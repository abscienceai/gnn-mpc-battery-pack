"""
bigru_error_injection.py
========================
Reviewer Comment 8: "Voltage noise ≠ SOC estimation error — inject BiGRU residuals"

BiGRU MAE = 7.98% absolute SOC. Injects realistic error modes into the
closed-loop evaluation to show that GraphOptimizer is robust to SOC
estimation uncertainty at the BiGRU accuracy level.

Error modes (all at BiGRU-level magnitude σ = 0.0798):
  1. Independent per-cell Gaussian: ε_i ~ N(0, σ²) each step, i.i.d.
  2. Common-mode Gaussian:          ε_shared ~ N(0, σ²) same for all cells each step
  3. Independent cell bias:         fixed per-cell offset b_i ~ N(0, σ²), constant episode
  4. Correlated drift:              random walk, σ_step = σ/√T so std≈σ at T=12 steps

For each mode, N=20 episodes. Controller observes noisy SOC; physics evolves on true SOC.
"""

import sys, csv, warnings, torch, numpy as np
from pathlib import Path
from copy import deepcopy

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from graph_battery_pack import PackGNN, build_pack_from_ecm, BatteryPackGraph
from safe_fast_charge_optimizer import default_config, GraphGuidedOptimizer, CCCVController

ECM_DIR = Path(__file__).parent.parent / "results/ecm"
MDL_DIR = Path(__file__).parent.parent / "results/models"
OUT_DIR = Path(__file__).parent.parent / "results"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

BIGRU_MAE = 0.0798   # absolute SOC units (7.98%)
N_STEPS_TYPICAL = 12  # typical CC-CV steps for drift scaling


def inject_noisy_soc(pack: BatteryPackGraph, error_mode: str,
                     ep_seed: int, step: int, cell_biases=None):
    """
    Temporarily perturb cell SOC values with the given error mode.
    Returns the original SOC values so they can be restored afterward.
    """
    rng = np.random.default_rng(ep_seed * 1000 + step)
    original_soc = [c.SOC for c in pack.cells]
    n = len(pack.cells)
    sigma = BIGRU_MAE

    if error_mode == "independent":
        # Independent per-cell per-step Gaussian
        errors = rng.normal(0, sigma, n)

    elif error_mode == "common_mode":
        # Same additive error for all cells (correlated)
        shared = rng.normal(0, sigma)
        errors = np.full(n, shared)

    elif error_mode == "cell_bias":
        # Fixed per-cell bias for the whole episode (supplied as cell_biases)
        errors = cell_biases if cell_biases is not None else np.zeros(n)

    elif error_mode == "drift":
        # Random walk: step std = sigma/sqrt(T) so cumulative std ≈ sigma at T steps
        step_sigma = sigma / np.sqrt(N_STEPS_TYPICAL)
        errors = rng.normal(0, step_sigma, n)
        # (In the caller, drifts accumulate — here we apply a single step increment)

    else:
        errors = np.zeros(n)

    # Apply errors, clip to valid SOC range
    for i, c in enumerate(pack.cells):
        c.SOC = float(np.clip(c.SOC + errors[i], 0.02, 0.97))

    return original_soc, errors


def restore_soc(pack: BatteryPackGraph, original_soc):
    for i, c in enumerate(pack.cells):
        c.SOC = original_soc[i]


def run_eval_with_soc_error(pack, ctrl, cfg, error_mode, ep_seed):
    """Episode runner that injects SOC errors before each controller query."""
    ctrl.reset()
    steps = 0; viol = 0; T_max = 0.0; aging_cum = 0.0

    # For cell_bias: fixed per-episode per-cell offset
    n = len(pack.cells)
    rng_ep = np.random.default_rng(ep_seed * 999)
    cell_biases = rng_ep.normal(0, BIGRU_MAE, n) if error_mode == "cell_bias" else None

    # For drift: cumulative drift state per cell
    drift_state = np.zeros(n)

    while steps < cfg["max_steps"]:
        if np.mean([c.SOC for c in pack.cells]) >= cfg["target_soc"]:
            break

        # Inject SOC error before controller sees pack state
        if error_mode == "drift":
            # Accumulate drift
            step_sigma = BIGRU_MAE / np.sqrt(N_STEPS_TYPICAL)
            rng_step = np.random.default_rng(ep_seed * 1000 + steps)
            drift_state += rng_step.normal(0, step_sigma, n)
            # Apply accumulated drift to cell SOC
            orig_soc = [c.SOC for c in pack.cells]
            for i, c in enumerate(pack.cells):
                c.SOC = float(np.clip(c.SOC + drift_state[i], 0.02, 0.97))
            action = ctrl.get_currents(pack)
            restore_soc(pack, orig_soc)
        elif error_mode != "none":
            orig_soc, _ = inject_noisy_soc(pack, error_mode, ep_seed, steps, cell_biases)
            action = ctrl.get_currents(pack)
            restore_soc(pack, orig_soc)
        else:
            action = ctrl.get_currents(pack)

        # Physics update on TRUE SOC (no error)
        m = pack.step(action, dt=cfg["dt_s"])
        viol += m["n_violations"]
        T_max = max(T_max, m["T_max"])
        aging_cum += m["aging_cost"]
        steps += 1

    socs = [c.SOC for c in pack.cells]
    T_f  = [c.T_C for c in pack.cells]
    return {
        "time_min":   round(steps * cfg["dt_s"] / 60.0, 2),
        "sigma_pct":  round(float(np.std(socs)) * 100, 4),
        "T_max":      round(T_max, 2),
        "dT":         round(float(max(T_f) - min(T_f)), 3),
        "violations": viol,
        "aging_e4":   round(aging_cum * 1e4, 4),
    }


def main():
    cfg = default_config()
    ecm = sorted(ECM_DIR.glob("ecm_params_*.parquet"))[-1]

    ckpt = torch.load(str(sorted(MDL_DIR.glob("pack_gnn_*.pt"))[-1]), map_location=DEVICE)
    gnn  = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(DEVICE)
    gnn.load_state_dict(ckpt.get("model_state", ckpt)); gnn.eval()

    go_ctrl   = GraphGuidedOptimizer(cfg, gnn)
    cccv_ctrl = CCCVController(cfg)   # reference: not affected by SOC estimation

    error_modes = {
        "none":        "Clean (no error)",
        "independent": "Independent per-cell N(0,σ²)",
        "common_mode": "Common-mode N(0,σ²)",
        "cell_bias":   "Fixed per-cell bias N(0,σ²)",
        "drift":       "Correlated drift (RW, std≈σ at 12 steps)",
    }

    N_EP = 20
    rows = []

    print(f"BiGRU MAE = {BIGRU_MAE*100:.2f}%  →  injected σ = {BIGRU_MAE:.4f}")
    print(f"\n{'Error Mode':<38} {'Time':>7} {'σ_SOC%':>8} {'ΔT':>6} {'Viol':>6}")
    print("─" * 70)

    for mode_key, mode_label in error_modes.items():
        eps = []
        for ep in range(N_EP):
            np.random.seed(ep * 7); torch.manual_seed(ep * 7)
            pack = build_pack_from_ecm(ecm_parquet=ecm, n_cells=12, chemistry="LFP",
                                        soc_init=0.20, soc_noise=0.03, T_amb=25.0,
                                        seed=ep * 7)
            eps.append(run_eval_with_soc_error(pack, go_ctrl, cfg, mode_key, ep))

        def a(k): return round(float(np.mean([r[k] for r in eps])), 4)
        def s(k): return round(float(np.std( [r[k] for r in eps])), 4)

        row = {"error_mode": mode_label, "controller": "GraphOptimizer",
               "time_min": a("time_min"),    "time_std": s("time_min"),
               "sigma_pct": a("sigma_pct"),  "sigma_std": s("sigma_pct"),
               "T_max": a("T_max"),
               "dT": a("dT"),                "dT_std": s("dT"),
               "violations": a("violations"),
               "aging_e4": a("aging_e4"),
               "N": N_EP}
        rows.append(row)
        print(f"{mode_label:<38} {row['time_min']:>7} {row['sigma_pct']:>8} "
              f"{row['dT']:>6} {row['violations']:>6}")

    # CC-CV reference (not affected by SOC error)
    cccv_eps = []
    for ep in range(N_EP):
        np.random.seed(ep * 7); torch.manual_seed(ep * 7)
        pack = build_pack_from_ecm(ecm_parquet=ecm, n_cells=12, chemistry="LFP",
                                    soc_init=0.20, soc_noise=0.03, T_amb=25.0,
                                    seed=ep * 7)
        cccv_eps.append(run_eval_with_soc_error(pack, cccv_ctrl, cfg, "none", ep))
    def ca(k): return round(float(np.mean([r[k] for r in cccv_eps])), 4)
    cccv_row = {"error_mode": "CC-CV (reference, no error)",
                "controller": "CC-CV",
                "time_min": ca("time_min"), "time_std": 0,
                "sigma_pct": ca("sigma_pct"), "sigma_std": 0,
                "T_max": ca("T_max"), "dT": ca("dT"), "dT_std": 0,
                "violations": ca("violations"), "aging_e4": ca("aging_e4"), "N": N_EP}
    rows.append(cccv_row)
    print(f"{'CC-CV (reference, no error)':<38} {cccv_row['time_min']:>7} "
          f"{cccv_row['sigma_pct']:>8} {cccv_row['dT']:>6} {cccv_row['violations']:>6}")

    out = OUT_DIR / "bigru_error_injection_n20.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved → {out}")

    # Summary: max degradation relative to clean baseline
    clean = next(r for r in rows if "Clean" in r["error_mode"])
    print("\n── Degradation vs clean baseline ──────────────────────")
    print(f"Clean:  σ={clean['sigma_pct']}%  ΔT={clean['dT']}°C")
    for r in rows:
        if "Clean" in r["error_mode"] or "CC-CV" in r["error_mode"]: continue
        delta_sigma = r["sigma_pct"] - clean["sigma_pct"]
        print(f"{r['error_mode']:<38}: Δσ={delta_sigma:+.4f}%  viol={r['violations']}")


if __name__ == "__main__":
    main()
