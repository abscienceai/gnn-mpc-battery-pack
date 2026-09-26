"""
weight_sweep.py  —  Paket 3
=============================
Reviewer question: "No exhaustive weight sweep — how sensitive are results to λ?"

Step 1 — Grid search: w_time × w_gradient, N=10 per combo
  w_time    ∈ {4, 8, 12, 20}   (higher = more speed pressure)
  w_gradient ∈ {1, 3, 5}        (higher = more thermal-uniformity pressure)
  w_violation = 50 (fixed)
  All other weights from default_config

Step 2 — 3 representative modes rerun with N=30:
  Fast-safe:    w_time=20, w_gradient=1
  Balanced:     w_time=8,  w_gradient=3
  Safety-first: w_time=4,  w_gradient=5  (similar to paper default)

Outputs:
  results/weight_sweep_raw_n10.csv
  results/weight_sweep_selected_modes_n30.csv
"""

import sys, csv, time, warnings, itertools
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from graph_battery_pack import PackGNN, build_pack_from_ecm
from safe_fast_charge_optimizer import default_config, GraphGuidedOptimizer

ECM_DIR = Path(__file__).parent.parent / "results" / "ecm"
MDL_DIR = Path(__file__).parent.parent / "results" / "models"
OUT_DIR = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ─── Episode runner ───────────────────────────────────────────────────────

def run_eval(pack, controller, cfg) -> dict:
    controller.reset()
    steps = 0; aging_cum = 0.0; viol_cum = 0; T_max_ep = 0.0
    while steps < cfg["max_steps"]:
        if np.mean([c.SOC for c in pack.cells]) >= cfg["target_soc"]:
            break
        m = pack.step(controller.get_currents(pack), dt=cfg["dt_s"])
        aging_cum += m["aging_cost"]; viol_cum += m["n_violations"]
        T_max_ep   = max(T_max_ep, m["T_max"]); steps += 1
    socs = [c.SOC for c in pack.cells]; T_f = [c.T_C for c in pack.cells]
    return {
        "charge_time_min": round(steps * cfg["dt_s"] / 60.0, 2),
        "sigma_SOC_pct":   round(float(np.std(socs)) * 100.0, 4),
        "T_max_C":         round(T_max_ep, 2),
        "delta_T_C":       round(float(max(T_f) - min(T_f)), 3),
        "aging_proxy":     round(aging_cum * 1e4, 4),
        "violations":      viol_cum,
    }


def run_grid(gnn, parquet_path, cfg_base, w_times, w_gradients,
             n_episodes: int = 10):
    rows = []
    combos = list(itertools.product(w_times, w_gradients))
    total  = len(combos)

    print(f"\n  Grid search: {total} combos × N={n_episodes}")
    for i, (wt, wg) in enumerate(combos, 1):
        cfg_i              = dict(cfg_base)
        cfg_i["w_time"]    = wt
        cfg_i["w_gradient"]= wg
        ctrl               = GraphGuidedOptimizer(cfg_i, gnn)

        ep_r = []
        for ep in range(n_episodes):
            np.random.seed(ep * 7); torch.manual_seed(ep * 7)
            pack = build_pack_from_ecm(
                ecm_parquet=parquet_path,
                n_cells=cfg_i["n_cells"], chemistry=cfg_i["chemistry"],
                soc_init=cfg_i["soc_init"], soc_noise=cfg_i["soc_noise"],
                T_amb=cfg_i["T_amb"], seed=ep * 7,
            )
            ep_r.append(run_eval(pack, ctrl, cfg_i))

        def agg(k): return round(float(np.mean([r[k] for r in ep_r])), 4)

        row = {
            "w_time":           wt,
            "w_gradient":       wg,
            "w_violation":      cfg_i["w_violation"],
            "n_episodes":       n_episodes,
            "charge_time_min":  agg("charge_time_min"),
            "sigma_SOC_pct":    agg("sigma_SOC_pct"),
            "T_max_C":          agg("T_max_C"),
            "delta_T_C":        agg("delta_T_C"),
            "aging_proxy_e4":   agg("aging_proxy"),
            "violations":       agg("violations"),
        }
        rows.append(row)
        print(f"  [{i:2d}/{total}] w_time={wt:>2}, w_grad={wg} | "
              f"t={row['charge_time_min']}min  σ={row['sigma_SOC_pct']}%  "
              f"ΔT={row['delta_T_C']}°C  viol={row['violations']}")

    return rows


def run_selected_modes(gnn, parquet_path, cfg_base, n_episodes: int = 30):
    modes = [
        ("Fast-safe",    {"w_time": 20, "w_gradient": 1}),
        ("Balanced",     {"w_time":  8, "w_gradient": 3}),
        ("Safety-first", {"w_time":  4, "w_gradient": 5}),
        ("Default",      {"w_time":  4, "w_gradient": 3}),  # paper config
    ]
    rows = []
    print(f"\n  Selected modes (N={n_episodes}):")

    for mode_name, overrides in modes:
        cfg_m = dict(cfg_base)
        cfg_m.update(overrides)
        ctrl  = GraphGuidedOptimizer(cfg_m, gnn)

        ep_r  = []
        t0    = time.time()
        for ep in range(n_episodes):
            np.random.seed(ep * 7); torch.manual_seed(ep * 7)
            pack = build_pack_from_ecm(
                ecm_parquet=parquet_path,
                n_cells=cfg_m["n_cells"], chemistry=cfg_m["chemistry"],
                soc_init=cfg_m["soc_init"], soc_noise=cfg_m["soc_noise"],
                T_amb=cfg_m["T_amb"], seed=ep * 7,
            )
            ep_r.append(run_eval(pack, ctrl, cfg_m))

        runtime = time.time() - t0

        def agg(k): return round(float(np.mean([r[k] for r in ep_r])), 4)
        def std(k): return round(float(np.std( [r[k] for r in ep_r])), 4)

        row = {
            "mode":             mode_name,
            "w_time":           cfg_m["w_time"],
            "w_gradient":       cfg_m["w_gradient"],
            "w_violation":      cfg_m["w_violation"],
            "n_episodes":       n_episodes,
            "charge_time_min":  agg("charge_time_min"),
            "charge_time_std":  std("charge_time_min"),
            "sigma_SOC_pct":    agg("sigma_SOC_pct"),
            "sigma_SOC_std":    std("sigma_SOC_pct"),
            "T_max_C":          agg("T_max_C"),
            "delta_T_C":        agg("delta_T_C"),
            "delta_T_std":      std("delta_T_C"),
            "aging_proxy_e4":   agg("aging_proxy"),
            "violations":       agg("violations"),
            "runtime_s":        round(runtime, 1),
        }
        rows.append(row)
        print(f"  {mode_name:<16} (wt={cfg_m['w_time']:>2}, wg={cfg_m['w_gradient']}) | "
              f"t={row['charge_time_min']}±{row['charge_time_std']}min  "
              f"σ={row['sigma_SOC_pct']}%  ΔT={row['delta_T_C']}°C")

    return rows


def main():
    cfg_base = default_config()

    ecm_files = sorted(ECM_DIR.glob("ecm_params_*.parquet"))
    assert ecm_files, f"No ECM parquet in {ECM_DIR}"
    parquet_path = ecm_files[-1]
    print(f"ECM: {parquet_path.name}")

    ckpt_files = sorted(MDL_DIR.glob("pack_gnn_*.pt"))
    assert ckpt_files, f"No GNN checkpoint in {MDL_DIR}"
    ckpt_path  = ckpt_files[-1]
    print(f"GNN: {ckpt_path.name}")

    gnn = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(DEVICE)
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    gnn.load_state_dict(ckpt.get("model_state", ckpt.get("model_state_dict", ckpt)))
    gnn.eval()

    # ── Step 1: Grid search N=10 ───────────────────────────────────────────
    W_TIMES     = [4, 8, 12, 20]
    W_GRADIENTS = [1, 3, 5]
    grid_rows   = run_grid(gnn, parquet_path, cfg_base,
                           W_TIMES, W_GRADIENTS, n_episodes=10)

    csv_grid = OUT_DIR / "weight_sweep_raw_n10.csv"
    with open(csv_grid, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=grid_rows[0].keys())
        w.writeheader(); w.writerows(grid_rows)
    print(f"\n  Saved grid → {csv_grid}")

    # ── Step 2: Selected modes N=30 ───────────────────────────────────────
    mode_rows = run_selected_modes(gnn, parquet_path, cfg_base, n_episodes=30)

    csv_modes = OUT_DIR / "weight_sweep_selected_modes_n30.csv"
    with open(csv_modes, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mode_rows[0].keys())
        w.writeheader(); w.writerows(mode_rows)
    print(f"  Saved modes → {csv_modes}")

    # ── Pareto summary ─────────────────────────────────────────────────────
    print("\n  Speed–safety Pareto (grid, N=10):")
    print(f"  {'w_time':>7} {'w_grad':>7} {'Time(min)':>10} {'σ_SOC%':>9} {'ΔT°C':>7}")
    for r in sorted(grid_rows, key=lambda x: x["charge_time_min"]):
        print(f"  {r['w_time']:>7} {r['w_gradient']:>7} "
              f"{r['charge_time_min']:>10} {r['sigma_SOC_pct']:>9} {r['delta_T_C']:>7}")


if __name__ == "__main__":
    main()
