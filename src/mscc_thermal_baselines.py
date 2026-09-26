"""
mscc_thermal_baselines.py  —  Paket 4
=======================================
Reviewer question: "Why not include MSCC or a thermal-aware baseline?"

New baselines (N=30, same seeds as Table 4):
  MSCCController              — Multi-Stage Constant Current charging
  ThermalAwareProportional    — Proportional with thermal derating

Also runs CC-CV and GraphOptimizer for direct comparison.

MSCC protocol:
  Stage 1: I = 2.0C  →  until any cell V ≥ 4.10 V  or T ≥ 40°C
  Stage 2: I = 1.0C  →  until any cell V ≥ 4.15 V  or T ≥ 42°C
  Stage 3: I = 0.5C  →  until target SOC reached

ThermalAwareProportional:
  Base:     I_i ∝ (target_SOC − SOC_i)  like Proportional
  Derating: cells hotter than (T_pack_mean + 1°C) receive reduced current
            factor_i = max(0.1, 1 − (T_i − T_mean − 1) / 5)

Output: results/mscc_thermal_prop_baselines_n30.csv
"""

import sys, csv, time, warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from graph_battery_pack import BatteryPackGraph, PackGNN, build_pack_from_ecm
from safe_fast_charge_optimizer import (
    default_config, CCCVController, GraphGuidedOptimizer
)

ECM_DIR = Path(__file__).parent.parent / "results" / "ecm"
MDL_DIR = Path(__file__).parent.parent / "results" / "models"
OUT_DIR = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════
#  MSCC CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════

class MSCCController:
    """
    Multi-Stage Constant Current (MSCC) charging.

    Standard reference baseline in battery-charging literature.
    Three stages with decreasing current as voltage limits are approached:
      Stage 1 (2.0C): aggressive until any cell near V_lim or T threshold
      Stage 2 (1.0C): moderate until tighter V/T thresholds
      Stage 3 (0.5C): conservative until target SOC

    Uniform current across all cells (same as CC-CV, no balancing).
    """
    STAGE_RATES  = [2.0, 1.0, 0.5]          # C-rates per stage
    V_THRESHOLDS = [4.10, 4.15, float("inf")]# V_term triggers (V)
    T_THRESHOLDS = [40.0, 42.0, float("inf")]# T triggers (°C)

    def __init__(self, cfg: dict):
        self.cfg   = cfg
        self.stage = 0

    def reset(self):
        self.stage = 0

    def get_currents(self, pack: BatteryPackGraph) -> np.ndarray:
        """Apply uniform MSCC current, advancing stage when thresholds crossed."""
        # Advance stage if any cell exceeds threshold
        V_terms = np.array([c.V_term for c in pack.cells])
        T_vals  = np.array([c.T_C   for c in pack.cells])
        Q_nom   = np.array([c.Q_nom_Ah for c in pack.cells])

        while self.stage < 2:
            if (V_terms.max() >= self.V_THRESHOLDS[self.stage] or
                    T_vals.max()  >= self.T_THRESHOLDS[self.stage]):
                self.stage += 1
            else:
                break

        I_rate = self.STAGE_RATES[self.stage]
        I_max  = self.cfg["I_max_C"] * Q_nom   # per-cell max

        # Uniform current, clipped to per-cell and pack limits
        I_uni  = np.minimum(I_rate * Q_nom, I_max)
        # Pack-level current constraint
        I_pack_max = self.cfg["I_pack_max_C"] * Q_nom.sum()
        if I_uni.sum() > I_pack_max:
            I_uni *= I_pack_max / I_uni.sum()

        return I_uni.astype(np.float32)

    def __repr__(self):
        return f"MSCCController(stage={self.stage})"


# ═══════════════════════════════════════════════════════════════════════════
#  THERMAL-AWARE PROPORTIONAL CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════

class ThermalAwareProportionalController:
    """
    SOC-deficit-proportional current allocation with per-cell thermal derating.

    Extends the Proportional baseline by reducing current to cells that are
    hotter than the pack mean, improving thermal uniformity without explicit
    graph-based coordination.

    Derating rule:
        T_excess_i = max(0, T_i − T_mean − T_margin)
        factor_i   = max(factor_min, 1 − T_excess_i / T_range)
        I_i        = I_proportional_i × factor_i   (then renormalised)

    Parameters (chosen to be simple and interpretable):
        T_margin  = 1.0°C  above pack mean before derating starts
        T_range   = 5.0°C  derating reaches factor_min at +6°C above mean
        factor_min = 0.10   minimum allowed derating factor
    """
    def __init__(self, cfg: dict,
                 T_margin: float = 1.0,
                 T_range:  float = 5.0,
                 factor_min: float = 0.10):
        self.cfg        = cfg
        self.T_margin   = T_margin
        self.T_range    = T_range
        self.factor_min = factor_min

    def reset(self):
        pass

    def get_currents(self, pack: BatteryPackGraph) -> np.ndarray:
        target  = self.cfg["target_soc"]
        socs    = np.array([c.SOC     for c in pack.cells])
        T_vals  = np.array([c.T_C     for c in pack.cells])
        Q_nom   = np.array([c.Q_nom_Ah for c in pack.cells])
        I_max   = self.cfg["I_max_C"] * Q_nom

        # Proportional base allocation
        deficit = np.maximum(target - socs, 0.0)
        if deficit.sum() < 1e-9:
            return np.zeros(pack.n_cells, dtype=np.float32)
        I_base = I_max * deficit / (deficit.max() + 1e-9)

        # Thermal derating
        T_mean     = T_vals.mean()
        T_excess   = np.maximum(T_vals - T_mean - self.T_margin, 0.0)
        factor     = np.maximum(self.factor_min,
                                1.0 - T_excess / (self.T_range + 1e-9))
        I_derated  = I_base * factor

        # Pack-level current constraint
        I_pack_max = self.cfg["I_pack_max_C"] * Q_nom.sum()
        if I_derated.sum() > I_pack_max:
            I_derated *= I_pack_max / I_derated.sum()

        return np.clip(I_derated, 0.0, I_max).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_eval(pack: BatteryPackGraph, controller, cfg: dict) -> dict:
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


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    cfg = default_config()

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

    controllers = {
        "CC-CV":                      CCCVController(cfg),
        "MSCC":                       MSCCController(cfg),
        "ThermalAwareProportional":   ThermalAwareProportionalController(cfg),
        "GraphOptimizer":             GraphGuidedOptimizer(cfg, gnn),
    }

    N_EPISODES = 30
    all_rows   = []

    print(f"\n{'='*60}")
    print(f"  MSCC + Thermal-aware Proportional Baselines: N={N_EPISODES}")
    print(f"{'='*60}")

    for name, ctrl in controllers.items():
        print(f"\n  Controller: {name}")
        ep_results = []
        t0 = time.time()

        for ep in range(N_EPISODES):
            np.random.seed(ep * 7); torch.manual_seed(ep * 7)
            pack = build_pack_from_ecm(
                ecm_parquet=parquet_path,
                n_cells=cfg["n_cells"], chemistry=cfg["chemistry"],
                soc_init=cfg["soc_init"], soc_noise=cfg["soc_noise"],
                T_amb=cfg["T_amb"], seed=ep * 7,
            )
            res = run_eval(pack, ctrl, cfg)
            ep_results.append(res)
            if (ep + 1) % 10 == 0:
                print(f"    ep {ep+1}/{N_EPISODES} | "
                      f"t={res['charge_time_min']:.1f}min "
                      f"σ={res['sigma_SOC_pct']:.3f}% "
                      f"viol={res['violations']}")

        runtime = time.time() - t0

        def agg(k): return round(float(np.mean([r[k] for r in ep_results])), 4)
        def std(k): return round(float(np.std( [r[k] for r in ep_results])), 4)

        row = {
            "method":               name,
            "charge_time_min":      agg("charge_time_min"),
            "charge_time_std":      std("charge_time_min"),
            "sigma_SOC_pct":        agg("sigma_SOC_pct"),
            "sigma_SOC_std":        std("sigma_SOC_pct"),
            "T_max_C":              agg("T_max_C"),
            "delta_T_C":            agg("delta_T_C"),
            "delta_T_std":          std("delta_T_C"),
            "aging_proxy_e4":       agg("aging_proxy"),
            "violations":           agg("violations"),
            "runtime_s":            round(runtime, 1),
            "n_episodes":           N_EPISODES,
        }
        all_rows.append(row)
        print(f"    → t={row['charge_time_min']}±{row['charge_time_std']}min  "
              f"σ={row['sigma_SOC_pct']}%  "
              f"ΔT={row['delta_T_C']}°C  "
              f"viol={row['violations']}")

    out_csv = OUT_DIR / "mscc_thermal_prop_baselines_n30.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n  Saved → {out_csv}")
    print("\n  Summary:")
    print(f"  {'Method':<28} {'Time(min)':>10} {'σ_SOC(%)':>10} "
          f"{'ΔT(°C)':>8} {'Viol':>6}")
    for r in all_rows:
        print(f"  {r['method']:<28} {r['charge_time_min']:>10} "
              f"{r['sigma_SOC_pct']:>10} {r['delta_T_C']:>8} "
              f"{r['violations']:>6}")

    # MSCC stage info
    print("\n  Note: MSCC stage thresholds: 2C→V≥4.10 or T≥40°C, "
          "1C→V≥4.15 or T≥42°C, 0.5C until target SOC")


if __name__ == "__main__":
    main()
