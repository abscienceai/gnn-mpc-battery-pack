"""
no_deltaT_ablation.py  —  Paket 2
===================================
Reviewer question: "Is the explicit ΔT penalty really necessary?"

Variants tested (N=30 each, same seeds as Table 4):
  1. Full Model          : w_gradient = 3.0  (default)
  2. No ΔT Penalty       : w_gradient = 0.0  (ΔT term removed from cost)
  3. No Graph Edges      : NodeOnlyGNN (from ablation_study.py)
  4. CC-CV               : baseline

Output: results/no_deltaT_ablation_n30.csv
"""

import sys, csv, time, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from graph_battery_pack import BatteryPackGraph, PackGNN, build_pack_from_ecm
from safe_fast_charge_optimizer import (
    default_config, compute_cost,
    CCCVController, GraphGuidedOptimizer
)

ECM_DIR = Path(__file__).parent.parent / "results" / "ecm"
MDL_DIR = Path(__file__).parent.parent / "results" / "models"
OUT_DIR = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════
#  NODE-ONLY GNN (no graph edges — reuse from ablation architecture)
# ═══════════════════════════════════════════════════════════════════════════

class NodeOnlyGNN(nn.Module):
    """
    No message passing — each node processed independently by MLP.
    Removes graph edge structure while keeping same output interface.
    Mirrors the variant in ablation_study.py.
    """
    def __init__(self, node_feat: int = 7, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 4),         # SOC, dT, aging, imbalance_contrib
        )

    def forward(self, x: torch.Tensor,
                edge_index=None, edge_attr=None, **_) -> dict:
        out   = self.mlp(x)               # (N, 4)
        imb   = torch.relu(out[:, 3].mean())
        return {
            "soc_pred":     torch.sigmoid(out[:, 0]),
            "delta_T_pred": out[:, 1],
            "aging_pred":   torch.relu(out[:, 2]),
            "imbalance":    imb,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_eval(pack: BatteryPackGraph, controller, cfg: dict) -> dict:
    controller.reset()
    steps = 0; aging_cum = 0.0; viol_cum = 0; T_max_ep = 0.0

    while steps < cfg["max_steps"]:
        if np.mean([c.SOC for c in pack.cells]) >= cfg["target_soc"]:
            break
        metrics   = pack.step(controller.get_currents(pack), dt=cfg["dt_s"])
        aging_cum += metrics["aging_cost"]
        viol_cum  += metrics["n_violations"]
        T_max_ep   = max(T_max_ep, metrics["T_max"])
        steps     += 1

    socs = [c.SOC for c in pack.cells]
    T_f  = [c.T_C for c in pack.cells]
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

    # Load trained PackGNN
    gnn = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(DEVICE)
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    gnn.load_state_dict(ckpt.get("model_state", ckpt.get("model_state_dict", ckpt)))
    gnn.eval()

    # NodeOnlyGNN (no graph) — untrained baseline weight (fair: matches ablation_study.py)
    node_gnn = NodeOnlyGNN(node_feat=7, hidden=64).to(DEVICE)
    node_gnn.eval()

    # Config for No-ΔT variant
    cfg_no_dT              = dict(cfg)
    cfg_no_dT["w_gradient"] = 0.0

    # Controller variants
    variants = [
        ("Full Model (w_grad=3)", GraphGuidedOptimizer(cfg, gnn)),
        ("No ΔT Penalty (w_grad=0)", GraphGuidedOptimizer(cfg_no_dT, gnn)),
        ("No Graph Edges",        GraphGuidedOptimizer(cfg, node_gnn)),
        ("CC-CV",                 CCCVController(cfg)),
    ]

    N_EPISODES = 30
    all_rows   = []

    print(f"\n{'='*60}")
    print(f"  No-ΔT Ablation: N={N_EPISODES} episodes")
    print(f"  Baseline w_gradient = {cfg['w_gradient']}")
    print(f"  No-ΔT  w_gradient  = {cfg_no_dT['w_gradient']}")
    print(f"{'='*60}")

    for name, ctrl in variants:
        print(f"\n  Variant: {name}")
        ep_results = []
        t0 = time.time()

        for ep in range(N_EPISODES):
            np.random.seed(ep * 7)
            torch.manual_seed(ep * 7)
            pack = build_pack_from_ecm(
                ecm_parquet=parquet_path,
                n_cells=cfg["n_cells"], chemistry=cfg["chemistry"],
                soc_init=cfg["soc_init"], soc_noise=cfg["soc_noise"],
                T_amb=cfg["T_amb"], seed=ep * 7,
            )
            res = run_eval(pack, ctrl, cfg if "No ΔT" not in name else cfg_no_dT)
            ep_results.append(res)

        runtime = time.time() - t0

        def agg(k): return float(np.mean([r[k] for r in ep_results]))
        def std(k): return float(np.std([r[k] for r in ep_results]))

        row = {
            "variant":          name,
            "charge_time_min":  round(agg("charge_time_min"), 2),
            "sigma_SOC_pct":    round(agg("sigma_SOC_pct"), 4),
            "sigma_SOC_std":    round(std("sigma_SOC_pct"), 4),
            "T_max_C":          round(agg("T_max_C"), 2),
            "delta_T_C":        round(agg("delta_T_C"), 3),
            "delta_T_std":      round(std("delta_T_C"), 3),
            "aging_proxy_e4":   round(agg("aging_proxy"), 4),
            "violations":       round(agg("violations"), 2),
            "runtime_s":        round(runtime, 1),
            "w_gradient_used":  0.0 if "No ΔT" in name else cfg["w_gradient"],
        }
        all_rows.append(row)
        print(f"    t={row['charge_time_min']}min  σ={row['sigma_SOC_pct']}%  "
              f"ΔT={row['delta_T_C']}°C  viol={row['violations']}")

    out_csv = OUT_DIR / "no_deltaT_ablation_n30.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n  Saved → {out_csv}")
    print("\n  Summary:")
    print(f"  {'Variant':<30} {'Time':>8} {'σ_SOC%':>9} {'ΔT°C':>7} {'Viol':>6}")
    for r in all_rows:
        print(f"  {r['variant']:<30} {r['charge_time_min']:>8} "
              f"{r['sigma_SOC_pct']:>9} {r['delta_T_C']:>7} {r['violations']:>6}")


if __name__ == "__main__":
    main()
