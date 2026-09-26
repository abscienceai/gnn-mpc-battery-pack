"""
cross_chem_ablation.py
======================
Reviewer Comment 6: "Chemistry-agnostic hypothesis not isolated from local features"

Tests: Full GNN vs No-edge GNN vs MLP Surrogate vs CC-CV on NMC and LCO
Same N=30, seed=7*ep protocol as primary results.

If No-edge GNN and MLP perform WORSE than Full GNN on cross-chemistry transfer,
graph edges are specifically responsible for the benefit — not just local features.
"""
import sys, csv, warnings, torch, numpy as np, torch.nn as nn
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


class NodeOnlyGNN(nn.Module):
    """No message passing — processes each node independently (no graph edges)."""
    def __init__(self, node_feat=7, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 4),
        )
    def forward(self, x, edge_index=None, edge_attr=None, **_):
        out = self.mlp(x)
        return {"soc_pred":     torch.sigmoid(out[:, 0]),
                "delta_T_pred": out[:, 1],
                "aging_pred":   torch.relu(out[:, 2]),
                "imbalance":    torch.relu(out[:, 3].mean())}


class PackMLP(nn.Module):
    """Flat MLP: all cell features concatenated, no graph structure."""
    def __init__(self, n_cells=12, node_feat=7, hidden=256):
        super().__init__()
        self.n_cells = n_cells
        in_dim = n_cells * node_feat
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.head_soc = nn.Linear(hidden//2, n_cells)
        self.head_dT  = nn.Linear(hidden//2, n_cells)
        self.head_age = nn.Linear(hidden//2, n_cells)
        self.head_imb = nn.Linear(hidden//2, 1)

    def forward(self, x, edge_index=None, edge_attr=None, **_):
        flat = x.reshape(1, -1)
        h = self.backbone(flat)
        return {"soc_pred":     torch.sigmoid(self.head_soc(h)).squeeze(0),
                "delta_T_pred": self.head_dT(h).squeeze(0),
                "aging_pred":   torch.relu(self.head_age(h)).squeeze(0),
                "imbalance":    torch.relu(self.head_imb(h)).squeeze()}


def run_eval(pack, ctrl, cfg):
    ctrl.reset()
    steps=0; viol=0; T_max=0.0
    while steps < cfg["max_steps"]:
        if np.mean([c.SOC for c in pack.cells]) >= cfg["target_soc"]: break
        m = pack.step(ctrl.get_currents(pack), dt=cfg["dt_s"])
        viol += m["n_violations"]; T_max = max(T_max, m["T_max"]); steps += 1
    socs = [c.SOC for c in pack.cells]; Tf = [c.T_C for c in pack.cells]
    return {"time_min":  round(steps * cfg["dt_s"] / 60.0, 2),
            "sigma_pct": round(float(np.std(socs)) * 100, 4),
            "T_max":     round(T_max, 2),
            "dT":        round(float(max(Tf) - min(Tf)), 3),
            "violations":viol}


def main():
    cfg = default_config()
    ecm = sorted(ECM_DIR.glob("ecm_params_*.parquet"))[-1]
    print(f"ECM: {ecm.name}")

    # Load Full GNN (LFP-trained)
    ckpt_path = sorted(MDL_DIR.glob("pack_gnn_*.pt"))[-1]
    gnn = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(DEVICE)
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    gnn.load_state_dict(ckpt.get("model_state", ckpt)); gnn.eval()
    print(f"GNN: {ckpt_path.name}")

    # No-edge GNN: UNTRAINED baseline (random init), intentionally not loaded
    # from a checkpoint. NOTE: this is NOT the "No-edge GNN (trained)" comparison
    # used in the manuscript (Tables 8/9); that comparison uses
    # train_node_only_gnn.py + fair_ablation.py's genuinely-retrained checkpoint
    # (results/models/node_only_gnn_*.pt). This script's untrained variant is
    # retained only as a legacy reference baseline, not a manuscript source.
    node_gnn = NodeOnlyGNN().to(DEVICE); node_gnn.eval()

    # MLP surrogate (load trained if available, else untrained)
    mlp = PackMLP().to(DEVICE)
    mlp_ckpts = sorted(MDL_DIR.glob("pack_mlp_*.pt"))
    if mlp_ckpts:
        mc = torch.load(str(mlp_ckpts[-1]), map_location=DEVICE)
        mlp.load_state_dict(mc.get("model_state_dict", mc)); mlp.eval()
        print(f"MLP: {mlp_ckpts[-1].name}")
    else:
        mlp.eval(); print("MLP: untrained (no checkpoint found)")

    surrogates = {
        "CC-CV":         CCCVController(cfg),
        "Full GNN":      GraphGuidedOptimizer(cfg, gnn),
        "No-edge GNN (untrained baseline)": GraphGuidedOptimizer(cfg, node_gnn),
        "MLP Surrogate": GraphGuidedOptimizer(cfg, mlp),
    }

    N_EP = 30
    rows = []
    print(f"\n{'Chem':<6} {'Method':<18} {'Time':>7} {'σ_SOC%':>8} {'T_max':>7} {'ΔT°C':>7} {'Viol':>6}")

    for chemistry in ["NMC", "LCO"]:
        print(f"\n── {chemistry} ──────────────────────────────────────")
        for name, ctrl in surrogates.items():
            eps = []
            for ep in range(N_EP):
                np.random.seed(ep * 7); torch.manual_seed(ep * 7)
                pack = build_pack_from_ecm(ecm_parquet=ecm, n_cells=12,
                                            chemistry=chemistry, soc_init=0.20,
                                            soc_noise=0.03, T_amb=25.0,
                                            seed=ep * 7)
                eps.append(run_eval(pack, ctrl, cfg))

            def a(k): return round(float(np.mean([r[k] for r in eps])), 4)

            row = {"chemistry": chemistry, "method": name,
                   "time_min":   a("time_min"),
                   "sigma_pct":  a("sigma_pct"),
                   "T_max":      a("T_max"),
                   "dT":         a("dT"),
                   "violations": a("violations"),
                   "N": N_EP}
            rows.append(row)
            print(f"{chemistry:<6} {name:<18} {row['time_min']:>7} {row['sigma_pct']:>8} "
                  f"{row['T_max']:>7} {row['dT']:>7} {row['violations']:>6}")

    out = OUT_DIR / "cross_chem_ablation_n30.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved → {out}")

    # Summary: key comparison
    print("\n── KEY COMPARISON (GraphEdge isolation) ──")
    for chem in ["NMC", "LCO"]:
        for name in ["Full GNN", "No-edge GNN (untrained baseline)", "MLP Surrogate"]:
            r = next(x for x in rows if x["chemistry"]==chem and x["method"]==name)
            print(f"  {chem} {name:<18}: σ={r['sigma_pct']}% viol={r['violations']}")


if __name__ == "__main__":
    main()
