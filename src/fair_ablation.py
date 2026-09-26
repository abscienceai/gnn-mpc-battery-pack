"""Fair surrogate ablation used for the manuscript architecture comparison.

This script evaluates the frozen full PackGNN, the independently trained
NodeOnlyGNN, and the independently trained flat MLP under the same H=1
controller protocol. Results are written to results/fair_ablation/ as JSON
and CSV in addition to being printed.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_battery_pack import PackGNN, build_pack_from_ecm
from safe_fast_charge_optimizer import default_config, GraphGuidedOptimizer, CCCVController
from train_node_only_gnn import NodeOnlyGNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = ROOT / "results"
MODEL_DIR = RESULTS / "models"
ECM_DIR = RESULTS / "ecm"
OUT_DIR = RESULTS / "fair_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def latest(pattern: str, folder: Path) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {folder}")
    return matches[-1]


ecm = latest("ecm_params_*.parquet", ECM_DIR)
cfg = default_config()
cfg["horizon"] = 1  # architecture comparison uses the same H=1 control protocol

ckpt = torch.load(MODEL_DIR / "pack_gnn_20260630_220402.pt", map_location=DEVICE)
gnn = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(DEVICE)
gnn.load_state_dict(ckpt.get("model_state", ckpt))
gnn.eval()


class PackMLP(nn.Module):
    def __init__(self, n_cells=12, node_feat=7, hidden=256):
        super().__init__()
        in_dim = n_cells * node_feat
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.head_soc = nn.Linear(hidden // 2, n_cells)
        self.head_dT = nn.Linear(hidden // 2, n_cells)
        self.head_age = nn.Linear(hidden // 2, n_cells)
        self.head_imb = nn.Linear(hidden // 2, 1)

    def forward(self, x, edge_index=None, edge_attr=None, **_):
        h = self.backbone(x.reshape(1, -1))
        return {
            "soc_pred": torch.sigmoid(self.head_soc(h)).squeeze(0),
            "delta_T_pred": self.head_dT(h).squeeze(0),
            "aging_pred": torch.relu(self.head_age(h)).squeeze(0),
            "imbalance": torch.relu(self.head_imb(h)).squeeze(),
        }


mlp_ckpt = latest("pack_mlp_fair5k_*.pt", MODEL_DIR)
mlp = PackMLP().to(DEVICE)
mc = torch.load(mlp_ckpt, map_location=DEVICE)
mlp.load_state_dict(mc.get("model_state_dict", mc))
mlp.eval()

node_ckpt = latest("node_only_gnn_*.pt", MODEL_DIR)
node_gnn = NodeOnlyGNN().to(DEVICE)
nc = torch.load(node_ckpt, map_location=DEVICE)
node_gnn.load_state_dict(nc.get("model_state", nc))
node_gnn.eval()


def run_ep(pack, ctrl, cfg):
    ctrl.reset()
    steps = 0
    viol = 0
    t_max = 0.0
    while steps < cfg["max_steps"]:
        if np.mean([c.SOC for c in pack.cells]) >= cfg["target_soc"]:
            break
        metrics = pack.step(ctrl.get_currents(pack), dt=cfg["dt_s"])
        viol += metrics["n_violations"]
        t_max = max(t_max, metrics["T_max"])
        steps += 1
    socs = [c.SOC for c in pack.cells]
    tf = [c.T_C for c in pack.cells]
    return {
        "time_min": steps * cfg["dt_s"] / 60,
        "sigma_soc_pct": float(np.std(socs)) * 100,
        "T_max_C": t_max,
        "dT_C": float(max(tf) - min(tf)),
        "violations": float(viol),
    }


def evaluate(chem, name, controller, n=30):
    rows = []
    t0 = time.time()
    for ep in range(n):
        seed = ep * 7
        np.random.seed(seed)
        torch.manual_seed(seed)
        pack = build_pack_from_ecm(
            ecm_parquet=ecm, n_cells=12, chemistry=chem,
            soc_init=0.20, soc_noise=0.03, T_amb=25.0, seed=seed,
        )
        r = run_ep(pack, controller, cfg)
        r.update({"chemistry": chem, "variant": name, "episode": ep, "seed": seed})
        rows.append(r)
        eta = (time.time() - t0) / (ep + 1) * (n - ep - 1)
        print(f"[{chem} | {name}] {ep+1}/{n} sigma={r['sigma_soc_pct']:.3f}% dT={r['dT_C']:.4f} ETA={eta:.0f}s", flush=True)
    summary = {k: float(np.mean([r[k] for r in rows])) for k in ["time_min", "sigma_soc_pct", "T_max_C", "dT_C", "violations"]}
    summary.update({"chemistry": chem, "variant": name, "n_episodes": n, "device": DEVICE})
    return rows, summary


def main():
    all_rows, summaries = [], []
    lfp_variants = [
        ("Full GNN", GraphGuidedOptimizer(cfg, gnn)),
        ("No-edge GNN trained", GraphGuidedOptimizer(cfg, node_gnn)),
        ("MLP fair 5k", GraphGuidedOptimizer(cfg, mlp)),
        ("CC-CV", CCCVController(cfg)),
    ]
    for name, ctrl in lfp_variants:
        rows, summary = evaluate("LFP", name, ctrl)
        all_rows += rows; summaries.append(summary)
    for chem in ["NMC", "LCO"]:
        for name, ctrl in [
            ("Full GNN", GraphGuidedOptimizer(cfg, gnn)),
            ("No-edge GNN trained", GraphGuidedOptimizer(cfg, node_gnn)),
            ("MLP fair 5k", GraphGuidedOptimizer(cfg, mlp)),
        ]:
            rows, summary = evaluate(chem, name, ctrl)
            all_rows += rows; summaries.append(summary)

    with (OUT_DIR / "fair_ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"config": cfg, "summaries": summaries}, f, indent=2)
    fieldnames = list(all_rows[0].keys())
    with (OUT_DIR / "fair_ablation_episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(all_rows)
    with (OUT_DIR / "fair_ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys())); w.writeheader(); w.writerows(summaries)
    print(f"Saved fair-ablation outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
