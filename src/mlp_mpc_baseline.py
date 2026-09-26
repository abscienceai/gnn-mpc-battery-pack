"""
mlp_mpc_baseline.py  —  Paket 1
================================
Reviewer question: "Is GNN really necessary? Would a flat MLP surrogate suffice?"

Method:
  1. Train PackMLP (flat MLP, no graph edges) on physics rollouts
  2. Drop PackMLP into GraphGuidedOptimizer as surrogate (same CEM, same constraints)
  3. Run N=30 episodes with same seeds as Table 4
  4. Compare: GraphOptimizer (GNN) vs MLPOptimizer (MLP) vs CC-CV vs Proportional

Output: results/mlp_mpc_vs_packgnn_n30.csv
"""

import sys, csv, time, warnings
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from graph_battery_pack import BatteryPackGraph, PackGNN, build_pack_from_ecm
from safe_fast_charge_optimizer import (
    default_config, compute_cost,
    CCCVController, ProportionalController, GraphGuidedOptimizer
)

ECM_DIR = Path(__file__).parent.parent / "results" / "ecm"
MDL_DIR = Path(__file__).parent.parent / "results" / "models"
OUT_DIR = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════
#  PACK MLP  —  flat MLP, no graph structure, GNN-compatible output dict
# ═══════════════════════════════════════════════════════════════════════════

class PackMLP(nn.Module):
    """
    Flat MLP surrogate for battery pack dynamics.
    Input : (N_cells, node_feat) tensor — same as PackGNN node feature matrix
    Output: same dict format as PackGNN for drop-in compatibility with
            GraphGuidedOptimizer (edge_index / edge_attr args ignored).

    ~97 K parameters (comparable to PackGNN 84 K) for fair comparison.
    """
    def __init__(self, n_cells: int = 12, node_feat: int = 7, hidden: int = 256):
        super().__init__()
        self.n_cells   = n_cells
        self.node_feat = node_feat
        in_dim         = n_cells * node_feat   # 84

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.head_soc = nn.Linear(hidden // 2, n_cells)
        self.head_dT  = nn.Linear(hidden // 2, n_cells)
        self.head_age = nn.Linear(hidden // 2, n_cells)
        self.head_imb = nn.Linear(hidden // 2, 1)

    def _backbone(self, flat: torch.Tensor):
        return self.backbone(flat)

    def forward(self, x: torch.Tensor,
                edge_index=None, edge_attr=None, **_) -> dict:
        """
        Inference mode — accepts same positional args as PackGNN.forward()
        so GraphGuidedOptimizer can use this as a drop-in GNN replacement.
        edge_index and edge_attr are ignored (no graph structure).
        x: (N_cells, node_feat)
        """
        flat = x.reshape(1, -1)           # (1, N*F)
        h    = self._backbone(flat)
        return {
            "soc_pred":     torch.sigmoid(self.head_soc(h)).squeeze(0),  # (N,)
            "delta_T_pred": self.head_dT(h).squeeze(0),                  # (N,)
            "aging_pred":   torch.relu(self.head_age(h)).squeeze(0),     # (N,)
            "imbalance":    torch.relu(self.head_imb(h)).squeeze(),       # scalar
        }

    def forward_batch(self, X: torch.Tensor):
        """Training mode — X: (B, N*F)."""
        h = self._backbone(X)
        return (torch.sigmoid(self.head_soc(h)),
                self.head_dT(h),
                torch.relu(self.head_age(h)),
                torch.relu(self.head_imb(h)))


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_training_data(parquet_path, cfg: dict,
                            n_rollouts: int = 200, steps_per_rollout: int = 18):
    """
    Generate (state → next_state) pairs from physics simulation.
    Uses seeds disjoint from eval episodes (eval uses ep*13, we use r*7+10000).
    """
    print(f"\n  Generating training data: {n_rollouts} rollouts × {steps_per_rollout} steps ...")
    X, ys, ydT, yage, yimb = [], [], [], [], []

    for r in range(n_rollouts):
        np.random.seed(r * 7 + 10000)
        soc0 = np.random.uniform(0.10, 0.50)
        ns   = np.random.uniform(0.01, 0.08)
        try:
            pack = build_pack_from_ecm(
                ecm_parquet=parquet_path,
                n_cells=cfg["n_cells"], chemistry=cfg["chemistry"],
                soc_init=soc0, soc_noise=ns, T_amb=cfg["T_amb"],
            )
        except Exception:
            continue

        Q_nom = np.array([c.Q_nom_Ah for c in pack.cells])

        for _ in range(steps_per_rollout):
            x_before  = pack.node_features()                        # (N, 7)
            T_before  = np.array([c.T_C for c in pack.cells])
            action    = np.random.uniform(0, cfg["I_max_C"]) * Q_nom
            metrics   = pack.step(action, dt=cfg["dt_s"])
            soc_after = np.array([c.SOC for c in pack.cells])
            T_after   = np.array([c.T_C for c in pack.cells])

            X.append(x_before.flatten())
            ys.append(soc_after)
            ydT.append(T_after - T_before)
            yage.append(np.full(cfg["n_cells"], metrics["aging_cost"]))
            yimb.append([metrics["SOC_imbalance"]])

            if np.mean(soc_after) >= 0.92:
                break

    to_t = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
    X, ys, ydT, yage, yimb = to_t(X), to_t(ys), to_t(ydT), to_t(yage), to_t(yimb)
    print(f"  Dataset: {X.shape[0]} samples, input dim {X.shape[1]}")
    return X, ys, ydT, yage, yimb


def train_pack_mlp(mlp: PackMLP, X, ys, ydT, yage, yimb,
                   epochs: int = 35, lr: float = 3e-4, bs: int = 128) -> PackMLP:
    mlp = mlp.to(DEVICE)
    opt = optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xd, ysd, ydTd, yaged, yimbd = [t.to(DEVICE) for t in [X, ys, ydT, yage, yimb]]
    N   = Xd.shape[0]

    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"\n  Training PackMLP ({n_params:,} params) on {N} samples "
          f"for {epochs} epochs ...")

    for ep in range(epochs):
        mlp.train()
        idx = torch.randperm(N, device=DEVICE)
        tot = 0.0; nb = 0
        for s in range(0, N, bs):
            b             = idx[s:s+bs]
            sp, dTp, ap, ip = mlp.forward_batch(Xd[b])
            loss = (10.0 * nn.MSELoss()(sp,  ysd[b])
                  +  1.0 * nn.MSELoss()(dTp, ydTd[b])
                  +  0.1 * nn.MSELoss()(ap,  yaged[b])
                  +  5.0 * nn.MSELoss()(ip,  yimbd[b]))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sch.step()
        if (ep + 1) % 10 == 0:
            print(f"    Epoch {ep+1:3d}/{epochs} | loss={tot/nb:.6f} | lr={sch.get_last_lr()[0]:.2e}")

    mlp.eval()
    print("  Training complete.")
    return mlp


# ═══════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER (self-contained, doesn't depend on run_episode internals)
# ═══════════════════════════════════════════════════════════════════════════

def run_eval(pack: BatteryPackGraph, controller, cfg: dict) -> dict:
    controller.reset()
    steps      = 0
    aging_cum  = 0.0
    viol_cum   = 0
    T_max_ep   = 0.0

    while steps < cfg["max_steps"]:
        if np.mean([c.SOC for c in pack.cells]) >= cfg["target_soc"]:
            break
        action   = controller.get_currents(pack)
        metrics  = pack.step(action, dt=cfg["dt_s"])
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

    # Paths
    ecm_files = sorted(ECM_DIR.glob("ecm_params_*.parquet"))
    assert ecm_files, f"No ECM parquet found in {ECM_DIR}"
    parquet_path = ecm_files[-1]
    print(f"ECM: {parquet_path.name}")

    ckpt_files = sorted(MDL_DIR.glob("pack_gnn_*.pt"))
    assert ckpt_files, f"No GNN checkpoint found in {MDL_DIR}"
    ckpt_path = ckpt_files[-1]
    print(f"GNN checkpoint: {ckpt_path.name}")

    # Load PackGNN
    gnn = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(DEVICE)
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    gnn.load_state_dict(ckpt.get("model_state", ckpt.get("model_state_dict", ckpt)))
    gnn.eval()

    # Train PackMLP
    mlp = PackMLP(n_cells=cfg["n_cells"]).to(DEVICE)
    X, ys, ydT, yage, yimb = generate_training_data(parquet_path, cfg,
                                                      n_rollouts=200,
                                                      steps_per_rollout=18)
    mlp = train_pack_mlp(mlp, X, ys, ydT, yage, yimb, epochs=35)

    # Save MLP weights for reproducibility
    mlp_path = MDL_DIR / f"pack_mlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({"model_state_dict": mlp.state_dict(),
                "n_cells": cfg["n_cells"], "hidden": 256}, str(mlp_path))
    print(f"  MLP saved → {mlp_path.name}")

    # Controllers
    controllers = {
        "GraphOptimizer": GraphGuidedOptimizer(cfg, gnn),
        "MLPOptimizer":   GraphGuidedOptimizer(cfg, mlp),  # drop-in
        "CC-CV":          CCCVController(cfg),
        "Proportional":   ProportionalController(cfg),
    }

    N_EPISODES = 30
    all_rows   = []

    print(f"\n{'='*60}")
    print(f"  Evaluation: N={N_EPISODES} episodes, same seeds as Table 4")
    print(f"{'='*60}")

    for name, ctrl in controllers.items():
        print(f"\n  Controller: {name}")
        ep_results = []
        t_start    = time.time()

        for ep in range(N_EPISODES):
            np.random.seed(ep * 7)
            torch.manual_seed(ep * 7)
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

        runtime = time.time() - t_start
        # Aggregate
        def agg(key): return float(np.mean([r[key] for r in ep_results]))
        row = {
            "method":           name,
            "charge_time_min":  round(agg("charge_time_min"), 2),
            "sigma_SOC_pct":    round(agg("sigma_SOC_pct"), 4),
            "T_max_C":          round(agg("T_max_C"), 2),
            "delta_T_C":        round(agg("delta_T_C"), 3),
            "aging_proxy_e4":   round(agg("aging_proxy"), 4),
            "violations":       round(agg("violations"), 2),
            "runtime_s":        round(runtime, 1),
            "n_episodes":       N_EPISODES,
        }
        all_rows.append(row)
        print(f"    → time={row['charge_time_min']}min  "
              f"σ={row['sigma_SOC_pct']}%  "
              f"viol={row['violations']}")

    # Save CSV
    out_csv = OUT_DIR / "mlp_mpc_vs_packgnn_n30.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n  Saved → {out_csv}")
    print("\n  Summary:")
    print(f"  {'Method':<20} {'Time(min)':>10} {'σ_SOC(%)':>10} "
          f"{'T_max':>8} {'ΔT':>7} {'Viol':>6}")
    for r in all_rows:
        print(f"  {r['method']:<20} {r['charge_time_min']:>10} {r['sigma_SOC_pct']:>10} "
              f"{r['T_max_C']:>8} {r['delta_T_C']:>7} {r['violations']:>6}")


if __name__ == "__main__":
    main()
