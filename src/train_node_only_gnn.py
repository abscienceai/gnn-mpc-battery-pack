"""
train_node_only_gnn.py
======================
Trains NodeOnlyGNN (no graph edges, per-node MLP) on the SAME 5,000 rollouts
used to train PackGNN, using the same training protocol.

This provides a fair ablation comparison:
  - Same training data    (seed=42, n_rollouts=5000, chemistry=LFP)
  - Same training budget  (50 epochs, AdamW, cosine LR, same loss weights)
  - Comparable parameter count (~85K vs PackGNN 84K)
  - Same train/val split

Output: results/models/node_only_gnn_{timestamp}.pt

Usage:
  python train_node_only_gnn.py [--epochs 50] [--n_rollouts 5000]
"""

import sys, argparse, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from graph_battery_pack import BatteryPackGraph
from train_gnn import generate_dataset, GraphRolloutDataset, collate_fn

MODEL_DIR = Path(__file__).parent.parent / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════
#  NODE-ONLY GNN  (no edge message passing, per-node MLP)
# ═══════════════════════════════════════════════════════════════════════════

class NodeOnlyGNN(nn.Module):
    """
    Per-node MLP surrogate — no inter-cell message passing, no graph edges.
    Same output interface as PackGNN for drop-in compatibility.

    Architecture: 7 → 256 → 256 → 64 → heads
    Parameters:   ~85,444  (comparable to PackGNN 84,804)
    """
    def __init__(self, node_feat: int = 7, hidden: int = 256, hidden2: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(node_feat, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
        )
        # Per-node prediction heads (same as PackGNN)
        self.soc_head    = nn.Linear(hidden2, 1)
        self.dT_head     = nn.Linear(hidden2, 1)
        self.aging_head  = nn.Linear(hidden2, 1)
        # Pack-level imbalance head (aggregated from all nodes)
        self.imb_head    = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor,
                edge_index=None, edge_attr=None, **_) -> dict:
        """
        x: (N_cells, node_feat)
        edge_index, edge_attr: ignored (no graph structure)
        Returns same dict format as PackGNN.
        """
        h = self.backbone(x)                                 # (N, hidden2)
        soc_pred   = torch.sigmoid(self.soc_head(h)).squeeze(-1)   # (N,)
        dT_pred    = self.dT_head(h).squeeze(-1)                   # (N,)
        aging_pred = torch.relu(self.aging_head(h)).squeeze(-1)    # (N,)
        imbalance  = torch.relu(self.imb_head(h).mean())           # scalar
        return {
            "soc_pred":     soc_pred,
            "delta_T_pred": dT_pred,
            "aging_pred":   aging_pred,
            "imbalance":    imbalance,
        }


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train(samples, epochs=50, batch_size=256, lr=1e-3):
    model = NodeOnlyGNN().to(DEVICE)
    print(f"  NodeOnlyGNN params: {count_params(model):,}  "
          f"(PackGNN reference: 84,804)")

    n = len(samples)
    idx = np.random.RandomState(42).permutation(n)   # same split as PackGNN
    n_train = int(0.85 * n)
    train_ds = GraphRolloutDataset([samples[i] for i in idx[:n_train]])
    val_ds   = GraphRolloutDataset([samples[i] for i in idx[n_train:]])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=4,
                               pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False,
                               collate_fn=collate_fn, num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=1e-5)
    mse = nn.MSELoss()
    w_soc, w_dT, w_aging = 10.0, 1.0, 0.1

    best_val, best_state = float("inf"), None
    history = {"train_loss": [], "val_loss": [], "val_soc_mae": [], "val_dT_mae": []}

    print(f"\n  {'Ep':>4} {'Train':>10} {'Val':>10} "
          f"{'SOC_MAE':>9} {'dT_MAE':>8} {'LR':>9}")
    print(f"  {'-'*55}")

    for epoch in range(1, epochs + 1):
        model.train(); t_loss = 0.0
        for x, ei, ea, y_soc, y_dT, y_aging in train_loader:
            x = x.to(DEVICE); y_soc = y_soc.to(DEVICE)
            y_dT = y_dT.to(DEVICE); y_aging = y_aging.to(DEVICE)
            opt.zero_grad()
            loss_b = torch.tensor(0.0, device=DEVICE)
            for b in range(x.shape[0]):
                out = model(x[b])   # edge_index/attr ignored
                loss_b += (w_soc   * mse(out["soc_pred"],     y_soc[b])  +
                           w_dT    * mse(out["delta_T_pred"],  y_dT[b])   +
                           w_aging * mse(out["aging_pred"],    y_aging[b]))
            loss_b = loss_b / x.shape[0]
            loss_b.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss_b.item() * x.shape[0]
        t_loss /= len(train_ds); sch.step()

        model.eval(); v_loss = 0.0; soc_maes = []; dT_maes = []
        with torch.no_grad():
            for x, ei, ea, y_soc, y_dT, y_aging in val_loader:
                x = x.to(DEVICE); y_soc = y_soc.to(DEVICE)
                y_dT = y_dT.to(DEVICE); y_aging = y_aging.to(DEVICE)
                for b in range(x.shape[0]):
                    out = model(x[b])
                    v_loss += (w_soc   * mse(out["soc_pred"],    y_soc[b]) +
                               w_dT    * mse(out["delta_T_pred"], y_dT[b])  +
                               w_aging * mse(out["aging_pred"],   y_aging[b])).item()
                    soc_maes.append(float(torch.abs(out["soc_pred"]-y_soc[b]).mean()))
                    dT_maes.append(float(torch.abs(out["delta_T_pred"]-y_dT[b]).mean()))
        v_loss /= len(val_ds)
        soc_mae = float(np.mean(soc_maes))
        dT_mae  = float(np.mean(dT_maes))

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_soc_mae"].append(soc_mae)
        history["val_dT_mae"].append(dT_mae)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  {epoch:>4} {t_loss:>10.6f} {v_loss:>10.6f} "
                  f"{soc_mae*100:>8.3f}% {dT_mae:>8.4f}°C "
                  f"{sch.get_last_lr()[0]:>9.2e}", flush=True)

    model.load_state_dict(best_state)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MODEL_DIR / f"node_only_gnn_{ts}.pt"
    torch.save({"model_state":      best_state,
                "architecture":     "NodeOnlyGNN",
                "node_feat":        7,
                "hidden":           256,
                "hidden2":          64,
                "n_params":         count_params(model),
                "best_val_loss":    best_val,
                "history":          history}, path)
    print(f"\n  ✅ Saved → {path}")
    print(f"  Best val loss : {best_val:.6f}")
    print(f"  Final SOC MAE : {history['val_soc_mae'][-1]*100:.3f}%")
    print(f"  Final ΔT  MAE : {history['val_dT_mae'][-1]:.4f}°C")
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rollouts", type=int, default=5000)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch",      type=int, default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"\n── Generating dataset (n_rollouts={args.n_rollouts}, "
          f"same seed=42 as PackGNN) ──")
    samples = generate_dataset(args.n_rollouts, n_cells=12, chemistry="LFP")
    print(f"  Total samples: {len(samples):,}")

    print(f"\n── Training NodeOnlyGNN ({args.epochs} epochs) ──")
    ckpt = train(samples, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
    print(f"\nDone. Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
