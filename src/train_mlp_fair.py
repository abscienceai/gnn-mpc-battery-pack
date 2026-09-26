"""Retrain flat MLP on same 5000 rollouts as PackGNN for fair comparison."""
import sys, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from train_gnn import generate_dataset, GraphRolloutDataset, collate_fn

MODEL_DIR = Path(__file__).parent.parent / "results" / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PackMLP(nn.Module):
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
        return {"soc_pred": torch.sigmoid(self.head_soc(h)).squeeze(0),
                "delta_T_pred": self.head_dT(h).squeeze(0),
                "aging_pred": torch.relu(self.head_age(h)).squeeze(0),
                "imbalance": torch.relu(self.head_imb(h)).squeeze()}

    def forward_batch(self, X):
        h = self.backbone(X)
        return (torch.sigmoid(self.head_soc(h)), self.head_dT(h),
                torch.relu(self.head_age(h)), torch.relu(self.head_imb(h)))

print(f"Device: {DEVICE}")
print("Generating 5000 rollouts (same seed=42 as PackGNN)...")
samples = generate_dataset(5000, n_cells=12, chemistry="LFP")
print(f"Total samples: {len(samples):,}")

mlp = PackMLP().to(DEVICE)
n_params = sum(p.numel() for p in mlp.parameters())
print(f"PackMLP params: {n_params:,}")

n = len(samples)
idx = np.random.RandomState(42).permutation(n)
n_train = int(0.85 * n)
train_ds = GraphRolloutDataset([samples[i] for i in idx[:n_train]])
val_ds   = GraphRolloutDataset([samples[i] for i in idx[n_train:]])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                           collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False,
                           collate_fn=collate_fn, num_workers=2)

opt = torch.optim.AdamW(mlp.parameters(), lr=3e-4, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35, eta_min=1e-5)
mse = nn.MSELoss()

print(f"\n{'Ep':>4} {'Train':>10} {'Val':>10} {'SOC_MAE':>9} {'dT_MAE':>8}")
best_val = float("inf"); best_state = None
for epoch in range(1, 36):
    mlp.train(); t_loss = 0.0
    for x, ei, ea, y_soc, y_dT, y_aging in train_loader:
        Xb = x.to(DEVICE).reshape(x.shape[0], -1)
        sp, dTp, ap, ip = mlp.forward_batch(Xb)
        loss = (10.0*mse(sp, y_soc.to(DEVICE)) + 1.0*mse(dTp, y_dT.to(DEVICE)) +
                0.1*mse(ap, y_aging.to(DEVICE)) + 5.0*mse(ip, y_soc.to(DEVICE).std(1,keepdim=True)))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        opt.step()
        t_loss += loss.item() * x.shape[0]
    t_loss /= len(train_ds); sch.step()

    mlp.eval(); v_loss = 0.0; soc_m = []; dT_m = []
    with torch.no_grad():
        for x, ei, ea, y_soc, y_dT, y_aging in val_loader:
            Xb = x.to(DEVICE).reshape(x.shape[0], -1)
            sp, dTp, ap, ip = mlp.forward_batch(Xb)
            v_loss += (10.0*mse(sp,y_soc.to(DEVICE))+mse(dTp,y_dT.to(DEVICE))).item()*x.shape[0]
            soc_m.append(float((sp-y_soc.to(DEVICE)).abs().mean()))
            dT_m.append(float((dTp-y_dT.to(DEVICE)).abs().mean()))
    v_loss /= len(val_ds)
    if v_loss < best_val: best_val = v_loss; best_state = {k:v.clone() for k,v in mlp.state_dict().items()}
    if epoch%5==0 or epoch==1:
        print(f"{epoch:>4} {t_loss:>10.6f} {v_loss:>10.6f} {np.mean(soc_m)*100:>8.3f}% {np.mean(dT_m):>8.4f}°C", flush=True)

mlp.load_state_dict(best_state)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
path = MODEL_DIR / f"pack_mlp_fair5k_{ts}.pt"
torch.save({"model_state_dict": best_state, "n_cells": 12, "hidden": 256}, path)
print(f"\nSaved → {path}")
