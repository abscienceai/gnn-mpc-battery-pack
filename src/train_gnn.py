"""
train_gnn.py
============
Dynamic Graph-Based Safe Fast Charging Optimization Project
------------------------------------------------------------
PURPOSE : Train PackGNN on simulated battery pack rollouts.

PIPELINE:
  1. Generate rollouts: random current profiles → pack states
  2. Build graph dataset: (graph_t, targets_t+1) pairs
  3. Train PackGNN with MSE loss on SOC, ΔT, aging predictions
  4. Save trained model → results/models/pack_gnn_{timestamp}.pt

TARGETS (per node, per step):
  - next_SOC     : SOC at t+1
  - delta_T      : T(t+1) - T(t)
  - aging_rate   : per-step aging cost

USAGE:
  python train_gnn.py [--n_rollouts 5000] [--epochs 50]
                      [--n_cells 12] [--chemistry LFP]
                      [--output_dir results/models]
"""

import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from graph_battery_pack import (
    BatteryPackGraph, PackGNN, build_pack_from_ecm
)

ECM_DIR  = Path("results/ecm")
MODEL_DIR = Path("results/models")


# ═══════════════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_rollout(pack: BatteryPackGraph, n_steps: int = 20,
                     I_max_C: float = 3.0,
                     rng: np.random.Generator = None) -> list:
    """
    Generate one rollout with random current profiles.
    Returns list of (x, edge_index, edge_attr, y_soc, y_dT, y_aging) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()

    samples = []
    Q_nom = np.array([c.Q_nom_Ah for c in pack.cells])
    I_max = I_max_C * Q_nom
    dt = 60.0

    for step in range(n_steps):
        # Record current graph state
        x          = pack.node_features().copy()        # (N, 7)
        edge_index = pack.edge_index.copy()              # (2, E)
        edge_attr  = pack.edge_features().copy()         # (E, 3)

        # Random current (mix of charging and rest)
        mode = rng.choice(["charge", "rest", "partial"],
                          p=[0.6, 0.1, 0.3])
        if mode == "charge":
            currents = rng.uniform(0.3, 1.0, size=len(pack.cells)) * I_max
        elif mode == "rest":
            currents = np.zeros(len(pack.cells))
        else:
            currents = rng.uniform(0.0, 0.5, size=len(pack.cells)) * I_max

        # Record pre-step SOC and T
        soc_before = np.array([c.SOC for c in pack.cells])
        T_before   = np.array([c.T_C for c in pack.cells])

        # Step
        metrics = pack.step(currents.astype(np.float32), dt=dt)

        # Targets
        soc_after = np.array([c.SOC for c in pack.cells])
        T_after   = np.array([c.T_C for c in pack.cells])
        delta_T   = T_after - T_before

        # Per-cell aging cost (from metrics, uniform across cells for now)
        aging_per_cell = np.full(len(pack.cells),
                                  metrics["aging_cost"] / len(pack.cells))

        samples.append((
            x.astype(np.float32),
            edge_index,
            edge_attr.astype(np.float32),
            soc_after.astype(np.float32),
            delta_T.astype(np.float32),
            aging_per_cell.astype(np.float32),
        ))

        # Stop if fully charged or any major violation
        if metrics["SOC_mean"] >= 0.95:
            break

    return samples


def generate_dataset(n_rollouts: int, n_cells: int, chemistry: str,
                     I_max_C: float = 3.0) -> list:
    """Generate n_rollouts simulation rollouts."""
    ecm_parquet = sorted(ECM_DIR.glob("*.parquet"))
    ecm_parquet = ecm_parquet[-1] if ecm_parquet else None

    all_samples = []
    rng = np.random.default_rng(42)

    # Pre-load ECM data once
    import pandas as pd
    if ecm_parquet:
        import pandas as _pd
        _df_full = _pd.read_parquet(ecm_parquet)
        _chem_ds = {'LFP':'MATR','NMC':'RWTH','LCO':'CALCE'}
        ecm_df = _df_full[_df_full['dataset']==_chem_ds.get(chemistry,'MATR')].dropna(subset=['IR_ohm']).reset_index(drop=True)
        print(f'  ECM df: {len(ecm_df)} rows', flush=True)
    else:
        ecm_df = None

    print(f"  Generating {n_rollouts} rollouts...", flush=True)
    import time as _time
    t_gen = _time.time()
    for i in range(n_rollouts):
        if i % 100 == 0:
            elapsed = _time.time() - t_gen
            eta = elapsed / max(i,1) * (n_rollouts - i)
            print(f"  [{i}/{n_rollouts}] samples={len(all_samples)} "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
        soc_init  = float(rng.uniform(0.10, 0.85))
        soc_noise = float(rng.uniform(0.01, 0.05))

        pack = build_pack_from_ecm(
            n_cells=n_cells, chemistry=chemistry,
            soc_init=soc_init, soc_noise=soc_noise,
            T_amb=float(rng.uniform(20, 35)),
            ecm_df=ecm_df,
        )

        n_steps = int(rng.integers(3, 10))
        samples = generate_rollout(pack, n_steps=n_steps,
                                    I_max_C=I_max_C, rng=rng)
        all_samples.extend(samples)

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_rollouts} rollouts | "
                  f"{len(all_samples)} samples so far")

    print(f"  Total samples: {len(all_samples)}")
    return all_samples


# ═══════════════════════════════════════════════════════════════════════════
#  PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════════════════

class GraphRolloutDataset(Dataset):
    """Dataset of (graph_state, targets) pairs from simulation rollouts."""

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, edge_index, edge_attr, y_soc, y_dT, y_aging = self.samples[idx]
        return (
            torch.tensor(x,          dtype=torch.float32),
            torch.tensor(edge_index, dtype=torch.long),
            torch.tensor(edge_attr,  dtype=torch.float32),
            torch.tensor(y_soc,      dtype=torch.float32),
            torch.tensor(y_dT,       dtype=torch.float32),
            torch.tensor(y_aging,    dtype=torch.float32),
        )


def collate_fn(batch):
    """Custom collate: handle variable graph sizes."""
    xs, eis, eas, y_socs, y_dTs, y_agings = zip(*batch)

    # Stack per-sample (all have same n_cells in our case)
    return (
        torch.stack(xs),
        torch.stack(eis),
        torch.stack(eas),
        torch.stack(y_socs),
        torch.stack(y_dTs),
        torch.stack(y_agings),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_gnn(samples: list, n_epochs: int = 50, batch_size: int = 256,
              lr: float = 1e-3, device: torch.device = None,
              output_dir: Path = None) -> dict:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Training PackGNN | device={device} | "
          f"samples={len(samples)} | epochs={n_epochs}")

    # Split train/val
    n = len(samples)
    idx = np.random.permutation(n)
    n_train = int(0.85 * n)
    train_samples = [samples[i] for i in idx[:n_train]]
    val_samples   = [samples[i] for i in idx[n_train:]]

    train_ds = GraphRolloutDataset(train_samples)
    val_ds   = GraphRolloutDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, collate_fn=collate_fn,
                               num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2,
                               shuffle=False, collate_fn=collate_fn,
                               num_workers=2)

    # Model
    model = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  PackGNN params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    # Loss weights
    w_soc   = 10.0   # SOC is most important
    w_dT    = 1.0
    w_aging = 0.1

    mse = nn.MSELoss()
    history = {"train_loss": [], "val_loss": [],
               "val_soc_mae": [], "val_dT_mae": []}
    best_val_loss = float("inf")
    best_state    = None

    print(f"\n  {'Ep':>4} {'Train':>10} {'Val':>10} "
          f"{'SOC_MAE':>9} {'dT_MAE':>8} {'LR':>9}")
    print(f"  {'-'*55}")

    for epoch in range(1, n_epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for x, ei, ea, y_soc, y_dT, y_aging in train_loader:
            x      = x.to(device)       # (B, N, 7)
            ei     = ei.to(device)      # (B, 2, E)
            ea     = ea.to(device)      # (B, E, 3)
            y_soc  = y_soc.to(device)   # (B, N)
            y_dT   = y_dT.to(device)
            y_aging= y_aging.to(device)

            optimizer.zero_grad()

            # Process each sample in batch (graph ops are per-sample)
            loss_batch = torch.tensor(0.0, device=device)
            for b in range(x.shape[0]):
                out = model(x[b], ei[b], ea[b])
                soc_pred  = out["soc_pred"].squeeze(-1)    # (N,)
                dT_pred   = out["delta_T_pred"].squeeze(-1) # (N,)
                aging_pred= out["aging_pred"].squeeze(-1)   # (N,)

                loss_b = (w_soc   * mse(soc_pred,   y_soc[b])  +
                          w_dT    * mse(dT_pred,    y_dT[b])   +
                          w_aging * mse(aging_pred, y_aging[b]))
                loss_batch = loss_batch + loss_b

            loss_batch = loss_batch / x.shape[0]
            loss_batch.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss_batch.item() * x.shape[0]

        train_loss /= len(train_ds)
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        soc_maes, dT_maes = [], []

        with torch.no_grad():
            for x, ei, ea, y_soc, y_dT, y_aging in val_loader:
                x = x.to(device); ei = ei.to(device)
                ea = ea.to(device)
                y_soc = y_soc.to(device); y_dT = y_dT.to(device)
                y_aging = y_aging.to(device)

                for b in range(x.shape[0]):
                    out = model(x[b], ei[b], ea[b])
                    soc_pred  = out["soc_pred"].squeeze(-1)
                    dT_pred   = out["delta_T_pred"].squeeze(-1)
                    aging_pred= out["aging_pred"].squeeze(-1)

                    loss_b = (w_soc   * mse(soc_pred,   y_soc[b]) +
                              w_dT    * mse(dT_pred,    y_dT[b])  +
                              w_aging * mse(aging_pred, y_aging[b]))
                    val_loss += loss_b.item()

                    soc_maes.append(
                        float(torch.abs(soc_pred - y_soc[b]).mean()))
                    dT_maes.append(
                        float(torch.abs(dT_pred - y_dT[b]).mean()))

        val_loss  /= len(val_ds)
        soc_mae    = float(np.mean(soc_maes))
        dT_mae     = float(np.mean(dT_maes))
        lr_now     = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_soc_mae"].append(soc_mae)
        history["val_dT_mae"].append(dT_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"  {epoch:>4} {train_loss:>10.6f} {val_loss:>10.6f} "
                  f"{soc_mae*100:>8.3f}% {dT_mae:>8.4f}°C {lr_now:>9.2e}")

    # Save
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"pack_gnn_{ts}.pt"

    torch.save({
        "model_state": best_state,
        "n_features":  7,
        "edge_feat":   3,
        "hidden":      64,
        "n_layers":    3,
        "best_val_loss": best_val_loss,
        "history":     history,
    }, model_path)

    print(f"\n  ✅ Model saved → {model_path}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Final SOC MAE: {history['val_soc_mae'][-1]*100:.3f}%")
    print(f"  Final ΔT  MAE: {history['val_dT_mae'][-1]:.4f}°C")

    return {
        "model_path":    str(model_path),
        "best_val_loss": best_val_loss,
        "final_soc_mae": history["val_soc_mae"][-1],
        "final_dT_mae":  history["val_dT_mae"][-1],
        "history":       history,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train PackGNN on simulation rollouts.")
    parser.add_argument("--n_rollouts", type=int,   default=5000)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch",      type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--n_cells",    type=int,   default=12)
    parser.add_argument("--chemistry",  type=str,   default="LFP")
    parser.add_argument("--output_dir", type=str,   default="results/models")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  PackGNN Training — Dynamic Graph Charging Project")
    print("=" * 65)
    print(f"  Device    : {device}")
    if device.type == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  Rollouts  : {args.n_rollouts}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Chemistry : {args.chemistry} | Cells: {args.n_cells}")

    # Generate data
    print("\n── Step 1: Data Generation ─────────────────────────────────")
    samples = generate_dataset(
        n_rollouts = args.n_rollouts,
        n_cells    = args.n_cells,
        chemistry  = args.chemistry,
    )

    # Train
    print("\n── Step 2: GNN Training ─────────────────────────────────────")
    metrics = train_gnn(
        samples    = samples,
        n_epochs   = args.epochs,
        batch_size = args.batch,
        lr         = args.lr,
        device     = device,
        output_dir = Path(args.output_dir),
    )

    print("\n" + "=" * 65)
    print("  ✅ GNN Training Complete!")
    print(f"  SOC MAE : {metrics['final_soc_mae']*100:.3f}%")
    print(f"  ΔT  MAE : {metrics['final_dT_mae']:.4f}°C")
    print("=" * 65)


if __name__ == "__main__":
    main()
