#!/usr/bin/env python3
"""
Train action-conditioned PackGNN v2.

Scientific protocol
-------------------
Each supervised sample is:

    (G_t, u_t) -> (SOC_{t+1}, DeltaT_{t+1}, aging_{t+1})

where u_t is the candidate per-cell charging action normalised by the
configured maximum current of that cell.

Validation splitting is performed by complete rollout ID, never by timestep,
so adjacent samples from one simulated trajectory cannot appear in both
training and validation.

Pack-realisation seeds are unique per rollout and intentionally placed far
outside the canonical evaluation seed set {0, 7, 14, ...}.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from graph_battery_pack import (
    PackGNN,
    build_pack_from_ecm,
)


ROOT = Path(__file__).resolve().parent.parent
ECM_DIR = ROOT / "results" / "ecm"
MODEL_DIR = ROOT / "results" / "models"


def generate_rollout(
    pack,
    rollout_id: str,
    pack_seed: int,
    n_steps: int = 20,
    I_max_C: float = 3.0,
    rng: np.random.Generator | None = None,
):
    """
    Generate one causal action-conditioned rollout.

    Every sample stores:
        x_t
        edge_index
        edge_attr_t
        u_t
        SOC_{t+1}
        DeltaT_{t+1}
        aging proxy
        rollout_id
        pack_seed
    """
    if rng is None:
        rng = np.random.default_rng()

    samples = []

    q_nom = np.asarray(
        [c.Q_nom_Ah for c in pack.cells],
        dtype=np.float64,
    )

    i_max = (
        float(I_max_C)
        * q_nom
    )

    dt = 60.0

    for _step in range(n_steps):

        x = pack.node_features().copy()
        edge_index = pack.edge_index.copy()
        edge_attr = pack.edge_features().copy()

        mode = rng.choice(
            ["charge", "rest", "partial"],
            p=[0.60, 0.10, 0.30],
        )

        if mode == "charge":
            currents = (
                rng.uniform(
                    0.30,
                    1.00,
                    size=len(pack.cells),
                )
                * i_max
            )

        elif mode == "rest":
            currents = np.zeros(
                len(pack.cells),
                dtype=np.float64,
            )

        else:
            currents = (
                rng.uniform(
                    0.00,
                    0.50,
                    size=len(pack.cells),
                )
                * i_max
            )

        # Explicit action feature used by PackGNN v2.
        action = np.clip(
            currents
            / (i_max + 1e-12),
            0.0,
            1.0,
        ).astype(np.float32)

        T_before = np.asarray(
            [c.T_C for c in pack.cells],
            dtype=np.float64,
        )

        metrics = pack.step(
            currents.astype(np.float32),
            dt=dt,
        )

        soc_after = np.asarray(
            [c.SOC for c in pack.cells],
            dtype=np.float32,
        )

        T_after = np.asarray(
            [c.T_C for c in pack.cells],
            dtype=np.float64,
        )

        delta_T = (
            T_after - T_before
        ).astype(np.float32)

        aging_per_cell = np.full(
            len(pack.cells),
            metrics["aging_cost"]
            / len(pack.cells),
            dtype=np.float32,
        )

        samples.append({
            "x": x.astype(np.float32),
            "edge_index": edge_index,
            "edge_attr": edge_attr.astype(np.float32),
            "action": action,
            "y_soc": soc_after,
            "y_dT": delta_T,
            "y_aging": aging_per_cell,
            "rollout_id": str(rollout_id),
            "pack_seed": int(pack_seed),
        })

        if metrics["SOC_mean"] >= 0.95:
            break

    return samples


def generate_dataset(
    n_rollouts: int,
    n_cells: int,
    chemistry: str,
    I_max_C: float = 3.0,
    seed: int = 42,
):
    """
    Generate independent pack rollouts.

    Each rollout receives a unique pack-realisation seed. The offset keeps
    training pack seeds disjoint from canonical evaluation seeds such as
    0, 7, 14, ..., 203.
    """

    chemistry = chemistry.upper()

    ecm_candidates = sorted(
        ECM_DIR.glob("*.parquet")
    )

    ecm_parquet = (
        ecm_candidates[-1]
        if ecm_candidates
        else None
    )

    if ecm_parquet is None:
        raise FileNotFoundError(
            f"No ECM parquet found in {ECM_DIR}"
        )

    import pandas as pd

    df_full = pd.read_parquet(
        ecm_parquet
    )

    chemistry_dataset = {
        "LFP": "MATR",
        "NMC": "RWTH",
        "LCO": "CALCE",
    }

    dataset_name = chemistry_dataset.get(
        chemistry
    )

    if dataset_name is None:
        raise ValueError(
            f"Unsupported chemistry: {chemistry}"
        )

    ecm_df = (
        df_full[
            df_full["dataset"]
            == dataset_name
        ]
        .dropna(
            subset=["IR_ohm"]
        )
        .reset_index(drop=True)
    )

    if len(ecm_df) == 0:
        raise RuntimeError(
            f"No valid ECM rows for {chemistry}/{dataset_name}"
        )

    print(
        f"  ECM df: {len(ecm_df)} rows "
        f"[{dataset_name}]",
        flush=True,
    )

    master_rng = np.random.default_rng(
        seed
    )

    all_samples = []

    print(
        f"  Generating {n_rollouts} independent rollouts...",
        flush=True,
    )

    import time
    t0 = time.time()

    for i in range(n_rollouts):

        if i % 100 == 0:
            elapsed = time.time() - t0
            eta = (
                elapsed
                / max(i, 1)
                * (n_rollouts - i)
            )

            print(
                f"  [{i}/{n_rollouts}] "
                f"samples={len(all_samples)} "
                f"elapsed={elapsed:.0f}s "
                f"ETA={eta:.0f}s",
                flush=True,
            )

        soc_init = float(
            master_rng.uniform(
                0.10,
                0.85,
            )
        )

        soc_noise = float(
            master_rng.uniform(
                0.01,
                0.05,
            )
        )

        T_amb = float(
            master_rng.uniform(
                20.0,
                35.0,
            )
        )

        # Explicitly disjoint from canonical evaluation seeds.
        pack_seed = (
            1_000_000
            + int(seed) * 100_000
            + i
        )

        rollout_id = (
            f"{chemistry}_"
            f"{i:06d}_"
            f"seed{pack_seed}"
        )

        pack = build_pack_from_ecm(
            n_cells=n_cells,
            chemistry=chemistry,
            soc_init=soc_init,
            soc_noise=soc_noise,
            T_amb=T_amb,
            ecm_df=ecm_df,
            seed=pack_seed,
        )

        n_steps = int(
            master_rng.integers(
                3,
                10,
            )
        )

        rollout_rng = np.random.default_rng(
            pack_seed + 7919
        )

        samples = generate_rollout(
            pack=pack,
            rollout_id=rollout_id,
            pack_seed=pack_seed,
            n_steps=n_steps,
            I_max_C=I_max_C,
            rng=rollout_rng,
        )

        all_samples.extend(
            samples
        )

    print(
        f"  Total samples: {len(all_samples)}"
    )

    print(
        f"  Unique rollouts: "
        f"{len(set(s['rollout_id'] for s in all_samples))}"
    )

    return all_samples


def split_by_rollout(
    samples,
    val_fraction: float = 0.15,
    seed: int = 42,
):
    """
    Group-safe train/validation split.

    No rollout ID can occur in both partitions.
    """

    rollout_ids = sorted(
        set(
            s["rollout_id"]
            for s in samples
        )
    )

    if len(rollout_ids) < 2:
        raise ValueError(
            "At least two rollouts are required for group split."
        )

    rng = np.random.default_rng(
        seed
    )

    ids = np.asarray(
        rollout_ids,
        dtype=object,
    )

    rng.shuffle(ids)

    n_val = max(
        1,
        int(
            round(
                len(ids)
                * val_fraction
            )
        ),
    )

    n_val = min(
        n_val,
        len(ids) - 1,
    )

    val_ids = set(
        ids[:n_val].tolist()
    )

    train_ids = set(
        ids[n_val:].tolist()
    )

    if train_ids & val_ids:
        raise RuntimeError(
            "Rollout leakage detected."
        )

    train_samples = [
        s
        for s in samples
        if s["rollout_id"]
        in train_ids
    ]

    val_samples = [
        s
        for s in samples
        if s["rollout_id"]
        in val_ids
    ]

    return (
        train_samples,
        val_samples,
        train_ids,
        val_ids,
    )


class GraphRolloutDataset(Dataset):

    def __init__(
        self,
        samples,
    ):
        self.samples = samples

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx,
    ):
        s = self.samples[idx]

        return (
            torch.tensor(
                s["x"],
                dtype=torch.float32,
            ),
            torch.tensor(
                s["edge_index"],
                dtype=torch.long,
            ),
            torch.tensor(
                s["edge_attr"],
                dtype=torch.float32,
            ),
            torch.tensor(
                s["action"],
                dtype=torch.float32,
            ),
            torch.tensor(
                s["y_soc"],
                dtype=torch.float32,
            ),
            torch.tensor(
                s["y_dT"],
                dtype=torch.float32,
            ),
            torch.tensor(
                s["y_aging"],
                dtype=torch.float32,
            ),
        )


def collate_fn(
    batch,
):
    (
        xs,
        eis,
        eas,
        actions,
        y_socs,
        y_dTs,
        y_agings,
    ) = zip(*batch)

    return (
        torch.stack(xs),
        torch.stack(eis),
        torch.stack(eas),
        torch.stack(actions),
        torch.stack(y_socs),
        torch.stack(y_dTs),
        torch.stack(y_agings),
    )


def train_gnn(
    samples,
    n_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device=None,
    output_dir: Path | None = None,
    pretrained: str | None = None,
    split_seed: int = 42,
):
    if output_dir is None:
        output_dir = MODEL_DIR

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    (
        train_samples,
        val_samples,
        train_ids,
        val_ids,
    ) = split_by_rollout(
        samples,
        val_fraction=0.15,
        seed=split_seed,
    )

    print()
    print(
        f"  Training PackGNN v2 | "
        f"device={device} | "
        f"samples={len(samples)} | "
        f"epochs={n_epochs}"
    )

    print(
        f"  Group split: "
        f"train={len(train_ids)} rollouts/"
        f"{len(train_samples)} samples | "
        f"val={len(val_ids)} rollouts/"
        f"{len(val_samples)} samples"
    )

    overlap = (
        set(train_ids)
        & set(val_ids)
    )

    if overlap:
        raise RuntimeError(
            f"Train/val rollout leakage: {overlap}"
        )

    train_ds = GraphRolloutDataset(
        train_samples
    )

    val_ds = GraphRolloutDataset(
        val_samples
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = PackGNN(
        node_feat=7,
        edge_feat=3,
        hidden=64,
        n_layers=3,
        action_feat=1,
    ).to(device)

    if pretrained:

        ckpt = torch.load(
            pretrained,
            map_location=device,
        )

        if (
            ckpt.get(
                "model_format_version"
            )
            != 2
        ):
            raise RuntimeError(
                "Pretrained checkpoint is not PackGNN v2."
            )

        if not ckpt.get(
            "action_conditioned",
            False,
        ):
            raise RuntimeError(
                "Pretrained checkpoint is not action-conditioned."
            )

        model.load_state_dict(
            ckpt["model_state"],
            strict=True,
        )

        print(
            f"  Loaded v2 pretrained weights: "
            f"{Path(pretrained).name}"
        )

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"  PackGNN v2 params: {n_params:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=n_epochs,
            eta_min=1e-5,
        )
    )

    # Explicitly supervised outputs.
    w_soc = 10.0
    w_dT = 1.0
    w_aging = 0.1
    w_imbalance = 1.0

    mse = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_soc_mae": [],
        "val_dT_mae": [],
        "val_imbalance_mae": [],
    }

    best_val_loss = float(
        "inf"
    )

    best_state = None
    best_epoch = None
    best_metrics = None

    print()
    print(
        f"  {'Ep':>4} "
        f"{'Train':>10} "
        f"{'Val':>10} "
        f"{'SOC_MAE':>9} "
        f"{'dT_MAE':>8} "
        f"{'IMB_MAE':>8} "
        f"{'LR':>9}"
    )

    print(
        "  "
        + "-" * 70
    )

    for epoch in range(
        1,
        n_epochs + 1,
    ):

        model.train()

        train_loss = 0.0

        for (
            x,
            ei,
            ea,
            action,
            y_soc,
            y_dT,
            y_aging,
        ) in train_loader:

            x = x.to(device)
            ei = ei.to(device)
            ea = ea.to(device)
            action = action.to(device)

            y_soc = y_soc.to(device)
            y_dT = y_dT.to(device)
            y_aging = y_aging.to(device)

            optimizer.zero_grad()

            loss_batch = torch.tensor(
                0.0,
                device=device,
            )

            for b in range(
                x.shape[0]
            ):

                out = model(
                    x[b],
                    ei[b],
                    ea[b],
                    action[b],
                )

                soc_pred = (
                    out["soc_pred"]
                    .squeeze(-1)
                )

                dT_pred = (
                    out["delta_T_pred"]
                    .squeeze(-1)
                )

                aging_pred = (
                    out["aging_pred"]
                    .squeeze(-1)
                )

                imb_pred = out[
                    "imbalance"
                ]

                # Next-state SOC imbalance is the pack-level supervision target.
                imb_target = torch.std(
                    y_soc[b],
                    unbiased=False,
                )

                loss_b = (
                    w_soc
                    * mse(
                        soc_pred,
                        y_soc[b],
                    )
                    +
                    w_dT
                    * mse(
                        dT_pred,
                        y_dT[b],
                    )
                    +
                    w_aging
                    * mse(
                        aging_pred,
                        y_aging[b],
                    )
                    +
                    w_imbalance
                    * mse(
                        imb_pred,
                        imb_target,
                    )
                )

                loss_batch = (
                    loss_batch
                    + loss_b
                )

            loss_batch = (
                loss_batch
                / x.shape[0]
            )

            loss_batch.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            train_loss += (
                loss_batch.item()
                * x.shape[0]
            )

        train_loss /= len(
            train_ds
        )

        scheduler.step()

        # =============================================================
        # Validation
        # =============================================================
        model.eval()

        val_loss = 0.0
        soc_maes = []
        dT_maes = []
        imb_maes = []

        with torch.no_grad():

            for (
                x,
                ei,
                ea,
                action,
                y_soc,
                y_dT,
                y_aging,
            ) in val_loader:

                x = x.to(device)
                ei = ei.to(device)
                ea = ea.to(device)
                action = action.to(device)

                y_soc = y_soc.to(device)
                y_dT = y_dT.to(device)
                y_aging = y_aging.to(device)

                for b in range(
                    x.shape[0]
                ):

                    out = model(
                        x[b],
                        ei[b],
                        ea[b],
                        action[b],
                    )

                    soc_pred = (
                        out["soc_pred"]
                        .squeeze(-1)
                    )

                    dT_pred = (
                        out["delta_T_pred"]
                        .squeeze(-1)
                    )

                    aging_pred = (
                        out["aging_pred"]
                        .squeeze(-1)
                    )

                    imb_pred = out[
                        "imbalance"
                    ]

                    imb_target = torch.std(
                        y_soc[b],
                        unbiased=False,
                    )

                    loss_b = (
                        w_soc
                        * mse(
                            soc_pred,
                            y_soc[b],
                        )
                        +
                        w_dT
                        * mse(
                            dT_pred,
                            y_dT[b],
                        )
                        +
                        w_aging
                        * mse(
                            aging_pred,
                            y_aging[b],
                        )
                        +
                        w_imbalance
                        * mse(
                            imb_pred,
                            imb_target,
                        )
                    )

                    val_loss += (
                        loss_b.item()
                    )

                    soc_maes.append(
                        float(
                            torch.abs(
                                soc_pred
                                - y_soc[b]
                            ).mean()
                        )
                    )

                    dT_maes.append(
                        float(
                            torch.abs(
                                dT_pred
                                - y_dT[b]
                            ).mean()
                        )
                    )

                    imb_maes.append(
                        float(
                            torch.abs(
                                imb_pred
                                - imb_target
                            )
                        )
                    )

        val_loss /= len(
            val_ds
        )

        soc_mae = float(
            np.mean(
                soc_maes
            )
        )

        dT_mae = float(
            np.mean(
                dT_maes
            )
        )

        imb_mae = float(
            np.mean(
                imb_maes
            )
        )

        lr_now = (
            scheduler
            .get_last_lr()[0]
        )

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "val_loss"
        ].append(
            val_loss
        )

        history[
            "val_soc_mae"
        ].append(
            soc_mae
        )

        history[
            "val_dT_mae"
        ].append(
            dT_mae
        )

        history[
            "val_imbalance_mae"
        ].append(
            imb_mae
        )

        if (
            val_loss
            < best_val_loss
        ):
            best_val_loss = (
                val_loss
            )

            best_epoch = epoch

            best_state = {
                k:
                v.detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict().items()
            }

            best_metrics = {
                "soc_mae":
                    soc_mae,
                "dT_mae":
                    dT_mae,
                "imbalance_mae":
                    imb_mae,
            }

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == n_epochs
        ):
            print(
                f"  {epoch:>4} "
                f"{train_loss:>10.6f} "
                f"{val_loss:>10.6f} "
                f"{soc_mae*100:>8.3f}% "
                f"{dT_mae:>8.4f} "
                f"{imb_mae:>8.4f} "
                f"{lr_now:>9.2e}"
            )

    if best_state is None:
        raise RuntimeError(
            "Training produced no checkpoint."
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ts = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    model_path = (
        output_dir
        / f"pack_gnn_action_v2_{ts}.pt"
    )

    train_pack_seeds = sorted(
        set(
            int(s["pack_seed"])
            for s in train_samples
        )
    )

    val_pack_seeds = sorted(
        set(
            int(s["pack_seed"])
            for s in val_samples
        )
    )

    torch.save({
        "model_format_version": 2,
        "action_conditioned": True,
        "action_definition":
            "u_i = I_i / (I_max_C * Q_nom_i)",
        "model_state": best_state,
        "node_features": 7,
        "action_features": 1,
        "edge_features": 3,
        "hidden": 64,
        "n_layers": 3,
        "best_epoch": best_epoch,
        "best_val_loss":
            best_val_loss,
        "best_metrics":
            best_metrics,
        "history":
            history,
        "split_method":
            "group_by_rollout_id",
        "split_seed":
            split_seed,
        "train_rollout_ids":
            sorted(train_ids),
        "val_rollout_ids":
            sorted(val_ids),
        "train_pack_seeds":
            train_pack_seeds,
        "val_pack_seeds":
            val_pack_seeds,
    }, model_path)

    print()
    print(
        f"  Model saved -> {model_path}"
    )

    print(
        f"  Best epoch: {best_epoch}"
    )

    print(
        f"  Best val loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"  Best SOC MAE: "
        f"{best_metrics['soc_mae']*100:.3f}%"
    )

    print(
        f"  Best DeltaT MAE: "
        f"{best_metrics['dT_mae']:.4f} C"
    )

    print(
        f"  Best imbalance MAE: "
        f"{best_metrics['imbalance_mae']:.5f}"
    )

    return {
        "model_path":
            str(model_path),
        "best_epoch":
            best_epoch,
        "best_val_loss":
            best_val_loss,
        "best_soc_mae":
            best_metrics["soc_mae"],
        "best_dT_mae":
            best_metrics["dT_mae"],
        "best_imbalance_mae":
            best_metrics[
                "imbalance_mae"
            ],
        "train_rollouts":
            len(train_ids),
        "val_rollouts":
            len(val_ids),
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train action-conditioned PackGNN v2 "
            "with rollout-level validation split."
        )
    )

    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--n_cells",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--chemistry",
        type=str,
        default="LFP",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/models",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    args = parser.parse_args()

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    if args.device == "cpu":
        device = torch.device(
            "cpu"
        )

    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        device = torch.device(
            "cuda"
        )

    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print("=" * 72)
    print(
        "PackGNN v2 Action-Conditioned Training"
    )
    print("=" * 72)

    print(
        f"Device:      {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU:         "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Rollouts:    {args.n_rollouts}"
    )

    print(
        f"Epochs:      {args.epochs}"
    )

    print(
        f"Chemistry:   "
        f"{args.chemistry.upper()}"
    )

    print(
        f"Cells:       {args.n_cells}"
    )

    print()
    print(
        "Step 1: independent rollout generation"
    )

    samples = generate_dataset(
        n_rollouts=args.n_rollouts,
        n_cells=args.n_cells,
        chemistry=args.chemistry,
        I_max_C=3.0,
        seed=args.seed,
    )

    print()
    print(
        "Step 2: group-safe PackGNN v2 training"
    )

    metrics = train_gnn(
        samples=samples,
        n_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=device,
        output_dir=Path(
            args.output_dir
        ),
        pretrained=args.pretrained,
        split_seed=args.seed,
    )

    print()
    print("=" * 72)
    print(
        "PackGNN v2 training complete"
    )
    print("=" * 72)

    print(
        f"SOC MAE:       "
        f"{metrics['best_soc_mae']*100:.3f}%"
    )

    print(
        f"DeltaT MAE:    "
        f"{metrics['best_dT_mae']:.4f} C"
    )

    print(
        f"Imbalance MAE: "
        f"{metrics['best_imbalance_mae']:.5f}"
    )

    print(
        f"Train rollouts:"
        f" {metrics['train_rollouts']}"
    )

    print(
        f"Val rollouts:  "
        f"{metrics['val_rollouts']}"
    )


if __name__ == "__main__":
    main()
