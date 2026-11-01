"""
Action-conditioned flat MLP v2 ablation.

Fair comparison with PackGNN v2:
  - same 5,000 rollout dataset
  - same rollout-level train/validation split
  - same explicit candidate-action input
  - same supervised targets
  - same loss weights
  - same optimizer and scheduler
  - same 50-epoch training budget
  - nearly identical parameter count
  - no graph topology or edge attributes

The MLP sees all 12 cells jointly through one flattened pack vector.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from train_gnn import (
    generate_dataset,
    split_by_rollout,
    GraphRolloutDataset,
    collate_fn,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "results" / "extended_evaluation" / "models"


class PackMLPV2(nn.Module):
    """
    Flat, action-conditioned whole-pack MLP.

    Each cell contributes:
        7 state features + 1 candidate action

    Input dimension:
        12 * 8 = 96

    Graph edges and edge attributes are deliberately ignored.
    """

    MODEL_FORMAT_VERSION = 2

    def __init__(
        self,
        n_cells: int = 12,
        node_feat: int = 7,
        action_feat: int = 1,
        hidden: int = 212,
        hidden2: int = 73,
    ):
        super().__init__()

        self.n_cells = int(n_cells)
        self.node_feat = int(node_feat)
        self.action_feat = int(action_feat)
        self.hidden = int(hidden)
        self.hidden2 = int(hidden2)

        in_dim = (
            self.n_cells
            * (
                self.node_feat
                + self.action_feat
            )
        )

        self.backbone = nn.Sequential(
            nn.Linear(
                in_dim,
                hidden,
            ),
            nn.LayerNorm(
                hidden
            ),
            nn.GELU(),

            nn.Linear(
                hidden,
                hidden,
            ),
            nn.LayerNorm(
                hidden
            ),
            nn.GELU(),

            nn.Linear(
                hidden,
                hidden2,
            ),
            nn.GELU(),
        )

        self.soc_head = nn.Sequential(
            nn.Linear(
                hidden2,
                n_cells,
            ),
            nn.Sigmoid(),
        )

        self.temp_head = nn.Linear(
            hidden2,
            n_cells,
        )

        self.aging_head = nn.Sequential(
            nn.Linear(
                hidden2,
                n_cells,
            ),
            nn.Softplus(),
        )

        self.pack_head = nn.Sequential(
            nn.Linear(
                hidden2,
                1,
            ),
            nn.Sigmoid(),
        )

    def _prepare(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:

        if action is None:
            raise ValueError(
                "PackMLPV2 requires explicit candidate action u_t."
            )

        x_was_single = (x.ndim == 2)

        if x_was_single:
            x = x.unsqueeze(0)

        if action.ndim == 1:
            action = action.unsqueeze(0).unsqueeze(-1)

        elif action.ndim == 2:
            if x_was_single and action.shape[-1] == 1:
                action = action.unsqueeze(0)
            else:
                action = action.unsqueeze(-1)

        if x.ndim != 3:
            raise ValueError(
                f"x must be (N,F) or (B,N,F); got {tuple(x.shape)}"
            )

        if action.ndim != 3:
            raise ValueError(
                f"action must be (N,), (B,N), or (B,N,1); "
                f"got {tuple(action.shape)}"
            )

        if x.shape[0] != action.shape[0]:
            raise ValueError(
                "Batch-size mismatch."
            )

        if x.shape[1] != action.shape[1]:
            raise ValueError(
                "Cell-count mismatch."
            )

        action = torch.clamp(
            action.to(
                device=x.device,
                dtype=x.dtype,
            ),
            0.0,
            1.0,
        )

        xa = torch.cat(
            [x, action],
            dim=-1,
        )

        return xa.reshape(
            xa.shape[0],
            -1,
        )

    def forward_batch(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
    ) -> dict:

        flat = self._prepare(
            x,
            action,
        )

        h = self.backbone(
            flat
        )

        return {
            "soc_pred":
                self.soc_head(h),
            "delta_T_pred":
                self.temp_head(h),
            "aging_pred":
                self.aging_head(h),
            "imbalance":
                self.pack_head(h),
        }

    def forward(
        self,
        x: torch.Tensor,
        edge_index=None,
        edge_attr=None,
        action: torch.Tensor | None = None,
    ) -> dict:

        out = self.forward_batch(
            x,
            action,
        )

        return {
            "soc_pred":
                out["soc_pred"][0],
            "delta_T_pred":
                out["delta_T_pred"][0],
            "aging_pred":
                out["aging_pred"][0],
            "imbalance":
                out["imbalance"][0].squeeze(),
        }


def count_params(
    model,
):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def train_mlp(
    samples,
    n_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    split_seed: int = 42,
    train_seed: int = 42,
    device=None,
):

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    torch.manual_seed(
        train_seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            train_seed
        )

    np.random.seed(
        train_seed
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

    model = PackMLPV2().to(
        device
    )

    n_params = count_params(
        model
    )

    print()
    print(
        f"PackMLP v2 params: {n_params:,}"
    )
    print(
        "PackGNN v2 reference: 84,868"
    )
    print(
        f"Difference: {n_params - 84868:+,} "
        f"({100.0*(n_params-84868)/84868:+.3f}%)"
    )

    print(
        f"Group split: "
        f"train={len(train_ids)} rollouts/"
        f"{len(train_samples)} samples | "
        f"val={len(val_ids)} rollouts/"
        f"{len(val_samples)} samples"
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
        f"{'Ep':>4} "
        f"{'Train':>10} "
        f"{'Val':>10} "
        f"{'SOC_MAE':>9} "
        f"{'dT_MAE':>8} "
        f"{'IMB_MAE':>8} "
        f"{'LR':>9}"
    )
    print(
        "-" * 70
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
            action = action.to(device)
            y_soc = y_soc.to(device)
            y_dT = y_dT.to(device)
            y_aging = y_aging.to(device)

            out = model.forward_batch(
                x,
                action,
            )

            imb_target = torch.std(
                y_soc,
                dim=1,
                unbiased=False,
                keepdim=True,
            )

            loss = (
                w_soc
                * mse(
                    out["soc_pred"],
                    y_soc,
                )
                +
                w_dT
                * mse(
                    out["delta_T_pred"],
                    y_dT,
                )
                +
                w_aging
                * mse(
                    out["aging_pred"],
                    y_aging,
                )
                +
                w_imbalance
                * mse(
                    out["imbalance"],
                    imb_target,
                )
            )

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            train_loss += (
                loss.item()
                * x.shape[0]
            )

        train_loss /= len(
            train_ds
        )

        scheduler.step()

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
                action = action.to(device)
                y_soc = y_soc.to(device)
                y_dT = y_dT.to(device)
                y_aging = y_aging.to(device)

                out = model.forward_batch(
                    x,
                    action,
                )

                imb_target = torch.std(
                    y_soc,
                    dim=1,
                    unbiased=False,
                    keepdim=True,
                )

                loss = (
                    w_soc
                    * mse(
                        out["soc_pred"],
                        y_soc,
                    )
                    +
                    w_dT
                    * mse(
                        out["delta_T_pred"],
                        y_dT,
                    )
                    +
                    w_aging
                    * mse(
                        out["aging_pred"],
                        y_aging,
                    )
                    +
                    w_imbalance
                    * mse(
                        out["imbalance"],
                        imb_target,
                    )
                )

                val_loss += (
                    loss.item()
                    * x.shape[0]
                )

                soc_maes.append(
                    float(
                        torch.abs(
                            out["soc_pred"]
                            - y_soc
                        ).mean()
                    )
                )

                dT_maes.append(
                    float(
                        torch.abs(
                            out["delta_T_pred"]
                            - y_dT
                        ).mean()
                    )
                )

                imb_maes.append(
                    float(
                        torch.abs(
                            out["imbalance"]
                            - imb_target
                        ).mean()
                    )
                )

        val_loss /= len(
            val_ds
        )

        soc_mae = float(
            np.mean(soc_maes)
        )
        dT_mae = float(
            np.mean(dT_maes)
        )
        imb_mae = float(
            np.mean(imb_maes)
        )

        history["train_loss"].append(
            train_loss
        )
        history["val_loss"].append(
            val_loss
        )
        history["val_soc_mae"].append(
            soc_mae
        )
        history["val_dT_mae"].append(
            dT_mae
        )
        history[
            "val_imbalance_mae"
        ].append(
            imb_mae
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
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
                f"{epoch:>4} "
                f"{train_loss:>10.6f} "
                f"{val_loss:>10.6f} "
                f"{soc_mae*100:>8.3f}% "
                f"{dT_mae:>8.4f} "
                f"{imb_mae:>8.4f} "
                f"{scheduler.get_last_lr()[0]:>9.2e}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError(
            "Training produced no checkpoint."
        )

    model.load_state_dict(
        best_state,
        strict=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ts = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    model_path = (
        MODEL_DIR
        / f"pack_mlp_action_v2_seed{train_seed}_{ts}.pt"
    )

    torch.save(
        {
            "model_format_version": 2,
            "architecture":
                "PackMLPV2",
            "action_conditioned":
                True,
            "graph_message_passing":
                False,
            "uses_edge_attributes":
                False,
            "action_definition":
                "u_i = I_i / (I_max_C * Q_nom_i)",
            "model_state":
                best_state,
            "n_cells": 12,
            "node_features": 7,
            "action_features": 1,
            "hidden": 212,
            "hidden2": 73,
            "n_params":
                n_params,
            "best_epoch":
                best_epoch,
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
            "train_seed":
                train_seed,
            "train_rollout_ids":
                sorted(train_ids),
            "val_rollout_ids":
                sorted(val_ids),
            "train_pack_seeds":
                sorted(
                    set(
                        int(s["pack_seed"])
                        for s in train_samples
                    )
                ),
            "val_pack_seeds":
                sorted(
                    set(
                        int(s["pack_seed"])
                        for s in val_samples
                    )
                ),
        },
        model_path,
    )

    print()
    print(
        f"Model saved -> {model_path}"
    )
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        f"Best val loss: {best_val_loss:.6f}"
    )
    print(
        f"Best SOC MAE: "
        f"{best_metrics['soc_mae']*100:.4f}%"
    )
    print(
        f"Best dT MAE: "
        f"{best_metrics['dT_mae']:.6f} C"
    )
    print(
        f"Best imbalance MAE: "
        f"{best_metrics['imbalance_mae']:.6f}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Generating {args.n_rollouts} "
        f"LFP rollouts, data_seed={args.data_seed}"
    )

    samples = generate_dataset(
        args.n_rollouts,
        n_cells=12,
        chemistry="LFP",
        seed=args.data_seed,
    )

    print(
        f"Total samples: {len(samples):,}"
    )

    train_mlp(
        samples=samples,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        split_seed=args.split_seed,
        train_seed=args.train_seed,
        device=device,
    )


if __name__ == "__main__":
    main()
