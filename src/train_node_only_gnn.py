"""
Action-conditioned NodeOnly v2 ablation.

Scientific control for PackGNN v2:
  - identical 5,000-rollout dataset generator
  - identical rollout-group train/validation split
  - identical explicit candidate-action conditioning
  - identical supervised outputs and loss weights
  - identical optimizer and LR schedule
  - identical prediction heads
  - comparable parameter count
  - no edge use and no inter-cell message passing

This isolates the contribution of graph message passing.
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


class NodeOnlyLayer(nn.Module):
    """
    Per-node residual MLP.

    No edge_index or edge_attr is used.
    Each cell is processed independently.
    """

    def __init__(
        self,
        node_dim: int = 64,
        expansion: int = 192,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(node_dim, expansion),
            nn.GELU(),
            nn.Linear(expansion, node_dim),
        )

        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.norm(
            x + self.mlp(x)
        )


class NodeOnlyGNN(nn.Module):
    """
    Action-conditioned non-graph ablation for PackGNN v2.

    Input:
        x_t : per-cell state features
        u_t : per-cell candidate action

    Output interface is identical to PackGNN v2.

    edge_index and edge_attr are accepted only for drop-in API compatibility
    and are deliberately ignored.
    """

    MODEL_FORMAT_VERSION = 2

    def __init__(
        self,
        node_feat: int = 7,
        hidden: int = 64,
        n_layers: int = 3,
        action_feat: int = 1,
        expansion: int = 192,
    ):
        super().__init__()

        self.node_feat = int(node_feat)
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.action_feat = int(action_feat)
        self.expansion = int(expansion)

        self.input_proj = nn.Linear(
            self.node_feat + self.action_feat,
            hidden,
        )

        self.layers = nn.ModuleList([
            NodeOnlyLayer(
                node_dim=hidden,
                expansion=expansion,
            )
            for _ in range(n_layers)
        ])

        self.soc_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.temp_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self.aging_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        self.pack_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        edge_attr: torch.Tensor | None,
        action: torch.Tensor,
    ) -> dict:

        if action is None:
            raise ValueError(
                "NodeOnlyGNN v2 requires explicit candidate action u_t."
            )

        if action.ndim == 1:
            action = action.unsqueeze(-1)

        if action.ndim != 2:
            raise ValueError(
                f"action must have shape (N,) or (N,1); "
                f"got {tuple(action.shape)}"
            )

        if action.shape[0] != x.shape[0]:
            raise ValueError(
                f"Node/action count mismatch: "
                f"x={tuple(x.shape)}, action={tuple(action.shape)}"
            )

        if action.shape[1] != self.action_feat:
            raise ValueError(
                f"Expected action feature width {self.action_feat}; "
                f"got {action.shape[1]}"
            )

        action = torch.clamp(
            action.to(
                device=x.device,
                dtype=x.dtype,
            ),
            0.0,
            1.0,
        )

        h = self.input_proj(
            torch.cat(
                [x, action],
                dim=-1,
            )
        )

        for layer in self.layers:
            h = layer(h)

        return {
            "soc_pred":
                self.soc_head(h),
            "delta_T_pred":
                self.temp_head(h),
            "aging_pred":
                self.aging_head(h),
            "imbalance":
                self.pack_head(
                    h.mean(
                        0,
                        keepdim=True,
                    )
                ),
        }


def count_params(
    model,
):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def train_node_only(
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

    model = NodeOnlyGNN(
        node_feat=7,
        hidden=64,
        n_layers=3,
        action_feat=1,
        expansion=192,
    ).to(device)

    n_params = count_params(
        model
    )

    print()
    print(
        f"NodeOnlyGNN v2 params: {n_params:,}"
    )
    print(
        "PackGNN v2 reference: 84,868"
    )
    print(
        f"Difference: {n_params - 84868:+,} "
        f"({100.0 * (n_params - 84868) / 84868:+.2f}%)"
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

                imb_pred = (
                    out["imbalance"]
                    .squeeze()
                )

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

                    imb_pred = (
                    out["imbalance"]
                    .squeeze()
                )

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
                f"{scheduler.get_last_lr()[0]:>9.2e}"
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
        / f"node_only_action_v2_seed{train_seed}_{ts}.pt"
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

    torch.save(
        {
            "model_format_version": 2,
            "architecture":
                "NodeOnlyGNN_v2",
            "action_conditioned": True,
            "graph_message_passing":
                False,
            "action_definition":
                "u_i = I_i / (I_max_C * Q_nom_i)",
            "model_state":
                best_state,
            "node_features": 7,
            "action_features": 1,
            "hidden": 64,
            "n_layers": 3,
            "expansion": 192,
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
                train_pack_seeds,
            "val_pack_seeds":
                val_pack_seeds,
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

    return model_path


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

    train_node_only(
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
