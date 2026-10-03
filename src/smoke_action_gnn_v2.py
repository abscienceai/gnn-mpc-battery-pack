#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

sys.path.insert(
    0,
    str(SRC),
)

from graph_battery_pack import (
    PackGNN,
    build_pack_from_ecm,
)

from train_gnn import (
    generate_dataset,
    split_by_rollout,
    train_gnn,
)

from safe_fast_charge_optimizer import (
    default_config,
    GraphGuidedOptimizer,
)

from submission_revision_controllers import (
    apply_dataset_voltage_limits,
)


def main():

    print("=" * 78)
    print(
        "ACTION-CONDITIONED PACKGNN V2 SMOKE TEST"
    )
    print("=" * 78)

    # ---------------------------------------------------------------
    # Small independent dataset.
    # ---------------------------------------------------------------
    samples = generate_dataset(
        n_rollouts=24,
        n_cells=12,
        chemistry="LFP",
        I_max_C=3.0,
        seed=123,
    )

    (
        train_samples,
        val_samples,
        train_ids,
        val_ids,
    ) = split_by_rollout(
        samples,
        val_fraction=0.15,
        seed=123,
    )

    overlap = (
        set(train_ids)
        & set(val_ids)
    )

    print(
        f"Unique train rollouts: "
        f"{len(train_ids)}"
    )

    print(
        f"Unique val rollouts:   "
        f"{len(val_ids)}"
    )

    print(
        f"Rollout overlap:       "
        f"{len(overlap)}"
    )

    if overlap:
        raise SystemExit(
            "FAIL: rollout leakage."
        )

    train_seeds = set(
        s["pack_seed"]
        for s in train_samples
    )

    val_seeds = set(
        s["pack_seed"]
        for s in val_samples
    )

    if train_seeds & val_seeds:
        raise SystemExit(
            "FAIL: pack-realisation seed leakage."
        )

    canonical_eval_seeds = {
        7 * i
        for i in range(30)
    }

    used_training_seeds = (
        train_seeds
        | val_seeds
    )

    eval_overlap = (
        used_training_seeds
        & canonical_eval_seeds
    )

    print(
        f"Canonical evaluation seed overlap: "
        f"{len(eval_overlap)}"
    )

    if eval_overlap:
        raise SystemExit(
            "FAIL: training/evaluation pack-seed overlap."
        )

    # ---------------------------------------------------------------
    # Tiny CPU training run.
    # ---------------------------------------------------------------
    out_dir = (
        ROOT
        / "results"
        / "submission_revision"
        / "action_gnn_smoke_models"
    )

    metrics = train_gnn(
        samples=samples,
        n_epochs=2,
        batch_size=16,
        lr=1e-3,
        device=torch.device("cpu"),
        output_dir=out_dir,
        split_seed=123,
    )

    ckpt_path = Path(
        metrics["model_path"]
    )

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
    )

    print()
    print(
        "Checkpoint model_format_version:",
        ckpt.get(
            "model_format_version"
        ),
    )

    print(
        "Checkpoint action_conditioned:",
        ckpt.get(
            "action_conditioned"
        ),
    )

    print(
        "Checkpoint split_method:",
        ckpt.get(
            "split_method"
        ),
    )

    if (
        ckpt.get(
            "model_format_version"
        )
        != 2
    ):
        raise SystemExit(
            "FAIL: wrong checkpoint version."
        )

    if not ckpt.get(
        "action_conditioned",
        False,
    ):
        raise SystemExit(
            "FAIL: checkpoint not action-conditioned."
        )

    if (
        ckpt.get(
            "split_method"
        )
        != "group_by_rollout_id"
    ):
        raise SystemExit(
            "FAIL: checkpoint split metadata incorrect."
        )

    # ---------------------------------------------------------------
    # Action sensitivity test:
    # exact same graph state, two different candidate actions.
    # ---------------------------------------------------------------
    model = PackGNN(
        node_feat=7,
        edge_feat=3,
        hidden=64,
        n_layers=3,
        action_feat=1,
    )

    model.load_state_dict(
        ckpt["model_state"],
        strict=True,
    )

    model.eval()

    ECM = (
        ROOT
        / "results"
        / "ecm"
        / "ecm_params_20260630_202031.parquet"
    )

    pack = build_pack_from_ecm(
        ECM,
        n_cells=12,
        chemistry="LFP",
        soc_init=0.20,
        soc_noise=0.03,
        seed=999,
    )

    g = pack.to_torch_graph()

    u_zero = torch.zeros(
        pack.n_cells,
        1,
    )

    u_full = torch.ones(
        pack.n_cells,
        1,
    )

    with torch.no_grad():

        out_zero = model(
            g["x"],
            g["edge_index"],
            g["edge_attr"],
            u_zero,
        )

        out_full = model(
            g["x"],
            g["edge_index"],
            g["edge_attr"],
            u_full,
        )

    soc_action_delta = float(
        torch.max(
            torch.abs(
                out_full["soc_pred"]
                - out_zero["soc_pred"]
            )
        )
    )

    temp_action_delta = float(
        torch.max(
            torch.abs(
                out_full["delta_T_pred"]
                - out_zero["delta_T_pred"]
            )
        )
    )

    print()
    print(
        f"Same-state action sensitivity "
        f"(SOC max delta): "
        f"{soc_action_delta:.8f}"
    )

    print(
        f"Same-state action sensitivity "
        f"(DeltaT max delta): "
        f"{temp_action_delta:.8f}"
    )

    if (
        soc_action_delta <= 1e-7
        and temp_action_delta <= 1e-7
    ):
        raise SystemExit(
            "FAIL: model output is insensitive to action."
        )

    # ---------------------------------------------------------------
    # GraphGuidedOptimizer surrogate rollout integration test.
    # ---------------------------------------------------------------
    cfg = default_config()

    cfg.update({
        "chemistry": "LFP",
        "n_cells": 12,
        "soc_init": 0.20,
        "soc_noise": 0.03,
        "target_soc": 0.80,
    })

    cfg = apply_dataset_voltage_limits(
        cfg,
        repo_root=ROOT,
    )

    controller = GraphGuidedOptimizer(
        cfg,
        model,
    )

    q = np.asarray(
        [c.Q_nom_Ah for c in pack.cells],
        dtype=np.float64,
    )

    i_max = (
        cfg["I_max_C"]
        * q
    )

    seq_zero = np.zeros(
        (
            cfg["horizon"],
            pack.n_cells,
        ),
        dtype=np.float32,
    )

    seq_full = np.tile(
        i_max[None, :],
        (
            cfg["horizon"],
            1,
        ),
    ).astype(np.float32)

    cost_zero = (
        controller
        ._gnn_rollout_cost(
            pack,
            seq_zero,
        )
    )

    cost_full = (
        controller
        ._gnn_rollout_cost(
            pack,
            seq_full,
        )
    )

    print()
    print(
        f"Surrogate rollout cost, zero action: "
        f"{cost_zero:.6f}"
    )

    print(
        f"Surrogate rollout cost, full action: "
        f"{cost_full:.6f}"
    )

    if not (
        np.isfinite(cost_zero)
        and np.isfinite(cost_full)
    ):
        raise SystemExit(
            "FAIL: non-finite surrogate cost."
        )

    if abs(
        cost_zero
        - cost_full
    ) <= 1e-8:
        raise SystemExit(
            "FAIL: surrogate rollout cost is action-invariant."
        )

    print()
    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)

    print(
        "Rollout-level split:             PASS"
    )

    print(
        "Train/validation seed isolation: PASS"
    )

    print(
        "Canonical eval seed isolation:   PASS"
    )

    print(
        "Checkpoint v2 metadata:          PASS"
    )

    print(
        "Explicit action sensitivity:     PASS"
    )

    print(
        "GraphOptimizer v2 integration:   PASS"
    )

    print()
    print(
        "ACTION-CONDITIONED PACKGNN V2 SMOKE TEST: PASS"
    )


if __name__ == "__main__":
    main()
