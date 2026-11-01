#!/usr/bin/env python3

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import run_surrogate_ablation_n30 as base


ROOT = Path(__file__).resolve().parent.parent

MODEL_ROOT = (
    ROOT
    / "results/extended_evaluation/models"
)

OUT_DIR = (
    ROOT
    / "results/extended_evaluation"
    / "training_seed_stability"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_SEEDS = [
    101,
    202,
]

MODEL_NAMES = [
    "PackGNN",
    "NodeOnly",
    "FlatMLP",
]


def latest(pattern, directory=MODEL_ROOT):
    paths = sorted(
        directory.glob(pattern)
    )

    if not paths:
        raise FileNotFoundError(
            f"No checkpoint: {directory}/{pattern}"
        )

    return paths[-1]


def checkpoint_paths(train_seed):

    return {
        "PackGNN":
            latest(
                "pack_gnn_action_v2_*.pt",
                MODEL_ROOT
                / f"packgnn_seed{train_seed}",
            ),

        "NodeOnly":
            latest(
                f"node_only_action_v2_seed{train_seed}_*.pt"
            ),

        "FlatMLP":
            latest(
                f"pack_mlp_action_v2_seed{train_seed}_*.pt"
            ),
    }


def load_models(train_seed):

    paths = checkpoint_paths(
        train_seed
    )

    ckpts = {
        name:
            torch.load(
                path,
                map_location="cpu",
            )
        for name, path
        in paths.items()
    }

    g = ckpts[
        "PackGNN"
    ]

    graph = base.PackGNN(
        node_feat=
            g["node_features"],
        edge_feat=
            g["edge_features"],
        hidden=
            g["hidden"],
        n_layers=
            g["n_layers"],
        action_feat=
            g["action_features"],
    ).to(
        base.device
    )

    graph.load_state_dict(
        g["model_state"],
        strict=True,
    )

    graph.eval()


    n = ckpts[
        "NodeOnly"
    ]

    node = base.NodeOnlyGNN(
        node_feat=
            n.get(
                "node_features",
                7,
            ),
        hidden=
            n.get(
                "hidden",
                64,
            ),
        n_layers=
            n.get(
                "n_layers",
                3,
            ),
        action_feat=
            n.get(
                "action_features",
                1,
            ),
        expansion=
            n.get(
                "expansion",
                192,
            ),
    ).to(
        base.device
    )

    node.load_state_dict(
        n["model_state"],
        strict=True,
    )

    node.eval()


    m = ckpts[
        "FlatMLP"
    ]

    mlp = base.PackMLPV2(
        n_cells=
            m.get(
                "n_cells",
                12,
            ),
        node_feat=
            m.get(
                "node_features",
                7,
            ),
        action_feat=
            m.get(
                "action_features",
                1,
            ),
        hidden=
            m.get(
                "hidden",
                212,
            ),
        hidden2=
            m.get(
                "hidden2",
                73,
            ),
    ).to(
        base.device
    )

    mlp.load_state_dict(
        m["model_state"],
        strict=True,
    )

    mlp.eval()

    models = {
        "PackGNN":
            graph,
        "NodeOnly":
            node,
        "FlatMLP":
            mlp,
    }

    reference = ckpts[
        "PackGNN"
    ]

    for name, c in ckpts.items():

        if (
            c["train_rollout_ids"]
            != reference[
                "train_rollout_ids"
            ]
        ):
            raise RuntimeError(
                f"{name}: train IDs differ"
            )

        if (
            c["val_rollout_ids"]
            != reference[
                "val_rollout_ids"
            ]
        ):
            raise RuntimeError(
                f"{name}: val IDs differ"
            )

        overlap = (
            set(
                base.CANONICAL_SEEDS
            )
            & (
                set(
                    c.get(
                        "train_pack_seeds",
                        [],
                    )
                )
                |
                set(
                    c.get(
                        "val_pack_seeds",
                        [],
                    )
                )
            )
        )

        if overlap:
            raise RuntimeError(
                f"{name}: evaluation leakage "
                f"{sorted(overlap)}"
            )

    return (
        models,
        ckpts,
        paths,
    )


def append_jsonl(
    path,
    row,
):

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                row,
                sort_keys=True,
            )
            + "\n"
        )


def read_jsonl(path):

    if not path.exists():
        return []

    return [
        json.loads(line)
        for line
        in path.read_text().splitlines()
        if line.strip()
    ]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if not (
        1
        <= args.n_seeds
        <= 30
    ):
        raise ValueError(
            "--n-seeds must be 1..30"
        )

    eval_seeds = (
        base.CANONICAL_SEEDS[
            :args.n_seeds
        ]
    )

    out = (
        OUT_DIR
        / f"training_seed_stability_n{args.n_seeds}.jsonl"
    )

    rows = read_jsonl(
        out
    )

    completed = {
        (
            int(r["train_seed"]),
            int(r["seed"]),
            r["model"],
        )
        for r in rows
    }

    original_models = (
        base.MODELS
    )

    try:

        for train_seed in TRAIN_SEEDS:

            (
                models,
                ckpts,
                paths,
            ) = load_models(
                train_seed
            )

            base.MODELS = (
                models
            )

            print()
            print(
                "=" * 86
            )

            print(
                f"TRAINING SEED {train_seed}"
            )

            print(
                "=" * 86
            )

            for name in MODEL_NAMES:

                bm = ckpts[
                    name
                ][
                    "best_metrics"
                ]

                print(
                    f"{name:<9} "
                    f"SOC="
                    f"{float(bm['soc_mae'])*100:.4f}% "
                    f"dT="
                    f"{float(bm['dT_mae']):.6f} C | "
                    f"{paths[name].name}"
                )

            print()

            for idx, seed in enumerate(
                eval_seeds,
                1,
            ):

                base_pack = (
                    base.build_pack_from_ecm(
                        ecm_parquet=
                            base.ECM,
                        n_cells=
                            base.cfg[
                                "n_cells"
                            ],
                        chemistry=
                            base.cfg[
                                "chemistry"
                            ],
                        T_amb=
                            base.cfg[
                                "T_amb"
                            ],
                        soc_init=
                            base.cfg[
                                "soc_init"
                            ],
                        soc_noise=
                            base.cfg[
                                "soc_noise"
                            ],
                        seed=seed,
                        strict=True,
                    )
                )

                for cell in base_pack.cells:

                    cell.V_min_limit = float(
                        base.cfg[
                            "V_min"
                        ]
                    )

                    cell.V_max_limit = float(
                        base.cfg[
                            "V_max"
                        ]
                    )

                print(
                    f"Seed {idx:02d}/{len(eval_seeds)} "
                    f"| pack_seed={seed}",
                    flush=True,
                )

                for model_name in MODEL_NAMES:

                    key = (
                        train_seed,
                        seed,
                        model_name,
                    )

                    if key in completed:

                        print(
                            f"  SKIP {model_name:<9}",
                            flush=True,
                        )

                        continue

                    row = base.run_episode(
                        deepcopy(
                            base_pack
                        ),
                        model_name,
                        seed,
                    )

                    row[
                        "train_seed"
                    ] = int(
                        train_seed
                    )

                    append_jsonl(
                        out,
                        row,
                    )

                    rows.append(
                        row
                    )

                    completed.add(
                        key
                    )

                    print(
                        f"  {model_name:<9} "
                        f"time="
                        f"{row['charging_time_min']:.2f} "
                        f"SOCsig="
                        f"{row['final_SOC_sigma_pct']:.4f}% "
                        f"dT="
                        f"{row['final_T_gradient_C']:.4f} "
                        f"peakT="
                        f"{row['peak_T_C']:.4f} "
                        f"viol="
                        f"{row['total_violations']}",
                        flush=True,
                    )

    finally:

        base.MODELS = (
            original_models
        )

    print()
    print(
        "=" * 86
    )
    print(
        "SUMMARY"
    )
    print(
        "=" * 86
    )

    for train_seed in TRAIN_SEEDS:

        for model_name in MODEL_NAMES:

            rr = [
                r
                for r in rows
                if (
                    int(
                        r["train_seed"]
                    )
                    == train_seed
                    and
                    r["model"]
                    == model_name
                )
            ]

            if not rr:
                continue

            def mean(k):
                return float(
                    np.mean(
                        [
                            r[k]
                            for r in rr
                        ]
                    )
                )

            print(
                f"train_seed={train_seed:3d} "
                f"{model_name:<9} "
                f"N={len(rr):2d} "
                f"time={mean('charging_time_min'):.4f} "
                f"SOCsig={mean('final_SOC_sigma_pct'):.5f}% "
                f"dT={mean('final_T_gradient_C'):.5f} "
                f"peakT={mean('peak_T_C'):.5f} "
                f"viol="
                f"{sum(r['total_violations'] for r in rr)}"
            )

    print()
    print(
        f"Results -> {out}"
    )


if __name__ == "__main__":
    main()
