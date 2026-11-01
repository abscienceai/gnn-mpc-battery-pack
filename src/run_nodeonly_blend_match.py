#!/usr/bin/env python3

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

import run_surrogate_ablation_n30 as base


ROOT = Path(__file__).resolve().parent.parent

OUT_DIR = (
    ROOT
    / "results/extended_evaluation"
    / "blend_match"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Original controller = 0.60 CEM + 0.40 SOC-deficit.
# Increase CEM share to reduce the fixed aggressive deficit component.
CEM_WEIGHTS = [
    0.90,
]


class BlendOptimizer(
    base.AblationOptimizer
):

    def __init__(
        self,
        cfg,
        gnn,
        cem_weight,
    ):
        super().__init__(
            cfg,
            gnn,
        )

        self.cem_weight = float(
            cem_weight
        )

    def get_currents(
        self,
        pack,
    ):

        I_base = self._cem_optimise(
            pack
        )

        g = pack.to_torch_graph()

        with base.torch.no_grad():

            action_base = (
                self._action_tensor(
                    pack,
                    I_base,
                )
            )

            out = self.gnn(
                g["x"].to(
                    self.device
                ),
                g["edge_index"].to(
                    self.device
                ),
                g["edge_attr"].to(
                    self.device
                ),
                action_base,
            )

        soc_pred = (
            out["soc_pred"]
            .cpu()
            .numpy()
            .flatten()
        )

        target = self.cfg[
            "target_soc"
        ]

        deficit = np.maximum(
            target - soc_pred,
            0.0,
        )

        Q_nom = np.array(
            [
                c.Q_nom_Ah
                for c in pack.cells
            ]
        )

        I_max = (
            self.cfg["I_max_C"]
            * Q_nom
        )

        if (
            deficit.sum()
            > 1e-6
        ):

            I_deficit = (
                I_max
                * deficit
                / (
                    deficit.max()
                    + 1e-9
                )
            )

            I_final = (
                self.cem_weight
                * I_base
                +
                (
                    1.0
                    - self.cem_weight
                )
                * I_deficit
            )

        else:

            I_final = I_base

        I_final = (
            base.project_current_vector(
                I_final,
                pack,
                self.cfg,
            )
        )

        return (
            self._physics_safety_filter(
                pack,
                I_final,
            )
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


def read_jsonl(
    path,
):

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
        default=5,
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

    seeds = (
        base.CANONICAL_SEEDS[
            :args.n_seeds
        ]
    )

    out = (
        OUT_DIR
        / f"nodeonly_blend_match_n{args.n_seeds}.jsonl"
    )

    rows = read_jsonl(
        out
    )

    completed = {
        (
            int(r["seed"]),
            float(r["cem_weight"]),
        )
        for r in rows
    }

    print(
        f"Device: {base.device}"
    )

    print(
        f"Seeds: {seeds}"
    )

    print(
        f"CEM weights: {CEM_WEIGHTS}"
    )

    print()

    original_factory = (
        base.make_controller
    )

    try:

        for cem_weight in CEM_WEIGHTS:

            print(
                "=" * 78
            )

            print(
                f"CEM weight = {cem_weight:.2f} "
                f"| deficit weight = "
                f"{1.0-cem_weight:.2f}"
            )

            print(
                "=" * 78
            )

            def factory(
                model_name,
                cw=cem_weight,
            ):

                return BlendOptimizer(
                    base.cfg,
                    base.MODELS[
                        model_name
                    ],
                    cw,
                )

            base.make_controller = (
                factory
            )

            for idx, seed in enumerate(
                seeds,
                1,
            ):

                key = (
                    seed,
                    float(
                        cem_weight
                    ),
                )

                if key in completed:

                    print(
                        f"Seed {idx:02d} "
                        f"seed={seed}: SKIP",
                        flush=True,
                    )

                    continue

                pack = (
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

                for cell in pack.cells:

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
                    f"Seed {idx:02d}/{len(seeds)} "
                    f"| seed={seed} ",
                    end="",
                    flush=True,
                )

                row = base.run_episode(
                    deepcopy(pack),
                    "NodeOnly",
                    seed,
                )

                row[
                    "cem_weight"
                ] = float(
                    cem_weight
                )

                row[
                    "deficit_weight"
                ] = float(
                    1.0
                    - cem_weight
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
                    f"time="
                    f"{row['charging_time_min']:.3f} | "
                    f"dT="
                    f"{row['final_T_gradient_C']:.4f} | "
                    f"SOCsig="
                    f"{row['final_SOC_sigma_pct']:.4f}% | "
                    f"peakT="
                    f"{row['peak_T_C']:.4f} | "
                    f"target="
                    f"{row['target_reached']}",
                    flush=True,
                )

    finally:

        base.make_controller = (
            original_factory
        )

    print()
    print(
        "SUMMARY"
    )

    print(
        "=" * 78
    )

    for cem_weight in CEM_WEIGHTS:

        rr = [
            r
            for r in rows
            if float(
                r[
                    "cem_weight"
                ]
            )
            == float(
                cem_weight
            )
        ]

        if not rr:
            continue

        def mean(key):

            return float(
                np.mean(
                    [
                        r[key]
                        for r in rr
                    ]
                )
            )

        success = sum(
            bool(
                r[
                    "target_reached"
                ]
            )
            for r in rr
        )

        print(
            f"CEM={cem_weight:.2f} "
            f"deficit={1.0-cem_weight:.2f} "
            f"N={len(rr):2d} "
            f"success={success}/{len(rr)} "
            f"time={mean('charging_time_min'):.4f} "
            f"dT={mean('final_T_gradient_C'):.5f} "
            f"SOCsig={mean('final_SOC_sigma_pct'):.5f}% "
            f"peakT={mean('peak_T_C'):.5f}"
        )

    print()
    print(
        f"Results -> {out}"
    )


if __name__ == "__main__":
    main()
