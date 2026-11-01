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
    / "thermal_pareto"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODELS = [
    "PackGNN",
    "NodeOnly",
]

WEIGHTS = [
    4.0,
    2.0,
    1.0,
    0.5,
]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path):
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
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

    if not 1 <= args.n_seeds <= 30:
        raise ValueError(
            "--n-seeds must be between 1 and 30"
        )

    seeds = base.CANONICAL_SEEDS[
        :args.n_seeds
    ]

    out = (
        OUT_DIR
        / f"time_pareto_n{args.n_seeds}.jsonl"
    )

    rows = read_jsonl(out)

    completed = {
        (
            int(r["seed"]),
            r["model"],
            float(r["w_time"]),
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
        f"Time weights: {WEIGHTS}"
    )

    print()

    original_time = float(
        base.cfg["w_time"]
    )

    try:

        for w_time in WEIGHTS:

            base.cfg["w_time"] = float(
                w_time
            )

            print(
                "=" * 78
            )

            print(
                f"w_time = {w_time:g}"
            )

            print(
                "=" * 78
            )

            for idx, seed in enumerate(
                seeds,
                1,
            ):

                base_pack = (
                    base.build_pack_from_ecm(
                        ecm_parquet=base.ECM,
                        n_cells=base.cfg["n_cells"],
                        chemistry=base.cfg["chemistry"],
                        T_amb=base.cfg["T_amb"],
                        soc_init=base.cfg["soc_init"],
                        soc_noise=base.cfg["soc_noise"],
                        seed=seed,
                        strict=True,
                    )
                )

                for cell in base_pack.cells:
                    cell.V_min_limit = float(
                        base.cfg["V_min"]
                    )
                    cell.V_max_limit = float(
                        base.cfg["V_max"]
                    )

                print(
                    f"Seed {idx:02d}/{len(seeds)} "
                    f"| seed={seed}",
                    flush=True,
                )

                for model_name in MODELS:

                    key = (
                        seed,
                        model_name,
                        float(w_time),
                    )

                    if key in completed:
                        print(
                            f"  SKIP {model_name:<8}",
                            flush=True,
                        )
                        continue

                    pack = deepcopy(
                        base_pack
                    )

                    print(
                        f"  RUN  {model_name:<8}",
                        end=" ",
                        flush=True,
                    )

                    row = base.run_episode(
                        pack,
                        model_name,
                        seed,
                    )

                    row[
                        "w_time"
                    ] = float(
                        w_time
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
                        f"time={row['charging_time_min']:.3f} | "
                        f"dT={row['final_T_gradient_C']:.4f} | "
                        f"SOCsig={row['final_SOC_sigma_pct']:.4f}% | "
                        f"peakT={row['peak_T_C']:.4f}",
                        flush=True,
                    )

    finally:

        base.cfg[
            "w_time"
        ] = original_time

    print()
    print(
        "SUMMARY"
    )

    print(
        "=" * 78
    )

    for w_time in WEIGHTS:

        for model_name in MODELS:

            rr = [
                r for r in rows
                if (
                    r["model"] == model_name
                    and float(
                        r["w_time"]
                    ) == float(
                        w_time
                    )
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

            print(
                f"w={w_time:>4g} "
                f"{model_name:<8} "
                f"N={len(rr):2d} "
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
