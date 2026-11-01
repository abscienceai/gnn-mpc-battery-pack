#!/usr/bin/env python3

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

import run_surrogate_ablation_n30 as base

from safe_fast_charge_optimizer import (
    SimpleMPCController,
    compute_cost,
    project_current_vector,
)


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = (
    ROOT
    / "results/extended_evaluation"
    / "physics_shared_objective"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


class PhysicsCEMSharedObjective(SimpleMPCController):
    """
    Direct-physics CEM with the same horizon-stage observability as PackGNN.

    Physics is still used to propagate candidate trajectories, but terminal
    voltage is deliberately excluded from the CEM horizon violation penalty
    because PackGNN does not predict voltage.

    Horizon violation term:
        count(SOC >= 0.98) + count(T >= T_max)

    The inherited final one-step physics safety filter remains unchanged.
    """

    def _physics_rollout_cost(
        self,
        pack,
        current_vector,
    ):
        virtual_pack = deepcopy(pack)

        total_cost = 0.0
        physics_steps = 0

        for h in range(self.horizon):

            currents = project_current_vector(
                current_vector,
                virtual_pack,
                self.cfg,
            )

            metrics = virtual_pack.step(
                currents,
                dt=self.cfg["dt_s"],
            )

            physics_steps += 1

            done = (
                metrics["SOC_mean"]
                >= self.cfg["target_soc"]
            )

            # Match PackGNN horizon-stage violation observability exactly.
            shared_n_viol = int(
                np.sum(
                    np.asarray(
                        metrics["SOC"],
                        dtype=float,
                    )
                    >= 0.98
                )
                +
                np.sum(
                    np.asarray(
                        metrics["T_C"],
                        dtype=float,
                    )
                    >= self.cfg["T_max"]
                )
            )

            shared_metrics = dict(metrics)
            shared_metrics["n_violations"] = shared_n_viol

            total_cost += compute_cost(
                shared_metrics,
                self.cfg,
                h,
                done,
            )

            if done:
                break

        return float(total_cost), physics_steps


def make_controller(model_name):

    if model_name == "PackGNN":
        return base.AblationOptimizer(
            base.cfg,
            base.MODELS["PackGNN"],
        )

    if model_name == "Physics-CEM-Shared":
        return PhysicsCEMSharedObjective(
            base.cfg
        )

    raise ValueError(model_name)


# Reuse the fixed canonical episode implementation.
base.make_controller = make_controller


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if not 1 <= args.n_seeds <= 30:
        raise ValueError("--n-seeds must be 1..30")

    seeds = base.CANONICAL_SEEDS[:args.n_seeds]

    out = (
        OUT_DIR
        / f"physics_shared_objective_n{args.n_seeds}.jsonl"
    )

    if out.exists():
        rows = [
            json.loads(x)
            for x in out.read_text().splitlines()
            if x.strip()
        ]
    else:
        rows = []

    completed = {
        (int(r["seed"]), r["model"])
        for r in rows
    }

    print("Shared horizon objective:")
    print("  Physics propagation retained")
    print("  Voltage excluded from horizon violation penalty")
    print("  SOC >= 0.98 and T >= T_max retained")
    print("  Final one-step physics safety filter retained")
    print()

    for i, seed in enumerate(seeds, 1):

        pack0 = base.build_pack_from_ecm(
            ecm_parquet=base.ECM,
            n_cells=base.cfg["n_cells"],
            chemistry=base.cfg["chemistry"],
            T_amb=base.cfg["T_amb"],
            soc_init=base.cfg["soc_init"],
            soc_noise=base.cfg["soc_noise"],
            seed=seed,
            strict=True,
        )

        for cell in pack0.cells:
            cell.V_min_limit = float(base.cfg["V_min"])
            cell.V_max_limit = float(base.cfg["V_max"])

        print(
            f"Seed {i:02d}/{len(seeds)} | seed={seed}",
            flush=True,
        )

        for model_name in [
            "Physics-CEM-Shared",
        ]:

            key = (int(seed), model_name)

            if key in completed:
                print(
                    f"  SKIP {model_name}",
                    flush=True,
                )
                continue

            row = base.run_episode(
                deepcopy(pack0),
                model_name,
                seed,
            )

            append_jsonl(out, row)
            rows.append(row)
            completed.add(key)

            print(
                f"  {model_name:<19} "
                f"time={row['charging_time_min']:.2f} "
                f"SOCsig={row['final_SOC_sigma_pct']:.4f}% "
                f"dT={row['final_T_gradient_C']:.4f} "
                f"peakT={row['peak_T_C']:.4f} "
                f"Vmax={row['peak_voltage_V']:.4f} "
                f"viol={row['total_violations']} "
                f"target={row['target_reached']}",
                flush=True,
            )

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    for model_name in [
        "Physics-CEM-Shared",
    ]:

        rr = [
            r for r in rows
            if r["model"] == model_name
        ]

        def mean(k):
            return float(
                np.mean([r[k] for r in rr])
            )

        success = sum(
            bool(r["target_reached"])
            for r in rr
        )

        print(
            f"{model_name:<19} "
            f"N={len(rr):2d} "
            f"success={success}/{len(rr)} "
            f"time={mean('charging_time_min'):.3f} "
            f"SOCsig={mean('final_SOC_sigma_pct'):.4f}% "
            f"dT={mean('final_T_gradient_C'):.4f} "
            f"peakT={mean('peak_T_C'):.4f} "
            f"Vmax={mean('peak_voltage_V'):.4f} "
            f"viol={sum(r['total_violations'] for r in rr)}"
        )

    print()
    print(f"Results -> {out}")


if __name__ == "__main__":
    main()
