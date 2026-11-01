#!/usr/bin/env python3

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import run_surrogate_ablation_n30 as base

from safe_fast_charge_optimizer import (
    PhysicsCEMController,
)


ROOT = Path(__file__).resolve().parent.parent

OUT_DIR = (
    ROOT
    / "results/extended_evaluation"
    / "model_mismatch"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SCENARIOS = {
    "combined_adverse": {
        "R0_factor": 1.20,
        "Q_factor": 0.90,
        "hA_factor": 0.80,
    },
}


CONTROLLERS = [
    "PackGNN",
    "Physics-CEM",
]


def perturb_plant(
    pack,
    scenario,
):

    spec = SCENARIOS[
        scenario
    ]

    R0_factor = float(
        spec["R0_factor"]
    )

    Q_factor = float(
        spec["Q_factor"]
    )

    hA_factor = float(
        spec["hA_factor"]
    )

    for cell, ecm, therm in zip(
        pack.cells,
        pack.ecm_models,
        pack.thermal_models,
    ):

        # Actual plant ohmic resistance.
        cell.R0 *= R0_factor
        ecm.R0 *= R0_factor

        # Actual plant capacity.
        cell.Q_nom_Ah *= Q_factor
        ecm.Q_nom *= Q_factor

        # Actual plant convective heat rejection.
        therm.hA *= hA_factor


def sync_observed_state(
    controller_pack,
    plant_pack,
):
    """
    Output-feedback update.

    Dynamic/measurable states are transferred from the actual plant to
    the controller representation, while controller model parameters
    R0, R1, C1, Q_nom and thermal hA remain nominal.
    """

    for c_nom, c_real in zip(
        controller_pack.cells,
        plant_pack.cells,
    ):

        c_nom.SOC = float(
            c_real.SOC
        )

        c_nom.T_C = float(
            c_real.T_C
        )

        c_nom.V_term = float(
            c_real.V_term
        )

        c_nom.V_oc = float(
            c_real.V_oc
        )

        c_nom.I_A = float(
            c_real.I_A
        )


def make_controller(
    name,
):

    if name == "PackGNN":

        return base.AblationOptimizer(
            base.cfg,
            base.MODELS[
                "PackGNN"
            ],
        )

    if name == "Physics-CEM":

        return PhysicsCEMController(
            base.cfg
        )

    raise ValueError(
        name
    )


def run_episode(
    nominal_pack,
    scenario,
    controller_name,
    seed,
):

    # Controller always retains nominal parameters.
    controller_pack = deepcopy(
        nominal_pack
    )

    # Actual plant starts from the same initial dynamic state,
    # then receives the requested parameter mismatch.
    plant_pack = deepcopy(
        nominal_pack
    )

    perturb_plant(
        plant_pack,
        scenario,
    )

    control_seed = (
        10_000_000
        + int(seed)
    )

    np.random.seed(
        control_seed
    )

    torch.manual_seed(
        control_seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            control_seed
        )

    controller = make_controller(
        controller_name
    )

    if hasattr(
        controller,
        "reset",
    ):
        controller.reset()

    initial_soc = np.asarray(
        [
            c.SOC
            for c in plant_pack.cells
        ],
        dtype=float,
    )

    initial_T = np.asarray(
        [
            c.T_C
            for c in plant_pack.cells
        ],
        dtype=float,
    )

    peak_T = float(
        np.max(
            initial_T
        )
    )

    max_T_gradient = float(
        np.max(initial_T)
        - np.min(initial_T)
    )

    peak_voltage = -np.inf
    min_voltage = np.inf

    cumulative_aging = 0.0
    pack_energy_Wh = 0.0

    violations = 0
    steps = 0

    t0 = time.time()

    for _ in range(
        base.cfg[
            "max_steps"
        ]
    ):

        actual_soc = np.asarray(
            [
                c.SOC
                for c in plant_pack.cells
            ],
            dtype=float,
        )

        if (
            float(
                np.mean(
                    actual_soc
                )
            )
            >= base.cfg[
                "target_soc"
            ]
        ):
            break

        # Controller decision is based only on the nominal internal model
        # plus the latest observed dynamic states.
        currents = np.asarray(
            controller.get_currents(
                controller_pack
            ),
            dtype=float,
        )

        # Apply exactly the commanded current to the mismatched plant.
        actual_metrics = (
            plant_pack.step(
                currents,
                dt=base.cfg[
                    "dt_s"
                ],
            )
        )

        # Advance nominal controller dynamics using the same command.
        # This preserves nominal ECM internal states such as V1.
        controller_pack.step(
            currents,
            dt=base.cfg[
                "dt_s"
            ],
        )

        # Then assimilate the newly observed plant state without revealing
        # the perturbed physical parameters to the controller.
        sync_observed_state(
            controller_pack,
            plant_pack,
        )

        steps += 1

        V = np.asarray(
            actual_metrics[
                "V_term"
            ],
            dtype=float,
        )

        T = np.asarray(
            actual_metrics[
                "T_C"
            ],
            dtype=float,
        )

        peak_voltage = max(
            peak_voltage,
            float(
                np.max(V)
            ),
        )

        min_voltage = min(
            min_voltage,
            float(
                np.min(V)
            ),
        )

        peak_T = max(
            peak_T,
            float(
                np.max(T)
            ),
        )

        max_T_gradient = max(
            max_T_gradient,
            float(
                np.max(T)
                - np.min(T)
            ),
        )

        violations += int(
            actual_metrics[
                "n_violations"
            ]
        )

        cumulative_aging += float(
            actual_metrics[
                "aging_cost"
            ]
        )

        dt_h = (
            base.cfg[
                "dt_s"
            ]
            / 3600.0
        )

        pack_energy_Wh += float(
            np.sum(
                currents
                * V
                * dt_h
            )
        )

    wall = (
        time.time()
        - t0
    )

    final_soc = np.asarray(
        [
            c.SOC
            for c in plant_pack.cells
        ],
        dtype=float,
    )

    final_T = np.asarray(
        [
            c.T_C
            for c in plant_pack.cells
        ],
        dtype=float,
    )

    target_reached = bool(
        float(
            np.mean(
                final_soc
            )
        )
        >= base.cfg[
            "target_soc"
        ]
    )

    return {
        "seed":
            int(seed),

        "scenario":
            scenario,

        "controller":
            controller_name,

        "control_seed":
            int(control_seed),

        "R0_factor":
            float(
                SCENARIOS[
                    scenario
                ][
                    "R0_factor"
                ]
            ),

        "Q_factor":
            float(
                SCENARIOS[
                    scenario
                ][
                    "Q_factor"
                ]
            ),

        "hA_factor":
            float(
                SCENARIOS[
                    scenario
                ][
                    "hA_factor"
                ]
            ),

        "steps":
            int(steps),

        "charging_time_min":
            float(
                steps
                * base.cfg[
                    "dt_s"
                ]
                / 60.0
            ),

        "target_reached":
            target_reached,

        "initial_SOC_sigma_pct":
            float(
                np.std(
                    initial_soc
                )
                * 100.0
            ),

        "final_SOC_mean":
            float(
                np.mean(
                    final_soc
                )
            ),

        "final_SOC_sigma_pct":
            float(
                np.std(
                    final_soc
                )
                * 100.0
            ),

        "final_T_gradient_C":
            float(
                np.max(
                    final_T
                )
                - np.min(
                    final_T
                )
            ),

        "peak_T_C":
            float(
                peak_T
            ),

        "max_T_gradient_C":
            float(
                max_T_gradient
            ),

        "peak_voltage_V":
            float(
                peak_voltage
            ),

        "min_voltage_V":
            float(
                min_voltage
            ),

        "total_violations":
            int(
                violations
            ),

        "cumulative_aging":
            float(
                cumulative_aging
            ),

        "pack_energy_Wh":
            float(
                pack_energy_Wh
            ),

        "wall_time_s":
            float(
                wall
            ),
    }


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
        / f"model_mismatch_n{args.n_seeds}.jsonl"
    )

    rows = read_jsonl(
        out
    )

    completed = {
        (
            int(r["seed"]),
            r["scenario"],
            r["controller"],
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
        "Controller model remains nominal."
    )

    print(
        "Dynamic SOC/T/V states are fed back after each actual plant step."
    )

    print()

    for scenario in SCENARIOS:

        print(
            "=" * 90
        )

        print(
            f"SCENARIO: {scenario} | "
            f"R0 x{SCENARIOS[scenario]['R0_factor']:.2f} | "
            f"Q x{SCENARIOS[scenario]['Q_factor']:.2f} | "
            f"hA x{SCENARIOS[scenario]['hA_factor']:.2f}"
        )

        print(
            "=" * 90
        )

        for idx, seed in enumerate(
            seeds,
            1,
        ):

            nominal_pack = (
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

            for cell in nominal_pack.cells:

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
                f"| seed={seed}",
                flush=True,
            )

            for controller_name in CONTROLLERS:

                key = (
                    seed,
                    scenario,
                    controller_name,
                )

                if key in completed:

                    print(
                        f"  SKIP {controller_name:<11}",
                        flush=True,
                    )

                    continue

                row = run_episode(
                    nominal_pack,
                    scenario,
                    controller_name,
                    seed,
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
                    f"  {controller_name:<11} "
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
    print(
        "=" * 90
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 90
    )

    for scenario in SCENARIOS:

        for controller_name in CONTROLLERS:

            rr = [
                r
                for r in rows
                if (
                    r["scenario"]
                    == scenario
                    and
                    r["controller"]
                    == controller_name
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

            success = sum(
                bool(
                    r[
                        "target_reached"
                    ]
                )
                for r in rr
            )

            print(
                f"{scenario:<18} "
                f"{controller_name:<11} "
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
    print(
        f"Results -> {out}"
    )


if __name__ == "__main__":
    main()
