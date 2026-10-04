#!/usr/bin/env python3

"""
Held-out HUST real-current trajectory replay, v2.

Purpose
-------
Use externally measured HUST LFP charging-current trajectories on the
same frozen evaluation plant/controller stack.

This is NOT claimed as external plant validation:
  * HUST supplies measured current/time/voltage trajectories.
  * The simulated plant remains the study's dataset-informed ECM/thermal model.
  * LFP ECM R0/SOH are supplied by the strict MATR-informed parameter table.

Four arms
---------
1. HUST-Raw
   External HUST C-rate trajectory applied without study actuator projection.
   Reference only; NOT a feasible comparator under the study actuator limits.

2. HUST-Projected
   Same external HUST trajectory after the exact common actuator projection.

3. Physics-CEM
   Matched direct-physics CEM controller.

4. GraphOptimizer
   Frozen action-conditioned PackGNN controller.

HUST profiles are physically resampled on a 60 s grid from real time_in_s.
There is no index-stride sampling.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent
    ),
)

from graph_battery_pack import (
    PackGNN,
    build_pack_from_ecm,
)

from safe_fast_charge_optimizer import (
    GraphGuidedOptimizer,
    PhysicsCEMController,
    default_config,
    project_current_vector,
)


ROOT = Path(__file__).resolve().parent.parent

HUST_DIR = (
    ROOT
    / "data/processed/BatteryML/HUST"
)

ECM = (
    ROOT
    / "results/ecm/"
      "ecm_params_20260630_202031.parquet"
)

CKPT = (
    ROOT
    / "results/models/"
      "pack_gnn_action_v2_20261003_232518.pt"
)

EXPECTED_ECM_SHA = (
    "5580ff78d2518c645588d5bdd58ace965"
    "07e14bb2622dd6e26cf080397d94988"
)

EXPECTED_CKPT_SHA = (
    "2f3acc96e77504be0a060f5bee9bb226"
    "3b62df08c8fcd2afca353005c02e0b5f"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def longest_true_run(mask):
    best = None
    start = None

    for i, flag in enumerate(mask):

        if flag and start is None:
            start = i

        if (
            start is not None
            and
            (
                not flag
                or i == len(mask) - 1
            )
        ):
            end = (
                i
                if flag
                and i == len(mask) - 1
                else i - 1
            )

            if (
                best is None
                or
                end - start
                > best[1] - best[0]
            ):
                best = (
                    start,
                    end,
                )

            start = None

    return best


def finite_array(x):
    return np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)


def load_hust_cycle1_profiles():
    """
    Extract cycle-1 positive-charge profiles from all processed HUST cells.

    Alignment:
      HUST charge capacity is integrated in physical seconds.
      Profile t=0 is set where cumulative charge first corresponds to
      approximately 20% of nominal capacity, matching study SOC_init~0.20.

    Returned action is represented as C-rate so it can be transferred to
    each simulated cell's own Q_nom_Ah.
    """

    files = sorted(
        HUST_DIR.glob("*.pkl")
    )

    if len(files) != 77:
        raise RuntimeError(
            f"Expected 77 HUST files; found {len(files)}."
        )

    profiles = []

    for path in files:

        with path.open(
            "rb"
        ) as f:
            data = pickle.load(f)

        cycles = data.get(
            "cycle_data",
            [],
        )

        if not cycles:
            continue

        cyc = cycles[0]

        I = finite_array(
            cyc.get(
                "current_in_A",
                [],
            )
        )

        V = finite_array(
            cyc.get(
                "voltage_in_V",
                [],
            )
        )

        t = finite_array(
            cyc.get(
                "time_in_s",
                [],
            )
        )

        Qc = finite_array(
            cyc.get(
                "charge_capacity_in_Ah",
                [],
            )
        )

        n = len(I)

        if not (
            n >= 100
            and
            len(V) == n
            and
            len(t) == n
        ):
            continue

        if not (
            np.all(np.isfinite(I))
            and
            np.all(np.isfinite(V))
            and
            np.all(np.isfinite(t))
        ):
            continue

        if (
            len(Qc) != n
            or
            not np.all(
                np.isfinite(Qc)
            )
        ):
            # Reconstruct from actual seconds if required.
            Qc = np.zeros(
                n,
                dtype=np.float64,
            )

            for k in range(
                1,
                n,
            ):
                dt = (
                    t[k]
                    - t[k - 1]
                )

                if (
                    dt > 0
                    and I[k] > 0
                ):
                    Qc[k] = (
                        Qc[k - 1]
                        + I[k]
                        * dt
                        / 3600.0
                    )
                else:
                    Qc[k] = (
                        Qc[k - 1]
                    )

        run = longest_true_run(
            I > 0.05
        )

        if run is None:
            continue

        a, b = run

        if (
            b - a + 1
            < 100
        ):
            continue

        I = I[a:b + 1]
        V = V[a:b + 1]
        t = t[a:b + 1]
        Qc = Qc[a:b + 1]

        # Enforce strictly increasing timestamp points for np.interp.
        keep = np.ones(
            len(t),
            dtype=bool,
        )

        keep[1:] = (
            np.diff(t) > 0
        )

        I = I[keep]
        V = V[keep]
        t = t[keep]
        Qc = Qc[keep]

        if len(t) < 100:
            continue

        q_nom = float(
            data.get(
                "nominal_capacity_in_Ah",
                1.1,
            )
            or 1.1
        )

        q_rel = (
            Qc
            - Qc[0]
        )

        q_frac = (
            q_rel
            / q_nom
        )

        # Need enough observed charging trajectory after 20% SOC.
        if (
            np.nanmax(q_frac)
            < 0.80
        ):
            continue

        # Physical time corresponding to nominal SOC progress 0.20.
        # q_frac is monotonic during the positive-current charge run.
        t_start = float(
            np.interp(
                0.20,
                q_frac,
                t,
            )
        )

        if (
            t[-1]
            <= t_start
        ):
            continue

        # Build exact 60-s physical grid.
        duration_s = float(
            t[-1]
            - t_start
        )

        grid_s = np.arange(
            0.0,
            duration_s + 1e-9,
            60.0,
            dtype=np.float64,
        )

        absolute_grid = (
            t_start
            + grid_s
        )

        current_A = np.interp(
            absolute_grid,
            t,
            I,
        )

        measured_V = np.interp(
            absolute_grid,
            t,
            V,
        )

        c_rate = (
            current_A
            / q_nom
        )

        profiles.append({
            "cell_id":
                str(
                    data.get(
                        "cell_id",
                        path.stem,
                    )
                ),

            "file":
                path.name,

            "cycle_number":
                int(
                    cyc.get(
                        "cycle_number",
                        1,
                    )
                ),

            "q_nom_Ah":
                q_nom,

            "grid_s":
                grid_s,

            "c_rate":
                c_rate,

            "measured_voltage_V":
                measured_V,

            "duration_s":
                duration_s,

            "max_raw_C":
                float(
                    np.max(c_rate)
                ),

            "mean_raw_C":
                float(
                    np.mean(c_rate)
                ),

            "measured_peak_voltage_V":
                float(
                    np.max(measured_V)
                ),
        })

    if len(profiles) < 60:
        raise RuntimeError(
            "Too few usable HUST profiles: "
            f"{len(profiles)}"
        )

    return profiles


def profile_c_rate(
    profile,
    elapsed_s,
):
    """
    Physical-time interpolation.

    After the source trajectory ends, external current is zero.
    """

    grid = profile[
        "grid_s"
    ]

    values = profile[
        "c_rate"
    ]

    if (
        elapsed_s < 0
        or elapsed_s
        > grid[-1]
    ):
        return 0.0

    return float(
        np.interp(
            elapsed_s,
            grid,
            values,
        )
    )


def explicit_study_violations(
    pack,
):
    """
    Study safety envelope, evaluated explicitly rather than relying solely
    on legacy cell.is_safe bookkeeping.
    """

    soc = np.asarray(
        [
            c.SOC
            for c in pack.cells
        ],
        dtype=np.float64,
    )

    T = np.asarray(
        [
            c.T_C
            for c in pack.cells
        ],
        dtype=np.float64,
    )

    V = np.asarray(
        [
            c.V_term
            for c in pack.cells
        ],
        dtype=np.float64,
    )

    bad = (
        (soc < 0.05)
        |
        (soc > 0.98)
        |
        (T < -10.0)
        |
        (T > 45.0)
        |
        (V < 2.0)
        |
        (V > 3.5)
    )

    return int(
        np.sum(bad)
    )


def make_cfg():
    cfg = default_config()

    cfg["n_cells"] = 12
    cfg["chemistry"] = "LFP"

    cfg["target_soc"] = 0.80
    cfg["soc_init"] = 0.20
    cfg["soc_noise"] = 0.03
    cfg["T_amb"] = 25.0

    cfg["V_min"] = 2.0
    cfg["V_max"] = 3.5

    cfg["horizon"] = 5
    cfg["cem_samples"] = 64
    cfg["cem_elite_frac"] = 0.25
    cfg["cem_iterations"] = 5

    return cfg


def load_model(device):
    ckpt = torch.load(
        CKPT,
        map_location="cpu",
    )

    if (
        ckpt[
            "model_format_version"
        ]
        != 2
    ):
        raise RuntimeError(
            "Expected model format v2."
        )

    if not ckpt[
        "action_conditioned"
    ]:
        raise RuntimeError(
            "Checkpoint is not action-conditioned."
        )

    model = PackGNN(
        node_feat=(
            ckpt[
                "node_features"
            ]
        ),
        edge_feat=(
            ckpt[
                "edge_features"
            ]
        ),
        hidden=(
            ckpt[
                "hidden"
            ]
        ),
        n_layers=(
            ckpt[
                "n_layers"
            ]
        ),
        action_feat=(
            ckpt[
                "action_features"
            ]
        ),
    ).to(device)

    model.load_state_dict(
        ckpt[
            "model_state"
        ],
        strict=True,
    )

    model.eval()

    return model


def base_pack(
    cfg,
    seed,
):
    pack = build_pack_from_ecm(
        ecm_parquet=ECM,
        n_cells=12,
        chemistry="LFP",
        T_amb=25.0,
        soc_init=0.20,
        soc_noise=0.03,
        seed=seed,
        strict=True,
    )

    for cell in pack.cells:
        # Preserve explicit study-limit metadata.
        cell.V_min_limit = 2.0
        cell.V_max_limit = 3.5

    if (
        pack.ecm_source_mode
        != "dataset"
        or
        pack.ecm_dataset
        != "MATR"
    ):
        raise RuntimeError(
            "Strict LFP pack was not MATR dataset-backed."
        )

    return pack


def initial_signature(
    pack,
):
    rows = []

    for c in pack.cells:
        rows.append({
            "SOC":
                float(c.SOC),
            "T_C":
                float(c.T_C),
            "SOH":
                float(c.SOH),
            "R0":
                float(c.R0),
            "R1":
                float(c.R1),
            "C1":
                float(c.C1),
            "Q_nom_Ah":
                float(c.Q_nom_Ah),
        })

    return hashlib.sha256(
        json.dumps(
            rows,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode()
    ).hexdigest()


def summarize_final(
    pack,
):
    soc = np.asarray(
        [
            c.SOC
            for c in pack.cells
        ],
        dtype=np.float64,
    )

    T = np.asarray(
        [
            c.T_C
            for c in pack.cells
        ],
        dtype=np.float64,
    )

    V = np.asarray(
        [
            c.V_term
            for c in pack.cells
        ],
        dtype=np.float64,
    )

    return {
        "final_SOC_mean":
            float(
                np.mean(soc)
            ),

        "final_SOC_sigma_pct":
            float(
                np.std(soc)
                * 100.0
            ),

        "final_T_gradient_C":
            float(
                np.max(T)
                - np.min(T)
            ),

        "final_T_max_C":
            float(
                np.max(T)
            ),

        "final_V_max_V":
            float(
                np.max(V)
            ),
    }


def run_hust_arm(
    initial_pack,
    profiles,
    cfg,
    projected,
):
    pack = deepcopy(
        initial_pack
    )

    dt = float(
        cfg["dt_s"]
    )

    max_steps = int(
        cfg["max_steps"]
    )

    source_end_s = max(
        float(
            p["grid_s"][-1]
        )
        for p in profiles
    )

    total_explicit_viol = 0
    total_plant_viol = 0
    actuator_exceed_steps = 0

    peak_T = -np.inf
    max_dT = 0.0
    peak_V = -np.inf
    cumulative_aging = 0.0
    energy_Wh = 0.0

    applied_C_max = 0.0
    raw_C_max = 0.0

    steps = 0
    source_exhausted = False

    for step in range(
        max_steps
    ):
        soc_mean = float(
            np.mean(
                [
                    c.SOC
                    for c in pack.cells
                ]
            )
        )

        if (
            soc_mean
            >= cfg["target_soc"]
        ):
            break

        elapsed_s = (
            step
            * dt
        )

        if (
            elapsed_s
            > source_end_s
        ):
            source_exhausted = True
            break

        c_rates = np.asarray(
            [
                profile_c_rate(
                    p,
                    elapsed_s,
                )
                for p in profiles
            ],
            dtype=np.float64,
        )

        q_nom = np.asarray(
            [
                c.Q_nom_Ah
                for c in pack.cells
            ],
            dtype=np.float64,
        )

        raw_currents = (
            c_rates
            * q_nom
        )

        raw_C_max = max(
            raw_C_max,
            float(
                np.max(
                    np.divide(
                        raw_currents,
                        q_nom,
                        out=np.zeros_like(
                            raw_currents
                        ),
                        where=q_nom > 0,
                    )
                )
            ),
        )

        feasible = (
            project_current_vector(
                raw_currents,
                pack,
                cfg,
            )
        )

        if not np.allclose(
            raw_currents,
            feasible,
            rtol=1e-7,
            atol=1e-8,
        ):
            actuator_exceed_steps += 1

        currents = (
            np.asarray(
                feasible,
                dtype=np.float64,
            )
            if projected
            else raw_currents
        )

        applied_C = np.divide(
            currents,
            q_nom,
            out=np.zeros_like(
                currents
            ),
            where=q_nom > 0,
        )

        applied_C_max = max(
            applied_C_max,
            float(
                np.max(
                    applied_C
                )
            ),
        )

        before_V = np.asarray(
            [
                c.V_term
                for c in pack.cells
            ],
            dtype=np.float64,
        )

        metrics = pack.step(
            currents,
            dt=dt,
        )

        after_V = np.asarray(
            [
                c.V_term
                for c in pack.cells
            ],
            dtype=np.float64,
        )

        # Trapezoidal terminal-energy approximation.
        energy_Wh += float(
            np.sum(
                currents
                * (
                    before_V
                    + after_V
                )
                * 0.5
                * dt
                / 3600.0
            )
        )

        total_explicit_viol += (
            explicit_study_violations(
                pack
            )
        )

        total_plant_viol += int(
            metrics[
                "n_violations"
            ]
        )

        peak_T = max(
            peak_T,
            float(
                metrics[
                    "T_max"
                ]
            ),
        )

        max_dT = max(
            max_dT,
            float(
                metrics[
                    "T_gradient"
                ]
            ),
        )

        peak_V = max(
            peak_V,
            float(
                np.max(
                    metrics[
                        "V_term"
                    ]
                )
            ),
        )

        cumulative_aging += float(
            metrics[
                "aging_cost"
            ]
        )

        steps += 1

    final = summarize_final(
        pack
    )

    final.update({
        "steps":
            int(steps),

        "charging_time_min":
            float(
                steps
                * dt
                / 60.0
            ),

        "target_reached":
            bool(
                final[
                    "final_SOC_mean"
                ]
                >= cfg[
                    "target_soc"
                ]
            ),

        "source_exhausted":
            bool(
                source_exhausted
            ),

        "explicit_study_violations":
            int(
                total_explicit_viol
            ),

        "plant_reported_violations":
            int(
                total_plant_viol
            ),

        "actuator_exceed_steps":
            int(
                actuator_exceed_steps
            ),

        "raw_C_max":
            float(
                raw_C_max
            ),

        "applied_C_max":
            float(
                applied_C_max
            ),

        "peak_T_C":
            float(
                peak_T
            ),

        "max_T_gradient_C":
            float(
                max_dT
            ),

        "peak_voltage_V":
            float(
                peak_V
            ),

        "cumulative_aging":
            float(
                cumulative_aging
            ),

        "energy_Wh":
            float(
                energy_Wh
            ),
    })

    return final


def run_controller_arm(
    initial_pack,
    controller,
    cfg,
    control_seed,
):
    pack = deepcopy(
        initial_pack
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

    if hasattr(
        controller,
        "reset",
    ):
        controller.reset()

    dt = float(
        cfg["dt_s"]
    )

    total_explicit_viol = 0
    total_plant_viol = 0

    peak_T = -np.inf
    max_dT = 0.0
    peak_V = -np.inf
    cumulative_aging = 0.0
    energy_Wh = 0.0

    steps = 0

    for _ in range(
        int(
            cfg[
                "max_steps"
            ]
        )
    ):
        if (
            np.mean(
                [
                    c.SOC
                    for c in pack.cells
                ]
            )
            >= cfg[
                "target_soc"
            ]
        ):
            break

        currents = np.asarray(
            controller.get_currents(
                pack
            ),
            dtype=np.float64,
        )

        projected = np.asarray(
            project_current_vector(
                currents,
                pack,
                cfg,
            ),
            dtype=np.float64,
        )

        if not np.allclose(
            currents,
            projected,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Controller returned action "
                "outside common actuator projection."
            )

        before_V = np.asarray(
            [
                c.V_term
                for c in pack.cells
            ],
            dtype=np.float64,
        )

        metrics = pack.step(
            currents,
            dt=dt,
        )

        after_V = np.asarray(
            [
                c.V_term
                for c in pack.cells
            ],
            dtype=np.float64,
        )

        energy_Wh += float(
            np.sum(
                currents
                * (
                    before_V
                    + after_V
                )
                * 0.5
                * dt
                / 3600.0
            )
        )

        total_explicit_viol += (
            explicit_study_violations(
                pack
            )
        )

        total_plant_viol += int(
            metrics[
                "n_violations"
            ]
        )

        peak_T = max(
            peak_T,
            float(
                metrics[
                    "T_max"
                ]
            ),
        )

        max_dT = max(
            max_dT,
            float(
                metrics[
                    "T_gradient"
                ]
            ),
        )

        peak_V = max(
            peak_V,
            float(
                np.max(
                    metrics[
                        "V_term"
                    ]
                )
            ),
        )

        cumulative_aging += float(
            metrics[
                "aging_cost"
            ]
        )

        steps += 1

    final = summarize_final(
        pack
    )

    final.update({
        "steps":
            int(steps),

        "charging_time_min":
            float(
                steps
                * dt
                / 60.0
            ),

        "target_reached":
            bool(
                final[
                    "final_SOC_mean"
                ]
                >= cfg[
                    "target_soc"
                ]
            ),

        "source_exhausted":
            False,

        "explicit_study_violations":
            int(
                total_explicit_viol
            ),

        "plant_reported_violations":
            int(
                total_plant_viol
            ),

        "actuator_exceed_steps":
            0,

        "raw_C_max":
            np.nan,

        "applied_C_max":
            np.nan,

        "peak_T_C":
            float(
                peak_T
            ),

        "max_T_gradient_C":
            float(
                max_dT
            ),

        "peak_voltage_V":
            float(
                peak_V
            ),

        "cumulative_aging":
            float(
                cumulative_aging
            ),

        "energy_Wh":
            float(
                energy_Wh
            ),
    })

    return final


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--n-episodes",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--seed-base",
        type=int,
        default=620001,
    )

    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = ap.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        args.output_dir
        / "hust_timestamp_replay_v2_raw.jsonl"
    )

    summary_path = (
        args.output_dir
        / "hust_timestamp_replay_v2_summary.json"
    )

    provenance_path = (
        args.output_dir
        / "hust_timestamp_replay_v2_provenance.json"
    )

    if sha256(
        ECM
    ) != EXPECTED_ECM_SHA:
        raise RuntimeError(
            "ECM SHA mismatch."
        )

    if sha256(
        CKPT
    ) != EXPECTED_CKPT_SHA:
        raise RuntimeError(
            "Checkpoint SHA mismatch."
        )

    cfg = make_cfg()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = load_model(
        device
    )

    profiles = (
        load_hust_cycle1_profiles()
    )

    print(
        "=" * 92
    )
    print(
        "HUST TIMESTAMP REPLAY V2"
    )
    print(
        "=" * 92
    )

    print(
        "device:",
        device,
    )

    print(
        "usable HUST cycle-1 profiles:",
        len(profiles),
    )

    print(
        "physical HUST native timestamp "
        "-> 60 s interpolation: ENABLED"
    )

    print(
        "strict ECM mode: ENABLED"
    )

    print(
        "HUST-Raw is reference only; "
        "not actuator-comparable."
    )

    existing = []

    if raw_path.exists():
        existing = [
            json.loads(line)
            for line
            in raw_path.read_text().splitlines()
            if line.strip()
        ]

    complete = {
        (
            int(r["episode"]),
            r["arm"],
        )
        for r in existing
    }

    rows = list(
        existing
    )

    git_commit = (
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )

    arms = [
        "HUST-Raw",
        "HUST-Projected",
        "Physics-CEM",
        "GraphOptimizer",
    ]

    for ep in range(
        args.n_episodes
    ):
        pack_seed = (
            args.seed_base
            + 7 * ep
        )

        select_rng = (
            np.random.default_rng(
                90_000_000
                + pack_seed
            )
        )

        idx = select_rng.choice(
            len(profiles),
            size=12,
            replace=False,
        )

        selected = [
            profiles[
                int(i)
            ]
            for i in idx
        ]

        initial = base_pack(
            cfg,
            pack_seed,
        )

        sig = initial_signature(
            initial
        )

        source_ids = [
            p["cell_id"]
            for p in selected
        ]

        source_peak_V = max(
            p[
                "measured_peak_voltage_V"
            ]
            for p in selected
        )

        source_max_C = max(
            p[
                "max_raw_C"
            ]
            for p in selected
        )

        print()
        print(
            f"===== episode "
            f"{ep + 1}/{args.n_episodes} "
            f"| seed={pack_seed} ====="
        )

        for arm in arms:

            key = (
                ep,
                arm,
            )

            if key in complete:
                print(
                    f"  SKIP {arm}"
                )
                continue

            t0 = time.time()

            if arm == "HUST-Raw":

                result = run_hust_arm(
                    initial,
                    selected,
                    cfg,
                    projected=False,
                )

            elif arm == "HUST-Projected":

                result = run_hust_arm(
                    initial,
                    selected,
                    cfg,
                    projected=True,
                )

            elif arm == "Physics-CEM":

                result = (
                    run_controller_arm(
                        initial,
                        PhysicsCEMController(
                            cfg
                        ),
                        cfg,
                        (
                            40_000_000
                            + pack_seed
                        ),
                    )
                )

            elif arm == "GraphOptimizer":

                result = (
                    run_controller_arm(
                        initial,
                        GraphGuidedOptimizer(
                            cfg,
                            model,
                        ),
                        cfg,
                        (
                            40_000_000
                            + pack_seed
                        ),
                    )
                )

            else:
                raise KeyError(
                    arm
                )

            result.update({
                "episode":
                    int(ep),

                "pack_seed":
                    int(
                        pack_seed
                    ),

                "arm":
                    arm,

                "initial_pack_sha256":
                    sig,

                "hust_source_cell_ids":
                    source_ids,

                "hust_source_cycle":
                    1,

                "source_measured_peak_voltage_V":
                    float(
                        source_peak_V
                    ),

                "source_max_C":
                    float(
                        source_max_C
                    ),

                "wall_time_s":
                    float(
                        time.time()
                        - t0
                    ),
            })

            with raw_path.open(
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(
                        result,
                        sort_keys=True,
                    )
                    + "\n"
                )

                f.flush()
                os.fsync(
                    f.fileno()
                )

            rows.append(
                result
            )

            complete.add(
                key
            )

            print(
                f"  {arm:15s} | "
                f"time="
                f"{result['charging_time_min']:.1f}m | "
                f"SOC="
                f"{result['final_SOC_mean']:.5f} | "
                f"sigma="
                f"{result['final_SOC_sigma_pct']:.4f}% | "
                f"dT="
                f"{result['final_T_gradient_C']:.4f}C | "
                f"Vpk="
                f"{result['peak_voltage_V']:.6f} | "
                f"viol="
                f"{result['explicit_study_violations']} | "
                f"target="
                f"{result['target_reached']} | "
                f"actX="
                f"{result['actuator_exceed_steps']}"
            )

    # Integrity
    rows = [
        json.loads(line)
        for line
        in raw_path.read_text().splitlines()
        if line.strip()
    ]

    expected = {
        (
            ep,
            arm,
        )
        for ep in range(
            args.n_episodes
        )
        for arm in arms
    }

    actual = {
        (
            int(
                r["episode"]
            ),
            r["arm"],
        )
        for r in rows
    }

    if actual != expected:
        raise RuntimeError(
            "Replay completeness failure."
        )

    for ep in range(
        args.n_episodes
    ):
        sub = [
            r
            for r in rows
            if int(
                r["episode"]
            ) == ep
        ]

        if len({
            r[
                "initial_pack_sha256"
            ]
            for r in sub
        }) != 1:
            raise RuntimeError(
                "Paired initial-pack mismatch."
            )

    summary = {}

    for arm in arms:
        sub = [
            r
            for r in rows
            if r[
                "arm"
            ] == arm
        ]

        summary[
            arm
        ] = {
            "n":
                len(sub),

            "target_successes":
                int(
                    sum(
                        bool(
                            r[
                                "target_reached"
                            ]
                        )
                        for r in sub
                    )
                ),

            "mean_time_min":
                float(
                    np.mean(
                        [
                            r[
                                "charging_time_min"
                            ]
                            for r in sub
                        ]
                    )
                ),

            "mean_sigma_SOC_pct":
                float(
                    np.mean(
                        [
                            r[
                                "final_SOC_sigma_pct"
                            ]
                            for r in sub
                        ]
                    )
                ),

            "mean_final_dT_C":
                float(
                    np.mean(
                        [
                            r[
                                "final_T_gradient_C"
                            ]
                            for r in sub
                        ]
                    )
                ),

            "maximum_peak_voltage_V":
                float(
                    np.max(
                        [
                            r[
                                "peak_voltage_V"
                            ]
                            for r in sub
                        ]
                    )
                ),

            "total_explicit_study_violations":
                int(
                    sum(
                        int(
                            r[
                                "explicit_study_violations"
                            ]
                        )
                        for r in sub
                    )
                ),

            "episodes_with_explicit_violation":
                int(
                    sum(
                        int(
                            r[
                                "explicit_study_violations"
                            ]
                        )
                        > 0
                        for r in sub
                    )
                ),

            "mean_aging":
                float(
                    np.mean(
                        [
                            r[
                                "cumulative_aging"
                            ]
                            for r in sub
                        ]
                    )
                ),

            "mean_energy_Wh":
                float(
                    np.mean(
                        [
                            r[
                                "energy_Wh"
                            ]
                            for r in sub
                        ]
                    )
                ),
        }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    provenance = {
        "git_commit":
            git_commit,

        "checkpoint_sha256":
            sha256(
                CKPT
            ),

        "ecm_sha256":
            sha256(
                ECM
            ),

        "hust_processed_file_count":
            len(
                list(
                    HUST_DIR.glob(
                        "*.pkl"
                    )
                )
            ),

        "usable_cycle1_profiles":
            len(
                profiles
            ),

        "n_episodes":
            args.n_episodes,

        "pack_seeds":
            [
                args.seed_base
                + 7 * i
                for i in range(
                    args.n_episodes
                )
            ],

        "timestamp_protocol":
            (
                "HUST real time_in_s; "
                "capacity-aligned at nominal SOC 0.20; "
                "physical interpolation every 60 s"
            ),

        "hust_arm_interpretation": {
            "HUST-Raw":
                (
                    "external current-profile reference only; "
                    "not constrained to study actuator limits"
                ),

            "HUST-Projected":
                (
                    "external HUST C-rate profile after exact "
                    "study actuator projection"
                ),

            "Physics-CEM":
                "matched direct-physics controller",

            "GraphOptimizer":
                "frozen action-conditioned PackGNN controller",
        },

        "claim_boundary":
            (
                "HUST provides external measured current/time/voltage "
                "trajectories. Simulated responses use the study "
                "MATR-informed ECM/thermal plant. This is external-action/"
                "protocol replay, not fully external plant validation."
            ),
    }

    provenance_path.write_text(
        json.dumps(
            provenance,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "=" * 92
    )
    print(
        "FINAL HUST REPLAY SUMMARY"
    )
    print(
        "=" * 92
    )

    for arm in arms:
        s = summary[
            arm
        ]

        print(
            f"{arm:15s} | "
            f"N={s['n']} | "
            f"target="
            f"{s['target_successes']}/{s['n']} | "
            f"time="
            f"{s['mean_time_min']:.3f}m | "
            f"sigma="
            f"{s['mean_sigma_SOC_pct']:.4f}% | "
            f"dT="
            f"{s['mean_final_dT_C']:.4f}C | "
            f"Vmax="
            f"{s['maximum_peak_voltage_V']:.6f} | "
            f"viol="
            f"{s['total_explicit_study_violations']}"
        )

    print()
    print(
        "HUST TIMESTAMP REPLAY V2: COMPLETE"
    )


if __name__ == "__main__":
    main()
