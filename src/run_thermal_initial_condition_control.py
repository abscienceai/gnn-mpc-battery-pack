#!/usr/bin/env python3

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")

from graph_battery_pack import (
    PackGNN,
    build_pack_from_ecm,
)

from safe_fast_charge_optimizer import (
    PhysicsCEMController,
    GraphGuidedOptimizer,
    default_config,
    project_current_vector,
)


ROOT = Path.cwd()

CKPT = (
    ROOT
    / "results/models/"
      "pack_gnn_action_v2_20261003_232518.pt"
)

ECM = (
    ROOT
    / "results/ecm/"
      "ecm_params_20260630_202031.parquet"
)

OUT_DIR = (
    ROOT
    / "results/diagnostics/"
      "thermal_initial_condition_n30"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW = OUT_DIR / "thermal_initial_condition_raw.jsonl"
SUMMARY = OUT_DIR / "thermal_initial_condition_summary.json"

PROV = (
    ROOT
    / "provenance"
    / "thermal_initial_condition_n30.json"
)

EXPECTED_CKPT_SHA = (
    "2f3acc96e77504be0a060f5bee9bb226"
    "3b62df08c8fcd2afca353005c02e0b5f"
)

EXPECTED_ECM_SHA = (
    "5580ff78d2518c645588d5bdd58ace965"
    "07e14bb2622dd6e26cf080397d94988"
)

# Completely disjoint diagnostic seeds.
SEEDS = [
    510001 + 7 * i
    for i in range(30)
]

CONDITIONS = [
    "heterogeneous_T",
    "uniform_25C",
]

CONTROLLERS = [
    "Physics-CEM",
    "GraphOptimizer",
]


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


assert CKPT.exists()
assert ECM.exists()

assert sha256(CKPT) == EXPECTED_CKPT_SHA
assert sha256(ECM) == EXPECTED_ECM_SHA


cfg = default_config()

cfg["n_cells"] = 12
cfg["chemistry"] = "LFP"

cfg["target_soc"] = 0.80
cfg["soc_init"] = 0.20
cfg["soc_noise"] = 0.03
cfg["T_amb"] = 25.0

cfg["V_min"] = 2.0
cfg["V_max"] = 3.5
cfg["T_max"] = 45.0

cfg["horizon"] = 5
cfg["cem_samples"] = 64
cfg["cem_elite_frac"] = 0.25
cfg["cem_iterations"] = 5


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

checkpoint = torch.load(
    CKPT,
    map_location="cpu",
)

assert checkpoint["model_format_version"] == 2
assert checkpoint["action_conditioned"] is True

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

gnn = PackGNN(
    node_feat=checkpoint["node_features"],
    edge_feat=checkpoint["edge_features"],
    hidden=checkpoint["hidden"],
    n_layers=checkpoint["n_layers"],
    action_feat=checkpoint["action_features"],
).to(device)

gnn.load_state_dict(
    checkpoint["model_state"],
    strict=True,
)

gnn.eval()


# ------------------------------------------------------------
# Counting controller wrappers
# ------------------------------------------------------------

class CountingPhysicsCEM(
    PhysicsCEMController
):
    def reset(self):
        super().reset()

        self.filter_calls = 0
        self.filter_activations = 0

    def _physics_safety_filter(
        self,
        pack,
        currents,
        n_bisection=24,
    ):
        proposed = project_current_vector(
            currents,
            pack,
            self.cfg,
        )

        filtered = super()._physics_safety_filter(
            pack,
            currents,
            n_bisection=n_bisection,
        )

        self.filter_calls += 1

        if not np.allclose(
            proposed,
            filtered,
            rtol=1e-6,
            atol=1e-6,
        ):
            self.filter_activations += 1

        return filtered


class CountingGraphOptimizer(
    GraphGuidedOptimizer
):
    def reset(self):
        super().reset()

        self.filter_calls = 0
        self.filter_activations = 0

    def _physics_safety_filter(
        self,
        pack,
        currents,
        n_bisection=24,
    ):
        proposed = project_current_vector(
            currents,
            pack,
            self.cfg,
        )

        filtered = super()._physics_safety_filter(
            pack,
            currents,
            n_bisection=n_bisection,
        )

        self.filter_calls += 1

        if not np.allclose(
            proposed,
            filtered,
            rtol=1e-6,
            atol=1e-6,
        ):
            self.filter_activations += 1

        return filtered


def make_controller(name):

    if name == "Physics-CEM":
        return CountingPhysicsCEM(
            cfg
        )

    if name == "GraphOptimizer":
        return CountingGraphOptimizer(
            cfg,
            gnn,
        )

    raise KeyError(name)


# ------------------------------------------------------------
# Initial-condition integrity
# ------------------------------------------------------------

def nonthermal_signature(pack):
    rows = []

    for c in pack.cells:
        rows.append({
            "SOC": float(c.SOC),
            "SOH": float(c.SOH),
            "V_oc": float(c.V_oc),
            "V_term": float(c.V_term),
            "R0": float(c.R0),
            "R1": float(c.R1),
            "C1": float(c.C1),
            "Q_nom_Ah": float(c.Q_nom_Ah),
            "chemistry": c.chemistry,
        })

    return hashlib.sha256(
        json.dumps(
            rows,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def full_signature(pack):
    rows = []

    for c in pack.cells:
        rows.append({
            "SOC": float(c.SOC),
            "SOH": float(c.SOH),
            "T_C": float(c.T_C),
            "V_oc": float(c.V_oc),
            "V_term": float(c.V_term),
            "R0": float(c.R0),
            "R1": float(c.R1),
            "C1": float(c.C1),
            "Q_nom_Ah": float(c.Q_nom_Ah),
            "chemistry": c.chemistry,
        })

    return hashlib.sha256(
        json.dumps(
            rows,
            sort_keys=True,
        ).encode()
    ).hexdigest()


# ------------------------------------------------------------
# Episode
# ------------------------------------------------------------

def run_episode(
    initial_pack,
    controller_name,
    seed,
):

    pack = deepcopy(
        initial_pack
    )

    # Same stochastic search stream for the
    # paired controllers and temperature conditions.
    control_seed = (
        30_000_000
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

    controller.reset()

    initial_T = np.asarray(
        [c.T_C for c in pack.cells],
        dtype=float,
    )

    violations = 0
    peak_T = -np.inf
    max_dT = 0.0
    peak_V = -np.inf
    aging = 0.0

    steps = 0

    t0 = time.time()

    for _ in range(
        cfg["max_steps"]
    ):

        soc_mean = float(
            np.mean(
                [
                    c.SOC
                    for c in pack.cells
                ]
            )
        )

        if soc_mean >= cfg["target_soc"]:
            break

        currents = np.asarray(
            controller.get_currents(
                pack
            ),
            dtype=float,
        )

        projected = np.asarray(
            project_current_vector(
                currents,
                pack,
                cfg,
            ),
            dtype=float,
        )

        assert np.allclose(
            currents,
            projected,
            rtol=1e-6,
            atol=1e-6,
        )

        m = pack.step(
            currents,
            dt=cfg["dt_s"],
        )

        steps += 1

        violations += int(
            m["n_violations"]
        )

        peak_T = max(
            peak_T,
            float(m["T_max"]),
        )

        max_dT = max(
            max_dT,
            float(m["T_gradient"]),
        )

        peak_V = max(
            peak_V,
            float(
                np.max(
                    m["V_term"]
                )
            ),
        )

        aging += float(
            m["aging_cost"]
        )

    wall = time.time() - t0

    final_soc = np.asarray(
        [c.SOC for c in pack.cells],
        dtype=float,
    )

    final_T = np.asarray(
        [c.T_C for c in pack.cells],
        dtype=float,
    )

    return {
        "seed":
            int(seed),

        "controller":
            controller_name,

        "control_seed":
            int(control_seed),

        "initial_T_mean_C":
            float(np.mean(initial_T)),

        "initial_T_gradient_C":
            float(
                np.max(initial_T)
                - np.min(initial_T)
            ),

        "steps":
            int(steps),

        "charging_time_min":
            float(
                steps
                * cfg["dt_s"]
                / 60.0
            ),

        "final_SOC_mean":
            float(
                np.mean(final_soc)
            ),

        "final_SOC_sigma_pct":
            float(
                np.std(final_soc)
                * 100.0
            ),

        "final_T_gradient_C":
            float(
                np.max(final_T)
                - np.min(final_T)
            ),

        "max_T_gradient_C":
            float(max_dT),

        "peak_T_C":
            float(peak_T),

        "peak_voltage_V":
            float(peak_V),

        "cumulative_aging":
            float(aging),

        "total_violations":
            int(violations),

        "target_reached":
            bool(
                np.mean(final_soc)
                >= cfg["target_soc"]
            ),

        "safety_filter_calls":
            int(
                controller.filter_calls
            ),

        "safety_filter_activations":
            int(
                controller.filter_activations
            ),

        "wall_time_s":
            float(wall),
    }


# ------------------------------------------------------------
# Incremental persistence
# ------------------------------------------------------------

def load_existing():

    if not RAW.exists():
        return []

    rows = []

    for line in RAW.read_text().splitlines():
        if line.strip():
            rows.append(
                json.loads(line)
            )

    return rows


def append_row(row):

    with open(
        RAW,
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

        f.flush()
        os.fsync(
            f.fileno()
        )


def write_summary(rows):

    output = {}

    for condition in CONDITIONS:

        output[condition] = {}

        for controller in CONTROLLERS:

            sub = [
                r
                for r in rows
                if (
                    r["condition"] == condition
                    and
                    r["controller"] == controller
                )
            ]

            if not sub:
                continue

            output[condition][controller] = {
                "n":
                    len(sub),

                "charging_time_mean_min":
                    float(
                        np.mean(
                            [
                                r["charging_time_min"]
                                for r in sub
                            ]
                        )
                    ),

                "sigma_SOC_mean_pct":
                    float(
                        np.mean(
                            [
                                r["final_SOC_sigma_pct"]
                                for r in sub
                            ]
                        )
                    ),

                "final_dT_mean_C":
                    float(
                        np.mean(
                            [
                                r["final_T_gradient_C"]
                                for r in sub
                            ]
                        )
                    ),

                "max_dT_mean_C":
                    float(
                        np.mean(
                            [
                                r["max_T_gradient_C"]
                                for r in sub
                            ]
                        )
                    ),

                "peak_T_mean_C":
                    float(
                        np.mean(
                            [
                                r["peak_T_C"]
                                for r in sub
                            ]
                        )
                    ),

                "violations_total":
                    int(
                        sum(
                            r["total_violations"]
                            for r in sub
                        )
                    ),

                "target_successes":
                    int(
                        sum(
                            r["target_reached"]
                            for r in sub
                        )
                    ),
            }

    SUMMARY.write_text(
        json.dumps(
            output,
            indent=2,
        )
        + "\n"
    )

    return output


# ------------------------------------------------------------
# Provenance
# ------------------------------------------------------------

git_commit = subprocess.check_output(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    text=True,
).strip()

provenance = {
    "git_commit":
        git_commit,

    "checkpoint_sha256":
        sha256(CKPT),

    "ecm_sha256":
        sha256(ECM),

    "diagnostic_seeds":
        SEEDS,

    "conditions":
        CONDITIONS,

    "controllers":
        CONTROLLERS,

    "design":
        (
            "2x2 paired diagnostic: "
            "natural heterogeneous initial T vs "
            "all cells forced to 25 C; "
            "Physics-CEM vs GraphOptimizer"
        ),

    "uniform_temperature_C":
        25.0,

    "control_rng":
        "30,000,000 + diagnostic pack seed",
}

PROV.write_text(
    json.dumps(
        provenance,
        indent=2,
    )
    + "\n"
)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

rows = load_existing()

completed = {
    (
        int(r["seed"]),
        r["condition"],
        r["controller"],
    )
    for r in rows
}

print("=" * 84)
print(
    "THERMAL INITIAL-CONDITION CONTROL N=30"
)
print("=" * 84)

print(
    "git:",
    git_commit,
)

print(
    "device:",
    device,
)

print(
    "rows already complete:",
    len(completed),
    "/ 120",
)

print()


for idx, seed in enumerate(
    SEEDS,
    1,
):

    base = build_pack_from_ecm(
        ecm_parquet=ECM,
        n_cells=12,
        chemistry="LFP",
        T_amb=25.0,
        soc_init=0.20,
        soc_noise=0.03,
        seed=seed,
        strict=True,
    )

    for c in base.cells:
        c.V_min_limit = 2.0
        c.V_max_limit = 3.5

    heterogeneous = deepcopy(
        base
    )

    uniform = deepcopy(
        base
    )

    for c in uniform.cells:
        c.T_C = 25.0

    # This is the key control assertion:
    # ONLY temperature may differ between conditions.
    assert (
        nonthermal_signature(
            heterogeneous
        )
        ==
        nonthermal_signature(
            uniform
        )
    )

    condition_packs = {
        "heterogeneous_T":
            heterogeneous,

        "uniform_25C":
            uniform,
    }

    print(
        f"===== seed {idx:02d}/30 "
        f"| {seed} =====",
        flush=True,
    )

    for condition in CONDITIONS:

        initial = condition_packs[
            condition
        ]

        condition_sha = full_signature(
            initial
        )

        initial_dT = (
            max(c.T_C for c in initial.cells)
            -
            min(c.T_C for c in initial.cells)
        )

        print(
            f"  {condition:16s} "
            f"initial dT="
            f"{initial_dT:.4f} C",
            flush=True,
        )

        for controller_name in CONTROLLERS:

            key = (
                seed,
                condition,
                controller_name,
            )

            if key in completed:
                print(
                    f"    SKIP "
                    f"{controller_name}",
                    flush=True,
                )
                continue

            row = run_episode(
                initial,
                controller_name,
                seed,
            )

            row["condition"] = condition
            row[
                "initial_condition_sha256"
            ] = condition_sha

            append_row(row)

            rows.append(row)
            completed.add(key)

            write_summary(rows)

            print(
                f"    {controller_name:15s} | "
                f"time="
                f"{row['charging_time_min']:.1f}m | "
                f"sigma="
                f"{row['final_SOC_sigma_pct']:.4f}% | "
                f"final dT="
                f"{row['final_T_gradient_C']:.4f}C | "
                f"max dT="
                f"{row['max_T_gradient_C']:.4f}C | "
                f"Tmax="
                f"{row['peak_T_C']:.3f}C | "
                f"viol="
                f"{row['total_violations']} | "
                f"filter="
                f"{row['safety_filter_activations']}/"
                f"{row['safety_filter_calls']}",
                flush=True,
            )

    print(
        f"  completed: "
        f"{len(completed)}/120",
        flush=True,
    )

    print()


# ------------------------------------------------------------
# Completeness / paired integrity
# ------------------------------------------------------------

rows = load_existing()

expected = {
    (
        seed,
        condition,
        controller,
    )
    for seed in SEEDS
    for condition in CONDITIONS
    for controller in CONTROLLERS
}

actual = {
    (
        int(r["seed"]),
        r["condition"],
        r["controller"],
    )
    for r in rows
}

assert len(rows) == 120
assert actual == expected


for seed in SEEDS:
    for condition in CONDITIONS:

        sub = [
            r
            for r in rows
            if (
                int(r["seed"]) == seed
                and
                r["condition"] == condition
            )
        ]

        assert len(sub) == 2

        assert len({
            r[
                "initial_condition_sha256"
            ]
            for r in sub
        }) == 1


summary = write_summary(
    rows
)


# ------------------------------------------------------------
# Paired thermal diagnostic
# ------------------------------------------------------------

def values(
    condition,
    controller,
    key,
):
    sub = sorted(
        [
            r
            for r in rows
            if (
                r["condition"] == condition
                and
                r["controller"] == controller
            )
        ],
        key=lambda x: x["seed"],
    )

    return np.asarray(
        [
            r[key]
            for r in sub
        ],
        dtype=float,
    )


print("=" * 84)
print("FINAL THERMAL CONTROL SUMMARY")
print("=" * 84)

for condition in CONDITIONS:

    print()
    print(condition)

    for controller in CONTROLLERS:

        s = summary[
            condition
        ][controller]

        print(
            f"  {controller:15s} | "
            f"N={s['n']} | "
            f"time="
            f"{s['charging_time_mean_min']:.3f}m | "
            f"sigma="
            f"{s['sigma_SOC_mean_pct']:.4f}% | "
            f"final dT="
            f"{s['final_dT_mean_C']:.4f}C | "
            f"max dT="
            f"{s['max_dT_mean_C']:.4f}C | "
            f"Tmax="
            f"{s['peak_T_mean_C']:.3f}C | "
            f"viol="
            f"{s['violations_total']} | "
            f"target="
            f"{s['target_successes']}/30"
        )


print()
print("=" * 84)
print("GRAPH - PHYSICS THERMAL DIFFERENCE")
print("=" * 84)

differences = {}

for condition in CONDITIONS:

    G = values(
        condition,
        "GraphOptimizer",
        "final_T_gradient_C",
    )

    P = values(
        condition,
        "Physics-CEM",
        "final_T_gradient_C",
    )

    d = G - P

    differences[condition] = d

    print(
        f"{condition:16s} | "
        f"mean ΔdT="
        f"{np.mean(d):+.6f} C | "
        f"median="
        f"{np.median(d):+.6f} C | "
        f"Graph lower in "
        f"{int(np.sum(d < 0))}/30"
    )


interaction = (
    differences["uniform_25C"]
    -
    differences["heterogeneous_T"]
)

print()
print(
    "temperature-condition interaction "
    "[(G-P)_uniform - (G-P)_heterogeneous]:"
)

print(
    "mean:",
    float(
        np.mean(interaction)
    ),
    "C"
)

print(
    "median:",
    float(
        np.median(interaction)
    ),
    "C"
)

print()
print(
    "THERMAL INITIAL-CONDITION CONTROL: COMPLETE"
)
