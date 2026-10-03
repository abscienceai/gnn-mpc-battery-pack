#!/usr/bin/env python3

from copy import deepcopy
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(
    0,
    str(Path(__file__).parent),
)

from graph_battery_pack import (
    PackGNN,
    build_pack_from_ecm,
)
from safe_fast_charge_optimizer import (
    CCCVController,
    BalancedCCCVController,
    ProportionalController,
    PhysicsCEMController,
    GraphGuidedOptimizer,
    default_config,
    project_current_vector,
)


# ================================================================
# Fixed canonical protocol
# ================================================================

ROOT = Path(__file__).resolve().parent.parent

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
    / "results/canonical_lfp_n30"
)

PROV_DIR = (
    ROOT.parent
    / "provenance"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROV_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

JSONL = (
    OUT_DIR
    / "canonical_lfp_n30_raw.jsonl"
)

SUMMARY_JSON = (
    OUT_DIR
    / "canonical_lfp_n30_summary.json"
)

SUMMARY_CSV = (
    OUT_DIR
    / "canonical_lfp_n30_summary.csv"
)

PROVENANCE_JSON = (
    PROV_DIR
    / "canonical_lfp_n30_provenance.json"
)

EXPECTED_CKPT_SHA = (
    "2f3acc96e77504be0a060f5bee9bb226"
    "3b62df08c8fcd2afca353005c02e0b5f"
)

SEEDS = [
    7 * i
    for i in range(30)
]

CONTROLLERS = [
    "CC-CV",
    "CC-CV-Balance",
    "Proportional",
    "Physics-CEM",
    "GraphOptimizer",
]


# ================================================================
# Configuration
# ================================================================

cfg = default_config()

cfg["n_cells"] = 12
cfg["chemistry"] = "LFP"

cfg["target_soc"] = 0.80
cfg["soc_init"] = 0.20
cfg["soc_noise"] = 0.03
cfg["T_amb"] = 25.0

# Dataset-derived MATR/LFP voltage limits.
cfg["V_min"] = 2.0
cfg["V_max"] = 3.5

cfg["T_max"] = 45.0

# Matched CEM budget.
cfg["horizon"] = 5
cfg["cem_samples"] = 64
cfg["cem_elite_frac"] = 0.25
cfg["cem_iterations"] = 5


# ================================================================
# General helpers
# ================================================================

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def git_head():
    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=ROOT,
        text=True,
    ).strip()


def initial_pack_record(pack):
    rows = []

    attrs = [
        "cell_id",
        "SOC",
        "SOH",
        "T_C",
        "V_oc",
        "V_term",
        "R0",
        "R1",
        "C1",
        "I_A",
        "Q_nom_Ah",
        "chemistry",
    ]

    for c in pack.cells:
        row = {}

        for attr in attrs:
            if hasattr(c, attr):
                value = getattr(
                    c,
                    attr,
                )

                if isinstance(
                    value,
                    (
                        np.floating,
                        np.integer,
                    ),
                ):
                    value = value.item()

                row[attr] = value

        rows.append(row)

    return rows


def fingerprint_pack(pack):
    record = initial_pack_record(
        pack
    )

    raw = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    return hashlib.sha256(
        raw
    ).hexdigest()


def append_jsonl(row):
    line = (
        json.dumps(
            row,
            sort_keys=True,
        )
        + "\n"
    )

    with open(
        JSONL,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(line)
        f.flush()
        os.fsync(
            f.fileno()
        )


def load_existing():
    rows = []

    if not JSONL.exists():
        return rows

    with open(
        JSONL,
        encoding="utf-8",
    ) as f:

        for lineno, line in enumerate(
            f,
            1,
        ):
            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except Exception as e:
                raise RuntimeError(
                    f"Invalid JSONL at line "
                    f"{lineno}: {e}"
                )

    return rows


# ================================================================
# GraphOptimizer instrumentation
# ================================================================

class CanonicalGraphOptimizer(
    GraphGuidedOptimizer
):
    def reset(self):
        super().reset()

        self.filter_calls = 0
        self.filter_activations = 0
        self.filter_scales = []

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

        filtered = (
            super()
            ._physics_safety_filter(
                pack,
                currents,
                n_bisection=n_bisection,
            )
        )

        self.filter_calls += 1

        denom = float(
            np.sum(proposed)
        )

        numer = float(
            np.sum(filtered)
        )

        scale = (
            numer / denom
            if denom > 1e-12
            else 1.0
        )

        self.filter_scales.append(
            float(scale)
        )

        if not np.allclose(
            proposed,
            filtered,
            rtol=1e-6,
            atol=1e-6,
        ):
            self.filter_activations += 1

        return filtered


# ================================================================
# Load PackGNN v2
# ================================================================

checkpoint_sha = sha256(
    CKPT
)

if (
    checkpoint_sha
    != EXPECTED_CKPT_SHA
):
    raise RuntimeError(
        "Canonical checkpoint SHA mismatch:\n"
        f"expected {EXPECTED_CKPT_SHA}\n"
        f"actual   {checkpoint_sha}"
    )

ckpt = torch.load(
    CKPT,
    map_location="cpu",
)

assert (
    ckpt["model_format_version"]
    == 2
)

assert (
    ckpt["action_conditioned"]
    is True
)

# Canonical seeds must remain untouched
# by training and validation.
training_pack_seeds = (
    set(
        ckpt["train_pack_seeds"]
    )
    |
    set(
        ckpt["val_pack_seeds"]
    )
)

canonical_overlap = (
    set(SEEDS)
    & training_pack_seeds
)

if canonical_overlap:
    raise RuntimeError(
        "Canonical evaluation seed leakage: "
        f"{sorted(canonical_overlap)}"
    )


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

gnn = PackGNN(
    node_feat=ckpt[
        "node_features"
    ],
    edge_feat=ckpt[
        "edge_features"
    ],
    hidden=ckpt[
        "hidden"
    ],
    n_layers=ckpt[
        "n_layers"
    ],
    action_feat=ckpt[
        "action_features"
    ],
).to(device)

gnn.load_state_dict(
    ckpt["model_state"],
    strict=True,
)

gnn.eval()


# ================================================================
# Controller factory
# ================================================================

def make_controller(name):

    if name == "CC-CV":
        return CCCVController(
            cfg
        )

    if name == "CC-CV-Balance":
        return BalancedCCCVController(
            cfg
        )

    if name == "Proportional":
        return ProportionalController(
            cfg
        )

    if name == "Physics-CEM":
        return PhysicsCEMController(
            cfg
        )

    if name == "GraphOptimizer":
        return CanonicalGraphOptimizer(
            cfg,
            gnn,
        )

    raise KeyError(name)


# ================================================================
# One canonical episode
# ================================================================

def run_episode(
    pack,
    controller_name,
    seed,
):

    # Same random stream initialization
    # for every controller at a given pack seed.
    #
    # This is especially important for the matched
    # Physics-CEM vs GraphOptimizer comparison.
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

    controller.reset()

    initial_soc = np.asarray(
        [
            c.SOC
            for c in pack.cells
        ],
        dtype=float,
    )

    initial_T = np.asarray(
        [
            c.T_C
            for c in pack.cells
        ],
        dtype=float,
    )

    initial_q = np.asarray(
        [
            c.Q_nom_Ah
            for c in pack.cells
        ],
        dtype=float,
    )

    n = pack.n_cells

    charge_Ah_cell = np.zeros(
        n,
        dtype=float,
    )

    energy_Wh_cell = np.zeros(
        n,
        dtype=float,
    )

    violations = 0

    peak_voltage = -np.inf
    min_voltage = np.inf

    peak_temperature = -np.inf
    max_temperature_gradient = 0.0

    cumulative_aging = 0.0

    current_sum_integral_Ah = 0.0

    steps = 0

    t0 = time.time()

    for step in range(
        cfg["max_steps"]
    ):

        soc_now = np.asarray(
            [
                c.SOC
                for c in pack.cells
            ],
            dtype=float,
        )

        if (
            float(
                np.mean(soc_now)
            )
            >= cfg["target_soc"]
        ):
            break

        currents = np.asarray(
            controller.get_currents(
                pack
            ),
            dtype=float,
        )

        # Defensive common actuator audit.
        projected = (
            project_current_vector(
                currents,
                pack,
                cfg,
            )
            .astype(float)
        )

        if not np.allclose(
            currents,
            projected,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"{controller_name} returned "
                "an action outside the common "
                "actuator feasible set."
            )

        metrics = pack.step(
            currents,
            dt=cfg["dt_s"],
        )

        steps += 1

        voltages = np.asarray(
            metrics["V_term"],
            dtype=float,
        )

        temps = np.asarray(
            metrics["T_C"],
            dtype=float,
        )

        violations += int(
            metrics[
                "n_violations"
            ]
        )

        peak_voltage = max(
            peak_voltage,
            float(
                np.max(
                    voltages
                )
            ),
        )

        min_voltage = min(
            min_voltage,
            float(
                np.min(
                    voltages
                )
            ),
        )

        peak_temperature = max(
            peak_temperature,
            float(
                np.max(
                    temps
                )
            ),
        )

        max_temperature_gradient = max(
            max_temperature_gradient,
            float(
                np.max(temps)
                - np.min(temps)
            ),
        )

        cumulative_aging += float(
            metrics[
                "aging_cost"
            ]
        )

        dt_h = (
            cfg["dt_s"]
            / 3600.0
        )

        charge_Ah_cell += (
            currents
            * dt_h
        )

        # Energy delivered to each individual
        # cell during the simulated step.
        energy_Wh_cell += (
            currents
            * voltages
            * dt_h
        )

        current_sum_integral_Ah += (
            float(
                np.sum(currents)
            )
            * dt_h
        )

    wall = (
        time.time()
        - t0
    )

    final_soc = np.asarray(
        [
            c.SOC
            for c in pack.cells
        ],
        dtype=float,
    )

    final_T = np.asarray(
        [
            c.T_C
            for c in pack.cells
        ],
        dtype=float,
    )

    target_reached = bool(
        float(
            np.mean(final_soc)
        )
        >= cfg["target_soc"]
    )

    if steps == 0:
        peak_voltage = float(
            max(
                c.V_term
                for c in pack.cells
            )
        )

        min_voltage = float(
            min(
                c.V_term
                for c in pack.cells
            )
        )

        peak_temperature = float(
            np.max(
                final_T
            )
        )

    filter_calls = (
        int(
            getattr(
                controller,
                "filter_calls",
                0,
            )
        )
    )

    filter_activations = (
        int(
            getattr(
                controller,
                "filter_activations",
                0,
            )
        )
    )

    filter_scales = list(
        getattr(
            controller,
            "filter_scales",
            [],
        )
    )

    row = {
        "seed":
            int(seed),

        "controller":
            controller_name,

        "control_seed":
            int(control_seed),

        "steps":
            int(steps),

        "charging_time_min":
            float(
                steps
                * cfg["dt_s"]
                / 60.0
            ),

        "target_reached":
            target_reached,

        "initial_SOC_mean":
            float(
                np.mean(initial_soc)
            ),

        "initial_SOC_sigma":
            float(
                np.std(initial_soc)
            ),

        "initial_SOC_sigma_pct":
            float(
                np.std(initial_soc)
                * 100.0
            ),

        "initial_T_mean_C":
            float(
                np.mean(initial_T)
            ),

        "initial_T_gradient_C":
            float(
                np.max(initial_T)
                - np.min(initial_T)
            ),

        "initial_Q_mean_Ah":
            float(
                np.mean(initial_q)
            ),

        "initial_Q_sigma_Ah":
            float(
                np.std(initial_q)
            ),

        "final_SOC_mean":
            float(
                np.mean(final_soc)
            ),

        "final_SOC_sigma":
            float(
                np.std(final_soc)
            ),

        "final_SOC_sigma_pct":
            float(
                np.std(final_soc)
                * 100.0
            ),

        "final_T_max_C":
            float(
                np.max(final_T)
            ),

        "final_T_gradient_C":
            float(
                np.max(final_T)
                - np.min(final_T)
            ),

        "peak_T_C":
            float(
                peak_temperature
            ),

        "max_T_gradient_C":
            float(
                max_temperature_gradient
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

        "pack_charge_throughput_Ah":
            float(
                current_sum_integral_Ah
            ),

        "per_cell_charge_Ah":
            [
                float(x)
                for x in charge_Ah_cell
            ],

        "per_cell_energy_Wh":
            [
                float(x)
                for x in energy_Wh_cell
            ],

        "pack_energy_Wh":
            float(
                np.sum(
                    energy_Wh_cell
                )
            ),

        "wall_time_s":
            float(wall),

        "safety_filter_calls":
            filter_calls,

        "safety_filter_activations":
            filter_activations,

        "minimum_safety_filter_scale":
            float(
                min(
                    filter_scales
                )
                if filter_scales
                else 1.0
            ),
    }

    return row


# ================================================================
# Incremental summary
# ================================================================

METRICS = [
    "charging_time_min",
    "final_SOC_mean",
    "final_SOC_sigma_pct",
    "peak_T_C",
    "final_T_gradient_C",
    "max_T_gradient_C",
    "peak_voltage_V",
    "total_violations",
    "cumulative_aging",
    "pack_energy_Wh",
    "wall_time_s",
]


def metric_stats(values):
    x = np.asarray(
        values,
        dtype=float,
    )

    return {
        "n":
            int(len(x)),

        "mean":
            float(
                np.mean(x)
            ),

        "std":
            float(
                np.std(
                    x,
                    ddof=1,
                )
            )
            if len(x) > 1
            else 0.0,

        "min":
            float(
                np.min(x)
            ),

        "max":
            float(
                np.max(x)
            ),
    }


def write_summary(rows):

    summary = {}

    for name in CONTROLLERS:

        subset = [
            r
            for r in rows
            if r["controller"]
            == name
        ]

        if not subset:
            continue

        block = {
            "n":
                len(subset),

            "target_successes":
                int(
                    sum(
                        bool(
                            r[
                                "target_reached"
                            ]
                        )
                        for r in subset
                    )
                ),

            "zero_violation_episodes":
                int(
                    sum(
                        int(
                            r[
                                "total_violations"
                            ]
                        )
                        == 0
                        for r in subset
                    )
                ),
        }

        for metric in METRICS:
            block[metric] = (
                metric_stats(
                    [
                        r[metric]
                        for r in subset
                    ]
                )
            )

        block[
            "total_safety_filter_activations"
        ] = int(
            sum(
                r[
                    "safety_filter_activations"
                ]
                for r in subset
            )
        )

        summary[name] = block

    payload = {
        "git_commit":
            git_head(),

        "checkpoint_sha256":
            checkpoint_sha,

        "n_expected_seeds":
            len(SEEDS),

        "controllers":
            CONTROLLERS,

        "summary":
            summary,
    }

    tmp = (
        SUMMARY_JSON
        .with_suffix(
            ".json.tmp"
        )
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n"
    )

    os.replace(
        tmp,
        SUMMARY_JSON,
    )

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "controller",
            "n",
            "target_successes",
            "zero_violation_episodes",
            "time_mean_min",
            "time_std_min",
            "sigma_SOC_mean_pct",
            "sigma_SOC_std_pct",
            "peak_T_mean_C",
            "final_dT_mean_C",
            "peak_voltage_max_V",
            "violations_mean",
            "aging_mean",
            "energy_mean_Wh",
            "filter_activations",
        ])

        for name in CONTROLLERS:

            if name not in summary:
                continue

            s = summary[name]

            w.writerow([
                name,
                s["n"],
                s[
                    "target_successes"
                ],
                s[
                    "zero_violation_episodes"
                ],
                s[
                    "charging_time_min"
                ]["mean"],
                s[
                    "charging_time_min"
                ]["std"],
                s[
                    "final_SOC_sigma_pct"
                ]["mean"],
                s[
                    "final_SOC_sigma_pct"
                ]["std"],
                s[
                    "peak_T_C"
                ]["mean"],
                s[
                    "final_T_gradient_C"
                ]["mean"],
                s[
                    "peak_voltage_V"
                ]["max"],
                s[
                    "total_violations"
                ]["mean"],
                s[
                    "cumulative_aging"
                ]["mean"],
                s[
                    "pack_energy_Wh"
                ]["mean"],
                s[
                    "total_safety_filter_activations"
                ],
            ])

    return payload


# ================================================================
# Preflight provenance
# ================================================================

provenance = {
    "git_commit":
        git_head(),

    "checkpoint":
        str(CKPT),

    "checkpoint_sha256":
        checkpoint_sha,

    "ecm":
        str(ECM),

    "ecm_sha256":
        sha256(
            ECM
        ),

    "device":
        str(device),

    "gpu":
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),

    "model_format_version":
        ckpt[
            "model_format_version"
        ],

    "action_conditioned":
        ckpt[
            "action_conditioned"
        ],

    "canonical_seeds":
        SEEDS,

    "training_validation_seed_overlap":
        len(
            canonical_overlap
        ),

    "controllers":
        CONTROLLERS,

    "config": {
        k: v
        for k, v in cfg.items()
        if isinstance(
            v,
            (
                str,
                int,
                float,
                bool,
                type(None),
            ),
        )
    },

    "protocol":
        {
            "paired_initial_pack":
                True,

            "control_rng":
                (
                    "np.random and torch reset "
                    "to 10,000,000 + pack seed "
                    "before every controller episode"
                ),

            "target_rule":
                "mean SOC >= 0.80",

            "voltage_window_V":
                [
                    2.0,
                    3.5,
                ],

            "per_cell_energy_recorded":
                True,

            "incremental_jsonl":
                True,

            "resume_supported":
                True,
        },
}

PROVENANCE_JSON.write_text(
    json.dumps(
        provenance,
        indent=2,
    )
    + "\n"
)


# ================================================================
# Run
# ================================================================

existing = load_existing()

completed = {
    (
        int(r["seed"]),
        r["controller"],
    )
    for r in existing
}

print("=" * 80)
print("CANONICAL LFP N=30 PAIRED EVALUATION")
print("=" * 80)

print(
    "git:",
    git_head(),
)

print(
    "checkpoint SHA:",
    checkpoint_sha,
)

print(
    "ECM SHA:",
    provenance[
        "ecm_sha256"
    ],
)

print(
    "device:",
    device,
)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

print(
    "seeds:",
    SEEDS,
)

print(
    "controllers:",
    CONTROLLERS,
)

print(
    "already completed:",
    len(completed),
    "/",
    len(SEEDS)
    * len(CONTROLLERS),
)

print(
    "training/evaluation seed overlap:",
    len(canonical_overlap),
)

print()


for seed_idx, seed in enumerate(
    SEEDS,
    1,
):

    # Generate ONE canonical initial pack for the seed.
    base_pack = build_pack_from_ecm(
        ecm_parquet=ECM,
        n_cells=cfg["n_cells"],
        chemistry=cfg["chemistry"],
        T_amb=cfg["T_amb"],
        soc_init=cfg["soc_init"],
        soc_noise=cfg["soc_noise"],
        seed=seed,
    )

    for cell in base_pack.cells:
        cell.V_min_limit = float(
            cfg["V_min"]
        )

        cell.V_max_limit = float(
            cfg["V_max"]
        )

    pack_fp = fingerprint_pack(
        base_pack
    )

    print(
        f"===== SEED "
        f"{seed_idx:02d}/30 "
        f"| seed={seed} "
        f"| pack={pack_fp[:12]} "
        f"=====",
        flush=True,
    )

    # If this seed was partly completed before,
    # verify the deterministic pack fingerprint.
    previous_this_seed = [
        r
        for r in existing
        if int(
            r["seed"]
        ) == seed
    ]

    for old in previous_this_seed:
        if (
            old.get(
                "initial_pack_sha256"
            )
            != pack_fp
        ):
            raise RuntimeError(
                f"Pack fingerprint mismatch "
                f"while resuming seed {seed}"
            )

    for controller_name in CONTROLLERS:

        key = (
            seed,
            controller_name,
        )

        if key in completed:
            print(
                f"  SKIP "
                f"{controller_name:<18} "
                "(already complete)",
                flush=True,
            )
            continue

        pack = deepcopy(
            base_pack
        )

        print(
            f"  RUN  "
            f"{controller_name:<18}",
            end=" ",
            flush=True,
        )

        row = run_episode(
            pack,
            controller_name,
            seed,
        )

        row[
            "initial_pack_sha256"
        ] = pack_fp

        append_jsonl(
            row
        )

        existing.append(
            row
        )

        completed.add(
            key
        )

        write_summary(
            existing
        )

        print(
            f"| time="
            f"{row['charging_time_min']:.1f}m "
            f"| SOC="
            f"{row['final_SOC_mean']:.5f} "
            f"| sigma="
            f"{row['final_SOC_sigma_pct']:.4f}% "
            f"| Tmax="
            f"{row['peak_T_C']:.3f}C "
            f"| dT="
            f"{row['final_T_gradient_C']:.4f}C "
            f"| Vpk="
            f"{row['peak_voltage_V']:.6f} "
            f"| viol="
            f"{row['total_violations']} "
            f"| target="
            f"{row['target_reached']} "
            f"| wall="
            f"{row['wall_time_s']:.1f}s",
            flush=True,
        )

    print(
        f"  completed total: "
        f"{len(completed)}/150",
        flush=True,
    )

    print()


# ================================================================
# Final validation
# ================================================================

final_rows = load_existing()

keys = [
    (
        int(r["seed"]),
        r["controller"],
    )
    for r in final_rows
]

if len(keys) != len(set(keys)):
    raise RuntimeError(
        "Duplicate seed/controller rows "
        "detected in canonical JSONL."
    )

expected = {
    (
        seed,
        controller,
    )
    for seed in SEEDS
    for controller in CONTROLLERS
}

actual = set(keys)

missing = (
    expected - actual
)

extra = (
    actual - expected
)

if missing:
    raise RuntimeError(
        f"Canonical run incomplete: "
        f"{len(missing)} rows missing."
    )

if extra:
    raise RuntimeError(
        f"Unexpected rows: {extra}"
    )


# Every controller must see the exact same
# initial pack fingerprint for a given seed.
for seed in SEEDS:

    fps = {
        r[
            "initial_pack_sha256"
        ]
        for r in final_rows
        if int(
            r["seed"]
        ) == seed
    }

    if len(fps) != 1:
        raise RuntimeError(
            f"Paired-pack failure "
            f"for seed {seed}: {fps}"
        )


summary_payload = write_summary(
    final_rows
)

print("=" * 80)
print("FINAL CANONICAL SUMMARY")
print("=" * 80)

for name in CONTROLLERS:

    s = (
        summary_payload[
            "summary"
        ][name]
    )

    print(
        f"{name:18s} "
        f"N={s['n']:2d} | "
        f"time="
        f"{s['charging_time_min']['mean']:.3f}"
        f"±"
        f"{s['charging_time_min']['std']:.3f} min | "
        f"sigmaSOC="
        f"{s['final_SOC_sigma_pct']['mean']:.4f}"
        f"±"
        f"{s['final_SOC_sigma_pct']['std']:.4f}% | "
        f"Tmax="
        f"{s['peak_T_C']['mean']:.3f} C | "
        f"dT="
        f"{s['final_T_gradient_C']['mean']:.4f} C | "
        f"Vpeak(max)="
        f"{s['peak_voltage_V']['max']:.6f} V | "
        f"viol="
        f"{s['total_violations']['mean']:.3f}/ep | "
        f"target="
        f"{s['target_successes']}/30"
    )

print()

print(
    "raw:",
    JSONL,
)

print(
    "summary JSON:",
    SUMMARY_JSON,
)

print(
    "summary CSV:",
    SUMMARY_CSV,
)

print(
    "provenance:",
    PROVENANCE_JSON,
)

print()

print(
    "CANONICAL LFP N=30 "
    "PAIRED EVALUATION: COMPLETE"
)
