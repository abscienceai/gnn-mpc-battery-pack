#!/usr/bin/env python3

import argparse
from copy import deepcopy
from pathlib import Path
import hashlib
import json
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
    GraphGuidedOptimizer,
    default_config,
    project_current_vector,
)

from train_node_only_gnn import (
    NodeOnlyGNN,
)

from train_mlp_fair import (
    PackMLPV2,
)


ROOT = Path(__file__).resolve().parent.parent

GRAPH_CKPT = (
    ROOT
    / "results/models/"
      "pack_gnn_action_v2_20261003_232518.pt"
)

ECM = (
    ROOT
    / "results/ecm/"
      "ecm_params_20260630_202031.parquet"
)

EXPERIMENT_ROOT = (
    ROOT
    / "results/extended_evaluation"
)

MODEL_DIR = (
    EXPERIMENT_ROOT
    / "models"
)

OUT_DIR = (
    EXPERIMENT_ROOT
    / "surrogate_ablation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# Canonical configuration
# ================================================================

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


CANONICAL_SEEDS = [
    7 * i
    for i in range(30)
]


# ================================================================
# Helpers
# ================================================================

def sha256(path):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for block in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):
            h.update(
                block
            )

    return h.hexdigest()


def load_ckpt(path):
    return torch.load(
        path,
        map_location="cpu",
    )


def find_full_checkpoint(pattern):

    candidates = sorted(
        MODEL_DIR.glob(
            pattern
        )
    )

    valid = []

    for path in candidates:

        try:
            ckpt = load_ckpt(
                path
            )
        except Exception:
            continue

        train_ids = ckpt.get(
            "train_rollout_ids",
            [],
        )

        val_ids = ckpt.get(
            "val_rollout_ids",
            [],
        )

        if (
            len(train_ids) == 4250
            and len(val_ids) == 750
        ):
            valid.append(
                path
            )

    if not valid:
        raise FileNotFoundError(
            f"No full 5000-rollout checkpoint "
            f"found for pattern: {pattern}"
        )

    return valid[-1]


NODE_CKPT = find_full_checkpoint(
    "node_only_action_v2_seed42_*.pt"
)

MLP_CKPT = find_full_checkpoint(
    "pack_mlp_action_v2_seed42_*.pt"
)


# ================================================================
# Instrumented common controller
# ================================================================

class AblationOptimizer(
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
            np.sum(
                proposed
            )
        )

        numer = float(
            np.sum(
                filtered
            )
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
# Load three pretrained surrogate models
# ================================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


graph_ckpt = load_ckpt(
    GRAPH_CKPT
)

if (
    graph_ckpt.get(
        "model_format_version"
    )
    != 2
):
    raise RuntimeError(
        "PackGNN checkpoint is not v2."
    )

if not graph_ckpt.get(
    "action_conditioned",
    False,
):
    raise RuntimeError(
        "PackGNN checkpoint is not "
        "action-conditioned."
    )


graph_model = PackGNN(
    node_feat=
        graph_ckpt["node_features"],
    edge_feat=
        graph_ckpt["edge_features"],
    hidden=
        graph_ckpt["hidden"],
    n_layers=
        graph_ckpt["n_layers"],
    action_feat=
        graph_ckpt["action_features"],
).to(device)

graph_model.load_state_dict(
    graph_ckpt["model_state"],
    strict=True,
)

graph_model.eval()


node_ckpt = load_ckpt(
    NODE_CKPT
)

node_model = NodeOnlyGNN(
    node_feat=
        node_ckpt.get(
            "node_features",
            7,
        ),
    hidden=
        node_ckpt.get(
            "hidden",
            64,
        ),
    n_layers=
        node_ckpt.get(
            "n_layers",
            3,
        ),
    action_feat=
        node_ckpt.get(
            "action_features",
            1,
        ),
    expansion=
        node_ckpt.get(
            "expansion",
            192,
        ),
).to(device)

node_model.load_state_dict(
    node_ckpt["model_state"],
    strict=True,
)

node_model.eval()


mlp_ckpt = load_ckpt(
    MLP_CKPT
)

mlp_model = PackMLPV2(
    n_cells=
        mlp_ckpt.get(
            "n_cells",
            12,
        ),
    node_feat=
        mlp_ckpt.get(
            "node_features",
            7,
        ),
    action_feat=
        mlp_ckpt.get(
            "action_features",
            1,
        ),
    hidden=
        mlp_ckpt.get(
            "hidden",
            212,
        ),
    hidden2=
        mlp_ckpt.get(
            "hidden2",
            73,
        ),
).to(device)

mlp_model.load_state_dict(
    mlp_ckpt["model_state"],
    strict=True,
)

mlp_model.eval()


MODELS = {
    "PackGNN":
        graph_model,

    "NodeOnly":
        node_model,

    "FlatMLP":
        mlp_model,
}


CHECKPOINTS = {
    "PackGNN":
        GRAPH_CKPT,

    "NodeOnly":
        NODE_CKPT,

    "FlatMLP":
        MLP_CKPT,
}


CKPT_DATA = {
    "PackGNN":
        graph_ckpt,

    "NodeOnly":
        node_ckpt,

    "FlatMLP":
        mlp_ckpt,
}


# ================================================================
# Leakage audit
# ================================================================

for name, ckpt in CKPT_DATA.items():

    training_pack_seeds = (
        set(
            ckpt.get(
                "train_pack_seeds",
                [],
            )
        )
        |
        set(
            ckpt.get(
                "val_pack_seeds",
                [],
            )
        )
    )

    overlap = (
        set(
            CANONICAL_SEEDS
        )
        & training_pack_seeds
    )

    if overlap:
        raise RuntimeError(
            f"{name} has canonical "
            f"evaluation seed leakage: "
            f"{sorted(overlap)}"
        )


# ================================================================
# Controller factory
# ================================================================

def make_controller(
    model_name,
):

    return AblationOptimizer(
        cfg,
        MODELS[
            model_name
        ],
    )


# ================================================================
# Canonical episode
# ================================================================

def run_episode(
    pack,
    model_name,
    seed,
):

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
        model_name
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
                np.mean(
                    soc_now
                )
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
            dtype=float,
        )

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
                f"{model_name} returned "
                "an action outside the "
                "common actuator feasible set."
            )

        metrics = pack.step(
            currents,
            dt=cfg[
                "dt_s"
            ],
        )

        steps += 1

        voltages = np.asarray(
            metrics[
                "V_term"
            ],
            dtype=float,
        )

        temps = np.asarray(
            metrics[
                "T_C"
            ],
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
                np.max(
                    temps
                )
                -
                np.min(
                    temps
                )
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

        energy_Wh_cell += (
            currents
            * voltages
            * dt_h
        )

        current_sum_integral_Ah += (
            float(
                np.sum(
                    currents
                )
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
            np.mean(
                final_soc
            )
        )
        >= cfg[
            "target_soc"
        ]
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

    filter_calls = int(
        getattr(
            controller,
            "filter_calls",
            0,
        )
    )

    filter_activations = int(
        getattr(
            controller,
            "filter_activations",
            0,
        )
    )

    filter_scales = list(
        getattr(
            controller,
            "filter_scales",
            [],
        )
    )

    return {
        "seed":
            int(seed),

        "model":
            model_name,

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
                np.mean(
                    initial_soc
                )
            ),

        "initial_SOC_sigma_pct":
            float(
                np.std(
                    initial_soc
                )
                * 100.0
            ),

        "initial_T_mean_C":
            float(
                np.mean(
                    initial_T
                )
            ),

        "initial_T_gradient_C":
            float(
                np.max(
                    initial_T
                )
                -
                np.min(
                    initial_T
                )
            ),

        "initial_Q_mean_Ah":
            float(
                np.mean(
                    initial_q
                )
            ),

        "initial_Q_sigma_Ah":
            float(
                np.std(
                    initial_q
                )
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

        "final_T_max_C":
            float(
                np.max(
                    final_T
                )
            ),

        "final_T_gradient_C":
            float(
                np.max(
                    final_T
                )
                -
                np.min(
                    final_T
                )
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
                for x
                in charge_Ah_cell
            ],

        "per_cell_energy_Wh":
            [
                float(x)
                for x
                in energy_Wh_cell
            ],

        "pack_energy_Wh":
            float(
                np.sum(
                    energy_Wh_cell
                )
            ),

        "wall_time_s":
            float(
                wall
            ),

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


# ================================================================
# Output helpers
# ================================================================

KEY_METRICS = [
    "charging_time_min",
    "final_SOC_sigma_pct",
    "final_T_gradient_C",
    "peak_T_C",
    "total_violations",
    "cumulative_aging",
    "pack_energy_Wh",
    "wall_time_s",
]


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

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(
                        line
                    )
                )

    return rows


def print_summary(
    rows,
):

    print()
    print(
        "=" * 90
    )
    print(
        "SURROGATE ABLATION SUMMARY"
    )
    print(
        "=" * 90
    )

    for name in MODELS:

        rr = [
            r
            for r in rows
            if r["model"]
            == name
        ]

        if not rr:
            continue

        print()
        print(
            f"{name} | N={len(rr)}"
        )

        print(
            "  target success:"
            f" {sum(r['target_reached'] for r in rr)}"
            f"/{len(rr)}"
        )

        for metric in KEY_METRICS:

            x = np.asarray(
                [
                    r[metric]
                    for r in rr
                ],
                dtype=float,
            )

            print(
                f"  {metric:<26}"
                f" mean={np.mean(x):.8g}"
                f" std={np.std(x, ddof=1) if len(x) > 1 else 0.0:.8g}"
            )


# ================================================================
# Main
# ================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    if (
        args.n_seeds < 1
        or args.n_seeds > 30
    ):
        raise ValueError(
            "--n-seeds must be between 1 and 30"
        )

    seeds = CANONICAL_SEEDS[
        :args.n_seeds
    ]

    out_jsonl = (
        OUT_DIR
        / f"surrogate_ablation_n{args.n_seeds}.jsonl"
    )

    existing = read_jsonl(
        out_jsonl
    )

    completed = {
        (
            int(r["seed"]),
            r["model"],
        )
        for r in existing
    }

    print(
        f"Device: {device}"
    )

    print(
        f"ECM: {ECM}"
    )

    print()

    for name, path in CHECKPOINTS.items():

        print(
            f"{name:<10} "
            f"{path.name}"
        )

        print(
            f"           SHA256="
            f"{sha256(path)}"
        )

        print(
            f"           params="
            f"{sum(p.numel() for p in MODELS[name].parameters()):,}"
        )

    print()

    print(
        "Canonical controller protocol:"
    )

    print(
        f"  H={cfg['horizon']}, "
        f"K={cfg['cem_samples']}, "
        f"elite={cfg['cem_elite_frac']}, "
        f"iterations={cfg['cem_iterations']}"
    )

    print(
        f"  seeds={seeds}"
    )

    print(
        "  training/evaluation leakage: NONE"
    )

    print()

    for seed_idx, seed in enumerate(
        seeds,
        1,
    ):

        base_pack = build_pack_from_ecm(
            ecm_parquet=ECM,
            n_cells=
                cfg["n_cells"],
            chemistry=
                cfg["chemistry"],
            T_amb=
                cfg["T_amb"],
            soc_init=
                cfg["soc_init"],
            soc_noise=
                cfg["soc_noise"],
            seed=seed,
            strict=True,
        )

        for cell in base_pack.cells:

            cell.V_min_limit = float(
                cfg["V_min"]
            )

            cell.V_max_limit = float(
                cfg["V_max"]
            )

        print(
            f"===== SEED "
            f"{seed_idx:02d}/{len(seeds)} "
            f"| seed={seed} =====",
            flush=True,
        )

        for model_name in MODELS:

            key = (
                seed,
                model_name,
            )

            if key in completed:

                print(
                    f"  SKIP {model_name:<10} "
                    "(already complete)",
                    flush=True,
                )

                continue

            pack = deepcopy(
                base_pack
            )

            print(
                f"  RUN  {model_name:<10}",
                end=" ",
                flush=True,
            )

            row = run_episode(
                pack,
                model_name,
                seed,
            )

            append_jsonl(
                out_jsonl,
                row,
            )

            existing.append(
                row
            )

            completed.add(
                key
            )

            print(
                f"time={row['charging_time_min']:.3f} min | "
                f"SOCsig={row['final_SOC_sigma_pct']:.4f}% | "
                f"dT={row['final_T_gradient_C']:.4f} C | "
                f"peakT={row['peak_T_C']:.4f} C | "
                f"viol={row['total_violations']} | "
                f"wall={row['wall_time_s']:.2f}s",
                flush=True,
            )

    print_summary(
        existing
    )

    summary_path = (
        OUT_DIR
        / f"surrogate_ablation_n{args.n_seeds}_summary.json"
    )

    summary = {
        "n_seeds":
            args.n_seeds,

        "seeds":
            seeds,

        "configuration": {
            "horizon":
                cfg["horizon"],
            "cem_samples":
                cfg["cem_samples"],
            "cem_elite_frac":
                cfg["cem_elite_frac"],
            "cem_iterations":
                cfg["cem_iterations"],
            "target_soc":
                cfg["target_soc"],
            "I_max_C":
                cfg["I_max_C"],
            "V_min":
                cfg["V_min"],
            "V_max":
                cfg["V_max"],
            "T_max":
                cfg["T_max"],
        },

        "checkpoints": {
            name: {
                "path":
                    str(path),
                "sha256":
                    sha256(path),
                "params":
                    int(
                        sum(
                            p.numel()
                            for p
                            in MODELS[name].parameters()
                        )
                    ),
            }
            for name, path
            in CHECKPOINTS.items()
        },

        "rows":
            existing,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            sort_keys=True,
        )

    print()
    print(
        f"Raw results -> {out_jsonl}"
    )

    print(
        f"Summary -> {summary_path}"
    )


if __name__ == "__main__":
    main()
