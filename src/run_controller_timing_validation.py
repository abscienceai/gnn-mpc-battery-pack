#!/usr/bin/env python3

"""
Software controller timing / pseudo-HIL deadline validation.

This is NOT hardware-in-the-loop validation.

It measures end-to-end software decision latency for:
  1. matched Physics-CEM
  2. GraphOptimizer

Both controllers are evaluated on exactly the same 30 dataset-backed
LFP pack states spanning the operating SOC range.

Reported latency includes:
  controller.get_currents(...)
  + common actuator projection

The manuscript claim is limited to deadline feasibility on the reported
compute platform for the study's 60 s control interval.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
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

OUT = (
    ROOT
    / "results/diagnostics/"
      "controller_timing_validation"
)

PROVENANCE = Path(
    "/home/msoylu/alper/Graph-Guided/"
    "provenance/"
    "controller_timing_validation.json"
)

EXPECTED_ECM_SHA = (
    "5580ff78d2518c645588d5bdd58ace965"
    "07e14bb2622dd6e26cf080397d94988"
)

EXPECTED_CKPT_SHA = (
    "2f3acc96e77504be0a060f5bee9bb226"
    "3b62df08c8fcd2afca353005c02e0b5f"
)

CONTROL_DEADLINE_S = 60.0


def sha256(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def cfg_for_timing():
    cfg = default_config()

    cfg["n_cells"] = 12
    cfg["chemistry"] = "LFP"

    cfg["target_soc"] = 0.80
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

    assert (
        ckpt["model_format_version"]
        == 2
    )

    assert ckpt[
        "action_conditioned"
    ]

    model = PackGNN(
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

    model.load_state_dict(
        ckpt["model_state"],
        strict=True,
    )

    model.eval()

    return model


def pack_signature(pack):
    rows = []

    for c in pack.cells:
        rows.append({
            "SOC": float(c.SOC),
            "T_C": float(c.T_C),
            "SOH": float(c.SOH),
            "R0": float(c.R0),
            "R1": float(c.R1),
            "C1": float(c.C1),
            "Q_nom_Ah":
                float(c.Q_nom_Ah),
        })

    return hashlib.sha256(
        json.dumps(
            rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def make_states():
    """
    30 paired states:
      6 nominal SOC levels x 5 independent pack seeds.
    """

    levels = [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
    ]

    states = []

    state_id = 0

    for level_idx, soc0 in enumerate(levels):

        for replicate in range(5):

            seed = (
                710001
                + level_idx * 100
                + replicate * 7
            )

            pack = build_pack_from_ecm(
                ecm_parquet=ECM,
                n_cells=12,
                chemistry="LFP",
                T_amb=25.0,
                soc_init=soc0,
                soc_noise=0.03,
                seed=seed,
                strict=True,
            )

            if (
                pack.ecm_source_mode
                != "dataset"
                or
                pack.ecm_dataset
                != "MATR"
            ):
                raise RuntimeError(
                    "Timing state is not "
                    "strict MATR-backed."
                )

            states.append({
                "state_id":
                    state_id,

                "nominal_SOC":
                    soc0,

                "seed":
                    seed,

                "pack":
                    pack,

                "signature":
                    pack_signature(pack),
            })

            state_id += 1

    assert len(states) == 30

    return states


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_decision(
    controller,
    pack,
    cfg,
):
    cuda_sync()

    t0 = time.perf_counter()

    currents = np.asarray(
        controller.get_currents(
            pack
        ),
        dtype=np.float64,
    )

    currents = np.asarray(
        project_current_vector(
            currents,
            pack,
            cfg,
        ),
        dtype=np.float64,
    )

    cuda_sync()

    elapsed = (
        time.perf_counter()
        - t0
    )

    if len(currents) != len(pack.cells):
        raise RuntimeError(
            "Controller action length mismatch."
        )

    if not np.all(
        np.isfinite(currents)
    ):
        raise RuntimeError(
            "Non-finite controller action."
        )

    return (
        elapsed,
        currents,
    )


def summarize(
    frame,
    controller_name,
):
    x = (
        frame[
            frame["controller"]
            == controller_name
        ]["latency_s"]
        .to_numpy(
            dtype=float
        )
    )

    return {
        "controller":
            controller_name,

        "n":
            int(len(x)),

        "mean_s":
            float(np.mean(x)),

        "median_s":
            float(np.median(x)),

        "p95_s":
            float(
                np.percentile(
                    x,
                    95,
                )
            ),

        "p99_s":
            float(
                np.percentile(
                    x,
                    99,
                )
            ),

        "max_s":
            float(np.max(x)),

        "deadline_s":
            CONTROL_DEADLINE_S,

        "deadline_misses":
            int(
                np.sum(
                    x
                    > CONTROL_DEADLINE_S
                )
            ),

        "deadline_success_pct":
            float(
                np.mean(
                    x
                    <= CONTROL_DEADLINE_S
                )
                * 100.0
            ),

        "p95_deadline_utilization_pct":
            float(
                np.percentile(
                    x,
                    95,
                )
                / CONTROL_DEADLINE_S
                * 100.0
            ),

        "max_deadline_utilization_pct":
            float(
                np.max(x)
                / CONTROL_DEADLINE_S
                * 100.0
            ),
    }


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROVENANCE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if sha256(ECM) != EXPECTED_ECM_SHA:
        raise RuntimeError(
            "ECM hash mismatch."
        )

    if sha256(CKPT) != EXPECTED_CKPT_SHA:
        raise RuntimeError(
            "Checkpoint hash mismatch."
        )

    cfg = cfg_for_timing()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = load_model(
        device
    )

    states = make_states()

    print("=" * 100)
    print(
        "SOFTWARE CONTROLLER TIMING VALIDATION"
    )
    print("=" * 100)

    print(
        "device:",
        device,
    )

    print(
        "states:",
        len(states),
    )

    print(
        "control deadline:",
        CONTROL_DEADLINE_S,
        "s",
    )

    print(
        "claim boundary: software timing "
        "emulation, NOT hardware HIL"
    )

    # Warm-up calls are excluded.
    warm_pack = deepcopy(
        states[0]["pack"]
    )

    print()
    print("===== WARM-UP =====")

    physics_warm = (
        PhysicsCEMController(cfg)
    )

    graph_warm = (
        GraphGuidedOptimizer(
            cfg,
            model,
        )
    )

    for i in range(3):

        p_t, _ = timed_decision(
            physics_warm,
            deepcopy(warm_pack),
            cfg,
        )

        g_t, _ = timed_decision(
            graph_warm,
            deepcopy(warm_pack),
            cfg,
        )

        print(
            f"warmup {i+1}: "
            f"Physics={p_t:.4f}s | "
            f"Graph={g_t:.4f}s"
        )

    rows = []

    print()
    print("===== PAIRED TIMING STATES =====")

    # Persistent controller objects mimic
    # repeated software control-loop usage.
    physics = PhysicsCEMController(
        cfg
    )

    graph = GraphGuidedOptimizer(
        cfg,
        model,
    )

    for s in states:

        pack_p = deepcopy(
            s["pack"]
        )

        pack_g = deepcopy(
            s["pack"]
        )

        p_t, p_I = timed_decision(
            physics,
            pack_p,
            cfg,
        )

        g_t, g_I = timed_decision(
            graph,
            pack_g,
            cfg,
        )

        for name, latency, action in [
            (
                "Physics-CEM",
                p_t,
                p_I,
            ),
            (
                "GraphOptimizer",
                g_t,
                g_I,
            ),
        ]:

            rows.append({
                "state_id":
                    s["state_id"],

                "nominal_SOC":
                    s["nominal_SOC"],

                "seed":
                    s["seed"],

                "initial_pack_sha256":
                    s["signature"],

                "controller":
                    name,

                "latency_s":
                    latency,

                "deadline_s":
                    CONTROL_DEADLINE_S,

                "deadline_met":
                    bool(
                        latency
                        <= CONTROL_DEADLINE_S
                    ),

                "mean_current_A":
                    float(
                        np.mean(action)
                    ),

                "max_current_A":
                    float(
                        np.max(action)
                    ),
            })

        print(
            f"state={s['state_id']:02d} | "
            f"SOC={s['nominal_SOC']:.2f} | "
            f"Physics={p_t:8.4f}s | "
            f"Graph={g_t:8.4f}s | "
            f"deadline="
            f"{CONTROL_DEADLINE_S:.0f}s"
        )

    df = pd.DataFrame(
        rows
    )

    assert len(df) == 60

    for state_id, sub in df.groupby(
        "state_id"
    ):

        assert len(sub) == 2

        assert (
            sub[
                "initial_pack_sha256"
            ].nunique()
            == 1
        )

    summaries = [
        summarize(
            df,
            "Physics-CEM",
        ),
        summarize(
            df,
            "GraphOptimizer",
        ),
    ]

    raw_csv = (
        OUT
        / "controller_timing_raw.csv"
    )

    summary_json = (
        OUT
        / "controller_timing_summary.json"
    )

    df.to_csv(
        raw_csv,
        index=False,
    )

    summary_json.write_text(
        json.dumps(
            summaries,
            indent=2,
        )
        + "\n"
    )

    gpu_info = None

    if torch.cuda.is_available():
        gpu_info = {
            "name":
                torch.cuda.get_device_name(
                    0
                ),

            "torch_cuda":
                torch.version.cuda,

            "allocated_GB":
                float(
                    torch.cuda.memory_allocated(
                        0
                    )
                    / 1e9
                ),

            "reserved_GB":
                float(
                    torch.cuda.memory_reserved(
                        0
                    )
                    / 1e9
                ),
        }

    cpu_model = None

    try:
        cpu_model = (
            subprocess.check_output(
                [
                    "bash",
                    "-lc",
                    "lscpu | grep 'Model name' | "
                    "head -1 | cut -d: -f2-",
                ],
                text=True,
            )
            .strip()
        )
    except Exception:
        pass

    provenance = {
        "description":
            (
                "Software controller timing / "
                "pseudo-HIL deadline feasibility test"
            ),

        "not_hardware_HIL":
            True,

        "control_interval_s":
            CONTROL_DEADLINE_S,

        "n_paired_states":
            30,

        "state_SOC_levels":
            [
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
            ],

        "git_head":
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            ).strip(),

        "checkpoint_sha256":
            sha256(CKPT),

        "ecm_sha256":
            sha256(ECM),

        "python":
            sys.version,

        "platform":
            platform.platform(),

        "cpu_model":
            cpu_model,

        "gpu":
            gpu_info,

        "timed_region":
            (
                "controller.get_currents plus "
                "common actuator projection, "
                "with CUDA synchronization before "
                "and after timing"
            ),

        "summary":
            summaries,

        "claim_boundary":
            (
                "Results demonstrate software deadline "
                "feasibility only on the reported compute "
                "platform for the 60 s study control interval; "
                "they do not constitute hardware-in-the-loop "
                "or embedded real-time validation."
            ),
    }

    PROVENANCE.write_text(
        json.dumps(
            provenance,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 100)
    print("TIMING SUMMARY")
    print("=" * 100)

    for s in summaries:

        print(
            f"{s['controller']:15s} | "
            f"N={s['n']:2d} | "
            f"median={s['median_s']:.4f}s | "
            f"p95={s['p95_s']:.4f}s | "
            f"max={s['max_s']:.4f}s | "
            f"miss={s['deadline_misses']} | "
            f"p95 budget="
            f"{s['p95_deadline_utilization_pct']:.2f}%"
        )

    print()
    print(
        "raw:",
        raw_csv,
    )

    print(
        "summary:",
        summary_json,
    )

    print(
        "provenance:",
        PROVENANCE,
    )

    print()
    print(
        "SOFTWARE TIMING VALIDATION: COMPLETE"
    )


if __name__ == "__main__":
    main()
