#!/usr/bin/env python3

from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_battery_pack import build_pack_from_ecm
from safe_fast_charge_optimizer import (
    default_config,
    SimpleMPCController,
)
from submission_revision_controllers import (
    apply_dataset_voltage_limits,
)

ECM = (
    ROOT
    / "results"
    / "ecm"
    / "ecm_params_20260630_202031.parquet"
)

OUT = (
    ROOT
    / "results"
    / "submission_revision"
    / "matched_physics_cem_smoke"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

    cfg = default_config()

    cfg["n_cells"] = 12
    cfg["chemistry"] = "LFP"
    cfg["soc_init"] = 0.20
    cfg["soc_noise"] = 0.03
    cfg["target_soc"] = 0.80

    cfg = apply_dataset_voltage_limits(
        cfg,
        repo_root=ROOT,
    )

    print("=" * 78)
    print("MATCHED DIRECT-PHYSICS CEM FAIRNESS TEST")
    print("=" * 78)

    print(
        "Config:",
        f"H={cfg['horizon']}",
        f"K={cfg['cem_samples']}",
        f"elite_frac={cfg['cem_elite_frac']}",
        f"iterations={cfg['cem_iterations']}",
    )

    expected_elite = int(
        cfg["cem_samples"]
        * cfg["cem_elite_frac"]
    )

    # Submission protocol gates.
    assert cfg["horizon"] == 5, (
        f"Expected H=5, got {cfg['horizon']}"
    )

    assert cfg["cem_samples"] == 64, (
        f"Expected K=64, got {cfg['cem_samples']}"
    )

    assert expected_elite == 16, (
        f"Expected elite=16, got {expected_elite}"
    )

    assert cfg["cem_iterations"] == 5, (
        f"Expected iterations=5, got "
        f"{cfg['cem_iterations']}"
    )

    rows = []

    for seed in [0, 7, 14]:

        np.random.seed(seed)

        pack = build_pack_from_ecm(
            ECM,
            n_cells=cfg["n_cells"],
            chemistry=cfg["chemistry"],
            T_amb=cfg["T_amb"],
            soc_init=cfg["soc_init"],
            soc_noise=cfg["soc_noise"],
            seed=seed,
        )

        for cell in pack.cells:
            cell.V_min_limit = cfg["V_min"]
            cell.V_max_limit = cfg["V_max"]

        ctrl = SimpleMPCController(cfg)

        # -------------------------------------------------------------
        # First call: cold-start CEM
        # -------------------------------------------------------------
        action1 = ctrl.get_currents(pack)

        expected_candidates = (
            cfg["cem_samples"]
            * cfg["cem_iterations"]
        )

        expected_physics_steps = (
            expected_candidates
            * cfg["horizon"]
        )

        row1 = {
            "seed": seed,
            "call": 1,
            "warm_start": ctrl.last_used_warm_start,
            "candidate_evaluations":
                ctrl.last_candidate_evaluations,
            "physics_steps":
                ctrl.last_physics_steps,
            "elite_k":
                ctrl.last_elite_k,
            "iterations":
                ctrl.last_iterations,
            "horizon":
                ctrl.last_horizon,
            "best_cost":
                ctrl.last_best_cost,
            "action_sum_A":
                float(action1.sum()),
            "action_min_A":
                float(action1.min()),
            "action_max_A":
                float(action1.max()),
        }

        rows.append(row1)

        q = np.asarray(
            [c.Q_nom_Ah for c in pack.cells],
            dtype=float,
        )

        per_cell_max = (
            cfg["I_max_C"]
            * q
        )

        pack_max = (
            cfg["I_pack_max_C"]
            * q.sum()
        )

        assert not ctrl.last_used_warm_start
        assert ctrl.last_candidate_evaluations == expected_candidates
        assert ctrl.last_physics_steps == expected_physics_steps
        assert ctrl.last_elite_k == 16
        assert ctrl.last_iterations == 5
        assert ctrl.last_horizon == 5

        assert np.all(
            action1 >= -1e-7
        )

        assert np.all(
            action1
            <= per_cell_max + 1e-5
        )

        assert float(action1.sum()) <= (
            pack_max + 1e-5
        )

        # Advance actual pack by one control step.
        metrics = pack.step(
            action1,
            dt=cfg["dt_s"],
        )

        # -------------------------------------------------------------
        # Second call: must warm-start from previous elite distribution
        # -------------------------------------------------------------
        np.random.seed(seed + 1000)

        action2 = ctrl.get_currents(pack)

        row2 = {
            "seed": seed,
            "call": 2,
            "warm_start": ctrl.last_used_warm_start,
            "candidate_evaluations":
                ctrl.last_candidate_evaluations,
            "physics_steps":
                ctrl.last_physics_steps,
            "elite_k":
                ctrl.last_elite_k,
            "iterations":
                ctrl.last_iterations,
            "horizon":
                ctrl.last_horizon,
            "best_cost":
                ctrl.last_best_cost,
            "action_sum_A":
                float(action2.sum()),
            "action_min_A":
                float(action2.min()),
            "action_max_A":
                float(action2.max()),
            "post_step_SOC":
                float(metrics["SOC_mean"]),
        }

        rows.append(row2)

        assert ctrl.last_used_warm_start
        assert ctrl.last_candidate_evaluations == expected_candidates
        assert ctrl.last_physics_steps == expected_physics_steps
        assert ctrl.last_elite_k == 16
        assert ctrl.last_iterations == 5
        assert ctrl.last_horizon == 5

        assert np.all(
            action2 >= -1e-7
        )

        assert np.all(
            action2
            <= per_cell_max + 1e-5
        )

        assert float(action2.sum()) <= (
            pack_max + 1e-5
        )

        print(
            f"seed={seed:2d} | "
            f"K={ctrl.n_samples} "
            f"elite={ctrl.last_elite_k} "
            f"iter={ctrl.last_iterations} "
            f"H={ctrl.last_horizon} | "
            f"eval={ctrl.last_candidate_evaluations} "
            f"physics_steps={ctrl.last_physics_steps} | "
            f"warm-start(second)={ctrl.last_used_warm_start}"
        )

    out = OUT / "matched_physics_cem_budget.json"

    out.write_text(
        json.dumps(
            {
                "config": {
                    "horizon": cfg["horizon"],
                    "cem_samples": cfg["cem_samples"],
                    "cem_elite_frac":
                        cfg["cem_elite_frac"],
                    "cem_iterations":
                        cfg["cem_iterations"],
                    "expected_elite":
                        expected_elite,
                },
                "rows": rows,
            },
            indent=2,
        )
    )

    print()
    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)
    print("Horizon:                  5")
    print("Candidates/iteration:     64")
    print("Elite/iteration:          16")
    print("CEM iterations:           5")
    print("Candidate evaluations:    320/control step")
    print("Physics rollout steps:    1600/control step")
    print("Cold-start tests:         3/3")
    print("Warm-start tests:         3/3")
    print("Actuator projection:      PASS")
    print("Saved:", out)
    print()
    print("MATCHED PHYSICS-CEM SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
