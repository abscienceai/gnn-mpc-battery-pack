#!/usr/bin/env python3

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_battery_pack import build_pack_from_ecm
from safe_fast_charge_optimizer import (
    default_config,
    CCCVController,
    BalancedCCCVController,
    run_episode,
)
from submission_revision_controllers import (
    apply_dataset_voltage_limits,
)

ECM = ROOT / "results/ecm/ecm_params_20260630_202031.parquet"

OUT = ROOT / "results/submission_revision/integrated_cccv_smoke"
OUT.mkdir(parents=True, exist_ok=True)


def main():

    cfg = default_config()

    cfg["n_cells"] = 12
    cfg["chemistry"] = "LFP"
    cfg["soc_init"] = 0.20
    cfg["soc_noise"] = 0.03
    cfg["target_soc"] = 0.80
    cfg["max_steps"] = 120

    cfg = apply_dataset_voltage_limits(
        cfg,
        repo_root=ROOT,
    )

    print("=" * 78)
    print("MAIN-PATH CC-CV INTEGRATION SMOKE TEST")
    print("=" * 78)
    print(
        f"Resolved limits: "
        f"{cfg['V_min']:.3f}--{cfg['V_max']:.3f} V "
        f"from {cfg['voltage_limit_dataset']}"
    )

    all_rows = []

    controller_types = [
        ("CC-CV", CCCVController),
        ("CC-CV-Balance", BalancedCCCVController),
    ]

    for seed in [0, 7, 14]:

        for label, cls in controller_types:

            pack = build_pack_from_ecm(
                ECM,
                n_cells=cfg["n_cells"],
                chemistry=cfg["chemistry"],
                T_amb=cfg["T_amb"],
                soc_init=cfg["soc_init"],
                soc_noise=cfg["soc_noise"],
                seed=seed,
            )

            controller = cls(cfg)

            result = run_episode(
                pack,
                controller,
                cfg,
                label,
                verbose=False,
            )

            row = {
                "seed": seed,
                "controller": label,
                "final_SOC": result["final_SOC"],
                "peak_cell_voltage": result["peak_cell_voltage"],
                "violations": result["total_violations"],
                "time_min": result["charging_time_min"],
                "final_SOC_imbalance": result["final_SOC_imbalance"],
                "final_T_gradient": result["final_T_gradient"],
                "cv_entered": bool(
                    getattr(controller, "cv_mode", False)
                ),
            }

            all_rows.append(row)

            print(
                f"{label:<16} "
                f"seed={seed:2d} "
                f"SOC={row['final_SOC']:.4f} "
                f"Vpeak={row['peak_cell_voltage']:.6f} "
                f"viol={row['violations']} "
                f"CV={row['cv_entered']} "
                f"time={row['time_min']:.1f} min"
            )

    out_file = OUT / "integrated_cccv_n3.json"

    out_file.write_text(
        json.dumps(
            {
                "config": cfg,
                "results": all_rows,
            },
            indent=2,
        )
    )

    voltage_failures = [
        r for r in all_rows
        if r["peak_cell_voltage"] > cfg["V_max"] + 5e-4
    ]

    violation_failures = [
        r for r in all_rows
        if r["violations"] != 0
    ]

    soc_failures = [
        r for r in all_rows
        if r["final_SOC"] < cfg["target_soc"] - 0.006
    ]

    cv_failures = [
        r for r in all_rows
        if not r["cv_entered"]
    ]

    print()
    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)
    print(
        "Voltage-limit failures:",
        len(voltage_failures),
    )
    print(
        "Plant-violation failures:",
        len(violation_failures),
    )
    print(
        "Target-SOC failures:",
        len(soc_failures),
    )
    print(
        "Episodes entering CV:",
        f"{len(all_rows)-len(cv_failures)}/{len(all_rows)}",
    )
    print("Saved:", out_file)

    if voltage_failures:
        raise SystemExit("FAIL: voltage limit exceeded.")

    if violation_failures:
        raise SystemExit("FAIL: plant reported safety violations.")

    if soc_failures:
        raise SystemExit("FAIL: target SOC not reached.")

    if cv_failures:
        raise SystemExit("FAIL: one or more episodes never entered CV.")

    print()
    print("MAIN INTEGRATION SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
