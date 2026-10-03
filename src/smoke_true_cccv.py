#!/usr/bin/env python3

from pathlib import Path
import json
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_battery_pack import build_pack_from_ecm
from submission_revision_controllers import (
    VoltageFeedbackCCCVController,
    uniform_cc_current,
)


ECM = ROOT / "results" / "ecm" / "ecm_params_20260630_202031.parquet"

MATR = (
    ROOT
    / "data"
    / "processed"
    / "BatteryML"
    / "MATR"
)

OUT = (
    ROOT
    / "results"
    / "submission_revision"
    / "true_cccv_smoke"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def get_matr_voltage_limit():
    """
    Read one representative cell from each MATR batch.

    BatteryML stores the dataset voltage limit in metadata.
    """
    representatives = [
        MATR / "MATR_b1c0.pkl",
        MATR / "MATR_b2c0.pkl",
        MATR / "MATR_b3c0.pkl",
        MATR / "MATR_b4c0.pkl",
    ]

    values = []

    for p in representatives:
        if not p.exists():
            continue

        with open(p, "rb") as f:
            d = pickle.load(f)

        value = d.get("max_voltage_limit_in_V")

        if value is not None:
            values.append(float(value))

    if not values:
        raise RuntimeError(
            "Could not read MATR max_voltage_limit_in_V."
        )

    spread = max(values) - min(values)

    print(
        "MATR representative Vmax values:",
        values,
    )

    if spread > 1e-6:
        print(
            "WARNING: MATR batches contain differing Vmax metadata."
        )

    return float(np.median(values))


def run_episode(
    seed,
    v_ref,
    soc_init,
    target_soc,
    soc_noise=0.03,
    max_steps=120,
):
    cfg = {
        "I_max_C": 3.0,
        "I_pack_max_C": 2.5,
        "V_max": float(v_ref),
        "T_max": 45.0,
        "target_soc": float(target_soc),
        "dt_s": 60.0,
    }

    pack = build_pack_from_ecm(
        ecm_parquet=ECM,
        n_cells=12,
        chemistry="LFP",
        T_amb=25.0,
        soc_init=float(soc_init),
        soc_noise=float(soc_noise),
        seed=int(seed),
    )

    controller = VoltageFeedbackCCCVController(cfg)
    controller.reset()

    cc_current = uniform_cc_current(pack, cfg)

    history = []
    max_overshoot = -1e9

    for step in range(max_steps):
        currents = controller.get_currents(pack)

        phase = controller.last_phase
        predicted_vmax = controller.last_predicted_max_voltage

        metrics = pack.step(
            currents,
            dt=cfg["dt_s"],
        )

        vmax = max(float(c.V_term) for c in pack.cells)
        soc_mean = float(np.mean([c.SOC for c in pack.cells]))
        tmax = max(float(c.T_C) for c in pack.cells)

        overshoot = vmax - v_ref
        max_overshoot = max(max_overshoot, overshoot)

        history.append({
            "step": int(step),
            "time_min": float((step + 1)),
            "phase": phase,
            "current_A": float(currents[0]),
            "predicted_vmax_V": (
                None
                if predicted_vmax is None
                else float(predicted_vmax)
            ),
            "actual_vmax_V": float(vmax),
            "soc_mean": soc_mean,
            "tmax_C": float(tmax),
        })

        if soc_mean >= target_soc:
            break

    cv_rows = [
        h
        for h in history
        if h["phase"] == "CV"
    ]

    min_cv_current = (
        min(h["current_A"] for h in cv_rows)
        if cv_rows
        else None
    )

    summary = {
        "seed": int(seed),
        "soc_init": float(soc_init),
        "target_soc": float(target_soc),
        "v_ref_V": float(v_ref),
        "cc_current_A": float(cc_current),
        "steps": len(history),
        "final_soc_mean": history[-1]["soc_mean"],
        "final_vmax_V": history[-1]["actual_vmax_V"],
        "max_voltage_overshoot_V": float(max_overshoot),
        "cv_entered": bool(cv_rows),
        "n_cv_steps": len(cv_rows),
        "min_cv_current_A": (
            None
            if min_cv_current is None
            else float(min_cv_current)
        ),
    }

    return summary, history


def main():
    if not ECM.exists():
        raise FileNotFoundError(ECM)

    v_ref = get_matr_voltage_limit()

    print()
    print("=" * 72)
    print("TRUE VOLTAGE-FEEDBACK CC-CV SMOKE TEST")
    print("=" * 72)
    print(f"Dataset-derived LFP Vref: {v_ref:.6f} V")
    print()

    all_results = []

    # A. Primary-like experiments
    for seed in [0, 7, 14]:
        summary, history = run_episode(
            seed=seed,
            v_ref=v_ref,
            soc_init=0.20,
            target_soc=0.80,
            soc_noise=0.03,
        )

        all_results.append({
            "scenario": "primary_like",
            "summary": summary,
            "history": history,
        })

        print(
            "PRIMARY",
            f"seed={seed:2d}",
            f"steps={summary['steps']:3d}",
            f"SOC={summary['final_soc_mean']:.4f}",
            f"Vmax={summary['final_vmax_V']:.6f}",
            f"overshoot={summary['max_voltage_overshoot_V']:+.8f}",
            f"CV={summary['cv_entered']}",
            f"CVsteps={summary['n_cv_steps']}",
        )

    print()

    # B. High-SOC diagnostic that should force the CV branch.
    for seed in [0, 7, 14]:
        summary, history = run_episode(
            seed=seed,
            v_ref=v_ref,
            soc_init=0.72,
            target_soc=0.95,
            soc_noise=0.01,
        )

        all_results.append({
            "scenario": "forced_cv",
            "summary": summary,
            "history": history,
        })

        print(
            "FORCED-CV",
            f"seed={seed:2d}",
            f"steps={summary['steps']:3d}",
            f"SOC={summary['final_soc_mean']:.4f}",
            f"Vmax={summary['final_vmax_V']:.6f}",
            f"overshoot={summary['max_voltage_overshoot_V']:+.8f}",
            f"CV={summary['cv_entered']}",
            f"CVsteps={summary['n_cv_steps']}",
            f"minI={summary['min_cv_current_A']}",
        )

    out_file = OUT / "true_cccv_smoke.json"

    out_file.write_text(
        json.dumps(
            all_results,
            indent=2,
        )
    )

    # Validation gates
    summaries = [
        x["summary"]
        for x in all_results
    ]

    worst_overshoot = max(
        x["max_voltage_overshoot_V"]
        for x in summaries
    )

    forced = [
        x["summary"]
        for x in all_results
        if x["scenario"] == "forced_cv"
    ]

    forced_cv_count = sum(
        int(x["cv_entered"])
        for x in forced
    )

    tapered_count = sum(
        int(
            x["min_cv_current_A"] is not None
            and
            x["min_cv_current_A"] < x["cc_current_A"] - 1e-5
        )
        for x in forced
    )

    print()
    print("=" * 72)
    print("VALIDATION")
    print("=" * 72)
    print(
        "Worst one-step voltage overshoot:",
        f"{worst_overshoot:+.8f} V",
    )
    print(
        "Forced-CV episodes entering CV:",
        f"{forced_cv_count}/{len(forced)}",
    )
    print(
        "Forced-CV episodes showing current taper:",
        f"{tapered_count}/{len(forced)}",
    )
    print(
        "Saved:",
        out_file,
    )

    # Numerical tolerance allows float32 plant arithmetic.
    if worst_overshoot > 5e-4:
        raise SystemExit(
            "FAIL: voltage overshoot exceeded 0.5 mV."
        )

    if forced_cv_count != len(forced):
        raise SystemExit(
            "FAIL: not all forced-CV episodes entered CV mode."
        )

    if tapered_count != len(forced):
        raise SystemExit(
            "FAIL: not all forced-CV episodes demonstrated current taper."
        )

    print()
    print("SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
