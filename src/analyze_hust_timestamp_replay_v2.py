#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel


ROOT = Path(__file__).resolve().parent.parent

OUT = (
    ROOT
    / "results/diagnostics/"
      "hust_timestamp_replay_v2_n30"
)

RAW = (
    OUT
    / "hust_timestamp_replay_v2_raw.jsonl"
)

SUMMARY = (
    OUT
    / "hust_timestamp_replay_v2_summary.json"
)

PROV = (
    OUT
    / "hust_timestamp_replay_v2_provenance.json"
)

STATS = (
    OUT
    / "hust_timestamp_replay_v2_statistics.csv"
)

SAFETY = (
    OUT
    / "hust_timestamp_replay_v2_safety.csv"
)

FREEZE = Path(
    "/home/msoylu/alper/Graph-Guided/"
    "provenance/"
    "hust_timestamp_replay_v2_n30_freeze.json"
)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def bootstrap_ci(diff, n=20000, seed=20261004):
    rng = np.random.default_rng(seed)

    diff = np.asarray(
        diff,
        dtype=float,
    )

    idx = rng.integers(
        0,
        len(diff),
        size=(
            n,
            len(diff),
        ),
    )

    means = diff[idx].mean(
        axis=1
    )

    return np.percentile(
        means,
        [2.5, 97.5],
    )


def bootstrap_relative_ci(a, b, n=20000, seed=20261005):
    rng = np.random.default_rng(seed)

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    idx = rng.integers(
        0,
        len(a),
        size=(
            n,
            len(a),
        ),
    )

    am = a[idx].mean(
        axis=1
    )

    bm = b[idx].mean(
        axis=1
    )

    rel = (
        (am - bm)
        / bm
        * 100.0
    )

    return np.percentile(
        rel,
        [2.5, 97.5],
    )


def holm_adjust(p_values):
    p = np.asarray(
        p_values,
        dtype=float,
    )

    order = np.argsort(p)

    adjusted = np.empty_like(
        p
    )

    running = 0.0
    m = len(p)

    for rank, idx in enumerate(order):
        value = min(
            1.0,
            (m - rank)
            * p[idx],
        )

        running = max(
            running,
            value,
        )

        adjusted[idx] = running

    return adjusted


rows = [
    json.loads(line)
    for line in RAW.read_text().splitlines()
    if line.strip()
]

df = pd.DataFrame(rows)

assert len(df) == 120

expected_arms = {
    "HUST-Raw",
    "HUST-Projected",
    "Physics-CEM",
    "GraphOptimizer",
}

assert set(df["arm"]) == expected_arms

counts = (
    df.groupby("arm")
    .size()
    .to_dict()
)

assert all(
    counts[x] == 30
    for x in expected_arms
)

for ep, sub in df.groupby(
    "episode"
):
    assert len(sub) == 4

    assert (
        sub[
            "initial_pack_sha256"
        ].nunique()
        == 1
    )


metrics = {
    "charging_time_min":
        "Charging time (min)",

    "final_SOC_sigma_pct":
        "Final SOC sigma (%)",

    "final_T_gradient_C":
        "Final thermal gradient (C)",

    "cumulative_aging":
        "Cumulative aging proxy",

    "energy_Wh":
        "Energy (Wh)",
}


comparisons = [
    (
        "primary",
        "GraphOptimizer",
        "Physics-CEM",
    ),
    (
        "secondary",
        "GraphOptimizer",
        "HUST-Projected",
    ),
]


results = []

for family, arm_a, arm_b in comparisons:

    A = (
        df[
            df["arm"] == arm_a
        ]
        .sort_values("episode")
        .reset_index(drop=True)
    )

    B = (
        df[
            df["arm"] == arm_b
        ]
        .sort_values("episode")
        .reset_index(drop=True)
    )

    assert np.array_equal(
        A["episode"].to_numpy(),
        B["episode"].to_numpy(),
    )

    for metric, label in metrics.items():

        a = A[
            metric
        ].to_numpy(
            dtype=float
        )

        b = B[
            metric
        ].to_numpy(
            dtype=float
        )

        d = a - b

        low, high = bootstrap_ci(
            d
        )

        rlow, rhigh = (
            bootstrap_relative_ci(
                a,
                b,
            )
        )

        relative = (
            (
                np.mean(a)
                - np.mean(b)
            )
            / np.mean(b)
            * 100.0
        )

        if np.allclose(
            d,
            0,
        ):
            wp = 1.0
            tp = 1.0
        else:
            wp = float(
                wilcoxon(
                    d,
                    alternative="two-sided",
                    zero_method="wilcox",
                ).pvalue
            )

            tp = float(
                ttest_rel(
                    a,
                    b,
                ).pvalue
            )

        sd = np.std(
            d,
            ddof=1,
        )

        dz = (
            float(
                np.mean(d)
                / sd
            )
            if sd > 0
            else np.nan
        )

        results.append({
            "family":
                family,

            "comparison":
                f"{arm_a} - {arm_b}",

            "metric":
                metric,

            "metric_label":
                label,

            "n":
                len(d),

            "mean_A":
                float(
                    np.mean(a)
                ),

            "mean_B":
                float(
                    np.mean(b)
                ),

            "mean_difference":
                float(
                    np.mean(d)
                ),

            "bootstrap95_low":
                float(low),

            "bootstrap95_high":
                float(high),

            "relative_difference_pct":
                float(relative),

            "relative_bootstrap95_low_pct":
                float(rlow),

            "relative_bootstrap95_high_pct":
                float(rhigh),

            "wilcoxon_p":
                wp,

            "paired_t_p":
                tp,

            "cohen_dz":
                dz,
        })


stats = pd.DataFrame(
    results
)

stats[
    "wilcoxon_holm_p"
] = np.nan

for family in stats[
    "family"
].unique():

    mask = (
        stats[
            "family"
        ]
        == family
    )

    stats.loc[
        mask,
        "wilcoxon_holm_p",
    ] = holm_adjust(
        stats.loc[
            mask,
            "wilcoxon_p",
        ].to_numpy()
    )

stats.to_csv(
    STATS,
    index=False,
)


safety_rows = []

for arm in [
    "HUST-Raw",
    "HUST-Projected",
    "Physics-CEM",
    "GraphOptimizer",
]:

    sub = df[
        df["arm"] == arm
    ]

    safety_rows.append({
        "arm":
            arm,

        "episodes":
            len(sub),

        "target_successes":
            int(
                sub[
                    "target_reached"
                ].sum()
            ),

        "episodes_with_explicit_violation":
            int(
                (
                    sub[
                        "explicit_study_violations"
                    ]
                    > 0
                ).sum()
            ),

        "total_explicit_study_violations":
            int(
                sub[
                    "explicit_study_violations"
                ].sum()
            ),

        "max_peak_voltage_V":
            float(
                sub[
                    "peak_voltage_V"
                ].max()
            ),

        "max_peak_temperature_C":
            float(
                sub[
                    "peak_T_C"
                ].max()
            ),

        "mean_actuator_exceed_steps":
            float(
                sub[
                    "actuator_exceed_steps"
                ].mean()
            ),
    })


safety = pd.DataFrame(
    safety_rows
)

safety.to_csv(
    SAFETY,
    index=False,
)


print("=" * 118)
print("HUST TIMESTAMP REPLAY V2 - PAIRED STATISTICS")
print("=" * 118)

for family in [
    "primary",
    "secondary",
]:
    print()
    print(
        family.upper()
    )
    print("-" * 118)

    sub = stats[
        stats["family"]
        == family
    ]

    for _, r in sub.iterrows():

        print(
            f"{r['metric_label']:28s} | "
            f"A={r['mean_A']:.8g} | "
            f"B={r['mean_B']:.8g} | "
            f"diff={r['mean_difference']:+.8g} | "
            f"95%CI=[{r['bootstrap95_low']:+.8g}, "
            f"{r['bootstrap95_high']:+.8g}] | "
            f"rel={r['relative_difference_pct']:+.2f}% | "
            f"W p={r['wilcoxon_p']:.4g} | "
            f"Holm={r['wilcoxon_holm_p']:.4g} | "
            f"dz={r['cohen_dz']:+.3f}"
        )


print()
print("=" * 118)
print("SAFETY")
print("=" * 118)

print(
    safety.to_string(
        index=False
    )
)


FREEZE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

freeze = {
    "analysis":
        "HUST timestamp replay v2 N=30",

    "git_head":
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),

    "files": {
        str(
            RAW.relative_to(ROOT)
        ):
            sha256(RAW),

        str(
            SUMMARY.relative_to(ROOT)
        ):
            sha256(SUMMARY),

        str(
            PROV.relative_to(ROOT)
        ):
            sha256(PROV),

        str(
            STATS.relative_to(ROOT)
        ):
            sha256(STATS),

        str(
            SAFETY.relative_to(ROOT)
        ):
            sha256(SAFETY),

        "src/analyze_hust_timestamp_replay_v2.py":
            sha256(
                ROOT
                / "src/"
                  "analyze_hust_timestamp_replay_v2.py"
            ),
    },

    "claim_boundary":
        (
            "External HUST measured current/time/voltage "
            "trajectory replay on the study MATR-informed "
            "ECM/thermal plant; not external plant validation."
        ),

    "primary_comparison":
        "GraphOptimizer versus matched Physics-CEM",

    "secondary_comparison":
        "GraphOptimizer versus actuator-projected HUST protocol",
}

FREEZE.write_text(
    json.dumps(
        freeze,
        indent=2,
    )
    + "\n"
)

print()
print(
    "statistics:",
    STATS
)

print(
    "safety:",
    SAFETY
)

print(
    "freeze:",
    FREEZE
)
