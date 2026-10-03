#!/usr/bin/env python3

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

RAW = (
    ROOT
    / "results/canonical_lfp_n30/"
      "canonical_lfp_n30_raw.jsonl"
)

OUT_DIR = (
    ROOT
    / "results/canonical_lfp_n30/statistics"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = [7 * i for i in range(30)]

CONTROLLERS = [
    "CC-CV",
    "CC-CV-Balance",
    "Proportional",
    "Physics-CEM",
    "GraphOptimizer",
]

METRICS = {
    "charging_time_min": "lower",
    "final_SOC_sigma_pct": "lower",
    "final_T_gradient_C": "lower",
    "peak_T_C": "lower",
    "cumulative_aging": "lower",
    "pack_energy_Wh": "lower",
}

# Pre-specified primary comparison:
# GraphOptimizer vs matched direct-physics CEM.
#
# Secondary comparisons:
# GraphOptimizer vs conventional baselines.
COMPARISONS = [
    ("GraphOptimizer", "Physics-CEM", "primary"),
    ("GraphOptimizer", "CC-CV", "secondary"),
    ("GraphOptimizer", "CC-CV-Balance", "secondary"),
    ("GraphOptimizer", "Proportional", "secondary"),
]

rng = np.random.default_rng(20261004)


def load_rows():
    rows = []

    with RAW.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    assert len(rows) == 150

    return pd.DataFrame(rows)


def bootstrap_mean_ci(d, n_boot=20000):
    d = np.asarray(d, dtype=float)
    n = len(d)

    idx = rng.integers(
        0,
        n,
        size=(n_boot, n),
    )

    boot = d[idx].mean(axis=1)

    lo, hi = np.percentile(
        boot,
        [2.5, 97.5],
    )

    return float(lo), float(hi)


def bootstrap_relative_ci(a, b, n_boot=20000):
    """
    Relative change of A vs B:
        (mean(A)-mean(B))/mean(B) * 100

    Negative = A is lower than B.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    n = len(a)

    idx = rng.integers(
        0,
        n,
        size=(n_boot, n),
    )

    ma = a[idx].mean(axis=1)
    mb = b[idx].mean(axis=1)

    rel = (
        (ma - mb)
        / mb
        * 100.0
    )

    lo, hi = np.percentile(
        rel,
        [2.5, 97.5],
    )

    return (
        float(
            (a.mean() - b.mean())
            / b.mean()
            * 100.0
        ),
        float(lo),
        float(hi),
    )


def cohen_dz(d):
    d = np.asarray(d, dtype=float)

    sd = np.std(
        d,
        ddof=1,
    )

    if sd == 0:
        if np.mean(d) == 0:
            return 0.0
        return math.copysign(
            float("inf"),
            np.mean(d),
        )

    return float(
        np.mean(d)
        / sd
    )


def rank_biserial_from_wilcoxon(d):
    """
    Paired rank-biserial effect size.

    Positive means positive differences dominate;
    negative means negative differences dominate.
    """
    d = np.asarray(d, dtype=float)

    nz = d[
        ~np.isclose(
            d,
            0.0,
            atol=1e-12,
        )
    ]

    if len(nz) == 0:
        return 0.0

    ranks = stats.rankdata(
        np.abs(nz)
    )

    Wp = float(
        ranks[nz > 0].sum()
    )

    Wm = float(
        ranks[nz < 0].sum()
    )

    denom = (
        len(nz)
        * (len(nz) + 1)
        / 2.0
    )

    return float(
        (Wp - Wm)
        / denom
    )


df = load_rows()

# ------------------------------------------------------------
# Integrity audit
# ------------------------------------------------------------

assert set(df["seed"]) == set(SEEDS)
assert set(df["controller"]) == set(CONTROLLERS)

counts = (
    df
    .groupby("controller")
    .size()
    .to_dict()
)

assert all(
    counts[c] == 30
    for c in CONTROLLERS
)

for seed in SEEDS:
    sub = df[df.seed == seed]

    assert len(sub) == 5

    assert (
        sub[
            "initial_pack_sha256"
        ].nunique()
        == 1
    )

print(
    "PAIRED DATA INTEGRITY: PASS"
)


# ------------------------------------------------------------
# Descriptive table
# ------------------------------------------------------------

desc_rows = []

for controller in CONTROLLERS:

    sub = (
        df[df.controller == controller]
        .sort_values("seed")
    )

    row = {
        "controller":
            controller,

        "n":
            len(sub),

        "target_success":
            int(
                sub[
                    "target_reached"
                ].sum()
            ),

        "zero_violation_episodes":
            int(
                (
                    sub[
                        "total_violations"
                    ]
                    == 0
                ).sum()
            ),

        "total_violations":
            int(
                sub[
                    "total_violations"
                ].sum()
            ),
    }

    for m in METRICS:
        x = sub[m].to_numpy(
            dtype=float
        )

        row[
            m + "_mean"
        ] = float(
            np.mean(x)
        )

        row[
            m + "_sd"
        ] = float(
            np.std(
                x,
                ddof=1,
            )
        )

    desc_rows.append(row)

desc = pd.DataFrame(
    desc_rows
)

desc.to_csv(
    OUT_DIR
    / "descriptive_statistics.csv",
    index=False,
)


# ------------------------------------------------------------
# Paired inference
# ------------------------------------------------------------

paired_rows = []

for A, B, tier in COMPARISONS:

    a = (
        df[df.controller == A]
        .sort_values("seed")
        .set_index("seed")
    )

    b = (
        df[df.controller == B]
        .sort_values("seed")
        .set_index("seed")
    )

    assert list(a.index) == list(b.index)

    for metric in METRICS:

        xa = a[metric].to_numpy(
            dtype=float
        )

        xb = b[metric].to_numpy(
            dtype=float
        )

        # Difference definition:
        # GraphOptimizer - comparator.
        d = xa - xb

        mean_diff = float(
            np.mean(d)
        )

        ci_lo, ci_hi = (
            bootstrap_mean_ci(d)
        )

        rel, rel_lo, rel_hi = (
            bootstrap_relative_ci(
                xa,
                xb,
            )
        )

        t_res = stats.ttest_rel(
            xa,
            xb,
            nan_policy="raise",
        )

        try:
            w_res = stats.wilcoxon(
                xa,
                xb,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            W = float(
                w_res.statistic
            )

            p_w = float(
                w_res.pvalue
            )

        except ValueError:
            W = 0.0
            p_w = 1.0

        paired_rows.append({
            "tier":
                tier,

            "method_A":
                A,

            "method_B":
                B,

            "metric":
                metric,

            "n":
                len(d),

            "mean_A":
                float(
                    np.mean(xa)
                ),

            "mean_B":
                float(
                    np.mean(xb)
                ),

            "mean_difference_A_minus_B":
                mean_diff,

            "bootstrap95_difference_low":
                ci_lo,

            "bootstrap95_difference_high":
                ci_hi,

            "relative_change_A_vs_B_pct":
                rel,

            "relative_change95_low_pct":
                rel_lo,

            "relative_change95_high_pct":
                rel_hi,

            "paired_t":
                float(
                    t_res.statistic
                ),

            "paired_t_p":
                float(
                    t_res.pvalue
                ),

            "wilcoxon_W":
                W,

            "wilcoxon_p":
                p_w,

            "cohen_dz":
                cohen_dz(d),

            "rank_biserial":
                rank_biserial_from_wilcoxon(
                    d
                ),
        })

paired = pd.DataFrame(
    paired_rows
)


# Holm correction separately within
# primary and secondary families.
paired[
    "wilcoxon_p_holm"
] = np.nan

for tier in [
    "primary",
    "secondary",
]:

    idx = paired.index[
        paired["tier"]
        == tier
    ].tolist()

    pvals = paired.loc[
        idx,
        "wilcoxon_p",
    ].to_numpy()

    order = np.argsort(
        pvals
    )

    m = len(pvals)

    adjusted_sorted = np.empty(
        m,
        dtype=float,
    )

    running = 0.0

    for rank, pos in enumerate(
        order
    ):
        adj = (
            (m - rank)
            * pvals[pos]
        )

        running = max(
            running,
            adj,
        )

        adjusted_sorted[pos] = min(
            running,
            1.0,
        )

    paired.loc[
        idx,
        "wilcoxon_p_holm",
    ] = adjusted_sorted


paired.to_csv(
    OUT_DIR
    / "paired_statistics.csv",
    index=False,
)


# ------------------------------------------------------------
# Safety table
# ------------------------------------------------------------

safety_rows = []

for controller in CONTROLLERS:

    sub = df[
        df.controller
        == controller
    ]

    safety_rows.append({
        "controller":
            controller,

        "episodes":
            len(sub),

        "target_successes":
            int(
                sub[
                    "target_reached"
                ].sum()
            ),

        "episodes_with_violation":
            int(
                (
                    sub[
                        "total_violations"
                    ]
                    > 0
                ).sum()
            ),

        "total_cell_step_violations":
            int(
                sub[
                    "total_violations"
                ].sum()
            ),

        "maximum_peak_voltage_V":
            float(
                sub[
                    "peak_voltage_V"
                ].max()
            ),

        "maximum_peak_temperature_C":
            float(
                sub[
                    "peak_T_C"
                ].max()
            ),
    })

safety = pd.DataFrame(
    safety_rows
)

safety.to_csv(
    OUT_DIR
    / "safety_statistics.csv",
    index=False,
)


# ------------------------------------------------------------
# Human-readable primary output
# ------------------------------------------------------------

primary = paired[
    (
        paired.method_A
        == "GraphOptimizer"
    )
    &
    (
        paired.method_B
        == "Physics-CEM"
    )
].copy()

print()
print("=" * 88)
print(
    "PRIMARY PAIRED COMPARISON: "
    "GraphOptimizer vs Physics-CEM"
)
print("=" * 88)

for _, r in primary.iterrows():

    print(
        f"{r['metric']:24s} | "
        f"G={r['mean_A']:.6g} | "
        f"P={r['mean_B']:.6g} | "
        f"Δ={r['mean_difference_A_minus_B']:+.6g} "
        f"[{r['bootstrap95_difference_low']:+.6g}, "
        f"{r['bootstrap95_difference_high']:+.6g}] | "
        f"rel={r['relative_change_A_vs_B_pct']:+.2f}% "
        f"[{r['relative_change95_low_pct']:+.2f}, "
        f"{r['relative_change95_high_pct']:+.2f}] | "
        f"W-p={r['wilcoxon_p']:.4g} | "
        f"Holm={r['wilcoxon_p_holm']:.4g} | "
        f"dz={r['cohen_dz']:+.3f}"
    )

print()
print("=" * 88)
print("SAFETY")
print("=" * 88)
print(
    safety.to_string(
        index=False
    )
)

print()
print(
    "saved:",
    OUT_DIR
    / "descriptive_statistics.csv",
)
print(
    "saved:",
    OUT_DIR
    / "paired_statistics.csv",
)
print(
    "saved:",
    OUT_DIR
    / "safety_statistics.csv",
)

print()
print(
    "CANONICAL PAIRED STATISTICS: COMPLETE"
)
