"""
paired_statistical_tests.py
============================
Reviewer Comment 9: "Paired design should use paired t-tests, not Welch/MWU"

Since all controllers share the same 30 episode seeds, this IS a paired design.
Runs paired t-tests and Wilcoxon signed-rank tests for:
  - GraphOptimizer vs CC-CV        (primary claim)
  - GraphOptimizer vs SimpleMPC    (GNN contribution)
  - GraphOptimizer vs Proportional (thermal gradient claim)
  - GraphOptimizer vs CC-CV-Balance

Also reports effect sizes (Cohen's d for paired samples).
"""

import json, csv, warnings, numpy as np
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

EXP_DIRS = {
    "LFP": Path("results/canonical_LFP_postpatch"),
    "NMC": Path("results/canonical_NMC_postpatch"),
    "LCO": Path("results/canonical_LCO_postpatch"),
}
OUT_DIR = Path("results")


def load_episodes(chemistry="LFP"):
    d = EXP_DIRS[chemistry]
    files = sorted(d.glob("*.json"))
    assert files, f"No JSON in {d}"
    data = json.load(open(files[-1]))
    return data["raw_results"]  # dict: ctrl -> list of 30 episode dicts


def cohen_d_paired(a, b):
    """Cohen's d for paired samples."""
    diff = np.array(a) - np.array(b)
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def run_paired_tests(raw, ctrl_a, ctrl_b, metric, label=None):
    """Run paired t-test and Wilcoxon for metric between ctrl_a and ctrl_b."""
    a = np.array([ep[metric] for ep in raw[ctrl_a]])
    b = np.array([ep[metric] for ep in raw[ctrl_b]])

    # Paired t-test
    t_stat, p_t = stats.ttest_rel(a, b)
    # Wilcoxon signed-rank (requires non-zero differences)
    diffs = a - b
    if np.all(diffs == 0):
        p_w = 1.0; w_stat = 0.0
    else:
        w_stat, p_w = stats.wilcoxon(diffs, alternative="two-sided",
                                      zero_method="wilcox")
    d = cohen_d_paired(a, b)

    name = label or f"{ctrl_a} vs {ctrl_b}"
    return {
        "comparison": name,
        "metric": metric,
        "mean_a": round(float(np.mean(a)), 6),
        "mean_b": round(float(np.mean(b)), 6),
        "mean_diff": round(float(np.mean(diffs)), 6),
        "t_stat": round(float(t_stat), 4),
        "p_paired_t": round(float(p_t), 6),
        "w_stat": round(float(w_stat), 4),
        "p_wilcoxon": round(float(p_w), 6),
        "cohens_d": round(float(d), 4),
        "significant_t": p_t < 0.05,
        "significant_w": p_w < 0.05,
    }


def main():
    print("Loading LFP episode data...")
    raw = load_episodes("LFP")
    ctrls = list(raw.keys())
    print(f"Controllers: {ctrls}")
    print(f"Episodes per controller: {len(raw[ctrls[0]])}")

    metrics = [
        ("final_SOC_imbalance", "σ_SOC (fraction)"),
        ("final_T_gradient",    "ΔT (°C)"),
        ("charging_time_min",   "Time (min)"),
        ("cumulative_aging",    "Aging proxy"),
        ("total_violations",    "Violations"),
    ]

    # All pairs involving GraphOptimizer + key other pairs
    pairs = [
        ("GraphOptimizer", "CC-CV",          "GO vs CC-CV"),
        ("GraphOptimizer", "SimpleMPC",      "GO vs SimpleMPC"),
        ("GraphOptimizer", "Proportional",   "GO vs Proportional"),
        ("GraphOptimizer", "CC-CV-Balance",  "GO vs CC-CV-Balance"),
    ]

    rows = []
    print(f"\n{'Comparison':<25} {'Metric':<22} {'Mean A':>10} {'Mean B':>10} "
          f"{'p(paired-t)':>12} {'p(Wilcox)':>10} {'d':>7} {'Sig?':>5}")
    print("─" * 105)

    for ctrl_a, ctrl_b, label in pairs:
        if ctrl_a not in raw or ctrl_b not in raw:
            print(f"  SKIP: {ctrl_a} or {ctrl_b} not in data")
            continue
        for metric, metric_label in metrics:
            r = run_paired_tests(raw, ctrl_a, ctrl_b, metric, label)
            r["metric_label"] = metric_label
            rows.append(r)
            sig = "***" if r["p_paired_t"] < 0.001 else ("**" if r["p_paired_t"] < 0.01
                   else ("*" if r["p_paired_t"] < 0.05 else "n.s."))
            print(f"{label:<25} {metric_label:<22} {r['mean_a']:>10.5f} {r['mean_b']:>10.5f} "
                  f"{r['p_paired_t']:>12.6f} {r['p_wilcoxon']:>10.6f} {r['cohens_d']:>7.3f} {sig:>5}")
        print()

    # Save CSV
    out = OUT_DIR / "paired_statistical_tests.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved → {out}")

    # LaTeX-ready summary table
    print("\n── LaTeX-ready significance summary (key metrics) ──")
    key_metric = "final_SOC_imbalance"
    print(f"{'Comparison':<28} {'Mean GO':>9} {'Mean B':>9} {'p(paired-t)':>13} {'Cohen d':>9} {'Sig':>5}")
    for r in rows:
        if r["metric"] == key_metric:
            sig = "p<0.001" if r["p_paired_t"] < 0.001 else f"p={r['p_paired_t']:.4f}"
            print(f"{r['comparison']:<28} {r['mean_a']:>9.4f} {r['mean_b']:>9.4f} "
                  f"{sig:>13} {r['cohens_d']:>9.3f} {'***' if r['p_paired_t']<0.001 else '*'}")


if __name__ == "__main__":
    main()
