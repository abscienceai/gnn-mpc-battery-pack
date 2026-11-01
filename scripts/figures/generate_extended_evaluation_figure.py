#!/usr/bin/env python3
"""Generate the manuscript extended-evaluation robustness figure.

The figure uses only frozen repository outputs and does not rerun any
controller experiment. It creates a vector PDF and a 600-dpi PNG.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    "primary": ROOT / "results/canonical_lfp_n30/canonical_lfp_n30_raw.jsonl",
    "ablation": ROOT / "results/extended_evaluation/surrogate_ablation/surrogate_ablation_n30.jsonl",
    "node90": ROOT / "results/extended_evaluation/blend_match/nodeonly_blend_match_n30.jsonl",
    "shared": ROOT / "results/extended_evaluation/physics_shared_objective/physics_shared_objective_n30.jsonl",
    "mismatch": ROOT / "results/extended_evaluation/model_mismatch/model_mismatch_n30.jsonl",
    "seed_stability": ROOT / "results/extended_evaluation/training_seed_stability/training_seed_stability_n10.jsonl",
}
OUTDIR = ROOT / "figures/submission_v2"
OUT_PDF = OUTDIR / "fig_extended_evaluation_robustness.pdf"
OUT_PNG = OUTDIR / "fig_extended_evaluation_robustness.png"
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20261101


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def by_value(rows, key, value):
    return [r for r in rows if r.get(key) == value]


def index_by_seed(rows):
    return {int(r["seed"]): r for r in rows}


def paired_differences(rows_a, rows_b, metric):
    a, b = index_by_seed(rows_a), index_by_seed(rows_b)
    common = sorted(set(a) & set(b))
    if not common:
        raise RuntimeError(f"No common seeds for {metric}")
    return np.asarray([float(a[s][metric]) - float(b[s][metric]) for s in common])


def paired_bootstrap_ci(differences):
    d = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(d), size=(BOOTSTRAP_RESAMPLES, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def yerr(m, lo, hi):
    return np.asarray([[m - lo], [hi - m]], dtype=float)


def arch_summary(rows, model):
    x = by_value(rows, "model", model)
    return {
        "time": mean(x, "charging_time_min"),
        "soc": mean(x, "final_SOC_sigma_pct"),
        "thermal": mean(x, "final_T_gradient_C"),
    }


def main():
    primary = load_jsonl(PATHS["primary"])
    ablation = load_jsonl(PATHS["ablation"])
    node90 = load_jsonl(PATHS["node90"])
    shared = load_jsonl(PATHS["shared"])
    mismatch = load_jsonl(PATHS["mismatch"])
    stability = load_jsonl(PATHS["seed_stability"])
    OUTDIR.mkdir(parents=True, exist_ok=True)

    graph_primary = by_value(primary, "controller", "GraphOptimizer")
    physics_primary = by_value(primary, "controller", "Physics-CEM")

    arch = {m: arch_summary(ablation, m) for m in ["PackGNN", "NodeOnly", "FlatMLP"]}
    arch["NodeOnly 90/10"] = {
        "time": mean(node90, "charging_time_min"),
        "soc": mean(node90, "final_SOC_sigma_pct"),
        "thermal": mean(node90, "final_T_gradient_C"),
    }

    eval_seeds = sorted({int(r["seed"]) for r in stability})
    seed_results = {}
    for model in ["PackGNN", "NodeOnly", "FlatMLP"]:
        rows42 = [r for r in ablation if r["model"] == model and int(r["seed"]) in eval_seeds]
        seed_results[(42, model)] = {
            "time": mean(rows42, "charging_time_min"),
            "soc": mean(rows42, "final_SOC_sigma_pct"),
            "thermal": mean(rows42, "final_T_gradient_C"),
        }
        for s in [101, 202]:
            rr = [r for r in stability if int(r["train_seed"]) == s and r["model"] == model]
            seed_results[(s, model)] = {
                "time": mean(rr, "charging_time_min"),
                "soc": mean(rr, "final_SOC_sigma_pct"),
                "thermal": mean(rr, "final_T_gradient_C"),
            }

    physics_shared = by_value(shared, "model", "Physics-CEM-Shared")
    graph_mismatch = by_value(mismatch, "controller", "PackGNN")
    physics_mismatch = by_value(mismatch, "controller", "Physics-CEM")

    time_conditions = [
        ("Primary", paired_differences(graph_primary, physics_primary, "charging_time_min")),
        ("Shared\nobjective", paired_differences(graph_primary, physics_shared, "charging_time_min")),
        ("Model-plant\nmismatch", paired_differences(graph_mismatch, physics_mismatch, "charging_time_min")),
    ]
    thermal_conditions = [
        ("Primary", paired_differences(graph_primary, physics_primary, "final_T_gradient_C")),
        ("Shared\nobjective", paired_differences(graph_primary, physics_shared, "final_T_gradient_C")),
        ("Model-plant\nmismatch", paired_differences(graph_mismatch, physics_mismatch, "final_T_gradient_C")),
    ]

    mismatch_graph_viol = int(sum(int(r["total_violations"]) for r in graph_mismatch))
    mismatch_physics_viol = int(sum(int(r["total_violations"]) for r in physics_mismatch))

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "legend.fontsize": 7.3, "figure.dpi": 150, "savefig.dpi": 600,
        "axes.linewidth": 0.8, "lines.linewidth": 1.25,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    cmap = plt.get_cmap("tab10")
    colors = {"PackGNN": cmap(0), "NodeOnly": cmap(1), "FlatMLP": cmap(2), "NodeOnly 90/10": cmap(3)}
    markers = {"PackGNN": "o", "NodeOnly": "s", "FlatMLP": "^", "NodeOnly 90/10": "D"}

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.25), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    offsets = {"PackGNN": (5, -18), "NodeOnly": (5, 7), "FlatMLP": (-58, 7), "NodeOnly 90/10": (-90, -15)}
    for label in ["PackGNN", "NodeOnly", "FlatMLP", "NodeOnly 90/10"]:
        v = arch[label]
        ax_a.scatter(v["time"], v["thermal"], s=52, marker=markers[label], color=colors[label],
                     edgecolor="black", linewidth=0.55, zorder=3)
        ax_a.annotate(f"{label}\n$\\sigma_{{\\mathrm{{SOC}}}}$={v['soc']:.3f}%",
                      xy=(v["time"], v["thermal"]), xytext=offsets[label], textcoords="offset points",
                      fontsize=6.8, arrowprops={"arrowstyle": "-", "linewidth": 0.5})
    ax_a.set_xlabel("Mean charging time (min)")
    ax_a.set_ylabel(r"Final temperature spread, $\Delta T$ ($^\circ$C)")
    ax_a.set_title("(a) Architecture attribution", loc="left", fontweight="bold")
    ax_a.grid(alpha=0.20, linewidth=0.5)

    for model in ["PackGNN", "NodeOnly", "FlatMLP"]:
        xs = [seed_results[(s, model)]["time"] for s in [42, 101, 202]]
        ys = [seed_results[(s, model)]["thermal"] for s in [42, 101, 202]]
        ax_b.plot(xs, ys, marker=markers[model], markersize=5.4, color=colors[model], label=model)
        for x, y, s in zip(xs, ys, [42, 101, 202]):
            ax_b.annotate(str(s), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6.4)
    ax_b.set_xlabel("Mean charging time (min)")
    ax_b.set_ylabel(r"Final temperature spread, $\Delta T$ ($^\circ$C)")
    ax_b.set_title("(b) Training-seed stability", loc="left", fontweight="bold")
    ax_b.legend(frameon=False, loc="best", handlelength=1.6)
    ax_b.grid(alpha=0.20, linewidth=0.5)

    x = np.arange(3)
    for ax, conditions, ylabel, title, decimals in [
        (ax_c, time_conditions, "Charging-time difference\nGraphOptimizer - Physics-CEM (min)",
         "(c) Charging-time robustness and fairness", 2),
        (ax_d, thermal_conditions, r"Temperature-spread difference,\nGraphOptimizer - Physics-CEM ($^\circ$C)",
         "(d) Thermal trade-off robustness", 3),
    ]:
        means = []
        for i, (_, d) in enumerate(conditions):
            m, lo, hi = paired_bootstrap_ci(d)
            means.append(m)
            ax.errorbar(i, m, yerr=yerr(m, lo, hi), fmt="o", markersize=5.8, capsize=3.5,
                        color=cmap(i), markeredgecolor="black", markeredgewidth=0.45,
                        linewidth=1.15, zorder=3)
            txt = f"{m:.{decimals}f}" if decimals == 2 else f"{m:+.{decimals}f}"
            ax.annotate(txt, (i, m), xytext=(0, -16 if ax is ax_c else 8),
                        textcoords="offset points", ha="center", fontsize=6.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([lab for lab, _ in conditions])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.20, linewidth=0.5)

    ax_d.annotate(f"Mismatch violations:\nGraphOptimizer {mismatch_graph_viol}, Physics-CEM {mismatch_physics_viol}",
                  xy=(2, paired_bootstrap_ci(thermal_conditions[2][1])[0]), xytext=(-8, 24),
                  textcoords="offset points", ha="right", fontsize=6.7,
                  bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "linewidth": 0.5})

    fig.suptitle("Architecture attribution, training stability, and robustness controls",
                 fontsize=10.3, fontweight="bold")
    fig.text(0.5, -0.012,
             "Panels (c,d) report paired GraphOptimizer-minus-Physics-CEM mean differences with 95% paired-bootstrap "
             "confidence intervals (20,000 resamples). Negative charging-time differences favour GraphOptimizer; "
             "positive thermal differences indicate larger final inter-cell temperature spread.",
             ha="center", va="top", fontsize=6.6, wrap=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.04, dpi=600)
    plt.close(fig)
    print(OUT_PDF.relative_to(ROOT))
    print(OUT_PNG.relative_to(ROOT))


if __name__ == "__main__":
    main()
