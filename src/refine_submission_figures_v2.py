from pathlib import Path
import hashlib
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures/submission_v2"
THERM = ROOT / "results/diagnostics/thermal_initial_condition_n30"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    plt.close(fig)


# ============================================================
# FIGURE 1: manually prepared publication schematic
# ============================================================

manual_fig1 = OUT / "fig1_controller_architecture.png"

if not manual_fig1.exists():
    raise FileNotFoundError(
        f"Manual publication Figure 1 is missing: {manual_fig1}"
    )

print("Preserving manual publication Fig. 1; no refinement applied.")


# ============================================================
# FIGURE 3: cleaner paired thermal diagnostic
# ============================================================

rows = []
with open(THERM / "thermal_initial_condition_raw.jsonl") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

raw = pd.DataFrame(rows)
stats = pd.read_csv(
    THERM / "thermal_initial_condition_statistics.csv"
)

td = raw[
    raw["controller"].isin(["GraphOptimizer", "Physics-CEM"])
].pivot_table(
    index=["seed", "condition"],
    columns="controller",
    values="final_T_gradient_C",
)

td["difference"] = (
    td["GraphOptimizer"] - td["Physics-CEM"]
)

wide = td["difference"].reset_index().pivot(
    index="seed",
    columns="condition",
    values="difference",
)

fig, ax = plt.subplots(figsize=(5.7, 4.55))

x = np.array([0, 1])

# all paired seeds in neutral grey
for _, r in wide.iterrows():
    ax.plot(
        x,
        [r["heterogeneous_T"], r["uniform_25C"]],
        marker="o",
        markersize=2.7,
        linewidth=0.75,
        alpha=0.22,
        color="0.45",
    )

labels = [
    "Graph-Physics final dT | heterogeneous_T",
    "Graph-Physics final dT | uniform_25C",
]

means, lows, highs = [], [], []

for label in labels:
    r = stats.loc[stats["comparison"] == label].iloc[0]
    means.append(float(r["mean_difference_C"]))
    lows.append(float(r["bootstrap95_low_C"]))
    highs.append(float(r["bootstrap95_high_C"]))

means = np.array(means)
lows = np.array(lows)
highs = np.array(highs)

ax.errorbar(
    x,
    means,
    yerr=np.vstack([means-lows, highs-means]),
    fmt="o",
    markersize=7,
    linewidth=1.7,
    capsize=4,
    color="black",
    label="Mean paired difference ± 95% bootstrap CI",
    zorder=5,
)

ax.plot(
    x,
    means,
    linewidth=1.2,
    color="black",
    zorder=4,
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=0.9,
    color="0.35",
)

ax.set_xticks(
    x,
    ["Heterogeneous initial T", "Uniform initial T = 25°C"],
)

ax.set_ylabel(
    "Final thermal-gradient difference\n"
    "GraphOptimizer - Physics-CEM (°C)"
)

ax.set_title(
    "Controlled thermal-initialisation diagnostic (N=30)"
)

interaction = stats.loc[
    stats["comparison"].str.startswith("interaction:")
].iloc[0]

ax.text(
    0.03, 0.97,
    f"Interaction = {interaction['mean_difference_C']:+.4f} °C\n"
    f"Wilcoxon p = {interaction['wilcoxon_p']:.3f}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="0.7",
        alpha=0.95,
    ),
)

ax.legend(
    frameon=False,
    loc="lower center",
)

ax.margins(x=0.08)

save(fig, "fig3_thermal_initial_condition")


# ============================================================
# Refresh final figure manifest after Fig. 3 refinement
# ============================================================

MANIFEST = OUT / "figure_manifest.json"

if not MANIFEST.exists():
    raise FileNotFoundError(
        "Figure manifest is missing. "
        "Run generate_submission_figures_v2.py first."
    )

manifest = json.loads(
    MANIFEST.read_text(
        encoding="utf-8"
    )
)

FINAL_FIGURE_NAMES = [
    "fig1_controller_architecture.png",
    "fig2_canonical_paired_tradeoff.pdf",
    "fig2_canonical_paired_tradeoff.png",
    "fig3_thermal_initial_condition.pdf",
    "fig3_thermal_initial_condition.png",
    "fig4_hust_paired_tradeoff.pdf",
    "fig4_hust_paired_tradeoff.png",
    "fig5_surrogate_ranking_fidelity.pdf",
    "fig5_surrogate_ranking_fidelity.png",
    "figS1_action_conditioning.pdf",
    "figS1_action_conditioning.png",
]

final_output_paths = [
    OUT / name
    for name in FINAL_FIGURE_NAMES
]

for p in final_output_paths:
    if not p.exists():
        raise FileNotFoundError(
            f"Expected publication figure is missing: {p}"
        )

generator_path = (
    ROOT
    / "src"
    / "generate_submission_figures_v2.py"
)

manifest["updated_utc"] = (
    datetime.now(timezone.utc).isoformat()
)

manifest[
    "frozen_computational_evidence_git_head"
] = (
    "a28143ea3bb60081214ae9aa91e9051b1c202e72"
)

manifest["generation_pipeline"] = [
    "src/generate_submission_figures_v2.py",
    "src/refine_submission_figures_v2.py",
]

manifest["generator_sha256"] = {
    "generate_submission_figures_v2.py":
        sha256(generator_path),

    "refine_submission_figures_v2.py":
        sha256(Path(__file__).resolve()),
}

manifest["figure_policy"] = {
    "fig1_controller_architecture.png":
        (
            "Manually prepared publication schematic; "
            "preserved and not regenerated."
        ),

    "fig2_to_fig5_and_figS1":
        (
            "Generated from frozen v2 "
            "computational evidence."
        ),
}

manifest["outputs"] = {
    str(p.relative_to(ROOT)): sha256(p)
    for p in final_output_paths
}

manifest["scientific_note"] = (
    "Figures 2-5 and S1 are derived from frozen v2 "
    "computational evidence. Figure 1 is a manually "
    "prepared explanatory architecture schematic. "
    "No experiment was rerun and no frozen "
    "computational-evidence file was modified."
)

MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print(
    "Refined Fig. 3 successfully; "
    "manual Fig. 1 preserved."
)

print(
    "Final figure manifest refreshed:",
    MANIFEST,
)
