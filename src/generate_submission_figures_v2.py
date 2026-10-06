#!/usr/bin/env python3
"""
Generate final submission figures from the frozen v2 evidence.

READ-ONLY inputs:
  results/canonical_lfp_n30/
  results/diagnostics/
  provenance/

Outputs:
  figures/submission_v2/

No experiment is rerun and no frozen evidence file is modified.
"""

from pathlib import Path
import json
import hashlib
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / "provenance"
OUT = ROOT / "figures" / "submission_v2"
OUT.mkdir(parents=True, exist_ok=True)

CANON = ROOT / "results" / "canonical_lfp_n30"
THERM = ROOT / "results" / "diagnostics" / "thermal_initial_condition_n30"
HUST = ROOT / "results" / "diagnostics" / "hust_timestamp_replay_v2_n30"

ACTION_JSON = PROV / "pack_gnn_action_v2_heldout_action_test.json"
RANK_JSON = PROV / "pack_gnn_action_v2_ranking_fidelity.json"


# ============================================================
# Publication defaults
# ============================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def assert_close(name, got, expected, atol=1e-6):
    if not np.isclose(float(got), float(expected), atol=atol, rtol=0):
        raise RuntimeError(
            f"Frozen-evidence assertion failed for {name}: "
            f"got {got}, expected {expected}"
        )


def savefig(fig, stem):
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    print(f"saved: {pdf}")
    print(f"saved: {png}")
    return [pdf, png]


# ============================================================
# Load frozen inputs
# ============================================================

canonical_summary = pd.read_csv(CANON / "canonical_lfp_n30_summary.csv")
canonical_raw = read_jsonl(CANON / "canonical_lfp_n30_raw.jsonl")
canonical_stats = pd.read_csv(CANON / "statistics" / "paired_statistics.csv")

thermal_raw = read_jsonl(THERM / "thermal_initial_condition_raw.jsonl")
thermal_stats = pd.read_csv(THERM / "thermal_initial_condition_statistics.csv")

hust_raw = read_jsonl(HUST / "hust_timestamp_replay_v2_raw.jsonl")
hust_stats = pd.read_csv(HUST / "hust_timestamp_replay_v2_statistics.csv")
hust_summary = read_json(HUST / "hust_timestamp_replay_v2_summary.json")

action = read_json(ACTION_JSON)
ranking = read_json(RANK_JSON)


# ============================================================
# Frozen evidence assertions
# ============================================================

def canonical_value(controller, column):
    return canonical_summary.loc[
        canonical_summary["controller"] == controller, column
    ].iloc[0]


assert_close(
    "canonical Graph time",
    canonical_value("GraphOptimizer", "time_mean_min"),
    21.6333333333333,
)
assert_close(
    "canonical Physics time",
    canonical_value("Physics-CEM", "time_mean_min"),
    25.0333333333333,
)
assert_close(
    "canonical Graph final dT",
    canonical_value("GraphOptimizer", "final_dT_mean_C"),
    0.155344,
    atol=2e-6,
)
assert_close(
    "canonical Physics final dT",
    canonical_value("Physics-CEM", "final_dT_mean_C"),
    0.115964,
    atol=2e-6,
)

assert_close(
    "HUST Graph time",
    hust_summary["GraphOptimizer"]["mean_time_min"],
    21.5,
)
assert_close(
    "HUST Physics time",
    hust_summary["Physics-CEM"]["mean_time_min"],
    24.933333333333334,
)
assert_close(
    "HUST raw violations",
    hust_summary["HUST-Raw"]["total_explicit_study_violations"],
    903,
)

assert_close(
    "action normal SOC MAE",
    action["summary"]["normal"]["soc_mae_percent"],
    0.08986631943569373,
)
assert_close(
    "ranking shared rho mean",
    ranking["summary"]["rho_shared"]["mean"],
    0.5848489010989011,
)
assert_close(
    "ranking full rho mean",
    ranking["summary"]["rho_full"]["mean"],
    0.8325137362637363,
)

print("Frozen evidence assertions: PASS")


# ============================================================
# Figure 1
# Manually prepared publication schematic
# ============================================================

manual_fig1 = OUT / "fig1_controller_architecture.png"

if not manual_fig1.exists():
    raise FileNotFoundError(
        "Manual publication Figure 1 is missing: "
        f"{manual_fig1}"
    )

print(
    "Figure 1: preserving manually prepared publication PNG "
    "(not regenerated)."
)

outputs = [manual_fig1]


# ============================================================
# Figure 2
# Canonical paired speed–thermal trade-off
# ============================================================

sub = canonical_raw[
    canonical_raw["controller"].isin(["GraphOptimizer", "Physics-CEM"])
].copy()

pt = sub.pivot(
    index="seed",
    columns="controller",
    values=["charging_time_min", "final_T_gradient_C", "cumulative_aging"],
)

dx = (
    pt["charging_time_min"]["GraphOptimizer"]
    - pt["charging_time_min"]["Physics-CEM"]
)
dy = (
    pt["final_T_gradient_C"]["GraphOptimizer"]
    - pt["final_T_gradient_C"]["Physics-CEM"]
)
da = (
    pt["cumulative_aging"]["GraphOptimizer"]
    - pt["cumulative_aging"]["Physics-CEM"]
)

if len(dx) != 30:
    raise RuntimeError(f"Expected 30 canonical pairs, got {len(dx)}")

fig, ax = plt.subplots(figsize=(6.6, 4.6))

ax.scatter(dx, dy, s=33, alpha=0.78, label="Paired canonical seeds")
ax.axvline(0, linestyle="--", linewidth=0.9)
ax.axhline(0, linestyle="--", linewidth=0.9)

ax.scatter(
    [dx.mean()],
    [dy.mean()],
    marker="X",
    s=120,
    label="Mean paired difference",
    zorder=5,
)

ax.set_xlabel(
    "Charging-time difference, GraphOptimizer - Physics-CEM (min)"
)
ax.set_ylabel(
    "Final thermal-gradient difference, GraphOptimizer - Physics-CEM (°C)"
)
ax.set_title("Canonical LFP: paired speed–thermal trade-off (N=30)")
ax.legend(frameon=False)

relative_ageing = (
    100
    * canonical_value("GraphOptimizer", "aging_mean")
    / canonical_value("Physics-CEM", "aging_mean")
    - 100
)

ax.text(
    0.03, 0.97,
    f"Mean Δtime = {dx.mean():.2f} min\n"
    f"Mean Δthermal = +{dy.mean():.3f} °C\n"
    f"Ageing proxy = {relative_ageing:.2f}%",
    transform=ax.transAxes,
    va="top",
)

outputs += savefig(fig, "fig2_canonical_paired_tradeoff")


# ============================================================
# Figure 3
# Thermal initial-condition controlled diagnostic
# ============================================================

td = thermal_raw[
    thermal_raw["controller"].isin(["GraphOptimizer", "Physics-CEM"])
].pivot_table(
    index=["seed", "condition"],
    columns="controller",
    values="final_T_gradient_C",
)

td["Graph_minus_Physics"] = (
    td["GraphOptimizer"] - td["Physics-CEM"]
)

tw = td["Graph_minus_Physics"].reset_index().pivot(
    index="seed",
    columns="condition",
    values="Graph_minus_Physics",
)

expected_conditions = ["heterogeneous_T", "uniform_25C"]
for c in expected_conditions:
    if c not in tw.columns:
        raise RuntimeError(f"Missing thermal condition: {c}")

fig, ax = plt.subplots(figsize=(5.9, 4.6))

x = np.array([0.0, 1.0])

for _, row in tw.iterrows():
    ax.plot(
        x,
        [row["heterogeneous_T"], row["uniform_25C"]],
        marker="o",
        markersize=2.8,
        linewidth=0.7,
        alpha=0.18,
    )

means = []
low = []
high = []

for key in [
    "Graph-Physics final dT | heterogeneous_T",
    "Graph-Physics final dT | uniform_25C",
]:
    r = thermal_stats.loc[
        thermal_stats["comparison"] == key
    ].iloc[0]
    means.append(r["mean_difference_C"])
    low.append(r["bootstrap95_low_C"])
    high.append(r["bootstrap95_high_C"])

means = np.asarray(means)
yerr = np.vstack([
    means - np.asarray(low),
    np.asarray(high) - means
])

ax.errorbar(
    x,
    means,
    yerr=yerr,
    fmt="X",
    markersize=8,
    capsize=4,
    linewidth=1.5,
    label="Mean paired difference ± 95% bootstrap CI",
    zorder=5,
)

ax.axhline(0, linestyle="--", linewidth=0.9)
ax.set_xticks(x, ["Heterogeneous T", "Uniform 25°C"])
ax.set_ylabel(
    "Final thermal-gradient difference\nGraphOptimizer - Physics-CEM (°C)"
)
ax.set_title("Controlled thermal-initialisation diagnostic (N=30)")
ax.legend(frameon=False, loc="best")

interaction = thermal_stats.loc[
    thermal_stats["comparison"].str.startswith("interaction:")
].iloc[0]

ax.text(
    0.03, 0.97,
    f"Interaction = {interaction['mean_difference_C']:+.4f} °C\n"
    f"p = {interaction['wilcoxon_p']:.3f}",
    transform=ax.transAxes,
    va="top",
)

outputs += savefig(fig, "fig3_thermal_initial_condition")


# ============================================================
# Figure 4
# HUST paired speed–thermal trade-off
# ============================================================

hs = hust_raw[
    hust_raw["arm"].isin(["GraphOptimizer", "Physics-CEM"])
].copy()

hp = hs.pivot(
    index="episode",
    columns="arm",
    values=["charging_time_min", "final_T_gradient_C", "cumulative_aging"],
)

hx = (
    hp["charging_time_min"]["GraphOptimizer"]
    - hp["charging_time_min"]["Physics-CEM"]
)
hy = (
    hp["final_T_gradient_C"]["GraphOptimizer"]
    - hp["final_T_gradient_C"]["Physics-CEM"]
)

if len(hx) != 30:
    raise RuntimeError(f"Expected 30 HUST pairs, got {len(hx)}")

fig, ax = plt.subplots(figsize=(6.6, 4.6))

ax.scatter(hx, hy, s=33, alpha=0.78, label="Paired HUST replay episodes")
ax.axvline(0, linestyle="--", linewidth=0.9)
ax.axhline(0, linestyle="--", linewidth=0.9)

ax.scatter(
    [hx.mean()],
    [hy.mean()],
    marker="X",
    s=120,
    label="Mean paired difference",
    zorder=5,
)

ax.set_xlabel(
    "Charging-time difference, GraphOptimizer - Physics-CEM (min)"
)
ax.set_ylabel(
    "Final thermal-gradient difference, GraphOptimizer - Physics-CEM (°C)"
)
ax.set_title("Held-out HUST protocol replay: paired trade-off (N=30)")
ax.legend(frameon=False)

hust_age_rel = (
    100
    * hust_summary["GraphOptimizer"]["mean_aging"]
    / hust_summary["Physics-CEM"]["mean_aging"]
    - 100
)

ax.text(
    0.03, 0.97,
    f"Mean Δtime = {hx.mean():.2f} min\n"
    f"Mean Δthermal = +{hy.mean():.3f} °C\n"
    f"Ageing proxy = {hust_age_rel:.2f}%\n"
    f"HUST-Raw reference: 903 explicit violations",
    transform=ax.transAxes,
    va="top",
)

outputs += savefig(fig, "fig4_hust_paired_tradeoff")


# ============================================================
# Figure 5
# Candidate-ranking fidelity across SOC
# ============================================================

states = pd.DataFrame(ranking["states"]).sort_values("soc_init")

fig, ax = plt.subplots(figsize=(6.6, 4.6))

ax.plot(
    states["soc_init"],
    states["rho_shared"],
    marker="o",
    markersize=4,
    label=(
        "Shared objective "
        f"(mean ρ={ranking['summary']['rho_shared']['mean']:.3f})"
    ),
)

ax.plot(
    states["soc_init"],
    states["rho_full"],
    marker="s",
    markersize=4,
    label=(
        "Complete physics objective "
        f"(mean ρ={ranking['summary']['rho_full']['mean']:.3f})"
    ),
)

ax.axhline(0, linestyle="--", linewidth=0.9)

ax.set_xlabel("Initial SOC used for ranking diagnostic")
ax.set_ylabel("Spearman rank correlation, ρ")
ax.set_title("PackGNN v2 candidate-ranking fidelity")
ax.legend(frameon=False)

ax.text(
    0.03, 0.04,
    "PackGNN has no voltage output head;\n"
    "complete-physics ranking agreement is not a voltage-prediction claim.",
    transform=ax.transAxes,
    va="bottom",
)

outputs += savefig(fig, "fig5_surrogate_ranking_fidelity")


# ============================================================
# Supplementary Figure S1
# Action-conditioning intervention
# ============================================================

s = action["summary"]

conditions = ["Correct action", "Zero action", "Shuffled action"]

soc = np.array([
    s["normal"]["soc_mae_percent"],
    s["zero"]["soc_mae_percent"],
    s["shuffled"]["soc_mae_percent"],
])

dt = np.array([
    s["normal"]["dT_mae_C"],
    s["zero"]["dT_mae_C"],
    s["shuffled"]["dT_mae_C"],
])

soc_ratio = soc / soc[0]
dt_ratio = dt / dt[0]

x = np.arange(len(conditions))
width = 0.36

fig, ax = plt.subplots(figsize=(6.2, 4.5))

ax.bar(
    x - width/2,
    soc_ratio,
    width,
    label="SOC MAE / correct-action SOC MAE",
)
ax.bar(
    x + width/2,
    dt_ratio,
    width,
    label="ΔT MAE / correct-action ΔT MAE",
)

ax.set_yscale("log")
ax.set_ylabel("MAE relative to correct-action condition (×)")
ax.set_xticks(x, conditions)
ax.set_title("Held-out action-conditioning intervention (2,901 samples)")
ax.legend(frameon=False)

outputs += savefig(fig, "figS1_action_conditioning")


# ============================================================
# Draft captions / LaTeX snippets
# ============================================================

captions = r"""
% ============================================================
% Draft figure insertions for the final v2 manuscript
% Review placement before copying into main.tex.
% ============================================================

% FIGURE 1
\begin{figure*}[t]
\centering
\includegraphics[width=0.94\textwidth]{figures/submission_v2/fig1_controller_architecture.png}
\caption{Final decision architecture for GraphOptimizer and the matched Physics-CEM comparator. Both controllers use the same CEM horizon and sampling budget, 60/40 base-action to SOC-deficit blend, actuator projection, and final one-step model-based safety verification. GraphOptimizer uses action-conditioned PackGNN v2 for candidate ranking and SOC prediction, whereas Physics-CEM uses direct ECM/thermal rollout.}
\label{fig:controller_architecture}
\end{figure*}

% FIGURE 2
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/submission_v2/fig2_canonical_paired_tradeoff.pdf}
\caption{Paired canonical LFP trade-off for GraphOptimizer relative to matched Physics-CEM across 30 identical initial-pack seeds. Negative horizontal values indicate faster GraphOptimizer charging, whereas positive vertical values indicate a larger final inter-cell thermal gradient. The cross marks the mean paired difference.}
\label{fig:canonical_tradeoff}
\end{figure}

% FIGURE 3
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/submission_v2/fig3_thermal_initial_condition.pdf}
\caption{Controlled thermal-initialisation diagnostic. Each line connects the GraphOptimizer-minus-Physics-CEM final thermal-gradient difference for the same diagnostic pack seed under heterogeneous and uniform 25~$^\circ$C initial cell temperatures. Crosses show paired means with 95\% bootstrap confidence intervals. The thermal-gradient disadvantage persists under uniform initialisation, while the interaction is not statistically significant.}
\label{fig:thermal_initial_condition}
\end{figure}

% FIGURE 4
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/submission_v2/fig4_hust_paired_tradeoff.pdf}
\caption{Paired GraphOptimizer-minus-Physics-CEM trade-off in the held-out HUST measured-current protocol replay. The directional pattern reproduces the canonical result: shorter charging time for GraphOptimizer is accompanied by a larger final inter-cell thermal gradient. HUST-Raw is an external-protocol reference and is not an actuator-matched controller comparator.}
\label{fig:hust_tradeoff}
\end{figure}

% FIGURE 5
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/submission_v2/fig5_surrogate_ranking_fidelity.pdf}
\caption{Candidate-ranking fidelity of PackGNN v2 over 30 independent diagnostic states and 64 candidates per state. Spearman correlation is reported against the shared surrogate-compatible objective and the complete downstream physics objective. Ranking is generally informative but is not uniform across the full SOC range.}
\label{fig:ranking_fidelity}
\end{figure}

% SUPPLEMENTARY FIGURE S1
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/submission_v2/figS1_action_conditioning.pdf}
\caption{Held-out action-conditioning intervention. Zeroing or shuffling the candidate-action input substantially increases both SOC and temperature-change prediction error relative to the correct-action condition, supporting genuine action sensitivity of PackGNN v2.}
\label{fig:action_conditioning}
\end{figure}
"""

(OUT / "latex_figure_snippets.tex").write_text(captions.strip() + "\n")


# ============================================================
# Figure provenance manifest
# ============================================================
# ============================================================
# Figure provenance manifest
# ============================================================

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

refiner_path = (
    ROOT
    / "src"
    / "refine_submission_figures_v2.py"
)

manifest = {
    "created_utc":
        datetime.now(timezone.utc).isoformat(),

    "frozen_computational_evidence_git_head":
        "a28143ea3bb60081214ae9aa91e9051b1c202e72",

    "generation_pipeline": [
        "src/generate_submission_figures_v2.py",
        "src/refine_submission_figures_v2.py",
    ],

    "generator_sha256": {
        "generate_submission_figures_v2.py":
            sha256(Path(__file__).resolve()),

        "refine_submission_figures_v2.py":
            sha256(refiner_path),
    },

    "figure_policy": {
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
    },

    "inputs": {
        str(p.relative_to(ROOT)): sha256(p)
        for p in input_paths
    },

    "outputs": {
        str(p.relative_to(ROOT)): sha256(p)
        for p in final_output_paths
    },

    "scientific_note":
        (
            "Figures 2-5 and S1 are derived from frozen v2 "
            "computational evidence. Figure 1 is a manually "
            "prepared explanatory architecture schematic. "
            "No experiment was rerun and no frozen "
            "computational-evidence file was modified."
        ),
}

manifest_path = (
    OUT
    / "figure_manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("=" * 72)
print("FINAL FIGURE GENERATION: PASS")
print(
    "frozen evidence head:",
    manifest[
        "frozen_computational_evidence_git_head"
    ],
)
print("output directory:", OUT)
print("figure manifest:", manifest_path)
print(
    "figure manifest sha256:",
    sha256(manifest_path),
)
print("=" * 72)
