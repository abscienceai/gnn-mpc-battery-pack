"""
cost_sensitivity.py
===================
Dynamic Graph-Based Safe Fast Charging Optimization Project
------------------------------------------------------------
PURPOSE : Cost function sensitivity analysis.

Sweeps λ1 (time weight) and λ2 (imbalance weight) over a grid,
runs GraphOptimizer for each combination, records outcomes.

Produces:
  fig_lambda_heatmap.pdf  — 2D heatmap: λ1 vs λ2 → σ_SOC and time
  fig_pareto_extended.pdf — Extended Pareto front with λ points
  lambda_sweep_results.json

USAGE: python cost_sensitivity.py [--n_episodes 10] [--output_dir results/sensitivity]
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from graph_battery_pack import build_pack_from_ecm, PackGNN
from safe_fast_charge_optimizer import (
    GraphGuidedOptimizer, CCCVController,
    default_config, run_episode
)

ECM_DIR = Path("results/ecm")
MODEL_DIR = Path("results/models")

plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


# ═══════════════════════════════════════════════════════════════════════════
#  LAMBDA SWEEP
# ═══════════════════════════════════════════════════════════════════════════

def run_lambda_sweep(n_episodes: int = 10, n_cells: int = 12,
                     output_dir: Path = None) -> dict:
    """
    Grid sweep over λ1 (time) and λ2 (imbalance).
    Other weights fixed: λ3=3 (temp), λ4=2 (aging), λ5=50 (violation).

    Grid: λ1 ∈ {1, 2, 4, 6, 8}, λ2 ∈ {1, 2, 3, 5, 8}
    """
    lambda1_values = [1, 2, 4, 6, 8]   # time weight
    lambda2_values = [1, 2, 3, 5, 8]   # imbalance weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecm_parquet = sorted(ECM_DIR.glob("*.parquet"))
    ecm_parquet = ecm_parquet[-1] if ecm_parquet else None

    # Load trained GNN
    gnn = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(device)
    trained = sorted(MODEL_DIR.glob("pack_gnn_*.pt"))
    if trained:
        ckpt = torch.load(trained[-1], map_location=device)
        gnn.load_state_dict(ckpt["model_state"])
        print(f"  Loaded GNN: {trained[-1].name}")

    results = []
    total = len(lambda1_values) * len(lambda2_values)
    done  = 0

    print(f"\n  Sweeping {total} (λ1, λ2) combinations × {n_episodes} episodes...")

    for l1 in lambda1_values:
        for l2 in lambda2_values:
            cfg = default_config()
            cfg["n_cells"]   = n_cells
            cfg["w_time"]    = float(l1)
            cfg["w_imbalance"] = float(l2)
            # Keep others fixed
            cfg["w_temperature"] = 3.0
            cfg["w_aging"]       = 2.0
            cfg["w_violation"]   = 50.0

            ep_results = []
            for ep in range(n_episodes):
                np.random.seed(ep * 17)
                torch.manual_seed(ep * 17)
                pack = build_pack_from_ecm(
                    ecm_parquet, n_cells=n_cells, chemistry="LFP",
                    soc_init=0.20, soc_noise=0.03)
                ctrl = GraphGuidedOptimizer(cfg, gnn)
                res  = run_episode(pack, ctrl, cfg,
                                    f"λ1={l1},λ2={l2}", verbose=False)
                ep_results.append(res)

            soc_imb  = np.mean([r["final_SOC_imbalance"] for r in ep_results])
            time_min = np.mean([r["charging_time_min"]   for r in ep_results])
            T_grad   = np.mean([r["final_T_gradient"]    for r in ep_results])
            aging    = np.mean([r["cumulative_aging"]     for r in ep_results])
            viol     = np.mean([r["total_violations"]     for r in ep_results])

            results.append({
                "lambda1":      l1,
                "lambda2":      l2,
                "soc_imbalance": round(float(soc_imb),  5),
                "time_min":      round(float(time_min), 2),
                "T_gradient":    round(float(T_grad),   4),
                "aging":         round(float(aging),    6),
                "violations":    round(float(viol),     2),
            })

            done += 1
            print(f"  [{done:2d}/{total}] λ1={l1} λ2={l2} → "
                  f"σ={soc_imb*100:.2f}% t={time_min:.1f}min "
                  f"ΔT={T_grad:.3f}°C", flush=True)

    # Mark the paper's chosen weights
    chosen = {"lambda1": 4, "lambda2": 3}
    for r in results:
        r["is_chosen"] = (r["lambda1"] == chosen["lambda1"] and
                          r["lambda2"] == chosen["lambda2"])

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = output_dir / f"lambda_sweep_{ts}.json"
        with open(out, "w") as f:
            json.dump({
                "results":         results,
                "chosen_weights":  chosen,
                "lambda1_values":  lambda1_values,
                "lambda2_values":  lambda2_values,
                "n_episodes":      n_episodes,
            }, f, indent=2)
        print(f"\n  ✅ Saved → {out.name}")

    return results, lambda1_values, lambda2_values


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 1 — 2D Heatmaps (λ1 × λ2 → σ_SOC and time)
# ═══════════════════════════════════════════════════════════════════════════

def fig_lambda_heatmap(results, l1_vals, l2_vals, out_dir: Path, fmt: str):
    n1, n2 = len(l1_vals), len(l2_vals)

    # Build matrices
    soc_mat  = np.zeros((n1, n2))
    time_mat = np.zeros((n1, n2))
    dT_mat   = np.zeros((n1, n2))

    for r in results:
        i = l1_vals.index(r["lambda1"])
        j = l2_vals.index(r["lambda2"])
        soc_mat[i, j]  = r["soc_imbalance"] * 100
        time_mat[i, j] = r["time_min"]
        dT_mat[i, j]   = r["T_gradient"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, mat, title, cmap, unit in zip(
        axes,
        [soc_mat, time_mat, dT_mat],
        ["(a) SOC Imbalance σ (%)",
         "(b) Charging Time (min)",
         "(c) Thermal Gradient ΔT (°C)"],
        ["RdYlGn_r", "RdYlGn", "RdYlGn_r"],
        ["%", "min", "°C"],
    ):
        im = ax.imshow(mat, cmap=cmap, aspect="auto",
                       origin="lower", interpolation="nearest")
        plt.colorbar(im, ax=ax, label=unit, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n2))
        ax.set_xticklabels([str(v) for v in l2_vals])
        ax.set_yticks(range(n1))
        ax.set_yticklabels([str(v) for v in l1_vals])
        ax.set_xlabel("λ₂ (imbalance weight)")
        ax.set_ylabel("λ₁ (time weight)")
        ax.set_title(title)

        # Annotate cells
        for i in range(n1):
            for j in range(n2):
                is_chosen = (l1_vals[i] == 4 and l2_vals[j] == 3)
                txt = f"{mat[i,j]:.2f}"
                color = "white" if is_chosen else "black"
                weight = "bold" if is_chosen else "normal"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=color, fontweight=weight)
                if is_chosen:
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                          fill=False, edgecolor="blue",
                                          linewidth=2.5)
                    ax.add_patch(rect)

    fig.suptitle("Cost Function Sensitivity Analysis: "
                 "λ₁ (time) × λ₂ (imbalance) Weight Sweep\n"
                 "Blue box = paper's chosen weights (λ₁=4, λ₂=3)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    p = out_dir / f"fig_lambda_heatmap.{fmt}"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {p.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 2 — Extended Pareto Front
# ═══════════════════════════════════════════════════════════════════════════

def fig_pareto_extended(results, out_dir: Path, fmt: str,
                         optimizer_results_dir: Path = None):
    """
    Extended Pareto front showing:
    - λ sweep points (coloured by λ2 value)
    - CC-CV, Proportional baselines
    - Paper's chosen point highlighted
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Plot 1: Time vs σ_SOC ──
    ax = axes[0]
    l2_unique = sorted(set(r["lambda2"] for r in results))
    cmap      = cm.get_cmap("viridis", len(l2_unique))

    for r in results:
        l2_idx = l2_unique.index(r["lambda2"])
        color  = cmap(l2_idx / max(len(l2_unique)-1, 1))
        marker = "D" if r["is_chosen"] else "o"
        size   = 120 if r["is_chosen"] else 50
        zorder = 5 if r["is_chosen"] else 3
        ax.scatter(r["time_min"], r["soc_imbalance"]*100,
                   c=[color], s=size, marker=marker,
                   edgecolors="black" if r["is_chosen"] else "white",
                   linewidths=1.5 if r["is_chosen"] else 0.5,
                   zorder=zorder, alpha=0.85)

    # Label chosen point
    chosen_r = next((r for r in results if r["is_chosen"]), None)
    if chosen_r:
        ax.annotate("Paper's\nchosen λ\n(λ₁=4, λ₂=3)",
                     xy=(chosen_r["time_min"],
                         chosen_r["soc_imbalance"]*100),
                     xytext=(chosen_r["time_min"]+3,
                             chosen_r["soc_imbalance"]*100+0.1),
                     fontsize=9, color="blue", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

    # Baselines from optimizer results if available
    baseline_data = {}
    if optimizer_results_dir:
        opt_files = sorted(optimizer_results_dir.glob("experiment_results_*.json"))
        if opt_files:
            with open(opt_files[-1]) as f:
                opt_data = json.load(f)
            agg = opt_data.get("aggregate", {})
            for ctrl_name, color, marker in [
                ("CC-CV",        "#e74c3c", "^"),
                ("Proportional", "#3498db", "s"),
            ]:
                if ctrl_name in agg:
                    t = agg[ctrl_name]["charging_time_min_mean"]
                    s = agg[ctrl_name]["final_SOC_imbalance_mean"] * 100
                    ax.scatter(t, s, c=color, s=150, marker=marker,
                               edgecolors="black", linewidths=1.2,
                               zorder=6, label=ctrl_name)
                    baseline_data[ctrl_name] = (t, s)

    # Colorbar for λ2
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=Normalize(vmin=min(l2_unique),
                                               vmax=max(l2_unique)))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax)
    cb.set_label("λ₂ (imbalance weight)")

    ax.set_xlabel("Charging Time (min)  [lower = better →]")
    ax.set_ylabel("SOC Imbalance σ (%)  [lower = better ↓]")
    ax.set_title("(a) Pareto Front: Speed vs Balance\n"
                 "Each point = one (λ₁, λ₂) configuration")

    # Pareto-optimal frontier annotation
    ax.annotate("",
                 xy=(ax.get_xlim()[0]+2, ax.get_ylim()[0]+0.05),
                 xytext=(ax.get_xlim()[0]+8, ax.get_ylim()[0]+0.3),
                 arrowprops=dict(arrowstyle="<-", color="grey",
                                  lw=1.5, ls="dashed"))
    ax.text(ax.get_xlim()[0]+3, ax.get_ylim()[0]+0.2,
             "Pareto\nfrontier", fontsize=8, color="grey", style="italic")

    if baseline_data:
        ax.legend(loc="upper right", fontsize=9)

    # ── Plot 2: λ1/λ2 ratio vs outcome ──
    ax2 = axes[1]
    ratios = [r["lambda1"] / r["lambda2"] for r in results]
    socs   = [r["soc_imbalance"]*100 for r in results]
    times  = [r["time_min"] for r in results]

    ax2.scatter(ratios, socs,  color="#e74c3c", s=60, alpha=0.7,
                label="σ_SOC (%)", marker="o")
    ax2_t = ax2.twinx()
    ax2_t.scatter(ratios, times, color="#2980b9", s=60, alpha=0.7,
                   label="Time (min)", marker="s")

    # Mark chosen
    chosen_ratio = 4/3
    if chosen_r:
        ax2.axvline(chosen_ratio, color="blue", ls="--", lw=1.5,
                     label=f"Chosen λ₁/λ₂={chosen_ratio:.2f}")

    ax2.set_xlabel("λ₁ / λ₂ Ratio (time/imbalance priority)")
    ax2.set_ylabel("SOC Imbalance σ (%)", color="#e74c3c")
    ax2_t.set_ylabel("Charging Time (min)", color="#2980b9")
    ax2.set_title("(b) Weight Ratio vs Outcome\n"
                  "Higher ratio = more speed priority")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_t.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
                loc="center right", fontsize=9)

    fig.suptitle("Cost Function Sensitivity: "
                 "Impact of λ₁ (time) and λ₂ (imbalance) Weights\n"
                 "on Charging Performance",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    p = out_dir / f"fig_pareto_extended.{fmt}"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {p.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cost function sensitivity analysis.")
    parser.add_argument("--n_episodes",  type=int, default=10)
    parser.add_argument("--n_cells",     type=int, default=12)
    parser.add_argument("--output_dir",  default="results/sensitivity")
    parser.add_argument("--figures_dir", default="results/figures")
    parser.add_argument("--format",      default="pdf",
                        choices=["pdf","png","svg"])
    parser.add_argument("--skip_sweep",  action="store_true",
                        help="Skip sweep, only plot from saved JSON")
    args = parser.parse_args()

    out_dir  = Path(args.output_dir)
    fig_dir  = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Cost Function Sensitivity Analysis")
    print("=" * 60)

    if args.skip_sweep:
        # Load existing results
        saved = sorted(out_dir.glob("lambda_sweep_*.json"))
        if not saved:
            print("[ERROR] No saved sweep found. Run without --skip_sweep.")
            return
        with open(saved[-1]) as f:
            d = json.load(f)
        results     = d["results"]
        l1_vals     = d["lambda1_values"]
        l2_vals     = d["lambda2_values"]
        print(f"  Loaded: {saved[-1].name} ({len(results)} configs)")
    else:
        results, l1_vals, l2_vals = run_lambda_sweep(
            n_episodes  = args.n_episodes,
            n_cells     = args.n_cells,
            output_dir  = out_dir,
        )

    print("\n── Generating figures ──────────────────────────────────")
    fig_lambda_heatmap(results, l1_vals, l2_vals, fig_dir, args.format)
    fig_pareto_extended(
        results, fig_dir, args.format,
        optimizer_results_dir=Path("results/optimizer_calibrated")
    )

    # Summary: what's optimal?
    print("\n── Optimal λ configurations ────────────────────────────")
    sorted_by_soc  = sorted(results, key=lambda r: r["soc_imbalance"])
    sorted_by_time = sorted(results, key=lambda r: r["time_min"])
    print(f"  Best SOC balance : λ1={sorted_by_soc[0]['lambda1']} "
          f"λ2={sorted_by_soc[0]['lambda2']} → "
          f"σ={sorted_by_soc[0]['soc_imbalance']*100:.3f}% "
          f"t={sorted_by_soc[0]['time_min']:.1f}min")
    print(f"  Fastest charging : λ1={sorted_by_time[0]['lambda1']} "
          f"λ2={sorted_by_time[0]['lambda2']} → "
          f"σ={sorted_by_time[0]['soc_imbalance']*100:.3f}% "
          f"t={sorted_by_time[0]['time_min']:.1f}min")

    chosen = next((r for r in results if r.get("is_chosen")), None)
    if chosen:
        print(f"  Paper's choice   : λ1={chosen['lambda1']} "
              f"λ2={chosen['lambda2']} → "
              f"σ={chosen['soc_imbalance']*100:.3f}% "
              f"t={chosen['time_min']:.1f}min  ← balanced trade-off")

    print(f"\n  ✅ Figures saved → {fig_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
