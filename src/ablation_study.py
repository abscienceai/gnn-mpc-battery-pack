"""
ablation_study.py
=================
Dynamic Graph-Based Safe Fast Charging Optimization Project
------------------------------------------------------------
PURPOSE : Ablation study — quantify contribution of each component.

VARIANTS:
  1. Full Model        : GNN + MPC (H=5) + CEM (K=128) + Graph edges
  2. No Graph Edges    : Node features only, no edge message passing
  3. No MPC Horizon    : H=1 (greedy, single-step lookahead)
  4. No CEM            : Random action selection (K=1)
  5. Rule-Based Only   : CC-CV baseline (no learning/optimization)

This directly answers: "Which component contributes what?"

USAGE: python ablation_study.py [--n_episodes 30] [--n_cells 12]
                                 [--output_dir results/ablation_components]
"""

import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from graph_battery_pack import (
    BatteryPackGraph, PackGNN, CellState,
    build_pack_from_ecm
)
from safe_fast_charge_optimizer import (
    CCCVController, default_config, run_episode
)

ECM_DIR = Path("results/ecm")


# ═══════════════════════════════════════════════════════════════════════════
#  ABLATION VARIANT 1: No Graph Edges (node-only MLP)
# ═══════════════════════════════════════════════════════════════════════════

class NodeOnlyGNN(nn.Module):
    """
    Ablation: No message passing — each node processed independently.
    Removes graph structure; equivalent to per-cell MLP.
    """
    def __init__(self, node_feat: int = 7, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_feat, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.soc_head   = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.temp_head  = nn.Linear(hidden, 1)
        self.aging_head = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.pack_head  = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x, edge_index, edge_attr):
        # Ignore edges completely
        h = self.mlp(x)
        return {
            "soc_pred":     self.soc_head(h),
            "delta_T_pred": self.temp_head(h),
            "aging_pred":   self.aging_head(h),
            "imbalance":    self.pack_head(h.mean(0, keepdim=True)).squeeze(),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  ABLATION VARIANT 2: No MPC (greedy, H=1)
# ═══════════════════════════════════════════════════════════════════════════

class GreedyOptimizer:
    """
    Ablation: No multi-step horizon (H=1 greedy).
    CEM still used but only evaluates immediate next step.
    """
    def __init__(self, cfg: dict, gnn):
        self.cfg = cfg
        self.gnn = gnn
        self.device = next(gnn.parameters()).device

    def reset(self): pass

    def get_currents(self, pack: BatteryPackGraph) -> np.ndarray:
        n = pack.n_cells
        Q_nom = np.array([c.Q_nom_Ah for c in pack.cells])
        I_max = self.cfg["I_max_C"] * Q_nom

        n_samples = self.cfg["cem_samples"]
        elite_k   = max(1, int(n_samples * self.cfg["cem_elite_frac"]))
        mu  = I_max * 0.5
        sig = I_max * 0.3

        for _ in range(self.cfg["cem_iterations"]):
            samples = np.random.normal(mu, sig, size=(n_samples, n))
            samples = np.clip(samples, 0, I_max)

            costs = []
            for s in samples:
                pack_copy = deepcopy(pack)
                # H=1: only evaluate one step (no GNN rollout)
                metrics = pack_copy.step(s, dt=self.cfg["dt_s"])
                cost = (self.cfg["w_imbalance"] * metrics["SOC_imbalance"] +
                        self.cfg["w_temperature"] * max(metrics["T_max"]-38,0)/10 +
                        self.cfg["w_time"] * 1.0 +
                        self.cfg["w_violation"] * metrics["n_violations"])
                costs.append(cost)

            elite = samples[np.argsort(costs)[:elite_k]]
            mu  = elite.mean(0)
            sig = elite.std(0) + 1e-6

        return mu.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  ABLATION VARIANT 3: No CEM (random action)
# ═══════════════════════════════════════════════════════════════════════════

class RandomActionOptimizer:
    """
    Ablation: No CEM — random current allocation.
    Tests whether optimisation matters vs random exploration.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def reset(self): pass

    def get_currents(self, pack: BatteryPackGraph) -> np.ndarray:
        Q_nom = np.array([c.Q_nom_Ah for c in pack.cells])
        I_max = self.cfg["I_max_C"] * Q_nom
        # Random uniform between 0 and I_max
        currents = np.random.uniform(0, I_max)
        return currents.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  FULL MODEL (from safe_fast_charge_optimizer)
# ═══════════════════════════════════════════════════════════════════════════

def make_full_optimizer(cfg, device):
    """Full model: trained PackGNN + MPC + CEM."""
    from safe_fast_charge_optimizer import GraphGuidedOptimizer
    from pathlib import Path
    gnn = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(device)
    # Load trained weights
    trained = sorted(Path("results/models").glob("pack_gnn_*.pt"))
    if trained:
        ckpt = torch.load(trained[-1], map_location=device)
        gnn.load_state_dict(ckpt["model_state"])
        print(f"    [Full] Loaded trained GNN: {trained[-1].name}", flush=True)
    return GraphGuidedOptimizer(cfg, gnn)


def make_no_edge_optimizer(cfg, device):
    """No-edge ablation: trained PackGNN but edge_attr zeroed out.
    Same weights, but edge message passing receives no coupling information.
    This is the cleanest ablation of graph structure contribution."""
    from safe_fast_charge_optimizer import GraphGuidedOptimizer
    from pathlib import Path

    class ZeroEdgePackGNN(PackGNN):
        """PackGNN with zeroed edge features — no inter-cell coupling."""
        def forward(self, x, edge_index, edge_attr):
            # Zero out edge attributes → message passing gets no coupling info
            return super().forward(x, edge_index,
                                   torch.zeros_like(edge_attr))

    gnn = ZeroEdgePackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(device)
    trained = sorted(Path("results/models").glob("pack_gnn_*.pt"))
    if trained:
        ckpt = torch.load(trained[-1], map_location=device)
        gnn.load_state_dict(ckpt["model_state"])
        print(f"    [NoEdge] Loaded trained GNN (edges zeroed): {trained[-1].name}", flush=True)
    return GraphGuidedOptimizer(cfg, gnn)


# ═══════════════════════════════════════════════════════════════════════════
#  RUN ABLATION
# ═══════════════════════════════════════════════════════════════════════════

def run_ablation(n_cells: int = 12, n_episodes: int = 30,
                 chemistry: str = "LFP",
                 output_dir: Path = None) -> dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecm_parquet = sorted(ECM_DIR.glob("*.parquet"))
    ecm_parquet = ecm_parquet[-1] if ecm_parquet else None

    cfg = default_config()
    cfg["n_cells"] = n_cells
    cfg["chemistry"] = chemistry

    # Define ablation variants
    variants = {
        "Full Model\n(GNN+MPC+CEM+Graph)": lambda: make_full_optimizer(cfg, device),
        "No Graph Edges\n(Node-only MLP)":  lambda: make_no_edge_optimizer(cfg, device),
        "No MPC Horizon\n(Greedy H=1)":     lambda: GreedyOptimizer(cfg,
                                                PackGNN(7,3,64,3).to(device)),
        "No CEM\n(Random Action)":           lambda: RandomActionOptimizer(cfg),
        "CC-CV\n(Rule-based)":              lambda: CCCVController(cfg),
    }

    all_results = {}
    print(f"\n{'='*65}")
    print(f"  Component Ablation Study — {chemistry} | {n_cells} cells | {n_episodes} ep")
    print(f"{'='*65}")

    for variant_name, make_ctrl in variants.items():
        label = variant_name.replace("\n", " ")
        print(f"\n  ▶ {label}")
        ep_results = []

        for ep in range(n_episodes):
            print(f"    ep {ep+1}/{n_episodes}...", flush=True)
            np.random.seed(ep * 13)
            torch.manual_seed(ep * 13)

            pack = build_pack_from_ecm(
                ecm_parquet, n_cells=n_cells,
                chemistry=chemistry, soc_init=0.20, soc_noise=0.03
            )
            ctrl = make_ctrl()
            res  = run_episode(pack, ctrl, cfg, label, verbose=False)
            ep_results.append(res)

        # Aggregate
        def ms(key):
            vals = [r[key] for r in ep_results]
            return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

        summary = {
            "n_episodes":        n_episodes,
            "time_min":          ms("charging_time_min"),
            "soc_imbalance":     ms("final_SOC_imbalance"),
            "T_max":             ms("final_T_max"),
            "T_gradient":        ms("final_T_gradient"),
            "aging":             ms("cumulative_aging"),
            "violations":        ms("total_violations"),
        }
        all_results[label] = summary

        si = summary["soc_imbalance"]
        tg = summary["T_gradient"]
        tm = summary["time_min"]
        vl = summary["violations"]
        print(f"    σ_SOC={si[0]*100:.3f}% | ΔT={tg[0]:.3f}°C | "
              f"t={tm[0]:.1f}min | viol={vl[0]:.1f}")

    # Print comparison table
    print(f"\n  {'Variant':<35} {'σ_SOC%':>8} {'ΔT°C':>7} {'Time':>7} {'Viol':>6}")
    print(f"  {'-'*65}")
    full_si = all_results["Full Model (GNN+MPC+CEM+Graph)"]["soc_imbalance"][0]
    for name, res in all_results.items():
        si = res["soc_imbalance"][0]
        tg = res["T_gradient"][0]
        tm = res["time_min"][0]
        vl = res["violations"][0]
        delta = f"(+{(si-full_si)*100:.2f}%)" if si > full_si else ""
        print(f"  {name[:35]:<35} {si*100:>7.3f} {tg:>7.3f} {tm:>7.1f} {vl:>6.1f} {delta}")

    # Compute relative degradation vs full model
    degradation = {}
    full = all_results["Full Model (GNN+MPC+CEM+Graph)"]
    for name, res in all_results.items():
        if name == "Full Model (GNN+MPC+CEM+Graph)":
            continue
        si_full = full["soc_imbalance"][0]
        si_abl  = res["soc_imbalance"][0]
        deg = (si_abl - si_full) / max(si_full, 1e-6) * 100
        degradation[name] = {
            "soc_imbalance_degradation_pct": round(deg, 1),
            "T_gradient_change": round(
                res["T_gradient"][0] - full["T_gradient"][0], 3),
        }
        print(f"  → Removing '{name}': σ_SOC +{deg:.1f}%")

    result = {
        "variants":    all_results,
        "degradation": degradation,
        "chemistry":   chemistry,
        "n_cells":     n_cells,
        "n_episodes":  n_episodes,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = output_dir / f"ablation_components_{ts}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  ✅ Saved → {out}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  LATEX TABLE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_latex_table(result: dict) -> str:
    variants  = result["variants"]
    chemistry = result["chemistry"]
    n_ep      = result["n_episodes"]

    rows = []
    for name, res in variants.items():
        si = res["soc_imbalance"]
        tg = res["T_gradient"]
        tm = res["time_min"]
        vl = res["violations"]

        # Mark full model bold
        is_full = "Full" in name
        fmt = lambda x, prec: (f"\\textbf{{{x:.{prec}f}}}" if is_full
                                else f"{x:.{prec}f}")

        rows.append(
            f"  {name.replace(chr(10),' '):<38} & "
            f"{fmt(si[0]*100,3)}"
            f"{{\\tiny$\\pm${si[1]*100:.3f}}} & "
            f"{fmt(tg[0],3)}"
            f"{{\\tiny$\\pm${tg[1]:.3f}}} & "
            f"{fmt(tm[0],1)}"
            f"{{\\tiny$\\pm${tm[1]:.1f}}} & "
            f"{fmt(vl[0],1)} \\\\"
        )

    table = (
        f"% Ablation Study Table — {chemistry}\n"
        f"\\begin{{table}}[!t]\n"
        f"\\caption{{Component Ablation Study ({chemistry}, $N={n_ep}$ episodes)}}\n"
        f"\\label{{tab:ablation_components}}\n"
        f"\\centering\n"
        f"\\renewcommand{{\\arraystretch}}{{1.2}}\n"
        f"\\setlength{{\\tabcolsep}}{{4pt}}\n"
        f"\\begin{{tabular}}{{lrrrr}}\n"
        f"\\toprule\n"
        f"Variant & $\\sigma_{{\\SOC}}$ (\\%) & $\\Delta T$ ($^\\circ$C) "
        f"& Time (min) & Viol. \\\\\n"
        f"\\midrule\n"
        + "\n".join(rows) + "\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )
    return table


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Component ablation study.")
    parser.add_argument("--n_episodes", type=int, default=30)
    parser.add_argument("--n_cells",    type=int, default=12)
    parser.add_argument("--chemistry",  type=str, default="LFP")
    parser.add_argument("--output_dir", type=str,
                        default="results/ablation_components")
    args = parser.parse_args()

    result = run_ablation(
        n_cells    = args.n_cells,
        n_episodes = args.n_episodes,
        chemistry  = args.chemistry,
        output_dir = Path(args.output_dir),
    )

    # Generate and save LaTeX table
    latex = generate_latex_table(result)
    tex_path = Path(args.output_dir) / "ablation_table.tex"
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"\n  ✅ LaTeX table → {tex_path}")
    print("\n" + latex)


if __name__ == "__main__":
    main()
