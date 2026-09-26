"""
cost_component_breakdown.py
=============================
Reports the raw and weighted contribution of each cost-function component
(time, SOC imbalance, peak-temperature comfort, thermal gradient, aging
proxy, violations) for a representative GraphOptimizer episode, averaged
across control steps. Addresses the reviewer's request to make explicit
which terms actually drive the "safety-first" operating point given the
very different numerical scales of the raw quantities.
"""
import sys, csv, numpy as np, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from graph_battery_pack import PackGNN, build_pack_from_ecm
from safe_fast_charge_optimizer import default_config, GraphGuidedOptimizer

ECM_DIR = Path(__file__).parent.parent / "results/ecm"
MDL_DIR = Path(__file__).parent.parent / "results/models"
OUT_DIR = Path(__file__).parent.parent / "results"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(str(sorted(MDL_DIR.glob("pack_gnn_*.pt"))[-1]), map_location=DEVICE)
gnn  = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(DEVICE)
gnn.load_state_dict(ckpt.get("model_state", ckpt)); gnn.eval()
ecm  = sorted(ECM_DIR.glob("ecm_params_*.parquet"))[-1]
cfg  = default_config()
ctrl = GraphGuidedOptimizer(cfg, gnn)

T_c = cfg.get("T_comfort", 38.0)

N_EP = 30
component_logs = {"time": [], "soc_imb": [], "peak_T": [], "grad_T": [],
                   "aging": [], "viol": []}

for ep in range(N_EP):
    np.random.seed(ep * 7); torch.manual_seed(ep * 7)
    pack = build_pack_from_ecm(ecm_parquet=ecm, n_cells=12, chemistry="LFP",
                                soc_init=cfg["soc_init"], soc_noise=cfg["soc_noise"],
                                T_amb=cfg["T_amb"], seed=ep * 7)
    ctrl.reset()
    steps = 0
    while steps < cfg["max_steps"]:
        socs = np.array([c.SOC for c in pack.cells])
        if np.mean(socs) >= cfg["target_soc"]:
            break
        action = ctrl.get_currents(pack)
        m = pack.step(action, dt=cfg["dt_s"])

        time_term  = 1.0  # indicator: not yet done
        soc_term   = m["SOC_imbalance"]
        Tvals      = np.array([c.T_C for c in pack.cells])
        peak_term  = max(float(np.max(Tvals)) - T_c, 0.0)
        grad_term  = m["T_gradient"]
        aging_term = m["aging_cost"]
        viol_term  = float(m["n_violations"])

        component_logs["time"].append(time_term)
        component_logs["soc_imb"].append(soc_term)
        component_logs["peak_T"].append(peak_term)
        component_logs["grad_T"].append(grad_term)
        component_logs["aging"].append(aging_term)
        component_logs["viol"].append(viol_term)
        steps += 1

lam = {"time": cfg["w_time"], "soc_imb": cfg["w_imbalance"],
       "peak_T": cfg["w_temperature"], "grad_T": cfg["w_gradient"],
       "aging": cfg["w_aging"], "viol": cfg["w_violation"]}

names = {"time": "Time (indicator)", "soc_imb": "SOC imbalance",
         "peak_T": "Peak-temperature comfort", "grad_T": "Thermal gradient",
         "aging": "Aging proxy", "viol": "Violations"}

raw_means = {k: float(np.mean(v)) for k, v in component_logs.items()}
weighted  = {k: lam[k] * raw_means[k] for k in raw_means}
total     = sum(weighted.values())

rows = []
print(f"{'Component':<26} {'lambda':>7} {'Raw mean':>12} {'Weighted':>12} {'% of total':>10}")
print("-" * 72)
for k in ["time", "soc_imb", "peak_T", "grad_T", "aging", "viol"]:
    pct = 100 * weighted[k] / total if total > 0 else 0
    print(f"{names[k]:<26} {lam[k]:>7.1f} {raw_means[k]:>12.6f} {weighted[k]:>12.6f} {pct:>9.2f}%")
    rows.append({"component": names[k], "lambda": lam[k],
                 "raw_mean": round(raw_means[k], 6),
                 "weighted_mean": round(weighted[k], 6),
                 "pct_of_total": round(pct, 2)})

print(f"\nTotal mean per-step cost: {total:.6f}")

out = OUT_DIR / "cost_component_breakdown.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"Saved -> {out}")
