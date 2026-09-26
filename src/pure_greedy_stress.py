"""
Pure Greedy stress test across scenarios where GraphOptimizer should win.
Reviewer: "Pure Greedy dominates nominally — show where it fails."
"""
import sys, csv, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from graph_battery_pack import PackGNN, build_pack_from_ecm
from safe_fast_charge_optimizer import default_config, GraphGuidedOptimizer, CCCVController
from ablation_study import GreedyOptimizer

ECM_DIR = Path(__file__).parent.parent / "results/ecm"
MDL_DIR = Path(__file__).parent.parent / "results/models"
OUT_DIR = Path(__file__).parent.parent / "results"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(str(sorted(MDL_DIR.glob("pack_gnn_*.pt"))[-1]), map_location=DEVICE)
gnn  = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(DEVICE)
gnn.load_state_dict(ckpt.get("model_state", ckpt)); gnn.eval()
ecm  = sorted(ECM_DIR.glob("ecm_params_*.parquet"))[-1]
cfg  = default_config()

def run_ep(pack, ctrl, cfg):
    ctrl.reset(); steps=0; viol=0; T_max=0
    while steps < cfg["max_steps"]:
        if np.mean([c.SOC for c in pack.cells]) >= cfg["target_soc"]: break
        m = pack.step(ctrl.get_currents(pack), dt=cfg["dt_s"])
        viol += m["n_violations"]; T_max = max(T_max, m["T_max"]); steps += 1
    socs=[c.SOC for c in pack.cells]; Tf=[c.T_C for c in pack.cells]
    return {"time":steps*cfg["dt_s"]/60, "sigma":float(np.std(socs))*100,
            "T_max":T_max, "dT":float(max(Tf)-min(Tf)), "viol":viol}

scenarios = [
    ("LFP",  0.03, 25.0,  "LFP nominal"),
    ("NMC",  0.03, 25.0,  "NMC zero-shot"),
    ("LCO",  0.03, 25.0,  "LCO zero-shot"),
    ("LFP",  0.15, 25.0,  "LFP high-imbalance"),
    ("LFP",  0.25, 25.0,  "LFP extreme-imbalance"),
    ("LFP",  0.03, 40.0,  "LFP hot-start 40C"),
    ("LFP",  0.03, 44.0,  "LFP hot-start 44C"),
]

rows = []; N = 30
print(f"{'Scenario':<28} {'Controller':<18} {'σ%':>7} {'ΔT':>7} {'Viol':>6}")
print("-"*72)

for chem, sigma_init, T_amb, label in scenarios:
    ctrls = [("GraphOptimizer", GraphGuidedOptimizer(cfg, gnn)),
              ("Pure Greedy",    GreedyOptimizer(cfg, gnn)),
              ("CC-CV",          CCCVController(cfg))]
    for ctrl_name, ctrl in ctrls:
        eps=[]; t0=time.time()
        for ep in range(N):
            np.random.seed(ep*7); torch.manual_seed(ep*7)
            pack=build_pack_from_ecm(ecm_parquet=ecm, n_cells=12, chemistry=chem,
                                      soc_init=0.20, soc_noise=sigma_init, T_amb=T_amb,
                                      seed=ep*7)
            eps.append(run_ep(pack, ctrl, cfg))
            if (ep+1)%10==0:
                print(f"  {label} {ctrl_name} ep {ep+1}/{N} "
                      f"σ={eps[-1]['sigma']:.3f}% viol={eps[-1]['viol']}", flush=True)
        def a(k): return round(float(np.mean([r[k] for r in eps])),4)
        row={"scenario":label,"chemistry":chem,"sigma_init":sigma_init,
             "T_amb":T_amb,"controller":ctrl_name,
             "time_min":a("time"),"sigma_pct":a("sigma"),
             "T_max":a("T_max"),"dT":a("dT"),"violations":a("viol"),"N":N}
        rows.append(row)
        print(f"{label:<28} {ctrl_name:<18} {row['sigma_pct']:>7.3f} "
              f"{row['dT']:>7.4f} {row['violations']:>6.2f}", flush=True)

out = OUT_DIR / "pure_greedy_stress_n30.csv"
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"\nSaved → {out}")
