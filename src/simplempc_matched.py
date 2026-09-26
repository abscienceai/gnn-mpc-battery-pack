"""
SimpleMPC matched-budget comparison.
Answers: "Is GNN the source of improvement, or just the larger budget?"
Compare:
  CC-CV | SimpleMPC(K=32,H=3) | SimpleMPC(K=64,H=5) | GraphOptimizer(K=64,H=5)
"""
import sys, csv, warnings, torch, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from graph_battery_pack import PackGNN, build_pack_from_ecm
from safe_fast_charge_optimizer import (default_config, GraphGuidedOptimizer,
                                         CCCVController, SimpleMPCController)

ECM_DIR = Path(__file__).parent.parent / "results/ecm"
MDL_DIR = Path(__file__).parent.parent / "results/models"
OUT_DIR = Path(__file__).parent.parent / "results"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

def run_eval(pack, ctrl, cfg):
    ctrl.reset()
    steps=0; viol=0; T_max=0.0
    while steps < cfg["max_steps"]:
        if np.mean([c.SOC for c in pack.cells]) >= cfg["target_soc"]: break
        m = pack.step(ctrl.get_currents(pack), dt=cfg["dt_s"])
        viol += m["n_violations"]; T_max = max(T_max, m["T_max"]); steps += 1
    socs=[c.SOC for c in pack.cells]; Tf=[c.T_C for c in pack.cells]
    return {"time_min":round(steps*cfg["dt_s"]/60,2),
            "sigma_pct":round(float(np.std(socs))*100,4),
            "T_max":round(T_max,2), "dT":round(float(max(Tf)-min(Tf)),3),
            "viol":viol}

def main():
    cfg = default_config()
    ecm = sorted(ECM_DIR.glob("ecm_params_*.parquet"))[-1]
    ckpt= torch.load(str(sorted(MDL_DIR.glob("pack_gnn_*.pt"))[-1]), map_location=DEVICE)
    gnn = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(DEVICE)
    gnn.load_state_dict(ckpt.get("model_state", ckpt)); gnn.eval()

    controllers = {
        "CC-CV":                 CCCVController(cfg),
        "SimpleMPC (K=32,H=3)":  SimpleMPCController(cfg, horizon=3, n_samples=32),
        "SimpleMPC (K=64,H=5)":  SimpleMPCController(cfg, horizon=5, n_samples=64),
        "GraphOptimizer (K=64,H=5)": GraphGuidedOptimizer(cfg, gnn),
    }
    rows = []
    print(f"{'Method':<30} {'Time':>7} {'σ_SOC%':>8} {'ΔT°C':>7} {'Viol':>6}")
    for name, ctrl in controllers.items():
        eps=[]
        for ep in range(30):
            np.random.seed(ep*7); torch.manual_seed(ep*7)
            pack=build_pack_from_ecm(ecm_parquet=ecm, n_cells=12, chemistry="LFP",
                                     soc_init=0.20, soc_noise=0.03, T_amb=25.0,
                                     seed=ep*7)
            eps.append(run_eval(pack, ctrl, cfg))
        def a(k): return round(float(np.mean([r[k] for r in eps])),4)
        row={"method":name,"time_min":a("time_min"),"sigma_pct":a("sigma_pct"),
             "T_max":a("T_max"),"dT":a("dT"),"violations":a("viol"),"N":30}
        rows.append(row)
        print(f"{name:<30} {row['time_min']:>7} {row['sigma_pct']:>8} {row['dT']:>7} {row['violations']:>6}")

    out=OUT_DIR/"simplempc_matched_n30.csv"
    with open(out,"w",newline="") as f:
        csv.DictWriter(f,fieldnames=rows[0].keys()).writeheader()
        csv.DictWriter(f,fieldnames=rows[0].keys()).writerows(rows)
    print(f"\nSaved → {out}")

if __name__=="__main__": main()
