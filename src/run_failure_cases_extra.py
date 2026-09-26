import sys, time, json
sys.path.insert(0, "src")
import numpy as np, torch
from pathlib import Path
from graph_battery_pack import build_pack_from_ecm, PackGNN
from safe_fast_charge_optimizer import default_config, GraphGuidedOptimizer, run_episode

ecm = sorted(Path("results/ecm").glob("ecm_params_*.parquet"))[-1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = sorted(Path("results/models").glob("pack_gnn_*.pt"))[-1]
gnn = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(DEVICE)
ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
gnn.load_state_dict(ckpt.get("model_state", ckpt))
gnn.eval()
cfg = default_config()
N = 20

def run_batch(label, pack_fn):
    results = []
    t0 = time.time()
    for ep in range(N):
        pack = pack_fn(ep)
        ctrl = GraphGuidedOptimizer(cfg, gnn)
        ctrl.reset()
        r = run_episode(pack, ctrl, cfg, label, verbose=False)
        results.append(r)
        if (ep + 1) % 5 == 0:
            print(f"  [{label}] ep {ep+1}/{N} sigma={r['final_SOC_imbalance']*100:.3f}% "
                  f"viol={r['total_violations']}", flush=True)
    sig = np.mean([r["final_SOC_imbalance"] for r in results]) * 100
    viol = np.mean([r["total_violations"] for r in results])
    tmax = np.mean([r["final_T_max"] for r in results])
    print(f"RESULT {label}: sigma={sig:.4f}% viol={viol:.4f} Tmax={tmax:.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return {"label": label, "sigma_SOC": sig, "viol": viol, "T_max": tmax}

def pack_r0x5(ep):
    p = build_pack_from_ecm(ecm_parquet=ecm, n_cells=12, chemistry="LFP",
                             soc_init=0.20, soc_noise=0.03, T_amb=25.0, seed=ep * 7)
    for c in p.cells:
        c.R0 = c.R0 * 5.0
    return p

def pack_hotstart45(ep):
    p = build_pack_from_ecm(ecm_parquet=ecm, n_cells=12, chemistry="LFP",
                             soc_init=0.20, soc_noise=0.03, T_amb=45.0, seed=ep * 7)
    for c in p.cells:
        c.T_C = 45.0
    return p

all_results = []
all_results.append(run_batch("R0x5_mismatch", pack_r0x5))
all_results.append(run_batch("HotStart_45C", pack_hotstart45))

with open("results/failure_cases_extra_postpatch.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved -> results/failure_cases_extra_postpatch.json")
