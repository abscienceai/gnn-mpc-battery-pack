"""
Sensitivity analysis for the two heuristic rollout constants:
  - alpha: SOC correction blend factor (currently 0.3)
  - beta:  Temperature scale factor    (currently 3.0)

SOC update:  SOC_next = SOC + coulomb + alpha*(GNN_SOC - SOC)
Temp update: T_next   = T   + GNN_dT  * (I/I_max) * beta
"""
import sys, csv, numpy as np, torch, re
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))
from graph_battery_pack import PackGNN, build_pack_from_ecm
from safe_fast_charge_optimizer import default_config, GraphGuidedOptimizer

ECM_DIR = Path(__file__).parent.parent / "results/ecm"
MDL_DIR = Path(__file__).parent.parent / "results/models"
OUT_DIR = Path(__file__).parent.parent / "results"
SRC     = Path(__file__).parent / "safe_fast_charge_optimizer.py"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(str(sorted(MDL_DIR.glob("pack_gnn_*.pt"))[-1]), map_location=DEVICE)
gnn  = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(DEVICE)
gnn.load_state_dict(ckpt.get("model_state", ckpt)); gnn.eval()
ecm  = sorted(ECM_DIR.glob("ecm_params_*.parquet"))[-1]

def run_with_constants(alpha, beta, n_ep=20):
    """Temporarily patch rollout constants and evaluate."""
    code = SRC.read_text()
    # Patch the two constants
    code_patched = re.sub(
        r'soc_virtual\s*=\s*np\.clip\(\s*soc_virtual\s*\+\s*coulomb\s*\+\s*0\.3\s*\*',
        f'soc_virtual = np.clip(soc_virtual + coulomb + {alpha} *',
        code
    )
    code_patched = re.sub(
        r'T_virtual\s*\+=\s*dT_pred\s*\*\s*i_ratio\s*\*\s*3\.0',
        f'T_virtual += dT_pred * i_ratio * {beta}',
        code_patched
    )
    # Write to temp file and import
    tmp = Path("/tmp/sfo_patched.py")
    tmp.write_text(code_patched)
    import importlib.util
    spec = importlib.util.spec_from_file_location("sfo_patched", tmp)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["sfo_patched"] = mod
    spec.loader.exec_module(mod)

    cfg  = default_config()
    ctrl = mod.GraphGuidedOptimizer(cfg, gnn)
    results = []
    for ep in range(n_ep):
        np.random.seed(ep*7); torch.manual_seed(ep*7)
        pack = build_pack_from_ecm(ecm_parquet=ecm, n_cells=12, chemistry="LFP",
                                    soc_init=0.20, soc_noise=0.03, T_amb=25.0,
                                    seed=ep*7)
        ctrl.reset(); steps=0; viol=0; T_max=0
        while steps < cfg["max_steps"]:
            if np.mean([c.SOC for c in pack.cells]) >= cfg["target_soc"]: break
            m = pack.step(ctrl.get_currents(pack), dt=cfg["dt_s"])
            viol += m["n_violations"]; T_max = max(T_max, m["T_max"]); steps += 1
        socs=[c.SOC for c in pack.cells]
        results.append({"sigma":float(np.std(socs))*100, "viol":viol, "T_max":T_max})
    return (round(float(np.mean([r["sigma"] for r in results])),4),
            round(float(np.mean([r["viol"]  for r in results])),4),
            round(float(np.mean([r["T_max"] for r in results])),2))

# Grid: alpha x beta
alphas = [0.0, 0.1, 0.3, 0.5, 1.0]   # SOC blend
betas  = [1.0, 2.0, 3.0, 4.0, 5.0]   # T scale

rows = []
print(f"{'alpha':>7} {'beta':>6} {'σ_SOC%':>9} {'Viol':>6} {'T_max':>8}")
print("-"*45)
for alpha in alphas:
    for beta in betas:
        sigma, viol, T_max = run_with_constants(alpha, beta, n_ep=20)
        mark = " ← CURRENT" if (alpha==0.3 and beta==3.0) else ""
        print(f"{alpha:>7.1f} {beta:>6.1f} {sigma:>9.4f} {viol:>6.2f} {T_max:>8.2f}{mark}",flush=True)
        rows.append({"alpha":alpha,"beta":beta,"sigma_pct":sigma,"violations":viol,"T_max":T_max})

out = OUT_DIR / "rollout_sensitivity.csv"
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"\nSaved → {out}")
