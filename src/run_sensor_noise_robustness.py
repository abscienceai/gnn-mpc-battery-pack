import sys, time, json
sys.path.insert(0, "src")
import numpy as np, torch
from copy import deepcopy
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

# Noise levels exactly as stated in the manuscript's Section 4.7
NOISE_LEVELS = [
    ("clean",  None),
    ("low",    {"V_mV": 5,  "I_mA": 10,  "T_C": 0.1}),
    ("medium", {"V_mV": 20, "I_mA": 50,  "T_C": 0.5}),
    ("high",   {"V_mV": 50, "I_mA": 100, "T_C": 1.0}),
]

class NoisyObserverWrapper:
    """Wraps GraphGuidedOptimizer: injects zero-mean Gaussian sensor noise
    into V_term, I_A, T_C on a deep-copied 'observed' pack passed to the
    underlying controller's decision logic. The real, unperturbed pack is
    what is actually charged (physics proceeds on ground truth). Mirrors
    the SOC-error-injection design in bigru_error_injection.py. Only
    terminal voltage (the physically measured quantity) is perturbed,
    not open-circuit voltage."""
    def __init__(self, inner_ctrl, level, seed):
        self.inner = inner_ctrl
        self.level = level
        self.rng = np.random.default_rng(seed)
    def reset(self):
        self.inner.reset()
    def get_currents(self, pack):
        if self.level is None:
            return self.inner.get_currents(pack)
        noisy = deepcopy(pack)
        v_sigma = self.level["V_mV"] / 1000.0
        i_sigma = self.level["I_mA"] / 1000.0
        t_sigma = self.level["T_C"]
        for c in noisy.cells:
            c.V_term = c.V_term + self.rng.normal(0, v_sigma)
            c.I_A    = c.I_A    + self.rng.normal(0, i_sigma)
            c.T_C    = c.T_C    + self.rng.normal(0, t_sigma)
        return self.inner.get_currents(noisy)

results = {}
for label, level in NOISE_LEVELS:
    eps = []
    t0 = time.time()
    for ep in range(N):
        np.random.seed(ep * 7)
        torch.manual_seed(ep * 7)
        pack = build_pack_from_ecm(ecm_parquet=ecm, n_cells=12, chemistry="LFP",
                                    soc_init=0.20, soc_noise=0.03, T_amb=25.0, seed=ep * 7)
        base_ctrl = GraphGuidedOptimizer(cfg, gnn)
        ctrl = NoisyObserverWrapper(base_ctrl, level, seed=ep * 7 + 100000)
        r = run_episode(pack, ctrl, cfg, label, verbose=False)
        eps.append(r)
        if (ep + 1) % 5 == 0:
            print(f"  [{label}] ep {ep+1}/{N} sigma={r['final_SOC_imbalance']*100:.3f}% "
                  f"viol={r['total_violations']}", flush=True)
    sigma = np.mean([r["final_SOC_imbalance"] for r in eps]) * 100
    sigma_std = np.std([r["final_SOC_imbalance"] for r in eps]) * 100
    viol = np.mean([r["total_violations"] for r in eps])
    tmax = np.mean([r["final_T_max"] for r in eps])
    print(f"RESULT {label}: sigma={sigma:.4f}+-{sigma_std:.4f}% viol={viol:.4f} "
          f"Tmax={tmax:.2f} ({time.time()-t0:.0f}s)", flush=True)
    results[label] = {"sigma_SOC": sigma, "sigma_SOC_std": sigma_std, "viol": viol, "T_max": tmax}

print("\n--- Degradation vs clean ---")
clean_sigma = results["clean"]["sigma_SOC"]
for label, _ in NOISE_LEVELS:
    if label == "clean":
        continue
    delta = results[label]["sigma_SOC"] - clean_sigma
    pct = (results[label]["sigma_SOC"] / clean_sigma - 1) * 100
    print(f"{label}: sigma={results[label]['sigma_SOC']:.4f}% "
          f"(delta={delta:+.4f}pp, {pct:+.1f}% relative)")

with open("results/sensor_noise_robustness_postpatch.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved -> results/sensor_noise_robustness_postpatch.json")
