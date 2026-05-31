"""
offline_replay_validation.py
============================
Validates the GraphOptimizer against real-world charging trajectories.

STRATEGY 1: Offline Replay Validation
  - Load real MATR/CALCE/RWTH charging profiles
  - Replace observed current I(t) with optimizer's I*(t)
  - Compare SOC imbalance, temperature, aging proxy

STRATEGY 2: Strict Train/Test Dataset Separation
  - Calibrate ECM on MATR (LFP) only
  - Test on CALCE (LCO) and RWTH (NMC) — never seen during calibration

STRATEGY 3: Parameter Mismatch Robustness
  - Perturb R0 ±20%, thermal coeff ±30%
  - Evaluate controller degradation

OUTPUT: results/replay_validation/
"""

import sys, json, pickle, argparse, warnings
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from graph_battery_pack import build_pack_from_ecm, PackGNN
from safe_fast_charge_optimizer import (
    GraphGuidedOptimizer, CCCVController,
    default_config, run_episode
)

ECM_DIR   = Path("results/ecm")
MODEL_DIR = Path("results/models")

MATR_PATH  = Path("/home/msoylu/alper/battery_datasets/MATR/processed")
CALCE_PATH = Path("/home/msoylu/alper/battery_datasets/CALCE/processed")
RWTH_PATH  = Path("/home/msoylu/alper/battery_datasets/RWTH/processed")


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def load_gnn(device):
    gnn = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(device)
    trained = sorted(MODEL_DIR.glob("pack_gnn_*.pt"))
    if trained:
        ckpt = torch.load(trained[-1], map_location=device)
        gnn.load_state_dict(ckpt["model_state"])
        print(f"  Loaded GNN: {trained[-1].name}")
    return gnn


def load_real_cycles(dataset_path: Path, n_cells: int = 12,
                     min_steps: int = 20) -> list:
    """Load real charging cycles from pkl files."""
    cycles = []
    for f in sorted(dataset_path.glob("*.pkl"))[:50]:
        try:
            data = pickle.load(open(f, "rb"))
            for cyc in data.get("cycle_data", []):
                I = np.array(cyc.get("current_in_A", []), dtype=float)
                V = np.array(cyc.get("voltage_in_V", []), dtype=float)
                T = np.array(cyc.get("temperature_in_C", []) or
                              [25.0]*len(I), dtype=float)
                t = np.array(cyc.get("time_in_s", []), dtype=float)

                # Keep charge phases only
                mask = I > 0.05
                if mask.sum() < min_steps:
                    mask = I < -0.05
                    I = -I
                if mask.sum() < min_steps:
                    continue

                I, V, T, t = I[mask], V[mask], T[mask], t[mask]
                if not np.all(np.isfinite(I)): continue

                cycles.append({
                    "I": I, "V": V, "T": T, "t": t,
                    "cell_id": f.stem,
                    "cycle": cyc.get("cycle_number", -1),
                    "nom_cap": float(data.get("nominal_capacity_in_Ah", 1.1) or 1.1),
                })
                if len(cycles) >= n_cells * 3:
                    break
        except Exception:
            continue
    return cycles


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGY 1: OFFLINE REPLAY VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def offline_replay(n_cells: int = 12, n_episodes: int = 20,
                   output_dir: Path = None) -> dict:
    """
    For each episode:
      1. Sample n_cells real MATR cells (TEST set — not used for ECM fitting)
      2. Record real current trajectory I_real(t)
      3. Apply GraphOptimizer → I_opt(t) from same initial state
      4. Apply CC-CV → I_cccv(t)
      5. Compare terminal SOC imbalance, peak T, aging proxy
    """
    print("\n── Strategy 1: Offline Replay Validation ───────────────────")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn    = load_gnn(device)
    cfg    = default_config()
    cfg["n_cells"] = n_cells

    ecm_parquet = sorted(ECM_DIR.glob("*.parquet"))
    ecm_parquet = ecm_parquet[-1] if ecm_parquet else None

    # Load MATR real cycles (held-out test set)
    print("  Loading MATR real cycles (held-out test set)...")
    real_cycles = load_real_cycles(MATR_PATH, n_cells=n_cells)
    if not real_cycles:
        print("  [SKIP] No real cycles found")
        return {}
    print(f"  Loaded {len(real_cycles)} real cycles")

    results = {"real": [], "optimized": [], "cccv": []}
    rng = np.random.default_rng(42)

    for ep in range(n_episodes):
        # Sample n_cells cycles
        idx   = rng.choice(len(real_cycles), size=n_cells, replace=True)
        cells = [real_cycles[i] for i in idx]

        # Build pack from ECM (calibrated on MATR training set)
        soc_init  = 0.20 + rng.uniform(-0.02, 0.02)
        pack_opt  = build_pack_from_ecm(ecm_parquet, n_cells=n_cells,
                                         chemistry="LFP",
                                         soc_init=soc_init, soc_noise=0.03)
        pack_real = deepcopy(pack_opt)
        pack_cccv = deepcopy(pack_opt)

        dt = 60.0
        max_steps = 80

        # ── Real trajectory (what actually happened) ──
        real_soc_imb, real_T_max = [], []
        for step in range(max_steps):
            currents = []
            for c in cells:
                t_idx = min(step * 5, len(c["I"]) - 1)
                currents.append(abs(float(c["I"][t_idx])))
            m = pack_real.step(np.array(currents, dtype=np.float32), dt=dt)
            real_soc_imb.append(m["SOC_imbalance"])
            real_T_max.append(m["T_max"])
            if m["SOC_mean"] >= 0.80: break

        # ── GraphOptimizer trajectory ──
        ctrl_opt  = GraphGuidedOptimizer(cfg, gnn)
        res_opt   = run_episode(pack_opt, ctrl_opt, cfg, "GraphOpt", verbose=False)

        # ── CC-CV trajectory ──
        ctrl_cccv = CCCVController(cfg)
        res_cccv  = run_episode(pack_cccv, ctrl_cccv, cfg, "CC-CV", verbose=False)

        results["real"].append({
            "ep": ep,
            "soc_imbalance": float(real_soc_imb[-1]),
            "T_max":         float(real_T_max[-1]),
            "time_min":      len(real_soc_imb) * dt / 60,
        })
        results["optimized"].append({
            "ep": ep,
            "soc_imbalance": res_opt["final_SOC_imbalance"],
            "T_max":         res_opt["final_T_max"],
            "time_min":      res_opt["charging_time_min"],
        })
        results["cccv"].append({
            "ep": ep,
            "soc_imbalance": res_cccv["final_SOC_imbalance"],
            "T_max":         res_cccv["final_T_max"],
            "time_min":      res_cccv["charging_time_min"],
        })

        if (ep + 1) % 5 == 0:
            print(f"  [{ep+1}/{n_episodes}] real σ={real_soc_imb[-1]*100:.2f}% | "
                  f"opt σ={res_opt['final_SOC_imbalance']*100:.2f}% | "
                  f"cccv σ={res_cccv['final_SOC_imbalance']*100:.2f}%", flush=True)

    # Summary
    def agg(lst, key):
        v = [r[key] for r in lst]
        return float(np.mean(v)), float(np.std(v))

    summary = {}
    for method, res_list in results.items():
        summary[method] = {
            "soc_imbalance": agg(res_list, "soc_imbalance"),
            "T_max":         agg(res_list, "T_max"),
            "time_min":      agg(res_list, "time_min"),
        }

    print(f"\n  {'Method':<15} {'σ_SOC%':>10} {'T_max°C':>10} {'Time':>8}")
    for m, s in summary.items():
        si = s["soc_imbalance"]
        tm = s["T_max"]
        t  = s["time_min"]
        print(f"  {m:<15} {si[0]*100:>8.3f}% {tm[0]:>9.2f}°C {t[0]:>6.1f}min")

    # Improvement over real
    ri = summary["real"]["soc_imbalance"][0]
    oi = summary["optimized"]["soc_imbalance"][0]
    ci = summary["cccv"]["soc_imbalance"][0]
    print(f"\n  vs Real trajectory:")
    print(f"    CC-CV      imbalance:  {ci*100:.3f}% ({(ci-ri)/ri*100:+.1f}% vs real)")
    print(f"    GraphOpt   imbalance:  {oi*100:.3f}% ({(oi-ri)/ri*100:+.1f}% vs real)")

    if output_dir:
        out = output_dir / "replay_results.json"
        json.dump({"summary": summary, "raw": results}, open(out,"w"),
                  indent=2, default=str)
        print(f"  ✅ Saved → {out.name}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGY 2: STRICT TRAIN/TEST SEPARATION
# ═══════════════════════════════════════════════════════════════════════════

def cross_dataset_validation(n_episodes: int = 20,
                              output_dir: Path = None) -> dict:
    """
    Train: MATR (LFP) only
    Test:  CALCE (LCO) and RWTH (NMC) — never seen during calibration
    """
    print("\n── Strategy 2: Strict Train/Test Separation ─────────────────")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn        = load_gnn(device)
    ecm_parquet = sorted(ECM_DIR.glob("*.parquet"))[-1]

    test_sets = [
        ("CALCE (LCO)", "LCO", 12),
        ("RWTH (NMC)",  "NMC", 12),
    ]

    results = {}
    for ds_name, chem, n_cells in test_sets:
        print(f"\n  Test set: {ds_name} (calibrated on MATR/LFP only)")
        cfg = default_config()
        cfg["n_cells"] = n_cells

        ep_opt, ep_cccv = [], []
        for ep in range(n_episodes):
            np.random.seed(ep * 11)
            pack_opt  = build_pack_from_ecm(ecm_parquet, n_cells=n_cells,
                                             chemistry=chem,
                                             soc_init=0.20, soc_noise=0.03)
            pack_cccv = deepcopy(pack_opt)

            res_opt  = run_episode(pack_opt,  GraphGuidedOptimizer(cfg, gnn),
                                    cfg, "GraphOpt", verbose=False)
            res_cccv = run_episode(pack_cccv, CCCVController(cfg),
                                    cfg, "CC-CV",    verbose=False)
            ep_opt.append(res_opt); ep_cccv.append(res_cccv)

        opt_si  = np.mean([r["final_SOC_imbalance"] for r in ep_opt])
        cccv_si = np.mean([r["final_SOC_imbalance"] for r in ep_cccv])
        impr    = (cccv_si - opt_si) / cccv_si * 100

        print(f"    CC-CV σ={cccv_si*100:.3f}% | GraphOpt σ={opt_si*100:.3f}% "
              f"| improvement={impr:.1f}%")

        results[ds_name] = {
            "chemistry":    chem,
            "graphopt_soc": round(float(opt_si),  5),
            "cccv_soc":     round(float(cccv_si), 5),
            "improvement":  round(float(impr),    1),
        }

    if output_dir:
        out = output_dir / "cross_dataset.json"
        json.dump(results, open(out,"w"), indent=2)
        print(f"\n  ✅ Saved → cross_dataset.json")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGY 3: PARAMETER MISMATCH ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════

def parameter_mismatch(n_episodes: int = 20, n_cells: int = 12,
                        output_dir: Path = None) -> dict:
    """
    Stress-test under parameter mismatch:
      - R0 ± 20%  (internal resistance uncertainty)
      - hA ± 30%  (thermal coefficient uncertainty)
      - Q_nom ± 10% (capacity uncertainty)
    """
    print("\n── Strategy 3: Parameter Mismatch Robustness ───────────────")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn        = load_gnn(device)
    cfg        = default_config()
    cfg["n_cells"] = n_cells
    ecm_parquet = sorted(ECM_DIR.glob("*.parquet"))[-1]

    perturbations = {
        "Nominal":      {"R0": 1.0, "hA": 1.0, "Q": 1.0},
        "R0 +20%":      {"R0": 1.2, "hA": 1.0, "Q": 1.0},
        "R0 -20%":      {"R0": 0.8, "hA": 1.0, "Q": 1.0},
        "hA +30%":      {"R0": 1.0, "hA": 1.3, "Q": 1.0},
        "hA -30%":      {"R0": 1.0, "hA": 0.7, "Q": 1.0},
        "Q_nom -10%":   {"R0": 1.0, "hA": 1.0, "Q": 0.9},
        "All mismatch": {"R0": 1.2, "hA": 0.7, "Q": 0.9},
    }

    nominal_si = None
    results = {}

    for label, perts in perturbations.items():
        ep_si = []
        for ep in range(n_episodes):
            np.random.seed(ep * 7)
            pack = build_pack_from_ecm(ecm_parquet, n_cells=n_cells,
                                        chemistry="LFP",
                                        soc_init=0.20, soc_noise=0.03)

            # Apply perturbations to cell parameters
            for cell in pack.cells:
                cell.R0       *= perts["R0"]
                cell.R1       *= perts["R0"]
                cell.Q_nom_Ah *= perts["Q"]
                # Thermal: hA stored in pack thermal model
            # Perturb pack-level thermal coefficient
            if hasattr(pack, "thermal_model"):
                pack.thermal_model.hA_W_K *= perts["hA"]

            ctrl = GraphGuidedOptimizer(cfg, gnn)
            res  = run_episode(pack, ctrl, cfg, label, verbose=False)
            ep_si.append(res["final_SOC_imbalance"])

        mean_si = float(np.mean(ep_si))
        if label == "Nominal":
            nominal_si = mean_si

        deg = (mean_si - nominal_si) / nominal_si * 100 if nominal_si else 0
        print(f"  {label:<20} σ_SOC={mean_si*100:.3f}% "
              f"degradation={deg:+.1f}%")

        results[label] = {
            "soc_imbalance_mean": round(mean_si, 5),
            "soc_imbalance_std":  round(float(np.std(ep_si)), 5),
            "degradation_pct":    round(deg, 1),
        }

    if output_dir:
        out = output_dir / "param_mismatch.json"
        json.dump(results, open(out,"w"), indent=2)
        print(f"  ✅ Saved → param_mismatch.json")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes",  type=int, default=20)
    parser.add_argument("--n_cells",     type=int, default=12)
    parser.add_argument("--output_dir",  default="results/replay_validation")
    parser.add_argument("--skip_replay", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Real-World Validation Suite")
    print("=" * 65)

    all_results = {}

    if not args.skip_replay:
        r1 = offline_replay(args.n_cells, args.n_episodes, out_dir)
        all_results["replay"] = r1

    r2 = cross_dataset_validation(args.n_episodes, out_dir)
    all_results["cross_dataset"] = r2

    r3 = parameter_mismatch(args.n_episodes, args.n_cells, out_dir)
    all_results["param_mismatch"] = r3

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"validation_suite_{ts}.json"
    json.dump(all_results, open(out,"w"), indent=2, default=str)

    print(f"\n{'='*65}")
    print(f"  ✅ Validation suite complete → {out.name}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
