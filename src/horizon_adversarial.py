"""
horizon_adversarial.py
======================
Reviewer Comment 5: "H=1 appears to dominate H=5 — show when H=1 fails"

Tests H=1 vs H=5 under increasing initial SOC heterogeneity.
Under adversarial high-imbalance conditions, the greedy H=1 controller
is expected to produce voltage violations by pushing high-SOC cells
past V_max without anticipating the downstream consequence.
"""
import sys, csv, warnings, torch, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from graph_battery_pack import PackGNN, build_pack_from_ecm
from safe_fast_charge_optimizer import default_config, GraphGuidedOptimizer

ECM_DIR = Path(__file__).parent.parent / "results/ecm"
MDL_DIR = Path(__file__).parent.parent / "results/models"
OUT_DIR = Path(__file__).parent.parent / "results"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


def run_eval(pack, ctrl, cfg):
    ctrl.reset()
    steps=0; viol=0; T_max=0.0; aging_cum=0.0
    while steps < cfg["max_steps"]:
        if np.mean([c.SOC for c in pack.cells]) >= cfg["target_soc"]: break
        m = pack.step(ctrl.get_currents(pack), dt=cfg["dt_s"])
        viol += m["n_violations"]; T_max = max(T_max, m["T_max"])
        aging_cum += m["aging_cost"]; steps += 1
    socs = [c.SOC for c in pack.cells]; Tf = [c.T_C for c in pack.cells]
    return {"time_min":     round(steps * cfg["dt_s"] / 60.0, 2),
            "sigma_pct":    round(float(np.std(socs)) * 100, 4),
            "T_max":        round(T_max, 2),
            "dT":           round(float(max(Tf) - min(Tf)), 3),
            "violations":   viol,
            "aging_proxy":  round(aging_cum * 1e4, 4)}


def main():
    cfg_base = default_config()
    ecm = sorted(ECM_DIR.glob("ecm_params_*.parquet"))[-1]

    ckpt = torch.load(str(sorted(MDL_DIR.glob("pack_gnn_*.pt"))[-1]), map_location=DEVICE)
    gnn  = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(DEVICE)
    gnn.load_state_dict(ckpt.get("model_state", ckpt)); gnn.eval()

    # sigma_init levels: nominal → moderate → adversarial
    sigma_inits = [0.03, 0.08, 0.15, 0.25]
    horizons    = [(1, "H=1 Greedy"), (5, "H=5 Full")]
    N_EP = 30
    rows = []

    print(f"\n{'σ_init':<10} {'Horizon':<14} {'Time':>7} {'σ_SOC%':>8} {'ΔT':>6} {'Viol':>6} {'Aging':>8}")
    print("─" * 65)

    total_runs = len(sigma_inits) * len(horizons)
    run_idx = 0

    for sigma_init in sigma_inits:
        for H, label in horizons:
            run_idx += 1
            cfg = dict(cfg_base); cfg["horizon"] = H
            ctrl = GraphGuidedOptimizer(cfg, gnn)
            eps  = []
            for ep in range(N_EP):
                np.random.seed(ep * 7); torch.manual_seed(ep * 7)
                pack = build_pack_from_ecm(ecm_parquet=ecm, n_cells=12,
                                            chemistry="LFP", soc_init=0.20,
                                            soc_noise=sigma_init, T_amb=25.0,
                                            seed=ep * 7)
                eps.append(run_eval(pack, ctrl, cfg))
                if (ep + 1) % 10 == 0:
                    import sys
                    print(f"  [{run_idx}/{total_runs}] σ_init={sigma_init} {label} "
                          f"ep {ep+1}/{N_EP} | last σ={eps[-1]['sigma_pct']:.3f}%", flush=True)

            def a(k): return round(float(np.mean([r[k] for r in eps])), 4)

            row = {"sigma_init": sigma_init, "horizon": label, "H": H,
                   "time_min":   a("time_min"),
                   "sigma_pct":  a("sigma_pct"),
                   "T_max":      a("T_max"),
                   "dT":         a("dT"),
                   "violations": a("violations"),
                   "aging_proxy":a("aging_proxy"),
                   "N": N_EP}
            rows.append(row)
            print(f"{sigma_init:<10} {label:<14} {row['time_min']:>7} {row['sigma_pct']:>8} "
                  f"{row['dT']:>6} {row['violations']:>6} {row['aging_proxy']:>8}")

        print()  # blank line between sigma levels

    out = OUT_DIR / "horizon_adversarial_n30.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Saved → {out}")

    # Summary: where does H=1 fail?
    print("\n── VIOLATION SUMMARY ──────────────────────────────────")
    for r in rows:
        if r["violations"] > 0:
            print(f"  *** VIOLATION: σ_init={r['sigma_init']} {r['horizon']} → {r['violations']} viol/ep ***")
    h1_viols = [r for r in rows if r["H"]==1 and r["violations"]>0]
    h5_viols = [r for r in rows if r["H"]==5 and r["violations"]>0]
    print(f"\n  H=1 violation cases: {len(h1_viols)}")
    print(f"  H=5 violation cases: {len(h5_viols)}")


if __name__ == "__main__":
    main()
