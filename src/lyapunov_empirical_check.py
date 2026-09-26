"""
lyapunov_empirical_check.py
============================
Empirically verifies the conditions underlying Theorem 5.4 (One-step Lyapunov
decrease) using real closed-loop GraphOptimizer trajectories.

At each control step, computes:
  e(t)  = SOC(t) - mean(SOC(t))            [centered SOC vector, P*s(t)]
  w(t)  = r(t)  - mean(r(t))                [centered current-rate vector, P*r(t)]
          where r(t) = I(t) / Q_nom(t)  (units: 1/hour)

Reports:
  1. % of control steps satisfying e^T w < 0        (directional alignment, Eq. 30)
  2. % of control steps satisfying ||e_next||^2 <= ||e||^2   (SOC-imbalance energy descent,
     the mu=0 special case of the Lyapunov function V(x) explicitly noted in the paper)
  3. Empirical range of beta = -e^T w / ||e||^2      (observed contraction rate)
  4. Empirical range of L    = ||w|| / ||e||         (observed current-response gain)

These are the three concrete quantities the reviewer requested to substantiate
Remark 5.5 / the sufficient stability conditions in Section 5.7.
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

EPS = 1e-6  # threshold below which ||e|| is considered "at equilibrium" (excluded from beta/L stats)


def run_episode_with_logging(pack, ctrl, cfg):
    """Runs one episode, logging e(t), w(t), and next-state e(t+1) at every control step."""
    ctrl.reset()
    steps = 0
    log = []  # list of dicts: {e, w, e_next}

    dt_h = cfg["dt_s"] / 3600.0

    while steps < cfg["max_steps"]:
        socs_now = np.array([c.SOC for c in pack.cells])
        if np.mean(socs_now) >= cfg["target_soc"]:
            break

        Q_nom = np.array([c.Q_nom_Ah for c in pack.cells])
        I_applied = ctrl.get_currents(pack)          # actual applied action I*(t)
        r = I_applied / Q_nom                          # units: 1/h (A / Ah)

        e = socs_now - np.mean(socs_now)                # P * s(t)
        w = r - np.mean(r)                              # P * r(t)

        pack.step(I_applied, dt=cfg["dt_s"])             # advance real physics

        socs_next = np.array([c.SOC for c in pack.cells])
        e_next = socs_next - np.mean(socs_next)

        log.append({
            "e": e, "w": w, "e_next": e_next,
        })
        steps += 1

    return log


def main():
    cfg = default_config()
    ecm = sorted(ECM_DIR.glob("ecm_params_*.parquet"))[-1]

    ckpt = torch.load(str(sorted(MDL_DIR.glob("pack_gnn_*.pt"))[-1]), map_location=DEVICE)
    gnn  = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(DEVICE)
    gnn.load_state_dict(ckpt.get("model_state", ckpt))
    gnn.eval()

    ctrl = GraphGuidedOptimizer(cfg, gnn)

    N_EP = 30
    all_logs = []

    print(f"Running {N_EP} canonical LFP episodes with per-step Lyapunov logging...")
    for ep in range(N_EP):
        np.random.seed(ep * 7)
        torch.manual_seed(ep * 7)
        pack = build_pack_from_ecm(
            ecm_parquet=ecm, n_cells=12, chemistry="LFP",
            soc_init=cfg["soc_init"], soc_noise=cfg["soc_noise"], T_amb=cfg["T_amb"],
            seed=ep * 7,
        )
        log = run_episode_with_logging(pack, ctrl, cfg)
        all_logs.extend(log)
        if (ep + 1) % 10 == 0:
            print(f"  ep {ep+1}/{N_EP} | steps logged so far: {len(all_logs)}", flush=True)

    print(f"\nTotal control steps logged: {len(all_logs)}")

    # ── Metric 1: e^T w < 0 (directional alignment) ──────────────────────────
    inner_products = np.array([np.dot(d["e"], d["w"]) for d in all_logs])
    frac_aligned = float(np.mean(inner_products < 0))

    # ── Metric 2: ||e_next||^2 <= ||e||^2 (SOC-imbalance energy descent) ─────
    norm_e_sq   = np.array([np.dot(d["e"], d["e"]) for d in all_logs])
    norm_enext_sq = np.array([np.dot(d["e_next"], d["e_next"]) for d in all_logs])
    frac_descent = float(np.mean(norm_enext_sq <= norm_e_sq))

    # ── Metric 3 & 4: empirical beta and L (only where ||e|| > EPS) ──────────
    norm_e = np.sqrt(norm_e_sq)
    norm_w = np.array([np.linalg.norm(d["w"]) for d in all_logs])
    valid = norm_e > EPS

    beta_est = -inner_products[valid] / norm_e_sq[valid]
    L_est    = norm_w[valid] / norm_e[valid]

    print("\n" + "=" * 60)
    print("EMPIRICAL LYAPUNOV CONDITION CHECK")
    print("=" * 60)
    print(f"Total steps analysed        : {len(all_logs)}")
    print(f"Steps with ||e|| > {EPS} (non-equilibrium): {int(valid.sum())} "
          f"({100*valid.mean():.1f}%)")
    print()
    print(f"1. Fraction of steps with e^T w < 0 (alignment, Eq. 30):")
    print(f"   {frac_aligned*100:.2f}%")
    print()
    print(f"2. Fraction of steps with ||e_next||^2 <= ||e||^2 (SOC-imbalance descent):")
    print(f"   {frac_descent*100:.2f}%")
    print()
    print(f"3. Empirical beta = -e^T w / ||e||^2  (over non-equilibrium steps):")
    print(f"   mean={np.mean(beta_est):.4f}  median={np.median(beta_est):.4f}  "
          f"[p5={np.percentile(beta_est,5):.4f}, p95={np.percentile(beta_est,95):.4f}]")
    print()
    print(f"4. Empirical L = ||w|| / ||e||  (over non-equilibrium steps):")
    print(f"   mean={np.mean(L_est):.4f}  median={np.median(L_est):.4f}  "
          f"[p5={np.percentile(L_est,5):.4f}, p95={np.percentile(L_est,95):.4f}]")
    print()

    # Check the sufficient condition alpha_s = 2*beta*dt_h - L^2*dt_h^2 > 0
    dt_h = cfg["dt_s"] / 3600.0
    alpha_s_est = 2 * beta_est * dt_h - (L_est ** 2) * (dt_h ** 2)
    frac_alpha_s_positive = float(np.mean(alpha_s_est > 0))
    print(f"5. Fraction of non-equilibrium steps with alpha_s = 2*beta*dt_h - L^2*dt_h^2 > 0:")
    print(f"   {frac_alpha_s_positive*100:.2f}%  (dt_h={dt_h:.6f} h)")

    # Save summary
    summary = {
        "n_steps_total": len(all_logs),
        "n_steps_nonequilibrium": int(valid.sum()),
        "frac_aligned_pct": round(frac_aligned * 100, 2),
        "frac_descent_pct": round(frac_descent * 100, 2),
        "beta_mean": round(float(np.mean(beta_est)), 4),
        "beta_median": round(float(np.median(beta_est)), 4),
        "beta_p5": round(float(np.percentile(beta_est, 5)), 4),
        "beta_p95": round(float(np.percentile(beta_est, 95)), 4),
        "L_mean": round(float(np.mean(L_est)), 4),
        "L_median": round(float(np.median(L_est)), 4),
        "L_p5": round(float(np.percentile(L_est, 5)), 4),
        "L_p95": round(float(np.percentile(L_est, 95)), 4),
        "frac_alpha_s_positive_pct": round(frac_alpha_s_positive * 100, 2),
        "dt_h": round(dt_h, 6),
    }

    out = OUT_DIR / "lyapunov_empirical_check.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary.keys())
        w.writeheader()
        w.writerow(summary)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
