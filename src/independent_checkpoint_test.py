"""
independent_checkpoint_test.py
================================
Frozen-checkpoint generalisation test (one-step MAE) on independently
resampled LFP ECM cell realisations, excluding the 12 rows sampled with
the original training seed (seed=42). The checkpoint is NOT modified.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from graph_battery_pack import build_pack_from_ecm, PackGNN
from train_gnn import generate_rollout
from safe_fast_charge_optimizer import default_config

ECM_DIR = Path(__file__).parent.parent / "results" / "ecm"
MODEL_DIR = Path(__file__).parent.parent / "results" / "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rollouts", type=int, default=60)
    ap.add_argument("--n_steps", type=int, default=5)
    ap.add_argument("--output_dir", default="results/independent_checkpoint_test")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = default_config()
    I_max_C = cfg["I_max_C"]

    ecm_parquet = sorted(ECM_DIR.glob("*.parquet"))[-1]
    df_full = pd.read_parquet(ecm_parquet)
    sub = df_full[df_full["dataset"] == "MATR"].dropna(subset=["IR_ohm"])

    train_rows = sub.sample(n=12, replace=len(sub) < 12, random_state=42)
    train_idx = set(train_rows.index.tolist())
    remaining = sub.drop(index=list(train_idx))
    print(f"[INFO] Excluded {len(train_idx)} training-seed rows; "
          f"{len(remaining)} of {len(sub)} MATR/LFP rows remain in the independent pool.")

    gnn = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3).to(device)
    ckpt_path = sorted(MODEL_DIR.glob("pack_gnn_*.pt"))[-1]
    ckpt = torch.load(str(ckpt_path), map_location=device)
    gnn.load_state_dict(ckpt.get("model_state", ckpt))
    gnn.eval()
    print(f"[INFO] Loaded FROZEN checkpoint: {ckpt_path.name} (not modified)")

    rng = np.random.default_rng(999_999)

    soc_abs_errs, dT_abs_errs = [], []
    dT_key_used = None
    dT_key_candidates = ["delta_T_pred", "dT_pred", "delta_t_pred", "dt_pred", "deltaT_pred", "temp_pred"]

    for i in range(args.n_rollouts):
        test_seed = int(rng.integers(10_000_000, 2**31 - 1))
        soc_init = float(rng.uniform(0.10, 0.85))
        soc_noise = float(rng.uniform(0.01, 0.05))
        pack = build_pack_from_ecm(
            n_cells=12, chemistry="LFP",
            soc_init=soc_init, soc_noise=soc_noise,
            T_amb=float(rng.uniform(20, 35)),
            ecm_df=remaining,
            seed=test_seed,
        )
        samples = generate_rollout(pack, n_steps=args.n_steps, I_max_C=I_max_C, rng=rng)

        for (x, edge_index, edge_attr, y_soc, y_dT, y_aging) in samples:
            with torch.no_grad():
                out = gnn(
                    torch.tensor(x, dtype=torch.float32).to(device),
                    torch.tensor(edge_index, dtype=torch.long).to(device),
                    torch.tensor(edge_attr, dtype=torch.float32).to(device),
                )
            soc_pred = out["soc_pred"].cpu().numpy().flatten()
            soc_abs_errs.append(np.mean(np.abs(soc_pred - np.asarray(y_soc).flatten())))

            if dT_key_used is None:
                for key in dT_key_candidates:
                    if key in out:
                        dT_key_used = key
                        break
                if dT_key_used is None:
                    print(f"[WARN] no delta-T key found; available keys: {list(out.keys())}")
            if dT_key_used is not None:
                dT_pred = out[dT_key_used].cpu().numpy().flatten()
                dT_abs_errs.append(np.mean(np.abs(dT_pred - np.asarray(y_dT).flatten())))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.n_rollouts}] running SOC MAE={np.mean(soc_abs_errs)*100:.3f}%")

    results = {
        "checkpoint": ckpt_path.name,
        "n_rollouts": args.n_rollouts,
        "n_steps_per_rollout": args.n_steps,
        "n_excluded_training_rows": len(train_idx),
        "n_independent_pool_rows": len(remaining),
        "soc_mae_pct": float(np.mean(soc_abs_errs) * 100),
        "soc_mae_pct_std": float(np.std(soc_abs_errs) * 100),
        "dT_key_used": dT_key_used,
        "dT_mae_C": float(np.mean(dT_abs_errs)) if dT_abs_errs else None,
        "dT_mae_C_std": float(np.std(dT_abs_errs)) if dT_abs_errs else None,
        "n_samples_evaluated": len(soc_abs_errs),
    }
    out_path = out_dir / "independent_checkpoint_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved -> {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
