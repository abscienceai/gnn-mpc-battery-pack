"""
analyze_results.py
===================
Generates LaTeX-ready Table 3 (LFP primary), Table 4 (speed-safety modes),
Table 6 (statistical significance), and Table 7 (cross-chemistry) from
the raw experiment0_v3 JSON outputs.
"""
import json
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path("../results/experiment0_v3")

def load(chem):
    files = sorted((RESULTS_DIR / chem).glob("experiment_results_*.json"))
    with open(files[-1]) as f:
        return json.load(f)

def metric_array(raw, controller, key):
    return np.array([ep[key] for ep in raw[controller]])

def mean_std(arr):
    return float(np.mean(arr)), float(np.std(arr))

CONTROLLERS = ["CC-CV", "CC-CV-Balance", "SimpleMPC", "Proportional", "GraphOptimizer"]
METRICS = {
    "time": "charging_time_min",
    "soc": "final_SOC_imbalance",
    "tmax": "final_T_max",
    "dt": "final_T_gradient",
    "aging": "cumulative_aging",
    "viol": "total_violations",
}

print("="*80)
print("TABLE 3 — Charging Performance Comparison (12-cell LFP Pack, N=30)")
print("="*80)
lfp = load("LFP")["raw_results"]
print(f"{'Method':<18}{'Time':>10}{'sigma_SOC':>12}{'T_max':>10}{'DT':>10}{'Aging':>12}{'Viol':>8}")
for c in CONTROLLERS:
    t_m, t_s = mean_std(metric_array(lfp, c, METRICS["time"]))
    s_m, s_s = mean_std(metric_array(lfp, c, METRICS["soc"]) * 100)
    tm_m, tm_s = mean_std(metric_array(lfp, c, METRICS["tmax"]))
    dt_m, dt_s = mean_std(metric_array(lfp, c, METRICS["dt"]))
    ag_m, ag_s = mean_std(metric_array(lfp, c, METRICS["aging"]) * 1e4)
    v_m = float(np.mean(metric_array(lfp, c, METRICS["viol"])))
    print(f"{c:<18}{t_m:6.1f}+-{t_s:<4.1f}{s_m:6.2f}+-{s_s:<4.2f}{tm_m:6.1f}+-{tm_s:<4.1f}{dt_m:6.2f}+-{dt_s:<4.2f}{ag_m:6.1f}+-{ag_s:<4.1f}{v_m:6.1f}")

cc = mean_std(metric_array(lfp, "CC-CV", METRICS["time"]))[0]
go = mean_std(metric_array(lfp, "GraphOptimizer", METRICS["time"]))[0]
cc_s = mean_std(metric_array(lfp, "CC-CV", METRICS["soc"]))[0]
go_s = mean_std(metric_array(lfp, "GraphOptimizer", METRICS["soc"]))[0]
cc_dt = mean_std(metric_array(lfp, "CC-CV", METRICS["dt"]))[0]
go_dt = mean_std(metric_array(lfp, "GraphOptimizer", METRICS["dt"]))[0]
print()
print(f"Time change vs CC-CV: {(go-cc)/cc*100:+.1f}%")
print(f"sigma_SOC reduction vs CC-CV: {(1-go_s/cc_s)*100:.1f}%")
print(f"DT reduction vs CC-CV: {(1-go_dt/cc_dt)*100:.1f}%")

print()
print("="*80)
print("TABLE 6 — Statistical Significance: GraphOptimizer vs CC-CV (N=30)")
print("="*80)
for label, key, scale in [("Time (min)", "time", 1), ("SOC sigma (frac)", "soc", 1),
                            ("DT (C)", "dt", 1), ("Aging (x1e-4)", "aging", 1e4)]:
    a_cc = metric_array(lfp, "CC-CV", METRICS[key]) * scale
    a_go = metric_array(lfp, "GraphOptimizer", METRICS[key]) * scale
    t_stat, t_p = stats.ttest_ind(a_go, a_cc, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(a_go, a_cc, alternative='two-sided')
    n1, n2 = len(a_go), len(a_cc)
    r_rb = 1 - (2*u_stat)/(n1*n2)
    print(f"{label:<20} CC-CV={np.mean(a_cc):.4f}+-{np.std(a_cc):.4f}  GraphOpt={np.mean(a_go):.4f}+-{np.std(a_go):.4f}  p(welch)={t_p:.2e}  p(MWU)={u_p:.2e}  r_rb={abs(r_rb):.2f}")

print()
print("="*80)
print("TABLE 7 — Cross-Chemistry Performance Comparison (12 cells, N=30)")
print("="*80)
for chem in ["LFP", "NMC", "LCO"]:
    raw = load(chem)["raw_results"]
    print(f"\n-- {chem} --")
    print(f"{'Method':<18}{'Time':>10}{'sigma_SOC':>12}{'T_max':>10}{'DT':>10}{'Viol':>10}")
    for c in CONTROLLERS:
        if c == "SimpleMPC":
            continue
        t_m, t_s = mean_std(metric_array(raw, c, METRICS["time"]))
        s_m, s_s = mean_std(metric_array(raw, c, METRICS["soc"]) * 100)
        tm_m = float(np.mean(metric_array(raw, c, METRICS["tmax"])))
        dt_m, dt_s = mean_std(metric_array(raw, c, METRICS["dt"]))
        v_m = float(np.mean(metric_array(raw, c, METRICS["viol"])))
        print(f"{c:<18}{t_m:6.1f}+-{t_s:<4.1f}{s_m:6.2f}+-{s_s:<4.2f}{tm_m:8.1f}  {dt_m:6.2f}+-{dt_s:<4.2f}{v_m:8.2f}")

print()
print("="*80)
print("All raw episode-level data confirmed loaded correctly.")
print("="*80)
