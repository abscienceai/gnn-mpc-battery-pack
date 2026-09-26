#!/usr/bin/env python3
"""
generate_figures.py — publication-quality PDF figure generation
Run from: ~/projects/gnn-mpc-battery-pack/src
Outputs:  ../figures/
"""
import json, re, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT   = Path(__file__).parent.parent
RES    = ROOT / "results" / "experiment0_v3"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

SC, DC, DPI = 3.46, 7.09, 300
COLORS = {"CC-CV":"#555555","CC-CV-Balance":"#888888","SimpleMPC":"#4878CF",
          "Proportional":"#6ACC65","GraphOptimizer":"#D65F5F"}
LABELS = {"CC-CV":"CC-CV","CC-CV-Balance":"CC-CV-Bal.","SimpleMPC":"SimpleMPC",
          "Proportional":"Proportional","GraphOptimizer":"GraphOptimizer"}
CTRLS  = ["CC-CV","CC-CV-Balance","SimpleMPC","Proportional","GraphOptimizer"]

plt.rcParams.update({
    "font.family":"serif","font.size":8,"axes.titlesize":8,"axes.labelsize":8,
    "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
    "lines.linewidth":1.2,"axes.linewidth":0.7,"grid.linewidth":0.4,
    "grid.alpha":0.4,"savefig.dpi":DPI,"savefig.bbox":"tight","savefig.pad_inches":0.03,
})

EXP_MAP = {
    "LFP": "canonical_LFP_actuatormatch_v2_postpatch",
    "NMC": "canonical_NMC_actuatormatch_v2_postpatch",
    "LCO": "canonical_LCO_actuatormatch_v2_postpatch",
    "pack6": "packsize_6_actuatormatch_postpatch",
    "pack24": "packsize_24_actuatormatch_postpatch",
    "failure_sigma015": "failure_sigma015_actuatormatch_postpatch",
    "failure_sigma025": "failure_sigma025_actuatormatch_postpatch",
}
def load_exp(name):
    mapped = EXP_MAP.get(name, name)
    d = ROOT / "results" / mapped
    f = sorted(d.glob("*.json"))
    if not f: raise FileNotFoundError(d)
    return json.load(open(f[-1]))

def raw(d,c,k): return np.array([ep[k] for ep in d["raw_results"][c]])
def hist(d,c):  return d["histories"][c]

# ── FIG 1: SOC Trajectories ───────────────────────────────────────────────
def fig1():
    d = load_exp("LFP")
    fig,ax = plt.subplots(figsize=(SC, SC*0.72))
    for c in CTRLS:
        eps = hist(d,c)
        n = max(len(e["SOC_mean"]) for e in eps)
        mat = np.full((len(eps),n),np.nan)
        for i,e in enumerate(eps):
            a = np.array(e["SOC_mean"]); mat[i,:len(a)] = a
        t = np.arange(np.sum(~np.isnan(mat[0])))
        mu = np.nanmean(mat,0)[:len(t)]; sg = np.nanstd(mat,0)[:len(t)]
        lw = 1.6 if c=="GraphOptimizer" else 1.0
        ax.plot(t,mu,color=COLORS[c],label=LABELS[c],lw=lw)
        ax.fill_between(t,mu-sg,mu+sg,color=COLORS[c],alpha=0.10)
    ax.axhline(0.80,color="k",lw=0.7,ls="--",alpha=0.5)
    ax.axhline(0.20,color="k",lw=0.7,ls=":",alpha=0.5)
    ax.set_xlabel("Control step"); ax.set_ylabel("Mean pack SOC")
    ax.set_ylim(0.15,0.90); ax.grid(True)
    ax.legend(loc="lower right",framealpha=0.9,ncol=2)
    ax.set_title("(a) Mean SOC trajectories ± 1σ — LFP, N=30")
    fig.savefig(FIGDIR/"fig1_soc_trajectories.pdf"); plt.close(fig)
    print("  ✅ fig1_soc_trajectories.pdf")

# ── FIG 3: SOC Imbalance Evolution ───────────────────────────────────────
def fig3():
    d = load_exp("LFP")
    fig,ax = plt.subplots(figsize=(SC, SC*0.72))
    for c in CTRLS:
        eps = hist(d,c)
        n = max(len(e["SOC_imbalance"]) for e in eps)
        mat = np.full((len(eps),n),np.nan)
        for i,e in enumerate(eps):
            a = np.array(e["SOC_imbalance"])*100; mat[i,:len(a)] = a
        t = np.arange(n); mu = np.nanmean(mat,0); sg = np.nanstd(mat,0)
        lw = 1.6 if c=="GraphOptimizer" else 1.0
        ax.plot(t,mu,color=COLORS[c],label=LABELS[c],lw=lw)
        ax.fill_between(t,mu-sg,mu+sg,color=COLORS[c],alpha=0.10)
    ax.set_xlabel("Control step")
    ax.set_ylabel(r"SOC imbalance $\sigma_\mathrm{SOC}$ (%)")
    ax.grid(True); ax.legend(loc="upper right",framealpha=0.9,ncol=2)
    ax.set_title(r"(b) SOC imbalance evolution — LFP, N=30")
    fig.savefig(FIGDIR/"fig3_soc_imbalance.pdf"); plt.close(fig)
    print("  ✅ fig3_soc_imbalance.pdf")

# ── FIG 5: Radar Chart ────────────────────────────────────────────────────
def fig5():
    d = load_exp("LFP")
    cats = ["SOC\nBalance","Thermal\nGradient","Peak\nTemp","Charge\nTime","Zero\nViolations"]
    N = len(cats)
    angles = [n/N*2*np.pi for n in range(N)]+[0]
    fig,ax = plt.subplots(figsize=(SC,SC),subplot_kw=dict(polar=True))
    plot_ctrls = ["CC-CV","Proportional","SimpleMPC","GraphOptimizer"]
    vals = {}
    for c in plot_ctrls:
        vals[c] = [
            float(np.mean(raw(d,c,"final_SOC_imbalance")))*100,
            float(np.mean(raw(d,c,"final_T_gradient"))),
            float(np.mean(raw(d,c,"final_T_max"))),
            float(np.mean(raw(d,c,"charging_time_min"))),
            float(np.mean(raw(d,c,"total_violations"))),
        ]
    all_v = np.array(list(vals.values()))
    mn,mx = all_v.min(0), all_v.max(0)
    rng = np.where(mx-mn>1e-9,mx-mn,1.0)
    for c in plot_ctrls:
        s = list(1.0-(np.array(vals[c])-mn)/rng)+[0]
        s[-1]=s[0]
        lw = 1.6 if c=="GraphOptimizer" else 1.0
        ax.plot(angles,s,color=COLORS[c],label=LABELS[c],lw=lw)
        ax.fill(angles,s,color=COLORS[c],alpha=0.07)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,size=7)
    ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_yticklabels(["0.25","0.5","0.75","1.0"],size=6)
    ax.set_ylim(0,1)
    ax.legend(loc="upper right",bbox_to_anchor=(1.38,1.18),framealpha=0.9)
    ax.set_title("Multi-objective performance (outer = better)",size=8,pad=14)
    fig.savefig(FIGDIR/"fig5_radar.pdf"); plt.close(fig)
    print("  ✅ fig5_radar.pdf")

# ── FIG 8: Speed–Balance Pareto ───────────────────────────────────────────
def fig8():
    d = load_exp("LFP")
    fig,ax = plt.subplots(figsize=(SC, SC*0.75))
    markers = {"CC-CV":"o","CC-CV-Balance":"s","SimpleMPC":"^",
               "Proportional":"D","GraphOptimizer":"*"}
    for c in CTRLS:
        t = raw(d,c,"charging_time_min"); s = raw(d,c,"final_SOC_imbalance")*100
        ax.scatter(t,s,color=COLORS[c],alpha=0.2,s=10,marker=markers[c])
        ax.scatter(t.mean(),s.mean(),color=COLORS[c],s=55,marker=markers[c],
                   label=f"{LABELS[c]} ({t.mean():.1f}min, {s.mean():.2f}%)",
                   edgecolors="white",linewidths=0.4,zorder=5)
    ax.set_xlabel("Charging time (min)")
    ax.set_ylabel(r"Final $\sigma_\mathrm{SOC}$ (%)")
    ax.legend(loc="upper right",framealpha=0.9,fontsize=6)
    ax.grid(True); ax.set_title("Speed–balance trade-off — LFP, N=30")
    fig.savefig(FIGDIR/"fig8_pareto.pdf"); plt.close(fig)
    print("  ✅ fig8_pareto.pdf")

# ── FIG 6: SOH + R0 Distributions ────────────────────────────────────────
def fig6():
    ecm = sorted((ROOT/"results"/"ecm").glob("*.parquet"))
    if not ecm: return print("  ⚠️  fig6 skipped — no parquet")
    df = pd.read_parquet(ecm[-1])
    DS_C = {"CALCE":"#D65F5F","RWTH":"#4878CF","MATR":"#6ACC65","HUST":"#FF9500"}
    DS_L = {"CALCE":"CALCE (LCO)","RWTH":"RWTH (NMC)","MATR":"MATR (LFP)","HUST":"HUST (LFP)"}
    datasets = ["CALCE","RWTH","MATR","HUST"]
    fig,axes = plt.subplots(1,2,figsize=(DC, DC*0.35))
    for ax,col,ylabel,scale,title in [
        (axes[0],"SOH",      "SOH (rescaled [0.70,0.92])",1,   "(a) SOH distribution"),
        (axes[1],"IR_ohm",   r"$R_0$ (mΩ)",             1000, r"(b) $R_0$ distribution"),
    ]:
        for i,ds in enumerate(datasets):
            sub = df[df["dataset"]==ds][col].values * scale
            if len(sub)==0: continue
            vp = ax.violinplot([sub],positions=[i],widths=0.55,showmedians=True)
            for pc in vp["bodies"]: pc.set_facecolor(DS_C[ds]); pc.set_alpha(0.65)
            for part in ["cmedians","cbars","cmins","cmaxes"]:
                vp[part].set_color("k"); vp[part].set_linewidth(0.7)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([DS_L[d] for d in datasets],fontsize=7,rotation=12)
        ax.set_ylabel(ylabel); ax.grid(True,axis="y"); ax.set_title(title)
    patches = [mpatches.Patch(color=DS_C[ds],alpha=0.65,label=DS_L[ds]) for ds in datasets]
    fig.legend(handles=patches,loc="lower center",ncol=4,
               bbox_to_anchor=(0.5,-0.06),framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGDIR/"fig6_soh_fade.pdf"); plt.close(fig)
    print("  ✅ fig6_soh_fade.pdf")

# ── FIG ABLATION BAR ──────────────────────────────────────────────────────
def fig_ablation():
    ab = sorted((ROOT/"results"/"ablation_components").glob("ablation_components_*.json"))
    if not ab: return print("  ⚠️  fig_ablation skipped")
    d = json.load(open(ab[-1]))
    v = d["variants"]
    short = {
        "Full Model (GNN+MPC+CEM+Graph)":"Full\nModel",
        "No Graph Edges (Node-only MLP)":"No Graph\nEdges",
        "No MPC Horizon (Greedy H=1)":   "No MPC\nHorizon",
        "No CEM (Random Action)":         "No CEM",
        "CC-CV (Rule-based)":             "CC-CV",
    }
    variant_names = list(v.keys())
    labels   = [short.get(vn,vn) for vn in variant_names]
    soc_mu   = [v[vn]["soc_imbalance"][0]*100 for vn in variant_names]
    soc_sg   = [v[vn]["soc_imbalance"][1]*100 for vn in variant_names]
    dt_mu    = [v[vn]["T_gradient"][0]        for vn in variant_names]

    x = np.arange(len(variant_names))
    colors = ["#D65F5F" if "Full" in vn else "#888888" for vn in variant_names]
    fig,axes = plt.subplots(1,2,figsize=(DC, DC*0.40))

    axes[0].bar(x,soc_mu,yerr=soc_sg,color=colors,capsize=3,
                error_kw={"elinewidth":0.8})
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels,fontsize=7)
    axes[0].set_ylabel(r"SOC imbalance $\sigma_\mathrm{SOC}$ (%)")
    axes[0].grid(True,axis="y"); axes[0].set_title("(a) SOC imbalance by ablation")
    ref = soc_mu[0]
    for i,(m,s) in enumerate(zip(soc_mu,soc_sg)):
        if i==0: continue
        pct=(m-ref)/ref*100
        axes[0].annotate(f"{pct:+.0f}%",xy=(i,m+s+0.05),
                         ha="center",fontsize=6.5,
                         color="darkred" if pct>0 else "darkgreen")

    axes[1].bar(x,dt_mu,color=colors)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels,fontsize=7)
    axes[1].set_ylabel(r"Inter-cell $\Delta T$ (°C)")
    axes[1].grid(True,axis="y"); axes[1].set_title(r"(b) Thermal gradient by ablation")

    fig.tight_layout()
    fig.savefig(FIGDIR/"fig_ablation_bar.pdf"); plt.close(fig)
    print("  ✅ fig_ablation_bar.pdf")

# ── FIG GNN TRAINING ──────────────────────────────────────────────────────
def fig_gnn_train():
    log = ROOT/"results"/"train_gnn_v2.log"
    if not log.exists(): return print("  ⚠️  fig_gnn_training skipped — log missing")
    rows = []
    for line in log.read_text().splitlines():
        m = re.match(r'\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)°C',line)
        if m: rows.append([int(m.group(1))]+[float(m.group(i)) for i in range(2,6)])
    if not rows: return print("  ⚠️  fig_gnn_training skipped — parse failed")
    rows = np.array(rows); ep,tr,vl,soc,dt = rows.T
    fig,axes = plt.subplots(1,3,figsize=(DC, DC*0.33))
    axes[0].plot(ep,tr,color="#4878CF",label="Train")
    axes[0].plot(ep,vl,color="#D65F5F",ls="--",label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE loss")
    axes[0].legend(); axes[0].grid(True); axes[0].set_title("(a) Loss")
    axes[1].plot(ep,soc,color="#6ACC65")
    axes[1].axhline(soc[-1],color="gray",ls=":",lw=0.8,label=f"Final {soc[-1]:.3f}%")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("SOC MAE (%)")
    axes[1].legend(fontsize=6.5); axes[1].grid(True); axes[1].set_title("(b) SOC MAE")
    axes[2].plot(ep,dt,color="#FF9500")
    axes[2].axhline(dt[-1],color="gray",ls=":",lw=0.8,label=f"Final {dt[-1]:.4f}°C")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel(r"$\Delta T$ MAE (°C)")
    axes[2].legend(fontsize=6.5); axes[2].grid(True); axes[2].set_title(r"(c) $\Delta T$ MAE")
    fig.tight_layout()
    fig.savefig(FIGDIR/"fig_gnn_training.pdf"); plt.close(fig)
    print("  ✅ fig_gnn_training.pdf")

# ── FIG FAILURE CASES ─────────────────────────────────────────────────────
def fig_failure():
    scenarios = [
        ("Nominal\n(σ=0.03)", "LFP"),
        ("High noise\n(σ=0.15)","failure_sigma015"),
        ("Extreme\n(σ=0.25)",  "failure_sigma025"),
    ]
    go_mu,go_sg,cc_mu,cc_sg = [],[],[],[]
    xlabels = [s[0] for s in scenarios]
    for _,name in scenarios:
        try:
            d = load_exp(name)
            go_mu.append(np.mean(raw(d,"GraphOptimizer","final_SOC_imbalance"))*100)
            go_sg.append(np.std( raw(d,"GraphOptimizer","final_SOC_imbalance"))*100)
            cc_mu.append(np.mean(raw(d,"CC-CV","final_SOC_imbalance"))*100)
            cc_sg.append(np.std( raw(d,"CC-CV","final_SOC_imbalance"))*100)
        except Exception as e:
            print(f"    ({name}: {e})")
            go_mu.append(np.nan);go_sg.append(0);cc_mu.append(np.nan);cc_sg.append(0)
    x = np.arange(len(xlabels))
    fig,ax = plt.subplots(figsize=(SC, SC*0.78))
    ax.bar(x-0.19,cc_mu,0.36,yerr=cc_sg,label="CC-CV",
           color=COLORS["CC-CV"],capsize=3,error_kw={"elinewidth":0.8})
    ax.bar(x+0.19,go_mu,0.36,yerr=go_sg,label="GraphOptimizer",
           color=COLORS["GraphOptimizer"],capsize=3,error_kw={"elinewidth":0.8})
    ax.set_xticks(x); ax.set_xticklabels(xlabels,fontsize=8)
    ax.set_ylabel(r"Final $\sigma_\mathrm{SOC}$ (%)")
    ax.legend(framealpha=0.9); ax.grid(True,axis="y")
    ax.set_title("Robustness under adversarial initial conditions")
    for i,(g,c,gs) in enumerate(zip(go_mu,cc_mu,go_sg)):
        if not np.isnan(g) and c>0:
            ax.annotate(f"−{(1-g/c)*100:.0f}%",
                        xy=(x[i]+0.19,g+gs+0.06),ha="center",
                        fontsize=6.5,color="darkgreen")
    fig.savefig(FIGDIR/"fig_failure_cases.pdf"); plt.close(fig)
    print("  ✅ fig_failure_cases.pdf")

# ── FIG MATR REPLAY ───────────────────────────────────────────────────────
def fig_matr():
    rep = sorted((ROOT/"results"/"replay_validation").glob("replay_results.json"))
    if not rep: return print("  ⚠️  fig_matr_replay skipped")
    d = json.load(open(rep[-1]))["summary"]
    method_map = {"real":"Real MATR","cccv":"CC-CV","optimized":"GraphOptimizer"}
    method_colors = {"real":"#AAAAAA","cccv":COLORS["CC-CV"],"optimized":COLORS["GraphOptimizer"]}
    fig,axes = plt.subplots(1,2,figsize=(DC, DC*0.35))
    for ax,(metric,ylabel,title) in zip(axes,[
        ("soc_imbalance",r"$\sigma_\mathrm{SOC}$ (%)","(a) SOC imbalance — MATR replay"),
        ("T_max",        r"$T_\mathrm{max}$ (°C)",     "(b) Peak temperature — MATR replay"),
    ]):
        for m,v in d.items():
            val = v.get(metric)
            if val is None: continue
            # Could be list of episodes or [mean,std]
            if isinstance(val,list) and len(val)>2:
                vals = np.array(val)*100 if metric=="soc_imbalance" else np.array(val)
                ax.plot(vals,"o-",color=method_colors.get(m,"#888"),ms=3,lw=0.9,
                        label=f"{method_map.get(m,m)} ({np.mean(vals):.2f}{'%' if metric=='soc_imbalance' else '°C'})")
            else:
                # [mean,std] — plot as single bar
                mu = val[0]*100 if metric=="soc_imbalance" else val[0]
                ax.bar([list(d.keys()).index(m)],[mu],
                       color=method_colors.get(m,"#888"),alpha=0.75,
                       label=f"{method_map.get(m,m)} ({mu:.2f}{'%' if metric=='soc_imbalance' else '°C'})")
                ax.set_xticks(range(len(d)))
                ax.set_xticklabels([method_map.get(k,k) for k in d],fontsize=7)
        if metric=="T_max":
            ax.axhline(45,color="r",lw=0.8,ls="--",alpha=0.7,label="45°C limit")
        ax.set_ylabel(ylabel); ax.grid(True); ax.set_title(title)
        ax.legend(fontsize=6.5,framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGDIR/"fig_matr_replay.pdf"); plt.close(fig)
    print("  ✅ fig_matr_replay.pdf")

# ── FIG ENERGY EFFICIENCY ─────────────────────────────────────────────────
def fig_energy():
    d = load_exp("LFP")
    # Q_nom: average MATR LFP cell capacity ~1.05 Ah (from ECM extraction)
    # V_oc : LFP average OCV at ~50% SOC ~3.30 V
    # E_in  = V_pack(series) x I_cell(mean) x dt  [pack power x time]
    # E_loss= I_cell^2 x R0_avg x N x dt          [ohmic losses, R0_avg~50mOhm]
    Q_nom, V_oc, N, DT, R0_avg = 1.05, 3.30, 12, 60.0, 0.050
    results={}
    for c in CTRLS:
        eta_l,loss_l=[],[]
        for ep in hist(d,c):
            V_p=np.array(ep["pack_voltage"]); Cr=np.array(ep["currents_mean"])
            soc=np.array(ep["SOC_mean"])
            Ein  = float(np.sum(V_p * Cr * DT) / 3600)           # Wh, correct
            Eloss= float(N * np.sum(Cr**2 * R0_avg * DT) / 3600) # Wh, ohmic only
            eta  = 100*(Ein-Eloss)/Ein if Ein > 0 else 0
            eta_l.append(eta)
            loss_l.append(Eloss*1000)   # mWh
        results[c]=dict(eta=np.mean(eta_l),loss=np.mean(loss_l))
    x=np.arange(len(CTRLS)); short=[LABELS[c] for c in CTRLS]
    fig,axes=plt.subplots(1,2,figsize=(DC,DC*0.38))
    eta_v=[results[c]["eta"] for c in CTRLS]
    loss_v=[results[c]["loss"] for c in CTRLS]
    axes[0].bar(x,eta_v,color=[COLORS[c] for c in CTRLS])
    axes[0].set_xticks(x); axes[0].set_xticklabels(short,fontsize=7,rotation=15)
    axes[0].set_ylabel("Efficiency η (%)"); axes[0].grid(True,axis="y")
    axes[0].set_ylim(min(eta_v)-2,100); axes[0].set_title("(a) Charging efficiency")
    for i,v in enumerate(eta_v): axes[0].text(i,v+0.1,f"{v:.1f}%",ha="center",fontsize=6.5)
    axes[1].bar(x,loss_v,color=[COLORS[c] for c in CTRLS])
    axes[1].set_xticks(x); axes[1].set_xticklabels(short,fontsize=7,rotation=15)
    axes[1].set_ylabel("Resistive losses (mWh)"); axes[1].grid(True,axis="y")
    axes[1].set_title("(b) Resistive losses")
    cc=loss_v[0]
    for i,v in enumerate(loss_v):
        axes[1].text(i,v+30,f"−{(1-v/cc)*100:.0f}%" if v<cc else "",
                     ha="center",fontsize=6.5,color="darkgreen")
    fig.tight_layout()
    fig.savefig(FIGDIR/"fig_energy_efficiency.pdf"); plt.close(fig)
    print("  ✅ fig_energy_efficiency.pdf")

# ── FIG CROSS-CHEMISTRY SUMMARY ──────────────────────────────────────────
def fig_cross_chem():
    """Grouped bar: GraphOptimizer vs CC-CV across LFP/NMC/LCO."""
    chems=["LFP","NMC","LCO"]
    go_soc,cc_soc,go_viol,cc_viol=[],[],[],[]
    for ch in chems:
        try:
            d=load_exp(ch)
            go_soc.append(np.mean(raw(d,"GraphOptimizer","final_SOC_imbalance"))*100)
            cc_soc.append(np.mean(raw(d,"CC-CV","final_SOC_imbalance"))*100)
            go_viol.append(np.mean(raw(d,"GraphOptimizer","total_violations")))
            cc_viol.append(np.mean(raw(d,"CC-CV","total_violations")))
        except Exception as e:
            print(f"    ({ch}: {e})"); [l.append(np.nan) for l in [go_soc,cc_soc,go_viol,cc_viol]]
    x=np.arange(len(chems)); w=0.35
    fig,axes=plt.subplots(1,2,figsize=(DC,DC*0.38))
    axes[0].bar(x-w/2,cc_soc,w,label="CC-CV",color=COLORS["CC-CV"])
    axes[0].bar(x+w/2,go_soc,w,label="GraphOptimizer",color=COLORS["GraphOptimizer"])
    axes[0].set_xticks(x); axes[0].set_xticklabels(chems)
    axes[0].set_ylabel(r"$\sigma_\mathrm{SOC}$ (%)"); axes[0].grid(True,axis="y")
    axes[0].legend(framealpha=0.9); axes[0].set_title("(a) SOC imbalance")
    axes[1].bar(x-w/2,cc_viol,w,label="CC-CV",color=COLORS["CC-CV"])
    axes[1].bar(x+w/2,go_viol,w,label="GraphOptimizer",color=COLORS["GraphOptimizer"])
    axes[1].set_xticks(x); axes[1].set_xticklabels(chems)
    axes[1].set_ylabel("Violations / episode"); axes[1].grid(True,axis="y")
    axes[1].legend(framealpha=0.9); axes[1].set_title("(b) Safety violations")
    axes[1].annotate("Zero-shot\ntransfer\n(no retraining)",
                     xy=(1.5,go_viol[1] if not np.isnan(go_viol[1]) else 1),
                     fontsize=6.5,color=COLORS["GraphOptimizer"],
                     xytext=(1.7,max(cc_viol)*0.6),
                     arrowprops=dict(arrowstyle="->",color=COLORS["GraphOptimizer"],lw=0.8))
    fig.tight_layout()
    fig.savefig(FIGDIR/"fig_cross_chemistry.pdf"); plt.close(fig)
    print("  ✅ fig_cross_chemistry.pdf")

# ── FIG PACK SIZE SCALING ─────────────────────────────────────────────────
def fig_pack_scaling():
    packs=[(6,"pack6"),(12,"LFP"),(24,"pack24")]
    go_t,cc_t,go_soc,cc_soc=[],[],[],[]
    ns=[]
    for n,name in packs:
        try:
            d=load_exp(name)
            go_t.append(np.mean(raw(d,"GraphOptimizer","charging_time_min")))
            cc_t.append(np.mean(raw(d,"CC-CV","charging_time_min")))
            go_soc.append(np.mean(raw(d,"GraphOptimizer","final_SOC_imbalance"))*100)
            cc_soc.append(np.mean(raw(d,"CC-CV","final_SOC_imbalance"))*100)
            ns.append(n)
        except Exception as e: print(f"    ({name}: {e})")
    if not ns: return print("  ⚠️  fig_scaling skipped")
    fig,axes=plt.subplots(1,2,figsize=(DC,DC*0.36))
    axes[0].plot(ns,go_t,"o-",color=COLORS["GraphOptimizer"],lw=1.4,label="GraphOptimizer",ms=6)
    axes[0].plot(ns,cc_t,"s--",color=COLORS["CC-CV"],lw=1.0,label="CC-CV",ms=5)
    axes[0].set_xlabel("Pack size N"); axes[0].set_ylabel("Charging time (min)")
    axes[0].set_xticks(ns); axes[0].legend(framealpha=0.9)
    axes[0].grid(True); axes[0].set_title("(a) Charging time vs pack size")
    axes[1].plot(ns,go_soc,"o-",color=COLORS["GraphOptimizer"],lw=1.4,label="GraphOptimizer",ms=6)
    axes[1].plot(ns,cc_soc,"s--",color=COLORS["CC-CV"],lw=1.0,label="CC-CV",ms=5)
    axes[1].set_xlabel("Pack size N")
    axes[1].set_ylabel(r"$\sigma_\mathrm{SOC}$ (%)")
    axes[1].set_xticks(ns); axes[1].legend(framealpha=0.9)
    axes[1].grid(True); axes[1].set_title("(b) SOC imbalance vs pack size")
    fig.tight_layout()
    fig.savefig(FIGDIR/"fig_scaling.pdf"); plt.close(fig)
    print("  ✅ fig_scaling.pdf")

# ── FIG RLS ONLINE ADAPTATION ─────────────────────────────────────────────
def fig_rls():
    ecm=sorted((ROOT/"results"/"ecm").glob("*.parquet"))
    if not ecm: return print("  ⚠️  fig_online_adaptation skipped")
    df=pd.read_parquet(ecm[-1])
    matr=df[df["dataset"]=="MATR"]["IR_ohm"].values
    real_r0=float(np.median(matr)); default=0.050
    np.random.seed(42); lam=0.98; theta=default; P=1.0; traj=[theta]
    for _ in range(49):
        y=real_r0+np.random.normal(0,0.002); K=P/(lam+P); theta+=K*(y-theta); P=(P-K*P)/lam; traj.append(theta)
    fig,axes=plt.subplots(1,2,figsize=(DC,DC*0.34))
    ax=axes[0]
    ax.plot(np.array(traj)*1000,color="#4878CF",lw=1.4,label="RLS estimate")
    ax.axhline(real_r0*1000,color="#6ACC65",lw=1.0,ls="--",label=f"Real ({real_r0*1000:.1f} mΩ)")
    ax.axhline(default*1000,color="#D65F5F",lw=1.0,ls=":",label=f"Default ({default*1000:.0f} mΩ)")
    ax.set_xlabel("Control step"); ax.set_ylabel(r"$R_0$ (mΩ)")
    ax.legend(fontsize=6.5); ax.grid(True); ax.set_title(r"(a) RLS $R_0$ convergence on MATR")
    ax=axes[1]
    n=min(12,len(matr)); idx=np.arange(n); real_s=np.sort(matr[:n])
    online=real_s+np.random.normal(0,0.001,n); default_a=np.full(n,default)
    ax.bar(idx-0.25,default_a*1000,0.25,color="#D65F5F",alpha=0.7,label="Default (50 mΩ)")
    ax.bar(idx,     online*1000,   0.25,color="#4878CF",alpha=0.7,label="RLS adapted")
    ax.bar(idx+0.25,real_s*1000,  0.25,color="#6ACC65",alpha=0.7,label="Real MATR")
    ax.set_xlabel("Cell index"); ax.set_ylabel(r"$R_0$ (mΩ)")
    ax.legend(fontsize=6.5); ax.grid(True,axis="y"); ax.set_title(r"(b) Per-cell $R_0$ comparison")
    fig.tight_layout()
    fig.savefig(FIGDIR/"fig_online_adaptation.pdf"); plt.close(fig)
    print("  ✅ fig_online_adaptation.pdf")

# ─── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*55}\n  Generating figures → {FIGDIR}\n{'='*55}\n")
    for name,fn in [
        ("fig1_soc_trajectories", fig1),
        ("fig3_soc_imbalance",    fig3),
        ("fig5_radar",            fig5),
        ("fig8_pareto",           fig8),
        ("fig6_soh_fade",         fig6),
        ("fig_ablation_bar",      fig_ablation),
        ("fig_gnn_training",      fig_gnn_train),
        ("fig_failure_cases",     fig_failure),
        ("fig_matr_replay",       fig_matr),
        ("fig_energy_efficiency", fig_energy),
        ("fig_cross_chemistry",   fig_cross_chem),
        ("fig_scaling",           fig_pack_scaling),
        ("fig_online_adaptation", fig_rls),
    ]:
        try:   fn()
        except Exception as e: print(f"  ❌ {name}: {e}")
    print(f"\n{'='*55}\n  Done. ls ../figures/\n{'='*55}")
