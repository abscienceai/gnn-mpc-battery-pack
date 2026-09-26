"""
Reports GNN one-step-ahead prediction MAE at each horizon step h=1..5
by comparing recursive surrogate rollout predictions against physics ground truth.
"""
import sys, csv, numpy as np, torch
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))
from graph_battery_pack import PackGNN, build_pack_from_ecm
from safe_fast_charge_optimizer import default_config

DEVICE='cuda'
ecm=sorted((Path(__file__).parent.parent/'results/ecm').glob('ecm_params_*.parquet'))[-1]
ckpt=torch.load(str(sorted((Path(__file__).parent.parent/'results/models').glob('pack_gnn_*.pt'))[-1]),map_location=DEVICE)
gnn=PackGNN(node_feat=7,edge_feat=3,hidden=64,n_layers=3).to(DEVICE)
gnn.load_state_dict(ckpt.get('model_state',ckpt)); gnn.eval()
cfg=default_config()

H = 5
soc_errs = {h: [] for h in range(1, H+1)}
dT_errs  = {h: [] for h in range(1, H+1)}

N_EP = 30
for ep in range(N_EP):
    np.random.seed(ep*7); torch.manual_seed(ep*7)
    pack = build_pack_from_ecm(ecm_parquet=ecm, n_cells=12, chemistry='LFP',
                                soc_init=0.20, soc_noise=0.03, T_amb=25.0,
                                seed=ep*7)
    Q_nom = np.array([c.Q_nom_Ah for c in pack.cells])
    I_max = cfg['I_max_C'] * Q_nom
    action = I_max * 0.5  # fixed representative action

    # Ground truth: apply action for H physics steps, track state each step
    pack_true = deepcopy(pack)
    true_socs = [np.array([c.SOC for c in pack_true.cells])]
    true_temps = [np.array([c.T_C for c in pack_true.cells])]
    for h in range(H):
        pack_true.step(action, dt=cfg['dt_s'])
        true_socs.append(np.array([c.SOC for c in pack_true.cells]))
        true_temps.append(np.array([c.T_C for c in pack_true.cells]))

    # Surrogate: recursive GNN rollout (mimics _gnn_rollout_cost logic)
    g = pack.to_torch_graph()
    x = g['x'].to(DEVICE); edge_index = g['edge_index'].to(DEVICE); edge_attr = g['edge_attr'].to(DEVICE)
    soc_virtual = np.array([c.SOC for c in pack.cells])
    T_virtual   = np.array([c.T_C for c in pack.cells])

    with torch.no_grad():
        for h in range(1, H+1):
            out = gnn(x, edge_index, edge_attr)
            soc_pred = out['soc_pred'].cpu().numpy().flatten()
            dT_pred  = out['delta_T_pred'].cpu().numpy().flatten()

            i_ratio = action / (I_max + 1e-9)
            coulomb = (action * cfg['dt_s']) / (Q_nom * 3600.0)
            soc_virtual = np.clip(soc_virtual + coulomb + 0.3*(soc_pred - soc_virtual), 0, 1)
            T_virtual   = T_virtual + dT_pred * i_ratio * 3.0

            soc_errs[h].append(np.mean(np.abs(soc_virtual - true_socs[h])) * 100)
            dT_errs[h].append(np.mean(np.abs(T_virtual - true_temps[h])))

            # Update node/edge features for next GNN call
            x_np = x.cpu().numpy().copy()
            x_np[:,0] = soc_virtual; x_np[:,1] = (T_virtual-25)/30
            x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
            soc_diffs = soc_virtual[:-1]-soc_virtual[1:]; T_diffs = T_virtual[:-1]-T_virtual[1:]
            e_np = edge_attr.cpu().numpy().copy(); half=e_np.shape[0]//2
            for e in range(half):
                e_np[2*e,0]=T_diffs[e]/10 if e<len(T_diffs) else 0
                e_np[2*e,1]=soc_diffs[e] if e<len(soc_diffs) else 0
                e_np[2*e+1,0]=-e_np[2*e,0]; e_np[2*e+1,1]=-e_np[2*e,1]
            edge_attr = torch.tensor(e_np, dtype=torch.float32, device=DEVICE)

print(f"{'h':>4} {'SOC MAE%':>10} {'dT MAE':>10}")
rows=[]
for h in range(1, H+1):
    sm = np.mean(soc_errs[h]); dm = np.mean(dT_errs[h])
    print(f"{h:>4} {sm:>10.4f} {dm:>10.4f}")
    rows.append({"h":h,"soc_mae_pct":round(sm,4),"dT_mae":round(dm,4)})

out = Path(__file__).parent.parent/'results/horizon_wise_mae.csv'
with open(out,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"Saved → {out}")
