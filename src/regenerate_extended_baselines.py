"""Regenerate Table 14 (extended baselines) with CURRENT code (post multi-step-rollout fix)
to verify whether the Table 4 vs Table 14 discrepancy was a code-version issue."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from graph_battery_pack import PackGNN, build_pack_from_ecm
from safe_fast_charge_optimizer import default_config, GraphGuidedOptimizer, CCCVController

DEVICE='cuda'
ckpt=torch.load('results/models/pack_gnn_20260630_220402.pt',map_location=DEVICE)
gnn=PackGNN(node_feat=7,edge_feat=3,hidden=64,n_layers=3).to(DEVICE)
gnn.load_state_dict(ckpt.get('model_state',ckpt)); gnn.eval()
ecm=sorted(Path('results/ecm').glob('ecm_params_*.parquet'))[-1]
cfg=default_config()

def run_ep(pack,ctrl,cfg):
    ctrl.reset(); steps=0; viol=0; T_max=0
    while steps<cfg['max_steps']:
        if np.mean([c.SOC for c in pack.cells])>=cfg['target_soc']: break
        m=pack.step(ctrl.get_currents(pack),dt=cfg['dt_s'])
        viol+=m['n_violations']; T_max=max(T_max,m['T_max']); steps+=1
    socs=[c.SOC for c in pack.cells]; Tf=[c.T_C for c in pack.cells]
    return {'time':steps*cfg['dt_s']/60,'sigma':float(np.std(socs))*100,
            'T_max':T_max,'dT':float(max(Tf)-min(Tf)),'viol':viol}

ctrl=GraphGuidedOptimizer(cfg,gnn)
eps=[]
for ep in range(30):
    np.random.seed(ep*7); torch.manual_seed(ep*7)
    pack=build_pack_from_ecm(ecm_parquet=ecm,n_cells=12,chemistry='LFP',soc_init=0.20,soc_noise=0.03,T_amb=25.0,seed=ep*7)
    eps.append(run_ep(pack,ctrl,cfg))
    if (ep+1)%10==0: print(f'ep {ep+1}/30',flush=True)

a=lambda k: float(np.mean([r[k] for r in eps]))
print(f"\nCURRENT CODE result (should match Table 4 if seed=protocol is truly identical):")
print(f"  time={a('time'):.2f}min  sigma={a('sigma'):.4f}%  dT={a('dT'):.4f}  T_max={a('T_max'):.2f}  viol={a('viol'):.2f}")
print(f"\nTable 4 reference: time=27.9min sigma=0.82% dT=0.13")
print(f"Table 14 (old) reference: time=28.9min sigma=0.81% dT=0.12")
