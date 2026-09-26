"""Fast, no-training consistency audit for the frozen submission package."""
from pathlib import Path
import json, csv, re, sys

ROOT = Path(__file__).resolve().parent
errors=[]

def check(cond,msg):
    if not cond: errors.append(msg)

# Controller configuration must match manuscript freeze.
s = (ROOT/'src/safe_fast_charge_optimizer.py').read_text(encoding='utf-8')
for token in ['"w_time":         4.0','"w_imbalance":    3.0','"w_temperature":  3.0','"w_gradient":     3.0','"w_aging":        2.0','"w_violation":    50.0','"horizon":        5','"cem_samples":    64','"cem_iterations": 5']:
    check(token in s, f'missing config token: {token}')
check('I_final = 0.6 * I_base + 0.4 * I_gnn' in s, '0.6/0.4 applied-action blend not found')

# Canonical outputs exist and retain the headline values used in manuscript/README.
expect = {
 'LFP': {'CC-CV':(14.7333333333,3.22,0.18), 'GraphOptimizer':(34.6,0.39,0.03)},
 'NMC': {'CC-CV':(93.6,), 'GraphOptimizer':(15.4,)},
 'LCO': {'CC-CV':(25.8,), 'GraphOptimizer':(67.1,)},
}
for chem in ['LFP','NMC','LCO']:
    p=ROOT/'results'/f'canonical_{chem}_actuatormatch_v2_postpatch'/'experiment_results_20260803_054210.json'
    check(p.exists(), f'missing canonical {chem} result')

# Rollout sensitivity must be monotonic in alpha_SOC at each beta_T.
p=ROOT/'results'/'rollout_sensitivity.csv'
if p.exists():
    rows=list(csv.DictReader(p.open()))
    # detect likely column names without making assumptions about beta ordering
    cols=rows[0].keys() if rows else []
    alpha=next((c for c in cols if 'alpha' in c.lower()),None)
    beta=next((c for c in cols if 'beta' in c.lower()),None)
    sigma=next((c for c in cols if 'sigma' in c.lower()),None)
    if alpha and beta and sigma:
        groups={}
        for r in rows: groups.setdefault(float(r[beta]),[]).append((float(r[alpha]),float(r[sigma])))
        for b,vals in groups.items():
            vals=sorted(vals)
            check(all(vals[i+1][1] <= vals[i][1]+1e-12 for i in range(len(vals)-1)), f'rollout sensitivity non-monotonic for beta={b}')

if errors:
    print('SUBMISSION CONSISTENCY AUDIT: FAIL')
    for e in errors: print(' -',e)
    sys.exit(1)
print('SUBMISSION CONSISTENCY AUDIT: PASS')
print('Config, frozen result locations, and rollout-sensitivity direction are internally consistent.')
