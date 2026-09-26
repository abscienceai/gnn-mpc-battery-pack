# Results manifest for the submission freeze

This repository contains several experiment families. Values from different families must not be mixed because some use different seed sets, CEM budgets, horizon lengths, stress conditions, or evaluation scripts.

## Canonical manuscript results

- **Primary LFP, Table 5 / main headline:** `results/canonical_LFP_actuatormatch_v2_postpatch/experiment_results_20260803_054210.json`
- **Zero-shot NMC:** `results/canonical_NMC_actuatormatch_v2_postpatch/experiment_results_20260803_054210.json`
- **Zero-shot LCO:** `results/canonical_LCO_actuatormatch_v2_postpatch/experiment_results_20260803_054210.json`
- **Paired statistics:** `results/paired_statistical_tests.csv`
- **Horizon-wise surrogate error:** `results/horizon_wise_mae.csv`
- **Rollout calibration sensitivity:** `results/rollout_sensitivity.csv`
- **Cost-component diagnostic:** `results/cost_component_breakdown.csv`

The canonical LFP headline is: CC-CV 14.7 min / 3.22% SOC imbalance / 0.18 °C inter-cell gradient / 0.03 violations per episode; GraphOptimizer 34.6 min / 0.39% / 0.03 °C / 0.0. This corresponds to 87.9% lower SOC imbalance and 82.0% lower inter-cell thermal gradient.

## Architecture and diagnostic experiments

`src/fair_ablation.py` is the canonical **retrained** architecture comparison (Full GNN vs independently trained NodeOnlyGNN vs independently trained MLP). It writes outputs to `results/fair_ablation/` when executed.

`results/cross_chem_ablation_n30.csv` is retained as an older diagnostic experiment and must not be substituted for the independently retrained fair-ablation results described in the manuscript.

Other CSV/JSON files are explicitly diagnostic/ablation outputs. Their numerical values may differ from the primary table because the corresponding scripts use different conditions documented in the manuscript and in each script.

## Reproducibility rule

When checking a manuscript claim, use the exact source listed for that table/analysis rather than selecting the numerically closest result from another experiment family.
