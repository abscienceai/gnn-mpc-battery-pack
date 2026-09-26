# Graph-Guided MPC for Safety-Aware Cell-Level Charging of Lithium-Ion Battery Packs

Reference implementation for **"Graph-Guided Model Predictive Control for Safety-Aware Cell-Level Charging of Lithium-Ion Battery Packs."**

**Authors:** Alper Bingöl¹, Mücahit Soylu², Ali Baheri³  
¹ Department of Physics, Faculty of Arts and Sciences, İnönü University, Malatya, Turkey · ² Department of Software Engineering, Faculty of Engineering, İnönü University, Malatya, Turkey · ³ Department of Mechanical Engineering, Safe AI Lab, Rochester Institute of Technology, NY, USA

**Status:** Manuscript prepared for submission. See `CITATION.cff` for citation metadata. This will be updated with journal, volume, and DOI upon acceptance.

---

## Overview

This repository implements **GraphOptimizer**, a graph-guided safety-aware current-allocation framework for heterogeneous lithium-ion battery packs. Cells are modeled as nodes of a dynamic graph; thermal-adjacency edges encode inter-cell temperature, SOC-difference, and distance features. A Graph Neural Network (**PackGNN**), trained on 5,000 simulated rollouts using ECM parameters informed by 419,657 real charge/discharge cycles, serves as a fast multi-step surrogate inside a Model Predictive Control loop solved by the Cross-Entropy Method (CEM).

This is a **real-data-informed simulation study**, not a physical pack-level validation. All results below are computational; see the manuscript's Limitations section for the physical-validation roadmap.

## Key Results

**Primary evaluation (12-cell LFP pack, N=30 episodes, `src/safe_fast_charge_optimizer.py`):**

| Controller | Time (min) | σ_SOC (%) | ΔT (°C) | Violations/ep |
|---|---|---|---|---|
| CC-CV | 14.7 | 3.22 | 0.18 | 0.03 |
| CC-CV-Balance | 14.7 | 1.95 | 0.22 | 0.0 |
| Proportional | 14.7 | 0.10 | 0.37 | 0.0 |
| SimpleMPC (physics-only MPC) | 30.5 | 4.07 | 0.38 | 0.0 |
| **GraphOptimizer** | 34.6 | **0.39** | **0.03** | **0.0** |

All controllers are evaluated under a common per-cell and pack-level actuator-current constraint. On the trained chemistry (LFP), GraphOptimizer reduces SOC imbalance by 87.9% and inter-cell thermal gradient by 82.0% relative to CC-CV, at the cost of longer charging time. This reflects a deliberate safety-first operating point rather than a throughput-maximizing one.

## Important: Cross-Chemistry Safety Is Chemistry-Dependent

Zero-shot transfer of the LFP-trained controller to NMC and LCO packs (no retraining) gives a genuinely mixed result, and we report it as such:

| Chemistry | CC-CV violations/ep | GraphOptimizer violations/ep | Outcome |
|---|---|---|---|
| NMC | 93.6 | **15.4** | Large safety improvement |
| LCO | 25.8 | **67.1** | **No improvement**; exceeds CC-CV and CC-CV-Balance (24.5) |

A companion surrogate ablation (`src/cross_chem_ablation.py`, `src/fair_ablation.py`) shows the same asymmetry. A **genuinely retrained** No-edge GNN (no graph message passing) outperforms the full graph-based GraphOptimizer on **both** SOC balance and violation count, specifically on LCO, while the graph-based model is clearly better on LFP and NMC. We do not currently have a mechanistic explanation for why the relational-transfer benefit fails on this specific chemistry. This is flagged as an open problem in the manuscript's Limitations section rather than being smoothed over.

**Practical takeaway:** the released checkpoint is a strong candidate controller for LFP, its trained chemistry, and shows a real, substantial safety benefit under zero-shot transfer to NMC. It should **not** be assumed safe for LCO packs without chemistry-specific validation or retraining.

## Repository Structure

```text
src/
  graph_battery_pack.py           Dynamic graph pack simulator, ECM/thermal/aging models, PackGNN
  safe_fast_charge_optimizer.py   GraphGuidedOptimizer (MPC-CEM), all baseline controllers, run_experiment()
  train_gnn.py                    PackGNN training (5,000 rollouts)
  train_mlp_fair.py               Flat-MLP surrogate, trained from scratch for fair comparison
  train_node_only_gnn.py          Edge-disabled GNN, trained from scratch for fair comparison

  # Cross-chemistry
  cross_chem_ablation.py          Full GNN vs. genuinely-trained No-edge GNN vs. MLP, on NMC/LCO
  fair_ablation.py                Canonical source for Table 11 (cross-chem surrogate ablation)

  # Ablations
  ablation_study.py                Component ablation (CEM, Pure Greedy, No-edge-at-inference)
  no_deltaT_ablation.py            Effect of removing the thermal-gradient cost term
  cost_component_breakdown.py      Per-term contribution to the stage cost
  pure_greedy_stress.py            GraphOptimizer vs. Pure Greedy across 7 stress scenarios
  horizon_adversarial.py           H=1 vs. H=5 across initial-SOC-heterogeneity levels
  horizon_wise_mae.py              Surrogate rollout prediction error by horizon step
  simplempc_matched.py             SimpleMPC matched-budget comparison
  mlp_mpc_baseline.py              MLP surrogate vs. GraphOptimizer, primary LFP setting

  # Robustness / real-data validation
  bigru_error_injection.py         Closed-loop robustness to SOC-estimation error models
  hardware_aware_validation.py     Pseudo-HIL: ADC noise, latency, slew-rate constraints
  offline_replay_validation.py     MATR closed-loop replay, strict train/test separation, parameter mismatch
  independent_checkpoint_test.py   Frozen-checkpoint SOC/ΔT generalisation test on independently-resampled cells
  lyapunov_empirical_check.py      Empirical verification of Theorem 1's sufficient conditions
  regenerate_extended_baselines.py MSCC / Thermal-Aware Proportional baselines
  mscc_thermal_baselines.py

  # Sensitivity sweeps
  beta_dT_sensitivity.py, weight_sweep.py, rollout_sensitivity.py

  # Statistics and figures
  paired_statistical_tests.py      Paired t-test / Wilcoxon, GraphOptimizer vs. CC-CV
  generate_figures.py              All 12 data-driven manuscript figures
  generate_fig_attention.py        GNN edge-importance interpretability figure

results/
  models/                          Trained checkpoints (PackGNN, No-edge GNN, fair MLP)
  ecm/                             Extracted ECM parameters (parquet)
  canonical_{LFP,NMC,LCO}_actuatormatch_v2_postpatch/   Primary N=30 evaluation results; all controllers share a common actuator constraint
  packsize_{6,12,24}_actuatormatch_postpatch/            Scalability ablation (N=20), same common actuator constraint
  failure_sigma0{15,25}_actuatormatch_postpatch/         Extreme initial-imbalance stress tests (N=20), same common actuator constraint
  mismatch_hAfix_postpatch/                              Parameter-mismatch robustness (Table 8), corrected $h_A$ perturbation
  hardware_validation_obsfix_postpatch/                  Pseudo-HIL results (Table 18), corrected to feed the controller its observed (noisy/delayed) state rather than the true state
  independent_checkpoint_test/                           Frozen-checkpoint generalisation test on independently-resampled cells
  replay_validation/                                     MATR closed-loop replay results (Table 17)
  ablation_components/                                   Component ablation results
  *.csv, *.json                                          Remaining experiment outputs (one per script above)

data/           See data/README.md for dataset download instructions
figures/        14 manuscript figures (PDF)
```

The manuscript LaTeX source is not included in this repository. See `CITATION.cff` for how to cite this work in the interim.

## Requirements

```bash
python >= 3.10
torch >= 2.4.0, < 2.5.0
```

Compatible version ranges are pinned in `requirements.txt`. Install into a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

ECM parameters are extracted from four of the six public battery-cycling datasets listed below (419,657 of 427,857+ total cycles). The remaining two, NASA Randomized and Oxford Deg1, are used only for a generalisability check and a qualitative long-term degradation consistency check, respectively. See the table below. See `data/README.md` for download instructions via [BatteryML](https://github.com/microsoft/BatteryML) and the expected directory layout.

| Dataset | Chemistry | Cells | Cycles | Used for |
|---|---|---|---|---|
| MATR | LFP | 180 | 154,231 | ECM extraction, primary training chemistry |
| CALCE | LCO | 13 | 14,298 | ECM extraction, cross-chemistry evaluation |
| RWTH | NMC | 48 | 105,006 | ECM extraction, cross-chemistry evaluation |
| HUST | LFP | 77 | 146,122 | ECM extraction |
| NASA (Randomized) | Li-ion | 59 | Not applicable | Generalizability check |
| Oxford Deg1 | NMC | 8 | 8,200 | Independent SOH validation only |

## Submission-Freeze Navigation

- `RESULTS_MANIFEST.md` maps the manuscript headline and diagnostic analyses to their source result files.
- `REPRODUCIBILITY.md` records the seed/environment protocol and exact rerun commands.
- `audit_submission_consistency.py` performs a fast no-training consistency check.

## Reproducing the Results

```bash
# 1. Train the PackGNN (uses ECM parameters in results/ecm/)
python3 src/train_gnn.py

# 2. Primary LFP evaluation (Table 5)
python3 src/safe_fast_charge_optimizer.py --chemistry LFP --n_episodes 30

# 3. Cross-chemistry evaluation (Table 10)
python3 src/safe_fast_charge_optimizer.py --chemistry NMC --n_episodes 30
python3 src/safe_fast_charge_optimizer.py --chemistry LCO --n_episodes 30

# 4. Cross-chemistry surrogate ablation (Table 11)
python3 src/fair_ablation.py

# 5. Regenerate all figures
cd src && python3 generate_figures.py && python3 generate_fig_attention.py
```

All pack initializations use episode-specific seeds (`seed = 7 × episode_index`) for exact reproducibility given identical code, package versions, and hardware. See Appendix C of the manuscript for full reproducibility details and the hyperparameter table.

## Trained Checkpoints

| Checkpoint | Role |
|---|---|
| `results/models/pack_gnn_20260630_220402.pt` | PackGNN (84,804 params), primary surrogate used throughout |
| `results/models/node_only_gnn_20260714_040356.pt` | No-edge GNN, trained from scratch (85,572 params), Table 11/13 fair ablation |
| `results/models/pack_mlp_fair5k_20260718_185127.pt` | Flat MLP surrogate, trained from scratch (126,245 params), Table 11/13 fair ablation |

## Citation

Citation metadata is maintained in `CITATION.cff`. The manuscript is currently prepared for submission. Please check back for the final published reference, or cite this repository directly in the interim.

## License

MIT License. See `LICENSE`.

## Contact

Alper Bingöl  
alper1ton@gmail.com