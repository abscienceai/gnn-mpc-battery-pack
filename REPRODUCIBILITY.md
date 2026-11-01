# Reproducibility

## 1. Study scope

This repository supports the reported 12-cell LFP GraphOptimizer study. MATR informs the LFP ECM/SOH parameterisation used by the simulated plant. HUST is held out and used only as a source of measured current, time, and voltage protocols.

## 2. Environment

Reference environment:

```text
Python       3.10.12
NumPy        1.26.4
pandas       2.3.3
SciPy        1.15.3
PyTorch      2.4.1+cu121
matplotlib   3.10.9
pyarrow      25.0.1
```

Install with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Frozen model inputs

MATR-informed ECM table:

```text
results/ecm/ecm_params_20260630_202031.parquet
```

Primary PackGNN v2 checkpoint:

```text
results/models/pack_gnn_action_v2_20261003_232518.pt
```

Additional NodeOnly, FlatMLP, and PackGNN training-seed checkpoints are retained under `results/extended_evaluation/models/`.

## 4. Primary paired N=30 evaluation

Run:

```bash
python src/run_canonical_lfp_n30.py
python src/analyze_canonical_lfp_n30.py
```

Protocol:

- 30 deterministic evaluation seeds: 0, 7, 14, ..., 203;
- 12 LFP cells;
- initial mean SOC approximately 0.20;
- target mean SOC 0.80;
- 60 s control interval;
- 3C per-cell current limit;
- 2.5C pack-average current limit;
- voltage envelope 2.0 to 3.5 V;
- temperature envelope -10 to 45 °C;
- SOC envelope 0.05 to 0.98.

GraphOptimizer and Physics-CEM use the same `H=5`, `K=64`, 16 elites, 5 refinement iterations, actuator projection, 60/40 charging-pressure blend, and one-step model-based safety screening.

Outputs are stored in `results/canonical_lfp_n30/`. The folder name is historical; the manuscript calls this the primary paired evaluation.

## 5. Architecture and robustness controls

Run as needed:

```bash
python src/run_surrogate_ablation_n30.py
python src/run_nodeonly_blend_match.py
python src/run_training_seed_stability.py
python src/run_physics_shared_objective.py
python src/run_model_mismatch_robustness.py
python src/run_thermal_pareto_sweep.py
```

Reported outputs are under `results/extended_evaluation/`.

The strict architecture ablation compares action-conditioned PackGNN, NodeOnly, and parameter-matched FlatMLP under the same downstream controller chain. The 90/10 NodeOnly case is a separate Pareto sensitivity rather than a strict architecture ablation.

## 6. Thermal initial-condition diagnostic

```bash
python src/run_thermal_initial_condition_control.py
```

The numerical diagnostic remains part of the manuscript evidence even though its standalone plot is not included in the final main-text figure set.

Outputs:

```text
results/diagnostics/thermal_initial_condition_n30/
```

## 7. HUST measured-protocol replay

Obtain and preprocess HUST according to the upstream dataset/BatteryML terms, then place processed files under:

```text
data/processed/BatteryML/HUST/
```

Run:

```bash
python src/run_hust_timestamp_replay_v2.py
python src/analyze_hust_timestamp_replay_v2.py
```

HUST supplies measured current, physical timestamps, and voltage context. Simulated state responses remain those of the MATR-informed study plant. This is measured-protocol replay, not external plant validation.

## 8. Software timing

```bash
python src/run_controller_timing_validation.py
```

The timing result measures workstation software control-loop latency. It is not HIL or embedded real-time validation.

## 9. Frozen diagnostics

Action-conditioning evidence:

```text
provenance/pack_gnn_action_v2_heldout_action_test.json
```

Candidate-ranking evidence:

```text
provenance/pack_gnn_action_v2_ranking_fidelity.json
```

The original one-off generators for these two frozen diagnostics were not retained. `src/smoke_action_gnn_v2.py` provides a smaller independent action-sensitivity smoke test and is not presented as a reproduction of the 500-rollout frozen diagnostic.

## 10. Figure generation

The new multi-panel architecture/fairness/robustness figure is generated from frozen result files without rerunning controller experiments:

```bash
python scripts/figures/generate_extended_evaluation_figure.py
```

Outputs:

```text
figures/submission_v2/fig_extended_evaluation_robustness.pdf
figures/submission_v2/fig_extended_evaluation_robustness.png
```

The other retained manuscript figures are frozen presentation artifacts derived from the corresponding result/provenance files listed in `RESULTS_MANIFEST.md`.

## 11. Third-party data

Raw and processed MATR/HUST files are not redistributed. See `data/README.md`.

## 12. Interpretation boundaries

- Observed violations are empirical counts under the evaluated models and scenarios.
- One-step screening is not a formal recursive or robust-safety guarantee.
- The model-based ageing quantity is not measured capacity fade or lifetime.
- Independent per-cell currents require an appropriate active/reconfigurable hardware interface.
- Retraining can produce small floating-point differences across hardware and library versions.
