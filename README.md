# Action-Conditioned Graph-Guided MPC for Cell-Level Lithium-Ion Battery Charging

Reproducibility repository for the manuscript:

> **Action-Conditioned Graph-Guided Model Predictive Control for Cell-Level Charging of Lithium-Ion Battery Packs**  
> Alper Bingöl<sup>a,*</sup>, Mücahit Soylu<sup>b</sup> and Ali Baheri<sup>c</sup>  

<sup>a</sup> Department of Physics, Faculty of Arts and Sciences, İnönü University, Malatya, Turkey  
<sup>b</sup> Department of Software Engineering, Faculty of Engineering, İnönü University, Malatya, Turkey  
<sup>c</sup> Department of Mechanical Engineering, Safe AI Lab, Rochester Institute of Technology, Rochester, NY, USA

**Repository:** https://github.com/abscienceai/gnn-mpc-battery-pack

---

## Overview

This repository contains the source code, trained surrogate checkpoints, frozen evaluation outputs, statistical results, provenance records, and manuscript figures supporting a computational study of **cell-level charging control for heterogeneous lithium-ion battery packs**.

The proposed controller, **GraphOptimizer**, embeds an explicitly **candidate-action-conditioned graph neural network surrogate (PackGNN v2)** inside an online **cross-entropy method (CEM)** model-predictive control loop. PackGNN v2 is not a direct charging policy. Instead, it recursively predicts candidate-conditioned future cell states so that CEM can rank alternative per-cell current vectors over a finite horizon.

A matched **Physics-CEM** comparator uses direct equivalent-circuit/thermal model rollouts for the same candidate-search problem while sharing the same search budget and downstream action-processing chain. This design isolates the candidate-evaluation model as the principal methodological difference.

The study focuses on a **12-cell lithium iron phosphate (LFP) pack** with dataset-informed equivalent-circuit parameters and evaluates charging time, SOC dispersion, thermal uniformity, peak temperature, voltage behaviour, a model-based degradation-stress proxy, constraint events, architecture attribution, training sensitivity, model-plant mismatch, measured-protocol replay, and software timing.

---

## Scientific question

The central question is:

> Can an explicitly candidate-action-conditioned graph dynamics surrogate provide useful recursive multi-step candidate ranking inside online cell-level CEM, relative to matched direct-physics candidate evaluation under the same search budget and downstream controller chain?

The manuscript does **not** claim that graph learning is universally superior to direct physics, that the learned surrogate is computationally faster, or that the present simulation establishes hardware-ready safety.

---

## Controller architecture

Both GraphOptimizer and Physics-CEM use the same overall optimisation and action-processing structure:

| Component | Setting |
|---|---|
| Pack size | 12 LFP cells |
| Prediction horizon | `H = 5` |
| CEM population | `K = 64` candidate current vectors |
| CEM elites | 16 |
| CEM refinement iterations | 5 |
| Control interval | 60 s |
| Per-cell current limit | 3C |
| Pack-average current limit | 2.5C |
| Target mean SOC | 0.80 |
| Voltage envelope | 2.0 to 3.5 V |
| Temperature envelope | -10 to 45 °C |
| SOC envelope | 0.05 to 0.98 |
| Downstream blend | 60% CEM base action + 40% capacity-aware SOC-deficit action |
| Final feasibility layer | One-step direct ECM/thermal model-based screening |

### GraphOptimizer

GraphOptimizer uses **PackGNN v2**, an 84,868-parameter graph surrogate. Each cell node receives seven state features together with the **candidate cell current being evaluated**. The candidate action is supplied explicitly at every recursive prediction step.

PackGNN v2 predicts:

- next SOC;
- temperature change;
- the model-based ageing/degradation-stress proxy;
- an auxiliary pack-level imbalance quantity used for training diagnostics.

PackGNN v2 has **no terminal-voltage prediction head**. Final voltage feasibility is therefore checked by the common one-step direct ECM/thermal screening layer.

### Physics-CEM

Physics-CEM performs the same CEM search using recursive direct ECM/thermal rollouts. It uses the same horizon, candidate count, elite count, refinement count, actuator projection, 60/40 charging-pressure blend, and final one-step screening chain.

A secondary **Physics-CEM-SharedObjective** sensitivity removes horizon voltage information unavailable to PackGNN while retaining direct-physics propagation.

---

## Primary paired result

The primary evaluation uses **30 paired initial-pack seeds**. For each seed, GraphOptimizer and Physics-CEM begin from the same physical state and parameter realisation.

| Metric | GraphOptimizer | Physics-CEM | Paired interpretation |
|---|---:|---:|---|
| Charging time (min) | **21.633** | 25.033 | **13.58% shorter** |
| Final SOC dispersion (%) | 0.3663 | 0.4144 | Difference not statistically significant |
| Final inter-cell temperature spread (°C) | 0.1553 | **0.1160** | GraphOptimizer **33.96% higher** |
| Peak temperature (°C) | 26.319 | 26.257 | Small absolute increase |
| Model-based cumulative ageing proxy | **lower** | higher | GraphOptimizer **5.10% lower** |
| Target success | 30/30 | 30/30 | Both reached target |
| Observed primary constraint violations | 0 | 0 | Empirical result under evaluated model |

For charging time, the paired GraphOptimizer-minus-Physics-CEM difference is **-3.40 min**, with a 95% paired-bootstrap confidence interval of approximately **[-3.63, -3.13] min**.

The primary conclusion is therefore a **charging-time versus thermal-uniformity trade-off**, not dominance across all objectives.

---

## Architecture attribution

The manuscript includes matched learned-surrogate controls to test whether the observed behaviour is specific to graph message passing rather than merely to using a learned state predictor.

| Surrogate/controller | Charging time (min) | Final SOC dispersion (%) | Final temperature spread (°C) |
|---|---:|---:|---:|
| PackGNN | 21.633 | **0.3663** | 0.1553 |
| NodeOnly | **18.933** | 0.3705 | 0.1852 |
| FlatMLP | 19.700 | 7.5013 | 0.2455 |
| NodeOnly, 90/10 sensitivity | 22.367 | 1.3547 | **0.1302** |

Interpretation:

- NodeOnly charges faster under the strict 60/40 configuration but is less thermally uniform.
- The parameter-matched FlatMLP produces severe final SOC dispersion.
- Retuning NodeOnly toward thermal uniformity improves temperature spread but makes charging slower than PackGNN and substantially worsens SOC dispersion.
- The evidence therefore supports a **configuration-limited graph-specific Pareto contribution**, not universal graph superiority.

---

## Training-seed stability

Training-initialisation sensitivity is evaluated for seeds **42, 101, and 202** while holding data generation and rollout-level partitioning fixed.

Across the three seeds:

- the qualitative PackGNN/NodeOnly thermal-ordering pattern persists;
- FlatMLP retains severe SOC dispersion;
- PackGNN charging time is sensitive to training initialisation.

This experiment is intentionally interpreted as a robustness boundary rather than evidence of complete seed invariance.

---

## Shared-objective fairness sensitivity

Physics-CEM has access to terminal voltage during its direct-physics horizon rollout, while PackGNN v2 does not predict voltage. The **Physics-CEM-SharedObjective** sensitivity removes this information asymmetry from the optimised horizon cost.

Under this comparison:

- GraphOptimizer charging time remains **12.53% shorter**;
- GraphOptimizer retains lower final SOC dispersion;
- Physics-CEM-SharedObjective retains lower final temperature spread;
- both controllers reach the target in 30/30 episodes with zero observed violations.

This supports the conclusion that the primary charging-time advantage is not explained solely by the comparator's additional horizon voltage information.

---

## Model-plant mismatch stress test

A combined adverse mismatch keeps the controller-side model nominal while perturbing the simulated plant:

- `R0 × 1.20`;
- `Qnom × 0.90`;
- `hA × 0.80`.

Across 30 paired episodes:

| Metric | GraphOptimizer | Physics-CEM |
|---|---:|---:|
| Charging time (min) | **19.333** | 22.467 |
| Final SOC dispersion (%) | 0.3932 | 0.4346 |
| Final temperature spread (°C) | 0.2417 | **0.1553** |
| Observed plant violations | **0** | 3 |

GraphOptimizer remains approximately **13.95% shorter in charging time**, while again showing a larger thermal spread.

This experiment evaluates one bounded adverse mismatch direction. It does **not** establish formal robust safety.

---

## Action-conditioning and candidate-ranking diagnostics

Two frozen diagnostic analyses support the intended role of PackGNN v2.

### Action conditioning

The held-out action-conditioning diagnostic compares prediction errors when the correct candidate current is supplied with errors after the action is zeroed or shuffled.

The substantial degradation under action intervention supports that PackGNN v2 uses the explicit candidate-current input rather than predicting only from state.

Frozen evidence:

```text
provenance/pack_gnn_action_v2_heldout_action_test.json
```

### Candidate-ranking fidelity

Because PackGNN v2 is used to **rank candidate actions**, ranking fidelity is evaluated directly rather than relying only on one-step prediction error.

Frozen evidence:

```text
provenance/pack_gnn_action_v2_ranking_fidelity.json
```

The diagnostic shows generally informative candidate ordering but also weaker shared-objective ranking in parts of the higher-SOC operating region. This limitation is retained explicitly in the manuscript.

---

## Thermal initial-condition diagnostic

A controlled diagnostic repeats the GraphOptimizer-versus-Physics-CEM comparison with both heterogeneous initial cell temperatures and uniform `25 °C` initial temperatures.

The larger final temperature-spread tendency under GraphOptimizer persists under uniform initialisation, while the interaction between initial-temperature condition and controller difference is not statistically significant.

The numerical experiment remains part of the manuscript evidence even though its previous standalone plot is not retained in the final main-text figure set.

Outputs:

```text
results/diagnostics/thermal_initial_condition_n30/
```

---

## Held-out HUST measured-protocol replay

HUST is held out from the MATR-based parameterisation pipeline and is used as a source of:

- measured charging current;
- physical timestamps;
- measured voltage context.

The measured protocols are replayed against the MATR-informed simulated study plant.

This is therefore **measured-protocol replay**, not validation on a HUST-identified physical plant.

The replay preserves the main directional finding:

- GraphOptimizer reaches the target sooner than Physics-CEM;
- GraphOptimizer shows a larger final inter-cell temperature spread;
- final SOC-dispersion and energy differences are not significant in the reported paired comparison.

`HUST-Raw` is included as contextual external-protocol evidence and may exceed the study actuator envelope. `HUST-Projected` applies the common actuator projection and serves as the actuator-feasible protocol reference.

---

## Software control-loop timing

Software timing was measured on paired pack states across six nominal SOC levels.

| Controller | Median decision time | p95 decision time | 60 s deadline misses |
|---|---:|---:|---:|
| Physics-CEM | **0.602 s** | **0.618 s** | 0 |
| GraphOptimizer | 2.076 s | 2.110 s | 0 |

GraphOptimizer is therefore **computationally slower than Physics-CEM in the reported implementation**, although both remain comfortably below the 60 s software control interval.

These are workstation software timings, not hardware-in-the-loop or embedded-controller validation.

---

## Data sources

The study uses two public LFP datasets in different roles.

### MATR

MATR informs the LFP ECM/SOH parameterisation used for PackGNN training and primary simulated evaluation.

The repository includes the frozen resulting ECM table:

```text
results/ecm/ecm_params_20260630_202031.parquet
```

The processed MATR collection used in the study contained **180 batteries**.

The plant should be described as **MATR-informed** or **dataset-informed**, not fully data-derived. Processed data inform `R0` and SOH, while `R1`, `C1`, the base OCV representation, and base nominal capacity retain chemistry/model defaults.

### HUST

The processed HUST collection used for the study contained **77 LFP cells**.

Third-party raw and processed MATR/HUST datasets are intentionally **not redistributed** in this repository. Obtain them from their original providers under the applicable licences and terms.

The preprocessing workflow uses BatteryML:

https://github.com/microsoft/BatteryML

See:

```text
data/README.md
```

for dataset roles and expected local paths.

---

## Repository structure

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── RESULTS_MANIFEST.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
│
├── data/
│   └── README.md
│
├── figures/
│   ├── README.md
│   └── submission_v2/
│       ├── fig1_controller_architecture.png
│       ├── fig2_primary_paired_tradeoff.pdf
│       ├── fig2_primary_paired_tradeoff.png
│       ├── fig_extended_evaluation_robustness.pdf
│       ├── fig_extended_evaluation_robustness.png
│       ├── fig4_surrogate_ranking_fidelity.pdf
│       ├── fig4_surrogate_ranking_fidelity.png
│       ├── fig5_hust_paired_tradeoff.pdf
│       ├── fig5_hust_paired_tradeoff.png
│       ├── figS1_action_conditioning.pdf
│       └── figS1_action_conditioning.png
│
├── provenance/
│   ├── controller_timing_validation.json
│   ├── pack_gnn_action_v2_canonical.json
│   ├── pack_gnn_action_v2_heldout_action_test.json
│   ├── pack_gnn_action_v2_ranking_fidelity.json
│   ├── repository_snapshot_manifest.json
│   └── thermal_initial_condition_n30.json
│
├── results/
│   ├── canonical_lfp_n30/
│   │   ├── canonical_lfp_n30_raw.jsonl
│   │   ├── canonical_lfp_n30_summary.csv
│   │   ├── canonical_lfp_n30_summary.json
│   │   └── statistics/
│   │
│   ├── diagnostics/
│   │   ├── controller_timing_validation/
│   │   ├── hust_timestamp_replay_v2_n30/
│   │   └── thermal_initial_condition_n30/
│   │
│   ├── ecm/
│   │   └── ecm_params_20260630_202031.parquet
│   │
│   ├── extended_evaluation/
│   │   ├── blend_match/
│   │   ├── model_mismatch/
│   │   ├── models/
│   │   ├── physics_shared_objective/
│   │   ├── surrogate_ablation/
│   │   ├── thermal_pareto/
│   │   └── training_seed_stability/
│   │
│   └── models/
│       └── pack_gnn_action_v2_20261003_232518.pt
│
├── scripts/
│   └── figures/
│       └── generate_extended_evaluation_figure.py
│
└── src/
    ├── analyze_canonical_lfp_n30.py
    ├── analyze_hust_timestamp_replay_v2.py
    ├── ecm_fitting.py
    ├── evaluation_controllers.py
    ├── graph_battery_pack.py
    ├── run_canonical_lfp_n30.py
    ├── run_controller_timing_validation.py
    ├── run_hust_timestamp_replay_v2.py
    ├── run_model_mismatch_robustness.py
    ├── run_nodeonly_blend_match.py
    ├── run_physics_shared_objective.py
    ├── run_surrogate_ablation_n30.py
    ├── run_thermal_initial_condition_control.py
    ├── run_thermal_pareto_sweep.py
    ├── run_training_seed_stability.py
    ├── safe_fast_charge_optimizer.py
    ├── train_gnn.py
    ├── train_mlp_fair.py
    └── train_node_only_gnn.py
```

The internal folder name `canonical_lfp_n30` is retained for compatibility with the original analysis pipeline. In the manuscript, this experiment is referred to as the **primary paired LFP evaluation**.

---

## Environment

Reference computational environment:

```text
Python       3.10.12
NumPy        1.26.4
pandas       2.3.3
SciPy        1.15.3
PyTorch      2.4.1+cu121
matplotlib   3.10.9
pyarrow      25.0.1
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The publication source tree does **not** require PyTorch Geometric. Graph operations used by PackGNN are implemented directly in this repository.

---

## Frozen model checkpoints

Primary PackGNN v2 checkpoint:

```text
results/models/pack_gnn_action_v2_20261003_232518.pt
```

Additional NodeOnly, FlatMLP, and PackGNN training-seed checkpoints are stored under:

```text
results/extended_evaluation/models/
```

---

## Reproducing the main analyses

### 1. Train PackGNN v2

```bash
python src/train_gnn.py \
  --n_rollouts 5000 \
  --epochs 50 \
  --batch 128 \
  --lr 0.001 \
  --n_cells 12 \
  --chemistry LFP \
  --seed 42 \
  --device auto
```

The reported primary evaluation uses the frozen reference checkpoint provided in `results/models/`.

### 2. Primary paired LFP evaluation

```bash
python src/run_canonical_lfp_n30.py
python src/analyze_canonical_lfp_n30.py
```

Primary outputs:

```text
results/canonical_lfp_n30/canonical_lfp_n30_raw.jsonl
results/canonical_lfp_n30/canonical_lfp_n30_summary.csv
results/canonical_lfp_n30/canonical_lfp_n30_summary.json
results/canonical_lfp_n30/statistics/descriptive_statistics.csv
results/canonical_lfp_n30/statistics/paired_statistics.csv
results/canonical_lfp_n30/statistics/safety_statistics.csv
```

### 3. Architecture ablation

```bash
python src/run_surrogate_ablation_n30.py
```

### 4. Thermal-matched NodeOnly sensitivity

```bash
python src/run_nodeonly_blend_match.py
```

### 5. Training-seed stability

```bash
python src/run_training_seed_stability.py
```

### 6. Shared-objective comparator sensitivity

```bash
python src/run_physics_shared_objective.py
```

### 7. Model-plant mismatch

```bash
python src/run_model_mismatch_robustness.py
```

### 8. Thermal Pareto exploratory sweep

```bash
python src/run_thermal_pareto_sweep.py
```

### 9. Controlled thermal initial-condition diagnostic

```bash
python src/run_thermal_initial_condition_control.py
```

### 10. HUST measured-protocol replay

After preparing the required third-party HUST data as documented in `data/README.md`:

```bash
python src/run_hust_timestamp_replay_v2.py
python src/analyze_hust_timestamp_replay_v2.py
```

### 11. Software timing

```bash
python src/run_controller_timing_validation.py
```

---

## Regenerating the extended-evaluation figure

The manuscript's four-panel architecture/fairness/robustness synthesis figure is generated directly from frozen result files:

```bash
python scripts/figures/generate_extended_evaluation_figure.py
```

Outputs:

```text
figures/submission_v2/fig_extended_evaluation_robustness.pdf
figures/submission_v2/fig_extended_evaluation_robustness.png
```

The figure summarises:

1. architecture attribution;
2. training-seed stability;
3. charging-time robustness/fairness;
4. thermal-trade-off robustness.

The other retained manuscript figures are frozen presentation artifacts tied to the result/provenance files listed in `RESULTS_MANIFEST.md`.

---

## Reproducibility notes

### Rollout-group-safe split

PackGNN v2 was trained from **5,000 independent simulated rollout groups** comprising **29,513 state-action samples**.

- Training: 4,250 rollouts / 25,172 samples
- Validation: 750 rollouts / 4,341 samples
- Evaluation seeds: disjoint from both training and validation pack seeds

The split is performed at the rollout-group level rather than by randomly mixing individual time steps.

### Paired evaluation

The primary GraphOptimizer-versus-Physics-CEM comparison uses the deterministic evaluation seeds:

```text
0, 7, 14, ..., 203
```

Within each seed, the controllers receive identical initial pack states and parameter realisations.

### Statistical analysis

The primary paired analysis uses:

- paired episode-level differences;
- 20,000 paired bootstrap resamples for confidence intervals;
- Wilcoxon signed-rank tests;
- paired t-tests as secondary statistics;
- Holm correction where reported.

Frozen primary statistics are stored in:

```text
results/canonical_lfp_n30/statistics/
```

### Snapshot integrity

SHA256 hashes for the cleaned repository snapshot are stored in:

```text
provenance/repository_snapshot_manifest.json
```

---

## Important interpretation boundaries

This repository intentionally preserves the limits of the evidence.

### Safety

- Zero observed violations are **empirical outcomes**, not a formal safety certificate.
- Final action screening is **one-step and model-based**.
- Recursive feasibility and worst-case robust safety are not established.

### Hardware

- Independent effective per-cell current commands require an appropriate **active, modular, or reconfigurable hardware layer**.
- The study does not validate converter-level efficiency, switching losses, topology-specific feasibility, electromagnetic effects, thermal loading of power electronics, or protection coordination.

### Plant fidelity

- The simulated plant is **MATR-informed**, not fully identified from experimental cell data.
- State-estimator uncertainty is not included in the reported closed-loop evaluation.

### Degradation

- The ageing quantity is a **relative model-based degradation-stress proxy**.
- It is not experimentally calibrated capacity fade, SOH loss, or lifetime extension.

### HUST replay

- HUST supplies measured protocol information.
- The simulated state response remains the MATR-informed study plant.
- The replay is not physical validation on HUST cells.

### Generalisation

- The reported study is limited to the evaluated 12-cell LFP setting, topology, surrogate families, and controller configuration.
- Cross-chemistry and broader pack-size generalisation are not established.

### Computation

- GraphOptimizer is computationally slower than Physics-CEM in the reported implementation.
- Both meet the 60 s software interval on the reported workstation.
- This is not HIL or embedded real-time validation.

---

## Files to read first

For a quick audit of the repository:

1. `README.md` - study overview and reproduction commands
2. `REPRODUCIBILITY.md` - detailed reproduction protocol
3. `RESULTS_MANIFEST.md` - mapping from manuscript evidence to repository files
4. `data/README.md` - third-party dataset roles and expected paths
5. `provenance/repository_snapshot_manifest.json` - snapshot file hashes

---

## Code availability statement

A concise manuscript-compatible statement is:

> Source code, trained model checkpoints, evaluation outputs, and reproducibility materials are publicly available at https://github.com/abscienceai/gnn-mpc-battery-pack. Third-party MATR and HUST datasets are not redistributed; instructions for obtaining and preprocessing them are provided in the repository.

---

## Citation

If you use this repository, please cite the accompanying manuscript:

**Alper Bingöl, Mücahit Soylu, and Ali Baheri.**  
*Action-Conditioned Graph-Guided Model Predictive Control for Cell-Level Charging of Lithium-Ion Battery Packs.*

Machine-readable citation metadata are provided in:

```text
CITATION.cff
```

Journal, volume, pages, and DOI can be added after publication.

---

## License

The source code in this repository is released under the **MIT License**.

Third-party datasets remain subject to their original providers' licences and terms.

---
