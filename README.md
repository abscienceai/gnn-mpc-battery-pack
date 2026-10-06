# Graph-Guided MPC for Safety-Aware Cell-Level Charging of Lithium-Ion Battery Packs

Repository for the manuscript:

**Graph-Guided Model Predictive Control for Safety-Aware Cell-Level Charging of Lithium-Ion Battery Packs**

Alper Bingöl, Mücahit Soylu, and Ali Baheri.

## Overview

This repository contains the final publication code, frozen model checkpoint, evaluation outputs, diagnostics, provenance records, and figures for a graph-guided model predictive charging framework.

The primary study considers a 12-cell LFP pack. GraphOptimizer uses an explicitly action-conditioned PackGNN v2 surrogate inside CEM-based MPC. Its matched Physics-CEM comparator uses direct ECM/thermal rollouts with the same search budget and downstream action-processing chain.

Both optimization controllers use:

- horizon H = 5;
- 64 CEM candidates per iteration;
- elite size 16;
- 5 CEM iterations;
- the same 60/40 base-action to SOC-deficit blend;
- the same actuator projection;
- the same one-step model-based safety verification.

The study assumes independent per-cell charging-current commands. Practical deployment therefore requires suitable cell-level power electronics, bypass converters, or another architecture capable of realizing those commands.

## Final Canonical LFP Evaluation

Thirty paired 12-cell LFP episodes were evaluated using the same initial pack realization for every controller within a seed.

| Controller | Charge time (min) | Final SOC sigma (%) | Peak T (C) | Final thermal gradient (C) | Observed violations |
|---|---:|---:|---:|---:|---:|
| CC-CV | 15.067 | 3.2209 | 26.641 | 0.1721 | 1 total |
| CC-CV-Balance | 15.000 | 1.7509 | 26.670 | 0.2153 | 0 |
| Proportional | 15.000 | 0.1182 | 26.742 | 0.3373 | 335 total |
| Physics-CEM | 25.033 | 0.4144 | 26.257 | 0.1160 | 0 |
| GraphOptimizer | 21.633 | 0.3663 | 26.319 | 0.1553 | 0 |

For the paired GraphOptimizer versus Physics-CEM comparison:

- GraphOptimizer charged 3.40 min faster on average, approximately 13.6%;
- both controllers had zero observed safety violations in the 30 canonical episodes;
- the SOC-imbalance difference was not statistically significant;
- GraphOptimizer had a larger final thermal gradient by approximately 0.039 C;
- the ageing proxy was approximately 5.1% lower for GraphOptimizer;
- the energy difference was not statistically significant.

These results do not establish universal superiority or Pareto dominance.

## HUST Measured-Protocol Replay

HUST supplies held-out measured charging-current, time, and voltage trajectories. The trajectories are resampled in physical time at 60 s intervals.

The simulated plant remains the study's MATR-informed LFP ECM/thermal plant. Therefore this experiment is an **external measured-protocol replay**, not external plant validation.

Against Physics-CEM in the paired HUST replay:

- GraphOptimizer charged approximately 13.8% faster;
- SOC-imbalance difference was not statistically significant;
- GraphOptimizer had a larger final thermal gradient;
- the ageing proxy was approximately 5.2% lower;
- both controller arms had zero observed study-constraint violations.

`HUST-Raw` is an unconstrained external-profile reference and is not actuator-fair. `HUST-Projected` applies the study's common actuator projection.

## Additional Diagnostics

### Action conditioning

The frozen held-out action intervention contains 500 independent rollouts and 2,901 samples. Zeroing or shuffling the candidate-action input substantially worsens SOC and temperature-change prediction, providing direct evidence that PackGNN v2 uses its action input.

The original one-off generator for this 500-rollout diagnostic was not retained. The frozen diagnostic JSON is included under `provenance/`. `src/smoke_action_gnn_v2.py` provides a smaller independent action-sensitivity smoke test, but it is not presented as a reproduction of the 500-rollout experiment.

### Candidate-ranking fidelity

The frozen ranking diagnostic evaluates 30 independent states with 64 candidate actions per state and horizon H = 5.

Mean Spearman correlation was approximately:

- 0.585 against the shared surrogate-compatible objective;
- 0.833 against the complete downstream physics objective.

Ranking was informative overall but was not uniformly reliable across the entire SOC range. PackGNN v2 has no voltage-prediction head.

The original one-off ranking-diagnostic generator was not retained. The frozen diagnostic JSON is included under `provenance/`.

### Thermal initial-condition control

The Graph-versus-Physics thermal-gradient difference persisted when all cells were initialized at 25 C. The temperature-condition interaction was not statistically significant. The observed thermal difference therefore cannot be attributed solely to heterogeneous initial temperature.

### Software controller timing

Timing measurements are software-only and are not HIL validation.

On the reported Xeon Gold 6430 plus NVIDIA L40S platform:

- Physics-CEM median decision latency: approximately 0.602 s;
- GraphOptimizer median decision latency: approximately 2.076 s;
- both had zero 60 s deadline misses.

GraphOptimizer was approximately 3.45 times slower computationally than Physics-CEM despite producing shorter charging times in the evaluation episodes.

## Safety Claim Boundary

The final canonical LFP limits are:

- SOC: 0.05 to 0.98;
- temperature: -10 to 45 C;
- voltage: 2.0 to 3.5 V.

The final safety filter performs one-step model-based verification after action blending and actuator projection. Zero observed violations in the reported Physics-CEM and GraphOptimizer experiments are empirical results. They are not a formal recursive-feasibility or closed-loop safety guarantee.

## Data Scope

The final manuscript scope uses:

- **MATR**: informs the LFP ECM/SOH parameterization and simulated evaluation plant;
- **HUST**: supplies held-out measured current/time/voltage charging protocols.

CALCE, RWTH, NASA, Oxford, NMC, and LCO experiments from earlier development stages are not part of the final publication evaluation and have been removed from this publication branch.

See `data/README.md`.

## Repository Structure

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── RESULTS_MANIFEST.md
├── requirements.txt
├── audit_submission_consistency.py
├── data/
│   └── README.md
├── figures/
│   └── submission_v2/
├── provenance/
├── results/
│   ├── canonical_lfp_n30/
│   ├── diagnostics/
│   ├── ecm/
│   └── models/
└── src/
```

## Environment

The final server environment used for the publication snapshot reports:

- Python 3.10.12
- NumPy 1.26.4
- pandas 2.3.3
- SciPy 1.15.3
- PyTorch 2.4.1 + CUDA 12.1 build
- matplotlib 3.10.9
- pyarrow 25.0.1

The final code does not require PyTorch Geometric. The graph operations used by PackGNN are implemented within the repository.

Create an environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Main Reproduction Commands

Run commands from the repository root.

### PackGNN v2 training

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

The publication repository already includes the frozen final checkpoint:

```text
results/models/pack_gnn_action_v2_20261003_232518.pt
```

### Canonical paired LFP N=30 evaluation

```bash
python src/run_canonical_lfp_n30.py
python src/analyze_canonical_lfp_n30.py
```

### Thermal initial-condition diagnostic

```bash
python src/run_thermal_initial_condition_control.py
```

### HUST physical-time protocol replay

After preparing the HUST BatteryML files as described in `data/README.md`:

```bash
python src/run_hust_timestamp_replay_v2.py
python src/analyze_hust_timestamp_replay_v2.py
```

### Software controller timing

```bash
python src/run_controller_timing_validation.py
```

### Publication figures

Figure 1 is the manually prepared publication schematic and is intentionally preserved by the figure scripts.

Figures 2-5 and Supplementary Figure S1 are generated from the frozen evidence:

```bash
python src/generate_submission_figures_v2.py
python src/refine_submission_figures_v2.py
```

### Final repository audit

```bash
python audit_submission_consistency.py
git diff --check
```

See `REPRODUCIBILITY.md` for protocol details and `RESULTS_MANIFEST.md` for the mapping between scripts, evidence files, and manuscript results.

## Frozen Computational Evidence

The original computational-evidence freeze was made at commit:

```text
a28143ea3bb60081214ae9aa91e9051b1c202e72
```

The historical manifest is:

```text
provenance/submission_evidence_freeze_20261004.json
```

The publication branch contains later packaging-only changes such as repository cleanup, portable provenance paths, documentation, and final figure packaging. Therefore source-file hashes in the historical freeze manifest refer to the frozen evidence commit, not necessarily to the later publication-packaging commit.

## Citation

See `CITATION.cff`.

## License

MIT License.
