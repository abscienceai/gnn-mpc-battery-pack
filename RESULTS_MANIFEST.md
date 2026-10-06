# Final Results Manifest

This file maps the final manuscript evidence to repository files.

## Frozen computational-evidence commit

```text
a28143ea3bb60081214ae9aa91e9051b1c202e72
```

Historical evidence manifest:

```text
provenance/submission_evidence_freeze_20261004.json
```

## Core model inputs

| Role | File |
|---|---|
| MATR-informed ECM/SOH table | `results/ecm/ecm_params_20260630_202031.parquet` |
| Final action-conditioned PackGNN v2 checkpoint | `results/models/pack_gnn_action_v2_20261003_232518.pt` |

## Canonical LFP N=30

Generator:

```text
src/run_canonical_lfp_n30.py
```

Analysis:

```text
src/analyze_canonical_lfp_n30.py
```

Outputs:

```text
results/canonical_lfp_n30/canonical_lfp_n30_raw.jsonl
results/canonical_lfp_n30/canonical_lfp_n30_summary.csv
results/canonical_lfp_n30/canonical_lfp_n30_summary.json
results/canonical_lfp_n30/statistics/descriptive_statistics.csv
results/canonical_lfp_n30/statistics/paired_statistics.csv
results/canonical_lfp_n30/statistics/safety_statistics.csv
```

## Thermal Initial-Condition Diagnostic

Generator:

```text
src/run_thermal_initial_condition_control.py
```

Outputs:

```text
results/diagnostics/thermal_initial_condition_n30/thermal_initial_condition_raw.jsonl
results/diagnostics/thermal_initial_condition_n30/thermal_initial_condition_summary.json
results/diagnostics/thermal_initial_condition_n30/thermal_initial_condition_statistics.csv
provenance/thermal_initial_condition_n30.json
```

## HUST Physical-Time Replay

Generator:

```text
src/run_hust_timestamp_replay_v2.py
```

Analysis:

```text
src/analyze_hust_timestamp_replay_v2.py
```

Outputs:

```text
results/diagnostics/hust_timestamp_replay_v2_n30/hust_timestamp_replay_v2_raw.jsonl
results/diagnostics/hust_timestamp_replay_v2_n30/hust_timestamp_replay_v2_summary.json
results/diagnostics/hust_timestamp_replay_v2_n30/hust_timestamp_replay_v2_statistics.csv
results/diagnostics/hust_timestamp_replay_v2_n30/hust_timestamp_replay_v2_safety.csv
results/diagnostics/hust_timestamp_replay_v2_n30/hust_timestamp_replay_v2_provenance.json
```

Interpretation: held-out measured HUST protocol replay on the study MATR-informed simulated plant. It is not external plant validation.

## Controller Software Timing

Generator:

```text
src/run_controller_timing_validation.py
```

Outputs:

```text
results/diagnostics/controller_timing_validation/controller_timing_raw.csv
results/diagnostics/controller_timing_validation/controller_timing_summary.json
provenance/controller_timing_validation.json
```

Interpretation: software timing only, not HIL.

## Action-Conditioning Evidence

Frozen evidence:

```text
provenance/pack_gnn_action_v2_heldout_action_test.json
```

The original one-off 500-rollout generator was not retained. `src/smoke_action_gnn_v2.py` is a smaller independent smoke diagnostic and is not claimed to reproduce the frozen 500-rollout result.

## Candidate-Ranking Evidence

Frozen evidence:

```text
provenance/pack_gnn_action_v2_ranking_fidelity.json
```

The original one-off ranking generator was not retained.

## Figures

Manual publication schematic:

```text
figures/submission_v2/fig1_controller_architecture.png
```

Generated evidence figures:

```text
figures/submission_v2/fig2_canonical_paired_tradeoff.pdf
figures/submission_v2/fig2_canonical_paired_tradeoff.png
figures/submission_v2/fig3_thermal_initial_condition.pdf
figures/submission_v2/fig3_thermal_initial_condition.png
figures/submission_v2/fig4_hust_paired_tradeoff.pdf
figures/submission_v2/fig4_hust_paired_tradeoff.png
figures/submission_v2/fig5_surrogate_ranking_fidelity.pdf
figures/submission_v2/fig5_surrogate_ranking_fidelity.png
figures/submission_v2/figS1_action_conditioning.pdf
figures/submission_v2/figS1_action_conditioning.png
```

Figure metadata:

```text
figures/submission_v2/figure_manifest.json
figures/submission_v2/latex_figure_snippets.tex
```

## Excluded Development Evidence

The publication branch intentionally excludes earlier development-stage:

- NMC and LCO evaluations;
- cross-chemistry extrapolation experiments;
- pseudo-HIL claims;
- SimpleMPC development comparator;
- MLP and node-only surrogate ablations;
- older checkpoints;
- earlier manuscript figures;
- obsolete robustness and parameter-sweep outputs.

These remain recoverable from Git history and the pre-publication backup branch but are not part of the final manuscript evidence.
