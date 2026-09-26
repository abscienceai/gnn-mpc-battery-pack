# Reproducibility notes

## Environment

The submission freeze is designed for Python 3.10+ and the compatible package ranges in `requirements.txt`. The manuscript run environment used an NVIDIA L40S GPU, Ubuntu 24.04, CUDA 12.1 and PyTorch 2.4.1.

## Deterministic episode protocol

Primary controller evaluations use `seed = 7 * episode_index`. Both NumPy and PyTorch random states are initialised once per episode. CEM uses NumPy's episode-seeded random stream; there is no separate CEM-specific seed.

## Primary results

Run from repository root:

```bash
python3 src/safe_fast_charge_optimizer.py --chemistry LFP --n_episodes 30
python3 src/safe_fast_charge_optimizer.py --chemistry NMC --n_episodes 30
python3 src/safe_fast_charge_optimizer.py --chemistry LCO --n_episodes 30
```

The frozen canonical outputs used by the manuscript are retained under `results/canonical_*_actuatormatch_v2_postpatch/`.

## Fair architecture ablation

```bash
python3 src/fair_ablation.py
```

This compares independently trained Full-GNN, No-edge-GNN and MLP surrogates under the same H=1 protocol and writes both episode-level and summary outputs to `results/fair_ablation/`.

## Important protocol distinction

Some ablation and diagnostic scripts intentionally use different horizons, CEM sample budgets, stress cases or seed protocols. Those results are not interchangeable with the primary LFP/NMC/LCO values. See `RESULTS_MANIFEST.md` and the corresponding manuscript notes before comparing values across files.
