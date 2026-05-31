# Safety-First Charging Optimisation: Thermal Gradient and SOC Balance at the Cost of Speed

**Dynamic Graph-Based Safe Charging Optimisation for Lithium-Ion Battery Packs**

Alper Bingöl¹ · Mücahit Soylu² · Ali Baheri³

¹ Inonu University, Faculty of Arts and Sciences, Department of Physics, Malatya, Turkey  
² Inonu University, Faculty of Engineering, Department of Software Engineering, Malatya, Turkey  
³ Rochester Institute of Technology, Department of Mechanical Engineering, Safe AI Lab, Rochester, NY, USA

---

## Overview

This repository contains the source code and pre-trained models for the paper:

> **Safety-First Charging Optimisation: Thermal Gradient and SOC Balance at the Cost of Speed**  
> *Joint Minimisation of Thermal Gradient and SOC Imbalance in Lithium-Ion Battery Packs*

We model a lithium-ion battery pack as a time-varying graph - cells as nodes, thermal and electrical couplings as edges - and train a Graph Neural Network (GNN) as a fast surrogate for multi-step state prediction within a Model Predictive Control (MPC) + Cross-Entropy Method (CEM) optimisation loop.

**Key results on a 12-cell LFP pack (N=30 episodes, episode-specific random seeds):**

| Controller | Time (min) | σ_SOC (%) | T_max (°C) | ΔT (°C) | Aging (×10⁻⁴) | Violations |
|---|---|---|---|---|---|---|
| CC-CV | 10.7 ± 0.5 | 8.76 ± 0.3 | 27.9 ± 0.3 | 1.02 ± 0.1 | 8.0 ± 0.0 | 0.1 |
| CC-CV-Balance | 11.0 ± 0.0 | 6.70 ± 0.0 | 27.6 ± 0.1 | 0.91 ± 0.1 | 8.0 ± 0.0 | 0.0 |
| Proportional | 13.0 ± 0.4 | 0.53 ± 0.1 | 27.0 ± 0.2 | 1.04 ± 0.1 | 8.0 ± 0.0 | 0.0 |
| SimpleMPC | 30.4 ± 1.1 | 1.92 ± 0.3 | 26.1 ± 0.2 | 0.64 ± 0.1 | 10.0 ± 0.0 | 0.0 |
| **GraphOptimizer (ours)** | **26.1 ± 0.6** | **0.54 ± 0.1** | **25.6 ± 0.1** | **0.21 ± 0.1** | **8.0 ± 0.0** | **0.0** |

GraphOptimizer reduces thermal gradient by **79.4%** and SOC imbalance by **93.8%** relative to CC-CV,  
with **5× lower thermal gradient** than the best SOC-focused baseline (Proportional: 1.04°C vs. 0.21°C).

---

## Repository Structure

```
├── src/
│   ├── graph_battery_pack.py          # BatteryPackGraph, ECM1RC, ThermalModel,
│   │                                  #   AgingCostModel, PackGNN
│   ├── safe_fast_charge_optimizer.py  # GraphGuidedOptimizer (MPC+CEM+GNN),
│   │                                  #   CC-CV, CC-CV-Balance, Proportional,
│   │                                  #   SimpleMPC baselines
│   ├── train_gnn.py                   # PackGNN training on simulation rollouts
│   ├── ablation_study.py              # Component ablation study (Table 7)
│   ├── cost_sensitivity.py            # Cost weight (λ) sensitivity analysis
│   ├── offline_replay_validation.py   # Closed-loop replay on real MATR
│   │                                  #   trajectories (Table 9) + cross-chemistry
│   │                                  #   validation (Table 6) + parameter mismatch
│   │                                  #   robustness (Table 4)
│   └── hardware_aware_validation.py   # Pseudo-HIL validation with ADC noise,
│                                      #   slew-rate limiting, latency (Table 10)
├── results/
│   └── models/
│       └── pack_gnn_20260507_031905.pt  # Pre-trained PackGNN checkpoint
├── README.md
└── requirements.txt
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

**Hardware used in experiments:** NVIDIA L40S GPU (48 GB VRAM), Intel Xeon CPU,  
256 GB RAM, Ubuntu 22.04, CUDA 13.0, PyTorch 2.11.0.

---

## Quickstart

### 1. Reproduce main results (Table 2)

```bash
cd src
python safe_fast_charge_optimizer.py \
    --n_episodes 30 \
    --n_cells 12 \
    --chemistry LFP \
    --output_dir ../results/optimizer_final
```

The script auto-loads the pre-trained GNN from `results/models/pack_gnn_*.pt`.  
Results are saved to `results/optimizer_final/experiment_results_*.json`.

> **Note on reproducibility:** Pack initialisation uses episode-specific random seeds
> (`seed = 7 × episode_index`). GNN training uses a fixed seed of 42.
> All results in the paper are reproducible with these settings.

### 2. Train PackGNN from scratch

ECM parameters must first be extracted from the datasets (requires dataset access - see [Datasets](#datasets) below):

```bash
# Step 1: extract ECM parameters from raw PKL files
python ecm_fitting.py --datasets all --output_dir ../results/ecm

# Step 2: train the PackGNN surrogate
python train_gnn.py \
    --n_rollouts 5000 \
    --epochs 50 \
    --chemistry LFP \
    --output_dir ../results/models
```

### 3. Component ablation study (Table 7)

```bash
python ablation_study.py \
    --n_episodes 30 \
    --n_cells 12 \
    --output_dir ../results/ablation_components
```

### 4. Cost weight sensitivity analysis

```bash
python cost_sensitivity.py \
    --n_episodes 10 \
    --output_dir ../results/sensitivity
```

### 5. Cross-chemistry & parameter mismatch validation (Tables 4, 6)

```bash
python offline_replay_validation.py \
    --n_episodes 20 \
    --output_dir ../results/replay_validation
```

### 6. Pseudo-HIL validation (Table 10)

```bash
python hardware_aware_validation.py \
    --n_episodes 20 \
    --output_dir ../results/hardware_validation
```

---

## Pre-trained Model

The pre-trained PackGNN checkpoint is provided at `results/models/pack_gnn_20260507_031905.pt`.

| Property | Value |
|---|---|
| Architecture | 3-layer edge-conditioned message passing GNN |
| Node features | 7 (SOC, T_norm, SOH, V_term, R0, \|I\|, V_oc) |
| Edge features | 3 (ΔT/10, ΔSOC, 1/d) |
| Hidden dimension | 64 |
| Total parameters | 84,804 |
| Training data | 5,000 LFP simulation rollouts, 50 epochs |
| Training seed | 42 |
| Final SOC MAE | 1.24% (per-step ΔSOC prediction) |
| Final ΔT MAE | 0.081°C |
| Best val loss | 0.014895 |

---

## Datasets

ECM parameters were extracted from **six public datasets** totalling **419,657 charge/discharge cycles** across **385 cells**. The datasets are **not included** in this repository. Download links and usage in the paper are listed below.

| Dataset | Chemistry | Form | Cells | Cycles | Used for | Reference |
|---|---|---|---|---|---|---|
| [MATR](https://data.matr.io/1/) | LFP | 18650 | 180 | 154,231 | ECM extraction, thermal calibration, SOC-GRU training, pack simulator sampling, closed-loop replay (Table 9) | Toyota Research Institute, 2022 |
| [CALCE](https://calce.umd.edu/battery-data) | LCO | Prismatic | 13 | 14,298 | ECM extraction, cross-chemistry test (Table 6) | CALCE Battery Group, UMD, 2020 |
| [RWTH](https://publications.rwth-aachen.de/record/818642) | NMC | 18650 | 48 | 105,006 | ECM extraction, cross-chemistry test (Table 6) | RWTH Aachen University, 2021 |
| [HUST](https://doi.org/10.1016/j.ensm.2021.08.019) | LFP | - | 77 | 146,122 | ECM extraction (IR-based) | HUST Battery Test Lab, 2020 |
| [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | Li-ion | - | 59 | - | Thermal model validation | Saha & Goebel, NASA, 2007 |
| [Oxford Deg1](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac) | NMC | Pouch | 8 | 8,200 cyc | SOH fade validation (Section 5.4) | Howey & Birkl, Oxford, 2017 |

> **Train/test separation:** The GNN and ECM parameters are calibrated exclusively on MATR (LFP).
> CALCE (LCO) and RWTH (NMC) are used only for evaluation - never seen during training.

---

## Hyperparameters

All hyperparameters are fixed across all experiments (no per-experiment tuning).

| Parameter | Value | Description |
|---|---|---|
| N cells | 12 | Default pack size |
| Δt | 60 s | Control timestep |
| SOC range | 20% → 80% | Charging window |
| K | 64 | CEM samples per iteration |
| K_elite | 16 | CEM elite samples (top 25%) |
| H | 5 | MPC horizon (steps) |
| T_iter | 5 | CEM refinement iterations |
| λ = [λ₁,λ₂,λ₃,λ₄,λ₅] | [4, 3, 3, 2, 50] | Cost weights: time, SOC imbalance, thermal, aging, violations |
| I_max | 3.0C | Per-cell C-rate limit |
| T_max | 45°C | Hard temperature limit |
| T_comfort | 38°C | Soft thermal penalty threshold |
| V_min / V_max | 2.5 V / 4.2 V | Voltage limits |
| GNN hidden | 64 | Hidden dimension |
| GNN layers | 3 | Message passing layers |
| Learning rate | 10⁻³ | Adam optimiser |
| Batch size | 256 | Training batch size |
| GNN rollouts | 5,000 | Training episodes |
| GNN epochs | 50 | Training epochs |

---

## Method Summary

```
Pack state G_t (cells as nodes, thermal/electrical edges)
        │
        ▼
PackGNN (edge-conditioned message passing)
   → Predicts ΔSOC, ΔT, aging_rate per cell
        │
        ▼
MPC-CEM loop (K=64 samples, H=5 horizon, T_iter=5 iterations)
   → Optimises: time + SOC imbalance + thermal + aging + violations
   → Selects best current vector I*(t)
        │
        ▼
Pack step (1-RC ECM + lumped thermal + aging model)
   → Updates G_{t+1}
```

The GNN serves as a **fast surrogate** (≈50× faster than physics simulation) within the CEM loop for rollout cost estimation. The actual pack state update always uses the physics simulator.

---

## Citation

If you use this code, please cite:

```bibtex
@article{bingolSoyluBaheri2026safety,
  title   = {Safety-First Charging Optimisation: Thermal Gradient and
             {SOC} Balance at the Cost of Speed},
  author  = {Bing{\"o}l, Alper and Soylu, M{\"u}cahit and Baheri, Ali},
  journal = {[TO BE ADDED UPON ACCEPTANCE]},
  year    = {2026}
}
```

---

## License

This code is released for research purposes. The pre-trained model weights are provided under the same terms.
