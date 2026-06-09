# Safety-First Charging Optimisation: Thermal Gradient and SOC Balance at the Cost of Speed

**Dynamic Graph-Based Safe Charging Optimisation for Lithium-Ion Battery Packs**

Alper Bingöl¹ · Mücahit Soylu² · Ali Baheri³

¹ İnönü University, Faculty of Arts and Sciences, Department of Physics, Malatya, Turkey  
² İnönü University, Faculty of Engineering, Department of Software Engineering, Malatya, Turkey  
³ Rochester Institute of Technology, Department of Mechanical Engineering, Safe AI Lab, Rochester, NY, USA

---

## Overview

This repository contains the source code, experiment scripts, and pre-trained PackGNN model for the manuscript:

> **Safety-First Charging Optimisation: Thermal Gradient and SOC Balance at the Cost of Speed**

The method models a lithium-ion battery pack as a time-varying graph: cells are represented as nodes, and thermal/electrical couplings are represented as edges. A Graph Neural Network (PackGNN) is trained as a fast surrogate for multi-step state prediction and embedded inside a Model Predictive Control (MPC) + Cross-Entropy Method (CEM) optimisation loop.

The controller is intentionally **safety-first** rather than throughput-maximising. It prioritises SOC uniformity, inter-cell thermal-gradient reduction, and constraint satisfaction, accepting longer charging time when required.

**Primary results on a 12-cell LFP pack (Table 3 in the manuscript; N = 30 episodes, episode-specific random seeds):**

| Controller | Time (min) | σ_SOC (%) | T_max (°C) | ΔT (°C) | Aging (×10⁻⁴) | Violations (avg/ep) |
|---|---:|---:|---:|---:|---:|---:|
| CC-CV | 10.7 ± 0.5 | 8.76 ± 0.3 | 27.9 ± 0.3 | 1.02 ± 0.1 | 8.0 ± 0.0 | 0.1 |
| CC-CV-Balance | 11.0 ± 0.0 | 6.70 ± 0.0 | 27.6 ± 0.1 | 0.91 ± 0.1 | 8.0 ± 0.0 | 0.0 |
| Proportional | 13.0 ± 0.4 | 0.53 ± 0.1 | 27.0 ± 0.2 | 1.04 ± 0.1 | 8.0 ± 0.0 | 0.0 |
| SimpleMPC | 30.4 ± 1.1 | 1.92 ± 0.3 | 26.1 ± 0.2 | 0.64 ± 0.1 | 10.0 ± 0.0 | 0.0 |
| **GraphOptimizer (ours)** | **26.1 ± 0.6** | **0.54 ± 0.1** | **25.6 ± 0.1** | **0.21 ± 0.1** | **8.0 ± 0.0** | **0.0** |

Relative to CC-CV, GraphOptimizer achieves:

- **93.8% lower SOC imbalance**
- **79.4% lower inter-cell thermal gradient**
- **zero observed safety violations** across 30 primary test episodes
- approximately **5× lower thermal gradient** than the SOC-focused Proportional baseline (0.21°C vs. 1.04°C)

The longer charging time (26.1 min vs. 10.7 min for CC-CV) is deliberate and reflects the safety-first speed–safety trade-off studied in the manuscript.

---

## Repository Structure

```text
├── src/
│   ├── graph_battery_pack.py          # BatteryPackGraph, ECM1RC, ThermalModel,
│   │                                  # AgingCostModel, PackGNN
│   ├── safe_fast_charge_optimizer.py  # GraphGuidedOptimizer (MPC+CEM+GNN),
│   │                                  # CC-CV, CC-CV-Balance, Proportional,
│   │                                  # SimpleMPC baselines
│   ├── train_gnn.py                   # PackGNN training on simulation rollouts
│   ├── ablation_study.py              # Component ablation study (Table 8)
│   ├── cost_sensitivity.py            # Optional cost-weight sensitivity analysis
│   ├── offline_replay_validation.py   # Parameter mismatch robustness (Table 5),
│   │                                  # cross-chemistry validation (Table 7),
│   │                                  # and MATR closed-loop replay (Table 10)
│   └── hardware_aware_validation.py   # Pseudo-HIL validation with ADC noise,
│                                      # slew-rate limits and latency (Table 11)
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

**Hardware used in the manuscript experiments:** NVIDIA L40S GPU (48 GB VRAM), Intel Xeon CPU, 256 GB RAM, Ubuntu 22.04, CUDA 13.0, PyTorch 2.11.0.

---

## Quickstart

### 1. Reproduce primary charging results (Table 3)

```bash
cd src
python safe_fast_charge_optimizer.py \
    --n_episodes 30 \
    --n_cells 12 \
    --chemistry LFP \
    --output_dir ../results/optimizer_final
```

The script auto-loads the pre-trained GNN checkpoint from `results/models/pack_gnn_*.pt`. Results are saved to `results/optimizer_final/experiment_results_*.json`.

> **Reproducibility note:** Primary experiments use episode-specific random seeds for pack initialisation. GNN training uses a fixed seed of 42. Minor differences can appear if library versions, GPU kernels, or floating-point backends differ.

### 2. Train PackGNN from scratch

ECM parameters must first be extracted from the public datasets listed in [Datasets](#datasets). The raw datasets are not included in this repository.

```bash
# Step 1: extract ECM parameters from raw dataset files
python ecm_fitting.py --datasets all --output_dir ../results/ecm

# Step 2: train the PackGNN surrogate
python train_gnn.py \
    --n_rollouts 5000 \
    --epochs 50 \
    --chemistry LFP \
    --output_dir ../results/models
```

### 3. Component ablation study (Table 8; Figure 6)

```bash
python ablation_study.py \
    --n_episodes 30 \
    --n_cells 12 \
    --output_dir ../results/ablation_components
```

The main manuscript reports the bar-chart ablation result as **Figure 6**. The corresponding multi-objective ablation radar is provided as **Supplementary Figure S1**.

### 4. Optional cost-weight sensitivity analysis

```bash
python cost_sensitivity.py \
    --n_episodes 10 \
    --output_dir ../results/sensitivity
```

This script is provided for exploration of alternative speed–safety operating points. The manuscript focuses on the safety-first configuration `λ = [4, 3, 3, 2, 50]`.

### 5. Parameter mismatch, cross-chemistry and MATR replay validation (Tables 5, 7 and 10)

```bash
python offline_replay_validation.py \
    --n_episodes 20 \
    --output_dir ../results/replay_validation
```

This script covers:

- parameter mismatch robustness (**Table 5**),
- cross-chemistry evaluation on LFP/NMC/LCO (**Table 7**),
- closed-loop replay using real MATR-derived initial conditions (**Table 10**).

### 6. Hardware-aware pseudo-HIL validation (Table 11)

```bash
python hardware_aware_validation.py \
    --n_episodes 20 \
    --output_dir ../results/hardware_validation
```

This script evaluates robustness to ADC noise, current slew-rate limits, and control-loop latency.

---

## Pre-trained Model

The pre-trained PackGNN checkpoint is provided at:

```text
results/models/pack_gnn_20260507_031905.pt
```

| Property | Value |
|---|---:|
| Architecture | 3-layer edge-conditioned message passing GNN |
| Node features | 7: SOC, T_norm, SOH, V_term, R0, \\|I\\|, V_oc |
| Edge features | 3: ΔT/10, ΔSOC, 1/d |
| Hidden dimension | 64 |
| Total parameters | 84,804 |
| Training data | 5,000 LFP simulation rollouts |
| Training epochs | 50 |
| Training seed | 42 |
| Final SOC MAE | 1.24% per-step ΔSOC prediction |
| Final ΔT MAE | 0.081°C |
| Best validation loss | 0.014895 |

---

## Datasets

The manuscript uses six public battery datasets totalling **427,857+ cycles across 385 cells**. ECM parameters are extracted from MATR, CALCE, RWTH and HUST, totalling **419,657 cycles**. Oxford Deg1 is reserved for independent SOH validation, and NASA PCoE is used for thermal-model validation context.

The datasets are **not included** in this repository. Download links and manuscript usage are listed below.

| Dataset | Chemistry | Form | Cells | Cycles | Manuscript usage | Reference / source |
|---|---|---|---:|---:|---|---|
| [MATR](https://data.matr.io/1/) | LFP | 18650 | 180 | 154,231 | ECM extraction, thermal anchoring, SOC-GRU training, pack simulator sampling, MATR closed-loop replay (Table 10) | Toyota Research Institute / MATR |
| [CALCE](https://calce.umd.edu/battery-data) | LCO | Prismatic | 13 | 14,298 | ECM extraction and LCO cross-chemistry evaluation (Table 7) | CALCE Battery Group, University of Maryland |
| [RWTH](https://publications.rwth-aachen.de/record/818642) | NMC | 18650 | 48 | 105,006 | ECM extraction and NMC cross-chemistry evaluation (Table 7) | RWTH Aachen University |
| [HUST](https://doi.org/10.1016/j.ensm.2021.08.019) | LFP | — | 77 | 146,122 | ECM extraction using internal-resistance-based parameterisation | HUST Battery Test Lab |
| [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | Li-ion | — | 59 | — | Thermal-model validation context | NASA Prognostics Center of Excellence |
| [Oxford Deg1](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac) | NMC | Pouch | 8 | 8,200 cyc | Independent SOH fade validation (Figure 5 / Section 4.9) | Oxford Battery Degradation Dataset |

> **Train/test separation:** PackGNN training uses MATR-derived LFP simulation rollouts. CALCE (LCO) and RWTH (NMC) are used only for held-out cross-chemistry evaluation and are not used during GNN training.

---

## Hyperparameters

All hyperparameters are fixed across the manuscript experiments unless explicitly stated otherwise.

| Parameter | Value | Description |
|---|---:|---|
| N cells | 12 | Default pack size |
| Δt | 60 s | Control timestep |
| SOC range | 20% → 80% | Charging window |
| K | 64 | CEM samples per iteration |
| K_elite | 16 | CEM elite samples (top 25%) |
| H | 5 | MPC horizon (steps) |
| T_iter | 5 | CEM refinement iterations |
| λ = [λ₁, λ₂, λ₃, λ₄, λ₅] | [4, 3, 3, 2, 50] | Cost weights: time, SOC imbalance, thermal, aging, violations |
| I_max | 3.0C | Per-cell C-rate limit |
| T_lim | 45°C | Hard temperature limit |
| T_comfort | 38°C | Soft thermal penalty threshold |
| V_min / V_max | 2.5 V / 4.2 V | Voltage limits |
| GNN hidden dimension | 64 | Hidden dimension |
| GNN layers | 3 | Message-passing layers |
| Learning rate | 10⁻³ | Adam optimiser |
| Batch size | 256 | Training batch size |
| GNN rollouts | 5,000 | Training rollouts |
| GNN epochs | 50 | Training epochs |

---

## Method Summary

```text
Pack state G_t
(cells as nodes, thermal/electrical edges)
        │
        ▼
PackGNN
(edge-conditioned message passing)
        │
        ├── predicts SOC-related state changes
        ├── predicts temperature changes
        └── predicts per-cell aging-rate proxy
        │
        ▼
MPC-CEM loop
(K = 64 samples, H = 5 horizon, T_iter = 5 iterations)
        │
        ├── evaluates candidate current vectors
        ├── penalises time, SOC imbalance, thermal stress,
        │   aging proxy and constraint violations
        └── selects best current vector I*(t)
        │
        ▼
Physics simulator step
(1-RC ECM + lumped thermal model + empirical aging proxy)
        │
        ▼
Updated pack graph G_{t+1}
```

The PackGNN is used as a fast surrogate inside the MPC-CEM rollout loop. The closed-loop plant update uses the physics simulator rather than replacing the plant dynamics with the neural network.

---

## Important Interpretation Notes

- The proposed controller is a **safety-first charging layer**, not a direct claim of universally faster charging than CC-CV.
- The aging term is an empirical **per-episode proxy and regularisation term**. It is not calibrated for absolute lifetime forecasting.
- The reported thermal metric ΔT is the **inter-cell pack-level temperature spread** (`max_i T_i - min_i T_i`), not the internal temperature gradient within a single cell.
- Per-cell current allocation assumes an actuator layer such as active cell-level power converters or a reconfigurable BMS, consistent with the hardware assumption stated in the manuscript.

---

## Citation

If you use this code, please cite:

```bibtex
@article{bingolSoyluBaheri2026safety,
  title   = {Safety-First Charging Optimisation: Thermal Gradient and
             {SOC} Balance at the Cost of Speed},
  author  = {Bing{"o}l, Alper and Soylu, M{"u}cahit and Baheri, Ali},
  journal = {[TO BE ADDED UPON ACCEPTANCE]},
  year    = {2026}
}
```

---

## License

This code is released for research purposes. The pre-trained model weights are provided under the same terms.
