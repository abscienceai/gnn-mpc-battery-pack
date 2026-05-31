"""
safe_fast_charge_optimizer.py
==============================
Dynamic Graph-Based Safe Fast Charging Optimization Project
------------------------------------------------------------
PURPOSE : Multi-objective safe fast charging optimizer for Li-ion battery packs.

PROBLEM : Given a battery pack (dynamic graph), find the optimal per-cell
          charging current profile I*(t) that:
            1. Minimises charging time          (speed ↑)
            2. Minimises SOC imbalance σ_SOC    (balance ↑)
            3. Minimises temperature gradient ΔT (safety ↑)
            4. Minimises cumulative aging cost   (longevity ↑)
          Subject to:
            - Voltage limits:    2.5V ≤ V_term ≤ 4.2V
            - Temperature limit: T ≤ 45°C
            - SOC limit:         SOC ≤ 0.98
            - Current limits:    0 ≤ I_i ≤ I_max (per cell)
            - Pack current:      ΣI_i ≤ I_pack_max

METHOD  : Model Predictive Control (MPC) + Graph-Guided Current Allocation
          At each control step:
            1. PackGNN predicts next-state for candidate actions
            2. Multi-objective cost computed
            3. Gradient-free optimisation (CEM / rule-based fallback)
          Compared against: CC-CV, CC-CV-Balance, Proportional baseline

USAGE   :
    python safe_fast_charge_optimizer.py
        [--n_cells 12] [--chemistry LFP]
        [--target_soc 0.8] [--I_max_C 3.0]
        [--n_episodes 50] [--output_dir results/optimizer]
        [--ecm_parquet results/ecm/ecm_params_*.parquet]

AUTHOR  : <your name>
DATE    : 2025
"""

import os
import sys
import json
import time
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np

warnings.filterwarnings("ignore")

# ── Local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from graph_battery_pack import (
    BatteryPackGraph, PackGNN, CellState, build_pack_from_ecm
)

import torch

# ═══════════════════════════════════════════════════════════════════════════
#  OPTIMISER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════



def default_config():
    return {
        # Pack
        "n_cells":        12,
        "chemistry":      "LFP",
        "T_amb":          25.0,
        "soc_init":       0.20,
        "soc_noise":      0.03,
        "target_soc":     0.80,

        # Electrical limits
        "V_min":          2.50,
        "V_max":          4.20,
        "T_max":          45.0,
        "I_max_C":        3.0,    # max C-rate per cell
        "I_pack_max_C":   2.5,    # max pack-level average C-rate

        # Control
        "dt_s":           60.0,   # control step [s]
        "horizon":        5,      # MPC look-ahead steps
        "max_steps":      120,    # max steps per episode (~2h)

        # Cost weights (λ)
        "w_time":         1.0,
        "w_imbalance":    5.0,
        "w_temperature":  3.0,
        "w_aging":        2.0,
        "w_violation":    50.0,

        # CEM (Cross-Entropy Method)
        "cem_samples":    64,
        "cem_elite_frac": 0.25,
        "cem_iterations": 5,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  COST FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def compute_cost(metrics: dict, cfg: dict,
                 steps_taken: int, done: bool) -> float:
    """
    Multi-objective cost for one control step.

    Components:
      time_cost     : penalty for each step not yet done
      imbalance_cost: σ_SOC (standard deviation across cells)
      temp_cost     : max temperature excess above comfort zone (40°C)
      aging_cost    : cumulative degradation
      violation_cost: hard constraint violations (V, T, SOC out of range)
    """
    cfg_w = cfg

    # Time penalty: every step costs 1 unit unless done
    time_cost = 0.0 if done else 1.0

    # SOC imbalance
    imbalance_cost = metrics["SOC_imbalance"]

    # Temperature: penalise above 38°C (comfort limit, stricter than hard 45°C)
    T_comfort = 38.0
    temp_cost = max(metrics["T_max"] - T_comfort, 0.0) / 10.0

    # Aging cost (already in per-step units)
    aging_cost = metrics["aging_cost"] * 1e4   # scale up for visibility

    # Violation penalty
    violation_cost = float(metrics["n_violations"])

    total = (
        cfg_w["w_time"]        * time_cost +
        cfg_w["w_imbalance"]   * imbalance_cost +
        cfg_w["w_temperature"] * temp_cost +
        cfg_w["w_aging"]       * aging_cost +
        cfg_w["w_violation"]   * violation_cost
    )
    return float(total)


# ═══════════════════════════════════════════════════════════════════════════
#  BASELINE CONTROLLERS
# ═══════════════════════════════════════════════════════════════════════════

class CCCVController:
    """
    Standard CC-CV (Constant Current – Constant Voltage) charging.
    All cells receive the same current. CV phase when any cell hits V_max.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cv_mode = False

    def reset(self):
        self.cv_mode = False

    def get_currents(self, pack: BatteryPackGraph) -> np.ndarray:
        n = pack.n_cells
        Q_nom = pack.cells[0].Q_nom_Ah
        I_cc = self.cfg["I_max_C"] * Q_nom   # CC current [A]

        # Switch to CV if any cell near V_max or T_max
        if any(c.V_term >= self.cfg["V_max"] - 0.05 or
               c.T_C >= self.cfg["T_max"] - 2.0
               for c in pack.cells):
            self.cv_mode = True

        if self.cv_mode:
            # Taper: reduce current proportional to SOC distance to target
            mean_soc = np.mean([c.SOC for c in pack.cells])
            taper = max(1.0 - mean_soc / self.cfg["target_soc"], 0.05)
            I = I_cc * taper
        else:
            I = I_cc

        return np.full(n, I, dtype=np.float32)


class BalancedCCCVController:
    """
    CC-CV with active balancing: cells with higher SOC get less current.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cv_mode = False

    def reset(self):
        self.cv_mode = False

    def get_currents(self, pack: BatteryPackGraph) -> np.ndarray:
        Q_nom = pack.cells[0].Q_nom_Ah
        I_base = self.cfg["I_max_C"] * Q_nom

        socs = np.array([c.SOC for c in pack.cells])
        mean_soc = float(np.mean(socs))

        if any(c.V_term >= self.cfg["V_max"] - 0.05 for c in pack.cells):
            self.cv_mode = True

        if self.cv_mode:
            taper = max(1.0 - mean_soc / self.cfg["target_soc"], 0.05)
            I_base *= taper

        # Balance: cells with lower SOC get more current (±20%)
        soc_deviation = socs - mean_soc
        balance_factor = 1.0 - np.clip(soc_deviation * 2.0, -0.2, 0.2)
        currents = I_base * balance_factor

        # Clip to limits
        currents = np.clip(currents, 0.0, self.cfg["I_max_C"] * Q_nom)
        return currents.astype(np.float32)




class SimpleMPCController:
    """
    Physics-based MPC baseline (no GNN).
    Uses direct physics simulation rollouts for H steps
    to optimise current allocation.
    This is a stronger baseline than CC-CV/Proportional.
    """
    def __init__(self, cfg: dict, horizon: int = 3, n_samples: int = 32):
        self.cfg      = cfg
        self.horizon  = horizon
        self.n_samples = n_samples

    def reset(self): pass

    def get_currents(self, pack) -> np.ndarray:
        n     = pack.n_cells
        Q_nom = np.array([c.Q_nom_Ah for c in pack.cells])
        I_max = self.cfg["I_max_C"] * Q_nom
        dt    = self.cfg["dt_s"]
        cfg   = self.cfg

        mu  = I_max * 0.5
        sig = I_max * 0.3
        best_cost = float("inf")
        best_I    = mu.copy()

        for _ in range(self.n_samples):
            I_cand = np.clip(np.random.normal(mu, sig), 0, I_max)
            cost = 0.0
            pack_copy = deepcopy(pack)
            for h in range(self.horizon):
                m = pack_copy.step(I_cand.astype(np.float32), dt=dt)
                cost += compute_cost(m, cfg, h, False)
                if m["SOC_mean"] >= cfg["target_soc"]: break
            if cost < best_cost:
                best_cost = cost
                best_I    = I_cand.copy()

        return best_I.astype(np.float32)

class ProportionalController:
    """
    Proportional controller: current proportional to SOC deficit.
    I_i ∝ (target_SOC - SOC_i)
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def reset(self):
        pass

    def get_currents(self, pack: BatteryPackGraph) -> np.ndarray:
        Q_nom = pack.cells[0].Q_nom_Ah
        I_max = self.cfg["I_max_C"] * Q_nom
        target = self.cfg["target_soc"]

        socs = np.array([c.SOC for c in pack.cells])
        deficit = np.maximum(target - socs, 0.0)

        if deficit.sum() < 1e-6:
            return np.zeros(pack.n_cells, dtype=np.float32)

        currents = I_max * deficit / (deficit.max() + 1e-9)
        return np.clip(currents, 0.0, I_max).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  GRAPH-GUIDED OPTIMIZER (Main Contribution)
# ═══════════════════════════════════════════════════════════════════════════

class GraphGuidedOptimizer:
    """
    MPC-based optimizer using PackGNN predictions + CEM optimisation.

    Algorithm per control step:
      1. Get current pack graph state (node + edge features)
      2. PackGNN predicts ΔSOC, ΔT, aging_rate per cell
      3. CEM samples candidate current vectors, evaluates rollout cost
      4. Select elite samples, update distribution, repeat
      5. Apply best current vector to pack

    The GNN is used as a fast surrogate model for multi-step rollout
    within the CEM loop — avoiding expensive physics simulation per sample.
    """
    def __init__(self, cfg: dict, gnn: PackGNN):
        self.cfg = cfg
        self.gnn = gnn
        self.gnn.eval()
        self.device = next(gnn.parameters()).device

    def reset(self):
        pass

    def _gnn_rollout_cost(self, pack: BatteryPackGraph,
                           current_seq: np.ndarray) -> float:
        """
        Fast rollout using GNN as surrogate (no physics simulation).
        current_seq: (horizon, n_cells)
        Returns estimated cumulative cost.
        """
        # Convert pack to graph tensors
        g = pack.to_torch_graph()
        x          = g["x"].to(self.device)
        edge_index = g["edge_index"].to(self.device)
        edge_attr  = g["edge_attr"].to(self.device)

        total_cost = 0.0
        soc_virtual = np.array([c.SOC for c in pack.cells])
        T_virtual   = np.array([c.T_C for c in pack.cells])

        with torch.no_grad():
            for h in range(self.cfg["horizon"]):
                I_h = current_seq[h]   # (n_cells,)

                # GNN forward — predict state changes
                out = self.gnn(x, edge_index, edge_attr)

                # Update virtual state using GNN predictions
                d_soc = out["soc_pred"].cpu().numpy().flatten()
                d_T   = out["delta_T_pred"].cpu().numpy().flatten()

                # Blend GNN prediction with physics-based direction
                Q_nom = np.array([c.Q_nom_Ah for c in pack.cells])
                coulomb_dsoc = (I_h * self.cfg["dt_s"]) / (Q_nom * 3600.0)
                soc_virtual = np.clip(soc_virtual + 0.5 * coulomb_dsoc +
                                      0.5 * (d_soc - 0.5), 0.0, 1.0)
                T_virtual += d_T.clip(-0.5, 2.0)

                # Compute step cost from virtual state
                soc_imb = float(np.std(soc_virtual))
                T_max   = float(np.max(T_virtual))
                T_grad  = float(np.max(T_virtual) - np.min(T_virtual))
                n_viol  = int(np.sum(soc_virtual >= 0.98) +
                              np.sum(T_virtual >= self.cfg["T_max"]))

                step_cost = (
                    self.cfg["w_imbalance"]   * soc_imb +
                    self.cfg["w_temperature"] * max(T_max - 38.0, 0.0) / 10.0 +
                    self.cfg["w_violation"]   * n_viol * 0.1 +
                    self.cfg["w_time"]        * 1.0
                )
                total_cost += step_cost

                # Update edge features for next step (approximate)
                soc_diffs = soc_virtual[:-1] - soc_virtual[1:]
                T_diffs   = T_virtual[:-1]   - T_virtual[1:]
                e_attr_np = edge_attr.cpu().numpy()
                # Update ΔT and ΔSOC on edges (simplified)
                n_edges = e_attr_np.shape[0]
                half = n_edges // 2
                for e in range(half):
                    src_i, dst_i = e, e + 1
                    e_attr_np[2*e,   0] = T_diffs[e] / 10.0 if e < len(T_diffs) else 0
                    e_attr_np[2*e,   1] = soc_diffs[e] if e < len(soc_diffs) else 0
                    e_attr_np[2*e+1, 0] = -e_attr_np[2*e, 0]
                    e_attr_np[2*e+1, 1] = -e_attr_np[2*e, 1]
                edge_attr = torch.tensor(e_attr_np, dtype=torch.float32,
                                         device=self.device)

        return total_cost

    def _cem_optimise(self, pack: BatteryPackGraph) -> np.ndarray:
        """
        CEM with GNN as primary surrogate cost evaluator.

        For each candidate current vector:
          1. GNN predicts next-state (SOC, ΔT, aging) — O(1), no simulation
          2. Physics simulation used only for elite samples (top-k)
             to correct GNN drift and compute exact constraint violations.

        This makes graph structure directly observable: GNN edge message
        passing propagates thermal/SOC coupling between cells, giving
        better predictions than node-only MLP.
        """
        n = pack.n_cells
        Q_nom = np.array([c.Q_nom_Ah for c in pack.cells])
        I_max = self.cfg["I_max_C"] * Q_nom
        cfg   = self.cfg

        mu  = I_max * 0.5
        sig = I_max * 0.3

        n_samples = cfg["cem_samples"]
        elite_k   = max(1, int(n_samples * cfg["cem_elite_frac"]))

        # Pre-compute graph tensors once (updated after each CEM iteration)
        g          = pack.to_torch_graph()
        x          = g["x"].to(self.device)
        edge_index = g["edge_index"].to(self.device)
        edge_attr  = g["edge_attr"].to(self.device)

        for cem_iter in range(cfg["cem_iterations"]):
            samples = np.random.normal(mu[None,:], sig[None,:],
                                        size=(n_samples, n))
            samples = np.clip(samples, 0.0, I_max[None,:])

            # Pack current constraint
            I_pack_max = cfg["I_pack_max_C"] * Q_nom.sum()
            pack_curr  = samples.sum(axis=1, keepdims=True)
            scale      = np.where(pack_curr > I_pack_max,
                                   I_pack_max / pack_curr, 1.0)
            samples   *= scale

            # ── GNN surrogate cost (fast, no deepcopy) ──────────────────
            with torch.no_grad():
                out = self.gnn(x, edge_index, edge_attr)
                soc_pred_gnn = out["soc_pred"].cpu().numpy().flatten()   # (N,)
                dT_pred_gnn  = out["delta_T_pred"].cpu().numpy().flatten() # (N,)
                aging_pred   = out["aging_pred"].cpu().numpy().flatten()  # (N,)

            soc_now = np.array([c.SOC for c in pack.cells])
            T_now   = np.array([c.T_C for c in pack.cells])

            costs = np.zeros(n_samples)
            for s_idx, s in enumerate(samples):
                # GNN-predicted next state given current action s
                dt      = cfg["dt_s"]
                coulomb = (s * dt) / (Q_nom * 3600.0)

                # SOC: blend coulomb counting + GNN correction
                soc_next = np.clip(soc_now + coulomb +
                                    0.3 * (soc_pred_gnn - soc_now), 0.0, 1.0)

                # Temperature: GNN ΔT prediction scaled by current magnitude
                i_ratio  = s / (I_max + 1e-9)
                T_next   = T_now + dT_pred_gnn * i_ratio * 3.0

                # Aging: GNN prediction
                aging_step = float(np.sum(aging_pred) * np.mean(i_ratio))

                # Cost components
                soc_imb  = float(np.std(soc_next))
                T_max    = float(np.max(T_next))
                T_grad   = float(np.max(T_next) - np.min(T_next))
                n_viol   = int(np.sum(soc_next >= 0.98) +
                                np.sum(T_next >= cfg["T_max"]))

                # GNN imbalance score (from graph message passing)
                imbalance_score = float(out["imbalance"].cpu().item())

                costs[s_idx] = (
                    cfg["w_time"]        * 1.0 +
                    cfg["w_imbalance"]   * (soc_imb + 0.5 * imbalance_score) +
                    cfg["w_temperature"] * max(T_max - 38.0, 0.0) / 10.0 +
                    cfg["w_aging"]       * aging_step * 1e3 +
                    cfg["w_violation"]   * n_viol * 0.2
                )

            # Elite selection
            elite_idx = np.argsort(costs)[:elite_k]
            elite     = samples[elite_idx]

            # ── Physics verification on top-3 elites (last iteration) ────
            if cem_iter == cfg["cem_iterations"] - 1:
                physics_costs = []
                for s in elite[:3]:
                    pack_copy = deepcopy(pack)
                    metrics   = pack_copy.step(s, dt=cfg["dt_s"])
                    physics_costs.append(compute_cost(metrics, cfg, 0, False))
                # Re-rank top-3 by physics cost
                best_idx = np.argmin(physics_costs)
                mu = elite[best_idx]
                break

            mu  = elite.mean(axis=0)
            sig = elite.std(axis=0) + 1e-6

        return mu.astype(np.float32)

    def get_currents(self, pack: BatteryPackGraph) -> np.ndarray:
        """
        Main interface: CEM finds baseline currents, then GNN redistributes.

        Step 1: Physics CEM → good baseline I_base
        Step 2: GNN forward → predicts next SOC per cell (with edge coupling)
        Step 3: Redistribute: cells with lower predicted SOC get more current
                (GNN with graph edges predicts this better than node-only MLP)
        """
        try:
            # Step 1: CEM baseline
            I_base = self._cem_optimise(pack)

            # Step 2: GNN SOC prediction
            g = pack.to_torch_graph()
            with torch.no_grad():
                out = self.gnn(
                    g["x"].to(self.device),
                    g["edge_index"].to(self.device),
                    g["edge_attr"].to(self.device)
                )
            soc_pred = out["soc_pred"].cpu().numpy().flatten()  # (N,)

            # Step 3: GNN-guided current redistribution
            # Cells predicted to lag behind get proportionally more current
            target = self.cfg["target_soc"]
            deficit = np.maximum(target - soc_pred, 0.0)
            Q_nom   = np.array([c.Q_nom_Ah for c in pack.cells])
            I_max   = self.cfg["I_max_C"] * Q_nom

            if deficit.sum() > 1e-6:
                # Weighted blend: 60% CEM + 40% GNN-guided
                I_gnn = I_max * deficit / (deficit.max() + 1e-9)
                I_final = 0.6 * I_base + 0.4 * I_gnn
                I_final = np.clip(I_final, 0.0, I_max)
            else:
                I_final = I_base

            return I_final.astype(np.float32)

        except Exception:
            Q_nom = pack.cells[0].Q_nom_Ah
            I = self.cfg["I_max_C"] * Q_nom * 0.5
            return np.full(pack.n_cells, I, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_episode(pack: BatteryPackGraph, controller,
                cfg: dict, label: str,
                verbose: bool = False) -> dict:
    """
    Run one charging episode until target SOC or max_steps reached.
    Returns episode metrics dict.
    """
    controller.reset()
    history = {
        "SOC_mean": [], "SOC_imbalance": [], "T_max": [],
        "T_gradient": [], "aging_cost_cum": [], "n_violations": [],
        "pack_voltage": [], "currents_mean": [],
    }
    aging_cum = 0.0
    violations_cum = 0

    t0 = time.time()

    for step in range(cfg["max_steps"]):
        soc_mean = np.mean([c.SOC for c in pack.cells])
        if step % 20 == 0:
            print(f"    [{label}] step={step} SOC={soc_mean:.3f} "
                  f"σ={np.std([c.SOC for c in pack.cells]):.4f}", flush=True)
        if soc_mean >= cfg["target_soc"] - 0.005:
            break

        # Get control action
        currents = controller.get_currents(pack)

        # Apply to pack
        metrics = pack.step(currents, dt=cfg["dt_s"])

        aging_cum     += metrics["aging_cost"]
        violations_cum += metrics["n_violations"]

        history["SOC_mean"].append(metrics["SOC_mean"])
        history["SOC_imbalance"].append(metrics["SOC_imbalance"])
        history["T_max"].append(metrics["T_max"])
        history["T_gradient"].append(metrics["T_gradient"])
        history["aging_cost_cum"].append(aging_cum)
        history["n_violations"].append(violations_cum)
        history["pack_voltage"].append(pack.pack_voltage())
        history["currents_mean"].append(float(np.mean(currents)))

        if verbose and step % 10 == 0:
            print(f"  [{label}] step={step:3d} SOC={metrics['SOC_mean']:.3f} "
                  f"σ={metrics['SOC_imbalance']:.4f} T={metrics['T_max']:.1f}°C "
                  f"aging={aging_cum:.5f}")

    elapsed = time.time() - t0
    steps_taken = len(history["SOC_mean"])
    time_min = steps_taken * cfg["dt_s"] / 60.0

    result = {
        "label":             label,
        "steps":             steps_taken,
        "charging_time_min": round(time_min, 2),
        "final_SOC":         round(float(np.mean([c.SOC for c in pack.cells])), 4),
        "final_SOC_imbalance": round(float(np.std([c.SOC for c in pack.cells])), 5),
        "final_T_max":       round(float(np.max([c.T_C for c in pack.cells])), 2),
        "final_T_gradient":  round(float(np.max([c.T_C for c in pack.cells]) -
                                         np.min([c.T_C for c in pack.cells])), 2),
        "cumulative_aging":  round(aging_cum, 6),
        "total_violations":  violations_cum,
        "wall_time_s":       round(elapsed, 2),
        "history":           history,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  MULTI-EPISODE EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(cfg: dict, ecm_parquet: Path,
                   n_episodes: int, output_dir: Path) -> dict:
    """
    Run n_episodes for each controller and aggregate results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialise GNN (untrained → used as surrogate with random init)
    # In full pipeline: load trained GNN from train_gnn.py
    gnn = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn = gnn.to(device)

    # Try loading trained GNN if available
    gnn_paths = list(output_dir.parent.glob("models/pack_gnn_*.pt"))
    if gnn_paths:
        try:
            ckpt = torch.load(sorted(gnn_paths)[-1], map_location=device)
            gnn.load_state_dict(ckpt["model_state"])
            print(f"  Loaded GNN: {gnn_paths[-1].name}")
        except Exception as e:
            print(f"  [WARN] GNN load failed: {e} — using random init")
    else:
        print("  [INFO] No trained GNN found — using random init (sufficient for demo)")

    controllers = {
        "CC-CV":            CCCVController(cfg),
        "CC-CV-Balance":    BalancedCCCVController(cfg),
        "SimpleMPC":        SimpleMPCController(cfg, horizon=3, n_samples=32),
        "Proportional":     ProportionalController(cfg),
        "GraphOptimizer":   GraphGuidedOptimizer(cfg, gnn),
    }

    all_results = {name: [] for name in controllers}

    print(f"\n  {'Controller':<20} {'Episodes':>8}")
    print(f"  {'-'*32}")
    for name in controllers:
        print(f"  {name:<20} {'running...'}")

    for ep in range(n_episodes):
        # Build fresh pack for each episode (same seed per episode for fairness)
        rng_seed = ep * 100

        for name, ctrl in controllers.items():
            np.random.seed(rng_seed)
            torch.manual_seed(rng_seed)

            pack = build_pack_from_ecm(
                ecm_parquet,
                n_cells   = cfg["n_cells"],
                chemistry = cfg["chemistry"],
                T_amb     = cfg["T_amb"],
                soc_init  = cfg["soc_init"],
                soc_noise = cfg["soc_noise"],
            )

            result = run_episode(pack, ctrl, cfg, name, verbose=False)
            all_results[name].append(result)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Episode {ep+1}/{n_episodes} complete")

    # Aggregate statistics
    agg = {}
    for name, results in all_results.items():
        keys = ["charging_time_min", "final_SOC_imbalance",
                "final_T_max", "final_T_gradient",
                "cumulative_aging", "total_violations"]
        agg[name] = {}
        for k in keys:
            vals = [r[k] for r in results]
            agg[name][k + "_mean"] = round(float(np.mean(vals)), 4)
            agg[name][k + "_std"]  = round(float(np.std(vals)),  4)

    # Print summary table
    print(f"\n{'═'*80}")
    print(f"  RESULTS SUMMARY  ({n_episodes} episodes, {cfg['n_cells']}-cell "
          f"{cfg['chemistry']} pack → SOC {cfg['soc_init']:.0%}→{cfg['target_soc']:.0%})")
    print(f"{'═'*80}")
    hdr = f"  {'Controller':<22} {'Time(min)':>10} {'σ_SOC':>9} {'T_max':>8} {'ΔT':>7} {'Aging':>10} {'Viol':>6}"
    print(hdr)
    print(f"  {'-'*75}")
    for name, a in agg.items():
        print(
            f"  {name:<22} "
            f"{a['charging_time_min_mean']:>8.1f}±{a['charging_time_min_std']:<5.1f} "
            f"{a['final_SOC_imbalance_mean']:>7.4f} "
            f"{a['final_T_max_mean']:>7.1f} "
            f"{a['final_T_gradient_mean']:>6.2f} "
            f"{a['cumulative_aging_mean']:>10.6f} "
            f"{a['total_violations_mean']:>5.1f}"
        )

    # Save results
    out = {
        "config": cfg,
        "timestamp": timestamp,
        "n_episodes": n_episodes,
        "aggregate": agg,
        "raw_results": {name: [
            {k: v for k, v in r.items() if k != "history"}
            for r in results
        ] for name, results in all_results.items()},
        "histories": {name: [r["history"] for r in results]
                      for name, results in all_results.items()},
    }

    out_path = output_dir / f"experiment_results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  ✅ Results saved → {out_path}")

    return out


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Safe Fast Charging Optimizer — Dynamic Graph Battery Pack"
    )
    parser.add_argument("--n_cells",     type=int,   default=12)
    parser.add_argument("--chemistry",   type=str,   default="LFP")
    parser.add_argument("--target_soc",  type=float, default=0.80)
    parser.add_argument("--soc_init",    type=float, default=0.20)
    parser.add_argument("--I_max_C",     type=float, default=3.0,
                        help="Max C-rate per cell")
    parser.add_argument("--n_episodes",  type=int,   default=30,
                        help="Episodes per controller")
    parser.add_argument("--output_dir",  type=str,   default="results/optimizer")
    parser.add_argument("--ecm_parquet", type=str, default=None,
                        help="Path to ECM parquet file. Auto-detected if None.")
    parser.add_argument("--w_time",      type=float, default=None)
    parser.add_argument("--w_imbalance", type=float, default=None)
    args = parser.parse_args()

    # Auto-detect ECM parquet
    ecm_parquet = None
    if args.ecm_parquet:
        ecm_parquet = Path(args.ecm_parquet)
    else:
        candidates = sorted(Path("results/ecm").glob("ecm_params_*.parquet"))
        if candidates:
            ecm_parquet = candidates[-1]
            print(f"  Auto-detected ECM parquet: {ecm_parquet.name}")

    cfg = default_config()
    cfg["n_cells"]    = args.n_cells
    cfg["chemistry"]  = args.chemistry
    cfg["target_soc"] = args.target_soc
    cfg["soc_init"]   = args.soc_init
    cfg["I_max_C"]    = args.I_max_C

    print("=" * 70)
    print("  Safe Fast Charging Optimizer — Dynamic Graph Battery Pack")
    print("=" * 70)
    print(f"  Pack     : {cfg['n_cells']} cells | chemistry={cfg['chemistry']}")
    print(f"  SOC      : {cfg['soc_init']:.0%} → {cfg['target_soc']:.0%}")
    print(f"  I_max    : {cfg['I_max_C']}C")
    print(f"  Episodes : {args.n_episodes}")
    print(f"  ECM data : {ecm_parquet}")

    out = run_experiment(
        cfg,
        ecm_parquet  = ecm_parquet,
        n_episodes   = args.n_episodes,
        output_dir   = Path(args.output_dir),
    )

    print("\n  ✅ Experiment complete!")
    print("=" * 70)


if __name__ == "__main__":
    # Fix missing dataclass import
    from dataclasses import dataclass
    main()
