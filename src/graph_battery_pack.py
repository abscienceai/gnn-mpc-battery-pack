"""
graph_battery_pack.py
=====================
Dynamic Graph-Based Safe Fast Charging Optimization Project
------------------------------------------------------------
PURPOSE : Model a lithium-ion battery pack as a dynamic graph where:
          - Nodes  = individual cells  (features: SOC, T, SOH, R0, V)
          - Edges  = thermal + electrical coupling between adjacent cells
          - Graph topology updates every control step (dynamic)

          Implements:
            BatteryPackGraph   — graph construction & state update
            PackGNN            — Graph Neural Network for SOC/T propagation
            GraphDataset       — training data builder

USAGE   : imported by safe_fast_charge_optimizer.py and train_gnn.py

AUTHOR  : <your name>
DATE    : 2025
"""

import math
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════
#  CELL STATE DATACLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CellState:
    """State of a single cell at time t."""
    cell_id:   str
    SOC:       float   # State of Charge       [0, 1]
    T_C:       float   # Temperature           [°C]
    SOH:       float   # State of Health       [0, 1]
    V_oc:      float   # Open Circuit Voltage  [V]
    V_term:    float   # Terminal Voltage      [V]
    R0:        float   # Ohmic resistance      [Ω]
    R1:        float   # Polarisation R        [Ω]
    C1:        float   # Polarisation C        [F]
    I_A:       float   # Applied current       [A]  (+ = charge)
    Q_nom_Ah:  float   # Nominal capacity      [Ah]
    chemistry: str     = "LFP"

    def feature_vector(self) -> np.ndarray:
        """Node feature vector: [SOC, T_norm, SOH, V_term, R0, |I|, V_oc]"""
        return np.array([
            self.SOC,
            (self.T_C - 25.0) / 30.0,   # normalise around 25°C
            self.SOH,
            self.V_term / 4.5,           # normalise to max voltage
            np.clip(self.R0 * 100, 0, 10),  # mΩ → scaled
            np.abs(self.I_A) / 10.0,     # normalise to 10A
            self.V_oc / 4.5,
        ], dtype=np.float32)

    @property
    def is_safe(self) -> bool:
        """True iff cell is within safe operating limits."""
        return (
            0.05 <= self.SOC <= 0.98 and
            self.T_C <= 45.0 and
            self.T_C >= -10.0 and
            self.V_term <= 4.25 and
            self.V_term >= 2.5
        )


# ═══════════════════════════════════════════════════════════════════════════
#  ECM FORWARD MODEL  (1-RC Thevenin)
# ═══════════════════════════════════════════════════════════════════════════

class ECM1RC:
    """
    1-RC Equivalent Circuit Model for one cell.
    State: [SOC, V1]  where V1 = voltage across RC branch.
    """
    def __init__(self, R0: float, R1: float, C1: float,
                 Q_nom_Ah: float, T_ref: float = 25.0):
        self.R0     = max(R0, 1e-4)
        self.R1     = max(R1, 1e-4)
        self.C1     = max(C1, 1.0)
        self.tau1   = self.R1 * self.C1
        self.Q_nom  = Q_nom_Ah * 3600.0   # [As]
        self.T_ref  = T_ref
        self._V1    = 0.0   # polarisation voltage

    def ocv_from_soc(self, soc: float, chemistry: str = "LFP") -> float:
        """
        Lookup OCV from SOC using empirical polynomial.
        Chemistry-specific coefficients (fitted from dataset averages).
        """
        s = np.clip(soc, 0.0, 1.0)
        if chemistry == "LFP":
            # LFP: flat plateau around 3.3V
            return 3.0 + 0.6 * s - 0.3 * s**2 + 0.15 * s**3
        elif chemistry == "NMC":
            # NMC: monotonic from 3.0V to 4.2V
            return 3.0 + 1.2 * s + 0.15 * s**2 - 0.1 * s**3
        elif chemistry == "LCO":
            # LCO: steep curve 3.6–4.2V
            return 3.6 + 0.55 * s + 0.05 * s**2
        else:
            return 3.2 + 1.0 * s

    def step(self, I: float, dt: float, soc: float,
             chemistry: str = "LFP") -> Tuple[float, float, float]:
        """
        Advance ECM by one timestep dt [s] with current I [A].
        Returns (new_soc, V_terminal, V_oc).
        Convention: I > 0 = charging.
        """
        # SOC update (coulomb counting)
        d_soc = (I * dt) / self.Q_nom
        new_soc = float(np.clip(soc + d_soc, 0.0, 1.0))

        # RC branch update (Euler)
        d_V1 = (I * self.R1 - self._V1) / self.tau1
        self._V1 += d_V1 * dt
        self._V1 = float(np.clip(self._V1, -1.0, 1.0))

        # OCV
        V_oc = self.ocv_from_soc(new_soc, chemistry)

        # Terminal voltage: V = OCV - I*R0 - V1
        V_term = V_oc - I * self.R0 - self._V1

        return new_soc, float(V_term), float(V_oc)

    def reset(self):
        self._V1 = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  THERMAL MODEL  (lumped single-node)
# ═══════════════════════════════════════════════════════════════════════════

class ThermalModel:
    """
    Lumped thermal model for one cell.
        m·cp·dT/dt = Q_gen - h·A·(T - T_amb) + k_couple·(T_neighbour - T)
    """
    def __init__(self, m_kg: float = 0.065, cp_J_kgK: float = 1100.0,
                 hA_W_K: float = 0.13, T_amb: float = 25.0):
        self.m_cp    = m_kg * cp_J_kgK  # [J/K]
        self.hA      = hA_W_K           # convective loss [W/K]
        self.T_amb   = T_amb

    def heat_generation(self, I: float, V_term: float,
                         V_oc: float, R0: float) -> float:
        """
        Bernardi model: Q = I(V_oc - V_term) ≈ I²R0 + entropic
        """
        Q = I * (V_oc - V_term)   # [W]
        return max(Q, 0.0)

    def step(self, T: float, I: float, V_term: float, V_oc: float,
             R0: float, dt: float,
             T_neighbours: Optional[List[float]] = None,
             k_couple: float = 0.1) -> float:
        """
        Advance temperature by dt seconds.
        T_neighbours: temperatures of thermally coupled adjacent cells.
        """
        Q_gen   = self.heat_generation(I, V_term, V_oc, R0)
        Q_conv  = self.hA * (T - self.T_amb)

        Q_couple = 0.0
        if T_neighbours:
            for T_n in T_neighbours:
                Q_couple += k_couple * (T_n - T)

        dT_dt = (Q_gen - Q_conv + Q_couple) / self.m_cp
        return T + dT_dt * dt


# ═══════════════════════════════════════════════════════════════════════════
#  AGING COST MODEL
# ═══════════════════════════════════════════════════════════════════════════

class AgingCostModel:
    """
    Cycle-level aging cost based on empirical stress factors.
    Combines: DOD stress, temperature stress, C-rate stress.

    Reference: Wang et al. (2014) semi-empirical model adapted for
               multi-chemistry packs.
    """
    # Chemistry-specific base degradation rates [%SOH per cycle]
    BASE_RATE = {"LFP": 0.010, "NMC": 0.015, "LCO": 0.020}

    def __init__(self, chemistry: str = "LFP", T_ref: float = 25.0):
        self.alpha_base = self.BASE_RATE.get(chemistry, 0.015) / 100.0
        self.T_ref = T_ref

    def stress_dod(self, dod: float) -> float:
        """DOD stress factor: higher DOD → more degradation."""
        return 1.0 + 1.5 * dod**2

    def stress_temperature(self, T_C: float) -> float:
        """Arrhenius temperature stress (simplified)."""
        Ea_R = 4500.0   # Ea/R [K]
        T_K  = T_C + 273.15
        T0_K = self.T_ref + 273.15
        return math.exp(Ea_R * (1.0/T0_K - 1.0/T_K))

    def stress_crate(self, c_rate: float) -> float:
        """C-rate stress: fast charging accelerates SEI growth."""
        return 1.0 + 0.4 * max(c_rate - 1.0, 0.0)**1.5

    def cycle_cost(self, dod: float, T_C: float,
                   c_rate: float) -> float:
        """
        Estimated SOH loss for one cycle.
        Returns fraction (e.g., 0.0002 = 0.02% SOH loss).
        """
        cost = (self.alpha_base *
                self.stress_dod(dod) *
                self.stress_temperature(T_C) *
                self.stress_crate(c_rate))
        return float(cost)


# ═══════════════════════════════════════════════════════════════════════════
#  BATTERY PACK GRAPH
# ═══════════════════════════════════════════════════════════════════════════

class BatteryPackGraph:
    """
    Represents a battery pack as a dynamic graph.

    Topology: cells arranged in a 1D string (series connection).
    Edges:
      - Electrical edges: all cells (series → same current)
      - Thermal edges: adjacent cells (nearest-neighbour coupling)

    Node features  : [SOC, T_norm, SOH, V_term, R0, |I|, V_oc]  (7-dim)
    Edge features  : [ΔT, ΔSOC, d_ij]                            (3-dim)
    """
    N_NODE_FEAT = 7
    N_EDGE_FEAT = 3

    def __init__(self, cell_states: List[CellState],
                 topology: str = "series_1d",
                 T_amb: float = 25.0,
                 k_thermal_couple: float = 0.15):
        self.cells   = cell_states
        self.n_cells = len(cell_states)
        self.topology = topology
        self.T_amb   = T_amb
        self.k_couple = k_thermal_couple

        # Build sub-models for each cell
        self.ecm_models     = []
        self.thermal_models = []
        self.aging_models   = []
        for c in self.cells:
            self.ecm_models.append(
                ECM1RC(c.R0, c.R1, c.C1, c.Q_nom_Ah)
            )
            self.thermal_models.append(
                ThermalModel(T_amb=T_amb)
            )
            self.aging_models.append(
                AgingCostModel(chemistry=c.chemistry)
            )

        # Build edge list (bidirectional thermal + electrical)
        self.edge_index = self._build_edges()

    def _build_edges(self) -> np.ndarray:
        """
        Build edge_index [2, E]:
          - Thermal: i↔i+1 for all adjacent cells
          - Self-loops suppressed
        Returns shape (2, E).
        """
        src, dst = [], []
        for i in range(self.n_cells - 1):
            src += [i, i+1]
            dst += [i+1, i]
        return np.array([src, dst], dtype=np.int64)

    def node_features(self) -> np.ndarray:
        """Stack all cell feature vectors → (n_cells, N_NODE_FEAT)."""
        return np.stack([c.feature_vector() for c in self.cells], axis=0)

    def edge_features(self) -> np.ndarray:
        """
        Edge features: [ΔT, ΔSOC, 1/d] for each edge.
        d = distance between cells (index difference).
        """
        E = self.edge_index.shape[1]
        feats = np.zeros((E, self.N_EDGE_FEAT), dtype=np.float32)
        for e, (i, j) in enumerate(zip(self.edge_index[0], self.edge_index[1])):
            ci, cj = self.cells[i], self.cells[j]
            feats[e, 0] = (ci.T_C - cj.T_C) / 10.0          # ΔT normalised
            feats[e, 1] = ci.SOC - cj.SOC                     # ΔSOC
            feats[e, 2] = 1.0 / max(abs(int(i) - int(j)), 1) # proximity
        return feats

    def step(self, currents: np.ndarray, dt: float = 1.0) -> dict:
        """
        Advance pack state by dt seconds given current array [A] (n_cells,).
        Returns dict with step metrics.
        """
        assert len(currents) == self.n_cells

        new_socs, new_Ts, V_terms = [], [], []
        aging_costs = []
        violations  = []

        # Collect neighbour temperatures for coupling
        T_prev = [c.T_C for c in self.cells]

        for idx, (cell, ecm, therm, aging, I) in enumerate(zip(
                self.cells, self.ecm_models, self.thermal_models,
                self.aging_models, currents)):

            # ECM step
            new_soc, V_t, V_oc = ecm.step(I, dt, cell.SOC, cell.chemistry)

            # Neighbours for thermal coupling
            neigh_T = []
            if idx > 0:
                neigh_T.append(T_prev[idx - 1])
            if idx < self.n_cells - 1:
                neigh_T.append(T_prev[idx + 1])

            # Thermal step
            new_T = therm.step(cell.T_C, I, V_t, V_oc, cell.R0, dt,
                               T_neighbours=neigh_T, k_couple=self.k_couple)

            # Update cell state
            cell.SOC    = new_soc
            cell.T_C    = new_T
            cell.V_term = V_t
            cell.V_oc   = V_oc
            cell.I_A    = I

            # Aging cost (per-step fraction; accumulate over cycle)
            c_rate = abs(I) / max(cell.Q_nom_Ah, 0.1)
            dod    = 1.0 - new_soc
            ac     = aging.cycle_cost(dod, new_T, c_rate) * dt / 3600.0
            aging_costs.append(ac)

            new_socs.append(new_soc)
            new_Ts.append(new_T)
            V_terms.append(V_t)
            violations.append(not cell.is_safe)

        # Pack-level metrics
        soc_arr = np.array(new_socs)
        T_arr   = np.array(new_Ts)

        return {
            "SOC":              soc_arr,
            "T_C":              T_arr,
            "V_term":           np.array(V_terms),
            "SOC_mean":         float(np.mean(soc_arr)),
            "SOC_imbalance":    float(np.std(soc_arr)),       # σ_SOC
            "T_max":            float(np.max(T_arr)),
            "T_mean":           float(np.mean(T_arr)),
            "T_gradient":       float(np.max(T_arr) - np.min(T_arr)),
            "aging_cost":       float(np.sum(aging_costs)),
            "n_violations":     int(np.sum(violations)),
            "any_violation":    bool(np.any(violations)),
        }

    def reset_to(self, soc_init: float = 0.2, T_init: float = 25.0,
                 soc_noise: float = 0.02):
        """Reset pack to initial state with optional SOC imbalance noise."""
        rng = np.random.default_rng()
        for i, (cell, ecm) in enumerate(zip(self.cells, self.ecm_models)):
            cell.SOC = float(np.clip(
                soc_init + rng.normal(0, soc_noise), 0.05, 0.95
            ))
            cell.T_C = T_init + rng.normal(0, 0.5)
            ecm.reset()

    def soc_imbalance(self) -> float:
        """Standard deviation of SOC across pack."""
        return float(np.std([c.SOC for c in self.cells]))

    def pack_voltage(self) -> float:
        """Series pack voltage = sum of terminal voltages."""
        return sum(c.V_term for c in self.cells)

    def to_torch_graph(self) -> dict:
        """
        Convert current pack state to PyTorch tensors for GNN.
        Returns dict: {x, edge_index, edge_attr}
        """
        x          = torch.tensor(self.node_features(),   dtype=torch.float32)
        edge_index = torch.tensor(self.edge_index,        dtype=torch.long)
        edge_attr  = torch.tensor(self.edge_features(),   dtype=torch.float32)
        return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}


# ═══════════════════════════════════════════════════════════════════════════
#  GRAPH NEURAL NETWORK  (GNN Message Passing)
# ═══════════════════════════════════════════════════════════════════════════

class EdgeConv(nn.Module):
    """
    Edge-conditioned message passing layer.
    m_ij = MLP([h_i || h_j || e_ij])
    h_i' = AGG({m_ij}) + h_i   (residual)
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        x          : (N, node_dim)
        edge_index : (2, E)
        edge_attr  : (E, edge_dim)
        """
        src, dst = edge_index[0], edge_index[1]
        # Build messages
        msg_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)   # (E, node_dim)

        # Aggregate (mean) messages per destination node
        agg = torch.zeros_like(x)
        count = torch.zeros(x.shape[0], 1, device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        count.scatter_add_(0, dst.unsqueeze(-1),
                           torch.ones(msg.shape[0], 1, device=x.device))
        count = count.clamp(min=1)
        agg = agg / count

        # Residual + norm
        return self.norm(x + agg)


class PackGNN(nn.Module):
    """
    GNN for battery pack state prediction.

    Input : pack graph state {x, edge_index, edge_attr}
    Output: predicted next-step node features (SOC, ΔT, aging_rate)
            + pack-level imbalance score

    Architecture:
      Input projection → 3× EdgeConv → Output heads
    """
    def __init__(self, node_feat: int = 7, edge_feat: int = 3,
                 hidden: int = 64, n_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(node_feat, hidden)

        self.convs = nn.ModuleList([
            EdgeConv(hidden, edge_feat, hidden * 2)
            for _ in range(n_layers)
        ])

        # Node-level prediction heads
        self.soc_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid()     # SOC ∈ [0,1]
        )
        self.temp_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(),
            nn.Linear(32, 1)                    # ΔT [°C]
        )
        self.aging_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Softplus()     # aging ≥ 0
        )

        # Pack-level head (global mean pooling → imbalance score)
        self.pack_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid()      # imbalance score [0,1]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> dict:
        """
        Returns:
            soc_pred    : (N, 1)
            delta_T_pred: (N, 1)
            aging_pred  : (N, 1)
            imbalance   : (1,)  pack-level imbalance score
        """
        h = self.input_proj(x)                  # (N, hidden)

        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)   # (N, hidden)

        soc_pred    = self.soc_head(h)           # (N, 1)
        delta_T_pred= self.temp_head(h)          # (N, 1)
        aging_pred  = self.aging_head(h)         # (N, 1)
        imbalance   = self.pack_head(h.mean(0, keepdim=True))  # (1, 1)

        return {
            "soc_pred":     soc_pred,
            "delta_T_pred": delta_T_pred,
            "aging_pred":   aging_pred,
            "imbalance":    imbalance.squeeze(),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  PACK FACTORY  (build a pack from ECM parquet + chemistry)
# ═══════════════════════════════════════════════════════════════════════════

def build_pack_from_ecm(ecm_parquet=None, n_cells: int = 12,
                         chemistry: str = "LFP",
                         T_amb: float = 25.0,
                         soc_init: float = 0.2,
                         soc_noise: float = 0.03,
                         ecm_df=None) -> BatteryPackGraph:
    """
    Build a BatteryPackGraph by sampling real ECM parameters from
    the extracted parquet file.

    n_cells  : number of cells in the pack (series string)
    chemistry: filter rows by dataset matching chemistry
    """
    try:
        import pandas as pd
        if ecm_df is not None:
            df = ecm_df
        elif ecm_parquet is not None:
            df = pd.read_parquet(ecm_parquet)
        else:
            df = None

        # Filter by chemistry / dataset
        chem_to_ds = {"LFP": "MATR", "NMC": "RWTH", "LCO": "CALCE"}
        ds_name = chem_to_ds.get(chemistry, "MATR")
        sub = df[df["dataset"] == ds_name].dropna(subset=["IR_ohm"])

        if len(sub) == 0:
            sub = df.dropna(subset=["IR_ohm"])

        # Sample n_cells rows (with replacement if needed)
        sampled = sub.sample(n=n_cells, replace=len(sub) < n_cells,
                             random_state=42)

    except Exception:
        sampled = None

    rng = np.random.default_rng(42)
    cells = []
    for i in range(n_cells):
        # ECM parameters — from data or chemistry defaults
        if sampled is not None and i < len(sampled):
            row = sampled.iloc[i]
            R0  = float(row.get("IR_ohm", 0.05))
            SOH = float(row.get("SOH", 0.95))
            # NaN veya geçersiz değerleri düzelt (NaN truthy olduğu için or çalışmaz)
            if np.isnan(R0)  or R0  <= 0 or R0  > 1.0: R0  = 0.05
            if np.isnan(SOH) or SOH <= 0 or SOH > 1.2:  SOH = 0.92
        else:
            R0  = 0.03 + rng.uniform(-0.01, 0.01)
            SOH = 0.90 + rng.uniform(0, 0.10)

        R0  = np.clip(R0,  0.005, 0.5)
        SOH = np.clip(SOH, 0.70,  1.0)

        # Chemistry-based defaults
        defaults = {
            "LFP": {"Q": 1.1, "R1": 0.02, "C1": 1500.0, "V_oc": 3.3},
            "NMC": {"Q": 3.0, "R1": 0.03, "C1": 1000.0, "V_oc": 3.7},
            "LCO": {"Q": 1.5, "R1": 0.025,"C1": 1200.0, "V_oc": 3.8},
        }.get(chemistry, {"Q": 1.1, "R1": 0.02, "C1": 1500.0, "V_oc": 3.3})

        SOC_i = float(np.clip(
            soc_init + rng.normal(0, soc_noise), 0.05, 0.95
        ))
        T_i   = T_amb + rng.normal(0, 1.0)

        cell = CellState(
            cell_id   = f"cell_{i:02d}",
            SOC       = SOC_i,
            T_C       = T_i,
            SOH       = SOH,
            V_oc      = defaults["V_oc"],
            V_term    = defaults["V_oc"] - SOC_i * 0.1,
            R0        = R0,
            R1        = defaults["R1"],
            C1        = defaults["C1"],
            I_A       = 0.0,
            Q_nom_Ah  = defaults["Q"] * SOH,
            chemistry = chemistry,
        )
        cells.append(cell)

    return BatteryPackGraph(cells, T_amb=T_amb)


# ═══════════════════════════════════════════════════════════════════════════
#  QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  BatteryPackGraph — Self-Test")
    print("=" * 60)

    # Build a 12-cell LFP pack without parquet
    rng = np.random.default_rng(0)
    cells = []
    for i in range(12):
        cells.append(CellState(
            cell_id=f"cell_{i:02d}",
            SOC=0.20 + rng.uniform(-0.03, 0.03),
            T_C=25.0 + rng.uniform(-1, 1),
            SOH=0.92 + rng.uniform(-0.05, 0.05),
            V_oc=3.3, V_term=3.25,
            R0=0.035 + rng.uniform(-0.005, 0.005),
            R1=0.020, C1=1500.0,
            I_A=0.0, Q_nom_Ah=1.0,
            chemistry="LFP",
        ))

    pack = BatteryPackGraph(cells, T_amb=25.0)

    print(f"  Pack: {pack.n_cells} cells | "
          f"Initial SOC imbalance σ={pack.soc_imbalance():.4f}")
    print(f"  Edge index shape: {pack.edge_index.shape}")
    print(f"  Node feature dim: {BatteryPackGraph.N_NODE_FEAT}")

    # Simulate 10 steps at 1C charge
    I_charge = np.full(12, 1.1)   # 1C ≈ 1.1A
    for step in range(10):
        metrics = pack.step(I_charge, dt=60.0)  # 60s per step

    print(f"\n  After 10 min charge at 1C:")
    print(f"    SOC mean   = {metrics['SOC_mean']:.4f}")
    print(f"    SOC σ      = {metrics['SOC_imbalance']:.4f}")
    print(f"    T max      = {metrics['T_max']:.2f} °C")
    print(f"    ΔT         = {metrics['T_gradient']:.2f} °C")
    print(f"    Aging cost = {metrics['aging_cost']:.6f}")
    print(f"    Violations = {metrics['n_violations']}")

    # GNN forward pass test
    g = pack.to_torch_graph()
    model = PackGNN(node_feat=7, edge_feat=3, hidden=64, n_layers=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  PackGNN params: {n_params:,}")
    with torch.no_grad():
        out = model(g["x"], g["edge_index"], g["edge_attr"])
    print(f"  SOC pred shape   : {out['soc_pred'].shape}")
    print(f"  ΔT pred shape    : {out['delta_T_pred'].shape}")
    print(f"  Imbalance score  : {out['imbalance'].item():.4f}")
    print("\n  ✅ Self-test passed!")
    print("=" * 60)
