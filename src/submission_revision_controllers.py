"""
Submission-revision controllers.

Additive implementation used to validate corrected baselines before replacing
the legacy manuscript controllers.
"""

from copy import deepcopy
from pathlib import Path
import pickle
import numpy as np


def project_current_vector(currents, pack, cfg):
    """
    Project a per-cell current vector onto the common actuator feasible set.

    Per-cell:
        0 <= I_i <= C_max * Q_i

    Pack:
        sum(I_i) <= C_pack,max * sum(Q_i)
    """
    currents = np.asarray(currents, dtype=np.float64).copy()

    q_nom = np.asarray(
        [float(c.Q_nom_Ah) for c in pack.cells],
        dtype=np.float64,
    )

    per_cell_max = float(cfg["I_max_C"]) * q_nom

    currents = np.clip(
        currents,
        0.0,
        per_cell_max,
    )

    pack_max = float(cfg["I_pack_max_C"]) * float(q_nom.sum())
    total = float(currents.sum())

    if total > pack_max and total > 0:
        currents *= pack_max / total

    return currents.astype(np.float32)


def uniform_cc_current(pack, cfg):
    """
    Largest uniform current allowed by BOTH the weakest-cell current limit
    and the aggregate pack-current limit.
    """
    q_nom = np.asarray(
        [float(c.Q_nom_Ah) for c in pack.cells],
        dtype=np.float64,
    )

    per_cell_caps = float(cfg["I_max_C"]) * q_nom
    pack_cap = float(cfg["I_pack_max_C"]) * float(q_nom.sum())

    uniform_from_pack = pack_cap / len(q_nom)

    return float(min(
        float(per_cell_caps.min()),
        uniform_from_pack,
    ))


class VoltageFeedbackCCCVController:
    """
    True voltage-feedback uniform-current CC-CV baseline.

    CC:
        Apply the largest uniform current allowed by the shared actuator
        constraints.

    CV:
        If the next physics step at CC current would exceed V_ref, solve for
        the largest uniform current whose predicted next-step maximum cell
        voltage remains <= V_ref.

    This is intentionally a uniform pack-level baseline. It does NOT perform
    cell-level balancing.
    """

    def __init__(
        self,
        cfg,
        voltage_tolerance=1e-5,
        binary_search_iterations=32,
    ):
        self.cfg = dict(cfg)
        self.voltage_tolerance = float(voltage_tolerance)
        self.binary_search_iterations = int(binary_search_iterations)

        self.cv_mode = False
        self.last_phase = "CC"
        self.last_current_A = 0.0
        self.last_predicted_max_voltage = None

    def reset(self):
        self.cv_mode = False
        self.last_phase = "CC"
        self.last_current_A = 0.0
        self.last_predicted_max_voltage = None

    @property
    def dt(self):
        return float(self.cfg.get("dt_s", 60.0))

    @property
    def voltage_limit(self):
        return float(self.cfg["V_max"])

    def _predict_uniform_step(self, pack, current_A):
        """
        One-step ECM/thermal prediction from the CURRENT pack state.

        deepcopy ensures the candidate check does not alter the real plant.
        """
        candidate_pack = deepcopy(pack)

        currents = np.full(
            candidate_pack.n_cells,
            float(current_A),
            dtype=np.float32,
        )

        currents = project_current_vector(
            currents,
            candidate_pack,
            self.cfg,
        )

        candidate_pack.step(currents, dt=self.dt)

        vmax = max(float(c.V_term) for c in candidate_pack.cells)
        tmax = max(float(c.T_C) for c in candidate_pack.cells)

        return vmax, tmax

    def _solve_cv_current(self, pack, upper_current):
        """
        Binary search for the largest uniform current satisfying:

            max_i V_i(t + dt) <= V_ref

        under the existing ECM plant.
        """
        v_ref = self.voltage_limit
        tol = self.voltage_tolerance

        v_zero, _ = self._predict_uniform_step(pack, 0.0)

        # Existing polarisation can temporarily leave voltage above V_ref.
        # In that case charging current must be zero.
        if v_zero > v_ref + tol:
            self.last_predicted_max_voltage = v_zero
            return 0.0

        lo = 0.0
        hi = float(max(upper_current, 0.0))

        v_hi, _ = self._predict_uniform_step(pack, hi)

        # Full CC current is already feasible.
        if v_hi <= v_ref + tol:
            self.last_predicted_max_voltage = v_hi
            return hi

        for _ in range(self.binary_search_iterations):
            mid = 0.5 * (lo + hi)

            v_mid, _ = self._predict_uniform_step(pack, mid)

            if v_mid <= v_ref:
                lo = mid
            else:
                hi = mid

        v_final, _ = self._predict_uniform_step(pack, lo)
        self.last_predicted_max_voltage = v_final

        return float(lo)

    def get_currents(self, pack):
        n = pack.n_cells

        I_cc = uniform_cc_current(pack, self.cfg)

        predicted_v_cc, _ = self._predict_uniform_step(
            pack,
            I_cc,
        )

        # Enter CV BEFORE applying an action that would cross V_ref.
        if predicted_v_cc > self.voltage_limit + self.voltage_tolerance:
            self.cv_mode = True

        if self.cv_mode:
            I = self._solve_cv_current(pack, I_cc)
            self.last_phase = "CV"
        else:
            I = I_cc
            self.last_predicted_max_voltage = predicted_v_cc
            self.last_phase = "CC"

        currents = np.full(
            n,
            I,
            dtype=np.float32,
        )

        currents = project_current_vector(
            currents,
            pack,
            self.cfg,
        )

        self.last_current_A = float(currents[0])

        return currents


# ---------------------------------------------------------------------
# Dataset-derived voltage limits
# ---------------------------------------------------------------------

_CHEMISTRY_DATASET = {
    "LFP": "MATR",
    "NMC": "RWTH",
    "LCO": "CALCE",
}


def resolve_dataset_voltage_limits(chemistry, repo_root=None):
    """
    Resolve voltage limits from BatteryML-processed dataset metadata.

    No silent chemistry fallback is allowed for canonical submission runs.
    This prevents a nominal 4.20-V default from being applied to datasets
    whose documented charge protocol uses another upper cutoff.
    """
    chemistry = str(chemistry).upper()

    if chemistry not in _CHEMISTRY_DATASET:
        raise ValueError(
            f"Unsupported chemistry for canonical voltage-limit resolution: "
            f"{chemistry}"
        )

    dataset = _CHEMISTRY_DATASET[chemistry]

    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    else:
        repo_root = Path(repo_root)

    processed = repo_root / "data" / "processed" / "BatteryML" / dataset

    if not processed.exists():
        raise FileNotFoundError(
            f"Processed BatteryML dataset not found for {chemistry}: "
            f"{processed}. Process {dataset} before running canonical "
            f"{chemistry} experiments."
        )

    files = sorted(processed.glob("*.pkl"))

    if not files:
        raise FileNotFoundError(
            f"No BatteryML PKL files found in {processed}"
        )

    # MATR has four batches; sample one known cell from each where available.
    if dataset == "MATR":
        preferred = [
            processed / "MATR_b1c0.pkl",
            processed / "MATR_b2c0.pkl",
            processed / "MATR_b3c0.pkl",
            processed / "MATR_b4c0.pkl",
        ]
        sample_files = [p for p in preferred if p.exists()]
    else:
        # For other processed datasets, several deterministic representatives
        # are enough because these are metadata fields rather than timeseries.
        sample_files = files[:min(8, len(files))]

    if not sample_files:
        sample_files = files[:1]

    vmins = []
    vmaxs = []

    for p in sample_files:
        with open(p, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, dict):
            continue

        vmin = obj.get("min_voltage_limit_in_V")
        vmax = obj.get("max_voltage_limit_in_V")

        if vmin is not None:
            vmins.append(float(vmin))
        if vmax is not None:
            vmaxs.append(float(vmax))

    if not vmaxs:
        raise RuntimeError(
            f"No max_voltage_limit_in_V metadata found for {dataset}"
        )

    import numpy as _np

    vmax = float(_np.median(vmaxs))
    vmin = float(_np.median(vmins)) if vmins else 2.5

    return {
        "chemistry": chemistry,
        "dataset": dataset,
        "V_min": vmin,
        "V_max": vmax,
        "source": "BatteryML processed dataset metadata",
        "representative_files": [p.name for p in sample_files],
        "observed_Vmax": vmaxs,
        "observed_Vmin": vmins,
    }


def apply_dataset_voltage_limits(cfg, repo_root=None):
    """
    Return a copied config with dataset-derived V_min/V_max.
    """
    cfg = dict(cfg)

    limits = resolve_dataset_voltage_limits(
        cfg["chemistry"],
        repo_root=repo_root,
    )

    cfg["V_min"] = limits["V_min"]
    cfg["V_max"] = limits["V_max"]
    cfg["voltage_limit_dataset"] = limits["dataset"]
    cfg["voltage_limit_source"] = limits["source"]

    return cfg


class BalancedVoltageFeedbackCCCVController:
    """
    Active-balancing CC-CV baseline with voltage-feedback CV regulation.

    The balancing rule creates a feasible per-cell current vector.
    If its one-step physics prediction crosses V_ref, a scalar CV search
    finds the largest feasible multiple of that vector.
    """

    def __init__(
        self,
        cfg,
        voltage_tolerance=1e-5,
        binary_search_iterations=32,
    ):
        self.cfg = dict(cfg)
        self.voltage_tolerance = float(voltage_tolerance)
        self.binary_search_iterations = int(binary_search_iterations)
        self.cv_mode = False
        self.last_phase = "CC"
        self.last_predicted_max_voltage = None

    def reset(self):
        self.cv_mode = False
        self.last_phase = "CC"
        self.last_predicted_max_voltage = None

    def _predict(self, pack, currents):
        candidate = deepcopy(pack)
        currents = project_current_vector(
            currents,
            candidate,
            self.cfg,
        )
        candidate.step(
            currents,
            dt=float(self.cfg.get("dt_s", 60.0)),
        )
        return max(float(c.V_term) for c in candidate.cells)

    def _balanced_candidate(self, pack):
        q_nom = np.asarray(
            [float(c.Q_nom_Ah) for c in pack.cells],
            dtype=np.float64,
        )

        socs = np.asarray(
            [float(c.SOC) for c in pack.cells],
            dtype=np.float64,
        )

        mean_soc = float(socs.mean())

        per_cell_max = float(self.cfg["I_max_C"]) * q_nom

        # Preserve the released ±20% balancing rule.
        soc_deviation = socs - mean_soc
        balance_factor = (
            1.0
            - np.clip(
                soc_deviation * 2.0,
                -0.2,
                0.2,
            )
        )

        candidate = per_cell_max * balance_factor

        return project_current_vector(
            candidate,
            pack,
            self.cfg,
        )

    def get_currents(self, pack):
        candidate = self._balanced_candidate(pack)

        v_ref = float(self.cfg["V_max"])

        vmax_full = self._predict(
            pack,
            candidate,
        )

        if vmax_full > v_ref + self.voltage_tolerance:
            self.cv_mode = True

        if not self.cv_mode:
            self.last_phase = "CC"
            self.last_predicted_max_voltage = vmax_full
            return candidate.astype(np.float32)

        self.last_phase = "CV"

        v_zero = self._predict(
            pack,
            np.zeros_like(candidate),
        )

        if v_zero > v_ref + self.voltage_tolerance:
            self.last_predicted_max_voltage = v_zero
            return np.zeros_like(candidate, dtype=np.float32)

        lo = 0.0
        hi = 1.0

        for _ in range(self.binary_search_iterations):
            mid = 0.5 * (lo + hi)

            vmax_mid = self._predict(
                pack,
                candidate * mid,
            )

            if vmax_mid <= v_ref:
                lo = mid
            else:
                hi = mid

        final = project_current_vector(
            candidate * lo,
            pack,
            self.cfg,
        )

        self.last_predicted_max_voltage = self._predict(
            pack,
            final,
        )

        return final.astype(np.float32)
