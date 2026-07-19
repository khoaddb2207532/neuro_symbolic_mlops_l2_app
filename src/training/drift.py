"""Fidelity drift policy used to decide whether stages 2--4 must be rebuilt."""
from __future__ import annotations

from typing import Dict

import numpy as np


def weighted_fidelity(reference_predictions, surrogate_predictions) -> float:
    reference = np.asarray(reference_predictions)
    surrogate = np.asarray(surrogate_predictions)
    if reference.shape != surrogate.shape or reference.size == 0:
        raise ValueError("prediction arrays must be non-empty and have identical shapes")
    return float(np.mean(reference == surrogate))


def drift_decision(current_fidelity: float, fitted_fidelity: float, relative_drop: float) -> Dict[str, float | bool]:
    if fitted_fidelity <= 0:
        raise ValueError("fitted_fidelity must be positive")
    drop = max(0.0, (fitted_fidelity - current_fidelity) / fitted_fidelity)
    return {"current_fidelity": current_fidelity, "fitted_fidelity": fitted_fidelity,
            "relative_drop": drop, "threshold": relative_drop, "refit_required": drop > relative_drop}


def select_relative_drift_threshold(fidelities, fitted_fidelity: float,
                                    minimum: float = 0.10, maximum: float = 0.15) -> float:
    """Separate ordinary checkpoint noise from an abnormal relative fidelity drop."""
    values = np.asarray(list(fidelities), dtype=float)
    if values.size < 2:
        return minimum
    relative_steps = np.abs(np.diff(values)) / max(fitted_fidelity, 1e-12)
    median = float(np.median(relative_steps))
    mad = float(np.median(np.abs(relative_steps - median)))
    noise_ceiling = median + 3.0 * 1.4826 * mad
    return float(np.clip(max(minimum, noise_ceiling), minimum, maximum))
