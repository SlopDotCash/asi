"""Prospective optimization-readiness diagnostic for development evaluation.

The equations follow Wang et al., arXiv:2605.09044v1.  This module evaluates
already-collected mini-batch gradients; it neither trains nor reads benchmark
data, and therefore cannot itself create a performance result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

OPTIMIZATION_READINESS_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.optimization-readiness.protocol.v1",
        "paper_revision": "arXiv:2605.09044v1",
        "paper_revision_date": "2026-05-09",
        "diagnostics": (
            "optimization_readiness",
            "gradient_norm",
            "representation_energy_rank_0.99",
            "curvature_energy_rank_0.99",
            "parameter_norm",
        ),
        "target": "future_relative_loss_reduction_after_matched_updates",
        "matched_axes": ("seed", "updates", "observations", "mini_batch_size"),
        "persistent_bytes_accounting_required": True,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)


@dataclass(frozen=True)
class OptimizationReadiness:
    """Equation-level diagnostic values for one checkpoint/task pair."""

    gradient_squared_norm: float
    expected_batch_gradient_squared_norm: float
    gradient_strength: float
    gradient_reliability: float
    optimization_readiness: float
    gradient_norm: float
    batch_count: int
    parameter_count: int


def _finite_matrix(value: object, *, name: str) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must have a real numeric dtype")
    resolved = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(resolved)):
        raise ValueError(f"{name} must contain only finite values")
    return resolved


def estimate_optimization_readiness(
    *, loss: float, batch_gradients: object, include_reliability: bool = True
) -> OptimizationReadiness:
    """Estimate OR from independent mini-batch gradient vectors.

    The row mean estimates the population gradient.  The mean row squared norm
    estimates the expected squared mini-batch-gradient norm.  Setting
    ``include_reliability=False`` is the predeclared gradient-strength-only
    mechanism-off reduction.
    """
    if type(loss) is not float and type(loss) is not int:
        raise ValueError("loss must be a finite positive real number")
    resolved_loss = float(loss)
    if not math.isfinite(resolved_loss) or resolved_loss <= 0.0:
        raise ValueError("loss must be a finite positive real number")
    if type(include_reliability) is not bool:
        raise ValueError("include_reliability must be a bool")
    gradients = _finite_matrix(batch_gradients, name="batch_gradients")
    mean_gradient = np.mean(gradients, axis=0)
    gradient_squared_norm = float(np.dot(mean_gradient, mean_gradient))
    expected_squared_norm = float(np.mean(np.sum(np.square(gradients), axis=1)))
    strength = gradient_squared_norm / resolved_loss
    reliability = (
        gradient_squared_norm / expected_squared_norm if expected_squared_norm > 0.0 else 0.0
    )
    readiness = strength * reliability if include_reliability else strength
    return OptimizationReadiness(
        gradient_squared_norm=gradient_squared_norm,
        expected_batch_gradient_squared_norm=expected_squared_norm,
        gradient_strength=strength,
        gradient_reliability=reliability,
        optimization_readiness=readiness,
        gradient_norm=math.sqrt(gradient_squared_norm),
        batch_count=int(gradients.shape[0]),
        parameter_count=int(gradients.shape[1]),
    )


def energy_rank(matrix: object, *, threshold: float = 0.99) -> int:
    """Return the smallest singular-value count reaching squared-energy mass."""
    if type(threshold) is not float:
        raise ValueError("threshold must be a float in (0, 1]")
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be a float in (0, 1]")
    resolved = _finite_matrix(matrix, name="matrix")
    singular_values = np.linalg.svd(resolved, compute_uv=False)
    squared = np.square(singular_values)
    total = float(np.sum(squared))
    if total == 0.0:
        return 0
    return int(np.searchsorted(np.cumsum(squared) / total, threshold, side="left") + 1)
