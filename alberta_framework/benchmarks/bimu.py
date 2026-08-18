"""Equation-level BiMU primitives for a separate binary/Bayesian lane."""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
from jax import Array

BIMU_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.bimu.protocol.v1",
        "paper_revision": "arXiv:2605.30198v1",
        "paper_revision_date": "2026-05-28",
        "lane": "binary_bayesian_permuted_mnist",
        "weight_domain": (-1, 1),
        "primary_metric": "mean_test_accuracy_over_last_5_tasks",
        "whole_stream_online_accuracy_is_separate": True,
        "paper_axes": {"tasks": 1000, "hidden_units": 100, "batch_size": 1},
        "adaptation_difference": "ASI implementation exposes equation primitives; no run is frozen",
        "learner_observes_task_boundary": False,
        "matched_axes": ("seed", "updates", "observations", "label_queries"),
        "persistent_bytes_accounting_required": True,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)


def _finite_vector(value: object, *, name: str) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 1 or array.size < 1 or not jnp.issubdtype(array.dtype, jnp.floating):
        raise ValueError(f"{name} must be a non-empty floating vector")
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return array


def posterior_probability(natural_parameter: object) -> Array:
    """Return ``P(weight=+1) = sigmoid(2 lambda)`` (paper equation 2)."""
    state = _finite_vector(natural_parameter, name="natural_parameter")
    return jnp.asarray(1.0 / (1.0 + jnp.exp(-2.0 * state)), dtype=state.dtype)


def bimu_update(
    natural_parameter: object,
    loss_gradient: object,
    prior_natural_parameter: object,
    *,
    memory_window: int | None,
    alpha_max: float,
) -> Array:
    """Apply BiMU equations 6--7 to one flat natural-parameter vector.

    ``memory_window=None`` is the predeclared mechanism-off reduction: it
    removes controlled forgetting while retaining the bounded metaplastic step.
    """
    state = _finite_vector(natural_parameter, name="natural_parameter")
    gradient = _finite_vector(loss_gradient, name="loss_gradient")
    prior = _finite_vector(prior_natural_parameter, name="prior_natural_parameter")
    if state.shape != gradient.shape or state.shape != prior.shape:
        raise ValueError("state, gradient, and prior must have identical shapes")
    if type(alpha_max) is not float or not math.isfinite(alpha_max) or alpha_max <= 0.0:
        raise ValueError("alpha_max must be a finite positive float")
    if memory_window is not None and (type(memory_window) is not int or memory_window < 1):
        raise ValueError("memory_window must be None or a positive integer")
    uncertainty = 1.0 / jnp.cosh(state) ** 2
    reciprocal_eta = (
        uncertainty
        + 2.0 * jnp.tanh(state) * gradient
        + 1.0 / alpha_max
        + 2.0 * jnp.abs(gradient)
    )
    eta = 1.0 / reciprocal_eta
    forgetting = 0.0 if memory_window is None else (state - prior) * uncertainty / memory_window
    return state - eta * (gradient + forgetting)


def late_window_mean(task_accuracies: Sequence[float], *, window: int = 5) -> float:
    """Compute BiMU's late-task metric without conflating whole-stream accuracy."""
    if type(window) is not int or window < 1:
        raise ValueError("window must be a positive integer")
    values = np.asarray(task_accuracies)
    if values.ndim != 1 or values.size < window or values.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("task_accuracies must be a numeric vector at least window long")
    resolved = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(resolved)) or np.any((resolved < 0.0) | (resolved > 1.0)):
        raise ValueError("task_accuracies must be finite and in [0, 1]")
    return float(np.mean(resolved[-window:]))
