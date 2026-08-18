"""Bounded, nonpromoting plasticity comparator primitives.

These kernels make the mechanisms in issues #1559--#1567 executable without
claiming that a development comparison has been run.  They share one matched
protocol/accounting record so a future IPMNIST or streaming-control runner can
compare arms without changing seeds, examples, updates, or information.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final, Literal, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array, core


@dataclasses.dataclass(frozen=True, slots=True)
class ComparatorProtocol:
    """Immutable development-only binding for one comparator mechanism."""

    name: str
    paper: str
    adaptation: str
    mechanism_off: str
    persistent_bytes: int
    environment_or_data_steps: int
    model_queries: int
    timing_telemetry_seconds: float
    matched_axes: tuple[str, ...] = ("seed", "updates", "observations", "example_order")
    development_only: bool = True
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        for field in ("name", "paper", "adaptation", "mechanism_off"):
            if type(getattr(self, field)) is not str or not getattr(self, field):
                raise ValueError(f"{field} must be an exact non-empty string")
        required_axes = ("seed", "updates", "observations", "example_order")
        if type(self.matched_axes) is not tuple or self.matched_axes != required_axes:
            raise ValueError(f"matched_axes must equal {required_axes!r}")
        for field in ("persistent_bytes", "environment_or_data_steps", "model_queries"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative exact integer")
        if (
            type(self.timing_telemetry_seconds) not in (int, float)
            or not math.isfinite(self.timing_telemetry_seconds)
            or self.timing_telemetry_seconds < 0
        ):
            raise ValueError("timing_telemetry_seconds must be finite and nonnegative")
        if self.development_only is not True or self.scientific_promotion_allowed is not False:
            raise ValueError("comparator protocols are permanently nonpromoting")


PAPER_REVISIONS: Final[dict[str, str]] = {
    "l2_er": "arXiv:2509.22335v3",
    "adamo": "arXiv:2606.09762v1",
    "intentional_updates": "arXiv:2604.19033v1",
    "growing_elastic": "arXiv:2608.01475v1",
    "utility_pull": "ASI protocol extension; no paper claimed",
    "nap": "arXiv:2407.01800v1",
    "c_chain": "arXiv:2506.00592v1",
    "smooth_leaky": "arXiv:2509.22562v4",
    "aid": "arXiv:2502.01342v2",
    "deep_fourier": "arXiv:2410.20634v1",
    "noise_curvature": "arXiv:2509.19698v3",
}


def protocol(
    name: str,
    *,
    persistent_bytes: int = 0,
    environment_or_data_steps: int = 0,
    model_queries: int = 0,
    timing_telemetry_seconds: float = 0.0,
) -> ComparatorProtocol:
    """Construct the predeclared matched protocol for a supported arm."""
    if type(name) is not str or name not in PAPER_REVISIONS:
        raise ValueError("unknown plasticity comparator")
    adaptations = {
        "l2_er": "streaming feature windows replace paper minibatches",
        "adamo": "dense matrices only; task moments exclude isometry gradient",
        "intentional_updates": "batch-size-one supervised and TD/control scalar kernels",
        "growing_elastic": "static maximum shape with an active-unit mask",
        "utility_pull": "utility-scaled interpolation toward retained initialization",
        "nap": "projection kernel; caller supplies normalized architecture",
        "c_chain": "caller supplies disjoint reference predictions",
        "smooth_leaky": "deterministic activation kernel",
        "aid": "Threefry-keyed simplified interval dropout",
        "deep_fourier": "concatenated sine/cosine at every selected layer",
        "noise_curvature": "caller supplies independently estimated layer indicators",
    }
    off = {
        "l2_er": "l2_strength=rank_strength=0",
        "adamo": "isometry_strength=0",
        "intentional_updates": "use caller fixed step size",
        "growing_elastic": "growth=pruning=0",
        "utility_pull": "strength=0",
        "nap": "projection_enabled=False",
        "c_chain": "strength=0",
        "smooth_leaky": "alpha=1",
        "aid": "relu_probability=1",
        "deep_fourier": "feature_enabled=False",
        "noise_curvature": "scheduler_enabled=False",
    }
    return ComparatorProtocol(
        name=name,
        paper=PAPER_REVISIONS[name],
        adaptation=adaptations[name],
        mechanism_off=off[name],
        persistent_bytes=persistent_bytes,
        environment_or_data_steps=environment_or_data_steps,
        model_queries=model_queries,
        timing_telemetry_seconds=timing_telemetry_seconds,
    )


def _finite_scalar(name: str, value: object, *, minimum: float = 0.0) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be an exact finite scalar >= {minimum}")
    scalar = cast(int | float, value)
    if not math.isfinite(scalar) or scalar < minimum:
        raise ValueError(f"{name} must be an exact finite scalar >= {minimum}")
    return float(scalar)


def _probability(name: str, value: object, *, include_one: bool = True) -> float:
    scalar = _finite_scalar(name, value)
    if scalar > 1.0 or (not include_one and scalar == 1.0):
        boundary = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError(f"{name} must lie in {boundary}")
    return scalar


def _array(name: str, value: object) -> Array:
    if type(value) is not np.ndarray and not isinstance(value, (Array, core.Tracer)):
        raise ValueError(f"{name} must be a NumPy or JAX array")
    return jnp.asarray(value)


def persistent_array_bytes(*arrays: Array | np.ndarray) -> int:
    """Return exact resident numeric payload bytes for comparator-owned arrays."""
    total = 0
    for value in arrays:
        array: Array | np.ndarray
        if type(value) is np.ndarray:
            array = value
        elif isinstance(value, Array):
            array = value
        else:
            raise ValueError("persistent comparator values must be arrays")
        if np.dtype(array.dtype).kind not in "biufc":
            raise ValueError("persistent comparator arrays must be numeric")
        total += int(array.nbytes)
    return total


def effective_rank(features: Array, *, epsilon: float = 1e-12) -> Array:
    """Roy--Vetterli effective rank of a stacked feature matrix."""
    matrix = _array("features", features)
    if matrix.ndim != 2:
        raise ValueError("features must be rank two")
    _finite_scalar("epsilon", epsilon, minimum=np.nextafter(0.0, 1.0))
    singular = jnp.linalg.svd(matrix, compute_uv=False)
    total = jnp.sum(singular)
    probabilities = jnp.where(total > 0, singular / total, 0.0)
    safe_probabilities = jnp.where(probabilities > epsilon, probabilities, 1.0)
    entropy = -jnp.sum(
        jnp.where(probabilities > epsilon, probabilities * jnp.log(safe_probabilities), 0)
    )
    return jnp.where(total > 0, jnp.exp(entropy), 0.0)


def l2_er_objective(
    task_loss: Array,
    parameters: tuple[Array, ...],
    feature_batches: tuple[Array, ...],
    *,
    l2_strength: float,
    rank_strength: float,
) -> Array:
    """Paper objective: task loss + L2 - mean effective rank."""
    if not parameters or not feature_batches:
        raise ValueError("parameters and feature_batches must be non-empty tuples")
    _finite_scalar("l2_strength", l2_strength)
    _finite_scalar("rank_strength", rank_strength)
    loss = jnp.asarray(task_loss)
    if loss.ndim != 0:
        raise ValueError("task_loss must be scalar")
    l2 = sum((jnp.sum(jnp.square(value)) for value in parameters), jnp.asarray(0.0))
    ranks = jnp.stack(tuple(effective_rank(value) for value in feature_batches))
    return loss + l2_strength * l2 - rank_strength * jnp.mean(ranks)


def isometry_gradient(weights: Array) -> Array:
    """Gradient of the rectangular Gram-deviation penalty (paper Eq. 16)."""
    weights = _array("weights", weights)
    if weights.ndim != 2:
        raise ValueError("weights must be rank two")
    rows, columns = weights.shape
    if rows >= columns:
        gram_error = weights.T @ weights - jnp.eye(columns, dtype=weights.dtype)
        return 4.0 * weights @ gram_error
    gram_error = weights @ weights.T - jnp.eye(rows, dtype=weights.dtype)
    return 4.0 * gram_error @ weights


def isometry_penalty(weights: Array) -> Array:
    """Squared rectangular Gram deviation used by AdamO Eq. 16."""
    weights = _array("weights", weights)
    if weights.ndim != 2:
        raise ValueError("weights must be rank two")
    rows, columns = weights.shape
    gram = weights.T @ weights if rows >= columns else weights @ weights.T
    return jnp.sum(jnp.square(gram - jnp.eye(min(rows, columns), dtype=weights.dtype)))


def adamo_update(
    weights: Array,
    task_gradient: Array,
    first_moment: Array,
    second_moment: Array,
    *,
    step: int,
    learning_rate: float,
    isometry_strength: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[Array, Array, Array]:
    """AdamO Eq. 19--20; only task gradients enter Adam moments."""
    if type(step) is not int or step < 1:
        raise ValueError("step must be a positive exact integer")
    _finite_scalar("learning_rate", learning_rate)
    _finite_scalar("isometry_strength", isometry_strength)
    _probability("beta1", beta1, include_one=False)
    _probability("beta2", beta2, include_one=False)
    _finite_scalar("epsilon", epsilon, minimum=np.nextafter(0.0, 1.0))
    weights = _array("weights", weights)
    gradient = _array("task_gradient", task_gradient)
    first_moment = _array("first_moment", first_moment)
    second_moment = _array("second_moment", second_moment)
    if (
        weights.ndim != 2
        or gradient.shape != weights.shape
        or first_moment.shape != weights.shape
        or second_moment.shape != weights.shape
    ):
        raise ValueError("AdamO weights, gradients, and moments must be matching matrices")
    moment = beta1 * first_moment + (1.0 - beta1) * gradient
    variance = beta2 * second_moment + (1.0 - beta2) * jnp.square(gradient)
    corrected_moment = moment / (1.0 - beta1**step)
    corrected_variance = variance / (1.0 - beta2**step)
    task_delta = learning_rate * corrected_moment / (jnp.sqrt(corrected_variance) + epsilon)
    iso_delta = learning_rate * isometry_strength * isometry_gradient(weights)
    return weights - task_delta - iso_delta, moment, variance


def intentional_td_step_size(
    gradient: Array,
    *,
    intended_fraction: float,
    diagonal_scale: Array | None = None,
    epsilon: float = 1e-12,
) -> Array:
    """Intentional TD(0) Eq. 7, including optional diagonal scaling."""
    gradient = _array("gradient", gradient)
    scale = (
        jnp.ones_like(gradient)
        if diagonal_scale is None
        else _array("diagonal_scale", diagonal_scale)
    )
    if gradient.size == 0 or scale.shape != gradient.shape:
        raise ValueError("gradient and diagonal_scale must be matching non-empty arrays")
    _finite_scalar("intended_fraction", intended_fraction, minimum=np.nextafter(0.0, 1.0))
    _finite_scalar("epsilon", epsilon, minimum=np.nextafter(0.0, 1.0))
    denominator = jnp.vdot(gradient, scale * gradient).real
    return jnp.asarray(intended_fraction) / jnp.maximum(denominator, epsilon)


def intentional_trace_step_size(
    trace: Array,
    diagonal_scale: Array,
    *,
    intended_fraction: float,
    discounted_gradient_energy: Array,
    epsilon: float = 1e-12,
) -> Array:
    """Conservative Intentional TD(lambda) step size from paper Eq. 12."""
    trace = _array("trace", trace)
    diagonal_scale = _array("diagonal_scale", diagonal_scale)
    energy = _array("discounted_gradient_energy", discounted_gradient_energy)
    if trace.size == 0 or diagonal_scale.shape != trace.shape or energy.ndim != 0:
        raise ValueError("trace/scale must match and discounted_gradient_energy must be scalar")
    _finite_scalar("intended_fraction", intended_fraction, minimum=np.nextafter(0.0, 1.0))
    _finite_scalar("epsilon", epsilon, minimum=np.nextafter(0.0, 1.0))
    denominator = energy * jnp.vdot(
        trace, diagonal_scale * trace
    ).real
    return jnp.asarray(intended_fraction) / jnp.sqrt(jnp.maximum(denominator, epsilon))


def bounded_elastic_mask(
    active: Array | np.ndarray,
    utilities: Array | np.ndarray,
    *,
    grow: int,
    prune: int,
) -> Array:
    """Static-shape adaptive elastic transition under a hard peak-width bound."""
    if type(active) is not np.ndarray and not isinstance(active, Array):
        raise ValueError("active must be a NumPy or JAX array")
    if type(utilities) is not np.ndarray and not isinstance(utilities, Array):
        raise ValueError("utilities must be a NumPy or JAX array")
    active_np = np.asarray(active)
    utilities_np = np.asarray(utilities)
    if active_np.ndim != 1 or utilities_np.shape != active_np.shape:
        raise ValueError("active and utilities must be matching vectors")
    if active_np.dtype != np.bool_ or utilities_np.dtype.kind not in "iuf":
        raise ValueError("active must be bool and utilities must be real numeric")
    if not np.isfinite(utilities_np).all():
        raise ValueError("utilities must be finite")
    if type(grow) is not int or type(prune) is not int or grow < 0 or prune < 0:
        raise ValueError("grow and prune must be nonnegative exact integers")
    initially_inactive = np.flatnonzero(~active_np)
    active_indices = np.flatnonzero(active_np)
    if prune > active_indices.size or grow > initially_inactive.size:
        raise ValueError("growth and pruning must fit the active-mask capacity")
    result = active_np.copy()
    for index in active_indices[np.argsort(utilities_np[active_indices], kind="stable")[:prune]]:
        result[index] = False
    result[initially_inactive[:grow]] = True
    return jnp.asarray(result)


def utility_scaled_pull(
    weights: Array,
    initial_weights: Array,
    utilities: Array,
    *,
    strength: float,
    mode: Literal["utility", "utility_free", "l2_init", "hard_reset"] = "utility",
) -> Array:
    """Continuous partial reset and its utility-free/hard-reset reductions."""
    weights = _array("weights", weights)
    initial = _array("initial_weights", initial_weights)
    utility = _array("utilities", utilities)
    if weights.shape != initial.shape or utility.shape != weights.shape:
        raise ValueError("weights, initialization, and utilities must match")
    _probability("strength", strength)
    if strength == 0:
        return weights
    if mode == "utility":
        maximum = jnp.max(jnp.maximum(utility, 0.0))
        normalized = jnp.where(maximum > 0, jnp.maximum(utility, 0.0) / maximum, 0.0)
        rate = strength * (1.0 - normalized)
    elif mode in ("utility_free", "l2_init"):
        rate = jnp.full_like(weights, strength)
    elif mode == "hard_reset":
        rate = jnp.where(utility <= 0, 1.0, 0.0)
    else:
        raise ValueError("unknown utility pull mode")
    return weights + jnp.clip(rate, 0.0, 1.0) * (initial - weights)


def nap_project(weights: Array, *, initial_norm: float, enabled: bool = True) -> Array:
    """NaP weight projection onto the initialization Frobenius radius."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be an exact bool")
    _finite_scalar("initial_norm", initial_norm)
    weights = _array("weights", weights)
    if not enabled:
        return weights
    norm = jnp.linalg.norm(weights)
    return jnp.where(norm > 0, weights * (initial_norm / norm), weights)


def churn_loss(before: Array, after: Array, *, strength: float) -> Array:
    """C-CHAIN Eq. 8 on a caller-owned disjoint reference batch."""
    before = _array("before", before)
    after = _array("after", after)
    if before.shape != after.shape:
        raise ValueError("reference predictions must have matching shapes")
    if before.size == 0:
        raise ValueError("reference predictions must be non-empty")
    _finite_scalar("strength", strength)
    return 0.5 * strength * jnp.mean(jnp.square(after - before))


def ntk_threshold_rank(jacobian: Array, *, threshold: float = 0.99) -> Array:
    """Number of empirical-NTK singular values carrying a target energy fraction."""
    jacobian = _array("jacobian", jacobian)
    if jacobian.ndim != 2 or not jacobian.size:
        raise ValueError("jacobian must be a non-empty matrix")
    target = _probability("threshold", threshold)
    if target == 0.0:
        raise ValueError("threshold must lie in (0, 1]")
    singular = jnp.linalg.svd(jacobian @ jacobian.T, compute_uv=False)
    total = jnp.sum(singular)
    cumulative = jnp.cumsum(singular)
    rank = jnp.searchsorted(cumulative, target * total, side="left") + 1
    return jnp.where(total > 0, rank, 0)


def smooth_leaky(value: Array, *, alpha: float, power: float, curvature: float) -> Array:
    """Smooth-Leaky activation from arXiv:2509.22562 Eq. 1."""
    value = _array("value", value)
    _probability("alpha", alpha)
    _finite_scalar("power", power, minimum=np.nextafter(0.0, 1.0))
    _finite_scalar("curvature", curvature)
    return alpha * value + (1.0 - alpha) * value * jax_sigmoid(curvature * value / power)


def jax_sigmoid(value: Array) -> Array:
    """Stable sigmoid without importing a larger neural-network facade."""
    return jnp.where(
        value >= 0,
        1.0 / (1.0 + jnp.exp(-value)),
        jnp.exp(value) / (1 + jnp.exp(value)),
    )


def interval_dropout(
    value: Array,
    key: Array,
    *,
    relu_probability: float,
    training: bool = True,
) -> Array:
    """Simplified AID Algorithm 2, with its deterministic evaluation rule."""
    probability = _probability("relu_probability", relu_probability)
    if probability < 0.5:
        raise ValueError("relu_probability must lie in [0.5, 1]")
    if type(training) is not bool:
        raise ValueError("training must be an exact bool")
    value = _array("value", value)
    if not training:
        return jnp.where(value >= 0, probability * value, (1.0 - probability) * value)
    use_relu = jr.bernoulli(key, probability, value.shape)
    return jnp.where(use_relu, jnp.maximum(value, 0), jnp.minimum(value, 0))


def deep_fourier_features(value: Array, *, enabled: bool = True) -> Array:
    """Deep Fourier feature map [sin(z), cos(z)]."""
    value = _array("value", value)
    if type(enabled) is not bool:
        raise ValueError("enabled must be an exact bool")
    if value.ndim == 0:
        raise ValueError("Deep Fourier features require a non-scalar input")
    return jnp.concatenate((jnp.sin(value), jnp.cos(value)), axis=-1) if enabled else value


def noise_curvature_critical_step_size(
    *,
    batch_size: int,
    squared_gradient_mean: float,
    per_sample_gradient_variance: float,
    normalized_curvature_variance: float,
    volatility_inflation: float = 1.0,
    safety_margin: float = 0.01,
) -> float:
    """Paper Eq. 2 joint gradient-noise/curvature-volatility safe bound."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive exact integer")
    gradient_mean = _finite_scalar("squared_gradient_mean", squared_gradient_mean)
    sampling_variance = _finite_scalar(
        "per_sample_gradient_variance", per_sample_gradient_variance
    )
    curvature_variance = _finite_scalar(
        "normalized_curvature_variance", normalized_curvature_variance
    )
    inflation = _probability("volatility_inflation", volatility_inflation)
    margin = _probability("safety_margin", safety_margin, include_one=False)
    combined_variance = sampling_variance + inflation * gradient_mean * curvature_variance
    if combined_variance == 0.0:
        return math.inf
    return (1.0 - margin) * batch_size * gradient_mean / combined_variance


def noise_curvature_step_size(
    base_step_size: float,
    *,
    effective_step_size: float,
    safe_bound: float,
    early_training: bool,
    adjustment: float = 0.01,
    enabled: bool = True,
) -> float:
    """One bounded per-layer cooling/warming decision from paper Algorithm 1."""
    base = _finite_scalar("base_step_size", base_step_size)
    effective = _finite_scalar("effective_step_size", effective_step_size)
    if type(safe_bound) not in (int, float) or math.isnan(safe_bound) or safe_bound < 0:
        raise ValueError("safe_bound must be an exact nonnegative scalar")
    bound = float(safe_bound)
    rate = _probability("adjustment", adjustment, include_one=False)
    if type(early_training) is not bool:
        raise ValueError("early_training must be an exact bool")
    if type(enabled) is not bool:
        raise ValueError("enabled must be an exact bool")
    if not enabled:
        return base
    if effective > bound and effective > 0.12:
        return base * (1.0 - rate)
    if early_training and effective < 0.1 * bound:
        return base * (1.0 + rate)
    return base
