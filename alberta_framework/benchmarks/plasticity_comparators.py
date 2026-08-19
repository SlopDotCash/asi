"""Bounded, nonpromoting plasticity comparator primitives.

These kernels make the mechanisms in issues #1559--#1567 executable without
claiming that a development comparison has been run.  They share one matched
protocol/accounting record so a future IPMNIST or streaming-control runner can
compare arms without changing seeds, examples, updates, or information.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array, core

_INT32_MAX: Final[int] = (1 << 31) - 1
_MAX_ARRAY_ELEMENTS: Final[int] = 1_000_000
_MAX_PERSISTENT_BYTES: Final[int] = 256 * 1024 * 1024
_MAX_PROTOCOL_STRING_BYTES: Final[int] = 4_096
_MAX_PROTOCOL_CONTAINER_ITEMS: Final[int] = 4_096


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
            value = getattr(self, field)
            if type(value) is not str or not value:
                raise ValueError(f"{field} must be an exact non-empty string")
            try:
                encoded = value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(f"{field} must be valid UTF-8") from error
            if len(encoded) > _MAX_PROTOCOL_STRING_BYTES or "\x00" in value:
                raise ValueError(f"{field} exceeds its bounded string domain")
        required_axes = ("seed", "updates", "observations", "example_order")
        if (
            type(self.matched_axes) is not tuple
            or len(self.matched_axes) != len(required_axes)
            or any(type(value) is not str for value in self.matched_axes)
            or self.matched_axes != required_axes
        ):
            raise ValueError(f"matched_axes must equal {required_axes!r}")
        for field in ("persistent_bytes", "environment_or_data_steps", "model_queries"):
            _exact_int32(field, getattr(self, field))
        try:
            timing = float(self.timing_telemetry_seconds)
        except (OverflowError, TypeError, ValueError):
            timing = math.nan
        if type(self.timing_telemetry_seconds) not in (int, float) or not (
            math.isfinite(timing) and timing >= 0
        ):
            raise ValueError("timing_telemetry_seconds must be finite and nonnegative")
        if self.development_only is not True or self.scientific_promotion_allowed is not False:
            raise ValueError("comparator protocols are permanently nonpromoting")


PAPER_REVISIONS: Final[Mapping[str, str]] = MappingProxyType({
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
})


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
    try:
        scalar = float(cast(int | float, value))
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an exact finite scalar >= {minimum}") from error
    if not math.isfinite(scalar) or scalar < minimum:
        raise ValueError(f"{name} must be an exact finite scalar >= {minimum}")
    return scalar


def _exact_int32(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _INT32_MAX:
        raise ValueError(
            f"{name} must be an exact signed-int32 integer in [{minimum}, {_INT32_MAX}]"
        )
    return value


def _checked_product(name: str, *factors: int) -> int:
    product = 1
    for factor in factors:
        if factor < 0 or (factor and product > _INT32_MAX // factor):
            raise ValueError(f"{name} exceeds the signed-int32 resource limit")
        product *= factor
    return product


def _probability(name: str, value: object, *, include_one: bool = True) -> float:
    scalar = _finite_scalar(name, value)
    if scalar > 1.0 or (not include_one and scalar == 1.0):
        boundary = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError(f"{name} must lie in {boundary}")
    return scalar


def _array(name: str, value: object) -> Array:
    actual_type = type(value)
    if actual_type is not np.ndarray and not issubclass(actual_type, (Array, core.Tracer)):
        raise ValueError(f"{name} must be a NumPy or JAX array")
    array = jnp.asarray(value)
    if array.size > _MAX_ARRAY_ELEMENTS:
        raise ValueError(f"{name} exceeds the {_MAX_ARRAY_ELEMENTS}-element limit")
    return array


def _threefry_key(name: str, value: object) -> Array:
    actual_type = type(value)
    if not issubclass(actual_type, (Array, core.Tracer)):
        raise ValueError(f"{name} must be a scalar typed Threefry JAX key")
    trusted = cast(Array, value)
    try:
        words = jr.key_data(trusted)
        implementation = str(jr.key_impl(trusted))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a scalar typed Threefry JAX key") from error
    if (
        trusted.shape != ()
        or words.shape != (2,)
        or words.dtype != jnp.dtype(jnp.uint32)
        or implementation != "threefry2x32"
    ):
        raise ValueError(f"{name} must be a scalar typed Threefry JAX key")
    return trusted


def _finite_result(candidate: Array, fallback: Array, *, name: str) -> Array:
    valid = jnp.all(jnp.isfinite(candidate))
    safe = jnp.where(valid, candidate, fallback)
    if isinstance(valid, core.Tracer):
        return jnp.where(valid, safe, jnp.full_like(safe, jnp.nan))
    if not bool(valid):
        raise ValueError(f"{name} must produce only finite values")
    return safe


def _floating_array(name: str, value: object, *, nonempty: bool = True) -> Array:
    array = _array(name, value)
    if (nonempty and array.size == 0) or not jnp.issubdtype(array.dtype, jnp.floating):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{name} must be a {qualifier}floating array")
    if not isinstance(array, core.Tracer) and not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return array


def persistent_array_bytes(*arrays: Array | np.ndarray) -> int:
    """Return exact resident numeric payload bytes for comparator-owned arrays."""
    total = 0
    for value in arrays:
        array: Array | np.ndarray
        actual_type = type(value)
        if actual_type is np.ndarray:
            array = value
        elif issubclass(actual_type, Array):
            array = value
        else:
            raise ValueError("persistent comparator values must be arrays")
        if np.dtype(array.dtype).kind not in "biufc":
            raise ValueError("persistent comparator arrays must be numeric")
        size = int(array.nbytes)
        if size > _MAX_PERSISTENT_BYTES - total:
            raise ValueError("persistent comparator bytes exceed the 256 MiB limit")
        total += size
    return total


def effective_rank(features: Array, *, epsilon: float = 1e-12) -> Array:
    """Roy--Vetterli effective rank of a stacked feature matrix."""
    matrix = _floating_array("features", features)
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
    candidate = jnp.where(total > 0, jnp.exp(entropy), 0.0)
    return _finite_result(candidate, jnp.asarray(0.0, dtype=matrix.dtype), name="effective rank")


def l2_er_objective(
    task_loss: Array,
    parameters: tuple[Array, ...],
    feature_batches: tuple[Array, ...],
    *,
    l2_strength: float,
    rank_strength: float,
) -> Array:
    """Paper objective: task loss + L2 - mean effective rank."""
    if type(parameters) is not tuple or type(feature_batches) is not tuple:
        raise ValueError("parameters and feature_batches must be exact tuples")
    if (
        not parameters
        or not feature_batches
        or len(parameters) > _MAX_PROTOCOL_CONTAINER_ITEMS
        or len(feature_batches) > _MAX_PROTOCOL_CONTAINER_ITEMS
    ):
        raise ValueError("parameters and feature_batches must be non-empty tuples")
    _finite_scalar("l2_strength", l2_strength)
    _finite_scalar("rank_strength", rank_strength)
    loss = _floating_array("task_loss", task_loss)
    if loss.ndim != 0:
        raise ValueError("task_loss must be scalar")
    checked_parameters = tuple(
        _floating_array(f"parameters[{index}]", value)
        for index, value in enumerate(parameters)
    )
    l2 = sum(
        (jnp.sum(jnp.square(value)) for value in checked_parameters), jnp.asarray(0.0)
    )
    ranks = jnp.stack(tuple(effective_rank(value) for value in feature_batches))
    candidate = loss + l2_strength * l2 - rank_strength * jnp.mean(ranks)
    return _finite_result(candidate, loss, name="L2-ER objective")


def isometry_gradient(weights: Array) -> Array:
    """Gradient of the rectangular Gram-deviation penalty (paper Eq. 16)."""
    weights = _floating_array("weights", weights)
    if weights.ndim != 2:
        raise ValueError("weights must be rank two")
    rows, columns = weights.shape
    _checked_product(
        "isometry Gram bytes",
        min(rows, columns),
        min(rows, columns),
        weights.dtype.itemsize,
    )
    if rows >= columns:
        gram_error = weights.T @ weights - jnp.eye(columns, dtype=weights.dtype)
        candidate = 4.0 * weights @ gram_error
        return _finite_result(candidate, jnp.zeros_like(weights), name="isometry gradient")
    gram_error = weights @ weights.T - jnp.eye(rows, dtype=weights.dtype)
    candidate = 4.0 * gram_error @ weights
    return _finite_result(candidate, jnp.zeros_like(weights), name="isometry gradient")


def isometry_penalty(weights: Array) -> Array:
    """Squared rectangular Gram deviation used by AdamO Eq. 16."""
    weights = _floating_array("weights", weights)
    if weights.ndim != 2:
        raise ValueError("weights must be rank two")
    rows, columns = weights.shape
    _checked_product(
        "isometry Gram bytes",
        min(rows, columns),
        min(rows, columns),
        weights.dtype.itemsize,
    )
    gram = weights.T @ weights if rows >= columns else weights @ weights.T
    candidate = jnp.sum(
        jnp.square(gram - jnp.eye(min(rows, columns), dtype=weights.dtype))
    )
    return _finite_result(
        candidate, jnp.asarray(0.0, dtype=weights.dtype), name="isometry penalty"
    )


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
    _exact_int32("step", step, minimum=1)
    _finite_scalar("learning_rate", learning_rate)
    _finite_scalar("isometry_strength", isometry_strength)
    _probability("beta1", beta1, include_one=False)
    _probability("beta2", beta2, include_one=False)
    _finite_scalar("epsilon", epsilon, minimum=np.nextafter(0.0, 1.0))
    weights = _floating_array("weights", weights)
    gradient = _floating_array("task_gradient", task_gradient)
    first_moment = _floating_array("first_moment", first_moment)
    second_moment = _floating_array("second_moment", second_moment)
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
    candidate = weights - task_delta - iso_delta
    valid = (
        jnp.all(jnp.isfinite(candidate))
        & jnp.all(jnp.isfinite(moment))
        & jnp.all(jnp.isfinite(variance))
    )
    if isinstance(valid, core.Tracer):
        return (
            jnp.where(valid, candidate, jnp.full_like(candidate, jnp.nan)),
            jnp.where(valid, moment, jnp.full_like(moment, jnp.nan)),
            jnp.where(valid, variance, jnp.full_like(variance, jnp.nan)),
        )
    if not bool(valid):
        raise ValueError("AdamO update must produce only finite values")
    return candidate, moment, variance


def intentional_td_step_size(
    gradient: Array,
    *,
    intended_fraction: float,
    diagonal_scale: Array | None = None,
    epsilon: float = 1e-12,
) -> Array:
    """Intentional TD(0) Eq. 7, including optional diagonal scaling."""
    gradient = _floating_array("gradient", gradient)
    scale = (
        jnp.ones_like(gradient)
        if diagonal_scale is None
        else _floating_array("diagonal_scale", diagonal_scale)
    )
    if gradient.size == 0 or scale.shape != gradient.shape:
        raise ValueError("gradient and diagonal_scale must be matching non-empty arrays")
    _finite_scalar("intended_fraction", intended_fraction, minimum=np.nextafter(0.0, 1.0))
    _finite_scalar("epsilon", epsilon, minimum=np.nextafter(0.0, 1.0))
    denominator = jnp.vdot(gradient, scale * gradient).real
    candidate = jnp.asarray(intended_fraction) / jnp.maximum(denominator, epsilon)
    return _finite_result(
        candidate, jnp.asarray(0.0, dtype=gradient.dtype), name="intentional TD step size"
    )


def intentional_trace_step_size(
    trace: Array,
    diagonal_scale: Array,
    *,
    intended_fraction: float,
    discounted_gradient_energy: Array,
    epsilon: float = 1e-12,
) -> Array:
    """Conservative Intentional TD(lambda) step size from paper Eq. 12."""
    trace = _floating_array("trace", trace)
    diagonal_scale = _floating_array("diagonal_scale", diagonal_scale)
    energy = _floating_array("discounted_gradient_energy", discounted_gradient_energy)
    if trace.size == 0 or diagonal_scale.shape != trace.shape or energy.ndim != 0:
        raise ValueError("trace/scale must match and discounted_gradient_energy must be scalar")
    _finite_scalar("intended_fraction", intended_fraction, minimum=np.nextafter(0.0, 1.0))
    _finite_scalar("epsilon", epsilon, minimum=np.nextafter(0.0, 1.0))
    denominator = energy * jnp.vdot(
        trace, diagonal_scale * trace
    ).real
    candidate = jnp.asarray(intended_fraction) / jnp.sqrt(
        jnp.maximum(denominator, epsilon)
    )
    return _finite_result(
        candidate,
        jnp.asarray(0.0, dtype=trace.dtype),
        name="intentional trace step size",
    )


def bounded_elastic_mask(
    active: Array | np.ndarray,
    utilities: Array | np.ndarray,
    *,
    grow: int,
    prune: int,
) -> Array:
    """Static-shape adaptive elastic transition under a hard peak-width bound."""
    active_type = type(active)
    utilities_type = type(utilities)
    if active_type is not np.ndarray and not issubclass(active_type, Array):
        raise ValueError("active must be a NumPy or JAX array")
    if utilities_type is not np.ndarray and not issubclass(utilities_type, Array):
        raise ValueError("utilities must be a NumPy or JAX array")
    active_np = np.asarray(active)
    utilities_np = np.asarray(utilities)
    if active_np.ndim != 1 or utilities_np.shape != active_np.shape:
        raise ValueError("active and utilities must be matching vectors")
    if active_np.size > _MAX_ARRAY_ELEMENTS:
        raise ValueError("active mask exceeds the 1000000-element limit")
    if active_np.dtype != np.bool_ or utilities_np.dtype.kind not in "iuf":
        raise ValueError("active must be bool and utilities must be real numeric")
    if not np.isfinite(utilities_np).all():
        raise ValueError("utilities must be finite")
    grow = _exact_int32("grow", grow)
    prune = _exact_int32("prune", prune)
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
    weights = _floating_array("weights", weights)
    initial = _floating_array("initial_weights", initial_weights)
    utility = _floating_array("utilities", utilities)
    if weights.shape != initial.shape or utility.shape != weights.shape:
        raise ValueError("weights, initialization, and utilities must match")
    _probability("strength", strength)
    modes = ("utility", "utility_free", "l2_init", "hard_reset")
    if type(mode) is not str or mode not in modes:
        raise ValueError("unknown utility pull mode")
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
    candidate = weights + jnp.clip(rate, 0.0, 1.0) * (initial - weights)
    return _finite_result(candidate, weights, name="utility pull")


def nap_project(weights: Array, *, initial_norm: float, enabled: bool = True) -> Array:
    """NaP weight projection onto the initialization Frobenius radius."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be an exact bool")
    _finite_scalar("initial_norm", initial_norm)
    weights = _floating_array("weights", weights)
    if not enabled:
        return weights
    norm = jnp.linalg.norm(weights)
    candidate = jnp.where(norm > 0, weights * (initial_norm / norm), weights)
    return _finite_result(candidate, weights, name="NaP projection")


def churn_loss(before: Array, after: Array, *, strength: float) -> Array:
    """C-CHAIN Eq. 8 on a caller-owned disjoint reference batch."""
    before = _floating_array("before", before)
    after = _floating_array("after", after)
    if before.shape != after.shape:
        raise ValueError("reference predictions must have matching shapes")
    if before.size == 0:
        raise ValueError("reference predictions must be non-empty")
    _finite_scalar("strength", strength)
    candidate = 0.5 * strength * jnp.mean(jnp.square(after - before))
    return _finite_result(candidate, jnp.asarray(0.0, dtype=before.dtype), name="churn loss")


def ntk_threshold_rank(jacobian: Array, *, threshold: float = 0.99) -> Array:
    """Number of empirical-NTK singular values carrying a target energy fraction."""
    jacobian = _floating_array("jacobian", jacobian)
    if jacobian.ndim != 2 or not jacobian.size:
        raise ValueError("jacobian must be a non-empty matrix")
    _checked_product(
        "empirical NTK bytes",
        jacobian.shape[0],
        jacobian.shape[0],
        jacobian.dtype.itemsize,
    )
    target = _probability("threshold", threshold)
    if target == 0.0:
        raise ValueError("threshold must lie in (0, 1]")
    gram = jacobian @ jacobian.T
    gram_valid = jnp.all(jnp.isfinite(gram))
    safe_gram = jnp.where(gram_valid, gram, jnp.zeros_like(gram))
    if not isinstance(gram_valid, core.Tracer) and not bool(gram_valid):
        raise ValueError("empirical NTK must contain only finite values")
    singular = jnp.linalg.svd(safe_gram, compute_uv=False)
    total = jnp.sum(singular)
    cumulative = jnp.cumsum(singular)
    rank = jnp.searchsorted(cumulative, target * total, side="left") + 1
    return jnp.where(total > 0, rank, 0)


def smooth_leaky(value: Array, *, alpha: float, power: float, curvature: float) -> Array:
    """Smooth-Leaky activation from arXiv:2509.22562 Eq. 1."""
    value = _floating_array("value", value)
    _probability("alpha", alpha)
    _finite_scalar("power", power, minimum=np.nextafter(0.0, 1.0))
    _finite_scalar("curvature", curvature)
    candidate = alpha * value + (1.0 - alpha) * value * jax_sigmoid(
        curvature * value / power
    )
    return _finite_result(candidate, jnp.zeros_like(value), name="Smooth-Leaky activation")


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
    value = _floating_array("value", value)
    key = _threefry_key("key", key)
    if not training:
        candidate = jnp.where(
            value >= 0, probability * value, (1.0 - probability) * value
        )
        return _finite_result(candidate, jnp.zeros_like(value), name="AID activation")
    use_relu = jr.bernoulli(key, probability, value.shape)
    candidate = jnp.where(use_relu, jnp.maximum(value, 0), jnp.minimum(value, 0))
    return _finite_result(candidate, jnp.zeros_like(value), name="AID activation")


def deep_fourier_features(value: Array, *, enabled: bool = True) -> Array:
    """Deep Fourier feature map [sin(z), cos(z)]."""
    value = _floating_array("value", value)
    if type(enabled) is not bool:
        raise ValueError("enabled must be an exact bool")
    if value.ndim == 0:
        raise ValueError("Deep Fourier features require a non-scalar input")
    if enabled:
        _checked_product("Deep Fourier output bytes", value.size, 2, value.dtype.itemsize)
    candidate = (
        jnp.concatenate((jnp.sin(value), jnp.cos(value)), axis=-1) if enabled else value
    )
    return _finite_result(candidate, jnp.zeros_like(candidate), name="Deep Fourier features")


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
    batch_size = _exact_int32("batch_size", batch_size, minimum=1)
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
    result = (1.0 - margin) * batch_size * gradient_mean / combined_variance
    if not math.isfinite(result):
        raise ValueError("critical step size overflowed its finite scalar domain")
    return result


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
        result = base * (1.0 - rate)
    elif early_training and effective < 0.1 * bound:
        result = base * (1.0 + rate)
    else:
        result = base
    if not math.isfinite(result):
        raise ValueError("noise_curvature step size overflowed its finite scalar domain")
    return result
