"""Layerwise noise-curvature scheduling for the nonpromoting IPMNIST lane.

This is an executable adaptation of Baveja, Lewandowski & Schmidt,
arXiv:2509.19698v3.  The paper uses minibatch random-label MNIST.  ASI's
screening runner is online, so this implementation retains the most recent
``control_interval`` examples as the diagnostic minibatch.  The optimizer
still performs exactly one update per arriving example.

The module contains no result or promotion logic.  Strict development receipt
validation lives in :mod:`alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from typing import Final, Literal, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from alberta_framework._scan_resources import (
    ScanBudget,
    require_parallel_count,
    require_scan_steps,
    require_step_units,
)
from alberta_framework.core.baseline_optimizers import Adam, AdamParamState
from alberta_framework.core.normalizers import _saturating_int32_counter_increment
from alberta_framework.core.update_safety import floating_tree_is_finite

PAPER_REVISION: Final[str] = "arXiv:2509.19698v3"
LAYER_PARAMETER_NAMES: Final[tuple[tuple[str, str], ...]] = (
    ("w1", "b1"),
    ("w2", "b2"),
    ("w3", "b3"),
)
PARAMETER_NAMES: Final[frozenset[str]] = frozenset(
    name for layer in LAYER_PARAMETER_NAMES for name in layer
)
_INT32_MAX: Final[int] = (1 << 31) - 1
_MAX_POWER_ITERATIONS: Final[int] = 10_000
_MAX_CONTROLLER_POWER_UNITS: Final[int] = 1_000_000
_POWER_ITERATION_BUDGET = ScanBudget(
    "noise-curvature controller",
    maximum_steps=_MAX_POWER_ITERATIONS,
    maximum_parallel=_MAX_CONTROLLER_POWER_UNITS,
    maximum_step_units=_MAX_CONTROLLER_POWER_UNITS,
)
_MAX_ARRAY_ELEMENTS: Final[int] = 1_000_000
_MAX_PERSISTENT_BYTES: Final[int] = 256 * 1024 * 1024

LossFn = Callable[[dict[str, Array], Array, Array], tuple[Array, Array]]
ControllerMode = Literal["fixed", "gradient_only", "volatility_only", "combined"]


def _exact_int(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _INT32_MAX:
        raise ValueError(
            f"{name} must be an exact signed-int32 integer in [{minimum}, {_INT32_MAX}]"
        )
    return value


def _finite_float(
    name: str,
    value: object,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
    positive: bool = False,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite float")
    resolved = value
    if resolved < minimum or (positive and resolved == 0.0):
        raise ValueError(f"{name} is outside its registered domain")
    if maximum is not None and resolved > maximum:
        raise ValueError(f"{name} is outside its registered domain")
    return resolved


@dataclasses.dataclass(frozen=True, slots=True)
class NoiseCurvatureConfig:
    """Host-validated static configuration for one scheduler arm."""

    mode: ControllerMode
    total_steps: int
    control_interval: int = 40
    power_iterations: int = 1
    base_step_size: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 1e-3
    ema_decay: float = 0.9
    volatility_epsilon: float = 1e-8
    volatility_inflation: float = 1.0
    volatility_kappa: float = 1.0
    safety_factor: float = 0.8
    cool_rate: float = 0.99
    warm_rate: float = 1.01
    warm_fraction: float = 0.3
    timid_fraction: float = 0.1
    effective_step_floor: float = 0.12

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in (
            "fixed",
            "gradient_only",
            "volatility_only",
            "combined",
        ):
            raise ValueError("mode must be a registered controller mode")
        total_steps = _exact_int("total_steps", self.total_steps, minimum=1)
        interval = _exact_int("control_interval", self.control_interval, minimum=1)
        power_iterations = require_scan_steps(
            "power_iterations",
            _exact_int("power_iterations", self.power_iterations, minimum=1),
            _POWER_ITERATION_BUDGET,
        )
        if self.total_steps % interval:
            raise ValueError("total_steps must be divisible by control_interval")
        controller_updates = require_parallel_count(
            "controller update count",
            total_steps // interval,
            _POWER_ITERATION_BUDGET,
        )
        require_step_units(power_iterations, controller_updates, _POWER_ITERATION_BUDGET)
        _finite_float("base_step_size", self.base_step_size, positive=True)
        _finite_float("beta1", self.beta1, maximum=1.0)
        _finite_float("beta2", self.beta2, maximum=1.0)
        if self.beta1 == 1.0 or self.beta2 == 1.0:
            raise ValueError("Adam beta values must lie in [0,1)")
        _finite_float("adam_epsilon", self.adam_epsilon, positive=True)
        _finite_float("weight_decay", self.weight_decay)
        _finite_float("ema_decay", self.ema_decay, maximum=1.0)
        if self.ema_decay == 1.0:
            raise ValueError("ema_decay must lie in [0,1)")
        _finite_float("volatility_epsilon", self.volatility_epsilon, positive=True)
        _finite_float("volatility_inflation", self.volatility_inflation, maximum=1.0)
        _finite_float("volatility_kappa", self.volatility_kappa, positive=True)
        _finite_float("safety_factor", self.safety_factor, positive=True, maximum=1.0)
        _finite_float("cool_rate", self.cool_rate, positive=True, maximum=1.0)
        _finite_float("warm_rate", self.warm_rate, positive=True)
        if self.warm_rate < 1.0:
            raise ValueError("warm_rate must be at least one")
        _finite_float("warm_fraction", self.warm_fraction, maximum=1.0)
        _finite_float("timid_fraction", self.timid_fraction, maximum=1.0)
        _finite_float("effective_step_floor", self.effective_step_floor)


@chex.dataclass(frozen=True)
class NoiseCurvatureState:
    """All optimizer and diagnostic state charged to one comparator arm."""

    adam: dict[str, AdamParamState]
    examples: Array
    labels: Array
    buffer_index: Array
    buffer_count: Array
    power_vectors: dict[str, Array]
    learning_rates: Array
    curvature_mean: Array
    curvature_variance: Array
    controller_count: Array
    cool_counts: Array
    warm_counts: Array
    diagnostic_failures: Array


def _validated_params(params: object) -> dict[str, Array]:
    if type(params) is not dict or set(params) != PARAMETER_NAMES:
        raise ValueError(f"params must have exactly {sorted(PARAMETER_NAMES)}")
    checked: dict[str, Array] = {}
    total = 0
    for name in sorted(PARAMETER_NAMES):
        value = params[name]
        actual_type = type(value)
        if actual_type is not np.ndarray and not issubclass(actual_type, Array):
            raise ValueError(f"params.{name} must be an exact NumPy or JAX array")
        array = jnp.asarray(value)
        if not jnp.issubdtype(array.dtype, jnp.floating) or array.size < 1:
            raise ValueError(f"params.{name} must be a nonempty floating array")
        if not bool(jnp.all(jnp.isfinite(array))):
            raise ValueError(f"params.{name} must contain only finite values")
        total += array.size
        if total > _MAX_ARRAY_ELEMENTS:
            raise ValueError("parameter tree exceeds the one-million-element limit")
        checked[name] = array
    w1, b1 = checked["w1"], checked["b1"]
    w2, b2 = checked["w2"], checked["b2"]
    w3, b3 = checked["w3"], checked["b3"]
    if (
        w1.ndim != 2
        or w2.ndim != 2
        or w3.ndim != 2
        or b1.shape != (w1.shape[1],)
        or w2.shape[0] != w1.shape[1]
        or b2.shape != (w2.shape[1],)
        or w3.shape[0] != w2.shape[1]
        or b3.shape != (w3.shape[1],)
    ):
        raise ValueError("params do not form the registered three-layer MLP")
    return checked


def noise_curvature_persistent_bytes(
    *, parameter_count: int, input_dim: int, control_interval: int
) -> int:
    """Exact canonical numeric payload bytes for one scheduler run.

    Scope: parameters, Adam's two moments plus five scalar fields for each of
    six tensors, one retained block-power vector, diagnostic examples/labels,
    three layerwise LR/EMA arrays, and all counters.  Runner keys and the
    externally supplied data/schedule are excluded.
    """

    parameter_count = _exact_int("parameter_count", parameter_count, minimum=1)
    input_dim = _exact_int("input_dim", input_dim, minimum=1)
    control_interval = _exact_int("control_interval", control_interval, minimum=1)
    if parameter_count > _MAX_ARRAY_ELEMENTS:
        raise ValueError("parameter_count exceeds the one-million-element limit")
    if control_interval > _INT32_MAX // input_dim:
        raise ValueError("diagnostic buffer element count exceeds signed int32")
    buffer_elements = control_interval * input_dim
    if buffer_elements > _MAX_ARRAY_ELEMENTS:
        raise ValueError("diagnostic buffer exceeds the one-million-element limit")
    # params + Adam m/v + power vector = 4P floats. Adam owns five scalar
    # float fields for each of six parameter tensors. Remaining state is nine
    # layer floats and ten int32 counters/scalars.
    scalar_count = 4 * parameter_count + buffer_elements + control_interval + 30 + 9 + 10
    if scalar_count > _INT32_MAX // 4:
        raise ValueError("scheduler persistent bytes exceed signed int32")
    result = 4 * scalar_count
    if result > _MAX_PERSISTENT_BYTES:
        raise ValueError("scheduler persistent bytes exceed 256 MiB")
    return result


def init_noise_curvature_state(params: object, config: NoiseCurvatureConfig) -> NoiseCurvatureState:
    """Initialize Adam and deterministic retained block-power directions."""

    if type(config) is not NoiseCurvatureConfig:
        raise TypeError("config must be an exact NoiseCurvatureConfig")
    checked = _validated_params(params)
    parameter_count = sum(value.size for value in checked.values())
    input_dim = checked["w1"].shape[0]
    noise_curvature_persistent_bytes(
        parameter_count=parameter_count,
        input_dim=input_dim,
        control_interval=config.control_interval,
    )
    optimizer = Adam(
        step_size=config.base_step_size,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    adam = {name: optimizer.init_for_shape(value.shape) for name, value in checked.items()}
    power_vectors: dict[str, Array] = {}
    for names in LAYER_PARAMETER_NAMES:
        count = sum(checked[name].size for name in names)
        scale = jnp.asarray(1.0 / math.sqrt(count), dtype=jnp.float32)
        for name in names:
            power_vectors[name] = jnp.ones_like(checked[name]) * scale
    return NoiseCurvatureState(  # type: ignore[call-arg]
        adam=adam,
        examples=jnp.zeros((config.control_interval, input_dim), dtype=jnp.float32),
        labels=jnp.zeros((config.control_interval,), dtype=jnp.int32),
        buffer_index=jnp.asarray(0, dtype=jnp.int32),
        buffer_count=jnp.asarray(0, dtype=jnp.int32),
        power_vectors=power_vectors,
        learning_rates=jnp.full((3,), config.base_step_size, dtype=jnp.float32),
        curvature_mean=jnp.zeros((3,), dtype=jnp.float32),
        curvature_variance=jnp.zeros((3,), dtype=jnp.float32),
        controller_count=jnp.asarray(0, dtype=jnp.int32),
        cool_counts=jnp.zeros((3,), dtype=jnp.int32),
        warm_counts=jnp.zeros((3,), dtype=jnp.int32),
        diagnostic_failures=jnp.asarray(0, dtype=jnp.int32),
    )


def noise_curvature_safe_bound(
    *,
    mode: ControllerMode,
    batch_size: int,
    squared_gradient_mean: Array,
    per_sample_gradient_variance: Array,
    curvature_volatility: Array,
    volatility_inflation: float,
    volatility_kappa: float,
    safety_factor: float,
) -> Array:
    """Evaluate paper Eq. 1/Eq. 2 or either causal single-signal ablation."""

    if mode not in ("fixed", "gradient_only", "volatility_only", "combined"):
        raise ValueError("mode must be a registered controller mode")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be an exact positive integer")
    gradient_norm = jnp.maximum(jnp.asarray(squared_gradient_mean), 0.0)
    gradient_variance = jnp.maximum(jnp.asarray(per_sample_gradient_variance), 0.0)
    volatility = jnp.maximum(jnp.asarray(curvature_volatility), 0.0)
    if mode == "volatility_only":
        denominator = jnp.asarray(volatility_kappa, jnp.float32) * volatility
        return jnp.where(denominator > 0.0, safety_factor / denominator, jnp.inf)
    beta = 0.0 if mode == "gradient_only" else volatility_inflation
    combined_variance = gradient_variance + beta * gradient_norm * volatility
    denominator = jnp.asarray(combined_variance, jnp.float32)
    numerator = jnp.asarray(safety_factor * batch_size, jnp.float32) * gradient_norm
    return jnp.where(denominator > 0.0, numerator / denominator, jnp.inf)


def noise_curvature_learning_rate(
    learning_rate: Array,
    *,
    effective_step: Array,
    safe_bound: Array,
    is_early: Array,
    mode: ControllerMode,
    config: NoiseCurvatureConfig,
) -> tuple[Array, Array, Array]:
    """Apply paper Algorithm 1 with ``≪`` operationalized as 0.1x."""

    if mode == "fixed":
        false = jnp.asarray(False, dtype=jnp.bool_)
        return learning_rate, false, false
    cool = (effective_step > safe_bound) & (effective_step > config.effective_step_floor)
    warm = (~cool) & is_early & (effective_step < config.timid_fraction * safe_bound)
    candidate = jnp.where(
        cool,
        learning_rate * config.cool_rate,
        jnp.where(warm, learning_rate * config.warm_rate, learning_rate),
    )
    return candidate, cool, warm


def _layer_squared_norm(tree: Mapping[str, Array], names: tuple[str, str]) -> Array:
    return sum(
        (jnp.sum(jnp.square(tree[name])) for name in names),
        start=jnp.asarray(0.0, dtype=jnp.float32),
    )


def _layer_effective_step(
    adam: Mapping[str, AdamParamState], names: tuple[str, str], learning_rate: Array
) -> Array:
    total = jnp.asarray(0.0, dtype=jnp.float32)
    count = 0
    for name in names:
        state = adam[name]
        v_hat = state.v / (1.0 - state.beta2**state.t)
        multiplier = learning_rate / ((1.0 - state.beta1**state.t) * (jnp.sqrt(v_hat) + state.eps))
        total = total + jnp.sum(multiplier)
        count += state.v.size
    return total / jnp.asarray(count, dtype=jnp.float32)


def _layer_normalizing_step(
    adam: Mapping[str, AdamParamState], names: tuple[str, str], learning_rate: Array
) -> Array:
    """Paper Appendix C.1 ``alpha_agg`` for normalized sharpness."""

    second_moment_total = jnp.asarray(0.0, dtype=jnp.float32)
    count = 0
    for name in names:
        state = adam[name]
        v_hat = state.v / (1.0 - state.beta2**state.t)
        second_moment_total = second_moment_total + jnp.sum(v_hat)
        count += state.v.size
    rms_sqrt_v = jnp.sqrt(
        jnp.maximum(
            second_moment_total / jnp.asarray(count, dtype=jnp.float32),
            0.0,
        )
    )
    return learning_rate / (rms_sqrt_v + adam[names[0]].eps)


def _controller_update(
    params: dict[str, Array],
    state: NoiseCurvatureState,
    loss_fn: LossFn,
    config: NoiseCurvatureConfig,
) -> NoiseCurvatureState:
    examples, labels = state.examples, state.labels

    def scalar_loss(candidate: dict[str, Array], x: Array, y: Array) -> Array:
        return loss_fn(candidate, x, y)[0]

    sample_grads = jax.vmap(jax.grad(scalar_loss), in_axes=(None, 0, 0))(params, examples, labels)
    mean_grads = {name: jnp.mean(value, axis=0) for name, value in sample_grads.items()}

    def batch_loss(candidate: dict[str, Array]) -> Array:
        return jnp.mean(jax.vmap(scalar_loss, in_axes=(None, 0, 0))(candidate, examples, labels))

    gradient_fn = jax.grad(batch_loss)
    next_power = dict(state.power_vectors)
    sharpness_values: list[Array] = []
    gradient_norms: list[Array] = []
    gradient_variances: list[Array] = []
    effective_steps: list[Array] = []
    normalizing_steps: list[Array] = []
    for layer_index, names in enumerate(LAYER_PARAMETER_NAMES):
        vector = {name: jnp.zeros_like(value) for name, value in params.items()}
        for name in names:
            vector[name] = state.power_vectors[name]
        hessian_vector = vector
        rayleigh = jnp.asarray(0.0, dtype=jnp.float32)
        for _ in range(config.power_iterations):
            hessian_vector = jax.jvp(gradient_fn, (params,), (vector,))[1]
            rayleigh = sum(
                (jnp.sum(vector[name] * hessian_vector[name]) for name in names),
                start=jnp.asarray(0.0, dtype=jnp.float32),
            )
            norm = jnp.sqrt(_layer_squared_norm(hessian_vector, names))
            valid_norm = jnp.isfinite(norm) & (norm > 0.0)
            for name in names:
                normalized = hessian_vector[name] / jnp.maximum(norm, 1e-30)
                vector[name] = jnp.where(valid_norm, normalized, vector[name])
        for name in names:
            next_power[name] = vector[name]
        sharpness_values.append(jnp.maximum(jnp.abs(rayleigh), 0.0))
        gradient_norms.append(_layer_squared_norm(mean_grads, names))
        gradient_variances.append(
            sum(
                (
                    jnp.mean(
                        jnp.sum(
                            jnp.square(sample_grads[name] - mean_grads[name]),
                            axis=tuple(range(1, sample_grads[name].ndim)),
                        )
                    )
                    for name in names
                ),
                start=jnp.asarray(0.0, dtype=jnp.float32),
            )
        )
        effective_steps.append(
            _layer_effective_step(state.adam, names, state.learning_rates[layer_index])
        )
        normalizing_steps.append(
            _layer_normalizing_step(state.adam, names, state.learning_rates[layer_index])
        )

    sharpness = jnp.stack(sharpness_values)
    gradient_norm = jnp.stack(gradient_norms)
    gradient_variance = jnp.stack(gradient_variances)
    effective_step = jnp.stack(effective_steps)
    normalized_sharpness = jnp.stack(normalizing_steps) * sharpness
    controller_count = _saturating_int32_counter_increment(state.controller_count)
    decay = jnp.asarray(config.ema_decay, dtype=jnp.float32)
    mean_uncorrected = decay * state.curvature_mean + (1.0 - decay) * normalized_sharpness
    correction = 1.0 - decay ** controller_count.astype(jnp.float32)
    mean = mean_uncorrected / correction
    deviations = jnp.square(normalized_sharpness - mean)
    variance_uncorrected = decay * state.curvature_variance + (1.0 - decay) * deviations
    variance = variance_uncorrected / correction
    volatility = variance / (mean + config.volatility_epsilon)

    bounds = jnp.stack(
        [
            noise_curvature_safe_bound(
                mode=config.mode,
                batch_size=config.control_interval,
                squared_gradient_mean=gradient_norm[index],
                per_sample_gradient_variance=gradient_variance[index],
                curvature_volatility=volatility[index],
                volatility_inflation=config.volatility_inflation,
                volatility_kappa=config.volatility_kappa,
                safety_factor=config.safety_factor,
            )
            for index in range(3)
        ]
    )
    steps_seen = state.buffer_count
    is_early = steps_seen < int(config.warm_fraction * config.total_steps)
    lr_updates = [
        noise_curvature_learning_rate(
            state.learning_rates[index],
            effective_step=effective_step[index],
            safe_bound=bounds[index],
            is_early=is_early,
            mode=config.mode,
            config=config,
        )
        for index in range(3)
    ]
    learning_rates = jnp.stack([update[0] for update in lr_updates])
    cool = jnp.stack([update[1] for update in lr_updates]).astype(jnp.int32)
    warm = jnp.stack([update[2] for update in lr_updates]).astype(jnp.int32)
    valid = (
        floating_tree_is_finite(next_power)
        & jnp.all(jnp.isfinite(mean_uncorrected))
        & jnp.all(jnp.isfinite(variance_uncorrected))
        & jnp.all(jnp.isfinite(learning_rates))
        & jnp.all(learning_rates > 0.0)
    )
    candidate_state = state.replace(  # type: ignore[attr-defined]
        power_vectors=jax.tree.map(
            lambda candidate, prior: jnp.where(valid, candidate, prior),
            next_power,
            state.power_vectors,
        ),
        learning_rates=jnp.where(valid, learning_rates, state.learning_rates),
        curvature_mean=jnp.where(valid, mean_uncorrected, state.curvature_mean),
        curvature_variance=jnp.where(valid, variance_uncorrected, state.curvature_variance),
        controller_count=jnp.where(valid, controller_count, state.controller_count),
        cool_counts=state.cool_counts + jnp.where(valid, cool, 0),
        warm_counts=state.warm_counts + jnp.where(valid, warm, 0),
        diagnostic_failures=state.diagnostic_failures + (~valid).astype(jnp.int32),
    )
    return cast(NoiseCurvatureState, candidate_state)


def noise_curvature_step(
    params: dict[str, Array],
    state: NoiseCurvatureState,
    grads: dict[str, Array],
    x: Array,
    y: Array,
    loss_fn: LossFn,
    config: NoiseCurvatureConfig,
) -> tuple[dict[str, Array], NoiseCurvatureState]:
    """Apply one AdamW update, buffer the example, and conditionally control."""

    optimizer = Adam(
        step_size=config.base_step_size,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    new_params: dict[str, Array] = {}
    new_adam: dict[str, AdamParamState] = {}
    all_updates_valid = jnp.asarray(True, dtype=jnp.bool_)
    for layer_index, names in enumerate(LAYER_PARAMETER_NAMES):
        for name in names:
            layer_state = state.adam[name].replace(  # type: ignore[attr-defined]
                step_size=state.learning_rates[layer_index]
            )
            update = optimizer.update_from_gradient_checked(
                layer_state, grads[name], error=None, param=params[name]
            )
            new_params[name] = params[name] - update.step
            new_adam[name] = cast(AdamParamState, update.new_state)
            all_updates_valid = all_updates_valid & update.update_applied
    new_params = jax.tree.map(
        lambda candidate, prior: jnp.where(all_updates_valid, candidate, prior),
        new_params,
        params,
    )
    new_adam = jax.tree.map(
        lambda candidate, prior: jnp.where(all_updates_valid, candidate, prior),
        new_adam,
        state.adam,
    )
    index = state.buffer_index
    examples = state.examples.at[index].set(x)
    labels = state.labels.at[index].set(y)
    next_index = (index + 1) % config.control_interval
    buffer_count = _saturating_int32_counter_increment(state.buffer_count)
    candidate_state = state.replace(  # type: ignore[attr-defined]
        adam=new_adam,
        examples=examples,
        labels=labels,
        buffer_index=next_index,
        buffer_count=buffer_count,
    )
    should_control = (buffer_count % config.control_interval) == 0
    controlled_state = jax.lax.cond(
        should_control,
        lambda value: _controller_update(new_params, value, loss_fn, config),
        lambda value: value,
        candidate_state,
    )
    return new_params, controlled_state
