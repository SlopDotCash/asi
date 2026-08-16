# mypy: disable-error-code="call-arg,name-defined"
"""Fixed-budget JAX prototype memory for Step 2 retention.

An online nearest-prototype classifier (cf. learning vector quantization,
Kohonen 1990; nearest-class-mean classification, Mensink et al. 2013): each
class owns a fixed number of prototype slots, a matched prototype is
EMA-updated toward the observation, and a sufficiently novel observation
claims an empty slot or recycles the least-used/oldest one (a fixed-budget,
distance-only variant of the allocation rule in Platt 1991).  Prediction is a
softmax over nearest-prototype class logits.

The budget is static, every step can update the memory, and the state is a
JAX PyTree, so the learner runs under ``jax.lax.scan``.  It is the core of
the packaged Step 2 retained-view memory
(:func:`alberta_framework.steps.step2.make_step2_memory_learner`).
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Mapping
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.update_safety import (
    floating_tree_is_finite,
    neutralize_array,
    select_transaction,
)

_INT32_MAX: int = 2**31 - 1
_FLOAT32_MIN_NORMAL: float = float.fromhex("0x1.0p-126")
_ACTUAL_INT_TYPES: tuple[type, ...] = (
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.longlong,
    np.ulonglong,
)


def _require_unit_interval(name: str, value: object) -> float:
    return validated_float32_scalar(name, value, lower=0.0, upper=1.0)


def _require_half_open_unit_interval(name: str, value: object) -> float:
    return validated_float32_scalar(
        name,
        value,
        positive=True,
        upper=1.0,
    )


def _require_nonnegative_real(name: str, value: object) -> float:
    return validated_float32_scalar(name, value, lower=0.0)


def _require_positive_normal_real(name: str, value: object) -> float:
    return validated_float32_scalar(name, value, lower=_FLOAT32_MIN_NORMAL)


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer")
    number = operator.index(cast(SupportsIndex, value))
    if minimum is not None and number < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive")
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative")
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _require_float32_resource(
    name: str,
    *,
    vector_scalars: int,
    fixed_scalars: int = 0,
) -> None:
    total_scalars = vector_scalars + fixed_scalars
    if total_scalars > _INT32_MAX:
        raise ValueError(f"{name} scalar count must fit signed int32")
    if 4 * total_scalars > _INT32_MAX:
        raise ValueError(f"{name} byte count must fit signed int32")


def _require_sequence_resource(
    name: str,
    *,
    float32_scalars: int,
    bool_scalars: int,
) -> None:
    if float32_scalars + bool_scalars > _INT32_MAX:
        raise ValueError(f"{name} scalar count must fit signed int32")
    if 4 * float32_scalars + bool_scalars > _INT32_MAX:
        raise ValueError(f"{name} byte count must fit signed int32")


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array with shape and dtype metadata")
    actual_shape = tuple(value.shape)
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {actual_shape}")
    expected_dtype = jnp.dtype(dtype)
    actual_dtype = jnp.dtype(value.dtype)
    if actual_dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}; got {actual_dtype}")


def _require_shape(name: str, value: Array, shape: tuple[int, ...]) -> None:
    actual_shape = tuple(value.shape)
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {actual_shape}")


def _saturating_increment(value: Array) -> Array:
    one = jnp.asarray(1, dtype=jnp.int32)
    return jnp.minimum(value, jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)) + one


@chex.dataclass(frozen=True)
class PrototypeMemoryConfig:
    """Configuration for :class:`PrototypeMemoryLearner`.

    Args:
        feature_dim: Observation dimensionality.
        n_classes: Number of one-hot classes.
        slots_per_class: Fixed prototype budget for each class.
        update_rate: EMA rate for updating a matched prototype.
        novelty_threshold: Mean-squared-distance threshold for allocating a
            new prototype instead of updating the nearest existing one.
        bandwidth: Distance-to-logit bandwidth for softmax prediction.

    ``novelty_threshold`` and ``bandwidth`` are both in units of *mean
    per-dimension* squared distance, so they are insensitive to
    ``feature_dim`` but scale with the square of the input magnitude. The
    defaults are the development-selected Step 2 MNIST-pixel calibration
    (:class:`~alberta_framework.steps.step2.Step2MemoryConfig` uses the same
    values); retune them for features on a different scale.
    """

    feature_dim: int
    n_classes: int
    slots_per_class: int = 20
    update_rate: float = 0.3
    novelty_threshold: float = 0.08
    bandwidth: float = 0.01

    def __post_init__(self) -> None:
        """Validate and canonicalize configuration."""
        _validate_config(self)

    def to_config(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "type": "PrototypeMemoryConfig",
            "feature_dim": self.feature_dim,
            "n_classes": self.n_classes,
            "slots_per_class": self.slots_per_class,
            "update_rate": self.update_rate,
            "novelty_threshold": self.novelty_threshold,
            "bandwidth": self.bandwidth,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PrototypeMemoryConfig:
        """Reconstruct from :meth:`to_config` output."""
        if not issubclass(type(config), Mapping):
            raise ValueError("PrototypeMemoryConfig payload must be a mapping")
        try:
            payload = dict(config)
        except Exception as error:
            raise ValueError("PrototypeMemoryConfig mapping could not be read") from error
        payload.pop("type", None)
        return cls(**payload)


@chex.dataclass(frozen=True)
class PrototypeMemoryState:
    """State for :class:`PrototypeMemoryLearner`."""

    means: Float[Array, "n_classes slots_per_class feature_dim"]
    counts: Float[Array, "n_classes slots_per_class"]
    last_update: Array
    step_count: Array


@chex.dataclass(frozen=True)
class PrototypeMemoryUpdateResult:
    """Result of one prototype-memory update."""

    state: PrototypeMemoryState
    predictions: Float[Array, " n_classes"]
    errors: Float[Array, " n_classes"]
    metrics: Float[Array, " 6"]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeMemoryLearningResult:
    """Result from :func:`run_prototype_memory_arrays`."""

    state: PrototypeMemoryState
    predictions: Float[Array, "steps n_classes"]
    metrics: Float[Array, "steps 6"]
    updates_applied: Bool[Array, " steps"]


def _validate_config(config: PrototypeMemoryConfig) -> None:
    feature_dim = _require_int(
        "feature_dim", config.feature_dim, minimum=1, maximum=_INT32_MAX
    )
    n_classes = _require_int(
        "n_classes", config.n_classes, minimum=2, maximum=_INT32_MAX
    )
    slots_per_class = _require_int(
        "slots_per_class", config.slots_per_class, minimum=1, maximum=_INT32_MAX
    )
    update_rate = _require_half_open_unit_interval("update_rate", config.update_rate)
    novelty_threshold = _require_nonnegative_real(
        "novelty_threshold", config.novelty_threshold
    )
    bandwidth = _require_positive_normal_real("bandwidth", config.bandwidth)
    object.__setattr__(config, "feature_dim", feature_dim)
    object.__setattr__(config, "n_classes", n_classes)
    object.__setattr__(config, "slots_per_class", slots_per_class)
    object.__setattr__(config, "update_rate", update_rate)
    object.__setattr__(config, "novelty_threshold", novelty_threshold)
    object.__setattr__(config, "bandwidth", bandwidth)
    if n_classes * slots_per_class > _INT32_MAX:
        raise ValueError("PrototypeMemoryConfig dimensions must fit signed int32")
    if n_classes * slots_per_class * feature_dim > _INT32_MAX:
        raise ValueError("PrototypeMemoryConfig dimensions must fit signed int32")
    total_means_scalars = n_classes * slots_per_class * feature_dim
    _require_float32_resource(
        "PrototypeMemoryConfig state",
        vector_scalars=total_means_scalars,
        fixed_scalars=2 * n_classes * slots_per_class + 1,
    )


def _softmax(logits: Array) -> Array:
    shifted = logits - jnp.max(logits)
    exp = jnp.exp(shifted)
    return exp / jnp.maximum(jnp.sum(exp), 1e-12)


class PrototypeMemoryLearner:
    """Fixed-budget multi-prototype classifier.

    The learner assumes one-hot classification targets.  Non-finite or
    non-simplex targets are ignored by the memory update but still produce a
    prediction and metrics.  This keeps the learner safe in mixed-head streams.
    """

    def __init__(self, config: PrototypeMemoryConfig):
        _validate_config(config)
        self._config = config

    @property
    def config(self) -> PrototypeMemoryConfig:
        """Learner configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize the learner configuration."""
        return {
            "type": "PrototypeMemoryLearner",
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PrototypeMemoryLearner:
        """Reconstruct a learner from :meth:`to_config` output."""
        if not issubclass(type(config), Mapping):
            raise ValueError("PrototypeMemoryLearner payload must be a mapping")
        try:
            payload = dict(config)
        except Exception as error:
            raise ValueError("PrototypeMemoryLearner mapping could not be read") from error
        inner_raw = payload.get("config")
        if not issubclass(type(inner_raw), Mapping):
            raise ValueError("PrototypeMemoryLearner config must be a mapping")
        try:
            inner = dict(cast(Mapping[str, Any], inner_raw))
        except Exception as error:
            raise ValueError("PrototypeMemoryLearner config could not be read") from error
        return cls(PrototypeMemoryConfig.from_config(inner))

    def _validate_state_static_contract(self, state: PrototypeMemoryState) -> None:
        """Reject malformed adopted state before any traced computation."""
        if type(state) is not PrototypeMemoryState:
            raise TypeError("state must be a PrototypeMemoryState")
        cfg = self._config
        slot_shape = (cfg.n_classes, cfg.slots_per_class)
        _require_array(
            state.means,
            name="state.means",
            shape=(*slot_shape, cfg.feature_dim),
            dtype=jnp.float32,
        )
        _require_array(
            state.counts,
            name="state.counts",
            shape=slot_shape,
            dtype=jnp.float32,
        )
        _require_array(
            state.last_update,
            name="state.last_update",
            shape=slot_shape,
            dtype=jnp.int32,
        )
        _require_array(
            state.step_count,
            name="state.step_count",
            shape=(),
            dtype=jnp.int32,
        )

    @staticmethod
    def _state_is_valid(state: PrototypeMemoryState) -> Bool[Array, ""]:
        return (
            floating_tree_is_finite(state)
            & jnp.all(state.counts >= 0.0)
            & jnp.all(state.last_update >= 0)
            & (state.step_count >= 0)
            & jnp.all(state.last_update <= state.step_count)
        )

    def init(self) -> PrototypeMemoryState:
        """Create an empty fixed-budget memory."""
        c = self._config
        return PrototypeMemoryState(
            means=jnp.zeros(
                (c.n_classes, c.slots_per_class, c.feature_dim),
                dtype=jnp.float32,
            ),
            counts=jnp.zeros((c.n_classes, c.slots_per_class), dtype=jnp.float32),
            last_update=jnp.zeros((c.n_classes, c.slots_per_class), dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def class_logits(
        self,
        state: PrototypeMemoryState,
        observation: Float[Array, " feature_dim"],
    ) -> Float[Array, " n_classes"]:
        """Return class logits from nearest active prototype distances."""
        self._validate_state_static_contract(state)
        x = jnp.asarray(observation, dtype=jnp.float32)
        _require_shape("observation", x, (self._config.feature_dim,))
        diffs = state.means - x[None, None, :]
        distances = jnp.mean(diffs * diffs, axis=2)
        slot_logits = -distances / jnp.asarray(self._config.bandwidth, dtype=jnp.float32)
        slot_logits = jnp.where(state.counts > 0.0, slot_logits, -jnp.inf)
        logits = jnp.max(slot_logits, axis=1)
        any_active = jnp.any(state.counts > 0.0, axis=1)
        logits = jnp.where(any_active, logits, -1e9)
        logits = jnp.where(jnp.any(any_active), logits, jnp.zeros_like(logits))
        inputs_valid = self._state_is_valid(state) & jnp.all(jnp.isfinite(x))
        return jnp.where(inputs_valid, logits, jnp.full_like(logits, jnp.nan))

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: PrototypeMemoryState,
        observation: Float[Array, " feature_dim"],
    ) -> Float[Array, " n_classes"]:
        """Return class probabilities for one observation."""
        return _softmax(self.class_logits(state, observation))

    @staticmethod
    def valid_one_hot_target(target: Array) -> Array:
        """Return whether ``target`` is a finite one-hot/simplex target."""
        finite = jnp.all(jnp.isfinite(target))
        target_sum = jnp.sum(target)
        max_target = jnp.max(target)
        non_negative = jnp.all(target >= -1e-6)
        return finite & non_negative & (jnp.abs(target_sum - 1.0) <= 1e-5) & (
            max_target >= 0.999
        )

    def _replacement_slot(self, state: PrototypeMemoryState, head: Array) -> Array:
        """Choose least-used, then oldest, slot for a full class budget."""
        class_counts = state.counts[head]
        class_last_update = state.last_update[head]
        min_count = jnp.min(class_counts)
        tied = class_counts <= (min_count + 1e-6)
        oldest_among_tied = jnp.where(
            tied,
            class_last_update,
            jnp.array(2_147_483_647, dtype=class_last_update.dtype),
        )
        return jnp.argmin(oldest_among_tied)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_with_novelty_threshold(
        self,
        state: PrototypeMemoryState,
        observation: Float[Array, " feature_dim"],
        target: Float[Array, " n_classes"],
        novelty_threshold: Float[Array, ""],
    ) -> PrototypeMemoryUpdateResult:
        """Perform one causal update with a runtime novelty threshold."""
        self._validate_state_static_contract(state)
        observation_arr = jnp.asarray(observation, dtype=jnp.float32)
        target_arr = jnp.asarray(target, dtype=jnp.float32)
        threshold_arr = jnp.asarray(novelty_threshold, dtype=jnp.float32)
        _require_shape("observation", observation_arr, (self._config.feature_dim,))
        _require_shape("target", target_arr, (self._config.n_classes,))
        _require_shape("novelty_threshold", threshold_arr, ())
        inputs_valid = (
            jnp.all(jnp.isfinite(observation_arr))
            & jnp.isfinite(threshold_arr)
            & (threshold_arr >= 0.0)
        )
        safe_observation = jnp.where(
            inputs_valid, observation_arr, jnp.zeros_like(observation_arr)
        )
        prediction = self.predict(state, safe_observation)
        valid_target = self.valid_one_hot_target(target_arr)
        safe_target = jnp.where(jnp.isfinite(target_arr), target_arr, 0.0)
        errors = prediction - safe_target
        mse = jnp.mean(errors * errors)
        confidence = jnp.max(prediction)
        correct = jnp.where(
            valid_target,
            (jnp.argmax(prediction) == jnp.argmax(safe_target)).astype(jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
        )

        def do_update(current: PrototypeMemoryState) -> tuple[PrototypeMemoryState, Array]:
            head = jnp.argmax(safe_target)
            used = current.counts[head] > 0.0
            has_used = jnp.any(used)
            has_empty = jnp.any(~used)
            distances = jnp.mean(
                (current.means[head] - safe_observation[None, :]) ** 2,
                axis=1,
            )
            used_distances = jnp.where(used, distances, jnp.inf)
            nearest_slot = jnp.argmin(used_distances)
            nearest_distance = used_distances[nearest_slot]
            empty_slot = jnp.argmax((~used).astype(jnp.int32))
            replacement_slot = self._replacement_slot(current, head)
            novel = (~has_used) | (nearest_distance > threshold_arr)
            slot = jnp.where(
                ~has_used,
                jnp.array(0, dtype=nearest_slot.dtype),
                jnp.where(
                    novel & has_empty,
                    empty_slot,
                    jnp.where(novel, replacement_slot, nearest_slot),
                ),
            )
            old_mean = current.means[head, slot]
            eta = jnp.asarray(self._config.update_rate, dtype=jnp.float32)
            new_mean = jnp.where(
                novel,
                safe_observation,
                old_mean + eta * (safe_observation - old_mean),
            )
            new_count = jnp.where(novel, 1.0, current.counts[head, slot] + 1.0)
            next_step = _saturating_increment(current.step_count)
            next_state = PrototypeMemoryState(
                means=current.means.at[head, slot].set(new_mean),
                counts=current.counts.at[head, slot].set(new_count),
                last_update=current.last_update.at[head, slot].set(next_step),
                step_count=next_step,
            )
            return next_state, novel.astype(jnp.float32)

        def skip_update(current: PrototypeMemoryState) -> tuple[PrototypeMemoryState, Array]:
            next_step = _saturating_increment(current.step_count)
            return (
                PrototypeMemoryState(
                    means=current.means,
                    counts=current.counts,
                    last_update=current.last_update,
                    step_count=next_step,
                ),
                jnp.array(0.0, dtype=jnp.float32),
            )

        candidate_state, allocated = jax.lax.cond(
            valid_target, do_update, skip_update, state
        )
        active = jnp.sum(candidate_state.counts > 0.0).astype(jnp.float32)
        metrics = jnp.asarray(
            [
                mse,
                correct,
                confidence,
                active,
                valid_target.astype(jnp.float32),
                allocated,
            ],
            dtype=jnp.float32,
        )
        update_applied = (
            inputs_valid
            & self._state_is_valid(state)
            & self._state_is_valid(candidate_state)
            & jnp.all(jnp.isfinite(prediction))
            & jnp.all(jnp.isfinite(errors))
            & jnp.all(jnp.isfinite(metrics))
        )
        return PrototypeMemoryUpdateResult(
            state=select_transaction(update_applied, candidate_state, state),
            predictions=neutralize_array(update_applied, prediction),
            errors=neutralize_array(update_applied, errors),
            metrics=neutralize_array(update_applied, metrics),
            update_applied=update_applied,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: PrototypeMemoryState,
        observation: Float[Array, " feature_dim"],
        target: Float[Array, " n_classes"],
    ) -> PrototypeMemoryUpdateResult:
        """Perform one causal online memory update."""
        return cast(
            PrototypeMemoryUpdateResult,
            self.update_with_novelty_threshold(
                state,
                observation,
                target,
                jnp.asarray(self._config.novelty_threshold, dtype=jnp.float32),
            ),
        )


def run_prototype_memory_arrays(
    learner: PrototypeMemoryLearner,
    observations: Float[Array, "steps feature_dim"],
    targets: Float[Array, "steps n_classes"],
    *,
    state: PrototypeMemoryState | None = None,
) -> PrototypeMemoryLearningResult:
    """Run the prototype memory over arrays with ``jax.lax.scan``.

    Metric columns are ``mse, correct, confidence, active_prototypes,
    valid_update, allocated``.
    """
    observation_array = jnp.asarray(observations, dtype=jnp.float32)
    target_array = jnp.asarray(targets, dtype=jnp.float32)
    if observation_array.ndim != 2 or observation_array.shape[1] != learner.config.feature_dim:
        raise ValueError(
            "observations must have shape "
            f"(steps, {learner.config.feature_dim}); got {tuple(observation_array.shape)}"
        )
    if target_array.ndim != 2 or target_array.shape[1] != learner.config.n_classes:
        raise ValueError(
            "targets must have shape "
            f"(steps, {learner.config.n_classes}); got {tuple(target_array.shape)}"
        )
    if observation_array.shape[0] != target_array.shape[0]:
        raise ValueError("observations and targets must have the same step count")
    steps = int(observation_array.shape[0])
    _require_sequence_resource(
        "prototype memory scan outputs",
        float32_scalars=steps * (learner.config.n_classes + 6),
        bool_scalars=steps,
    )
    if state is None:
        state = learner.init()
    learner._validate_state_static_contract(state)

    def step_fn(
        carry: PrototypeMemoryState,
        batch: tuple[Array, Array],
    ) -> tuple[PrototypeMemoryState, tuple[Array, Array, Array]]:
        observation, target = batch
        result = learner.update(carry, observation, target)
        return result.state, (
            result.predictions,
            result.metrics,
            result.update_applied,
        )

    final_state, (predictions, metrics, updates_applied) = jax.lax.scan(
        step_fn,
        state,
        (observation_array, target_array),
    )
    return PrototypeMemoryLearningResult(
        state=final_state,
        predictions=predictions,
        metrics=metrics,
        updates_applied=updates_applied,
    )
