# mypy: disable-error-code="call-arg"
"""Causal, typed learning-signal estimates from ensemble predictions.

This module is a small development-only mechanism for typed surprise,
learning progress, and change-detection channels. It deliberately does not
combine its outputs into a reward, objective,
priority, or generic score.  A consumer must choose a named signal and
preserve its units.

The causal contract is strict.  At time ``t`` a caller must:

1. obtain all ensemble means and aleatoric variances before updating the
   ensemble on the current target;
2. observe the target and the corresponding pre-update loss; and
3. call :meth:`LearningSignalEstimator.observe`.

The returned signals use only those predict-before-update values and state
from earlier calls.  The returned state incorporates the current residual and
loss for use at later times.  Passing post-update predictions violates the
contract and would make learning progress and surprise optimistic.

Signal units are explicit:

* epistemic disagreement is a population variance in squared target units;
* epistemic surprise is disagreement divided by predicted aleatoric variance,
  and is dimensionless;
* aleatoric uncertainty is predicted variance in squared target units;
* normalized residual is squared prediction error divided by total predicted
  variance, and is dimensionless;
* learning progress is slow-window loss minus fast-window loss, in the
  caller's observed-loss units; and
* change probability is a dimensionless probability in ``[0, 1]``.

The change detector first freezes a Welford calibration of normalized
residuals, maps later calibrated residual z-scores through a configured
logistic curve, and then exponentially smooths those instantaneous
probabilities.  This is an internally calibrated detector, not evidence of
calibration on an external environment or a scientific-result claim.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core._float32_scalars import validated_float32_scalar

_INT32_MAX = 2_147_483_647
_ACTUAL_INT_TYPES = frozenset(
    {
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
    }
)


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _skip_zero_scale(scale: Array, value: Array) -> Array:
    """Skip ``0 * inf`` so a disabled EMA decay does not poison the next value."""
    return jnp.where(scale == 0.0, jnp.zeros_like(value), scale * value)


def _saturating_increment(value: Array) -> Array:
    """Increment a non-negative int32 counter without lifetime wraparound."""
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return jnp.minimum(value, maximum - 1) + jnp.asarray(1, dtype=jnp.int32)


def _saturating_counter_sum(left: Array, right: Array) -> Array:
    """Add non-negative int32 counters without overflowing."""
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return left + jnp.minimum(right, maximum - left)


def _positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


def _unit_interval(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1)")


def _positive_integer(name: str, value: int, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclasses.dataclass(frozen=True)
class LearningSignalEstimatorConfig:
    """Static shape, timescale, calibration, and numerical-safety contract.

    ``fast_loss_decay`` must be smaller than ``slow_loss_decay`` so the former
    reacts more quickly.  ``change_calibration_steps`` valid residuals are used
    only for calibration; change probabilities become available on the next
    valid observation.  ``change_decay`` controls how much persistence is
    required: a single observation can contribute at most
    ``1 - change_decay`` to the sustained probability.

    The magnitude bounds reject otherwise finite inputs that could overflow
    float32 second moments.  ``max_normalized_residual`` is a documented
    saturation bound for dimensionless diagnostics and detector inputs.
    """

    ensemble_size: int
    target_dim: int
    variance_floor: float = 1.0e-6
    fast_loss_decay: float = 0.8
    slow_loss_decay: float = 0.99
    progress_warmup_steps: int = 2
    change_calibration_steps: int = 16
    change_z_threshold: float = 3.0
    change_temperature: float = 0.5
    change_decay: float = 0.95
    calibration_scale_floor: float = 0.25
    max_normalized_residual: float = 1.0e6
    max_input_magnitude: float = 1.0e12
    max_predicted_variance: float = 1.0e24
    max_observed_loss: float = 1.0e24

    def __post_init__(self) -> None:
        """Reject invalid static shapes, timescales, and safety bounds."""
        object.__setattr__(
            self,
            "ensemble_size",
            _require_int32("ensemble_size", self.ensemble_size, minimum=2),
        )
        object.__setattr__(
            self,
            "target_dim",
            _require_int32("target_dim", self.target_dim, minimum=1),
        )
        object.__setattr__(
            self,
            "progress_warmup_steps",
            _require_int32(
                "progress_warmup_steps",
                self.progress_warmup_steps,
                minimum=2,
            ),
        )
        object.__setattr__(
            self,
            "change_calibration_steps",
            _require_int32(
                "change_calibration_steps",
                self.change_calibration_steps,
                minimum=2,
            ),
        )
        if self.change_calibration_steps >= _INT32_MAX:
            raise ValueError("change_calibration_steps must fit in int32")
        _positive_finite("variance_floor", self.variance_floor)
        fast_loss_decay = validated_float32_scalar(
            "fast_loss_decay",
            self.fast_loss_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        slow_loss_decay = validated_float32_scalar(
            "slow_loss_decay",
            self.slow_loss_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        if fast_loss_decay >= slow_loss_decay:
            raise ValueError("fast_loss_decay must be smaller than slow_loss_decay")
        change_decay = validated_float32_scalar(
            "change_decay",
            self.change_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        object.__setattr__(self, "fast_loss_decay", fast_loss_decay)
        object.__setattr__(self, "slow_loss_decay", slow_loss_decay)
        object.__setattr__(self, "change_decay", change_decay)
        _positive_finite("change_z_threshold", self.change_z_threshold)
        _positive_finite("change_temperature", self.change_temperature)
        _positive_finite("calibration_scale_floor", self.calibration_scale_floor)
        _positive_finite("max_normalized_residual", self.max_normalized_residual)
        _positive_finite("max_input_magnitude", self.max_input_magnitude)
        _positive_finite("max_predicted_variance", self.max_predicted_variance)
        _positive_finite("max_observed_loss", self.max_observed_loss)
        if self.variance_floor > self.max_predicted_variance:
            raise ValueError("variance_floor must not exceed max_predicted_variance")

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration."""
        payload = dataclasses.asdict(self)
        payload["type"] = "LearningSignalEstimatorConfig"
        payload["development_only"] = True
        payload["accepted_scientific_evidence"] = False
        return payload

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
    ) -> LearningSignalEstimatorConfig:
        """Reconstruct a configuration and reject a mismatched type marker."""
        payload = dict(config)
        type_name = payload.pop("type", "LearningSignalEstimatorConfig")
        if type_name != "LearningSignalEstimatorConfig":
            raise ValueError("type must be LearningSignalEstimatorConfig")
        development_only = payload.pop("development_only", True)
        if development_only is not True:
            raise ValueError("learning signal estimator is development_only")
        accepted_evidence = payload.pop("accepted_scientific_evidence", False)
        if accepted_evidence is not False:
            raise ValueError("learning signal estimator is not accepted scientific evidence")
        return cls(**payload)


@dataclasses.dataclass(frozen=True)
class LearningSignalResourceBudget:
    """Exact logical scalar and byte counts.

    Counts exclude transient compiler buffers and device-specific alignment.
    Persistent state contains four int32 and five float32 scalars.  Output
    contains eight float32 values and six boolean flags.
    """

    input_float_scalars_per_step: int
    persistent_float32_scalars: int
    persistent_int32_scalars: int
    persistent_state_scalars: int
    persistent_state_bytes: int
    output_float32_scalars: int
    output_bool_scalars: int
    output_logical_bytes: int
    trainable_scalars: int

    def to_config(self) -> dict[str, int]:
        """Return a JSON-compatible budget description."""
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class LearningSignalEstimatorState:
    """Fixed-size causal state for loss windows and change calibration."""

    step_count: Int[Array, ""]
    valid_count: Int[Array, ""]
    invalid_count: Int[Array, ""]
    calibration_count: Int[Array, ""]
    calibration_mean: Float[Array, ""]
    calibration_m2: Float[Array, ""]
    fast_loss_ema: Float[Array, ""]
    slow_loss_ema: Float[Array, ""]
    sustained_change_probability: Float[Array, ""]


@chex.dataclass(frozen=True)
class LearningSignalAvailability:
    """Named availability flags; there is intentionally no aggregate score."""

    input_valid: Bool[Array, ""]
    epistemic: Bool[Array, ""]
    aleatoric: Bool[Array, ""]
    normalized_residual: Bool[Array, ""]
    learning_progress: Bool[Array, ""]
    change_probability: Bool[Array, ""]


@chex.dataclass(frozen=True)
class TypedLearningSignals:
    """Separately typed signal values from one predict-before-update event."""

    epistemic_disagreement: Float[Array, ""]
    epistemic_surprise: Float[Array, ""]
    aleatoric_uncertainty: Float[Array, ""]
    normalized_residual: Float[Array, ""]
    learning_progress: Float[Array, ""]
    calibrated_residual_z: Float[Array, ""]
    instantaneous_change_probability: Float[Array, ""]
    change_probability: Float[Array, ""]
    availability: LearningSignalAvailability


class LearningSignalEstimator:
    """Fixed-state producer for typed, predict-before-update learning signals."""

    def __init__(self, config: LearningSignalEstimatorConfig):
        self._config = config

    @property
    def config(self) -> LearningSignalEstimatorConfig:
        """Return the immutable estimator configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize the estimator's static configuration."""
        return self._config.to_config()

    def resource_budget(self) -> LearningSignalResourceBudget:
        """Return exact logical resource counts for this implementation."""
        input_scalars = (
            2 * self._config.ensemble_size * self._config.target_dim + self._config.target_dim + 1
        )
        return LearningSignalResourceBudget(
            input_float_scalars_per_step=input_scalars,
            persistent_float32_scalars=5,
            persistent_int32_scalars=4,
            persistent_state_scalars=9,
            persistent_state_bytes=36,
            output_float32_scalars=8,
            output_bool_scalars=6,
            output_logical_bytes=38,
            trainable_scalars=0,
        )

    def init(self) -> LearningSignalEstimatorState:
        """Return a zeroed fixed-shape state."""
        zero_float = jnp.asarray(0.0, dtype=jnp.float32)
        zero_int = jnp.asarray(0, dtype=jnp.int32)
        return LearningSignalEstimatorState(
            step_count=zero_int,
            valid_count=zero_int,
            invalid_count=zero_int,
            calibration_count=zero_int,
            calibration_mean=zero_float,
            calibration_m2=zero_float,
            fast_loss_ema=zero_float,
            slow_loss_ema=zero_float,
            sustained_change_probability=zero_float,
        )

    @staticmethod
    def _floating_array(value: Array, shape: tuple[int, ...], *, name: str) -> Array:
        array = jnp.asarray(value)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            raise ValueError(f"{name} must have a floating dtype")
        return jnp.asarray(array, dtype=jnp.float32)

    @staticmethod
    def _validate_state_shapes(state: LearningSignalEstimatorState) -> None:
        integer_values = {
            "step_count": state.step_count,
            "valid_count": state.valid_count,
            "invalid_count": state.invalid_count,
            "calibration_count": state.calibration_count,
        }
        float_values = {
            "calibration_mean": state.calibration_mean,
            "calibration_m2": state.calibration_m2,
            "fast_loss_ema": state.fast_loss_ema,
            "slow_loss_ema": state.slow_loss_ema,
            "sustained_change_probability": state.sustained_change_probability,
        }
        for name, value in integer_values.items():
            array = jnp.asarray(value)
            if array.shape != ():
                raise ValueError(f"state.{name} must be scalar")
            if not jnp.issubdtype(array.dtype, jnp.integer):
                raise ValueError(f"state.{name} must have an integer dtype")
        for name, value in float_values.items():
            array = jnp.asarray(value)
            if array.shape != ():
                raise ValueError(f"state.{name} must be scalar")
            if not jnp.issubdtype(array.dtype, jnp.inexact):
                raise ValueError(f"state.{name} must have a floating dtype")

    def observe(
        self,
        state: LearningSignalEstimatorState,
        member_means: Array,
        predicted_aleatoric_variances: Array,
        observed_target: Array,
        observed_loss: Array | float,
    ) -> tuple[LearningSignalEstimatorState, TypedLearningSignals]:
        """Consume one predict-before-update ensemble event.

        ``member_means`` and ``predicted_aleatoric_variances`` both have shape
        ``(ensemble_size, target_dim)``.  ``observed_target`` has shape
        ``(target_dim,)`` and ``observed_loss`` is a non-negative scalar in the
        caller's native loss units.

        Runtime non-finite values, negative variances/losses, excessive
        magnitudes, or a corrupt state fail closed: all availability flags and
        signal values are zero.  A valid state still records an invalid input
        attempt in ``step_count`` and ``invalid_count`` without changing any
        calibration or EMA statistic.
        """
        self._validate_state_shapes(state)
        means = self._floating_array(
            member_means,
            (self._config.ensemble_size, self._config.target_dim),
            name="member_means",
        )
        variances = self._floating_array(
            predicted_aleatoric_variances,
            (self._config.ensemble_size, self._config.target_dim),
            name="predicted_aleatoric_variances",
        )
        target = self._floating_array(
            observed_target,
            (self._config.target_dim,),
            name="observed_target",
        )
        loss = self._floating_array(jnp.asarray(observed_loss), (), name="observed_loss")

        checked_fast_loss = (
            jnp.zeros_like(state.fast_loss_ema)
            if self._config.fast_loss_decay == 0.0
            else state.fast_loss_ema
        )
        checked_slow_loss = (
            jnp.zeros_like(state.slow_loss_ema)
            if self._config.slow_loss_decay == 0.0
            else state.slow_loss_ema
        )
        checked_change = (
            jnp.zeros_like(state.sustained_change_probability)
            if self._config.change_decay == 0.0
            else state.sustained_change_probability
        )
        state_valid = (
            (state.step_count >= 0)
            & (state.valid_count >= 0)
            & (state.invalid_count >= 0)
            & (
                state.step_count
                == _saturating_counter_sum(
                    state.valid_count,
                    state.invalid_count,
                )
            )
            & (state.calibration_count >= 0)
            & (state.calibration_count <= self._config.change_calibration_steps)
            & (state.calibration_count <= state.valid_count)
            & jnp.isfinite(state.calibration_mean)
            & (state.calibration_mean >= 0.0)
            & (state.calibration_mean <= self._config.max_normalized_residual)
            & jnp.isfinite(state.calibration_m2)
            & (state.calibration_m2 >= 0.0)
            & jnp.isfinite(checked_fast_loss)
            & (checked_fast_loss >= 0.0)
            & (checked_fast_loss <= self._config.max_observed_loss)
            & jnp.isfinite(checked_slow_loss)
            & (checked_slow_loss >= 0.0)
            & (checked_slow_loss <= self._config.max_observed_loss)
            & jnp.isfinite(checked_change)
            & (checked_change >= 0.0)
            & (checked_change <= 1.0)
        )
        input_valid = (
            jnp.all(jnp.isfinite(means))
            & jnp.all(jnp.isfinite(variances))
            & jnp.all(variances >= 0.0)
            & jnp.all(jnp.isfinite(target))
            & jnp.isfinite(loss)
            & (loss >= 0.0)
            & jnp.all(jnp.abs(means) <= self._config.max_input_magnitude)
            & jnp.all(jnp.abs(target) <= self._config.max_input_magnitude)
            & jnp.all(variances <= self._config.max_predicted_variance)
            & (loss <= self._config.max_observed_loss)
        )
        event_valid = state_valid & input_valid

        # Sanitizing before arithmetic prevents invalid branches from producing
        # NaNs/Infs that could escape through compiler transformations.
        safe_means = jnp.nan_to_num(
            means,
            nan=0.0,
            posinf=self._config.max_input_magnitude,
            neginf=-self._config.max_input_magnitude,
        )
        safe_means = jnp.clip(
            safe_means,
            -self._config.max_input_magnitude,
            self._config.max_input_magnitude,
        )
        safe_variances = jnp.nan_to_num(
            variances,
            nan=0.0,
            posinf=self._config.max_predicted_variance,
            neginf=0.0,
        )
        safe_variances = jnp.clip(
            safe_variances,
            0.0,
            self._config.max_predicted_variance,
        )
        safe_target = jnp.nan_to_num(
            target,
            nan=0.0,
            posinf=self._config.max_input_magnitude,
            neginf=-self._config.max_input_magnitude,
        )
        safe_target = jnp.clip(
            safe_target,
            -self._config.max_input_magnitude,
            self._config.max_input_magnitude,
        )
        safe_loss = jnp.nan_to_num(
            loss,
            nan=0.0,
            posinf=self._config.max_observed_loss,
            neginf=0.0,
        )
        safe_loss = jnp.clip(safe_loss, 0.0, self._config.max_observed_loss)

        ensemble_mean = jnp.mean(safe_means, axis=0)
        per_dimension_epistemic = jnp.mean(
            jnp.square(safe_means - ensemble_mean[None, :]),
            axis=0,
        )
        per_dimension_aleatoric = jnp.mean(safe_variances, axis=0)
        epistemic_disagreement = jnp.mean(per_dimension_epistemic)
        epistemic_surprise = jnp.mean(
            per_dimension_epistemic
            / jnp.maximum(per_dimension_aleatoric, self._config.variance_floor)
        )
        epistemic_surprise = jnp.minimum(
            epistemic_surprise,
            self._config.max_normalized_residual,
        )
        aleatoric_uncertainty = jnp.mean(per_dimension_aleatoric)
        total_variance = jnp.maximum(
            per_dimension_epistemic + per_dimension_aleatoric,
            self._config.variance_floor,
        )
        normalized_residual = jnp.mean(jnp.square(safe_target - ensemble_mean) / total_variance)
        normalized_residual = jnp.minimum(
            normalized_residual,
            self._config.max_normalized_residual,
        )

        first_valid_event = state.valid_count == 0
        fast_decay = jnp.asarray(self._config.fast_loss_decay, dtype=jnp.float32)
        slow_decay = jnp.asarray(self._config.slow_loss_decay, dtype=jnp.float32)
        change_decay = jnp.asarray(self._config.change_decay, dtype=jnp.float32)
        fast_loss = jnp.where(
            first_valid_event,
            safe_loss,
            _skip_zero_scale(fast_decay, state.fast_loss_ema)
            + (1.0 - fast_decay) * safe_loss,
        )
        slow_loss = jnp.where(
            first_valid_event,
            safe_loss,
            _skip_zero_scale(slow_decay, state.slow_loss_ema)
            + (1.0 - slow_decay) * safe_loss,
        )
        learning_progress = slow_loss - fast_loss
        next_valid_count = _saturating_increment(state.valid_count)
        progress_available = event_valid & (next_valid_count >= self._config.progress_warmup_steps)

        calibration_ready = state.calibration_count >= self._config.change_calibration_steps
        calibrating = ~calibration_ready
        next_calibration_count = _saturating_increment(
            state.calibration_count
        )
        calibration_delta = normalized_residual - state.calibration_mean
        next_calibration_mean = (
            state.calibration_mean + calibration_delta / next_calibration_count.astype(jnp.float32)
        )
        next_calibration_m2 = state.calibration_m2 + calibration_delta * (
            normalized_residual - next_calibration_mean
        )

        calibration_denominator = jnp.maximum(
            state.calibration_count - jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
        ).astype(jnp.float32)
        calibration_variance = state.calibration_m2 / calibration_denominator
        calibration_scale = jnp.maximum(
            jnp.sqrt(jnp.maximum(calibration_variance, 0.0)),
            self._config.calibration_scale_floor,
        )
        calibrated_residual_z = (normalized_residual - state.calibration_mean) / calibration_scale
        calibrated_residual_z = jnp.clip(
            calibrated_residual_z,
            -self._config.max_normalized_residual,
            self._config.max_normalized_residual,
        )
        instantaneous_change_probability = jax.nn.sigmoid(
            (calibrated_residual_z - self._config.change_z_threshold)
            / self._config.change_temperature
        )
        sustained_change_probability = (
            _skip_zero_scale(change_decay, state.sustained_change_probability)
            + (1.0 - change_decay) * instantaneous_change_probability
        )
        change_available = event_valid & calibration_ready

        zero = jnp.asarray(0.0, dtype=jnp.float32)

        def available_value(value: Array, available: Array) -> Array:
            finite_value = jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return jnp.where(available, finite_value, zero)

        immediate_available = event_valid
        signals = TypedLearningSignals(
            epistemic_disagreement=available_value(
                epistemic_disagreement,
                immediate_available,
            ),
            epistemic_surprise=available_value(
                epistemic_surprise,
                immediate_available,
            ),
            aleatoric_uncertainty=available_value(
                aleatoric_uncertainty,
                immediate_available,
            ),
            normalized_residual=available_value(
                normalized_residual,
                immediate_available,
            ),
            learning_progress=available_value(
                learning_progress,
                progress_available,
            ),
            calibrated_residual_z=available_value(
                calibrated_residual_z,
                change_available,
            ),
            instantaneous_change_probability=available_value(
                instantaneous_change_probability,
                change_available,
            ),
            change_probability=available_value(
                sustained_change_probability,
                change_available,
            ),
            availability=LearningSignalAvailability(
                input_valid=event_valid,
                epistemic=immediate_available,
                aleatoric=immediate_available,
                normalized_residual=immediate_available,
                learning_progress=progress_available,
                change_probability=change_available,
            ),
        )

        valid_calibration_update = event_valid & calibrating
        next_state_if_valid = LearningSignalEstimatorState(
            step_count=_saturating_increment(state.step_count),
            valid_count=next_valid_count,
            invalid_count=state.invalid_count,
            calibration_count=jnp.where(
                valid_calibration_update,
                next_calibration_count,
                state.calibration_count,
            ),
            calibration_mean=jnp.where(
                valid_calibration_update,
                next_calibration_mean,
                state.calibration_mean,
            ),
            calibration_m2=jnp.where(
                valid_calibration_update,
                jnp.maximum(next_calibration_m2, 0.0),
                state.calibration_m2,
            ),
            fast_loss_ema=fast_loss,
            slow_loss_ema=slow_loss,
            sustained_change_probability=jnp.where(
                change_available,
                jnp.clip(sustained_change_probability, 0.0, 1.0),
                state.sustained_change_probability,
            ),
        )
        next_state_if_invalid_input = LearningSignalEstimatorState(
            step_count=_saturating_increment(state.step_count),
            valid_count=state.valid_count,
            invalid_count=_saturating_increment(state.invalid_count),
            calibration_count=state.calibration_count,
            calibration_mean=state.calibration_mean,
            calibration_m2=state.calibration_m2,
            fast_loss_ema=state.fast_loss_ema,
            slow_loss_ema=state.slow_loss_ema,
            sustained_change_probability=state.sustained_change_probability,
        )
        next_state = jax.tree_util.tree_map(
            lambda valid_value, invalid_value, corrupt_value: jnp.where(
                state_valid,
                jnp.where(input_valid, valid_value, invalid_value),
                corrupt_value,
            ),
            next_state_if_valid,
            next_state_if_invalid_input,
            state,
        )
        return next_state, signals

    def scan(
        self,
        state: LearningSignalEstimatorState,
        member_means: Array,
        predicted_aleatoric_variances: Array,
        observed_targets: Array,
        observed_losses: Array,
    ) -> tuple[LearningSignalEstimatorState, TypedLearningSignals]:
        """Process a fixed-shape sequence with :func:`jax.lax.scan`."""
        means = jnp.asarray(member_means)
        variances = jnp.asarray(predicted_aleatoric_variances)
        targets = jnp.asarray(observed_targets)
        losses = jnp.asarray(observed_losses)
        if means.ndim != 3:
            raise ValueError("member_means sequence must have rank 3")
        num_steps = means.shape[0]
        expected_ensemble_shape = (
            num_steps,
            self._config.ensemble_size,
            self._config.target_dim,
        )
        if means.shape != expected_ensemble_shape:
            raise ValueError(
                "member_means sequence must have shape "
                f"{expected_ensemble_shape}, got {means.shape}"
            )
        if variances.shape != expected_ensemble_shape:
            raise ValueError(
                "predicted_aleatoric_variances sequence must have shape "
                f"{expected_ensemble_shape}, got {variances.shape}"
            )
        expected_target_shape = (num_steps, self._config.target_dim)
        if targets.shape != expected_target_shape:
            raise ValueError(
                f"observed_targets must have shape {expected_target_shape}, got {targets.shape}"
            )
        if losses.shape != (num_steps,):
            raise ValueError(f"observed_losses must have shape {(num_steps,)}, got {losses.shape}")

        def scan_step(
            carry: LearningSignalEstimatorState,
            inputs: tuple[Array, Array, Array, Array],
        ) -> tuple[LearningSignalEstimatorState, TypedLearningSignals]:
            next_state, signal = self.observe(carry, *inputs)
            return next_state, signal

        return jax.lax.scan(
            scan_step,
            state,
            (means, variances, targets, losses),
        )


__all__ = [
    "LearningSignalAvailability",
    "LearningSignalEstimator",
    "LearningSignalEstimatorConfig",
    "LearningSignalEstimatorState",
    "LearningSignalResourceBudget",
    "TypedLearningSignals",
]
