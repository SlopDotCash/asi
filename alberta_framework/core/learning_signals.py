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
import operator
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core._float32_scalars import (
    validated_float32_scalar,
    validated_float32_scalar_with_ratio,
)
from alberta_framework.core.normalizers import _saturating_int32_counter_increment

_INT32_MAX = 2_147_483_647
_FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
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


def _learning_signal_observe_working_set_bytes(
    ensemble_size: int,
    target_dim: int,
) -> int:
    """Source persist, proposed persist, inputs, sanitized copies, and returned leaves.

    ``observe`` keeps the source state, the valid-path proposal, and the
    invalid-path proposal live together with the raw ensemble inputs, their
    sanitized copies, member residual squares, per-dimension temps, and the
    returned signal / availability leaves.
    """

    persist_scalars = 9
    input_scalars = 2 * ensemble_size * target_dim + target_dim + 1
    observe_scalars = (
        3 * persist_scalars
        + 2 * input_scalars
        + ensemble_size * target_dim
        + 5 * target_dim
        + 16
    )
    return 4 * observe_scalars


def _preflight_learning_signal_observe_working_set(
    ensemble_size: int,
    target_dim: int,
) -> None:
    """Reject an observe envelope the estimator cannot name."""

    working_set_bytes = _learning_signal_observe_working_set_bytes(
        ensemble_size,
        target_dim,
    )
    if working_set_bytes > _INT32_MAX:
        raise ValueError(
            "learning-signal observe working set byte count must fit signed int32"
        )


def _ratio_less(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] * right[1] < right[0] * left[1]


def _ratio_less_equal(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] * right[1] <= right[0] * left[1]


def _stable_mean(values: Array, *, axis: int | None = None) -> Array:
    """Average bounded float32 values without overflowing the reduction sum."""
    count = values.size if axis is None else values.shape[axis]
    return jnp.sum(values / jnp.asarray(count, dtype=jnp.float32), axis=axis)


def _skip_zero_scale(scale: Array, value: Array) -> Array:
    """Skip ``0 * inf`` so a disabled EMA decay does not poison the next value."""
    return jnp.where(scale == 0.0, jnp.zeros_like(value), scale * value)


def _saturating_increment(value: Array) -> Array:
    return _saturating_int32_counter_increment(value)


def _saturating_counter_sum(left: Array, right: Array) -> Array:
    """Add non-negative int32 counters without overflowing."""
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return left + jnp.minimum(right, maximum - left)


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
        input_scalars = 2 * self.ensemble_size * self.target_dim + self.target_dim + 1
        if input_scalars > _INT32_MAX:
            raise ValueError(
                "ensemble_size and target_dim input resource budget must fit signed int32"
            )
        _preflight_learning_signal_observe_working_set(
            self.ensemble_size,
            self.target_dim,
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
                maximum=_INT32_MAX - 1,
            ),
        )
        positive_values = {
            field_name: validated_float32_scalar_with_ratio(
                field_name, getattr(self, field_name), positive=True
            )
            for field_name in (
                "variance_floor",
                "change_z_threshold",
                "change_temperature",
                "calibration_scale_floor",
                "max_normalized_residual",
                "max_input_magnitude",
                "max_predicted_variance",
                "max_observed_loss",
            )
        }
        fast_loss_decay, fast_n, fast_d = validated_float32_scalar_with_ratio(
            "fast_loss_decay",
            self.fast_loss_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        slow_loss_decay, slow_n, slow_d = validated_float32_scalar_with_ratio(
            "slow_loss_decay",
            self.slow_loss_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        if not _ratio_less((fast_n, fast_d), (slow_n, slow_d)) or not (
            fast_loss_decay < slow_loss_decay
        ):
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
        for field_name, (stored, _, _) in positive_values.items():
            object.__setattr__(self, field_name, stored)
        _, variance_floor_n, variance_floor_d = positive_values["variance_floor"]
        _, max_variance_n, max_variance_d = positive_values["max_predicted_variance"]
        if not _ratio_less_equal(
            (variance_floor_n, variance_floor_d),
            (max_variance_n, max_variance_d),
        ) or self.variance_floor > self.max_predicted_variance:
            raise ValueError("variance_floor must not exceed max_predicted_variance")
        max_input, max_input_n, max_input_d = positive_values["max_input_magnitude"]
        float32_max_n, float32_max_d = _FLOAT32_MAX.as_integer_ratio()
        total_variance_left = (
            4 * max_input_n * max_input_n * max_variance_d
            + max_variance_n * max_input_d * max_input_d
        ) * float32_max_d
        total_variance_right = (
            float32_max_n * max_input_d * max_input_d * max_variance_d
        )
        if total_variance_left > total_variance_right or (
            4.0 * max_input * max_input + self.max_predicted_variance > _FLOAT32_MAX
        ):
            raise ValueError(
                "four times max_input_magnitude squared plus "
                "max_predicted_variance must fit float32"
            )
        max_residual, max_residual_n, max_residual_d = positive_values[
            "max_normalized_residual"
        ]
        calibration_left = (
            _INT32_MAX * max_residual_n * max_residual_n * float32_max_d
        )
        calibration_right = float32_max_n * max_residual_d * max_residual_d
        if calibration_left > calibration_right or (
            _INT32_MAX * max_residual * max_residual > _FLOAT32_MAX
        ):
            raise ValueError(
                "max_normalized_residual squared times the counter lifetime "
                "must fit float32"
            )

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
        if type(config) is not dict:
            raise ValueError("config must be an actual dict")
        payload = dict(config)
        type_name = payload.pop("type", "LearningSignalEstimatorConfig")
        if type(type_name) is not str or type_name != "LearningSignalEstimatorConfig":
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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_float_scalars_per_step",
            _require_int32(
                "input_float_scalars_per_step",
                self.input_float_scalars_per_step,
                minimum=1,
            ),
        )
        # Producer counts have the exact form D * (2E + 1) + 1 for
        # ensemble size E >= 2 and target dimension D >= 1.  Removing all
        # factors of two from count - 1 therefore leaves an odd factor >= 5.
        odd_factor = self.input_float_scalars_per_step - 1
        while odd_factor > 0 and odd_factor % 2 == 0:
            odd_factor //= 2
        if odd_factor < 5:
            raise ValueError(
                "input_float_scalars_per_step is not attainable by a legal estimator config"
            )
        object.__setattr__(
            self,
            "persistent_float32_scalars",
            _require_int32(
                "persistent_float32_scalars",
                self.persistent_float32_scalars,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "persistent_int32_scalars",
            _require_int32(
                "persistent_int32_scalars",
                self.persistent_int32_scalars,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "persistent_state_scalars",
            _require_int32(
                "persistent_state_scalars",
                self.persistent_state_scalars,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "persistent_state_bytes",
            _require_int32(
                "persistent_state_bytes",
                self.persistent_state_bytes,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "output_float32_scalars",
            _require_int32(
                "output_float32_scalars",
                self.output_float32_scalars,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "output_bool_scalars",
            _require_int32("output_bool_scalars", self.output_bool_scalars, minimum=0),
        )
        object.__setattr__(
            self,
            "output_logical_bytes",
            _require_int32(
                "output_logical_bytes",
                self.output_logical_bytes,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "trainable_scalars",
            _require_int32("trainable_scalars", self.trainable_scalars, minimum=0),
        )
        expected = {
            "persistent_float32_scalars": 5,
            "persistent_int32_scalars": 4,
            "persistent_state_scalars": 9,
            "persistent_state_bytes": 36,
            "output_float32_scalars": 8,
            "output_bool_scalars": 6,
            "output_logical_bytes": 38,
            "trainable_scalars": 0,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} does not match the learning-signal implementation")

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
            if array.dtype != jnp.dtype(jnp.int32):
                raise ValueError(f"state.{name} must have dtype int32")
        for name, value in float_values.items():
            array = jnp.asarray(value)
            if array.shape != ():
                raise ValueError(f"state.{name} must be scalar")
            if array.dtype != jnp.dtype(jnp.float32):
                raise ValueError(f"state.{name} must have dtype float32")

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

        ensemble_mean = _stable_mean(safe_means, axis=0)
        per_dimension_epistemic = _stable_mean(
            jnp.square(safe_means - ensemble_mean[None, :]),
            axis=0,
        )
        per_dimension_aleatoric = _stable_mean(safe_variances, axis=0)
        epistemic_disagreement = _stable_mean(per_dimension_epistemic)
        epistemic_surprise = _stable_mean(
            per_dimension_epistemic
            / jnp.maximum(per_dimension_aleatoric, self._config.variance_floor)
        )
        epistemic_surprise = jnp.minimum(
            epistemic_surprise,
            self._config.max_normalized_residual,
        )
        aleatoric_uncertainty = _stable_mean(per_dimension_aleatoric)
        total_variance = jnp.maximum(
            per_dimension_epistemic + per_dimension_aleatoric,
            self._config.variance_floor,
        )
        normalized_residual = _stable_mean(
            jnp.square(safe_target - ensemble_mean) / total_variance
        )
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
        self._validate_state_shapes(state)
        means = jnp.asarray(member_means)
        variances = jnp.asarray(predicted_aleatoric_variances)
        targets = jnp.asarray(observed_targets)
        losses = jnp.asarray(observed_losses)
        if means.ndim != 3:
            raise ValueError("member_means sequence must have rank 3")
        if means.shape[0] < 1:
            raise ValueError("member_means sequence must be non-empty")
        num_steps = _require_int32("scan sequence length", means.shape[0], minimum=1)
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
