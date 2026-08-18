# mypy: disable-error-code="call-arg"
"""Online reward models for selective model-based updates.

Implements a linear recursive-least-squares (RLS) scalar reward predictor
with exponential forgetting (standard exponentially weighted RLS; see e.g.
Haykin, *Adaptive Filter Theory*).  Its role in the model-based lane is to
supply calibrated reward targets for imagined (dream) transitions when the
shared multi-head dynamics model's reward head is too biased or too slow to
calibrate.  The ``abs_error_ema`` diagnostic gives callers an online
reliability estimate for deciding whether imagined rewards are currently
trustworthy — the "selective" part of selective model-based updates.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework.core._float32_scalars import validated_float32_scalar

_INT32_MAX = 2**31 - 1
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
_ACTUAL_FLOAT_TYPES = frozenset(
    {float, *(np.dtype(code).type for code in ("e", "f", "d", "g"))}
)


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _preflight_state_resources(feature_dim: int) -> None:
    state_scalars = feature_dim * feature_dim + feature_dim + 2
    if state_scalars > _INT32_MAX:
        raise ValueError("reward-model state scalars must fit signed int32")
    if 4 * state_scalars > _INT32_MAX:
        raise ValueError("reward-model state bytes must fit signed int32")


def _preflight_update_working_set(feature_dim: int) -> None:
    # Covariance, proposed covariance, and the outer-product rank-one term,
    # plus the live feature-width vectors (x, weights, Px, gain, next weights).
    update_scalars = 3 * feature_dim * feature_dim + 5 * feature_dim + 8
    if 4 * update_scalars > _INT32_MAX:
        raise ValueError(
            "reward-model update working set byte count must fit signed int32"
        )


def _validated_config_float(name: str, value: object, **bounds: Any) -> float:
    if type(value) not in (_ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES):
        raise ValueError(f"{name} must be a finite real scalar")
    return validated_float32_scalar(name, value, **bounds)


def _skip_zero_scale(scale: Array, value: Array) -> Array:
    """Skip ``0 * inf`` so a disabled error EMA does not poison the diagnostic."""
    return jnp.where(scale == 0.0, jnp.zeros_like(value), scale * value)


def _saturating_int32_increment(value: Array) -> Array:
    """Increment a diagnostic counter without signed-int32 wraparound."""
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


@dataclass(frozen=True)
class RLSRewardModelConfig:
    """Configuration for a linear recursive-least-squares reward model.

    Args:
        feature_dim: Number of scalar input features.
        forgetting: Exponential forgetting factor. Values near one favor stable
            estimates; lower values adapt faster to nonstationarity but risk
            covariance windup (see :meth:`RLSRewardModel.update`).
        ridge: Initial precision regularizer. Larger values make the initial
            covariance smaller and therefore more conservative.
        error_decay: EMA decay for absolute reward-prediction error diagnostics.
    """

    feature_dim: int
    forgetting: float = 0.995
    ridge: float = 10.0
    error_decay: float = 0.99

    def __post_init__(self) -> None:
        feature_dim = _require_int32("feature_dim", self.feature_dim, minimum=1)
        _preflight_state_resources(feature_dim)
        forgetting = _validated_config_float(
            "forgetting", self.forgetting, positive=True, upper=1.0
        )
        ridge = _validated_config_float("ridge", self.ridge, positive=True)
        error_decay = _validated_config_float(
            "error_decay", self.error_decay, lower=0.0, upper=1.0, upper_inclusive=False
        )

        object.__setattr__(self, "feature_dim", feature_dim)
        object.__setattr__(self, "forgetting", forgetting)
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "error_decay", error_decay)

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "type": "RLSRewardModelConfig",
            "feature_dim": self.feature_dim,
            "forgetting": self.forgetting,
            "ridge": self.ridge,
            "error_decay": self.error_decay,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> RLSRewardModelConfig:
        """Reconstruct from :meth:`to_config` output."""
        if type(payload) is not dict:
            raise ValueError("payload must be an exact built-in dict")
        expected = {"type", "feature_dim", "forgetting", "ridge", "error_decay"}
        if any(type(key) is not str for key in payload) or set(payload) != expected:
            raise ValueError("payload fields do not match the serialized schema")
        if type(payload["type"]) is not str or payload["type"] != "RLSRewardModelConfig":
            raise ValueError("payload type differs")
        data = dict(payload)
        data.pop("type")
        return cls(**data)


@chex.dataclass(frozen=True)
class RLSRewardModelState:
    """State for :class:`RLSRewardModel`."""

    weights: Float[Array, " feature_dim"]
    covariance: Float[Array, "feature_dim feature_dim"]
    abs_error_ema: Float[Array, ""]
    step_count: Array


@chex.dataclass(frozen=True)
class RLSRewardModelUpdateResult:
    """Result from one reward-model update."""

    state: RLSRewardModelState
    prediction: Float[Array, ""]
    error: Float[Array, ""]
    gain: Float[Array, " feature_dim"]
    update_applied: Bool[Array, ""]


class RLSRewardModel:
    """Linear RLS scalar reward predictor.

    This model is intentionally narrow: it learns calibrated scalar reward
    predictions from caller-provided features. It is useful when imagined
    updates need reward targets but a shared multi-head dynamics model is too
    biased or too slow to calibrate.
    """

    def __init__(self, config: RLSRewardModelConfig):
        """Initialize the model."""
        if type(config) is not RLSRewardModelConfig:
            raise TypeError("config must be an RLSRewardModelConfig")
        self._config = config

    @property
    def config(self) -> RLSRewardModelConfig:
        """Model configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize the model configuration."""
        return {
            "type": "RLSRewardModel",
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> RLSRewardModel:
        """Reconstruct from :meth:`to_config` output."""
        if type(payload) is not dict:
            raise ValueError("payload must be an exact built-in dict")
        if any(type(key) is not str for key in payload) or set(payload) != {"type", "config"}:
            raise ValueError("payload fields do not match the serialized schema")
        if type(payload["type"]) is not str or payload["type"] != "RLSRewardModel":
            raise ValueError("payload type differs")
        if type(payload["config"]) is not dict:
            raise ValueError("payload config must be an exact built-in dict")
        return cls(RLSRewardModelConfig.from_config(payload["config"]))

    @functools.partial(jax.jit, static_argnums=(0,))
    def init(self) -> RLSRewardModelState:
        """Initialize model state."""
        feature_dim = self._config.feature_dim
        _preflight_update_working_set(feature_dim)
        return RLSRewardModelState(
            weights=jnp.zeros((feature_dim,), dtype=jnp.float32),
            covariance=(jnp.eye(feature_dim, dtype=jnp.float32) / self._config.ridge),
            abs_error_ema=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, state: RLSRewardModelState, features: Array) -> Array:
        """Predict reward from one feature vector."""
        x = jnp.asarray(features, dtype=jnp.float32)
        if x.shape != (self._config.feature_dim,):
            raise ValueError(
                f"features must have shape ({self._config.feature_dim},), got {x.shape}"
            )
        return jnp.dot(state.weights, x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: RLSRewardModelState,
        features: Array,
        reward: Array,
    ) -> RLSRewardModelUpdateResult:
        """Update from one real reward observation.

        Standard exponentially weighted RLS (Haykin, *Adaptive Filter
        Theory*):

        1. ``gain = P x / (forgetting + x^T P x)``
        2. ``w <- w + gain * error``
        3. ``P <- (P - gain (P x)^T) / forgetting``

        Caveat (covariance windup): with ``forgetting < 1``, any direction
        of feature space that receives no excitation has its covariance
        grown by ``1 / forgetting`` every step, so under persistently
        poorly excited features the covariance — and hence the gain when
        excitation returns — can grow without bound. Keep ``forgetting=1``
        unless the reward function is genuinely nonstationary and the
        feature stream stays exciting.
        """
        x = jnp.asarray(features, dtype=jnp.float32)
        if x.shape != (self._config.feature_dim,):
            raise ValueError(
                f"features must have shape ({self._config.feature_dim},), got {x.shape}"
            )
        raw_target = jnp.asarray(reward)
        if raw_target.shape != ():
            raise ValueError("reward must be a scalar")
        if not (
            jnp.issubdtype(raw_target.dtype, jnp.floating)
            or jnp.issubdtype(raw_target.dtype, jnp.integer)
        ):
            raise TypeError("reward must be real numeric")
        target = raw_target.astype(jnp.float32)
        prediction = jnp.dot(state.weights, x)
        error = target - prediction
        covariance_features = state.covariance @ x
        forgetting = jnp.asarray(self._config.forgetting, dtype=jnp.float32)
        denominator = forgetting + jnp.dot(x, covariance_features)
        gain = covariance_features / denominator
        next_weights = state.weights + gain * error
        next_covariance = (state.covariance - jnp.outer(gain, covariance_features)) / forgetting

        error_decay = jnp.asarray(self._config.error_decay, dtype=jnp.float32)
        abs_error = jnp.abs(error)
        next_abs_error_ema = jnp.where(
            state.step_count == 0,
            abs_error,
            _skip_zero_scale(error_decay, state.abs_error_ema) + (1.0 - error_decay) * abs_error,
        )
        next_state = RLSRewardModelState(
            weights=next_weights,
            covariance=next_covariance,
            abs_error_ema=next_abs_error_ema,
            step_count=_saturating_int32_increment(state.step_count),
        )
        # Inf reward * a silent feature's zero gain is 0*inf = NaN, and
        # that channel stays poisoned. Hold the previous finite state.
        checked_error_ema = (
            jnp.zeros_like(state.abs_error_ema)
            if self._config.error_decay == 0.0
            else state.abs_error_ema
        )
        source_finite = (
            jnp.all(jnp.isfinite(state.weights))
            & jnp.all(jnp.isfinite(state.covariance))
            & jnp.isfinite(checked_error_ema)
        )
        inputs_valid = jnp.all(jnp.isfinite(x)) & jnp.isfinite(jnp.squeeze(target))
        proposed_finite = (
            jnp.all(jnp.isfinite(next_weights))
            & jnp.all(jnp.isfinite(next_covariance))
            & jnp.isfinite(next_abs_error_ema)
        )
        update_applied = source_finite & inputs_valid & proposed_finite
        committed = jax.lax.cond(
            update_applied,
            lambda: next_state,
            lambda: state,
        )
        return RLSRewardModelUpdateResult(
            state=committed,
            prediction=jnp.where(update_applied, prediction, jnp.zeros_like(prediction)),
            error=jnp.where(update_applied, error, jnp.zeros_like(error)),
            gain=jnp.where(update_applied, gain, jnp.zeros_like(gain)),
            update_applied=update_applied,
        )

__all__ = [
    "RLSRewardModel",
    "RLSRewardModelConfig",
    "RLSRewardModelState",
    "RLSRewardModelUpdateResult",
]
