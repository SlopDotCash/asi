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
from dataclasses import dataclass, replace
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework.core._float32_scalars import validated_float32_scalar


def _skip_zero_scale(scale: Array, value: Array) -> Array:
    """Skip ``0 * inf`` so a disabled error EMA does not poison the diagnostic."""
    return jnp.where(scale == 0.0, jnp.zeros_like(value), scale * value)


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
        data = dict(payload)
        data.pop("type", None)
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
        self._config = self._validated_config(config)

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
        data = dict(payload)
        data.pop("type", None)
        return cls(RLSRewardModelConfig.from_config(data["config"]))

    @functools.partial(jax.jit, static_argnums=(0,))
    def init(self) -> RLSRewardModelState:
        """Initialize model state."""
        feature_dim = self._config.feature_dim
        return RLSRewardModelState(
            weights=jnp.zeros((feature_dim,), dtype=jnp.float32),
            covariance=(
                jnp.eye(feature_dim, dtype=jnp.float32) / self._config.ridge
            ),
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
        target = jnp.asarray(reward, dtype=jnp.float32)
        prediction = jnp.dot(state.weights, x)
        error = target - prediction
        covariance_features = state.covariance @ x
        forgetting = jnp.asarray(self._config.forgetting, dtype=jnp.float32)
        denominator = forgetting + jnp.dot(x, covariance_features)
        gain = covariance_features / denominator
        next_weights = state.weights + gain * error
        next_covariance = (
            state.covariance - jnp.outer(gain, covariance_features)
        ) / forgetting

        error_decay = jnp.asarray(self._config.error_decay, dtype=jnp.float32)
        abs_error = jnp.abs(error)
        next_abs_error_ema = jnp.where(
            state.step_count == 0,
            abs_error,
            _skip_zero_scale(error_decay, state.abs_error_ema)
            + (1.0 - error_decay) * abs_error,
        )
        next_state = RLSRewardModelState(
            weights=next_weights,
            covariance=next_covariance,
            abs_error_ema=next_abs_error_ema,
            step_count=state.step_count + 1,
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

    def _validated_config(self, config: RLSRewardModelConfig) -> RLSRewardModelConfig:
        if type(config.feature_dim) is not int or config.feature_dim <= 0:
            raise ValueError("feature_dim must be a positive builtin integer")
        forgetting = validated_float32_scalar(
            "forgetting", config.forgetting, positive=True, upper=1.0
        )
        ridge = validated_float32_scalar("ridge", config.ridge, positive=True)
        error_decay = validated_float32_scalar(
            "error_decay", config.error_decay, lower=0.0, upper=1.0, upper_inclusive=False
        )
        return replace(
            config,
            forgetting=forgetting,
            ridge=ridge,
            error_decay=error_decay,
        )


__all__ = [
    "RLSRewardModel",
    "RLSRewardModelConfig",
    "RLSRewardModelState",
    "RLSRewardModelUpdateResult",
]
