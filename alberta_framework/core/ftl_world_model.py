# mypy: disable-error-code="call-arg"
"""Fixed-shape, lifetime-statistics world model for continual planning.

This module implements a one-dimensional LoSSE-style analogue of the sparse
online world model in Liu et al. (2025), *Continual Reinforcement Learning by
Planning with Online World Models*.  It models

``(observation_t, action_t) -> observation_{t+1} - observation_t``

with fixed random projections, soft-binned sparse features, and accumulated
ridge-regression sufficient statistics.  Each transition is predicted before
it is incorporated.  No replay buffer, task identifier, boundary signal, or
growing transition store is used.

The default ``statistics_decay=1`` retains sufficient statistics from the
entire experience stream and performs the paper's sparse active-block update.
It is not the exact global ridge minimizer at every step.  The paper's
no-regret theorem applies only without decay and under its stated assumptions.
A value below one is an explicitly forgetting, exponentially weighted variant
for genuinely changing dynamics, outside that guarantee.

The feature map has ``projection_dim * bins`` coordinates and stores exactly
``2 * projection_dim`` adjacent support slots per input.  At bin boundaries,
one slot per projection can have value zero, so there are *at most* that many
non-zero features.  The implementation stores a dense Gram matrix for clarity
and updates only the support block.  Its state shapes are fixed in time but
require quadratic memory in the configured feature dimension.  The float32
sufficient-statistic magnitudes are not bounded, so shape-bounded memory does
not by itself provide indefinite numerical precision.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import operator
from numbers import Real
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Float, Int

from alberta_framework._float32 import round_real_to_float32
from alberta_framework.core._float32_scalars import validated_float32_scalar

_FLOAT32_TINY = 2.0**-126
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


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _finite_positive_normal_float32(name: str, value: object) -> float:
    """Return a concrete real after validation in the model's execution dtype."""
    message = f"{name} must be a finite positive normal float32"
    preserve_builtin_payload = type(value) is int or type(value) is float
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        narrowed = round_real_to_float32(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(narrowed) or narrowed < _FLOAT32_TINY:
        raise ValueError(message)
    if preserve_builtin_payload:
        concrete = float(value)
        if round_real_to_float32(cast(Real, concrete)) == narrowed:
            return concrete
    return narrowed


@dataclasses.dataclass(frozen=True)
class SparseFTLWorldModelConfig:
    """Configuration for :class:`SparseFTLWorldModel`.

    Args:
        observation_dim: Flat state/observation dimension.
        action_dim: Flat action-vector dimension. Discrete callers should pass
            a one-hot action vector.
        projection_dim: Number of fixed random scalar projections.
        bins: Soft bins per projection; two adjacent support slots are stored.
        ridge: Positive diagonal ridge coefficient in each local solve.
        statistics_decay: Sufficient-statistic retention. ``1`` is the
            lifetime-statistics, no-decay active-block variant; values below
            one exponentially forget old dynamics.
        prediction_clip: Absolute bound on each predicted state delta.
        error_decay: EMA decay for prequential squared prediction error.
    """

    observation_dim: int
    action_dim: int
    projection_dim: int = 32
    bins: int = 7
    ridge: float = 0.01
    statistics_decay: float = 1.0
    prediction_clip: float = 10.0
    error_decay: float = 0.99

    def __post_init__(self) -> None:
        """Validate all static dimensions and numerical parameters."""
        object.__setattr__(
            self,
            "observation_dim",
            _require_int32("observation_dim", self.observation_dim, minimum=1),
        )
        object.__setattr__(
            self,
            "action_dim",
            _require_int32("action_dim", self.action_dim, minimum=1),
        )
        object.__setattr__(
            self,
            "projection_dim",
            _require_int32("projection_dim", self.projection_dim, minimum=1),
        )
        object.__setattr__(
            self,
            "bins",
            _require_int32("bins", self.bins, minimum=2),
        )
        ridge = _finite_positive_normal_float32("ridge", self.ridge)
        statistics_decay = validated_float32_scalar(
            "statistics_decay",
            self.statistics_decay,
            positive=True,
            upper=1.0,
            upper_inclusive=True,
        )
        prediction_clip = _finite_positive_normal_float32(
            "prediction_clip",
            self.prediction_clip,
        )
        error_decay = validated_float32_scalar(
            "error_decay",
            self.error_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "statistics_decay", statistics_decay)
        object.__setattr__(self, "prediction_clip", prediction_clip)
        object.__setattr__(self, "error_decay", error_decay)

    @property
    def input_dim(self) -> int:
        """Concatenated observation-action dimension."""
        return self.observation_dim + self.action_dim

    @property
    def feature_dim(self) -> int:
        """Total soft-binned feature dimension."""
        return self.projection_dim * self.bins

    @property
    def active_feature_count(self) -> int:
        """Fixed support-block width; boundary entries can have value zero."""
        return 2 * self.projection_dim

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration."""
        payload = dataclasses.asdict(self)
        payload["type"] = type(self).__name__
        return payload

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SparseFTLWorldModelConfig:
        """Reconstruct a configuration from :meth:`to_config`."""
        payload = dict(config)
        payload.pop("type", None)
        return cls(**payload)


@chex.dataclass(frozen=True)
class SparseFeatures:
    """Sparse soft-binned feature representation of one input."""

    indices: Int[Array, " active_features"]
    values: Float[Array, " active_features"]
    dense: Float[Array, " feature_dim"]


@chex.dataclass(frozen=True)
class SparseFTLWorldModelState:
    """Fixed-size sufficient statistics and weights."""

    projection: Float[Array, "projection_dim input_dim"]
    gram: Float[Array, "feature_dim feature_dim"]
    cross: Float[Array, "feature_dim observation_dim"]
    weights: Float[Array, "feature_dim observation_dim"]
    prediction_error_ema: Float[Array, ""]
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class SparseFTLWorldModelPrediction:
    """One action-conditioned state prediction."""

    delta: Float[Array, " observation_dim"]
    next_observation: Float[Array, " observation_dim"]
    features: SparseFeatures


@chex.dataclass(frozen=True)
class SparseFTLWorldModelUpdateResult:
    """Predict-before-update diagnostics and updated state."""

    state: SparseFTLWorldModelState
    prediction: SparseFTLWorldModelPrediction
    target_delta: Float[Array, " observation_dim"]
    error: Float[Array, " observation_dim"]
    squared_error: Float[Array, ""]


@chex.dataclass(frozen=True)
class SparseFTLWorldModelLearningResult:
    """Outputs of an uninterrupted scan over transitions."""

    state: SparseFTLWorldModelState
    predicted_next_observations: Float[Array, "num_steps observation_dim"]
    target_deltas: Float[Array, "num_steps observation_dim"]
    errors: Float[Array, "num_steps observation_dim"]
    squared_errors: Float[Array, " num_steps"]


class SparseFTLWorldModel:
    """One-dimensional sparse active-block ridge model with lifetime statistics."""

    def __init__(self, config: SparseFTLWorldModelConfig):
        self._config = config

    @property
    def config(self) -> SparseFTLWorldModelConfig:
        """Static model configuration."""
        return self._config

    @property
    def state_nbytes(self) -> int:
        """Configured serialized array size, independent of lifetime."""
        cfg = self._config
        float_count = (
            cfg.projection_dim * cfg.input_dim
            + cfg.feature_dim * cfg.feature_dim
            + 2 * cfg.feature_dim * cfg.observation_dim
            + 1
        )
        # All configured arrays are float32; step_count is int32.
        return 4 * (float_count + 1)

    def to_config(self) -> dict[str, Any]:
        """Serialize model type and static configuration."""
        return {
            "type": type(self).__name__,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SparseFTLWorldModel:
        """Reconstruct from :meth:`to_config`."""
        payload = dict(config)
        payload.pop("type", None)
        return cls(SparseFTLWorldModelConfig.from_config(payload["config"]))

    def init(self, key: Array) -> SparseFTLWorldModelState:
        """Initialize fixed projections and zero sufficient statistics."""
        cfg = self._config
        projection = jr.normal(
            key,
            (cfg.projection_dim, cfg.input_dim),
            dtype=jnp.float32,
        ) / jnp.sqrt(jnp.asarray(cfg.input_dim, dtype=jnp.float32))
        return SparseFTLWorldModelState(
            projection=projection,
            gram=jnp.zeros((cfg.feature_dim, cfg.feature_dim), dtype=jnp.float32),
            cross=jnp.zeros(
                (cfg.feature_dim, cfg.observation_dim),
                dtype=jnp.float32,
            ),
            weights=jnp.zeros(
                (cfg.feature_dim, cfg.observation_dim),
                dtype=jnp.float32,
            ),
            prediction_error_ema=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def _input(self, observation: Array, action: Array) -> Array:
        cfg = self._config
        obs = jnp.asarray(observation, dtype=jnp.float32).reshape((cfg.observation_dim,))
        act = jnp.asarray(action, dtype=jnp.float32).reshape((cfg.action_dim,))
        return jnp.concatenate((obs, act))

    @functools.partial(jax.jit, static_argnums=(0,))
    def sparse_features(
        self,
        state: SparseFTLWorldModelState,
        observation: Array,
        action: Array,
    ) -> SparseFeatures:
        """Project and softly bin an observation-action pair."""
        cfg = self._config
        projected = jax.nn.sigmoid(state.projection @ self._input(observation, action))
        location = (cfg.bins - 1) * projected
        lower = jnp.minimum(
            jnp.floor(location).astype(jnp.int32),
            jnp.asarray(cfg.bins - 2, dtype=jnp.int32),
        )
        fraction = location - lower.astype(jnp.float32)
        offsets = jnp.arange(cfg.projection_dim, dtype=jnp.int32) * cfg.bins
        lower_indices = offsets + lower
        upper_indices = lower_indices + 1
        indices = jnp.stack((lower_indices, upper_indices), axis=1).reshape((-1,))
        values = jnp.stack((1.0 - fraction, fraction), axis=1).reshape((-1,))
        dense = jnp.zeros((cfg.feature_dim,), dtype=jnp.float32).at[indices].add(values)
        return SparseFeatures(indices=indices, values=values, dense=dense)

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: SparseFTLWorldModelState,
        observation: Array,
        action: Array,
    ) -> SparseFTLWorldModelPrediction:
        """Predict the next observation without changing model state."""
        cfg = self._config
        obs = jnp.asarray(observation, dtype=jnp.float32).reshape((cfg.observation_dim,))
        features = self.sparse_features(state, obs, action)
        delta = jnp.clip(
            features.values @ state.weights[features.indices],
            -cfg.prediction_clip,
            cfg.prediction_clip,
        )
        return SparseFTLWorldModelPrediction(
            delta=delta,
            next_observation=obs + delta,
            features=features,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: SparseFTLWorldModelState,
        observation: Array,
        action: Array,
        next_observation: Array,
    ) -> SparseFTLWorldModelUpdateResult:
        """Predict, then incorporate exactly one real transition."""
        cfg = self._config
        obs = jnp.asarray(observation, dtype=jnp.float32).reshape((cfg.observation_dim,))
        next_obs = jnp.asarray(next_observation, dtype=jnp.float32).reshape((cfg.observation_dim,))
        prediction = self.predict(state, obs, action)
        target_delta = next_obs - obs
        error = target_delta - prediction.delta
        squared_error = jnp.mean(error**2)

        decay = jnp.asarray(cfg.statistics_decay, dtype=jnp.float32)
        gram = decay * state.gram
        cross = decay * state.cross
        indices = prediction.features.indices
        values = prediction.features.values
        gram = gram.at[indices[:, None], indices[None, :]].add(values[:, None] * values[None, :])
        cross = cross.at[indices].add(values[:, None] * target_delta[None, :])

        # Exact minimizer for the active block with all inactive weights held
        # at their previous values:
        #   W_s = (A_ss + ridge I)^-1 (B_s - A_s,bar W_bar).
        active_gram = gram[indices[:, None], indices[None, :]]
        active_cross = cross[indices]
        all_contribution = gram[indices] @ state.weights
        inactive_contribution = all_contribution - active_gram @ state.weights[indices]
        system = active_gram + cfg.ridge * jnp.eye(
            cfg.active_feature_count,
            dtype=jnp.float32,
        )
        active_weights = jnp.linalg.solve(
            system,
            active_cross - inactive_contribution,
        )
        weights = state.weights.at[indices].set(active_weights)

        first = state.step_count == 0
        error_ema = jnp.where(
            first,
            squared_error,
            cfg.error_decay * state.prediction_error_ema + (1.0 - cfg.error_decay) * squared_error,
        )
        max_step_count = jnp.array(2_147_483_647, dtype=jnp.int32)
        next_step_count = jnp.minimum(
            state.step_count, max_step_count - jnp.array(1, jnp.int32)
        ) + jnp.array(1, dtype=jnp.int32)
        new_state = SparseFTLWorldModelState(
            projection=state.projection,
            gram=gram,
            cross=cross,
            weights=weights,
            prediction_error_ema=error_ema,
            step_count=next_step_count,
        )
        return SparseFTLWorldModelUpdateResult(
            state=new_state,
            prediction=prediction,
            target_delta=target_delta,
            error=error,
            squared_error=squared_error,
        )


def run_sparse_ftl_world_model(
    model: SparseFTLWorldModel,
    state: SparseFTLWorldModelState,
    observations: Array,
    actions: Array,
    next_observations: Array,
) -> SparseFTLWorldModelLearningResult:
    """Run one uninterrupted, predict-before-update transition stream."""

    def step_fn(
        carry: SparseFTLWorldModelState,
        transition: tuple[Array, Array, Array],
    ) -> tuple[SparseFTLWorldModelState, tuple[Array, ...]]:
        observation, action, next_observation = transition
        result = model.update(carry, observation, action, next_observation)
        return result.state, (
            result.prediction.next_observation,
            result.target_delta,
            result.error,
            result.squared_error,
        )

    final_state, outputs = jax.lax.scan(
        step_fn,
        state,
        (observations, actions, next_observations),
    )
    predictions, targets, errors, squared_errors = outputs
    return SparseFTLWorldModelLearningResult(
        state=final_state,
        predicted_next_observations=predictions,
        target_deltas=targets,
        errors=errors,
        squared_errors=squared_errors,
    )


__all__ = [
    "SparseFTLWorldModel",
    "SparseFTLWorldModelConfig",
    "SparseFTLWorldModelLearningResult",
    "SparseFTLWorldModelPrediction",
    "SparseFTLWorldModelState",
    "SparseFTLWorldModelUpdateResult",
    "SparseFeatures",
    "run_sparse_ftl_world_model",
]
