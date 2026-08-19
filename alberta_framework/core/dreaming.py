# mypy: disable-error-code="call-arg"
"""Guarded self-simulation helpers.

Dreaming is deliberately separated from world-model learning. The world model
learns only from real transitions; this module decides whether a predicted
transition is safe enough to expose to a control learner.

This is the planning half of a Dyna loop (Sutton 1991): model-generated
transitions stand in for real experience, but here each one must pass
explicit warmup, model-error, and uncertainty gates before it reaches the
control learner, so a poor or not-yet-trained model cannot corrupt control
updates.  Rejections are recorded with per-gate reasons rather than silently
dropped, keeping the selection auditable.

References:
    Sutton (1991). "Dyna, an Integrated Architecture for Learning, Planning,
        and Reacting."
"""

from __future__ import annotations

import dataclasses
import functools
import operator
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, Literal, Protocol, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Float, Int

from alberta_framework.core._float32_scalars import validated_float32_scalar_with_ratio
from alberta_framework.core.behavior_model import (
    BehaviorModel,
    BehaviorModelState,
    floor_and_renormalize_probabilities,
    selected_action_probabilities,
)
from alberta_framework.core.normalizers import _saturating_int32_counter_increment
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelState,
)
from alberta_framework.core.world_model import (
    WorldModelPrediction as ActionWorldModelPrediction,
)

_INT32_MAX = 2**31 - 1
_ACTUAL_INT_TYPES: frozenset[type] = frozenset(
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
_ACTUAL_FLOAT_TYPES: frozenset[type] = frozenset(
    {
        float,
        Fraction,
        *(np.dtype(c).type for c in ("e", "f", "d", "g")),
    }
)
_ALLOWED_REAL_TYPES: frozenset[type] = _ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES


def _require_int(
    name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX
) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer")
    return canonical


def _validated_config_float(
    name: str,
    value: object,
    *,
    positive: bool = False,
    lower: float | None = None,
    upper: float | None = None,
    upper_inclusive: bool = True,
) -> float:
    if type(value) not in _ALLOWED_REAL_TYPES:
        raise ValueError(f"{name} must be a finite real number")
    stored, numerator, denominator = validated_float32_scalar_with_ratio(
        name,
        value,
        positive=positive,
        lower=lower,
        upper=upper,
        upper_inclusive=upper_inclusive,
    )
    if numerator != 0 and abs(numerator) * (1 << 149) <= denominator:
        raise ValueError(f"{name} must remain nonzero once narrowed to float32")
    return stored


def _require_float32_resource(
    name: str, *, vector_scalars: int, fixed_scalars: int = 0
) -> None:
    total = vector_scalars + fixed_scalars
    if total > _INT32_MAX:
        raise ValueError(f"{name} scalar count must fit signed int32")
    if 4 * total > _INT32_MAX:
        raise ValueError(f"{name} byte count must fit signed int32")


def _copy_mapping(payload: object, *, name: str) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} payload must be a mapping")
    try:
        data = dict(payload)
    except Exception as error:
        raise ValueError(f"{name} payload could not be read") from error
    for key in data:
        if type(key) is not str:
            raise ValueError(f"{name} payload has exact strings as keys")
    return data




def _require_bool(value: object, name: str) -> bool:
    """Reject truthy stand-ins for exact bools."""
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


@dataclasses.dataclass(frozen=True)
class DreamingConfig:
    """Configuration for guarded model-generated transitions.

    Args:
        warmup_steps: Real model updates required before any dream can be
            accepted.
        max_model_error_ema: Maximum allowed world-model real-transition error
            EMA. Set high for smoke experiments; tune from real-error quantiles
            for serious runs.
        max_uncertainty: Maximum allowed external uncertainty estimate, e.g.
            ensemble disagreement. Single-model callers can pass ``0``.
        min_discount: Lower discount clamp for synthetic transitions.
        max_discount: Upper discount clamp for synthetic transitions. When
            ``None``, the world model's ``gamma`` is used.
    """

    warmup_steps: int = 100
    max_model_error_ema: float = 1.0
    max_uncertainty: float = 1.0
    min_discount: float = 0.0
    max_discount: float | None = None
    rollout_horizon: int = 1
    confidence_threshold: float = 0.0
    max_model_error: float = 1.0e30
    discount_floor: float = 0.0
    stop_on_terminal: bool = True

    def __post_init__(self) -> None:
        """Validate scalar configuration, rejecting NaN and type stand-ins."""
        object.__setattr__(
            self,
            "warmup_steps",
            _require_int("warmup_steps", self.warmup_steps, minimum=0),
        )
        for name in (
            "max_model_error_ema",
            "max_uncertainty",
            "min_discount",
            "confidence_threshold",
            "max_model_error",
            "discount_floor",
        ):
            object.__setattr__(
                self,
                name,
                _validated_config_float(name, getattr(self, name), lower=0.0),
            )
        if self.max_discount is not None:
            object.__setattr__(
                self,
                "max_discount",
                _validated_config_float(
                    "max_discount", self.max_discount, lower=0.0
                ),
            )
        object.__setattr__(
            self,
            "rollout_horizon",
            _require_int("rollout_horizon", self.rollout_horizon, minimum=1),
        )
        object.__setattr__(
            self, "stop_on_terminal", _require_bool(self.stop_on_terminal, "stop_on_terminal")
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = dataclasses.asdict(self)
        payload["type"] = "DreamingConfig"
        return payload

    @classmethod
    def from_config(cls, config: object) -> DreamingConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = _copy_mapping(config, name="DreamingConfig")
        config_type = data.pop("type", None)
        if type(config_type) is not str or config_type != "DreamingConfig":
            raise ValueError("DreamingConfig payload type must be DreamingConfig")
        allowed = {
            "warmup_steps",
            "max_model_error_ema",
            "max_uncertainty",
            "min_discount",
            "max_discount",
            "rollout_horizon",
            "confidence_threshold",
            "max_model_error",
            "discount_floor",
            "stop_on_terminal",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError("DreamingConfig payload has unknown fields")
        return cls(**data)  # type: ignore[arg-type]


@chex.dataclass(frozen=True)
class DreamTransition:
    """One synthetic transition proposed by a world model."""

    observation: Float[Array, " observation_dim"]
    action: Int[Array, ""]
    reward: Float[Array, ""]
    discount: Float[Array, ""]
    next_observation: Float[Array, " observation_dim"]


@chex.dataclass(frozen=True)
class DreamProposal:
    """A guarded dream transition plus diagnostics."""

    transition: DreamTransition
    prediction: ActionWorldModelPrediction
    accepted: Array
    reject_code: Int[Array, ""]
    uncertainty: Float[Array, ""]


@chex.dataclass(frozen=True)
class RecentObservationBufferState:
    """Fixed-size ring buffer for real-state dream anchors."""

    observations: Float[Array, "capacity observation_dim"]
    size: Int[Array, ""]
    index: Int[Array, ""]


@dataclasses.dataclass(frozen=True)
class DreamSelectionConfig:
    """Configuration for selecting useful imagined or replay candidates.

    ``surprise`` should measure prediction error or novelty. ``utility`` can be
    reward, positive TD error, value improvement, or another task-relevant
    benefit. The score is deliberately simple so experiments can audit exactly
    why a candidate was selected.
    """

    max_items: int = 1
    surprise_weight: float = 1.0
    utility_weight: float = 1.0
    confidence_weight: float = 0.0
    model_error_weight: float = 1.0
    min_surprise: float = 0.0
    min_utility: float = -1.0e30
    min_confidence: float = 0.0
    max_model_error: float = 1.0e30

    def __post_init__(self) -> None:
        """Validate scalar configuration, rejecting NaN and type stand-ins."""
        object.__setattr__(
            self, "max_items", _require_int("max_items", self.max_items, minimum=1)
        )
        for name in (
            "surprise_weight",
            "utility_weight",
            "confidence_weight",
            "model_error_weight",
            "min_surprise",
            "min_utility",
        ):
            object.__setattr__(
                self, name, _validated_config_float(name, getattr(self, name))
            )
        for name in ("min_confidence", "max_model_error"):
            object.__setattr__(
                self,
                name,
                _validated_config_float(name, getattr(self, name), lower=0.0),
            )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = dataclasses.asdict(self)
        payload["type"] = "DreamSelectionConfig"
        return payload

    @classmethod
    def from_config(cls, config: object) -> DreamSelectionConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = _copy_mapping(config, name="DreamSelectionConfig")
        config_type = data.pop("type", None)
        if type(config_type) is not str or config_type != "DreamSelectionConfig":
            raise ValueError(
                "DreamSelectionConfig payload type must be DreamSelectionConfig"
            )
        allowed = {
            "max_items",
            "surprise_weight",
            "utility_weight",
            "confidence_weight",
            "model_error_weight",
            "min_surprise",
            "min_utility",
            "min_confidence",
            "max_model_error",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError("DreamSelectionConfig payload has unknown fields")
        return cls(**data)  # type: ignore[arg-type]


@chex.dataclass(frozen=True)
class DreamSelectionResult:
    """Scores and selected indices for a candidate dream set."""

    selected_indices: Int[Array, " max_items"]
    scores: Float[Array, " num_candidates"]
    accepted: Array
    selected_mask: Array


class GuardedDreamer:
    """Propose short, real-state-anchored dream transitions."""

    # Reject-code constants. Kept numeric so the proposal remains JAX-friendly.
    ACCEPT = 0
    REJECT_WARMUP = 1
    REJECT_MODEL_ERROR = 2
    REJECT_UNCERTAINTY = 3
    REJECT_NONFINITE = 4

    def __init__(self, config: DreamingConfig | None = None):
        """Initialize a guarded dream proposer."""
        self._config = config or DreamingConfig()
        # DreamingConfig.__post_init__ validates on construction; re-validate here
        # so instances mutated through object.__setattr__ (or crafted subclasses)
        # cannot smuggle NaN or type stand-ins past the guard.
        _require_int("warmup_steps", self._config.warmup_steps, minimum=0)
        _validated_config_float(
            "max_model_error_ema", self._config.max_model_error_ema, lower=0.0
        )
        _validated_config_float(
            "max_uncertainty", self._config.max_uncertainty, lower=0.0
        )
        _validated_config_float("min_discount", self._config.min_discount, lower=0.0)
        if self._config.max_discount is not None:
            _validated_config_float(
                "max_discount", self._config.max_discount, lower=0.0
            )
        _require_int("rollout_horizon", self._config.rollout_horizon, minimum=1)
        _validated_config_float(
            "confidence_threshold", self._config.confidence_threshold, lower=0.0
        )
        _validated_config_float(
            "max_model_error", self._config.max_model_error, lower=0.0
        )
        _validated_config_float(
            "discount_floor", self._config.discount_floor, lower=0.0
        )
        _require_bool(self._config.stop_on_terminal, "stop_on_terminal")

    @property
    def config(self) -> DreamingConfig:
        """Dreaming guard configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {"type": "GuardedDreamer", "config": self._config.to_config()}

    @classmethod
    def from_config(cls, config: object) -> GuardedDreamer:
        """Reconstruct from :meth:`to_config` output."""
        data = _copy_mapping(config, name="GuardedDreamer")
        config_type = data.pop("type", None)
        if type(config_type) is not str or config_type != "GuardedDreamer":
            raise ValueError("GuardedDreamer payload type must be GuardedDreamer")
        if set(data) != {"config"}:
            raise ValueError("GuardedDreamer payload has unknown fields")
        raw = data["config"]
        if not isinstance(raw, Mapping):
            raise ValueError("GuardedDreamer config must be a mapping")
        return cls(DreamingConfig.from_config(raw))

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def propose(
        self,
        model: ActionConditionedWorldModel,
        model_state: ActionConditionedWorldModelState,
        observation: Array,
        action: Array,
        uncertainty: Array | None = None,
    ) -> DreamProposal:
        """Return a guarded synthetic transition proposal.

        The transition is always returned, but callers should use ``accepted``
        to decide whether to update a control learner from it.
        """
        if uncertainty is None:
            uncertainty = jnp.array(0.0, dtype=jnp.float32)
        uncertainty_arr = jnp.asarray(uncertainty, dtype=jnp.float32)
        prediction = model.predict(model_state, observation, action)
        max_discount = (
            model.config.gamma
            if self._config.max_discount is None
            else self._config.max_discount
        )
        discount = jnp.clip(
            prediction.discount,
            self._config.min_discount,
            max_discount,
        )
        transition = DreamTransition(
            observation=jnp.asarray(observation, dtype=jnp.float32).reshape(
                (model.config.observation_dim,)
            ),
            action=jnp.asarray(action, dtype=jnp.int32),
            reward=prediction.reward,
            discount=discount,
            next_observation=prediction.next_observation,
        )

        enough_data = model_state.step_count >= self._config.warmup_steps
        low_error = model_state.model_error_ema <= self._config.max_model_error_ema
        low_uncertainty = uncertainty_arr <= self._config.max_uncertainty
        finite = (
            jnp.all(jnp.isfinite(transition.observation))
            & jnp.isfinite(transition.reward)
            & jnp.isfinite(transition.discount)
            & jnp.all(jnp.isfinite(transition.next_observation))
        )
        accepted = enough_data & low_error & low_uncertainty & finite
        reject_code = jnp.where(
            accepted,
            self.ACCEPT,
            jnp.where(
                ~enough_data,
                self.REJECT_WARMUP,
                jnp.where(
                    ~low_error,
                    self.REJECT_MODEL_ERROR,
                    jnp.where(~low_uncertainty, self.REJECT_UNCERTAINTY, self.REJECT_NONFINITE),
                ),
            ),
        ).astype(jnp.int32)

        return DreamProposal(
            transition=transition,
            prediction=prediction,
            accepted=accepted,
            reject_code=reject_code,
            uncertainty=uncertainty_arr,
        )


class BehaviorModelDreamPolicy:
    """Adapter from :class:`BehaviorModel` to the dream behavior protocol."""

    def __init__(self, model: BehaviorModel):
        """Initialize the adapter."""
        self._model = model

    @property
    def model(self) -> BehaviorModel:
        """Wrapped behavior model."""
        return self._model

    def sample_action(
        self,
        state: BehaviorModelState,
        observation: Array,
        key: Array,
    ) -> DreamBehaviorModelPrediction:
        """Sample an imagined action without mutating real behavior state."""
        probabilities = floor_and_renormalize_probabilities(
            self._model.predict_probabilities(state, observation),
            min_probability=self._model.config.min_probability,
        )
        action = jr.categorical(key, jnp.log(probabilities)).astype(jnp.int32)
        action_prob = selected_action_probabilities(
            probabilities,
            action,
            min_probability=self._model.config.min_probability,
        )
        return DreamBehaviorModelPrediction(
            action=action,
            action_probability=action_prob,
            log_probability=jnp.log(action_prob),
        )


def score_dream_candidates(
    surprises: Array,
    utilities: Array,
    *,
    confidences: Array | None = None,
    model_errors: Array | None = None,
    valid: Array | None = None,
    config: DreamSelectionConfig | None = None,
) -> DreamSelectionResult:
    """Score and select surprising/useful candidate dreams.

    Candidates that fail hard gates get score ``-inf`` and cannot be selected
    unless every candidate is rejected, in which case the returned mask remains
    false for those indices.
    """
    cfg = config or DreamSelectionConfig()
    surprise_arr = jnp.ravel(jnp.asarray(surprises, dtype=jnp.float32))
    utility_arr = jnp.ravel(jnp.asarray(utilities, dtype=jnp.float32))
    if surprise_arr.shape != utility_arr.shape:
        raise ValueError("surprises and utilities must have the same shape")
    confidence_arr = (
        jnp.ones_like(surprise_arr)
        if confidences is None
        else jnp.ravel(jnp.asarray(confidences, dtype=jnp.float32))
    )
    error_arr = (
        jnp.zeros_like(surprise_arr)
        if model_errors is None
        else jnp.ravel(jnp.asarray(model_errors, dtype=jnp.float32))
    )
    valid_arr = (
        jnp.ones_like(surprise_arr, dtype=jnp.bool_)
        if valid is None
        else jnp.ravel(jnp.asarray(valid, dtype=jnp.bool_))
    )
    if confidence_arr.shape != surprise_arr.shape:
        raise ValueError("confidences must match surprises")
    if error_arr.shape != surprise_arr.shape:
        raise ValueError("model_errors must match surprises")
    if valid_arr.shape != surprise_arr.shape:
        raise ValueError("valid must match surprises")

    inputs_finite = (
        jnp.isfinite(surprise_arr)
        & jnp.isfinite(utility_arr)
        & jnp.isfinite(confidence_arr)
        & jnp.isfinite(error_arr)
    )
    accepted = (
        valid_arr
        & inputs_finite
        & (surprise_arr >= jnp.asarray(cfg.min_surprise, dtype=jnp.float32))
        & (utility_arr >= jnp.asarray(cfg.min_utility, dtype=jnp.float32))
        & (confidence_arr >= jnp.asarray(cfg.min_confidence, dtype=jnp.float32))
        & (error_arr <= jnp.asarray(cfg.max_model_error, dtype=jnp.float32))
    )
    def weighted_term(weight: float, values: Array) -> Array:
        weight_arr = jnp.asarray(weight, dtype=jnp.float32)
        return jnp.where(
            weight_arr == 0.0,
            jnp.zeros_like(values),
            weight_arr * values,
        )

    raw_scores = (
        weighted_term(cfg.surprise_weight, surprise_arr)
        + weighted_term(cfg.utility_weight, utility_arr)
        + weighted_term(cfg.confidence_weight, confidence_arr)
        - weighted_term(cfg.model_error_weight, error_arr)
    )
    accepted = accepted & jnp.isfinite(raw_scores)
    scores = jnp.where(accepted, raw_scores, -jnp.inf)
    selected_indices = jnp.argsort(-scores)[: cfg.max_items].astype(jnp.int32)
    selected_accepted = accepted[selected_indices]
    selected_mask = jnp.zeros_like(accepted).at[selected_indices].set(selected_accepted)
    return DreamSelectionResult(
        selected_indices=selected_indices,
        scores=scores,
        accepted=accepted,
        selected_mask=selected_mask,
    )


class RecentObservationBuffer:
    """Fixed-size buffer for anchoring dreams in recently observed states."""

    def __init__(self, capacity: int, observation_dim: int):
        """Initialize the buffer shape."""
        self._capacity = _require_int("capacity", capacity, minimum=1)
        self._observation_dim = _require_int(
            "observation_dim", observation_dim, minimum=1
        )
        _require_float32_resource(
            "RecentObservationBuffer",
            vector_scalars=self._capacity * self._observation_dim,
        )

    @property
    def capacity(self) -> int:
        """Maximum number of observations retained."""
        return self._capacity

    @property
    def observation_dim(self) -> int:
        """Observation dimensionality."""
        return self._observation_dim

    def init(self) -> RecentObservationBufferState:
        """Return an empty buffer state."""
        _require_float32_resource(
            "RecentObservationBuffer init",
            vector_scalars=self._capacity * self._observation_dim,
        )
        return RecentObservationBufferState(
            observations=jnp.zeros(
                (self._capacity, self._observation_dim),
                dtype=jnp.float32,
            ),
            size=jnp.array(0, dtype=jnp.int32),
            index=jnp.array(0, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def add(
        self,
        state: RecentObservationBufferState,
        observation: Array,
    ) -> RecentObservationBufferState:
        """Insert one observation into the ring buffer."""
        obs = jnp.asarray(observation, dtype=jnp.float32).reshape(
            (self._observation_dim,)
        )
        next_observations = state.observations.at[state.index].set(obs)
        return RecentObservationBufferState(
            observations=next_observations,
            size=jnp.minimum(state.size + 1, self._capacity).astype(jnp.int32),
            index=((state.index + 1) % self._capacity).astype(jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def sample(
        self,
        state: RecentObservationBufferState,
        key: Array,
    ) -> tuple[Array, Array]:
        """Sample one retained observation.

        Returns the sampled observation and the sampled index. If the buffer is
        empty, index ``0`` and the zero observation are returned.
        """
        sample_size = jnp.maximum(state.size, 1)
        idx = jr.randint(key, (), 0, sample_size).astype(jnp.int32)
        return state.observations[idx], idx


class DreamBehaviorModel(Protocol):
    """Minimal behavior model protocol for rollout-level self-simulation."""

    def sample_action(
        self,
        state: Any,
        observation: Array,
        key: Array,
    ) -> DreamBehaviorModelPrediction:
        """Sample or choose an imagined action."""
        ...


class DreamWorldModel(Protocol):
    """Minimal world model protocol for rollout-level self-simulation."""

    def predict(
        self,
        state: Any,
        observation: Array,
        action: Array,
        key: Array,
    ) -> DreamWorldModelPrediction:
        """Predict one imagined transition."""
        ...


@dataclasses.dataclass(frozen=True)
class DreamRolloutConfig:
    """Configuration for bounded short model rollouts.

    This config complements :class:`DreamingConfig`, which guards one-step
    proposals from the concrete action-conditioned world model.
    """

    rollout_horizon: int = 1
    confidence_threshold: float = 0.0
    max_model_error: float = 1.0e30
    discount_floor: float = 0.0
    stop_on_terminal: bool = True

    def __post_init__(self) -> None:
        """Validate scalar configuration, rejecting NaN and type stand-ins."""
        object.__setattr__(
            self,
            "rollout_horizon",
            _require_int("rollout_horizon", self.rollout_horizon, minimum=1),
        )
        for name in ("confidence_threshold", "max_model_error", "discount_floor"):
            object.__setattr__(
                self,
                name,
                _validated_config_float(name, getattr(self, name), lower=0.0),
            )
        object.__setattr__(
            self, "stop_on_terminal", _require_bool(self.stop_on_terminal, "stop_on_terminal")
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = dataclasses.asdict(self)
        payload["type"] = "DreamRolloutConfig"
        return payload

    @classmethod
    def from_config(cls, config: object) -> DreamRolloutConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = _copy_mapping(config, name="DreamRolloutConfig")
        config_type = data.pop("type", None)
        if type(config_type) is not str or config_type != "DreamRolloutConfig":
            raise ValueError(
                "DreamRolloutConfig payload type must be DreamRolloutConfig"
            )
        allowed = {
            "rollout_horizon",
            "confidence_threshold",
            "max_model_error",
            "discount_floor",
            "stop_on_terminal",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError("DreamRolloutConfig payload has unknown fields")
        return cls(**data)  # type: ignore[arg-type]


@chex.dataclass(frozen=True)
class DreamBehaviorModelPrediction:
    """Action prediction used by rollout-level dreaming."""

    action: Array
    action_probability: Float[Array, ""]
    log_probability: Float[Array, ""]


@chex.dataclass(frozen=True)
class DreamWorldModelPrediction:
    """World-model prediction used by rollout-level dreaming."""

    next_observation: Array
    reward: Float[Array, ""]
    discount: Float[Array, ""]
    terminated: Array
    confidence: Float[Array, ""]
    model_error: Float[Array, ""]


BehaviorModelPrediction = DreamBehaviorModelPrediction
WorldModelPrediction = DreamWorldModelPrediction


@chex.dataclass(frozen=True)
class DreamRolloutState:
    """State carried through an imagined rollout."""

    observation: Array
    rng_key: Array
    active: Array
    cumulative_confidence: Float[Array, ""]
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class ImaginedTransition:
    """Transition generated by a world model and behavior model."""

    observation: Array
    action: Array
    reward: Float[Array, ""]
    next_observation: Array
    discount: Float[Array, ""]
    terminated: Array
    confidence: Float[Array, ""]
    model_error: Float[Array, ""]
    behavior_probability: Float[Array, ""]
    valid: Array
    step_index: Int[Array, ""]


@chex.dataclass(frozen=True)
class DreamRolloutResult:
    """Result from a bounded imagined rollout."""

    state: DreamRolloutState
    transitions: ImaginedTransition


@chex.dataclass(frozen=True)
class DreamSupervisedTrainingItem:
    """Supervised item derived from an imagined transition."""

    inputs: Array
    targets: Array
    weights: Float[Array, ""]


@chex.dataclass(frozen=True)
class DreamGVFTrainingItem:
    """GVF/Horde-style item derived from imagined transitions."""

    observations: Array
    cumulants: Array
    next_observations: Array
    discounts: Array
    weights: Array


@chex.dataclass(frozen=True)
class DreamSARSATrainingItem:
    """SARSA-style item derived from imagined transitions."""

    observations: Array
    actions: Array
    rewards: Array
    next_observations: Array
    discounts: Array
    next_actions: Array
    weights: Array


class ActionConditionedDreamWorld:
    """Adapter from :class:`ActionConditionedWorldModel` to rollout protocol."""

    def __init__(
        self,
        model: ActionConditionedWorldModel,
        *,
        confidence: float = 1.0,
    ):
        """Initialize the adapter."""
        self._model = model
        self._confidence = confidence

    def predict(
        self,
        state: ActionConditionedWorldModelState,
        observation: Array,
        action: Array,
        key: Array,
    ) -> DreamWorldModelPrediction:
        """Predict one dream transition from the wrapped model."""
        del key
        prediction = self._model.predict(state, observation, action)
        return DreamWorldModelPrediction(
            next_observation=prediction.next_observation,
            reward=prediction.reward,
            discount=prediction.discount,
            terminated=prediction.discount <= 0.0,
            confidence=jnp.asarray(self._confidence, dtype=jnp.float32),
            model_error=state.model_error_ema,
        )


def init_dream_rollout_state(
    observation: Array,
    key: Array,
    *,
    active: bool = True,
) -> DreamRolloutState:
    """Create an initial rollout state from a real observation."""
    return DreamRolloutState(
        observation=jnp.asarray(observation),
        rng_key=key,
        active=jnp.asarray(active, dtype=jnp.bool_),
        cumulative_confidence=jnp.array(1.0, dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
    )


def dream_one_step(
    world_model: DreamWorldModel,
    world_state: Any,
    behavior_model: DreamBehaviorModel,
    behavior_state: Any,
    rollout_state: DreamRolloutState,
    config: DreamRolloutConfig | None = None,
) -> tuple[DreamRolloutState, ImaginedTransition]:
    """Generate one imagined transition without mutating real environment state."""
    cfg = config or DreamRolloutConfig()
    key, action_key, model_key = jr.split(rollout_state.rng_key, 3)
    behavior_prediction = behavior_model.sample_action(
        behavior_state,
        rollout_state.observation,
        action_key,
    )
    world_prediction = world_model.predict(
        world_state,
        rollout_state.observation,
        behavior_prediction.action,
        model_key,
    )
    confidence_ok = world_prediction.confidence >= jnp.asarray(
        cfg.confidence_threshold,
        dtype=jnp.float32,
    )
    error_ok = world_prediction.model_error <= jnp.asarray(
        cfg.max_model_error,
        dtype=jnp.float32,
    )
    discount_terminal = world_prediction.discount <= jnp.asarray(
        cfg.discount_floor,
        dtype=jnp.float32,
    )
    terminated = jnp.logical_or(world_prediction.terminated, discount_terminal)
    # A non-finite imagined step can never be valid: it would otherwise ship to
    # the control learner with full weight and poison every later step of the
    # rollout through the carried observation. The anchor observation and the
    # sampled action are gated alongside the model prediction because both are
    # copied verbatim into the emitted transition (mirroring the one-step
    # guard's REJECT_NONFINITE, which also requires a finite anchor).
    finite = (
        jnp.all(jnp.isfinite(jnp.asarray(rollout_state.observation, dtype=jnp.float32)))
        & jnp.all(jnp.isfinite(jnp.asarray(behavior_prediction.action, dtype=jnp.float32)))
        & jnp.all(jnp.isfinite(world_prediction.next_observation))
        & jnp.all(jnp.isfinite(world_prediction.reward))
        & jnp.all(jnp.isfinite(world_prediction.discount))
        & jnp.all(jnp.isfinite(world_prediction.confidence))
        & jnp.all(jnp.isfinite(world_prediction.model_error))
    )
    valid = jnp.logical_and(
        rollout_state.active,
        jnp.logical_and(finite, jnp.logical_and(confidence_ok, error_ok)),
    )
    next_active = jnp.logical_and(valid, jnp.logical_not(terminated))
    if not cfg.stop_on_terminal:
        next_active = valid
    next_observation = jnp.where(
        valid,
        world_prediction.next_observation,
        rollout_state.observation,
    )
    next_state = DreamRolloutState(
        observation=next_observation,
        rng_key=key,
        active=next_active,
        cumulative_confidence=jnp.where(
            valid,
            rollout_state.cumulative_confidence * world_prediction.confidence,
            rollout_state.cumulative_confidence,
        ),
        step_count=_saturating_int32_counter_increment(rollout_state.step_count),
    )
    transition = ImaginedTransition(
        observation=rollout_state.observation,
        action=behavior_prediction.action,
        reward=jnp.squeeze(jnp.asarray(world_prediction.reward, dtype=jnp.float32)),
        next_observation=world_prediction.next_observation,
        discount=jnp.squeeze(jnp.asarray(world_prediction.discount, dtype=jnp.float32)),
        terminated=terminated,
        confidence=jnp.squeeze(jnp.asarray(world_prediction.confidence, dtype=jnp.float32)),
        model_error=jnp.squeeze(jnp.asarray(world_prediction.model_error, dtype=jnp.float32)),
        behavior_probability=jnp.squeeze(
            jnp.asarray(behavior_prediction.action_probability, dtype=jnp.float32)
        ),
        valid=valid,
        step_index=rollout_state.step_count,
    )
    return next_state, transition


def dream_rollout(
    world_model: DreamWorldModel,
    world_state: Any,
    behavior_model: DreamBehaviorModel,
    behavior_state: Any,
    rollout_state: DreamRolloutState,
    config: DreamRolloutConfig,
) -> DreamRolloutResult:
    """Generate a bounded short rollout using ``jax.lax.scan``."""

    def step_fn(
        carry: DreamRolloutState,
        _: Array,
    ) -> tuple[DreamRolloutState, ImaginedTransition]:
        return dream_one_step(
            world_model,
            world_state,
            behavior_model,
            behavior_state,
            carry,
            config,
        )

    final_state, transitions = jax.lax.scan(
        step_fn,
        rollout_state,
        jnp.arange(config.rollout_horizon, dtype=jnp.int32),
    )
    return DreamRolloutResult(state=final_state, transitions=transitions)


def slice_imagined_transition(
    transitions: ImaginedTransition,
    index: int,
) -> ImaginedTransition:
    """Select one transition from a time-leading rollout."""
    return cast(
        ImaginedTransition,
        jax.tree_util.tree_map(lambda value: value[index], transitions),
    )


def action_features(action: Array, n_actions: int | None = None) -> Array:
    """Return float action features for training-item conversion."""
    if n_actions is None:
        return jnp.ravel(jnp.asarray(action, dtype=jnp.float32))
    n_actions = _require_int("n_actions", n_actions, minimum=1)
    action_array = jnp.asarray(action)
    if action_array.shape != () or not jnp.issubdtype(action_array.dtype, jnp.integer):
        raise ValueError("discrete action must be a scalar integer array")
    valid = (action_array >= 0) & (action_array < n_actions)
    action_index = jnp.where(valid, action_array, 0).astype(jnp.int32)
    return jax.nn.one_hot(action_index, n_actions, dtype=jnp.float32) * valid


def _neutralize_invalid(value: Array, valid: Array) -> Array:
    """Return zeros for rejected dream rows before weighted arithmetic."""
    array = jnp.asarray(value)
    mask = jnp.asarray(valid, dtype=jnp.bool_)
    while mask.ndim < array.ndim:
        mask = mask[..., None]
    return jnp.where(mask, array, jnp.zeros_like(array))


def imagined_transition_to_supervised_item(
    transition: ImaginedTransition,
    *,
    n_actions: int | None = None,
    target: Literal["next_observation", "reward", "reward_next_observation"] = (
        "next_observation"
    ),
) -> DreamSupervisedTrainingItem:
    """Convert one imagined transition to a supervised model-learning item."""
    inputs = jnp.concatenate(
        [
            jnp.ravel(jnp.asarray(transition.observation, dtype=jnp.float32)),
            action_features(transition.action, n_actions),
        ],
        axis=0,
    )
    reward = jnp.reshape(jnp.asarray(transition.reward, dtype=jnp.float32), (1,))
    next_observation = jnp.ravel(
        jnp.asarray(transition.next_observation, dtype=jnp.float32)
    )
    if type(target) is not str or target not in {
        "next_observation",
        "reward",
        "reward_next_observation",
    }:
        raise ValueError("supervised target is unsupported")
    if target == "next_observation":
        targets = next_observation
    elif target == "reward":
        targets = reward
    else:
        targets = jnp.concatenate([reward, next_observation], axis=0)
    return DreamSupervisedTrainingItem(
        inputs=_neutralize_invalid(inputs, transition.valid),
        targets=_neutralize_invalid(targets, transition.valid),
        weights=jnp.asarray(transition.valid, dtype=jnp.float32),
    )


def imagined_transition_to_gvf_item(
    transition: ImaginedTransition,
    cumulants: Array | None = None,
) -> DreamGVFTrainingItem:
    """Convert one imagined transition to a GVF/Horde update item."""
    cumulant_array = (
        jnp.reshape(jnp.asarray(transition.reward, dtype=jnp.float32), (1,))
        if cumulants is None
        else jnp.ravel(jnp.asarray(cumulants, dtype=jnp.float32))
    )
    return DreamGVFTrainingItem(
        observations=_neutralize_invalid(transition.observation, transition.valid),
        cumulants=_neutralize_invalid(cumulant_array, transition.valid),
        next_observations=_neutralize_invalid(transition.next_observation, transition.valid),
        discounts=_neutralize_invalid(
            jnp.reshape(jnp.asarray(transition.discount, dtype=jnp.float32), (1,)),
            transition.valid,
        ),
        weights=jnp.reshape(jnp.asarray(transition.valid, dtype=jnp.float32), (1,)),
    )


def imagined_rollout_to_gvf_items(
    rollout: DreamRolloutResult,
    cumulants: Array | None = None,
) -> DreamGVFTrainingItem:
    """Convert a rollout to time-leading GVF/Horde arrays."""
    transitions = rollout.transitions
    cumulant_array = (
        jnp.reshape(jnp.asarray(transitions.reward, dtype=jnp.float32), (-1, 1))
        if cumulants is None
        else jnp.asarray(cumulants, dtype=jnp.float32)
    )
    return DreamGVFTrainingItem(
        observations=_neutralize_invalid(transitions.observation, transitions.valid),
        cumulants=_neutralize_invalid(cumulant_array, transitions.valid),
        next_observations=_neutralize_invalid(
            transitions.next_observation, transitions.valid
        ),
        discounts=_neutralize_invalid(
            jnp.asarray(transitions.discount, dtype=jnp.float32), transitions.valid
        ),
        weights=jnp.asarray(transitions.valid, dtype=jnp.float32),
    )


def imagined_rollout_to_sarsa_items(
    rollout: DreamRolloutResult,
    bootstrap_action: Array | None = None,
) -> DreamSARSATrainingItem:
    """Convert a rollout to SARSA-style arrays with shifted next actions."""
    transitions = rollout.transitions
    actions = transitions.action
    if bootstrap_action is None:
        next_actions = jnp.concatenate([actions[1:], jnp.zeros_like(actions[-1:])], axis=0)
        weights = jnp.asarray(transitions.valid, dtype=jnp.float32)
        weights = weights.at[-1].set(0.0)
    else:
        bootstrap = jnp.expand_dims(jnp.asarray(bootstrap_action, dtype=actions.dtype), axis=0)
        next_actions = jnp.concatenate([actions[1:], bootstrap], axis=0)
        weights = jnp.asarray(transitions.valid, dtype=jnp.float32)
    return DreamSARSATrainingItem(
        observations=_neutralize_invalid(transitions.observation, transitions.valid),
        actions=_neutralize_invalid(actions, transitions.valid),
        rewards=_neutralize_invalid(
            jnp.asarray(transitions.reward, dtype=jnp.float32), transitions.valid
        ),
        next_observations=_neutralize_invalid(
            transitions.next_observation, transitions.valid
        ),
        discounts=_neutralize_invalid(
            jnp.asarray(transitions.discount, dtype=jnp.float32), transitions.valid
        ),
        next_actions=_neutralize_invalid(next_actions, transitions.valid),
        weights=weights,
    )


__all__ = [
    "ActionConditionedDreamWorld",
    "BehaviorModelPrediction",
    "BehaviorModelDreamPolicy",
    "DreamBehaviorModel",
    "DreamBehaviorModelPrediction",
    "DreamGVFTrainingItem",
    "DreamProposal",
    "DreamRolloutConfig",
    "DreamRolloutResult",
    "DreamRolloutState",
    "DreamSARSATrainingItem",
    "DreamSelectionConfig",
    "DreamSelectionResult",
    "DreamSupervisedTrainingItem",
    "DreamTransition",
    "DreamWorldModel",
    "DreamWorldModelPrediction",
    "DreamingConfig",
    "GuardedDreamer",
    "ImaginedTransition",
    "RecentObservationBuffer",
    "RecentObservationBufferState",
    "WorldModelPrediction",
    "action_features",
    "dream_one_step",
    "dream_rollout",
    "imagined_rollout_to_gvf_items",
    "imagined_rollout_to_sarsa_items",
    "imagined_transition_to_gvf_item",
    "imagined_transition_to_supervised_item",
    "init_dream_rollout_state",
    "score_dream_candidates",
    "slice_imagined_transition",
]
