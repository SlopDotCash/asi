# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Core types and algorithms for Intelligence Amplification (Alberta Plan Step 12).

Step 12 — "Prototype-IA: Intelligence Amplification" — shifts from autonomous
agent to *augmenting a partner agent's intelligence*.  The IA agent observes
the same environment as the partner and provides two complementary streams of
augmentation:

**Exo-cerebellum** — A multi-output online linear predictor that continuously
learns to predict future observation features from the current observation.  The
prediction vector is broadcast to the partner so it can use anticipated future
state information as augmented features.  This implements the "sensorimotor
predictions" concept from Pilarski & Sutton's communicative capital work
(Mathewson et al. 2023).

**Exo-cortex** — An :class:`~alberta_framework.core.oak.OaKAgent` that observes
the partner's states and rewards, learning its own Q-function over the same
environment.  It broadcasts an action recommendation at each step by taking the
argmax of its current Q-values.  The partner can accept or ignore the
recommendation.

Together, the :class:`IAAgent` provides at every step:
* A prediction vector ``predictions`` of shape ``(n_demons,)`` — future feature
  estimates from the exo-cerebellum.
* An action recommendation ``recommendation`` of shape ``()`` — the cortex's
  greedy action choice.
* An augmented observation ``augmented_obs`` of shape
  ``(obs_dim + n_demons,)`` = ``concat(partner_obs, predictions)``, ready to
  drop into the partner's feature pipeline.

References:
    Sutton, Bowling, & Pilarski (2022). "The Alberta Plan for AI Research."
    Mathewson et al. (2023). "Communicative Capital: A Key Resource for
        Human-Machine Shared Agency." *Neural Computing & Applications* 35(23).
"""

from __future__ import annotations

import dataclasses
import math
import operator
from numbers import Real
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework._float32 import round_real_to_float32
from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.normalizers import (
    _checked_lifetime_words_increment,
    _lifetime_counter_valid,
    _saturating_int32_counter_increment,
)
from alberta_framework.core.oak import (
    OAK_LIFETIME_COUNTER_NBYTES,
    OaKAgent,
    OaKConfig,
    OaKState,
    OaKUpdateResult,
    _default_stomp_config,
    _oak_outer_state_validity,
    measure_oak_state_nbytes,
    migrate_legacy_oak_state,
    oak_total_lifetime_counter_nbytes,
)
from alberta_framework.core.update_safety import checked_integer_action_array

EXO_CEREBELLUM_STATE_SCHEMA = "alberta.exo-cerebellum-state.v2"
IA_STATE_SCHEMA = "alberta.intelligence-amplification-state.v2"
RECOMMENDATION_PROTOCOL_STATE_SCHEMA = "alberta.recommendation-protocol-state.v2"

# Every compatibility telemetry scalar occupies four bytes and every exact
# big-endian uint32 word pair occupies eight.  The IA total includes its outer
# and cerebellum clocks plus OaK's outer, STOMP, and conditional base-update
# clocks.  Only the first two pairs are introduced by this module's v2 state.
EXO_CEREBELLUM_LIFETIME_COUNTER_NBYTES = 12
EXO_CEREBELLUM_LIFETIME_COUNTER_DELTA_NBYTES = 8
IA_LIFETIME_COUNTER_NBYTES = (
    2 * EXO_CEREBELLUM_LIFETIME_COUNTER_NBYTES + 3 * OAK_LIFETIME_COUNTER_NBYTES
)
IA_LIFETIME_COUNTER_DELTA_NBYTES = 2 * EXO_CEREBELLUM_LIFETIME_COUNTER_DELTA_NBYTES
RECOMMENDATION_PROTOCOL_LIFETIME_COUNTER_NBYTES = 36
RECOMMENDATION_PROTOCOL_LIFETIME_COUNTER_DELTA_NBYTES = 24

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
    actual_type = type(value)
    if not any(actual_type is allowed_type for allowed_type in _ACTUAL_INT_TYPES):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _require_derived_int32(name: str, value: int, *, minimum: int = 0) -> int:
    if not minimum <= value <= _INT32_MAX:
        raise ValueError(f"derived {name} must fit signed int32")
    return value


def _require_float32_resource(name: str, *, float32_scalars: int) -> None:
    _require_derived_int32(f"{name} scalar count", float32_scalars)
    if 4 * float32_scalars > _INT32_MAX:
        raise ValueError(f"derived {name} byte count must fit signed int32")


def _preflight_cerebellum_resources(n_demons: int, obs_dim: int) -> None:
    """Reject persistent and update working sets the cerebellum cannot name."""

    weight_scalars = n_demons * obs_dim
    # The agent owns the float32 weights, one int32 cumulant-index vector,
    # one int32 telemetry scalar, and two uint32 lifetime words.  Every leaf
    # occupies four bytes, so account for the aggregate rather than merely
    # proving that the largest single array is nameable.
    persistent_scalars = weight_scalars + n_demons + 3
    _require_float32_resource(
        "exo-cerebellum persistent state",
        float32_scalars=persistent_scalars,
    )
    # During update, source weights, the outer-product delta, candidate
    # weights, and the selected result weights are simultaneously live.  The
    # remaining term covers the retained cumulant indices, predictions,
    # targets, errors, neutral result vectors, two raw/safe observations,
    # lifetime words, and transaction predicates. Charging every element as
    # four bytes conservatively covers int32, uint32, and boolean arrays too.
    update_scalars = 4 * weight_scalars + 6 * n_demons + 4 * obs_dim + 16
    _require_float32_resource(
        "exo-cerebellum update working set",
        float32_scalars=update_scalars,
    )
    _require_derived_int32("augmented observation dim", n_demons + obs_dim, minimum=2)
    _require_float32_resource(
        "augmented observation",
        float32_scalars=n_demons + obs_dim,
    )


def _positive_float32_scalar(name: str, value: object) -> float:
    """Validate and canonicalize a positive scalar in its execution dtype."""
    message = f"{name} must be a finite positive float32 scalar"
    preserve_builtin_float = type(value) is float
    preserve_builtin_int = type(value) is int
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        if value <= 0:
            raise ValueError
        narrowed = round_real_to_float32(value)
    except (FloatingPointError, OverflowError, TypeError, ValueError) as error:
        raise ValueError(message) from error
    if not math.isfinite(narrowed) or narrowed <= 0.0:
        raise ValueError(message)

    # A valid built-in float already has an exact binary64 payload, so keeping
    # it preserves existing serialized configs without changing its float32
    # sink. Other Real implementations are stored as that exact sink. A small
    # built-in integer is likewise safe only when it is exactly representable.
    if preserve_builtin_float:
        return cast(float, value)
    if preserve_builtin_int and value == narrowed:
        return float(value)
    return narrowed


# ---------------------------------------------------------------------------
# Exo-cerebellum
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExoCerebellumConfig:
    """Configuration for the exo-cerebellum online predictor.

    Each demon i predicts ``next_obs[i % obs_dim]`` from the current
    observation using a linear TD(0) update.

    Args:
        n_demons: Number of prediction heads.
        obs_dim: Flat observation dimensionality.
        step_size: Learning rate for weight updates.
    """

    n_demons: int = 4
    obs_dim: int = 4
    step_size: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_demons",
            _require_int32("n_demons", self.n_demons, minimum=1),
        )
        object.__setattr__(
            self,
            "obs_dim",
            _require_int32("obs_dim", self.obs_dim, minimum=1),
        )
        step_size = _positive_float32_scalar("step_size", self.step_size)
        object.__setattr__(self, "step_size", step_size)
        _preflight_cerebellum_resources(self.n_demons, self.obs_dim)

    def to_config(self) -> dict[str, Any]:
        return {"type": "ExoCerebellumConfig", **dataclasses.asdict(self)}

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> ExoCerebellumConfig:
        data = dict(payload)
        data.pop("type", None)
        return cls(**data)


@chex.dataclass(frozen=True)
class ExoCerebellumState:
    """State of the exo-cerebellum predictor.

    Attributes:
        weights: Linear prediction weights; shape ``(n_demons, obs_dim)``.
        step_count: Saturating int32 update-count telemetry.
        step_words: Exact big-endian uint32 update identity.  The all-ones
            value is terminal and is never wrapped.
    """

    weights: Float[Array, "n_demons obs_dim"]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"] = dataclasses.field(
        default_factory=lambda: jnp.zeros((2,), dtype=jnp.uint32)
    )


@chex.dataclass(frozen=True)
class ExoCerebellumUpdateResult:
    """Transactional result for one exo-cerebellum update."""

    state: ExoCerebellumState
    predictions: Float[Array, " n_demons"]
    errors: Float[Array, " n_demons"]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    source_state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


class ExoCerebellumAgent:
    """Online multi-output linear predictor for Step 12 IA.

    Demon ``i`` learns to predict ``next_obs[i % obs_dim]`` from the current
    observation via a one-step supervised / TD(0) update:

    ``error = next_obs[i % obs_dim] - weights[i] @ obs``
    ``weights[i] += alpha * error * obs``
    """

    def __init__(self, config: ExoCerebellumConfig) -> None:
        self._config = config
        self._cumulant_indices = jnp.arange(config.n_demons, dtype=jnp.int32) % config.obs_dim

    @property
    def config(self) -> ExoCerebellumConfig:
        return self._config

    def to_config(self) -> dict[str, Any]:
        return self._config.to_config()

    def init(self) -> ExoCerebellumState:
        """Initialise with zero prediction weights."""
        return ExoCerebellumState(
            weights=jnp.zeros((self._config.n_demons, self._config.obs_dim), dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _require_state_contract(self, state: ExoCerebellumState) -> None:
        """Reject a malformed PyTree before any numerical update is staged."""

        if not isinstance(state, ExoCerebellumState):
            raise TypeError("state must be ExoCerebellumState")
        if (
            state.weights.shape != (self._config.n_demons, self._config.obs_dim)
            or state.weights.dtype != jnp.float32
        ):
            raise ValueError("state.weights has an invalid shape or dtype")
        if state.step_count.shape != () or state.step_count.dtype != jnp.int32:
            raise ValueError("state.step_count must be scalar int32")
        if state.step_words.shape != (2,) or state.step_words.dtype != jnp.uint32:
            raise ValueError("state.step_words must have shape (2,) and dtype uint32")

    def state_is_valid(self, state: ExoCerebellumState) -> Bool[Array, ""]:
        """Authenticate weights and the exact/telemetry lifetime relation."""

        self._require_state_contract(state)
        return jnp.all(jnp.isfinite(state.weights)) & _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )

    def predict(self, state: ExoCerebellumState, observation: Array) -> Array:
        """Compute current predictions from an observation.

        Args:
            state: Current cerebellum state.
            observation: Shape ``(obs_dim,)`` float32.

        Returns:
            Shape ``(n_demons,)`` predictions.
        """
        self._require_state_contract(state)
        obs = jnp.asarray(observation, dtype=jnp.float32)
        if obs.shape != (self._config.obs_dim,):
            raise ValueError(
                f"observation must have shape {(self._config.obs_dim,)}, got {obs.shape}"
            )
        return state.weights @ obs

    def update_result(
        self,
        state: ExoCerebellumState,
        observation: Array,
        next_observation: Array,
    ) -> ExoCerebellumUpdateResult:
        """Stage and atomically commit one exact-identity predictor update."""

        self._require_state_contract(state)
        obs = jnp.asarray(observation, dtype=jnp.float32)
        next_obs = jnp.asarray(next_observation, dtype=jnp.float32)
        expected_shape = (self._config.obs_dim,)
        if obs.shape != expected_shape:
            raise ValueError(f"observation must have shape {expected_shape}, got {obs.shape}")
        if next_obs.shape != expected_shape:
            raise ValueError(
                f"next_observation must have shape {expected_shape}, got {next_obs.shape}"
            )

        source_state_valid = self.state_is_valid(state)
        input_valid = jnp.all(jnp.isfinite(obs)) & jnp.all(jnp.isfinite(next_obs))
        safe_obs = jnp.where(jnp.isfinite(obs), obs, jnp.float32(0.0))
        safe_next_obs = jnp.where(jnp.isfinite(next_obs), next_obs, jnp.float32(0.0))
        proposed_step_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        alpha = jnp.asarray(self._config.step_size, dtype=jnp.float32)
        predictions = state.weights @ safe_obs
        targets = safe_next_obs[self._cumulant_indices]
        errors = targets - predictions
        candidate_state = ExoCerebellumState(
            weights=state.weights + alpha * jnp.outer(errors, safe_obs),
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_step_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        update_applied = (
            source_state_valid & input_valid & lifetime_capacity_available & candidate_state_valid
        )
        new_state = jax.tree.map(
            lambda candidate, source: jnp.where(update_applied, candidate, source),
            candidate_state,
            state,
        )
        return ExoCerebellumUpdateResult(
            state=new_state,
            predictions=jnp.where(update_applied, predictions, jnp.zeros_like(predictions)),
            errors=jnp.where(update_applied, errors, jnp.zeros_like(errors)),
            pre_step_words=state.step_words,
            post_step_words=new_state.step_words,
            source_state_valid=source_state_valid,
            input_valid=input_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            candidate_state_valid=candidate_state_valid,
            update_applied=update_applied,
        )

    def update(
        self,
        state: ExoCerebellumState,
        observation: Array,
        next_observation: Array,
    ) -> tuple[ExoCerebellumState, Array, Array]:
        """One-step supervised update for all demons.

        Args:
            state: Current cerebellum state.
            observation: Current observation ``s_t``, shape ``(obs_dim,)``.
            next_observation: Next observation ``s_{t+1}``, shape ``(obs_dim,)``.

        Returns:
            ``(new_state, predictions, errors)`` where predictions are computed
            *before* the weight update, matching the typical RL convention.
        """
        result = self.update_result(state, observation, next_observation)
        return result.state, result.predictions, result.errors


# ---------------------------------------------------------------------------
# Exo-cortex (thin wrapper around OaKAgent)
# ---------------------------------------------------------------------------


def _default_oak_config() -> OaKConfig:
    """Default OaK config for the exo-cortex."""
    from alberta_framework.core.oak import OaKConfig  # local to avoid circular at module level

    return OaKConfig(
        stomp=_default_stomp_config(),
        utility_ema_decay=0.99,
        curation_threshold=0.0,
    )


# ExoCortexConfig is just OaKConfig; the type alias makes the Step 12 API clear.
ExoCortexConfig = OaKConfig
ExoCortexState = OaKState


def _checked_partner_action(
    action: Array,
    *,
    n_primitive_actions: int,
) -> tuple[Int[Array, ""], Bool[Array, ""]]:
    """Validate one primitive action eagerly and poison invalid traced input."""
    raw = jnp.asarray(action)
    if raw.shape != ():
        raise ValueError("partner_action must be scalar")
    if not jnp.issubdtype(raw.dtype, jnp.integer):
        raise ValueError("partner_action must have an integer dtype")
    valid = (raw >= 0) & (raw < n_primitive_actions)
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError(f"partner_action must be in [0, {n_primitive_actions})")
    executed = jnp.where(valid, raw, 0).astype(jnp.int32)
    return (
        executed,
        valid,
    )


class ExoCortexAgent:
    """Exo-cortex: an OaKAgent that provides action recommendations.

    Learns from the partner's (obs, reward, next_obs) experience and
    broadcasts its greedy action recommendation at each step.
    """

    def __init__(self, config: ExoCortexConfig) -> None:
        self._oak = OaKAgent(config)

    @property
    def config(self) -> ExoCortexConfig:
        return self._oak.config

    @property
    def oak_agent(self) -> OaKAgent:
        return self._oak

    def to_config(self) -> dict[str, Any]:
        return self._oak.to_config()

    def init(self, key: Array) -> ExoCortexState:
        return self._oak.init(key)

    def start(self, state: ExoCortexState, initial_obs: Array) -> ExoCortexState:
        return self._oak.start(state, initial_obs)

    def recommend(self, state: ExoCortexState, observation: Array) -> Int[Array, ""]:
        """Return the greedy primitive action for a given observation."""
        obs = jnp.asarray(observation, dtype=jnp.float32)
        q_vals = self._oak.base_q_values(state, obs)
        n_prim = self._oak.config.n_primitive_actions
        q_prim = q_vals[:n_prim]
        return jnp.argmax(q_prim).astype(jnp.int32)

    def state_is_valid(self, state: ExoCortexState) -> Bool[Array, ""]:
        """Authenticate the OaK wrapper, STOMP subtree, and aligned clocks."""

        outer_valid = _oak_outer_state_validity(state, self.config)[-1]
        nested_valid = self._oak.stomp_agent.state_valid(state.stomp_state)
        return (
            outer_valid & nested_valid & jnp.all(state.step_words == state.stomp_state.step_words)
        )

    def _prepare_update_source(
        self,
        state: ExoCortexState,
        partner_action: Array | None,
        discount: Array | None,
    ) -> tuple[ExoCortexState, Array | None, Bool[Array, ""]]:
        """Bind the executed primitive to STOMP's authenticated action owner.

        ``ExoCortexAgent`` is an off-policy adapter around OaK.  OaK still
        validates its complete source unchanged; this adapter first makes the
        explicit partner execution (or the legacy selected owner) internally
        consistent across STOMP's three dispatch fields.
        """

        routed_discount = discount
        action_valid = jnp.asarray(True, dtype=jnp.bool_)
        stomp_state = state.stomp_state
        if partner_action is not None:
            n_prim = self._oak.config.n_primitive_actions
            executed, action_valid = _checked_partner_action(
                partner_action,
                n_primitive_actions=n_prim,
            )
            supplied_discount = (
                jnp.array(1.0, dtype=jnp.float32)
                if discount is None
                else jnp.asarray(discount, dtype=jnp.float32).reshape(())
            )
            routed_discount = jnp.where(
                action_valid,
                supplied_discount,
                jnp.array(jnp.nan, dtype=jnp.float32),
            )
            stomp_state = stomp_state.replace(
                base_last_action=executed,
                last_primitive_action=executed,
                option_last_intra_action=executed,
                executing_option=jnp.array(-1, dtype=jnp.int32),
            )
        else:
            # Historical ExoCortex callers supplied the selected base/option
            # owner but predated STOMP's explicit duplicated primitive owner.
            # Reassert that same owner; no clock, learning value, or action is
            # inferred from outside the stored dispatch tuple.
            owned_primitive = jnp.where(
                stomp_state.executing_option == -1,
                stomp_state.base_last_action,
                stomp_state.option_last_intra_action,
            )
            stomp_state = stomp_state.replace(
                last_primitive_action=owned_primitive,
            )
        return (
            cast(ExoCortexState, state.replace(stomp_state=stomp_state)),
            routed_discount,
            action_valid,
        )

    def _update_result(
        self,
        state: ExoCortexState,
        partner_reward: Array,
        partner_next_obs: Array,
        partner_action: Array | None = None,
        *,
        discount: Array | None = None,
        decision_observation: Array | None = None,
        execution_boundary: Array | bool = False,
    ) -> tuple[OaKUpdateResult, Int[Array, ""], Float[Array, ""]]:
        """Return the complete OaK transaction without widening the public API."""

        state, routed_discount, action_valid = self._prepare_update_source(
            state,
            partner_action,
            discount,
        )
        decision_obs = partner_next_obs if decision_observation is None else decision_observation
        result = self._oak.update(
            state,
            partner_reward,
            partner_next_obs,
            routed_discount,
            decision_observation=decision_obs,
            execution_boundary=execution_boundary,
        )
        recommendation = self.recommend(result.state, decision_obs)
        output_td_error = jnp.where(
            action_valid,
            result.td_error,
            jnp.asarray(jnp.nan, dtype=jnp.float32),
        )
        return result, recommendation, output_td_error

    def update(
        self,
        state: ExoCortexState,
        partner_reward: Array,
        partner_next_obs: Array,
        partner_action: Array | None = None,
        *,
        discount: Array | None = None,
        decision_observation: Array | None = None,
        execution_boundary: Array | bool = False,
    ) -> tuple[ExoCortexState, Int[Array, ""], Float[Array, ""]]:
        """Update cortex from partner experience and return recommendation.

        When ``partner_action`` is provided, the Q-update credits the primitive
        action the partner actually *executed* (the ``effective_action`` from
        :func:`update_recommendation_protocol`) rather than the action the
        cortex's own epsilon-greedy selection happened to pick — off-policy
        learning about the recommendation policy from the partner's behaviour
        stream.  Each partner step is treated as a primitive (duration-1)
        transition, since the partner — not the cortex — controls execution.
        No importance-sampling correction is needed: the differential
        Q-learning target bootstraps through ``max`` and is off-policy by
        construction. This executed-action path intentionally exits any
        exo-cortex option and updates the base learner only.

        When ``partner_action`` is ``None`` the legacy behaviour (crediting
        the cortex's own selected action, including its option lifecycle) is
        preserved.

        ``discount`` is the environment's effective continuation multiplier
        for this transition. Supplying it reaches whichever base/intra-option
        path owns the cortex transition. With an explicit ``partner_action``,
        that is the primitive base learner described above. Omitting the
        discount preserves OaK's historical primitive/``option_gamma``
        behaviour.

        Returns ``(new_state, recommendation, td_error)``.
        """
        result, recommendation, output_td_error = self._update_result(
            state,
            partner_reward,
            partner_next_obs,
            partner_action=partner_action,
            discount=discount,
            decision_observation=decision_observation,
            execution_boundary=execution_boundary,
        )
        return result.state, recommendation, output_td_error


# ---------------------------------------------------------------------------
# Combined IA agent
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class IAConfig:
    """Configuration for the Intelligence Amplification agent.

    Args:
        cerebellum: Exo-cerebellum configuration.
        cortex: Exo-cortex (OaK) configuration.
    """

    cerebellum: ExoCerebellumConfig = dataclasses.field(default_factory=ExoCerebellumConfig)
    cortex: ExoCortexConfig = dataclasses.field(default_factory=_default_oak_config)

    def __post_init__(self) -> None:
        if self.cerebellum.obs_dim != self.cortex.observation_dim:
            raise ValueError(
                f"cerebellum.obs_dim ({self.cerebellum.obs_dim}) must equal "
                f"cortex.observation_dim ({self.cortex.observation_dim})"
            )
        _require_derived_int32(
            "augmented observation dim",
            self.augmented_obs_dim,
            minimum=2,
        )
        _require_float32_resource(
            "augmented observation",
            float32_scalars=self.augmented_obs_dim,
        )

    @property
    def augmented_obs_dim(self) -> int:
        """Dimension of the concatenated [partner_obs, predictions] vector."""
        return self.cortex.observation_dim + self.cerebellum.n_demons

    def to_config(self) -> dict[str, Any]:
        return {
            "type": "IAConfig",
            "cerebellum": self.cerebellum.to_config(),
            "cortex": self.cortex.to_config(),
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> IAConfig:
        data = dict(payload)
        data.pop("type", None)
        cerebellum = ExoCerebellumConfig.from_config(cast(dict[str, Any], data["cerebellum"]))
        cortex = OaKConfig.from_config(cast(dict[str, Any], data["cortex"]))
        return cls(cerebellum=cerebellum, cortex=cortex)


@chex.dataclass(frozen=True)
class IAState:
    """Combined IA agent state.

    Attributes:
        cerebellum_state: Exo-cerebellum weight state.
        cortex_state: Exo-cortex OaK agent state.
        step_count: Saturating int32 primitive-step telemetry.
        step_words: Exact big-endian uint32 primitive-step identity.  It is
            aligned with the cerebellum, OaK, and nested STOMP primitive
            clocks; ``start`` consumes no primitive step.
    """

    cerebellum_state: ExoCerebellumState
    cortex_state: ExoCortexState
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"] = dataclasses.field(
        default_factory=lambda: jnp.zeros((2,), dtype=jnp.uint32)
    )


@chex.dataclass(frozen=True)
class IAUpdateResult:
    """Result of one IA primitive step.

    Attributes:
        state: New combined IA state.
        predictions: Exo-cerebellum output *before* the weight update;
            shape ``(n_demons,)``.
        cerebellum_errors: Per-demon prediction errors; shape ``(n_demons,)``.
        recommendation: Exo-cortex greedy action recommendation.
        augmented_obs: ``concat(partner_obs, predictions)``; shape
            ``(obs_dim + n_demons,)``.
        cortex_td_error: TD error from the cortex Q-update.
    """

    state: IAState
    predictions: Float[Array, " n_demons"]
    cerebellum_errors: Float[Array, " n_demons"]
    recommendation: Int[Array, ""]
    augmented_obs: Float[Array, " augmented_dim"]
    cortex_td_error: Float[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    source_state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    child_clocks_aligned: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    cerebellum_update_applied: Bool[Array, ""]
    cortex_update_applied: Bool[Array, ""]
    proposed_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class IAArrayResult:
    """Scan result for the IA agent over transition arrays."""

    state: IAState
    predictions: Float[Array, "num_steps n_demons"]
    cerebellum_errors: Float[Array, "num_steps n_demons"]
    recommendations: Int[Array, " num_steps"]
    augmented_obs: Float[Array, "num_steps augmented_dim"]
    cortex_td_errors: Float[Array, " num_steps"]
    pre_step_words: UInt[Array, "num_steps 2"]
    post_step_words: UInt[Array, "num_steps 2"]
    source_state_valid: Bool[Array, " num_steps"]
    input_valid: Bool[Array, " num_steps"]
    child_clocks_aligned: Bool[Array, " num_steps"]
    lifetime_capacity_available: Bool[Array, " num_steps"]
    cerebellum_updates_applied: Bool[Array, " num_steps"]
    cortex_updates_applied: Bool[Array, " num_steps"]
    proposed_state_valid: Bool[Array, " num_steps"]
    updates_applied: Bool[Array, " num_steps"]


@dataclasses.dataclass(frozen=True)
class RecommendationProtocolConfig:
    """Configuration for recommendation acceptance/rejection feedback."""

    acceptance_ema_decay: float = 0.95

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "acceptance_ema_decay",
            validated_float32_scalar(
                "acceptance_ema_decay",
                self.acceptance_ema_decay,
                lower=0.0,
                upper=1.0,
                upper_inclusive=False,
            ),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "type": "RecommendationProtocolConfig",
            "acceptance_ema_decay": self.acceptance_ema_decay,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> RecommendationProtocolConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = dict(payload)
        data.pop("type", None)
        return cls(**data)


@chex.dataclass(frozen=True)
class RecommendationProtocolState:
    """State for partner acceptance/rejection feedback.

    The three int32 counters are compatibility telemetry.  Their word pairs
    are exact authorities, and ``accepted_words + rejected_words ==
    step_words`` must hold without uint64 overflow.
    """

    accepted_count: Int[Array, ""]
    rejected_count: Int[Array, ""]
    acceptance_ema: Float[Array, ""]
    step_count: Int[Array, ""]
    accepted_words: UInt[Array, " 2"]
    rejected_words: UInt[Array, " 2"]
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class RecommendationProtocolResult:
    """Result of one recommendation feedback event."""

    state: RecommendationProtocolState
    recommendation: Int[Array, ""]
    partner_action: Int[Array, ""]
    effective_action: Int[Array, ""]
    accepted: Bool[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    source_state_valid: Bool[Array, ""]
    exact_partition_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    selected_counter_capacity_available: Bool[Array, ""]
    proposed_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


def _words_sum(
    left: Array,
    right: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Add two exact word pairs and report whether uint64 overflow was avoided."""

    for name, words in (("left", left), ("right", right)):
        if getattr(words, "shape", None) != (2,):
            raise ValueError(f"{name} words must have shape (2,)")
        if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
            raise TypeError(f"{name} words must have dtype uint32")
    low = left[1] + right[1]
    carry = (low < left[1]).astype(jnp.uint32)
    high_without_carry = left[0] + right[0]
    overflow_high = high_without_carry < left[0]
    high = high_without_carry + carry
    overflow_carry = high < high_without_carry
    return jnp.stack((high, low)), ~(overflow_high | overflow_carry)


def _recommendation_protocol_state_contract(
    state: RecommendationProtocolState,
) -> None:
    """Require the exact protocol PyTree layout."""

    if not isinstance(state, RecommendationProtocolState):
        raise TypeError("state must be RecommendationProtocolState")
    for name in ("accepted_count", "rejected_count", "step_count"):
        value = getattr(state, name)
        if value.shape != () or value.dtype != jnp.int32:
            raise ValueError(f"state.{name} must be scalar int32")
    if state.acceptance_ema.shape != () or state.acceptance_ema.dtype != jnp.float32:
        raise ValueError("state.acceptance_ema must be scalar float32")
    for name in ("accepted_words", "rejected_words", "step_words"):
        value = getattr(state, name)
        if value.shape != (2,) or value.dtype != jnp.uint32:
            raise ValueError(f"state.{name} must have shape (2,) and dtype uint32")


def recommendation_protocol_state_is_valid(
    state: RecommendationProtocolState,
) -> Bool[Array, ""]:
    """Authenticate exact counters, saturated telemetry, and the partition."""

    _recommendation_protocol_state_contract(state)
    total_words, total_available = _words_sum(
        state.accepted_words,
        state.rejected_words,
    )
    exact_partition_valid = total_available & jnp.all(total_words == state.step_words)
    return (
        _lifetime_counter_valid(state.accepted_words, state.accepted_count)
        & _lifetime_counter_valid(state.rejected_words, state.rejected_count)
        & _lifetime_counter_valid(state.step_words, state.step_count)
        & exact_partition_valid
        & jnp.isfinite(state.acceptance_ema)
        & (state.acceptance_ema >= 0.0)
        & (state.acceptance_ema <= 1.0)
    )


def init_recommendation_protocol_state() -> RecommendationProtocolState:
    """Initialize recommendation feedback counters."""
    return RecommendationProtocolState(
        accepted_count=jnp.array(0, dtype=jnp.int32),
        rejected_count=jnp.array(0, dtype=jnp.int32),
        acceptance_ema=jnp.array(0.0, dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
        accepted_words=jnp.zeros((2,), dtype=jnp.uint32),
        rejected_words=jnp.zeros((2,), dtype=jnp.uint32),
        step_words=jnp.zeros((2,), dtype=jnp.uint32),
    )


def update_recommendation_protocol(
    config: RecommendationProtocolConfig,
    state: RecommendationProtocolState,
    recommendation: Array,
    partner_action: Array,
    accept_recommendation: Array | None = None,
) -> RecommendationProtocolResult:
    """Apply a partner-controlled recommendation before environment execution.

    ``partner_action`` is the partner's proposed fallback action, not an action
    that has already been executed.  ``accept_recommendation`` is a scalar
    boolean decision made by the partner from information available at action
    selection time.  When it is true, the recommendation is the effective
    action even when it differs from the fallback; when false, the fallback is
    preserved.  The protocol therefore selects one of two already-bounded
    action candidates and has no access to reward, next observation, or other
    post-transition information.

    For compatibility, omitting ``accept_recommendation`` retains the original
    agreement-accounting behaviour: matching candidates count as accepted and
    differing candidates count as rejected.  That legacy path cannot intervene
    and should not be used by new closed-loop integrations.

    Feed ``effective_action`` into the environment and then back into
    :meth:`IAAgent.update` as ``partner_action`` so the exo-cortex credits the
    action that was actually executed.

    Args:
        config: Acceptance statistics configuration.
        state: Current protocol counters.
        recommendation: IA action recommendation, produced before acting.
        partner_action: Partner's proposed fallback action, produced before
            acting.
        accept_recommendation: Partner's explicit pre-action accept/reject
            decision.  ``None`` enables the legacy agreement-only path.

    Returns:
        Updated counters and the action selected for environment execution.
    """
    _recommendation_protocol_state_contract(state)
    raw_rec = jnp.asarray(recommendation)
    raw_action = jnp.asarray(partner_action)
    if raw_rec.ndim != 0 or raw_action.ndim != 0:
        raise ValueError("recommendation and partner_action must be scalar")
    if not jnp.issubdtype(raw_rec.dtype, jnp.integer) or not jnp.issubdtype(
        raw_action.dtype,
        jnp.integer,
    ):
        raise TypeError("recommendation and partner_action must have integer dtypes")
    rec = raw_rec.astype(jnp.int32)
    action = raw_action.astype(jnp.int32)
    if accept_recommendation is None:
        accepted = rec == action
    else:
        accepted = jnp.asarray(accept_recommendation)
        if accepted.ndim != 0 or accepted.dtype != jnp.bool_:
            raise TypeError("accept_recommendation must be a scalar boolean")
    accepted_f = accepted.astype(jnp.float32)
    decay = jnp.asarray(config.acceptance_ema_decay, dtype=jnp.float32)
    source_state_valid = recommendation_protocol_state_is_valid(state)
    source_total_words, source_sum_available = _words_sum(
        state.accepted_words,
        state.rejected_words,
    )
    exact_partition_valid = source_sum_available & jnp.all(source_total_words == state.step_words)
    next_step_words, lifetime_capacity_available = _checked_lifetime_words_increment(
        state.step_words
    )
    next_accepted_words, accepted_capacity_available = _checked_lifetime_words_increment(
        state.accepted_words
    )
    next_rejected_words, rejected_capacity_available = _checked_lifetime_words_increment(
        state.rejected_words
    )
    selected_counter_capacity_available = jnp.where(
        accepted,
        accepted_capacity_available,
        rejected_capacity_available,
    )
    candidate_state = RecommendationProtocolState(
        accepted_count=jnp.where(
            accepted,
            _saturating_int32_counter_increment(state.accepted_count),
            state.accepted_count,
        ),
        rejected_count=jnp.where(
            accepted,
            state.rejected_count,
            _saturating_int32_counter_increment(state.rejected_count),
        ),
        acceptance_ema=decay * state.acceptance_ema + (1.0 - decay) * accepted_f,
        step_count=_saturating_int32_counter_increment(state.step_count),
        accepted_words=jnp.where(
            accepted,
            next_accepted_words,
            state.accepted_words,
        ),
        rejected_words=jnp.where(
            accepted,
            state.rejected_words,
            next_rejected_words,
        ),
        step_words=next_step_words,
    )
    proposed_state_valid = recommendation_protocol_state_is_valid(candidate_state)
    update_applied = (
        source_state_valid
        & lifetime_capacity_available
        & selected_counter_capacity_available
        & proposed_state_valid
    )
    new_state = jax.tree_util.tree_map(
        lambda candidate, source: jnp.where(update_applied, candidate, source),
        candidate_state,
        state,
    )
    return RecommendationProtocolResult(
        state=new_state,
        recommendation=rec,
        partner_action=action,
        effective_action=jnp.where(update_applied & accepted, rec, action),
        accepted=update_applied & accepted,
        pre_step_words=state.step_words,
        post_step_words=new_state.step_words,
        source_state_valid=source_state_valid,
        exact_partition_valid=exact_partition_valid,
        lifetime_capacity_available=lifetime_capacity_available,
        selected_counter_capacity_available=selected_counter_capacity_available,
        proposed_state_valid=proposed_state_valid,
        update_applied=update_applied,
    )


class IAAgent:
    """Alberta Plan Step 12 Intelligence Amplification agent.

    Combines an :class:`ExoCerebellumAgent` and an :class:`ExoCortexAgent` to
    augment a partner's decision-making.  At each step the IA agent:

    1. Computes cerebellum predictions from ``partner_obs``.
    2. Updates the cerebellum weights from ``(partner_obs, partner_next_obs)``.
    3. Updates the cortex OaK Q-function from ``(partner_reward, partner_next_obs)``,
       crediting the partner's executed action when it is provided.
    4. Computes a greedy cortex action recommendation from ``partner_next_obs``.
    5. Returns the augmented observation ``[partner_obs, predictions]``.
    """

    def __init__(self, config: IAConfig) -> None:
        self._config = config
        self._cerebellum = ExoCerebellumAgent(config.cerebellum)
        self._cortex = ExoCortexAgent(config.cortex)

    @property
    def config(self) -> IAConfig:
        return self._config

    @property
    def cerebellum(self) -> ExoCerebellumAgent:
        return self._cerebellum

    @property
    def cortex(self) -> ExoCortexAgent:
        return self._cortex

    def to_config(self) -> dict[str, Any]:
        return self._config.to_config()

    def _require_state_contract(self, state: IAState) -> None:
        """Require the IA-owned layout before staging any child mutation."""

        if not isinstance(state, IAState):
            raise TypeError("state must be IAState")
        if state.step_count.shape != () or state.step_count.dtype != jnp.int32:
            raise ValueError("state.step_count must be scalar int32")
        if state.step_words.shape != (2,) or state.step_words.dtype != jnp.uint32:
            raise ValueError("state.step_words must have shape (2,) and dtype uint32")
        self._cerebellum._require_state_contract(state.cerebellum_state)
        if not isinstance(state.cortex_state, OaKState):
            raise TypeError("state.cortex_state must be OaKState")

    @staticmethod
    def _child_clocks_aligned(state: IAState) -> Bool[Array, ""]:
        """Apply the zero-cost-start primitive-history relation."""

        return (
            jnp.all(state.step_words == state.cerebellum_state.step_words)
            & jnp.all(state.step_words == state.cortex_state.step_words)
            & jnp.all(state.step_words == state.cortex_state.stomp_state.step_words)
        )

    def state_is_valid(self, state: IAState) -> Bool[Array, ""]:
        """Authenticate IA-owned values, both children, and their history."""

        self._require_state_contract(state)
        return (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & self._cerebellum.state_is_valid(state.cerebellum_state)
            & self._cortex.state_is_valid(state.cortex_state)
            & self._child_clocks_aligned(state)
        )

    def init(self, key: Array) -> IAState:
        """Initialise IA state."""
        cortex_state = self._cortex.init(key)
        return IAState(
            cerebellum_state=self._cerebellum.init(),
            cortex_state=cortex_state,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def start(self, state: IAState, initial_observation: Array) -> IAState:
        """Prime the IA agent without consuming a primitive-step identity."""

        self._require_state_contract(state)
        observation = jnp.asarray(initial_observation, dtype=jnp.float32)
        expected_shape = (self._config.cortex.observation_dim,)
        if observation.shape != expected_shape:
            raise ValueError(
                f"initial_observation must have shape {expected_shape}, got {observation.shape}"
            )
        source_state_valid = self.state_is_valid(state)
        input_valid = jnp.all(jnp.isfinite(observation))
        safe_observation = jnp.where(jnp.isfinite(observation), observation, 0.0)
        new_cortex = self._cortex.start(state.cortex_state, safe_observation)
        candidate = cast(IAState, state.replace(cortex_state=new_cortex))
        candidate_valid = self.state_is_valid(candidate)
        start_applied = source_state_valid & input_valid & candidate_valid
        return jax.tree_util.tree_map(
            lambda proposed, source: (
                jnp.where(start_applied, proposed, source) if isinstance(source, Array) else source
            ),
            candidate,
            state,
        )

    def update(
        self,
        state: IAState,
        partner_obs: Array,
        partner_reward: Array,
        partner_next_obs: Array,
        partner_action: Array | None = None,
        *,
        discount: Array | None = None,
        decision_observation: Array | None = None,
        execution_boundary: Array | bool = False,
    ) -> IAUpdateResult:
        """Process one IA step from partner experience.

        Args:
            state: Current IA state.
            partner_obs: Partner's current observation ``s_t``.
            partner_reward: Partner's received reward ``r_{t+1}``.
            partner_next_obs: Partner's next observation ``s_{t+1}``.
            partner_action: Primitive action the partner actually executed at
                ``s_t`` (the ``effective_action`` from
                :func:`update_recommendation_protocol`).  When provided, the
                exo-cortex Q-update credits this executed action; when
                ``None``, the cortex's own selected action is credited.
            discount: Optional effective continuation multiplier for the
                partner transition. ``None`` preserves the legacy OaK update
                semantics.

        Returns:
            :class:`IAUpdateResult` with augmented observation and diagnostics.
        """
        self._require_state_contract(state)
        obs = jnp.asarray(partner_obs, dtype=jnp.float32)
        next_obs = jnp.asarray(partner_next_obs, dtype=jnp.float32)
        reward = jnp.asarray(partner_reward, dtype=jnp.float32)
        expected_obs_shape = (self._config.cortex.observation_dim,)
        if obs.shape != expected_obs_shape:
            raise ValueError(f"partner_obs must have shape {expected_obs_shape}, got {obs.shape}")
        if next_obs.shape != expected_obs_shape:
            raise ValueError(
                f"partner_next_obs must have shape {expected_obs_shape}, got {next_obs.shape}"
            )
        if reward.shape != ():
            raise ValueError("partner_reward must be scalar")
        routed_decision_obs = (
            next_obs
            if decision_observation is None
            else jnp.asarray(decision_observation, dtype=jnp.float32)
        )
        if routed_decision_obs.shape != expected_obs_shape:
            raise ValueError(
                "decision_observation must have shape "
                f"{expected_obs_shape}, got {routed_decision_obs.shape}"
            )
        boundary = jnp.asarray(execution_boundary)
        if boundary.shape != () or boundary.dtype != jnp.bool_:
            raise TypeError("execution_boundary must be a scalar boolean")
        discount_valid = jnp.asarray(True, dtype=jnp.bool_)
        if discount is not None:
            supplied_discount = jnp.asarray(discount, dtype=jnp.float32)
            if supplied_discount.shape != ():
                raise ValueError("discount must be scalar")
            discount_valid = (
                jnp.isfinite(supplied_discount)
                & (supplied_discount >= 0.0)
                & (supplied_discount <= 1.0)
            )
        action_valid = jnp.asarray(True, dtype=jnp.bool_)
        if partner_action is not None:
            _, action_valid = _checked_partner_action(
                partner_action,
                n_primitive_actions=self._config.cortex.n_primitive_actions,
            )
        prepared_cortex_state, _, prepared_action_valid = self._cortex._prepare_update_source(
            state.cortex_state,
            partner_action,
            discount,
        )
        action_valid = action_valid & prepared_action_valid
        input_valid = (
            jnp.all(jnp.isfinite(obs))
            & jnp.all(jnp.isfinite(next_obs))
            & jnp.all(jnp.isfinite(routed_decision_obs))
            & jnp.isfinite(reward)
            & discount_valid
            & action_valid
        )
        child_clocks_aligned = self._child_clocks_aligned(state)
        source_state_valid = (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & self._cerebellum.state_is_valid(state.cerebellum_state)
            & self._cortex.state_is_valid(prepared_cortex_state)
            & child_clocks_aligned
        )
        proposed_step_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )

        # Cerebellum: predict from obs, update from (obs, next_obs)
        cerebellum_result = self._cerebellum.update_result(state.cerebellum_state, obs, next_obs)

        # Cortex: update Q from (reward, next_obs) crediting the partner's
        # executed action when known, get recommendation
        cortex_result, recommendation, td_error = self._cortex._update_result(
            state.cortex_state,
            reward,
            next_obs,
            partner_action=partner_action,
            discount=discount,
            decision_observation=decision_observation,
            execution_boundary=execution_boundary,
        )

        candidate_state = IAState(
            cerebellum_state=cerebellum_result.state,
            cortex_state=cortex_result.state,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_step_words,
        )
        proposed_state_valid = self.state_is_valid(candidate_state)
        update_applied = (
            source_state_valid
            & input_valid
            & child_clocks_aligned
            & lifetime_capacity_available
            & cerebellum_result.update_applied
            & cortex_result.update_applied
            & proposed_state_valid
        )
        new_state = jax.tree_util.tree_map(
            lambda candidate, source: (
                jnp.where(update_applied, candidate, source)
                if isinstance(source, Array)
                else source
            ),
            candidate_state,
            state,
        )
        predictions = jnp.where(
            update_applied,
            cerebellum_result.predictions,
            jnp.zeros_like(cerebellum_result.predictions),
        )
        errors = jnp.where(
            update_applied,
            cerebellum_result.errors,
            jnp.zeros_like(cerebellum_result.errors),
        )
        safe_obs = jnp.where(jnp.isfinite(obs), obs, jnp.float32(0.0))
        augmented_obs = jnp.concatenate([safe_obs, predictions])
        output_td_error = jnp.where(
            update_applied | ~action_valid,
            td_error,
            jnp.asarray(0.0, dtype=jnp.float32),
        )

        return IAUpdateResult(
            state=new_state,
            predictions=predictions,
            cerebellum_errors=errors,
            recommendation=recommendation,
            augmented_obs=augmented_obs,
            cortex_td_error=output_td_error,
            pre_step_words=state.step_words,
            post_step_words=new_state.step_words,
            source_state_valid=source_state_valid,
            input_valid=input_valid,
            child_clocks_aligned=child_clocks_aligned,
            lifetime_capacity_available=lifetime_capacity_available,
            cerebellum_update_applied=cerebellum_result.update_applied,
            cortex_update_applied=cortex_result.update_applied,
            proposed_state_valid=proposed_state_valid,
            update_applied=update_applied,
        )

    def scan(
        self,
        state: IAState,
        partner_obs: Array,
        partner_rewards: Array,
        partner_next_obs: Array,
        partner_actions: Array | None = None,
        *,
        discounts: Array | None = None,
        partner_decision_obs: Array | None = None,
        execution_boundaries: Array | None = None,
    ) -> IAArrayResult:
        """Run the IA agent over pre-collected partner transition arrays.

        Args:
            state: Starting IA state.
            partner_obs: Shape ``(T, obs_dim)`` partner observations.
            partner_rewards: Shape ``(T,)`` partner rewards.
            partner_next_obs: Shape ``(T, obs_dim)`` partner next observations.
            partner_actions: Optional shape ``(T,)`` int32 primitive actions
                the partner executed; when provided, each cortex Q-update
                credits the executed action (see :meth:`update`).
            discounts: Optional shape ``(T,)`` effective continuation
                multipliers. ``None`` preserves legacy update semantics.

        Returns:
            :class:`IAArrayResult` with per-step diagnostics.
        """
        use_actions = partner_actions is not None
        use_discounts = discounts is not None
        use_decision_observations = partner_decision_obs is not None

        def step_fn(
            carry: IAState,
            inputs: tuple[Array, ...],
        ) -> tuple[IAState, tuple[Array, ...]]:
            (
                obs,
                reward,
                next_ob,
                decision_ob,
                action,
                transition_discount,
                execution_boundary,
            ) = inputs
            result = self.update(
                carry,
                obs,
                reward,
                next_ob,
                partner_action=action if use_actions else None,
                discount=transition_discount if use_discounts else None,
                decision_observation=(decision_ob if use_decision_observations else None),
                execution_boundary=execution_boundary,
            )
            return result.state, (
                result.predictions,
                result.cerebellum_errors,
                result.recommendation,
                result.augmented_obs,
                result.cortex_td_error,
                result.pre_step_words,
                result.post_step_words,
                result.source_state_valid,
                result.input_valid,
                result.child_clocks_aligned,
                result.lifetime_capacity_available,
                result.cerebellum_update_applied,
                result.cortex_update_applied,
                result.proposed_state_valid,
                result.update_applied,
            )

        partner_rewards_shape = tuple(partner_rewards.shape)
        if len(partner_rewards_shape) != 1 or partner_rewards_shape[0] < 1:
            raise ValueError("partner_rewards must have shape (num_steps,)")
        num_steps = partner_rewards_shape[0]
        if partner_actions is not None:
            _, _ = checked_integer_action_array(
                partner_actions,
                self._config.cortex.n_primitive_actions,
                name="partner_actions",
                expected_shape=(num_steps,),
                range_message="partner_actions must lie in the primitive action range",
            )
            # Preserve the original integer dtype until each scalar reaches
            # ``_checked_partner_action``; narrowing here would erase wide
            # invalid values such as uint64(2**32).
            scan_actions = jnp.asarray(partner_actions)
        else:
            scan_actions = jnp.zeros((num_steps,), dtype=jnp.int32)
        scan_discounts = (
            jnp.asarray(discounts, dtype=jnp.float32)
            if discounts is not None
            else jnp.ones((num_steps,), dtype=jnp.float32)
        )
        scan_decision_observations = (
            jnp.asarray(partner_next_obs, dtype=jnp.float32)
            if partner_decision_obs is None
            else jnp.asarray(partner_decision_obs, dtype=jnp.float32)
        )
        scan_execution_boundaries = (
            jnp.zeros((num_steps,), dtype=jnp.bool_)
            if execution_boundaries is None
            else jnp.asarray(execution_boundaries, dtype=jnp.bool_)
        )
        xs: tuple[Array, ...] = (
            partner_obs,
            partner_rewards,
            partner_next_obs,
            scan_decision_observations,
            scan_actions,
            scan_discounts,
            scan_execution_boundaries,
        )

        (
            final_state,
            (
                predictions,
                cerebellum_errors,
                recommendations,
                augmented_obs,
                cortex_td_errors,
                pre_step_words,
                post_step_words,
                source_state_valid,
                input_valid,
                child_clocks_aligned,
                lifetime_capacity_available,
                cerebellum_updates_applied,
                cortex_updates_applied,
                proposed_state_valid,
                updates_applied,
            ),
        ) = jax.lax.scan(step_fn, state, xs)

        return IAArrayResult(
            state=final_state,
            predictions=predictions,
            cerebellum_errors=cerebellum_errors,
            recommendations=recommendations,
            augmented_obs=augmented_obs,
            cortex_td_errors=cortex_td_errors,
            pre_step_words=pre_step_words,
            post_step_words=post_step_words,
            source_state_valid=source_state_valid,
            input_valid=input_valid,
            child_clocks_aligned=child_clocks_aligned,
            lifetime_capacity_available=lifetime_capacity_available,
            cerebellum_updates_applied=cerebellum_updates_applied,
            cortex_updates_applied=cortex_updates_applied,
            proposed_state_valid=proposed_state_valid,
            updates_applied=updates_applied,
        )


# ---------------------------------------------------------------------------
# Exact-state migration and resource accounting
# ---------------------------------------------------------------------------


def _host_field_mapping(value: Any, *, name: str) -> dict[str, Any]:
    """Return an exact shallow field mapping for one legacy state."""

    if type(value) is dict:
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    raise TypeError(f"legacy {name} state must be a mapping or dataclass")


def _strict_legacy_counter(value: Any, *, name: str) -> int:
    """Read one uniquely recoverable pre-v2 non-negative int32 counter."""

    counter = jnp.asarray(value)
    if counter.shape != () or counter.dtype != jnp.dtype(jnp.int32):
        raise TypeError(f"legacy {name} must be scalar int32")
    result = int(counter)
    if result < 0:
        raise ValueError(f"negative legacy {name} indicates wrap")
    if result >= _INT32_MAX:
        raise ValueError(f"saturated legacy {name} is ambiguous")
    return result


def migrate_legacy_exo_cerebellum_state(
    legacy_state: Any,
    *,
    config: ExoCerebellumConfig,
) -> ExoCerebellumState:
    """Migrate only the exact pre-v2 cerebellum field manifest."""

    fields = _host_field_mapping(legacy_state, name="exo-cerebellum")
    if set(fields) != {"weights", "step_count"}:
        missing = sorted({"weights", "step_count"} - set(fields))
        extra = sorted(set(fields) - {"weights", "step_count"})
        raise ValueError(
            f"legacy exo-cerebellum field manifest is not exact; missing={missing}, extra={extra}"
        )
    step = _strict_legacy_counter(fields["step_count"], name="exo-cerebellum step_count")
    state = ExoCerebellumState(
        weights=jnp.asarray(fields["weights"]),
        step_count=jnp.asarray(fields["step_count"]),
        step_words=jnp.asarray((0, step), dtype=jnp.uint32),
    )
    agent = ExoCerebellumAgent(config)
    agent._require_state_contract(state)
    if not bool(jax.device_get(agent.state_is_valid(state))):
        raise ValueError("legacy exo-cerebellum state violates the v2 contract")
    return state


def migrate_legacy_ia_state(
    legacy_state: Any,
    *,
    config: IAConfig,
) -> IAState:
    """Migrate one mutually authenticated, unsaturated pre-v2 IA state."""

    fields = _host_field_mapping(legacy_state, name="IA")
    expected = {"cerebellum_state", "cortex_state", "step_count"}
    if set(fields) != expected:
        missing = sorted(expected - set(fields))
        extra = sorted(set(fields) - expected)
        raise ValueError(f"legacy IA field manifest is not exact; missing={missing}, extra={extra}")
    step = _strict_legacy_counter(fields["step_count"], name="IA step_count")
    cerebellum = migrate_legacy_exo_cerebellum_state(
        fields["cerebellum_state"],
        config=config.cerebellum,
    )
    cortex_raw = fields["cortex_state"]
    cortex = (
        cortex_raw if isinstance(cortex_raw, OaKState) else migrate_legacy_oak_state(cortex_raw)
    )
    state = IAState(
        cerebellum_state=cerebellum,
        cortex_state=cortex,
        step_count=jnp.asarray(fields["step_count"]),
        step_words=jnp.asarray((0, step), dtype=jnp.uint32),
    )
    agent = IAAgent(config)
    agent._require_state_contract(state)
    if not bool(jax.device_get(agent.state_is_valid(state))):
        raise ValueError("legacy IA child histories do not authenticate one lifetime")
    return state


def migrate_legacy_recommendation_protocol_state(
    legacy_state: Any,
) -> RecommendationProtocolState:
    """Migrate exact pre-v2 protocol counters before signed saturation."""

    fields = _host_field_mapping(legacy_state, name="recommendation protocol")
    expected = {"accepted_count", "rejected_count", "acceptance_ema", "step_count"}
    if set(fields) != expected:
        missing = sorted(expected - set(fields))
        extra = sorted(set(fields) - expected)
        raise ValueError(
            "legacy recommendation-protocol field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )
    accepted = _strict_legacy_counter(
        fields["accepted_count"],
        name="recommendation-protocol accepted_count",
    )
    rejected = _strict_legacy_counter(
        fields["rejected_count"],
        name="recommendation-protocol rejected_count",
    )
    step = _strict_legacy_counter(
        fields["step_count"],
        name="recommendation-protocol step_count",
    )
    if accepted + rejected != step:
        raise ValueError("legacy recommendation-protocol counters do not partition step_count")
    state = RecommendationProtocolState(
        accepted_count=jnp.asarray(fields["accepted_count"]),
        rejected_count=jnp.asarray(fields["rejected_count"]),
        acceptance_ema=jnp.asarray(fields["acceptance_ema"]),
        step_count=jnp.asarray(fields["step_count"]),
        accepted_words=jnp.asarray((0, accepted), dtype=jnp.uint32),
        rejected_words=jnp.asarray((0, rejected), dtype=jnp.uint32),
        step_words=jnp.asarray((0, step), dtype=jnp.uint32),
    )
    _recommendation_protocol_state_contract(state)
    if not bool(jax.device_get(recommendation_protocol_state_is_valid(state))):
        raise ValueError("legacy recommendation-protocol state violates the v2 contract")
    return state


def measure_exo_cerebellum_state_nbytes(state: ExoCerebellumState) -> int:
    """Measure all persistent JAX-array bytes in a cerebellum state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def measure_ia_state_nbytes(state: IAState) -> int:
    """Measure all persistent JAX-array bytes in the complete IA state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def measure_ia_wrapper_state_nbytes(state: IAState) -> int:
    """Measure IA-owned outer/cerebellum arrays, excluding nested OaK."""

    return measure_ia_state_nbytes(state) - measure_oak_state_nbytes(state.cortex_state)


def measure_recommendation_protocol_state_nbytes(
    state: RecommendationProtocolState,
) -> int:
    """Measure persistent protocol telemetry, EMA, and exact authorities."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def exo_cerebellum_lifetime_counter_nbytes() -> int:
    """Return cerebellum compatibility telemetry plus its exact authority."""

    return EXO_CEREBELLUM_LIFETIME_COUNTER_NBYTES


def ia_lifetime_counter_nbytes() -> int:
    """Return bytes for IA, cerebellum, and all nested OaK clocks."""

    return 2 * EXO_CEREBELLUM_LIFETIME_COUNTER_NBYTES + oak_total_lifetime_counter_nbytes()


def recommendation_protocol_lifetime_counter_nbytes() -> int:
    """Return telemetry and exact bytes for all three protocol counters."""

    return RECOMMENDATION_PROTOCOL_LIFETIME_COUNTER_NBYTES


__all__ = [
    "EXO_CEREBELLUM_LIFETIME_COUNTER_DELTA_NBYTES",
    "EXO_CEREBELLUM_LIFETIME_COUNTER_NBYTES",
    "EXO_CEREBELLUM_STATE_SCHEMA",
    "IA_LIFETIME_COUNTER_DELTA_NBYTES",
    "IA_LIFETIME_COUNTER_NBYTES",
    "IA_STATE_SCHEMA",
    "RECOMMENDATION_PROTOCOL_LIFETIME_COUNTER_DELTA_NBYTES",
    "RECOMMENDATION_PROTOCOL_LIFETIME_COUNTER_NBYTES",
    "RECOMMENDATION_PROTOCOL_STATE_SCHEMA",
    "ExoCerebellumAgent",
    "ExoCerebellumConfig",
    "ExoCerebellumState",
    "ExoCerebellumUpdateResult",
    "ExoCortexAgent",
    "ExoCortexConfig",
    "ExoCortexState",
    "IAAgent",
    "IAArrayResult",
    "IAConfig",
    "IAState",
    "IAUpdateResult",
    "RecommendationProtocolConfig",
    "RecommendationProtocolResult",
    "RecommendationProtocolState",
    "exo_cerebellum_lifetime_counter_nbytes",
    "ia_lifetime_counter_nbytes",
    "init_recommendation_protocol_state",
    "measure_exo_cerebellum_state_nbytes",
    "measure_ia_state_nbytes",
    "measure_ia_wrapper_state_nbytes",
    "measure_recommendation_protocol_state_nbytes",
    "migrate_legacy_exo_cerebellum_state",
    "migrate_legacy_ia_state",
    "migrate_legacy_recommendation_protocol_state",
    "recommendation_protocol_lifetime_counter_nbytes",
    "recommendation_protocol_state_is_valid",
    "update_recommendation_protocol",
]
