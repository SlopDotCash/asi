"""A tiny recurring two-agent world for continual-control experiments.

The world is deliberately small enough to support fast, exhaustive seeded
experiments while retaining the properties needed by a continual-control
testbed:

* operation is continuing -- transitions never terminate or reset;
* two agents act jointly in a bounded one-dimensional world;
* damped velocity dynamics remain stable under an infinite action stream;
* the visible context recurs ``meet -> avoid -> meet -> ...``;
* a scripted partner changes behaviour with that visible context, and can be
  replaced by any pure JAX-compatible policy hook;
* independent nuisance channels are replaced on every transition; and
* evaluator-only oracle data is returned separately from agent observations.

Transition convention
---------------------
At step ``t``, ``transition.observation`` and the reward objective use the
context visible in ``state``.  The action is then applied, ``step_count`` is
incremented, and ``transition.next_observation`` shows the next context.  Thus a
transition at a context boundary has ``oracle.context_switched == True`` while
its reward still matches the context that was visible when the action was
chosen.

The state and transition records are immutable chex dataclasses.  ``init``,
``observe``, ``step``, and ``step_with_partner`` are pure functions suitable
for ``jax.jit`` and ``jax.lax.scan``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, PRNGKeyArray

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)

MEET_CONTEXT = 0
AVOID_CONTEXT = 1
N_AGENTS = 2

# The first six channels have stable meanings.  Nuisance channels, if any,
# follow them.
OWN_POSITION_INDEX = 0
RELATIVE_POSITION_INDEX = 1
OWN_VELOCITY_INDEX = 2
OTHER_VELOCITY_INDEX = 3
MEET_CONTEXT_INDEX = 4
AVOID_CONTEXT_INDEX = 5
BASE_OBSERVATION_DIM = 6

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1

RECURRING_TWO_AGENT_CONFIG_SCHEMA = "alberta.recurring-two-agent-config.v2"
RECURRING_TWO_AGENT_STATE_SCHEMA = "alberta.recurring-two-agent-state.v2"
RECURRING_TWO_AGENT_CHECKPOINT_SCHEMA = "alberta.recurring-two-agent-checkpoint.v2"
_LEGACY_RECURRING_TWO_AGENT_CHECKPOINT_SCHEMA = "alberta.recurring-two-agent-checkpoint.v1"
RECURRING_TWO_AGENT_CLOCK_NBYTES = 12
RECURRING_TWO_AGENT_CLOCK_DELTA_NBYTES = 8
_RECURRING_TWO_AGENT_CONFIG_FIELDS = frozenset(
    {
        "type",
        "config_schema",
        "state_schema",
        "context_length",
        "nuisance_dim",
        "nuisance_scale",
        "world_limit",
        "damping",
        "acceleration",
        "time_delta",
        "max_speed",
        "initial_positions",
        "partner_policy",
    }
)

type PartnerPolicy = Callable[[Array, Array], Array]


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int = 0,
    maximum: int = _INT32_MAX,
) -> int:
    # This record is a JSON/provenance boundary.  Require the exact built-in
    # identity before comparing the value so hostile ``int`` subclasses cannot
    # execute comparison hooks and NumPy scalars cannot leak into ``to_dict``.
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return value


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")
    return array


def _checked_words_increment(words: Array) -> tuple[Array, Array]:
    _require_array(
        words,
        name="recurring two-agent step_words",
        shape=(2,),
        dtype=jnp.dtype(jnp.uint32),
    )
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry_word = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    candidate = jnp.stack((words[0] + carry_word, low)).astype(jnp.uint32)
    return jnp.where(available, candidate, words), available


def _words_to_int32_telemetry(words: Array) -> Array:
    saturated = (words[0] > jnp.asarray(0, dtype=jnp.uint32)) | (
        words[1] >= jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return jnp.where(
        saturated,
        jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        words[1].astype(jnp.int32),
    )


def _divmod_words_by_u32(words: Array, divisor: int | Array) -> tuple[Array, Array]:
    """Exact 64-by-32 long division without enabling JAX x64."""
    divisor_array = jnp.asarray(divisor, dtype=jnp.uint32)

    def body(index: Array, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        remainder, quotient_high, quotient_low = carry
        in_high = index < 32
        bit_index = jnp.asarray(31, dtype=jnp.int32) - jnp.mod(index, 32)
        source = jnp.where(in_high, words[0], words[1])
        bit = jnp.bitwise_and(
            jnp.right_shift(source, bit_index.astype(jnp.uint32)),
            jnp.asarray(1, dtype=jnp.uint32),
        )
        doubled = remainder + remainder + bit
        subtract = doubled >= divisor_array
        remainder = jnp.where(subtract, doubled - divisor_array, doubled)
        mask = jnp.left_shift(jnp.asarray(1, dtype=jnp.uint32), bit_index.astype(jnp.uint32))
        quotient_high = jnp.where(
            in_high & subtract,
            jnp.bitwise_or(quotient_high, mask),
            quotient_high,
        )
        quotient_low = jnp.where(
            (~in_high) & subtract,
            jnp.bitwise_or(quotient_low, mask),
            quotient_low,
        )
        return remainder, quotient_high, quotient_low

    zero = jnp.asarray(0, dtype=jnp.uint32)
    remainder, high, low = jax.lax.fori_loop(0, 64, body, (zero, zero, zero))
    return jnp.stack((high, low)).astype(jnp.uint32), remainder


@chex.dataclass(frozen=True)
class RecurringTwoAgentState:
    """Immutable dynamic state of :class:`RecurringTwoAgentWorld`.

    ``nuisance`` is stored in the state so repeated calls to :meth:`observe`
    are referentially transparent.  It affects neither physics nor reward.
    """

    key: PRNGKeyArray
    positions: Float[Array, " 2"]
    velocities: Float[Array, " 2"]
    nuisance: Float[Array, "2 nuisance_dim"]
    step_count: Int[Array, ""]
    step_words: Array | None = None

    def __post_init__(self) -> None:
        """Migrate omitted unsaturated compatibility clocks at construction."""
        if self.step_words is None:
            if self.step_count is None:
                # JAX tree transformations construct a transient all-None
                # placeholder before unflattening real leaves.
                return
            count_array = jnp.asarray(self.step_count)
            if count_array.shape != () or count_array.dtype != jnp.dtype(jnp.int32):
                raise TypeError("legacy recurring two-agent step_count must be scalar int32")
            count = int(count_array)
            if count < 0 or count >= _INT32_MAX:
                raise ValueError("legacy recurring two-agent step_count is ambiguous")
            object.__setattr__(
                self,
                "step_words",
                jnp.asarray((0, count), dtype=jnp.uint32),
            )


@chex.dataclass(frozen=True)
class RecurringTwoAgentOracle:
    """Evaluator-only diagnostics for one transition.

    These fields are intentionally separate from ``observation`` and
    ``next_observation``.  In particular, agents are not given the absolute
    step, segment/cycle indices, imminent-boundary flag, counterfactual reward,
    or unclipped requested action.
    """

    step_count: Int[Array, ""]
    step_words: Array
    segment_index: Int[Array, ""]
    segment_words: Array
    cycle_index: Int[Array, ""]
    cycle_words: Array
    context_id: Int[Array, ""]
    next_context_id: Int[Array, ""]
    context_switched: Bool[Array, ""]
    positions_before: Float[Array, " 2"]
    positions_after: Float[Array, " 2"]
    velocities_before: Float[Array, " 2"]
    velocities_after: Float[Array, " 2"]
    distance_before: Float[Array, ""]
    distance_after: Float[Array, ""]
    meet_reward: Float[Array, ""]
    avoid_reward: Float[Array, ""]
    requested_actions: Float[Array, " 2"]
    applied_actions: Float[Array, " 2"]
    hit_boundary: Bool[Array, " 2"]


@chex.dataclass(frozen=True)
class RecurringTwoAgentTransition:
    """One continuing joint-action transition."""

    observation: Float[Array, "2 observation_dim"]
    action: Float[Array, " 2"]
    reward: Float[Array, " 2"]
    next_observation: Float[Array, "2 observation_dim"]
    terminated: Bool[Array, ""]
    discount: Float[Array, ""]
    oracle: RecurringTwoAgentOracle


@chex.dataclass(frozen=True)
class RecurringTwoAgentStepResult:
    """One transactional environment proposal and its commit diagnostics."""

    transition: RecurringTwoAgentTransition
    state: RecurringTwoAgentState
    pre_step_words: Array
    post_step_words: Array
    state_valid: Array
    candidate_state_valid: Array
    lifetime_capacity_available: Array
    input_valid: Array
    update_applied: Array
    update_rejected: Array


@dataclass(frozen=True)
class RecurringTwoAgentResourceBudget:
    """Exact fixed-shape environment-state accounting."""

    state_nbytes: int
    exact_clock_nbytes: int
    exact_clock_delta_nbytes: int
    trainable_scalars: int = 0
    replay_capacity: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_nbytes", _require_int("state_nbytes", self.state_nbytes))
        object.__setattr__(
            self,
            "exact_clock_nbytes",
            _require_int("exact_clock_nbytes", self.exact_clock_nbytes),
        )
        object.__setattr__(
            self,
            "exact_clock_delta_nbytes",
            _require_int("exact_clock_delta_nbytes", self.exact_clock_delta_nbytes),
        )
        object.__setattr__(
            self,
            "trainable_scalars",
            _require_int("trainable_scalars", self.trainable_scalars),
        )
        object.__setattr__(
            self,
            "replay_capacity",
            _require_int("replay_capacity", self.replay_capacity),
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "type": type(self).__name__,
            "state_nbytes": self.state_nbytes,
            "exact_clock_nbytes": self.exact_clock_nbytes,
            "exact_clock_delta_nbytes": self.exact_clock_delta_nbytes,
            "trainable_scalars": self.trainable_scalars,
            "replay_capacity": self.replay_capacity,
        }


def scripted_meet_avoid_partner_policy(
    partner_observation: Array,
    key: Array,
) -> Array:
    """Move the partner toward agent 0 in ``meet`` and away in ``avoid``.

    The hook consumes only the partner's ordinary observation and a PRNG key;
    it has no access to evaluator-only oracle data.  The key is accepted to
    match stochastic custom-policy hooks but is unused by this deterministic
    script.
    """

    del key
    relative_position = partner_observation[RELATIVE_POSITION_INDEX]
    objective_direction = (
        partner_observation[MEET_CONTEXT_INDEX] - partner_observation[AVOID_CONTEXT_INDEX]
    )
    return jnp.sign(relative_position) * objective_direction


class RecurringTwoAgentWorld:
    """Bounded recurring-context continual-control environment.

    Each agent observes, in order:

    ``[own_position, relative_position, own_velocity, other_velocity,
    meet_visible, avoid_visible, nuisance...]``.

    Positions are divided by ``world_limit``, relative position by
    ``2 * world_limit``, and velocities by ``max_speed``.  The context
    channels are a one-hot visible cue.  No segment, cycle, boundary, or
    counterfactual-reward oracle is present in the agent input.

    Context A (id ``MEET_CONTEXT``) rewards proximity:

    ``meet_reward = 1 - distance / (2 * world_limit)``.

    Context B (id ``AVOID_CONTEXT``) rewards separation:

    ``avoid_reward = distance / (2 * world_limit)``.

    Both agents receive the same cooperative reward in ``[0, 1]``.

    Args:
        context_length: Number of transitions per A or B segment.
        nuisance_dim: Ephemeral nuisance features per agent.
        nuisance_scale: Standard deviation of nuisance features.
        world_limit: Symmetric position bound; positions lie in
            ``[-world_limit, world_limit]``.
        damping: Velocity retention in ``[0, 1)``.
        acceleration: Action-to-velocity gain.
        time_delta: Position integration interval.
        max_speed: Absolute velocity bound.
        initial_positions: Fixed two-agent initial positions.
        partner_policy: Optional pure hook ``policy(observation, key) -> scalar``.
            The default is :func:`scripted_meet_avoid_partner_policy`.
    """

    def __init__(
        self,
        *,
        context_length: int = 64,
        nuisance_dim: int = 4,
        nuisance_scale: float = 1.0,
        world_limit: float = 1.0,
        damping: float = 0.75,
        acceleration: float = 0.15,
        time_delta: float = 1.0,
        max_speed: float = 0.25,
        initial_positions: tuple[float, float] = (-0.5, 0.5),
        partner_policy: PartnerPolicy | None = None,
    ) -> None:
        if (
            type(context_length) is bool
            or type(context_length) is not int
            or context_length <= 0
        ):
            raise ValueError("context_length must be positive")
        if context_length > _INT32_MAX // 2:
            raise ValueError("2 * context_length must fit in signed schedule telemetry")
        if type(nuisance_dim) is bool or type(nuisance_dim) is not int or nuisance_dim < 0:
            raise ValueError("nuisance_dim must be non-negative")
        for name, value in (
            ("nuisance_scale", nuisance_scale),
            ("world_limit", world_limit),
            ("damping", damping),
            ("acceleration", acceleration),
            ("time_delta", time_delta),
            ("max_speed", max_speed),
        ):
            if type(value) is bool or (
                type(value) is not int and type(value) is not float
            ):
                raise ValueError(f"{name} must be a finite real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if nuisance_scale < 0.0:
            raise ValueError("nuisance_scale must be non-negative")
        if world_limit <= 0.0:
            raise ValueError("world_limit must be positive")
        if not 0.0 <= damping < 1.0:
            raise ValueError("damping must lie in [0, 1)")
        if acceleration <= 0.0:
            raise ValueError("acceleration must be positive")
        if time_delta <= 0.0:
            raise ValueError("time_delta must be positive")
        if max_speed <= 0.0:
            raise ValueError("max_speed must be positive")
        if type(initial_positions) is not tuple or len(initial_positions) != N_AGENTS:
            raise ValueError(f"initial_positions must be an exact {N_AGENTS}-tuple")
        canonical_positions = tuple(
            validated_float32_scalar(f"initial_positions[{index}]", position)
            for index, position in enumerate(initial_positions)
        )
        if any(abs(position) > world_limit for position in canonical_positions):
            raise ValueError("initial_positions must lie within [-world_limit, world_limit]")

        self._context_length = int(context_length)
        self._nuisance_dim = int(nuisance_dim)
        self._nuisance_scale = float(nuisance_scale)
        self._world_limit = float(world_limit)
        self._damping = float(damping)
        self._acceleration = float(acceleration)
        self._time_delta = float(time_delta)
        self._max_speed = float(max_speed)
        self._initial_positions_tuple = canonical_positions
        self._initial_positions = jnp.asarray(canonical_positions, dtype=jnp.float32)
        self._partner_policy_is_default = partner_policy is None
        self._partner_policy = (
            scripted_meet_avoid_partner_policy if partner_policy is None else partner_policy
        )

    @property
    def n_agents(self) -> int:
        """Number of jointly acting agents (always two)."""

        return N_AGENTS

    @property
    def action_dim(self) -> int:
        """Scalar action dimensions per agent."""

        return 1

    @property
    def observation_dim(self) -> int:
        """Number of observation channels per agent."""

        return BASE_OBSERVATION_DIM + self._nuisance_dim

    @property
    def feature_dim(self) -> int:
        """Alias for the per-agent observation dimension."""

        return self.observation_dim

    @property
    def nuisance_dim(self) -> int:
        """Number of ephemeral channels per agent."""

        return self._nuisance_dim

    @property
    def context_length(self) -> int:
        """Number of transitions in each recurring context segment."""

        return self._context_length

    @property
    def world_limit(self) -> float:
        """Absolute position bound."""

        return self._world_limit

    @property
    def max_speed(self) -> float:
        """Absolute velocity bound."""

        return self._max_speed

    @property
    def resource_budget(self) -> RecurringTwoAgentResourceBudget:
        """Return exact persistent-state and lifetime-clock accounting."""
        state = self.init(jr.key(0))
        return RecurringTwoAgentResourceBudget(
            state_nbytes=measure_recurring_two_agent_state_nbytes(state),
            exact_clock_nbytes=RECURRING_TWO_AGENT_CLOCK_NBYTES,
            exact_clock_delta_nbytes=RECURRING_TWO_AGENT_CLOCK_DELTA_NBYTES,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the default-policy world under strict v2 schemas."""
        if not self._partner_policy_is_default:
            raise ValueError("custom partner policies are not serializable")
        return {
            "type": type(self).__name__,
            "config_schema": RECURRING_TWO_AGENT_CONFIG_SCHEMA,
            "state_schema": RECURRING_TWO_AGENT_STATE_SCHEMA,
            "context_length": self._context_length,
            "nuisance_dim": self._nuisance_dim,
            "nuisance_scale": self._nuisance_scale,
            "world_limit": self._world_limit,
            "damping": self._damping,
            "acceleration": self._acceleration,
            "time_delta": self._time_delta,
            "max_speed": self._max_speed,
            "initial_positions": list(self._initial_positions_tuple),
            "partner_policy": "scripted_meet_avoid_partner_policy",
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> RecurringTwoAgentWorld:
        """Strictly reconstruct a default-policy v2 environment."""
        if type(config) is not dict:
            raise ValueError("recurring two-agent config must be an exact dictionary")
        if any(type(key) is not str for key in config):
            raise ValueError("recurring two-agent config keys must be exact strings")
        values = dict(config)
        if set(values) != _RECURRING_TWO_AGENT_CONFIG_FIELDS:
            raise ValueError("recurring two-agent config fields are invalid")
        type_name = values.pop("type")
        if type(type_name) is not str or type_name != cls.__name__:
            raise ValueError("recurring two-agent config type is unsupported")
        config_schema = values.pop("config_schema")
        if type(config_schema) is not str or config_schema != RECURRING_TWO_AGENT_CONFIG_SCHEMA:
            raise ValueError("recurring two-agent config schema is unsupported")
        state_schema = values.pop("state_schema")
        if type(state_schema) is not str or state_schema != RECURRING_TWO_AGENT_STATE_SCHEMA:
            raise ValueError("recurring two-agent state schema is unsupported")
        partner_policy = values.pop("partner_policy")
        if (
            type(partner_policy) is not str
            or partner_policy != "scripted_meet_avoid_partner_policy"
        ):
            raise ValueError("recurring two-agent partner policy is unsupported")
        initial = values.pop("initial_positions")
        if type(initial) is not list:
            raise ValueError("recurring two-agent initial_positions must be an exact list")
        if len(initial) != N_AGENTS:
            raise ValueError(f"recurring two-agent initial_positions must have length {N_AGENTS}")
        return cls(initial_positions=tuple(initial), **values)

    def _require_state_contract(self, state: RecurringTwoAgentState) -> None:
        """Require every fixed-shape v2 state leaf."""
        _require_array(state.positions, name="positions", shape=(2,), dtype=jnp.dtype(jnp.float32))
        _require_array(
            state.velocities,
            name="velocities",
            shape=(2,),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.nuisance,
            name="nuisance",
            shape=(2, self._nuisance_dim),
            dtype=jnp.dtype(jnp.float32),
        )
        _require_array(
            state.step_count,
            name="step_count",
            shape=(),
            dtype=jnp.dtype(jnp.int32),
        )
        _require_array(
            cast(Array, state.step_words),
            name="step_words",
            shape=(2,),
            dtype=jnp.dtype(jnp.uint32),
        )
        key_data = jr.key_data(state.key)
        if key_data.shape != (2,) or key_data.dtype != jnp.dtype(jnp.uint32):
            raise TypeError("recurring two-agent key must be a scalar JAX PRNG key")

    def state_is_valid(self, state: RecurringTwoAgentState) -> Array:
        """Authenticate exact time and bounded finite physical state."""
        self._require_state_contract(state)
        words = cast(Array, state.step_words)
        return (
            (state.step_count == _words_to_int32_telemetry(words))
            & jnp.all(jnp.isfinite(state.positions))
            & jnp.all(jnp.isfinite(state.velocities))
            & jnp.all(jnp.isfinite(state.nuisance))
            & jnp.all(jnp.abs(state.positions) <= self._world_limit)
            & jnp.all(jnp.abs(state.velocities) <= self._max_speed)
        )

    def context_id(self, state: RecurringTwoAgentState) -> Array:
        """Return the visible recurring context id for ``state``."""
        self._require_state_contract(state)
        segment_words, _remainder = _divmod_words_by_u32(
            cast(Array, state.step_words), self._context_length
        )
        return jnp.bitwise_and(segment_words[1], jnp.asarray(1, dtype=jnp.uint32)).astype(jnp.int32)

    def _schedule_identities(self, words: Array) -> tuple[Array, Array, Array, Array]:
        """Return exact segment/cycle identities and bounded schedule phase."""
        segment_words, segment_step = _divmod_words_by_u32(words, self._context_length)
        cycle_words, _parity = _divmod_words_by_u32(segment_words, 2)
        context = jnp.bitwise_and(segment_words[1], jnp.asarray(1, dtype=jnp.uint32)).astype(
            jnp.int32
        )
        return segment_words, cycle_words, segment_step.astype(jnp.int32), context

    def init(self, key: Array) -> RecurringTwoAgentState:
        """Create a deterministic initial state from ``key``."""

        next_key, nuisance_key = jr.split(key)
        nuisance = self._sample_nuisance(nuisance_key)
        return RecurringTwoAgentState(  # type: ignore[call-arg]
            key=next_key,
            positions=self._initial_positions,
            velocities=jnp.zeros((N_AGENTS,), dtype=jnp.float32),
            nuisance=nuisance,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def observe(self, state: RecurringTwoAgentState) -> Array:
        """Construct the ordinary, non-oracle observation for both agents."""

        other = jnp.array([1, 0], dtype=jnp.int32)
        own_positions = state.positions / self._world_limit
        relative_positions = (state.positions[other] - state.positions) / (2.0 * self._world_limit)
        own_velocities = state.velocities / self._max_speed
        other_velocities = state.velocities[other] / self._max_speed

        context = self.context_id(state)
        context_features = jnp.stack(
            (
                (context == MEET_CONTEXT).astype(jnp.float32),
                (context == AVOID_CONTEXT).astype(jnp.float32),
            )
        )
        shared_context = jnp.broadcast_to(context_features, (N_AGENTS, 2))
        physical = jnp.stack(
            (
                own_positions,
                relative_positions,
                own_velocities,
                other_velocities,
            ),
            axis=1,
        )
        return jnp.concatenate((physical, shared_context, state.nuisance), axis=1)

    def step(
        self,
        state: RecurringTwoAgentState,
        joint_action: Array,
    ) -> tuple[RecurringTwoAgentTransition, RecurringTwoAgentState]:
        """Apply an externally supplied joint continuous action.

        ``joint_action`` has shape ``(2,)`` and is clipped componentwise to
        ``[-1, 1]``.  The input state is never mutated.
        """

        result = self.step_result(state, joint_action)
        return result.transition, result.state

    def step_result(
        self,
        state: RecurringTwoAgentState,
        joint_action: Array,
    ) -> RecurringTwoAgentStepResult:
        """Stage one joint action and expose whether the whole event committed."""
        next_key, nuisance_key = jr.split(state.key)
        return self._advance_result(state, joint_action, next_key, nuisance_key)

    def step_with_partner(
        self,
        state: RecurringTwoAgentState,
        learner_action: Array,
    ) -> tuple[RecurringTwoAgentTransition, RecurringTwoAgentState]:
        """Apply agent 0's action and obtain agent 1's action from the hook.

        The policy hook sees only agent 1's ordinary observation and an
        independent PRNG key.  A scalar or one-element learner action is
        accepted for convenience.
        """

        result = self.step_with_partner_result(state, learner_action)
        return result.transition, result.state

    def step_with_partner_result(
        self,
        state: RecurringTwoAgentState,
        learner_action: Array,
    ) -> RecurringTwoAgentStepResult:
        """Stage learner and partner actions as one atomic environment event."""
        next_key, nuisance_key, policy_key = jr.split(state.key, 3)
        observation = self.observe(state)
        partner_action = jnp.asarray(
            self._partner_policy(observation[1], policy_key),
            dtype=jnp.float32,
        )
        learner_action_array = jnp.asarray(learner_action, dtype=jnp.float32)
        if partner_action.size != 1:
            raise ValueError("partner policy must return a scalar or one-element action")
        if learner_action_array.size != 1:
            raise ValueError("learner_action must be scalar or one-element")
        joint_action = jnp.stack(
            (
                jnp.reshape(learner_action_array, ()),
                jnp.reshape(partner_action, ()),
            )
        )
        return self._advance_result(state, joint_action, next_key, nuisance_key)

    def _sample_nuisance(self, key: Array) -> Array:
        return self._nuisance_scale * jr.normal(
            key,
            (N_AGENTS, self._nuisance_dim),
            dtype=jnp.float32,
        )

    def _advance_result(
        self,
        state: RecurringTwoAgentState,
        joint_action: Array,
        next_key: Array,
        nuisance_key: Array,
    ) -> RecurringTwoAgentStepResult:
        self._require_state_contract(state)
        requested_actions = jnp.asarray(joint_action)
        if requested_actions.shape != (N_AGENTS,):
            raise ValueError(
                f"joint_action must have shape ({N_AGENTS},), got {requested_actions.shape}"
            )
        if requested_actions.dtype != jnp.dtype(jnp.float32):
            raise TypeError("joint_action must have dtype float32")
        state_valid = self.state_is_valid(state)
        proposed_words, lifetime_capacity_available = _checked_words_increment(
            cast(Array, state.step_words)
        )
        input_valid = jnp.all(jnp.isfinite(requested_actions))
        safe_actions = jnp.where(jnp.isfinite(requested_actions), requested_actions, 0.0)
        applied_actions = jnp.clip(safe_actions, -1.0, 1.0)

        accelerated = self._damping * state.velocities + self._acceleration * applied_actions
        candidate_velocities = jnp.clip(
            accelerated,
            -self._max_speed,
            self._max_speed,
        )
        candidate_positions = state.positions + self._time_delta * candidate_velocities
        positions = jnp.clip(
            candidate_positions,
            -self._world_limit,
            self._world_limit,
        )
        hit_boundary = candidate_positions != positions
        velocities = jnp.where(hit_boundary, 0.0, candidate_velocities)

        distance_before = jnp.abs(state.positions[0] - state.positions[1])
        distance_after = jnp.abs(positions[0] - positions[1])
        normalized_distance = jnp.clip(
            distance_after / (2.0 * self._world_limit),
            0.0,
            1.0,
        )
        meet_reward = 1.0 - normalized_distance
        avoid_reward = normalized_distance

        context = self.context_id(state)
        reward_scalar = jnp.where(
            context == MEET_CONTEXT,
            meet_reward,
            avoid_reward,
        ).astype(jnp.float32)
        reward = jnp.full((N_AGENTS,), reward_scalar, dtype=jnp.float32)

        candidate_state = RecurringTwoAgentState(  # type: ignore[call-arg]
            key=next_key,
            positions=positions,
            velocities=velocities,
            nuisance=self._sample_nuisance(nuisance_key),
            step_count=_words_to_int32_telemetry(proposed_words),
            step_words=proposed_words,
        )
        candidate_state_valid = self.state_is_valid(candidate_state)
        update_applied = (
            state_valid & input_valid & lifetime_capacity_available & candidate_state_valid
        )
        new_state = cast(
            RecurringTwoAgentState,
            jax.tree.map(
                lambda proposed, current: jnp.where(update_applied, proposed, current),
                candidate_state,
                state,
            ),
        )
        next_context = self.context_id(candidate_state)
        segment_words, cycle_words, _segment_step, _context = self._schedule_identities(
            cast(Array, state.step_words)
        )
        oracle = RecurringTwoAgentOracle(  # type: ignore[call-arg]
            step_count=state.step_count,
            step_words=cast(Array, state.step_words),
            segment_index=_words_to_int32_telemetry(segment_words),
            segment_words=segment_words,
            cycle_index=_words_to_int32_telemetry(cycle_words),
            cycle_words=cycle_words,
            context_id=context,
            next_context_id=next_context,
            context_switched=context != next_context,
            positions_before=state.positions,
            positions_after=positions,
            velocities_before=state.velocities,
            velocities_after=velocities,
            distance_before=distance_before,
            distance_after=distance_after,
            meet_reward=meet_reward,
            avoid_reward=avoid_reward,
            requested_actions=safe_actions,
            applied_actions=applied_actions,
            hit_boundary=hit_boundary,
        )
        transition = RecurringTwoAgentTransition(  # type: ignore[call-arg]
            observation=self.observe(state),
            action=applied_actions,
            reward=reward,
            next_observation=self.observe(candidate_state),
            terminated=~update_applied,
            discount=jnp.where(
                update_applied,
                jnp.asarray(1.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            oracle=oracle,
        )
        transition = cast(
            RecurringTwoAgentTransition,
            jax.tree.map(
                lambda value: jnp.where(
                    update_applied,
                    value,
                    (
                        jnp.full_like(value, jnp.nan)
                        if jnp.issubdtype(value.dtype, jnp.floating)
                        else (
                            jnp.zeros_like(value)
                            if jnp.issubdtype(value.dtype, jnp.bool_)
                            else jnp.full_like(value, -1)
                        )
                    ),
                ),
                transition,
            ),
        ).replace(  # type: ignore[attr-defined]
            terminated=~update_applied,
            discount=jnp.where(update_applied, 1.0, 0.0).astype(jnp.float32),
        )
        return RecurringTwoAgentStepResult(  # type: ignore[call-arg]
            transition=transition,
            state=new_state,
            pre_step_words=cast(Array, state.step_words),
            post_step_words=cast(Array, new_state.step_words),
            state_valid=state_valid,
            candidate_state_valid=candidate_state_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            input_valid=input_valid,
            update_applied=update_applied,
            update_rejected=~update_applied,
        )


def measure_recurring_two_agent_state_nbytes(state: RecurringTwoAgentState) -> int:
    """Measure every persistent JAX-array byte in one environment state."""
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def _recurring_checkpoint_storage_state(
    state: RecurringTwoAgentState,
) -> RecurringTwoAgentState:
    """Encode empty nuisance arrays as Orbax-compatible storage sentinels."""
    return cast(
        RecurringTwoAgentState,
        jax.tree.map(
            lambda leaf: (
                jnp.zeros((1,), dtype=leaf.dtype)
                if isinstance(leaf, Array) and leaf.size == 0
                else leaf
            ),
            state,
        ),
    )


def _recurring_empty_leaf_indices(state: RecurringTwoAgentState) -> list[int]:
    return [
        index
        for index, leaf in enumerate(jax.tree.leaves(state))
        if isinstance(leaf, Array) and leaf.size == 0
    ]


def _restore_recurring_empty_arrays(
    restored: RecurringTwoAgentState,
    template: RecurringTwoAgentState,
) -> RecurringTwoAgentState:
    return cast(
        RecurringTwoAgentState,
        jax.tree.map(
            lambda stored, expected: (
                expected if isinstance(expected, Array) and expected.size == 0 else stored
            ),
            restored,
            template,
        ),
    )


def migrate_legacy_recurring_two_agent_state(
    legacy_state: Any,
    *,
    world: RecurringTwoAgentWorld,
) -> RecurringTwoAgentState:
    """Migrate only an unsaturated pre-v2 environment lifetime."""
    if isinstance(legacy_state, Mapping):
        fields = dict(legacy_state)
    elif dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        fields = {
            field.name: getattr(legacy_state, field.name)
            for field in dataclasses.fields(legacy_state)
        }
    else:
        raise TypeError("legacy recurring two-agent state must be a mapping or dataclass")
    expected = {"key", "positions", "velocities", "nuisance", "step_count"}
    if set(fields) != expected:
        raise ValueError("legacy recurring two-agent state fields are invalid")
    count_array = _require_array(
        fields["step_count"],
        name="legacy recurring two-agent step_count",
        shape=(),
        dtype=jnp.dtype(jnp.int32),
    )
    count = int(count_array)
    if count < 0:
        raise ValueError("negative legacy recurring two-agent step_count indicates wrap")
    if count >= _INT32_MAX:
        raise ValueError("saturated legacy recurring two-agent step_count is ambiguous")
    fields["step_words"] = jnp.asarray((0, count), dtype=jnp.uint32)
    migrated = RecurringTwoAgentState(**fields)
    world._require_state_contract(migrated)
    if not bool(jax.device_get(world.state_is_valid(migrated))):
        raise ValueError("legacy recurring two-agent state violates the v2 contract")
    return migrated


def save_recurring_two_agent_checkpoint(
    world: RecurringTwoAgentWorld,
    state: RecurringTwoAgentState,
    path: str | Path,
) -> None:
    """Persist one valid exact-clock default-policy environment state."""
    world._require_state_contract(state)
    if not bool(jax.device_get(world.state_is_valid(state))):
        raise ValueError("recurring two-agent checkpoint state is invalid")
    save_checkpoint(
        _recurring_checkpoint_storage_state(state),
        path,
        metadata={
            "schema": RECURRING_TWO_AGENT_CHECKPOINT_SCHEMA,
            "world_config": world.to_config(),
            "memory_accounting": world.resource_budget.to_dict(),
            "zero_sized_array_leaf_indices": _recurring_empty_leaf_indices(state),
        },
    )


def load_recurring_two_agent_checkpoint(
    path: str | Path,
) -> tuple[RecurringTwoAgentWorld, RecurringTwoAgentState]:
    """Restore only a strict v2 exact-clock environment checkpoint."""
    metadata = load_checkpoint_metadata(path)
    expected = {
        "schema",
        "world_config",
        "memory_accounting",
        "zero_sized_array_leaf_indices",
    }
    if set(metadata) != expected:
        raise ValueError("recurring two-agent checkpoint metadata fields are invalid")
    schema = metadata.get("schema")
    if schema == _LEGACY_RECURRING_TWO_AGENT_CHECKPOINT_SCHEMA:
        raise ValueError(
            "legacy recurring two-agent checkpoint lacks exact step_words; "
            "migrate its state and resave it"
        )
    if schema != RECURRING_TWO_AGENT_CHECKPOINT_SCHEMA:
        raise ValueError("recurring two-agent checkpoint schema is unsupported")
    raw_config = metadata.get("world_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("recurring two-agent checkpoint world_config is invalid")
    world = RecurringTwoAgentWorld.from_config(raw_config)
    template = world.init(jr.key(0))
    expected_empty = _recurring_empty_leaf_indices(template)
    if metadata.get("zero_sized_array_leaf_indices") != expected_empty:
        raise ValueError("recurring two-agent empty-array manifest does not match")
    restored, restored_metadata = load_checkpoint(
        _recurring_checkpoint_storage_state(template), path
    )
    if restored_metadata != metadata:
        raise ValueError("recurring two-agent checkpoint metadata changed between reads")
    state = _restore_recurring_empty_arrays(cast(RecurringTwoAgentState, restored), template)
    world._require_state_contract(state)
    if not bool(jax.device_get(world.state_is_valid(state))):
        raise ValueError("restored recurring two-agent state is invalid")
    if world.resource_budget.to_dict() != metadata.get("memory_accounting"):
        raise ValueError("recurring two-agent checkpoint resource contract does not match")
    return world, state
