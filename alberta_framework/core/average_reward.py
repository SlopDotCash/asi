# mypy: disable-error-code="attr-defined,call-arg,name-defined"
"""Average-reward prediction and control primitives.

This module provides the linear primitives used by the Alberta Plan Step 5 and 6 facades:
linear differential TD prediction and linear differential SARSA control in a
continuing, temporally uniform setting.  The algorithms update their value
weights, traces, policies, and average-reward estimate on every transition.
They intentionally stay small and linear so they can serve as a stable target
for later nonlinear Horde/GQ/GTD and actor-critic integrations.

References:
    Wan, Naik, & Sutton (2021). "Learning and Planning in Average-Reward
        Markov Decision Processes."  (Differential TD/Q-learning; the
        TD-error-driven reward-rate update used throughout this module.)
    Sutton et al. (2009). "Fast Gradient-Descent Methods for
        Temporal-Difference Learning with Linear Function Approximation."
        (GTD2/TDC secondary-weight scheme.)
    Maei (2011). "Gradient Temporal-Difference Learning Algorithms."
        PhD thesis, University of Alberta.
    Sutton & Barto (2018), ch. 10.  (Differential semi-gradient SARSA.)
"""

from __future__ import annotations

import dataclasses
import functools
import operator
import time
from collections.abc import Mapping
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core._float32_scalars import (
    validated_float32_scalar,
    validated_float32_scalar_with_ratio,
)
from alberta_framework.core.learners import _update_from_gradient_with_diagnostics
from alberta_framework.core.multi_head_learner import MultiHeadMLPLearner, MultiHeadMLPState
from alberta_framework.core.optimizers import Autostep, AutostepParamState, optimizer_from_config
from alberta_framework.core.update_safety import (
    floating_tree_is_finite as _floating_tree_is_finite,
)

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_FLOAT32_HALF_MIN_SUBNORMAL_DENOMINATOR = 1 << 150
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


# Matches the class of ceiling already established for other scan-driven
# array-loop runners in ``core`` (e.g. ``sarsa._SARSA_SEQUENCE_MAX_STEPS``,
# ``learners._LEARNING_LOOP_MAX_STEPS``). Set with headroom above this
# module's largest exercised sequence (20,000 steps, see
# ``test_differential_gtd_scan_learns_average_reward_cycle``) while still
# bounding the leading axis that every ``run_*_from_arrays`` runner below
# hands straight to ``jax.lax.scan``.
_AVERAGE_REWARD_SEQUENCE_MAX_STEPS = 50_000


def _require_avg_reward_sequence_length(name: str, value: object) -> int:
    """Reject an oversized or malformed leading axis before it drives a scan.

    The array-loop runners in this module (``run_differential_td_from_arrays``,
    ``run_differential_gtd_from_arrays``, ``run_differential_sarsa_from_arrays``,
    ``run_average_reward_horde_actor_critic_from_arrays``) hand their step
    arrays straight to ``jax.lax.scan`` with no bound on the leading (step)
    axis. A hostile or mistaken caller supplying a huge leading length forces
    JAX to materialize per-step outputs (and, for large enough arrays, the
    inputs) at that length, exhausting memory or hanging the process well
    before any step executes.
    """
    if not isinstance(value, jax.Array):
        raise TypeError(f"{name} must be a JAX array")
    if value.ndim < 1:
        raise ValueError(f"{name} must have a leading step axis")
    length = int(value.shape[0])
    if length < 1 or length > _AVERAGE_REWARD_SEQUENCE_MAX_STEPS:
        raise ValueError(
            f"{name} length must be an integer in [1, {_AVERAGE_REWARD_SEQUENCE_MAX_STEPS}]"
        )
    return length


def _require_avg_reward_matching_length(name: str, value: object, *, expected: int) -> None:
    if not isinstance(value, jax.Array):
        raise TypeError(f"{name} must be a JAX array")
    if value.ndim < 1 or int(value.shape[0]) != expected:
        raise ValueError(f"{name} must share the same leading length as the primary sequence")


def _validate_hidden_sizes(hidden_sizes: object) -> tuple[int, ...]:
    if type(hidden_sizes) is not tuple:
        raise ValueError("hidden_sizes must be an actual tuple")
    return tuple(_require_int32("hidden_sizes element", size, minimum=1) for size in hidden_sizes)


def _decode_hidden_sizes(hidden_sizes: object) -> tuple[int, ...]:
    if type(hidden_sizes) is not list:
        raise ValueError("serialized hidden_sizes must be an actual list")
    sequence = cast(list[object], hidden_sizes)
    return _validate_hidden_sizes(tuple(sequence))


def _require_state_resources(name: str, *, scalars: int, nbytes: int) -> None:
    if scalars > _INT32_MAX:
        raise ValueError(f"{name} state scalars must fit signed int32")
    if nbytes > _INT32_MAX:
        raise ValueError(f"{name} state bytes must fit signed int32")


def _validated_nonnegative_float32_scalar(
    name: str,
    value: object,
    *,
    upper: float | None = None,
) -> float:
    """Validate a nonnegative float32 sink without losing a nonzero value.

    Exact zero is a supported way to disable these updates. A positive host
    value that rounds to binary32 zero is not: accepting it would silently
    turn an enabled learning rate or trace into the disabled configuration.
    """
    stored, numerator, denominator = validated_float32_scalar_with_ratio(
        name,
        value,
        lower=0.0,
        upper=upper,
    )
    if (
        numerator != 0
        and numerator * _FLOAT32_HALF_MIN_SUBNORMAL_DENOMINATOR <= denominator
    ):
        raise ValueError(f"{name} must remain nonzero once narrowed to float32")
    return stored


def _validated_float32_scalar_preserving_nonzero(name: str, value: object) -> float:
    """Validate a signed float32 scalar without erasing an exact nonzero."""
    stored, numerator, denominator = validated_float32_scalar_with_ratio(name, value)
    if (
        numerator != 0
        and abs(numerator) * _FLOAT32_HALF_MIN_SUBNORMAL_DENOMINATOR <= denominator
    ):
        raise ValueError(f"{name} must remain nonzero once narrowed to float32")
    return stored


def _preflight_actor_state(
    n_actions: int,
    observation_dim: int,
    hidden_sizes: tuple[int, ...],
) -> None:
    actor_dim = hidden_sizes[-1] if hidden_sizes else observation_dim
    actor_parameters = n_actions * actor_dim
    float32_scalars = 4 * actor_parameters + 6 * n_actions + observation_dim + 8
    actor_scalar_count = float32_scalars + 5
    layer_sizes = (observation_dim, *hidden_sizes)
    trunk_parameters = sum(
        fan_out * (fan_in + 1)
        for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:], strict=False)
    )
    critic_parameters = trunk_parameters + actor_dim + 1
    # The critic's default MultiHead learner owns one scalar LMS state for
    # every trunk and head weight/bias array, in addition to parameters,
    # traces, utilities, counters, and its average-reward vector.
    critic_lms_state_scalars = 2 * len(hidden_sizes) + 2
    critic_scalar_count = (
        2 * critic_parameters
        + sum(hidden_sizes)
        + critic_lms_state_scalars
        + 5
    )
    scalar_count = actor_scalar_count + critic_scalar_count
    _require_state_resources(
        "average-reward actor-critic",
        scalars=scalar_count,
        nbytes=4 * scalar_count,
    )


def _preflight_differential_sarsa_state(n_actions: int, feature_dim: int) -> None:
    parameters = n_actions * feature_dim
    float32_scalars = 2 * parameters + 2 * n_actions + feature_dim + 2
    # Two int32 leaves, a two-word RNG key, and a two-word lifetime counter.
    scalar_count = float32_scalars + 6
    _require_state_resources(
        "differential SARSA",
        scalars=scalar_count,
        nbytes=4 * scalar_count,
    )


def _preflight_differential_td_state(feature_dim: int) -> None:
    scalar_count = 2 * feature_dim + 4
    _require_state_resources(
        "differential TD",
        scalars=scalar_count,
        nbytes=4 * scalar_count,
    )


def _differential_td_persistent_bytes(feature_dim: int) -> int:
    """Named persist already preflighted by ``DifferentialTDLearner.init``."""
    return 4 * (2 * feature_dim + 4)


def _differential_td_update_result_extras_bytes() -> int:
    """Returned ``DifferentialTDUpdateResult`` extras excluding persist.

    Nested ``state`` is persist and is already counted in the simultaneous
    persist copies. These extras are the published prediction, TD-error,
    reward-rate, metric, and diagnostic leaves.
    """
    return 33


def _differential_td_update_working_set_bytes(feature_dim: int) -> int:
    """Source persist, proposed persist, committed persist, and returned extras."""
    return 3 * _differential_td_persistent_bytes(
        feature_dim
    ) + _differential_td_update_result_extras_bytes()


def _preflight_differential_td_update_working_set(feature_dim: int) -> None:
    """Reject an update envelope the host cannot name in signed int32."""
    if _differential_td_update_working_set_bytes(feature_dim) > _INT32_MAX:
        raise ValueError(
            "differential TD update working set byte count must fit signed int32"
        )


def _preflight_differential_gtd_state(feature_dim: int) -> None:
    scalar_count = 3 * feature_dim + 5
    _require_state_resources(
        "differential GTD",
        scalars=scalar_count,
        nbytes=4 * scalar_count,
    )


DIFFERENTIAL_SARSA_STATE_SCHEMA = "alberta.differential-sarsa-state.v2"
DIFFERENTIAL_SARSA_LIFETIME_COUNTER_NBYTES = 12
DIFFERENTIAL_SARSA_LIFETIME_COUNTER_DELTA_NBYTES = 8


def _saturating_int32_increment(value: Array) -> Array:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


def _checked_lifetime_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose the next exact transition identity without all-ones wrap."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("differential SARSA lifetime words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("differential SARSA lifetime words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity_available = ~jnp.all(words == maximum)
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = words[1] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low))
    return (
        jnp.where(capacity_available, proposed, words).astype(jnp.uint32),
        capacity_available,
    )


def _lifetime_counter_valid(words: Array, telemetry: Array) -> Bool[Array, ""]:
    """Validate the exact identity against saturating compatibility telemetry."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("differential SARSA lifetime words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("differential SARSA lifetime words must have dtype uint32")
    if getattr(telemetry, "shape", None) != ():
        raise ValueError("differential SARSA step_count must be scalar")
    if getattr(telemetry, "dtype", None) != jnp.dtype(jnp.int32):
        raise TypeError("differential SARSA step_count must have dtype int32")
    maximum_i32 = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return (telemetry >= 0) & jnp.where(
        below_saturation,
        telemetry == words[1].astype(jnp.int32),
        telemetry == maximum_i32,
    )


@dataclasses.dataclass(frozen=True)
class DifferentialTDConfig:
    """Configuration for linear differential TD prediction.

    Attributes:
        step_size: Step-size for value weights and bias.
        average_reward_step_size: Step-size for the reward-rate estimate.
        trace_decay: Accumulating trace decay `lambda`.
    """

    step_size: float = 0.05
    average_reward_step_size: float = 0.01
    trace_decay: float = 0.0

    def __post_init__(self) -> None:
        """Validate scalar hyperparameters."""
        for name in ("step_size", "average_reward_step_size"):
            object.__setattr__(
                self,
                name,
                _validated_nonnegative_float32_scalar(name, getattr(self, name)),
            )
        object.__setattr__(
            self,
            "trace_decay",
            _validated_nonnegative_float32_scalar(
                "trace_decay",
                self.trace_decay,
                upper=1.0,
            ),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize this config to a dictionary."""
        return {
            "type": "DifferentialTDConfig",
            "step_size": self.step_size,
            "average_reward_step_size": self.average_reward_step_size,
            "trace_decay": self.trace_decay,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DifferentialTDConfig:
        """Reconstruct a config from :meth:`to_config` output."""
        config = dict(config)
        config.pop("type", None)
        return cls(**config)


@chex.dataclass(frozen=True)
class DifferentialTDState:
    """State for linear differential TD prediction.

    Attributes:
        weights: Linear differential value weights.
        bias: Scalar differential value bias.
        eligibility_traces: Accumulating value-weight trace.
        bias_eligibility_trace: Accumulating bias trace.
        average_reward: Running estimate of the continuing reward rate.
        step_count: Number of transition updates.
        birth_timestamp: Wall-clock seconds at initialization.
        uptime_s: Cumulative wall-clock seconds spent in array loops.
    """

    weights: Float[Array, " feature_dim"]
    bias: Float[Array, ""]
    eligibility_traces: Float[Array, " feature_dim"]
    bias_eligibility_trace: Float[Array, ""]
    average_reward: Float[Array, ""]
    step_count: Int[Array, ""]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class DifferentialTDUpdateResult:
    """Result from one differential TD transition update."""

    state: DifferentialTDState
    prediction: Float[Array, ""]
    next_prediction: Float[Array, ""]
    td_error: Float[Array, ""]
    average_reward: Float[Array, ""]
    metrics: Float[Array, " 4"]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class DifferentialTDArrayResult:
    """Result from scanning differential TD over arrays."""

    state: DifferentialTDState
    predictions: Float[Array, " num_steps"]
    td_errors: Float[Array, " num_steps"]
    average_rewards: Float[Array, " num_steps"]
    metrics: Float[Array, "num_steps 4"]
    updates_applied: Bool[Array, " num_steps"]


@dataclasses.dataclass(frozen=True)
class DifferentialGTDConfig:
    """Configuration for linear differential GTD/TDC prediction.

    Attributes:
        value_step_size: Step-size for primary value weights and bias.
        secondary_step_size: Step-size for the GTD secondary weights.
        average_reward_step_size: Step-size for the reward-rate estimate.
        trace_decay: Accumulating trace decay `lambda`.
        ratio_clip: Maximum importance-sampling ratio.
    """

    value_step_size: float = 0.05
    secondary_step_size: float = 0.01
    average_reward_step_size: float = 0.01
    trace_decay: float = 0.0
    ratio_clip: float = 10.0

    def __post_init__(self) -> None:
        """Validate scalar hyperparameters."""
        for name in (
            "value_step_size",
            "secondary_step_size",
            "average_reward_step_size",
        ):
            object.__setattr__(
                self,
                name,
                _validated_nonnegative_float32_scalar(name, getattr(self, name)),
            )
        object.__setattr__(
            self,
            "trace_decay",
            _validated_nonnegative_float32_scalar(
                "trace_decay",
                self.trace_decay,
                upper=1.0,
            ),
        )
        object.__setattr__(
            self,
            "ratio_clip",
            validated_float32_scalar("ratio_clip", self.ratio_clip, positive=True),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize this config to a dictionary."""
        return {
            "type": "DifferentialGTDConfig",
            "value_step_size": self.value_step_size,
            "secondary_step_size": self.secondary_step_size,
            "average_reward_step_size": self.average_reward_step_size,
            "trace_decay": self.trace_decay,
            "ratio_clip": self.ratio_clip,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DifferentialGTDConfig:
        """Reconstruct a config from :meth:`to_config` output."""
        config = dict(config)
        config.pop("type", None)
        return cls(**config)


@chex.dataclass(frozen=True)
class DifferentialGTDState:
    """State for linear differential GTD/TDC prediction."""

    weights: Float[Array, " feature_dim"]
    bias: Float[Array, ""]
    secondary_weights: Float[Array, " feature_dim"]
    secondary_bias: Float[Array, ""]
    eligibility_traces: Float[Array, " feature_dim"]
    bias_eligibility_trace: Float[Array, ""]
    average_reward: Float[Array, ""]
    step_count: Int[Array, ""]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class DifferentialGTDUpdateResult:
    """Result from one differential GTD/TDC transition update."""

    state: DifferentialGTDState
    prediction: Float[Array, ""]
    next_prediction: Float[Array, ""]
    td_error: Float[Array, ""]
    rho_clipped: Float[Array, ""]
    average_reward: Float[Array, ""]
    metrics: Float[Array, " 6"]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class DifferentialGTDArrayResult:
    """Result from scanning differential GTD/TDC over arrays."""

    state: DifferentialGTDState
    predictions: Float[Array, " num_steps"]
    td_errors: Float[Array, " num_steps"]
    average_rewards: Float[Array, " num_steps"]
    rho_clipped: Float[Array, " num_steps"]
    metrics: Float[Array, "num_steps 6"]
    updates_applied: Bool[Array, " num_steps"]


@chex.dataclass(frozen=True)
class AverageRewardHordeState:
    """State for a shared-trunk average-reward Horde."""

    learner_state: MultiHeadMLPState
    average_rewards: Float[Array, " n_demons"]
    step_count: Int[Array, ""]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class AverageRewardHordeUpdateResult:
    """Result from one average-reward Horde update."""

    state: AverageRewardHordeState
    predictions: Float[Array, " n_demons"]
    next_predictions: Float[Array, " n_demons"]
    td_errors: Float[Array, " n_demons"]
    td_targets: Float[Array, " n_demons"]
    average_rewards: Float[Array, " n_demons"]
    per_demon_metrics: Float[Array, "n_demons 3"]
    head_updates_applied: Bool[Array, " n_demons"]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AverageRewardHordeLearningResult:
    """Result from scanning an average-reward Horde over arrays."""

    state: AverageRewardHordeState
    td_errors: Float[Array, "num_steps n_demons"]
    average_rewards: Float[Array, "num_steps n_demons"]
    per_demon_metrics: Float[Array, "num_steps n_demons 3"]
    head_updates_applied: Bool[Array, "num_steps n_demons"]
    updates_applied: Bool[Array, " num_steps"]


@dataclasses.dataclass(frozen=True)
class AverageRewardHordeActorCriticConfig:
    """Config for a Horde-critic average-reward actor-critic agent."""

    n_actions: int
    hidden_sizes: tuple[int, ...] = (16,)
    critic_step_size: float = 0.02
    average_reward_step_size: float = 0.01
    temperature: float = 1.0
    epsilon: float = 0.0
    actor_update_clip: float = 0.1
    logit_clip: float = 20.0

    def __post_init__(self) -> None:
        """Validate scalar hyperparameters."""
        object.__setattr__(
            self,
            "n_actions",
            _require_int32("n_actions", self.n_actions, minimum=1),
        )
        object.__setattr__(
            self,
            "hidden_sizes",
            _validate_hidden_sizes(self.hidden_sizes),
        )
        for name in ("critic_step_size", "average_reward_step_size"):
            object.__setattr__(
                self,
                name,
                validated_float32_scalar(name, getattr(self, name), lower=0.0),
            )
        for name in ("temperature", "actor_update_clip", "logit_clip"):
            object.__setattr__(
                self,
                name,
                validated_float32_scalar(name, getattr(self, name), positive=True),
            )
        object.__setattr__(
            self,
            "epsilon",
            validated_float32_scalar("epsilon", self.epsilon, lower=0.0, upper=1.0),
        )
        if self.hidden_sizes:
            _preflight_actor_state(self.n_actions, 1, self.hidden_sizes)

    def to_config(self) -> dict[str, Any]:
        """Serialize this config to a dictionary."""
        return {
            "type": "AverageRewardHordeActorCriticConfig",
            "n_actions": self.n_actions,
            "hidden_sizes": list(self.hidden_sizes),
            "critic_step_size": self.critic_step_size,
            "average_reward_step_size": self.average_reward_step_size,
            "temperature": self.temperature,
            "epsilon": self.epsilon,
            "actor_update_clip": self.actor_update_clip,
            "logit_clip": self.logit_clip,
        }

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
    ) -> AverageRewardHordeActorCriticConfig:
        """Reconstruct a config from :meth:`to_config` output."""
        if type(config) is not dict:
            raise ValueError("actor-critic config must be an actual dict")
        payload = dict(config)
        expected = {field.name for field in dataclasses.fields(cls)} | {"type"}
        if set(payload) != expected:
            raise ValueError("actor-critic config fields do not match its schema")
        if payload.pop("type") != "AverageRewardHordeActorCriticConfig":
            raise ValueError("actor-critic config type is invalid")
        payload["hidden_sizes"] = _decode_hidden_sizes(payload["hidden_sizes"])
        return cls(**payload)


@chex.dataclass(frozen=True)
class DiscretePolicySample:
    """Auditable target/behavior probabilities for one sampled action."""

    action: Int[Array, ""]
    target_policy: Float[Array, " n_actions"]
    behavior_policy: Float[Array, " n_actions"]
    target_probability: Float[Array, ""]
    behavior_probability: Float[Array, ""]
    target_log_probability: Float[Array, ""]
    behavior_log_probability: Float[Array, ""]


@chex.dataclass(frozen=True)
class AverageRewardHordeActorCriticState:
    """State for the average-reward Horde actor-critic."""

    critic_state: AverageRewardHordeState
    actor_weights: Float[Array, "n_actions feature_dim"]
    actor_bias: Float[Array, " n_actions"]
    actor_opt_w: AutostepParamState
    actor_opt_b: AutostepParamState
    last_observation: Float[Array, " observation_dim"]
    last_action: Int[Array, ""]
    last_policy_sample: DiscretePolicySample
    rng_key: Array
    step_count: Int[Array, ""]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class AverageRewardHordeActorCriticUpdateResult:
    """Result from one average-reward Horde actor-critic update."""

    state: AverageRewardHordeActorCriticState
    action: Int[Array, ""]
    # Actual epsilon-mixture policy that sampled ``action``.
    policy: Float[Array, " n_actions"]
    # Softmax target component before fixed uniform exploration is mixed in.
    target_policy: Float[Array, " n_actions"]
    behavior_action_probability: Float[Array, ""]
    target_action_probability: Float[Array, ""]
    # Multiplier converting grad log(target) to grad log(behavior) on the
    # transition that was just learned from.
    actor_score_scale: Float[Array, ""]
    td_error: Float[Array, ""]
    average_reward: Float[Array, ""]
    critic_prediction: Float[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AverageRewardHordeActorCriticArrayResult:
    """Result from scanning average-reward Horde actor-critic updates."""

    state: AverageRewardHordeActorCriticState
    actions: Int[Array, " num_steps"]
    # Actual epsilon-mixture behavior policies.
    policies: Float[Array, "num_steps n_actions"]
    target_policies: Float[Array, "num_steps n_actions"]
    behavior_action_probabilities: Float[Array, " num_steps"]
    target_action_probabilities: Float[Array, " num_steps"]
    actor_score_scales: Float[Array, " num_steps"]
    td_errors: Float[Array, " num_steps"]
    average_rewards: Float[Array, " num_steps"]
    updates_applied: Bool[Array, " num_steps"]


class AverageRewardHordeActorCriticAgent:
    """Average-reward actor-critic using a nonlinear Horde critic.

    The critic is a one-head :class:`AverageRewardHordeLearner`.  The actor is a
    softmax target policy over the critic trunk's learned nonlinear feature
    vector.  The acting policy is the explicit epsilon mixture
    ``b(a|s) = (1-epsilon) pi(a|s) + epsilon / |A|``.  Actor gradients optimize
    that behavior policy exactly:

    ``grad log b = ((1-epsilon) pi(a|s) / b(a|s)) grad log pi``.

    This distinction matters at large exploration rates: treating actions
    sampled from the mixture as if they came directly from ``pi`` biases the
    score gradient.  In particular, ``epsilon=1`` makes the behavior independent
    of actor parameters, so its actor gradient must be exactly zero.

    The actor deliberately keeps no eligibility trace in this implementation
    slice; every transition still updates the actor, critic, and reward-rate
    baseline.
    """

    def __init__(
        self,
        config: AverageRewardHordeActorCriticConfig,
        actor_optimizer: Autostep | None = None,
    ):
        """Initialize the agent."""
        self._config = config
        self._actor_optimizer = (
            actor_optimizer if actor_optimizer is not None else Autostep(initial_step_size=0.05)
        )
        self._critic = AverageRewardHordeLearner(
            n_demons=1,
            hidden_sizes=config.hidden_sizes,
            step_size=config.critic_step_size,
            average_reward_step_size=config.average_reward_step_size,
            sparsity=0.0,
            use_layer_norm=False,
        )

    @property
    def config(self) -> AverageRewardHordeActorCriticConfig:
        """Agent configuration."""
        return self._config

    @property
    def actor_optimizer(self) -> Autostep:
        """Per-weight Autostep optimizer for the actor."""
        return self._actor_optimizer

    @property
    def critic(self) -> AverageRewardHordeLearner:
        """Underlying one-head average-reward Horde critic."""
        return self._critic

    @property
    def actor_feature_dim(self) -> int:
        """Actor feature dimension implied by the critic trunk."""
        if self._config.hidden_sizes:
            return self._config.hidden_sizes[-1]
        raise ValueError("actor_feature_dim requires initialized feature_dim")

    def to_config(self) -> dict[str, Any]:
        """Serialize this agent to a dictionary."""
        return {
            "type": "AverageRewardHordeActorCriticAgent",
            "config": self._config.to_config(),
            "actor_optimizer": self._actor_optimizer.to_config(),
        }

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
    ) -> AverageRewardHordeActorCriticAgent:
        """Reconstruct an agent from :meth:`to_config` output."""
        if type(config) is not dict:
            raise ValueError("actor-critic agent config must be an actual dict")
        if set(config) != {"type", "config", "actor_optimizer"}:
            raise ValueError("actor-critic agent config fields do not match its schema")
        if config["type"] != "AverageRewardHordeActorCriticAgent":
            raise ValueError("actor-critic agent config type is invalid")
        nested = config["config"]
        optimizer_config = config["actor_optimizer"]
        if type(nested) is not dict or type(optimizer_config) is not dict:
            raise ValueError("actor-critic nested configs must be actual dicts")
        cfg = AverageRewardHordeActorCriticConfig.from_config(nested)
        actor_opt = cast(Autostep, optimizer_from_config(optimizer_config))
        return cls(cfg, actor_optimizer=actor_opt)

    def init(self, observation_dim: int, key: Array) -> AverageRewardHordeActorCriticState:
        """Initialize critic and actor state."""
        observation_dim = _require_int32("observation_dim", observation_dim, minimum=1)
        actor_dim = self._config.hidden_sizes[-1] if self._config.hidden_sizes else observation_dim
        _preflight_actor_state(
            self._config.n_actions, observation_dim, self._config.hidden_sizes
        )
        key, critic_key = jr.split(key)
        critic_state = self._critic.init(observation_dim, critic_key)
        actor_opt_w = self._actor_optimizer.init_for_shape((self._config.n_actions, actor_dim))
        actor_opt_b = self._actor_optimizer.init_for_shape((self._config.n_actions,))
        initial_policy = jnp.full(
            (self._config.n_actions,),
            1.0 / self._config.n_actions,
            dtype=jnp.float32,
        )
        initial_sample = DiscretePolicySample(
            action=jnp.array(-1, dtype=jnp.int32),
            target_policy=initial_policy,
            behavior_policy=initial_policy,
            target_probability=jnp.array(0.0, dtype=jnp.float32),
            behavior_probability=jnp.array(0.0, dtype=jnp.float32),
            target_log_probability=jnp.array(-jnp.inf, dtype=jnp.float32),
            behavior_log_probability=jnp.array(-jnp.inf, dtype=jnp.float32),
        )
        return AverageRewardHordeActorCriticState(
            critic_state=critic_state,
            actor_weights=jnp.zeros((self._config.n_actions, actor_dim), dtype=jnp.float32),
            actor_bias=jnp.zeros((self._config.n_actions,), dtype=jnp.float32),
            actor_opt_w=actor_opt_w,
            actor_opt_b=actor_opt_b,
            last_observation=jnp.zeros((observation_dim,), dtype=jnp.float32),
            last_action=jnp.array(-1, dtype=jnp.int32),
            last_policy_sample=initial_sample,
            rng_key=key,
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    def _actor_features(
        self,
        state: AverageRewardHordeActorCriticState,
        observation: Array,
    ) -> Array:
        learner_state = state.critic_state.learner_state
        return MultiHeadMLPLearner._trunk_forward(
            learner_state.trunk_params.weights,
            learner_state.trunk_params.biases,
            observation,
            self._critic._leaky_relu_slope,  # noqa: SLF001
            self._critic._use_layer_norm,  # noqa: SLF001
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy(
        self,
        state: AverageRewardHordeActorCriticState,
        observation: Array,
    ) -> Float[Array, " n_actions"]:
        """Compute softmax target-policy probabilities."""
        features = self._actor_features(state, observation)
        logits = state.actor_weights @ features + state.actor_bias
        logits = jnp.clip(logits, -self._config.logit_clip, self._config.logit_clip)
        return jax.nn.softmax(logits / self._config.temperature)

    @functools.partial(jax.jit, static_argnums=(0,))
    def behavior_policy(
        self,
        state: AverageRewardHordeActorCriticState,
        observation: Array,
    ) -> Float[Array, " n_actions"]:
        """Return the epsilon-mixture policy used to sample actions."""

        target = self.policy(state, observation)
        uniform = jnp.asarray(
            1.0 / self._config.n_actions,
            dtype=jnp.float32,
        )
        return jnp.asarray(
            (1.0 - self._config.epsilon) * target + self._config.epsilon * uniform,
            dtype=jnp.float32,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def sample_policy(
        self,
        state: AverageRewardHordeActorCriticState,
        observation: Array,
    ) -> tuple[DiscretePolicySample, Array]:
        """Sample and log the exact target/behavior policy contract."""

        key, sample_key = jr.split(state.rng_key)
        target = self.policy(state, observation)
        behavior = self.behavior_policy(state, observation)
        action = jr.categorical(
            sample_key,
            jnp.log(jnp.maximum(behavior, 1e-8)),
        ).astype(jnp.int32)
        target_probability = target[action]
        behavior_probability = behavior[action]
        return DiscretePolicySample(
            action=action,
            target_policy=target,
            behavior_policy=behavior,
            target_probability=target_probability,
            behavior_probability=behavior_probability,
            target_log_probability=jnp.log(jnp.maximum(target_probability, 1e-8)),
            behavior_log_probability=jnp.log(jnp.maximum(behavior_probability, 1e-8)),
        ), key

    @functools.partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        state: AverageRewardHordeActorCriticState,
        observation: Array,
    ) -> tuple[Int[Array, ""], Array]:
        """Sample an epsilon-soft action (compatibility convenience API)."""

        sample, key = self.sample_policy(state, observation)
        return sample.action, key

    @functools.partial(jax.jit, static_argnums=(0,))
    def start(
        self,
        state: AverageRewardHordeActorCriticState,
        observation: Array,
    ) -> tuple[AverageRewardHordeActorCriticState, Int[Array, ""]]:
        """Select and store the first action."""
        sample, key = self.sample_policy(state, observation)
        return state.replace(
            last_observation=observation,
            last_action=sample.action,
            last_policy_sample=sample,
            rng_key=key,
        ), sample.action

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: AverageRewardHordeActorCriticState,
        reward: Array,
        next_observation: Array,
    ) -> AverageRewardHordeActorCriticUpdateResult:
        """Apply one average-reward actor-critic update."""
        action_valid = (state.last_action >= 0) & (state.last_action < self._config.n_actions)
        safe_last_action = jnp.clip(
            state.last_action,
            0,
            self._config.n_actions - 1,
        )
        old_features = self._actor_features(state, state.last_observation)
        old_sample = state.last_policy_sample
        critic_result = self._critic.update(
            state.critic_state,
            state.last_observation,
            jnp.atleast_1d(jnp.asarray(reward, dtype=jnp.float32)),
            next_observation,
        )
        td_error = critic_result.td_errors[0]
        actor_td_error = jnp.clip(
            td_error,
            -self._config.logit_clip,
            self._config.logit_clip,
        )
        action_mask = jax.nn.one_hot(
            safe_last_action,
            self._config.n_actions,
            dtype=jnp.float32,
        )
        actor_score_scale = (
            (1.0 - self._config.epsilon)
            * old_sample.target_probability
            / jnp.maximum(old_sample.behavior_probability, 1e-8)
        )
        score = (
            actor_score_scale * (action_mask - old_sample.target_policy) / self._config.temperature
        )
        grad_log_policy = score[:, None] * old_features[None, :]
        bias_grad = score
        raw_w, new_opt_w, weight_update_applied = _update_from_gradient_with_diagnostics(
            self._actor_optimizer,
            state.actor_opt_w,
            grad_log_policy,
            error=actor_td_error,
        )
        raw_b, new_opt_b, bias_update_applied = _update_from_gradient_with_diagnostics(
            self._actor_optimizer,
            state.actor_opt_b,
            bias_grad,
            error=actor_td_error,
        )
        weight_step = jnp.clip(
            actor_td_error * raw_w,
            -self._config.actor_update_clip,
            self._config.actor_update_clip,
        )
        bias_step = jnp.clip(
            actor_td_error * raw_b,
            -self._config.actor_update_clip,
            self._config.actor_update_clip,
        )
        candidate_state = state.replace(
            critic_state=critic_result.state,
            actor_weights=state.actor_weights + weight_step,
            actor_bias=state.actor_bias + bias_step,
            actor_opt_w=new_opt_w,
            actor_opt_b=new_opt_b,
            step_count=_saturating_int32_increment(state.step_count),
        )
        # The cached prior decision above owns the transition being learned.
        # Sample the successor only after the critic trunk, actor, optimizer,
        # reward-rate baseline, and step counter have committed atomically, so
        # the stored policy is exactly the current policy at next_observation.
        next_sample, key = self.sample_policy(candidate_state, next_observation)
        proposed_state = candidate_state.replace(
            last_observation=next_observation,
            last_action=next_sample.action,
            last_policy_sample=next_sample,
            rng_key=key,
        )
        inputs_valid = (
            action_valid
            & jnp.isfinite(jnp.squeeze(reward))
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.all(jnp.isfinite(old_features))
            & jnp.isfinite(td_error)
            & jnp.isfinite(actor_score_scale)
        )
        update_applied = (
            inputs_valid
            & _floating_tree_is_finite(state)
            & _floating_tree_is_finite(proposed_state)
            & critic_result.update_applied
            & critic_result.head_updates_applied[0]
            & weight_update_applied
            & bias_update_applied
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda: proposed_state,
            lambda: state,
        )
        return AverageRewardHordeActorCriticUpdateResult(
            state=new_state,
            action=jnp.where(
                update_applied,
                next_sample.action,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            policy=jnp.where(
                update_applied,
                next_sample.behavior_policy,
                jnp.zeros_like(next_sample.behavior_policy),
            ),
            target_policy=jnp.where(
                update_applied,
                next_sample.target_policy,
                jnp.zeros_like(next_sample.target_policy),
            ),
            behavior_action_probability=jnp.where(
                update_applied, next_sample.behavior_probability, 0.0
            ),
            target_action_probability=jnp.where(
                update_applied, next_sample.target_probability, 0.0
            ),
            actor_score_scale=jnp.where(update_applied, actor_score_scale, 0.0),
            td_error=jnp.where(update_applied, td_error, 0.0),
            average_reward=jnp.where(update_applied, critic_result.average_rewards[0], 0.0),
            critic_prediction=jnp.where(update_applied, critic_result.predictions[0], 0.0),
            update_applied=update_applied,
        )


class AverageRewardHordeLearner:
    """Shared-trunk nonlinear Horde for differential GVF prediction.

    Each head learns a continuing differential value target
    `r_i - rbar_i + v_i(s')`, while `rbar_i` tracks the per-demon average
    cumulant.  The trunk is updated every step through ``MultiHeadMLPLearner``;
    per-head traces can be enabled without trunk temporal traces, matching the
    Step 3 Horde trace constraint.
    """

    def __init__(
        self,
        n_demons: int,
        *,
        hidden_sizes: tuple[int, ...] = (32,),
        step_size: float = 0.01,
        average_reward_step_size: float = 0.01,
        trace_decay: float = 0.0,
        sparsity: float = 0.0,
        use_layer_norm: bool = False,
        leaky_relu_slope: float = 0.01,
    ):
        """Initialize the average-reward Horde."""
        n_demons = _require_int32("n_demons", n_demons, minimum=1)
        hidden_sizes = _validate_hidden_sizes(hidden_sizes)
        step_size = _validated_nonnegative_float32_scalar("step_size", step_size)
        average_reward_step_size = _validated_nonnegative_float32_scalar(
            "average_reward_step_size",
            average_reward_step_size,
        )
        trace_decay = _validated_nonnegative_float32_scalar(
            "trace_decay",
            trace_decay,
            upper=1.0,
        )
        sparsity = validated_float32_scalar("sparsity", sparsity, lower=0.0, upper=1.0)
        leaky_relu_slope = validated_float32_scalar("leaky_relu_slope", leaky_relu_slope, lower=0.0)
        if type(use_layer_norm) is not bool:
            raise ValueError("use_layer_norm must be an actual bool")
        self._n_demons = n_demons
        self._hidden_sizes = hidden_sizes
        self._step_size = step_size
        self._average_reward_step_size = average_reward_step_size
        self._trace_decay = trace_decay
        self._sparsity = sparsity
        self._use_layer_norm = use_layer_norm
        self._leaky_relu_slope = leaky_relu_slope
        self._learner = MultiHeadMLPLearner(
            n_heads=n_demons,
            hidden_sizes=hidden_sizes,
            step_size=step_size,
            gamma=0.0,
            lamda=0.0,
            per_head_gamma_lamda=tuple(trace_decay for _ in range(n_demons)),
            sparsity=sparsity,
            use_layer_norm=use_layer_norm,
            leaky_relu_slope=leaky_relu_slope,
        )

    @property
    def n_demons(self) -> int:
        """Number of average-reward GVF heads."""
        return self._n_demons

    @property
    def learner(self) -> MultiHeadMLPLearner:
        """Underlying shared-trunk learner."""
        return self._learner

    def to_config(self) -> dict[str, Any]:
        """Serialize this learner to a dictionary."""
        return {
            "type": "AverageRewardHordeLearner",
            "n_demons": self._n_demons,
            "hidden_sizes": list(self._hidden_sizes),
            "step_size": self._step_size,
            "average_reward_step_size": self._average_reward_step_size,
            "trace_decay": self._trace_decay,
            "sparsity": self._sparsity,
            "use_layer_norm": self._use_layer_norm,
            "leaky_relu_slope": self._leaky_relu_slope,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AverageRewardHordeLearner:
        """Reconstruct a learner from :meth:`to_config` output."""
        config = dict(config)
        config.pop("type", None)
        config["hidden_sizes"] = tuple(config["hidden_sizes"])
        return cls(**config)

    def init(self, feature_dim: int, key: Array) -> AverageRewardHordeState:
        """Initialize shared-trunk and per-demon reward-rate state."""
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        return AverageRewardHordeState(
            learner_state=self._learner.init(feature_dim, key),
            average_rewards=jnp.zeros(self._n_demons, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, state: AverageRewardHordeState, observation: Array) -> Array:
        """Predict all average-reward GVF heads."""
        return cast(Array, self._learner.predict(state.learner_state, observation))

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: AverageRewardHordeState,
        observation: Array,
        cumulants: Array,
        next_observation: Array,
    ) -> AverageRewardHordeUpdateResult:
        """Apply one shared-trunk differential Horde update."""
        cumulants = jnp.asarray(cumulants)
        expected_shape = (self._n_demons,)
        if cumulants.shape != expected_shape:
            raise ValueError(f"cumulants must have shape {expected_shape}, got {cumulants.shape}")
        next_predictions = self._learner.predict(state.learner_state, next_observation)
        # NaN remains the inactive-head sentinel. Other non-finite cumulants
        # are rejected per head and reported separately from inactivity.
        requested = ~jnp.isnan(cumulants)
        raw_targets = cumulants - state.average_rewards + next_predictions
        active = requested & jnp.isfinite(cumulants) & jnp.isfinite(raw_targets)
        targets = jnp.where(active, raw_targets, jnp.nan)
        result = self._learner.update(state.learner_state, observation, targets)
        td_errors = result.errors
        safe_td_errors = jnp.where(active, td_errors, 0.0)
        proposed_average_rewards = (
            state.average_rewards + self._average_reward_step_size * safe_td_errors
        )
        inputs_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.all(jnp.isfinite(state.average_rewards))
        )
        transaction_applied = (
            inputs_valid
            & result.update_applied
            & jnp.all(jnp.isfinite(proposed_average_rewards))
            & (jnp.any(active) | jnp.all(~requested))
        )
        proposed_state = state.replace(
            learner_state=result.state,
            average_rewards=proposed_average_rewards,
            step_count=_saturating_int32_increment(state.step_count),
        )
        new_state = jax.lax.cond(
            transaction_applied,
            lambda: proposed_state,
            lambda: state,
        )
        head_updates_applied = active & transaction_applied

        def report_per_head(value: Array) -> Array:
            neutral = jnp.zeros_like(value)
            inactive = jnp.full_like(value, jnp.nan)
            return jnp.where(
                active,
                jnp.where(transaction_applied, value, neutral),
                jnp.where(requested, neutral, inactive),
            )

        return AverageRewardHordeUpdateResult(
            state=new_state,
            predictions=jnp.where(
                transaction_applied,
                jnp.where(
                    requested & ~head_updates_applied,
                    jnp.zeros_like(result.predictions),
                    result.predictions,
                ),
                jnp.zeros_like(result.predictions),
            ),
            next_predictions=jnp.where(
                transaction_applied,
                jnp.where(
                    requested & ~head_updates_applied,
                    jnp.zeros_like(next_predictions),
                    next_predictions,
                ),
                jnp.zeros_like(next_predictions),
            ),
            td_errors=report_per_head(td_errors),
            td_targets=report_per_head(targets),
            average_rewards=new_state.average_rewards,
            per_demon_metrics=jnp.stack(
                [report_per_head(result.per_head_metrics[:, i]) for i in range(3)],
                axis=1,
            ),
            head_updates_applied=head_updates_applied,
            update_applied=transaction_applied,
        )


class DifferentialGTDLearner:
    """Linear off-policy differential GTD/TDC prediction.

    This is the average-reward counterpart to the framework's off-policy TD
    primitives.  It maintains a primary value function, a secondary correction
    vector, accumulating importance-weighted traces, and a learned reward-rate
    baseline.  Setting ``rho = 1`` gives an on-policy differential TDC update.

    The secondary weights follow the GTD2/TDC scheme (Sutton et al. 2009;
    Maei 2011): they run an LMS estimate of the expected TD error given the
    features, and the primary update subtracts the resulting
    gradient-correction term so the value weights follow the true (rather
    than semi-) gradient under off-policy sampling.  Note that the config
    defaults run the secondary weights slower than the primary (0.01 vs
    0.05), whereas TDC's two-timescale convergence analysis assumes the
    secondary estimate adapts on the faster timescale.
    """

    def __init__(self, config: DifferentialGTDConfig | None = None):
        """Initialize the learner."""
        self._config = config or DifferentialGTDConfig()

    @property
    def config(self) -> DifferentialGTDConfig:
        """Learner configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize this learner to a dictionary."""
        return {
            "type": "DifferentialGTDLearner",
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DifferentialGTDLearner:
        """Reconstruct a learner from :meth:`to_config` output."""
        config = dict(config)
        config.pop("type", None)
        return cls(DifferentialGTDConfig.from_config(config["config"]))

    def init(
        self,
        feature_dim: int,
        *,
        average_reward: float = 0.0,
    ) -> DifferentialGTDState:
        """Initialize primary weights, secondary weights, and traces."""
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        _preflight_differential_gtd_state(feature_dim)
        average_reward = _validated_float32_scalar_preserving_nonzero(
            "average_reward", average_reward
        )
        return DifferentialGTDState(
            weights=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias=jnp.array(0.0, dtype=jnp.float32),
            secondary_weights=jnp.zeros(feature_dim, dtype=jnp.float32),
            secondary_bias=jnp.array(0.0, dtype=jnp.float32),
            eligibility_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias_eligibility_trace=jnp.array(0.0, dtype=jnp.float32),
            average_reward=jnp.array(average_reward, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: DifferentialGTDState,
        observation: Array,
    ) -> Float[Array, ""]:
        """Compute the scalar differential value estimate."""
        return jnp.dot(state.weights, observation) + state.bias

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: DifferentialGTDState,
        observation: Array,
        reward: Array,
        next_observation: Array,
        rho: Array,
    ) -> DifferentialGTDUpdateResult:
        """Apply one off-policy average-reward GTD/TDC update."""
        cfg = self._config
        alpha = jnp.asarray(cfg.value_step_size, dtype=jnp.float32)
        beta = jnp.asarray(cfg.secondary_step_size, dtype=jnp.float32)
        eta = jnp.asarray(cfg.average_reward_step_size, dtype=jnp.float32)
        lamda = jnp.asarray(cfg.trace_decay, dtype=jnp.float32)
        ratio_clip = jnp.asarray(cfg.ratio_clip, dtype=jnp.float32)

        reward_s = jnp.squeeze(jnp.asarray(reward, dtype=jnp.float32))
        rho_s = jnp.maximum(
            0.0,
            jnp.minimum(jnp.squeeze(jnp.asarray(rho, dtype=jnp.float32)), ratio_clip),
        )
        prediction = self.predict(state, observation)
        next_prediction = self.predict(state, next_observation)
        td_error = reward_s - state.average_reward + next_prediction - prediction

        traces = rho_s * (lamda * state.eligibility_traces + observation)
        bias_trace = rho_s * (lamda * state.bias_eligibility_trace + 1.0)
        secondary_dot_trace = (
            jnp.dot(state.secondary_weights, traces) + state.secondary_bias * bias_trace
        )
        secondary_dot_obs = jnp.dot(state.secondary_weights, observation) + state.secondary_bias

        primary_weight_step = alpha * (
            td_error * traces - (1.0 - lamda) * secondary_dot_trace * next_observation
        )
        primary_bias_step = alpha * (td_error * bias_trace - (1.0 - lamda) * secondary_dot_trace)
        secondary_weight_step = beta * (td_error * traces - secondary_dot_obs * observation)
        secondary_bias_step = beta * (td_error * bias_trace - secondary_dot_obs)
        new_average_reward = state.average_reward + eta * rho_s * td_error

        proposed_state = state.replace(
            weights=state.weights + primary_weight_step,
            bias=state.bias + primary_bias_step,
            secondary_weights=state.secondary_weights + secondary_weight_step,
            secondary_bias=state.secondary_bias + secondary_bias_step,
            eligibility_traces=traces,
            bias_eligibility_trace=bias_trace,
            average_reward=new_average_reward,
            step_count=_saturating_int32_increment(state.step_count),
        )
        # Inf reward makes alpha * delta * z = 0*inf = NaN on silent features
        # and sends rbar to inf. Differential SARSA already refuses that.
        inputs_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.isfinite(reward_s)
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.isfinite(jnp.squeeze(jnp.asarray(rho, dtype=jnp.float32)))
        )
        proposed_finite = (
            jnp.all(jnp.isfinite(proposed_state.weights))
            & jnp.isfinite(proposed_state.bias)
            & jnp.all(jnp.isfinite(proposed_state.secondary_weights))
            & jnp.isfinite(proposed_state.secondary_bias)
            & jnp.all(jnp.isfinite(proposed_state.eligibility_traces))
            & jnp.isfinite(proposed_state.bias_eligibility_trace)
            & jnp.isfinite(proposed_state.average_reward)
        )
        update_applied = inputs_valid & _floating_tree_is_finite(state) & proposed_finite
        new_state = jax.lax.cond(
            update_applied,
            lambda: proposed_state,
            lambda: state,
        )
        metrics = jnp.array(
            [
                td_error**2,
                td_error,
                rho_s,
                new_state.average_reward,
                jnp.mean(jnp.abs(new_state.eligibility_traces)),
                jnp.sqrt(jnp.mean(new_state.secondary_weights**2)),
            ],
            dtype=jnp.float32,
        )
        return DifferentialGTDUpdateResult(
            state=new_state,
            prediction=jnp.where(update_applied, prediction, jnp.zeros_like(prediction)),
            next_prediction=jnp.where(
                update_applied, next_prediction, jnp.zeros_like(next_prediction)
            ),
            td_error=jnp.where(update_applied, td_error, jnp.zeros_like(td_error)),
            rho_clipped=jnp.where(update_applied, rho_s, jnp.zeros_like(rho_s)),
            average_reward=new_state.average_reward,
            metrics=jnp.where(update_applied, metrics, jnp.zeros_like(metrics)),
            update_applied=update_applied,
        )


class DifferentialTDLearner:
    """Linear differential TD(lambda) for continuing prediction.

    For a transition `(S_t, R_{t+1}, S_{t+1})`, the learner forms
    `delta_t = R_{t+1} - rbar_t + v(S_{t+1}) - v(S_t)`, updates the
    reward-rate estimate `rbar_{t+1} = rbar_t + beta * delta_t`, and applies
    the semi-gradient value update along accumulating traces.
    """

    def __init__(self, config: DifferentialTDConfig | None = None):
        """Initialize the learner."""
        self._config = config or DifferentialTDConfig()

    @property
    def config(self) -> DifferentialTDConfig:
        """Learner configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize this learner to a dictionary."""
        return {
            "type": "DifferentialTDLearner",
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DifferentialTDLearner:
        """Reconstruct a learner from :meth:`to_config` output."""
        config = dict(config)
        config.pop("type", None)
        return cls(DifferentialTDConfig.from_config(config["config"]))

    def init(
        self,
        feature_dim: int,
        *,
        average_reward: float = 0.0,
    ) -> DifferentialTDState:
        """Initialize value weights, traces, and reward-rate estimate."""
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        _preflight_differential_td_state(feature_dim)
        _preflight_differential_td_update_working_set(feature_dim)
        average_reward = _validated_float32_scalar_preserving_nonzero(
            "average_reward", average_reward
        )
        return DifferentialTDState(
            weights=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias=jnp.array(0.0, dtype=jnp.float32),
            eligibility_traces=jnp.zeros(feature_dim, dtype=jnp.float32),
            bias_eligibility_trace=jnp.array(0.0, dtype=jnp.float32),
            average_reward=jnp.array(average_reward, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: DifferentialTDState,
        observation: Array,
    ) -> Float[Array, ""]:
        """Compute the scalar differential value estimate."""
        return jnp.dot(state.weights, observation) + state.bias

    @functools.partial(jax.jit, static_argnums=(0,))
    def td_error(
        self,
        state: DifferentialTDState,
        observation: Array,
        reward: Array,
        next_observation: Array,
    ) -> Float[Array, ""]:
        """Compute `R - rbar + v(S') - v(S)` without changing state."""
        reward_s = jnp.squeeze(jnp.asarray(reward, dtype=jnp.float32))
        value = self.predict(state, observation)
        next_value = self.predict(state, next_observation)
        return cast(Array, reward_s - state.average_reward + next_value - value)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: DifferentialTDState,
        observation: Array,
        reward: Array,
        next_observation: Array,
    ) -> DifferentialTDUpdateResult:
        """Apply one continuing differential TD update."""
        cfg = self._config
        alpha = jnp.asarray(cfg.step_size, dtype=jnp.float32)
        beta = jnp.asarray(cfg.average_reward_step_size, dtype=jnp.float32)
        lamda = jnp.asarray(cfg.trace_decay, dtype=jnp.float32)

        reward_s = jnp.squeeze(jnp.asarray(reward, dtype=jnp.float32))
        prediction = self.predict(state, observation)
        next_prediction = self.predict(state, next_observation)
        td_error = reward_s - state.average_reward + next_prediction - prediction

        traces = lamda * state.eligibility_traces + observation
        bias_trace = lamda * state.bias_eligibility_trace + 1.0
        proposed_state = state.replace(
            weights=state.weights + alpha * td_error * traces,
            bias=state.bias + alpha * td_error * bias_trace,
            eligibility_traces=traces,
            bias_eligibility_trace=bias_trace,
            average_reward=state.average_reward + beta * td_error,
            step_count=_saturating_int32_increment(state.step_count),
        )
        # Inf reward makes alpha * delta * z = 0*inf = NaN on silent features
        # and sends rbar to inf. Differential SARSA already refuses that.
        inputs_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.isfinite(reward_s)
            & jnp.all(jnp.isfinite(next_observation))
        )
        proposed_finite = (
            jnp.all(jnp.isfinite(proposed_state.weights))
            & jnp.isfinite(proposed_state.bias)
            & jnp.all(jnp.isfinite(proposed_state.eligibility_traces))
            & jnp.isfinite(proposed_state.bias_eligibility_trace)
            & jnp.isfinite(proposed_state.average_reward)
        )
        update_applied = inputs_valid & _floating_tree_is_finite(state) & proposed_finite
        new_state = jax.lax.cond(
            update_applied,
            lambda: proposed_state,
            lambda: state,
        )
        metrics = jnp.array(
            [
                td_error**2,
                td_error,
                new_state.average_reward,
                jnp.mean(jnp.abs(new_state.eligibility_traces)),
            ],
            dtype=jnp.float32,
        )
        return DifferentialTDUpdateResult(
            state=new_state,
            prediction=jnp.where(update_applied, prediction, jnp.zeros_like(prediction)),
            next_prediction=jnp.where(
                update_applied, next_prediction, jnp.zeros_like(next_prediction)
            ),
            td_error=jnp.where(update_applied, td_error, jnp.zeros_like(td_error)),
            average_reward=new_state.average_reward,
            metrics=jnp.where(update_applied, metrics, jnp.zeros_like(metrics)),
            update_applied=update_applied,
        )


def run_differential_td_from_arrays(
    learner: DifferentialTDLearner,
    state: DifferentialTDState,
    observations: Float[Array, "num_steps feature_dim"],
    rewards: Float[Array, " num_steps"],
    next_observations: Float[Array, "num_steps feature_dim"],
) -> DifferentialTDArrayResult:
    """Run differential TD updates over transition arrays with ``lax.scan``.

    Raises:
        TypeError: If an input is not a JAX array.
        ValueError: If ``observations`` is empty, exceeds the documented
            scan-length ceiling (``_AVERAGE_REWARD_SEQUENCE_MAX_STEPS``), or
            the other step arrays do not share its leading length.
    """
    num_steps = _require_avg_reward_sequence_length("observations", observations)
    _require_avg_reward_matching_length("rewards", rewards, expected=num_steps)
    _require_avg_reward_matching_length(
        "next_observations", next_observations, expected=num_steps
    )
    start = time.time()

    def _scan_fn(
        carry: DifferentialTDState,
        inputs: tuple[Array, Array, Array],
    ) -> tuple[DifferentialTDState, tuple[Array, Array, Array, Array, Array]]:
        obs, reward, next_obs = inputs
        result = learner.update(carry, obs, reward, next_obs)
        return result.state, (
            result.prediction,
            result.td_error,
            result.average_reward,
            result.metrics,
            result.update_applied,
        )

    final_state, (predictions, td_errors, average_rewards, metrics, updates_applied) = jax.lax.scan(
        _scan_fn,
        state,
        (observations, rewards, next_observations),
    )
    elapsed = time.time() - start
    final_state = final_state.replace(uptime_s=final_state.uptime_s + elapsed)
    return DifferentialTDArrayResult(
        state=final_state,
        predictions=predictions,
        td_errors=td_errors,
        average_rewards=average_rewards,
        metrics=metrics,
        updates_applied=updates_applied,
    )


def run_differential_gtd_from_arrays(
    learner: DifferentialGTDLearner,
    state: DifferentialGTDState,
    observations: Float[Array, "num_steps feature_dim"],
    rewards: Float[Array, " num_steps"],
    next_observations: Float[Array, "num_steps feature_dim"],
    rhos: Float[Array, " num_steps"],
) -> DifferentialGTDArrayResult:
    """Run differential GTD/TDC updates over transition arrays with ``lax.scan``.

    Raises:
        TypeError: If an input is not a JAX array.
        ValueError: If ``observations`` is empty, exceeds the documented
            scan-length ceiling (``_AVERAGE_REWARD_SEQUENCE_MAX_STEPS``), or
            the other step arrays do not share its leading length.
    """
    num_steps = _require_avg_reward_sequence_length("observations", observations)
    _require_avg_reward_matching_length("rewards", rewards, expected=num_steps)
    _require_avg_reward_matching_length(
        "next_observations", next_observations, expected=num_steps
    )
    _require_avg_reward_matching_length("rhos", rhos, expected=num_steps)
    start = time.time()

    def _scan_fn(
        carry: DifferentialGTDState,
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[
        DifferentialGTDState,
        tuple[Array, Array, Array, Array, Array, Array],
    ]:
        obs, reward, next_obs, rho = inputs
        result = learner.update(carry, obs, reward, next_obs, rho)
        return result.state, (
            result.prediction,
            result.td_error,
            result.average_reward,
            result.rho_clipped,
            result.metrics,
            result.update_applied,
        )

    (
        final_state,
        (
            predictions,
            td_errors,
            average_rewards,
            rho_clipped,
            metrics,
            updates_applied,
        ),
    ) = jax.lax.scan(
        _scan_fn,
        state,
        (observations, rewards, next_observations, rhos),
    )
    elapsed = time.time() - start
    final_state = final_state.replace(uptime_s=final_state.uptime_s + elapsed)
    return DifferentialGTDArrayResult(
        state=final_state,
        predictions=predictions,
        td_errors=td_errors,
        average_rewards=average_rewards,
        rho_clipped=rho_clipped,
        metrics=metrics,
        updates_applied=updates_applied,
    )


def run_average_reward_horde_from_arrays(
    learner: AverageRewardHordeLearner,
    state: AverageRewardHordeState,
    observations: Float[Array, "num_steps feature_dim"],
    cumulants: Float[Array, "num_steps n_demons"],
    next_observations: Float[Array, "num_steps feature_dim"],
) -> AverageRewardHordeLearningResult:
    """Run a shared-trunk average-reward Horde over transition arrays."""
    observations = jnp.asarray(observations)
    cumulants = jnp.asarray(cumulants)
    next_observations = jnp.asarray(next_observations)
    if observations.ndim != 2:
        raise ValueError(
            f"observations must have shape (num_steps, feature_dim), got {observations.shape}"
        )
    if next_observations.ndim != 2:
        raise ValueError(
            "next_observations must have shape (num_steps, feature_dim), "
            f"got {next_observations.shape}"
        )
    if observations.shape != next_observations.shape:
        raise ValueError(
            "observations and next_observations must have matching shapes, got "
            f"{observations.shape} and {next_observations.shape}"
        )
    expected_cumulant_shape = (observations.shape[0], learner.n_demons)
    if cumulants.shape != expected_cumulant_shape:
        raise ValueError(
            f"cumulants must have shape {expected_cumulant_shape}, got {cumulants.shape}"
        )
    num_steps = observations.shape[0]
    if num_steps < 1 or num_steps > _AVERAGE_REWARD_SEQUENCE_MAX_STEPS:
        raise ValueError(
            "observations length must be an integer in "
            f"[1, {_AVERAGE_REWARD_SEQUENCE_MAX_STEPS}]"
        )
    start = time.time()

    def _scan_fn(
        carry: AverageRewardHordeState,
        inputs: tuple[Array, Array, Array],
    ) -> tuple[AverageRewardHordeState, tuple[Array, Array, Array, Array, Array]]:
        obs, cumulant, next_obs = inputs
        result = learner.update(carry, obs, cumulant, next_obs)
        return result.state, (
            result.td_errors,
            result.average_rewards,
            result.per_demon_metrics,
            result.head_updates_applied,
            result.update_applied,
        )

    (
        final_state,
        (
            td_errors,
            average_rewards,
            per_demon_metrics,
            head_updates_applied,
            updates_applied,
        ),
    ) = jax.lax.scan(
        _scan_fn,
        state,
        (observations, cumulants, next_observations),
    )
    elapsed = time.time() - start
    final_state = final_state.replace(uptime_s=final_state.uptime_s + elapsed)
    return AverageRewardHordeLearningResult(
        state=final_state,
        td_errors=td_errors,
        average_rewards=average_rewards,
        per_demon_metrics=per_demon_metrics,
        head_updates_applied=head_updates_applied,
        updates_applied=updates_applied,
    )


def run_average_reward_horde_actor_critic_from_arrays(
    agent: AverageRewardHordeActorCriticAgent,
    state: AverageRewardHordeActorCriticState,
    rewards: Float[Array, " num_steps"],
    next_observations: Float[Array, "num_steps observation_dim"],
) -> AverageRewardHordeActorCriticArrayResult:
    """Run average-reward Horde actor-critic over transition arrays.

    Raises:
        TypeError: If an input is not a JAX array.
        ValueError: If ``rewards`` is empty, exceeds the documented
            scan-length ceiling (``_AVERAGE_REWARD_SEQUENCE_MAX_STEPS``), or
            ``next_observations`` does not share its leading length.
    """
    num_steps = _require_avg_reward_sequence_length("rewards", rewards)
    _require_avg_reward_matching_length(
        "next_observations", next_observations, expected=num_steps
    )
    start = time.time()

    def _scan_fn(
        carry: AverageRewardHordeActorCriticState,
        inputs: tuple[Array, Array],
    ) -> tuple[
        AverageRewardHordeActorCriticState,
        tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array],
    ]:
        reward, next_observation = inputs
        result = agent.update(carry, reward, next_observation)
        committed_state = jax.lax.cond(
            result.update_applied,
            lambda: result.state,
            lambda: carry,
        )
        return committed_state, (
            result.action,
            result.policy,
            result.target_policy,
            result.behavior_action_probability,
            result.target_action_probability,
            result.actor_score_scale,
            result.td_error,
            result.average_reward,
            result.update_applied,
        )

    (
        final_state,
        (
            actions,
            policies,
            target_policies,
            behavior_action_probabilities,
            target_action_probabilities,
            actor_score_scales,
            td_errors,
            average_rewards,
            updates_applied,
        ),
    ) = jax.lax.scan(_scan_fn, state, (rewards, next_observations))
    elapsed = time.time() - start
    final_state = final_state.replace(uptime_s=final_state.uptime_s + elapsed)
    return AverageRewardHordeActorCriticArrayResult(
        state=final_state,
        actions=actions,
        policies=policies,
        target_policies=target_policies,
        behavior_action_probabilities=behavior_action_probabilities,
        target_action_probabilities=target_action_probabilities,
        actor_score_scales=actor_score_scales,
        td_errors=td_errors,
        average_rewards=average_rewards,
        updates_applied=updates_applied,
    )


@dataclasses.dataclass(frozen=True)
class DifferentialSARSAConfig:
    """Configuration for linear differential SARSA control.

    The reward-rate estimate ``rbar`` learns from the same TD error as the
    action values.  Wan, Naik & Sutton (2021) parameterize its step-size as
    the fraction ``eta`` of the value step-size; the defaults here
    (``average_reward_step_size=0.01`` vs ``q_step_size=0.05``) correspond to
    ``eta = 0.2``, so ``rbar`` tracks more slowly than the values it
    baselines.

    ``use_bias`` controls the per-action bias term.  The bias is an always-on
    shared parameter: in context-gated representations (where per-task weight
    blocks are otherwise mutually exclusive) it becomes the one pathway
    through which learning in one task overwrites another's policy, so
    continual-memory experiments disable it.
    """

    n_actions: int
    q_step_size: float = 0.05
    average_reward_step_size: float = 0.01
    trace_decay: float = 0.0
    epsilon_start: float = 0.1
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 0
    use_bias: bool = True

    def __post_init__(self) -> None:
        """Validate scalar hyperparameters."""
        object.__setattr__(
            self,
            "n_actions",
            _require_int32("n_actions", self.n_actions, minimum=1),
        )
        object.__setattr__(
            self,
            "epsilon_decay_steps",
            _require_int32("epsilon_decay_steps", self.epsilon_decay_steps, minimum=0),
        )
        for name in ("q_step_size", "average_reward_step_size"):
            object.__setattr__(
                self,
                name,
                validated_float32_scalar(name, getattr(self, name), lower=0.0),
            )
        for name in ("trace_decay", "epsilon_start", "epsilon_end"):
            object.__setattr__(
                self,
                name,
                validated_float32_scalar(
                    name,
                    getattr(self, name),
                    lower=0.0,
                    upper=1.0,
                ),
            )
        if type(self.use_bias) is not bool:
            raise ValueError("use_bias must be boolean")
        object.__setattr__(self, "use_bias", bool(self.use_bias))

    def to_config(self) -> dict[str, Any]:
        """Serialize this config to a dictionary."""
        return {
            "type": "DifferentialSARSAConfig",
            "n_actions": self.n_actions,
            "q_step_size": self.q_step_size,
            "average_reward_step_size": self.average_reward_step_size,
            "trace_decay": self.trace_decay,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DifferentialSARSAConfig:
        """Reconstruct a config from :meth:`to_config` output."""
        if type(config) is not dict:
            raise ValueError("differential SARSA config must be an actual dict")
        payload = dict(config)
        expected = {field.name for field in dataclasses.fields(cls)} | {"type"}
        # Historical configs predate the use_bias field and retain the documented
        # compatibility default. No other schema omission is accepted.
        if set(payload) == expected - {"use_bias"}:
            payload["use_bias"] = True
        elif set(payload) != expected:
            raise ValueError("differential SARSA config fields do not match its schema")
        if payload.pop("type") != "DifferentialSARSAConfig":
            raise ValueError("differential SARSA config type is invalid")
        return cls(**payload)


@chex.dataclass(frozen=True)
class DifferentialSARSAState:
    """State for linear differential SARSA."""

    q_weights: Float[Array, "n_actions feature_dim"]
    q_bias: Float[Array, " n_actions"]
    q_trace_weights: Float[Array, "n_actions feature_dim"]
    q_trace_bias: Float[Array, " n_actions"]
    average_reward: Float[Array, ""]
    last_observation: Float[Array, " feature_dim"]
    last_action: Int[Array, ""]
    epsilon: Float[Array, ""]
    rng_key: Array
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class DifferentialSARSAUpdateResult:
    """Result from one differential SARSA transition update."""

    state: DifferentialSARSAState
    action: Int[Array, ""]
    q_values: Float[Array, " n_actions"]
    td_error: Float[Array, ""]
    average_reward: Float[Array, ""]
    reward: Float[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    input_valid: Bool[Array, ""]
    candidate_state_finite: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class DifferentialSARSAArrayResult:
    """Result from scanning differential SARSA over continuing transitions."""

    state: DifferentialSARSAState
    q_values: Float[Array, "num_steps n_actions"]
    td_errors: Float[Array, " num_steps"]
    average_rewards: Float[Array, " num_steps"]
    actions: Int[Array, " num_steps"]
    updates_applied: Bool[Array, " num_steps"]


class DifferentialSARSAAgent:
    """Linear epsilon-greedy differential SARSA control agent.

    The update is the continuing-control counterpart of SARSA:
    `delta_t = R_{t+1} - rbar_t
    + gamma_{t+1} Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)`.
    The same scalar TD error updates the reward-rate estimate and all Q
    parameters through an action-indexed accumulating trace whose prior value
    continues by `gamma_{t+1} lambda`. The compatibility default
    `gamma_{t+1}=1` retains the original continuing-stream behavior.
    """

    def __init__(self, config: DifferentialSARSAConfig):
        """Initialize the control agent."""
        self._config = config

    @property
    def config(self) -> DifferentialSARSAConfig:
        """Agent configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize this agent to a dictionary."""
        return {
            "type": "DifferentialSARSAAgent",
            "state_schema": DIFFERENTIAL_SARSA_STATE_SCHEMA,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DifferentialSARSAAgent:
        """Reconstruct an agent from :meth:`to_config` output."""
        if type(config) is not dict:
            raise ValueError("differential SARSA agent config must be an actual dict")
        payload = dict(config)
        if set(payload) != {"type", "state_schema", "config"}:
            raise ValueError(
                "differential SARSA agent config must contain exactly "
                "type, state_schema, and config"
            )
        if payload["type"] != "DifferentialSARSAAgent":
            raise ValueError("differential SARSA agent config type is invalid")
        if payload["state_schema"] != DIFFERENTIAL_SARSA_STATE_SCHEMA:
            raise ValueError("differential SARSA state schema is unsupported")
        nested = payload["config"]
        if type(nested) is not dict:
            raise ValueError("differential SARSA nested config must be an actual dict")
        return cls(DifferentialSARSAConfig.from_config(nested))

    def init(
        self,
        feature_dim: int,
        key: Array,
        *,
        average_reward: float = 0.0,
    ) -> DifferentialSARSAState:
        """Initialize Q weights, traces, reward-rate estimate, and RNG."""
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        _preflight_differential_sarsa_state(self._config.n_actions, feature_dim)
        average_reward = validated_float32_scalar("average_reward", average_reward)
        cfg = self._config
        zeros_q = jnp.zeros((cfg.n_actions, feature_dim), dtype=jnp.float32)
        zeros_bias = jnp.zeros((cfg.n_actions,), dtype=jnp.float32)
        return DifferentialSARSAState(
            q_weights=zeros_q,
            q_bias=zeros_bias,
            q_trace_weights=zeros_q,
            q_trace_bias=zeros_bias,
            average_reward=jnp.array(average_reward, dtype=jnp.float32),
            last_observation=jnp.zeros(feature_dim, dtype=jnp.float32),
            last_action=jnp.array(-1, dtype=jnp.int32),
            epsilon=jnp.array(cfg.epsilon_start, dtype=jnp.float32),
            rng_key=key,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    def _require_state_contract(self, state: DifferentialSARSAState) -> int:
        """Validate fixed shapes/dtypes and return the deployed feature width."""

        q_weights = jnp.asarray(state.q_weights)
        if q_weights.ndim != 2 or q_weights.shape[0] != self._config.n_actions:
            raise ValueError(
                "differential SARSA q_weights must have shape (n_actions, feature_dim)"
            )
        feature_dim = q_weights.shape[1]
        if feature_dim < 1:
            raise ValueError("differential SARSA feature_dim must be positive")
        float_contracts = (
            ("q_weights", state.q_weights, (self._config.n_actions, feature_dim)),
            ("q_bias", state.q_bias, (self._config.n_actions,)),
            (
                "q_trace_weights",
                state.q_trace_weights,
                (self._config.n_actions, feature_dim),
            ),
            ("q_trace_bias", state.q_trace_bias, (self._config.n_actions,)),
            ("average_reward", state.average_reward, ()),
            ("last_observation", state.last_observation, (feature_dim,)),
            ("epsilon", state.epsilon, ()),
        )
        for name, value, shape in float_contracts:
            array = jnp.asarray(value)
            if array.shape != shape:
                raise ValueError(f"differential SARSA {name} must have shape {shape}")
            if array.dtype != jnp.dtype(jnp.float32):
                raise TypeError(f"differential SARSA {name} must have dtype float32")
        for name, value in (
            ("last_action", state.last_action),
            ("step_count", state.step_count),
        ):
            array = jnp.asarray(value)
            if array.shape != ():
                raise ValueError(f"differential SARSA {name} must be scalar")
            if array.dtype != jnp.dtype(jnp.int32):
                raise TypeError(f"differential SARSA {name} must have dtype int32")
        _checked_lifetime_words_increment(state.step_words)
        key_words = jnp.asarray(jr.key_data(state.rng_key))
        if key_words.shape != (2,) or key_words.dtype != jnp.dtype(jnp.uint32):
            raise TypeError("differential SARSA rng_key must be a scalar PRNG key")
        return feature_dim

    def _state_values_valid(self, state: DifferentialSARSAState) -> Bool[Array, ""]:
        """Return dynamic validity excluding the separately reported clock."""

        return (
            jnp.all(jnp.isfinite(state.q_weights))
            & jnp.all(jnp.isfinite(state.q_bias))
            & jnp.all(jnp.isfinite(state.q_trace_weights))
            & jnp.all(jnp.isfinite(state.q_trace_bias))
            & jnp.isfinite(state.average_reward)
            & jnp.all(jnp.isfinite(state.last_observation))
            & jnp.isfinite(state.epsilon)
            & (state.epsilon >= 0.0)
            & (state.epsilon <= 1.0)
            & (state.last_action >= 0)
            & (state.last_action < self._config.n_actions)
            & jnp.isfinite(jnp.asarray(state.birth_timestamp, dtype=jnp.float32))
            & jnp.isfinite(jnp.asarray(state.uptime_s, dtype=jnp.float32))
            & (jnp.asarray(state.uptime_s, dtype=jnp.float32) >= 0.0)
        )

    @staticmethod
    def _candidate_finite(state: DifferentialSARSAState) -> Bool[Array, ""]:
        """Require every floating candidate leaf to be finite."""

        checks: list[Array] = []
        for leaf in jax.tree.leaves(state):
            value = jnp.asarray(leaf)
            if jnp.issubdtype(value.dtype, jnp.inexact):
                checks.append(jnp.all(jnp.isfinite(value)))
        return jnp.all(jnp.stack(checks)) if checks else jnp.asarray(True, dtype=jnp.bool_)

    @functools.partial(jax.jit, static_argnums=(0,))
    def q_values(
        self,
        state: DifferentialSARSAState,
        observation: Array,
    ) -> Float[Array, " n_actions"]:
        """Compute all action values for one observation."""
        return state.q_weights @ observation + state.q_bias

    @functools.partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        state: DifferentialSARSAState,
        observation: Array,
    ) -> tuple[Int[Array, ""], Array]:
        """Select an epsilon-greedy action with uniform tie-breaking."""
        key, explore_key, noise_key, random_key = jr.split(state.rng_key, 4)
        q_values = self.q_values(state, observation)
        maximum = jnp.max(q_values)
        tie_noise = jnp.where(
            q_values == maximum,
            jr.gumbel(noise_key, shape=q_values.shape),
            -jnp.inf,
        )
        greedy_action = jnp.argmax(tie_noise).astype(jnp.int32)
        random_action = jr.randint(random_key, (), 0, self._config.n_actions).astype(jnp.int32)
        explore = jr.uniform(explore_key) < state.epsilon
        return jax.lax.select(explore, random_action, greedy_action), key

    @functools.partial(jax.jit, static_argnums=(0,))
    def start(
        self,
        state: DifferentialSARSAState,
        observation: Array,
    ) -> tuple[DifferentialSARSAState, Int[Array, ""]]:
        """Select and store the first action for a continuing stream."""
        action, key = self.select_action(state, observation)
        return state.replace(
            last_observation=observation,
            last_action=action,
            rng_key=key,
        ), action

    @functools.partial(jax.jit, static_argnums=(0,))
    def start_with_action(
        self,
        state: DifferentialSARSAState,
        observation: Array,
        action: Array,
    ) -> tuple[DifferentialSARSAState, Int[Array, ""]]:
        """Store an externally scored first action without sampling internally.

        This is the causal start surface for planners that combine the current
        Q-values with pre-action model scores and own the exploration RNG.
        It does not advance the agent RNG, update parameters, increment the
        continuing step counter, or reinterpret the supplied action.

        The caller is responsible for supplying a scalar action ID in
        ``[0, n_actions)``; integrated protocols must validate that invariant
        in their own fail-closed diagnostics.
        """
        observation_f = jnp.asarray(observation, dtype=jnp.float32)
        action_i = jnp.asarray(action)
        if action_i.shape != ():
            raise ValueError("differential SARSA action must be scalar")
        if action_i.dtype != jnp.dtype(jnp.int32):
            raise TypeError("differential SARSA action must have dtype int32")
        return (
            state.replace(
                last_observation=observation_f,
                last_action=action_i,
            ),
            action_i,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def td_error(
        self,
        state: DifferentialSARSAState,
        reward: Array,
        next_observation: Array,
        next_action: Array,
        *,
        discount: Array | float = 1.0,
    ) -> Float[Array, ""]:
        """Compute a discounted differential SARSA TD error without updating."""
        reward_s = jnp.squeeze(jnp.asarray(reward, dtype=jnp.float32))
        discount_s = jnp.squeeze(jnp.asarray(discount, dtype=jnp.float32))
        q_prev = self.q_values(state, state.last_observation)[state.last_action]
        q_next = self.q_values(state, next_observation)[next_action]
        return cast(
            Array,
            reward_s - state.average_reward + discount_s * q_next - q_prev,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: DifferentialSARSAState,
        reward: Array,
        next_observation: Array,
        next_action: Array | None = None,
        *,
        discount: Array | float = 1.0,
    ) -> DifferentialSARSAUpdateResult:
        """Apply one differential SARSA update.

        Args:
            state: Current state with a valid previous observation/action.
            reward: Reward received after taking ``state.last_action``.
            next_observation: New observation.
            next_action: Optional preselected next action. When omitted, the
                current policy samples one and stores it for the following step.
            discount: Continuation discount for this transition. The default
                ``1.0`` preserves the original continuing-task behavior.
        """
        cfg = self._config
        feature_dim = self._require_state_contract(state)
        raw_next_observation = jnp.asarray(next_observation)
        if raw_next_observation.shape != (feature_dim,):
            raise ValueError(
                f"differential SARSA next_observation must have shape ({feature_dim},)"
            )
        if not (
            jnp.issubdtype(raw_next_observation.dtype, jnp.floating)
            or jnp.issubdtype(raw_next_observation.dtype, jnp.integer)
        ):
            raise TypeError("differential SARSA next_observation must be real numeric")
        next_observation_f = raw_next_observation.astype(jnp.float32)
        raw_reward = jnp.squeeze(jnp.asarray(reward))
        if raw_reward.shape != ():
            raise ValueError("differential SARSA reward must squeeze to a scalar")
        if not (
            jnp.issubdtype(raw_reward.dtype, jnp.floating)
            or jnp.issubdtype(raw_reward.dtype, jnp.integer)
        ):
            raise TypeError("differential SARSA reward must be real numeric")
        reward_s = raw_reward.astype(jnp.float32)
        raw_discount = jnp.squeeze(jnp.asarray(discount))
        if raw_discount.shape != ():
            raise ValueError("differential SARSA discount must squeeze to a scalar")
        if not (
            jnp.issubdtype(raw_discount.dtype, jnp.floating)
            or jnp.issubdtype(raw_discount.dtype, jnp.integer)
        ):
            raise TypeError("differential SARSA discount must be real numeric")
        discount_s = raw_discount.astype(jnp.float32)
        proposed_step_words, lifetime_capacity_available = _checked_lifetime_words_increment(
            state.step_words
        )
        lifetime_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        if next_action is None:
            selected_next_action, key = self.select_action(state, next_observation_f)
            action_contract_valid = jnp.asarray(True, dtype=jnp.bool_)
        else:
            raw_next_action = jnp.asarray(next_action)
            if raw_next_action.shape != ():
                raise ValueError("differential SARSA next_action must be scalar")
            if raw_next_action.dtype != jnp.dtype(jnp.int32):
                raise TypeError("differential SARSA next_action must have dtype int32")
            selected_next_action = raw_next_action
            key = state.rng_key
            action_contract_valid = (selected_next_action >= 0) & (
                selected_next_action < cfg.n_actions
            )

        alpha = jnp.asarray(cfg.q_step_size, dtype=jnp.float32)
        beta = jnp.asarray(cfg.average_reward_step_size, dtype=jnp.float32)
        lamda = jnp.asarray(cfg.trace_decay, dtype=jnp.float32)
        state_values_valid = self._state_values_valid(state)
        state_valid = lifetime_counter_valid & state_values_valid
        input_valid = (
            jnp.isfinite(reward_s)
            & jnp.all(jnp.isfinite(next_observation_f))
            & jnp.isfinite(discount_s)
            & (discount_s >= 0.0)
            & (discount_s <= 1.0)
            & action_contract_valid
        )

        q_prev_all = self.q_values(state, state.last_observation)
        q_next_all = self.q_values(state, next_observation_f)
        q_prev = q_prev_all[state.last_action]
        q_next = q_next_all[selected_next_action]
        td_error = reward_s - state.average_reward + discount_s * q_next - q_prev

        action_mask = jax.nn.one_hot(
            state.last_action,
            cfg.n_actions,
            dtype=jnp.float32,
        )
        grad_weights = action_mask[:, None] * state.last_observation[None, :]
        # With use_bias=False the bias gradient is zeroed, so q_bias stays at
        # its zero init and the Q-function is purely feature-driven.
        grad_bias = action_mask * jnp.float32(cfg.use_bias)
        trace_continuation = discount_s * lamda
        traces = trace_continuation * state.q_trace_weights + grad_weights
        bias_traces = trace_continuation * state.q_trace_bias + grad_bias
        new_step_count = _saturating_int32_increment(state.step_count)
        new_epsilon = jax.lax.cond(
            cfg.epsilon_decay_steps > 0,
            lambda: jnp.maximum(
                cfg.epsilon_end,
                cfg.epsilon_start
                - (cfg.epsilon_start - cfg.epsilon_end) * new_step_count / cfg.epsilon_decay_steps,
            ),
            lambda: state.epsilon,
        )
        new_average_reward = state.average_reward + beta * td_error
        proposed_state = state.replace(
            q_weights=state.q_weights + alpha * td_error * traces,
            q_bias=state.q_bias + alpha * td_error * bias_traces,
            q_trace_weights=traces,
            q_trace_bias=bias_traces,
            average_reward=new_average_reward,
            last_observation=next_observation_f,
            last_action=selected_next_action,
            epsilon=new_epsilon,
            rng_key=key,
            step_count=new_step_count,
            step_words=proposed_step_words,
        )
        candidate_state_finite = self._candidate_finite(proposed_state)
        update_available = (
            state_valid & input_valid & lifetime_capacity_available & candidate_state_finite
        )
        new_state = jax.lax.cond(
            update_available,
            lambda _: proposed_state,
            lambda _: state,
            operand=None,
        )
        return DifferentialSARSAUpdateResult(
            state=new_state,
            action=jnp.where(update_available, selected_next_action, state.last_action),
            q_values=jnp.where(update_available, q_next_all, jnp.zeros_like(q_next_all)),
            td_error=jnp.where(
                update_available,
                td_error,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            average_reward=new_state.average_reward,
            reward=jnp.where(
                update_available,
                reward_s,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            pre_step_words=state.step_words,
            post_step_words=new_state.step_words,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            state_valid=state_valid,
            input_valid=input_valid,
            candidate_state_finite=candidate_state_finite,
            update_applied=update_available,
        )


def run_differential_sarsa_from_arrays(
    agent: DifferentialSARSAAgent,
    state: DifferentialSARSAState,
    rewards: Float[Array, " num_steps"],
    next_observations: Float[Array, "num_steps feature_dim"],
    discounts: Float[Array, " num_steps"] | None = None,
) -> DifferentialSARSAArrayResult:
    """Run differential SARSA over continuing transition arrays.

    The input state must already be primed with ``agent.start`` so it contains
    the first observation and action. Rewards are therefore aligned with
    transitions from the stored `(S_t, A_t)` to each ``next_observations[t]``.
    When ``discounts`` is omitted, every transition uses ``1.0`` for backward
    compatibility with the original continuing-task runner.

    Raises:
        TypeError: If an input is not a JAX array.
        ValueError: If ``rewards`` is empty, exceeds the documented
            scan-length ceiling (``_AVERAGE_REWARD_SEQUENCE_MAX_STEPS``), or
            ``next_observations``/``discounts`` do not share its leading
            length.
    """
    num_steps = _require_avg_reward_sequence_length("rewards", rewards)
    _require_avg_reward_matching_length(
        "next_observations", next_observations, expected=num_steps
    )
    if discounts is not None:
        _require_avg_reward_matching_length("discounts", discounts, expected=num_steps)
    start = time.time()
    if discounts is None:
        discounts = jnp.ones_like(rewards, dtype=jnp.float32)

    def _scan_fn(
        carry: DifferentialSARSAState,
        inputs: tuple[Array, Array, Array],
    ) -> tuple[DifferentialSARSAState, tuple[Array, Array, Array, Array, Array]]:
        reward, next_observation, discount = inputs
        result = agent.update(
            carry,
            reward,
            next_observation,
            discount=discount,
        )
        return result.state, (
            result.q_values,
            result.td_error,
            result.average_reward,
            result.action,
            result.update_applied,
        )

    (
        final_state,
        (
            q_values,
            td_errors,
            average_rewards,
            actions,
            updates_applied,
        ),
    ) = jax.lax.scan(
        _scan_fn,
        state,
        (rewards, next_observations, discounts),
    )
    elapsed = time.time() - start
    if bool(jax.device_get(jnp.any(updates_applied))):
        final_state = final_state.replace(uptime_s=final_state.uptime_s + elapsed)
    return DifferentialSARSAArrayResult(
        state=final_state,
        q_values=q_values,
        td_errors=td_errors,
        average_rewards=average_rewards,
        actions=actions,
        updates_applied=updates_applied,
    )


def differential_sarsa_lifetime_counter_nbytes() -> int:
    """Return bytes occupied by compatibility telemetry plus exact identity."""

    return DIFFERENTIAL_SARSA_LIFETIME_COUNTER_NBYTES


def measure_differential_sarsa_state_nbytes(state: DifferentialSARSAState) -> int:
    """Measure persistent JAX-array bytes in one differential SARSA state."""

    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(state)
        if isinstance(leaf, Array)
    )


def migrate_legacy_differential_sarsa_state(
    legacy_state: Any,
) -> DifferentialSARSAState:
    """Migrate a pre-v2 state only when its int32 clock is unambiguous."""

    if isinstance(legacy_state, Mapping):
        fields = dict(legacy_state)
    elif dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        fields = {
            field.name: getattr(legacy_state, field.name)
            for field in dataclasses.fields(legacy_state)
        }
    else:
        raise TypeError("legacy differential SARSA state must be a mapping or dataclass")
    current_names = {
        field.name
        for field in dataclasses.fields(DifferentialSARSAState)  # type: ignore[arg-type]
    }
    legacy_names = current_names - {"step_words"}
    supplied_names = set(fields)
    if supplied_names != legacy_names:
        missing = sorted(legacy_names - supplied_names)
        extra = sorted(supplied_names - legacy_names)
        raise ValueError(
            "legacy differential SARSA field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )
    step_count = jnp.asarray(fields["step_count"])
    if step_count.shape != () or step_count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy differential SARSA step_count must be scalar int32")
    step = int(step_count)
    if step < 0:
        raise ValueError("negative legacy differential SARSA step_count indicates wrap")
    if step >= _INT32_MAX:
        raise ValueError("saturated legacy differential SARSA step_count is ambiguous")
    fields["step_words"] = jnp.asarray((0, step), dtype=jnp.uint32)
    return DifferentialSARSAState(**fields)
