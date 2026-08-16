# mypy: disable-error-code="attr-defined,call-arg"
"""Core types and algorithms for the STOMP progression (Alberta Plan Step 10).

The STOMP progression (SubTasks, Options, Models, Planning) introduces
temporally extended actions (options) to the continuing agent architecture.
Each option is defined by a subtask — a feature-reaching sub-problem with its
own pseudo-reward and termination condition.  Solving each subtask produces an
intra-option policy (an option).  Online experience with each option trains a
multi-step outcome model.  The top-level agent can then plan with option models
the same way it plans with one-step environment models.

This module provides JAX-compatible, scan-friendly implementations of all four
STOMP components.  All shapes are statically fixed so that JIT compilation and
``jax.lax.scan`` work without recompilation per step.

References:
    Sutton, Bowling, & Pilarski (2022). "The Alberta Plan for AI Research."
    Sutton, Precup, & Singh (1999). "Between MDPs and semi-MDPs: A Framework
        for Temporal Abstraction in Reinforcement Learning." AIJ.
    Precup (2000). "Temporal Abstraction in Reinforcement Learning." PhD thesis.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
from collections.abc import Mapping
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPLearner,
    MultiHeadMLPState,
    migrate_legacy_multi_head_mlp_state,
)
from alberta_framework.core.normalizers import (
    _checked_lifetime_words_increment,
    _lifetime_counter_valid,
    _saturating_int32_counter_increment,
)
from alberta_framework.core.types import LMSState

STOMP_STATE_SCHEMA = "alberta.stomp-state.v2"
STOMP_LIFETIME_COUNTER_NBYTES = 12
STOMP_LIFETIME_COUNTER_DELTA_NBYTES = 8

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
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


def _validate_hidden_sizes(sizes: object) -> tuple[int, ...]:
    if type(sizes) is not tuple:
        raise ValueError("base_hidden_sizes must be an actual tuple")
    return tuple(_require_int32("base_hidden_sizes element", s, minimum=1) for s in sizes)


def _stomp_direct_array_scalars(config: STOMPConfig) -> int:
    """Return exact directly allocated agent/config and state array scalars."""

    n_options = config.n_options
    n_actions = config.n_primitive_actions
    observation_dim = config.observation_dim
    n_heads = config.n_total_actions
    layer_sizes = (observation_dim, *config.base_hidden_sizes)
    trunk_parameters = sum(
        fan_out * (fan_in + 1)
        for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:], strict=False)
    )
    final_width = config.base_hidden_sizes[-1] if config.base_hidden_sizes else observation_dim
    head_parameters = n_heads * (final_width + 1)
    base_learner_scalars = (
        2 * (trunk_parameters + head_parameters)
        + sum(config.base_hidden_sizes)
        + 2 * len(config.base_hidden_sizes)
        + 2 * n_heads
        + 3
    )
    return (
        base_learner_scalars
        + 2 * n_options * n_actions * observation_dim
        + n_options * observation_dim * observation_dim
        + 11 * n_options
        + 2 * observation_dim
        + 15
    )


# ---------------------------------------------------------------------------
# Subtask specification (Python-level; JAX arrays extracted for scan use)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SubtaskSpec:
    """Defines one subtask as a linear feature-reaching pseudo-reward.

    The pseudo-reward is ``pseudo_reward_scale * observation[feature_index]``.
    The option terminates when the pseudo-reward reaches ``threshold`` or when
    ``max_option_steps`` primitive actions have been executed.

    Args:
        feature_index: Index of the observation feature the option drives toward.
        threshold: Pseudo-reward value at which the option is considered
            complete.  Must be finite and positive; choose relative to the
            feature scale.
        pseudo_reward_scale: Multiplicative scale for the pseudo-reward signal.
            Must be finite and positive.
        max_option_steps: Hard cap on option duration to prevent infinite loops.
    """

    feature_index: int
    threshold: float = 0.5
    pseudo_reward_scale: float = 1.0
    max_option_steps: int = 8

    def __post_init__(self) -> None:
        """Validate subtask specification."""
        object.__setattr__(
            self,
            "feature_index",
            _require_int32("feature_index", self.feature_index, minimum=0),
        )
        object.__setattr__(
            self,
            "max_option_steps",
            _require_int32("max_option_steps", self.max_option_steps, minimum=1),
        )
        object.__setattr__(
            self,
            "threshold",
            validated_float32_scalar("threshold", self.threshold, positive=True),
        )
        object.__setattr__(
            self,
            "pseudo_reward_scale",
            validated_float32_scalar(
                "pseudo_reward_scale", self.pseudo_reward_scale, positive=True
            ),
        )


@dataclasses.dataclass(frozen=True)
class STOMPSpecArrays:
    """JAX arrays extracted from a list of :class:`SubtaskSpec` for scan use.

    All arrays have shape ``(n_options,)`` or a compatible leading dimension.
    """

    feature_indices: Int[Array, " n_options"]
    thresholds: Float[Array, " n_options"]
    pseudo_reward_scales: Float[Array, " n_options"]
    max_option_steps: Int[Array, " n_options"]

    @staticmethod
    def from_specs(specs: list[SubtaskSpec]) -> STOMPSpecArrays:
        """Build JAX arrays from a list of :class:`SubtaskSpec`."""
        return STOMPSpecArrays(
            feature_indices=jnp.array([s.feature_index for s in specs], dtype=jnp.int32),
            thresholds=jnp.array([s.threshold for s in specs], dtype=jnp.float32),
            pseudo_reward_scales=jnp.array(
                [s.pseudo_reward_scale for s in specs], dtype=jnp.float32
            ),
            max_option_steps=jnp.array([s.max_option_steps for s in specs], dtype=jnp.int32),
        )

    def to_list(self) -> list[SubtaskSpec]:
        """Recover a list of :class:`SubtaskSpec` from this array collection."""
        n = int(self.feature_indices.shape[0])
        return [
            SubtaskSpec(
                feature_index=int(self.feature_indices[i]),
                threshold=float(self.thresholds[i]),
                pseudo_reward_scale=float(self.pseudo_reward_scales[i]),
                max_option_steps=int(self.max_option_steps[i]),
            )
            for i in range(n)
        ]


# ---------------------------------------------------------------------------
# Intra-option policy state (batched over options)
# ---------------------------------------------------------------------------


@chex.dataclass(frozen=True)
class IntraOptionPoliciesState:
    """Linear differential Q-policies for all options.

    Weights are stored batched over options so that a single indexed update
    can be expressed as a masked scatter inside ``jax.lax.scan``.

    Attributes:
        q_weights: Shape ``(n_options, n_primitive_actions, observation_dim)``.
        traces: Accumulating eligibility traces; same shape as ``q_weights``.
        average_rewards: Per-option differential reward rates; shape
            ``(n_options,)``.
    """

    q_weights: Float[Array, "n_options n_actions obs_dim"]
    traces: Float[Array, "n_options n_actions obs_dim"]
    average_rewards: Float[Array, " n_options"]


# ---------------------------------------------------------------------------
# Option outcome model state (batched over options)
# ---------------------------------------------------------------------------


@chex.dataclass(frozen=True)
class OptionModelsState:
    """Online outcome models for all options.

    Each option model represents the expected multi-step return of executing
    the option from a state, decomposed into:

    * expected cumulative pseudo-reward (EMA over completed option runs),
    * expected discounted environment return,
    * expected primitive-step duration,
    * expected discounted baseline mass ``Σ_{k=0}^{T-1} γ^k``,
    * expected option discount ``γ^T`` where ``T`` is the option duration,
    * a linear predictor for the expected next-state delta.

    Checkpoint schema note:
        These fields are part of the JAX PyTree. Generic Orbax restores require
        an exact template match; use :func:`load_stomp_state_with_migration`
        to load checkpoints written before the environment-return, duration,
        and baseline-mass fields were added.

    Attributes:
        cumreward_ema: Shape ``(n_options,)``.  EMA of observed cumulative
            pseudo-reward per option execution.
        env_return_ema: Shape ``(n_options,)``. EMA of the discounted real
            environment return observed per option execution.
        duration_ema: Shape ``(n_options,)``. EMA of primitive-step duration.
        baseline_mass_ema: Shape ``(n_options,)``. EMA of the discounted
            baseline mass ``Σ_{k=0}^{T-1} γ^k``. This is the coefficient on
            the average-reward baseline in the discounted differential
            semi-MDP target and equals duration when ``γ = 1``.
        discount_ema: Shape ``(n_options,)``.  EMA of ``γ^T`` observed at
            option termination.
        next_state_weights: Shape ``(n_options, obs_dim, obs_dim)``.  Linear
            weights predicting ``Δobs = next_obs - start_obs`` from ``start_obs``.
        n_completions: Shape ``(n_options,)`` int32.  Number of times each
            option has successfully terminated.
    """

    cumreward_ema: Float[Array, " n_options"]
    env_return_ema: Float[Array, " n_options"]
    duration_ema: Float[Array, " n_options"]
    baseline_mass_ema: Float[Array, " n_options"]
    discount_ema: Float[Array, " n_options"]
    next_state_weights: Float[Array, "n_options obs_dim obs_dim"]
    n_completions: Int[Array, " n_options"]


# ---------------------------------------------------------------------------
# Full STOMP agent state
# ---------------------------------------------------------------------------


@chex.dataclass(frozen=True)
class STOMPState:
    """Combined state for the full STOMP agent.

    The base control is a differential Q-function over the *extended*
    action set ``{a_0, …, a_{K-1}, o_0, …, o_{N-1}}`` where K is the number
    of primitive actions and N is the number of options.  When
    ``STOMPConfig.base_hidden_sizes`` is empty the base Q is linear (one head
    per extended action); when non-empty it is a shared-trunk MLP.

    Attributes:
        base_learner_state: Extended Q-function state (MultiHeadMLPLearner
            with ``n_heads = K + N``).
        base_average_reward: Scalar continuing reward rate for base agent.
        base_last_obs: Most recent observation seen by the base agent.
        base_last_action: Last extended action index taken (0..K+N-1).
        last_primitive_action: Primitive action most recently selected for
            dispatch to the environment.
        rng_key: JAX PRNG key.
        option_policies: Batched intra-option policies.
        option_models: Batched option outcome models.
        executing_option: Scalar int32; −1 means no option is executing.
        option_start_obs: Observation at the start of the current option.
        option_last_intra_action: Primitive action taken on the previous
            intra-option step (for option Q-update).
        option_cumreward: Accumulated pseudo-reward in the current option.
        option_env_cumreward: Discounted environment return accumulated in
            the current option; grounds the base extended-Q update in task
            reward on option termination.
        option_baseline_mass: Discounted baseline mass
            ``Σ_{k=0}^{T-1} γ^k`` accumulated in the current option.
        option_discount: Accumulated discount ``∏ γ`` in current option.
        option_steps: Number of primitive steps taken inside current option.
        step_count: Saturating int32 primitive-step telemetry.
        step_words: Exact big-endian ``[high, low]`` uint32 primitive-step
            identity. This is the scheduling and lifetime authority.
    """

    base_learner_state: MultiHeadMLPState
    base_average_reward: Float[Array, ""]
    base_last_obs: Float[Array, " obs_dim"]
    base_last_action: Int[Array, ""]
    last_primitive_action: Int[Array, ""]
    rng_key: Array
    option_policies: IntraOptionPoliciesState
    option_models: OptionModelsState
    executing_option: Int[Array, ""]
    option_start_obs: Float[Array, " obs_dim"]
    option_last_intra_action: Int[Array, ""]
    option_cumreward: Float[Array, ""]
    option_env_cumreward: Float[Array, ""]
    option_baseline_mass: Float[Array, ""]
    option_discount: Float[Array, ""]
    option_steps: Int[Array, ""]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


DISPATCH_OWNER_INVALID = -1
DISPATCH_OWNER_BASE_PRIMITIVE = 0
DISPATCH_OWNER_OPTION = 1


@chex.dataclass(frozen=True)
class DispatchedPrimitiveActionDecision:
    """Fail-closed ownership and safety audit for one dispatch replacement."""

    owner: Int[Array, ""]
    state_static_contract_valid: Bool[Array, ""]
    state_values_finite: Bool[Array, ""]
    state_counters_valid: Bool[Array, ""]
    rng_key_valid: Bool[Array, ""]
    ownership_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    observation_static_contract_valid: Bool[Array, ""]
    observation_valid: Bool[Array, ""]
    observation_matches: Bool[Array, ""]
    proposed_action_static_contract_valid: Bool[Array, ""]
    proposed_action_valid: Bool[Array, ""]
    safety_action_mask_static_contract_valid: Bool[Array, ""]
    counterfactual_action_safe: Bool[Array, ""]
    proposed_action_safe: Bool[Array, ""]
    counterfactual_action: Int[Array, ""]
    proposed_action: Int[Array, ""]
    effective_action: Int[Array, ""]
    used_safe_base_fallback: Bool[Array, ""]
    applied: Bool[Array, ""]
    failed_closed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class DispatchedPrimitiveActionReplacementResult:
    """Potentially replaced STOMP dispatch state plus its exact audit."""

    state: STOMPState
    decision: DispatchedPrimitiveActionDecision


def _all_floating_leaves_finite(tree: Any) -> Array:
    """Bounded full-tree finiteness check, excluding typed PRNG keys."""
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "dtype") and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            continue
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(array))
    return valid


def _all_integer_leaves_nonnegative(tree: Any) -> Array:
    """Check learner/model counter leaves without constraining float values."""
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "dtype") and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            continue
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.integer):
            valid = valid & jnp.all(array >= 0)
    return valid


def _checked_lifetime_words_advance(
    words: Array,
    delta: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Advance one uint64 word clock by a non-negative uint32 delta."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("lifetime counter words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("lifetime counter words must have dtype uint32")
    increment = jnp.asarray(delta, dtype=jnp.uint32).reshape(())
    low = words[1] + increment
    carry = (low < words[1]).astype(jnp.uint32)
    high = words[0] + carry
    overflow = (carry != 0) & (high == jnp.asarray(0, dtype=jnp.uint32))
    candidate = jnp.stack((high, low)).astype(jnp.uint32)
    return jnp.where(overflow, words, candidate), ~overflow


def _lifetime_words_mod(words: Array, divisor: Array) -> UInt[Array, ""]:
    """Return an exact uint64 word-pair remainder without enabling JAX x64."""

    modulus = jnp.asarray(divisor, dtype=jnp.uint32).reshape(())

    def fold_word(remainder: Array, word: Array) -> Array:
        def fold_bit(bit_index: int, current: Array) -> Array:
            shift = jnp.asarray(31 - bit_index, dtype=jnp.uint32)
            bit = (word >> shift) & jnp.asarray(1, dtype=jnp.uint32)
            doubled = current + current + bit
            return jnp.where(doubled >= modulus, doubled - modulus, doubled)

        return jax.lax.fori_loop(0, 32, fold_bit, remainder)

    return fold_word(
        fold_word(jnp.asarray(0, dtype=jnp.uint32), words[0]),
        words[1],
    )


def _lifetime_words_at_least(words: Array, threshold: int) -> Bool[Array, ""]:
    """Compare a word clock with one validated host uint64 threshold."""

    threshold_high = jnp.asarray(threshold >> 32, dtype=jnp.uint32)
    threshold_low = jnp.asarray(threshold & _UINT32_MAX, dtype=jnp.uint32)
    return (words[0] > threshold_high) | (
        (words[0] == threshold_high) & (words[1] >= threshold_low)
    )


def _stomp_action_ownership_valid(
    state: STOMPState,
    *,
    n_options: int,
    n_primitive_actions: int,
) -> Bool[Array, ""]:
    """Authenticate the extended-action owner of the dispatched primitive."""

    primitive_valid = (state.last_primitive_action >= 0) & (
        state.last_primitive_action < n_primitive_actions
    )
    idle = state.executing_option == -1
    active = (state.executing_option >= 0) & (state.executing_option < n_options)
    idle_valid = (
        idle
        & (state.base_last_action == state.last_primitive_action)
        & (state.base_last_action >= 0)
        & (state.base_last_action < n_primitive_actions)
    )
    active_valid = (
        active
        & (state.base_last_action == n_primitive_actions + state.executing_option)
        & (state.option_last_intra_action == state.last_primitive_action)
    )
    return primitive_valid & (idle_valid | active_valid)


def _stomp_static_dispatch_contract(
    state: STOMPState,
    *,
    n_options: int,
    n_primitive_actions: int,
    observation_dim: int,
) -> tuple[bool, bool]:
    """Return fixed shape/dtype and typed-key contract flags."""

    float32_shapes = (
        (state.base_last_obs, (observation_dim,)),
        (state.option_start_obs, (observation_dim,)),
        (state.option_policies.q_weights, (n_options, n_primitive_actions, observation_dim)),
        (state.option_policies.traces, (n_options, n_primitive_actions, observation_dim)),
        (state.option_policies.average_rewards, (n_options,)),
        (state.option_models.cumreward_ema, (n_options,)),
        (state.option_models.env_return_ema, (n_options,)),
        (state.option_models.duration_ema, (n_options,)),
        (state.option_models.baseline_mass_ema, (n_options,)),
        (state.option_models.discount_ema, (n_options,)),
        (
            state.option_models.next_state_weights,
            (n_options, observation_dim, observation_dim),
        ),
    )
    scalar_float32 = (
        state.base_average_reward,
        state.option_cumreward,
        state.option_env_cumreward,
        state.option_baseline_mass,
        state.option_discount,
    )
    scalar_int32 = (
        state.base_last_action,
        state.last_primitive_action,
        state.executing_option,
        state.option_last_intra_action,
        state.option_steps,
        state.step_count,
        state.base_learner_state.step_count,
    )
    static_valid = all(
        value.shape == expected and value.dtype == jnp.float32 for value, expected in float32_shapes
    )
    static_valid = static_valid and all(
        value.shape == () and value.dtype == jnp.float32 for value in scalar_float32
    )
    static_valid = static_valid and all(
        value.shape == () and value.dtype == jnp.int32 for value in scalar_int32
    )
    static_valid = static_valid and (
        state.option_models.n_completions.shape == (n_options,)
        and state.option_models.n_completions.dtype == jnp.int32
    )
    static_valid = static_valid and (
        state.step_words.shape == (2,)
        and state.step_words.dtype == jnp.uint32
        and state.base_learner_state.step_words.shape == (2,)
        and state.base_learner_state.step_words.dtype == jnp.uint32
    )
    n_total_actions = n_primitive_actions + n_options
    learner = state.base_learner_state
    head_collections_valid = (
        len(learner.head_params.weights) == n_total_actions
        and len(learner.head_params.biases) == n_total_actions
        and len(learner.head_optimizer_states) == n_total_actions
        and len(learner.head_traces) == n_total_actions
    )
    static_valid = static_valid and head_collections_valid
    trunk_depth = len(learner.trunk_params.weights)
    trunk_collections_valid = (
        len(learner.trunk_params.biases) == trunk_depth
        and len(learner.hidden_unit_utilities) == trunk_depth
        and len(learner.trunk_traces) == 2 * trunk_depth
        and len(learner.trunk_optimizer_states) == 2 * trunk_depth
        and learner.normalizer_state is None
    )
    static_valid = static_valid and trunk_collections_valid
    trunk_input_width = observation_dim
    if trunk_collections_valid:
        for layer_idx in range(trunk_depth):
            weight = learner.trunk_params.weights[layer_idx]
            bias = learner.trunk_params.biases[layer_idx]
            utility = learner.hidden_unit_utilities[layer_idx]
            expected_weight_shape = (
                int(weight.shape[0]) if weight.ndim == 2 else 0,
                trunk_input_width,
            )
            expected_bias_shape = (expected_weight_shape[0],)
            static_valid = static_valid and (
                weight.ndim == 2
                and weight.shape == expected_weight_shape
                and weight.dtype == jnp.float32
                and bias.shape == expected_bias_shape
                and bias.dtype == jnp.float32
                and utility.shape == expected_bias_shape
                and utility.dtype == jnp.float32
                and learner.trunk_traces[2 * layer_idx].shape == weight.shape
                and learner.trunk_traces[2 * layer_idx].dtype == jnp.float32
                and learner.trunk_traces[2 * layer_idx + 1].shape == bias.shape
                and learner.trunk_traces[2 * layer_idx + 1].dtype == jnp.float32
            )
            trunk_input_width = expected_weight_shape[0]
        for optimizer_state in learner.trunk_optimizer_states:
            static_valid = static_valid and (
                isinstance(optimizer_state, LMSState)
                and optimizer_state.step_size.shape == ()
                and optimizer_state.step_size.dtype == jnp.float32
            )
    if head_collections_valid:
        for head_idx in range(n_total_actions):
            weight = learner.head_params.weights[head_idx]
            bias = learner.head_params.biases[head_idx]
            trace_pair = learner.head_traces[head_idx]
            optimizer_pair = learner.head_optimizer_states[head_idx]
            pairs_valid = (
                isinstance(trace_pair, tuple)
                and len(trace_pair) == 2
                and isinstance(optimizer_pair, tuple)
                and len(optimizer_pair) == 2
            )
            static_valid = static_valid and pairs_valid
            if not pairs_valid:
                continue
            weight_trace, bias_trace = trace_pair
            static_valid = static_valid and (
                weight.shape == (1, trunk_input_width)
                and weight.dtype == jnp.float32
                and bias.shape == (1,)
                and bias.dtype == jnp.float32
                and weight_trace.shape == weight.shape
                and weight_trace.dtype == jnp.float32
                and bias_trace.shape == bias.shape
                and bias_trace.dtype == jnp.float32
            )
            for optimizer_state in optimizer_pair:
                static_valid = static_valid and (
                    isinstance(optimizer_state, LMSState)
                    and optimizer_state.step_size.shape == ()
                    and optimizer_state.step_size.dtype == jnp.float32
                )
    rng_key_valid = state.rng_key.shape == () and jax.dtypes.issubdtype(
        state.rng_key.dtype, jax.dtypes.prng_key
    )
    return static_valid, rng_key_valid


def replace_dispatched_primitive_action(
    state: STOMPState,
    decision_observation: Array,
    proposed_action: Array,
    safety_action_mask: Array | None = None,
) -> DispatchedPrimitiveActionReplacementResult:
    """Replace only the decision owner that will receive next-step credit.

    The current primitive action is the safe fallback. An invalid proposal,
    stale observation, inconsistent owner state, or unsafe fallback fails
    closed as an exact state no-op. A valid but unsafe proposal uses the safe
    current action without consuming RNG or changing credit ownership.
    """

    if not isinstance(state, STOMPState):
        raise TypeError("state must be a STOMPState")
    if state.option_policies.q_weights.ndim != 3:
        raise TypeError("state.option_policies.q_weights must have rank 3")
    n_options = state.option_policies.q_weights.shape[0]
    n_primitive_actions = state.option_policies.q_weights.shape[1]
    observation_dim = state.option_policies.q_weights.shape[2]
    if n_options < 0 or n_primitive_actions < 1 or observation_dim < 1:
        raise TypeError(
            "STOMP dispatch requires nonnegative option count and positive "
            "primitive/observation dimensions"
        )

    raw_observation = jnp.asarray(decision_observation)
    observation_static_contract_valid = (
        raw_observation.shape == (observation_dim,) and raw_observation.dtype == jnp.float32
    )
    obs = (
        raw_observation
        if observation_static_contract_valid
        else jnp.zeros((observation_dim,), dtype=jnp.float32)
    )
    raw_proposal = jnp.asarray(proposed_action)
    proposed_action_static_contract_valid = (
        raw_proposal.shape == () and raw_proposal.dtype == jnp.int32
    )
    proposal = (
        raw_proposal if proposed_action_static_contract_valid else jnp.asarray(-1, dtype=jnp.int32)
    )
    if safety_action_mask is None:
        safety = jnp.ones((n_primitive_actions,), dtype=jnp.bool_)
        safety_action_mask_static_contract_valid = True
    else:
        raw_safety = jnp.asarray(safety_action_mask)
        safety_action_mask_static_contract_valid = (
            raw_safety.shape == (n_primitive_actions,) and raw_safety.dtype == jnp.bool_
        )
        safety = (
            raw_safety
            if safety_action_mask_static_contract_valid
            else jnp.zeros((n_primitive_actions,), dtype=jnp.bool_)
        )

    no_option = state.executing_option == -1
    option_index_valid = (state.executing_option >= 0) & (state.executing_option < n_options)
    owner = jnp.where(
        no_option,
        jnp.asarray(DISPATCH_OWNER_BASE_PRIMITIVE, dtype=jnp.int32),
        jnp.where(
            option_index_valid,
            jnp.asarray(DISPATCH_OWNER_OPTION, dtype=jnp.int32),
            jnp.asarray(DISPATCH_OWNER_INVALID, dtype=jnp.int32),
        ),
    )
    counterfactual = state.last_primitive_action
    counterfactual_valid = (counterfactual >= 0) & (counterfactual < n_primitive_actions)
    base_owner_valid = (
        no_option
        & (state.base_last_action >= 0)
        & (state.base_last_action < n_primitive_actions)
        & (state.base_last_action == counterfactual)
    )
    option_owner_valid = (
        option_index_valid
        & (state.base_last_action == n_primitive_actions + state.executing_option)
        & (state.option_last_intra_action == counterfactual)
    )
    ownership_valid = counterfactual_valid & (base_owner_valid | option_owner_valid)
    static_contract_valid, typed_rng_key_valid = _stomp_static_dispatch_contract(
        state,
        n_options=n_options,
        n_primitive_actions=n_primitive_actions,
        observation_dim=observation_dim,
    )
    state_values_finite = _all_floating_leaves_finite(state)
    state_counters_valid = (
        _all_integer_leaves_nonnegative(state.base_learner_state)
        & _lifetime_counter_valid(state.step_words, state.step_count)
        & _lifetime_counter_valid(
            state.base_learner_state.step_words,
            state.base_learner_state.step_count,
        )
        & jnp.all(state.option_models.n_completions >= 0)
        & (state.option_steps >= 0)
        & (state.step_count >= 0)
        & (state.option_steps <= state.step_count)
        & (state.option_baseline_mass >= 0.0)
        & (state.option_discount >= 0.0)
        & (state.option_discount <= 1.0)
        & jnp.all(state.option_models.duration_ema >= 0.0)
        & jnp.all(state.option_models.baseline_mass_ema >= 0.0)
        & jnp.all(state.option_models.discount_ema >= 0.0)
        & jnp.all(state.option_models.discount_ema <= 1.0)
    )
    state_valid = (
        jnp.asarray(static_contract_valid, dtype=jnp.bool_)
        & jnp.asarray(typed_rng_key_valid, dtype=jnp.bool_)
        & state_values_finite
        & state_counters_valid
        & ownership_valid
    )
    observation_valid = jnp.all(jnp.isfinite(obs))
    observation_matches = (
        jnp.asarray(observation_static_contract_valid, dtype=jnp.bool_)
        & observation_valid
        & jnp.all(jnp.isfinite(state.base_last_obs))
        & jnp.array_equal(
            jax.lax.bitcast_convert_type(obs, jnp.int32),
            jax.lax.bitcast_convert_type(state.base_last_obs, jnp.int32),
        )
    )
    proposed_valid = (
        jnp.asarray(proposed_action_static_contract_valid, dtype=jnp.bool_)
        & (proposal >= 0)
        & (proposal < n_primitive_actions)
    )
    safe_counterfactual_index = jnp.clip(counterfactual, 0, n_primitive_actions - 1)
    safe_proposal_index = jnp.clip(proposal, 0, n_primitive_actions - 1)
    counterfactual_safe = counterfactual_valid & safety[safe_counterfactual_index]
    proposed_safe = proposed_valid & safety[safe_proposal_index]
    common_valid = (
        state_valid
        & observation_matches
        & jnp.asarray(safety_action_mask_static_contract_valid, dtype=jnp.bool_)
        & counterfactual_safe
    )
    apply_proposal = common_valid & proposed_valid & proposed_safe
    use_fallback = common_valid & proposed_valid & ~proposed_safe
    failed_closed = ~common_valid | ~proposed_valid
    effective_action = jnp.where(
        apply_proposal,
        proposal,
        jnp.where(
            use_fallback,
            counterfactual,
            jnp.asarray(-1, dtype=jnp.int32),
        ),
    )
    changed = apply_proposal & (proposal != counterfactual)
    next_state = state.replace(
        last_primitive_action=jnp.where(
            apply_proposal,
            proposal,
            state.last_primitive_action,
        ),
        base_last_action=jnp.where(
            apply_proposal & (owner == DISPATCH_OWNER_BASE_PRIMITIVE),
            proposal,
            state.base_last_action,
        ),
        option_last_intra_action=jnp.where(
            apply_proposal & (owner == DISPATCH_OWNER_OPTION),
            proposal,
            state.option_last_intra_action,
        ),
    )
    decision = DispatchedPrimitiveActionDecision(
        owner=jnp.where(
            state_valid,
            owner,
            jnp.asarray(DISPATCH_OWNER_INVALID, dtype=jnp.int32),
        ),
        state_static_contract_valid=jnp.asarray(static_contract_valid, dtype=jnp.bool_),
        state_values_finite=state_values_finite,
        state_counters_valid=state_counters_valid,
        rng_key_valid=jnp.asarray(typed_rng_key_valid, dtype=jnp.bool_),
        ownership_valid=ownership_valid,
        state_valid=state_valid,
        observation_static_contract_valid=jnp.asarray(
            observation_static_contract_valid, dtype=jnp.bool_
        ),
        observation_valid=observation_valid,
        observation_matches=observation_matches,
        proposed_action_static_contract_valid=jnp.asarray(
            proposed_action_static_contract_valid, dtype=jnp.bool_
        ),
        proposed_action_valid=proposed_valid,
        safety_action_mask_static_contract_valid=jnp.asarray(
            safety_action_mask_static_contract_valid, dtype=jnp.bool_
        ),
        counterfactual_action_safe=counterfactual_safe,
        proposed_action_safe=proposed_safe,
        counterfactual_action=counterfactual,
        proposed_action=proposal,
        effective_action=effective_action,
        used_safe_base_fallback=use_fallback,
        applied=changed,
        failed_closed=failed_closed,
    )
    return DispatchedPrimitiveActionReplacementResult(
        state=next_state,
        decision=decision,
    )


# ---------------------------------------------------------------------------
# Update-result types
# ---------------------------------------------------------------------------


@chex.dataclass(frozen=True)
class STOMPStartResult:
    """Primed STOMP state and the first primitive action to dispatch."""

    state: STOMPState
    primitive_action: Int[Array, ""]


@chex.dataclass(frozen=True)
class STOMPUpdateResult:
    """Result of one primitive STOMP transition."""

    state: STOMPState
    td_error: Float[Array, ""]
    average_reward: Float[Array, ""]
    primitive_action: Int[Array, ""]
    executing_option: Int[Array, ""]
    option_terminated: Array
    pseudo_reward: Float[Array, ""]
    option_importance_ratio: Float[Array, ""]
    planning_backups: Int[Array, ""]
    planning_td_error: Float[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    inputs_valid: Bool[Array, ""]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    nested_lifetime_counter_valid: Bool[Array, ""]
    nested_lifetime_capacity_available: Bool[Array, ""]
    nested_updates_required: Int[Array, ""]
    nested_updates_applied: Int[Array, ""]
    proposed_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class STOMPArrayResult:
    """Result of a scan-based STOMP run over transition arrays."""

    state: STOMPState
    td_errors: Float[Array, " num_steps"]
    average_rewards: Float[Array, " num_steps"]
    primitive_actions: Int[Array, " num_steps"]
    executing_options: Int[Array, " num_steps"]
    option_terminations: Array
    pseudo_rewards: Float[Array, " num_steps"]
    option_importance_ratios: Float[Array, " num_steps"]
    planning_backups: Int[Array, " num_steps"]
    planning_td_errors: Float[Array, " num_steps"]
    pre_step_words: UInt[Array, "num_steps 2"]
    post_step_words: UInt[Array, "num_steps 2"]
    inputs_valid: Bool[Array, " num_steps"]
    lifetime_counter_valid: Bool[Array, " num_steps"]
    lifetime_capacity_available: Bool[Array, " num_steps"]
    nested_lifetime_counter_valid: Bool[Array, " num_steps"]
    nested_lifetime_capacity_available: Bool[Array, " num_steps"]
    nested_updates_required: Int[Array, " num_steps"]
    nested_updates_applied: Int[Array, " num_steps"]
    proposed_state_valid: Bool[Array, " num_steps"]
    update_applied: Bool[Array, " num_steps"]


# ---------------------------------------------------------------------------
# Helper: pseudo-reward and termination conditions
# ---------------------------------------------------------------------------


def compute_pseudo_reward(
    spec_arrays: STOMPSpecArrays,
    option_idx: Array,
    observation: Array,
) -> Float[Array, ""]:
    """Compute pseudo-reward for one option given an observation."""
    feat_idx = spec_arrays.feature_indices[option_idx]
    scale = spec_arrays.pseudo_reward_scales[option_idx]
    return scale * observation[feat_idx]


def check_option_terminated(
    spec_arrays: STOMPSpecArrays,
    option_idx: Array,
    observation: Array,
    option_steps: Array,
) -> Array:
    """Return True if the option should terminate."""
    pseudo_r = compute_pseudo_reward(spec_arrays, option_idx, observation)
    goal_reached = pseudo_r >= spec_arrays.thresholds[option_idx]
    max_exceeded = option_steps >= spec_arrays.max_option_steps[option_idx]
    return goal_reached | max_exceeded


# ---------------------------------------------------------------------------
# Core update functions
# ---------------------------------------------------------------------------


def _q_values_for_obs(q_weights: Array, observation: Array) -> Array:
    """Compute Q(s, ·) = q_weights @ obs for all actions."""
    return q_weights @ observation


def _select_action_epsilon_greedy(
    q_weights: Array,
    observation: Array,
    key: Array,
    epsilon: float,
    n_actions: int,
) -> tuple[Array, Array]:
    """ε-greedy action selection with Gumbel tie-breaking."""
    key, explore_key, noise_key = jr.split(key, 3)
    q_vals = _q_values_for_obs(q_weights, observation)
    greedy = jnp.argmax(q_vals + 1e-6 * jr.gumbel(noise_key, (n_actions,))).astype(jnp.int32)
    random_action = jr.randint(explore_key, (), 0, n_actions).astype(jnp.int32)
    explore = jr.uniform(key) < jnp.asarray(epsilon, dtype=jnp.float32)
    action = jnp.where(explore, random_action, greedy)
    return action, key


def _select_action_epsilon_greedy_from_q(
    q_vals: Array,
    key: Array,
    epsilon: float,
    n_actions: int,
) -> tuple[Array, Array]:
    """ε-greedy action selection from pre-computed Q values."""
    key, explore_key, noise_key = jr.split(key, 3)
    greedy = jnp.argmax(q_vals + 1e-6 * jr.gumbel(noise_key, (n_actions,))).astype(jnp.int32)
    random_action = jr.randint(explore_key, (), 0, n_actions).astype(jnp.int32)
    explore = jr.uniform(key) < jnp.asarray(epsilon, dtype=jnp.float32)
    action = jnp.where(explore, random_action, greedy)
    return action, key


def _select_action_epsilon_greedy_from_q_masked(
    q_vals: Array,
    key: Array,
    epsilon: float,
    action_mask: Array,
) -> tuple[Array, Array]:
    """Select only eligible actions while preserving all-true RNG parity.

    The key split, Gumbel draw, exploration draw, and returned key are exactly
    the same as :func:`_select_action_epsilon_greedy_from_q`.  With an all-true
    mask the selected action is therefore bit-identical to the legacy path.
    """

    n_actions = q_vals.shape[0]
    key, explore_key, noise_key = jr.split(key, 3)
    noisy = q_vals + 1e-6 * jr.gumbel(noise_key, (n_actions,))
    masked_noisy = jnp.where(action_mask, noisy, -jnp.inf)
    greedy = jnp.argmax(masked_noisy).astype(jnp.int32)
    eligible_count = jnp.sum(action_mask.astype(jnp.int32))
    safe_count = jnp.maximum(eligible_count, jnp.asarray(1, dtype=jnp.int32))
    random_rank = jr.randint(explore_key, (), 0, safe_count).astype(jnp.int32)
    eligible_indices = jnp.nonzero(
        action_mask,
        size=n_actions,
        fill_value=0,
    )[0]
    random_action = eligible_indices[random_rank]
    explore = jr.uniform(key) < jnp.asarray(epsilon, dtype=jnp.float32)
    action = jnp.where(explore, random_action, greedy)
    return action, key


def _epsilon_greedy_action_probabilities(q_values: Array, epsilon: Array) -> Array:
    """Return epsilon-greedy probabilities with uniform tie handling."""
    q = jnp.asarray(q_values, dtype=jnp.float32)
    n_actions = q.shape[0]
    eps = jnp.asarray(epsilon, dtype=jnp.float32)
    max_q = jnp.max(q)
    greedy_mask = jnp.isclose(q, max_q, atol=1e-6, rtol=0.0).astype(jnp.float32)
    n_greedy = jnp.maximum(jnp.sum(greedy_mask), jnp.array(1.0, dtype=jnp.float32))
    return eps / n_actions + (1.0 - eps) * greedy_mask / n_greedy


def _clipped_epsilon_greedy_importance_ratio(
    q_weights: Array,
    observation: Array,
    action: Array,
    *,
    behavior_epsilon: float,
    target_epsilon: float,
    clip: float,
) -> Array:
    """Return clipped target/behavior probability ratio for one action."""
    q_values = _q_values_for_obs(q_weights, observation)
    behavior = _epsilon_greedy_action_probabilities(
        q_values,
        jnp.asarray(behavior_epsilon, dtype=jnp.float32),
    )
    target = _epsilon_greedy_action_probabilities(
        q_values,
        jnp.asarray(target_epsilon, dtype=jnp.float32),
    )
    selected_behavior = behavior[action]
    selected_target = target[action]
    ratio = selected_target / jnp.maximum(
        selected_behavior,
        jnp.asarray(1.0e-6, dtype=jnp.float32),
    )
    return jnp.minimum(ratio, jnp.asarray(clip, dtype=jnp.float32))


def _differential_q_update(
    q_weights: Array,
    traces: Array,
    average_reward: Array,
    last_obs: Array,
    last_action: Array,
    reward: Array,
    next_obs: Array,
    *,
    step_size: float,
    avg_reward_step_size: float,
    trace_decay: float,
    n_actions: int,
) -> tuple[Array, Array, Array, Array, Bool[Array, ""]]:
    """One differential SARSA Q-update step.

    Returns (new_q_weights, new_traces, new_average_reward, td_error).
    """
    alpha = jnp.asarray(step_size, dtype=jnp.float32)
    beta = jnp.asarray(avg_reward_step_size, dtype=jnp.float32)
    lam = jnp.asarray(trace_decay, dtype=jnp.float32)

    q_prev = q_weights[last_action] @ last_obs
    q_next = jnp.max(_q_values_for_obs(q_weights, next_obs))
    td_error = reward - average_reward + q_next - q_prev

    action_mask = jax.nn.one_hot(last_action, n_actions, dtype=jnp.float32)
    new_traces = lam * traces + action_mask[:, None] * last_obs[None, :]
    delta_w = alpha * td_error * new_traces
    proposed_q = q_weights + delta_w
    proposed_rbar = average_reward + beta * td_error
    inputs_valid = (
        jnp.all(jnp.isfinite(last_obs))
        & jnp.isfinite(jnp.asarray(reward, dtype=jnp.float32))
        & jnp.all(jnp.isfinite(next_obs))
        & (last_action >= 0)
        & (last_action < n_actions)
        & jnp.all(jnp.isfinite(q_weights))
        & jnp.all(jnp.isfinite(traces))
        & jnp.isfinite(average_reward)
    )
    proposed_finite = (
        jnp.all(jnp.isfinite(proposed_q))
        & jnp.all(jnp.isfinite(new_traces))
        & jnp.isfinite(proposed_rbar)
    )
    update_applied = inputs_valid & proposed_finite
    new_q_weights, new_traces, new_average_reward = jax.lax.cond(
        update_applied,
        lambda: (proposed_q, new_traces, proposed_rbar),
        lambda: (q_weights, traces, average_reward),
    )
    return (
        new_q_weights,
        new_traces,
        new_average_reward,
        jnp.where(update_applied, td_error, jnp.zeros_like(td_error)),
        update_applied,
    )


def _differential_semidp_q_update(
    q_weights: Array,
    traces: Array,
    average_reward: Array,
    last_obs: Array,
    last_action: Array,
    reward: Array,
    next_obs: Array,
    *,
    step_size: float,
    avg_reward_step_size: float,
    trace_decay: float,
    n_actions: int,
    baseline_mass: Array,
    discount: Array,
) -> tuple[Array, Array, Array, Array, Bool[Array, ""]]:
    """Discounted differential Q-update for semi-MDP option returns.

    Extends :func:`_differential_q_update` to correctly account for
    a discounted multi-step return and its matching baseline mass:

    .. code-block::

        td = R_o^γ - avg_r * Σ(k=0..T_o-1) γ^k
             + γ^T_o * V(s') - Q(s, o)

    For primitive steps pass ``baseline_mass=1, discount=1`` to recover the
    standard single-step update exactly.

    Args:
        baseline_mass: Discounted baseline coefficient
            ``Σ(k=0..T_o-1) γ^k``. ``1`` for primitive actions. When
            ``γ = 1`` this equals the raw option duration ``T_o``.
        discount: Cumulative per-step discount across the option (γ^{T_o}).
            ``1.0`` for primitive actions.

    Returns:
        ``(new_q_weights, new_traces, new_average_reward, td_error)``.
    """
    alpha = jnp.asarray(step_size, dtype=jnp.float32)
    beta = jnp.asarray(avg_reward_step_size, dtype=jnp.float32)
    lam = jnp.asarray(trace_decay, dtype=jnp.float32)
    baseline_coefficient = jnp.asarray(baseline_mass, dtype=jnp.float32)
    gamma_o = jnp.asarray(discount, dtype=jnp.float32)

    q_prev = q_weights[last_action] @ last_obs
    q_next = jnp.max(_q_values_for_obs(q_weights, next_obs))
    # The discounted reward and baseline must use the same γ powers.
    td_error = reward - average_reward * baseline_coefficient + gamma_o * q_next - q_prev

    action_mask = jax.nn.one_hot(last_action, n_actions, dtype=jnp.float32)
    new_traces = lam * traces + action_mask[:, None] * last_obs[None, :]
    delta_w = alpha * td_error * new_traces
    proposed_q = q_weights + delta_w
    proposed_rbar = average_reward + beta * td_error
    inputs_valid = (
        jnp.all(jnp.isfinite(last_obs))
        & jnp.isfinite(jnp.asarray(reward, dtype=jnp.float32))
        & jnp.all(jnp.isfinite(next_obs))
        & jnp.isfinite(baseline_coefficient)
        & jnp.isfinite(gamma_o)
        & (last_action >= 0)
        & (last_action < n_actions)
        & jnp.all(jnp.isfinite(q_weights))
        & jnp.all(jnp.isfinite(traces))
        & jnp.isfinite(average_reward)
    )
    proposed_finite = (
        jnp.all(jnp.isfinite(proposed_q))
        & jnp.all(jnp.isfinite(new_traces))
        & jnp.isfinite(proposed_rbar)
    )
    update_applied = inputs_valid & proposed_finite
    new_q_weights, new_traces, new_average_reward = jax.lax.cond(
        update_applied,
        lambda: (proposed_q, new_traces, proposed_rbar),
        lambda: (q_weights, traces, average_reward),
    )
    return (
        new_q_weights,
        new_traces,
        new_average_reward,
        jnp.where(update_applied, td_error, jnp.zeros_like(td_error)),
        update_applied,
    )


def _update_option_model(
    models: OptionModelsState,
    option_idx: Array,
    start_obs: Array,
    pseudo_return: Array,
    env_return: Array,
    duration: Array,
    baseline_mass: Array,
    discount: Array,
    end_obs: Array,
    *,
    model_decay: float,
    model_step_size: float,
) -> OptionModelsState:
    """Update the model for one option from a completed trajectory."""
    decay = jnp.asarray(model_decay, dtype=jnp.float32)
    lr = jnp.asarray(model_step_size, dtype=jnp.float32)

    new_cumreward = decay * models.cumreward_ema + (1.0 - decay) * pseudo_return
    new_env_return = decay * models.env_return_ema + (1.0 - decay) * env_return
    new_duration = decay * models.duration_ema + (1.0 - decay) * duration
    new_baseline_mass = decay * models.baseline_mass_ema + (1.0 - decay) * baseline_mass
    new_discount = decay * models.discount_ema + (1.0 - decay) * discount

    predicted_delta = models.next_state_weights[option_idx] @ start_obs
    actual_delta = end_obs - start_obs
    delta_error = actual_delta - predicted_delta
    ns_update = lr * jnp.outer(delta_error, start_obs)

    mask = (jnp.arange(models.cumreward_ema.shape[0], dtype=jnp.int32) == option_idx).astype(
        jnp.float32
    )
    new_ns_weights = models.next_state_weights + mask[:, None, None] * ns_update[None, :, :]
    incremented_completions = _saturating_int32_counter_increment(models.n_completions)
    new_completions = jnp.where(
        mask.astype(jnp.bool_),
        incremented_completions,
        models.n_completions,
    )

    return OptionModelsState(
        cumreward_ema=jnp.where(mask, new_cumreward, models.cumreward_ema),
        env_return_ema=jnp.where(mask, new_env_return, models.env_return_ema),
        duration_ema=jnp.where(mask, new_duration, models.duration_ema),
        baseline_mass_ema=jnp.where(mask, new_baseline_mass, models.baseline_mass_ema),
        discount_ema=jnp.where(mask, new_discount, models.discount_ema),
        next_state_weights=new_ns_weights,
        n_completions=new_completions,
    )


def _reset_linear_stomp_feature_axes(
    base_state: MultiHeadMLPState,
    option_policies: IntraOptionPoliciesState,
    option_models: OptionModelsState,
    feature_mask: Array,
) -> tuple[MultiHeadMLPState, IntraOptionPoliciesState, OptionModelsState]:
    """Scrub recycled representation axes after learning, before selection.

    This deliberately supports only STOMP's linear base learner.  It removes
    every selected input coefficient and eligibility trace from the base and
    intra-option heads, plus both the input and predicted-output axes from the
    option transition models.  Clocks, scalar baselines, and current-transition
    learning diagnostics are unchanged.
    """

    selected = jnp.asarray(feature_mask, dtype=jnp.bool_)
    head_params = cast(
        Any,
        base_state.head_params.replace(
            weights=tuple(
                jnp.where(selected[None, :], 0.0, weights)
                for weights in base_state.head_params.weights
            )
        ),
    )
    head_traces = tuple(
        (
            jnp.where(selected[None, :], 0.0, weight_trace),
            bias_trace,
        )
        for weight_trace, bias_trace in base_state.head_traces
    )
    scrubbed_base = cast(
        MultiHeadMLPState,
        base_state.replace(
            head_params=head_params,
            head_traces=head_traces,
        ),
    )
    scrubbed_policies = IntraOptionPoliciesState(
        q_weights=jnp.where(
            selected[None, None, :],
            0.0,
            option_policies.q_weights,
        ),
        traces=jnp.where(
            selected[None, None, :],
            0.0,
            option_policies.traces,
        ),
        average_rewards=option_policies.average_rewards,
    )
    model_axis_mask = selected[None, :, None] | selected[None, None, :]
    scrubbed_models = OptionModelsState(
        cumreward_ema=option_models.cumreward_ema,
        env_return_ema=option_models.env_return_ema,
        duration_ema=option_models.duration_ema,
        baseline_mass_ema=option_models.baseline_mass_ema,
        discount_ema=option_models.discount_ema,
        next_state_weights=jnp.where(
            model_axis_mask,
            0.0,
            option_models.next_state_weights,
        ),
        n_completions=option_models.n_completions,
    )
    return scrubbed_base, scrubbed_policies, scrubbed_models


def _update_intra_option_policy(
    option_policies: IntraOptionPoliciesState,
    option_idx: Array,
    last_obs: Array,
    last_intra_action: Array,
    pseudo_reward: Array,
    next_obs: Array,
    terminated: Array,
    discount: Array,
    *,
    step_size: float,
    avg_reward_step_size: float,
    trace_decay: float,
    n_primitive_actions: int,
    importance_ratio: Array,
) -> tuple[IntraOptionPoliciesState, Array, Bool[Array, ""]]:
    """Update one intra-option Q-function with a transition discount.

    ``terminated`` is the option's own termination decision (goal, duration,
    or environmental termination).  It always zeros the bootstrap.  Otherwise
    the supplied transition discount controls both bootstrapping and trace
    carry, so fractional continuation is not silently promoted to one.
    """
    q_i = option_policies.q_weights[option_idx]
    traces_i = option_policies.traces[option_idx]
    avg_r_i = option_policies.average_rewards[option_idx]
    transition_discount = jnp.asarray(discount, dtype=jnp.float32)
    bootstrap_discount = jnp.where(terminated, 0.0, transition_discount).astype(jnp.float32)

    q_prev = q_i[last_intra_action] @ last_obs
    q_next = jnp.max(_q_values_for_obs(q_i, next_obs)) * bootstrap_discount
    td_error = pseudo_reward - avg_r_i + q_next - q_prev

    alpha = jnp.asarray(step_size, dtype=jnp.float32)
    beta = jnp.asarray(avg_reward_step_size, dtype=jnp.float32)
    lam = jnp.asarray(trace_decay, dtype=jnp.float32) * bootstrap_discount
    rho = jnp.asarray(importance_ratio, dtype=jnp.float32)

    action_mask = jax.nn.one_hot(last_intra_action, n_primitive_actions, dtype=jnp.float32)
    new_traces_i = rho * (lam * traces_i + action_mask[:, None] * last_obs[None, :])
    proposed_q_i = q_i + alpha * td_error * new_traces_i
    proposed_avg_r_i = avg_r_i + beta * rho * td_error
    inputs_valid = (
        jnp.all(jnp.isfinite(last_obs))
        & jnp.isfinite(jnp.asarray(pseudo_reward, dtype=jnp.float32))
        & jnp.all(jnp.isfinite(next_obs))
        & jnp.isfinite(transition_discount)
        & jnp.isfinite(rho)
        & (transition_discount >= 0.0)
        & (transition_discount <= 1.0)
        & (last_intra_action >= 0)
        & (last_intra_action < n_primitive_actions)
        & jnp.all(jnp.isfinite(q_i))
        & jnp.all(jnp.isfinite(traces_i))
        & jnp.isfinite(avg_r_i)
    )
    proposed_finite = (
        jnp.all(jnp.isfinite(proposed_q_i))
        & jnp.all(jnp.isfinite(new_traces_i))
        & jnp.isfinite(proposed_avg_r_i)
    )
    update_applied = inputs_valid & proposed_finite
    new_q_i, new_traces_i, new_avg_r_i = jax.lax.cond(
        update_applied,
        lambda: (proposed_q_i, new_traces_i, proposed_avg_r_i),
        lambda: (q_i, traces_i, avg_r_i),
    )

    n_opts = option_policies.average_rewards.shape[0]
    option_mask = jnp.arange(n_opts, dtype=jnp.int32) == option_idx

    new_q_weights = option_policies.q_weights.at[option_idx].set(new_q_i)
    new_traces = option_policies.traces.at[option_idx].set(new_traces_i)
    new_avg_rewards = jnp.where(option_mask, new_avg_r_i, option_policies.average_rewards)

    return (
        IntraOptionPoliciesState(
            q_weights=new_q_weights,
            traces=new_traces,
            average_rewards=new_avg_rewards,
        ),
        jnp.where(update_applied, td_error, jnp.zeros_like(td_error)),
        update_applied,
    )


# ---------------------------------------------------------------------------
# Configuration (Python-level, not a JAX type)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class STOMPConfig:
    """Configuration for the STOMP agent.

    Args:
        subtask_specs: List of subtask specifications (one per option).
        observation_dim: Flat observation dimensionality.
        n_primitive_actions: Number of primitive discrete actions.
        base_step_size: Step-size for the base extended Q-function.
        base_avg_reward_step_size: Average-reward rate step-size for base.
        base_trace_decay: Eligibility trace decay for the base agent.
        option_step_size: Step-size for intra-option Q-functions.
        option_avg_reward_step_size: Per-option average-reward rate step-size.
        option_trace_decay: Trace decay for intra-option Q-functions.
        option_gamma: Discount within option execution.
        option_model_decay: EMA decay for option outcome model updates.
        option_model_step_size: Step-size for next-state delta predictor.
        option_planning_backups_per_step: Fixed number of Dyna-style option
            model backups after each real transition. ``0`` disables planning
            and preserves the model-free update path. This is a static JIT
            configuration value.
        epsilon_base: Exploration rate for the base extended action selection.
        epsilon_option: Exploration rate for intra-option action selection.
    """

    subtask_specs: tuple[SubtaskSpec, ...] = ()
    observation_dim: int = 4
    n_primitive_actions: int = 2
    base_step_size: float = 0.05
    base_avg_reward_step_size: float = 0.01
    base_trace_decay: float = 0.0
    base_hidden_sizes: tuple[int, ...] = ()
    option_step_size: float = 0.05
    option_avg_reward_step_size: float = 0.01
    option_trace_decay: float = 0.0
    option_gamma: float = 0.99
    option_model_decay: float = 0.95
    option_model_step_size: float = 0.1
    option_planning_backups_per_step: int = 0
    epsilon_base: float = 0.1
    epsilon_option: float = 0.1
    option_target_epsilon: float | None = None
    option_importance_clip: float = 10.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if type(self.subtask_specs) is not tuple or any(
            type(spec) is not SubtaskSpec for spec in self.subtask_specs
        ):
            raise ValueError("subtask_specs must be an actual tuple of SubtaskSpec values")
        observation_dim = _require_int32("observation_dim", self.observation_dim, minimum=1)
        n_primitive_actions = _require_int32(
            "n_primitive_actions", self.n_primitive_actions, minimum=1
        )
        base_hidden_sizes = _validate_hidden_sizes(self.base_hidden_sizes)
        backups = _require_int32(
            "option_planning_backups_per_step",
            self.option_planning_backups_per_step,
            minimum=0,
            maximum=_INT32_MAX - 1,
        )

        object.__setattr__(self, "observation_dim", observation_dim)
        object.__setattr__(self, "n_primitive_actions", n_primitive_actions)
        object.__setattr__(self, "base_hidden_sizes", base_hidden_sizes)
        object.__setattr__(self, "option_planning_backups_per_step", backups)
        if len(self.subtask_specs) > _INT32_MAX - n_primitive_actions:
            raise ValueError("derived n_total_actions must fit signed int32")

        for spec in self.subtask_specs:
            if spec.feature_index >= observation_dim:
                raise ValueError(
                    f"SubtaskSpec.feature_index={spec.feature_index} >= "
                    f"observation_dim={observation_dim}"
                )
            if spec.max_option_steps > _INT32_MAX:
                raise ValueError("SubtaskSpec.max_option_steps must fit int32 telemetry")
        for step_size_name in (
            "base_step_size",
            "base_avg_reward_step_size",
            "option_step_size",
            "option_avg_reward_step_size",
            "option_model_step_size",
        ):
            object.__setattr__(
                self,
                step_size_name,
                validated_float32_scalar(
                    step_size_name, getattr(self, step_size_name), lower=0.0
                ),
            )
        for unit_interval_name in (
            "base_trace_decay",
            "option_trace_decay",
            "option_model_decay",
            "epsilon_base",
            "epsilon_option",
        ):
            object.__setattr__(
                self,
                unit_interval_name,
                validated_float32_scalar(
                    unit_interval_name,
                    getattr(self, unit_interval_name),
                    lower=0.0,
                    upper=1.0,
                ),
            )
        object.__setattr__(
            self,
            "option_gamma",
            validated_float32_scalar("option_gamma", self.option_gamma, lower=0.0, upper=1.0),
        )
        if self.option_target_epsilon is not None:
            object.__setattr__(
                self,
                "option_target_epsilon",
                validated_float32_scalar(
                    "option_target_epsilon",
                    self.option_target_epsilon,
                    lower=0.0,
                    upper=1.0,
                ),
            )
        object.__setattr__(
            self,
            "option_importance_clip",
            validated_float32_scalar(
                "option_importance_clip", self.option_importance_clip, positive=True
            ),
        )
        if not self.subtask_specs and self.option_planning_backups_per_step != 0:
            raise ValueError("primitive-only STOMP requires option_planning_backups_per_step == 0")
        direct_scalars = _stomp_direct_array_scalars(self)
        if direct_scalars > _INT32_MAX or 4 * direct_scalars > _INT32_MAX:
            raise ValueError("derived STOMP direct array bytes must fit signed int32")

    @property
    def n_options(self) -> int:
        """Number of options."""
        return len(self.subtask_specs)

    @property
    def n_total_actions(self) -> int:
        """Total extended action count (primitive + options)."""
        return self.n_primitive_actions + self.n_options

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "type": "STOMPConfig",
            "subtask_specs": [dataclasses.asdict(s) for s in self.subtask_specs],
            "observation_dim": self.observation_dim,
            "n_primitive_actions": self.n_primitive_actions,
            "base_step_size": self.base_step_size,
            "base_avg_reward_step_size": self.base_avg_reward_step_size,
            "base_trace_decay": self.base_trace_decay,
            "base_hidden_sizes": list(self.base_hidden_sizes),
            "option_step_size": self.option_step_size,
            "option_avg_reward_step_size": self.option_avg_reward_step_size,
            "option_trace_decay": self.option_trace_decay,
            "option_gamma": self.option_gamma,
            "option_model_decay": self.option_model_decay,
            "option_model_step_size": self.option_model_step_size,
            "option_planning_backups_per_step": self.option_planning_backups_per_step,
            "epsilon_base": self.epsilon_base,
            "epsilon_option": self.epsilon_option,
            "option_target_epsilon": self.option_target_epsilon,
            "option_importance_clip": self.option_importance_clip,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> STOMPConfig:
        """Reconstruct from :meth:`to_config` output."""
        if not issubclass(type(config), Mapping):
            raise ValueError("STOMPConfig payload must be a mapping")
        try:
            payload = dict(config)
        except Exception as error:
            raise ValueError("STOMPConfig mapping could not be read") from error
        payload.pop("type", None)
        specs_raw = payload.pop("subtask_specs", [])
        if type(specs_raw) not in (list, tuple):
            raise ValueError("serialized subtask_specs must be an actual list or tuple")
        specs: list[SubtaskSpec] = []
        for raw in specs_raw:
            if not issubclass(type(raw), Mapping):
                raise ValueError("serialized subtask_specs entries must be mappings")
            try:
                specs.append(SubtaskSpec(**dict(raw)))
            except Exception as error:
                raise ValueError("serialized SubtaskSpec is invalid") from error
        if "base_hidden_sizes" in payload:
            raw_hidden = payload["base_hidden_sizes"]
            if type(raw_hidden) not in (list, tuple):
                raise ValueError(
                    "serialized base_hidden_sizes must be an actual list or tuple"
                )
            sequence = cast(list[object] | tuple[object, ...], raw_hidden)
            payload["base_hidden_sizes"] = tuple(sequence)
        return cls(subtask_specs=tuple(specs), **payload)


# ---------------------------------------------------------------------------
# STOMP agent
# ---------------------------------------------------------------------------


class STOMPAgent:
    """Alberta Plan Step 10 STOMP agent.

    Combines a continuing base control agent (extended Q over primitive and
    option actions) with N intra-option policies and N option outcome models,
    one per :class:`SubtaskSpec`.

    The base agent uses differential Q-learning (average-reward formulation)
    over the extended action set.  Intra-option policies similarly use
    differential Q-learning with subtask pseudo-rewards.  Option models are
    updated online after each option termination.
    """

    def __init__(self, config: STOMPConfig):
        """Initialize the STOMP agent with a given configuration."""
        self._config = config
        self._spec_arrays = STOMPSpecArrays.from_specs(list(config.subtask_specs))
        self._base_learner = MultiHeadMLPLearner(
            n_heads=config.n_total_actions,
            hidden_sizes=config.base_hidden_sizes,
            step_size=config.base_step_size,
            gamma=0.0,
            lamda=0.0,
            per_head_gamma_lamda=(config.base_trace_decay,) * config.n_total_actions,
            sparsity=0.0,
        )

    @property
    def config(self) -> STOMPConfig:
        """Agent configuration."""
        return self._config

    @property
    def spec_arrays(self) -> STOMPSpecArrays:
        """JAX arrays derived from subtask specifications."""
        return self._spec_arrays

    @property
    def base_learner(self) -> MultiHeadMLPLearner:
        """Underlying base Q-function learner."""
        return self._base_learner

    def base_q_values(self, state: STOMPState, observation: Array) -> Array:
        """Compute Q-values for all extended actions from a STOMPState."""
        return self._base_learner.predict(state.base_learner_state, observation)

    def state_valid(self, state: STOMPState) -> Bool[Array, ""]:
        """Return the complete source-state validity gate for an update."""

        cfg = self._config
        static_valid, rng_valid = _stomp_static_dispatch_contract(
            state,
            n_options=cfg.n_options,
            n_primitive_actions=cfg.n_primitive_actions,
            observation_dim=cfg.observation_dim,
        )
        outer_clock_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        nested_clock_valid = _lifetime_counter_valid(
            state.base_learner_state.step_words,
            state.base_learner_state.step_count,
        )
        values_valid = (
            _all_floating_leaves_finite(state)
            & (state.option_baseline_mass >= 0.0)
            & (state.option_discount >= 0.0)
            & (state.option_discount <= 1.0)
            & jnp.all(state.option_models.duration_ema >= 0.0)
            & jnp.all(state.option_models.baseline_mass_ema >= 0.0)
            & jnp.all(state.option_models.discount_ema >= 0.0)
            & jnp.all(state.option_models.discount_ema <= 1.0)
        )
        counters_valid = (
            _all_integer_leaves_nonnegative(state.base_learner_state)
            & jnp.all(state.option_models.n_completions >= 0)
            & (state.option_steps >= 0)
            & (state.option_steps <= state.step_count)
        )
        ownership_valid = _stomp_action_ownership_valid(
            state,
            n_options=cfg.n_options,
            n_primitive_actions=cfg.n_primitive_actions,
        )
        return (
            jnp.asarray(static_valid, dtype=jnp.bool_)
            & jnp.asarray(rng_valid, dtype=jnp.bool_)
            & outer_clock_valid
            & nested_clock_valid
            & values_valid
            & counters_valid
            & ownership_valid
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize agent configuration."""
        return self._config.to_config()

    def init(self, key: Array) -> STOMPState:
        """Initialize agent state for a given observation dimensionality."""
        obs_dim = self._config.observation_dim
        n_prim = self._config.n_primitive_actions
        n_opt = self._config.n_options

        policy_key, learner_key, option_key = jr.split(key, 3)
        scale = 0.01
        base_learner_state = self._base_learner.init(obs_dim, learner_key)
        option_q_weights = scale * jr.normal(
            option_key, (n_opt, n_prim, obs_dim), dtype=jnp.float32
        )

        obs_zero = jnp.zeros(obs_dim, dtype=jnp.float32)
        return STOMPState(
            base_learner_state=base_learner_state,
            base_average_reward=jnp.array(0.0, dtype=jnp.float32),
            base_last_obs=obs_zero,
            base_last_action=jnp.array(0, dtype=jnp.int32),
            last_primitive_action=jnp.array(0, dtype=jnp.int32),
            rng_key=policy_key,
            option_policies=IntraOptionPoliciesState(
                q_weights=option_q_weights,
                traces=jnp.zeros((n_opt, n_prim, obs_dim), dtype=jnp.float32),
                average_rewards=jnp.zeros(n_opt, dtype=jnp.float32),
            ),
            option_models=OptionModelsState(
                cumreward_ema=jnp.zeros(n_opt, dtype=jnp.float32),
                env_return_ema=jnp.zeros(n_opt, dtype=jnp.float32),
                duration_ema=jnp.zeros(n_opt, dtype=jnp.float32),
                baseline_mass_ema=jnp.zeros(n_opt, dtype=jnp.float32),
                discount_ema=jnp.ones(n_opt, dtype=jnp.float32),
                next_state_weights=jnp.zeros((n_opt, obs_dim, obs_dim), dtype=jnp.float32),
                n_completions=jnp.zeros(n_opt, dtype=jnp.int32),
            ),
            executing_option=jnp.array(-1, dtype=jnp.int32),
            option_start_obs=obs_zero,
            option_last_intra_action=jnp.array(0, dtype=jnp.int32),
            option_cumreward=jnp.array(0.0, dtype=jnp.float32),
            option_env_cumreward=jnp.array(0.0, dtype=jnp.float32),
            option_baseline_mass=jnp.array(0.0, dtype=jnp.float32),
            option_discount=jnp.array(1.0, dtype=jnp.float32),
            option_steps=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def start(self, state: STOMPState, initial_observation: Array) -> STOMPState:
        """Prime state and select the first primitive action for dispatch.

        Use :meth:`start_with_action` when the caller needs the action
        explicitly. The state also records it in ``last_primitive_action``.
        """
        cfg = self._config
        obs = jnp.asarray(initial_observation, dtype=jnp.float32).reshape((cfg.observation_dim,))
        key = state.rng_key
        q_vals = self._base_learner.predict(state.base_learner_state, obs)
        extended_action, key = _select_action_epsilon_greedy_from_q(
            q_vals, key, cfg.epsilon_base, cfg.n_total_actions
        )
        if cfg.n_options == 0:
            return cast(
                STOMPState,
                state.replace(
                    base_last_obs=obs,
                    base_last_action=extended_action,
                    last_primitive_action=extended_action,
                    rng_key=key,
                    executing_option=jnp.asarray(-1, dtype=jnp.int32),
                    option_cumreward=jnp.asarray(0.0, dtype=jnp.float32),
                    option_env_cumreward=jnp.asarray(0.0, dtype=jnp.float32),
                    option_baseline_mass=jnp.asarray(0.0, dtype=jnp.float32),
                    option_discount=jnp.asarray(1.0, dtype=jnp.float32),
                    option_steps=jnp.asarray(0, dtype=jnp.int32),
                ),
            )
        is_starting_option = extended_action >= jnp.asarray(
            cfg.n_primitive_actions, dtype=jnp.int32
        )
        selected_option = jnp.clip(
            extended_action - cfg.n_primitive_actions,
            0,
            cfg.n_options - 1,
        )
        intra_action, key = _select_action_epsilon_greedy(
            state.option_policies.q_weights[selected_option],
            obs,
            key,
            cfg.epsilon_option,
            cfg.n_primitive_actions,
        )
        primitive_action = jnp.where(
            is_starting_option,
            intra_action,
            extended_action,
        )
        return cast(
            STOMPState,
            state.replace(
                base_last_obs=obs,
                base_last_action=extended_action,
                last_primitive_action=primitive_action,
                rng_key=key,
                executing_option=jnp.where(
                    is_starting_option,
                    selected_option,
                    jnp.array(-1, dtype=jnp.int32),
                ),
                option_start_obs=jnp.where(
                    is_starting_option,
                    obs,
                    state.option_start_obs,
                ),
                option_last_intra_action=jnp.where(
                    is_starting_option,
                    intra_action,
                    state.option_last_intra_action,
                ),
                option_cumreward=jnp.array(0.0, dtype=jnp.float32),
                option_env_cumreward=jnp.array(0.0, dtype=jnp.float32),
                option_baseline_mass=jnp.array(0.0, dtype=jnp.float32),
                option_discount=jnp.array(1.0, dtype=jnp.float32),
                option_steps=jnp.array(0, dtype=jnp.int32),
            ),
        )

    def start_with_action(
        self,
        state: STOMPState,
        initial_observation: Array,
    ) -> STOMPStartResult:
        """Prime the agent and return the first primitive action to execute."""
        primed = self.start(state, initial_observation)
        return STOMPStartResult(
            state=primed,
            primitive_action=primed.last_primitive_action,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def start_with_extended_action_mask(
        self,
        state: STOMPState,
        initial_observation: Array,
        extended_action_mask: Array,
    ) -> STOMPStartResult:
        """Prime control while keeping inactive option heads behavior-ineligible.

        The mask spans primitive actions followed by option actions. Every
        primitive action must remain eligible; callers may disable any subset
        of option slots. Invalid mask values or persistent state are exact
        state no-ops. An all-true mask preserves legacy :meth:`start` state and
        RNG bits exactly.
        """

        cfg = self._config
        raw_mask = jnp.asarray(extended_action_mask)
        if raw_mask.shape != (cfg.n_total_actions,):
            raise ValueError(
                "extended_action_mask must have shape "
                f"({cfg.n_total_actions},), got {raw_mask.shape}"
            )
        if raw_mask.dtype != jnp.bool_:
            raise TypeError(f"extended_action_mask must have dtype bool, got {raw_mask.dtype}")
        obs = jnp.asarray(initial_observation, dtype=jnp.float32).reshape((cfg.observation_dim,))
        mask_valid = jnp.all(raw_mask[: cfg.n_primitive_actions]) & jnp.any(raw_mask)
        values_valid = jnp.all(jnp.isfinite(obs))
        key = state.rng_key
        q_vals = self._base_learner.predict(state.base_learner_state, obs)
        extended_action, key = _select_action_epsilon_greedy_from_q_masked(
            q_vals,
            key,
            cfg.epsilon_base,
            raw_mask,
        )
        if cfg.n_options == 0:
            proposed = cast(
                STOMPState,
                state.replace(
                    base_last_obs=obs,
                    base_last_action=extended_action,
                    last_primitive_action=extended_action,
                    rng_key=key,
                    executing_option=jnp.asarray(-1, dtype=jnp.int32),
                    option_cumreward=jnp.asarray(0.0, dtype=jnp.float32),
                    option_env_cumreward=jnp.asarray(0.0, dtype=jnp.float32),
                    option_baseline_mass=jnp.asarray(0.0, dtype=jnp.float32),
                    option_discount=jnp.asarray(1.0, dtype=jnp.float32),
                    option_steps=jnp.asarray(0, dtype=jnp.int32),
                ),
            )
            applied = self.state_valid(state) & mask_valid & values_valid
            next_state = jax.lax.cond(
                applied,
                lambda _: proposed,
                lambda _: state,
                None,
            )
            return STOMPStartResult(
                state=next_state,
                primitive_action=jnp.where(
                    applied,
                    extended_action,
                    state.last_primitive_action,
                ),
            )
        is_starting_option = extended_action >= jnp.asarray(
            cfg.n_primitive_actions, dtype=jnp.int32
        )
        selected_option = jnp.clip(
            extended_action - cfg.n_primitive_actions,
            0,
            cfg.n_options - 1,
        )
        intra_action, key = _select_action_epsilon_greedy(
            state.option_policies.q_weights[selected_option],
            obs,
            key,
            cfg.epsilon_option,
            cfg.n_primitive_actions,
        )
        primitive_action = jnp.where(
            is_starting_option,
            intra_action,
            extended_action,
        )
        proposed = cast(
            STOMPState,
            state.replace(
                base_last_obs=obs,
                base_last_action=extended_action,
                last_primitive_action=primitive_action,
                rng_key=key,
                executing_option=jnp.where(
                    is_starting_option,
                    selected_option,
                    jnp.asarray(-1, dtype=jnp.int32),
                ),
                option_start_obs=jnp.where(
                    is_starting_option,
                    obs,
                    state.option_start_obs,
                ),
                option_last_intra_action=jnp.where(
                    is_starting_option,
                    intra_action,
                    state.option_last_intra_action,
                ),
                option_cumreward=jnp.asarray(0.0, dtype=jnp.float32),
                option_env_cumreward=jnp.asarray(0.0, dtype=jnp.float32),
                option_baseline_mass=jnp.asarray(0.0, dtype=jnp.float32),
                option_discount=jnp.asarray(1.0, dtype=jnp.float32),
                option_steps=jnp.asarray(0, dtype=jnp.int32),
            ),
        )
        applied = self.state_valid(state) & mask_valid & values_valid
        next_state = jax.lax.cond(applied, lambda _: proposed, lambda _: state, None)
        return STOMPStartResult(
            state=next_state,
            primitive_action=jnp.where(
                applied,
                primitive_action,
                state.last_primitive_action,
            ),
        )

    @staticmethod
    def current_primitive_action(state: STOMPState) -> Int[Array, ""]:
        """Return the primitive action currently selected for dispatch."""
        return state.last_primitive_action

    def _apply_option_model_planning(
        self,
        learner_state: MultiHeadMLPState,
        models: OptionModelsState,
        anchor_observation: Array,
        average_reward: Array,
        selection_words: Array,
        extended_action_mask: Array,
    ) -> tuple[MultiHeadMLPState, Array, Array]:
        """Apply a fixed number of Dyna backups from completed option models.

        Every imagined backup starts from ``anchor_observation``, which is a
        state encountered on the current real transition. Completed models are
        selected round-robin with a deterministic offset, so the compute
        budget is static and no extra policy RNG is consumed.

        The returned diagnostics are ``(applied_count, mean_td_error)``.
        Average reward is intentionally not part of the carry: imagined
        outcomes must not update the real experience reward-rate estimate.
        """
        cfg = self._config
        completed_mask = (models.n_completions > 0) & extended_action_mask[
            cfg.n_primitive_actions :
        ]
        n_completed = jnp.sum(completed_mask.astype(jnp.int32))
        completed_indices = jnp.nonzero(
            completed_mask,
            size=cfg.n_options,
            fill_value=0,
        )[0]
        backup_count = cfg.option_planning_backups_per_step

        def apply_backups(_: None) -> tuple[MultiHeadMLPState, Array, Array]:
            def backup_body(
                backup_idx: int,
                carry: tuple[MultiHeadMLPState, Array],
            ) -> tuple[MultiHeadMLPState, Array]:
                current_learner_state, td_sum = carry
                exact_phase = _lifetime_words_mod(
                    selection_words,
                    n_completed,
                ).astype(jnp.int32)
                completed_rank = jnp.mod(exact_phase + backup_idx, n_completed)
                model_idx = completed_indices[completed_rank]
                option_action = model_idx + cfg.n_primitive_actions

                predicted_delta = models.next_state_weights[model_idx] @ anchor_observation
                predicted_next = anchor_observation + predicted_delta
                next_q = self._base_learner.predict(current_learner_state, predicted_next)
                eligible_next_q = jnp.where(
                    extended_action_mask,
                    next_q,
                    -jnp.inf,
                )
                target = (
                    models.env_return_ema[model_idx]
                    - average_reward * models.baseline_mass_ema[model_idx]
                    + models.discount_ema[model_idx] * jnp.max(eligible_next_q)
                )
                targets = (
                    jnp.full(cfg.n_total_actions, jnp.nan, dtype=jnp.float32)
                    .at[option_action]
                    .set(target)
                )
                update_result = self._base_learner.update(
                    current_learner_state,
                    anchor_observation,
                    targets,
                )
                td_error = update_result.errors[option_action]
                return update_result.state, td_sum + td_error

            planned_state, td_sum = jax.lax.fori_loop(
                0,
                backup_count,
                backup_body,
                (learner_state, jnp.array(0.0, dtype=jnp.float32)),
            )
            count = jnp.asarray(backup_count, dtype=jnp.int32)
            mean_td = td_sum / jnp.asarray(backup_count, dtype=jnp.float32)
            return planned_state, count, mean_td

        def skip_backups(_: None) -> tuple[MultiHeadMLPState, Array, Array]:
            return (
                learner_state,
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0.0, dtype=jnp.float32),
            )

        return jax.lax.cond(
            n_completed > 0,
            apply_backups,
            skip_backups,
            None,
        )

    def _update_primitive_only(
        self,
        state: STOMPState,
        env_reward: Array,
        next_observation: Array,
        discount: Array | None,
        *,
        decision_observation: Array | None,
        execution_boundary: Array | bool,
        extended_action_mask: Array | None,
        enable_planning: bool,
        preselection_feature_reset_mask: Array | None,
    ) -> STOMPUpdateResult:
        """Apply one base-control transition when the option set is empty."""

        cfg = self._config
        if cfg.n_options != 0:
            raise RuntimeError("primitive-only update requires zero configured options")
        if enable_planning and cfg.option_planning_backups_per_step != 0:
            raise RuntimeError("primitive-only STOMP cannot execute option-model backups")

        if preselection_feature_reset_mask is None:
            reset_mask = jnp.zeros((cfg.observation_dim,), dtype=jnp.bool_)
        else:
            if cfg.base_hidden_sizes:
                raise ValueError("preselection feature reset requires a linear STOMP base learner")
            raw_reset_mask = jnp.asarray(preselection_feature_reset_mask)
            if raw_reset_mask.shape != (cfg.observation_dim,):
                raise ValueError(
                    "preselection_feature_reset_mask must have shape "
                    f"({cfg.observation_dim},), got {raw_reset_mask.shape}"
                )
            if raw_reset_mask.dtype != jnp.bool_:
                raise TypeError(
                    "preselection_feature_reset_mask must have dtype bool, "
                    f"got {raw_reset_mask.dtype}"
                )
            reset_mask = raw_reset_mask

        if extended_action_mask is None:
            action_mask = jnp.ones((cfg.n_primitive_actions,), dtype=jnp.bool_)
            action_mask_valid = jnp.asarray(True, dtype=jnp.bool_)
        else:
            raw_action_mask = jnp.asarray(extended_action_mask)
            if raw_action_mask.shape != (cfg.n_primitive_actions,):
                raise ValueError(
                    "extended_action_mask must have shape "
                    f"({cfg.n_primitive_actions},), got {raw_action_mask.shape}"
                )
            if raw_action_mask.dtype != jnp.bool_:
                raise TypeError(
                    f"extended_action_mask must have dtype bool, got {raw_action_mask.dtype}"
                )
            action_mask = raw_action_mask
            action_mask_valid = jnp.all(action_mask) & jnp.any(action_mask)

        bootstrap_obs = jnp.asarray(next_observation, dtype=jnp.float32).reshape(
            (cfg.observation_dim,)
        )
        decision_obs = (
            bootstrap_obs
            if decision_observation is None
            else jnp.asarray(decision_observation, dtype=jnp.float32).reshape(
                (cfg.observation_dim,)
            )
        )
        reward = jnp.asarray(env_reward, dtype=jnp.float32).reshape(())
        _ = jnp.asarray(execution_boundary, dtype=jnp.bool_).reshape(())
        input_values_valid = (
            jnp.isfinite(reward)
            & jnp.all(jnp.isfinite(bootstrap_obs))
            & jnp.all(jnp.isfinite(decision_obs))
            & action_mask_valid
        )
        if discount is None:
            primitive_discount = jnp.asarray(1.0, dtype=jnp.float32)
        else:
            supplied_discount = jnp.asarray(discount, dtype=jnp.float32).reshape(())
            valid_discount = (
                jnp.isfinite(supplied_discount)
                & (supplied_discount >= 0.0)
                & (supplied_discount <= 1.0)
            )
            primitive_discount = jnp.where(
                valid_discount,
                supplied_discount,
                jnp.asarray(jnp.nan, dtype=jnp.float32),
            )
            input_values_valid = input_values_valid & valid_discount

        outer_words, outer_capacity = _checked_lifetime_words_increment(state.step_words)
        outer_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        nested_counter_valid = _lifetime_counter_valid(
            state.base_learner_state.step_words,
            state.base_learner_state.step_count,
        )
        source_state_valid = self.state_valid(state)
        nested_updates_required = jnp.asarray(1, dtype=jnp.int32)
        expected_nested_words, nested_capacity = _checked_lifetime_words_advance(
            state.base_learner_state.step_words,
            nested_updates_required,
        )
        transaction_preflight = (
            source_state_valid
            & input_values_valid
            & outer_capacity
            & nested_counter_valid
            & nested_capacity
        )

        next_q_values = self._base_learner.predict(
            state.base_learner_state,
            bootstrap_obs,
        )
        eligible_next_q = jnp.where(action_mask, next_q_values, -jnp.inf)
        td_target = (
            reward - state.base_average_reward + primitive_discount * jnp.max(eligible_next_q)
        )
        targets = (
            jnp.full(
                cfg.n_primitive_actions,
                jnp.nan,
                dtype=jnp.float32,
            )
            .at[state.base_last_action]
            .set(td_target)
        )
        trace_adjusted_state = cast(
            MultiHeadMLPState,
            state.base_learner_state.replace(
                head_traces=jax.tree_util.tree_map(
                    lambda trace: primitive_discount * trace,
                    state.base_learner_state.head_traces,
                )
            ),
        )
        base_update = self._base_learner.update(
            trace_adjusted_state,
            state.base_last_obs,
            targets,
        )
        base_td = base_update.errors[state.base_last_action]
        new_average_reward = (
            state.base_average_reward
            + jnp.asarray(cfg.base_avg_reward_step_size, dtype=jnp.float32) * base_td
        )
        new_base_state = base_update.state
        new_option_policies = state.option_policies
        new_option_models = state.option_models
        if preselection_feature_reset_mask is not None:
            (
                new_base_state,
                new_option_policies,
                new_option_models,
            ) = _reset_linear_stomp_feature_axes(
                new_base_state,
                new_option_policies,
                new_option_models,
                reset_mask,
            )

        next_q_values = self._base_learner.predict(new_base_state, decision_obs)
        next_action, next_key = _select_action_epsilon_greedy_from_q_masked(
            next_q_values,
            state.rng_key,
            cfg.epsilon_base,
            action_mask,
        )
        proposed_state = STOMPState(
            base_learner_state=new_base_state,
            base_average_reward=new_average_reward,
            base_last_obs=decision_obs,
            base_last_action=next_action,
            last_primitive_action=next_action,
            rng_key=next_key,
            option_policies=new_option_policies,
            option_models=new_option_models,
            executing_option=jnp.asarray(-1, dtype=jnp.int32),
            option_start_obs=state.option_start_obs,
            option_last_intra_action=state.option_last_intra_action,
            option_cumreward=jnp.asarray(0.0, dtype=jnp.float32),
            option_env_cumreward=jnp.asarray(0.0, dtype=jnp.float32),
            option_baseline_mass=jnp.asarray(0.0, dtype=jnp.float32),
            option_discount=jnp.asarray(1.0, dtype=jnp.float32),
            option_steps=jnp.asarray(0, dtype=jnp.int32),
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=outer_words,
        )
        nested_post_matches = jnp.all(
            proposed_state.base_learner_state.step_words == expected_nested_words
        )
        proposed_state_valid = self.state_valid(proposed_state)
        transaction_applied = (
            transaction_preflight
            & base_update.update_applied
            & nested_post_matches
            & proposed_state_valid
        )
        new_state = jax.lax.cond(
            transaction_applied,
            lambda: proposed_state,
            lambda: state,
        )
        return STOMPUpdateResult(
            state=new_state,
            td_error=jnp.where(
                transaction_applied,
                base_td,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            average_reward=jnp.where(
                transaction_applied,
                new_average_reward,
                state.base_average_reward,
            ),
            primitive_action=jnp.where(
                transaction_applied,
                next_action,
                state.last_primitive_action,
            ),
            executing_option=jnp.asarray(-1, dtype=jnp.int32),
            option_terminated=jnp.asarray(False, dtype=jnp.bool_),
            pseudo_reward=jnp.asarray(0.0, dtype=jnp.float32),
            option_importance_ratio=jnp.asarray(0.0, dtype=jnp.float32),
            planning_backups=jnp.asarray(0, dtype=jnp.int32),
            planning_td_error=jnp.asarray(0.0, dtype=jnp.float32),
            pre_step_words=state.step_words,
            post_step_words=new_state.step_words,
            inputs_valid=input_values_valid,
            lifetime_counter_valid=outer_counter_valid,
            lifetime_capacity_available=outer_capacity,
            nested_lifetime_counter_valid=nested_counter_valid,
            nested_lifetime_capacity_available=nested_capacity,
            nested_updates_required=nested_updates_required,
            nested_updates_applied=jnp.where(
                transaction_applied,
                nested_updates_required,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            proposed_state_valid=proposed_state_valid,
            update_applied=transaction_applied,
        )

    @functools.partial(
        jax.jit,
        static_argnums=(0,),
        static_argnames=("enable_planning",),
    )
    def update(
        self,
        state: STOMPState,
        env_reward: Array,
        next_observation: Array,
        discount: Array | None = None,
        *,
        decision_observation: Array | None = None,
        execution_boundary: Array | bool = False,
        extended_action_mask: Array | None = None,
        enable_planning: bool = True,
        preselection_feature_reset_mask: Array | None = None,
    ) -> STOMPUpdateResult:
        """Process one real-time transition update.

        The function:

        1. Determines whether an option is currently executing.
        2. If executing: advances the intra-option policy and checks termination.
           On termination, updates the option outcome model (pseudo-reward
           outcome) and the base Q-function using the environment reward
           accumulated across the option.
        3. If not executing: updates the base Q-function and selects the next
           extended action (primitive or option).
        4. Optionally applies fixed-budget planning. Callers replaying an
           imagined transition must pass ``enable_planning=False`` so option
           models only plan from real observation anchors.
        5. Returns diagnostics for logging.

        ``next_observation`` is the bootstrap observation used by every
        learning target. ``decision_observation`` is the state from which the
        next action is selected; it defaults to ``next_observation`` for
        continuing callers. This distinction matters for autoreset wrappers,
        which return a final observation for bootstrapping and a reset
        observation for the next decision.

        ``execution_boundary`` interrupts an active option lifecycle after
        this transition without changing its Bellman discount. It is intended
        for censored episode truncations: the observed transition updates the
        intra-option policy and the base option value, then option traces are
        cleared, but the partial execution is not recorded as a completed
        option-model outcome.

        ``discount`` is the effective continuation multiplier for this
        transition.  Explicit callers should always supply it.  ``None`` is a
        compatibility mode: primitive and intra-option bootstraps use one,
        while option-return accumulation uses ``config.option_gamma`` exactly
        as releases before the explicit transition contract did.

        ``extended_action_mask`` is an opt-in behavior-eligibility boundary
        over primitive actions followed by options. Every primitive action
        must remain true. The mask affects a newly selected extended action
        plus real-transition and option-model-planning bootstraps. Inactive
        option models are not selected for planning. An already executing
        option continues until its normal lifecycle boundary. ``None`` is the
        exact legacy all-actions-eligible path.

        ``preselection_feature_reset_mask`` is the narrow causal recycling
        hook.  It is valid only for the linear base learner.  Selected feature
        axes are scrubbed from all linear STOMP consumers after this real
        transition's learning and before the sole next-action selection.
        """
        cfg = self._config
        if cfg.n_options == 0:
            return self._update_primitive_only(
                state,
                env_reward,
                next_observation,
                discount,
                decision_observation=decision_observation,
                execution_boundary=execution_boundary,
                extended_action_mask=extended_action_mask,
                enable_planning=enable_planning,
                preselection_feature_reset_mask=preselection_feature_reset_mask,
            )
        spec = self._spec_arrays
        if preselection_feature_reset_mask is None:
            reset_mask = jnp.zeros((cfg.observation_dim,), dtype=jnp.bool_)
        else:
            if cfg.base_hidden_sizes:
                raise ValueError("preselection feature reset requires a linear STOMP base learner")
            raw_reset_mask = jnp.asarray(preselection_feature_reset_mask)
            if raw_reset_mask.shape != (cfg.observation_dim,):
                raise ValueError(
                    "preselection_feature_reset_mask must have shape "
                    f"({cfg.observation_dim},), got {raw_reset_mask.shape}"
                )
            if raw_reset_mask.dtype != jnp.bool_:
                raise TypeError(
                    "preselection_feature_reset_mask must have dtype bool, "
                    f"got {raw_reset_mask.dtype}"
                )
            reset_mask = raw_reset_mask
        if extended_action_mask is None:
            action_mask = jnp.ones((cfg.n_total_actions,), dtype=jnp.bool_)
            action_mask_valid = jnp.asarray(True, dtype=jnp.bool_)
        else:
            raw_action_mask = jnp.asarray(extended_action_mask)
            if raw_action_mask.shape != (cfg.n_total_actions,):
                raise ValueError(
                    "extended_action_mask must have shape "
                    f"({cfg.n_total_actions},), got {raw_action_mask.shape}"
                )
            if raw_action_mask.dtype != jnp.bool_:
                raise TypeError(
                    f"extended_action_mask must have dtype bool, got {raw_action_mask.dtype}"
                )
            action_mask = raw_action_mask
            action_mask_valid = jnp.all(action_mask[: cfg.n_primitive_actions]) & jnp.any(
                action_mask
            )
        bootstrap_obs = jnp.asarray(next_observation, dtype=jnp.float32).reshape(
            (cfg.observation_dim,)
        )
        decision_obs = (
            bootstrap_obs
            if decision_observation is None
            else jnp.asarray(decision_observation, dtype=jnp.float32).reshape(
                (cfg.observation_dim,)
            )
        )
        boundary = jnp.asarray(execution_boundary, dtype=jnp.bool_).reshape(())
        reward = jnp.asarray(env_reward, dtype=jnp.float32).reshape(())
        input_values_valid = (
            jnp.isfinite(reward)
            & jnp.all(jnp.isfinite(bootstrap_obs))
            & jnp.all(jnp.isfinite(decision_obs))
            & action_mask_valid
        )
        if discount is None:
            primitive_discount = jnp.array(1.0, dtype=jnp.float32)
            intra_option_discount = jnp.array(1.0, dtype=jnp.float32)
            option_step_discount = jnp.asarray(cfg.option_gamma, dtype=jnp.float32)
            environmental_termination = jnp.array(False)
        else:
            supplied_discount = jnp.asarray(discount, dtype=jnp.float32).reshape(())
            valid_discount = (
                jnp.isfinite(supplied_discount)
                & (supplied_discount >= 0.0)
                & (supplied_discount <= 1.0)
            )
            # Shape errors fail while tracing; value errors become non-finite
            # rather than being silently clipped under JIT. PrototypeAgent's
            # eager boundary raises a ValueError before reaching this path.
            supplied_discount = jnp.where(
                valid_discount,
                supplied_discount,
                jnp.array(jnp.nan, dtype=jnp.float32),
            )
            primitive_discount = supplied_discount
            intra_option_discount = supplied_discount
            option_step_discount = supplied_discount
            environmental_termination = supplied_discount <= 0.0
            input_values_valid = input_values_valid & valid_discount

        outer_words, outer_capacity = _checked_lifetime_words_increment(state.step_words)
        outer_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        nested_counter_valid = _lifetime_counter_valid(
            state.base_learner_state.step_words,
            state.base_learner_state.step_count,
        )
        source_state_valid = self.state_valid(state)

        is_executing = state.executing_option >= 0
        option_idx = jnp.clip(
            state.executing_option,
            jnp.array(0, dtype=jnp.int32),
            jnp.asarray(cfg.n_options - 1, dtype=jnp.int32),
        )

        # Compute pseudo-reward for the currently-executing (or notional) option
        pseudo_r = compute_pseudo_reward(spec, option_idx, bootstrap_obs)
        target_epsilon = (
            cfg.epsilon_option if cfg.option_target_epsilon is None else cfg.option_target_epsilon
        )
        option_importance_ratio = _clipped_epsilon_greedy_importance_ratio(
            state.option_policies.q_weights[option_idx],
            state.base_last_obs,
            state.option_last_intra_action,
            behavior_epsilon=cfg.epsilon_option,
            target_epsilon=target_epsilon,
            clip=cfg.option_importance_clip,
        )

        # Option termination check
        new_option_steps = _saturating_int32_counter_increment(state.option_steps)
        option_completes = (
            check_option_terminated(spec, option_idx, bootstrap_obs, new_option_steps)
            | environmental_termination
        )
        option_lifecycle_ends = option_completes | boundary
        should_update_model = is_executing & option_completes
        should_update_base = (~is_executing) | (is_executing & option_lifecycle_ends)
        completed_after = (state.option_models.n_completions > 0) | (
            jnp.arange(cfg.n_options, dtype=jnp.int32) == option_idx
        ) & should_update_model
        has_completed_model_after = jnp.any(
            completed_after & action_mask[cfg.n_primitive_actions :]
        )
        if enable_planning and cfg.option_planning_backups_per_step > 0:
            planning_updates_required = jnp.where(
                has_completed_model_after,
                jnp.asarray(
                    cfg.option_planning_backups_per_step,
                    dtype=jnp.int32,
                ),
                jnp.asarray(0, dtype=jnp.int32),
            )
        else:
            planning_updates_required = jnp.asarray(0, dtype=jnp.int32)
        nested_updates_required = should_update_base.astype(jnp.int32) + planning_updates_required
        expected_nested_words, nested_capacity = _checked_lifetime_words_advance(
            state.base_learner_state.step_words,
            nested_updates_required,
        )
        transaction_preflight = (
            source_state_valid
            & input_values_valid
            & outer_capacity
            & nested_counter_valid
            & nested_capacity
        )

        # --- Intra-option policy update (only active when executing) ---
        # Gated on is_executing: idle steps must not pollute option 0 (the
        # clamped index) with spurious pseudo-reward updates.
        def do_intra_update(
            _: None,
        ) -> tuple[IntraOptionPoliciesState, Array, Bool[Array, ""]]:
            return _update_intra_option_policy(
                state.option_policies,
                option_idx,
                state.base_last_obs,
                state.option_last_intra_action,
                pseudo_r,
                bootstrap_obs,
                option_completes,
                intra_option_discount,
                step_size=cfg.option_step_size,
                avg_reward_step_size=cfg.option_avg_reward_step_size,
                trace_decay=cfg.option_trace_decay,
                n_primitive_actions=cfg.n_primitive_actions,
                importance_ratio=option_importance_ratio,
            )

        def skip_intra_update(
            _: None,
        ) -> tuple[IntraOptionPoliciesState, Array, Bool[Array, ""]]:
            return (
                state.option_policies,
                jnp.array(0.0, dtype=jnp.float32),
                jnp.asarray(True, dtype=jnp.bool_),
            )

        new_option_policies, option_td, intra_update_applied = jax.lax.cond(
            is_executing, do_intra_update, skip_intra_update, None
        )
        # A censored boundary retains the positive bootstrap above, so the
        # boundary sample contributes to both the weight and trace update.
        # Clear the stored trace only afterwards to prevent eligibility from
        # leaking into the reset episode or a newly selected option execution.
        clear_option_trace = is_executing & option_lifecycle_ends
        active_traces = new_option_policies.traces[option_idx]
        new_option_policies = cast(
            IntraOptionPoliciesState,
            new_option_policies.replace(
                traces=new_option_policies.traces.at[option_idx].set(
                    jnp.where(
                        clear_option_trace,
                        jnp.zeros_like(active_traces),
                        active_traces,
                    )
                )
            ),
        )

        # Accumulate option trajectory stats. Pseudo-reward feeds only the
        # subtask learner/model. The base control learner receives the
        # discounted environment return:
        #     r_1 + d_1*r_2 + ... + (Π_{k<T} d_k)*r_T.
        # Before this transition, ``option_discount`` is the product of all
        # preceding transition discounts in the current option.
        new_option_cumreward = state.option_cumreward + pseudo_r
        new_option_env_cumreward = jnp.where(
            is_executing,
            state.option_env_cumreward + state.option_discount * reward,
            state.option_env_cumreward,
        )
        new_option_baseline_mass = jnp.where(
            is_executing,
            state.option_baseline_mass + state.option_discount,
            state.option_baseline_mass,
        )
        new_option_discount = state.option_discount * option_step_discount

        # --- Option model update (only on termination while executing) ---
        # A pure execution boundary is a censored option trajectory, not an
        # observed option-model completion. If the subtask also completes
        # naturally (or the environment terminates), preserve completion.
        def do_update_model(_: None) -> OptionModelsState:
            return _update_option_model(
                state.option_models,
                option_idx,
                state.option_start_obs,
                new_option_cumreward,
                new_option_env_cumreward,
                jnp.asarray(new_option_steps, dtype=jnp.float32),
                new_option_baseline_mass,
                new_option_discount,
                bootstrap_obs,
                model_decay=cfg.option_model_decay,
                model_step_size=cfg.option_model_step_size,
            )

        def skip_update_model(_: None) -> OptionModelsState:
            return state.option_models

        new_option_models = jax.lax.cond(
            should_update_model, do_update_model, skip_update_model, None
        )

        # --- Base Q-function update ---
        # Always grounded in real environment reward: the one-step reward for
        # primitive actions, the reward accumulated across the option on
        # termination.  Pseudo-reward never enters here — option values must
        # stay in task-reward units to support reward-maximizing planning.
        base_reward = jnp.where(
            is_executing & option_lifecycle_ends,
            new_option_env_cumreward,
            reward,
        )
        # Discounted differential semi-MDP correction: the average-reward
        # baseline is weighted by the same γ powers as the environment return.
        # At unit discounts this mass equals raw duration T_o. Primitive
        # transitions use mass=1 and the supplied one-step discount.
        base_baseline_mass = jnp.where(
            is_executing & option_lifecycle_ends,
            new_option_baseline_mass,
            jnp.array(1.0, dtype=jnp.float32),
        )
        base_discount = jnp.where(
            is_executing & option_lifecycle_ends,
            new_option_discount,
            primitive_discount,
        )
        # Only update base Q on: (a) primitive steps, or (b) option termination
        n_total = cfg.n_total_actions
        beta = jnp.asarray(cfg.base_avg_reward_step_size, dtype=jnp.float32)
        # The base extended action was selected at the option's start state.
        # Keep ``base_last_obs`` free to track consecutive primitive
        # observations for the intra-option learner, but restore semi-MDP
        # credit assignment at option termination by updating from the stored
        # start observation. Primitive actions still update from their most
        # recent observation.
        base_update_obs = jnp.where(
            is_executing & option_lifecycle_ends,
            state.option_start_obs,
            state.base_last_obs,
        )

        def do_base_update(
            _: None,
        ) -> tuple[MultiHeadMLPState, Array, Array, Array]:
            next_q_vals = self._base_learner.predict(state.base_learner_state, bootstrap_obs)
            eligible_next_q = jnp.where(action_mask, next_q_vals, -jnp.inf)
            max_next_q = base_discount * jnp.max(eligible_next_q)
            td_target = base_reward - state.base_average_reward * base_baseline_mass + max_next_q
            targets = (
                jnp.full(n_total, jnp.nan, dtype=jnp.float32)
                .at[state.base_last_action]
                .set(td_target)
            )
            trace_adjusted_state = cast(
                MultiHeadMLPState,
                state.base_learner_state.replace(
                    head_traces=jax.tree_util.tree_map(
                        lambda trace: base_discount * trace,
                        state.base_learner_state.head_traces,
                    )
                ),
            )
            result = self._base_learner.update(trace_adjusted_state, base_update_obs, targets)
            td_err = result.errors[state.base_last_action]
            new_avg_reward = state.base_average_reward + beta * td_err
            return result.state, new_avg_reward, td_err, result.update_applied

        def skip_base_update(
            _: None,
        ) -> tuple[MultiHeadMLPState, Array, Array, Array]:
            prev_q = self._base_learner.predict(state.base_learner_state, state.base_last_obs)
            next_q = self._base_learner.predict(state.base_learner_state, bootstrap_obs)
            eligible_next_q = jnp.where(action_mask, next_q, -jnp.inf)
            td = primitive_discount * jnp.max(eligible_next_q) - prev_q[state.base_last_action]
            return (
                state.base_learner_state,
                state.base_average_reward,
                td,
                jnp.asarray(True, dtype=jnp.bool_),
            )

        (
            new_base_learner_state,
            new_avg_r,
            base_td,
            real_base_update_applied,
        ) = jax.lax.cond(should_update_base, do_base_update, skip_base_update, None)

        # --- Fixed-budget option-model planning ---
        # The zero-backup default is a Python-level static branch: it consumes
        # no RNG and executes no additional learner operations.
        if enable_planning and cfg.option_planning_backups_per_step > 0:
            (
                new_base_learner_state,
                planning_backups,
                planning_td_error,
            ) = self._apply_option_model_planning(
                new_base_learner_state,
                new_option_models,
                bootstrap_obs,
                new_avg_r,
                state.step_words,
                action_mask,
            )
        else:
            planning_backups = jnp.array(0, dtype=jnp.int32)
            planning_td_error = jnp.array(0.0, dtype=jnp.float32)

        if preselection_feature_reset_mask is not None:
            (
                new_base_learner_state,
                new_option_policies,
                new_option_models,
            ) = _reset_linear_stomp_feature_axes(
                new_base_learner_state,
                new_option_policies,
                new_option_models,
                reset_mask,
            )

        # --- Select next extended action ---
        # After primitive or option termination: select from extended action space.
        # During option execution (not terminating): use intra-option policy.
        key = state.rng_key
        key, ext_key, intra_key = jr.split(key, 3)

        ext_q_vals = self._base_learner.predict(new_base_learner_state, decision_obs)
        extended_action, _ = _select_action_epsilon_greedy_from_q_masked(
            ext_q_vals,
            ext_key,
            cfg.epsilon_base,
            action_mask,
        )
        next_select_extended = (~is_executing) | (is_executing & option_lifecycle_ends)
        selected_option = jnp.clip(
            extended_action - cfg.n_primitive_actions,
            0,
            cfg.n_options - 1,
        )
        # A continuing option uses its current policy. If this transition
        # starts a new option, sample from that selected option's policy
        # rather than from the idle clamped index (option 0).
        intra_policy_idx = jnp.where(
            is_executing & (~option_lifecycle_ends),
            option_idx,
            selected_option,
        )
        intra_action, _ = _select_action_epsilon_greedy(
            new_option_policies.q_weights[intra_policy_idx],
            decision_obs,
            intra_key,
            cfg.epsilon_option,
            cfg.n_primitive_actions,
        )

        # The actual primitive action dispatched to the environment:
        # If primitive extended action: use extended_action directly.
        # If option extended action: use intra-option policy action.
        new_executing_option = jnp.where(
            is_executing & (~option_lifecycle_ends),
            option_idx,
            jnp.where(
                next_select_extended
                & (extended_action >= jnp.asarray(cfg.n_primitive_actions, jnp.int32)),
                selected_option,
                jnp.array(-1, dtype=jnp.int32),
            ),
        )
        is_starting_option = next_select_extended & (
            extended_action >= jnp.asarray(cfg.n_primitive_actions, jnp.int32)
        )

        # Primitive action sent to environment
        primitive_action = jnp.where(
            is_starting_option | (is_executing & (~option_lifecycle_ends)),
            intra_action,
            extended_action,
        )
        primitive_action = jnp.minimum(
            primitive_action,
            jnp.asarray(cfg.n_primitive_actions - 1, dtype=jnp.int32),
        )

        # Reset option tracking on termination or new option start
        new_option_start_obs = jnp.where(is_starting_option, decision_obs, state.option_start_obs)
        new_option_cumreward = jnp.where(
            (is_executing & option_lifecycle_ends) | is_starting_option,
            jnp.array(0.0, dtype=jnp.float32),
            new_option_cumreward,
        )
        new_option_env_cumreward = jnp.where(
            (is_executing & option_lifecycle_ends) | is_starting_option,
            jnp.array(0.0, dtype=jnp.float32),
            new_option_env_cumreward,
        )
        new_option_baseline_mass = jnp.where(
            (is_executing & option_lifecycle_ends) | is_starting_option,
            jnp.array(0.0, dtype=jnp.float32),
            new_option_baseline_mass,
        )
        new_option_discount = jnp.where(
            (is_executing & option_lifecycle_ends) | is_starting_option,
            jnp.array(1.0, dtype=jnp.float32),
            new_option_discount,
        )
        new_option_steps = jnp.where(
            (is_executing & option_lifecycle_ends) | is_starting_option,
            jnp.array(0, dtype=jnp.int32),
            new_option_steps,
        )

        proposed_state = STOMPState(
            base_learner_state=new_base_learner_state,
            base_average_reward=new_avg_r,
            base_last_obs=decision_obs,
            base_last_action=jnp.where(
                next_select_extended, extended_action, state.base_last_action
            ),
            last_primitive_action=primitive_action,
            rng_key=key,
            option_policies=new_option_policies,
            option_models=new_option_models,
            executing_option=new_executing_option,
            option_start_obs=new_option_start_obs,
            option_last_intra_action=jnp.where(
                is_starting_option | (is_executing & (~option_lifecycle_ends)),
                intra_action,
                state.option_last_intra_action,
            ),
            option_cumreward=new_option_cumreward,
            option_env_cumreward=new_option_env_cumreward,
            option_baseline_mass=new_option_baseline_mass,
            option_discount=new_option_discount,
            option_steps=new_option_steps,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=outer_words,
        )
        nested_post_matches = jnp.all(
            proposed_state.base_learner_state.step_words == expected_nested_words
        )
        proposed_state_valid = self.state_valid(proposed_state)
        transaction_applied = (
            transaction_preflight
            & intra_update_applied
            & real_base_update_applied
            & nested_post_matches
            & proposed_state_valid
        )
        new_state = jax.lax.cond(
            transaction_applied,
            lambda: proposed_state,
            lambda: state,
        )
        committed_base_td = jnp.where(
            transaction_applied,
            base_td,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        committed_planning_backups = jnp.where(
            transaction_applied,
            planning_backups,
            jnp.asarray(0, dtype=jnp.int32),
        )
        committed_planning_td_error = jnp.where(
            transaction_applied,
            planning_td_error,
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return STOMPUpdateResult(
            state=new_state,
            td_error=committed_base_td,
            average_reward=jnp.where(
                transaction_applied,
                new_avg_r,
                state.base_average_reward,
            ),
            primitive_action=jnp.where(
                transaction_applied,
                primitive_action,
                state.last_primitive_action,
            ),
            executing_option=jnp.where(
                transaction_applied,
                new_executing_option,
                state.executing_option,
            ),
            option_terminated=transaction_applied & is_executing & option_lifecycle_ends,
            pseudo_reward=jnp.where(
                transaction_applied & is_executing,
                pseudo_r,
                jnp.array(0.0, dtype=jnp.float32),
            ),
            option_importance_ratio=jnp.where(
                transaction_applied,
                option_importance_ratio,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            planning_backups=committed_planning_backups,
            planning_td_error=committed_planning_td_error,
            pre_step_words=state.step_words,
            post_step_words=new_state.step_words,
            inputs_valid=input_values_valid,
            lifetime_counter_valid=outer_counter_valid,
            lifetime_capacity_available=outer_capacity,
            nested_lifetime_counter_valid=nested_counter_valid,
            nested_lifetime_capacity_available=nested_capacity,
            nested_updates_required=nested_updates_required,
            nested_updates_applied=jnp.where(
                transaction_applied,
                nested_updates_required,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            proposed_state_valid=proposed_state_valid,
            update_applied=transaction_applied,
        )

    def scan(
        self,
        state: STOMPState,
        env_rewards: Array,
        next_observations: Array,
        discounts: Array | None = None,
        *,
        decision_observations: Array | None = None,
        execution_boundaries: Array | None = None,
        extended_action_masks: Array | None = None,
    ) -> STOMPArrayResult:
        """Run STOMP over transition arrays via scan.

        Supplying ``discounts`` selects the explicit transition contract.
        Omitting it preserves the historical ``option_gamma`` behavior.
        ``decision_observations`` and ``execution_boundaries`` provide the
        batched autoreset-boundary split accepted by :meth:`update`.
        ``extended_action_masks`` threads the per-transition action-eligibility
        contract, including bootstrap and planning eligibility.
        """

        def step_fn(
            carry: STOMPState,
            inputs: tuple[Array, Array, Array, Array, Array, Array],
        ) -> tuple[STOMPState, tuple[Array, ...]]:
            (
                reward,
                next_obs,
                transition_discount,
                decision_obs,
                execution_boundary,
                extended_action_mask,
            ) = inputs
            result = self.update(
                carry,
                reward,
                next_obs,
                transition_discount if discounts is not None else None,
                decision_observation=decision_obs,
                execution_boundary=execution_boundary,
                extended_action_mask=(
                    extended_action_mask if extended_action_masks is not None else None
                ),
            )
            return result.state, (
                result.td_error,
                result.average_reward,
                result.primitive_action,
                result.executing_option,
                result.option_terminated,
                result.pseudo_reward,
                result.option_importance_ratio,
                result.planning_backups,
                result.planning_td_error,
                result.pre_step_words,
                result.post_step_words,
                result.inputs_valid,
                result.lifetime_counter_valid,
                result.lifetime_capacity_available,
                result.nested_lifetime_counter_valid,
                result.nested_lifetime_capacity_available,
                result.nested_updates_required,
                result.nested_updates_applied,
                result.proposed_state_valid,
                result.update_applied,
            )

        if discounts is None:
            scan_discounts = jnp.ones_like(env_rewards, dtype=jnp.float32)
        else:
            scan_discounts = jnp.asarray(discounts, dtype=jnp.float32)
        scan_decision_observations = (
            next_observations
            if decision_observations is None
            else jnp.asarray(decision_observations, dtype=jnp.float32)
        )
        scan_execution_boundaries = (
            jnp.zeros_like(env_rewards, dtype=jnp.bool_)
            if execution_boundaries is None
            else jnp.asarray(execution_boundaries, dtype=jnp.bool_)
        )
        scan_extended_action_masks = (
            jnp.ones(
                (env_rewards.shape[0], self._config.n_total_actions),
                dtype=jnp.bool_,
            )
            if extended_action_masks is None
            else jnp.asarray(extended_action_masks)
        )
        if scan_extended_action_masks.shape != (
            env_rewards.shape[0],
            self._config.n_total_actions,
        ):
            raise ValueError(
                "extended_action_masks must have shape "
                f"({env_rewards.shape[0]}, {self._config.n_total_actions})"
            )
        if scan_extended_action_masks.dtype != jnp.bool_:
            raise TypeError("extended_action_masks must have dtype bool")

        (
            final_state,
            (
                td_errors,
                average_rewards,
                primitive_actions,
                executing_options,
                option_terminations,
                pseudo_rewards,
                option_importance_ratios,
                planning_backups,
                planning_td_errors,
                pre_step_words,
                post_step_words,
                inputs_valid,
                lifetime_counter_valid,
                lifetime_capacity_available,
                nested_lifetime_counter_valid,
                nested_lifetime_capacity_available,
                nested_updates_required,
                nested_updates_applied,
                proposed_state_valid,
                update_applied,
            ),
        ) = jax.lax.scan(
            step_fn,
            state,
            (
                env_rewards,
                next_observations,
                scan_discounts,
                scan_decision_observations,
                scan_execution_boundaries,
                scan_extended_action_masks,
            ),
        )
        return STOMPArrayResult(
            state=final_state,
            td_errors=td_errors,
            average_rewards=average_rewards,
            primitive_actions=primitive_actions,
            executing_options=executing_options,
            option_terminations=option_terminations,
            pseudo_rewards=pseudo_rewards,
            option_importance_ratios=option_importance_ratios,
            planning_backups=planning_backups,
            planning_td_errors=planning_td_errors,
            pre_step_words=pre_step_words,
            post_step_words=post_step_words,
            inputs_valid=inputs_valid,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            nested_lifetime_counter_valid=nested_lifetime_counter_valid,
            nested_lifetime_capacity_available=nested_lifetime_capacity_available,
            nested_updates_required=nested_updates_required,
            nested_updates_applied=nested_updates_applied,
            proposed_state_valid=proposed_state_valid,
            update_applied=update_applied,
        )


def subtasks_from_feature_scores(
    feature_scores: Float[Array, " feature_dim"] | list[float],
    *,
    top_k: int = 2,
    threshold: float = 0.5,
    pseudo_reward_scale: float = 1.0,
    max_option_steps: int = 16,
    min_score: float = 0.0,
) -> list[SubtaskSpec]:
    """Create SubtaskSpecs for the top-K highest-scoring features.

    This is the auto-discovery pathway for Step 10 STOMP: instead of
    hand-specifying subtasks, the caller supplies per-feature relevance scores
    and this function converts the top-ranked features into ``SubtaskSpec``
    objects.

    The feature scores may come from any source:
    - ``jnp.sum(relevance.weight_relevance, axis=0)`` for path-norm relevance
    - Per-head weight norms for a specific prediction target
    - Domain-specific utility signal

    Args:
        feature_scores: 1-D array or list of per-feature importance scores.
            Higher score = more relevant feature → higher-priority subtask.
        top_k: Number of subtasks to create. Selects features with the
            ``top_k`` highest scores.
        threshold: Pseudo-reward threshold for subtask completion. The option
            terminates when ``pseudo_reward_scale * obs[feature_index] >= threshold``.
        pseudo_reward_scale: Multiplier for the feature value in the
            pseudo-reward signal.
        max_option_steps: Maximum primitive steps per option execution.
        min_score: Features with score below this value are excluded even if
            they would otherwise be in the top-K. Set to 0.0 to keep all.

    Returns:
        List of up to ``top_k`` :class:`SubtaskSpec` objects sorted by
        descending feature score. May be shorter than ``top_k`` if fewer
        features exceed ``min_score``.

    """
    import numpy as _np

    scores = _np.asarray(feature_scores, dtype=_np.float32)
    eligible = [i for i in range(len(scores)) if float(scores[i]) >= min_score]
    eligible.sort(key=lambda i: float(scores[i]), reverse=True)
    selected = eligible[:top_k]

    return [
        SubtaskSpec(
            feature_index=int(i),
            threshold=threshold,
            pseudo_reward_scale=pseudo_reward_scale,
            max_option_steps=max_option_steps,
        )
        for i in selected
    ]


# ---------------------------------------------------------------------------
# Checkpoint payloads and STOMP-state migration
# ---------------------------------------------------------------------------

#: Option-model fields added by the environment-return/duration/baseline-mass
#: expansion.  Pre-expansion checkpoints lack exactly these keys.
STOMP_OPTION_MODEL_EXPANSION_FIELDS = (
    "env_return_ema",
    "duration_ema",
    "baseline_mass_ema",
)

#: Top-level ``STOMPState`` accumulators added by the same expansion.
STOMP_STATE_EXPANSION_FIELDS = (
    "option_env_cumreward",
    "option_baseline_mass",
)

#: Exact-lifetime fields introduced by ``STOMP_STATE_SCHEMA`` v2.
STOMP_STATE_LIFETIME_FIELDS = ("step_words",)


def measure_stomp_state_nbytes(state: STOMPState) -> int:
    """Measure STOMP-owned arrays plus its nested base-learner arrays."""

    persistent_learner = state.base_learner_state.replace(
        birth_timestamp=0.0,
        uptime_s=0.0,
    )
    persistent_state = state.replace(base_learner_state=persistent_learner)
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(persistent_state)
        if isinstance(leaf, Array)
    )


def measure_stomp_wrapper_state_nbytes(state: STOMPState) -> int:
    """Measure STOMP-owned arrays, excluding the nested base learner."""

    persistent_learner = state.base_learner_state.replace(
        birth_timestamp=0.0,
        uptime_s=0.0,
    )
    nested_nbytes = sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(persistent_learner)
        if isinstance(leaf, Array)
    )
    return measure_stomp_state_nbytes(state) - nested_nbytes


def stomp_lifetime_counter_nbytes() -> int:
    """Return bytes for STOMP's primitive and base-update exact clocks."""

    return 2 * STOMP_LIFETIME_COUNTER_NBYTES


def stomp_state_to_checkpoint_payload(state: STOMPState) -> dict[str, Any]:
    """Flatten a :class:`STOMPState` into a field-keyed checkpoint payload.

    The payload maps every ``STOMPState`` field name to its value, with
    ``option_models`` and ``option_policies`` expanded into nested field-keyed
    dictionaries.  ``load_stomp_state_with_migration`` accepts exactly this
    layout, so ``load(to_payload(state))`` round-trips losslessly.

    Args:
        state: The STOMP agent state to flatten.

    Returns:
        Nested plain-dict payload suitable for generic checkpointers.
    """
    payload: dict[str, Any] = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(STOMPState)  # type: ignore[arg-type]
    }
    payload["option_models"] = {
        field.name: getattr(state.option_models, field.name)
        for field in dataclasses.fields(OptionModelsState)  # type: ignore[arg-type]
    }
    payload["option_policies"] = {
        field.name: getattr(state.option_policies, field.name)
        for field in dataclasses.fields(IntraOptionPoliciesState)  # type: ignore[arg-type]
    }
    return payload


def _sub_payload_as_dict(value: Any, name: str, cls: type) -> dict[str, Any]:
    """Normalize a nested payload entry to a plain field-keyed dict."""
    if isinstance(value, cls):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(cls)}
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(
        f"'{name}' must be a field-keyed dict or {cls.__name__}, got {type(value).__name__}"
    )


def load_stomp_state_with_migration(payload: dict[str, Any]) -> STOMPState:
    """Load a STOMP checkpoint payload, migrating pre-expansion templates.

    Pre-expansion checkpoints were written before the option models tracked
    the discounted environment return, primitive-step duration, and discounted
    baseline mass, and before ``STOMPState`` carried the matching in-flight
    accumulators.  This loader detects that old template by the missing keys
    and fills principled defaults:

    * ``option_models.env_return_ema``, ``option_models.duration_ema``, and
      ``option_models.baseline_mass_ema`` become zeros of shape
      ``(n_options,)`` — the same "no completed option executions observed"
      prior that :meth:`STOMPAgent.init` uses, so the EMAs warm-start exactly
      as on a fresh agent.
    * ``option_env_cumreward`` and ``option_baseline_mass`` become scalar
      zeros — any option mid-flight at checkpoint time restarts its
      environment-return and baseline-mass accumulation, since the old
      format never recorded those quantities.

    Everything present in the payload is required: missing pre-expansion
    fields or unknown keys raise ``ValueError`` (fail-closed) rather than
    being silently defaulted or dropped.

    Args:
        payload: Field-keyed payload as produced by
            :func:`stomp_state_to_checkpoint_payload` (new format) or its
            pre-expansion equivalent (old format).

    Returns:
        A fully populated :class:`STOMPState`.

    Raises:
        ValueError: If any non-expansion field is missing or the payload
            contains keys that are not ``STOMPState`` fields.
    """
    data = dict(payload)
    state_field_names = {
        field.name
        for field in dataclasses.fields(STOMPState)  # type: ignore[arg-type]
    }
    unknown = sorted(set(data) - state_field_names)
    if unknown:
        raise ValueError(f"Unknown STOMPState checkpoint fields: {unknown}")
    missing_state_fields = state_field_names - set(data)
    pre_lifetime_fields = set(STOMP_STATE_LIFETIME_FIELDS)
    pre_expansion_fields = set(STOMP_STATE_EXPANSION_FIELDS) | pre_lifetime_fields
    allowed_missing_state_fields: tuple[set[str], ...] = (
        set(),
        pre_lifetime_fields,
        pre_expansion_fields,
    )
    if missing_state_fields not in allowed_missing_state_fields:
        raise ValueError(
            "STOMP checkpoint state-field manifest is not a known schema; "
            f"missing={sorted(missing_state_fields)}"
        )

    models = _sub_payload_as_dict(data.pop("option_models"), "option_models", OptionModelsState)
    model_field_names = {
        field.name
        for field in dataclasses.fields(OptionModelsState)  # type: ignore[arg-type]
    }
    unknown_models = sorted(set(models) - model_field_names)
    if unknown_models:
        raise ValueError(f"Unknown OptionModelsState checkpoint fields: {unknown_models}")
    missing_model_fields = model_field_names - set(models)
    expected_missing_model_fields = (
        set(STOMP_OPTION_MODEL_EXPANSION_FIELDS)
        if missing_state_fields == pre_expansion_fields
        else set()
    )
    if missing_model_fields != expected_missing_model_fields:
        raise ValueError(
            "STOMP option-model field manifest does not match the state schema; "
            f"missing={sorted(missing_model_fields)}"
        )
    cumreward_ema = jnp.asarray(models["cumreward_ema"], dtype=jnp.float32)
    for name in STOMP_OPTION_MODEL_EXPANSION_FIELDS:
        if name in models:
            models[name] = jnp.asarray(models[name], dtype=jnp.float32)
        else:
            models[name] = jnp.zeros_like(cumreward_ema)
    option_models = OptionModelsState(**models)

    policies = _sub_payload_as_dict(
        data.pop("option_policies"), "option_policies", IntraOptionPoliciesState
    )
    policy_field_names = {
        field.name
        for field in dataclasses.fields(IntraOptionPoliciesState)  # type: ignore[arg-type]
    }
    if set(policies) != policy_field_names:
        raise ValueError(
            f"STOMP option-policy payload fields {sorted(policies)} != {sorted(policy_field_names)}"
        )
    option_policies = IntraOptionPoliciesState(**policies)

    for name in STOMP_STATE_EXPANSION_FIELDS:
        if name in data:
            data[name] = jnp.asarray(data[name], dtype=jnp.float32)
        else:
            data[name] = jnp.array(0.0, dtype=jnp.float32)

    base_learner = data["base_learner_state"]
    if not isinstance(base_learner, MultiHeadMLPState):
        base_fields = _sub_payload_as_dict(
            base_learner,
            "base_learner_state",
            MultiHeadMLPState,
        )
        current_base_names = {
            field.name
            for field in dataclasses.fields(MultiHeadMLPState)  # type: ignore[arg-type]
        }
        if set(base_fields) == current_base_names:
            base_learner = MultiHeadMLPState(**base_fields)
        else:
            base_learner = migrate_legacy_multi_head_mlp_state(base_fields)
    if not bool(
        _lifetime_counter_valid(
            base_learner.step_words,
            base_learner.step_count,
        )
    ):
        raise ValueError("STOMP base-learner lifetime counter is invalid")
    data["base_learner_state"] = base_learner

    if "step_words" not in data:
        legacy_step = jnp.asarray(data["step_count"])
        if legacy_step.shape != () or legacy_step.dtype != jnp.dtype(jnp.int32):
            raise TypeError("legacy STOMP step_count must be scalar int32")
        legacy_step_value = int(legacy_step)
        if legacy_step_value < 0:
            raise ValueError("negative legacy STOMP step_count indicates wrap")
        if legacy_step_value >= _INT32_MAX:
            raise ValueError("saturated legacy STOMP step_count is ambiguous")
        data["step_words"] = jnp.asarray(
            (0, legacy_step_value),
            dtype=jnp.uint32,
        )
    else:
        data["step_words"] = jnp.asarray(data["step_words"])
    if not bool(_lifetime_counter_valid(data["step_words"], data["step_count"])):
        raise ValueError("STOMP lifetime counter is invalid")

    return STOMPState(
        option_models=option_models,
        option_policies=option_policies,
        **data,
    )


__all__ = [
    "DISPATCH_OWNER_BASE_PRIMITIVE",
    "DISPATCH_OWNER_INVALID",
    "DISPATCH_OWNER_OPTION",
    "DispatchedPrimitiveActionDecision",
    "DispatchedPrimitiveActionReplacementResult",
    "IntraOptionPoliciesState",
    "OptionModelsState",
    "STOMPAgent",
    "STOMPArrayResult",
    "STOMPConfig",
    "STOMP_LIFETIME_COUNTER_DELTA_NBYTES",
    "STOMP_LIFETIME_COUNTER_NBYTES",
    "STOMP_STATE_SCHEMA",
    "STOMPSpecArrays",
    "STOMPStartResult",
    "STOMPState",
    "STOMPUpdateResult",
    "STOMP_OPTION_MODEL_EXPANSION_FIELDS",
    "STOMP_STATE_EXPANSION_FIELDS",
    "STOMP_STATE_LIFETIME_FIELDS",
    "SubtaskSpec",
    "check_option_terminated",
    "compute_pseudo_reward",
    "load_stomp_state_with_migration",
    "measure_stomp_state_nbytes",
    "measure_stomp_wrapper_state_nbytes",
    "replace_dispatched_primitive_action",
    "stomp_state_to_checkpoint_payload",
    "stomp_lifetime_counter_nbytes",
    "subtasks_from_feature_scores",
]
