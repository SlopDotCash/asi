# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Core types and algorithms for the OaK architecture (Alberta Plan Step 11).

OaK (Options and Knowledge) extends the STOMP progression (Step 10) with two
additional mechanisms:

1. **Utility tracking** — Each option's execution frequency and accumulated
   pseudo-reward are tracked online via EMA.  This produces a per-option
   utility score without any expensive periodic evaluation pass.
2. **Curation** — The ``curate()`` method compares each option's utility EMA
   against a configurable threshold.  The lowest-utility option is replaced
   with a new :class:`~alberta_framework.core.options.SubtaskSpec` targeting a
   different observation feature, and its weights/models are reset.
3. **Option keyboard** — A real-valued keyboard vector w ∈ R^N encodes a
   *chord*: a weighted blend of N option Q-functions that produces a single
   Q-vector over primitive actions.  The policy is greedy w.r.t. this blend
   (Barreto et al. 2019, Option Keyboard, §3.1):
   ``Q_w(s, a) = Σ_i w_i Q_i(s, a)``.

Together these realise the **FC-STOMP cycle**:
Feature Construction → SubTask → Option → Model → Planning → (Curation) →
where curation drives continuous self-improvement by replacing unhelpful
options with new subtasks on higher-utility features.

References:
    Sutton, Bowling, & Pilarski (2022). "The Alberta Plan for AI Research."
    Barreto et al. (2019). "The Option Keyboard: Combining Skills in RL."
    Sutton (RLC 2025). "The OaK Architecture."
    Wan, Naik, & Sutton (2021). "Average-Reward Learning and Planning
        with Options."
"""

from __future__ import annotations

import dataclasses
import math
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

from alberta_framework.core.normalizers import (
    _checked_lifetime_words_increment,
    _lifetime_counter_valid,
    _saturating_int32_counter_increment,
)
from alberta_framework.core.options import (
    DispatchedPrimitiveActionDecision,
    IntraOptionPoliciesState,
    OptionModelsState,
    STOMPAgent,
    STOMPConfig,
    STOMPState,
    STOMPUpdateResult,
    SubtaskSpec,
    _checked_lifetime_words_advance,
    _lifetime_words_at_least,
    load_stomp_state_with_migration,
    measure_stomp_state_nbytes,
    replace_dispatched_primitive_action,
    subtasks_from_feature_scores,
)
from alberta_framework.core.types import MLPParams

# OaK owns one telemetry/word clock. The nested STOMP state owns a second,
# exactly aligned clock; ``oak_lifetime_counter_nbytes`` reports their total.
OAK_STATE_SCHEMA = "alberta.oak-state.v2"
OAK_LIFETIME_COUNTER_NBYTES = 12
OAK_LIFETIME_COUNTER_DELTA_NBYTES = 8

_INT32_MAX = 2**31 - 1
_UINT64_MAX = 2**64 - 1
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


def _require_uint64(
    name: str, value: object, *, minimum: int = 0, maximum: int = _UINT64_MAX
) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


# ---------------------------------------------------------------------------
# Default config helper
# ---------------------------------------------------------------------------


def _default_stomp_config() -> STOMPConfig:
    return STOMPConfig(subtask_specs=(SubtaskSpec(feature_index=0),))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OaKConfig:
    """Configuration for the OaK agent.

    Args:
        stomp: Underlying STOMP configuration. An empty subtask tuple selects
            exact primitive-only base control with no option side effects.
        utility_ema_decay: EMA decay for per-option utility tracking.
            Higher → slower adaptation.  Range [0, 1].
        curation_threshold: Utility EMA value below which an option is
            eligible for replacement.  Set to 0 to disable automatic
            threshold gating (replace the worst option unconditionally when
            :meth:`curate` is called).
        min_steps_before_curation: Minimum number of primitive steps
            (``step_count``) before :meth:`curate` may evict an option.
            Guards against evicting on untrained utility EMAs.  0 (default)
            disables the guard, preserving unconditional curation.
    """

    stomp: STOMPConfig = dataclasses.field(default_factory=_default_stomp_config)
    utility_ema_decay: float = 0.99
    curation_threshold: float = 0.0
    min_steps_before_curation: int = 0

    def __post_init__(self) -> None:
        min_steps = _require_uint64(
            "min_steps_before_curation",
            self.min_steps_before_curation,
            minimum=0,
            maximum=_UINT64_MAX,
        )
        object.__setattr__(self, "min_steps_before_curation", min_steps)

        if not math.isfinite(self.utility_ema_decay) or not 0.0 <= self.utility_ema_decay <= 1.0:
            raise ValueError("utility_ema_decay must be finite and in [0, 1]")
        if not math.isfinite(self.curation_threshold) or self.curation_threshold < 0.0:
            raise ValueError("curation_threshold must be finite and non-negative")

    @property
    def n_options(self) -> int:
        return len(self.stomp.subtask_specs)

    @property
    def n_primitive_actions(self) -> int:
        return self.stomp.n_primitive_actions

    @property
    def observation_dim(self) -> int:
        return self.stomp.observation_dim

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "type": "OaKConfig",
            "stomp": self.stomp.to_config(),
            "utility_ema_decay": self.utility_ema_decay,
            "curation_threshold": self.curation_threshold,
            "min_steps_before_curation": self.min_steps_before_curation,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> OaKConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = dict(payload)
        data.pop("type", None)
        stomp_raw = data.pop("stomp")
        stomp = STOMPConfig.from_config(stomp_raw)
        return cls(stomp=stomp, **data)


# ---------------------------------------------------------------------------
# State and result types
# ---------------------------------------------------------------------------


@chex.dataclass(frozen=True)
class OaKState:
    """Combined OaK agent state.

    Attributes:
        stomp_state: Full STOMP agent state (weights, traces, models, RNG).
        execution_counts: Integer count of times each option has been
            started; shape ``(n_options,)``.
        cumulative_pseudo_rewards: Running sum of pseudo-reward accumulated
            while each option was active; shape ``(n_options,)``.
        utility_ema: EMA utility score for each option; shape
            ``(n_options,)``.  Updated every primitive step that an option is
            executing.
        step_count: Saturating int32 primitive-step telemetry.
        step_words: Exact big-endian ``[high, low]`` uint32 primitive-step
            identity, aligned exactly with the nested STOMP clock.
    """

    stomp_state: STOMPState
    execution_counts: Int[Array, " n_options"]
    cumulative_pseudo_rewards: Float[Array, " n_options"]
    utility_ema: Float[Array, " n_options"]
    step_count: Int[Array, ""]
    step_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class OaKUpdateResult:
    """Result of one primitive OaK transition."""

    state: OaKState
    td_error: Float[Array, ""]
    average_reward: Float[Array, ""]
    primitive_action: Int[Array, ""]
    executing_option: Int[Array, ""]
    option_terminated: Array
    pseudo_reward: Float[Array, ""]
    utility_ema: Float[Array, " n_options"]
    planning_backups: Int[Array, ""]
    planning_td_error: Float[Array, ""]
    pre_step_words: UInt[Array, " 2"]
    post_step_words: UInt[Array, " 2"]
    outer_state_valid: Bool[Array, ""]
    lifetime_counter_valid: Bool[Array, ""]
    lifetime_capacity_available: Bool[Array, ""]
    nested_counter_aligned: Bool[Array, ""]
    nested_update_applied: Bool[Array, ""]
    proposed_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class OaKSTOMPUpdateAdoptionResult:
    """Audit for adopting one externally evaluated STOMP transition.

    The seam checks exact source identity, clock provenance, endpoint fields,
    and the diagnostics carried by :class:`STOMPUpdateResult`.  It deliberately
    does not rerun STOMP and therefore cannot independently authenticate that
    the supplied result was derived by the claimed caller.  The lifecycle that
    invokes this method remains the authority for that claim.
    """

    update: OaKUpdateResult
    source_state_valid: Bool[Array, ""]
    source_state_matches: Bool[Array, ""]
    result_static_contract_valid: Bool[Array, ""]
    result_clock_binding_valid: Bool[Array, ""]
    result_endpoint_binding_valid: Bool[Array, ""]
    result_diagnostics_valid: Bool[Array, ""]
    derivation_recomputed: Bool[Array, ""]
    caller_authority_required: Bool[Array, ""]
    caller_authenticated: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class OaKTracedUpdateResult:
    """One OaK update plus the exact sole STOMP evaluation it adopted."""

    update: OaKUpdateResult
    stomp_result: STOMPUpdateResult
    stomp_update_evaluations: Int[Array, ""]
    derivation_recomputed: Bool[Array, ""]


@chex.dataclass(frozen=True)
class OaKOptionSlotRebindResult:
    """Audit for one exact, quiescent option-slot STOMP rebind."""

    state: OaKState
    reset_slot_mask: Bool[Array, " n_options"]
    source_state_valid: Bool[Array, ""]
    rebound_state_valid: Bool[Array, ""]
    source_quiescent: Bool[Array, ""]
    reset_requested: Bool[Array, ""]
    clocks_preserved: Bool[Array, ""]
    policy_rng_preserved: Bool[Array, ""]
    unchanged_slots_preserved: Bool[Array, ""]
    proposed_state_valid: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]


@dataclasses.dataclass(frozen=True, slots=True)
class OaKExternalSTOMPAdoptionResourceBudget:
    """Exact allocation/work and authority facts for external STOMP adoption."""

    persistent_state_nbytes_before: int
    persistent_state_nbytes_after: int
    persistent_state_growth_bytes: int
    stomp_update_evaluations_per_adopt: int
    stomp_update_evaluations_per_delegated_update: int
    derivation_recomputed_on_adopt: bool
    source_result_integrity_checked: bool
    caller_authority_required: bool
    caller_authenticated: bool


@chex.dataclass(frozen=True)
class OaKArrayResult:
    """Scan result for OaK over transition arrays."""

    state: OaKState
    td_errors: Float[Array, " num_steps"]
    average_rewards: Float[Array, " num_steps"]
    primitive_actions: Int[Array, " num_steps"]
    executing_options: Int[Array, " num_steps"]
    option_terminations: Array
    pseudo_rewards: Float[Array, " num_steps"]
    utility_emas: Float[Array, "num_steps n_options"]
    planning_backups: Int[Array, " num_steps"]
    planning_td_errors: Float[Array, " num_steps"]
    pre_step_words: UInt[Array, "num_steps 2"]
    post_step_words: UInt[Array, "num_steps 2"]
    outer_state_valid: Bool[Array, " num_steps"]
    lifetime_counter_valid: Bool[Array, " num_steps"]
    lifetime_capacity_available: Bool[Array, " num_steps"]
    nested_counter_aligned: Bool[Array, " num_steps"]
    nested_update_applied: Bool[Array, " num_steps"]
    proposed_state_valid: Bool[Array, " num_steps"]
    update_applied: Bool[Array, " num_steps"]


@chex.dataclass(frozen=True)
class OaKKeyboardPolicyProposal:
    """Deterministic fixed-chord proposal bound to one OaK decision state.

    The first three fields intentionally match the partner-fusion option
    proposal surface. ``declared_score`` is the selected keyboard Q-value;
    it is descriptive and carries no calibration or benefit claim.
    """

    available: Bool[Array, ""]
    action: Int[Array, ""]
    declared_score: Float[Array, ""]
    decision_observation: Float[Array, " observation_dim"]
    keyboard_vector: Float[Array, " n_options"]
    q_values: Float[Array, " n_primitive_actions"]
    outer_state_static_contract_valid: Bool[Array, ""]
    outer_state_values_finite: Bool[Array, ""]
    outer_state_counters_valid: Bool[Array, ""]
    outer_state_valid: Bool[Array, ""]
    stomp_state_valid: Bool[Array, ""]
    state_valid: Bool[Array, ""]
    observation_static_contract_valid: Bool[Array, ""]
    observation_valid: Bool[Array, ""]
    observation_matches: Bool[Array, ""]
    keyboard_vector_static_contract_valid: Bool[Array, ""]
    keyboard_vector_valid: Bool[Array, ""]
    q_values_valid: Bool[Array, ""]


@chex.dataclass(frozen=True)
class OaKKeyboardDispatchDecision:
    """Keyboard proposal and ownership-correct effective dispatch audit."""

    proposal: OaKKeyboardPolicyProposal
    replacement: DispatchedPrimitiveActionDecision


@chex.dataclass(frozen=True)
class OaKKeyboardDispatchResult:
    """OaK state after one strict keyboard-dispatch attempt."""

    state: OaKState
    decision: OaKKeyboardDispatchDecision


def _oak_outer_state_validity(
    state: OaKState,
    config: OaKConfig,
) -> tuple[Array, Array, Array, Array]:
    """Validate all OaK-owned dynamic leaves around the nested STOMP state."""

    n_options = config.n_options
    static_contract_valid = (
        state.execution_counts.shape == (n_options,)
        and state.execution_counts.dtype == jnp.int32
        and state.cumulative_pseudo_rewards.shape == (n_options,)
        and state.cumulative_pseudo_rewards.dtype == jnp.float32
        and state.utility_ema.shape == (n_options,)
        and state.utility_ema.dtype == jnp.float32
        and state.step_count.shape == ()
        and state.step_count.dtype == jnp.int32
        and state.step_words.shape == (2,)
        and state.step_words.dtype == jnp.uint32
        and state.stomp_state.step_words.shape == (2,)
        and state.stomp_state.step_words.dtype == jnp.uint32
        and state.stomp_state.option_policies.q_weights.shape
        == (
            config.n_options,
            config.n_primitive_actions,
            config.observation_dim,
        )
    )
    values_finite = jnp.all(jnp.isfinite(state.cumulative_pseudo_rewards)) & jnp.all(
        jnp.isfinite(state.utility_ema)
    )
    counter_ceiling = jnp.where(
        state.step_count < jnp.int32(2_147_483_647),
        state.step_count + jnp.int32(1),
        state.step_count,
    )
    counters_valid = (
        _lifetime_counter_valid(state.step_words, state.step_count)
        & (state.step_count >= 0)
        & (state.step_count == state.stomp_state.step_count)
        & jnp.all(state.execution_counts >= 0)
        & jnp.all(state.execution_counts <= counter_ceiling)
    )
    valid = jnp.asarray(static_contract_valid, dtype=jnp.bool_) & values_finite & counters_valid
    return (
        jnp.asarray(static_contract_valid, dtype=jnp.bool_),
        values_finite,
        counters_valid,
        valid,
    )


def _float_bits_equal(left: Array, right: Array) -> Bool[Array, ""]:
    """Compare float arrays without treating signed zero as interchangeable."""

    if left.dtype == jnp.float16:
        return jnp.array_equal(
            jax.lax.bitcast_convert_type(left, jnp.uint16),
            jax.lax.bitcast_convert_type(right, jnp.uint16),
        )
    if left.dtype == jnp.float32:
        return jnp.array_equal(
            jax.lax.bitcast_convert_type(left, jnp.uint32),
            jax.lax.bitcast_convert_type(right, jnp.uint32),
        )
    if left.dtype == jnp.float64:
        return jnp.array_equal(
            jax.lax.bitcast_convert_type(left, jnp.uint64),
            jax.lax.bitcast_convert_type(right, jnp.uint64),
        )
    return jnp.array_equal(left, right)


def _exact_tree_equal(left: Any, right: Any) -> Bool[Array, ""]:
    """Compare a fixed-shape PyTree by exact typed leaf content."""

    if type(left) is not type(right):
        return jnp.asarray(False, dtype=jnp.bool_)
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if cast(object, left_structure) != cast(object, right_structure) or len(left_leaves) != len(
        right_leaves
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    equal = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            leaf_equal = jnp.array_equal(
                jr.key_data(left_array),
                jr.key_data(right_array),
            )
        elif jnp.issubdtype(left_array.dtype, jnp.floating):
            leaf_equal = _float_bits_equal(left_array, right_array)
        else:
            leaf_equal = jnp.array_equal(left_array, right_array)
        equal = equal & leaf_equal
    return equal


def _require_result_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    """Validate one static STOMP result leaf and return it as an array."""

    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    expected_dtype = jnp.dtype(dtype)
    if array.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}, got {array.dtype}")
    return array


def _validate_stomp_update_result_contract(result: STOMPUpdateResult) -> None:
    """Reject malformed static result manifests before staging an adoption."""

    if type(result) is not STOMPUpdateResult:
        raise TypeError("stomp_result must be a STOMPUpdateResult")
    if type(result.state) is not STOMPState:
        raise TypeError("stomp_result.state must be a STOMPState")
    float_scalars = (
        "td_error",
        "average_reward",
        "pseudo_reward",
        "option_importance_ratio",
        "planning_td_error",
    )
    int_scalars = (
        "primitive_action",
        "executing_option",
        "planning_backups",
        "nested_updates_required",
        "nested_updates_applied",
    )
    bool_scalars = (
        "option_terminated",
        "inputs_valid",
        "lifetime_counter_valid",
        "lifetime_capacity_available",
        "nested_lifetime_counter_valid",
        "nested_lifetime_capacity_available",
        "proposed_state_valid",
        "update_applied",
    )
    for name in float_scalars:
        _require_result_array(
            getattr(result, name),
            name=f"stomp_result.{name}",
            shape=(),
            dtype=jnp.float32,
        )
    for name in int_scalars:
        _require_result_array(
            getattr(result, name),
            name=f"stomp_result.{name}",
            shape=(),
            dtype=jnp.int32,
        )
    for name in bool_scalars:
        _require_result_array(
            getattr(result, name),
            name=f"stomp_result.{name}",
            shape=(),
            dtype=jnp.bool_,
        )
    for name in ("pre_step_words", "post_step_words"):
        _require_result_array(
            getattr(result, name),
            name=f"stomp_result.{name}",
            shape=(2,),
            dtype=jnp.uint32,
        )


def _merge_rebound_stomp_option_slots(
    source: STOMPState,
    rebound: STOMPState,
    reset_slots: Array,
    *,
    n_primitive_actions: int,
) -> STOMPState:
    """Copy only reset-slot policy, model, and base-head state from rebound."""

    mask1 = reset_slots
    mask3 = reset_slots[:, None, None]
    policies = cast(
        IntraOptionPoliciesState,
        source.option_policies.replace(
            q_weights=jnp.where(
                mask3,
                rebound.option_policies.q_weights,
                source.option_policies.q_weights,
            ),
            traces=jnp.where(
                mask3,
                rebound.option_policies.traces,
                source.option_policies.traces,
            ),
            average_rewards=jnp.where(
                mask1,
                rebound.option_policies.average_rewards,
                source.option_policies.average_rewards,
            ),
        ),
    )
    models = cast(
        OptionModelsState,
        source.option_models.replace(
            cumreward_ema=jnp.where(
                mask1,
                rebound.option_models.cumreward_ema,
                source.option_models.cumreward_ema,
            ),
            env_return_ema=jnp.where(
                mask1,
                rebound.option_models.env_return_ema,
                source.option_models.env_return_ema,
            ),
            duration_ema=jnp.where(
                mask1,
                rebound.option_models.duration_ema,
                source.option_models.duration_ema,
            ),
            baseline_mass_ema=jnp.where(
                mask1,
                rebound.option_models.baseline_mass_ema,
                source.option_models.baseline_mass_ema,
            ),
            discount_ema=jnp.where(
                mask1,
                rebound.option_models.discount_ema,
                source.option_models.discount_ema,
            ),
            next_state_weights=jnp.where(
                mask3,
                rebound.option_models.next_state_weights,
                source.option_models.next_state_weights,
            ),
            n_completions=jnp.where(
                mask1,
                rebound.option_models.n_completions,
                source.option_models.n_completions,
            ),
        ),
    )
    source_learner = source.base_learner_state
    rebound_learner = rebound.base_learner_state
    merged_weights: list[Array] = []
    merged_biases: list[Array] = []
    merged_optimizer_states: list[Any] = []
    merged_traces: list[Any] = []
    for head in range(len(source_learner.head_params.weights)):
        reset = (
            jnp.asarray(False, dtype=jnp.bool_)
            if head < n_primitive_actions
            else reset_slots[head - n_primitive_actions]
        )
        merged_weights.append(
            jnp.where(
                reset,
                rebound_learner.head_params.weights[head],
                source_learner.head_params.weights[head],
            )
        )
        merged_biases.append(
            jnp.where(
                reset,
                rebound_learner.head_params.biases[head],
                source_learner.head_params.biases[head],
            )
        )
        merged_optimizer_states.append(
            jax.tree_util.tree_map(
                lambda fresh, old: jnp.where(reset, fresh, old),
                rebound_learner.head_optimizer_states[head],
                source_learner.head_optimizer_states[head],
            )
        )
        merged_traces.append(
            jax.tree_util.tree_map(
                lambda fresh, old: jnp.where(reset, fresh, old),
                rebound_learner.head_traces[head],
                source_learner.head_traces[head],
            )
        )
    learner = cast(
        Any,
        source_learner.replace(
            head_params=source_learner.head_params.replace(
                weights=tuple(merged_weights),
                biases=tuple(merged_biases),
            ),
            head_optimizer_states=tuple(merged_optimizer_states),
            head_traces=tuple(merged_traces),
        ),
    )
    return cast(
        STOMPState,
        source.replace(
            base_learner_state=learner,
            option_policies=policies,
            option_models=models,
        ),
    )


def measure_oak_state_nbytes(state: OaKState) -> int:
    """Measure all persistent JAX-array bytes in one concrete OaK state."""

    outer_state_nbytes = sum(
        int(value.size) * int(value.dtype.itemsize)
        for value in (
            state.execution_counts,
            state.cumulative_pseudo_rewards,
            state.utility_ema,
            state.step_count,
            state.step_words,
        )
    )
    return measure_stomp_state_nbytes(state.stomp_state) + outer_state_nbytes


def measure_oak_wrapper_state_nbytes(state: OaKState) -> int:
    """Measure only OaK-owned arrays, excluding the nested STOMP state."""

    return measure_oak_state_nbytes(state) - measure_stomp_state_nbytes(state.stomp_state)


def oak_lifetime_counter_nbytes() -> int:
    """Return bytes for the aligned OaK and nested STOMP primitive clocks.

    This excludes STOMP's independent nested base-learner update clock.
    """

    return 2 * OAK_LIFETIME_COUNTER_NBYTES


def oak_total_lifetime_counter_nbytes() -> int:
    """Return bytes for all three exact clocks in a concrete OaK state."""

    # OaK outer + STOMP primitive + MultiHead base-update identity.
    return 3 * OAK_LIFETIME_COUNTER_NBYTES


def _oak_host_field_mapping(value: Any) -> dict[str, Any]:
    """Return a shallow host mapping for one legacy state dataclass."""

    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    raise TypeError("legacy OaK state must be a mapping or dataclass")


def migrate_legacy_oak_state(legacy_state: Any) -> OaKState:
    """Migrate the exact pre-v2 OaK field manifest on the host.

    The legacy outer scalar is accepted only below int32 saturation, where it
    still identifies one unique lifetime. The nested STOMP state is migrated
    through its own strict seam and must authenticate the identical primitive
    step. Mixed, saturated, negative, or misaligned histories are rejected.
    """

    fields = _oak_host_field_mapping(legacy_state)
    current_names = {
        field.name
        for field in dataclasses.fields(OaKState)  # type: ignore[arg-type]
    }
    legacy_names = current_names - {"step_words"}
    supplied_names = set(fields)
    if supplied_names != legacy_names:
        missing = sorted(legacy_names - supplied_names)
        extra = sorted(supplied_names - legacy_names)
        raise ValueError(
            f"legacy OaK field manifest is not exact; missing={missing}, extra={extra}"
        )

    step_count = jnp.asarray(fields["step_count"])
    if step_count.shape != () or step_count.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy OaK step_count must be scalar int32")
    step = int(step_count)
    if step < 0:
        raise ValueError("negative legacy OaK step_count indicates wrap")
    if step >= 2**31 - 1:
        raise ValueError("saturated legacy OaK step_count is ambiguous")
    step_words = jnp.asarray((0, step), dtype=jnp.uint32)

    nested = fields["stomp_state"]
    if not isinstance(nested, STOMPState):
        nested = load_stomp_state_with_migration(_oak_host_field_mapping(nested))
    if not bool(_lifetime_counter_valid(nested.step_words, nested.step_count)):
        raise ValueError("legacy OaK nested STOMP lifetime counter is invalid")
    if int(nested.step_count) != step or not bool(jnp.all(nested.step_words == step_words)):
        raise ValueError("legacy OaK and nested STOMP clocks are not aligned")

    execution_counts = jnp.asarray(fields["execution_counts"])
    cumulative_pseudo_rewards = jnp.asarray(fields["cumulative_pseudo_rewards"])
    utility_ema = jnp.asarray(fields["utility_ema"])
    n_options = int(execution_counts.shape[0]) if execution_counts.ndim == 1 else -1
    owned_contract_valid = (
        n_options >= 0
        and execution_counts.dtype == jnp.int32
        and cumulative_pseudo_rewards.shape == (n_options,)
        and cumulative_pseudo_rewards.dtype == jnp.float32
        and utility_ema.shape == (n_options,)
        and utility_ema.dtype == jnp.float32
        and nested.option_policies.q_weights.ndim == 3
        and nested.option_policies.q_weights.shape[0] == n_options
    )
    if not owned_contract_valid:
        raise ValueError("legacy OaK owned-array contract is invalid")
    counter_ceiling = min(step + 1, 2**31 - 1)
    if not bool(jnp.all(execution_counts >= 0) & jnp.all(execution_counts <= counter_ceiling)):
        raise ValueError("legacy OaK execution counters are invalid")
    if not bool(
        jnp.all(jnp.isfinite(cumulative_pseudo_rewards)) & jnp.all(jnp.isfinite(utility_ema))
    ):
        raise ValueError("legacy OaK utility values are non-finite")

    return OaKState(
        stomp_state=nested,
        execution_counts=execution_counts,
        cumulative_pseudo_rewards=cumulative_pseudo_rewards,
        utility_ema=utility_ema,
        step_count=step_count,
        step_words=step_words,
    )


# ---------------------------------------------------------------------------
# Learned feature construction and keyboard chord learning
# ---------------------------------------------------------------------------


def learned_feature_subtask_specs(
    oak_state: OaKState,
    *,
    n_subtasks: int = 4,
    threshold: float = 0.5,
    pseudo_reward_scale: float = 1.0,
    max_option_steps: int = 20,
    min_score: float = 0.0,
) -> tuple[SubtaskSpec, ...]:
    """Construct subtask specs from learned OaK feature importance.

    Feature scores combine base extended-action Q weights and intra-option
    primitive-action Q weights.  The highest-scoring observation features are
    converted into :class:`SubtaskSpec` objects for curation or replacement.
    """
    bls = oak_state.stomp_state.base_learner_state
    if len(bls.trunk_params.weights) == 0:
        base_q = jnp.stack([w[0] for w in bls.head_params.weights])
    else:
        base_q = bls.trunk_params.weights[0]
    base_scores = jnp.max(jnp.abs(base_q), axis=0)

    option_q = oak_state.stomp_state.option_policies.q_weights
    obs_dim = int(option_q.shape[-1])
    option_scores = (
        jnp.max(jnp.abs(option_q).reshape(-1, obs_dim), axis=0)
        if option_q.shape[0] > 0
        else jnp.zeros((obs_dim,), dtype=jnp.float32)
    )
    combined_scores = base_scores + option_scores
    specs = subtasks_from_feature_scores(
        combined_scores,
        top_k=n_subtasks,
        threshold=threshold,
        pseudo_reward_scale=pseudo_reward_scale,
        max_option_steps=max_option_steps,
        min_score=min_score,
    )
    return tuple(specs)


@dataclasses.dataclass(frozen=True)
class KeyboardChordLearnerConfig:
    """Bandit-style learner for option-keyboard chord vectors."""

    n_options: int
    step_size: float = 0.1
    baseline_decay: float = 0.9
    l2_penalty: float = 0.0
    max_norm: float = 10.0

    def __post_init__(self) -> None:
        n_options = _require_int32("n_options", self.n_options, minimum=1)
        object.__setattr__(self, "n_options", n_options)

        if not math.isfinite(self.step_size) or self.step_size < 0.0:
            raise ValueError("step_size must be finite and non-negative")
        if not math.isfinite(self.baseline_decay) or not 0.0 <= self.baseline_decay < 1.0:
            raise ValueError("baseline_decay must be finite and in [0, 1)")
        if not math.isfinite(self.l2_penalty) or self.l2_penalty < 0.0:
            raise ValueError("l2_penalty must be finite and non-negative")
        if not math.isfinite(self.max_norm) or self.max_norm <= 0.0:
            raise ValueError("max_norm must be finite and positive")

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = dataclasses.asdict(self)
        payload["type"] = "KeyboardChordLearnerConfig"
        return payload

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> KeyboardChordLearnerConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = dict(payload)
        data.pop("type", None)
        return cls(**data)


@chex.dataclass(frozen=True)
class KeyboardChordLearnerState:
    """State for bandit-style chord-vector learning."""

    chord_vector: Float[Array, " n_options"]
    reward_baseline: Float[Array, ""]
    step_count: Int[Array, ""]


def init_keyboard_chord_learner(
    config: KeyboardChordLearnerConfig,
) -> KeyboardChordLearnerState:
    """Initialize keyboard-chord learner state."""
    return KeyboardChordLearnerState(
        chord_vector=jnp.ones(config.n_options, dtype=jnp.float32) / config.n_options,
        reward_baseline=jnp.array(0.0, dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
    )


def update_keyboard_chord_learner(
    config: KeyboardChordLearnerConfig,
    state: KeyboardChordLearnerState,
    selected_chord: Array,
    reward: Array,
) -> KeyboardChordLearnerState:
    """Apply one bandit-style reward update for a selected chord.

    Positive advantage moves the learned chord vector toward the selected
    chord; negative advantage moves it away.  The reward baseline is an EMA.
    """
    chord = jnp.asarray(selected_chord, dtype=jnp.float32).reshape((config.n_options,))
    chord_norm = chord / (jnp.linalg.norm(chord) + 1.0e-8)
    reward_arr = jnp.asarray(reward, dtype=jnp.float32)
    baseline = (
        config.baseline_decay * state.reward_baseline + (1.0 - config.baseline_decay) * reward_arr
    )
    advantage = reward_arr - state.reward_baseline
    proposed_vector = (
        state.chord_vector * (1.0 - config.step_size * config.l2_penalty)
        + config.step_size * advantage * chord_norm
    )
    norm = jnp.linalg.norm(proposed_vector)
    scale = jnp.minimum(1.0, jnp.asarray(config.max_norm, dtype=jnp.float32) / (norm + 1.0e-8))
    proposed_state = KeyboardChordLearnerState(
        chord_vector=proposed_vector * scale,
        reward_baseline=baseline,
        step_count=state.step_count + 1,
    )
    # Inf reward * a zero chord coordinate is 0*inf = NaN, then the
    # max-norm rescaling is nan/inf and the whole vector stays poisoned.
    inputs_valid = jnp.all(jnp.isfinite(chord)) & jnp.isfinite(reward_arr)
    proposed_finite = jnp.all(jnp.isfinite(proposed_state.chord_vector)) & jnp.isfinite(
        proposed_state.reward_baseline
    )
    return jax.lax.cond(
        inputs_valid & proposed_finite,
        lambda: proposed_state,
        lambda: state,
    )


# ---------------------------------------------------------------------------
# Option keyboard (standalone JAX functions)
# ---------------------------------------------------------------------------


def keyboard_q_values(
    stomp_state: STOMPState,
    observation: Float[Array, " obs_dim"],
    keyboard_vector: Float[Array, " n_options"],
) -> Float[Array, " n_primitive_actions"]:
    """Compute primitive Q-values for a chord keyboard vector.

    Per Barreto et al. (2019) Eq. 6:
    ``Q_w(s, a) = Σ_i w_i · Q_i(s, a)``
    where ``Q_i(s, a) = option_q_weights[i, a, :] @ s``.

    The keyboard vector is internally L1-normalised so the blend is
    well-defined regardless of scale.

    Args:
        stomp_state: Current STOMP agent state; provides
            ``option_policies.q_weights`` of shape
            ``(n_options, n_prim, obs_dim)``.
        observation: Current observation, shape ``(obs_dim,)``.
        keyboard_vector: Chord weights, shape ``(n_options,)``.

    Returns:
        Shape ``(n_prim,)`` blended Q-values for each primitive action.
    """
    w = keyboard_vector / (jnp.sum(jnp.abs(keyboard_vector)) + 1e-8)
    blended = jnp.einsum("o,oap->ap", w, stomp_state.option_policies.q_weights)
    return blended @ observation


def keyboard_action(
    stomp_state: STOMPState,
    observation: Float[Array, " obs_dim"],
    keyboard_vector: Float[Array, " n_options"],
    key: Array,
    *,
    epsilon: float,
    n_primitive_actions: int,
) -> tuple[Int[Array, ""], Array]:
    """Select a primitive action using a chord keyboard vector.

    Uses the blended Q-values from :func:`keyboard_q_values` with ε-greedy
    exploration and Gumbel-noise tie-breaking.

    Args:
        stomp_state: Current STOMP agent state.
        observation: Current observation.
        keyboard_vector: Chord weights.
        key: JAX PRNG key.
        epsilon: Exploration probability.
        n_primitive_actions: Number of primitive actions.

    Returns:
        ``(action, new_key)`` pair.
    """
    q_vals = keyboard_q_values(stomp_state, observation, keyboard_vector)
    key, explore_key, noise_key = jr.split(key, 3)
    greedy = jnp.argmax(q_vals + 1e-6 * jr.gumbel(noise_key, (n_primitive_actions,))).astype(
        jnp.int32
    )
    random_action = jr.randint(explore_key, (), 0, n_primitive_actions).astype(jnp.int32)
    action = jnp.where(
        jr.uniform(key) < jnp.asarray(epsilon, dtype=jnp.float32),
        random_action,
        greedy,
    )
    return action, key


# ---------------------------------------------------------------------------
# OaK agent
# ---------------------------------------------------------------------------


class OaKAgent:
    """Alberta Plan Step 11 OaK agent.

    Wraps a :class:`~alberta_framework.core.options.STOMPAgent` with:

    * **Utility tracking**: an EMA utility score per option, updated on every
      step that an option is actively executing.
    * **Curation**: :meth:`curate` replaces the lowest-utility option when its
      score falls below ``config.curation_threshold``.
    * **Option keyboard**: :meth:`keyboard_q_values` and
      :meth:`keyboard_action` provide chord-blended Q-inference.
    """

    def __init__(self, config: OaKConfig) -> None:
        self._config = config
        self._stomp = STOMPAgent(config.stomp)

    @property
    def config(self) -> OaKConfig:
        return self._config

    @property
    def stomp_agent(self) -> STOMPAgent:
        return self._stomp

    def base_q_values(self, state: OaKState, observation: Array) -> Array:
        """Compute Q-values for all extended actions."""
        return self._stomp.base_q_values(state.stomp_state, observation)

    def to_config(self) -> dict[str, Any]:
        return self._config.to_config()

    def init(self, key: Array) -> OaKState:
        """Initialise OaK state (zeros for utility tracking)."""
        stomp_state = self._stomp.init(key)
        n_opt = self._config.n_options
        return OaKState(
            stomp_state=stomp_state,
            execution_counts=jnp.zeros(n_opt, dtype=jnp.int32),
            cumulative_pseudo_rewards=jnp.zeros(n_opt, dtype=jnp.float32),
            utility_ema=jnp.zeros(n_opt, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def start(
        self,
        state: OaKState,
        initial_observation: Array,
        *,
        extended_action_mask: Array | None = None,
    ) -> OaKState:
        """Prime the agent, optionally excluding cold option actions.

        ``extended_action_mask`` spans primitive actions followed by options.
        Every primitive action must remain live.  Omitting it executes the
        historical path exactly; an all-true supplied mask preserves that
        path's state and policy-RNG bits for a valid source.
        """

        if extended_action_mask is None:
            new_stomp = self._stomp.start(state.stomp_state, initial_observation)
            start_applied = jnp.asarray(True, dtype=jnp.bool_)
        else:
            start_result = self._stomp.start_with_extended_action_mask(
                state.stomp_state,
                initial_observation,
                extended_action_mask,
            )
            new_stomp = start_result.state
            mask = jnp.asarray(extended_action_mask)
            observation = jnp.asarray(initial_observation, dtype=jnp.float32).reshape(
                (self._config.observation_dim,)
            )
            outer_state_valid = _oak_outer_state_validity(state, self._config)[-1]
            start_applied = (
                outer_state_valid
                & self._stomp.state_valid(state.stomp_state)
                & jnp.all(state.step_words == state.stomp_state.step_words)
                & jnp.all(mask[: self._config.n_primitive_actions])
                & jnp.any(mask)
                & jnp.all(jnp.isfinite(observation))
            )
        started_option = new_stomp.executing_option
        started_mask = jnp.arange(self._config.n_options, dtype=jnp.int32) == jnp.maximum(
            started_option, jnp.array(0, dtype=jnp.int32)
        )
        new_execution_counts = jnp.where(
            started_mask & (started_option >= 0),
            _saturating_int32_counter_increment(state.execution_counts),
            state.execution_counts,
        )
        proposed = cast(
            OaKState,
            state.replace(
                stomp_state=new_stomp,
                execution_counts=new_execution_counts,
            ),
        )
        if extended_action_mask is None:
            return proposed
        proposed_valid = (
            _oak_outer_state_validity(proposed, self._config)[-1]
            & self._stomp.state_valid(proposed.stomp_state)
            & jnp.all(proposed.step_words == proposed.stomp_state.step_words)
        )
        return jax.lax.cond(
            start_applied & proposed_valid,
            lambda: proposed,
            lambda: state,
        )

    def adopt_stomp_update(
        self,
        state: OaKState,
        *,
        source_state: OaKState,
        stomp_result: STOMPUpdateResult,
    ) -> OaKSTOMPUpdateAdoptionResult:
        """Adopt one authoritative STOMP result without evaluating STOMP again.

        The supplied ``source_state`` must match ``state`` bit for bit, including
        signed-zero and policy-RNG representation.  The result must bind to that
        source's exact primitive and nested clocks, its exposed endpoint fields,
        and its own success diagnostics.  Failure returns the complete current
        OaK state unchanged.

        This is an integrity seam, not a derivation proof: it does not recompute
        the STOMP transition and carries no caller credential.  A trusted outer
        lifecycle must establish that ``stomp_result`` came from its one
        authorized STOMP evaluation.
        """

        if type(state) is not OaKState:
            raise TypeError("state must be an OaKState")
        if type(source_state) is not OaKState:
            raise TypeError("source_state must be an OaKState")
        _validate_stomp_update_result_contract(stomp_result)
        cfg = self._config
        n_options = cfg.n_options

        outer_state_valid = _oak_outer_state_validity(state, cfg)[-1]
        source_outer_valid = _oak_outer_state_validity(source_state, cfg)[-1]
        outer_counter_valid = _lifetime_counter_valid(
            state.step_words,
            state.step_count,
        )
        nested_counter_aligned = jnp.all(state.step_words == state.stomp_state.step_words)
        source_nested_counter_aligned = jnp.all(
            source_state.step_words == source_state.stomp_state.step_words
        )
        source_state_valid = (
            source_outer_valid
            & source_nested_counter_aligned
            & self._stomp.state_valid(source_state.stomp_state)
        )
        source_state_matches = _exact_tree_equal(state, source_state)

        proposed_step_words, outer_capacity = _checked_lifetime_words_increment(
            source_state.step_words
        )
        expected_nested_words, computed_nested_capacity = _checked_lifetime_words_advance(
            source_state.stomp_state.base_learner_state.step_words,
            stomp_result.nested_updates_required,
        )
        expected_step_count = _saturating_int32_counter_increment(source_state.step_count)
        result_clock_binding_valid = (
            jnp.array_equal(
                stomp_result.pre_step_words,
                source_state.stomp_state.step_words,
            )
            & jnp.array_equal(stomp_result.post_step_words, proposed_step_words)
            & jnp.array_equal(
                stomp_result.state.step_words,
                proposed_step_words,
            )
            & (stomp_result.state.step_count == expected_step_count)
            & jnp.array_equal(
                stomp_result.state.base_learner_state.step_words,
                expected_nested_words,
            )
            & computed_nested_capacity
        )

        prior_option = source_state.stomp_state.executing_option
        prior_active = prior_option >= jnp.asarray(0, dtype=jnp.int32)
        output_values_finite = (
            jnp.isfinite(stomp_result.td_error)
            & jnp.isfinite(stomp_result.average_reward)
            & jnp.isfinite(stomp_result.pseudo_reward)
            & jnp.isfinite(stomp_result.option_importance_ratio)
            & jnp.isfinite(stomp_result.planning_td_error)
        )
        idle_outputs_valid = prior_active | (
            (~stomp_result.option_terminated)
            & _float_bits_equal(
                jnp.asarray(stomp_result.pseudo_reward),
                jnp.asarray(0.0, dtype=jnp.float32),
            )
        )
        continuing_owner_valid = (
            (~prior_active)
            | stomp_result.option_terminated
            | (stomp_result.state.executing_option == prior_option)
        )
        result_endpoint_binding_valid = (
            self._stomp.state_valid(stomp_result.state)
            & output_values_finite
            & _float_bits_equal(
                jnp.asarray(stomp_result.average_reward),
                jnp.asarray(stomp_result.state.base_average_reward),
            )
            & (stomp_result.primitive_action == stomp_result.state.last_primitive_action)
            & (stomp_result.executing_option == stomp_result.state.executing_option)
            & (stomp_result.primitive_action >= 0)
            & (stomp_result.primitive_action < cfg.n_primitive_actions)
            & (stomp_result.executing_option >= -1)
            & (stomp_result.executing_option < n_options)
            & idle_outputs_valid
            & continuing_owner_valid
        )
        inferred_real_updates = stomp_result.nested_updates_applied - stomp_result.planning_backups
        result_diagnostics_valid = (
            stomp_result.inputs_valid
            & stomp_result.lifetime_counter_valid
            & stomp_result.lifetime_capacity_available
            & stomp_result.nested_lifetime_counter_valid
            & stomp_result.nested_lifetime_capacity_available
            & stomp_result.proposed_state_valid
            & stomp_result.update_applied
            & (stomp_result.nested_updates_required >= 0)
            & (stomp_result.nested_updates_applied == stomp_result.nested_updates_required)
            & (stomp_result.planning_backups >= 0)
            & (stomp_result.planning_backups <= cfg.stomp.option_planning_backups_per_step)
            & (inferred_real_updates >= 0)
            & (inferred_real_updates <= 1)
            & (stomp_result.option_importance_ratio >= 0.0)
            & (stomp_result.option_importance_ratio <= cfg.stomp.option_importance_clip)
        )

        prior_option_index = jnp.maximum(
            prior_option,
            jnp.asarray(0, dtype=jnp.int32),
        )
        prior_option_mask = jnp.arange(n_options, dtype=jnp.int32) == prior_option_index
        decay = jnp.asarray(cfg.utility_ema_decay, dtype=jnp.float32)
        new_utility_ema = jnp.where(
            prior_option_mask & prior_active,
            decay * source_state.utility_ema + (1.0 - decay) * stomp_result.pseudo_reward,
            source_state.utility_ema,
        )

        post_option = stomp_result.state.executing_option
        post_option_index = jnp.maximum(
            post_option,
            jnp.asarray(0, dtype=jnp.int32),
        )
        post_option_mask = jnp.arange(n_options, dtype=jnp.int32) == post_option_index
        post_active = post_option >= jnp.asarray(0, dtype=jnp.int32)
        just_started = post_active & ((~prior_active) | stomp_result.option_terminated)
        new_execution_counts = jnp.where(
            post_option_mask & just_started,
            _saturating_int32_counter_increment(source_state.execution_counts),
            source_state.execution_counts,
        )
        new_cumulative_pseudo_rewards = source_state.cumulative_pseudo_rewards + jnp.where(
            prior_option_mask & prior_active,
            jnp.full(
                n_options,
                stomp_result.pseudo_reward,
                dtype=jnp.float32,
            ),
            jnp.zeros(n_options, dtype=jnp.float32),
        )
        proposed_state = OaKState(
            stomp_state=stomp_result.state,
            execution_counts=new_execution_counts,
            cumulative_pseudo_rewards=new_cumulative_pseudo_rewards,
            utility_ema=new_utility_ema,
            step_count=expected_step_count,
            step_words=proposed_step_words,
        )
        post_nested_counter_aligned = jnp.all(stomp_result.state.step_words == proposed_step_words)
        proposed_state_valid = (
            _oak_outer_state_validity(proposed_state, cfg)[-1]
            & self._stomp.state_valid(proposed_state.stomp_state)
            & post_nested_counter_aligned
        )
        transaction_applied = (
            outer_state_valid
            & outer_counter_valid
            & outer_capacity
            & nested_counter_aligned
            & source_state_valid
            & source_state_matches
            & result_clock_binding_valid
            & result_endpoint_binding_valid
            & result_diagnostics_valid
            & proposed_state_valid
        )
        new_state = jax.lax.cond(
            transaction_applied,
            lambda: proposed_state,
            lambda: state,
        )
        update = OaKUpdateResult(
            state=new_state,
            td_error=jnp.where(
                transaction_applied,
                stomp_result.td_error,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            average_reward=jnp.where(
                transaction_applied,
                stomp_result.average_reward,
                state.stomp_state.base_average_reward,
            ),
            primitive_action=jnp.where(
                transaction_applied,
                stomp_result.primitive_action,
                state.stomp_state.last_primitive_action,
            ),
            executing_option=jnp.where(
                transaction_applied,
                stomp_result.executing_option,
                state.stomp_state.executing_option,
            ),
            option_terminated=transaction_applied & stomp_result.option_terminated,
            pseudo_reward=jnp.where(
                transaction_applied,
                stomp_result.pseudo_reward,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            utility_ema=jnp.where(
                transaction_applied,
                new_utility_ema,
                state.utility_ema,
            ),
            planning_backups=jnp.where(
                transaction_applied,
                stomp_result.planning_backups,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            planning_td_error=jnp.where(
                transaction_applied,
                stomp_result.planning_td_error,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            pre_step_words=state.step_words,
            post_step_words=new_state.step_words,
            outer_state_valid=outer_state_valid,
            lifetime_counter_valid=outer_counter_valid,
            lifetime_capacity_available=outer_capacity,
            nested_counter_aligned=nested_counter_aligned,
            nested_update_applied=transaction_applied & stomp_result.update_applied,
            proposed_state_valid=proposed_state_valid,
            update_applied=transaction_applied,
        )
        return OaKSTOMPUpdateAdoptionResult(
            update=update,
            source_state_valid=source_state_valid,
            source_state_matches=source_state_matches,
            result_static_contract_valid=jnp.asarray(True, dtype=jnp.bool_),
            result_clock_binding_valid=result_clock_binding_valid,
            result_endpoint_binding_valid=result_endpoint_binding_valid,
            result_diagnostics_valid=result_diagnostics_valid,
            derivation_recomputed=jnp.asarray(False, dtype=jnp.bool_),
            caller_authority_required=jnp.asarray(True, dtype=jnp.bool_),
            caller_authenticated=jnp.asarray(False, dtype=jnp.bool_),
            transaction_applied=transaction_applied,
        )

    def rebind_option_slots(
        self,
        state: OaKState,
        rebound_stomp_state: STOMPState,
        reset_slot_mask: Array,
    ) -> OaKOptionSlotRebindResult:
        """Adopt an exact quiescent STOMP option-slot rebind.

        Only reset-slot intra-option policy/model leaves and their corresponding
        base extended-action heads may differ.  Global learner state, primitive
        heads, unchanged option slots, every clock, and the live policy RNG must
        remain bit-identical.  On success only the reset slots' OaK execution,
        pseudo-reward, and utility statistics are zeroed.
        """

        if type(state) is not OaKState:
            raise TypeError("state must be an OaKState")
        if type(rebound_stomp_state) is not STOMPState:
            raise TypeError("rebound_stomp_state must be a STOMPState")
        reset_slots = jnp.asarray(reset_slot_mask)
        if reset_slots.shape != (self._config.n_options,):
            raise ValueError(
                "reset_slot_mask must have shape "
                f"({self._config.n_options},), got {reset_slots.shape}"
            )
        if reset_slots.dtype != jnp.bool_:
            raise TypeError(f"reset_slot_mask must have dtype bool, got {reset_slots.dtype}")

        source_state_valid = (
            _oak_outer_state_validity(state, self._config)[-1]
            & self._stomp.state_valid(state.stomp_state)
            & jnp.all(state.step_words == state.stomp_state.step_words)
        )
        rebound_state_valid = self._stomp.state_valid(rebound_stomp_state)
        source_quiescent = state.stomp_state.executing_option < 0
        reset_requested = jnp.any(reset_slots)
        clocks_preserved = (
            jnp.array_equal(
                rebound_stomp_state.step_words,
                state.stomp_state.step_words,
            )
            & (rebound_stomp_state.step_count == state.stomp_state.step_count)
            & jnp.array_equal(
                rebound_stomp_state.base_learner_state.step_words,
                state.stomp_state.base_learner_state.step_words,
            )
            & (
                rebound_stomp_state.base_learner_state.step_count
                == state.stomp_state.base_learner_state.step_count
            )
        )
        policy_rng_preserved = jnp.array_equal(
            jr.key_data(rebound_stomp_state.rng_key),
            jr.key_data(state.stomp_state.rng_key),
        )
        expected_rebound = _merge_rebound_stomp_option_slots(
            state.stomp_state,
            rebound_stomp_state,
            reset_slots,
            n_primitive_actions=self._config.n_primitive_actions,
        )
        unchanged_slots_preserved = _exact_tree_equal(
            rebound_stomp_state,
            expected_rebound,
        )
        proposed_state = cast(
            OaKState,
            state.replace(
                stomp_state=rebound_stomp_state,
                execution_counts=jnp.where(
                    reset_slots,
                    jnp.zeros_like(state.execution_counts),
                    state.execution_counts,
                ),
                cumulative_pseudo_rewards=jnp.where(
                    reset_slots,
                    jnp.zeros_like(state.cumulative_pseudo_rewards),
                    state.cumulative_pseudo_rewards,
                ),
                utility_ema=jnp.where(
                    reset_slots,
                    jnp.zeros_like(state.utility_ema),
                    state.utility_ema,
                ),
            ),
        )
        proposed_state_valid = (
            _oak_outer_state_validity(proposed_state, self._config)[-1]
            & self._stomp.state_valid(proposed_state.stomp_state)
            & jnp.all(proposed_state.step_words == proposed_state.stomp_state.step_words)
        )
        transaction_applied = (
            source_state_valid
            & rebound_state_valid
            & source_quiescent
            & reset_requested
            & clocks_preserved
            & policy_rng_preserved
            & unchanged_slots_preserved
            & proposed_state_valid
        )
        next_state = jax.lax.cond(
            transaction_applied,
            lambda: proposed_state,
            lambda: state,
        )
        return OaKOptionSlotRebindResult(
            state=next_state,
            reset_slot_mask=reset_slots,
            source_state_valid=source_state_valid,
            rebound_state_valid=rebound_state_valid,
            source_quiescent=source_quiescent,
            reset_requested=reset_requested,
            clocks_preserved=clocks_preserved,
            policy_rng_preserved=policy_rng_preserved,
            unchanged_slots_preserved=unchanged_slots_preserved,
            proposed_state_valid=proposed_state_valid,
            transaction_applied=transaction_applied,
        )

    def external_stomp_adoption_resource_budget(
        self,
        state: OaKState,
    ) -> OaKExternalSTOMPAdoptionResourceBudget:
        """Report exact persistent cost and the adoption seam's trust limit."""

        persistent_nbytes = measure_oak_state_nbytes(state)
        return OaKExternalSTOMPAdoptionResourceBudget(
            persistent_state_nbytes_before=persistent_nbytes,
            persistent_state_nbytes_after=persistent_nbytes,
            persistent_state_growth_bytes=0,
            stomp_update_evaluations_per_adopt=0,
            stomp_update_evaluations_per_delegated_update=1,
            derivation_recomputed_on_adopt=False,
            source_result_integrity_checked=True,
            caller_authority_required=True,
            caller_authenticated=False,
        )

    def update(
        self,
        state: OaKState,
        env_reward: Array,
        next_observation: Array,
        discount: Array | None = None,
        *,
        decision_observation: Array | None = None,
        execution_boundary: Array | bool = False,
        extended_action_mask: Array | None = None,
        enable_option_planning: bool = True,
        preselection_feature_reset_mask: Array | None = None,
    ) -> OaKUpdateResult:
        """Process one real-time primitive STOMP + utility-tracking step.

        All branching is via ``jnp.where`` so this method is ``jax.lax.scan``
        compatible.

        Args:
            state: Current OaK state.
            env_reward: Scalar environment reward.
            next_observation: Next observation from the environment.
            discount: Effective continuation multiplier. ``None`` preserves
                STOMP's historical primitive/``option_gamma`` behavior.
            decision_observation: Optional distinct observation for selecting
                the next action after an autoreset boundary. Learning always
                uses ``next_observation``.
            execution_boundary: Whether this transition interrupts the
                current option lifecycle without zeroing ``discount``.
            extended_action_mask: Optional live-action mask spanning primitive
                actions followed by options. It is forwarded to the sole STOMP
                evaluation for selection, bootstrap, and planning eligibility.

        Returns:
            :class:`OaKUpdateResult` with new state and per-step diagnostics.
        """
        return self.update_with_stomp_trace(
            state,
            env_reward,
            next_observation,
            discount,
            decision_observation=decision_observation,
            execution_boundary=execution_boundary,
            extended_action_mask=extended_action_mask,
            enable_option_planning=enable_option_planning,
            preselection_feature_reset_mask=preselection_feature_reset_mask,
        ).update

    def update_with_stomp_trace(
        self,
        state: OaKState,
        env_reward: Array,
        next_observation: Array,
        discount: Array | None = None,
        *,
        decision_observation: Array | None = None,
        execution_boundary: Array | bool = False,
        extended_action_mask: Array | None = None,
        enable_option_planning: bool = True,
        preselection_feature_reset_mask: Array | None = None,
    ) -> OaKTracedUpdateResult:
        """Evaluate STOMP exactly once and return its transient adoption trace."""

        stomp_result: STOMPUpdateResult = self._stomp.update(
            state.stomp_state,
            env_reward,
            next_observation,
            discount,
            decision_observation=decision_observation,
            execution_boundary=execution_boundary,
            extended_action_mask=extended_action_mask,
            enable_planning=enable_option_planning,
            preselection_feature_reset_mask=(preselection_feature_reset_mask),
        )
        update = self.adopt_stomp_update(
            state,
            source_state=state,
            stomp_result=stomp_result,
        ).update
        return OaKTracedUpdateResult(
            update=update,
            stomp_result=stomp_result,
            stomp_update_evaluations=jnp.asarray(1, dtype=jnp.int32),
            derivation_recomputed=jnp.asarray(False, dtype=jnp.bool_),
        )

    def scan(
        self,
        state: OaKState,
        env_rewards: Array,
        next_observations: Array,
        discounts: Array | None = None,
        *,
        decision_observations: Array | None = None,
        execution_boundaries: Array | None = None,
        extended_action_masks: Array | None = None,
    ) -> OaKArrayResult:
        """Run OaK over arrays with optional boundary and live-option masks."""

        def step_fn(
            carry: OaKState,
            inputs: tuple[Array, Array, Array, Array, Array, Array],
        ) -> tuple[OaKState, tuple[Array, ...]]:
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
                result.utility_ema,
                result.planning_backups,
                result.planning_td_error,
                result.pre_step_words,
                result.post_step_words,
                result.outer_state_valid,
                result.lifetime_counter_valid,
                result.lifetime_capacity_available,
                result.nested_counter_aligned,
                result.nested_update_applied,
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
                (env_rewards.shape[0], self._config.stomp.n_total_actions),
                dtype=jnp.bool_,
            )
            if extended_action_masks is None
            else jnp.asarray(extended_action_masks)
        )

        (
            final_state,
            (
                td_errors,
                average_rewards,
                primitive_actions,
                executing_options,
                option_terminations,
                pseudo_rewards,
                utility_emas,
                planning_backups,
                planning_td_errors,
                pre_step_words,
                post_step_words,
                outer_state_valid,
                lifetime_counter_valid,
                lifetime_capacity_available,
                nested_counter_aligned,
                nested_update_applied,
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

        return OaKArrayResult(
            state=final_state,
            td_errors=td_errors,
            average_rewards=average_rewards,
            primitive_actions=primitive_actions,
            executing_options=executing_options,
            option_terminations=option_terminations,
            pseudo_rewards=pseudo_rewards,
            utility_emas=utility_emas,
            planning_backups=planning_backups,
            planning_td_errors=planning_td_errors,
            pre_step_words=pre_step_words,
            post_step_words=post_step_words,
            outer_state_valid=outer_state_valid,
            lifetime_counter_valid=lifetime_counter_valid,
            lifetime_capacity_available=lifetime_capacity_available,
            nested_counter_aligned=nested_counter_aligned,
            nested_update_applied=nested_update_applied,
            proposed_state_valid=proposed_state_valid,
            update_applied=update_applied,
        )

    def curate(
        self,
        state: OaKState,
        key: Array,
        available_feature_indices: list[int] | None = None,
    ) -> tuple[OaKAgent, OaKState]:
        """Replace the lowest-utility option with a new subtask.

        The replacement creates a new :class:`OaKAgent` with updated subtask
        specs.  The replaced option's Q-weights, eligibility traces, option
        model, and utility statistics are reset to initial values.

        Curation is a **Python-level operation** — it runs outside
        ``jax.lax.scan`` / JIT and materialises JAX array values.

        When ``config.min_steps_before_curation > 0`` and fewer primitive
        steps have elapsed, no eviction occurs and ``(self, state)`` is
        returned unchanged. Curation is also deferred when the lowest-utility
        option is currently executing, so a subtask specification is never
        changed in the middle of its trajectory.

        Args:
            state: Current OaK state.
            key: JAX PRNG key for selecting the replacement feature index.
            available_feature_indices: Pool of feature indices to draw the
                new subtask from.  Defaults to all indices not already in
                use; if all are in use, samples from the full range.

        Returns:
            ``(new_agent, new_state)`` where ``new_agent`` has updated subtask
            specs and ``new_state`` has the replaced option's arrays zeroed.
        """
        cfg = self._config
        if cfg.n_options == 0:
            return self, state
        utility = state.utility_ema

        outer_valid = _oak_outer_state_validity(state, cfg)[-1]
        nested_aligned = jnp.all(state.step_words == state.stomp_state.step_words)
        if not bool(outer_valid & nested_aligned & self._stomp.state_valid(state.stomp_state)):
            return self, state

        # Minimum-uptime guard: never evict before utilities have had time
        # to be learned (untrained EMAs would make the choice arbitrary)
        if not bool(
            _lifetime_words_at_least(
                state.step_words,
                cfg.min_steps_before_curation,
            )
        ):
            return self, state

        # Find option with lowest utility
        worst_idx = int(jnp.argmin(utility))
        worst_utility = float(utility[worst_idx])

        # Skip if utility is above threshold and threshold > 0
        if cfg.curation_threshold > 0.0 and worst_utility >= cfg.curation_threshold:
            return self, state

        # Replacing an active option would reinterpret its already-dispatched
        # action and accumulated trajectory under a different SubtaskSpec.
        # Defer this curation attempt until the option has terminated.
        if int(state.stomp_state.executing_option) == worst_idx:
            return self, state

        # Pick replacement feature index
        current_feat_indices = {s.feature_index for s in cfg.stomp.subtask_specs}
        obs_dim = cfg.observation_dim
        if available_feature_indices is None:
            pool = [i for i in range(obs_dim) if i not in current_feat_indices]
            if not pool:
                pool = list(range(obs_dim))
        else:
            pool = list(available_feature_indices)

        key, subkey = jr.split(key)
        new_feat_idx = pool[int(jr.randint(subkey, (), 0, len(pool)))]

        # Build new spec list (preserve threshold / scale / max_steps)
        new_specs = list(cfg.stomp.subtask_specs)
        old = new_specs[worst_idx]
        new_specs[worst_idx] = SubtaskSpec(
            feature_index=new_feat_idx,
            threshold=old.threshold,
            pseudo_reward_scale=old.pseudo_reward_scale,
            max_option_steps=old.max_option_steps,
        )

        # Reset STOMP state for the replaced option
        idx = worst_idx
        n_prim = cfg.n_primitive_actions

        new_op_weights = state.stomp_state.option_policies.q_weights.at[idx].set(
            jnp.zeros_like(state.stomp_state.option_policies.q_weights[idx])
        )
        new_op_traces = state.stomp_state.option_policies.traces.at[idx].set(
            jnp.zeros_like(state.stomp_state.option_policies.traces[idx])
        )
        new_op_average_rewards = state.stomp_state.option_policies.average_rewards.at[idx].set(0.0)
        new_option_policies = cast(
            IntraOptionPoliciesState,
            state.stomp_state.option_policies.replace(
                q_weights=new_op_weights,
                traces=new_op_traces,
                average_rewards=new_op_average_rewards,
            ),
        )

        new_ns_weights = state.stomp_state.option_models.next_state_weights.at[idx].set(
            jnp.zeros_like(state.stomp_state.option_models.next_state_weights[idx])
        )
        new_option_models = cast(
            OptionModelsState,
            state.stomp_state.option_models.replace(
                cumreward_ema=state.stomp_state.option_models.cumreward_ema.at[idx].set(0.0),
                env_return_ema=state.stomp_state.option_models.env_return_ema.at[idx].set(0.0),
                duration_ema=state.stomp_state.option_models.duration_ema.at[idx].set(0.0),
                baseline_mass_ema=state.stomp_state.option_models.baseline_mass_ema.at[idx].set(
                    0.0
                ),
                discount_ema=state.stomp_state.option_models.discount_ema.at[idx].set(1.0),
                next_state_weights=new_ns_weights,
                n_completions=state.stomp_state.option_models.n_completions.at[idx].set(0),
            ),
        )

        base_action_idx = n_prim + idx
        ls = state.stomp_state.base_learner_state
        new_head_weights = tuple(
            jnp.zeros_like(w) if i == base_action_idx else w
            for i, w in enumerate(ls.head_params.weights)
        )
        new_head_biases = tuple(
            jnp.zeros_like(b) if i == base_action_idx else b
            for i, b in enumerate(ls.head_params.biases)
        )
        new_head_traces = tuple(
            (jnp.zeros_like(tw), jnp.zeros_like(tb)) if i == base_action_idx else (tw, tb)
            for i, (tw, tb) in enumerate(ls.head_traces)
        )
        # Optimizer state must be reset to a *fresh init*, not zeros: LMS
        # stores the step-size itself and IDBD stores log step-sizes
        # (exp(0) = 1.0), so zeroed state would freeze or corrupt learning.
        key, init_key = jr.split(key)
        fresh_ls = self._stomp.base_learner.init(cfg.observation_dim, init_key)
        new_head_opt_states = tuple(
            fresh_ls.head_optimizer_states[i] if i == base_action_idx else opt
            for i, opt in enumerate(ls.head_optimizer_states)
        )
        new_base_learner_state = ls.replace(
            head_params=MLPParams(weights=new_head_weights, biases=new_head_biases),
            head_traces=new_head_traces,
            head_optimizer_states=new_head_opt_states,
        )

        new_stomp_state = cast(
            STOMPState,
            state.stomp_state.replace(
                base_learner_state=new_base_learner_state,
                option_policies=new_option_policies,
                option_models=new_option_models,
            ),
        )

        # Reset utility stats for replaced option
        replace_mask = jnp.arange(cfg.n_options, dtype=jnp.int32) == idx
        new_state = OaKState(
            stomp_state=new_stomp_state,
            execution_counts=jnp.where(replace_mask, 0, state.execution_counts),
            cumulative_pseudo_rewards=jnp.where(replace_mask, 0.0, state.cumulative_pseudo_rewards),
            utility_ema=jnp.where(replace_mask, 0.0, state.utility_ema),
            step_count=state.step_count,
            step_words=state.step_words,
        )

        # Build new agent with updated config
        new_stomp_cfg = dataclasses.replace(cfg.stomp, subtask_specs=tuple(new_specs))
        new_oak_cfg = dataclasses.replace(cfg, stomp=new_stomp_cfg)
        return OaKAgent(new_oak_cfg), new_state

    def keyboard_q_values(
        self,
        state: OaKState,
        observation: Array,
        keyboard_vector: Array,
    ) -> Array:
        """Compute blended Q-values for a keyboard chord vector."""
        return keyboard_q_values(state.stomp_state, observation, keyboard_vector)

    def propose_keyboard_policy(
        self,
        state: OaKState,
        decision_observation: Array,
        keyboard_vector: Array,
    ) -> OaKKeyboardPolicyProposal:
        """Return a strict deterministic proposal for one fixed chord.

        The observation must be the exact float32 observation already stored
        by STOMP for its current dispatched action. The chord must be a finite
        float32 vector of shape ``(n_options,)`` with positive L1 norm. This is
        a pure counterfactual query: it consumes no RNG and changes no state.

        Invalid values or static input mismatches return ``available=False``
        with finite sentinel diagnostics.
        """

        raw_observation = jnp.asarray(decision_observation)
        observation_static_contract_valid = (
            raw_observation.shape == (self._config.observation_dim,)
            and raw_observation.dtype == jnp.float32
        )
        observation = (
            raw_observation
            if observation_static_contract_valid
            else jnp.zeros((self._config.observation_dim,), dtype=jnp.float32)
        )
        raw_chord = jnp.asarray(keyboard_vector)
        keyboard_vector_static_contract_valid = (
            raw_chord.shape == (self._config.n_options,) and raw_chord.dtype == jnp.float32
        )
        chord = (
            raw_chord
            if keyboard_vector_static_contract_valid
            else jnp.zeros((self._config.n_options,), dtype=jnp.float32)
        )

        observation_valid = jnp.asarray(
            observation_static_contract_valid, dtype=jnp.bool_
        ) & jnp.all(jnp.isfinite(observation))
        chord_l1 = jnp.sum(jnp.abs(chord))
        keyboard_vector_valid = (
            jnp.asarray(keyboard_vector_static_contract_valid, dtype=jnp.bool_)
            & jnp.all(jnp.isfinite(chord))
            & jnp.isfinite(chord_l1)
            & (chord_l1 > 0.0)
        )
        safe_observation = jnp.where(observation_valid, observation, 0.0)
        safe_chord = jnp.where(keyboard_vector_valid, chord, 0.0)
        raw_q_values = keyboard_q_values(
            state.stomp_state,
            safe_observation,
            safe_chord,
        )
        q_values_valid = jnp.all(jnp.isfinite(raw_q_values))
        safe_q_values = jnp.where(q_values_valid, raw_q_values, 0.0)
        candidate_action = jnp.argmax(safe_q_values).astype(jnp.int32)

        ownership_audit = replace_dispatched_primitive_action(
            state.stomp_state,
            observation,
            candidate_action,
        ).decision
        (
            outer_static_contract_valid,
            outer_values_finite,
            outer_counters_valid,
            outer_state_valid,
        ) = _oak_outer_state_validity(state, self._config)
        whole_state_valid = ownership_audit.state_valid & outer_state_valid
        available = (
            observation_valid
            & keyboard_vector_valid
            & q_values_valid
            & whole_state_valid
            & ownership_audit.observation_matches
        )
        declared_score = safe_q_values[candidate_action]
        return OaKKeyboardPolicyProposal(
            available=available,
            action=jnp.where(available, candidate_action, jnp.int32(-1)),
            declared_score=jnp.where(available, declared_score, jnp.float32(0.0)),
            decision_observation=jnp.where(observation_valid, observation, 0.0),
            keyboard_vector=jnp.where(keyboard_vector_valid, chord, 0.0),
            q_values=safe_q_values,
            outer_state_static_contract_valid=outer_static_contract_valid,
            outer_state_values_finite=outer_values_finite,
            outer_state_counters_valid=outer_counters_valid,
            outer_state_valid=outer_state_valid,
            stomp_state_valid=ownership_audit.state_valid,
            state_valid=whole_state_valid,
            observation_static_contract_valid=jnp.asarray(
                observation_static_contract_valid, dtype=jnp.bool_
            ),
            observation_valid=observation_valid,
            observation_matches=(
                jnp.asarray(observation_static_contract_valid, dtype=jnp.bool_)
                & ownership_audit.observation_matches
            ),
            keyboard_vector_static_contract_valid=jnp.asarray(
                keyboard_vector_static_contract_valid, dtype=jnp.bool_
            ),
            keyboard_vector_valid=keyboard_vector_valid,
            q_values_valid=q_values_valid,
        )

    def dispatch_keyboard_policy(
        self,
        state: OaKState,
        decision_observation: Array,
        keyboard_vector: Array,
        safety_action_mask: Array | None = None,
    ) -> OaKKeyboardDispatchResult:
        """Propose and atomically commit one ownership-correct chord action.

        The unconstrained keyboard argmax is audited against the optional hard
        primitive-action mask. An unsafe proposal uses the independently safe
        already-selected OaK action. Invalid inputs or an unsafe base action
        fail closed with effective action ``-1`` and exact state preservation.
        """

        proposal = self.propose_keyboard_policy(
            state,
            decision_observation,
            keyboard_vector,
        )
        replacement = replace_dispatched_primitive_action(
            state.stomp_state,
            decision_observation,
            proposal.action,
            safety_action_mask=safety_action_mask,
        )
        return OaKKeyboardDispatchResult(
            state=cast(
                OaKState,
                state.replace(stomp_state=replacement.state),
            ),
            decision=OaKKeyboardDispatchDecision(
                proposal=proposal,
                replacement=replacement.decision,
            ),
        )

    def keyboard_action(
        self,
        state: OaKState,
        observation: Array,
        keyboard_vector: Array,
        key: Array,
        *,
        epsilon: float = 0.0,
    ) -> tuple[Array, Array]:
        """Select a primitive action for a chord keyboard vector."""
        return keyboard_action(
            state.stomp_state,
            observation,
            keyboard_vector,
            key,
            epsilon=epsilon,
            n_primitive_actions=self._config.n_primitive_actions,
        )


__all__ = [
    "KeyboardChordLearnerConfig",
    "KeyboardChordLearnerState",
    "OAK_LIFETIME_COUNTER_DELTA_NBYTES",
    "OAK_LIFETIME_COUNTER_NBYTES",
    "OAK_STATE_SCHEMA",
    "OaKAgent",
    "OaKArrayResult",
    "OaKConfig",
    "OaKExternalSTOMPAdoptionResourceBudget",
    "OaKKeyboardDispatchDecision",
    "OaKKeyboardDispatchResult",
    "OaKKeyboardPolicyProposal",
    "OaKOptionSlotRebindResult",
    "OaKState",
    "OaKSTOMPUpdateAdoptionResult",
    "OaKTracedUpdateResult",
    "OaKUpdateResult",
    "init_keyboard_chord_learner",
    "keyboard_action",
    "keyboard_q_values",
    "learned_feature_subtask_specs",
    "measure_oak_state_nbytes",
    "measure_oak_wrapper_state_nbytes",
    "migrate_legacy_oak_state",
    "oak_lifetime_counter_nbytes",
    "oak_total_lifetime_counter_nbytes",
    "update_keyboard_chord_learner",
]
