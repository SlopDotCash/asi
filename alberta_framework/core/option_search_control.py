# mypy: disable-error-code="attr-defined,call-arg"
"""Bounded support-aware search control for learned STOMP option models.

This module is an opt-in L0 composition over the public STOMP surfaces.  It
does not alter option models, action ownership, the real average-reward rate,
or policy RNG.  At one real decision observation it ranks completed option
models by the absolute differential semi-MDP Bellman residual and applies a
fixed number of base-value backups.  Candidate priorities are recomputed after
every accepted backup.

This boundary is value integration only.  It never refreshes an action that
OaK/STOMP already selected and cached for dispatch.  An accepted value update
can first affect behavior at the next extended-action selection boundary,
which may be several primitive decisions later while an option is active.

Completion count is only an observed-support gate.  It is not calibrated
uncertainty, reachability, model validity evidence, or a benefit claim.
Likewise, Bellman-residual magnitude is a transparent L0 search priority, not
a universal learning-value score or a learned search controller.
"""

from __future__ import annotations

import dataclasses
import operator
from collections.abc import Mapping
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core.multi_head_learner import MultiHeadMLPState
from alberta_framework.core.options import (
    OptionModelsState,
    STOMPAgent,
    STOMPState,
    replace_dispatched_primitive_action,
)
from alberta_framework.core.types import LMSState, MLPParams

OPTION_SEARCH_CONTROL_CONFIG_SCHEMA = "alberta.option-search-control.config.v1"
OPTION_SEARCH_CONTROL_MECHANISM_STATUS = "development_mechanism_only"
OPTION_SEARCH_CONTROL_SCIENTIFIC_PROMOTION_ALLOWED = False

_CONFIG_TYPE = "OptionSearchControlConfig"
_MAX_BACKUP_BUDGET = 4_096
_MAX_CANDIDATE_DIAGNOSTIC_SLOTS = 262_144
_INT32_MAX = 2_147_483_647
_ACTUAL_INT_TYPES = frozenset({int, *(np.dtype(code).type for code in "bBhHiIlLqQpP")})


def _require_int32(name: str, value: object, *, minimum: int = 1, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


@dataclasses.dataclass(frozen=True)
class OptionSearchControlConfig:
    """Static work and observed-support bounds for one search call."""

    backup_budget: int = 1
    min_model_completions: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backup_budget",
            _require_int32(
                "backup_budget",
                self.backup_budget,
                minimum=1,
                maximum=_MAX_BACKUP_BUDGET,
            ),
        )
        object.__setattr__(
            self,
            "min_model_completions",
            _require_int32(
                "min_model_completions",
                self.min_model_completions,
                minimum=1,
                maximum=_INT32_MAX,
            ),
        )

    def to_config(self) -> dict[str, object]:
        """Return the exact JSON-compatible L0 configuration."""

        return {
            "schema": OPTION_SEARCH_CONTROL_CONFIG_SCHEMA,
            "type": _CONFIG_TYPE,
            "mechanism_status": OPTION_SEARCH_CONTROL_MECHANISM_STATUS,
            "scientific_promotion_allowed": (OPTION_SEARCH_CONTROL_SCIENTIFIC_PROMOTION_ALLOWED),
            "backup_budget": self.backup_budget,
            "min_model_completions": self.min_model_completions,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> OptionSearchControlConfig:
        """Strictly reconstruct only the exact v1 configuration schema."""

        if type(config) is not dict:
            raise ValueError("option search control config must be an actual dict")
        payload = dict(config)
        expected = {
            "schema",
            "type",
            "mechanism_status",
            "scientific_promotion_allowed",
            "backup_budget",
            "min_model_completions",
        }
        if set(payload) != expected:
            raise ValueError("option search control config fields do not match v1")
        if payload.pop("schema") != OPTION_SEARCH_CONTROL_CONFIG_SCHEMA:
            raise ValueError("unexpected option search control config schema")
        if payload.pop("type") != _CONFIG_TYPE:
            raise ValueError("unexpected option search control config type")
        if payload.pop("mechanism_status") != OPTION_SEARCH_CONTROL_MECHANISM_STATUS:
            raise ValueError("option search control must remain mechanism-only")
        if payload.pop("scientific_promotion_allowed") is not False:
            raise ValueError("option search control config cannot claim promotion")
        if type(payload["backup_budget"]) is not int:
            raise ValueError("serialized backup_budget must be a JSON integer")
        if type(payload["min_model_completions"]) is not int:
            raise ValueError("serialized min_model_completions must be a JSON integer")
        return cls(**cast(dict[str, Any], payload))


@dataclasses.dataclass(frozen=True)
class OptionSearchControlResourceBudget:
    """Exact logical work bounds for the stateless option search boundary."""

    n_options: int
    observation_dim: int
    backup_budget: int
    persistent_state_bytes: int
    rng_draws_per_call: int
    candidate_values_per_evaluation: int
    max_candidate_evaluations_per_call: int
    max_base_learner_updates_per_call: int
    max_model_matrix_vector_products_per_call: int
    max_base_value_forward_calls_per_call: int
    max_base_value_backward_calls_per_call: int
    stomp_self_audits_per_call: int
    max_diagnostic_payload_bytes_per_call: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_options",
            _require_int32("n_options", self.n_options, minimum=1),
        )
        object.__setattr__(
            self,
            "observation_dim",
            _require_int32("observation_dim", self.observation_dim, minimum=1),
        )
        object.__setattr__(
            self,
            "backup_budget",
            _require_int32("backup_budget", self.backup_budget, minimum=1),
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
            "rng_draws_per_call",
            _require_int32("rng_draws_per_call", self.rng_draws_per_call, minimum=0),
        )
        object.__setattr__(
            self,
            "candidate_values_per_evaluation",
            _require_int32(
                "candidate_values_per_evaluation",
                self.candidate_values_per_evaluation,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "max_candidate_evaluations_per_call",
            _require_int32(
                "max_candidate_evaluations_per_call",
                self.max_candidate_evaluations_per_call,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "max_base_learner_updates_per_call",
            _require_int32(
                "max_base_learner_updates_per_call",
                self.max_base_learner_updates_per_call,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "max_model_matrix_vector_products_per_call",
            _require_int32(
                "max_model_matrix_vector_products_per_call",
                self.max_model_matrix_vector_products_per_call,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "max_base_value_forward_calls_per_call",
            _require_int32(
                "max_base_value_forward_calls_per_call",
                self.max_base_value_forward_calls_per_call,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "max_base_value_backward_calls_per_call",
            _require_int32(
                "max_base_value_backward_calls_per_call",
                self.max_base_value_backward_calls_per_call,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "stomp_self_audits_per_call",
            _require_int32(
                "stomp_self_audits_per_call",
                self.stomp_self_audits_per_call,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "max_diagnostic_payload_bytes_per_call",
            _require_int32(
                "max_diagnostic_payload_bytes_per_call",
                self.max_diagnostic_payload_bytes_per_call,
                minimum=0,
            ),
        )
        candidate_slots = self.n_options * self.backup_budget
        diagnostic_bytes = (
            4 * self.observation_dim
            + 18
            + 4 * self.n_options
            + 16 * candidate_slots
            + 19 * self.backup_budget
            + 4
        )
        expected = {
            "persistent_state_bytes": 0,
            "rng_draws_per_call": 0,
            "candidate_values_per_evaluation": self.n_options,
            "max_candidate_evaluations_per_call": candidate_slots,
            "max_base_learner_updates_per_call": self.backup_budget,
            "max_model_matrix_vector_products_per_call": candidate_slots,
            "max_base_value_forward_calls_per_call": (
                self.n_options + 2
            ) * self.backup_budget,
            "max_base_value_backward_calls_per_call": self.backup_budget,
            "stomp_self_audits_per_call": 1,
            "max_diagnostic_payload_bytes_per_call": diagnostic_bytes,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} does not match the option-search implementation")

    def to_config(self) -> dict[str, int]:
        """Return the exact JSON-compatible logical resource record."""

        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class OptionSearchControlDiagnostics:
    """Complete per-backup candidate ranking and application audit.

    Invalid candidate values are represented by finite zero diagnostics plus
    false validity.  ``completion_counts`` is observed support only.
    Candidate arrays have shape ``(backup_budget, n_options)`` because scores
    are recomputed after each accepted backup.
    """

    decision_observation: Float[Array, " observation_dim"]
    decision_observation_static_contract_valid: Bool[Array, ""]
    decision_observation_values_finite: Bool[Array, ""]
    option_model_static_contract_valid: Bool[Array, ""]
    base_state_static_contract_valid: Bool[Array, ""]
    base_state_values_finite: Bool[Array, ""]
    option_model_values_finite: Bool[Array, ""]
    state_counters_valid: Bool[Array, ""]
    base_update_capacity_available: Bool[Array, ""]
    stomp_state_static_contract_valid: Bool[Array, ""]
    stomp_state_values_finite: Bool[Array, ""]
    stomp_rng_key_valid: Bool[Array, ""]
    stomp_action_ownership_valid: Bool[Array, ""]
    stomp_state_valid: Bool[Array, ""]
    decision_observation_matches_state: Bool[Array, ""]
    cached_decision_action_refreshed: Bool[Array, ""]
    value_effect_deferred_to_next_extended_action_selection: Bool[Array, ""]
    average_reward_valid: Bool[Array, ""]
    planner_inputs_valid: Bool[Array, ""]
    completion_counts: Int[Array, " n_options"]
    completion_supported: Bool[Array, "backup_budget n_options"]
    candidate_semantics_valid: Bool[Array, "backup_budget n_options"]
    candidate_predictions_finite: Bool[Array, "backup_budget n_options"]
    candidate_targets: Float[Array, "backup_budget n_options"]
    candidate_bellman_residuals: Float[Array, "backup_budget n_options"]
    candidate_priorities: Float[Array, "backup_budget n_options"]
    candidate_valid: Bool[Array, "backup_budget n_options"]
    selected_option_indices: Int[Array, " backup_budget"]
    selected_extended_action_indices: Int[Array, " backup_budget"]
    selected_priorities: Float[Array, " backup_budget"]
    td_errors: Float[Array, " backup_budget"]
    candidate_update_finite: Bool[Array, " backup_budget"]
    trace_isolation_preserved: Bool[Array, " backup_budget"]
    applied: Bool[Array, " backup_budget"]
    applied_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class OptionSearchControlResult:
    """STOMP state after a planner-only transaction and its exact audit."""

    state: STOMPState
    diagnostics: OptionSearchControlDiagnostics


def _floating_tree_is_finite(value: Any) -> Bool[Array, ""]:
    """Return whether every inexact leaf is finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree_util.tree_leaves(value):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(array))
    return valid


def _trees_exactly_equal(left: Any, right: Any) -> Bool[Array, ""]:
    """Return one JAX boolean for exact equality of matching array PyTrees."""

    predicate = jnp.asarray(True, dtype=jnp.bool_)
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return jnp.asarray(False, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        predicate = predicate & jnp.array_equal(
            jnp.asarray(left_leaf),
            jnp.asarray(right_leaf),
        )
    return predicate


def _all_integer_leaves_nonnegative(value: Any) -> Bool[Array, ""]:
    """Return whether every integer leaf is nonnegative."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree_util.tree_leaves(value):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.integer):
            valid = valid & jnp.all(array >= 0)
    return valid


def _array_has_contract(
    value: Any,
    shape: tuple[int, ...],
    dtype: Any,
) -> bool:
    """Return a nonthrowing exact array shape/dtype predicate."""

    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and value.shape == shape
        and value.dtype == dtype
    )


class OptionSearchControl:
    """Stateless stable Bellman-residual search over completed option models."""

    def __init__(
        self,
        stomp_agent: STOMPAgent,
        config: OptionSearchControlConfig | None = None,
    ) -> None:
        self._agent = stomp_agent
        self._config = config or OptionSearchControlConfig()
        n_options = self._agent.config.n_options
        if type(n_options) is not int:
            raise ValueError(
                f"n_options must be an integer in [1, {_MAX_BACKUP_BUDGET}] "
                "for option-search arange work"
            )
        if n_options < 1:
            raise ValueError("option search control requires at least one option")
        if n_options > _MAX_BACKUP_BUDGET:
            raise ValueError(
                f"n_options must be an integer in [1, {_MAX_BACKUP_BUDGET}] "
                "for option-search arange work"
            )
        self._n_options = n_options
        candidate_slots = n_options * self._config.backup_budget
        if candidate_slots > _MAX_CANDIDATE_DIAGNOSTIC_SLOTS:
            raise ValueError(
                "backup_budget * n_options exceeds the option-search "
                f"diagnostic slot ceiling ({_MAX_CANDIDATE_DIAGNOSTIC_SLOTS})"
            )

    @property
    def config(self) -> OptionSearchControlConfig:
        """Return the immutable search configuration."""

        return self._config

    @property
    def resource_budget(self) -> OptionSearchControlResourceBudget:
        """Return exact logical work maxima for one call."""

        n_options = self._n_options
        backup_budget = self._config.backup_budget
        candidate_slots = n_options * backup_budget
        # Dense logical payload: decision observation; eighteen boolean scalar
        # validity flags; completion counts; four boolean and three float32
        # candidate matrices; two int32, two float32, and three boolean backup
        # vectors; and the final int32 applied count.
        diagnostic_payload_bytes = (
            4 * self._agent.config.observation_dim
            + 18
            + 4 * n_options
            + 16 * candidate_slots
            + 19 * backup_budget
            + 4
        )
        # Every backup iteration performs one current-anchor Q forward, one
        # successor forward per option, and at most one learner-update
        # forward/backward pass for the selected candidate.
        return OptionSearchControlResourceBudget(
            n_options=n_options,
            observation_dim=self._agent.config.observation_dim,
            backup_budget=backup_budget,
            persistent_state_bytes=0,
            rng_draws_per_call=0,
            candidate_values_per_evaluation=n_options,
            max_candidate_evaluations_per_call=n_options * backup_budget,
            max_base_learner_updates_per_call=backup_budget,
            max_model_matrix_vector_products_per_call=n_options * backup_budget,
            max_base_value_forward_calls_per_call=(n_options + 2) * backup_budget,
            max_base_value_backward_calls_per_call=backup_budget,
            stomp_self_audits_per_call=1,
            max_diagnostic_payload_bytes_per_call=diagnostic_payload_bytes,
        )

    @staticmethod
    def _lms_state_static_contract_valid(state: Any) -> bool:
        """Return whether one STOMP optimizer leaf is scalar float32 LMS."""

        return isinstance(state, LMSState) and _array_has_contract(state.step_size, (), jnp.float32)

    def _base_state_static_contract_valid(self, state: STOMPState) -> bool:
        """Validate every base-learner collection before prediction/update."""

        learner = state.base_learner_state
        if not isinstance(learner, MultiHeadMLPState):
            return False
        hidden_sizes = self._agent.config.base_hidden_sizes
        n_heads = self._agent.config.n_total_actions
        trunk_depth = len(hidden_sizes)
        if not (
            isinstance(learner.trunk_params, MLPParams)
            and isinstance(learner.head_params, MLPParams)
            and isinstance(learner.trunk_params.weights, tuple)
            and isinstance(learner.trunk_params.biases, tuple)
            and isinstance(learner.trunk_optimizer_states, tuple)
            and isinstance(learner.trunk_traces, tuple)
            and isinstance(learner.hidden_unit_utilities, tuple)
            and isinstance(learner.head_params.weights, tuple)
            and isinstance(learner.head_params.biases, tuple)
            and isinstance(learner.head_optimizer_states, tuple)
            and isinstance(learner.head_traces, tuple)
            and len(learner.trunk_params.weights) == trunk_depth
            and len(learner.trunk_params.biases) == trunk_depth
            and len(learner.trunk_optimizer_states) == 2 * trunk_depth
            and len(learner.trunk_traces) == 2 * trunk_depth
            and len(learner.hidden_unit_utilities) == trunk_depth
            and len(learner.head_params.weights) == n_heads
            and len(learner.head_params.biases) == n_heads
            and len(learner.head_optimizer_states) == n_heads
            and len(learner.head_traces) == n_heads
            and learner.normalizer_state is None
            and _array_has_contract(learner.step_count, (), jnp.int32)
            and _array_has_contract(state.base_average_reward, (), jnp.float32)
            and _array_has_contract(state.option_steps, (), jnp.int32)
            and _array_has_contract(state.step_count, (), jnp.int32)
        ):
            return False

        input_width = self._agent.config.observation_dim
        for layer_index, output_width in enumerate(hidden_sizes):
            weight = learner.trunk_params.weights[layer_index]
            bias = learner.trunk_params.biases[layer_index]
            weight_trace = learner.trunk_traces[2 * layer_index]
            bias_trace = learner.trunk_traces[2 * layer_index + 1]
            utility = learner.hidden_unit_utilities[layer_index]
            if not (
                _array_has_contract(
                    weight,
                    (output_width, input_width),
                    jnp.float32,
                )
                and _array_has_contract(bias, (output_width,), jnp.float32)
                and _array_has_contract(
                    weight_trace,
                    (output_width, input_width),
                    jnp.float32,
                )
                and _array_has_contract(
                    bias_trace,
                    (output_width,),
                    jnp.float32,
                )
                and _array_has_contract(utility, (output_width,), jnp.float32)
                and self._lms_state_static_contract_valid(
                    learner.trunk_optimizer_states[2 * layer_index]
                )
                and self._lms_state_static_contract_valid(
                    learner.trunk_optimizer_states[2 * layer_index + 1]
                )
            ):
                return False
            input_width = output_width

        for head_index in range(n_heads):
            weight = learner.head_params.weights[head_index]
            bias = learner.head_params.biases[head_index]
            trace_pair = learner.head_traces[head_index]
            optimizer_pair = learner.head_optimizer_states[head_index]
            if not (
                isinstance(trace_pair, tuple)
                and len(trace_pair) == 2
                and isinstance(optimizer_pair, tuple)
                and len(optimizer_pair) == 2
            ):
                return False
            weight_trace, bias_trace = trace_pair
            if not (
                _array_has_contract(weight, (1, input_width), jnp.float32)
                and _array_has_contract(bias, (1,), jnp.float32)
                and _array_has_contract(
                    weight_trace,
                    (1, input_width),
                    jnp.float32,
                )
                and _array_has_contract(bias_trace, (1,), jnp.float32)
                and self._lms_state_static_contract_valid(optimizer_pair[0])
                and self._lms_state_static_contract_valid(optimizer_pair[1])
            ):
                return False
        return True

    def _option_model_static_contract_valid(self, state: STOMPState) -> bool:
        """Validate model shapes/dtypes before any indexed JAX computation."""

        models = state.option_models
        if not isinstance(models, OptionModelsState):
            return False
        n_options = self._n_options
        observation_dim = self._agent.config.observation_dim
        return (
            _array_has_contract(models.cumreward_ema, (n_options,), jnp.float32)
            and _array_has_contract(
                models.env_return_ema,
                (n_options,),
                jnp.float32,
            )
            and _array_has_contract(models.duration_ema, (n_options,), jnp.float32)
            and _array_has_contract(
                models.baseline_mass_ema,
                (n_options,),
                jnp.float32,
            )
            and _array_has_contract(models.discount_ema, (n_options,), jnp.float32)
            and _array_has_contract(
                models.next_state_weights,
                (n_options, observation_dim, observation_dim),
                jnp.float32,
            )
            and _array_has_contract(models.n_completions, (n_options,), jnp.int32)
        )

    def unavailable_diagnostics(
        self,
        decision_observation: Array,
        *,
        option_model_static_contract_valid: bool | Array = True,
        base_state_static_contract_valid: bool | Array = True,
        state_counters_valid: bool | Array = False,
        base_update_capacity_available: bool | Array = False,
        stomp_state_static_contract_valid: bool | Array = False,
        stomp_state_values_finite: bool | Array = False,
        stomp_rng_key_valid: bool | Array = False,
        stomp_action_ownership_valid: bool | Array = False,
        stomp_state_valid: bool | Array = False,
        decision_observation_matches_state: bool | Array = False,
    ) -> OptionSearchControlDiagnostics:
        """Return fixed finite diagnostics for a nonexecuted search call."""

        raw_observation = jnp.asarray(decision_observation)
        static_valid = (
            raw_observation.shape == (self._agent.config.observation_dim,)
            and raw_observation.dtype == jnp.float32
        )
        observation = (
            raw_observation
            if static_valid
            else jnp.zeros(
                (self._agent.config.observation_dim,),
                dtype=jnp.float32,
            )
        )
        values_finite = jnp.asarray(static_valid, dtype=jnp.bool_) & jnp.all(
            jnp.isfinite(observation)
        )
        budget = self._config.backup_budget
        n_options = self._n_options
        matrix_shape = (budget, n_options)
        return OptionSearchControlDiagnostics(
            decision_observation=jnp.where(values_finite, observation, 0.0),
            decision_observation_static_contract_valid=jnp.asarray(static_valid, dtype=jnp.bool_),
            decision_observation_values_finite=values_finite,
            option_model_static_contract_valid=jnp.asarray(
                option_model_static_contract_valid, dtype=jnp.bool_
            ),
            base_state_static_contract_valid=jnp.asarray(
                base_state_static_contract_valid, dtype=jnp.bool_
            ),
            base_state_values_finite=jnp.asarray(False, dtype=jnp.bool_),
            option_model_values_finite=jnp.asarray(False, dtype=jnp.bool_),
            state_counters_valid=jnp.asarray(state_counters_valid, dtype=jnp.bool_),
            base_update_capacity_available=jnp.asarray(
                base_update_capacity_available, dtype=jnp.bool_
            ),
            stomp_state_static_contract_valid=jnp.asarray(
                stomp_state_static_contract_valid, dtype=jnp.bool_
            ),
            stomp_state_values_finite=jnp.asarray(stomp_state_values_finite, dtype=jnp.bool_),
            stomp_rng_key_valid=jnp.asarray(stomp_rng_key_valid, dtype=jnp.bool_),
            stomp_action_ownership_valid=jnp.asarray(stomp_action_ownership_valid, dtype=jnp.bool_),
            stomp_state_valid=jnp.asarray(stomp_state_valid, dtype=jnp.bool_),
            decision_observation_matches_state=jnp.asarray(
                decision_observation_matches_state, dtype=jnp.bool_
            ),
            cached_decision_action_refreshed=jnp.asarray(False, dtype=jnp.bool_),
            value_effect_deferred_to_next_extended_action_selection=(
                jnp.asarray(False, dtype=jnp.bool_)
            ),
            average_reward_valid=jnp.asarray(False, dtype=jnp.bool_),
            planner_inputs_valid=jnp.asarray(False, dtype=jnp.bool_),
            completion_counts=jnp.zeros((n_options,), dtype=jnp.int32),
            completion_supported=jnp.zeros(matrix_shape, dtype=jnp.bool_),
            candidate_semantics_valid=jnp.zeros(matrix_shape, dtype=jnp.bool_),
            candidate_predictions_finite=jnp.zeros(matrix_shape, dtype=jnp.bool_),
            candidate_targets=jnp.zeros(matrix_shape, dtype=jnp.float32),
            candidate_bellman_residuals=jnp.zeros(matrix_shape, dtype=jnp.float32),
            candidate_priorities=jnp.zeros(matrix_shape, dtype=jnp.float32),
            candidate_valid=jnp.zeros(matrix_shape, dtype=jnp.bool_),
            selected_option_indices=jnp.full((budget,), -1, dtype=jnp.int32),
            selected_extended_action_indices=jnp.full((budget,), -1, dtype=jnp.int32),
            selected_priorities=jnp.zeros((budget,), dtype=jnp.float32),
            td_errors=jnp.zeros((budget,), dtype=jnp.float32),
            candidate_update_finite=jnp.zeros((budget,), dtype=jnp.bool_),
            trace_isolation_preserved=jnp.zeros((budget,), dtype=jnp.bool_),
            applied=jnp.zeros((budget,), dtype=jnp.bool_),
            applied_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def _evaluate_candidates(
        self,
        learner_state: MultiHeadMLPState,
        state: STOMPState,
        observation: Array,
        planner_inputs_valid: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        """Evaluate all option targets and residuals from one learner state."""

        models = state.option_models
        n_options = self._n_options
        base_values = self._agent.base_learner.predict(
            learner_state,
            observation,
        )
        option_values = base_values[self._agent.config.n_primitive_actions :]
        completion_supported = models.n_completions >= jnp.asarray(
            self._config.min_model_completions,
            dtype=jnp.int32,
        )
        candidate_semantics_valid = (
            (models.n_completions >= 0)
            & jnp.isfinite(models.env_return_ema)
            & jnp.isfinite(models.duration_ema)
            & (models.duration_ema > 0.0)
            & jnp.isfinite(models.baseline_mass_ema)
            & (models.baseline_mass_ema > 0.0)
            & jnp.isfinite(models.discount_ema)
            & (models.discount_ema >= 0.0)
            & (models.discount_ema <= 1.0)
            & jnp.all(jnp.isfinite(models.next_state_weights), axis=(1, 2))
        )

        indices = jnp.arange(n_options, dtype=jnp.int32)

        def evaluate_one(index: Array) -> tuple[Array, Array, Array, Array]:
            safe_model = completion_supported[index] & candidate_semantics_valid[index]
            weights = jnp.where(
                safe_model,
                models.next_state_weights[index],
                jnp.zeros_like(models.next_state_weights[index]),
            )
            predicted_next = observation + weights @ observation
            next_values = self._agent.base_learner.predict(
                learner_state,
                predicted_next,
            )
            discount = models.discount_ema[index]
            bootstrap = jnp.where(
                discount == 0.0,
                jnp.zeros_like(discount),
                discount * jnp.max(next_values),
            )
            target = (
                models.env_return_ema[index]
                - state.base_average_reward * models.baseline_mass_ema[index]
                + bootstrap
            )
            residual = target - option_values[index]
            prediction_finite = (
                ((discount == 0.0) | jnp.all(jnp.isfinite(predicted_next)))
                & ((discount == 0.0) | jnp.all(jnp.isfinite(next_values)))
                & jnp.isfinite(option_values[index])
                & jnp.isfinite(target)
                & jnp.isfinite(residual)
            )
            valid = planner_inputs_valid & safe_model & prediction_finite
            return (
                jnp.where(valid, target, jnp.float32(0.0)),
                jnp.where(valid, residual, jnp.float32(0.0)),
                prediction_finite,
                valid,
            )

        targets, residuals, predictions_finite, candidate_valid = jax.vmap(evaluate_one)(indices)
        priorities = jnp.where(candidate_valid, jnp.abs(residuals), 0.0)
        return (
            completion_supported,
            candidate_semantics_valid,
            predictions_finite,
            targets,
            residuals,
            candidate_valid,
            priorities,
        )

    def apply(
        self,
        state: STOMPState,
        decision_observation: Array,
    ) -> OptionSearchControlResult:
        """Apply fixed-budget planner-only backups at one real decision state.

        Stable ``argmax`` resolves exact priority ties by lowest option index.
        Each accepted learner update preserves the real eligibility traces and
        normalizer state exactly.  Invalid or unsupported candidates are
        finite diagnostic no-ops; no policy RNG is consumed.
        """

        if not isinstance(state, STOMPState):
            raise TypeError("state must be a STOMPState")

        raw_observation = jnp.asarray(decision_observation)
        observation_static_valid = (
            raw_observation.shape == (self._agent.config.observation_dim,)
            and raw_observation.dtype == jnp.float32
        )
        model_static_valid = self._option_model_static_contract_valid(state)
        base_static_valid = self._base_state_static_contract_valid(state)
        if not observation_static_valid or not model_static_valid or not base_static_valid:
            return OptionSearchControlResult(
                state=state,
                diagnostics=self.unavailable_diagnostics(
                    raw_observation,
                    option_model_static_contract_valid=model_static_valid,
                    base_state_static_contract_valid=base_static_valid,
                ),
            )

        observation = raw_observation
        try:
            stomp_audit = replace_dispatched_primitive_action(
                state,
                observation,
                state.last_primitive_action,
                jnp.ones(
                    (self._agent.config.n_primitive_actions,),
                    dtype=jnp.bool_,
                ),
            ).decision
        except (AttributeError, IndexError, TypeError):
            return OptionSearchControlResult(
                state=state,
                diagnostics=self.unavailable_diagnostics(
                    observation,
                    option_model_static_contract_valid=model_static_valid,
                    base_state_static_contract_valid=base_static_valid,
                ),
            )
        observation_values_finite = jnp.all(jnp.isfinite(observation))
        base_state_values_finite = _floating_tree_is_finite(state.base_learner_state)
        option_model_values_finite = _floating_tree_is_finite(state.option_models)
        state_counters_valid = stomp_audit.state_counters_valid & _all_integer_leaves_nonnegative(
            state.base_learner_state
        )
        max_step_count_before_call = jnp.asarray(
            _INT32_MAX - self._config.backup_budget,
            dtype=jnp.int32,
        )
        base_update_capacity_available = (
            state.base_learner_state.step_count <= max_step_count_before_call
        )
        average_reward_valid = jnp.isfinite(state.base_average_reward)
        planner_inputs_valid = (
            observation_values_finite
            & base_state_values_finite
            & option_model_values_finite
            & state_counters_valid
            & base_update_capacity_available
            & stomp_audit.state_valid
            & stomp_audit.observation_matches
            & average_reward_valid
        )
        safe_observation = jnp.where(
            observation_values_finite,
            observation,
            jnp.zeros_like(observation),
        )

        budget = self._config.backup_budget
        n_options = self._n_options
        matrix_shape = (budget, n_options)
        completion_supported_log = jnp.zeros(matrix_shape, dtype=jnp.bool_)
        semantics_valid_log = jnp.zeros(matrix_shape, dtype=jnp.bool_)
        predictions_finite_log = jnp.zeros(matrix_shape, dtype=jnp.bool_)
        targets_log = jnp.zeros(matrix_shape, dtype=jnp.float32)
        residuals_log = jnp.zeros(matrix_shape, dtype=jnp.float32)
        priorities_log = jnp.zeros(matrix_shape, dtype=jnp.float32)
        candidate_valid_log = jnp.zeros(matrix_shape, dtype=jnp.bool_)
        selected_options = jnp.full((budget,), -1, dtype=jnp.int32)
        selected_extended_actions = jnp.full((budget,), -1, dtype=jnp.int32)
        selected_priorities = jnp.zeros((budget,), dtype=jnp.float32)
        td_errors = jnp.zeros((budget,), dtype=jnp.float32)
        update_finite_log = jnp.zeros((budget,), dtype=jnp.bool_)
        trace_isolation_log = jnp.zeros((budget,), dtype=jnp.bool_)
        applied_log = jnp.zeros((budget,), dtype=jnp.bool_)

        carry = (
            state.base_learner_state,
            completion_supported_log,
            semantics_valid_log,
            predictions_finite_log,
            targets_log,
            residuals_log,
            priorities_log,
            candidate_valid_log,
            selected_options,
            selected_extended_actions,
            selected_priorities,
            td_errors,
            update_finite_log,
            trace_isolation_log,
            applied_log,
        )

        def backup_body(index: int, loop_carry: tuple[Any, ...]) -> tuple[Any, ...]:
            (
                learner_state,
                completion_log,
                semantics_log,
                prediction_log,
                target_log,
                residual_log,
                priority_log,
                valid_log,
                selected_option_log,
                selected_extended_log,
                selected_priority_log,
                td_error_log,
                update_finite,
                trace_isolation,
                applied,
            ) = loop_carry
            (
                completion_supported,
                candidate_semantics_valid,
                predictions_finite,
                candidate_targets,
                candidate_residuals,
                candidate_valid,
                candidate_priorities,
            ) = self._evaluate_candidates(
                learner_state,
                state,
                safe_observation,
                planner_inputs_valid,
            )
            ranking_scores = jnp.where(
                candidate_valid,
                candidate_priorities,
                -jnp.inf,
            )
            any_valid = jnp.any(candidate_valid)
            selected_option = jnp.argmax(ranking_scores).astype(jnp.int32)
            selected_option = jnp.where(
                any_valid,
                selected_option,
                jnp.int32(-1),
            )
            safe_option = jnp.maximum(selected_option, jnp.int32(0))
            selected_extended = safe_option + jnp.asarray(
                self._agent.config.n_primitive_actions,
                dtype=jnp.int32,
            )
            selected_target = candidate_targets[safe_option]
            selected_priority = candidate_priorities[safe_option]

            # Planning uses one-step synthetic traces.  The real traces are
            # restored after the parameter/optimizer proposal, preventing an
            # imagined backup from leaking eligibility into the next real
            # transition.
            trace_neutral_state = cast(
                MultiHeadMLPState,
                learner_state.replace(
                    trunk_traces=tuple(
                        jnp.zeros_like(trace) for trace in learner_state.trunk_traces
                    ),
                    head_traces=tuple(
                        (
                            jnp.zeros_like(weight_trace),
                            jnp.zeros_like(bias_trace),
                        )
                        for weight_trace, bias_trace in learner_state.head_traces
                    ),
                ),
            )
            targets = (
                jnp.full(
                    (self._agent.config.n_total_actions,),
                    jnp.nan,
                    dtype=jnp.float32,
                )
                .at[selected_extended]
                .set(selected_target)
            )

            def propose_update(_: None) -> tuple[MultiHeadMLPState, Array]:
                update = self._agent.base_learner.update(
                    trace_neutral_state,
                    safe_observation,
                    targets,
                )
                restored = cast(
                    MultiHeadMLPState,
                    update.state.replace(
                        trunk_traces=learner_state.trunk_traces,
                        head_traces=learner_state.head_traces,
                        normalizer_state=learner_state.normalizer_state,
                    ),
                )
                return restored, update.errors[selected_extended]

            def skip_update(_: None) -> tuple[MultiHeadMLPState, Array]:
                return learner_state, jnp.float32(0.0)

            candidate_state, candidate_td_error = jax.lax.cond(
                any_valid,
                propose_update,
                skip_update,
                operand=None,
            )
            candidate_finite = (
                any_valid
                & _floating_tree_is_finite(candidate_state)
                & jnp.isfinite(candidate_td_error)
            )
            traces_preserved = (
                any_valid
                & _trees_exactly_equal(
                    candidate_state.trunk_traces,
                    learner_state.trunk_traces,
                )
                & _trees_exactly_equal(
                    candidate_state.head_traces,
                    learner_state.head_traces,
                )
            )
            accepted = candidate_finite & traces_preserved
            next_learner_state = jax.lax.cond(
                accepted,
                lambda _: candidate_state,
                lambda _: learner_state,
                operand=None,
            )

            return (
                next_learner_state,
                completion_log.at[index].set(completion_supported),
                semantics_log.at[index].set(candidate_semantics_valid),
                prediction_log.at[index].set(predictions_finite),
                target_log.at[index].set(candidate_targets),
                residual_log.at[index].set(candidate_residuals),
                priority_log.at[index].set(candidate_priorities),
                valid_log.at[index].set(candidate_valid),
                selected_option_log.at[index].set(selected_option),
                selected_extended_log.at[index].set(
                    jnp.where(accepted, selected_extended, jnp.int32(-1))
                ),
                selected_priority_log.at[index].set(
                    jnp.where(any_valid, selected_priority, jnp.float32(0.0))
                ),
                td_error_log.at[index].set(
                    jnp.where(accepted, candidate_td_error, jnp.float32(0.0))
                ),
                update_finite.at[index].set(candidate_finite),
                trace_isolation.at[index].set(traces_preserved),
                applied.at[index].set(accepted),
            )

        (
            final_learner_state,
            completion_supported_log,
            semantics_valid_log,
            predictions_finite_log,
            targets_log,
            residuals_log,
            priorities_log,
            candidate_valid_log,
            selected_options,
            selected_extended_actions,
            selected_priorities,
            td_errors,
            update_finite_log,
            trace_isolation_log,
            applied_log,
        ) = jax.lax.fori_loop(0, budget, backup_body, carry)

        final_state = cast(
            STOMPState,
            state.replace(base_learner_state=final_learner_state),
        )
        diagnostics = OptionSearchControlDiagnostics(
            decision_observation=jnp.where(
                observation_values_finite,
                observation,
                0.0,
            ),
            decision_observation_static_contract_valid=jnp.asarray(
                observation_static_valid, dtype=jnp.bool_
            ),
            decision_observation_values_finite=observation_values_finite,
            option_model_static_contract_valid=jnp.asarray(model_static_valid, dtype=jnp.bool_),
            base_state_static_contract_valid=jnp.asarray(base_static_valid, dtype=jnp.bool_),
            base_state_values_finite=base_state_values_finite,
            option_model_values_finite=option_model_values_finite,
            state_counters_valid=state_counters_valid,
            base_update_capacity_available=base_update_capacity_available,
            stomp_state_static_contract_valid=(stomp_audit.state_static_contract_valid),
            stomp_state_values_finite=stomp_audit.state_values_finite,
            stomp_rng_key_valid=stomp_audit.rng_key_valid,
            stomp_action_ownership_valid=stomp_audit.ownership_valid,
            stomp_state_valid=stomp_audit.state_valid,
            decision_observation_matches_state=(stomp_audit.observation_matches),
            cached_decision_action_refreshed=jnp.asarray(False, dtype=jnp.bool_),
            value_effect_deferred_to_next_extended_action_selection=(jnp.any(applied_log)),
            average_reward_valid=average_reward_valid,
            planner_inputs_valid=planner_inputs_valid,
            completion_counts=jnp.where(
                model_static_valid,
                state.option_models.n_completions,
                jnp.zeros((n_options,), dtype=jnp.int32),
            ),
            completion_supported=completion_supported_log,
            candidate_semantics_valid=semantics_valid_log,
            candidate_predictions_finite=predictions_finite_log,
            candidate_targets=targets_log,
            candidate_bellman_residuals=residuals_log,
            candidate_priorities=priorities_log,
            candidate_valid=candidate_valid_log,
            selected_option_indices=selected_options,
            selected_extended_action_indices=selected_extended_actions,
            selected_priorities=selected_priorities,
            td_errors=td_errors,
            candidate_update_finite=update_finite_log,
            trace_isolation_preserved=trace_isolation_log,
            applied=applied_log,
            applied_count=jnp.sum(applied_log.astype(jnp.int32)),
        )
        return OptionSearchControlResult(
            state=final_state,
            diagnostics=diagnostics,
        )


__all__ = [
    "OPTION_SEARCH_CONTROL_CONFIG_SCHEMA",
    "OPTION_SEARCH_CONTROL_MECHANISM_STATUS",
    "OPTION_SEARCH_CONTROL_SCIENTIFIC_PROMOTION_ALLOWED",
    "OptionSearchControl",
    "OptionSearchControlConfig",
    "OptionSearchControlDiagnostics",
    "OptionSearchControlResourceBudget",
    "OptionSearchControlResult",
]
