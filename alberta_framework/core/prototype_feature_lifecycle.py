# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Bounded L0 lifecycle for discovered pair features and linear OaK consumers.

This module composes two existing mechanisms without changing either one:

* :class:`FixedBudgetInteractionLearner` proposes a fixed-width bank of
  pair-product features; and
* :class:`FeatureBankRouter` moves downstream linear state by descriptor
  identity when that bank changes.

The lifecycle owns only discovery and routing metadata.  A supplied
:class:`OaKState` remains caller-owned.  At a caller-declared safe boundary,
one descriptor change is routed atomically through the linear base heads,
eligibility traces, intra-option policies, option-start cache, and both axes
of every option-model matrix.  Unsafe changes are deferred: ordinary causal
feature learning is retained, while the proposed descriptor mutation is
rolled back to ``InteractionFeatureUpdateResult.pre_curation_state``.  A
deferred mutation is not queued; the learner may propose again only at a
later curation opportunity under its fixed replacement schedule.

This is development mechanism evidence only.  It does not establish a
benefit, an Alberta Plan completion claim, or eligibility for scientific
promotion.
"""

from __future__ import annotations

import dataclasses
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.feature_bank_router import (
    FeatureBankRouteDiagnostics,
    FeatureBankRouter,
    FeatureBankRouterConfig,
    FeatureBankRouterState,
)
from alberta_framework.core.feature_discovery import GENERATOR_RANDOM
from alberta_framework.core.interaction_features import (
    FixedBudgetInteractionLearner,
    InteractionFeatureState,
)
from alberta_framework.core.oak import OaKAgent, OaKConfig, OaKState
from alberta_framework.core.options import (
    STOMPConfig,
    SubtaskSpec,
    replace_dispatched_primitive_action,
)

PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA = (
    "alberta.prototype-feature-lifecycle.config.v1"
)
PROTOTYPE_FEATURE_LIFECYCLE_CHECKPOINT_SCHEMA = (
    "alberta.prototype-feature-lifecycle.checkpoint.v1"
)
PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS = "development_mechanism_only"
PROTOTYPE_FEATURE_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED = False

_CONFIG_TYPE = "PrototypeFeatureLifecycleConfig"
_INT32_MAX = 2_147_483_647
_MAX_TOTAL_FEATURE_DIM = 4_096
_MAX_PAIR_SLOTS = 262_144
_MAX_AXIS_PRODUCT_SCALARS = 4_194_304
_MAX_MANAGED_CONSUMER_SCALARS = 8_388_608
_MAX_DESCRIPTOR_COMPARISON_CELLS = 4_194_304
_MAX_ENUMERATED_PAIR_SPACE = 65_536
_MAX_PYTHON_COLLECTION_LENGTH = 4_096


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int = _INT32_MAX,
) -> int:
    """Validate a Python integer without accepting booleans or coercions."""

    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be a strict integer in [{minimum}, {maximum}]"
        )
    return value


def _strict_float(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float,
    maximum_inclusive: bool = True,
) -> float:
    """Validate an exact Python float and a finite closed/half-open range."""

    if type(value) is not float:
        bracket = "]" if maximum_inclusive else ")"
        raise ValueError(
            f"{name} must be a strict float in [{minimum}, {maximum}{bracket}"
        )
    upper_valid = value <= maximum if maximum_inclusive else value < maximum
    if not math.isfinite(value) or value < minimum or not upper_valid:
        bracket = "]" if maximum_inclusive else ")"
        raise ValueError(
            f"{name} must be a strict float in [{minimum}, {maximum}{bracket}"
        )
    try:
        float32_value = struct.unpack("!f", struct.pack("!f", value))[0]
    except OverflowError as error:
        raise ValueError(f"{name} must remain finite as float32") from error
    if not math.isfinite(float32_value) or (value != 0.0 and float32_value == 0.0):
        raise ValueError(f"{name} must remain finite and nonzero as float32")
    float32_upper_valid = (
        float32_value <= maximum
        if maximum_inclusive
        else float32_value < maximum
    )
    if float32_value < minimum or not float32_upper_valid:
        bracket = "]" if maximum_inclusive else ")"
        raise ValueError(
            f"{name} must remain in [{minimum}, {maximum}{bracket} as float32"
        )
    return value


@dataclasses.dataclass(frozen=True)
class PrototypeFeatureLifecycleConfig:
    """Static capacity, learner, and downstream-layout contract.

    ``option_subtask_feature_indices`` is a caller attestation.  OaKState does
    not persist SubtaskSpecs, so this standalone boundary cannot derive their
    indices from a supplied state.  Every attested index is restricted to the
    stable base prefix; an integrator must call
    :meth:`PrototypeFeatureLifecycle.require_compatible_oak_config` when it
    binds this lifecycle to the actual OaK configuration.
    """

    base_feature_dim: int
    active_pair_slots: int
    candidate_pair_slots: int
    n_tasks: int
    n_options: int
    n_primitive_actions: int
    option_subtask_feature_indices: tuple[int, ...]
    step_size_output: float = 0.03
    utility_decay: float = 0.995
    replacement_interval: int = 100
    min_feature_age: int = 50
    candidate_min_age: int = 25
    promotion_margin: float = 1.05
    scale_normalizer_decay: float = 0.99
    scale_normalizer_epsilon: float = 1.0e-6
    carry_survivors: bool = True
    max_observations: int = _INT32_MAX - 1

    def __post_init__(self) -> None:
        _strict_int(
            self.base_feature_dim,
            name="base_feature_dim",
            minimum=2,
            maximum=_MAX_TOTAL_FEATURE_DIM,
        )
        _strict_int(
            self.active_pair_slots,
            name="active_pair_slots",
            minimum=1,
            maximum=_MAX_PAIR_SLOTS,
        )
        _strict_int(
            self.candidate_pair_slots,
            name="candidate_pair_slots",
            minimum=0,
            maximum=_MAX_PAIR_SLOTS,
        )
        _strict_int(
            self.n_tasks,
            name="n_tasks",
            minimum=1,
            maximum=_MAX_PYTHON_COLLECTION_LENGTH,
        )
        _strict_int(
            self.n_options,
            name="n_options",
            minimum=1,
            maximum=_MAX_PYTHON_COLLECTION_LENGTH,
        )
        _strict_int(
            self.n_primitive_actions,
            name="n_primitive_actions",
            minimum=1,
            maximum=_MAX_PYTHON_COLLECTION_LENGTH,
        )
        if (
            type(self.option_subtask_feature_indices) is not tuple
            or len(self.option_subtask_feature_indices) != self.n_options
        ):
            raise ValueError(
                "option_subtask_feature_indices must be an exact tuple with one "
                "entry per option"
            )
        for feature_index in self.option_subtask_feature_indices:
            if (
                type(feature_index) is not int
                or not 0 <= feature_index < self.base_feature_dim
            ):
                raise ValueError(
                    "every option subtask must index the stable base prefix"
                )
        _strict_int(
            self.replacement_interval,
            name="replacement_interval",
            minimum=0,
            maximum=_INT32_MAX - 1,
        )
        if self.candidate_pair_slots == 0 and self.replacement_interval > 0:
            raise ValueError(
                "positive replacement_interval requires candidate_pair_slots > 0"
            )
        _strict_int(
            self.min_feature_age,
            name="min_feature_age",
            minimum=0,
            maximum=_INT32_MAX - 1,
        )
        _strict_int(
            self.candidate_min_age,
            name="candidate_min_age",
            minimum=0,
            maximum=_INT32_MAX - 1,
        )
        _strict_int(
            self.max_observations,
            name="max_observations",
            minimum=1,
            maximum=_INT32_MAX - 1,
        )
        _strict_float(
            self.step_size_output,
            name="step_size_output",
            minimum=0.0,
            maximum=float("inf"),
        )
        if self.step_size_output == 0.0:
            raise ValueError("step_size_output must be positive")
        _strict_float(
            self.utility_decay,
            name="utility_decay",
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )
        _strict_float(
            self.promotion_margin,
            name="promotion_margin",
            minimum=0.0,
            maximum=float("inf"),
        )
        if self.promotion_margin == 0.0:
            raise ValueError("promotion_margin must be positive")
        _strict_float(
            self.scale_normalizer_decay,
            name="scale_normalizer_decay",
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )
        _strict_float(
            self.scale_normalizer_epsilon,
            name="scale_normalizer_epsilon",
            minimum=0.0,
            maximum=float("inf"),
        )
        if self.scale_normalizer_epsilon == 0.0:
            raise ValueError("scale_normalizer_epsilon must be positive")
        if type(self.carry_survivors) is not bool:
            raise ValueError("carry_survivors must be a strict boolean")

        pair_space = self.base_feature_dim * (self.base_feature_dim - 1) // 2
        if self.active_pair_slots > pair_space:
            raise ValueError("active_pair_slots exceeds the canonical pair space")
        if self.candidate_pair_slots > pair_space:
            raise ValueError("candidate_pair_slots exceeds the canonical pair space")
        if self.active_pair_slots**2 > _MAX_DESCRIPTOR_COMPARISON_CELLS:
            raise ValueError(
                "active descriptor comparison matrix exceeds the allocation ceiling"
            )
        if self.candidate_pair_slots**2 > _MAX_DESCRIPTOR_COMPARISON_CELLS:
            raise ValueError(
                "candidate descriptor comparison matrix exceeds the allocation ceiling"
            )
        if (
            self.candidate_pair_slots > 0
            and pair_space > _MAX_ENUMERATED_PAIR_SPACE
        ):
            raise ValueError(
                "all-pairs candidate enumeration exceeds the allocation ceiling"
            )
        if self.n_total_actions > _MAX_PYTHON_COLLECTION_LENGTH:
            raise ValueError(
                "linear base-head collection exceeds the allocation ceiling"
            )
        if self.total_feature_dim > _MAX_TOTAL_FEATURE_DIM:
            raise ValueError(
                "total_feature_dim exceeds the lifecycle allocation ceiling"
            )
        discovery_axis_product = self.n_tasks * (
            self.active_pair_slots + self.candidate_pair_slots
        )
        if discovery_axis_product > _MAX_AXIS_PRODUCT_SCALARS:
            raise ValueError(
                "task-by-pair discovery state exceeds the allocation ceiling"
            )
        option_model_scalars = (
            self.n_options * self.total_feature_dim * self.total_feature_dim
        )
        if option_model_scalars > _MAX_AXIS_PRODUCT_SCALARS:
            raise ValueError(
                "option-model feature matrix state exceeds the allocation ceiling"
            )
        input_groups = (
            2 * self.n_total_actions
            + 2 * self.n_options * self.n_primitive_actions
            + self.n_options * self.total_feature_dim
            + 1
        )
        if input_groups * self.total_feature_dim > _MAX_MANAGED_CONSUMER_SCALARS:
            raise ValueError(
                "managed linear OaK consumers exceed the allocation ceiling"
            )

    @property
    def total_feature_dim(self) -> int:
        """Width of ``[base prefix | discovered pair tail]``."""

        return self.base_feature_dim + self.active_pair_slots

    @property
    def n_total_actions(self) -> int:
        """Number of primitive plus option heads in the linear base learner."""

        return self.n_primitive_actions + self.n_options

    def to_config(self) -> dict[str, object]:
        """Return the exact JSON-compatible L0 configuration."""

        return {
            "schema": PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA,
            "type": _CONFIG_TYPE,
            "mechanism_status": PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS,
            "scientific_promotion_allowed": (
                PROTOTYPE_FEATURE_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED
            ),
            "base_feature_dim": self.base_feature_dim,
            "active_pair_slots": self.active_pair_slots,
            "candidate_pair_slots": self.candidate_pair_slots,
            "n_tasks": self.n_tasks,
            "n_options": self.n_options,
            "n_primitive_actions": self.n_primitive_actions,
            "option_subtask_feature_indices": list(
                self.option_subtask_feature_indices
            ),
            "step_size_output": self.step_size_output,
            "utility_decay": self.utility_decay,
            "replacement_interval": self.replacement_interval,
            "min_feature_age": self.min_feature_age,
            "candidate_min_age": self.candidate_min_age,
            "promotion_margin": self.promotion_margin,
            "scale_normalizer_decay": self.scale_normalizer_decay,
            "scale_normalizer_epsilon": self.scale_normalizer_epsilon,
            "carry_survivors": self.carry_survivors,
            "max_observations": self.max_observations,
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> PrototypeFeatureLifecycleConfig:
        """Strictly reconstruct only the exact versioned mechanism schema."""

        payload = dict(config)
        expected = {
            "schema",
            "type",
            "mechanism_status",
            "scientific_promotion_allowed",
            "base_feature_dim",
            "active_pair_slots",
            "candidate_pair_slots",
            "n_tasks",
            "n_options",
            "n_primitive_actions",
            "option_subtask_feature_indices",
            "step_size_output",
            "utility_decay",
            "replacement_interval",
            "min_feature_age",
            "candidate_min_age",
            "promotion_margin",
            "scale_normalizer_decay",
            "scale_normalizer_epsilon",
            "carry_survivors",
            "max_observations",
        }
        if set(payload) != expected:
            raise ValueError("prototype feature lifecycle config fields do not match v1")
        if payload.pop("schema") != PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA:
            raise ValueError("unexpected prototype feature lifecycle config schema")
        if payload.pop("type") != _CONFIG_TYPE:
            raise ValueError("unexpected prototype feature lifecycle config type")
        if (
            payload.pop("mechanism_status")
            != PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS
        ):
            raise ValueError("prototype feature lifecycle must remain mechanism-only")
        if payload.pop("scientific_promotion_allowed") is not False:
            raise ValueError("prototype feature lifecycle config cannot claim promotion")
        raw_subtask_indices = payload.get("option_subtask_feature_indices")
        if type(raw_subtask_indices) is not list or not all(
            type(index) is int for index in raw_subtask_indices
        ):
            raise ValueError(
                "serialized option_subtask_feature_indices must be a JSON integer list"
            )
        payload["option_subtask_feature_indices"] = tuple(raw_subtask_indices)
        return PrototypeFeatureLifecycleConfig(**cast(dict[str, Any], payload))


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleState:
    """Owned discovery state, descriptor version, and bounded counters."""

    learner_state: InteractionFeatureState
    router_state: FeatureBankRouterState
    observe_count: Int[Array, ""]
    deferred_curation_count: Int[Array, ""]
    committed_curation_count: Int[Array, ""]
    rolled_back_curation_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFeatureConsumerBinding:
    """Exact feature-bank identity carried atomically with one OaK consumer.

    This is caller-owned metadata. A standalone caller must persist and
    restore it together with the OaK state; rebuilding it from a separately
    restored lifecycle state would be an attestation, not proof that the OaK
    weights share that feature-bank identity.
    """

    semantic_generation: Int[Array, ""]
    descriptors: Int[Array, " active_pair_slots 2"]


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleEvent:
    """One causal discovery observation and a caller-owned commit boundary.

    ``observation`` and ``targets`` are supplied directly to the interaction
    learner.  NaN targets are its documented missing-head sentinel.
    ``next_observation`` must correspond to the supplied OaK state's cached
    ``base_last_obs`` under the current descriptor bank.  The cache is rebuilt
    under a newly committed bank, preventing mixed-generation decisions.
    """

    observation: Float[Array, " base_feature_dim"]
    targets: Float[Array, " n_tasks"]
    next_observation: Float[Array, " base_feature_dim"]
    allow_curation: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class PrototypeFeatureLifecycleResourceBudget:
    """Exact persistent array bytes and static logical work bounds."""

    mechanism_status: str
    scientific_promotion_allowed: bool
    base_feature_slots: int
    active_pair_slots: int
    candidate_pair_slots: int
    managed_oak_feature_width: int
    learner_persistent_state_nbytes: int
    router_persistent_state_nbytes: int
    lifecycle_counter_nbytes: int
    lifecycle_state_nbytes: int
    consumer_binding_persistent_nbytes: int
    internal_learner_template_nbytes: int
    internal_oak_template_nbytes: int
    internal_template_nbytes: int
    owned_persistent_state_nbytes: int
    managed_oak_consumer_nbytes: int
    rebuilt_base_cache_nbytes: int
    input_route_feature_groups: int
    output_route_feature_groups: int
    router_calls_per_observe: int
    router_calls_per_committed_curation: int
    max_active_pair_products_per_observe: int
    max_candidate_pair_products_per_observe: int
    max_observations: int

    def to_config(self) -> dict[str, str | int | bool]:
        """Return an exact JSON-compatible resource record."""

        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class PrototypePairGradientPullback:
    """Base-coordinate pullback of a gradient over the augmented vector."""

    gradient: Float[Array, " base_feature_dim"]
    valid: Bool[Array, ""]
    semantic_generation: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleDiagnostics:
    """Fixed-shape audit for one observe/defer/route transaction."""

    available: Bool[Array, ""]
    state_values_valid: Bool[Array, ""]
    oak_values_valid: Bool[Array, ""]
    consumer_binding_valid: Bool[Array, ""]
    event_values_valid: Bool[Array, ""]
    next_observation_matches_oak_cache: Bool[Array, ""]
    update_capacity_available: Bool[Array, ""]
    learner_update_rejected: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    curation_proposed: Bool[Array, ""]
    safe_curation_boundary: Bool[Array, ""]
    curation_deferred: Bool[Array, ""]
    routing_attempted: Bool[Array, ""]
    input_route_valid: Bool[Array, ""]
    output_route_valid: Bool[Array, ""]
    route_states_match: Bool[Array, ""]
    routed_values_finite: Bool[Array, ""]
    curation_committed: Bool[Array, ""]
    curation_rolled_back: Bool[Array, ""]
    postcondition_checked: Bool[Array, ""]
    postcondition_valid: Bool[Array, ""]
    postcondition_rolled_back: Bool[Array, ""]
    semantic_generation_before: Int[Array, ""]
    semantic_generation_after: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleResult:
    """Owned state, caller-owned OaK state, and complete route audit."""

    state: PrototypeFeatureLifecycleState
    oak_state: OaKState
    consumer_binding: PrototypeFeatureConsumerBinding
    next_augmented_observation: Float[Array, " total_feature_dim"]
    predictions: Float[Array, " n_tasks"]
    errors: Float[Array, " n_tasks"]
    metrics: Float[Array, " 7"]
    input_route_diagnostics: FeatureBankRouteDiagnostics
    output_route_diagnostics: FeatureBankRouteDiagnostics
    diagnostics: PrototypeFeatureLifecycleDiagnostics


def _array_has_contract(value: Any, shape: tuple[int, ...], dtype: Any) -> bool:
    """Return an exact noncoercing array shape/dtype predicate."""

    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and value.shape == shape
        and value.dtype == dtype
    )


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    """Reject shape and effective-dtype mismatches before indexed work."""

    if not hasattr(value, "shape") or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not hasattr(value, "dtype") or value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    return cast(Array, value)


def _static_tree_matches(value: Any, template: Any) -> bool:
    """Compare exact PyTree nodes and leaf contracts, tolerating host timers."""

    try:
        value_leaves, value_tree = jax.tree_util.tree_flatten(value)
        template_leaves, template_tree = jax.tree_util.tree_flatten(template)
    except (AttributeError, TypeError, ValueError):
        return False
    if (
        value_tree != template_tree  # type: ignore[operator]
        or len(value_leaves) != len(template_leaves)
    ):
        return False
    for actual, expected in zip(value_leaves, template_leaves, strict=True):
        if type(expected) is float:
            if type(actual) is float:
                continue
            if not (
                hasattr(actual, "shape")
                and hasattr(actual, "dtype")
                and actual.shape == ()
                and actual.dtype == jnp.float32
            ):
                return False
            continue
        if not (
            hasattr(actual, "shape")
            and hasattr(actual, "dtype")
            and hasattr(expected, "shape")
            and hasattr(expected, "dtype")
            and actual.shape == expected.shape
            and actual.dtype == expected.dtype
        ):
            return False
    return True


def _floating_tree_is_finite(value: Any) -> Bool[Array, ""]:
    """Return whether all inexact leaves, including host timers, are finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree_util.tree_leaves(value):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(array))
    return valid


def _tree_nbytes(value: Any) -> int:
    """Count physical bytes reported by every persistent PyTree leaf."""

    return sum(
        int(getattr(leaf, "nbytes", 0))
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _trees_exactly_equal(left: Any, right: Any) -> Bool[Array, ""]:
    """Return one JAX boolean for exact equality of matching array PyTrees."""

    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if (
        left_tree != right_tree  # type: ignore[operator]
        or len(left_leaves) != len(right_leaves)
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        valid = valid & jnp.array_equal(jnp.asarray(left_leaf), jnp.asarray(right_leaf))
    return valid


def _exact_json_tree_equal(left: Any, right: Any) -> bool:
    """Compare JSON-like trees without Python's bool/int or int/float aliases."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _exact_json_tree_equal(left[key], right[key])
            for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_json_tree_equal(left_value, right_value)
            for left_value, right_value in zip(
                left,
                right,
                strict=True,
            )
        )
    if type(left) is float:
        return struct.pack("!d", left) == struct.pack("!d", right)
    return bool(left == right)


def _float32_arrays_bit_exact(left: Array, right: Array) -> Bool[Array, ""]:
    """Compare float32 arrays by representation, distinguishing signed zero."""

    return jnp.array_equal(
        jax.lax.bitcast_convert_type(left, jnp.int32),
        jax.lax.bitcast_convert_type(right, jnp.int32),
    )


def _nonnegative_int32_sum_within(
    values: Array,
    limit: Array,
) -> Bool[Array, ""]:
    """Check a sum against a limit without allowing int32 wraparound."""

    def step(carry: tuple[Array, Array], value: Array) -> tuple[tuple[Array, Array], None]:
        total, valid = carry
        fits = (value >= 0) & (value <= limit - total)
        safe_value = jnp.where(fits, value, jnp.asarray(0, dtype=jnp.int32))
        return (total + safe_value, valid & fits), None

    (_, valid), _ = jax.lax.scan(
        step,
        (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(True, dtype=jnp.bool_)),
        values,
    )
    return valid


class PrototypeFeatureLifecycle:
    """Standalone fixed-budget feature discovery and route transaction."""

    def __init__(self, config: PrototypeFeatureLifecycleConfig):
        if type(config) is not PrototypeFeatureLifecycleConfig:
            raise TypeError("config must be a PrototypeFeatureLifecycleConfig")
        self._config = config
        self._learner = FixedBudgetInteractionLearner(
            n_features=config.active_pair_slots,
            n_tasks=config.n_tasks,
            step_size_output=config.step_size_output,
            utility_decay=config.utility_decay,
            replacement_interval=config.replacement_interval,
            min_feature_age=config.min_feature_age,
            candidate_count=config.candidate_pair_slots,
            candidate_min_age=config.candidate_min_age,
            promotion_margin=config.promotion_margin,
            promotion_blend=1.0,
            generator_mix=(1.0, 0.0, 0.0),
            candidate_strategy="all_pairs",
            utility_aggregation="mean",
            utility_task_balancing="active",
            task_activity_decay=config.utility_decay,
            future_utility_mix=0.0,
            refresh_candidates=False,
            refresh_promoted_candidate=False,
            include_squares=False,
            use_obgd=False,
            scale_robust=True,
            scale_normalizer_decay=config.scale_normalizer_decay,
            scale_normalizer_epsilon=config.scale_normalizer_epsilon,
        )
        self._router = FeatureBankRouter(
            FeatureBankRouterConfig(
                base_dim=config.base_feature_dim,
                active_slots=config.active_pair_slots,
            )
        )
        self._learner_template = self._initial_learner_state(jr.key(0))
        self._oak_template = self._make_oak_template()

    @property
    def config(self) -> PrototypeFeatureLifecycleConfig:
        """Return the immutable lifecycle configuration."""

        return self._config

    @property
    def learner(self) -> FixedBudgetInteractionLearner:
        """Return the unchanged fixed-budget interaction learner."""

        return self._learner

    @property
    def router(self) -> FeatureBankRouter:
        """Return the unchanged descriptor-identity router."""

        return self._router

    def to_config(self) -> dict[str, object]:
        """Serialize the exact lifecycle configuration."""

        return self._config.to_config()

    def require_compatible_oak_config(self, oak_config: OaKConfig) -> None:
        """Validate the caller attestation against an actual OaK config.

        This bind-time check cannot be recovered from ``OaKState`` later,
        because the state intentionally contains no SubtaskSpecs.
        """

        if type(oak_config) is not OaKConfig:
            raise TypeError("oak_config must be an OaKConfig")
        stomp = oak_config.stomp
        if type(stomp) is not STOMPConfig:
            raise TypeError("oak_config.stomp must be an exact STOMPConfig")
        specs = stomp.subtask_specs
        exact_specs = type(specs) is tuple and all(
            type(spec) is SubtaskSpec and type(spec.feature_index) is int
            for spec in specs
        )
        indices = tuple(spec.feature_index for spec in specs)
        compatible = (
            exact_specs
            and type(stomp.observation_dim) is int
            and type(stomp.n_primitive_actions) is int
            and type(stomp.base_hidden_sizes) is tuple
            and stomp.observation_dim == self._config.total_feature_dim
            and stomp.n_primitive_actions == self._config.n_primitive_actions
            and stomp.n_options == self._config.n_options
            and stomp.base_hidden_sizes == ()
            and indices == self._config.option_subtask_feature_indices
        )
        if not compatible:
            raise ValueError(
                "actual OaK config does not match the linear layout and subtask attestation"
            )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> PrototypeFeatureLifecycle:
        """Construct from the strict versioned configuration."""

        return PrototypeFeatureLifecycle(
            PrototypeFeatureLifecycleConfig.from_config(config)
        )

    def _canonical_active_descriptors(self) -> Array:
        pairs: list[tuple[int, int]] = []
        for left in range(self._config.base_feature_dim):
            for right in range(left + 1, self._config.base_feature_dim):
                pairs.append((left, right))
                if len(pairs) == self._config.active_pair_slots:
                    break
            if len(pairs) == self._config.active_pair_slots:
                break
        return jnp.asarray(
            pairs,
            dtype=jnp.int32,
        )

    def _initial_learner_state(self, key: Array) -> InteractionFeatureState:
        state = self._learner.init(
            feature_dim=self._config.base_feature_dim,
            key=key,
        )
        descriptors = self._canonical_active_descriptors()
        return cast(
            InteractionFeatureState,
            state.replace(
                feature_left=descriptors[:, 0],
                feature_right=descriptors[:, 1],
            ),
        )

    def _make_oak_template(self) -> OaKState:
        specs = tuple(
            SubtaskSpec(feature_index=feature_index)
            for feature_index in self._config.option_subtask_feature_indices
        )
        agent = OaKAgent(
            OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=specs,
                    observation_dim=self._config.total_feature_dim,
                    n_primitive_actions=self._config.n_primitive_actions,
                    base_hidden_sizes=(),
                )
            )
        )
        return agent.init(jr.key(0))

    def init(self, key: Array) -> PrototypeFeatureLifecycleState:
        """Initialize a unique canonical bank and zero lifecycle counters."""

        if not (
            hasattr(key, "shape")
            and hasattr(key, "dtype")
            and key.shape == ()
            and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
        ):
            raise TypeError("key must be a scalar typed JAX PRNG key")
        learner_state = self._initial_learner_state(key)
        descriptors = jnp.stack(
            (learner_state.feature_left, learner_state.feature_right),
            axis=1,
        )
        zero = jnp.asarray(0, dtype=jnp.int32)
        return PrototypeFeatureLifecycleState(
            learner_state=learner_state,
            router_state=self._router.init(descriptors),
            observe_count=zero,
            deferred_curation_count=zero,
            committed_curation_count=zero,
            rolled_back_curation_count=zero,
        )

    def init_bound(
        self,
        key: Array,
    ) -> tuple[PrototypeFeatureLifecycleState, PrototypeFeatureConsumerBinding]:
        """Initialize lifecycle state and its inseparable consumer identity."""

        state = self.init(key)
        return state, PrototypeFeatureConsumerBinding(
            semantic_generation=state.router_state.generation_count,
            descriptors=state.router_state.descriptors,
        )

    def _consumer_binding_static_contract_valid(self, binding: Any) -> bool:
        return (
            type(binding) is PrototypeFeatureConsumerBinding
            and _array_has_contract(
                binding.semantic_generation,
                (),
                jnp.int32,
            )
            and _array_has_contract(
                binding.descriptors,
                (self._config.active_pair_slots, 2),
                jnp.int32,
            )
        )

    def consumer_binding_valid(
        self,
        state: PrototypeFeatureLifecycleState,
        binding: PrototypeFeatureConsumerBinding,
    ) -> Bool[Array, ""]:
        """Return whether a caller-owned OaK binding exactly matches ``state``."""

        if not self._state_static_contract_valid(state):
            return jnp.asarray(False, dtype=jnp.bool_)
        if not self._consumer_binding_static_contract_valid(binding):
            return jnp.asarray(False, dtype=jnp.bool_)
        return (
            (binding.semantic_generation >= 0)
            & (
                binding.semantic_generation
                == state.router_state.generation_count
            )
            & jnp.array_equal(
                binding.descriptors,
                state.router_state.descriptors,
            )
        )

    def _state_static_contract_valid(self, state: Any) -> bool:
        if type(state) is not PrototypeFeatureLifecycleState:
            return False
        template = PrototypeFeatureLifecycleState(
            learner_state=self._learner_template,
            router_state=self._router.init(self._canonical_active_descriptors()),
            observe_count=jnp.asarray(0, dtype=jnp.int32),
            deferred_curation_count=jnp.asarray(0, dtype=jnp.int32),
            committed_curation_count=jnp.asarray(0, dtype=jnp.int32),
            rolled_back_curation_count=jnp.asarray(0, dtype=jnp.int32),
        )
        return _static_tree_matches(state, template)

    def _oak_static_contract_valid(self, state: Any) -> bool:
        return type(state) is OaKState and _static_tree_matches(
            state,
            self._oak_template,
        )

    def state_valid(
        self,
        state: PrototypeFeatureLifecycleState,
    ) -> Bool[Array, ""]:
        """Validate exact structure, descriptor identity, values, and counters."""

        if not self._state_static_contract_valid(state):
            return jnp.asarray(False, dtype=jnp.bool_)
        learner = state.learner_state
        active_descriptors = jnp.stack(
            (learner.feature_left, learner.feature_right),
            axis=1,
        )
        active_validation = self._router.validate_descriptors(active_descriptors)
        router_validation = self._router.validate_descriptors(
            state.router_state.descriptors
        )
        canonical_descriptors = self._canonical_active_descriptors()
        canonical_row_distance = jnp.sum(
            jnp.any(active_descriptors != canonical_descriptors, axis=1),
            dtype=jnp.int32,
        )
        descriptor_history_valid = (
            canonical_row_distance
            <= state.router_state.generation_count
        )

        candidate_left = learner.candidate_left
        candidate_right = learner.candidate_right
        candidate_live = (
            (candidate_left >= 0)
            & (candidate_left < candidate_right)
            & (candidate_right < self._config.base_feature_dim)
        )
        candidate_descriptors = jnp.stack(
            (candidate_left, candidate_right),
            axis=1,
        )
        candidate_equal = jnp.all(
            candidate_descriptors[:, None, :] == candidate_descriptors[None, :, :],
            axis=-1,
        )
        candidate_duplicate = jnp.any(
            candidate_equal
            & ~jnp.eye(
                self._config.candidate_pair_slots,
                dtype=jnp.bool_,
            )
        )

        nonnegative_counters = (
            (learner.step_count >= 0)
            & jnp.all(learner.ages >= 0)
            & jnp.all(learner.candidate_ages >= 0)
            & jnp.all(learner.evidence_idle_steps >= 0)
            & jnp.all(learner.utility_evidence_streak >= 0)
            & jnp.all(learner.candidate_promotion_evidence_streak >= 0)
            & (state.router_state.route_count >= 0)
            & (state.router_state.generation_count >= 0)
            & (state.observe_count >= 0)
            & (state.deferred_curation_count >= 0)
            & (state.committed_curation_count >= 0)
            & (state.rolled_back_curation_count >= 0)
        )
        learner_counter_progress_valid = (
            jnp.all(learner.ages <= learner.step_count)
            & jnp.all(learner.candidate_ages <= learner.step_count)
            & jnp.all(learner.evidence_idle_steps <= learner.step_count)
            & jnp.all(learner.utility_evidence_streak <= learner.step_count)
            & jnp.all(
                learner.candidate_promotion_evidence_streak
                <= learner.step_count
            )
        )
        bounded_counters = (
            (state.observe_count <= self._config.max_observations)
            & (learner.step_count == state.observe_count)
            & _nonnegative_int32_sum_within(
                jnp.stack(
                    (
                        state.deferred_curation_count,
                        state.committed_curation_count,
                        state.rolled_back_curation_count,
                    )
                ),
                state.observe_count,
            )
            & (state.router_state.route_count == state.committed_curation_count)
            & (
                state.router_state.generation_count
                == state.committed_curation_count
            )
        )
        provenance_valid = (
            jnp.all(learner.feature_parent_a == -1)
            & jnp.all(learner.feature_parent_b == -1)
            & jnp.all(learner.candidate_parent_a == -1)
            & jnp.all(learner.candidate_parent_b == -1)
            & jnp.all(learner.feature_generator == GENERATOR_RANDOM)
            & jnp.all(learner.candidate_generator == GENERATOR_RANDOM)
        )
        moments_valid = (
            jnp.all(learner.utilities >= 0.0)
            & jnp.all(learner.utilities <= 1.0)
            & jnp.all(learner.candidate_utilities >= 0.0)
            & jnp.all(learner.candidate_utilities <= 1.0)
            & jnp.all(learner.feature_second_moments >= 0.0)
            & jnp.all(learner.candidate_second_moments >= 0.0)
            & jnp.all(learner.target_second_moments >= 0.0)
            & jnp.all(learner.task_activity_ema >= 0.0)
            & jnp.all(learner.task_activity_ema <= 1.0)
        )
        timer_values_valid = (
            learner.birth_timestamp >= 0.0
        ) & (learner.uptime_s >= 0.0)
        fixed_disabled_substate_valid = (
            _float32_arrays_bit_exact(
                learner.relevance_probe_weights,
                jnp.zeros_like(learner.relevance_probe_weights),
            )
            & _float32_arrays_bit_exact(
                learner.relevance_probe_biases,
                learner.output_biases,
            )
            & jnp.all(learner.evidence_idle_steps == 0)
            & jnp.all(learner.utility_evidence_streak == 0)
            & jnp.all(learner.candidate_promotion_evidence_streak == 0)
            & ~jnp.any(learner.active_output_memory_committed)
            & ~jnp.any(learner.candidate_reacquisition_required)
        )
        return (
            active_validation.valid
            & jnp.all(active_validation.live_mask)
            & router_validation.valid
            & jnp.all(router_validation.live_mask)
            & jnp.all(candidate_live)
            & ~candidate_duplicate
            & jnp.array_equal(active_descriptors, state.router_state.descriptors)
            & descriptor_history_valid
            & _floating_tree_is_finite(learner)
            & nonnegative_counters
            & learner_counter_progress_valid
            & bounded_counters
            & provenance_valid
            & moments_valid
            & timer_values_valid
            & fixed_disabled_substate_valid
        )

    def _oak_values_valid(self, state: OaKState) -> Bool[Array, ""]:
        """Apply the public STOMP ownership audit plus outer OaK checks."""

        stomp = state.stomp_state
        dispatch = replace_dispatched_primitive_action(
            stomp,
            stomp.base_last_obs,
            stomp.last_primitive_action,
            jnp.ones(
                (self._config.n_primitive_actions,),
                dtype=jnp.bool_,
            ),
        ).decision
        execution_limit = jnp.where(
            state.step_count < _INT32_MAX,
            state.step_count + jnp.asarray(1, dtype=jnp.int32),
            state.step_count,
        )
        option_completions = stomp.option_models.n_completions
        timer_values_valid = (
            jnp.asarray(stomp.base_learner_state.birth_timestamp) >= 0.0
        ) & (jnp.asarray(stomp.base_learner_state.uptime_s) >= 0.0)
        return (
            dispatch.state_valid
            & dispatch.observation_matches
            & _floating_tree_is_finite(state)
            & jnp.all(state.execution_counts >= 0)
            & _nonnegative_int32_sum_within(
                state.execution_counts,
                execution_limit,
            )
            & jnp.all(option_completions <= state.execution_counts)
            & _nonnegative_int32_sum_within(
                option_completions,
                state.step_count,
            )
            & timer_values_valid
            & (state.step_count >= 0)
            & (state.step_count == stomp.step_count)
        )

    def augment(
        self,
        state: PrototypeFeatureLifecycleState,
        observation: Array,
    ) -> Float[Array, " total_feature_dim"]:
        """Return the fixed-width base prefix plus live pair products.

        Static violations raise before indexed work.  Dynamic invalidity is a
        finite all-zero fail-closed result so this method is safe inside JIT
        and scan composition.
        """

        if not self._state_static_contract_valid(state):
            raise ValueError("prototype feature lifecycle state has an invalid static contract")
        raw = _require_array(
            observation,
            name="observation",
            shape=(self._config.base_feature_dim,),
            dtype=jnp.float32,
        )
        augmented = self._learner.augmented_observation(state.learner_state, raw)
        valid = (
            self.state_valid(state)
            & jnp.all(jnp.isfinite(raw))
            & jnp.all(jnp.isfinite(augmented))
        )
        return jnp.where(valid, augmented, jnp.zeros_like(augmented))

    def pullback_pair_gradient(
        self,
        state: PrototypeFeatureLifecycleState,
        observation: Array,
        augmented_gradient: Array,
        expected_generation: Array,
        expected_bank_descriptors: Array,
    ) -> PrototypePairGradientPullback:
        """Apply the chain rule only for the exact generation and descriptor bank."""

        if not self._state_static_contract_valid(state):
            raise ValueError("prototype feature lifecycle state has an invalid static contract")
        raw = _require_array(
            observation,
            name="observation",
            shape=(self._config.base_feature_dim,),
            dtype=jnp.float32,
        )
        gradient = _require_array(
            augmented_gradient,
            name="augmented_gradient",
            shape=(self._config.total_feature_dim,),
            dtype=jnp.float32,
        )
        generation = _require_array(
            expected_generation,
            name="expected_generation",
            shape=(),
            dtype=jnp.int32,
        )
        expected_descriptors = _require_array(
            expected_bank_descriptors,
            name="expected_bank_descriptors",
            shape=(self._config.active_pair_slots, 2),
            dtype=jnp.int32,
        )
        left = state.learner_state.feature_left
        right = state.learner_state.feature_right
        live = (
            (left >= 0)
            & (left < right)
            & (right < self._config.base_feature_dim)
        )
        safe_left = jnp.where(live, left, 0)
        safe_right = jnp.where(live, right, 0)
        pair_gradient = gradient[self._config.base_feature_dim :]
        pulled = gradient[: self._config.base_feature_dim]
        pulled = pulled.at[safe_left].add(
            jnp.where(live, pair_gradient * raw[safe_right], 0.0)
        )
        pulled = pulled.at[safe_right].add(
            jnp.where(live, pair_gradient * raw[safe_left], 0.0)
        )
        valid = (
            self.state_valid(state)
            & jnp.all(jnp.isfinite(raw))
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(pulled))
            & (generation == state.router_state.generation_count)
            & jnp.array_equal(
                expected_descriptors,
                state.router_state.descriptors,
            )
        )
        return PrototypePairGradientPullback(
            gradient=jnp.where(valid, pulled, jnp.zeros_like(pulled)),
            valid=valid,
            semantic_generation=jnp.where(
                valid,
                state.router_state.generation_count,
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )

    def resource_budget(
        self,
        state: PrototypeFeatureLifecycleState | None = None,
    ) -> PrototypeFeatureLifecycleResourceBudget:
        """Return exact owned bytes and static consumer/work bounds."""

        measured = self.init(jr.key(0)) if state is None else state
        if not self._state_static_contract_valid(measured):
            raise ValueError("prototype feature lifecycle state has an invalid static contract")
        width = self._config.total_feature_dim
        heads = self._config.n_total_actions
        options = self._config.n_options
        primitive_actions = self._config.n_primitive_actions
        input_groups = (
            heads
            + heads
            + options * primitive_actions
            + options * primitive_actions
            + options * width
            + 1
        )

        output_groups = options * width
        managed_consumer_scalars = input_groups * width
        lifecycle_state_nbytes = _tree_nbytes(measured)
        consumer_binding_nbytes = _tree_nbytes(
            PrototypeFeatureConsumerBinding(
                semantic_generation=measured.router_state.generation_count,
                descriptors=measured.router_state.descriptors,
            )
        )
        internal_learner_template_nbytes = _tree_nbytes(self._learner_template)
        internal_oak_template_nbytes = _tree_nbytes(self._oak_template)
        internal_template_nbytes = (
            internal_learner_template_nbytes + internal_oak_template_nbytes
        )
        return PrototypeFeatureLifecycleResourceBudget(
            mechanism_status=PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS,
            scientific_promotion_allowed=(
                PROTOTYPE_FEATURE_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED
            ),
            base_feature_slots=self._config.base_feature_dim,
            active_pair_slots=self._config.active_pair_slots,
            candidate_pair_slots=self._config.candidate_pair_slots,
            managed_oak_feature_width=width,
            learner_persistent_state_nbytes=_tree_nbytes(measured.learner_state),
            router_persistent_state_nbytes=_tree_nbytes(measured.router_state),
            lifecycle_counter_nbytes=16,
            lifecycle_state_nbytes=lifecycle_state_nbytes,
            consumer_binding_persistent_nbytes=consumer_binding_nbytes,
            internal_learner_template_nbytes=internal_learner_template_nbytes,
            internal_oak_template_nbytes=internal_oak_template_nbytes,
            internal_template_nbytes=internal_template_nbytes,
            owned_persistent_state_nbytes=(
                lifecycle_state_nbytes + internal_template_nbytes
            ),
            managed_oak_consumer_nbytes=4 * (managed_consumer_scalars + width),
            rebuilt_base_cache_nbytes=4 * width,
            input_route_feature_groups=input_groups,
            output_route_feature_groups=output_groups,
            # Both pure route candidates execute on every observe so the
            # method remains one fixed JIT/scan program.  Their state is
            # adopted only at a committed curation boundary.
            router_calls_per_observe=2,
            router_calls_per_committed_curation=2,
            # Current implementation evaluates the active bank for the old
            # cache audit, learner update, provisional committed cache,
            # candidate postcondition cache, and final returned cache.
            max_active_pair_products_per_observe=(
                5 * self._config.active_pair_slots
            ),
            max_candidate_pair_products_per_observe=(
                self._config.candidate_pair_slots
            ),
            max_observations=self._config.max_observations,
        )

    def unavailable_diagnostics(
        self,
        semantic_generation: Array,
    ) -> PrototypeFeatureLifecycleDiagnostics:
        """Return finite neutral diagnostics for an outer rejected branch.

        This constructor performs no lifecycle, OaK, route, or postcondition
        audit.  It exists so an integrating JAX ``lax.cond`` can keep a fixed
        diagnostics PyTree when a prerequisite outside this boundary rejects
        the call.
        """

        generation = _require_array(
            semantic_generation,
            name="semantic_generation",
            shape=(),
            dtype=jnp.int32,
        )
        false = jnp.asarray(False, dtype=jnp.bool_)
        return PrototypeFeatureLifecycleDiagnostics(
            available=false,
            state_values_valid=false,
            oak_values_valid=false,
            consumer_binding_valid=false,
            event_values_valid=false,
            next_observation_matches_oak_cache=false,
            update_capacity_available=false,
            learner_update_rejected=false,
            transaction_applied=false,
            curation_proposed=false,
            safe_curation_boundary=false,
            curation_deferred=false,
            routing_attempted=false,
            input_route_valid=false,
            output_route_valid=false,
            route_states_match=false,
            routed_values_finite=false,
            curation_committed=false,
            curation_rolled_back=false,
            postcondition_checked=false,
            postcondition_valid=false,
            postcondition_rolled_back=false,
            semantic_generation_before=generation,
            semantic_generation_after=generation,
        )

    def _input_consumers(self, oak_state: OaKState) -> dict[str, Any]:
        stomp = oak_state.stomp_state
        return {
            "base_head_weights": stomp.base_learner_state.head_params.weights,
            "base_head_weight_traces": tuple(
                trace_pair[0]
                for trace_pair in stomp.base_learner_state.head_traces
            ),
            "option_policy_weights": stomp.option_policies.q_weights,
            "option_policy_traces": stomp.option_policies.traces,
            "option_model_input_weights": stomp.option_models.next_state_weights,
            "option_start_observation": stomp.option_start_obs,
        }

    def _routed_oak_state(
        self,
        old: OaKState,
        routed_inputs: dict[str, Any],
        routed_model: Array,
        next_augmented_observation: Array,
        commit: Array,
    ) -> OaKState:
        stomp = old.stomp_state
        learner = stomp.base_learner_state
        candidate_head_weights = cast(
            tuple[Array, ...],
            routed_inputs["base_head_weights"],
        )
        candidate_weight_traces = cast(
            tuple[Array, ...],
            routed_inputs["base_head_weight_traces"],
        )
        head_weights = tuple(
            jnp.where(commit, candidate, original)
            for candidate, original in zip(
                candidate_head_weights,
                learner.head_params.weights,
                strict=True,
            )
        )
        head_traces = tuple(
            (
                jnp.where(commit, candidate_weight, original_pair[0]),
                original_pair[1],
            )
            for candidate_weight, original_pair in zip(
                candidate_weight_traces,
                learner.head_traces,
                strict=True,
            )
        )
        next_learner = learner.replace(
            head_params=learner.head_params.replace(weights=head_weights),
            head_traces=head_traces,
        )
        policies = stomp.option_policies.replace(
            q_weights=jnp.where(
                commit,
                routed_inputs["option_policy_weights"],
                stomp.option_policies.q_weights,
            ),
            traces=jnp.where(
                commit,
                routed_inputs["option_policy_traces"],
                stomp.option_policies.traces,
            ),
        )
        models = stomp.option_models.replace(
            next_state_weights=jnp.where(
                commit,
                routed_model,
                stomp.option_models.next_state_weights,
            )
        )
        next_stomp = stomp.replace(
            base_learner_state=next_learner,
            base_last_obs=jnp.where(
                commit,
                next_augmented_observation,
                stomp.base_last_obs,
            ),
            option_policies=policies,
            option_models=models,
            option_start_obs=jnp.where(
                commit,
                routed_inputs["option_start_observation"],
                stomp.option_start_obs,
            ),
        )
        return cast(OaKState, old.replace(stomp_state=next_stomp))

    def observe_and_route(
        self,
        state: PrototypeFeatureLifecycleState,
        oak_state: OaKState,
        consumer_binding: PrototypeFeatureConsumerBinding,
        event: PrototypeFeatureLifecycleEvent,
    ) -> PrototypeFeatureLifecycleResult:
        """Learn once, then defer or atomically route a descriptor mutation.

        Static mismatches raise before indexed work.  Dynamic invalidity is an
        exact no-op for both owned state and the supplied OaK state.  A route
        failure preserves ordinary learning via ``pre_curation_state`` and
        rolls back only the descriptor mutation.
        """

        if not self._state_static_contract_valid(state):
            raise ValueError("prototype feature lifecycle state has an invalid static contract")
        if type(consumer_binding) is not PrototypeFeatureConsumerBinding:
            raise TypeError(
                "consumer_binding must be a PrototypeFeatureConsumerBinding"
            )
        _require_array(
            consumer_binding.semantic_generation,
            name="consumer_binding.semantic_generation",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            consumer_binding.descriptors,
            name="consumer_binding.descriptors",
            shape=(self._config.active_pair_slots, 2),
            dtype=jnp.int32,
        )
        if type(event) is not PrototypeFeatureLifecycleEvent:
            raise TypeError("event must be a PrototypeFeatureLifecycleEvent")
        observation = _require_array(
            event.observation,
            name="event.observation",
            shape=(self._config.base_feature_dim,),
            dtype=jnp.float32,
        )
        targets = _require_array(
            event.targets,
            name="event.targets",
            shape=(self._config.n_tasks,),
            dtype=jnp.float32,
        )
        next_observation = _require_array(
            event.next_observation,
            name="event.next_observation",
            shape=(self._config.base_feature_dim,),
            dtype=jnp.float32,
        )
        allow_curation = _require_array(
            event.allow_curation,
            name="event.allow_curation",
            shape=(),
            dtype=jnp.bool_,
        )
        if not self._oak_static_contract_valid(oak_state):
            raise ValueError(
                "oak_state must satisfy the exact supported linear OaK static contract"
            )

        state_values_valid = self.state_valid(state)
        oak_values_valid = self._oak_values_valid(oak_state)
        consumer_binding_values_valid = self.consumer_binding_valid(
            state,
            consumer_binding,
        )
        event_values_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.all(jnp.isfinite(targets) | jnp.isnan(targets))
        )
        old_next_augmented = self.augment(state, next_observation)
        next_observation_matches_oak_cache = _float32_arrays_bit_exact(
            oak_state.stomp_state.base_last_obs,
            old_next_augmented,
        )
        update_capacity_available = (
            state.observe_count
            < jnp.asarray(self._config.max_observations, dtype=jnp.int32)
        )
        composition_valid = (
            state_values_valid
            & oak_values_valid
            & consumer_binding_values_valid
            & event_values_valid
            & next_observation_matches_oak_cache
            & update_capacity_available
        )

        safe_observation = jnp.where(jnp.isfinite(observation), observation, 0.0)
        safe_targets = jnp.where(
            jnp.isfinite(targets) | jnp.isnan(targets),
            targets,
            jnp.nan,
        )
        learner_update = self._learner.update(
            state.learner_state,
            safe_observation,
            safe_targets,
        )
        learner_update_valid = composition_valid & ~learner_update.update_rejected
        proposed_descriptors = jnp.stack(
            (
                learner_update.state.feature_left,
                learner_update.state.feature_right,
            ),
            axis=1,
        )
        pre_curation_descriptors = jnp.stack(
            (
                learner_update.pre_curation_state.feature_left,
                learner_update.pre_curation_state.feature_right,
            ),
            axis=1,
        )
        curation_proposed = learner_update_valid & jnp.any(
            proposed_descriptors != pre_curation_descriptors
        )
        stomp = oak_state.stomp_state
        safe_curation_boundary = (
            allow_curation
            & (stomp.executing_option == -1)
            & (stomp.base_last_action < self._config.n_primitive_actions)
        )
        curation_deferred = (
            curation_proposed & ~safe_curation_boundary
        )
        routing_attempted = curation_proposed & safe_curation_boundary

        input_route = self._router.route(
            state.router_state,
            self._input_consumers(oak_state),
            proposed_descriptors,
            carry_survivors=self._config.carry_survivors,
        )
        routed_inputs = cast(dict[str, Any], input_route.consumers)
        output_route = self._router.route(
            state.router_state,
            routed_inputs["option_model_input_weights"],
            proposed_descriptors,
            feature_axes=1,
            carry_survivors=self._config.carry_survivors,
        )
        route_states_match_raw = _trees_exactly_equal(
            input_route.state,
            output_route.state,
        )
        routed_values_finite_raw = (
            _floating_tree_is_finite(routed_inputs)
            & _floating_tree_is_finite(output_route.consumers)
        )
        route_valid = (
            input_route.diagnostics.valid
            & output_route.diagnostics.valid
            & route_states_match_raw
            & routed_values_finite_raw
            & input_route.diagnostics.descriptors_changed
            & output_route.diagnostics.descriptors_changed
        )
        provisional_curation_committed = routing_attempted & route_valid
        route_curation_rolled_back = routing_attempted & ~route_valid

        learned_state = jax.lax.cond(
            curation_proposed & ~provisional_curation_committed,
            lambda _: learner_update.pre_curation_state,
            lambda _: learner_update.state,
            operand=None,
        )
        next_learner_state = jax.lax.cond(
            learner_update_valid,
            lambda _: learned_state,
            lambda _: state.learner_state,
            operand=None,
        )
        one = jnp.asarray(1, dtype=jnp.int32)
        next_observe_count = jnp.where(
            learner_update_valid,
            state.observe_count + one,
            state.observe_count,
        )
        next_deferred_count = jnp.where(
            curation_deferred,
            state.deferred_curation_count + one,
            state.deferred_curation_count,
        )
        next_committed_count = jnp.where(
            provisional_curation_committed,
            state.committed_curation_count + one,
            state.committed_curation_count,
        )
        next_rolled_back_count = jnp.where(
            route_curation_rolled_back,
            state.rolled_back_curation_count + one,
            state.rolled_back_curation_count,
        )
        candidate_router_state = input_route.state
        next_router_state = FeatureBankRouterState(
            descriptors=jnp.where(
                provisional_curation_committed,
                candidate_router_state.descriptors,
                state.router_state.descriptors,
            ),
            route_count=jnp.where(
                provisional_curation_committed,
                candidate_router_state.route_count,
                state.router_state.route_count,
            ),
            generation_count=jnp.where(
                provisional_curation_committed,
                candidate_router_state.generation_count,
                state.router_state.generation_count,
            ),
        )
        candidate_state = PrototypeFeatureLifecycleState(
            learner_state=next_learner_state,
            router_state=next_router_state,
            observe_count=next_observe_count,
            deferred_curation_count=next_deferred_count,
            committed_curation_count=next_committed_count,
            rolled_back_curation_count=next_rolled_back_count,
        )
        candidate_consumer_binding = PrototypeFeatureConsumerBinding(
            semantic_generation=next_router_state.generation_count,
            descriptors=next_router_state.descriptors,
        )

        safe_next_observation = jnp.where(
            jnp.isfinite(next_observation),
            next_observation,
            0.0,
        )
        committed_next_augmented = self._learner.augmented_observation(
            learner_update.state,
            safe_next_observation,
        )
        candidate_oak_state = self._routed_oak_state(
            oak_state,
            routed_inputs,
            cast(Array, output_route.consumers),
            committed_next_augmented,
            provisional_curation_committed,
        )
        candidate_next_augmented = self._learner.augmented_observation(
            next_learner_state,
            safe_next_observation,
        )
        candidate_state_valid = self.state_valid(candidate_state)
        candidate_oak_values_valid = self._oak_values_valid(candidate_oak_state)
        candidate_cache_valid = _float32_arrays_bit_exact(
            candidate_oak_state.stomp_state.base_last_obs,
            candidate_next_augmented,
        )
        candidate_postcondition_valid = (
            candidate_state_valid
            & candidate_oak_values_valid
            & self.consumer_binding_valid(
                candidate_state,
                candidate_consumer_binding,
            )
            & candidate_cache_valid
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.all(jnp.isfinite(candidate_next_augmented))
        )
        postcondition_valid = learner_update_valid & candidate_postcondition_valid
        postcondition_rolled_back = learner_update_valid & ~candidate_postcondition_valid
        transaction_applied = learner_update_valid & candidate_postcondition_valid
        curation_committed = (
            provisional_curation_committed & candidate_postcondition_valid
        )
        curation_rolled_back = (
            route_curation_rolled_back
            | (postcondition_rolled_back & curation_proposed)
        )
        next_state = jax.lax.cond(
            postcondition_rolled_back,
            lambda _: state,
            lambda _: candidate_state,
            operand=None,
        )
        next_oak_state = self._routed_oak_state(
            oak_state,
            routed_inputs,
            cast(Array, output_route.consumers),
            committed_next_augmented,
            curation_committed,
        )
        next_consumer_binding = PrototypeFeatureConsumerBinding(
            semantic_generation=jnp.where(
                curation_committed,
                candidate_consumer_binding.semantic_generation,
                consumer_binding.semantic_generation,
            ),
            descriptors=jnp.where(
                curation_committed,
                candidate_consumer_binding.descriptors,
                consumer_binding.descriptors,
            ),
        )
        final_next_augmented = self._learner.augmented_observation(
            next_state.learner_state,
            safe_next_observation,
        )
        final_next_valid = (
            self.state_valid(next_state)
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.all(jnp.isfinite(final_next_augmented))
        )
        final_next_augmented = jnp.where(
            final_next_valid,
            final_next_augmented,
            jnp.zeros_like(final_next_augmented),
        )

        diagnostics = PrototypeFeatureLifecycleDiagnostics(
            available=jnp.asarray(True, dtype=jnp.bool_),
            state_values_valid=state_values_valid,
            oak_values_valid=oak_values_valid,
            consumer_binding_valid=consumer_binding_values_valid,
            event_values_valid=event_values_valid,
            next_observation_matches_oak_cache=(
                next_observation_matches_oak_cache
            ),
            update_capacity_available=update_capacity_available,
            learner_update_rejected=(
                ~composition_valid | learner_update.update_rejected
            ),
            transaction_applied=transaction_applied,
            curation_proposed=curation_proposed,
            safe_curation_boundary=safe_curation_boundary,
            curation_deferred=curation_deferred,
            routing_attempted=routing_attempted,
            input_route_valid=(
                routing_attempted & input_route.diagnostics.valid
            ),
            output_route_valid=(
                routing_attempted & output_route.diagnostics.valid
            ),
            route_states_match=routing_attempted & route_states_match_raw,
            routed_values_finite=(
                routing_attempted & routed_values_finite_raw
            ),
            curation_committed=curation_committed,
            curation_rolled_back=curation_rolled_back,
            postcondition_checked=learner_update_valid,
            postcondition_valid=postcondition_valid,
            postcondition_rolled_back=postcondition_rolled_back,
            semantic_generation_before=state.router_state.generation_count,
            semantic_generation_after=next_state.router_state.generation_count,
        )
        return PrototypeFeatureLifecycleResult(
            state=next_state,
            oak_state=next_oak_state,
            consumer_binding=next_consumer_binding,
            next_augmented_observation=final_next_augmented,
            predictions=jnp.where(
                transaction_applied,
                learner_update.predictions,
                jnp.full_like(learner_update.predictions, jnp.nan),
            ),
            errors=jnp.where(
                transaction_applied,
                learner_update.errors,
                jnp.full_like(learner_update.errors, jnp.nan),
            ),
            metrics=jnp.where(
                transaction_applied,
                learner_update.metrics,
                jnp.zeros_like(learner_update.metrics),
            ),
            input_route_diagnostics=input_route.diagnostics,
            output_route_diagnostics=output_route.diagnostics,
            diagnostics=diagnostics,
        )


def save_prototype_feature_lifecycle_checkpoint(
    lifecycle: PrototypeFeatureLifecycle,
    state: PrototypeFeatureLifecycleState,
    path: str | Path,
) -> None:
    """Persist only a valid owned lifecycle state and its exact L0 contract."""

    if type(lifecycle) is not PrototypeFeatureLifecycle:
        raise TypeError("lifecycle must be a PrototypeFeatureLifecycle")
    if not bool(lifecycle.state_valid(state)):
        raise ValueError("prototype feature lifecycle state is invalid")
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": PROTOTYPE_FEATURE_LIFECYCLE_CHECKPOINT_SCHEMA,
            "mechanism_status": PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS,
            "scientific_promotion_allowed": (
                PROTOTYPE_FEATURE_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED
            ),
            "config": lifecycle.to_config(),
            "resource_budget": lifecycle.resource_budget(state).to_config(),
        },
    )


def load_prototype_feature_lifecycle_checkpoint(
    path: str | Path,
) -> tuple[PrototypeFeatureLifecycle, PrototypeFeatureLifecycleState]:
    """Restore only the exact v1 schema, structure, resources, and state."""

    metadata = load_checkpoint_metadata(path)
    expected = {
        "schema",
        "mechanism_status",
        "scientific_promotion_allowed",
        "config",
        "resource_budget",
    }
    if set(metadata) != expected:
        raise ValueError("prototype feature lifecycle checkpoint fields are invalid")
    if metadata.get("schema") != PROTOTYPE_FEATURE_LIFECYCLE_CHECKPOINT_SCHEMA:
        raise ValueError("prototype feature lifecycle checkpoint schema is unsupported")
    if (
        metadata.get("mechanism_status")
        != PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS
    ):
        raise ValueError("prototype feature lifecycle checkpoint is not mechanism-only")
    if metadata.get("scientific_promotion_allowed") is not False:
        raise ValueError("prototype feature lifecycle checkpoint cannot claim promotion")
    raw_config = metadata.get("config")
    if type(raw_config) is not dict:
        raise ValueError("prototype feature lifecycle checkpoint config is invalid")
    lifecycle = PrototypeFeatureLifecycle.from_config(raw_config)
    template = lifecycle.init(jr.key(0))
    restored, restored_metadata = load_checkpoint(template, path)
    if not _exact_json_tree_equal(restored_metadata, metadata):
        raise ValueError("prototype feature lifecycle checkpoint metadata changed between reads")
    state = cast(PrototypeFeatureLifecycleState, restored)
    if not bool(lifecycle.state_valid(state)):
        raise ValueError("prototype feature lifecycle checkpoint state is invalid")
    budget = metadata.get("resource_budget")
    if type(budget) is not dict:
        raise ValueError("prototype feature lifecycle checkpoint resource budget is invalid")
    expected_budget = lifecycle.resource_budget(state).to_config()
    if not _exact_json_tree_equal(budget, expected_budget):
        raise ValueError("prototype feature lifecycle checkpoint resource contract changed")
    return lifecycle, state


__all__ = [
    "PROTOTYPE_FEATURE_LIFECYCLE_CHECKPOINT_SCHEMA",
    "PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA",
    "PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS",
    "PROTOTYPE_FEATURE_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED",
    "PrototypeFeatureConsumerBinding",
    "PrototypeFeatureLifecycle",
    "PrototypeFeatureLifecycleConfig",
    "PrototypeFeatureLifecycleDiagnostics",
    "PrototypeFeatureLifecycleEvent",
    "PrototypeFeatureLifecycleResourceBudget",
    "PrototypeFeatureLifecycleResult",
    "PrototypeFeatureLifecycleState",
    "PrototypePairGradientPullback",
    "load_prototype_feature_lifecycle_checkpoint",
    "save_prototype_feature_lifecycle_checkpoint",
]
