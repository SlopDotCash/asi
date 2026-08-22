# mypy: disable-error-code="call-arg,name-defined"
"""Bounded, fail-closed experiential memory for continuing agents.

The memory stores fixed-width episodic exemplars and retrieves from the state
that existed before the current exemplar is written.  Retrieval keeps reward,
safety, uncertainty, reliability, and eviction utility as separate signals;
in particular, reward is never folded into either reliability or utility.

All persistent state is held in fixed-shape JAX arrays.  The implementation is
therefore compatible with ``jax.jit`` and ``jax.lax.scan`` and exposes exact
array-byte and entry-count accounting.

The design follows the episodic-memory lineage: stored transitions reused as
learning signal (Lin 1992) and similarity-weighted nearest-neighbor retrieval
over exemplar keys (Blundell et al. 2016), here with explicit reliability,
staleness, and safety gating instead of unconditional recall.

References:
    Lin (1992). "Self-Improving Reactive Agents Based on Reinforcement
        Learning, Planning and Teaching."
    Blundell et al. (2016). "Model-Free Episodic Control."
"""

from __future__ import annotations

import functools
import operator
from dataclasses import asdict, dataclass, fields
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core._float32_scalars import validated_float32_scalar

_INT32_MAX = 2_147_483_647
_UINT32_MAX = 4_294_967_295

_ACTUAL_INT_TYPES = frozenset({int, *(np.dtype(code).type for code in "bBhHiIlLqQpP")})
_ACTUAL_FLOAT_TYPES = frozenset(
    {float, *(np.dtype(code).type for code in ("e", "f", "d", "g"))}
)

@dataclass(frozen=True)
class ExperientialMemoryConfig:
    """Static allocation and retrieval policy for experiential memory.

    Args:
        capacity: Maximum number of persistent exemplars.
        observation_dim: Width of the stored grounded observation.
        key_dim: Width of the retrieval key.
        action_dim: Width of the stored action.
        outcome_dim: Width of the stored outcome vector.
        top_k: Maximum neighbors considered by one query.
        min_neighbors: Minimum eligible neighbors required to retrieve.
        distance_scale: Positive scale for the RBF key similarity
            ``exp(-mean_squared_distance / distance_scale)``, where the
            distance is the per-dimension mean of squared key differences.
            It is therefore in squared key units and must be calibrated to
            the key representation's magnitude.
        min_similarity: Minimum RBF similarity for an eligible neighbor.
            With the defaults, a neighbor qualifies when its mean squared
            per-dimension key distance is at most
            ``distance_scale * ln(2) ~= 0.69``.
        min_effective_reliability: Minimum reliability after staleness decay.
        max_uncertainty: Maximum query and exemplar uncertainty.
        max_safety_cost: Maximum exemplar safety cost.
        max_age: Maximum chronological exemplar age.
        staleness_scale: Exponential reliability-decay timescale.
        utility_decay: Per-step decay for explicit eviction utility.
        eviction_utility_weight: Utility contribution to retention priority.
        eviction_recency_weight: Recency contribution to retention priority.
        recency_scale: Scale of the reciprocal recency score.
    """

    capacity: int
    observation_dim: int
    key_dim: int
    action_dim: int
    outcome_dim: int
    top_k: int = 4
    min_neighbors: int = 1
    distance_scale: float = 1.0
    min_similarity: float = 0.5
    min_effective_reliability: float = 0.1
    max_uncertainty: float = 1.0
    max_safety_cost: float = 1.0
    max_age: int = 10_000
    staleness_scale: float = 1_000.0
    utility_decay: float = 0.999
    eviction_utility_weight: float = 1.0
    eviction_recency_weight: float = 1.0
    recency_scale: float = 100.0

    def __post_init__(self) -> None:
        _validate_config(self)

    def to_config(self) -> dict[str, object]:
        """Return a JSON-serializable configuration."""
        payload = asdict(self)
        payload["type"] = "ExperientialMemoryConfig"
        return payload

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExperientialMemoryConfig:
        """Reconstruct and validate a configuration dictionary."""
        if type(config) is not dict:
            raise ValueError("config must be an exact built-in dict")
        expected = {field.name for field in fields(cls)} | {"type"}
        if any(type(key) is not str for key in config) or set(config) != expected:
            raise ValueError("config fields do not match the serialized schema")
        if type(config["type"]) is not str or config["type"] != "ExperientialMemoryConfig":
            raise ValueError("config type differs")
        payload = dict(config)
        payload.pop("type")
        return cls(**payload)

@chex.dataclass(frozen=True)
class ExperientialMemoryEntry:
    """One typed experiential exemplar presented for bounded storage."""

    observation: Float[Array, " observation_dim"]
    key: Float[Array, " key_dim"]
    action: Float[Array, " action_dim"]
    outcome: Float[Array, " outcome_dim"]
    reward: Float[Array, ""]
    uncertainty: Float[Array, ""]
    uncertainty_available: Bool[Array, ""]
    safety_cost: Float[Array, ""]
    safety_cost_available: Bool[Array, ""]
    reliability: Float[Array, ""]
    utility: Float[Array, ""]
    utility_available: Bool[Array, ""]
    representation_version: Int[Array, ""]
    valid: Bool[Array, ""]
    age: Int[Array, ""]
    provenance_id: Int[Array, ""]
    source_id: Int[Array, ""]

@chex.dataclass(frozen=True)
class ExperientialMemoryEntries:
    """Fixed-capacity structure-of-arrays for persistent exemplars.

    ``ages`` is the exemplar's chronological age. ``recency_ages`` is local
    time since its last accepted retrieval: zero on write and on access, then
    incremented once per later memory step while the slot remains valid.
    """

    observations: Float[Array, "capacity observation_dim"]
    keys: Float[Array, "capacity key_dim"]
    actions: Float[Array, "capacity action_dim"]
    outcomes: Float[Array, "capacity outcome_dim"]
    rewards: Float[Array, " capacity"]
    uncertainties: Float[Array, " capacity"]
    uncertainty_available: Bool[Array, " capacity"]
    safety_costs: Float[Array, " capacity"]
    safety_cost_available: Bool[Array, " capacity"]
    reliabilities: Float[Array, " capacity"]
    utilities: Float[Array, " capacity"]
    utility_available: Bool[Array, " capacity"]
    representation_versions: Int[Array, " capacity"]
    valid: Bool[Array, " capacity"]
    ages: Int[Array, " capacity"]
    recency_ages: Int[Array, " capacity"]
    provenance_ids: Int[Array, " capacity"]
    source_ids: Int[Array, " capacity"]
    retrieval_counts: Int[Array, " capacity"]

@chex.dataclass(frozen=True)
class ExperientialMemoryState:
    """Complete fixed-shape persistent memory state."""

    entries: ExperientialMemoryEntries
    active_count: Int[Array, ""]
    step_count: Int[Array, ""]
    query_count: Int[Array, ""]
    accepted_query_count: Int[Array, ""]
    write_count: Int[Array, ""]
    rejected_write_count: Int[Array, ""]
    eviction_count: Int[Array, ""]
    persistent_bytes: Array

@chex.dataclass(frozen=True)
class ExperientialMemoryRetrieval:
    """A gated retrieval and its auditable neighbor diagnostics.

    Retrieved payloads are all-zero when ``accepted`` is false.  Neighbor
    diagnostics remain available so an evaluator can localize an abstention.
    """

    accepted: Bool[Array, ""]
    observation: Float[Array, " observation_dim"]
    action: Float[Array, " action_dim"]
    outcome: Float[Array, " outcome_dim"]
    reward: Float[Array, ""]
    uncertainty: Float[Array, ""]
    safety_cost: Float[Array, ""]
    effective_reliability: Float[Array, ""]
    neighbor_indices: Int[Array, " top_k"]
    neighbor_mask: Bool[Array, " top_k"]
    neighbor_weights: Float[Array, " top_k"]
    neighbor_similarities: Float[Array, " top_k"]
    neighbor_reliabilities: Float[Array, " top_k"]
    neighbor_ages: Int[Array, " top_k"]
    neighbor_provenance_ids: Int[Array, " top_k"]
    state_valid: Bool[Array, ""]
    query_valid: Bool[Array, ""]
    version_compatible: Bool[Array, ""]
    freshness_ok: Bool[Array, ""]
    uncertainty_available: Bool[Array, ""]
    safety_cost_available: Bool[Array, ""]
    uncertainty_ok: Bool[Array, ""]
    safety_ok: Bool[Array, ""]
    has_neighbors: Bool[Array, ""]

@chex.dataclass(frozen=True)
class ExperientialMemoryWriteResult:
    """State and accounting returned by one bounded write."""

    state: ExperientialMemoryState
    wrote: Bool[Array, ""]
    slot: Int[Array, ""]
    evicted: Bool[Array, ""]
    evicted_provenance_id: Int[Array, ""]

@chex.dataclass(frozen=True)
class ExperientialMemoryStepResult:
    """One causal query-before-write operation."""

    state: ExperientialMemoryState
    retrieval: ExperientialMemoryRetrieval
    wrote: Bool[Array, ""]
    slot: Int[Array, ""]
    evicted: Bool[Array, ""]
    evicted_provenance_id: Int[Array, ""]

@chex.dataclass(frozen=True)
class ExperientialMemoryAccounting:
    """Exact persistent allocation and lifetime operation counts."""

    active_entries: Int[Array, ""]
    capacity_entries: Int[Array, ""]
    slot_bytes: Array
    persistent_bytes: Array
    queries: Int[Array, ""]
    accepted_queries: Int[Array, ""]
    writes: Int[Array, ""]
    rejected_writes: Int[Array, ""]
    evictions: Int[Array, ""]

def _require_positive_int(name: str, value: object) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be a positive integer in [1, {_INT32_MAX}]")
    number = operator.index(cast(SupportsIndex, value))
    if number < 1 or number > _INT32_MAX:
        raise ValueError(f"{name} must be a positive integer in [1, {_INT32_MAX}]")
    return number

def _require_nonnegative_int(name: str, value: object) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [0, {_INT32_MAX}]")
    number = operator.index(cast(SupportsIndex, value))
    if number < 0 or number > _INT32_MAX:
        raise ValueError(f"{name} must be an integer in [0, {_INT32_MAX}]")
    return number

def _validated_config_float(name: str, value: object, **bounds: Any) -> float:
    if type(value) not in (frozenset(_ACTUAL_INT_TYPES) | _ACTUAL_FLOAT_TYPES):
        raise ValueError(f"{name} must be a finite real scalar")
    return validated_float32_scalar(name, value, **bounds)


def _require_float32_allocation(name: str, scalars: int) -> None:
    if scalars > _INT32_MAX or 4 * scalars > _INT32_MAX:
        raise ValueError(f"{name} scalar and byte counts must fit signed int32")


def _experiential_persistent_bytes(
    *,
    capacity: int,
    observation_dim: int,
    key_dim: int,
    action_dim: int,
    outcome_dim: int,
) -> int:
    """Named persist already preflighted by ``_validate_config``."""
    vector_values = observation_dim + key_dim + action_dim + outcome_dim
    return capacity * (4 * (vector_values + 11) + 4) + 32


def _experiential_step_result_extras_bytes(
    *,
    observation_dim: int,
    action_dim: int,
    outcome_dim: int,
    top_k: int,
) -> int:
    """Returned ``ExperientialMemoryStepResult`` extras excluding persist.

    Nested ``state`` is persist and is already counted in the simultaneous
    persist copies. These extras are the published retrieval and write
    diagnostic leaves.
    """
    payload_values = observation_dim + action_dim + outcome_dim
    return 36 + 4 * payload_values + 25 * top_k


def _experiential_update_working_set_bytes(
    *,
    capacity: int,
    observation_dim: int,
    key_dim: int,
    action_dim: int,
    outcome_dim: int,
    top_k: int,
) -> int:
    """Source persist, proposed persist, committed persist, and returned extras."""
    persist_bytes = _experiential_persistent_bytes(
        capacity=capacity,
        observation_dim=observation_dim,
        key_dim=key_dim,
        action_dim=action_dim,
        outcome_dim=outcome_dim,
    )
    extras_bytes = _experiential_step_result_extras_bytes(
        observation_dim=observation_dim,
        action_dim=action_dim,
        outcome_dim=outcome_dim,
        top_k=top_k,
    )
    return 3 * persist_bytes + extras_bytes


def _preflight_experiential_update_working_set(config: ExperientialMemoryConfig) -> None:
    """Reject an update envelope the host cannot name in signed int32."""
    working_set_bytes = _experiential_update_working_set_bytes(
        capacity=config.capacity,
        observation_dim=config.observation_dim,
        key_dim=config.key_dim,
        action_dim=config.action_dim,
        outcome_dim=config.outcome_dim,
        top_k=config.top_k,
    )
    if working_set_bytes > _INT32_MAX:
        raise ValueError(
            "experiential memory update working set byte count must fit signed int32"
        )


def _validate_config(config: ExperientialMemoryConfig) -> None:
    for name in (
        "capacity",
        "observation_dim",
        "key_dim",
        "action_dim",
        "outcome_dim",
        "top_k",
        "min_neighbors",
    ):
        canonical = _require_positive_int(name, getattr(config, name))
        object.__setattr__(config, name, canonical)
    if config.top_k > config.capacity:
        raise ValueError("top_k must be <= capacity")
    if config.min_neighbors > config.top_k:
        raise ValueError("min_neighbors must be <= top_k")
    canonical_max_age = _require_nonnegative_int("max_age", config.max_age)
    object.__setattr__(config, "max_age", canonical_max_age)

    vector_values = (
        config.observation_dim + config.key_dim + config.action_dim + config.outcome_dim
    )
    if vector_values > _INT32_MAX:
        raise ValueError("ExperientialMemoryConfig dimensions must fit signed int32")
    logical_scalars = config.capacity * (vector_values + 15) + 8
    persistent_bytes = config.capacity * (4 * (vector_values + 11) + 4) + 32
    if logical_scalars > _INT32_MAX:
        raise ValueError("ExperientialMemoryConfig state scalar count must fit signed int32")
    if persistent_bytes > _INT32_MAX:
        raise ValueError("ExperientialMemoryConfig state byte count must fit signed int32")
    if persistent_bytes > _UINT32_MAX:
        raise ValueError("persistent memory allocation exceeds uint32 byte accounting")
    for name, scalars in (
        ("query key-distance work", config.capacity * config.key_dim),
        ("query observation gather", config.top_k * config.observation_dim),
        ("query key gather", config.top_k * config.key_dim),
        ("query action gather", config.top_k * config.action_dim),
        ("query outcome gather", config.top_k * config.outcome_dim),
        ("query gathered payload", config.top_k * vector_values),
        ("query neighbor diagnostics", 7 * config.top_k),
    ):
        _require_float32_allocation(f"ExperientialMemoryConfig {name}", scalars)
    payload_values = config.observation_dim + config.action_dim + config.outcome_dim
    # Conservative simultaneous query envelope, expressed as four-byte scalar
    # equivalents.  It includes the persistent state plus every full-width
    # logical temporary in _query_jit: row-finiteness predicates; safe keys,
    # key differences and squares; safe payload matrices; nineteen capacity
    # vectors; gathered values and their weighted products; top-k diagnostics;
    # and reduced payload outputs.  Booleans can be smaller, but are charged at
    # four bytes so this remains an upper bound across JAX backends.
    query_temporary_scalars = (
        config.capacity * vector_values
        + 3 * config.capacity * config.key_dim
        + config.capacity * payload_values
        + 19 * config.capacity
        + 2 * config.top_k * (payload_values + 4)
        + 9 * config.top_k
        + payload_values
        + 4
    )
    if persistent_bytes + 4 * query_temporary_scalars > _INT32_MAX:
        raise ValueError(
            "ExperientialMemoryConfig aggregate query working-set bytes "
            "must fit signed int32"
        )
    _preflight_experiential_update_working_set(config)

    object.__setattr__(
        config,
        "distance_scale",
        _validated_config_float("distance_scale", config.distance_scale, positive=True),
    )
    object.__setattr__(
        config,
        "min_similarity",
        _validated_config_float("min_similarity", config.min_similarity, lower=0, upper=1),
    )
    object.__setattr__(
        config,
        "min_effective_reliability",
        _validated_config_float(
            "min_effective_reliability", config.min_effective_reliability, positive=True, upper=1
        ),
    )
    object.__setattr__(
        config,
        "max_uncertainty",
        _validated_config_float("max_uncertainty", config.max_uncertainty, lower=0),
    )
    object.__setattr__(
        config,
        "max_safety_cost",
        _validated_config_float("max_safety_cost", config.max_safety_cost, lower=0),
    )
    object.__setattr__(
        config,
        "staleness_scale",
        _validated_config_float("staleness_scale", config.staleness_scale, positive=True),
    )
    object.__setattr__(
        config,
        "utility_decay",
        _validated_config_float("utility_decay", config.utility_decay, lower=0, upper=1),
    )
    object.__setattr__(
        config,
        "eviction_utility_weight",
        _validated_config_float(
            "eviction_utility_weight", config.eviction_utility_weight, lower=0
        ),
    )
    object.__setattr__(
        config,
        "eviction_recency_weight",
        _validated_config_float(
            "eviction_recency_weight", config.eviction_recency_weight, lower=0
        ),
    )
    utility_weight_f32 = float(np.asarray(config.eviction_utility_weight, dtype=np.float32))
    recency_weight_f32 = float(np.asarray(config.eviction_recency_weight, dtype=np.float32))
    if utility_weight_f32 + recency_weight_f32 <= 0.0:
        raise ValueError("at least one eviction retention weight must be positive")
    object.__setattr__(
        config,
        "recency_scale",
        _validated_config_float("recency_scale", config.recency_scale, positive=True),
    )
def _saturating_increment(value: Array) -> Array:
    maximum_minus_one = jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)
    return jnp.minimum(value, maximum_minus_one) + jnp.asarray(1, dtype=jnp.int32)

def _tree_nbytes(tree: Any) -> int:
    return sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in jax.tree.leaves(tree))

def _configured_nbytes(config: ExperientialMemoryConfig) -> tuple[int, int]:
    vector_values = config.observation_dim + config.key_dim + config.action_dim + config.outcome_dim
    # Five float scalars, six int32 scalars, and four bools per slot.
    slot_bytes = 4 * (vector_values + 5 + 6) + 4
    # Seven int32 counters and the uint32 byte-count scalar live beside slots.
    persistent_bytes = config.capacity * slot_bytes + 8 * 4
    return persistent_bytes, slot_bytes

def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    """Reject structural drift before eager execution or JAX tracing."""
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array with shape and dtype metadata")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")
    expected_dtype = jnp.dtype(dtype)
    actual_dtype = jnp.dtype(value.dtype)
    if actual_dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}; got {actual_dtype}")

class ExperientialMemory:
    """Fixed-capacity episodic memory with conservative retrieval.

    ``query`` is read-only.  ``step`` first performs that query against the
    previous state, records any accepted access, advances ages once, and only
    then writes the supplied exemplar.  This ordering prevents target leakage
    from a current transition into its own prediction.
    """

    def __init__(self, config: ExperientialMemoryConfig):
        _validate_config(config)
        self._config = config
        persistent_bytes, slot_bytes = _configured_nbytes(config)
        if persistent_bytes > _UINT32_MAX:
            raise ValueError("persistent memory allocation exceeds uint32 byte accounting")
        self._persistent_bytes = persistent_bytes
        self._slot_bytes = slot_bytes

    @property
    def config(self) -> ExperientialMemoryConfig:
        """Memory configuration."""
        return self._config

    @property
    def persistent_bytes(self) -> int:
        """Exact bytes occupied by all persistent JAX array leaves."""
        return self._persistent_bytes

    @property
    def slot_bytes(self) -> int:
        """Exact bytes occupied by one fixed-capacity entry slot."""
        return self._slot_bytes

    def to_config(self) -> dict[str, object]:
        """Serialize the memory construction configuration."""
        return {
            "type": "ExperientialMemory",
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExperientialMemory:
        """Reconstruct a memory from :meth:`to_config` output."""
        if type(config) is not dict:
            raise ValueError("config must be an exact built-in dict")
        if any(type(key) is not str for key in config) or set(config) != {"type", "config"}:
            raise ValueError("config fields do not match the serialized schema")
        if type(config["type"]) is not str or config["type"] != "ExperientialMemory":
            raise ValueError("config type differs")
        if type(config["config"]) is not dict:
            raise ValueError("nested config must be an exact built-in dict")
        inner = cast(dict[str, Any], config["config"])
        return cls(ExperientialMemoryConfig.from_config(inner))

    def _validate_state_static_contract(self, state: ExperientialMemoryState) -> None:
        """Validate every persistent leaf without reshaping or lossy casting."""
        if not isinstance(state, ExperientialMemoryState):
            raise TypeError("state must be an ExperientialMemoryState")
        if not isinstance(state.entries, ExperientialMemoryEntries):
            raise TypeError("state.entries must be ExperientialMemoryEntries")
        cfg = self._config
        entries = state.entries
        float_fields = {
            "observations": (cfg.capacity, cfg.observation_dim),
            "keys": (cfg.capacity, cfg.key_dim),
            "actions": (cfg.capacity, cfg.action_dim),
            "outcomes": (cfg.capacity, cfg.outcome_dim),
            "rewards": (cfg.capacity,),
            "uncertainties": (cfg.capacity,),
            "safety_costs": (cfg.capacity,),
            "reliabilities": (cfg.capacity,),
            "utilities": (cfg.capacity,),
        }
        for field_name, shape in float_fields.items():
            _require_array(
                getattr(entries, field_name),
                name=f"state.entries.{field_name}",
                shape=shape,
                dtype=jnp.float32,
            )
        for field_name in (
            "uncertainty_available",
            "safety_cost_available",
            "utility_available",
            "valid",
        ):
            _require_array(
                getattr(entries, field_name),
                name=f"state.entries.{field_name}",
                shape=(cfg.capacity,),
                dtype=jnp.bool_,
            )
        for field_name in (
            "representation_versions",
            "ages",
            "recency_ages",
            "provenance_ids",
            "source_ids",
            "retrieval_counts",
        ):
            _require_array(
                getattr(entries, field_name),
                name=f"state.entries.{field_name}",
                shape=(cfg.capacity,),
                dtype=jnp.int32,
            )
        for field_name in (
            "active_count",
            "step_count",
            "query_count",
            "accepted_query_count",
            "write_count",
            "rejected_write_count",
            "eviction_count",
        ):
            _require_array(
                getattr(state, field_name),
                name=f"state.{field_name}",
                shape=(),
                dtype=jnp.int32,
            )
        _require_array(
            state.persistent_bytes,
            name="state.persistent_bytes",
            shape=(),
            dtype=jnp.uint32,
        )

    def _validate_entry_static_contract(self, entry: ExperientialMemoryEntry) -> None:
        """Reject same-size wrong shapes and dtype aliases at the write boundary."""
        if not isinstance(entry, ExperientialMemoryEntry):
            raise TypeError("entry must be an ExperientialMemoryEntry")
        cfg = self._config
        for field_name, shape in {
            "observation": (cfg.observation_dim,),
            "key": (cfg.key_dim,),
            "action": (cfg.action_dim,),
            "outcome": (cfg.outcome_dim,),
            "reward": (),
            "uncertainty": (),
            "safety_cost": (),
            "reliability": (),
            "utility": (),
        }.items():
            _require_array(
                getattr(entry, field_name),
                name=f"entry.{field_name}",
                shape=shape,
                dtype=jnp.float32,
            )
        for field_name in (
            "uncertainty_available",
            "safety_cost_available",
            "utility_available",
            "valid",
        ):
            _require_array(
                getattr(entry, field_name),
                name=f"entry.{field_name}",
                shape=(),
                dtype=jnp.bool_,
            )
        for field_name in (
            "representation_version",
            "age",
            "provenance_id",
            "source_id",
        ):
            _require_array(
                getattr(entry, field_name),
                name=f"entry.{field_name}",
                shape=(),
                dtype=jnp.int32,
            )

    def _state_is_valid(self, state: ExperientialMemoryState) -> Bool[Array, ""]:
        """Return the global dynamic invariant for one structurally valid state."""
        entries = state.entries
        valid = entries.valid
        all_float_payload_finite = (
            jnp.all(jnp.isfinite(entries.observations))
            & jnp.all(jnp.isfinite(entries.keys))
            & jnp.all(jnp.isfinite(entries.actions))
            & jnp.all(jnp.isfinite(entries.outcomes))
            & jnp.all(jnp.isfinite(entries.rewards))
            & jnp.all(jnp.isfinite(entries.uncertainties))
            & jnp.all(jnp.isfinite(entries.safety_costs))
            & jnp.all(jnp.isfinite(entries.reliabilities))
            & jnp.all(jnp.isfinite(entries.utilities))
        )
        scalar_ranges_valid = (
            jnp.all(entries.uncertainties >= 0.0)
            & jnp.all(entries.safety_costs >= 0.0)
            & jnp.all((entries.reliabilities >= 0.0) & (entries.reliabilities <= 1.0))
            & jnp.all(entries.utilities >= 0.0)
            & jnp.all(entries.ages >= 0)
            & jnp.all(entries.recency_ages >= 0)
            & jnp.all(entries.retrieval_counts >= 0)
        )
        availability_honest = (
            jnp.all(entries.uncertainty_available | (entries.uncertainties == 0.0))
            & jnp.all(entries.safety_cost_available | (entries.safety_costs == 0.0))
            & jnp.all(entries.utility_available | (entries.utilities == 0.0))
        )
        active_metadata_valid = jnp.all(
            (~valid)
            | (
                (entries.representation_versions >= 0)
                & (entries.provenance_ids >= 0)
                & (entries.source_ids >= 0)
            )
        )
        inactive_metadata_valid = jnp.all(
            valid
            | (
                (entries.representation_versions == -1)
                & (entries.provenance_ids == -1)
                & (entries.source_ids == -1)
                & (~entries.uncertainty_available)
                & (~entries.safety_cost_available)
                & (~entries.utility_available)
            )
        )
        active_count = jnp.sum(valid.astype(jnp.int32))
        counters_nonnegative = (
            (state.active_count >= 0)
            & (state.step_count >= 0)
            & (state.query_count >= 0)
            & (state.accepted_query_count >= 0)
            & (state.write_count >= 0)
            & (state.rejected_write_count >= 0)
            & (state.eviction_count >= 0)
        )
        counter_relations_valid = (
            (state.active_count == active_count)
            & (state.active_count <= self._config.capacity)
            & (state.accepted_query_count <= state.query_count)
            & (state.eviction_count <= state.write_count)
            & (state.write_count >= state.active_count)
        )
        return (
            all_float_payload_finite
            & scalar_ranges_valid
            & availability_honest
            & active_metadata_valid
            & inactive_metadata_valid
            & counters_nonnegative
            & counter_relations_valid
            & (
                state.persistent_bytes
                == jnp.asarray(self._persistent_bytes, dtype=jnp.uint32)
            )
        )

    def _make_initial_state(self, persistent_bytes: int) -> ExperientialMemoryState:
        cfg = self._config
        zeros = functools.partial(jnp.zeros, dtype=jnp.float32)
        entries = ExperientialMemoryEntries(
            observations=zeros((cfg.capacity, cfg.observation_dim)),
            keys=zeros((cfg.capacity, cfg.key_dim)),
            actions=zeros((cfg.capacity, cfg.action_dim)),
            outcomes=zeros((cfg.capacity, cfg.outcome_dim)),
            rewards=zeros((cfg.capacity,)),
            uncertainties=zeros((cfg.capacity,)),
            uncertainty_available=jnp.zeros((cfg.capacity,), dtype=jnp.bool_),
            safety_costs=zeros((cfg.capacity,)),
            safety_cost_available=jnp.zeros((cfg.capacity,), dtype=jnp.bool_),
            reliabilities=zeros((cfg.capacity,)),
            utilities=zeros((cfg.capacity,)),
            utility_available=jnp.zeros((cfg.capacity,), dtype=jnp.bool_),
            representation_versions=jnp.full((cfg.capacity,), -1, dtype=jnp.int32),
            valid=jnp.zeros((cfg.capacity,), dtype=jnp.bool_),
            ages=jnp.zeros((cfg.capacity,), dtype=jnp.int32),
            recency_ages=jnp.zeros((cfg.capacity,), dtype=jnp.int32),
            provenance_ids=jnp.full((cfg.capacity,), -1, dtype=jnp.int32),
            source_ids=jnp.full((cfg.capacity,), -1, dtype=jnp.int32),
            retrieval_counts=jnp.zeros((cfg.capacity,), dtype=jnp.int32),
        )
        zero = jnp.asarray(0, dtype=jnp.int32)
        return ExperientialMemoryState(
            entries=entries,
            active_count=zero,
            step_count=zero,
            query_count=zero,
            accepted_query_count=zero,
            write_count=zero,
            rejected_write_count=zero,
            eviction_count=zero,
            persistent_bytes=jnp.asarray(persistent_bytes, dtype=jnp.uint32),
        )

    def init(self) -> ExperientialMemoryState:
        """Return an empty, fixed-shape memory state."""
        state = self._make_initial_state(persistent_bytes=self._persistent_bytes)
        if _tree_nbytes(state) != self._persistent_bytes:
            raise RuntimeError("persistent byte accounting disagrees with allocated state")
        return state

    def _canonical_entry(self, entry: ExperientialMemoryEntry) -> ExperientialMemoryEntry:
        cfg = self._config
        return ExperientialMemoryEntry(
            observation=jnp.asarray(entry.observation, dtype=jnp.float32).reshape(
                (cfg.observation_dim,)
            ),
            key=jnp.asarray(entry.key, dtype=jnp.float32).reshape((cfg.key_dim,)),
            action=jnp.asarray(entry.action, dtype=jnp.float32).reshape((cfg.action_dim,)),
            outcome=jnp.asarray(entry.outcome, dtype=jnp.float32).reshape((cfg.outcome_dim,)),
            reward=jnp.asarray(entry.reward, dtype=jnp.float32).reshape(()),
            uncertainty=jnp.asarray(entry.uncertainty, dtype=jnp.float32).reshape(()),
            uncertainty_available=jnp.asarray(
                entry.uncertainty_available, dtype=jnp.bool_
            ).reshape(()),
            safety_cost=jnp.asarray(entry.safety_cost, dtype=jnp.float32).reshape(()),
            safety_cost_available=jnp.asarray(
                entry.safety_cost_available, dtype=jnp.bool_
            ).reshape(()),
            reliability=jnp.asarray(entry.reliability, dtype=jnp.float32).reshape(()),
            utility=jnp.asarray(entry.utility, dtype=jnp.float32).reshape(()),
            utility_available=jnp.asarray(entry.utility_available, dtype=jnp.bool_).reshape(()),
            representation_version=jnp.asarray(
                entry.representation_version, dtype=jnp.int32
            ).reshape(()),
            valid=jnp.asarray(entry.valid, dtype=jnp.bool_).reshape(()),
            age=jnp.asarray(entry.age, dtype=jnp.int32).reshape(()),
            provenance_id=jnp.asarray(entry.provenance_id, dtype=jnp.int32).reshape(()),
            source_id=jnp.asarray(entry.source_id, dtype=jnp.int32).reshape(()),
        )

    @staticmethod
    def _entry_is_valid(entry: ExperientialMemoryEntry) -> Array:
        finite_payload = (
            jnp.all(jnp.isfinite(entry.observation))
            & jnp.all(jnp.isfinite(entry.key))
            & jnp.all(jnp.isfinite(entry.action))
            & jnp.all(jnp.isfinite(entry.outcome))
            & jnp.isfinite(entry.reward)
            & jnp.isfinite(entry.uncertainty)
            & jnp.isfinite(entry.safety_cost)
            & jnp.isfinite(entry.reliability)
            & jnp.isfinite(entry.utility)
        )
        valid_metadata = (
            (entry.uncertainty >= 0.0)
            & (entry.uncertainty_available | (entry.uncertainty == 0.0))
            & (entry.safety_cost >= 0.0)
            & (entry.safety_cost_available | (entry.safety_cost == 0.0))
            & (entry.reliability >= 0.0)
            & (entry.reliability <= 1.0)
            & (entry.utility >= 0.0)
            & (entry.utility_available | (entry.utility == 0.0))
            & (entry.representation_version >= 0)
            & (entry.age >= 0)
            & (entry.provenance_id >= 0)
            & (entry.source_id >= 0)
        )
        return entry.valid & finite_payload & valid_metadata

    def query(
        self,
        state: ExperientialMemoryState,
        key: Float[Array, " key_dim"],
        representation_version: Int[Array, ""],
        query_uncertainty: Float[Array, ""],
        query_uncertainty_available: Bool[Array, ""],
    ) -> ExperientialMemoryRetrieval:
        """Retrieve an eligible neighborhood without mutating persistent state.

        ``query_uncertainty_available`` is explicit: an unavailable estimate is
        not interchangeable with a measured zero.  Such a query abstains.
        """
        self._validate_state_static_contract(state)
        _require_array(key, name="key", shape=(self._config.key_dim,), dtype=jnp.float32)
        _require_array(
            representation_version,
            name="representation_version",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            query_uncertainty,
            name="query_uncertainty",
            shape=(),
            dtype=jnp.float32,
        )
        _require_array(
            query_uncertainty_available,
            name="query_uncertainty_available",
            shape=(),
            dtype=jnp.bool_,
        )
        return cast(
            ExperientialMemoryRetrieval,
            self._query_jit(
                state,
                key,
                representation_version,
                query_uncertainty,
                query_uncertainty_available,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _query_jit(
        self,
        state: ExperientialMemoryState,
        key: Array,
        representation_version: Array,
        query_uncertainty: Array,
        query_uncertainty_available: Array,
    ) -> ExperientialMemoryRetrieval:
        """Compiled query after the public structural contract is established."""
        cfg = self._config
        entries = state.entries
        query_key = jnp.asarray(key, dtype=jnp.float32)
        query_version = jnp.asarray(representation_version, dtype=jnp.int32)
        query_uncertainty_value = jnp.asarray(query_uncertainty, dtype=jnp.float32)
        query_uncertainty_is_available = jnp.asarray(
            query_uncertainty_available, dtype=jnp.bool_
        )
        state_valid = self._state_is_valid(state)

        query_valid = (
            jnp.all(jnp.isfinite(query_key))
            & jnp.isfinite(query_uncertainty_value)
            & (query_uncertainty_value >= 0.0)
            & (query_version >= 0)
            & query_uncertainty_is_available
        )

        finite_rows = (
            jnp.all(jnp.isfinite(entries.observations), axis=1)
            & jnp.all(jnp.isfinite(entries.keys), axis=1)
            & jnp.all(jnp.isfinite(entries.actions), axis=1)
            & jnp.all(jnp.isfinite(entries.outcomes), axis=1)
            & jnp.isfinite(entries.rewards)
            & jnp.isfinite(entries.uncertainties)
            & jnp.isfinite(entries.safety_costs)
            & jnp.isfinite(entries.reliabilities)
            & jnp.isfinite(entries.utilities)
        )
        sane_rows = (
            entries.valid
            & finite_rows
            & (entries.uncertainties >= 0.0)
            & (entries.safety_costs >= 0.0)
            & (entries.reliabilities >= 0.0)
            & (entries.reliabilities <= 1.0)
            & (entries.utilities >= 0.0)
            & (entries.representation_versions >= 0)
            & (entries.ages >= 0)
            & (entries.recency_ages >= 0)
            & (entries.provenance_ids >= 0)
            & (entries.source_ids >= 0)
            & (entries.retrieval_counts >= 0)
        )
        same_version = sane_rows & (entries.representation_versions == query_version)
        fresh = same_version & (entries.ages <= cfg.max_age)
        uncertainty_available = fresh & entries.uncertainty_available
        safety_cost_available = fresh & entries.safety_cost_available
        uncertainty_eligible = uncertainty_available & (
            entries.uncertainties <= cfg.max_uncertainty
        )
        safety_eligible = safety_cost_available & (
            entries.safety_costs <= cfg.max_safety_cost
        )

        safe_query_key = jnp.where(jnp.isfinite(query_key), query_key, 0.0)
        safe_keys = jnp.where(finite_rows[:, None], entries.keys, 0.0)
        squared_distance = jnp.mean(
            (safe_keys - safe_query_key[None, :]) ** 2,
            axis=1,
        )
        similarities = jnp.exp(
            -squared_distance / jnp.asarray(cfg.distance_scale, dtype=jnp.float32)
        )
        safe_ages = jnp.where(entries.ages >= 0, entries.ages, 0)
        staleness_scale = jnp.asarray(cfg.staleness_scale, dtype=jnp.float32)
        # Every positive float32 scale is legal. Cap the mathematically huge
        # quotient before division so a tiny scale yields exact zero staleness
        # without first creating an infinity in the JAX operation graph.
        maximum_exponent = jnp.asarray(
            np.finfo(np.float32).max / 2.0,
            dtype=jnp.float32,
        )
        bounded_staleness_age = jnp.minimum(
            safe_ages.astype(jnp.float32),
            maximum_exponent * jnp.minimum(staleness_scale, 1.0),
        )
        staleness = jnp.exp(-bounded_staleness_age / staleness_scale)
        safe_reliabilities = jnp.where(
            jnp.isfinite(entries.reliabilities)
            & (entries.reliabilities >= 0.0)
            & (entries.reliabilities <= 1.0),
            entries.reliabilities,
            0.0,
        )
        effective_reliabilities = safe_reliabilities * staleness
        eligible = (
            uncertainty_eligible
            & safety_eligible
            & (similarities >= cfg.min_similarity)
            & (effective_reliabilities >= cfg.min_effective_reliability)
        )
        scores = jnp.where(eligible, similarities * effective_reliabilities, -jnp.inf)
        top_scores, indices = jax.lax.top_k(scores, cfg.top_k)
        neighbor_mask = jnp.isfinite(top_scores) & (top_scores > 0.0)
        positive_scores = jnp.where(neighbor_mask, top_scores, 0.0)
        score_sum = jnp.sum(positive_scores)
        weight_denominator = jnp.where(score_sum > 0.0, score_sum, 1.0)
        neighbor_weights = positive_scores / weight_denominator

        neighbor_similarities = similarities[indices]
        neighbor_reliabilities = effective_reliabilities[indices]
        neighbor_ages = entries.ages[indices]
        neighbor_provenance_ids = entries.provenance_ids[indices]

        safe_observations = jnp.where(
            jnp.isfinite(entries.observations), entries.observations, 0.0
        )
        safe_actions = jnp.where(jnp.isfinite(entries.actions), entries.actions, 0.0)
        safe_outcomes = jnp.where(jnp.isfinite(entries.outcomes), entries.outcomes, 0.0)
        safe_rewards = jnp.where(jnp.isfinite(entries.rewards), entries.rewards, 0.0)
        safe_uncertainties = jnp.where(
            jnp.isfinite(entries.uncertainties), entries.uncertainties, 0.0
        )
        safe_safety_costs = jnp.where(
            jnp.isfinite(entries.safety_costs), entries.safety_costs, 0.0
        )
        weighted_observation = jnp.sum(
            neighbor_weights[:, None] * safe_observations[indices], axis=0
        )
        weighted_action = jnp.sum(
            neighbor_weights[:, None] * safe_actions[indices], axis=0
        )
        weighted_outcome = jnp.sum(
            neighbor_weights[:, None] * safe_outcomes[indices], axis=0
        )
        weighted_reward = jnp.sum(neighbor_weights * safe_rewards[indices])
        weighted_uncertainty = jnp.sum(
            neighbor_weights * safe_uncertainties[indices]
        )
        weighted_safety_cost = jnp.sum(
            neighbor_weights * safe_safety_costs[indices]
        )
        weighted_reliability = jnp.sum(neighbor_weights * effective_reliabilities[indices])

        has_neighbors = jnp.sum(neighbor_mask.astype(jnp.int32)) >= cfg.min_neighbors
        version_compatible = jnp.any(same_version)
        freshness_ok = jnp.any(fresh)
        uncertainty_is_available = (
            state_valid
            & query_uncertainty_is_available
            & jnp.any(uncertainty_available)
        )
        safety_is_available = state_valid & jnp.any(safety_cost_available)
        uncertainty_ok = (
            uncertainty_is_available
            & (query_uncertainty_value <= cfg.max_uncertainty)
            & jnp.any(uncertainty_eligible)
        )
        safety_ok = jnp.any(safety_eligible)
        aggregate_ok = (
            (weighted_uncertainty <= cfg.max_uncertainty)
            & (weighted_safety_cost <= cfg.max_safety_cost)
            & jnp.isfinite(weighted_reliability)
        )
        accepted = (
            state_valid
            & query_valid
            & version_compatible
            & freshness_ok
            & uncertainty_ok
            & safety_ok
            & has_neighbors
            & aggregate_ok
        )

        def gated(value: Array) -> Array:
            return jnp.where(accepted, value, jnp.zeros_like(value))

        return ExperientialMemoryRetrieval(
            accepted=accepted,
            observation=gated(weighted_observation),
            action=gated(weighted_action),
            outcome=gated(weighted_outcome),
            reward=gated(weighted_reward),
            uncertainty=gated(weighted_uncertainty),
            safety_cost=gated(weighted_safety_cost),
            effective_reliability=gated(weighted_reliability),
            neighbor_indices=indices.astype(jnp.int32),
            neighbor_mask=neighbor_mask,
            neighbor_weights=neighbor_weights,
            neighbor_similarities=neighbor_similarities,
            neighbor_reliabilities=neighbor_reliabilities,
            neighbor_ages=neighbor_ages,
            neighbor_provenance_ids=neighbor_provenance_ids,
            state_valid=state_valid,
            query_valid=query_valid,
            version_compatible=version_compatible,
            freshness_ok=freshness_ok,
            uncertainty_available=uncertainty_is_available,
            safety_cost_available=safety_is_available,
            uncertainty_ok=uncertainty_ok,
            safety_ok=safety_ok,
            has_neighbors=has_neighbors,
        )

    def _advance(self, state: ExperientialMemoryState) -> ExperientialMemoryState:
        entries = state.entries
        valid = entries.valid
        ages = jnp.where(valid, _saturating_increment(entries.ages), entries.ages)
        recency_ages = jnp.where(
            valid,
            _saturating_increment(entries.recency_ages),
            entries.recency_ages,
        )
        utilities = jnp.where(
            valid,
            entries.utilities
            * jnp.asarray(self._config.utility_decay, dtype=jnp.float32),
            entries.utilities,
        )
        return ExperientialMemoryState(
            entries=ExperientialMemoryEntries(
                observations=entries.observations,
                keys=entries.keys,
                actions=entries.actions,
                outcomes=entries.outcomes,
                rewards=entries.rewards,
                uncertainties=entries.uncertainties,
                uncertainty_available=entries.uncertainty_available,
                safety_costs=entries.safety_costs,
                safety_cost_available=entries.safety_cost_available,
                reliabilities=entries.reliabilities,
                utilities=utilities,
                utility_available=entries.utility_available,
                representation_versions=entries.representation_versions,
                valid=entries.valid,
                ages=ages,
                recency_ages=recency_ages,
                provenance_ids=entries.provenance_ids,
                source_ids=entries.source_ids,
                retrieval_counts=entries.retrieval_counts,
            ),
            active_count=state.active_count,
            step_count=_saturating_increment(state.step_count),
            query_count=state.query_count,
            accepted_query_count=state.accepted_query_count,
            write_count=state.write_count,
            rejected_write_count=state.rejected_write_count,
            eviction_count=state.eviction_count,
            persistent_bytes=state.persistent_bytes,
        )

    @staticmethod
    def _record_query(
        state: ExperientialMemoryState,
        retrieval: ExperientialMemoryRetrieval,
    ) -> ExperientialMemoryState:
        entries = state.entries
        access_increments = (
            jnp.zeros_like(entries.retrieval_counts)
            .at[retrieval.neighbor_indices]
            .add(retrieval.neighbor_mask.astype(jnp.int32))
        )
        access_mask = access_increments > 0
        accepted_access = retrieval.accepted & access_mask
        retrieval_counts = jnp.where(
            accepted_access,
            _saturating_increment(entries.retrieval_counts),
            entries.retrieval_counts,
        )
        recency_ages = jnp.where(accepted_access, 0, entries.recency_ages)
        return ExperientialMemoryState(
            entries=ExperientialMemoryEntries(
                observations=entries.observations,
                keys=entries.keys,
                actions=entries.actions,
                outcomes=entries.outcomes,
                rewards=entries.rewards,
                uncertainties=entries.uncertainties,
                uncertainty_available=entries.uncertainty_available,
                safety_costs=entries.safety_costs,
                safety_cost_available=entries.safety_cost_available,
                reliabilities=entries.reliabilities,
                utilities=entries.utilities,
                utility_available=entries.utility_available,
                representation_versions=entries.representation_versions,
                valid=entries.valid,
                ages=entries.ages,
                recency_ages=recency_ages,
                provenance_ids=entries.provenance_ids,
                source_ids=entries.source_ids,
                retrieval_counts=retrieval_counts,
            ),
            active_count=state.active_count,
            step_count=state.step_count,
            query_count=_saturating_increment(state.query_count),
            accepted_query_count=jnp.where(
                retrieval.accepted,
                _saturating_increment(state.accepted_query_count),
                state.accepted_query_count,
            ),
            write_count=state.write_count,
            rejected_write_count=state.rejected_write_count,
            eviction_count=state.eviction_count,
            persistent_bytes=state.persistent_bytes,
        )

    def _write_advanced(
        self,
        state: ExperientialMemoryState,
        raw_entry: ExperientialMemoryEntry,
    ) -> ExperientialMemoryWriteResult:
        entry = self._canonical_entry(raw_entry)
        can_write = self._entry_is_valid(entry)
        cfg = self._config

        def do_write(
            current: ExperientialMemoryState,
        ) -> tuple[ExperientialMemoryState, Array, Array, Array]:
            current_entries = current.entries
            has_empty = jnp.any(~current_entries.valid)
            empty_slot = jnp.argmax((~current_entries.valid).astype(jnp.int32))
            recency_scale = jnp.asarray(cfg.recency_scale, dtype=jnp.float32)
            # Algebraically equivalent to 1 / (1 + age / scale), but remains
            # finite for every positive float32 scale in the public contract.
            recency_score = recency_scale / (
                recency_scale + current_entries.recency_ages.astype(jnp.float32)
            )
            eviction_utilities = jnp.where(
                current_entries.utility_available,
                current_entries.utilities,
                0.0,
            )
            utility_weight = jnp.asarray(cfg.eviction_utility_weight, dtype=jnp.float32)
            recency_weight = jnp.asarray(cfg.eviction_recency_weight, dtype=jnp.float32)
            # These weights are relative. Normalization preserves ordering and
            # prevents finite configured weights from overflowing their products.
            weight_scale = jnp.maximum(utility_weight, recency_weight)
            retention_score = (
                utility_weight / weight_scale * eviction_utilities
                + recency_weight / weight_scale * recency_score
            )
            retention_score = jnp.where(current_entries.valid, retention_score, jnp.inf)
            eviction_slot = jnp.argmin(retention_score)
            slot = jnp.where(has_empty, empty_slot, eviction_slot).astype(jnp.int32)
            evicted = ~has_empty
            evicted_provenance_id = jnp.where(
                evicted, current_entries.provenance_ids[slot], -1
            ).astype(jnp.int32)

            next_entries = ExperientialMemoryEntries(
                observations=current_entries.observations.at[slot].set(entry.observation),
                keys=current_entries.keys.at[slot].set(entry.key),
                actions=current_entries.actions.at[slot].set(entry.action),
                outcomes=current_entries.outcomes.at[slot].set(entry.outcome),
                rewards=current_entries.rewards.at[slot].set(entry.reward),
                uncertainties=current_entries.uncertainties.at[slot].set(entry.uncertainty),
                uncertainty_available=current_entries.uncertainty_available.at[slot].set(
                    entry.uncertainty_available
                ),
                safety_costs=current_entries.safety_costs.at[slot].set(entry.safety_cost),
                safety_cost_available=current_entries.safety_cost_available.at[slot].set(
                    entry.safety_cost_available
                ),
                reliabilities=current_entries.reliabilities.at[slot].set(entry.reliability),
                utilities=current_entries.utilities.at[slot].set(entry.utility),
                utility_available=current_entries.utility_available.at[slot].set(
                    entry.utility_available
                ),
                representation_versions=current_entries.representation_versions.at[slot].set(
                    entry.representation_version
                ),
                valid=current_entries.valid.at[slot].set(True),
                ages=current_entries.ages.at[slot].set(entry.age),
                # Recency is time since this memory last retrieved the
                # exemplar, not the exemplar's imported chronological age.
                recency_ages=current_entries.recency_ages.at[slot].set(0),
                provenance_ids=current_entries.provenance_ids.at[slot].set(entry.provenance_id),
                source_ids=current_entries.source_ids.at[slot].set(entry.source_id),
                retrieval_counts=current_entries.retrieval_counts.at[slot].set(0),
            )
            next_state = ExperientialMemoryState(
                entries=next_entries,
                active_count=jnp.where(
                    has_empty,
                    _saturating_increment(current.active_count),
                    current.active_count,
                ),
                step_count=current.step_count,
                query_count=current.query_count,
                accepted_query_count=current.accepted_query_count,
                write_count=_saturating_increment(current.write_count),
                rejected_write_count=current.rejected_write_count,
                eviction_count=jnp.where(
                    evicted,
                    _saturating_increment(current.eviction_count),
                    current.eviction_count,
                ),
                persistent_bytes=current.persistent_bytes,
            )
            return next_state, slot, evicted, evicted_provenance_id

        def reject_write(
            current: ExperientialMemoryState,
        ) -> tuple[ExperientialMemoryState, Array, Array, Array]:
            rejected = ExperientialMemoryState(
                entries=current.entries,
                active_count=current.active_count,
                step_count=current.step_count,
                query_count=current.query_count,
                accepted_query_count=current.accepted_query_count,
                write_count=current.write_count,
                rejected_write_count=_saturating_increment(current.rejected_write_count),
                eviction_count=current.eviction_count,
                persistent_bytes=current.persistent_bytes,
            )
            return (
                rejected,
                jnp.asarray(-1, dtype=jnp.int32),
                jnp.asarray(False),
                jnp.asarray(-1, dtype=jnp.int32),
            )

        next_state, slot, evicted, evicted_provenance_id = jax.lax.cond(
            can_write, do_write, reject_write, state
        )
        return ExperientialMemoryWriteResult(
            state=next_state,
            wrote=can_write,
            slot=slot,
            evicted=evicted,
            evicted_provenance_id=evicted_provenance_id,
        )

    def write(
        self,
        state: ExperientialMemoryState,
        entry: ExperientialMemoryEntry,
    ) -> ExperientialMemoryWriteResult:
        """Advance time once and attempt one bounded exemplar write.

        A dynamically corrupt state is an exact no-op rather than a substrate
        into which new data is partially written.
        """
        self._validate_state_static_contract(state)
        self._validate_entry_static_contract(entry)
        return cast(ExperientialMemoryWriteResult, self._write_jit(state, entry))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _write_jit(
        self,
        state: ExperientialMemoryState,
        entry: ExperientialMemoryEntry,
    ) -> ExperientialMemoryWriteResult:
        def apply(_: None) -> ExperientialMemoryWriteResult:
            return self._write_advanced(self._advance(state), entry)

        def reject(_: None) -> ExperientialMemoryWriteResult:
            return ExperientialMemoryWriteResult(
                state=state,
                wrote=jnp.asarray(False),
                slot=jnp.asarray(-1, dtype=jnp.int32),
                evicted=jnp.asarray(False),
                evicted_provenance_id=jnp.asarray(-1, dtype=jnp.int32),
            )

        return cast(
            ExperientialMemoryWriteResult,
            jax.lax.cond(self._state_is_valid(state), apply, reject, operand=None),
        )

    def step(
        self,
        state: ExperientialMemoryState,
        query_key: Float[Array, " key_dim"],
        representation_version: Int[Array, ""],
        query_uncertainty: Float[Array, ""],
        query_uncertainty_available: Bool[Array, ""],
        entry: ExperientialMemoryEntry,
    ) -> ExperientialMemoryStepResult:
        """Query the pre-write state, then age/access/write exactly once."""
        self._validate_state_static_contract(state)
        self._validate_entry_static_contract(entry)
        _require_array(
            query_key,
            name="query_key",
            shape=(self._config.key_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            representation_version,
            name="representation_version",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            query_uncertainty,
            name="query_uncertainty",
            shape=(),
            dtype=jnp.float32,
        )
        _require_array(
            query_uncertainty_available,
            name="query_uncertainty_available",
            shape=(),
            dtype=jnp.bool_,
        )
        return cast(
            ExperientialMemoryStepResult,
            self._step_jit(
                state,
                query_key,
                representation_version,
                query_uncertainty,
                query_uncertainty_available,
                entry,
            ),
        )

    def _step_from_retrieval(
        self,
        state: ExperientialMemoryState,
        retrieval: ExperientialMemoryRetrieval,
        entry: ExperientialMemoryEntry,
    ) -> ExperientialMemoryStepResult:
        """Age, record one already-computed pre-state query, and write once."""

        def apply(_: None) -> ExperientialMemoryStepResult:
            advanced = self._advance(state)
            accessed = self._record_query(advanced, retrieval)
            write_result = self._write_advanced(accessed, entry)
            return ExperientialMemoryStepResult(
                state=write_result.state,
                retrieval=retrieval,
                wrote=write_result.wrote,
                slot=write_result.slot,
                evicted=write_result.evicted,
                evicted_provenance_id=write_result.evicted_provenance_id,
            )

        def reject(_: None) -> ExperientialMemoryStepResult:
            return ExperientialMemoryStepResult(
                state=state,
                retrieval=retrieval,
                wrote=jnp.asarray(False),
                slot=jnp.asarray(-1, dtype=jnp.int32),
                evicted=jnp.asarray(False),
                evicted_provenance_id=jnp.asarray(-1, dtype=jnp.int32),
            )

        return cast(
            ExperientialMemoryStepResult,
            jax.lax.cond(self._state_is_valid(state), apply, reject, operand=None),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_jit(
        self,
        state: ExperientialMemoryState,
        query_key: Array,
        representation_version: Array,
        query_uncertainty: Array,
        query_uncertainty_available: Array,
        entry: ExperientialMemoryEntry,
    ) -> ExperientialMemoryStepResult:
        retrieval = self._query_jit(
            state,
            query_key,
            representation_version,
            query_uncertainty,
            query_uncertainty_available,
        )
        return self._step_from_retrieval(state, retrieval, entry)

    @functools.partial(jax.jit, static_argnums=(0,))
    def accounting(
        self,
        state: ExperientialMemoryState,
    ) -> ExperientialMemoryAccounting:
        """Return exact capacity, byte, and lifetime-operation accounting."""
        return ExperientialMemoryAccounting(
            active_entries=state.active_count,
            capacity_entries=jnp.asarray(self._config.capacity, dtype=jnp.int32),
            slot_bytes=jnp.asarray(self._slot_bytes, dtype=jnp.uint32),
            persistent_bytes=state.persistent_bytes,
            queries=state.query_count,
            accepted_queries=state.accepted_query_count,
            writes=state.write_count,
            rejected_writes=state.rejected_write_count,
            evictions=state.eviction_count,
        )

__all__ = [
    "ExperientialMemory",
    "ExperientialMemoryAccounting",
    "ExperientialMemoryConfig",
    "ExperientialMemoryEntries",
    "ExperientialMemoryEntry",
    "ExperientialMemoryRetrieval",
    "ExperientialMemoryState",
    "ExperientialMemoryStepResult",
    "ExperientialMemoryWriteResult",
]
