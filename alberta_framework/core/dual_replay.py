# mypy: disable-error-code="attr-defined,call-arg"
"""Fixed-budget dual replay memory for mechanism development.

This module is a mechanism-only replay substrate.  It does not train a model,
critic, or actor and it carries no scientific-evidence claim.  A configured
total slot budget is split between a short-term FIFO and a long-term memory.
The long-term policy is static: either exact reservoir sampling (Algorithm R,
Vitter 1985) or a ``"calibrated"`` priority policy — a weight-normalized convex
combination of scale-normalized epistemic surprise, observation-coverage
novelty (distance to the nearest retained observation), and learning progress.
The calibration itself lives with the caller: scales, weights, and the
admission threshold are per-protocol inputs, and the defaults (unit scales,
equal weights, midpoint threshold) are neutral placeholders, not tuned values.

The write boundary is deliberately causal.  :meth:`DualReplayMemory.record`
accepts a :class:`ReplayPrediction` that must have been produced before the
outcome and a separate :class:`ReplayOutcome`; only then is a transition
assembled and atomically written.  Optional signals always have explicit
availability flags.  Missing estimates are never interpreted as measured zero,
and no raw prediction-error field participates in retention priority.

All persistent state uses fixed-shape JAX arrays.  Record and sample operations
are compatible with eager execution, ``jax.jit``, and ``jax.lax.scan``.  Invalid
dynamic state and exhausted counters fail closed without changing persistent
state.  Invalid transitions only advance the bounded attempt/rejection counters;
neither replay stratum nor the PRNG is partially updated.

Reference:
    Vitter, J. S. (1985). "Random Sampling with a Reservoir."
    ACM Transactions on Mathematical Software 11(1), 37-57.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import operator
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Literal, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.typing as npt
from jax import Array

from alberta_framework._float32 import round_real_to_float32

DUAL_REPLAY_CONFIG_SCHEMA = "alberta.dual-replay.config.v1"
DUAL_REPLAY_CHECKPOINT_SCHEMA = "alberta.dual-replay.checkpoint.v1"
MECHANISM_STATUS = "mechanism-only-no-training-integration"

LongTermPolicy = Literal["reservoir", "calibrated"]
AleatoricControl = Literal["veto", "downweight"]

SHORT_TERM_STRATUM = 0
LONG_TERM_STRATUM = 1

_INT32_MAX = 2_147_483_647
_UINT32_MAX = 4_294_967_295
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


@dataclass(frozen=True)
class DualReplayConfig:
    """Static allocation, retention, and sampling configuration.

    ``total_capacity - short_term_capacity`` is the long-term capacity.  Both
    strata must contain at least one slot.  Calibrated retention uses all three
    positive-weight components and requires explicit epistemic, learning-
    progress, and aleatoric availability.

    Calibrated-retention fields: each ``*_scale`` divides its raw signal before
    clipping to ``[0, 1]``, so it encodes the magnitude at which that signal
    saturates; the ``*_weight`` values form a convex combination, so the
    resulting priority also lies in ``[0, 1]``.
    ``calibrated_priority_threshold`` is the admission floor on that priority
    (default ``0.5`` — the midpoint of the attainable range), and
    ``calibrated_replacement_margin`` is the extra priority a candidate must
    carry over the lowest-priority resident once the stratum is full.  The
    defaults are neutral placeholders; protocols supply their own calibrated
    values.
    """

    total_capacity: int
    short_term_capacity: int
    observation_dim: int
    action_dim: int
    short_term_sample_size: int
    long_term_sample_size: int
    long_term_policy: LongTermPolicy = "reservoir"
    max_representation_lag: int = 0
    surprise_scale: float = 1.0
    coverage_scale: float = 1.0
    progress_scale: float = 1.0
    surprise_weight: float = 1.0
    coverage_weight: float = 1.0
    progress_weight: float = 1.0
    calibrated_priority_threshold: float = 0.5
    calibrated_replacement_margin: float = 0.0
    aleatoric_control: AleatoricControl = "veto"
    max_aleatoric_uncertainty: float = 1.0
    aleatoric_downweight_scale: float = 1.0
    max_persistent_bytes: int | None = None

    def __post_init__(self) -> None:
        """Validate float32-consumed scalars in their sink domain and canonicalize them."""
        for name in _POSITIVE_FLOAT32_FIELDS:
            object.__setattr__(
                self,
                name,
                _require_float32_real(name, getattr(self, name), strictly_positive=True),
            )
        object.__setattr__(
            self,
            "calibrated_priority_threshold",
            _require_float32_real(
                "calibrated_priority_threshold",
                self.calibrated_priority_threshold,
                strictly_positive=False,
                maximum=1,
            ),
        )
        for name in ("calibrated_replacement_margin", "max_aleatoric_uncertainty"):
            object.__setattr__(
                self,
                name,
                _require_float32_real(name, getattr(self, name), strictly_positive=False),
            )
        _validate_config(self)

    @property
    def long_term_capacity(self) -> int:
        """Number of long-term slots under the fixed total budget."""
        return self.total_capacity - self.short_term_capacity

    @property
    def batch_size(self) -> int:
        """Fixed number of sample positions, including unavailable padding."""
        return self.short_term_sample_size + self.long_term_sample_size

    def to_config(self) -> dict[str, object]:
        """Return a strict, versioned JSON-compatible configuration."""
        return {
            "schema": DUAL_REPLAY_CONFIG_SCHEMA,
            "type": "DualReplayConfig",
            "mechanism_status": MECHANISM_STATUS,
            **asdict(self),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, Any]) -> DualReplayConfig:
        """Reconstruct and validate an exact configuration payload."""
        values = _copy_mapping(payload, name="dual replay config")
        expected = {
            "schema",
            "type",
            "mechanism_status",
            *asdict(
                cls(
                    total_capacity=2,
                    short_term_capacity=1,
                    observation_dim=1,
                    action_dim=1,
                    short_term_sample_size=1,
                    long_term_sample_size=1,
                )
            ),
        }
        if set(values) != expected:
            raise ValueError("dual replay config fields do not match the v1 schema")
        if type(values["schema"]) is not str or values["schema"] != DUAL_REPLAY_CONFIG_SCHEMA:
            raise ValueError("unexpected dual replay config schema")
        if type(values["type"]) is not str or values["type"] != "DualReplayConfig":
            raise ValueError("unexpected dual replay config type")
        if (
            type(values["mechanism_status"]) is not str
            or values["mechanism_status"] != MECHANISM_STATUS
        ):
            raise ValueError("dual replay config must remain mechanism-only")
        for name in ("schema", "type", "mechanism_status"):
            values.pop(name)
        return cls(**values)


@chex.dataclass(frozen=True)
class ReplayPrediction:
    """Information available before observing the transition outcome."""

    observation: Array
    action: Array
    old_behavior_probability: Array
    old_behavior_probability_available: Array
    old_behavior_logit: Array
    old_behavior_logit_available: Array
    old_value_target: Array
    old_value_target_available: Array
    epistemic_surprise: Array
    epistemic_surprise_available: Array
    aleatoric_uncertainty: Array
    aleatoric_uncertainty_available: Array
    representation_version: Array
    provenance_id: Array
    source_id: Array
    valid: Array


@chex.dataclass(frozen=True)
class ReplayOutcome:
    """Information revealed only after the prediction/action boundary.

    ``next_observation`` is the successor observation used as the transition's
    model target, including the final pre-reset observation on a time-limit
    truncation.  A post-reset decision observation belongs to the next real
    event and is intentionally not stored in this transition.  ``discount`` is
    zero exactly at terminal boundaries.  ``truncated`` is independent: a
    truncation-only boundary keeps a positive bootstrap discount, while a
    simultaneous terminal and truncation boundary has zero discount.
    """

    next_observation: Array
    reward: Array
    discount: Array
    terminated: Array
    truncated: Array
    learning_progress: Array
    learning_progress_available: Array
    safety_cost: Array
    safety_cost_available: Array
    valid: Array


@chex.dataclass(frozen=True)
class ReplayTransition:
    """One fixed-shape transition after joining prediction and outcome."""

    observation: Array
    next_observation: Array
    action: Array
    reward: Array
    discount: Array
    terminated: Array
    truncated: Array
    old_behavior_probability: Array
    old_behavior_probability_available: Array
    old_behavior_logit: Array
    old_behavior_logit_available: Array
    old_value_target: Array
    old_value_target_available: Array
    epistemic_surprise: Array
    epistemic_surprise_available: Array
    aleatoric_uncertainty: Array
    aleatoric_uncertainty_available: Array
    learning_progress: Array
    learning_progress_available: Array
    safety_cost: Array
    safety_cost_available: Array
    representation_version: Array
    provenance_id: Array
    source_id: Array
    valid: Array


@chex.dataclass(frozen=True)
class ReplayEntries:
    """Fixed-capacity structure-of-arrays used by either replay stratum."""

    observations: Array
    next_observations: Array
    actions: Array
    rewards: Array
    discounts: Array
    terminated: Array
    truncated: Array
    old_behavior_probabilities: Array
    old_behavior_probability_available: Array
    old_behavior_logits: Array
    old_behavior_logit_available: Array
    old_value_targets: Array
    old_value_target_available: Array
    epistemic_surprises: Array
    epistemic_surprise_available: Array
    aleatoric_uncertainties: Array
    aleatoric_uncertainty_available: Array
    learning_progress: Array
    learning_progress_available: Array
    safety_costs: Array
    safety_cost_available: Array
    representation_versions: Array
    provenance_ids: Array
    source_ids: Array
    eviction_provenance_ids: Array
    insertion_priorities: Array
    insertion_priority_available: Array
    valid: Array


@chex.dataclass(frozen=True)
class DualReplayState:
    """Complete fixed-shape dual replay state, including deterministic PRNG."""

    short_term: ReplayEntries
    long_term: ReplayEntries
    rng_key_data: Array
    short_term_size: Array
    short_term_head: Array
    long_term_size: Array
    write_attempt_count: Array
    accepted_transition_count: Array
    rejected_transition_count: Array
    long_term_candidate_count: Array
    long_term_write_count: Array
    short_term_eviction_count: Array
    long_term_eviction_count: Array
    long_term_rejection_count: Array
    sample_count: Array
    persistent_bytes: Array


@chex.dataclass(frozen=True)
class ReservoirSelection:
    """Exact Algorithm-R (Vitter 1985) selection calculation for one one-based candidate."""

    selected: Array
    slot: Array
    candidate_number: Array
    population_size: Array
    draw: Array
    draw_available: Array
    capacity: Array


@chex.dataclass(frozen=True)
class ReplayWriteResult:
    """Atomic dual-write result and auditable long-term selection decision."""

    state: DualReplayState
    state_valid: Array
    input_valid: Array
    counter_available: Array
    wrote_short_term: Array
    short_term_slot: Array
    short_term_evicted: Array
    short_term_evicted_provenance_id: Array
    long_term_candidate_number: Array
    long_term_priority: Array
    long_term_priority_available: Array
    surprise_component: Array
    coverage_component: Array
    progress_component: Array
    aleatoric_control_passed: Array
    aleatoric_control_available: Array
    reservoir_draw: Array
    reservoir_draw_available: Array
    reservoir_population_size: Array
    wrote_long_term: Array
    long_term_slot: Array
    long_term_evicted: Array
    long_term_evicted_provenance_id: Array


@chex.dataclass(frozen=True)
class ReplayBatch:
    """Fixed-size stratified batch; ``valid`` marks non-padding positions."""

    entries: ReplayEntries
    valid: Array
    stratum: Array
    slot: Array
    representation_lag: Array


@chex.dataclass(frozen=True)
class ReplaySampleResult:
    """Read result with explicit compatibility and availability diagnostics."""

    state: DualReplayState
    batch: ReplayBatch
    state_valid: Array
    representation_version_valid: Array
    counter_available: Array
    stale_detected: Array
    stale_short_term_count: Array
    stale_long_term_count: Array
    future_short_term_count: Array
    future_long_term_count: Array
    eligible_short_term_count: Array
    eligible_long_term_count: Array


@chex.dataclass(frozen=True)
class DualReplayResourceAccounting:
    """Exact fixed allocation plus lifetime bounded-operation counts."""

    total_capacity: Array
    short_term_capacity: Array
    long_term_capacity: Array
    active_entries: Array
    short_term_entries: Array
    long_term_entries: Array
    slot_bytes: Array
    persistent_bytes: Array
    write_attempts: Array
    accepted_transitions: Array
    rejected_transitions: Array
    long_term_candidates: Array
    long_term_writes: Array
    short_term_evictions: Array
    long_term_evictions: Array
    long_term_rejections: Array
    samples: Array


_POSITIVE_FLOAT32_FIELDS = (
    "surprise_scale",
    "coverage_scale",
    "progress_scale",
    "surprise_weight",
    "coverage_weight",
    "progress_weight",
    "aleatoric_downweight_scale",
)


def _require_float32_real(
    name: str,
    value: object,
    *,
    strictly_positive: bool,
    maximum: int | None = None,
) -> float:
    """Validate one host scalar in its exact domain and its float32 sink.

    Zero is only admitted where the exact domain allows it; a nonzero value
    that narrows to zero, or a finite value that narrows to infinity, is
    rejected because the compiled sink would silently run a different rule.
    Built-in payloads that already narrow exactly are preserved for
    serialization; other reals are canonicalized to the exact sink value.
    """
    domain = "positive" if strictly_positive else "non-negative"
    if maximum is not None:
        domain = f"{domain} and at most {maximum}"
    preserve_builtin_payload = type(value) is int or type(value) is float
    actual_type = type(value)
    if issubclass(actual_type, bool | np.bool_) or not issubclass(actual_type, Real):
        raise ValueError(f"{name} must be finite and {domain}")
    real_value = cast(Any, value)

    def in_domain(candidate: Any) -> bool:
        if strictly_positive:
            if not candidate > 0:
                return False
        elif not candidate >= 0:
            return False
        return maximum is None or bool(candidate <= maximum)

    try:
        if not in_domain(real_value):
            raise ValueError
        narrowed = round_real_to_float32(real_value)
    except Exception as error:
        raise ValueError(f"{name} must be finite and {domain}") from error
    if not math.isfinite(narrowed):
        raise ValueError(f"{name} must narrow to a finite float32")
    try:
        exact_nonzero = bool(real_value != 0)
    except Exception as error:
        raise ValueError(f"{name} must be finite and {domain}") from error
    if exact_nonzero and narrowed == 0.0:
        raise ValueError(f"{name} must not underflow to zero in float32")
    if not in_domain(narrowed):
        raise ValueError(f"{name} must remain {domain} once narrowed to float32")
    if not preserve_builtin_payload:
        return narrowed
    number = float(real_value)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        renarrowed = float(np.float32(number))
    return cast(float, value) if narrowed == renarrowed else narrowed


def _validate_positive_int(
    name: str,
    value: object,
    *,
    minimum: int = 1,
    maximum: int = _INT32_MAX,
) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _copy_mapping(payload: object, *, name: str) -> dict[str, Any]:
    """Copy one supported mapping while normalizing hostile hooks."""
    if not issubclass(type(payload), Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        return dict(cast(Mapping[str, Any], payload))
    except Exception as error:
        raise ValueError(f"{name} must be a readable mapping") from error


def _allocation_sizes(config: DualReplayConfig) -> tuple[int, int]:
    """Return exact slot and persistent bytes without allocating JAX arrays."""
    slot_bytes = 4 * (2 * config.observation_dim + config.action_dim + 14) + 11
    persistent_bytes = config.total_capacity * slot_bytes + 60
    if persistent_bytes > _UINT32_MAX:
        raise ValueError("dual replay allocation exceeds uint32 byte accounting")
    if (
        config.max_persistent_bytes is not None
        and persistent_bytes > config.max_persistent_bytes
    ):
        raise ValueError("dual replay allocation exceeds max_persistent_bytes")
    return slot_bytes, persistent_bytes


def _dual_replay_record_extras_bytes(observation_dim: int, action_dim: int) -> int:
    """Prediction, outcome, and ``ReplayWriteResult`` extras excluding persist."""
    prediction_bytes = 4 * observation_dim + 4 * action_dim + 38
    outcome_bytes = 4 * observation_dim + 21
    write_result_extras = 55
    return prediction_bytes + outcome_bytes + write_result_extras


def _dual_replay_sample_extras_bytes(batch_size: int, slot_bytes: int) -> int:
    """Returned ``ReplaySampleResult`` extras excluding persist."""
    return batch_size * (slot_bytes + 13) + 28


def _dual_replay_update_working_set_bytes(
    *,
    total_capacity: int,
    observation_dim: int,
    action_dim: int,
    batch_size: int,
) -> int:
    """Source persist, proposed persist, committed persist, and returned extras.

    ``record`` keeps the source state, the candidate persist, and the
    transaction-selected result live together with the prediction, outcome, and
    write-result extras.  ``sample`` keeps the same three persist copies live
    with the returned batch and sample diagnostics.  The host names the larger
    envelope.
    """
    slot_bytes = 4 * (2 * observation_dim + action_dim + 14) + 11
    persist_bytes = total_capacity * slot_bytes + 60
    extras = max(
        _dual_replay_record_extras_bytes(observation_dim, action_dim),
        _dual_replay_sample_extras_bytes(batch_size, slot_bytes),
    )
    return 3 * persist_bytes + extras


def _preflight_dual_replay_update_working_set(
    *,
    total_capacity: int,
    observation_dim: int,
    action_dim: int,
    batch_size: int,
) -> None:
    """Reject a record/sample envelope the host cannot name in signed int32."""
    working_set_bytes = _dual_replay_update_working_set_bytes(
        total_capacity=total_capacity,
        observation_dim=observation_dim,
        action_dim=action_dim,
        batch_size=batch_size,
    )
    if working_set_bytes > _INT32_MAX:
        raise ValueError(
            "dual replay update working set byte count must fit signed int32"
        )


def _validate_config(config: DualReplayConfig) -> None:
    for name in (
        "total_capacity",
        "short_term_capacity",
        "observation_dim",
        "action_dim",
        "short_term_sample_size",
        "long_term_sample_size",
    ):
        canonical = _validate_positive_int(name, getattr(config, name))
        object.__setattr__(config, name, canonical)
    if config.short_term_capacity >= config.total_capacity:
        raise ValueError("short_term_capacity must leave at least one long-term slot")
    if config.short_term_sample_size > config.short_term_capacity:
        raise ValueError("short_term_sample_size exceeds short-term capacity")
    if config.long_term_sample_size > config.long_term_capacity:
        raise ValueError("long_term_sample_size exceeds long-term capacity")
    if type(config.long_term_policy) is not str or config.long_term_policy not in {
        "reservoir",
        "calibrated",
    }:
        raise ValueError("long_term_policy must be 'reservoir' or 'calibrated'")
    if type(config.aleatoric_control) is not str or config.aleatoric_control not in {
        "veto",
        "downweight",
    }:
        raise ValueError("aleatoric_control must be 'veto' or 'downweight'")
    object.__setattr__(
        config,
        "max_representation_lag",
        _validate_positive_int(
            "max_representation_lag", config.max_representation_lag, minimum=0
        ),
    )

    with np.errstate(over="ignore"):
        weight_sum = float(
            np.float32(config.surprise_weight)
            + np.float32(config.coverage_weight)
            + np.float32(config.progress_weight)
        )
    if not math.isfinite(weight_sum):
        raise ValueError(
            "surprise_weight, coverage_weight, and progress_weight must have a "
            "finite float32 sum"
        )
    if config.max_persistent_bytes is not None:
        object.__setattr__(
            config,
            "max_persistent_bytes",
            _validate_positive_int(
                "max_persistent_bytes",
                config.max_persistent_bytes,
                maximum=_UINT32_MAX,
            ),
        )
    _allocation_sizes(config)
    _preflight_dual_replay_update_working_set(
        total_capacity=config.total_capacity,
        observation_dim=config.observation_dim,
        action_dim=config.action_dim,
        batch_size=config.batch_size,
    )


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array with shape and dtype metadata")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")
    expected = jnp.dtype(dtype)
    actual = jnp.dtype(value.dtype)
    if actual != expected:
        raise TypeError(f"{name} must have dtype {expected}; got {actual}")


def _tree_nbytes(tree: object) -> int:
    return sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in jax.tree.leaves(tree))


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _payload_digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _saturating_increment(value: Array) -> Array:
    return jnp.minimum(value, jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)) + jnp.asarray(
        1, dtype=jnp.int32
    )


def _key_from_data(key_data: Array) -> Array:
    return cast(Array, jr.wrap_key_data(key_data))


def _key_data(key: Array) -> Array:
    return jnp.asarray(jr.key_data(key), dtype=jnp.uint32)


def reservoir_selection(
    key: Array,
    candidate_number: Array,
    capacity: int,
) -> ReservoirSelection:
    """Apply the exact one-based Algorithm-R (Vitter 1985) selection contract.

    Candidates ``1..capacity`` deterministically fill slots ``0..capacity-1``.
    For candidate ``n > capacity``, one integer is drawn uniformly from
    ``[0, n)`` and the candidate replaces slot ``draw`` iff ``draw < capacity``.
    The caller owns key advancement; this pure calculation does not return a key.
    """
    if type(capacity) is not int or capacity < 1:
        raise ValueError("capacity must be a positive integer")
    _require_array(candidate_number, name="candidate_number", shape=(), dtype=jnp.int32)
    candidate_valid = candidate_number >= 1
    fill = candidate_valid & (candidate_number <= jnp.asarray(capacity, dtype=jnp.int32))
    safe_population = jnp.maximum(candidate_number, jnp.asarray(1, dtype=jnp.int32))
    draw = jr.randint(
        key,
        shape=(),
        minval=0,
        maxval=safe_population,
        dtype=jnp.int32,
    )
    fill_slot = jnp.maximum(candidate_number - 1, 0)
    selected = candidate_valid & (fill | (draw < jnp.asarray(capacity, dtype=jnp.int32)))
    slot = jnp.where(fill, fill_slot, jnp.where(selected, draw, -1)).astype(jnp.int32)
    return ReservoirSelection(
        selected=selected,
        slot=slot,
        candidate_number=candidate_number,
        population_size=candidate_number,
        draw=jnp.where(candidate_valid & (~fill), draw, -1).astype(jnp.int32),
        draw_available=candidate_valid & (~fill),
        capacity=jnp.asarray(capacity, dtype=jnp.int32),
    )


_ENTRY_FLOAT_FIELDS = (
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "discounts",
    "old_behavior_probabilities",
    "old_behavior_logits",
    "old_value_targets",
    "epistemic_surprises",
    "aleatoric_uncertainties",
    "learning_progress",
    "safety_costs",
    "insertion_priorities",
)
_ENTRY_BOOL_FIELDS = (
    "terminated",
    "truncated",
    "old_behavior_probability_available",
    "old_behavior_logit_available",
    "old_value_target_available",
    "epistemic_surprise_available",
    "aleatoric_uncertainty_available",
    "learning_progress_available",
    "safety_cost_available",
    "insertion_priority_available",
    "valid",
)
_ENTRY_INT_FIELDS = (
    "representation_versions",
    "provenance_ids",
    "source_ids",
    "eviction_provenance_ids",
)
_STATE_COUNTER_FIELDS = (
    "short_term_size",
    "short_term_head",
    "long_term_size",
    "write_attempt_count",
    "accepted_transition_count",
    "rejected_transition_count",
    "long_term_candidate_count",
    "long_term_write_count",
    "short_term_eviction_count",
    "long_term_eviction_count",
    "long_term_rejection_count",
    "sample_count",
)


class DualReplayMemory:
    """Fixed-total-capacity FIFO plus long-term replay memory."""

    def __init__(self, config: DualReplayConfig):
        _validate_config(config)
        self._config = config
        slot_bytes, persistent_bytes = _allocation_sizes(config)
        initial = self._make_initial_state(jnp.zeros((2,), dtype=jnp.uint32), 0)
        if _tree_nbytes(initial) != persistent_bytes:
            raise RuntimeError("dual replay allocation formula disagrees with allocated state")
        short_bytes = _tree_nbytes(initial.short_term)
        long_bytes = _tree_nbytes(initial.long_term)
        short_slot_bytes = short_bytes // config.short_term_capacity
        long_slot_bytes = long_bytes // config.long_term_capacity
        if short_slot_bytes != long_slot_bytes:
            raise RuntimeError("replay strata disagree on fixed slot bytes")
        self._persistent_bytes = persistent_bytes
        if short_slot_bytes != slot_bytes:
            raise RuntimeError("dual replay slot formula disagrees with allocated state")
        self._slot_bytes = slot_bytes

    @property
    def config(self) -> DualReplayConfig:
        """Static memory configuration."""
        return self._config

    @property
    def persistent_bytes(self) -> int:
        """Exact bytes in every persistent JAX array leaf."""
        return self._persistent_bytes

    @property
    def slot_bytes(self) -> int:
        """Exact bytes in one transition slot."""
        return self._slot_bytes

    def to_config(self) -> dict[str, object]:
        """Return a strict construction payload."""
        return {
            "schema": DUAL_REPLAY_CONFIG_SCHEMA,
            "type": "DualReplayMemory",
            "mechanism_status": MECHANISM_STATUS,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, Any]) -> DualReplayMemory:
        """Reconstruct a memory from :meth:`to_config`."""
        values = _copy_mapping(payload, name="dual replay memory config")
        expected = {"schema", "type", "mechanism_status", "config"}
        if set(values) != expected:
            raise ValueError("dual replay memory config fields do not match the v1 schema")
        if type(values["schema"]) is not str or values["schema"] != DUAL_REPLAY_CONFIG_SCHEMA:
            raise ValueError("unexpected dual replay memory schema")
        if type(values["type"]) is not str or values["type"] != "DualReplayMemory":
            raise ValueError("unexpected dual replay memory type")
        if (
            type(values["mechanism_status"]) is not str
            or values["mechanism_status"] != MECHANISM_STATUS
        ):
            raise ValueError("dual replay memory must remain mechanism-only")
        inner = values["config"]
        return cls(DualReplayConfig.from_config(cast(Mapping[str, Any], inner)))

    @staticmethod
    def _empty_entries(capacity: int, observation_dim: int, action_dim: int) -> ReplayEntries:
        def floats(shape: tuple[int, ...]) -> Array:
            return jnp.zeros(shape, dtype=jnp.float32)

        def flags(shape: tuple[int, ...]) -> Array:
            return jnp.zeros(shape, dtype=jnp.bool_)

        def sentinel(shape: tuple[int, ...]) -> Array:
            return jnp.full(shape, -1, dtype=jnp.int32)

        return ReplayEntries(
            observations=floats((capacity, observation_dim)),
            next_observations=floats((capacity, observation_dim)),
            actions=floats((capacity, action_dim)),
            rewards=floats((capacity,)),
            discounts=floats((capacity,)),
            terminated=flags((capacity,)),
            truncated=flags((capacity,)),
            old_behavior_probabilities=floats((capacity,)),
            old_behavior_probability_available=flags((capacity,)),
            old_behavior_logits=floats((capacity,)),
            old_behavior_logit_available=flags((capacity,)),
            old_value_targets=floats((capacity,)),
            old_value_target_available=flags((capacity,)),
            epistemic_surprises=floats((capacity,)),
            epistemic_surprise_available=flags((capacity,)),
            aleatoric_uncertainties=floats((capacity,)),
            aleatoric_uncertainty_available=flags((capacity,)),
            learning_progress=floats((capacity,)),
            learning_progress_available=flags((capacity,)),
            safety_costs=floats((capacity,)),
            safety_cost_available=flags((capacity,)),
            representation_versions=sentinel((capacity,)),
            provenance_ids=sentinel((capacity,)),
            source_ids=sentinel((capacity,)),
            eviction_provenance_ids=sentinel((capacity,)),
            insertion_priorities=floats((capacity,)),
            insertion_priority_available=flags((capacity,)),
            valid=flags((capacity,)),
        )

    def _make_initial_state(self, key_data: Array, persistent_bytes: int) -> DualReplayState:
        cfg = self._config
        zero = jnp.asarray(0, dtype=jnp.int32)
        return DualReplayState(
            short_term=self._empty_entries(
                cfg.short_term_capacity, cfg.observation_dim, cfg.action_dim
            ),
            long_term=self._empty_entries(
                cfg.long_term_capacity, cfg.observation_dim, cfg.action_dim
            ),
            rng_key_data=jnp.asarray(key_data, dtype=jnp.uint32),
            short_term_size=zero,
            short_term_head=zero,
            long_term_size=zero,
            write_attempt_count=zero,
            accepted_transition_count=zero,
            rejected_transition_count=zero,
            long_term_candidate_count=zero,
            long_term_write_count=zero,
            short_term_eviction_count=zero,
            long_term_eviction_count=zero,
            long_term_rejection_count=zero,
            sample_count=zero,
            persistent_bytes=jnp.asarray(persistent_bytes, dtype=jnp.uint32),
        )

    def init(self, key: Array) -> DualReplayState:
        """Allocate an empty replay state using the supplied deterministic key."""
        key_data = _key_data(key)
        _require_array(key_data, name="key data", shape=(2,), dtype=jnp.uint32)
        state = self._make_initial_state(key_data, self._persistent_bytes)
        if _tree_nbytes(state) != self._persistent_bytes:
            raise RuntimeError("persistent byte accounting disagrees with allocated state")
        return state

    def _validate_entries_static(
        self,
        entries: ReplayEntries,
        *,
        capacity: int,
        name: str,
    ) -> None:
        if not isinstance(entries, ReplayEntries):
            raise TypeError(f"{name} must be ReplayEntries")
        cfg = self._config
        vector_fields = {
            "observations": (capacity, cfg.observation_dim),
            "next_observations": (capacity, cfg.observation_dim),
            "actions": (capacity, cfg.action_dim),
        }
        float_fields = {
            **vector_fields,
            **{
                field: (capacity,)
                for field in (
                    "rewards",
                    "discounts",
                    "old_behavior_probabilities",
                    "old_behavior_logits",
                    "old_value_targets",
                    "epistemic_surprises",
                    "aleatoric_uncertainties",
                    "learning_progress",
                    "safety_costs",
                    "insertion_priorities",
                )
            },
        }
        for field, shape in float_fields.items():
            _require_array(
                getattr(entries, field),
                name=f"{name}.{field}",
                shape=shape,
                dtype=jnp.float32,
            )
        for field in (
            "terminated",
            "truncated",
            "old_behavior_probability_available",
            "old_behavior_logit_available",
            "old_value_target_available",
            "epistemic_surprise_available",
            "aleatoric_uncertainty_available",
            "learning_progress_available",
            "safety_cost_available",
            "insertion_priority_available",
            "valid",
        ):
            _require_array(
                getattr(entries, field),
                name=f"{name}.{field}",
                shape=(capacity,),
                dtype=jnp.bool_,
            )
        for field in (
            "representation_versions",
            "provenance_ids",
            "source_ids",
            "eviction_provenance_ids",
        ):
            _require_array(
                getattr(entries, field),
                name=f"{name}.{field}",
                shape=(capacity,),
                dtype=jnp.int32,
            )

    def _validate_state_static_contract(self, state: DualReplayState) -> None:
        if not isinstance(state, DualReplayState):
            raise TypeError("state must be a DualReplayState")
        cfg = self._config
        self._validate_entries_static(
            state.short_term,
            capacity=cfg.short_term_capacity,
            name="state.short_term",
        )
        self._validate_entries_static(
            state.long_term,
            capacity=cfg.long_term_capacity,
            name="state.long_term",
        )
        _require_array(
            state.rng_key_data,
            name="state.rng_key_data",
            shape=(2,),
            dtype=jnp.uint32,
        )
        for field in (
            "short_term_size",
            "short_term_head",
            "long_term_size",
            "write_attempt_count",
            "accepted_transition_count",
            "rejected_transition_count",
            "long_term_candidate_count",
            "long_term_write_count",
            "short_term_eviction_count",
            "long_term_eviction_count",
            "long_term_rejection_count",
            "sample_count",
        ):
            _require_array(
                getattr(state, field),
                name=f"state.{field}",
                shape=(),
                dtype=jnp.int32,
            )
        _require_array(
            state.persistent_bytes,
            name="state.persistent_bytes",
            shape=(),
            dtype=jnp.uint32,
        )

    def _validate_prediction_static_contract(self, prediction: ReplayPrediction) -> None:
        if not isinstance(prediction, ReplayPrediction):
            raise TypeError("prediction must be a ReplayPrediction")
        _require_array(
            prediction.observation,
            name="prediction.observation",
            shape=(self._config.observation_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            prediction.action,
            name="prediction.action",
            shape=(self._config.action_dim,),
            dtype=jnp.float32,
        )
        for field in (
            "old_behavior_probability",
            "old_behavior_logit",
            "old_value_target",
            "epistemic_surprise",
            "aleatoric_uncertainty",
        ):
            _require_array(
                getattr(prediction, field),
                name=f"prediction.{field}",
                shape=(),
                dtype=jnp.float32,
            )
        for field in (
            "old_behavior_probability_available",
            "old_behavior_logit_available",
            "old_value_target_available",
            "epistemic_surprise_available",
            "aleatoric_uncertainty_available",
            "valid",
        ):
            _require_array(
                getattr(prediction, field),
                name=f"prediction.{field}",
                shape=(),
                dtype=jnp.bool_,
            )
        for field in ("representation_version", "provenance_id", "source_id"):
            _require_array(
                getattr(prediction, field),
                name=f"prediction.{field}",
                shape=(),
                dtype=jnp.int32,
            )

    def _validate_outcome_static_contract(self, outcome: ReplayOutcome) -> None:
        if not isinstance(outcome, ReplayOutcome):
            raise TypeError("outcome must be a ReplayOutcome")
        _require_array(
            outcome.next_observation,
            name="outcome.next_observation",
            shape=(self._config.observation_dim,),
            dtype=jnp.float32,
        )
        for field in ("reward", "discount", "learning_progress", "safety_cost"):
            _require_array(
                getattr(outcome, field),
                name=f"outcome.{field}",
                shape=(),
                dtype=jnp.float32,
            )
        for field in (
            "terminated",
            "truncated",
            "learning_progress_available",
            "safety_cost_available",
            "valid",
        ):
            _require_array(
                getattr(outcome, field),
                name=f"outcome.{field}",
                shape=(),
                dtype=jnp.bool_,
            )

    @staticmethod
    def _entries_are_valid(entries: ReplayEntries) -> Array:
        valid = entries.valid
        float_payload_finite = (
            jnp.all(jnp.isfinite(entries.observations))
            & jnp.all(jnp.isfinite(entries.next_observations))
            & jnp.all(jnp.isfinite(entries.actions))
            & jnp.all(jnp.isfinite(entries.rewards))
            & jnp.all(jnp.isfinite(entries.discounts))
            & jnp.all(jnp.isfinite(entries.old_behavior_probabilities))
            & jnp.all(jnp.isfinite(entries.old_behavior_logits))
            & jnp.all(jnp.isfinite(entries.old_value_targets))
            & jnp.all(jnp.isfinite(entries.epistemic_surprises))
            & jnp.all(jnp.isfinite(entries.aleatoric_uncertainties))
            & jnp.all(jnp.isfinite(entries.learning_progress))
            & jnp.all(jnp.isfinite(entries.safety_costs))
            & jnp.all(jnp.isfinite(entries.insertion_priorities))
        )
        ranges_valid = (
            jnp.all((entries.discounts >= 0.0) & (entries.discounts <= 1.0))
            & jnp.all((~valid) | ((entries.discounts == 0.0) == entries.terminated))
            & jnp.all(
                (~entries.old_behavior_probability_available)
                | (
                    (entries.old_behavior_probabilities > 0.0)
                    & (entries.old_behavior_probabilities <= 1.0)
                )
            )
            & jnp.all(entries.epistemic_surprises >= 0.0)
            & jnp.all(entries.aleatoric_uncertainties >= 0.0)
            & jnp.all(entries.safety_costs >= 0.0)
            & jnp.all(
                (~entries.insertion_priority_available)
                | (
                    (entries.insertion_priorities >= 0.0)
                    & (entries.insertion_priorities <= 1.0)
                )
            )
        )
        availability_honest = (
            jnp.all(
                entries.old_behavior_probability_available
                | (entries.old_behavior_probabilities == 0.0)
            )
            & jnp.all(
                entries.old_behavior_logit_available | (entries.old_behavior_logits == 0.0)
            )
            & jnp.all(entries.old_value_target_available | (entries.old_value_targets == 0.0))
            & jnp.all(
                entries.epistemic_surprise_available | (entries.epistemic_surprises == 0.0)
            )
            & jnp.all(
                entries.aleatoric_uncertainty_available
                | (entries.aleatoric_uncertainties == 0.0)
            )
            & jnp.all(entries.learning_progress_available | (entries.learning_progress == 0.0))
            & jnp.all(entries.safety_cost_available | (entries.safety_costs == 0.0))
            & jnp.all(
                entries.insertion_priority_available | (entries.insertion_priorities == 0.0)
            )
        )
        active_metadata = jnp.all(
            (~valid)
            | (
                (entries.representation_versions >= 0)
                & (entries.provenance_ids >= 0)
                & (entries.source_ids >= 0)
                & (entries.eviction_provenance_ids >= -1)
            )
        )
        inactive_metadata = jnp.all(
            valid
            | (
                (entries.representation_versions == -1)
                & (entries.provenance_ids == -1)
                & (entries.source_ids == -1)
                & (entries.eviction_provenance_ids == -1)
                & (~entries.old_behavior_probability_available)
                & (~entries.old_behavior_logit_available)
                & (~entries.old_value_target_available)
                & (~entries.epistemic_surprise_available)
                & (~entries.aleatoric_uncertainty_available)
                & (~entries.learning_progress_available)
                & (~entries.safety_cost_available)
                & (~entries.insertion_priority_available)
            )
        )
        inactive_payload_neutral = jnp.all(
            valid
            | (
                jnp.all(entries.observations == 0.0, axis=1)
                & jnp.all(entries.next_observations == 0.0, axis=1)
                & jnp.all(entries.actions == 0.0, axis=1)
                & (entries.rewards == 0.0)
                & (entries.discounts == 0.0)
                & (~entries.terminated)
                & (~entries.truncated)
                & (entries.old_behavior_probabilities == 0.0)
                & (entries.old_behavior_logits == 0.0)
                & (entries.old_value_targets == 0.0)
                & (entries.epistemic_surprises == 0.0)
                & (entries.aleatoric_uncertainties == 0.0)
                & (entries.learning_progress == 0.0)
                & (entries.safety_costs == 0.0)
                & (entries.insertion_priorities == 0.0)
            )
        )
        return (
            float_payload_finite
            & ranges_valid
            & availability_honest
            & active_metadata
            & inactive_metadata
            & inactive_payload_neutral
        )

    def _state_is_valid(self, state: DualReplayState) -> Array:
        cfg = self._config
        counters = (
            state.short_term_size,
            state.short_term_head,
            state.long_term_size,
            state.write_attempt_count,
            state.accepted_transition_count,
            state.rejected_transition_count,
            state.long_term_candidate_count,
            state.long_term_write_count,
            state.short_term_eviction_count,
            state.long_term_eviction_count,
            state.long_term_rejection_count,
            state.sample_count,
        )
        counters_nonnegative = jnp.asarray(True)
        counters_in_range = jnp.asarray(True)
        for counter in counters:
            counters_nonnegative = counters_nonnegative & (counter >= 0)
            counters_in_range = counters_in_range & (counter <= _INT32_MAX)
        short_count = jnp.sum(state.short_term.valid.astype(jnp.int32))
        long_count = jnp.sum(state.long_term.valid.astype(jnp.int32))
        expected_short_evictions = jnp.maximum(
            state.accepted_transition_count
            - jnp.asarray(cfg.short_term_capacity, dtype=jnp.int32),
            0,
        )
        expected_short_size = jnp.minimum(
            state.accepted_transition_count,
            jnp.asarray(cfg.short_term_capacity, dtype=jnp.int32),
        )
        expected_long_size = jnp.minimum(
            state.long_term_write_count,
            jnp.asarray(cfg.long_term_capacity, dtype=jnp.int32),
        )
        expected_long_evictions = jnp.maximum(
            state.long_term_write_count
            - jnp.asarray(cfg.long_term_capacity, dtype=jnp.int32),
            0,
        )
        expected_head = jnp.mod(
            state.accepted_transition_count,
            jnp.asarray(cfg.short_term_capacity, dtype=jnp.int32),
        )
        expected_short_valid = jnp.arange(cfg.short_term_capacity) < state.short_term_size
        expected_long_valid = jnp.arange(cfg.long_term_capacity) < state.long_term_size
        if cfg.long_term_policy == "calibrated":
            policy_consistent = jnp.all(
                (~state.long_term.valid)
                | (
                    state.long_term.insertion_priority_available
                    & state.long_term.epistemic_surprise_available
                    & state.long_term.aleatoric_uncertainty_available
                    & state.long_term.learning_progress_available
                )
            )
        else:
            policy_consistent = (
                jnp.all(~state.short_term.insertion_priority_available)
                & jnp.all(~state.long_term.insertion_priority_available)
            )
        counter_relations = (
            (state.short_term_size == short_count)
            & (state.long_term_size == long_count)
            & (state.short_term_size == expected_short_size)
            & (state.long_term_size == expected_long_size)
            & (state.short_term_head == expected_head)
            & (
                state.rejected_transition_count
                == state.write_attempt_count - state.accepted_transition_count
            )
            & (state.long_term_candidate_count == state.accepted_transition_count)
            & (state.long_term_write_count <= state.long_term_candidate_count)
            & (
                state.long_term_rejection_count
                == state.long_term_candidate_count - state.long_term_write_count
            )
            & (state.short_term_eviction_count == expected_short_evictions)
            & (state.long_term_eviction_count == expected_long_evictions)
            & jnp.all(state.short_term.valid == expected_short_valid)
            & jnp.all(state.long_term.valid == expected_long_valid)
            & policy_consistent
        )
        return (
            self._entries_are_valid(state.short_term)
            & self._entries_are_valid(state.long_term)
            & counters_nonnegative
            & counters_in_range
            & counter_relations
            & (
                state.persistent_bytes
                == jnp.asarray(self._persistent_bytes, dtype=jnp.uint32)
            )
        )

    @staticmethod
    def _prediction_is_valid(prediction: ReplayPrediction) -> Array:
        finite = (
            jnp.all(jnp.isfinite(prediction.observation))
            & jnp.all(jnp.isfinite(prediction.action))
            & jnp.isfinite(prediction.old_behavior_probability)
            & jnp.isfinite(prediction.old_behavior_logit)
            & jnp.isfinite(prediction.old_value_target)
            & jnp.isfinite(prediction.epistemic_surprise)
            & jnp.isfinite(prediction.aleatoric_uncertainty)
        )
        availability_honest = (
            (
                prediction.old_behavior_probability_available
                | (prediction.old_behavior_probability == 0.0)
            )
            & (
                prediction.old_behavior_logit_available
                | (prediction.old_behavior_logit == 0.0)
            )
            & (
                prediction.old_value_target_available | (prediction.old_value_target == 0.0)
            )
            & (
                prediction.epistemic_surprise_available
                | (prediction.epistemic_surprise == 0.0)
            )
            & (
                prediction.aleatoric_uncertainty_available
                | (prediction.aleatoric_uncertainty == 0.0)
            )
        )
        ranges = (
            (
                (~prediction.old_behavior_probability_available)
                | (
                    (prediction.old_behavior_probability > 0.0)
                    & (prediction.old_behavior_probability <= 1.0)
                )
            )
            & (prediction.epistemic_surprise >= 0.0)
            & (prediction.aleatoric_uncertainty >= 0.0)
            & (prediction.representation_version >= 0)
            & (prediction.provenance_id >= 0)
            & (prediction.source_id >= 0)
        )
        return prediction.valid & finite & availability_honest & ranges

    @staticmethod
    def _outcome_is_valid(outcome: ReplayOutcome) -> Array:
        finite = (
            jnp.all(jnp.isfinite(outcome.next_observation))
            & jnp.isfinite(outcome.reward)
            & jnp.isfinite(outcome.discount)
            & jnp.isfinite(outcome.learning_progress)
            & jnp.isfinite(outcome.safety_cost)
        )
        availability_honest = (
            (outcome.learning_progress_available | (outcome.learning_progress == 0.0))
            & (outcome.safety_cost_available | (outcome.safety_cost == 0.0))
        )
        ranges = (
            (outcome.discount >= 0.0)
            & (outcome.discount <= 1.0)
            & ((outcome.discount == 0.0) == outcome.terminated)
            & (outcome.safety_cost >= 0.0)
        )
        return outcome.valid & finite & availability_honest & ranges

    @staticmethod
    def _join_transition(
        prediction: ReplayPrediction,
        outcome: ReplayOutcome,
    ) -> ReplayTransition:
        return ReplayTransition(
            observation=prediction.observation,
            next_observation=outcome.next_observation,
            action=prediction.action,
            reward=outcome.reward,
            discount=outcome.discount,
            terminated=outcome.terminated,
            truncated=outcome.truncated,
            old_behavior_probability=prediction.old_behavior_probability,
            old_behavior_probability_available=prediction.old_behavior_probability_available,
            old_behavior_logit=prediction.old_behavior_logit,
            old_behavior_logit_available=prediction.old_behavior_logit_available,
            old_value_target=prediction.old_value_target,
            old_value_target_available=prediction.old_value_target_available,
            epistemic_surprise=prediction.epistemic_surprise,
            epistemic_surprise_available=prediction.epistemic_surprise_available,
            aleatoric_uncertainty=prediction.aleatoric_uncertainty,
            aleatoric_uncertainty_available=prediction.aleatoric_uncertainty_available,
            learning_progress=outcome.learning_progress,
            learning_progress_available=outcome.learning_progress_available,
            safety_cost=outcome.safety_cost,
            safety_cost_available=outcome.safety_cost_available,
            representation_version=prediction.representation_version,
            provenance_id=prediction.provenance_id,
            source_id=prediction.source_id,
            valid=prediction.valid & outcome.valid,
        )

    @staticmethod
    def _set_entry(
        entries: ReplayEntries,
        slot: Array,
        transition: ReplayTransition,
        eviction_provenance_id: Array,
        insertion_priority: Array,
        insertion_priority_available: Array,
    ) -> ReplayEntries:
        return ReplayEntries(
            observations=entries.observations.at[slot].set(transition.observation),
            next_observations=entries.next_observations.at[slot].set(
                transition.next_observation
            ),
            actions=entries.actions.at[slot].set(transition.action),
            rewards=entries.rewards.at[slot].set(transition.reward),
            discounts=entries.discounts.at[slot].set(transition.discount),
            terminated=entries.terminated.at[slot].set(transition.terminated),
            truncated=entries.truncated.at[slot].set(transition.truncated),
            old_behavior_probabilities=entries.old_behavior_probabilities.at[slot].set(
                transition.old_behavior_probability
            ),
            old_behavior_probability_available=entries.old_behavior_probability_available.at[
                slot
            ].set(transition.old_behavior_probability_available),
            old_behavior_logits=entries.old_behavior_logits.at[slot].set(
                transition.old_behavior_logit
            ),
            old_behavior_logit_available=entries.old_behavior_logit_available.at[slot].set(
                transition.old_behavior_logit_available
            ),
            old_value_targets=entries.old_value_targets.at[slot].set(
                transition.old_value_target
            ),
            old_value_target_available=entries.old_value_target_available.at[slot].set(
                transition.old_value_target_available
            ),
            epistemic_surprises=entries.epistemic_surprises.at[slot].set(
                transition.epistemic_surprise
            ),
            epistemic_surprise_available=entries.epistemic_surprise_available.at[slot].set(
                transition.epistemic_surprise_available
            ),
            aleatoric_uncertainties=entries.aleatoric_uncertainties.at[slot].set(
                transition.aleatoric_uncertainty
            ),
            aleatoric_uncertainty_available=entries.aleatoric_uncertainty_available.at[
                slot
            ].set(transition.aleatoric_uncertainty_available),
            learning_progress=entries.learning_progress.at[slot].set(
                transition.learning_progress
            ),
            learning_progress_available=entries.learning_progress_available.at[slot].set(
                transition.learning_progress_available
            ),
            safety_costs=entries.safety_costs.at[slot].set(transition.safety_cost),
            safety_cost_available=entries.safety_cost_available.at[slot].set(
                transition.safety_cost_available
            ),
            representation_versions=entries.representation_versions.at[slot].set(
                transition.representation_version
            ),
            provenance_ids=entries.provenance_ids.at[slot].set(transition.provenance_id),
            source_ids=entries.source_ids.at[slot].set(transition.source_id),
            eviction_provenance_ids=entries.eviction_provenance_ids.at[slot].set(
                eviction_provenance_id
            ),
            insertion_priorities=entries.insertion_priorities.at[slot].set(
                insertion_priority
            ),
            insertion_priority_available=entries.insertion_priority_available.at[slot].set(
                insertion_priority_available
            ),
            valid=entries.valid.at[slot].set(True),
        )

    def _calibrated_priority(
        self,
        long_term: ReplayEntries,
        transition: ReplayTransition,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        cfg = self._config
        surprise = jnp.clip(
            transition.epistemic_surprise
            / jnp.asarray(cfg.surprise_scale, dtype=jnp.float32),
            0.0,
            1.0,
        )
        progress = jnp.clip(
            jnp.maximum(transition.learning_progress, 0.0)
            / jnp.asarray(cfg.progress_scale, dtype=jnp.float32),
            0.0,
            1.0,
        )
        distances = jnp.linalg.norm(
            long_term.observations - transition.observation[None, :], axis=1
        )
        nearest = jnp.min(jnp.where(long_term.valid, distances, jnp.inf))
        coverage = jnp.where(
            jnp.any(long_term.valid),
            jnp.clip(nearest / jnp.asarray(cfg.coverage_scale, dtype=jnp.float32), 0.0, 1.0),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        surprise_weight = jnp.asarray(cfg.surprise_weight, dtype=jnp.float32)
        coverage_weight = jnp.asarray(cfg.coverage_weight, dtype=jnp.float32)
        progress_weight = jnp.asarray(cfg.progress_weight, dtype=jnp.float32)
        weight_sum = surprise_weight + coverage_weight + progress_weight
        raw_priority = jnp.clip(
            (surprise_weight * surprise + coverage_weight * coverage + progress_weight * progress)
            / weight_sum,
            0.0,
            1.0,
        )
        components_available = (
            transition.epistemic_surprise_available
            & transition.learning_progress_available
            & transition.aleatoric_uncertainty_available
        )
        if cfg.aleatoric_control == "veto":
            control_passed = (
                transition.aleatoric_uncertainty_available
                & (
                    transition.aleatoric_uncertainty
                    <= jnp.asarray(cfg.max_aleatoric_uncertainty, dtype=jnp.float32)
                )
            )
            priority = raw_priority
        else:
            control_passed = transition.aleatoric_uncertainty_available
            priority = raw_priority / (
                1.0
                + transition.aleatoric_uncertainty
                / jnp.asarray(cfg.aleatoric_downweight_scale, dtype=jnp.float32)
            )
        priority = jnp.where(components_available, priority, 0.0).astype(jnp.float32)
        return (
            priority,
            components_available,
            surprise.astype(jnp.float32),
            coverage.astype(jnp.float32),
            progress.astype(jnp.float32),
            control_passed,
            transition.aleatoric_uncertainty_available,
        )

    def _empty_write_result(
        self,
        state: DualReplayState,
        *,
        state_valid: Array,
        input_valid: Array,
        counter_available: Array,
    ) -> ReplayWriteResult:
        false = jnp.asarray(False)
        minus_one = jnp.asarray(-1, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return ReplayWriteResult(
            state=state,
            state_valid=state_valid,
            input_valid=input_valid,
            counter_available=counter_available,
            wrote_short_term=false,
            short_term_slot=minus_one,
            short_term_evicted=false,
            short_term_evicted_provenance_id=minus_one,
            long_term_candidate_number=minus_one,
            long_term_priority=zero,
            long_term_priority_available=false,
            surprise_component=zero,
            coverage_component=zero,
            progress_component=zero,
            aleatoric_control_passed=false,
            aleatoric_control_available=false,
            reservoir_draw=minus_one,
            reservoir_draw_available=false,
            reservoir_population_size=minus_one,
            wrote_long_term=false,
            long_term_slot=minus_one,
            long_term_evicted=false,
            long_term_evicted_provenance_id=minus_one,
        )

    def _record_valid_transition(
        self,
        state: DualReplayState,
        transition: ReplayTransition,
        state_valid: Array,
        input_valid: Array,
        counter_available: Array,
    ) -> ReplayWriteResult:
        cfg = self._config
        short_slot = state.short_term_head
        short_evicted = state.short_term.valid[short_slot]
        short_evicted_provenance = jnp.where(
            short_evicted,
            state.short_term.provenance_ids[short_slot],
            -1,
        ).astype(jnp.int32)
        candidate_number = _saturating_increment(state.long_term_candidate_count)

        if cfg.long_term_policy == "calibrated":
            (
                priority,
                priority_available,
                surprise_component,
                coverage_component,
                progress_component,
                aleatoric_passed,
                aleatoric_available,
            ) = self._calibrated_priority(state.long_term, transition)
            long_has_empty = state.long_term_size < cfg.long_term_capacity
            empty_slot = jnp.argmax((~state.long_term.valid).astype(jnp.int32))
            retention_priority = jnp.where(
                state.long_term.valid & state.long_term.insertion_priority_available,
                state.long_term.insertion_priorities,
                jnp.inf,
            )
            replacement_slot = jnp.argmin(retention_priority).astype(jnp.int32)
            replacement_floor = state.long_term.insertion_priorities[replacement_slot]
            clears_replacement = priority >= (
                replacement_floor
                + jnp.asarray(cfg.calibrated_replacement_margin, dtype=jnp.float32)
            )
            clears_threshold = priority >= jnp.asarray(
                cfg.calibrated_priority_threshold, dtype=jnp.float32
            )
            wrote_long = (
                priority_available
                & aleatoric_passed
                & clears_threshold
                & (long_has_empty | clears_replacement)
            )
            long_slot = jnp.where(
                wrote_long,
                jnp.where(long_has_empty, empty_slot, replacement_slot),
                -1,
            ).astype(jnp.int32)
            reservoir_draw = jnp.asarray(-1, dtype=jnp.int32)
            reservoir_draw_available = jnp.asarray(False)
            reservoir_population = jnp.asarray(-1, dtype=jnp.int32)
            next_key_data = state.rng_key_data
        else:
            priority = jnp.asarray(0.0, dtype=jnp.float32)
            priority_available = jnp.asarray(False)
            surprise_component = jnp.asarray(0.0, dtype=jnp.float32)
            coverage_component = jnp.asarray(0.0, dtype=jnp.float32)
            progress_component = jnp.asarray(0.0, dtype=jnp.float32)
            aleatoric_passed = jnp.asarray(False)
            aleatoric_available = jnp.asarray(False)
            next_key, draw_key = jr.split(_key_from_data(state.rng_key_data))
            selection = reservoir_selection(draw_key, candidate_number, cfg.long_term_capacity)
            wrote_long = selection.selected
            long_slot = selection.slot
            reservoir_draw = selection.draw
            reservoir_draw_available = selection.draw_available
            reservoir_population = selection.population_size
            next_key_data = jnp.where(
                selection.draw_available,
                _key_data(next_key),
                state.rng_key_data,
            ).astype(jnp.uint32)

        long_evicted = wrote_long & state.long_term.valid[jnp.maximum(long_slot, 0)]
        long_evicted_provenance = jnp.where(
            long_evicted,
            state.long_term.provenance_ids[jnp.maximum(long_slot, 0)],
            -1,
        ).astype(jnp.int32)

        next_short = self._set_entry(
            state.short_term,
            short_slot,
            transition,
            short_evicted_provenance,
            priority,
            priority_available,
        )

        def write_long(entries: ReplayEntries) -> ReplayEntries:
            return self._set_entry(
                entries,
                long_slot,
                transition,
                long_evicted_provenance,
                priority,
                priority_available,
            )

        next_long = jax.lax.cond(
            wrote_long,
            write_long,
            lambda entries: entries,
            state.long_term,
        )
        next_state = state.replace(
            short_term=next_short,
            long_term=next_long,
            rng_key_data=next_key_data,
            short_term_size=jnp.minimum(
                _saturating_increment(state.short_term_size),
                jnp.asarray(cfg.short_term_capacity, dtype=jnp.int32),
            ),
            short_term_head=jnp.mod(
                _saturating_increment(state.short_term_head),
                jnp.asarray(cfg.short_term_capacity, dtype=jnp.int32),
            ),
            long_term_size=jnp.where(
                wrote_long,
                jnp.minimum(
                    _saturating_increment(state.long_term_size),
                    jnp.asarray(cfg.long_term_capacity, dtype=jnp.int32),
                ),
                state.long_term_size,
            ),
            write_attempt_count=_saturating_increment(state.write_attempt_count),
            accepted_transition_count=_saturating_increment(
                state.accepted_transition_count
            ),
            long_term_candidate_count=candidate_number,
            long_term_write_count=jnp.where(
                wrote_long,
                _saturating_increment(state.long_term_write_count),
                state.long_term_write_count,
            ),
            short_term_eviction_count=jnp.where(
                short_evicted,
                _saturating_increment(state.short_term_eviction_count),
                state.short_term_eviction_count,
            ),
            long_term_eviction_count=jnp.where(
                long_evicted,
                _saturating_increment(state.long_term_eviction_count),
                state.long_term_eviction_count,
            ),
            long_term_rejection_count=jnp.where(
                wrote_long,
                state.long_term_rejection_count,
                _saturating_increment(state.long_term_rejection_count),
            ),
        )
        return ReplayWriteResult(
            state=cast(DualReplayState, next_state),
            state_valid=state_valid,
            input_valid=input_valid,
            counter_available=counter_available,
            wrote_short_term=jnp.asarray(True),
            short_term_slot=short_slot,
            short_term_evicted=short_evicted,
            short_term_evicted_provenance_id=short_evicted_provenance,
            long_term_candidate_number=candidate_number,
            long_term_priority=priority,
            long_term_priority_available=priority_available,
            surprise_component=surprise_component,
            coverage_component=coverage_component,
            progress_component=progress_component,
            aleatoric_control_passed=aleatoric_passed,
            aleatoric_control_available=aleatoric_available,
            reservoir_draw=reservoir_draw,
            reservoir_draw_available=reservoir_draw_available,
            reservoir_population_size=reservoir_population,
            wrote_long_term=wrote_long,
            long_term_slot=long_slot,
            long_term_evicted=long_evicted,
            long_term_evicted_provenance_id=long_evicted_provenance,
        )

    def record(
        self,
        state: DualReplayState,
        prediction: ReplayPrediction,
        outcome: ReplayOutcome,
    ) -> ReplayWriteResult:
        """Atomically record one prediction-before-outcome transition.

        ``prediction`` and ``outcome`` are separate types so an observed target
        cannot be smuggled into the old behavior/value fields at the memory
        boundary.  The FIFO always receives a valid transition.  The static
        long-term policy independently decides whether to retain it.
        """
        self._validate_state_static_contract(state)
        self._validate_prediction_static_contract(prediction)
        self._validate_outcome_static_contract(outcome)
        return cast(ReplayWriteResult, self._record_jit(state, prediction, outcome))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _record_jit(
        self,
        state: DualReplayState,
        prediction: ReplayPrediction,
        outcome: ReplayOutcome,
    ) -> ReplayWriteResult:
        state_valid = self._state_is_valid(state)
        input_valid = self._prediction_is_valid(prediction) & self._outcome_is_valid(outcome)
        counters = jnp.stack(
            (
                state.write_attempt_count,
                state.accepted_transition_count,
                state.rejected_transition_count,
                state.long_term_candidate_count,
                state.long_term_write_count,
                state.short_term_eviction_count,
                state.long_term_eviction_count,
                state.long_term_rejection_count,
            )
        )
        counter_available = jnp.all(counters < _INT32_MAX)
        transition = self._join_transition(prediction, outcome)

        def apply(_: None) -> ReplayWriteResult:
            def accept(_: None) -> ReplayWriteResult:
                return self._record_valid_transition(
                    state,
                    transition,
                    state_valid,
                    input_valid,
                    counter_available,
                )

            def reject(_: None) -> ReplayWriteResult:
                rejected_state = state.replace(
                    write_attempt_count=_saturating_increment(state.write_attempt_count),
                    rejected_transition_count=_saturating_increment(
                        state.rejected_transition_count
                    ),
                )
                return self._empty_write_result(
                    cast(DualReplayState, rejected_state),
                    state_valid=state_valid,
                    input_valid=input_valid,
                    counter_available=counter_available,
                )

            return cast(
                ReplayWriteResult,
                jax.lax.cond(input_valid, accept, reject, operand=None),
            )

        def fail_closed(_: None) -> ReplayWriteResult:
            return self._empty_write_result(
                state,
                state_valid=state_valid,
                input_valid=input_valid,
                counter_available=counter_available,
            )

        return cast(
            ReplayWriteResult,
            jax.lax.cond(
                state_valid & counter_available,
                apply,
                fail_closed,
                operand=None,
            ),
        )

    @staticmethod
    def _gather_entries(entries: ReplayEntries, indices: Array) -> ReplayEntries:
        return cast(ReplayEntries, jax.tree.map(lambda value: value[indices], entries))

    @staticmethod
    def _mask_entries(entries: ReplayEntries, mask: Array) -> ReplayEntries:
        vector_mask = mask[:, None]

        def vectors(value: Array) -> Array:
            return jnp.where(vector_mask, value, jnp.zeros_like(value))

        def floats(value: Array) -> Array:
            return jnp.where(mask, value, jnp.zeros_like(value))

        def flags(value: Array) -> Array:
            return mask & value

        def metadata(value: Array) -> Array:
            return jnp.where(mask, value, jnp.full_like(value, -1))

        return ReplayEntries(
            observations=vectors(entries.observations),
            next_observations=vectors(entries.next_observations),
            actions=vectors(entries.actions),
            rewards=floats(entries.rewards),
            discounts=floats(entries.discounts),
            terminated=flags(entries.terminated),
            truncated=flags(entries.truncated),
            old_behavior_probabilities=floats(entries.old_behavior_probabilities),
            old_behavior_probability_available=flags(
                entries.old_behavior_probability_available
            ),
            old_behavior_logits=floats(entries.old_behavior_logits),
            old_behavior_logit_available=flags(entries.old_behavior_logit_available),
            old_value_targets=floats(entries.old_value_targets),
            old_value_target_available=flags(entries.old_value_target_available),
            epistemic_surprises=floats(entries.epistemic_surprises),
            epistemic_surprise_available=flags(entries.epistemic_surprise_available),
            aleatoric_uncertainties=floats(entries.aleatoric_uncertainties),
            aleatoric_uncertainty_available=flags(
                entries.aleatoric_uncertainty_available
            ),
            learning_progress=floats(entries.learning_progress),
            learning_progress_available=flags(entries.learning_progress_available),
            safety_costs=floats(entries.safety_costs),
            safety_cost_available=flags(entries.safety_cost_available),
            representation_versions=metadata(entries.representation_versions),
            provenance_ids=metadata(entries.provenance_ids),
            source_ids=metadata(entries.source_ids),
            eviction_provenance_ids=metadata(entries.eviction_provenance_ids),
            insertion_priorities=floats(entries.insertion_priorities),
            insertion_priority_available=flags(entries.insertion_priority_available),
            valid=mask & entries.valid,
        )

    @staticmethod
    def _concat_entries(left: ReplayEntries, right: ReplayEntries) -> ReplayEntries:
        return cast(
            ReplayEntries,
            jax.tree.map(lambda a, b: jnp.concatenate((a, b), axis=0), left, right),
        )

    def _sample_stratum(
        self,
        entries: ReplayEntries,
        key: Array,
        current_representation_version: Array,
        sample_size: int,
    ) -> tuple[ReplayEntries, Array, Array, Array, Array, Array, Array]:
        lag = current_representation_version - entries.representation_versions
        stale = (
            entries.valid
            & (lag >= 0)
            & (lag > jnp.asarray(self._config.max_representation_lag, dtype=jnp.int32))
        )
        future = entries.valid & (lag < 0)
        eligible = (
            entries.valid
            & (lag >= 0)
            & (lag <= jnp.asarray(self._config.max_representation_lag, dtype=jnp.int32))
        )
        scores = jr.uniform(key, shape=entries.valid.shape, dtype=jnp.float32)
        ranked_scores = jnp.where(eligible, scores, -jnp.inf)
        _, indices = jax.lax.top_k(ranked_scores, sample_size)
        selected = eligible[indices]
        gathered = self._mask_entries(self._gather_entries(entries, indices), selected)
        selected_lag = jnp.where(selected, lag[indices], -1).astype(jnp.int32)
        selected_indices = jnp.where(selected, indices, -1).astype(jnp.int32)
        return (
            gathered,
            selected_indices,
            selected,
            selected_lag,
            jnp.sum(stale.astype(jnp.int32)),
            jnp.sum(future.astype(jnp.int32)),
            jnp.sum(eligible.astype(jnp.int32)),
        )

    def _empty_batch(self) -> ReplayBatch:
        cfg = self._config
        entries = self._empty_entries(cfg.batch_size, cfg.observation_dim, cfg.action_dim)
        return ReplayBatch(
            entries=entries,
            valid=jnp.zeros((cfg.batch_size,), dtype=jnp.bool_),
            stratum=jnp.concatenate(
                (
                    jnp.full(
                        (cfg.short_term_sample_size,),
                        SHORT_TERM_STRATUM,
                        dtype=jnp.int32,
                    ),
                    jnp.full(
                        (cfg.long_term_sample_size,),
                        LONG_TERM_STRATUM,
                        dtype=jnp.int32,
                    ),
                )
            ),
            slot=jnp.full((cfg.batch_size,), -1, dtype=jnp.int32),
            representation_lag=jnp.full((cfg.batch_size,), -1, dtype=jnp.int32),
        )

    def sample(
        self,
        state: DualReplayState,
        current_representation_version: Array,
    ) -> ReplaySampleResult:
        """Draw a bounded without-replacement batch from both strata.

        Each stratum has a fixed quota.  Shortages remain explicit padding;
        they are not silently backfilled from the other stratum.  Entries newer
        than the current representation and entries older than
        ``max_representation_lag`` are counted and excluded.
        """
        self._validate_state_static_contract(state)
        _require_array(
            current_representation_version,
            name="current_representation_version",
            shape=(),
            dtype=jnp.int32,
        )
        return cast(
            ReplaySampleResult,
            self._sample_jit(state, current_representation_version),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _sample_jit(
        self,
        state: DualReplayState,
        current_representation_version: Array,
    ) -> ReplaySampleResult:
        state_valid = self._state_is_valid(state)
        version_valid = current_representation_version >= 0
        counter_available = state.sample_count < _INT32_MAX

        def apply(_: None) -> ReplaySampleResult:
            next_key, short_key, long_key = jr.split(_key_from_data(state.rng_key_data), 3)
            (
                short_entries,
                short_indices,
                short_valid,
                short_lag,
                stale_short,
                future_short,
                eligible_short,
            ) = self._sample_stratum(
                state.short_term,
                short_key,
                current_representation_version,
                self._config.short_term_sample_size,
            )
            (
                long_entries,
                long_indices,
                long_valid,
                long_lag,
                stale_long,
                future_long,
                eligible_long,
            ) = self._sample_stratum(
                state.long_term,
                long_key,
                current_representation_version,
                self._config.long_term_sample_size,
            )
            valid = jnp.concatenate((short_valid, long_valid))
            batch = ReplayBatch(
                entries=self._concat_entries(short_entries, long_entries),
                valid=valid,
                stratum=jnp.concatenate(
                    (
                        jnp.full(
                            (self._config.short_term_sample_size,),
                            SHORT_TERM_STRATUM,
                            dtype=jnp.int32,
                        ),
                        jnp.full(
                            (self._config.long_term_sample_size,),
                            LONG_TERM_STRATUM,
                            dtype=jnp.int32,
                        ),
                    )
                ),
                slot=jnp.concatenate((short_indices, long_indices)),
                representation_lag=jnp.concatenate((short_lag, long_lag)),
            )
            next_state = state.replace(
                rng_key_data=_key_data(next_key),
                sample_count=_saturating_increment(state.sample_count),
            )
            return ReplaySampleResult(
                state=cast(DualReplayState, next_state),
                batch=batch,
                state_valid=state_valid,
                representation_version_valid=version_valid,
                counter_available=counter_available,
                stale_detected=(stale_short + stale_long + future_short + future_long) > 0,
                stale_short_term_count=stale_short,
                stale_long_term_count=stale_long,
                future_short_term_count=future_short,
                future_long_term_count=future_long,
                eligible_short_term_count=eligible_short,
                eligible_long_term_count=eligible_long,
            )

        def fail_closed(_: None) -> ReplaySampleResult:
            zero = jnp.asarray(0, dtype=jnp.int32)
            return ReplaySampleResult(
                state=state,
                batch=self._empty_batch(),
                state_valid=state_valid,
                representation_version_valid=version_valid,
                counter_available=counter_available,
                stale_detected=jnp.asarray(False),
                stale_short_term_count=zero,
                stale_long_term_count=zero,
                future_short_term_count=zero,
                future_long_term_count=zero,
                eligible_short_term_count=zero,
                eligible_long_term_count=zero,
            )

        return cast(
            ReplaySampleResult,
            jax.lax.cond(
                state_valid & version_valid & counter_available,
                apply,
                fail_closed,
                operand=None,
            ),
        )

    def accounting(self, state: DualReplayState) -> DualReplayResourceAccounting:
        """Return exact allocation and bounded lifetime counters."""
        self._validate_state_static_contract(state)
        return DualReplayResourceAccounting(
            total_capacity=jnp.asarray(self._config.total_capacity, dtype=jnp.int32),
            short_term_capacity=jnp.asarray(
                self._config.short_term_capacity, dtype=jnp.int32
            ),
            long_term_capacity=jnp.asarray(
                self._config.long_term_capacity, dtype=jnp.int32
            ),
            active_entries=state.short_term_size + state.long_term_size,
            short_term_entries=state.short_term_size,
            long_term_entries=state.long_term_size,
            slot_bytes=jnp.asarray(self._slot_bytes, dtype=jnp.uint32),
            persistent_bytes=state.persistent_bytes,
            write_attempts=state.write_attempt_count,
            accepted_transitions=state.accepted_transition_count,
            rejected_transitions=state.rejected_transition_count,
            long_term_candidates=state.long_term_candidate_count,
            long_term_writes=state.long_term_write_count,
            short_term_evictions=state.short_term_eviction_count,
            long_term_evictions=state.long_term_eviction_count,
            long_term_rejections=state.long_term_rejection_count,
            samples=state.sample_count,
        )

    def validate_state(self, state: DualReplayState) -> None:
        """Raise if a checkpoint/restored state violates any invariant."""
        self._validate_state_static_contract(state)
        if not bool(jax.device_get(self._state_is_valid(state))):
            raise ValueError("dual replay state violates dynamic invariants")

    def state_valid(self, state: DualReplayState) -> Array:
        """Return the dynamic state verdict after exact static validation."""
        self._validate_state_static_contract(state)
        return self._state_is_valid(state)

    @staticmethod
    def _entries_checkpoint_payload(entries: ReplayEntries) -> dict[str, object]:
        payload: dict[str, object] = {}
        for field in (*_ENTRY_FLOAT_FIELDS, *_ENTRY_BOOL_FIELDS, *_ENTRY_INT_FIELDS):
            payload[field] = np.asarray(jax.device_get(getattr(entries, field))).tolist()
        return payload

    def checkpoint_payload(self, state: DualReplayState) -> dict[str, object]:
        """Return a deterministic, digest-bound JSON-compatible checkpoint."""
        self.validate_state(state)
        memory_payload = self.to_config()
        state_payload: dict[str, object] = {
            "short_term": self._entries_checkpoint_payload(state.short_term),
            "long_term": self._entries_checkpoint_payload(state.long_term),
            "rng_key_data": np.asarray(
                jax.device_get(state.rng_key_data), dtype=np.uint32
            ).tolist(),
            **{
                field: int(jax.device_get(getattr(state, field)))
                for field in _STATE_COUNTER_FIELDS
            },
            "persistent_bytes": int(jax.device_get(state.persistent_bytes)),
        }
        return {
            "schema": DUAL_REPLAY_CHECKPOINT_SCHEMA,
            "mechanism_status": MECHANISM_STATUS,
            "memory": memory_payload,
            "config_digest": _payload_digest(memory_payload),
            "state": state_payload,
            "state_digest": _payload_digest(state_payload),
        }

    @staticmethod
    def _checkpoint_array(
        value: object,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> Array:
        def leaves(node: object, remaining_shape: tuple[int, ...]) -> list[object]:
            if not remaining_shape:
                return [node]
            if type(node) is not list or len(node) != remaining_shape[0]:
                raise ValueError(f"{name} must have shape {shape}")
            result: list[object] = []
            for child in node:
                result.extend(leaves(child, remaining_shape[1:]))
            return result

        untyped = leaves(value, shape)
        expected = np.dtype(dtype)
        if np.issubdtype(expected, np.floating):
            if any(
                type(item) is not int and type(item) is not float
                for item in untyped
            ):
                raise ValueError(f"{name} must contain only JSON real numbers")
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                array: npt.NDArray[np.float32] = np.asarray(value, dtype=np.float32)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain finite float32 values")
            return jnp.asarray(array, dtype=jnp.float32)
        if np.issubdtype(expected, np.bool_):
            if any(type(item) is not bool for item in untyped):
                raise ValueError(f"{name} must contain only booleans")
            return jnp.asarray(value, dtype=jnp.bool_)
        if any(type(item) is not int for item in untyped):
            raise ValueError(f"{name} must contain only JSON integers")
        integer_values = cast(list[int], untyped)
        minimum = 0 if expected == np.dtype(np.uint32) else -_INT32_MAX - 1
        maximum = _UINT32_MAX if expected == np.dtype(np.uint32) else _INT32_MAX
        if any(item < minimum or item > maximum for item in integer_values):
            raise ValueError(f"{name} contains an out-of-range integer")
        output_dtype = jnp.uint32 if expected == np.dtype(np.uint32) else jnp.int32
        return jnp.asarray(value, dtype=output_dtype)

    @staticmethod
    def _checkpoint_counter(value: object, *, name: str, uint32: bool = False) -> Array:
        if type(value) is not int:
            raise ValueError(f"{name} must be a JSON integer")
        maximum = _UINT32_MAX if uint32 else _INT32_MAX
        if not 0 <= value <= maximum:
            raise ValueError(f"{name} is outside its counter range")
        return jnp.asarray(value, dtype=jnp.uint32 if uint32 else jnp.int32)

    @classmethod
    def _entries_from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        *,
        name: str,
        capacity: int,
        observation_dim: int,
        action_dim: int,
    ) -> ReplayEntries:
        expected = {*_ENTRY_FLOAT_FIELDS, *_ENTRY_BOOL_FIELDS, *_ENTRY_INT_FIELDS}
        if set(payload) != expected:
            raise ValueError(f"{name} fields do not match the replay entry schema")
        shapes = {
            "observations": (capacity, observation_dim),
            "next_observations": (capacity, observation_dim),
            "actions": (capacity, action_dim),
            **{
                field: (capacity,)
                for field in (
                    *_ENTRY_FLOAT_FIELDS[3:],
                    *_ENTRY_BOOL_FIELDS,
                    *_ENTRY_INT_FIELDS,
                )
            },
        }
        arrays: dict[str, Array] = {}
        for field in _ENTRY_FLOAT_FIELDS:
            arrays[field] = cls._checkpoint_array(
                payload[field],
                name=f"{name}.{field}",
                shape=shapes[field],
                dtype=np.float32,
            )
        for field in _ENTRY_BOOL_FIELDS:
            arrays[field] = cls._checkpoint_array(
                payload[field],
                name=f"{name}.{field}",
                shape=shapes[field],
                dtype=np.bool_,
            )
        for field in _ENTRY_INT_FIELDS:
            arrays[field] = cls._checkpoint_array(
                payload[field],
                name=f"{name}.{field}",
                shape=shapes[field],
                dtype=np.int32,
            )
        return ReplayEntries(**arrays)

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> tuple[DualReplayMemory, DualReplayState]:
        """Strictly restore a memory/state pair from a v1 checkpoint payload."""
        expected = {
            "schema",
            "mechanism_status",
            "memory",
            "config_digest",
            "state",
            "state_digest",
        }
        if set(payload) != expected:
            raise ValueError("checkpoint fields do not match the dual replay v1 schema")
        if payload.get("schema") != DUAL_REPLAY_CHECKPOINT_SCHEMA:
            raise ValueError("unexpected dual replay checkpoint schema")
        if payload.get("mechanism_status") != MECHANISM_STATUS:
            raise ValueError("dual replay checkpoint must remain mechanism-only")
        memory_payload = payload.get("memory")
        state_payload = payload.get("state")
        if not isinstance(memory_payload, Mapping) or not isinstance(state_payload, Mapping):
            raise ValueError("checkpoint memory and state must be mappings")
        config_digest = payload.get("config_digest")
        state_digest = payload.get("state_digest")
        if type(config_digest) is not str or config_digest != _payload_digest(memory_payload):
            raise ValueError("dual replay checkpoint config digest mismatch")
        if type(state_digest) is not str or state_digest != _payload_digest(state_payload):
            raise ValueError("dual replay checkpoint state digest mismatch")
        memory = cls.from_config(memory_payload)
        cfg = memory.config
        expected_state = {
            "short_term",
            "long_term",
            "rng_key_data",
            *_STATE_COUNTER_FIELDS,
            "persistent_bytes",
        }
        if set(state_payload) != expected_state:
            raise ValueError("checkpoint state fields do not match the replay state schema")
        short_payload = state_payload.get("short_term")
        long_payload = state_payload.get("long_term")
        if not isinstance(short_payload, Mapping) or not isinstance(long_payload, Mapping):
            raise ValueError("checkpoint replay strata must be mappings")
        short_term = cls._entries_from_checkpoint(
            short_payload,
            name="state.short_term",
            capacity=cfg.short_term_capacity,
            observation_dim=cfg.observation_dim,
            action_dim=cfg.action_dim,
        )
        long_term = cls._entries_from_checkpoint(
            long_payload,
            name="state.long_term",
            capacity=cfg.long_term_capacity,
            observation_dim=cfg.observation_dim,
            action_dim=cfg.action_dim,
        )
        state = DualReplayState(
            short_term=short_term,
            long_term=long_term,
            rng_key_data=cls._checkpoint_array(
                state_payload["rng_key_data"],
                name="state.rng_key_data",
                shape=(2,),
                dtype=np.uint32,
            ),
            **{
                field: cls._checkpoint_counter(state_payload[field], name=f"state.{field}")
                for field in _STATE_COUNTER_FIELDS
            },
            persistent_bytes=cls._checkpoint_counter(
                state_payload["persistent_bytes"],
                name="state.persistent_bytes",
                uint32=True,
            ),
        )
        memory.validate_state(state)
        return memory, state


__all__ = [
    "AleatoricControl",
    "DUAL_REPLAY_CHECKPOINT_SCHEMA",
    "DUAL_REPLAY_CONFIG_SCHEMA",
    "DualReplayConfig",
    "DualReplayMemory",
    "DualReplayResourceAccounting",
    "DualReplayState",
    "LONG_TERM_STRATUM",
    "LongTermPolicy",
    "MECHANISM_STATUS",
    "ReplayBatch",
    "ReplayEntries",
    "ReplayOutcome",
    "ReplayPrediction",
    "ReplaySampleResult",
    "ReplayTransition",
    "ReplayWriteResult",
    "ReservoirSelection",
    "SHORT_TERM_STRATUM",
    "reservoir_selection",
]
