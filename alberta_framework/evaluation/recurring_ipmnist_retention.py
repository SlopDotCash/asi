"""Threshold-free recurring-IPMNIST retention diagnostics.

The existing IPMNIST screening lane reports predict-before-update accuracy and
same-example, post-update plasticity on a sequence of unique permutations.
Those measurements describe online prediction and immediate adaptation, but a
one-pass schedule cannot directly measure whether a previously learned
permutation survived later interference.

This module supplies the smallest strict recurrence contract needed to add
that missing diagnostic surface.  The learner-visible trace contains only one
binary predict-before-update outcome and one post-update plasticity value per
observation.  Permutation identities, zero-based exposure indices, and frozen
sentinel correctness remain evaluator-only.  The fixed A/B/A schedule requires
an evaluator-owned sentinel probe for every permutation seen so far at every
phase boundary.  In particular, the probes after B and before the second A
measure A directly without allowing A updates to mask forgetting.

The resulting measurements are descriptive and development-only.  They apply
no performance thresholds, make no retention or anti-forgetting claim, and
cannot promote scientific evidence.  There is deliberately no runner, seed,
artifact writer, or command-line entry point here; a later MNIST adapter can
construct this contract from its already-ordered online loop.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import operator
import re
from collections.abc import Mapping, Sequence
from typing import SupportsIndex, cast

import numpy as np

RECURRING_IPMNIST_PROTOCOL_SCHEMA = "alberta.recurring-ipmnist-retention.protocol.v1"
RECURRING_IPMNIST_TRACE_SCHEMA = "alberta.recurring-ipmnist-retention.trace.v1"
RECURRING_IPMNIST_REPORT_SCHEMA = "alberta.recurring-ipmnist-retention.report.v1"
DEVELOPMENT_STATUS = "development-only-not-assessed"

LEARNER_VISIBLE_TRACE_FIELDS = (
    "pre_update_online_accuracy",
    "post_update_one_step_plasticity",
)
EVALUATOR_ONLY_FIELDS = (
    "phase_index",
    "permutation_id",
    "permutation_sha256",
    "exposure_index",
    "sentinel_set_id",
    "sentinel_set_sha256",
    "sentinel_correctness",
)

METRIC_DEFINITIONS: Mapping[str, str] = {
    "pre_revisit_sentinel_accuracy": (
        "accuracy on the recurring permutation's fixed evaluator-only sentinel set from "
        "the frozen learner state after the intervening phase and before any revisit update"
    ),
    "retention_change_from_acquisition": (
        "pre-revisit sentinel accuracy minus first-acquisition-end sentinel accuracy"
    ),
    "peak_to_revisit_forgetting": (
        "non-negative peak sentinel accuracy before revisit minus the pre-revisit score"
    ),
    "revisit_recovery": ("revisit-end sentinel accuracy minus pre-revisit sentinel accuracy"),
    "relearning_savings_mistakes": (
        "first-exposure leading-window mistakes minus revisit leading-window mistakes"
    ),
    "relearning_savings_accuracy": (
        "revisit leading-window pre-update accuracy minus first-exposure leading-window "
        "pre-update accuracy"
    ),
    "one_step_plasticity": (
        "the screening lane's clipped same-example post-update loss improvement; this is an "
        "immediate adaptation diagnostic and is not a retention measurement"
    ),
}

LIMITATIONS = (
    "one fixed A/B/A recurrence is a development diagnostic, not external validation",
    "sentinel cases and their ordered digests are evaluator supplied and not calibrated here",
    "state hashes detect declared probe mutation but cannot audit hidden side effects",
    "relearning savings can coexist with poor direct retention and is not proof of memory",
    "one-step plasticity measures immediate same-example improvement, not retained knowledge",
    "no thresholds, matched baselines, uncertainty estimates, or scientific promotion apply",
)

_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _identifier(value: object, *, name: str, versioned: bool = False) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical lowercase identifier")
    if versioned and ".v" not in value:
        raise ValueError(f"{name} must be versioned")
    return value


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


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


def _nonnegative_int(value: object, *, name: str, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be a non-negative integer in [0, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not 0 <= canonical <= maximum:
        raise ValueError(f"{name} must be a non-negative integer in [0, {maximum}]")
    return canonical


def _positive_int(value: object, *, name: str, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be a positive integer in [1, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not 1 <= canonical <= maximum:
        raise ValueError(f"{name} must be a positive integer in [1, {maximum}]")
    return canonical


def _require_int32_workload(name: str, value: int) -> None:
    if not 1 <= value <= _INT32_MAX:
        raise ValueError(f"derived {name} must fit in signed int32")


def _unit_float(value: object, *, name: str, binary: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        qualifier = "finite binary" if binary else "finite in [0, 1]"
        raise ValueError(f"{name} must be {qualifier}")
    if binary and value not in {0.0, 1.0}:
        raise ValueError(f"{name} must be binary (0.0 or 1.0)")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _signed_int(value: object, *, name: str, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be a signed integer in [{-maximum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not -maximum <= canonical <= maximum:
        raise ValueError(f"{name} must be a signed integer in [{-maximum}, {maximum}]")
    return canonical


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class SentinelProbeBinding:
    """Frozen evaluator-owned permutation and ordered sentinel-set identity.

    ``permutation_sha256`` binds the complete integer pixel permutation.
    ``sentinel_set_sha256`` binds the ordered example identities, labels, and
    transformed inputs used only for evaluation.  A repeated permutation has
    exactly one binding, so its sentinel cases cannot change at revisit time.
    """

    permutation_id: str
    permutation_sha256: str
    sentinel_set_id: str
    sentinel_set_sha256: str
    sentinel_case_count: int

    def __post_init__(self) -> None:
        _identifier(self.permutation_id, name="permutation_id", versioned=True)
        _sha256(self.permutation_sha256, name="permutation_sha256")
        _identifier(self.sentinel_set_id, name="sentinel_set_id", versioned=True)
        _sha256(self.sentinel_set_sha256, name="sentinel_set_sha256")
        object.__setattr__(
            self,
            "sentinel_case_count",
            _positive_int(self.sentinel_case_count, name="sentinel_case_count"),
        )

    def to_config(self) -> dict[str, object]:
        return {
            "permutation_id": self.permutation_id,
            "permutation_sha256": self.permutation_sha256,
            "sentinel_set_id": self.sentinel_set_id,
            "sentinel_set_sha256": self.sentinel_set_sha256,
            "sentinel_case_count": self.sentinel_case_count,
        }


@dataclasses.dataclass(frozen=True)
class RecurringIPMNISTPhase:
    """One evaluator-owned contiguous phase in the fixed A/B/A schedule."""

    phase_index: int
    start_step: int
    length: int
    permutation_id: str
    exposure_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase_index",
            _nonnegative_int(self.phase_index, name="phase_index"),
        )
        object.__setattr__(
            self,
            "start_step",
            _nonnegative_int(self.start_step, name="start_step"),
        )
        object.__setattr__(
            self,
            "length",
            _positive_int(self.length, name="length"),
        )
        if self.start_step > _INT32_MAX - self.length:
            raise ValueError("stop_step must fit signed int32")
        _identifier(self.permutation_id, name="permutation_id", versioned=True)
        object.__setattr__(
            self,
            "exposure_index",
            _nonnegative_int(self.exposure_index, name="exposure_index"),
        )

    @property
    def stop_step(self) -> int:
        """Exclusive end and frozen-probe checkpoint for this phase."""
        return self.start_step + self.length

    def to_config(self) -> dict[str, object]:
        return {
            "phase_index": self.phase_index,
            "start_step": self.start_step,
            "length": self.length,
            "permutation_id": self.permutation_id,
            "exposure_index": self.exposure_index,
        }


@dataclasses.dataclass(frozen=True)
class SentinelProbeRequirement:
    """One mandatory phase-boundary probe derived entirely from the protocol."""

    phase_index: int
    checkpoint_step: int
    permutation_id: str
    permutation_sha256: str
    exposure_index: int
    sentinel_set_id: str
    sentinel_set_sha256: str
    sentinel_case_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase_index",
            _nonnegative_int(self.phase_index, name="phase_index"),
        )
        object.__setattr__(
            self,
            "checkpoint_step",
            _positive_int(self.checkpoint_step, name="checkpoint_step"),
        )
        _identifier(self.permutation_id, name="permutation_id", versioned=True)
        _sha256(self.permutation_sha256, name="permutation_sha256")
        object.__setattr__(
            self,
            "exposure_index",
            _nonnegative_int(self.exposure_index, name="exposure_index"),
        )
        _identifier(self.sentinel_set_id, name="sentinel_set_id", versioned=True)
        _sha256(self.sentinel_set_sha256, name="sentinel_set_sha256")
        object.__setattr__(
            self,
            "sentinel_case_count",
            _positive_int(self.sentinel_case_count, name="sentinel_case_count"),
        )

    @property
    def ordering_key(self) -> tuple[int, int, str, int]:
        return (
            self.phase_index,
            self.checkpoint_step,
            self.permutation_id,
            self.exposure_index,
        )

    @property
    def frozen_binding(self) -> tuple[str, str, str, int]:
        return (
            self.permutation_sha256,
            self.sentinel_set_id,
            self.sentinel_set_sha256,
            self.sentinel_case_count,
        )

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RecurringIPMNISTProtocol:
    """Strict evaluator-owned A/B/A schedule and probe manifest.

    Exposure indices are zero based and must equal the number of prior phases
    with that exact permutation identity.  The first and recurring A phases
    must have equal lengths so their leading adaptation windows are directly
    comparable.
    """

    protocol_id: str
    phases: tuple[RecurringIPMNISTPhase, ...]
    sentinel_bindings: tuple[SentinelProbeBinding, ...]
    relearning_window: int

    def __post_init__(self) -> None:
        _identifier(self.protocol_id, name="protocol_id", versioned=True)
        if type(self.phases) is not tuple or len(self.phases) != 3:
            raise ValueError("phases must be an exact three-phase A/B/A tuple")
        if any(type(phase) is not RecurringIPMNISTPhase for phase in self.phases):
            raise TypeError("phases must contain RecurringIPMNISTPhase values")

        expected_start = 0
        exposure_counts: dict[str, int] = {}
        for index, phase in enumerate(self.phases):
            if phase.phase_index != index:
                raise ValueError("phase_index values must be contiguous from zero")
            if phase.start_step != expected_start:
                raise ValueError("phase start_step values must be contiguous from zero")
            expected_exposure = exposure_counts.get(phase.permutation_id, 0)
            if phase.exposure_index != expected_exposure:
                raise ValueError(
                    f"phase {index} exposure_index must be {expected_exposure} for "
                    f"{phase.permutation_id}"
                )
            exposure_counts[phase.permutation_id] = expected_exposure + 1
            expected_start = phase.stop_step

        identities = tuple(phase.permutation_id for phase in self.phases)
        if identities[0] != identities[2] or identities[0] == identities[1]:
            raise ValueError("phase permutation identities must form exact A/B/A recurrence")
        if self.phases[0].length != self.phases[2].length:
            raise ValueError("first and recurring A phases must have equal lengths")

        if type(self.sentinel_bindings) is not tuple or len(self.sentinel_bindings) != 2:
            raise ValueError("sentinel_bindings must contain exactly the A and B bindings")
        if any(type(binding) is not SentinelProbeBinding for binding in self.sentinel_bindings):
            raise TypeError("sentinel_bindings must contain SentinelProbeBinding values")
        first_occurrence_order = (identities[0], identities[1])
        binding_order = tuple(binding.permutation_id for binding in self.sentinel_bindings)
        if binding_order != first_occurrence_order:
            raise ValueError(
                "sentinel_bindings must exactly follow first permutation occurrence order"
            )
        permutation_digests = tuple(
            binding.permutation_sha256 for binding in self.sentinel_bindings
        )
        if len(set(permutation_digests)) != len(permutation_digests):
            raise ValueError("A and B must bind distinct pixel permutation digests")
        sentinel_digests = tuple(binding.sentinel_set_sha256 for binding in self.sentinel_bindings)
        if len(set(sentinel_digests)) != len(sentinel_digests):
            raise ValueError("A and B must bind distinct sentinel set digests")
        sentinel_ids = tuple(binding.sentinel_set_id for binding in self.sentinel_bindings)
        if len(set(sentinel_ids)) != len(sentinel_ids):
            raise ValueError("A and B must bind distinct sentinel set identities")

        object.__setattr__(
            self,
            "relearning_window",
            _positive_int(self.relearning_window, name="relearning_window"),
        )
        if self.relearning_window > self.phases[0].length:
            raise ValueError("relearning_window cannot exceed either equal-length A phase")
        _require_int32_workload("trace scalar count", 2 * self.trace_length)
        binding_counts = {
            binding.permutation_id: binding.sentinel_case_count
            for binding in self.sentinel_bindings
        }
        sentinel_evaluations = sum(
            binding_counts[requirement.permutation_id]
            for requirement in self.required_probe_snapshots
        )
        _require_int32_workload("sentinel probe evaluations", sentinel_evaluations)

    @property
    def trace_length(self) -> int:
        """Exact number of predict-before-update observations required."""
        return self.phases[-1].stop_step

    @property
    def required_probe_snapshots(self) -> tuple[SentinelProbeRequirement, ...]:
        """Return the exhaustive, ordered phase-boundary probe manifest.

        Every permutation seen so far is probed after every completed phase.
        This policy makes omission post hoc impossible and exposes both A's
        pre-revisit retention and B's survival through the final A phase.
        """
        bindings = {binding.permutation_id: binding for binding in self.sentinel_bindings}
        seen_order: list[str] = []
        latest_exposure: dict[str, int] = {}
        requirements: list[SentinelProbeRequirement] = []
        for phase in self.phases:
            if phase.permutation_id not in latest_exposure:
                seen_order.append(phase.permutation_id)
            latest_exposure[phase.permutation_id] = phase.exposure_index
            for permutation_id in seen_order:
                binding = bindings[permutation_id]
                requirements.append(
                    SentinelProbeRequirement(
                        phase_index=phase.phase_index,
                        checkpoint_step=phase.stop_step,
                        permutation_id=permutation_id,
                        permutation_sha256=binding.permutation_sha256,
                        exposure_index=latest_exposure[permutation_id],
                        sentinel_set_id=binding.sentinel_set_id,
                        sentinel_set_sha256=binding.sentinel_set_sha256,
                        sentinel_case_count=binding.sentinel_case_count,
                    )
                )
        return tuple(requirements)

    def to_config(self) -> dict[str, object]:
        return {
            "schema": RECURRING_IPMNIST_PROTOCOL_SCHEMA,
            "type": type(self).__name__,
            "schedule": "A/B/A",
            "exposure_index_origin": 0,
            "probe_ownership": "evaluator-only-frozen",
            "probe_timing": "after-phase-before-next-online-update",
            "learner_visible_trace_fields": list(LEARNER_VISIBLE_TRACE_FIELDS),
            "evaluator_only_fields": list(EVALUATOR_ONLY_FIELDS),
            "protocol_id": self.protocol_id,
            "phases": [phase.to_config() for phase in self.phases],
            "sentinel_bindings": [binding.to_config() for binding in self.sentinel_bindings],
            "required_probe_snapshots": [
                requirement.to_config() for requirement in self.required_probe_snapshots
            ],
            "relearning_window": self.relearning_window,
        }


@dataclasses.dataclass(frozen=True)
class RecurringIPMNISTTrace:
    """Task-ID-free per-observation metrics in exact online processing order.

    Accuracy is the binary prediction outcome emitted before the learner sees
    the corresponding label update.  Plasticity is emitted after that update
    from the same example, matching the current IPMNIST screening semantics.
    The evaluator joins positions to phases only after the run.
    """

    pre_update_online_accuracy: tuple[float, ...]
    post_update_one_step_plasticity: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.pre_update_online_accuracy) is not tuple:
            raise TypeError("pre_update_online_accuracy must be an exact tuple")
        if type(self.post_update_one_step_plasticity) is not tuple:
            raise TypeError("post_update_one_step_plasticity must be an exact tuple")
        if not self.pre_update_online_accuracy:
            raise ValueError("the online trace must be non-empty")
        if len(self.pre_update_online_accuracy) != len(self.post_update_one_step_plasticity):
            raise ValueError("accuracy and plasticity traces must have equal length")
        _require_int32_workload(
            "online trace scalar count", 2 * len(self.pre_update_online_accuracy)
        )
        for index, value in enumerate(self.pre_update_online_accuracy):
            _unit_float(value, name=f"pre_update_online_accuracy[{index}]", binary=True)
        for index, value in enumerate(self.post_update_one_step_plasticity):
            _unit_float(value, name=f"post_update_one_step_plasticity[{index}]")

    def to_config(self) -> dict[str, object]:
        return {
            "schema": RECURRING_IPMNIST_TRACE_SCHEMA,
            "ordering": "predict-before-update",
            "learner_visible_fields": list(LEARNER_VISIBLE_TRACE_FIELDS),
            "pre_update_online_accuracy": list(self.pre_update_online_accuracy),
            "post_update_one_step_plasticity": list(self.post_update_one_step_plasticity),
        }


@dataclasses.dataclass(frozen=True)
class SentinelProbeSnapshot:
    """Correctness vector from one required non-learning sentinel evaluation."""

    phase_index: int
    checkpoint_step: int
    permutation_id: str
    permutation_sha256: str
    exposure_index: int
    sentinel_set_id: str
    sentinel_set_sha256: str
    learner_state_sha256_before: str
    learner_state_sha256_after: str
    correctness: tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase_index",
            _nonnegative_int(self.phase_index, name="phase_index"),
        )
        object.__setattr__(
            self,
            "checkpoint_step",
            _positive_int(self.checkpoint_step, name="checkpoint_step"),
        )
        _identifier(self.permutation_id, name="permutation_id", versioned=True)
        _sha256(self.permutation_sha256, name="permutation_sha256")
        object.__setattr__(
            self,
            "exposure_index",
            _nonnegative_int(self.exposure_index, name="exposure_index"),
        )
        _identifier(self.sentinel_set_id, name="sentinel_set_id", versioned=True)
        _sha256(self.sentinel_set_sha256, name="sentinel_set_sha256")
        before = _sha256(
            self.learner_state_sha256_before,
            name="learner_state_sha256_before",
        )
        after = _sha256(
            self.learner_state_sha256_after,
            name="learner_state_sha256_after",
        )
        if before != after:
            raise ValueError("a frozen sentinel probe must not mutate learner state")
        if type(self.correctness) is not tuple or not self.correctness:
            raise ValueError("correctness must be a non-empty exact tuple")
        if any(type(value) is not bool for value in self.correctness):
            raise ValueError("correctness must contain only canonical booleans")
        _require_int32_workload("sentinel correctness cases", len(self.correctness))

    @classmethod
    def from_requirement(
        cls,
        requirement: SentinelProbeRequirement,
        *,
        learner_state_sha256_before: str,
        learner_state_sha256_after: str,
        correctness: tuple[bool, ...],
    ) -> SentinelProbeSnapshot:
        """Construct a snapshot without manually copying evaluator bindings."""
        if type(requirement) is not SentinelProbeRequirement:
            raise TypeError("requirement must be SentinelProbeRequirement")
        return cls(
            phase_index=requirement.phase_index,
            checkpoint_step=requirement.checkpoint_step,
            permutation_id=requirement.permutation_id,
            permutation_sha256=requirement.permutation_sha256,
            exposure_index=requirement.exposure_index,
            sentinel_set_id=requirement.sentinel_set_id,
            sentinel_set_sha256=requirement.sentinel_set_sha256,
            learner_state_sha256_before=learner_state_sha256_before,
            learner_state_sha256_after=learner_state_sha256_after,
            correctness=correctness,
        )

    @property
    def ordering_key(self) -> tuple[int, int, str, int]:
        return (
            self.phase_index,
            self.checkpoint_step,
            self.permutation_id,
            self.exposure_index,
        )

    @property
    def frozen_binding(self) -> tuple[str, str, str, int]:
        return (
            self.permutation_sha256,
            self.sentinel_set_id,
            self.sentinel_set_sha256,
            len(self.correctness),
        )

    @property
    def accuracy(self) -> float:
        return sum(self.correctness) / len(self.correctness)

    def to_config(self) -> dict[str, object]:
        return {
            "phase_index": self.phase_index,
            "checkpoint_step": self.checkpoint_step,
            "permutation_id": self.permutation_id,
            "permutation_sha256": self.permutation_sha256,
            "exposure_index": self.exposure_index,
            "sentinel_set_id": self.sentinel_set_id,
            "sentinel_set_sha256": self.sentinel_set_sha256,
            "learner_state_sha256_before": self.learner_state_sha256_before,
            "learner_state_sha256_after": self.learner_state_sha256_after,
            "correctness": list(self.correctness),
        }


@dataclasses.dataclass(frozen=True)
class PhaseOnlineSummary:
    """Descriptive online and immediate-plasticity metrics for one phase."""

    phase_index: int
    permutation_id: str
    exposure_index: int
    observation_count: int
    mean_pre_update_online_accuracy: float
    online_mistakes: int
    mean_post_update_one_step_plasticity: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase_index",
            _nonnegative_int(self.phase_index, name="phase_index"),
        )
        _identifier(self.permutation_id, name="permutation_id", versioned=True)
        object.__setattr__(
            self,
            "exposure_index",
            _nonnegative_int(self.exposure_index, name="exposure_index"),
        )
        object.__setattr__(
            self,
            "observation_count",
            _positive_int(self.observation_count, name="observation_count"),
        )
        object.__setattr__(
            self,
            "mean_pre_update_online_accuracy",
            _unit_float(
                self.mean_pre_update_online_accuracy,
                name="mean_pre_update_online_accuracy",
            ),
        )
        object.__setattr__(
            self,
            "online_mistakes",
            _nonnegative_int(self.online_mistakes, name="online_mistakes"),
        )
        object.__setattr__(
            self,
            "mean_post_update_one_step_plasticity",
            _unit_float(
                self.mean_post_update_one_step_plasticity,
                name="mean_post_update_one_step_plasticity",
            ),
        )

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SentinelProbeScore:
    """Derived score retaining its exact protocol and frozen-state bindings."""

    phase_index: int
    checkpoint_step: int
    permutation_id: str
    exposure_index: int
    sentinel_set_id: str
    sentinel_case_count: int
    correct_count: int
    accuracy: float
    learner_state_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase_index",
            _nonnegative_int(self.phase_index, name="phase_index"),
        )
        object.__setattr__(
            self,
            "checkpoint_step",
            _positive_int(self.checkpoint_step, name="checkpoint_step"),
        )
        _identifier(self.permutation_id, name="permutation_id", versioned=True)
        object.__setattr__(
            self,
            "exposure_index",
            _nonnegative_int(self.exposure_index, name="exposure_index"),
        )
        _identifier(self.sentinel_set_id, name="sentinel_set_id", versioned=True)
        object.__setattr__(
            self,
            "sentinel_case_count",
            _positive_int(self.sentinel_case_count, name="sentinel_case_count"),
        )
        object.__setattr__(
            self,
            "correct_count",
            _nonnegative_int(self.correct_count, name="correct_count"),
        )
        object.__setattr__(
            self,
            "accuracy",
            _unit_float(self.accuracy, name="accuracy"),
        )
        _sha256(self.learner_state_sha256, name="learner_state_sha256")

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RecurrenceRetentionMetrics:
    """Direct retention, forgetting, recovery, savings, and plasticity for A."""

    permutation_id: str
    first_exposure_index: int
    revisit_exposure_index: int
    relearning_window: int
    acquisition_end_sentinel_accuracy: float
    peak_before_revisit_sentinel_accuracy: float
    pre_revisit_sentinel_accuracy: float
    retention_change_from_acquisition: float
    peak_to_revisit_forgetting: float
    revisit_end_sentinel_accuracy: float
    revisit_recovery: float
    first_exposure_leading_pre_update_accuracy: float
    revisit_leading_pre_update_accuracy: float
    first_exposure_leading_mistakes: int
    revisit_leading_mistakes: int
    relearning_savings_mistakes: int
    relearning_savings_accuracy: float
    first_exposure_leading_one_step_plasticity: float
    revisit_leading_one_step_plasticity: float

    def __post_init__(self) -> None:
        _identifier(self.permutation_id, name="permutation_id", versioned=True)
        object.__setattr__(
            self,
            "first_exposure_index",
            _nonnegative_int(self.first_exposure_index, name="first_exposure_index"),
        )
        object.__setattr__(
            self,
            "revisit_exposure_index",
            _nonnegative_int(self.revisit_exposure_index, name="revisit_exposure_index"),
        )
        object.__setattr__(
            self,
            "relearning_window",
            _positive_int(self.relearning_window, name="relearning_window"),
        )
        for name in (
            "acquisition_end_sentinel_accuracy",
            "peak_before_revisit_sentinel_accuracy",
            "pre_revisit_sentinel_accuracy",
            "revisit_end_sentinel_accuracy",
            "first_exposure_leading_pre_update_accuracy",
            "revisit_leading_pre_update_accuracy",
            "first_exposure_leading_one_step_plasticity",
            "revisit_leading_one_step_plasticity",
        ):
            object.__setattr__(self, name, _unit_float(getattr(self, name), name=name))
        for name in (
            "retention_change_from_acquisition",
            "peak_to_revisit_forgetting",
            "revisit_recovery",
            "relearning_savings_accuracy",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name=name))
        for name in (
            "first_exposure_leading_mistakes",
            "revisit_leading_mistakes",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "relearning_savings_mistakes",
            _signed_int(self.relearning_savings_mistakes, name="relearning_savings_mistakes"),
        )

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RecurringIPMNISTRetentionReport:
    """Validated descriptive report with no acceptance decision."""

    protocol: RecurringIPMNISTProtocol
    protocol_sha256: str
    trace_sha256: str
    sentinel_snapshots_sha256: str
    phase_summaries: tuple[PhaseOnlineSummary, ...]
    sentinel_scores: tuple[SentinelProbeScore, ...]
    recurrence: RecurrenceRetentionMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, RecurringIPMNISTProtocol):
            raise TypeError("protocol must be a RecurringIPMNISTProtocol")
        _sha256(self.protocol_sha256, name="protocol_sha256")
        _sha256(self.trace_sha256, name="trace_sha256")
        _sha256(self.sentinel_snapshots_sha256, name="sentinel_snapshots_sha256")
        if type(self.phase_summaries) is not tuple or not all(
            type(s) is PhaseOnlineSummary for s in self.phase_summaries
        ):
            raise TypeError("phase_summaries must be a tuple of PhaseOnlineSummary")
        if type(self.sentinel_scores) is not tuple or not all(
            type(s) is SentinelProbeScore for s in self.sentinel_scores
        ):
            raise TypeError("sentinel_scores must be a tuple of SentinelProbeScore")
        if not isinstance(self.recurrence, RecurrenceRetentionMetrics):
            raise TypeError("recurrence must be a RecurrenceRetentionMetrics")

    def to_config(self) -> dict[str, object]:
        return {
            "schema": RECURRING_IPMNIST_REPORT_SCHEMA,
            "development_status": DEVELOPMENT_STATUS,
            "assessment_status": "not-assessed",
            "evidence_class": "development_recurring_ipmnist_diagnostic",
            "scientific_promotion_allowed": False,
            "performance_thresholds_applied": False,
            "retention_claimed": False,
            "catastrophic_forgetting_absence_claimed": False,
            "protocol": self.protocol.to_config(),
            "protocol_sha256": self.protocol_sha256,
            "trace_sha256": self.trace_sha256,
            "sentinel_snapshots_sha256": self.sentinel_snapshots_sha256,
            "metric_definitions": dict(METRIC_DEFINITIONS),
            "phase_summaries": [item.to_config() for item in self.phase_summaries],
            "sentinel_scores": [item.to_config() for item in self.sentinel_scores],
            "recurrence": self.recurrence.to_config(),
            "limitations": list(LIMITATIONS),
        }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _validate_sentinel_snapshots(
    protocol: RecurringIPMNISTProtocol,
    snapshots: Sequence[SentinelProbeSnapshot],
) -> tuple[SentinelProbeSnapshot, ...]:
    resolved = tuple(snapshots)
    expected = protocol.required_probe_snapshots
    if len(resolved) != len(expected):
        raise ValueError("sentinel snapshots must appear once each in the exact required order")
    checkpoint_states: dict[int, str] = {}
    for index, (snapshot, requirement) in enumerate(zip(resolved, expected, strict=True)):
        if type(snapshot) is not SentinelProbeSnapshot:
            raise TypeError(f"sentinel_snapshots[{index}] must be SentinelProbeSnapshot")
        if snapshot.ordering_key != requirement.ordering_key:
            raise ValueError("sentinel snapshots must appear once each in the exact required order")
        if snapshot.frozen_binding != requirement.frozen_binding:
            if len(snapshot.correctness) != requirement.sentinel_case_count:
                raise ValueError(
                    f"sentinel snapshot {index} correctness case count does not match "
                    "its frozen requirement"
                )
            raise ValueError(f"sentinel snapshot {index} does not match its frozen requirement")
        prior_state = checkpoint_states.setdefault(
            snapshot.checkpoint_step,
            snapshot.learner_state_sha256_before,
        )
        if prior_state != snapshot.learner_state_sha256_before:
            raise ValueError("all sentinel probes at one checkpoint must use one frozen state")
    states_by_digest: dict[str, list[int]] = {}
    for checkpoint_step, digest in checkpoint_states.items():
        states_by_digest.setdefault(digest, []).append(checkpoint_step)
    for shared_steps in states_by_digest.values():
        if len(shared_steps) > 1:
            raise ValueError(
                "sentinel probes at different checkpoints must use distinct frozen states; "
                f"checkpoint steps {sorted(shared_steps)} all declare one learner state"
            )
    return resolved


def _phase_summaries(
    protocol: RecurringIPMNISTProtocol,
    trace: RecurringIPMNISTTrace,
) -> tuple[PhaseOnlineSummary, ...]:
    summaries: list[PhaseOnlineSummary] = []
    for phase in protocol.phases:
        accuracies = trace.pre_update_online_accuracy[phase.start_step : phase.stop_step]
        plasticities = trace.post_update_one_step_plasticity[phase.start_step : phase.stop_step]
        summaries.append(
            PhaseOnlineSummary(
                phase_index=phase.phase_index,
                permutation_id=phase.permutation_id,
                exposure_index=phase.exposure_index,
                observation_count=phase.length,
                mean_pre_update_online_accuracy=_mean(accuracies),
                online_mistakes=sum(value == 0.0 for value in accuracies),
                mean_post_update_one_step_plasticity=_mean(plasticities),
            )
        )
    return tuple(summaries)


def _sentinel_scores(
    snapshots: Sequence[SentinelProbeSnapshot],
) -> tuple[SentinelProbeScore, ...]:
    return tuple(
        SentinelProbeScore(
            phase_index=snapshot.phase_index,
            checkpoint_step=snapshot.checkpoint_step,
            permutation_id=snapshot.permutation_id,
            exposure_index=snapshot.exposure_index,
            sentinel_set_id=snapshot.sentinel_set_id,
            sentinel_case_count=len(snapshot.correctness),
            correct_count=sum(snapshot.correctness),
            accuracy=snapshot.accuracy,
            learner_state_sha256=snapshot.learner_state_sha256_before,
        )
        for snapshot in snapshots
    )


def _recurrence_metrics(
    protocol: RecurringIPMNISTProtocol,
    trace: RecurringIPMNISTTrace,
    snapshots: Sequence[SentinelProbeSnapshot],
) -> RecurrenceRetentionMetrics:
    first_phase, _, revisit_phase = protocol.phases
    recurring_id = first_phase.permutation_id
    by_phase_and_permutation = {
        (snapshot.phase_index, snapshot.permutation_id): snapshot for snapshot in snapshots
    }
    acquisition = by_phase_and_permutation[(first_phase.phase_index, recurring_id)].accuracy
    pre_revisit = by_phase_and_permutation[(revisit_phase.phase_index - 1, recurring_id)].accuracy
    revisit_end = by_phase_and_permutation[(revisit_phase.phase_index, recurring_id)].accuracy
    before_revisit = [
        snapshot.accuracy
        for snapshot in snapshots
        if snapshot.permutation_id == recurring_id
        and first_phase.phase_index <= snapshot.phase_index < revisit_phase.phase_index
    ]
    peak_before_revisit = max(before_revisit)

    window = protocol.relearning_window
    first_accuracy = trace.pre_update_online_accuracy[
        first_phase.start_step : first_phase.start_step + window
    ]
    revisit_accuracy = trace.pre_update_online_accuracy[
        revisit_phase.start_step : revisit_phase.start_step + window
    ]
    first_plasticity = trace.post_update_one_step_plasticity[
        first_phase.start_step : first_phase.start_step + window
    ]
    revisit_plasticity = trace.post_update_one_step_plasticity[
        revisit_phase.start_step : revisit_phase.start_step + window
    ]
    first_mistakes = sum(value == 0.0 for value in first_accuracy)
    revisit_mistakes = sum(value == 0.0 for value in revisit_accuracy)

    return RecurrenceRetentionMetrics(
        permutation_id=recurring_id,
        first_exposure_index=first_phase.exposure_index,
        revisit_exposure_index=revisit_phase.exposure_index,
        relearning_window=window,
        acquisition_end_sentinel_accuracy=acquisition,
        peak_before_revisit_sentinel_accuracy=peak_before_revisit,
        pre_revisit_sentinel_accuracy=pre_revisit,
        retention_change_from_acquisition=pre_revisit - acquisition,
        peak_to_revisit_forgetting=max(0.0, peak_before_revisit - pre_revisit),
        revisit_end_sentinel_accuracy=revisit_end,
        revisit_recovery=revisit_end - pre_revisit,
        first_exposure_leading_pre_update_accuracy=_mean(first_accuracy),
        revisit_leading_pre_update_accuracy=_mean(revisit_accuracy),
        first_exposure_leading_mistakes=first_mistakes,
        revisit_leading_mistakes=revisit_mistakes,
        relearning_savings_mistakes=first_mistakes - revisit_mistakes,
        relearning_savings_accuracy=_mean(revisit_accuracy) - _mean(first_accuracy),
        first_exposure_leading_one_step_plasticity=_mean(first_plasticity),
        revisit_leading_one_step_plasticity=_mean(revisit_plasticity),
    )


def build_recurring_ipmnist_retention_report(
    *,
    protocol: RecurringIPMNISTProtocol,
    trace: RecurringIPMNISTTrace,
    sentinel_snapshots: Sequence[SentinelProbeSnapshot],
) -> RecurringIPMNISTRetentionReport:
    """Validate one complete A/B/A trace and derive threshold-free metrics."""
    if type(protocol) is not RecurringIPMNISTProtocol:
        raise TypeError("protocol must be RecurringIPMNISTProtocol")
    if type(trace) is not RecurringIPMNISTTrace:
        raise TypeError("trace must be RecurringIPMNISTTrace")
    if len(trace.pre_update_online_accuracy) != protocol.trace_length:
        raise ValueError(
            "the predict-before-update trace length must exactly match the A/B/A protocol"
        )
    snapshots = _validate_sentinel_snapshots(protocol, sentinel_snapshots)
    return RecurringIPMNISTRetentionReport(
        protocol=protocol,
        protocol_sha256=_digest(protocol.to_config()),
        trace_sha256=_digest(trace.to_config()),
        sentinel_snapshots_sha256=_digest([snapshot.to_config() for snapshot in snapshots]),
        phase_summaries=_phase_summaries(protocol, trace),
        sentinel_scores=_sentinel_scores(snapshots),
        recurrence=_recurrence_metrics(protocol, trace, snapshots),
    )


__all__ = [
    "DEVELOPMENT_STATUS",
    "EVALUATOR_ONLY_FIELDS",
    "LEARNER_VISIBLE_TRACE_FIELDS",
    "LIMITATIONS",
    "METRIC_DEFINITIONS",
    "RECURRING_IPMNIST_PROTOCOL_SCHEMA",
    "RECURRING_IPMNIST_REPORT_SCHEMA",
    "RECURRING_IPMNIST_TRACE_SCHEMA",
    "PhaseOnlineSummary",
    "RecurrenceRetentionMetrics",
    "RecurringIPMNISTPhase",
    "RecurringIPMNISTProtocol",
    "RecurringIPMNISTRetentionReport",
    "RecurringIPMNISTTrace",
    "SentinelProbeBinding",
    "SentinelProbeRequirement",
    "SentinelProbeScore",
    "SentinelProbeSnapshot",
    "build_recurring_ipmnist_retention_report",
]
