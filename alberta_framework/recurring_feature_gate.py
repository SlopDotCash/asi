"""Reproducible recurring pair-feature retention and forgetting probe.

This module isolates one narrow Step 2 question: can a fixed-budget online
learner retain three recurring pair products while evicting a fourth pair
that appeared once from its deployed bank?  A single learner state
experiences the uninterrupted sequence ``A, B, A, D, A, C, A, B, C``.
Predictions are scored before each update, and final scores use a separately
generated held-out set that never updates the learner.

The probe is deliberately small.  It uses Gaussian vectors, supplied task
identities, pair-product targets, and squared error.  Passing it is evidence
for pairwise active-bank allocation under the stated budgets; it is not
evidence that the obsolete pair was erased from the candidate archive, general
feature discovery, continual control, or completion of the Alberta Plan.
"""

from __future__ import annotations

import functools
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.typing as npt
from jax import Array

from alberta_framework._seed_validation import require_jax_seed, require_unique_jax_seeds
from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.interaction_features import (
    FixedBudgetInteractionLearner,
    InteractionFeatureState,
)

type Pair = tuple[int, int]
type TaskName = Literal["A", "B", "C", "D"]
type VariantName = Literal["retained", "no_retention"]

TASK_NAMES: tuple[TaskName, ...] = ("A", "B", "C", "D")
"""Output-head order used by held-out metrics."""

TASK_PAIRS: tuple[Pair, ...] = ((0, 1), (2, 3), (4, 5), (0, 5))
"""Pair target for each output head; D is the one-off obsolete target."""

CRITICAL_TASKS: tuple[TaskName, ...] = ("A", "B", "C")
PHASE_TASKS: tuple[TaskName, ...] = ("A", "B", "A", "D", "A", "C", "A", "B", "C")
DEVELOPMENT_SEEDS: tuple[int, ...] = tuple(range(30))
"""Seeds used while calibrating the canonical lifecycle configuration."""

EVIDENCE_SEEDS: tuple[int, ...] = tuple(range(30, 60))
"""Disjoint held-out seeds reserved for the frozen canonical protocol."""

PAIRWISE_PROBE_SCOPE = (
    "Pairwise Gaussian/L2 active-bank allocation probe only; the all-pairs candidate "
    "archive remains counted memory, and this does not establish general feature "
    "discovery, continual control, or Alberta Plan completion."
)

# Bound the exhaustive C(n, 2) candidate universe before constructing it.  The
# frozen protocol uses six inputs; 64 retains bounded noncanonical development
# diagnostics without permitting scalar-controlled pair-set explosions.
MAX_RECURRING_FEATURE_DIM = 64


def _require_builtin_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a built-in integer")
    if minimum is not None and maximum is not None:
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    elif minimum is not None and value < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive")
        raise ValueError(f"{name} must be >= {minimum}")
    elif maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _require_exact_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _require_task_name(name: str, value: object) -> TaskName:
    if type(value) is not str or value not in TASK_NAMES:
        raise ValueError(f"{name} must be an exact registered task name")
    return value


def _require_nmse(name: str, value: object) -> float:
    """Accept nonnegative exact floats, retaining ``inf`` as unrecovered evidence."""
    if type(value) is not float or math.isnan(value) or value < 0.0:
        raise ValueError(f"{name} must be a nonnegative exact float or positive infinity")
    return value


def _require_recovery_steps(name: str, value: object) -> int | None:
    if value is None:
        return None
    return _require_builtin_int(name, value, minimum=0, maximum=2**31 - 1)


def _require_pairs(name: str, value: object) -> tuple[Pair, ...]:
    if type(value) is not tuple or len(value) > 4096:
        raise ValueError(f"{name} must be a bounded exact tuple of pairs")
    pairs: list[Pair] = []
    for index, raw_pair in enumerate(value):
        if type(raw_pair) is not tuple or len(raw_pair) != 2:
            raise ValueError(f"{name}[{index}] must be an exact pair")
        left = _require_builtin_int(
            f"{name}[{index}][0]", raw_pair[0], minimum=0, maximum=2**31 - 1
        )
        right = _require_builtin_int(
            f"{name}[{index}][1]", raw_pair[1], minimum=0, maximum=2**31 - 1
        )
        if left >= right:
            raise ValueError(f"{name}[{index}] must be in increasing index order")
        pairs.append((left, right))
    return tuple(pairs)


def _preflight_seed_array_resources(
    *, feature_dim: int, output_heads: int, total_steps: int, heldout_samples: int
) -> None:
    scalar_count = total_steps * (feature_dim + output_heads + 2) + heldout_samples * (
        feature_dim + output_heads
    )
    byte_count = 4 * scalar_count
    if scalar_count > 2**31 - 1 or byte_count > 2**31 - 1:
        raise ValueError("per-seed arrays must fit signed-int32 scalar and byte resources")
    if byte_count > 256 * 1024 * 1024:
        raise ValueError("per-seed arrays must fit the 256 MiB resource budget")


@dataclass(frozen=True, slots=True)
class RecurringFeatureProtocol:
    """Fully specified stream and learner settings for the recurring probe."""

    feature_dim: int = 6
    output_heads: int = 4
    steps_per_phase: int = 400
    target_amplitude: float = 1.0
    active_pair_budget: int = 3
    candidate_pair_budget: int = 15
    step_size_output: float = 0.03
    utility_decay: float = 0.99
    retained_utility_decay: float = 0.999
    replacement_interval: int = 20
    min_feature_age: int = 40
    candidate_min_age: int = 40
    promotion_margin: float = 1.02
    promotion_blend: float = 1.0
    candidate_strategy: Literal["all_pairs"] = "all_pairs"
    utility_aggregation: Literal["max"] = "max"
    utility_task_balancing: Literal["active"] = "active"
    future_utility_mix: float = 0.5
    candidate_utility_retention_decay: float | None = None
    refresh_candidates: bool = False
    refresh_promoted_candidate: bool = False
    use_obgd: bool = True
    obgd_kappa: float = 2.0
    heldout_samples: int = 4096
    recovery_window: int = 40
    recovery_nmse_threshold: float = 0.1

    @property
    def total_steps(self) -> int:
        """Number of uninterrupted online updates."""
        return len(PHASE_TASKS) * self.steps_per_phase

    def validate(self) -> None:
        """Reject malformed protocols before allocating experiment arrays."""
        _require_builtin_int(
            "feature_dim",
            self.feature_dim,
            minimum=6,
            maximum=MAX_RECURRING_FEATURE_DIM,
        )
        _require_builtin_int("output_heads", self.output_heads)
        if self.output_heads != len(TASK_NAMES):
            raise ValueError(f"output_heads must be {len(TASK_NAMES)}")
        _require_builtin_int("steps_per_phase", self.steps_per_phase, minimum=1)
        _require_builtin_int("active_pair_budget", self.active_pair_budget, minimum=1)
        _require_builtin_int("candidate_pair_budget", self.candidate_pair_budget, minimum=1)
        _require_builtin_int("heldout_samples", self.heldout_samples, minimum=1)
        _require_builtin_int("replacement_interval", self.replacement_interval, minimum=0)
        _require_builtin_int("min_feature_age", self.min_feature_age, minimum=0)
        _require_builtin_int("candidate_min_age", self.candidate_min_age, minimum=0)
        _require_builtin_int(
            "recovery_window",
            self.recovery_window,
            minimum=1,
            maximum=self.steps_per_phase,
        )
        _preflight_seed_array_resources(
            feature_dim=self.feature_dim,
            output_heads=self.output_heads,
            total_steps=self.total_steps,
            heldout_samples=self.heldout_samples,
        )
        target_amplitude = validated_float32_scalar(
            "target_amplitude", self.target_amplitude, positive=True
        )
        recovery_nmse_threshold = validated_float32_scalar(
            "recovery_nmse_threshold", self.recovery_nmse_threshold, positive=True
        )
        step_size_output = validated_float32_scalar(
            "step_size_output", self.step_size_output, lower=0.0
        )
        utility_decay = validated_float32_scalar(
            "utility_decay",
            self.utility_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        retained_utility_decay = validated_float32_scalar(
            "retained_utility_decay",
            self.retained_utility_decay,
            lower=utility_decay,
            upper=1.0,
            upper_inclusive=False,
        )
        promotion_margin = validated_float32_scalar(
            "promotion_margin", self.promotion_margin, positive=True
        )
        promotion_blend = validated_float32_scalar(
            "promotion_blend", self.promotion_blend, lower=0.0, upper=1.0
        )
        future_utility_mix = validated_float32_scalar(
            "future_utility_mix", self.future_utility_mix, lower=0.0, upper=1.0
        )
        obgd_kappa = validated_float32_scalar("obgd_kappa", self.obgd_kappa, positive=True)
        candidate_utility_retention_decay = None
        if self.candidate_utility_retention_decay is not None:
            candidate_utility_retention_decay = validated_float32_scalar(
                "candidate_utility_retention_decay",
                self.candidate_utility_retention_decay,
                lower=utility_decay,
                upper=1.0,
                upper_inclusive=False,
            )
        if self.candidate_strategy != "all_pairs":
            raise ValueError("candidate_strategy must be 'all_pairs'")
        if self.utility_aggregation != "max":
            raise ValueError("utility_aggregation must be 'max'")
        if self.utility_task_balancing != "active":
            raise ValueError("utility_task_balancing must be 'active'")
        _require_exact_bool("refresh_candidates", self.refresh_candidates)
        _require_exact_bool("refresh_promoted_candidate", self.refresh_promoted_candidate)
        _require_exact_bool("use_obgd", self.use_obgd)
        canonical = {
            "target_amplitude": target_amplitude,
            "recovery_nmse_threshold": recovery_nmse_threshold,
            "step_size_output": step_size_output,
            "utility_decay": utility_decay,
            "retained_utility_decay": retained_utility_decay,
            "promotion_margin": promotion_margin,
            "promotion_blend": promotion_blend,
            "future_utility_mix": future_utility_mix,
            "obgd_kappa": obgd_kappa,
            "candidate_utility_retention_decay": candidate_utility_retention_decay,
        }
        for field, value in canonical.items():
            object.__setattr__(self, field, value)


@dataclass(frozen=True, slots=True)
class FeatureMemoryBudget:
    """Explicit active and candidate capacities used by the probe.

    Pair slots are the representation-selection budget.  Output-weight slots
    show the additional per-head learned storage in those two banks.  This is
    capacity accounting, not a byte-level accounting of JAX array metadata.
    """

    active_pair_slots: int
    candidate_pair_slots: int
    output_heads: int

    def __post_init__(self) -> None:
        """Canonicalize the bounded capacities used by result accounting."""
        for name in ("active_pair_slots", "candidate_pair_slots", "output_heads"):
            object.__setattr__(
                self,
                name,
                _require_builtin_int(
                    name,
                    getattr(self, name),
                    minimum=1,
                    maximum=2**31 - 1,
                ),
            )

    @property
    def total_pair_slots(self) -> int:
        """Total pair descriptors retained across both banks."""
        return self.active_pair_slots + self.candidate_pair_slots

    @property
    def active_output_weight_slots(self) -> int:
        """Learned task weights in the deployed bank."""
        return self.active_pair_slots * self.output_heads

    @property
    def candidate_output_weight_slots(self) -> int:
        """Learned task weights in the testing/archive bank."""
        return self.candidate_pair_slots * self.output_heads

    @property
    def total_output_weight_slots(self) -> int:
        """Learned task weights across both banks."""
        return self.total_pair_slots * self.output_heads


@dataclass(frozen=True, slots=True)
class PhaseEvidence:
    """Predict-before-update evidence for one contiguous task phase."""

    phase_index: int
    task: TaskName
    occurrence: int
    prequential_nmse: float
    recovery_steps: int | None

    def __post_init__(self) -> None:
        """Reject leftover scalar identities before phase records are traversed."""
        _require_builtin_int("phase_index", self.phase_index, minimum=0, maximum=2**31 - 1)
        _require_task_name("task", self.task)
        _require_builtin_int("occurrence", self.occurrence, minimum=1, maximum=2**31 - 1)
        _require_nmse("prequential_nmse", self.prequential_nmse)
        _require_recovery_steps("recovery_steps", self.recovery_steps)


@dataclass(frozen=True, slots=True)
class TaskRecoveryEvidence:
    """Acquisition recovery and all later recurrence recoveries for one task."""

    task: TaskName
    acquisition_steps: int | None
    recurrence_steps: tuple[int | None, ...]

    def __post_init__(self) -> None:
        """Validate exact task and recovery identities at construction."""
        _require_task_name("task", self.task)
        _require_recovery_steps("acquisition_steps", self.acquisition_steps)
        if type(self.recurrence_steps) is not tuple or len(self.recurrence_steps) > 4096:
            raise ValueError("recurrence_steps must be a bounded exact tuple")
        for index, value in enumerate(self.recurrence_steps):
            _require_recovery_steps(f"recurrence_steps[{index}]", value)


@dataclass(frozen=True, slots=True)
class RecurringFeatureSeedEvidence:
    """All reproducible evidence from one seed and one learner variant."""

    seed: int
    final_heldout_nmse: tuple[float, ...]
    active_pairs: tuple[Pair, ...]
    candidate_pairs: tuple[Pair, ...]
    phase_evidence: tuple[PhaseEvidence, ...]
    task_recovery: tuple[TaskRecoveryEvidence, ...]
    steps_seen: int

    def __post_init__(self) -> None:
        """Validate exact nested evidence before set/statistics consumers see it."""
        require_jax_seed(self.seed)
        if (
            type(self.final_heldout_nmse) is not tuple
            or not self.final_heldout_nmse
            or len(self.final_heldout_nmse) > 4096
        ):
            raise ValueError("final_heldout_nmse must be a bounded non-empty exact tuple")
        for index, value in enumerate(self.final_heldout_nmse):
            _require_nmse(f"final_heldout_nmse[{index}]", value)
        _require_pairs("active_pairs", self.active_pairs)
        _require_pairs("candidate_pairs", self.candidate_pairs)
        if type(self.phase_evidence) is not tuple or len(self.phase_evidence) > 4096:
            raise ValueError("phase_evidence must be a bounded exact tuple")
        if any(type(item) is not PhaseEvidence for item in self.phase_evidence):
            raise ValueError("phase_evidence must contain exact PhaseEvidence records")
        if type(self.task_recovery) is not tuple or len(self.task_recovery) > 4096:
            raise ValueError("task_recovery must be a bounded exact tuple")
        if any(type(item) is not TaskRecoveryEvidence for item in self.task_recovery):
            raise ValueError("task_recovery must contain exact TaskRecoveryEvidence records")
        _require_builtin_int("steps_seen", self.steps_seen, minimum=0, maximum=2**31 - 1)

    @property
    def critical_pairs_retained(self) -> tuple[bool, bool, bool]:
        """Whether A, B, and C occupy active slots at the end."""
        active = set(self.active_pairs)
        return tuple(TASK_PAIRS[TASK_NAMES.index(name)] in active for name in CRITICAL_TASKS)  # type: ignore[return-value]

    @property
    def retained_all_critical_pairs(self) -> bool:
        """Whether all three recurring pairs survived in the active budget."""
        return all(self.critical_pairs_retained)

    @property
    def obsolete_pair_evicted(self) -> bool:
        """Whether one-off pair D is absent from the final active bank."""
        return TASK_PAIRS[TASK_NAMES.index("D")] not in set(self.active_pairs)


@dataclass(frozen=True, slots=True)
class RecurringFeatureVariantEvidence:
    """Multi-seed evidence for one utility-retention setting."""

    name: VariantName
    utility_retention_decay: float | None
    seeds: tuple[RecurringFeatureSeedEvidence, ...]

    def __post_init__(self) -> None:
        """Validate exact variant metadata and bounded seed records."""
        if type(self.name) is not str or self.name not in {"retained", "no_retention"}:
            raise ValueError("name must be an exact registered variant name")
        if self.utility_retention_decay is not None:
            value = self.utility_retention_decay
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError("utility_retention_decay must be an exact float in [0,1)")
        if type(self.seeds) is not tuple or len(self.seeds) > 4096:
            raise ValueError("seeds must be a bounded exact tuple")
        if any(type(item) is not RecurringFeatureSeedEvidence for item in self.seeds):
            raise ValueError("seeds must contain exact RecurringFeatureSeedEvidence records")

    @property
    def all_critical_retention_rate(self) -> float:
        """Fraction of seeds ending with A, B, and C all active."""
        if not self.seeds:
            return 0.0
        return sum(seed.retained_all_critical_pairs for seed in self.seeds) / len(self.seeds)

    @property
    def obsolete_eviction_rate(self) -> float:
        """Fraction of seeds ending without obsolete pair D."""
        if not self.seeds:
            return 0.0
        return sum(seed.obsolete_pair_evicted for seed in self.seeds) / len(self.seeds)

    @property
    def median_critical_heldout_nmse(self) -> float:
        """Median final NMSE over critical heads and seeds."""
        values = [value for seed in self.seeds for value in seed.final_heldout_nmse[:3]]
        return float(statistics.median(values)) if values else math.inf

    @property
    def median_heldout_nmse_by_task(self) -> tuple[float, ...]:
        """Per-head medians in the public ``A, B, C, D`` order."""
        if not self.seeds:
            return tuple(math.inf for _ in TASK_NAMES)
        return tuple(
            float(statistics.median(seed.final_heldout_nmse[task_index] for seed in self.seeds))
            for task_index in range(len(TASK_NAMES))
        )

    @property
    def maximum_critical_task_median_nmse(self) -> float:
        """Worst per-task median across A, B, and C.

        This prevents two near-zero heads from hiding a forgotten third head in
        a pooled median.
        """
        return max(self.median_heldout_nmse_by_task[:3])

    @property
    def median_obsolete_heldout_nmse(self) -> float:
        """Median final NMSE for one-off head D."""
        values = [seed.final_heldout_nmse[3] for seed in self.seeds]
        return float(statistics.median(values)) if values else -math.inf

    @property
    def median_acquisition_recovery_steps(self) -> float:
        """Median recovery length for first A/B/C occurrences."""
        values = [
            recovery.acquisition_steps
            for seed in self.seeds
            for recovery in seed.task_recovery
            if recovery.task in CRITICAL_TASKS and recovery.acquisition_steps is not None
        ]
        return float(statistics.median(values)) if values else math.inf

    @property
    def median_recurrence_recovery_steps(self) -> float:
        """Median recovery length over later A/B/C occurrences."""
        values = [
            step
            for seed in self.seeds
            for recovery in seed.task_recovery
            if recovery.task in CRITICAL_TASKS
            for step in recovery.recurrence_steps
            if step is not None
        ]
        return float(statistics.median(values)) if values else math.inf


@dataclass(frozen=True, slots=True)
class RecurringFeatureGateCriteria:
    """Fail-closed thresholds for the canonical pairwise probe."""

    minimum_seeds: int = 30
    minimum_heldout_samples: int = 512
    minimum_retained_all_critical_rate: float = 1.0
    minimum_obsolete_eviction_rate: float = 1.0
    maximum_median_critical_nmse: float = 0.05
    minimum_median_obsolete_nmse: float = 0.8
    minimum_retention_rate_gain_over_baseline: float = 0.5
    minimum_critical_nmse_gain_over_baseline: float = 0.5
    require_recurrence_faster_than_acquisition: bool = True

    def __post_init__(self) -> None:
        """Validate and canonicalize the immutable criteria at construction."""
        self.validate()

    def validate(self) -> None:
        """Reject boolean or non-finite criteria before comparing evidence."""
        _require_builtin_int("minimum_seeds", self.minimum_seeds, minimum=1)
        _require_builtin_int(
            "minimum_heldout_samples", self.minimum_heldout_samples, minimum=1
        )
        minimum_retained_all_critical_rate = validated_float32_scalar(
            "minimum_retained_all_critical_rate",
            self.minimum_retained_all_critical_rate,
            lower=0.0,
            upper=1.0,
        )
        minimum_obsolete_eviction_rate = validated_float32_scalar(
            "minimum_obsolete_eviction_rate",
            self.minimum_obsolete_eviction_rate,
            lower=0.0,
            upper=1.0,
        )
        maximum_median_critical_nmse = validated_float32_scalar(
            "maximum_median_critical_nmse",
            self.maximum_median_critical_nmse,
            lower=0.0,
        )
        minimum_median_obsolete_nmse = validated_float32_scalar(
            "minimum_median_obsolete_nmse",
            self.minimum_median_obsolete_nmse,
            lower=0.0,
        )
        minimum_retention_rate_gain_over_baseline = validated_float32_scalar(
            "minimum_retention_rate_gain_over_baseline",
            self.minimum_retention_rate_gain_over_baseline,
        )
        minimum_critical_nmse_gain_over_baseline = validated_float32_scalar(
            "minimum_critical_nmse_gain_over_baseline",
            self.minimum_critical_nmse_gain_over_baseline,
        )
        _require_exact_bool(
            "require_recurrence_faster_than_acquisition",
            self.require_recurrence_faster_than_acquisition,
        )
        canonical = {
            "minimum_retained_all_critical_rate": minimum_retained_all_critical_rate,
            "minimum_obsolete_eviction_rate": minimum_obsolete_eviction_rate,
            "maximum_median_critical_nmse": maximum_median_critical_nmse,
            "minimum_median_obsolete_nmse": minimum_median_obsolete_nmse,
            "minimum_retention_rate_gain_over_baseline": (
                minimum_retention_rate_gain_over_baseline
            ),
            "minimum_critical_nmse_gain_over_baseline": (
                minimum_critical_nmse_gain_over_baseline
            ),
        }
        for field, value in canonical.items():
            object.__setattr__(self, field, value)


class RecurringFeatureGateError(AssertionError):
    """Raised when canonical evidence does not satisfy every gate criterion."""


@dataclass(frozen=True, slots=True)
class RecurringFeatureGateDecision:
    """Acceptance decision plus concrete diagnostics for every failed check."""

    accepted: bool
    summary: str
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject leftover bool/int/string identities before they become a verdict."""
        _require_exact_bool("accepted", self.accepted)
        if type(self.summary) is not str:
            raise ValueError("summary must be an exact string")
        if not self.summary:
            raise ValueError("summary must be non-empty")
        if type(self.failures) is not tuple:
            raise ValueError("failures must be an exact tuple")
        if any(type(item) is not str or not item for item in self.failures):
            raise ValueError("failures must contain non-empty exact strings")
        if self.accepted == bool(self.failures):
            raise ValueError("accepted must be true exactly when failures is empty")

    def require(self) -> None:
        """Raise an evidence-rich error unless every check passed."""
        if not self.accepted:
            details = "\n - ".join(self.failures)
            raise RecurringFeatureGateError(f"{self.summary}\n - {details}")


@dataclass(frozen=True, slots=True)
class RecurringFeatureGateResult:
    """Canonical retained learner and matched no-retention baseline evidence."""

    protocol: RecurringFeatureProtocol
    memory_budget: FeatureMemoryBudget
    retained: RecurringFeatureVariantEvidence
    no_retention: RecurringFeatureVariantEvidence
    scope: str = PAIRWISE_PROBE_SCOPE

    def __post_init__(self) -> None:
        """Reject leftover container/string identities at the public result boundary."""
        if type(self.protocol) is not RecurringFeatureProtocol:
            raise ValueError("protocol must be an exact RecurringFeatureProtocol")
        if type(self.memory_budget) is not FeatureMemoryBudget:
            raise ValueError("memory_budget must be an exact FeatureMemoryBudget")
        if type(self.retained) is not RecurringFeatureVariantEvidence:
            raise ValueError("retained must be an exact RecurringFeatureVariantEvidence")
        if type(self.no_retention) is not RecurringFeatureVariantEvidence:
            raise ValueError("no_retention must be an exact RecurringFeatureVariantEvidence")
        if self.retained.name != "retained" or self.no_retention.name != "no_retention":
            raise ValueError("variant records must occupy their declared result roles")
        if type(self.scope) is not str or self.scope != PAIRWISE_PROBE_SCOPE:
            raise ValueError("scope must be the exact pairwise probe scope")

    def decision(
        self,
        criteria: RecurringFeatureGateCriteria | None = None,
    ) -> RecurringFeatureGateDecision:
        """Evaluate this result without trusting a cached pass flag."""
        return evaluate_recurring_feature_gate(self, criteria=criteria)

    def require_pass(
        self,
        criteria: RecurringFeatureGateCriteria | None = None,
    ) -> RecurringFeatureGateDecision:
        """Re-evaluate and fail closed, returning the passing decision."""
        decision = self.decision(criteria)
        decision.require()
        return decision


def _validated_protocol(protocol: object) -> RecurringFeatureProtocol:
    if type(protocol) is not RecurringFeatureProtocol:
        raise ValueError("protocol must be an exact RecurringFeatureProtocol")
    checked = replace(protocol)
    for name, expected in (
        ("candidate_strategy", "all_pairs"),
        ("utility_aggregation", "max"),
        ("utility_task_balancing", "active"),
    ):
        if type(getattr(checked, name)) is not str:
            raise ValueError(f"protocol.{name} must be an exact string")
        if getattr(checked, name) != expected:
            raise ValueError(f"protocol.{name} must be {expected!r}")
    checked.validate()
    return checked


def _validated_phase_evidence(value: object) -> PhaseEvidence:
    if type(value) is not PhaseEvidence:
        raise ValueError("phase_evidence must contain exact PhaseEvidence records")
    return PhaseEvidence(
        phase_index=value.phase_index,
        task=value.task,
        occurrence=value.occurrence,
        prequential_nmse=value.prequential_nmse,
        recovery_steps=value.recovery_steps,
    )


def _validated_task_recovery(value: object) -> TaskRecoveryEvidence:
    if type(value) is not TaskRecoveryEvidence:
        raise ValueError("task_recovery must contain exact TaskRecoveryEvidence records")
    return TaskRecoveryEvidence(
        task=value.task,
        acquisition_steps=value.acquisition_steps,
        recurrence_steps=value.recurrence_steps,
    )


def _validated_seed_evidence(value: object) -> RecurringFeatureSeedEvidence:
    if type(value) is not RecurringFeatureSeedEvidence:
        raise ValueError("seeds must contain exact RecurringFeatureSeedEvidence records")
    if type(value.phase_evidence) is not tuple or len(value.phase_evidence) > 4096:
        raise ValueError("phase_evidence must be a bounded exact tuple")
    if type(value.task_recovery) is not tuple or len(value.task_recovery) > 4096:
        raise ValueError("task_recovery must be a bounded exact tuple")
    return RecurringFeatureSeedEvidence(
        seed=value.seed,
        final_heldout_nmse=value.final_heldout_nmse,
        active_pairs=value.active_pairs,
        candidate_pairs=value.candidate_pairs,
        phase_evidence=tuple(_validated_phase_evidence(item) for item in value.phase_evidence),
        task_recovery=tuple(_validated_task_recovery(item) for item in value.task_recovery),
        steps_seen=value.steps_seen,
    )


def _validated_variant_evidence(value: object) -> RecurringFeatureVariantEvidence:
    if type(value) is not RecurringFeatureVariantEvidence:
        raise ValueError("variants must be exact RecurringFeatureVariantEvidence records")
    if type(value.seeds) is not tuple or len(value.seeds) > 4096:
        raise ValueError("seeds must be a bounded exact tuple")
    return RecurringFeatureVariantEvidence(
        name=value.name,
        utility_retention_decay=value.utility_retention_decay,
        seeds=tuple(_validated_seed_evidence(seed) for seed in value.seeds),
    )


def _validate_seed_cross_fields(
    variant: RecurringFeatureVariantEvidence,
    protocol: RecurringFeatureProtocol,
) -> None:
    """Bind nested evidence identities to the frozen stream and pair universe."""
    seed_ids = tuple(seed.seed for seed in variant.seeds)
    if len(set(seed_ids)) != len(seed_ids):
        raise ValueError(f"{variant.name} seed identities must be unique")

    expected_pairs = {
        (left, right)
        for left in range(protocol.feature_dim)
        for right in range(left + 1, protocol.feature_dim)
    }
    occurrences: dict[TaskName, int] = {task: 0 for task in TASK_NAMES}
    expected_phases: list[tuple[int, TaskName, int]] = []
    for phase_index, task in enumerate(PHASE_TASKS):
        occurrences[task] += 1
        expected_phases.append((phase_index, task, occurrences[task]))
    expected_phase_tuple = tuple(expected_phases)

    for seed in variant.seeds:
        actual_phases = tuple(
            (phase.phase_index, phase.task, phase.occurrence) for phase in seed.phase_evidence
        )
        if actual_phases != expected_phase_tuple:
            raise ValueError(
                f"{variant.name} seed {seed.seed} phase evidence must match the frozen stream"
            )

        expected_recoveries = []
        for task in TASK_NAMES:
            recoveries = tuple(
                phase.recovery_steps for phase in seed.phase_evidence if phase.task == task
            )
            expected_recoveries.append((task, recoveries[0], recoveries[1:]))
        actual_recoveries = tuple(
            (recovery.task, recovery.acquisition_steps, recovery.recurrence_steps)
            for recovery in seed.task_recovery
        )
        if actual_recoveries != tuple(expected_recoveries):
            raise ValueError(
                f"{variant.name} seed {seed.seed} task recovery must summarize phase evidence"
            )

        active_pairs = set(seed.active_pairs)
        if len(active_pairs) != len(seed.active_pairs) or not active_pairs <= expected_pairs:
            raise ValueError(
                f"{variant.name} seed {seed.seed} active pairs must be unique members "
                "of the protocol pair universe"
            )


def _validated_gate_result(value: object) -> RecurringFeatureGateResult:
    """Return a recursively validated snapshot before any evidence is consumed."""
    if type(value) is not RecurringFeatureGateResult:
        raise ValueError("result must be an exact RecurringFeatureGateResult")
    protocol = _validated_protocol(value.protocol)
    if type(value.memory_budget) is not FeatureMemoryBudget:
        raise ValueError("memory_budget must be an exact FeatureMemoryBudget")
    budget = FeatureMemoryBudget(
        active_pair_slots=value.memory_budget.active_pair_slots,
        candidate_pair_slots=value.memory_budget.candidate_pair_slots,
        output_heads=value.memory_budget.output_heads,
    )
    expected_budget = FeatureMemoryBudget(
        active_pair_slots=protocol.active_pair_budget,
        candidate_pair_slots=protocol.candidate_pair_budget,
        output_heads=protocol.output_heads,
    )
    if budget != expected_budget:
        raise ValueError("memory_budget capacities must match the protocol")
    checked = RecurringFeatureGateResult(
        protocol=protocol,
        memory_budget=budget,
        retained=_validated_variant_evidence(value.retained),
        no_retention=_validated_variant_evidence(value.no_retention),
        scope=value.scope,
    )
    _validate_seed_cross_fields(checked.retained, protocol)
    _validate_seed_cross_fields(checked.no_retention, protocol)
    return checked


def _validated_gate_criteria(value: object) -> RecurringFeatureGateCriteria:
    if value is None:
        return RecurringFeatureGateCriteria()
    if type(value) is not RecurringFeatureGateCriteria:
        raise ValueError("criteria must be an exact RecurringFeatureGateCriteria")
    checked = replace(value)
    checked.validate()
    return checked


@dataclass(frozen=True, slots=True)
class _SeedArrays:
    observations: Array
    targets: Array
    scalar_targets: Array
    active_tasks: Array
    heldout_observations: Array
    heldout_targets: Array
    learner_key: Array


def _task_index(task: TaskName) -> int:
    return TASK_NAMES.index(task)


def _make_seed_arrays(seed: int, protocol: RecurringFeatureProtocol) -> _SeedArrays:
    learner_key, stream_key, heldout_key = jr.split(jr.key(seed), 3)
    observations = jr.normal(
        stream_key,
        (protocol.total_steps, protocol.feature_dim),
        dtype=jnp.float32,
    )
    phase_task_indices = jnp.asarray(
        [_task_index(task) for task in PHASE_TASKS],
        dtype=jnp.int32,
    )
    active_tasks = jnp.repeat(phase_task_indices, protocol.steps_per_phase)
    pair_array = jnp.asarray(TASK_PAIRS, dtype=jnp.int32)
    active_pairs = pair_array[active_tasks]
    rows = jnp.arange(protocol.total_steps)
    scalar_targets = (
        protocol.target_amplitude
        * observations[rows, active_pairs[:, 0]]
        * observations[rows, active_pairs[:, 1]]
    )
    targets = jnp.full(
        (protocol.total_steps, protocol.output_heads),
        jnp.nan,
        dtype=jnp.float32,
    )
    targets = targets.at[rows, active_tasks].set(scalar_targets)

    heldout_observations = jr.normal(
        heldout_key,
        (protocol.heldout_samples, protocol.feature_dim),
        dtype=jnp.float32,
    )
    heldout_targets = protocol.target_amplitude * (
        heldout_observations[:, pair_array[:, 0]] * heldout_observations[:, pair_array[:, 1]]
    )
    return _SeedArrays(
        observations=observations,
        targets=targets,
        scalar_targets=scalar_targets,
        active_tasks=active_tasks,
        heldout_observations=heldout_observations,
        heldout_targets=heldout_targets,
        learner_key=learner_key,
    )


def _make_learner(
    protocol: RecurringFeatureProtocol,
    utility_retention_decay: float | None,
) -> FixedBudgetInteractionLearner:
    return FixedBudgetInteractionLearner(
        n_features=protocol.active_pair_budget,
        n_tasks=protocol.output_heads,
        step_size_output=protocol.step_size_output,
        utility_decay=protocol.utility_decay,
        utility_retention_decay=utility_retention_decay,
        replacement_interval=protocol.replacement_interval,
        min_feature_age=protocol.min_feature_age,
        candidate_count=protocol.candidate_pair_budget,
        candidate_min_age=protocol.candidate_min_age,
        promotion_margin=protocol.promotion_margin,
        promotion_blend=protocol.promotion_blend,
        candidate_strategy=protocol.candidate_strategy,
        utility_aggregation=protocol.utility_aggregation,
        utility_task_balancing=protocol.utility_task_balancing,
        future_utility_mix=protocol.future_utility_mix,
        candidate_utility_retention_decay=protocol.candidate_utility_retention_decay,
        refresh_candidates=protocol.refresh_candidates,
        refresh_promoted_candidate=protocol.refresh_promoted_candidate,
        use_obgd=protocol.use_obgd,
        obgd_kappa=protocol.obgd_kappa,
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _train_sequence(
    learner: FixedBudgetInteractionLearner,
    state: InteractionFeatureState,
    observations: Array,
    targets: Array,
    active_tasks: Array,
) -> tuple[InteractionFeatureState, Array]:
    """Train once through the stream and retain predict-before-update errors."""

    def step(
        carry: InteractionFeatureState,
        inputs: tuple[Array, Array, Array],
    ) -> tuple[InteractionFeatureState, Array]:
        observation, target, task = inputs
        update = learner.update(carry, observation, target)
        return update.state, update.errors[task]

    return jax.lax.scan(step, state, (observations, targets, active_tasks))


@jax.jit
def _predict_batch(
    state: InteractionFeatureState,
    observations: Array,
) -> Array:
    features = observations[:, state.feature_left] * observations[:, state.feature_right]
    return features @ state.output_weights.T + state.output_biases


def _pairs(left: Array, right: Array) -> tuple[Pair, ...]:
    left_values = np.asarray(left, dtype=np.int64).tolist()
    right_values = np.asarray(right, dtype=np.int64).tolist()
    return tuple(
        (int(left_value), int(right_value))
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )


def _nmse(
    errors: npt.NDArray[np.float64],
    targets: npt.NDArray[np.float64],
) -> float:
    denominator = float(np.mean(np.square(targets)))
    if not math.isfinite(denominator) or denominator <= 0.0:
        return math.inf
    return float(np.mean(np.square(errors)) / denominator)


def _recovery_steps(
    errors: npt.NDArray[np.float64],
    targets: npt.NDArray[np.float64],
    *,
    window: int,
    threshold: float,
) -> int | None:
    """First full rolling window whose empirical NMSE reaches threshold."""
    squared_errors = np.square(errors)
    squared_targets = np.square(targets)
    kernel = np.ones(window, dtype=np.float64)
    error_sums = np.convolve(squared_errors, kernel, mode="valid")
    target_sums = np.convolve(squared_targets, kernel, mode="valid")
    rolling_nmse = error_sums / np.maximum(target_sums, np.finfo(np.float64).tiny)
    recovered = np.flatnonzero(rolling_nmse <= threshold)
    if recovered.size == 0:
        return None
    return int(recovered[0]) + window


def _phase_evidence(
    scalar_errors: Array,
    scalar_targets: Array,
    protocol: RecurringFeatureProtocol,
) -> tuple[PhaseEvidence, ...]:
    errors = np.asarray(scalar_errors, dtype=np.float64).reshape(
        len(PHASE_TASKS),
        protocol.steps_per_phase,
    )
    targets = np.asarray(scalar_targets, dtype=np.float64).reshape(
        len(PHASE_TASKS),
        protocol.steps_per_phase,
    )
    occurrences: dict[TaskName, int] = {name: 0 for name in TASK_NAMES}
    evidence: list[PhaseEvidence] = []
    for phase_index, task in enumerate(PHASE_TASKS):
        occurrences[task] += 1
        phase_errors = errors[phase_index]
        phase_targets = targets[phase_index]
        evidence.append(
            PhaseEvidence(
                phase_index=phase_index,
                task=task,
                occurrence=occurrences[task],
                prequential_nmse=_nmse(phase_errors, phase_targets),
                recovery_steps=_recovery_steps(
                    phase_errors,
                    phase_targets,
                    window=protocol.recovery_window,
                    threshold=protocol.recovery_nmse_threshold,
                ),
            )
        )
    return tuple(evidence)


def _task_recovery(
    phases: tuple[PhaseEvidence, ...],
) -> tuple[TaskRecoveryEvidence, ...]:
    evidence: list[TaskRecoveryEvidence] = []
    for task in TASK_NAMES:
        recoveries = tuple(phase.recovery_steps for phase in phases if phase.task == task)
        evidence.append(
            TaskRecoveryEvidence(
                task=task,
                acquisition_steps=recoveries[0],
                recurrence_steps=recoveries[1:],
            )
        )
    return tuple(evidence)


def _run_seed(
    learner: FixedBudgetInteractionLearner,
    seed: int,
    arrays: _SeedArrays,
    protocol: RecurringFeatureProtocol,
) -> RecurringFeatureSeedEvidence:
    state = learner.init(protocol.feature_dim, arrays.learner_key)
    final_state, scalar_errors = _train_sequence(
        learner,
        state,
        arrays.observations,
        arrays.targets,
        arrays.active_tasks,
    )

    # This separate held-out prediction is intentionally the last learner
    # operation.  It cannot update weights, utilities, ages, or step count.
    heldout_predictions = _predict_batch(
        final_state,
        arrays.heldout_observations,
    )
    predictions = np.asarray(heldout_predictions, dtype=np.float64)
    targets = np.asarray(arrays.heldout_targets, dtype=np.float64)
    heldout_nmse = tuple(
        _nmse(predictions[:, task] - targets[:, task], targets[:, task])
        for task in range(protocol.output_heads)
    )
    phases = _phase_evidence(scalar_errors, arrays.scalar_targets, protocol)
    return RecurringFeatureSeedEvidence(
        seed=seed,
        final_heldout_nmse=heldout_nmse,
        active_pairs=_pairs(final_state.feature_left, final_state.feature_right),
        candidate_pairs=_pairs(
            final_state.candidate_left,
            final_state.candidate_right,
        ),
        phase_evidence=phases,
        task_recovery=_task_recovery(phases),
        steps_seen=int(final_state.step_count),
    )


def run_recurring_feature_gate(
    seeds: Iterable[int] = EVIDENCE_SEEDS,
    *,
    protocol: RecurringFeatureProtocol | None = None,
) -> RecurringFeatureGateResult:
    """Run the canonical learner and its matched no-retention ablation.

    Both variants receive identical observations, targets, held-out probes,
    initial active pairs, candidate archive, and PRNG state for every seed.
    They differ only in active-feature utility retention: ``0.999`` versus
    ``None``.  There is one initialization and no reset across all nine phases.

    Passing fewer than thirty seeds is allowed for exploration, but the default
    acceptance criteria fail closed on insufficient evidence.
    """
    chosen_protocol = protocol or RecurringFeatureProtocol()
    chosen_protocol.validate()
    chosen_seeds = require_unique_jax_seeds(tuple(seeds))

    retained_learner = _make_learner(
        chosen_protocol,
        chosen_protocol.retained_utility_decay,
    )
    baseline_learner = _make_learner(chosen_protocol, None)
    retained_runs: list[RecurringFeatureSeedEvidence] = []
    baseline_runs: list[RecurringFeatureSeedEvidence] = []
    for seed in chosen_seeds:
        arrays = _make_seed_arrays(seed, chosen_protocol)
        retained_runs.append(_run_seed(retained_learner, seed, arrays, chosen_protocol))
        baseline_runs.append(_run_seed(baseline_learner, seed, arrays, chosen_protocol))

    budget = FeatureMemoryBudget(
        active_pair_slots=chosen_protocol.active_pair_budget,
        candidate_pair_slots=chosen_protocol.candidate_pair_budget,
        output_heads=chosen_protocol.output_heads,
    )
    return RecurringFeatureGateResult(
        protocol=chosen_protocol,
        memory_budget=budget,
        retained=RecurringFeatureVariantEvidence(
            name="retained",
            utility_retention_decay=chosen_protocol.retained_utility_decay,
            seeds=tuple(retained_runs),
        ),
        no_retention=RecurringFeatureVariantEvidence(
            name="no_retention",
            utility_retention_decay=None,
            seeds=tuple(baseline_runs),
        ),
    )


def _canonical_protocol_failures(protocol: RecurringFeatureProtocol) -> list[str]:
    canonical = RecurringFeatureProtocol(heldout_samples=protocol.heldout_samples)
    checked_fields = (
        "feature_dim",
        "output_heads",
        "steps_per_phase",
        "target_amplitude",
        "active_pair_budget",
        "candidate_pair_budget",
        "step_size_output",
        "utility_decay",
        "retained_utility_decay",
        "replacement_interval",
        "min_feature_age",
        "candidate_min_age",
        "promotion_margin",
        "promotion_blend",
        "candidate_strategy",
        "utility_aggregation",
        "utility_task_balancing",
        "future_utility_mix",
        "candidate_utility_retention_decay",
        "refresh_candidates",
        "refresh_promoted_candidate",
        "use_obgd",
        "obgd_kappa",
        "recovery_window",
        "recovery_nmse_threshold",
    )
    failures = []
    for field in checked_fields:
        actual = getattr(protocol, field)
        expected = getattr(canonical, field)
        if actual != expected:
            failures.append(f"protocol.{field}={actual!r}, expected canonical {expected!r}")
    return failures


def _seed_failures(
    result: RecurringFeatureGateResult,
) -> list[str]:
    protocol = result.protocol
    failures: list[str] = []
    expected_candidate_pairs = {
        (left, right)
        for left in range(protocol.feature_dim)
        for right in range(left + 1, protocol.feature_dim)
    }
    for variant in (result.retained, result.no_retention):
        for seed in variant.seeds:
            if len(seed.final_heldout_nmse) != protocol.output_heads or not all(
                math.isfinite(value) and value >= 0.0 for value in seed.final_heldout_nmse
            ):
                failures.append(
                    f"{variant.name} seed {seed.seed} has invalid held-out NMSE "
                    f"{seed.final_heldout_nmse!r}"
                )
            if len(seed.phase_evidence) != len(PHASE_TASKS) or not all(
                math.isfinite(phase.prequential_nmse) and phase.prequential_nmse >= 0.0
                for phase in seed.phase_evidence
            ):
                failures.append(
                    f"{variant.name} seed {seed.seed} has incomplete/non-finite "
                    "prequential phase evidence"
                )
            if seed.steps_seen != protocol.total_steps:
                failures.append(
                    f"{variant.name} seed {seed.seed} saw {seed.steps_seen} updates, "
                    f"expected uninterrupted {protocol.total_steps}"
                )
            if len(seed.active_pairs) != protocol.active_pair_budget:
                failures.append(
                    f"{variant.name} seed {seed.seed} has {len(seed.active_pairs)} "
                    f"active slots, expected {protocol.active_pair_budget}"
                )
            candidate_set = set(seed.candidate_pairs)
            if (
                len(seed.candidate_pairs) != protocol.candidate_pair_budget
                or candidate_set != expected_candidate_pairs
            ):
                failures.append(
                    f"{variant.name} seed {seed.seed} candidate archive has "
                    f"{len(seed.candidate_pairs)} slots/{len(candidate_set)} unique pairs; "
                    f"expected {protocol.candidate_pair_budget} slots covering "
                    f"{len(expected_candidate_pairs)} all-pairs entries"
                )
    return failures


def evaluate_recurring_feature_gate(
    result: RecurringFeatureGateResult,
    *,
    criteria: RecurringFeatureGateCriteria | None = None,
) -> RecurringFeatureGateDecision:
    """Recompute a fail-closed decision from raw per-seed evidence."""
    result = _validated_gate_result(result)
    chosen = _validated_gate_criteria(criteria)
    failures = _canonical_protocol_failures(result.protocol)
    failures.extend(_seed_failures(result))

    retained = result.retained
    baseline = result.no_retention
    retained_seed_ids = tuple(seed.seed for seed in retained.seeds)
    baseline_seed_ids = tuple(seed.seed for seed in baseline.seeds)
    if retained.name != "retained" or retained.utility_retention_decay != (
        result.protocol.retained_utility_decay
    ):
        failures.append("retained variant label/retention setting does not match the protocol")
    if baseline.name != "no_retention" or baseline.utility_retention_decay is not None:
        failures.append("baseline is not the matched no-retention variant")
    if retained_seed_ids != baseline_seed_ids:
        failures.append(
            f"variant seed mismatch: retained={retained_seed_ids!r}, "
            f"no_retention={baseline_seed_ids!r}"
        )
    if len(retained.seeds) < chosen.minimum_seeds:
        failures.append(
            f"only {len(retained.seeds)} matched seeds, require at least {chosen.minimum_seeds}"
        )
    if result.protocol.heldout_samples < chosen.minimum_heldout_samples:
        failures.append(
            f"only {result.protocol.heldout_samples} held-out samples per seed, "
            f"require at least {chosen.minimum_heldout_samples}"
        )

    retention_rate = retained.all_critical_retention_rate
    if retention_rate < chosen.minimum_retained_all_critical_rate:
        failed_seeds = [
            f"{seed.seed}:{seed.active_pairs!r}"
            for seed in retained.seeds
            if not seed.retained_all_critical_pairs
        ]
        failures.append(
            f"retained all-critical rate {retention_rate:.3f} is below "
            f"{chosen.minimum_retained_all_critical_rate:.3f}; "
            f"failed seeds/active pairs={failed_seeds}"
        )

    eviction_rate = retained.obsolete_eviction_rate
    if eviction_rate < chosen.minimum_obsolete_eviction_rate:
        failed_seeds = [
            f"{seed.seed}:{seed.active_pairs!r}"
            for seed in retained.seeds
            if not seed.obsolete_pair_evicted
        ]
        failures.append(
            f"obsolete-D eviction rate {eviction_rate:.3f} is below "
            f"{chosen.minimum_obsolete_eviction_rate:.3f}; "
            f"failed seeds/active pairs={failed_seeds}"
        )

    retained_nmse = retained.maximum_critical_task_median_nmse
    if retained_nmse > chosen.maximum_median_critical_nmse:
        failures.append(
            f"retained median critical held-out NMSE {retained_nmse:.6f} exceeds "
            f"{chosen.maximum_median_critical_nmse:.6f}"
        )
    obsolete_nmse = retained.median_obsolete_heldout_nmse
    if obsolete_nmse < chosen.minimum_median_obsolete_nmse:
        failures.append(
            f"retained median obsolete-D held-out NMSE {obsolete_nmse:.6f} is below "
            f"forgetting floor {chosen.minimum_median_obsolete_nmse:.6f}"
        )

    retention_gain = retained.all_critical_retention_rate - baseline.all_critical_retention_rate
    if retention_gain < chosen.minimum_retention_rate_gain_over_baseline:
        failures.append(
            f"all-critical retention gain over matched no-retention baseline "
            f"is {retention_gain:.3f}, require "
            f"{chosen.minimum_retention_rate_gain_over_baseline:.3f}; "
            f"rates={retained.all_critical_retention_rate:.3f} vs "
            f"{baseline.all_critical_retention_rate:.3f}"
        )
    nmse_gain = (
        baseline.maximum_critical_task_median_nmse - retained.maximum_critical_task_median_nmse
    )
    if nmse_gain < chosen.minimum_critical_nmse_gain_over_baseline:
        failures.append(
            f"median critical held-out NMSE gain over matched no-retention "
            f"baseline is {nmse_gain:.6f}, require "
            f"{chosen.minimum_critical_nmse_gain_over_baseline:.6f}; "
            f"worst task-median NMSE={retained.maximum_critical_task_median_nmse:.6f} "
            f"vs {baseline.maximum_critical_task_median_nmse:.6f}"
        )

    acquisition = retained.median_acquisition_recovery_steps
    recurrence = retained.median_recurrence_recovery_steps
    if chosen.require_recurrence_faster_than_acquisition and not (
        math.isfinite(acquisition) and math.isfinite(recurrence) and recurrence < acquisition
    ):
        failures.append(
            f"median recurrence recovery ({recurrence!r} samples) is not faster "
            f"than acquisition ({acquisition!r} samples) at rolling "
            f"NMSE <= {result.protocol.recovery_nmse_threshold}"
        )

    budget = result.memory_budget
    if (
        budget.active_pair_slots != result.protocol.active_pair_budget
        or budget.candidate_pair_slots != result.protocol.candidate_pair_budget
        or budget.output_heads != result.protocol.output_heads
    ):
        failures.append(
            "reported memory budget does not match protocol: "
            f"budget={budget!r}, active={result.protocol.active_pair_budget}, "
            f"candidate={result.protocol.candidate_pair_budget}, "
            f"heads={result.protocol.output_heads}"
        )

    retained_medians = tuple(round(value, 6) for value in retained.median_heldout_nmse_by_task)
    baseline_medians = tuple(round(value, 6) for value in baseline.median_heldout_nmse_by_task)
    summary = (
        "Recurring pairwise feature gate "
        f"{'PASSED' if not failures else 'FAILED'}: "
        f"seeds={len(retained.seeds)}, "
        f"all-critical={retained.all_critical_retention_rate:.3f} "
        f"(baseline {baseline.all_critical_retention_rate:.3f}), "
        f"D-eviction={retained.obsolete_eviction_rate:.3f}, "
        f"heldout-medians={retained_medians} "
        f"(baseline {baseline_medians}), "
        f"D-NMSE={retained.median_obsolete_heldout_nmse:.6f}, "
        f"recovery={acquisition:.1f}->{recurrence:.1f}, "
        f"pair-slots={budget.active_pair_slots}+{budget.candidate_pair_slots}."
    )
    return RecurringFeatureGateDecision(
        accepted=not failures,
        summary=summary,
        failures=tuple(failures),
    )


__all__ = [
    "CRITICAL_TASKS",
    "DEVELOPMENT_SEEDS",
    "EVIDENCE_SEEDS",
    "PAIRWISE_PROBE_SCOPE",
    "PHASE_TASKS",
    "TASK_NAMES",
    "TASK_PAIRS",
    "FeatureMemoryBudget",
    "PhaseEvidence",
    "RecurringFeatureGateCriteria",
    "RecurringFeatureGateDecision",
    "RecurringFeatureGateError",
    "RecurringFeatureGateResult",
    "RecurringFeatureProtocol",
    "RecurringFeatureSeedEvidence",
    "RecurringFeatureVariantEvidence",
    "TaskRecoveryEvidence",
    "evaluate_recurring_feature_gate",
    "run_recurring_feature_gate",
]
