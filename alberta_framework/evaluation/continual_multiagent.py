"""Fail-closed A-B-A benchmark for the recurring two-agent stream.

This benchmark is intentionally narrow.  It tests whether a tiny, fixed-memory
online controller can learn two visibly cued coordination regimes, retain the
first regime while learning the second, and exploit an online-learning partner.
It does not establish general feature discovery or completion of the Alberta
Plan.

The learner receives only ordinary observations, its selected action, and the
shared reward.  It is never passed a phase index, change flag, oracle field, or
task-boundary callback.  Evaluation knows the A-B-A schedule, but its policy
probes are read-only counterfactual rollouts and never replace the continuing
life state.

Three paired conditions have identical two-scalar action and controller-state
budgets:

``frozen``
    Neither agent writes learning updates.
``learner_only``
    Agent 0 learns while agent 1 remains frozen.
``joint_adaptive``
    Both agents learn.  Comparing this with ``learner_only`` reports the
    benefit of partner learning (coadaptation uplift).  It is not an
    intelligence-amplification or recommendation-intervention estimate.

The deliberately simple controller is an epsilon-greedy, per-agent contextual
bandit over the two visible context channels and two actions.  Separate
contextual values make this a retention sanity check, not a feature-finding
solution.  All updates are predict-act-observe-update and use no replay.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from time import perf_counter, perf_counter_ns
from typing import Any, Literal, SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from numpy.typing import NDArray

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.evaluation._measurement_validation import (
    finite_real,
    nonnegative_finite_real,
    real_number,
    validate_interval_bounds,
)
from alberta_framework.streams.recurring_multiagent import (
    AVOID_CONTEXT,
    AVOID_CONTEXT_INDEX,
    MEET_CONTEXT,
    MEET_CONTEXT_INDEX,
    RecurringTwoAgentState,
    RecurringTwoAgentTransition,
    RecurringTwoAgentWorld,
)
from alberta_framework.utils.metrics import (
    ContinualLearningSummary,
    compute_recovery_lengths,
    summarize_continual_learning,
)

type ConditionName = Literal["frozen", "learner_only", "joint_adaptive"]
type StepFunction = Callable[
    [RecurringTwoAgentState, Array],
    tuple[RecurringTwoAgentTransition, RecurringTwoAgentState],
]

CONDITION_MASKS: tuple[tuple[ConditionName, tuple[bool, bool]], ...] = (
    ("frozen", (False, False)),
    ("learner_only", (True, False)),
    ("joint_adaptive", (True, True)),
)
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_MAX_CONFIGURED_ARRAY_NBYTES = 256 * 1024 * 1024
_ACTUAL_INT_TYPES = frozenset({int, *(np.dtype(code).type for code in "bBhHiIlLqQpP")})


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _require_uint32(name: str, value: object) -> int:
    return _require_int32(name, value, minimum=0, maximum=_UINT32_MAX)


def _require_ndarray(
    name: str,
    value: object,
    *,
    dtype: np.dtype[Any],
    ndim: int = 1,
) -> NDArray[Any]:
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must be an exact numpy.ndarray")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    return value


def _freeze_ndarray(value: NDArray[Any]) -> NDArray[Any]:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _require_seed(value: object) -> int:
    try:
        return _require_int32("seed", value, minimum=0)
    except ValueError as error:
        raise ValueError("seeds must lie in [0, 2**31)") from error


def _condition_working_nbytes(phase_steps: int) -> int:
    """Exact bytes in the two phase-length float64 work vectors."""
    return 2 * (3 * phase_steps) * np.dtype(np.float64).itemsize


def _condition_result_array_nbytes(phase_steps: int) -> int:
    """Exact retained NumPy payload for one completed condition."""
    float64_nbytes = np.dtype(np.float64).itemsize
    int64_nbytes = np.dtype(np.int64).itemsize
    return (
        3 * phase_steps * float64_nbytes
        + 3 * float64_nbytes
        + 3 * 2 * float64_nbytes
        + 2 * int64_nbytes
    )


def _world_array_nbytes(nuisance_dim: int) -> tuple[int, int]:
    """Exact persistent-state and one-observation array bytes for the world."""
    state_nbytes = 36 + 2 * nuisance_dim * np.dtype(np.float32).itemsize
    observation_nbytes = 2 * (6 + nuisance_dim) * np.dtype(np.float32).itemsize
    return state_nbytes, observation_nbytes


def _bootstrap_working_nbytes(resamples: int, sample_size: int) -> int:
    """Peak owned NumPy payload for indices, sampled values, and means."""
    itemsize = np.dtype(np.float64).itemsize
    return 2 * resamples * sample_size * itemsize + resamples * itemsize


def _require_resource_limit(name: str, nbytes: int) -> None:
    if nbytes > _MAX_CONFIGURED_ARRAY_NBYTES:
        raise ValueError(f"{name} requires {nbytes} bytes; limit is {_MAX_CONFIGURED_ARRAY_NBYTES}")


@dataclass(frozen=True)
class ContinualMultiAgentConfig:
    """Scientific and resource configuration for one A-B-A life."""

    phase_steps: int = 64
    nuisance_dim: int = 4
    learning_rate: float = 0.15
    exploration_rate: float = 0.20
    probe_horizon: int = 12
    probe_tail_steps: int = 4
    recovery_reward_threshold: float = 0.70
    recovery_window: int = 4
    stability_reference_reward: float = 0.75
    bootstrap_resamples: int = 10_000
    confidence_level: float = 0.95
    bootstrap_seed: int = 2_026_073_000

    def __post_init__(self) -> None:
        phase_steps = _require_int32("phase_steps", self.phase_steps, minimum=2)
        nuisance_dim = _require_int32("nuisance_dim", self.nuisance_dim, minimum=0)
        probe_horizon = _require_int32(
            "probe_horizon", self.probe_horizon, minimum=1, maximum=phase_steps
        )
        probe_tail_steps = _require_int32(
            "probe_tail_steps", self.probe_tail_steps, minimum=1, maximum=probe_horizon
        )
        recovery_window = _require_int32(
            "recovery_window", self.recovery_window, minimum=1, maximum=phase_steps
        )
        bootstrap_resamples = _require_int32(
            "bootstrap_resamples",
            self.bootstrap_resamples,
            minimum=1000,
            maximum=_MAX_CONFIGURED_ARRAY_NBYTES // (3 * np.dtype(np.float64).itemsize),
        )
        bootstrap_seed = _require_uint32("bootstrap_seed", self.bootstrap_seed)

        _require_resource_limit(
            "per-condition phase work arrays",
            _condition_working_nbytes(phase_steps),
        )
        world_state_nbytes, observation_nbytes = _world_array_nbytes(nuisance_dim)
        _require_resource_limit("recurring world state", world_state_nbytes)
        _require_resource_limit("recurring world observation", observation_nbytes)

        object.__setattr__(self, "phase_steps", phase_steps)
        object.__setattr__(self, "nuisance_dim", nuisance_dim)
        object.__setattr__(self, "probe_horizon", probe_horizon)
        object.__setattr__(self, "probe_tail_steps", probe_tail_steps)
        object.__setattr__(self, "recovery_window", recovery_window)
        object.__setattr__(self, "bootstrap_resamples", bootstrap_resamples)
        object.__setattr__(self, "bootstrap_seed", bootstrap_seed)

        object.__setattr__(
            self,
            "learning_rate",
            validated_float32_scalar("learning_rate", self.learning_rate, positive=True, upper=1.0),
        )
        for name in (
            "exploration_rate",
            "recovery_reward_threshold",
            "stability_reference_reward",
        ):
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
        object.__setattr__(
            self,
            "confidence_level",
            validated_float32_scalar(
                "confidence_level",
                self.confidence_level,
                positive=True,
                upper=1.0,
                upper_inclusive=False,
            ),
        )


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Frozen thresholds for the multi-seed acceptance decision.

    Values were calibrated on development seeds 0--29 and frozen before the
    single promoted run on held-out seeds 30--59 (see docs/status.md);
    ``evidence_seed_start=30`` with ``minimum_seed_count=30`` pins exactly
    that consumed schedule.  The uplift thresholds gate the bootstrap
    interval's *lower* bound, not the point estimate.  Retuning any value
    after seeing held-out results is disallowed — a failed gate stays a
    valid rejection.
    """

    minimum_seed_count: int = 30
    evidence_seed_start: int = 30
    minimum_reward_uplift_over_frozen: float = 0.15
    minimum_partner_uplift: float = 0.20
    minimum_recurrent_a_probe_reward: float = 0.90
    maximum_mean_forgetting: float = 0.05
    maximum_interference_forgetting: float = 0.01
    minimum_recurrence_recovery_fraction: float = 0.95
    maximum_mean_recurrence_recovery_steps: float = 16.0
    maximum_mean_stability_gap: float = 0.20
    maximum_update_latency_ms: float = 5.0

    def __post_init__(self) -> None:
        minimum_seed_count = _require_int32(
            "minimum_seed_count", self.minimum_seed_count, minimum=1
        )
        evidence_seed_start = _require_int32(
            "evidence_seed_start", self.evidence_seed_start, minimum=0
        )
        if evidence_seed_start + minimum_seed_count > _INT32_MAX + 1:
            raise ValueError("evidence_seed_start + minimum_seed_count must not exceed 2147483648")
        object.__setattr__(
            self,
            "minimum_seed_count",
            minimum_seed_count,
        )
        object.__setattr__(
            self,
            "evidence_seed_start",
            evidence_seed_start,
        )
        for name in (
            "minimum_reward_uplift_over_frozen",
            "minimum_partner_uplift",
            "minimum_recurrent_a_probe_reward",
            "maximum_mean_forgetting",
            "maximum_interference_forgetting",
            "minimum_recurrence_recovery_fraction",
            "maximum_mean_recurrence_recovery_steps",
            "maximum_mean_stability_gap",
            "maximum_update_latency_ms",
        ):
            object.__setattr__(
                self,
                name,
                validated_float32_scalar(name, getattr(self, name)),
            )


@dataclass(frozen=True)
class ControllerBudget:
    """Persistent dynamic state and action accounting for one condition."""

    state_scalars: int
    state_bytes: int
    action_scalars_per_step: int

    def __post_init__(self) -> None:
        for name, minimum in (
            ("state_scalars", 0),
            ("state_bytes", 0),
            ("action_scalars_per_step", 1),
        ):
            object.__setattr__(
                self,
                name,
                _require_int32(name, getattr(self, name), minimum=minimum),
            )


@dataclass(frozen=True)
class TimingMetrics:
    """Steady-state timing after one environment compilation warm-up."""

    wall_seconds: float
    mean_step_latency_ms: float
    mean_update_latency_ms: float
    p95_update_latency_ms: float

    def __post_init__(self) -> None:
        for name in (
            "wall_seconds",
            "mean_step_latency_ms",
            "mean_update_latency_ms",
            "p95_update_latency_ms",
        ):
            object.__setattr__(
                self,
                name,
                nonnegative_finite_real(name, getattr(self, name)),
            )


@dataclass(frozen=True)
class BootstrapInterval:
    """Deterministic non-parametric interval over paired seed statistics."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int
    sample_size: int
    method: str = "paired-percentile-bootstrap"

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimate", finite_real("estimate", self.estimate))
        object.__setattr__(self, "lower", finite_real("lower", self.lower))
        object.__setattr__(self, "upper", finite_real("upper", self.upper))
        object.__setattr__(
            self,
            "confidence_level",
            finite_real("confidence_level", self.confidence_level),
        )
        object.__setattr__(
            self,
            "resamples",
            _require_int32("resamples", self.resamples, minimum=1),
        )
        object.__setattr__(
            self,
            "sample_size",
            _require_int32("sample_size", self.sample_size, minimum=1),
        )
        if type(self.method) is not str or not self.method:
            raise ValueError("method must be a non-empty string")
        validate_interval_bounds(
            lower=self.lower,
            upper=self.upper,
            confidence_level=self.confidence_level,
        )


@dataclass(frozen=True)
class ConditionResult:
    """All evidence from one seed and one uninterrupted condition."""

    seed: int
    condition: ConditionName
    learning_mask: tuple[bool, bool]
    online_rewards: NDArray[np.float64]
    phase_mean_rewards: NDArray[np.float64]
    performance_matrix: NDArray[np.float64]
    summary: ContinualLearningSummary
    recovery_lengths: NDArray[np.int64]
    recurrence_recovery_steps: int
    interference_forgetting: float
    controller_budget: ControllerBudget
    timing: TimingMetrics

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _require_seed(self.seed))
        known = dict(CONDITION_MASKS)
        if type(self.condition) is not str or self.condition not in known:
            raise ValueError("condition must be a known multiagent condition name")
        if (
            type(self.learning_mask) is not tuple
            or len(self.learning_mask) != 2
            or any(type(flag) is not bool for flag in self.learning_mask)
        ):
            raise ValueError("learning_mask must be a pair of booleans")
        if self.learning_mask != known[self.condition]:
            raise ValueError("learning_mask must match the named multiagent condition")
        rewards = _require_ndarray(
            "online_rewards", self.online_rewards, dtype=np.dtype(np.float64)
        )
        if int(rewards.shape[0]) < 6 or int(rewards.shape[0]) % 3 != 0:
            raise ValueError("online_rewards must contain three equal non-empty phases")
        phase_steps = int(rewards.shape[0]) // 3
        _require_resource_limit(
            "condition result arrays",
            _condition_result_array_nbytes(phase_steps),
        )
        phase_rewards = _require_ndarray(
            "phase_mean_rewards",
            self.phase_mean_rewards,
            dtype=np.dtype(np.float64),
        )
        if phase_rewards.shape != (3,):
            raise ValueError("phase_mean_rewards must have shape (3,)")
        performance = _require_ndarray(
            "performance_matrix",
            self.performance_matrix,
            dtype=np.dtype(np.float64),
            ndim=2,
        )
        if performance.shape != (3, 2):
            raise ValueError("performance_matrix must have shape (3, 2)")
        if not all(np.all(np.isfinite(array)) for array in (rewards, phase_rewards, performance)):
            raise ValueError("condition result floating arrays must contain only finite values")
        expected_phase_rewards = np.mean(rewards.reshape(3, phase_steps), axis=1)
        if not np.array_equal(phase_rewards, expected_phase_rewards):
            raise ValueError("phase_mean_rewards must reconstruct from online_rewards")
        if type(self.summary) is not ContinualLearningSummary:
            raise ValueError("summary must be a ContinualLearningSummary")
        summary = ContinualLearningSummary(
            **{
                field.name: getattr(self.summary, field.name)
                for field in fields(ContinualLearningSummary)
            }
        )
        summary_without_reference = summarize_continual_learning(
            performance,
            [0, 1],
            rewards,
            0.0,
        )
        for name in (
            "final_performance",
            "prequential_performance",
            "mean_forgetting",
            "max_forgetting",
            "backward_transfer",
        ):
            if getattr(summary, name) != getattr(summary_without_reference, name):
                raise ValueError(f"summary.{name} must reconstruct from primitive arrays")
        for name in (
            "per_task_final_performance",
            "per_task_forgetting",
            "per_task_backward_transfer",
        ):
            if not np.array_equal(getattr(summary, name), getattr(summary_without_reference, name)):
                raise ValueError(f"summary.{name} must reconstruct from performance_matrix")
        recovery = _require_ndarray(
            "recovery_lengths",
            self.recovery_lengths,
            dtype=np.dtype(np.int64),
        )
        if recovery.shape != (2,) or np.any(recovery < -1) or np.any(recovery > phase_steps):
            raise ValueError("recovery_lengths must be a shape-(2,) vector in [-1, phase_steps]")
        recurrence = _require_int32(
            "recurrence_recovery_steps",
            self.recurrence_recovery_steps,
            minimum=-1,
            maximum=phase_steps,
        )
        if recurrence != int(recovery[1]):
            raise ValueError("recurrence_recovery_steps must match recovery_lengths[1]")
        interference = finite_real("interference_forgetting", self.interference_forgetting)
        expected_interference = max(
            0.0,
            float(performance[0, MEET_CONTEXT]) - float(performance[1, MEET_CONTEXT]),
        )
        if interference != expected_interference:
            raise ValueError("interference_forgetting must reconstruct from performance_matrix")
        if type(self.controller_budget) is not ControllerBudget:
            raise ValueError("controller_budget must be a ControllerBudget")
        budget = ControllerBudget(
            **{
                field.name: getattr(self.controller_budget, field.name)
                for field in fields(ControllerBudget)
            }
        )
        if type(self.timing) is not TimingMetrics:
            raise ValueError("timing must be a TimingMetrics")
        timing = TimingMetrics(
            **{field.name: getattr(self.timing, field.name) for field in fields(TimingMetrics)}
        )
        object.__setattr__(self, "online_rewards", _freeze_ndarray(rewards))
        object.__setattr__(self, "phase_mean_rewards", _freeze_ndarray(phase_rewards))
        object.__setattr__(self, "performance_matrix", _freeze_ndarray(performance))
        object.__setattr__(self, "recovery_lengths", _freeze_ndarray(recovery))
        object.__setattr__(self, "recurrence_recovery_steps", recurrence)
        object.__setattr__(self, "interference_forgetting", interference)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "controller_budget", budget)
        object.__setattr__(self, "timing", timing)


@dataclass(frozen=True)
class AggregateEvidence:
    """Paired multi-seed evidence used by the acceptance evaluator."""

    seeds: tuple[int, ...]
    frozen_prequential_reward: float
    learner_only_prequential_reward: float
    joint_adaptive_prequential_reward: float
    reward_uplift_over_frozen: float
    reward_uplift_interval: BootstrapInterval
    partner_uplift: float
    partner_uplift_interval: BootstrapInterval
    joint_adaptive_phase_rewards: NDArray[np.float64]
    joint_adaptive_performance_matrix: NDArray[np.float64]
    recurrent_a_probe_reward: float
    mean_forgetting: float
    max_forgetting: float
    mean_interference_forgetting: float
    recurrence_recovery_fraction: float
    mean_recurrence_recovery_steps: float
    mean_stability_gap: float
    maximum_update_latency_ms: float
    state_scalars: int
    state_bytes: int
    action_scalars_per_step: int
    budgets_identical: bool
    all_values_finite: bool

    def __post_init__(self) -> None:
        """Reject leftover seed/bool/measurement identities before acceptance."""
        if type(self.seeds) is not tuple or not self.seeds:
            raise ValueError("seeds must be a non-empty exact tuple")
        _require_resource_limit(
            "aggregate seed identities",
            len(self.seeds) * np.dtype(np.int64).itemsize,
        )
        seeds = tuple(_require_seed(seed) for seed in self.seeds)
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        object.__setattr__(self, "seeds", seeds)
        for name in (
            "frozen_prequential_reward",
            "learner_only_prequential_reward",
            "joint_adaptive_prequential_reward",
            "reward_uplift_over_frozen",
            "partner_uplift",
            "recurrent_a_probe_reward",
            "mean_forgetting",
            "max_forgetting",
            "mean_interference_forgetting",
            "recurrence_recovery_fraction",
            "mean_stability_gap",
            "maximum_update_latency_ms",
        ):
            object.__setattr__(self, name, finite_real(name, getattr(self, name)))
        mean_recovery = real_number(
            "mean_recurrence_recovery_steps",
            self.mean_recurrence_recovery_steps,
        )
        if mean_recovery < 0.0 or mean_recovery == float("-inf"):
            raise ValueError("mean_recurrence_recovery_steps must be nonnegative")
        object.__setattr__(self, "mean_recurrence_recovery_steps", mean_recovery)
        for name in (
            "mean_forgetting",
            "max_forgetting",
            "mean_interference_forgetting",
            "mean_stability_gap",
            "maximum_update_latency_ms",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 <= self.recurrence_recovery_fraction <= 1.0:
            raise ValueError("recurrence_recovery_fraction must lie in [0, 1]")
        if type(self.reward_uplift_interval) is not BootstrapInterval:
            raise ValueError("reward_uplift_interval must be a BootstrapInterval")
        if type(self.partner_uplift_interval) is not BootstrapInterval:
            raise ValueError("partner_uplift_interval must be a BootstrapInterval")
        reward_interval = BootstrapInterval(
            **{
                field.name: getattr(self.reward_uplift_interval, field.name)
                for field in fields(BootstrapInterval)
            }
        )
        partner_interval = BootstrapInterval(
            **{
                field.name: getattr(self.partner_uplift_interval, field.name)
                for field in fields(BootstrapInterval)
            }
        )
        if reward_interval.sample_size != len(seeds) or partner_interval.sample_size != len(seeds):
            raise ValueError("bootstrap interval sample_size must match seeds")
        if reward_interval.estimate != self.reward_uplift_over_frozen:
            raise ValueError("reward interval estimate must match reward_uplift_over_frozen")
        if partner_interval.estimate != self.partner_uplift:
            raise ValueError("partner interval estimate must match partner_uplift")
        object.__setattr__(self, "reward_uplift_interval", reward_interval)
        object.__setattr__(self, "partner_uplift_interval", partner_interval)
        phase_rewards = _require_ndarray(
            "joint_adaptive_phase_rewards",
            self.joint_adaptive_phase_rewards,
            dtype=np.dtype(np.float64),
            ndim=1,
        )
        performance = _require_ndarray(
            "joint_adaptive_performance_matrix",
            self.joint_adaptive_performance_matrix,
            dtype=np.dtype(np.float64),
            ndim=2,
        )
        if phase_rewards.shape != (3,):
            raise ValueError("joint_adaptive_phase_rewards must have shape (3,)")
        if performance.shape != (3, 2):
            raise ValueError("joint_adaptive_performance_matrix must have shape (3, 2)")
        if not np.all(np.isfinite(phase_rewards)) or not np.all(np.isfinite(performance)):
            raise ValueError("aggregate arrays must contain only finite values")
        if self.recurrent_a_probe_reward != float(performance[-1, MEET_CONTEXT]):
            raise ValueError(
                "recurrent_a_probe_reward must match joint_adaptive_performance_matrix"
            )
        object.__setattr__(self, "joint_adaptive_phase_rewards", _freeze_ndarray(phase_rewards))
        object.__setattr__(
            self, "joint_adaptive_performance_matrix", _freeze_ndarray(performance)
        )
        object.__setattr__(
            self, "state_scalars", _require_int32("state_scalars", self.state_scalars, minimum=0)
        )
        object.__setattr__(
            self, "state_bytes", _require_int32("state_bytes", self.state_bytes, minimum=0)
        )
        object.__setattr__(
            self,
            "action_scalars_per_step",
            _require_int32("action_scalars_per_step", self.action_scalars_per_step, minimum=1),
        )
        if type(self.budgets_identical) is not bool:
            raise ValueError("budgets_identical must be an exact bool")
        if type(self.all_values_finite) is not bool:
            raise ValueError("all_values_finite must be an exact bool")
        if not self.all_values_finite:
            raise ValueError("all_values_finite must match validated finite aggregate payloads")


@dataclass(frozen=True)
class AcceptanceEvidence:
    """One fail-closed threshold comparison and its observed value."""

    name: str
    passed: bool
    actual: float
    comparator: str
    threshold: float
    detail: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("name must be a non-empty string")
        if type(self.passed) is not bool:
            raise ValueError("passed must be a boolean")
        actual = real_number("actual", self.actual)
        if self.passed and not np.isfinite(actual):
            raise ValueError("a passed check must have a finite actual value")
        object.__setattr__(self, "actual", actual)
        if type(self.comparator) is not str or not self.comparator:
            raise ValueError("comparator must be a non-empty string")
        object.__setattr__(self, "threshold", finite_real("threshold", self.threshold))
        if type(self.detail) is not str:
            raise ValueError("detail must be a string")


@dataclass(frozen=True)
class AcceptanceResult:
    """Acceptance never skips: every failed check carries numeric evidence."""

    passed: bool
    checks: tuple[AcceptanceEvidence, ...]

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise ValueError("passed must be a boolean")
        if not isinstance(self.checks, tuple) or not all(
            isinstance(c, AcceptanceEvidence) for c in self.checks
        ):
            raise ValueError("checks must be a tuple of AcceptanceEvidence")

    @property
    def failures(self) -> tuple[AcceptanceEvidence, ...]:
        """Return all failed checks without hiding later failures."""

        return tuple(check for check in self.checks if not check.passed)


@dataclass(frozen=True)
class ContinualMultiAgentReport:
    """Complete benchmark report across paired seeds and conditions."""

    config: ContinualMultiAgentConfig
    thresholds: AcceptanceThresholds
    condition_results: tuple[ConditionResult, ...]
    aggregate: AggregateEvidence
    acceptance: AcceptanceResult


@dataclass
class _ControllerState:
    """Fixed-size contextual values plus counter-based random state."""

    action_values: NDArray[np.float32]
    random_state: NDArray[np.uint64]


def _initial_controller(seed: int) -> _ControllerState:
    return _ControllerState(
        action_values=np.zeros((2, 2, 2), dtype=np.float32),
        random_state=np.asarray((seed, 0), dtype=np.uint64),
    )


def _controller_budget(controller: _ControllerState) -> ControllerBudget:
    arrays = (controller.action_values, controller.random_state)
    return ControllerBudget(
        state_scalars=sum(int(array.size) for array in arrays),
        state_bytes=sum(int(array.nbytes) for array in arrays),
        action_scalars_per_step=2,
    )


def _visible_contexts(observation: NDArray[np.float32]) -> NDArray[np.int64]:
    context_cues = observation[:, MEET_CONTEXT_INDEX : AVOID_CONTEXT_INDEX + 1]
    return np.argmax(context_cues, axis=1).astype(np.int64)


def _select_actions(
    controller: _ControllerState,
    observation: NDArray[np.float32],
    exploration_rate: float,
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]]:
    """Select two actions using only ordinary observation and stored state."""

    contexts = _visible_contexts(observation)
    seed = int(controller.random_state[0])
    step = int(controller.random_state[1])
    # Counter-based stream: draws are a pure function of (seed, step), so all
    # three conditions of one seed see identical exploration randomness (the
    # common random numbers the paired bootstrap relies on) at a fixed
    # two-scalar state cost.  Distinct (seed, step) pairs map to distinct
    # stream seeds because the multiplier 100_003 (prime) exceeds any life
    # length used here (3 * phase_steps = 192 at the defaults).
    generator = np.random.default_rng(seed * 100_003 + step)
    explore_draws = generator.random(2)
    random_actions = generator.integers(0, 2, size=2, dtype=np.int64)

    action_indices = np.empty((2,), dtype=np.int64)
    for agent in range(2):
        values = controller.action_values[agent, contexts[agent]]
        tied = bool(values[0] == values[1])
        if tied or explore_draws[agent] < exploration_rate:
            action_indices[agent] = random_actions[agent]
        else:
            action_indices[agent] = int(np.argmax(values))
    actions = (2 * action_indices - 1).astype(np.float32)
    return actions, action_indices, contexts


def _update_controller(
    controller: _ControllerState,
    action_indices: NDArray[np.int64],
    contexts: NDArray[np.int64],
    reward: float,
    learning_rate: float,
    learning_mask: tuple[bool, bool],
) -> _ControllerState:
    """Perform one post-reward update; frozen agents suppress candidate writes."""

    values = controller.action_values.copy()
    for agent in range(2):
        action = int(action_indices[agent])
        context = int(contexts[agent])
        old_value = float(values[agent, context, action])
        candidate = old_value + learning_rate * (reward - old_value)
        if learning_mask[agent]:
            values[agent, context, action] = np.float32(candidate)

    random_state = controller.random_state.copy()
    random_state[1] += np.uint64(1)
    return _ControllerState(action_values=values, random_state=random_state)


def _probe_state(
    world: RecurringTwoAgentWorld,
    *,
    context: int,
    phase_steps: int,
) -> RecurringTwoAgentState:
    initial = world.init(jr.key(2_147_483_647))
    return RecurringTwoAgentState(  # type: ignore[call-arg]
        key=initial.key,
        positions=jnp.asarray((-0.5, 0.5), dtype=jnp.float32),
        velocities=jnp.zeros((2,), dtype=jnp.float32),
        nuisance=jnp.zeros_like(initial.nuisance),
        step_count=jnp.asarray(context * phase_steps, dtype=jnp.int32),
    )


def _probe_policy(
    world: RecurringTwoAgentWorld,
    step_world: StepFunction,
    controller: _ControllerState,
    *,
    context: int,
    config: ContinualMultiAgentConfig,
) -> float:
    """Read-only deterministic rollout from a fixed evaluator-owned state."""

    state = _probe_state(world, context=context, phase_steps=config.phase_steps)
    action_preferences = (
        controller.action_values[:, context, 1] - controller.action_values[:, context, 0]
    )
    actions = np.sign(action_preferences).astype(np.float32)
    rewards = np.empty((config.probe_horizon,), dtype=np.float64)
    for step_index in range(config.probe_horizon):
        transition, state = step_world(state, jnp.asarray(actions))
        rewards[step_index] = float(transition.reward[0])
    return float(np.mean(rewards[-config.probe_tail_steps :]))


def _run_condition(
    world: RecurringTwoAgentWorld,
    step_world: StepFunction,
    *,
    seed: int,
    condition: ConditionName,
    learning_mask: tuple[bool, bool],
    config: ContinualMultiAgentConfig,
) -> ConditionResult:
    state = world.init(jr.key(seed))
    observation = np.asarray(world.observe(state), dtype=np.float32)
    controller = _initial_controller(seed)
    total_steps = 3 * config.phase_steps
    rewards = np.empty((total_steps,), dtype=np.float64)
    update_latencies_ns = np.empty((total_steps,), dtype=np.float64)
    performance_rows: list[list[float]] = []

    wall_start = perf_counter()
    for step_index in range(total_steps):
        actions, action_indices, contexts = _select_actions(
            controller,
            observation,
            config.exploration_rate,
        )
        transition, state = step_world(state, jnp.asarray(actions))
        reward = float(transition.reward[0])
        rewards[step_index] = reward
        observation = np.asarray(transition.next_observation, dtype=np.float32)

        update_start = perf_counter_ns()
        controller = _update_controller(
            controller,
            action_indices,
            contexts,
            reward,
            config.learning_rate,
            learning_mask,
        )
        update_latencies_ns[step_index] = perf_counter_ns() - update_start

        # Evaluator-only checkpoint: the continuing state and learner are not
        # modified by either counterfactual rollout.
        if (step_index + 1) % config.phase_steps == 0:
            performance_rows.append(
                [
                    _probe_policy(
                        world,
                        step_world,
                        controller,
                        context=MEET_CONTEXT,
                        config=config,
                    ),
                    _probe_policy(
                        world,
                        step_world,
                        controller,
                        context=AVOID_CONTEXT,
                        config=config,
                    ),
                ]
            )
    wall_seconds = perf_counter() - wall_start

    performance_matrix = np.asarray(performance_rows, dtype=np.float64)
    phase_mean_rewards = np.mean(
        rewards.reshape(3, config.phase_steps),
        axis=1,
    )
    summary = summarize_continual_learning(
        performance_matrix,
        [0, 1],
        rewards,
        config.stability_reference_reward,
    )
    recovery_lengths = compute_recovery_lengths(
        rewards,
        [config.phase_steps, 2 * config.phase_steps],
        config.recovery_reward_threshold,
        window_size=config.recovery_window,
    )
    interference_forgetting = max(
        0.0,
        float(performance_matrix[0, MEET_CONTEXT]) - float(performance_matrix[1, MEET_CONTEXT]),
    )
    update_latency_ms = update_latencies_ns / 1_000_000.0
    return ConditionResult(
        seed=seed,
        condition=condition,
        learning_mask=learning_mask,
        online_rewards=rewards,
        phase_mean_rewards=np.asarray(phase_mean_rewards, dtype=np.float64),
        performance_matrix=performance_matrix,
        summary=summary,
        recovery_lengths=recovery_lengths,
        recurrence_recovery_steps=int(recovery_lengths[1]),
        interference_forgetting=interference_forgetting,
        controller_budget=_controller_budget(controller),
        timing=TimingMetrics(
            wall_seconds=wall_seconds,
            mean_step_latency_ms=1_000.0 * wall_seconds / total_steps,
            mean_update_latency_ms=float(np.mean(update_latency_ms)),
            p95_update_latency_ms=float(np.percentile(update_latency_ms, 95)),
        ),
    )


def _condition_results(
    results: Sequence[ConditionResult],
    condition: ConditionName,
) -> tuple[ConditionResult, ...]:
    selected = tuple(result for result in results if result.condition == condition)
    if not selected:
        raise ValueError(f"missing required condition: {condition}")
    return selected


def paired_bootstrap_mean_interval(
    paired_differences: NDArray[np.float64],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> BootstrapInterval:
    """Return a reproducible percentile interval for a paired mean.

    The caller supplies one already-paired difference per seed.  Resampling
    those differences, rather than independently resampling each condition,
    preserves the common-random-number experimental design.
    """

    values = np.asarray(paired_differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired_differences must be a non-empty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("paired_differences must contain only finite values")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    resamples = _require_int32(
        "resamples",
        resamples,
        minimum=1,
        maximum=_MAX_CONFIGURED_ARRAY_NBYTES // (3 * np.dtype(np.float64).itemsize),
    )
    seed = _require_uint32("seed", seed)
    _require_resource_limit(
        "paired bootstrap working arrays",
        _bootstrap_working_nbytes(resamples, int(values.size)),
    )

    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        values.size,
        size=(resamples, values.size),
        dtype=np.int64,
    )
    bootstrap_means = np.mean(values[indices], axis=1)
    tail_probability = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        bootstrap_means,
        (tail_probability, 1.0 - tail_probability),
    )
    return BootstrapInterval(
        estimate=float(np.mean(values)),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        resamples=resamples,
        sample_size=int(values.size),
    )


def aggregate_evidence(
    results: Sequence[ConditionResult],
    *,
    config: ContinualMultiAgentConfig,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 2_026_073_000,
) -> AggregateEvidence:
    """Aggregate paired condition results without discarding failed seeds."""

    if type(config) is not ContinualMultiAgentConfig:
        raise ValueError("config must be a ContinualMultiAgentConfig")
    config = ContinualMultiAgentConfig(
        **{field.name: getattr(config, field.name) for field in fields(ContinualMultiAgentConfig)}
    )
    if type(results) not in (list, tuple):
        raise ValueError("results must be an exact list or tuple")
    if not results:
        raise ValueError("results must be non-empty")
    _require_resource_limit(
        "aggregate condition result arrays",
        len(results) * _condition_result_array_nbytes(config.phase_steps),
    )
    validated: list[ConditionResult] = []
    for result in results:
        if type(result) is not ConditionResult:
            raise ValueError("results must contain exact ConditionResult records")
        if type(result.controller_budget) is not ControllerBudget:
            raise ValueError("controller_budget must be a ControllerBudget")
        if type(result.timing) is not TimingMetrics:
            raise ValueError("timing must be a TimingMetrics")
        budget = ControllerBudget(
            **{
                field.name: getattr(result.controller_budget, field.name)
                for field in fields(ControllerBudget)
            }
        )
        timing = TimingMetrics(
            **{field.name: getattr(result.timing, field.name) for field in fields(TimingMetrics)}
        )
        result = ConditionResult(
            seed=result.seed,
            condition=result.condition,
            learning_mask=result.learning_mask,
            online_rewards=result.online_rewards,
            phase_mean_rewards=result.phase_mean_rewards,
            performance_matrix=result.performance_matrix,
            summary=result.summary,
            recovery_lengths=result.recovery_lengths,
            recurrence_recovery_steps=result.recurrence_recovery_steps,
            interference_forgetting=result.interference_forgetting,
            controller_budget=budget,
            timing=timing,
        )
        if result.online_rewards.size != 3 * config.phase_steps:
            raise ValueError("result length does not match config.phase_steps")
        expected_recoveries = compute_recovery_lengths(
            result.online_rewards,
            [config.phase_steps, 2 * config.phase_steps],
            config.recovery_reward_threshold,
            window_size=config.recovery_window,
        )
        if not np.array_equal(result.recovery_lengths, expected_recoveries):
            raise ValueError("recovery_lengths do not match rewards and config")
        expected_summary = summarize_continual_learning(
            result.performance_matrix,
            [0, 1],
            result.online_rewards,
            config.stability_reference_reward,
        )
        summary_matches = all(
            np.array_equal(
                getattr(result.summary, field.name),
                getattr(expected_summary, field.name),
            )
            if isinstance(getattr(result.summary, field.name), np.ndarray)
            else getattr(result.summary, field.name) == getattr(expected_summary, field.name)
            for field in fields(ContinualLearningSummary)
        )
        if not summary_matches:
            raise ValueError("summary does not match primitive arrays and config")
        validated.append(result)
    results = tuple(validated)
    frozen = _condition_results(results, "frozen")
    learner_only = _condition_results(results, "learner_only")
    joint_adaptive = _condition_results(results, "joint_adaptive")
    seed_sets = tuple(
        tuple(result.seed for result in group) for group in (frozen, learner_only, joint_adaptive)
    )
    if seed_sets[0] != seed_sets[1] or seed_sets[0] != seed_sets[2]:
        raise ValueError("conditions must contain identical seeds in identical order")
    if len(set(seed_sets[0])) != len(seed_sets[0]):
        raise ValueError("paired condition seeds must be unique")

    frozen_rewards = np.asarray(
        [result.summary.prequential_performance for result in frozen],
        dtype=np.float64,
    )
    learner_only_rewards = np.asarray(
        [result.summary.prequential_performance for result in learner_only],
        dtype=np.float64,
    )
    adaptive_rewards = np.asarray(
        [result.summary.prequential_performance for result in joint_adaptive],
        dtype=np.float64,
    )
    reward_uplift_differences = adaptive_rewards - frozen_rewards
    partner_uplift_differences = adaptive_rewards - learner_only_rewards
    reward_uplift_interval = paired_bootstrap_mean_interval(
        reward_uplift_differences,
        confidence_level=confidence_level,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    partner_uplift_interval = paired_bootstrap_mean_interval(
        partner_uplift_differences,
        confidence_level=confidence_level,
        resamples=bootstrap_resamples,
        seed=(bootstrap_seed + 1) % 2**32,
    )
    adaptive_phase_rewards = np.mean(
        np.stack([result.phase_mean_rewards for result in joint_adaptive]),
        axis=0,
    )
    adaptive_performance_matrix = np.mean(
        np.stack([result.performance_matrix for result in joint_adaptive]),
        axis=0,
    )
    recurrence_steps = np.asarray(
        [result.recurrence_recovery_steps for result in joint_adaptive],
        dtype=np.int64,
    )
    recovered = recurrence_steps >= 0
    mean_recovery = (
        float(np.mean(recurrence_steps[recovered])) if np.any(recovered) else float("inf")
    )
    budgets = tuple(result.controller_budget for result in results)
    first_budget = budgets[0]
    budgets_identical = all(budget == first_budget for budget in budgets[1:])

    numeric_arrays = (
        [result.online_rewards for result in results]
        + [result.phase_mean_rewards for result in results]
        + [result.performance_matrix for result in results]
    )
    timings = np.asarray(
        [
            (
                result.timing.wall_seconds,
                result.timing.mean_step_latency_ms,
                result.timing.mean_update_latency_ms,
                result.timing.p95_update_latency_ms,
            )
            for result in results
        ],
        dtype=np.float64,
    )
    all_values_finite = all(np.all(np.isfinite(array)) for array in numeric_arrays) and bool(
        np.all(np.isfinite(timings))
    )

    return AggregateEvidence(
        seeds=seed_sets[0],
        frozen_prequential_reward=float(np.mean(frozen_rewards)),
        learner_only_prequential_reward=float(np.mean(learner_only_rewards)),
        joint_adaptive_prequential_reward=float(np.mean(adaptive_rewards)),
        reward_uplift_over_frozen=reward_uplift_interval.estimate,
        reward_uplift_interval=reward_uplift_interval,
        partner_uplift=partner_uplift_interval.estimate,
        partner_uplift_interval=partner_uplift_interval,
        joint_adaptive_phase_rewards=np.asarray(adaptive_phase_rewards, dtype=np.float64),
        joint_adaptive_performance_matrix=np.asarray(
            adaptive_performance_matrix,
            dtype=np.float64,
        ),
        recurrent_a_probe_reward=float(adaptive_performance_matrix[-1, MEET_CONTEXT]),
        mean_forgetting=float(
            np.mean([result.summary.mean_forgetting for result in joint_adaptive])
        ),
        max_forgetting=float(np.max([result.summary.max_forgetting for result in joint_adaptive])),
        mean_interference_forgetting=float(
            np.mean([result.interference_forgetting for result in joint_adaptive])
        ),
        recurrence_recovery_fraction=float(np.mean(recovered)),
        mean_recurrence_recovery_steps=mean_recovery,
        mean_stability_gap=float(
            np.mean([result.summary.stability_gap_mean for result in joint_adaptive])
        ),
        maximum_update_latency_ms=float(
            np.max([result.timing.p95_update_latency_ms for result in results])
        ),
        state_scalars=first_budget.state_scalars,
        state_bytes=first_budget.state_bytes,
        action_scalars_per_step=first_budget.action_scalars_per_step,
        budgets_identical=budgets_identical,
        all_values_finite=all_values_finite,
    )


def _minimum_check(
    name: str,
    actual: float,
    threshold: float,
    detail: str,
) -> AcceptanceEvidence:
    passed = bool(np.isfinite(actual) and actual >= threshold)
    return AcceptanceEvidence(
        name=name,
        passed=passed,
        actual=actual,
        comparator=">=",
        threshold=threshold,
        detail=detail,
    )


def _maximum_check(
    name: str,
    actual: float,
    threshold: float,
    detail: str,
) -> AcceptanceEvidence:
    passed = bool(np.isfinite(actual) and actual <= threshold)
    return AcceptanceEvidence(
        name=name,
        passed=passed,
        actual=actual,
        comparator="<=",
        threshold=threshold,
        detail=detail,
    )


def evaluate_acceptance(
    aggregate: AggregateEvidence,
    thresholds: AcceptanceThresholds | None = None,
) -> AcceptanceResult:
    """Evaluate every threshold and return evidence for every failure.

    Missing/non-finite evidence fails the relevant comparison.  There is no
    skipped or "not evaluated" acceptance state.
    """

    limits = AcceptanceThresholds() if thresholds is None else thresholds
    seed_schedule_matches = len(aggregate.seeds) == limits.minimum_seed_count and all(
        seed == limits.evidence_seed_start + offset for offset, seed in enumerate(aggregate.seeds)
    )
    checks = (
        _minimum_check(
            "seed_count",
            float(len(aggregate.seeds)),
            float(limits.minimum_seed_count),
            "Promoted evidence requires the pre-registered number of unique paired seeds.",
        ),
        _minimum_check(
            "evidence_seed_schedule",
            float(seed_schedule_matches),
            1.0,
            "Promoted evidence must use the frozen held-out consecutive seed schedule.",
        ),
        _minimum_check(
            "all_values_finite",
            float(aggregate.all_values_finite),
            1.0,
            "Every reward, probe, and timing value must be finite.",
        ),
        _minimum_check(
            "budgets_identical",
            float(aggregate.budgets_identical),
            1.0,
            "All conditions must use identical persistent state and actions.",
        ),
        _minimum_check(
            "reward_uplift_over_frozen",
            aggregate.reward_uplift_interval.lower,
            limits.minimum_reward_uplift_over_frozen,
            "Lower confidence bound for paired joint-adaptive minus frozen reward.",
        ),
        _minimum_check(
            "partner_uplift",
            aggregate.partner_uplift_interval.lower,
            limits.minimum_partner_uplift,
            "Lower confidence bound for paired joint-adaptive minus learner-only "
            "reward (coadaptation, not an IA intervention).",
        ),
        _minimum_check(
            "recurrent_a_probe_reward",
            aggregate.recurrent_a_probe_reward,
            limits.minimum_recurrent_a_probe_reward,
            "Read-only A probe after the full A-B-A life.",
        ),
        _maximum_check(
            "mean_forgetting",
            aggregate.mean_forgetting,
            limits.maximum_mean_forgetting,
            "Peak-to-final degradation across both probe tasks.",
        ),
        _maximum_check(
            "interference_forgetting",
            aggregate.mean_interference_forgetting,
            limits.maximum_interference_forgetting,
            "A-probe degradation from immediately after A1 to after B.",
        ),
        _minimum_check(
            "recurrence_recovery_fraction",
            aggregate.recurrence_recovery_fraction,
            limits.minimum_recurrence_recovery_fraction,
            "Fraction of seeds recovering during recurrent A.",
        ),
        _maximum_check(
            "mean_recurrence_recovery_steps",
            aggregate.mean_recurrence_recovery_steps,
            limits.maximum_mean_recurrence_recovery_steps,
            "Mean steps to a sustained reward threshold among seeds that "
            "recovered after A recurred; recovery fraction is gated separately.",
        ),
        _maximum_check(
            "mean_stability_gap",
            aggregate.mean_stability_gap,
            limits.maximum_mean_stability_gap,
            "Online reward deficit below the configured reference.",
        ),
        _maximum_check(
            "update_latency_ms",
            aggregate.maximum_update_latency_ms,
            limits.maximum_update_latency_ms,
            "Worst per-run p95 controller-update latency.",
        ),
    )
    return AcceptanceResult(
        passed=all(check.passed for check in checks),
        checks=checks,
    )


def run_continual_multiagent_benchmark(
    *,
    seeds: Sequence[int] = tuple(range(30, 60)),
    config: ContinualMultiAgentConfig | None = None,
    thresholds: AcceptanceThresholds | None = None,
) -> ContinualMultiAgentReport:
    """Run paired seeded conditions and return a fail-closed report."""

    benchmark_config = ContinualMultiAgentConfig() if config is None else config
    acceptance_thresholds = AcceptanceThresholds() if thresholds is None else thresholds
    if type(seeds) not in (list, tuple, range):
        raise ValueError("seeds must be an actual list, tuple, or range")
    seed_count = len(seeds)
    if seed_count == 0:
        raise ValueError("seeds must be non-empty")
    _require_resource_limit(
        "retained condition-result arrays",
        seed_count
        * len(CONDITION_MASKS)
        * _condition_result_array_nbytes(benchmark_config.phase_steps),
    )
    seed_tuple = tuple(_require_seed(seed) for seed in seeds)
    if len(set(seed_tuple)) != len(seed_tuple):
        raise ValueError("seeds must be unique")

    # Immediate dynamics make the causal effect of each bounded action visible
    # while preserving an uninterrupted physical state across phase switches.
    world = RecurringTwoAgentWorld(
        context_length=benchmark_config.phase_steps,
        nuisance_dim=benchmark_config.nuisance_dim,
        damping=0.0,
        acceleration=0.25,
        max_speed=0.25,
    )
    step_world: StepFunction = jax.jit(world.step)
    warm_state = world.init(jr.key(seed_tuple[0]))
    warm_transition, _ = step_world(warm_state, jnp.zeros((2,), dtype=jnp.float32))
    warm_transition.reward.block_until_ready()

    results = tuple(
        _run_condition(
            world,
            step_world,
            seed=seed,
            condition=condition,
            learning_mask=mask,
            config=benchmark_config,
        )
        for seed in seed_tuple
        for condition, mask in CONDITION_MASKS
    )
    aggregate = aggregate_evidence(
        results,
        config=benchmark_config,
        confidence_level=benchmark_config.confidence_level,
        bootstrap_resamples=benchmark_config.bootstrap_resamples,
        bootstrap_seed=benchmark_config.bootstrap_seed,
    )
    acceptance = evaluate_acceptance(aggregate, acceptance_thresholds)
    return ContinualMultiAgentReport(
        config=benchmark_config,
        thresholds=acceptance_thresholds,
        condition_results=results,
        aggregate=aggregate,
        acceptance=acceptance,
    )
