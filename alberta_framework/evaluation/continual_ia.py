"""Held-out causal evaluation for the narrow Step-12 IA prototype.

The primary estimand is the paired reward difference between two conditions
with identical partner, IA, recommendation-protocol, random-number, and state
budgets:

``recommendation_p05 - observe_only``.

Both arms run the same IA computation and credit the exo-cortex with the
primitive action actually executed.  ``observe_only`` sets recommendation
acceptance to zero and must reproduce ``partner_alone`` bitwise.  The treatment
accepts with the development-selected probability 0.5.  An intervention is
counted only on a primitive transition where an accepted recommendation
differs from the partner proposal and therefore changes the executed action.

Seeds 0--11 and the selected hyperparameters are development/calibration data.
The promoted schedule is the disjoint, frozen seed set 30--59.  This evaluator
is intentionally limited to a deterministic, hand-designed, hidden-phase
two-state MDP.  It does not demonstrate a general partner effect, autonomous
feature discovery, or completion of the Alberta Plan.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from numpy.typing import NDArray

from alberta_framework._seed_validation import JAX_KEY_SEED_MAX
from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    DifferentialSARSAState,
)
from alberta_framework.core.intelligence_amplification import (
    ExoCerebellumConfig,
    IAAgent,
    IAConfig,
    IAState,
    RecommendationProtocolConfig,
    RecommendationProtocolState,
    init_recommendation_protocol_state,
    update_recommendation_protocol,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.evaluation._measurement_validation import (
    finite_real,
    nonnegative_finite_real,
    validate_interval_bounds,
)
from alberta_framework.streams.closed_loop import (
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
    SwitchingTwoStateState,
)

type IAConditionName = Literal[
    "partner_alone",
    "observe_only",
    "recommendation_p05",
    "accept_always",
    "augmented_predictions",
    "augmented_noise",
]

DEVELOPMENT_SEEDS = tuple(range(12))
PROMOTED_EVIDENCE_SEEDS = tuple(range(30, 60))
CONDITION_NAMES: tuple[IAConditionName, ...] = (
    "partner_alone",
    "observe_only",
    "recommendation_p05",
    "accept_always",
    "augmented_predictions",
    "augmented_noise",
)
RECOMMENDATION_CONDITIONS: tuple[IAConditionName, ...] = (
    "observe_only",
    "recommendation_p05",
    "accept_always",
)

_INT32_MAX = 2**31 - 1
_MAX_BOOTSTRAP_RESAMPLES = 1_000_000
_MAX_BOOTSTRAP_DRAW_COUNT = 50_000_000
_ACTUAL_INT_TYPES = frozenset(
    {int, *(np.dtype(code).type for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"))}
)


def _require_integer(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int = _INT32_MAX,
) -> int:
    """Canonicalize only trusted Python/NumPy integer scalar families."""
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _require_derived_int32(name: str, value: int) -> None:
    if not 0 <= value <= _INT32_MAX:
        raise ValueError(f"derived {name} must be in [0, {_INT32_MAX}]")


def _preflight_static_resources(
    *,
    num_steps: int,
    phase_length: int,
    n_demons: int,
) -> None:
    """Bound all config-only controller, scan, history, and work dimensions."""
    n_phases = num_steps // phase_length
    # Exact state-tree measurements for the fixed two-state/two-action/one-option
    # protocol. Host timing floats are included at their deployed eight bytes.
    recommendation_state_scalars = 113 + 2 * n_demons
    recommendation_state_bytes = 468 + 8 * n_demons
    augmented_state_scalars = 103 + 7 * n_demons
    augmented_state_bytes = 428 + 28 * n_demons
    # The recommendation scan retains 21 bytes/step of outputs while its
    # uint32 split-key schedule is live. The three recommendation results each
    # retain 41 bytes/step; the three plain results each retain 33 because one
    # int64 sentinel vector is shared by two result fields. Every arm also
    # retains float64 phase means and int64 recovery lengths.
    recommendation_scan_and_key_bytes = 29 * num_steps
    per_seed_history_bytes = 222 * num_steps + 96 * n_phases - 48
    per_seed_transition_work = len(CONDITION_NAMES) * num_steps
    scan_key_bytes = 8 * num_steps
    for name, value in (
        ("n_phases", n_phases),
        ("recommendation_controller_state_scalars", recommendation_state_scalars),
        ("recommendation_controller_state_bytes", recommendation_state_bytes),
        ("augmented_controller_state_scalars", augmented_state_scalars),
        ("augmented_controller_state_bytes", augmented_state_bytes),
        ("scan_key_bytes", scan_key_bytes),
        ("recommendation_scan_and_key_bytes", recommendation_scan_and_key_bytes),
        ("per_seed_history_bytes", per_seed_history_bytes),
        ("per_seed_transition_work", per_seed_transition_work),
    ):
        _require_derived_int32(name, value)


def _preflight_run_resources(config: ContinualIAConfig, seed_count: int) -> None:
    """Bound retained histories, total transitions, and bootstrap peak arrays."""
    per_seed_history_bytes = 222 * config.num_steps + 96 * config.n_phases - 48
    total_history_bytes = seed_count * per_seed_history_bytes
    total_transition_work = len(CONDITION_NAMES) * seed_count * config.num_steps
    bootstrap_draw_count = config.bootstrap_resamples * seed_count
    bootstrap_peak_bytes = 16 * bootstrap_draw_count + 8 * config.bootstrap_resamples
    _require_derived_int32("total_history_bytes", total_history_bytes)
    _require_derived_int32("total_transition_work", total_transition_work)
    _require_derived_int32("bootstrap_peak_bytes", bootstrap_peak_bytes)
    if bootstrap_draw_count > _MAX_BOOTSTRAP_DRAW_COUNT:
        raise ValueError(
            "derived bootstrap_draw_count must not exceed "
            f"{_MAX_BOOTSTRAP_DRAW_COUNT}"
        )


@dataclass(frozen=True)
class ContinualIAConfig:
    """Frozen environment, learner, intervention, and statistics protocol."""

    num_steps: int = 1_200
    phase_length: int = 200
    observation_dim: int = 2
    n_actions: int = 2
    n_demons: int = 2
    partner_q_step_size: float = 0.02
    partner_average_reward_step_size: float = 0.01
    partner_epsilon: float = 0.10
    cortex_base_step_size: float = 0.30
    recommendation_acceptance_probability: float = 0.50
    recovery_window: int = 20
    recovery_mean_reward_threshold: float = 0.75
    bootstrap_resamples: int = 10_000
    confidence_level: float = 0.95
    bootstrap_seed: int = 2_026_073_012

    def __post_init__(self) -> None:
        num_steps = _require_integer("num_steps", self.num_steps, minimum=1)
        phase_length = _require_integer(
            "phase_length", self.phase_length, minimum=1, maximum=num_steps
        )
        observation_dim = _require_integer(
            "observation_dim", self.observation_dim, minimum=2, maximum=2
        )
        n_actions = _require_integer("n_actions", self.n_actions, minimum=2, maximum=2)
        n_demons = _require_integer("n_demons", self.n_demons, minimum=1)
        recovery_window = _require_integer(
            "recovery_window", self.recovery_window, minimum=1, maximum=phase_length
        )
        bootstrap_resamples = _require_integer(
            "bootstrap_resamples",
            self.bootstrap_resamples,
            minimum=1_000,
            maximum=_MAX_BOOTSTRAP_RESAMPLES,
        )
        bootstrap_seed = _require_integer(
            "bootstrap_seed", self.bootstrap_seed, minimum=0, maximum=JAX_KEY_SEED_MAX
        )
        if num_steps % phase_length != 0:
            raise ValueError("num_steps must be divisible by phase_length")
        _preflight_static_resources(
            num_steps=num_steps,
            phase_length=phase_length,
            n_demons=n_demons,
        )
        object.__setattr__(self, "num_steps", num_steps)
        object.__setattr__(self, "phase_length", phase_length)
        object.__setattr__(self, "observation_dim", observation_dim)
        object.__setattr__(self, "n_actions", n_actions)
        object.__setattr__(self, "n_demons", n_demons)
        object.__setattr__(self, "recovery_window", recovery_window)
        object.__setattr__(self, "bootstrap_resamples", bootstrap_resamples)
        object.__setattr__(self, "bootstrap_seed", bootstrap_seed)
        for name, bounds in (
            ("partner_q_step_size", {"lower": 0.0}),
            ("partner_average_reward_step_size", {"lower": 0.0}),
            ("partner_epsilon", {"lower": 0.0, "upper": 1.0}),
            ("cortex_base_step_size", {"lower": 0.0}),
            (
                "recommendation_acceptance_probability",
                {"lower": 0.0, "upper": 1.0},
            ),
            ("recovery_mean_reward_threshold", {"lower": 0.0, "upper": 1.0}),
            (
                "confidence_level",
                {"positive": True, "upper": 1.0, "upper_inclusive": False},
            ),
        ):
            object.__setattr__(
                self,
                name,
                validated_float32_scalar(
                    name, getattr(self, name), **cast(dict[str, Any], bounds)
                ),
            )

    @property
    def n_phases(self) -> int:
        return self.num_steps // self.phase_length


@dataclass(frozen=True)
class IAAcceptanceThresholds:
    """Preregistered held-out thresholds, frozen before seeds 30--59."""

    minimum_seed_count: int = 30
    evidence_seed_start: int = 30
    minimum_primary_uplift_lower_ci: float = 0.10
    minimum_changed_action_intervention_rate: float = 0.10
    minimum_augmentation_vs_alone_lower_ci: float = 0.05
    minimum_augmentation_vs_noise_lower_ci: float = 0.05
    require_observe_only_exact_identity: bool = True
    maximum_executed_action_credit_mismatches: int = 0

    def __post_init__(self) -> None:
        minimum_seed_count = _require_integer(
            "minimum_seed_count", self.minimum_seed_count, minimum=1
        )
        evidence_seed_start = _require_integer(
            "evidence_seed_start",
            self.evidence_seed_start,
            minimum=0,
            maximum=JAX_KEY_SEED_MAX,
        )
        maximum_mismatches = _require_integer(
            "maximum_executed_action_credit_mismatches",
            self.maximum_executed_action_credit_mismatches,
            minimum=0,
        )
        schedule_end = evidence_seed_start + minimum_seed_count
        if schedule_end > JAX_KEY_SEED_MAX + 1:
            raise ValueError("derived evidence seed schedule exceeds the uint32 JAX key domain")
        object.__setattr__(self, "minimum_seed_count", minimum_seed_count)
        object.__setattr__(self, "evidence_seed_start", evidence_seed_start)
        object.__setattr__(
            self, "maximum_executed_action_credit_mismatches", maximum_mismatches
        )
        for name, bounds in (
            ("minimum_primary_uplift_lower_ci", {"lower": -1.0, "upper": 1.0}),
            (
                "minimum_changed_action_intervention_rate",
                {"lower": 0.0, "upper": 1.0},
            ),
            ("minimum_augmentation_vs_alone_lower_ci", {"lower": -1.0, "upper": 1.0}),
            ("minimum_augmentation_vs_noise_lower_ci", {"lower": -1.0, "upper": 1.0}),
        ):
            object.__setattr__(
                self,
                name,
                validated_float32_scalar(
                    name, getattr(self, name), **cast(dict[str, Any], bounds)
                ),
            )
        if type(self.require_observe_only_exact_identity) is not bool:
            raise ValueError("require_observe_only_exact_identity must be an actual bool")


@dataclass(frozen=True)
class PairedBootstrapInterval:
    """Deterministic percentile interval over one paired seed-difference vector."""

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
            _require_integer(
                "resamples",
                self.resamples,
                minimum=1,
                maximum=_MAX_BOOTSTRAP_RESAMPLES,
            ),
        )
        object.__setattr__(
            self,
            "sample_size",
            _require_integer("sample_size", self.sample_size, minimum=1),
        )
        if type(self.method) is not str or not self.method:
            raise ValueError("method must be a non-empty string")
        validate_interval_bounds(
            lower=self.lower,
            upper=self.upper,
            confidence_level=self.confidence_level,
        )


@dataclass(frozen=True)
class ControllerBudget:
    """Persistent controller state and per-step interface budget."""

    state_scalars: int
    state_bytes: int
    observation_scalars: int
    action_scalars_per_step: int
    interaction_steps: int
    ia_attached: bool

    def __post_init__(self) -> None:
        for name, minimum in (
            ("state_scalars", 0),
            ("state_bytes", 0),
            ("observation_scalars", 1),
            ("action_scalars_per_step", 1),
            ("interaction_steps", 1),
        ):
            object.__setattr__(
                self,
                name,
                _require_integer(name, getattr(self, name), minimum=minimum),
            )
        if type(self.ia_attached) is not bool:
            raise ValueError("ia_attached must be an actual bool")


@dataclass(frozen=True)
class ConditionTiming:
    """Host-dependent wall-clock diagnostic, excluded from scientific digest."""

    wall_seconds: float
    mean_step_latency_ms: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "wall_seconds",
            nonnegative_finite_real("wall_seconds", self.wall_seconds),
        )
        object.__setattr__(
            self,
            "mean_step_latency_ms",
            nonnegative_finite_real("mean_step_latency_ms", self.mean_step_latency_ms),
        )


@dataclass(frozen=True)
class IAConditionResult:
    """Primitive and summary evidence for one condition and paired seed."""

    seed: int
    condition: IAConditionName
    rewards: NDArray[np.float64]
    executed_actions: NDArray[np.int64]
    credited_actions: NDArray[np.int64]
    recommendations: NDArray[np.int64]
    partner_proposals: NDArray[np.int64]
    accepted_recommendations: NDArray[np.bool_]
    mean_reward: float
    phase_mean_rewards: NDArray[np.float64]
    recovery_lengths: NDArray[np.int64]
    nominal_recommendation_decisions: int
    nominal_accepted_recommendations: int
    executed_accepted_recommendations: int
    action_changing_interventions: int
    changed_action_intervention_rate: float
    executed_action_credit_mismatches: int
    controller_budget: ControllerBudget
    timing: ConditionTiming


@dataclass(frozen=True)
class IAAggregateEvidence:
    """Paired held-out aggregate used by primary and secondary gates."""

    seeds: tuple[int, ...]
    mean_rewards: dict[IAConditionName, float]
    primary_uplift: float
    primary_uplift_interval: PairedBootstrapInterval
    accept_always_uplift: float
    accept_always_uplift_interval: PairedBootstrapInterval
    augmentation_vs_alone: float
    augmentation_vs_alone_interval: PairedBootstrapInterval
    augmentation_vs_noise: float
    augmentation_vs_noise_interval: PairedBootstrapInterval
    mean_changed_action_intervention_rate: float
    total_action_changing_interventions: int
    observe_only_exact_reward_identity: bool
    observe_only_exact_action_identity: bool
    executed_action_credit_mismatches: int
    primary_state_budget_matched: bool
    primary_interaction_budget_matched: bool
    augmentation_noise_state_budget_matched: bool
    augmentation_state_bytes_above_alone: int
    condition_budgets: dict[IAConditionName, ControllerBudget]
    condition_phase_mean_rewards: dict[IAConditionName, NDArray[np.float64]]
    condition_recovery_fraction: dict[IAConditionName, float]
    condition_mean_recovery_steps: dict[IAConditionName, float]
    all_values_finite: bool


@dataclass(frozen=True)
class IAAcceptanceEvidence:
    """One numeric, fail-closed acceptance comparison."""

    name: str
    scope: Literal["primary", "secondary"]
    passed: bool
    actual: float
    comparator: str
    threshold: float
    detail: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.scope not in ("primary", "secondary"):
            raise ValueError("scope must be 'primary' or 'secondary'")
        if type(self.passed) is not bool:
            raise ValueError("passed must be a boolean")
        object.__setattr__(self, "actual", finite_real("actual", self.actual))
        if type(self.comparator) is not str or not self.comparator:
            raise ValueError("comparator must be a non-empty string")
        object.__setattr__(self, "threshold", finite_real("threshold", self.threshold))
        if type(self.detail) is not str:
            raise ValueError("detail must be a string")


@dataclass(frozen=True)
class IAAcceptanceResult:
    """Primary and secondary decisions; no check may be skipped."""

    passed: bool
    primary_passed: bool
    secondary_passed: bool
    checks: tuple[IAAcceptanceEvidence, ...]

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise ValueError("passed must be a boolean")
        if type(self.primary_passed) is not bool:
            raise ValueError("primary_passed must be a boolean")
        if type(self.secondary_passed) is not bool:
            raise ValueError("secondary_passed must be a boolean")
        if not isinstance(self.checks, tuple) or not all(
            isinstance(c, IAAcceptanceEvidence) for c in self.checks
        ):
            raise ValueError("checks must be a tuple of IAAcceptanceEvidence")

    @property
    def failures(self) -> tuple[IAAcceptanceEvidence, ...]:
        return tuple(check for check in self.checks if not check.passed)


@dataclass(frozen=True)
class ContinualIAReport:
    """Complete paired report for one fixed seed schedule."""

    config: ContinualIAConfig
    thresholds: IAAcceptanceThresholds
    condition_results: tuple[IAConditionResult, ...]
    aggregate: IAAggregateEvidence
    acceptance: IAAcceptanceResult


def paired_bootstrap_mean_interval(
    paired_differences: NDArray[np.float64],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> PairedBootstrapInterval:
    """Resample already-paired within-seed differences."""

    values = np.asarray(paired_differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired_differences must be a non-empty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("paired_differences must be finite")
    confidence_level = validated_float32_scalar(
        "confidence_level",
        confidence_level,
        positive=True,
        upper=1.0,
        upper_inclusive=False,
    )
    resamples = _require_integer(
        "resamples", resamples, minimum=1, maximum=_MAX_BOOTSTRAP_RESAMPLES
    )
    seed = _require_integer("seed", seed, minimum=0, maximum=JAX_KEY_SEED_MAX)
    bootstrap_draw_count = resamples * int(values.size)
    if bootstrap_draw_count > _MAX_BOOTSTRAP_DRAW_COUNT:
        raise ValueError(
            "derived bootstrap_draw_count must not exceed "
            f"{_MAX_BOOTSTRAP_DRAW_COUNT}"
        )
    _require_derived_int32(
        "bootstrap_peak_bytes",
        16 * bootstrap_draw_count + 8 * resamples,
    )

    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        values.size,
        size=(resamples, values.size),
        dtype=np.int64,
    )
    means = np.mean(values[indices], axis=1)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(means, (tail, 1.0 - tail))
    return PairedBootstrapInterval(
        estimate=float(np.mean(values)),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        resamples=resamples,
        sample_size=int(values.size),
    )


def _make_partner(config: ContinualIAConfig) -> DifferentialSARSAAgent:
    return DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=config.n_actions,
            q_step_size=config.partner_q_step_size,
            average_reward_step_size=config.partner_average_reward_step_size,
            epsilon_start=config.partner_epsilon,
            epsilon_end=config.partner_epsilon,
        )
    )


def _make_ia(config: ContinualIAConfig) -> IAAgent:
    stomp = STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),),
        observation_dim=config.observation_dim,
        n_primitive_actions=config.n_actions,
        base_step_size=config.cortex_base_step_size,
        epsilon_base=0.0,
        epsilon_option=0.0,
    )
    cerebellum = ExoCerebellumConfig(
        n_demons=config.n_demons,
        obs_dim=config.observation_dim,
    )
    return IAAgent(
        IAConfig(
            cerebellum=cerebellum,
            cortex=OaKConfig(stomp=stomp),
        )
    )


def _make_env(config: ContinualIAConfig) -> SwitchingTwoStateMDP:
    environment_config = SwitchingTwoStateConfig(  # type: ignore[call-arg]
        phase_length=config.phase_length
    )
    return SwitchingTwoStateMDP(environment_config)


type AloneScanOutput = tuple[Array, Array]
type RecommendationScanOutput = tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    RecommendationProtocolState,
]
type AugmentedScanOutput = tuple[Array, Array, Array]


def _run_alone(
    env: SwitchingTwoStateMDP,
    partner: DifferentialSARSAAgent,
    key: Array,
    *,
    num_steps: int,
) -> AloneScanOutput:
    """Partner-alone loop with the same four-way key split as recommendation."""

    k_env, k_agent, _k_ia, k_steps = jr.split(key, 4)
    env_state = env.init(k_env)
    obs0 = env.observe(env_state)
    partner_state = partner.init(env.feature_dim, k_agent)
    partner_state, _ = partner.start(partner_state, obs0)

    def step(
        carry: tuple[SwitchingTwoStateState, DifferentialSARSAState],
        step_key: Array,
    ) -> tuple[
        tuple[SwitchingTwoStateState, DifferentialSARSAState],
        tuple[Array, Array],
    ]:
        env_state, partner_state = carry
        k_env_step, _k_accept = jr.split(step_key)
        action = partner_state.last_action
        next_obs, reward, env_state = env.step(
            env_state,
            action,
            k_env_step,
        )
        proposal, partner_key = partner.select_action(partner_state, next_obs)
        partner_state = partner_state.replace(  # type: ignore[attr-defined]
            rng_key=partner_key
        )
        update = partner.update(
            partner_state,
            reward,
            next_obs,
            next_action=proposal,
        )
        return (env_state, update.state), (reward, action)

    _, (rewards, actions) = jax.lax.scan(
        step,
        (env_state, partner_state),
        jr.split(k_steps, num_steps),
    )
    return rewards, actions


type RecommendationCarry = tuple[
    SwitchingTwoStateState,
    DifferentialSARSAState,
    IAState,
    RecommendationProtocolState,
    Array,
    Array,
    Array,
]


def _run_recommendation(
    env: SwitchingTwoStateMDP,
    partner: DifferentialSARSAAgent,
    ia: IAAgent,
    key: Array,
    *,
    num_steps: int,
    p_accept: float,
) -> RecommendationScanOutput:
    """Run one recommendation arm with primitive intervention provenance."""

    protocol_config = RecommendationProtocolConfig()
    k_env, k_agent, k_ia, k_steps = jr.split(key, 4)
    env_state = env.init(k_env)
    obs0 = env.observe(env_state)
    partner_state = partner.init(env.feature_dim, k_agent)
    partner_state, _ = partner.start(partner_state, obs0)
    ia_state = ia.start(ia.init(k_ia), obs0)
    protocol_state = init_recommendation_protocol_state()

    initial_action = partner_state.last_action
    carry: RecommendationCarry = (
        env_state,
        partner_state,
        ia_state,
        protocol_state,
        initial_action,
        initial_action,
        jnp.asarray(False),
    )

    def step(
        carry: RecommendationCarry,
        step_key: Array,
    ) -> tuple[RecommendationCarry, tuple[Array, ...]]:
        (
            env_state,
            partner_state,
            ia_state,
            protocol_state,
            prior_recommendation,
            prior_proposal,
            prior_accepted,
        ) = carry
        k_env_step, k_accept = jr.split(step_key)
        observation = partner_state.last_observation
        executed_action = partner_state.last_action
        credited_action = executed_action
        next_obs, reward, env_state = env.step(
            env_state,
            executed_action,
            k_env_step,
        )
        ia_update = ia.update(
            ia_state,
            observation,
            reward,
            next_obs,
            partner_action=credited_action,
        )
        proposal, partner_key = partner.select_action(partner_state, next_obs)
        partner_state = partner_state.replace(  # type: ignore[attr-defined]
            rng_key=partner_key
        )
        accept = jr.uniform(k_accept) < p_accept
        protocol = update_recommendation_protocol(
            protocol_config,
            protocol_state,
            ia_update.recommendation,
            proposal,
            accept,
        )
        partner_update = partner.update(
            partner_state,
            reward,
            next_obs,
            next_action=protocol.effective_action,
        )
        next_carry: RecommendationCarry = (
            env_state,
            partner_update.state,
            ia_update.state,
            protocol.state,
            ia_update.recommendation,
            proposal,
            protocol.accepted,
        )
        return next_carry, (
            reward,
            executed_action,
            credited_action,
            prior_recommendation,
            prior_proposal,
            prior_accepted,
        )

    final_carry, outputs = jax.lax.scan(
        step,
        carry,
        jr.split(k_steps, num_steps),
    )
    (
        rewards,
        executed_actions,
        credited_actions,
        recommendations,
        proposals,
        accepted,
    ) = outputs
    return (
        rewards,
        executed_actions,
        credited_actions,
        recommendations,
        proposals,
        accepted,
        final_carry[3],
    )


def _run_augmented(
    env: SwitchingTwoStateMDP,
    partner: DifferentialSARSAAgent,
    ia: IAAgent,
    key: Array,
    *,
    num_steps: int,
    noise_features: bool,
) -> AugmentedScanOutput:
    """Run prediction-feature or same-shape uniform-noise augmentation."""

    k_env, k_agent, k_ia, k_steps = jr.split(key, 4)
    env_state = env.init(k_env)
    obs0 = env.observe(env_state)
    partner_state = partner.init(env.feature_dim + ia.config.cerebellum.n_demons, k_agent)
    ia_state = ia.start(ia.init(k_ia), obs0)
    predictions0 = ia.cerebellum.predict(
        ia_state.cerebellum_state,
        obs0,
    )
    partner_state, _ = partner.start(
        partner_state,
        jnp.concatenate((obs0, predictions0)),
    )

    def step(
        carry: tuple[
            SwitchingTwoStateState,
            DifferentialSARSAState,
            IAState,
            Array,
        ],
        step_key: Array,
    ) -> tuple[
        tuple[
            SwitchingTwoStateState,
            DifferentialSARSAState,
            IAState,
            Array,
        ],
        tuple[Array, Array, Array],
    ]:
        env_state, partner_state, ia_state, raw_obs = carry
        k_env_step, k_noise = jr.split(step_key)
        executed_action = partner_state.last_action
        credited_action = executed_action
        next_obs, reward, env_state = env.step(
            env_state,
            executed_action,
            k_env_step,
        )
        ia_update = ia.update(
            ia_state,
            raw_obs,
            reward,
            next_obs,
            partner_action=credited_action,
        )
        features = ia.cerebellum.predict(
            ia_update.state.cerebellum_state,
            next_obs,
        )
        if noise_features:
            features = jr.uniform(k_noise, (ia.config.cerebellum.n_demons,))
        partner_update = partner.update(
            partner_state,
            reward,
            jnp.concatenate((next_obs, features)),
        )
        return (
            env_state,
            partner_update.state,
            ia_update.state,
            next_obs,
        ), (reward, executed_action, credited_action)

    _, outputs = jax.lax.scan(
        step,
        (env_state, partner_state, ia_state, obs0),
        jr.split(k_steps, num_steps),
    )
    return outputs


def _controller_budget(
    states: tuple[object, ...],
    *,
    observation_scalars: int,
    interaction_steps: int,
    ia_attached: bool,
) -> ControllerBudget:
    arrays: list[NDArray[np.generic]] = []
    for leaf in jax.tree_util.tree_leaves(states):
        # Typed JAX PRNG keys deliberately reject ``np.asarray``.  Their
        # physical state is the underlying uint32 key data, which is what a
        # persistent-memory budget must count.
        if isinstance(leaf, jax.Array) and jnp.issubdtype(
            leaf.dtype,
            jax.dtypes.prng_key,
        ):
            arrays.append(np.asarray(jr.key_data(leaf)))
        else:
            arrays.append(np.asarray(leaf))
    return ControllerBudget(
        state_scalars=sum(int(array.size) for array in arrays),
        state_bytes=sum(int(array.nbytes) for array in arrays),
        observation_scalars=observation_scalars,
        action_scalars_per_step=1,
        interaction_steps=interaction_steps,
        ia_attached=ia_attached,
    )


def _condition_budgets(
    config: ContinualIAConfig,
    partner: DifferentialSARSAAgent,
    ia: IAAgent,
) -> dict[IAConditionName, ControllerBudget]:
    key_agent, key_ia = jr.split(jr.key(2_026_073_012))
    alone_state = partner.init(config.observation_dim, key_agent)
    recommendation_partner = partner.init(config.observation_dim, key_agent)
    ia_state = ia.init(key_ia)
    protocol_state = init_recommendation_protocol_state()
    augmented_partner = partner.init(
        config.observation_dim + config.n_demons,
        key_agent,
    )
    alone = _controller_budget(
        (alone_state,),
        observation_scalars=config.observation_dim,
        interaction_steps=config.num_steps,
        ia_attached=False,
    )
    recommendation = _controller_budget(
        (recommendation_partner, ia_state, protocol_state),
        observation_scalars=config.observation_dim,
        interaction_steps=config.num_steps,
        ia_attached=True,
    )
    augmented = _controller_budget(
        (augmented_partner, ia_state),
        observation_scalars=config.observation_dim + config.n_demons,
        interaction_steps=config.num_steps,
        ia_attached=True,
    )
    return {
        "partner_alone": alone,
        "observe_only": recommendation,
        "recommendation_p05": recommendation,
        "accept_always": recommendation,
        "augmented_predictions": augmented,
        "augmented_noise": augmented,
    }


def condition_controller_budgets(
    config: ContinualIAConfig | None = None,
) -> dict[IAConditionName, ControllerBudget]:
    """Recompute canonical controller budgets from actual frozen state trees."""

    benchmark_config = ContinualIAConfig() if config is None else config
    return _condition_budgets(
        benchmark_config,
        _make_partner(benchmark_config),
        _make_ia(benchmark_config),
    )


def _phase_means(
    rewards: NDArray[np.float64],
    config: ContinualIAConfig,
) -> NDArray[np.float64]:
    return np.mean(
        rewards.reshape(config.n_phases, config.phase_length),
        axis=1,
    )


def _recovery_lengths(
    rewards: NDArray[np.float64],
    config: ContinualIAConfig,
) -> NDArray[np.int64]:
    """Steps through first post-switch window reaching the frozen mean target."""

    recoveries = np.full((config.n_phases - 1,), -1, dtype=np.int64)
    for phase_index in range(1, config.n_phases):
        start = phase_index * config.phase_length
        stop = start + config.phase_length
        for window_start in range(start, stop - config.recovery_window + 1):
            window = rewards[window_start : window_start + config.recovery_window]
            if np.all(np.isfinite(window)) and float(np.mean(window)) >= (
                config.recovery_mean_reward_threshold
            ):
                recoveries[phase_index - 1] = window_start + config.recovery_window - start
                break
    return recoveries


def _recommendation_result(
    *,
    seed: int,
    condition: IAConditionName,
    output: RecommendationScanOutput,
    budget: ControllerBudget,
    wall_seconds: float,
    config: ContinualIAConfig,
) -> IAConditionResult:
    (
        rewards_raw,
        actions_raw,
        credits_raw,
        recommendations_raw,
        proposals_raw,
        accepted_raw,
        protocol_state,
    ) = output
    rewards = np.asarray(rewards_raw, dtype=np.float64)
    actions = np.asarray(actions_raw, dtype=np.int64)
    credits = np.asarray(credits_raw, dtype=np.int64)
    recommendations = np.asarray(recommendations_raw, dtype=np.int64)
    proposals = np.asarray(proposals_raw, dtype=np.int64)
    accepted = np.asarray(accepted_raw, dtype=np.bool_)
    changed = accepted & (recommendations != proposals)
    mismatches = int(np.count_nonzero(actions != credits))
    return IAConditionResult(
        seed=seed,
        condition=condition,
        rewards=rewards,
        executed_actions=actions,
        credited_actions=credits,
        recommendations=recommendations,
        partner_proposals=proposals,
        accepted_recommendations=accepted,
        mean_reward=float(np.mean(rewards)),
        phase_mean_rewards=_phase_means(rewards, config),
        recovery_lengths=_recovery_lengths(rewards, config),
        nominal_recommendation_decisions=int(protocol_state.step_count),
        nominal_accepted_recommendations=int(protocol_state.accepted_count),
        executed_accepted_recommendations=int(np.count_nonzero(accepted)),
        action_changing_interventions=int(np.count_nonzero(changed)),
        changed_action_intervention_rate=float(np.mean(changed)),
        executed_action_credit_mismatches=mismatches,
        controller_budget=budget,
        timing=ConditionTiming(
            wall_seconds=wall_seconds,
            mean_step_latency_ms=1_000.0 * wall_seconds / config.num_steps,
        ),
    )


def _plain_result(
    *,
    seed: int,
    condition: IAConditionName,
    rewards_raw: Array,
    actions_raw: Array,
    credits_raw: Array,
    budget: ControllerBudget,
    wall_seconds: float,
    config: ContinualIAConfig,
) -> IAConditionResult:
    rewards = np.asarray(rewards_raw, dtype=np.float64)
    actions = np.asarray(actions_raw, dtype=np.int64)
    credits = np.asarray(credits_raw, dtype=np.int64)
    empty_actions = np.full(actions.shape, -1, dtype=np.int64)
    empty_acceptance = np.zeros(actions.shape, dtype=np.bool_)
    return IAConditionResult(
        seed=seed,
        condition=condition,
        rewards=rewards,
        executed_actions=actions,
        credited_actions=credits,
        recommendations=empty_actions,
        partner_proposals=empty_actions,
        accepted_recommendations=empty_acceptance,
        mean_reward=float(np.mean(rewards)),
        phase_mean_rewards=_phase_means(rewards, config),
        recovery_lengths=_recovery_lengths(rewards, config),
        nominal_recommendation_decisions=0,
        nominal_accepted_recommendations=0,
        executed_accepted_recommendations=0,
        action_changing_interventions=0,
        changed_action_intervention_rate=0.0,
        executed_action_credit_mismatches=int(np.count_nonzero(actions != credits)),
        controller_budget=budget,
        timing=ConditionTiming(
            wall_seconds=wall_seconds,
            mean_step_latency_ms=1_000.0 * wall_seconds / config.num_steps,
        ),
    )


def _timed_call[T](function: Callable[[Array], T], key: Array) -> tuple[T, float]:
    start = perf_counter()
    output = function(key)
    jax.block_until_ready(output)  # type: ignore[no-untyped-call]
    return output, perf_counter() - start


def _condition_group(
    results: Sequence[IAConditionResult],
    condition: IAConditionName,
) -> tuple[IAConditionResult, ...]:
    selected = tuple(result for result in results if result.condition == condition)
    if not selected:
        raise ValueError(f"missing condition: {condition}")
    return selected


def aggregate_ia_evidence(
    results: Sequence[IAConditionResult],
    *,
    config: ContinualIAConfig,
) -> IAAggregateEvidence:
    """Aggregate paired conditions without dropping seeds or failed recoveries."""

    if not results:
        raise ValueError("results must be non-empty")
    groups = {condition: _condition_group(results, condition) for condition in CONDITION_NAMES}
    seed_sets = {
        condition: tuple(result.seed for result in group) for condition, group in groups.items()
    }
    first_seeds = seed_sets["partner_alone"]
    if any(seeds != first_seeds for seeds in seed_sets.values()):
        raise ValueError("all conditions must contain identical ordered seeds")
    if len(set(first_seeds)) != len(first_seeds):
        raise ValueError("paired seeds must be unique")
    for condition, group in groups.items():
        first_budget = group[0].controller_budget
        if any(result.controller_budget != first_budget for result in group[1:]):
            raise ValueError(f"controller budget changes across seeds for {condition}")

    rewards = {
        condition: np.asarray(
            [result.mean_reward for result in group],
            dtype=np.float64,
        )
        for condition, group in groups.items()
    }
    primary_differences = rewards["recommendation_p05"] - rewards["observe_only"]
    accept_always_differences = rewards["accept_always"] - rewards["observe_only"]
    augmentation_alone_differences = rewards["augmented_predictions"] - rewards["partner_alone"]
    augmentation_noise_differences = rewards["augmented_predictions"] - rewards["augmented_noise"]
    intervals = (
        paired_bootstrap_mean_interval(
            primary_differences,
            confidence_level=config.confidence_level,
            resamples=config.bootstrap_resamples,
            seed=config.bootstrap_seed,
        ),
        paired_bootstrap_mean_interval(
            accept_always_differences,
            confidence_level=config.confidence_level,
            resamples=config.bootstrap_resamples,
            seed=(config.bootstrap_seed + 1) % 2**32,
        ),
        paired_bootstrap_mean_interval(
            augmentation_alone_differences,
            confidence_level=config.confidence_level,
            resamples=config.bootstrap_resamples,
            seed=(config.bootstrap_seed + 2) % 2**32,
        ),
        paired_bootstrap_mean_interval(
            augmentation_noise_differences,
            confidence_level=config.confidence_level,
            resamples=config.bootstrap_resamples,
            seed=(config.bootstrap_seed + 3) % 2**32,
        ),
    )

    alone = groups["partner_alone"]
    observe = groups["observe_only"]
    treatment = groups["recommendation_p05"]
    observe_rewards_equal = all(
        np.array_equal(left.rewards, right.rewards)
        for left, right in zip(alone, observe, strict=True)
    )
    observe_actions_equal = all(
        np.array_equal(left.executed_actions, right.executed_actions)
        for left, right in zip(alone, observe, strict=True)
    )
    budgets = {condition: group[0].controller_budget for condition, group in groups.items()}
    primary_budget_matched = budgets["recommendation_p05"] == budgets["observe_only"]
    primary_interaction_matched = (
        budgets["recommendation_p05"].interaction_steps == budgets["observe_only"].interaction_steps
        and budgets["recommendation_p05"].action_scalars_per_step
        == budgets["observe_only"].action_scalars_per_step
    )
    augmentation_budget_matched = budgets["augmented_predictions"] == budgets["augmented_noise"]

    phase_rewards = {
        condition: np.mean(
            np.stack([result.phase_mean_rewards for result in group]),
            axis=0,
        )
        for condition, group in groups.items()
    }
    recovery_fraction: dict[IAConditionName, float] = {}
    mean_recovery: dict[IAConditionName, float] = {}
    for condition, group in groups.items():
        values = np.concatenate([result.recovery_lengths for result in group])
        recovered = values >= 0
        recovery_fraction[condition] = float(np.mean(recovered))
        mean_recovery[condition] = (
            float(np.mean(values[recovered])) if np.any(recovered) else float("inf")
        )

    numeric_arrays = [
        array
        for group in groups.values()
        for result in group
        for array in (
            result.rewards,
            result.phase_mean_rewards,
        )
    ]
    all_values_finite = all(bool(np.all(np.isfinite(array))) for array in numeric_arrays)
    changed_rates = np.asarray(
        [result.changed_action_intervention_rate for result in treatment],
        dtype=np.float64,
    )
    return IAAggregateEvidence(
        seeds=first_seeds,
        mean_rewards={condition: float(np.mean(values)) for condition, values in rewards.items()},
        primary_uplift=intervals[0].estimate,
        primary_uplift_interval=intervals[0],
        accept_always_uplift=intervals[1].estimate,
        accept_always_uplift_interval=intervals[1],
        augmentation_vs_alone=intervals[2].estimate,
        augmentation_vs_alone_interval=intervals[2],
        augmentation_vs_noise=intervals[3].estimate,
        augmentation_vs_noise_interval=intervals[3],
        mean_changed_action_intervention_rate=float(np.mean(changed_rates)),
        total_action_changing_interventions=sum(
            result.action_changing_interventions for result in treatment
        ),
        observe_only_exact_reward_identity=observe_rewards_equal,
        observe_only_exact_action_identity=observe_actions_equal,
        executed_action_credit_mismatches=sum(
            result.executed_action_credit_mismatches
            for condition in (
                "observe_only",
                "recommendation_p05",
                "accept_always",
                "augmented_predictions",
                "augmented_noise",
            )
            for result in groups[condition]
        ),
        primary_state_budget_matched=primary_budget_matched,
        primary_interaction_budget_matched=primary_interaction_matched,
        augmentation_noise_state_budget_matched=augmentation_budget_matched,
        augmentation_state_bytes_above_alone=(
            budgets["augmented_predictions"].state_bytes - budgets["partner_alone"].state_bytes
        ),
        condition_budgets=budgets,
        condition_phase_mean_rewards=phase_rewards,
        condition_recovery_fraction=recovery_fraction,
        condition_mean_recovery_steps=mean_recovery,
        all_values_finite=all_values_finite,
    )


def _minimum_check(
    name: str,
    scope: Literal["primary", "secondary"],
    actual: float,
    threshold: float,
    detail: str,
) -> IAAcceptanceEvidence:
    return IAAcceptanceEvidence(
        name=name,
        scope=scope,
        passed=bool(np.isfinite(actual) and actual >= threshold),
        actual=actual,
        comparator=">=",
        threshold=threshold,
        detail=detail,
    )


def _maximum_check(
    name: str,
    scope: Literal["primary", "secondary"],
    actual: float,
    threshold: float,
    detail: str,
) -> IAAcceptanceEvidence:
    return IAAcceptanceEvidence(
        name=name,
        scope=scope,
        passed=bool(np.isfinite(actual) and actual <= threshold),
        actual=actual,
        comparator="<=",
        threshold=threshold,
        detail=detail,
    )


def evaluate_ia_acceptance(
    aggregate: IAAggregateEvidence,
    thresholds: IAAcceptanceThresholds | None = None,
) -> IAAcceptanceResult:
    """Evaluate every preregistered primary and coherent secondary check."""

    limits = IAAcceptanceThresholds() if thresholds is None else thresholds
    seed_schedule_matches = len(aggregate.seeds) == limits.minimum_seed_count and all(
        seed == limits.evidence_seed_start + offset
        for offset, seed in enumerate(aggregate.seeds)
    )
    identity = (
        aggregate.observe_only_exact_reward_identity
        and aggregate.observe_only_exact_action_identity
    )
    checks = (
        _minimum_check(
            "seed_count",
            "primary",
            float(len(aggregate.seeds)),
            float(limits.minimum_seed_count),
            "At least thirty unique paired held-out seeds are required.",
        ),
        _minimum_check(
            "evidence_seed_schedule",
            "primary",
            float(seed_schedule_matches),
            1.0,
            "Promoted evidence must use exactly frozen seeds 30-59.",
        ),
        _minimum_check(
            "all_values_finite",
            "primary",
            float(aggregate.all_values_finite),
            1.0,
            "Every scientific reward, effect, and intervention value must be finite.",
        ),
        _minimum_check(
            "observe_only_exact_identity",
            "primary",
            float(identity),
            float(limits.require_observe_only_exact_identity),
            "Observe-only reward and executed-action traces must equal partner-alone bitwise.",
        ),
        _minimum_check(
            "primary_state_budget_matched",
            "primary",
            float(aggregate.primary_state_budget_matched),
            1.0,
            "p=.5 treatment and p=0 control must have identical controller state budgets.",
        ),
        _minimum_check(
            "primary_interaction_budget_matched",
            "primary",
            float(aggregate.primary_interaction_budget_matched),
            1.0,
            "p=.5 treatment and p=0 control must have identical interaction budgets.",
        ),
        _maximum_check(
            "executed_action_credit_mismatches",
            "primary",
            float(aggregate.executed_action_credit_mismatches),
            float(limits.maximum_executed_action_credit_mismatches),
            "IA updates must be credited on the primitive action actually executed.",
        ),
        _minimum_check(
            "primary_recommendation_uplift",
            "primary",
            aggregate.primary_uplift_interval.lower,
            limits.minimum_primary_uplift_lower_ci,
            "Lower paired 95% CI for p=.5 minus observe-only mean reward.",
        ),
        _minimum_check(
            "changed_action_intervention_rate",
            "primary",
            aggregate.mean_changed_action_intervention_rate,
            limits.minimum_changed_action_intervention_rate,
            "Mean primitive-step rate of accepted, action-changing recommendations.",
        ),
        _minimum_check(
            "augmentation_noise_state_budget_matched",
            "secondary",
            float(aggregate.augmentation_noise_state_budget_matched),
            1.0,
            "Prediction and noise augmentation controls must have equal state budgets.",
        ),
        _minimum_check(
            "augmentation_state_budget_above_alone",
            "secondary",
            float(aggregate.augmentation_state_bytes_above_alone),
            1.0,
            "Both equal-budget augmentation arms must exceed partner-alone state bytes.",
        ),
        _minimum_check(
            "augmentation_vs_alone",
            "secondary",
            aggregate.augmentation_vs_alone_interval.lower,
            limits.minimum_augmentation_vs_alone_lower_ci,
            "Lower paired 95% CI for prediction augmentation minus partner-alone.",
        ),
        _minimum_check(
            "augmentation_vs_noise",
            "secondary",
            aggregate.augmentation_vs_noise_interval.lower,
            limits.minimum_augmentation_vs_noise_lower_ci,
            "Lower paired 95% CI for predictions minus same-shape uniform noise.",
        ),
    )
    primary = tuple(check for check in checks if check.scope == "primary")
    secondary = tuple(check for check in checks if check.scope == "secondary")
    primary_passed = all(check.passed for check in primary)
    secondary_passed = all(check.passed for check in secondary)
    return IAAcceptanceResult(
        passed=primary_passed and secondary_passed,
        primary_passed=primary_passed,
        secondary_passed=secondary_passed,
        checks=checks,
    )


def run_continual_ia_benchmark(
    *,
    seeds: Sequence[int] = PROMOTED_EVIDENCE_SEEDS,
    config: ContinualIAConfig | None = None,
    thresholds: IAAcceptanceThresholds | None = None,
) -> ContinualIAReport:
    """Run all paired conditions once per seed and return fail-closed evidence."""

    benchmark_config = ContinualIAConfig() if config is None else config
    limits = IAAcceptanceThresholds() if thresholds is None else thresholds
    if type(seeds) not in (tuple, list):
        raise ValueError("seeds must be an actual tuple or list")
    seed_count = len(seeds)
    if seed_count < 1:
        raise ValueError("seeds must be non-empty")
    _preflight_run_resources(benchmark_config, seed_count)
    seed_tuple = tuple(
        _require_integer(
            f"seeds[{index}]", seed, minimum=0, maximum=JAX_KEY_SEED_MAX
        )
        for index, seed in enumerate(seeds)
    )
    if len(set(seed_tuple)) != len(seed_tuple):
        raise ValueError("seeds must be unique")

    env = _make_env(benchmark_config)
    partner = _make_partner(benchmark_config)
    ia = _make_ia(benchmark_config)
    budgets = _condition_budgets(benchmark_config, partner, ia)

    alone_fn = jax.jit(
        functools.partial(
            _run_alone,
            env,
            partner,
            num_steps=benchmark_config.num_steps,
        )
    )
    observe_fn = jax.jit(
        functools.partial(
            _run_recommendation,
            env,
            partner,
            ia,
            num_steps=benchmark_config.num_steps,
            p_accept=0.0,
        )
    )
    treatment_fn = jax.jit(
        functools.partial(
            _run_recommendation,
            env,
            partner,
            ia,
            num_steps=benchmark_config.num_steps,
            p_accept=benchmark_config.recommendation_acceptance_probability,
        )
    )
    accept_always_fn = jax.jit(
        functools.partial(
            _run_recommendation,
            env,
            partner,
            ia,
            num_steps=benchmark_config.num_steps,
            p_accept=1.0,
        )
    )
    augmented_fn = jax.jit(
        functools.partial(
            _run_augmented,
            env,
            partner,
            ia,
            num_steps=benchmark_config.num_steps,
            noise_features=False,
        )
    )
    noise_fn = jax.jit(
        functools.partial(
            _run_augmented,
            env,
            partner,
            ia,
            num_steps=benchmark_config.num_steps,
            noise_features=True,
        )
    )
    warm_key = jr.key(2_147_483_000)
    for function in (
        alone_fn,
        observe_fn,
        treatment_fn,
        accept_always_fn,
        augmented_fn,
        noise_fn,
    ):
        jax.block_until_ready(function(warm_key))  # type: ignore[no-untyped-call]

    results: list[IAConditionResult] = []
    for seed in seed_tuple:
        key = jr.key(seed)
        alone_output, alone_wall = _timed_call(alone_fn, key)
        alone_rewards, alone_actions = alone_output
        results.append(
            _plain_result(
                seed=seed,
                condition="partner_alone",
                rewards_raw=alone_rewards,
                actions_raw=alone_actions,
                credits_raw=alone_actions,
                budget=budgets["partner_alone"],
                wall_seconds=alone_wall,
                config=benchmark_config,
            )
        )
        for condition, function in (
            ("observe_only", observe_fn),
            ("recommendation_p05", treatment_fn),
            ("accept_always", accept_always_fn),
        ):
            condition = cast(IAConditionName, condition)
            output, wall_seconds = _timed_call(function, key)
            results.append(
                _recommendation_result(
                    seed=seed,
                    condition=condition,
                    output=output,
                    budget=budgets[condition],
                    wall_seconds=wall_seconds,
                    config=benchmark_config,
                )
            )
        for condition, function in (
            ("augmented_predictions", augmented_fn),
            ("augmented_noise", noise_fn),
        ):
            condition = cast(IAConditionName, condition)
            output, wall_seconds = _timed_call(function, key)
            rewards_raw, actions_raw, credits_raw = output
            results.append(
                _plain_result(
                    seed=seed,
                    condition=condition,
                    rewards_raw=rewards_raw,
                    actions_raw=actions_raw,
                    credits_raw=credits_raw,
                    budget=budgets[condition],
                    wall_seconds=wall_seconds,
                    config=benchmark_config,
                )
            )

    aggregate = aggregate_ia_evidence(results, config=benchmark_config)
    acceptance = evaluate_ia_acceptance(aggregate, limits)
    return ContinualIAReport(
        config=benchmark_config,
        thresholds=limits,
        condition_results=tuple(results),
        aggregate=aggregate,
        acceptance=acceptance,
    )
