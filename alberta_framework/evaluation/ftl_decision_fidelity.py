"""Narrow multi-step decision-fidelity evaluation for a sparse world model.

This module evaluates whether a learned transition model is useful for choosing
among a fixed menu of open-loop action sequences.  It is inspired by the
decision-fidelity protocol in WorldModelGym and by the known-reward planning
setup in Liu et al. (2025), *Continual Reinforcement Learning by Planning with
Online World Models*:

* true menu returns are precomputed with a separate deterministic environment;
* every imagined sequence branches from the same immutable model state;
* the planner applies the known reward function to imagined states and uses
  only an argmax over predicted returns;
* menu-normalized regret is
  ``(best_true_return - picked_true_return) / (best - worst)``;
* per-step reward and cumulative-return errors expose lucky correct choices.

The five action sequences have the same first action and diverge later.
Consequently, the evaluation requires a multi-step rollout rather than a
one-step action comparison.

The training stream visits a negative state domain (A), a positive domain (B),
and then A again without providing a phase label to either learner.  A raw
online ridge model is included as a small, fixed-memory baseline.  It receives
the same transitions and ridge coefficient without replay, but is not matched
to the sparse model's feature count or memory budget.  The evaluation is
deliberately synthetic and deterministic: it is not evidence of general
control, exact global FTL, equivalence to a learned visual world model, or
completion of the Alberta Plan.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from numpy.typing import NDArray

from alberta_framework.core._float32_scalars import (
    validated_float32_scalar,
    validated_float32_scalar_with_ratio,
)
from alberta_framework.core.ftl_world_model import (
    SparseFTLWorldModel,
    SparseFTLWorldModelConfig,
    SparseFTLWorldModelState,
    run_sparse_ftl_world_model,
)

CONDITION_NAMES = (
    "sparse_untrained",
    "sparse_after_a1",
    "sparse_after_b",
    "sparse_after_a2",
    "linear_after_a1",
    "linear_after_b",
    "linear_after_a2",
)

# The configuration and assertion margins were developed while inspecting
# seeds 0..29.  Keep that schedule explicit and disjoint from the frozen
# promoted-evidence default.  Do not tune on EVIDENCE_SEEDS after inspecting
# their outcomes.
DEVELOPMENT_SEEDS = tuple(range(30))
EVIDENCE_SEEDS = tuple(range(30, 60))
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_BOOTSTRAP_MAX_SEED_OFFSET = 103
_FLOAT32_MAX_BELOW_OVERFLOW = float(
    np.nextafter(np.float32(np.finfo(np.float32).max), np.float32(0.0))
)
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


def _require_uint32(name: str, value: object) -> int:
    return _require_int32(name, value, minimum=0, maximum=_UINT32_MAX)


def _require_derived_int32(name: str, value: int) -> int:
    if value > _INT32_MAX:
        raise ValueError(f"derived {name} must be at most {_INT32_MAX}")
    return value


def _validate_bootstrap_capacity(config: DecisionFidelityConfig, sample_count: int) -> None:
    """Reject the exact bootstrap working set before NumPy allocates it."""

    draws = _require_derived_int32(
        "bootstrap draw count", config.bootstrap_resamples * sample_count
    )
    # NumPy holds the int64 index matrix and indexed float64 samples together,
    # then materializes one float64 mean per row.
    _require_derived_int32(
        "bootstrap working bytes", 16 * draws + 8 * config.bootstrap_resamples
    )


@dataclass(frozen=True)
class DecisionFidelityConfig:
    """Scientific configuration for the deterministic decision probe.

    Args:
        phase_steps: Online transitions in each visitation phase (A1, B, A2).
        horizon: Length of every open-loop menu sequence; imagined rollouts
            are scored over exactly this many steps.
        probes_per_domain: Held-out decision probes drawn per domain; the
            probe set holds twice this many decisions.
        menu_amplitudes: Peak amplitude of each half-sine action profile, one
            menu entry per amplitude.  Every profile starts at zero, so the
            menus share their first action and diverge only later.
        projection_dim: Random projections in the sparse model's feature map
            (forwarded to :class:`SparseFTLWorldModelConfig`).
        bins: Soft bins per projection in the sparse feature map.
        ridge: Ridge coefficient shared by the sparse model and the raw
            linear baseline.
        prediction_clip: Absolute bound on each model's predicted state delta
            during imagined rollouts.
        state_bound: The true environment clips states to this magnitude;
            probe goals stay within ``0.89 * state_bound``.
        action_cost: Quadratic control penalty in the known reward
            ``-(s' - g)^2 - action_cost * u^2``.
        bootstrap_resamples: Frozen resample count for percentile intervals.
        confidence_level: Two-sided confidence level of those intervals.
        bootstrap_seed: Base seed of the bootstrap RNG, offset once per
            condition-metric pair.
    """

    phase_steps: int = 180
    horizon: int = 6
    probes_per_domain: int = 12
    menu_amplitudes: tuple[float, ...] = (-0.85, -0.45, 0.0, 0.45, 0.85)
    projection_dim: int = 12
    bins: int = 7
    ridge: float = 0.01
    prediction_clip: float = 3.0
    state_bound: float = 1.75
    action_cost: float = 0.025
    bootstrap_resamples: int = 5_000
    confidence_level: float = 0.95
    bootstrap_seed: int = 2_026_073_001

    def __post_init__(self) -> None:
        phase_steps = _require_int32("phase_steps", self.phase_steps, minimum=16)
        horizon = _require_int32("horizon", self.horizon, minimum=3)
        probes_per_domain = _require_int32("probes_per_domain", self.probes_per_domain, minimum=2)
        projection_dim = _require_int32("projection_dim", self.projection_dim, minimum=1)
        bins = _require_int32("bins", self.bins, minimum=2)
        bootstrap_resamples = _require_int32(
            "bootstrap_resamples", self.bootstrap_resamples, minimum=1000
        )
        bootstrap_seed = _require_uint32("bootstrap_seed", self.bootstrap_seed)

        object.__setattr__(self, "phase_steps", phase_steps)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "probes_per_domain", probes_per_domain)
        object.__setattr__(self, "projection_dim", projection_dim)
        object.__setattr__(self, "bins", bins)
        object.__setattr__(self, "bootstrap_resamples", bootstrap_resamples)
        object.__setattr__(self, "bootstrap_seed", bootstrap_seed)

        if type(self.menu_amplitudes) is not tuple:
            raise ValueError("menu_amplitudes must be an exact tuple")
        menu_count = len(self.menu_amplitudes)
        if menu_count < 3:
            raise ValueError("menu_amplitudes must contain at least three choices")
        probe_count = _require_derived_int32("probe count", 2 * probes_per_domain)
        _require_derived_int32("phase schedule length", 3 * phase_steps)
        _require_derived_int32("phase array bytes", 12 * phase_steps)
        _require_derived_int32("action menu scalars", menu_count * horizon)
        rollout_scalars = _require_derived_int32(
            "probe rollout work", probe_count * menu_count * horizon
        )
        _require_derived_int32(
            "probe and rollout array bytes",
            16 * rollout_scalars
            + 8 * probe_count * menu_count
            + 8 * menu_count * horizon
            + 16 * probe_count,
        )
        menu_values: list[float] = []
        menu_ratios: list[tuple[int, int]] = []
        for index, value in enumerate(self.menu_amplitudes):
            stored, numerator, denominator = validated_float32_scalar_with_ratio(
                f"menu_amplitudes[{index}]", value
            )
            menu_values.append(stored)
            menu_ratios.append((numerator, denominator))
        if len(set(menu_ratios)) != len(menu_ratios) or len(set(menu_values)) != len(menu_values):
            raise ValueError("menu_amplitudes must be unique")
        menu_amplitudes = tuple(menu_values)
        ridge = validated_float32_scalar("ridge", self.ridge, positive=True)
        prediction_clip = validated_float32_scalar(
            "prediction_clip", self.prediction_clip, positive=True
        )
        state_bound = validated_float32_scalar("state_bound", self.state_bound, positive=True)
        action_cost = validated_float32_scalar("action_cost", self.action_cost, lower=0.0)
        confidence_level = validated_float32_scalar(
            "confidence_level",
            self.confidence_level,
            positive=True,
            upper=1.0,
            upper_inclusive=False,
        )
        object.__setattr__(self, "menu_amplitudes", menu_amplitudes)
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "prediction_clip", prediction_clip)
        object.__setattr__(self, "state_bound", state_bound)
        object.__setattr__(self, "action_cost", action_cost)
        object.__setattr__(self, "confidence_level", confidence_level)

        sparse_config = SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=projection_dim,
            bins=bins,
            ridge=ridge,
            statistics_decay=1.0,
            prediction_clip=prediction_clip,
        )
        _require_derived_int32("SparseFTL feature dimension", sparse_config.feature_dim)
        _require_derived_int32(
            "SparseFTL persistent state bytes", SparseFTLWorldModel(sparse_config).state_nbytes
        )
        reward_term_limit = math.sqrt(
            _FLOAT32_MAX_BELOW_OVERFLOW / (2.0 * horizon)
        )
        # Probe initial states are bounded by 1.30 independently of state_bound;
        # goals are clipped to 0.89 * state_bound. Slightly wider decimal
        # coefficients conservatively cover float32 narrowing at both sinks.
        maximum_predicted_distance = 1.31 + 0.90 * state_bound + horizon * prediction_clip
        if (
            not math.isfinite(maximum_predicted_distance)
            or maximum_predicted_distance > reward_term_limit
        ):
            raise ValueError(
                "state_bound and prediction_clip exceed the float32 return domain"
            )
        maximum_amplitude = max(abs(value) for value in menu_amplitudes)
        if action_cost > 0.0 and maximum_amplitude > reward_term_limit / math.sqrt(action_cost):
            raise ValueError("menu_amplitudes and action_cost exceed the float32 return domain")
        if bootstrap_seed + _BOOTSTRAP_MAX_SEED_OFFSET > _UINT32_MAX:
            raise ValueError(
                "bootstrap_seed plus the frozen maximum offset must lie in [0, 2**32)"
            )
        _validate_bootstrap_capacity(self, len(EVIDENCE_SEEDS))


@dataclass(frozen=True)
class DecisionProbeSet:
    """Held-out decisions with separately precomputed true trajectories."""

    seed: int
    initial_observations: NDArray[np.float32]
    goals: NDArray[np.float32]
    domain_indices: NDArray[np.int64]
    action_sequences: NDArray[np.float32]
    true_next_observations: NDArray[np.float64]
    true_rewards: NDArray[np.float64]
    true_returns: NDArray[np.float64]


@dataclass(frozen=True)
class PredictedRollouts:
    """Imagined trajectories and known rewards evaluated on those trajectories."""

    next_observations: NDArray[np.float64]
    rewards: NDArray[np.float64]
    returns: NDArray[np.float64]


@dataclass(frozen=True)
class DecisionMetrics:
    """Decision and prediction metrics for one seed and model snapshot."""

    condition: str
    normalized_regret: float
    domain_a_normalized_regret: float
    domain_b_normalized_regret: float
    oracle_pick_rate: float
    reward_mae: float
    return_mae: float
    normalized_return_mae: float


@dataclass(frozen=True)
class SeedDecisionResult:
    """All paired model conditions for one probe seed.

    Whether that seed is development, promoted evidence, or a custom
    diagnostic schedule is determined by the caller.
    """

    seed: int
    metrics: tuple[DecisionMetrics, ...]


@dataclass(frozen=True)
class BootstrapEstimate:
    """Deterministic percentile-bootstrap estimate over paired seeds."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int
    sample_size: int


@dataclass(frozen=True)
class ConditionAggregate:
    """Bootstrap summaries for one model snapshot."""

    condition: str
    normalized_regret: BootstrapEstimate
    domain_a_normalized_regret: BootstrapEstimate
    domain_b_normalized_regret: BootstrapEstimate
    oracle_pick_rate: BootstrapEstimate
    reward_mae: BootstrapEstimate
    return_mae: BootstrapEstimate
    normalized_return_mae: BootstrapEstimate


@dataclass(frozen=True)
class PairedComparison:
    """A named paired difference whose sign is defined explicitly."""

    name: str
    definition: str
    estimate: BootstrapEstimate


@dataclass(frozen=True)
class DecisionFidelityReport:
    """Complete paired multi-seed decision-fidelity evidence."""

    config: DecisionFidelityConfig
    seeds: tuple[int, ...]
    seed_results: tuple[SeedDecisionResult, ...]
    aggregates: tuple[ConditionAggregate, ...]
    comparisons: tuple[PairedComparison, ...]

    def aggregate(self, condition: str) -> ConditionAggregate:
        """Return one named condition or fail rather than silently substituting."""

        for aggregate in self.aggregates:
            if aggregate.condition == condition:
                return aggregate
        raise KeyError(condition)

    def comparison(self, name: str) -> PairedComparison:
        """Return one named paired comparison."""

        for comparison in self.comparisons:
            if comparison.name == name:
                return comparison
        raise KeyError(name)


class _LinearRidgeState(NamedTuple):
    """Fixed-memory raw-feature online ridge baseline."""

    gram: Array
    cross: Array
    weights: Array
    step_count: Array


def _true_next_observation(
    observation: NDArray[np.float64],
    action: NDArray[np.float64],
    config: DecisionFidelityConfig,
) -> NDArray[np.float64]:
    """Deterministic nonlinear one-dimensional evaluation environment.

    The state delta mixes a linear control gain with ``sin`` and cubic state
    terms that lie outside the raw-linear baseline's feature span
    ``(1, s, u, s*u)``, so that baseline is deliberately misspecified.  Both
    nonlinear terms are odd in the state, so local dynamics differ between
    the negative (A) and positive (B) probe domains.  The next state is
    clipped to ``±state_bound``.
    """

    state = float(observation[0])
    control = float(action[0])
    delta = 0.46 * control + 0.18 * np.sin(2.4 * state) + 0.12 * state * control - 0.055 * state**3
    return np.asarray(
        (np.clip(state + delta, -config.state_bound, config.state_bound),),
        dtype=np.float64,
    )


def _known_reward(
    next_observation: NDArray[np.float64],
    goal: NDArray[np.float64],
    action: NDArray[np.float64],
    config: DecisionFidelityConfig,
) -> float:
    distance = float(next_observation[0] - goal[0])
    control = float(action[0])
    return -(distance**2) - config.action_cost * control**2


def _known_reward_jax(
    next_observation: Array,
    goal: Array,
    action: Array,
    config: DecisionFidelityConfig,
) -> Array:
    distance = next_observation[0] - goal[0]
    control = action[0]
    return -(distance**2) - config.action_cost * control**2


def fixed_action_sequence_menu(config: DecisionFidelityConfig) -> NDArray[np.float32]:
    """Create fixed open-loop choices with an identical first action."""

    time = np.linspace(0.0, 1.0, config.horizon, dtype=np.float64)
    profile = np.sin(np.pi * time)
    sequences = np.stack(
        [amplitude * profile for amplitude in config.menu_amplitudes],
        axis=0,
    )[..., None]
    return sequences.astype(np.float32)


def _probe_inputs(
    config: DecisionFidelityConfig,
    seed: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.int64],
    NDArray[np.float32],
]:
    generator = np.random.default_rng(50_000 + seed)
    count = config.probes_per_domain
    domain_a = generator.uniform(-1.30, -0.22, size=count)
    domain_b = generator.uniform(0.22, 1.30, size=count)
    initial = np.concatenate((domain_a, domain_b)).astype(np.float32)[:, None]
    offsets = generator.choice(
        np.asarray((-0.95, -0.65, 0.65, 0.95), dtype=np.float32),
        size=2 * count,
    )
    goals = np.clip(
        initial[:, 0] + offsets,
        -0.89 * config.state_bound,
        0.89 * config.state_bound,
    ).astype(np.float32)[:, None]
    domains = np.concatenate(
        (
            np.zeros((count,), dtype=np.int64),
            np.ones((count,), dtype=np.int64),
        )
    )
    return initial, goals, domains, fixed_action_sequence_menu(config)


def precompute_decision_probes(
    config: DecisionFidelityConfig,
    seed: int,
) -> DecisionProbeSet:
    """Precompute true outcomes without consulting a learned model."""

    seed = _require_int32("seed", seed, minimum=0)
    initial, goals, domains, action_sequences = _probe_inputs(config, seed)
    probe_count = initial.shape[0]
    menu_size = action_sequences.shape[0]
    true_states = np.empty(
        (probe_count, menu_size, config.horizon, 1),
        dtype=np.float64,
    )
    true_rewards = np.empty(
        (probe_count, menu_size, config.horizon),
        dtype=np.float64,
    )

    for probe_index in range(probe_count):
        for menu_index in range(menu_size):
            observation = initial[probe_index].astype(np.float64)
            goal = goals[probe_index].astype(np.float64)
            for step, action_f32 in enumerate(action_sequences[menu_index]):
                action = action_f32.astype(np.float64)
                observation = _true_next_observation(observation, action, config)
                true_states[probe_index, menu_index, step] = observation
                true_rewards[probe_index, menu_index, step] = _known_reward(
                    observation,
                    goal,
                    action,
                    config,
                )

    true_returns = true_rewards.sum(axis=2)
    if np.any(np.ptp(true_returns, axis=1) <= 1.0e-10):
        raise RuntimeError("decision menu contains a zero-return-range probe")
    return DecisionProbeSet(
        seed=seed,
        initial_observations=initial,
        goals=goals,
        domain_indices=domains,
        action_sequences=action_sequences,
        true_next_observations=true_states,
        true_rewards=true_rewards,
        true_returns=true_returns,
    )


def _phase_transitions(
    config: DecisionFidelityConfig,
    seed: int,
    *,
    domain: int,
    phase_index: int,
) -> tuple[Array, Array, Array]:
    """Generate one deterministic online visitation phase.

    Observations sweep ``center ± 0.55`` (domain A centers at -0.78, B at
    +0.78), covering the disjoint state ranges the held-out probes are drawn
    from.  The observation and action sinusoids use incommensurate
    frequencies (0.137 vs 0.193 rad/step), so the visited (state, action)
    pairs fill a two-dimensional region rather than retracing one curve; the
    seed and phase-index offsets decorrelate the sweep across seeds and
    across repeat visits to the same domain.
    """

    time = np.arange(config.phase_steps, dtype=np.float64)
    center = -0.78 if domain == 0 else 0.78
    observations = center + 0.55 * np.sin(0.137 * time + 0.17 * seed + 0.73 * phase_index)
    actions = 0.90 * np.sin(0.193 * time + 0.31 * seed + 1.11 * phase_index)
    observations_f64 = observations[:, None]
    actions_f64 = actions[:, None]
    next_observations = np.stack(
        [
            _true_next_observation(observation, action, config)
            for observation, action in zip(
                observations_f64,
                actions_f64,
                strict=True,
            )
        ],
        axis=0,
    )
    return (
        jnp.asarray(observations_f64, dtype=jnp.float32),
        jnp.asarray(actions_f64, dtype=jnp.float32),
        jnp.asarray(next_observations, dtype=jnp.float32),
    )


def _sparse_rollout_arrays(
    model: SparseFTLWorldModel,
    state: SparseFTLWorldModelState,
    initial_observations: Array,
    goals: Array,
    action_sequences: Array,
    config: DecisionFidelityConfig,
) -> tuple[Array, Array, Array]:
    def rollout_sequence(
        initial_observation: Array,
        goal: Array,
        actions: Array,
    ) -> tuple[Array, Array, Array]:
        def step_fn(observation: Array, action: Array) -> tuple[Array, tuple[Array, Array]]:
            next_observation = model.predict(state, observation, action).next_observation
            reward = _known_reward_jax(next_observation, goal, action, config)
            return next_observation, (next_observation, reward)

        _, (next_observations, rewards) = jax.lax.scan(
            step_fn,
            initial_observation,
            actions,
        )
        return next_observations, rewards, rewards.sum()

    def rollout_probe(
        initial_observation: Array,
        goal: Array,
    ) -> tuple[Array, Array, Array]:
        return jax.vmap(lambda actions: rollout_sequence(initial_observation, goal, actions))(
            action_sequences
        )

    return jax.vmap(rollout_probe)(initial_observations, goals)


def predict_sparse_probe_rollouts(
    model: SparseFTLWorldModel,
    state: SparseFTLWorldModelState,
    probes: DecisionProbeSet,
    config: DecisionFidelityConfig,
) -> PredictedRollouts:
    """Branch every menu choice from one read-only sparse-model snapshot."""

    next_observations, rewards, returns = _sparse_rollout_arrays(
        model,
        state,
        jnp.asarray(probes.initial_observations),
        jnp.asarray(probes.goals),
        jnp.asarray(probes.action_sequences),
        config,
    )
    return PredictedRollouts(
        next_observations=np.asarray(next_observations, dtype=np.float64),
        rewards=np.asarray(rewards, dtype=np.float64),
        returns=np.asarray(returns, dtype=np.float64),
    )


def _linear_features(observation: Array, action: Array) -> Array:
    state = observation[0]
    control = action[0]
    return jnp.asarray((1.0, state, control, state * control), dtype=jnp.float32)


def _initial_linear_state() -> _LinearRidgeState:
    return _LinearRidgeState(
        gram=jnp.zeros((4, 4), dtype=jnp.float32),
        cross=jnp.zeros((4, 1), dtype=jnp.float32),
        weights=jnp.zeros((4, 1), dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
    )


def _run_linear_phase(
    state: _LinearRidgeState,
    observations: Array,
    actions: Array,
    next_observations: Array,
    ridge: float,
) -> _LinearRidgeState:
    def step_fn(
        carry: _LinearRidgeState,
        transition: tuple[Array, Array, Array],
    ) -> tuple[_LinearRidgeState, None]:
        observation, action, next_observation = transition
        features = _linear_features(observation, action)
        target_delta = next_observation - observation
        gram = carry.gram + jnp.outer(features, features)
        cross = carry.cross + jnp.outer(features, target_delta)
        weights = jnp.linalg.solve(
            gram + ridge * jnp.eye(4, dtype=jnp.float32),
            cross,
        )
        next_state = _LinearRidgeState(
            gram=gram,
            cross=cross,
            weights=weights,
            step_count=carry.step_count + jnp.array(1, dtype=jnp.int32),
        )
        return next_state, None

    final_state, _ = jax.lax.scan(
        step_fn,
        state,
        (observations, actions, next_observations),
    )
    return final_state


def _linear_rollout_arrays(
    state: _LinearRidgeState,
    initial_observations: Array,
    goals: Array,
    action_sequences: Array,
    config: DecisionFidelityConfig,
) -> tuple[Array, Array, Array]:
    def rollout_sequence(
        initial_observation: Array,
        goal: Array,
        actions: Array,
    ) -> tuple[Array, Array, Array]:
        def step_fn(observation: Array, action: Array) -> tuple[Array, tuple[Array, Array]]:
            delta = jnp.clip(
                _linear_features(observation, action) @ state.weights,
                -config.prediction_clip,
                config.prediction_clip,
            )
            next_observation = observation + delta
            reward = _known_reward_jax(next_observation, goal, action, config)
            return next_observation, (next_observation, reward)

        _, (next_observations, rewards) = jax.lax.scan(
            step_fn,
            initial_observation,
            actions,
        )
        return next_observations, rewards, rewards.sum()

    def rollout_probe(
        initial_observation: Array,
        goal: Array,
    ) -> tuple[Array, Array, Array]:
        return jax.vmap(lambda actions: rollout_sequence(initial_observation, goal, actions))(
            action_sequences
        )

    return jax.vmap(rollout_probe)(initial_observations, goals)


def _decision_metrics(
    condition: str,
    probes: DecisionProbeSet,
    predicted: PredictedRollouts,
) -> DecisionMetrics:
    if not (
        np.all(np.isfinite(predicted.next_observations))
        and np.all(np.isfinite(predicted.rewards))
        and np.all(np.isfinite(predicted.returns))
    ):
        raise FloatingPointError(f"{condition} produced non-finite rollouts")
    true_returns = probes.true_returns
    best_returns = true_returns.max(axis=1)
    worst_returns = true_returns.min(axis=1)
    return_ranges = best_returns - worst_returns
    selected = np.argmax(predicted.returns, axis=1)
    selected_true_returns = true_returns[
        np.arange(true_returns.shape[0]),
        selected,
    ]
    normalized_regrets = (best_returns - selected_true_returns) / return_ranges
    normalized_return_errors = np.abs(predicted.returns - true_returns).mean(axis=1) / return_ranges
    oracle_picks = np.isclose(
        selected_true_returns,
        best_returns,
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    domain_a = probes.domain_indices == 0
    domain_b = probes.domain_indices == 1
    return DecisionMetrics(
        condition=condition,
        normalized_regret=float(normalized_regrets.mean()),
        domain_a_normalized_regret=float(normalized_regrets[domain_a].mean()),
        domain_b_normalized_regret=float(normalized_regrets[domain_b].mean()),
        oracle_pick_rate=float(oracle_picks.mean()),
        reward_mae=float(np.abs(predicted.rewards - probes.true_rewards).mean()),
        return_mae=float(np.abs(predicted.returns - true_returns).mean()),
        normalized_return_mae=float(normalized_return_errors.mean()),
    )


def evaluate_sparse_snapshot(
    condition: str,
    model: SparseFTLWorldModel,
    state: SparseFTLWorldModelState,
    probes: DecisionProbeSet,
    config: DecisionFidelityConfig,
) -> DecisionMetrics:
    """Score a sparse model without mutating its persistent state."""

    return _decision_metrics(
        condition,
        probes,
        predict_sparse_probe_rollouts(model, state, probes, config),
    )


def _bootstrap_mean(
    values: NDArray[np.float64],
    *,
    config: DecisionFidelityConfig,
    seed_offset: int,
) -> BootstrapEstimate:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    _validate_bootstrap_capacity(config, int(values.size))
    generator = np.random.default_rng(config.bootstrap_seed + seed_offset)
    indices = generator.integers(
        0,
        values.size,
        size=(config.bootstrap_resamples, values.size),
        dtype=np.int64,
    )
    resampled_means = values[indices].mean(axis=1)
    tail = (1.0 - config.confidence_level) / 2.0
    lower, upper = np.quantile(resampled_means, (tail, 1.0 - tail))
    return BootstrapEstimate(
        estimate=float(values.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence_level=config.confidence_level,
        resamples=config.bootstrap_resamples,
        sample_size=int(values.size),
    )


def _metric_vector(
    seed_results: Sequence[SeedDecisionResult],
    condition: str,
    field: str,
) -> NDArray[np.float64]:
    values: list[float] = []
    for result in seed_results:
        matched = [metric for metric in result.metrics if metric.condition == condition]
        if len(matched) != 1:
            raise RuntimeError(f"expected one {condition!r} metric for seed {result.seed}")
        values.append(float(getattr(matched[0], field)))
    return np.asarray(values, dtype=np.float64)


def _aggregate_conditions(
    seed_results: Sequence[SeedDecisionResult],
    config: DecisionFidelityConfig,
) -> tuple[ConditionAggregate, ...]:
    aggregates: list[ConditionAggregate] = []
    offset = 0
    for condition in CONDITION_NAMES:
        estimates: dict[str, BootstrapEstimate] = {}
        for field in (
            "normalized_regret",
            "domain_a_normalized_regret",
            "domain_b_normalized_regret",
            "oracle_pick_rate",
            "reward_mae",
            "return_mae",
            "normalized_return_mae",
        ):
            estimates[field] = _bootstrap_mean(
                _metric_vector(seed_results, condition, field),
                config=config,
                seed_offset=offset,
            )
            offset += 1
        aggregates.append(
            ConditionAggregate(
                condition=condition,
                normalized_regret=estimates["normalized_regret"],
                domain_a_normalized_regret=estimates["domain_a_normalized_regret"],
                domain_b_normalized_regret=estimates["domain_b_normalized_regret"],
                oracle_pick_rate=estimates["oracle_pick_rate"],
                reward_mae=estimates["reward_mae"],
                return_mae=estimates["return_mae"],
                normalized_return_mae=estimates["normalized_return_mae"],
            )
        )
    return tuple(aggregates)


def _paired_comparisons(
    seed_results: Sequence[SeedDecisionResult],
    config: DecisionFidelityConfig,
) -> tuple[PairedComparison, ...]:
    definitions = (
        (
            "sparse_after_b_vs_untrained_regret_reduction",
            "sparse_untrained regret - sparse_after_b regret; positive favors learning",
            "sparse_untrained",
            "sparse_after_b",
            "normalized_regret",
        ),
        (
            "sparse_after_b_vs_linear_regret_reduction",
            "linear_after_b regret - sparse_after_b regret; positive favors sparse",
            "linear_after_b",
            "sparse_after_b",
            "normalized_regret",
        ),
        (
            "sparse_domain_a_regret_change_after_b",
            "sparse_after_b A regret - sparse_after_a1 A regret; positive is interference",
            "sparse_after_b",
            "sparse_after_a1",
            "domain_a_normalized_regret",
        ),
        (
            "sparse_domain_a_recovery_after_a2",
            "sparse_after_b A regret - sparse_after_a2 A regret; positive is recovery",
            "sparse_after_b",
            "sparse_after_a2",
            "domain_a_normalized_regret",
        ),
    )
    comparisons: list[PairedComparison] = []
    for offset, (name, definition, left, right, field) in enumerate(definitions, 100):
        difference = _metric_vector(
            seed_results,
            left,
            field,
        ) - _metric_vector(seed_results, right, field)
        comparisons.append(
            PairedComparison(
                name=name,
                definition=definition,
                estimate=_bootstrap_mean(
                    difference,
                    config=config,
                    seed_offset=offset,
                ),
            )
        )
    return tuple(comparisons)


def run_ftl_decision_fidelity_evaluation(
    *,
    config: DecisionFidelityConfig | None = None,
    seeds: Sequence[int] = EVIDENCE_SEEDS,
) -> DecisionFidelityReport:
    """Run paired untrained, sparse, and raw-linear decision probes.

    The default is the frozen evidence schedule.  Explicit custom schedules
    are useful for development, but callers must not describe them as
    promoted held-out evidence without separately recording their role.
    """

    resolved = DecisionFidelityConfig() if config is None else config
    if type(seeds) not in (tuple, list):
        raise ValueError("seeds must be an actual tuple or list of exact integers")
    seed_count = len(seeds)
    if seed_count == 0:
        raise ValueError("seeds must be non-empty")
    probe_count = 2 * resolved.probes_per_domain
    menu_count = len(resolved.menu_amplitudes)
    _require_derived_int32(
        "evaluation phase transitions", seed_count * 3 * resolved.phase_steps
    )
    _require_derived_int32(
        "evaluation rollout work",
        seed_count * probe_count * menu_count * resolved.horizon * len(CONDITION_NAMES),
    )
    _validate_bootstrap_capacity(resolved, seed_count)
    raw_seeds = tuple(seeds)
    seed_tuple = tuple(
        _require_int32(f"seeds[{index}]", seed, minimum=0)
        for index, seed in enumerate(raw_seeds)
    )
    if len(set(seed_tuple)) != len(seed_tuple):
        raise ValueError("seeds must be unique for paired evidence")

    sparse_model = SparseFTLWorldModel(
        SparseFTLWorldModelConfig(
            observation_dim=1,
            action_dim=1,
            projection_dim=resolved.projection_dim,
            bins=resolved.bins,
            ridge=resolved.ridge,
            statistics_decay=1.0,
            prediction_clip=resolved.prediction_clip,
        )
    )
    train_sparse = jax.jit(
        lambda state, observations, actions, next_observations: (
            run_sparse_ftl_world_model(
                sparse_model,
                state,
                observations,
                actions,
                next_observations,
            ).state
        )
    )
    train_linear = jax.jit(
        lambda state, observations, actions, next_observations: _run_linear_phase(
            state,
            observations,
            actions,
            next_observations,
            resolved.ridge,
        )
    )
    sparse_rollouts = jax.jit(
        lambda state, initial, goals, action_sequences: _sparse_rollout_arrays(
            sparse_model,
            state,
            initial,
            goals,
            action_sequences,
            resolved,
        )
    )
    linear_rollouts = jax.jit(
        lambda state, initial, goals, action_sequences: _linear_rollout_arrays(
            state,
            initial,
            goals,
            action_sequences,
            resolved,
        )
    )

    results: list[SeedDecisionResult] = []
    phase_specs = (
        ("a1", 0, 0),
        ("b", 1, 1),
        ("a2", 0, 2),
    )
    for seed in seed_tuple:
        probes = precompute_decision_probes(resolved, seed)
        initial = jnp.asarray(probes.initial_observations)
        goals = jnp.asarray(probes.goals)
        action_sequences = jnp.asarray(probes.action_sequences)
        sparse_state = sparse_model.init(jr.key(seed))
        linear_state = _initial_linear_state()
        seed_metrics: list[DecisionMetrics] = []

        sparse_arrays = sparse_rollouts(
            sparse_state,
            initial,
            goals,
            action_sequences,
        )
        seed_metrics.append(
            _decision_metrics(
                "sparse_untrained",
                probes,
                PredictedRollouts(
                    next_observations=np.asarray(
                        sparse_arrays[0],
                        dtype=np.float64,
                    ),
                    rewards=np.asarray(sparse_arrays[1], dtype=np.float64),
                    returns=np.asarray(sparse_arrays[2], dtype=np.float64),
                ),
            )
        )

        sparse_snapshots: dict[str, SparseFTLWorldModelState] = {}
        linear_snapshots: dict[str, _LinearRidgeState] = {}
        for phase_name, domain, phase_index in phase_specs:
            transitions = _phase_transitions(
                resolved,
                seed,
                domain=domain,
                phase_index=phase_index,
            )
            sparse_state = train_sparse(sparse_state, *transitions)
            linear_state = train_linear(linear_state, *transitions)
            sparse_snapshots[phase_name] = sparse_state
            linear_snapshots[phase_name] = linear_state

        for learner_name, snapshots, rollout in (
            ("sparse", sparse_snapshots, sparse_rollouts),
            ("linear", linear_snapshots, linear_rollouts),
        ):
            for phase_name, _, _ in phase_specs:
                arrays = rollout(
                    snapshots[phase_name],
                    initial,
                    goals,
                    action_sequences,
                )
                predicted = PredictedRollouts(
                    next_observations=np.asarray(arrays[0], dtype=np.float64),
                    rewards=np.asarray(arrays[1], dtype=np.float64),
                    returns=np.asarray(arrays[2], dtype=np.float64),
                )
                seed_metrics.append(
                    _decision_metrics(
                        f"{learner_name}_after_{phase_name}",
                        probes,
                        predicted,
                    )
                )

        ordered_metrics = tuple(
            next(metric for metric in seed_metrics if metric.condition == condition)
            for condition in CONDITION_NAMES
        )
        results.append(SeedDecisionResult(seed=seed, metrics=ordered_metrics))

    result_tuple = tuple(results)
    return DecisionFidelityReport(
        config=resolved,
        seeds=seed_tuple,
        seed_results=result_tuple,
        aggregates=_aggregate_conditions(result_tuple, resolved),
        comparisons=_paired_comparisons(result_tuple, resolved),
    )


__all__ = [
    "BootstrapEstimate",
    "CONDITION_NAMES",
    "ConditionAggregate",
    "DEVELOPMENT_SEEDS",
    "DecisionFidelityConfig",
    "DecisionFidelityReport",
    "DecisionMetrics",
    "DecisionProbeSet",
    "EVIDENCE_SEEDS",
    "PairedComparison",
    "PredictedRollouts",
    "SeedDecisionResult",
    "evaluate_sparse_snapshot",
    "fixed_action_sequence_menu",
    "precompute_decision_probes",
    "predict_sparse_probe_rollouts",
    "run_ftl_decision_fidelity_evaluation",
]
