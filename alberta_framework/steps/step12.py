# mypy: disable-error-code="attr-defined,call-arg"
"""Production-facing Step 12 Intelligence Amplification facade.

Step 12 of the Alberta Plan — "Prototype-IA: Intelligence Amplification" —
demonstrates that an IA agent can increase the decision-making capacity of a
*partner* agent in non-trivial ways.  The IA agent is not a standalone
autonomous system; it amplifies another agent's intelligence.

Two augmentation streams are provided:

* **Exo-cerebellum** — An online multi-output linear predictor that learns to
  anticipate future observation features.  Its prediction vector becomes an
  augmented feature channel for the partner.
* **Exo-cortex** — An OaK-based (Step 11) agent that learns from the partner's
  experience and broadcasts greedy action recommendations.  The partner can
  accept or ignore these recommendations.

At each step the IA agent returns:

* ``predictions`` — shape ``(n_demons,)`` cerebellum predictions.
* ``recommendation`` — scalar int32 cortex action recommendation.
* ``augmented_obs`` — ``concat(partner_obs, predictions)``, a drop-in
  replacement for the partner's raw observation that adds predictive context.

References:
    Sutton, Bowling, & Pilarski (2022). "The Alberta Plan for AI Research."
    Mathewson et al. (2023). "Communicative Capital." *Neural Comp. & Apps.*
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._float32 import round_real_to_float32
from alberta_framework.core.intelligence_amplification import (
    ExoCerebellumConfig,
    IAAgent,
    IAArrayResult,
    IAConfig,
    IAState,
    IAUpdateResult,
    RecommendationProtocolConfig,
    RecommendationProtocolResult,
    RecommendationProtocolState,
    init_recommendation_protocol_state,
    update_recommendation_protocol,
)
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec


@dataclass(frozen=True)
class Step12IAConfig:
    """Configuration for the production Step 12 IA facade.

    Args:
        n_demons: Number of exo-cerebellum prediction heads.
        cerebellum_step_size: Learning rate for cerebellum weight updates.
        subtask_specs: Subtask specs for the exo-cortex OaK agent.
        observation_dim: Flat observation dimensionality.
        n_primitive_actions: Number of primitive discrete actions.
        base_step_size: Cortex base Q step-size.
        base_avg_reward_step_size: Cortex base average-reward step-size.
        option_step_size: Cortex intra-option Q step-size.
        option_gamma: Cortex option discount.
        option_planning_backups_per_step: Fixed cortex option-model planning
            backup budget per real transition. ``0`` disables planning.
        epsilon_base: Cortex exploration rate.
        utility_ema_decay: Cortex option utility EMA decay.
    """

    n_demons: int = 4
    cerebellum_step_size: float = 0.05
    subtask_specs: tuple[SubtaskSpec, ...] = ()
    observation_dim: int = 4
    n_primitive_actions: int = 2
    base_step_size: float = 0.05
    base_avg_reward_step_size: float = 0.01
    option_step_size: float = 0.05
    option_gamma: float = 0.99
    option_planning_backups_per_step: int = 0
    epsilon_base: float = 0.1
    utility_ema_decay: float = 0.99

    def __post_init__(self) -> None:
        """Reject illegal dimensions and scientific scalars, then canonicalize."""
        _validate_ia_facade_config(self)

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "type": "Step12IAConfig",
            "n_demons": self.n_demons,
            "cerebellum_step_size": self.cerebellum_step_size,
            "subtask_specs": [asdict(s) for s in self.subtask_specs],
            "observation_dim": self.observation_dim,
            "n_primitive_actions": self.n_primitive_actions,
            "base_step_size": self.base_step_size,
            "base_avg_reward_step_size": self.base_avg_reward_step_size,
            "option_step_size": self.option_step_size,
            "option_gamma": self.option_gamma,
            "option_planning_backups_per_step": self.option_planning_backups_per_step,
            "epsilon_base": self.epsilon_base,
            "utility_ema_decay": self.utility_ema_decay,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> Step12IAConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = dict(payload)
        data.pop("type", None)
        specs_raw = data.pop("subtask_specs", [])
        specs = tuple(SubtaskSpec(**s) for s in specs_raw)
        return cls(subtask_specs=specs, **data)

    def to_ia_config(self) -> IAConfig:
        """Convert to the core :class:`IAConfig`."""
        specs = self.subtask_specs
        if not specs:
            specs = (SubtaskSpec(feature_index=0),)
        stomp = STOMPConfig(
            subtask_specs=specs,
            observation_dim=self.observation_dim,
            n_primitive_actions=self.n_primitive_actions,
            base_step_size=self.base_step_size,
            base_avg_reward_step_size=self.base_avg_reward_step_size,
            option_step_size=self.option_step_size,
            option_gamma=self.option_gamma,
            option_planning_backups_per_step=self.option_planning_backups_per_step,
            epsilon_base=self.epsilon_base,
        )
        cortex = OaKConfig(stomp=stomp, utility_ema_decay=self.utility_ema_decay)
        cerebellum = ExoCerebellumConfig(
            n_demons=self.n_demons,
            obs_dim=self.observation_dim,
            step_size=self.cerebellum_step_size,
        )
        return IAConfig(cerebellum=cerebellum, cortex=cortex)


_INT32_MAX = 2**31 - 1


def _require_real(name: str, value: object) -> float:
    """Return a concrete real scalar after direct float32 sink validation."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        narrowed = round_real_to_float32(value)
    except (FloatingPointError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite in float32, got {value!r}") from exc
    if not bool(np.isfinite(narrowed)):
        raise ValueError(f"{name} must be finite in float32, got {value!r}")
    return narrowed


def _require_unit_interval(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    original: Any = value
    if original < 0.0 or original > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return _require_real(name, value)


def _require_nonnegative_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    original: Any = value
    if original < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return _require_real(name, value)


def _require_positive_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    original: Any = value
    if original <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    narrowed = _require_real(name, value)
    if narrowed <= 0.0:
        raise ValueError(f"{name} must remain positive in float32, got {value!r}")
    return narrowed


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    exclusive_maximum: int | None = None,
) -> int:
    actual_type = type(value)
    if actual_type is bool or not issubclass(actual_type, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    number = int(cast(Integral, value))
    if minimum is not None and number < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive, got {value!r}")
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative, got {value!r}")
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    if exclusive_maximum is not None and number >= exclusive_maximum:
        raise ValueError(f"{name} must be smaller than int32 max, got {value!r}")
    return number


def _validate_ia_facade_config(config: Step12IAConfig) -> None:
    n_demons = _require_int("n_demons", config.n_demons, minimum=1)
    observation_dim = _require_int("observation_dim", config.observation_dim, minimum=1)
    n_primitive_actions = _require_int(
        "n_primitive_actions",
        config.n_primitive_actions,
        minimum=1,
    )
    option_planning_backups_per_step = _require_int(
        "option_planning_backups_per_step",
        config.option_planning_backups_per_step,
        minimum=0,
        exclusive_maximum=_INT32_MAX,
    )
    if not isinstance(config.subtask_specs, tuple):
        raise ValueError(
            f"subtask_specs must be a tuple of SubtaskSpec, got {config.subtask_specs!r}"
        )
    canonical_specs: list[SubtaskSpec] = []
    for spec in config.subtask_specs:
        if not isinstance(spec, SubtaskSpec):
            raise ValueError(f"subtask_specs must contain SubtaskSpec values, got {spec!r}")
        feature_index = _require_int("feature_index", spec.feature_index, minimum=0)
        if feature_index >= observation_dim:
            raise ValueError(
                f"feature_index must be < observation_dim, got {spec.feature_index!r}"
            )
        threshold = _require_positive_real("threshold", spec.threshold)
        pseudo_reward_scale = _require_real(
            "pseudo_reward_scale",
            spec.pseudo_reward_scale,
        )
        max_option_steps = _require_int(
            "max_option_steps",
            spec.max_option_steps,
            minimum=1,
        )
        if max_option_steps > _INT32_MAX:
            raise ValueError("max_option_steps must fit int32 telemetry")
        canonical_specs.append(
            SubtaskSpec(
                feature_index=feature_index,
                threshold=threshold,
                pseudo_reward_scale=pseudo_reward_scale,
                max_option_steps=max_option_steps,
            )
        )
    cerebellum_step_size = _require_positive_real(
        "cerebellum_step_size",
        config.cerebellum_step_size,
    )
    base_step_size = _require_nonnegative_real("base_step_size", config.base_step_size)
    base_avg_reward_step_size = _require_nonnegative_real(
        "base_avg_reward_step_size",
        config.base_avg_reward_step_size,
    )
    option_step_size = _require_nonnegative_real(
        "option_step_size",
        config.option_step_size,
    )
    option_gamma = _require_unit_interval("option_gamma", config.option_gamma)
    epsilon_base = _require_unit_interval("epsilon_base", config.epsilon_base)
    utility_ema_decay = _require_unit_interval(
        "utility_ema_decay",
        config.utility_ema_decay,
    )
    object.__setattr__(config, "n_demons", n_demons)
    object.__setattr__(config, "subtask_specs", tuple(canonical_specs))
    object.__setattr__(config, "observation_dim", observation_dim)
    object.__setattr__(config, "n_primitive_actions", n_primitive_actions)
    object.__setattr__(config, "cerebellum_step_size", cerebellum_step_size)
    object.__setattr__(config, "base_step_size", base_step_size)
    object.__setattr__(config, "base_avg_reward_step_size", base_avg_reward_step_size)
    object.__setattr__(config, "option_step_size", option_step_size)
    object.__setattr__(config, "option_gamma", option_gamma)
    object.__setattr__(
        config,
        "option_planning_backups_per_step",
        option_planning_backups_per_step,
    )
    object.__setattr__(config, "epsilon_base", epsilon_base)
    object.__setattr__(config, "utility_ema_decay", utility_ema_decay)


@dataclass(frozen=True)
class Step12SmokeResult:
    """Summary returned by :func:`run_step12_smoke`."""

    config: Step12IAConfig
    steps: int
    seed: int
    predictions_shape: tuple[int, ...]
    cerebellum_errors_shape: tuple[int, ...]
    recommendations_shape: tuple[int, ...]
    augmented_obs_shape: tuple[int, ...]
    cortex_td_errors_shape: tuple[int, ...]
    finite: bool
    agent_config: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "config": self.config.to_config(),
            "steps": self.steps,
            "seed": self.seed,
            "predictions_shape": list(self.predictions_shape),
            "cerebellum_errors_shape": list(self.cerebellum_errors_shape),
            "recommendations_shape": list(self.recommendations_shape),
            "augmented_obs_shape": list(self.augmented_obs_shape),
            "cortex_td_errors_shape": list(self.cortex_td_errors_shape),
            "finite": self.finite,
            "agent_config": self.agent_config,
        }


def make_step12_ia_agent(config: Step12IAConfig | None = None) -> IAAgent:
    """Create an :class:`IAAgent` from a :class:`Step12IAConfig`.

    Args:
        config: Step 12 configuration.  Defaults to 4 cerebellum demons and
            one cortex subtask on feature 0.

    Returns:
        Initialised :class:`IAAgent`.
    """
    if config is None:
        config = Step12IAConfig()
    return IAAgent(config.to_ia_config())


def init_step12_state(
    agent: IAAgent,
    *,
    key: Array,
    initial_observation: Array,
) -> IAState:
    """Initialise and prime the Step 12 IA state.

    Args:
        agent: The :class:`IAAgent` to initialise.
        key: JAX PRNG key.
        initial_observation: First real observation from the environment.

    Returns:
        Primed :class:`IAState`.
    """
    init_key, _ = jr.split(key)
    state = agent.init(init_key)
    obs = jnp.asarray(initial_observation, dtype=jnp.float32)
    return agent.start(state, obs)


def step12_update(
    agent: IAAgent,
    state: IAState,
    partner_obs: Array,
    partner_reward: Array,
    partner_next_obs: Array,
) -> IAUpdateResult:
    """Run one IA step from partner experience.

    Args:
        agent: The IA agent.
        state: Current IA state.
        partner_obs: Partner's current observation.
        partner_reward: Partner's received reward.
        partner_next_obs: Partner's next observation.

    Returns:
        :class:`IAUpdateResult` with predictions, recommendation, and
        augmented observation.
    """
    return agent.update(state, partner_obs, partner_reward, partner_next_obs)


def run_step12_scan(
    agent: IAAgent,
    state: IAState,
    partner_obs: Array,
    partner_rewards: Array,
    partner_next_obs: Array,
) -> IAArrayResult:
    """Run the IA agent over pre-collected partner transition arrays.

    Args:
        agent: The IA agent.
        state: Starting IA state.
        partner_obs: Shape ``(T, obs_dim)`` partner observations.
        partner_rewards: Shape ``(T,)`` partner rewards.
        partner_next_obs: Shape ``(T, obs_dim)`` partner next observations.

    Returns:
        :class:`IAArrayResult` with per-step diagnostics.
    """
    return agent.scan(state, partner_obs, partner_rewards, partner_next_obs)


def run_step12_smoke(
    config: Step12IAConfig | None = None,
    *,
    steps: int = 64,
    seed: int = 0,
) -> Step12SmokeResult:
    """Run a deterministic Step 12 IA integration probe.

    Args:
        config: Step 12 configuration.  Defaults to 4 cerebellum demons,
            one cortex subtask on feature 0.
        steps: Number of transition steps to run.
        seed: PRNG seed for reproducibility.

    Returns:
        :class:`Step12SmokeResult` with shape/fineness summary.
    """
    steps = _require_int("steps", steps, minimum=1)

    cfg = config or Step12IAConfig()
    agent = make_step12_ia_agent(cfg)
    obs_dim = cfg.observation_dim

    data_key, state_key = jr.split(jr.key(seed))
    observations = jr.normal(data_key, (steps + 1, obs_dim), dtype=jnp.float32)
    rewards = jnp.tanh(observations[1:, 0])

    state = init_step12_state(agent, key=state_key, initial_observation=observations[0])
    result = run_step12_scan(
        agent,
        state,
        observations[:-1],
        rewards,
        observations[1:],
    )
    result.cortex_td_errors.block_until_ready()

    finite = bool(
        jnp.all(jnp.isfinite(result.predictions))
        & jnp.all(jnp.isfinite(result.cerebellum_errors))
        & jnp.all(jnp.isfinite(result.cortex_td_errors))
        & jnp.all(jnp.isfinite(result.augmented_obs))
        & jnp.all(result.recommendations >= 0)
        & jnp.all(result.recommendations < cfg.n_primitive_actions)
        & jnp.all(result.updates_applied)
    )

    return Step12SmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        predictions_shape=tuple(int(d) for d in result.predictions.shape),
        cerebellum_errors_shape=tuple(int(d) for d in result.cerebellum_errors.shape),
        recommendations_shape=tuple(int(d) for d in result.recommendations.shape),
        augmented_obs_shape=tuple(int(d) for d in result.augmented_obs.shape),
        cortex_td_errors_shape=tuple(int(d) for d in result.cortex_td_errors.shape),
        finite=finite,
        agent_config=agent.to_config(),
    )


__all__ = [
    "RecommendationProtocolConfig",
    "RecommendationProtocolResult",
    "RecommendationProtocolState",
    "Step12IAConfig",
    "Step12SmokeResult",
    "init_step12_state",
    "init_recommendation_protocol_state",
    "make_step12_ia_agent",
    "run_step12_scan",
    "run_step12_smoke",
    "step12_update",
    "update_recommendation_protocol",
]
