# mypy: disable-error-code="call-arg"
"""Public Step 6 average-reward control facade (Control II).

Step 6 of the Alberta Plan is continuing control: the agent maximizes the
long-run reward *rate* rather than a discounted or episodic return.  The
agent behind this facade is the linear differential SARSA learner in
:mod:`alberta_framework.core.average_reward` — epsilon-greedy over
action-values learned relative to an online reward-rate estimate
(``td_error = reward - average_reward + q(s', a') - q(s, a)``).  As in Step 5
prediction, the reward-rate estimate is updated from the same TD error with a
smaller step-size than the action-values, since it enters every TD target.

References:
    Wan, Naik, & Sutton (2021). "Learning and Planning in Average-Reward
        Markov Decision Processes."
    Sutton & Barto (2018). "Reinforcement Learning: An Introduction," §10.3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAArrayResult,
    DifferentialSARSAConfig,
    DifferentialSARSAState,
    DifferentialSARSAUpdateResult,
    run_differential_sarsa_from_arrays,
)
from alberta_framework.steps._float32_validation import finite_real_and_float32
from alberta_framework.steps._smoke_record_validation import require_step_shape

_INT32_MAX = 2**31 - 1
_ACTUAL_INT_TYPES = (
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
)


def _compatible_float32_storage(value: object, narrowed: float) -> float:
    """Preserve compatible builtin payloads without changing the JAX sink."""
    if type(value) is float and (narrowed != 0.0 or value == 0):
        return value
    if type(value) is int and value == narrowed:
        return float(value)
    return narrowed


def _require_unit_interval(name: str, value: object) -> float:
    real, numerator, denominator, narrowed = finite_real_and_float32(name, value)
    if (
        real < 0.0
        or not real <= 1.0
        or numerator < 0
        or numerator > denominator
        or narrowed < 0.0
        or not narrowed <= 1.0
    ):
        raise ValueError(f"{name} must be in [0, 1]")
    return _compatible_float32_storage(real, narrowed)


def _require_nonnegative_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = finite_real_and_float32(name, value)
    if real < 0.0 or numerator < 0 or narrowed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return _compatible_float32_storage(real, narrowed)


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    actual_type = type(value)
    if all(actual_type is not allowed for allowed in _ACTUAL_INT_TYPES):
        raise ValueError(f"{name} must be an integer")
    number = int(cast(Integral, value))
    if minimum is not None and number < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive")
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative")
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most int32 max")
    return number


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a built-in bool")
    return value


def _validate_step6_config(config: Step6DifferentialSARSAConfig) -> None:
    if type(config) is not Step6DifferentialSARSAConfig:
        raise TypeError("config must be an exact Step6DifferentialSARSAConfig")
    n_actions = _require_int(
        "n_actions",
        config.n_actions,
        minimum=1,
        maximum=_INT32_MAX,
    )
    q_step_size = _require_nonnegative_real("q_step_size", config.q_step_size)
    average_reward_step_size = _require_nonnegative_real(
        "average_reward_step_size",
        config.average_reward_step_size,
    )
    trace_decay = _require_unit_interval("trace_decay", config.trace_decay)
    epsilon_start = _require_unit_interval("epsilon_start", config.epsilon_start)
    epsilon_end = _require_unit_interval("epsilon_end", config.epsilon_end)
    epsilon_decay_steps = _require_int(
        "epsilon_decay_steps",
        config.epsilon_decay_steps,
        minimum=0,
        maximum=_INT32_MAX,
    )
    object.__setattr__(config, "n_actions", n_actions)
    object.__setattr__(config, "q_step_size", q_step_size)
    object.__setattr__(config, "average_reward_step_size", average_reward_step_size)
    object.__setattr__(config, "trace_decay", trace_decay)
    object.__setattr__(config, "epsilon_start", epsilon_start)
    object.__setattr__(config, "epsilon_end", epsilon_end)
    object.__setattr__(config, "epsilon_decay_steps", epsilon_decay_steps)
    config.to_core_config()


@dataclass(frozen=True)
class Step6DifferentialSARSAConfig:
    """Config for the public Step 6 differential SARSA facade."""

    n_actions: int = 2
    q_step_size: float = 0.05
    average_reward_step_size: float = 0.01
    trace_decay: float = 0.0
    epsilon_start: float = 0.1
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 0

    def __post_init__(self) -> None:
        """Reject illegal parameters and canonicalize scalars."""
        _validate_step6_config(self)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "n_actions": int(self.n_actions),
            "q_step_size": float(self.q_step_size),
            "average_reward_step_size": float(self.average_reward_step_size),
            "trace_decay": float(self.trace_decay),
            "epsilon_start": float(self.epsilon_start),
            "epsilon_end": float(self.epsilon_end),
            "epsilon_decay_steps": int(self.epsilon_decay_steps),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step6DifferentialSARSAConfig:
        """Reconstruct from :meth:`to_dict` output."""
        if type(payload) is not dict:
            raise ValueError("Step6DifferentialSARSAConfig payload must be an exact dictionary")
        raw = cast(dict[object, object], payload)
        if any(type(key) is not str for key in raw):
            raise ValueError("Step6DifferentialSARSAConfig payload keys must be exact strings")
        if cast(set[str], set(raw)) != frozenset(cls.__dataclass_fields__):
            raise ValueError("Step6DifferentialSARSAConfig payload fields do not match the schema")
        return cls(**cast(Any, dict(raw)))

    def to_core_config(self) -> DifferentialSARSAConfig:
        """Return the core differential SARSA config."""
        return DifferentialSARSAConfig(
            n_actions=self.n_actions,
            q_step_size=self.q_step_size,
            average_reward_step_size=self.average_reward_step_size,
            trace_decay=self.trace_decay,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_steps=self.epsilon_decay_steps,
        )


@dataclass(frozen=True)
class Step6SmokeResult:
    """Summary returned by :func:`run_step6_smoke`."""

    config: Step6DifferentialSARSAConfig
    steps: int
    seed: int
    q_values_shape: tuple[int, ...]
    td_errors_shape: tuple[int, ...]
    average_rewards_shape: tuple[int, ...]
    actions_shape: tuple[int, ...]
    finite: bool
    agent_config: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self.config) is not Step6DifferentialSARSAConfig:
            raise TypeError("config must be an exact Step6DifferentialSARSAConfig")
        if type(self.agent_config) is not dict:
            raise TypeError("agent_config must be an exact dictionary")
        object.__setattr__(
            self, "steps", _require_int("steps", self.steps, minimum=1, maximum=_INT32_MAX)
        )
        object.__setattr__(self, "seed", require_jax_seed(self.seed, name="seed"))
        for name in (
            "q_values_shape",
            "td_errors_shape",
            "average_rewards_shape",
            "actions_shape",
        ):
            object.__setattr__(
                self,
                name,
                require_step_shape(name, getattr(self, name), steps=self.steps),
            )
        object.__setattr__(self, "finite", _require_bool("finite", self.finite))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        payload["q_values_shape"] = list(self.q_values_shape)
        payload["td_errors_shape"] = list(self.td_errors_shape)
        payload["average_rewards_shape"] = list(self.average_rewards_shape)
        payload["actions_shape"] = list(self.actions_shape)
        return payload


def make_step6_differential_sarsa_agent(
    config: Step6DifferentialSARSAConfig | None = None,
) -> DifferentialSARSAAgent:
    """Create the public Step 6 differential SARSA agent."""
    if config is None:
        cfg = Step6DifferentialSARSAConfig()
    elif type(config) is Step6DifferentialSARSAConfig:
        cfg = config
    else:
        raise TypeError("config must be an exact Step6DifferentialSARSAConfig")
    return DifferentialSARSAAgent(cfg.to_core_config())


def _checked_product(name: str, *factors: int) -> int:
    product = 1
    for factor in factors:
        if factor < 0 or (factor != 0 and product > _INT32_MAX // factor):
            raise ValueError(f"derived {name} must fit signed int32")
        product *= factor
    return product


def _checked_sum(name: str, *terms: int) -> int:
    total = 0
    for term in terms:
        if term < 0 or term > _INT32_MAX - total:
            raise ValueError(f"derived {name} must fit signed int32")
        total += term
    return total


def _preflight_step6_smoke_resources(
    config: Step6DifferentialSARSAConfig,
    *,
    steps: int,
    feature_dim: int,
) -> None:
    state_parameters = _checked_product(
        "Step 6 state parameter count", config.n_actions, feature_dim
    )
    _checked_product("Step 6 state parameter bytes", 8, state_parameters)
    rows = _checked_sum("Step 6 observation row count", steps, 1)
    observations = _checked_product("Step 6 observation count", rows, feature_dim)
    q_values = _checked_product("Step 6 q-value count", steps, config.n_actions)
    _checked_sum(
        "Step 6 smoke array bytes",
        _checked_product("Step 6 observation bytes", 4, observations),
        _checked_product("Step 6 q-value bytes", 4, q_values),
        # rewards, TD errors, average rewards, actions, and validity flags.
        _checked_product("Step 6 scalar output bytes", 17, steps),
    )


def init_step6_state(
    agent: DifferentialSARSAAgent,
    *,
    feature_dim: int,
    key: Array,
    initial_features: Array,
) -> DifferentialSARSAState:
    """Initialize and prime a differential SARSA state."""
    feature_dim = _require_int(
        "feature_dim", feature_dim, minimum=1, maximum=_INT32_MAX
    )
    if type(agent) is not DifferentialSARSAAgent:
        raise TypeError("agent must be an exact DifferentialSARSAAgent")
    _checked_product("Step 6 state parameter bytes", 8, agent.config.n_actions, feature_dim)
    state = agent.init(feature_dim, key)
    state, _action = agent.start(state, initial_features)
    return cast(DifferentialSARSAState, state)


def step6_update(
    agent: DifferentialSARSAAgent,
    state: DifferentialSARSAState,
    reward: Array,
    next_features: Array,
) -> DifferentialSARSAUpdateResult:
    """Run one continuing differential SARSA transition update."""
    return cast(DifferentialSARSAUpdateResult, agent.update(state, reward, next_features))


def run_step6_scan(
    agent: DifferentialSARSAAgent,
    state: DifferentialSARSAState,
    rewards: Array,
    next_features: Array,
) -> DifferentialSARSAArrayResult:
    """Run Step 6 differential SARSA over pre-collected transition arrays."""
    return run_differential_sarsa_from_arrays(agent, state, rewards, next_features)


def run_step6_smoke(
    config: Step6DifferentialSARSAConfig | None = None,
    *,
    steps: int = 32,
    feature_dim: int = 6,
    seed: int = 0,
) -> Step6SmokeResult:
    """Run a tiny deterministic Step 6 integration probe."""
    steps = _require_int("steps", steps, minimum=1, maximum=_INT32_MAX)
    feature_dim = _require_int(
        "feature_dim", feature_dim, minimum=1, maximum=_INT32_MAX
    )
    seed = require_jax_seed(seed, name="seed")

    if config is None:
        cfg = Step6DifferentialSARSAConfig()
    elif type(config) is Step6DifferentialSARSAConfig:
        cfg = config
    else:
        raise TypeError("config must be an exact Step6DifferentialSARSAConfig")
    _preflight_step6_smoke_resources(cfg, steps=steps, feature_dim=feature_dim)
    agent = make_step6_differential_sarsa_agent(cfg)
    data_key, state_key = jr.split(jr.key(seed))
    observations = jr.normal(data_key, (steps + 1, feature_dim), dtype=jnp.float32)
    rewards = jnp.tanh(observations[1:, 0])
    state = init_step6_state(
        agent,
        feature_dim=feature_dim,
        key=state_key,
        initial_features=observations[0],
    )
    result = run_differential_sarsa_from_arrays(
        agent,
        state,
        rewards,
        observations[1:],
    )
    result.td_errors.block_until_ready()
    finite = bool(
        jnp.all(jnp.isfinite(result.q_values))
        & jnp.all(jnp.isfinite(result.td_errors))
        & jnp.all(jnp.isfinite(result.average_rewards))
        & jnp.all(result.actions >= 0)
        & jnp.all(result.actions < cfg.n_actions)
        & jnp.all(result.updates_applied)
    )
    return Step6SmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        q_values_shape=tuple(int(dim) for dim in result.q_values.shape),
        td_errors_shape=tuple(int(dim) for dim in result.td_errors.shape),
        average_rewards_shape=tuple(int(dim) for dim in result.average_rewards.shape),
        actions_shape=tuple(int(dim) for dim in result.actions.shape),
        finite=finite,
        agent_config=agent.to_config(),
    )
