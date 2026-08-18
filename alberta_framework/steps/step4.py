# mypy: disable-error-code="attr-defined,call-arg"
"""Public Step 4 SARSA control facade (Alberta Plan Step 4, Control I).

This module keeps the packaged Step 4 surface narrow: construct a SARSA agent,
prime it with an initial feature vector, run one online transition, or scan over
pre-collected feature/reward arrays.

The science lives in :mod:`alberta_framework.core.sarsa`: a Horde-backed
epsilon-greedy semi-gradient SARSA(lambda) (Sutton & Barto 2018, §10.1) with
one control demon per action.  The defaults follow the streaming deep-RL
stability recipe of Elsayed et al. 2024 ("Streaming Deep Reinforcement
Learning Finally Works"): sparse weight initialization (``sparsity``),
parameterless layer normalization (``use_layer_norm``), and global ObGD
update bounding (``bounder="obgd"`` with sensitivity ``bounder_kappa``).

References:
    Sutton & Barto (2018). "Reinforcement Learning: An Introduction," §10.1.
    Elsayed et al. (2024). "Streaming Deep Reinforcement Learning Finally
        Works."
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.core.optimizers import (
    IDBD,
    LMS,
    Autostep,
    Bounder,
    ObGDBounding,
)
from alberta_framework.core.sarsa import (
    SARSAAgent,
    SARSAArrayResult,
    SARSAConfig,
    SARSAState,
    SARSAUpdateResult,
)
from alberta_framework.core.types import GVFSpec, TraceMode
from alberta_framework.steps._float32_validation import (
    canonical_float32_storage,
    finite_real_and_float32,
)
from alberta_framework.steps._smoke_record_validation import require_step_shape

Step4OptimizerName = Literal["lms", "idbd", "autostep"]
Step4BounderName = Literal["none", "obgd"]
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
_FLOAT32_MIN_NORMAL = float.fromhex("0x1.0p-126")


@dataclass(frozen=True)
class Step4SARSAConfig:
    """Config for the public Step 4 SARSA facade."""

    n_actions: int = 2
    hidden_sizes: tuple[int, ...] = (16,)
    gamma: float = 0.99
    epsilon_start: float = 0.1
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 0
    lamda: float = 0.0
    optimizer: Step4OptimizerName = "lms"
    bounder: Step4BounderName = "obgd"
    step_size: float = 0.03
    meta_step_size: float = 0.01
    bounder_kappa: float = 0.5
    sparsity: float = 0.5
    use_layer_norm: bool = True
    trace_mode: Literal["accumulating", "replacing"] = "accumulating"

    def __post_init__(self) -> None:
        """Reject illegal SARSA scientific scalars and canonicalize reals."""
        _validate_sarsa_config(self)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Step4SARSAConfig:
        """Reconstruct from :meth:`to_dict` output."""
        config = dict(payload)
        config["hidden_sizes"] = tuple(cast(list[int], config["hidden_sizes"]))
        return cls(**cast(Any, config))

    def to_sarsa_config(self) -> SARSAConfig:
        """Return the core SARSA configuration."""
        return SARSAConfig(
            n_actions=self.n_actions,
            gamma=self.gamma,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_steps=self.epsilon_decay_steps,
        )


@dataclass(frozen=True)
class Step4SmokeResult:
    """Summary returned by :func:`run_step4_smoke`."""

    config: Step4SARSAConfig
    steps: int
    seed: int
    q_values_shape: tuple[int, ...]
    td_errors_shape: tuple[int, ...]
    actions_shape: tuple[int, ...]
    finite: bool
    agent_config: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", _require_positive_int("steps", self.steps))
        object.__setattr__(self, "seed", require_jax_seed(self.seed, name="seed"))
        for name in ("q_values_shape", "td_errors_shape", "actions_shape"):
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
        payload["actions_shape"] = list(self.actions_shape)
        return payload


@chex.dataclass(frozen=True)
class Step4OneStepResult:
    """Result from one Step 4 transition."""

    state: SARSAState
    action: Array
    q_values: Array
    td_error: Array
    reward: Array


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
    return canonical_float32_storage(real, narrowed)


def _require_gvf_probability(name: str, value: object) -> float:
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
    if (
        (numerator != 0 and numerator << 126 < denominator)
        or (real != 0.0 and real < _FLOAT32_MIN_NORMAL)
        or (narrowed != 0.0 and narrowed < _FLOAT32_MIN_NORMAL)
    ):
        raise ValueError(f"{name} must be zero or a normal float32 value in [0, 1]")
    return canonical_float32_storage(real, narrowed)


def _require_nonnegative_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = finite_real_and_float32(name, value)
    if real < 0.0 or numerator < 0 or narrowed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return canonical_float32_storage(real, narrowed)


def _require_positive_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = finite_real_and_float32(name, value)
    if real <= 0.0 or numerator <= 0 or narrowed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return canonical_float32_storage(real, narrowed)


def _require_positive_int(name: str, value: object) -> int:
    actual_type = type(value)
    if actual_type not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be a positive integer")
    number = int(cast(Integral, value))
    if number < 1:
        raise ValueError(f"{name} must be positive")
    if number > _INT32_MAX:
        raise ValueError(f"{name} must be at most int32 max")
    return number


def _require_nonneg_int(name: str, value: object) -> int:
    actual_type = type(value)
    if actual_type not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be a non-negative integer")
    number = int(cast(Integral, value))
    if number < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if number > _INT32_MAX:
        raise ValueError(f"{name} must be at most int32 max")
    return number


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_choice(name: str, value: object, choices: tuple[str, ...]) -> str:
    if type(value) is not str or value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")
    return value


def _validate_sarsa_config(config: Step4SARSAConfig) -> None:
    n_actions = _require_positive_int("n_actions", config.n_actions)
    if type(config.hidden_sizes) is not tuple:
        raise ValueError("hidden_sizes must be an actual tuple")
    hidden_sizes = tuple(
        _require_positive_int("hidden_sizes", size) for size in config.hidden_sizes
    )
    gamma = _require_gvf_probability("gamma", config.gamma)
    epsilon_start = _require_unit_interval("epsilon_start", config.epsilon_start)
    epsilon_end = _require_unit_interval("epsilon_end", config.epsilon_end)
    epsilon_decay_steps = _require_nonneg_int("epsilon_decay_steps", config.epsilon_decay_steps)
    lamda = _require_gvf_probability("lamda", config.lamda)
    step_size = _require_nonnegative_real("step_size", config.step_size)
    meta_step_size = _require_nonnegative_real("meta_step_size", config.meta_step_size)
    bounder_kappa = _require_positive_real("bounder_kappa", config.bounder_kappa)
    sparsity = _require_unit_interval("sparsity", config.sparsity)
    use_layer_norm = _require_bool("use_layer_norm", config.use_layer_norm)
    optimizer = _require_choice("optimizer", config.optimizer, ("lms", "idbd", "autostep"))
    bounder = _require_choice("bounder", config.bounder, ("none", "obgd"))
    trace_mode = _require_choice(
        "trace_mode", config.trace_mode, ("accumulating", "replacing")
    )
    object.__setattr__(config, "n_actions", n_actions)
    object.__setattr__(config, "hidden_sizes", hidden_sizes)
    object.__setattr__(config, "gamma", gamma)
    object.__setattr__(config, "epsilon_start", epsilon_start)
    object.__setattr__(config, "epsilon_end", epsilon_end)
    object.__setattr__(config, "epsilon_decay_steps", epsilon_decay_steps)
    object.__setattr__(config, "lamda", lamda)
    object.__setattr__(config, "step_size", step_size)
    object.__setattr__(config, "meta_step_size", meta_step_size)
    object.__setattr__(config, "bounder_kappa", bounder_kappa)
    object.__setattr__(config, "sparsity", sparsity)
    object.__setattr__(config, "use_layer_norm", use_layer_norm)
    object.__setattr__(config, "optimizer", optimizer)
    object.__setattr__(config, "bounder", bounder)
    object.__setattr__(config, "trace_mode", trace_mode)


def make_step4_optimizer(config: Step4SARSAConfig) -> Any:
    """Construct the configured Step 4 optimizer."""
    if config.optimizer == "lms":
        return LMS(step_size=config.step_size)
    if config.optimizer == "idbd":
        return IDBD(
            initial_step_size=config.step_size,
            meta_step_size=config.meta_step_size,
        )
    if config.optimizer == "autostep":
        return Autostep(
            initial_step_size=config.step_size,
            meta_step_size=config.meta_step_size,
        )
    raise ValueError("unknown Step 4 optimizer")


def make_step4_bounder(config: Step4SARSAConfig) -> Bounder | None:
    """Construct the configured Step 4 update bounder."""
    if config.bounder == "none":
        return None
    if config.bounder == "obgd":
        return ObGDBounding(kappa=config.bounder_kappa)
    raise ValueError("unknown Step 4 bounder")


def make_step4_sarsa_agent(
    config: Step4SARSAConfig | None = None,
    *,
    prediction_demons: tuple[GVFSpec, ...] | None = None,
) -> SARSAAgent:
    """Create the public Step 4 SARSA agent."""
    cfg = config or Step4SARSAConfig()
    return SARSAAgent(
        sarsa_config=cfg.to_sarsa_config(),
        hidden_sizes=cfg.hidden_sizes,
        optimizer=make_step4_optimizer(cfg),
        bounder=make_step4_bounder(cfg),
        sparsity=cfg.sparsity,
        use_layer_norm=cfg.use_layer_norm,
        lamda=cfg.lamda,
        trace_mode=TraceMode(cfg.trace_mode),
        prediction_demons=list(prediction_demons) if prediction_demons else None,
    )


def init_step4_state(
    agent: SARSAAgent,
    *,
    feature_dim: int,
    key: Array,
    initial_features: Array,
) -> SARSAState:
    """Initialize and prime a SARSA state with the first feature vector."""
    state = agent.init(feature_dim, key)
    action, next_key = agent.select_action(state, initial_features)
    return cast(
        SARSAState,
        state.replace(
            last_action=action,
            last_observation=initial_features,
            rng_key=next_key,
        ),
    )


def step4_update(
    agent: SARSAAgent,
    state: SARSAState,
    reward: Array,
    next_features: Array,
    terminated: Array,
    prediction_cumulants: Array | None = None,
) -> Step4OneStepResult:
    """Run one SARSA transition update and select the next action."""
    next_action, next_key = agent.select_action(state, next_features)
    ready_state = state.replace(rng_key=next_key)
    result: SARSAUpdateResult = agent.update(
        ready_state,
        reward,
        next_features,
        terminated,
        next_action,
        prediction_cumulants=prediction_cumulants,
    )
    return Step4OneStepResult(
        state=result.state,
        action=result.action,
        q_values=result.q_values,
        td_error=result.td_error,
        reward=result.reward,
    )


def run_step4_scan(
    agent: SARSAAgent,
    state: SARSAState,
    next_features: Array,
    rewards: Array,
    terminated: Array,
) -> SARSAArrayResult:
    """Run Step 4 SARSA over pre-collected transition arrays."""

    def step_fn(
        carry: SARSAState,
        inputs: tuple[Array, Array, Array],
    ) -> tuple[SARSAState, tuple[Array, Array, Array]]:
        s = carry
        features_t, reward_t, terminated_t = inputs
        result = step4_update(agent, s, reward_t, features_t, terminated_t)
        return result.state, (result.q_values, result.td_error, result.action)

    final_state, (q_values, td_errors, actions) = jax.lax.scan(
        step_fn,
        state,
        (next_features, rewards, terminated),
    )
    return SARSAArrayResult(
        state=final_state,
        q_values=q_values,
        td_errors=td_errors,
        actions=actions,
    )


def run_step4_smoke(
    config: Step4SARSAConfig | None = None,
    *,
    steps: int = 32,
    feature_dim: int = 6,
    seed: int = 0,
) -> Step4SmokeResult:
    """Run a tiny deterministic Step 4 integration probe."""
    steps = _require_positive_int("steps", steps)
    feature_dim = _require_positive_int("feature_dim", feature_dim)
    seed = require_jax_seed(seed, name="seed")

    cfg = config or Step4SARSAConfig()
    agent = make_step4_sarsa_agent(cfg)
    data_key, state_key = jr.split(jr.key(seed))
    observations = jr.normal(data_key, (steps + 1, feature_dim), dtype=jnp.float32)
    rewards = jnp.tanh(observations[1:, 0])
    terminated = jnp.zeros(steps, dtype=jnp.float32)
    state = init_step4_state(
        agent,
        feature_dim=feature_dim,
        key=state_key,
        initial_features=observations[0],
    )
    result = run_step4_scan(
        agent,
        state,
        observations[1:],
        rewards,
        terminated,
    )
    result.q_values.block_until_ready()

    finite = bool(
        jnp.all(jnp.isfinite(result.q_values))
        & jnp.all(jnp.isfinite(result.td_errors))
        & jnp.all(result.actions >= 0)
        & jnp.all(result.actions < cfg.n_actions)
    )
    return Step4SmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        q_values_shape=tuple(int(dim) for dim in result.q_values.shape),
        td_errors_shape=tuple(int(dim) for dim in result.td_errors.shape),
        actions_shape=tuple(int(dim) for dim in result.actions.shape),
        finite=finite,
        agent_config=agent.to_config(),
    )
