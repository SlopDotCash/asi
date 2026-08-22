# mypy: disable-error-code="attr-defined,call-arg"
"""Public Step 10 STOMP facade.

Step 10 of the Alberta Plan introduces the STOMP progression: SubTasks,
Options, Models, Planning.  This is the first step that enables temporal
abstraction — the agent can execute temporally extended actions (options)
defined by feature-reaching subtasks, learn multi-step outcome models for
each option, and plan at the option level.

Architecture:

* **Subtasks** — Feature-reaching sub-problems with pseudo-rewards.
  Each subtask defines one option.
* **Options** — Temporally extended actions.  Each option has its own
  intra-option differential Q-policy trained with subtask pseudo-rewards.
* **Models** — Per-option outcome models tracking cumulative pseudo-reward,
  expected discount, and next-state delta prediction.  Updated at option
  termination.
* **Planning** — The base agent acts over the extended action space
  {primitives, options}.  When an option is selected, its intra-option
  policy drives primitive actions until the option terminates.

The base control is a linear differential Q-function (average-reward
formulation) over the extended action set, exactly as in Step 6.

The facade rejects illegal dimensions and scientific scalars — epsilons,
gamma, decays, step sizes, and clipping/threshold values — before
constructing the core agent. Accepted numbers are canonicalized to builtin
ints and floats; legal endpoints stay valid.

References:
    Sutton, Bowling, & Pilarski (2022). "The Alberta Plan for AI Research."
    Sutton, Precup, & Singh (1999). "Between MDPs and semi-MDPs: A Framework
        for Temporal Abstraction in Reinforcement Learning."
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from numbers import Integral
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.core.options import (
    STOMPAgent,
    STOMPArrayResult,
    STOMPConfig,
    STOMPStartResult,
    STOMPState,
    STOMPUpdateResult,
    SubtaskSpec,
)
from alberta_framework.steps._float32_validation import (
    canonical_float32_storage,
    finite_real_and_float32,
)
from alberta_framework.steps._smoke_record_validation import require_step_shape


@dataclass(frozen=True)
class Step10STOMPConfig:
    """Configuration for the public Step 10 STOMP facade.

    This thin wrapper around :class:`STOMPConfig` adds standard
    dict serialization consistent with the Step 1–9 facades.

    Args:
        subtask_specs: Feature-reaching subtask definitions.  Each entry
            becomes one option.  At least one entry is required at runtime.
        observation_dim: Flat observation dimensionality.
        n_primitive_actions: Number of primitive discrete actions.
        base_step_size: Step-size for the extended base Q-function.
        base_avg_reward_step_size: Average-reward rate step-size for base.
        base_trace_decay: Eligibility trace decay for the base agent.
        option_step_size: Step-size for intra-option Q-functions.
        option_avg_reward_step_size: Per-option average-reward step-size.
        option_trace_decay: Trace decay for intra-option Q-functions.
        option_gamma: Discount within option execution.
        option_model_decay: EMA decay for option outcome model updates.
        option_model_step_size: Step-size for next-state delta predictor.
        option_planning_backups_per_step: Fixed Dyna option-model backup
            budget after each real transition. ``0`` disables planning.
        epsilon_base: Exploration rate for extended action selection.
        epsilon_option: Exploration rate for intra-option action selection.
        option_target_epsilon: Optional target-policy epsilon for clipped
            intra-option importance sampling. ``None`` matches
            ``epsilon_option`` and recovers on-policy updates.
        option_importance_clip: Maximum per-decision target/behavior ratio for
            intra-option updates.
    """

    subtask_specs: tuple[SubtaskSpec, ...] = ()
    observation_dim: int = 4
    n_primitive_actions: int = 2
    base_step_size: float = 0.05
    base_avg_reward_step_size: float = 0.01
    base_trace_decay: float = 0.0
    option_step_size: float = 0.05
    option_avg_reward_step_size: float = 0.01
    option_trace_decay: float = 0.0
    option_gamma: float = 0.99
    option_model_decay: float = 0.95
    option_model_step_size: float = 0.1
    option_planning_backups_per_step: int = 0
    epsilon_base: float = 0.1
    epsilon_option: float = 0.1
    option_target_epsilon: float | None = None
    option_importance_clip: float = 10.0

    def __post_init__(self) -> None:
        """Reject illegal dimensions and scientific scalars, then canonicalize."""
        _validate_stomp_facade_config(self)

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload: dict[str, Any] = {
            "type": "Step10STOMPConfig",
            "subtask_specs": [asdict(s) for s in self.subtask_specs],
            "observation_dim": self.observation_dim,
            "n_primitive_actions": self.n_primitive_actions,
            "base_step_size": self.base_step_size,
            "base_avg_reward_step_size": self.base_avg_reward_step_size,
            "base_trace_decay": self.base_trace_decay,
            "option_step_size": self.option_step_size,
            "option_avg_reward_step_size": self.option_avg_reward_step_size,
            "option_trace_decay": self.option_trace_decay,
            "option_gamma": self.option_gamma,
            "option_model_decay": self.option_model_decay,
            "option_model_step_size": self.option_model_step_size,
            "option_planning_backups_per_step": self.option_planning_backups_per_step,
            "epsilon_base": self.epsilon_base,
            "epsilon_option": self.epsilon_option,
            "option_target_epsilon": self.option_target_epsilon,
            "option_importance_clip": self.option_importance_clip,
        }
        return payload

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> Step10STOMPConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = _require_payload(
            payload,
            name="Step10STOMPConfig payload",
            fields=_STEP10_CONFIG_FIELDS,
        )
        if type(data["type"]) is not str or data["type"] != "Step10STOMPConfig":
            raise ValueError("Step10STOMPConfig payload type must be 'Step10STOMPConfig'")
        raw_specs = data.pop("subtask_specs")
        if type(raw_specs) is not list:
            raise ValueError("subtask_specs payload must be an exact list")
        values = cast(list[object], raw_specs)
        _require_subtask_count(len(values))
        specs: list[SubtaskSpec] = []
        for index in range(len(values)):
            raw = _require_payload(
                values[index],
                name=f"subtask_specs[{index}]",
                fields=_SUBTASK_SPEC_FIELDS,
            )
            specs.append(
                SubtaskSpec(
                    feature_index=_require_int("feature_index", raw["feature_index"], minimum=0),
                    threshold=_require_positive_real("threshold", raw["threshold"]),
                    pseudo_reward_scale=_require_positive_real(
                        "pseudo_reward_scale", raw["pseudo_reward_scale"]
                    ),
                    max_option_steps=_require_int(
                        "max_option_steps", raw["max_option_steps"], minimum=1,
                        maximum=_INT32_MAX
                    ),
                )
            )
        data.pop("type")
        return cls(subtask_specs=tuple(specs), **data)

    def to_stomp_config(self) -> STOMPConfig:
        """Convert to the core :class:`STOMPConfig`."""
        return STOMPConfig(
            subtask_specs=self.subtask_specs,
            observation_dim=self.observation_dim,
            n_primitive_actions=self.n_primitive_actions,
            base_step_size=self.base_step_size,
            base_avg_reward_step_size=self.base_avg_reward_step_size,
            base_trace_decay=self.base_trace_decay,
            option_step_size=self.option_step_size,
            option_avg_reward_step_size=self.option_avg_reward_step_size,
            option_trace_decay=self.option_trace_decay,
            option_gamma=self.option_gamma,
            option_model_decay=self.option_model_decay,
            option_model_step_size=self.option_model_step_size,
            option_planning_backups_per_step=self.option_planning_backups_per_step,
            epsilon_base=self.epsilon_base,
            epsilon_option=self.epsilon_option,
            option_target_epsilon=self.option_target_epsilon,
            option_importance_clip=self.option_importance_clip,
        )


_INT32_MAX = 2**31 - 1
_MAX_SUBTASK_SPECS = 4_096
_MAX_PLANNING_BACKUPS_PER_STEP = 4_096
_STEP10_CONFIG_FIELDS = frozenset(
    {"type", "subtask_specs", *Step10STOMPConfig.__dataclass_fields__}
)
_SUBTASK_SPEC_FIELDS = frozenset(SubtaskSpec.__dataclass_fields__)
_ACTUAL_INT_TYPES: tuple[type, ...] = (int, *(np.dtype(code).type for code in "bBhHiIlLqQpP"))
_ACTUAL_REAL_TYPES = _ACTUAL_INT_TYPES + (
    float,
    Fraction,
    np.dtype("e").type,
    np.dtype("f").type,
    np.dtype("d").type,
    np.dtype("g").type,
)


def _finite_real_and_float32(name: str, value: object) -> tuple[Any, int, int, float]:
    """Validate the runtime type without invoking hooks on its metaclass."""
    actual_type = type(value)
    if not any(actual_type is allowed_type for allowed_type in _ACTUAL_REAL_TYPES):
        mro = type.__getattribute__(actual_type, "__mro__")
        has_real_lineage = actual_type is not bool and any(
            base is int or base is float or base is Fraction for base in mro
        )
        requirement = "finite" if has_real_lineage else "a real number"
        raise ValueError(f"{name} must be {requirement}")
    return finite_real_and_float32(name, value)


def _require_unit_interval(name: str, value: object) -> float:
    real, numerator, denominator, narrowed = _finite_real_and_float32(name, value)
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


def _require_nonnegative_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = _finite_real_and_float32(name, value)
    if real < 0.0 or numerator < 0 or narrowed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return canonical_float32_storage(real, narrowed)


def _require_positive_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = _finite_real_and_float32(name, value)
    if real <= 0.0 or numerator <= 0 or narrowed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return canonical_float32_storage(real, narrowed)


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    exclusive_maximum: int | None = None,
) -> int:
    actual_type = type(value)
    if not any(actual_type is allowed_type for allowed_type in _ACTUAL_INT_TYPES):
        raise ValueError(f"{name} must be an integer")
    number = int(cast(Integral, value))
    if minimum is not None and number < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive")
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative")
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    if exclusive_maximum is not None and number >= exclusive_maximum:
        raise ValueError(f"{name} must be smaller than int32 max")
    return number


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a built-in bool")
    return value


def _require_payload(
    value: object, *, name: str, fields: frozenset[str]
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an exact dictionary")
    raw = cast(dict[object, object], value)
    if any(type(key) is not str for key in raw):
        raise ValueError(f"{name} keys must be exact strings")
    if cast(set[str], set(raw)) != fields:
        raise ValueError(f"{name} fields do not match the schema")
    return cast(dict[str, Any], dict(raw))


def _require_subtask_count(count: int) -> None:
    if count > _MAX_SUBTASK_SPECS:
        raise ValueError(f"subtask_specs must contain at most {_MAX_SUBTASK_SPECS} values")


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


def _preflight_step10_smoke_resources(config: Step10STOMPConfig, steps: int) -> None:
    rows = _checked_sum("Step 10 observation row count", steps, 1)
    observations = _checked_product(
        "Step 10 observation count", rows, config.observation_dim
    )
    option_outputs = _checked_product(
        "Step 10 option output count", steps, len(config.subtask_specs)
    )
    _checked_sum(
        "Step 10 smoke array bytes",
        _checked_product("Step 10 observation bytes", 4, observations),
        _checked_product("Step 10 scalar output bytes", 21, steps),
        _checked_product("Step 10 option output bytes", 4, option_outputs),
    )


def _validate_stomp_facade_config(config: Step10STOMPConfig) -> None:
    if type(config) is not Step10STOMPConfig:
        raise TypeError("config must be an exact Step10STOMPConfig")
    # Observation dimensions and action indices flow into int32 JAX sinks;
    # bounding both keeps every feature_index (< observation_dim) in range.
    observation_dim = _require_int(
        "observation_dim",
        config.observation_dim,
        minimum=1,
        maximum=_INT32_MAX,
    )
    n_primitive_actions = _require_int(
        "n_primitive_actions",
        config.n_primitive_actions,
        minimum=1,
        maximum=_INT32_MAX,
    )
    option_planning_backups_per_step = _require_int(
        "option_planning_backups_per_step",
        config.option_planning_backups_per_step,
        minimum=0,
        maximum=_MAX_PLANNING_BACKUPS_PER_STEP,
    )
    if type(config.subtask_specs) is not tuple:
        raise ValueError("subtask_specs must be a tuple of SubtaskSpec")
    _require_subtask_count(len(config.subtask_specs))
    canonical_specs: list[SubtaskSpec] = []
    for spec in config.subtask_specs:
        if type(spec) is not SubtaskSpec:
            raise ValueError("subtask_specs must contain SubtaskSpec values")
        feature_index = _require_int("feature_index", spec.feature_index, minimum=0)
        if feature_index >= observation_dim:
            raise ValueError("feature_index must be < observation_dim")
        threshold = _require_positive_real("threshold", spec.threshold)
        pseudo_reward_scale = _require_positive_real(
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
    base_step_size = _require_nonnegative_real("base_step_size", config.base_step_size)
    base_avg_reward_step_size = _require_nonnegative_real(
        "base_avg_reward_step_size",
        config.base_avg_reward_step_size,
    )
    base_trace_decay = _require_unit_interval("base_trace_decay", config.base_trace_decay)
    option_step_size = _require_nonnegative_real(
        "option_step_size",
        config.option_step_size,
    )
    option_avg_reward_step_size = _require_nonnegative_real(
        "option_avg_reward_step_size",
        config.option_avg_reward_step_size,
    )
    option_trace_decay = _require_unit_interval(
        "option_trace_decay",
        config.option_trace_decay,
    )
    option_gamma = _require_unit_interval("option_gamma", config.option_gamma)
    option_model_decay = _require_unit_interval(
        "option_model_decay",
        config.option_model_decay,
    )
    option_model_step_size = _require_nonnegative_real(
        "option_model_step_size",
        config.option_model_step_size,
    )
    epsilon_base = _require_unit_interval("epsilon_base", config.epsilon_base)
    epsilon_option = _require_unit_interval("epsilon_option", config.epsilon_option)
    option_target_epsilon = (
        None
        if config.option_target_epsilon is None
        else _require_unit_interval("option_target_epsilon", config.option_target_epsilon)
    )
    option_importance_clip = _require_positive_real(
        "option_importance_clip",
        config.option_importance_clip,
    )
    object.__setattr__(config, "subtask_specs", tuple(canonical_specs))
    object.__setattr__(config, "observation_dim", observation_dim)
    object.__setattr__(config, "n_primitive_actions", n_primitive_actions)
    object.__setattr__(config, "base_step_size", base_step_size)
    object.__setattr__(config, "base_avg_reward_step_size", base_avg_reward_step_size)
    object.__setattr__(config, "base_trace_decay", base_trace_decay)
    object.__setattr__(config, "option_step_size", option_step_size)
    object.__setattr__(config, "option_avg_reward_step_size", option_avg_reward_step_size)
    object.__setattr__(config, "option_trace_decay", option_trace_decay)
    object.__setattr__(config, "option_gamma", option_gamma)
    object.__setattr__(config, "option_model_decay", option_model_decay)
    object.__setattr__(config, "option_model_step_size", option_model_step_size)
    object.__setattr__(
        config,
        "option_planning_backups_per_step",
        option_planning_backups_per_step,
    )
    object.__setattr__(config, "epsilon_base", epsilon_base)
    object.__setattr__(config, "epsilon_option", epsilon_option)
    object.__setattr__(config, "option_target_epsilon", option_target_epsilon)
    object.__setattr__(config, "option_importance_clip", option_importance_clip)
    config.to_stomp_config()


@dataclass(frozen=True)
class Step10SmokeResult:
    """Summary returned by :func:`run_step10_smoke`."""

    config: Step10STOMPConfig
    steps: int
    seed: int
    td_errors_shape: tuple[int, ...]
    average_rewards_shape: tuple[int, ...]
    primitive_actions_shape: tuple[int, ...]
    executing_options_shape: tuple[int, ...]
    pseudo_rewards_shape: tuple[int, ...]
    finite: bool
    option_termination_count: int
    agent_config: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self.config) is not Step10STOMPConfig:
            raise TypeError("config must be an exact Step10STOMPConfig")
        if type(self.agent_config) is not dict:
            raise TypeError("agent_config must be an exact dictionary")
        object.__setattr__(
            self, "steps", _require_int("steps", self.steps, minimum=1, maximum=_INT32_MAX)
        )
        object.__setattr__(self, "seed", require_jax_seed(self.seed, name="seed"))
        for name in (
            "td_errors_shape",
            "average_rewards_shape",
            "primitive_actions_shape",
            "executing_options_shape",
            "pseudo_rewards_shape",
        ):
            object.__setattr__(
                self,
                name,
                require_step_shape(name, getattr(self, name), steps=self.steps),
            )
        object.__setattr__(self, "finite", _require_bool("finite", self.finite))
        object.__setattr__(
            self,
            "option_termination_count",
            _require_int(
                "option_termination_count",
                self.option_termination_count,
                minimum=0,
                maximum=_INT32_MAX,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "config": self.config.to_config(),
            "steps": self.steps,
            "seed": self.seed,
            "td_errors_shape": list(self.td_errors_shape),
            "average_rewards_shape": list(self.average_rewards_shape),
            "primitive_actions_shape": list(self.primitive_actions_shape),
            "executing_options_shape": list(self.executing_options_shape),
            "pseudo_rewards_shape": list(self.pseudo_rewards_shape),
            "finite": self.finite,
            "option_termination_count": self.option_termination_count,
            "agent_config": self.agent_config,
        }


def make_step10_stomp_agent(config: Step10STOMPConfig | None = None) -> STOMPAgent:
    """Create a :class:`STOMPAgent` from a :class:`Step10STOMPConfig`.

    Args:
        config: Step 10 configuration.  Defaults to
            :class:`Step10STOMPConfig` with one default subtask if *None*.

    Returns:
        Initialized :class:`STOMPAgent`.
    """
    if config is None:
        config = Step10STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
        )
    elif type(config) is not Step10STOMPConfig:
        raise TypeError("config must be an exact Step10STOMPConfig")
    if not config.subtask_specs:
        raise ValueError("Step 10 STOMP requires at least one subtask")
    return STOMPAgent(config.to_stomp_config())


def init_step10_state(
    agent: STOMPAgent,
    *,
    key: Array,
    initial_observation: Array,
) -> STOMPState:
    """Initialize and prime the Step 10 STOMP state.

    Args:
        agent: The :class:`STOMPAgent` to initialize.
        key: JAX PRNG key.
        initial_observation: First real observation from the environment.
            Shape must match ``agent.config.observation_dim``.

    Returns:
        Primed :class:`STOMPState` with ``base_last_obs`` set.
    """
    return init_step10_state_and_action(
        agent,
        key=key,
        initial_observation=initial_observation,
    ).state


def init_step10_state_and_action(
    agent: STOMPAgent,
    *,
    key: Array,
    initial_observation: Array,
) -> STOMPStartResult:
    """Initialize Step 10 and return the first primitive action to dispatch.

    The returned action must be executed before passing its reward and next
    observation to :func:`step10_update`.
    """
    init_key, _ = jr.split(key)
    state = agent.init(init_key)
    obs = jnp.asarray(initial_observation, dtype=jnp.float32)
    return agent.start_with_action(state, obs)


def step10_update(
    agent: STOMPAgent,
    state: STOMPState,
    env_reward: Array,
    next_observation: Array,
) -> STOMPUpdateResult:
    """Run one real-time STOMP transition.

    Delegates directly to :meth:`STOMPAgent.update`.

    Args:
        agent: The STOMP agent.
        state: Current agent state.
        env_reward: Scalar environment reward.
        next_observation: Next real observation from the environment.

    Returns:
        :class:`STOMPUpdateResult` containing the new state and diagnostics.
    """
    return cast(STOMPUpdateResult, agent.update(state, env_reward, next_observation))


def _has_trusted_array_type(value: object) -> bool:
    actual_type = type(value)
    return (
        actual_type is np.ndarray
        or issubclass(
            actual_type,
            (
                jax.Array,
                jax.core.Tracer,
                jax.ShapeDtypeStruct,
                jax.core.ShapedArray,
            ),
        )
    )


def _trusted_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    """Validate static array metadata without dispatching on hostile objects."""
    if not _has_trusted_array_type(value):
        raise TypeError(f"{name} must be a trusted array")
    trusted = cast(Array, value)
    try:
        actual_shape = tuple(trusted.shape)
        actual_dtype = np.dtype(trusted.dtype)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must expose trusted shape and dtype metadata") from error
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if actual_dtype != np.dtype(dtype):
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}")
    return trusted


def run_step10_scan(
    agent: STOMPAgent,
    state: STOMPState,
    rewards: Array,
    next_observations: Array,
) -> STOMPArrayResult:
    """Run the STOMP agent over pre-collected continuing transition arrays.

    JIT-compiled via :meth:`STOMPAgent.scan` / ``jax.lax.scan``.

    Args:
        agent: The STOMP agent.
        state: Starting agent state.
        rewards: Shape ``(T,)`` float32 environment rewards.
        next_observations: Shape ``(T, obs_dim)`` float32 observations.

    Returns:
        :class:`STOMPArrayResult` with per-step diagnostics arrays.
    """
    if type(agent) is not STOMPAgent:
        raise TypeError("agent must be an exact STOMPAgent")
    if type(state) is not STOMPState:
        raise TypeError("state must be an exact STOMPState")

    if not _has_trusted_array_type(rewards):
        raise TypeError("rewards must be a trusted array")
    try:
        steps = int(rewards.shape[0])
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise TypeError("rewards must expose trusted shape metadata") from error
    if not 1 <= steps <= _INT32_MAX:
        raise ValueError("rewards must contain between 1 and signed-int32 steps")

    checked_rewards = _trusted_array("rewards", rewards, shape=(steps,), dtype=jnp.float32)
    checked_next_obs = _trusted_array(
        "next_observations",
        next_observations,
        shape=(steps, agent.config.observation_dim),
        dtype=jnp.float32,
    )
    return agent.scan(state, checked_rewards, checked_next_obs)


def run_step10_smoke(
    config: Step10STOMPConfig | None = None,
    *,
    steps: int = 64,
    seed: int = 0,
) -> Step10SmokeResult:
    """Run a deterministic Step 10 STOMP integration probe.

    Generates a random stream, runs the STOMP scan, and verifies that all
    outputs are finite and correctly shaped.

    Args:
        config: Step 10 configuration.  Defaults to one subtask on feature 0.
        steps: Number of transition steps to run.
        seed: PRNG seed for reproducibility.

    Returns:
        :class:`Step10SmokeResult` with shape/fineness summary.
    """
    steps = _require_int("steps", steps, minimum=1, maximum=_INT32_MAX)
    seed = require_jax_seed(seed, name="seed")

    cfg = config
    if cfg is None:
        cfg = Step10STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
        )
    elif type(cfg) is not Step10STOMPConfig:
        raise TypeError("config must be an exact Step10STOMPConfig")

    _preflight_step10_smoke_resources(cfg, steps)

    agent = make_step10_stomp_agent(cfg)
    obs_dim = cfg.observation_dim

    data_key, state_key = jr.split(jr.key(seed))
    observations = jr.normal(data_key, (steps + 1, obs_dim), dtype=jnp.float32)
    rewards = jnp.tanh(observations[1:, 0])

    state = init_step10_state(agent, key=state_key, initial_observation=observations[0])
    result = run_step10_scan(agent, state, rewards, observations[1:])
    result.td_errors.block_until_ready()

    finite = bool(
        jnp.all(jnp.isfinite(result.td_errors))
        & jnp.all(jnp.isfinite(result.average_rewards))
        & jnp.all(jnp.isfinite(result.pseudo_rewards))
        & jnp.all(result.primitive_actions >= 0)
        & jnp.all(result.primitive_actions < cfg.n_primitive_actions)
    )

    return Step10SmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        td_errors_shape=tuple(int(d) for d in result.td_errors.shape),
        average_rewards_shape=tuple(int(d) for d in result.average_rewards.shape),
        primitive_actions_shape=tuple(int(d) for d in result.primitive_actions.shape),
        executing_options_shape=tuple(int(d) for d in result.executing_options.shape),
        pseudo_rewards_shape=tuple(int(d) for d in result.pseudo_rewards.shape),
        finite=finite,
        option_termination_count=int(jnp.sum(result.option_terminations)),
        agent_config=agent.to_config(),
    )


__all__ = [
    "Step10SmokeResult",
    "Step10STOMPConfig",
    "init_step10_state",
    "init_step10_state_and_action",
    "make_step10_stomp_agent",
    "run_step10_scan",
    "run_step10_smoke",
    "step10_update",
]
