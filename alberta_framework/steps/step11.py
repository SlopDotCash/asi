# mypy: disable-error-code="attr-defined,call-arg"
"""Public Step 11 OaK facade.

Step 11 of the Alberta Plan introduces the OaK (Options and Knowledge)
architecture.  OaK extends the STOMP progression from Step 10 with three
additional mechanisms:

1. **Utility tracking** — online EMA utility scores for each option.
2. **Curation** — low-utility options are detected and replaced with new
   subtasks targeting higher-utility state features.
3. **Option keyboard** — a real-valued chord vector blends option Q-functions
   into a composite Q-vector over primitive actions, enabling exponentially
   many behaviours from a finite option set.

This facade exposes a minimal, stable surface over the core
:class:`~alberta_framework.core.oak.OaKAgent` implementation.

The facade rejects illegal dimensions and scientific scalars — epsilons,
gamma, decays, step sizes, and curation thresholds — before constructing the
core agent. Accepted numbers are canonicalized to builtin ints and floats;
legal endpoints stay valid.

References:
    Sutton, Bowling, & Pilarski (2022). "The Alberta Plan for AI Research."
    Sutton (RLC 2025). "The OaK Architecture: A Vision of SuperIntelligence."
    Barreto et al. (2019). "The Option Keyboard: Combining Skills in RL."
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
from alberta_framework.core.oak import (
    KeyboardChordLearnerConfig,
    KeyboardChordLearnerState,
    OaKAgent,
    OaKArrayResult,
    OaKConfig,
    OaKState,
    OaKUpdateResult,
    init_keyboard_chord_learner,
    keyboard_action,
    keyboard_q_values,
    learned_feature_subtask_specs,
    update_keyboard_chord_learner,
)
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.steps._float32_validation import (
    canonical_float32_storage,
    finite_real_and_float32,
)
from alberta_framework.steps._smoke_record_validation import require_step_shape


@dataclass(frozen=True)
class Step11OaKConfig:
    """Configuration for the public Step 11 OaK facade.

    Thin wrapper around :class:`~alberta_framework.core.oak.OaKConfig` with
    standard dict serialization consistent with Steps 1–10.

    Args:
        subtask_specs: Feature-reaching subtask definitions.
        observation_dim: Flat observation dimensionality.
        n_primitive_actions: Number of primitive discrete actions.
        base_step_size: Step-size for the extended base Q-function.
        base_avg_reward_step_size: Average-reward rate step-size for base.
        base_trace_decay: Eligibility trace decay for base agent.
        option_step_size: Step-size for intra-option Q-functions.
        option_avg_reward_step_size: Per-option average-reward step-size.
        option_trace_decay: Trace decay for intra-option Q-functions.
        option_gamma: Discount within option execution.
        option_model_decay: EMA decay for option outcome model updates.
        option_model_step_size: Step-size for next-state delta predictor.
        option_planning_backups_per_step: Fixed option-model planning backup
            budget per real transition. ``0`` disables planning.
        epsilon_base: Exploration rate for extended action selection.
        epsilon_option: Exploration rate for intra-option selection.
        utility_ema_decay: EMA decay for per-option utility tracking.
        curation_threshold: Utility threshold below which curation fires.
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
    utility_ema_decay: float = 0.99
    curation_threshold: float = 0.0

    def __post_init__(self) -> None:
        """Reject illegal dimensions and scientific scalars, then canonicalize."""
        _validate_oak_facade_config(self)

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "type": "Step11OaKConfig",
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
            "utility_ema_decay": self.utility_ema_decay,
            "curation_threshold": self.curation_threshold,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> Step11OaKConfig:
        """Reconstruct from :meth:`to_config` output."""
        data = _require_payload(
            payload,
            name="Step11OaKConfig payload",
            fields=_STEP11_CONFIG_FIELDS,
        )
        if type(data["type"]) is not str or data["type"] != "Step11OaKConfig":
            raise ValueError("Step11OaKConfig payload type must be 'Step11OaKConfig'")
        raw_specs = data.pop("subtask_specs")
        if type(specs_raw := raw_specs) is not list:
            raise ValueError("subtask_specs payload must be an exact list")
        values = cast(list[object], specs_raw)
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
                    feature_index=_require_int(
                        "feature_index", raw["feature_index"], minimum=0, maximum=_INT32_MAX
                    ),
                    threshold=_require_positive_real("threshold", raw["threshold"]),
                    pseudo_reward_scale=_require_positive_real(
                        "pseudo_reward_scale", raw["pseudo_reward_scale"]
                    ),
                    max_option_steps=_require_int(
                        "max_option_steps", raw["max_option_steps"], minimum=1, maximum=_INT32_MAX
                    ),
                )
            )
        data.pop("type")
        return cls(subtask_specs=tuple(specs), **data)

    def to_oak_config(self) -> OaKConfig:
        """Convert to the core :class:`OaKConfig`."""
        stomp = STOMPConfig(
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
        )
        return OaKConfig(
            stomp=stomp,
            utility_ema_decay=self.utility_ema_decay,
            curation_threshold=self.curation_threshold,
        )


_INT32_MAX = 2**31 - 1
_MAX_SUBTASK_SPECS = 4_096
_MAX_PLANNING_BACKUPS_PER_STEP = 4_096
_STEP11_CONFIG_FIELDS = frozenset({"type", *Step11OaKConfig.__dataclass_fields__})
_SUBTASK_SPEC_FIELDS = frozenset(SubtaskSpec.__dataclass_fields__)
_ACTUAL_INT_TYPES = (int, *(np.dtype(code).type for code in "bBhHiIlLqQpP"))
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


def _require_real(name: str, value: object) -> float:
    real, _, _, narrowed = _finite_real_and_float32(name, value)
    return canonical_float32_storage(real, narrowed)


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
    return number


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
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
        actual_dtype = jnp.dtype(trusted.dtype)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must expose trusted shape and dtype metadata") from error
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if actual_dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")
    return trusted


def _has_trusted_array_type(value: object) -> bool:
    actual_type = type(value)
    return (
        actual_type is np.ndarray
        or issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
    )


def _require_typed_key(name: str, value: object) -> Array:
    actual_type = type(value)
    if not (issubclass(actual_type, jax.Array) or issubclass(actual_type, jax.core.Tracer)):
        raise TypeError(f"{name} must be a scalar typed JAX PRNG key")
    key = cast(Array, value)
    try:
        shape = tuple(key.shape)
        words = jr.key_data(key)
        implementation = str(jr.key_impl(key))
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a scalar typed JAX PRNG key") from error
    if shape != () or words.shape != (2,) or words.dtype != jnp.uint32:
        raise TypeError(f"{name} must be a scalar typed JAX PRNG key")
    if implementation != "threefry2x32":
        raise ValueError(f"{name} must use Threefry2x32")
    return key


def _require_agent(agent: object) -> OaKAgent:
    if type(agent) is not OaKAgent:
        raise TypeError("agent must be an actual OaKAgent")
    return agent


def _require_state(state: object) -> OaKState:
    if type(state) is not OaKState:
        raise TypeError("state must be an actual OaKState")
    return state


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


def _preflight_step11_smoke_resources(config: Step11OaKConfig, steps: int) -> None:
    rows = _checked_sum("Step 11 observation row count", steps, 1)
    observations = _checked_product(
        "Step 11 observation count", rows, config.observation_dim
    )
    utility_outputs = _checked_product(
        "Step 11 utility output count", steps, len(config.subtask_specs)
    )
    # The scan exposes seven scalar 32-bit outputs, four uint32 counter words,
    # and eight boolean outputs per step, plus the option utility matrix.
    _checked_sum(
        "Step 11 smoke array bytes",
        _checked_product("Step 11 observation bytes", 4, observations),
        _checked_product("Step 11 reward bytes", 4, steps),
        _checked_product("Step 11 scalar output bytes", 52, steps),
        _checked_product("Step 11 utility output bytes", 4, utility_outputs),
    )


def _validate_oak_facade_config(config: Step11OaKConfig) -> None:
    if type(config) is not Step11OaKConfig:
        raise ValueError("config must be an actual Step11OaKConfig")
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
        feature_index = _require_int(
            "feature_index",
            spec.feature_index,
            minimum=0,
            maximum=_INT32_MAX,
        )
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
            maximum=_INT32_MAX,
        )
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
    utility_ema_decay = _require_unit_interval(
        "utility_ema_decay",
        config.utility_ema_decay,
    )
    curation_threshold = _require_nonnegative_real(
        "curation_threshold",
        config.curation_threshold,
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
    object.__setattr__(config, "utility_ema_decay", utility_ema_decay)
    object.__setattr__(config, "curation_threshold", curation_threshold)
    # Exercise the core constructor now, before the facade is accepted.  This
    # carries the exact derived action-count and direct-array byte ceilings to
    # direct records and JSON restores instead of deferring them to agent use.
    config.to_oak_config()


@dataclass(frozen=True)
class Step11SmokeResult:
    """Summary returned by :func:`run_step11_smoke`."""

    config: Step11OaKConfig
    steps: int
    seed: int
    td_errors_shape: tuple[int, ...]
    average_rewards_shape: tuple[int, ...]
    primitive_actions_shape: tuple[int, ...]
    utility_emas_shape: tuple[int, ...]
    finite: bool
    option_termination_count: int
    agent_config: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self) is not Step11SmokeResult:
            raise ValueError("result must be an actual Step11SmokeResult")
        if type(self.config) is not Step11OaKConfig:
            raise ValueError("config must be an actual Step11OaKConfig")
        object.__setattr__(
            self, "steps", _require_int("steps", self.steps, minimum=1, maximum=_INT32_MAX)
        )
        object.__setattr__(self, "seed", require_jax_seed(self.seed, name="seed"))
        for name in (
            "td_errors_shape",
            "average_rewards_shape",
            "primitive_actions_shape",
            "utility_emas_shape",
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
        if self.td_errors_shape != (self.steps,):
            raise ValueError("td_errors_shape must be exactly (steps,)")
        if self.average_rewards_shape != (self.steps,):
            raise ValueError("average_rewards_shape must be exactly (steps,)")
        if self.primitive_actions_shape != (self.steps,):
            raise ValueError("primitive_actions_shape must be exactly (steps,)")
        if self.utility_emas_shape != (self.steps, len(self.config.subtask_specs)):
            raise ValueError("utility_emas_shape must be exactly (steps, n_options)")
        if self.option_termination_count > self.steps:
            raise ValueError("option_termination_count must not exceed steps")
        if type(self.agent_config) is not dict:
            raise ValueError("agent_config must be an actual dict")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "config": self.config.to_config(),
            "steps": self.steps,
            "seed": self.seed,
            "td_errors_shape": list(self.td_errors_shape),
            "average_rewards_shape": list(self.average_rewards_shape),
            "primitive_actions_shape": list(self.primitive_actions_shape),
            "utility_emas_shape": list(self.utility_emas_shape),
            "finite": self.finite,
            "option_termination_count": self.option_termination_count,
            "agent_config": self.agent_config,
        }


def make_step11_oak_agent(config: Step11OaKConfig | None = None) -> OaKAgent:
    """Create an :class:`OaKAgent` from a :class:`Step11OaKConfig`.

    Args:
        config: Step 11 configuration.  Defaults to one subtask on feature 0.

    Returns:
        Initialized :class:`OaKAgent`.
    """
    if config is None:
        config = Step11OaKConfig(subtask_specs=(SubtaskSpec(feature_index=0),))
    elif type(config) is not Step11OaKConfig:
        raise TypeError("config must be an actual Step11OaKConfig")
    return OaKAgent(config.to_oak_config())


def init_step11_state(
    agent: OaKAgent,
    *,
    key: Array,
    initial_observation: Array,
) -> OaKState:
    """Initialise and prime the Step 11 OaK state.

    Args:
        agent: The :class:`OaKAgent` to initialise.
        key: JAX PRNG key.
        initial_observation: First real observation from the environment.

    Returns:
        Primed :class:`OaKState`.
    """
    checked_agent = _require_agent(agent)
    checked_key = _require_typed_key("key", key)
    observation = _trusted_array(
        "initial_observation",
        initial_observation,
        shape=(checked_agent.config.observation_dim,),
        dtype=jnp.float32,
    )
    init_key, start_key = jr.split(checked_key)
    del start_key
    state = checked_agent.init(init_key)
    return checked_agent.start(state, observation)


def step11_update(
    agent: OaKAgent,
    state: OaKState,
    env_reward: Array,
    next_observation: Array,
) -> OaKUpdateResult:
    """Run one real-time OaK transition.

    Args:
        agent: The OaK agent.
        state: Current agent state.
        env_reward: Scalar environment reward.
        next_observation: Next real observation.

    Returns:
        :class:`OaKUpdateResult` with new state and diagnostics.
    """
    checked_agent = _require_agent(agent)
    checked_state = _require_state(state)
    reward = _trusted_array("env_reward", env_reward, shape=(), dtype=jnp.float32)
    observation = _trusted_array(
        "next_observation",
        next_observation,
        shape=(checked_agent.config.observation_dim,),
        dtype=jnp.float32,
    )
    return checked_agent.update(checked_state, reward, observation)


def run_step11_scan(
    agent: OaKAgent,
    state: OaKState,
    rewards: Array,
    next_observations: Array,
) -> OaKArrayResult:
    """Run OaK over pre-collected continuing transition arrays.

    Args:
        agent: The OaK agent.
        state: Starting agent state.
        rewards: Shape ``(T,)`` float32 environment rewards.
        next_observations: Shape ``(T, obs_dim)`` float32 observations.

    Returns:
        :class:`OaKArrayResult` with per-step diagnostics.
    """
    checked_agent = _require_agent(agent)
    checked_state = _require_state(state)
    if not _has_trusted_array_type(rewards):
        raise TypeError("rewards must be a trusted array")
    try:
        steps = int(rewards.shape[0])
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise TypeError("rewards must expose trusted shape metadata") from error
    if not 1 <= steps <= _INT32_MAX:
        raise ValueError("rewards must contain between 1 and signed-int32 steps")
    checked_rewards = _trusted_array("rewards", rewards, shape=(steps,), dtype=jnp.float32)
    checked_observations = _trusted_array(
        "next_observations",
        next_observations,
        shape=(steps, checked_agent.config.observation_dim),
        dtype=jnp.float32,
    )
    return checked_agent.scan(checked_state, checked_rewards, checked_observations)


def run_step11_smoke(
    config: Step11OaKConfig | None = None,
    *,
    steps: int = 64,
    seed: int = 0,
) -> Step11SmokeResult:
    """Run a deterministic Step 11 OaK integration probe.

    Args:
        config: Step 11 configuration.  Defaults to one subtask on feature 0.
        steps: Number of transition steps to run.
        seed: PRNG seed for reproducibility.

    Returns:
        :class:`Step11SmokeResult` with shape/fineness summary.
    """
    steps = _require_int("steps", steps, minimum=1, maximum=_INT32_MAX)
    seed = require_jax_seed(seed, name="seed")

    cfg = config
    if cfg is None:
        cfg = Step11OaKConfig(subtask_specs=(SubtaskSpec(feature_index=0),))
    elif type(cfg) is not Step11OaKConfig:
        raise TypeError("config must be an actual Step11OaKConfig")

    _preflight_step11_smoke_resources(cfg, steps)
    agent = make_step11_oak_agent(cfg)
    obs_dim = cfg.observation_dim

    data_key, state_key = jr.split(jr.key(seed))
    observations = jr.normal(data_key, (steps + 1, obs_dim), dtype=jnp.float32)
    rewards = jnp.tanh(observations[1:, 0])

    state = init_step11_state(agent, key=state_key, initial_observation=observations[0])
    result = run_step11_scan(agent, state, rewards, observations[1:])
    result.td_errors.block_until_ready()

    finite = bool(
        jnp.all(jnp.isfinite(result.td_errors))
        & jnp.all(jnp.isfinite(result.average_rewards))
        & jnp.all(jnp.isfinite(result.pseudo_rewards))
        & jnp.all(jnp.isfinite(result.utility_emas))
        & jnp.all(result.primitive_actions >= 0)
        & jnp.all(result.primitive_actions < cfg.n_primitive_actions)
    )

    return Step11SmokeResult(
        config=cfg,
        steps=steps,
        seed=seed,
        td_errors_shape=tuple(int(d) for d in result.td_errors.shape),
        average_rewards_shape=tuple(int(d) for d in result.average_rewards.shape),
        primitive_actions_shape=tuple(int(d) for d in result.primitive_actions.shape),
        utility_emas_shape=tuple(int(d) for d in result.utility_emas.shape),
        finite=finite,
        option_termination_count=int(jnp.sum(result.option_terminations)),
        agent_config=agent.to_config(),
    )


__all__ = [
    "KeyboardChordLearnerConfig",
    "KeyboardChordLearnerState",
    "Step11OaKConfig",
    "Step11SmokeResult",
    "init_step11_state",
    "init_keyboard_chord_learner",
    "keyboard_action",
    "keyboard_q_values",
    "learned_feature_subtask_specs",
    "make_step11_oak_agent",
    "run_step11_scan",
    "run_step11_smoke",
    "step11_update",
    "update_keyboard_chord_learner",
]
