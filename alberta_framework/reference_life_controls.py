"""Development-only exact-dispatch controls for the reference-life runner.

These adapters are matched development controls, not ``reference-dev``, a
checkpoint surface, or scientific evidence.  They intentionally support only
the primitive continuing one-hot/discrete slices used by
``SwitchingTwoStateMDP`` and ``RiverSwimMDP``.  Every live state is immutable
and bound to one process-local adapter owner.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from numbers import Real
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    DifferentialSARSAState,
)
from alberta_framework.core.sarsa import SARSAAgent, SARSAConfig, SARSAState
from alberta_framework.core.types import LMSState
from alberta_framework.reference_agent import (
    MAX_DECISION_INDEX,
    REFERENCE_AGENT_MANIFEST_SCHEMA,
    AgentCapabilities,
    AgentManifest,
    ArrayValue,
    AuthorizationStatus,
    Decision,
    DecisionOwnershipError,
    DispatchAuthorization,
    DispatchStatus,
    ReferenceAgentUpdate,
    SpaceSpec,
    Transaction,
    canonical_config_sha256,
)
from alberta_framework.reference_life import REFERENCE_LIFE_PRNG_IMPLEMENTATION
from alberta_framework.streams.closed_loop import (
    RiverSwimConfig,
    RiverSwimMDP,
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
)

# All selected simulators use signed-int32 transition counters.  An accepted
# event at this capacity leaves decision ``int32.max`` armed, but no selected
# life may request another transition.
REFERENCE_CONTROL_MAX_ACCEPTED_EVENTS = int(np.iinfo(np.int32).max)
REFERENCE_CONTROL_CONFIG_SCHEMA = "asi.reference_life_control_config.preview1"
REFERENCE_CONTROL_OBSERVATION_SEMANTIC_ID = (
    "asi.reference_life_control.one_hot_observation.preview1"
)
REFERENCE_CONTROL_ACTION_SEMANTIC_ID = (
    "asi.reference_life_control.primitive_action.preview1"
)

UNIFORM_RANDOM_IMPLEMENTATION_ID = "asi.uniform_random_exact_control.preview1"
UNIFORM_RANDOM_STATE_SCHEMA = "asi.uniform_random_exact_control_state.preview1"
ANALYTIC_ORACLE_IMPLEMENTATION_ID = "asi.analytic_oracle_exact_control.preview1"
ANALYTIC_ORACLE_STATE_SCHEMA = "asi.analytic_oracle_exact_control_state.preview1"
DIFFERENTIAL_SARSA_IMPLEMENTATION_ID = "asi.differential_sarsa_exact_control.preview1"
DIFFERENTIAL_SARSA_STATE_SCHEMA = "asi.differential_sarsa_exact_control_state.preview1"
DISCOUNTED_SARSA_IMPLEMENTATION_ID = "asi.discounted_sarsa_exact_control.preview1"
DISCOUNTED_SARSA_STATE_SCHEMA = "asi.discounted_sarsa_exact_control_state.preview1"

_RIVERSWIM_REFERENCE_MAX_STATES = 12
_INT32_MAX = int(np.iinfo(np.int32).max)
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_MAX_HIDDEN_LAYERS = 32
_MAX_HIDDEN_WIDTH = 4096
_MAX_NETWORK_PARAMETERS = 1 << 20
_MAX_ORACLE_HORIZON = 100_000
_THREEFRY_IMPLEMENTATION = REFERENCE_LIFE_PRNG_IMPLEMENTATION
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ID_LENGTH = 256
_RANDOM_DOMAIN = 0x4354524C

EnvironmentKind = Literal["switching_two_state", "riverswim"]


def _require_environment_shape(
    environment_kind: EnvironmentKind,
    observation_dim: int,
    n_actions: int,
) -> None:
    if environment_kind not in ("switching_two_state", "riverswim"):
        raise ValueError("environment_kind must be switching_two_state or riverswim")
    if (
        isinstance(observation_dim, bool)
        or not isinstance(observation_dim, int)
        or observation_dim < 2
    ):
        raise ValueError("observation_dim must be an integer of at least two")
    if environment_kind == "switching_two_state" and observation_dim != 2:
        raise ValueError("SwitchingTwoState controls require observation_dim == 2")
    if environment_kind == "riverswim" and observation_dim > _RIVERSWIM_REFERENCE_MAX_STATES:
        raise ValueError(
            "RiverSwim controls require observation_dim <= "
            f"{_RIVERSWIM_REFERENCE_MAX_STATES}"
        )
    if isinstance(n_actions, bool) or not isinstance(n_actions, int) or n_actions != 2:
        raise ValueError("selected reference controls require exactly two actions")


def _nonnegative_int(value: Any, *, name: str, maximum: int = _INT32_MAX) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise ValueError(f"{name} must be a nonnegative integer within int32 capacity")
    return value


def _canonical_float32(
    value: Any,
    *,
    name: str,
    minimum: float = -_FLOAT32_MAX,
    maximum: float = _FLOAT32_MAX,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    """Return the actual executed float32 value after checking both domains."""

    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real non-boolean float32 scalar")
    try:
        raw = float(value)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            converted = float(np.float32(raw))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite float32 value") from exc
    if not math.isfinite(raw) or not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite float32 value")

    def in_bounds(candidate: float) -> bool:
        lower_ok = candidate >= minimum if minimum_inclusive else candidate > minimum
        upper_ok = candidate <= maximum if maximum_inclusive else candidate < maximum
        return lower_ok and upper_ok

    if not in_bounds(raw):
        left = "[" if minimum_inclusive else "("
        right = "]" if maximum_inclusive else ")"
        raise ValueError(
            f"{name} must remain in {left}{minimum}, {maximum}{right} as float32"
        )
    if not minimum_inclusive and raw > minimum and converted == minimum:
        raise ValueError(f"{name} must not collapse to its strict float32 lower bound")
    if not maximum_inclusive and raw < maximum and converted == maximum:
        raise ValueError(f"{name} must not round to its strict float32 upper bound")
    if not in_bounds(converted):
        left = "[" if minimum_inclusive else "("
        right = "]" if maximum_inclusive else ")"
        raise ValueError(
            f"{name} must remain in {left}{minimum}, {maximum}{right} as float32"
        )
    # Canonicalize both spellings of zero to the same manifest/config value.
    return 0.0 if converted == 0.0 else converted


def _canonical_payoff_matrix(value: Any, *, name: str) -> np.ndarray[Any, Any]:
    try:
        source = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("switching payoffs must be numeric 2x2 matrices") from exc
    if source.shape != (2, 2):
        raise ValueError("switching payoffs must be 2x2 matrices")
    result = np.empty((2, 2), dtype=np.float32)
    for index in np.ndindex(source.shape):
        result[index] = _canonical_float32(source[index], name=f"{name}{index}")
    return result


def _canonical_hidden_sizes(
    value: Any,
    *,
    observation_dim: int,
    n_actions: int,
) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("hidden_sizes must be a bounded tuple or list")
    if len(value) > _MAX_HIDDEN_LAYERS:
        raise ValueError(f"hidden_sizes must contain at most {_MAX_HIDDEN_LAYERS} layers")
    hidden_sizes = tuple(value)
    if any(
        type(size) is not int or size <= 0 or size > min(_INT32_MAX, _MAX_HIDDEN_WIDTH)
        for size in hidden_sizes
    ):
        raise ValueError(
            "hidden_sizes must contain positive int32 widths no greater than "
            f"{_MAX_HIDDEN_WIDTH}"
        )
    layer_inputs = (observation_dim, *hidden_sizes[:-1]) if hidden_sizes else ()
    parameter_count = sum(
        fan_in * fan_out + fan_out
        for fan_in, fan_out in zip(layer_inputs, hidden_sizes, strict=True)
    )
    final_width = hidden_sizes[-1] if hidden_sizes else observation_dim
    parameter_count += n_actions * (final_width + 1)
    if parameter_count > _MAX_NETWORK_PARAMETERS:
        raise ValueError(
            "discounted-SARSA network parameter resource bound exceeded: "
            f"{parameter_count} > {_MAX_NETWORK_PARAMETERS}"
        )
    return hidden_sizes


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_switching_environment(
    config: SwitchingTwoStateConfig,
) -> tuple[dict[str, Any], int]:
    if not isinstance(config, SwitchingTwoStateConfig):
        raise ValueError("switching environment config has the wrong type")
    phase_length = config.phase_length
    if (
        isinstance(phase_length, bool)
        or not isinstance(phase_length, int)
        or phase_length <= 0
        or phase_length > REFERENCE_CONTROL_MAX_ACCEPTED_EVENTS
    ):
        raise ValueError("phase_length must lie within signed-int32 capacity")
    payoffs_a = _canonical_payoff_matrix(config.payoffs_a, name="payoffs_a")
    payoffs_b = _canonical_payoff_matrix(config.payoffs_b, name="payoffs_b")
    canonical = SwitchingTwoStateConfig(  # type: ignore[call-arg]
        phase_length=phase_length,
        payoffs_a=tuple(tuple(float(value) for value in row) for row in payoffs_a),
        payoffs_b=tuple(tuple(float(value) for value in row) for row in payoffs_b),
    )
    # Retain the kernel validation as part of the same config gate.
    SwitchingTwoStateMDP(canonical)
    payload = {
        "phase_length": phase_length,
        "payoffs_a": [list(row) for row in canonical.payoffs_a],
        "payoffs_b": [list(row) for row in canonical.payoffs_b],
    }
    return payload, phase_length


def _canonical_riverswim_environment(
    config: RiverSwimConfig,
) -> dict[str, Any]:
    if not isinstance(config, RiverSwimConfig):
        raise ValueError("RiverSwim environment config has the wrong type")
    n_states = config.n_states
    if (
        isinstance(n_states, bool)
        or not isinstance(n_states, int)
        or n_states < 2
        or n_states > _RIVERSWIM_REFERENCE_MAX_STATES
    ):
        raise ValueError(
            "RiverSwim n_states must lie in "
            f"[2, {_RIVERSWIM_REFERENCE_MAX_STATES}]"
        )
    initial_state = config.initial_state
    if (
        isinstance(initial_state, bool)
        or not isinstance(initial_state, int)
        or initial_state < 0
        or initial_state >= n_states
    ):
        raise ValueError("RiverSwim initial_state is outside the configured chain")
    p_right_up = _canonical_float32(
        config.p_right_up,
        name="p_right_up",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    p_right_down = _canonical_float32(
        config.p_right_down,
        name="p_right_down",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    reward_left = _canonical_float32(config.reward_left, name="reward_left")
    reward_right = _canonical_float32(config.reward_right, name="reward_right")
    if p_right_up + p_right_down > 1.0:
        raise ValueError("RiverSwim right-transition probabilities must sum to at most one")
    canonical = RiverSwimConfig(  # type: ignore[call-arg]
        n_states=n_states,
        p_right_up=p_right_up,
        p_right_down=p_right_down,
        reward_left=reward_left,
        reward_right=reward_right,
        initial_state=initial_state,
    )
    RiverSwimMDP(canonical)
    payload = {
        "n_states": n_states,
        "p_right_up": p_right_up,
        "p_right_down": p_right_down,
        "reward_left": reward_left,
        "reward_right": reward_right,
        "initial_state": initial_state,
    }
    return payload


def _canonical_oracle_horizon(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_ORACLE_HORIZON:
        raise ValueError(
            f"analytic-oracle horizon must lie in [1, {_MAX_ORACLE_HORIZON}]"
        )
    return value


def _finite_horizon_oracle_policy(
    *,
    environment_kind: EnvironmentKind,
    environment_config: dict[str, Any],
    horizon: int,
) -> np.ndarray[Any, np.dtype[np.int32]]:
    """Solve the exact selected finite-horizon control problem backwards.

    Values are accumulated in float64 from the simulator's canonical float32
    rewards and transition tensor. ``np.argmax`` supplies the declared
    lowest-action tie break.  The resulting policy is indexed by accepted
    decision index and observed state.
    """

    horizon = _canonical_oracle_horizon(horizon)
    if environment_kind == "switching_two_state":
        phase_length = cast(int, environment_config["phase_length"])
        payoffs = np.asarray(
            (
                environment_config["payoffs_a"],
                environment_config["payoffs_b"],
            ),
            dtype=np.float32,
        ).astype(np.float64)
        value = np.zeros(2, dtype=np.float64)
        policy = np.zeros((horizon + 1, 2), dtype=np.int32)
        for index in range(horizon - 1, -1, -1):
            phase = (index // phase_length) % 2
            action_values = payoffs[phase] + value[np.newaxis, :]
            actions = np.argmax(action_values, axis=1).astype(np.int32)
            policy[index] = actions
            value = action_values[np.arange(2), actions]
    elif environment_kind == "riverswim":
        canonical = RiverSwimConfig(  # type: ignore[call-arg]
            n_states=environment_config["n_states"],
            p_right_up=environment_config["p_right_up"],
            p_right_down=environment_config["p_right_down"],
            reward_left=environment_config["reward_left"],
            reward_right=environment_config["reward_right"],
            initial_state=environment_config["initial_state"],
        )
        kernel = RiverSwimMDP(canonical)
        transitions = kernel.transition_tensor.astype(np.float64)
        rewards = kernel.reward_tensor.astype(np.float64)
        value = np.zeros(kernel.n_states, dtype=np.float64)
        policy = np.zeros((horizon + 1, kernel.n_states), dtype=np.int32)
        for index in range(horizon - 1, -1, -1):
            continuation = np.einsum("asn,n->sa", transitions, value)
            action_values = rewards + continuation
            actions = np.argmax(action_values, axis=1).astype(np.int32)
            policy[index] = actions
            value = action_values[np.arange(kernel.n_states), actions]
    else:
        raise ValueError("unsupported analytic-oracle environment")
    if not bool(np.all(np.isfinite(value))):
        raise ValueError("analytic-oracle dynamic program became non-finite")
    return policy


def _oracle_policy_sha256(policy: np.ndarray[Any, Any]) -> str:
    canonical = np.asarray(policy, dtype="<i4", order="C")
    header = f"int32le:{canonical.shape[0]}:{canonical.shape[1]}:".encode("ascii")
    return hashlib.sha256(header + canonical.tobytes(order="C")).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class UniformRandomReferenceConfig:
    """Canonical uniform-random control configuration."""

    environment_kind: EnvironmentKind
    observation_dim: int
    n_actions: int = 2

    def __post_init__(self) -> None:
        _require_environment_shape(
            self.environment_kind,
            self.observation_dim,
            self.n_actions,
        )

    @classmethod
    def for_switching(
        cls,
        environment_config: SwitchingTwoStateConfig,
    ) -> UniformRandomReferenceConfig:
        _canonical_switching_environment(environment_config)
        return cls(
            environment_kind="switching_two_state",
            observation_dim=2,
        )

    @classmethod
    def for_riverswim(
        cls,
        environment_config: RiverSwimConfig,
    ) -> UniformRandomReferenceConfig:
        payload = _canonical_riverswim_environment(environment_config)
        return cls(
            environment_kind="riverswim",
            observation_dim=cast(int, payload["n_states"]),
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "schema": REFERENCE_CONTROL_CONFIG_SCHEMA,
            "algorithm": "uniform_random",
            "environment_kind": self.environment_kind,
            "observation_dim": self.observation_dim,
            "n_actions": self.n_actions,
            "action_distribution": "uniform",
            "rng_schedule": "jax_fold_in_uint64_decision_index.preview1",
            "rng_implementation": _THREEFRY_IMPLEMENTATION,
            "initial_key_contract": "scalar_threefry2x32_typed_or_legacy.preview1",
            "boundary_mode": "continuing_unit_discount_only",
        }


@dataclasses.dataclass(frozen=True, slots=True)
class DifferentialSARSAReferenceConfig:
    """Canonical linear differential-SARSA control configuration."""

    environment_kind: EnvironmentKind
    observation_dim: int
    n_actions: int = 2
    q_step_size: float = 0.1
    average_reward_step_size: float = 0.01
    trace_decay: float = 0.0
    epsilon_start: float = 0.5
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 2500
    use_bias: bool = True

    def __post_init__(self) -> None:
        _require_environment_shape(
            self.environment_kind,
            self.observation_dim,
            self.n_actions,
        )
        object.__setattr__(
            self,
            "q_step_size",
            _canonical_float32(
                self.q_step_size,
                name="q_step_size",
                minimum=0.0,
                maximum=_FLOAT32_MAX,
            ),
        )
        object.__setattr__(
            self,
            "average_reward_step_size",
            _canonical_float32(
                self.average_reward_step_size,
                name="average_reward_step_size",
                minimum=0.0,
                maximum=_FLOAT32_MAX,
            ),
        )
        for name in ("trace_decay", "epsilon_start", "epsilon_end"):
            object.__setattr__(
                self,
                name,
                _canonical_float32(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        object.__setattr__(
            self,
            "epsilon_decay_steps",
            _nonnegative_int(self.epsilon_decay_steps, name="epsilon_decay_steps"),
        )
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end must not exceed epsilon_start")
        if not isinstance(self.use_bias, bool):
            raise ValueError("use_bias must be boolean")

    @classmethod
    def for_switching(
        cls,
        environment_config: SwitchingTwoStateConfig,
        **overrides: Any,
    ) -> DifferentialSARSAReferenceConfig:
        _canonical_switching_environment(environment_config)
        return cls(
            environment_kind="switching_two_state",
            observation_dim=2,
            **overrides,
        )

    @classmethod
    def for_riverswim(
        cls,
        environment_config: RiverSwimConfig,
        **overrides: Any,
    ) -> DifferentialSARSAReferenceConfig:
        payload = _canonical_riverswim_environment(environment_config)
        return cls(
            environment_kind="riverswim",
            observation_dim=cast(int, payload["n_states"]),
            **overrides,
        )

    def core_config(self) -> DifferentialSARSAConfig:
        return DifferentialSARSAConfig(
            n_actions=self.n_actions,
            q_step_size=self.q_step_size,
            average_reward_step_size=self.average_reward_step_size,
            trace_decay=self.trace_decay,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_steps=self.epsilon_decay_steps,
            use_bias=self.use_bias,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "schema": REFERENCE_CONTROL_CONFIG_SCHEMA,
            "algorithm": "differential_sarsa",
            "environment_kind": self.environment_kind,
            "observation_dim": self.observation_dim,
            "n_actions": self.n_actions,
            "q_step_size": self.q_step_size,
            "average_reward_step_size": self.average_reward_step_size,
            "trace_decay": self.trace_decay,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "use_bias": self.use_bias,
            "rng_schedule": "owned_differential_sarsa_key.preview1",
            "rng_implementation": _THREEFRY_IMPLEMENTATION,
            "initial_key_contract": "scalar_threefry2x32_typed_or_legacy.preview1",
            "boundary_mode": "continuing_unit_discount_only",
        }


@dataclasses.dataclass(frozen=True, slots=True)
class DiscountedSARSAReferenceConfig:
    """Canonical discounted continuing-SARSA control configuration."""

    environment_kind: EnvironmentKind
    observation_dim: int
    n_actions: int = 2
    gamma: float = 0.9
    epsilon_start: float = 0.5
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 2500
    hidden_sizes: tuple[int, ...] = ()
    step_size: float = 0.05
    sparsity: float = 0.0
    leaky_relu_slope: float = 0.01
    use_layer_norm: bool = False
    lamda: float = 0.0

    def __post_init__(self) -> None:
        _require_environment_shape(
            self.environment_kind,
            self.observation_dim,
            self.n_actions,
        )
        gamma = _canonical_float32(
            self.gamma,
            name="gamma",
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )
        object.__setattr__(self, "gamma", gamma)
        for name in ("epsilon_start", "epsilon_end", "lamda"):
            object.__setattr__(
                self,
                name,
                _canonical_float32(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        object.__setattr__(
            self,
            "sparsity",
            _canonical_float32(
                self.sparsity,
                name="sparsity",
                minimum=0.0,
                maximum=1.0,
                maximum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "step_size",
            _canonical_float32(
                self.step_size,
                name="step_size",
                minimum=0.0,
                maximum=_FLOAT32_MAX,
            ),
        )
        object.__setattr__(
            self,
            "leaky_relu_slope",
            _canonical_float32(
                self.leaky_relu_slope,
                name="leaky_relu_slope",
                minimum=0.0,
                maximum=_FLOAT32_MAX,
            ),
        )
        object.__setattr__(
            self,
            "epsilon_decay_steps",
            _nonnegative_int(self.epsilon_decay_steps, name="epsilon_decay_steps"),
        )
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end must not exceed epsilon_start")
        object.__setattr__(
            self,
            "hidden_sizes",
            _canonical_hidden_sizes(
                self.hidden_sizes,
                observation_dim=self.observation_dim,
                n_actions=self.n_actions,
            ),
        )
        if not isinstance(self.use_layer_norm, bool):
            raise ValueError("use_layer_norm must be boolean")

    @classmethod
    def for_switching(
        cls,
        environment_config: SwitchingTwoStateConfig,
        **overrides: Any,
    ) -> DiscountedSARSAReferenceConfig:
        _canonical_switching_environment(environment_config)
        return cls(
            environment_kind="switching_two_state",
            observation_dim=2,
            **overrides,
        )

    @classmethod
    def for_riverswim(
        cls,
        environment_config: RiverSwimConfig,
        **overrides: Any,
    ) -> DiscountedSARSAReferenceConfig:
        payload = _canonical_riverswim_environment(environment_config)
        return cls(
            environment_kind="riverswim",
            observation_dim=cast(int, payload["n_states"]),
            **overrides,
        )

    def core_config(self) -> SARSAConfig:
        return SARSAConfig(  # type: ignore[call-arg]
            n_actions=self.n_actions,
            gamma=self.gamma,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_steps=self.epsilon_decay_steps,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "schema": REFERENCE_CONTROL_CONFIG_SCHEMA,
            "algorithm": "discounted_sarsa",
            "environment_kind": self.environment_kind,
            "observation_dim": self.observation_dim,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "hidden_sizes": list(self.hidden_sizes),
            "step_size": self.step_size,
            "sparsity": self.sparsity,
            "leaky_relu_slope": self.leaky_relu_slope,
            "use_layer_norm": self.use_layer_norm,
            "lamda": self.lamda,
            "rng_schedule": "owned_sarsa_key.preview1",
            "rng_implementation": _THREEFRY_IMPLEMENTATION,
            "initial_key_contract": "scalar_threefry2x32_typed_or_legacy.preview1",
            "boundary_mode": "continuing_unit_discount_only",
        }


@dataclasses.dataclass(frozen=True, slots=True)
class AnalyticOracleReferenceConfig:
    """Exact environment-bound finite-horizon privileged control."""

    environment_kind: EnvironmentKind
    observation_dim: int
    n_actions: int
    horizon: int
    policy_sha256: str
    environment_config_json: str = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        _require_environment_shape(
            self.environment_kind,
            self.observation_dim,
            self.n_actions,
        )
        horizon = _canonical_oracle_horizon(self.horizon)
        if type(self.policy_sha256) is not str or _SHA256.fullmatch(
            self.policy_sha256
        ) is None:
            raise ValueError("oracle policy_sha256 must be a lowercase SHA-256 digest")
        try:
            decoded = json.loads(self.environment_config_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("oracle environment config must be canonical JSON") from exc
        if (
            not isinstance(decoded, dict)
            or _canonical_json(decoded) != self.environment_config_json
        ):
            raise ValueError("oracle environment config must use canonical JSON")
        if self.environment_kind == "switching_two_state":
            try:
                switching_source = SwitchingTwoStateConfig(  # type: ignore[call-arg]
                    phase_length=decoded["phase_length"],
                    payoffs_a=tuple(tuple(row) for row in decoded["payoffs_a"]),
                    payoffs_b=tuple(tuple(row) for row in decoded["payoffs_b"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("oracle switching config is malformed") from exc
            expected, _ = _canonical_switching_environment(
                switching_source
            )
            if set(decoded) != {"phase_length", "payoffs_a", "payoffs_b"}:
                raise ValueError("oracle switching config has unknown fields")
            if decoded != expected:
                raise ValueError("oracle switching config is not canonical")
        else:
            try:
                river_source = RiverSwimConfig(  # type: ignore[call-arg]
                    n_states=decoded["n_states"],
                    p_right_up=decoded["p_right_up"],
                    p_right_down=decoded["p_right_down"],
                    reward_left=decoded["reward_left"],
                    reward_right=decoded["reward_right"],
                    initial_state=decoded["initial_state"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("oracle RiverSwim config is malformed") from exc
            expected = _canonical_riverswim_environment(river_source)
            if set(decoded) != {
                "n_states",
                "p_right_up",
                "p_right_down",
                "reward_left",
                "reward_right",
                "initial_state",
            }:
                raise ValueError("oracle RiverSwim config has unknown fields")
            if decoded != expected:
                raise ValueError("oracle RiverSwim config is not canonical")
        policy = _finite_horizon_oracle_policy(
            environment_kind=self.environment_kind,
            environment_config=expected,
            horizon=horizon,
        )
        if policy.shape != (horizon + 1, self.observation_dim):
            raise ValueError("oracle policy has an invalid shape")
        if _oracle_policy_sha256(policy) != self.policy_sha256:
            raise ValueError("oracle policy digest does not match the exact dynamic program")

    @classmethod
    def for_switching(
        cls,
        environment_config: SwitchingTwoStateConfig,
        *,
        horizon: int,
    ) -> AnalyticOracleReferenceConfig:
        payload, _ = _canonical_switching_environment(environment_config)
        canonical_horizon = _canonical_oracle_horizon(horizon)
        policy = _finite_horizon_oracle_policy(
            environment_kind="switching_two_state",
            environment_config=payload,
            horizon=canonical_horizon,
        )
        return cls(
            environment_kind="switching_two_state",
            observation_dim=2,
            n_actions=2,
            horizon=canonical_horizon,
            policy_sha256=_oracle_policy_sha256(policy),
            environment_config_json=_canonical_json(payload),
        )

    @classmethod
    def for_riverswim(
        cls,
        environment_config: RiverSwimConfig,
        *,
        horizon: int,
    ) -> AnalyticOracleReferenceConfig:
        payload = _canonical_riverswim_environment(environment_config)
        canonical_horizon = _canonical_oracle_horizon(horizon)
        policy = _finite_horizon_oracle_policy(
            environment_kind="riverswim",
            environment_config=payload,
            horizon=canonical_horizon,
        )
        return cls(
            environment_kind="riverswim",
            observation_dim=cast(int, payload["n_states"]),
            n_actions=2,
            horizon=canonical_horizon,
            policy_sha256=_oracle_policy_sha256(policy),
            environment_config_json=_canonical_json(payload),
        )

    @property
    def environment_config(self) -> dict[str, Any]:
        decoded = json.loads(self.environment_config_json)
        assert isinstance(decoded, dict)
        return decoded

    @property
    def environment_config_sha256(self) -> str:
        return canonical_config_sha256(self.environment_config)

    def to_config(self) -> dict[str, Any]:
        return {
            "schema": REFERENCE_CONTROL_CONFIG_SCHEMA,
            "algorithm": "analytic_oracle",
            "environment_kind": self.environment_kind,
            "observation_dim": self.observation_dim,
            "n_actions": self.n_actions,
            "horizon": self.horizon,
            "policy_shape": [self.horizon + 1, self.observation_dim],
            "post_horizon_decision": "unused_lowest_action_sentinel_for_final_accept",
            "policy_sha256": self.policy_sha256,
            "solver": "finite_horizon_backward_dynamic_program.float64.tie_low.preview1",
            "environment_config": self.environment_config,
            "environment_config_sha256": self.environment_config_sha256,
            "privileged": True,
            "privileged_environment_model": True,
            "privileged_inputs": (
                ["environment_dynamics", "phase_schedule"]
                if self.environment_kind == "switching_two_state"
                else ["environment_dynamics"]
            ),
            "tie_break": "lowest_action_index",
            "rng_schedule": "none_deterministic",
            "rng_implementation": None,
            "initial_key_contract": "scalar_jax_key_ignored.preview1",
            "boundary_mode": "continuing_unit_discount_only",
        }


ControlConfig = (
    UniformRandomReferenceConfig
    | AnalyticOracleReferenceConfig
    | DifferentialSARSAReferenceConfig
    | DiscountedSARSAReferenceConfig
)
ControlAgentState = DifferentialSARSAState | SARSAState | None


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceLifeControlState:
    """Immutable process-local envelope shared by the four control adapters."""

    schema: str
    manifest_id: str
    config_sha256: str
    lifecycle_id: str
    decision_index: int
    current_observation_id: str | None
    current_observation: ArrayValue | None
    current_action: ArrayValue | None
    random_key: Array | None
    agent_state: ControlAgentState
    _owner_token: object = dataclasses.field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.schema) is not str or _SAFE_ID.fullmatch(self.schema) is None:
            raise ValueError("control state schema must be a safe identifier")
        if type(self.manifest_id) is not str or _SHA256.fullmatch(self.manifest_id) is None:
            raise ValueError("control state manifest_id must be a SHA-256 digest")
        if (
            type(self.config_sha256) is not str
            or _SHA256.fullmatch(self.config_sha256) is None
        ):
            raise ValueError("control state config_sha256 must be a SHA-256 digest")
        if (
            type(self.lifecycle_id) is not str
            or len(self.lifecycle_id) > _MAX_ID_LENGTH
            or _SAFE_ID.fullmatch(self.lifecycle_id) is None
        ):
            raise ValueError("control state lifecycle_id must be a safe identifier")
        if (
            isinstance(self.decision_index, bool)
            or not isinstance(self.decision_index, int)
            or self.decision_index < 0
            or self.decision_index > MAX_DECISION_INDEX
        ):
            raise ValueError("control state decision_index must be uint64")
        armed_fields = (
            self.current_observation_id,
            self.current_observation,
            self.current_action,
        )
        if any(value is None for value in armed_fields) and not all(
            value is None for value in armed_fields
        ):
            raise ValueError("control state decision cache must be wholly fresh or armed")
        if self.current_observation_id is not None and (
            type(self.current_observation_id) is not str
            or _SAFE_ID.fullmatch(self.current_observation_id) is None
        ):
            raise ValueError("current_observation_id must be a safe identifier")
        if self.current_observation is not None and not isinstance(
            self.current_observation, ArrayValue
        ):
            raise ValueError("current_observation must be an ArrayValue")
        if self.current_action is not None and not isinstance(self.current_action, ArrayValue):
            raise ValueError("current_action must be an ArrayValue")

    def __reduce__(self) -> Any:
        raise TypeError(
            "reference-life control state is process-local and has no checkpoint codec"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceControlResourceUsage:
    """Primitive resource counts for one in-memory control state."""

    array_leaves: int
    array_elements: int
    persistent_bytes: int
    floating_array_leaves: int


def control_state_resource_usage(
    state: ReferenceLifeControlState,
) -> ReferenceControlResourceUsage:
    """Count numeric state leaves without making a checkpoint claim."""

    if not isinstance(state, ReferenceLifeControlState):
        raise ValueError("state must be a ReferenceLifeControlState")
    arrays: list[np.ndarray[Any, Any]] = []
    for encoded in (state.current_observation, state.current_action):
        if encoded is not None:
            arrays.append(encoded.to_numpy())
    roots: tuple[Any, ...] = (state.random_key, state.agent_state)
    for root in roots:
        if root is None:
            continue
        for leaf in jax.tree.leaves(root):
            try:
                dtype = getattr(leaf, "dtype", None)
                typed_key = dtype is not None and bool(
                    jax.dtypes.issubdtype(  # type: ignore[attr-defined]
                        dtype,
                        jax.dtypes.prng_key,
                    )
                )
                value = (
                    np.asarray(jr.key_data(leaf))
                    if typed_key
                    else np.asarray(leaf)
                )
            except (TypeError, ValueError):
                continue
            if value.dtype.kind in {"b", "f", "i", "u", "c"}:
                arrays.append(value)
    return ReferenceControlResourceUsage(
        array_leaves=len(arrays),
        array_elements=sum(int(value.size) for value in arrays),
        persistent_bytes=sum(int(value.nbytes) for value in arrays),
        floating_array_leaves=sum(value.dtype.kind in {"f", "c"} for value in arrays),
    )


def _observation_spec(observation_dim: int) -> SpaceSpec:
    return SpaceSpec.box(
        shape=(observation_dim,),
        dtype="float32",
        low=(0.0,) * observation_dim,
        high=(1.0,) * observation_dim,
        semantic_id=REFERENCE_CONTROL_OBSERVATION_SEMANTIC_ID,
    )


def _action_spec(n_actions: int) -> SpaceSpec:
    return SpaceSpec.discrete(
        cardinality=n_actions,
        dtype="int32",
        semantic_id=REFERENCE_CONTROL_ACTION_SEMANTIC_ID,
    )


def _require_prng_key(
    key: Any,
    *,
    name: str,
    implementation: str | None = None,
) -> str:
    """Validate one scalar typed or legacy key and return its named PRNG impl."""

    try:
        key_shape = tuple(key.shape)
        words = np.asarray(jr.key_data(key))
        key_implementation = str(jr.key_impl(key))
        dtype = key.dtype
        typed = bool(
            jax.dtypes.issubdtype(  # type: ignore[attr-defined]
                dtype,
                jax.dtypes.prng_key,
            )
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise DecisionOwnershipError(f"{name} must be a scalar JAX PRNG key") from exc
    if (
        words.ndim != 1
        or words.dtype != np.dtype(np.uint32)
        or (typed and key_shape != ())
        or (not typed and key_shape != words.shape)
    ):
        raise DecisionOwnershipError(f"{name} must be a scalar JAX PRNG key")
    if implementation is not None and key_implementation != implementation:
        raise DecisionOwnershipError(
            f"{name} must use explicit {implementation}, got {key_implementation}"
        )
    if implementation == _THREEFRY_IMPLEMENTATION and words.shape != (2,):
        raise DecisionOwnershipError(f"{name} must use a scalar threefry2x32 key")
    return key_implementation


def _explicit_threefry_key(key: Any, *, name: str) -> Array:
    _require_prng_key(key, name=name, implementation=_THREEFRY_IMPLEMENTATION)
    words = jnp.asarray(jr.key_data(key), dtype=jnp.uint32)
    return cast(Array, jr.wrap_key_data(words, impl=_THREEFRY_IMPLEMENTATION))


def _float32_scalar(value: Any, *, name: str) -> np.float32:
    try:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            converted = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DecisionOwnershipError(f"{name} must be a finite float32 scalar") from exc
    if converted.shape != () or not bool(np.isfinite(converted)):
        raise DecisionOwnershipError(f"{name} must be a finite float32 scalar")
    return np.float32(converted)


def _tree_exactly_equal(left: Any, right: Any) -> bool:
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if cast(Any, left_structure) != right_structure or len(left_leaves) != len(
        right_leaves
    ):
        return False
    return all(
        np.asarray(left_leaf).dtype == np.asarray(right_leaf).dtype
        and np.asarray(left_leaf).shape == np.asarray(right_leaf).shape
        and np.array_equal(np.asarray(left_leaf), np.asarray(right_leaf))
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    )


def _tree_finite(tree: Any) -> bool:
    for leaf in jax.tree.leaves(tree):
        try:
            array = np.asarray(leaf)
        except (TypeError, ValueError):
            continue
        if array.dtype.kind in {"f", "c"} and not bool(np.all(np.isfinite(array))):
            return False
    return True


def _require_float32_array(value: Any, *, name: str, shape: tuple[int, ...]) -> None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise DecisionOwnershipError(f"{name} violates its float32 array contract") from exc
    if array.shape != shape or array.dtype != np.dtype(np.float32):
        raise DecisionOwnershipError(
            f"{name} must have shape {shape} and dtype float32"
        )
    if not bool(np.all(np.isfinite(array))):
        raise DecisionOwnershipError(f"{name} must contain only finite float32 values")


def _require_float32_tree(
    value: Any,
    *,
    name: str,
    expected_shapes: tuple[tuple[int, ...], ...],
) -> None:
    leaves = jax.tree.leaves(value)
    if len(leaves) != len(expected_shapes):
        raise DecisionOwnershipError(f"{name} violates its array-tree structure contract")
    for index, (leaf, shape) in enumerate(zip(leaves, expected_shapes, strict=True)):
        _require_float32_array(leaf, name=f"{name}[{index}]", shape=shape)


def _require_lms_optimizer_tree(
    value: Any,
    *,
    name: str,
    group_count: int,
    step_size: float,
) -> None:
    if type(value) is not tuple or len(value) != group_count:
        raise DecisionOwnershipError(f"{name} violates its LMS group structure")
    expected_step = np.asarray(step_size, dtype=np.float32)
    for group_index, group in enumerate(value):
        if type(group) is not tuple or len(group) != 2:
            raise DecisionOwnershipError(
                f"{name}[{group_index}] must contain weight and bias LMS states"
            )
        for state_index, optimizer_state in enumerate(group):
            if not isinstance(optimizer_state, LMSState):
                raise DecisionOwnershipError(
                    f"{name}[{group_index}][{state_index}] must be an LMSState"
                )
            observed = np.asarray(optimizer_state.step_size)
            if (
                observed.shape != ()
                or observed.dtype != np.dtype(np.float32)
                or not np.array_equal(observed, expected_step)
            ):
                raise DecisionOwnershipError(
                    f"{name}[{group_index}][{state_index}] step size differs from config"
                )


def _counter_words(index: int) -> np.ndarray[Any, np.dtype[np.uint32]]:
    return np.asarray(((index >> 32) & 0xFFFFFFFF, index & 0xFFFFFFFF), dtype=np.uint32)


class _BaseReferenceControlAdapter:
    """Shared exact-dispatch ownership and continuing-outcome machinery."""

    _agent: DifferentialSARSAAgent | SARSAAgent | None
    _agent_config_json: str | None
    _config: ControlConfig
    _frozen: bool
    _manifest: AgentManifest
    _owner_token: object
    _state_schema: str

    __slots__ = (
        "_agent",
        "_agent_config_json",
        "_config",
        "_frozen",
        "_manifest",
        "_owner_token",
        "_state_schema",
    )

    def __init__(
        self,
        config: ControlConfig,
        *,
        implementation_id: str,
        state_schema: str,
        agent: DifferentialSARSAAgent | SARSAAgent | None,
    ) -> None:
        object.__setattr__(self, "_frozen", False)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_state_schema", state_schema)
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(
            self,
            "_agent_config_json",
            None if agent is None else _canonical_json(agent.to_config()),
        )
        object.__setattr__(self, "_owner_token", object())
        object.__setattr__(
            self,
            "_manifest",
            AgentManifest.from_config(
                schema=REFERENCE_AGENT_MANIFEST_SCHEMA,
                implementation_id=implementation_id,
                state_schema=state_schema,
                config=config.to_config(),
                observation_spec=_observation_spec(config.observation_dim),
                action_spec=_action_spec(config.n_actions),
                capabilities=AgentCapabilities(dispatch_rebinding=False),
            ),
        )
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("reference-life control adapter is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("reference-life control adapter is immutable")
        object.__delattr__(self, name)

    def __reduce__(self) -> Any:
        raise TypeError("reference-life control adapter is process-local")

    @property
    def manifest(self) -> AgentManifest:
        return self._manifest

    @property
    def config(self) -> ControlConfig:
        return self._config

    @property
    def max_accepted_events(self) -> int:
        return REFERENCE_CONTROL_MAX_ACCEPTED_EVENTS

    @property
    def static_numeric_bytes(self) -> int:
        return 0

    def _base_state(
        self,
        *,
        lifecycle_id: str,
        random_key: Array | None,
        agent_state: ControlAgentState,
    ) -> ReferenceLifeControlState:
        return ReferenceLifeControlState(
            schema=self._state_schema,
            manifest_id=self.manifest.manifest_id,
            config_sha256=self.manifest.config_sha256,
            lifecycle_id=lifecycle_id,
            decision_index=0,
            current_observation_id=None,
            current_observation=None,
            current_action=None,
            random_key=random_key,
            agent_state=agent_state,
            _owner_token=self._owner_token,
        )

    def _require_bound_state(self, state: ReferenceLifeControlState) -> None:
        if not isinstance(state, ReferenceLifeControlState):
            raise DecisionOwnershipError("state must be a ReferenceLifeControlState")
        if state._owner_token is not self._owner_token:
            raise DecisionOwnershipError("control state belongs to another adapter owner")
        if state.schema != self._state_schema:
            raise DecisionOwnershipError("control state has another algorithm schema")
        if state.manifest_id != self.manifest.manifest_id:
            raise DecisionOwnershipError("control state belongs to another manifest")
        if state.config_sha256 != self.manifest.config_sha256:
            raise DecisionOwnershipError("control state belongs to another configuration")
        if state.decision_index > self.max_accepted_events:
            raise DecisionOwnershipError("control state exceeds adapter capacity")

    def _one_hot(self, value: Any) -> np.ndarray[Any, np.dtype[np.float32]]:
        try:
            encoded = self.manifest.observation_spec.encode(value)
        except (TypeError, ValueError) as exc:
            raise DecisionOwnershipError(
                "observation does not match the control codec"
            ) from exc
        observation = encoded.to_numpy()
        if (
            observation.dtype != np.dtype(np.float32)
            or observation.shape != (self.config.observation_dim,)
            or not np.all((observation == 0.0) | (observation == 1.0))
            or int(np.count_nonzero(observation)) != 1
        ):
            raise DecisionOwnershipError("control observation must be exact float32 one-hot")
        return observation

    def _action_index(self, value: Any) -> int:
        try:
            encoded = self.manifest.action_spec.encode(value)
        except (TypeError, ValueError) as exc:
            raise DecisionOwnershipError("action does not match the control codec") from exc
        action = encoded.to_numpy()
        if action.shape != () or action.dtype != np.dtype(np.int32):
            raise DecisionOwnershipError("control action must be scalar int32")
        return int(action)

    def _initial_payload(
        self,
        key: Array,
    ) -> tuple[Array | None, ControlAgentState]:
        raise NotImplementedError

    def _start_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
    ) -> tuple[Array | None, ControlAgentState, int]:
        raise NotImplementedError

    def _validate_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
    ) -> None:
        raise NotImplementedError

    def _advance_payload(
        self,
        state: ReferenceLifeControlState,
        *,
        reward: np.float32,
        discount: np.float32,
        next_observation: np.ndarray[Any, np.dtype[np.float32]],
        next_index: int,
    ) -> tuple[Array | None, ControlAgentState, int, bool]:
        raise NotImplementedError

    def init(self, key: Array, *, lifecycle_id: str) -> ReferenceLifeControlState:
        _require_prng_key(key, name="initial control key")
        if (
            type(lifecycle_id) is not str
            or len(lifecycle_id) > _MAX_ID_LENGTH
            or _SAFE_ID.fullmatch(lifecycle_id) is None
        ):
            raise ValueError("lifecycle_id must be a safe identifier")
        random_key, agent_state = self._initial_payload(key)
        return self._base_state(
            lifecycle_id=lifecycle_id,
            random_key=random_key,
            agent_state=agent_state,
        )

    def start(
        self,
        state: ReferenceLifeControlState,
        *,
        observation_id: str,
        observation: Any,
    ) -> tuple[ReferenceLifeControlState, Decision]:
        self._require_bound_state(state)
        if (
            state.decision_index != 0
            or state.current_observation_id is not None
            or state.current_observation is not None
            or state.current_action is not None
        ):
            raise DecisionOwnershipError("start requires a fresh control state")
        encoded_observation = self.manifest.observation_spec.encode(observation)
        one_hot = self._one_hot(encoded_observation)
        random_key, agent_state, action = self._start_payload(state, one_hot)
        encoded_action = self.manifest.action_spec.encode(np.asarray(action, dtype=np.int32))
        started = dataclasses.replace(
            state,
            current_observation_id=observation_id,
            current_observation=encoded_observation,
            current_action=encoded_action,
            random_key=random_key,
            agent_state=agent_state,
        )
        decision = self.current_decision(started)
        return started, decision

    def current_decision(self, state: ReferenceLifeControlState) -> Decision:
        self._require_bound_state(state)
        if (
            state.current_observation_id is None
            or state.current_observation is None
            or state.current_action is None
        ):
            raise DecisionOwnershipError("fresh control state has no armed decision")
        observation = self._one_hot(state.current_observation)
        action = self._action_index(state.current_action)
        self._validate_payload(state, observation, action)
        return self.manifest.make_decision(
            lifecycle_id=state.lifecycle_id,
            decision_index=state.decision_index,
            observation_id=state.current_observation_id,
            observation=state.current_observation,
            proposed_action=state.current_action,
            armed=True,
        )

    def validate_state(self, state: ReferenceLifeControlState) -> None:
        self.current_decision(state)

    def settle_dispatch(
        self,
        state: ReferenceLifeControlState,
        authorization: DispatchAuthorization,
    ) -> tuple[ReferenceLifeControlState, Literal[False]]:
        if not isinstance(authorization, DispatchAuthorization):
            raise DecisionOwnershipError("settlement requires a DispatchAuthorization")
        current = self.current_decision(state)
        if authorization.decision != current:
            raise DecisionOwnershipError(
                "authorization decision does not match the current control cache"
            )
        if authorization.status is not AuthorizationStatus.EXACT:
            raise DecisionOwnershipError("control adapters allow exact authorization only")
        if authorization.authorized_action != current.proposed_action:
            raise DecisionOwnershipError("authorized action does not match the control action")
        return state, False

    @staticmethod
    def _rejected(state: Any, reason: str) -> ReferenceAgentUpdate:
        return ReferenceAgentUpdate(
            state=state,
            next_decision=None,
            accepted=False,
            parameters_changed=False,
            rejection_reason=reason,
        )

    def _require_current_transaction(
        self,
        state: ReferenceLifeControlState,
        transaction: Transaction,
    ) -> tuple[np.float32, np.float32, np.ndarray[Any, np.dtype[np.float32]]]:
        if not isinstance(transaction, Transaction):
            raise DecisionOwnershipError("outcome must be a Transaction")
        current = self.current_decision(state)
        if transaction.decision != current:
            raise DecisionOwnershipError("outcome does not own the current control decision")
        authorization = transaction.receipt.dispatch.authorization
        if (
            authorization.status is not AuthorizationStatus.EXACT
            or transaction.receipt.dispatch.status is not DispatchStatus.EXACT
            or transaction.receipt.applied_action != current.proposed_action
        ):
            raise DecisionOwnershipError("outcome does not preserve exact action ownership")
        if transaction.is_boundary or transaction.autoreset:
            raise DecisionOwnershipError("control adapters support continuing outcomes only")
        if transaction.discount != 1.0:
            raise DecisionOwnershipError(
                "selected continuing controls require unit environment discount"
            )
        if (
            transaction.next_decision_observation_id is None
            or transaction.next_decision_observation is None
        ):
            raise DecisionOwnershipError("continuing outcome lacks a next observation")
        if state.decision_index >= self.max_accepted_events:
            raise DecisionOwnershipError("control adapter capacity is exhausted")
        reward = _float32_scalar(transaction.reward, name="reward")
        discount = _float32_scalar(transaction.discount, name="discount")
        next_observation = self._one_hot(transaction.next_decision_observation)
        return reward, discount, next_observation

    def apply_outcome(
        self,
        state: Any,
        transaction: Transaction,
    ) -> ReferenceAgentUpdate:
        """Stage one update, preserving the exact input state on every failure."""

        if not isinstance(state, ReferenceLifeControlState):
            return self._rejected(state, "state must be a ReferenceLifeControlState")
        try:
            reward, discount, next_observation = self._require_current_transaction(
                state,
                transaction,
            )
            assert transaction.next_decision_observation_id is not None
            assert transaction.next_decision_observation is not None
            next_index = state.decision_index + 1
            random_key, agent_state, action, parameters_changed = self._advance_payload(
                state,
                reward=reward,
                discount=discount,
                next_observation=next_observation,
                next_index=next_index,
            )
            candidate = dataclasses.replace(
                state,
                decision_index=next_index,
                current_observation_id=transaction.next_decision_observation_id,
                current_observation=transaction.next_decision_observation,
                current_action=self.manifest.action_spec.encode(
                    np.asarray(action, dtype=np.int32)
                ),
                random_key=random_key,
                agent_state=agent_state,
            )
            decision = self.current_decision(candidate)
        except Exception as exc:
            return self._rejected(state, f"control transition rejected: {exc}")
        return ReferenceAgentUpdate(
            state=candidate,
            next_decision=decision,
            accepted=True,
            parameters_changed=parameters_changed,
            rejection_reason=None,
        )


class UniformRandomReferenceAdapter(_BaseReferenceControlAdapter):
    """Owner-bound uniform random primitive-action control."""

    __slots__ = ()

    def __init__(self, config: UniformRandomReferenceConfig) -> None:
        if not isinstance(config, UniformRandomReferenceConfig):
            raise ValueError("uniform-random adapter config has the wrong type")
        super().__init__(
            config,
            implementation_id=UNIFORM_RANDOM_IMPLEMENTATION_ID,
            state_schema=UNIFORM_RANDOM_STATE_SCHEMA,
            agent=None,
        )

    @property
    def config(self) -> UniformRandomReferenceConfig:
        return cast(UniformRandomReferenceConfig, super().config)

    def _random_action(self, key: Array, decision_index: int) -> int:
        _require_prng_key(
            key,
            name="uniform-random root key",
            implementation=_THREEFRY_IMPLEMENTATION,
        )
        scheduled = jr.fold_in(key, _RANDOM_DOMAIN)
        scheduled = jr.fold_in(scheduled, (decision_index >> 32) & 0xFFFFFFFF)
        scheduled = jr.fold_in(scheduled, decision_index & 0xFFFFFFFF)
        return int(jr.randint(scheduled, (), 0, self.config.n_actions, dtype=jnp.int32))

    def _initial_payload(
        self,
        key: Array,
    ) -> tuple[Array | None, ControlAgentState]:
        return _explicit_threefry_key(key, name="uniform-random initial key"), None

    def _start_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
    ) -> tuple[Array | None, ControlAgentState, int]:
        del observation
        if state.random_key is None or state.agent_state is not None:
            raise DecisionOwnershipError("uniform-random fresh state is inconsistent")
        return state.random_key, None, self._random_action(state.random_key, 0)

    def _validate_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
    ) -> None:
        del observation
        if state.random_key is None or state.agent_state is not None:
            raise DecisionOwnershipError("uniform-random state payload is inconsistent")
        expected = self._random_action(state.random_key, state.decision_index)
        if action != expected:
            raise DecisionOwnershipError("uniform-random action cache does not match its key")

    def _advance_payload(
        self,
        state: ReferenceLifeControlState,
        *,
        reward: np.float32,
        discount: np.float32,
        next_observation: np.ndarray[Any, np.dtype[np.float32]],
        next_index: int,
    ) -> tuple[Array | None, ControlAgentState, int, bool]:
        del reward, discount, next_observation
        assert state.random_key is not None
        return state.random_key, None, self._random_action(state.random_key, next_index), False


class AnalyticOracleReferenceAdapter(_BaseReferenceControlAdapter):
    """Privileged finite-horizon dynamic-programming control."""

    _oracle_policy_bytes: bytes
    __slots__ = ("_oracle_policy_bytes",)

    def __init__(self, config: AnalyticOracleReferenceConfig) -> None:
        if not isinstance(config, AnalyticOracleReferenceConfig):
            raise ValueError("analytic-oracle adapter config has the wrong type")
        policy = _finite_horizon_oracle_policy(
            environment_kind=config.environment_kind,
            environment_config=config.environment_config,
            horizon=config.horizon,
        )
        object.__setattr__(
            self,
            "_oracle_policy_bytes",
            np.asarray(policy, dtype="<i4", order="C").tobytes(order="C"),
        )
        super().__init__(
            config,
            implementation_id=ANALYTIC_ORACLE_IMPLEMENTATION_ID,
            state_schema=ANALYTIC_ORACLE_STATE_SCHEMA,
            agent=None,
        )

    @property
    def config(self) -> AnalyticOracleReferenceConfig:
        return cast(AnalyticOracleReferenceConfig, super().config)

    @property
    def max_accepted_events(self) -> int:
        return self.config.horizon

    @property
    def static_numeric_bytes(self) -> int:
        return len(self._oracle_policy_bytes)

    def _oracle_action(
        self,
        observation: np.ndarray[Any, np.dtype[np.float32]],
        decision_index: int,
    ) -> int:
        if decision_index < 0 or decision_index > self.config.horizon:
            raise DecisionOwnershipError("analytic-oracle decision exceeds its horizon")
        state_index = int(np.argmax(observation))
        policy = np.frombuffer(self._oracle_policy_bytes, dtype="<i4").reshape(
            self.config.horizon + 1,
            self.config.observation_dim,
        )
        return int(policy[decision_index, state_index])

    def _initial_payload(
        self,
        key: Array,
    ) -> tuple[Array | None, ControlAgentState]:
        del key
        return None, None

    def _start_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
    ) -> tuple[Array | None, ControlAgentState, int]:
        if state.random_key is not None or state.agent_state is not None:
            raise DecisionOwnershipError("analytic-oracle fresh state is inconsistent")
        return None, None, self._oracle_action(observation, 0)

    def _validate_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
    ) -> None:
        if state.random_key is not None or state.agent_state is not None:
            raise DecisionOwnershipError("analytic-oracle state payload is inconsistent")
        if action != self._oracle_action(observation, state.decision_index):
            raise DecisionOwnershipError("analytic-oracle action cache is inconsistent")

    def _advance_payload(
        self,
        state: ReferenceLifeControlState,
        *,
        reward: np.float32,
        discount: np.float32,
        next_observation: np.ndarray[Any, np.dtype[np.float32]],
        next_index: int,
    ) -> tuple[Array | None, ControlAgentState, int, bool]:
        del state, reward, discount
        return None, None, self._oracle_action(next_observation, next_index), False


class DifferentialSARSAReferenceAdapter(_BaseReferenceControlAdapter):
    """Exact-dispatch wrapper for the linear differential-SARSA control."""

    __slots__ = ()

    def __init__(self, config: DifferentialSARSAReferenceConfig) -> None:
        if not isinstance(config, DifferentialSARSAReferenceConfig):
            raise ValueError("differential-SARSA adapter config has the wrong type")
        agent = DifferentialSARSAAgent(config.core_config())
        super().__init__(
            config,
            implementation_id=DIFFERENTIAL_SARSA_IMPLEMENTATION_ID,
            state_schema=DIFFERENTIAL_SARSA_STATE_SCHEMA,
            agent=agent,
        )

    @property
    def config(self) -> DifferentialSARSAReferenceConfig:
        return cast(DifferentialSARSAReferenceConfig, super().config)

    @property
    def _differential_agent(self) -> DifferentialSARSAAgent:
        agent = self._agent
        if (
            not isinstance(agent, DifferentialSARSAAgent)
            or self._agent_config_json is None
            or _canonical_json(agent.to_config()) != self._agent_config_json
        ):
            raise DecisionOwnershipError(
                "differential-SARSA core configuration was mutated"
            )
        return agent

    def _initial_payload(
        self,
        key: Array,
    ) -> tuple[Array | None, ControlAgentState]:
        explicit_key = _explicit_threefry_key(
            key,
            name="differential-SARSA initial key",
        )
        return None, self._differential_agent.init(self.config.observation_dim, explicit_key)

    def _start_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
    ) -> tuple[Array | None, ControlAgentState, int]:
        if (
            not isinstance(state.agent_state, DifferentialSARSAState)
            or state.random_key is not None
        ):
            raise DecisionOwnershipError("differential-SARSA fresh state is inconsistent")
        started, action = self._differential_agent.start(
            state.agent_state,
            jnp.asarray(observation, dtype=jnp.float32),
        )
        return None, started, int(action)

    def _validate_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
    ) -> None:
        learner = state.agent_state
        if not isinstance(learner, DifferentialSARSAState) or state.random_key is not None:
            raise DecisionOwnershipError("differential-SARSA state payload is inconsistent")
        expected_float_shapes = (
            ("q_weights", learner.q_weights, (self.config.n_actions, self.config.observation_dim)),
            ("q_bias", learner.q_bias, (self.config.n_actions,)),
            (
                "q_trace_weights",
                learner.q_trace_weights,
                (self.config.n_actions, self.config.observation_dim),
            ),
            ("q_trace_bias", learner.q_trace_bias, (self.config.n_actions,)),
            ("average_reward", learner.average_reward, ()),
            ("last_observation", learner.last_observation, (self.config.observation_dim,)),
            ("epsilon", learner.epsilon, ()),
        )
        for name, value, shape in expected_float_shapes:
            array = np.asarray(value)
            if array.shape != shape or array.dtype != np.dtype(np.float32):
                raise DecisionOwnershipError(
                    f"differential-SARSA {name} violates shape/dtype contract"
                )
        for name, value in (
            ("last_action", learner.last_action),
            ("step_count", learner.step_count),
        ):
            array = np.asarray(value)
            if array.shape != () or array.dtype != np.dtype(np.int32):
                raise DecisionOwnershipError(
                    f"differential-SARSA {name} must be scalar int32"
                )
        step_words = np.asarray(learner.step_words)
        if step_words.shape != (2,) or step_words.dtype != np.dtype(np.uint32):
            raise DecisionOwnershipError(
                "differential-SARSA step_words must have shape (2,) and dtype uint32"
            )
        _require_prng_key(
            learner.rng_key,
            name="differential-SARSA key",
            implementation=_THREEFRY_IMPLEMENTATION,
        )
        if not _tree_finite(learner):
            raise DecisionOwnershipError("differential-SARSA state must be finite")
        if int(learner.step_count) != state.decision_index:
            raise DecisionOwnershipError("differential-SARSA counter does not match decision")
        if not np.array_equal(step_words, _counter_words(state.decision_index)):
            raise DecisionOwnershipError("differential-SARSA exact counter is inconsistent")
        if not np.array_equal(np.asarray(learner.last_observation), observation):
            raise DecisionOwnershipError("differential-SARSA observation cache is inconsistent")
        if int(learner.last_action) != action:
            raise DecisionOwnershipError("differential-SARSA action cache is inconsistent")
        epsilon = float(learner.epsilon)
        if not 0.0 <= epsilon <= 1.0:
            raise DecisionOwnershipError("differential-SARSA epsilon is invalid")

    def _advance_payload(
        self,
        state: ReferenceLifeControlState,
        *,
        reward: np.float32,
        discount: np.float32,
        next_observation: np.ndarray[Any, np.dtype[np.float32]],
        next_index: int,
    ) -> tuple[Array | None, ControlAgentState, int, bool]:
        del next_index
        learner = state.agent_state
        assert isinstance(learner, DifferentialSARSAState)
        result = self._differential_agent.update(
            learner,
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(next_observation, dtype=jnp.float32),
            discount=jnp.asarray(discount, dtype=jnp.float32),
        )
        if not bool(np.asarray(result.update_applied)):
            raise DecisionOwnershipError("differential-SARSA functional update was unavailable")
        old_parameters = (learner.q_weights, learner.q_bias, learner.average_reward)
        new_parameters = (
            result.state.q_weights,
            result.state.q_bias,
            result.state.average_reward,
        )
        return (
            None,
            result.state,
            int(result.action),
            not _tree_exactly_equal(old_parameters, new_parameters),
        )


class DiscountedSARSAReferenceAdapter(_BaseReferenceControlAdapter):
    """Exact-dispatch wrapper for discounted SARSA on continuing outcomes."""

    __slots__ = ()

    def __init__(self, config: DiscountedSARSAReferenceConfig) -> None:
        if not isinstance(config, DiscountedSARSAReferenceConfig):
            raise ValueError("discounted-SARSA adapter config has the wrong type")
        agent = SARSAAgent(
            sarsa_config=config.core_config(),
            hidden_sizes=config.hidden_sizes,
            step_size=config.step_size,
            sparsity=config.sparsity,
            leaky_relu_slope=config.leaky_relu_slope,
            use_layer_norm=config.use_layer_norm,
            lamda=config.lamda,
        )
        super().__init__(
            config,
            implementation_id=DISCOUNTED_SARSA_IMPLEMENTATION_ID,
            state_schema=DISCOUNTED_SARSA_STATE_SCHEMA,
            agent=agent,
        )

    @property
    def config(self) -> DiscountedSARSAReferenceConfig:
        return cast(DiscountedSARSAReferenceConfig, super().config)

    @property
    def _sarsa_agent(self) -> SARSAAgent:
        agent = self._agent
        if (
            not isinstance(agent, SARSAAgent)
            or self._agent_config_json is None
            or _canonical_json(agent.to_config()) != self._agent_config_json
        ):
            raise DecisionOwnershipError("discounted-SARSA core configuration was mutated")
        return agent

    def _initial_payload(
        self,
        key: Array,
    ) -> tuple[Array | None, ControlAgentState]:
        explicit_key = _explicit_threefry_key(
            key,
            name="discounted-SARSA initial key",
        )
        learner = self._sarsa_agent.init(self.config.observation_dim, explicit_key)
        # The core learner exposes host-side lifecycle timers for its timed loop
        # helpers.  This exact-dispatch adapter never uses those helpers, and a
        # JIT update otherwise promotes both Python floats into persistent JAX
        # leaves on the first transition.  Canonical scalar arrays keep the
        # whole-life state shape fixed and exclude wall-clock time from state.
        inner = learner.learner_state.replace(  # type: ignore[attr-defined]
            birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
            uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
        )
        return None, learner.replace(learner_state=inner)  # type: ignore[attr-defined]

    def _start_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
    ) -> tuple[Array | None, ControlAgentState, int]:
        if not isinstance(state.agent_state, SARSAState) or state.random_key is not None:
            raise DecisionOwnershipError("discounted-SARSA fresh state is inconsistent")
        observation_array = jnp.asarray(observation, dtype=jnp.float32)
        action, key = self._sarsa_agent.select_action(
            state.agent_state,
            observation_array,
        )
        started = state.agent_state.replace(  # type: ignore[attr-defined]
            last_action=action,
            last_observation=observation_array,
            rng_key=key,
        )
        return None, started, int(action)

    def _validate_inner_learner_contract(self, learner: SARSAState) -> None:
        inner = learner.learner_state
        hidden_sizes = self.config.hidden_sizes
        input_widths = (
            (self.config.observation_dim, *hidden_sizes[:-1]) if hidden_sizes else ()
        )
        trunk_weights = inner.trunk_params.weights
        trunk_biases = inner.trunk_params.biases
        if (
            type(trunk_weights) is not tuple
            or type(trunk_biases) is not tuple
            or len(trunk_weights) != len(hidden_sizes)
            or len(trunk_biases) != len(hidden_sizes)
        ):
            raise DecisionOwnershipError(
                "discounted-SARSA trunk parameter structure violates its contract"
            )
        trunk_shapes: list[tuple[int, ...]] = []
        for index, (fan_in, fan_out) in enumerate(
            zip(input_widths, hidden_sizes, strict=True)
        ):
            weight_shape = (fan_out, fan_in)
            bias_shape = (fan_out,)
            _require_float32_array(
                trunk_weights[index],
                name=f"discounted-SARSA trunk weight {index}",
                shape=weight_shape,
            )
            _require_float32_array(
                trunk_biases[index],
                name=f"discounted-SARSA trunk bias {index}",
                shape=bias_shape,
            )
            trunk_shapes.extend((weight_shape, bias_shape))

        final_width = hidden_sizes[-1] if hidden_sizes else self.config.observation_dim
        head_weights = inner.head_params.weights
        head_biases = inner.head_params.biases
        if (
            type(head_weights) is not tuple
            or type(head_biases) is not tuple
            or len(head_weights) != self.config.n_actions
            or len(head_biases) != self.config.n_actions
        ):
            raise DecisionOwnershipError(
                "discounted-SARSA head parameter structure violates its contract"
            )
        for index in range(self.config.n_actions):
            _require_float32_array(
                head_weights[index],
                name=f"discounted-SARSA head weight {index}",
                shape=(1, final_width),
            )
            _require_float32_array(
                head_biases[index],
                name=f"discounted-SARSA head bias {index}",
                shape=(1,),
            )

        _require_lms_optimizer_tree(
            inner.trunk_optimizer_states,
            name="discounted-SARSA trunk optimizer state",
            group_count=len(hidden_sizes),
            step_size=self.config.step_size,
        )
        _require_lms_optimizer_tree(
            inner.head_optimizer_states,
            name="discounted-SARSA head optimizer state",
            group_count=self.config.n_actions,
            step_size=self.config.step_size,
        )
        _require_float32_tree(
            inner.trunk_traces,
            name="discounted-SARSA trunk traces",
            expected_shapes=tuple(trunk_shapes),
        )
        _require_float32_tree(
            inner.head_traces,
            name="discounted-SARSA head traces",
            expected_shapes=tuple(
                shape
                for _ in range(self.config.n_actions)
                for shape in ((1, final_width), (1,))
            ),
        )
        _require_float32_tree(
            inner.hidden_unit_utilities,
            name="discounted-SARSA hidden-unit utilities",
            expected_shapes=tuple((width,) for width in hidden_sizes),
        )
        if inner.normalizer_state is not None:
            raise DecisionOwnershipError(
                "discounted-SARSA state must not contain an undeclared normalizer"
            )
        for name in ("birth_timestamp", "uptime_s"):
            _require_float32_array(
                getattr(inner, name),
                name=f"discounted-SARSA {name}",
                shape=(),
            )

    def _validate_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
    ) -> None:
        learner = state.agent_state
        if not isinstance(learner, SARSAState) or state.random_key is not None:
            raise DecisionOwnershipError("discounted-SARSA state payload is inconsistent")
        self._validate_inner_learner_contract(learner)
        last_observation = np.asarray(learner.last_observation)
        last_action = np.asarray(learner.last_action)
        step_count = np.asarray(learner.step_count)
        epsilon = np.asarray(learner.epsilon)
        if (
            last_observation.shape != (self.config.observation_dim,)
            or last_observation.dtype != np.dtype(np.float32)
        ):
            raise DecisionOwnershipError("discounted-SARSA observation cache is malformed")
        if last_action.shape != () or last_action.dtype != np.dtype(np.int32):
            raise DecisionOwnershipError("discounted-SARSA action cache is malformed")
        if step_count.shape != () or step_count.dtype != np.dtype(np.int32):
            raise DecisionOwnershipError("discounted-SARSA step_count must be scalar int32")
        if epsilon.shape != () or epsilon.dtype != np.dtype(np.float32):
            raise DecisionOwnershipError("discounted-SARSA epsilon must be scalar float32")
        _require_prng_key(
            learner.rng_key,
            name="discounted-SARSA key",
            implementation=_THREEFRY_IMPLEMENTATION,
        )
        if not _tree_finite(learner):
            raise DecisionOwnershipError("discounted-SARSA state must be finite")
        if int(step_count) != state.decision_index:
            raise DecisionOwnershipError("discounted-SARSA counter does not match decision")
        inner_count = np.asarray(learner.learner_state.step_count)
        inner_words = np.asarray(learner.learner_state.step_words)
        if inner_words.shape != (2,) or inner_words.dtype != np.dtype(np.uint32):
            raise DecisionOwnershipError(
                "discounted-SARSA step_words must have shape (2,) and dtype uint32"
            )
        if (
            inner_count.shape != ()
            or inner_count.dtype != np.dtype(np.int32)
            or int(inner_count) != state.decision_index
            or not np.array_equal(inner_words, _counter_words(state.decision_index))
        ):
            raise DecisionOwnershipError("discounted-SARSA learner counter is inconsistent")
        if not np.array_equal(last_observation, observation):
            raise DecisionOwnershipError("discounted-SARSA observation cache is inconsistent")
        if int(last_action) != action:
            raise DecisionOwnershipError("discounted-SARSA action cache is inconsistent")
        if not 0.0 <= float(epsilon) <= 1.0:
            raise DecisionOwnershipError("discounted-SARSA epsilon is invalid")
        try:
            q_values = np.asarray(
                self._sarsa_agent.horde.predict(
                    learner.learner_state,
                    jnp.asarray(observation, dtype=jnp.float32),
                )[: self.config.n_actions]
            )
        except Exception as exc:
            raise DecisionOwnershipError(
                "discounted-SARSA learner structure is invalid"
            ) from exc
        if (
            q_values.shape != (self.config.n_actions,)
            or q_values.dtype != np.dtype(np.float32)
            or not np.all(np.isfinite(q_values))
        ):
            raise DecisionOwnershipError("discounted-SARSA action values are invalid")

    def _advance_payload(
        self,
        state: ReferenceLifeControlState,
        *,
        reward: np.float32,
        discount: np.float32,
        next_observation: np.ndarray[Any, np.dtype[np.float32]],
        next_index: int,
    ) -> tuple[Array | None, ControlAgentState, int, bool]:
        del discount, next_index
        learner = state.agent_state
        assert isinstance(learner, SARSAState)
        observation_array = jnp.asarray(next_observation, dtype=jnp.float32)
        next_action, key = self._sarsa_agent.select_action(learner, observation_array)
        selected = learner.replace(rng_key=key)  # type: ignore[attr-defined]
        result = self._sarsa_agent.update(
            selected,
            jnp.asarray(reward, dtype=jnp.float32),
            observation_array,
            jnp.asarray(False, dtype=jnp.bool_),
            next_action,
        )
        old_parameters = (
            learner.learner_state.trunk_params,
            learner.learner_state.head_params,
        )
        new_parameters = (
            result.state.learner_state.trunk_params,
            result.state.learner_state.head_params,
        )
        return (
            None,
            result.state,
            int(result.action),
            not _tree_exactly_equal(old_parameters, new_parameters),
        )


__all__ = [
    "ANALYTIC_ORACLE_IMPLEMENTATION_ID",
    "ANALYTIC_ORACLE_STATE_SCHEMA",
    "DIFFERENTIAL_SARSA_IMPLEMENTATION_ID",
    "DIFFERENTIAL_SARSA_STATE_SCHEMA",
    "DISCOUNTED_SARSA_IMPLEMENTATION_ID",
    "DISCOUNTED_SARSA_STATE_SCHEMA",
    "REFERENCE_CONTROL_ACTION_SEMANTIC_ID",
    "REFERENCE_CONTROL_CONFIG_SCHEMA",
    "REFERENCE_CONTROL_MAX_ACCEPTED_EVENTS",
    "REFERENCE_CONTROL_OBSERVATION_SEMANTIC_ID",
    "UNIFORM_RANDOM_IMPLEMENTATION_ID",
    "UNIFORM_RANDOM_STATE_SCHEMA",
    "AnalyticOracleReferenceAdapter",
    "AnalyticOracleReferenceConfig",
    "DifferentialSARSAReferenceAdapter",
    "DifferentialSARSAReferenceConfig",
    "DiscountedSARSAReferenceAdapter",
    "DiscountedSARSAReferenceConfig",
    "ReferenceControlResourceUsage",
    "ReferenceLifeControlState",
    "UniformRandomReferenceAdapter",
    "UniformRandomReferenceConfig",
    "control_state_resource_usage",
]
