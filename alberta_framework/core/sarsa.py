"""SARSA agent: on-policy control via Horde (Sutton & Barto Ch. 10).

Wraps ``HordeLearner`` with epsilon-greedy action selection and SARSA
target computation. Each action maps to a control demon (head) in the
Horde. The SARSA target ``r + gamma * Q(s', a')`` is computed externally
and passed as the cumulant to the Horde; the control heads' transition
discounts are zeroed via ``update_with_discounts`` so no internal
bootstrap is added on top of the external target.

The control demons carry the real ``SARSAConfig.gamma`` in their spec so
head eligibility traces decay by ``gamma * lamda`` (SARSA(lambda)); with
``lamda=0`` this reduces to one-step SARSA. Traces are reset at episode
boundaries.

Optionally, prediction demons can coexist with control demons in the
same Horde — they learn alongside the Q-heads without interference.

Reference: Sutton & Barto 2018, Section 10.1 (Episodic Semi-gradient SARSA)
"""

import dataclasses
import functools
import operator
import time
from collections.abc import Mapping
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Float, Int

from alberta_framework.core._float32_scalars import validated_float32_scalar_with_ratio
from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.multi_head_learner import (
    MULTI_HEAD_MLP_STATE_SCHEMA,
    AnyOptimizer,
    MultiHeadMLPState,
)
from alberta_framework.core.normalizers import (
    EMANormalizerState,
    Normalizer,
    WelfordNormalizerState,
)
from alberta_framework.core.optimizers import Bounder
from alberta_framework.core.types import (
    DemonType,
    GVFSpec,
    TraceMode,
    create_horde_spec,
)
from alberta_framework.core.update_safety import (
    floating_tree_is_finite as _floating_tree_is_finite,
)

# =============================================================================
# Types
# =============================================================================


_INT32_MAX = 2**31 - 1
# Matches the documented learning-loop step ceiling established for
# scan-driven array loops elsewhere in ``core`` (see
# ``learners._LEARNING_LOOP_MAX_STEPS`` and ``utils.nexting``). SARSA's
# array-based loops below have no other cap on the scanned sequence length.
_SARSA_SEQUENCE_MAX_STEPS = 10_000
_SARSA_CONFIG_FIELDS = {
    "n_actions",
    "gamma",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay_steps",
}
_SARSA_AGENT_CONFIG_FIELDS = {
    "type",
    "state_schema",
    "sarsa_config",
    "hidden_sizes",
    "optimizer",
    "bounder",
    "normalizer",
    "head_optimizer",
    "sparsity",
    "leaky_relu_slope",
    "use_layer_norm",
    "lamda",
    "prediction_demons",
    "trace_mode",
    "utility_decay",
}
_PREDICTION_DEMON_CONFIG_FIELDS = {
    "name",
    "demon_type",
    "gamma",
    "lamda",
    "cumulant_index",
    "terminal_reward",
}
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
_ACTUAL_FLOAT_TYPES = frozenset({float, *(np.dtype(code).type for code in ("e", "f", "d", "g"))})


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _require_sarsa_host_steps(name: str, value: object) -> int:
    """Reject an oversized host loop count before ``range`` materializes it.

    ``run_sarsa_episode`` and ``run_sarsa_continuing`` are Python env loops.
    Origin accepted any signed-int32 count and handed it to ``range`` — hang,
    not leftover INT32 arithmetic. Public last-fit matches the array-scan
    ceiling ``_SARSA_SEQUENCE_MAX_STEPS``.
    """
    if type(value) is not int or not 1 <= value <= _SARSA_SEQUENCE_MAX_STEPS:
        raise ValueError(
            f"{name} must be an integer in [1, {_SARSA_SEQUENCE_MAX_STEPS}]"
        )
    return value


def _require_sarsa_sequence_length(name: str, value: object) -> int:
    """Reject an oversized or malformed leading axis before it drives a scan.

    ``run_sarsa_from_arrays`` and ``run_sarsa_from_arrays_final_state`` hand
    their step arrays straight to ``jax.lax.scan`` with no bound on the
    leading (step) dimension. A hostile or mistaken caller supplying a huge
    ``num_steps`` can force JAX to trace/compile a scan of that length,
    hanging the process well before any step executes.
    """
    if not isinstance(value, jax.Array):
        raise TypeError(f"{name} must be a JAX array")
    if value.ndim < 1:
        raise ValueError(f"{name} must have a leading step axis")
    length = int(value.shape[0])
    if length < 1 or length > _SARSA_SEQUENCE_MAX_STEPS:
        raise ValueError(
            f"{name} length must be an integer in [1, {_SARSA_SEQUENCE_MAX_STEPS}]"
        )
    return length


def _require_sarsa_matching_length(name: str, value: object, *, expected: int) -> None:
    if not isinstance(value, jax.Array):
        raise TypeError(f"{name} must be a JAX array")
    if value.ndim < 1 or int(value.shape[0]) != expected:
        raise ValueError(f"{name} must share the same leading length as observations")


def _validated_config_float_with_ratio(
    name: str, value: object, **bounds: Any
) -> tuple[float, int, int]:
    """Validate only concrete built-in/NumPy host scalars before conversion."""
    if type(value) not in (_ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES):
        raise ValueError(f"{name} must be a finite real scalar")
    normalized, numerator, denominator = validated_float32_scalar_with_ratio(
        name, value, **bounds
    )
    if numerator != 0 and normalized == 0.0:
        raise ValueError(f"{name} must not narrow from an exact nonzero value to float32 zero")
    return normalized, numerator, denominator


def _validated_config_float(name: str, value: object, **bounds: Any) -> float:
    return _validated_config_float_with_ratio(name, value, **bounds)[0]


def _canonical_prediction_demon(demon: object, *, name: str) -> GVFSpec:
    """Validate one direct prediction spec without invoking hostile hooks."""
    if type(demon) is not GVFSpec:
        raise ValueError(f"{name} must be an actual GVFSpec")
    spec = demon
    if type(spec.name) is not str:
        raise ValueError(f"{name}.name must be an actual string")
    if type(spec.demon_type) is not DemonType:
        raise ValueError(f"{name}.demon_type must be an actual DemonType")
    return GVFSpec(  # type: ignore[call-arg]
        name=spec.name,
        demon_type=spec.demon_type,
        gamma=_validated_config_float(f"{name}.gamma", spec.gamma, lower=0.0, upper=1.0),
        lamda=_validated_config_float(f"{name}.lamda", spec.lamda, lower=0.0, upper=1.0),
        cumulant_index=_require_int32(
            f"{name}.cumulant_index", spec.cumulant_index, minimum=-1
        ),
        terminal_reward=_validated_config_float(
            f"{name}.terminal_reward", spec.terminal_reward
        ),
    )


def _prediction_demon_from_config(config: object, *, index: int) -> GVFSpec:
    """Decode one exact serialized prediction-demon schema."""
    name = f"prediction_demons[{index}]"
    if type(config) is not dict:
        raise ValueError(f"serialized {name} must be an actual dict")
    payload = cast(dict[object, object], config)
    if (
        any(type(key) is not str for key in payload)
        or set(payload) != _PREDICTION_DEMON_CONFIG_FIELDS
    ):
        raise ValueError(f"serialized {name} fields do not match the schema")
    if type(payload["name"]) is not str:
        raise ValueError(f"serialized {name}.name must be an actual string")
    demon_type = payload["demon_type"]
    if type(demon_type) is not str or demon_type not in {member.value for member in DemonType}:
        raise ValueError(f"serialized {name}.demon_type is unsupported")
    return GVFSpec(  # type: ignore[call-arg]
        name=payload["name"],
        demon_type=DemonType(demon_type),
        gamma=_validated_config_float(
            f"serialized {name}.gamma", payload["gamma"], lower=0.0, upper=1.0
        ),
        lamda=_validated_config_float(
            f"serialized {name}.lamda", payload["lamda"], lower=0.0, upper=1.0
        ),
        cumulant_index=_require_int32(
            f"serialized {name}.cumulant_index", payload["cumulant_index"], minimum=-1
        ),
        terminal_reward=_validated_config_float(
            f"serialized {name}.terminal_reward", payload["terminal_reward"]
        ),
    )


def _preflight_sarsa_direct_state(
    n_actions: int,
    n_heads: int,
    hidden_sizes: tuple[int, ...],
    feature_dim: int,
) -> None:
    """Bound the Horde arrays plus SARSA-owned state before JAX allocation."""
    layer_sizes = (feature_dim, *hidden_sizes)
    trunk_parameters = sum(
        fan_out * (fan_in + 1)
        for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:], strict=False)
    )
    final_width = hidden_sizes[-1] if hidden_sizes else feature_dim
    head_parameters = n_heads * (final_width + 1)
    horde_direct_scalars = 2 * (trunk_parameters + head_parameters) + sum(hidden_sizes) + 3
    # last_observation, last_action, epsilon, step_count, and the two-word
    # Threefry key are all four-byte public-state leaves.
    aggregate_scalars = horde_direct_scalars + feature_dim + 5
    persist_bytes = 4 * aggregate_scalars
    for name, value in (
        ("aggregate_direct_state_scalars", aggregate_scalars),
        ("aggregate_direct_state_bytes", persist_bytes),
    ):
        if not 1 <= value <= _INT32_MAX:
            raise ValueError(f"derived SARSA {name} must be at most {_INT32_MAX}")
    # Source persist, proposed persist, and returned action/Q/td/reward extras.
    # Prediction heads are persisted but update() returns Q values only for
    # the control-action prefix.
    update_working_set_bytes = 2 * persist_bytes + 12 + 4 * n_actions
    if update_working_set_bytes > _INT32_MAX:
        raise ValueError("SARSA update working set byte count must fit signed int32")


@chex.dataclass(frozen=True)
class SARSAConfig:
    """Configuration for SARSA agent.

    Attributes:
        n_actions: Number of discrete actions
        gamma: Discount factor for SARSA targets (default: 0.99)
        epsilon_start: Initial exploration rate (default: 0.1)
        epsilon_end: Final exploration rate (default: 0.01)
        epsilon_decay_steps: Steps over which epsilon decays linearly.
            0 = no decay (constant epsilon_start).
    """

    n_actions: int
    gamma: float = 0.99
    epsilon_start: float = 0.1
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 0

    def __post_init__(self) -> None:
        """Validate and canonicalize host configuration before JAX use."""
        n_actions = _require_int32("n_actions", self.n_actions, minimum=1)
        epsilon_decay_steps = _require_int32(
            "epsilon_decay_steps", self.epsilon_decay_steps, minimum=0
        )
        gamma = _validated_config_float("gamma", self.gamma, lower=0.0, upper=1.0)
        epsilon_start, start_numerator, start_denominator = _validated_config_float_with_ratio(
            "epsilon_start", self.epsilon_start, lower=0.0, upper=1.0
        )
        epsilon_end, end_numerator, end_denominator = _validated_config_float_with_ratio(
            "epsilon_end", self.epsilon_end, lower=0.0, upper=1.0
        )
        if (
            epsilon_decay_steps > 0
            and end_numerator * start_denominator > start_numerator * end_denominator
        ):
            raise ValueError("epsilon_end must not exceed epsilon_start when decaying")
        object.__setattr__(self, "n_actions", n_actions)
        object.__setattr__(self, "epsilon_decay_steps", epsilon_decay_steps)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "epsilon_start", epsilon_start)
        object.__setattr__(self, "epsilon_end", epsilon_end)

    def to_config(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SARSAConfig":
        """Reconstruct from config dict."""
        if type(config) is not dict:
            raise ValueError("SARSA config must be an actual dict")
        if not all(type(key) is str for key in config) or set(config) != _SARSA_CONFIG_FIELDS:
            raise ValueError("SARSA config fields do not match the compatibility schema")
        for name, value in config.items():
            expected_type = int if name in {"n_actions", "epsilon_decay_steps"} else float
            if type(value) is not expected_type:
                raise ValueError("serialized SARSA config values must use exact JSON scalar types")
        return cls(**config)


@chex.dataclass(frozen=True)
class SARSAState:
    """State for the SARSA agent.

    Attributes:
        learner_state: Underlying Horde/MultiHeadMLPLearner state
        last_action: Action taken at previous step (a_t)
        last_observation: Observation at previous step (s_t)
        epsilon: Current exploration rate
        rng_key: JAX random key for action selection
        step_count: Number of SARSA update steps taken
    """

    learner_state: MultiHeadMLPState
    last_action: Int[Array, ""]
    last_observation: Float[Array, " feature_dim"]
    epsilon: Float[Array, ""]
    rng_key: Array
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class SARSAUpdateResult:
    """Result of a single SARSA update step.

    Attributes:
        state: Updated SARSA state (includes new action a_{t+1})
        action: Next action a_{t+1} selected for the new state
        q_values: Q-values for all actions at s_{t+1}
        td_error: TD error for the taken action
        reward: Reward received
    """

    state: SARSAState
    action: Int[Array, ""]
    q_values: Float[Array, " n_actions"]
    td_error: Float[Array, ""]
    reward: Float[Array, ""]


@dataclasses.dataclass(frozen=True)
class SARSAEpisodeResult:
    """Result from running one episode of SARSA.

    Not a chex dataclass — used in Python loops with native Python types.

    Attributes:
        state: Final SARSA state
        total_reward: Sum of rewards in the episode
        num_steps: Number of steps taken
        rewards: Per-step rewards
        q_values: Per-step Q-values
        td_errors: Per-step TD errors
    """

    state: SARSAState
    total_reward: float
    num_steps: int
    rewards: list[float]
    q_values: list[Array]
    td_errors: list[float]


@dataclasses.dataclass(frozen=True)
class SARSAContinuingResult:
    """Result from running SARSA in continuing mode.

    Not a chex dataclass — used in Python loops with native Python types.

    Attributes:
        state: Final SARSA state
        total_reward: Sum of rewards over all steps
        rewards: Per-step rewards
        q_values: Per-step Q-values
        td_errors: Per-step TD errors
    """

    state: SARSAState
    total_reward: float
    rewards: list[float]
    q_values: list[Array]
    td_errors: list[float]


@chex.dataclass(frozen=True)
class SARSAArrayResult:
    """Result from scan-based SARSA on pre-collected arrays.

    Attributes:
        state: Final SARSA state
        q_values: Per-step Q-values, shape ``(num_steps, n_actions)``
        td_errors: Per-step TD errors, shape ``(num_steps,)``
        actions: Per-step actions taken, shape ``(num_steps,)``
    """

    state: SARSAState
    q_values: Float[Array, "num_steps n_actions"]
    td_errors: Float[Array, " num_steps"]
    actions: Int[Array, " num_steps"]


# =============================================================================
# SARSAAgent
# =============================================================================


def _make_control_demons(
    n_actions: int,
    gamma: float,
    lamda: float = 0.0,
) -> list[GVFSpec]:
    """Create n_actions control demons (SARSA targets computed externally).

    The demon ``gamma`` carries the real discount so each head's
    eligibility trace decays by ``gamma * lamda`` (SARSA(lambda)).
    The TD target does not bootstrap internally: ``SARSAAgent.update``
    zeroes the control heads' transition discounts via
    ``update_with_discounts``, so the externally computed SARSA target
    is the full target.

    Args:
        n_actions: Number of discrete actions
        gamma: Real discount factor (drives head trace decay only)
        lamda: Trace decay for head eligibility traces

    Returns:
        List of GVFSpec for control demons
    """
    return [
        GVFSpec(  # type: ignore[call-arg]
            name=f"q_{i}",
            demon_type=DemonType.CONTROL,
            gamma=gamma,
            lamda=lamda,
            cumulant_index=-1,  # external cumulant (SARSA target)
        )
        for i in range(n_actions)
    ]


class SARSAAgent:
    """On-policy SARSA control agent via Horde architecture.

    Wraps ``HordeLearner`` with epsilon-greedy action selection and
    SARSA target computation. Each action maps to a control demon (head)
    in the Horde. The SARSA target ``r + gamma * Q(s', a')`` is computed
    externally and passed as the cumulant; the control heads' transition
    discounts are zeroed at update time, while their eligibility traces
    decay by the real ``gamma * lamda`` (SARSA(lambda)).

    Optionally, additional prediction demons can coexist with the control
    demons — they learn alongside the Q-heads.

    Single-Step (Daemon) Usage
    --------------------------
    Both ``select_action()`` and ``update()`` work with single unbatched
    observations (1D arrays). JIT-compiled automatically.

    Attributes:
        sarsa_config: SARSA configuration
        horde: The underlying HordeLearner
        n_actions: Number of discrete actions
    """

    def __init__(
        self,
        sarsa_config: SARSAConfig,
        hidden_sizes: tuple[int, ...] = (128, 128),
        optimizer: AnyOptimizer | None = None,
        step_size: float = 1.0,
        bounder: Bounder | None = None,
        normalizer: (
            Normalizer[EMANormalizerState] | Normalizer[WelfordNormalizerState] | None
        ) = None,
        sparsity: float = 0.9,
        leaky_relu_slope: float = 0.01,
        use_layer_norm: bool = True,
        head_optimizer: AnyOptimizer | None = None,
        prediction_demons: list[GVFSpec] | None = None,
        lamda: float = 0.0,
        trace_mode: TraceMode = TraceMode.ACCUMULATING,
        utility_decay: float = 0.99,
    ):
        """Initialize the SARSA agent.

        Args:
            sarsa_config: SARSA configuration (n_actions, gamma, epsilon)
            hidden_sizes: Tuple of hidden layer sizes (default: two layers of 128)
            optimizer: Optimizer for weight updates. Defaults to LMS(step_size).
            step_size: Base learning rate (used only when optimizer is None)
            bounder: Optional update bounder (e.g. ObGDBounding)
            normalizer: Optional feature normalizer
            sparsity: Fraction of weights zeroed out per neuron (default: 0.9)
            leaky_relu_slope: Negative slope for LeakyReLU (default: 0.01)
            use_layer_norm: Whether to apply parameterless layer normalization
            head_optimizer: Optional separate optimizer for heads
            prediction_demons: Optional additional prediction demons to
                learn alongside Q-heads. These are appended after the
                control demons in the Horde.
            lamda: Trace decay for control demon heads (default: 0.0)
            trace_mode: Eligibility trace mode (ACCUMULATING or REPLACING)
            utility_decay: EMA decay for hidden-unit utility diagnostics.
        """
        if type(sarsa_config) is not SARSAConfig:
            raise ValueError("sarsa_config must be an actual SARSAConfig")
        lamda = _validated_config_float("lamda", lamda, lower=0.0, upper=1.0)
        step_size = _validated_config_float("step_size", step_size, lower=0.0)
        sparsity = _validated_config_float("sparsity", sparsity, lower=0.0, upper=1.0)
        leaky_relu_slope = _validated_config_float("leaky_relu_slope", leaky_relu_slope, lower=0.0)
        utility_decay = _validated_config_float(
            "utility_decay",
            utility_decay,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        if type(use_layer_norm) is not bool:
            raise ValueError("use_layer_norm must be an actual bool")
        if type(trace_mode) is not TraceMode:
            raise ValueError("trace_mode must be an actual TraceMode")
        if prediction_demons is not None and type(prediction_demons) is not list:
            raise ValueError("prediction_demons must be an actual list or None")
        canonical_predictions = (
            [
                _canonical_prediction_demon(demon, name=f"prediction_demons[{index}]")
                for index, demon in enumerate(prediction_demons)
            ]
            if prediction_demons is not None
            else None
        )
        n_predictions = len(canonical_predictions) if canonical_predictions is not None else 0
        total_heads = _require_int32(
            "total control and prediction demons",
            sarsa_config.n_actions + n_predictions,
            minimum=1,
        )
        # MultiHeadMLPLearner canonicalizes the tuple and its elements. This
        # minimum-dimension preflight happens before constructing one Python
        # GVF object per action, so an impossible state cannot first exhaust
        # host memory in `_make_control_demons`.
        if type(hidden_sizes) is not tuple:
            raise ValueError("hidden_sizes must be an actual tuple")
        canonical_hidden = tuple(
            _require_int32(f"hidden_sizes[{index}]", width, minimum=1)
            for index, width in enumerate(hidden_sizes)
        )
        _preflight_sarsa_direct_state(
            sarsa_config.n_actions,
            total_heads,
            canonical_hidden,
            1,
        )

        self._sarsa_config = sarsa_config
        self._hidden_sizes = canonical_hidden
        self._lamda = lamda

        # Build HordeSpec: control demons first, then prediction demons
        control_demons = _make_control_demons(
            sarsa_config.n_actions, gamma=sarsa_config.gamma, lamda=lamda
        )
        all_demons: list[GVFSpec] = list(control_demons)
        if canonical_predictions is not None:
            all_demons.extend(canonical_predictions)
        self._n_prediction_demons = n_predictions

        horde_spec = create_horde_spec(all_demons)

        self._horde = HordeLearner(
            horde_spec=horde_spec,
            hidden_sizes=canonical_hidden,
            optimizer=optimizer,
            step_size=step_size,
            bounder=bounder,
            normalizer=normalizer,
            sparsity=sparsity,
            leaky_relu_slope=leaky_relu_slope,
            use_layer_norm=use_layer_norm,
            head_optimizer=head_optimizer,
            trace_mode=trace_mode,
            utility_decay=utility_decay,
        )

    @property
    def sarsa_config(self) -> SARSAConfig:
        """The SARSA configuration."""
        return self._sarsa_config

    @property
    def horde(self) -> HordeLearner:
        """The underlying HordeLearner."""
        return self._horde

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return self._sarsa_config.n_actions

    def to_config(self) -> dict[str, Any]:
        """Serialize agent configuration to dict."""
        horde_config = self._horde.to_config()
        # Remove fields managed by SARSAAgent
        horde_config.pop("type", None)
        horde_config.pop("horde_spec", None)

        # Extract prediction demon specs if any
        pred_demons = None
        if self._n_prediction_demons > 0:
            all_demons = self._horde.horde_spec.demons
            pred_demons = [d.to_config() for d in all_demons[self._sarsa_config.n_actions :]]

        return {
            "type": "SARSAAgent",
            "sarsa_config": self._sarsa_config.to_config(),
            "lamda": self._lamda,
            "prediction_demons": pred_demons,
            **horde_config,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SARSAAgent":
        """Reconstruct from config dict."""
        from alberta_framework.core.normalizers import normalizer_from_config
        from alberta_framework.core.optimizers import (
            bounder_from_config,
            optimizer_from_config,
        )

        if type(config) is not dict:
            raise ValueError("SARSA agent config must be an actual dict")
        if not all(type(key) is str for key in config) or set(config) != _SARSA_AGENT_CONFIG_FIELDS:
            raise ValueError("SARSA agent config fields do not match the schema")
        config = dict(config)
        serialized_type = config.pop("type")
        if type(serialized_type) is not str or serialized_type != "SARSAAgent":
            raise ValueError("unexpected SARSA agent config type")
        state_schema = config.pop("state_schema")
        if type(state_schema) is not str or state_schema != MULTI_HEAD_MLP_STATE_SCHEMA:
            raise ValueError("unsupported SARSA Horde state schema")

        if type(config["sarsa_config"]) is not dict:
            raise ValueError("serialized sarsa_config must be an actual dict")
        if type(config["hidden_sizes"]) is not list:
            raise ValueError("serialized hidden_sizes must be an actual list")
        if (
            config["prediction_demons"] is not None
            and type(config["prediction_demons"]) is not list
        ):
            raise ValueError("serialized prediction_demons must be an actual list or None")
        if type(config["trace_mode"]) is not str:
            raise ValueError("serialized trace_mode must be an actual string")
        if any(type(width) is not int for width in config["hidden_sizes"]):
            raise ValueError("serialized hidden_sizes entries must be actual integers")
        for name in ("sparsity", "leaky_relu_slope", "lamda", "utility_decay"):
            if type(config[name]) is not float:
                raise ValueError("serialized SARSA float fields must be actual floats")
        if type(config["use_layer_norm"]) is not bool:
            raise ValueError("serialized use_layer_norm must be an actual bool")

        sarsa_config = SARSAConfig.from_config(config.pop("sarsa_config"))
        optimizer = optimizer_from_config(config.pop("optimizer"))
        bounder_cfg = config.pop("bounder", None)
        bounder = bounder_from_config(bounder_cfg) if bounder_cfg is not None else None
        normalizer_cfg = config.pop("normalizer", None)
        normalizer = normalizer_from_config(normalizer_cfg) if normalizer_cfg is not None else None
        head_opt_cfg = config.pop("head_optimizer", None)
        head_optimizer = optimizer_from_config(head_opt_cfg) if head_opt_cfg is not None else None
        pred_demons_cfg = config.pop("prediction_demons", None)
        prediction_demons = None
        if pred_demons_cfg is not None:
            prediction_demons = [
                _prediction_demon_from_config(demon, index=index)
                for index, demon in enumerate(pred_demons_cfg)
            ]

        trace_mode_str = config.pop("trace_mode", None)
        trace_mode = (
            TraceMode(trace_mode_str) if trace_mode_str is not None else TraceMode.ACCUMULATING
        )

        return cls(
            sarsa_config=sarsa_config,
            hidden_sizes=tuple(config.pop("hidden_sizes")),
            optimizer=optimizer,
            bounder=bounder,
            normalizer=normalizer,
            head_optimizer=head_optimizer,
            prediction_demons=prediction_demons,
            trace_mode=trace_mode,
            **config,
        )

    def _require_state_contract(self, state: SARSAState) -> int:
        """Validate SARSA-owned static state leaves and return feature width."""

        if type(state) is not SARSAState:
            raise TypeError("state must be an actual SARSAState")
        if type(state.learner_state) is not MultiHeadMLPState:
            raise TypeError("learner_state must be an actual MultiHeadMLPState")
        last_observation = jnp.asarray(state.last_observation)
        if last_observation.ndim != 1 or last_observation.shape[0] < 1:
            raise ValueError("SARSA last_observation must be a nonempty vector")
        if last_observation.dtype != jnp.dtype(jnp.float32):
            raise TypeError("SARSA last_observation must have dtype float32")
        for name, value, dtype in (
            ("last_action", state.last_action, jnp.int32),
            ("epsilon", state.epsilon, jnp.float32),
            ("step_count", state.step_count, jnp.int32),
        ):
            array = jnp.asarray(value)
            if array.shape != ():
                raise ValueError(f"SARSA {name} must be scalar")
            if array.dtype != jnp.dtype(dtype):
                raise TypeError(f"SARSA {name} has the wrong dtype")
        key_words = jnp.asarray(jr.key_data(state.rng_key))
        if key_words.shape != (2,) or key_words.dtype != jnp.dtype(jnp.uint32):
            raise TypeError("SARSA rng_key must be a scalar Threefry key")
        return int(last_observation.shape[0])

    def _state_values_valid(self, state: SARSAState) -> Array:
        """Return dynamic validity for the complete source transaction."""

        return (
            jnp.all(jnp.isfinite(state.last_observation))
            & jnp.isfinite(state.epsilon)
            & (state.epsilon >= 0.0)
            & (state.epsilon <= 1.0)
            & (state.last_action >= -1)
            & (state.last_action < self.n_actions)
            & (state.step_count >= 0)
        )

    def init(self, feature_dim: int, key: Array) -> SARSAState:
        """Initialize SARSA agent state.

        Args:
            feature_dim: Dimension of the input feature vector
            key: JAX random key

        Returns:
            Initial SARSAState with zeroed last_action/observation
        """
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        _preflight_sarsa_direct_state(
            self.n_actions,
            self._horde.n_demons,
            self._hidden_sizes,
            feature_dim,
        )
        key, subkey = jr.split(key)
        learner_state = self._horde.init(feature_dim, subkey)

        return SARSAState(  # type: ignore[call-arg]
            learner_state=learner_state,
            last_action=jnp.array(-1, dtype=jnp.int32),
            last_observation=jnp.zeros(feature_dim, dtype=jnp.float32),
            epsilon=jnp.array(self._sarsa_config.epsilon_start, dtype=jnp.float32),
            rng_key=key,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        state: SARSAState,
        observation: Array,
    ) -> tuple[Int[Array, ""], Array]:
        """Select action via epsilon-greedy over Q-values.

        JIT-compiled. Uses Gumbel trick for uniform tie-breaking among
        equal Q-values (avoids left-side bias from ``jnp.argmax``).

        Args:
            state: Current SARSA state (uses rng_key and epsilon)
            observation: Input feature vector

        Returns:
            Tuple of (action, new_rng_key)
        """
        feature_dim = self._require_state_contract(state)
        raw_observation = jnp.asarray(observation)
        if raw_observation.shape != (feature_dim,):
            raise ValueError(f"SARSA observation must have shape ({feature_dim},)")
        if raw_observation.dtype != jnp.dtype(jnp.float32):
            raise TypeError("SARSA observation must have dtype float32")
        input_valid = self._state_values_valid(state) & jnp.all(jnp.isfinite(raw_observation))
        safe_observation = jnp.where(input_valid, raw_observation, jnp.zeros_like(raw_observation))
        key, explore_key, noise_key, random_key = jr.split(state.rng_key, 4)

        # Get Q-values (first n_actions heads are control demons)
        all_preds = self._horde.predict(state.learner_state, safe_observation)
        q_values = all_preds[: self._sarsa_config.n_actions]
        policy_valid = input_valid & jnp.all(jnp.isfinite(q_values))

        # Greedy action with Gumbel tie-breaking
        # Add small noise only to max-valued actions for uniform tie-breaking
        maximum = jnp.max(q_values)
        tie_noise = jnp.where(
            q_values == maximum,
            jr.gumbel(noise_key, shape=q_values.shape),
            -jnp.inf,
        )
        greedy_action = jnp.argmax(tie_noise).astype(jnp.int32)

        # Random action
        random_action = jr.randint(random_key, (), 0, self._sarsa_config.n_actions).astype(
            jnp.int32
        )

        # Epsilon-greedy selection
        explore = jr.uniform(explore_key) < state.epsilon
        action = jax.lax.select(explore, random_action, greedy_action)

        return (
            jnp.where(policy_valid, action, jnp.asarray(-1, dtype=jnp.int32)),
            jax.lax.cond(policy_valid, lambda: key, lambda: state.rng_key),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: SARSAState,
        reward: Array,
        observation: Array,
        terminated: Array,
        next_action: Array,
        prediction_cumulants: Array | None = None,
    ) -> SARSAUpdateResult:
        """Perform one SARSA update step.

        Computes the SARSA target ``r + gamma * Q(s', a')`` and updates
        the Horde. Only the previously-taken action's head receives the
        target; all other Q-heads get NaN (no update).

        Args:
            state: Current SARSA state
            reward: Reward r received after taking last_action in last_obs
            observation: New observation s' (state we transitioned to)
            terminated: Whether s' is terminal (scalar bool/float)
            next_action: Action a' selected for s' (pre-computed)
            prediction_cumulants: Optional cumulants for prediction demons,
                shape ``(n_prediction_demons,)``. NaN for inactive demons.

        Returns:
            SARSAUpdateResult with updated state, Q-values, TD error
        """
        n_actions = self._sarsa_config.n_actions
        gamma = self._sarsa_config.gamma
        feature_dim = self._require_state_contract(state)
        raw_observation = jnp.asarray(observation)
        if raw_observation.shape != (feature_dim,):
            raise ValueError(f"SARSA observation must have shape ({feature_dim},)")
        if raw_observation.dtype != jnp.dtype(jnp.float32):
            raise TypeError("SARSA observation must have dtype float32")
        raw_reward = jnp.asarray(reward)
        if raw_reward.shape != ():
            raise ValueError("SARSA reward must be scalar")
        if raw_reward.dtype != jnp.dtype(jnp.float32):
            raise TypeError("SARSA reward must have dtype float32")
        raw_terminated = jnp.asarray(terminated)
        if raw_terminated.shape != ():
            raise ValueError("SARSA terminated must be scalar")
        if raw_terminated.dtype not in (jnp.dtype(jnp.bool_), jnp.dtype(jnp.float32)):
            raise TypeError("SARSA terminated must have dtype bool or float32")
        raw_next_action = jnp.asarray(next_action)
        if raw_next_action.shape != ():
            raise ValueError("SARSA next_action must be scalar")
        if raw_next_action.dtype != jnp.dtype(jnp.int32):
            raise TypeError("SARSA next_action must have dtype int32")
        prediction_values = None
        prediction_values_valid = jnp.asarray(True, dtype=jnp.bool_)
        if prediction_cumulants is not None:
            prediction_values = jnp.asarray(prediction_cumulants)
            if prediction_values.shape != (self._n_prediction_demons,):
                raise ValueError(
                    f"SARSA prediction_cumulants must have shape ({self._n_prediction_demons},)"
                )
            if prediction_values.dtype != jnp.dtype(jnp.float32):
                raise TypeError("SARSA prediction_cumulants must have dtype float32")
            prediction_values_valid = jnp.all(~jnp.isinf(prediction_values))

        state_valid = self._state_values_valid(state)
        next_action_valid = (raw_next_action >= 0) & (raw_next_action < n_actions)
        terminated_valid = (raw_terminated == 0) | (raw_terminated == 1)
        inputs_valid = (
            jnp.all(jnp.isfinite(raw_observation))
            & jnp.isfinite(raw_reward)
            & terminated_valid
            & next_action_valid
            & prediction_values_valid
        )
        safe_observation = jnp.where(
            jnp.all(jnp.isfinite(raw_observation)),
            raw_observation,
            jnp.zeros_like(raw_observation),
        )
        safe_reward = jnp.where(jnp.isfinite(raw_reward), raw_reward, jnp.zeros_like(raw_reward))
        safe_terminated = jnp.where(terminated_valid, raw_terminated, jnp.ones_like(raw_terminated))
        safe_next_action = jnp.clip(raw_next_action, 0, n_actions - 1)

        # Q(s', :) for all actions
        all_preds = self._horde.predict(state.learner_state, safe_observation)
        q_next = all_preds[:n_actions]
        q_previous = self._horde.predict(
            state.learner_state,
            state.last_observation,
        )[:n_actions]

        # SARSA target: r + gamma * Q(s', a') with terminal handling.
        # A terminal or zero-gamma transition must not multiply Q(s', a');
        # jnp.where still evaluates 0 * inf, which is NaN.
        q_sa_next = q_next[safe_next_action]
        gamma_arr = jnp.asarray(gamma, dtype=q_sa_next.dtype)
        skip_bootstrap = (safe_terminated != 0) | (gamma_arr == 0.0)
        bootstrap = jnp.where(
            skip_bootstrap,
            jnp.zeros_like(q_sa_next),
            gamma_arr * q_sa_next,
        )
        sarsa_target = safe_reward + bootstrap

        # Build cumulants: NaN for all except last_action gets sarsa_target
        cumulants = jnp.full(self._horde.n_demons, jnp.nan, dtype=jnp.float32)
        # Only update the head corresponding to the action we took at s_t.
        # An update before select_action leaves last_action == -1, which
        # modular-indexes to the last head; gate it out instead so no control
        # head learns from a premature update.
        action_valid = (state.last_action >= 0) & (state.last_action < n_actions)
        safe_last_action = jnp.clip(
            state.last_action,
            0,
            n_actions - 1,
        )
        updated_cumulants = cumulants.at[safe_last_action].set(sarsa_target)
        cumulants = jnp.where(action_valid, updated_cumulants, cumulants)

        # Add prediction demon cumulants if any
        if prediction_values is not None:
            cumulants = cumulants.at[n_actions:].set(
                jnp.where(
                    action_valid & inputs_valid,
                    prediction_values,
                    jnp.full_like(prediction_values, jnp.nan),
                )
            )

        # Horde update: learns from (s_t, cumulants, s'). Control heads use
        # zero transition discounts (the SARSA target above already contains
        # the gamma * Q(s', a') bootstrap); prediction demons keep their
        # configured gammas. Head traces still decay by the spec's
        # gamma * lamda, which is what makes this SARSA(lambda).
        discounts = self._horde.horde_spec.gammas.at[:n_actions].set(0.0)
        horde_result = self._horde.update_with_discounts(
            state.learner_state,
            state.last_observation,
            cumulants,
            safe_observation,
            discounts,
        )

        # SARSA(lambda) trace maintenance. The inner learner decays only the
        # active head's trace (inactive heads are frozen by NaN masking), so
        # decay the untouched control heads here and clear all control-head
        # traces at episode boundaries so credit never crosses a reset.
        new_learner_state = horde_result.state
        if self._lamda > 0.0:
            gl = jnp.asarray(gamma * self._lamda, dtype=jnp.float32)
            head_traces = list(new_learner_state.head_traces)
            for i in range(n_actions):
                w_trace, b_trace = head_traces[i]
                decay = jnp.where(state.last_action == i, 1.0, gl)
                skipped_w = jnp.where(decay == 0.0, jnp.zeros_like(w_trace), decay * w_trace)
                skipped_b = jnp.where(decay == 0.0, jnp.zeros_like(b_trace), decay * b_trace)
                new_w = jnp.where(safe_terminated, jnp.zeros_like(w_trace), skipped_w)
                new_b = jnp.where(safe_terminated, jnp.zeros_like(b_trace), skipped_b)
                head_traces[i] = (new_w, new_b)
            new_learner_state = new_learner_state.replace(head_traces=tuple(head_traces))

        # TD error for the taken action
        q_old = q_previous[safe_last_action]
        td_error = jnp.where(action_valid, sarsa_target - q_old, 0.0)

        # Epsilon decay
        cfg = self._sarsa_config
        new_step_count = jnp.minimum(state.step_count, _INT32_MAX - 1) + 1
        new_epsilon = jax.lax.cond(
            cfg.epsilon_decay_steps > 0,
            lambda: jnp.maximum(
                cfg.epsilon_end,
                cfg.epsilon_start
                - (cfg.epsilon_start - cfg.epsilon_end) * new_step_count / cfg.epsilon_decay_steps,
            ),
            lambda: state.epsilon,
        )

        proposed_state = SARSAState(  # type: ignore[call-arg]
            learner_state=new_learner_state,
            last_action=safe_next_action,
            last_observation=safe_observation,
            epsilon=new_epsilon,
            rng_key=state.rng_key,
            step_count=new_step_count,
        )
        candidate_valid = _floating_tree_is_finite(proposed_state)
        zero_decay_trace_recovery = jnp.asarray(
            self._lamda > 0.0 and gamma == 0.0,
            dtype=jnp.bool_,
        )
        transaction_applied = (
            state_valid
            & action_valid
            & inputs_valid
            & (horde_result.update_applied | zero_decay_trace_recovery)
            & candidate_valid
        )
        diagnostic_valid = (
            action_valid
            & next_action_valid
            & jnp.isfinite(raw_reward)
            & terminated_valid
            & (jnp.all(jnp.isfinite(raw_observation)) | skip_bootstrap)
            & jnp.isfinite(td_error)
        )
        new_state = jax.lax.cond(transaction_applied, lambda: proposed_state, lambda: state)

        return SARSAUpdateResult(  # type: ignore[call-arg]
            state=new_state,
            action=jnp.where(transaction_applied, safe_next_action, -1),
            q_values=jnp.where(transaction_applied, q_next, jnp.zeros_like(q_next)),
            td_error=jnp.where(diagnostic_valid, td_error, 0.0),
            reward=jnp.where(jnp.isfinite(raw_reward), raw_reward, 0.0),
        )


# =============================================================================
# Learning Loops
# =============================================================================


def run_sarsa_episode(
    agent: SARSAAgent,
    state: SARSAState,
    env: Any,
    max_steps: int = 10000,
) -> SARSAEpisodeResult:
    """Run one episode of SARSA on a Gymnasium environment.

    Python loop (env interaction not JIT-able). Follows the SARSA
    pattern: select a' *before* updating, so the update uses the
    on-policy next action.

    Args:
        agent: SARSA agent
        state: Initial SARSA state
        env: Gymnasium environment
        max_steps: Maximum steps per episode

    Returns:
        SARSAEpisodeResult with episode metrics
    """
    max_steps = _require_sarsa_host_steps("max_steps", max_steps)
    obs, _info = env.reset()
    obs = jnp.asarray(obs, dtype=jnp.float32).flatten()

    # Select initial action
    action, new_key = agent.select_action(state, obs)
    state = state.replace(  # type: ignore[attr-defined]
        last_action=action,
        last_observation=obs,
        rng_key=new_key,
    )

    rewards: list[float] = []
    q_values_list: list[Array] = []
    td_errors: list[float] = []
    total_reward = 0.0

    for _ in range(max_steps):
        # Step environment
        next_obs, reward, terminated, truncated, _info = env.step(int(action))
        next_obs = jnp.asarray(next_obs, dtype=jnp.float32).flatten()
        reward_arr = jnp.array(reward, dtype=jnp.float32)
        term_arr = jnp.array(terminated, dtype=jnp.float32)

        # Select next action a' (on-policy)
        next_action, new_key = agent.select_action(state, next_obs)
        state = state.replace(rng_key=new_key)  # type: ignore[attr-defined]

        # SARSA update
        result = agent.update(state, reward_arr, next_obs, term_arr, next_action)
        state = result.state

        rewards.append(float(reward))
        q_values_list.append(result.q_values)
        td_errors.append(float(result.td_error))
        total_reward += float(reward)

        action = next_action

        if terminated or truncated:
            break

    return SARSAEpisodeResult(
        state=state,
        total_reward=total_reward,
        num_steps=len(rewards),
        rewards=rewards,
        q_values=q_values_list,
        td_errors=td_errors,
    )


def run_sarsa_continuing(
    agent: SARSAAgent,
    state: SARSAState,
    env: Any,
    num_steps: int,
) -> SARSAContinuingResult:
    """Run SARSA in continuing mode for a fixed number of steps.

    At episode boundaries, the environment auto-resets. gamma is set to 0
    at pseudo-boundaries (terminal/truncated) to prevent bootstrapping
    across resets, matching the ``ContinuingWrapper`` pattern.

    Args:
        agent: SARSA agent
        state: Initial SARSA state
        env: Gymnasium environment
        num_steps: Number of steps to run

    Returns:
        SARSAContinuingResult with step-level metrics
    """
    num_steps = _require_sarsa_host_steps("num_steps", num_steps)
    obs, _info = env.reset()
    obs = jnp.asarray(obs, dtype=jnp.float32).flatten()

    # Select initial action
    action, new_key = agent.select_action(state, obs)
    state = state.replace(  # type: ignore[attr-defined]
        last_action=action,
        last_observation=obs,
        rng_key=new_key,
    )

    rewards: list[float] = []
    q_values_list: list[Array] = []
    td_errors: list[float] = []
    total_reward = 0.0

    for _ in range(num_steps):
        next_obs, reward, terminated, truncated, _info = env.step(int(action))
        next_obs = jnp.asarray(next_obs, dtype=jnp.float32).flatten()
        reward_arr = jnp.array(reward, dtype=jnp.float32)

        # Continuing mode: gamma=0 at pseudo-boundaries
        is_boundary = terminated or truncated
        term_arr = jnp.array(is_boundary, dtype=jnp.float32)

        if is_boundary:
            next_obs_reset, _info = env.reset()
            next_obs = jnp.asarray(next_obs_reset, dtype=jnp.float32).flatten()

        # Select next action
        next_action, new_key = agent.select_action(state, next_obs)
        state = state.replace(rng_key=new_key)  # type: ignore[attr-defined]

        # SARSA update
        result = agent.update(state, reward_arr, next_obs, term_arr, next_action)
        state = result.state

        rewards.append(float(reward))
        q_values_list.append(result.q_values)
        td_errors.append(float(result.td_error))
        total_reward += float(reward)

        action = next_action

    return SARSAContinuingResult(
        state=state,
        total_reward=total_reward,
        rewards=rewards,
        q_values=q_values_list,
        td_errors=td_errors,
    )


def run_sarsa_from_arrays(
    agent: SARSAAgent,
    state: SARSAState,
    observations: Float[Array, "num_steps feature_dim"],
    rewards: Float[Array, " num_steps"],
    terminated: Float[Array, " num_steps"],
    next_observations: Float[Array, "num_steps feature_dim"],
) -> SARSAArrayResult:
    """Run SARSA on pre-collected arrays via ``jax.lax.scan``.

    JIT-compiled for maximum throughput. Actions are selected on-policy
    within the scan. Use this loop when transitions are pre-collected
    arrays rather than produced by a live environment interaction loop.

    Args:
        agent: SARSA agent
        state: Initial SARSA state (must have valid last_action, last_observation)
        observations: Current observations, shape ``(num_steps, feature_dim)``
        rewards: Rewards, shape ``(num_steps,)``
        terminated: Termination flags, shape ``(num_steps,)``
        next_observations: Next observations, shape ``(num_steps, feature_dim)``

    Returns:
        SARSAArrayResult with per-step Q-values, TD errors, and actions

    Raises:
        TypeError: If an input is not a JAX array.
        ValueError: If ``observations`` is empty, exceeds the documented
            scan-length ceiling (``_SARSA_SEQUENCE_MAX_STEPS``), or the other
            step arrays do not share its leading length.
    """
    num_steps = _require_sarsa_sequence_length("observations", observations)
    _require_sarsa_matching_length("rewards", rewards, expected=num_steps)
    _require_sarsa_matching_length("terminated", terminated, expected=num_steps)
    _require_sarsa_matching_length(
        "next_observations", next_observations, expected=num_steps
    )

    @jax.jit
    def _scan_fn(
        carry: SARSAState,
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[SARSAState, tuple[Array, Array, Array]]:
        s = carry
        obs, r, term, next_obs = inputs

        # Select next action for next_obs
        next_action, new_key = agent.select_action(s, next_obs)
        s = s.replace(rng_key=new_key)  # type: ignore[attr-defined]

        # Update using current obs/reward/next_obs
        result = agent.update(s, r, next_obs, term, next_action)

        return result.state, (result.q_values, result.td_error, result.action)

    t0 = time.time()
    final_state, (q_vals, td_errs, actions) = jax.lax.scan(
        _scan_fn, state, (observations, rewards, terminated, next_observations)
    )
    elapsed = time.time() - t0

    # Update uptime on the inner learner state
    final_learner = final_state.learner_state.replace(  # type: ignore[attr-defined]
        uptime_s=final_state.learner_state.uptime_s + elapsed,
    )
    final_state = final_state.replace(learner_state=final_learner)  # type: ignore[attr-defined]

    return SARSAArrayResult(  # type: ignore[call-arg]
        state=final_state,
        q_values=q_vals,
        td_errors=td_errs,
        actions=actions,
    )


def run_sarsa_from_arrays_final_state(
    agent: SARSAAgent,
    state: SARSAState,
    observations: Float[Array, "num_steps feature_dim"],
    rewards: Float[Array, " num_steps"],
    terminated: Float[Array, " num_steps"],
    next_observations: Float[Array, "num_steps feature_dim"],
) -> SARSAState:
    """Run the scan-compatible SARSA loop and return only the final state.

    Throughput benchmarks use this helper to avoid materializing per-step
    Q-values, TD errors, and actions.

    Raises:
        TypeError: If an input is not a JAX array.
        ValueError: If ``observations`` is empty, exceeds the documented
            scan-length ceiling (``_SARSA_SEQUENCE_MAX_STEPS``), or the other
            step arrays do not share its leading length.
    """
    num_steps = _require_sarsa_sequence_length("observations", observations)
    _require_sarsa_matching_length("rewards", rewards, expected=num_steps)
    _require_sarsa_matching_length("terminated", terminated, expected=num_steps)
    _require_sarsa_matching_length(
        "next_observations", next_observations, expected=num_steps
    )

    @jax.jit
    def _scan_fn(
        carry: SARSAState,
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[SARSAState, None]:
        s = carry
        _obs, r, term, next_obs = inputs
        next_action, new_key = agent.select_action(s, next_obs)
        s = s.replace(rng_key=new_key)  # type: ignore[attr-defined]
        result = agent.update(s, r, next_obs, term, next_action)
        return result.state, None

    t0 = time.time()
    final_state, _ = jax.lax.scan(
        _scan_fn,
        state,
        (observations, rewards, terminated, next_observations),
    )
    elapsed = time.time() - t0
    final_learner = final_state.learner_state.replace(  # type: ignore[attr-defined]
        uptime_s=final_state.learner_state.uptime_s + elapsed,
    )
    return final_state.replace(learner_state=final_learner)  # type: ignore[no-any-return, attr-defined]
