# mypy: disable-error-code="call-arg,name-defined"
"""Lightweight working-memory features for predictive state construction.

The module keeps causal, fixed-budget traces of observations, actions, and
rewards. It is intentionally smaller than a learned recurrent network: callers
can concatenate the feature vector into world models, behavior models, Horde
demons, or actor-critic inputs while preserving temporal-uniform per-step
updates.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.update_safety import (
    floating_tree_is_finite,
    neutralize_array,
    select_transaction,
)


@dataclass(frozen=True)
class WorkingMemoryConfig:
    """Configuration for :class:`WorkingMemoryFeaturizer`.

    A decay rate ``beta`` yields an EMA with effective timescale
    ``1 / (1 - beta)``, so the default observation rates ``(0.5, 0.9, 0.99)``
    span memory horizons of roughly 2, 10, and 100 steps — a multi-timescale
    view of recent history, the same construction motivated in
    :mod:`alberta_framework.core.history_features`.

    Args:
        observation_dim: Observation vector dimensionality.
        action_dim: Action-feature dimensionality, usually one-hot actions.
        reward_dim: Reward/cumulant vector dimensionality.
        observation_decay_rates: EMA rates for observation traces.
        action_decay_rates: EMA rates for action traces.
        reward_decay_rates: EMA rates for reward traces.
        include_current_observation: Include the current observation in output.
        include_current_action: Include the current action vector in output.
        include_current_reward: Include the current reward vector in output.
        include_traces: Include all trace banks in output.
        include_innovations: Include current-minus-fast-trace innovations.
        gated_update: If true, trace updates are scaled by a surprise gate.
        gate_threshold: Surprise level where gated updates start opening.
        gate_temperature: Positive softness for the surprise gate.
    """

    observation_dim: int
    action_dim: int = 0
    reward_dim: int = 1
    observation_decay_rates: tuple[float, ...] = (0.5, 0.9, 0.99)
    action_decay_rates: tuple[float, ...] = (0.5, 0.9)
    reward_decay_rates: tuple[float, ...] = (0.5, 0.9)
    include_current_observation: bool = True
    include_current_action: bool = True
    include_current_reward: bool = True
    include_traces: bool = True
    include_innovations: bool = False
    gated_update: bool = False
    gate_threshold: float = 0.0
    gate_temperature: float = 1.0

    def __post_init__(self) -> None:
        _validate_config(self)

    def feature_dim(self) -> int:
        """Return the working-memory feature dimensionality."""
        dim = 0
        if self.include_current_observation:
            dim += self.observation_dim
        if self.include_current_action:
            dim += self.action_dim
        if self.include_current_reward:
            dim += self.reward_dim
        if self.include_traces:
            dim += self.observation_dim * len(self.observation_decay_rates)
            dim += self.action_dim * len(self.action_decay_rates)
            dim += self.reward_dim * len(self.reward_decay_rates)
        if self.include_innovations:
            dim += self.observation_dim * int(len(self.observation_decay_rates) > 0)
            dim += self.action_dim * int(len(self.action_decay_rates) > 0)
            dim += self.reward_dim * int(len(self.reward_decay_rates) > 0)
        return dim

    def to_config(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        payload = asdict(self)
        payload["observation_decay_rates"] = list(self.observation_decay_rates)
        payload["action_decay_rates"] = list(self.action_decay_rates)
        payload["reward_decay_rates"] = list(self.reward_decay_rates)
        payload["type"] = "WorkingMemoryConfig"
        return payload

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> WorkingMemoryConfig:
        """Reconstruct from :meth:`to_config` output."""
        if not issubclass(type(config), Mapping):
            raise ValueError("WorkingMemoryConfig payload must be a mapping")
        try:
            payload = dict(config)
        except Exception as error:
            raise ValueError("WorkingMemoryConfig mapping could not be read") from error
        if payload.pop("type", None) != "WorkingMemoryConfig":
            raise ValueError("WorkingMemoryConfig type is invalid")
        if set(payload) != {field.name for field in fields(cls)}:
            raise ValueError("WorkingMemoryConfig fields do not match its schema")
        for key in (
            "observation_decay_rates",
            "action_decay_rates",
            "reward_decay_rates",
        ):
            if key in payload:
                value = payload[key]
                if type(value) is list:
                    payload[key] = tuple(value)
                elif type(value) is not tuple:
                    raise ValueError(f"{key} must be an actual list or tuple")
                if len(payload[key]) > _MAX_WORKING_MEMORY_DECAY_RATES:
                    raise ValueError(
                        f"{key} must contain at most "
                        f"{_MAX_WORKING_MEMORY_DECAY_RATES} decay rates"
                    )
                if any(type(item) is not float for item in payload[key]):
                    raise ValueError(f"serialized {key} values must be JSON numbers")
        for key in ("observation_dim", "action_dim", "reward_dim"):
            if type(payload[key]) is not int:
                raise ValueError(f"serialized {key} must be a JSON integer")
        for key in (
            "include_current_observation",
            "include_current_action",
            "include_current_reward",
            "include_traces",
            "include_innovations",
            "gated_update",
        ):
            if type(payload[key]) is not bool:
                raise ValueError(f"serialized {key} must be a JSON boolean")
        for key in ("gate_threshold", "gate_temperature"):
            if type(payload[key]) is not float:
                raise ValueError(f"serialized {key} must be a JSON number")
        return cls(**payload)


@chex.dataclass(frozen=True)
class WorkingMemoryState:
    """State for :class:`WorkingMemoryFeaturizer`."""

    observation_traces: Float[Array, "n_observation_decays observation_dim"]
    action_traces: Float[Array, "n_action_decays action_dim"]
    reward_traces: Float[Array, "n_reward_decays reward_dim"]
    step_count: Array
    last_gate: Float[Array, " 3"]


@chex.dataclass(frozen=True)
class WorkingMemoryDiagnostics:
    """Scalar diagnostics for the current memory state."""

    step_count: Array
    trace_energy: Array
    effective_dimension: Array
    observation_energy: Array
    action_energy: Array
    reward_energy: Array
    last_gate: Float[Array, " 3"]


@chex.dataclass(frozen=True)
class WorkingMemoryUpdateResult:
    """Checked state transition for one working-memory event."""

    state: WorkingMemoryState
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True, mappable_dataclass=False)
class WorkingMemoryStepResult:
    """Checked feature-and-update result with legacy two-value unpacking.

    Iteration intentionally yields ``(state, features)`` so existing callers
    retain their pre-1.0 unpacking surface. Transaction-aware callers should
    inspect :attr:`update_applied` before consuming the features.
    """

    state: WorkingMemoryState
    features: Float[Array, " feature_dim"]
    update_applied: Bool[Array, ""]

    def __iter__(self) -> Iterator[Any]:
        yield self.state
        yield self.features


@chex.dataclass(frozen=True, mappable_dataclass=False)
class WorkingMemoryArrayResult:
    """Checked causal transform with one transaction verdict per event.

    Iteration preserves the historical ``(state, features)`` return surface;
    new callers should also consume :attr:`updates_applied`.
    """

    state: WorkingMemoryState
    features: Float[Array, "steps feature_dim"]
    updates_applied: Bool[Array, " steps"]

    def __iter__(self) -> Iterator[Any]:
        yield self.state
        yield self.features


_INT32_MAX = 2**31 - 1
# Bounded cardinality for decay-rate tuples, mirroring the ceiling added to
# HistoryFeatureExtractor in #2124 (unbounded per-item walks are a hostile
# resource risk before aggregate validation runs).
_MAX_WORKING_MEMORY_DECAY_RATES = 4_096
_FLOAT32_MIN_NORMAL = float.fromhex("0x1.0p-126")
_ACTUAL_INT_TYPES: tuple[type, ...] = (
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


def _require_int32(name: str, value: object, *, minimum: int) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer")
    number = int(cast(int, value))
    if number < minimum or number > _INT32_MAX:
        raise ValueError(f"{name} must be in [{minimum}, {_INT32_MAX}]")
    return number


_UINT32_MAX: int = 4294967295


def _require_float32_resource(
    name: str,
    *,
    vector_scalars: int,
    fixed_scalars: int = 0,
) -> None:
    total_scalars = vector_scalars + fixed_scalars
    if total_scalars > _INT32_MAX:
        raise ValueError(f"{name} scalar count must fit signed int32")
    if 4 * total_scalars > _INT32_MAX:
        raise ValueError(f"{name} byte count must fit signed int32")


def _require_sequence_resource(name: str, *, float32_scalars: int, bool_scalars: int) -> None:
    if float32_scalars + bool_scalars > _INT32_MAX:
        raise ValueError(f"{name} scalar count must fit signed int32")
    if 4 * float32_scalars + bool_scalars > _INT32_MAX:
        raise ValueError(f"{name} byte count must fit signed int32")


def _require_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any = jnp.float32,
) -> None:
    try:
        actual_shape = tuple(value.shape)  # type: ignore[attr-defined]
        actual_dtype = jnp.dtype(value.dtype)  # type: ignore[attr-defined]
    except Exception as error:
        raise TypeError(f"{name} must expose array shape and dtype metadata") from error
    if actual_shape != shape:
        if name == "vector":
            raise ValueError("vector must have shape matching its configured dimension")
        raise ValueError(f"{name} must have the configured shape")
    if actual_dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} has an invalid dtype")


def _validate_decay_rates(name: str, rates: object) -> tuple[float, ...]:
    if type(rates) is not tuple:
        raise ValueError(f"{name} must be an actual tuple")
    if len(rates) > _MAX_WORKING_MEMORY_DECAY_RATES:
        raise ValueError(
            f"{name} must contain at most {_MAX_WORKING_MEMORY_DECAY_RATES} decay rates"
        )
    return tuple(
        validated_float32_scalar(
            f"{name}[{index}]",
            value,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        for index, value in enumerate(rates)
    )


def _validate_config(config: WorkingMemoryConfig) -> None:
    observation_dim = _require_int32("observation_dim", config.observation_dim, minimum=1)
    action_dim = _require_int32("action_dim", config.action_dim, minimum=0)
    reward_dim = _require_int32("reward_dim", config.reward_dim, minimum=0)
    observation_decay_rates = _validate_decay_rates(
        "observation_decay_rates", config.observation_decay_rates
    )
    action_decay_rates = _validate_decay_rates(
        "action_decay_rates", config.action_decay_rates
    )
    reward_decay_rates = _validate_decay_rates(
        "reward_decay_rates", config.reward_decay_rates
    )
    gate_threshold = validated_float32_scalar(
        "gate_threshold", config.gate_threshold, lower=0.0
    )
    gate_temperature = validated_float32_scalar(
        "gate_temperature", config.gate_temperature, lower=_FLOAT32_MIN_NORMAL
    )
    for name in (
        "include_current_observation",
        "include_current_action",
        "include_current_reward",
        "include_traces",
        "include_innovations",
        "gated_update",
    ):
        if type(getattr(config, name)) is not bool:
            raise ValueError(f"{name} must be an actual bool")

    object.__setattr__(config, "observation_dim", observation_dim)
    object.__setattr__(config, "action_dim", action_dim)
    object.__setattr__(config, "reward_dim", reward_dim)
    object.__setattr__(config, "observation_decay_rates", observation_decay_rates)
    object.__setattr__(config, "action_decay_rates", action_decay_rates)
    object.__setattr__(config, "reward_decay_rates", reward_decay_rates)
    object.__setattr__(config, "gate_threshold", gate_threshold)
    object.__setattr__(config, "gate_temperature", gate_temperature)

    feature_dim = config.feature_dim()
    if feature_dim < 1 or feature_dim > _INT32_MAX:
        raise ValueError(
            f"configuration feature_dim must be in [1, {_INT32_MAX}], got {feature_dim}"
        )
    trace_scalars = (
        observation_dim * len(observation_decay_rates)
        + action_dim * len(action_decay_rates)
        + reward_dim * len(reward_decay_rates)
    )
    if trace_scalars > _INT32_MAX:
        raise ValueError("WorkingMemoryConfig dimensions must fit signed int32")
    total_state_scalars = trace_scalars + 4
    _require_float32_resource(
        "WorkingMemoryConfig state",
        vector_scalars=trace_scalars,
        fixed_scalars=4,
    )
    persistent_bytes = 4 * total_state_scalars
    if persistent_bytes > _INT32_MAX:
        raise ValueError("WorkingMemoryConfig state byte count must fit signed int32")
    if persistent_bytes > _UINT32_MAX:
        raise ValueError("working memory allocation exceeds uint32 byte accounting")
    decay_scalars = (
        len(observation_decay_rates)
        + len(action_decay_rates)
        + len(reward_decay_rates)
    )
    signal_scalars = observation_dim + action_dim + reward_dim
    feature_work = total_state_scalars + feature_dim + 2 * signal_scalars
    update_work = (
        3 * total_state_scalars
        + 6 * trace_scalars
        + 4 * signal_scalars
        + decay_scalars
        + feature_dim
        + 32
    )
    diagnostics_work = total_state_scalars + 3 * trace_scalars + 9
    _require_float32_resource("WorkingMemoryConfig features", vector_scalars=feature_dim)
    _require_float32_resource(
        "WorkingMemoryConfig feature operation", vector_scalars=feature_work
    )
    _require_float32_resource(
        "WorkingMemoryConfig update operation", vector_scalars=update_work
    )
    _require_float32_resource(
        "WorkingMemoryConfig diagnostics operation", vector_scalars=diagnostics_work
    )


def _empty_or_vector(value: Array, dim: int) -> Array:
    _require_array("vector", value, shape=(dim,))
    return jnp.asarray(value)


def _trace_bank(decay_count: int, dim: int) -> Array:
    return jnp.zeros((decay_count, dim), dtype=jnp.float32)


def _flatten_traces(state: WorkingMemoryState) -> Array:
    return jnp.concatenate(
        [
            state.observation_traces.reshape(-1),
            state.action_traces.reshape(-1),
            state.reward_traces.reshape(-1),
        ],
        axis=0,
    )


def _root_mean_square(values: Array) -> Array:
    if values.size == 0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    scale = jnp.max(jnp.abs(values))
    safe_scale = jnp.where(scale > 0.0, scale, 1.0)
    normalized = jnp.where(
        jnp.abs(values) == scale,
        jnp.sign(values),
        values / safe_scale,
    )
    return jnp.where(scale > 0.0, scale * jnp.sqrt(jnp.mean(normalized * normalized)), 0.0)


def _effective_dimension(values: Array) -> Array:
    if values.size == 0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    scale = jnp.max(jnp.abs(values))
    safe_scale = jnp.where(scale > 0.0, scale, 1.0)
    normalized = jnp.where(
        jnp.abs(values) == scale,
        jnp.sign(values),
        values / safe_scale,
    )
    squared = normalized * normalized
    energy = jnp.sum(squared)
    fourth = jnp.sum(squared * squared)
    return jnp.where(fourth > 0.0, (energy * energy) / fourth, 0.0)


class WorkingMemoryFeaturizer:
    """Causal observation/action/reward trace features.

    ``features(state, observation, action, reward)`` exposes the current
    signals plus pre-update traces. ``update`` then advances memory with the
    same transition. This ordering lets callers predict the next environment
    event from information available before the current event is written into
    memory, while still allowing current observation/action/reward to be part
    of the model input when configured.
    """

    def __init__(self, config: WorkingMemoryConfig):
        _validate_config(config)
        self._config = config

    @property
    def config(self) -> WorkingMemoryConfig:
        """Featurizer configuration."""
        return self._config

    def feature_dim(self) -> int:
        """Return the working-memory feature dimensionality."""
        return self._config.feature_dim()

    def to_config(self) -> dict[str, Any]:
        """Serialize the featurizer configuration."""
        return {
            "type": "WorkingMemoryFeaturizer",
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> WorkingMemoryFeaturizer:
        """Reconstruct a featurizer from :meth:`to_config` output."""
        if not issubclass(type(config), Mapping):
            raise ValueError("WorkingMemoryFeaturizer payload must be a mapping")
        try:
            payload = dict(config)
        except Exception as error:
            raise ValueError("WorkingMemoryFeaturizer mapping could not be read") from error
        if set(payload) != {"type", "config"}:
            raise ValueError("WorkingMemoryFeaturizer fields do not match its schema")
        if payload["type"] != "WorkingMemoryFeaturizer":
            raise ValueError("WorkingMemoryFeaturizer type is invalid")
        inner = payload["config"]
        if not issubclass(type(inner), Mapping):
            raise ValueError("WorkingMemoryFeaturizer config must be a mapping")
        return cls(WorkingMemoryConfig.from_config(cast(Mapping[str, Any], inner)))

    def _validate_state_static_contract(self, state: WorkingMemoryState) -> None:
        """Reject malformed adopted state before traced computation."""
        if type(state) is not WorkingMemoryState:
            raise TypeError("state must be a WorkingMemoryState")
        cfg = self._config
        expected = (
            (
                "state.observation_traces",
                state.observation_traces,
                (len(cfg.observation_decay_rates), cfg.observation_dim),
                jnp.float32,
            ),
            (
                "state.action_traces",
                state.action_traces,
                (len(cfg.action_decay_rates), cfg.action_dim),
                jnp.float32,
            ),
            (
                "state.reward_traces",
                state.reward_traces,
                (len(cfg.reward_decay_rates), cfg.reward_dim),
                jnp.float32,
            ),
            ("state.step_count", state.step_count, (), jnp.int32),
            ("state.last_gate", state.last_gate, (3,), jnp.float32),
        )
        for name, value, shape, dtype in expected:
            _require_array(name, value, shape=shape, dtype=dtype)

    @staticmethod
    def _state_is_valid(state: WorkingMemoryState) -> Bool[Array, ""]:
        return (
            floating_tree_is_finite(state)
            & (state.step_count >= 0)
            & jnp.all(state.last_gate >= 0.0)
            & jnp.all(state.last_gate <= 1.0)
        )

    def init(self) -> WorkingMemoryState:
        """Return an all-zero memory state."""
        cfg = self._config
        return WorkingMemoryState(
            observation_traces=_trace_bank(
                len(cfg.observation_decay_rates),
                cfg.observation_dim,
            ),
            action_traces=_trace_bank(len(cfg.action_decay_rates), cfg.action_dim),
            reward_traces=_trace_bank(len(cfg.reward_decay_rates), cfg.reward_dim),
            step_count=jnp.array(0, dtype=jnp.int32),
            last_gate=jnp.ones((3,), dtype=jnp.float32),
        )

    def reset(self) -> WorkingMemoryState:
        """Reset memory to its initial all-zero state."""
        return self.init()

    def zero_action(self) -> Float[Array, " action_dim"]:
        """Return a zero action vector with the configured dimension."""
        return jnp.zeros((self._config.action_dim,), dtype=jnp.float32)

    def zero_reward(self) -> Float[Array, " reward_dim"]:
        """Return a zero reward vector with the configured dimension."""
        return jnp.zeros((self._config.reward_dim,), dtype=jnp.float32)

    def _surprise_gate(self, traces: Array, value: Array, threshold: Array) -> Array:
        if (not self._config.gated_update) or traces.shape[0] == 0 or value.size == 0:
            return jnp.asarray(1.0, dtype=jnp.float32)
        surprise = _root_mean_square(value - traces[0])
        temperature = jnp.asarray(self._config.gate_temperature, dtype=jnp.float32)
        return jax.nn.sigmoid((surprise - threshold) / temperature)

    @staticmethod
    def _update_trace_bank(
        traces: Array,
        value: Array,
        decay_rates: tuple[float, ...],
        gate: Array,
    ) -> Array:
        if len(decay_rates) == 0:
            return traces
        decay = jnp.asarray(decay_rates, dtype=jnp.float32)[:, None]
        update_rate = (1.0 - decay) * gate
        persist = 1.0 - update_rate
        delta_update = traces + update_rate * (value[None, :] - traces)
        return jnp.where(
            persist == 0.0,
            jnp.broadcast_to(value[None, :], traces.shape),
            jnp.where(update_rate == 0.0, traces, delta_update),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def features(
        self,
        state: WorkingMemoryState,
        observation: Float[Array, " observation_dim"],
        action: Float[Array, " action_dim"],
        reward: Float[Array, " reward_dim"],
    ) -> Float[Array, " feature_dim"]:
        """Return current working-memory features without advancing state."""
        self._validate_state_static_contract(state)
        cfg = self._config
        obs = _empty_or_vector(observation, cfg.observation_dim)
        act = _empty_or_vector(action, cfg.action_dim)
        rew = _empty_or_vector(reward, cfg.reward_dim)

        blocks = []
        if cfg.include_current_observation:
            blocks.append(obs)
        if cfg.include_current_action:
            blocks.append(act)
        if cfg.include_current_reward:
            blocks.append(rew)
        if cfg.include_traces:
            blocks.extend(
                [
                    state.observation_traces.reshape(-1),
                    state.action_traces.reshape(-1),
                    state.reward_traces.reshape(-1),
                ]
            )
        if cfg.include_innovations:
            if len(cfg.observation_decay_rates) > 0:
                blocks.append(obs - state.observation_traces[0])
            if len(cfg.action_decay_rates) > 0:
                blocks.append(act - state.action_traces[0])
            if len(cfg.reward_decay_rates) > 0:
                blocks.append(rew - state.reward_traces[0])
        return jnp.concatenate(blocks, axis=0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_checked(
        self,
        state: WorkingMemoryState,
        observation: Float[Array, " observation_dim"],
        action: Float[Array, " action_dim"],
        reward: Float[Array, " reward_dim"],
        external_gate: Float[Array, ""] | float = 1.0,
    ) -> WorkingMemoryUpdateResult:
        """Advance one event and report whether the transaction committed."""
        self._validate_state_static_contract(state)
        cfg = self._config
        obs = _empty_or_vector(observation, cfg.observation_dim)
        act = _empty_or_vector(action, cfg.action_dim)
        rew = _empty_or_vector(reward, cfg.reward_dim)
        raw_outer_gate = jnp.asarray(external_gate, dtype=jnp.float32)
        if raw_outer_gate.shape != ():
            raise ValueError(
                f"external_gate must have scalar shape (); got {raw_outer_gate.shape}"
            )
        inputs_valid = (
            jnp.all(jnp.isfinite(obs))
            & jnp.all(jnp.isfinite(act))
            & jnp.all(jnp.isfinite(rew))
            & jnp.all(jnp.isfinite(raw_outer_gate))
        )
        safe_obs = jnp.where(inputs_valid, obs, jnp.zeros_like(obs))
        safe_act = jnp.where(inputs_valid, act, jnp.zeros_like(act))
        safe_rew = jnp.where(inputs_valid, rew, jnp.zeros_like(rew))
        outer_gate = jnp.clip(
            jnp.where(inputs_valid, raw_outer_gate, jnp.zeros_like(raw_outer_gate)),
            0.0,
            1.0,
        )
        threshold = jnp.asarray(cfg.gate_threshold, dtype=jnp.float32)

        observation_gate = outer_gate * self._surprise_gate(
            state.observation_traces,
            safe_obs,
            threshold,
        )
        action_gate = outer_gate * self._surprise_gate(
            state.action_traces,
            safe_act,
            threshold,
        )
        reward_gate = outer_gate * self._surprise_gate(
            state.reward_traces,
            safe_rew,
            threshold,
        )

        candidate = WorkingMemoryState(
            observation_traces=self._update_trace_bank(
                state.observation_traces,
                safe_obs,
                cfg.observation_decay_rates,
                observation_gate,
            ),
            action_traces=self._update_trace_bank(
                state.action_traces,
                safe_act,
                cfg.action_decay_rates,
                action_gate,
            ),
            reward_traces=self._update_trace_bank(
                state.reward_traces,
                safe_rew,
                cfg.reward_decay_rates,
                reward_gate,
            ),
            step_count=jnp.minimum(
                state.step_count,
                jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32),
            )
            + jnp.asarray(1, dtype=jnp.int32),
            last_gate=jnp.stack([observation_gate, action_gate, reward_gate]),
        )
        previous_checked = state
        if cfg.observation_decay_rates and all(rate == 0.0 for rate in cfg.observation_decay_rates):
            previous_checked = previous_checked.replace(  # type: ignore[attr-defined]
                observation_traces=jnp.zeros_like(state.observation_traces),
            )
        if cfg.action_decay_rates and all(rate == 0.0 for rate in cfg.action_decay_rates):
            previous_checked = previous_checked.replace(  # type: ignore[attr-defined]
                action_traces=jnp.zeros_like(state.action_traces),
            )
        if cfg.reward_decay_rates and all(rate == 0.0 for rate in cfg.reward_decay_rates):
            previous_checked = previous_checked.replace(  # type: ignore[attr-defined]
                reward_traces=jnp.zeros_like(state.reward_traces),
            )
        update_applied = (
            inputs_valid
            & self._state_is_valid(previous_checked)
            & self._state_is_valid(candidate)
        )
        return WorkingMemoryUpdateResult(
            state=select_transaction(update_applied, candidate, state),
            update_applied=update_applied,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: WorkingMemoryState,
        observation: Float[Array, " observation_dim"],
        action: Float[Array, " action_dim"],
        reward: Float[Array, " reward_dim"],
        external_gate: Float[Array, ""] | float = 1.0,
    ) -> WorkingMemoryState:
        """Advance memory, retaining the legacy state-only return surface.

        Transaction-aware callers should use :meth:`update_checked`.
        """
        return cast(
            WorkingMemoryState,
            self.update_checked(
                state,
                observation,
                action,
                reward,
                external_gate,
            ).state,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: WorkingMemoryState,
        observation: Float[Array, " observation_dim"],
        action: Float[Array, " action_dim"],
        reward: Float[Array, " reward_dim"],
        external_gate: Float[Array, ""] | float = 1.0,
    ) -> WorkingMemoryStepResult:
        """Return neutral pre-update features and an explicit update verdict."""
        raw_features = self.features(state, observation, action, reward)
        update = self.update_checked(state, observation, action, reward, external_gate)
        update_applied = update.update_applied & jnp.all(jnp.isfinite(raw_features))
        return WorkingMemoryStepResult(
            state=select_transaction(update_applied, update.state, state),
            features=neutralize_array(update_applied, raw_features),
            update_applied=update_applied,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def diagnostics(self, state: WorkingMemoryState) -> WorkingMemoryDiagnostics:
        """Return memory energy, participation-ratio dimension, and gates."""
        self._validate_state_static_contract(state)
        flat = _flatten_traces(state)
        return WorkingMemoryDiagnostics(
            step_count=state.step_count,
            trace_energy=_root_mean_square(flat),
            effective_dimension=_effective_dimension(flat),
            observation_energy=_root_mean_square(state.observation_traces.reshape(-1)),
            action_energy=_root_mean_square(state.action_traces.reshape(-1)),
            reward_energy=_root_mean_square(state.reward_traces.reshape(-1)),
            last_gate=state.last_gate,
        )


def transform_working_memory_arrays(
    featurizer: WorkingMemoryFeaturizer,
    observations: Float[Array, "steps observation_dim"],
    actions: Float[Array, "steps action_dim"],
    rewards: Float[Array, "steps reward_dim"],
    *,
    state: WorkingMemoryState | None = None,
    external_gates: Float[Array, " steps"] | None = None,
) -> WorkingMemoryArrayResult:
    """Transform arrays and expose one checked transaction mask per event."""
    if type(featurizer) is not WorkingMemoryFeaturizer:
        raise TypeError("featurizer must be an actual WorkingMemoryFeaturizer")
    if state is not None and type(state) is not WorkingMemoryState:
        raise TypeError("state must be an actual WorkingMemoryState")
    cfg = featurizer.config
    for name, arr in (("observations", observations), ("actions", actions), ("rewards", rewards)):
        actual_type = type(cast(object, arr))
        if not (
            actual_type is np.ndarray
            or isinstance(arr, (jax.Array, jax.core.Tracer))
            or actual_type is jax.ShapeDtypeStruct
            or issubclass(actual_type, jax.core.ShapedArray)
            or (hasattr(arr, "shape") and hasattr(arr, "dtype"))
        ):
            raise TypeError(f"{name} must be a trusted array")
    if external_gates is not None:
        actual_gate_type = type(cast(object, external_gates))
        if not (
            actual_gate_type is np.ndarray
            or isinstance(external_gates, (jax.Array, jax.core.Tracer))
            or actual_gate_type is jax.ShapeDtypeStruct
            or issubclass(actual_gate_type, jax.core.ShapedArray)
            or (hasattr(external_gates, "shape") and hasattr(external_gates, "dtype"))
        ):
            raise TypeError("external_gates must be a trusted array")

    if getattr(observations, "ndim", len(getattr(observations, "shape", ()))) != 2:
        raise ValueError("observations must be 2-dimensional (steps, observation_dim)")
    if getattr(actions, "ndim", len(getattr(actions, "shape", ()))) != 2:
        raise ValueError("actions must be 2-dimensional (steps, action_dim)")
    if getattr(rewards, "ndim", len(getattr(rewards, "shape", ()))) != 2:
        raise ValueError("rewards must be 2-dimensional (steps, reward_dim)")
    if (
        external_gates is not None
        and getattr(external_gates, "ndim", len(getattr(external_gates, "shape", ()))) != 1
    ):
        raise ValueError("external_gates must be 1-dimensional (steps,)")

    try:
        observation_shape = tuple(observations.shape)
        action_shape = tuple(actions.shape)
        reward_shape = tuple(rewards.shape)
    except Exception as error:
        raise TypeError("transform inputs must expose array metadata") from error
    if len(observation_shape) != 2 or observation_shape[1] != cfg.observation_dim:
        raise ValueError("observations have an invalid shape")
    steps = observation_shape[0]
    if type(steps) is not int or not 1 <= steps <= _INT32_MAX:
        raise ValueError("transform step count must be between 1 and signed-int32 steps")
    if action_shape != (steps, cfg.action_dim):
        raise ValueError("actions have an invalid shape")
    if reward_shape != (steps, cfg.reward_dim):
        raise ValueError("rewards have an invalid shape")
    if external_gates is not None:
        try:
            gate_shape = tuple(external_gates.shape)
        except Exception as error:
            raise TypeError("external_gates must expose array metadata") from error
        if gate_shape != (steps,):
            raise ValueError("external_gates have an invalid shape")
    _require_sequence_resource(
        "working memory transform outputs",
        float32_scalars=steps * cfg.feature_dim(),
        bool_scalars=steps,
    )
    trace_scalars = (
        cfg.observation_dim * len(cfg.observation_decay_rates)
        + cfg.action_dim * len(cfg.action_decay_rates)
        + cfg.reward_dim * len(cfg.reward_decay_rates)
    )
    state_scalars = trace_scalars + 4
    signal_scalars = cfg.observation_dim + cfg.action_dim + cfg.reward_dim
    decay_scalars = (
        len(cfg.observation_decay_rates)
        + len(cfg.action_decay_rates)
        + len(cfg.reward_decay_rates)
    )
    per_step_work = (
        3 * state_scalars
        + 6 * trace_scalars
        + 4 * signal_scalars
        + decay_scalars
        + cfg.feature_dim()
        + 32
    )
    _require_sequence_resource(
        "working memory transform aggregate",
        float32_scalars=steps * (signal_scalars + cfg.feature_dim() + 1)
        + per_step_work,
        bool_scalars=steps,
    )
    _require_array("observations", observations, shape=(steps, cfg.observation_dim))
    _require_array("actions", actions, shape=(steps, cfg.action_dim))
    _require_array("rewards", rewards, shape=(steps, cfg.reward_dim))
    if external_gates is not None:
        _require_array("external_gates", external_gates, shape=(steps,))
    _obs = jnp.asarray(observations)
    _act = jnp.asarray(actions)
    _rew = jnp.asarray(rewards)
    if state is None:
        state = featurizer.init()
    featurizer._validate_state_static_contract(state)
    gates = (
        jnp.ones((steps,), dtype=jnp.float32)
        if external_gates is None
        else jnp.asarray(external_gates)
    )

    def step_fn(
        carry: WorkingMemoryState,
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[WorkingMemoryState, tuple[Array, Array]]:
        obs, act, rew, gate = inputs
        result = featurizer.step(carry, obs, act, rew, gate)
        return result.state, (result.features, result.update_applied)

    final_state, (features, updates_applied) = jax.lax.scan(
        step_fn,
        state,
        (_obs, _act, _rew, gates),
    )
    return WorkingMemoryArrayResult(
        state=final_state,
        features=features,
        updates_applied=updates_applied,
    )
