"""Actor-critic control with discrete and continuous policies.

This module provides the Step 4b control cores for daemon-style use:
``ActorCriticAgent`` for discrete (softmax) actions and
``ContinuousActorCriticAgent`` for continuous (diagonal-Gaussian) actions.
Both share the same linear-critic AC(lambda) semantics, separate eligibility
traces, and pure single-step APIs compatible with ``jax.jit`` and
``jax.lax.scan``.

The Horde-backed critic integration point is the scalar ``value``/TD-error
path in ``update``: replace the linear critic estimate and critic trace update
with a GVF value adapter that preserves the actor's advantage signal. That
adapter lives in :mod:`alberta_framework.core.horde_actor_critic`
(``HordeActorCriticAgent`` and ``QHordeActorCriticAgent``); it is kept out of
this core slice so the linear AC(lambda) semantics here remain explicit and
covered by focused tests.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import operator
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework._float32 import round_real_to_float32
from alberta_framework.core._float32_scalars import validated_float32_scalar_with_ratio
from alberta_framework.core.optimizers import Bounder, bounder_from_config
from alberta_framework.core.update_safety import (
    checked_integer_action_array,
)
from alberta_framework.core.update_safety import (
    floating_tree_is_finite as _floating_tree_is_finite,
)

_INT32_MAX = 2**31 - 1
_ACTUAL_INT_TYPES = frozenset(
    {int, *(np.dtype(code).type for code in "bBhHiIlLqQpP")}
)
_ACTUAL_REAL_TYPES = _ACTUAL_INT_TYPES | frozenset(
    {float, *(np.dtype(code).type for code in "efdg")}
)


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _require_float32(
    name: str,
    value: object,
    *,
    positive: bool = False,
    lower: float | None = None,
    upper: float | None = None,
    preserve_nonzero: bool = False,
) -> float:
    if type(value) not in _ACTUAL_REAL_TYPES:
        raise ValueError(f"{name} must be a finite real number")
    stored, numerator, denominator = validated_float32_scalar_with_ratio(
        name, value, positive=positive, lower=lower, upper=upper
    )
    if preserve_nonzero and numerator != 0 and np.float32(numerator / denominator) == 0.0:
        raise ValueError(f"{name} must remain nonzero once narrowed to float32")
    return stored


def _read_mapping(name: str, value: object) -> dict[str, Any]:
    if not issubclass(type(value), Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        return dict(cast(Mapping[str, Any], value))
    except Exception as error:
        raise ValueError(f"{name} must be a readable mapping") from error


def _serialized_mapping(
    name: str, value: object, *, fields: frozenset[str]
) -> dict[str, Any]:
    payload = _read_mapping(name, value)
    if any(type(key) is not str for key in payload) or set(payload) != fields:
        raise ValueError(f"{name} fields do not match the serialized schema")
    return payload


def _require_discrete_state_resources(n_actions: int, feature_dim: int) -> None:
    state_scalars = 2 * n_actions * feature_dim + 3 * feature_dim + 2 * n_actions + 6
    state_bytes = 8 * n_actions * feature_dim + 12 * feature_dim + 8 * n_actions + 24
    if state_scalars > _INT32_MAX or state_bytes > _INT32_MAX:
        raise ValueError("derived actor-critic state exceeds the signed-int32 budget")


def _actor_critic_persistent_bytes(n_actions: int, feature_dim: int) -> int:
    """Named persist already counted inside ``_require_discrete_state_resources``."""
    return 8 * n_actions * feature_dim + 12 * feature_dim + 8 * n_actions + 24


def _actor_critic_update_result_extras_bytes(n_actions: int) -> int:
    """Returned ``ActorCriticUpdateResult`` extras excluding persist.

    Nested ``state`` is persist and is already counted in the simultaneous
    persist copies. These extras are the published action, policy, value,
    next-value, TD-error, bounder, and acceptance leaves.
    """
    return 4 * n_actions + 21


def _actor_critic_update_working_set_bytes(n_actions: int, feature_dim: int) -> int:
    """Source persist, proposed persist, committed persist, and returned extras."""
    return 3 * _actor_critic_persistent_bytes(
        n_actions, feature_dim
    ) + _actor_critic_update_result_extras_bytes(n_actions)


def _preflight_actor_critic_update_working_set(n_actions: int, feature_dim: int) -> None:
    """Reject an update envelope the host cannot name in signed int32."""
    if _actor_critic_update_working_set_bytes(n_actions, feature_dim) > _INT32_MAX:
        raise ValueError(
            "actor-critic update working set byte count must fit signed int32"
        )


def _require_continuous_state_resources(action_dim: int, feature_dim: int) -> None:
    state_scalars = 2 * action_dim * feature_dim + 5 * action_dim + 3 * feature_dim + 5
    state_bytes = 8 * action_dim * feature_dim + 20 * action_dim + 12 * feature_dim + 20
    if state_scalars > _INT32_MAX or state_bytes > _INT32_MAX:
        raise ValueError("derived continuous actor-critic state exceeds the signed-int32 budget")


def _continuous_actor_critic_persistent_bytes(action_dim: int, feature_dim: int) -> int:
    """Named persist already counted inside ``_require_continuous_state_resources``."""
    return 8 * action_dim * feature_dim + 20 * action_dim + 12 * feature_dim + 20


def _continuous_actor_critic_update_result_extras_bytes(action_dim: int) -> int:
    """Returned ``ContinuousActorCriticUpdateResult`` extras excluding persist.

    Nested ``state`` is persist and is already counted in the simultaneous
    persist copies. These extras are the published action, mean, sigma, value,
    next-value, TD-error, bounder, and acceptance leaves.
    """
    return 12 * action_dim + 17


def _continuous_actor_critic_update_working_set_bytes(
    action_dim: int, feature_dim: int
) -> int:
    """Source persist, proposed persist, committed persist, and returned extras."""
    return 3 * _continuous_actor_critic_persistent_bytes(
        action_dim, feature_dim
    ) + _continuous_actor_critic_update_result_extras_bytes(action_dim)


def _preflight_continuous_actor_critic_update_working_set(
    action_dim: int, feature_dim: int
) -> None:
    """Reject an update envelope the host cannot name in signed int32."""
    if _continuous_actor_critic_update_working_set_bytes(action_dim, feature_dim) > _INT32_MAX:
        raise ValueError(
            "continuous actor-critic update working set byte count must fit signed int32"
        )


def _require_discrete_scan_resources(
    *, n_actions: int, feature_dim: int, num_steps: int
) -> None:
    """Preflight a backend-independent logical upper bound for the complete scan."""
    # State counts the typed key as its two physical uint32 words. Inputs count
    # both observation matrices and the materialized reward, terminal,
    # discount, and action vectors. Outputs count action, policy, value,
    # TD-error, and acceptance rows. The scan-body envelope is independent of
    # num_steps because lax.scan reuses it: eight additional state-sized
    # buffers cover updated/held/proposed/final states and full-tree predicates;
    # the remaining terms cover every matrix/vector-width gradient, trace,
    # step, policy, selection, and scalar temporary. Charging every temporary
    # as four bytes also upper-bounds int32, uint32, and bool predicates.
    state_scalars = 2 * n_actions * feature_dim + 3 * feature_dim + 2 * n_actions + 6
    state_bytes = 8 * n_actions * feature_dim + 12 * feature_dim + 8 * n_actions + 24
    input_scalars = num_steps * (2 * feature_dim + 4)
    input_bytes = num_steps * (8 * feature_dim + 13)
    output_scalars = num_steps * (n_actions + 4)
    output_bytes = num_steps * (4 * n_actions + 13)
    temporary_scalars = (
        8 * state_scalars
        + 12 * n_actions * feature_dim
        + 24 * n_actions
        + 16 * feature_dim
        + 64
    )
    # The caller retains the initial carry while ``lax.scan`` materialises the
    # returned final carry. Count both state trees in addition to the reusable
    # source-level update/select temporary envelope.
    total_scalars = 2 * state_scalars + input_scalars + output_scalars + temporary_scalars
    total_bytes = 2 * state_bytes + input_bytes + output_bytes + 4 * temporary_scalars
    if total_scalars > _INT32_MAX or total_bytes > _INT32_MAX:
        raise ValueError("derived actor-critic scan working set exceeds the signed-int32 budget")


def _require_continuous_scan_resources(
    *, action_dim: int, feature_dim: int, num_steps: int
) -> None:
    """Preflight a backend-independent logical upper bound for the complete scan."""
    # This is the continuous analogue of _require_discrete_scan_resources:
    # action inputs and action/mean/sigma outputs have action_dim width, while
    # the reusable workspace includes Gaussian-policy gradients and samples.
    state_scalars = 2 * action_dim * feature_dim + 5 * action_dim + 3 * feature_dim + 5
    state_bytes = 8 * action_dim * feature_dim + 20 * action_dim + 12 * feature_dim + 20
    input_scalars = num_steps * (2 * feature_dim + action_dim + 3)
    input_bytes = num_steps * (8 * feature_dim + 4 * action_dim + 9)
    output_scalars = num_steps * (3 * action_dim + 3)
    output_bytes = num_steps * (12 * action_dim + 9)
    temporary_scalars = (
        8 * state_scalars
        + 12 * action_dim * feature_dim
        + 32 * action_dim
        + 16 * feature_dim
        + 64
    )
    # Retain both carry trees and add the reusable update/select envelope.
    total_scalars = 2 * state_scalars + input_scalars + output_scalars + temporary_scalars
    total_bytes = 2 * state_bytes + input_bytes + output_bytes + 4 * temporary_scalars
    if total_scalars > _INT32_MAX or total_bytes > _INT32_MAX:
        raise ValueError(
            "derived continuous actor-critic scan working set exceeds the signed-int32 budget"
        )


def _array(name: str, value: object, shape: tuple[int, ...], dtype: Any) -> Array:
    if shape == () and dtype == jnp.bool_ and type(value) is bool:
        return jnp.asarray(value, dtype=dtype)
    if shape == () and dtype == jnp.float32 and type(value) in _ACTUAL_REAL_TYPES:
        return jnp.asarray(_require_float32(name, value), dtype=dtype)
    if not (type(value) is np.ndarray or isinstance(value, jax.Array)):
        raise ValueError(f"{name} must expose trusted array metadata")
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if array.dtype != jnp.dtype(dtype):
        raise ValueError(f"{name} must have dtype {jnp.dtype(dtype)}")
    return array


def _trusted_shape(name: str, value: object) -> tuple[int, ...]:
    actual_type = type(value)
    if not (
        actual_type is np.ndarray
        or issubclass(
            actual_type,
            (jax.Array, jax.core.Tracer, jax.ShapeDtypeStruct, jax.core.ShapedArray),
        )
    ):
        raise ValueError(f"{name} must expose trusted array metadata")
    return tuple(cast(Array, value).shape)


def _checked_terminated(name: str, value: object) -> Array:
    """Validate a ``terminated`` flag array dtype before boolean coercion."""
    actual_type = type(value)
    if not (
        actual_type is np.ndarray
        or issubclass(
            actual_type,
            (jax.Array, jax.core.Tracer, jax.ShapeDtypeStruct, jax.core.ShapedArray),
        )
    ):
        raise ValueError(f"{name} must expose trusted array metadata")
    trusted = cast(Array, value)
    if trusted.dtype not in (jnp.dtype(jnp.bool_), jnp.dtype(jnp.float32)):
        raise TypeError(f"{name} must have dtype bool or float32")
    return trusted


def _validated_bounder_result(
    name: str,
    result: object,
    templates: tuple[Array, ...],
) -> tuple[tuple[Array, ...], Array]:
    """Validate one third-party bounder result before using it in state arithmetic."""
    if type(result) is not tuple or len(result) != 2:
        raise ValueError(f"{name} bounder result must be a (steps, metric) tuple")
    steps, metric = result
    if type(steps) is not tuple or len(steps) != len(templates):
        raise ValueError(f"{name} bounder steps must match the parameter tree")
    for step, template in zip(steps, templates, strict=True):
        if (
            not isinstance(step, jax.Array)
            or tuple(step.shape) != tuple(template.shape)
            or step.dtype != template.dtype
        ):
            raise ValueError(f"{name} bounder steps must match parameter shapes and dtypes")
    if (
        not isinstance(metric, jax.Array)
        or tuple(metric.shape) != ()
        or metric.dtype != jnp.float32
    ):
        raise ValueError(f"{name} bounder metric must be a scalar float32 JAX array")
    return steps, metric


def _require_key(key: object) -> Array:
    if not isinstance(key, jax.Array):
        raise ValueError("key must be a Threefry JAX key")
    try:
        data = jr.key_data(cast(Any, key))
    except Exception as error:
        raise ValueError("key must be a Threefry JAX key") from error
    if data.shape != (2,) or data.dtype != jnp.uint32:
        raise ValueError("key must be a Threefry JAX key")
    return key


def _require_action_bound(name: str, value: object) -> float:
    if type(value) not in _ACTUAL_REAL_TYPES | {Fraction}:
        raise ValueError(f"{name} must be a finite real number")
    try:
        host = float(cast(Any, value))
        narrowed = round_real_to_float32(cast(Any, value))
    except Exception as error:
        raise ValueError(f"{name} must be finite when set") from error
    if not math.isfinite(host):
        raise ValueError(f"{name} must be finite when set")
    if not math.isfinite(narrowed):
        raise ValueError(f"{name} must remain finite once narrowed to float32")
    return host if type(value) is float else narrowed


@dataclasses.dataclass(frozen=True)
class ActorCriticConfig:
    """Configuration for a linear softmax actor-critic agent.

    Attributes:
        n_actions: Number of discrete actions.
        gamma: Discount factor.
        actor_step_size: Step-size for policy parameters.
        critic_step_size: Step-size for value parameters.
        actor_lamda: Eligibility trace decay for the actor.
        critic_lamda: Eligibility trace decay for the critic.
        temperature: Softmax temperature. Values below 1 sharpen the policy.
    """

    n_actions: int
    gamma: float = 0.99
    actor_step_size: float = 0.01
    critic_step_size: float = 0.05
    actor_lamda: float = 0.9
    critic_lamda: float = 0.9
    temperature: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_actions",
            _require_int32("n_actions", self.n_actions, minimum=1),
        )
        object.__setattr__(
            self,
            "gamma",
            _require_float32(
                "gamma", self.gamma, lower=0.0, upper=1.0, preserve_nonzero=True
            ),
        )
        object.__setattr__(
            self,
            "actor_step_size",
            _require_float32(
                "actor_step_size", self.actor_step_size, lower=0.0, preserve_nonzero=True
            ),
        )
        object.__setattr__(
            self,
            "critic_step_size",
            _require_float32(
                "critic_step_size", self.critic_step_size, lower=0.0, preserve_nonzero=True
            ),
        )
        object.__setattr__(
            self,
            "actor_lamda",
            _require_float32(
                "actor_lamda", self.actor_lamda, lower=0.0, upper=1.0, preserve_nonzero=True
            ),
        )
        object.__setattr__(
            self,
            "critic_lamda",
            _require_float32(
                "critic_lamda", self.critic_lamda, lower=0.0, upper=1.0, preserve_nonzero=True
            ),
        )
        object.__setattr__(
            self,
            "temperature",
            _require_float32("temperature", self.temperature, positive=True),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize this configuration to a dictionary."""
        return {
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "actor_step_size": self.actor_step_size,
            "critic_step_size": self.critic_step_size,
            "actor_lamda": self.actor_lamda,
            "critic_lamda": self.critic_lamda,
            "temperature": self.temperature,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ActorCriticConfig:
        """Reconstruct an ``ActorCriticConfig`` from a dictionary."""
        fields = frozenset(field.name for field in dataclasses.fields(cls))
        payload = _serialized_mapping("config", config, fields=fields)
        if type(payload["n_actions"]) is not int or any(
            type(payload[name]) is not float for name in fields - {"n_actions"}
        ):
            raise ValueError("serialized config scalars must be exact JSON numbers")
        return cls(**payload)


@chex.dataclass(frozen=True)
class ActorCriticState:
    """Immutable state for a linear actor-critic agent.

    Attributes:
        actor_weights: Policy weight matrix, shape ``(n_actions, feature_dim)``.
        actor_bias: Policy bias vector, shape ``(n_actions,)``.
        critic_weights: Value weight vector, shape ``(feature_dim,)``.
        critic_bias: Scalar value bias.
        actor_trace_weights: Eligibility trace for actor weights.
        actor_trace_bias: Eligibility trace for actor bias.
        critic_trace_weights: Eligibility trace for critic weights.
        critic_trace_bias: Eligibility trace for critic bias.
        last_observation: Previous observation ``s_t``.
        last_action: Previous action ``a_t``.
        rng_key: Random key used for action sampling.
        step_count: Number of update steps taken.
    """

    actor_weights: Float[Array, "n_actions feature_dim"]
    actor_bias: Float[Array, " n_actions"]
    critic_weights: Float[Array, " feature_dim"]
    critic_bias: Float[Array, ""]
    actor_trace_weights: Float[Array, "n_actions feature_dim"]
    actor_trace_bias: Float[Array, " n_actions"]
    critic_trace_weights: Float[Array, " feature_dim"]
    critic_trace_bias: Float[Array, ""]
    last_observation: Float[Array, " feature_dim"]
    last_action: Int[Array, ""]
    rng_key: Array
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class ActorCriticUpdateResult:
    """Result from one actor-critic transition update.

    Attributes:
        state: Updated agent state.
        action: Next action selected for the new observation.
        policy: Policy probabilities at the new observation.
        value: Value estimate at the previous observation.
        next_value: Value estimate at the new observation.
        td_error: One-step TD error.
        bound_metric: Mean bounder metric, or 1.0 when no bounder is used.
    """

    state: ActorCriticState
    action: Int[Array, ""]
    policy: Float[Array, " n_actions"]
    value: Float[Array, ""]
    next_value: Float[Array, ""]
    td_error: Float[Array, ""]
    bound_metric: Float[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ActorCriticArrayResult:
    """Result from scan-based actor-critic learning on arrays.

    Attributes:
        state: Final agent state.
        actions: Per-step actions, shape ``(num_steps,)``.
        policies: Per-step pre-update agent policy probabilities at
            ``observations[t]``, shape ``(num_steps, n_actions)``. When actions
            are sampled by this runner, this is the distribution that produced
            ``actions[t]``. When fixed actions are supplied, this is the current
            agent policy evaluated at that state, not provenance for the
            external behavior policy.
        values: Per-step previous-state value estimates, shape ``(num_steps,)``.
        td_errors: Per-step TD errors, shape ``(num_steps,)``.
    """

    state: ActorCriticState
    actions: Int[Array, " num_steps"]
    policies: Float[Array, "num_steps n_actions"]
    values: Float[Array, " num_steps"]
    td_errors: Float[Array, " num_steps"]
    updates_applied: Bool[Array, " num_steps"]


class ActorCriticAgent:
    """Linear actor-critic agent with a discrete softmax policy.

    The actor is a softmax over linear logits and the critic is a scalar
    linear value function. Both components maintain accumulating eligibility
    traces and update at every time step from the same TD error.

    The implemented objective is the continuing or episodic AC(lambda)
    semi-gradient update. For transition ``S_t, A_t, R_{t+1}, S_{t+1}``, the
    critic forms ``delta_t = R_{t+1} + gamma_t V(S_{t+1}) - V(S_t)`` and
    updates value parameters along accumulating traces
    ``e^v_t = gamma_t lambda_v e^v_{t-1} + grad V(S_t)``. The actor updates
    linear softmax logits in the policy-gradient direction
    ``delta_t e^pi_t``, with
    ``e^pi_t = gamma_t lambda_pi e^pi_{t-1} + grad log pi(A_t | S_t)``.
    Because logits are divided by ``temperature`` before the softmax,
    ``grad log pi`` includes the corresponding ``1 / temperature`` factor.
    """

    def __init__(
        self,
        config: ActorCriticConfig,
        bounder: Bounder | None = None,
    ):
        """Initialize the actor-critic agent.

        Args:
            config: Actor-critic hyperparameters.
            bounder: Optional update bounder compatible with the framework
                ``Bounder`` ABC. When present, actor and critic proposed steps
                are bounded independently using the TD error.
        """
        if type(config) is not ActorCriticConfig:
            raise ValueError("config must be an ActorCriticConfig")
        if bounder is not None and not isinstance(bounder, Bounder):
            raise ValueError("bounder must be a Bounder or None")
        self._config = config
        self._bounder = bounder

    @property
    def config(self) -> ActorCriticConfig:
        """Actor-critic configuration."""
        return self._config

    @property
    def bounder(self) -> Bounder | None:
        """Optional update bounder."""
        return self._bounder

    def to_config(self) -> dict[str, Any]:
        """Serialize this agent to a dictionary."""
        return {
            "type": "ActorCriticAgent",
            "config": self._config.to_config(),
            "bounder": self._bounder.to_config() if self._bounder is not None else None,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ActorCriticAgent:
        """Reconstruct an ``ActorCriticAgent`` from a dictionary."""
        payload = _serialized_mapping(
            "config", config, fields=frozenset({"type", "config", "bounder"})
        )
        if type(payload["type"]) is not str or payload.pop("type") != cls.__name__:
            raise ValueError("config type differs")
        ac_config = ActorCriticConfig.from_config(payload.pop("config"))
        bounder_config = payload.pop("bounder")
        if bounder_config is not None and not issubclass(type(bounder_config), Mapping):
            raise ValueError("serialized bounder must be a mapping or None")
        bounder = bounder_from_config(bounder_config) if bounder_config is not None else None
        return cls(config=ac_config, bounder=bounder)

    def init(self, feature_dim: int, key: Array) -> ActorCriticState:
        """Initialize actor and critic state.

        Args:
            feature_dim: Input feature dimension.
            key: JAX random key.

        Returns:
            Initial immutable actor-critic state.
        """
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        _require_discrete_state_resources(self._config.n_actions, feature_dim)
        _preflight_actor_critic_update_working_set(self._config.n_actions, feature_dim)
        key = _require_key(key)
        zeros_actor = jnp.zeros((self._config.n_actions, feature_dim), dtype=jnp.float32)
        zeros_policy_bias = jnp.zeros((self._config.n_actions,), dtype=jnp.float32)
        zeros_critic = jnp.zeros((feature_dim,), dtype=jnp.float32)
        return ActorCriticState(  # type: ignore[call-arg]
            actor_weights=zeros_actor,
            actor_bias=zeros_policy_bias,
            critic_weights=zeros_critic,
            critic_bias=jnp.array(0.0, dtype=jnp.float32),
            actor_trace_weights=zeros_actor,
            actor_trace_bias=zeros_policy_bias,
            critic_trace_weights=zeros_critic,
            critic_trace_bias=jnp.array(0.0, dtype=jnp.float32),
            last_observation=jnp.zeros((feature_dim,), dtype=jnp.float32),
            last_action=jnp.array(-1, dtype=jnp.int32),
            rng_key=key,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def _feature_dim(self, state: ActorCriticState) -> int:
        if type(state) is not ActorCriticState:
            raise ValueError("state must be an ActorCriticState")
        n_actions = self._config.n_actions
        if state.actor_weights.ndim != 2 or state.actor_weights.shape[0] != n_actions:
            raise ValueError("state.actor_weights must match n_actions and feature_dim")
        feature_dim = state.actor_weights.shape[1]
        matrix_shape = (n_actions, feature_dim)
        for name in ("actor_weights", "actor_trace_weights"):
            leaf = getattr(state, name)
            if leaf.shape != matrix_shape or leaf.dtype != jnp.float32:
                raise ValueError(f"state.{name} must have shape {matrix_shape} and dtype float32")
        for name in ("actor_bias", "actor_trace_bias"):
            leaf = getattr(state, name)
            if leaf.shape != (n_actions,) or leaf.dtype != jnp.float32:
                raise ValueError(f"state.{name} must have shape ({n_actions},) and dtype float32")
        for name in ("critic_weights", "critic_trace_weights", "last_observation"):
            leaf = getattr(state, name)
            if leaf.shape != (feature_dim,) or leaf.dtype != jnp.float32:
                raise ValueError(f"state.{name} must have shape ({feature_dim},) and dtype float32")
        for name in ("critic_bias", "critic_trace_bias"):
            leaf = getattr(state, name)
            if leaf.shape != () or leaf.dtype != jnp.float32:
                raise ValueError(f"state.{name} must be a scalar float32")
        if state.last_action.shape != () or state.last_action.dtype != jnp.int32:
            raise ValueError("state.last_action must be a scalar int32")
        if state.step_count.shape != () or state.step_count.dtype != jnp.int32:
            raise ValueError("state.step_count must be a scalar int32")
        _require_key(state.rng_key)
        return feature_dim

    def _observation(self, state: ActorCriticState, value: object) -> Array:
        return _array("observation", value, (self._feature_dim(state),), jnp.float32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy(
        self,
        state: ActorCriticState,
        observation: Array,
    ) -> Float[Array, " n_actions"]:
        """Compute softmax action probabilities for one observation."""
        observation = self._observation(state, observation)
        logits = state.actor_weights @ observation + state.actor_bias
        return jax.nn.softmax(logits / self._config.temperature)

    @functools.partial(jax.jit, static_argnums=(0,))
    def value(self, state: ActorCriticState, observation: Array) -> Float[Array, ""]:
        """Compute the critic value estimate for one observation."""
        observation = self._observation(state, observation)
        return jnp.dot(state.critic_weights, observation) + state.critic_bias

    @functools.partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        state: ActorCriticState,
        observation: Array,
    ) -> tuple[Int[Array, ""], Array, Float[Array, " n_actions"]]:
        """Sample one action from the current softmax policy.

        Args:
            state: Current agent state.
            observation: Input feature vector.

        Returns:
            Tuple ``(action, new_rng_key, probabilities)``.
        """
        observation = self._observation(state, observation)
        key, sample_key = jr.split(state.rng_key)
        probs = self.policy(state, observation)
        action = jr.categorical(sample_key, jnp.log(jnp.maximum(probs, 1e-8))).astype(jnp.int32)
        return action, key, probs

    @functools.partial(jax.jit, static_argnums=(0,))
    def start(
        self,
        state: ActorCriticState,
        observation: Array,
    ) -> tuple[ActorCriticState, Int[Array, ""], Float[Array, " n_actions"]]:
        """Select and store the first action for a new stream or episode."""
        observation = self._observation(state, observation)
        action, key, probs = self.select_action(state, observation)
        new_state = state.replace(  # type: ignore[attr-defined]
            last_observation=observation,
            last_action=action,
            rng_key=key,
        )
        return new_state, action, probs

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: ActorCriticState,
        reward: Array,
        observation: Array,
        terminated: Array | None = None,
        discount: Array | None = None,
    ) -> ActorCriticUpdateResult:
        """Update actor and critic from one transition.

        The transition is ``(state.last_observation, state.last_action,
        reward, observation)`` plus either a scalar transition ``discount`` or
        the legacy ``terminated`` flag. A next action is sampled and stored in
        the returned state for the following update.

        Args:
            state: Current agent state with a valid previous observation/action.
            reward: Scalar reward.
            observation: Next observation.
            terminated: Backward-compatible scalar terminal flag. Non-zero
                maps to transition discount ``0``; false maps to
                ``config.gamma``. Ignored when ``discount`` is provided.
            discount: Optional scalar per-transition discount ``gamma_t``.
                Use this for continuing logs, variable discounts, time-limit
                truncation semantics, and pre-collected trajectories.

        Returns:
            ``ActorCriticUpdateResult`` containing the updated state and metrics.
        """
        observation = self._observation(state, observation)
        reward = _array("reward", reward, (), jnp.float32)
        if terminated is not None:
            terminated = _array("terminated", terminated, (), jnp.bool_)
        if discount is not None:
            discount = _array("discount", discount, (), jnp.float32)
        cfg = self._config
        prev_obs = state.last_observation
        action = state.last_action

        old_policy = self.policy(state, prev_obs)
        value = self.value(state, prev_obs)
        next_value = self.value(state, observation)
        if discount is None:
            if terminated is None:
                discount = jnp.array(cfg.gamma, dtype=jnp.float32)
            else:
                discount = jnp.where(terminated, 0.0, cfg.gamma)
        discount = jnp.asarray(discount, dtype=jnp.float32)
        # Terminal 0 * inf V(s') is NaN and would freeze the critic.
        bootstrap = jnp.where(discount == 0.0, jnp.zeros_like(next_value), discount * next_value)
        td_error = reward + bootstrap - value

        one_hot = jax.nn.one_hot(action, cfg.n_actions, dtype=jnp.float32)
        actor_grad_bias = (one_hot - old_policy) / cfg.temperature
        actor_grad_weights = actor_grad_bias[:, None] * prev_obs[None, :]

        actor_decay = discount * cfg.actor_lamda
        critic_decay = discount * cfg.critic_lamda
        actor_trace_weights = (
            jnp.where(
                actor_decay == 0.0,
                jnp.zeros_like(state.actor_trace_weights),
                actor_decay * state.actor_trace_weights,
            )
            + actor_grad_weights
        )
        actor_trace_bias = (
            jnp.where(
                actor_decay == 0.0,
                jnp.zeros_like(state.actor_trace_bias),
                actor_decay * state.actor_trace_bias,
            )
            + actor_grad_bias
        )
        critic_trace_weights = (
            jnp.where(
                critic_decay == 0.0,
                jnp.zeros_like(state.critic_trace_weights),
                critic_decay * state.critic_trace_weights,
            )
            + prev_obs
        )
        critic_trace_bias = (
            jnp.where(
                critic_decay == 0.0,
                jnp.zeros_like(state.critic_trace_bias),
                critic_decay * state.critic_trace_bias,
            )
            + 1.0
        )

        actor_steps: tuple[Array, ...] = (
            cfg.actor_step_size * actor_trace_weights,
            cfg.actor_step_size * actor_trace_bias,
        )
        critic_steps: tuple[Array, ...] = (
            cfg.critic_step_size * critic_trace_weights,
            cfg.critic_step_size * critic_trace_bias,
        )
        actor_metric = jnp.array(1.0, dtype=jnp.float32)
        critic_metric = jnp.array(1.0, dtype=jnp.float32)
        if self._bounder is not None:
            actor_steps, actor_metric = _validated_bounder_result(
                "actor",
                self._bounder.bound(
                    actor_steps,
                    td_error,
                    (state.actor_weights, state.actor_bias),
                ),
                actor_steps,
            )
            critic_steps, critic_metric = _validated_bounder_result(
                "critic",
                self._bounder.bound(
                    critic_steps,
                    td_error,
                    (state.critic_weights, state.critic_bias),
                ),
                critic_steps,
            )
        actor_steps = tuple(td_error * step for step in actor_steps)
        critic_steps = tuple(td_error * step for step in critic_steps)

        carry_traces = discount != 0.0
        stored_actor_trace_weights = jnp.where(
            carry_traces, actor_trace_weights, jnp.zeros_like(actor_trace_weights)
        )
        stored_actor_trace_bias = jnp.where(
            carry_traces, actor_trace_bias, jnp.zeros_like(actor_trace_bias)
        )
        stored_critic_trace_weights = jnp.where(
            carry_traces, critic_trace_weights, jnp.zeros_like(critic_trace_weights)
        )
        stored_critic_trace_bias = jnp.where(
            carry_traces, critic_trace_bias, jnp.zeros_like(critic_trace_bias)
        )
        updated = state.replace(  # type: ignore[attr-defined]
            actor_weights=state.actor_weights + actor_steps[0],
            actor_bias=state.actor_bias + actor_steps[1],
            critic_weights=state.critic_weights + critic_steps[0],
            critic_bias=state.critic_bias + critic_steps[1],
            actor_trace_weights=stored_actor_trace_weights,
            actor_trace_bias=stored_actor_trace_bias,
            critic_trace_weights=stored_critic_trace_weights,
            critic_trace_bias=stored_critic_trace_bias,
            step_count=jnp.minimum(state.step_count, _INT32_MAX - 1) + 1,
        )
        # Reject the complete transition if its inputs or proposed persistent
        # state are non-finite. The result carries the rejection explicitly.
        inputs_valid = (
            jnp.isfinite(jnp.squeeze(reward))
            & jnp.all(jnp.isfinite(observation))
            & jnp.isfinite(td_error)
            & jnp.isfinite(discount)
            & (discount >= 0.0)
            & (discount <= 1.0)
            & (action >= 0)
            & (action < cfg.n_actions)
            & jnp.all(jnp.isfinite(old_policy))
            & jnp.isfinite(value)
            & ((discount == 0.0) | jnp.isfinite(next_value))
            & jnp.isfinite(actor_metric)
            & jnp.isfinite(critic_metric)
            & (state.step_count >= 0)
        )
        candidate_ok = (
            inputs_valid & _floating_tree_is_finite(state) & _floating_tree_is_finite(updated)
        )
        held = jax.lax.cond(
            candidate_ok,
            lambda: updated,
            lambda: state,
        )
        safe_observation = jnp.where(candidate_ok, observation, state.last_observation)
        next_action, key, next_policy = self.select_action(held, safe_observation)
        proposed_final_state = held.replace(
            last_observation=observation,
            last_action=next_action,
            rng_key=key,
        )
        bound_metric = actor_metric / 2.0 + critic_metric / 2.0
        params_ok = (
            candidate_ok
            & _floating_tree_is_finite(proposed_final_state)
            & jnp.all(jnp.isfinite(next_policy))
            & (next_action >= 0)
            & (next_action < cfg.n_actions)
            & jnp.isfinite(bound_metric)
        )
        new_state = jax.lax.cond(
            params_ok,
            lambda: proposed_final_state,
            lambda: state,
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)

        return ActorCriticUpdateResult(  # type: ignore[call-arg]
            state=new_state,
            action=jnp.where(
                params_ok,
                next_action,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            policy=jnp.where(params_ok, next_policy, jnp.zeros_like(next_policy)),
            value=jnp.where(params_ok, value, zero),
            next_value=jnp.where(params_ok & jnp.isfinite(next_value), next_value, zero),
            td_error=jnp.where(params_ok, td_error, zero),
            bound_metric=jnp.where(params_ok, bound_metric, zero),
            update_applied=params_ok,
        )


def run_actor_critic_from_arrays(
    agent: ActorCriticAgent,
    state: ActorCriticState,
    observations: Float[Array, "num_steps feature_dim"],
    rewards: Float[Array, " num_steps"],
    terminated: Float[Array, " num_steps"] | None,
    next_observations: Float[Array, "num_steps feature_dim"],
    actions: Int[Array, " num_steps"] | None = None,
    discounts: Float[Array, " num_steps"] | None = None,
) -> ActorCriticArrayResult:
    """Run actor-critic updates over arrays with ``jax.lax.scan``.

    By default the scan is on-policy with respect to the current actor. At each
    row it starts from ``observations[t]``, samples/stores an action, and
    applies the transition ending at ``next_observations[t]``. When ``actions``
    is provided, those fixed behavior actions are used instead, which is the
    path intended for pre-collected logs. When ``discounts`` is provided it is
    used as the per-transition discount; otherwise ``terminated`` is mapped to
    ``0`` or ``agent.config.gamma`` for backward compatibility.

    Args:
        agent: Actor-critic agent.
        state: Initial actor-critic state.
        observations: Current observations, shape ``(num_steps, feature_dim)``.
        rewards: Rewards, shape ``(num_steps,)``.
        terminated: Terminal flags, shape ``(num_steps,)``. Required unless
            ``discounts`` is provided.
        next_observations: Next observations, shape ``(num_steps, feature_dim)``.
        actions: Optional fixed current actions, shape ``(num_steps,)``. Their
            behavior-policy probabilities are not known; returned ``policies``
            are the current agent policy evaluated before each update.
        discounts: Optional transition discounts, shape ``(num_steps,)``.

    Returns:
        ``ActorCriticArrayResult`` with final state and per-step metrics.
    """
    if type(agent) is not ActorCriticAgent:
        raise TypeError("agent must be an exact ActorCriticAgent")
    if type(state) is not ActorCriticState:
        raise TypeError("state must be an exact ActorCriticState")

    feature_dim = agent._feature_dim(state)
    observations_shape = _trusted_shape("observations", observations)
    if len(observations_shape) != 2 or not 1 <= observations_shape[0] <= _INT32_MAX:
        raise ValueError("observations must have shape (num_steps, feature_dim)")
    num_steps = observations_shape[0]
    expected_observations_shape = (num_steps, feature_dim)
    if observations_shape != expected_observations_shape or _trusted_shape(
        "next_observations", next_observations
    ) != expected_observations_shape:
        raise ValueError("observations and next_observations must match state feature_dim")
    if _trusted_shape("rewards", rewards) != (num_steps,):
        raise ValueError("rewards must have shape (num_steps,)")
    if terminated is None and discounts is None:
        raise ValueError("terminated or discounts must be provided")
    if terminated is not None and _trusted_shape("terminated", terminated) != (num_steps,):
        raise ValueError("terminated must have shape (num_steps,)")
    if discounts is not None and _trusted_shape("discounts", discounts) != (num_steps,):
        raise ValueError("discounts must have shape (num_steps,)")
    if actions is not None and _trusted_shape("actions", actions) != (num_steps,):
        raise ValueError("actions must have shape (num_steps,)")
    _require_discrete_scan_resources(
        n_actions=agent.config.n_actions,
        feature_dim=feature_dim,
        num_steps=num_steps,
    )
    observations = jnp.asarray(observations, dtype=jnp.float32)
    next_observations = jnp.asarray(next_observations, dtype=jnp.float32)
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    if terminated is not None:
        terminated = jnp.asarray(_checked_terminated("terminated", terminated), dtype=jnp.bool_)
    if discounts is not None:
        discounts = jnp.asarray(discounts, dtype=jnp.float32)
    if terminated is None:
        terminated = jnp.zeros_like(rewards, dtype=jnp.bool_)
    if discounts is None:
        discounts = jnp.where(terminated, 0.0, agent.config.gamma).astype(jnp.float32)
    if actions is None:
        actions = jnp.full_like(rewards, -1, dtype=jnp.int32)
        actions_valid = jnp.ones((num_steps,), dtype=jnp.bool_)
        use_fixed_actions = False
    else:
        actions, actions_valid = checked_integer_action_array(
            actions,
            agent.config.n_actions,
            name="actions",
            expected_shape=(num_steps,),
            range_message="actions must lie in [0, n_actions)",
        )
        use_fixed_actions = True

    def _scan_fn(
        carry: ActorCriticState,
        inputs: tuple[Array, Array, Array, Array, Array, Array],
    ) -> tuple[ActorCriticState, tuple[Array, Array, Array, Array, Array]]:
        obs, reward, term_discount, next_obs, fixed_action, fixed_action_valid = inputs
        if use_fixed_actions:
            started_state = carry.replace(  # type: ignore[attr-defined]
                last_observation=obs,
                last_action=fixed_action.astype(jnp.int32),
            )
            current_action = fixed_action.astype(jnp.int32)
            current_policy = agent.policy(started_state, obs)
        else:
            started_state, current_action, current_policy = agent.start(carry, obs)
        result = agent.update(
            started_state,
            reward,
            next_obs,
            discount=term_discount,
        )
        update_applied = result.update_applied & fixed_action_valid
        next_carry = jax.lax.cond(
            update_applied,
            lambda: result.state,
            lambda: carry,
        )
        return next_carry, (
            jnp.where(
                update_applied,
                current_action,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            jnp.where(update_applied, current_policy, jnp.zeros_like(current_policy)),
            jnp.where(update_applied, result.value, 0.0),
            jnp.where(update_applied, result.td_error, 0.0),
            update_applied,
        )

    final_state, (actions, policies, values, td_errors, updates_applied) = jax.lax.scan(
        _scan_fn,
        state,
        (observations, rewards, discounts, next_observations, actions, actions_valid),
    )
    return ActorCriticArrayResult(  # type: ignore[call-arg]
        state=final_state,
        actions=actions,
        policies=policies,
        values=values,
        td_errors=td_errors,
        updates_applied=updates_applied,
    )


# ---------------------------------------------------------------------------
# Continuous-action actor-critic (Step 4 preview)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ContinuousActorCriticConfig:
    """Configuration for a continuous-action linear actor-critic.

    The actor models a diagonal-Gaussian policy ``a ~ N(mu(s), sigma^2)`` with
    a linear mean ``mu(s) = W_mu s + b_mu`` and a per-dimension log-standard-
    deviation parameter ``log_sigma`` (state-independent). Action samples are
    optionally clipped to ``[action_low, action_high]`` after sampling. The
    critic is a scalar linear value function ``V(s) = w_v . s + b_v``. Both
    actor and critic carry their own accumulating eligibility traces and share
    the same TD error.

    Attributes:
        action_dim: Dimensionality of the continuous action vector.
        gamma: Discount factor.
        actor_step_size: Step-size for the actor mean and log-sigma parameters.
        critic_step_size: Step-size for the critic value parameters.
        actor_lamda: Eligibility trace decay for the actor.
        critic_lamda: Eligibility trace decay for the critic.
        log_sigma_init: Initial value for ``log_sigma`` per action dimension.
        log_sigma_min: Lower bound clamp on ``log_sigma`` after each update.
        log_sigma_max: Upper bound clamp on ``log_sigma`` after each update.
        action_low: Lower bound for action clipping. ``None`` disables clipping.
        action_high: Upper bound for action clipping. ``None`` disables clipping.
    """

    action_dim: int
    gamma: float = 0.99
    actor_step_size: float = 0.001
    critic_step_size: float = 0.05
    actor_lamda: float = 0.9
    critic_lamda: float = 0.9
    log_sigma_init: float = -0.5
    log_sigma_min: float = -5.0
    log_sigma_max: float = 2.0
    action_low: float | None = None
    action_high: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_dim",
            _require_int32("action_dim", self.action_dim, minimum=1),
        )
        object.__setattr__(
            self,
            "gamma",
            _require_float32(
                "gamma", self.gamma, lower=0.0, upper=1.0, preserve_nonzero=True
            ),
        )
        object.__setattr__(
            self,
            "actor_step_size",
            _require_float32(
                "actor_step_size", self.actor_step_size, lower=0.0, preserve_nonzero=True
            ),
        )
        object.__setattr__(
            self,
            "critic_step_size",
            _require_float32(
                "critic_step_size", self.critic_step_size, lower=0.0, preserve_nonzero=True
            ),
        )
        object.__setattr__(
            self,
            "actor_lamda",
            _require_float32(
                "actor_lamda", self.actor_lamda, lower=0.0, upper=1.0, preserve_nonzero=True
            ),
        )
        object.__setattr__(
            self,
            "critic_lamda",
            _require_float32(
                "critic_lamda", self.critic_lamda, lower=0.0, upper=1.0, preserve_nonzero=True
            ),
        )
        object.__setattr__(
            self,
            "log_sigma_init",
            _require_float32("log_sigma_init", self.log_sigma_init),
        )
        object.__setattr__(
            self,
            "log_sigma_min",
            _require_float32("log_sigma_min", self.log_sigma_min),
        )
        object.__setattr__(
            self,
            "log_sigma_max",
            _require_float32("log_sigma_max", self.log_sigma_max),
        )
        if self.log_sigma_min > self.log_sigma_max:
            raise ValueError("log_sigma_min must be <= log_sigma_max")
        action_low = (
            None
            if self.action_low is None
            else _require_action_bound("action_low", self.action_low)
        )
        action_high = (
            None
            if self.action_high is None
            else _require_action_bound("action_high", self.action_high)
        )
        if action_low is not None and action_high is not None and action_low > action_high:
            raise ValueError("action_low must be <= action_high")
        object.__setattr__(self, "action_low", action_low)
        object.__setattr__(self, "action_high", action_high)

    def to_config(self) -> dict[str, Any]:
        """Serialize this configuration to a dictionary."""
        return {
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "actor_step_size": self.actor_step_size,
            "critic_step_size": self.critic_step_size,
            "actor_lamda": self.actor_lamda,
            "critic_lamda": self.critic_lamda,
            "log_sigma_init": self.log_sigma_init,
            "log_sigma_min": self.log_sigma_min,
            "log_sigma_max": self.log_sigma_max,
            "action_low": self.action_low,
            "action_high": self.action_high,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ContinuousActorCriticConfig:
        """Reconstruct a ``ContinuousActorCriticConfig`` from a dictionary."""
        fields = frozenset(field.name for field in dataclasses.fields(cls))
        payload = _serialized_mapping("config", config, fields=fields)
        if type(payload["action_dim"]) is not int or any(
            payload[name] is not None and type(payload[name]) is not float
            for name in fields - {"action_dim"}
        ):
            raise ValueError("serialized config scalars must be exact JSON numbers or None")
        return cls(**payload)


@chex.dataclass(frozen=True)
class ContinuousActorCriticState:
    """Immutable state for a continuous-action linear actor-critic.

    Attributes:
        mean_weights: Mean head weights, shape ``(action_dim, feature_dim)``.
        mean_bias: Mean head bias, shape ``(action_dim,)``.
        log_sigma: Per-dimension log-standard-deviation, shape ``(action_dim,)``.
        critic_weights: Value weight vector, shape ``(feature_dim,)``.
        critic_bias: Scalar value bias.
        mean_trace_weights: Trace for mean weights.
        mean_trace_bias: Trace for mean bias.
        log_sigma_trace: Trace for ``log_sigma``.
        critic_trace_weights: Trace for critic weights.
        critic_trace_bias: Trace for critic bias.
        last_observation: Previous observation ``s_t``.
        last_action: Previous (continuous) action vector ``a_t``.
        rng_key: Random key used for action sampling.
        step_count: Number of update steps taken.
    """

    mean_weights: Float[Array, "action_dim feature_dim"]
    mean_bias: Float[Array, " action_dim"]
    log_sigma: Float[Array, " action_dim"]
    critic_weights: Float[Array, " feature_dim"]
    critic_bias: Float[Array, ""]
    mean_trace_weights: Float[Array, "action_dim feature_dim"]
    mean_trace_bias: Float[Array, " action_dim"]
    log_sigma_trace: Float[Array, " action_dim"]
    critic_trace_weights: Float[Array, " feature_dim"]
    critic_trace_bias: Float[Array, ""]
    last_observation: Float[Array, " feature_dim"]
    last_action: Float[Array, " action_dim"]
    rng_key: Array
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class ContinuousActorCriticUpdateResult:
    """Result from one continuous actor-critic transition update.

    Attributes:
        state: Updated agent state.
        action: Next action vector sampled at the new observation.
        mean: Mean of the policy at the new observation.
        sigma: Standard deviation of the policy.
        value: Value estimate at the previous observation.
        next_value: Value estimate at the new observation.
        td_error: One-step TD error.
        bound_metric: Mean bounder metric, or 1.0 when no bounder is used.
    """

    state: ContinuousActorCriticState
    action: Float[Array, " action_dim"]
    mean: Float[Array, " action_dim"]
    sigma: Float[Array, " action_dim"]
    value: Float[Array, ""]
    next_value: Float[Array, ""]
    td_error: Float[Array, ""]
    bound_metric: Float[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ContinuousActorCriticArrayResult:
    """Result from scan-based continuous actor-critic learning on arrays.

    Attributes:
        state: Final agent state.
        actions: Per-step actions, shape ``(num_steps, action_dim)``.
        means: Per-step policy means, shape ``(num_steps, action_dim)``.
        sigmas: Per-step policy standard deviations, shape ``(num_steps, action_dim)``.
        values: Per-step previous-state value estimates, shape ``(num_steps,)``.
        td_errors: Per-step TD errors, shape ``(num_steps,)``.
    """

    state: ContinuousActorCriticState
    actions: Float[Array, "num_steps action_dim"]
    means: Float[Array, "num_steps action_dim"]
    sigmas: Float[Array, "num_steps action_dim"]
    values: Float[Array, " num_steps"]
    td_errors: Float[Array, " num_steps"]
    updates_applied: Bool[Array, " num_steps"]


class ContinuousActorCriticAgent:
    """Linear continuous-action actor-critic with a diagonal-Gaussian policy.

    The actor parameterises a diagonal Gaussian
    ``pi(a | s) = N(mu(s), diag(sigma^2))`` with linear mean
    ``mu(s) = W_mu s + b_mu`` and a state-independent log-sigma vector. The
    critic is a scalar linear value function. Both components carry their own
    accumulating eligibility traces and update at every time step from the
    same TD error, mirroring the discrete ``ActorCriticAgent``.

    Policy gradient. With a Gaussian policy, the score function is

    ``grad_{mu_i} log pi(a | s) = (a_i - mu_i) / sigma_i^2``,

    ``grad_{log_sigma_i} log pi(a | s) = (a_i - mu_i)^2 / sigma_i^2 - 1``.

    These gradients enter the actor traces and are scaled by the TD error
    when applied. ``log_sigma`` is optionally clamped after each update for
    numerical stability and to prevent collapse.
    """

    def __init__(
        self,
        config: ContinuousActorCriticConfig,
        bounder: Bounder | None = None,
    ):
        """Initialize the continuous actor-critic agent.

        Args:
            config: Continuous actor-critic hyperparameters.
            bounder: Optional update bounder compatible with the framework
                ``Bounder`` ABC. When present, actor and critic proposed steps
                are bounded independently using the TD error.
        """
        if type(config) is not ContinuousActorCriticConfig:
            raise ValueError("config must be a ContinuousActorCriticConfig")
        if bounder is not None and not isinstance(bounder, Bounder):
            raise ValueError("bounder must be a Bounder or None")
        self._config = config
        self._bounder = bounder

    @property
    def config(self) -> ContinuousActorCriticConfig:
        """Continuous actor-critic configuration."""
        return self._config

    @property
    def bounder(self) -> Bounder | None:
        """Optional update bounder."""
        return self._bounder

    def to_config(self) -> dict[str, Any]:
        """Serialize this agent to a dictionary."""
        return {
            "type": "ContinuousActorCriticAgent",
            "config": self._config.to_config(),
            "bounder": self._bounder.to_config() if self._bounder is not None else None,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ContinuousActorCriticAgent:
        """Reconstruct a ``ContinuousActorCriticAgent`` from a dictionary."""
        payload = _serialized_mapping(
            "config", config, fields=frozenset({"type", "config", "bounder"})
        )
        if type(payload["type"]) is not str or payload.pop("type") != cls.__name__:
            raise ValueError("config type differs")
        ac_config = ContinuousActorCriticConfig.from_config(payload.pop("config"))
        bounder_config = payload.pop("bounder")
        if bounder_config is not None and not issubclass(type(bounder_config), Mapping):
            raise ValueError("serialized bounder must be a mapping or None")
        bounder = bounder_from_config(bounder_config) if bounder_config is not None else None
        return cls(config=ac_config, bounder=bounder)

    def init(self, feature_dim: int, key: Array) -> ContinuousActorCriticState:
        """Initialize actor and critic state.

        Args:
            feature_dim: Input feature dimension.
            key: JAX random key.

        Returns:
            Initial immutable continuous actor-critic state.
        """
        feature_dim = _require_int32("feature_dim", feature_dim, minimum=1)
        cfg = self._config
        _require_continuous_state_resources(cfg.action_dim, feature_dim)
        _preflight_continuous_actor_critic_update_working_set(cfg.action_dim, feature_dim)
        key = _require_key(key)
        zeros_mean = jnp.zeros((cfg.action_dim, feature_dim), dtype=jnp.float32)
        zeros_mean_bias = jnp.zeros((cfg.action_dim,), dtype=jnp.float32)
        log_sigma = jnp.full(
            (cfg.action_dim,),
            cfg.log_sigma_init,
            dtype=jnp.float32,
        )
        zeros_critic = jnp.zeros((feature_dim,), dtype=jnp.float32)
        return ContinuousActorCriticState(  # type: ignore[call-arg]
            mean_weights=zeros_mean,
            mean_bias=zeros_mean_bias,
            log_sigma=log_sigma,
            critic_weights=zeros_critic,
            critic_bias=jnp.array(0.0, dtype=jnp.float32),
            mean_trace_weights=zeros_mean,
            mean_trace_bias=zeros_mean_bias,
            log_sigma_trace=jnp.zeros_like(log_sigma),
            critic_trace_weights=zeros_critic,
            critic_trace_bias=jnp.array(0.0, dtype=jnp.float32),
            last_observation=jnp.zeros((feature_dim,), dtype=jnp.float32),
            last_action=jnp.zeros((cfg.action_dim,), dtype=jnp.float32),
            rng_key=key,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def _feature_dim(self, state: ContinuousActorCriticState) -> int:
        if type(state) is not ContinuousActorCriticState:
            raise ValueError("state must be a ContinuousActorCriticState")
        action_dim = self._config.action_dim
        if state.mean_weights.ndim != 2 or state.mean_weights.shape[0] != action_dim:
            raise ValueError("state.mean_weights must match action_dim and feature_dim")
        feature_dim = state.mean_weights.shape[1]
        matrix_shape = (action_dim, feature_dim)
        for name in ("mean_weights", "mean_trace_weights"):
            leaf = getattr(state, name)
            if leaf.shape != matrix_shape or leaf.dtype != jnp.float32:
                raise ValueError(f"state.{name} must have shape {matrix_shape} and dtype float32")
        for name in (
            "mean_bias",
            "log_sigma",
            "mean_trace_bias",
            "log_sigma_trace",
            "last_action",
        ):
            leaf = getattr(state, name)
            if leaf.shape != (action_dim,) or leaf.dtype != jnp.float32:
                raise ValueError(f"state.{name} must have shape ({action_dim},) and dtype float32")
        for name in ("critic_weights", "critic_trace_weights", "last_observation"):
            leaf = getattr(state, name)
            if leaf.shape != (feature_dim,) or leaf.dtype != jnp.float32:
                raise ValueError(f"state.{name} must have shape ({feature_dim},) and dtype float32")
        for name in ("critic_bias", "critic_trace_bias"):
            leaf = getattr(state, name)
            if leaf.shape != () or leaf.dtype != jnp.float32:
                raise ValueError(f"state.{name} must be a scalar float32")
        if state.step_count.shape != () or state.step_count.dtype != jnp.int32:
            raise ValueError("state.step_count must be a scalar int32")
        _require_key(state.rng_key)
        return feature_dim

    def _observation(self, state: ContinuousActorCriticState, value: object) -> Array:
        return _array("observation", value, (self._feature_dim(state),), jnp.float32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy_params(
        self,
        state: ContinuousActorCriticState,
        observation: Array,
    ) -> tuple[Float[Array, " action_dim"], Float[Array, " action_dim"]]:
        """Compute Gaussian policy mean and standard deviation for one observation."""
        observation = self._observation(state, observation)
        mean = state.mean_weights @ observation + state.mean_bias
        sigma = jnp.exp(state.log_sigma)
        return mean, sigma

    @functools.partial(jax.jit, static_argnums=(0,))
    def value(
        self,
        state: ContinuousActorCriticState,
        observation: Array,
    ) -> Float[Array, ""]:
        """Compute the critic value estimate for one observation."""
        observation = self._observation(state, observation)
        return jnp.dot(state.critic_weights, observation) + state.critic_bias

    def _maybe_clip_action(self, action: Array) -> Array:
        cfg = self._config
        if cfg.action_low is None and cfg.action_high is None:
            return action
        low = -jnp.inf if cfg.action_low is None else cfg.action_low
        high = jnp.inf if cfg.action_high is None else cfg.action_high
        return jnp.clip(action, low, high)

    @functools.partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        state: ContinuousActorCriticState,
        observation: Array,
    ) -> tuple[
        Float[Array, " action_dim"],
        Array,
        Float[Array, " action_dim"],
        Float[Array, " action_dim"],
    ]:
        """Sample one action from the current Gaussian policy.

        Args:
            state: Current agent state.
            observation: Input feature vector.

        Returns:
            Tuple ``(action, new_rng_key, mean, sigma)`` where ``action`` is
            optionally clipped to the configured action bounds.
        """
        observation = self._observation(state, observation)
        key, sample_key = jr.split(state.rng_key)
        mean, sigma = self.policy_params(state, observation)
        noise = jr.normal(sample_key, shape=mean.shape, dtype=jnp.float32)
        raw_action = mean + sigma * noise
        action = self._maybe_clip_action(raw_action)
        return action, key, mean, sigma

    @functools.partial(jax.jit, static_argnums=(0,))
    def start(
        self,
        state: ContinuousActorCriticState,
        observation: Array,
    ) -> tuple[
        ContinuousActorCriticState,
        Float[Array, " action_dim"],
        Float[Array, " action_dim"],
        Float[Array, " action_dim"],
    ]:
        """Select and store the first action for a new stream or episode."""
        observation = self._observation(state, observation)
        action, key, mean, sigma = self.select_action(state, observation)
        new_state = state.replace(  # type: ignore[attr-defined]
            last_observation=observation,
            last_action=action,
            rng_key=key,
        )
        return new_state, action, mean, sigma

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: ContinuousActorCriticState,
        reward: Array,
        observation: Array,
        terminated: Array | None = None,
        discount: Array | None = None,
    ) -> ContinuousActorCriticUpdateResult:
        """Update actor and critic from one transition.

        The transition is ``(state.last_observation, state.last_action,
        reward, observation)`` plus either a scalar transition ``discount`` or
        the legacy ``terminated`` flag. A next action is sampled and stored in
        the returned state for the following update.

        Args:
            state: Current agent state with a valid previous observation/action.
            reward: Scalar reward.
            observation: Next observation.
            terminated: Backward-compatible scalar terminal flag. Non-zero
                maps to transition discount ``0``; false maps to
                ``config.gamma``. Ignored when ``discount`` is provided.
            discount: Optional scalar per-transition discount ``gamma_t``.

        Returns:
            ``ContinuousActorCriticUpdateResult`` containing the updated state.
        """
        observation = self._observation(state, observation)
        reward = _array("reward", reward, (), jnp.float32)
        if terminated is not None:
            terminated = _array("terminated", terminated, (), jnp.bool_)
        if discount is not None:
            discount = _array("discount", discount, (), jnp.float32)
        cfg = self._config
        prev_obs = state.last_observation
        action = state.last_action

        prev_mean, prev_sigma = self.policy_params(state, prev_obs)
        value = self.value(state, prev_obs)
        next_value = self.value(state, observation)
        if discount is None:
            if terminated is None:
                discount = jnp.array(cfg.gamma, dtype=jnp.float32)
            else:
                discount = jnp.where(terminated, 0.0, cfg.gamma)
        discount = jnp.asarray(discount, dtype=jnp.float32)
        # Terminal 0 * inf V(s') is NaN and would freeze the critic.
        bootstrap = jnp.where(discount == 0.0, jnp.zeros_like(next_value), discount * next_value)
        td_error = reward + bootstrap - value

        sigma_sq = prev_sigma * prev_sigma + 1e-8
        diff = action - prev_mean
        # Gaussian score function (per-dimension):
        #   grad log pi w.r.t. mean   = diff / sigma^2
        #   grad log pi w.r.t. log_sigma = diff^2 / sigma^2 - 1
        mean_grad_bias = diff / sigma_sq
        mean_grad_weights = mean_grad_bias[:, None] * prev_obs[None, :]
        log_sigma_grad = (diff * diff) / sigma_sq - 1.0

        actor_decay = discount * cfg.actor_lamda
        critic_decay = discount * cfg.critic_lamda
        mean_trace_weights = (
            jnp.where(
                actor_decay == 0.0,
                jnp.zeros_like(state.mean_trace_weights),
                actor_decay * state.mean_trace_weights,
            )
            + mean_grad_weights
        )
        mean_trace_bias = (
            jnp.where(
                actor_decay == 0.0,
                jnp.zeros_like(state.mean_trace_bias),
                actor_decay * state.mean_trace_bias,
            )
            + mean_grad_bias
        )
        log_sigma_trace = (
            jnp.where(
                actor_decay == 0.0,
                jnp.zeros_like(state.log_sigma_trace),
                actor_decay * state.log_sigma_trace,
            )
            + log_sigma_grad
        )
        critic_trace_weights = (
            jnp.where(
                critic_decay == 0.0,
                jnp.zeros_like(state.critic_trace_weights),
                critic_decay * state.critic_trace_weights,
            )
            + prev_obs
        )
        critic_trace_bias = (
            jnp.where(
                critic_decay == 0.0,
                jnp.zeros_like(state.critic_trace_bias),
                critic_decay * state.critic_trace_bias,
            )
            + 1.0
        )

        actor_steps: tuple[Array, ...] = (
            cfg.actor_step_size * mean_trace_weights,
            cfg.actor_step_size * mean_trace_bias,
            cfg.actor_step_size * log_sigma_trace,
        )
        critic_steps: tuple[Array, ...] = (
            cfg.critic_step_size * critic_trace_weights,
            cfg.critic_step_size * critic_trace_bias,
        )
        actor_metric = jnp.array(1.0, dtype=jnp.float32)
        critic_metric = jnp.array(1.0, dtype=jnp.float32)
        if self._bounder is not None:
            actor_steps, actor_metric = _validated_bounder_result(
                "actor",
                self._bounder.bound(
                    actor_steps,
                    td_error,
                    (state.mean_weights, state.mean_bias, state.log_sigma),
                ),
                actor_steps,
            )
            critic_steps, critic_metric = _validated_bounder_result(
                "critic",
                self._bounder.bound(
                    critic_steps,
                    td_error,
                    (state.critic_weights, state.critic_bias),
                ),
                critic_steps,
            )
        actor_steps = tuple(td_error * step for step in actor_steps)
        critic_steps = tuple(td_error * step for step in critic_steps)

        carry_traces = discount != 0.0
        stored_mean_trace_weights = jnp.where(
            carry_traces, mean_trace_weights, jnp.zeros_like(mean_trace_weights)
        )
        stored_mean_trace_bias = jnp.where(
            carry_traces, mean_trace_bias, jnp.zeros_like(mean_trace_bias)
        )
        stored_log_sigma_trace = jnp.where(
            carry_traces, log_sigma_trace, jnp.zeros_like(log_sigma_trace)
        )
        stored_critic_trace_weights = jnp.where(
            carry_traces, critic_trace_weights, jnp.zeros_like(critic_trace_weights)
        )
        stored_critic_trace_bias = jnp.where(
            carry_traces, critic_trace_bias, jnp.zeros_like(critic_trace_bias)
        )
        new_log_sigma = jnp.clip(
            state.log_sigma + actor_steps[2],
            cfg.log_sigma_min,
            cfg.log_sigma_max,
        )
        updated = state.replace(  # type: ignore[attr-defined]
            mean_weights=state.mean_weights + actor_steps[0],
            mean_bias=state.mean_bias + actor_steps[1],
            log_sigma=new_log_sigma,
            critic_weights=state.critic_weights + critic_steps[0],
            critic_bias=state.critic_bias + critic_steps[1],
            mean_trace_weights=stored_mean_trace_weights,
            mean_trace_bias=stored_mean_trace_bias,
            log_sigma_trace=stored_log_sigma_trace,
            critic_trace_weights=stored_critic_trace_weights,
            critic_trace_bias=stored_critic_trace_bias,
            step_count=jnp.minimum(state.step_count, _INT32_MAX - 1) + 1,
        )
        inputs_valid = (
            jnp.isfinite(jnp.squeeze(reward))
            & jnp.all(jnp.isfinite(observation))
            & jnp.isfinite(td_error)
            & jnp.isfinite(discount)
            & (discount >= 0.0)
            & (discount <= 1.0)
            & jnp.all(jnp.isfinite(prev_mean))
            & jnp.all(jnp.isfinite(prev_sigma))
            & jnp.isfinite(value)
            & ((discount == 0.0) | jnp.isfinite(next_value))
            & jnp.isfinite(actor_metric)
            & jnp.isfinite(critic_metric)
            & (state.step_count >= 0)
        )
        candidate_ok = (
            inputs_valid & _floating_tree_is_finite(state) & _floating_tree_is_finite(updated)
        )
        held = jax.lax.cond(
            candidate_ok,
            lambda: updated,
            lambda: state,
        )
        safe_observation = jnp.where(candidate_ok, observation, state.last_observation)
        next_action, key, next_mean, next_sigma = self.select_action(held, safe_observation)
        proposed_final_state = held.replace(
            last_observation=observation,
            last_action=next_action,
            rng_key=key,
        )
        bound_metric = actor_metric / 2.0 + critic_metric / 2.0
        params_ok = (
            candidate_ok
            & _floating_tree_is_finite(proposed_final_state)
            & jnp.all(jnp.isfinite(next_action))
            & jnp.all(jnp.isfinite(next_mean))
            & jnp.all(jnp.isfinite(next_sigma))
            & jnp.isfinite(bound_metric)
        )
        new_state = jax.lax.cond(
            params_ok,
            lambda: proposed_final_state,
            lambda: state,
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)

        return ContinuousActorCriticUpdateResult(  # type: ignore[call-arg]
            state=new_state,
            action=jnp.where(params_ok, next_action, jnp.zeros_like(next_action)),
            mean=jnp.where(params_ok, next_mean, jnp.zeros_like(next_mean)),
            sigma=jnp.where(params_ok, next_sigma, jnp.zeros_like(next_sigma)),
            value=jnp.where(params_ok, value, zero),
            next_value=jnp.where(params_ok & jnp.isfinite(next_value), next_value, zero),
            td_error=jnp.where(params_ok, td_error, zero),
            bound_metric=jnp.where(params_ok, bound_metric, zero),
            update_applied=params_ok,
        )


def run_continuous_actor_critic_from_arrays(
    agent: ContinuousActorCriticAgent,
    state: ContinuousActorCriticState,
    observations: Float[Array, "num_steps feature_dim"],
    rewards: Float[Array, " num_steps"],
    terminated: Float[Array, " num_steps"] | None,
    next_observations: Float[Array, "num_steps feature_dim"],
    actions: Float[Array, "num_steps action_dim"] | None = None,
    discounts: Float[Array, " num_steps"] | None = None,
) -> ContinuousActorCriticArrayResult:
    """Run continuous actor-critic updates over arrays with ``jax.lax.scan``.

    Mirrors :func:`run_actor_critic_from_arrays` for the continuous-action
    variant. By default the scan is on-policy with respect to the current
    actor; pass ``actions`` to use fixed behavior actions.

    Args:
        agent: Continuous actor-critic agent.
        state: Initial agent state.
        observations: Current observations, shape ``(num_steps, feature_dim)``.
        rewards: Rewards, shape ``(num_steps,)``.
        terminated: Terminal flags, shape ``(num_steps,)``. Required unless
            ``discounts`` is provided.
        next_observations: Next observations, shape ``(num_steps, feature_dim)``.
        actions: Optional fixed current actions, shape ``(num_steps, action_dim)``.
        discounts: Optional transition discounts, shape ``(num_steps,)``.

    Returns:
        ``ContinuousActorCriticArrayResult`` with final state and per-step metrics.
    """
    if type(agent) is not ContinuousActorCriticAgent:
        raise TypeError("agent must be an exact ContinuousActorCriticAgent")
    if type(state) is not ContinuousActorCriticState:
        raise TypeError("state must be an exact ContinuousActorCriticState")

    feature_dim = agent._feature_dim(state)
    observations_shape = _trusted_shape("observations", observations)
    if len(observations_shape) != 2 or not 1 <= observations_shape[0] <= _INT32_MAX:
        raise ValueError("observations must have shape (num_steps, feature_dim)")
    num_steps = observations_shape[0]
    expected_observations_shape = (num_steps, feature_dim)
    if observations_shape != expected_observations_shape or _trusted_shape(
        "next_observations", next_observations
    ) != expected_observations_shape:
        raise ValueError("observations and next_observations must match state feature_dim")
    if _trusted_shape("rewards", rewards) != (num_steps,):
        raise ValueError("rewards must have shape (num_steps,)")
    action_dim = agent.config.action_dim
    if terminated is None and discounts is None:
        raise ValueError("terminated or discounts must be provided")
    if terminated is not None and _trusted_shape("terminated", terminated) != (num_steps,):
        raise ValueError("terminated must have shape (num_steps,)")
    if discounts is not None and _trusted_shape("discounts", discounts) != (num_steps,):
        raise ValueError("discounts must have shape (num_steps,)")
    if actions is not None and _trusted_shape("actions", actions) != (num_steps, action_dim):
        raise ValueError("actions must have shape (num_steps, action_dim)")
    _require_continuous_scan_resources(
        action_dim=action_dim,
        feature_dim=feature_dim,
        num_steps=num_steps,
    )
    observations = jnp.asarray(observations, dtype=jnp.float32)
    next_observations = jnp.asarray(next_observations, dtype=jnp.float32)
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    if terminated is not None:
        terminated = jnp.asarray(_checked_terminated("terminated", terminated), dtype=jnp.bool_)
    if discounts is not None:
        discounts = jnp.asarray(discounts, dtype=jnp.float32)
    if terminated is None:
        terminated = jnp.zeros_like(rewards, dtype=jnp.bool_)
    if discounts is None:
        discounts = jnp.where(terminated, 0.0, agent.config.gamma).astype(jnp.float32)
    if actions is None:
        actions = jnp.zeros((rewards.shape[0], action_dim), dtype=jnp.float32)
        use_fixed_actions = False
    else:
        actions = jnp.asarray(actions, dtype=jnp.float32)
        use_fixed_actions = True

    def _scan_fn(
        carry: ContinuousActorCriticState,
        inputs: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[
        ContinuousActorCriticState,
        tuple[Array, Array, Array, Array, Array, Array],
    ]:
        obs, reward, term_discount, next_obs, fixed_action = inputs
        if use_fixed_actions:
            started_state = carry.replace(  # type: ignore[attr-defined]
                last_observation=obs,
                last_action=fixed_action.astype(jnp.float32),
            )
            current_action = fixed_action.astype(jnp.float32)
            current_mean, current_sigma = agent.policy_params(started_state, obs)
        else:
            started_state, current_action, current_mean, current_sigma = agent.start(carry, obs)
        result = agent.update(
            started_state,
            reward,
            next_obs,
            discount=term_discount,
        )
        next_carry = jax.lax.cond(
            result.update_applied,
            lambda: result.state,
            lambda: carry,
        )
        return next_carry, (
            jnp.where(result.update_applied, current_action, jnp.zeros_like(current_action)),
            jnp.where(result.update_applied, current_mean, jnp.zeros_like(current_mean)),
            jnp.where(result.update_applied, current_sigma, jnp.zeros_like(current_sigma)),
            result.value,
            result.td_error,
            result.update_applied,
        )

    (
        final_state,
        (actions_out, means_out, sigmas_out, values, td_errors, updates_applied),
    ) = jax.lax.scan(
        _scan_fn,
        state,
        (observations, rewards, discounts, next_observations, actions),
    )
    return ContinuousActorCriticArrayResult(  # type: ignore[call-arg]
        state=final_state,
        actions=actions_out,
        means=means_out,
        sigmas=sigmas_out,
        values=values,
        td_errors=td_errors,
        updates_applied=updates_applied,
    )
