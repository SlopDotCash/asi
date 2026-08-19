# mypy: disable-error-code="attr-defined,call-arg,name-defined"
"""Conventional option-return and option-duration prediction.

This module implements the narrow mechanism called for in Alberta Plan Step 5:
learn an option's conventional cumulative reward *and* its expected remaining
primitive-step duration.  Each supplied option owns two linear GVF-equivalent
TD(0) heads:

.. math::

    V^r_o(s_t) &= \\mathbb{E}[R_{t+1} + \\gamma_{t+1} V^r_o(s_{t+1})], \\
    V^\tau_o(s_t) &= \\mathbb{E}[1 + \\gamma_{t+1} V^\tau_o(s_{t+1})].

The caller supplies the effective per-transition continuation discount
``gamma`` explicitly: zero on option termination, one while an undiscounted
option continues, or a value in ``[0, 1]`` for a discounted question.  The
reward target deliberately does **not** subtract an average-reward baseline.
That conventional prediction is separate from the differential learners in
``average_reward.py``.

The implementation is linear, online, and fixed-memory.  It assumes option
identities and features are supplied; it does not discover options, learn their
policies, or establish Step 5 completion on its own.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
import time
from collections.abc import Mapping
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework.core._float32_scalars import validated_float32_scalar_with_ratio

REWARD_HEAD = 0
DURATION_HEAD = 1
N_HEADS = 2
_INT32_MAX = 2_147_483_647
_MAX_PERSISTENT_STATE_BYTES = 256 * 1024 * 1024
_ACTUAL_INT_TYPES = frozenset(
    {int, *(np.dtype(code).type for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"))}
)
def _require_int32(name: str, value: object, *, minimum: int = 1) -> int:
    """Canonicalize one concrete Python/NumPy integer without invoking hooks."""
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= _INT32_MAX:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    return canonical


def _require_float32_real(
    name: str,
    value: object,
    *,
    strictly_positive: bool,
) -> float:
    """Validate one host scalar in its exact domain and float32 sink."""
    domain = "positive" if strictly_positive else "non-negative"
    try:
        stored, numerator, _ = validated_float32_scalar_with_ratio(
            name,
            value,
            positive=strictly_positive,
            lower=None if strictly_positive else 0.0,
        )
    except Exception as error:
        raise ValueError(f"{name} must be finite and {domain}") from error
    narrowed = float(np.float32(stored))
    if numerator != 0 and narrowed == 0.0:
        raise ValueError(f"{name} must not underflow to zero in float32")
    return stored


def _state_resources(n_options: int, feature_dim: int) -> dict[str, int]:
    """Exact persistent array envelope, excluding Python/compiled/transient data."""
    trainable_parameters = n_options * N_HEADS * feature_dim
    persistent_scalars = trainable_parameters + n_options + 1
    persistent_bytes = 4 * persistent_scalars
    if trainable_parameters > _INT32_MAX or persistent_scalars > _INT32_MAX:
        raise ValueError("derived option-duration persistent scalars must fit signed int32")
    if persistent_bytes > _MAX_PERSISTENT_STATE_BYTES:
        raise ValueError("derived option-duration persistent state exceeds 256 MiB")
    return {
        "trainable_parameters": trainable_parameters,
        "persistent_scalars": persistent_scalars,
        "persistent_bytes": persistent_bytes,
    }


def _saturating_increment(value: Array) -> Array:
    """Increment an int32 counter without wrapping at its lifetime limit."""
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return jnp.where(value < maximum, value + jnp.asarray(1, dtype=jnp.int32), maximum)


@dataclasses.dataclass(frozen=True)
class OptionValueDurationConfig:
    """Hyperparameters for the two conventional TD(0) heads.

    Args:
        reward_step_size: Step-size for the cumulative-reward head.
        duration_step_size: Step-size for the remaining-duration head.
        duration_floor: Positive denominator floor used only when reporting
            reward-per-step scores.  It does not alter either TD target.
    """

    reward_step_size: float = 0.1
    duration_step_size: float = 0.1
    duration_floor: float = 1e-6

    def __post_init__(self) -> None:
        """Validate float32 execution values and canonicalize host scalars."""
        object.__setattr__(
            self,
            "reward_step_size",
            _require_float32_real(
                "reward_step_size",
                self.reward_step_size,
                strictly_positive=False,
            ),
        )
        object.__setattr__(
            self,
            "duration_step_size",
            _require_float32_real(
                "duration_step_size",
                self.duration_step_size,
                strictly_positive=False,
            ),
        )
        object.__setattr__(
            self,
            "duration_floor",
            _require_float32_real(
                "duration_floor",
                self.duration_floor,
                strictly_positive=True,
            ),
        )

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration."""
        return {
            "type": "OptionValueDurationConfig",
            "reward_step_size": self.reward_step_size,
            "duration_step_size": self.duration_step_size,
            "duration_floor": self.duration_floor,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> OptionValueDurationConfig:
        """Reconstruct a config from :meth:`to_config` output."""
        if not issubclass(type(config), Mapping):
            raise ValueError("config must be a mapping")
        try:
            payload = dict(config)
        except Exception as error:
            raise ValueError("config must be a readable mapping") from error
        if any(type(key) is not str for key in payload):
            raise ValueError("config keys must be strings")
        payload.pop("type", None)
        try:
            return cls(**payload)
        except TypeError as error:
            raise ValueError("config has unsupported fields") from error


@chex.dataclass(frozen=True)
class OptionValueDurationState:
    """Fixed-memory state for all supplied options.

    ``weights[o, 0]`` is option ``o``'s reward head and ``weights[o, 1]`` is
    its duration head.  There are exactly
    ``2 * n_options * feature_dim`` trainable scalar parameters.
    """

    weights: Float[Array, "n_options 2 feature_dim"]
    option_update_counts: Int[Array, " n_options"]
    step_count: Int[Array, ""]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class OptionValueDurationPrediction:
    """Predictions for every option at one feature vector.

    ``reward_rates`` is the descriptive ratio ``reward_values / durations``.
    It is a valid direct decision score in renewal settings where the options
    return to the same decision condition, as in the accompanying diagnostic.
    It is not a general replacement for a semi-Markov control backup when
    options lead to different downstream states.
    """

    reward_values: Float[Array, " n_options"]
    durations: Float[Array, " n_options"]
    reward_rates: Float[Array, " n_options"]


@chex.dataclass(frozen=True)
class OptionValueDurationUpdateResult:
    """Diagnostics from one selected-option TD(0) update.

    The two-vector fields are ordered ``[reward, duration]``.
    """

    state: OptionValueDurationState
    predictions: Float[Array, " 2"]
    next_predictions: Float[Array, " 2"]
    td_targets: Float[Array, " 2"]
    td_errors: Float[Array, " 2"]
    continuation_discount: Float[Array, ""]
    head_updates_applied: Bool[Array, " 2"]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class OptionValueDurationArrayResult:
    """Result of scanning the learner over primitive option transitions."""

    state: OptionValueDurationState
    predictions: Float[Array, "num_steps 2"]
    next_predictions: Float[Array, "num_steps 2"]
    td_targets: Float[Array, "num_steps 2"]
    td_errors: Float[Array, "num_steps 2"]
    continuation_discounts: Float[Array, " num_steps"]
    head_updates_applied: Bool[Array, "num_steps 2"]
    updates_applied: Bool[Array, " num_steps"]


class OptionValueDurationLearner:
    """Two conventional linear TD(0) heads for each supplied option.

    This learner is intentionally independent of differential option/control
    values.  To use it, call :meth:`update` for every primitive transition
    generated while an option runs, passing that option's index and an explicit
    continuation discount.  A terminating transition must have discount zero.
    """

    def __init__(
        self,
        n_options: int,
        config: OptionValueDurationConfig | None = None,
    ):
        """Create a fixed-capacity option predictor."""
        self._n_options = _require_int32("n_options", n_options)
        _state_resources(self._n_options, 1)
        if config is None:
            config = OptionValueDurationConfig()
        if type(config) is not OptionValueDurationConfig:
            raise ValueError("config must be an actual OptionValueDurationConfig")
        self._config = config

    @property
    def n_options(self) -> int:
        """Number of supplied option identities."""
        return self._n_options

    @property
    def config(self) -> OptionValueDurationConfig:
        """Learner configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-serializable learner configuration."""
        return {
            "type": "OptionValueDurationLearner",
            "n_options": self._n_options,
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> OptionValueDurationLearner:
        """Reconstruct a learner from :meth:`to_config` output."""
        if not issubclass(type(config), Mapping):
            raise ValueError("config must be a mapping")
        try:
            payload = dict(config)
        except Exception as error:
            raise ValueError("config must be a readable mapping") from error
        if any(type(key) is not str for key in payload):
            raise ValueError("config keys must be strings")
        payload.pop("type", None)
        try:
            return cls(
                n_options=payload["n_options"],
                config=OptionValueDurationConfig.from_config(payload["config"]),
            )
        except (KeyError, TypeError) as error:
            raise ValueError("config has missing or unsupported fields") from error

    def trainable_parameter_count(self, feature_dim: int) -> int:
        """Return the exact history-independent trainable parameter count."""
        validated_feature_dim = _require_int32("feature_dim", feature_dim)
        return _state_resources(self._n_options, validated_feature_dim)[
            "trainable_parameters"
        ]

    def persistent_resource_budget(self, feature_dim: int) -> dict[str, int]:
        """Return the exact retained-array budget before initialization."""
        validated_feature_dim = _require_int32("feature_dim", feature_dim)
        return _state_resources(self._n_options, validated_feature_dim)

    def init(self, feature_dim: int) -> OptionValueDurationState:
        """Initialize all heads to zero."""
        feature_dim = _require_int32("feature_dim", feature_dim)
        _state_resources(self._n_options, feature_dim)
        return OptionValueDurationState(
            weights=jnp.zeros(
                (self._n_options, N_HEADS, feature_dim),
                dtype=jnp.float32,
            ),
            option_update_counts=jnp.zeros((self._n_options,), dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    def _require_state_contract(
        self, state: object
    ) -> tuple[OptionValueDurationState, int]:
        if type(state) is not OptionValueDurationState:
            raise ValueError("state must be an actual OptionValueDurationState")
        canonical = state
        try:
            weights_ndim = canonical.weights.ndim
            weights_shape = canonical.weights.shape
            weights_dtype = canonical.weights.dtype
            counts_shape = canonical.option_update_counts.shape
            counts_dtype = canonical.option_update_counts.dtype
            step_shape = canonical.step_count.shape
            step_dtype = canonical.step_count.dtype
        except Exception as error:
            raise ValueError("state arrays must expose readable shape and dtype") from error
        if weights_ndim != 3:
            raise ValueError("state.weights must be rank 3")
        feature_dim = weights_shape[2]
        if (
            weights_shape != (self._n_options, N_HEADS, feature_dim)
            or weights_dtype != jnp.float32
        ):
            raise ValueError("state.weights has an incompatible shape or dtype")
        if (
            counts_shape != (self._n_options,)
            or counts_dtype != jnp.int32
        ):
            raise ValueError("state.option_update_counts has an incompatible shape or dtype")
        if step_shape != () or step_dtype != jnp.int32:
            raise ValueError("state.step_count must be scalar int32")
        _state_resources(self._n_options, feature_dim)
        return canonical, feature_dim

    @staticmethod
    def _require_array(
        name: str,
        value: object,
        *,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> Array:
        actual_type = type(value)
        if not (
            actual_type is np.ndarray
            or issubclass(actual_type, (jax.Array, jax.core.Tracer))
        ):
            raise TypeError(f"{name} must expose trusted array metadata")
        actual_shape = tuple(value.shape)
        actual_dtype = jnp.dtype(value.dtype)
        if actual_shape != shape or actual_dtype != jnp.dtype(dtype):
            raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
        return jnp.asarray(value)

    @staticmethod
    def _state_values_valid(state: OptionValueDurationState) -> Array:
        return (
            jnp.all(jnp.isfinite(state.weights))
            & jnp.all(state.option_update_counts >= 0)
            & (state.step_count >= 0)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict_heads(
        self,
        state: OptionValueDurationState,
        observation: Array,
    ) -> Float[Array, "n_options 2"]:
        """Return raw ``[reward, duration]`` head predictions for all options."""
        state, feature_dim = self._require_state_contract(state)
        observation = self._require_array(
            "observation",
            observation,
            shape=(feature_dim,),
            dtype=jnp.float32,
        )
        heads = jnp.einsum("ohf,f->oh", state.weights, observation)
        return jnp.where(self._state_values_valid(state), heads, jnp.zeros_like(heads))

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: OptionValueDurationState,
        observation: Array,
    ) -> OptionValueDurationPrediction:
        """Return conventional values, durations, and reward-per-step scores."""
        heads = self.predict_heads(state, observation)
        reward_values = heads[:, REWARD_HEAD]
        durations = heads[:, DURATION_HEAD]
        safe_durations = jnp.maximum(
            durations,
            jnp.asarray(self._config.duration_floor, dtype=jnp.float32),
        )
        return OptionValueDurationPrediction(
            reward_values=reward_values,
            durations=durations,
            reward_rates=reward_values / safe_durations,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: OptionValueDurationState,
        observation: Array,
        option_index: Array,
        reward: Array,
        next_observation: Array,
        continuation_discount: Array,
    ) -> OptionValueDurationUpdateResult:
        """Apply one primitive transition to one option's two heads.

        The targets are exactly
        ``[reward, 1] + continuation_discount * next_predictions``.  No
        average-reward estimate is present or subtracted.  The caller is
        responsible for supplying a scalar discount in ``[0, 1]`` and for
        setting it to zero when the option terminates.
        """
        state, feature_dim = self._require_state_contract(state)
        observation = self._require_array(
            "observation", observation, shape=(feature_dim,), dtype=jnp.float32
        )
        next_observation = self._require_array(
            "next_observation",
            next_observation,
            shape=(feature_dim,),
            dtype=jnp.float32,
        )
        option_index = self._require_array(
            "option_index", option_index, shape=(), dtype=jnp.int32
        )
        option_index_valid = (option_index >= 0) & (option_index < self._n_options)
        safe_option_index = jnp.clip(option_index, 0, self._n_options - 1)
        reward = self._require_array("reward", reward, shape=(), dtype=jnp.float32)
        continuation_discount = self._require_array(
            "continuation_discount",
            continuation_discount,
            shape=(),
            dtype=jnp.float32,
        )

        option_weights = state.weights[safe_option_index]
        predictions = option_weights @ observation
        next_predictions = option_weights @ next_observation
        cumulants = jnp.stack(
            (reward, jnp.array(1.0, dtype=jnp.float32)),
        )
        terminated = continuation_discount == 0.0
        bootstrap = jnp.where(
            terminated,
            jnp.zeros_like(next_predictions),
            continuation_discount * next_predictions,
        )
        td_targets = cumulants + bootstrap
        td_errors = td_targets - predictions
        step_sizes = jnp.array(
            [
                self._config.reward_step_size,
                self._config.duration_step_size,
            ],
            dtype=jnp.float32,
        )
        proposed_option_weights = (
            option_weights + step_sizes[:, None] * td_errors[:, None] * observation[None, :]
        )
        source_valid = self._state_values_valid(state)
        next_needed = continuation_discount != 0.0
        shared_inputs_valid = (
            jnp.all(jnp.isfinite(observation))
            & ((~next_needed) | jnp.all(jnp.isfinite(next_observation)))
            & jnp.isfinite(continuation_discount)
            & (continuation_discount >= 0.0)
            & (continuation_discount <= 1.0)
            & option_index_valid
            & source_valid
        )
        head_inputs_valid = jnp.stack(
            (
                jnp.isfinite(reward),
                jnp.asarray(True, dtype=jnp.bool_),
            )
        )
        candidate_finite = jnp.all(jnp.isfinite(proposed_option_weights), axis=1)
        head_updates_applied = (
            shared_inputs_valid & head_inputs_valid & candidate_finite
        )
        updated_option_weights = jnp.where(
            head_updates_applied[:, None], proposed_option_weights, option_weights
        )
        update_applied = jnp.any(head_updates_applied)
        proposed_state = state.replace(
            weights=state.weights.at[safe_option_index].set(updated_option_weights),
            option_update_counts=state.option_update_counts.at[safe_option_index].set(
                _saturating_increment(state.option_update_counts[safe_option_index])
            ),
            step_count=_saturating_increment(state.step_count),
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda: proposed_state,
            lambda: state,
        )
        return OptionValueDurationUpdateResult(
            state=new_state,
            predictions=jnp.where(
                head_updates_applied, predictions, jnp.zeros_like(predictions)
            ),
            next_predictions=jnp.where(
                head_updates_applied & (~terminated),
                next_predictions,
                jnp.zeros_like(next_predictions),
            ),
            td_targets=jnp.where(
                head_updates_applied, td_targets, jnp.zeros_like(td_targets)
            ),
            td_errors=jnp.where(
                head_updates_applied, td_errors, jnp.zeros_like(td_errors)
            ),
            continuation_discount=jnp.where(
                update_applied,
                continuation_discount,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            head_updates_applied=head_updates_applied,
            update_applied=update_applied,
        )


def run_option_value_duration_from_arrays(
    learner: OptionValueDurationLearner,
    state: OptionValueDurationState,
    observations: Float[Array, "num_steps feature_dim"],
    option_indices: Int[Array, " num_steps"],
    rewards: Float[Array, " num_steps"],
    next_observations: Float[Array, "num_steps feature_dim"],
    continuation_discounts: Float[Array, " num_steps"],
) -> OptionValueDurationArrayResult:
    """Scan online TD(0) updates over primitive option transitions."""
    if type(learner) is not OptionValueDurationLearner:
        raise ValueError("learner must be an actual OptionValueDurationLearner")
    state, feature_dim = learner._require_state_contract(state)
    try:
        observations = jnp.asarray(observations)
    except Exception as error:
        raise ValueError("observations must be a readable float32 matrix") from error
    if observations.ndim != 2 or observations.shape[1] != feature_dim:
        raise ValueError("observations have an incompatible shape")
    if observations.dtype != jnp.float32:
        raise ValueError("observations must have dtype float32")
    num_steps = _require_int32("scan sequence length", observations.shape[0], minimum=1)
    output_scalars = num_steps * (4 * learner.n_options + 3)
    if output_scalars > _INT32_MAX or 4 * output_scalars > _INT32_MAX:
        raise ValueError("option value duration scan result bytes must fit signed int32")
    next_observations = learner._require_array(
        "next_observations",
        next_observations,
        shape=(num_steps, feature_dim),
        dtype=jnp.float32,
    )
    option_indices = learner._require_array(
        "option_indices", option_indices, shape=(num_steps,), dtype=jnp.int32
    )
    rewards = learner._require_array(
        "rewards", rewards, shape=(num_steps,), dtype=jnp.float32
    )
    continuation_discounts = learner._require_array(
        "continuation_discounts",
        continuation_discounts,
        shape=(num_steps,),
        dtype=jnp.float32,
    )
    start = time.time()

    def _scan_fn(
        carry: OptionValueDurationState,
        inputs: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[
        OptionValueDurationState,
        tuple[Array, Array, Array, Array, Array, Array, Array],
    ]:
        observation, option_index, reward, next_observation, discount = inputs
        result = learner.update(
            carry,
            observation,
            option_index,
            reward,
            next_observation,
            discount,
        )
        return result.state, (
            result.predictions,
            result.next_predictions,
            result.td_targets,
            result.td_errors,
            result.continuation_discount,
            result.head_updates_applied,
            result.update_applied,
        )

    (
        final_state,
        (
            predictions,
            next_predictions,
            td_targets,
            td_errors,
            discounts,
            head_updates_applied,
            updates_applied,
        ),
    ) = jax.lax.scan(
        _scan_fn,
        state,
        (
            observations,
            option_indices,
            rewards,
            next_observations,
            continuation_discounts,
        ),
    )
    final_state = final_state.replace(uptime_s=final_state.uptime_s + (time.time() - start))
    return OptionValueDurationArrayResult(
        state=final_state,
        predictions=predictions,
        next_predictions=next_predictions,
        td_targets=td_targets,
        td_errors=td_errors,
        continuation_discounts=discounts,
        head_updates_applied=head_updates_applied,
        updates_applied=updates_applied,
    )
