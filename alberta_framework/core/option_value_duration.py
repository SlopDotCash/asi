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
from numbers import Real
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework._float32 import round_real_to_float32

REWARD_HEAD = 0
DURATION_HEAD = 1
N_HEADS = 2
_INT32_MAX = 2_147_483_647
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


def _require_int32(name: str, value: object, *, minimum: int = 1) -> int:
    """Validate one actual host integer in the signed-int32 domain."""
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
    preserve_builtin_payload = type(value) is int or type(value) is float
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and {domain}")
    try:
        if strictly_positive:
            if value <= 0:
                raise ValueError
        elif value < 0:
            raise ValueError
        narrowed = round_real_to_float32(value)
    except (FloatingPointError, OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and {domain}") from error
    if not bool(np.isfinite(narrowed)):
        raise ValueError(f"{name} must narrow to a finite float32")
    if value != 0 and narrowed == 0.0:
        raise ValueError(f"{name} must not underflow to zero in float32")

    # Preserve already-valid built-in payloads for serialization compatibility.
    # Non-built-in Real scalars are canonicalized to the exact execution value.
    if not preserve_builtin_payload:
        return narrowed
    number = float(value)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        renarrowed = float(np.float32(number))
    return number if narrowed == renarrowed else narrowed


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
    def from_config(cls, config: dict[str, Any]) -> OptionValueDurationConfig:
        """Reconstruct a config from :meth:`to_config` output."""
        payload = dict(config)
        payload.pop("type", None)
        return cls(**payload)


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
        self._config = config or OptionValueDurationConfig()

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
    def from_config(cls, config: dict[str, Any]) -> OptionValueDurationLearner:
        """Reconstruct a learner from :meth:`to_config` output."""
        payload = dict(config)
        payload.pop("type", None)
        return cls(
            n_options=payload["n_options"],
            config=OptionValueDurationConfig.from_config(payload["config"]),
        )

    def trainable_parameter_count(self, feature_dim: int) -> int:
        """Return the exact history-independent trainable parameter count."""
        validated_feature_dim = _require_int32("feature_dim", feature_dim)
        return self._n_options * N_HEADS * validated_feature_dim

    def init(self, feature_dim: int) -> OptionValueDurationState:
        """Initialize all heads to zero."""
        validated_feature_dim = _require_int32("feature_dim", feature_dim)
        return OptionValueDurationState(
            weights=jnp.zeros(
                (self._n_options, N_HEADS, validated_feature_dim),
                dtype=jnp.float32,
            ),
            option_update_counts=jnp.zeros((self._n_options,), dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict_heads(
        self,
        state: OptionValueDurationState,
        observation: Array,
    ) -> Float[Array, "n_options 2"]:
        """Return raw ``[reward, duration]`` head predictions for all options."""
        observation = jnp.asarray(observation, dtype=jnp.float32)
        return jnp.einsum("ohf,f->oh", state.weights, observation)

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
        observation = jnp.asarray(observation, dtype=jnp.float32)
        next_observation = jnp.asarray(next_observation, dtype=jnp.float32)
        option_index = jnp.asarray(option_index, dtype=jnp.int32)
        option_index_valid = (option_index >= 0) & (option_index < self._n_options)
        safe_option_index = jnp.clip(option_index, 0, self._n_options - 1)
        reward = jnp.squeeze(jnp.asarray(reward, dtype=jnp.float32))
        continuation_discount = jnp.squeeze(jnp.asarray(continuation_discount, dtype=jnp.float32))

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
        source_finite = jnp.all(jnp.isfinite(state.weights))
        next_needed = continuation_discount != 0.0
        shared_inputs_valid = (
            jnp.all(jnp.isfinite(observation))
            & ((~next_needed) | jnp.all(jnp.isfinite(next_observation)))
            & jnp.isfinite(continuation_discount)
            & (continuation_discount >= 0.0)
            & (continuation_discount <= 1.0)
            & option_index_valid
            & source_finite
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
