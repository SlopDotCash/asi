# mypy: disable-error-code="call-arg"
"""Action-conditioned one-step world models.

The prediction surface is intentionally small: predict the next observation
(or its delta), reward, and discount from the current observation and action.
This is enough to support GVF-style environment prediction and guarded
Dyna-style dream updates without committing the core API to a large latent
dynamics architecture too early.

References:
    Sutton (1991). "Dyna, an Integrated Architecture for Learning,
        Planning, and Reacting." SIGART Bulletin 2(4).
"""

from __future__ import annotations

import dataclasses
import functools
import operator
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework.core._float32_scalars import (
    validated_float32_scalar,
    validated_float32_scalar_with_ratio,
)
from alberta_framework.core.multi_head_learner import (
    AnyOptimizer,
    MultiHeadMLPLearner,
    MultiHeadMLPState,
    MultiHeadMLPUpdateResult,
)
from alberta_framework.core.normalizers import (
    EMANormalizerState,
    Normalizer,
    WelfordNormalizerState,
)
from alberta_framework.core.optimizers import Bounder
from alberta_framework.core.types import TraceMode
from alberta_framework.core.update_safety import (
    floating_tree_is_finite,
    neutralize_array,
    safe_discrete_action,
    select_transaction,
)

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
_ACTUAL_FLOAT_TYPES = frozenset(
    {float, Fraction, *(np.dtype(code).type for code in ("e", "f", "d", "g"))}
)


def _require_int32(name: str, value: object, *, minimum: int) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= _INT32_MAX:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    return canonical


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _validated_config_float(name: str, value: object, **bounds: Any) -> float:
    if type(value) not in (_ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES):
        raise ValueError(f"{name} must be a finite real scalar")
    return validated_float32_scalar(name, value, **bounds)


_SMALLEST_NORMAL_FLOAT32 = float(np.finfo(np.float32).tiny)


def _validated_normalizing_scale(name: str, value: object) -> float:
    """Accept only a scale that still divides and multiplies once narrowed to float32.

    A subnormal float32 is returned as an exact zero by every arithmetic operand on
    the accelerator backends this runs on, so a scale below the smallest normal
    cannot rescale anything: the head it belongs to is structurally unable to
    express a change, whatever it predicts. Refusing it here keeps that a
    construction error rather than a model that converges and predicts nothing.
    """
    scale = _validated_config_float(name, value, positive=True)
    if abs(float(np.float32(scale))) < _SMALLEST_NORMAL_FLOAT32:
        raise ValueError(f"{name} must remain a normal float32 once narrowed, not a subnormal")
    return scale


def _validated_step_size(name: str, value: object) -> float:
    """Accept an exact zero freeze without silently underflowing learning to zero."""
    if type(value) not in (_ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES):
        raise ValueError(f"{name} must be a finite real scalar")
    stored, numerator, denominator = validated_float32_scalar_with_ratio(
        name, value, lower=0.0
    )
    # Values at or below half the smallest binary32 subnormal round to zero
    # (the exact halfway case ties to the even zero significand).
    if numerator > 0 and numerator * (1 << 150) <= denominator:
        raise ValueError(f"{name} must remain positive once narrowed to float32 or be exact zero")
    return stored


def _validate_hidden_sizes(value: object) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise ValueError("hidden_sizes must be an actual tuple")
    return tuple(
        _require_int32(f"hidden_sizes[{index}]", width, minimum=1)
        for index, width in enumerate(value)
    )


def _checked_product(name: str, left: int, right: int) -> int:
    if left > _INT32_MAX // right:
        raise ValueError(f"derived {name} must fit in signed int32")
    return left * right


def _world_model_direct_state_scalars(
    *,
    observation_dim: int,
    action_feature_dim: int,
    hidden_sizes: tuple[int, ...],
    n_heads: int,
    outer_state_scalars: int,
) -> int:
    if observation_dim > _INT32_MAX - action_feature_dim:
        raise ValueError("derived input_dim must fit in signed int32")
    input_dim = observation_dim + action_feature_dim
    layer_sizes = (input_dim, *hidden_sizes)
    for index, (fan_in, fan_out) in enumerate(
        zip(layer_sizes, layer_sizes[1:], strict=False)
    ):
        _checked_product(f"hidden_layer[{index}]_scalars", fan_in, fan_out)
    head_input = hidden_sizes[-1] if hidden_sizes else input_dim
    _checked_product("head_weight_scalars", n_heads, head_input)
    trunk_parameters = sum(
        fan_out * (fan_in + 1)
        for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:], strict=False)
    )
    head_parameters = n_heads * (head_input + 1)
    return (
        2 * (trunk_parameters + head_parameters)
        + sum(hidden_sizes)
        + 3
        + outer_state_scalars
    )


def _world_model_update_result_extras_bytes(*, observation_dim: int, n_heads: int) -> int:
    """Published ``WorldModelUpdateResult`` extras excluding persistent state.

    The +4 float32 scalars cover JIT-materialized ``birth_timestamp`` and
    ``uptime_s`` on both the source and proposed learner states. Persistent
    byte counts omit those host-only leaves; the update working set cannot.
    """
    extras_scalars = (
        2 * observation_dim
        + 11 * n_heads
        + 7
        + 4
    )
    extras_bools = 6
    return 4 * extras_scalars + extras_bools


def _world_model_update_working_set_bytes(
    *,
    observation_dim: int,
    action_feature_dim: int,
    hidden_sizes: tuple[int, ...],
    n_heads: int,
    outer_state_scalars: int,
) -> int:
    persist_bytes = 4 * _world_model_direct_state_scalars(
        observation_dim=observation_dim,
        action_feature_dim=action_feature_dim,
        hidden_sizes=hidden_sizes,
        n_heads=n_heads,
        outer_state_scalars=outer_state_scalars,
    )
    extras_bytes = _world_model_update_result_extras_bytes(
        observation_dim=observation_dim,
        n_heads=n_heads,
    )
    return 2 * persist_bytes + extras_bytes


def _preflight_world_model_update_working_set(
    *,
    observation_dim: int,
    action_feature_dim: int,
    hidden_sizes: tuple[int, ...],
    n_heads: int,
    outer_state_scalars: int,
) -> None:
    working_set_bytes = _world_model_update_working_set_bytes(
        observation_dim=observation_dim,
        action_feature_dim=action_feature_dim,
        hidden_sizes=hidden_sizes,
        n_heads=n_heads,
        outer_state_scalars=outer_state_scalars,
    )
    if working_set_bytes > _INT32_MAX:
        raise ValueError(
            "world-model update working set byte count must fit signed int32"
        )


def _validate_world_model_resources(
    *,
    observation_dim: int,
    action_feature_dim: int,
    hidden_sizes: tuple[int, ...],
    n_heads: int,
    outer_state_scalars: int,
) -> None:
    direct_scalars = _world_model_direct_state_scalars(
        observation_dim=observation_dim,
        action_feature_dim=action_feature_dim,
        hidden_sizes=hidden_sizes,
        n_heads=n_heads,
        outer_state_scalars=outer_state_scalars,
    )
    for name, value in (
        ("combined_direct_state_scalars", direct_scalars),
        ("combined_direct_state_bytes", 4 * direct_scalars),
    ):
        if value > _INT32_MAX:
            raise ValueError(f"derived {name} must fit in signed int32")


def _serialized_sequence(name: str, value: object) -> tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"serialized {name} must be an actual list or tuple")
    return tuple(cast(list[Any] | tuple[Any, ...], value))


def _require_scan_resource(name: str, *, float32_scalars: int, bool_scalars: int) -> None:
    if float32_scalars + bool_scalars > _INT32_MAX:
        raise ValueError(f"derived {name} scalar count must fit in signed int32")
    if 4 * float32_scalars + bool_scalars > _INT32_MAX:
        raise ValueError(f"derived {name} byte count must fit in signed int32")


def _scan_array_metadata(name: str, value: object) -> tuple[int, ...]:
    """Read shape and numeric dtype before any user-controlled array conversion."""
    try:
        raw_shape = object.__getattribute__(value, "shape")
        raw_dtype = object.__getattribute__(value, "dtype")
    except (AttributeError, TypeError):
        raise ValueError(f"{name} must expose array shape and dtype metadata") from None
    if type(raw_shape) is not tuple:
        raise ValueError(f"{name} shape metadata must be an exact tuple")
    shape = tuple(
        _require_int32(f"{name} shape dimension", dimension, minimum=0)
        for dimension in raw_shape
    )
    try:
        dtype = np.dtype(raw_dtype)
    except Exception:
        raise ValueError(f"{name} dtype metadata must be readable") from None
    if dtype.kind not in "biuf":
        raise ValueError(f"{name} must have a real numeric dtype")
    return shape


def _require_scan_array(name: str, value: object, expected: tuple[int, ...]) -> None:
    if _scan_array_metadata(name, value) != expected:
        raise ValueError(f"{name} must have shape {expected}")


def _saturating_increment(value: Array) -> Array:
    one = jnp.asarray(1, dtype=jnp.int32)
    return jnp.minimum(value, jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)) + one


def _float32_operand(
    name: str,
    value: Array | float | int,
    shape: tuple[int, ...],
) -> Array:
    """Narrow one numeric operand while preserving its exact shape contract."""
    array = jnp.asarray(value, dtype=jnp.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


@dataclasses.dataclass(frozen=True)
class ActionConditionedWorldModelConfig:
    """Configuration for :class:`ActionConditionedWorldModel`.

    Args:
        observation_dim: Flat observation dimensionality.
        n_actions: Number of discrete actions.
        gamma: Maximum environment discount used for clipping predicted
            discounts.
        observation_scale: Per-observation-dimension scale for normalized delta
            targets, dividing on encode and multiplying on decode. When ``None``,
            all dimensions use scale ``1``. Each entry must narrow to a normal
            float32; a subnormal is flushed by the decode multiply, leaving that
            head unable to express any change.
        reward_scale: Scalar reward target scale.
        predict_delta: When ``True`` (default), the observation heads are
            trained on the normalized change ``next_obs - obs`` and decoded
            by adding the predicted change back to the current observation;
            when ``False`` they predict the normalized next observation
            directly.
        hidden_sizes: Shared MLP trunk sizes. Use ``()`` for a linear model.
        step_size: Base learner step-size when ``optimizer`` is omitted.
        sparsity: Sparse initialization fraction for MLP weights.
        leaky_relu_slope: Negative slope for hidden activations.
        use_layer_norm: Whether to use parameterless layer normalization.
        trace_mode: Eligibility trace mode passed to the underlying learner.
        utility_decay: Hidden-unit utility EMA decay.
        error_decay: EMA decay for real one-step model error diagnostics.
        observation_clip_margin: Margin around observed min/max bounds used
            when producing imagined next observations.
        max_delta_scale: Clip predicted normalized deltas to this absolute
            magnitude before rescaling. This guards dream rollouts.
        include_action_interactions: Whether to append observation-by-action
            product features to the model input. This lets a linear world model
            represent simple action-conditioned slopes without requiring a
            nonlinear trunk.
    """

    observation_dim: int
    n_actions: int
    gamma: float = 0.99
    observation_scale: tuple[float, ...] | None = None
    reward_scale: float = 1.0
    predict_delta: bool = True
    hidden_sizes: tuple[int, ...] = (64, 64)
    step_size: float = 0.03
    sparsity: float = 0.9
    leaky_relu_slope: float = 0.01
    use_layer_norm: bool = True
    trace_mode: TraceMode = TraceMode.ACCUMULATING
    utility_decay: float = 0.99
    error_decay: float = 0.99
    observation_clip_margin: float = 0.05
    max_delta_scale: float = 5.0
    include_action_interactions: bool = False

    def __post_init__(self) -> None:
        """Validate and canonicalize the complete static construction."""
        observation_dim = _require_int32("observation_dim", self.observation_dim, minimum=1)
        n_actions = _require_int32("n_actions", self.n_actions, minimum=1)
        hidden_sizes = _validate_hidden_sizes(self.hidden_sizes)
        for name in ("predict_delta", "use_layer_norm", "include_action_interactions"):
            object.__setattr__(self, name, _require_bool(name, getattr(self, name)))
        if type(self.trace_mode) is not TraceMode:
            raise ValueError("trace_mode must be a TraceMode")

        observation_scale = self.observation_scale
        if observation_scale is not None:
            if type(observation_scale) is not tuple:
                raise ValueError("observation_scale must be an actual tuple or None")
            if len(observation_scale) != observation_dim:
                raise ValueError("observation_scale length must equal observation_dim")
            observation_scale = tuple(
                _validated_normalizing_scale(f"observation_scale[{index}]", scale)
                for index, scale in enumerate(observation_scale)
            )

        scalar_specs: tuple[tuple[str, dict[str, Any]], ...] = (
            ("gamma", {"lower": 0.0, "upper": 1.0}),
            ("reward_scale", {"positive": True}),
            ("sparsity", {"lower": 0.0, "upper": 1.0}),
            ("leaky_relu_slope", {"lower": 0.0, "upper": 1.0}),
            ("utility_decay", {"lower": 0.0, "upper": 1.0, "upper_inclusive": False}),
            ("error_decay", {"lower": 0.0, "upper": 1.0, "upper_inclusive": False}),
            ("observation_clip_margin", {"lower": 0.0}),
            ("max_delta_scale", {"positive": True}),
        )
        object.__setattr__(self, "observation_dim", observation_dim)
        object.__setattr__(self, "n_actions", n_actions)
        object.__setattr__(self, "hidden_sizes", hidden_sizes)
        object.__setattr__(self, "observation_scale", observation_scale)
        object.__setattr__(self, "step_size", _validated_step_size("step_size", self.step_size))
        for name, bounds in scalar_specs:
            object.__setattr__(
                self,
                name,
                _validated_config_float(name, getattr(self, name), **bounds),
            )
        if (
            observation_scale is not None
            and max(observation_scale) * self.max_delta_scale
            > float(np.finfo(np.float32).max)
        ):
            raise ValueError("observation_scale * max_delta_scale must remain finite")

        action_feature_dim = n_actions
        if self.include_action_interactions:
            interactions = _checked_product(
                "action_interaction_scalars", observation_dim, n_actions
            )
            if action_feature_dim > _INT32_MAX - interactions:
                raise ValueError("derived action_feature_dim must fit in signed int32")
            action_feature_dim += interactions
        if observation_dim > _INT32_MAX - 2:
            raise ValueError("derived n_heads must fit in signed int32")
        _validate_world_model_resources(
            observation_dim=observation_dim,
            action_feature_dim=action_feature_dim,
            hidden_sizes=hidden_sizes,
            n_heads=observation_dim + 2,
            outer_state_scalars=2 * observation_dim + 4,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = dataclasses.asdict(self)
        payload["type"] = "ActionConditionedWorldModelConfig"
        payload["hidden_sizes"] = list(self.hidden_sizes)
        payload["trace_mode"] = self.trace_mode.value
        if self.observation_scale is not None:
            payload["observation_scale"] = list(self.observation_scale)
        return payload

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ActionConditionedWorldModelConfig:
        """Reconstruct from :meth:`to_config` output."""
        if not issubclass(type(config), Mapping):
            raise ValueError("config must be an actual mapping")
        try:
            payload = dict(config)
        except Exception as error:
            raise ValueError("config must be a readable mapping") from error
        if any(type(key) is not str for key in payload):
            raise ValueError("config keys must be exact strings")
        type_name = payload.pop("type", None)
        if type_name is not None and (
            type(type_name) is not str or type_name != cls.__name__
        ):
            raise ValueError("config type differs")
        if "hidden_sizes" in payload:
            payload["hidden_sizes"] = _serialized_sequence(
                "hidden_sizes", payload["hidden_sizes"]
            )
        if "observation_scale" in payload and payload["observation_scale"] is not None:
            payload["observation_scale"] = _serialized_sequence(
                "observation_scale", payload["observation_scale"]
            )
        if "trace_mode" in payload:
            if type(payload["trace_mode"]) is not str:
                raise ValueError("serialized trace_mode must be an actual string")
            payload["trace_mode"] = TraceMode(payload["trace_mode"])
        return cls(**payload)


@chex.dataclass(frozen=True)
class ActionConditionedWorldModelState:
    """State for :class:`ActionConditionedWorldModel`."""

    learner_state: MultiHeadMLPState
    observation_min: Float[Array, " observation_dim"]
    observation_max: Float[Array, " observation_dim"]
    reward_min: Float[Array, ""]
    reward_max: Float[Array, ""]
    model_error_ema: Float[Array, ""]
    step_count: Array


@chex.dataclass(frozen=True)
class WorldModelPrediction:
    """Decoded world-model prediction."""

    next_observation: Float[Array, " observation_dim"]
    reward: Float[Array, ""]
    raw_predictions: Float[Array, " model_heads"]
    discount: Float[Array, ""]


@chex.dataclass(frozen=True)
class WorldModelUpdateResult:
    """Result from one real transition update."""

    state: Any
    prediction: WorldModelPrediction
    targets: Float[Array, " model_heads"]
    errors: Float[Array, " model_heads"]
    per_head_metrics: Float[Array, "model_heads 3"]
    prediction_error: Float[Array, ""]
    observation_mse: Float[Array, ""]
    reward_error: Float[Array, ""]
    next_observation_errors: Float[Array, " observation_dim"]
    discount_error: Float[Array, ""]
    learner_result: MultiHeadMLPUpdateResult
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class ActionConditionedWorldModelLearningResult:
    """Result from scan-based action-conditioned world-model learning."""

    state: ActionConditionedWorldModelState
    next_observation_predictions: Float[Array, "num_steps observation_dim"]
    reward_predictions: Float[Array, " num_steps"]
    discount_predictions: Float[Array, " num_steps"]
    raw_predictions: Float[Array, "num_steps model_heads"]
    targets: Float[Array, "num_steps model_heads"]
    errors: Float[Array, "num_steps model_heads"]
    prediction_errors: Float[Array, " num_steps"]
    observation_mse: Float[Array, " num_steps"]
    reward_errors: Float[Array, " num_steps"]
    next_observation_errors: Float[Array, "num_steps observation_dim"]
    discount_errors: Float[Array, " num_steps"]
    per_head_metrics: Float[Array, "num_steps model_heads metrics"]
    updates_applied: Bool[Array, " num_steps"]


def _rollback_multi_head_result(
    result: MultiHeadMLPUpdateResult,
    previous_state: MultiHeadMLPState,
    outer_update_applied: Array,
) -> MultiHeadMLPUpdateResult:
    """Make a nested learner result describe the outer committed transaction."""
    requested = ~jnp.isnan(result.errors)
    reported_errors = jnp.where(
        requested,
        neutralize_array(outer_update_applied, result.errors),
        jnp.nan,
    )
    reported_metrics = jnp.where(
        requested[:, None],
        neutralize_array(outer_update_applied, result.per_head_metrics),
        jnp.nan,
    )
    return cast(
        MultiHeadMLPUpdateResult,
        dataclasses.replace(
            cast(Any, result),
            state=select_transaction(outer_update_applied, result.state, previous_state),
            predictions=neutralize_array(outer_update_applied, result.predictions),
            errors=reported_errors,
            per_head_metrics=reported_metrics,
            trunk_bounding_metric=neutralize_array(
                outer_update_applied, result.trunk_bounding_metric
            ),
            post_step_words=jnp.where(
                outer_update_applied, result.post_step_words, result.pre_step_words
            ),
            update_applied=result.update_applied & outer_update_applied,
        ),
    )


def _action_world_model_state_is_valid(
    state: ActionConditionedWorldModelState,
    observation_dim: int,
) -> Bool[Array, ""]:
    """Accept either the intentional empty-bound sentinels or finite learned bounds."""
    direct_structure_valid = (
        state.observation_min.shape == (observation_dim,)
        and state.observation_max.shape == (observation_dim,)
        and state.reward_min.shape == ()
        and state.reward_max.shape == ()
        and state.model_error_ema.shape == ()
        and state.step_count.shape == ()
        and state.observation_min.dtype == jnp.float32
        and state.observation_max.dtype == jnp.float32
        and state.reward_min.dtype == jnp.float32
        and state.reward_max.dtype == jnp.float32
        and state.model_error_ema.dtype == jnp.float32
        and state.step_count.dtype == jnp.int32
    )
    finite_bounds = (
        jnp.all(jnp.isfinite(state.observation_min))
        & jnp.all(jnp.isfinite(state.observation_max))
        & jnp.isfinite(state.reward_min)
        & jnp.isfinite(state.reward_max)
    )
    empty_bounds = (
        (state.step_count == 0)
        & jnp.all(jnp.isposinf(state.observation_min))
        & jnp.all(jnp.isneginf(state.observation_max))
        & jnp.isposinf(state.reward_min)
        & jnp.isneginf(state.reward_max)
    )
    return (
        direct_structure_valid
        & floating_tree_is_finite(state.learner_state)
        & jnp.isfinite(state.model_error_ema)
        & (state.step_count >= 0)
        & (finite_bounds | empty_bounds)
    )


class ActionConditionedWorldModel:
    """One-step model for ``(observation, action) -> (next_obs, reward, discount)``.

    The model predicts normalized observation deltas rather than raw next
    observations, which avoids spending model capacity on the identity map and
    makes one-step dynamics errors easier to compare across channels.
    """

    def __init__(
        self,
        config: ActionConditionedWorldModelConfig,
        optimizer: AnyOptimizer | None = None,
        bounder: Bounder | None = None,
        normalizer: (
            Normalizer[EMANormalizerState] | Normalizer[WelfordNormalizerState] | None
        ) = None,
        head_optimizer: AnyOptimizer | None = None,
    ):
        """Initialize the world model."""
        config = self._validate_config(config)
        self._config = config
        self._observation_scale = (
            tuple(1.0 for _ in range(config.observation_dim))
            if config.observation_scale is None
            else tuple(config.observation_scale)
        )
        self._learner = MultiHeadMLPLearner(
            n_heads=config.observation_dim + 2,
            hidden_sizes=config.hidden_sizes,
            optimizer=optimizer,
            step_size=config.step_size,
            bounder=bounder,
            gamma=0.0,
            lamda=0.0,
            normalizer=normalizer,
            sparsity=config.sparsity,
            leaky_relu_slope=config.leaky_relu_slope,
            use_layer_norm=config.use_layer_norm,
            head_optimizer=head_optimizer,
            trace_mode=config.trace_mode,
            utility_decay=config.utility_decay,
        )

    @property
    def config(self) -> ActionConditionedWorldModelConfig:
        """Model configuration."""
        return self._config

    @property
    def learner(self) -> MultiHeadMLPLearner:
        """Underlying multi-head learner."""
        return self._learner

    @property
    def input_dim(self) -> int:
        """World-model input dimension."""
        base_dim = self._config.observation_dim + self._config.n_actions
        if self._config.include_action_interactions:
            return base_dim + self._config.observation_dim * self._config.n_actions
        return base_dim

    @property
    def n_heads(self) -> int:
        """Number of prediction heads."""
        return self._config.observation_dim + 2

    def to_config(self) -> dict[str, Any]:
        """Serialize model configuration and learner components."""
        learner_cfg = self._learner.to_config()
        return {
            "type": "ActionConditionedWorldModel",
            "config": self._config.to_config(),
            "learner": learner_cfg,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ActionConditionedWorldModel:
        """Reconstruct from :meth:`to_config` output.

        This restores constructor-level model hyperparameters. Optimizer,
        bounder, and normalizer objects are represented in the nested learner
        config, so this path mirrors their serialized settings where supported.
        """
        from alberta_framework.core.normalizers import normalizer_from_config
        from alberta_framework.core.optimizers import (
            bounder_from_config,
            optimizer_from_config,
        )

        payload = dict(config)
        payload.pop("type", None)
        model_config = ActionConditionedWorldModelConfig.from_config(payload["config"])
        learner_cfg = dict(payload["learner"])
        optimizer = optimizer_from_config(learner_cfg["optimizer"])
        bounder_cfg = learner_cfg.get("bounder")
        normalizer_cfg = learner_cfg.get("normalizer")
        head_opt_cfg = learner_cfg.get("head_optimizer")
        return cls(
            config=model_config,
            optimizer=optimizer,
            bounder=bounder_from_config(bounder_cfg) if bounder_cfg is not None else None,
            normalizer=(
                normalizer_from_config(normalizer_cfg)
                if normalizer_cfg is not None
                else None
            ),
            head_optimizer=(
                optimizer_from_config(head_opt_cfg) if head_opt_cfg is not None else None
            ),
        )

    def init(self, key: Array) -> ActionConditionedWorldModelState:
        """Initialize model state."""
        obs_dim = self._config.observation_dim
        _preflight_world_model_update_working_set(
            observation_dim=obs_dim,
            action_feature_dim=self.input_dim - obs_dim,
            hidden_sizes=self._config.hidden_sizes,
            n_heads=self.n_heads,
            outer_state_scalars=2 * obs_dim + 4,
        )
        return ActionConditionedWorldModelState(
            learner_state=self._learner.init(self.input_dim, key),
            observation_min=jnp.full((obs_dim,), jnp.inf, dtype=jnp.float32),
            observation_max=jnp.full((obs_dim,), -jnp.inf, dtype=jnp.float32),
            reward_min=jnp.array(jnp.inf, dtype=jnp.float32),
            reward_max=jnp.array(-jnp.inf, dtype=jnp.float32),
            model_error_ema=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def input_features(self, observation: Array, action: Array) -> Array:
        """Return features, leaving any invalid public operand visible as NaN."""
        features, features_valid, _ = self._safe_input_features(observation, action)
        return jnp.where(
            features_valid,
            features,
            jnp.full_like(features, jnp.nan),
        )

    def _safe_input_features(
        self,
        observation: Array,
        action: Array,
    ) -> tuple[Array, Bool[Array, ""], Array]:
        """Build finite internal operands and return their traced validity.

        The safe observation is returned separately because prediction decoding
        must not form arithmetic with a non-finite public observation before
        publishing the fail-visible result.
        """
        obs = _float32_operand(
            "observation", observation, (self._config.observation_dim,)
        )
        action_one_hot = self.encode_action(action)
        features_valid = jnp.all(jnp.isfinite(obs)) & jnp.all(
            jnp.isfinite(action_one_hot)
        )
        # Inf observation times a silent one-hot is 0*inf = NaN. Zero both
        # factors before the product so that product is never formed.
        safe_obs = jnp.where(features_valid, obs, jnp.zeros_like(obs))
        safe_action = jnp.where(
            features_valid, action_one_hot, jnp.zeros_like(action_one_hot)
        )
        if self._config.include_action_interactions:
            interactions = (safe_obs[:, None] * safe_action[None, :]).reshape((-1,))
            features = jnp.concatenate([safe_obs, safe_action, interactions], axis=0)
        else:
            features = jnp.concatenate([safe_obs, safe_action], axis=0)
        return features, features_valid, safe_obs

    @functools.partial(jax.jit, static_argnums=(0,))
    def encode_action(self, action: Array) -> Array:
        """Return the action code, leaving invalid discrete inputs visible."""
        action = _float32_operand("action", action, ())
        safe_action, action_valid = safe_discrete_action(
            action,
            self._config.n_actions,
        )
        encoded = jax.nn.one_hot(
            safe_action,
            self._config.n_actions,
            dtype=jnp.float32,
        )
        return jnp.where(
            action_valid,
            encoded,
            jnp.full_like(encoded, jnp.nan),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def targets(
        self,
        observation: Array,
        reward: Array,
        discount: Array,
        next_observation: Array,
    ) -> Array:
        """Build normalized ``[delta_obs, reward, discount]`` targets.

        The observation divisor here is the same ``observation_scale`` the decode in
        :meth:`_prediction_and_raw_diagnostics` multiplies by, so a head that fits
        its target reconstructs the delta it was shown.
        """
        obs = _float32_operand(
            "observation", observation, (self._config.observation_dim,)
        )
        next_obs = _float32_operand(
            "next_observation",
            next_observation,
            (self._config.observation_dim,),
        )
        reward_arr = _float32_operand("reward", reward, ())
        discount_arr = _float32_operand("discount", discount, ())
        # The divisor is the raw declared scale, not a floored one: the decode in
        # `_prediction_and_raw_diagnostics` multiplies by the raw value, and a head
        # that fits a target divided by anything else reconstructs that ratio of the
        # delta it trained on while still reporting a converged fit. Construction
        # refuses a scale that does not narrow to a normal float32, which is the only
        # way this divisor reaches zero.
        obs_scale = jnp.asarray(self._observation_scale, dtype=jnp.float32)
        normalized_delta = jnp.where(
            self._config.predict_delta,
            (next_obs - obs) / obs_scale,
            next_obs / obs_scale,
        )
        reward_target = jnp.reshape(
            reward_arr / self._config.reward_scale,
            (1,),
        )
        discount_target = jnp.reshape(discount_arr, (1,))
        return jnp.concatenate([normalized_delta, reward_target, discount_target], axis=0)

    def _prediction_and_raw_diagnostics(
        self,
        state: ActionConditionedWorldModelState,
        observation: Array,
        action: Array,
    ) -> tuple[WorldModelPrediction, Array, Array]:
        inputs, inputs_valid, safe_obs = self._safe_input_features(
            observation,
            action,
        )
        raw_predictions = self._learner.predict(state.learner_state, inputs)

        obs_scale = jnp.asarray(self._observation_scale, dtype=jnp.float32)
        normalized_delta = jnp.clip(
            raw_predictions[: self._config.observation_dim],
            -self._config.max_delta_scale,
            self._config.max_delta_scale,
        )
        decoded_next_observation = jnp.where(
            self._config.predict_delta,
            safe_obs + normalized_delta * obs_scale,
            normalized_delta * obs_scale,
        )

        has_bounds = state.step_count > 0
        low = state.observation_min - self._config.observation_clip_margin
        high = state.observation_max + self._config.observation_clip_margin
        clipped_next = jnp.clip(decoded_next_observation, low, high)
        next_observation = jnp.where(
            has_bounds, clipped_next, decoded_next_observation
        )

        raw_reward = raw_predictions[self._config.observation_dim] * self._config.reward_scale
        reward_low = state.reward_min - self._config.observation_clip_margin
        reward_high = state.reward_max + self._config.observation_clip_margin
        clipped_reward = jnp.clip(raw_reward, reward_low, reward_high)
        reward = jnp.where(has_bounds, clipped_reward, raw_reward)

        discount = jnp.clip(
            raw_predictions[self._config.observation_dim + 1],
            0.0,
            self._config.gamma,
        )

        invalid_observation = jnp.full_like(next_observation, jnp.nan)
        invalid_scalar = jnp.asarray(jnp.nan, dtype=jnp.float32)
        invalid_raw = jnp.full_like(raw_predictions, jnp.nan)
        next_observation = jnp.where(
            inputs_valid,
            next_observation,
            invalid_observation,
        )
        reward = jnp.where(inputs_valid, reward, invalid_scalar)
        decoded_next_observation = jnp.where(
            inputs_valid,
            decoded_next_observation,
            invalid_observation,
        )
        raw_reward = jnp.where(inputs_valid, raw_reward, invalid_scalar)
        discount = jnp.where(inputs_valid, discount, invalid_scalar)
        raw_predictions = jnp.where(
            inputs_valid,
            raw_predictions,
            invalid_raw,
        )

        return (
            WorldModelPrediction(
                next_observation=next_observation,
                reward=reward,
                raw_predictions=raw_predictions,
                discount=discount,
            ),
            decoded_next_observation,
            raw_reward,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: ActionConditionedWorldModelState,
        observation: Array,
        action: Array,
    ) -> WorldModelPrediction:
        """Predict the guarded next observation, reward, and discount."""
        prediction, _, _ = self._prediction_and_raw_diagnostics(
            state, observation, action
        )
        return prediction

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: ActionConditionedWorldModelState,
        observation: Array,
        action: Array,
        reward: Array,
        discount_or_next_observation: Array,
        next_observation: Array | None = None,
    ) -> WorldModelUpdateResult:
        """Update from one real transition."""
        if next_observation is None:
            discount = jnp.asarray(self._config.gamma, dtype=jnp.float32)
            next_observation = discount_or_next_observation
        else:
            discount = discount_or_next_observation

        obs = _float32_operand(
            "observation", observation, (self._config.observation_dim,)
        )
        action = _float32_operand("action", action, ())
        action_arr, action_valid = safe_discrete_action(
            action,
            self._config.n_actions,
        )
        reward_arr = _float32_operand("reward", reward, ())
        discount_arr = _float32_operand("discount", discount, ())
        next_obs = _float32_operand(
            "next_observation",
            next_observation,
            (self._config.observation_dim,),
        )
        inputs_valid = (
            jnp.all(jnp.isfinite(obs))
            & action_valid
            & jnp.all(jnp.isfinite(reward_arr))
            & jnp.all(jnp.isfinite(discount_arr))
            & jnp.all(discount_arr >= 0.0)
            & jnp.all(discount_arr <= 1.0)
            & jnp.all(jnp.isfinite(next_obs))
        )
        safe_obs = jnp.where(inputs_valid, obs, jnp.zeros_like(obs))
        safe_action = jnp.where(inputs_valid, action_arr, jnp.zeros_like(action_arr))
        safe_reward = jnp.where(inputs_valid, reward_arr, jnp.zeros_like(reward_arr))
        safe_discount = jnp.where(
            inputs_valid, discount_arr, jnp.zeros_like(discount_arr)
        )
        safe_next_obs = jnp.where(
            inputs_valid, next_obs, jnp.zeros_like(next_obs)
        )

        prediction, decoded_next_observation, decoded_reward = (
            self._prediction_and_raw_diagnostics(state, safe_obs, safe_action)
        )
        targets = self.targets(
            safe_obs, safe_reward, safe_discount, safe_next_obs
        )
        inputs = self.input_features(safe_obs, safe_action)
        learner_result = self._learner.update(state.learner_state, inputs, targets)

        next_observation_errors = decoded_next_observation - safe_next_obs
        observation_mse = jnp.mean(next_observation_errors**2)
        reward_error = decoded_reward - safe_reward
        discount_error = prediction.discount - safe_discount
        prediction_error = observation_mse + reward_error**2 + discount_error**2

        error_decay = jnp.asarray(self._config.error_decay, dtype=jnp.float32)
        next_error_ema = jnp.where(
            state.step_count == 0,
            prediction_error,
            error_decay * state.model_error_ema + (1.0 - error_decay) * prediction_error,
        )

        observed_stack_min = jnp.minimum(safe_obs, safe_next_obs)
        observed_stack_max = jnp.maximum(safe_obs, safe_next_obs)
        candidate_state = ActionConditionedWorldModelState(
            learner_state=learner_result.state,
            observation_min=jnp.minimum(state.observation_min, observed_stack_min),
            observation_max=jnp.maximum(state.observation_max, observed_stack_max),
            reward_min=jnp.minimum(state.reward_min, safe_reward),
            reward_max=jnp.maximum(state.reward_max, safe_reward),
            model_error_ema=next_error_ema,
            step_count=_saturating_increment(state.step_count),
        )

        diagnostics_finite = (
            floating_tree_is_finite(prediction)
            & jnp.all(jnp.isfinite(targets))
            & jnp.all(jnp.isfinite(learner_result.errors))
            & jnp.all(jnp.isfinite(learner_result.per_head_metrics))
            & jnp.isfinite(prediction_error)
            & jnp.isfinite(observation_mse)
            & jnp.isfinite(reward_error)
            & jnp.all(jnp.isfinite(next_observation_errors))
            & jnp.isfinite(discount_error)
        )
        update_applied = (
            inputs_valid
            & learner_result.update_applied
            & _action_world_model_state_is_valid(state, self._config.observation_dim)
            & floating_tree_is_finite(candidate_state)
            & diagnostics_finite
        )
        committed_state = select_transaction(update_applied, candidate_state, state)
        reported_learner_result = _rollback_multi_head_result(
            learner_result, state.learner_state, update_applied
        )
        reported_prediction = WorldModelPrediction(
            next_observation=neutralize_array(
                update_applied, prediction.next_observation
            ),
            reward=neutralize_array(update_applied, prediction.reward),
            raw_predictions=neutralize_array(
                update_applied, prediction.raw_predictions
            ),
            discount=neutralize_array(update_applied, prediction.discount),
        )

        return WorldModelUpdateResult(
            state=committed_state,
            prediction=reported_prediction,
            targets=neutralize_array(update_applied, targets),
            errors=reported_learner_result.errors,
            per_head_metrics=reported_learner_result.per_head_metrics,
            prediction_error=neutralize_array(update_applied, prediction_error),
            observation_mse=neutralize_array(update_applied, observation_mse),
            reward_error=neutralize_array(update_applied, reward_error),
            next_observation_errors=neutralize_array(
                update_applied, next_observation_errors
            ),
            discount_error=neutralize_array(update_applied, discount_error),
            learner_result=reported_learner_result,
            update_applied=update_applied,
        )

    def _validate_config(
        self, config: ActionConditionedWorldModelConfig
    ) -> ActionConditionedWorldModelConfig:
        """Fail closed on malformed configuration and return its canonical float32 form."""
        if type(config) is not ActionConditionedWorldModelConfig:
            raise ValueError("config must be an ActionConditionedWorldModelConfig")
        return config


def run_action_conditioned_world_model_learning_loop(
    model: ActionConditionedWorldModel,
    state: ActionConditionedWorldModelState,
    observations: Float[Array, "num_steps observation_dim"],
    actions: Array,
    rewards: Float[Array, " num_steps"],
    next_observations: Float[Array, "num_steps observation_dim"],
    discounts: Float[Array, " num_steps"] | None = None,
) -> ActionConditionedWorldModelLearningResult:
    """Run online one-step model learning over transition arrays."""
    observation_dim = model.config.observation_dim
    n_heads = observation_dim + 2
    observation_shape = _scan_array_metadata("observations", observations)
    if len(observation_shape) != 2 or observation_shape[1] != observation_dim:
        raise ValueError(f"observations must have shape (num_steps, {observation_dim})")
    num_steps = observation_shape[0]
    _require_scan_array("actions", actions, (num_steps,))
    _require_scan_array("rewards", rewards, (num_steps,))
    _require_scan_array(
        "next_observations", next_observations, (num_steps, observation_dim)
    )
    if discounts is not None:
        _require_scan_array("discounts", discounts, (num_steps,))
    _require_scan_resource(
        "action-conditioned world-model learning result",
        float32_scalars=(
            2 * num_steps * observation_dim + 6 * num_steps + 6 * num_steps * n_heads
        ),
        bool_scalars=num_steps,
    )
    observations = jnp.asarray(observations, dtype=jnp.float32)
    actions = jnp.asarray(actions, dtype=jnp.float32)
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    next_observations = jnp.asarray(next_observations, dtype=jnp.float32)
    if discounts is None:
        discounts = jnp.full(
            (num_steps,),
            jnp.asarray(model.config.gamma, dtype=jnp.float32),
            dtype=jnp.float32,
        )
    else:
        discounts = jnp.asarray(discounts, dtype=jnp.float32)

    def _scan_fn(
        carry: ActionConditionedWorldModelState,
        inputs: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[ActionConditionedWorldModelState, tuple[Array, ...]]:
        obs, action, reward, discount, next_obs = inputs
        result = model.update(carry, obs, action, reward, discount, next_obs)
        return result.state, (
            result.prediction.next_observation,
            result.prediction.reward,
            result.prediction.discount,
            result.prediction.raw_predictions,
            result.targets,
            result.errors,
            result.prediction_error,
            result.observation_mse,
            result.reward_error,
            result.next_observation_errors,
            result.discount_error,
            result.per_head_metrics,
            result.update_applied,
        )

    final_state, (
        next_observation_predictions,
        reward_predictions,
        discount_predictions,
        raw_predictions,
        targets,
        errors,
        prediction_errors,
        observation_mse,
        reward_errors,
        next_observation_errors,
        discount_errors,
        per_head_metrics,
        updates_applied,
    ) = jax.lax.scan(
        _scan_fn,
        state,
        (observations, actions, rewards, discounts, next_observations),
    )
    return ActionConditionedWorldModelLearningResult(
        state=final_state,
        next_observation_predictions=next_observation_predictions,
        reward_predictions=reward_predictions,
        discount_predictions=discount_predictions,
        raw_predictions=raw_predictions,
        targets=targets,
        errors=errors,
        prediction_errors=prediction_errors,
        observation_mse=observation_mse,
        reward_errors=reward_errors,
        next_observation_errors=next_observation_errors,
        discount_errors=discount_errors,
        per_head_metrics=per_head_metrics,
        updates_applied=updates_applied,
    )


__all__ = [
    "ActionConditionedWorldModel",
    "ActionConditionedWorldModelConfig",
    "ActionConditionedWorldModelLearningResult",
    "ActionConditionedWorldModelState",
    "OneStepWorldModel",
    "WorldModelConfig",
    "WorldModelLearningResult",
    "WorldModelPrediction",
    "WorldModelState",
    "WorldModelUpdateResult",
    "run_action_conditioned_world_model_learning_loop",
    "run_world_model_learning_loop",
]


@dataclasses.dataclass(frozen=True)
class WorldModelConfig:
    """Configuration for the Step 8 one-step world model.

    This compatibility surface predicts reward and next observation from
    ``concat(observation, action_encoding)``. It intentionally has no discount
    head; use :class:`ActionConditionedWorldModel` when dream rollouts need a
    learned discount/termination prediction.
    """

    observation_dim: int
    n_actions: int | None = 2
    action_dim: int = 1
    hidden_sizes: tuple[int, ...] = (64,)
    step_size: float = 0.05
    sparsity: float = 0.9
    leaky_relu_slope: float = 0.01
    use_layer_norm: bool = True
    predict_delta: bool = False
    utility_decay: float = 0.99

    def __post_init__(self) -> None:
        """Validate and canonicalize the complete static construction."""
        observation_dim = _require_int32("observation_dim", self.observation_dim, minimum=1)
        n_actions = (
            _require_int32("n_actions", self.n_actions, minimum=1)
            if self.n_actions is not None
            else None
        )
        action_dim = _require_int32("action_dim", self.action_dim, minimum=1)
        hidden_sizes = _validate_hidden_sizes(self.hidden_sizes)
        for name in ("use_layer_norm", "predict_delta"):
            object.__setattr__(self, name, _require_bool(name, getattr(self, name)))
        scalar_specs: tuple[tuple[str, dict[str, Any]], ...] = (
            ("sparsity", {"lower": 0.0, "upper": 1.0}),
            ("leaky_relu_slope", {"lower": 0.0, "upper": 1.0}),
            ("utility_decay", {"lower": 0.0, "upper": 1.0, "upper_inclusive": False}),
        )
        object.__setattr__(self, "observation_dim", observation_dim)
        object.__setattr__(self, "n_actions", n_actions)
        object.__setattr__(self, "action_dim", action_dim)
        object.__setattr__(self, "hidden_sizes", hidden_sizes)
        object.__setattr__(self, "step_size", _validated_step_size("step_size", self.step_size))
        for name, bounds in scalar_specs:
            object.__setattr__(
                self,
                name,
                _validated_config_float(name, getattr(self, name), **bounds),
            )
        if observation_dim == _INT32_MAX:
            raise ValueError("derived n_heads must fit in signed int32")
        _validate_world_model_resources(
            observation_dim=observation_dim,
            action_feature_dim=n_actions if n_actions is not None else action_dim,
            hidden_sizes=hidden_sizes,
            n_heads=observation_dim + 1,
            outer_state_scalars=1,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = dataclasses.asdict(self)
        payload["type"] = "WorldModelConfig"
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> WorldModelConfig:
        """Reconstruct from :meth:`to_config` output."""
        if not issubclass(type(config), Mapping):
            raise ValueError("config must be an actual mapping")
        try:
            payload = dict(config)
        except Exception as error:
            raise ValueError("config must be a readable mapping") from error
        if any(type(key) is not str for key in payload):
            raise ValueError("config keys must be exact strings")
        type_name = payload.pop("type", None)
        if type_name is not None and (
            type(type_name) is not str or type_name != cls.__name__
        ):
            raise ValueError("config type differs")
        if "hidden_sizes" in payload:
            payload["hidden_sizes"] = _serialized_sequence(
                "hidden_sizes", payload["hidden_sizes"]
            )
        return cls(**payload)


@chex.dataclass(frozen=True)
class WorldModelState:
    """State for :class:`OneStepWorldModel`."""

    learner_state: MultiHeadMLPState
    step_count: Array


@chex.dataclass(frozen=True)
class WorldModelLearningResult:
    """Scan result for :func:`run_world_model_learning_loop`."""

    state: WorldModelState
    reward_predictions: Float[Array, " num_steps"]
    next_observation_predictions: Float[Array, "num_steps observation_dim"]
    reward_errors: Float[Array, " num_steps"]
    next_observation_errors: Float[Array, "num_steps observation_dim"]
    per_head_metrics: Float[Array, "num_steps model_heads 3"]
    updates_applied: Bool[Array, " num_steps"]


class OneStepWorldModel:
    """Step 8 one-step environment predictor.

    Predicts one scalar reward head and one head per next-observation channel.
    Discrete actions are one-hot encoded; continuous/vector actions are passed
    through directly when ``n_actions=None``.
    """

    def __init__(
        self,
        config: WorldModelConfig,
        optimizer: AnyOptimizer | None = None,
        bounder: Bounder | None = None,
        normalizer: (
            Normalizer[EMANormalizerState] | Normalizer[WelfordNormalizerState] | None
        ) = None,
        head_optimizer: AnyOptimizer | None = None,
    ):
        """Initialize the model."""
        self._validate_config(config)
        self._config = config
        self._action_feature_dim = (
            config.n_actions if config.n_actions is not None else config.action_dim
        )
        self._learner = MultiHeadMLPLearner(
            n_heads=config.observation_dim + 1,
            hidden_sizes=config.hidden_sizes,
            optimizer=optimizer,
            step_size=config.step_size,
            bounder=bounder,
            gamma=0.0,
            lamda=0.0,
            normalizer=normalizer,
            sparsity=config.sparsity,
            leaky_relu_slope=config.leaky_relu_slope,
            use_layer_norm=config.use_layer_norm,
            head_optimizer=head_optimizer,
            utility_decay=config.utility_decay,
        )

    @property
    def config(self) -> WorldModelConfig:
        """Model configuration."""
        return self._config

    @property
    def learner(self) -> MultiHeadMLPLearner:
        """Underlying learner."""
        return self._learner

    @property
    def input_dim(self) -> int:
        """Encoded input dimensionality."""
        return self._config.observation_dim + self._action_feature_dim

    def to_config(self) -> dict[str, Any]:
        """Serialize model configuration."""
        return {
            "type": "OneStepWorldModel",
            "config": self._config.to_config(),
            "learner": self._learner.to_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> OneStepWorldModel:
        """Reconstruct from :meth:`to_config` output."""
        from alberta_framework.core.normalizers import normalizer_from_config
        from alberta_framework.core.optimizers import (
            bounder_from_config,
            optimizer_from_config,
        )

        payload = dict(config)
        payload.pop("type", None)
        model_config = WorldModelConfig.from_config(payload["config"])
        learner_cfg = dict(payload["learner"])
        optimizer = optimizer_from_config(learner_cfg["optimizer"])
        bounder_cfg = learner_cfg.get("bounder")
        normalizer_cfg = learner_cfg.get("normalizer")
        head_opt_cfg = learner_cfg.get("head_optimizer")
        return cls(
            model_config,
            optimizer=optimizer,
            bounder=bounder_from_config(bounder_cfg) if bounder_cfg is not None else None,
            normalizer=(
                normalizer_from_config(normalizer_cfg)
                if normalizer_cfg is not None
                else None
            ),
            head_optimizer=(
                optimizer_from_config(head_opt_cfg) if head_opt_cfg is not None else None
            ),
        )

    def init(self, key: Array) -> WorldModelState:
        """Initialize model state."""
        _preflight_world_model_update_working_set(
            observation_dim=self._config.observation_dim,
            action_feature_dim=self._action_feature_dim,
            hidden_sizes=self._config.hidden_sizes,
            n_heads=self._config.observation_dim + 1,
            outer_state_scalars=1,
        )
        return WorldModelState(
            learner_state=self._learner.init(self.input_dim, key),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def encode_action(self, action: Array) -> Array:
        """Encode a discrete or vector action."""
        if self._config.n_actions is not None:
            action = _float32_operand("action", action, ())
            safe_action, action_valid = safe_discrete_action(
                action,
                self._config.n_actions,
            )
            encoded = jax.nn.one_hot(
                safe_action,
                self._config.n_actions,
                dtype=jnp.float32,
            )
            return jnp.where(
                action_valid,
                encoded,
                jnp.full_like(encoded, jnp.nan),
            )
        return _float32_operand("action", action, (self._config.action_dim,))

    @functools.partial(jax.jit, static_argnums=(0,))
    def input_features(self, observation: Array, action: Array) -> Array:
        """Return ``concat(observation, encoded_action)``."""
        obs = _float32_operand(
            "observation", observation, (self._config.observation_dim,)
        )
        return jnp.concatenate([obs, self.encode_action(action)], axis=0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def targets(self, observation: Array, reward: Array, next_observation: Array) -> Array:
        """Build ``[reward, next_obs_or_delta]`` targets."""
        obs = _float32_operand(
            "observation", observation, (self._config.observation_dim,)
        )
        reward_arr = _float32_operand("reward", reward, ())
        next_obs = _float32_operand(
            "next_observation",
            next_observation,
            (self._config.observation_dim,),
        )
        obs_target = next_obs - obs if self._config.predict_delta else next_obs
        return jnp.concatenate(
            [jnp.reshape(reward_arr, (1,)), obs_target],
            axis=0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: WorldModelState,
        observation: Array,
        action: Array,
    ) -> WorldModelPrediction:
        """Predict reward and next observation."""
        obs = _float32_operand(
            "observation", observation, (self._config.observation_dim,)
        )
        raw_predictions = self._learner.predict(
            state.learner_state,
            self.input_features(obs, action),
        )
        reward = raw_predictions[0]
        obs_prediction = raw_predictions[1:]
        next_observation = (
            obs + obs_prediction if self._config.predict_delta else obs_prediction
        )
        return WorldModelPrediction(
            next_observation=next_observation,
            reward=reward,
            raw_predictions=raw_predictions,
            discount=jnp.array(jnp.nan, dtype=jnp.float32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: WorldModelState,
        observation: Array,
        action: Array,
        reward: Array,
        next_observation: Array,
    ) -> WorldModelUpdateResult:
        """Update from one real transition."""
        observation = _float32_operand(
            "observation", observation, (self._config.observation_dim,)
        )
        reward = _float32_operand("reward", reward, ())
        next_observation = _float32_operand(
            "next_observation",
            next_observation,
            (self._config.observation_dim,),
        )
        if self._config.n_actions is not None:
            action = _float32_operand("action", action, ())
            safe_action, action_valid = safe_discrete_action(
                action,
                self._config.n_actions,
            )
        else:
            action_arr = _float32_operand(
                "action", action, (self._config.action_dim,)
            )
            action_valid = jnp.all(jnp.isfinite(action_arr))
            safe_action = jnp.where(
                action_valid,
                action_arr,
                jnp.zeros_like(action_arr),
            )
        prediction = self.predict(state, observation, safe_action)
        targets = self.targets(observation, reward, next_observation)
        learner_result = self._learner.update(
            state.learner_state,
            self.input_features(observation, safe_action),
            targets,
        )
        next_obs = next_observation
        reward_arr = reward
        next_observation_errors = prediction.next_observation - next_obs
        reward_error = prediction.reward - reward_arr
        observation_mse = jnp.nanmean(next_observation_errors**2)
        prediction_error = jnp.nanmean(learner_result.errors**2)
        candidate_state = WorldModelState(
            learner_state=learner_result.state,
            step_count=_saturating_increment(state.step_count),
        )
        diagnostics_valid = (
            jnp.all(jnp.isfinite(prediction.next_observation))
            & jnp.isfinite(prediction.reward)
            & jnp.all(jnp.isfinite(prediction.raw_predictions))
            & jnp.all(~jnp.isinf(targets))
            & jnp.all(~jnp.isinf(learner_result.errors))
            & jnp.all(~jnp.isinf(learner_result.per_head_metrics))
            & jnp.all(~jnp.isinf(next_observation_errors))
            & ~jnp.isinf(reward_error)
            & ~jnp.isinf(observation_mse)
            & ~jnp.isinf(prediction_error)
        )
        update_applied = (
            action_valid
            & learner_result.update_applied
            & floating_tree_is_finite(state)
            & (state.step_count.shape == ())
            & (state.step_count.dtype == jnp.int32)
            & (state.step_count >= 0)
            & floating_tree_is_finite(candidate_state)
            & diagnostics_valid
        )
        new_state = select_transaction(update_applied, candidate_state, state)
        reported_learner_result = _rollback_multi_head_result(
            learner_result, state.learner_state, update_applied
        )
        reported_prediction = WorldModelPrediction(
            next_observation=neutralize_array(
                update_applied, prediction.next_observation
            ),
            reward=neutralize_array(update_applied, prediction.reward),
            raw_predictions=neutralize_array(
                update_applied, prediction.raw_predictions
            ),
            discount=prediction.discount,
        )
        reported_targets = jnp.where(
            jnp.isnan(targets),
            jnp.nan,
            neutralize_array(update_applied, targets),
        )
        reported_next_errors = jnp.where(
            jnp.isnan(next_observation_errors),
            jnp.nan,
            neutralize_array(update_applied, next_observation_errors),
        )
        return WorldModelUpdateResult(
            state=new_state,
            prediction=reported_prediction,
            targets=reported_targets,
            errors=reported_learner_result.errors,
            per_head_metrics=reported_learner_result.per_head_metrics,
            prediction_error=jnp.where(update_applied, prediction_error, 0.0),
            observation_mse=jnp.where(update_applied, observation_mse, 0.0),
            reward_error=jnp.where(update_applied, reward_error, 0.0),
            next_observation_errors=reported_next_errors,
            discount_error=jnp.array(jnp.nan, dtype=jnp.float32),
            learner_result=reported_learner_result,
            update_applied=update_applied,
        )

    def _validate_config(self, config: WorldModelConfig) -> None:
        if type(config) is not WorldModelConfig:
            raise ValueError("config must be a WorldModelConfig")


def run_world_model_learning_loop(
    model: OneStepWorldModel,
    state: WorldModelState,
    observations: Array,
    actions: Array,
    rewards: Array,
    next_observations: Array,
) -> WorldModelLearningResult:
    """Run one-step world-model learning with ``jax.lax.scan``."""
    observation_dim = model.config.observation_dim
    n_heads = observation_dim + 1
    observation_shape = _scan_array_metadata("observations", observations)
    if len(observation_shape) != 2 or observation_shape[1] != observation_dim:
        raise ValueError(f"observations must have shape (num_steps, {observation_dim})")
    num_steps = observation_shape[0]
    action_shape = (
        (num_steps,)
        if model.config.n_actions is not None
        else (num_steps, model.config.action_dim)
    )
    _require_scan_array("actions", actions, action_shape)
    _require_scan_array("rewards", rewards, (num_steps,))
    _require_scan_array(
        "next_observations", next_observations, (num_steps, observation_dim)
    )
    _require_scan_resource(
        "world-model learning result",
        float32_scalars=(
            2 * num_steps * observation_dim + 2 * num_steps + 3 * num_steps * n_heads
        ),
        bool_scalars=num_steps,
    )
    observations = jnp.asarray(observations, dtype=jnp.float32)
    actions = jnp.asarray(actions, dtype=jnp.float32)
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    next_observations = jnp.asarray(next_observations, dtype=jnp.float32)

    def step_fn(
        carry: WorldModelState,
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[WorldModelState, tuple[Array, Array, Array, Array, Array, Array]]:
        obs, action, reward, next_obs = inputs
        result = model.update(carry, obs, action, reward, next_obs)
        return result.state, (
            result.prediction.reward,
            result.prediction.next_observation,
            result.reward_error,
            result.next_observation_errors,
            result.per_head_metrics,
            result.update_applied,
        )

    final_state, (
        reward_predictions,
        next_observation_predictions,
        reward_errors,
        next_observation_errors,
        per_head_metrics,
        updates_applied,
    ) = jax.lax.scan(step_fn, state, (observations, actions, rewards, next_observations))
    return WorldModelLearningResult(
        state=final_state,
        reward_predictions=reward_predictions,
        next_observation_predictions=next_observation_predictions,
        reward_errors=reward_errors,
        next_observation_errors=next_observation_errors,
        per_head_metrics=per_head_metrics,
        updates_applied=updates_applied,
    )
