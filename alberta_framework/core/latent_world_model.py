# mypy: disable-error-code="call-arg"
"""Latent action-conditioned world model with surprise diagnostics.

This module is a low-dimensional, online analogue of latent-prediction world
models in the JEPA family (LeCun 2022, "A Path Towards Autonomous Machine
Intelligence") such as LeWorldModel (LeWM; arXiv:2603.19312): dynamics are
predicted in an embedding space, ``(z_t, a_t) -> (z_{t+1}, r, gamma)``, rather
than in observation space.

Two deliberate simplifications keep the first research question about
predictive latent dynamics, surprise, and dream selection rather than
unstable joint representation learning:

1. **Fixed random encoder by default** — observations are embedded by an
   untrained ``tanh`` random projection, i.e. a random-features map (cf.
   Rahimi & Recht 2007; the sparse FTL world model in
   :mod:`alberta_framework.core.ftl_world_model` makes the same trade for
   the same reason).  Because the default encoder receives no gradient,
   representation collapse cannot be caused by training here.
2. **Collapse tracking as diagnostic, not loss** — an EMA of per-dimension
   target-latent variance is maintained, and ``collapse_score`` reports the
   fraction of latent dimensions whose std falls below ``min_latent_std``.
   With the fixed encoder it flags degenerate inputs or encoders; with the
   trainable-encoder variant below it becomes an anti-collapse gate.

**Trainable-encoder variant** (``encoder_learning=True``, default off): the
encoder additionally takes one bounded gradient step per accepted transition
on the latent prediction loss
``mean((z_t + predicted_delta - encode(next_obs))**2)``, differentiated
through both the predictor input branch and the target branch with the
predictor weights held fixed at their pre-update values.  The step is
elementwise-clipped to ``max_encoder_update`` and is skipped fail-closed
whenever the fresh ``collapse_score`` exceeds
``encoder_collapse_gate_threshold`` or any gradient/candidate value is
non-finite — a skipped step leaves the encoder bytes unchanged.  With the
flag at its default ``False`` the update path is exactly the fixed-encoder
computation.

This module is development/L0 mechanism code and makes no efficacy,
scientific-evidence, or SOTA claim.  ("Development" and "L0" are this
repository's evidence vocabulary, defined in ``docs/status.md``: L0
covers mechanism-level checks — API, shape, finite-value, serialization,
local update behavior — and development modules can never promote
scientific claims.)
"""

from __future__ import annotations

import dataclasses
import functools
import operator
from collections.abc import Mapping
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.multi_head_learner import (
    AnyOptimizer,
    MultiHeadMLPLearner,
    MultiHeadMLPState,
    MultiHeadMLPUpdateResult,
)
from alberta_framework.core.optimizers import Bounder
from alberta_framework.core.types import TraceMode
from alberta_framework.core.update_safety import (
    floating_tree_is_finite,
    neutralize_array,
    safe_discrete_action,
    select_transaction,
)

EVIDENCE_LEVEL = "L0"
SCIENTIFIC_PROMOTION_ALLOWED = False
_INT32_MAX = 2**31 - 1
# Smallest normal binary32 (float(np.finfo(np.float32).tiny)); every admitted
# observation_scale must be at least this so its float32 reciprocal stays
# finite when the encoder divides by the declared scale.
_SMALLEST_NORMAL_FLOAT32 = 1.1754943508222875e-38
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


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _validate_hidden_sizes(sizes: object) -> tuple[int, ...]:
    if type(sizes) is not tuple:
        raise ValueError("hidden_sizes must be an actual tuple")
    return tuple(
        _require_int32(f"hidden_sizes[{index}]", size, minimum=1)
        for index, size in enumerate(sizes)
    )


def _require_int32_product(name: str, left: int, right: int) -> int:
    if left > _INT32_MAX // right:
        raise ValueError(f"derived {name} must fit in signed int32")
    return left * right


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _saturating_int32_increment(value: Array) -> Array:
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return jnp.where(value < maximum, value + jnp.asarray(1, dtype=jnp.int32), maximum)


def _latent_direct_state_scalars(
    *,
    observation_dim: int,
    n_actions: int,
    latent_dim: int,
    hidden_sizes: tuple[int, ...],
    include_action_interactions: bool,
) -> int:
    """Named persist scalars already preflighted by ``LatentWorldModelConfig``."""
    input_dim = latent_dim + n_actions
    if include_action_interactions:
        input_dim += latent_dim * n_actions
    layer_sizes = (input_dim, *hidden_sizes)
    trunk_parameters = sum(
        fan_out * (fan_in + 1)
        for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:], strict=False)
    )
    n_heads = latent_dim + 2
    head_input = hidden_sizes[-1] if hidden_sizes else input_dim
    head_parameters = n_heads * (head_input + 1)
    learner_direct_scalars = (
        2 * (trunk_parameters + head_parameters) + sum(hidden_sizes) + 3
    )
    outer_state_scalars = observation_dim * latent_dim + 3 * latent_dim + 4
    return learner_direct_scalars + outer_state_scalars


def _latent_update_result_extras_bytes(*, latent_dim: int) -> int:
    """Returned ``LatentWorldModelUpdateResult`` extras excluding persist.

    Nested ``learner_result.state`` is callee persist and is already counted
    in the simultaneous persist copies.  These extras are the published
    prediction, target, error, metric, and learner-result diagnostic leaves.
    """
    n_heads = latent_dim + 2
    return 12 * latent_dim + 44 * n_heads + 64


def _latent_update_working_set_bytes(
    *,
    observation_dim: int,
    n_actions: int,
    latent_dim: int,
    hidden_sizes: tuple[int, ...],
    include_action_interactions: bool,
) -> int:
    """Source persist, proposed persist, committed persist, and returned extras."""
    persist_bytes = 4 * _latent_direct_state_scalars(
        observation_dim=observation_dim,
        n_actions=n_actions,
        latent_dim=latent_dim,
        hidden_sizes=hidden_sizes,
        include_action_interactions=include_action_interactions,
    )
    extras_bytes = _latent_update_result_extras_bytes(latent_dim=latent_dim)
    return 3 * persist_bytes + extras_bytes


def _preflight_latent_update_working_set(
    *,
    observation_dim: int,
    n_actions: int,
    latent_dim: int,
    hidden_sizes: tuple[int, ...],
    include_action_interactions: bool,
) -> None:
    """Reject an update envelope the host cannot name in signed int32."""
    working_set_bytes = _latent_update_working_set_bytes(
        observation_dim=observation_dim,
        n_actions=n_actions,
        latent_dim=latent_dim,
        hidden_sizes=hidden_sizes,
        include_action_interactions=include_action_interactions,
    )
    if working_set_bytes > _INT32_MAX:
        raise ValueError(
            "latent world-model update working set byte count must fit signed int32"
        )


@dataclasses.dataclass(frozen=True)
class LatentWorldModelConfig:
    """Configuration for :class:`LatentWorldModel`.

    The encoder is fixed by default (``encoder_learning=False``), keeping
    anti-collapse a pure diagnostic.  Setting ``encoder_learning=True``
    enables the development-only (L0) trainable-encoder variant: one bounded
    encoder gradient step per accepted transition, elementwise-clipped to
    ``max_encoder_update`` and skipped whenever the fresh ``collapse_score``
    exceeds ``encoder_collapse_gate_threshold`` (the anti-collapse diagnostic
    acting as a gate) or any gradient/candidate value is non-finite.
    """

    observation_dim: int
    n_actions: int
    latent_dim: int = 8
    gamma: float = 0.99
    observation_scale: tuple[float, ...] | None = None
    reward_scale: float = 1.0
    encoder_scale: float = 1.0
    encoder_bias_scale: float = 0.0
    predict_delta: bool = True
    hidden_sizes: tuple[int, ...] = (64,)
    step_size: float = 0.03
    sparsity: float = 0.9
    leaky_relu_slope: float = 0.01
    use_layer_norm: bool = True
    trace_mode: TraceMode = TraceMode.ACCUMULATING
    utility_decay: float = 0.99
    surprise_decay: float = 0.99
    collapse_decay: float = 0.99
    min_latent_std: float = 0.05
    max_latent_delta: float = 5.0
    include_action_interactions: bool = False
    encoder_learning: bool = False
    encoder_step_size: float = 0.01
    max_encoder_update: float = 0.1
    encoder_collapse_gate_threshold: float = 0.0

    def __post_init__(self) -> None:
        """Validate and canonicalize the complete static construction."""
        object.__setattr__(
            self,
            "observation_dim",
            _require_int32("observation_dim", self.observation_dim, minimum=1),
        )
        object.__setattr__(
            self,
            "n_actions",
            _require_int32("n_actions", self.n_actions, minimum=1),
        )
        object.__setattr__(
            self,
            "latent_dim",
            _require_int32("latent_dim", self.latent_dim, minimum=1),
        )
        object.__setattr__(
            self,
            "hidden_sizes",
            _validate_hidden_sizes(self.hidden_sizes),
        )
        for name in (
            "predict_delta",
            "use_layer_norm",
            "include_action_interactions",
            "encoder_learning",
        ):
            object.__setattr__(self, name, _require_bool(name, getattr(self, name)))
        if type(self.trace_mode) is not TraceMode:
            raise ValueError("trace_mode must be a TraceMode")

        observation_scale = self.observation_scale
        if observation_scale is not None:
            if type(observation_scale) is not tuple:
                raise ValueError("observation_scale must be an actual tuple or None")
            if len(observation_scale) != self.observation_dim:
                raise ValueError("observation_scale length must equal observation_dim")
            observation_scale = tuple(
                validated_float32_scalar(
                    f"observation_scale[{index}]",
                    scale,
                    lower=_SMALLEST_NORMAL_FLOAT32,
                )
                for index, scale in enumerate(observation_scale)
            )
            object.__setattr__(self, "observation_scale", observation_scale)

        scalar_specs: tuple[tuple[str, dict[str, Any]], ...] = (
            ("gamma", {"lower": 0.0, "upper": 1.0}),
            ("reward_scale", {"positive": True}),
            ("encoder_scale", {"positive": True}),
            ("encoder_bias_scale", {"lower": 0.0}),
            ("step_size", {"positive": True}),
            ("sparsity", {"lower": 0.0, "upper": 1.0}),
            ("leaky_relu_slope", {"lower": 0.0, "upper": 1.0}),
            ("utility_decay", {"lower": 0.0, "upper": 1.0, "upper_inclusive": False}),
            ("surprise_decay", {"lower": 0.0, "upper": 1.0, "upper_inclusive": False}),
            ("collapse_decay", {"lower": 0.0, "upper": 1.0, "upper_inclusive": False}),
            ("min_latent_std", {"lower": 0.0}),
            ("max_latent_delta", {"positive": True}),
            ("encoder_step_size", {"positive": True}),
            ("max_encoder_update", {"positive": True}),
            ("encoder_collapse_gate_threshold", {"lower": 0.0, "upper": 1.0}),
        )
        for name, bounds in scalar_specs:
            object.__setattr__(
                self,
                name,
                validated_float32_scalar(name, getattr(self, name), **bounds),
            )

        _require_int32_product(
            "encoder_matrix_scalars", self.observation_dim, self.latent_dim
        )
        if self.latent_dim > _INT32_MAX - 2:
            raise ValueError("derived n_heads must fit in signed int32")
        n_heads = self.latent_dim + 2
        if self.latent_dim > _INT32_MAX - self.n_actions:
            raise ValueError("derived input_dim must fit in signed int32")
        input_dim = self.latent_dim + self.n_actions
        if self.include_action_interactions:
            interactions = _require_int32_product(
                "action_interaction_scalars", self.latent_dim, self.n_actions
            )
            if input_dim > _INT32_MAX - interactions:
                raise ValueError("derived input_dim must fit in signed int32")
            input_dim += interactions
        layer_sizes = (input_dim, *self.hidden_sizes)
        for index, (fan_in, fan_out) in enumerate(
            zip(layer_sizes, layer_sizes[1:], strict=False)
        ):
            _require_int32_product(f"hidden_layer[{index}]_scalars", fan_in, fan_out)
        head_input = self.hidden_sizes[-1] if self.hidden_sizes else input_dim
        _require_int32_product("head_weight_scalars", n_heads, head_input)
        combined_direct_state_scalars = _latent_direct_state_scalars(
            observation_dim=self.observation_dim,
            n_actions=self.n_actions,
            latent_dim=self.latent_dim,
            hidden_sizes=self.hidden_sizes,
            include_action_interactions=self.include_action_interactions,
        )
        for name, value in (
            ("combined_direct_state_scalars", combined_direct_state_scalars),
            ("combined_direct_state_bytes", 4 * combined_direct_state_scalars),
        ):
            if value > _INT32_MAX:
                raise ValueError(f"derived {name} must fit in signed int32")
        _preflight_latent_update_working_set(
            observation_dim=self.observation_dim,
            n_actions=self.n_actions,
            latent_dim=self.latent_dim,
            hidden_sizes=self.hidden_sizes,
            include_action_interactions=self.include_action_interactions,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = dataclasses.asdict(self)
        payload["type"] = "LatentWorldModelConfig"
        payload["hidden_sizes"] = list(self.hidden_sizes)
        payload["trace_mode"] = self.trace_mode.value
        if self.observation_scale is not None:
            payload["observation_scale"] = list(self.observation_scale)
        return payload

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> LatentWorldModelConfig:
        """Reconstruct from :meth:`to_config` output."""
        if not issubclass(type(config), Mapping):
            raise ValueError("config must be an actual mapping")
        payload = dict(config)
        config_type = payload.pop("type", None)
        if type(config_type) is not str or config_type != "LatentWorldModelConfig":
            raise ValueError("unexpected latent world model config type")
        expected_fields = {field.name for field in dataclasses.fields(cls)}
        legacy_optional = {
            "encoder_learning",
            "encoder_step_size",
            "max_encoder_update",
            "encoder_collapse_gate_threshold",
        }
        if set(payload) - expected_fields or (expected_fields - legacy_optional) - set(payload):
            raise ValueError("config fields do not match the serialized schema")
        if "hidden_sizes" in payload:
            if type(payload["hidden_sizes"]) is not list:
                raise ValueError("hidden_sizes must be a list in serialized config")
            payload["hidden_sizes"] = tuple(payload["hidden_sizes"])
        if "observation_scale" in payload and payload["observation_scale"] is not None:
            if type(payload["observation_scale"]) is not list:
                raise ValueError("observation_scale must be a list in serialized config")
            payload["observation_scale"] = tuple(payload["observation_scale"])
        if "trace_mode" in payload:
            if type(payload["trace_mode"]) is not str:
                raise ValueError("trace_mode must be a string in serialized config")
            payload["trace_mode"] = TraceMode(payload["trace_mode"])
        return cls(**cast(dict[str, Any], payload))


@chex.dataclass(frozen=True)
class LatentWorldModelState:
    """Immutable state for :class:`LatentWorldModel`."""

    encoder_matrix: Float[Array, "observation_dim latent_dim"]
    encoder_bias: Float[Array, " latent_dim"]
    learner_state: MultiHeadMLPState
    latent_mean_ema: Float[Array, " latent_dim"]
    latent_var_ema: Float[Array, " latent_dim"]
    surprise_ema: Float[Array, ""]
    prediction_error_ema: Float[Array, ""]
    collapse_score_ema: Float[Array, ""]
    step_count: Array


@chex.dataclass(frozen=True)
class LatentWorldModelPrediction:
    """Decoded latent dynamics prediction."""

    latent: Float[Array, " latent_dim"]
    next_latent: Float[Array, " latent_dim"]
    reward: Float[Array, ""]
    discount: Float[Array, ""]
    raw_predictions: Float[Array, " model_heads"]


@chex.dataclass(frozen=True)
class LatentWorldModelUpdateResult:
    """Result from one real latent-dynamics update.

    ``encoder_update_applied``, ``encoder_collapse_gated``, and
    ``encoder_gradient_norm`` describe the optional trainable-encoder step;
    with ``encoder_learning=False`` they are the constants ``False``,
    ``False``, and ``0.0``.
    """

    state: LatentWorldModelState
    prediction: LatentWorldModelPrediction
    target_next_latent: Float[Array, " latent_dim"]
    targets: Float[Array, " model_heads"]
    errors: Float[Array, " model_heads"]
    surprise: Float[Array, ""]
    reward_error: Float[Array, ""]
    discount_error: Float[Array, ""]
    prediction_error: Float[Array, ""]
    latent_std_mean: Float[Array, ""]
    collapse_score: Float[Array, ""]
    per_head_metrics: Float[Array, "model_heads 3"]
    learner_result: MultiHeadMLPUpdateResult
    encoder_update_applied: Bool[Array, ""]
    encoder_collapse_gated: Bool[Array, ""]
    encoder_gradient_norm: Float[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class LatentWorldModelLearningResult:
    """Scan result for latent world-model learning."""

    state: LatentWorldModelState
    latent_predictions: Float[Array, "num_steps latent_dim"]
    next_latent_predictions: Float[Array, "num_steps latent_dim"]
    reward_predictions: Float[Array, " num_steps"]
    discount_predictions: Float[Array, " num_steps"]
    target_next_latents: Float[Array, "num_steps latent_dim"]
    surprises: Float[Array, " num_steps"]
    prediction_errors: Float[Array, " num_steps"]
    reward_errors: Float[Array, " num_steps"]
    discount_errors: Float[Array, " num_steps"]
    latent_std_means: Float[Array, " num_steps"]
    collapse_scores: Float[Array, " num_steps"]
    per_head_metrics: Float[Array, "num_steps model_heads metrics"]
    encoder_updates_applied: Bool[Array, " num_steps"]
    encoder_collapse_gates: Bool[Array, " num_steps"]
    encoder_gradient_norms: Float[Array, " num_steps"]
    updates_applied: Bool[Array, " num_steps"]


def _rollback_multi_head_result(
    result: MultiHeadMLPUpdateResult,
    previous_state: MultiHeadMLPState,
    outer_update_applied: Array,
) -> MultiHeadMLPUpdateResult:
    """Make the nested learner result match the committed latent transaction."""
    requested = ~jnp.isnan(result.errors)
    return cast(
        MultiHeadMLPUpdateResult,
        dataclasses.replace(
            cast(Any, result),
            state=select_transaction(outer_update_applied, result.state, previous_state),
            predictions=neutralize_array(outer_update_applied, result.predictions),
            errors=jnp.where(
                requested,
                neutralize_array(outer_update_applied, result.errors),
                jnp.nan,
            ),
            per_head_metrics=jnp.where(
                requested[:, None],
                neutralize_array(outer_update_applied, result.per_head_metrics),
                jnp.nan,
            ),
            trunk_bounding_metric=neutralize_array(
                outer_update_applied, result.trunk_bounding_metric
            ),
            post_step_words=jnp.where(
                outer_update_applied, result.post_step_words, result.pre_step_words
            ),
            update_applied=result.update_applied & outer_update_applied,
        ),
    )


class LatentWorldModel:
    """Latent model for ``(z_t, a_t) -> (z_{t+1}, r, gamma)``.

    The encoder is a fixed ``tanh`` random projection by default; with
    ``encoder_learning=True`` (development-only, L0) it additionally takes
    bounded, collapse-gated gradient steps on the latent prediction loss.
    """

    def __init__(
        self,
        config: LatentWorldModelConfig,
        optimizer: AnyOptimizer | None = None,
        bounder: Bounder | None = None,
        head_optimizer: AnyOptimizer | None = None,
    ):
        """Initialize the latent world model."""
        config = self._validate_config(config)
        self._config = config
        self._observation_scale = (
            tuple(1.0 for _ in range(config.observation_dim))
            if config.observation_scale is None
            else tuple(config.observation_scale)
        )
        self._learner = MultiHeadMLPLearner(
            n_heads=config.latent_dim + 2,
            hidden_sizes=config.hidden_sizes,
            optimizer=optimizer,
            step_size=config.step_size,
            bounder=bounder,
            gamma=0.0,
            lamda=0.0,
            sparsity=config.sparsity,
            leaky_relu_slope=config.leaky_relu_slope,
            use_layer_norm=config.use_layer_norm,
            head_optimizer=head_optimizer,
            trace_mode=config.trace_mode,
            utility_decay=config.utility_decay,
        )

    @property
    def config(self) -> LatentWorldModelConfig:
        """Model configuration."""
        return self._config

    @property
    def learner(self) -> MultiHeadMLPLearner:
        """Underlying multi-head predictor."""
        return self._learner

    @property
    def input_dim(self) -> int:
        """Latent predictor input dimension."""
        base_dim = self._config.latent_dim + self._config.n_actions
        if self._config.include_action_interactions:
            return base_dim + self._config.latent_dim * self._config.n_actions
        return base_dim

    @property
    def n_heads(self) -> int:
        """Number of prediction heads."""
        return self._config.latent_dim + 2

    def to_config(self) -> dict[str, Any]:
        """Serialize model configuration and learner components."""
        return {
            "type": "LatentWorldModel",
            "config": self._config.to_config(),
            "learner": self._learner.to_config(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> LatentWorldModel:
        """Reconstruct from :meth:`to_config` output."""
        from alberta_framework.core.optimizers import (
            bounder_from_config,
            optimizer_from_config,
        )

        if not issubclass(type(config), Mapping):
            raise ValueError("model config must be an actual mapping")
        if set(config) != {"type", "config", "learner"}:
            raise ValueError("model config fields do not match the serialized schema")
        if config.get("type") != "LatentWorldModel":
            raise ValueError("unexpected latent world model type")
        payload = dict(config)
        payload.pop("type")
        if not issubclass(type(payload["config"]), Mapping) or type(payload["learner"]) is not dict:
            raise ValueError("nested model configs must use canonical mappings")
        model_config = LatentWorldModelConfig.from_config(
            cast(Mapping[str, object], payload["config"])
        )
        learner_cfg = dict(payload["learner"])
        optimizer = optimizer_from_config(learner_cfg["optimizer"])
        bounder_cfg = learner_cfg.get("bounder")
        head_opt_cfg = learner_cfg.get("head_optimizer")
        return cls(
            model_config,
            optimizer=optimizer,
            bounder=bounder_from_config(bounder_cfg) if bounder_cfg is not None else None,
            head_optimizer=(
                optimizer_from_config(head_opt_cfg) if head_opt_cfg is not None else None
            ),
        )

    def init(self, key: Array) -> LatentWorldModelState:
        """Initialize fixed encoder and predictor state."""
        encoder_key, bias_key, learner_key = jr.split(key, 3)
        obs_dim = self._config.observation_dim
        latent_dim = self._config.latent_dim
        encoder_matrix = (
            jr.normal(encoder_key, (obs_dim, latent_dim), dtype=jnp.float32)
            * self._config.encoder_scale
            / jnp.sqrt(jnp.asarray(obs_dim, dtype=jnp.float32))
        )
        encoder_bias = (
            jr.uniform(
                bias_key,
                (latent_dim,),
                minval=-self._config.encoder_bias_scale,
                maxval=self._config.encoder_bias_scale,
                dtype=jnp.float32,
            )
            if self._config.encoder_bias_scale > 0.0
            else jnp.zeros((latent_dim,), dtype=jnp.float32)
        )
        if not bool(
            jnp.all(jnp.isfinite(encoder_matrix))
            & jnp.all(jnp.isfinite(encoder_bias))
        ):
            raise ValueError("encoder initialization produced non-finite parameters")
        return LatentWorldModelState(
            encoder_matrix=encoder_matrix,
            encoder_bias=encoder_bias,
            learner_state=self._learner.init(self.input_dim, learner_key),
            latent_mean_ema=jnp.zeros((latent_dim,), dtype=jnp.float32),
            latent_var_ema=jnp.zeros((latent_dim,), dtype=jnp.float32),
            surprise_ema=jnp.array(0.0, dtype=jnp.float32),
            prediction_error_ema=jnp.array(0.0, dtype=jnp.float32),
            collapse_score_ema=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def _encode_with(
        self,
        encoder_matrix: Array,
        encoder_bias: Array,
        observation: Array,
    ) -> Float[Array, " latent_dim"]:
        """Encode one observation with explicit (differentiable) encoder params."""
        obs = jnp.asarray(observation, dtype=jnp.float32).reshape((self._config.observation_dim,))
        # Divide by the declared per-dimension scale. The validator bounds every
        # admitted scale at or above the smallest normal float32, so ``1/scale``
        # stays finite; no ``max(scale, 1e-6)`` floor silently substitutes it.
        scale = jnp.asarray(self._observation_scale, dtype=jnp.float32)
        scaled = obs / scale
        return jnp.tanh(scaled @ encoder_matrix + encoder_bias)

    @functools.partial(jax.jit, static_argnums=(0,))
    def encode(
        self,
        state: LatentWorldModelState,
        observation: Array,
    ) -> Float[Array, " latent_dim"]:
        """Encode one observation into the current latent space."""
        return self._encode_with(state.encoder_matrix, state.encoder_bias, observation)

    @functools.partial(jax.jit, static_argnums=(0,))
    def input_features_from_latent(
        self,
        latent: Array,
        action: Array,
    ) -> Float[Array, " input_dim"]:
        """Return ``concat(latent, one_hot(action), optional interactions)``."""
        z = jnp.asarray(latent, dtype=jnp.float32).reshape((self._config.latent_dim,))
        safe_action, action_valid = safe_discrete_action(
            action,
            self._config.n_actions,
        )
        action_one_hot = jax.nn.one_hot(
            safe_action,
            self._config.n_actions,
            dtype=jnp.float32,
        )
        latent_valid = jnp.all(jnp.isfinite(z))
        features_valid = action_valid & latent_valid
        # Inf latent times a silent one-hot is 0*inf = NaN. Zero both
        # factors before the product; keep the invalid-action NaN reject.
        safe_z = jnp.where(features_valid, z, jnp.zeros_like(z))
        safe_action_one_hot = jnp.where(
            features_valid, action_one_hot, jnp.zeros_like(action_one_hot)
        )
        features = [safe_z, safe_action_one_hot]
        if self._config.include_action_interactions:
            interactions = (safe_z[:, None] * safe_action_one_hot[None, :]).reshape((-1,))
            features.append(interactions)
        encoded = jnp.concatenate(features, axis=0)
        return jnp.where(
            features_valid,
            encoded,
            jnp.full_like(encoded, jnp.nan),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def input_features(
        self,
        state: LatentWorldModelState,
        observation: Array,
        action: Array,
    ) -> Float[Array, " input_dim"]:
        """Encode observation and return latent predictor inputs."""
        return cast(
            Array,
            self.input_features_from_latent(self.encode(state, observation), action),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def targets(
        self,
        state: LatentWorldModelState,
        observation: Array,
        reward: Array,
        discount: Array,
        next_observation: Array,
    ) -> Float[Array, " model_heads"]:
        """Build ``[next_latent_or_delta, reward, discount]`` targets."""
        latent = self.encode(state, observation)
        next_latent = self.encode(state, next_observation)
        latent_target = jnp.where(
            self._config.predict_delta,
            next_latent - latent,
            next_latent,
        )
        reward_target = jnp.reshape(
            jnp.asarray(reward, dtype=jnp.float32) / self._config.reward_scale,
            (1,),
        )
        discount_target = jnp.reshape(jnp.asarray(discount, dtype=jnp.float32), (1,))
        return jnp.concatenate([latent_target, reward_target, discount_target], axis=0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict_from_latent(
        self,
        state: LatentWorldModelState,
        latent: Array,
        action: Array,
    ) -> LatentWorldModelPrediction:
        """Predict the next latent, reward, and discount from a latent state."""
        z = jnp.asarray(latent, dtype=jnp.float32).reshape((self._config.latent_dim,))
        raw_predictions = self._learner.predict(
            state.learner_state,
            self.input_features_from_latent(z, action),
        )
        latent_part = jnp.clip(
            raw_predictions[: self._config.latent_dim],
            -self._config.max_latent_delta,
            self._config.max_latent_delta,
        )
        next_latent = jnp.where(
            self._config.predict_delta,
            z + latent_part,
            latent_part,
        )
        reward = raw_predictions[self._config.latent_dim] * self._config.reward_scale
        discount = jnp.clip(
            raw_predictions[self._config.latent_dim + 1],
            0.0,
            self._config.gamma,
        )
        return LatentWorldModelPrediction(
            latent=z,
            next_latent=next_latent,
            reward=reward,
            discount=discount,
            raw_predictions=raw_predictions,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: LatentWorldModelState,
        observation: Array,
        action: Array,
    ) -> LatentWorldModelPrediction:
        """Predict from raw observation by first encoding it."""
        return cast(
            LatentWorldModelPrediction,
            self.predict_from_latent(state, self.encode(state, observation), action),
        )

    def _encoder_step(
        self,
        state: LatentWorldModelState,
        observation: Array,
        action: Array,
        next_observation: Array,
        collapse_score: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Return one bounded, collapse-gated encoder update candidate.

        The latent prediction loss is differentiated with respect to the
        encoder parameters through both the predictor input branch and the
        target branch, with the predictor weights held fixed at their
        pre-update values.  The step is elementwise-clipped to
        ``max_encoder_update`` and skipped fail-closed — encoder bytes
        unchanged — when the fresh ``collapse_score`` exceeds
        ``encoder_collapse_gate_threshold`` or any gradient/candidate value
        is non-finite.

        Returns:
            ``(encoder_matrix, encoder_bias, applied, gate_blocked,
            gradient_norm)``.
        """
        config = self._config

        def latent_prediction_loss(
            encoder_matrix: Array,
            encoder_bias: Array,
        ) -> Array:
            latent = self._encode_with(encoder_matrix, encoder_bias, observation)
            target_next_latent = self._encode_with(encoder_matrix, encoder_bias, next_observation)
            raw_predictions = self._learner.predict(
                state.learner_state,
                self.input_features_from_latent(latent, action),
            )
            latent_part = jnp.clip(
                raw_predictions[: config.latent_dim],
                -config.max_latent_delta,
                config.max_latent_delta,
            )
            predicted_next_latent = jnp.where(
                config.predict_delta,
                latent + latent_part,
                latent_part,
            )
            return jnp.mean((predicted_next_latent - target_next_latent) ** 2)

        grad_matrix, grad_bias = jax.grad(latent_prediction_loss, argnums=(0, 1))(
            state.encoder_matrix,
            state.encoder_bias,
        )
        grads_finite = jnp.all(jnp.isfinite(grad_matrix)) & jnp.all(jnp.isfinite(grad_bias))
        gradient_norm = jnp.where(
            grads_finite,
            jnp.sqrt(jnp.sum(grad_matrix**2) + jnp.sum(grad_bias**2)),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        step_size = jnp.asarray(config.encoder_step_size, dtype=jnp.float32)
        bound = jnp.asarray(config.max_encoder_update, dtype=jnp.float32)
        candidate_matrix = state.encoder_matrix + jnp.clip(-step_size * grad_matrix, -bound, bound)
        candidate_bias = state.encoder_bias + jnp.clip(-step_size * grad_bias, -bound, bound)
        gate_blocked = collapse_score > jnp.asarray(
            config.encoder_collapse_gate_threshold, dtype=jnp.float32
        )
        applied = (
            grads_finite
            & ~gate_blocked
            & jnp.all(jnp.isfinite(candidate_matrix))
            & jnp.all(jnp.isfinite(candidate_bias))
        )
        encoder_matrix = jnp.where(applied, candidate_matrix, state.encoder_matrix)
        encoder_bias = jnp.where(applied, candidate_bias, state.encoder_bias)
        return encoder_matrix, encoder_bias, applied, gate_blocked, gradient_norm

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: LatentWorldModelState,
        observation: Array,
        action: Array,
        reward: Array,
        discount: Array,
        next_observation: Array,
    ) -> LatentWorldModelUpdateResult:
        """Update from one real transition.

        With ``encoder_learning=True`` the fixed-encoder transaction is
        followed by one bounded, collapse-gated encoder gradient step; the
        updated encoder takes effect from the next event.  With the default
        ``encoder_learning=False`` the computation is exactly the
        fixed-encoder path.
        """
        observation_arr = jnp.asarray(observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )
        action_arr, action_valid = safe_discrete_action(
            action,
            self._config.n_actions,
        )
        reward_arr = jnp.asarray(reward, dtype=jnp.float32)
        discount_arr = jnp.asarray(discount, dtype=jnp.float32)
        next_observation_arr = jnp.asarray(next_observation, dtype=jnp.float32).reshape(
            (self._config.observation_dim,)
        )
        inputs_valid = (
            jnp.all(jnp.isfinite(observation_arr))
            & action_valid
            & jnp.all(jnp.isfinite(reward_arr))
            & jnp.all(jnp.isfinite(discount_arr))
            & jnp.all(discount_arr >= 0.0)
            & jnp.all(discount_arr <= 1.0)
            & jnp.all(jnp.isfinite(next_observation_arr))
        )
        safe_observation = jnp.where(inputs_valid, observation_arr, jnp.zeros_like(observation_arr))
        safe_action = jnp.where(inputs_valid, action_arr, jnp.zeros_like(action_arr))
        safe_reward = jnp.where(inputs_valid, reward_arr, jnp.zeros_like(reward_arr))
        safe_discount = jnp.where(inputs_valid, discount_arr, jnp.zeros_like(discount_arr))
        safe_next_observation = jnp.where(
            inputs_valid,
            next_observation_arr,
            jnp.zeros_like(next_observation_arr),
        )

        prediction = self.predict(state, safe_observation, safe_action)
        targets = self.targets(
            state,
            safe_observation,
            safe_reward,
            safe_discount,
            safe_next_observation,
        )
        learner_result = self._learner.update(
            state.learner_state,
            self.input_features_from_latent(prediction.latent, safe_action),
            targets,
        )
        target_next_latent = self.encode(state, safe_next_observation)
        surprise = jnp.mean((prediction.next_latent - target_next_latent) ** 2)
        reward_error = prediction.reward - safe_reward
        discount_error = prediction.discount - safe_discount
        prediction_error = surprise + reward_error**2 + discount_error**2

        collapse_decay = jnp.asarray(self._config.collapse_decay, dtype=jnp.float32)
        surprise_decay = jnp.asarray(self._config.surprise_decay, dtype=jnp.float32)
        first = state.step_count == 0
        next_mean = jnp.where(
            first,
            target_next_latent,
            collapse_decay * state.latent_mean_ema + (1.0 - collapse_decay) * target_next_latent,
        )
        centered = target_next_latent - next_mean
        next_var = jnp.where(
            first,
            centered**2,
            collapse_decay * state.latent_var_ema + (1.0 - collapse_decay) * centered**2,
        )
        latent_std = jnp.sqrt(jnp.maximum(next_var, jnp.asarray(1e-8, dtype=jnp.float32)))
        latent_std_mean = jnp.mean(latent_std)
        collapse_score = jnp.mean(
            (latent_std < jnp.asarray(self._config.min_latent_std, dtype=jnp.float32)).astype(
                jnp.float32
            )
        )
        next_surprise_ema = jnp.where(
            first,
            surprise,
            surprise_decay * state.surprise_ema + (1.0 - surprise_decay) * surprise,
        )
        next_prediction_error_ema = jnp.where(
            first,
            prediction_error,
            surprise_decay * state.prediction_error_ema + (1.0 - surprise_decay) * prediction_error,
        )
        next_collapse_score_ema = jnp.where(
            first,
            collapse_score,
            collapse_decay * state.collapse_score_ema + (1.0 - collapse_decay) * collapse_score,
        )

        if self._config.encoder_learning:
            (
                encoder_matrix,
                encoder_bias,
                encoder_update_applied,
                encoder_collapse_gated,
                encoder_gradient_norm,
            ) = self._encoder_step(
                state,
                safe_observation,
                safe_action,
                safe_next_observation,
                collapse_score,
            )
        else:
            encoder_matrix = state.encoder_matrix
            encoder_bias = state.encoder_bias
            encoder_update_applied = jnp.asarray(False, dtype=jnp.bool_)
            encoder_collapse_gated = jnp.asarray(False, dtype=jnp.bool_)
            encoder_gradient_norm = jnp.asarray(0.0, dtype=jnp.float32)

        candidate_state = LatentWorldModelState(
            encoder_matrix=encoder_matrix,
            encoder_bias=encoder_bias,
            learner_state=learner_result.state,
            latent_mean_ema=next_mean,
            latent_var_ema=next_var,
            surprise_ema=next_surprise_ema,
            prediction_error_ema=next_prediction_error_ema,
            collapse_score_ema=next_collapse_score_ema,
            step_count=_saturating_int32_increment(state.step_count),
        )
        diagnostics_finite = (
            floating_tree_is_finite(prediction)
            & jnp.all(jnp.isfinite(target_next_latent))
            & jnp.all(jnp.isfinite(targets))
            & jnp.all(jnp.isfinite(learner_result.errors))
            & jnp.all(jnp.isfinite(learner_result.per_head_metrics))
            & jnp.isfinite(surprise)
            & jnp.isfinite(reward_error)
            & jnp.isfinite(discount_error)
            & jnp.isfinite(prediction_error)
            & jnp.isfinite(latent_std_mean)
            & jnp.isfinite(collapse_score)
            & jnp.isfinite(encoder_gradient_norm)
        )
        update_applied = (
            inputs_valid
            & learner_result.update_applied
            & floating_tree_is_finite(state)
            & floating_tree_is_finite(candidate_state)
            & diagnostics_finite
        )
        committed_state = select_transaction(update_applied, candidate_state, state)
        reported_learner_result = _rollback_multi_head_result(
            learner_result, state.learner_state, update_applied
        )
        reported_prediction = LatentWorldModelPrediction(
            latent=neutralize_array(update_applied, prediction.latent),
            next_latent=neutralize_array(update_applied, prediction.next_latent),
            reward=neutralize_array(update_applied, prediction.reward),
            discount=neutralize_array(update_applied, prediction.discount),
            raw_predictions=neutralize_array(update_applied, prediction.raw_predictions),
        )
        return LatentWorldModelUpdateResult(
            state=committed_state,
            prediction=reported_prediction,
            target_next_latent=neutralize_array(update_applied, target_next_latent),
            targets=neutralize_array(update_applied, targets),
            errors=reported_learner_result.errors,
            surprise=neutralize_array(update_applied, surprise),
            reward_error=neutralize_array(update_applied, reward_error),
            discount_error=neutralize_array(update_applied, discount_error),
            prediction_error=neutralize_array(update_applied, prediction_error),
            latent_std_mean=neutralize_array(update_applied, latent_std_mean),
            collapse_score=neutralize_array(update_applied, collapse_score),
            per_head_metrics=reported_learner_result.per_head_metrics,
            learner_result=reported_learner_result,
            encoder_update_applied=encoder_update_applied & update_applied,
            encoder_collapse_gated=encoder_collapse_gated & update_applied,
            encoder_gradient_norm=neutralize_array(update_applied, encoder_gradient_norm),
            update_applied=update_applied,
        )

    def _validate_config(self, config: LatentWorldModelConfig) -> LatentWorldModelConfig:
        """Return the fail-closed canonical dataclass construction."""
        if type(config) is not LatentWorldModelConfig:
            raise ValueError("config must be a LatentWorldModelConfig")
        return config


def _require_scan_array_metadata(name: str, array: object) -> tuple[int, ...]:
    actual_type = type(array)
    if not (
        actual_type is np.ndarray
        or issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
    ):
        raise TypeError(f"{name} must be a trusted array")
    try:
        shape = tuple(int(dim) for dim in array.shape)  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must expose trusted shape metadata") from error
    return shape


def _require_scan_resource(num_steps: int, latent_dim: int, n_heads: int) -> None:
    output_scalars = num_steps * (3 * latent_dim + 3 * n_heads + 12)
    if output_scalars > _INT32_MAX or 4 * output_scalars > _INT32_MAX:
        raise ValueError("latent world model scan result bytes must fit signed int32")


def run_latent_world_model_learning_loop(
    model: LatentWorldModel,
    state: LatentWorldModelState,
    observations: Float[Array, "num_steps observation_dim"],
    actions: Array,
    rewards: Float[Array, " num_steps"],
    next_observations: Float[Array, "num_steps observation_dim"],
    discounts: Float[Array, " num_steps"] | None = None,
) -> LatentWorldModelLearningResult:
    """Run online latent world-model learning over transition arrays."""
    if type(model) is not LatentWorldModel:
        raise TypeError("model must be an exact LatentWorldModel")
    if type(state) is not LatentWorldModelState:
        raise TypeError("state must be an exact LatentWorldModelState")

    obs_shape = _require_scan_array_metadata("observations", observations)
    if len(obs_shape) != 2 or obs_shape[1] != model.config.observation_dim:
        raise ValueError(
            f"observations must have shape (num_steps, {model.config.observation_dim})"
        )
    num_steps = _require_int32("scan sequence length", obs_shape[0], minimum=1)

    next_obs_shape = _require_scan_array_metadata("next_observations", next_observations)
    if next_obs_shape != (num_steps, model.config.observation_dim):
        raise ValueError(
            f"next_observations must have shape ({num_steps}, {model.config.observation_dim})"
        )

    act_shape = _require_scan_array_metadata("actions", actions)
    if model.config.n_actions is not None:
        if act_shape != (num_steps,):
            raise ValueError(f"actions must have shape ({num_steps},)")
    else:
        if act_shape != (num_steps, model.config.action_dim):
            raise ValueError(f"actions must have shape ({num_steps}, {model.config.action_dim})")

    rew_shape = _require_scan_array_metadata("rewards", rewards)
    if rew_shape != (num_steps,):
        raise ValueError(f"rewards must have shape ({num_steps},)")

    if discounts is None:
        discounts_array = jnp.full(
            (num_steps,),
            jnp.asarray(model.config.gamma, dtype=jnp.float32),
            dtype=jnp.float32,
        )
    else:
        disc_shape = _require_scan_array_metadata("discounts", discounts)
        if disc_shape != (num_steps,):
            raise ValueError(f"discounts must have shape ({num_steps},)")
        discounts_array = jnp.asarray(discounts, dtype=jnp.float32)

    _require_scan_resource(num_steps, model.config.latent_dim, model.config.latent_dim + 2)

    obs_array = jnp.asarray(observations, dtype=jnp.float32)
    next_obs_array = jnp.asarray(next_observations, dtype=jnp.float32)
    rew_array = jnp.asarray(rewards, dtype=jnp.float32)
    act_array = (
        jnp.asarray(actions, dtype=jnp.int32)
        if model.config.n_actions is not None
        else jnp.asarray(actions, dtype=jnp.float32)
    )

    def _scan_fn(
        carry: LatentWorldModelState,
        inputs: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[LatentWorldModelState, tuple[Array, ...]]:
        obs, action, reward, discount, next_obs = inputs
        result = model.update(carry, obs, action, reward, discount, next_obs)
        return result.state, (
            result.prediction.latent,
            result.prediction.next_latent,
            result.prediction.reward,
            result.prediction.discount,
            result.target_next_latent,
            result.surprise,
            result.prediction_error,
            result.reward_error,
            result.discount_error,
            result.latent_std_mean,
            result.collapse_score,
            result.per_head_metrics,
            result.encoder_update_applied,
            result.encoder_collapse_gated,
            result.encoder_gradient_norm,
            result.update_applied,
        )

    (
        final_state,
        (
            latent_predictions,
            next_latent_predictions,
            reward_predictions,
            discount_predictions,
            target_next_latents,
            surprises,
            prediction_errors,
            reward_errors,
            discount_errors,
            latent_std_means,
            collapse_scores,
            per_head_metrics,
            encoder_updates_applied,
            encoder_collapse_gates,
            encoder_gradient_norms,
            updates_applied,
        ),
    ) = jax.lax.scan(
        _scan_fn,
        state,
        (obs_array, act_array, rew_array, discounts_array, next_obs_array),
    )
    return LatentWorldModelLearningResult(
        state=final_state,
        latent_predictions=latent_predictions,
        next_latent_predictions=next_latent_predictions,
        reward_predictions=reward_predictions,
        discount_predictions=discount_predictions,
        target_next_latents=target_next_latents,
        surprises=surprises,
        prediction_errors=prediction_errors,
        reward_errors=reward_errors,
        discount_errors=discount_errors,
        latent_std_means=latent_std_means,
        collapse_scores=collapse_scores,
        per_head_metrics=per_head_metrics,
        encoder_updates_applied=encoder_updates_applied,
        encoder_collapse_gates=encoder_collapse_gates,
        encoder_gradient_norms=encoder_gradient_norms,
        updates_applied=updates_applied,
    )


__all__ = [
    "EVIDENCE_LEVEL",
    "LatentWorldModel",
    "LatentWorldModelConfig",
    "LatentWorldModelLearningResult",
    "LatentWorldModelPrediction",
    "LatentWorldModelState",
    "LatentWorldModelUpdateResult",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "run_latent_world_model_learning_loop",
]
