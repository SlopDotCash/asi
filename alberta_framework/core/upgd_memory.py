# mypy: disable-error-code="call-arg,name-defined"
"""Single Step 2 learner combining UPGD with fixed-budget prototype memory.

Two complementary mechanisms form one learner: target-structure UPGD
(:class:`~alberta_framework.core.upgd.UPGDLearner`) provides differentiable
plastic features, and a fixed-budget multi-prototype memory
(:class:`~alberta_framework.core.prototype_memory.PrototypeMemoryLearner`)
retains one-hot class views.  Both components update on every step.  Their
predictions are blended by one learned scalar logit plus causal
confidence/reliability signals, so the deployed object is one learner
rather than a route-selecting portfolio.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.optimizers import ObGDBounding
from alberta_framework.core.prototype_memory import (
    PrototypeMemoryConfig,
    PrototypeMemoryLearner,
    PrototypeMemoryState,
)
from alberta_framework.core.update_safety import (
    floating_tree_is_finite,
    neutralize_array,
    select_transaction,
)
from alberta_framework.core.upgd import UPGDLearner, UPGDState

_INT32_MAX: int = 2**31 - 1
_UINT32_MAX: int = 2**32 - 1
_MAX_PERSISTENT_STATE_BYTES: int = 256 * 1024 * 1024
_FLOAT32_MIN_NORMAL: float = float(np.finfo(np.float32).tiny)
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


def _require_resource(
    name: str,
    *,
    float32_scalars: int = 0,
    int32_scalars: int = 0,
    uint32_scalars: int = 0,
    bool_scalars: int = 0,
) -> None:
    total_scalars = float32_scalars + int32_scalars + uint32_scalars + bool_scalars
    if total_scalars > _INT32_MAX:
        raise ValueError(f"{name} scalar count must fit signed int32")
    total_bytes = 4 * (float32_scalars + int32_scalars + uint32_scalars) + bool_scalars
    if total_bytes > _INT32_MAX:
        raise ValueError(f"{name} byte count must fit signed int32")


def _combined_state_resource_counts(
    feature_dim: int,
    n_heads: int,
    hidden_sizes: tuple[int, ...],
    slots_per_class: int,
) -> tuple[int, int, int]:
    """Return exact serialized float32, int32, and uint32 state leaf counts."""
    layer_sizes = (feature_dim, *hidden_sizes)
    trunk_weights = sum(
        layer_sizes[index] * layer_sizes[index + 1]
        for index in range(len(layer_sizes) - 1)
    )
    trunk_biases = sum(hidden_sizes)
    head_input_dim = hidden_sizes[-1] if hidden_sizes else feature_dim
    head_weights = n_heads * head_input_dim
    slots = n_heads * slots_per_class
    # UPGD serializes ordinary and fast head leaves separately, even though
    # initialization aliases their arrays. It also stores utilities, the
    # label adapter, targets, nine control scalars, and two telemetry scalars.
    upgd_float32 = (
        2 * trunk_weights
        + trunk_biases
        + 2 * head_weights
        + 3 * n_heads
        + n_heads * n_heads
        + 11
    )
    # Prototype means/counts plus the six wrapper EMA/logit scalars.
    memory_and_wrapper_float32 = slots * feature_dim + slots + 6
    # Prototype last-update words and three component/wrapper step counters.
    int32_scalars = slots + 3
    # One typed Threefry key has two underlying uint32 words.
    return upgd_float32 + memory_and_wrapper_float32, int32_scalars, 2


def _require_array(
    name: str, value: object, shape: tuple[int, ...], dtype: Any
) -> None:
    try:
        actual_shape = tuple(getattr(value, "shape"))
        actual_dtype = jnp.dtype(getattr(value, "dtype"))
        expected_dtype = jnp.dtype(dtype)
    except Exception as error:
        raise TypeError(f"{name} must expose valid array shape and dtype metadata") from error
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if actual_dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}")


def _require_typed_key(name: str, value: object) -> Array:
    """Require one scalar typed Threefry key without accepting legacy words."""
    try:
        implementation = str(jr.key_impl(value))  # type: ignore[arg-type]
        words = jr.key_data(value)  # type: ignore[arg-type]
        shape = tuple(getattr(value, "shape"))
    except Exception as error:
        raise TypeError(f"{name} must be a typed scalar threefry2x32 key") from error
    if (
        shape != ()
        or implementation != "threefry2x32"
        or words.shape != (2,)
        or words.dtype != jnp.uint32
    ):
        raise TypeError(f"{name} must be a typed scalar threefry2x32 key")
    return cast(Array, value)


def _require_real(name: str, value: object) -> float:
    return validated_float32_scalar(name, value)


def _require_unit_interval(name: str, value: object) -> float:
    return validated_float32_scalar(name, value, lower=0.0, upper=1.0)


def _require_half_open_unit_interval(name: str, value: object) -> float:
    return validated_float32_scalar(
        name,
        value,
        positive=True,
        upper=1.0,
    )


def _require_half_open_zero_one_interval(name: str, value: object) -> float:
    return validated_float32_scalar(
        name,
        value,
        lower=0.0,
        upper=1.0,
        upper_inclusive=False,
    )


def _require_nonnegative_real(name: str, value: object) -> float:
    return validated_float32_scalar(name, value, lower=0.0)


def _require_positive_real(name: str, value: object) -> float:
    return validated_float32_scalar(name, value, positive=True)


def _require_positive_normal_real(name: str, value: object) -> float:
    return validated_float32_scalar(name, value, lower=_FLOAT32_MIN_NORMAL)


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer")
    number = operator.index(cast(SupportsIndex, value))
    if minimum is not None and number < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive, got invalid value")
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative, got invalid value")
        raise ValueError(f"{name} must be >= {minimum}, got invalid value")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got invalid value")
    return number


def _skip_zero_scale(scale: Array, value: Array) -> Array:
    """Skip ``0 * inf`` so a disabled reliability EMA does not poison trackers."""
    return jnp.where(scale == 0.0, jnp.zeros_like(value), scale * value)


def _finite_or_zero(value: Array) -> Array:
    """Replace only poisoned disabled-EMA history, retaining finite history exactly."""
    return jnp.where(jnp.isfinite(value), value, jnp.zeros_like(value))


UPGDMemoryReadoutMode = Literal["linear_mse", "softmax_ce"]


@dataclass(frozen=True)
class UPGDMemoryConfig:
    """Configuration for :class:`UPGDMemoryLearner`.

    Args:
        feature_dim: Observation dimensionality.
        n_heads: Output dimensionality.  For classification this is the number
            of one-hot classes.
        hidden_sizes: UPGD hidden-layer widths.
        readout_mode: UPGD readout/loss mode.  ``"softmax_ce"`` is the intended
            mode when prototype memory is active.
        upgd_step_size: Base UPGD step-size.
        upgd_head_step_size_multiplier: Fixed multiplier for output-head
            weight and bias updates.
        upgd_head_bias_step_size_multiplier: Extra multiplier for output-head
            bias updates after ``upgd_head_step_size_multiplier``.
        upgd_head_loss_pressure_gate_ratio: Fast/slow loss ratio at which the
            output head receives an additional plasticity multiplier.
        upgd_head_loss_pressure_multiplier: Maximum additional output-head
            plasticity under loss pressure.
        upgd_head_loss_pressure_warmup_steps: Initial updates before
            loss-pressure head plasticity is enabled.
        upgd_head_repetition_multiplier: Maximum additional output-head
            plasticity under repeated-target pressure.
        upgd_head_repetition_decay: EMA decay for repeated-target detection.
        upgd_head_repetition_delta_threshold: Mean absolute target-vector
            change treated as a repeated target.
        upgd_head_repetition_pressure_threshold: Repetition EMA level below
            which repeated-target pressure is ignored.
        upgd_head_repetition_warmup_steps: Initial updates before
            repeated-target head plasticity is enabled.
        slots_per_class: Fixed prototype slots per class.
        memory_update_rate: EMA rate for matched prototypes.
        initial_novelty_threshold: Initial mean-squared distance threshold for
            allocating a fresh prototype.
        memory_bandwidth: Distance-to-logit bandwidth for prototype memory.
        initial_memory_logit: Learned base logit for memory-vs-UPGD blending.
        memory_logit_step_size: Online gradient step-size for the blend logit.
        confidence_logit_scale: Fixed coefficient for memory confidence minus
            UPGD confidence.
        reliability_logit_scale: Fixed coefficient for UPGD loss EMA minus
            memory loss EMA.
        reliability_decay: EMA decay for component losses and allocation rate.
        target_trace_blend_scale: Update-time blend toward the previous
            target vector under repeated-target pressure.  This is a causal
            temporal prior for prequential streams with persistent targets.
            Only ``update`` applies it; ordinary ``predict`` calls stay
            observation-based so held-out batch evaluation is not biased toward
            the last observed target.
        target_trace_pressure_threshold: Repetition EMA level below which the
            target-trace prior is ignored.
        novelty_adaptation_rate: Online log-threshold adaptation step-size.
        target_allocation_rate: Target prototype allocation frequency.  When
            allocation EMA is higher than this, the threshold rises; when lower,
            it falls.
        min_novelty_threshold: Lower threshold clamp.
        max_novelty_threshold: Upper threshold clamp.
    """

    feature_dim: int
    n_heads: int
    hidden_sizes: tuple[int, ...] = (64,)
    readout_mode: UPGDMemoryReadoutMode = "softmax_ce"
    upgd_step_size: float = 0.03
    upgd_head_step_size_multiplier: float = 1.0
    upgd_head_bias_step_size_multiplier: float = 1.0
    upgd_head_loss_pressure_gate_ratio: float = 0.0
    upgd_head_loss_pressure_multiplier: float = 0.0
    upgd_head_loss_pressure_warmup_steps: int = 0
    upgd_head_repetition_multiplier: float = 0.0
    upgd_head_repetition_decay: float = 0.9
    upgd_head_repetition_delta_threshold: float = 0.05
    upgd_head_repetition_pressure_threshold: float = 0.0
    upgd_head_repetition_warmup_steps: int = 0
    slots_per_class: int = 20
    memory_update_rate: float = 0.3
    initial_novelty_threshold: float = 0.08
    memory_bandwidth: float = 0.01
    initial_memory_logit: float = 0.0
    memory_logit_step_size: float = 0.25
    confidence_logit_scale: float = 2.0
    reliability_logit_scale: float = 8.0
    reliability_decay: float = 0.98
    target_trace_blend_scale: float = 0.8
    target_trace_pressure_threshold: float = 0.5
    novelty_adaptation_rate: float = 0.02
    target_allocation_rate: float = 0.18
    min_novelty_threshold: float = 1e-4
    max_novelty_threshold: float = 1.0

    def __post_init__(self) -> None:
        """Validate and canonicalize configuration."""
        _validate_config(self)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload

    def to_config(self) -> dict[str, object]:
        """Serialize to a plain config dictionary."""
        payload = self.to_dict()
        payload["type"] = "UPGDMemoryConfig"
        return payload

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> UPGDMemoryConfig:
        """Reconstruct from :meth:`to_config` output."""
        return cls._from_serialized(config)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UPGDMemoryConfig:
        """Reconstruct from :meth:`to_dict` output."""
        return cls._from_serialized(data)

    @classmethod
    def _from_serialized(cls, data: Mapping[str, Any]) -> UPGDMemoryConfig:
        if not issubclass(type(data), Mapping):
            raise ValueError("UPGDMemoryConfig payload must be a mapping")
        try:
            payload = dict(data)
        except Exception as error:
            raise ValueError("UPGDMemoryConfig mapping could not be read") from error
        marker = payload.pop("type", None)
        if marker is not None and type(marker) is not str:
            raise ValueError("UPGDMemoryConfig type marker must be an actual string")
        if "hidden_sizes" in payload:
            hidden = payload["hidden_sizes"]
            if type(hidden) not in {list, tuple}:
                raise ValueError("hidden_sizes must be a list or tuple")
            payload["hidden_sizes"] = tuple(hidden)
        return cls(**payload)


@chex.dataclass(frozen=True)
class UPGDMemoryState:
    """State for :class:`UPGDMemoryLearner`."""

    upgd_state: UPGDState
    memory_state: PrototypeMemoryState
    memory_logit: Array
    novelty_log_threshold: Array
    upgd_loss_ema: Array
    memory_loss_ema: Array
    blended_loss_ema: Array
    allocation_ema: Array
    step_count: Array


@chex.dataclass(frozen=True)
class UPGDMemoryUpdateResult:
    """Result of one UPGD-memory update."""

    state: UPGDMemoryState
    predictions: Float[Array, " n_heads"]
    errors: Float[Array, " n_heads"]
    metrics: Float[Array, " 10"]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class UPGDMemoryLearningResult:
    """Result from :func:`run_upgd_memory_arrays`."""

    state: UPGDMemoryState
    predictions: Float[Array, "steps n_heads"]
    metrics: Float[Array, "steps 10"]
    updates_applied: Bool[Array, " steps"]


def _validate_config(config: UPGDMemoryConfig) -> None:
    feature_dim = _require_int(
        "feature_dim", config.feature_dim, minimum=1, maximum=_INT32_MAX
    )
    n_heads = _require_int("n_heads", config.n_heads, minimum=2, maximum=_INT32_MAX)
    if type(config.hidden_sizes) is not tuple:
        raise TypeError(
            f"hidden_sizes must be an actual tuple, got {type(config.hidden_sizes).__name__}"
        )
    canonical_hidden = tuple(
        _require_int("hidden_sizes element", size, minimum=1, maximum=_INT32_MAX)
        for size in config.hidden_sizes
    )
    readout_mode = config.readout_mode
    if type(readout_mode) is not str:
        raise TypeError(
            "readout_mode must be an actual string, got invalid value"
        )
    if readout_mode not in {"linear_mse", "softmax_ce"}:
        raise ValueError("readout_mode must be 'linear_mse' or 'softmax_ce'")
    canonical_readout_mode = str(readout_mode)
    upgd_step_size = _require_positive_real("upgd_step_size", config.upgd_step_size)
    upgd_head_step_size_multiplier = _require_positive_real(
        "upgd_head_step_size_multiplier", config.upgd_head_step_size_multiplier
    )
    upgd_head_bias_step_size_multiplier = _require_nonnegative_real(
        "upgd_head_bias_step_size_multiplier",
        config.upgd_head_bias_step_size_multiplier,
    )
    upgd_head_loss_pressure_gate_ratio = _require_nonnegative_real(
        "upgd_head_loss_pressure_gate_ratio",
        config.upgd_head_loss_pressure_gate_ratio,
    )
    upgd_head_loss_pressure_multiplier = _require_nonnegative_real(
        "upgd_head_loss_pressure_multiplier",
        config.upgd_head_loss_pressure_multiplier,
    )
    upgd_head_loss_pressure_warmup_steps = _require_int(
        "upgd_head_loss_pressure_warmup_steps",
        config.upgd_head_loss_pressure_warmup_steps,
        minimum=0,
        maximum=_INT32_MAX,
    )
    upgd_head_repetition_multiplier = _require_nonnegative_real(
        "upgd_head_repetition_multiplier",
        config.upgd_head_repetition_multiplier,
    )
    upgd_head_repetition_decay = _require_half_open_zero_one_interval(
        "upgd_head_repetition_decay",
        config.upgd_head_repetition_decay,
    )
    upgd_head_repetition_delta_threshold = _require_nonnegative_real(
        "upgd_head_repetition_delta_threshold",
        config.upgd_head_repetition_delta_threshold,
    )
    upgd_head_repetition_pressure_threshold = _require_half_open_zero_one_interval(
        "upgd_head_repetition_pressure_threshold",
        config.upgd_head_repetition_pressure_threshold,
    )
    upgd_head_repetition_warmup_steps = _require_int(
        "upgd_head_repetition_warmup_steps",
        config.upgd_head_repetition_warmup_steps,
        minimum=0,
        maximum=_INT32_MAX,
    )
    slots_per_class = _require_int(
        "slots_per_class", config.slots_per_class, minimum=1, maximum=_INT32_MAX
    )
    memory_update_rate = _require_half_open_unit_interval(
        "memory_update_rate", config.memory_update_rate
    )
    initial_novelty_threshold = _require_positive_normal_real(
        "initial_novelty_threshold", config.initial_novelty_threshold
    )
    memory_bandwidth = _require_positive_normal_real(
        "memory_bandwidth", config.memory_bandwidth
    )
    initial_memory_logit = _require_real(
        "initial_memory_logit", config.initial_memory_logit
    )
    memory_logit_step_size = _require_nonnegative_real(
        "memory_logit_step_size", config.memory_logit_step_size
    )
    confidence_logit_scale = _require_real(
        "confidence_logit_scale", config.confidence_logit_scale
    )
    reliability_logit_scale = _require_real(
        "reliability_logit_scale", config.reliability_logit_scale
    )
    reliability_decay = _require_half_open_zero_one_interval(
        "reliability_decay", config.reliability_decay
    )
    target_trace_blend_scale = _require_unit_interval(
        "target_trace_blend_scale", config.target_trace_blend_scale
    )
    target_trace_pressure_threshold = _require_half_open_zero_one_interval(
        "target_trace_pressure_threshold", config.target_trace_pressure_threshold
    )
    novelty_adaptation_rate = _require_nonnegative_real(
        "novelty_adaptation_rate", config.novelty_adaptation_rate
    )
    target_allocation_rate = _require_unit_interval(
        "target_allocation_rate", config.target_allocation_rate
    )
    min_novelty_threshold = _require_positive_normal_real(
        "min_novelty_threshold", config.min_novelty_threshold
    )
    max_novelty_threshold = _require_positive_normal_real(
        "max_novelty_threshold", config.max_novelty_threshold
    )
    if min_novelty_threshold > max_novelty_threshold:
        raise ValueError("min_novelty_threshold must be <= max_novelty_threshold")

    object.__setattr__(config, "feature_dim", feature_dim)
    object.__setattr__(config, "n_heads", n_heads)
    object.__setattr__(config, "hidden_sizes", canonical_hidden)
    object.__setattr__(config, "readout_mode", canonical_readout_mode)
    object.__setattr__(config, "upgd_step_size", upgd_step_size)
    object.__setattr__(
        config,
        "upgd_head_step_size_multiplier",
        upgd_head_step_size_multiplier,
    )
    object.__setattr__(
        config,
        "upgd_head_bias_step_size_multiplier",
        upgd_head_bias_step_size_multiplier,
    )
    object.__setattr__(
        config,
        "upgd_head_loss_pressure_gate_ratio",
        upgd_head_loss_pressure_gate_ratio,
    )
    object.__setattr__(
        config,
        "upgd_head_loss_pressure_multiplier",
        upgd_head_loss_pressure_multiplier,
    )
    object.__setattr__(
        config,
        "upgd_head_loss_pressure_warmup_steps",
        upgd_head_loss_pressure_warmup_steps,
    )
    object.__setattr__(
        config,
        "upgd_head_repetition_multiplier",
        upgd_head_repetition_multiplier,
    )
    object.__setattr__(
        config,
        "upgd_head_repetition_decay",
        upgd_head_repetition_decay,
    )
    object.__setattr__(
        config,
        "upgd_head_repetition_delta_threshold",
        upgd_head_repetition_delta_threshold,
    )
    object.__setattr__(
        config,
        "upgd_head_repetition_pressure_threshold",
        upgd_head_repetition_pressure_threshold,
    )
    object.__setattr__(
        config,
        "upgd_head_repetition_warmup_steps",
        upgd_head_repetition_warmup_steps,
    )
    object.__setattr__(config, "slots_per_class", slots_per_class)
    object.__setattr__(config, "memory_update_rate", memory_update_rate)
    object.__setattr__(
        config, "initial_novelty_threshold", initial_novelty_threshold
    )
    object.__setattr__(config, "memory_bandwidth", memory_bandwidth)
    object.__setattr__(config, "initial_memory_logit", initial_memory_logit)
    object.__setattr__(
        config, "memory_logit_step_size", memory_logit_step_size
    )
    object.__setattr__(
        config, "confidence_logit_scale", confidence_logit_scale
    )
    object.__setattr__(
        config, "reliability_logit_scale", reliability_logit_scale
    )
    object.__setattr__(config, "reliability_decay", reliability_decay)
    object.__setattr__(
        config, "target_trace_blend_scale", target_trace_blend_scale
    )
    object.__setattr__(
        config,
        "target_trace_pressure_threshold",
        target_trace_pressure_threshold,
    )
    object.__setattr__(
        config, "novelty_adaptation_rate", novelty_adaptation_rate
    )
    object.__setattr__(
        config, "target_allocation_rate", target_allocation_rate
    )
    object.__setattr__(
        config, "min_novelty_threshold", min_novelty_threshold
    )
    object.__setattr__(
        config, "max_novelty_threshold", max_novelty_threshold
    )
    float32_scalars, int32_scalars, uint32_scalars = _combined_state_resource_counts(
        feature_dim,
        n_heads,
        canonical_hidden,
        slots_per_class,
    )
    _require_resource(
        "UPGDMemoryConfig combined state",
        float32_scalars=float32_scalars,
        int32_scalars=int32_scalars,
        uint32_scalars=uint32_scalars,
    )
    persistent_bytes = 4 * (float32_scalars + int32_scalars + uint32_scalars)
    if persistent_bytes > _UINT32_MAX:
        raise ValueError("UPGDMemoryConfig combined state bytes must fit unsigned int32")
    if persistent_bytes > _MAX_PERSISTENT_STATE_BYTES:
        raise ValueError("UPGDMemoryConfig persistent state exceeds 256 MiB")


def _active_mse(prediction: Array, target: Array) -> Array:
    active = jnp.isfinite(target)
    safe_target = jnp.where(active, target, 0.0)
    squared = jnp.where(active, (prediction - safe_target) ** 2, 0.0)
    return jnp.sum(squared) / jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)


def _normalize_simplex(prediction: Array) -> Array:
    clipped = jnp.maximum(prediction, 0.0)
    return clipped / jnp.maximum(jnp.sum(clipped), 1e-12)


class UPGDMemoryLearner:
    """UPGD plus adaptive fixed-budget prototype memory as one learner."""

    def __init__(self, config: UPGDMemoryConfig):
        _validate_config(config)
        self._config = config
        # The UPGD sub-configuration is frozen here; UPGDMemoryConfig exposes
        # only step-size and head-plasticity knobs.  The fixed values match
        # ``UPGDLearner.step2_default``: ObGD bounding at kappa 0.5, init
        # sparsity 0.5, layer norm, Rademacher perturbation of sigma 1e-4
        # every 16 steps, and target-structure loss normalization for one-hot
        # targets.  Relative to raw ``UPGDLearner`` defaults (sigma 1e-3 every
        # step, sparsity 0.9, no update bounding) this perturbs far more
        # gently and keeps more weights alive at init.
        self._upgd = UPGDLearner(
            n_heads=config.n_heads,
            hidden_sizes=config.hidden_sizes,
            step_size=config.upgd_step_size,
            bounder=ObGDBounding(kappa=0.5),
            sparsity=0.5,
            use_layer_norm=True,
            perturbation_sigma=1e-4,
            perturbation_noise="rademacher",
            utility_decay=0.995,
            perturbation_beta=2.0,
            perturbation_interval=16,
            loss_normalization="target_structure",
            readout_mode=config.readout_mode,
            track_unit_utilities=False,
            track_gradient_history=False,
            head_step_size_multiplier=config.upgd_head_step_size_multiplier,
            head_bias_step_size_multiplier=(config.upgd_head_bias_step_size_multiplier),
            head_loss_pressure_gate_ratio=(config.upgd_head_loss_pressure_gate_ratio),
            head_loss_pressure_multiplier=(config.upgd_head_loss_pressure_multiplier),
            head_loss_pressure_warmup_steps=(config.upgd_head_loss_pressure_warmup_steps),
            head_repetition_multiplier=config.upgd_head_repetition_multiplier,
            head_repetition_decay=config.upgd_head_repetition_decay,
            head_repetition_delta_threshold=(config.upgd_head_repetition_delta_threshold),
            head_repetition_pressure_threshold=(config.upgd_head_repetition_pressure_threshold),
            head_repetition_warmup_steps=(config.upgd_head_repetition_warmup_steps),
        )
        self._memory = PrototypeMemoryLearner(
            PrototypeMemoryConfig(
                feature_dim=config.feature_dim,
                n_classes=config.n_heads,
                slots_per_class=config.slots_per_class,
                update_rate=config.memory_update_rate,
                novelty_threshold=config.initial_novelty_threshold,
                bandwidth=config.memory_bandwidth,
            )
        )

    @property
    def config(self) -> UPGDMemoryConfig:
        """Learner configuration."""
        return self._config

    @property
    def upgd(self) -> UPGDLearner:
        """Underlying UPGD component."""
        return self._upgd

    @property
    def memory(self) -> PrototypeMemoryLearner:
        """Underlying fixed-budget prototype memory component."""
        return self._memory

    def to_config(self) -> dict[str, object]:
        """Serialize the learner configuration."""
        return {
            "type": "UPGDMemoryLearner",
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> UPGDMemoryLearner:
        """Reconstruct from :meth:`to_config` output."""
        if not issubclass(type(config), Mapping):
            raise ValueError("UPGDMemoryLearner payload must be a mapping")
        try:
            payload = dict(config)
        except Exception as error:
            raise ValueError("UPGDMemoryLearner mapping could not be read") from error
        marker = payload.get("type")
        if marker is not None and type(marker) is not str:
            raise ValueError("UPGDMemoryLearner type marker must be an actual string")
        inner = payload.get("config")
        if not issubclass(type(inner), Mapping):
            raise ValueError("UPGDMemoryLearner config must be a mapping")
        return cls(UPGDMemoryConfig.from_config(cast(Mapping[str, Any], inner)))

    def _validate_state_static_contract(self, state: UPGDMemoryState) -> None:
        """Reject malformed combined state before any traced computation."""
        if type(state) is not UPGDMemoryState:
            raise TypeError("state must be a UPGDMemoryState")
        if type(state.upgd_state) is not UPGDState:
            raise TypeError("state.upgd_state must be a UPGDState")
        self._memory._validate_state_static_contract(state.memory_state)  # noqa: SLF001
        cfg = self._config
        upgd = state.upgd_state
        layer_sizes = (cfg.feature_dim, *cfg.hidden_sizes)
        if (
            type(upgd.trunk_params.weights) is not tuple
            or type(upgd.trunk_params.biases) is not tuple
        ):
            raise TypeError("UPGD trunk parameters must use tuple containers")
        if len(upgd.trunk_params.weights) != len(cfg.hidden_sizes):
            raise ValueError("UPGD trunk layer count is invalid")
        if type(upgd.utilities) is not tuple or len(upgd.utilities) != len(cfg.hidden_sizes):
            raise ValueError("UPGD utility layer count is invalid")
        for index in range(len(cfg.hidden_sizes)):
            shape = (layer_sizes[index + 1], layer_sizes[index])
            _require_array(
                f"state.upgd_state.trunk_params.weights[{index}]",
                upgd.trunk_params.weights[index],
                shape,
                jnp.float32,
            )
            _require_array(
                f"state.upgd_state.trunk_params.biases[{index}]",
                upgd.trunk_params.biases[index],
                (layer_sizes[index + 1],),
                jnp.float32,
            )
            _require_array(
                f"state.upgd_state.utilities[{index}]",
                upgd.utilities[index],
                shape,
                jnp.float32,
            )
        head_input = cfg.hidden_sizes[-1] if cfg.hidden_sizes else cfg.feature_dim
        for label, params in (
            ("head_params", upgd.head_params),
            ("readout_fast_head_params", upgd.readout_fast_head_params),
        ):
            if type(params.weights) is not tuple or type(params.biases) is not tuple:
                raise TypeError(f"UPGD {label} must use tuple containers")
            if len(params.weights) != cfg.n_heads or len(params.biases) != cfg.n_heads:
                raise ValueError(f"UPGD {label} count is invalid")
            for index in range(cfg.n_heads):
                _require_array(
                    f"state.upgd_state.{label}.weights[{index}]",
                    params.weights[index],
                    (1, head_input),
                    jnp.float32,
                )
                _require_array(
                    f"state.upgd_state.{label}.biases[{index}]",
                    params.biases[index],
                    (1,),
                    jnp.float32,
                )
        _require_array(
            "state.upgd_state.readout_label_adapter",
            upgd.readout_label_adapter,
            (cfg.n_heads, cfg.n_heads),
            jnp.float32,
        )
        for name in (
            "unit_utilities",
            "unit_long_utilities",
            "unit_gradient_emas",
            "unit_ages",
            "previous_trunk_weight_grads",
            "previous_trunk_bias_grads",
            "previous_head_weight_grads",
            "previous_head_bias_grads",
        ):
            if getattr(upgd, name) != ():
                raise ValueError(f"state.upgd_state.{name} must be empty")
        _require_array(
            "state.upgd_state.unit_replacement_counts",
            upgd.unit_replacement_counts,
            (0,),
            jnp.float32,
        )
        _require_array(
            "state.upgd_state.unit_replacement_accumulators",
            upgd.unit_replacement_accumulators,
            (0,),
            jnp.float32,
        )
        _require_array(
            "state.upgd_state.previous_targets",
            upgd.previous_targets,
            (cfg.n_heads,),
            jnp.float32,
        )
        for name in (
            "loss_fast_ema",
            "loss_slow_ema",
            "target_repeat_ema",
            "target_simplex_ema",
            "meta_trunk_log_scale",
            "meta_head_weight_log_scale",
            "meta_head_bias_log_scale",
            "meta_repetition_log_scale",
            "adaptive_kappa_log_scale",
            "birth_timestamp",
            "uptime_s",
        ):
            _require_array(f"state.upgd_state.{name}", getattr(upgd, name), (), jnp.float32)
        _require_array("state.upgd_state.step_count", upgd.step_count, (), jnp.int32)
        _require_typed_key("state.upgd_state.key", upgd.key)
        for name in (
            "memory_logit",
            "novelty_log_threshold",
            "upgd_loss_ema",
            "memory_loss_ema",
            "blended_loss_ema",
            "allocation_ema",
        ):
            _require_array(f"state.{name}", getattr(state, name), (), jnp.float32)
        _require_array("state.step_count", state.step_count, (), jnp.int32)

    def _state_is_valid(self, state: UPGDMemoryState) -> Bool[Array, ""]:
        return (
            floating_tree_is_finite(state)
            & (state.step_count >= 0)
            & (state.upgd_state.step_count >= 0)
            & self._memory._state_is_valid(state.memory_state)  # noqa: SLF001
        )

    def init(self, key: Array | None = None) -> UPGDMemoryState:
        """Initialize both components and adaptive blend state."""
        if key is None:
            key = jr.key(0)
        key = _require_typed_key("key", key)
        cfg = self._config
        raw_upgd_state = self._upgd.init(cfg.feature_dim, key)
        upgd_state = raw_upgd_state.replace(  # type: ignore[attr-defined]
            birth_timestamp=jnp.asarray(raw_upgd_state.birth_timestamp, dtype=jnp.float32),
            uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
        )
        return UPGDMemoryState(
            upgd_state=upgd_state,
            memory_state=self._memory.init(),
            memory_logit=jnp.asarray(cfg.initial_memory_logit, dtype=jnp.float32),
            novelty_log_threshold=jnp.log(
                jnp.asarray(cfg.initial_novelty_threshold, dtype=jnp.float32)
            ),
            upgd_loss_ema=jnp.array(0.0, dtype=jnp.float32),
            memory_loss_ema=jnp.array(0.0, dtype=jnp.float32),
            blended_loss_ema=jnp.array(0.0, dtype=jnp.float32),
            allocation_ema=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def _blend_gate(
        self,
        state: UPGDMemoryState,
        upgd_prediction: Array,
        memory_prediction: Array,
    ) -> Array:
        active_memory = (jnp.sum(state.memory_state.counts > 0.0) > 0).astype(jnp.float32)
        confidence_delta = jnp.max(memory_prediction) - jnp.max(upgd_prediction)
        if self._config.reliability_decay == 0.0:
            # These prior losses still determine the prediction gate before
            # the zero-decay EMAs are overwritten.  Preserve every finite
            # value; only a poisoned historical value is recoverable here.
            upgd_loss_ema = _finite_or_zero(state.upgd_loss_ema)
            memory_loss_ema = _finite_or_zero(state.memory_loss_ema)
        else:
            upgd_loss_ema = state.upgd_loss_ema
            memory_loss_ema = state.memory_loss_ema
        reliability_delta = upgd_loss_ema - memory_loss_ema
        logit = (
            state.memory_logit
            + self._config.confidence_logit_scale * confidence_delta
            + self._config.reliability_logit_scale * reliability_delta
        )
        return active_memory * jax.nn.sigmoid(logit)

    def _blend_predictions(
        self,
        state: UPGDMemoryState,
        upgd_prediction: Array,
        memory_prediction: Array,
        *,
        include_target_trace: bool,
    ) -> tuple[Array, Array]:
        gate = self._blend_gate(state, upgd_prediction, memory_prediction)
        prediction = (1.0 - gate) * upgd_prediction + gate * memory_prediction
        if self._config.readout_mode == "softmax_ce":
            prediction = _normalize_simplex(prediction)
        trace_scale = jnp.where(
            include_target_trace,
            jnp.asarray(self._config.target_trace_blend_scale, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
        )
        threshold = jnp.asarray(
            self._config.target_trace_pressure_threshold,
            dtype=jnp.float32,
        )
        trace_pressure = jnp.clip(
            (state.upgd_state.target_repeat_ema - threshold) / jnp.maximum(1.0 - threshold, 1e-6),
            0.0,
            1.0,
        )
        trace_gate = trace_scale * trace_pressure
        trace_prediction = _normalize_simplex(state.upgd_state.previous_targets)
        prediction = (1.0 - trace_gate) * prediction + trace_gate * trace_prediction
        if self._config.readout_mode == "softmax_ce":
            prediction = _normalize_simplex(prediction)
        return prediction, gate

    def predict(
        self,
        state: UPGDMemoryState,
        observation: Float[Array, " feature_dim"],
    ) -> Float[Array, " n_heads"]:
        """Predict with the current learned UPGD-memory blend."""
        self._validate_state_static_contract(state)
        _require_array("observation", observation, (self._config.feature_dim,), jnp.float32)
        return cast(Array, self._predict_jitted(state, observation))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _predict_jitted(
        self,
        state: UPGDMemoryState,
        observation: Float[Array, " feature_dim"],
    ) -> Float[Array, " n_heads"]:
        upgd_prediction = self._upgd.predict(state.upgd_state, observation)
        memory_prediction = self._memory.predict(state.memory_state, observation)
        prediction, _gate = self._blend_predictions(
            state,
            upgd_prediction,
            memory_prediction,
            include_target_trace=False,
        )
        return prediction

    def update(
        self,
        state: UPGDMemoryState,
        observation: Float[Array, " feature_dim"],
        target: Float[Array, " n_heads"],
    ) -> UPGDMemoryUpdateResult:
        """Update UPGD, memory, blend reliability, and novelty threshold."""
        self._validate_state_static_contract(state)
        _require_array("observation", observation, (self._config.feature_dim,), jnp.float32)
        _require_array("target", target, (self._config.n_heads,), jnp.float32)
        return cast(UPGDMemoryUpdateResult, self._update_jitted(state, observation, target))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_jitted(
        self,
        state: UPGDMemoryState,
        observation: Float[Array, " feature_dim"],
        target: Float[Array, " n_heads"],
    ) -> UPGDMemoryUpdateResult:
        observation_arr = jnp.asarray(observation)
        target_arr = jnp.asarray(target)
        observation_valid = jnp.all(jnp.isfinite(observation_arr))
        target_valid = jnp.all(~jnp.isinf(target_arr))
        safe_observation = jnp.where(
            observation_valid, observation_arr, jnp.zeros_like(observation_arr)
        )
        safe_update_target = jnp.where(jnp.isinf(target_arr), jnp.nan, target_arr)
        upgd_prediction = self._upgd.predict(state.upgd_state, safe_observation)
        memory_prediction = self._memory.predict(state.memory_state, safe_observation)
        prediction, gate = self._blend_predictions(
            state,
            upgd_prediction,
            memory_prediction,
            include_target_trace=True,
        )
        safe_target = jnp.where(jnp.isfinite(safe_update_target), safe_update_target, 0.0)
        errors = prediction - safe_target
        blended_loss = _active_mse(prediction, safe_update_target)
        upgd_loss = _active_mse(upgd_prediction, safe_update_target)
        memory_loss = _active_mse(memory_prediction, safe_update_target)

        def blend_loss(memory_logit: Array) -> Array:
            probe_prediction, _probe_gate = self._blend_predictions(
                state.replace(memory_logit=memory_logit),  # type: ignore[attr-defined]
                upgd_prediction,
                memory_prediction,
                include_target_trace=True,
            )
            return _active_mse(probe_prediction, safe_update_target)

        dloss_dlogit = jax.grad(blend_loss)(state.memory_logit)
        next_memory_logit = state.memory_logit - (
            jnp.asarray(self._config.memory_logit_step_size, dtype=jnp.float32) * dloss_dlogit
        )
        # Bound the learned base logit: sigmoid(+/-8) is already ~0.9997, so
        # the clip costs nothing in gate range but stops unbounded drift during
        # long one-sided regimes, keeping the additive confidence/reliability
        # terms able to reverse the gate after a regime change.
        next_memory_logit = jnp.clip(next_memory_logit, -8.0, 8.0)

        threshold = jnp.exp(state.novelty_log_threshold)
        upgd_result = self._upgd.update(
            state.upgd_state, safe_observation, safe_update_target
        )
        safe_upgd_state = upgd_result.state.replace(
            step_count=(
                jnp.minimum(
                    state.upgd_state.step_count,
                    jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32),
                )
                + 1
            )
        )
        memory_result = self._memory.update_with_novelty_threshold(
            state.memory_state,
            safe_observation,
            safe_update_target,
            threshold,
        )
        allocated = memory_result.metrics[5]
        decay = jnp.asarray(self._config.reliability_decay, dtype=jnp.float32)
        one_minus_decay = 1.0 - decay
        next_allocation_ema = (
            _skip_zero_scale(decay, state.allocation_ema) + one_minus_decay * allocated
        )
        allocation_error = next_allocation_ema - jnp.asarray(
            self._config.target_allocation_rate,
            dtype=jnp.float32,
        )
        next_log_threshold = state.novelty_log_threshold + (
            jnp.asarray(self._config.novelty_adaptation_rate, dtype=jnp.float32) * allocation_error
        )
        next_log_threshold = jnp.clip(
            next_log_threshold,
            jnp.log(jnp.asarray(self._config.min_novelty_threshold, dtype=jnp.float32)),
            jnp.log(jnp.asarray(self._config.max_novelty_threshold, dtype=jnp.float32)),
        )

        candidate_state = UPGDMemoryState(
            upgd_state=safe_upgd_state,
            memory_state=memory_result.state,
            memory_logit=next_memory_logit,
            novelty_log_threshold=next_log_threshold,
            upgd_loss_ema=_skip_zero_scale(decay, state.upgd_loss_ema)
            + one_minus_decay * upgd_loss,
            memory_loss_ema=_skip_zero_scale(decay, state.memory_loss_ema)
            + one_minus_decay * memory_loss,
            blended_loss_ema=(
                _skip_zero_scale(decay, state.blended_loss_ema)
                + one_minus_decay * blended_loss
            ),
            allocation_ema=next_allocation_ema,
            step_count=(
                jnp.minimum(
                    state.step_count,
                    jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32),
                )
                + 1
            ),
        )
        metrics = jnp.asarray(
            [
                blended_loss,
                upgd_loss,
                memory_loss,
                gate,
                next_memory_logit,
                threshold,
                next_allocation_ema,
                jnp.sum(memory_result.state.counts > 0.0).astype(jnp.float32),
                jnp.max(upgd_prediction),
                jnp.max(memory_prediction),
            ],
            dtype=jnp.float32,
        )
        checked_state = (
            state.replace(  # type: ignore[attr-defined]
                allocation_ema=_finite_or_zero(state.allocation_ema),
                upgd_loss_ema=_finite_or_zero(state.upgd_loss_ema),
                memory_loss_ema=_finite_or_zero(state.memory_loss_ema),
                blended_loss_ema=_finite_or_zero(state.blended_loss_ema),
            )
            if self._config.reliability_decay == 0.0
            else state
        )
        update_applied = (
            observation_valid
            & target_valid
            & memory_result.update_applied
            & self._state_is_valid(checked_state)
            & floating_tree_is_finite(candidate_state)
            & jnp.all(jnp.isfinite(prediction))
            & jnp.all(jnp.isfinite(errors))
            & jnp.all(jnp.isfinite(metrics))
        )
        return UPGDMemoryUpdateResult(
            state=select_transaction(update_applied, candidate_state, state),
            predictions=neutralize_array(update_applied, prediction),
            errors=neutralize_array(update_applied, errors),
            metrics=neutralize_array(update_applied, metrics),
            update_applied=update_applied,
        )


def run_upgd_memory_arrays(
    learner: UPGDMemoryLearner,
    state: UPGDMemoryState,
    observations: Float[Array, "steps feature_dim"],
    targets: Float[Array, "steps n_heads"],
) -> UPGDMemoryLearningResult:
    """Run a UPGD-memory learner over arrays with ``jax.lax.scan``.

    Metric columns are ``blend_mse, upgd_mse, memory_mse, gate, memory_logit,
    novelty_threshold, allocation_ema, active_prototypes, upgd_conf,
    memory_conf``.
    """
    if type(learner) is not UPGDMemoryLearner:
        raise TypeError("learner must be a UPGDMemoryLearner")
    learner._validate_state_static_contract(state)  # noqa: SLF001
    try:
        observation_shape = tuple(getattr(observations, "shape"))
        observation_dtype = jnp.dtype(getattr(observations, "dtype"))
    except Exception as error:
        raise TypeError("observations must expose valid array metadata") from error
    try:
        target_shape = tuple(getattr(targets, "shape"))
        target_dtype = jnp.dtype(getattr(targets, "dtype"))
    except Exception as error:
        raise TypeError("targets must expose valid array metadata") from error
    if len(observation_shape) != 2 or observation_shape[1:] != (
        learner.config.feature_dim,
    ):
        raise ValueError(
            f"observations must have shape (steps, {learner.config.feature_dim})"
        )
    steps = observation_shape[0]
    if type(steps) is not int or steps < 0:
        raise ValueError("UPGD memory step count must be a non-negative integer")
    if target_shape != (steps, learner.config.n_heads):
        raise ValueError(f"targets must have shape ({steps}, {learner.config.n_heads})")
    if observation_dtype != jnp.dtype(jnp.float32):
        raise TypeError("observations must have dtype float32")
    if target_dtype != jnp.dtype(jnp.float32):
        raise TypeError("targets must have dtype float32")
    _require_resource(
        "UPGD memory batch outputs",
        float32_scalars=steps * (learner.config.n_heads + 10),
        bool_scalars=steps,
    )
    float32_scalars, int32_scalars, uint32_scalars = _combined_state_resource_counts(
        learner.config.feature_dim,
        learner.config.n_heads,
        learner.config.hidden_sizes,
        learner.config.slots_per_class,
    )
    working_set_bytes = (
        4 * (float32_scalars + int32_scalars + uint32_scalars)
        + steps
        * (4 * (learner.config.feature_dim + 2 * learner.config.n_heads + 10) + 1)
    )
    if working_set_bytes > _UINT32_MAX:
        raise ValueError("UPGD memory scan working set must fit unsigned int32")
    if working_set_bytes > _MAX_PERSISTENT_STATE_BYTES:
        raise ValueError("UPGD memory scan working set exceeds 256 MiB")

    def step_fn(
        carry: UPGDMemoryState,
        batch: tuple[Array, Array],
    ) -> tuple[UPGDMemoryState, tuple[Array, Array, Array]]:
        observation, target = batch
        result = learner.update(carry, observation, target)
        return result.state, (
            result.predictions,
            result.metrics,
            result.update_applied,
        )

    final_state, (predictions, metrics, updates_applied) = jax.lax.scan(
        step_fn,
        state,
        (observations, targets),
    )
    return UPGDMemoryLearningResult(
        state=final_state,
        predictions=predictions,
        metrics=metrics,
        updates_applied=updates_applied,
    )
