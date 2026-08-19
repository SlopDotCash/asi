# mypy: disable-error-code="call-arg,name-defined,unused-ignore"
"""Fixed-budget online associative memory for Step 2 sequence features.

This module implements a sparse key/value associative memory as a small
JAX-compatible online learner.  It is deliberately narrower than a transformer:
contexts are integer token windows, features are causal token/local-conjunction
keys, and a fixed-size table maps those keys to value logits.  Feature utility
is learned online from feature-level loss advantage and the table budget is
reused by replacing low-utility rows.
"""

from __future__ import annotations

import functools
import math
import operator
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from numbers import Real
from typing import Any, Literal, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

from alberta_framework._float32 import round_real_to_float32_with_ratio
from alberta_framework.core.update_safety import (
    floating_tree_is_finite,
    neutralize_array,
    safe_discrete_action,
    select_transaction,
)

_INT32_MAX: int = 2**31 - 1
_UINT32_MAX: int = 4294967295
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


def _require_sequence_resource(name: str, *, scalars: int, bools: int = 0) -> None:
    if scalars + bools > _INT32_MAX:
        raise ValueError(f"{name} scalar count must fit signed int32")
    if 4 * scalars + bools > _INT32_MAX:
        raise ValueError(f"{name} byte count must fit signed int32")


def _read_mapping(name: str, value: object) -> dict[str, Any]:
    if not issubclass(type(value), Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        return dict(cast(Mapping[str, Any], value))
    except Exception as error:
        raise ValueError(f"{name} mapping could not be read") from error


def _require_array(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    try:
        actual_shape = tuple(value.shape)  # type: ignore[attr-defined]
        actual_dtype = jnp.dtype(value.dtype)  # type: ignore[attr-defined]
    except Exception as error:
        raise TypeError(f"{name} must expose array shape and dtype metadata") from error
    if actual_shape != shape:
        raise ValueError(f"{name} has an invalid shape")
    if actual_dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} has an invalid dtype")


def _saturating_increment(value: Array) -> Array:
    return jnp.minimum(value, jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)) + jnp.asarray(
        1, dtype=jnp.int32
    )


def _saturating_add_bool(value: Array, increment: Array) -> Array:
    room = value < jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    return value + (increment & room).astype(jnp.int32)


def finite_real_and_float32(name: str, value: object) -> tuple[Real, int, int, float]:
    """Return the original real, exact ratio, and finite binary32 rounding."""
    actual_type = type(value)
    if issubclass(actual_type, bool) or not issubclass(actual_type, Real):
        raise ValueError(f"{name} must be a real number")
    real = cast(Real, value)
    try:
        numerator, denominator, narrowed = round_real_to_float32_with_ratio(real)
    except Exception:
        raise ValueError(f"{name} must narrow to a finite float32") from None
    if not math.isfinite(narrowed):
        raise ValueError(f"{name} must narrow to a finite float32")
    return real, numerator, denominator, narrowed


def canonical_float32_storage(value: Real, narrowed: float) -> float:
    if not isinstance(value, (int, float, np.floating)):
        return narrowed
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return narrowed
    if not math.isfinite(number):
        raise ValueError("scalar must be finite")
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        renarrowed = np.asarray(number, dtype=np.float32)
    if not bool(np.array_equal(narrowed, renarrowed)):
        number = float(narrowed)
    return number


def _require_unit_interval(name: str, value: object) -> float:
    real, numerator, denominator, narrowed = finite_real_and_float32(name, value)
    if (
        real < 0.0
        or not real <= 1.0
        or numerator < 0
        or numerator > denominator
        or narrowed < 0.0
        or not narrowed <= 1.0
    ):
        raise ValueError(f"{name} must be in [0, 1], got invalid value")
    return canonical_float32_storage(real, narrowed)


def _require_half_open_unit_interval(name: str, value: object) -> float:
    real, numerator, denominator, narrowed = finite_real_and_float32(name, value)
    if (
        real <= 0.0
        or not real <= 1.0
        or numerator <= 0
        or numerator > denominator
        or narrowed <= 0.0
        or not narrowed <= 1.0
    ):
        raise ValueError(f"{name} must be in (0, 1], got invalid value")
    return canonical_float32_storage(real, narrowed)


def _require_nonnegative_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = finite_real_and_float32(name, value)
    if real < 0.0 or numerator < 0 or narrowed < 0.0:
        raise ValueError(f"{name} must be non-negative, got invalid value")
    return canonical_float32_storage(real, narrowed)


def _require_positive_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = finite_real_and_float32(name, value)
    if real <= 0.0 or numerator <= 0 or narrowed <= 0.0:
        raise ValueError(f"{name} must be positive, got invalid value")
    return canonical_float32_storage(real, narrowed)


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


AssociativeFeatureFamily = Literal[
    "position_token",
    "suffix_pair",
    "token_suffix_pair",
]

KEY_WIDTH = 5
EMPTY_KEY_VALUE = -1
FAMILY_COUNT = 2
POSITION_TOKEN_FAMILY = 0
SUFFIX_PAIR_FAMILY = 1


@dataclass(frozen=True)
class AssociativeMemoryConfig:
    """Configuration for :class:`AssociativeMemoryLearner`.

    Args:
        vocab_size: Number of discrete labels/tokens.
        block_size: Context length.
        suffix_length: Number of recent tokens used for local pair features.
        feature_family: Active feature family. ``"token_suffix_pair"``
            enables both the position-token and suffix-pair families.
        max_features: Fixed row budget for the associative table.
        write_lr: Additive write rate for the observed label.
        retention: Multiplicative decay applied to prior and row values.
        utility_lr: Step-size for feature utility updates.
        utility_decay: Decay for feature utility traces.
        min_weight: Lower clamp for exponentiated feature utility.
        max_weight: Upper clamp for exponentiated feature utility.
        logit_scale: Scale applied to weighted row evidence.
        normalize_by_weight: Divide logits by active feature weight mass.
        adaptive_feature_family: Learn a soft gate over token-position and
            suffix-pair feature families from per-family loss advantage.
        adaptive_window: Learn a soft suffix-window selector over lengths
            ``2..suffix_length`` from per-window loss advantage.
        adaptive_budget: Learn an effective table budget gate from replacement
            pressure and loss relative to the uniform baseline.
        scope_lr: Step-size shared by the optional scope controllers.
        budget_lr: Step-size for the optional adaptive budget controller.
        initial_budget_fraction: Initial fraction of ``max_features`` exposed
            by the adaptive budget gate when ``adaptive_budget`` is enabled.
        min_effective_budget: Lower bound for the adaptive effective budget.
        scope_logit_clip: Absolute clamp for learned scope logits.
    """

    vocab_size: int
    block_size: int
    suffix_length: int = 8
    feature_family: AssociativeFeatureFamily = "token_suffix_pair"
    max_features: int = 4096
    write_lr: float = 1.0
    retention: float = 0.80
    utility_lr: float = 0.10
    utility_decay: float = 0.995
    min_weight: float = 0.02
    max_weight: float = 8.0
    logit_scale: float = 4.0
    normalize_by_weight: bool = True
    adaptive_feature_family: bool = False
    adaptive_window: bool = False
    adaptive_budget: bool = False
    scope_lr: float = 0.05
    budget_lr: float = 0.05
    initial_budget_fraction: float = 0.5
    min_effective_budget: int = 1
    scope_logit_clip: float = 8.0

    def __post_init__(self) -> None:
        """Validate and canonicalize configuration."""
        _validate_config(self)

    def to_config(self) -> dict[str, object]:
        """Serialize to a plain config dictionary."""
        payload = asdict(self)
        payload["type"] = "AssociativeMemoryConfig"
        return payload

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> AssociativeMemoryConfig:
        """Reconstruct from :meth:`to_config` output."""
        payload = _read_mapping("AssociativeMemoryConfig payload", config)
        if payload.pop("type", None) != "AssociativeMemoryConfig":
            raise ValueError("AssociativeMemoryConfig type is invalid")
        if set(payload) != {field.name for field in fields(cls)}:
            raise ValueError("AssociativeMemoryConfig fields do not match its schema")
        integer_fields = {
            "vocab_size",
            "block_size",
            "suffix_length",
            "max_features",
            "min_effective_budget",
        }
        boolean_fields = {
            "normalize_by_weight",
            "adaptive_feature_family",
            "adaptive_window",
            "adaptive_budget",
        }
        string_fields = {"feature_family"}
        for name, value in payload.items():
            if name in integer_fields and type(value) is not int:
                raise ValueError(f"serialized {name} must be a JSON integer")
            if name in boolean_fields and type(value) is not bool:
                raise ValueError(f"serialized {name} must be a JSON boolean")
            if name in string_fields and type(value) is not str:
                raise ValueError(f"serialized {name} must be a JSON string")
            if (
                name not in integer_fields | boolean_fields | string_fields
                and type(value) is not float
            ):
                raise ValueError(f"serialized {name} must be a JSON number")
        return cls(**payload)


@chex.dataclass(frozen=True)
class AssociativeMemoryState:
    """State for :class:`AssociativeMemoryLearner`."""

    keys: Int[Array, "max_features key_width"]
    values: Float[Array, "max_features vocab_size"]
    utility: Float[Array, " max_features"]
    counts: Int[Array, " max_features"]
    last_update: Int[Array, " max_features"]
    prior: Float[Array, " vocab_size"]
    family_logits: Float[Array, " family_count"]
    window_logits: Float[Array, " window_count"]
    budget_logit: Float[Array, ""]
    allocations: Int[Array, ""]
    replacements: Int[Array, ""]
    step_count: Int[Array, ""]


@chex.dataclass(frozen=True)
class AssociativeMemoryPrediction:
    """Prediction internals returned for diagnostics and update reuse."""

    logits: Float[Array, " vocab_size"]
    probabilities: Float[Array, " vocab_size"]
    feature_keys: Int[Array, "max_active key_width"]
    feature_mask: Int[Array, " max_active"]
    found: Int[Array, " max_active"]
    indices: Int[Array, " max_active"]
    base_weights: Float[Array, " max_active"]
    scope_weights: Float[Array, " max_active"]
    weights: Float[Array, " max_active"]
    total_weight: Float[Array, ""]
    family_probs: Float[Array, " family_count"]
    window_probs: Float[Array, " window_count"]
    effective_budget: Float[Array, ""]


@chex.dataclass(frozen=True)
class AssociativeMemoryUpdateResult:
    """Result from one online associative-memory update."""

    state: AssociativeMemoryState
    predictions: Float[Array, " vocab_size"]
    logits: Float[Array, " vocab_size"]
    metrics: Float[Array, " 8"]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class AssociativeMemoryLearningResult:
    """Result from :func:`run_associative_memory_arrays`."""

    state: AssociativeMemoryState
    predictions: Float[Array, "steps vocab_size"]
    metrics: Float[Array, "steps 8"]
    updates_applied: Bool[Array, " steps"]


def _associative_resource_counts(
    *, vocab_size: int, block_size: int, suffix_length: int, max_features: int
) -> tuple[int, int, int, int, int]:
    pair_count = suffix_length * (suffix_length - 1) // 2
    active = block_size + pair_count
    windows = suffix_length - 1
    persistent = max_features * vocab_size + 8 * max_features + vocab_size + suffix_length + 5
    descriptors = 5 * pair_count + 2 * block_size + suffix_length - 1
    window_matrix = pair_count * windows
    query = (
        persistent
        + block_size
        + 16 * active
        + active * max_features
        + max_features
        + active * vocab_size
        + window_matrix
        + 4 * vocab_size
        + 5 * windows
        + 32
    )
    update = (
        query
        + 2 * persistent
        + active * vocab_size
        + 10 * active
        + 5 * vocab_size
        + 5 * windows
        + 64
    )
    return pair_count, active, descriptors, query, update


def _validate_config(config: AssociativeMemoryConfig) -> None:
    vocab_size = _require_int(
        "vocab_size", config.vocab_size, minimum=2, maximum=_INT32_MAX
    )
    block_size = _require_int(
        "block_size", config.block_size, minimum=1, maximum=_INT32_MAX
    )
    suffix_length = _require_int(
        "suffix_length", config.suffix_length, minimum=2, maximum=block_size
    )
    feature_family = config.feature_family
    if type(feature_family) is not str:
        raise ValueError("feature_family must be an actual string")
    if feature_family not in {
        "position_token",
        "suffix_pair",
        "token_suffix_pair",
    }:
        raise ValueError("unknown feature_family")
    canonical_feature_family = str(feature_family)
    max_features = _require_int(
        "max_features", config.max_features, minimum=1, maximum=_INT32_MAX
    )
    write_lr = _require_positive_real("write_lr", config.write_lr)
    retention = _require_unit_interval("retention", config.retention)
    utility_lr = _require_nonnegative_real("utility_lr", config.utility_lr)
    utility_decay = _require_unit_interval("utility_decay", config.utility_decay)
    min_weight = _require_positive_real("min_weight", config.min_weight)
    max_weight = _require_positive_real("max_weight", config.max_weight)
    if max_weight < min_weight:
        raise ValueError("max_weight must be >= min_weight")
    logit_scale = _require_positive_real("logit_scale", config.logit_scale)
    if type(config.normalize_by_weight) is not bool:
        raise ValueError(
            "normalize_by_weight must be a bool"
        )
    for name in (
        "adaptive_feature_family",
        "adaptive_window",
        "adaptive_budget",
    ):
        val = getattr(config, name)
        if type(val) is not bool:
            raise ValueError(f"{name} must be a boolean")
        object.__setattr__(config, name, bool(val))
    scope_lr = _require_nonnegative_real("scope_lr", config.scope_lr)
    budget_lr = _require_nonnegative_real("budget_lr", config.budget_lr)
    initial_budget_fraction = _require_half_open_unit_interval(
        "initial_budget_fraction", config.initial_budget_fraction
    )
    min_effective_budget = _require_int(
        "min_effective_budget",
        config.min_effective_budget,
        minimum=1,
        maximum=max_features,
    )
    scope_logit_clip = _require_positive_real(
        "scope_logit_clip", config.scope_logit_clip
    )

    object.__setattr__(config, "vocab_size", vocab_size)
    object.__setattr__(config, "block_size", block_size)
    object.__setattr__(config, "suffix_length", suffix_length)
    object.__setattr__(config, "feature_family", canonical_feature_family)
    object.__setattr__(config, "max_features", max_features)
    object.__setattr__(config, "write_lr", write_lr)
    object.__setattr__(config, "retention", retention)
    object.__setattr__(config, "utility_lr", utility_lr)
    object.__setattr__(config, "utility_decay", utility_decay)
    object.__setattr__(config, "min_weight", min_weight)
    object.__setattr__(config, "max_weight", max_weight)
    object.__setattr__(config, "logit_scale", logit_scale)
    object.__setattr__(config, "normalize_by_weight", bool(config.normalize_by_weight))
    object.__setattr__(config, "scope_lr", scope_lr)
    object.__setattr__(config, "budget_lr", budget_lr)
    object.__setattr__(config, "initial_budget_fraction", initial_budget_fraction)
    object.__setattr__(config, "min_effective_budget", min_effective_budget)
    object.__setattr__(config, "scope_logit_clip", scope_logit_clip)
    if max_features * vocab_size > _INT32_MAX:
        raise ValueError("AssociativeMemoryConfig dimensions must fit signed int32")
    if max_features * block_size > _INT32_MAX:
        raise ValueError("AssociativeMemoryConfig dimensions must fit signed int32")
    pair_count, active_features, descriptor_scalars, query_scalars, update_scalars = (
        _associative_resource_counts(
            vocab_size=vocab_size,
            block_size=block_size,
            suffix_length=suffix_length,
            max_features=max_features,
        )
    )
    for name, count in (
        ("pair count", pair_count),
        ("active feature count", active_features),
        ("feature-key count", 5 * active_features),
        ("lookup match count", active_features * max_features),
        ("row-value count", active_features * vocab_size),
        ("window-scope count", pair_count * (suffix_length - 1)),
    ):
        if count > _INT32_MAX:
            raise ValueError(f"AssociativeMemoryConfig {name} must fit signed int32")
    total_values_scalars = max_features * vocab_size
    fixed_state_scalars = (
        8 * max_features + vocab_size + suffix_length + 5
    )
    _require_float32_resource(
        "AssociativeMemoryConfig state",
        vector_scalars=total_values_scalars,
        fixed_scalars=fixed_state_scalars,
    )
    persistent_bytes = 4 * (total_values_scalars + fixed_state_scalars)
    if persistent_bytes > _INT32_MAX:
        raise ValueError("AssociativeMemoryConfig state byte count must fit signed int32")
    if persistent_bytes > _UINT32_MAX:
        raise ValueError("associative memory allocation exceeds uint32 byte accounting")
    _require_float32_resource(
        "AssociativeMemoryConfig descriptors", vector_scalars=descriptor_scalars
    )
    _require_float32_resource(
        "AssociativeMemoryConfig query", vector_scalars=query_scalars
    )
    _require_float32_resource(
        "AssociativeMemoryConfig update", vector_scalars=update_scalars
    )


def _softmax(logits: Array) -> Array:
    shifted = logits - jnp.max(logits)
    exp = jnp.exp(shifted)
    return exp / jnp.maximum(jnp.sum(exp), 1e-12)


def _masked_softmax(logits: Array, mask: Array) -> Array:
    active = mask > 0
    masked = jnp.where(active, logits, -1.0e9)
    shifted = masked - jnp.max(masked)
    exp = jnp.where(active, jnp.exp(shifted), 0.0)
    return exp / jnp.maximum(jnp.sum(exp), 1e-12)


def _cross_entropy_from_logits(logits: Array, label: Array) -> Array:
    safe_label = jnp.clip(label.astype(jnp.int32), 0, logits.shape[0] - 1)
    shifted = logits - jnp.max(logits)
    log_z = jnp.log(jnp.sum(jnp.exp(shifted))) + jnp.max(logits)
    return log_z - logits[safe_label]


class AssociativeMemoryLearner:
    """Fixed-budget feature-to-label associative learner.

    The learner predicts before writing the current example.  Rows are keyed by
    causal features of the current token context; row values are decayed and
    updated toward the observed next label.  Feature utilities are updated from
    whether each row would have produced lower loss than the aggregate
    prediction, and low-utility rows are replaced when the fixed table is full.
    """

    def __init__(self, config: AssociativeMemoryConfig):
        _validate_config(config)
        self._config = config
        self._pair_left, self._pair_right = self._build_pair_indices(
            config.suffix_length
        )
        self._family_ids = jnp.concatenate(
            [
                jnp.full(
                    (config.block_size,),
                    POSITION_TOKEN_FAMILY,
                    dtype=jnp.int32,
                ),
                jnp.full(
                    (self._pair_left.shape[0],),
                    SUFFIX_PAIR_FAMILY,
                    dtype=jnp.int32,
                ),
            ],
            axis=0,
        )
        self._window_lengths = jnp.arange(2, config.suffix_length + 1, dtype=jnp.int32)
        self._pair_required_window = config.suffix_length - self._pair_left
        self._feature_required_window = jnp.concatenate(
            [
                jnp.zeros((config.block_size,), dtype=jnp.int32),
                self._pair_required_window,
            ],
            axis=0,
        )

    @staticmethod
    def _build_pair_indices(suffix_length: int) -> tuple[Array, Array]:
        left: list[int] = []
        right: list[int] = []
        for i in range(suffix_length):
            for j in range(i + 1, suffix_length):
                left.append(i)
                right.append(j)
        return (
            jnp.asarray(left, dtype=jnp.int32),
            jnp.asarray(right, dtype=jnp.int32),
        )

    @property
    def config(self) -> AssociativeMemoryConfig:
        """Learner configuration."""
        return self._config

    @property
    def max_active_features(self) -> int:
        """Maximum number of context features generated per step."""
        pair_count = self._config.suffix_length * (self._config.suffix_length - 1) // 2
        return self._config.block_size + pair_count

    def to_config(self) -> dict[str, object]:
        """Serialize the learner."""
        return {
            "type": "AssociativeMemoryLearner",
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> AssociativeMemoryLearner:
        """Reconstruct from :meth:`to_config` output."""
        payload = _read_mapping("AssociativeMemoryLearner payload", config)
        if set(payload) != {"type", "config"}:
            raise ValueError("AssociativeMemoryLearner fields do not match its schema")
        if payload["type"] != "AssociativeMemoryLearner":
            raise ValueError("AssociativeMemoryLearner type is invalid")
        inner = payload["config"]
        if not issubclass(type(inner), Mapping):
            raise ValueError("AssociativeMemoryLearner config must be a mapping")
        return cls(AssociativeMemoryConfig.from_config(cast(Mapping[str, Any], inner)))

    def _validate_state_static_contract(self, state: AssociativeMemoryState) -> None:
        if type(state) is not AssociativeMemoryState:
            raise TypeError("state must be an AssociativeMemoryState")
        c = self._config
        expected = (
            ("state.keys", state.keys, (c.max_features, KEY_WIDTH), jnp.int32),
            ("state.values", state.values, (c.max_features, c.vocab_size), jnp.float32),
            ("state.utility", state.utility, (c.max_features,), jnp.float32),
            ("state.counts", state.counts, (c.max_features,), jnp.int32),
            ("state.last_update", state.last_update, (c.max_features,), jnp.int32),
            ("state.prior", state.prior, (c.vocab_size,), jnp.float32),
            ("state.family_logits", state.family_logits, (FAMILY_COUNT,), jnp.float32),
            ("state.window_logits", state.window_logits, (c.suffix_length - 1,), jnp.float32),
            ("state.budget_logit", state.budget_logit, (), jnp.float32),
            ("state.allocations", state.allocations, (), jnp.int32),
            ("state.replacements", state.replacements, (), jnp.int32),
            ("state.step_count", state.step_count, (), jnp.int32),
        )
        for name, value, shape, dtype in expected:
            _require_array(name, value, shape=shape, dtype=dtype)

    @staticmethod
    def _state_is_valid(state: AssociativeMemoryState) -> Bool[Array, ""]:
        return (
            floating_tree_is_finite(state)
            & jnp.all(state.counts >= 0.0)
            & jnp.all(state.last_update >= 0)
            & (state.allocations >= 0)
            & (state.replacements >= 0)
            & (state.step_count >= 0)
            & jnp.all(state.last_update <= state.step_count)
        )

    def _validate_context_static_contract(self, context: object) -> None:
        _require_array(
            "context",
            context,
            shape=(self._config.block_size,),
            dtype=jnp.int32,
        )

    def init(self) -> AssociativeMemoryState:
        """Create an empty associative table."""
        c = self._config
        clipped_fraction = min(max(c.initial_budget_fraction, 1.0e-3), 1.0 - 1.0e-3)
        initial_budget_logit = math.log(clipped_fraction / (1.0 - clipped_fraction))
        return AssociativeMemoryState(
            keys=jnp.full(
                (c.max_features, KEY_WIDTH),
                EMPTY_KEY_VALUE,
                dtype=jnp.int32,
            ),
            values=jnp.zeros((c.max_features, c.vocab_size), dtype=jnp.float32),
            utility=jnp.zeros((c.max_features,), dtype=jnp.float32),
            counts=jnp.zeros((c.max_features,), dtype=jnp.int32),
            last_update=jnp.zeros((c.max_features,), dtype=jnp.int32),
            prior=jnp.zeros((c.vocab_size,), dtype=jnp.float32),
            family_logits=jnp.zeros((FAMILY_COUNT,), dtype=jnp.float32),
            window_logits=jnp.zeros((c.suffix_length - 1,), dtype=jnp.float32),
            budget_logit=jnp.array(initial_budget_logit, dtype=jnp.float32),
            allocations=jnp.array(0, dtype=jnp.int32),
            replacements=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def feature_keys(
        self,
        context: Int[Array, " block_size"],
    ) -> tuple[Int[Array, "max_active key_width"], Int[Array, " max_active"]]:
        """Return fixed-shape active feature keys and a 0/1 mask."""
        self._validate_context_static_contract(context)
        return cast(tuple[Array, Array], self._feature_keys_jit(context))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _feature_keys_jit(
        self,
        context: Int[Array, " block_size"],
    ) -> tuple[Int[Array, "max_active key_width"], Int[Array, " max_active"]]:
        c = self._config
        tokens = jnp.asarray(context)
        token_positions = jnp.arange(c.block_size, dtype=jnp.int32)
        token_keys = jnp.stack(
            [
                jnp.zeros((c.block_size,), dtype=jnp.int32),
                token_positions,
                tokens,
                jnp.full((c.block_size,), EMPTY_KEY_VALUE, dtype=jnp.int32),
                jnp.full((c.block_size,), EMPTY_KEY_VALUE, dtype=jnp.int32),
            ],
            axis=1,
        )
        suffix = tokens[c.block_size - c.suffix_length :]
        left_tokens = suffix[self._pair_left]
        right_tokens = suffix[self._pair_right]
        pair_keys = jnp.stack(
            [
                jnp.ones_like(self._pair_left),
                self._pair_left,
                self._pair_right,
                left_tokens,
                right_tokens,
            ],
            axis=1,
        )
        keys = jnp.concatenate([token_keys, pair_keys], axis=0)
        token_enabled = c.feature_family in {"position_token", "token_suffix_pair"}
        pair_enabled = c.feature_family in {"suffix_pair", "token_suffix_pair"}
        token_mask = jnp.full((c.block_size,), token_enabled, dtype=jnp.bool_)
        pair_mask = jnp.full((self._pair_left.shape[0],), pair_enabled, dtype=jnp.bool_)
        return keys, jnp.concatenate([token_mask, pair_mask], axis=0).astype(jnp.int32)

    def _lookup(
        self,
        state: AssociativeMemoryState,
        keys: Array,
        mask: Array,
    ) -> tuple[Array, Array]:
        occupied = state.counts > 0.0
        matches = jnp.all(state.keys[None, :, :] == keys[:, None, :], axis=2)
        matches = matches & occupied[None, :] & (mask[:, None] > 0)
        found = jnp.any(matches, axis=1)
        indices = jnp.argmax(matches.astype(jnp.int32), axis=1)
        return found, indices.astype(jnp.int32)

    def _feature_weights(self, utility: Array) -> Array:
        # Mirror of the stored-trace clip in ``update``; keeps ``exp`` finite
        # for arbitrary restored states.  The [min_weight, max_weight] clamp
        # below is what actually bounds a feature's influence.
        clipped = jnp.clip(utility, -8.0, 8.0)
        return jnp.clip(jnp.exp(clipped), self._config.min_weight, self._config.max_weight)

    def _family_enabled(self) -> Array:
        c = self._config
        if c.feature_family == "position_token":
            return jnp.asarray([1.0, 0.0], dtype=jnp.float32)
        if c.feature_family == "suffix_pair":
            return jnp.asarray([0.0, 1.0], dtype=jnp.float32)
        return jnp.asarray([1.0, 1.0], dtype=jnp.float32)

    def _family_scope_weights(self, state: AssociativeMemoryState) -> tuple[Array, Array]:
        enabled = self._family_enabled()
        enabled_count = jnp.maximum(jnp.sum(enabled), 1.0)
        if self._config.adaptive_feature_family:
            probs = _masked_softmax(state.family_logits, enabled)
            scope = probs[self._family_ids] * enabled_count
        else:
            probs = enabled / enabled_count
            scope = jnp.ones((self.max_active_features,), dtype=jnp.float32)
        return scope, probs

    def _window_scope_weights(self, state: AssociativeMemoryState) -> tuple[Array, Array]:
        if self._config.adaptive_window:
            probs = _softmax(state.window_logits)
            pair_in_window = (
                self._window_lengths[None, :] >= self._pair_required_window[:, None]
            ).astype(jnp.float32)
            pair_scope = pair_in_window @ probs
        else:
            probs = jax.nn.one_hot(
                self._config.suffix_length - 2,
                self._config.suffix_length - 1,
                dtype=jnp.float32,
            )
            pair_scope = jnp.ones((self._pair_left.shape[0],), dtype=jnp.float32)
        scope = jnp.concatenate(
            [
                jnp.ones((self._config.block_size,), dtype=jnp.float32),
                pair_scope,
            ],
            axis=0,
        )
        return scope, probs

    def _effective_budget(self, state: AssociativeMemoryState) -> Array:
        if not self._config.adaptive_budget:
            return jnp.asarray(self._config.max_features, dtype=jnp.float32)
        fraction = jax.nn.sigmoid(state.budget_logit)
        budget_span = self._config.max_features - self._config.min_effective_budget
        return jnp.asarray(self._config.min_effective_budget, dtype=jnp.float32) + (
            fraction * budget_span
        )

    def _weighted_feature_loss(self, row_values: Array, weights: Array, label: Array) -> Array:
        evidence = jnp.sum(weights[:, None] * row_values, axis=0)
        logits = self._config.logit_scale * evidence
        total_weight = jnp.sum(weights)
        if self._config.normalize_by_weight:
            logits = jnp.where(total_weight > 0.0, logits / total_weight, logits)
        loss = _cross_entropy_from_logits(logits, label)
        uniform_loss = jnp.log(jnp.asarray(self._config.vocab_size, dtype=jnp.float32))
        return jnp.where(total_weight > 0.0, loss, uniform_loss)

    def predict(
        self,
        state: AssociativeMemoryState,
        context: Int[Array, " block_size"],
    ) -> AssociativeMemoryPrediction:
        """Predict label probabilities before any write."""
        self._validate_state_static_contract(state)
        self._validate_context_static_contract(context)
        return cast(AssociativeMemoryPrediction, self._predict_jit(state, context))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _predict_jit(
        self,
        state: AssociativeMemoryState,
        context: Int[Array, " block_size"],
    ) -> AssociativeMemoryPrediction:
        keys, mask = self._feature_keys_jit(context)
        found, indices = self._lookup(state, keys, mask)
        row_values = state.values[indices]
        row_utility = state.utility[indices]
        base_weights = jnp.where(found, self._feature_weights(row_utility), 0.0)
        family_scope, family_probs = self._family_scope_weights(state)
        window_scope, window_probs = self._window_scope_weights(state)
        scope_weights = family_scope * window_scope * mask.astype(jnp.float32)
        weights = base_weights * scope_weights
        weight_col = weights[:, None]
        # Silent features have weight 0. An inf stored value times that
        # weight is 0*inf = NaN. Skip the product on exact zeros.
        evidence = jnp.sum(
            jnp.where(weight_col == 0.0, jnp.zeros_like(row_values), weight_col * row_values),
            axis=0,
        )
        # The decayed label-frequency prior enters with a small fixed weight:
        # it dominates only when no feature rows match (evidence is zero, so
        # the prediction falls back to base rates) and is otherwise a weak
        # tiebreak against feature evidence scaled by ``logit_scale``.
        logits = 0.05 * state.prior + self._config.logit_scale * evidence
        total_weight = jnp.sum(weights)
        if self._config.normalize_by_weight:
            logits = jnp.where(total_weight > 0.0, logits / total_weight, logits)
        probabilities = _softmax(logits)
        return AssociativeMemoryPrediction(
            logits=logits,
            probabilities=probabilities,
            feature_keys=keys,
            feature_mask=mask,
            found=found.astype(jnp.int32),
            indices=indices,
            base_weights=base_weights,
            scope_weights=scope_weights,
            weights=weights,
            total_weight=total_weight,
            family_probs=family_probs,
            window_probs=window_probs,
            effective_budget=self._effective_budget(state),
        )

    def _replacement_slot(self, state: AssociativeMemoryState) -> tuple[Array, Array]:
        eligible = (
            jnp.arange(self._config.max_features, dtype=jnp.float32)
            < self._effective_budget(state)
        )
        empty = (state.counts <= 0.0) & eligible
        has_empty = jnp.any(empty)
        empty_slot = jnp.argmax(empty.astype(jnp.int32))
        utility_score = jnp.where(eligible & (~empty), state.utility, jnp.inf)
        low_utility_slot = jnp.argmin(utility_score)
        return jnp.where(has_empty, empty_slot, low_utility_slot), has_empty

    def _update_family_scope(
        self,
        next_state: AssociativeMemoryState,
        state: AssociativeMemoryState,
        prediction: AssociativeMemoryPrediction,
        label: Array,
        loss: Array,
    ) -> AssociativeMemoryState:
        if not self._config.adaptive_feature_family:
            return next_state
        row_values = state.values[prediction.indices]
        window_scope, _ = self._window_scope_weights(state)
        enabled = self._family_enabled()

        def family_loss(family_id: Array) -> Array:
            family_mask = (
                (self._family_ids == family_id)
                & (prediction.feature_mask > 0)
                & (prediction.found > 0)
            )
            weights = prediction.base_weights * window_scope * family_mask.astype(jnp.float32)
            return self._weighted_feature_loss(row_values, weights, label)

        family_ids = jnp.arange(FAMILY_COUNT, dtype=jnp.int32)
        family_losses = jax.vmap(family_loss)(family_ids)
        advantages = (loss - family_losses) * enabled
        mean_advantage = jnp.sum(advantages) / jnp.maximum(jnp.sum(enabled), 1.0)
        centered = (advantages - mean_advantage) * enabled
        logits = jnp.clip(
            state.family_logits + self._config.scope_lr * centered,
            -self._config.scope_logit_clip,
            self._config.scope_logit_clip,
        )
        return cast(
            AssociativeMemoryState,
            next_state.replace(family_logits=logits),  # type: ignore[attr-defined]
        )

    def _update_window_scope(
        self,
        next_state: AssociativeMemoryState,
        state: AssociativeMemoryState,
        prediction: AssociativeMemoryPrediction,
        label: Array,
        loss: Array,
    ) -> AssociativeMemoryState:
        if not self._config.adaptive_window:
            return next_state
        if self._config.feature_family == "position_token":
            return next_state
        row_values = state.values[prediction.indices]

        def window_loss(window_length: Array) -> Array:
            window_mask = (
                (self._family_ids == SUFFIX_PAIR_FAMILY)
                & (self._feature_required_window <= window_length)
                & (prediction.feature_mask > 0)
                & (prediction.found > 0)
            )
            weights = prediction.base_weights * window_mask.astype(jnp.float32)
            return self._weighted_feature_loss(row_values, weights, label)

        window_losses = jax.vmap(window_loss)(self._window_lengths)
        advantages = loss - window_losses
        centered = advantages - jnp.mean(advantages)
        logits = jnp.clip(
            state.window_logits + self._config.scope_lr * centered,
            -self._config.scope_logit_clip,
            self._config.scope_logit_clip,
        )
        return cast(
            AssociativeMemoryState,
            next_state.replace(window_logits=logits),  # type: ignore[attr-defined]
        )

    def _update_budget_scope(
        self,
        next_state: AssociativeMemoryState,
        state: AssociativeMemoryState,
        prediction: AssociativeMemoryPrediction,
        loss: Array,
    ) -> AssociativeMemoryState:
        if not self._config.adaptive_budget:
            return next_state
        active_count = jnp.sum(prediction.feature_mask.astype(jnp.float32))
        replacement_delta = (
            next_state.replacements.astype(jnp.float32)
            - state.replacements.astype(jnp.float32)
        )
        replacement_rate = jnp.clip(
            replacement_delta / jnp.maximum(active_count, 1.0),
            0.0,
            1.0,
        )
        uniform_loss = jnp.log(jnp.asarray(self._config.vocab_size, dtype=jnp.float32))
        loss_pressure = jnp.clip((loss / uniform_loss) - 1.0, -1.0, 1.0)
        grow = replacement_rate * (1.0 + jnp.maximum(loss_pressure, 0.0))
        shrink = (1.0 - replacement_rate) * jnp.maximum(-loss_pressure, 0.0)
        # Asymmetric controller: shrink pressure is discounted 4x relative to
        # growth, so the exposed budget contracts more slowly than it expands.
        budget_delta = self._config.budget_lr * (grow - 0.25 * shrink)
        budget_logit = jnp.clip(
            state.budget_logit + budget_delta,
            -self._config.scope_logit_clip,
            self._config.scope_logit_clip,
        )
        return cast(
            AssociativeMemoryState,
            next_state.replace(budget_logit=budget_logit),  # type: ignore[attr-defined]
        )

    def update(
        self,
        state: AssociativeMemoryState,
        context: Int[Array, " block_size"],
        label: Int[Array, ""],
    ) -> AssociativeMemoryUpdateResult:
        """Predict, then update active associative rows."""
        self._validate_state_static_contract(state)
        self._validate_context_static_contract(context)
        safe_label, label_valid = self._prepare_label(label)
        return cast(
            AssociativeMemoryUpdateResult,
            self._update(state, context, safe_label, label_valid),
        )

    def _prepare_label(self, label: object) -> tuple[Array, Array]:
        """Validate a discrete label before JAX can narrow or reshape it."""
        actual_type = type(label)
        if issubclass(actual_type, jax.core.Tracer):
            array = jnp.asarray(label)
            if not (jnp.issubdtype(array.dtype, jnp.integer) and array.shape == ()):
                return jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False)
            # Static checks cannot see traced values, so the vocabulary-domain
            # verdict must be a runtime predicate folded into update_applied.
            return safe_discrete_action(array, self._config.vocab_size)

        allowed = (
            actual_type is int
            or issubclass(actual_type, np.integer)
            or actual_type is np.ndarray
            or issubclass(actual_type, jax.Array)
        )
        if not allowed:
            return jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False)
        try:
            host = np.asarray(label)
        except (OverflowError, TypeError, ValueError):
            return jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False)
        if host.shape != () or host.dtype.kind not in ("i", "u"):
            return jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False)
        value = int(host.item())
        if value < 0 or value >= self._config.vocab_size:
            return jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False)
        return jnp.asarray(value, dtype=jnp.int32), jnp.asarray(True)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update(
        self,
        state: AssociativeMemoryState,
        context: Int[Array, " block_size"],
        label: Int[Array, ""],
        label_valid: Bool[Array, ""],
    ) -> AssociativeMemoryUpdateResult:
        """Execute one already-domain-checked associative transaction."""
        prediction = self._predict_jit(state, context)
        loss = _cross_entropy_from_logits(prediction.logits, label)
        accuracy = (jnp.argmax(prediction.logits) == label).astype(jnp.float32)
        next_state = state.replace(  # type: ignore[attr-defined]
            prior=state.prior * self._config.retention,
        )
        next_state = next_state.replace(  # type: ignore[attr-defined]
            prior=next_state.prior.at[label].add(self._config.write_lr)
        )

        def row_step(
            carry: AssociativeMemoryState,
            inputs: tuple[Array, Array],
        ) -> tuple[AssociativeMemoryState, Array]:
            key, active = inputs
            found, indices = self._lookup(carry, key[None, :], active[None])
            found_scalar = found[0]
            existing_slot = indices[0]
            replacement_slot, used_empty = self._replacement_slot(carry)
            slot = jnp.where(found_scalar, existing_slot, replacement_slot)
            # A slot allocated or reused for a new key starts empty: the values,
            # utility, and count of an evicted occupant must not be inherited.
            slot_values = carry.values[slot]
            old_row = jnp.where(found_scalar, slot_values, jnp.zeros_like(slot_values))
            old_utility = jnp.where(found_scalar, carry.utility[slot], 0.0)
            row_logits = self._config.logit_scale * old_row
            feature_loss = jnp.where(
                found_scalar,
                _cross_entropy_from_logits(row_logits, label),
                jnp.log(jnp.asarray(self._config.vocab_size, dtype=jnp.float32)),
            )
            new_utility = (
                self._config.utility_decay * old_utility
                + self._config.utility_lr * (loss - feature_loss)
            )
            # Clip the stored trace, not just the derived weight: ±8 is far
            # outside the range that moves the clamped weight (ln of the
            # default weight clamps is about [-3.9, +2.1]), so this bound
            # caps how deeply a row can saturate and keeps the number of
            # opposing-sign updates needed to recover bounded.
            new_utility = jnp.clip(new_utility, -8.0, 8.0)
            new_row = old_row * self._config.retention
            new_row = new_row.at[label].add(self._config.write_lr)
            active_bool = active > 0
            allocations = _saturating_add_bool(
                carry.allocations,
                active_bool & (~found_scalar) & used_empty,
            )
            replacements = _saturating_add_bool(
                carry.replacements,
                active_bool & (~found_scalar) & (~used_empty),
            )
            next_carry = carry.replace(  # type: ignore[attr-defined]
                keys=jnp.where(
                    active_bool,
                    carry.keys.at[slot].set(key),
                    carry.keys,
                ),
                values=jnp.where(
                    active_bool,
                    carry.values.at[slot].set(new_row),
                    carry.values,
                ),
                utility=jnp.where(
                    active_bool,
                    carry.utility.at[slot].set(new_utility),
                    carry.utility,
                ),
                counts=jnp.where(
                    active_bool,
                    carry.counts.at[slot].set(
                        jnp.where(
                            found_scalar,
                            _saturating_increment(carry.counts[slot]),
                            jnp.asarray(1, dtype=jnp.int32),
                        )
                    ),
                    carry.counts,
                ),
                last_update=jnp.where(
                    active_bool,
                    carry.last_update.at[slot].set(carry.step_count),
                    carry.last_update,
                ),
                allocations=allocations,
                replacements=replacements,
            )
            return next_carry, jnp.asarray(0, dtype=jnp.int32)

        next_state = jax.lax.scan(
            row_step,
            next_state,
            (prediction.feature_keys, prediction.feature_mask),
        )[0]
        next_state = self._update_family_scope(next_state, state, prediction, label, loss)
        next_state = self._update_window_scope(next_state, state, prediction, label, loss)
        next_state = self._update_budget_scope(next_state, state, prediction, loss)
        next_state = next_state.replace(  # type: ignore[attr-defined]
            step_count=_saturating_increment(state.step_count)
        )
        active_count = jnp.sum(prediction.feature_mask.astype(jnp.float32))
        occupied_count = jnp.sum((next_state.counts > 0.0).astype(jnp.float32))
        mean_weight = jnp.sum(prediction.weights) / jnp.maximum(
            jnp.sum(prediction.found.astype(jnp.float32)),
            1.0,
        )
        metrics = jnp.asarray(
            [
                loss,
                accuracy,
                active_count,
                occupied_count,
                mean_weight,
                next_state.allocations.astype(jnp.float32),
                next_state.replacements.astype(jnp.float32),
                prediction.total_weight,
            ],
            dtype=jnp.float32,
        )
        update_applied = (
            label_valid
            & self._state_is_valid(state)
            & floating_tree_is_finite(prediction)
            & jnp.isfinite(loss)
            & floating_tree_is_finite(next_state)
            & jnp.all(jnp.isfinite(metrics))
        )
        return AssociativeMemoryUpdateResult(
            state=select_transaction(update_applied, next_state, state),
            predictions=neutralize_array(update_applied, prediction.probabilities),
            logits=neutralize_array(update_applied, prediction.logits),
            metrics=neutralize_array(update_applied, metrics),
            update_applied=update_applied,
        )


def run_associative_memory_arrays(
    learner: AssociativeMemoryLearner,
    state: AssociativeMemoryState,
    contexts: Int[Array, "steps block_size"],
    labels: Int[Array, " steps"],
) -> AssociativeMemoryLearningResult:
    """Run a scan-compatible online associative learner over arrays."""
    if type(learner) is not AssociativeMemoryLearner:
        raise TypeError("learner must be an actual AssociativeMemoryLearner")
    if type(state) is not AssociativeMemoryState:
        raise TypeError("state must be an actual AssociativeMemoryState")
    learner._validate_state_static_contract(state)
    c = learner.config
    actual_context_type = type(cast(object, contexts))
    if not (
        actual_context_type is np.ndarray
        or isinstance(contexts, (jax.Array, jax.core.Tracer))
        or actual_context_type is jax.ShapeDtypeStruct
        or issubclass(actual_context_type, jax.core.ShapedArray)
        or (hasattr(contexts, "shape") and hasattr(contexts, "dtype"))
    ):
        raise TypeError("contexts must be a trusted array")
    actual_label_type = type(cast(object, labels))
    if not (
        actual_label_type is np.ndarray
        or isinstance(labels, (jax.Array, jax.core.Tracer))
        or actual_label_type is jax.ShapeDtypeStruct
        or issubclass(actual_label_type, jax.core.ShapedArray)
        or (hasattr(labels, "shape") and hasattr(labels, "dtype"))
    ):
        raise TypeError("labels must be a trusted array")
    if getattr(contexts, "ndim", len(getattr(contexts, "shape", ()))) != 2:
        raise ValueError("contexts must be 2-dimensional (steps, block_size)")
    if getattr(labels, "ndim", len(getattr(labels, "shape", ()))) != 1:
        raise ValueError("labels must be 1-dimensional (steps,)")
    try:
        steps = int(contexts.shape[0])
        block_size = int(contexts.shape[1])
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise TypeError("contexts must expose trusted shape metadata") from error
    try:
        label_steps = int(labels.shape[0])
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise TypeError("labels must expose trusted shape metadata") from error
    if not 1 <= steps <= _INT32_MAX:
        raise ValueError("associative memory step count must be between 1 and signed-int32 steps")
    if block_size != c.block_size:
        raise ValueError(
            f"contexts block_size ({block_size}) must match config block_size ({c.block_size})"
        )
    if label_steps != steps:
        raise ValueError(
            f"labels step count ({label_steps}) must match contexts step count ({steps})"
        )
    _, _, _, query_scalars, update_scalars = _associative_resource_counts(
        vocab_size=c.vocab_size,
        block_size=c.block_size,
        suffix_length=c.suffix_length,
        max_features=c.max_features,
    )
    _require_sequence_resource(
        "associative memory scan outputs",
        scalars=steps * (c.vocab_size + 8),
        bools=steps,
    )
    _require_sequence_resource(
        "associative memory scan aggregate",
        scalars=steps * (c.block_size + 1 + c.vocab_size + 8)
        + query_scalars
        + update_scalars,
        bools=steps,
    )

    def step_fn(
        carry: AssociativeMemoryState,
        inputs: tuple[Array, Array],
    ) -> tuple[AssociativeMemoryState, tuple[Array, Array, Array]]:
        context_t, label_t = inputs
        result = learner.update(carry, context_t, label_t)
        return result.state, (
            result.predictions,
            result.metrics,
            result.update_applied,
        )

    final_state, (predictions, metrics, updates_applied) = jax.lax.scan(
        step_fn,
        state,
        (contexts, labels),
    )
    return AssociativeMemoryLearningResult(
        state=final_state,
        predictions=predictions,
        metrics=metrics,
        updates_applied=updates_applied,
    )
