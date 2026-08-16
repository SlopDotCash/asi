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
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Literal, cast

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


def finite_real_and_float32(name: str, value: object) -> tuple[Real, int, int, float]:
    """Return the original real, exact ratio, and finite binary32 rounding."""
    actual_type = type(value)
    if issubclass(actual_type, bool) or not issubclass(actual_type, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    real = cast(Real, value)
    try:
        numerator, denominator, narrowed = round_real_to_float32_with_ratio(real)
    except (FloatingPointError, OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must narrow to a finite float32, got {value!r}") from None
    if not math.isfinite(narrowed):
        raise ValueError(f"{name} must narrow to a finite float32, got {value!r}")
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
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
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
        raise ValueError(f"{name} must be in (0, 1], got {value!r}")
    return canonical_float32_storage(real, narrowed)


def _require_nonnegative_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = finite_real_and_float32(name, value)
    if real < 0.0 or numerator < 0 or narrowed < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return canonical_float32_storage(real, narrowed)


def _require_positive_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = finite_real_and_float32(name, value)
    if real <= 0.0 or numerator <= 0 or narrowed <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return canonical_float32_storage(real, narrowed)


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    actual_type = type(value)
    if issubclass(actual_type, bool) or not issubclass(actual_type, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    number = int(cast(Integral, value))
    if minimum is not None and number < minimum:
        if minimum == 1:
            raise ValueError(f"{name} must be positive, got {value!r}")
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative, got {value!r}")
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value!r}")
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
    def from_config(cls, config: dict[str, Any]) -> AssociativeMemoryConfig:
        """Reconstruct from :meth:`to_config` output."""
        payload = dict(config)
        payload.pop("type", None)
        return cls(**payload)


@chex.dataclass(frozen=True)
class AssociativeMemoryState:
    """State for :class:`AssociativeMemoryLearner`."""

    keys: Int[Array, "max_features key_width"]
    values: Float[Array, "max_features vocab_size"]
    utility: Float[Array, " max_features"]
    counts: Float[Array, " max_features"]
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
        raise ValueError(
            f"feature_family must be an actual string, got {feature_family!r}"
        )
    if feature_family not in {
        "position_token",
        "suffix_pair",
        "token_suffix_pair",
    }:
        raise ValueError(f"unknown feature_family: {feature_family!r}")
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
            f"normalize_by_weight must be a bool, got {config.normalize_by_weight!r}"
        )
    for name in (
        "adaptive_feature_family",
        "adaptive_window",
        "adaptive_budget",
    ):
        val = getattr(config, name)
        if type(val) is not bool:
            raise ValueError(f"{name} must be a boolean, got {val!r}")
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
    def from_config(cls, config: dict[str, Any]) -> AssociativeMemoryLearner:
        """Reconstruct from :meth:`to_config` output."""
        return cls(
            AssociativeMemoryConfig.from_config(cast(dict[str, Any], config["config"]))
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
            counts=jnp.zeros((c.max_features,), dtype=jnp.float32),
            last_update=jnp.zeros((c.max_features,), dtype=jnp.int32),
            prior=jnp.zeros((c.vocab_size,), dtype=jnp.float32),
            family_logits=jnp.zeros((FAMILY_COUNT,), dtype=jnp.float32),
            window_logits=jnp.zeros((c.suffix_length - 1,), dtype=jnp.float32),
            budget_logit=jnp.array(initial_budget_logit, dtype=jnp.float32),
            allocations=jnp.array(0, dtype=jnp.int32),
            replacements=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def feature_keys(
        self,
        context: Int[Array, " block_size"],
    ) -> tuple[Int[Array, "max_active key_width"], Int[Array, " max_active"]]:
        """Return fixed-shape active feature keys and a 0/1 mask."""
        c = self._config
        tokens = jnp.asarray(context, dtype=jnp.int32)
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

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        state: AssociativeMemoryState,
        context: Int[Array, " block_size"],
    ) -> AssociativeMemoryPrediction:
        """Predict label probabilities before any write."""
        keys, mask = self.feature_keys(context)
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
        prediction = self.predict(state, context)
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
            allocations = carry.allocations + (
                active_bool & (~found_scalar) & used_empty
            ).astype(jnp.int32)
            replacements = carry.replacements + (
                active_bool & (~found_scalar) & (~used_empty)
            ).astype(jnp.int32)
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
                        jnp.where(found_scalar, carry.counts[slot] + 1.0, 1.0)
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
            step_count=state.step_count + 1
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
            & floating_tree_is_finite(state)
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
