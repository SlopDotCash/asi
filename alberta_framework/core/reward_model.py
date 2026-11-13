# mypy: disable-error-code="call-arg"
"""Online reward models for selective model-based updates.

Implements a linear recursive-least-squares (RLS) scalar reward predictor
with exponential forgetting (standard exponentially weighted RLS; see e.g.
Haykin, *Adaptive Filter Theory*).  Its role in the model-based lane is to
supply calibrated reward targets for imagined (dream) transitions when the
shared multi-head dynamics model's reward head is too biased or too slow to
calibrate.  The ``abs_error_ema`` diagnostic gives callers an online
reliability estimate for deciding whether imagined rewards are currently
trustworthy — the "selective" part of selective model-based updates.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float

from alberta_framework.core._float32_scalars import validated_float32_scalar

_INT32_MAX = 2**31 - 1
_ACTUAL_INT_TYPES = frozenset({int, *(np.dtype(code).type for code in "bBhHiIlLqQpP")})
_ACTUAL_FLOAT_TYPES = frozenset(
    {float, *(np.dtype(code).type for code in ("e", "f", "d", "g"))}
)


def _require_int32(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    actual_type = type(value)
    if not any(actual_type is allowed_type for allowed_type in _ACTUAL_INT_TYPES):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _preflight_state_resources(feature_dim: int) -> None:
    state_scalars = feature_dim * feature_dim + feature_dim + 2
    if state_scalars > _INT32_MAX:
        raise ValueError("reward-model state scalars must fit signed int32")
    if 4 * state_scalars > _INT32_MAX:
        raise ValueError("reward-model state bytes must fit signed int32")


def _preflight_update_working_set(feature_dim: int) -> None:
    # Logical arrays, independently of backend donation/fusion: five covariance
    # banks (source, outer, unsymmetrized, symmetrized, selected), plus the
    # symmetrizer's two gathered values, two half-scaled values, and sum.
    # Its i/j/min/max int32 index grids and equality boolean mask also count.
    # Nine float32 vectors retain the existing update terms plus the two
    # normalized operands used by the overflow-safe prediction dot product.
    # Twenty-two scalar slots conservatively cover both update and stable-dot
    # values and predicates, with every slot charged at four bytes.
    matrix_cells = feature_dim * feature_dim
    update_bytes = 4 * (10 * matrix_cells + 9 * feature_dim + 22)
    update_bytes += (4 * 4 + 1) * matrix_cells
    if update_bytes > _INT32_MAX:
        raise ValueError(
            "reward-model update working set byte count must fit signed int32"
        )


def _validated_config_float(name: str, value: object, **bounds: Any) -> float:
    actual_type = type(value)
    if not any(
        actual_type is allowed_type
        for allowed_type in (*_ACTUAL_INT_TYPES, *_ACTUAL_FLOAT_TYPES)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return validated_float32_scalar(name, value, **bounds)


def _validated_feature_vector(features: Array, *, feature_dim: int) -> Array:
    """Validate feature identity before narrowing to the model's float32 domain."""
    raw = jnp.asarray(features)
    if raw.shape != (feature_dim,):
        raise ValueError(f"features must have shape ({feature_dim},), got {raw.shape}")
    if not (
        jnp.issubdtype(raw.dtype, jnp.floating)
        or jnp.issubdtype(raw.dtype, jnp.integer)
    ):
        raise TypeError("features must be real numeric")
    return raw.astype(jnp.float32)


def _skip_zero_scale(scale: Array, value: Array) -> Array:
    """Skip ``0 * inf`` so a disabled error EMA does not poison the diagnostic."""
    return jnp.where(scale == 0.0, jnp.zeros_like(value), scale * value)


def _saturating_int32_increment(value: Array) -> Array:
    """Increment a diagnostic counter without signed-int32 wraparound."""
    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    counter = jnp.asarray(value, dtype=jnp.int32)
    return jnp.minimum(jnp.maximum(counter, 0), maximum - 1) + 1


def _symmetrize_matrix(matrix: Array) -> Array:
    """Enforce exact float32 matrix symmetry without intermediate addition overflow."""
    half = jnp.asarray(0.5, dtype=matrix.dtype)
    n = matrix.shape[0]
    i, j = jnp.indices((n, n), dtype=jnp.int32)
    i_min = jnp.minimum(i, j)
    j_max = jnp.maximum(i, j)
    val_a = matrix[i_min, j_max]
    val_b = matrix[j_max, i_min]
    # Equal entries already are symmetric; halving tiny normal values can flush to zero.
    return jnp.where(val_a == val_b, val_a, half * val_a + half * val_b)


def _split_unit_mantissa(mantissa: Array) -> tuple[Array, Array]:
    """Split a float32 mantissa in ``(-1, 1)`` into exact high and low parts."""
    high = mantissa.astype(jnp.float16).astype(jnp.float32)
    return high, mantissa - high


def _neumaier_sum(values: Array) -> tuple[Array, Array]:
    """Return ``(sum, compensation)`` for a 1D float32 vector via Neumaier summation."""

    def _step(
        carry: tuple[Array, Array], x: Array
    ) -> tuple[tuple[Array, Array], None]:
        total, comp = carry
        next_total = total + x
        err = jnp.where(
            jnp.abs(total) >= jnp.abs(x),
            (total - next_total) + x,
            (x - next_total) + total,
        )
        return (next_total, comp + err), None

    (total, comp), _ = jax.lax.scan(
        _step,
        (jnp.float32(0.0), jnp.float32(0.0)),
        values,
    )
    return total, comp


def _tier_cancelling_sum(
    prod_hi: Array,
    prod_lo: Array,
    pair_exp: Array,
    mask: Array,
) -> Array:
    """Sum exact mantissa products within one exponent tier using compensated summation."""
    any_active = jnp.any(mask)
    max_exp = jnp.max(jnp.where(mask, pair_exp, jnp.int32(-1024)))
    scale_exp = jnp.where(any_active, max_exp - jnp.int32(60), jnp.int32(0))
    rel_exp = jnp.clip(pair_exp - scale_exp, -149, 64)
    scaled_hi = jnp.where(mask, jnp.ldexp(prod_hi, rel_exp), jnp.float32(0.0))
    scaled_lo = jnp.where(mask, jnp.ldexp(prod_lo, rel_exp), jnp.float32(0.0))
    sum_hi, comp_hi = _neumaier_sum(scaled_hi)
    sum_lo, comp_lo = _neumaier_sum(scaled_lo)
    total_scaled = (sum_hi + sum_lo) + (comp_hi + comp_lo)
    return jnp.where(any_active, jnp.ldexp(total_scaled, scale_exp), jnp.float32(0.0))


def _finite_cancelling_dot(left: Array, right: Array) -> Array:
    """Preserve a representable dot product when direct accumulation overflows."""
    left_f32 = jnp.asarray(left, dtype=jnp.float32)
    right_f32 = jnp.asarray(right, dtype=jnp.float32)
    direct = jnp.dot(left_f32, right_f32)
    if left_f32.size == 0:
        return direct
    all_finite = jnp.all(jnp.isfinite(left_f32)) & jnp.all(jnp.isfinite(right_f32))
    active = (left_f32 != jnp.float32(0.0)) & (right_f32 != jnp.float32(0.0))
    _, exp_l = jnp.frexp(jnp.where(all_finite, left_f32, jnp.float32(0.0)))
    _, exp_r = jnp.frexp(jnp.where(all_finite, right_f32, jnp.float32(0.0)))
    pair_exp = exp_l + exp_r
    max_pair_exp = jnp.max(jnp.where(active, pair_exp, jnp.int32(-1024)))
    use_fallback = all_finite & (
        (~jnp.isfinite(direct)) | (max_pair_exp > jnp.int32(100))
    )

    def compensated() -> Array:
        mant_l, exp_left = jnp.frexp(left_f32)
        mant_r, exp_right = jnp.frexp(right_f32)
        p_exp = exp_left + exp_right
        l_hi, l_lo = _split_unit_mantissa(mant_l)
        r_hi, r_lo = _split_unit_mantissa(mant_r)
        prod_hi = mant_l * mant_r
        prod_lo = ((l_hi * r_hi - prod_hi) + l_hi * r_lo + l_lo * r_hi) + l_lo * r_lo
        high_mask = active & (p_exp >= max_pair_exp - jnp.int32(90))
        low_mask = active & (~high_mask)
        return _tier_cancelling_sum(
            prod_hi, prod_lo, p_exp, high_mask
        ) + _tier_cancelling_sum(prod_hi, prod_lo, p_exp, low_mask)

    return cast(Array, jax.lax.cond(use_fallback, compensated, lambda: direct))


@dataclass(frozen=True)
class RLSRewardModelConfig:
    """Configuration for a linear recursive-least-squares reward model.

    Args:
        feature_dim: Number of scalar input features.
        forgetting: Exponential forgetting factor. Values near one favor stable
            estimates; lower values adapt faster to nonstationarity but risk
            covariance windup (see :meth:`RLSRewardModel.update`).
        ridge: Initial precision regularizer. Larger values make the initial
            covariance smaller and therefore more conservative.
        error_decay: EMA decay for absolute reward-prediction error diagnostics.
    """

    feature_dim: int
    forgetting: float = 0.995
    ridge: float = 10.0
    error_decay: float = 0.99

    def __post_init__(self) -> None:
        feature_dim = _require_int32("feature_dim", self.feature_dim, minimum=1)
        _preflight_state_resources(feature_dim)
        forgetting = _validated_config_float(
            "forgetting", self.forgetting, positive=True, upper=1.0
        )
        ridge = _validated_config_float("ridge", self.ridge, positive=True)
        minimum_ridge = float(np.finfo(np.float32).tiny)
        if ridge < minimum_ridge:
            raise ValueError(f"ridge must be at least {minimum_ridge} in float32")
        error_decay = _validated_config_float(
            "error_decay", self.error_decay, lower=0.0, upper=1.0, upper_inclusive=False
        )

        object.__setattr__(self, "feature_dim", feature_dim)
        object.__setattr__(self, "forgetting", forgetting)
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "error_decay", error_decay)

    def to_config(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "type": "RLSRewardModelConfig",
            "feature_dim": self.feature_dim,
            "forgetting": self.forgetting,
            "ridge": self.ridge,
            "error_decay": self.error_decay,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> RLSRewardModelConfig:
        """Reconstruct from :meth:`to_config` output."""
        if type(payload) is not dict:
            raise ValueError("payload must be an exact built-in dict")
        expected = {"type", "feature_dim", "forgetting", "ridge", "error_decay"}
        if any(type(key) is not str for key in payload) or set(payload) != expected:
            raise ValueError("payload fields do not match the serialized schema")
        if type(payload["type"]) is not str or payload["type"] != "RLSRewardModelConfig":
            raise ValueError("payload type differs")
        data = dict(payload)
        data.pop("type")
        return cls(**data)


@chex.dataclass(frozen=True)
class RLSRewardModelState:
    """State for :class:`RLSRewardModel`."""

    weights: Float[Array, " feature_dim"]
    covariance: Float[Array, "feature_dim feature_dim"]
    abs_error_ema: Float[Array, ""]
    step_count: Array


@chex.dataclass(frozen=True)
class RLSRewardModelUpdateResult:
    """Result from one reward-model update."""

    state: RLSRewardModelState
    prediction: Float[Array, ""]
    error: Float[Array, ""]
    gain: Float[Array, " feature_dim"]
    update_applied: Bool[Array, ""]


class RLSRewardModel:
    """Linear RLS scalar reward predictor.

    This model is intentionally narrow: it learns calibrated scalar reward
    predictions from caller-provided features. It is useful when imagined
    updates need reward targets but a shared multi-head dynamics model is too
    biased or too slow to calibrate.
    """

    def __init__(self, config: RLSRewardModelConfig):
        """Initialize the model."""
        if type(config) is not RLSRewardModelConfig:
            raise TypeError("config must be an RLSRewardModelConfig")
        self._config = config

    @property
    def config(self) -> RLSRewardModelConfig:
        """Model configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize the model configuration."""
        return {
            "type": "RLSRewardModel",
            "config": self._config.to_config(),
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> RLSRewardModel:
        """Reconstruct from :meth:`to_config` output."""
        if type(payload) is not dict:
            raise ValueError("payload must be an exact built-in dict")
        if any(type(key) is not str for key in payload) or set(payload) != {"type", "config"}:
            raise ValueError("payload fields do not match the serialized schema")
        if type(payload["type"]) is not str or payload["type"] != "RLSRewardModel":
            raise ValueError("payload type differs")
        if type(payload["config"]) is not dict:
            raise ValueError("payload config must be an exact built-in dict")
        return cls(RLSRewardModelConfig.from_config(payload["config"]))

    @functools.partial(jax.jit, static_argnums=(0,))
    def init(self) -> RLSRewardModelState:
        """Initialize model state."""
        feature_dim = self._config.feature_dim
        _preflight_update_working_set(feature_dim)
        return RLSRewardModelState(
            weights=jnp.zeros((feature_dim,), dtype=jnp.float32),
            covariance=(jnp.eye(feature_dim, dtype=jnp.float32) / self._config.ridge),
            abs_error_ema=jnp.array(0.0, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, state: RLSRewardModelState, features: Array) -> Array:
        """Predict reward from one feature vector."""
        x = _validated_feature_vector(features, feature_dim=self._config.feature_dim)
        return _finite_cancelling_dot(state.weights, x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: RLSRewardModelState,
        features: Array,
        reward: Array,
    ) -> RLSRewardModelUpdateResult:
        """Update from one real reward observation.

        Standard exponentially weighted RLS (Haykin, *Adaptive Filter
        Theory*):

        1. ``gain = P x / (forgetting + x^T P x)``
        2. ``w <- w + gain * error``
        3. ``P <- (P - gain (P x)^T) / forgetting``

        Caveat (covariance windup): with ``forgetting < 1``, any direction
        of feature space that receives no excitation has its covariance
        grown by ``1 / forgetting`` every step, so under persistently
        poorly excited features the covariance — and hence the gain when
        excitation returns — can grow without bound. Keep ``forgetting=1``
        unless the reward function is genuinely nonstationary and the
        feature stream stays exciting.
        """
        x = _validated_feature_vector(features, feature_dim=self._config.feature_dim)
        raw_target = jnp.asarray(reward)
        if raw_target.shape != ():
            raise ValueError("reward must be a scalar")
        if not (
            jnp.issubdtype(raw_target.dtype, jnp.floating)
            or jnp.issubdtype(raw_target.dtype, jnp.integer)
        ):
            raise TypeError("reward must be real numeric")
        target = raw_target.astype(jnp.float32)
        prediction = _finite_cancelling_dot(state.weights, x)
        error = target - prediction
        covariance_features = jax.vmap(
            lambda row: _finite_cancelling_dot(row, x)
        )(state.covariance)
        forgetting = jnp.asarray(self._config.forgetting, dtype=jnp.float32)
        denominator = forgetting + _finite_cancelling_dot(x, covariance_features)
        gain = covariance_features / denominator
        next_weights = state.weights + gain * error
        next_covariance = (state.covariance - jnp.outer(gain, covariance_features)) / forgetting
        next_covariance = _symmetrize_matrix(next_covariance)

        error_decay = jnp.asarray(self._config.error_decay, dtype=jnp.float32)
        abs_error = jnp.abs(error)
        next_abs_error_ema = jnp.where(
            state.step_count == 0,
            abs_error,
            _skip_zero_scale(error_decay, state.abs_error_ema) + (1.0 - error_decay) * abs_error,
        )
        next_state = RLSRewardModelState(
            weights=next_weights,
            covariance=next_covariance,
            abs_error_ema=next_abs_error_ema,
            step_count=_saturating_int32_increment(state.step_count),
        )
        # Inf reward * a silent feature's zero gain is 0*inf = NaN, and
        # that channel stays poisoned. Hold the previous finite state.
        checked_error_ema = (
            jnp.zeros_like(state.abs_error_ema)
            if self._config.error_decay == 0.0
            else state.abs_error_ema
        )
        source_valid = (
            jnp.all(jnp.isfinite(state.weights))
            & jnp.all(jnp.isfinite(state.covariance))
            & jnp.isfinite(checked_error_ema)
            & (state.step_count >= 0)
        )
        inputs_valid = jnp.all(jnp.isfinite(x)) & jnp.isfinite(jnp.squeeze(target))
        # Float32 rounding can leave the covariance indefinite; a nonpositive
        # innovation variance would flip the gain's sign.
        denominator_valid = jnp.isfinite(denominator) & (denominator > 0.0)
        proposed_finite = (
            jnp.all(jnp.isfinite(next_weights))
            & jnp.all(jnp.isfinite(next_covariance))
            & jnp.isfinite(next_abs_error_ema)
        )
        update_applied = (
            source_valid & inputs_valid & denominator_valid & proposed_finite
        )
        committed = jax.lax.cond(
            update_applied,
            lambda: next_state,
            lambda: state,
        )
        return RLSRewardModelUpdateResult(
            state=committed,
            prediction=jnp.where(update_applied, prediction, jnp.zeros_like(prediction)),
            error=jnp.where(update_applied, error, jnp.zeros_like(error)),
            gain=jnp.where(update_applied, gain, jnp.zeros_like(gain)),
            update_applied=update_applied,
        )

__all__ = [
    "RLSRewardModel",
    "RLSRewardModelConfig",
    "RLSRewardModelState",
    "RLSRewardModelUpdateResult",
]
