"""Intentional value updates for a caller-supplied prediction gradient.

Port of ``IntentionalOptimizerValue`` at Intentional_RL commit
e86e26fd8613ac212e9a52c3fed8a01d0a31f685 (arXiv:2604.19033v1).
The caller computes the semi-gradient and the frozen TD target. This kernel
normalizes globally over one parameter array, including its bias coordinates.
The development consumer uses linear action values; this is not a reproduction
of the paper's deep-RL results or a reference-life adapter.
"""

import dataclasses
import math

import chex
import jax.numpy as jnp
import numpy as np
from jax import Array

from alberta_framework.core._float32_scalars import validated_float32_scalar

_INT32_MAX = 2**31 - 1


def _stable_weighted_square_ema(previous: Array, value: Array, decay: float) -> Array:
    """Preserve a finite EMA when only the unweighted square overflows."""

    increment = 1.0 - decay
    direct_innovation = increment * value**2
    scaled_value = math.sqrt(increment) * value
    innovation = jnp.where(jnp.isfinite(direct_innovation), direct_innovation, scaled_value**2)
    if float(np.float32(decay)) == 1.0:
        # Apply the exact configured decrement together with the innovation.
        # Multiplying by the narrowed decay would discard it on every step.
        return previous + (innovation - increment * previous)
    return decay * previous + innovation


@dataclasses.dataclass(frozen=True)
class IntentionalTDConfig:
    """Value-optimizer controls; defaults match the pinned reference code.

    ``enabled=False`` is a fixed-step accumulating-trace TD control, without
    diagonal normalization, adaptive step selection, or delta clipping.
    ``use_adaptive_clip=False`` means reference-code clipping to [-1, 1].
    RMS and sigma statistics survive episode boundaries; only traces reset.
    """

    eta: float = 0.5
    gamma: float = 0.99
    lamda: float = 0.0
    beta2: float = 0.999
    beta_clip: float = 0.9998
    clip_mult: float = 20.0
    use_rmsprop: bool = True
    use_adaptive_clip: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("eta", "gamma", "lamda", "beta2", "beta_clip", "clip_mult"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        for name in ("eta", "clip_mult"):
            object.__setattr__(
                self,
                name,
                validated_float32_scalar(name, getattr(self, name), positive=True),
            )
        if self.eta <= 0 or self.clip_mult <= 0:
            raise ValueError("eta and clip_mult must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.lamda <= 1:
            raise ValueError("gamma and lamda must be in [0, 1]")
        if self.gamma * self.lamda >= 1:
            raise ValueError("gamma * lamda must be less than 1 for sigma correction")
        if not 0 <= self.beta2 < 1 or not 0 <= self.beta_clip < 1:
            raise ValueError("EMA decays must be in [0, 1)")
        for name in ("use_rmsprop", "use_adaptive_clip", "enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")


@chex.dataclass(frozen=True)
class IntentionalTDState:
    """Fixed-size optimizer state, with no retained transition replay."""

    trace: Array
    squared_gradient: Array
    sigma: Array
    clip_squared_error: Array
    step: Array


def init_intentional_td(weights: Array) -> IntentionalTDState:
    """Allocate optimizer state matching a float32 parameter array."""
    if weights.dtype != jnp.float32 or weights.size == 0:
        raise ValueError("weights must be a nonempty float32 array")
    return IntentionalTDState(  # type: ignore[call-arg]
        trace=jnp.zeros_like(weights),
        squared_gradient=jnp.zeros_like(weights),
        sigma=jnp.array(0.0, dtype=jnp.float32),
        clip_squared_error=jnp.array(0.0, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )


def _bias_correction(decay: float, count: Array) -> Array:
    """Compute 1 - decay**count without rounding a near-one decay to one."""
    if decay == 0.0:
        return jnp.ones_like(count)
    return -jnp.expm1(count * math.log(decay))


def _scaled_trace_step(scale: float | Array, error: Array, trace_term: Array) -> Array:
    """Compute (scale * error) * trace_term without overflow or 0 * inf NaN."""
    scale_arr = jnp.asarray(scale, dtype=trace_term.dtype)
    direct = (scale_arr * error) * trace_term
    m_s, e_s = jnp.frexp(scale_arr)
    m_e, e_e = jnp.frexp(error)
    m_t, e_t = jnp.frexp(trace_term)
    rescaled = jnp.ldexp((m_s * m_e) * m_t, e_s + e_e + e_t)
    step = jnp.where(jnp.isfinite(direct), direct, rescaled)
    return jnp.where(trace_term == 0.0, jnp.zeros_like(trace_term), step)


def intentional_td_update(
    weights: Array,
    state: IntentionalTDState,
    gradient: Array,
    td_error: Array,
    config: IntentionalTDConfig,
    terminated: bool | Array = False,
) -> tuple[Array, IntentionalTDState]:
    """Apply one value update, then clear the trace if the episode ended.

    ``gradient`` is dV/dw (or dQ/dw), not a loss gradient multiplied by the
    error. The caller must zero the bootstrap discount at a true terminal.
    As in the official code, sigma is a bias-corrected discounted *mean*,
    whereas paper Eq. 12 describes an unnormalized sum. That distinction is
    retained explicitly; this ports the implementation, not literal Eq. 12.
    All dynamic arrays are float32 except the int32 update counter.
    """
    if gradient.shape != weights.shape or gradient.dtype != weights.dtype:
        raise ValueError("gradient must match weights")
    decay = config.gamma * config.lamda
    if float(np.float32(decay)) == 1.0:
        decrement = (1.0 - decay) * state.trace
        # Opposite signs cancel safely before the small decrement. With the
        # same sign, subtract the decrement before the potentially large sum.
        opposite_sign = (state.trace < 0.0) != (gradient < 0.0)
        trace = jnp.where(
            opposite_sign,
            (state.trace + gradient) - decrement,
            state.trace + (gradient - decrement),
        )
    else:
        trace = decay * state.trace + gradient
    maximum_step = jnp.array(_INT32_MAX, dtype=jnp.int32)
    step = jnp.minimum(state.step, maximum_step - 1) + jnp.array(
        1, dtype=jnp.int32
    )
    if not config.enabled:
        changed = weights + _scaled_trace_step(config.eta, td_error, trace)
        return changed, state.replace(  # type: ignore[attr-defined]
            trace=jnp.where(terminated, jnp.zeros_like(trace), trace), step=step
        )

    gradient_squared = gradient**2
    # Weight before squaring when the direct square overflows.  The weighted
    # moment can still be representable even when a finite float32 value
    # squared is not.
    second = _stable_weighted_square_ema(
        state.squared_gradient,
        gradient,
        config.beta2,
    )
    count = step.astype(jnp.float32)
    if config.use_rmsprop:
        second_bias = _bias_correction(config.beta2, count)
        direct_rms = jnp.sqrt(second / second_bias)
        stable_rms = jnp.sqrt(second) / jnp.sqrt(second_bias)
        divisor = jnp.where(jnp.isfinite(direct_rms), direct_rms, stable_rms) + 1e-8
    else:
        divisor = jnp.ones_like(weights)
    direct_norm_gradient = jnp.sum(gradient_squared / divisor)
    stable_norm_gradient = jnp.sum((gradient / divisor) * gradient)
    norm_gradient = jnp.where(
        jnp.isfinite(direct_norm_gradient), direct_norm_gradient, stable_norm_gradient
    )
    sigma_increment = 1.0 - decay
    direct_sigma_delta = sigma_increment * (norm_gradient - state.sigma)
    scaled_gradient = math.sqrt(sigma_increment) * gradient
    stable_sigma_innovation = jnp.sum((scaled_gradient / divisor) * scaled_gradient)
    fallback_sigma = (
        state.sigma + (stable_sigma_innovation - sigma_increment * state.sigma)
        if float(np.float32(decay)) == 1.0
        else decay * state.sigma + stable_sigma_innovation
    )
    sigma = jnp.where(
        jnp.isfinite(direct_sigma_delta),
        state.sigma + direct_sigma_delta,
        fallback_sigma,
    )
    sigma_bias = _bias_correction(decay, count)
    sigma_corrected = sigma / sigma_bias
    direct_sqrt_sigma = jnp.sqrt(sigma_corrected)
    stable_sqrt_sigma = jnp.sqrt(sigma) / jnp.sqrt(sigma_bias)
    sqrt_sigma = jnp.where(jnp.isfinite(direct_sqrt_sigma), direct_sqrt_sigma, stable_sqrt_sigma)
    direct_norm_trace = jnp.sum(trace**2 / divisor)
    stable_norm_trace = jnp.sum((trace / divisor) * trace)
    norm_trace = jnp.where(jnp.isfinite(direct_norm_trace), direct_norm_trace, stable_norm_trace)
    trace_scale = jnp.max(jnp.abs(trace))
    safe_trace_scale = jnp.where(trace_scale > 0.0, trace_scale, jnp.ones_like(trace_scale))
    unit_trace = trace / safe_trace_scale
    scaled_sqrt_norm_trace = trace_scale * jnp.sqrt(
        jnp.sum((unit_trace / divisor) * unit_trace)
    )
    sqrt_norm_trace = jnp.where(
        jnp.isfinite(norm_trace), jnp.sqrt(norm_trace), scaled_sqrt_norm_trace
    )
    direct_normalizer = jnp.sqrt(sigma_corrected * norm_trace)
    stable_normalizer = sqrt_sigma * sqrt_norm_trace
    normalizer = jnp.where(jnp.isfinite(direct_normalizer), direct_normalizer, stable_normalizer)
    step_size = config.eta / jnp.maximum(normalizer, 1e-8)
    clip_second = _stable_weighted_square_ema(
        state.clip_squared_error,
        td_error,
        config.beta_clip,
    )
    if config.use_adaptive_clip:
        clip_bias = _bias_correction(config.beta_clip, count)
        direct_cap = config.clip_mult * jnp.sqrt(clip_second / clip_bias)
        stable_cap = config.clip_mult * jnp.sqrt(clip_second) / jnp.sqrt(clip_bias)
        cap = jnp.where(jnp.isfinite(direct_cap), direct_cap, stable_cap)
    else:
        cap = jnp.array(1.0, dtype=jnp.float32)
    safe_error = jnp.clip(td_error, -cap, cap)
    scaled_trace = trace / divisor
    direct_delta = _scaled_trace_step(step_size, safe_error, scaled_trace)
    factored_delta = _scaled_trace_step(
        config.eta / jnp.maximum(sqrt_sigma, 1e-4),
        safe_error,
        scaled_trace / jnp.maximum(sqrt_norm_trace, 1e-4),
    )
    changed = weights + jnp.where(jnp.isfinite(normalizer), direct_delta, factored_delta)
    return changed, IntentionalTDState(  # type: ignore[call-arg]
        trace=jnp.where(terminated, jnp.zeros_like(trace), trace),
        squared_gradient=second,
        sigma=sigma,
        clip_squared_error=clip_second,
        step=step,
    )
