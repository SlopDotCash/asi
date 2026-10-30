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
        if float(np.float32(self.gamma * self.lamda)) >= 1.0:
            raise ValueError(
                "gamma * lamda must remain less than 1 in float32 for sigma correction"
            )
        if not 0 <= self.beta2 < 1 or not 0 <= self.beta_clip < 1:
            raise ValueError("EMA decays must be in [0, 1)")
        for name in ("beta2", "beta_clip"):
            if float(np.float32(getattr(self, name))) >= 1.0:
                raise ValueError(f"{name} must remain less than 1 in float32")
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
    trace = decay * state.trace + gradient
    step = state.step + jnp.array(1, dtype=jnp.int32)
    if not config.enabled:
        changed = weights + (config.eta * td_error) * trace
        return changed, state.replace(  # type: ignore[attr-defined]
            trace=jnp.where(terminated, jnp.zeros_like(trace), trace), step=step
        )

    second = config.beta2 * state.squared_gradient + (1 - config.beta2) * gradient**2
    count = step.astype(jnp.float32)
    divisor = (
        jnp.sqrt(second / (1 - config.beta2**count)) + 1e-8
        if config.use_rmsprop
        else jnp.ones_like(weights)
    )
    norm_gradient = jnp.sum(gradient**2 / divisor)
    sigma = state.sigma + (1 - decay) * (norm_gradient - state.sigma)
    sigma_corrected = sigma / (1 - decay**count)
    norm_trace = jnp.sum(trace**2 / divisor)
    step_size = config.eta / jnp.maximum(jnp.sqrt(sigma_corrected * norm_trace), 1e-8)
    clip_second = config.beta_clip * state.clip_squared_error + (1 - config.beta_clip) * td_error**2
    cap = (
        config.clip_mult * jnp.sqrt(clip_second / (1 - config.beta_clip**count))
        if config.use_adaptive_clip
        else jnp.array(1.0, dtype=jnp.float32)
    )
    safe_error = jnp.clip(td_error, -cap, cap)
    changed = weights + (safe_error * step_size) * (trace / divisor)
    return changed, IntentionalTDState(  # type: ignore[call-arg]
        trace=jnp.where(terminated, jnp.zeros_like(trace), trace),
        squared_gradient=second,
        sigma=sigma,
        clip_squared_error=clip_second,
        step=step,
    )
