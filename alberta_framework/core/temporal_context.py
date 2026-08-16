# mypy: disable-error-code="call-arg,name-defined"
"""Causal temporal/context features for non-stationary Step 2 streams.

The featurizer augments each observation with cheap context blocks a
downstream linear or shallow learner can exploit under drift:

1. **EMA copy** — a slow exponential average of the observation stream, a
   causal summary of the recent input regime.
2. **Innovation** — the difference between the current observation and that
   EMA, isolating what just changed.
3. **Phase code** — ``sin``/``cos`` of the absolute step count at fixed
   periods (two features per period), a Fourier-style time encoding that lets
   even a linear readout represent target functions that vary periodically in
   time; the sin/cos pair covers arbitrary phase offsets.

Caveat: the phase code depends on the absolute step counter, not on the
observations, so it leaks global time into the feature vector — the same
observation maps to different features at different steps.  It helps only
when the stream's nonstationarity is genuinely periodic near the configured
periods; on aperiodic streams it is a spurious clock signal a downstream
learner can overfit.
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

from alberta_framework._float32 import round_real_to_float32_with_ratio

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


def _require_half_open_zero_one_interval(name: str, value: object) -> float:
    real, numerator, denominator, narrowed = finite_real_and_float32(name, value)
    if (
        real < 0.0
        or not real < 1.0
        or numerator < 0
        or numerator >= denominator
        or narrowed < 0.0
        or not narrowed < 1.0
    ):
        raise ValueError(f"{name} must be in [0, 1), got {value!r}")
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


@dataclass(frozen=True)
class TemporalContextConfig:
    """Configuration for :class:`TemporalContextFeaturizer`.

    The featurizer is causal: features at time ``t`` use the pre-update EMA and
    the current step counter, then the EMA is advanced after the observation is
    exposed.  This is meant for streams whose target changes with slowly moving
    latent context, such as rotating relevant subspaces.

    The default ``periods`` (50, 100, 200) span drift timescales of tens to a
    few hundred steps; set them to the stream's known drift periods when those
    are available.
    """

    input_dim: int
    include_raw: bool = True
    include_ema: bool = True
    include_delta: bool = True
    include_phase_products: bool = False
    ema_decay: float = 0.95
    periods: tuple[float, ...] = (50.0, 100.0, 200.0)

    def __post_init__(self) -> None:
        """Validate and canonicalize configuration."""
        _validate_config(self)

    def output_dim(self) -> int:
        """Return the transformed feature dimensionality."""
        copies = int(self.include_raw) + int(self.include_ema) + int(self.include_delta)
        phase_dim = 2 * len(self.periods)
        product_dim = phase_dim * self.input_dim * int(self.include_phase_products)
        return copies * self.input_dim + phase_dim + product_dim

    def to_config(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        payload = asdict(self)
        payload["periods"] = list(self.periods)
        payload["type"] = "TemporalContextConfig"
        return payload

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> TemporalContextConfig:
        """Reconstruct from :meth:`to_config` output."""
        payload = dict(config)
        payload.pop("type", None)
        if "periods" in payload:
            payload["periods"] = tuple(payload["periods"])
        return cls(**payload)


@chex.dataclass(frozen=True)
class TemporalContextState:
    """State for :class:`TemporalContextFeaturizer`."""

    observation_ema: Float[Array, " input_dim"]
    step_count: Array


def _validate_config(config: TemporalContextConfig) -> None:
    input_dim = _require_int(
        "input_dim", config.input_dim, minimum=1, maximum=_INT32_MAX
    )
    if type(config.include_raw) is not bool:
        raise ValueError(f"include_raw must be a bool, got {config.include_raw!r}")
    if type(config.include_ema) is not bool:
        raise ValueError(f"include_ema must be a bool, got {config.include_ema!r}")
    if type(config.include_delta) is not bool:
        raise ValueError(f"include_delta must be a bool, got {config.include_delta!r}")
    if type(config.include_phase_products) is not bool:
        raise ValueError(
            f"include_phase_products must be a bool, got {config.include_phase_products!r}"
        )
    if not (config.include_raw or config.include_ema or config.include_delta):
        raise ValueError("at least one observation feature block must be included")
    ema_decay = _require_half_open_zero_one_interval("ema_decay", config.ema_decay)
    if type(config.periods) is not tuple:
        raise ValueError(
            f"periods must be an actual tuple, got {type(config.periods).__name__}"
        )
    canonical_periods = tuple(
        _require_positive_real("period", p) for p in config.periods
    )
    object.__setattr__(config, "input_dim", input_dim)
    object.__setattr__(config, "include_raw", bool(config.include_raw))
    object.__setattr__(config, "include_ema", bool(config.include_ema))
    object.__setattr__(config, "include_delta", bool(config.include_delta))
    object.__setattr__(
        config, "include_phase_products", bool(config.include_phase_products)
    )
    object.__setattr__(config, "ema_decay", ema_decay)
    object.__setattr__(config, "periods", canonical_periods)


class TemporalContextFeaturizer:
    """Causal feature wrapper exposing EMA, innovation, and phase features."""

    def __init__(self, config: TemporalContextConfig):
        _validate_config(config)
        self._config = config

    @property
    def config(self) -> TemporalContextConfig:
        """Featurizer configuration."""
        return self._config

    def init(self) -> TemporalContextState:
        """Return an all-zero initial context state."""
        return TemporalContextState(
            observation_ema=jnp.zeros(self._config.input_dim, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def features(
        self,
        state: TemporalContextState,
        observation: Float[Array, " input_dim"],
    ) -> Float[Array, " output_dim"]:
        """Return current causal context features without advancing state."""
        cfg = self._config
        obs = jnp.asarray(observation, dtype=jnp.float32)
        coordinate_valid = jnp.isfinite(obs)
        safe_obs = jnp.where(coordinate_valid, obs, jnp.zeros_like(obs))
        blocks = []
        if cfg.include_raw:
            blocks.append(safe_obs)
        if cfg.include_ema:
            blocks.append(state.observation_ema)
        if cfg.include_delta:
            delta = obs - state.observation_ema
            blocks.append(jnp.where(coordinate_valid, delta, jnp.zeros_like(delta)))
        if cfg.periods:
            step = state.step_count.astype(jnp.float32)
            periods = jnp.asarray(cfg.periods, dtype=jnp.float32)
            angles = (2.0 * jnp.pi * step) / periods
            phase = jnp.ravel(jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=1))
            blocks.append(phase)
            if cfg.include_phase_products:
                blocks.append(jnp.ravel(phase[:, None] * safe_obs[None, :]))
        return jnp.concatenate(blocks, axis=0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: TemporalContextState,
        observation: Float[Array, " input_dim"],
    ) -> TemporalContextState:
        """Advance the context state after observing one input."""
        decay = jnp.asarray(self._config.ema_decay, dtype=jnp.float32)
        obs = jnp.asarray(observation, dtype=jnp.float32)
        observation_valid = jnp.all(jnp.isfinite(obs))
        decayed_ema = jnp.where(
            decay == 0.0,
            jnp.zeros_like(state.observation_ema),
            decay * state.observation_ema,
        )
        proposed = TemporalContextState(
            observation_ema=decayed_ema + (1.0 - decay) * obs,
            step_count=state.step_count + 1,
        )
        return cast(
            TemporalContextState,
            jax.lax.cond(observation_valid, lambda: proposed, lambda: state),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: TemporalContextState,
        observation: Float[Array, " input_dim"],
    ) -> tuple[TemporalContextState, Float[Array, " output_dim"]]:
        """Return features and then advance context state."""
        features = self.features(state, observation)
        next_state = self.update(state, observation)
        return next_state, features


def transform_temporal_context_arrays(
    featurizer: TemporalContextFeaturizer,
    observations: Float[Array, "steps input_dim"],
    *,
    state: TemporalContextState | None = None,
) -> tuple[TemporalContextState, Float[Array, "steps output_dim"]]:
    """Transform an observation array with a causal scan."""
    if state is None:
        state = featurizer.init()

    def step_fn(
        carry: TemporalContextState,
        observation: Array,
    ) -> tuple[TemporalContextState, Array]:
        return cast(tuple[TemporalContextState, Array], featurizer.step(carry, observation))

    return cast(
        tuple[TemporalContextState, Float[Array, "steps output_dim"]],
        jax.lax.scan(step_fn, state, observations),
    )
