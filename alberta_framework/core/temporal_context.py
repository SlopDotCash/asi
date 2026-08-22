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
import operator
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from numbers import Real
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

from alberta_framework._float32 import round_real_to_float32_with_ratio

_INT32_MAX: int = 2**31 - 1
# Public last-fit in tests is 3 array steps. Origin scanned the leading
# observation axis with no reject — hang/OOM, not an INT32 leftover.
_TEMPORAL_CONTEXT_LOOP_MAX_STEPS = 10_000

_ACTUAL_INT_TYPES = frozenset({int, *(np.dtype(code).type for code in "bBhHiIlLqQpP")})
_ACTUAL_FLOAT_TYPES = frozenset(
    {float, Fraction, *(np.dtype(code).type for code in ("e", "f", "d", "g"))}
)
_ALLOWED_REAL_TYPES = _ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES


def _copy_mapping(payload: object, *, name: str) -> dict[str, Any]:
    if not issubclass(type(payload), Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        values = dict(cast(Mapping[str, Any], payload))
    except Exception as error:
        raise ValueError(f"{name} must be a readable mapping") from error
    if any(type(key) is not str for key in values):
        raise ValueError(f"{name} keys must be exact strings")
    return values


def _require_exact_str(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def finite_real_and_float32(name: str, value: object) -> tuple[Real, int, int, float]:
    """Return the original real, exact ratio, and finite binary32 rounding."""
    if type(value) not in _ALLOWED_REAL_TYPES:
        raise ValueError(f"{name} must be a real number")
    real_obj = cast(Real, value)
    try:
        numerator, denominator, narrowed = round_real_to_float32_with_ratio(
            real_obj
        )
    except (FloatingPointError, OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must narrow to a finite float32") from None
    if not math.isfinite(narrowed):
        raise ValueError(f"{name} must narrow to a finite float32")
    return real_obj, numerator, denominator, narrowed


def canonical_float32_storage(value: object, narrowed: float) -> float:
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
        raise ValueError(f"{name} must be in [0, 1)")
    return canonical_float32_storage(real, narrowed)


def _require_positive_real(name: str, value: object) -> float:
    real, numerator, _, narrowed = finite_real_and_float32(name, value)
    if real <= 0.0 or numerator <= 0 or narrowed <= 0.0:
        raise ValueError(f"{name} must be positive")
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
            raise ValueError(f"{name} must be positive")
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative")
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _require_temporal_context_loop_steps(name: str, value: object) -> int:
    """Reject scan lengths above the public last-fit before ``jax.lax.scan``."""
    if type(value) is not int or value < 1 or value > _TEMPORAL_CONTEXT_LOOP_MAX_STEPS:
        raise ValueError(
            f"{name} must be an integer in [1, {_TEMPORAL_CONTEXT_LOOP_MAX_STEPS}]"
        )
    return value


def _require_temporal_context_array_steps(observations: object) -> int:
    """Reject pre-collected scan lengths above the public last-fit before scan."""
    if not isinstance(observations, jax.Array):
        raise TypeError("observations must be a JAX array")
    if observations.ndim != 2:
        raise ValueError("observations must be rank-2")
    num_steps = observations.shape[0]
    return _require_temporal_context_loop_steps("observations num_steps", num_steps)


def _temporal_context_persist_bytes(input_dim: int) -> int:
    """EMA float32 vector plus the signed-int32 step counter."""

    return 4 * input_dim + 4


def _temporal_context_output_dim(
    input_dim: int,
    *,
    include_raw: bool,
    include_ema: bool,
    include_delta: bool,
    include_phase_products: bool,
    n_periods: int,
) -> int:
    copies = int(include_raw) + int(include_ema) + int(include_delta)
    phase_dim = 2 * n_periods
    product_dim = phase_dim * input_dim * int(include_phase_products and n_periods > 0)
    return copies * input_dim + phase_dim + product_dim


def _temporal_context_update_working_set_bytes(
    input_dim: int,
    *,
    include_raw: bool,
    include_ema: bool,
    include_delta: bool,
    include_phase_products: bool,
    n_periods: int,
) -> int:
    """Source persist, proposed persist, committed persist, and returned extras.

    ``step`` keeps the source EMA/counter, the proposed EMA/counter, and the
    transaction-selected result live together with the observation, sanitized
    copies, optional delta pair, phase code, optional phase×obs products, and
    the returned feature leaf.
    """

    persist_bytes = _temporal_context_persist_bytes(input_dim)
    copies = int(include_raw) + int(include_ema) + int(include_delta)
    phase_dim = 2 * n_periods
    product_dim = phase_dim * input_dim * int(include_phase_products and n_periods > 0)
    output_dim = copies * input_dim + phase_dim + product_dim
    extra_bytes = (
        4 * input_dim
        + 4 * input_dim
        + 4 * input_dim
        + 4 * input_dim
        + 8 * input_dim * int(include_delta)
        + 4 * phase_dim
        + 4 * product_dim
        + 4 * output_dim
        + 16
    )
    return 3 * persist_bytes + extra_bytes


def _preflight_temporal_context_update_working_set(
    input_dim: int,
    *,
    include_raw: bool,
    include_ema: bool,
    include_delta: bool,
    include_phase_products: bool,
    n_periods: int,
) -> None:
    """Reject a step envelope the host cannot name in signed int32."""

    persist_bytes = _temporal_context_persist_bytes(input_dim)
    if persist_bytes > _INT32_MAX:
        raise ValueError(
            "temporal-context persistent state byte count must fit signed int32"
        )
    output_dim = _temporal_context_output_dim(
        input_dim,
        include_raw=include_raw,
        include_ema=include_ema,
        include_delta=include_delta,
        include_phase_products=include_phase_products,
        n_periods=n_periods,
    )
    if output_dim > _INT32_MAX or 4 * output_dim > _INT32_MAX:
        raise ValueError(
            "temporal-context returned feature byte count must fit signed int32"
        )
    working_set_bytes = _temporal_context_update_working_set_bytes(
        input_dim,
        include_raw=include_raw,
        include_ema=include_ema,
        include_delta=include_delta,
        include_phase_products=include_phase_products,
        n_periods=n_periods,
    )
    if working_set_bytes > _INT32_MAX:
        raise ValueError(
            "temporal-context update working set byte count must fit signed int32"
        )


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
        return _temporal_context_output_dim(
            self.input_dim,
            include_raw=self.include_raw,
            include_ema=self.include_ema,
            include_delta=self.include_delta,
            include_phase_products=self.include_phase_products,
            n_periods=len(self.periods),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        payload = asdict(self)
        payload["periods"] = list(self.periods)
        payload["type"] = "TemporalContextConfig"
        return payload

    @classmethod
    def from_config(cls, config: object) -> TemporalContextConfig:
        """Reconstruct from :meth:`to_config` output."""
        payload = _copy_mapping(config, name="TemporalContextConfig")
        payload.pop("type", None)
        if "periods" in payload:
            raw = payload["periods"]
            if type(raw) not in (list, tuple):
                raise ValueError("periods must be a list or tuple")
            payload["periods"] = tuple(raw)
        return cls(**payload)


@chex.dataclass(frozen=True)
class TemporalContextState:
    """State for :class:`TemporalContextFeaturizer`."""

    observation_ema: Float[Array, " input_dim"]
    step_count: Array


def _require_array_metadata(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: object,
) -> Array:
    """Require trusted JAX metadata before any coercion or arithmetic."""
    actual_type = type(value)
    if not (
        issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
    ):
        raise ValueError(f"{name} must be a JAX array")
    array = cast(Array, value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if array.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    return array


def _require_state_metadata(
    state: object,
    *,
    input_dim: int,
) -> TemporalContextState:
    """Require the exact public state schema without reading hostile objects."""
    if type(state) is not TemporalContextState:
        raise ValueError("state must be an exact TemporalContextState")
    trusted = state
    _require_array_metadata(
        "state.observation_ema",
        trusted.observation_ema,
        shape=(input_dim,),
        dtype=jnp.float32,
    )
    _require_array_metadata(
        "state.step_count",
        trusted.step_count,
        shape=(),
        dtype=jnp.int32,
    )
    return trusted


def _require_observation_batch_metadata(value: object, *, input_dim: int) -> Array:
    """Require one trusted float32 batch without touching arbitrary hooks."""
    actual_type = type(value)
    if not (
        issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
    ):
        raise ValueError("observations must be a JAX array")
    observations = cast(Array, value)
    if observations.ndim != 2 or observations.shape[1] != input_dim:
        raise ValueError(f"observations must have shape (steps, {input_dim})")
    if observations.dtype != jnp.float32:
        raise ValueError("observations must have dtype float32")
    return observations


def _validate_config(config: TemporalContextConfig) -> None:
    input_dim = _require_int(
        "input_dim", config.input_dim, minimum=1, maximum=_INT32_MAX
    )
    if type(config.include_raw) is not bool:
        raise ValueError("include_raw must be a bool")
    if type(config.include_ema) is not bool:
        raise ValueError("include_ema must be a bool")
    if type(config.include_delta) is not bool:
        raise ValueError("include_delta must be a bool")
    if type(config.include_phase_products) is not bool:
        raise ValueError(
            "include_phase_products must be a bool"
        )
    if not (config.include_raw or config.include_ema or config.include_delta):
        raise ValueError("at least one observation feature block must be included")
    ema_decay = _require_half_open_zero_one_interval("ema_decay", config.ema_decay)
    if type(config.periods) is not tuple:
        raise ValueError(
            "periods must be an actual tuple"
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
    _preflight_temporal_context_update_working_set(
        input_dim,
        include_raw=config.include_raw,
        include_ema=config.include_ema,
        include_delta=config.include_delta,
        include_phase_products=config.include_phase_products,
        n_periods=len(canonical_periods),
    )


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

    def features(
        self,
        state: TemporalContextState,
        observation: Float[Array, " input_dim"],
    ) -> Float[Array, " output_dim"]:
        """Return current causal context features without advancing state."""
        cfg = self._config
        state = _require_state_metadata(state, input_dim=cfg.input_dim)
        obs = _require_array_metadata(
            "observation",
            observation,
            shape=(cfg.input_dim,),
            dtype=jnp.float32,
        )
        return cast(Array, self._features_jit(state, obs))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _features_jit(
        self,
        state: TemporalContextState,
        obs: Array,
    ) -> Array:
        """Compiled feature implementation after public metadata gates."""
        cfg = self._config
        coordinate_valid = jnp.isfinite(obs)
        safe_obs = jnp.where(coordinate_valid, obs, jnp.zeros_like(obs))
        ema_valid = jnp.all(jnp.isfinite(state.observation_ema))
        safe_ema = jnp.where(
            ema_valid,
            state.observation_ema,
            jnp.zeros_like(state.observation_ema),
        )
        blocks = []
        if cfg.include_raw:
            blocks.append(safe_obs)
        if cfg.include_ema:
            blocks.append(safe_ema)
        if cfg.include_delta:
            delta = obs - safe_ema
            blocks.append(jnp.where(coordinate_valid, delta, jnp.zeros_like(delta)))
        if cfg.periods:
            counter_valid = state.step_count >= jnp.asarray(0, dtype=jnp.int32)
            safe_step_count = jnp.where(
                counter_valid,
                state.step_count,
                jnp.asarray(0, dtype=jnp.int32),
            )
            step = safe_step_count.astype(jnp.float32)
            periods = jnp.asarray(cfg.periods, dtype=jnp.float32)
            angles = (2.0 * jnp.pi * step) / periods
            phase = jnp.ravel(jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=1))
            blocks.append(phase)
            if cfg.include_phase_products:
                blocks.append(jnp.ravel(phase[:, None] * safe_obs[None, :]))
        return jnp.concatenate(blocks, axis=0)

    def update(
        self,
        state: TemporalContextState,
        observation: Float[Array, " input_dim"],
    ) -> TemporalContextState:
        """Advance the context state after observing one input.

        ``step_count`` is a signed-int32 lifetime coordinate. At exhaustion it
        remains at ``INT32_MAX``: EMA learning continues, while phase features
        deliberately stay at the final representable coordinate. Invalid
        observations, counters, or proposed EMA values commit no state field.
        """
        state = _require_state_metadata(state, input_dim=self._config.input_dim)
        obs = _require_array_metadata(
            "observation",
            observation,
            shape=(self._config.input_dim,),
            dtype=jnp.float32,
        )
        return cast(TemporalContextState, self._update_jit(state, obs))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_jit(
        self,
        state: TemporalContextState,
        obs: Array,
    ) -> TemporalContextState:
        """Compiled atomic update after public metadata gates."""
        decay = jnp.asarray(self._config.ema_decay, dtype=jnp.float32)
        observation_valid = jnp.all(jnp.isfinite(obs))
        counter_valid = state.step_count >= jnp.asarray(0, dtype=jnp.int32)
        decayed_ema = jnp.where(
            decay == 0.0,
            jnp.zeros_like(state.observation_ema),
            decay * state.observation_ema,
        )
        proposed_ema = decayed_ema + (1.0 - decay) * obs
        proposed = TemporalContextState(
            observation_ema=proposed_ema,
            step_count=jnp.minimum(
                state.step_count,
                jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32),
            )
            + jnp.asarray(1, dtype=jnp.int32),
        )
        return cast(
            TemporalContextState,
            jax.lax.cond(
                observation_valid
                & counter_valid
                & jnp.all(jnp.isfinite(proposed_ema)),
                lambda: proposed,
                lambda: state,
            ),
        )

    def step(
        self,
        state: TemporalContextState,
        observation: Float[Array, " input_dim"],
    ) -> tuple[TemporalContextState, Float[Array, " output_dim"]]:
        """Return features and then advance context state."""
        state = _require_state_metadata(state, input_dim=self._config.input_dim)
        obs = _require_array_metadata(
            "observation",
            observation,
            shape=(self._config.input_dim,),
            dtype=jnp.float32,
        )
        return cast(
            tuple[TemporalContextState, Float[Array, " output_dim"]],
            self._step_jit(state, obs),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_jit(
        self,
        state: TemporalContextState,
        obs: Array,
    ) -> tuple[TemporalContextState, Array]:
        """Compiled combined path after public metadata gates."""
        features = self._features_jit(state, obs)
        next_state = self._update_jit(state, obs)
        return next_state, features


def transform_temporal_context_arrays(
    featurizer: TemporalContextFeaturizer,
    observations: Float[Array, "steps input_dim"],
    *,
    state: TemporalContextState | None = None,
) -> tuple[TemporalContextState, Float[Array, "steps output_dim"]]:
    """Transform an observation array with a causal scan.

    Raises:
        ValueError: If ``observations`` length is not an exact integer in
            ``[1, 10_000]``.
    """
    observations = _require_observation_batch_metadata(
        observations,
        input_dim=featurizer.config.input_dim,
    )
    _require_temporal_context_array_steps(observations)
    if state is None:
        state = featurizer.init()

    def step_fn(
        carry: TemporalContextState,
        observation: Array,
    ) -> tuple[TemporalContextState, Array]:
        return featurizer.step(carry, observation)

    return cast(
        tuple[TemporalContextState, Float[Array, "steps output_dim"]],
        jax.lax.scan(step_fn, state, observations),
    )
