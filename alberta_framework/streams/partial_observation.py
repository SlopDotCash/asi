"""Partial-observation stream wrapper for POMDP testbeds.

Wraps any ``ScanStream`` so that a configurable subset of observation
channels is masked at each step. Useful for testing whether agents can
recover the missing information from history (e.g. via
``HistoryFeatureExtractor`` in ``core/history_features.py``).

Three masking modes are supported:
    - ``MaskMode.FIXED``: same channels are masked every step.
    - ``MaskMode.RANDOM``: each step samples an i.i.d. random mask with
      Bernoulli probability ``mask_prob`` per channel.
    - ``MaskMode.PERIODIC``: the mask cycles through a hand-specified
      schedule of mask vectors with period ``len(schedule)``.

The wrapped TimeStep retains the SAME ``feature_dim`` -- masked positions
are replaced with a sentinel value (default 0.0). The ``target`` is
unchanged. This means the agent's job is unchanged dimensionally; it
just receives less information about the underlying state.
"""

from __future__ import annotations

import enum
from typing import TypeVar, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, PRNGKeyArray

from alberta_framework.core._float32_scalars import validated_float32_scalar
from alberta_framework.core.types import TimeStep
from alberta_framework.streams.base import ScanStream

InnerStateT = TypeVar("InnerStateT")


# =============================================================================
# Modes
# =============================================================================


class MaskMode(enum.Enum):
    """Channel-masking mode for ``PartialObservationWrapper``."""

    FIXED = "fixed"
    RANDOM = "random"
    PERIODIC = "periodic"


# Bound schedule construction by both row count and total boolean payload.
_MAX_PERIODIC_SCHEDULE_LENGTH = 4_096
# Cap the final boolean schedule payload at 64 MiB. The schedule is converted
# in one operation below: converting each row separately makes construction
# time grow with thousands of JAX dispatches even when every row is a repeated
# pointer to the same tiny host array.
_MAX_PERIODIC_SCHEDULE_VALUES = 64 * 1024 * 1024


def _trusted_boolean_mask(name: str, value: object, feature_dim: int) -> Array:
    """Validate mask metadata before JAX conversion can dispatch user hooks."""
    actual_type = type(value)
    if actual_type is not np.ndarray and not issubclass(
        actual_type, (jax.Array, jax.core.Tracer)
    ):
        raise ValueError(f"{name} must be an exact NumPy or JAX array")
    trusted = cast(Array, value)
    if trusted.shape != (feature_dim,) or trusted.dtype != jnp.dtype(jnp.bool_):
        raise ValueError(
            f"{name} must have shape (feature_dim={feature_dim},) and dtype bool"
        )
    return trusted


def _require_unit_interval_probability(name: str, value: object) -> float:
    """Return a canonical probability valid at the float32 execution sink."""
    return validated_float32_scalar(name, value, lower=0.0, upper=1.0)


def _require_mode(mode: object) -> MaskMode:
    """Reject leftover string, bool, and int identities for ``mode``.

    Leftover ``"FIXED"`` / ``"fixed"`` compare unequal to
    ``MaskMode.FIXED``, so the wrapper stored the leftover, skipped the
    fixed mask, and fell through to the periodic branch (fail-open).
    """
    if type(mode) is not MaskMode:
        raise ValueError("mode must be an exact MaskMode")
    return mode


def _require_feature_dim(value: object) -> int:
    """Reject non-builtin protocol dimensions before resource arithmetic."""
    if type(value) is not int or not 1 <= value <= 2**31 - 1:
        raise ValueError("inner.feature_dim must be an integer in [1, 2147483647]")
    return value


def _require_typed_threefry_key(name: str, value: object) -> Array:
    """Reject legacy or implementation-ambiguous keys before stream dispatch."""
    actual_type = type(value)
    if not (
        issubclass(actual_type, jax.Array)
        or issubclass(actual_type, jax.core.Tracer)
    ):
        raise ValueError(f"{name} must be a typed scalar threefry2x32 key")
    key = cast(Array, value)
    try:
        words = jr.key_data(key)
        implementation = str(jr.key_impl(key))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a typed scalar threefry2x32 key"
        ) from error
    if (
        key.shape != ()
        or not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)  # type: ignore[attr-defined]
        or words.shape != (2,)
        or words.dtype != jnp.uint32
        or implementation != "threefry2x32"
    ):
        raise ValueError(f"{name} must be a typed scalar threefry2x32 key")
    return key


# =============================================================================
# State
# =============================================================================


@chex.dataclass(frozen=True)
class PartialObservationState[InnerStateT]:
    """State for the partial-observation wrapper.

    Attributes:
        inner_state: The wrapped stream's state.
        key: JAX random key used by ``MaskMode.RANDOM``.
        period_index: Step counter modulo schedule length for
            ``MaskMode.PERIODIC``.
    """

    inner_state: InnerStateT
    key: PRNGKeyArray
    period_index: Array


# =============================================================================
# Wrapper
# =============================================================================


class PartialObservationWrapper[InnerStateT]:
    """ScanStream wrapper that masks observation channels.

    Args:
        inner: The underlying ``ScanStream`` whose observations will be
            partially masked.
        mode: Exact ``MaskMode`` member (FIXED / RANDOM / PERIODIC).
            Leftover string, bool, and int identities are rejected.
        fixed_mask: Exact NumPy or JAX boolean array of shape
            ``(feature_dim,)`` for FIXED. ``True`` means VISIBLE;
            ``False`` means HIDDEN.
        mask_prob: Per-channel KEEP probability for RANDOM. So
            ``mask_prob = 0.5`` keeps half the channels each step in
            expectation.
        schedule: Non-empty exact tuple of at most 4,096 NumPy or JAX
            boolean arrays of shape ``(feature_dim,)`` and at most
            67,108,864 total values; cycled under PERIODIC mode.
        sentinel: Finite real that replaces masked entries (default ``0.0``).
            Boolean and non-finite values are rejected so a hidden channel
            cannot silently become ``1.0``, ``NaN``, or ``Inf``.

    Examples
    --------
    ```python
    inner = RandomWalkStream(feature_dim=10, drift_rate=0.0)
    fixed = jnp.array([True] * 5 + [False] * 5)  # hide last 5 channels
    wrapper = PartialObservationWrapper(
        inner, mode=MaskMode.FIXED, fixed_mask=fixed
    )
    state = wrapper.init(jr.key(0))
    timestep, state = wrapper.step(state, jnp.array(0))
    # timestep.observation has zeros in positions 5..9
    ```
    """

    def __init__(
        self,
        inner: ScanStream[InnerStateT],
        mode: MaskMode = MaskMode.FIXED,
        fixed_mask: Bool[Array, " feature_dim"] | None = None,
        mask_prob: float = 0.5,
        schedule: tuple[Bool[Array, " feature_dim"], ...] | None = None,
        sentinel: float = 0.0,
    ):
        mode = _require_mode(mode)
        self._inner = inner
        self._mode = mode
        self._mask_prob = mask_prob
        self._sentinel = validated_float32_scalar("sentinel", sentinel)

        feature_dim = _require_feature_dim(inner.feature_dim)

        if mode == MaskMode.FIXED:
            if fixed_mask is None:
                raise ValueError("MaskMode.FIXED requires fixed_mask.")
            mask = jnp.asarray(
                _trusted_boolean_mask("fixed_mask shape", fixed_mask, feature_dim)
            )
            self._fixed_mask: Array | None = mask
        else:
            self._fixed_mask = None

        if mode == MaskMode.PERIODIC:
            if type(schedule) is not tuple:
                raise ValueError("periodic schedule must be an exact tuple")
            if len(schedule) == 0:
                raise ValueError(
                    "MaskMode.PERIODIC requires a non-empty schedule."
                )
            if len(schedule) > _MAX_PERIODIC_SCHEDULE_LENGTH:
                raise ValueError(
                    "periodic schedule length must be an integer in "
                    f"[1, {_MAX_PERIODIC_SCHEDULE_LENGTH}]"
                )
            schedule_values = len(schedule) * feature_dim
            if schedule_values > _MAX_PERIODIC_SCHEDULE_VALUES:
                raise ValueError(
                    "periodic schedule working set exceeds the bounded "
                    f"{_MAX_PERIODIC_SCHEDULE_VALUES}-value payload"
                )
            for row in schedule:
                _trusted_boolean_mask("schedule masks", row, feature_dim)
            # Convert the complete trusted tuple once.  Per-row conversion plus
            # stacking turns the admitted 4,096-row boundary into thousands of
            # independent JAX dispatches even for pointer-repeated rows.
            sched = jnp.asarray(schedule)
            assert sched.shape == (len(schedule), feature_dim)
            assert sched.dtype == jnp.dtype(jnp.bool_)
            self._schedule: Array | None = sched
        else:
            self._schedule = None

        if mode == MaskMode.RANDOM:
            self._mask_prob = _require_unit_interval_probability("mask_prob", mask_prob)

    @property
    def feature_dim(self) -> int:
        """Same as the wrapped stream."""
        return self._inner.feature_dim

    @property
    def mode(self) -> MaskMode:
        """Masking mode."""
        return self._mode

    def init(self, key: Array) -> PartialObservationState[InnerStateT]:
        """Initialize wrapper state."""
        key = _require_typed_threefry_key("key", key)
        k_inner, k_mask = jr.split(key)
        inner_state = self._inner.init(k_inner)
        return PartialObservationState(  # type: ignore[call-arg]
            inner_state=inner_state,
            key=k_mask,
            period_index=jnp.array(0, dtype=jnp.int32),
        )

    def step(
        self,
        state: PartialObservationState[InnerStateT],
        idx: Array,
    ) -> tuple[TimeStep, PartialObservationState[InnerStateT]]:
        """Step the wrapped stream and apply the channel mask."""
        timestep, new_inner = self._inner.step(state.inner_state, idx)

        new_key = state.key
        new_period_index = state.period_index

        if self._mode == MaskMode.FIXED:
            assert self._fixed_mask is not None
            mask = self._fixed_mask
        elif self._mode == MaskMode.RANDOM:
            new_key, k_use = jr.split(state.key)
            mask = jr.uniform(k_use, (self._inner.feature_dim,)) < self._mask_prob
        else:  # PERIODIC
            assert self._schedule is not None
            schedule_length = jnp.asarray(self._schedule.shape[0], dtype=jnp.int32)
            # Persist the bounded phase itself: a saturated lifetime counter
            # would pin one mask forever once it reached int32 max.
            phase = jnp.mod(state.period_index, schedule_length)
            mask = self._schedule[phase]
            new_period_index = jnp.mod(phase + jnp.int32(1), schedule_length)

        masked_obs = jnp.where(
            mask,
            timestep.observation,
            jnp.full_like(timestep.observation, self._sentinel),
        )

        new_timestep = TimeStep(  # type: ignore[call-arg]
            observation=masked_obs,
            target=timestep.target,
        )
        new_state: PartialObservationState[InnerStateT] = PartialObservationState(
            inner_state=new_inner,
            key=new_key,
            period_index=new_period_index,
        )
        return new_timestep, new_state
