"""History-feature extractor for recurrent state construction (Step 3).

Implements decaying-trace ("EMA") features over observation channels --
the simplest form of memory needed for partially observable settings.

Sutton, Bowling, & Pilarski (2022, p.8) Step 3: features in Step 3 must
include "not just nonlinear combinations, but also incorporation of older
signals and traces." A history-feature bank with multiple decay rates is
the simplest realization of this idea.

Mathematically, for each observation channel ``i`` and each decay rate
``beta_k`` in ``decay_rates``, the trace feature is::

    h_{i,k}(t) = beta_k * h_{i,k}(t-1) + (1 - beta_k) * obs_i(t)

This is an EMA with timescale ``1 / (1 - beta_k)``, giving an effective
memory horizon. With several decay rates we get a multi-timescale view
of the recent observation history -- the kind of representation that
lets a Horde demon condition predictions on what happened in the past.

Pairs cleanly with ``streams/partial_observation.py``: a POMDP wrapper
masks part of the observation, and the agent recovers the missing
information by conditioning on history features.

Reference: Sutton & Tanner 2004 (Temporal-Difference Networks);
the multi-timescale trace idea is also central to Modayil et al. 2014
nexting work.
"""

from __future__ import annotations

import functools
import operator
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

from alberta_framework.core._float32_scalars import validated_float32_scalar

# =============================================================================
# Types
# =============================================================================


@chex.dataclass(frozen=True)
class HistoryFeatureState:
    """State for the history-feature extractor.

    Attributes:
        traces: Per-decay-rate, per-channel trace values, shape
            ``(n_decay_rates, n_channels)``.
    """

    traces: Float[Array, "n_decays n_channels"]


# =============================================================================
# Extractor
# =============================================================================

_INT32_MAX = 2**31 - 1
# One 12-bit cardinality budget bounds direct tuple validation, serialized-list
# normalization, and the implicit all-channel expansion before per-item work.
_MAX_HISTORY_CONFIGURATION_ITEMS = 1 << 12
_MAX_HISTORY_DECAY_RATES = _MAX_HISTORY_CONFIGURATION_ITEMS
_MAX_HISTORY_CHANNELS = _MAX_HISTORY_CONFIGURATION_ITEMS
_ACTUAL_INT_TYPES = frozenset({int, *(np.dtype(code).type for code in "bBhHiIlLqQpP")})


def _require_int32(name: str, value: object, *, minimum: int, maximum: int) -> int:
    actual_type = type(value)
    if not any(actual_type is allowed_type for allowed_type in _ACTUAL_INT_TYPES):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return canonical


def _require_float32_resource(name: str, *, float32_scalars: int) -> None:
    if float32_scalars > _INT32_MAX:
        raise ValueError(f"{name} scalar count must fit signed int32")
    if 4 * float32_scalars > _INT32_MAX:
        raise ValueError(f"{name} byte count must fit signed int32")


def _preflight_history_resources(
    *, raw_dim: int, n_decays: int, n_channels: int, feature_dim: int
) -> None:
    """Bound the trace state, output, and complete source-level step envelope."""

    trace_scalars = n_decays * n_channels
    _require_float32_resource(
        "history-feature traces",
        float32_scalars=trace_scalars,
    )
    _require_float32_resource(
        "history-feature output",
        float32_scalars=feature_dim,
    )
    # Charge source/product/zero/carried/contribution/proposed/finite-mask/
    # selected trace banks; raw/finite/zero/safe observation vectors; channel
    # indices and tracked observations; decay temporaries; the returned output;
    # and scalar predicates. Every logical element is charged at four bytes,
    # conservatively covering float32, int32, and boolean arrays.
    step_scalars = (
        8 * trace_scalars
        + 4 * raw_dim
        + 2 * n_channels
        + 3 * n_decays
        + feature_dim
        + 16
    )
    _require_float32_resource(
        "history-feature step working set",
        float32_scalars=step_scalars,
    )


def _require_decay_rates(value: object) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise ValueError("decay_rates must be an actual tuple")
    rates = cast(tuple[object, ...], value)
    if len(rates) > _MAX_HISTORY_DECAY_RATES:
        raise ValueError(
            f"decay_rates must contain at most {_MAX_HISTORY_DECAY_RATES} rates"
        )
    return tuple(
        validated_float32_scalar(
            f"decay_rates[{index}]",
            rate,
            lower=0.0,
            upper=1.0,
            upper_inclusive=False,
        )
        for index, rate in enumerate(rates)
    )


def _require_channels(value: object, raw_dim: int) -> tuple[int, ...] | None:
    if value is None:
        return None
    if type(value) is not tuple:
        raise ValueError("channels must be an actual tuple or None")
    channels = cast(tuple[object, ...], value)
    if len(channels) > _MAX_HISTORY_CHANNELS:
        raise ValueError(
            f"channels must contain at most {_MAX_HISTORY_CHANNELS} indices"
        )
    return tuple(
        _require_int32(
            f"channels[{index}]",
            channel,
            minimum=0,
            maximum=raw_dim - 1,
        )
        for index, channel in enumerate(channels)
    )


class HistoryFeatureExtractor:
    """Decaying-trace history-feature extractor.

    Given an observation ``obs`` of shape ``(raw_dim,)``, produces an
    augmented observation of shape ``(out_dim,)``::

        out_dim = (raw_dim if include_raw else 0)
                  + len(channels) * len(decay_rates)

    Channels chosen for tracing are ``range(raw_dim)`` by default, or
    a custom subset if ``channels`` is given. ``include_raw`` controls
    whether the raw observation is concatenated to the front (default True
    -- this is the "augmented_observation" pattern used by Step 2's
    ``FixedBudgetInteractionLearner``).

    JIT-compiled. Pure functional, no mutation.

    Examples
    --------
    ```python
    extractor = HistoryFeatureExtractor(
        raw_dim=4,
        decay_rates=(0.5, 0.9, 0.99),
    )
    state = extractor.init()
    obs = jnp.array([1.0, 0.0, -0.3, 0.7])
    aug, state = extractor.step(state, obs)
    # aug has shape (4 + 4*3,) = (16,)
    ```

    Attributes:
        raw_dim: Dimension of the raw observation
        decay_rates: Tuple of EMA decay rates beta_k in [0, 1)
        channels: Indices of observation channels to track
        include_raw: Whether the raw observation is concatenated to the front
    """

    def __init__(
        self,
        raw_dim: int,
        decay_rates: tuple[float, ...] = (0.5, 0.9, 0.99),
        channels: tuple[int, ...] | None = None,
        include_raw: bool = True,
    ):
        """Initialize the history-feature extractor.

        Args:
            raw_dim: Dimension of the raw observation vector
            decay_rates: EMA decay rates ``beta_k`` in ``[0, 1)``. Each rate
                yields one trace feature per selected channel.
            channels: Indices of observation channels to track. ``None``
                means all channels (``range(raw_dim)``).
            include_raw: If True (default), the raw observation is
                concatenated to the front of the augmented observation.
        """
        raw_dim = _require_int32(
            "raw_dim", raw_dim, minimum=1, maximum=_INT32_MAX
        )
        decay_rates = _require_decay_rates(decay_rates)
        canonical_channels = _require_channels(channels, raw_dim)
        if type(include_raw) is not bool:
            raise ValueError("include_raw must be an actual bool")

        channel_count = raw_dim if canonical_channels is None else len(canonical_channels)
        feature_dim = channel_count * len(decay_rates)
        if include_raw:
            feature_dim += raw_dim
        if feature_dim < 1 or feature_dim > _INT32_MAX:
            raise ValueError(
                f"feature_dim must be in [1, {_INT32_MAX}], got {feature_dim}"
            )
        _preflight_history_resources(
            raw_dim=raw_dim,
            n_decays=len(decay_rates),
            n_channels=channel_count,
            feature_dim=feature_dim,
        )
        if channel_count > _MAX_HISTORY_CHANNELS:
            raise ValueError(
                f"channels must contain at most {_MAX_HISTORY_CHANNELS} indices"
            )
        if canonical_channels is None:
            canonical_channels = tuple(range(raw_dim))

        self._raw_dim = raw_dim
        self._decay_rates = decay_rates
        self._channels = canonical_channels
        self._include_raw = include_raw

    @property
    def raw_dim(self) -> int:
        """Dimension of the raw observation."""
        return self._raw_dim

    @property
    def decay_rates(self) -> tuple[float, ...]:
        """Tuple of EMA decay rates."""
        return self._decay_rates

    @property
    def channels(self) -> tuple[int, ...]:
        """Tracked observation-channel indices."""
        return self._channels

    @property
    def include_raw(self) -> bool:
        """Whether the raw observation is included in the augmented output."""
        return self._include_raw

    def feature_dim(self) -> int:
        """Dimension of the augmented observation."""
        out = len(self._channels) * len(self._decay_rates)
        if self._include_raw:
            out += self._raw_dim
        return out

    def init(self) -> HistoryFeatureState:
        """Initialize traces to zero."""
        traces = jnp.zeros(
            (len(self._decay_rates), len(self._channels)), dtype=jnp.float32
        )
        return HistoryFeatureState(traces=traces)  # type: ignore[call-arg]

    def _require_state_contract(self, state: HistoryFeatureState) -> None:
        if type(state) is not HistoryFeatureState:
            raise TypeError("state must be an exact HistoryFeatureState")
        traces_type = type(state.traces)
        if not (
            issubclass(traces_type, jax.Array)
            or issubclass(traces_type, jax.core.Tracer)
        ):
            raise TypeError("state.traces must be a JAX array")
        expected_shape = (len(self._decay_rates), len(self._channels))
        if state.traces.shape != expected_shape or state.traces.dtype != jnp.float32:
            raise ValueError("state.traces has an invalid shape or dtype")

    def step(
        self,
        state: HistoryFeatureState,
        observation: Float[Array, " raw_dim"],
    ) -> tuple[Float[Array, " out_dim"], HistoryFeatureState]:
        """Update traces and produce the augmented observation.

        Args:
            state: Current history-feature state
            observation: Raw observation, shape ``(raw_dim,)``

        Returns:
            Tuple ``(augmented, new_state)``:
            - ``augmented`` has shape ``(out_dim,)`` -- raw observation
              (if ``include_raw``) followed by the trace bank flattened
              channel-major within each decay rate
            - ``new_state`` carries the updated trace values
        """
        self._require_state_contract(state)
        return cast(
            tuple[Float[Array, " out_dim"], HistoryFeatureState],
            self._step_jit(state, observation),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_jit(
        self,
        state: HistoryFeatureState,
        observation: Float[Array, " raw_dim"],
    ) -> tuple[Float[Array, " out_dim"], HistoryFeatureState]:
        """Execute one validated history update."""
        # Select tracked channels
        channel_indices = jnp.asarray(self._channels, dtype=jnp.int32)
        observation = jnp.asarray(observation, dtype=jnp.float32)
        if observation.shape != (self._raw_dim,):
            raise ValueError(
                f"observation must have shape ({self._raw_dim},), "
                f"got {observation.shape}"
            )
        observation_valid = jnp.all(jnp.isfinite(observation))
        safe_observation = jnp.where(
            jnp.isfinite(observation), observation, jnp.zeros_like(observation)
        )
        obs_tracked = observation[channel_indices]

        # EMA decay per trace row
        decay = jnp.asarray(self._decay_rates, dtype=jnp.float32)[:, None]
        # decay=0 times an inf stored trace is 0*inf = NaN. Skip that product.
        carried = jnp.where(decay == 0.0, jnp.zeros_like(state.traces), decay * state.traces)
        proposed_traces = carried + (1.0 - decay) * obs_tracked[None, :]
        traces_finite = jnp.all(jnp.isfinite(proposed_traces))
        new_traces = jnp.where(
            observation_valid & traces_finite, proposed_traces, state.traces
        )

        # Flatten and concatenate
        flat_traces = new_traces.reshape(-1)
        if self._include_raw:
            augmented = jnp.concatenate([safe_observation, flat_traces])
        else:
            augmented = flat_traces

        return augmented, HistoryFeatureState(traces=new_traces)  # type: ignore[call-arg]

    def to_config(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "type": "HistoryFeatureExtractor",
            "raw_dim": self._raw_dim,
            "decay_rates": list(self._decay_rates),
            "channels": list(self._channels),
            "include_raw": self._include_raw,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> HistoryFeatureExtractor:
        """Reconstruct from config dict."""
        config = dict(config)
        config.pop("type", None)
        raw_rates = config["decay_rates"]
        if type(raw_rates) is list:
            if len(raw_rates) > _MAX_HISTORY_DECAY_RATES:
                raise ValueError(
                    f"decay_rates must contain at most {_MAX_HISTORY_DECAY_RATES} rates"
                )
            raw_rates = tuple(raw_rates)
        raw_channels = config["channels"]
        if type(raw_channels) is list:
            if len(raw_channels) > _MAX_HISTORY_CHANNELS:
                raise ValueError(
                    f"channels must contain at most {_MAX_HISTORY_CHANNELS} indices"
                )
            raw_channels = tuple(raw_channels)
        return cls(
            raw_dim=config["raw_dim"],
            decay_rates=raw_rates,
            channels=raw_channels,
            include_raw=config["include_raw"],
        )
