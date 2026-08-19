"""Saturating periodic clock keeps schedule-phase identity at int32 exhaustion.

``PartialObservationWrapper`` under ``MaskMode.PERIODIC`` tracks
``period_index`` as an int32 JAX scalar and indexes the mask schedule with
``period_index % len(schedule)``. Incrementing that counter with plain
``+ 1`` silently wraps ``INT32_MAX`` to ``INT32_MIN`` (signed int32
overflow), which desyncs the observed schedule phase from the true
(unbounded) step count: ``2**31 % 3 == 2`` but ``(-2**31) % 3 == 1``. A run
long enough to exhaust int32 would silently start masking the wrong
channels forever after, with no error. This mirrors the already-fixed
lifetime/step counters elsewhere in ``alberta_framework.streams`` (Step 1,
gauntlet, feature discovery, synthetic, out-of-class): saturate instead of
wrapping.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.streams.partial_observation import (
    MaskMode,
    PartialObservationWrapper,
)
from alberta_framework.streams.synthetic import RandomWalkStream

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def test_unsaturated_wraparound_would_desync_the_schedule_phase() -> None:
    """Document the corruption a plain ``+ 1`` would cause at exhaustion.

    This is a static arithmetic fact about int32 wraparound, independent of
    the fix: the wrapped (negative) counter reduces to a different residue
    than the true, unbounded step count would.
    """
    schedule_length = 3
    true_next_step_count = 2**31  # INT32_MAX + 1, in unbounded arithmetic
    wrapped_step_count = _INT32_MIN  # what a signed int32 `+ 1` overflow stores

    true_phase = true_next_step_count % schedule_length
    wrapped_phase = wrapped_step_count % schedule_length

    assert true_phase != wrapped_phase


def test_periodic_period_index_saturates_at_int32_max() -> None:
    """The counter must not overflow to a negative int32 value."""
    inner = RandomWalkStream(feature_dim=3, drift_rate=0.0, noise_std=0.0)
    schedule = (
        jnp.array([True, False, False]),
        jnp.array([False, True, False]),
        jnp.array([False, False, True]),
    )
    wrapper = PartialObservationWrapper(inner, mode=MaskMode.PERIODIC, schedule=schedule)

    planted = wrapper.init(jr.key(0)).replace(
        period_index=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    _, advanced = wrapper.step(planted, jnp.asarray(0, dtype=jnp.int32))

    assert int(advanced.period_index) == _INT32_MAX
    assert int(advanced.period_index) >= 0


def test_periodic_period_index_stays_nonnegative_across_repeated_saturation() -> None:
    """Once saturated, further steps must keep the counter pinned, not wrap."""
    inner = RandomWalkStream(feature_dim=2, drift_rate=0.0, noise_std=0.0)
    schedule = (jnp.array([True, False]), jnp.array([False, True]))
    wrapper = PartialObservationWrapper(inner, mode=MaskMode.PERIODIC, schedule=schedule)

    state = wrapper.init(jr.key(1)).replace(
        period_index=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    for step in range(3):
        _, state = wrapper.step(state, jnp.asarray(step, dtype=jnp.int32))
        assert int(state.period_index) == _INT32_MAX
