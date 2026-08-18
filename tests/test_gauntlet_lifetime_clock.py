"""Saturating lifetime clocks keep GauntletStream segment identity at int32 exhaustion."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.streams.gauntlet import NUM_SEGMENTS, GauntletStream

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def test_gauntlet_wrap_forges_a_different_segment() -> None:
    """The old bare increment wraps INT32_MAX + 1 to INT32_MIN and resets the course."""

    stream = GauntletStream()
    wrap = jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    assert int(wrap) == _INT32_MIN
    assert int(stream.segment_of(wrap)) == 0
    assert int(stream.segment_of(jnp.asarray(_INT32_MAX, dtype=jnp.int32))) == NUM_SEGMENTS - 1


def test_gauntlet_clock_saturates_at_int32_max() -> None:
    """A planted INT32_MAX clock saturates instead of wrapping negative."""

    stream = GauntletStream()
    predecessor = stream.init(jr.key(0)).replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)
    )
    _, advanced = stream.step(predecessor, jnp.asarray(0, dtype=jnp.int32))
    assert int(advanced.step_count) == _INT32_MAX
    assert int(advanced.step_count) >= 0
    assert int(stream.segment_of(advanced.step_count)) == NUM_SEGMENTS - 1

    exhausted = stream.step(advanced, jnp.asarray(0, dtype=jnp.int32))
    assert int(exhausted[1].step_count) == _INT32_MAX
    assert int(exhausted[1].step_count) >= 0
    assert int(stream.segment_of(exhausted[1].step_count)) == NUM_SEGMENTS - 1
