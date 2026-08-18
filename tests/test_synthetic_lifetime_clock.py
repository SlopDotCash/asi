"""Saturating lifetime clocks keep synthetic schedule identity at int32 exhaustion."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.streams.synthetic import (
    AbruptChangeStream,
    CyclicStream,
    PeriodicChangeStream,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def test_cyclic_stream_wrap_would_change_config_slot() -> None:
    """INT32_MAX+1 wrap selects a different cyclic configuration than saturate."""

    stream = CyclicStream(
        feature_dim=3,
        cycle_length=1,
        num_configurations=2,
        noise_std=0.0,
        feature_std=1.0,
    )
    planted = stream.init(jr.key(0)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    _, advanced = stream.step(planted, jnp.asarray(0, dtype=jnp.int32))
    assert int(advanced.step_count) == _INT32_MAX
    wrap_slot = int((jnp.asarray(_INT32_MIN, dtype=jnp.int32) // 1) % 2)
    sat_slot = int((advanced.step_count // 1) % 2)
    assert sat_slot != wrap_slot
    assert sat_slot == 1
    assert wrap_slot == 0


def test_abrupt_change_wrap_would_silence_the_change_schedule() -> None:
    """Wrap stores INT32_MIN, so the next interval-1 change is skipped."""

    stream = AbruptChangeStream(
        feature_dim=3,
        change_interval=1,
        noise_std=0.0,
        feature_std=1.0,
    )
    planted = stream.init(jr.key(1)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    _, advanced = stream.step(planted, jnp.asarray(0, dtype=jnp.int32))
    assert int(advanced.step_count) == _INT32_MAX
    wrap_should_change = bool(
        (jnp.asarray(_INT32_MIN, dtype=jnp.int32) > 0)
        & (jnp.asarray(_INT32_MIN, dtype=jnp.int32) % 1 == 0)
    )
    sat_should_change = bool(
        (advanced.step_count > 0) & (advanced.step_count % 1 == 0)
    )
    assert sat_should_change
    assert not wrap_should_change


def test_periodic_change_wrap_would_flip_the_oscillation_phase() -> None:
    """float32(INT32_MIN) is the negated phase of float32(INT32_MAX)."""

    stream = PeriodicChangeStream(
        feature_dim=2,
        period=8,
        amplitude=1.0,
        noise_std=0.0,
        feature_std=1.0,
    )
    planted = stream.init(jr.key(2)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    _, advanced = stream.step(planted, jnp.asarray(0, dtype=jnp.int32))
    assert int(advanced.step_count) == _INT32_MAX
    t_sat = advanced.step_count.astype(jnp.float32)
    t_wrap = jnp.asarray(_INT32_MIN, dtype=jnp.int32).astype(jnp.float32)
    phase_sat = jnp.sin(2.0 * jnp.pi * t_sat / jnp.float32(8.0))
    phase_wrap = jnp.sin(2.0 * jnp.pi * t_wrap / jnp.float32(8.0))
    assert float(phase_sat) != float(phase_wrap)
    assert float(phase_sat) == -float(phase_wrap)
