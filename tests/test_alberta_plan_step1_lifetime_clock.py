"""Saturating lifetime clocks keep Step 1 scale-change identity at int32 exhaustion."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.streams.alberta_plan_step1 import (
    AlbertaPlanStep1Stream,
    XDistShiftStream,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def test_xdist_shift_wrap_would_silence_the_scale_change_schedule() -> None:
    stream = XDistShiftStream(
        feature_dim=3,
        num_relevant=1,
        scale_change_interval=1,
        noise_std=0.0,
        noise_in_target=False,
    )
    planted = stream.init(jr.key(0)).replace(
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


def test_alberta_plan_step1_clock_saturates_at_int32_max() -> None:
    stream = AlbertaPlanStep1Stream(feature_dim=4, num_relevant=1, noise_std=0.0)
    planted = stream.init(jr.key(1)).replace(
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    _, advanced = stream.step(planted, jnp.asarray(0, dtype=jnp.int32))
    assert int(advanced.step_count) == _INT32_MAX
