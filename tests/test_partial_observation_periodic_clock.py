"""The periodic mask clock stores the exact bounded schedule phase."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from alberta_framework.streams.partial_observation import (
    MaskMode,
    PartialObservationWrapper,
)
from alberta_framework.streams.synthetic import RandomWalkStream

_INT32_MAX = 2**31 - 1
def test_periodic_period_index_remains_the_exact_schedule_phase() -> None:
    inner = RandomWalkStream(feature_dim=3, drift_rate=0.0, noise_std=0.0)
    schedule = (
        jnp.array([True, False, False]),
        jnp.array([False, True, False]),
        jnp.array([False, False, True]),
    )
    wrapper = PartialObservationWrapper(inner, mode=MaskMode.PERIODIC, schedule=schedule)

    state = wrapper.init(jr.key(0))
    for step, expected_phase in enumerate((1, 2, 0, 1)):
        _, state = wrapper.step(state, jnp.asarray(step, dtype=jnp.int32))
        assert int(state.period_index) == expected_phase


def test_legacy_saturated_period_index_resumes_the_schedule() -> None:
    inner = RandomWalkStream(feature_dim=3, drift_rate=0.0, noise_std=0.0)
    schedule = (
        jnp.array([True, False, False]),
        jnp.array([False, True, False]),
        jnp.array([False, False, True]),
    )
    wrapper = PartialObservationWrapper(inner, mode=MaskMode.PERIODIC, schedule=schedule)

    state = wrapper.init(jr.key(1)).replace(
        period_index=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    compiled_step = jax.jit(wrapper.step)
    for step, expected_phase in enumerate((2, 0, 1, 2)):
        _, state = compiled_step(state, jnp.asarray(step, dtype=jnp.int32))
        assert int(state.period_index) == expected_phase
