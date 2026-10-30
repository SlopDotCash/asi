"""Finite float32 sentinels must survive lower-precision observations."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.types import TimeStep
from alberta_framework.streams.partial_observation import PartialObservationWrapper


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("sentinel", [-100000.0, 100000.0])
@pytest.mark.parametrize("execution", ["eager", "jit", "scan"])
def test_sentinel_precision(dtype: Any, sentinel: float, execution: str) -> None:
    class Stream:
        feature_dim = 2

        def init(self, key: jax.Array) -> jax.Array:
            return key

        def step(self, state: jax.Array, idx: jax.Array) -> tuple[TimeStep, jax.Array]:
            return TimeStep(
                observation=jnp.asarray([1.0, 2.0], dtype), target=jnp.asarray([3.0], dtype)
            ), state

    wrapper = PartialObservationWrapper(
        Stream(), fixed_mask=np.asarray([True, False]), sentinel=sentinel
    )
    state = wrapper.init(jax.random.key(0))
    if execution == "scan":
        def body(carry: Any, index: jax.Array) -> tuple[Any, TimeStep]:
            timestep, updated = wrapper.step(carry, index)
            return updated, timestep
        _, timesteps = jax.lax.scan(body, state, jnp.arange(3, dtype=jnp.int32))
        observation, target = timesteps.observation[-1], timesteps.target[-1]
    else:
        step = jax.jit(wrapper.step) if execution == "jit" else wrapper.step
        timestep, _ = step(state, jnp.asarray(0, dtype=jnp.int32))
        observation, target = timestep.observation, timestep.target
    np.testing.assert_array_equal(observation, [1.0, sentinel])
    assert observation.dtype == jnp.float32
    np.testing.assert_array_equal(target, jnp.asarray([3.0], dtype))
    assert target.dtype == dtype
