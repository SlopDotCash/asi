"""Zero-scale regression for ``ScaledStreamWrapper``."""

from typing import Any

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.types import TimeStep
from alberta_framework.streams.synthetic import ScaledStreamWrapper


class _NonFiniteChannelStream:
    """Minimal ScanStream whose middle channel diverges to +inf."""

    feature_dim = 3

    def init(self, key: Any) -> tuple[()]:
        del key
        return ()

    def step(self, state: tuple[()], idx: Any) -> tuple[TimeStep, tuple[()]]:
        del idx
        observation = jnp.array([1.0, jnp.inf, 3.0], dtype=jnp.float32)
        target = jnp.array([0.0], dtype=jnp.float32)
        return TimeStep(observation=observation, target=target), state


def test_zero_scale_suppresses_a_non_finite_inner_channel() -> None:
    scales = jnp.array([1.0, 0.0, 2.0], dtype=jnp.float32)
    wrapper = ScaledStreamWrapper(_NonFiniteChannelStream(), scales)
    inner_observation, _ = _NonFiniteChannelStream().step((), jnp.array(0))

    # The raw scale*value product is the 0*inf that used to leak a NaN.
    raw = inner_observation.observation * scales
    assert not bool(jnp.isfinite(raw[1]))

    timestep, _ = wrapper.step(wrapper.init(jr.key(0)), jnp.array(0))
    assert bool(jnp.all(jnp.isfinite(timestep.observation)))
    assert float(timestep.observation[1]) == 0.0
    # Non-zero scales are untouched.
    assert float(timestep.observation[0]) == 1.0
    assert float(timestep.observation[2]) == 6.0


def test_non_finite_feature_scales_are_still_rejected() -> None:
    scales = jnp.array([1.0, jnp.inf, 2.0], dtype=jnp.float32)
    with pytest.raises(ValueError, match="finite"):
        ScaledStreamWrapper(_NonFiniteChannelStream(), scales)
