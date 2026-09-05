"""Regression: closed context gates must not turn poisoned x into NaN."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from alberta_framework.core.types import TimeStep
from alberta_framework.streams.gauntlet import ContextGatedFeatures, GauntletConfig

pytestmark = pytest.mark.unit


class _PoisonedInner:
    """Stub stream that returns a fixed observation."""

    def __init__(self, observation: Any) -> None:
        self.config = GauntletConfig(
            relevant_dim=2,
            irrelevant_dim=0,
            segment_length=1,
            noise_std=0.0,
            context_noise_std=0.0,
        )
        self._observation = observation

    def step(self, state: Any, idx: Any) -> tuple[TimeStep, Any]:
        del idx
        target = jnp.zeros((1,), dtype=jnp.float32)
        return TimeStep(observation=self._observation, target=target), state


def test_closed_context_gates_do_not_multiply_inf_features() -> None:
    """Exact-zero gates times poisoned x is 0*inf = NaN without a skip."""
    # ctx=(1,0) => exclusive base_gate=0 and ctx[1]=0; ctx[0]=1 keeps one open block.
    observation = jnp.asarray([jnp.inf, -jnp.inf, 1.0, 0.0], dtype=jnp.float32)
    raw_closed = jnp.asarray(0.0, dtype=jnp.float32) * observation[:2]
    assert bool(jnp.any(~jnp.isfinite(raw_closed)))

    wrapper = ContextGatedFeatures(_PoisonedInner(observation), exclusive=True)
    timestep, _ = wrapper.step(None, jnp.asarray(0, dtype=jnp.int32))
    gated = timestep.observation

    assert gated.shape == (2 + 2 + 2 + 2,)
    assert not bool(jnp.any(jnp.isnan(gated)))
    np_gated = jnp.asarray(gated)
    assert bool(jnp.all(np_gated[:2] == 0.0))
    assert bool(jnp.all(np_gated[4:6] == observation[:2]))
    assert bool(jnp.all(np_gated[6:] == 0.0))
