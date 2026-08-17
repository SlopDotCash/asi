"""Regressions for actor-critic runtime sinks lost during integration."""

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from alberta_framework.core.actor_critic import (
    ActorCriticAgent,
    ActorCriticConfig,
    ContinuousActorCriticAgent,
    ContinuousActorCriticConfig,
    run_actor_critic_from_arrays,
    run_continuous_actor_critic_from_arrays,
)
from alberta_framework.core.optimizers import Bounder


class _MaxMetricBounder(Bounder):
    def to_config(self) -> dict[str, Any]:
        return {"type": "test-only"}

    def bound(
        self,
        steps: tuple[Array, ...],
        error: Array,
        params: tuple[Array, ...],
    ) -> tuple[tuple[Array, ...], Array]:
        del error, params
        return steps, jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)


class _HostileArray:
    @property
    def shape(self) -> tuple[int, int]:
        return (2**31 - 1, 2)

    def __jax_array__(self) -> jax.Array:
        raise AssertionError("conversion executed before trusted metadata rejection")


@pytest.mark.parametrize("continuous", [False, True])
def test_bound_metric_average_of_finite_extremes_remains_finite(continuous: bool) -> None:
    if continuous:
        agent = ContinuousActorCriticAgent(
            ContinuousActorCriticConfig(action_dim=2), bounder=_MaxMetricBounder()
        )
        state = agent.init(2, jr.key(0))
    else:
        agent = ActorCriticAgent(ActorCriticConfig(n_actions=2), bounder=_MaxMetricBounder())
        state = agent.init(2, jr.key(0)).replace(  # type: ignore[attr-defined]
            last_action=jnp.asarray(0, dtype=jnp.int32)
        )
    result = agent.update(  # type: ignore[union-attr]
        state, jnp.asarray(0.0, dtype=jnp.float32), jnp.zeros((2,), dtype=jnp.float32)
    )
    assert bool(result.update_applied)
    assert bool(jnp.isfinite(result.bound_metric))


@pytest.mark.parametrize("continuous", [False, True])
def test_scan_rejects_hostile_input_before_jax_conversion(continuous: bool) -> None:
    if continuous:
        agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=2))
        state = agent.init(2, jr.key(0))
        runner = run_continuous_actor_critic_from_arrays
    else:
        agent = ActorCriticAgent(ActorCriticConfig(n_actions=2))
        state = agent.init(2, jr.key(0))
        runner = run_actor_critic_from_arrays
    with pytest.raises(ValueError, match="trusted array metadata"):
        runner(  # type: ignore[arg-type]
            agent,
            state,
            _HostileArray(),
            jnp.zeros((1,), dtype=jnp.float32),
            None,
            jnp.zeros((1, 2), dtype=jnp.float32),
            discounts=jnp.zeros((1,), dtype=jnp.float32),
        )
