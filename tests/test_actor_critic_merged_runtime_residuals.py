"""Regressions for actor-critic runtime sinks lost during integration."""

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from alberta_framework.core import actor_critic as actor_critic_module
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


class _MalformedBounder(Bounder):
    def to_config(self) -> dict[str, Any]:
        return {"type": "test-only"}

    def bound(
        self,
        steps: tuple[Array, ...],
        error: Array,
        params: tuple[Array, ...],
    ) -> tuple[tuple[Array, ...], Array]:
        del error, params
        return (steps[0][None, ...], *steps[1:]), jnp.ones((1,), dtype=jnp.float32)


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
def test_bounder_result_must_match_parameter_tree_and_scalar_metric(continuous: bool) -> None:
    if continuous:
        agent = ContinuousActorCriticAgent(
            ContinuousActorCriticConfig(action_dim=2), bounder=_MalformedBounder()
        )
        state = agent.init(2, jr.key(0))
    else:
        agent = ActorCriticAgent(
            ActorCriticConfig(n_actions=2), bounder=_MalformedBounder()
        )
        state = agent.init(2, jr.key(0)).replace(  # type: ignore[attr-defined]
            last_action=jnp.asarray(0, dtype=jnp.int32)
        )
    with pytest.raises(ValueError, match="bounder steps"):
        agent.update(  # type: ignore[union-attr]
            state, jnp.asarray(0.0, dtype=jnp.float32), jnp.zeros((2,), dtype=jnp.float32)
        )


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


def test_discrete_scan_working_set_formula_has_exact_byte_boundary() -> None:
    last_legal = (2**31 - 1 - 52) // 38
    actor_critic_module._require_discrete_scan_resources(
        n_actions=1, feature_dim=1, num_steps=last_legal
    )
    with pytest.raises(ValueError, match="working set"):
        actor_critic_module._require_discrete_scan_resources(
            n_actions=1, feature_dim=1, num_steps=last_legal + 1
        )


def test_continuous_scan_working_set_formula_has_exact_byte_boundary() -> None:
    last_legal = (2**31 - 1 - 60) // 42
    actor_critic_module._require_continuous_scan_resources(
        action_dim=1, feature_dim=1, num_steps=last_legal
    )
    with pytest.raises(ValueError, match="working set"):
        actor_critic_module._require_continuous_scan_resources(
            action_dim=1, feature_dim=1, num_steps=last_legal + 1
        )


@pytest.mark.parametrize("continuous", [False, True])
def test_scan_working_set_preflight_precedes_jax_conversion(
    continuous: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dim = 1_000
    num_steps = 1_000_000
    if continuous:
        agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=1))
        state = agent.init(feature_dim, jr.key(0))
        runner = run_continuous_actor_critic_from_arrays
    else:
        agent = ActorCriticAgent(ActorCriticConfig(n_actions=1))
        state = agent.init(feature_dim, jr.key(0))
        runner = run_actor_critic_from_arrays

    scalar = np.zeros((1,), dtype=np.float32)
    observations = np.lib.stride_tricks.as_strided(
        scalar,
        shape=(num_steps, feature_dim),
        strides=(0, 0),
    )
    rewards = np.lib.stride_tricks.as_strided(
        scalar,
        shape=(num_steps,),
        strides=(0,),
    )

    def unexpected_conversion(*args: object, **kwargs: object) -> None:
        raise AssertionError("JAX conversion ran before the working-set preflight")

    monkeypatch.setattr(actor_critic_module.jnp, "asarray", unexpected_conversion)
    with pytest.raises(ValueError, match="working set"):
        runner(  # type: ignore[arg-type]
            agent,
            state,
            observations,
            rewards,
            None,
            observations,
            discounts=rewards,
        )
