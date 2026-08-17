"""Regressions for actor-critic runtime sinks lost during integration."""

from typing import Any

import chex
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
    # Two 52-byte carry trees plus an 880-byte reusable workspace.
    last_legal = (2**31 - 1 - 984) // 38
    actor_critic_module._require_discrete_scan_resources(
        n_actions=1, feature_dim=1, num_steps=last_legal
    )
    with pytest.raises(ValueError, match="working set"):
        actor_critic_module._require_discrete_scan_resources(
            n_actions=1, feature_dim=1, num_steps=last_legal + 1
        )


def test_continuous_scan_working_set_formula_has_exact_byte_boundary() -> None:
    # Two 60-byte carry trees plus a 976-byte reusable workspace.
    last_legal = (2**31 - 1 - 1_096) // 42
    actor_critic_module._require_continuous_scan_resources(
        action_dim=1, feature_dim=1, num_steps=last_legal
    )
    with pytest.raises(ValueError, match="working set"):
        actor_critic_module._require_continuous_scan_resources(
            action_dim=1, feature_dim=1, num_steps=last_legal + 1
        )


@pytest.mark.parametrize(
    ("continuous", "exact_bytes"),
    [
        (False, 4_410),
        (True, 5_022),
    ],
)
def test_scan_working_set_formula_covers_multidimensional_terms(
    continuous: bool,
    exact_bytes: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # At action count 3, feature count 5, and horizon 7, these are the exact
    # conservative-envelope byte totals, including both retained carry trees.
    monkeypatch.setattr(actor_critic_module, "_INT32_MAX", exact_bytes)
    if continuous:
        helper = actor_critic_module._require_continuous_scan_resources
        dimensions = {"action_dim": 3, "feature_dim": 5, "num_steps": 7}
    else:
        helper = actor_critic_module._require_discrete_scan_resources
        dimensions = {"n_actions": 3, "feature_dim": 5, "num_steps": 7}
    helper(**dimensions)
    monkeypatch.setattr(actor_critic_module, "_INT32_MAX", exact_bytes - 1)
    with pytest.raises(ValueError, match="working set"):
        helper(**dimensions)


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


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("malformed_name", ["terminated", "discounts", "actions"])
def test_scan_validates_all_host_metadata_before_any_jax_conversion(
    continuous: bool,
    malformed_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if continuous:
        agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=2))
        state = agent.init(2, jr.key(0))
        runner = run_continuous_actor_critic_from_arrays
        good_actions = np.zeros((1, 2), dtype=np.float32)
    else:
        agent = ActorCriticAgent(ActorCriticConfig(n_actions=2))
        state = agent.init(2, jr.key(0))
        runner = run_actor_critic_from_arrays
        good_actions = np.zeros((1,), dtype=np.int32)

    observations = np.zeros((1, 2), dtype=np.float32)
    rewards = np.zeros((1,), dtype=np.float32)
    terminated: object = np.zeros((1,), dtype=np.bool_)
    discounts: object = np.zeros((1,), dtype=np.float32)
    actions: object = good_actions
    if malformed_name == "terminated":
        terminated = np.zeros((2,), dtype=np.bool_)
    elif malformed_name == "discounts":
        discounts = np.zeros((2,), dtype=np.float32)
    else:
        actions = np.zeros((2,), dtype=good_actions.dtype)

    def unexpected_conversion(*args: object, **kwargs: object) -> None:
        raise AssertionError("JAX conversion ran before complete metadata validation")

    monkeypatch.setattr(actor_critic_module.jnp, "asarray", unexpected_conversion)
    with pytest.raises(ValueError, match=f"{malformed_name} must have shape"):
        runner(  # type: ignore[arg-type]
            agent,
            state,
            observations,
            rewards,
            terminated,
            observations,
            actions=actions,
            discounts=discounts,
        )


@pytest.mark.parametrize(
    "actions, message",
    [
        (jnp.asarray([0.75], dtype=jnp.float32), "integer dtype"),
        (jnp.asarray([-1], dtype=jnp.int32), r"\[0, n_actions\)"),
        (jnp.asarray([2], dtype=jnp.int32), r"\[0, n_actions\)"),
        (np.asarray([2**32], dtype=np.uint64), r"\[0, n_actions\)"),
    ],
)
def test_discrete_scan_rejects_actions_that_cannot_name_policy_entries(
    actions: Array,
    message: str,
) -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2))
    state = agent.init(2, jr.key(0))

    with pytest.raises(ValueError, match=message):
        run_actor_critic_from_arrays(
            agent,
            state,
            jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.asarray([False]),
            jnp.asarray([[0.0, 1.0]], dtype=jnp.float32),
            actions=actions,
        )


def test_discrete_scan_action_contract_is_jittable_and_invalid_values_are_atomic() -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2))
    state = agent.init(2, jr.key(0))
    observations = jnp.asarray([[1.0, 0.0]], dtype=jnp.float32)
    rewards = jnp.asarray([1.0], dtype=jnp.float32)
    terminated = jnp.asarray([False])
    next_observations = jnp.asarray([[0.0, 1.0]], dtype=jnp.float32)

    compiled = jax.jit(
        lambda initial_state, fixed_actions: run_actor_critic_from_arrays(
            agent,
            initial_state,
            observations,
            rewards,
            terminated,
            next_observations,
            actions=fixed_actions,
        )
    )
    valid = compiled(state, jnp.asarray([1], dtype=jnp.int32))
    assert bool(valid.updates_applied[0])

    invalid = compiled(state, jnp.asarray([2], dtype=jnp.int32))
    chex.assert_trees_all_equal(
        invalid.state.replace(rng_key=jr.key_data(invalid.state.rng_key)),
        state.replace(rng_key=jr.key_data(state.rng_key)),
    )
    assert not bool(jnp.any(invalid.updates_applied))
    assert not bool(jnp.any(invalid.actions))
    assert not bool(jnp.any(invalid.policies))
    assert not bool(jnp.any(invalid.values))
    assert not bool(jnp.any(invalid.td_errors))

    for bad_actions in (
        jnp.asarray([0.5], dtype=jnp.float32),
        jnp.asarray([True], dtype=jnp.bool_),
        jnp.asarray([[0]], dtype=jnp.int32),
    ):
        with pytest.raises(ValueError):
            compiled(state, bad_actions)


@pytest.mark.parametrize("continuous", [False, True])
def test_scan_requires_transition_semantics_before_jax_conversion(
    continuous: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if continuous:
        agent = ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=1))
        state = agent.init(1, jr.key(0))
        runner = run_continuous_actor_critic_from_arrays
    else:
        agent = ActorCriticAgent(ActorCriticConfig(n_actions=1))
        state = agent.init(1, jr.key(0))
        runner = run_actor_critic_from_arrays
    observations = np.zeros((1, 1), dtype=np.float32)
    rewards = np.zeros((1,), dtype=np.float32)

    def unexpected_conversion(*args: object, **kwargs: object) -> None:
        raise AssertionError("JAX conversion ran before transition-semantics validation")

    monkeypatch.setattr(actor_critic_module.jnp, "asarray", unexpected_conversion)
    with pytest.raises(ValueError, match="terminated or discounts"):
        runner(  # type: ignore[arg-type]
            agent,
            state,
            observations,
            rewards,
            None,
            observations,
        )
