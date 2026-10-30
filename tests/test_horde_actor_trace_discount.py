"""Actor eligibility uses the incoming discount, separately from bootstrapping."""

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.horde_actor_critic import (
    HordeActorCriticAgent,
    HordeActorCriticConfig,
    NonlinearHordeActorCriticAgent,
    NonlinearHordeActorCriticConfig,
    NonlinearQHordeActorCriticAgent,
    NonlinearQHordeActorCriticConfig,
    QHordeActorCriticAgent,
    QHordeActorCriticConfig,
    run_horde_actor_critic_from_arrays,
    run_nonlinear_horde_actor_critic_from_arrays,
)
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec

pytestmark = pytest.mark.unit


def _fixture(nonlinear: bool, lamda: float = 0.8):
    # Lambda zero isolates actor credit from the critic's separate trace rule.
    critic = HordeLearner(
        create_horde_spec(
            [
                GVFSpec(
                    name="value",
                    demon_type=DemonType.PREDICTION,
                    gamma=0.9,
                    lamda=0.0,
                    cumulant_index=0,
                )
            ]
        ),
        hidden_sizes=(),
        use_layer_norm=False,
        step_size=0.1,
    )
    if nonlinear:
        agent = NonlinearHordeActorCriticAgent(
            NonlinearHordeActorCriticConfig(
                n_actions=2, hidden_sizes=(2,), use_layer_norm=False, actor_lamda=lamda
            ),
            critic,
            actor_optimizer=LMS(step_size=0.1),  # type: ignore[arg-type]
        )
    else:
        agent = HordeActorCriticAgent(
            HordeActorCriticConfig(n_actions=2, actor_step_size=0.1, actor_lamda=lamda), critic
        )
    state = agent.init(2, jr.key(210))
    if nonlinear:
        state = state.replace(
            actor_trunk=state.actor_trunk.replace(weights=(jnp.eye(2),), biases=(jnp.ones(2),)),
            actor_head_w=jnp.array([[0.2, 0.4], [-0.2, -0.4]], dtype=jnp.float32),
        )
    heads = state.critic_state.head_params
    state = state.replace(
        critic_state=state.critic_state.replace(
            head_params=heads.replace(
                weights=tuple(jnp.zeros_like(w) for w in heads.weights),
                biases=tuple(jnp.zeros_like(b) for b in heads.biases),
            )
        )
    )
    return agent, state


def _params(state, nonlinear: bool):
    if nonlinear:
        return tuple(
            leaf
            for pair in zip(state.actor_trunk.weights, state.actor_trunk.biases, strict=True)
            for leaf in pair
        ) + (state.actor_head_w, state.actor_head_b)
    return state.actor_weights, state.actor_bias


def _traces(state, nonlinear: bool):
    if nonlinear:
        return (*state.actor_trunk_traces, state.actor_head_trace_w, state.actor_head_trace_b)
    return state.actor_trace_weights, state.actor_trace_bias


def _clear_traces(state, nonlinear: bool):
    if nonlinear:
        return state.replace(
            actor_trunk_traces=tuple(jnp.zeros_like(t) for t in state.actor_trunk_traces),
            actor_head_trace_w=jnp.zeros_like(state.actor_head_trace_w),
            actor_head_trace_b=jnp.zeros_like(state.actor_head_trace_b),
        )
    return state.replace(
        actor_trace_weights=jnp.zeros_like(state.actor_trace_weights),
        actor_trace_bias=jnp.zeros_like(state.actor_trace_bias),
    )


def _warm(agent, state, discount=0.2):
    state = agent.start(state, jnp.array([1.0, 0.0]))[0]
    first = agent.update(
        state,
        jnp.float32(0.0),
        jnp.array([0.0, 1.0]),
        discount=None if discount is None else jnp.float32(discount),
    )
    assert bool(first.update_applied)
    return first.state


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("outgoing", [0.0, 0.7])
@pytest.mark.parametrize("disable_jit", [False, True])
def test_prior_discount_controls_delayed_credit(nonlinear, outgoing, disable_jit):
    with jax.disable_jit(disable_jit):
        agent, initial = _fixture(nonlinear)
        state = _warm(agent, initial)
        traces = _traces(state, nonlinear)
        assert all(float(jnp.linalg.norm(trace)) > 0.0 for trace in traces)
        result = agent.update(state, jnp.float32(1.0), jnp.ones(2), discount=jnp.float32(outgoing))
        ablated = agent.update(
            _clear_traces(state, nonlinear),
            jnp.float32(1.0),
            jnp.ones(2),
            discount=jnp.float32(outgoing),
        )
        assert bool(result.update_applied) and bool(ablated.update_applied)
        np.testing.assert_array_equal(result.td_error, ablated.td_error)
        assert float(result.td_error) == 1.0
        for actual, control, trace in zip(
            _params(result.state, nonlinear), _params(ablated.state, nonlinear), traces, strict=True
        ):
            np.testing.assert_allclose(actual - control, 0.1 * 0.2 * 0.8 * trace, atol=1e-7)
        if outgoing == 0.0:
            for trace in _traces(result.state, nonlinear):
                np.testing.assert_array_equal(trace, jnp.zeros_like(trace))


@pytest.mark.parametrize("outgoing", [0.0, 0.7])
def test_public_array_runner_preserves_delayed_reward(outgoing):
    agent, state = _fixture(False)
    result = run_horde_actor_critic_from_arrays(
        agent,
        state,
        jnp.eye(2),
        jnp.array([0.0, 1.0]),
        jnp.array([[0.0, 1.0], [0.0, 0.0]]),
        actions=jnp.array([0, 0]),
        discounts=jnp.array([0.2, outgoing]),
    )
    np.testing.assert_array_equal(result.updates_applied, [True, True])
    np.testing.assert_allclose(
        result.state.actor_weights, [[0.008, 0.05], [-0.008, -0.05]], atol=1e-7
    )
    np.testing.assert_allclose(result.state.actor_bias, [0.058, -0.058], atol=1e-7)


def _assert_same_tree(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jr.key_data(a), jr.key_data(b)
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_rejection_retry_and_episode_boundary(nonlinear):
    agent, initial = _fixture(nonlinear)
    state = _warm(agent, initial)
    rejected = agent.update(state, jnp.float32(jnp.nan), jnp.ones(2), discount=jnp.float32(0.0))
    assert not bool(rejected.update_applied)
    _assert_same_tree(rejected.state, state)
    retry = agent.update(rejected.state, jnp.float32(1.0), jnp.ones(2), discount=jnp.float32(0.0))
    direct = agent.update(state, jnp.float32(1.0), jnp.ones(2), discount=jnp.float32(0.0))
    assert bool(retry.update_applied)
    _assert_same_tree(retry, direct)
    assert float(retry.state.previous_discount) == 0.0
    for trace in _traces(retry.state, nonlinear):
        np.testing.assert_array_equal(trace, jnp.zeros_like(trace))
    # Even stale finite eligibility cannot cross a persisted terminal barrier.
    restarted = agent.start(retry.state, jnp.array([1.0, 0.0]))[0]
    if nonlinear:
        stale = restarted.replace(
            actor_trunk_traces=state.actor_trunk_traces,
            actor_head_trace_w=state.actor_head_trace_w,
            actor_head_trace_b=state.actor_head_trace_b,
        )
    else:
        stale = restarted.replace(
            actor_trace_weights=state.actor_trace_weights, actor_trace_bias=state.actor_trace_bias
        )
    clean_result = agent.update(restarted, jnp.float32(0.5), jnp.ones(2))
    stale_result = agent.update(stale, jnp.float32(0.5), jnp.ones(2))
    assert bool(clean_result.update_applied) and bool(stale_result.update_applied)
    _assert_same_tree(stale_result, clean_result)


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("lamda", [0.0, 0.8])
def test_default_discount_matches_explicit_fixed_discount(nonlinear, lamda):
    agent, initial = _fixture(nonlinear, lamda)
    implicit, explicit = _warm(agent, initial, None), _warm(agent, initial, 0.9)
    _assert_same_tree(implicit, explicit)
    implicit_result = agent.update(implicit, jnp.float32(1.0), jnp.ones(2))
    explicit_result = agent.update(
        explicit, jnp.float32(1.0), jnp.ones(2), discount=jnp.float32(0.9)
    )
    assert bool(implicit_result.update_applied) and bool(explicit_result.update_applied)
    _assert_same_tree(implicit_result, explicit_result)
    if lamda == 0.0:
        ablated = agent.update(_clear_traces(implicit, nonlinear), jnp.float32(1.0), jnp.ones(2))
        _assert_same_tree(implicit_result, ablated)


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("history", [-0.1, 1.1, float("nan")])
def test_invalid_history_rejects_entire_transaction_even_with_lambda_zero(nonlinear, history):
    agent, initial = _fixture(nonlinear, 0.0)
    state = _warm(agent, initial).replace(previous_discount=jnp.float32(history))
    result = agent.update(state, jnp.float32(1.0), jnp.ones(2), discount=jnp.float32(0.7))
    assert not bool(result.update_applied)
    _assert_same_tree(result.state, state)
    assert not bool(result.critic_result.update_applied)


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("history", [jnp.array([0.2]), jnp.int32(0)])
def test_history_requires_scalar_float32(nonlinear, history):
    agent, initial = _fixture(nonlinear)
    state = _warm(agent, initial).replace(previous_discount=history)
    with pytest.raises(ValueError, match="previous_discount must be a scalar float32"):
        agent.update(state, jnp.float32(1.0), jnp.ones(2))


@pytest.mark.parametrize("nonlinear", [False, True])
def test_array_runner_rejected_row_preserves_history_and_rng(nonlinear):
    agent, initial = _fixture(nonlinear)
    runner = (
        run_nonlinear_horde_actor_critic_from_arrays
        if nonlinear
        else run_horde_actor_critic_from_arrays
    )
    observations = jnp.array([[1.0, 0.0], [9.0, 9.0], [0.0, 1.0]])
    next_observations = jnp.array([[0.0, 1.0], [9.0, 9.0], [0.0, 0.0]])
    rewards, discounts = jnp.array([0.0, jnp.nan, 1.0]), jnp.array([0.2, 0.0, 0.7])
    result = runner(agent, initial, observations, rewards, next_observations, discounts=discounts)
    kept = jnp.array([0, 2])
    control = runner(
        agent,
        initial,
        observations[kept],
        rewards[kept],
        next_observations[kept],
        discounts=discounts[kept],
    )
    np.testing.assert_array_equal(result.updates_applied, [True, False, True])
    _assert_same_tree(result.state, control.state)
    assert float(result.state.previous_discount) == pytest.approx(0.7)


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("q_critic", [False, True])
def test_actor_budget_matches_allocated_state(nonlinear, q_critic):
    agent_class, config_class = (
        (
            (NonlinearQHordeActorCriticAgent, NonlinearQHordeActorCriticConfig)
            if q_critic
            else (NonlinearHordeActorCriticAgent, NonlinearHordeActorCriticConfig)
        )
        if nonlinear
        else (
            (QHordeActorCriticAgent, QHordeActorCriticConfig)
            if q_critic
            else (HordeActorCriticAgent, HordeActorCriticConfig)
        )
    )
    config = config_class(n_actions=2, **({"hidden_sizes": (3,)} if nonlinear else {}))
    critic = HordeLearner(
        create_horde_spec(
            [
                GVFSpec(
                    name=f"head{i}",
                    demon_type=DemonType.CONTROL if q_critic else DemonType.PREDICTION,
                    gamma=0.0 if q_critic else 0.9,
                    lamda=0.0,
                    cumulant_index=0,
                )
                for i in range(2 if q_critic else 1)
            ]
        ),
        hidden_sizes=(),
    )
    state = agent_class(config, critic).init(4, jr.key(210))
    leaves = jax.tree.leaves(
        {
            field.name: getattr(state, field.name)
            for field in dataclasses.fields(state)
            if field.name != "critic_state"
        }
    )
    leaves = [
        jr.key_data(x) if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key) else x for x in leaves
    ]
    budget = config.actor_resource_budget(4)
    assert budget["state_nbytes"] == sum(x.nbytes for x in leaves)
    assert budget["state_scalars"] == sum(x.size for x in leaves)
    assert budget["float32_state_scalars"] == sum(x.size for x in leaves if x.dtype == jnp.float32)
