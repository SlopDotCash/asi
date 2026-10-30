"""Terminal Q-Horde actor updates must retain within-episode eligibility."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.horde_actor_critic import (
    NonlinearQHordeActorCriticAgent,
    NonlinearQHordeActorCriticConfig,
    QHordeActorCriticAgent,
    QHordeActorCriticConfig,
)
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec


def _fixture(nonlinear: bool, critic_target: str, actor_update: str, gamma: float = 0.9):
    critic = HordeLearner(
        create_horde_spec(
            [
                GVFSpec(
                    name=f"q{i}",
                    demon_type=DemonType.CONTROL,
                    gamma=0.0,
                    lamda=0.0,
                    cumulant_index=-1,
                )
                for i in range(2)
            ]
        ),
        hidden_sizes=(),
        use_layer_norm=False,
        step_size=0.1,
    )
    kwargs = dict(
        n_actions=2,
        gamma=gamma,
        actor_lamda=0.8,
        temperature=1.0,
        critic_target=critic_target,
        actor_update=actor_update,
    )
    if nonlinear:
        agent = NonlinearQHordeActorCriticAgent(
            NonlinearQHordeActorCriticConfig(hidden_sizes=(2,), use_layer_norm=False, **kwargs),
            critic,
            actor_optimizer=LMS(step_size=0.1),  # type: ignore[arg-type]
        )
    else:
        agent = QHordeActorCriticAgent(
            QHordeActorCriticConfig(actor_step_size=0.1, **kwargs), critic
        )
    state = agent.init(2, jr.key(180))
    if nonlinear:
        # Positive hidden activations and nonzero head weights exercise all
        # trunk/head trace paths without a ReLU derivative at zero.
        state = state.replace(
            actor_trunk=state.actor_trunk.replace(weights=(jnp.eye(2),), biases=(jnp.ones((2,)),)),
            actor_head_w=jnp.array([[0.2, 0.4], [-0.2, -0.4]], dtype=jnp.float32),
        )
    heads = state.critic_state.head_params
    biases = (1.0, -1.0) if actor_update == "expected_advantage" else (0.0, 0.0)
    state = state.replace(
        critic_state=state.critic_state.replace(
            head_params=heads.replace(
                weights=tuple(jnp.zeros_like(w) for w in heads.weights),
                biases=tuple(
                    jnp.full_like(b, value) for b, value in zip(heads.biases, biases, strict=True)
                ),
            )
        )
    )
    state = agent.start(state, jnp.array([1.0, 0.0], dtype=jnp.float32))[0]
    first = agent.update(state, jnp.float32(0.0), jnp.array([0.0, 1.0]), jnp.bool_(False))
    assert bool(first.update_applied)
    return agent, first.state


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


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("critic_target", ["expected_sarsa", "sampled_sarsa"])
@pytest.mark.parametrize("actor_update", ["td_error", "expected_advantage"])
def test_terminal_credits_retained_actor_traces(
    nonlinear: bool, critic_target: str, actor_update: str
) -> None:
    agent, state = _fixture(nonlinear, critic_target, actor_update)
    traces = _traces(state, nonlinear)
    assert all(float(jnp.linalg.norm(trace)) > 0.0 for trace in traces)
    # Ablate only the earlier eligibility. Both updates have identical current
    # parameters, critic, observation/action and RNG. Their difference must be
    # precisely the retained credit, under the fixed-step LMS actor optimizer.
    without_history = _clear_traces(state, nonlinear)
    for terminated in [False, True]:
        result = agent.update(state, jnp.float32(1.0), jnp.ones((2,)), jnp.bool_(terminated))
        ablated = agent.update(
            without_history, jnp.float32(1.0), jnp.ones((2,)), jnp.bool_(terminated)
        )
        assert bool(result.update_applied) and bool(ablated.update_applied)
        np.testing.assert_array_equal(result.td_error, ablated.td_error)
        signal = 1.0 if actor_update == "expected_advantage" else float(result.td_error)
        assert abs(signal) > 0.01
        for actual, control, trace in zip(
            _params(result.state, nonlinear), _params(ablated.state, nonlinear), traces, strict=True
        ):
            expected = (
                0.1 * signal * agent.config.gamma * agent.config.actor_lamda * np.asarray(trace)
            )
            np.testing.assert_allclose(actual - control, expected, rtol=1e-5, atol=1e-7)
        if terminated:
            assert float(result.target) == 1.0
            for trace in _traces(result.state, nonlinear):
                np.testing.assert_array_equal(trace, jnp.zeros_like(trace))


@pytest.mark.parametrize("nonlinear", [False, True])
def test_rejected_terminal_preserves_eligibility_for_retry(nonlinear: bool) -> None:
    agent, state = _fixture(nonlinear, "expected_sarsa", "td_error")
    rejected = agent.update(state, jnp.float32(jnp.nan), jnp.ones((2,)), jnp.bool_(True))
    assert not bool(rejected.update_applied)
    for old, new in zip(jax.tree.leaves(state), jax.tree.leaves(rejected.state), strict=True):
        if hasattr(old, "dtype") and jax.dtypes.issubdtype(old.dtype, jax.dtypes.prng_key):
            old, new = jr.key_data(old), jr.key_data(new)
        np.testing.assert_array_equal(old, new)
    retry = agent.update(rejected.state, jnp.float32(1.0), jnp.ones((2,)), jnp.bool_(True))
    assert bool(retry.update_applied)
    # A fresh episode must not carry the previous episode's terminal trace.
    for trace in _traces(retry.state, nonlinear):
        np.testing.assert_array_equal(trace, jnp.zeros_like(trace))
