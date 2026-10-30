"""Finite TD cancellation for linear actor-critic critics."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.actor_critic import (
    ActorCriticAgent,
    ActorCriticConfig,
    ContinuousActorCriticAgent,
    ContinuousActorCriticConfig,
)


def _discrete_agent(critic_step_size: float) -> ActorCriticAgent:
    return ActorCriticAgent(
        ActorCriticConfig(
            n_actions=1,
            gamma=1.0,
            actor_step_size=0.1,
            critic_step_size=critic_step_size,
            actor_lamda=0.0,
            critic_lamda=0.0,
        )
    )


def _continuous_agent(critic_step_size: float) -> ContinuousActorCriticAgent:
    return ContinuousActorCriticAgent(
        ContinuousActorCriticConfig(
            action_dim=1,
            gamma=1.0,
            actor_step_size=0.001,
            critic_step_size=critic_step_size,
            actor_lamda=0.0,
            critic_lamda=0.0,
        )
    )


def _discount() -> jax.Array:
    return jnp.asarray(1.0, dtype=jnp.float32)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_overflowing_reward_sum_keeps_the_critic_step(sign: float) -> None:
    """``R + V(s')`` overflows; cancelling it against V(s) leaves ``delta = R``.

    A one-action softmax has a zero policy gradient, so only the critic moves.
    With a fresh trace the weight and bias each step by ``alpha * R``.
    """
    magnitude = np.float32(sign * 2.0e38)
    alpha = np.float32(0.1)
    agent = _discrete_agent(0.1)
    state = agent.init(1, jr.key(0)).replace(
        critic_weights=jnp.asarray([magnitude], dtype=jnp.float32),
        last_observation=jnp.ones((1,), dtype=jnp.float32),
        last_action=jnp.int32(0),
    )
    reward = jnp.asarray(magnitude, dtype=jnp.float32)
    observation = jnp.ones((1,), dtype=jnp.float32)
    expected_weight = np.float32(magnitude + alpha * magnitude)
    expected_bias = np.float32(alpha * magnitude)

    with jax.disable_jit():
        eager = agent.update(state, reward, observation, discount=_discount())
    compiled = jax.jit(agent.update)(state, reward, observation, discount=_discount())

    for result in (eager, compiled):
        assert bool(result.update_applied)
        np.testing.assert_array_equal(np.asarray(result.td_error), magnitude)
        np.testing.assert_array_equal(np.asarray(result.state.critic_weights), expected_weight)
        np.testing.assert_array_equal(np.asarray(result.state.critic_bias), expected_bias)
        np.testing.assert_array_equal(
            np.asarray(result.state.critic_trace_weights),
            np.ones((1,), dtype=np.float32),
        )
        assert float(result.state.critic_trace_bias) == 1.0
        np.testing.assert_array_equal(np.asarray(result.state.actor_weights), 0.0)
        assert int(result.state.step_count) == 1


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_overflowing_reward_sum_keeps_the_continuous_critic_step(sign: float) -> None:
    """The diagonal-Gaussian critic has the same cancelling reward sum."""
    magnitude = np.float32(sign * 2.0e38)
    alpha = np.float32(0.1)
    actor_step = np.float32(0.001)
    agent = _continuous_agent(0.1)
    state = agent.init(1, jr.key(0)).replace(
        critic_weights=jnp.asarray([magnitude], dtype=jnp.float32),
        last_observation=jnp.ones((1,), dtype=jnp.float32),
        last_sample=jnp.zeros((1,), dtype=jnp.float32),
    )
    reward = jnp.asarray(magnitude, dtype=jnp.float32)
    observation = jnp.ones((1,), dtype=jnp.float32)
    expected_weight = np.float32(magnitude + alpha * magnitude)
    expected_bias = np.float32(alpha * magnitude)
    # A matched sample has score zero on the mean and -1 on log-sigma.
    # The huge but finite TD error drives log-sigma to its clamp.
    raw_log_sigma = np.float32(np.float32(agent._config.log_sigma_init) + actor_step * -magnitude)
    expected_log_sigma = np.clip(
        raw_log_sigma,
        np.float32(agent._config.log_sigma_min),
        np.float32(agent._config.log_sigma_max),
    )

    with jax.disable_jit():
        eager = agent.update(state, reward, observation, discount=_discount())
    compiled = jax.jit(agent.update)(state, reward, observation, discount=_discount())

    for result in (eager, compiled):
        assert bool(result.update_applied)
        np.testing.assert_array_equal(np.asarray(result.td_error), magnitude)
        np.testing.assert_array_equal(np.asarray(result.state.critic_weights), expected_weight)
        np.testing.assert_array_equal(np.asarray(result.state.critic_bias), expected_bias)
        np.testing.assert_array_equal(np.asarray(result.state.log_sigma), expected_log_sigma)
        np.testing.assert_array_equal(np.asarray(result.state.mean_weights), 0.0)
        assert int(result.state.step_count) == 1


def test_finite_actor_critic_sum_keeps_left_to_right_rounding() -> None:
    """A finite ``(R + V(s')) - V(s)`` keeps that association."""
    reward = np.float32(0.1)
    prediction = np.float32(1.0e10)
    expected_td = np.float32((reward + prediction) - prediction)
    alpha = np.float32(0.1)
    agent = _discrete_agent(0.1)
    state = agent.init(1, jr.key(0)).replace(
        critic_weights=jnp.asarray([prediction], dtype=jnp.float32),
        last_observation=jnp.ones((1,), dtype=jnp.float32),
        last_action=jnp.int32(0),
    )
    result = agent.update(
        state,
        jnp.asarray(reward, dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        discount=_discount(),
    )
    assert bool(result.update_applied)
    np.testing.assert_array_equal(np.asarray(result.td_error), expected_td)
    np.testing.assert_array_equal(
        np.asarray(result.state.critic_weights),
        np.float32(prediction + alpha * expected_td),
    )
    np.testing.assert_array_equal(
        np.asarray(result.state.critic_bias),
        np.float32(alpha * expected_td),
    )


@pytest.mark.parametrize(
    ("next_feature", "critic_step_size"),
    [(0.0, 0.1), (1.0, 1.0)],
)
def test_unrepresentable_actor_critic_step_still_rejects(
    next_feature: float,
    critic_step_size: float,
) -> None:
    """Reject a TD error or critic step that is still outside float32."""
    magnitude = np.float32(2.0e38)
    weight = np.float32(-magnitude if next_feature == 0.0 else magnitude)
    agent = _discrete_agent(critic_step_size)
    state = agent.init(1, jr.key(0)).replace(
        critic_weights=jnp.asarray([weight], dtype=jnp.float32),
        last_observation=jnp.ones((1,), dtype=jnp.float32),
        last_action=jnp.int32(0),
    )
    observation = jnp.asarray([next_feature], dtype=jnp.float32)
    reward = jnp.asarray(magnitude, dtype=jnp.float32)

    with jax.disable_jit():
        eager = agent.update(state, reward, observation, discount=_discount())
    compiled = jax.jit(agent.update)(state, reward, observation, discount=_discount())

    for result in (eager, compiled):
        assert not bool(result.update_applied)
        np.testing.assert_array_equal(np.asarray(result.td_error), np.float32(0.0))
        np.testing.assert_array_equal(np.asarray(result.state.critic_weights), weight)
        assert float(result.state.critic_bias) == 0.0
        assert int(result.state.step_count) == 0
