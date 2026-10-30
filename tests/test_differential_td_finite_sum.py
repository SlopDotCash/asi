"""Finite differential-TD cancellation for the average-reward learners."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.average_reward import (
    DifferentialGTDConfig,
    DifferentialGTDLearner,
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    DifferentialTDConfig,
    DifferentialTDLearner,
)

_MAGNITUDE = np.float32(2.0e38)


def _td_learner(step_size: float, average_reward_step_size: float = 0.01) -> DifferentialTDLearner:
    return DifferentialTDLearner(
        DifferentialTDConfig(
            step_size=step_size,
            average_reward_step_size=average_reward_step_size,
            trace_decay=0.0,
        )
    )


def _gtd_learner(
    value_step_size: float,
    average_reward_step_size: float = 0.01,
) -> DifferentialGTDLearner:
    return DifferentialGTDLearner(
        DifferentialGTDConfig(
            value_step_size=value_step_size,
            secondary_step_size=0.01,
            average_reward_step_size=average_reward_step_size,
            trace_decay=0.0,
        )
    )


def _sarsa_agent(
    q_step_size: float,
    average_reward_step_size: float = 0.01,
) -> DifferentialSARSAAgent:
    return DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=1,
            q_step_size=q_step_size,
            average_reward_step_size=average_reward_step_size,
            trace_decay=0.0,
            epsilon_decay_steps=0,
            use_bias=False,
        )
    )


def _feature(value: float) -> jax.Array:
    return jnp.asarray([value], dtype=jnp.float32)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_overflowing_differential_sum_keeps_the_td_step(sign: float) -> None:
    """``R - rbar + V(s')`` overflows; cancelling it against ``V(s)`` leaves ``R``.

    A fresh trace steps the weight and the bias by ``alpha * R`` and the
    reward rate by ``beta * R``.
    """
    magnitude = np.float32(sign * _MAGNITUDE)
    learner = _td_learner(0.1)
    alpha = np.float32(learner.config.step_size)
    beta = np.float32(learner.config.average_reward_step_size)
    state = learner.init(1).replace(
        weights=jnp.asarray([magnitude], dtype=jnp.float32),
    )
    reward = jnp.asarray(magnitude, dtype=jnp.float32)
    feature = _feature(1.0)
    expected_td = magnitude
    expected_weight = np.float32(magnitude + alpha * expected_td)
    expected_bias = np.float32(alpha * expected_td)
    expected_rbar = np.float32(beta * expected_td)

    with jax.disable_jit():
        eager = learner.update(state, feature, reward, feature)
    compiled = jax.jit(learner.update)(state, feature, reward, feature)

    for result in (eager, compiled):
        assert bool(result.update_applied)
        np.testing.assert_array_equal(np.asarray(result.td_error), expected_td)
        np.testing.assert_array_equal(np.asarray(result.state.weights), expected_weight)
        np.testing.assert_array_equal(np.asarray(result.state.bias), expected_bias)
        np.testing.assert_array_equal(np.asarray(result.state.average_reward), expected_rbar)
        np.testing.assert_array_equal(np.asarray(result.state.eligibility_traces), feature)
        assert float(result.state.bias_eligibility_trace) == 1.0
        assert int(result.state.step_count) == 1


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_overflowing_differential_sum_keeps_the_gtd_step(sign: float) -> None:
    """On-policy differential GTD has the same cancelling reward sum."""
    magnitude = np.float32(sign * _MAGNITUDE)
    learner = _gtd_learner(0.1)
    alpha = np.float32(learner.config.value_step_size)
    secondary = np.float32(learner.config.secondary_step_size)
    beta = np.float32(learner.config.average_reward_step_size)
    state = learner.init(1).replace(
        weights=jnp.asarray([magnitude], dtype=jnp.float32),
    )
    reward = jnp.asarray(magnitude, dtype=jnp.float32)
    feature = _feature(1.0)
    rho = jnp.asarray(1.0, dtype=jnp.float32)
    expected_td = magnitude
    expected_weight = np.float32(magnitude + alpha * expected_td)
    expected_bias = np.float32(alpha * expected_td)
    expected_secondary = np.float32(secondary * expected_td)
    expected_rbar = np.float32(beta * expected_td)

    with jax.disable_jit():
        eager = learner.update(state, feature, reward, feature, rho)
    compiled = jax.jit(learner.update)(state, feature, reward, feature, rho)

    for result in (eager, compiled):
        assert bool(result.update_applied)
        np.testing.assert_array_equal(np.asarray(result.td_error), expected_td)
        np.testing.assert_array_equal(np.asarray(result.state.weights), expected_weight)
        np.testing.assert_array_equal(np.asarray(result.state.bias), expected_bias)
        np.testing.assert_array_equal(
            np.asarray(result.state.secondary_weights),
            expected_secondary,
        )
        np.testing.assert_array_equal(np.asarray(result.state.secondary_bias), expected_secondary)
        np.testing.assert_array_equal(np.asarray(result.state.average_reward), expected_rbar)
        assert int(result.state.step_count) == 1


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_overflowing_differential_sum_keeps_the_sarsa_step(sign: float) -> None:
    """Differential SARSA drops the same sum before ``Q(s, a)`` can cancel it."""
    magnitude = np.float32(sign * _MAGNITUDE)
    agent = _sarsa_agent(0.1)
    alpha = np.float32(agent.config.q_step_size)
    beta = np.float32(agent.config.average_reward_step_size)
    feature = _feature(1.0)
    state = agent.init(1, jr.key(0)).replace(
        q_weights=jnp.asarray([[magnitude]], dtype=jnp.float32),
        last_observation=feature,
        last_action=jnp.int32(0),
    )
    reward = jnp.asarray(magnitude, dtype=jnp.float32)
    expected_td = magnitude
    expected_weight = np.float32(magnitude + alpha * expected_td)
    expected_rbar = np.float32(beta * expected_td)

    with jax.disable_jit():
        eager = agent.update(state, reward, feature, jnp.int32(0), discount=jnp.float32(1.0))
    compiled = jax.jit(agent.update)(
        state,
        reward,
        feature,
        jnp.int32(0),
        discount=jnp.float32(1.0),
    )

    for result in (eager, compiled):
        assert bool(result.update_applied)
        np.testing.assert_array_equal(np.asarray(result.td_error), expected_td)
        np.testing.assert_array_equal(np.asarray(result.state.q_weights), [[expected_weight]])
        np.testing.assert_array_equal(np.asarray(result.state.q_bias), np.float32(0.0))
        np.testing.assert_array_equal(np.asarray(result.state.average_reward), expected_rbar)
        assert int(result.state.step_count) == 1
        assert int(result.state.last_action) == 0


def test_finite_differential_sum_keeps_left_to_right_rounding() -> None:
    """A finite ``(R - rbar + V(s')) - V(s)`` keeps that association."""
    reward = np.float32(0.1)
    prediction = np.float32(1.0e10)
    baseline = np.float32(0.25)
    direct = np.float32(np.float32(reward - baseline) + prediction)
    expected_td = np.float32(direct - prediction)
    learner = _td_learner(0.1, average_reward_step_size=0.0)
    alpha = np.float32(learner.config.step_size)
    state = learner.init(1).replace(
        weights=jnp.asarray([prediction], dtype=jnp.float32),
        average_reward=jnp.asarray(baseline, dtype=jnp.float32),
    )
    feature = _feature(1.0)
    result = learner.update(
        state,
        feature,
        jnp.asarray(reward, dtype=jnp.float32),
        feature,
    )
    diagnostic = learner.td_error(
        state,
        feature,
        jnp.asarray(reward, dtype=jnp.float32),
        feature,
    )

    assert bool(result.update_applied)
    np.testing.assert_array_equal(np.asarray(diagnostic), expected_td)
    np.testing.assert_array_equal(np.asarray(result.td_error), expected_td)
    np.testing.assert_array_equal(
        np.asarray(result.state.weights),
        np.float32(prediction + alpha * expected_td),
    )
    np.testing.assert_array_equal(np.asarray(result.state.bias), np.float32(alpha * expected_td))
    np.testing.assert_array_equal(np.asarray(result.state.average_reward), baseline)


@pytest.mark.parametrize(
    ("next_feature", "step_size"),
    [(0.0, 0.1), (1.0, 1.0)],
)
def test_unrepresentable_differential_step_still_rejects(
    next_feature: float,
    step_size: float,
) -> None:
    """Reject a TD error or value step that is still outside float32."""
    magnitude = _MAGNITUDE
    weight = np.float32(-magnitude if next_feature == 0.0 else magnitude)
    learner = _td_learner(step_size)
    state = learner.init(1).replace(weights=jnp.asarray([weight], dtype=jnp.float32))
    reward = jnp.asarray(magnitude, dtype=jnp.float32)

    with jax.disable_jit():
        eager = learner.update(state, _feature(1.0), reward, _feature(next_feature))
    compiled = jax.jit(learner.update)(state, _feature(1.0), reward, _feature(next_feature))

    for result in (eager, compiled):
        assert not bool(result.update_applied)
        np.testing.assert_array_equal(np.asarray(result.td_error), np.float32(0.0))
        np.testing.assert_array_equal(np.asarray(result.state.weights), weight)
        assert float(result.state.bias) == 0.0
        assert float(result.state.average_reward) == 0.0
        assert int(result.state.step_count) == 0
