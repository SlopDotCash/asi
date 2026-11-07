"""Prototype behavior gradients must support the zero-option control surface."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeTransition,
    _reduction_safe_l2_norm,
)
from alberta_framework.core.representation_gradient_mixer import (
    RepresentationGradientMixerConfig,
)
from alberta_framework.core.state_builder import OnlineGatedStateBuilderConfig

pytestmark = pytest.mark.unit


def _agent() -> PrototypeAgent:
    builder = OnlineGatedStateBuilderConfig(
        observation_dim=2,
        n_actions=1,
        hidden_dim=1,
        step_size=0.01,
    )
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=(),
                    observation_dim=builder.feature_dim(),
                    n_primitive_actions=1,
                    base_step_size=0.1,
                    base_avg_reward_step_size=0.0,
                    epsilon_base=0.0,
                )
            ),
            state_builder=builder,
            representation_gradient_mixer=RepresentationGradientMixerConfig(
                representation_dim=builder.feature_dim(),
                mode="behavior_only",
            ),
        )
    )


def _transition(state: PrototypeAgentState) -> PrototypeTransition:
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(1.0, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        terminated=jnp.asarray(False),
        truncated=jnp.asarray(False),
        next_observation=jnp.asarray([0.5, -0.25], dtype=jnp.float32),
        next_decision_observation=jnp.asarray([0.5, -0.25], dtype=jnp.float32),
    )


def test_behavior_only_mixer_updates_without_an_option_bank() -> None:
    agent = _agent()
    state = agent.start(
        agent.init(jr.key(310, impl="threefry2x32")),
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
    )

    result = agent.update_transition(state, _transition(state))

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.behavior_gradient_result.valid)
    assert bool(result.representation_gradient_mix.applied)
    assert bool(result.state_builder_learning_diagnostics.applied)
    assert int(result.state.state_builder_state.update_count) == 1
    assert np.isfinite(np.asarray(result.behavior_representation_gradient)).all()


def test_compiled_behavior_only_mixer_updates_without_an_option_bank() -> None:
    agent = _agent()
    state = agent.start(
        agent.init(jr.key(311, impl="threefry2x32")),
        jnp.asarray([0.25, 0.75], dtype=jnp.float32),
    )

    result = jax.jit(agent.update_transition)(state, _transition(state))

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.behavior_gradient_result.valid)
    assert bool(result.representation_gradient_mix.applied)
    assert bool(result.state_builder_learning_diagnostics.applied)
    assert int(result.state.state_builder_state.update_count) == 1
    assert np.isfinite(np.asarray(result.behavior_representation_gradient)).all()


def test_behavior_gradient_keeps_representable_norm_when_square_sum_overflows() -> None:
    builder = OnlineGatedStateBuilderConfig(
        observation_dim=8,
        n_actions=1,
        hidden_dim=1,
        step_size=0.01,
    )
    agent = PrototypeAgent(
        PrototypeAgentConfig(
            oak=OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=(),
                    observation_dim=builder.feature_dim(),
                    n_primitive_actions=1,
                    base_step_size=0.1,
                    base_avg_reward_step_size=0.0,
                    epsilon_base=0.0,
                )
            ),
            state_builder=builder,
            representation_gradient_mixer=RepresentationGradientMixerConfig(
                representation_dim=builder.feature_dim(),
                mode="behavior_only",
            ),
        )
    )
    state = agent.start(
        agent.init(jr.key(312, impl="threefry2x32")),
        jnp.zeros((builder.observation_dim,), dtype=jnp.float32),
    )
    stomp_state = state.oak_state.stomp_state
    base_state = stomp_state.base_learner_state
    component = np.float32(7e18)
    weights = jnp.asarray(
        [[component] * builder.observation_dim + [np.float32(0.0)]],
        dtype=jnp.float32,
    )
    base_state = base_state.replace(
        head_params=base_state.head_params.replace(weights=(weights,))
    )
    state = state.replace(
        oak_state=state.oak_state.replace(
            stomp_state=stomp_state.replace(base_learner_state=base_state)
        )
    )
    expected_gradient = np.asarray(
        [-component] * builder.observation_dim + [np.float32(0.0)],
        dtype=np.float32,
    )
    expected_norm = float(np.linalg.norm(expected_gradient.astype(np.float64)))
    assert expected_norm < np.finfo(np.float32).max
    assert not np.isfinite(float(jnp.linalg.norm(jnp.asarray(expected_gradient))))
    transition = PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(1.0, dtype=jnp.float32),
        discount=jnp.asarray(0.0, dtype=jnp.float32),
        terminated=jnp.asarray(True),
        truncated=jnp.asarray(False),
        next_observation=jnp.zeros((builder.observation_dim,), dtype=jnp.float32),
        next_decision_observation=jnp.zeros(
            (builder.observation_dim,), dtype=jnp.float32
        ),
    )

    result = agent.update_transition(state, transition)

    assert bool(result.transition_diagnostics.valid)
    assert bool(result.behavior_gradient_result.diagnostics.gradient_finite)
    assert bool(result.behavior_gradient_result.valid)
    np.testing.assert_allclose(
        np.asarray(result.behavior_representation_gradient),
        expected_gradient,
        rtol=1e-6,
    )
    assert float(result.behavior_gradient_result.diagnostics.gradient_norm) == pytest.approx(
        expected_norm,
        rel=1e-6,
    )
    assert bool(result.representation_gradient_mix.applied)
    assert bool(result.state_builder_learning_diagnostics.applied)


def _unit_weight_state(agent: PrototypeAgent) -> PrototypeAgentState:
    """Plant a unit weight on the first representation feature."""
    state = agent.start(
        agent.init(jr.key(319, impl="threefry2x32")),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    stomp_state = state.oak_state.stomp_state
    base_state = stomp_state.base_learner_state
    weights = jnp.zeros((1, agent._base_representation_dim()), dtype=jnp.float32)
    weights = weights.at[0, 0].set(jnp.float32(1.0))
    base_state = base_state.replace(head_params=base_state.head_params.replace(weights=(weights,)))
    return state.replace(
        oak_state=state.oak_state.replace(
            stomp_state=stomp_state.replace(base_learner_state=base_state)
        )
    )


def _terminal_reward(state: PrototypeAgentState, reward: np.float32) -> PrototypeTransition:
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(0.0, dtype=jnp.float32),
        terminated=jnp.asarray(True),
        truncated=jnp.asarray(False),
        next_observation=jnp.zeros((2,), dtype=jnp.float32),
        next_decision_observation=jnp.zeros((2,), dtype=jnp.float32),
    )


def _recovered_half_square(error: np.float32) -> np.float32:
    half = np.float32(0.5)
    with np.errstate(over="ignore"):
        direct = half * (error * error)
        mantissa, exponent = np.frexp(error)
        recovered = np.ldexp(half * (mantissa * mantissa), int(exponent) + int(exponent))
    if np.isfinite(direct) or not np.isfinite(recovered):
        return np.float32(direct)
    return np.float32(recovered)


def test_behavior_gradient_keeps_direct_half_square_loss() -> None:
    """A modest terminal reward keeps the ordinary squared loss bit for bit."""
    agent = _agent()
    state = _unit_weight_state(agent)
    reward = np.float32(1.5)
    result = agent.update_transition(state, _terminal_reward(state, reward))
    loss = np.float32(result.behavior_gradient_result.diagnostics.loss)
    assert loss == _recovered_half_square(reward)
    expected = np.array([-reward, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(
        np.asarray(result.behavior_representation_gradient),
        expected,
    )


@pytest.mark.parametrize("sign", [np.float32(1.0), np.float32(-1.0)])
def test_behavior_gradient_keeps_representable_half_square_loss(sign: np.float32) -> None:
    """``0.5 * error**2`` stays finite when only the raw square overflows.

    A terminal reward of ``2e19`` is the TD error when the prediction is zero.
    Its square exceeds float32, while half of that square is about ``2e38``.
    The gradient is ``-error`` on the planted unit weight.
    """
    agent = _agent()
    state = _unit_weight_state(agent)
    reward = sign * np.float32(2.0e19)
    with np.errstate(over="ignore"):
        raw_square = reward * reward
    assert not np.isfinite(raw_square)
    assert np.isfinite(_recovered_half_square(reward))
    transition = _terminal_reward(state, reward)

    result = agent.update_transition(state, transition)
    compiled = jax.jit(agent.update_transition)(state, transition)
    diagnostics = result.behavior_gradient_result.diagnostics
    assert bool(result.behavior_gradient_result.valid)
    assert bool(diagnostics.loss_finite)
    assert np.float32(diagnostics.loss) == _recovered_half_square(reward)
    expected = np.array([-reward, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(result.behavior_representation_gradient), expected)
    np.testing.assert_array_equal(
        np.asarray(compiled.behavior_representation_gradient),
        expected,
    )
    assert bool(compiled.behavior_gradient_result.valid)


def test_behavior_gradient_rejects_unrepresentable_half_square_loss() -> None:
    """A half-square past float32 still rejects the behavior gradient."""
    agent = _agent()
    state = _unit_weight_state(agent)
    reward = np.float32(1.0e20)
    assert not np.isfinite(_recovered_half_square(reward))
    result = agent.update_transition(state, _terminal_reward(state, reward))
    assert bool(result.behavior_gradient_result.valid) is False
    assert bool(result.behavior_gradient_result.diagnostics.loss_finite) is False
    np.testing.assert_array_equal(
        np.asarray(result.behavior_representation_gradient),
        np.zeros((3,), dtype=np.float32),
    )


def test_behavior_gradient_norm_keeps_true_overflow_nonfinite() -> None:
    maximum = np.finfo(np.float32).max
    gradient = jnp.asarray([maximum, maximum], dtype=jnp.float32)

    assert np.isfinite(np.asarray(gradient)).all()
    assert not np.isfinite(float(_reduction_safe_l2_norm(gradient)))
