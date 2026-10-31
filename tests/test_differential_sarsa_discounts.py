"""Per-transition discount regressions for differential SARSA and Step 9."""

from __future__ import annotations

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    run_differential_sarsa_from_arrays,
)
from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
from alberta_framework.steps.step9 import (
    Step9DreamingConfig,
    Step9DreamingState,
    init_step9_state,
    make_step9_components,
    step9_update,
)


@pytest.mark.parametrize(
    ("discount", "expected_td_error", "expected_weight_traces", "expected_bias_traces"),
    [
        (
            0.0,
            1.25,
            [[1.0, 2.0], [0.0, 0.0]],
            [1.0, 0.0],
        ),
        (
            0.4,
            3.05,
            [[1.2, 2.2], [0.4, 0.4]],
            [1.2, 0.4],
        ),
    ],
)
def test_differential_sarsa_discount_hand_calculation(
    discount: float,
    expected_td_error: float,
    expected_weight_traces: list[list[float]],
    expected_bias_traces: list[float],
) -> None:
    """A constant discount scales both the next-action bootstrap and old trace.

    The old trace decays by the carried discount into the current state, so
    the hand calculation stores the same value as the transition discount.
    """
    agent = DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=2,
            q_step_size=0.0,
            average_reward_step_size=0.0,
            trace_decay=0.5,
            epsilon_start=0.0,
        )
    )
    state = agent.init(2, jr.key(0)).replace(
        q_weights=jnp.array(
            [[2.0, -1.0], [1.0, 0.5]],
            dtype=jnp.float32,
        ),
        q_bias=jnp.array([0.5, -0.5], dtype=jnp.float32),
        q_trace_weights=jnp.array(
            [[1.0, 1.0], [2.0, 2.0]],
            dtype=jnp.float32,
        ),
        q_trace_bias=jnp.array([1.0, 2.0], dtype=jnp.float32),
        previous_discount=jnp.array(discount, dtype=jnp.float32),
        average_reward=jnp.array(0.25, dtype=jnp.float32),
        last_observation=jnp.array([1.0, 2.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )

    result = agent.update(
        state,
        jnp.array(2.0, dtype=jnp.float32),
        jnp.array([3.0, 4.0], dtype=jnp.float32),
        next_action=jnp.array(1, dtype=jnp.int32),
        discount=jnp.array(discount, dtype=jnp.float32),
    )

    chex.assert_trees_all_close(
        result.td_error,
        jnp.array(expected_td_error, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.state.q_trace_weights,
        jnp.array(expected_weight_traces, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.state.q_trace_bias,
        jnp.array(expected_bias_traces, dtype=jnp.float32),
    )


def test_differential_sarsa_array_runner_uses_transition_discounts() -> None:
    """The array runner forwards each discount; omission retains gamma=1."""
    agent = DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=1,
            q_step_size=0.0,
            average_reward_step_size=0.0,
            epsilon_start=0.0,
        )
    )
    state = agent.init(1, jr.key(1)).replace(
        q_weights=jnp.array([[2.0]], dtype=jnp.float32),
        last_observation=jnp.array([1.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
    )
    rewards = jnp.array([1.0, 1.0], dtype=jnp.float32)
    next_observations = jnp.array([[3.0], [4.0]], dtype=jnp.float32)

    discounted = run_differential_sarsa_from_arrays(
        agent,
        state,
        rewards,
        next_observations,
        discounts=jnp.array([0.0, 0.5], dtype=jnp.float32),
    )
    compatibility_default = run_differential_sarsa_from_arrays(
        agent,
        state,
        rewards,
        next_observations,
    )

    chex.assert_trees_all_close(
        discounted.td_errors,
        jnp.array([-1.0, -1.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        compatibility_default.td_errors,
        jnp.array([5.0, 3.0], dtype=jnp.float32),
    )


def _step9_config(*, model_gamma: float, planning_budget: int) -> Step9DreamingConfig:
    return Step9DreamingConfig(
        control=Step6DifferentialSARSAConfig(
            n_actions=1,
            q_step_size=0.0,
            average_reward_step_size=0.0,
            epsilon_start=0.0,
        ),
        observation_dim=1,
        n_actions=1,
        model_hidden_sizes=(),
        model_step_size=0.0,
        model_sparsity=0.0,
        model_use_layer_norm=False,
        model_gamma=model_gamma,
        dreaming_warmup_steps=0,
        dreaming_max_model_error=1e30,
        behavior_model_step_size=0.0,
        planning_budget=planning_budget,
        dream_rollout_horizon=1,
        dream_candidate_count=1,
        buffer_capacity=2,
    )


def _set_constant_model_discount(
    state: Step9DreamingState,
    discount: float,
) -> Step9DreamingState:
    learner_state = state.world_model_state.learner_state
    head_params = learner_state.head_params
    discount_head = len(head_params.biases) - 1
    head_params = head_params.replace(
        weights=tuple(jnp.zeros_like(weight) for weight in head_params.weights),
        biases=tuple(
            jnp.full_like(bias, discount if index == discount_head else 0.0)
            for index, bias in enumerate(head_params.biases)
        ),
    )
    learner_state = learner_state.replace(head_params=head_params)
    return state.replace(
        world_model_state=state.world_model_state.replace(
            learner_state=learner_state,
        )
    )


def _step9_state_with_constant_q(
    config: Step9DreamingConfig,
):
    agent, model, buffer = make_step9_components(config)
    state = init_step9_state(
        agent,
        model,
        buffer,
        key=jr.key(2),
        initial_observation=jnp.zeros(1, dtype=jnp.float32),
    )
    state = state.replace(
        control_state=state.control_state.replace(
            q_weights=jnp.zeros((1, 1), dtype=jnp.float32),
            q_bias=jnp.array([2.0], dtype=jnp.float32),
            average_reward=jnp.array(0.0, dtype=jnp.float32),
            last_action=jnp.array(0, dtype=jnp.int32),
        )
    )
    return agent, model, buffer, state


def test_step9_real_update_uses_model_gamma_as_transition_discount() -> None:
    config = _step9_config(model_gamma=0.25, planning_budget=0)
    agent, model, buffer, state = _step9_state_with_constant_q(config)

    result = step9_update(
        config,
        agent,
        model,
        buffer,
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
    )

    # 1 + 0.25 * Q(next=2) - Q(previous=2) = -0.5.
    chex.assert_trees_all_close(
        result.real_control_result.td_error,
        jnp.array(-0.5, dtype=jnp.float32),
    )


def test_step9_dream_target_changes_with_predicted_continuation() -> None:
    """Changing only the learned discount head changes the dream TD target."""
    config = _step9_config(model_gamma=0.9, planning_budget=1)
    agent, model, buffer, base_state = _step9_state_with_constant_q(config)

    zero_continuation = step9_update(
        config,
        agent,
        model,
        buffer,
        _set_constant_model_discount(base_state, 0.0),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
    )
    half_continuation = step9_update(
        config,
        agent,
        model,
        buffer,
        _set_constant_model_discount(base_state, 0.5),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
    )

    assert bool(zero_continuation.dream_accepted[0])
    assert bool(half_continuation.dream_accepted[0])
    # With reward/rbar=0 and Qprev=Qnext=2:
    # discount=0 -> -2, while discount=0.5 -> -1.
    chex.assert_trees_all_close(
        zero_continuation.dream_td_errors,
        jnp.array([-2.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        half_continuation.dream_td_errors,
        jnp.array([-1.0], dtype=jnp.float32),
    )
