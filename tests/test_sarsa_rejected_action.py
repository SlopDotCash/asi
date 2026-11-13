"""Tests that a rejected SARSA update reports the prior action, not -1.

A rejected update rolls the state back to the incoming state, which keeps
its prior action. ``SARSAUpdateResult.action`` must agree with
``result.state.last_action``: the recorded action trajectory otherwise
diverges from the action the agent state credits in learning. This is the
same defect class #3672 fixed in ``ActorCriticAgent.update`` and #3564 fixed
in ``DifferentialSARSAAgent.update``; plain ``SARSAAgent.update`` still
returns the -1 refusal sentinel on the rejected path.
"""

import chex
import jax.numpy as jnp
import jax.random as jr

from alberta_framework import (
    SARSAAgent,
    SARSAConfig,
    run_sarsa_from_arrays,
)


def _make_agent(
    n_actions: int = 2,
    hidden_sizes: tuple[int, ...] = (),
    gamma: float = 0.99,
    epsilon_start: float = 0.0,
    **kwargs,
) -> SARSAAgent:
    """Helper to create a simple SARSA agent for tests."""
    config = SARSAConfig(
        n_actions=n_actions,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=kwargs.pop("epsilon_end", 0.01),
        epsilon_decay_steps=kwargs.pop("epsilon_decay_steps", 0),
    )
    return SARSAAgent(
        sarsa_config=config,
        hidden_sizes=hidden_sizes,
        sparsity=0.0,
        **kwargs,
    )


def _primed_state(agent: SARSAAgent, last_action: int) -> object:
    observation = jnp.ones(3, dtype=jnp.float32)
    return agent.init(3, jr.key(0)).replace(
        last_action=jnp.asarray(last_action, dtype=jnp.int32),
        last_observation=observation,
    )


def _assert_state_exact_ignoring_host_time(actual, expected) -> None:
    """Compare two SARSA states exactly, ignoring float32-narrowed host clocks.

    Same pattern as ``test_sarsa._assert_sarsa_state_exact_ignoring_host_time``:
    the state passes through ``jax.lax.cond`` in ``update``, which narrows
    ``birth_timestamp`` to float32 (see open #3719 for the runner form).
    """
    actual_learner = actual.learner_state.replace(birth_timestamp=0.0, uptime_s=0.0)
    expected_learner = expected.learner_state.replace(birth_timestamp=0.0, uptime_s=0.0)
    chex.assert_trees_all_equal(
        actual.replace(learner_state=actual_learner),
        expected.replace(learner_state=expected_learner),
    )


def test_rejected_update_reports_prior_action() -> None:
    """A NaN reward rejects the update but the prior action stays 1."""
    agent = _make_agent()
    state = _primed_state(agent, last_action=1)
    observation = jnp.ones(3, dtype=jnp.float32)
    result = agent.update(
        state,
        jnp.asarray(float("nan"), dtype=jnp.float32),
        observation,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    _assert_state_exact_ignoring_host_time(result.state, state)
    assert int(result.state.last_action) == 1
    assert int(result.action) == 1
    assert not bool(result.update_applied)


def test_rejected_update_with_invalid_next_action_reports_prior_action() -> None:
    """An invalid next action rejects the update; the prior action is kept."""
    agent = _make_agent()
    state = _primed_state(agent, last_action=1)
    observation = jnp.ones(3, dtype=jnp.float32)
    result = agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        observation,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(-1, dtype=jnp.int32),
    )
    _assert_state_exact_ignoring_host_time(result.state, state)
    assert int(result.action) == 1


def test_rejected_update_without_prior_action_keeps_refusal_sentinel() -> None:
    """With no prior action (last_action == -1) the -1 sentinel stays."""
    agent = _make_agent()
    state = agent.init(3, jr.key(0))
    assert int(state.last_action) == -1
    observation = jnp.ones(3, dtype=jnp.float32)
    result = agent.update(
        state,
        jnp.asarray(float("nan"), dtype=jnp.float32),
        observation,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert int(result.action) == -1


def test_accepted_update_reports_next_action() -> None:
    """The accepted path is unchanged: the selected next action is reported."""
    agent = _make_agent()
    state = _primed_state(agent, last_action=1)
    observation = jnp.ones(3, dtype=jnp.float32)
    result = agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        observation,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert int(result.state.last_action) == 0
    assert int(result.action) == 0
    assert bool(result.update_applied)


def test_array_runner_records_prior_action_on_rejected_row() -> None:
    """``run_sarsa_from_arrays`` records the prior action for a rejected row."""
    agent = _make_agent()
    state = agent.init(3, jr.key(0)).replace(
        last_action=jnp.asarray(1, dtype=jnp.int32),
        last_observation=jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32),
    )
    observations = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32
    )
    next_observations = jnp.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=jnp.float32
    )
    rewards = jnp.asarray([float("nan"), 1.0, 1.0], dtype=jnp.float32)
    terminated = jnp.zeros(3, dtype=jnp.float32)
    result = run_sarsa_from_arrays(
        agent, state, observations, rewards, terminated, next_observations
    )
    # Row 0 is rejected (NaN reward): the recorded action is the prior
    # action 1, not the -1 refusal sentinel. Rows 1-2 apply and report
    # valid selected actions.
    assert int(result.actions[0]) == 1
    assert all(int(a) >= 0 for a in result.actions)
