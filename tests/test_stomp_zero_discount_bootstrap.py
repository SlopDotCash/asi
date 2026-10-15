"""Terminal STOMP updates must not depend on unused next-state Q values."""

import functools

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.options import STOMPAgent, STOMPConfig, SubtaskSpec


def _init(agent):
    state = agent.init(jr.key(270, impl="threefry2x32"))
    # Isolate numeric transaction equality from legacy host-clock float32 coercion.
    return state.replace(
        base_learner_state=state.base_learner_state.replace(
            birth_timestamp=jnp.float32(0.0),
            uptime_s=jnp.float32(0.0),
        )
    )


@functools.cache
def _primitive_case(n_options):
    specs = tuple(
        SubtaskSpec(feature_index=1, threshold=10.0, max_option_steps=3) for _ in range(n_options)
    )
    agent = STOMPAgent(
        STOMPConfig(
            subtask_specs=specs,
            observation_dim=2,
            n_primitive_actions=1,
            base_step_size=0.1,
            base_avg_reward_step_size=0.0,
            epsilon_base=0.0,
        )
    )
    observation = jnp.array([0.0, 1.0], dtype=jnp.float32)
    state = _init(agent)
    params = state.base_learner_state.head_params.replace(
        weights=tuple(jnp.array([[2.0, 0.0]]) for _ in range(1 + n_options)),
        biases=tuple(jnp.zeros(1) for _ in range(1 + n_options)),
    )
    state = state.replace(
        base_learner_state=state.base_learner_state.replace(head_params=params),
        base_last_obs=observation,
    )
    assert bool(agent.state_valid(state))
    return agent, state, observation


def _assert_equal(actual, expected, *, ignore_uptime_s=False):
    actual_leaves = jax.tree_util.tree_flatten_with_path(actual)[0]
    expected_leaves = jax.tree_util.tree_flatten_with_path(expected)[0]
    for (actual_path, a), (expected_path, b) in zip(actual_leaves, expected_leaves, strict=True):
        assert actual_path == expected_path
        if ignore_uptime_s and getattr(actual_path[-1], "name", None) == "uptime_s":
            continue
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jr.key_data(a), jr.key_data(b)
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n_options", [0, 1])
def test_terminal_primitive_learns_despite_unused_bootstrap_overflow(n_options):
    agent, state, observation = _primitive_case(n_options)
    next_observation = jnp.array([3.0e38, 0.0], dtype=jnp.float32)
    assert bool(
        jnp.all(jnp.isinf(agent.base_learner.predict(state.base_learner_state, next_observation)))
    )
    result = agent.update(
        state,
        jnp.float32(1.0),
        next_observation,
        jnp.float32(0.0),
        decision_observation=observation,
    )
    control = agent.update(
        state,
        jnp.float32(1.0),
        jnp.zeros(2),
        jnp.float32(0.0),
        decision_observation=observation,
    )
    assert bool(control.update_applied)
    assert bool(result.update_applied)
    assert float(result.td_error) == pytest.approx(1.0)
    assert float(result.state.base_learner_state.head_params.weights[0][0, 1]) == pytest.approx(0.1)
    _assert_equal(result, control)


@pytest.mark.parametrize("n_options", [0, 1])
@pytest.mark.parametrize("discount", [1.0e-4, 0.5])
def test_positive_discount_retains_overflow_rejection(n_options, discount):
    agent, state, observation = _primitive_case(n_options)
    result = agent.update(
        state,
        jnp.float32(1.0),
        jnp.array([3.0e38, 0.0]),
        jnp.float32(discount),
        decision_observation=observation,
        execution_boundary=jnp.array(True),
    )
    assert not bool(result.update_applied)
    _assert_equal(result.state, state)


@pytest.mark.parametrize("n_options", [0, 1])
@pytest.mark.parametrize(
    "bad_input", ["next_observation", "decision_observation", "reward", "state"]
)
def test_zero_discount_does_not_relax_finite_input_gates(n_options, bad_input):
    agent, state, observation = _primitive_case(n_options)
    next_observation = jnp.array([3.0e38, 0.0])
    decision_observation = observation
    reward = jnp.float32(1.0)
    if bad_input == "next_observation":
        next_observation = jnp.array([jnp.inf, 0.0])
    elif bad_input == "decision_observation":
        decision_observation = jnp.array([jnp.nan, 0.0])
    elif bad_input == "reward":
        reward = jnp.float32(jnp.inf)
    else:
        state = state.replace(base_average_reward=jnp.float32(jnp.nan))
    result = agent.update(
        state,
        reward,
        next_observation,
        jnp.float32(0.0),
        decision_observation=decision_observation,
    )
    assert not bool(result.update_applied)
    _assert_equal(result.state, state)


def test_terminal_scan_keeps_learning_and_matches_safe_bootstrap():
    agent, state, observation = _primitive_case(0)
    rewards = jnp.ones(4, dtype=jnp.float32)
    decisions = jnp.tile(observation, (4, 1))
    discounts = jnp.zeros(4, dtype=jnp.float32)
    result = agent.scan(
        state,
        rewards,
        jnp.tile(jnp.array([3.0e38, 0.0]), (4, 1)),
        discounts,
        decision_observations=decisions,
    )
    control = agent.scan(
        state,
        rewards,
        jnp.zeros((4, 2)),
        discounts,
        decision_observations=decisions,
    )
    assert bool(jnp.all(result.update_applied))
    np.testing.assert_allclose(result.td_errors, np.array([1.0, 0.8, 0.64, 0.512]), rtol=1e-6)
    _assert_equal(result, control, ignore_uptime_s=True)


@pytest.mark.parametrize(
    ("termination", "overflow"),
    [
        ("environment", "base"),
        ("environment", "intra"),
        ("environment", "both"),
        ("goal", "intra"),
        ("duration", "intra"),
    ],
)
def test_completed_option_ignores_unused_bootstrap(termination, overflow):
    agent = STOMPAgent(
        STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=1,
                    threshold=1.0 if termination == "goal" else 10.0,
                    max_option_steps=1 if termination == "duration" else 3,
                ),
            ),
            observation_dim=2,
            n_primitive_actions=1,
            base_step_size=0.1,
            base_avg_reward_step_size=0.0,
            option_step_size=0.1,
            option_avg_reward_step_size=0.0,
            epsilon_base=0.0,
        )
    )
    observation = jnp.array([0.0, 1.0], dtype=jnp.float32)
    state = _init(agent)
    state = state.replace(
        base_learner_state=state.base_learner_state.replace(
            head_params=state.base_learner_state.head_params.replace(
                weights=tuple(
                    jnp.array([[2.0 if overflow in ("base", "both") else 0.0, 0.0]])
                    for _ in range(2)
                ),
                biases=(jnp.zeros(1), jnp.zeros(1)),
            ),
        ),
        base_last_obs=observation,
        base_last_action=jnp.int32(1),
        executing_option=jnp.int32(0),
        option_start_obs=observation,
        option_policies=state.option_policies.replace(
            q_weights=jnp.array(
                [[[2.0 if overflow in ("intra", "both") else 0.0, 0.0]]],
                dtype=jnp.float32,
            ),
        ),
    )
    assert bool(agent.state_valid(state))
    result = agent.update(
        state,
        jnp.float32(1.0),
        jnp.array([3.0e38, 1.0]),
        jnp.float32(0.0 if termination == "environment" else 0.5),
        decision_observation=observation,
    )
    assert bool(result.update_applied)
    assert bool(result.option_terminated)
    assert float(result.pseudo_reward) == pytest.approx(1.0)
    assert float(result.state.option_policies.q_weights[0, 0, 1]) == pytest.approx(0.1)
    assert int(result.state.option_models.n_completions[0]) == 1


@pytest.mark.parametrize("model_discount", [0.0, 0.5])
def test_terminal_model_planning_ignores_unused_prediction(model_discount):
    agent = STOMPAgent(
        STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=1, threshold=10.0),),
            observation_dim=2,
            n_primitive_actions=1,
            base_step_size=0.1,
            base_avg_reward_step_size=0.0,
            option_planning_backups_per_step=1,
            epsilon_base=0.0,
        )
    )
    observation = jnp.array([0.0, 1.0], dtype=jnp.float32)
    state = _init(agent)
    state = state.replace(
        base_learner_state=state.base_learner_state.replace(
            head_params=state.base_learner_state.head_params.replace(
                weights=(jnp.array([[2.0, 0.0]]), jnp.array([[2.0, 0.0]])),
                biases=(jnp.zeros(1), jnp.zeros(1)),
            ),
        ),
        base_last_obs=observation,
        option_models=state.option_models.replace(
            env_return_ema=jnp.array([1.0]),
            duration_ema=jnp.array([1.0]),
            baseline_mass_ema=jnp.array([1.0]),
            discount_ema=jnp.array([model_discount]),
            n_completions=jnp.array([1], dtype=jnp.int32),
            next_state_weights=jnp.array([[[0.0, 3.0e38], [0.0, 0.0]]]),
        ),
    )
    assert bool(agent.state_valid(state))
    result = agent.update(state, jnp.float32(0.0), observation, jnp.float32(0.0))
    assert bool(result.update_applied) == (model_discount == 0.0)
    if model_discount == 0.0:
        assert int(result.planning_backups) == 1
        assert float(result.planning_td_error) == pytest.approx(1.0)
        assert int(result.state.base_learner_state.step_count) == 2
        assert float(result.state.base_learner_state.head_params.weights[1][0, 1]) == pytest.approx(
            0.1
        )
    else:
        assert int(result.state.step_count) == 0
        _assert_equal(result.state, state)
