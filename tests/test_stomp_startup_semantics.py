"""Initial action dispatch and first intra-option credit regressions."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.options import STOMPAgent, STOMPConfig, SubtaskSpec


def test_starting_nonzero_option_dispatches_and_credits_its_own_action() -> None:
    """Option k>0 supplies both the first command and its subsequent credit."""
    config = STOMPConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=1.0e6,
                pseudo_reward_scale=10.0,
                max_option_steps=8,
            ),
            SubtaskSpec(
                feature_index=1,
                threshold=1.0e6,
                pseudo_reward_scale=10.0,
                max_option_steps=8,
            ),
        ),
        observation_dim=2,
        n_primitive_actions=2,
        epsilon_base=0.0,
        epsilon_option=0.0,
    )
    agent = STOMPAgent(config)
    initial_obs = jnp.array([1.0, 0.0], dtype=jnp.float32)
    state = agent.init(jr.key(0))

    # Force the base to choose extended action o_1 (index 3).
    base_weights = (
        jnp.array([[-2.0, 0.0]], dtype=jnp.float32),
        jnp.array([[-1.0, 0.0]], dtype=jnp.float32),
        jnp.array([[1.0, 0.0]], dtype=jnp.float32),
        jnp.array([[10.0, 0.0]], dtype=jnp.float32),
    )
    base_state = state.base_learner_state.replace(
        head_params=state.base_learner_state.head_params.replace(
            weights=base_weights,
            biases=tuple(jnp.zeros(1, dtype=jnp.float32) for _ in base_weights),
        )
    )

    # Option 0 prefers primitive 0 and option 1 prefers primitive 1, making an
    # incorrect idle-option dispatch observable.
    option_weights = jnp.zeros((2, 2, 2), dtype=jnp.float32)
    option_weights = option_weights.at[0, 0, 0].set(5.0)
    option_weights = option_weights.at[1, 1, 0].set(5.0)
    state = state.replace(
        base_learner_state=base_state,
        option_policies=state.option_policies.replace(q_weights=option_weights),
    )

    started = agent.start_with_action(state, initial_obs)

    assert int(started.state.base_last_action) == 3
    assert int(started.state.executing_option) == 1
    assert int(started.primitive_action) == 1
    assert int(started.state.last_primitive_action) == 1
    assert int(started.state.option_last_intra_action) == 1
    np.testing.assert_array_equal(
        np.asarray(started.state.option_start_obs), np.asarray(initial_obs)
    )

    before = np.asarray(started.state.option_policies.q_weights).copy()
    result = agent.update(
        started.state,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array([1.0, 1.0], dtype=jnp.float32),
    )
    after = np.asarray(result.state.option_policies.q_weights)

    assert not bool(result.option_terminated)
    assert float(result.pseudo_reward) == 10.0
    # The first transition must credit option 1's dispatched primitive 1.
    assert not np.array_equal(after[1, 1], before[1, 1])
    np.testing.assert_array_equal(after[1, 0], before[1, 0])
    np.testing.assert_array_equal(after[0], before[0])


def test_primitive_start_action_is_recorded_for_first_real_update() -> None:
    config = STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),),
        observation_dim=2,
        n_primitive_actions=2,
        epsilon_base=0.0,
    )
    agent = STOMPAgent(config)
    initial_obs = jnp.array([1.0, 0.0], dtype=jnp.float32)
    state = agent.init(jr.key(1))
    base_weights = (
        jnp.array([[-1.0, 0.0]], dtype=jnp.float32),
        jnp.array([[5.0, 0.0]], dtype=jnp.float32),
        jnp.array([[0.0, 0.0]], dtype=jnp.float32),
    )
    state = state.replace(
        base_learner_state=state.base_learner_state.replace(
            head_params=state.base_learner_state.head_params.replace(
                weights=base_weights,
                biases=tuple(jnp.zeros(1, dtype=jnp.float32) for _ in base_weights),
            )
        )
    )

    started = agent.start_with_action(state, initial_obs)

    assert int(started.primitive_action) == 1
    assert int(started.state.base_last_action) == 1
    assert int(started.state.executing_option) == -1
    assert int(agent.current_primitive_action(started.state)) == 1
