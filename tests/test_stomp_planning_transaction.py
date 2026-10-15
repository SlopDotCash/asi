"""Invalid imagined targets must roll back the complete STOMP transition."""

import functools

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.options import STOMPAgent, STOMPConfig, SubtaskSpec


@functools.cache
def _case(backups=1, n_options=1):
    agent = STOMPAgent(
        STOMPConfig(
            subtask_specs=tuple(
                SubtaskSpec(feature_index=0, threshold=10.0) for _ in range(n_options)
            ),
            observation_dim=2,
            n_primitive_actions=1,
            base_step_size=0.1,
            base_avg_reward_step_size=0.1,
            option_planning_backups_per_step=backups,
            epsilon_base=0.0,
        )
    )
    state = agent.init(jr.key(270, impl="threefry2x32"))
    state = state.replace(
        base_learner_state=state.base_learner_state.replace(
            head_params=state.base_learner_state.head_params.replace(
                weights=tuple(jnp.zeros((1, 2)) for _ in range(1 + n_options)),
                biases=tuple(jnp.zeros(1) for _ in range(1 + n_options)),
            ),
            birth_timestamp=jnp.float32(0.0),
            uptime_s=jnp.float32(0.0),
        ),
        base_last_obs=jnp.ones(2),
        option_models=state.option_models.replace(
            env_return_ema=jnp.ones(n_options),
            duration_ema=jnp.ones(n_options),
            baseline_mass_ema=jnp.ones(n_options),
            discount_ema=jnp.full(n_options, 0.5),
            n_completions=jnp.ones(n_options, dtype=jnp.int32),
            next_state_weights=jnp.full((n_options, 2, 2), 3.0e38),
        ),
    )
    assert bool(agent.state_valid(state))
    return agent, state


def _assert_equal(actual, expected, *, ignore_uptime_s=False):
    actual_leaves = jax.tree_util.tree_flatten_with_path(actual)[0]
    expected_leaves = jax.tree_util.tree_flatten_with_path(expected)[0]
    for (actual_path, a), (expected_path, b) in zip(
        actual_leaves, expected_leaves, strict=True
    ):
        assert actual_path == expected_path
        if ignore_uptime_s and getattr(actual_path[-1], "name", None) == "uptime_s":
            continue
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jr.key_data(a), jr.key_data(b)
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backups", [1, 3])
@pytest.mark.parametrize("active", [False, True])
def test_nan_planning_target_cannot_commit_or_advance_lifetime(backups, active):
    agent, state = _case(backups)
    if active:
        state = state.replace(
            base_last_action=jnp.int32(1),
            executing_option=jnp.int32(0),
            option_start_obs=jnp.ones(2),
        )
    assert bool(agent.state_valid(state))
    anchor = jnp.ones(2)
    predicted_next = anchor + state.option_models.next_state_weights[0] @ anchor
    assert bool(jnp.all(jnp.isinf(predicted_next)))
    assert bool(jnp.all(jnp.isnan(agent.base_q_values(state, predicted_next))))
    result = agent.update(state, jnp.float32(1.0), anchor, jnp.float32(0.5))
    assert bool(result.inputs_valid)
    assert bool(result.proposed_state_valid)
    assert not bool(result.update_applied)
    assert int(result.planning_backups) == 0
    assert int(result.nested_updates_applied) == 0
    assert float(result.planning_td_error) == 0.0
    _assert_equal(result.state, state)


@pytest.mark.parametrize("bad_model", [0, 1])
def test_one_bad_backup_rolls_back_real_learning_and_other_backups(bad_model):
    agent, state = _case(backups=2, n_options=2)
    state = state.replace(
        option_models=state.option_models.replace(
            next_state_weights=jnp.zeros((2, 2, 2)).at[bad_model].set(3.0e38),
        )
    )
    result = agent.update(state, jnp.float32(1.0), jnp.ones(2), jnp.float32(0.5))
    assert not bool(result.update_applied)
    assert int(result.nested_updates_required) == 3
    assert int(result.nested_updates_applied) == 0
    assert int(result.planning_backups) == 0
    _assert_equal(result.state, state)


@pytest.mark.parametrize("disabled", ["zero_budget", "masked", "uncompleted", "flag"])
def test_unrequested_bad_models_do_not_block_real_learning(disabled):
    agent, state = _case(backups=0 if disabled == "zero_budget" else 1)
    if disabled == "uncompleted":
        state = state.replace(
            option_models=state.option_models.replace(
                n_completions=jnp.zeros(1, dtype=jnp.int32),
            )
        )
    result = agent.update(
        state,
        jnp.float32(1.0),
        jnp.ones(2),
        jnp.float32(0.5),
        extended_action_mask=jnp.array([True, disabled != "masked"]),
        enable_planning=disabled != "flag",
    )
    assert bool(result.update_applied)
    assert int(result.planning_backups) == 0
    assert int(result.nested_updates_applied) == 1
    assert float(result.state.base_average_reward) == pytest.approx(0.1)
    assert float(result.state.base_learner_state.head_params.weights[0][0, 0]) == pytest.approx(0.1)


def test_rejected_scan_preserves_state_and_can_retry_after_model_repair():
    agent, state = _case()
    result = agent.scan(state, jnp.ones(3), jnp.ones((3, 2)), jnp.full(3, 0.5))
    np.testing.assert_array_equal(result.update_applied, [False, False, False])
    np.testing.assert_array_equal(result.planning_td_errors, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.nested_updates_applied, [0, 0, 0])
    _assert_equal(result.state, state, ignore_uptime_s=True)
    assert (
        result.state.base_learner_state.birth_timestamp
        == state.base_learner_state.birth_timestamp
    )
    assert result.state.base_learner_state.uptime_s > state.base_learner_state.uptime_s
    repaired = result.state.replace(
        option_models=result.state.option_models.replace(
            next_state_weights=jnp.zeros((1, 2, 2)),
        )
    )
    retry = agent.update(repaired, jnp.float32(1.0), jnp.ones(2), jnp.float32(0.5))
    direct = agent.update(
        state.replace(option_models=repaired.option_models),
        jnp.float32(1.0),
        jnp.ones(2),
        jnp.float32(0.5),
    )
    assert bool(retry.update_applied)
    assert int(retry.nested_updates_applied) == 2
    assert int(retry.planning_backups) == 1
    # The retry ran through the rejected scan first, so its host uptime carries
    # that attempt; everything else must match the direct update.
    _assert_equal(retry, direct, ignore_uptime_s=True)
