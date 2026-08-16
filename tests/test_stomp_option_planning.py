"""Fixed-budget option-model planning regressions for Alberta Plan Step 10.

The added option-model return, duration, and baseline-mass fields change the
STOMP state PyTree. Generic Orbax checkpoints written against the older tree
are not guaranteed to restore; there is no versioned STOMP checkpoint
migration loader in this repository.
"""

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.options import STOMPAgent, STOMPConfig, SubtaskSpec
from alberta_framework.steps.step10 import Step10STOMPConfig
from alberta_framework.steps.step11 import Step11OaKConfig
from alberta_framework.steps.step12 import Step12IAConfig

OBS_DIM = 2
N_PRIMITIVE = 2
OPTION_ACTION = N_PRIMITIVE
ANCHOR_OBS = jnp.array([1.0, 0.0], dtype=jnp.float32)


def _config(*, backups: int = 0, model_decay: float = 0.0) -> STOMPConfig:
    return STOMPConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=1.0e6,
                pseudo_reward_scale=10.0,
                max_option_steps=8,
            ),
        ),
        observation_dim=OBS_DIM,
        n_primitive_actions=N_PRIMITIVE,
        base_step_size=0.05,
        base_avg_reward_step_size=0.01,
        base_trace_decay=0.0,
        option_gamma=1.0,
        option_model_decay=model_decay,
        option_planning_backups_per_step=backups,
        epsilon_base=0.0,
    )


def _state_with_completed_model(
    agent: STOMPAgent,
    *,
    completions: int = 1,
):
    start_obs = jnp.array([0.0, 1.0], dtype=jnp.float32)
    state = agent.start(agent.init(jr.key(101)), start_obs)

    # Make every planning-target term identifiable:
    # max_a Q(anchor, a) = 2 and Q(anchor, option) = 0.5.
    head_weights = (
        jnp.array([[2.0, 0.0]], dtype=jnp.float32),
        jnp.array([[-1.0, 0.0]], dtype=jnp.float32),
        jnp.array([[0.5, 0.0]], dtype=jnp.float32),
    )
    head_biases = tuple(jnp.zeros(1, dtype=jnp.float32) for _ in head_weights)
    learner_state = state.base_learner_state.replace(
        head_params=state.base_learner_state.head_params.replace(
            weights=head_weights,
            biases=head_biases,
        )
    )
    model_state = state.option_models.replace(
        cumreward_ema=jnp.array([999.0], dtype=jnp.float32),
        env_return_ema=jnp.array([1.0], dtype=jnp.float32),
        duration_ema=jnp.array([2.0], dtype=jnp.float32),
        baseline_mass_ema=jnp.array([2.0], dtype=jnp.float32),
        discount_ema=jnp.array([0.5], dtype=jnp.float32),
        next_state_weights=jnp.zeros((1, OBS_DIM, OBS_DIM), dtype=jnp.float32),
        n_completions=jnp.array([completions], dtype=jnp.int32),
    )
    candidate = state.replace(
        base_learner_state=learner_state,
        base_average_reward=jnp.array(0.5, dtype=jnp.float32),
        base_last_obs=start_obs,
        base_last_action=jnp.array(OPTION_ACTION, dtype=jnp.int32),
        option_models=model_state,
        executing_option=jnp.array(0, dtype=jnp.int32),
        option_start_obs=start_obs,
        option_last_intra_action=state.last_primitive_action,
        option_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_env_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_baseline_mass=jnp.array(0.0, dtype=jnp.float32),
        option_discount=jnp.array(1.0, dtype=jnp.float32),
        option_steps=jnp.array(0, dtype=jnp.int32),
    )
    assert bool(agent.state_valid(candidate))
    return candidate


def _assert_learner_equal(actual, expected) -> None:
    chex.assert_trees_all_equal(actual, expected)


def test_completed_option_model_records_task_and_subtask_outcomes_separately() -> None:
    config = STOMPConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=1.0e6,
                pseudo_reward_scale=10.0,
                max_option_steps=2,
            ),
        ),
        observation_dim=OBS_DIM,
        n_primitive_actions=N_PRIMITIVE,
        option_gamma=0.5,
        option_model_decay=0.0,
        option_planning_backups_per_step=0,
        epsilon_base=0.0,
    )
    agent = STOMPAgent(config)
    observations = jnp.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=jnp.float32)
    rewards = jnp.array([2.0, 6.0], dtype=jnp.float32)
    state = agent.start(agent.init(jr.key(0)), observations[0])
    state = state.replace(
        executing_option=jnp.array(0, dtype=jnp.int32),
        base_last_action=jnp.array(OPTION_ACTION, dtype=jnp.int32),
        option_start_obs=observations[0],
        option_last_intra_action=state.last_primitive_action,
        option_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_env_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_baseline_mass=jnp.array(0.0, dtype=jnp.float32),
        option_discount=jnp.array(1.0, dtype=jnp.float32),
        option_steps=jnp.array(0, dtype=jnp.int32),
    )
    assert bool(agent.state_valid(state))

    first = agent.update(state, rewards[0], observations[1])
    assert bool(first.update_applied)
    result = agent.update(first.state, rewards[1], observations[2])
    assert bool(result.update_applied)
    models = result.state.option_models

    assert bool(result.option_terminated)
    np.testing.assert_allclose(float(models.cumreward_ema[0]), 50.0)
    np.testing.assert_allclose(float(models.env_return_ema[0]), 5.0)
    np.testing.assert_allclose(float(models.duration_ema[0]), 2.0)
    np.testing.assert_allclose(float(models.baseline_mass_ema[0]), 1.5)
    np.testing.assert_allclose(float(models.discount_ema[0]), 0.25)
    assert int(models.n_completions[0]) == 1


def test_default_zero_planning_is_exactly_explicit_zero() -> None:
    default_agent = STOMPAgent(_config())
    explicit_agent = STOMPAgent(_config(backups=0))
    observations = jr.normal(jr.key(2), (13, OBS_DIM), dtype=jnp.float32)
    rewards = jr.normal(jr.key(3), (12,), dtype=jnp.float32)
    shared_state = default_agent.start(default_agent.init(jr.key(4)), observations[0])

    default_result = default_agent.scan(shared_state, rewards, observations[1:])
    explicit_result = explicit_agent.scan(shared_state, rewards, observations[1:])

    chex.assert_trees_all_equal(default_result, explicit_result)
    np.testing.assert_array_equal(
        np.asarray(default_result.planning_backups), np.zeros(12, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(default_result.planning_td_errors),
        np.zeros(12, dtype=np.float32),
    )


def test_legacy_config_without_planning_budget_defaults_to_disabled() -> None:
    payload = _config(backups=3).to_config()
    payload.pop("option_planning_backups_per_step")

    restored = STOMPConfig.from_config(payload)

    assert restored.option_planning_backups_per_step == 0


def test_production_facades_propagate_planning_budget() -> None:
    spec = (SubtaskSpec(feature_index=0),)
    step10 = Step10STOMPConfig(subtask_specs=spec, option_planning_backups_per_step=2)
    step11 = Step11OaKConfig(subtask_specs=spec, option_planning_backups_per_step=3)
    step12 = Step12IAConfig(subtask_specs=spec, option_planning_backups_per_step=4)

    assert step10.to_stomp_config().option_planning_backups_per_step == 2
    assert step11.to_oak_config().stomp.option_planning_backups_per_step == 3
    assert step12.to_ia_config().cortex.stomp.option_planning_backups_per_step == 4


@pytest.mark.parametrize("invalid", [-1, 1.5, True])
def test_planning_budget_must_be_nonnegative_static_integer(invalid: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            option_planning_backups_per_step=invalid,  # type: ignore[arg-type]
        )


def test_no_planning_before_any_model_completion() -> None:
    agent = STOMPAgent(_config(backups=3))
    state = _state_with_completed_model(agent, completions=0)
    before = state.base_learner_state

    result = agent.update(state, jnp.array(0.0), ANCHOR_OBS)

    assert bool(result.update_applied)
    assert int(result.planning_backups) == 0
    assert float(result.planning_td_error) == 0.0
    _assert_learner_equal(result.state.base_learner_state, before)


def test_planning_can_be_suppressed_for_imagined_transition() -> None:
    agent = STOMPAgent(_config(backups=3))
    state = _state_with_completed_model(agent)
    before = state.base_learner_state

    result = agent.update(
        state,
        jnp.array(0.0),
        ANCHOR_OBS,
        enable_planning=False,
    )

    assert bool(result.update_applied)
    assert int(result.planning_backups) == 0
    assert float(result.planning_td_error) == 0.0
    _assert_learner_equal(result.state.base_learner_state, before)


def test_fixed_budget_updates_only_option_head_at_real_anchor() -> None:
    agent = STOMPAgent(_config(backups=3))
    state = _state_with_completed_model(agent)
    before = state.base_learner_state

    result = agent.update(state, jnp.array(0.0), ANCHOR_OBS)

    assert int(result.planning_backups) == 3
    assert int(result.state.base_learner_state.step_count) == 3
    np.testing.assert_array_equal(
        np.asarray(result.average_reward), np.asarray(state.base_average_reward)
    )
    after = result.state.base_learner_state
    for head in range(N_PRIMITIVE):
        np.testing.assert_array_equal(
            np.asarray(after.head_params.weights[head]),
            np.asarray(before.head_params.weights[head]),
        )
        np.testing.assert_array_equal(
            np.asarray(after.head_params.biases[head]),
            np.asarray(before.head_params.biases[head]),
        )
    option_delta = np.asarray(after.head_params.weights[OPTION_ACTION]) - np.asarray(
        before.head_params.weights[OPTION_ACTION]
    )
    assert abs(float(option_delta[0, 0])) > 1.0e-6
    np.testing.assert_allclose(option_delta[0, 1], 0.0, rtol=0.0, atol=1.0e-8)


def test_planning_selects_only_models_with_completions() -> None:
    config = STOMPConfig(
        subtask_specs=(
            SubtaskSpec(feature_index=0, threshold=1.0e6, max_option_steps=8),
            SubtaskSpec(feature_index=1, threshold=1.0e6, max_option_steps=8),
        ),
        observation_dim=OBS_DIM,
        n_primitive_actions=N_PRIMITIVE,
        base_step_size=0.05,
        option_planning_backups_per_step=2,
        epsilon_base=0.0,
    )
    agent = STOMPAgent(config)
    state = agent.start(agent.init(jr.key(7)), jnp.array([0.0, 1.0]))
    zero_weights = tuple(
        jnp.zeros_like(weight) for weight in state.base_learner_state.head_params.weights
    )
    learner_state = state.base_learner_state.replace(
        head_params=state.base_learner_state.head_params.replace(weights=zero_weights)
    )
    models = state.option_models.replace(
        env_return_ema=jnp.array([100.0, 1.0], dtype=jnp.float32),
        duration_ema=jnp.ones(2, dtype=jnp.float32),
        baseline_mass_ema=jnp.ones(2, dtype=jnp.float32),
        discount_ema=jnp.zeros(2, dtype=jnp.float32),
        n_completions=jnp.array([0, 1], dtype=jnp.int32),
    )
    state = state.replace(
        base_learner_state=learner_state,
        option_models=models,
        executing_option=jnp.array(0, dtype=jnp.int32),
        option_start_obs=jnp.array([0.0, 1.0], dtype=jnp.float32),
        base_last_obs=jnp.array([0.0, 1.0], dtype=jnp.float32),
        base_last_action=jnp.array(OPTION_ACTION, dtype=jnp.int32),
        option_last_intra_action=state.last_primitive_action,
        option_steps=jnp.array(0, dtype=jnp.int32),
    )
    assert bool(agent.state_valid(state))

    result = agent.update(state, jnp.array(0.0), ANCHOR_OBS)

    assert bool(result.update_applied)
    assert int(result.planning_backups) == 2
    after_weights = result.state.base_learner_state.head_params.weights
    completed_option_action = N_PRIMITIVE + 1
    for head in range(completed_option_action):
        np.testing.assert_array_equal(
            np.asarray(after_weights[head]), np.asarray(zero_weights[head])
        )
    assert bool(jnp.any(after_weights[completed_option_action] != 0.0))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("env_return_ema", jnp.array([2.0], dtype=jnp.float32)),
        ("baseline_mass_ema", jnp.array([4.0], dtype=jnp.float32)),
        ("discount_ema", jnp.array([0.0], dtype=jnp.float32)),
        (
            "next_state_weights",
            jnp.array([[[1.0, 0.0], [0.0, 0.0]]], dtype=jnp.float32),
        ),
    ],
)
def test_planning_target_reads_every_task_model_field(field: str, replacement: jnp.ndarray) -> None:
    agent = STOMPAgent(_config(backups=1))
    state = _state_with_completed_model(agent)
    baseline = agent.update(state, jnp.array(0.0), ANCHOR_OBS)
    mutated_models = state.option_models.replace(**{field: replacement})
    counterfactual = agent.update(
        state.replace(option_models=mutated_models),
        jnp.array(0.0),
        ANCHOR_OBS,
    )

    np.testing.assert_allclose(float(baseline.planning_td_error), 0.5)
    assert not np.isclose(
        float(counterfactual.planning_td_error),
        float(baseline.planning_td_error),
    )


def test_raw_duration_is_diagnostic_not_a_discounted_bellman_coefficient() -> None:
    """Changing T alone cannot create the old hybrid R^γ - avg*T target."""
    agent = STOMPAgent(_config(backups=1))
    state = _state_with_completed_model(agent)
    baseline = agent.update(state, jnp.array(0.0), ANCHOR_OBS)
    mutated = state.replace(
        option_models=state.option_models.replace(
            duration_ema=jnp.array([1000.0], dtype=jnp.float32)
        )
    )
    counterfactual = agent.update(mutated, jnp.array(0.0), ANCHOR_OBS)

    assert bool(baseline.update_applied)
    assert bool(counterfactual.update_applied)
    np.testing.assert_allclose(
        float(counterfactual.planning_td_error),
        float(baseline.planning_td_error),
    )
    _assert_learner_equal(
        counterfactual.state.base_learner_state,
        baseline.state.base_learner_state,
    )


def test_pseudo_return_is_not_consumed_by_base_planning() -> None:
    agent = STOMPAgent(_config(backups=1))
    state = _state_with_completed_model(agent)
    baseline = agent.update(state, jnp.array(0.0), ANCHOR_OBS)
    mutated = state.replace(
        option_models=state.option_models.replace(
            cumreward_ema=jnp.array([-1.0e6], dtype=jnp.float32)
        )
    )
    counterfactual = agent.update(mutated, jnp.array(0.0), ANCHOR_OBS)

    assert bool(baseline.update_applied)
    assert bool(counterfactual.update_applied)
    np.testing.assert_allclose(
        float(counterfactual.planning_td_error),
        float(baseline.planning_td_error),
    )
    _assert_learner_equal(
        counterfactual.state.base_learner_state,
        baseline.state.base_learner_state,
    )


def test_matched_model_backups_reduce_option_value_error() -> None:
    planning_agent = STOMPAgent(_config(backups=4))
    disabled_agent = STOMPAgent(_config(backups=0))
    state = _state_with_completed_model(planning_agent)
    target = 1.0

    disabled = disabled_agent.update(state, jnp.array(0.0), ANCHOR_OBS)
    planned = planning_agent.update(state, jnp.array(0.0), ANCHOR_OBS)
    q_disabled = disabled_agent.base_q_values(disabled.state, ANCHOR_OBS)[OPTION_ACTION]
    q_planned = planning_agent.base_q_values(planned.state, ANCHOR_OBS)[OPTION_ACTION]

    np.testing.assert_allclose(float(q_disabled), 0.5, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(float(q_planned), 0.67195, rtol=1.0e-5, atol=1.0e-6)
    assert abs(float(q_planned) - target) < abs(float(q_disabled) - target)
