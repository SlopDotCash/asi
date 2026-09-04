# mypy: disable-error-code="attr-defined,no-any-return"
"""Bootstrap/decision split regressions for episodic STOMP and OaK use."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.oak import OaKAgent, OaKConfig
from alberta_framework.core.options import STOMPAgent, STOMPConfig, STOMPState, SubtaskSpec

OBS_DIM = 2
N_PRIMITIVE = 2
OPTION_ACTION = N_PRIMITIVE
LAST_OBS = jnp.array([0.0, 1.0], dtype=jnp.float32)
BOOTSTRAP_OBS = jnp.array([1.0, 0.0], dtype=jnp.float32)
DECISION_OBS = jnp.array([0.0, 1.0], dtype=jnp.float32)


def _config(
    *,
    threshold: float = 100.0,
    base_step_size: float = 0.0,
    option_step_size: float = 0.0,
    option_trace_decay: float = 0.75,
    option_model_step_size: float = 1.0,
) -> STOMPConfig:
    return STOMPConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=threshold,
                max_option_steps=100,
            ),
        ),
        observation_dim=OBS_DIM,
        n_primitive_actions=N_PRIMITIVE,
        base_step_size=base_step_size,
        base_avg_reward_step_size=0.0,
        base_trace_decay=0.0,
        option_step_size=option_step_size,
        option_avg_reward_step_size=0.0,
        option_trace_decay=option_trace_decay,
        option_model_decay=0.0,
        option_model_step_size=option_model_step_size,
        epsilon_base=0.0,
        epsilon_option=0.0,
    )


def _with_base_weights(state: STOMPState, weights: tuple[ArrayLike, ...]) -> STOMPState:
    arrays = tuple(jnp.asarray(weight, dtype=jnp.float32) for weight in weights)
    learner_state = state.base_learner_state.replace(
        head_params=state.base_learner_state.head_params.replace(
            weights=arrays,
            biases=tuple(jnp.zeros(1, dtype=jnp.float32) for _ in arrays),
        )
    )
    return state.replace(base_learner_state=learner_state)


ArrayLike = list[list[float]] | np.ndarray


def _active_state(agent: STOMPAgent) -> STOMPState:
    state = agent.init(jr.key(0))
    return state.replace(
        base_last_obs=LAST_OBS,
        base_last_action=jnp.array(OPTION_ACTION, dtype=jnp.int32),
        last_primitive_action=jnp.array(0, dtype=jnp.int32),
        executing_option=jnp.array(0, dtype=jnp.int32),
        option_start_obs=LAST_OBS,
        option_last_intra_action=jnp.array(0, dtype=jnp.int32),
        option_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_env_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_baseline_mass=jnp.array(0.0, dtype=jnp.float32),
        option_discount=jnp.array(1.0, dtype=jnp.float32),
        option_steps=jnp.array(0, dtype=jnp.int32),
    )


def test_censored_boundary_bootstraps_intra_option_then_clears_trace() -> None:
    """A truncation sample learns, but cannot become an option completion."""
    agent = STOMPAgent(_config(option_step_size=0.5))
    state = _active_state(agent)
    option_q = jnp.array(
        [[[0.0, 1.0], [2.0, 0.0]]],
        dtype=jnp.float32,
    )
    state = state.replace(
        option_policies=state.option_policies.replace(q_weights=option_q)
    )

    result = agent.update(
        state,
        jnp.array(0.0, dtype=jnp.float32),
        BOOTSTRAP_OBS,
        jnp.array(0.5, dtype=jnp.float32),
        decision_observation=DECISION_OBS,
        execution_boundary=jnp.array(True),
    )

    # q_prev=1, pseudo=1 and 0.5*max(q(bootstrap))=1, hence TD=1.
    # The boundary sample changes the selected weight before its trace clears.
    np.testing.assert_allclose(
        np.asarray(result.state.option_policies.q_weights[0, 0]),
        np.array([0.0, 1.5], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(result.state.option_policies.traces[0]),
        np.zeros((N_PRIMITIVE, OBS_DIM), dtype=np.float32),
    )
    assert bool(result.option_terminated)
    assert float(result.pseudo_reward) == 1.0
    assert int(result.state.option_models.n_completions[0]) == 0


def test_goal_termination_credits_earlier_option_steps_through_the_trace() -> None:
    """The goal step's TD error must reach earlier option actions via the carried trace.

    Accumulating traces decay by the discount *entering* the current state, not by
    the termination that zeros the bootstrap: ``e_t = gamma_t * lambda * e_{t-1} +
    phi_t``. Zeroing the incoming trace on the goal step turns intra-option
    learning into TD(0) exactly where sparse pseudo-reward needs multi-step credit.
    """
    agent = STOMPAgent(_config(threshold=1.0, option_step_size=0.5, option_trace_decay=0.75))
    state = _active_state(agent)
    prior_trace = np.zeros((N_PRIMITIVE, OBS_DIM), dtype=np.float32)
    prior_trace[1, 1] = 1.0  # an earlier option step took action 1 on feature 1
    state = state.replace(
        option_policies=state.option_policies.replace(
            q_weights=jnp.zeros((1, N_PRIMITIVE, OBS_DIM), dtype=jnp.float32),
            traces=jnp.asarray(prior_trace)[None, :, :],
        )
    )

    discount = 0.5
    result = agent.update(
        state,
        jnp.array(0.0, dtype=jnp.float32),
        BOOTSTRAP_OBS,
        jnp.array(discount, dtype=jnp.float32),
        decision_observation=DECISION_OBS,
    )

    assert bool(result.option_terminated)
    assert float(result.pseudo_reward) == 1.0
    # q_prev = 0, bootstrap zeroed by termination, hence TD = pseudo_reward = 1.
    td_error = 1.0
    current = np.zeros((N_PRIMITIVE, OBS_DIM), dtype=np.float32)
    current[0] = np.asarray(LAST_OBS)  # the goal step took action 0 on LAST_OBS
    expected_trace = 0.75 * discount * prior_trace + current
    expected_q = 0.5 * td_error * expected_trace
    np.testing.assert_allclose(
        np.asarray(result.state.option_policies.q_weights[0]),
        expected_q,
        rtol=0.0,
        atol=1.0e-6,
    )
    # The lifecycle ended, so the stored trace is cleared only after the update.
    np.testing.assert_array_equal(
        np.asarray(result.state.option_policies.traces[0]),
        np.zeros((N_PRIMITIVE, OBS_DIM), dtype=np.float32),
    )


def test_censored_boundary_keeps_positive_base_bellman_bootstrap() -> None:
    agent = STOMPAgent(_config())
    state = _with_base_weights(
        _active_state(agent),
        (
            [[2.0, 0.0]],
            [[-1.0, 0.0]],
            [[0.0, 0.0]],
        ),
    )

    result = agent.update(
        state,
        jnp.array(1.0, dtype=jnp.float32),
        BOOTSTRAP_OBS,
        jnp.array(0.5, dtype=jnp.float32),
        decision_observation=DECISION_OBS,
        execution_boundary=True,
    )

    # Partial option return 1 + positive bootstrap 0.5 * maxQ=2.
    np.testing.assert_allclose(float(result.td_error), 2.0, rtol=0.0, atol=1.0e-6)
    assert int(result.state.option_models.n_completions[0]) == 0


def test_boundary_next_decision_and_new_option_start_use_reset_observation() -> None:
    agent = STOMPAgent(_config())
    state = _with_base_weights(
        _active_state(agent),
        (
            [[10.0, 0.0]],
            [[-10.0, 0.0]],
            [[0.0, 10.0]],
        ),
    )
    option_q = jnp.array(
        [[[0.0, -5.0], [0.0, 5.0]]],
        dtype=jnp.float32,
    )
    state = state.replace(
        option_policies=state.option_policies.replace(q_weights=option_q)
    )

    result = agent.update(
        state,
        jnp.array(0.0, dtype=jnp.float32),
        BOOTSTRAP_OBS,
        jnp.array(0.9, dtype=jnp.float32),
        decision_observation=DECISION_OBS,
        execution_boundary=True,
    )

    # Bootstrap obs would choose primitive 0; reset/decision obs chooses option 0.
    assert int(result.state.base_last_action) == OPTION_ACTION
    assert int(result.executing_option) == 0
    assert int(result.primitive_action) == 1
    np.testing.assert_array_equal(np.asarray(result.state.base_last_obs), DECISION_OBS)
    np.testing.assert_array_equal(np.asarray(result.state.option_start_obs), DECISION_OBS)


def test_natural_and_terminal_completions_still_train_option_model() -> None:
    natural_agent = STOMPAgent(_config(threshold=0.5))
    natural_state = _with_base_weights(
        _active_state(natural_agent),
        (
            [[10.0, 0.0]],
            [[-10.0, 0.0]],
            [[0.0, 10.0]],
        ),
    )
    natural = natural_agent.update(
        natural_state,
        jnp.array(0.0, dtype=jnp.float32),
        BOOTSTRAP_OBS,
        jnp.array(0.9, dtype=jnp.float32),
        decision_observation=DECISION_OBS,
    )

    assert bool(natural.option_terminated)
    assert int(natural.state.option_models.n_completions[0]) == 1
    # Outcome learning ends at the bootstrap observation, while the restarted
    # option begins at the distinct reset/decision observation.
    np.testing.assert_allclose(
        np.asarray(natural.state.option_models.next_state_weights[0]),
        np.array([[0.0, 1.0], [0.0, -1.0]], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(natural.state.option_start_obs),
        np.asarray(DECISION_OBS),
    )

    terminal_agent = STOMPAgent(_config())
    terminal = terminal_agent.update(
        _active_state(terminal_agent),
        jnp.array(0.0, dtype=jnp.float32),
        BOOTSTRAP_OBS,
        jnp.array(0.0, dtype=jnp.float32),
        decision_observation=DECISION_OBS,
    )
    assert bool(terminal.option_terminated)
    assert int(terminal.state.option_models.n_completions[0]) == 1


def test_oak_and_scan_forward_boundary_contract() -> None:
    config = OaKConfig(stomp=_config(), utility_ema_decay=0.5)
    agent = OaKAgent(config)
    initial = agent.init(jr.key(7))
    stomp_state = _with_base_weights(
        _active_state(agent.stomp_agent),
        (
            [[10.0, 0.0]],
            [[-10.0, 0.0]],
            [[0.0, 10.0]],
        ),
    )
    initial = initial.replace(stomp_state=stomp_state)

    result = agent.scan(
        initial,
        jnp.array([0.0], dtype=jnp.float32),
        BOOTSTRAP_OBS[None, :],
        jnp.array([0.75], dtype=jnp.float32),
        decision_observations=DECISION_OBS[None, :],
        execution_boundaries=jnp.array([True]),
    )

    assert bool(result.option_terminations[0])
    assert float(result.pseudo_rewards[0]) == 1.0
    assert int(result.state.stomp_state.option_models.n_completions[0]) == 0
    # The interrupted option is selected afresh from the decision observation.
    assert int(result.state.stomp_state.executing_option) == 0
    np.testing.assert_array_equal(
        np.asarray(result.state.execution_counts),
        np.array([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(result.state.stomp_state.base_last_obs),
        np.asarray(DECISION_OBS),
    )
