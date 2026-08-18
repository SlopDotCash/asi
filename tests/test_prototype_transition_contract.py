"""Analytic tests for explicit real and imagined transition discounts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.dreaming import DreamingConfig, DreamTransition
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPAgent, STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    _DREAM_NEXT_OBSERVATION_STREAM_TAG,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeTransition,
    PrototypeUpdateResult,
    _sample_one_hot_dream_observation,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig


def test_transition_and_checkpoint_helpers_are_publicly_exported() -> None:
    assert alberta.PrototypeTransition is core.PrototypeTransition
    assert alberta.PrototypeDecision is core.PrototypeDecision
    assert alberta.PrototypeAgent is core.PrototypeAgent
    assert (
        alberta.PROTOTYPE_CHECKPOINT_SCHEMA
        == core.PROTOTYPE_CHECKPOINT_SCHEMA
    )
    assert alberta.save_prototype_checkpoint is core.save_prototype_checkpoint
    assert alberta.load_prototype_checkpoint is core.load_prototype_checkpoint


def _stomp_config(
    *,
    option_gamma: float = 0.8,
    max_option_steps: int = 8,
    option_model_decay: float = 0.0,
) -> STOMPConfig:
    return STOMPConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=1.0e6,
                max_option_steps=max_option_steps,
            ),
        ),
        observation_dim=2,
        n_primitive_actions=2,
        base_step_size=0.05,
        base_avg_reward_step_size=0.01,
        option_gamma=option_gamma,
        option_model_decay=option_model_decay,
        epsilon_base=0.0,
        epsilon_option=0.0,
    )


def _prototype_config(
    *,
    world_model: bool = False,
    n_dreams: int = 0,
    horde: bool = False,
    dream_next_observation_mode: Literal[
        "model_prediction",
        "sample_one_hot",
    ] = "model_prediction",
) -> PrototypeAgentConfig:
    horde_spec = None
    if horde:
        horde_spec = create_horde_spec(
            (
                GVFSpec(
                    name="short",
                    demon_type=DemonType.PREDICTION,
                    gamma=0.2,
                    lamda=0.0,
                    cumulant_index=-1,
                ),
                GVFSpec(
                    name="long",
                    demon_type=DemonType.PREDICTION,
                    gamma=0.9,
                    lamda=0.0,
                    cumulant_index=-1,
                ),
            )
        )
    wm_config = (
        ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            step_size=0.1,
            gamma=0.95,
        )
        if world_model
        else None
    )
    return PrototypeAgentConfig(
        oak=OaKConfig(stomp=_stomp_config()),
        world_model=wm_config,
        dreaming=(
            DreamingConfig(warmup_steps=0, max_model_error_ema=1.0e9)
            if world_model
            else None
        ),
        buffer_capacity=8,
        n_dreams_per_step=n_dreams,
        dream_next_observation_mode=dream_next_observation_mode,
        horde_spec=horde_spec,
        horde_hidden_sizes=(),
        horde_step_size=0.05,
    )


def _materialize_typed_keys(tree: object) -> object:
    """Convert typed PRNG leaves so Chex can compare complete states."""

    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def _explicit_transition(
    state: PrototypeAgentState,
    *,
    reward: object,
    next_observation: object,
    discount: object,
    terminated: object | None = None,
    truncated: object = False,
    horde_cumulants: object | None = None,
    horde_discounts: object | None = None,
) -> PrototypeTransition:
    """Build an owned explicit transition from the state's decision cache."""
    discount_array = jnp.asarray(discount, dtype=jnp.float32)
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=reward,
        discount=discount,
        terminated=(discount_array == 0.0 if terminated is None else terminated),
        truncated=truncated,
        next_observation=next_observation,
        next_decision_observation=next_observation,
        horde_cumulants=horde_cumulants,
        horde_discounts=horde_discounts,
    )


@pytest.mark.parametrize("discount", [0.0, 0.25])
def test_primitive_backup_matches_hand_derived_discount(discount: float) -> None:
    agent = STOMPAgent(_stomp_config())
    last_obs = jnp.array([1.0, -0.5], dtype=jnp.float32)
    next_obs = jnp.array([-0.25, 2.0], dtype=jnp.float32)
    state = agent.start(agent.init(jr.key(0)), last_obs).replace(
        executing_option=jnp.array(-1, dtype=jnp.int32),
        base_last_obs=last_obs,
        base_last_action=jnp.array(0, dtype=jnp.int32),
        last_primitive_action=jnp.array(0, dtype=jnp.int32),
        base_average_reward=jnp.array(0.4, dtype=jnp.float32),
    )
    assert bool(agent.state_valid(state))
    reward = jnp.array(1.7, dtype=jnp.float32)
    q_previous = agent.base_learner.predict(state.base_learner_state, last_obs)[0]
    q_next = jnp.max(agent.base_learner.predict(state.base_learner_state, next_obs))
    expected = reward - state.base_average_reward + discount * q_next - q_previous

    result = agent.update(
        state,
        reward,
        next_obs,
        jnp.array(discount, dtype=jnp.float32),
    )

    np.testing.assert_allclose(result.td_error, expected, rtol=1e-6, atol=1e-6)


def test_option_return_product_baseline_and_terminal_are_hand_derived() -> None:
    agent = STOMPAgent(_stomp_config(max_option_steps=10))
    start_obs = jnp.array([1.0, 0.0], dtype=jnp.float32)
    observations = (
        jnp.array([0.5, 0.5], dtype=jnp.float32),
        jnp.array([0.25, 0.75], dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
    )
    rewards = (2.0, 3.0, 5.0)
    discounts = (0.5, 0.25, 0.0)
    option_action = 2
    state = agent.start(agent.init(jr.key(1)), start_obs)
    state = state.replace(
        executing_option=jnp.array(0, dtype=jnp.int32),
        base_last_obs=start_obs,
        base_last_action=jnp.array(option_action, dtype=jnp.int32),
        base_average_reward=jnp.array(0.3, dtype=jnp.float32),
        option_start_obs=start_obs,
        option_last_intra_action=state.last_primitive_action,
        option_env_cumreward=jnp.array(0.0, dtype=jnp.float32),
        option_baseline_mass=jnp.array(0.0, dtype=jnp.float32),
        option_discount=jnp.array(1.0, dtype=jnp.float32),
        option_steps=jnp.array(0, dtype=jnp.int32),
    )
    assert bool(agent.state_valid(state))

    first = agent.update(state, rewards[0], observations[0], discounts[0])
    assert not bool(first.option_terminated)
    np.testing.assert_allclose(first.state.option_env_cumreward, 2.0)
    np.testing.assert_allclose(first.state.option_baseline_mass, 1.0)
    np.testing.assert_allclose(first.state.option_discount, 0.5)

    second = agent.update(first.state, rewards[1], observations[1], discounts[1])
    assert not bool(second.option_terminated)
    np.testing.assert_allclose(second.state.option_env_cumreward, 2.0 + 0.5 * 3.0)
    np.testing.assert_allclose(second.state.option_baseline_mass, 1.0 + 0.5)
    np.testing.assert_allclose(second.state.option_discount, 0.5 * 0.25)

    q_previous = agent.base_learner.predict(
        second.state.base_learner_state,
        start_obs,
    )[option_action]
    expected_return = 2.0 + 0.5 * 3.0 + 0.5 * 0.25 * 5.0
    expected_mass = 1.0 + 0.5 + 0.5 * 0.25
    expected_td = (
        expected_return
        - second.state.base_average_reward * expected_mass
        - q_previous
    )
    terminal = agent.update(
        second.state,
        rewards[2],
        observations[2],
        discounts[2],
    )

    assert bool(terminal.option_terminated)
    np.testing.assert_allclose(terminal.td_error, expected_td, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        terminal.state.option_models.env_return_ema[0],
        expected_return,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        terminal.state.option_models.baseline_mass_ema[0],
        expected_mass,
        rtol=1e-6,
    )
    np.testing.assert_allclose(terminal.state.option_models.discount_ema[0], 0.0)


def test_legacy_stomp_update_retains_configured_option_gamma() -> None:
    agent = STOMPAgent(_stomp_config(option_gamma=0.6, max_option_steps=8))
    observation = jnp.array([0.0, 1.0], dtype=jnp.float32)
    state = agent.start(agent.init(jr.key(2)), observation)
    state = state.replace(
        executing_option=jnp.array(0, dtype=jnp.int32),
        base_last_obs=observation,
        base_last_action=jnp.array(2, dtype=jnp.int32),
        option_start_obs=observation,
        option_last_intra_action=state.last_primitive_action,
        option_discount=jnp.array(1.0, dtype=jnp.float32),
        option_steps=jnp.array(0, dtype=jnp.int32),
    )
    assert bool(agent.state_valid(state))

    result = agent.update(state, jnp.array(1.0), observation)

    np.testing.assert_allclose(result.state.option_discount, 0.6)


def test_world_model_error_uses_supplied_discount_target() -> None:
    agent = PrototypeAgent(_prototype_config(world_model=True))
    last_obs = jnp.array([0.4, -0.2], dtype=jnp.float32)
    next_obs = jnp.array([0.1, 0.3], dtype=jnp.float32)
    state = agent.start(agent.init(jr.key(3)), last_obs)
    assert agent._world_model is not None
    prediction = agent._world_model.predict(
        state.world_model_state,
        state.oak_state.stomp_state.base_last_obs,
        state.oak_state.stomp_state.last_primitive_action,
    )
    reward = jnp.array(0.7, dtype=jnp.float32)
    discount = jnp.array(0.37, dtype=jnp.float32)
    expected_error = (
        jnp.mean((prediction.next_observation - next_obs) ** 2)
        + (prediction.reward - reward) ** 2
        + (prediction.discount - discount) ** 2
    )

    result = agent.update_transition(
        state,
        _explicit_transition(
            state,
            reward=reward,
            next_observation=next_obs,
            discount=discount,
        ),
    )

    np.testing.assert_allclose(
        result.world_model_error,
        expected_error,
        rtol=1e-6,
        atol=1e-6,
    )


def test_rejected_world_model_update_rejects_the_complete_prototype_transition() -> None:
    agent = PrototypeAgent(_prototype_config(world_model=True))
    state = agent.start(agent.init(jr.key(303)), jnp.asarray([0.4, -0.2]))
    warm = agent.update_transition(
        state,
        _explicit_transition(
            state,
            reward=jnp.asarray(0.5),
            next_observation=jnp.asarray([0.1, 0.3]),
            discount=jnp.asarray(0.9),
        ),
    )
    state = warm.state
    maximum = jnp.asarray(2_147_483_647, dtype=jnp.int32)
    world_model_state = state.world_model_state
    exhausted_world_model_state = world_model_state.replace(
        learner_state=world_model_state.learner_state.replace(
            step_count=maximum,
            step_words=jnp.asarray([0xFFFFFFFF, 0xFFFFFFFF], dtype=jnp.uint32),
        ),
        step_count=maximum,
    )
    exhausted = state.replace(world_model_state=exhausted_world_model_state)

    result = agent.update_transition(
        exhausted,
        _explicit_transition(
            exhausted,
            reward=jnp.asarray(0.25),
            next_observation=jnp.asarray([-0.2, 0.6]),
            discount=jnp.asarray(0.9),
        ),
    )

    assert not bool(result.transition_diagnostics.valid)
    assert bool(result.transition_diagnostics.rejected)
    chex.assert_trees_all_equal(
        _materialize_typed_keys(result.state),
        _materialize_typed_keys(exhausted),
    )
    assert float(result.world_model_error) == 0.0
    assert int(result.state.buffer_state.size) == int(exhausted.buffer_state.size)


def test_legacy_prototype_wrapper_retains_split_discount_behavior() -> None:
    agent = PrototypeAgent(_prototype_config(world_model=True))
    last_obs = jnp.array([0.4, -0.2], dtype=jnp.float32)
    next_obs = jnp.array([0.1, 0.3], dtype=jnp.float32)
    state = agent.start(agent.init(jr.key(31)), last_obs)
    state = state.replace(
        oak_state=state.oak_state.replace(
            stomp_state=state.oak_state.stomp_state.replace(
                executing_option=jnp.array(0, dtype=jnp.int32),
                base_last_action=jnp.array(2, dtype=jnp.int32),
                option_start_obs=last_obs,
                option_last_intra_action=(
                    state.oak_state.stomp_state.last_primitive_action
                ),
                option_discount=jnp.array(1.0, dtype=jnp.float32),
                option_steps=jnp.array(0, dtype=jnp.int32),
            )
        )
    )
    assert bool(
        agent.oak_agent.stomp_agent.state_valid(state.oak_state.stomp_state)
    )
    assert agent._world_model is not None
    prediction = agent._world_model.predict(
        state.world_model_state,
        last_obs,
        state.oak_state.stomp_state.last_primitive_action,
    )
    reward = jnp.array(0.7, dtype=jnp.float32)
    expected_model_error = (
        jnp.mean((prediction.next_observation - next_obs) ** 2)
        + (prediction.reward - reward) ** 2
        + (prediction.discount - 0.95) ** 2
    )

    result = agent.update(state, reward, next_obs)

    np.testing.assert_allclose(result.world_model_error, expected_model_error, rtol=1e-6)
    # Compatibility mode keeps STOMP.option_gamma independent of the world
    # model's gamma.
    np.testing.assert_allclose(result.state.oak_state.stomp_state.option_discount, 0.8)


def test_horde_explicit_discounts_and_default_horizons() -> None:
    agent = PrototypeAgent(_prototype_config(horde=True))
    last_obs = jnp.array([0.4, -0.2], dtype=jnp.float32)
    next_obs = jnp.array([0.1, 0.3], dtype=jnp.float32)
    initial = agent.start(agent.init(jr.key(4)), last_obs)
    assert agent._horde is not None
    current = agent._horde.predict(initial.horde_state, last_obs)
    following = agent._horde.predict(initial.horde_state, next_obs)
    cumulants = jnp.array([1.0, -0.5], dtype=jnp.float32)

    explicit_discounts = jnp.array([0.0, 0.4], dtype=jnp.float32)
    explicit = agent.update_transition(
        initial,
        _explicit_transition(
            initial,
            reward=0.0,
            next_observation=next_obs,
            discount=0.5,
            horde_cumulants=cumulants,
            horde_discounts=explicit_discounts,
        ),
    )
    chex.assert_trees_all_close(
        explicit.horde_td_errors,
        cumulants + explicit_discounts * following - current,
        atol=1e-6,
    )

    defaults = agent.update_transition(
        initial,
        _explicit_transition(
            initial,
            reward=0.0,
            next_observation=next_obs,
            discount=0.5,
            horde_cumulants=cumulants,
        ),
    )
    configured = jnp.array([0.2, 0.9], dtype=jnp.float32)
    chex.assert_trees_all_close(
        defaults.horde_td_errors,
        cumulants + configured * following - current,
        atol=1e-6,
    )

    terminal = agent.update_transition(
        initial,
        _explicit_transition(
            initial,
            reward=0.0,
            next_observation=next_obs,
            discount=0.0,
            horde_cumulants=cumulants,
        ),
    )
    chex.assert_trees_all_close(
        terminal.horde_td_errors,
        cumulants - current,
        atol=1e-6,
    )


class _FixedDreamer:
    def __init__(self, transition: DreamTransition):
        self._transition = transition

    def propose(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            transition=self._transition,
            accepted=jnp.array(True),
        )


def test_one_hot_dream_sampling_is_jittable_and_preserves_categorical_support() -> None:
    prediction = jnp.array([0.0, 0.25, 0.75], dtype=jnp.float32)
    root = jr.key(17)
    keys = jax.vmap(lambda index: jr.fold_in(root, index))(jnp.arange(4096))
    samples, valid = jax.jit(
        jax.vmap(_sample_one_hot_dream_observation, in_axes=(None, 0))
    )(prediction, keys)

    chex.assert_shape(samples, (4096, 3))
    chex.assert_trees_all_equal(valid, jnp.ones(4096, dtype=jnp.bool_))
    chex.assert_trees_all_equal(samples[:, 0], jnp.zeros(4096))
    chex.assert_trees_all_equal(jnp.sum(samples, axis=1), jnp.ones(4096))
    chex.assert_trees_all_equal(
        (samples == 0.0) | (samples == 1.0),
        jnp.ones_like(samples, dtype=jnp.bool_),
    )
    np.testing.assert_allclose(
        np.asarray(jnp.mean(samples, axis=0)),
        np.array([0.0, 0.25, 0.75]),
        atol=0.03,
    )


@pytest.mark.parametrize(
    "prediction",
    (
        jnp.array([-1.0, 0.0], dtype=jnp.float32),
        jnp.array([jnp.nan, 1.0], dtype=jnp.float32),
        jnp.array([jnp.inf, 1.0], dtype=jnp.float32),
    ),
)
def test_one_hot_dream_sampling_marks_invalid_projection(
    prediction: jax.Array,
) -> None:
    sample, valid = jax.jit(_sample_one_hot_dream_observation)(
        prediction,
        jr.key(3),
    )
    chex.assert_trees_all_equal(valid, jnp.array(False))
    chex.assert_trees_all_equal(sample, jnp.array([1.0, 0.0]))


def _set_state_sensitive_q_values(
    state: PrototypeAgentState,
) -> PrototypeAgentState:
    """Return a prototype state whose next-state value distinguishes soft/one-hot inputs."""

    stomp = state.oak_state.stomp_state
    learner = stomp.base_learner_state
    weights = tuple(
        jnp.zeros_like(weight) for weight in learner.head_params.weights
    )
    weights = (
        weights[0],
        jnp.array([[0.0, 4.0]], dtype=jnp.float32),
        *weights[2:],
    )
    biases = tuple(
        jnp.zeros_like(bias) for bias in learner.head_params.biases
    )
    learner = learner.replace(
        head_params=learner.head_params.replace(
            weights=weights,
            biases=biases,
        )
    )
    return state.replace(
        oak_state=state.oak_state.replace(
            stomp_state=stomp.replace(
                base_learner_state=learner,
                base_average_reward=jnp.array(0.0, dtype=jnp.float32),
            )
        )
    )


def test_default_dream_backup_preserves_expectation_valued_model_prediction() -> None:
    agent = PrototypeAgent(_prototype_config(world_model=True, n_dreams=1))
    initial_obs = jnp.array([1.0, 0.0], dtype=jnp.float32)
    state = _set_state_sensitive_q_values(
        agent.start(agent.init(jr.key(5)), initial_obs)
    )
    assert agent._buffer is not None
    state = state.replace(
        buffer_state=agent._buffer.add(state.buffer_state, initial_obs)
    )
    transition = DreamTransition(
        observation=initial_obs,
        action=jnp.array(0, dtype=jnp.int32),
        reward=jnp.array(1.2, dtype=jnp.float32),
        discount=jnp.array(0.4, dtype=jnp.float32),
        next_observation=jnp.array([0.25, 0.75], dtype=jnp.float32),
    )
    agent._dreamer = _FixedDreamer(transition)  # type: ignore[assignment]
    stomp = state.oak_state.stomp_state
    q_next = jnp.max(
        agent.oak_agent.stomp_agent.base_learner.predict(
            stomp.base_learner_state,
            transition.next_observation,
        )
    )
    expected_td = transition.reward + transition.discount * q_next

    _, td_errors = agent._run_dreams(
        state.oak_state,
        state.world_model_state,
        state.buffer_state,
        jr.key(6),
    )

    np.testing.assert_allclose(td_errors[0], expected_td, rtol=1e-6, atol=1e-6)


def test_sample_one_hot_dream_backup_uses_named_rng_stream_and_isolates_real_state() -> None:
    agent = PrototypeAgent(
        _prototype_config(
            world_model=True,
            n_dreams=1,
            dream_next_observation_mode="sample_one_hot",
        )
    )
    initial_obs = jnp.array([1.0, 0.0], dtype=jnp.float32)
    state = agent.start(agent.init(jr.key(5)), initial_obs)
    state = _set_state_sensitive_q_values(state)
    assert agent._buffer is not None
    state = state.replace(
        buffer_state=agent._buffer.add(state.buffer_state, initial_obs)
    )
    transition = DreamTransition(
        observation=initial_obs,
        action=jnp.array(0, dtype=jnp.int32),
        reward=jnp.array(1.2, dtype=jnp.float32),
        discount=jnp.array(0.4, dtype=jnp.float32),
        next_observation=jnp.array([0.25, 0.75], dtype=jnp.float32),
    )
    agent._dreamer = _FixedDreamer(transition)  # type: ignore[assignment]
    dream_key = jr.key(6)
    observation_key = jr.fold_in(
        jr.fold_in(dream_key, _DREAM_NEXT_OBSERVATION_STREAM_TAG),
        0,
    )
    sampled_observation, valid = _sample_one_hot_dream_observation(
        transition.next_observation,
        observation_key,
    )
    assert bool(valid)
    stomp = state.oak_state.stomp_state
    q_next = jnp.max(
        agent.oak_agent.stomp_agent.base_learner.predict(
            stomp.base_learner_state,
            sampled_observation,
        )
    )
    expected_td = transition.reward + transition.discount * q_next

    dreamed, td_errors = agent._run_dreams(
        state.oak_state,
        state.world_model_state,
        state.buffer_state,
        dream_key,
    )

    np.testing.assert_allclose(td_errors[0], expected_td, rtol=1e-6, atol=1e-6)
    raw_q_next = jnp.max(
        agent.oak_agent.stomp_agent.base_learner.predict(
            stomp.base_learner_state,
            transition.next_observation,
        )
    )
    raw_td = transition.reward + transition.discount * raw_q_next
    assert not bool(jnp.isclose(td_errors[0], raw_td))
    normalized = dreamed.replace(
        stomp_state=dreamed.stomp_state.replace(
            base_learner_state=state.oak_state.stomp_state.base_learner_state
        )
    )
    chex.assert_trees_all_equal(normalized, state.oak_state)


def test_invalid_sample_one_hot_projection_rejects_entire_dream_backup() -> None:
    agent = PrototypeAgent(
        _prototype_config(
            world_model=True,
            n_dreams=1,
            dream_next_observation_mode="sample_one_hot",
        )
    )
    initial_obs = jnp.array([1.0, 0.0], dtype=jnp.float32)
    state = agent.start(agent.init(jr.key(5)), initial_obs)
    assert agent._buffer is not None
    state = state.replace(
        buffer_state=agent._buffer.add(state.buffer_state, initial_obs)
    )
    agent._dreamer = _FixedDreamer(  # type: ignore[assignment]
        DreamTransition(
            observation=initial_obs,
            action=jnp.array(0, dtype=jnp.int32),
            reward=jnp.array(1.2, dtype=jnp.float32),
            discount=jnp.array(0.4, dtype=jnp.float32),
            next_observation=jnp.array([-1.0, 0.0], dtype=jnp.float32),
        )
    )

    dreamed, td_errors = agent._run_dreams(
        state.oak_state,
        state.world_model_state,
        state.buffer_state,
        jr.key(6),
    )

    chex.assert_trees_all_equal(td_errors, jnp.zeros(1))
    chex.assert_trees_all_equal(dreamed, state.oak_state)


@pytest.mark.parametrize("discount", [0.0, 0.4])
def test_dream_backup_is_caused_by_predicted_discount(discount: float) -> None:
    agent = PrototypeAgent(_prototype_config(world_model=True, n_dreams=1))
    initial_obs = jnp.array([1.0, 0.0], dtype=jnp.float32)
    next_obs = jnp.array([0.0, 1.0], dtype=jnp.float32)
    state = agent.start(agent.init(jr.key(5)), initial_obs)
    assert agent._buffer is not None
    buffer_state = agent._buffer.add(state.buffer_state, initial_obs)
    state = state.replace(buffer_state=buffer_state)
    action = jnp.array(0, dtype=jnp.int32)
    reward = jnp.array(1.2, dtype=jnp.float32)
    transition = DreamTransition(
        observation=initial_obs,
        action=action,
        reward=reward,
        discount=jnp.array(discount, dtype=jnp.float32),
        next_observation=next_obs,
    )
    agent._dreamer = _FixedDreamer(transition)  # type: ignore[assignment]
    stomp = state.oak_state.stomp_state
    q_previous = agent.oak_agent.stomp_agent.base_learner.predict(
        stomp.base_learner_state,
        initial_obs,
    )[action]
    q_next = jnp.max(
        agent.oak_agent.stomp_agent.base_learner.predict(
            stomp.base_learner_state,
            next_obs,
        )
    )
    expected_td = reward - stomp.base_average_reward + discount * q_next - q_previous

    _, td_errors = agent._run_dreams(
        state.oak_state,
        state.world_model_state,
        state.buffer_state,
        jr.key(6),
    )

    np.testing.assert_allclose(td_errors[0], expected_td, rtol=1e-6, atol=1e-6)


def test_explicit_scan_matches_repeated_transition_updates() -> None:
    agent = PrototypeAgent(_prototype_config())
    initial_obs = jnp.array([0.2, -0.3], dtype=jnp.float32)
    initial = agent.start(agent.init(jr.key(7)), initial_obs)
    rewards = jnp.array([0.5, -0.2, 1.0], dtype=jnp.float32)
    observations = jnp.array(
        [[0.4, -0.1], [0.0, 0.5], [-0.3, 0.2]],
        dtype=jnp.float32,
    )
    discounts = jnp.array([0.9, 0.4, 0.0], dtype=jnp.float32)

    loop_state = initial
    loop_actions = []
    loop_tds = []
    loop_averages = []
    for reward, observation, discount in zip(
        rewards,
        observations,
        discounts,
        strict=True,
    ):
        result = agent.update_transition(
            loop_state,
            _explicit_transition(
                loop_state,
                reward=reward,
                next_observation=observation,
                discount=discount,
            ),
        )
        loop_state = result.state
        loop_actions.append(result.action)
        loop_tds.append(result.oak_td_error)
        loop_averages.append(result.oak_average_reward)

    scanned = agent.scan(
        initial,
        rewards,
        observations,
        discounts=discounts,
    )

    chex.assert_trees_all_close(
        _materialize_typed_keys(scanned.state),
        _materialize_typed_keys(loop_state),
        atol=1e-6,
    )
    chex.assert_trees_all_equal(scanned.actions, jnp.stack(loop_actions))
    chex.assert_trees_all_close(scanned.oak_td_errors, jnp.stack(loop_tds), atol=1e-6)
    chex.assert_trees_all_close(
        scanned.oak_average_rewards,
        jnp.stack(loop_averages),
        atol=1e-6,
    )


@pytest.mark.parametrize("discount", [-0.1, 1.1, np.nan, np.inf])
def test_explicit_transition_rejects_invalid_discount(discount: float) -> None:
    agent = PrototypeAgent(_prototype_config())
    state = agent.start(agent.init(jr.key(8)), jnp.zeros(2, dtype=jnp.float32))

    result = agent.update_transition(
        state,
        _explicit_transition(
            state,
            reward=0.0,
            next_observation=jnp.ones(2, dtype=jnp.float32),
            discount=discount,
        ),
    )

    chex.assert_trees_all_equal(
        _materialize_typed_keys(result.state),
        _materialize_typed_keys(state),
    )
    assert not bool(result.transition_diagnostics.valid)
    assert bool(result.transition_diagnostics.rejected)
    assert float(result.oak_td_error) == 0.0


def test_traced_invalid_discount_is_an_atomic_noop() -> None:
    agent = PrototypeAgent(_prototype_config())
    state = agent.start(agent.init(jr.key(9)), jnp.zeros(2, dtype=jnp.float32))

    @jax.jit
    def update(discount: jax.Array) -> PrototypeUpdateResult:
        return agent.update_transition(
            state,
            _explicit_transition(
                state,
                reward=0.0,
                next_observation=jnp.ones(2, dtype=jnp.float32),
                discount=discount,
            ),
        )

    result = update(jnp.array(1.5, dtype=jnp.float32))
    chex.assert_trees_all_equal(
        _materialize_typed_keys(result.state),
        _materialize_typed_keys(state),
    )
    assert not bool(result.transition_diagnostics.valid)
    assert bool(result.transition_diagnostics.rejected)
    assert float(result.oak_td_error) == 0.0


def test_stomp_config_rejects_nonfinite_or_out_of_range_option_gamma() -> None:
    for gamma in (-0.1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="option_gamma"):
            _stomp_config(option_gamma=gamma)
