"""Tests for Step 4b actor-critic core."""

from types import MappingProxyType

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework import ActorCriticAgent as TopLevelActorCriticAgent
from alberta_framework.core import ActorCriticAgent as CoreActorCriticAgent
from alberta_framework.core.actor_critic import (
    ActorCriticAgent,
    ActorCriticConfig,
    _require_discrete_state_resources,
    run_actor_critic_from_arrays,
)
from alberta_framework.core.optimizers import ObGDBounding


def _assert_actor_critic_numeric_state_finite(state) -> None:  # type: ignore[no-untyped-def]
    chex.assert_tree_all_finite(
        (
            state.actor_weights,
            state.actor_bias,
            state.critic_weights,
            state.critic_bias,
            state.actor_trace_weights,
            state.actor_trace_bias,
            state.critic_trace_weights,
            state.critic_trace_bias,
            state.last_observation,
        )
    )


def test_actor_critic_init_predict_and_start_shapes() -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=3))
    state = agent.init(feature_dim=4, key=jr.key(0))
    obs = jnp.array([1.0, -1.0, 0.5, 2.0], dtype=jnp.float32)

    policy = agent.policy(state, obs)
    value = agent.value(state, obs)
    next_state, action, start_policy = agent.start(state, obs)

    chex.assert_shape(policy, (3,))
    chex.assert_shape(start_policy, (3,))
    chex.assert_shape(value, ())
    chex.assert_shape(action, ())
    chex.assert_trees_all_close(jnp.sum(policy), 1.0)
    _assert_actor_critic_numeric_state_finite(next_state)
    assert int(next_state.last_action) in range(3)


def test_actor_critic_update_changes_actor_and_critic() -> None:
    config = ActorCriticConfig(
        n_actions=2,
        gamma=0.9,
        actor_step_size=0.1,
        critic_step_size=0.2,
        actor_lamda=0.8,
        critic_lamda=0.7,
    )
    agent = ActorCriticAgent(config)
    state = agent.init(feature_dim=3, key=jr.key(1))
    state, _action, _policy = agent.start(state, jnp.array([1.0, 0.0, -1.0], dtype=jnp.float32))

    result = agent.update(
        state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=jnp.array([0.0, 1.0, 0.5], dtype=jnp.float32),
        terminated=jnp.array(False),
    )

    assert int(result.state.step_count) == 1
    assert float(result.td_error) == 1.0
    assert not jnp.allclose(result.state.actor_weights, state.actor_weights)
    assert not jnp.allclose(result.state.critic_weights, state.critic_weights)
    _assert_actor_critic_numeric_state_finite(result.state)
    chex.assert_tree_all_finite((result.policy, result.value, result.next_value, result.td_error))


def test_actor_critic_temperature_scales_actor_gradient() -> None:
    obs = jnp.array([1.0, -2.0], dtype=jnp.float32)
    next_obs = jnp.array([0.0, 0.0], dtype=jnp.float32)
    warm_agent = ActorCriticAgent(
        ActorCriticConfig(
            n_actions=2,
            gamma=0.9,
            actor_step_size=0.1,
            critic_step_size=0.0,
            actor_lamda=0.0,
            critic_lamda=0.0,
            temperature=1.0,
        )
    )
    cool_agent = ActorCriticAgent(
        ActorCriticConfig(
            n_actions=2,
            gamma=0.9,
            actor_step_size=0.1,
            critic_step_size=0.0,
            actor_lamda=0.0,
            critic_lamda=0.0,
            temperature=0.5,
        )
    )
    warm_state = warm_agent.init(feature_dim=2, key=jr.key(10)).replace(  # type: ignore[attr-defined]
        last_observation=obs,
        last_action=jnp.array(0, dtype=jnp.int32),
    )
    cool_state = cool_agent.init(feature_dim=2, key=jr.key(10)).replace(  # type: ignore[attr-defined]
        last_observation=obs,
        last_action=jnp.array(0, dtype=jnp.int32),
    )

    warm = warm_agent.update(
        warm_state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=next_obs,
        discount=jnp.array(0.0, dtype=jnp.float32),
    )
    cool = cool_agent.update(
        cool_state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=next_obs,
        discount=jnp.array(0.0, dtype=jnp.float32),
    )

    chex.assert_trees_all_close(
        cool.state.actor_weights,
        2.0 * warm.state.actor_weights,
        atol=1e-6,
        rtol=1e-6,
    )


def test_actor_critic_terminal_update_resets_traces() -> None:
    agent = ActorCriticAgent(
        ActorCriticConfig(
            n_actions=2,
            actor_step_size=0.1,
            critic_step_size=0.1,
            actor_lamda=0.9,
            critic_lamda=0.9,
        )
    )
    state = agent.init(feature_dim=2, key=jr.key(11)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([1.0, 0.5], dtype=jnp.float32),
        last_action=jnp.array(1, dtype=jnp.int32),
        actor_trace_weights=jnp.ones((2, 2), dtype=jnp.float32),
        actor_trace_bias=jnp.ones((2,), dtype=jnp.float32),
        critic_trace_weights=jnp.ones((2,), dtype=jnp.float32),
        critic_trace_bias=jnp.array(1.0, dtype=jnp.float32),
    )

    result = agent.update(
        state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=jnp.array([0.0, 1.0], dtype=jnp.float32),
        terminated=jnp.array(True),
    )

    chex.assert_trees_all_close(
        (
            result.state.actor_trace_weights,
            result.state.actor_trace_bias,
            result.state.critic_trace_weights,
            result.state.critic_trace_bias,
        ),
        (
            jnp.zeros_like(result.state.actor_trace_weights),
            jnp.zeros_like(result.state.actor_trace_bias),
            jnp.zeros_like(result.state.critic_trace_weights),
            jnp.array(0.0, dtype=jnp.float32),
        ),
    )
    assert not jnp.allclose(result.state.actor_weights, state.actor_weights)


def test_actor_critic_explicit_discount_semantics() -> None:
    agent = ActorCriticAgent(
        ActorCriticConfig(
            n_actions=2,
            gamma=0.9,
            actor_step_size=0.0,
            critic_step_size=0.0,
        )
    )
    state = agent.init(feature_dim=2, key=jr.key(12)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
        critic_weights=jnp.array([2.0, 4.0], dtype=jnp.float32),
        critic_bias=jnp.array(0.5, dtype=jnp.float32),
    )
    next_obs = jnp.array([0.0, 1.0], dtype=jnp.float32)

    explicit = agent.update(
        state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=next_obs,
        terminated=jnp.array(True),
        discount=jnp.array(0.25, dtype=jnp.float32),
    )
    legacy = agent.update(
        state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=next_obs,
        terminated=jnp.array(True),
    )

    chex.assert_trees_all_close(explicit.td_error, jnp.array(-0.375, dtype=jnp.float32))
    chex.assert_trees_all_close(legacy.td_error, jnp.array(-1.5, dtype=jnp.float32))


def test_actor_critic_update_is_jittable() -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2))
    state = agent.init(feature_dim=2, key=jr.key(2))
    state, _action, _policy = agent.start(state, jnp.array([1.0, 0.0], dtype=jnp.float32))

    update = jax.jit(agent.update)
    result = update(
        state,
        jnp.array(0.5, dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
        jnp.array(False),
    )

    chex.assert_shape(result.policy, (2,))
    assert int(result.state.step_count) == 1


def test_run_actor_critic_from_arrays_scan() -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2))
    state = agent.init(feature_dim=2, key=jr.key(3))
    observations = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32)
    next_observations = jnp.array([[0.0, 1.0], [1.0, 1.0], [0.5, -0.5]], dtype=jnp.float32)
    rewards = jnp.array([1.0, 0.0, -1.0], dtype=jnp.float32)
    terminated = jnp.array([False, False, True])

    result = run_actor_critic_from_arrays(
        agent, state, observations, rewards, terminated, next_observations
    )

    chex.assert_shape(result.actions, (3,))
    chex.assert_shape(result.policies, (3, 2))
    chex.assert_shape(result.values, (3,))
    chex.assert_shape(result.td_errors, (3,))
    assert int(result.state.step_count) == 3
    _assert_actor_critic_numeric_state_finite(result.state)
    chex.assert_tree_all_finite((result.policies, result.values, result.td_errors))


def test_run_actor_critic_from_arrays_sampled_policies_align_with_actions() -> None:
    """On-policy rows report the distribution that sampled each action."""
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=3, actor_step_size=0.5, gamma=0.9))
    state = agent.init(feature_dim=2, key=jr.key(5))
    state = state.replace(  # type: ignore[attr-defined]
        actor_weights=jnp.array([[3.0, 0.0], [0.0, 3.0], [0.0, 0.0]], dtype=jnp.float32)
    )
    observations = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    next_observations = jnp.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    rewards = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
    terminated = jnp.array([False, False, False])

    result = run_actor_critic_from_arrays(
        agent, state, observations, rewards, terminated, next_observations
    )

    loop_state = state
    for step in range(3):
        expected_policy = agent.policy(loop_state, observations[step])
        chex.assert_trees_all_close(result.policies[step], expected_policy)
        started, _action, _probs = agent.start(loop_state, observations[step])
        started = started.replace(last_action=result.actions[step])  # type: ignore[attr-defined]
        loop_state = agent.update(
            started, rewards[step], next_observations[step], discount=0.9
        ).state
    for step in range(3):
        assert float(result.policies[step][result.actions[step]]) > 0.5


def test_run_actor_critic_from_arrays_fixed_actions_matches_loop() -> None:
    agent = ActorCriticAgent(
        ActorCriticConfig(
            n_actions=2,
            gamma=0.9,
            actor_step_size=0.05,
            critic_step_size=0.1,
            actor_lamda=0.7,
            critic_lamda=0.6,
        )
    )
    state = agent.init(feature_dim=2, key=jr.key(13))
    state = state.replace(  # type: ignore[attr-defined]
        actor_weights=jnp.array([[0.0, 3.0], [3.0, 0.0]], dtype=jnp.float32)
    )
    observations = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32)
    next_observations = jnp.array([[0.0, 1.0], [1.0, 1.0], [0.5, -0.5]], dtype=jnp.float32)
    rewards = jnp.array([1.0, 0.0, -1.0], dtype=jnp.float32)
    actions = jnp.array([0, 1, 0], dtype=jnp.int32)
    discounts = jnp.array([0.9, 0.9, 0.0], dtype=jnp.float32)

    scan_result = run_actor_critic_from_arrays(
        agent,
        state,
        observations,
        rewards,
        terminated=None,
        next_observations=next_observations,
        actions=actions,
        discounts=discounts,
    )

    loop_state = state
    loop_td_errors = []
    loop_policies = []
    for obs, reward, action, discount, next_obs in zip(
        observations, rewards, actions, discounts, next_observations, strict=True
    ):
        loop_policies.append(agent.policy(loop_state, obs))
        loop_state = loop_state.replace(  # type: ignore[attr-defined]
            last_observation=obs,
            last_action=action,
        )
        loop_result = agent.update(
            loop_state,
            reward=reward,
            observation=next_obs,
            discount=discount,
        )
        loop_state = loop_result.state
        loop_td_errors.append(loop_result.td_error)

    chex.assert_trees_all_close(scan_result.actions, actions)
    chex.assert_trees_all_close(scan_result.policies, jnp.stack(loop_policies))
    # The first supplied action is deliberately unlikely under the agent. The
    # returned row is a target-policy evaluation, not claimed behavior provenance.
    assert float(scan_result.policies[0, scan_result.actions[0]]) < 0.1
    chex.assert_trees_all_close(scan_result.td_errors, jnp.stack(loop_td_errors))
    chex.assert_trees_all_close(scan_result.state.actor_weights, loop_state.actor_weights)
    chex.assert_trees_all_close(scan_result.state.critic_weights, loop_state.critic_weights)


def test_actor_critic_config_round_trip_with_bounder() -> None:
    agent = ActorCriticAgent(
        ActorCriticConfig(n_actions=4, gamma=0.95, actor_step_size=0.03),
        bounder=ObGDBounding(kappa=3.0),
    )

    reconstructed = ActorCriticAgent.from_config(agent.to_config())

    assert reconstructed.config == agent.config
    assert reconstructed.bounder is not None
    assert reconstructed.bounder.to_config() == {"type": "ObGDBounding", "kappa": 3.0}


def test_actor_critic_bounder_hook_runs() -> None:
    agent = ActorCriticAgent(
        ActorCriticConfig(
            n_actions=2,
            actor_step_size=100.0,
            critic_step_size=100.0,
        ),
        bounder=ObGDBounding(kappa=2.0),
    )
    state = agent.init(feature_dim=2, key=jr.key(4))
    state, _action, _policy = agent.start(state, jnp.array([1.0, 1.0], dtype=jnp.float32))

    result = agent.update(
        state,
        reward=jnp.array(10.0, dtype=jnp.float32),
        observation=jnp.array([0.5, -1.0], dtype=jnp.float32),
        terminated=jnp.array(False),
    )

    assert float(result.bound_metric) < 1.0
    _assert_actor_critic_numeric_state_finite(result.state)
    chex.assert_tree_all_finite((result.policy, result.value, result.next_value, result.td_error))


def test_actor_critic_infinite_reward_with_obgd_does_not_poison_weights() -> None:
    """Inf TD error zeros the ObGD step, then td_error*step is 0*inf=NaN."""
    agent = ActorCriticAgent(
        ActorCriticConfig(n_actions=2, actor_step_size=0.1, critic_step_size=0.1),
        bounder=ObGDBounding(kappa=2.0),
    )
    state = agent.init(feature_dim=2, key=jr.key(0))
    state, _, _ = agent.start(state, jnp.ones(2, dtype=jnp.float32))

    poisoned = agent.update(
        state,
        reward=jnp.array(jnp.inf, dtype=jnp.float32),
        observation=jnp.array([0.5, -1.0], dtype=jnp.float32),
        terminated=jnp.array(False),
    )
    assert bool(jnp.all(jnp.isfinite(poisoned.state.actor_weights)))
    assert bool(jnp.all(jnp.isfinite(poisoned.state.critic_weights)))
    chex.assert_trees_all_close(poisoned.state.actor_weights, state.actor_weights)
    chex.assert_trees_all_close(poisoned.state.critic_weights, state.critic_weights)
    chex.assert_trees_all_equal(jr.key_data(poisoned.state.rng_key), jr.key_data(state.rng_key))
    chex.assert_trees_all_close(
        poisoned.state.replace(rng_key=jr.key_data(poisoned.state.rng_key)),
        state.replace(rng_key=jr.key_data(state.rng_key)),
    )
    assert not bool(poisoned.update_applied)
    assert float(poisoned.td_error) == 0.0
    chex.assert_trees_all_close(poisoned.policy, jnp.zeros_like(poisoned.policy))

    recovered = agent.update(
        poisoned.state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=jnp.array([0.0, 1.0], dtype=jnp.float32),
        terminated=jnp.array(False),
    )
    assert bool(jnp.all(jnp.isfinite(recovered.state.actor_weights)))
    assert bool(jnp.all(jnp.isfinite(recovered.state.critic_weights)))
    assert bool(recovered.update_applied)


def test_actor_critic_terminal_does_not_multiply_inf_next_value() -> None:
    """gamma=0 * inf V(s') is 0*inf = NaN and would freeze a terminal update."""
    agent = ActorCriticAgent(
        ActorCriticConfig(n_actions=2, actor_step_size=0.1, critic_step_size=0.1)
    )
    huge = jnp.float32(1e38)
    state = agent.init(feature_dim=2, key=jr.key(1)).replace(  # type: ignore[attr-defined]
        last_observation=jnp.array([0.0, 1.0], dtype=jnp.float32),
        last_action=jnp.array(0, dtype=jnp.int32),
        critic_weights=jnp.array([huge, 0.0], dtype=jnp.float32),
        critic_bias=jnp.array(0.0, dtype=jnp.float32),
    )
    next_obs = jnp.array([huge, 0.0], dtype=jnp.float32)
    raw = jnp.asarray(0.0, dtype=jnp.float32) * (huge * huge)
    assert not bool(jnp.isfinite(raw))

    result = agent.update(
        state,
        reward=jnp.array(3.0, dtype=jnp.float32),
        observation=next_obs,
        terminated=jnp.array(True),
    )
    assert bool(result.update_applied)
    chex.assert_trees_all_close(result.td_error, jnp.array(3.0, dtype=jnp.float32))
    _assert_actor_critic_numeric_state_finite(result.state)


def test_actor_critic_exports() -> None:
    assert TopLevelActorCriticAgent is ActorCriticAgent
    assert CoreActorCriticAgent is ActorCriticAgent


def test_actor_critic_integer_and_scalar_validation() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        ActorCriticConfig(n_actions=True)  # type: ignore[arg-type]
    assert ActorCriticConfig(n_actions=1).n_actions == 1
    with pytest.raises(ValueError, match="n_actions"):
        ActorCriticConfig(n_actions=2.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="gamma"):
        ActorCriticConfig(n_actions=2, gamma=1.5)
    with pytest.raises(ValueError, match="actor_step_size"):
        ActorCriticConfig(n_actions=2, actor_step_size=-0.1)

    cfg = ActorCriticConfig(n_actions=np.int32(4))
    assert cfg.n_actions == 4
    assert type(cfg.n_actions) is int

    agent = ActorCriticAgent(cfg)
    with pytest.raises(ValueError, match="feature_dim"):
        agent.init(feature_dim=True, key=jr.key(0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_dim"):
        agent.init(feature_dim=0, key=jr.key(0))

    state = agent.init(feature_dim=np.int32(5), key=jr.key(0))
    assert state.actor_weights.shape == (4, 5)


class _HostileFloat(float):
    def as_integer_ratio(self) -> tuple[int, int]:
        raise RuntimeError("hostile hook executed")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gamma", _HostileFloat(0.5)),
        ("gamma", np.float64(1e-100)),
        ("actor_step_size", np.float64(1e-100)),
        ("critic_lamda", 1e100),
        ("temperature", True),
    ],
)
def test_actor_critic_rejects_invalid_float32_config(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        ActorCriticConfig(n_actions=2, **{field: value})  # type: ignore[arg-type]


def test_actor_critic_mapping_config_and_resource_boundary() -> None:
    config = ActorCriticConfig(n_actions=np.uint16(2), gamma=np.float32(0.5)).to_config()
    clone = ActorCriticConfig.from_config(MappingProxyType(config))
    assert clone.to_config() == config
    _require_discrete_state_resources(1, 107_374_180)
    with pytest.raises(ValueError, match="state exceeds"):
        _require_discrete_state_resources(1, 107_374_181)


class _HostileArray:
    def __jax_array__(self) -> jax.Array:
        raise RuntimeError("hostile array hook executed")


def test_actor_critic_serialized_schema_and_hostile_runtime_inputs_fail_closed() -> None:
    config = ActorCriticConfig(n_actions=2).to_config()
    with pytest.raises(ValueError, match="serialized schema"):
        ActorCriticConfig.from_config({**config, "unknown": 1})
    with pytest.raises(ValueError, match="exact JSON"):
        ActorCriticConfig.from_config({**config, "n_actions": np.int32(2)})

    payload = ActorCriticAgent(ActorCriticConfig(n_actions=2)).to_config()
    with pytest.raises(ValueError, match="type differs"):
        ActorCriticAgent.from_config({**payload, "type": "wrong"})
    with pytest.raises(ValueError, match="serialized schema"):
        ActorCriticAgent.from_config({**payload, "unknown": None})
    with pytest.raises(ValueError, match="Bounder"):
        ActorCriticAgent(ActorCriticConfig(n_actions=2), bounder=object())  # type: ignore[arg-type]

    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2))
    with pytest.raises(ValueError, match="Threefry"):
        agent.init(2, _HostileArray())  # type: ignore[arg-type]
    state = agent.init(2, jr.key(0))
    with pytest.raises(ValueError, match="trusted array metadata"):
        agent.policy.__wrapped__(agent, state, _HostileArray())  # type: ignore[attr-defined,arg-type]


@pytest.mark.parametrize("shape", [(), (1,), (1, 2), (2, 1), (3,)])
def test_actor_critic_rejects_wrong_observation_shapes(shape: tuple[int, ...]) -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2))
    state = agent.init(2, jr.key(0))
    malformed = jnp.zeros(shape, dtype=jnp.float32)
    with pytest.raises(ValueError, match="observation"):
        agent.policy(state, malformed)
    with pytest.raises(ValueError, match="observation"):
        agent.update(state, jnp.asarray(0.0), malformed)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.uint8, jnp.int64])
def test_run_actor_critic_from_arrays_rejects_integer_terminated(dtype: object) -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2))
    state = agent.init(2, jr.key(0))
    observations = jnp.zeros((2, 2), dtype=jnp.float32)
    rewards = jnp.zeros((2,), dtype=jnp.float32)
    with pytest.raises(TypeError, match="terminated must have dtype bool or float32"):
        run_actor_critic_from_arrays(
            agent,
            state,
            observations,
            rewards,
            jnp.array([0, 1], dtype=dtype),
            observations,
        )


def test_run_actor_critic_from_arrays_accepts_bool_and_float32_terminated() -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2))
    state = agent.init(2, jr.key(0))
    observations = jnp.zeros((2, 2), dtype=jnp.float32)
    rewards = jnp.zeros((2,), dtype=jnp.float32)
    for flags in (
        jnp.array([False, True], dtype=jnp.bool_),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
    ):
        result = run_actor_critic_from_arrays(
            agent, state, observations, rewards, flags, observations
        )
        assert result.td_errors.shape == (2,)


def test_actor_critic_state_contract_and_counter_saturation() -> None:
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2))
    state = agent.init(2, jr.key(0))
    malformed = state.replace(actor_weights=jnp.zeros((2,), dtype=jnp.float32))
    with pytest.raises(ValueError, match="actor_weights"):
        agent.policy(malformed, jnp.zeros((2,), dtype=jnp.float32))
    saturated = state.replace(
        last_action=jnp.asarray(0, dtype=jnp.int32),
        step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
    )
    result = agent.update(
        saturated,
        jnp.asarray(0.0),
        jnp.zeros((2,), dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert int(result.state.step_count) == 2**31 - 1


def test_actor_critic_select_action_exact_distribution() -> None:
    """select_action samples from exact log probabilities without artificial floor."""
    agent = ActorCriticAgent(ActorCriticConfig(n_actions=2))
    state = agent.init(2, jr.key(42))
    weights = jnp.asarray([[100.0, 0.0], [-100.0, 0.0]], dtype=jnp.float32)
    state = state.replace(actor_weights=weights)
    obs = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    action, _next_key, probs = agent.select_action(state, obs)
    assert int(action) == 0
    assert float(probs[0]) > 0.999
