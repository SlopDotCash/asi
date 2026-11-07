"""Differential SARSA traces decay by the discount into the current state."""

from __future__ import annotations

import dataclasses

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    DifferentialSARSAState,
    migrate_legacy_differential_sarsa_state,
    run_differential_sarsa_from_arrays,
)
from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
from alberta_framework.steps.step9 import (
    Step9DreamingConfig,
    init_step9_state,
    make_step9_components,
    step9_update,
)

pytestmark = pytest.mark.unit

_ALPHA = 0.1
_LAMBDA = 0.8


def _tabular_agent() -> DifferentialSARSAAgent:
    return DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=1,
            q_step_size=_ALPHA,
            average_reward_step_size=0.0,
            trace_decay=_LAMBDA,
            epsilon_start=0.0,
            epsilon_end=0.0,
            use_bias=False,
        )
    )


def _one_hot(index: int, width: int = 4) -> jnp.ndarray:
    return jnp.zeros(width, dtype=jnp.float32).at[index].set(1.0)


def _step(
    agent: DifferentialSARSAAgent,
    state: DifferentialSARSAState,
    reward: float,
    next_index: int,
    discount: float,
) -> DifferentialSARSAState:
    result = agent.update(
        state,
        jnp.asarray(reward, dtype=jnp.float32),
        _one_hot(next_index),
        next_action=jnp.asarray(0, dtype=jnp.int32),
        discount=jnp.asarray(discount, dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    return result.state


def test_terminal_td_error_credits_the_whole_episode_trace() -> None:
    """s0 -> s1 -> s2 -> terminal(r=1, discount 0) credits all three states.

    Q starts at zero, so only the terminal TD error (1.0) is nonzero. The
    backward view gives z = (gamma lambda)^2 phi0 + gamma lambda phi1 + phi2
    with gamma lambda = 0.72, so Q moves by 0.1 * (0.5184, 0.72, 1.0).
    Decaying by the terminal discount instead wipes s0 and s1.
    """
    agent = _tabular_agent()
    state, _ = agent.start(agent.init(4, jr.key(0)), _one_hot(0))
    state = _step(agent, state, 0.0, 1, 0.9)
    state = _step(agent, state, 0.0, 2, 0.9)
    state = _step(agent, state, 1.0, 3, 0.0)

    chex.assert_trees_all_close(
        state.q_weights,
        jnp.asarray([[0.05184, 0.072, 0.1, 0.0]], dtype=jnp.float32),
        rtol=1e-6,
    )
    chex.assert_trees_all_equal(state.previous_discount, jnp.asarray(0.0, jnp.float32))


def test_next_episode_does_not_inherit_the_terminal_trace() -> None:
    """After a terminal transition, the next update credits only its own pair."""
    agent = _tabular_agent()
    state, _ = agent.start(agent.init(4, jr.key(0)), _one_hot(0))
    state = _step(agent, state, 0.0, 1, 0.9)
    state = _step(agent, state, 1.0, 2, 0.0)
    q_after_episode = state.q_weights

    # New episode starts in s3 and moves to s0 with reward 1.
    state, _ = agent.start_with_action(
        state, _one_hot(3), jnp.asarray(0, dtype=jnp.int32)
    )
    state = _step(agent, state, 1.0, 0, 0.9)

    delta = 1.0 + 0.9 * float(q_after_episode[0, 0])
    expected = np.asarray(q_after_episode).copy()
    expected[0, 3] += _ALPHA * delta
    chex.assert_trees_all_close(state.q_weights, jnp.asarray(expected), rtol=1e-6)
    chex.assert_trees_all_close(
        state.q_trace_weights,
        jnp.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32),
    )


def test_array_runner_matches_backward_view_with_varying_discounts() -> None:
    """The scan path follows z_t = gamma_t lambda z_{t-1} + phi_t exactly."""
    agent = _tabular_agent()
    state, _ = agent.start(agent.init(4, jr.key(0)), _one_hot(0))
    next_indices = [1, 2, 3, 0, 1, 2]
    discounts = [0.9, 0.5, 0.0, 1.0, 0.25, 0.9]
    rewards = [0.5, -1.0, 2.0, 0.0, 1.5, -0.5]

    result = run_differential_sarsa_from_arrays(
        agent,
        state,
        jnp.asarray(rewards, dtype=jnp.float32),
        jnp.stack([_one_hot(i) for i in next_indices]),
        discounts=jnp.asarray(discounts, dtype=jnp.float32),
    )

    weights = np.zeros(4, dtype=np.float32)
    trace = np.zeros(4, dtype=np.float32)
    prior = np.float32(1.0)
    current = 0
    for reward, nxt, discount in zip(rewards, next_indices, discounts, strict=True):
        phi = np.eye(4, dtype=np.float32)[current]
        delta = np.float32(reward) + np.float32(discount) * weights[nxt] - weights[current]
        trace = prior * np.float32(_LAMBDA) * trace + phi
        weights = weights + np.float32(_ALPHA) * delta * trace
        prior = np.float32(discount)
        current = nxt

    assert bool(jnp.all(result.updates_applied))
    np.testing.assert_allclose(np.asarray(result.state.q_weights[0]), weights, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(result.state.q_trace_weights[0]), trace, rtol=1e-6)


@pytest.mark.parametrize("bad", [-0.25, 1.5, float("nan"), float("inf")])
def test_out_of_range_previous_discount_rejects_the_update(bad: float) -> None:
    agent = _tabular_agent()
    state, _ = agent.start(agent.init(4, jr.key(0)), _one_hot(0))
    state = _step(agent, state, 0.0, 1, 0.9)
    tampered = state.replace(previous_discount=jnp.asarray(bad, dtype=jnp.float32))

    result = agent.update(
        tampered,
        jnp.asarray(1.0, dtype=jnp.float32),
        _one_hot(2),
        next_action=jnp.asarray(0, dtype=jnp.int32),
    )

    assert not bool(result.state_valid)
    assert not bool(result.update_applied)
    selected, _ = agent.select_action(tampered, _one_hot(1))
    assert int(selected) == -1


def test_previous_discount_shape_and_dtype_are_checked() -> None:
    agent = _tabular_agent()
    state, _ = agent.start(agent.init(4, jr.key(0)), _one_hot(0))
    with pytest.raises(TypeError, match="previous_discount must have dtype float32"):
        agent.update(
            state.replace(previous_discount=jnp.asarray(1.0, dtype=jnp.float16)),
            jnp.asarray(0.0, dtype=jnp.float32),
            _one_hot(1),
        )
    with pytest.raises(ValueError, match="previous_discount must have shape"):
        agent.update(
            state.replace(previous_discount=jnp.ones((1,), dtype=jnp.float32)),
            jnp.asarray(0.0, dtype=jnp.float32),
            _one_hot(1),
        )


_V1_FIELDS = (
    "q_weights",
    "q_bias",
    "q_trace_weights",
    "q_trace_bias",
    "average_reward",
    "last_observation",
    "last_action",
    "epsilon",
    "rng_key",
    "step_count",
    "birth_timestamp",
    "uptime_s",
)


def _historical_payload(state: DifferentialSARSAState, *, v2: bool) -> dict[str, object]:
    names = (*_V1_FIELDS, "step_words") if v2 else _V1_FIELDS
    return {name: getattr(state, name) for name in names}


@pytest.mark.parametrize("v2", [False, True])
def test_historical_manifests_migrate_with_a_unit_previous_discount(v2: bool) -> None:
    agent = _tabular_agent()
    state, _ = agent.start(agent.init(4, jr.key(0)), _one_hot(0))
    state = _step(agent, state, 0.0, 1, 0.9)
    state = _step(agent, state, 0.0, 2, 0.5)
    legacy = _historical_payload(state, v2=v2)

    migrated = migrate_legacy_differential_sarsa_state(legacy)

    chex.assert_trees_all_equal(migrated.previous_discount, jnp.asarray(1.0, jnp.float32))
    chex.assert_trees_all_equal(migrated.step_words, state.step_words)
    chex.assert_trees_all_equal(migrated.q_trace_weights, state.q_trace_weights)
    assert {f.name for f in dataclasses.fields(migrated)} == {  # type: ignore[arg-type]
        f.name for f in dataclasses.fields(state)  # type: ignore[arg-type]
    }
    assert bool(
        agent.update(
            migrated,
            jnp.asarray(0.0, dtype=jnp.float32),
            _one_hot(3),
            next_action=jnp.asarray(0, dtype=jnp.int32),
        ).update_applied
    )


def test_current_and_mixed_manifests_are_not_legacy() -> None:
    agent = _tabular_agent()
    state = agent.init(4, jr.key(0))
    current = {f.name: getattr(state, f.name) for f in dataclasses.fields(state)}  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="manifest"):
        migrate_legacy_differential_sarsa_state(current)
    v1_plus_discount = {
        **_historical_payload(state, v2=False),
        "previous_discount": state.previous_discount,
    }
    with pytest.raises(ValueError, match="manifest"):
        migrate_legacy_differential_sarsa_state(v1_plus_discount)


def _step9_dream_config() -> Step9DreamingConfig:
    return Step9DreamingConfig(
        control=Step6DifferentialSARSAConfig(
            n_actions=1,
            q_step_size=0.0,
            average_reward_step_size=0.0,
            trace_decay=0.5,
            epsilon_start=0.0,
        ),
        observation_dim=1,
        n_actions=1,
        model_hidden_sizes=(),
        model_step_size=0.0,
        model_sparsity=0.0,
        model_use_layer_norm=False,
        model_gamma=0.9,
        dreaming_warmup_steps=0,
        dreaming_max_model_error=1e30,
        behavior_model_step_size=0.0,
        planning_budget=1,
        dream_rollout_horizon=1,
        dream_candidate_count=1,
        buffer_capacity=2,
    )


def test_step9_dream_discount_does_not_reach_the_real_trace_decay() -> None:
    """An accepted dream with predicted discount 0 leaves the real carry at gamma."""
    config = _step9_dream_config()
    agent, model, buffer = make_step9_components(config)
    state = init_step9_state(
        agent,
        model,
        buffer,
        key=jr.key(2),
        initial_observation=jnp.zeros(1, dtype=jnp.float32),
    )
    learner_state = state.world_model_state.learner_state
    head_params = learner_state.head_params
    discount_head = len(head_params.biases) - 1
    head_params = head_params.replace(
        weights=tuple(jnp.zeros_like(weight) for weight in head_params.weights),
        biases=tuple(jnp.zeros_like(bias) for bias in head_params.biases),
    )
    assert discount_head >= 0
    state = state.replace(
        world_model_state=state.world_model_state.replace(
            learner_state=learner_state.replace(head_params=head_params),
        )
    )

    result = step9_update(
        config,
        agent,
        model,
        buffer,
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.ones(1, dtype=jnp.float32),
    )

    assert bool(result.dream_accepted[0])
    chex.assert_trees_all_close(
        result.state.control_state.previous_discount,
        jnp.asarray(0.9, dtype=jnp.float32),
    )
