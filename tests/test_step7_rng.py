"""Exact RNG-threading contracts for Step 7 real and planning updates.

Regression coverage for a defect where `_maybe_accept_planning_state` restored
the OLD `rng_key` into the carried control state whenever planning was rejected
(pre-warmup), freezing the planning RNG so every rejected planning iteration
re-sampled identical anchors and actions.  Transactionally rejected real or
rollout updates must likewise persist their already-consumed reserved key.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
from alberta_framework.steps.step7 import (
    Step7DynaConfig,
    Step7DynaState,
    _update_control_with_linear_rng,
    init_step7_state,
    make_step7_components,
    run_step7_scan,
    step7_update,
)
from alberta_framework.steps.step8 import Step8WorldModelConfig

OBS_DIM = 4
N_ACTIONS = 6


def _cfg(
    planning_steps: int,
    warmup: int,
    *,
    rollout_depth: int = 1,
) -> Step7DynaConfig:
    return Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=N_ACTIONS),
        world_model=Step8WorldModelConfig(observation_dim=OBS_DIM, n_actions=N_ACTIONS),
        planning_steps=planning_steps,
        planning_rollout_depth=rollout_depth,
        planning_warmup_steps=warmup,
        planning_memory_size=16,
        planning_strategy="random",
    )


def _init(cfg: Step7DynaConfig, seed: int = 0) -> tuple[object, object, Step7DynaState]:
    agent, model = make_step7_components(cfg)
    state = init_step7_state(
        agent,
        model,
        key=jr.key(seed),
        initial_observation=jnp.zeros(OBS_DIM),
        memory_size=cfg.planning_memory_size,
    )
    return agent, model, state


def _assert_key_equal(actual: jax.Array, expected: jax.Array) -> None:
    chex.assert_trees_all_equal(jr.key_data(actual), jr.key_data(expected))


def _assert_pairwise_distinct(keys: tuple[jax.Array, ...]) -> None:
    words = tuple(jr.key_data(key) for key in keys)
    for index, left in enumerate(words):
        for right in words[index + 1 :]:
            assert not bool(jnp.array_equal(left, right))


def _advance_planning_key(
    key: jax.Array,
    *,
    planning_steps: int,
    rollout_depth: int,
) -> jax.Array:
    for _ in range(planning_steps):
        scheduler_key, _action_key = jr.split(key)
        rollout_key, _anchor_key = jr.split(scheduler_key)
        key = rollout_key
        for _ in range(rollout_depth):
            key = jr.split(key, 4)[0]
    return key


class TestStep7RejectedPlanningRng:
    def test_rejected_planning_still_advances_rng_key(self) -> None:
        """Pre-warmup planning must consume RNG even though its output is dropped."""
        cfg = _cfg(planning_steps=2, warmup=100)
        agent, model, state = _init(cfg)
        result = step7_update(cfg, agent, model, state, jnp.array(0.0), jnp.zeros(OBS_DIM))
        assert not bool(jnp.all(result.planning_accepted))
        key_after_real = jr.key_data(result.real_control_result.state.rng_key)
        key_after_planning = jr.key_data(result.state.control_state.rng_key)
        assert not jnp.array_equal(key_after_real, key_after_planning), (
            "planning scan carried the pre-planning rng_key unchanged: rejected "
            "planning steps froze the RNG stream"
        )

    def test_rejected_planning_iterations_sample_distinct_actions(self) -> None:
        """Each rejected planning iteration must draw from a fresh key."""
        cfg = _cfg(planning_steps=8, warmup=100)
        agent, model, state = _init(cfg)
        result = step7_update(cfg, agent, model, state, jnp.array(0.0), jnp.zeros(OBS_DIM))
        assert not bool(jnp.all(result.planning_accepted))
        # With a frozen key every iteration re-samples the identical random
        # action; with threaded keys 8 uniform draws over 6 actions collide
        # all-equal only with probability 6**-7 (and the seed is fixed).
        assert int(jnp.unique(result.planning_actions).shape[0]) > 1, (
            "all planning iterations sampled the same action from a frozen rng_key"
        )

    def test_accepted_planning_advances_rng_key(self) -> None:
        """Accepted planning already threads the rollout key; keep it that way."""
        cfg = _cfg(planning_steps=2, warmup=0)
        agent, model, state = _init(cfg)
        result = step7_update(cfg, agent, model, state, jnp.array(0.0), jnp.zeros(OBS_DIM))
        assert bool(jnp.all(result.planning_accepted))
        key_after_real = jr.key_data(result.real_control_result.state.rng_key)
        key_after_planning = jr.key_data(result.state.control_state.rng_key)
        assert not jnp.array_equal(key_after_real, key_after_planning)


def test_zero_planning_rejected_real_update_keeps_the_original_rng_path() -> None:
    """Planning zero must not alter even a transactionally rejected real step."""
    cfg = _cfg(planning_steps=0, warmup=0)
    agent, model, state = _init(cfg, seed=13)
    saturated_control = state.control_state.replace(
        step_count=jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32),
        step_words=jnp.full(
            (2,),
            jnp.iinfo(jnp.uint32).max,
            dtype=jnp.uint32,
        ),
    )
    state = state.replace(control_state=saturated_control)
    reward = jnp.array(0.25, dtype=jnp.float32)
    next_observation = jnp.full((OBS_DIM,), 0.125, dtype=jnp.float32)
    expected = agent.update(saturated_control, reward, next_observation)
    result = step7_update(cfg, agent, model, state, reward, next_observation)

    assert not bool(expected.update_applied)
    chex.assert_trees_all_equal(result.real_control_result, expected)
    chex.assert_trees_all_equal(result.state.control_state, expected.state)


def test_rejected_real_update_reserves_its_advanced_key_before_planning() -> None:
    """Failed real action sampling cannot share siblings with the planner."""
    cfg = _cfg(planning_steps=1, warmup=0, rollout_depth=2)
    agent, model, state = _init(cfg, seed=17)
    parent = state.control_state.rng_key
    corrupt = state.replace(
        control_state=state.control_state.replace(
            q_weights=jnp.full_like(state.control_state.q_weights, jnp.nan)
        )
    )
    result = step7_update(
        cfg,
        agent,
        model,
        corrupt,
        jnp.array(0.25, dtype=jnp.float32),
        jnp.full((OBS_DIM,), 0.125, dtype=jnp.float32),
    )

    assert not bool(result.real_control_result.update_applied)
    real_reserved, real_explore, real_tie_break, real_random_action = jr.split(
        parent, 4
    )
    scheduler_key, planning_action = jr.split(real_reserved)
    rollout_root, anchor = jr.split(scheduler_key)
    expected_final = jr.split(jr.split(rollout_root, 4)[0], 4)[0]
    _assert_key_equal(result.real_control_result.state.rng_key, real_reserved)
    _assert_key_equal(result.state.control_state.rng_key, expected_final)
    _assert_pairwise_distinct(
        (
            real_explore,
            real_tie_break,
            real_random_action,
            planning_action,
            anchor,
            rollout_root,
        )
    )


def test_invalid_next_observation_rejects_before_building_a_control_candidate() -> None:
    """The ``-1`` selection sentinel must never index the SARSA target."""
    cfg = _cfg(planning_steps=1, warmup=0)
    agent, _model, state = _init(cfg, seed=23)
    control = state.control_state
    result = _update_control_with_linear_rng(
        agent,
        control,
        jnp.array(0.25, dtype=jnp.float32),
        jnp.full((OBS_DIM,), jnp.nan, dtype=jnp.float32),
    )

    assert not bool(result.update_applied)
    assert not bool(result.input_valid)
    assert bool(result.candidate_state_finite)
    expected_key = jr.split(control.rng_key, 4)[0]
    chex.assert_trees_all_equal(
        result.state,
        control.replace(rng_key=expected_key),
    )


def test_rejected_rollout_updates_advance_their_local_key_each_depth() -> None:
    """A rejected depth-two rollout cannot sample twice from the same root."""
    cfg = _cfg(planning_steps=1, warmup=0, rollout_depth=2)
    agent, model, state = _init(cfg, seed=18)
    maximum = jnp.iinfo(jnp.uint32).max
    almost_saturated = state.control_state.replace(
        step_count=jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32),
        step_words=jnp.asarray([maximum, maximum - 1], dtype=jnp.uint32),
    )
    state = state.replace(control_state=almost_saturated)
    result = step7_update(
        cfg,
        agent,
        model,
        state,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.zeros((OBS_DIM,), dtype=jnp.float32),
    )

    assert bool(result.real_control_result.update_applied)
    real_key = result.real_control_result.state.rng_key
    scheduler_key, _planning_action = jr.split(real_key)
    rollout_root, _anchor = jr.split(scheduler_key)
    expected_final = jr.split(jr.split(rollout_root, 4)[0], 4)[0]
    expected_control = result.real_control_result.state.replace(rng_key=expected_final)
    _assert_key_equal(result.state.control_state.rng_key, expected_final)
    chex.assert_trees_all_equal(result.state.control_state, expected_control)


@pytest.mark.parametrize("warmup", [0, 100])
@pytest.mark.parametrize("planning_steps", [1, 2])
def test_valid_planning_uses_the_exact_linear_key_chain(
    warmup: int,
    planning_steps: int,
) -> None:
    """Accepted and gate-rejected planning persist the same fresh RNG chain."""
    rollout_depth = 2
    cfg = _cfg(
        planning_steps=planning_steps,
        warmup=warmup,
        rollout_depth=rollout_depth,
    )
    agent, model, state = _init(cfg, seed=19)
    expected_real = agent.update(
        state.control_state,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.zeros((OBS_DIM,), dtype=jnp.float32),
    )
    result = step7_update(
        cfg,
        agent,
        model,
        state,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.zeros((OBS_DIM,), dtype=jnp.float32),
    )

    assert bool(jnp.all(result.planning_accepted)) is (warmup == 0)
    chex.assert_trees_all_equal(result.real_control_result, expected_real)
    expected = _advance_planning_key(
        result.real_control_result.state.rng_key,
        planning_steps=planning_steps,
        rollout_depth=rollout_depth,
    )
    _assert_key_equal(result.state.control_state.rng_key, expected)
    if warmup > 0:
        chex.assert_trees_all_equal(
            result.state.control_state,
            result.real_control_result.state.replace(rng_key=expected),
        )


@pytest.mark.parametrize("warmup", [0, 100])
def test_step7_eager_jit_and_scan_reproduce_rng_trajectories(
    warmup: int,
) -> None:
    cfg = _cfg(planning_steps=2, warmup=warmup, rollout_depth=2)
    agent, model, state = _init(cfg, seed=23)
    reward = jnp.array(0.125, dtype=jnp.float32)
    next_observation = jnp.full((OBS_DIM,), 0.25, dtype=jnp.float32)
    with jax.disable_jit():
        eager = step7_update(cfg, agent, model, state, reward, next_observation)
    compiled = jax.jit(step7_update, static_argnums=(0, 1, 2))(
        cfg,
        agent,
        model,
        state,
        reward,
        next_observation,
    )
    _assert_key_equal(
        eager.state.control_state.rng_key,
        compiled.state.control_state.rng_key,
    )
    chex.assert_trees_all_equal(eager.planning_actions, compiled.planning_actions)
    chex.assert_trees_all_equal(eager.planning_accepted, compiled.planning_accepted)

    rewards = jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float32)
    observations = jnp.asarray(
        [
            [0.1, 0.0, -0.1, 0.2],
            [0.2, -0.1, 0.3, 0.0],
            [-0.2, 0.4, 0.1, -0.3],
        ],
        dtype=jnp.float32,
    )
    first = run_step7_scan(cfg, agent, model, state, rewards, observations)
    second = run_step7_scan(cfg, agent, model, state, rewards, observations)
    first_without_clock = first.replace(  # type: ignore[attr-defined]
        state=first.state.replace(  # type: ignore[attr-defined]
            world_model_state=first.state.world_model_state.replace(  # type: ignore[attr-defined]
                learner_state=first.state.world_model_state.learner_state.replace(  # type: ignore[attr-defined]
                    uptime_s=0.0,
                ),
            ),
        ),
    )
    second_without_clock = second.replace(  # type: ignore[attr-defined]
        state=second.state.replace(  # type: ignore[attr-defined]
            world_model_state=second.state.world_model_state.replace(  # type: ignore[attr-defined]
                learner_state=second.state.world_model_state.learner_state.replace(  # type: ignore[attr-defined]
                    uptime_s=0.0,
                ),
            ),
        ),
    )
    chex.assert_trees_all_equal(first_without_clock, second_without_clock)
