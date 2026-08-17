"""Tests for the recurring two-agent continual-control world."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams import (
    AVOID_CONTEXT,
    AVOID_CONTEXT_INDEX,
    BASE_OBSERVATION_DIM,
    MEET_CONTEXT,
    MEET_CONTEXT_INDEX,
    RecurringTwoAgentState,
    RecurringTwoAgentTransition,
    RecurringTwoAgentWorld,
)
from alberta_framework.streams.recurring_multiagent import RecurringTwoAgentResourceBudget


def _with_step_count(
    state: RecurringTwoAgentState,
    step_count: int,
) -> RecurringTwoAgentState:
    return RecurringTwoAgentState(
        key=state.key,
        positions=state.positions,
        velocities=state.velocities,
        nuisance=state.nuisance,
        step_count=jnp.array(step_count, dtype=jnp.int32),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_length": 0}, "context_length"),
        ({"nuisance_dim": -1}, "nuisance_dim"),
        ({"nuisance_scale": -0.1}, "nuisance_scale"),
        ({"world_limit": 0.0}, "world_limit"),
        ({"damping": 1.0}, "damping"),
        ({"acceleration": 0.0}, "acceleration"),
        ({"time_delta": 0.0}, "time_delta"),
        ({"max_speed": 0.0}, "max_speed"),
        ({"initial_positions": (-2.0, 0.0)}, "initial_positions"),
        ({"initial_positions": (True, False)}, "initial_positions"),
        ({"initial_positions": (False, True)}, "initial_positions"),
        ({"initial_positions": (1.0, False)}, "initial_positions"),
    ],
)
def test_invalid_configuration_is_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RecurringTwoAgentWorld(**kwargs)  # type: ignore[arg-type]


def test_initial_positions_reject_boolean_identities() -> None:
    legal = RecurringTwoAgentWorld(initial_positions=(-0.5, 0.5))
    assert legal._initial_positions_tuple == (-0.5, 0.5)
    for positions in ((True, False), (np.True_, 0.0), (1.0, np.bool_(False))):
        with pytest.raises(ValueError, match="initial_positions"):
            RecurringTwoAgentWorld(initial_positions=positions)


def test_initial_positions_validate_container_and_numeric_hooks_before_allocation() -> None:
    class HostileTuple(tuple):
        def __len__(self) -> int:
            raise AssertionError("hostile length hook executed")

    class ExplodingFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            raise RuntimeError("hostile ratio hook")

        def __repr__(self) -> str:
            raise AssertionError("hostile repr hook executed")

    with pytest.raises(ValueError, match="exact 2-tuple"):
        RecurringTwoAgentWorld(initial_positions=HostileTuple((-0.5, 0.5)))
    with pytest.raises(ValueError, match=r"initial_positions\[0\]"):
        RecurringTwoAgentWorld(initial_positions=(ExplodingFloat(0.25), 0.5))


def test_initial_positions_canonicalize_supported_numpy_scalars() -> None:
    world = RecurringTwoAgentWorld(
        initial_positions=(np.float64(-0.25), np.int16(0)),
    )
    assert world._initial_positions_tuple == (-0.25, 0.0)
    assert all(type(value) is float for value in world._initial_positions_tuple)


def test_visible_context_sequence_recurs_a_b_a() -> None:
    world = RecurringTwoAgentWorld(context_length=2, nuisance_dim=0)
    state = world.init(jr.key(0))
    action = jnp.zeros((2,), dtype=jnp.float32)

    contexts: list[int] = []
    next_contexts: list[int] = []
    switched: list[bool] = []
    segments: list[int] = []
    cycles: list[int] = []
    visible_contexts: list[tuple[float, float]] = []

    for _ in range(5):
        observation = world.observe(state)
        visible_contexts.append(
            (
                float(observation[0, MEET_CONTEXT_INDEX]),
                float(observation[0, AVOID_CONTEXT_INDEX]),
            )
        )
        transition, state = world.step(state, action)
        contexts.append(int(transition.oracle.context_id))
        next_contexts.append(int(transition.oracle.next_context_id))
        switched.append(bool(transition.oracle.context_switched))
        segments.append(int(transition.oracle.segment_index))
        cycles.append(int(transition.oracle.cycle_index))

    assert contexts == [MEET_CONTEXT, MEET_CONTEXT, AVOID_CONTEXT, AVOID_CONTEXT, MEET_CONTEXT]
    assert next_contexts == [
        MEET_CONTEXT,
        AVOID_CONTEXT,
        AVOID_CONTEXT,
        MEET_CONTEXT,
        MEET_CONTEXT,
    ]
    assert switched == [False, True, False, True, False]
    assert segments == [0, 0, 1, 1, 2]
    assert cycles == [0, 0, 0, 0, 1]
    assert visible_contexts == [
        (1.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (1.0, 0.0),
    ]


def test_context_switch_does_not_reset_physics_or_terminate() -> None:
    world = RecurringTwoAgentWorld(context_length=2, nuisance_dim=0)
    initial_state = world.init(jr.key(1))
    action = jnp.array([0.6, 0.2], dtype=jnp.float32)

    _, state_before_switch = world.step(initial_state, action)
    transition, state_after_switch = world.step(state_before_switch, action)

    assert bool(transition.oracle.context_switched)
    assert int(transition.oracle.context_id) == MEET_CONTEXT
    assert int(transition.oracle.next_context_id) == AVOID_CONTEXT
    assert int(state_after_switch.step_count) == 2
    chex.assert_trees_all_close(
        transition.oracle.positions_before,
        state_before_switch.positions,
    )
    chex.assert_trees_all_close(
        transition.oracle.velocities_before,
        state_before_switch.velocities,
    )
    chex.assert_trees_all_close(
        transition.oracle.positions_after,
        state_after_switch.positions,
    )
    chex.assert_trees_all_close(
        transition.oracle.velocities_after,
        state_after_switch.velocities,
    )
    chex.assert_trees_all_close(
        transition.next_observation[:, 0],
        state_after_switch.positions / world.world_limit,
    )
    assert not jnp.array_equal(state_after_switch.positions, initial_state.positions)
    assert jnp.any(state_after_switch.velocities != 0.0)
    assert not bool(transition.terminated)
    assert float(transition.discount) == pytest.approx(1.0)


def test_joint_actions_have_opposite_predictable_effects_and_are_clipped() -> None:
    world = RecurringTwoAgentWorld(
        context_length=4,
        nuisance_dim=0,
        damping=0.0,
        acceleration=0.2,
        max_speed=0.5,
    )
    state = world.init(jr.key(2))

    toward, toward_state = world.step(
        state,
        jnp.array([1.0, -1.0], dtype=jnp.float32),
    )
    away, away_state = world.step(
        state,
        jnp.array([-1.0, 1.0], dtype=jnp.float32),
    )
    clipped, _ = world.step(
        state,
        jnp.array([5.0, -3.0], dtype=jnp.float32),
    )

    assert float(toward.oracle.distance_after) < float(toward.oracle.distance_before)
    assert float(away.oracle.distance_after) > float(away.oracle.distance_before)
    chex.assert_trees_all_close(toward.action, jnp.array([1.0, -1.0]))
    chex.assert_trees_all_close(clipped.action, jnp.array([1.0, -1.0]))
    chex.assert_trees_all_close(
        clipped.oracle.requested_actions,
        jnp.array([5.0, -3.0]),
    )
    chex.assert_trees_all_close(
        clipped.oracle.applied_actions,
        jnp.array([1.0, -1.0]),
    )
    assert toward_state.positions[0] > state.positions[0]
    assert toward_state.positions[1] < state.positions[1]
    assert away_state.positions[0] < state.positions[0]
    assert away_state.positions[1] > state.positions[1]
    # The frozen input state was not mutated by either counterfactual action.
    chex.assert_trees_all_close(state.positions, jnp.array([-0.5, 0.5]))
    chex.assert_trees_all_close(state.velocities, jnp.zeros((2,)))


def test_meet_and_avoid_rewards_use_post_action_distance() -> None:
    world = RecurringTwoAgentWorld(
        context_length=3,
        nuisance_dim=0,
        damping=0.0,
    )
    initial = world.init(jr.key(3))
    separated = RecurringTwoAgentState(
        key=initial.key,
        positions=jnp.array([-0.75, 0.75], dtype=jnp.float32),
        velocities=jnp.zeros((2,), dtype=jnp.float32),
        nuisance=initial.nuisance,
        step_count=jnp.array(0, dtype=jnp.int32),
    )
    avoid_state = _with_step_count(separated, world.context_length)
    zero_action = jnp.zeros((2,), dtype=jnp.float32)

    meet_transition, _ = world.step(separated, zero_action)
    avoid_transition, _ = world.step(avoid_state, zero_action)

    assert int(meet_transition.oracle.context_id) == MEET_CONTEXT
    assert int(avoid_transition.oracle.context_id) == AVOID_CONTEXT
    chex.assert_trees_all_close(meet_transition.reward, jnp.full((2,), 0.25))
    chex.assert_trees_all_close(avoid_transition.reward, jnp.full((2,), 0.75))
    assert float(meet_transition.oracle.meet_reward) == pytest.approx(0.25)
    assert float(meet_transition.oracle.avoid_reward) == pytest.approx(0.75)
    assert float(
        meet_transition.oracle.meet_reward + meet_transition.oracle.avoid_reward
    ) == pytest.approx(1.0)


def test_boundary_transition_reward_matches_context_visible_before_action() -> None:
    world = RecurringTwoAgentWorld(
        context_length=3,
        nuisance_dim=0,
        damping=0.0,
    )
    state = _with_step_count(world.init(jr.key(4)), 2)
    transition, _ = world.step(state, jnp.zeros((2,), dtype=jnp.float32))

    assert bool(transition.oracle.context_switched)
    assert int(transition.oracle.context_id) == MEET_CONTEXT
    assert int(transition.oracle.next_context_id) == AVOID_CONTEXT
    assert transition.observation[0, MEET_CONTEXT_INDEX] == 1.0
    assert transition.next_observation[0, AVOID_CONTEXT_INDEX] == 1.0
    chex.assert_trees_all_close(
        transition.reward,
        jnp.full((2,), transition.oracle.meet_reward),
    )


def test_default_and_custom_partner_policies_change_with_visible_context() -> None:
    default_world = RecurringTwoAgentWorld(context_length=2, nuisance_dim=0)
    meet_state = default_world.init(jr.key(5))
    avoid_state = _with_step_count(meet_state, default_world.context_length)

    meet_transition, _ = default_world.step_with_partner(
        meet_state,
        jnp.array(0.0),
    )
    avoid_transition, _ = default_world.step_with_partner(
        avoid_state,
        jnp.array(0.0),
    )

    # Agent 1 starts to the right of agent 0: left meets, right avoids.
    assert float(meet_transition.action[1]) == pytest.approx(-1.0)
    assert float(avoid_transition.action[1]) == pytest.approx(1.0)

    def custom_policy(observation: jax.Array, key: jax.Array) -> jax.Array:
        del key
        return jnp.where(
            observation[MEET_CONTEXT_INDEX] > 0.5,
            jnp.array(0.25, dtype=jnp.float32),
            jnp.array(-0.75, dtype=jnp.float32),
        )

    custom_world = RecurringTwoAgentWorld(
        context_length=2,
        nuisance_dim=1,
        partner_policy=custom_policy,
    )
    jit_step = jax.jit(custom_world.step_with_partner)
    custom_meet = custom_world.init(jr.key(6))
    custom_avoid = _with_step_count(custom_meet, custom_world.context_length)
    meet_transition, _ = jit_step(custom_meet, jnp.array([0.0]))
    avoid_transition, _ = jit_step(custom_avoid, jnp.array([0.0]))

    assert float(meet_transition.action[1]) == pytest.approx(0.25)
    assert float(avoid_transition.action[1]) == pytest.approx(-0.75)


def test_nuisance_is_seeded_ephemeral_and_causally_irrelevant() -> None:
    world = RecurringTwoAgentWorld(context_length=3, nuisance_dim=4)
    state_a = world.init(jr.key(7))
    state_a_repeat = world.init(jr.key(7))
    state_b = world.init(jr.key(8))

    chex.assert_trees_all_equal(state_a, state_a_repeat)
    assert not jnp.array_equal(state_a.nuisance, state_b.nuisance)

    transition, next_state = world.step(
        state_a,
        jnp.array([0.1, -0.2], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        transition.observation[:, BASE_OBSERVATION_DIM:],
        state_a.nuisance,
    )
    chex.assert_trees_all_close(
        transition.next_observation[:, BASE_OBSERVATION_DIM:],
        next_state.nuisance,
    )
    assert not jnp.array_equal(state_a.nuisance, next_state.nuisance)

    nuisance_counterfactual = RecurringTwoAgentState(
        key=state_a.key,
        positions=state_a.positions,
        velocities=state_a.velocities,
        nuisance=state_a.nuisance + 1000.0,
        step_count=state_a.step_count,
    )
    counterfactual_transition, counterfactual_state = world.step(
        nuisance_counterfactual,
        jnp.array([0.1, -0.2], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        transition.reward,
        counterfactual_transition.reward,
    )
    chex.assert_trees_all_close(
        next_state.positions,
        counterfactual_state.positions,
    )
    chex.assert_trees_all_close(
        next_state.velocities,
        counterfactual_state.velocities,
    )


def test_recurrence_oracle_is_not_encoded_in_agent_observation() -> None:
    world = RecurringTwoAgentWorld(context_length=3, nuisance_dim=2)
    first_cycle = world.init(jr.key(9))
    recurrent_cycle = _with_step_count(first_cycle, 2 * world.context_length)

    # Same physical/nuisance state and same recurring context yield exactly the
    # same agent input; absolute time and cycle remain available only in oracle.
    chex.assert_trees_all_equal(
        world.observe(first_cycle),
        world.observe(recurrent_cycle),
    )
    first_transition, _ = world.step(
        first_cycle,
        jnp.zeros((2,), dtype=jnp.float32),
    )
    recurrent_transition, _ = world.step(
        recurrent_cycle,
        jnp.zeros((2,), dtype=jnp.float32),
    )
    assert first_transition.observation.shape == (
        2,
        BASE_OBSERVATION_DIM + world.nuisance_dim,
    )
    assert int(first_transition.oracle.step_count) == 0
    assert int(first_transition.oracle.cycle_index) == 0
    assert int(recurrent_transition.oracle.step_count) == 6
    assert int(recurrent_transition.oracle.cycle_index) == 1


def test_seeded_transition_shapes_finiteness_and_reproducibility() -> None:
    world = RecurringTwoAgentWorld(context_length=5, nuisance_dim=3)
    state_a = world.init(jr.key(10))
    state_b = world.init(jr.key(10))
    action = jnp.array([0.4, -0.6], dtype=jnp.float32)

    transition_a, next_a = world.step(state_a, action)
    transition_b, next_b = world.step(state_b, action)

    assert isinstance(transition_a, RecurringTwoAgentTransition)
    chex.assert_trees_all_equal(transition_a, transition_b)
    chex.assert_trees_all_equal(next_a, next_b)
    chex.assert_shape(state_a.positions, (2,))
    chex.assert_shape(state_a.velocities, (2,))
    chex.assert_shape(state_a.nuisance, (2, 3))
    chex.assert_shape(transition_a.observation, (2, 9))
    chex.assert_shape(transition_a.action, (2,))
    chex.assert_shape(transition_a.reward, (2,))
    chex.assert_shape(transition_a.next_observation, (2, 9))
    chex.assert_shape(transition_a.oracle.positions_before, (2,))
    chex.assert_shape(transition_a.oracle.positions_after, (2,))
    chex.assert_shape(transition_a.oracle.hit_boundary, (2,))
    chex.assert_tree_all_finite(
        (
            transition_a.observation,
            transition_a.action,
            transition_a.reward,
            transition_a.next_observation,
            transition_a.oracle.distance_before,
            transition_a.oracle.distance_after,
            next_a.positions,
            next_a.velocities,
            next_a.nuisance,
        )
    )


def test_state_is_an_immutable_chex_pytree() -> None:
    world = RecurringTwoAgentWorld(nuisance_dim=1)
    state = world.init(jr.key(12))

    with pytest.raises(FrozenInstanceError):
        state.step_count = jnp.array(3, dtype=jnp.int32)

    leaves, tree = jax.tree.flatten(state)
    rebuilt = jax.tree.unflatten(tree, leaves)
    chex.assert_trees_all_equal(rebuilt, state)


def test_public_step_keeps_rejected_nonfinite_action_unaggregatable() -> None:
    world = RecurringTwoAgentWorld(nuisance_dim=0)
    state = world.init(jr.key(13)).replace(
        velocities=jnp.array([0.2, -0.1], dtype=jnp.float32),
    )

    transition, next_state = world.step(
        state,
        jnp.array([jnp.nan, 0.0], dtype=jnp.float32),
    )

    assert bool(transition.terminated)
    assert float(transition.discount) == 0.0
    for payload in (
        transition.observation,
        transition.action,
        transition.reward,
        transition.next_observation,
    ):
        assert bool(jnp.all(jnp.isnan(payload)))
    for leaf in jax.tree.leaves(transition.oracle):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert bool(jnp.all(jnp.isnan(leaf)))
    chex.assert_trees_all_equal(next_state, state)


def test_jitted_scan_runs_continually_and_remains_bounded() -> None:
    world = RecurringTwoAgentWorld(
        context_length=3,
        nuisance_dim=2,
        damping=0.9,
        acceleration=0.2,
        max_speed=0.3,
    )
    initial_state = world.init(jr.key(11))
    num_steps = 64
    actions = jnp.broadcast_to(
        jnp.array([-1.0, 1.0], dtype=jnp.float32),
        (num_steps, 2),
    )

    @jax.jit
    def rollout(
        state: RecurringTwoAgentState,
        joint_actions: jax.Array,
    ) -> tuple[RecurringTwoAgentState, tuple[jax.Array, ...]]:
        def scan_step(
            carry: RecurringTwoAgentState,
            action: jax.Array,
        ) -> tuple[RecurringTwoAgentState, tuple[jax.Array, ...]]:
            transition, next_state = world.step(carry, action)
            outputs = (
                transition.reward,
                transition.terminated,
                transition.discount,
                transition.oracle.context_id,
                transition.oracle.context_switched,
                transition.oracle.positions_after,
                transition.oracle.velocities_after,
                transition.next_observation,
            )
            return next_state, outputs

        return jax.lax.scan(scan_step, state, joint_actions)

    final_state, outputs = rollout(initial_state, actions)
    (
        rewards,
        terminated,
        discounts,
        contexts,
        switched,
        positions,
        velocities,
        observations,
    ) = outputs

    expected_contexts = (jnp.arange(num_steps) // world.context_length) % 2
    expected_next_contexts = ((jnp.arange(num_steps) + 1) // world.context_length) % 2
    chex.assert_trees_all_equal(contexts, expected_contexts)
    chex.assert_trees_all_equal(switched, contexts != expected_next_contexts)
    assert int(final_state.step_count) == num_steps
    assert not bool(jnp.any(terminated))
    chex.assert_trees_all_close(discounts, jnp.ones((num_steps,)))
    assert jnp.all(jnp.abs(positions) <= world.world_limit)
    assert jnp.all(jnp.abs(velocities) <= world.max_speed)
    chex.assert_shape(rewards, (num_steps, 2))
    chex.assert_shape(observations, (num_steps, 2, world.observation_dim))
    chex.assert_tree_all_finite((rewards, discounts, positions, velocities, observations))


def test_recurring_two_agent_world_initial_positions_reject_booleans() -> None:
    # Boolean initial positions
    with pytest.raises(ValueError, match="initial_positions"):
        RecurringTwoAgentWorld(initial_positions=(True, False))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="initial_positions"):
        RecurringTwoAgentWorld(initial_positions=(False, True))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="initial_positions"):
        RecurringTwoAgentWorld(initial_positions=(np.True_, 0.0))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="initial_positions"):
        RecurringTwoAgentWorld(initial_positions=(1.0, False))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="initial_positions"):
        RecurringTwoAgentWorld(initial_positions=(float("nan"), 0.5))

    # Valid initial positions
    world = RecurringTwoAgentWorld(initial_positions=(-0.5, 0.5))
    assert world.to_config()["initial_positions"] == [-0.5, 0.5]


def test_recurring_resource_budget_rejects_leftover_identities() -> None:
    """Public resource-budget records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="state_nbytes"):
        RecurringTwoAgentResourceBudget(
            state_nbytes=True,
            exact_clock_nbytes=12,
            exact_clock_delta_nbytes=8,
        )
    with pytest.raises(ValueError, match="trainable_scalars"):
        RecurringTwoAgentResourceBudget(
            state_nbytes=1,
            exact_clock_nbytes=12,
            exact_clock_delta_nbytes=8,
            trainable_scalars=True,
        )
    with pytest.raises(ValueError, match="replay_capacity"):
        RecurringTwoAgentResourceBudget(
            state_nbytes=1,
            exact_clock_nbytes=12,
            exact_clock_delta_nbytes=8,
            replay_capacity=True,
        )
    with pytest.raises(ValueError, match="state_nbytes"):
        RecurringTwoAgentResourceBudget(
            state_nbytes=float("nan"),
            exact_clock_nbytes=12,
            exact_clock_delta_nbytes=8,
        )

    legal = RecurringTwoAgentResourceBudget(
        state_nbytes=1,
        exact_clock_nbytes=12,
        exact_clock_delta_nbytes=8,
    )
    dumped = json.dumps(legal.to_dict(), allow_nan=False)
    assert '"state_nbytes": 1' in dumped
    assert '"exact_clock_nbytes": 12' in dumped
    assert '"trainable_scalars": 0' in dumped
    assert '"replay_capacity": 0' in dumped
    assert '"state_nbytes": true' not in dumped
    assert '"trainable_scalars": true' not in dumped
    assert '"replay_capacity": true' not in dumped
