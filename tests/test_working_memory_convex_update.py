"""Finite convex trace updates must survive overflowing intermediate differences."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.state_builder import (
    FixedTraceStateBuilder,
    FixedTraceStateBuilderConfig,
)
from alberta_framework.core.working_memory import (
    WorkingMemoryConfig,
    WorkingMemoryFeaturizer,
    transform_working_memory_arrays,
)


@pytest.mark.parametrize("bank", ["observation", "action", "reward"])
@pytest.mark.parametrize("gate", [0.25, 1.0])
def test_opposite_finite_inputs_commit_in_every_trace_bank(bank: str, gate: float) -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=1,
            action_dim=1,
            reward_dim=1,
            observation_decay_rates=(0.5,),
            action_decay_rates=(0.5,),
            reward_decay_rates=(0.5,),
        )
    )
    magnitude = np.float32(3e38)
    zero = jnp.zeros(1, dtype=jnp.float32)
    positive = {name: zero for name in ("observation", "action", "reward")}
    positive[bank] = jnp.asarray([magnitude])
    first = memory.step(memory.init(), **positive)
    assert bool(first.update_applied)

    negative = {**positive, bank: -positive[bank]}
    second = memory.step(first.state, **negative, external_gate=gate)

    assert bool(second.update_applied)
    assert int(second.state.step_count) == 2
    # Evaluate the mathematical convex combination in binary64, where the
    # difference between the two binary32 endpoints is representable.
    old = np.asarray(getattr(first.state, f"{bank}_traces"), dtype=np.float64)
    expected = ((1.0 - 0.5 * gate) * old - 0.5 * gate * float(magnitude)).astype(np.float32)
    np.testing.assert_allclose(getattr(second.state, f"{bank}_traces"), expected, rtol=2e-7)
    chex.assert_tree_all_finite(second)
    for other in positive.keys() - {bank}:
        np.testing.assert_array_equal(getattr(second.state, f"{other}_traces"), [[0.0]])


def test_scan_retains_each_event_after_a_large_sign_change() -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=1,
            reward_dim=0,
            observation_decay_rates=(0.5,),
        )
    )
    observations = jnp.asarray([[3e38], [-3e38], [0.0]], dtype=jnp.float32)
    result = jax.jit(
        lambda: transform_working_memory_arrays(
            memory, observations, jnp.empty((3, 0)), jnp.empty((3, 0))
        )
    )()

    np.testing.assert_array_equal(result.updates_applied, [True, True, True])
    assert int(result.state.step_count) == 3
    expected = np.float32(-float(np.float32(3e38)) / 8.0)
    np.testing.assert_allclose(result.state.observation_traces, [[expected]], rtol=2e-7)
    # The third row consumes the second event's trace, rather than stale memory.
    np.testing.assert_allclose(result.features[2, -1], 2.0 * expected, rtol=2e-7)
    chex.assert_tree_all_finite(result)


def test_fixed_trace_builder_keeps_a_finite_representation_after_sign_change() -> None:
    builder = FixedTraceStateBuilder(
        FixedTraceStateBuilderConfig(
            observation_dim=1,
            n_actions=2,
            observation_decay_rates=(0.5,),
        )
    )
    state = builder.init(jax.random.key(0))
    for value in (3e38, -3e38):
        state, representation = builder.update(
            state, jnp.asarray([value], dtype=jnp.float32), 0, 0.0, 1.0
        )
        chex.assert_tree_all_finite(representation)

    assert int(state.step_count) == 2
    expected = np.float32(-float(np.float32(3e38)) / 4.0)
    np.testing.assert_allclose(state.observation_traces, [[expected]], rtol=2e-7)


def test_closed_gate_preserves_the_existing_trace_across_sign_change() -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(observation_dim=1, observation_decay_rates=(0.5,))
    )
    state = memory.update(
        memory.init(), jnp.asarray([3e38]), memory.zero_action(), memory.zero_reward()
    )
    result = memory.step(
        state, jnp.asarray([-3e38]), memory.zero_action(), memory.zero_reward(), external_gate=0.0
    )
    assert bool(result.update_applied)
    assert int(result.state.step_count) == 2
    np.testing.assert_array_equal(result.state.observation_traces, state.observation_traces)


def test_equal_and_opposite_limits_cancel_without_rejecting() -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(observation_dim=1, observation_decay_rates=(0.5,))
    )
    maximum = jnp.finfo(jnp.float32).max
    state = memory.init().replace(observation_traces=jnp.asarray([[-maximum]]))
    result = memory.step(
        state, jnp.asarray([maximum]), memory.zero_action(), memory.zero_reward()
    )
    assert bool(result.update_applied)
    np.testing.assert_array_equal(result.state.observation_traces, [[0.0]])
    chex.assert_tree_all_finite(result)


def test_finite_delta_updates_keep_their_equal_input_fixed_point() -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(observation_dim=1, observation_decay_rates=(0.9,))
    )
    value = jnp.asarray([3e38], dtype=jnp.float32)
    state = memory.init().replace(observation_traces=value[None, :])
    result = memory.step(state, value, memory.zero_action(), memory.zero_reward())
    assert bool(result.update_applied)
    np.testing.assert_array_equal(result.state.observation_traces, state.observation_traces)
