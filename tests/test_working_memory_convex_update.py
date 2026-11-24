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


@pytest.mark.parametrize("bank", ["observation", "action", "reward"])
@pytest.mark.parametrize("external_gate", [0.25, 1.0])
def test_gated_opposite_finite_inputs_commit_in_every_trace_bank(
    bank: str, external_gate: float
) -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=1,
            action_dim=1,
            reward_dim=1,
            observation_decay_rates=(0.5,),
            action_decay_rates=(0.5,),
            reward_decay_rates=(0.5,),
            gated_update=True,
            gate_threshold=0.0,
            gate_temperature=1.0,
        )
    )
    magnitude = np.float32(3e38)
    zero = jnp.zeros(1, dtype=jnp.float32)
    positive = {name: zero for name in ("observation", "action", "reward")}
    positive[bank] = jnp.asarray([magnitude])
    first = memory.step(memory.init(), **positive)
    assert bool(first.update_applied)

    negative = {**positive, bank: -positive[bank]}
    second = memory.step(first.state, **negative, external_gate=external_gate)

    assert bool(second.update_applied)
    assert int(second.state.step_count) == 2
    bank_index = {"observation": 0, "action": 1, "reward": 2}[bank]
    np.testing.assert_allclose(second.state.last_gate[bank_index], external_gate, rtol=1e-6)
    old = np.asarray(getattr(first.state, f"{bank}_traces"), dtype=np.float64)
    expected = (
        (1.0 - 0.5 * external_gate) * old - 0.5 * external_gate * float(magnitude)
    ).astype(np.float32)
    np.testing.assert_allclose(getattr(second.state, f"{bank}_traces"), expected, rtol=2e-7)
    chex.assert_tree_all_finite(second)


def test_gated_surprise_matches_exact_sigmoid_across_overflowing_difference() -> None:
    limit = np.float32(np.finfo(np.float32).max)

    # Sparse overflowing coordinate (N=9): exact RMS = 2*limit / 3 < threshold = 3e38,
    # so the true gate is strictly below 0.5 rather than saturated to 1.0.
    sparse_memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=9,
            action_dim=0,
            reward_dim=0,
            observation_decay_rates=(0.5,),
            action_decay_rates=(),
            reward_decay_rates=(),
            gated_update=True,
            gate_threshold=3.0e38,
            gate_temperature=1.0e38,
        )
    )
    sparse_trace = np.zeros((1, 9), dtype=np.float32)
    sparse_trace[0, 0] = -limit
    sparse_obs = np.zeros((9,), dtype=np.float32)
    sparse_obs[0] = limit
    sparse_state = sparse_memory.init().replace(observation_traces=jnp.asarray(sparse_trace))
    sparse_result = sparse_memory.update_checked(
        sparse_state,
        jnp.asarray(sparse_obs),
        sparse_memory.zero_action(),
        sparse_memory.zero_reward(),
    )
    assert bool(sparse_result.update_applied)
    exact_sparse_rms = (2.0 * float(limit)) / 3.0
    exact_sparse_gate = 1.0 / (1.0 + np.exp(-((exact_sparse_rms - 3.0e38) / 1.0e38)))
    assert exact_sparse_gate < 0.5
    np.testing.assert_allclose(sparse_result.state.last_gate[0], exact_sparse_gate, rtol=1e-5)
    chex.assert_tree_all_finite(sparse_result)

    # Above-threshold overflowing difference: exact RMS = 3.6e38 > threshold = 3.4e38.
    above_memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=1,
            action_dim=0,
            reward_dim=0,
            observation_decay_rates=(0.5,),
            action_decay_rates=(),
            reward_decay_rates=(),
            gated_update=True,
            gate_threshold=3.4e38,
            gate_temperature=1.0e38,
        )
    )
    mag = np.float32(1.8e38)
    above_state = above_memory.init().replace(
        observation_traces=jnp.asarray([[-mag]], dtype=jnp.float32)
    )
    above_result = jax.jit(above_memory.update_checked)(
        above_state,
        jnp.asarray([mag], dtype=jnp.float32),
        above_memory.zero_action(),
        above_memory.zero_reward(),
    )
    assert bool(above_result.update_applied)
    exact_above_gate = 1.0 / (
        1.0
        + np.exp(
            -(
                (2.0 * float(mag) - float(np.float32(3.4e38)))
                / float(np.float32(1.0e38))
            )
        )
    )
    assert exact_above_gate > 0.5
    np.testing.assert_allclose(above_result.state.last_gate[0], exact_above_gate, rtol=1e-5)
    chex.assert_tree_all_finite(above_result)

    # Non-overflowing difference with huge temperature (> 1 / float32_min_normal)
    # must not flush margin / temperature to zero via XLA reciprocal FTZ.
    huge_temp_memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=1,
            action_dim=0,
            reward_dim=0,
            observation_decay_rates=(0.5,),
            action_decay_rates=(),
            reward_decay_rates=(),
            gated_update=True,
            gate_threshold=0.0,
            gate_temperature=1.0e38,
        )
    )
    huge_temp_result = jax.jit(huge_temp_memory.update_checked)(
        huge_temp_memory.init(),
        jnp.asarray([2.0e38], dtype=jnp.float32),
        huge_temp_memory.zero_action(),
        huge_temp_memory.zero_reward(),
    )
    assert bool(huge_temp_result.update_applied)
    exact_huge_temp_gate = 1.0 / (1.0 + np.exp(-2.0))
    np.testing.assert_allclose(
        huge_temp_result.state.last_gate[0], exact_huge_temp_gate, rtol=1e-5
    )
    chex.assert_tree_all_finite(huge_temp_result)


def test_gated_scan_retains_each_event_after_large_sign_change() -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=1,
            reward_dim=0,
            observation_decay_rates=(0.5,),
            gated_update=True,
            gate_threshold=0.0,
            gate_temperature=1.0,
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
    chex.assert_tree_all_finite(result)


@pytest.mark.parametrize("poison", [np.float32(np.inf), np.float32(-np.inf), np.float32(np.nan)])
def test_gated_update_rejects_poisoned_trace_even_with_zero_decay(poison: np.float32) -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=1,
            action_dim=0,
            reward_dim=0,
            observation_decay_rates=(0.0,),
            action_decay_rates=(),
            reward_decay_rates=(),
            gated_update=True,
            gate_threshold=0.0,
            gate_temperature=1.0,
        )
    )
    poisoned = memory.init().replace(observation_traces=jnp.asarray([[poison]], dtype=jnp.float32))
    result = memory.update_checked(
        poisoned,
        jnp.asarray([1.0], dtype=jnp.float32),
        memory.zero_action(),
        memory.zero_reward(),
    )
    assert not bool(result.update_applied)
    assert int(result.state.step_count) == 0
