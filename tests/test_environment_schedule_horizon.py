# mypy: disable-error-code="arg-type,attr-defined,call-arg,index,no-untyped-def,untyped-decorator"
"""Exact finite-horizon contracts for recurring environment schedules."""

from __future__ import annotations

import dataclasses

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.streams.gauntlet import (
    LIFETIME_GAUNTLET_CLOCK_DELTA_NBYTES,
    LIFETIME_GAUNTLET_CONFIG_SCHEMA,
    LIFETIME_GAUNTLET_STATE_SCHEMA,
    GauntletConfig,
    LifetimeGauntletStream,
    ema_smooth,
    load_lifetime_gauntlet_checkpoint,
    migrate_legacy_lifetime_gauntlet_state,
    save_lifetime_gauntlet_checkpoint,
)
from alberta_framework.streams.recurring_multiagent import (
    RECURRING_TWO_AGENT_CLOCK_DELTA_NBYTES,
    RECURRING_TWO_AGENT_CONFIG_SCHEMA,
    RECURRING_TWO_AGENT_STATE_SCHEMA,
    RecurringTwoAgentWorld,
    load_recurring_two_agent_checkpoint,
    migrate_legacy_recurring_two_agent_state,
    save_recurring_two_agent_checkpoint,
)

INT32_MAX = 2**31 - 1
pytestmark = pytest.mark.unit


def test_ema_smooth_preserves_leading_batch_axes() -> None:
    """EMA smoothing follows the last axis for higher-rank measurements."""
    values = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)

    actual = ema_smooth(values, halflife=1.0)

    expected = jnp.asarray(
        [
            [[0.0, 0.5, 1.25, 2.125], [4.0, 4.5, 5.25, 6.125], [8.0, 8.5, 9.25, 10.125]],
            [
                [12.0, 12.5, 13.25, 14.125],
                [16.0, 16.5, 17.25, 18.125],
                [20.0, 20.5, 21.25, 22.125],
            ],
        ],
        dtype=jnp.float32,
    )
    chex.assert_shape(actual, values.shape)
    chex.assert_trees_all_close(actual, expected)


def _words(value: int) -> jax.Array:
    high, low = divmod(value, 2**32)
    return jnp.asarray((high, low), dtype=jnp.uint32)


def _high_recurring(world: RecurringTwoAgentWorld, value: int):
    state = world.init(jr.key(1))
    return state.replace(
        step_count=jnp.asarray(INT32_MAX, dtype=jnp.int32),
        step_words=_words(value),
    )


def _high_lifetime(stream: LifetimeGauntletStream, value: int):
    state = stream.init(jr.key(2))
    return state.replace(
        step_count=jnp.asarray(INT32_MAX, dtype=jnp.int32),
        step_words=_words(value),
    )


@pytest.mark.parametrize("context_length", [3, 5, 17])
def test_recurring_schedule_division_is_exact_above_uint32(context_length: int) -> None:
    """Context, segment, and cycle identities must match Python integer math."""
    world = RecurringTwoAgentWorld(context_length=context_length, nuisance_dim=0)
    absolute = 3 * 2**32 + 123_457
    state = _high_recurring(world, absolute)

    result = jax.jit(world.step_result)(
        state,
        jnp.zeros((2,), dtype=jnp.float32),
    )

    segment = absolute // context_length
    cycle = segment // 2
    assert bool(result.update_applied)
    assert int(result.transition.oracle.context_id) == segment % 2
    chex.assert_trees_all_equal(result.transition.oracle.segment_words, _words(segment))
    chex.assert_trees_all_equal(result.transition.oracle.cycle_words, _words(cycle))
    chex.assert_trees_all_equal(result.state.step_words, _words(absolute + 1))
    assert int(result.state.step_count) == INT32_MAX


def test_recurring_scan_carries_low_word_and_switches_from_exact_phase() -> None:
    """A compiled scan must cross uint32 while retaining the true schedule."""
    world = RecurringTwoAgentWorld(context_length=3, nuisance_dim=0)
    start = 2**32 - 2
    state = _high_recurring(world, start)
    actions = jnp.zeros((6, 2), dtype=jnp.float32)

    def body(carry, action):
        result = world.step_result(carry, action)
        return result.state, (
            result.transition.oracle.context_id,
            result.update_applied,
        )

    final, (contexts, applied) = jax.jit(
        lambda initial: jax.lax.scan(body, initial, actions)
    )(state)
    expected = jnp.asarray(
        [((start + offset) // 3) % 2 for offset in range(6)], dtype=jnp.int32
    )
    chex.assert_trees_all_equal(contexts, expected)
    chex.assert_trees_all_equal(applied, jnp.ones((6,), dtype=jnp.bool_))
    chex.assert_trees_all_equal(final.step_words, _words(start + 6))


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf])
def test_recurring_nonfinite_action_is_bit_exact_rollback(bad: float) -> None:
    """Non-finite actions cannot consume RNG, physics, nuisance, or time."""
    world = RecurringTwoAgentWorld(context_length=3, nuisance_dim=2)
    state = world.init(jr.key(3))
    result = world.step_result(
        state,
        jnp.asarray((bad, 0.0), dtype=jnp.float32),
    )
    assert not bool(result.update_applied)
    assert not bool(result.input_valid)
    chex.assert_trees_all_equal(result.state, state)
    assert bool(result.transition.terminated)
    assert float(result.transition.discount) == 0.0


def test_recurring_terminal_and_corrupt_clocks_fail_closed() -> None:
    """Terminal exhaustion and unauthenticated telemetry are atomic no-ops."""
    world = RecurringTwoAgentWorld(context_length=7, nuisance_dim=0)
    terminal = _high_recurring(world, 2**64 - 1)
    terminal_result = world.step_result(
        terminal, jnp.zeros((2,), dtype=jnp.float32)
    )
    assert not bool(terminal_result.update_applied)
    assert not bool(terminal_result.lifetime_capacity_available)
    chex.assert_trees_all_equal(terminal_result.state, terminal)

    corrupt = terminal.replace(
        step_count=jnp.asarray(0, dtype=jnp.int32),
        step_words=jnp.asarray((1, 4), dtype=jnp.uint32),
    )
    corrupt_result = world.step_result(
        corrupt, jnp.zeros((2,), dtype=jnp.float32)
    )
    assert not bool(corrupt_result.state_valid)
    assert not bool(corrupt_result.update_applied)
    chex.assert_trees_all_equal(corrupt_result.state, corrupt)


def test_recurring_schema_migration_resources_and_checkpoint(tmp_path) -> None:
    """Persistence binds exact schemas and rejects saturated legacy history."""
    world = RecurringTwoAgentWorld(context_length=5, nuisance_dim=0)
    config = world.to_config()
    assert config["config_schema"] == RECURRING_TWO_AGENT_CONFIG_SCHEMA
    assert config["state_schema"] == RECURRING_TWO_AGENT_STATE_SCHEMA
    assert RecurringTwoAgentWorld.from_config(config).to_config() == config
    with pytest.raises(ValueError, match="fields"):
        RecurringTwoAgentWorld.from_config({**config, "extra": 1})
    assert (
        world.resource_budget.exact_clock_delta_nbytes
        == RECURRING_TWO_AGENT_CLOCK_DELTA_NBYTES
    )

    state = world.step_result(
        world.init(jr.key(4)), jnp.zeros((2,), dtype=jnp.float32)
    ).state
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state)
        if field.name != "step_words"
    }
    migrated = migrate_legacy_recurring_two_agent_state(legacy, world=world)
    chex.assert_trees_all_equal(migrated.step_words, jnp.asarray((0, 1), dtype=jnp.uint32))
    with pytest.raises(ValueError, match="saturated"):
        migrate_legacy_recurring_two_agent_state(
            {**legacy, "step_count": jnp.asarray(INT32_MAX, dtype=jnp.int32)},
            world=world,
        )

    path = tmp_path / "recurring"
    save_recurring_two_agent_checkpoint(world, state, path)
    restored_world, restored = load_recurring_two_agent_checkpoint(path)
    assert restored_world.to_config() == config
    chex.assert_trees_all_equal(restored, state)


@pytest.mark.parametrize("absolute", [2**32 + 7, 5 * 2**32 + 999_983])
def test_lifetime_schedule_is_exact_above_uint32(absolute: int) -> None:
    """Sub-segment, cycle, boundary, and scale cadence use exact time."""
    config = GauntletConfig(relevant_dim=2, irrelevant_dim=0, segment_length=7)
    stream = LifetimeGauntletStream(config, scale_cycle_period=3)
    state = _high_lifetime(stream, absolute)
    result = jax.jit(stream.step_result)(state, jnp.asarray(0, dtype=jnp.int32))

    cycle = absolute // stream.cycle_length
    cycle_step = absolute % stream.cycle_length
    sub = cycle_step // config.segment_length
    segment_step = cycle_step % config.segment_length
    assert bool(result.update_applied)
    chex.assert_trees_all_equal(result.cycle_words, _words(cycle))
    assert int(result.sub_segment) == sub
    assert int(result.segment_step) == segment_step
    assert bool(result.scaled_cycle) == (cycle % 3 == 2 and sub == 0)
    chex.assert_trees_all_equal(result.state.step_words, _words(absolute + 1))


def test_lifetime_scan_crosses_uint32_with_exact_program_phase() -> None:
    """Compiled repeated updates retain exact four-segment program ordering."""
    config = GauntletConfig(relevant_dim=2, irrelevant_dim=0, segment_length=3)
    stream = LifetimeGauntletStream(config, scale_cycle_period=3)
    start = 2**32 - 2
    state = _high_lifetime(stream, start)
    indices = jnp.arange(8, dtype=jnp.int32)

    def body(carry, idx):
        result = stream.step_result(carry, idx)
        return result.state, (result.sub_segment, result.update_applied)

    final, (segments, applied) = jax.jit(
        lambda initial: jax.lax.scan(body, initial, indices)
    )(state)
    expected = jnp.asarray(
        [((start + offset) % stream.cycle_length) // 3 for offset in range(8)],
        dtype=jnp.int32,
    )
    chex.assert_trees_all_equal(segments, expected)
    chex.assert_trees_all_equal(applied, jnp.ones((8,), dtype=jnp.bool_))
    chex.assert_trees_all_equal(final.step_words, _words(start + 8))


def test_lifetime_nonfinite_terminal_and_corrupt_state_are_atomic() -> None:
    """Invalid inputs, corruption, and lifetime exhaustion preserve RNG/tasks."""
    stream = LifetimeGauntletStream(
        GauntletConfig(relevant_dim=2, irrelevant_dim=0, segment_length=3),
        scale_cycle_period=2,
    )
    state = stream.init(jr.key(5))
    nonfinite = stream.step_result(state, jnp.asarray(jnp.nan, dtype=jnp.float32))
    assert not bool(nonfinite.input_valid)
    assert not bool(nonfinite.update_applied)
    chex.assert_trees_all_equal(nonfinite.state, state)

    corrupt = state.replace(w_c=state.w_c.at[0].set(jnp.inf))
    corrupt_result = stream.step_result(corrupt, jnp.asarray(0, dtype=jnp.int32))
    assert not bool(corrupt_result.state_valid)
    assert not bool(corrupt_result.update_applied)
    chex.assert_trees_all_equal(corrupt_result.state, corrupt)

    terminal = _high_lifetime(stream, 2**64 - 1)
    terminal_result = stream.step_result(terminal, jnp.asarray(0, dtype=jnp.int32))
    assert not bool(terminal_result.lifetime_capacity_available)
    assert not bool(terminal_result.update_applied)
    chex.assert_trees_all_equal(terminal_result.state, terminal)


def test_lifetime_schema_migration_resources_and_checkpoint(tmp_path) -> None:
    """Lifetime persistence is schema-strict and migration is unsaturated only."""
    config = GauntletConfig(relevant_dim=2, irrelevant_dim=1, segment_length=4)
    stream = LifetimeGauntletStream(config, scale_cycle_period=3)
    payload = stream.to_config()
    assert payload["config_schema"] == LIFETIME_GAUNTLET_CONFIG_SCHEMA
    assert payload["state_schema"] == LIFETIME_GAUNTLET_STATE_SCHEMA
    assert LifetimeGauntletStream.from_config(payload).to_config() == payload
    with pytest.raises(ValueError, match="fields"):
        LifetimeGauntletStream.from_config({**payload, "extra": 1})
    assert (
        stream.resource_budget.exact_clock_delta_nbytes
        == LIFETIME_GAUNTLET_CLOCK_DELTA_NBYTES
    )

    state = stream.step_result(
        stream.init(jr.key(6)), jnp.asarray(0, dtype=jnp.int32)
    ).state
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state)
        if field.name != "step_words"
    }
    migrated = migrate_legacy_lifetime_gauntlet_state(legacy, stream=stream)
    chex.assert_trees_all_equal(migrated.step_words, jnp.asarray((0, 1), dtype=jnp.uint32))
    with pytest.raises(ValueError, match="saturated"):
        migrate_legacy_lifetime_gauntlet_state(
            {**legacy, "step_count": jnp.asarray(INT32_MAX, dtype=jnp.int32)},
            stream=stream,
        )

    path = tmp_path / "lifetime"
    save_lifetime_gauntlet_checkpoint(stream, state, path)
    restored_stream, restored = load_lifetime_gauntlet_checkpoint(path)
    assert restored_stream.to_config() == payload
    chex.assert_trees_all_equal(restored, state)
