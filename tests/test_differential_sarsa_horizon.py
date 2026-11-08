"""Exact lifetime-clock contracts for differential SARSA control."""

from __future__ import annotations

import dataclasses
from typing import cast

import chex
import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.average_reward import (
    DIFFERENTIAL_SARSA_LIFETIME_COUNTER_DELTA_NBYTES,
    DIFFERENTIAL_SARSA_LIFETIME_COUNTER_NBYTES,
    DIFFERENTIAL_SARSA_STATE_SCHEMA,
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    DifferentialSARSAState,
    differential_sarsa_lifetime_counter_nbytes,
    measure_differential_sarsa_state_nbytes,
    migrate_legacy_differential_sarsa_state,
    run_differential_sarsa_from_arrays,
)

pytestmark = pytest.mark.unit


def _agent() -> DifferentialSARSAAgent:
    return DifferentialSARSAAgent(
        DifferentialSARSAConfig(
            n_actions=2,
            q_step_size=0.1,
            average_reward_step_size=0.01,
            epsilon_start=0.2,
            epsilon_end=0.2,
        )
    )


def _primed_state(agent: DifferentialSARSAAgent) -> DifferentialSARSAState:
    state = agent.init(2, jax.random.key(17))
    state, _ = agent.start_with_action(
        state,
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    return cast(DifferentialSARSAState, state)


def test_normal_update_commits_one_exact_transition() -> None:
    agent = _agent()
    state = _primed_state(agent)
    result = agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray((0.0, 1.0), dtype=jnp.float32),
        next_action=jnp.asarray(1, dtype=jnp.int32),
    )

    assert bool(result.lifetime_counter_valid)
    assert bool(result.lifetime_capacity_available)
    assert bool(result.state_valid)
    assert bool(result.input_valid)
    assert bool(result.candidate_state_finite)
    assert bool(result.update_applied)
    chex.assert_trees_all_equal(result.pre_step_words, jnp.asarray((0, 0), dtype=jnp.uint32))
    chex.assert_trees_all_equal(result.post_step_words, jnp.asarray((0, 1), dtype=jnp.uint32))
    chex.assert_trees_all_equal(result.state.step_words, result.post_step_words)
    assert int(result.state.step_count) == 1
    assert int(result.action) == 1


def test_word_carry_scans_and_all_ones_exhaustion_is_atomic() -> None:
    agent = _agent()
    state = cast(
        DifferentialSARSAState,
        _primed_state(agent).replace(  # type: ignore[attr-defined]
            step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
            step_words=jnp.asarray((0, 2**32 - 1), dtype=jnp.uint32),
        ),
    )

    @jax.jit
    def two_steps(
        initial: DifferentialSARSAState,
    ) -> tuple[DifferentialSARSAState, jax.Array]:
        def step(
            carry: DifferentialSARSAState,
            _: jax.Array,
        ) -> tuple[DifferentialSARSAState, jax.Array]:
            result = agent.update(
                carry,
                jnp.asarray(0.25, dtype=jnp.float32),
                jnp.asarray((0.0, 1.0), dtype=jnp.float32),
                next_action=jnp.asarray(1, dtype=jnp.int32),
            )
            return result.state, result.post_step_words

        return jax.lax.scan(step, initial, jnp.arange(2, dtype=jnp.int32))

    carried, words = two_steps(state)
    chex.assert_trees_all_equal(
        words,
        jnp.asarray(((1, 0), (1, 1)), dtype=jnp.uint32),
    )
    assert int(carried.step_count) == 2**31 - 1

    exhausted = cast(
        DifferentialSARSAState,
        carried.replace(
            step_words=jnp.full((2,), 2**32 - 1, dtype=jnp.uint32),
        ),
    )
    stopped = agent.update(
        exhausted,
        jnp.asarray(3.0, dtype=jnp.float32),
        jnp.asarray((1.0, 1.0), dtype=jnp.float32),
        next_action=jnp.asarray(0, dtype=jnp.int32),
    )
    assert bool(stopped.lifetime_counter_valid)
    assert not bool(stopped.lifetime_capacity_available)
    assert not bool(stopped.update_applied)
    chex.assert_trees_all_equal(stopped.state, exhausted)
    chex.assert_trees_all_equal(stopped.pre_step_words, stopped.post_step_words)
    assert int(stopped.action) == int(exhausted.last_action)


def test_counter_corruption_refuses_parameter_trace_and_rng_mutation() -> None:
    agent = _agent()
    invalid = cast(
        DifferentialSARSAState,
        _primed_state(agent).replace(  # type: ignore[attr-defined]
            step_count=jnp.asarray(1, dtype=jnp.int32),
        ),
    )
    result = jax.jit(agent.update)(
        invalid,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray((0.0, 1.0), dtype=jnp.float32),
    )

    assert not bool(result.lifetime_counter_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, invalid)
    assert int(result.action) == int(invalid.last_action)


@pytest.mark.parametrize(
    ("reward", "observation", "action", "discount"),
    [
        (float("nan"), (0.0, 1.0), 1, 1.0),
        (1.0, (float("inf"), 1.0), 1, 1.0),
        (1.0, (0.0, 1.0), 2, 1.0),
        (1.0, (0.0, 1.0), 1, -0.1),
        (1.0, (0.0, 1.0), 1, 1.1),
    ],
)
def test_invalid_dynamic_input_is_an_atomic_noop(
    reward: float,
    observation: tuple[float, float],
    action: int,
    discount: float,
) -> None:
    agent = _agent()
    state = _primed_state(agent)
    result = agent.update(
        state,
        jnp.asarray(reward, dtype=jnp.float32),
        jnp.asarray(observation, dtype=jnp.float32),
        next_action=jnp.asarray(action, dtype=jnp.int32),
        discount=jnp.asarray(discount, dtype=jnp.float32),
    )

    assert bool(result.state_valid)
    assert not bool(result.input_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


def test_nonfinite_state_and_overflowing_candidate_fail_closed() -> None:
    agent = _agent()
    state = _primed_state(agent)
    nonfinite = cast(
        DifferentialSARSAState,
        state.replace(  # type: ignore[attr-defined]
            q_weights=state.q_weights.at[0, 0].set(jnp.asarray(jnp.nan)),
        ),
    )
    rejected_state = agent.update(
        nonfinite,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray((0.0, 1.0), dtype=jnp.float32),
        next_action=jnp.asarray(1, dtype=jnp.int32),
    )
    assert not bool(rejected_state.state_valid)
    assert not bool(rejected_state.update_applied)
    chex.assert_trees_all_equal(rejected_state.state, nonfinite)

    overflowing = cast(
        DifferentialSARSAState,
        state.replace(  # type: ignore[attr-defined]
            q_weights=jnp.full_like(state.q_weights, 3.0e38),
            last_observation=jnp.ones_like(state.last_observation),
        ),
    )
    rejected_candidate = agent.update(
        overflowing,
        jnp.asarray(3.0e38, dtype=jnp.float32),
        jnp.ones_like(state.last_observation),
        next_action=jnp.asarray(1, dtype=jnp.int32),
    )
    assert bool(rejected_candidate.state_valid)
    assert bool(rejected_candidate.input_valid)
    assert not bool(rejected_candidate.candidate_state_finite)
    assert not bool(rejected_candidate.update_applied)
    chex.assert_trees_all_equal(rejected_candidate.state, overflowing)


def test_array_runner_exposes_rejected_updates_instead_of_plausible_progress() -> None:
    agent = _agent()
    state = cast(
        DifferentialSARSAState,
        _primed_state(agent).replace(  # type: ignore[attr-defined]
            step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
            step_words=jnp.full((2,), 2**32 - 1, dtype=jnp.uint32),
        ),
    )
    result = run_differential_sarsa_from_arrays(
        agent,
        state,
        jnp.asarray((1.0, 1.0), dtype=jnp.float32),
        jnp.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=jnp.float32),
    )

    chex.assert_trees_all_equal(
        result.updates_applied,
        jnp.asarray((False, False), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(result.state.q_weights, state.q_weights)
    chex.assert_trees_all_equal(result.state.step_words, state.step_words)


def test_schema_migration_and_counter_byte_accounting_are_strict() -> None:
    agent = _agent()
    state = _primed_state(agent)
    payload = agent.to_config()
    assert payload["state_schema"] == DIFFERENTIAL_SARSA_STATE_SCHEMA
    assert DifferentialSARSAAgent.from_config(payload).config == agent.config
    with pytest.raises(ValueError, match="state schema"):
        DifferentialSARSAAgent.from_config(
            {**payload, "state_schema": "alberta.differential-sarsa-state.v1"}
        )
    with pytest.raises(ValueError, match="state schema"):
        DifferentialSARSAAgent.from_config(
            {**payload, "state_schema": "alberta.differential-sarsa-state.v2"}
        )

    # The exact v1 manifest: no step_words (v2) and no previous_discount (v3).
    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state)  # type: ignore[arg-type]
        if field.name not in {"step_words", "previous_discount"}
    }
    legacy["step_count"] = jnp.asarray(19, dtype=jnp.int32)
    migrated = migrate_legacy_differential_sarsa_state(legacy)
    chex.assert_trees_all_equal(
        migrated.step_words,
        jnp.asarray((0, 19), dtype=jnp.uint32),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_differential_sarsa_state(
            {**legacy, "step_count": jnp.asarray(2**31 - 1, dtype=jnp.int32)}
        )
    with pytest.raises(ValueError, match="manifest"):
        migrate_legacy_differential_sarsa_state({**legacy, "extra": 1})

    measured = measure_differential_sarsa_state_nbytes(state)
    without_words = 0
    for field in dataclasses.fields(state):  # type: ignore[arg-type]
        value = getattr(state, field.name)
        if field.name != "step_words" and isinstance(value, jax.Array):
            without_words += int(value.size) * int(value.dtype.itemsize)
    assert measured == without_words + DIFFERENTIAL_SARSA_LIFETIME_COUNTER_DELTA_NBYTES
    assert differential_sarsa_lifetime_counter_nbytes() == 12
    assert DIFFERENTIAL_SARSA_LIFETIME_COUNTER_NBYTES == 12
    assert DIFFERENTIAL_SARSA_LIFETIME_COUNTER_DELTA_NBYTES == 8
