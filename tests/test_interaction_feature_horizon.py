"""Exact transaction-clock contracts for pairwise interaction discovery."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.checkpoints import save_checkpoint
from alberta_framework.core.interaction_features import (
    INTERACTION_FEATURE_CHECKPOINT_SCHEMA,
    INTERACTION_FEATURE_LIFETIME_COUNTER_DELTA_NBYTES,
    INTERACTION_FEATURE_LIFETIME_COUNTER_NBYTES,
    INTERACTION_FEATURE_STATE_SCHEMA,
    INTERACTION_FEATURE_TRANSACTION_CLOCK_DELTA_NBYTES,
    INTERACTION_FEATURE_TRANSACTION_CLOCK_NBYTES,
    FixedBudgetInteractionLearner,
    InteractionFeatureState,
    interaction_feature_lifetime_counter_nbytes,
    interaction_feature_transaction_clock_nbytes,
    load_interaction_feature_checkpoint,
    measure_interaction_feature_state_nbytes,
    migrate_legacy_interaction_feature_state,
    save_interaction_feature_checkpoint,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_OBSERVATION = jnp.asarray((0.5, -0.25), dtype=jnp.float32)
_TARGET = jnp.asarray((0.75,), dtype=jnp.float32)


def _learner(
    *,
    replacement_interval: int = 0,
    candidate_count: int = 0,
    scale_robust: bool = False,
) -> FixedBudgetInteractionLearner:
    return FixedBudgetInteractionLearner(
        n_features=1,
        n_tasks=1,
        candidate_count=candidate_count,
        replacement_interval=replacement_interval,
        min_feature_age=0,
        scale_robust=scale_robust,
    )


def _near_rollover_state(
    learner: FixedBudgetInteractionLearner,
    *,
    replacement_phase: int = 0,
) -> InteractionFeatureState:
    return learner.init(2, jr.key(17)).replace(  # type: ignore[attr-defined,no-any-return]
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.asarray((0, _UINT32_MAX), dtype=jnp.uint32),
        replacement_phase=jnp.asarray(replacement_phase, dtype=jnp.int32),
    )


def test_normal_update_commits_one_exact_transaction_to_both_snapshots() -> None:
    learner = _learner(replacement_interval=3)
    state = learner.init(2, jr.key(0))

    result = learner.update(state, _OBSERVATION, _TARGET)

    assert bool(result.lifetime_counter_valid)
    assert bool(result.lifetime_capacity_available)
    assert bool(result.state_valid)
    assert bool(result.candidate_state_valid)
    assert bool(result.update_applied)
    assert not bool(result.update_rejected)
    np.testing.assert_array_equal(result.pre_step_words, (0, 0))
    np.testing.assert_array_equal(result.post_step_words, (0, 1))
    np.testing.assert_array_equal(result.pre_curation_state.step_words, (0, 1))
    np.testing.assert_array_equal(result.state.step_words, (0, 1))
    assert int(result.pre_curation_state.replacement_phase) == 1
    assert int(result.state.replacement_phase) == 1
    assert int(result.state.step_count) == 1


def test_low_word_rollover_is_exact_under_eager_jit_and_scan() -> None:
    learner = _learner()
    state = _near_rollover_state(learner)

    with jax.disable_jit():
        eager = learner.update(state, _OBSERVATION, _TARGET)
    compiled = jax.jit(learner.update)(state, _OBSERVATION, _TARGET)

    for result in (eager, compiled):
        assert bool(result.update_applied)
        np.testing.assert_array_equal(result.pre_step_words, (0, _UINT32_MAX))
        np.testing.assert_array_equal(result.post_step_words, (1, 0))
        np.testing.assert_array_equal(result.pre_curation_state.step_words, (1, 0))
        assert int(result.state.step_count) == _INT32_MAX
    chex.assert_trees_all_equal(eager.state, compiled.state)

    def step(
        carry: InteractionFeatureState,
        _: jax.Array,
    ) -> tuple[InteractionFeatureState, jax.Array]:
        result = learner.update(carry, _OBSERVATION, _TARGET)
        return result.state, result.post_step_words

    final_state, words = jax.lax.scan(
        step,
        state,
        jnp.arange(2, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        words,
        np.asarray(((1, 0), (1, 1)), dtype=np.uint32),
    )
    np.testing.assert_array_equal(final_state.step_words, (1, 1))


def test_nondivisor_replacement_cadence_survives_low_word_rollover() -> None:
    learner = _learner(replacement_interval=7)
    # 2**32 - 1 == 3 (mod 7). The fourth accepted event is therefore due,
    # even though the first event crosses the low-word boundary.
    state = _near_rollover_state(learner, replacement_phase=3)

    def step(
        carry: InteractionFeatureState,
        _: jax.Array,
    ) -> tuple[InteractionFeatureState, tuple[jax.Array, jax.Array]]:
        result = learner.update(carry, _OBSERVATION, _TARGET)
        return result.state, (
            result.state.replacement_phase,
            result.curation_attempted,
        )

    final_state, (phases, attempted) = jax.lax.scan(
        step,
        state,
        jnp.arange(5, dtype=jnp.int32),
    )

    np.testing.assert_array_equal(phases, (4, 5, 6, 0, 1))
    np.testing.assert_array_equal(attempted, (False, False, False, True, False))
    np.testing.assert_array_equal(final_state.step_words, (1, 4))


def test_counter_corruption_and_phase_corruption_are_atomic_noops() -> None:
    learner = _learner(replacement_interval=3)
    state = learner.init(2, jr.key(2))
    corrupt_counter = state.replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(1, dtype=jnp.int32),
    )
    counter_result = jax.jit(learner.update)(
        corrupt_counter,
        _OBSERVATION,
        _TARGET,
    )

    assert not bool(counter_result.lifetime_counter_valid)
    assert bool(counter_result.lifetime_capacity_available)
    assert not bool(counter_result.state_valid)
    assert not bool(counter_result.update_applied)
    assert bool(counter_result.update_rejected)
    chex.assert_trees_all_equal(counter_result.state, corrupt_counter)
    chex.assert_trees_all_equal(counter_result.pre_curation_state, corrupt_counter)
    np.testing.assert_array_equal(
        counter_result.pre_step_words,
        counter_result.post_step_words,
    )

    for bad_phase in (1, 3):
        # Phase 1 is in range but disagrees with exact identity zero; phase 3
        # is out of range. Both must fail closed rather than shift cadence.
        corrupt_phase = state.replace(  # type: ignore[attr-defined]
            replacement_phase=jnp.asarray(bad_phase, dtype=jnp.int32),
        )
        phase_result = learner.update(corrupt_phase, _OBSERVATION, _TARGET)
        assert bool(phase_result.lifetime_counter_valid)
        assert not bool(phase_result.state_valid)
        assert not bool(phase_result.update_applied)
        assert bool(phase_result.update_rejected)
        chex.assert_trees_all_equal(phase_result.state, corrupt_phase)


def test_all_ones_exhaustion_is_a_diagnosed_atomic_noop() -> None:
    learner = _learner()
    state = learner.init(2, jr.key(3)).replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32),
    )

    result = learner.update(state, _OBSERVATION, _TARGET)

    assert bool(result.lifetime_counter_valid)
    assert not bool(result.lifetime_capacity_available)
    assert bool(result.state_valid)
    assert not bool(result.update_applied)
    assert bool(result.update_rejected)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.pre_curation_state, state)


def test_unused_nonfinite_leaf_and_invalid_counter_leaf_are_atomic_noops() -> None:
    learner = _learner(candidate_count=1)
    state = learner.init(2, jr.key(31))
    invalid_states = (
        state.replace(  # type: ignore[attr-defined]
            candidate_utilities=jnp.asarray((jnp.nan,), dtype=jnp.float32),
        ),
        state.replace(  # type: ignore[attr-defined]
            candidate_ages=jnp.asarray((-1,), dtype=jnp.int32),
        ),
    )

    for invalid in invalid_states:
        result = learner.update(invalid, _OBSERVATION, _TARGET)
        assert not bool(result.state_valid)
        assert not bool(result.update_applied)
        assert bool(result.update_rejected)
        chex.assert_trees_all_equal(result.state, invalid)
        chex.assert_trees_all_equal(result.pre_curation_state, invalid)


def test_nonfinite_candidate_state_is_rejected_after_finite_source() -> None:
    learner = _learner(candidate_count=1)
    state = learner.init(2, jr.key(32)).replace(  # type: ignore[attr-defined]
        candidate_output_weights=jnp.asarray(((3.0e38,),), dtype=jnp.float32),
    )

    result = learner.update(
        state,
        jnp.asarray((2.0, 2.0), dtype=jnp.float32),
        jnp.asarray((0.0,), dtype=jnp.float32),
    )

    assert bool(result.state_valid)
    assert not bool(result.candidate_state_valid)
    assert not bool(result.update_applied)
    assert bool(result.update_rejected)
    chex.assert_trees_all_equal(result.state, state)
    np.testing.assert_array_equal(result.pre_step_words, result.post_step_words)


@pytest.mark.parametrize(
    ("observation", "targets"),
    [
        ((jnp.nan, 0.0), (1.0,)),
        ((0.0, 1.0), (jnp.inf,)),
    ],
)
def test_invalid_dynamic_input_does_not_advance_clock_or_phase(
    observation: tuple[float, float],
    targets: tuple[float],
) -> None:
    learner = _learner(replacement_interval=3)
    state = learner.init(2, jr.key(4))

    result = learner.update(
        state,
        jnp.asarray(observation, dtype=jnp.float32),
        jnp.asarray(targets, dtype=jnp.float32),
    )

    assert bool(result.lifetime_counter_valid)
    assert bool(result.lifetime_capacity_available)
    assert not bool(result.update_applied)
    assert bool(result.update_rejected)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.pre_curation_state, state)


def test_static_state_clock_and_array_contracts_fail_before_update() -> None:
    learner = _learner()
    state = learner.init(2, jr.key(5))
    with pytest.raises(TypeError, match="lifetime words.*uint32"):
        learner.update(
            state.replace(  # type: ignore[attr-defined]
                step_words=jnp.zeros((2,), dtype=jnp.int32)
            ),
            _OBSERVATION,
            _TARGET,
        )
    with pytest.raises(TypeError, match="output_weights.*float32"):
        learner.update(
            state.replace(  # type: ignore[attr-defined]
                output_weights=jnp.zeros((1, 1), dtype=jnp.int32)
            ),
            _OBSERVATION,
            _TARGET,
        )


def test_state_schema_migration_and_byte_accounting_are_strict() -> None:
    learner = _learner(replacement_interval=7)
    state = learner.init(2, jr.key(6))
    payload = learner.to_config()
    assert payload["state_schema"] == INTERACTION_FEATURE_STATE_SCHEMA
    assert FixedBudgetInteractionLearner.from_config(payload).to_config() == payload
    with pytest.raises(ValueError, match="state schema"):
        FixedBudgetInteractionLearner.from_config(
            {**payload, "state_schema": "alberta.interaction-feature-state.v1"}
        )

    legacy = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state)  # type: ignore[arg-type]
        if field.name not in {"step_words", "replacement_phase"}
    }
    legacy["step_count"] = jnp.asarray(19, dtype=jnp.int32)
    migrated = migrate_legacy_interaction_feature_state(
        legacy,
        replacement_interval=7,
    )
    np.testing.assert_array_equal(migrated.step_words, (0, 19))
    assert int(migrated.replacement_phase) == 5
    with pytest.raises(ValueError, match="ambiguous"):
        migrate_legacy_interaction_feature_state(
            {**legacy, "step_count": jnp.asarray(_INT32_MAX, dtype=jnp.int32)},
            replacement_interval=7,
        )
    with pytest.raises(ValueError, match="manifest"):
        migrate_legacy_interaction_feature_state(
            {**legacy, "extra": 1},
            replacement_interval=7,
        )

    measured = measure_interaction_feature_state_nbytes(state)
    without_new_clock = 0
    for field in dataclasses.fields(state):  # type: ignore[arg-type]
        value = getattr(state, field.name)
        if field.name not in {"step_words", "replacement_phase"} and isinstance(
            value,
            jax.Array,
        ):
            without_new_clock += int(value.size) * int(value.dtype.itemsize)
    assert measured == (
        without_new_clock + INTERACTION_FEATURE_TRANSACTION_CLOCK_DELTA_NBYTES
    )
    accounting = learner.memory_accounting(state)
    assert accounting["lifetime_counter_bytes"] == 12
    assert accounting["replacement_phase_bytes"] == 4
    assert accounting["transaction_clock_bytes"] == 16
    assert interaction_feature_lifetime_counter_nbytes() == 12
    assert interaction_feature_transaction_clock_nbytes() == 16
    assert INTERACTION_FEATURE_LIFETIME_COUNTER_NBYTES == 12
    assert INTERACTION_FEATURE_LIFETIME_COUNTER_DELTA_NBYTES == 8
    assert INTERACTION_FEATURE_TRANSACTION_CLOCK_NBYTES == 16
    assert INTERACTION_FEATURE_TRANSACTION_CLOCK_DELTA_NBYTES == 12


def test_current_checkpoint_resumes_exactly_and_v1_is_rejected_precisely(
    tmp_path: Path,
) -> None:
    learner = _learner(
        replacement_interval=7,
        candidate_count=1,
        scale_robust=True,
    )
    state = _near_rollover_state(learner, replacement_phase=3)
    path = tmp_path / "current"
    save_interaction_feature_checkpoint(learner, state, path, feature_dim=2)

    loaded_learner, loaded_state = load_interaction_feature_checkpoint(path)
    chex.assert_trees_all_equal(loaded_state, state)
    direct = learner.update(state, _OBSERVATION, _TARGET)
    resumed = loaded_learner.update(loaded_state, _OBSERVATION, _TARGET)
    chex.assert_trees_all_equal(resumed, direct)
    assert INTERACTION_FEATURE_CHECKPOINT_SCHEMA.endswith(".v2")

    legacy_path = tmp_path / "legacy"
    save_checkpoint(
        state,
        legacy_path,
        metadata={
            "schema": "alberta.interaction-feature-checkpoint.v1",
            "learner_config": learner.to_config(),
            "feature_dim": 2,
            "memory_accounting": learner.memory_accounting(state),
        },
    )
    with pytest.raises(ValueError, match="lacks exact step_words.*resave"):
        load_interaction_feature_checkpoint(legacy_path)


def test_checkpoint_restore_is_independent_of_ambient_prng_default(tmp_path: Path) -> None:
    learner = _learner(candidate_count=1)
    state = learner.init(2, jr.key(13, impl="threefry2x32"))
    path = tmp_path / "checkpoint"
    save_interaction_feature_checkpoint(learner, state, path, feature_dim=2)

    with jax.default_prng_impl("rbg"):
        loaded_learner, loaded_state = load_interaction_feature_checkpoint(path)

    chex.assert_trees_all_equal(loaded_state, state)
    chex.assert_trees_all_equal(
        loaded_learner.update(loaded_state, _OBSERVATION, _TARGET),
        learner.update(state, _OBSERVATION, _TARGET),
    )


def test_init_rejects_non_threefry_key_before_constructing_state() -> None:
    with pytest.raises(TypeError, match="typed scalar threefry2x32"):
        _learner().init(2, jr.key(13, impl="rbg"))


def test_checkpoint_save_rejects_non_threefry_state(tmp_path: Path) -> None:
    learner = _learner()
    state = learner.init(2, jr.key(13, impl="threefry2x32")).replace(  # type: ignore[attr-defined,no-any-return]
        key=jr.key(14, impl="rbg")
    )
    path = tmp_path / "checkpoint"

    with pytest.raises(TypeError, match="typed scalar threefry2x32"):
        save_interaction_feature_checkpoint(learner, state, path, feature_dim=2)

    assert not path.exists()
