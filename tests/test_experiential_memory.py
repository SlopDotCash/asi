"""Tests for bounded experiential memory.

The negative-transfer cases below are controlled mechanism checks only.  They
do not constitute evidence for the WP8 scientific exit gate.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.checkpoints import load_checkpoint, save_checkpoint
from alberta_framework.core.experiential_memory import (
    ExperientialMemory,
    ExperientialMemoryConfig,
    ExperientialMemoryEntry,
    ExperientialMemoryState,
)


def test_experiential_memory_is_publicly_exported() -> None:
    assert alberta.ExperientialMemory is core.ExperientialMemory
    assert alberta.ExperientialMemoryConfig is core.ExperientialMemoryConfig
    assert alberta.ExperientialMemoryEntry is core.ExperientialMemoryEntry
    assert (
        alberta.ExperientialMemoryRetrieval
        is core.ExperientialMemoryRetrieval
    )


def _config(**overrides: Any) -> ExperientialMemoryConfig:
    values: dict[str, Any] = {
        "capacity": 3,
        "observation_dim": 2,
        "key_dim": 2,
        "action_dim": 2,
        "outcome_dim": 1,
        "top_k": 2,
        "min_neighbors": 1,
        "distance_scale": 0.25,
        "min_similarity": 0.5,
        "min_effective_reliability": 0.05,
        "max_uncertainty": 1.0,
        "max_safety_cost": 1.0,
        "max_age": 100,
        "staleness_scale": 100.0,
        "utility_decay": 1.0,
        "eviction_utility_weight": 1.0,
        "eviction_recency_weight": 1.0,
        "recency_scale": 10.0,
    }
    values.update(overrides)
    return ExperientialMemoryConfig(**values)


def _entry(
    provenance_id: int,
    *,
    key: tuple[float, float] = (0.0, 0.0),
    observation: tuple[float, float] | None = None,
    action: tuple[float, float] = (1.0, 0.0),
    outcome: float = 1.0,
    reward: float = 1.0,
    uncertainty: float = 0.1,
    uncertainty_available: bool = True,
    safety_cost: float = 0.0,
    safety_cost_available: bool = True,
    reliability: float = 1.0,
    utility: float = 1.0,
    utility_available: bool = True,
    representation_version: int = 1,
    valid: bool = True,
    age: int = 0,
    source_id: int = 7,
) -> ExperientialMemoryEntry:
    obs = key if observation is None else observation
    return ExperientialMemoryEntry(
        observation=jnp.asarray(obs, dtype=jnp.float32),
        key=jnp.asarray(key, dtype=jnp.float32),
        action=jnp.asarray(action, dtype=jnp.float32),
        outcome=jnp.asarray([outcome], dtype=jnp.float32),
        reward=jnp.asarray(reward, dtype=jnp.float32),
        uncertainty=jnp.asarray(uncertainty, dtype=jnp.float32),
        uncertainty_available=jnp.asarray(uncertainty_available),
        safety_cost=jnp.asarray(safety_cost, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(safety_cost_available),
        reliability=jnp.asarray(reliability, dtype=jnp.float32),
        utility=jnp.asarray(utility, dtype=jnp.float32),
        utility_available=jnp.asarray(utility_available),
        representation_version=jnp.asarray(representation_version, dtype=jnp.int32),
        valid=jnp.asarray(valid),
        age=jnp.asarray(age, dtype=jnp.int32),
        provenance_id=jnp.asarray(provenance_id, dtype=jnp.int32),
        source_id=jnp.asarray(source_id, dtype=jnp.int32),
    )


def _write(
    memory: ExperientialMemory,
    state: ExperientialMemoryState,
    entry: ExperientialMemoryEntry,
) -> ExperientialMemoryState:
    result = memory.write(state, entry)
    assert bool(result.wrote)
    return result.state


def _assert_trees_equal(left: Any, right: Any) -> None:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_config_json_roundtrip_and_memory_roundtrip() -> None:
    config = _config()
    encoded = json.dumps(config.to_config())
    restored = ExperientialMemoryConfig.from_config(json.loads(encoded))
    assert restored == config

    memory = ExperientialMemory(config)
    memory_payload = json.loads(json.dumps(memory.to_config()))
    reconstructed = ExperientialMemory.from_config(memory_payload)
    assert reconstructed.config == config
    assert reconstructed.persistent_bytes == memory.persistent_bytes


def test_single_tiny_positive_neighbor_score_still_has_unit_weight() -> None:
    memory = ExperientialMemory(
        _config(
            distance_scale=1.0,
            min_similarity=0.0,
            min_effective_reliability=1.0e-6,
        )
    )
    entry = _entry(8, key=(6.0, 6.0), action=(2.0, 4.0))
    state = _write(memory, memory.init(), entry)

    retrieval = memory.query(
        state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
    )

    assert bool(retrieval.accepted)
    np.testing.assert_array_equal(retrieval.neighbor_mask, [True, False])
    np.testing.assert_allclose(retrieval.neighbor_weights, [1.0, 0.0])
    np.testing.assert_allclose(retrieval.action, entry.action)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capacity", 0),
        ("capacity", 2.0),
        ("capacity", True),
        ("observation_dim", 0),
        ("key_dim", 0),
        ("action_dim", 0),
        ("outcome_dim", 0),
        ("top_k", 0),
        ("min_neighbors", 0),
        ("distance_scale", 0.0),
        ("distance_scale", float("nan")),
        ("min_similarity", float("inf")),
        ("min_similarity", -0.1),
        ("min_effective_reliability", 0.0),
        ("max_uncertainty", -0.1),
        ("max_safety_cost", float("nan")),
        ("max_age", -1),
        ("max_age", 2**31),
        ("max_age", 10**10),
        ("max_age", 1.5),
        ("staleness_scale", float("inf")),
        ("utility_decay", 1.1),
        ("eviction_utility_weight", -0.1),
        ("eviction_recency_weight", float("nan")),
        ("recency_scale", 0.0),
    ],
)
def test_config_rejects_invalid_or_nonfinite_values(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        ExperientialMemory(replace(_config(), **{field: value}))


def test_config_rejects_inconsistent_neighbor_and_eviction_settings() -> None:
    with pytest.raises(ValueError, match="top_k"):
        ExperientialMemory(_config(capacity=2, top_k=3))
    with pytest.raises(ValueError, match="min_neighbors"):
        ExperientialMemory(_config(top_k=1, min_neighbors=2))
    with pytest.raises(ValueError, match="retention weight"):
        ExperientialMemory(_config(eviction_utility_weight=0.0, eviction_recency_weight=0.0))


def test_config_rejects_allocation_larger_than_exact_byte_counter() -> None:
    with pytest.raises(ValueError, match="uint32|int32"):
        ExperientialMemory(_config(capacity=100_000_000, top_k=1))


def test_initial_allocation_and_accounting_are_exact() -> None:
    memory = ExperientialMemory(_config(capacity=5, top_k=3))
    state = memory.init()
    accounting = memory.accounting(state)

    exact_state_bytes = sum(
        int(leaf.size) * int(leaf.dtype.itemsize) for leaf in jax.tree.leaves(state)
    )
    exact_slot_bytes = (
        sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in jax.tree.leaves(state.entries))
        // memory.config.capacity
    )

    assert int(state.persistent_bytes) == exact_state_bytes
    assert memory.persistent_bytes == exact_state_bytes
    assert memory.slot_bytes == exact_slot_bytes
    assert int(accounting.persistent_bytes) == exact_state_bytes
    assert int(accounting.slot_bytes) == exact_slot_bytes
    assert int(accounting.active_entries) == 0
    assert int(accounting.capacity_entries) == 5
    assert state.entries.observations.shape == (5, 2)
    assert state.entries.keys.shape == (5, 2)
    assert state.entries.actions.shape == (5, 2)
    assert state.entries.outcomes.shape == (5, 1)


def test_new_entry_recency_starts_at_zero_not_chronological_age() -> None:
    """A freshly written exemplar has not gone ``entry.age`` steps without access."""
    memory = ExperientialMemory(_config())
    result = memory.write(memory.init(), _entry(17, age=500))

    assert bool(result.wrote)
    slot = int(result.slot)
    assert int(result.state.entries.ages[slot]) == 500
    assert int(result.state.entries.recency_ages[slot]) == 0


def test_imported_age_does_not_override_lru_write_recency() -> None:
    """A newly written old exemplar must not look older than prior writes."""
    memory = ExperientialMemory(
        _config(
            capacity=2,
            top_k=1,
            eviction_utility_weight=0.0,
            eviction_recency_weight=1.0,
        )
    )
    state = _write(memory, memory.init(), _entry(10))
    state = _write(memory, state, _entry(20))

    third = memory.write(state, _entry(30, age=500))
    assert bool(third.wrote)
    assert int(third.evicted_provenance_id) == 10
    fourth = memory.write(third.state, _entry(40))

    assert bool(fourth.wrote)
    assert int(fourth.evicted_provenance_id) == 20
    retained = set(
        np.asarray(fourth.state.entries.provenance_ids)[
            np.asarray(fourth.state.entries.valid)
        ].tolist()
    )
    assert retained == {30, 40}


def test_empty_query_abstains_with_finite_zero_payload() -> None:
    memory = ExperientialMemory(_config())
    retrieval = memory.query(
        memory.init(),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray(1),
        jnp.asarray(0.0),
        jnp.asarray(True),
    )

    assert not bool(retrieval.accepted)
    assert not bool(retrieval.has_neighbors)
    assert not bool(retrieval.version_compatible)
    chex.assert_tree_all_finite(retrieval)
    np.testing.assert_array_equal(retrieval.action, jnp.zeros((2,)))
    np.testing.assert_array_equal(retrieval.outcome, jnp.zeros((1,)))
    assert float(retrieval.reward) == 0.0


def test_step_queries_before_writing_current_exemplar() -> None:
    memory = ExperientialMemory(_config())
    current = _entry(10, key=(0.0, 0.0), action=(0.25, 0.75))

    first = memory.step(
        memory.init(),
        current.key,
        current.representation_version,
        jnp.asarray(0.0),
        jnp.asarray(True),
        current,
    )
    assert not bool(first.retrieval.accepted)
    assert bool(first.wrote)
    assert int(first.state.active_count) == 1
    assert int(first.state.query_count) == 1
    assert int(first.state.accepted_query_count) == 0

    later = memory.query(
        first.state,
        current.key,
        current.representation_version,
        jnp.asarray(0.0),
        jnp.asarray(True),
    )
    assert bool(later.accepted)
    np.testing.assert_allclose(later.action, current.action, atol=1e-6)


def test_similarity_reliability_and_staleness_shape_neighbor_weights() -> None:
    memory = ExperientialMemory(
        _config(
            capacity=2,
            top_k=2,
            min_similarity=0.0,
            min_effective_reliability=1e-5,
            staleness_scale=1.0,
        )
    )
    state = _write(
        memory,
        memory.init(),
        _entry(1, action=(1.0, 0.0), reliability=1.0, age=0),
    )
    state = _write(
        memory,
        state,
        _entry(2, action=(0.0, 1.0), reliability=1.0, age=5),
    )

    retrieval = memory.query(
        state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(True),
    )
    assert bool(retrieval.accepted)
    assert float(retrieval.action[0]) > 0.98
    assert float(retrieval.action[1]) < 0.02
    selected = retrieval.neighbor_mask
    assert int(jnp.sum(selected)) == 2
    assert float(jnp.max(retrieval.neighbor_reliabilities)) > float(
        jnp.min(retrieval.neighbor_reliabilities)
    )

    low_reliability_memory = ExperientialMemory(
        _config(capacity=1, top_k=1, min_effective_reliability=0.5)
    )
    low_state = _write(
        low_reliability_memory,
        low_reliability_memory.init(),
        _entry(3, reliability=0.1),
    )
    rejected = low_reliability_memory.query(
        low_state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(True),
    )
    assert not bool(rejected.accepted)
    assert not bool(rejected.has_neighbors)


def test_retrieval_gate_fails_closed_for_version_uncertainty_safety_and_distance() -> None:
    config = _config(capacity=1, top_k=1, min_similarity=0.8)
    memory = ExperientialMemory(config)
    safe_state = _write(memory, memory.init(), _entry(1))

    wrong_version = memory.query(
        safe_state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(True),
    )
    assert not bool(wrong_version.accepted)
    assert not bool(wrong_version.version_compatible)
    np.testing.assert_array_equal(wrong_version.action, jnp.zeros((2,)))

    uncertain_query = memory.query(
        safe_state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(1.01, dtype=jnp.float32),
        jnp.asarray(True),
    )
    assert not bool(uncertain_query.accepted)
    assert not bool(uncertain_query.uncertainty_ok)

    uncertain_state = _write(
        memory, memory.init(), _entry(2, uncertainty=config.max_uncertainty + 0.1)
    )
    uncertain_memory = memory.query(
        uncertain_state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(True),
    )
    assert not bool(uncertain_memory.accepted)
    assert not bool(uncertain_memory.uncertainty_ok)

    unsafe_state = _write(
        memory, memory.init(), _entry(3, safety_cost=config.max_safety_cost + 0.1)
    )
    unsafe = memory.query(
        unsafe_state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(True),
    )
    assert not bool(unsafe.accepted)
    assert not bool(unsafe.safety_ok)

    distant = memory.query(
        safe_state,
        jnp.asarray([10.0, 10.0], dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(True),
    )
    assert not bool(distant.accepted)
    assert bool(distant.version_compatible)
    assert not bool(distant.has_neighbors)


def test_nonfinite_or_invalid_entries_are_rejected_without_slot_mutation() -> None:
    memory = ExperientialMemory(_config())
    invalid = _entry(1, reward=float("nan"))
    result = memory.write(memory.init(), invalid)

    assert not bool(result.wrote)
    assert int(result.slot) == -1
    assert int(result.state.active_count) == 0
    assert int(result.state.write_count) == 0
    assert int(result.state.rejected_write_count) == 1
    assert int(result.state.step_count) == 1
    assert not bool(jnp.any(result.state.entries.valid))


def test_eviction_is_deterministic_and_reward_does_not_replace_explicit_utility() -> None:
    memory = ExperientialMemory(
        _config(
            capacity=2,
            top_k=1,
            utility_decay=1.0,
            eviction_utility_weight=1.0,
            eviction_recency_weight=0.0,
        )
    )
    state = _write(
        memory,
        memory.init(),
        _entry(10, utility=0.1, reward=1_000.0),
    )
    state = _write(memory, state, _entry(20, key=(1.0, 0.0), utility=0.9, reward=-1.0))
    result = memory.write(state, _entry(30, key=(2.0, 0.0), utility=0.5))

    assert bool(result.evicted)
    assert int(result.evicted_provenance_id) == 10
    assert int(result.slot) == 0
    assert set(np.asarray(result.state.entries.provenance_ids).tolist()) == {20, 30}
    accounting = memory.accounting(result.state)
    assert int(accounting.active_entries) == 2
    assert int(accounting.writes) == 3
    assert int(accounting.evictions) == 1


def test_accepted_access_updates_recency_before_deterministic_eviction() -> None:
    memory = ExperientialMemory(
        _config(
            capacity=2,
            top_k=1,
            min_similarity=0.9,
            utility_decay=1.0,
            eviction_utility_weight=0.0,
            eviction_recency_weight=1.0,
            recency_scale=1.0,
        )
    )
    state = _write(memory, memory.init(), _entry(10, key=(0.0, 0.0)))
    state = _write(memory, state, _entry(20, key=(1.0, 0.0)))

    result = memory.step(
        state,
        jnp.asarray([0.0, 0.0]),
        jnp.asarray(1),
        jnp.asarray(0.0),
        jnp.asarray(True),
        _entry(30, key=(2.0, 0.0)),
    )

    assert bool(result.retrieval.accepted)
    assert bool(result.evicted)
    assert int(result.evicted_provenance_id) == 20
    assert set(np.asarray(result.state.entries.provenance_ids).tolist()) == {10, 30}
    slot_for_ten = int(jnp.argmax(result.state.entries.provenance_ids == 10))
    assert int(result.state.entries.retrieval_counts[slot_for_ten]) == 1
    assert int(result.state.entries.recency_ages[slot_for_ten]) == 0


def _scan_inputs() -> tuple[jax.Array, ...]:
    keys = jnp.asarray(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [2.0, 0.0]],
        dtype=jnp.float32,
    )
    actions = jnp.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.9, 0.1], [0.5, 0.5]],
        dtype=jnp.float32,
    )
    rewards = jnp.arange(5, dtype=jnp.float32)
    provenance_ids = jnp.arange(100, 105, dtype=jnp.int32)
    return keys, actions, rewards, provenance_ids


def _scan_memory(
    memory: ExperientialMemory,
    state: ExperientialMemoryState,
    inputs: tuple[jax.Array, ...],
) -> tuple[ExperientialMemoryState, tuple[jax.Array, jax.Array]]:
    def body(
        carry: ExperientialMemoryState,
        values: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[ExperientialMemoryState, tuple[jax.Array, jax.Array]]:
        key, action, reward, provenance_id = values
        entry = ExperientialMemoryEntry(
            observation=key,
            key=key,
            action=action,
            outcome=reward[None],
            reward=reward,
            uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
            uncertainty_available=jnp.asarray(True),
            safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
            safety_cost_available=jnp.asarray(True),
            reliability=jnp.asarray(0.9, dtype=jnp.float32),
            utility=jnp.asarray(0.5, dtype=jnp.float32),
            utility_available=jnp.asarray(True),
            representation_version=jnp.asarray(1, dtype=jnp.int32),
            valid=jnp.asarray(True),
            age=jnp.asarray(0, dtype=jnp.int32),
            provenance_id=provenance_id,
            source_id=jnp.asarray(9, dtype=jnp.int32),
        )
        result = memory.step(
            carry,
            key,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0.1, dtype=jnp.float32),
            jnp.asarray(True),
            entry,
        )
        return result.state, (result.retrieval.accepted, result.retrieval.reward)

    return jax.lax.scan(body, state, inputs)


def test_jit_and_scan_match_eager_execution() -> None:
    memory = ExperientialMemory(_config(capacity=3, top_k=2, min_similarity=0.1))
    inputs = _scan_inputs()

    eager_state, eager_outputs = _scan_memory(memory, memory.init(), inputs)
    compiled = jax.jit(lambda state: _scan_memory(memory, state, inputs))
    compiled_state, compiled_outputs = compiled(memory.init())

    _assert_trees_equal(eager_state, compiled_state)
    _assert_trees_equal(eager_outputs, compiled_outputs)
    assert int(compiled_state.active_count) == memory.config.capacity
    assert int(compiled_state.step_count) == inputs[0].shape[0]
    assert int(compiled_state.query_count) == inputs[0].shape[0]
    assert int(compiled_state.write_count) == inputs[0].shape[0]


def test_checkpoint_resume_matches_uninterrupted_scan(tmp_path: Any) -> None:
    memory = ExperientialMemory(_config(capacity=3, top_k=2, min_similarity=0.1))
    inputs = _scan_inputs()
    prefix_inputs = tuple(values[:3] for values in inputs)
    suffix_inputs = tuple(values[3:] for values in inputs)

    prefix_state, _ = _scan_memory(memory, memory.init(), prefix_inputs)
    checkpoint_path = tmp_path / "experiential-memory"
    save_checkpoint(
        prefix_state,
        checkpoint_path,
        metadata={"memory": memory.to_config()},
    )
    loaded, metadata = load_checkpoint(memory.init(), checkpoint_path)
    loaded_state = cast(ExperientialMemoryState, loaded)

    assert metadata["memory"] == memory.to_config()
    _assert_trees_equal(prefix_state, loaded_state)

    resumed_state, resumed_outputs = _scan_memory(memory, loaded_state, suffix_inputs)
    direct_state, direct_outputs = _scan_memory(memory, prefix_state, suffix_inputs)
    uninterrupted_state, uninterrupted_outputs = _scan_memory(memory, memory.init(), inputs)

    _assert_trees_equal(resumed_state, direct_state)
    _assert_trees_equal(resumed_outputs, direct_outputs)
    _assert_trees_equal(resumed_state, uninterrupted_state)
    _assert_trees_equal(
        resumed_outputs,
        tuple(values[3:] for values in uninterrupted_outputs),
    )


def test_controlled_stale_wrong_version_negative_transfer_is_gated() -> None:
    """A stale/wrong-version exemplar cannot override a safe base action."""
    memory = ExperientialMemory(_config(capacity=2, top_k=1, max_age=2, min_similarity=0.9))
    harmful = _entry(
        40,
        action=(-1.0, 1.0),
        outcome=-100.0,
        reward=-100.0,
        representation_version=1,
        age=3,
    )
    state = _write(memory, memory.init(), harmful)
    retrieval = memory.query(
        state,
        harmful.key,
        jnp.asarray(2),
        jnp.asarray(0.0),
        jnp.asarray(True),
    )
    base_action = jnp.asarray([0.4, 0.6], dtype=jnp.float32)
    selected_action = jnp.where(retrieval.accepted, retrieval.action, base_action)

    assert not bool(retrieval.accepted)
    assert not bool(retrieval.version_compatible)
    np.testing.assert_array_equal(retrieval.action, jnp.zeros((2,)))
    np.testing.assert_array_equal(selected_action, base_action)

    fresh = _entry(
        41,
        action=(0.2, 0.8),
        representation_version=2,
        age=0,
    )
    state = _write(memory, state, fresh)
    compatible = memory.query(
        state,
        fresh.key,
        jnp.asarray(2),
        jnp.asarray(0.0),
        jnp.asarray(True),
    )
    assert bool(compatible.accepted)
    np.testing.assert_allclose(compatible.action, fresh.action, atol=1e-6)


@pytest.mark.parametrize(
    ("entry_overrides", "missing_field"),
    [
        ({"uncertainty": 0.0, "uncertainty_available": False}, "uncertainty_available"),
        ({"safety_cost": 0.0, "safety_cost_available": False}, "safety_cost_available"),
    ],
)
def test_unavailable_risk_metadata_is_not_treated_as_measured_zero(
    entry_overrides: dict[str, Any],
    missing_field: str,
) -> None:
    memory = ExperientialMemory(_config(capacity=1, top_k=1))
    state = _write(memory, memory.init(), _entry(1, **entry_overrides))

    retrieval = memory.query(
        state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(True),
    )

    assert bool(retrieval.state_valid)
    assert not bool(retrieval.accepted)
    assert not bool(getattr(retrieval, missing_field))
    np.testing.assert_array_equal(retrieval.action, jnp.zeros((2,), dtype=jnp.float32))


def test_unavailable_query_uncertainty_abstains_even_when_numeric_value_is_zero() -> None:
    memory = ExperientialMemory(_config(capacity=1, top_k=1))
    state = _write(memory, memory.init(), _entry(1))

    retrieval = memory.query(
        state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(False),
    )

    assert bool(retrieval.state_valid)
    assert not bool(retrieval.query_valid)
    assert not bool(retrieval.uncertainty_available)
    assert not bool(retrieval.accepted)


@pytest.mark.parametrize(
    "entry",
    [
        _entry(1, uncertainty=0.1, uncertainty_available=False),
        _entry(1, safety_cost=0.1, safety_cost_available=False),
        _entry(1, utility=0.1, utility_available=False),
    ],
)
def test_unavailable_metadata_must_use_an_explicit_zero(
    entry: ExperientialMemoryEntry,
) -> None:
    memory = ExperientialMemory(_config())
    result = memory.write(memory.init(), entry)

    assert not bool(result.wrote)
    assert int(result.state.rejected_write_count) == 1


def test_same_size_wrong_shapes_and_dtype_aliases_are_rejected_before_tracing() -> None:
    memory = ExperientialMemory(_config())
    state = memory.init()
    wrong_entry = _entry(1).replace(
        action=jnp.ones((1, 2), dtype=jnp.float32)
    )

    with pytest.raises(ValueError, match=r"entry\.action must have shape"):
        memory.write(state, wrong_entry)
    with pytest.raises(ValueError, match=r"entry\.action must have shape"):
        jax.jit(lambda current: memory.write(current, wrong_entry))(state)
    with pytest.raises(TypeError, match="key must have dtype float32"):
        memory.query(
            state,
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(True),
        )


def test_dynamic_state_corruption_makes_query_abstain_and_mutations_exact_noops() -> None:
    memory = ExperientialMemory(_config())
    original = memory.init()
    corrupt = original.replace(active_count=jnp.asarray(1, dtype=jnp.int32))
    entry = _entry(1)
    query_args = (
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(True),
    )

    retrieval = memory.query(corrupt, *query_args)
    write_result = memory.write(corrupt, entry)
    step_result = memory.step(corrupt, *query_args, entry)

    assert not bool(retrieval.state_valid)
    assert not bool(retrieval.accepted)
    chex.assert_tree_all_finite(retrieval)
    assert not bool(write_result.wrote)
    assert not bool(step_result.wrote)
    _assert_trees_equal(write_result.state, corrupt)
    _assert_trees_equal(step_result.state, corrupt)

    compiled_step = jax.jit(lambda current: memory.step(current, *query_args, entry))
    compiled_result = compiled_step(corrupt)
    _assert_trees_equal(compiled_result.state, corrupt)
    assert not bool(compiled_result.retrieval.state_valid)
