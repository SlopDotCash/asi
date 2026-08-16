# mypy: disable-error-code="call-arg,type-var"
"""Tests for experiential-memory policy proposal and transaction boundaries."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.experiential_memory import (
    ExperientialMemory,
    ExperientialMemoryConfig,
    ExperientialMemoryEntry,
    ExperientialMemoryState,
)
from alberta_framework.core.experiential_memory_policy import (
    EXPERIENTIAL_MEMORY_POLICY_SCHEMA,
    ExperientialMemoryPolicy,
    ExperientialMemoryPolicyProposal,
)

pytestmark = pytest.mark.unit


def _memory_config(**overrides: Any) -> ExperientialMemoryConfig:
    values: dict[str, Any] = {
        "capacity": 3,
        "observation_dim": 2,
        "key_dim": 2,
        "action_dim": 3,
        "outcome_dim": 1,
        "top_k": 2,
        "min_neighbors": 1,
        "distance_scale": 0.25,
        "min_similarity": 0.5,
        "min_effective_reliability": 0.05,
        "max_uncertainty": 0.5,
        "max_safety_cost": 0.5,
        "max_age": 3,
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
    action_mass: tuple[float, float, float] = (1.0, 3.0, 2.0),
    key: tuple[float, float] = (0.0, 0.0),
    representation_version: int = 1,
    age: int = 0,
    uncertainty: float = 0.1,
    uncertainty_available: bool = True,
    safety_cost: float = 0.0,
    safety_cost_available: bool = True,
) -> ExperientialMemoryEntry:
    return ExperientialMemoryEntry(
        observation=jnp.asarray(key, dtype=jnp.float32),
        key=jnp.asarray(key, dtype=jnp.float32),
        action=jnp.asarray(action_mass, dtype=jnp.float32),
        outcome=jnp.asarray([1.0], dtype=jnp.float32),
        reward=jnp.asarray(1.0, dtype=jnp.float32),
        uncertainty=jnp.asarray(uncertainty, dtype=jnp.float32),
        uncertainty_available=jnp.asarray(uncertainty_available, dtype=jnp.bool_),
        safety_cost=jnp.asarray(safety_cost, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(safety_cost_available, dtype=jnp.bool_),
        reliability=jnp.asarray(1.0, dtype=jnp.float32),
        utility=jnp.asarray(1.0, dtype=jnp.float32),
        utility_available=jnp.asarray(True, dtype=jnp.bool_),
        representation_version=jnp.asarray(representation_version, dtype=jnp.int32),
        valid=jnp.asarray(True, dtype=jnp.bool_),
        age=jnp.asarray(age, dtype=jnp.int32),
        provenance_id=jnp.asarray(provenance_id, dtype=jnp.int32),
        source_id=jnp.asarray(0, dtype=jnp.int32),
    )


def _write(
    memory: ExperientialMemory,
    state: ExperientialMemoryState,
    entry: ExperientialMemoryEntry,
) -> ExperientialMemoryState:
    result = memory.write(state, entry)
    assert bool(result.wrote)
    return result.state


def _propose(
    policy: ExperientialMemoryPolicy,
    state: ExperientialMemoryState,
    *,
    key: tuple[float, float] = (0.0, 0.0),
    version: int = 1,
    uncertainty: float = 0.1,
    uncertainty_available: bool = True,
    safety_mask: tuple[bool, bool, bool] = (True, True, True),
) -> ExperientialMemoryPolicyProposal:
    return policy.propose(
        state,
        jnp.asarray(key, dtype=jnp.float32),
        jnp.asarray(version, dtype=jnp.int32),
        jnp.asarray(uncertainty, dtype=jnp.float32),
        jnp.asarray(uncertainty_available, dtype=jnp.bool_),
        jnp.asarray(safety_mask, dtype=jnp.bool_),
    )


def _assert_trees_equal(left: object, right: object) -> None:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_policy_is_public_and_construction_roundtrip_is_strict() -> None:
    assert alberta.ExperientialMemoryPolicy is core.ExperientialMemoryPolicy
    assert alberta.ExperientialMemoryPolicyProposal is core.ExperientialMemoryPolicyProposal
    memory = ExperientialMemory(_memory_config())
    policy = ExperientialMemoryPolicy(memory)
    payload = policy.to_config()

    assert payload["schema"] == EXPERIENTIAL_MEMORY_POLICY_SCHEMA
    assert payload["action_semantics"] == "categorical-score-mass-not-action-identifiers"
    assert payload["calibrated_confidence_claimed"] is False
    assert payload["benefit_claimed"] is False
    restored = ExperientialMemoryPolicy.from_config(payload)
    assert restored.to_config() == payload
    assert restored.memory.to_config() == memory.to_config()

    changed = dict(payload)
    changed["calibrated_confidence_claimed"] = True
    with pytest.raises(ValueError, match="confidence"):
        ExperientialMemoryPolicy.from_config(changed)
    changed = dict(payload)
    changed["extra"] = 1
    with pytest.raises(ValueError, match="fields"):
        ExperientialMemoryPolicy.from_config(changed)


def test_resource_declaration_has_zero_owned_state_and_no_rng() -> None:
    memory = ExperientialMemory(_memory_config())
    resources = ExperientialMemoryPolicy(memory).resource_declaration()

    assert resources.n_actions == 3
    assert resources.owned_trainable_float32_scalars == 0
    assert resources.owned_persistent_state_bytes == 0
    assert resources.external_memory_persistent_state_bytes == memory.persistent_bytes
    assert resources.memory_queries_per_proposal == 1
    assert resources.random_draws_per_proposal == 0
    assert resources.score_mass_values_interpreted_per_proposal == 3
    assert resources.hard_safety_values_interpreted_per_proposal == 3
    assert resources.argmax_candidates_per_proposal == 3
    assert resources.to_config() == {
        "n_actions": 3,
        "owned_trainable_float32_scalars": 0,
        "owned_persistent_state_bytes": 0,
        "external_memory_persistent_state_bytes": memory.persistent_bytes,
        "memory_queries_per_proposal": 1,
        "random_draws_per_proposal": 0,
        "score_mass_values_interpreted_per_proposal": 3,
        "hard_safety_values_interpreted_per_proposal": 3,
        "argmax_candidates_per_proposal": 3,
    }


def test_empty_memory_abstains_and_api_does_not_accept_forged_retrieval() -> None:
    policy = ExperientialMemoryPolicy(ExperientialMemory(_memory_config()))
    proposal = _propose(policy, policy.memory.init())

    assert not bool(proposal.available)
    assert int(proposal.action) == -1
    assert not bool(proposal.retrieval.accepted)
    assert not bool(proposal.action_mass_valid)
    assert not bool(proposal.safe_positive_mass_available)
    np.testing.assert_array_equal(proposal.action_mass, jnp.zeros((3,), dtype=jnp.float32))
    np.testing.assert_array_equal(
        proposal.normalized_action_mass, jnp.zeros((3,), dtype=jnp.float32)
    )
    assert "retrieval" not in inspect.signature(policy.propose).parameters
    with pytest.raises(TypeError):
        policy.propose(retrieval=proposal.retrieval)


def test_public_state_valid_is_exact_and_jit_safe() -> None:
    memory = ExperientialMemory(_memory_config())
    policy = ExperientialMemoryPolicy(memory)
    state = _write(memory, memory.init(), _entry(5))

    assert bool(policy.state_valid(state))
    assert bool(jax.jit(policy.state_valid)(state))
    corrupted = replace(
        state,
        entries=replace(
            state.entries,
            actions=state.entries.actions.at[0, 0].set(jnp.nan),
        ),
    )
    assert not bool(policy.state_valid(corrupted))
    assert not bool(jax.jit(policy.state_valid)(corrupted))
    with pytest.raises(TypeError):
        policy.state_valid(object())  # type: ignore[arg-type]


def test_valid_retrieval_is_categorical_mass_with_separate_diagnostics() -> None:
    memory = ExperientialMemory(_memory_config())
    policy = ExperientialMemoryPolicy(memory)
    state = _write(memory, memory.init(), _entry(10, action_mass=(1.0, 3.0, 2.0)))
    before = jax.tree.map(lambda value: np.asarray(value).copy(), state)

    proposal = _propose(policy, state)

    assert bool(proposal.available)
    assert int(proposal.action) == 1
    np.testing.assert_array_equal(
        proposal.action_mass, jnp.asarray([1.0, 3.0, 2.0], dtype=jnp.float32)
    )
    np.testing.assert_allclose(
        proposal.normalized_action_mass,
        jnp.asarray([1.0 / 6.0, 0.5, 1.0 / 3.0], dtype=jnp.float32),
        rtol=1e-6,
    )
    assert float(proposal.total_action_mass) == pytest.approx(6.0)
    assert float(proposal.selected_action_mass) == pytest.approx(3.0)
    assert float(proposal.selected_normalized_action_mass) == pytest.approx(0.5)
    assert float(proposal.effective_reliability) > 0.0
    assert int(proposal.retrieval.neighbor_provenance_ids[0]) == 10
    assert bool(proposal.retrieval.neighbor_mask[0])
    _assert_trees_equal(state, before)


def test_policy_step_matches_memory_step_and_records_each_access() -> None:
    """The stateful policy path must query, account, age, and write exactly once."""
    memory = ExperientialMemory(
        _memory_config(
            capacity=4,
            top_k=2,
            max_age=100,
            eviction_utility_weight=0.0,
            eviction_recency_weight=1.0,
        )
    )
    policy = ExperientialMemoryPolicy(memory)
    policy_state = memory.init()
    direct_state = memory.init()
    key = jnp.zeros((2,), dtype=jnp.float32)
    version = jnp.asarray(1, dtype=jnp.int32)
    uncertainty = jnp.asarray(0.1, dtype=jnp.float32)
    uncertainty_available = jnp.asarray(True, dtype=jnp.bool_)
    safety_mask = jnp.ones((3,), dtype=jnp.bool_)

    probe_entry = _entry(99)
    eager_probe = policy.propose_and_step(
        policy_state,
        key,
        version,
        uncertainty,
        uncertainty_available,
        safety_mask,
        probe_entry,
    )
    compiled_probe = jax.jit(policy.propose_and_step)(
        policy_state,
        key,
        version,
        uncertainty,
        uncertainty_available,
        safety_mask,
        probe_entry,
    )
    _assert_trees_equal(eager_probe, compiled_probe)

    for index in range(12):
        entry = _entry(100 + index)
        proposal, policy_step = policy.propose_and_step(
            policy_state,
            key,
            version,
            uncertainty,
            uncertainty_available,
            safety_mask,
            entry,
        )
        direct_step = memory.step(
            direct_state,
            key,
            version,
            uncertainty,
            uncertainty_available,
            entry,
        )
        _assert_trees_equal(proposal.retrieval, direct_step.retrieval)
        _assert_trees_equal(policy_step, direct_step)
        policy_state = policy_step.state
        direct_state = direct_step.state

    accounting = memory.accounting(policy_state)
    assert int(accounting.queries) == 12
    assert int(accounting.accepted_queries) == 11
    assert int(accounting.writes) == 12
    np.testing.assert_array_equal(
        policy_state.entries.provenance_ids,
        direct_state.entries.provenance_ids,
    )
    np.testing.assert_array_equal(
        policy_state.entries.recency_ages,
        direct_state.entries.recency_ages,
    )
    np.testing.assert_array_equal(
        policy_state.entries.retrieval_counts,
        direct_state.entries.retrieval_counts,
    )


def test_finite_action_mass_sum_cannot_overflow_into_false_abstention() -> None:
    memory = ExperientialMemory(_memory_config())
    policy = ExperientialMemoryPolicy(memory)
    maximum = np.finfo(np.float32).max
    state = _write(
        memory,
        memory.init(),
        _entry(11, action_mass=(maximum, maximum, maximum)),
    )

    proposal = _propose(policy, state)

    assert bool(proposal.retrieval.accepted)
    assert bool(proposal.action_mass_valid)
    assert bool(proposal.available)
    assert int(proposal.action) == 0
    assert float(proposal.total_action_mass) == maximum
    np.testing.assert_allclose(
        proposal.normalized_action_mass,
        jnp.full((3,), 1.0 / 3.0, dtype=jnp.float32),
        rtol=1e-6,
    )


def test_large_finite_action_mass_total_never_overflows_on_rounding() -> None:
    """Regression for a reviewer-found rounding edge: reconstructing the
    saturated total via `_FLOAT32_MAX / scaled_total * scaled_total` can
    itself round upward past `_FLOAT32_MAX` on the multiply, so the fix must
    be provably finite (a hard `jnp.minimum` clamp) rather than relying on
    the division alone."""
    memory = ExperientialMemory(_memory_config())
    policy = ExperientialMemoryPolicy(memory)
    rng = np.random.default_rng(1234)
    float32_max = np.finfo(np.float32).max

    for _ in range(200):
        action_mass = tuple(
            float(v)
            for v in (rng.uniform(0.0, 1.0, size=3).astype(np.float32) * float32_max)
        )
        state = _write(
            memory,
            memory.init(),
            _entry(11, action_mass=action_mass),
        )
        proposal = _propose(policy, state)

        assert bool(jnp.isfinite(proposal.total_action_mass)), action_mass
        with np.errstate(over="ignore"):
            direct_sum = float(np.sum(np.asarray(action_mass, dtype=np.float32)))
        if np.isfinite(direct_sum):
            assert float(proposal.total_action_mass) == direct_sum, action_mass


def test_hard_safety_mask_and_lowest_index_tie_break_are_exact() -> None:
    memory = ExperientialMemory(_memory_config())
    policy = ExperientialMemoryPolicy(memory)
    state = _write(memory, memory.init(), _entry(11, action_mass=(2.0, 2.0, 1.0)))

    tied = _propose(policy, state)
    assert bool(tied.available)
    assert int(tied.action) == 0

    masked = _propose(policy, state, safety_mask=(False, True, True))
    assert bool(masked.available)
    assert int(masked.action) == 1
    np.testing.assert_array_equal(
        masked.hard_safety_mask, jnp.asarray([False, True, True], dtype=jnp.bool_)
    )

    no_safe_mass = _propose(policy, state, safety_mask=(False, False, True))
    assert bool(no_safe_mass.available)
    assert int(no_safe_mass.action) == 2
    none_safe = _propose(policy, state, safety_mask=(False, False, False))
    assert not bool(none_safe.available)
    assert int(none_safe.action) == -1
    assert float(none_safe.selected_action_mass) == 0.0

    zero_safe_state = _write(
        memory, memory.init(), _entry(12, action_mass=(2.0, 0.0, 0.0))
    )
    zero_safe_mass = _propose(
        policy, zero_safe_state, safety_mask=(False, True, True)
    )
    assert bool(zero_safe_mass.action_mass_valid)
    assert not bool(zero_safe_mass.safe_positive_mass_available)
    assert not bool(zero_safe_mass.available)
    assert int(zero_safe_mass.action) == -1


def test_neighbor_vectors_are_averaged_as_mass_not_integer_action_ids() -> None:
    memory = ExperientialMemory(_memory_config(top_k=2))
    policy = ExperientialMemoryPolicy(memory)
    state = _write(memory, memory.init(), _entry(20, action_mass=(1.0, 0.0, 0.0)))
    state = _write(memory, state, _entry(21, action_mass=(0.0, 1.0, 0.0)))

    proposal = _propose(policy, state)

    assert bool(proposal.available)
    assert int(proposal.action) == 1
    # Ages make the newer action-1 exemplar slightly heavier; the proposal
    # applies argmax to the three mass channels and never averages IDs 0 and 1.
    assert float(proposal.action_mass[1]) > float(proposal.action_mass[0])
    assert float(proposal.action_mass[2]) == 0.0


@pytest.mark.parametrize(
    ("state_entry", "query", "diagnostic"),
    [
        (_entry(30, representation_version=1), {"version": 2}, "version_compatible"),
        (_entry(31, age=4), {}, "freshness_ok"),
        (
            _entry(32, uncertainty=0.1),
            {"uncertainty": 0.0, "uncertainty_available": False},
            "uncertainty_available",
        ),
        (
            _entry(33, uncertainty=0.1),
            {"uncertainty": 1.0},
            "uncertainty_ok",
        ),
        (_entry(34, safety_cost=1.0), {}, "safety_ok"),
    ],
)
def test_memory_version_staleness_uncertainty_and_safety_gates_are_inherited(
    state_entry: ExperientialMemoryEntry,
    query: dict[str, object],
    diagnostic: str,
) -> None:
    memory = ExperientialMemory(_memory_config())
    policy = ExperientialMemoryPolicy(memory)
    state = _write(memory, memory.init(), state_entry)

    proposal = _propose(policy, state, **query)  # type: ignore[arg-type]

    assert not bool(proposal.available)
    assert int(proposal.action) == -1
    assert not bool(getattr(proposal.retrieval, diagnostic))
    assert float(proposal.effective_reliability) == 0.0


@pytest.mark.parametrize(
    "action_mass",
    [(-1.0, 2.0, 1.0), (0.0, 0.0, 0.0)],
)
def test_invalid_or_zero_retrieved_action_mass_fails_closed(
    action_mass: tuple[float, float, float],
) -> None:
    memory = ExperientialMemory(_memory_config())
    policy = ExperientialMemoryPolicy(memory)
    state = _write(memory, memory.init(), _entry(40, action_mass=action_mass))

    proposal = _propose(policy, state)

    assert bool(proposal.retrieval.accepted)
    assert not bool(proposal.action_mass_valid)
    assert not bool(proposal.safe_positive_mass_available)
    assert not bool(proposal.available)
    assert int(proposal.action) == -1
    np.testing.assert_array_equal(
        proposal.normalized_action_mass, jnp.zeros((3,), dtype=jnp.float32)
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"key": (float("nan"), 0.0)},
        {"version": -1},
        {"uncertainty": float("nan")},
        {"uncertainty": -0.1},
    ],
)
def test_dynamic_query_corruption_fails_closed(overrides: dict[str, object]) -> None:
    memory = ExperientialMemory(_memory_config())
    policy = ExperientialMemoryPolicy(memory)
    state = _write(memory, memory.init(), _entry(45))

    proposal = _propose(policy, state, **overrides)  # type: ignore[arg-type]

    assert bool(proposal.retrieval.state_valid)
    assert not bool(proposal.retrieval.query_valid)
    assert not bool(proposal.retrieval.accepted)
    assert not bool(proposal.available)
    assert int(proposal.action) == -1


def test_dynamic_state_corruption_fails_closed_without_mutation() -> None:
    memory = ExperientialMemory(_memory_config())
    policy = ExperientialMemoryPolicy(memory)
    state = _write(memory, memory.init(), _entry(50))
    corrupted_entries = replace(
        state.entries,
        actions=state.entries.actions.at[0, 0].set(jnp.nan),
    )
    corrupted = replace(state, entries=corrupted_entries)
    before = jax.tree.map(lambda value: np.asarray(value).copy(), corrupted)

    proposal = _propose(policy, corrupted)

    assert not bool(proposal.retrieval.state_valid)
    assert not bool(proposal.available)
    assert int(proposal.action) == -1
    _assert_trees_equal(corrupted, before)


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        ("hard_safety_mask", jnp.ones((2,), dtype=jnp.bool_), ValueError),
        ("hard_safety_mask", jnp.ones((3,), dtype=jnp.int32), TypeError),
        ("query_key", jnp.ones((1, 2), dtype=jnp.float32), ValueError),
        ("query_key", jnp.ones((2,), dtype=jnp.float16), TypeError),
        ("representation_version", jnp.asarray(1, dtype=jnp.float32), TypeError),
        ("query_uncertainty", jnp.asarray(0, dtype=jnp.int32), TypeError),
        ("query_uncertainty_available", jnp.asarray(1, dtype=jnp.int32), TypeError),
    ],
)
def test_malformed_static_inputs_are_rejected_before_tracing(
    argument: str,
    value: jax.Array,
    error: type[Exception],
) -> None:
    memory = ExperientialMemory(_memory_config())
    policy = ExperientialMemoryPolicy(memory)
    kwargs = {
        "state": memory.init(),
        "query_key": jnp.zeros((2,), dtype=jnp.float32),
        "representation_version": jnp.asarray(1, dtype=jnp.int32),
        "query_uncertainty": jnp.asarray(0.1, dtype=jnp.float32),
        "query_uncertainty_available": jnp.asarray(True, dtype=jnp.bool_),
        "hard_safety_mask": jnp.ones((3,), dtype=jnp.bool_),
    }
    kwargs[argument] = value
    with pytest.raises(error):
        policy.propose(**kwargs)  # type: ignore[arg-type]


def test_eager_jit_and_scan_are_exact_rng_free_and_nonmutating() -> None:
    memory = ExperientialMemory(_memory_config())
    policy = ExperientialMemoryPolicy(memory)
    state = _write(memory, memory.init(), _entry(60, action_mass=(1.0, 4.0, 2.0)))
    key = jnp.zeros((2,), dtype=jnp.float32)
    version = jnp.asarray(1, dtype=jnp.int32)
    uncertainty = jnp.asarray(0.1, dtype=jnp.float32)
    available = jnp.asarray(True, dtype=jnp.bool_)
    mask = jnp.asarray([True, True, True], dtype=jnp.bool_)
    before = jax.tree.map(lambda value: np.asarray(value).copy(), state)

    eager = policy.propose(state, key, version, uncertainty, available, mask)
    repeated = policy.propose(state, key, version, uncertainty, available, mask)
    compiled = jax.jit(policy.propose)(
        state, key, version, uncertainty, available, mask
    )
    _assert_trees_equal(eager, repeated)
    _assert_trees_equal(eager, compiled)

    masks = jnp.asarray(
        [[True, True, True], [True, False, True], [False, False, False]],
        dtype=jnp.bool_,
    )

    def body(
        current: ExperientialMemoryState,
        current_mask: jax.Array,
    ) -> tuple[ExperientialMemoryState, ExperientialMemoryPolicyProposal]:
        proposal = policy.propose(
            current, key, version, uncertainty, available, current_mask
        )
        return current, proposal

    eager_final, eager_outputs = jax.lax.scan(body, state, masks)
    compiled_final, compiled_outputs = jax.jit(
        lambda initial: jax.lax.scan(body, initial, masks)
    )(state)
    _assert_trees_equal(eager_final, state)
    _assert_trees_equal(eager_final, compiled_final)
    _assert_trees_equal(eager_outputs, compiled_outputs)
    np.testing.assert_array_equal(
        eager_outputs.action, jnp.asarray([1, 2, -1], dtype=jnp.int32)
    )
    _assert_trees_equal(state, before)
