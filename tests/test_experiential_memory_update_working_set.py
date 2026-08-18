"""#1383-complete update working-set preflight for experiential memory."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.experiential_memory import (
    ExperientialMemory,
    ExperientialMemoryConfig,
    ExperientialMemoryEntry,
    _experiential_persistent_bytes,
    _experiential_update_working_set_bytes,
    _preflight_experiential_update_working_set,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_CAPACITY = 11_200_000
_LAST_FIT_CAPACITY = 11_184_809
_FIRST_OVERFLOW_CAPACITY = 11_184_810
_UNIT = {
    "observation_dim": 1,
    "key_dim": 1,
    "action_dim": 1,
    "outcome_dim": 1,
    "top_k": 1,
}


def _unit_persist_bytes(capacity: int) -> int:
    return _experiential_persistent_bytes(
        capacity=capacity,
        observation_dim=1,
        key_dim=1,
        action_dim=1,
        outcome_dim=1,
    )


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _unit_persist_bytes(_OVERFLOW_CAPACITY)
    working_set_bytes = _experiential_update_working_set_bytes(
        capacity=_OVERFLOW_CAPACITY,
        **_UNIT,
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 716_800_032
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_CAPACITY <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        ExperientialMemoryConfig(capacity=_OVERFLOW_CAPACITY, **_UNIT)


def test_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for capacity in range(_LAST_FIT_CAPACITY, _FIRST_OVERFLOW_CAPACITY + 2):
        persist_bytes = _unit_persist_bytes(capacity)
        working_set_bytes = _experiential_update_working_set_bytes(
            capacity=capacity,
            **_UNIT,
        )
        extras_bytes = working_set_bytes - 3 * persist_bytes
        assert persist_bytes <= _INT32_MAX
        assert persist_bytes + extras_bytes <= _INT32_MAX
        assert 4 * capacity <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = capacity
        elif first_overflow is None:
            first_overflow = capacity
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _LAST_FIT_CAPACITY
    config = ExperientialMemoryConfig(capacity=last_fit, **_UNIT)
    assert config.capacity == last_fit
    with pytest.raises(ValueError, match="update working set byte count"):
        ExperientialMemoryConfig(capacity=first_overflow, **_UNIT)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    overflow = ExperientialMemoryConfig(capacity=4, **_UNIT)
    object.__setattr__(overflow, "capacity", _OVERFLOW_CAPACITY)
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_experiential_update_working_set(overflow)


def test_query_envelope_still_fires_before_working_set_when_it_overflows_first() -> None:
    with pytest.raises(ValueError, match="aggregate query working-set bytes"):
        ExperientialMemoryConfig(capacity=12_000_000, **_UNIT)


def test_persist_bound_still_fires_before_working_set() -> None:
    last_legal = (2**31 - 1 - 32) // 64
    with pytest.raises(ValueError, match="state byte count"):
        ExperientialMemoryConfig(capacity=last_legal + 1, **_UNIT)


def test_existing_query_last_legal_capacity_still_constructs() -> None:
    persist_bytes = _unit_persist_bytes(11_000_000)
    working_set_bytes = _experiential_update_working_set_bytes(
        capacity=11_000_000,
        **_UNIT,
    )
    assert persist_bytes == 704_000_032
    assert working_set_bytes <= _INT32_MAX
    config = ExperientialMemoryConfig(capacity=11_000_000, **_UNIT)
    assert config.capacity == 11_000_000


def test_legal_small_experiential_memory_still_steps() -> None:
    persist_bytes = _experiential_persistent_bytes(
        capacity=4,
        observation_dim=1,
        key_dim=1,
        action_dim=1,
        outcome_dim=1,
    )
    assert persist_bytes == 288
    config = ExperientialMemoryConfig(capacity=4, **_UNIT)
    memory = ExperientialMemory(config)
    state = memory.init()
    entry = ExperientialMemoryEntry(
        observation=jnp.zeros((1,), dtype=jnp.float32),
        key=jnp.zeros((1,), dtype=jnp.float32),
        action=jnp.zeros((1,), dtype=jnp.float32),
        outcome=jnp.zeros((1,), dtype=jnp.float32),
        reward=jnp.asarray(0.0, dtype=jnp.float32),
        uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
        uncertainty_available=jnp.asarray(True),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True),
        reliability=jnp.asarray(1.0, dtype=jnp.float32),
        utility=jnp.asarray(0.0, dtype=jnp.float32),
        utility_available=jnp.asarray(True),
        representation_version=jnp.asarray(1, dtype=jnp.int32),
        valid=jnp.asarray(True),
        age=jnp.asarray(0, dtype=jnp.int32),
        provenance_id=jnp.asarray(1, dtype=jnp.int32),
        source_id=jnp.asarray(1, dtype=jnp.int32),
    )
    memory.step(
        state,
        entry.key,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(True),
        entry,
    )
