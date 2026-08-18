"""#1383-complete step working-set preflight for Pavlovian streams."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.streams.pavlovian import (
    ClassicalConditioningStream,
    PavlovianPhase,
    _pavlovian_persistent_bytes,
    _pavlovian_step_working_set_bytes,
    _preflight_pavlovian_step_working_set,
    _require_pavlovian_resources,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_N_CS = 80_000_000
_LAST_FIT_N_CS = 76_695_840
_FIRST_OVERFLOW_N_CS = 76_695_841


def _unit_phase() -> PavlovianPhase:
    return PavlovianPhase(
        name="acq",
        n_steps=10,
        cs_us_contingency=1.0,
        cs_active=(0,),
    )


def _unit_stream(n_cs: int, n_distractors: int = 0) -> ClassicalConditioningStream:
    return ClassicalConditioningStream(
        phases=(_unit_phase(),),
        n_cs=n_cs,
        n_distractors=n_distractors,
    )


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _pavlovian_persistent_bytes(
        n_phases=1, n_cs=_OVERFLOW_N_CS, n_distractors=0
    )
    working_set_bytes = _pavlovian_step_working_set_bytes(
        n_phases=1, n_cs=_OVERFLOW_N_CS, n_distractors=0
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 640_000_040
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_N_CS <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="working set byte count"):
        _unit_stream(_OVERFLOW_N_CS)


def test_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for n_cs in range(_LAST_FIT_N_CS, _FIRST_OVERFLOW_N_CS + 2):
        persist_bytes = _pavlovian_persistent_bytes(
            n_phases=1, n_cs=n_cs, n_distractors=0
        )
        working_set_bytes = _pavlovian_step_working_set_bytes(
            n_phases=1, n_cs=n_cs, n_distractors=0
        )
        extras_bytes = working_set_bytes - 3 * persist_bytes
        assert persist_bytes <= _INT32_MAX
        assert persist_bytes + extras_bytes <= _INT32_MAX
        assert 4 * n_cs <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = n_cs
        elif first_overflow is None:
            first_overflow = n_cs
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _LAST_FIT_N_CS
    _preflight_pavlovian_step_working_set(
        n_phases=1, n_cs=last_fit, n_distractors=0
    )
    with pytest.raises(ValueError, match="working set byte count"):
        _preflight_pavlovian_step_working_set(
            n_phases=1, n_cs=first_overflow, n_distractors=0
        )
    with pytest.raises(ValueError, match="working set byte count"):
        _unit_stream(first_overflow)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="working set byte count"):
        _preflight_pavlovian_step_working_set(
            n_phases=1, n_cs=_OVERFLOW_N_CS, n_distractors=0
        )


def test_persist_bound_still_fires_before_working_set() -> None:
    _require_pavlovian_resources(n_phases=1, n_cs=1, n_distractors=0)
    with pytest.raises(ValueError, match="byte count must fit signed int32"):
        _unit_stream(_INT32_MAX // 4)


def test_legal_small_pavlovian_still_steps() -> None:
    persist_bytes = _pavlovian_persistent_bytes(n_phases=1, n_cs=3, n_distractors=2)
    assert persist_bytes == 72
    stream = _unit_stream(3, n_distractors=2)
    state = stream.init(jr.key(0))
    stream.step(state, jnp.asarray(0, dtype=jnp.int32))
