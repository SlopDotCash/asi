"""Focused validation for Pavlovian stream (hostile + resource)."""

from __future__ import annotations

from fractions import Fraction

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams.pavlovian import (
    ClassicalConditioningStream,
    PavlovianPhase,
    PavlovianState,
    _require_pavlovian_resources,
)

_INT32_MAX = 2**31 - 1


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr executed")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:  # type: ignore[override]
        type(self).calls += 1
        raise RuntimeError("ratio hook")


class _ClassSpoof:
    @property  # type: ignore[misc]
    def __class__(self) -> type:  # type: ignore[no-untyped-def]
        return float

    def __float__(self) -> float:  # pragma: no cover
        return 0.1


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook must not run")


def _phase(**overrides: object) -> PavlovianPhase:
    base: dict[str, object] = {
        "name": "acq",
        "n_steps": 10,
        "cs_us_contingency": 1.0,
        "cs_active": (0,),
    }
    base.update(overrides)
    return PavlovianPhase(**base)  # type: ignore[arg-type]


def _stream(**overrides: object) -> ClassicalConditioningStream:
    phases = overrides.pop("phases", (_phase(),))
    return ClassicalConditioningStream(phases=phases, **overrides)  # type: ignore[arg-type]


def test_pavlovian_int_validators_reject_hostile_without_hook() -> None:
    for field, ctor in [
        ("n_cs", lambda v: _stream(n_cs=v)),
        ("n_distractors", lambda v: _stream(n_distractors=v)),
        ("cs_us_delay", lambda v: _stream(cs_us_delay=v)),
        ("cs_duration", lambda v: _stream(cs_duration=v)),
    ]:
        with pytest.raises(ValueError, match=field):
            ctor(_HostileInt(2))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=field):
            ctor(_HostileInt(2))  # type: ignore[arg-type]


def test_pavlovian_int_validators_do_not_run_repr() -> None:
    for ctor in [
        lambda v: _stream(n_cs=v),
        lambda v: _stream(cs_us_delay=v),
    ]:
        with pytest.raises(ValueError):
            ctor(_RaisingRepr())  # type: ignore[arg-type]


def test_pavlovian_int_validators_reject_bool_and_numpy() -> None:
    for field in ("n_cs", "cs_us_delay", "cs_duration"):
        for bad in (True, np.bool_(True), np.int64(2), 1.0, "2"):
            with pytest.raises(ValueError, match=field):
                _stream(**{field: bad})  # type: ignore[arg-type]


def test_pavlovian_float_validators_reject_hostile_ratio() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="noise_std"):
        _stream(noise_std=_HostileFloat(0.1))  # type: ignore[arg-type]
    # Hostile as_integer_ratio must not be invoked (ratio check is guarded)
    assert _HostileFloat.calls == 0


def test_pavlovian_float_validators_reject_spoof_and_nan() -> None:
    for field, bad in [
        ("noise_std", float("nan")),
        ("noise_std", float("inf")),
        ("noise_std", -0.1),
        ("noise_std", _ClassSpoof()),  # type: ignore[arg-type]
        ("distractor_prob", float("nan")),
        ("distractor_prob", 1.5),
        ("distractor_prob", -0.1),
        ("distractor_prob", _ClassSpoof()),  # type: ignore[arg-type]
    ]:
        with pytest.raises(ValueError, match=field):
            _stream(**{field: bad})  # type: ignore[arg-type]


def test_pavlovian_phase_contingency_hostile_is_suppressed() -> None:
    with pytest.raises(ValueError, match="cs_us_contingency"):
        _stream(phases=(_phase(cs_us_contingency=_HostileFloat(0.5)),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cs_us_contingency"):
        _stream(phases=(_phase(cs_us_contingency=_ClassSpoof()),))  # type: ignore[arg-type]


def test_pavlovian_phase_name_requires_exact_str() -> None:
    with pytest.raises(ValueError, match="phase name"):
        _stream(phases=(_phase(name=_HostileInt(1)),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="phase name"):
        _stream(phases=(_phase(name=b"acq"),))  # type: ignore[arg-type]


def test_pavlovian_cs_active_hostile_and_range() -> None:
    # hostile int subclass must be rejected without running index hook
    with pytest.raises(ValueError, match="cs_active"):
        _stream(phases=(_phase(cs_active=(_HostileInt(0),)),))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _stream(phases=(_phase(cs_active=(_RaisingRepr(),)),))  # type: ignore[arg-type]
    # out of range
    with pytest.raises(ValueError, match="out of range"):
        _stream(phases=(_phase(cs_active=(5,)),), n_cs=2)
    # compound out of range
    with pytest.raises(ValueError, match="compound_index"):
        _stream(phases=(_phase(compound_index=5),), n_cs=2)
    # tuple type check
    with pytest.raises(ValueError, match="cs_active"):
        _stream(phases=(_phase(cs_active=[0]),))  # type: ignore[arg-type]


def test_pavlovian_resource_boundaries_include_static_arrays_state_and_key() -> None:
    _require_pavlovian_resources(n_phases=1, n_cs=1, n_distractors=0)
    with pytest.raises(ValueError, match="byte count must fit signed int32"):
        _require_pavlovian_resources(
            n_phases=1,
            n_cs=_INT32_MAX // 4,
            n_distractors=0,
        )


def test_pavlovian_resource_preflight_without_allocation() -> None:
    # feature_dim = n_cs + n_distractors, mask = n_phases * n_cs
    # small case passes
    s = _stream(n_cs=2, n_distractors=1, phases=(_phase(), _phase(name="ext")))
    assert s.feature_dim == 3
    _require_pavlovian_resources(n_phases=2, n_cs=2, n_distractors=1)
    with pytest.raises(ValueError, match="scalar count"):
        _require_pavlovian_resources(
            n_phases=_INT32_MAX,
            n_cs=_INT32_MAX,
            n_distractors=0,
        )


def test_pavlovian_phase_container_and_values_are_canonicalized() -> None:
    with pytest.raises(ValueError, match="exact tuple"):
        _stream(phases=[_phase()])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact PavlovianPhase"):
        _stream(phases=({"name": "bad"},))  # type: ignore[arg-type]

    stream = _stream(phases=(_phase(cs_us_contingency=Fraction(1, 2)),))
    assert type(stream.phases[0].cs_us_contingency) is float


def test_pavlovian_probabilities_reject_exact_host_values_above_one() -> None:
    above_one = Fraction(1, 1) + Fraction(1, 2**100)
    with pytest.raises(ValueError, match="distractor_prob"):
        _stream(distractor_prob=above_one)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cs_us_contingency"):
        _stream(phases=(_phase(cs_us_contingency=above_one),))


def test_pavlovian_invalid_dynamic_state_is_an_atomic_neutral_noop_under_jit() -> None:
    stream = _stream(n_cs=2, n_distractors=1)
    valid = stream.init(jr.key(3))
    invalid = valid.replace(phase_idx=jnp.asarray(-1, dtype=jnp.int32))
    timestep, returned = jax.jit(stream.step)(invalid, jnp.asarray(0, dtype=jnp.int32))
    chex.assert_trees_all_equal(returned, invalid)
    chex.assert_trees_all_equal(
        timestep.observation,
        jnp.zeros((stream.feature_dim,), dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(timestep.target, jnp.zeros((1,), dtype=jnp.float32))


def test_pavlovian_static_state_contract_fails_before_indexing() -> None:
    stream = _stream(n_cs=2)
    state = stream.init(jr.key(5))
    invalid = PavlovianState(
        key=state.key,
        cs_active_steps_remaining=jnp.zeros((1,), dtype=jnp.int32),
        us_pending_steps_remaining=state.us_pending_steps_remaining,
        phase_idx=state.phase_idx,
        step_in_phase=state.step_in_phase,
        n_distractor_active=state.n_distractor_active,
        iti_steps_remaining=state.iti_steps_remaining,
    )
    with pytest.raises(ValueError, match="cs_active_steps_remaining"):
        stream.step(invalid, jnp.asarray(0, dtype=jnp.int32))


def test_pavlovian_counters_saturate_and_compound_does_not_double_activate() -> None:
    stream = _stream(
        phases=(_phase(cs_active=(0,), compound_index=0),),
        cs_duration=_INT32_MAX,
        iti_min=0,
        iti_max=0,
        noise_std=0.0,
    )
    state = stream.init(jr.key(0)).replace(
        iti_steps_remaining=jnp.asarray(0, dtype=jnp.int32),
        step_in_phase=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    _, next_state = stream.step(state, jnp.asarray(0, dtype=jnp.int32))
    assert int(next_state.cs_active_steps_remaining[0]) == _INT32_MAX
    assert int(next_state.step_in_phase) == _INT32_MAX


def test_pavlovian_valid_construction_and_jit() -> None:
    stream = _stream(n_cs=2, n_distractors=1, noise_std=0.01, distractor_prob=0.1)
    state = stream.init(jr.key(0))
    ts, _ = stream.step(state, jnp.array(0))
    assert ts.observation.shape == (3,)
    assert ts.target.shape == (1,)
