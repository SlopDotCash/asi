"""Trust-boundary validation for streams/pavlovian sanitized errors."""

from __future__ import annotations

from typing import Any

import pytest

from alberta_framework.streams.pavlovian import (
    ClassicalConditioningStream,
    PavlovianPhase,
)


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")

    def __hash__(self) -> int:  # type: ignore[override]
        raise AssertionError("EvilStr.__hash__ must not be called")


class _StringSubclass(str):
    pass


def _phase(**overrides: Any) -> PavlovianPhase:
    base: dict[str, Any] = {
        "name": "acquisition",
        "n_steps": 10,
        "cs_us_contingency": 1.0,
        "cs_active": (0,),
        "compound_index": -1,
    }
    base.update(overrides)
    return PavlovianPhase(**base)  # type: ignore[arg-type]


def _raw_phase(**overrides: Any) -> PavlovianPhase:
    obj = object.__new__(PavlovianPhase)
    object.__setattr__(obj, "name", overrides.get("name", "raw"))
    object.__setattr__(obj, "n_steps", overrides.get("n_steps", 10))
    object.__setattr__(
        obj, "cs_us_contingency", overrides.get("cs_us_contingency", 1.0)
    )
    object.__setattr__(obj, "cs_active", overrides.get("cs_active", (0,)))
    object.__setattr__(obj, "compound_index", overrides.get("compound_index", -1))
    return obj


def _stream(**overrides: Any) -> ClassicalConditioningStream:
    phases = overrides.pop("phases", (_phase(),))
    return ClassicalConditioningStream(phases=phases, **overrides)  # type: ignore[arg-type]


def test_phase_name_rejects_subclass_without_repr() -> None:
    with pytest.raises(ValueError, match="phase name"):
        _phase(name=_StringSubclass("acq"))  # type: ignore[arg-type]


def test_phase_name_hostile_without_repr_leak() -> None:
    evil = _EvilStr("acq")
    with pytest.raises(ValueError, match="phase name") as exc:
        _phase(name=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_stream_contingency_error_sanitized() -> None:
    bad = _raw_phase(name="evil_phase", cs_us_contingency=2.0)
    with pytest.raises(ValueError, match="cs_us_contingency") as exc:
        _stream(phases=(bad,), n_cs=2)
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_phase'" in msg


def test_stream_cs_active_type_error_sanitized() -> None:
    raw = _raw_phase(name="type_phase", cs_active=(0.5,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="built-in integer") as exc:
        _stream(phases=(raw,), n_cs=2)
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'type_phase'" in msg


def test_stream_cs_active_range_error_sanitized() -> None:
    bad = _phase(name="range_phase", cs_active=(5,))
    with pytest.raises(ValueError, match="out of range") as exc:
        _stream(phases=(bad,), n_cs=2)
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'range_phase'" in msg


def test_stream_compound_index_error_sanitized() -> None:
    bad = _phase(name="compound_phase", compound_index=5)
    with pytest.raises(ValueError, match="compound_index") as exc:
        _stream(phases=(bad,), n_cs=2)
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'compound_phase'" in msg


def test_stream_rejects_hostile_phase_name_subclass() -> None:
    evil_sub = _StringSubclass("evil_sub")
    with pytest.raises(ValueError, match="phase name"):
        _phase(name=evil_sub)  # type: ignore[arg-type]


def test_valid_stream_still_passes() -> None:
    s = _stream(phases=(_phase(name="ok"),), n_cs=2)
    assert s.phases[0].name == "ok"
