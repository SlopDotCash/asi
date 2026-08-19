"""Hostile int gate for continual_ia _integer before returning the value."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.continual_ia_artifact import _integer

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")


def test_integer_rejects_hostile_int_identity() -> None:
    hostile = _HostileInt(9)
    _HostileInt.calls = 0
    errors: list[str] = []
    result = _integer(hostile, location="field", errors=errors)  # type: ignore[arg-type]
    assert result is None
    assert errors == ["field must be an integer"]
    assert _HostileInt.calls == 0


def test_integer_rejects_bool() -> None:
    errors: list[str] = []
    assert _integer(True, location="field", errors=errors) is None
    assert errors == ["field must be an integer"]


def test_integer_benign_still_works() -> None:
    errors: list[str] = []
    assert _integer(5, location="field", errors=errors) == 5
    assert errors == []
    errors = []
    assert _integer("bad", location="field", errors=errors) is None  # type: ignore[arg-type]
    assert errors == ["field must be an integer"]
