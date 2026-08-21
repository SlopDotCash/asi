"""Hostile string gate for forager container absolute/relative before contains."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks._forager_matched_container import (
    ContainerError,
    _absolute,
    _relative,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __contains__(self, item: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile contains must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_absolute_rejects_hostile_before_path() -> None:
    hostile = _HostileStr("/tmp")
    _HostileStr.calls = 0
    with pytest.raises(ContainerError, match="must be an exact string"):
        _absolute(hostile, label="p")  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_absolute_rejects_non_string() -> None:
    with pytest.raises(ContainerError, match="must be an exact string"):
        _absolute(123, label="p")  # type: ignore[arg-type]


def test_relative_rejects_hostile_before_contains() -> None:
    hostile = _HostileStr("a/b")
    _HostileStr.calls = 0
    with pytest.raises(ContainerError, match="must be an exact string"):
        _relative(hostile, label="p")  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_relative_rejects_non_string() -> None:
    with pytest.raises(ContainerError, match="must be an exact string"):
        _relative(123, label="p")  # type: ignore[arg-type]


def test_benign_absolute_passes() -> None:
    p = _absolute("/tmp", label="p")
    assert str(p) == "/tmp"


def test_benign_relative_passes() -> None:
    p = _relative("a/b", label="p")
    assert str(p) == "a/b"
