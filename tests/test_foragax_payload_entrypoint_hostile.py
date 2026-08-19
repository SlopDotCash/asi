"""Hostile string gate for foragax payload entrypoint before membership."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.foragax_open_screen import (
    ScreenError,
    _validate_payload_components,
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

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_entrypoint_rejects_hostile_before_membership() -> None:
    hostile = _HostileStr("src/continuing_main.py")
    _HostileStr.calls = 0
    with pytest.raises(ScreenError, match="entrypoint"):
        _validate_payload_components(
            payload=Path("/tmp"),
            seeds=[0],
            horizon=1,
            entrypoint=hostile,  # type: ignore[arg-type]
            result_root="results/foo",
            metadata_contract={},
        )
    assert _HostileStr.calls == 0


def test_entrypoint_rejects_unknown_benign() -> None:
    with pytest.raises(ScreenError, match="entrypoint"):
        _validate_payload_components(
            payload=Path("/tmp"),
            seeds=[0],
            horizon=1,
            entrypoint="unknown.py",
            result_root="results/foo",
            metadata_contract={},
        )


def test_non_string_rejects_before_membership() -> None:
    with pytest.raises(ScreenError, match="entrypoint"):
        _validate_payload_components(
            payload=Path("/tmp"),
            seeds=[0],
            horizon=1,
            entrypoint=123,  # type: ignore[arg-type]
            result_root="results/foo",
            metadata_contract={},
        )


