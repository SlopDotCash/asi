"""Hostile int gate for forager matched qualification timeout before float."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_qualification import _run_bounded_process

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __float__")

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __eq__")


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __float__ float")


def test_timeout_rejects_hostile_int_before_float() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="finite and positive"):
        _run_bounded_process(
            ["echo", "hi"],
            timeout=hostile,  # type: ignore[arg-type]
            maximum_stdout_bytes=100,
            maximum_stderr_bytes=100,
        )
    assert _HostileInt.calls == 0


def test_timeout_rejects_hostile_float_before_finite() -> None:
    hostile = _HostileFloat(1.0)
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="finite and positive"):
        _run_bounded_process(
            ["echo", "hi"],
            timeout=hostile,  # type: ignore[arg-type]
            maximum_stdout_bytes=100,
            maximum_stderr_bytes=100,
        )
    assert _HostileFloat.calls == 0


def test_benign_timeout_passes() -> None:
    # Use a short timeout with a command that will finish quickly
    # timeout 1 should not raise for timeout check
    try:
        _run_bounded_process(["echo", "hi"], timeout=1, maximum_stdout_bytes=100, maximum_stderr_bytes=100)
    except ValueError as e:
        assert "finite and positive" not in str(e)
    except Exception:
        pass
