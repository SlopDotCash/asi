"""Hostile int/float identity gates for the forager matched bounded-process
timeout gates before conversion.

The timeout gates in forager_matched_executor and forager_matched_qualification
still use isinstance before a trusted float() conversion; a hostile subclass
passes the gate and its overridden __float__ runs during validation.
"""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_executor import (
    _run_bounded_process as executor_run_bounded_process,
)
from alberta_framework.benchmarks.forager_matched_qualification import (
    _run_bounded_process as qualification_run_bounded_process,
)

pytestmark = pytest.mark.unit


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


class _HostileInt(int):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def _reset() -> None:
    _HostileFloat.calls = 0
    _HostileInt.calls = 0


def _run_gate(timeout: object, module: str) -> None:
    if module == "executor":
        executor_run_bounded_process(
            ("unused",),
            timeout=timeout,  # type: ignore[arg-type]
            maximum_stdout_bytes=1,
            maximum_stderr_bytes=1,
        )
    else:
        qualification_run_bounded_process(
            ("unused",),
            timeout=timeout,  # type: ignore[arg-type]
            maximum_stdout_bytes=1,
            maximum_stderr_bytes=1,
        )


@pytest.mark.parametrize("module", ["executor", "qualification"])
def test_bounded_process_rejects_hostile_float_timeout_before_float(
    module: str,
) -> None:
    _reset()
    with pytest.raises(ValueError, match="bounded process timeout"):
        _run_gate(_HostileFloat(1.0), module)
    assert _HostileFloat.calls == 0


@pytest.mark.parametrize("module", ["executor", "qualification"])
def test_bounded_process_rejects_hostile_int_timeout_before_float(
    module: str,
) -> None:
    _reset()
    with pytest.raises(ValueError, match="bounded process timeout"):
        _run_gate(_HostileInt(1), module)
    assert _HostileInt.calls == 0


@pytest.mark.parametrize("module", ["executor", "qualification"])
def test_bounded_process_rejects_bool_and_accepts_benign(module: str) -> None:
    _reset()
    assert _HostileFloat.calls == 0
    with pytest.raises(ValueError, match="bounded process timeout"):
        _run_gate(True, module)
    with pytest.raises(ValueError, match="bounded process timeout"):
        _run_gate(0, module)
    with pytest.raises(ValueError, match="bounded process timeout"):
        _run_gate(-1.0, module)
    with pytest.raises(ValueError, match="bounded process timeout"):
        _run_gate(float("inf"), module)
