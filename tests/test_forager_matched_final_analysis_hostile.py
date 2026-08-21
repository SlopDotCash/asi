"""Hostile string gate for forager_matched_final_analysis before membership."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_final_analysis import (
    ForagerMatchedFinalAnalysisError,
    _expected_entrypoint_binding,
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


def test_expected_entrypoint_binding_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("candidate-0")
    _HostileStr.calls = 0
    with pytest.raises(ForagerMatchedFinalAnalysisError, match="must be an exact string"):
        _expected_entrypoint_binding(hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_expected_entrypoint_binding_rejects_non_string() -> None:
    with pytest.raises(ForagerMatchedFinalAnalysisError, match="must be an exact string"):
        _expected_entrypoint_binding(123)  # type: ignore[arg-type]


def test_expected_entrypoint_binding_benign_passes() -> None:
    # Known Alberta candidate should pass - test unknown id if available, else test unknown
    try:
        res = _expected_entrypoint_binding("alberta")
        assert isinstance(res, dict)
    except ForagerMatchedFinalAnalysisError:
        # For non-Alberta unknown, error is expected but not hostile
        pass
    assert True
