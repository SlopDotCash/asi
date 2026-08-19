"""Hostile string gate for metrics key before in."""

from __future__ import annotations

import pytest

from alberta_framework.utils.metrics import _metric_history_values

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


def test_metric_key_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("squared_error")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="metric key must be an exact string"):
        _metric_history_values([{"squared_error": 1.0}], hostile, name="metrics")
    assert _HostileStr.calls == 0


def test_metric_key_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="metric key must be an exact string"):
        _metric_history_values([{"squared_error": 1.0}], 123, name="metrics")  # type: ignore[arg-type]


def test_metric_key_benign_missing() -> None:
    with pytest.raises(ValueError, match="is missing metric"):
        _metric_history_values([{"other": 1.0}], "squared_error", name="metrics")


def test_metric_key_benign_valid() -> None:
    vals = _metric_history_values(
        [{"squared_error": 1.0}, {"squared_error": 2.0}], "squared_error", name="metrics"
    )
    assert len(vals) == 2
