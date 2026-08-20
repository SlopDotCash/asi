# mypy: disable-error-code="arg-type"
"""Boolean-trace walks reject oversized host lists before the width hang."""

from __future__ import annotations

import time

import pytest

from alberta_framework.utils.metrics import (
    _BOOLEAN_TRACE_MAX_NODES,
    compute_cumulative_error,
    compute_running_mean,
)

pytestmark = pytest.mark.unit


def test_running_mean_rejects_origin_hang_class_before_trace_walk() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="boolean-trace value limit"):
        compute_running_mean([0.0] * (_BOOLEAN_TRACE_MAX_NODES + 1), window_size=2)
    assert time.perf_counter() - started < 0.25


def test_running_mean_rejects_pointer_repeat_origin_hang_n() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="boolean-trace value limit"):
        compute_running_mean([0.0] * 15_000_000, window_size=2)
    assert time.perf_counter() - started < 0.25


def test_cumulative_error_rejects_oversized_metrics_history_before_walk() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="boolean-trace value limit"):
        compute_cumulative_error([{"squared_error": 1.0}] * (_BOOLEAN_TRACE_MAX_NODES + 1))
    assert time.perf_counter() - started < 0.25


def test_running_mean_accepts_public_last_fit() -> None:
    result = compute_running_mean([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], window_size=3)
    assert result.shape == (6,)
