"""Protocol ceilings for CLEAR accuracy-matrix enumeration."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.clear_qualification import (
    MAX_METRIC_MATRIX_ROWS,
    ClearQualificationError,
    _metric_values,
)


def test_documented_protocol_ceiling() -> None:
    assert MAX_METRIC_MATRIX_ROWS == 10_000


def test_rejects_oversized_accuracy_matrix() -> None:
    with pytest.raises(ClearQualificationError, match="accuracy matrix"):
        _metric_values([[0.0] for _ in range(MAX_METRIC_MATRIX_ROWS + 1)])
