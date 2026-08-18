"""Leftover-identity gates for publication significance records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.utils.statistics import SignificanceResult


def test_significance_result_rejects_leftover_identities() -> None:
    """Public records must not keep leftover significant/name identities."""

    with pytest.raises(ValueError, match="significant"):
        SignificanceResult("t", 1.0, 0.01, 1, 0.05, 0.2, "a", "b")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="significant"):
        SignificanceResult("t", 1.0, 0.99, 0, 0.05, 0.2, "a", "b")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="significant"):
        SignificanceResult("t", 1.0, 0.01, "FIXED", 0.05, 0.2, "a", "b")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="test_name"):
        SignificanceResult(True, 1.0, 0.01, True, 0.05, 0.2, "a", "b")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="method_a"):
        SignificanceResult("t", 1.0, 0.01, True, 0.05, 0.2, 1, "b")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="method_b"):
        SignificanceResult("t", 1.0, 0.01, True, 0.05, 0.2, "a", False)  # type: ignore[arg-type]

    legal = SignificanceResult("t", 1.0, 0.01, True, 0.05, 0.2, "a", "b")
    dumped = json.dumps(legal._asdict(), allow_nan=False)
    assert dumped == (
        '{"test_name": "t", "statistic": 1.0, "p_value": 0.01, '
        '"significant": true, "alpha": 0.05, "effect_size": 0.2, '
        '"method_a": "a", "method_b": "b"}'
    )
    assert '"significant": 1' not in dumped
    assert '"significant": "FIXED"' not in dumped
    assert type(legal.significant) is bool
    rejected = SignificanceResult("t", 1.0, 0.99, False, 0.05, 0.2, "a", "b")
    assert json.dumps(rejected._asdict(), allow_nan=False).count('"significant": false') == 1
