"""Unit coverage for alberta_framework.streams.gymnasium.

Tests the fail-closed scalar gates and the discounted-bootstrap helper
(0*inf = 0 semantics), without constructing Gymnasium spaces.
"""

import pytest

from alberta_framework.streams.gymnasium import (
    PredictionMode,
    _discounted_bootstrap,
    _require_allocation,
    _require_exact_bool,
    _require_mode,
    _require_positive_int32,
)


def test_require_positive_int32() -> None:
    assert _require_positive_int32("x", 5) == 5
    assert _require_positive_int32("x", 2**31 - 1) == 2**31 - 1
    with pytest.raises(ValueError, match="\\[1, 2147483647\\]"):
        _require_positive_int32("x", 0)
    with pytest.raises(ValueError, match="\\[1, 2147483647\\]"):
        _require_positive_int32("x", 2**31)
    with pytest.raises(ValueError, match="\\[1, 2147483647\\]"):
        _require_positive_int32("x", 1.5)


def test_require_allocation() -> None:
    _require_allocation("x", 100, itemsize=4)  # ok
    with pytest.raises(ValueError, match="signed int32"):
        _require_allocation("x", 2**30, itemsize=8)  # bytes overflow


def test_require_float32_allocation() -> None:
    from alberta_framework.streams.gymnasium import _require_float32_allocation

    _require_float32_allocation("x", 100)
    with pytest.raises(ValueError, match="signed int32"):
        _require_float32_allocation("x", 2**31)


def test_require_exact_bool() -> None:
    assert _require_exact_bool("x", True) is True
    with pytest.raises(ValueError, match="exact bool"):
        _require_exact_bool("x", 1)
    with pytest.raises(ValueError, match="exact bool"):
        _require_exact_bool("x", "true")


def test_require_mode() -> None:
    assert _require_mode(PredictionMode.REWARD) is PredictionMode.REWARD
    with pytest.raises(ValueError, match="exact PredictionMode"):
        _require_mode("reward")
    with pytest.raises(ValueError, match="exact PredictionMode"):
        _require_mode(1)


def test_discounted_bootstrap() -> None:
    assert _discounted_bootstrap(0.99, 10.0) == 9.9
    assert _discounted_bootstrap(0.0, 10.0) == 0.0
    # gamma=0 with inf next value → 0 (not nan).
    assert _discounted_bootstrap(0.0, float("inf")) == 0.0
    assert _discounted_bootstrap(0.5, 0.0) == 0.0


def test_prediction_mode_values() -> None:
    assert PredictionMode.REWARD.value == "reward"
    assert PredictionMode.NEXT_STATE.value == "next_state"
    assert PredictionMode.VALUE.value == "value"
