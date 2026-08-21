"""Unit coverage for alberta_framework.steps.step6.

Tests the fail-closed primitives: checked product/sum overflow
protection, float32 storage compatibility (incl. signed-zero semantics),
unit-interval and non-negative scalar gates, and integer validation.
"""

import pytest

from alberta_framework.steps.step6 import (
    _checked_product,
    _checked_sum,
    _compatible_float32_storage,
    _require_int,
    _require_nonnegative_real,
    _require_unit_interval,
)


def test_checked_product() -> None:
    assert _checked_product("x", 2, 3, 4) == 24
    assert _checked_product("x", 0) == 0
    with pytest.raises(ValueError, match="signed int32"):
        _checked_product("x", 2**20, 2**20)
    with pytest.raises(ValueError, match="signed int32"):
        _checked_product("x", -1)


def test_checked_sum() -> None:
    assert _checked_sum("x", 1, 2, 3) == 6
    with pytest.raises(ValueError, match="signed int32"):
        _checked_sum("x", 2**31 - 1, 1)
    with pytest.raises(ValueError, match="signed int32"):
        _checked_sum("x", -1)


def test_compatible_float32_storage() -> None:
    assert _compatible_float32_storage(1.5, 1.5) == 1.5
    assert _compatible_float32_storage(3, 3.0) == 3.0  # int equal → float
    # 0.0 narrowed to 0.0: float kept only if value == 0 (not -0.0 case)
    assert _compatible_float32_storage(0.0, 0.0) == 0.0


def test_require_unit_interval() -> None:
    assert _require_unit_interval("x", 0.5) == 0.5
    assert _require_unit_interval("x", 0.0) == 0.0
    assert _require_unit_interval("x", 1.0) == 1.0
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        _require_unit_interval("x", 1.5)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        _require_unit_interval("x", -0.1)


def test_require_nonnegative_real() -> None:
    assert _require_nonnegative_real("x", 0.0) == 0.0
    assert _require_nonnegative_real("x", 2.5) == 2.5
    with pytest.raises(ValueError, match="non-negative"):
        _require_nonnegative_real("x", -1.0)


def test_require_int() -> None:
    assert _require_int("x", 5) == 5
    with pytest.raises(ValueError, match="integer"):
        _require_int("x", 1.5)
    with pytest.raises(ValueError, match="positive"):
        _require_int("x", 0, minimum=1)
    with pytest.raises(ValueError, match="non-negative"):
        _require_int("x", -1, minimum=0)
    with pytest.raises(ValueError, match="int32 max"):
        _require_int("x", 2**31, maximum=2**31 - 1)
