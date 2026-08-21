"""Unit coverage for alberta_framework.steps.step1.

Tests the validation primitives and resource math: finite real + float32
narrowing, canonical storage, unit-interval gates, checked resource
sum/product overflow protection, and the per-optimizer state scalar count.
"""

from fractions import Fraction

import numpy as np
import pytest

from alberta_framework.steps.step1 import (
    Step1KernelConfig,
    _checked_resource_product,
    _checked_resource_sum,
    _require_int,
    _require_unit_interval,
    _step1_state_scalar_count,
    canonical_float32_storage,
    finite_real_and_float32,
)


def test_finite_real_and_float32() -> None:
    real, num, den, narrowed = finite_real_and_float32("x", 0.5)
    assert real == 0.5
    assert (num, den) == (1, 2)
    assert narrowed == 0.5
    with pytest.raises(ValueError, match="real number"):
        finite_real_and_float32("x", "abc")
    with pytest.raises(ValueError, match="finite float32"):
        finite_real_and_float32("x", 1e40)  # narrows to inf


def test_finite_real_accepts_fraction() -> None:
    real, num, den, narrowed = finite_real_and_float32("x", Fraction(1, 3))
    assert (num, den) == (1, 3)


def test_canonical_storage() -> None:
    assert canonical_float32_storage(0.5, 0.5) == 0.5
    assert canonical_float32_storage(Fraction(1, 2), 0.5) == 0.5  # non-float → narrowed
    assert canonical_float32_storage(np.float32(0.5), 0.5) == 0.5


def test_require_unit_interval() -> None:
    assert _require_unit_interval("x", 0.5) == 0.5
    assert _require_unit_interval("x", 1.0) == 1.0
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        _require_unit_interval("x", 1.5)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        _require_unit_interval("x", -0.1)


def test_require_int() -> None:
    assert _require_int("x", 5) == 5
    with pytest.raises(ValueError, match="integer"):
        _require_int("x", 1.5)
    with pytest.raises(ValueError, match="positive"):
        _require_int("x", 0, minimum=1)
    with pytest.raises(ValueError, match="non-negative"):
        _require_int("x", -1, minimum=0)


def test_checked_sum() -> None:
    assert _checked_resource_sum("x", 1, 2, 3) == 6
    with pytest.raises(ValueError, match="signed int32"):
        _checked_resource_sum("x", 2**31 - 1, 1)
    with pytest.raises(ValueError, match="signed int32"):
        _checked_resource_sum("x", -1)


def test_checked_product() -> None:
    assert _checked_resource_product("x", 2, 3, 4) == 24
    assert _checked_resource_product("x", 0) == 0
    with pytest.raises(ValueError, match="signed int32"):
        _checked_resource_product("x", 2**20, 2**20)


def test_state_scalar_count_widths() -> None:
    cfg = Step1KernelConfig(
        feature_dim=10,
        optimizer="lms",
        normalizer="none",
        stream="alberta",
    )
    # width = 1 + 0 (lms) + 0 (none) + 1 (alberta) = 2 → 2*10 + 32 = 52
    assert _step1_state_scalar_count(cfg) == 52


def test_state_scalar_count_idbd() -> None:
    cfg = Step1KernelConfig(
        feature_dim=10,
        optimizer="idbd",
        normalizer="ema",
        stream="xdist_shift",
    )
    # width = 1 + 2 (idbd) + 2 (ema) + 2 (xdist) = 7 → 7*10 + 32 = 102
    assert _step1_state_scalar_count(cfg) == 102


def test_config_rejects_bad_optimizer() -> None:
    with pytest.raises(ValueError, match="unknown Step 1 optimizer"):
        Step1KernelConfig(feature_dim=8, optimizer="bogus")
