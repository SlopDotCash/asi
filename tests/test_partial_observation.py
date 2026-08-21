"""Unit coverage for alberta_framework.streams.partial_observation.

Tests the fail-closed wrapper gates: exact MaskMode enforcement (no
leftover string fall-through), boolean-mask shape/dtype validation,
probability scalar domain, and feature-dim bounds.
"""

import numpy as np
import pytest

from alberta_framework.streams.partial_observation import (
    MaskMode,
    _require_feature_dim,
    _require_mode,
    _require_unit_interval_probability,
    _trusted_boolean_mask,
)


def test_mask_mode_values() -> None:
    assert MaskMode.FIXED.value == "fixed"
    assert MaskMode.RANDOM.value == "random"
    assert MaskMode.PERIODIC.value == "periodic"


def test_require_mode_exact() -> None:
    assert _require_mode(MaskMode.FIXED) is MaskMode.FIXED
    # Leftover strings must be rejected (fail-open prevention).
    with pytest.raises(ValueError, match="exact MaskMode"):
        _require_mode("fixed")
    with pytest.raises(ValueError, match="exact MaskMode"):
        _require_mode("FIXED")
    with pytest.raises(ValueError, match="exact MaskMode"):
        _require_mode(1)


def test_require_feature_dim() -> None:
    assert _require_feature_dim(10) == 10
    assert _require_feature_dim(2**31 - 1) == 2**31 - 1
    with pytest.raises(ValueError, match="\\[1, 2147483647\\]"):
        _require_feature_dim(0)
    with pytest.raises(ValueError, match="\\[1, 2147483647\\]"):
        _require_feature_dim(2**31)
    with pytest.raises(ValueError, match="\\[1, 2147483647\\]"):
        _require_feature_dim(10.5)


def test_require_unit_interval_probability() -> None:
    assert _require_unit_interval_probability("x", 0.5) == 0.5
    assert _require_unit_interval_probability("x", 0.0) == 0.0
    assert _require_unit_interval_probability("x", 1.0) == 1.0
    with pytest.raises(ValueError):
        _require_unit_interval_probability("x", 1.5)
    with pytest.raises(ValueError):
        _require_unit_interval_probability("x", -0.1)


def test_trusted_boolean_mask_accepts() -> None:
    mask = np.array([True, False, True])
    assert _trusted_boolean_mask("mask", mask, 3).shape == (3,)


def test_trusted_boolean_mask_rejects_shape() -> None:
    mask = np.array([True, False])
    with pytest.raises(ValueError, match="shape"):
        _trusted_boolean_mask("mask", mask, 3)


def test_trusted_boolean_mask_rejects_dtype() -> None:
    mask = np.array([1, 0, 1])  # int, not bool
    with pytest.raises(ValueError, match="dtype bool"):
        _trusted_boolean_mask("mask", mask, 3)


def test_trusted_boolean_mask_rejects_type() -> None:
    with pytest.raises(ValueError, match="exact NumPy or JAX"):
        _trusted_boolean_mask("mask", [True, False, True], 3)
