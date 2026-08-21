"""Unit coverage for alberta_framework.evaluation.cchain_ipmnist_nonpromoting.

Tests the checked product (overflow protection), expected-hyperparameter
surface per arm, and the exact _int/_float/_strings/_object validation
helpers.
"""

import pytest

from alberta_framework.evaluation.cchain_ipmnist_nonpromoting import (
    _checked_product,
    _float,
    _int,
    _object,
    _strings,
)


def test_checked_product() -> None:
    assert _checked_product(2, 3, 4, context="x") == 24
    assert _checked_product(1, 1, context="x") == 1
    with pytest.raises(ValueError, match="signed int32"):
        _checked_product(2**20, 2**20, context="x")  # overflows


def test_checked_product_rejects_zero_or_negative() -> None:
    with pytest.raises(ValueError, match="signed int32"):
        _checked_product(0, context="x")
    with pytest.raises(ValueError, match="signed int32"):
        _checked_product(-1, context="x")


def test_int_validation() -> None:
    assert _int(0, context="x") == 0
    assert _int(7, context="x", minimum=3) == 7
    with pytest.raises(ValueError, match="exact integer"):
        _int(1.5, context="x")
    with pytest.raises(ValueError, match="exact integer"):
        _int(-1, context="x")
    with pytest.raises(ValueError, match="exact integer"):
        _int(2**31, context="x")


def test_float_validation() -> None:
    assert _float(0.5, context="x") == 0.5
    assert _float(1.5, context="x", minimum=1.0, maximum=2.0) == 1.5
    with pytest.raises(ValueError, match="exact finite float"):
        _float(1, context="x")  # int not float
    with pytest.raises(ValueError, match="exact finite float"):
        _float(float("nan"), context="x")
    with pytest.raises(ValueError, match="exact finite float"):
        _float(0.5, context="x", minimum=1.0)
    with pytest.raises(ValueError, match="exact finite float"):
        _float(2.5, context="x", maximum=2.0)


def test_strings_validation() -> None:
    assert _strings(["a", "b"], context="x") == ("a", "b")
    with pytest.raises(ValueError, match="duplicates"):
        _strings(["a", "a"], context="x")
    with pytest.raises(ValueError, match="non-empty"):
        _strings([""], context="x")
    with pytest.raises(ValueError, match="non-empty"):
        _strings(["ok", "\x00"], context="x")


def test_object_validation() -> None:
    assert _object({"a": 1}, frozenset({"a"}), context="x") == {"a": 1}
    with pytest.raises(ValueError, match="exact object"):
        _object([1], frozenset(), context="x")
    with pytest.raises(ValueError, match="exact object"):
        _object({"a": 1, "b": 2}, frozenset({"a"}), context="x")


def test_expected_hyperparameters() -> None:
    from alberta_framework.evaluation.cchain_ipmnist_nonpromoting import (
        _ARMS,
        _expected_hyperparameters,
    )

    for arm in _ARMS:
        hp = _expected_hyperparameters(arm)
        assert "churn_enabled" in hp
        assert "gradient_component" in hp
        assert hp["initial_coefficient"] == 1.0
