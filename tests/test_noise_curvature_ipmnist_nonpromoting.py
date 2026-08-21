"""Unit coverage for alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting.

Tests the registered-arm surface and the exact validation helpers
(_int, _float, _strings, _object).
"""

import pytest

from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    _float,
    _int,
    _object,
    _strings,
    matched_arm_names,
    registered_arms,
    registered_hyperparameters,
)


def test_registered_arms() -> None:
    arms = registered_arms()
    assert isinstance(arms, tuple)
    assert len(arms) >= 1
    assert "noise_curvature_combined" in arms


def test_matched_arm_names_includes_live_control() -> None:
    names = matched_arm_names()
    assert len(names) == len(registered_arms()) + 1
    assert set(names) >= set(registered_arms())


def test_registered_hyperparameters_valid() -> None:
    hp = registered_hyperparameters(registered_arms()[0])
    assert isinstance(hp, dict)
    assert "controller_mode" in hp
    assert len(hp) >= 2


def test_registered_hyperparameters_rejects_bad_arm() -> None:
    with pytest.raises(ValueError, match="arm must be one of"):
        registered_hyperparameters("bogus")
    with pytest.raises(ValueError, match="arm must be one of"):
        registered_hyperparameters(123)


def test_int_validation() -> None:
    assert _int(0, context="x") == 0
    assert _int(5, context="x", positive=True) == 5
    with pytest.raises(ValueError, match="nonnegative"):
        _int(-1, context="x")
    with pytest.raises(ValueError, match="positive"):
        _int(0, context="x", positive=True)
    with pytest.raises(ValueError, match="signed-int32"):
        _int(1.5, context="x")


def test_float_validation() -> None:
    assert _float(0.5, context="x") == 0.5
    assert _float(0.5, context="x", nonnegative=True) == 0.5
    assert _float(0.5, context="x", unit=True) == 0.5
    with pytest.raises(ValueError, match="nonnegative"):
        _float(-0.1, context="x", nonnegative=True)
    with pytest.raises(ValueError, match="\\[0,1\\]"):
        _float(1.5, context="x", unit=True)
    with pytest.raises(ValueError, match="finite"):
        _float(float("nan"), context="x")
    with pytest.raises(ValueError, match="finite"):
        _float(1, context="x")  # int is not float


def test_strings_validation() -> None:
    assert _strings(["a", "b"], context="x") == ("a", "b")
    with pytest.raises(ValueError, match="nonempty"):
        _strings([""], context="x")
    with pytest.raises(ValueError, match="nonempty"):
        _strings(["ok", "\x00"], context="x")
    with pytest.raises(ValueError, match="bounded list"):
        _strings("not-a-list", context="x")


def test_object_validation() -> None:
    assert _object({"a": 1}, frozenset({"a"}), context="x") == {"a": 1}
    with pytest.raises(ValueError, match="exact object"):
        _object([1], frozenset(), context="x")
    with pytest.raises(ValueError, match="keys must be exactly"):
        _object({"a": 1, "b": 2}, frozenset({"a"}), context="x")
