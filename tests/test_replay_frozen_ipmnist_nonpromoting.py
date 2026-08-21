"""Unit coverage for alberta_framework.evaluation.replay_frozen_ipmnist_nonpromoting.

Tests the validation helpers and parameter-count math: exact type gates,
bound checks, duplicate rejection, and the parameter-count formula.
"""

import pytest

from alberta_framework.evaluation.replay_frozen_ipmnist_nonpromoting import (
    _float,
    _int,
    _object,
    _parameter_count,
    _strings,
)


def test_parameter_count_formula() -> None:
    # input_dim*hidden1 + hidden1*hidden2 + hidden2*classes + hidden1+hidden2+classes
    assert _parameter_count(2, 3, 4, 5) == 2 * 3 + 3 * 4 + 4 * 5 + 3 + 4 + 5
    assert _parameter_count(1, 1, 1, 1) == 1 + 1 + 1 + 1 + 1 + 1


def test_parameter_count_overflow() -> None:
    with pytest.raises(ValueError, match="signed int32"):
        _parameter_count(2**20, 2**20, 2**20, 2**20)  # product overflows


def test_int_validation() -> None:
    assert _int(0, "x") == 0
    assert _int(5, "x", minimum=3) == 5
    with pytest.raises(ValueError, match="bounded integer"):
        _int(1.5, "x")
    with pytest.raises(ValueError, match="bounded integer"):
        _int(-1, "x", minimum=0)
    with pytest.raises(ValueError, match="bounded integer"):
        _int(2**31, "x")  # exceeds int32


def test_float_validation() -> None:
    assert _float(1.5, "x") == 1.5
    assert _float(0.0, "x") == 0.0
    with pytest.raises(ValueError, match="bounded nonnegative"):
        _float(-0.1, "x")
    with pytest.raises(ValueError, match="bounded nonnegative"):
        _float(1, "x")  # int not float
    with pytest.raises(ValueError, match="bounded nonnegative"):
        _float(float("nan"), "x")
    with pytest.raises(ValueError, match="bounded nonnegative"):
        _float(1e39, "x")  # exceeds float32 max


def test_strings_validation() -> None:
    assert _strings(["a", "b"], "x") == ("a", "b")
    with pytest.raises(ValueError, match="duplicates"):
        _strings(["a", "a"], "x")
    with pytest.raises(ValueError, match="bounded exact string"):
        _strings([""], "x")
    with pytest.raises(ValueError, match="bounded exact string"):
        _strings("abc", "x")
    with pytest.raises(ValueError, match="bounded exact string"):
        _strings(["\x00"], "x")


def test_object_validation() -> None:
    assert _object({"a": 1}, frozenset({"a"}), "x") == {"a": 1}
    with pytest.raises(ValueError, match="exactly"):
        _object({"a": 1, "b": 2}, frozenset({"a"}), "x")
    with pytest.raises(ValueError, match="exactly"):
        _object([1], frozenset(), "x")
