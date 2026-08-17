"""Hostile-safe validation for float32 scalars."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from alberta_framework.core._float32_scalars import (
    validated_float32_scalar,
    validated_float32_scalar_with_ratio,
)


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:
        type(self).calls += 1
        raise RuntimeError("ratio hook")

    def __float__(self) -> float:
        type(self).calls += 1
        raise RuntimeError("float hook")


class _StringSubclass(str):
    pass


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr hook")


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise RuntimeError("str hook")

    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook")


def test_rejects_hostile_float_without_hook() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="finite real"):
        validated_float32_scalar("x", _HostileFloat(1.0))
    assert _HostileFloat.calls == 0
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="finite real"):
        validated_float32_scalar_with_ratio("x", _HostileFloat(1.0))
    assert _HostileFloat.calls == 0


def test_rejects_string_subclass_name() -> None:
    with pytest.raises(ValueError, match="exact string"):
        validated_float32_scalar(_StringSubclass("x"), 1.0)
    with pytest.raises(ValueError, match="exact string"):
        validated_float32_scalar_with_ratio(_StringSubclass("x"), 1.0)


def test_hostile_name_repr_not_invoked() -> None:
    with pytest.raises(ValueError, match="exact string"):
        validated_float32_scalar(_EvilStr("x"), 1.0)
    with pytest.raises(ValueError, match="exact string"):
        validated_float32_scalar_with_ratio(_EvilStr("x"), 1.0)


def test_rejects_bool_value() -> None:
    with pytest.raises(ValueError, match="finite real"):
        validated_float32_scalar("x", True)
    with pytest.raises(ValueError, match="finite real"):
        validated_float32_scalar("x", np.bool_(True))


def test_rejects_string_subclass_value() -> None:
    with pytest.raises(ValueError, match="finite real"):
        validated_float32_scalar("x", _StringSubclass("1.0"))


def test_rejects_bool_positive_flag() -> None:
    with pytest.raises(ValueError, match="built-in bool"):
        validated_float32_scalar("x", 1.0, positive=1)
    with pytest.raises(ValueError, match="built-in bool"):
        validated_float32_scalar("x", 1.0, positive=_StringSubclass("true"))


def test_rejects_bool_upper_inclusive_flag() -> None:
    with pytest.raises(ValueError, match="built-in bool"):
        validated_float32_scalar("x", 0.5, upper=1.0, upper_inclusive=1)


def test_rejects_hostile_float_lower_without_hook() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="lower must be"):
        validated_float32_scalar("x", 0.5, lower=_HostileFloat(0.1))
    assert _HostileFloat.calls == 0


def test_rejects_string_subclass_lower() -> None:
    with pytest.raises(ValueError, match="lower must be"):
        validated_float32_scalar("x", 0.5, lower=_StringSubclass("0.1"))


def test_rejects_hostile_float_upper_without_hook() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="upper must be"):
        validated_float32_scalar("x", 0.5, upper=_HostileFloat(1.0))
    assert _HostileFloat.calls == 0


def test_valid_int_float_fraction_numpy_pass() -> None:
    assert validated_float32_scalar("x", 2) == pytest.approx(2.0)
    assert validated_float32_scalar("x", 0.5) == pytest.approx(0.5)
    assert validated_float32_scalar("x", Fraction(1, 3)) == pytest.approx(
        validated_float32_scalar("x", 1 / 3)
    )
    assert validated_float32_scalar("x", np.float64(0.5)) == pytest.approx(0.5)
    assert validated_float32_scalar("x", np.int32(7)) == pytest.approx(7.0)
    s, n, d = validated_float32_scalar_with_ratio("x", 1.5)
    assert s == pytest.approx(1.5)
    assert n == 3 and d == 2


def test_valid_bounds_and_domain_pass() -> None:
    assert validated_float32_scalar("x", 0.5, lower=0.0, upper=1.0) == pytest.approx(0.5)
    assert validated_float32_scalar(
        "x", 1.0, lower=0.0, upper=1.0, upper_inclusive=True
    ) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="must be"):
        validated_float32_scalar(
            "x", 1.0, lower=0.0, upper=1.0, upper_inclusive=False
        )
    with pytest.raises(ValueError, match="must be"):
        validated_float32_scalar("x", 0.0, positive=True)
    assert validated_float32_scalar("x", 0.1, positive=True) == pytest.approx(0.1)


def test_exact_fraction_bound_cannot_be_lost_through_binary64_conversion() -> None:
    just_above_one = Fraction(2**54 + 1, 2**54)
    with pytest.raises(ValueError, match="must remain"):
        validated_float32_scalar(
            "x",
            just_above_one,
            lower=just_above_one,
        )


def test_finite_wide_upper_bound_does_not_require_binary64_conversion() -> None:
    assert validated_float32_scalar("x", 1.0, upper=10**1000) == 1.0
