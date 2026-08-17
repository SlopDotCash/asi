"""Hostile validation for steps/_float32_validation sink gate."""

from fractions import Fraction

import numpy as np
import pytest

from alberta_framework.steps._float32_validation import finite_real_and_float32


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat.as_integer_ratio must not be called")

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat.__float__ must not be called")


def test_hostile_float_raises_finite_without_repr_leak() -> None:
    # This layer intentionally allows true float subclasses (see HiddenBoundaryFloat
    # tests) so the hostile hook IS invoked, but the error must be sanitized to
    # "must be finite" without leaking the hostile repr via !r.
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        finite_real_and_float32("x", _HostileFloat(1.0))
    assert _HostileFloat.calls == 1
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_string_subclass_name() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        finite_real_and_float32(_StringSubclass("x"), 1.0)  # type: ignore[arg-type]


def test_hostile_name_repr_not_invoked() -> None:
    evil = _EvilStr("x")
    with pytest.raises(ValueError, match="must be an exact string"):
        finite_real_and_float32(evil, 1.0)  # type: ignore[arg-type]


def test_rejects_bool_value() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        finite_real_and_float32("x", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a real number"):
        finite_real_and_float32("x", np.bool_(True))  # type: ignore[arg-type]


def test_rejects_string_subclass_value() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        finite_real_and_float32("x", _StringSubclass("1.0"))  # type: ignore[arg-type]


def test_hostile_float_value_raises_finite() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite"):
        finite_real_and_float32("x", _HostileFloat(0.5))
    assert _HostileFloat.calls == 1


def test_rejects_plain_string_without_repr() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        finite_real_and_float32("x", "1.0")  # type: ignore[arg-type]


def test_does_not_invoke_hostile_value_when_name_is_evil() -> None:
    evil = _EvilStr("x")
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be an exact string"):
        finite_real_and_float32(evil, _HostileFloat(1.0))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_valid_int_float_fraction_numpy_pass() -> None:
    _, n, d, _ = finite_real_and_float32("x", 1)
    assert (n, d) == (1, 1)
    _, n, d, _ = finite_real_and_float32("x", 1.0)
    assert d == 1
    _, n, d, _ = finite_real_and_float32("x", Fraction(1, 3))
    assert (n, d) == (1, 3)
    _, _, _, _ = finite_real_and_float32("x", np.float64(1.5))
    _, _, _, _ = finite_real_and_float32("x", np.int32(2))
    _, _, _, _ = finite_real_and_float32("x", np.float32(0.25))


def test_valid_zero_and_negative_finite_pass() -> None:
    # finite_real_and_float32 is domain-agnostic finite check, so 0 and negatives pass
    _, n, d, narrowed = finite_real_and_float32("x", 0.0)
    assert n == 0
    _, _, _, narrowed2 = finite_real_and_float32("x", -1.5)
    assert narrowed2 == pytest.approx(-1.5)  # type: ignore[arg-type]


def test_rejects_non_finite_narrowed() -> None:
    # 1e40 narrows to inf in float32, must be rejected as finite
    with pytest.raises(ValueError, match="must be finite"):
        finite_real_and_float32("x", 1e40)


def test_float_subclass_with_lying_ratio_is_canonicalized() -> None:
    class RatioFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return (3, 4)

    _, n, d, narrowed = finite_real_and_float32("x", RatioFloat(0.5))
    assert (n, d) == (3, 4)
    assert narrowed == pytest.approx(0.75)
