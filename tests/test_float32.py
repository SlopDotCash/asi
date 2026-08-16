"""Exact boundary tests for host-real to IEEE binary32 narrowing."""

from fractions import Fraction
from numbers import Real

import numpy as np
import pytest

from alberta_framework._float32 import round_real_to_float32


def _raw_float32(value: float) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32).item())


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (-Fraction(1, 2**60), np.float32(1.0)),
        (Fraction(0), np.float32(1.0)),
        (
            Fraction(1, 2**60),
            np.nextafter(np.float32(1.0), np.float32(2.0)),
        ),
    ],
    ids=("below", "tie-to-even", "above"),
)
def test_rounds_fraction_midpoint_exactly(offset: Fraction, expected: np.float32) -> None:
    midpoint = Fraction(1) + Fraction(1, 2**24)
    assert _raw_float32(round_real_to_float32(midpoint + offset)) == _raw_float32(
        float(expected)
    )


def test_rounds_overflow_midpoint_outward() -> None:
    float32_max = (2**24 - 1) * 2**104
    midpoint = Fraction(float32_max + 2**103)

    assert round_real_to_float32(midpoint - 1) == float(np.finfo(np.float32).max)
    assert round_real_to_float32(midpoint) == float("inf")
    assert round_real_to_float32(midpoint + 1) == float("inf")


def test_rounds_subnormal_midpoint_to_even_zero() -> None:
    midpoint = Fraction(1, 2**150)
    minimum_subnormal = np.nextafter(np.float32(0.0), np.float32(1.0))

    assert _raw_float32(round_real_to_float32(midpoint - Fraction(1, 2**200))) == 0
    assert _raw_float32(round_real_to_float32(midpoint)) == 0
    assert _raw_float32(round_real_to_float32(midpoint + Fraction(1, 2**200))) == (
        _raw_float32(float(minimum_subnormal))
    )


@pytest.mark.parametrize("value", [-0.0, np.float32(-0.0), np.longdouble(-0.0)])
def test_preserves_signed_zero(value: float) -> None:
    assert _raw_float32(round_real_to_float32(value)) == 0x80000000


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_rejects_bool_aliases(value: object) -> None:
    with pytest.raises(TypeError):
        round_real_to_float32(value)  # type: ignore[arg-type]


def test_rejects_non_real_whose_class_property_spoofs_float() -> None:
    class ClassSpoof:
        @property
        def __class__(self) -> type[float]:
            return float

        def as_integer_ratio(self) -> tuple[int, int]:
            return (1, 2)

    value = ClassSpoof()
    assert isinstance(value, Real)
    assert not issubclass(type(value), Real)

    with pytest.raises(TypeError, match="actual non-bool real"):
        round_real_to_float32(value)  # type: ignore[arg-type]


def test_float_subclass_cannot_spoof_integral_ratio_dispatch() -> None:
    class IntegralSpoofFloat(float):
        calls = 0

        @property
        def __class__(self) -> type[int]:
            return int

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            return (-1, 2**200)

    value = IntegralSpoofFloat(0.5)

    assert _raw_float32(round_real_to_float32(value)) == 0x80000000
    assert IntegralSpoofFloat.calls == 1


def test_rejects_ratio_component_whose_class_property_spoofs_int() -> None:
    class IntegerSpoof:
        @property
        def __class__(self) -> type[int]:
            return int

        def __int__(self) -> int:
            return 1

    class MalformedRatioFloat(float):
        def as_integer_ratio(self) -> tuple[object, int]:
            return (IntegerSpoof(), 2)

    with pytest.raises(TypeError, match="integer pair"):
        round_real_to_float32(MalformedRatioFloat(0.5))
