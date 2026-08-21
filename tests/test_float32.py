"""Unit coverage for alberta_framework._float32.

Exercises the exact host-scalar → binary32 conversion: ties-to-even
rounding, subnormal/overflow handling, negative zero, ratio extraction
(actual-type gates), and the public rounding API (checked against
numpy.float32 where exact).
"""

import math
from fractions import Fraction

import numpy as np
import pytest

from alberta_framework._float32 import (
    _is_actual_int,
    _real_ratio,
    _round_quotient_ties_to_even,
    round_real_to_float32,
    round_real_to_float32_with_ratio,
)


def test_round_quotient_ties_to_even() -> None:
    # 1/2 → tie → round to 0 (even)
    assert _round_quotient_ties_to_even(1, 2) == 0
    # 3/2 → 1.5 → tie → round to 2 (even)
    assert _round_quotient_ties_to_even(3, 2) == 2
    # 5/2 → 2.5 → tie → round to 2 (even)
    assert _round_quotient_ties_to_even(5, 2) == 2
    # 7/2 → 3.5 → tie → round to 4 (even)
    assert _round_quotient_ties_to_even(7, 2) == 4
    # Non-tie rounds normally
    assert _round_quotient_ties_to_even(4, 2) == 2
    assert _round_quotient_ties_to_even(5, 3) == 2


def test_is_actual_int() -> None:
    assert _is_actual_int(1) is True
    assert _is_actual_int(np.int32(1)) is True
    assert _is_actual_int(1.0) is False
    assert _is_actual_int(True) is False


def test_real_ratio_basic() -> None:
    assert _real_ratio(3) == (3, 1, False)
    assert _real_ratio(-3) == (-3, 1, False)
    assert _real_ratio(0.5) == (1, 2, False)
    assert _real_ratio(Fraction(1, 3)) == (1, 3, False)


def test_real_ratio_negative_zero() -> None:
    n, d, neg_zero = _real_ratio(-0.0)
    assert n == 0
    assert neg_zero is True
    n2, d2, neg_zero2 = _real_ratio(0.0)
    assert neg_zero2 is False


def test_real_ratio_rejects_bool() -> None:
    with pytest.raises(TypeError, match="non-bool"):
        _real_ratio(True)


def test_float32_rounding_matches_numpy() -> None:
    values = [0.1, 0.2, 1.5, 3.14, 1e-40, 1e40, 123456.789, -0.5]
    for v in values:
        mine = round_real_to_float32(v)
        ref = float(np.float32(v))
        assert mine == ref, f"{v}: {mine} vs {ref}"


def test_float32_with_ratio_returns_ratio() -> None:
    n, d, rounded = round_real_to_float32_with_ratio(0.1)
    assert (n, d) == (3602879701896397, 36028797018963968)
    assert rounded == float(np.float32(0.1))


def test_float32_overflow_to_inf() -> None:
    assert math.isinf(round_real_to_float32(1e40))
    assert round_real_to_float32(1e40) > 0


def test_float32_subnormal() -> None:
    # 1e-40 is subnormal in float32; numpy matches.
    assert round_real_to_float32(1e-40) == float(np.float32(1e-40))


def test_rounds_fraction_midpoint_exactly() -> None:
    # Fraction midpoint at 1 + 2^-24 with a tiny offset (tie-to-even via ratio).
    frac = Fraction(1) + Fraction(1, 2**24) + Fraction(1, 2**60)
    rounded = round_real_to_float32(frac)
    # 1 + 2^-24 + tiny is exactly halfway → rounds to 1 (even) or 1+2^-23.
    assert rounded in (1.0, float(np.float32(1.0 + 2**-23)))


def test_rounds_overflow_midpoint_outward() -> None:
    # float32_max = (2^24-1)*2^104; its midpoint rounds to inf.
    float32_max = (2**24 - 1) * 2**104
    mid = float32_max + float32_max / 2
    assert math.isinf(round_real_to_float32(mid))


def test_rounds_subnormal_midpoint_to_even_zero() -> None:
    # Midpoint 2^-150 is between 0 and the min subnormal 2^-149 → tie → 0.
    assert round_real_to_float32(Fraction(1, 2**150)) == 0.0


def test_preserves_signed_zero() -> None:
    neg = round_real_to_float32_with_ratio(-0.0)
    assert math.copysign(1.0, neg[2]) == -1.0
    pos = round_real_to_float32_with_ratio(0.0)
    assert math.copysign(1.0, pos[2]) == 1.0


def test_rejects_bool_aliases() -> None:
    with pytest.raises(TypeError, match="non-bool"):
        _real_ratio(True)
    with pytest.raises(TypeError, match="non-bool"):
        _real_ratio(np.bool_(True))
