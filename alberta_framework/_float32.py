"""Exact host-scalar conversion helpers for float32 JAX sinks."""

from __future__ import annotations

import struct
from fractions import Fraction
from typing import Any, cast

import numpy as np

_ACTUAL_INT_TYPES = frozenset({int, *(np.dtype(code).type for code in "bBhHiIlLqQpP")})
_ACTUAL_FLOAT_TYPES = frozenset(
    {
        float,
        Fraction,
        np.dtype("e").type,
        np.dtype("f").type,
        np.dtype("d").type,
        np.dtype("g").type,
    }
)
_ALLOWED_REAL_TYPES = _ACTUAL_INT_TYPES | _ACTUAL_FLOAT_TYPES


def _is_actual_int(value: object) -> bool:
    actual_type = type(value)
    return any(actual_type is allowed_type for allowed_type in _ACTUAL_INT_TYPES)


def _round_quotient_ties_to_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    doubled_remainder = remainder * 2
    if doubled_remainder > denominator or (
        doubled_remainder == denominator and quotient % 2 == 1
    ):
        return quotient + 1
    return quotient


def _float32_from_ratio(
    numerator: int,
    denominator: int,
    *,
    negative_zero: bool,
) -> float:
    """Round an exact ratio to IEEE 754 binary32 with ties-to-even."""
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    if denominator == 0:
        raise ValueError("ratio denominator must be nonzero")

    negative = numerator < 0 or (numerator == 0 and negative_zero)
    magnitude = abs(numerator)
    sign_bits = int(negative) << 31
    if magnitude == 0:
        bits = sign_bits
    else:
        exponent = magnitude.bit_length() - denominator.bit_length()
        if exponent >= 0:
            if magnitude < denominator << exponent:
                exponent -= 1
        elif magnitude << -exponent < denominator:
            exponent -= 1

        if exponent > 127:
            bits = sign_bits | 0x7F800000
        elif exponent >= -126:
            shift = 23 - exponent
            if shift >= 0:
                significand = _round_quotient_ties_to_even(
                    magnitude << shift,
                    denominator,
                )
            else:
                significand = _round_quotient_ties_to_even(
                    magnitude,
                    denominator << -shift,
                )
            if significand == 1 << 24:
                significand >>= 1
                exponent += 1
            if exponent > 127:
                bits = sign_bits | 0x7F800000
            else:
                bits = (
                    sign_bits
                    | ((exponent + 127) << 23)
                    | (significand - (1 << 23))
                )
        else:
            significand = _round_quotient_ties_to_even(
                magnitude << 149,
                denominator,
            )
            bits = sign_bits | significand
    return float(struct.unpack("!f", bits.to_bytes(4, byteorder="big"))[0])


def _real_ratio(value: object) -> tuple[int, int, bool]:
    """Return one normalized exact ratio and its zero-sign metadata."""
    actual_type = type(value)
    if not any(actual_type is allowed_type for allowed_type in _ALLOWED_REAL_TYPES):
        raise TypeError("value must be an actual non-bool real")
    if _is_actual_int(value):
        ratio: object = (int(cast(Any, value)), 1)
    elif actual_type is float:
        ratio = float.as_integer_ratio(cast(float, value))
    elif actual_type is Fraction:
        fraction = cast(Fraction, value)
        ratio = (fraction.numerator, fraction.denominator)
    else:
        # NumPy's concrete floating scalar classes are in the allow-list.
        # Dispatch through the concrete class after the exact-type gate so an
        # instance cannot substitute an attribute hook.
        ratio = cast(Any, actual_type).as_integer_ratio(value)
    if type(ratio) is not tuple:
        raise TypeError("as_integer_ratio must return an integer pair")
    ratio_tuple = cast(tuple[object, ...], ratio)
    if len(ratio_tuple) != 2:
        raise TypeError("as_integer_ratio must return an integer pair")
    numerator_raw, denominator_raw = ratio_tuple
    if not _is_actual_int(numerator_raw) or not _is_actual_int(denominator_raw):
        raise TypeError("as_integer_ratio must return an integer pair")
    numerator = int(cast(Any, numerator_raw))
    denominator = int(cast(Any, denominator_raw))
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    if denominator == 0:
        raise ValueError("ratio denominator must be nonzero")
    negative_zero = (
        numerator == 0
        and actual_type in _ACTUAL_FLOAT_TYPES
        and actual_type is not Fraction
        and bool(np.signbit(cast(Any, value)))
    )
    return numerator, denominator, negative_zero


def round_real_to_float32_with_ratio(value: object) -> tuple[int, int, float]:
    """Read one exact ratio and return it with its binary32 rounding.

    Returning the same ratio used for rounding lets domain validators retain
    facts that disappear at binary32 endpoints, such as a negative value that
    rounds to ``-0.0`` or a value above one that rounds to ``1.0``.
    """
    numerator, denominator, negative_zero = _real_ratio(value)
    rounded = _float32_from_ratio(
        numerator,
        denominator,
        negative_zero=negative_zero,
    )
    return numerator, denominator, rounded


def round_real_to_float32(value: object) -> float:
    """Round a standard exact-ratio real directly to IEEE binary32.

    Integer and ``as_integer_ratio`` inputs are rounded with IEEE
    round-to-nearest, ties-to-even semantics without an intermediate binary64
    conversion. Real implementations that cannot expose an exact ratio are
    rejected instead of being silently double-rounded.
    """
    _, _, rounded = round_real_to_float32_with_ratio(value)
    return rounded


__all__ = ["round_real_to_float32", "round_real_to_float32_with_ratio"]
