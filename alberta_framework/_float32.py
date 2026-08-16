"""Exact host-scalar conversion helpers for float32 JAX sinks."""

from __future__ import annotations

import math
import struct
from numbers import Integral, Real
from typing import cast


def _is_actual_integral(value: object) -> bool:
    """Return whether the runtime type is a non-bool Integral subtype."""
    actual_type = type(value)
    return not issubclass(actual_type, bool) and issubclass(actual_type, Integral)


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


def _real_ratio(value: Real) -> tuple[int, int, bool]:
    """Return one normalized exact ratio and its zero-sign metadata."""
    actual_type = type(value)
    if issubclass(actual_type, bool) or not issubclass(actual_type, Real):
        raise TypeError("value must be an actual non-bool real")
    if _is_actual_integral(value):
        ratio: object = (int(cast(Integral, value)), 1)
    else:
        ratio_method = getattr(value, "as_integer_ratio", None)
        if not callable(ratio_method):
            raise TypeError("real value must expose as_integer_ratio")
        ratio = ratio_method()
    if not issubclass(type(ratio), tuple):
        raise TypeError("as_integer_ratio must return an integer pair")
    ratio_tuple = cast(tuple[object, ...], ratio)
    if len(ratio_tuple) != 2:
        raise TypeError("as_integer_ratio must return an integer pair")
    numerator_raw, denominator_raw = ratio_tuple
    if not _is_actual_integral(numerator_raw) or not _is_actual_integral(
        denominator_raw
    ):
        raise TypeError("as_integer_ratio must return an integer pair")
    numerator = int(cast(Integral, numerator_raw))
    denominator = int(cast(Integral, denominator_raw))
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    if denominator == 0:
        raise ValueError("ratio denominator must be nonzero")
    negative_zero = numerator == 0 and math.copysign(1.0, float(value)) < 0.0
    return numerator, denominator, negative_zero


def round_real_to_float32_with_ratio(value: Real) -> tuple[int, int, float]:
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


def round_real_to_float32(value: Real) -> float:
    """Round a standard exact-ratio real directly to IEEE binary32.

    Integer and ``as_integer_ratio`` inputs are rounded with IEEE
    round-to-nearest, ties-to-even semantics without an intermediate binary64
    conversion. Real implementations that cannot expose an exact ratio are
    rejected instead of being silently double-rounded.
    """
    _, _, rounded = round_real_to_float32_with_ratio(value)
    return rounded


__all__ = ["round_real_to_float32", "round_real_to_float32_with_ratio"]
