"""Shared validation for configuration scalars that are consumed as float32.

A configuration scalar is checked in both domains that matter: the exact host
value (so a lying ``as_integer_ratio`` or a value that only *rounds* into
range cannot pass) and its binary32 rounding (so a host-finite value that
narrows to infinity, zero, or the excluded end of a half-open interval is
refused before it can freeze an EMA or divide by zero at the sink).  Only an
actual built-in ``float`` is stored as-is — JAX narrows it once, exactly as
validated here — while ints and other reals are stored as the validated
binary32 value.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, cast

from alberta_framework._float32 import round_real_to_float32_with_ratio


def validated_float32_scalar(
    name: str,
    value: object,
    *,
    positive: bool = False,
    lower: float | None = None,
    upper: float | None = None,
    upper_inclusive: bool = True,
) -> float:
    """Return the canonical stored value of one float32-consumed scalar or fail closed.

    Raises:
        ValueError: If ``value`` is not an actual non-bool real, does not narrow
            to a finite binary32, or leaves the declared domain either as the
            exact host value or once narrowed to binary32.
    """
    actual_type = type(value)
    if issubclass(actual_type, bool) or not issubclass(actual_type, Real):
        raise ValueError(f"{name} must be a finite real number")
    real = cast(Any, value)
    try:
        numerator, denominator, narrowed = round_real_to_float32_with_ratio(real)
    except Exception as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(narrowed):
        raise ValueError(f"{name} must remain finite once narrowed to float32")

    def narrowed_in_domain(candidate: float) -> bool:
        if positive and candidate <= 0.0:
            return False
        if lower is not None and candidate < lower:
            return False
        if upper is not None:
            if upper_inclusive:
                return bool(candidate <= upper)
            return bool(candidate < upper)
        return True

    def ratio_compares_to(bound: float) -> int:
        bound_numerator, bound_denominator = bound.as_integer_ratio()
        left = numerator * bound_denominator
        right = bound_numerator * denominator
        return (left > right) - (left < right)

    def exact_in_domain() -> bool:
        if positive and numerator <= 0:
            return False
        if lower is not None and ratio_compares_to(lower) < 0:
            return False
        if upper is not None:
            comparison = ratio_compares_to(upper)
            if comparison > 0 or (comparison == 0 and not upper_inclusive):
                return False
        return True

    domain = _describe_domain(positive, lower, upper, upper_inclusive)
    if not exact_in_domain():
        raise ValueError(f"{name} must be {domain}")
    if not narrowed_in_domain(narrowed):
        raise ValueError(f"{name} must remain {domain} once narrowed to float32")
    return real if type(real) is float else narrowed


def _describe_domain(
    positive: bool, lower: float | None, upper: float | None, upper_inclusive: bool
) -> str:
    if upper is not None:
        floor = lower if lower is not None else "-inf"
        bracket = "]" if upper_inclusive else ")"
        return f"in [{floor}, {upper}{bracket}"
    if positive:
        return "positive"
    if lower is not None:
        return f">= {lower}"
    return "finite"


__all__ = ["validated_float32_scalar"]
