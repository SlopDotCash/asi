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
from fractions import Fraction
from typing import Any, cast

import numpy as np

from alberta_framework._float32 import round_real_to_float32_with_ratio

_ACTUAL_INT_TYPES = frozenset(
    {
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    }
)
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


def _require_exact_str(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def _require_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a built-in bool")
    return value


def _require_optional_finite_real(
    bound_name: str,
    value: object,
) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{bound_name} must be a finite real number")
    actual_type = type(value)
    if actual_type not in _ALLOWED_REAL_TYPES:
        raise ValueError(f"{bound_name} must be a finite real number")
    if (
        actual_type in _ACTUAL_FLOAT_TYPES
        and actual_type is not Fraction
        and not bool(np.isfinite(cast(Any, value)))
    ):
        raise ValueError(f"{bound_name} must be a finite real number")
    if actual_type in _ACTUAL_INT_TYPES:
        return int(cast(int, value)), 1
    # Every remaining type is an exact, trusted Fraction or NumPy/built-in
    # floating scalar.  Their concrete implementations return an integer pair;
    # subclasses were rejected above before any method lookup.
    numerator, denominator = cast(Any, value).as_integer_ratio()
    return int(numerator), int(denominator)


def validated_float32_scalar(
    name: object,
    value: object,
    *,
    positive: object = False,
    lower: object | None = None,
    upper: object | None = None,
    upper_inclusive: object = True,
) -> float:
    """Return the canonical stored value of one float32-consumed scalar or fail closed.

    Raises:
        ValueError: If ``value`` is not an actual non-bool real, does not narrow
            to a finite binary32, or leaves the declared domain either as the
            exact host value or once narrowed to binary32.
    """
    stored, _, _ = validated_float32_scalar_with_ratio(
        name,
        value,
        positive=positive,
        lower=lower,
        upper=upper,
        upper_inclusive=upper_inclusive,
    )
    return stored


def validated_float32_scalar_with_ratio(
    name: object,
    value: object,
    *,
    positive: object = False,
    lower: object | None = None,
    upper: object | None = None,
    upper_inclusive: object = True,
) -> tuple[float, int, int]:
    """Validate once and also return the exact host numerator and denominator."""
    _require_exact_str("name", name)
    host_name = cast(str, name)
    _require_bool("positive", positive)
    _require_bool("upper_inclusive", upper_inclusive)
    pos = cast(bool, positive)
    upper_inc = cast(bool, upper_inclusive)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{host_name} must be a finite real number")
    if type(value) not in _ALLOWED_REAL_TYPES:
        raise ValueError(f"{host_name} must be a finite real number")
    lower_ratio = _require_optional_finite_real(f"{host_name} lower", lower)
    upper_ratio = _require_optional_finite_real(f"{host_name} upper", upper)
    real = cast(Any, value)
    try:
        numerator, denominator, narrowed = round_real_to_float32_with_ratio(real)
    except Exception as error:
        raise ValueError(f"{host_name} must be a finite real number") from error
    if not math.isfinite(narrowed):
        raise ValueError(f"{host_name} must remain finite once narrowed to float32")

    def narrowed_in_domain(candidate: float) -> bool:
        candidate_numerator, candidate_denominator = candidate.as_integer_ratio()
        if pos and candidate <= 0.0:
            return False
        if lower_ratio is not None and ratio_compares(
            candidate_numerator,
            candidate_denominator,
            lower_ratio,
        ) < 0:
            return False
        if upper_ratio is not None:
            comparison = ratio_compares(
                candidate_numerator,
                candidate_denominator,
                upper_ratio,
            )
            if upper_inc:
                return comparison <= 0
            return comparison < 0
        return True

    def ratio_compares(
        left_numerator: int,
        left_denominator: int,
        right_ratio: tuple[int, int],
    ) -> int:
        right_numerator, right_denominator = right_ratio
        left = left_numerator * right_denominator
        right = right_numerator * left_denominator
        return (left > right) - (left < right)

    def exact_in_domain() -> bool:
        if pos and numerator <= 0:
            return False
        if lower_ratio is not None and ratio_compares(
            numerator,
            denominator,
            lower_ratio,
        ) < 0:
            return False
        if upper_ratio is not None:
            comparison = ratio_compares(numerator, denominator, upper_ratio)
            if comparison > 0 or (comparison == 0 and not upper_inc):
                return False
        return True

    domain = _describe_domain(pos, lower, upper, upper_inc)
    if not exact_in_domain():
        raise ValueError(f"{host_name} must be {domain}")
    if not narrowed_in_domain(narrowed):
        raise ValueError(f"{host_name} must remain {domain} once narrowed to float32")
    stored = real if type(real) is float else narrowed
    return stored, numerator, denominator


def _describe_domain(
    positive: bool,
    lower: object | None,
    upper: object | None,
    upper_inclusive: bool,
) -> str:
    if upper is not None:
        floor: object = lower if lower is not None else "-inf"
        bracket = "]" if upper_inclusive else ")"
        return f"in [{floor}, {upper}{bracket}"
    if positive:
        return "positive"
    if lower is not None:
        return f">= {lower}"
    return "finite"


__all__ = ["validated_float32_scalar", "validated_float32_scalar_with_ratio"]
