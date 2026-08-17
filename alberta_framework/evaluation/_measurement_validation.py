"""Shared exact validation for public evaluation measurement records."""

from __future__ import annotations

import math
from typing import cast


def finite_real(name: str, value: object) -> float:
    """Return a finite builtin float without accepting facade identities."""

    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite real number")
    numeric = float(cast("int | float", value))
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite real number")
    return numeric


def nonnegative_finite_real(name: str, value: object) -> float:
    """Return one finite, non-negative measurement."""

    numeric = finite_real(name, value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def validate_interval_bounds(
    *, lower: float, upper: float, confidence_level: float
) -> None:
    """Validate ordering and the open-unit confidence domain."""

    if lower > upper:
        raise ValueError("lower must not exceed upper")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")


__all__ = ["finite_real", "nonnegative_finite_real", "validate_interval_bounds"]
