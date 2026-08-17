"""Shared validation policy for Step facade scalars consumed as float32."""

from __future__ import annotations

import math
from numbers import Real
from typing import cast

from alberta_framework._float32 import round_real_to_float32_with_ratio


def _require_exact_str(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def finite_real_and_float32(name: object, value: object) -> tuple[Real, int, int, float]:
    """Return the original real, exact ratio, and finite binary32 rounding."""
    host_name = _require_exact_str("name", name)
    actual_type = type(value)
    if issubclass(actual_type, bool) or not issubclass(actual_type, Real):
        raise ValueError(f"{host_name} must be a real number")
    real = cast(Real, value)
    try:
        numerator, denominator, narrowed = round_real_to_float32_with_ratio(real)
    except Exception:
        raise ValueError(f"{host_name} must be finite") from None
    if not math.isfinite(narrowed):
        raise ValueError(f"{host_name} must be finite")
    return real, numerator, denominator, narrowed


def canonical_float32_storage(value: Real, narrowed: float) -> float:
    """Canonicalize non-builtins while preserving builtin JSON float values.

    A builtin float is already an exact binary64 input, so the eventual JAX
    conversion performs the same single binary32 rounding validated here.
    Exact and third-party ``Real`` implementations are stored as the already
    rounded builtin float, avoiding both double rounding and non-JSON scalars.
    """
    if type(cast(object, value)) is float:
        return float(value)
    return narrowed


__all__ = ["canonical_float32_storage", "finite_real_and_float32"]
