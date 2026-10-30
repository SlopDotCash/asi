"""Shared host-side resource contracts for bounded JAX loops.

These checks run before ``arange`` or ``lax.scan`` tracing.  A signed-integer
shape bound is not a compute budget: callers declare the largest supported
step count and, when work is batched, the largest supported product.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax


@dataclass(frozen=True, slots=True)
class ScanBudget:
    """One named, auditable host-side scan budget."""

    label: str
    maximum_steps: int
    maximum_parallel: int = 1
    maximum_step_units: int | None = None

    def __post_init__(self) -> None:
        if type(self.label) is not str or not self.label:
            raise ValueError("scan budget label must be a nonempty exact string")
        for name, value in (
            ("maximum_steps", self.maximum_steps),
            ("maximum_parallel", self.maximum_parallel),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        units = self.maximum_step_units
        if units is not None and (type(units) is not int or units < 1):
            raise ValueError("maximum_step_units must be a positive exact integer")


def require_scan_steps(name: str, value: object, budget: ScanBudget) -> int:
    """Return an exact host step count within ``budget`` without coercion."""

    if type(name) is not str or not name:
        raise ValueError("scan step name must be a nonempty exact string")
    if type(value) is not int or not 1 <= value <= budget.maximum_steps:
        raise ValueError(
            f"{name} must be an integer in [1, {budget.maximum_steps}] "
            f"for the {budget.label} budget"
        )
    return value


def require_parallel_count(name: str, value: object, budget: ScanBudget) -> int:
    """Return an exact batch/seed count within ``budget`` without coercion."""

    if type(name) is not str or not name:
        raise ValueError("parallel count name must be a nonempty exact string")
    if type(value) is not int or not 1 <= value <= budget.maximum_parallel:
        raise ValueError(
            f"{name} must be an integer in [1, {budget.maximum_parallel}] "
            f"for the {budget.label} budget"
        )
    return value


def require_step_units(steps: int, parallel: int, budget: ScanBudget) -> None:
    """Reject a valid-axis combination whose total declared work is too large."""

    maximum = budget.maximum_step_units
    if maximum is not None and steps * parallel > maximum:
        raise ValueError(
            f"{budget.label} step-units exceed the declared resource budget ({maximum})"
        )


def require_jax_leading_length(
    name: str,
    value: object,
    budget: ScanBudget,
    *,
    ranks: tuple[int, ...] | None = None,
) -> int:
    """Validate a real JAX array and return its bounded leading dimension."""

    if type(name) is not str or not name:
        raise ValueError("name must be a nonempty exact string")
    if not isinstance(value, jax.Array):
        raise TypeError(f"{name} must be a JAX array")
    if ranks is not None and value.ndim not in ranks:
        raise ValueError(f"{name} must have rank in {ranks}")
    return require_scan_steps(f"{name} length", value.shape[0], budget)


def require_matching_jax_leading_length(
    name: str,
    value: object,
    *,
    expected: int,
) -> None:
    """Require a real JAX array with the already-bounded leading dimension."""

    if type(name) is not str or not name:
        raise ValueError("name must be a nonempty exact string")
    if not isinstance(value, jax.Array):
        raise TypeError(f"{name} must be a JAX array")
    if value.shape[0] != expected:
        raise ValueError(f"{name} must share the primary sequence length")
