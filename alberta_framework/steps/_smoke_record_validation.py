"""Shared validation for public Step smoke-result array identities."""

from __future__ import annotations

_INT32_MAX = 2**31 - 1


def require_step_shape(name: str, value: object, *, steps: int) -> tuple[int, ...]:
    """Require one exact, bounded array shape whose leading axis is ``steps``."""

    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a non-empty exact tuple")
    dimensions: list[int] = []
    for index, dimension in enumerate(value):
        if type(dimension) is not int:
            raise ValueError(f"{name}[{index}] must be an exact integer")
        if not 0 <= dimension <= _INT32_MAX:
            raise ValueError(f"{name}[{index}] must lie in [0, {_INT32_MAX}]")
        dimensions.append(dimension)
    if dimensions[0] != steps:
        raise ValueError(f"{name}[0] must equal steps")
    return tuple(dimensions)


__all__ = ["require_step_shape"]
