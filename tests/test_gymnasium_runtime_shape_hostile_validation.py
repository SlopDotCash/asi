"""Standalone trust-boundary validation for Gymnasium runtime shape checks."""

from __future__ import annotations

import pytest

gymnasium = pytest.importorskip("gymnasium")

from alberta_framework.streams.gymnasium import _require_runtime_shape  # noqa: E402


class _HostileDimension:
    """An object masquerading as a shape dimension with hostile hooks."""

    def __eq__(self, other: object) -> bool:  # pragma: no cover - only if reached
        raise AssertionError("hostile __eq__ must not run")

    def __ne__(self, other: object) -> bool:  # pragma: no cover - only if reached
        raise AssertionError("hostile __ne__ must not run")

    def __repr__(self) -> str:  # pragma: no cover - only if reached
        raise AssertionError("hostile __repr__ must not run")

    def __hash__(self) -> int:
        return 0


class _HostileNonRaisingDimension:
    """A hostile dimension whose comparison quietly returns False."""

    def __eq__(self, other: object) -> bool:
        return False

    def __repr__(self) -> str:  # pragma: no cover - only if reached
        raise AssertionError("hostile __repr__ must not run")

    def __hash__(self) -> int:
        return 0


class _HostileShapeHolder:
    """Reports a hostile ``.shape`` without being a trusted array type."""

    def __init__(self, dimension: object) -> None:
        self._dimension = dimension

    @property
    def shape(self) -> tuple[object, ...]:
        return (self._dimension,)


def test_require_runtime_shape_rejects_hostile_dimension_without_eq_hook() -> None:
    with pytest.raises(ValueError, match="built-in integers"):
        _require_runtime_shape("observation", _HostileShapeHolder(_HostileDimension()), (3,))


def test_require_runtime_shape_rejects_hostile_dimension_without_repr_hook() -> None:
    with pytest.raises(ValueError, match="built-in integers"):
        _require_runtime_shape(
            "action", _HostileShapeHolder(_HostileNonRaisingDimension()), (3,)
        )


def test_require_runtime_shape_accepts_matching_plain_int_shape() -> None:
    _require_runtime_shape("observation", _HostileShapeHolder(3), (3,))


def test_require_runtime_shape_still_rejects_mismatched_plain_int_shape() -> None:
    with pytest.raises(ValueError, match="must have declared shape"):
        _require_runtime_shape("observation", _HostileShapeHolder(4), (3,))
