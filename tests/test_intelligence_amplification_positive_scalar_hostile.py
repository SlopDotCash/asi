# mypy: disable-error-code="attr-defined,call-arg,override"
"""``_positive_float32_scalar`` must gate by exact type before any comparison.

``ExoCerebellumConfig.step_size`` is validated through
``_positive_float32_scalar``, which historically ran ``value <= 0`` on the
caller-supplied object *before* ``round_real_to_float32`` narrowed it through
its own exact-type gate (see ``alberta_framework/_float32.py::_real_ratio``,
which checks ``type(value)`` against an allow-list before touching any
instance dunder). A ``numbers.Real`` subclass overriding ``__le__``,
``__lt__``, ``__gt__``, ``__ge__``, or ``__bool__`` therefore had its hostile
dunder invoked during "trusted" validation, before the value's type was ever
confirmed safe -- the same class of defect closed file-by-file elsewhere in
this repository's evidence and stream validators (e.g. commit ``a81e159f``,
which hardened ``continual_ia_artifact.py``'s ``_number`` to gate on
``type(value) is int or type(value) is float`` before any conversion).
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import pytest

from alberta_framework.core.intelligence_amplification import ExoCerebellumConfig


class _HostileReal(Real):
    """A ``numbers.Real`` whose comparison/conversion dunders must never run."""

    calls: list[str] = []

    def __init__(self, magnitude: float) -> None:
        self._magnitude = magnitude

    def _record(self, name: str) -> None:
        type(self).calls.append(name)

    def __le__(self, other: object) -> bool:
        self._record("__le__")
        raise AssertionError("hostile __le__ ran")

    def __lt__(self, other: object) -> bool:
        self._record("__lt__")
        raise AssertionError("hostile __lt__ ran")

    def __gt__(self, other: object) -> bool:
        self._record("__gt__")
        raise AssertionError("hostile __gt__ ran")

    def __ge__(self, other: object) -> bool:
        self._record("__ge__")
        raise AssertionError("hostile __ge__ ran")

    def __bool__(self) -> bool:
        self._record("__bool__")
        raise AssertionError("hostile __bool__ ran")

    def __float__(self) -> float:
        self._record("__float__")
        raise AssertionError("hostile __float__ ran")

    def __eq__(self, other: object) -> bool:
        self._record("__eq__")
        return NotImplemented

    def __hash__(self) -> int:
        return 0

    # Remaining abstract members of numbers.Real -- unused by the validator,
    # but required for the ABC to instantiate. Each records and raises too,
    # so *any* touch of the hostile instance's numeric protocol is caught.
    def __abs__(self) -> Any:
        self._record("__abs__")
        raise AssertionError("hostile __abs__ ran")

    def __add__(self, other: object) -> Any:
        self._record("__add__")
        raise AssertionError("hostile __add__ ran")

    def __radd__(self, other: object) -> Any:
        self._record("__radd__")
        raise AssertionError("hostile __radd__ ran")

    def __neg__(self) -> Any:
        self._record("__neg__")
        raise AssertionError("hostile __neg__ ran")

    def __pos__(self) -> Any:
        self._record("__pos__")
        raise AssertionError("hostile __pos__ ran")

    def __mul__(self, other: object) -> Any:
        self._record("__mul__")
        raise AssertionError("hostile __mul__ ran")

    def __rmul__(self, other: object) -> Any:
        self._record("__rmul__")
        raise AssertionError("hostile __rmul__ ran")

    def __truediv__(self, other: object) -> Any:
        self._record("__truediv__")
        raise AssertionError("hostile __truediv__ ran")

    def __rtruediv__(self, other: object) -> Any:
        self._record("__rtruediv__")
        raise AssertionError("hostile __rtruediv__ ran")

    def __floordiv__(self, other: object) -> Any:
        self._record("__floordiv__")
        raise AssertionError("hostile __floordiv__ ran")

    def __rfloordiv__(self, other: object) -> Any:
        self._record("__rfloordiv__")
        raise AssertionError("hostile __rfloordiv__ ran")

    def __mod__(self, other: object) -> Any:
        self._record("__mod__")
        raise AssertionError("hostile __mod__ ran")

    def __rmod__(self, other: object) -> Any:
        self._record("__rmod__")
        raise AssertionError("hostile __rmod__ ran")

    def __pow__(self, other: object) -> Any:
        self._record("__pow__")
        raise AssertionError("hostile __pow__ ran")

    def __rpow__(self, other: object) -> Any:
        self._record("__rpow__")
        raise AssertionError("hostile __rpow__ ran")

    def __trunc__(self) -> Any:
        self._record("__trunc__")
        raise AssertionError("hostile __trunc__ ran")

    def __floor__(self) -> Any:
        self._record("__floor__")
        raise AssertionError("hostile __floor__ ran")

    def __ceil__(self) -> Any:
        self._record("__ceil__")
        raise AssertionError("hostile __ceil__ ran")

    def __round__(self, ndigits: int | None = None) -> Any:
        self._record("__round__")
        raise AssertionError("hostile __round__ ran")


def test_cerebellum_config_rejects_hostile_real_without_touching_its_dunders() -> None:
    _HostileReal.calls.clear()
    with pytest.raises(ValueError, match="step_size"):
        ExoCerebellumConfig(step_size=_HostileReal(0.05))  # type: ignore[arg-type]
    assert _HostileReal.calls == []


def test_cerebellum_config_rejects_hostile_real_reporting_as_positive() -> None:
    """A hostile instance cannot bypass the gate merely by "looking" positive."""
    _HostileReal.calls.clear()
    with pytest.raises(ValueError, match="step_size"):
        ExoCerebellumConfig(step_size=_HostileReal(1.0))  # type: ignore[arg-type]
    assert _HostileReal.calls == []


def test_cerebellum_config_rejects_plain_negative_and_zero_step_size() -> None:
    with pytest.raises(ValueError, match="step_size"):
        ExoCerebellumConfig(step_size=-0.05)
    with pytest.raises(ValueError, match="step_size"):
        ExoCerebellumConfig(step_size=0.0)
    with pytest.raises(ValueError, match="step_size"):
        ExoCerebellumConfig(step_size=-0.0)
