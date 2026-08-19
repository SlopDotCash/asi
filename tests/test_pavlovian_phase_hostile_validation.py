"""Standalone Pavlovian phase trust-boundary validation."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from alberta_framework.streams.pavlovian import (
    PavlovianPhase,
    partial_reinforcement_scenario,
)


def _phase(**overrides: Any) -> PavlovianPhase:
    values: dict[str, Any] = {
        "name": "acquisition",
        "n_steps": 10,
        "cs_us_contingency": 1.0,
        "cs_active": (0,),
        "compound_index": -1,
    }
    values.update(overrides)
    return PavlovianPhase(**values)


class _HostileFloat(float):
    def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
        raise AssertionError("numeric hook must not run")


@pytest.mark.parametrize("cs_active", [(), (0, 0), (True,), (-1,)])
def test_phase_rejects_invalid_local_cs_sets(cs_active: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="cs_active"):
        _phase(cs_active=cs_active)


def test_phase_rejects_hostile_values_without_numeric_hooks() -> None:
    with pytest.raises(ValueError, match="cs_us_contingency"):
        _phase(cs_us_contingency=_HostileFloat(0.5))


def test_phase_preserves_supported_exact_fraction_probability() -> None:
    phase = _phase(cs_us_contingency=Fraction(1, 2))
    assert phase.cs_us_contingency == Fraction(1, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", ""),
        ("n_steps", True),
        ("n_steps", 0),
        ("cs_active", [0]),
        ("compound_index", True),
        ("compound_index", -2),
    ],
)
def test_phase_rejects_noncanonical_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _phase(**{field: value})


class _HostileFormatReal:
    """A ``numbers.Real`` registrant whose format/str hooks must never run.

    ``partial_reinforcement_scenario`` rejects this in
    ``_require_unit_interval`` (it is not an exact allow-listed real type),
    which itself never touches the value in its error message. The
    surrounding ``except ValueError`` handler must preserve that guarantee
    instead of re-interpolating the still-untrusted original object.
    """

    def __format__(self, spec: str) -> str:  # pragma: no cover
        raise AssertionError("hostile __format__ must not run")

    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("hostile __str__ must not run")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("hostile __repr__ must not run")


def test_partial_reinforcement_scenario_rejects_hostile_p_without_format_hook() -> None:
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        partial_reinforcement_scenario(p=_HostileFormatReal())
