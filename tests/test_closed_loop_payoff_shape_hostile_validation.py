"""Standalone trust-boundary validation for closed-loop payoff shape checks.

``SwitchingTwoStateConfig.__post_init__`` canonicalizes ``payoffs_a`` and
``payoffs_b`` through ``closed_loop._canonical_two_state_payoffs``, which reads
shape metadata off the raw, still-untrusted field value with
``np.shape(raw_payoff)`` before any dimension's type is confirmed safe. A
hostile object whose ``.shape`` property returns a tuple containing a hostile
int-like element can therefore have its ``__eq__``/``__ne__`` invoked by the
subsequent ``payoff_shape != (2, 2)`` comparison before that element's type is
ever checked. This mirrors the defect class already closed in
``gymnasium.py::_require_runtime_shape`` (PR #1998) and
``closed_loop.py::_phase_payoff_np`` (PR #1995).
"""

from __future__ import annotations

import pytest

from alberta_framework.streams import SwitchingTwoStateConfig


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


class _HostilePayoffShapeHolder:
    """Reports a hostile ``.shape`` without being a trusted array/sequence type."""

    def __init__(self, dimension: object) -> None:
        self._dimension = dimension

    @property
    def shape(self) -> tuple[object, object]:
        return (self._dimension, 2)


def test_switching_two_state_config_rejects_hostile_payoff_dimension_without_eq_hook() -> None:
    with pytest.raises(ValueError, match="built-in integers"):
        SwitchingTwoStateConfig(payoffs_a=_HostilePayoffShapeHolder(_HostileDimension()))


def test_switching_two_state_config_rejects_hostile_payoff_dimension_without_repr_hook() -> None:
    with pytest.raises(ValueError, match="built-in integers"):
        SwitchingTwoStateConfig(
            payoffs_b=_HostilePayoffShapeHolder(_HostileNonRaisingDimension())
        )


def test_switching_two_state_config_still_accepts_plain_payoffs() -> None:
    config = SwitchingTwoStateConfig(
        payoffs_a=((0.0, 1.0), (1.0, 0.0)), payoffs_b=((1.0, 0.0), (0.0, 1.0))
    )
    assert config.payoffs_a == ((0.0, 1.0), (1.0, 0.0))
    assert config.payoffs_b == ((1.0, 0.0), (0.0, 1.0))


def test_switching_two_state_config_still_rejects_wrong_shape_payoffs() -> None:
    with pytest.raises(ValueError, match="2x2"):
        SwitchingTwoStateConfig(payoffs_a=((0.0, 1.0, 2.0), (1.0, 0.0, 3.0)))
