"""Hostile integer gate for micro_continual finite-real validation."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.micro_continual import _require_finite_real

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        return int.__hash__(self)


class _HostileMeta(type):
    calls = 0

    def __eq__(cls, other: object) -> bool:
        del other
        cls.calls += 1
        raise AssertionError("hostile metaclass eq must not run")


class _MetaclassHostileInt(int, metaclass=_HostileMeta):
    pass


def test_require_finite_real_rejects_hostile_before_float() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="finite real number"):
        _require_finite_real(hostile, "x")  # type: ignore[arg-type]
    assert _HostileInt.calls == 0

    _HostileMeta.calls = 0
    with pytest.raises(ValueError, match="finite real number"):
        _require_finite_real(_MetaclassHostileInt(1), "x")  # type: ignore[arg-type]
    assert _HostileMeta.calls == 0

    with pytest.raises(ValueError, match="finite real number"):
        _require_finite_real(True, "x")  # type: ignore[arg-type]

    assert _require_finite_real(1, "x") == 1.0
    assert _require_finite_real(1.0, "x") == 1.0
    assert _require_finite_real(0, "y") == 0.0

    with pytest.raises(ValueError, match="finite real number"):
        _require_finite_real(float("inf"), "x")
    with pytest.raises(ValueError, match="finite real number"):
        _require_finite_real(float("nan"), "x")


def test_validated_curve_rejects_hostile_without_dispatch() -> None:
    from alberta_framework.benchmarks.micro_continual import _validated_curve

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="finite real number"):
        _validated_curve(
            [hostile, 0.5],  # type: ignore[list-item]
            n_regimes=2,
            lower=0.0,
            upper=1.0,
            context="per_regime_accuracy",
        )
    assert _HostileInt.calls == 0
