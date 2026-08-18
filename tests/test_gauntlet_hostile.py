"""Hostile int/float gate for GauntletConfig before float/range."""

from __future__ import annotations

import pytest

from alberta_framework.streams.gauntlet import GauntletConfig

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile lt must not run")

    def __mod__(self, other: object) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile mod must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_gauntlet_rejects_hostile_int_before_range() -> None:
    hostile = _HostileInt(8)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="relevant_dim must be an even integer"):
        GauntletConfig(relevant_dim=hostile)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0

    hostile_seg = _HostileInt(3000)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="segment_length must be positive"):
        GauntletConfig(segment_length=hostile_seg)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0

    # bool must be rejected without dispatch
    with pytest.raises(ValueError, match="relevant_dim must be an even integer"):
        GauntletConfig(relevant_dim=True)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_gauntlet_rejects_hostile_float_before_float() -> None:
    hostile = _HostileFloat(0.1)
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="noise_std must be a finite real number"):
        GauntletConfig(noise_std=hostile)  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0

    hostile_int = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="noise_std must be a finite real number"):
        GauntletConfig(noise_std=hostile_int)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0

    # bool subclass
    with pytest.raises(ValueError, match="noise_std must be a finite real number"):
        GauntletConfig(noise_std=True)  # type: ignore[arg-type]


def test_gauntlet_benign_still_works() -> None:
    cfg = GauntletConfig()
    assert cfg.relevant_dim == 8
    assert cfg.noise_std == 0.1
    # benign exact int/float
    cfg2 = GauntletConfig(
        relevant_dim=4,
        noise_std=0.2,
        feature_std=1.0,
        scale_factor=5.0,
        drift_rate=0.02,
        context_noise_std=0.1,
    )
    assert cfg2.relevant_dim == 4
    # finite check

    with pytest.raises(ValueError, match="must be finite"):
        GauntletConfig(noise_std=float("inf"))  # type: ignore[arg-type]


def test_gauntlet_hostile_not_in_repr() -> None:
    hostile = _HostileInt(8)
    _HostileInt.calls = 0
    try:
        GauntletConfig(relevant_dim=hostile)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "_HostileInt" not in str(exc)
        assert _HostileInt.calls == 0
    else:
        raise AssertionError("should have raised")
