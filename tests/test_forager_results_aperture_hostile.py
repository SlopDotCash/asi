"""Hostile string gate for forager_results aperture before hash."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_results import _legacy_fov_config_aperture

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash")

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile len")

    def startswith(self, prefix: object, *args, **kwargs) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile startswith")


def test_aperture_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("Greedy")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        _legacy_fov_config_aperture(hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_aperture_rejects_hostile_dqn_before_startswith() -> None:
    hostile = _HostileStr("DQN-5")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        _legacy_fov_config_aperture(hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_benign_still_passes() -> None:
    assert _legacy_fov_config_aperture("Greedy") == 15
    assert _legacy_fov_config_aperture("Random") == 1
    assert _legacy_fov_config_aperture("DQN-5") == 5
