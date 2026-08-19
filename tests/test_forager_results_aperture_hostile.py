"""Hostile string gate for forager_results aperture before hash."""

from __future__ import annotations

from typing import SupportsIndex

import pytest

from alberta_framework.benchmarks.forager_results import (
    _legacy_fov_config_aperture,
    _legacy_fov_display_agent,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq")

    def __hash__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile hash")

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile len")

    def startswith(
        self,
        prefix: str | tuple[str, ...],
        start: SupportsIndex | None = None,
        end: SupportsIndex | None = None,
    ) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile startswith")


def test_aperture_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("Greedy")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        _legacy_fov_config_aperture(hostile)
    assert _HostileStr.calls == 0


def test_aperture_rejects_hostile_dqn_before_startswith() -> None:
    hostile = _HostileStr("DQN-5")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        _legacy_fov_config_aperture(hostile)
    assert _HostileStr.calls == 0


def test_display_agent_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("Greedy")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        _legacy_fov_display_agent(hostile)
    assert _HostileStr.calls == 0


def test_display_agent_rejects_hostile_dqn_before_startswith() -> None:
    hostile = _HostileStr("DQN-5")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        _legacy_fov_display_agent(hostile)
    assert _HostileStr.calls == 0


def test_non_str_types_rejected() -> None:
    for bad in [123, None, True, b"bytes", ["Greedy"]]:
        with pytest.raises(ValueError, match="exact string"):
            _legacy_fov_config_aperture(bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="exact string"):
            _legacy_fov_display_agent(bad)  # type: ignore[arg-type]


def test_benign_still_passes() -> None:
    assert _legacy_fov_config_aperture("Greedy") == 15
    assert _legacy_fov_config_aperture("Random") == 1
    assert _legacy_fov_config_aperture("DQN-5") == 5
    assert _legacy_fov_display_agent("Greedy") == "Search Oracle"
    assert _legacy_fov_display_agent("Greedy-122") == "Search Nearest"
    assert _legacy_fov_display_agent("Random") == "Random"
    assert _legacy_fov_display_agent("DQN-5") == "DQN"
