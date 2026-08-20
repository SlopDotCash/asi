# mypy: disable-error-code="call-arg"
"""HordeSpec rejects oversized demon lists before from_config reconstruction hang.

Origin ``HordeSpec.from_config`` rebuilt every demon dict with no count bound.
A cheap ``[spec] * 400_000`` pointer-repeat took 3.736s on origin/main.
"""

from __future__ import annotations

import time

import pytest

from alberta_framework.core.types import (
    _MAX_HORDE_DEMONS,
    DemonType,
    GVFSpec,
    HordeSpec,
    create_horde_spec,
)

_DEMON_PAYLOAD = {
    "name": "d",
    "demon_type": "prediction",
    "gamma": 0.9,
    "lamda": 0.0,
    "cumulant_index": 0,
    "terminal_reward": 0.0,
}


def _demon() -> GVFSpec:
    return GVFSpec(
        name="d",
        demon_type=DemonType.PREDICTION,
        gamma=0.9,
        lamda=0.0,
        cumulant_index=0,
    )


def test_frozen_demon_count_bound() -> None:
    assert _MAX_HORDE_DEMONS == 4096


def test_last_fit_demon_count_is_accepted() -> None:
    spec = create_horde_spec([_demon()] * _MAX_HORDE_DEMONS)
    assert len(spec.demons) == _MAX_HORDE_DEMONS
    restored = HordeSpec.from_config(
        {"demons": [_DEMON_PAYLOAD] * _MAX_HORDE_DEMONS}
    )
    assert len(restored.demons) == _MAX_HORDE_DEMONS


@pytest.mark.parametrize("count", [4097, 400_000])
def test_from_config_rejects_oversized_demon_list_before_rebuild(count: int) -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="at most 4096"):
        HordeSpec.from_config({"demons": [_DEMON_PAYLOAD] * count})
    assert time.perf_counter() - started < 0.5


def test_create_horde_spec_rejects_oversized_pointer_repeat() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="at most 4096"):
        create_horde_spec([_demon()] * 400_000)
    assert time.perf_counter() - started < 0.5
