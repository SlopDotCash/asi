"""Cardinality preflights for StackedLinearHorde demons and sequences."""

from __future__ import annotations

from typing import Any

import pytest

from alberta_framework.core.stacked_horde import (
    _MAX_STACKED_HORDE_DEMONS,
    StackedHordeConfig,
    nexting_spec,
)


class _HostileList(list[object]):
    calls = 0

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("list length hook executed")


def test_documented_protocol_ceilings() -> None:
    assert _MAX_STACKED_HORDE_DEMONS == 4096


def test_last_fit_demon_count_is_accepted() -> None:
    config = StackedHordeConfig(
        n_demons=_MAX_STACKED_HORDE_DEMONS,
        feature_dim=1,
        gammas=(0.9,) * _MAX_STACKED_HORDE_DEMONS,
        lamdas=(0.5,) * _MAX_STACKED_HORDE_DEMONS,
        cumulant_indices=(0,) * _MAX_STACKED_HORDE_DEMONS,
        step_size=0.05,
    )
    assert config.n_demons == _MAX_STACKED_HORDE_DEMONS
    assert len(config.gammas) == _MAX_STACKED_HORDE_DEMONS


def test_rejects_oversized_demon_count_before_element_walk() -> None:
    calls = 0

    class HostileFloat:
        def __float__(self) -> float:
            nonlocal calls
            calls += 1
            raise AssertionError("oversized sequence walked an element")

    hostile: Any = HostileFloat()
    with pytest.raises(ValueError, match="n_demons"):
        StackedHordeConfig(
            n_demons=_MAX_STACKED_HORDE_DEMONS + 1,
            feature_dim=1,
            gammas=(hostile,) * (_MAX_STACKED_HORDE_DEMONS + 1),
            lamdas=(0.5,) * (_MAX_STACKED_HORDE_DEMONS + 1),
            cumulant_indices=(0,) * (_MAX_STACKED_HORDE_DEMONS + 1),
            step_size=0.05,
        )
    assert calls == 0


@pytest.mark.parametrize(
    "field",
    [
        "gammas",
        "lamdas",
        "cumulant_indices",
    ],
)
def test_from_config_rejects_oversized_lists_before_tuple_copy(field: str) -> None:
    base = StackedHordeConfig(
        n_demons=1,
        feature_dim=1,
        gammas=(0.9,),
        lamdas=(0.5,),
        cumulant_indices=(0,),
    ).to_config()
    base[field] = [0.5 if field != "cumulant_indices" else 0] * (_MAX_STACKED_HORDE_DEMONS + 1)
    with pytest.raises(ValueError, match="at most 4096"):
        StackedHordeConfig.from_config(base)


@pytest.mark.parametrize(
    "field",
    [
        "gammas",
        "lamdas",
        "cumulant_indices",
    ],
)
def test_from_config_rejects_list_subclasses_before_length_hooks(field: str) -> None:
    base = StackedHordeConfig(
        n_demons=1,
        feature_dim=1,
        gammas=(0.9,),
        lamdas=(0.5,),
        cumulant_indices=(0,),
    ).to_config()
    base[field] = _HostileList()
    _HostileList.calls = 0
    with pytest.raises(ValueError, match="actual list or tuple"):
        StackedHordeConfig.from_config(base)
    assert _HostileList.calls == 0


def test_nexting_spec_rejects_oversized_combinations() -> None:
    with pytest.raises(ValueError, match="demon count"):
        nexting_spec(
            feature_dim=1,
            cumulant_indices=tuple(range(1025)),
            gammas=(0.0, 0.5, 0.9, 0.99),  # 1025 * 4 = 4100 > 4096
        )
