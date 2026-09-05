"""Cardinality preflights for stacked-horde demon count and sequences."""

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

    def __iter__(self):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        raise AssertionError("list iterator hook executed")


def _last_fit_config() -> StackedHordeConfig:
    n = _MAX_STACKED_HORDE_DEMONS
    return StackedHordeConfig(
        n_demons=n,
        feature_dim=1,
        gammas=(0.9,) * n,
        lamdas=(0.8,) * n,
        cumulant_indices=(0,) * n,
    )


def test_last_fit_demon_count_is_accepted() -> None:
    cfg = _last_fit_config()
    assert cfg.n_demons == _MAX_STACKED_HORDE_DEMONS


def test_rejects_oversized_n_demons_before_sequence_walk() -> None:
    calls = 0

    class HostileFloat:
        def __float__(self) -> float:
            nonlocal calls
            calls += 1
            raise AssertionError("oversized n_demons walked a sequence element")

    hostile: Any = HostileFloat()
    with pytest.raises(ValueError, match="4096"):
        StackedHordeConfig(
            n_demons=4097,
            feature_dim=1,
            gammas=(hostile,),  # type: ignore[arg-type]
            lamdas=(0.8,),
            cumulant_indices=(0,),
        )
    assert calls == 0


def test_from_config_rejects_oversized_lists_before_tuple_copy() -> None:
    payload = StackedHordeConfig(
        n_demons=1,
        feature_dim=1,
        gammas=(0.9,),
        lamdas=(0.8,),
        cumulant_indices=(0,),
    ).to_config()
    payload["gammas"] = [0.9] * 4097
    with pytest.raises(ValueError, match="at most 4096"):
        StackedHordeConfig.from_config(payload)


def test_from_config_rejects_list_subclasses_before_length_hooks() -> None:
    payload = StackedHordeConfig(
        n_demons=1,
        feature_dim=1,
        gammas=(0.9,),
        lamdas=(0.8,),
        cumulant_indices=(0,),
    ).to_config()
    payload["gammas"] = _HostileList()
    _HostileList.calls = 0
    with pytest.raises(ValueError, match="actual list or tuple"):
        StackedHordeConfig.from_config(payload)
    assert _HostileList.calls == 0


def test_last_fit_nexting_product_is_accepted() -> None:
    cfg = nexting_spec(
        feature_dim=1,
        cumulant_indices=tuple(range(64)),
        gammas=(0.0,) * 64,
    )
    assert cfg.n_demons == 4096


def test_nexting_spec_rejects_product_before_expanding_or_walking() -> None:
    calls = 0

    class HostileFloat:
        def __float__(self) -> float:
            nonlocal calls
            calls += 1
            raise AssertionError("oversized nexting product walked a gamma")

    hostile: Any = HostileFloat()
    with pytest.raises(ValueError, match="at most 4096"):
        nexting_spec(
            feature_dim=1,
            cumulant_indices=tuple(range(65)),
            gammas=(hostile,) * 64,  # type: ignore[arg-type]
        )
    assert calls == 0


def test_nexting_spec_rejects_oversized_sequence_before_per_item_walk() -> None:
    calls = 0

    class HostileFloat:
        def __float__(self) -> float:
            nonlocal calls
            calls += 1
            raise AssertionError("oversized gamma tuple walked an element")

    hostile: Any = HostileFloat()
    with pytest.raises(ValueError, match="at most 4096"):
        nexting_spec(
            feature_dim=1,
            cumulant_indices=(0,),
            gammas=(hostile,) * 4097,  # type: ignore[arg-type]
        )
    assert calls == 0
