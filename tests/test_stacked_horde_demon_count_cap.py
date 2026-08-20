"""Reject oversized stacked-Horde demon counts before per-demon walks hang."""

from __future__ import annotations

import pytest

from alberta_framework.core.stacked_horde import (
    _MAX_STACKED_HORDE_DEMONS,
    StackedHordeConfig,
    nexting_spec,
)


def test_stacked_horde_demon_cap_constant() -> None:
    assert _MAX_STACKED_HORDE_DEMONS == 4096


def test_stacked_horde_accepts_max_demon_count() -> None:
    n = _MAX_STACKED_HORDE_DEMONS
    StackedHordeConfig(
        n_demons=n,
        feature_dim=1,
        gammas=(0.9,) * n,
        lamdas=(0.8,) * n,
        cumulant_indices=(0,) * n,
    )


def test_stacked_horde_rejects_oversized_demon_count() -> None:
    n = _MAX_STACKED_HORDE_DEMONS + 1
    with pytest.raises(ValueError, match="n_demons"):
        StackedHordeConfig(
            n_demons=n,
            feature_dim=1,
            gammas=(0.9,) * n,
            lamdas=(0.8,) * n,
            cumulant_indices=(0,) * n,
        )


def test_nexting_spec_rejects_oversized_derived_demon_count() -> None:
    with pytest.raises(ValueError, match="derived n_demons"):
        nexting_spec(
            feature_dim=1,
            cumulant_indices=tuple(range(65)),
            gammas=tuple(i / 100.0 for i in range(65)),
        )
