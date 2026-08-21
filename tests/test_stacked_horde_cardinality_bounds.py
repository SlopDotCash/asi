"""Cardinality bounds for stacked-Horde demon counts (#2225).

Oversized demon counts (> 4096), oversized demon-list sequences, and
oversized nexting_spec demon-grid products must raise ValueError before
any element walk or array materialization.
"""

from __future__ import annotations

import pytest

from alberta_framework.core.stacked_horde import (
    StackedHordeConfig,
    _MAX_STACKED_HORDE_DEMONS,
    _decode_sequence,
    nexting_spec,
)


class _HostileList(list):
    calls = 0

    def __len__(self) -> int:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("hostile __len__ must not be called")

    def __iter__(self):  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("hostile __iter__ must not be called")


def test_max_constant_is_4096() -> None:
    assert _MAX_STACKED_HORDE_DEMONS == 4096


def test_config_rejects_oversized_n_demons() -> None:
    with pytest.raises(ValueError, match="n_demons must be in \\[1, 4096\\]"):
        StackedHordeConfig(
            n_demons=4097,
            feature_dim=4,
            gammas=(0.5,),
            lamdas=(0.7,),
            cumulant_indices=(0,),
        )


def test_config_accepts_boundary_n_demons() -> None:
    cfg = StackedHordeConfig(
        n_demons=4096,
        feature_dim=4,
        gammas=tuple([0.5] * 4096),
        lamdas=tuple([0.7] * 4096),
        cumulant_indices=tuple([0] * 4096),
    )
    assert cfg.n_demons == 4096


def test_decode_sequence_rejects_oversized() -> None:
    with pytest.raises(
        ValueError,
        match=r"gammas must contain at most 4096 elements",
    ):
        _decode_sequence("gammas", [0.5] * (4096 + 1))


def test_decode_sequence_rejects_hostile_subclass() -> None:
    with pytest.raises(ValueError, match="must be an actual list or tuple"):
        _decode_sequence("gammas", _HostileList([0.5]))


def test_nexting_spec_rejects_oversized_product() -> None:
    with pytest.raises(ValueError, match="exceeds 4096 demons"):
        nexting_spec(feature_dim=4, cumulant_indices=tuple(range(100)), gammas=tuple([0.5] * 50))


def test_nexting_spec_accepts_boundary_product() -> None:
    cfg = nexting_spec(feature_dim=4, cumulant_indices=tuple(range(1024)), gammas=(0.5, 0.9, 0.99, 0.999))
    assert cfg.n_demons == 4096
