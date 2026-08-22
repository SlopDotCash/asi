"""Reject oversized AdamO shape rank before dimension-walk hang."""

from __future__ import annotations

import pytest

from alberta_framework.core.adamo import _MAX_ADAMO_SHAPE_RANK, AdamO


def test_adamo_shape_rank_cap_constant() -> None:
    assert _MAX_ADAMO_SHAPE_RANK == 32


def test_adamo_accepts_max_shape_rank() -> None:
    AdamO().init_for_shape((1,) * _MAX_ADAMO_SHAPE_RANK)


def test_adamo_rejects_oversized_shape_rank() -> None:
    with pytest.raises(ValueError, match="shape length"):
        AdamO().init_for_shape((1,) * (_MAX_ADAMO_SHAPE_RANK + 1))
