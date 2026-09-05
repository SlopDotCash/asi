"""ArrayValue and SpaceSpec reject oversized rank before walking axes."""

from __future__ import annotations

import pytest

from alberta_framework.reference_agent import _MAX_ARRAY_RANK, ArrayValue, SpaceSpec

pytestmark = pytest.mark.unit


def test_array_value_rejects_oversized_rank_before_axis_walk() -> None:
    assert _MAX_ARRAY_RANK == 8
    with pytest.raises(ValueError, match="ArrayValue rank must be <= 8"):
        ArrayValue(
            semantic_id="a.b",
            dtype="float32",
            shape=(0,) + (1,) * _MAX_ARRAY_RANK,
            payload=b"",
        )


def test_space_spec_rejects_oversized_rank_before_axis_walk() -> None:
    with pytest.raises(ValueError, match="space rank must be <= 8"):
        SpaceSpec(
            kind="box",
            shape=(0,) + (1,) * _MAX_ARRAY_RANK,
            dtype="float32",
            semantic_id="a.b",
        )
