"""ArrayValue and SpaceSpec reject oversized rank before walking axes."""

from __future__ import annotations

import time

import pytest

from alberta_framework.reference_agent import _MAX_ARRAY_RANK, ArrayValue, SpaceSpec

pytestmark = pytest.mark.unit


def test_array_value_rejects_oversized_rank_before_axis_walk() -> None:
    assert _MAX_ARRAY_RANK == 8
    started = time.perf_counter()
    with pytest.raises(ValueError, match="ArrayValue rank must be <= 8"):
        ArrayValue(
            semantic_id="a.b",
            dtype="float32",
            shape=(1,) * (_MAX_ARRAY_RANK + 1),
            payload=b"",
        )
    assert time.perf_counter() - started < 0.25


def test_space_spec_rejects_oversized_rank_before_axis_walk() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="space rank must be <= 8"):
        SpaceSpec(
            kind="box",
            shape=(1,) * (_MAX_ARRAY_RANK + 1),
            dtype="float32",
            semantic_id="a.b",
        )
    assert time.perf_counter() - started < 0.25
