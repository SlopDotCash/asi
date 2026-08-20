"""SpaceSpec.box rejects oversized sequences before tuple() hang."""

from __future__ import annotations

import time

import pytest

from alberta_framework.reference_agent import _MAX_ARRAY_ELEMENTS, _MAX_ARRAY_RANK, SpaceSpec

pytestmark = pytest.mark.unit


def test_box_rejects_oversized_range_bounds_before_tuple_hang() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="array element limit"):
        SpaceSpec.box(
            shape=(2,),
            dtype="float32",
            low=range(_MAX_ARRAY_ELEMENTS + 1),
            high=range(_MAX_ARRAY_ELEMENTS + 1),
            semantic_id="obs",
        )
    assert time.perf_counter() - started < 0.25


def test_box_rejects_oversized_range_shape_before_tuple_hang() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="rank"):
        SpaceSpec.box(
            shape=range(_MAX_ARRAY_RANK + 1_000_000),  # type: ignore[arg-type]
            dtype="float32",
            low=None,
            high=None,
            semantic_id="obs",
        )
    assert time.perf_counter() - started < 0.25


def test_box_encode_rejects_oversized_range_before_numpy_convert() -> None:
    spec = SpaceSpec.box(
        shape=(2,),
        dtype="float32",
        low=None,
        high=None,
        semantic_id="obs",
    )
    started = time.perf_counter()
    with pytest.raises(ValueError, match="array element limit"):
        spec.encode(range(_MAX_ARRAY_ELEMENTS + 1))
    assert time.perf_counter() - started < 0.25


def test_box_still_accepts_matched_bounds() -> None:
    spec = SpaceSpec.box(
        shape=(2,),
        dtype="float32",
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        semantic_id="obs",
    )
    assert spec.low == (-1.0, -1.0)
    assert spec.high == (1.0, 1.0)
    encoded = spec.encode((0.0, 0.5))
    assert encoded.shape == (2,)
