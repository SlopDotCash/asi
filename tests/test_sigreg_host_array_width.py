"""Reject wide host arrays before SIGReg walks them for jnp.asarray."""

from __future__ import annotations

from typing import Any, cast

import pytest

from alberta_framework.core.sigreg import (
    _MAX_HOST_ARRAY_CONTAINER_ITEMS,
    _asarray_float32,
    epps_pulley_gaussian_statistic,
)

pytestmark = pytest.mark.unit


def test_wide_host_list_rejects_before_asarray(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _MAX_HOST_ARRAY_CONTAINER_ITEMS == 4096
    calls: list[int] = []

    def spy(value: object, dtype: object = None) -> Any:
        calls.append(1)
        raise AssertionError("jnp.asarray must not run on oversized host lists")

    monkeypatch.setattr("alberta_framework.core.sigreg.jnp.asarray", spy)
    with pytest.raises(
        ValueError,
        match=r"samples length must be an integer in \[0, 4096\]",
    ):
        epps_pulley_gaussian_statistic(
            cast(Any, [0.0] * (_MAX_HOST_ARRAY_CONTAINER_ITEMS + 1))
        )
    assert calls == []


def test_last_fit_host_list_still_converts() -> None:
    array = _asarray_float32(
        [0.0] * _MAX_HOST_ARRAY_CONTAINER_ITEMS, name="samples"
    )
    assert array.shape == (_MAX_HOST_ARRAY_CONTAINER_ITEMS,)
