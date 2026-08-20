"""Reject deep or cyclic host arrays before jnp.asarray RecursionError."""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
import pytest

from alberta_framework.core.sigreg import (
    _MAX_ARRAY_NESTING_DEPTH,
    _asarray_float32,
    epps_pulley_gaussian_statistic,
)


def _nested_list(depth: int, leaf: Any = 1.0) -> Any:
    value: Any = leaf
    for _ in range(depth):
        value = [value]
    return value


def test_deep_samples_never_reach_asarray(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def spy(value: object, dtype: object = None) -> Any:
        calls.append(1)
        raise AssertionError("jnp.asarray must not run on overflow nests")

    monkeypatch.setattr("alberta_framework.core.sigreg.jnp.asarray", spy)
    with pytest.raises(ValueError, match="nesting depth"):
        epps_pulley_gaussian_statistic(cast(Any, _nested_list(10_000)))
    assert calls == []


def test_last_fit_list_nesting_still_converts() -> None:
    array = _asarray_float32(_nested_list(_MAX_ARRAY_NESTING_DEPTH), name="samples")
    assert array.shape == (1,) * _MAX_ARRAY_NESTING_DEPTH
    assert bool(jnp.isfinite(epps_pulley_gaussian_statistic(array)))


def test_first_overflow_rejects_before_asarray(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def spy(value: object, dtype: object = None) -> Any:
        calls.append(1)
        raise AssertionError("jnp.asarray must not run on overflow nests")

    monkeypatch.setattr("alberta_framework.core.sigreg.jnp.asarray", spy)
    with pytest.raises(ValueError, match="nesting depth"):
        epps_pulley_gaussian_statistic(cast(Any, _nested_list(_MAX_ARRAY_NESTING_DEPTH + 1)))
    assert calls == []


def test_cyclic_list_rejects_before_asarray(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def spy(value: object, dtype: object = None) -> Any:
        calls.append(1)
        raise AssertionError("jnp.asarray must not run on cyclic host arrays")

    monkeypatch.setattr("alberta_framework.core.sigreg.jnp.asarray", spy)
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic host array"):
        epps_pulley_gaussian_statistic(cast(Any, cyclic))
    assert calls == []


def test_vector_samples_still_compute() -> None:
    samples = jnp.ones((32,), dtype=jnp.float32)
    assert bool(jnp.isfinite(epps_pulley_gaussian_statistic(samples)))


def test_recursionerror_from_asarray_is_valueerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(value: object, dtype: object = None) -> Any:
        del value, dtype
        raise RecursionError("simulated host-array overflow")

    monkeypatch.setattr("alberta_framework.core.sigreg.jnp.asarray", boom)
    with pytest.raises(ValueError, match="nesting depth"):
        _asarray_float32(1.0, name="samples")
