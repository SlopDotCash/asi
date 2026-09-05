"""Validation hardening for sparse initializers."""

from __future__ import annotations

from typing import NoReturn

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.initializers import sparse_init

_INT32_MAX = 2**31 - 1


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr executed")


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:  # type: ignore[override]
        type(self).calls += 1
        raise RuntimeError("ratio hook")


class _ClassSpoof:
    @property  # type: ignore[misc]
    def __class__(self) -> type:  # type: ignore[no-untyped-def]
        return float

    def __float__(self) -> float:  # pragma: no cover
        return 0.5


def test_sparse_init_rejects_hostile_int_shape_without_hook() -> None:
    key = jr.key(0)
    with pytest.raises(ValueError, match="must be an integer"):
        sparse_init(key, (_HostileInt(4), 4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        sparse_init(key, (4, _HostileInt(4)))  # type: ignore[arg-type]


def test_sparse_init_rejects_hostile_repr_shape() -> None:
    key = jr.key(0)
    with pytest.raises(ValueError):
        sparse_init(key, (_RaisingRepr(), 4))  # type: ignore[arg-type]


def test_sparse_init_rejects_non_integer_shape_and_out_of_range() -> None:
    key = jr.key(0)
    for bad_shape in [
        (True, 4),
        (4.0, 4),
        ("4", 4),
        (0, 4),
        (-1, 4),
        (_INT32_MAX + 1, 4),
        (4, 0),
        (4, -1),
    ]:
        with pytest.raises(ValueError, match="must be"):
            sparse_init(key, bad_shape)  # type: ignore[arg-type]


def test_sparse_init_numpy_int_canonicalizes() -> None:
    key = jr.key(0)
    for np_type in [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    ]:
        weights = sparse_init(key, (np_type(4), np_type(4)))  # type: ignore[arg-type]
        assert weights.shape == (4, 4)


def test_sparse_init_rejects_hostile_float_sparsity() -> None:
    key = jr.key(0)
    with pytest.raises(ValueError, match="sparsity"):
        sparse_init(key, (4, 4), sparsity=_HostileFloat(0.5))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    # The important check is that type gate prevents as_integer_ratio
    class HostileFloat2(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:  # type: ignore[override]
            type(self).calls += 1
            raise RuntimeError("hook")

    with pytest.raises(ValueError, match="sparsity"):
        sparse_init(key, (4, 4), sparsity=HostileFloat2(0.5))  # type: ignore[arg-type]
    assert HostileFloat2.calls == 0


def test_sparse_init_float_validators_reject_class_spoof_and_out_of_range() -> None:
    key = jr.key(0)
    for bad in [float("nan"), float("inf"), -0.1, 1.5, _ClassSpoof()]:  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="sparsity"):
            sparse_init(key, (4, 4), sparsity=bad)  # type: ignore[arg-type]


def test_sparse_init_rejects_subnormal_sparsity() -> None:
    key = jr.key(0)
    # 1e-45 flushes to zero in float32
    with pytest.raises(ValueError, match="subnormal"):
        sparse_init(key, (4, 4), sparsity=1e-45)  # type: ignore[arg-type]
    # Normal small value passes
    weights = sparse_init(key, (4, 4), sparsity=1e-6)
    assert weights.shape == (4, 4)


def test_sparse_init_accepts_valid_edge_sparsity() -> None:
    key = jr.key(0)
    w0 = sparse_init(key, (4, 4), sparsity=0.0)
    assert w0.shape == (4, 4)
    w1 = sparse_init(key, (4, 4), sparsity=1.0)
    assert jnp.all(w1 == 0.0)


def test_sparse_init_rejects_init_type_hostile_and_wrong() -> None:
    key = jr.key(0)
    with pytest.raises(ValueError, match="init_type"):
        sparse_init(key, (4, 4), init_type=_RaisingRepr())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="init_type"):
        sparse_init(key, (4, 4), init_type="bad")  # type: ignore[arg-type]
    # String subclass should be rejected (exact str)
    class StringSubclass(str):
        pass

    with pytest.raises(ValueError, match="init_type"):
        sparse_init(key, (4, 4), init_type=StringSubclass("uniform"))  # type: ignore[arg-type]


def test_sparse_init_accepts_valid_init_types() -> None:
    key = jr.key(0)
    w_uniform = sparse_init(key, (4, 4), init_type="uniform")
    w_normal = sparse_init(key, (4, 4), init_type="normal")
    assert w_uniform.shape == (4, 4)
    assert w_normal.shape == (4, 4)


def test_sparse_init_preflight_without_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    class AllocationReachedError(Exception):
        pass

    requested_shapes: list[tuple[int, int]] = []

    def stop_at_allocation(
        key: object, shape: tuple[int, int], **kwargs: object
    ) -> NoReturn:
        requested_shapes.append(shape)
        assert kwargs["dtype"] == jnp.float32
        raise AllocationReachedError

    # Test the byte boundary without constructing a roughly 2 GiB matrix and
    # its initialization intermediates. Real small-shape initialization stays
    # covered by the tests above; this test isolates the allocation preflight.
    monkeypatch.setattr(jr, "uniform", stop_at_allocation)
    key = jr.key(0)
    max_elements = _INT32_MAX // np.dtype(np.float32).itemsize
    for invalid_shape in ((_INT32_MAX, 2), (50_000, 50_000), (1, max_elements + 1)):
        with pytest.raises(ValueError, match="byte count"):
            sparse_init(key, invalid_shape)
    assert requested_shapes == []

    # The exact last representable element count passes validation and reaches
    # the allocator with the requested shape, while one more was rejected.
    last_fit_shape = (1, max_elements)
    with pytest.raises(AllocationReachedError):
        sparse_init(key, last_fit_shape)
    assert requested_shapes == [last_fit_shape]
