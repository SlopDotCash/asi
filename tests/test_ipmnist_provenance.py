"""Boundary checks for IPMNIST provenance array identities."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_provenance import array_identity


def test_array_identity_binds_zero_dimensional_shape() -> None:
    scalar = array_identity(np.array(1.0))
    vector = array_identity(np.array([1.0]))
    assert scalar["shape"] == []
    assert vector["shape"] == [1]
    assert scalar != vector


@pytest.mark.parametrize(
    "shape",
    [(0,), (0, 3), (3, 0)],
)
def test_array_identity_binds_empty_arrays(shape: tuple[int, ...]) -> None:
    identity = array_identity(np.zeros(shape, dtype=np.float32))
    assert identity == {
        "shape": list(shape),
        "dtype": "<f4",
        "sha256": hashlib.sha256(b"").hexdigest(),
    }


@pytest.mark.parametrize(
    "array",
    [
        np.array([object()], dtype=object),
        np.zeros(2, dtype=[("a", "<i4"), ("b", "<f4")]),
        np.zeros(2, dtype=[("a", "<f4"), ("b", "<i4")]),
        np.zeros(2, dtype="V8"),
    ],
    ids=["object", "structured-int-first", "structured-float-first", "void"],
)
def test_array_identity_rejects_dtypes_it_cannot_bind(array: np.ndarray) -> None:
    with pytest.raises(ValueError, match="dtype"):
        array_identity(array)


@pytest.mark.parametrize("value", [[1.0, 2.0], 1.0, None])
def test_array_identity_rejects_non_arrays(value: object) -> None:
    with pytest.raises(ValueError, match="NumPy array"):
        array_identity(value)  # type: ignore[arg-type]


def test_array_identity_keeps_c_order_bytes_for_fortran_layout() -> None:
    values = np.arange(6, dtype=np.int32).reshape(2, 3)
    identity = array_identity(np.asfortranarray(values))
    assert identity == {
        "shape": [2, 3],
        "dtype": "<i4",
        "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }
