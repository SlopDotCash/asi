"""Hostile string gates for forager matrix schema and artifact."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matrix import (
    ForagerMatrixError,
    ForagerMatrixManifestError,
    ForagerMatrixStateError,
    _matrix_rng_contract,
    _parse_variant,
    _safe_artifact_parts,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")

    def __contains__(self, item: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile contains must not run")


def test_matrix_rng_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("v2")
    _HostileStr.calls = 0
    with pytest.raises(ForagerMatrixError, match="matrix schema is invalid"):
        _matrix_rng_contract(hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_matrix_rng_rejects_non_string() -> None:
    with pytest.raises(ForagerMatrixError, match="matrix schema is invalid"):
        _matrix_rng_contract(123)  # type: ignore[arg-type]


def test_matrix_rng_benign_rejects_unknown() -> None:
    with pytest.raises(ForagerMatrixError, match="unsupported matrix schema"):
        _matrix_rng_contract("unknown")


def test_parse_variant_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("v2")
    _HostileStr.calls = 0
    with pytest.raises(ForagerMatrixManifestError, match="matrix schema is invalid"):
        _parse_variant({}, path="p", schema_version=hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_safe_artifact_rejects_hostile_before_contains() -> None:
    hostile = _HostileStr("a/b")
    _HostileStr.calls = 0
    with pytest.raises(ForagerMatrixStateError, match="artifact path must be an exact string"):
        _safe_artifact_parts(hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_safe_artifact_rejects_non_string() -> None:
    with pytest.raises(ForagerMatrixStateError, match="artifact path must be an exact string"):
        _safe_artifact_parts(123)  # type: ignore[arg-type]


def test_safe_artifact_benign_valid() -> None:
    parts = _safe_artifact_parts("a/b/c")
    assert parts == ("a", "b", "c")
