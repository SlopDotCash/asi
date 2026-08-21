"""Unit coverage for alberta_framework.benchmarks.development_provenance.

Tests the canonicalization and identity primitives: registry budget
limits, canonical sort/hash determinism, sha256 format validation, and
module hashing.
"""

import hashlib

import pytest

from alberta_framework.benchmarks.development_provenance import (
    DevelopmentIdentity,
    _canonical,
    _sha256,
    registry_sha256,
)


def test_canonical_sorts_keys() -> None:
    assert _canonical({"b": 1, "a": 2}) == {"a": 2, "b": 1}
    assert _canonical({"z": [3, 1], "y": {"m": 1}}) == {"y": {"m": 1}, "z": [3, 1]}


def test_registry_sha256_deterministic() -> None:
    a = registry_sha256({"b": [1, 2], "a": {"x": 3}})
    b = registry_sha256({"a": {"x": 3}, "b": [1, 2]})  # different insertion order
    assert a == b
    assert len(a) == 64


def test_registry_sha256_rejects_unsupported() -> None:
    with pytest.raises(TypeError, match="unsupported"):
        registry_sha256({"a": object()})
    with pytest.raises(TypeError, match="exact strings"):
        registry_sha256({1: "x"})
    with pytest.raises(ValueError, match="finite"):
        registry_sha256({"a": float("nan")})


def test_registry_sha256_rejects_huge_string() -> None:
    with pytest.raises(ValueError, match="byte limit"):
        registry_sha256({"a": "x" * 1024 * 1024})


def test_registry_sha256_rejects_deep_nesting() -> None:
    deep = value = {}
    for _ in range(40):
        child = {}
        value["k"] = child
        value = child
    with pytest.raises(ValueError, match="nesting-depth"):
        registry_sha256(deep)


def test_sha256_validates_format() -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    assert _sha256(digest, "name") == digest
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _sha256("not-a-hash", "name")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _sha256(digest.upper(), "name")


def test_development_identity_validates_sha() -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    ident = DevelopmentIdentity(
        lane_source_sha256=digest,
        dependency_source_sha256=(("numpy", digest),),
        runtime_identity=(("python", "3.12"),),
        dependency_versions=(("numpy", "1.26.4"),),
        workload_registry_sha256=digest,
        paper_registry_sha256=digest,
    )
    assert ident.lane_source_sha256 == digest


def test_development_identity_rejects_bad_sha() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        DevelopmentIdentity(
            lane_source_sha256="bad",
            dependency_source_sha256=(("numpy", "x" * 64),),
            runtime_identity=(("python", "3.12"),),
            dependency_versions=(("numpy", "1.26.4"),),
            workload_registry_sha256="y" * 64,
            paper_registry_sha256="z" * 64,
        )
