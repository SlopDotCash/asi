"""Supplementary coverage for continual_ia_artifact.py helpers.

Covers previously untested helpers: canonical_content_bytes (compact sorted
JSON) and scientific_content_sha256 (deterministic digest) — the write path
itself requires a full ContinualIAReport, so the serialization primitives
are the unit surface.
"""

import json

import pytest

from alberta_framework.evaluation.continual_ia_artifact import (
    canonical_content_bytes,
    scientific_content_sha256,
)


def test_canonical_content_bytes_compact_sorted() -> None:
    assert canonical_content_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_canonical_content_bytes_deterministic() -> None:
    payload = {"x": [1, 2], "y": {"k": "v"}}
    assert canonical_content_bytes(payload) == canonical_content_bytes(dict(payload))


def test_canonical_content_bytes_unicode() -> None:
    out = canonical_content_bytes({"s": "αβγ"})
    assert "αβγ".encode("utf-8") in out


def test_canonical_content_bytes_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_content_bytes({"v": float("nan")})


def test_scientific_sha256_deterministic() -> None:
    payload = {"a": 1, "b": [2, 3]}
    first = scientific_content_sha256(payload)
    second = scientific_content_sha256(dict(payload))
    assert first == second
    assert len(first) == 64  # hex sha256


def test_scientific_sha256_differs() -> None:
    assert scientific_content_sha256({"a": 1}) != scientific_content_sha256({"a": 2})


def test_scientific_sha256_roundtrip() -> None:
    # The digest is over the canonical bytes: re-serializing the same
    # payload with different key order produces the same digest.
    a = scientific_content_sha256({"x": 1, "y": 2})
    b = scientific_content_sha256({"y": 2, "x": 1})
    assert a == b
