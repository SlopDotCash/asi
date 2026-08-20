"""Origin-complete: deep evidence-manifest JSON must fail closed before RecursionError."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.evaluation.evidence_manifest import (
    _MAX_JSON_NESTING_DEPTH,
    _load_strict_json_object,
)


def _deep_object_bytes(depth: int) -> bytes:
    return (b'{"k":' * depth) + b"0" + (b"}" * depth)


def test_load_strict_json_object_rejects_origin_recursion_fixture(
    tmp_path: Path,
) -> None:
    """60,001-byte depth-10000 object RecursionError'd origin json.loads."""
    raw = _deep_object_bytes(10_000)
    assert len(raw) == 60_001
    path = tmp_path / "deep-object.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="JSON nesting limit"):
        _load_strict_json_object(path)


def test_load_strict_json_object_rejects_first_over_depth_object(
    tmp_path: Path,
) -> None:
    path = tmp_path / "over.json"
    path.write_bytes(_deep_object_bytes(_MAX_JSON_NESTING_DEPTH + 1))
    with pytest.raises(ValueError, match="JSON nesting limit"):
        _load_strict_json_object(path)


def test_load_strict_json_object_rejects_deep_array_nest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deep-array.json"
    path.write_text("[" * 10_000 + "0" + "]" * 10_000, encoding="utf-8")
    with pytest.raises(ValueError, match="JSON nesting limit"):
        _load_strict_json_object(path)


def test_load_strict_json_object_accepts_bounded_object(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ok.json"
    path.write_text('{"schema": "x"}\n', encoding="utf-8")
    assert _load_strict_json_object(path) == {"schema": "x"}
