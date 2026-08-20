"""Origin-complete: deep shard JSON must fail closed before RecursionError."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.evaluation.upgd_ipmnist_nonpromoting import (
    _MAX_JSON_NESTING_DEPTH,
    _decode_strict_json_object,
    _strict_json_object,
    validate_upgd_ipmnist_partials,
)


def _deep_object_bytes(depth: int) -> bytes:
    return (b'{"k":' * depth) + b"0" + (b"}" * depth)


def test_decode_strict_json_object_rejects_origin_recursion_fixture() -> None:
    """60,001-byte depth-10000 object RecursionError'd origin json.loads."""
    raw = _deep_object_bytes(10_000)
    assert len(raw) == 60_001
    with pytest.raises(ValueError, match="nesting-depth limit"):
        _decode_strict_json_object(raw)


def test_strict_json_object_rejects_first_over_depth_object(tmp_path: Path) -> None:
    path = tmp_path / "over.json"
    path.write_bytes(_deep_object_bytes(_MAX_JSON_NESTING_DEPTH + 1))
    with pytest.raises(ValueError, match="nesting-depth limit"):
        _strict_json_object(path)


def test_strict_json_object_accepts_bounded_object(tmp_path: Path) -> None:
    path = tmp_path / "ok.json"
    path.write_text('{"schema": "x"}\n', encoding="utf-8")
    assert _strict_json_object(path) == {"schema": "x"}


def test_validate_partials_rejects_deep_nests_before_schema_walk(tmp_path: Path) -> None:
    path = tmp_path / "shard.json"
    path.write_bytes(_deep_object_bytes(10_000))
    result = validate_upgd_ipmnist_partials([path])
    assert result.valid is False
    assert any("nesting-depth limit" in error for error in result.errors)
