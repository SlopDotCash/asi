"""Origin-complete: deep plan JSON must fail closed before RecursionError."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.upgd_label_emnist import (
    _MAX_JSON_NESTING_DEPTH,
    _strict_json_object,
    load_plan,
)


def _deep_object_bytes(depth: int) -> bytes:
    return (b'{"k":' * depth) + b"0" + (b"}" * depth)


def test_strict_json_object_rejects_origin_recursion_fixture(tmp_path: Path) -> None:
    """60,001-byte depth-10000 object RecursionError'd origin json.loads."""
    path = tmp_path / "deep.json"
    raw = _deep_object_bytes(10_000)
    path.write_bytes(raw)
    assert path.stat().st_size == 60_001
    with pytest.raises(ValueError, match="nesting-depth limit"):
        _strict_json_object(path)


def test_strict_json_object_rejects_first_over_depth_object(tmp_path: Path) -> None:
    path = tmp_path / "over.json"
    path.write_bytes(_deep_object_bytes(_MAX_JSON_NESTING_DEPTH + 1))
    with pytest.raises(ValueError, match="nesting-depth limit"):
        _strict_json_object(path)


def test_strict_json_object_accepts_bounded_object(tmp_path: Path) -> None:
    path = tmp_path / "ok.json"
    path.write_text('{"schema": "x"}\n', encoding="utf-8")
    assert _strict_json_object(path) == {"schema": "x"}


def test_load_plan_rejects_deep_nests_before_schema_walk(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_bytes(_deep_object_bytes(10_000))
    with pytest.raises(ValueError, match="nesting-depth limit"):
        load_plan(path)
