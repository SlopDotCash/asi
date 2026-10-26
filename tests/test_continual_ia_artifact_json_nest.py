"""Origin-complete: deep continual-IA evidence JSON must fail closed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alberta_framework.evaluation.continual_ia_artifact import (
    _MAX_JSON_NESTING_DEPTH,
    load_ia_evidence_artifact,
)


def _deep_object_bytes(depth: int) -> bytes:
    return (b'{"k":' * depth) + b"0" + (b"}" * depth)


def test_loader_rejects_origin_recursion_fixture(tmp_path: Path) -> None:
    """The depth-10000 object raised RecursionError before the preflight."""
    raw = _deep_object_bytes(10_000)
    assert len(raw) == 60_001
    path = tmp_path / "deep-object.json"
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="JSON nesting limit"):
        load_ia_evidence_artifact(path)


def test_loader_rejects_first_over_depth_object(tmp_path: Path) -> None:
    path = tmp_path / "over.json"
    path.write_bytes(_deep_object_bytes(_MAX_JSON_NESTING_DEPTH + 1))

    with pytest.raises(ValueError, match="JSON nesting limit"):
        load_ia_evidence_artifact(path)


def test_loader_rejects_deep_array_nest(tmp_path: Path) -> None:
    path = tmp_path / "deep-array.json"
    path.write_text("[" * 10_000 + "0" + "]" * 10_000, encoding="utf-8")

    with pytest.raises(ValueError, match="JSON nesting limit"):
        load_ia_evidence_artifact(path)


def test_loader_accepts_bounded_object(tmp_path: Path) -> None:
    path = tmp_path / "ok.json"
    path.write_text('{"schema": "x"}\n', encoding="utf-8")

    assert load_ia_evidence_artifact(path) == {"schema": "x"}


def test_loader_does_not_count_brackets_inside_strings(tmp_path: Path) -> None:
    value = '["\\' * 1_000
    path = tmp_path / "string-brackets.json"
    path.write_text(json.dumps({"value": value}), encoding="utf-8")

    assert load_ia_evidence_artifact(path) == {"value": value}
