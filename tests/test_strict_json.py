"""Unit coverage for alberta_framework._strict_json.

Exercises the strict JSON loading pipeline: path validation, finite-float
parsing, duplicate-key rejection, nesting preflight, bounded decode,
exact-tree validation, and the public loaders.
"""

import json
from pathlib import Path

import pytest

from alberta_framework._strict_json import (
    _parse_finite_float,
    _reject_duplicate_object_keys,
    _scan_json_nesting,
    _validate_exact_json_tree,
    load_strict_json_object,
    load_strict_json_object_from_text,
)


def test_parse_finite_float() -> None:
    assert _parse_finite_float("1.5") == 1.5
    assert _parse_finite_float("-2.25") == -2.25
    with pytest.raises(ValueError, match="exact string"):
        _parse_finite_float(1.5)
    with pytest.raises(ValueError, match="non-finite"):
        _parse_finite_float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        _parse_finite_float("inf")


def test_reject_duplicate_keys() -> None:
    assert _reject_duplicate_object_keys([("a", 1), ("b", 2)]) == {"a": 1, "b": 2}
    with pytest.raises(ValueError, match="duplicate"):
        _reject_duplicate_object_keys([("a", 1), ("a", 2)])
    with pytest.raises(ValueError, match="key-value pairs"):
        _reject_duplicate_object_keys([1, 2])
    with pytest.raises(ValueError, match="exact strings"):
        _reject_duplicate_object_keys([(1, 2)])


def test_scan_json_nesting() -> None:
    _scan_json_nesting('{"a": [1, 2]}')
    _scan_json_nesting('{"a": "string with {brackets}"}')
    # Build a >64-deep nested structure.
    deep = "{" * 70 + "}" * 70
    with pytest.raises(ValueError, match="nesting"):
        _scan_json_nesting(deep)


def test_validate_exact_tree() -> None:
    _validate_exact_json_tree({"a": [1, 2.5, "x", True, None]})
    with pytest.raises(ValueError):
        _validate_exact_json_tree({"a": object()})


def test_load_strict_json_object(tmp_path: Path) -> None:
    f = tmp_path / "data.json"
    f.write_text('{"a": 1}', encoding="utf-8")
    assert load_strict_json_object(f) == {"a": 1}


def test_load_strict_json_object_accepts_str_path(tmp_path: Path) -> None:
    f = tmp_path / "data.json"
    f.write_text('{"b": 2}', encoding="utf-8")
    assert load_strict_json_object(str(f)) == {"b": 2}


def test_load_strict_json_object_rejects_duplicate(tmp_path: Path) -> None:
    f = tmp_path / "data.json"
    f.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_strict_json_object(f)


def test_load_strict_json_object_rejects_nonstandard(tmp_path: Path) -> None:
    f = tmp_path / "data.json"
    f.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard"):
        load_strict_json_object(f)


def test_load_from_text() -> None:
    assert load_strict_json_object_from_text('{"x": 1}', label="test") == {"x": 1}
    with pytest.raises(ValueError, match="exact str"):
        load_strict_json_object_from_text({"x": 1}, label="test")
