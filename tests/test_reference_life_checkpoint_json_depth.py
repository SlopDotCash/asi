"""Depth ceiling for reference-life checkpoint JSON loads.

Origin ``json.loads`` RecursionError'd a 10_000-deep object nest that still
fit ``_MAX_JSON_BYTES``. The codec now uses the directory-tree depth bound
(32) and raises ``ValueError`` before the CPython decoder blows the stack.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.reference_life_checkpoint import (
    _MAX_TREE_DEPTH,
    _canonical_json_bytes,
    _load_canonical_json,
    _require_json_text_depth,
)

pytestmark = [pytest.mark.unit]


def _nested_dict(depth: int) -> dict[str, object]:
    nest: object = True
    for _ in range(depth):
        nest = {"k": nest}
    assert type(nest) is dict
    return nest


def _nested_json_bytes(depth: int) -> bytes:
    return ('{"k":' * depth + "true" + "}" * depth).encode("ascii")


def test_protocol_ceiling_matches_directory_tree_bound() -> None:
    assert _MAX_TREE_DEPTH == 32


def test_last_fit_nest_is_accepted(tmp_path: Path) -> None:
    encoded = _canonical_json_bytes(_nested_dict(_MAX_TREE_DEPTH))
    path = tmp_path / "manifest.json"
    path.write_bytes(encoded)
    loaded = _load_canonical_json(path)
    assert loaded == _nested_dict(_MAX_TREE_DEPTH)


def test_first_overflow_nest_is_value_error_not_recursion_error() -> None:
    with pytest.raises(ValueError, match="nesting limit"):
        _require_json_text_depth(
            _nested_json_bytes(_MAX_TREE_DEPTH + 1), name="manifest.json"
        )


def test_origin_hang_class_10000_is_value_error_not_recursion_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(_nested_json_bytes(10_000))
    with pytest.raises(ValueError, match="nesting limit"):
        _load_canonical_json(path)


def test_load_rejects_overflow_nest_without_recursion_error(tmp_path: Path) -> None:
    path = tmp_path / "life_state.json"
    path.write_bytes(_nested_json_bytes(_MAX_TREE_DEPTH + 1))
    with pytest.raises(ValueError, match="nesting limit"):
        _load_canonical_json(path)
