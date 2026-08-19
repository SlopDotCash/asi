"""Depth ceiling for checkpoint metadata JSON walks.

Origin ``json.dumps`` RecursionError'd a 10_000-deep metadata nest. The
protocol ceiling matches ``security._JSON_MAX_DEPTH`` (32).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework import LinearLearner
from alberta_framework.core.checkpoints import (
    _JSON_MAX_DEPTH,
    _require_json_safe_metadata,
    save_checkpoint,
)


def _nested_dict(depth: int) -> dict[str, object]:
    nest: object = True
    for _ in range(depth):
        nest = {"k": nest}
    assert type(nest) is dict
    return nest


def test_protocol_ceiling_matches_json_depth_bound() -> None:
    assert _JSON_MAX_DEPTH == 32


def test_last_fit_nest_is_accepted() -> None:
    _require_json_safe_metadata(_nested_dict(_JSON_MAX_DEPTH))


def test_first_overflow_nest_is_value_error_not_recursion_error() -> None:
    with pytest.raises(ValueError, match="nesting limit"):
        _require_json_safe_metadata(_nested_dict(_JSON_MAX_DEPTH + 1))


def test_origin_hang_class_10000_is_value_error_not_recursion_error() -> None:
    with pytest.raises(ValueError, match="nesting limit"):
        _require_json_safe_metadata(_nested_dict(10_000))


def test_save_checkpoint_rejects_overflow_nest_without_writing(tmp_path: Path) -> None:
    learner = LinearLearner()
    state = learner.init(3)
    path = tmp_path / "deep"
    with pytest.raises(ValueError, match="nesting limit"):
        save_checkpoint(state, path, metadata=_nested_dict(_JSON_MAX_DEPTH + 1))
    assert not path.exists()
