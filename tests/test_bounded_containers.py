"""Unit coverage for alberta_framework._bounded_containers.

Tests the iterative container-tree walk (depth, cycles, node budget) and
the JSON bracket-nesting preflight (string-aware, escape-aware).
"""

import pytest

from alberta_framework._bounded_containers import (
    require_bounded_container_tree,
    require_json_text_nesting,
)


def _list_children(value):
    if isinstance(value, list):
        return value
    return None


def test_container_leaf_passes() -> None:
    require_bounded_container_tree(
        "leaf", children=_list_children, max_depth=3, max_nodes=None, name="x", kind="list"
    )


def test_container_flat_list_passes() -> None:
    require_bounded_container_tree(
        [1, 2, 3], children=_list_children, max_depth=3, max_nodes=None, name="x", kind="list"
    )


def test_container_depth_limit() -> None:
    deep = [[[[1]]]]
    with pytest.raises(ValueError, match="nesting depth"):
        require_bounded_container_tree(
            deep, children=_list_children, max_depth=2, max_nodes=None, name="x", kind="list"
        )


def test_container_cycle_detection() -> None:
    a = [1]
    a.append(a)  # self-cycle
    with pytest.raises(ValueError, match="cyclic"):
        require_bounded_container_tree(
            a, children=_list_children, max_depth=10, max_nodes=None, name="x", kind="list"
        )


def test_container_shared_subtree_ok() -> None:
    shared = [1, 2]
    root = [shared, shared]  # shared but acyclic
    require_bounded_container_tree(
        root, children=_list_children, max_depth=5, max_nodes=None, name="x", kind="list"
    )


def test_container_node_budget() -> None:
    wide = list(range(100))
    with pytest.raises(ValueError, match="resource limit"):
        require_bounded_container_tree(
            wide, children=_list_children, max_depth=5, max_nodes=10, name="x", kind="list"
        )


def test_container_validation_args() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        require_bounded_container_tree([], children=_list_children, max_depth=0, max_nodes=None, name="x", kind="list")
    with pytest.raises(ValueError, match="name"):
        require_bounded_container_tree([], children=_list_children, max_depth=2, max_nodes=None, name="", kind="list")


def test_json_nesting_flat() -> None:
    assert require_json_text_nesting('{"a": [1, 2]}', max_depth=5, name="j") == '{"a": [1, 2]}'


def test_json_nesting_exceeds() -> None:
    with pytest.raises(ValueError, match="nesting limit"):
        require_json_text_nesting('{"a": {"b": {"c": 1}}}', max_depth=2, name="j")


def test_json_nesting_ignores_string_brackets() -> None:
    # Brackets inside strings must not count toward nesting.
    text = '{"key": "[not {real} depth]"}'
    require_json_text_nesting(text, max_depth=1, name="j")


def test_json_nesting_handles_escapes() -> None:
    # Escaped quote inside a string does not close it.
    text = '{"key": "a \\" quote"}'
    require_json_text_nesting(text, max_depth=1, name="j")
