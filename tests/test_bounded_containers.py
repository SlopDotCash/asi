"""Shared host-container preflight contracts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import pytest

from alberta_framework._bounded_containers import (
    _MAX_JSON_TEXT_CHARS,
    require_bounded_container_tree,
    require_json_text_nesting,
)


def _json_children(value: object) -> Iterable[object] | None:
    if type(value) is list:
        return cast(list[object], value)
    if type(value) is dict:
        return cast(dict[str, object], value).values()
    return None


def test_tree_preflight_allows_shared_acyclic_subtrees() -> None:
    shared: list[object] = [[1]]
    require_bounded_container_tree(
        [shared, shared],
        children=_json_children,
        max_depth=4,
        max_nodes=8,
        name="payload",
        kind="JSON",
    )


def test_tree_preflight_rejects_cycle_depth_and_node_budget() -> None:
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic JSON"):
        require_bounded_container_tree(
            cyclic,
            children=_json_children,
            max_depth=4,
            max_nodes=8,
            name="payload",
            kind="JSON",
        )
    with pytest.raises(ValueError, match="nesting depth"):
        require_bounded_container_tree(
            [[[[0]]]],
            children=_json_children,
            max_depth=3,
            max_nodes=None,
            name="payload",
            kind="JSON",
        )
    with pytest.raises(ValueError, match="resource limit"):
        require_bounded_container_tree(
            [0, 1, 2],
            children=_json_children,
            max_depth=2,
            max_nodes=3,
            name="payload",
            kind="JSON",
        )


def test_json_text_scanner_rejects_oversized_host_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _MAX_JSON_TEXT_CHARS == 16 * 1024 * 1024
    monkeypatch.setattr(
        "alberta_framework._bounded_containers._MAX_JSON_TEXT_CHARS", 8
    )
    with pytest.raises(
        ValueError,
        match=r"payload length must be an integer in \[0, 8\]",
    ):
        require_json_text_nesting("[" * 9, max_depth=3, name="payload")
    require_json_text_nesting("[]", max_depth=3, name="payload")


def test_json_text_scanner_ignores_brackets_inside_escaped_strings() -> None:
    text = '{"literal":"[\\\"]}","nested":[{}]}'
    assert require_json_text_nesting(text, max_depth=3, name="payload") == text
    with pytest.raises(ValueError, match="nesting limit"):
        require_json_text_nesting("[[[[]]]]", max_depth=3, name="payload")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_depth": 0, "max_nodes": None, "name": "x", "kind": "tree"}, "max_depth"),
        ({"max_depth": 1, "max_nodes": 0, "name": "x", "kind": "tree"}, "max_nodes"),
        ({"max_depth": 1, "max_nodes": None, "name": "", "kind": "tree"}, "name"),
        ({"max_depth": 1, "max_nodes": None, "name": "x", "kind": ""}, "kind"),
    ],
)
def test_tree_preflight_rejects_invalid_own_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        require_bounded_container_tree([], children=_json_children, **kwargs)  # type: ignore[arg-type]
