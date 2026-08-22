"""Iterative bounds for host containers before recursive library calls.

Python, JAX, NumPy, and JSON entry points do not share one failure mode for
deep or cyclic host values: depending on the implementation they can raise
``RecursionError``/``SystemError`` or spend unbounded work before a protocol's
own validator runs.  This module supplies the structural preflight only.  Each
consumer keeps its domain-specific type, shape, numeric, and resource checks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

ContainerChildren = Callable[[object], Iterable[object] | None]

# Match the 16 MiB JSON byte last-fit used by `_strict_json` and checkpoint
# codecs so this scanner cannot walk an arbitrarily large host string.
_MAX_JSON_TEXT_CHARS = 16 * 1024 * 1024


def require_bounded_container_tree(
    root: object,
    *,
    children: ContainerChildren,
    max_depth: int,
    max_nodes: int | None,
    name: str,
    kind: str,
) -> None:
    """Reject excessive depth, cycles, and optionally excessive node count.

    The walk is iterative and tracks only the active ancestry path, so shared
    acyclic subtrees remain valid while true cycles fail before a recursive
    downstream consumer sees them.  ``children`` must return ``None`` for a
    leaf and an iterable for a container.
    """

    if type(max_depth) is not int or max_depth < 1:
        raise ValueError("max_depth must be a positive exact integer")
    if max_nodes is not None and (type(max_nodes) is not int or max_nodes < 1):
        raise ValueError("max_nodes must be a positive exact integer or None")
    if type(name) is not str or not name:
        raise ValueError("name must be a non-empty exact string")
    if type(kind) is not str or not kind:
        raise ValueError("kind must be a non-empty exact string")

    root_children = children(root)
    if root_children is None:
        return
    nodes_seen = 1
    ancestors = {id(root)}
    frames: list[tuple[object, Iterator[object], int]] = [
        (root, iter(root_children), 1)
    ]
    while frames:
        node, iterator, depth = frames[-1]
        if depth > max_depth:
            raise ValueError(f"{name} exceeds the maximum {kind} nesting depth")
        try:
            child = next(iterator)
        except StopIteration:
            frames.pop()
            ancestors.discard(id(node))
            continue

        nodes_seen += 1
        if max_nodes is not None and nodes_seen > max_nodes:
            raise ValueError(f"{name} exceeds the {kind} value resource limit")
        child_children = children(child)
        if child_children is None:
            continue
        child_id = id(child)
        if child_id in ancestors:
            raise ValueError(f"{name} contains a cyclic {kind}")
        child_depth = depth + 1
        if child_depth > max_depth:
            raise ValueError(f"{name} exceeds the maximum {kind} nesting depth")
        ancestors.add(child_id)
        frames.append((child, iter(child_children), child_depth))


def require_json_text_nesting(
    text: object,
    *,
    max_depth: int,
    name: str,
) -> str:
    """Preflight JSON bracket depth without counting brackets inside strings."""

    if type(text) is not str:
        raise ValueError(f"{name} must be canonical JSON")
    if type(max_depth) is not int or max_depth < 1:
        raise ValueError("max_depth must be a positive exact integer")
    if type(name) is not str or not name:
        raise ValueError("name must be a non-empty exact string")
    if len(text) > _MAX_JSON_TEXT_CHARS:
        raise ValueError(
            f"{name} length must be an integer in [0, {_MAX_JSON_TEXT_CHARS}]"
        )

    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > max_depth:
                raise ValueError(f"{name} exceeds the JSON nesting limit")
        elif character in "]}":
            depth -= 1
    return text
