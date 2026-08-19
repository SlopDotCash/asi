"""Reject deep or cyclic pytrees before jax.tree.leaves SystemError."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import pytest

from alberta_framework.core.update_safety import (
    _MAX_PYTREE_NESTING_DEPTH,
    _tree_leaves,
    floating_tree_is_finite,
)


def _nested_list(depth: int, leaf: Any = 1.0) -> Any:
    value: Any = leaf
    for _ in range(depth):
        value = [value]
    return value


def _nested_tuple(depth: int, leaf: Any = 1.0) -> Any:
    value: Any = leaf
    for _ in range(depth):
        value = (value,)
    return value


def _nested_dict(depth: int, leaf: Any = 1.0) -> Any:
    value: Any = leaf
    for _ in range(depth):
        value = {"k": value}
    return value


def test_deep_list_never_reaches_tree_leaves(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def spy(tree: object) -> list[Any]:
        calls.append(1)
        raise AssertionError("jax.tree.leaves must not run on overflow nests")

    monkeypatch.setattr("alberta_framework.core.update_safety.jax.tree.leaves", spy)
    with pytest.raises(ValueError, match="nesting depth"):
        floating_tree_is_finite(_nested_list(10_000))
    assert calls == []


def test_last_fit_list_nesting_still_walks() -> None:
    tree = _nested_list(_MAX_PYTREE_NESTING_DEPTH)
    assert _tree_leaves(tree) == [1.0]
    assert bool(floating_tree_is_finite(tree))


def test_first_overflow_list_rejects_before_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def spy(tree: object) -> list[Any]:
        calls.append(1)
        raise AssertionError("jax.tree.leaves must not run on overflow nests")

    monkeypatch.setattr("alberta_framework.core.update_safety.jax.tree.leaves", spy)
    with pytest.raises(ValueError, match="nesting depth"):
        floating_tree_is_finite(_nested_list(_MAX_PYTREE_NESTING_DEPTH + 1))
    assert calls == []


def test_last_fit_tuple_and_dict_nests_still_walk() -> None:
    assert bool(floating_tree_is_finite(_nested_tuple(_MAX_PYTREE_NESTING_DEPTH)))
    assert bool(floating_tree_is_finite(_nested_dict(_MAX_PYTREE_NESTING_DEPTH)))


def test_cyclic_list_rejects_before_leaves(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def spy(tree: object) -> list[Any]:
        calls.append(1)
        raise AssertionError("jax.tree.leaves must not run on cyclic trees")

    monkeypatch.setattr("alberta_framework.core.update_safety.jax.tree.leaves", spy)
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic pytree"):
        floating_tree_is_finite(cyclic)
    assert calls == []


def test_shared_subtree_is_not_cyclic() -> None:
    inner = [jnp.asarray(1.0)]
    assert bool(floating_tree_is_finite([inner, inner]))


class _Pair(NamedTuple):
    left: Any
    right: Any


def test_namedtuple_container_is_walked() -> None:
    tree = _Pair(jnp.asarray(1.0), jnp.asarray(2.0))
    assert bool(floating_tree_is_finite(tree))


def test_finite_array_leaf_still_reports_nonfinite() -> None:
    assert not bool(floating_tree_is_finite(jnp.asarray([1.0, jnp.nan])))


def test_systemerror_from_leaves_is_valueerror(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(tree: object) -> list[Any]:
        del tree
        raise SystemError("simulated pytree overflow")

    monkeypatch.setattr("alberta_framework.core.update_safety.jax.tree.leaves", boom)
    with pytest.raises(ValueError, match="nesting depth"):
        floating_tree_is_finite(1.0)
