"""Shared JAX helpers for fail-closed numerical update transactions."""

from __future__ import annotations

import operator
from typing import Any, SupportsIndex, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Int

from alberta_framework._bounded_containers import require_bounded_container_tree

_ACTUAL_INT_TYPES = frozenset(
    {
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    }
)
_INT32_MAX = int(np.iinfo(np.int32).max)
# Origin ``jax.tree.leaves`` still returns at depth 8000 and SystemErrors at 10_000.
_MAX_PYTREE_NESTING_DEPTH = 4096


def _require_exact_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact bool")
    return value


def _require_non_negative_int(name: str, value: object) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [0, {_INT32_MAX}]")
    number = operator.index(cast(SupportsIndex, value))
    if not 0 <= number <= _INT32_MAX:
        raise ValueError(f"{name} must be an integer in [0, {_INT32_MAX}]")
    return number


def safe_discrete_action(
    action: Array | int,
    n_actions: int,
    *,
    allow_unset: bool = False,
) -> tuple[Array, Bool[Array, ""]]:
    """Return a safe scalar action code and its exact discrete-domain verdict.

    Casting a floating action to ``int32`` before validation can turn ``NaN``,
    infinity, fractions, and out-of-range values into an innocuous all-zero
    one-hot vector.  Validate the floating scalar first, then expose only a
    safe code to downstream one-hot arithmetic.  ``allow_unset`` admits the
    conventional ``-1`` episode-start sentinel.
    """

    n_actions = _require_non_negative_int("n_actions", n_actions)
    allow_unset = _require_exact_bool("allow_unset", allow_unset)
    if n_actions == 0:
        return (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(True, dtype=jnp.bool_),
        )
    raw = jnp.asarray(action, dtype=jnp.float32).reshape(())
    lower = -1.0 if allow_unset else 0.0
    valid = (
        jnp.isfinite(raw)
        & (raw == jnp.floor(raw))
        & (raw >= lower)
        & (raw < float(n_actions))
    )
    fallback = -1 if allow_unset else 0
    safe = jnp.where(valid, raw, jnp.asarray(fallback, dtype=jnp.float32))
    return safe.astype(jnp.int32), valid


def checked_integer_action_array(
    actions: object,
    n_actions: int,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    range_message: str,
) -> tuple[Int[Array, " *shape"], Bool[Array, " *shape"]]:
    """Validate an integer action array without narrowing away invalid values.

    Shape and dtype are static contracts, so they fail identically in eager
    and staged/JIT calls. Concrete out-of-domain values raise before work is
    staged. Traced values cannot synchronously raise from ordinary ``jax.jit``;
    they instead produce a safe index plus a validity mask that callers must
    include in their transaction commit predicate.
    """

    n_actions = _require_non_negative_int("n_actions", n_actions)
    try:
        shape = tuple(actions.shape)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as error:
        raise ValueError(f"{name} must expose an exact shape") from error
    if shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    try:
        dtype = np.dtype(actions.dtype)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as error:
        raise ValueError(f"{name} must expose an integer dtype") from error
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype")

    if not isinstance(actions, jax.core.Tracer):
        host_actions = np.asarray(actions)
        if not bool(np.all((host_actions >= 0) & (host_actions < n_actions))):
            raise ValueError(range_message)

    raw = jnp.asarray(actions)
    valid = (raw >= 0) & (raw < n_actions)
    safe = jnp.where(valid, raw, 0).astype(jnp.int32)
    return safe, valid


def _pytree_container_children(node: object) -> tuple[object, ...] | None:
    node_type = type(node)
    if node_type is dict:
        return tuple(cast(dict[Any, Any], node).values())
    if node_type is list:
        return tuple(cast(list[Any], node))
    if node_type is tuple:
        return cast(tuple[object, ...], node)
    if isinstance(node, tuple) and getattr(node_type, "_fields", None) is not None:
        return tuple(node)
    return None


def _require_pytree_nesting(tree: object, *, name: str = "tree") -> None:
    """Reject cycles and nesting that SystemError ``jax.tree.leaves``."""
    require_bounded_container_tree(
        tree,
        children=_pytree_container_children,
        max_depth=_MAX_PYTREE_NESTING_DEPTH,
        max_nodes=None,
        name=name,
        kind="pytree",
    )


def _tree_leaves(tree: object) -> list[Any]:
    _require_pytree_nesting(tree, name="tree")
    try:
        return jax.tree.leaves(tree)
    except RecursionError as exc:
        raise ValueError("tree exceeds the maximum pytree nesting depth") from exc
    except SystemError as exc:
        raise ValueError("tree exceeds the maximum pytree nesting depth") from exc


def floating_tree_is_finite(tree: object) -> Bool[Array, ""]:
    """Return whether every floating or complex leaf in ``tree`` is finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in _tree_leaves(tree):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(array))
    return valid


def select_transaction[T](applied: Array, candidate: T, current: T) -> T:
    """Commit ``candidate`` iff a scalar JAX transaction predicate is true."""

    return cast(
        T,
        jax.lax.cond(
            applied,
            lambda: candidate,
            lambda: current,
        ),
    )


def neutralize_array(applied: Array, candidate: Array) -> Array:
    """Return a finite zero diagnostic/update for a rejected transaction."""

    return jnp.where(applied, candidate, jnp.zeros_like(candidate))


def neutralize_metrics(
    applied: Array,
    metrics: dict[str, Array],
) -> dict[str, Array]:
    """Neutralize every metric while preserving the static dictionary shape."""

    return {name: neutralize_array(applied, value) for name, value in metrics.items()}


def masked_convex_weights(mask: Bool[Array, " n"], weights: Array) -> Array:
    """Renormalize non-negative ``weights`` to sum to one over ``mask``.

    A baseline built from the result is a convex combination of the masked-in
    values, so it is bounded by them and a uniform shift in all of them moves it
    by exactly that shift.  Flooring the divisor instead promises neither: once
    the masked-in entries carry less mass than the floor, the row sums to less
    than one and the baseline is pulled toward zero rather than toward the values
    that took part, which turns a centered advantage into the raw value.

    Masked-in entries whose total is not a normal float of ``weights.dtype`` fall
    back to a uniform allocation.  Every arithmetic operand below the smallest
    normal is returned as an exact zero by the accelerator backends this runs on,
    so their relative sizes cannot be recovered here and uniform is the only
    convex answer left.  A non-finite total is passed through so the caller's
    finiteness gate still sees it.
    """

    participating = jnp.where(mask, weights, jnp.zeros_like(weights))
    mass = jnp.sum(participating)
    smallest_normal = jnp.asarray(np.finfo(cast(Any, weights.dtype)).tiny, dtype=mass.dtype)
    usable = mass >= smallest_normal
    counted = jnp.sum(mask.astype(weights.dtype))
    uniform = jnp.where(mask, jnp.ones_like(weights), jnp.zeros_like(weights)) / jnp.maximum(
        counted, jnp.ones_like(counted)
    )
    return jnp.where(
        usable | ~jnp.isfinite(mass),
        participating / jnp.where(usable, mass, jnp.ones_like(mass)),
        uniform,
    )


def zero_if_collapsed_infinity(
    product: Array,
    infinite_input: Array,
    collapsed: Array,
) -> Array:
    """Replace only a bound-induced ``0 * inf`` NaN with exact zero.

    An unrelated NaN remains visible.  The repair applies only when the
    original input was infinite and the bound/clip scale actually collapsed.
    """

    return jnp.where(
        jnp.isnan(product) & jnp.isinf(infinite_input) & collapsed,
        jnp.zeros_like(product),
        product,
    )
