"""Unit coverage for alberta_framework._fixed_count_selection.

Tests the positive-int32 validation and the stable smallest-mask
selection primitive (tie-breaking by source index, non-finite handling,
validation gates).
"""

import jax
import jax.numpy as jnp
import pytest

from alberta_framework._fixed_count_selection import (
    require_positive_builtin_int,
    stable_smallest_mask,
)

_INT32_MAX = (1 << 31) - 1


def test_require_positive_int_accepts() -> None:
    assert require_positive_builtin_int(1, name="count") == 1
    assert require_positive_builtin_int(_INT32_MAX, name="count") == _INT32_MAX


def test_require_positive_int_rejects() -> None:
    with pytest.raises(ValueError, match="positive"):
        require_positive_builtin_int(0, name="count")
    with pytest.raises(ValueError, match="positive"):
        require_positive_builtin_int(-1, name="count")
    with pytest.raises(ValueError, match="positive"):
        require_positive_builtin_int(1.5, name="count")
    with pytest.raises(ValueError, match="positive"):
        require_positive_builtin_int(_INT32_MAX + 1, name="count")


def test_stable_mask_selects_smallest() -> None:
    scores = jnp.array([[3.0, 1.0, 2.0], [5.0, 4.0, 6.0]])
    mask = stable_smallest_mask(scores, 2)
    # Row 0: smallest two are index 1 (1.0) and index 2 (2.0).
    # Row 1: smallest two are index 0 (5.0) and index 1 (4.0).
    assert mask.tolist() == [[False, True, True], [True, True, False]]


def test_stable_mask_ties_break_by_index() -> None:
    scores = jnp.array([[1.0, 1.0, 1.0]])
    mask = stable_smallest_mask(scores, 1)
    # Tie → source index order: index 0 wins.
    assert mask.tolist() == [[True, False, False]]


def test_stable_mask_zero_count() -> None:
    scores = jnp.array([[1.0, 2.0]])
    mask = stable_smallest_mask(scores, 0)
    assert mask.tolist() == [[False, False]]


def test_stable_mask_full_count() -> None:
    scores = jnp.array([[1.0, 2.0]])
    mask = stable_smallest_mask(scores, 2)
    assert mask.tolist() == [[True, True]]


def test_stable_mask_nonfinite_mapped_to_inf() -> None:
    scores = jnp.array([[1.0, jnp.nan, 2.0]])
    mask = stable_smallest_mask(scores, 2)
    # nan → inf → never selected when 2 finite candidates exist.
    assert mask.tolist() == [[True, False, True]]


def test_stable_mask_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="count"):
        stable_smallest_mask(jnp.array([[1.0]]), 2)
    with pytest.raises(ValueError, match="JAX array"):
        stable_smallest_mask([[1.0]], 1)
    with pytest.raises(ValueError, match="floating"):
        stable_smallest_mask(jnp.array([[1, 2]]), 1)
    with pytest.raises(ValueError, match="candidate axis"):
        stable_smallest_mask(jnp.array(1.0), 1)
