"""Unit coverage for alberta_framework._seed_validation.

JAX seed-domain validation: exact built-in integer enforcement, uint32
range bounds, sequence length caps, uniqueness, and exact container
type checks.
"""

import pytest

from alberta_framework._seed_validation import (
    JAX_KEY_SEED_MAX,
    JAX_SEED_SEQUENCE_MAX_LENGTH,
    require_jax_seed,
    require_unique_jax_seeds,
)


def test_require_jax_seed_accepts_boundaries() -> None:
    assert require_jax_seed(0) == 0
    assert require_jax_seed(JAX_KEY_SEED_MAX) == JAX_KEY_SEED_MAX
    assert require_jax_seed(12345) == 12345


def test_require_jax_seed_rejects_non_int() -> None:
    with pytest.raises(ValueError, match="built-in integer"):
        require_jax_seed(1.5)
    with pytest.raises(ValueError, match="built-in integer"):
        require_jax_seed("1")
    with pytest.raises(ValueError, match="built-in integer"):
        require_jax_seed(True)  # bool is not a built-in integer contract


def test_require_jax_seed_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="uint32"):
        require_jax_seed(-1)
    with pytest.raises(ValueError, match="uint32"):
        require_jax_seed(JAX_KEY_SEED_MAX + 1)


def test_require_unique_seeds_accepts_list() -> None:
    assert require_unique_jax_seeds([1, 2, 3]) == (1, 2, 3)


def test_require_unique_seeds_accepts_tuple_identity() -> None:
    assert require_unique_jax_seeds((1, 2, 3)) == (1, 2, 3)


def test_require_unique_seeds_rejects_non_container() -> None:
    with pytest.raises(ValueError, match="exact list or tuple"):
        require_unique_jax_seeds({1, 2})
    with pytest.raises(ValueError, match="exact list or tuple"):
        require_unique_jax_seeds("12")


def test_require_unique_seeds_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        require_unique_jax_seeds([])


def test_require_unique_seeds_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="unique"):
        require_unique_jax_seeds([1, 1])


def test_require_unique_seeds_rejects_bad_element() -> None:
    with pytest.raises(ValueError, match="\\[1\\]"):
        require_unique_jax_seeds([1, 2.5])


def test_require_unique_seeds_rejects_too_long() -> None:
    with pytest.raises(ValueError, match="at most"):
        require_unique_jax_seeds(list(range(JAX_SEED_SEQUENCE_MAX_LENGTH + 1)))
