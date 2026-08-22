"""Tests for sparse initialization."""

import math

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework import sparse_init

# SparseInit is Algorithm 1 of Elsayed, Vasan, and Mahmood, "Streaming Deep
# Reinforcement Learning Finally Works" (arXiv:2410.14606v2).  Algorithm 1
# writes the per-output-neuron zero count as ``n <- s x fan_in``; the authors'
# reference implementation resolves that real-valued count with a ceiling:
#
#     num_zeros = int(math.ceil(sparsity * fan_in))
#
# (https://github.com/mohmdelsayed/streaming-drl/blob/40bd4a61/sparse_init.py).
# These pairs are exactly the ones where the ceiling and a round-to-nearest
# count disagree, so they are what separates the published rule from a
# look-alike.
_CEILING_DISAGREEMENT_CASES = [
    (0.9, 8, 8),
    (0.9, 16, 15),
    (0.9, 128, 116),
    (0.9, 256, 231),
    (0.8, 64, 52),
    (0.2, 32, 7),
    (0.95, 32, 31),
]


class TestSparseInit:
    """Tests for the sparse_init function."""

    def test_correct_output_shape(self):
        """sparse_init should return matrix of the requested shape."""
        key = jr.key(42)
        weights = sparse_init(key, (128, 10))
        chex.assert_shape(weights, (128, 10))

    def test_list_shape_preserves_legacy_sequence_input(self):
        """A two-element list remains a valid static shape input."""
        weights = sparse_init(jr.key(42), [8, 5])  # type: ignore[arg-type]
        chex.assert_shape(weights, (8, 5))

    def test_correct_sparsity_fraction(self):
        """Each output neuron should have approximately the right sparsity."""
        key = jr.key(42)
        fan_out, fan_in = 100, 50
        sparsity = 0.8
        weights = sparse_init(key, (fan_out, fan_in), sparsity=sparsity)

        # Count zeros per row
        zeros_per_row = jnp.sum(weights == 0, axis=1)
        expected_zeros = math.ceil(sparsity * fan_in)

        # Each row should have exactly expected_zeros zeros
        chex.assert_trees_all_close(zeros_per_row, jnp.full(fan_out, expected_zeros))

    @pytest.mark.parametrize(("sparsity", "fan_in", "expected_zeros"), _CEILING_DISAGREEMENT_CASES)
    def test_zero_count_is_the_published_ceiling(self, sparsity, fan_in, expected_zeros):
        """Zeros per output neuron equal ceil(sparsity * fan_in), per SparseInit.

        Elsayed et al. 2024 (arXiv:2410.14606v2) Algorithm 1 with the authors'
        reference ``sparse_init.py`` line ``num_zeros = int(math.ceil(sparsity *
        fan_in))``.  Every pair here is one where a round-to-nearest count would
        be one lower, so this pins the published rule and not a coincidence.
        """
        assert expected_zeros == math.ceil(sparsity * fan_in)
        assert expected_zeros != int(sparsity * fan_in + 0.5)

        weights = sparse_init(jr.key(7), (24, fan_in), sparsity=sparsity)
        zeros_per_row = jnp.sum(weights == 0, axis=1)
        chex.assert_trees_all_close(zeros_per_row, jnp.full(24, expected_zeros))

    @pytest.mark.parametrize(("sparsity", "fan_in"), [(0.8, 50), (0.9, 100), (0.5, 64), (0.2, 25)])
    def test_ceiling_uses_binary64_like_the_reference(self, sparsity, fan_in):
        """A whole-number ``sparsity * fan_in`` must not round up to one extra zero.

        The reference ``sparse_init.py`` evaluates ``math.ceil(sparsity *
        fan_in)`` in binary64, where ``0.9 * 100`` is exactly ``90.0``. Taking
        the ceiling of the *exact* value of the decimal literal instead would
        give 91, because binary64 ``0.9`` is slightly above nine tenths. This
        pins the reference's arithmetic, not just its ceiling.
        """
        expected_zeros = math.ceil(sparsity * fan_in)
        assert expected_zeros * 1.0 == sparsity * fan_in

        weights = sparse_init(jr.key(11), (16, fan_in), sparsity=sparsity)
        zeros_per_row = jnp.sum(weights == 0, axis=1)
        chex.assert_trees_all_close(zeros_per_row, jnp.full(16, expected_zeros))

    def test_ceiling_saturating_sparsity_zeros_every_input(self):
        """ceil(0.99 * 64) == 64 zeroes the whole row, unlike a rounded count.

        Reference: Elsayed et al. 2024 (arXiv:2410.14606v2) Algorithm 1 and
        ``sparse_init.py``; ``row_indices[:num_zeros]`` covers the full fan-in
        once the ceiling saturates, while round-to-nearest would leave one live
        input per output neuron.
        """
        fan_in = 64
        assert math.ceil(0.99 * fan_in) == fan_in
        assert int(0.99 * fan_in + 0.5) == fan_in - 1

        weights = sparse_init(jr.key(3), (16, fan_in), sparsity=0.99)
        assert jnp.all(weights == 0)

    def test_nonzero_values_within_paper_sparse_init_bounds(self):
        """Non-zero values should stay within SparseInit Algorithm 1 bounds."""
        key = jr.key(42)
        fan_out, fan_in = 64, 32
        weights = sparse_init(key, (fan_out, fan_in), sparsity=0.5)

        scale = 1.0 / fan_in**0.5
        nonzero_mask = weights != 0
        nonzero_values = weights[nonzero_mask]

        assert jnp.all(nonzero_values >= -scale)
        assert jnp.all(nonzero_values <= scale)

    def test_different_keys_give_different_results(self):
        """Different random keys should produce different weight matrices."""
        shape = (32, 16)
        w1 = sparse_init(jr.key(0), shape)
        w2 = sparse_init(jr.key(1), shape)

        assert not jnp.allclose(w1, w2)

    def test_zero_sparsity(self):
        """With sparsity=0, all weights should be non-zero."""
        key = jr.key(42)
        weights = sparse_init(key, (32, 16), sparsity=0.0)

        # All values should be non-zero (with very high probability)
        assert jnp.sum(weights == 0) == 0

    def test_normal_init_type(self):
        """Normal init type should produce valid weights."""
        key = jr.key(42)
        weights = sparse_init(key, (32, 16), sparsity=0.5, init_type="normal")

        chex.assert_shape(weights, (32, 16))
        chex.assert_tree_all_finite(weights)

        # Check sparsity
        zeros_per_row = jnp.sum(weights == 0, axis=1)
        expected_zeros = math.ceil(0.5 * 16)
        chex.assert_trees_all_close(zeros_per_row, jnp.full(32, expected_zeros))

    def test_invalid_init_type_raises(self):
        """Invalid init_type should raise ValueError."""
        key = jr.key(42)
        with pytest.raises(ValueError, match="init_type"):
            sparse_init(key, (32, 16), init_type="invalid")

    def test_high_sparsity(self):
        """90% sparsity should produce mostly zeros."""
        key = jr.key(42)
        fan_out, fan_in = 128, 100
        weights = sparse_init(key, (fan_out, fan_in), sparsity=0.9)

        total_zeros = jnp.sum(weights == 0)
        total_elements = fan_out * fan_in
        actual_sparsity = float(total_zeros) / total_elements

        assert actual_sparsity == pytest.approx(0.9, abs=0.01)

    def test_one_sparsity(self):
        """With sparsity=1.0, all weights should be zero."""
        key = jr.key(42)
        weights = sparse_init(key, (32, 16), sparsity=1.0)

        chex.assert_shape(weights, (32, 16))
        assert jnp.all(weights == 0)

    @pytest.mark.parametrize(
        "invalid_shape",
        [
            (0, 10),
            (10, 0),
            (-1, 10),
            (10, -5),
            (0, 0),
            (10,),
            (10, 10, 10),
            (),
            (True, 10),
            (10, False),
            (10.5, 5),
            (10, 5.5),
        ],
    )
    def test_invalid_shapes_raise(self, invalid_shape):
        """Invalid shape tuples should raise ValueError."""
        key = jr.key(42)
        with pytest.raises(ValueError, match=r"shape|dimension"):
            sparse_init(key, invalid_shape)

    @pytest.mark.parametrize(
        "invalid_sparsity",
        [
            -0.1,
            1.1,
            10**309,
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            False,
            "0.5",
        ],
    )
    def test_invalid_sparsity_raises(self, invalid_sparsity):
        """Out-of-bounds or non-finite sparsity should raise ValueError."""
        key = jr.key(42)
        with pytest.raises(ValueError, match="sparsity"):
            sparse_init(key, (32, 16), sparsity=invalid_sparsity)
