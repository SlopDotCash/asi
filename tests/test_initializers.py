"""Tests for sparse initialization."""

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework import sparse_init


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
        expected_zeros = int(sparsity * fan_in + 0.5)

        # Each row should have exactly expected_zeros zeros
        chex.assert_trees_all_close(zeros_per_row, jnp.full(fan_out, expected_zeros))

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
        expected_zeros = int(0.5 * 16 + 0.5)
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

    def test_default_sparsity_keeps_one_connection_when_rounding_hits_fan_in(self):
        """Default sparsity=0.9 on fan_in=5 must not wipe every incoming weight.

        ``int(0.9 * 5 + 0.5)`` is 5, and the ``num_zeros >= fan_in`` branch
        used to return an all-zero matrix. The constructor accepts any
        positive ``fan_in``; it does not require
        ``round(sparsity * fan_in) < fan_in``. One live connection per row
        remains when the requested sparsity is below 1.
        """
        weights = sparse_init(jr.key(0), (4, 5), sparsity=0.9)

        assert not bool(jnp.all(weights == 0))
        zeros_per_row = jnp.sum(weights == 0, axis=1)
        chex.assert_trees_all_close(zeros_per_row, jnp.full((4,), 4))
        assert float(jnp.mean(weights == 0)) == pytest.approx(0.8)
        nonzero_per_row = jnp.sum(weights != 0, axis=1)
        chex.assert_trees_all_close(nonzero_per_row, jnp.ones((4,), dtype=nonzero_per_row.dtype))

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
