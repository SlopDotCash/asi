"""Supplementary coverage for adalin.py activation helpers.

Covers previously untested helpers: adalin_relu (algebraically PReLU with
finite gating), adalin_tanh (Lipschitz-1 activation), and their
transaction variants (caller-visible validity bit).
"""

import jax.numpy as jnp
import pytest

from alberta_framework.benchmarks.adalin import (
    adalin_relu,
    adalin_relu_transaction,
    adalin_tanh,
    adalin_tanh_transaction,
)


def test_adalin_relu_positive_side() -> None:
    x = jnp.array([1.0, 2.0, 3.0])
    alpha = jnp.array([0.1, 0.2, 0.3])
    out = adalin_relu(x, alpha)
    # ReLU side: output = x * alpha + (1 - alpha) * relu(x) = x.
    assert out.tolist() == pytest.approx([1.0, 2.0, 3.0], abs=1e-5)


def test_adalin_relu_negative_side() -> None:
    x = jnp.array([-1.0, -2.0])
    alpha = jnp.array([0.5, 0.5])
    out = adalin_relu(x, alpha)
    # Negative side: output = alpha * x (PReLU).
    assert out.tolist() == pytest.approx([-0.5, -1.0], abs=1e-5)


def test_adalin_relu_transaction_valid() -> None:
    x = jnp.array([1.0, -1.0])
    alpha = jnp.array([0.1, 0.1])
    value, valid = adalin_relu_transaction(x, alpha)
    assert bool(valid)
    assert value.shape == (2,)


def test_adalin_tanh_shape() -> None:
    x = jnp.array([[1.0, -2.0], [3.0, -4.0]])
    alpha = jnp.array([0.5, 0.5])
    out = adalin_tanh(x, alpha)
    assert out.shape == (2, 2)
    assert jnp.all(jnp.isfinite(out))


def test_adalin_tanh_transaction() -> None:
    x = jnp.array([0.0, 0.5])
    alpha = jnp.array([0.1, 0.1])
    value, valid = adalin_tanh_transaction(x, alpha)
    assert bool(valid)
    assert jnp.all(jnp.isfinite(value))


def test_adalin_relu_rejects_bad_alpha() -> None:
    with pytest.raises((ValueError, TypeError)):
        adalin_relu(jnp.array([1.0]), "not-an-array")
