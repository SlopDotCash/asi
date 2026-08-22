"""Supplementary coverage for update_safety.py array guards.

Covers previously untested helpers: neutralize_array (rejected-transaction
zeroing), neutralize_metrics (dict-preserving zeroing), and
zero_if_collapsed_infinity (bound-induced 0*inf NaN repair that leaves
unrelated NaNs visible).
"""

import jax.numpy as jnp
import pytest

from alberta_framework.core.update_safety import (
    neutralize_array,
    neutralize_metrics,
    zero_if_collapsed_infinity,
)


def test_neutralize_array_applied() -> None:
    applied = jnp.array([True, True])
    candidate = jnp.array([1.0, 2.0])
    out = neutralize_array(applied, candidate)
    assert out.tolist() == [1.0, 2.0]


def test_neutralize_array_rejected() -> None:
    applied = jnp.array([False, False])
    candidate = jnp.array([1.0, 2.0])
    out = neutralize_array(applied, candidate)
    assert out.tolist() == [0.0, 0.0]


def test_neutralize_array_mixed() -> None:
    applied = jnp.array([True, False, True])
    candidate = jnp.array([1.0, 2.0, 3.0])
    out = neutralize_array(applied, candidate)
    assert out.tolist() == [1.0, 0.0, 3.0]


def test_neutralize_metrics_preserves_keys() -> None:
    metrics = {"loss": jnp.array([1.0, 2.0]), "acc": jnp.array([0.9, 0.8])}
    applied = jnp.array([True, False])
    out = neutralize_metrics(applied, metrics)
    assert set(out.keys()) == {"loss", "acc"}
    assert out["loss"].tolist() == [1.0, 0.0]
    assert out["acc"][0] == pytest.approx(0.9)
    assert out["acc"][1] == 0.0


def test_zero_if_collapsed_infinity_repairs_nan() -> None:
    product = jnp.array([jnp.nan, 1.0])
    infinite_input = jnp.array([jnp.inf, 1.0])
    collapsed = jnp.array([True, False])
    out = zero_if_collapsed_infinity(product, infinite_input, collapsed)
    assert out[0] == 0.0
    assert out[1] == 1.0


def test_zero_if_collapsed_infinity_leaves_unrelated_nan() -> None:
    product = jnp.array([jnp.nan])
    infinite_input = jnp.array([1.0])  # not infinite → NaN stays
    collapsed = jnp.array([True])
    out = zero_if_collapsed_infinity(product, infinite_input, collapsed)
    assert jnp.isnan(out[0])


def test_zero_if_collapsed_infinity_not_collapsed() -> None:
    product = jnp.array([jnp.nan])
    infinite_input = jnp.array([jnp.inf])
    collapsed = jnp.array([False])  # scale did not collapse → NaN stays
    out = zero_if_collapsed_infinity(product, infinite_input, collapsed)
    assert jnp.isnan(out[0])


def test_zero_if_collapsed_infinity_finite_path() -> None:
    product = jnp.array([5.0])
    infinite_input = jnp.array([jnp.inf])
    collapsed = jnp.array([True])
    out = zero_if_collapsed_infinity(product, infinite_input, collapsed)
    assert out[0] == 5.0
