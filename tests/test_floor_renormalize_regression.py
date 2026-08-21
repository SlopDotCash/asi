"""Regression coverage for #2238: floor_and_renormalize_probabilities must
return a valid simplex (sum-to-one) on degenerate input.

Two input classes previously produced a vector summing to n*min_probability
(3e-6 for three actions) instead of 1.0:
  - all-zero probability mass
  - float32 values whose ratios underflow to zero (e.g. 1e38 alongside 1.0)
"""

import jax.numpy as jnp
import pytest

from alberta_framework.core.behavior_model import floor_and_renormalize_probabilities


def _sum(out: jnp.ndarray) -> float:
    return float(out.sum())


def test_zero_mass_returns_uniform_simplex() -> None:
    out = floor_and_renormalize_probabilities(jnp.asarray([0.0, 0.0, 0.0], jnp.float32))
    assert _sum(out) == pytest.approx(1.0, abs=1e-6)
    assert out.min() == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_overflowing_values_return_simplex() -> None:
    out = floor_and_renormalize_probabilities(jnp.asarray([1e38, 1.0, 1.0], jnp.float32))
    assert _sum(out) == pytest.approx(1.0, abs=1e-6)


def test_well_formed_keeps_output() -> None:
    # The existing affine output for well-formed input must be preserved.
    out = floor_and_renormalize_probabilities(jnp.asarray([0.5, 0.3, 0.2], jnp.float32))
    assert _sum(out) == pytest.approx(1.0, abs=1e-6)
    assert out[0] == pytest.approx(0.49999952, abs=1e-6)


def test_negative_values_clipped() -> None:
    out = floor_and_renormalize_probabilities(jnp.asarray([-0.5, 1.5, 0.0], jnp.float32))
    assert _sum(out) == pytest.approx(1.0, abs=1e-6)
    assert out.min() >= 1e-6 - 1e-9


def test_single_action() -> None:
    out = floor_and_renormalize_probabilities(jnp.asarray([1.0], jnp.float32))
    assert _sum(out) == pytest.approx(1.0, abs=1e-6)


def test_two_actions() -> None:
    out = floor_and_renormalize_probabilities(jnp.asarray([0.1, 0.1], jnp.float32))
    assert _sum(out) == pytest.approx(1.0, abs=1e-6)
