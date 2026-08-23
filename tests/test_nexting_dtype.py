"""Regression coverage for nexting.py forward-view returns with
integer/boolean cumulant series (issue #2368)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.utils.nexting import forward_view_returns


class TestIntegerCumulantSeries:
    def test_gamma_not_truncated_for_int_cumulants(self) -> None:
        # Integer cumulant series: previously gamma was cast to int -> 0,
        # making every return degenerate.
        cumulants = jnp.array([1, 2, 3], dtype=jnp.int32)
        gamma = 0.95
        out = forward_view_returns(cumulants, gamma=gamma)
        # G_t = c_t + gamma*c_{t+1} + gamma^2*c_{t+2}
        expected = jnp.array(
            [1.0 + 0.95 * 2.0 + 0.95**2 * 3.0, 2.0 + 0.95 * 3.0, 3.0],
            dtype=jnp.float32,
        )
        assert out.dtype == jnp.float32
        assert jnp.allclose(out, expected, rtol=1e-5)

    def test_terminal_value_not_truncated_for_int_cumulants(self) -> None:
        cumulants = jnp.array([1, 2], dtype=jnp.int32)
        terminal_value = 3.75
        out = forward_view_returns(
            cumulants, gamma=0.9, terminal_value=terminal_value
        )
        # G_1 = c_1 + gamma*c_2 + gamma^2*terminal_value
        expected = jnp.array(
            [1.0 + 0.9 * 2.0 + 0.81 * 3.75, 2.0 + 0.9 * 3.75],
            dtype=jnp.float32,
        )
        assert jnp.allclose(out, expected, rtol=1e-5)


class TestBooleanCumulantSeries:
    def test_boolean_cumulants_keep_float_gamma(self) -> None:
        cumulants = jnp.array([True, True, False], dtype=jnp.bool_)
        gamma = 0.5
        out = forward_view_returns(cumulants, gamma=gamma)
        # [True,True,False] -> [1,1,0]: G1=1+.5*1+.25*0=1.5; G2=1+.5*0=1; G3=0
        expected = jnp.array([1.5, 1.0, 0.0], dtype=jnp.float32)
        assert out.dtype == jnp.float32
        assert jnp.allclose(out, expected, rtol=1e-5)


class TestFloatCumulantSeriesUnchanged:
    def test_float32_behavior_unchanged(self) -> None:
        cumulants = jnp.array([1.0, 2.0], dtype=jnp.float32)
        out = forward_view_returns(cumulants, gamma=0.5)
        expected = jnp.array([2.0, 2.0], dtype=jnp.float32)
        assert jnp.allclose(out, expected, rtol=1e-6)
