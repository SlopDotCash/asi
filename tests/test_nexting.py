"""Tests for the multi-timescale nexting evaluation harness."""

from __future__ import annotations

from typing import cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from alberta_framework.utils.nexting import (
    forward_view_returns,
    multi_channel_horizon_returns,
    multi_horizon_returns,
    per_horizon_rmse,
    per_horizon_running_rmse,
)


class TestForwardViewReturns:
    def test_gamma_zero_equals_next_cumulant(self) -> None:
        c = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g = forward_view_returns(c, gamma=0.0)
        chex.assert_trees_all_close(g, c)

    def test_gamma_zero_skips_inf_later_return(self) -> None:
        """gamma=0 is G_t = c_{t+1}; 0 * inf G_{t+1} is NaN.

        Fail-closed: a zero discount does not multiply the later return.
        """
        c = jnp.array([1.0, jnp.inf, 3.0])
        g = forward_view_returns(c, gamma=0.0)
        assert bool(jnp.isfinite(g[0]))
        assert bool(jnp.isfinite(g[2]))
        chex.assert_trees_all_close(g[0], jnp.float32(1.0))
        chex.assert_trees_all_close(g[2], jnp.float32(3.0))
        assert bool(jnp.isinf(g[1]))

    def test_gamma_one_undiscounted(self) -> None:
        c = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g = forward_view_returns(c, gamma=1.0)
        # Cumulative sum from the right
        expected = jnp.array([15.0, 14.0, 12.0, 9.0, 5.0])
        chex.assert_trees_all_close(g, expected)

    def test_gamma_half(self) -> None:
        c = jnp.array([1.0, 2.0, 4.0])
        # G_2 = 4
        # G_1 = 2 + 0.5 * 4 = 4
        # G_0 = 1 + 0.5 * 4 = 3
        expected = jnp.array([3.0, 4.0, 4.0])
        g = forward_view_returns(c, gamma=0.5)
        chex.assert_trees_all_close(g, expected, atol=1e-6)

    def test_terminal_value(self) -> None:
        c = jnp.array([0.0, 0.0, 1.0])
        # With terminal_value=10, gamma=1
        # G_2 = 1 + 1 * 10 = 11
        # G_1 = 0 + 1 * 11 = 11
        # G_0 = 0 + 1 * 11 = 11
        g = forward_view_returns(c, gamma=1.0, terminal_value=10.0)
        chex.assert_trees_all_close(g, jnp.array([11.0, 11.0, 11.0]))

    @pytest.mark.parametrize("gamma", [True, False, float("nan"), float("inf"), -0.1, 1.1])
    def test_gamma_rejects_boolean_and_non_discount_host_values(self, gamma: object) -> None:
        """True used to compile as undiscounted gamma=1.0; False as gamma=0.0."""

        c = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        with pytest.raises(ValueError, match="gamma"):
            forward_view_returns(c, gamma=cast(float, gamma))

    def test_gamma_true_is_not_the_undiscounted_identity(self) -> None:
        c = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        legal = forward_view_returns(c, gamma=1.0)
        chex.assert_trees_all_close(legal, jnp.array([6.0, 5.0, 3.0]))
        with pytest.raises(ValueError, match="boolean"):
            forward_view_returns(c, gamma=True)

    @pytest.mark.parametrize("terminal_value", [True, False, float("nan"), float("inf")])
    def test_terminal_value_rejects_boolean_and_nonfinite_hosts(
        self, terminal_value: object
    ) -> None:
        c = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        with pytest.raises(ValueError, match="terminal_value"):
            forward_view_returns(c, gamma=0.0, terminal_value=cast(float, terminal_value))

    def test_bool_gamma_vector_is_not_the_one_zero_horizon_pair(self) -> None:
        c = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        legal = multi_horizon_returns(c, jnp.array([1.0, 0.0], dtype=jnp.float32))
        chex.assert_shape(legal, (3, 2))
        with pytest.raises(ValueError, match="gammas"):
            multi_horizon_returns(c, jnp.array([True, False]))

    @pytest.mark.parametrize("array_type", [np.asarray, jnp.asarray])
    @pytest.mark.parametrize("gamma", [float("nan"), float("inf"), -0.1, 1.1])
    def test_concrete_array_gamma_is_validated(self, array_type, gamma: float) -> None:
        c = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        with pytest.raises(ValueError, match="gamma"):
            forward_view_returns(c, gamma=array_type(gamma))

    @pytest.mark.parametrize("array_type", [np.asarray, jnp.asarray])
    def test_concrete_gamma_vector_values_are_validated(self, array_type) -> None:
        c = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        with pytest.raises(ValueError, match="gammas"):
            multi_horizon_returns(c, array_type([0.5, float("nan")]))
        with pytest.raises(ValueError, match="gammas"):
            multi_horizon_returns(c, array_type([0.5, 1.1]))

    def test_jit_traced_legal_gamma_remains_supported(self) -> None:
        c = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        actual = jax.jit(forward_view_returns)(c, jnp.array(0.5, dtype=jnp.float32))
        chex.assert_trees_all_close(actual, jnp.array([2.75, 3.5, 3.0]))

    def test_float32_output_is_byte_identical_to_expected(self) -> None:
        """Guard the promotion fix against silently altering float32 arithmetic."""
        c = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
        g = forward_view_returns(c, gamma=0.9)
        assert g.dtype == jnp.float32
        expected = jnp.array(
            [1.6560999, 0.7289999, 0.80999994, 0.9, 1.0], dtype=jnp.float32
        )
        chex.assert_trees_all_equal(g, expected)

    @pytest.mark.parametrize(
        "dtype", [jnp.int16, jnp.int32, jnp.uint8, jnp.uint32]
    )
    def test_integer_cumulants_match_float_and_are_not_the_echo(self, dtype) -> None:
        """int/uint series used to truncate gamma to 0 and echo the raw cumulants."""
        c_int = jnp.array([1, 0, 0, 0, 1], dtype=dtype)
        g_int = forward_view_returns(c_int, gamma=0.9)
        g_float = forward_view_returns(c_int.astype(jnp.float32), gamma=0.9)
        assert jnp.issubdtype(g_int.dtype, jnp.floating)
        chex.assert_trees_all_close(g_int, g_float, atol=1e-6)
        # The pre-fix degenerate result was a raw cumulant echo [1,0,0,0,1].
        echo = c_int.astype(jnp.float32)
        assert not bool(jnp.allclose(g_int, echo))
        chex.assert_trees_all_close(
            g_int,
            jnp.array([1.6561, 0.729, 0.81, 0.9, 1.0], dtype=jnp.float32),
            atol=1e-4,
        )

    def test_boolean_cumulants_are_promoted_not_logical_or(self) -> None:
        """Boolean series used to collapse to all-True via the bootstrap OR."""
        c_bool = jnp.array([True, False, False, False, True])
        g_bool = forward_view_returns(c_bool, gamma=0.9)
        g_float = forward_view_returns(
            c_bool.astype(jnp.float32), gamma=0.9
        )
        assert jnp.issubdtype(g_bool.dtype, jnp.floating)
        chex.assert_trees_all_close(g_bool, g_float, atol=1e-6)
        # Pre-fix collapse produced all-True (1.0 everywhere); reject it.
        assert not bool(jnp.all(g_bool == 1.0))

    def test_fractional_terminal_value_not_truncated_with_integer_cumulants(
        self,
    ) -> None:
        c_int = jnp.array([1, 2, 3], dtype=jnp.int32)
        g = forward_view_returns(c_int, gamma=1.0, terminal_value=7.6)
        # G_2 = 3 + 7.6 = 10.6, G_1 = 2 + 10.6 = 12.6, G_0 = 1 + 12.6 = 13.6.
        chex.assert_trees_all_close(
            g, jnp.array([13.6, 12.6, 10.6], dtype=jnp.float32), atol=1e-5
        )

    def test_integer_multi_horizon_gives_distinct_decay_per_horizon(self) -> None:
        c_int = jnp.array([1, 0, 0, 0, 1], dtype=jnp.int32)
        gammas = jnp.array([0.5, 0.9], dtype=jnp.float32)
        g_int = multi_horizon_returns(c_int, gammas)
        g_float = multi_horizon_returns(c_int.astype(jnp.float32), gammas)
        chex.assert_shape(g_int, (5, 2))
        chex.assert_trees_all_close(g_int, g_float, atol=1e-6)
        # Distinct decay per horizon -- not the degenerate echo in both columns.
        assert not bool(jnp.allclose(g_int[:, 0], g_int[:, 1]))

    def test_integer_multi_channel_matches_float(self) -> None:
        c_int = jnp.array(
            [[1, 0], [0, 0], [0, 1]], dtype=jnp.int32
        )
        gammas = jnp.array([0.9], dtype=jnp.float32)
        g_int = multi_channel_horizon_returns(c_int, gammas)
        g_float = multi_channel_horizon_returns(
            c_int.astype(jnp.float32), gammas
        )
        chex.assert_shape(g_int, (3, 2, 1))
        chex.assert_trees_all_close(g_int, g_float, atol=1e-6)
        # Channel 1 (delayed pulse) decays distinctly from channel 0.
        assert not bool(jnp.allclose(g_int[:, 0, 0], g_int[:, 1, 0]))

    def test_per_horizon_rmse_verdict_not_inverted_by_integer_returns(self) -> None:
        """A perfect predictor must not score worse than a degenerate echo."""
        c_int = jnp.array([1, 0, 0, 0, 1], dtype=jnp.int32)
        truth = forward_view_returns(c_int, gamma=0.9).reshape(-1, 1)
        echo = c_int.astype(jnp.float32).reshape(-1, 1)
        perfect_rmse = per_horizon_rmse(truth, truth)
        echo_rmse = per_horizon_rmse(echo, truth)
        assert float(perfect_rmse[0]) < float(echo_rmse[0])


class TestMultiHorizon:
    def test_shape(self) -> None:
        c = jnp.arange(10, dtype=jnp.float32)
        gammas = jnp.array([0.0, 0.5, 0.9, 0.99], dtype=jnp.float32)
        g = multi_horizon_returns(c, gammas)
        chex.assert_shape(g, (10, 4))

    def test_each_column_matches_single_call(self) -> None:
        c = jnp.array([1.0, -1.0, 0.5, 0.5, -0.5, 0.0])
        gammas = jnp.array([0.1, 0.5, 0.9])
        g_multi = multi_horizon_returns(c, gammas)

        for i, gv in enumerate([0.1, 0.5, 0.9]):
            g_single = forward_view_returns(c, gamma=gv)
            chex.assert_trees_all_close(g_multi[:, i], g_single, atol=1e-6)

    def test_zero_gamma_column(self) -> None:
        c = jnp.array([0.0, 1.0, 0.0, 2.0])
        gammas = jnp.array([0.0, 0.9])
        g = multi_horizon_returns(c, gammas)
        chex.assert_trees_all_close(g[:, 0], c)


class TestMultiChannel:
    def test_shape_and_values(self) -> None:
        cumulants = jnp.array(
            [
                [1.0, -1.0],
                [2.0, 0.0],
                [3.0, 1.0],
            ]
        )
        gammas = jnp.array([0.0, 0.5])
        g = multi_channel_horizon_returns(cumulants, gammas)
        chex.assert_shape(g, (3, 2, 2))  # (T, C, H)

        # Channel 0 at gamma=0
        chex.assert_trees_all_close(g[:, 0, 0], cumulants[:, 0])
        # Channel 1 at gamma=0
        chex.assert_trees_all_close(g[:, 1, 0], cumulants[:, 1])

    def test_cross_channel_independent(self) -> None:
        # Two channels with different cumulant patterns
        c1 = jnp.array([1.0, 0.0, 0.0])
        c2 = jnp.array([0.0, 0.0, 1.0])
        cumulants = jnp.stack([c1, c2], axis=1)
        gammas = jnp.array([0.9])
        g = multi_channel_horizon_returns(cumulants, gammas)

        # Channel 0: G_0 = 1 + 0.9*0 + 0.81*0 = 1; G_1 = 0; G_2 = 0
        chex.assert_trees_all_close(g[:, 0, 0], jnp.array([1.0, 0.0, 0.0]), atol=1e-6)
        # Channel 1: G_0 = 0 + 0.9*0 + 0.81*1 = 0.81; G_1 = 0.9; G_2 = 1
        chex.assert_trees_all_close(g[:, 1, 0], jnp.array([0.81, 0.9, 1.0]), atol=1e-6)

    @pytest.mark.parametrize("array_type", [np.zeros, jnp.zeros])
    def test_channel_count_limits_checked_for_numpy_and_jax(self, array_type) -> None:
        cumulants_legal = array_type((5, 8), dtype=np.float32)
        gammas = jnp.array([0.5], dtype=jnp.float32)
        g = multi_channel_horizon_returns(cumulants_legal, gammas)
        chex.assert_shape(g, (5, 8, 1))

        cumulants_too_many = array_type((5, 9), dtype=np.float32)
        with pytest.raises(
            ValueError,
            match=r"cumulants channel count must be an integer in \[1, 8\]",
        ):
            multi_channel_horizon_returns(cumulants_too_many, gammas)

    @pytest.mark.parametrize("gamma_val", [True, float("nan"), float("inf"), -0.1, 1.1])
    def test_multi_channel_rejects_invalid_gammas(self, gamma_val: object) -> None:
        cumulants = jnp.zeros((3, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="gammas"):
            multi_channel_horizon_returns(cumulants, [cast(float, gamma_val)])

    @pytest.mark.parametrize("term_val", [True, float("nan"), float("inf")])
    def test_multi_channel_rejects_invalid_terminal_value(self, term_val: object) -> None:
        cumulants = jnp.zeros((3, 2), dtype=jnp.float32)
        gammas = jnp.array([0.5], dtype=jnp.float32)
        with pytest.raises(ValueError, match="terminal_value"):
            multi_channel_horizon_returns(cumulants, gammas, terminal_value=cast(float, term_val))


class TestRMSE:
    def test_large_finite_errors_do_not_overflow(self) -> None:
        predictions = jnp.asarray([[2.0e20], [2.0e20]], dtype=jnp.float32)
        returns = jnp.zeros_like(predictions)

        with jax.debug_infs(True):
            rmse = per_horizon_rmse(predictions, returns)

        assert bool(jnp.isfinite(rmse[0]))
        np.testing.assert_allclose(np.asarray(rmse), np.asarray([2.0e20], dtype=np.float32))

    def test_zero_error_does_not_form_a_discarded_zero_division(self) -> None:
        predictions = jnp.zeros((3, 2), dtype=jnp.float32)

        with jax.debug_nans(True):
            rmse = per_horizon_rmse(predictions, predictions)

        np.testing.assert_array_equal(np.asarray(rmse), np.zeros(2, dtype=np.float32))

    def test_near_max_finite_errors_do_not_underflow_during_scaling(self) -> None:
        values = np.linspace(-1.0e38, 1.0e38, 257, dtype=np.float32)[:, None]
        predictions = jnp.asarray(values)
        returns = jnp.zeros_like(predictions)
        reference = np.sqrt(np.mean(np.square(values.astype(np.float64)), axis=0))

        with jax.debug_infs(True):
            rmse = per_horizon_rmse(predictions, returns)

        assert bool(jnp.isfinite(rmse[0]))
        np.testing.assert_allclose(np.asarray(rmse), reference, rtol=2e-6)
        compiled = jax.jit(per_horizon_rmse)(predictions, returns)
        np.testing.assert_allclose(np.asarray(compiled), reference, rtol=2e-6)

    def test_finite_stable_rmse_has_no_checkify_float_errors(self) -> None:
        checked = checkify.checkify(per_horizon_rmse, errors=checkify.float_checks)

        for predictions in (
            jnp.zeros((3, 1), dtype=jnp.float32),
            jnp.full((3, 1), 2.0e20, dtype=jnp.float32),
            jnp.linspace(-1.0e38, 1.0e38, 257, dtype=jnp.float32)[:, None],
        ):
            error, rmse = checked(predictions, jnp.zeros_like(predictions))
            error.throw()
            chex.assert_tree_all_finite(rmse)

    def test_huge_finite_error_gradient_is_correct_eager_and_jit(self) -> None:
        predictions = jnp.asarray([[3.0e38], [-1.0e38]], dtype=jnp.float32)
        returns = jnp.zeros_like(predictions)

        def scalar_rmse(values: jax.Array) -> jax.Array:
            return per_horizon_rmse(values, returns)[0]

        reference_values = np.asarray(predictions, dtype=np.float64)
        reference_rmse = np.sqrt(np.mean(reference_values**2, axis=0))[0]
        expected = reference_values / (predictions.shape[0] * reference_rmse)

        eager = jax.grad(scalar_rmse)(predictions)
        compiled = jax.jit(jax.grad(scalar_rmse))(predictions)

        assert bool(jnp.all(jnp.isfinite(eager)))
        assert bool(jnp.all(jnp.isfinite(compiled)))
        np.testing.assert_allclose(np.asarray(eager), expected, rtol=2e-6, atol=0.0)
        np.testing.assert_allclose(np.asarray(compiled), expected, rtol=2e-6, atol=0.0)

    @pytest.mark.parametrize("magnitude", [2.0e20, 1.0e38, 3.0e38])
    def test_large_finite_rmse_has_stable_eager_and_jit_gradients(
        self,
        magnitude: float,
    ) -> None:
        predictions = jnp.full((2, 1), magnitude, dtype=jnp.float32)
        returns = jnp.zeros_like(predictions)

        def loss(values: jax.Array) -> jax.Array:
            return jnp.sum(per_horizon_rmse(values, returns))

        expected = jnp.full_like(predictions, 0.5)
        eager = jax.grad(loss)(predictions)
        compiled = jax.jit(jax.grad(loss))(predictions)

        chex.assert_trees_all_close(eager, expected, rtol=5e-6, atol=0.0)
        chex.assert_trees_all_close(compiled, expected, rtol=5e-6, atol=0.0)

    def test_zero_rmse_uses_zero_eager_jit_and_forward_gradient_convention(self) -> None:
        predictions = jnp.zeros((3, 2), dtype=jnp.float32)
        returns = jnp.zeros_like(predictions)

        def loss(values: jax.Array) -> jax.Array:
            return jnp.sum(per_horizon_rmse(values, returns))

        expected = jnp.zeros_like(predictions)
        eager = jax.grad(loss)(predictions)
        compiled = jax.jit(jax.grad(loss))(predictions)
        _primal, tangent = jax.jvp(
            lambda values: per_horizon_rmse(values, returns),
            (predictions,),
            (jnp.ones_like(predictions),),
        )

        chex.assert_trees_all_equal(eager, expected)
        chex.assert_trees_all_equal(compiled, expected)
        chex.assert_trees_all_equal(tangent, jnp.zeros((2,), dtype=jnp.float32))

    def test_zero_error_when_predictions_match(self) -> None:
        t, h = 50, 4
        truths = jnp.ones((t, h))
        rmse = per_horizon_rmse(truths, truths)
        chex.assert_trees_all_close(rmse, jnp.zeros(h), atol=1e-7)

    def test_constant_error(self) -> None:
        t, h = 100, 2
        preds = jnp.zeros((t, h))
        truths = jnp.ones((t, h)) * 3.0
        rmse = per_horizon_rmse(preds, truths)
        chex.assert_trees_all_close(rmse, jnp.array([3.0, 3.0]), atol=1e-7)

    def test_burn_in(self) -> None:
        # Errors only in the first 5 steps; with burn-in=5, RMSE should be 0.
        t, h = 20, 1
        preds = jnp.zeros((t, h))
        truths = jnp.zeros((t, h)).at[:5].set(10.0)
        rmse_no_burn = per_horizon_rmse(preds, truths, burn_in=0)
        rmse_burn = per_horizon_rmse(preds, truths, burn_in=5)
        assert float(rmse_no_burn[0]) > 0.0
        chex.assert_trees_all_close(rmse_burn, jnp.zeros(h), atol=1e-6)

    @pytest.mark.parametrize(
        "burn_in",
        [
            pytest.param(True, id="bool"),
            pytest.param(1.0, id="float"),
            pytest.param(np.int64(1), id="numpy-scalar"),
            pytest.param(jnp.asarray(1, dtype=jnp.int32), id="jax-scalar"),
        ],
    )
    def test_burn_in_requires_exact_builtin_int(self, burn_in: object) -> None:
        predictions = jnp.ones((3, 2))
        returns = jnp.zeros((3, 2))

        with pytest.raises(ValueError, match=r"^burn_in must be a built-in int$"):
            per_horizon_rmse(predictions, returns, burn_in=cast(int, burn_in))

    @pytest.mark.parametrize("burn_in", [-1, 3, 5])
    def test_burn_in_out_of_range_is_rejected(self, burn_in: int) -> None:
        predictions = jnp.ones((3, 2))
        returns = jnp.zeros((3, 2))

        with pytest.raises(ValueError) as exc_info:
            per_horizon_rmse(predictions, returns, burn_in=burn_in)

        assert str(exc_info.value) == (
            "burn_in must satisfy 0 <= burn_in < n_steps "
            f"(got burn_in={burn_in}, n_steps=3)"
        )

    @pytest.mark.parametrize(
        ("prediction_shape", "return_shape"),
        [
            pytest.param((), (), id="rank-zero"),
            pytest.param((3,), (3,), id="rank-one"),
            pytest.param((3, 2, 1), (3, 2, 1), id="rank-three"),
            pytest.param((3, 1), (3, 2), id="broadcast-horizon"),
            pytest.param((1, 2), (3, 2), id="broadcast-time"),
            pytest.param((0, 2), (0, 2), id="empty-time"),
            pytest.param((3, 0), (3, 0), id="empty-horizon"),
        ],
    )
    def test_inputs_require_equal_rank_two_nonempty_shape(
        self,
        prediction_shape: tuple[int, ...],
        return_shape: tuple[int, ...],
    ) -> None:
        predictions = jnp.zeros(prediction_shape)
        returns = jnp.zeros(return_shape)

        with pytest.raises(
            ValueError,
            match=r"must be rank-2 arrays with identical nonempty shape \(T, H\)",
        ):
            per_horizon_rmse(predictions, returns)

    def test_burn_in_boundaries_are_bitexact_eager_and_static_jit(self) -> None:
        predictions = jnp.array(
            [[1.0, 2.0], [1.0, 2.0], [5.0, -10.0]],
            dtype=jnp.float32,
        )
        returns = jnp.zeros_like(predictions)
        compiled = jax.jit(per_horizon_rmse, static_argnames=("burn_in",))
        expected_by_burn_in = {
            0: np.array([3.0, 6.0], dtype=np.float32),
            2: np.array([5.0, 10.0], dtype=np.float32),
        }

        for burn_in, expected in expected_by_burn_in.items():
            eager = per_horizon_rmse(predictions, returns, burn_in=burn_in)
            jitted = compiled(predictions, returns, burn_in=burn_in)
            np.testing.assert_array_equal(np.asarray(eager), expected)
            np.testing.assert_array_equal(np.asarray(jitted), expected)

        closed_over = jax.jit(lambda p, r: per_horizon_rmse(p, r, burn_in=2))
        np.testing.assert_array_equal(
            np.asarray(closed_over(predictions, returns)),
            expected_by_burn_in[2],
        )

    def test_nonfinite_errors_remain_visible(self) -> None:
        predictions = jnp.array([[1.0, 1.0], [jnp.nan, jnp.inf]])
        returns = jnp.zeros_like(predictions)

        rmse = per_horizon_rmse(predictions, returns)

        assert bool(jnp.isnan(rmse[0]))
        assert bool(jnp.isinf(rmse[1]))

    def test_dynamic_jit_burn_in_is_named_and_default_call_recovers(self) -> None:
        predictions = jnp.tile(jnp.array([[3.0, -4.0]], dtype=jnp.float32), (3, 1))
        returns = jnp.zeros_like(predictions)
        compiled = jax.jit(per_horizon_rmse)

        with pytest.raises(ValueError, match=r"^burn_in must be a built-in int"):
            compiled(predictions, returns, burn_in=1)

        recovered = compiled(predictions, returns)
        np.testing.assert_array_equal(np.asarray(recovered), np.array([3.0, 4.0]))

    def test_jit_rejects_broadcasting_at_the_shape_boundary(self) -> None:
        predictions = jnp.ones((3, 2))
        returns = jnp.zeros((3, 1))

        with pytest.raises(
            ValueError,
            match=r"must be rank-2 arrays with identical nonempty shape \(T, H\)",
        ):
            jax.jit(per_horizon_rmse)(predictions, returns)


class TestRunningRMSE:
    def test_warmup_rows_do_not_use_future_errors(self) -> None:
        predictions = jnp.asarray([[1.0], [3.0], [5.0]], dtype=jnp.float32)
        returns = jnp.zeros_like(predictions)

        running = per_horizon_running_rmse(predictions, returns, window_size=2)

        expected = np.asarray(
            [[np.nan], [np.sqrt(5.0)], [np.sqrt(17.0)]], dtype=np.float32
        )
        np.testing.assert_allclose(np.asarray(running), expected, rtol=1e-6)

    @pytest.mark.parametrize("nonfinite", [jnp.nan, jnp.inf])
    def test_future_nonfinite_does_not_poison_earlier_complete_window(
        self, nonfinite: float
    ) -> None:
        predictions = jnp.asarray([[1.0], [3.0], [nonfinite]], dtype=jnp.float32)
        returns = jnp.zeros_like(predictions)

        running = per_horizon_running_rmse(predictions, returns, window_size=2)

        assert bool(jnp.isnan(running[0, 0]))
        assert float(running[1, 0]) == pytest.approx(np.sqrt(5.0))
        assert not bool(jnp.isfinite(running[2, 0]))

    def test_large_finite_errors_do_not_overflow(self) -> None:
        predictions = jnp.asarray([[2.0e20], [2.0e20]], dtype=jnp.float32)
        returns = jnp.zeros_like(predictions)

        with jax.debug_infs(True):
            running = per_horizon_running_rmse(predictions, returns, window_size=2)

        assert bool(jnp.isnan(running[0, 0]))
        assert bool(jnp.isfinite(running[1, 0]))
        np.testing.assert_allclose(
            np.asarray(running[1:]),
            np.full((1, 1), 2.0e20, dtype=np.float32),
        )

    @pytest.mark.parametrize("magnitude", [2.0e20, 1.0e38, 3.0e38])
    def test_large_finite_running_rmse_has_stable_eager_jit_and_jvp_gradients(
        self, magnitude: float
    ) -> None:
        predictions = jnp.full((2, 1), magnitude, dtype=jnp.float32)
        returns = jnp.zeros_like(predictions)

        def loss(values: jax.Array) -> jax.Array:
            return per_horizon_running_rmse(values, returns, window_size=2)[1, 0]

        expected = jnp.full_like(predictions, 0.5)
        eager = jax.grad(loss)(predictions)
        compiled = jax.jit(jax.grad(loss))(predictions)
        _primal, tangent = jax.jvp(
            lambda values: per_horizon_running_rmse(values, returns, window_size=2),
            (predictions,),
            (jnp.ones_like(predictions),),
        )

        chex.assert_trees_all_close(eager, expected, rtol=5e-6, atol=0.0)
        chex.assert_trees_all_close(compiled, expected, rtol=5e-6, atol=0.0)
        expected_tangent = jnp.asarray([[0.0], [1.0]], dtype=jnp.float32)
        chex.assert_trees_all_close(tangent, expected_tangent, rtol=5e-6, atol=0.0)

    def test_zero_running_rmse_uses_zero_eager_jit_and_jvp_gradient_convention(
        self,
    ) -> None:
        predictions = jnp.zeros((3, 2), dtype=jnp.float32)
        returns = jnp.zeros_like(predictions)

        def loss(values: jax.Array) -> jax.Array:
            return jnp.sum(per_horizon_running_rmse(values, returns, window_size=2))

        expected = jnp.zeros_like(predictions)
        eager = jax.grad(loss)(predictions)
        compiled = jax.jit(jax.grad(loss))(predictions)
        _primal, tangent = jax.jvp(
            lambda values: per_horizon_running_rmse(values, returns, window_size=2),
            (predictions,),
            (jnp.ones_like(predictions),),
        )

        chex.assert_trees_all_equal(eager, expected)
        chex.assert_trees_all_equal(compiled, expected)
        chex.assert_trees_all_equal(tangent, jnp.zeros_like(tangent))

    def test_shape(self) -> None:
        t, h = 50, 3
        preds = jnp.zeros((t, h))
        truths = jnp.ones((t, h))
        running = per_horizon_running_rmse(preds, truths, window_size=10)
        chex.assert_shape(running, (t, h))

    def test_constant_error(self) -> None:
        t, h = 30, 2
        preds = jnp.zeros((t, h))
        truths = jnp.ones((t, h)) * 2.0
        running = per_horizon_running_rmse(preds, truths, window_size=5)
        assert bool(jnp.all(jnp.isnan(running[:4])))
        np.testing.assert_allclose(np.asarray(running[4:]), 2.0, atol=1e-6)

    def test_running_rmse_survives_a_high_error_phase_before_a_settled_phase(self) -> None:
        """A trailing window must not inherit cancellation from an earlier large-error phase.

        A global float32 prefix sum loses every squared error that is smaller than the
        prefix's ulp, so once 2,000 unit errors have accumulated a settled phase of
        1e-4 errors reads as exactly zero RMSE. The window's own errors are all 1e-4,
        so every complete settled window must report 1e-4.
        """
        n_steps, window = 4000, 100
        for settled in (1e-3, 1e-4, 1e-5):
            errors = np.full((n_steps, 1), settled, dtype=np.float32)
            errors[:2000] = 1.0
            running = np.asarray(
                per_horizon_running_rmse(
                    jnp.asarray(errors), jnp.zeros((n_steps, 1), jnp.float32), window_size=window
                )
            )
            settled_windows = running[2000 + window :, 0]
            np.testing.assert_allclose(settled_windows, settled, rtol=1e-3, atol=0.0)

    @pytest.mark.parametrize(
        "window_size",
        [
            pytest.param(True, id="bool"),
            pytest.param(1.0, id="float"),
            pytest.param(np.int64(1), id="numpy-scalar"),
            pytest.param(jnp.asarray(1, dtype=jnp.int32), id="jax-scalar"),
        ],
    )
    def test_window_size_requires_exact_builtin_int(self, window_size: object) -> None:
        predictions = jnp.ones((3, 2))
        returns = jnp.zeros((3, 2))

        with pytest.raises(ValueError, match=r"^window_size must be a built-in int$"):
            per_horizon_running_rmse(
                predictions,
                returns,
                window_size=cast(int, window_size),
            )

    @pytest.mark.parametrize("window_size", [-1, 0, 4])
    def test_window_size_out_of_range_is_rejected(self, window_size: int) -> None:
        predictions = jnp.ones((3, 2))
        returns = jnp.zeros((3, 2))

        with pytest.raises(ValueError) as exc_info:
            per_horizon_running_rmse(predictions, returns, window_size=window_size)

        assert str(exc_info.value) == (
            "window_size must satisfy 1 <= window_size <= n_steps "
            f"(got window_size={window_size}, n_steps=3)"
        )

    @pytest.mark.parametrize(
        ("prediction_shape", "return_shape"),
        [
            pytest.param((), (), id="rank-zero"),
            pytest.param((3,), (3,), id="rank-one"),
            pytest.param((3, 2, 1), (3, 2, 1), id="rank-three"),
            pytest.param((3, 1), (3, 2), id="broadcast-horizon"),
            pytest.param((1, 2), (3, 2), id="broadcast-time"),
            pytest.param((0, 2), (0, 2), id="empty-time"),
            pytest.param((3, 0), (3, 0), id="empty-horizon"),
        ],
    )
    def test_inputs_require_equal_rank_two_nonempty_shape(
        self,
        prediction_shape: tuple[int, ...],
        return_shape: tuple[int, ...],
    ) -> None:
        predictions = jnp.zeros(prediction_shape)
        returns = jnp.zeros(return_shape)

        with pytest.raises(
            ValueError,
            match=r"must be rank-2 arrays with identical nonempty shape \(T, H\)",
        ):
            per_horizon_running_rmse(predictions, returns, window_size=1)

    def test_window_boundaries_are_bitexact_eager_and_static_jit(self) -> None:
        predictions = jnp.array(
            [[1.0, 2.0], [1.0, 2.0], [5.0, -10.0]],
            dtype=jnp.float32,
        )
        returns = jnp.zeros_like(predictions)
        compiled = jax.jit(
            per_horizon_running_rmse,
            static_argnames=("window_size",),
        )
        expected_by_window = {
            1: np.array([[1.0, 2.0], [1.0, 2.0], [5.0, 10.0]], dtype=np.float32),
            3: np.array(
                [[np.nan, np.nan], [np.nan, np.nan], [3.0, 6.0]],
                dtype=np.float32,
            ),
        }

        for window_size, expected in expected_by_window.items():
            eager = per_horizon_running_rmse(
                predictions,
                returns,
                window_size=window_size,
            )
            jitted = compiled(predictions, returns, window_size=window_size)
            np.testing.assert_array_equal(np.asarray(eager), expected)
            np.testing.assert_array_equal(np.asarray(jitted), expected)

        closed_over = jax.jit(
            lambda p, r: per_horizon_running_rmse(p, r, window_size=3)
        )
        np.testing.assert_array_equal(
            np.asarray(closed_over(predictions, returns)),
            expected_by_window[3],
        )

    def test_nonfinite_errors_remain_visible(self) -> None:
        predictions = jnp.array([[1.0, 1.0], [jnp.nan, jnp.inf]])
        returns = jnp.zeros_like(predictions)

        running = per_horizon_running_rmse(predictions, returns, window_size=2)

        assert bool(jnp.all(jnp.isnan(running[:, 0])))
        assert bool(jnp.isnan(running[0, 1]))
        assert bool(jnp.isinf(running[1, 1]))

    def test_dynamic_jit_window_is_named_and_default_call_recovers(self) -> None:
        predictions = jnp.tile(jnp.array([[3.0, -4.0]], dtype=jnp.float32), (100, 1))
        returns = jnp.zeros_like(predictions)
        compiled = jax.jit(per_horizon_running_rmse)

        with pytest.raises(ValueError, match=r"^window_size must be a built-in int"):
            compiled(predictions, returns, window_size=1)

        recovered = compiled(predictions, returns)
        expected = np.full((100, 2), np.nan, dtype=np.float32)
        expected[-1] = np.array([3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(np.asarray(recovered), expected)

    def test_jit_rejects_broadcasting_at_the_shape_boundary(self) -> None:
        predictions = jnp.ones((3, 2))
        returns = jnp.zeros((1, 2))

        with pytest.raises(
            ValueError,
            match=r"must be rank-2 arrays with identical nonempty shape \(T, H\)",
        ):
            jax.jit(lambda p, r: per_horizon_running_rmse(p, r, window_size=1))(
                predictions,
                returns,
            )

    def test_decay(self) -> None:
        # First half big errors, second half no errors. Running RMSE
        # should drop in the second half.
        t = 40
        h = 1
        preds = jnp.zeros((t, h))
        truths = jnp.zeros((t, h)).at[:20].set(5.0)
        running = per_horizon_running_rmse(preds, truths, window_size=5)
        # End of series: window contains only zero-error steps
        assert float(running[-1, 0]) < 0.01
        # Mid-series transition: some error
        assert float(running[19, 0]) > 0.5
