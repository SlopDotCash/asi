"""Tests for Step 1 baseline optimizers.

The optimizers under test (:mod:`alberta_framework.core.baseline_optimizers`)
are the non-meta-learning comparators for Alberta Plan Step 1 (continual
supervised learning with given features): AdaGain (Jacobsen et al. 2019,
"Meta-Descent for Online, Continual Prediction"), Adam (Kingma & Ba 2014),
RMSprop (Tieleman & Hinton 2012), and NADALINE — per-feature normalized LMS
after Sutton (1988).  Coverage is mechanism-level: state shapes and init
values, finiteness over a few updates, config round-trips, and one
analytical contract per method (Adam's exact t=1 bias correction and
decoupled AdamW decay; NADALINE's scale-invariant step and plain-LMS bias).

Hyperparameter literals here carry no tuning content: init tests restate
the constructor defaults explicitly (e.g. AdaGain's ``meta_step_size=0.001,
forgetting_rate=0.1``), and round-trip tests use arbitrary nondefault
values so serialization bugs cannot hide behind defaults.
"""

import chex
import jax.numpy as jnp
import pytest

from alberta_framework.core.baseline_optimizers import (
    NADALINE,
    AdaGain,
    AdaGainState,
    Adam,
    AdamParamState,
    AdamState,
    NadalineState,
    RMSprop,
    RMSpropParamState,
    RMSpropState,
)

# =============================================================================
# AdaGain
# =============================================================================


class TestAdaGain:
    """Tests for the AdaGain optimizer."""

    def test_init_shapes(self):
        """AdaGain ``init`` should produce per-feature gains and traces."""
        optimizer = AdaGain(
            initial_step_size=0.05,
            meta_step_size=0.001,
            forgetting_rate=0.1,
        )
        state = optimizer.init(feature_dim=10)

        assert isinstance(state, AdaGainState)
        chex.assert_shape(state.step_sizes, (10,))
        chex.assert_shape(state.gradient_trace, (10,))
        chex.assert_trees_all_close(state.step_sizes, jnp.full(10, 0.05))
        chex.assert_trees_all_close(state.gradient_trace, jnp.zeros(10))
        assert float(state.bias_step_size) == pytest.approx(0.05)
        assert float(state.meta_step_size) == pytest.approx(0.001)
        assert float(state.forgetting_rate) == pytest.approx(0.1)

    def test_update_returns_finite_metrics(self, sample_observation):
        """AdaGain ``update`` should produce finite outputs over 5 steps."""
        optimizer = AdaGain()
        state = optimizer.init(feature_dim=len(sample_observation))

        for i in range(5):
            error = jnp.array(1.0 + 0.1 * i)
            result = optimizer.update(state, error, sample_observation)
            chex.assert_tree_all_finite(result.weight_delta)
            chex.assert_tree_all_finite(result.bias_delta)
            chex.assert_tree_all_finite(result.new_state)
            for v in result.metrics.values():
                chex.assert_tree_all_finite(v)
            state = result.new_state

    def test_to_from_config_roundtrip(self):
        """AdaGain config roundtrip should preserve all parameters."""
        original = AdaGain(
            initial_step_size=0.1,
            meta_step_size=0.01,
            forgetting_rate=0.2,
        )
        config = original.to_config()
        kwargs = {k: v for k, v in config.items() if k != "type"}
        recreated = AdaGain(**kwargs)

        assert recreated.to_config() == config

    def test_full_forget_does_not_multiply_inf_traces(self) -> None:
        """forgetting_rate=1 drops leftover traces; 0 * inf must not freeze."""
        optimizer = AdaGain(
            initial_step_size=0.05,
            meta_step_size=0.0,
            forgetting_rate=1.0,
        )
        state = optimizer.init(feature_dim=2)
        state = state.replace(
            gradient_trace=jnp.full(2, jnp.inf, dtype=jnp.float32),
            bias_gradient_trace=jnp.asarray(jnp.inf, dtype=jnp.float32),
        )
        raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
        assert not bool(jnp.isfinite(raw))

        observation = jnp.asarray([0.5, -0.25], dtype=jnp.float32)
        error = jnp.asarray(1.0, dtype=jnp.float32)
        result = optimizer.update(state, error, observation)
        assert bool(result.update_applied)
        chex.assert_trees_all_close(result.new_state.gradient_trace, error * observation)
        assert float(result.new_state.bias_gradient_trace) == pytest.approx(1.0)


# =============================================================================
# Adam
# =============================================================================


class TestAdam:
    """Tests for the Adam optimizer."""

    def test_init_shapes(self):
        """Adam ``init`` should produce arrays with feature_dim shape."""
        optimizer = Adam(step_size=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
        state = optimizer.init(feature_dim=10)

        assert isinstance(state, AdamState)
        chex.assert_shape(state.m, (10,))
        chex.assert_shape(state.v, (10,))
        chex.assert_shape(state.bias_m, ())
        chex.assert_shape(state.bias_v, ())
        chex.assert_shape(state.t, ())
        chex.assert_trees_all_close(state.m, jnp.zeros(10))
        chex.assert_trees_all_close(state.v, jnp.zeros(10))
        assert float(state.t) == pytest.approx(0.0)
        assert float(state.step_size) == pytest.approx(0.001)
        assert float(state.beta1) == pytest.approx(0.9)
        assert float(state.beta2) == pytest.approx(0.999)
        assert float(state.eps) == pytest.approx(1e-8)

    def test_update_returns_finite_metrics(self, sample_observation):
        """Adam ``update`` should produce finite outputs over 5 steps."""
        optimizer = Adam(step_size=0.01)
        state = optimizer.init(feature_dim=len(sample_observation))

        for i in range(5):
            error = jnp.array(1.0 + 0.1 * i)
            result = optimizer.update(state, error, sample_observation)
            chex.assert_tree_all_finite(result.weight_delta)
            chex.assert_tree_all_finite(result.bias_delta)
            chex.assert_tree_all_finite(result.new_state)
            for v in result.metrics.values():
                chex.assert_tree_all_finite(v)
            state = result.new_state

    def test_update_from_gradient_finite(self):
        """Adam ``update_from_gradient`` should produce finite outputs."""
        optimizer = Adam(step_size=0.01)
        state = optimizer.init_for_shape((8, 4))

        for i in range(5):
            gradient = jnp.ones((8, 4)) * 0.1 * (i + 1)
            error = jnp.array(0.5)
            step, state = optimizer.update_from_gradient(state, gradient, error=error)
            chex.assert_shape(step, (8, 4))
            chex.assert_tree_all_finite(step)
            chex.assert_tree_all_finite(state)

    def test_to_from_config_roundtrip(self):
        """Adam config roundtrip should preserve all parameters."""
        original = Adam(step_size=0.005, beta1=0.85, beta2=0.995, eps=1e-7)
        config = original.to_config()
        # Drop the type tag for direct reconstruction
        kwargs = {k: v for k, v in config.items() if k != "type"}
        recreated = Adam(**kwargs)

        assert recreated.to_config() == config

    def test_state_init_for_shape(self):
        """``init_for_shape((3, 4))`` should produce 2D-shaped moments."""
        optimizer = Adam(step_size=0.001)
        state = optimizer.init_for_shape((3, 4))

        assert isinstance(state, AdamParamState)
        chex.assert_shape(state.m, (3, 4))
        chex.assert_shape(state.v, (3, 4))
        chex.assert_shape(state.t, ())
        chex.assert_trees_all_close(state.m, jnp.zeros((3, 4)))
        chex.assert_trees_all_close(state.v, jnp.zeros((3, 4)))

    def test_bias_correction_at_t1(self):
        """At t=1, bias-corrected first moment should equal the gradient.

        ``m_hat = m / (1 - beta1) = ((1 - beta1) * g) / (1 - beta1) = g``
        """
        optimizer = Adam(step_size=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
        state = optimizer.init_for_shape((4,))

        # Pure descent path: error=None means gradient is the loss gradient
        gradient = jnp.array([1.0, -2.0, 3.0, -4.0], dtype=jnp.float32)
        step, new_state = optimizer.update_from_gradient(state, gradient, error=None)

        # m_hat at t=1 should equal the gradient exactly
        m_hat = new_state.m / (1.0 - new_state.beta1**new_state.t)
        chex.assert_trees_all_close(m_hat, gradient, atol=1e-6)

        # v_hat at t=1 should equal gradient**2
        v_hat = new_state.v / (1.0 - new_state.beta2**new_state.t)
        chex.assert_trees_all_close(v_hat, gradient**2, atol=1e-6)

    def test_t_counter_increments(self):
        """``t`` should increment by 1 on each call."""
        optimizer = Adam(step_size=0.001)
        state = optimizer.init_for_shape((3,))
        assert float(state.t) == pytest.approx(0.0)

        for expected_t in (1.0, 2.0, 3.0):
            _, state = optimizer.update_from_gradient(
                state, jnp.ones(3), error=jnp.array(1.0)
            )
            assert float(state.t) == pytest.approx(expected_t)

    def test_decoupled_weight_decay_zero_gradient_is_pure_decay(self):
        """With zero gradient, ``param - step`` must equal ``(1 - lr*wd) * param``.

        Zero gradient keeps both moments at zero, so the Adam step is zero
        and the returned step reduces to the decoupled decay term
        ``lr * wd * param`` (AdamW; Loshchilov & Hutter 2019 -- matching the
        official UPGD repository's Adam, which applies
        ``p.data.add_(p.data, alpha=-wd*lr)`` before the Adam step).
        """
        optimizer = Adam(step_size=0.01, weight_decay=0.1)
        state = optimizer.init_for_shape((4,))
        param = jnp.array([1.0, -2.0, 0.5, 0.0], dtype=jnp.float32)

        step, _ = optimizer.update_from_gradient(
            state, jnp.zeros(4), error=None, param=param
        )
        chex.assert_trees_all_close(param - step, (1.0 - 0.01 * 0.1) * param, atol=1e-7)

    def test_weight_decay_zero_matches_plain_adam(self):
        """``weight_decay=0.0`` must reproduce the plain Adam step exactly."""
        plain = Adam(step_size=0.01)
        decayed = Adam(step_size=0.01, weight_decay=0.0)
        gradient = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)
        param = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)

        step_plain, _ = plain.update_from_gradient(
            plain.init_for_shape((3,)), gradient, error=None
        )
        step_decayed, _ = decayed.update_from_gradient(
            decayed.init_for_shape((3,)), gradient, error=None, param=param
        )
        chex.assert_trees_all_close(step_plain, step_decayed, atol=0.0)

    def test_weight_decay_requires_param(self):
        """Nonzero weight decay without ``param`` must fail loudly."""
        optimizer = Adam(step_size=0.01, weight_decay=0.01)
        state = optimizer.init_for_shape((3,))
        with pytest.raises(ValueError, match="param"):
            optimizer.update_from_gradient(state, jnp.ones(3), error=None)

    def test_weight_decay_config_roundtrip(self):
        """``weight_decay`` must survive the config roundtrip."""
        from alberta_framework.core.optimizers import optimizer_from_config

        original = Adam(step_size=0.005, weight_decay=0.02)
        config = original.to_config()
        assert config["weight_decay"] == pytest.approx(0.02)
        recreated = optimizer_from_config(config)
        assert isinstance(recreated, Adam)
        assert recreated.to_config() == config

    def test_legacy_config_without_weight_decay_still_loads(self):
        """Configs serialized before weight decay existed must still load."""
        from alberta_framework.core.optimizers import optimizer_from_config

        legacy = {
            "type": "Adam",
            "step_size": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        }
        recreated = optimizer_from_config(legacy)
        assert isinstance(recreated, Adam)
        assert recreated.to_config()["weight_decay"] == pytest.approx(0.0)

    def test_zero_beta_does_not_multiply_inf_moments(self):
        """beta1=beta2=0 times an infinite moment EMA is NaN."""
        optimizer = Adam(step_size=0.01, beta1=0.0, beta2=0.0)
        state = optimizer.init(feature_dim=3).replace(
            m=jnp.full(3, jnp.inf, dtype=jnp.float32),
            v=jnp.full(3, jnp.inf, dtype=jnp.float32),
            bias_m=jnp.asarray(jnp.inf, dtype=jnp.float32),
            bias_v=jnp.asarray(jnp.inf, dtype=jnp.float32),
        )
        raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
        assert not bool(jnp.isfinite(raw))

        result = optimizer.update(
            state,
            jnp.asarray(0.5, dtype=jnp.float32),
            jnp.ones(3, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        chex.assert_tree_all_finite(result.new_state)
        chex.assert_tree_all_finite(result.weight_delta)
        chex.assert_tree_all_finite(result.bias_delta)

    def test_zero_beta_recovers_poisoned_per_parameter_moments(self):
        """The checked MLP path has the same zero-decay recovery contract."""
        optimizer = Adam(step_size=0.01, beta1=0.0, beta2=0.0)
        state = optimizer.init_for_shape((2, 3)).replace(
            m=jnp.full((2, 3), jnp.inf, dtype=jnp.float32),
            v=jnp.full((2, 3), jnp.inf, dtype=jnp.float32),
        )

        result = optimizer.update_from_gradient_checked(
            state,
            jnp.full((2, 3), 0.25, dtype=jnp.float32),
        )

        assert bool(result.update_applied)
        chex.assert_tree_all_finite(result.step)
        chex.assert_tree_all_finite(result.new_state)

    def test_zero_config_does_not_relax_a_nonzero_persisted_beta(self):
        """Recovery requires the persisted tracker decay itself to be disabled."""
        optimizer = Adam(step_size=0.01, beta1=0.0, beta2=0.0)
        state = optimizer.init_for_shape((3,)).replace(
            beta1=jnp.asarray(0.5, dtype=jnp.float32),
            m=jnp.full(3, jnp.inf, dtype=jnp.float32),
        )

        result = optimizer.update_from_gradient_checked(
            state,
            jnp.ones(3, dtype=jnp.float32),
        )

        assert not bool(result.update_applied)
        chex.assert_trees_all_equal(result.new_state, state)
        chex.assert_trees_all_equal(result.step, jnp.zeros(3, dtype=jnp.float32))


# =============================================================================
# RMSprop
# =============================================================================


class TestRMSprop:
    """Tests for the RMSprop optimizer."""

    def test_init_shapes(self):
        """RMSprop ``init`` should produce arrays with feature_dim shape."""
        optimizer = RMSprop(step_size=0.001, decay=0.99, eps=1e-8)
        state = optimizer.init(feature_dim=10)

        assert isinstance(state, RMSpropState)
        chex.assert_shape(state.v, (10,))
        chex.assert_shape(state.bias_v, ())
        chex.assert_trees_all_close(state.v, jnp.zeros(10))
        assert float(state.step_size) == pytest.approx(0.001)
        assert float(state.decay) == pytest.approx(0.99)
        assert float(state.eps) == pytest.approx(1e-8)

    def test_update_returns_finite_metrics(self, sample_observation):
        """RMSprop ``update`` should produce finite outputs over 5 steps."""
        optimizer = RMSprop(step_size=0.01)
        state = optimizer.init(feature_dim=len(sample_observation))

        for i in range(5):
            error = jnp.array(1.0 + 0.1 * i)
            result = optimizer.update(state, error, sample_observation)
            chex.assert_tree_all_finite(result.weight_delta)
            chex.assert_tree_all_finite(result.bias_delta)
            chex.assert_tree_all_finite(result.new_state)
            for v in result.metrics.values():
                chex.assert_tree_all_finite(v)
            state = result.new_state

    def test_update_from_gradient_finite(self):
        """RMSprop ``update_from_gradient`` should produce finite outputs."""
        optimizer = RMSprop(step_size=0.01)
        state = optimizer.init_for_shape((8, 4))

        for i in range(5):
            gradient = jnp.ones((8, 4)) * 0.1 * (i + 1)
            error = jnp.array(0.5)
            step, state = optimizer.update_from_gradient(state, gradient, error=error)
            chex.assert_shape(step, (8, 4))
            chex.assert_tree_all_finite(step)
            chex.assert_tree_all_finite(state)

    def test_to_from_config_roundtrip(self):
        """RMSprop config roundtrip should preserve all parameters."""
        original = RMSprop(step_size=0.005, decay=0.95, eps=1e-7)
        config = original.to_config()
        kwargs = {k: v for k, v in config.items() if k != "type"}
        recreated = RMSprop(**kwargs)

        assert recreated.to_config() == config

    def test_state_init_for_shape(self):
        """``init_for_shape((3, 4))`` should produce 2D-shaped second moment."""
        optimizer = RMSprop(step_size=0.001)
        state = optimizer.init_for_shape((3, 4))

        assert isinstance(state, RMSpropParamState)
        chex.assert_shape(state.v, (3, 4))
        chex.assert_trees_all_close(state.v, jnp.zeros((3, 4)))

    def test_zero_decay_does_not_multiply_inf_second_moment(self):
        """decay=0 times an infinite squared-gradient EMA is NaN."""
        optimizer = RMSprop(step_size=0.01, decay=0.0)
        state = optimizer.init(feature_dim=3).replace(
            v=jnp.full(3, jnp.inf, dtype=jnp.float32),
            bias_v=jnp.asarray(jnp.inf, dtype=jnp.float32),
        )
        raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
        assert not bool(jnp.isfinite(raw))

        result = optimizer.update(
            state,
            jnp.asarray(0.5, dtype=jnp.float32),
            jnp.ones(3, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        chex.assert_tree_all_finite(result.new_state)
        chex.assert_tree_all_finite(result.weight_delta)
        chex.assert_tree_all_finite(result.bias_delta)

    def test_zero_decay_recovers_poisoned_per_parameter_moment(self):
        """The checked MLP path skips a disabled poisoned second moment."""
        optimizer = RMSprop(step_size=0.01, decay=0.0)
        state = optimizer.init_for_shape((2, 3)).replace(
            v=jnp.full((2, 3), jnp.inf, dtype=jnp.float32)
        )

        result = optimizer.update_from_gradient_checked(
            state,
            jnp.full((2, 3), 0.25, dtype=jnp.float32),
        )

        assert bool(result.update_applied)
        chex.assert_tree_all_finite(result.step)
        chex.assert_tree_all_finite(result.new_state)

    def test_zero_config_does_not_relax_a_nonzero_persisted_decay(self):
        """A nonzero persisted decay still consumes and validates its history."""
        optimizer = RMSprop(step_size=0.01, decay=0.0)
        state = optimizer.init_for_shape((3,)).replace(
            decay=jnp.asarray(0.5, dtype=jnp.float32),
            v=jnp.full(3, jnp.inf, dtype=jnp.float32),
        )

        result = optimizer.update_from_gradient_checked(
            state,
            jnp.ones(3, dtype=jnp.float32),
        )

        assert not bool(result.update_applied)
        chex.assert_trees_all_equal(result.new_state, state)
        chex.assert_trees_all_equal(result.step, jnp.zeros(3, dtype=jnp.float32))

    def test_unit_decay_does_not_multiply_inf_squared_gradient(self):
        """decay=1 freezes the EMA, so an overflowing square must not enter it."""
        optimizer = RMSprop(step_size=0.01, decay=1.0)
        state = optimizer.init(feature_dim=3)
        observation = jnp.full(3, 1e30, dtype=jnp.float32)
        error = jnp.asarray(1.0, dtype=jnp.float32)
        raw = (1.0 - state.decay) * (-error * observation) ** 2
        assert not bool(jnp.all(jnp.isfinite(raw)))

        result = optimizer.update(state, error, observation)

        assert bool(result.update_applied)
        chex.assert_tree_all_finite(result.new_state)
        chex.assert_tree_all_finite(result.weight_delta)
        chex.assert_tree_all_finite(result.bias_delta)
        chex.assert_trees_all_equal(result.new_state.v, state.v)
        chex.assert_trees_all_equal(result.new_state.bias_v, state.bias_v)

    def test_unit_decay_still_applies_an_ordinary_update(self):
        """A frozen EMA keeps producing the ordinary normalized descent step."""
        optimizer = RMSprop(step_size=0.01, decay=1.0)
        state = optimizer.init(feature_dim=2).replace(
            v=jnp.full(2, 4.0, dtype=jnp.float32),
        )

        result = optimizer.update(
            state,
            jnp.asarray(0.5, dtype=jnp.float32),
            jnp.ones(2, dtype=jnp.float32),
        )

        assert bool(result.update_applied)
        chex.assert_trees_all_close(result.new_state.v, state.v)
        expected = 0.01 * 0.5 / (2.0 + float(state.eps))
        chex.assert_trees_all_close(
            result.weight_delta,
            jnp.full(2, expected, dtype=jnp.float32),
            rtol=1e-6,
        )

    def test_unit_decay_does_not_poison_the_per_parameter_moment(self):
        """The checked MLP path shares the frozen-EMA guard."""
        optimizer = RMSprop(step_size=0.01, decay=1.0)
        state = optimizer.init_for_shape((2, 3))

        result = optimizer.update_from_gradient_checked(
            state,
            jnp.full((2, 3), 1e30, dtype=jnp.float32),
        )

        assert bool(result.update_applied)
        chex.assert_tree_all_finite(result.step)
        chex.assert_tree_all_finite(result.new_state)
        chex.assert_trees_all_equal(result.new_state.v, state.v)


# =============================================================================
# NADALINE
# =============================================================================


class TestNADALINE:
    """Tests for the NADALINE optimizer."""

    def test_init_shapes(self):
        """NADALINE ``init`` should produce per-feature second-moment array."""
        optimizer = NADALINE(step_size=0.01, decay=0.99, eps=1e-8)
        state = optimizer.init(feature_dim=10)

        assert isinstance(state, NadalineState)
        chex.assert_shape(state.feature_second_moment, (10,))
        chex.assert_trees_all_close(state.feature_second_moment, jnp.zeros(10))
        assert float(state.step_size) == pytest.approx(0.01)
        assert float(state.decay) == pytest.approx(0.99)
        assert float(state.eps) == pytest.approx(1e-8)

    def test_update_returns_finite_metrics(self, sample_observation):
        """NADALINE ``update`` should produce finite outputs over 5 steps."""
        optimizer = NADALINE(step_size=0.01)
        state = optimizer.init(feature_dim=len(sample_observation))

        for i in range(5):
            error = jnp.array(1.0 + 0.1 * i)
            result = optimizer.update(state, error, sample_observation)
            chex.assert_tree_all_finite(result.weight_delta)
            chex.assert_tree_all_finite(result.bias_delta)
            chex.assert_tree_all_finite(result.new_state)
            for v in result.metrics.values():
                chex.assert_tree_all_finite(v)
            state = result.new_state

    def test_to_from_config_roundtrip(self):
        """NADALINE config roundtrip should preserve all parameters."""
        original = NADALINE(step_size=0.05, decay=0.95, eps=1e-7)
        config = original.to_config()
        kwargs = {k: v for k, v in config.items() if k != "type"}
        recreated = NADALINE(**kwargs)

        assert recreated.to_config() == config

    def test_normalization_reduces_step_for_large_features(self):
        """Per-feature normalization should make step magnitude scale-invariant.

        Feeding ``x = 100 * ones`` should produce a weight-step magnitude
        roughly equal to feeding ``x = ones``, because each weight is
        scaled by ``1 / max(eps, EMA(x_i^2))``.
        """
        optimizer = NADALINE(step_size=0.01, decay=0.5, eps=1e-8)
        feature_dim = 5
        error = jnp.array(1.0)

        # Run several steps with x = 1 to let EMA converge
        state_small = optimizer.init(feature_dim)
        small_obs = jnp.ones(feature_dim)
        for _ in range(20):
            r = optimizer.update(state_small, error, small_obs)
            state_small = r.new_state
        result_small = optimizer.update(state_small, error, small_obs)
        small_step_norm = float(jnp.linalg.norm(result_small.weight_delta))

        # Run several steps with x = 100 to let EMA converge to a much
        # larger value (10000), which the denominator will normalize by
        state_large = optimizer.init(feature_dim)
        large_obs = jnp.ones(feature_dim) * 100.0
        for _ in range(20):
            r = optimizer.update(state_large, error, large_obs)
            state_large = r.new_state
        result_large = optimizer.update(state_large, error, large_obs)
        large_step_norm = float(jnp.linalg.norm(result_large.weight_delta))

        # Without normalization, large_step_norm would be ~100x larger.
        # With normalization, alpha * x / E[x^2] ~ alpha * x / x^2 = alpha / x,
        # so the ratio of large to small should be roughly 1/100, not 100.
        ratio = large_step_norm / small_step_norm
        assert ratio < 0.1, (
            f"Expected normalization to keep step magnitude similar; "
            f"got ratio {ratio:.4f} (small={small_step_norm:.6f}, "
            f"large={large_step_norm:.6f})"
        )

    def test_bias_uses_plain_lms(self):
        """NADALINE bias delta should equal ``alpha * error`` with no normalization."""
        optimizer = NADALINE(step_size=0.05)
        state = optimizer.init(feature_dim=4)

        observation = jnp.array([0.5, 1.0, 2.0, 3.0])
        error = jnp.array(0.7)

        result = optimizer.update(state, error, observation)
        # bias_delta = alpha * error
        assert float(result.bias_delta) == pytest.approx(0.05 * 0.7, abs=1e-6)

    def test_zero_decay_does_not_multiply_inf_second_moment(self):
        """decay=0 times an infinite feature second-moment EMA is NaN."""
        optimizer = NADALINE(step_size=0.01, decay=0.0)
        state = optimizer.init(feature_dim=3).replace(
            feature_second_moment=jnp.full(3, jnp.inf, dtype=jnp.float32),
        )
        raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
        assert not bool(jnp.isfinite(raw))

        result = optimizer.update(
            state,
            jnp.asarray(0.5, dtype=jnp.float32),
            jnp.ones(3, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        chex.assert_tree_all_finite(result.new_state)
        chex.assert_tree_all_finite(result.weight_delta)
        chex.assert_tree_all_finite(result.bias_delta)

    def test_unit_decay_does_not_multiply_inf_squared_feature(self):
        """decay=1 freezes E[x^2], so an overflowing square must not enter it."""
        optimizer = NADALINE(step_size=0.01, decay=1.0)
        state = optimizer.init(feature_dim=3).replace(
            feature_second_moment=jnp.full(3, 4.0, dtype=jnp.float32),
        )
        observation = jnp.full(3, 1e30, dtype=jnp.float32)
        raw = (1.0 - state.decay) * observation**2
        assert not bool(jnp.all(jnp.isfinite(raw)))

        result = optimizer.update(
            state,
            jnp.asarray(0.5, dtype=jnp.float32),
            observation,
        )

        assert bool(result.update_applied)
        chex.assert_tree_all_finite(result.new_state)
        chex.assert_tree_all_finite(result.weight_delta)
        chex.assert_tree_all_finite(result.bias_delta)
        chex.assert_trees_all_equal(
            result.new_state.feature_second_moment,
            state.feature_second_moment,
        )

    def test_unit_decay_still_normalizes_by_the_frozen_moment(self):
        """A frozen E[x^2] keeps producing the ordinary normalized step."""
        optimizer = NADALINE(step_size=0.01, decay=1.0)
        state = optimizer.init(feature_dim=2).replace(
            feature_second_moment=jnp.full(2, 4.0, dtype=jnp.float32),
        )

        result = optimizer.update(
            state,
            jnp.asarray(0.5, dtype=jnp.float32),
            jnp.full(2, 2.0, dtype=jnp.float32),
        )

        assert bool(result.update_applied)
        chex.assert_trees_all_close(
            result.new_state.feature_second_moment,
            state.feature_second_moment,
        )
        chex.assert_trees_all_close(
            result.weight_delta,
            jnp.full(2, 0.01 * 0.5 * 2.0 / 4.0, dtype=jnp.float32),
            rtol=1e-6,
        )
