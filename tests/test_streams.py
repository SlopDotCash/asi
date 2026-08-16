"""Tests for experience streams."""

from fractions import Fraction

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework import (
    AbruptChangeStream,
    CyclicStream,
    DynamicScaleShiftStream,
    PeriodicChangeStream,
    RandomWalkStream,
    ScaleDriftStream,
    ScaledStreamWrapper,
    SuttonExperiment1Stream,
    TimeStep,
    make_scale_range,
)

_INT32_MAX = 2**31 - 1
_INVALID_SCHEDULE_MODULI = (0, -1, False, True, 1.5, None, 2**31, 10**100)


class TestRandomWalkStream:
    """Tests for the RandomWalkStream class."""

    def test_init_creates_valid_state(self, rng_key):
        """Stream init should create valid state with correct shapes."""
        stream = RandomWalkStream(feature_dim=10, drift_rate=0.001)
        state = stream.init(rng_key)

        assert state.key is not None
        chex.assert_shape(state.true_weights, (10,))

    def test_rejects_invalid_feature_dim(self):
        """Should reject non-positive, bool, or non-integer feature_dim."""
        for feature_dim in (0, -1, True, 2.5):
            with pytest.raises(ValueError, match="feature_dim"):
                RandomWalkStream(feature_dim=feature_dim)

    def test_rejects_non_finite_or_negative_float_params(self):
        """Should reject NaN/inf/negative drift_rate, noise_std, feature_std."""
        for name, value in (
            ("drift_rate", float("nan")),
            ("drift_rate", float("inf")),
            ("drift_rate", -1.0),
            ("noise_std", float("nan")),
            ("feature_std", -0.5),
        ):
            with pytest.raises(ValueError, match=name):
                RandomWalkStream(feature_dim=4, **{name: value})

    def test_step_produces_valid_timestep(self, rng_key):
        """Step should produce valid observation and target."""
        stream = RandomWalkStream(feature_dim=10)
        state = stream.init(rng_key)

        timestep, new_state = stream.step(state, jnp.array(0))

        chex.assert_shape(timestep.observation, (10,))
        chex.assert_shape(timestep.target, (1,))
        chex.assert_tree_all_finite(timestep.observation)
        chex.assert_tree_all_finite(timestep.target)

    def test_feature_dim_property(self):
        """Feature dim property should return correct dimension."""
        stream = RandomWalkStream(feature_dim=20)
        assert stream.feature_dim == 20

    def test_weights_drift_over_time(self, rng_key):
        """True weights should change from step to step."""
        stream = RandomWalkStream(feature_dim=10, drift_rate=0.1)  # High drift
        state = stream.init(rng_key)

        initial_weights = state.true_weights

        for i in range(10):
            _, state = stream.step(state, jnp.array(i))

        # Weights should have changed
        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(initial_weights, state.true_weights)

    def test_deterministic_with_same_key(self, rng_key):
        """Same key should produce same sequence."""
        stream = RandomWalkStream(feature_dim=10)

        state1 = stream.init(rng_key)
        timestep1, _ = stream.step(state1, jnp.array(0))

        state2 = stream.init(rng_key)
        timestep2, _ = stream.step(state2, jnp.array(0))

        chex.assert_trees_all_close(timestep1.observation, timestep2.observation)
        chex.assert_trees_all_close(timestep1.target, timestep2.target)

    def test_targets_are_non_constant(self, rng_key):
        """Targets should vary due to random features and noise."""
        stream = RandomWalkStream(feature_dim=10)
        state = stream.init(rng_key)

        targets = []
        for i in range(100):
            timestep, state = stream.step(state, jnp.array(i))
            targets.append(float(timestep.target[0]))

        # Targets should not all be the same
        assert len(set(targets)) > 1


class TestAbruptChangeStream:
    """Tests for the AbruptChangeStream class."""

    def test_rejects_non_finite_or_negative_float_params(self):
        """Should reject NaN/inf/negative noise_std and feature_std."""
        for name, value in (
            ("noise_std", float("nan")),
            ("noise_std", float("inf")),
            ("feature_std", -1.0),
        ):
            with pytest.raises(ValueError, match=name):
                AbruptChangeStream(feature_dim=4, **{name: value})

    @pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
    def test_initial_weights_last_for_first_full_segment(self, rng_key, compiled):
        """Initialized weights govern exactly one complete change interval."""
        change_interval = 3
        stream = AbruptChangeStream(feature_dim=10, change_interval=change_interval)
        state = stream.init(rng_key)
        initial_weights = state.true_weights
        step = jax.jit(stream.step) if compiled else stream.step

        for i in range(change_interval):
            _, state = step(state, jnp.array(i, dtype=jnp.int32))
            chex.assert_trees_all_equal(state.true_weights, initial_weights)

        _, state = step(state, jnp.array(change_interval, dtype=jnp.int32))
        assert not bool(jnp.array_equal(state.true_weights, initial_weights))

    def test_generates_valid_timesteps(self, rng_key):
        """Should generate valid TimeStep instances."""
        stream = AbruptChangeStream(feature_dim=5)
        state = stream.init(rng_key)

        for i in range(50):
            timestep, state = stream.step(state, jnp.array(i))
            assert isinstance(timestep, TimeStep)
            chex.assert_tree_all_finite(timestep.observation)


class TestSuttonExperiment1Stream:
    """Tests for the SuttonExperiment1Stream class."""

    def test_correct_feature_dim(self):
        """Feature dim should be num_relevant + num_irrelevant."""
        stream = SuttonExperiment1Stream(num_relevant=5, num_irrelevant=15)
        assert stream.feature_dim == 20

    def test_initial_signs_are_positive(self, rng_key):
        """All initial signs should be +1."""
        stream = SuttonExperiment1Stream()
        state = stream.init(rng_key)

        assert jnp.all(state.signs == 1.0)

    def test_sign_flips_at_interval(self, rng_key):
        """One sign should flip every change_interval steps."""
        stream = SuttonExperiment1Stream(change_interval=20)
        state = stream.init(rng_key)

        initial_signs = state.signs.copy()

        # Step past the change interval
        for i in range(21):
            _, state = stream.step(state, jnp.array(i))

        # At least one sign should have changed
        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(initial_signs, state.signs)

        # Exactly one sign should be different
        num_changes = jnp.sum(initial_signs != state.signs)
        assert num_changes == 1

    def test_target_only_depends_on_relevant_inputs(self, rng_key):
        """Target should only depend on first num_relevant inputs."""
        stream = SuttonExperiment1Stream(num_relevant=5, num_irrelevant=15)
        state = stream.init(rng_key)

        timestep, new_state = stream.step(state, jnp.array(0))

        # At step 0, no flip happens (step_count > 0 check), so signs remain all 1
        # Target = sum of first 5 inputs (weighted by signs which are all 1)
        expected = jnp.sum(timestep.observation[:5])

        assert jnp.isclose(timestep.target[0], expected, rtol=1e-5)


class TestCyclicStream:
    """Tests for the CyclicStream class."""

    def test_rejects_non_finite_or_negative_float_params(self):
        """Should reject NaN/inf/negative noise_std and feature_std."""
        for name, value in (
            ("noise_std", float("nan")),
            ("noise_std", float("inf")),
            ("feature_std", -1.0),
        ):
            with pytest.raises(ValueError, match=name):
                CyclicStream(feature_dim=4, **{name: value})

    def test_cycles_through_configurations(self, rng_key):
        """Should cycle through configurations."""
        stream = CyclicStream(
            feature_dim=10,
            cycle_length=5,
            num_configurations=4,
        )
        state = stream.init(rng_key)

        # Track which configuration index is used
        config_indices = []
        for i in range(25):  # Go through all 4 configs plus more
            config_idx = (state.step_count // 5) % 4
            config_indices.append(int(config_idx))
            _, state = stream.step(state, jnp.array(i))

        # Should see all 4 configurations
        assert 0 in config_indices
        assert 1 in config_indices
        assert 2 in config_indices
        assert 3 in config_indices

    def test_same_config_produces_consistent_weights(self, rng_key):
        """Same configuration should use same weights."""
        stream = CyclicStream(
            feature_dim=10,
            cycle_length=10,
            num_configurations=2,
            noise_std=0.0,  # No noise for easier testing
        )
        state = stream.init(rng_key)

        # Get the stored configurations
        config0_weights = state.configurations[0]

        # After one full cycle, we should be back to config 0
        for i in range(20):  # Go through both configs
            _, state = stream.step(state, jnp.array(i))

        # Config 0 weights should be unchanged (stored in configurations)
        chex.assert_trees_all_close(config0_weights, state.configurations[0])

    def test_generates_valid_timesteps(self, rng_key):
        """Should generate valid TimeStep instances."""
        stream = CyclicStream(feature_dim=5)
        state = stream.init(rng_key)

        for i in range(50):
            timestep, state = stream.step(state, jnp.array(i))
            assert isinstance(timestep, TimeStep)
            chex.assert_tree_all_finite(timestep.observation)


class TestPeriodicChangeStream:
    """Tests for the PeriodicChangeStream class."""

    def test_rejects_non_finite_or_negative_float_params(self):
        """Should reject NaN/inf/negative amplitude, noise_std, feature_std."""
        for name, value in (
            ("amplitude", float("nan")),
            ("amplitude", float("inf")),
            ("noise_std", -0.5),
            ("feature_std", float("nan")),
        ):
            with pytest.raises(ValueError, match=name):
                PeriodicChangeStream(feature_dim=4, **{name: value})

    def test_init_creates_valid_state(self, rng_key):
        """Stream init should create valid state with correct shapes."""
        stream = PeriodicChangeStream(feature_dim=10, period=100)
        state = stream.init(rng_key)

        assert state.key is not None
        chex.assert_shape(state.base_weights, (10,))
        chex.assert_shape(state.phases, (10,))
        assert state.step_count == 0

    def test_step_produces_valid_timestep(self, rng_key):
        """Step should produce valid observation and target."""
        stream = PeriodicChangeStream(feature_dim=10)
        state = stream.init(rng_key)

        timestep, new_state = stream.step(state, jnp.array(0))

        chex.assert_shape(timestep.observation, (10,))
        chex.assert_shape(timestep.target, (1,))
        chex.assert_tree_all_finite(timestep.observation)
        chex.assert_tree_all_finite(timestep.target)

    def test_feature_dim_property(self):
        """Feature dim property should return correct dimension."""
        stream = PeriodicChangeStream(feature_dim=20)
        assert stream.feature_dim == 20

    def test_weights_oscillate_periodically(self, rng_key):
        """Weights should return to similar values after one period."""
        period = 100
        stream = PeriodicChangeStream(feature_dim=5, period=period, amplitude=1.0)
        state = stream.init(rng_key)

        # Get weights at step 0 (after init, step_count=0)
        t = 0
        oscillation_0 = 1.0 * jnp.sin(2.0 * jnp.pi * t / period + state.phases)
        weights_at_0 = state.base_weights + oscillation_0

        # Run one full period
        for i in range(period):
            _, state = stream.step(state, jnp.array(i))

        # Get weights after one period
        t = period
        oscillation_period = 1.0 * jnp.sin(2.0 * jnp.pi * t / period + state.phases)
        weights_at_period = state.base_weights + oscillation_period

        # Should be back to same weights (sin(2π + φ) = sin(φ))
        chex.assert_trees_all_close(weights_at_0, weights_at_period, atol=1e-5)

    def test_weights_differ_at_half_period(self, rng_key):
        """Weights at half period should differ from initial (unless phase happens to align)."""
        period = 100
        stream = PeriodicChangeStream(feature_dim=10, period=period, amplitude=2.0)
        state = stream.init(rng_key)

        # Run to half period
        for i in range(period // 2):
            _, state = stream.step(state, jnp.array(i))

        # At t=period/2, oscillation should be different from t=0
        # sin(π + φ) = -sin(φ), so weights should differ
        t_half = period // 2
        oscillation_half = 2.0 * jnp.sin(2.0 * jnp.pi * t_half / period + state.phases)
        weights_half = state.base_weights + oscillation_half

        t_0 = 0
        oscillation_0 = 2.0 * jnp.sin(2.0 * jnp.pi * t_0 / period + state.phases)
        weights_0 = state.base_weights + oscillation_0

        # Weights should differ (they're inverted around base)
        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(weights_0, weights_half)

    def test_deterministic_with_same_key(self, rng_key):
        """Same key should produce same sequence."""
        stream = PeriodicChangeStream(feature_dim=10)

        state1 = stream.init(rng_key)
        timestep1, _ = stream.step(state1, jnp.array(0))

        state2 = stream.init(rng_key)
        timestep2, _ = stream.step(state2, jnp.array(0))

        chex.assert_trees_all_close(timestep1.observation, timestep2.observation)
        chex.assert_trees_all_close(timestep1.target, timestep2.target)


class TestScaledStreamWrapper:
    """Tests for the ScaledStreamWrapper class."""

    def test_scales_observations(self, rng_key):
        """Wrapper should scale observations by feature_scales."""
        inner = RandomWalkStream(feature_dim=5, feature_std=1.0)
        scales = jnp.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        wrapped = ScaledStreamWrapper(inner, feature_scales=scales)

        # Get inner stream output
        inner_state = inner.init(rng_key)
        inner_timestep, _ = inner.step(inner_state, jnp.array(0))

        # Get wrapped stream output (same key)
        wrapped_state = wrapped.init(rng_key)
        wrapped_timestep, _ = wrapped.step(wrapped_state, jnp.array(0))

        # Wrapped observation should be inner * scales
        expected = inner_timestep.observation * scales
        chex.assert_trees_all_close(wrapped_timestep.observation, expected)

    def test_preserves_target(self, rng_key):
        """Wrapper should not modify targets."""
        inner = RandomWalkStream(feature_dim=5)
        scales = jnp.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        wrapped = ScaledStreamWrapper(inner, feature_scales=scales)

        inner_state = inner.init(rng_key)
        inner_timestep, _ = inner.step(inner_state, jnp.array(0))

        wrapped_state = wrapped.init(rng_key)
        wrapped_timestep, _ = wrapped.step(wrapped_state, jnp.array(0))

        # Target should be unchanged
        chex.assert_trees_all_close(wrapped_timestep.target, inner_timestep.target)

    def test_feature_dim_property(self):
        """Feature dim should match inner stream."""
        inner = RandomWalkStream(feature_dim=20)
        wrapped = ScaledStreamWrapper(inner, feature_scales=jnp.ones(20))
        assert wrapped.feature_dim == 20

    def test_rejects_mismatched_scales(self):
        """Should raise error if scales don't match feature_dim."""
        inner = RandomWalkStream(feature_dim=10)
        scales = jnp.ones(5)  # Wrong size

        with pytest.raises(ValueError, match="must match"):
            ScaledStreamWrapper(inner, feature_scales=scales)

    def test_rejects_non_finite_scales(self):
        """Should raise error if scales contain NaN or infinity."""
        inner = RandomWalkStream(feature_dim=5)
        for scales in (
            jnp.array([1.0, float("nan"), 2.0, 3.0, 4.0]),
            jnp.array([1.0, float("inf"), 2.0, 3.0, 4.0]),
        ):
            with pytest.raises(ValueError, match="finite"):
                ScaledStreamWrapper(inner, feature_scales=scales)

    def test_works_with_different_streams(self, rng_key):
        """Should work with any stream implementing the protocol."""
        scales = jnp.array([0.01, 0.1, 1.0, 10.0, 100.0])

        # Test with AbruptChangeStream
        stream1 = ScaledStreamWrapper(AbruptChangeStream(feature_dim=5), feature_scales=scales)
        state1 = stream1.init(rng_key)
        ts1, _ = stream1.step(state1, jnp.array(0))
        chex.assert_shape(ts1.observation, (5,))

        # Test with CyclicStream
        stream2 = ScaledStreamWrapper(CyclicStream(feature_dim=5), feature_scales=scales)
        state2 = stream2.init(rng_key)
        ts2, _ = stream2.step(state2, jnp.array(0))
        chex.assert_shape(ts2.observation, (5,))


class TestMakeScaleRange:
    """Tests for the make_scale_range utility function."""

    def test_log_spaced_range(self):
        """Log-spaced scales should span min to max logarithmically."""
        scales = make_scale_range(5, min_scale=0.01, max_scale=100.0, log_spaced=True)

        chex.assert_shape(scales, (5,))
        chex.assert_trees_all_close(scales[0], jnp.array(0.01), rtol=1e-5)
        chex.assert_trees_all_close(scales[-1], jnp.array(100.0), rtol=1e-5)
        # Middle value should be geometric mean ≈ 1.0
        chex.assert_trees_all_close(scales[2], jnp.array(1.0), rtol=0.1)

    @pytest.mark.parametrize(
        ("min_scale", "max_scale"),
        [
            pytest.param(-1.0, 100.0, id="min-negative"),
            pytest.param(0.01, -100.0, id="max-negative"),
            pytest.param(0.0, 100.0, id="min-zero"),
            pytest.param(0.01, 0.0, id="max-zero"),
            pytest.param(float("nan"), 100.0, id="min-nan"),
            pytest.param(0.01, float("nan"), id="max-nan"),
            pytest.param(float("inf"), 100.0, id="min-positive-infinity"),
            pytest.param(0.01, float("inf"), id="max-positive-infinity"),
            pytest.param(float("-inf"), 100.0, id="min-negative-infinity"),
            pytest.param(0.01, float("-inf"), id="max-negative-infinity"),
            pytest.param(1e-50, 1.0, id="min-below-float32-range"),
            pytest.param(1.0, 1e-50, id="max-below-float32-range"),
            pytest.param(1e100, 1.0, id="min-above-float32-range"),
            pytest.param(1.0, 1e100, id="max-above-float32-range"),
        ],
    )
    def test_log_spaced_range_rejects_invalid_bounds(
        self,
        min_scale,
        max_scale,
    ):
        """Log-spaced scales require bounds in JAX's normal float32 domain."""
        with pytest.raises(ValueError, match="normal float32"):
            make_scale_range(
                5,
                min_scale=min_scale,
                max_scale=max_scale,
                log_spaced=True,
            )

    @pytest.mark.parametrize(
        ("min_scale", "max_scale"),
        [
            pytest.param(0.01, 100.0, id="ascending"),
            pytest.param(100.0, 0.01, id="descending"),
            pytest.param(np.float32(0.123), np.float32(0.123), id="equal"),
            pytest.param(
                np.float32(3.0),
                np.nextafter(np.float32(3.0), np.float32("inf")),
                id="adjacent-ascending",
            ),
            pytest.param(
                np.nextafter(np.float32(3.0), np.float32("inf")),
                np.float32(3.0),
                id="adjacent-descending",
            ),
            pytest.param(
                float(np.finfo(np.float32).tiny),
                float(np.finfo(np.float32).max),
                id="float32-boundaries",
            ),
            pytest.param(
                float(np.finfo(np.float32).tiny),
                float(np.finfo(np.float32).tiny),
                id="equal-float32-tiny",
            ),
            pytest.param(
                float(np.finfo(np.float32).max),
                float(np.finfo(np.float32).max),
                id="equal-float32-max",
            ),
        ],
    )
    def test_log_spaced_range_has_valid_float32_postconditions(
        self,
        min_scale,
        max_scale,
    ):
        """Generated ranges preserve canonical endpoints and ordering."""
        scales = make_scale_range(
            5,
            min_scale=min_scale,
            max_scale=max_scale,
            log_spaced=True,
        )
        values = np.asarray(scales)
        lower_bound = np.float32(min(min_scale, max_scale))
        upper_bound = np.float32(max(min_scale, max_scale))

        assert values.dtype == np.dtype(np.float32)
        assert values[0] == np.float32(min_scale)
        assert values[-1] == np.float32(max_scale)
        assert np.all(np.isfinite(values))
        assert np.all(values > 0.0)
        assert np.all(values >= lower_bound)
        assert np.all(values <= upper_bound)
        if min_scale <= max_scale:
            assert np.all(values[:-1] <= values[1:])
        else:
            assert np.all(values[:-1] >= values[1:])

    @pytest.mark.parametrize(
        "generated",
        [
            pytest.param(jnp.array([1.0, jnp.nan, 10.0]), id="non-finite"),
            pytest.param(jnp.array([1.0, 0.0, 10.0]), id="non-positive"),
            pytest.param(jnp.array([1.0, 9.0, 8.0, 10.0]), id="unordered"),
        ],
    )
    def test_log_spaced_range_rejects_invalid_generated_values(self, monkeypatch, generated):
        """Backend numerical failures must fail closed before scales escape."""
        monkeypatch.setattr(
            "alberta_framework.streams.synthetic.np.geomspace",
            lambda *args, **kwargs: generated,
        )

        with pytest.raises(ValueError, match="generated log-spaced scales"):
            make_scale_range(
                generated.size,
                min_scale=1.0,
                max_scale=10.0,
                log_spaced=True,
            )

    def test_linear_spaced_range(self):
        """Linear-spaced scales should span min to max linearly."""
        scales = make_scale_range(5, min_scale=0.0, max_scale=100.0, log_spaced=False)

        chex.assert_shape(scales, (5,))
        chex.assert_trees_all_close(scales[0], jnp.array(0.0), atol=1e-5)
        chex.assert_trees_all_close(scales[-1], jnp.array(100.0), rtol=1e-5)
        # Middle value should be arithmetic mean = 50.0
        chex.assert_trees_all_close(scales[2], jnp.array(50.0), rtol=1e-5)

    def test_default_range(self):
        """Default range should be 0.001 to 1000."""
        scales = make_scale_range(7)

        chex.assert_shape(scales, (7,))
        chex.assert_trees_all_close(scales[0], jnp.array(0.001), rtol=1e-5)
        chex.assert_trees_all_close(scales[-1], jnp.array(1000.0), rtol=1e-5)


class TestDynamicScaleShiftStream:
    """Tests for the DynamicScaleShiftStream class."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_scale": 0.0},
            {"min_scale": -1.0},
            {"min_scale": float("nan")},
            {"max_scale": float("inf")},
            {"min_scale": 10.0, "max_scale": 1.0},
            {"max_scale": 1e39},
            {"max_scale": Fraction(10**400)},
            {"min_scale": True},
            {"max_scale": True},
            {"min_scale": 1e-40},
        ],
    )
    def test_rejects_nonpositive_nonfinite_or_reversed_scale_bounds(self, kwargs):
        """These bounds produced NaN/inf observations or an inverted log-uniform range."""
        with pytest.raises(ValueError, match="scale"):
            DynamicScaleShiftStream(feature_dim=4, **kwargs)

    def test_scale_bounds_are_narrowed_once_and_compared_after_narrowing(self, rng_key):
        """Exact-ratio inputs round straight to float32; equal narrowed bounds are accepted."""
        midpoint = Fraction(1) + Fraction(1, 2**24)
        stream = DynamicScaleShiftStream(feature_dim=4, min_scale=midpoint, max_scale=midpoint)
        assert stream._min_scale == 1.0
        assert stream._max_scale == 1.0
        assert type(stream._min_scale) is float
        max_finite = float(np.finfo(np.float32).max)
        stream = DynamicScaleShiftStream(feature_dim=4, min_scale=1.0, max_scale=max_finite)
        assert stream._max_scale == max_finite
        stream = DynamicScaleShiftStream(feature_dim=4, min_scale=0.5, max_scale=0.5)
        timestep, _ = stream.step(stream.init(rng_key), jnp.array(0))
        chex.assert_tree_all_finite(timestep.observation)

    def test_init_creates_valid_state(self, rng_key):
        """Stream init should create valid state with correct shapes."""
        stream = DynamicScaleShiftStream(feature_dim=10)
        state = stream.init(rng_key)

        assert state.key is not None
        chex.assert_shape(state.true_weights, (10,))
        chex.assert_shape(state.current_scales, (10,))
        assert state.step_count == 0

    def test_step_produces_valid_timestep(self, rng_key):
        """Step should produce valid observation and target."""
        stream = DynamicScaleShiftStream(feature_dim=10)
        state = stream.init(rng_key)

        timestep, new_state = stream.step(state, jnp.array(0))

        chex.assert_shape(timestep.observation, (10,))
        chex.assert_shape(timestep.target, (1,))
        chex.assert_tree_all_finite(timestep.observation)
        chex.assert_tree_all_finite(timestep.target)

    def test_feature_dim_property(self):
        """Feature dim property should return correct dimension."""
        stream = DynamicScaleShiftStream(feature_dim=20)
        assert stream.feature_dim == 20

    @pytest.mark.parametrize("field", ["current_scales", "true_weights"])
    @pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
    def test_initial_parameters_last_for_first_full_segment(
        self, rng_key, field, compiled
    ):
        """Initialized scales and weights each govern one complete interval."""
        change_interval = 3
        intervals = {
            "scale_change_interval": change_interval if field == "current_scales" else 97,
            "weight_change_interval": change_interval if field == "true_weights" else 97,
        }
        stream = DynamicScaleShiftStream(feature_dim=10, **intervals)
        state = stream.init(rng_key)
        initial_value = getattr(state, field)
        step = jax.jit(stream.step) if compiled else stream.step

        for i in range(change_interval):
            _, state = step(state, jnp.array(i, dtype=jnp.int32))
            chex.assert_trees_all_equal(getattr(state, field), initial_value)

        _, state = step(state, jnp.array(change_interval, dtype=jnp.int32))
        assert not bool(jnp.array_equal(getattr(state, field), initial_value))

    def test_scales_within_bounds(self, rng_key):
        """Scales should be within min_scale and max_scale."""
        min_scale, max_scale = 0.01, 100.0
        stream = DynamicScaleShiftStream(
            feature_dim=10,
            scale_change_interval=5,
            min_scale=min_scale,
            max_scale=max_scale,
        )
        state = stream.init(rng_key)

        # Run many steps with scale changes
        for i in range(50):
            _, state = stream.step(state, jnp.array(i))

        # All scales should be within bounds
        assert jnp.all(state.current_scales >= min_scale)
        assert jnp.all(state.current_scales <= max_scale)

    def test_deterministic_with_same_key(self, rng_key):
        """Same key should produce same sequence."""
        stream = DynamicScaleShiftStream(feature_dim=10)

        state1 = stream.init(rng_key)
        timestep1, _ = stream.step(state1, jnp.array(0))

        state2 = stream.init(rng_key)
        timestep2, _ = stream.step(state2, jnp.array(0))

        chex.assert_trees_all_close(timestep1.observation, timestep2.observation)
        chex.assert_trees_all_close(timestep1.target, timestep2.target)


class TestScaleDriftStream:
    """Tests for the ScaleDriftStream class."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            (
                {"min_log_scale": 4.0, "max_log_scale": -4.0},
                "min_log_scale must be <= max_log_scale",
            ),
            ({"min_log_scale": float("nan")}, "min_log_scale must be finite"),
            ({"max_log_scale": float("inf")}, "max_log_scale must be finite"),
            ({"min_log_scale": -1e100, "max_log_scale": 1e100}, "min_log_scale must be finite"),
            ({"max_log_scale": 1e39}, "max_log_scale must be finite"),
            ({"max_log_scale": Fraction(10**400)}, "max_log_scale must be finite"),
            ({"min_log_scale": True}, "min_log_scale must be finite"),
            ({"max_log_scale": False}, "max_log_scale must be finite"),
            ({"min_log_scale": -88.0}, "min_log_scale must be finite"),
            ({"max_log_scale": 89.0}, "max_log_scale must be finite"),
        ],
    )
    def test_rejects_reversed_or_nonfinite_log_scale_bounds(self, kwargs, message):
        """Reversed clip bounds pin every scale to max_log_scale: a stationary stream."""
        with pytest.raises(ValueError, match=message):
            ScaleDriftStream(feature_dim=4, **kwargs)

    def test_equal_log_scale_bounds_are_accepted(self, rng_key):
        stream = ScaleDriftStream(feature_dim=4, min_log_scale=0.0, max_log_scale=0.0)
        timestep, _ = stream.step(stream.init(rng_key), jnp.array(0))
        chex.assert_tree_all_finite(timestep.observation)

    def test_log_scale_bounds_are_narrowed_once_before_comparison(self):
        """A Fraction midpoint rounds once (ties-to-even) to the float32 clip bound."""
        midpoint = Fraction(1) + Fraction(1, 2**24)
        stream = ScaleDriftStream(feature_dim=4, min_log_scale=midpoint, max_log_scale=1.0)
        assert stream._min_log_scale == 1.0
        assert stream._max_log_scale == 1.0
        assert type(stream._min_log_scale) is float

    def test_log_scale_bounds_have_positive_finite_float32_exponentials(self):
        """Accepted clip endpoints cannot collapse or overflow the scale factor."""
        min_safe = float(
            np.nextafter(
                np.float32(np.log(np.finfo(np.float32).tiny)), np.float32(np.inf)
            )
        )
        max_safe = float(
            np.nextafter(
                np.float32(np.log(np.finfo(np.float32).max)), np.float32(-np.inf)
            )
        )
        stream = ScaleDriftStream(
            feature_dim=4, min_log_scale=min_safe, max_log_scale=max_safe
        )
        endpoint_scales = jnp.exp(
            jnp.asarray([stream._min_log_scale, stream._max_log_scale], dtype=jnp.float32)
        )
        assert bool(jnp.all(endpoint_scales > 0.0))
        assert bool(jnp.all(jnp.isfinite(endpoint_scales)))

        with pytest.raises(ValueError, match="min_log_scale"):
            ScaleDriftStream(
                feature_dim=4,
                min_log_scale=float(np.nextafter(np.float32(min_safe), np.float32(-np.inf))),
            )
        with pytest.raises(ValueError, match="max_log_scale"):
            ScaleDriftStream(
                feature_dim=4,
                max_log_scale=float(np.nextafter(np.float32(max_safe), np.float32(np.inf))),
            )

    def test_clip_bounds_stay_finite_at_execution(self, rng_key):
        """The stored bounds are exactly what jnp.clip receives, so the walk is bounded."""
        stream = ScaleDriftStream(
            feature_dim=4, scale_drift_rate=10.0, min_log_scale=-1.0, max_log_scale=1.0
        )
        state = stream.init(rng_key)
        for step in range(50):
            _, state = stream.step(state, jnp.array(step))
        assert bool(jnp.all(jnp.isfinite(state.log_scales)))
        assert bool(jnp.all(state.log_scales >= -1.0))
        assert bool(jnp.all(state.log_scales <= 1.0))

    def test_init_creates_valid_state(self, rng_key):
        """Stream init should create valid state with correct shapes."""
        stream = ScaleDriftStream(feature_dim=10)
        state = stream.init(rng_key)

        assert state.key is not None
        chex.assert_shape(state.true_weights, (10,))
        chex.assert_shape(state.log_scales, (10,))
        assert state.step_count == 0
        # Initial log_scales should be 0 (scale = 1)
        chex.assert_trees_all_close(state.log_scales, jnp.zeros(10))

    def test_step_produces_valid_timestep(self, rng_key):
        """Step should produce valid observation and target."""
        stream = ScaleDriftStream(feature_dim=10)
        state = stream.init(rng_key)

        timestep, new_state = stream.step(state, jnp.array(0))

        chex.assert_shape(timestep.observation, (10,))
        chex.assert_shape(timestep.target, (1,))
        chex.assert_tree_all_finite(timestep.observation)
        chex.assert_tree_all_finite(timestep.target)

    def test_feature_dim_property(self):
        """Feature dim property should return correct dimension."""
        stream = ScaleDriftStream(feature_dim=20)
        assert stream.feature_dim == 20

    def test_scales_drift_over_time(self, rng_key):
        """Log-scales should change from step to step."""
        stream = ScaleDriftStream(feature_dim=10, scale_drift_rate=0.1)  # High drift
        state = stream.init(rng_key)

        initial_log_scales = state.log_scales.copy()

        for i in range(100):
            _, state = stream.step(state, jnp.array(i))

        # Log-scales should have changed
        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(initial_log_scales, state.log_scales)

    def test_weights_drift_over_time(self, rng_key):
        """Weights should change from step to step."""
        stream = ScaleDriftStream(feature_dim=10, weight_drift_rate=0.1)  # High drift
        state = stream.init(rng_key)

        initial_weights = state.true_weights.copy()

        for i in range(100):
            _, state = stream.step(state, jnp.array(i))

        # Weights should have changed
        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(initial_weights, state.true_weights)

    def test_log_scales_bounded(self, rng_key):
        """Log-scales should stay within bounds."""
        min_log, max_log = -2.0, 2.0
        stream = ScaleDriftStream(
            feature_dim=10,
            scale_drift_rate=0.5,  # High drift to test bounds
            min_log_scale=min_log,
            max_log_scale=max_log,
        )
        state = stream.init(rng_key)

        # Run many steps
        for i in range(500):
            _, state = stream.step(state, jnp.array(i))

        # Log-scales should be within bounds
        assert jnp.all(state.log_scales >= min_log)
        assert jnp.all(state.log_scales <= max_log)

    def test_deterministic_with_same_key(self, rng_key):
        """Same key should produce same sequence."""
        stream = ScaleDriftStream(feature_dim=10)

        state1 = stream.init(rng_key)
        timestep1, _ = stream.step(state1, jnp.array(0))

        state2 = stream.init(rng_key)
        timestep2, _ = stream.step(state2, jnp.array(0))

        chex.assert_trees_all_close(timestep1.observation, timestep2.observation)
        chex.assert_trees_all_close(timestep1.target, timestep2.target)

    def test_generates_valid_timesteps(self, rng_key):
        """Should generate valid TimeStep instances over many steps."""
        stream = ScaleDriftStream(feature_dim=5)
        state = stream.init(rng_key)

        for i in range(100):
            timestep, state = stream.step(state, jnp.array(i))
            assert isinstance(timestep, TimeStep)
            chex.assert_tree_all_finite(timestep.observation)
            chex.assert_tree_all_finite(timestep.target)


class TestScheduleModuliRejectInvalid:
    """Schedule divisors must be positive JAX-int32 integers before arithmetic."""

    @pytest.mark.parametrize("period", _INVALID_SCHEDULE_MODULI)
    def test_periodic_change_period_must_be_positive_int(self, period):
        with pytest.raises(
            ValueError,
            match=rf"period must be a positive integer in \[1, {_INT32_MAX}\]",
        ):
            PeriodicChangeStream(feature_dim=3, period=period)

    @pytest.mark.parametrize("cycle_length", _INVALID_SCHEDULE_MODULI)
    def test_cyclic_cycle_length_must_be_positive_int(self, cycle_length):
        with pytest.raises(
            ValueError,
            match=rf"cycle_length must be a positive integer in \[1, {_INT32_MAX}\]",
        ):
            CyclicStream(feature_dim=3, cycle_length=cycle_length)

    @pytest.mark.parametrize("num_configurations", _INVALID_SCHEDULE_MODULI)
    def test_cyclic_num_configurations_must_be_positive_int(self, num_configurations):
        with pytest.raises(
            ValueError,
            match=rf"num_configurations must be a positive integer in \[1, {_INT32_MAX}\]",
        ):
            CyclicStream(feature_dim=3, num_configurations=num_configurations)

    @pytest.mark.parametrize("change_interval", _INVALID_SCHEDULE_MODULI)
    def test_abrupt_change_interval_must_be_positive_int(self, change_interval):
        with pytest.raises(
            ValueError,
            match=rf"change_interval must be a positive integer in \[1, {_INT32_MAX}\]",
        ):
            AbruptChangeStream(feature_dim=3, change_interval=change_interval)

    @pytest.mark.parametrize("change_interval", _INVALID_SCHEDULE_MODULI)
    def test_sutton_change_interval_must_be_positive_int(self, change_interval):
        with pytest.raises(
            ValueError,
            match=rf"change_interval must be a positive integer in \[1, {_INT32_MAX}\]",
        ):
            SuttonExperiment1Stream(change_interval=change_interval)

    @pytest.mark.parametrize("name", ["scale_change_interval", "weight_change_interval"])
    @pytest.mark.parametrize("value", _INVALID_SCHEDULE_MODULI)
    def test_dynamic_scale_shift_intervals_must_be_positive_ints(self, name, value):
        with pytest.raises(
            ValueError,
            match=rf"{name} must be a positive integer in \[1, {_INT32_MAX}\]",
        ):
            DynamicScaleShiftStream(feature_dim=3, **{name: value})

    @pytest.mark.parametrize(
        "stream",
        [
            AbruptChangeStream(feature_dim=3, change_interval=_INT32_MAX),
            SuttonExperiment1Stream(change_interval=_INT32_MAX),
            CyclicStream(feature_dim=3, cycle_length=_INT32_MAX),
            PeriodicChangeStream(feature_dim=3, period=_INT32_MAX),
            DynamicScaleShiftStream(feature_dim=3, scale_change_interval=_INT32_MAX),
            DynamicScaleShiftStream(feature_dim=3, weight_change_interval=_INT32_MAX),
        ],
        ids=["abrupt", "sutton", "cyclic", "periodic", "dynamic-scale", "dynamic-weight"],
    )
    def test_int32_max_schedule_runs_first_eager_and_jit_step(self, stream, rng_key):
        state = stream.init(rng_key)
        eager_timestep, eager_state = stream.step(state, jnp.array(0, dtype=jnp.int32))
        jit_timestep, jit_state = jax.jit(stream.step)(state, jnp.array(0, dtype=jnp.int32))

        chex.assert_tree_all_finite(eager_timestep)
        chex.assert_tree_all_finite(jit_timestep)
        chex.assert_trees_all_close(eager_timestep, jit_timestep)
        jax.block_until_ready((eager_state, jit_state))

    def test_cyclic_int32_max_num_configurations_constructs(self):
        """The representable boundary is valid even when allocation would be impractical."""
        stream = CyclicStream(feature_dim=3, num_configurations=_INT32_MAX)
        assert stream.feature_dim == 3


def test_scaled_stream_wrapper_rejects_non_vector_feature_scales():
    """Per-feature scales must stay one-dimensional to preserve observation shape."""
    with pytest.raises(ValueError, match=r"feature_scales shape \(\(3, 1\)\)"):
        ScaledStreamWrapper(RandomWalkStream(feature_dim=3), jnp.ones((3, 1)))
