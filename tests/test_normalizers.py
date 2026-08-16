"""Tests for online feature normalization."""

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework import (
    EMANormalizer,
    EMANormalizerState,
    Normalizer,
    StreamingBatchNormalizer,
    StreamingBatchNormalizerState,
    WelfordNormalizer,
    WelfordNormalizerState,
    normalizer_from_config,
)

_NORMALIZERS = (EMANormalizer, WelfordNormalizer, StreamingBatchNormalizer)


@pytest.mark.parametrize("normalizer_type", _NORMALIZERS)
@pytest.mark.parametrize(
    "feature_dim",
    [
        4,
        np.int8(4),
        np.int32(4),
        np.int64(4),
        np.longlong(4),
        np.uint64(4),
        np.ulonglong(4),
    ],
)
def test_normalizer_init_accepts_supported_exact_integer_types(
    normalizer_type,
    feature_dim,
) -> None:
    state = normalizer_type().init(feature_dim)
    chex.assert_shape(state.mean, (4,))
    chex.assert_shape(state.var, (4,))


@pytest.mark.parametrize("normalizer_type", _NORMALIZERS)
@pytest.mark.parametrize(
    "feature_dim",
    [True, np.bool_(True), 1.5, np.float32(4), 0, -1, 2**31, "4", [4]],
)
def test_normalizer_init_rejects_invalid_feature_dimensions(
    normalizer_type,
    feature_dim,
) -> None:
    with pytest.raises(ValueError, match="feature_dim"):
        normalizer_type().init(feature_dim)


@pytest.mark.parametrize("normalizer_type", _NORMALIZERS)
def test_normalizer_init_rejects_integer_subclasses(normalizer_type) -> None:
    class LyingInt(int):
        def __int__(self) -> int:
            return 4

    with pytest.raises(ValueError, match="feature_dim"):
        normalizer_type().init(LyingInt(-1))


class TestEMANormalizer:
    """Tests for the EMANormalizer class."""

    def test_init_creates_correct_state(self, feature_dim):
        """EMANormalizer init should create state with zero mean, unit variance."""
        normalizer = EMANormalizer()
        state = normalizer.init(feature_dim)

        assert isinstance(state, EMANormalizerState)
        chex.assert_shape(state.mean, (feature_dim,))
        chex.assert_shape(state.var, (feature_dim,))
        chex.assert_trees_all_close(state.mean, jnp.zeros(feature_dim))
        chex.assert_trees_all_close(state.var, jnp.ones(feature_dim))
        assert state.sample_count == 0.0

    def test_normalize_updates_statistics(self, sample_observation):
        """Normalizing should update mean and variance estimates."""
        normalizer = EMANormalizer()
        state = normalizer.init(len(sample_observation))

        normalized, new_state = normalizer.normalize(state, sample_observation)

        # Count should increase
        assert new_state.sample_count == 1.0

        # Mean should have moved toward the observation
        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(new_state.mean, state.mean)

    def test_normalize_returns_finite_values(self, sample_observation):
        """Normalized output should always be finite."""
        normalizer = EMANormalizer()
        state = normalizer.init(len(sample_observation))

        normalized, _ = normalizer.normalize(state, sample_observation)

        chex.assert_tree_all_finite(normalized)

    def test_normalize_only_does_not_update_state(self, sample_observation):
        """normalize_only should not modify the state."""
        normalizer = EMANormalizer()
        state = normalizer.init(len(sample_observation))

        # First update state
        _, state = normalizer.normalize(state, sample_observation)
        original_count = state.sample_count

        # normalize_only should not change count
        _ = normalizer.normalize_only(state, sample_observation)

        assert state.sample_count == original_count

    def test_update_only_does_not_return_normalized(self, sample_observation):
        """update_only should only update state, returning new state."""
        normalizer = EMANormalizer()
        state = normalizer.init(len(sample_observation))

        new_state = normalizer.update_only(state, sample_observation)

        assert isinstance(new_state, EMANormalizerState)
        assert new_state.sample_count == 1.0

    def test_repeated_updates_converge(self, sample_observation):
        """Mean and variance should converge with repeated identical inputs."""
        normalizer = EMANormalizer(decay=0.9)
        state = normalizer.init(len(sample_observation))

        # Repeatedly normalize the same observation
        for _ in range(32):
            _, state = normalizer.normalize(state, sample_observation)

        # Mean should be close to the observation
        # (not exact due to decay and numerical issues)
        chex.assert_trees_all_close(state.mean, sample_observation, atol=0.5)

    @pytest.mark.parametrize("decay", [0.9, 1.0])
    def test_warmup_uses_initial_moment_pseudo_sample(self, decay: float) -> None:
        """Warmup treats the initialized moments as one explicit pseudo-sample."""
        normalizer = EMANormalizer(decay=decay)
        state = normalizer.init(1)

        _, state = normalizer.normalize(
            state,
            jnp.asarray([5.0], dtype=jnp.float32),
        )
        chex.assert_trees_all_equal(state.mean, jnp.asarray([2.5], dtype=jnp.float32))
        chex.assert_trees_all_equal(state.var, jnp.asarray([6.75], dtype=jnp.float32))

        _, state = normalizer.normalize(
            state,
            jnp.asarray([7.0], dtype=jnp.float32),
        )
        chex.assert_trees_all_equal(state.mean, jnp.asarray([4.0], dtype=jnp.float32))
        chex.assert_trees_all_close(
            state.var,
            jnp.asarray([9.0], dtype=jnp.float32),
            atol=1e-6,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        ("decay", "expected_mean"),
        [(0.0, 5.0), (0.1, 4.5), (0.49, 2.55)],
    )
    def test_warmup_respects_decay_below_one_half(
        self,
        decay: float,
        expected_mean: float,
    ) -> None:
        """The first effective decay is the smaller of the target and one half."""
        normalizer = EMANormalizer(decay=decay)
        state = normalizer.init(1)

        _, state = normalizer.normalize(
            state,
            jnp.asarray([5.0], dtype=jnp.float32),
        )

        chex.assert_trees_all_close(
            state.mean,
            jnp.asarray([expected_mean], dtype=jnp.float32),
            atol=1e-6,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        ("decay", "expected"),
        [
            (0.9, "ema-zero-mean-unit-variance-prior-bounded-recency"),
            (
                1.0,
                "ema-zero-mean-unit-variance-prior-"
                "cumulative-float32-fail-stop-at-2p24",
            ),
        ],
    )
    def test_config_names_prior_regularized_semantics(
        self,
        decay: float,
        expected: str,
    ) -> None:
        """Serialized semantics distinguish EMA's prior from other estimators."""
        assert EMANormalizer(decay=decay).to_config()["estimator_semantics"] == expected

    @pytest.mark.parametrize(
        ("decay", "legacy"),
        [
            (0.9, "bounded-recency"),
            (1.0, "cumulative-float32-fail-stop-at-2p24"),
        ],
    )
    def test_legacy_generic_semantics_load_and_canonicalize(
        self,
        decay: float,
        legacy: str,
    ) -> None:
        """Existing receipts still load, then serialize the explicit contract."""
        config = EMANormalizer(decay=decay).to_config()
        config["estimator_semantics"] = legacy

        restored = normalizer_from_config(config)

        assert isinstance(restored, EMANormalizer)
        assert restored.to_config()["estimator_semantics"].startswith(
            "ema-zero-mean-unit-variance-prior-"
        )

    def test_generic_semantics_alias_must_match_decay_family(self) -> None:
        """A legacy alias cannot disguise a bounded/cumulative mismatch."""
        config = EMANormalizer(decay=1.0).to_config()
        config["estimator_semantics"] = "bounded-recency"
        with pytest.raises(ValueError, match="estimator_semantics"):
            normalizer_from_config(config)

    @pytest.mark.parametrize("invalid", [[], {}])
    def test_estimator_semantics_requires_a_string(self, invalid: object) -> None:
        """Malformed JSON values fail with the loader's stable config error."""
        config = EMANormalizer(decay=0.9).to_config()
        config["estimator_semantics"] = invalid

        with pytest.raises(ValueError, match="estimator_semantics"):
            normalizer_from_config(config)

    def test_zero_decay_recovers_from_nonfinite_unused_moments(self) -> None:
        """A zero-decay update does not consume either previous moment."""
        normalizer = EMANormalizer(decay=0.0)
        obs = jnp.array([1.0, -2.0], dtype=jnp.float32)
        state = normalizer.normalize(normalizer.init(2), obs)[1]
        state = state.replace(
            mean=jnp.full(2, jnp.inf, dtype=jnp.float32),
            var=jnp.full(2, jnp.inf, dtype=jnp.float32),
        )
        result = normalizer.normalize_with_diagnostics(state, obs)
        assert isinstance(result.state, EMANormalizerState)
        assert bool(result.update_applied)
        chex.assert_trees_all_close(result.state.mean, obs)
        chex.assert_trees_all_close(
            result.state.var,
            jnp.full_like(obs, normalizer._epsilon),
        )
        chex.assert_trees_all_close(result.normalized, jnp.zeros_like(obs))

    def test_zero_decay_preserves_finite_update_arithmetic(self) -> None:
        """Finite zero-decay updates retain the established rounding path."""
        normalizer = EMANormalizer(decay=0.0)
        state = normalizer.normalize(
            normalizer.init(1),
            jnp.array([0.0], dtype=jnp.float32),
        )[1]
        state = state.replace(
            mean=jnp.array([1.0e20], dtype=jnp.float32),
            var=jnp.array([7.0], dtype=jnp.float32),
        )
        obs = jnp.array([1.0], dtype=jnp.float32)
        delta = obs - state.mean
        expected_mean = state.mean + delta
        delta2 = obs - expected_mean
        expected_var = jnp.maximum(
            jnp.asarray(0.0, dtype=jnp.float32) * state.var + delta * delta2,
            normalizer._epsilon,
        )

        result = normalizer.normalize_with_diagnostics(state, obs)
        assert isinstance(result.state, EMANormalizerState)
        assert bool(result.update_applied)
        chex.assert_trees_all_equal(result.state.mean, expected_mean)
        chex.assert_trees_all_equal(result.state.var, expected_var)
        assert not bool(jnp.array_equal(result.state.mean, obs))

    def test_zero_decay_config_does_not_relax_nonzero_persisted_decay(self) -> None:
        """A host/state decay mismatch remains fail-closed with poisoned moments."""
        normalizer = EMANormalizer(decay=0.0)
        state = normalizer.init(1).replace(
            mean=jnp.array([jnp.inf], dtype=jnp.float32),
            var=jnp.array([jnp.inf], dtype=jnp.float32),
            decay=jnp.asarray(0.5, dtype=jnp.float32),
        )

        result = normalizer.normalize_with_diagnostics(
            state,
            jnp.array([2.0], dtype=jnp.float32),
        )
        assert isinstance(result.state, EMANormalizerState)
        assert not bool(result.counter_valid)
        assert not bool(result.update_applied)
        assert bool(jnp.all(jnp.isfinite(result.normalized)))
        chex.assert_trees_all_equal(result.normalized, jnp.zeros(1, dtype=jnp.float32))
        chex.assert_trees_all_equal(result.state.sample_count_words, state.sample_count_words)


class TestWelfordNormalizer:
    """Tests for the WelfordNormalizer class."""

    def test_init_creates_correct_state(self, feature_dim):
        """WelfordNormalizer init should create state with zero mean, unit variance, zero p."""
        normalizer = WelfordNormalizer()
        state = normalizer.init(feature_dim)

        assert isinstance(state, WelfordNormalizerState)
        chex.assert_shape(state.mean, (feature_dim,))
        chex.assert_shape(state.var, (feature_dim,))
        chex.assert_shape(state.p, (feature_dim,))
        chex.assert_trees_all_close(state.mean, jnp.zeros(feature_dim))
        chex.assert_trees_all_close(state.var, jnp.ones(feature_dim))
        chex.assert_trees_all_close(state.p, jnp.zeros(feature_dim))
        assert state.sample_count == 0.0

    def test_var_is_one_when_count_less_than_two(self):
        """Variance should be 1.0 when fewer than 2 samples have been seen."""
        normalizer = WelfordNormalizer()
        state = normalizer.init(5)

        obs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, new_state = normalizer.normalize(state, obs)

        assert new_state.sample_count == 1.0
        chex.assert_trees_all_close(new_state.var, jnp.ones(5))

    def test_converges_to_true_mean_and_var(self):
        """After many samples, mean and var should match sample statistics."""
        normalizer = WelfordNormalizer()
        feature_dim = 5
        state = normalizer.init(feature_dim)

        true_mean = jnp.array([1.0, -2.0, 3.0, 0.5, -1.0])
        true_std = jnp.array([0.5, 1.0, 2.0, 0.3, 1.5])
        standardized = jnp.linspace(-1.5, 1.5, 32 * feature_dim).reshape(
            32, feature_dim
        )
        observations = true_mean + true_std * standardized
        for obs in observations:
            _, state = normalizer.normalize(state, obs)

        expected_mean = jnp.mean(observations, axis=0)
        expected_var = jnp.var(observations, axis=0, ddof=1)

        chex.assert_trees_all_close(state.mean, expected_mean, atol=1e-5)
        chex.assert_trees_all_close(state.var, expected_var, atol=1e-5)

    def test_normalize_returns_finite_values(self, sample_observation):
        """Normalized output should always be finite."""
        normalizer = WelfordNormalizer()
        state = normalizer.init(len(sample_observation))

        normalized, _ = normalizer.normalize(state, sample_observation)
        chex.assert_tree_all_finite(normalized)

    def test_normalize_only_does_not_update_state(self, sample_observation):
        """normalize_only should not modify the state."""
        normalizer = WelfordNormalizer()
        state = normalizer.init(len(sample_observation))

        _, state = normalizer.normalize(state, sample_observation)
        original_count = state.sample_count

        _ = normalizer.normalize_only(state, sample_observation)
        assert state.sample_count == original_count

    def test_update_only_returns_updated_state(self, sample_observation):
        """update_only should return updated state."""
        normalizer = WelfordNormalizer()
        state = normalizer.init(len(sample_observation))

        new_state = normalizer.update_only(state, sample_observation)

        assert isinstance(new_state, WelfordNormalizerState)
        assert new_state.sample_count == 1.0

    def test_p_field_has_correct_shape(self, feature_dim):
        """The p (M2) field should have shape (feature_dim,)."""
        normalizer = WelfordNormalizer()
        state = normalizer.init(feature_dim)
        chex.assert_shape(state.p, (feature_dim,))


class TestStreamingBatchNormalizer:
    """Tests for the StreamingBatchNormalizer class."""

    def test_init_creates_correct_state(self, feature_dim):
        """StreamingBatchNormalizer init should create running moments."""
        normalizer = StreamingBatchNormalizer(momentum=0.9)
        state = normalizer.init(feature_dim)

        assert isinstance(state, StreamingBatchNormalizerState)
        chex.assert_shape(state.mean, (feature_dim,))
        chex.assert_shape(state.var, (feature_dim,))
        chex.assert_trees_all_close(state.mean, jnp.zeros(feature_dim))
        chex.assert_trees_all_close(state.var, jnp.ones(feature_dim))
        assert state.sample_count == 0.0
        assert state.momentum == 0.9

    def test_normalize_updates_statistics(self, sample_observation):
        """Normalizing should update BatchNorm-style running moments."""
        normalizer = StreamingBatchNormalizer(momentum=0.5)
        state = normalizer.init(len(sample_observation))

        normalized, new_state = normalizer.normalize(state, sample_observation)

        chex.assert_tree_all_finite(normalized)
        assert new_state.sample_count == 1.0
        chex.assert_trees_all_close(new_state.mean, sample_observation)
        chex.assert_trees_all_close(new_state.var, jnp.ones_like(sample_observation))

    def test_roundtrip_config(self):
        """StreamingBatchNormalizer should serialize through the dispatcher."""
        normalizer = StreamingBatchNormalizer(momentum=0.75, epsilon=1e-4)
        config = normalizer.to_config()

        assert config["type"] == "StreamingBatchNormalizer"
        restored = normalizer_from_config(config)
        assert isinstance(restored, StreamingBatchNormalizer)
        assert restored._momentum == 0.75
        assert restored._epsilon == 1e-4

    def test_zero_momentum_preserves_finite_variance_update(self) -> None:
        """The previous mean still defines variance when momentum is zero."""
        normalizer = StreamingBatchNormalizer(momentum=0.0)
        state = normalizer.normalize(
            normalizer.init(2),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
        )[1]
        state = state.replace(
            mean=jnp.array([4.0, -5.0], dtype=jnp.float32),
            var=jnp.array([2.0, 3.0], dtype=jnp.float32),
        )
        obs = jnp.array([1.0, -2.0], dtype=jnp.float32)
        expected_mean = state.momentum * state.mean + (1.0 - state.momentum) * obs
        centered = obs - state.mean
        expected_var = jnp.maximum(
            state.momentum * state.var
            + (1.0 - state.momentum) * (centered * centered),
            normalizer._epsilon,
        )

        result = normalizer.normalize_with_diagnostics(state, obs)
        assert isinstance(result.state, StreamingBatchNormalizerState)
        assert bool(result.update_applied)
        chex.assert_trees_all_equal(result.state.mean, expected_mean)
        chex.assert_trees_all_equal(result.state.var, expected_var)
        assert bool(jnp.all(result.state.var > normalizer._epsilon))

    def test_zero_momentum_recovers_from_nonfinite_unused_variance(self) -> None:
        """Only the zero-scaled previous variance may be nonfinite."""
        normalizer = StreamingBatchNormalizer(momentum=0.0)
        state = normalizer.normalize(
            normalizer.init(2),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
        )[1]
        state = state.replace(
            mean=jnp.array([4.0, -5.0], dtype=jnp.float32),
            var=jnp.full(2, jnp.inf, dtype=jnp.float32),
        )
        obs = jnp.array([1.0, -2.0], dtype=jnp.float32)

        result = normalizer.normalize_with_diagnostics(state, obs)
        assert isinstance(result.state, StreamingBatchNormalizerState)
        assert bool(result.update_applied)
        chex.assert_trees_all_equal(result.state.mean, obs)
        chex.assert_trees_all_equal(result.state.var, (obs - state.mean) ** 2)
        chex.assert_tree_all_finite(result.normalized)

    def test_zero_momentum_rejects_nonfinite_consumed_mean(self) -> None:
        """The previous mean cannot be ignored because variance consumes it."""
        normalizer = StreamingBatchNormalizer(momentum=0.0)
        state = normalizer.normalize(
            normalizer.init(1),
            jnp.array([0.0], dtype=jnp.float32),
        )[1]
        state = state.replace(
            mean=jnp.array([jnp.inf], dtype=jnp.float32),
            var=jnp.array([jnp.inf], dtype=jnp.float32),
        )

        result = normalizer.normalize_with_diagnostics(
            state,
            jnp.array([2.0], dtype=jnp.float32),
        )
        assert isinstance(result.state, StreamingBatchNormalizerState)
        assert bool(result.counter_valid)
        assert not bool(result.update_applied)
        assert bool(jnp.all(jnp.isfinite(result.normalized)))
        chex.assert_trees_all_equal(result.normalized, jnp.zeros(1, dtype=jnp.float32))
        chex.assert_trees_all_equal(result.state.sample_count_words, state.sample_count_words)

    def test_zero_momentum_config_does_not_relax_nonzero_persisted_value(self) -> None:
        """A host/state momentum mismatch remains fail-closed with poisoned moments."""
        normalizer = StreamingBatchNormalizer(momentum=0.0)
        state = normalizer.init(1).replace(
            mean=jnp.array([jnp.inf], dtype=jnp.float32),
            var=jnp.array([jnp.inf], dtype=jnp.float32),
            momentum=jnp.asarray(0.5, dtype=jnp.float32),
        )

        result = normalizer.normalize_with_diagnostics(
            state,
            jnp.array([2.0], dtype=jnp.float32),
        )
        assert isinstance(result.state, StreamingBatchNormalizerState)
        assert not bool(result.counter_valid)
        assert not bool(result.update_applied)
        assert bool(jnp.all(jnp.isfinite(result.normalized)))
        chex.assert_trees_all_equal(result.normalized, jnp.zeros(1, dtype=jnp.float32))
        chex.assert_trees_all_equal(result.state.sample_count_words, state.sample_count_words)


class TestNormalizerABC:
    """Tests that both normalizers satisfy the ABC contract."""

    @pytest.mark.parametrize(
        "normalizer_cls",
        [EMANormalizer, WelfordNormalizer, StreamingBatchNormalizer],
    )
    def test_is_normalizer_subclass(self, normalizer_cls):
        """All normalizers should be subclasses of Normalizer."""
        assert issubclass(normalizer_cls, Normalizer)

    @pytest.mark.parametrize(
        "normalizer",
        [EMANormalizer(), WelfordNormalizer(), StreamingBatchNormalizer()],
        ids=["EMA", "Welford", "StreamingBatch"],
    )
    def test_init_returns_state_with_required_fields(self, normalizer, feature_dim):
        """All normalizer states should have mean, var, and sample_count."""
        state = normalizer.init(feature_dim)

        assert hasattr(state, "mean")
        assert hasattr(state, "var")
        assert hasattr(state, "sample_count")
        chex.assert_shape(state.mean, (feature_dim,))
        chex.assert_shape(state.var, (feature_dim,))

    @pytest.mark.parametrize(
        "normalizer",
        [EMANormalizer(), WelfordNormalizer(), StreamingBatchNormalizer()],
        ids=["EMA", "Welford", "StreamingBatch"],
    )
    def test_normalize_returns_tuple(self, normalizer, sample_observation):
        """normalize should return (array, state) tuple."""
        state = normalizer.init(len(sample_observation))
        result = normalizer.normalize(state, sample_observation)

        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.parametrize(
        "normalizer",
        [EMANormalizer(), WelfordNormalizer(), StreamingBatchNormalizer()],
        ids=["EMA", "Welford", "StreamingBatch"],
    )
    def test_normalize_only_returns_array(self, normalizer, sample_observation):
        """normalize_only should return just an array."""
        state = normalizer.init(len(sample_observation))
        _, state = normalizer.normalize(state, sample_observation)
        result = normalizer.normalize_only(state, sample_observation)
        chex.assert_tree_all_finite(result)

    @pytest.mark.parametrize(
        "normalizer",
        [EMANormalizer(), WelfordNormalizer(), StreamingBatchNormalizer()],
        ids=["EMA", "Welford", "StreamingBatch"],
    )
    def test_update_only_increments_count(self, normalizer, sample_observation):
        """update_only should increment sample_count."""
        state = normalizer.init(len(sample_observation))
        new_state = normalizer.update_only(state, sample_observation)
        assert new_state.sample_count == 1.0

    @pytest.mark.parametrize(
        "normalizer",
        [EMANormalizer(), WelfordNormalizer(), StreamingBatchNormalizer()],
        ids=["EMA", "Welford", "StreamingBatch"],
    )
    def test_inf_observation_holds_finite_statistics(self, normalizer) -> None:
        """Inf observation minus an inf mean is inf-inf = NaN.

        Fail-closed: skip the statistics update and keep the previous finite
        mean/variance so later finite samples are not poisoned.
        """
        state = normalizer.init(2)
        inf_obs = jnp.array([jnp.inf, 1.0], dtype=jnp.float32)
        rejected = normalizer.normalize_with_diagnostics(state, inf_obs)
        out, held = rejected.normalized, rejected.state
        assert not bool(rejected.update_applied)
        assert bool(jnp.all(jnp.isfinite(out)))
        assert bool(jnp.all(jnp.isfinite(held.mean)))
        assert bool(jnp.all(jnp.isfinite(held.var)))
        chex.assert_trees_all_close(held.mean, state.mean)
        chex.assert_trees_all_close(held.var, state.var)
        assert int(held.sample_count) == int(state.sample_count)

        finite_obs = jnp.array([0.0, 2.0], dtype=jnp.float32)
        accepted = normalizer.normalize_with_diagnostics(held, finite_obs)
        out2, recovered = accepted.normalized, accepted.state
        assert bool(accepted.update_applied)
        assert bool(jnp.all(jnp.isfinite(out2)))
        assert bool(jnp.all(jnp.isfinite(recovered.mean)))
        assert bool(jnp.all(jnp.isfinite(recovered.var)))
        assert int(recovered.sample_count) == 1

    @pytest.mark.parametrize(
        "normalizer",
        [EMANormalizer(), WelfordNormalizer(), StreamingBatchNormalizer()],
        ids=["EMA", "Welford", "StreamingBatch"],
    )
    def test_finite_overflow_is_not_rewritten_to_zero(self, normalizer) -> None:
        """Finite arithmetic overflow remains visible to downstream gates."""
        state = normalizer.init(1)
        max_f32 = jnp.finfo(jnp.float32).max
        state = state.replace(mean=jnp.array([-max_f32], dtype=jnp.float32))
        out = normalizer.normalize_only(state, jnp.array([max_f32], dtype=jnp.float32))
        assert not bool(jnp.isfinite(out[0]))
        assert float(out[0]) != 0.0


class _FloatSpoof:
    """Not a Real at all, but reports ``float`` through ``__class__``."""

    def __init__(self, value: float) -> None:
        self._value = value

    @property
    def __class__(self) -> type[float]:  # type: ignore[override]
        return float

    def __float__(self) -> float:
        return self._value


class _RaisingFloatSpoof:
    """A ``__class__`` spoof whose ``__float__`` hook raises when trusted."""

    @property
    def __class__(self) -> type[float]:  # type: ignore[override]
        return float

    def __float__(self) -> float:
        raise RuntimeError("untrusted __float__ hook executed")


class TestNormalizerScalarSpoofRejection:
    """``epsilon``/``decay``/``momentum`` must reject ``__class__``-spoofed reals.

    ``isinstance(value, Real)`` consults the overridable ``__class__``
    attribute, so an object that only lies about its class via a
    ``__class__`` property previously passed validation even though its
    actual type is not a registered ``Real``.
    """

    @pytest.mark.parametrize(
        ("ctor", "field"),
        [
            (EMANormalizer, "epsilon"),
            (EMANormalizer, "decay"),
            (StreamingBatchNormalizer, "epsilon"),
            (StreamingBatchNormalizer, "momentum"),
            (WelfordNormalizer, "epsilon"),
        ],
        ids=[
            "ema-epsilon",
            "ema-decay",
            "streaming-epsilon",
            "streaming-momentum",
            "welford-epsilon",
        ],
    )
    def test_rejects_class_spoofed_real_field(self, ctor, field) -> None:
        """A well-behaved ``__class__``-spoofed real must not be accepted."""
        in_domain = {"epsilon": 1e-6, "decay": 0.5, "momentum": 0.5}[field]
        with pytest.raises(TypeError, match="must be a finite real number"):
            ctor(**{field: _FloatSpoof(in_domain)})

    @pytest.mark.parametrize(
        ("ctor", "field"),
        [
            (EMANormalizer, "epsilon"),
            (EMANormalizer, "decay"),
            (StreamingBatchNormalizer, "epsilon"),
            (StreamingBatchNormalizer, "momentum"),
            (WelfordNormalizer, "epsilon"),
        ],
        ids=[
            "ema-epsilon",
            "ema-decay",
            "streaming-epsilon",
            "streaming-momentum",
            "welford-epsilon",
        ],
    )
    def test_raising_spoofed_field_stays_a_type_error(self, ctor, field) -> None:
        """A spoof with a raising ``__float__`` must not leak its raw exception."""
        with pytest.raises(TypeError, match="must be a finite real number"):
            ctor(**{field: _RaisingFloatSpoof()})

    def test_rejects_bool_epsilon(self) -> None:
        with pytest.raises(TypeError, match="epsilon must be a finite real number"):
            EMANormalizer(epsilon=True)

    def test_rejects_bool_decay(self) -> None:
        with pytest.raises(TypeError, match="decay must be a finite real number"):
            EMANormalizer(decay=False)

    def test_rejects_bool_momentum(self) -> None:
        with pytest.raises(TypeError, match="momentum must be a finite real number"):
            StreamingBatchNormalizer(momentum=True)

    def test_accepts_genuine_numpy_scalar_decay(self) -> None:
        """A real (non-spoofed) numpy scalar type must remain accepted."""
        np = pytest.importorskip("numpy")
        normalizer = EMANormalizer(decay=np.float64(0.75))
        assert normalizer._decay == pytest.approx(0.75)

    def test_accepts_genuine_int_momentum(self) -> None:
        """A real (non-spoofed) int must remain accepted and coerced to float."""
        normalizer = StreamingBatchNormalizer(momentum=1)
        assert normalizer._momentum == 1.0
