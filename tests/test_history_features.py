"""Tests for the history-feature extractor (Step 3 Phase D)."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.history_features import (
    HistoryFeatureExtractor,
    HistoryFeatureState,
)


class TestInit:
    def test_init_traces_zero(self) -> None:
        ex = HistoryFeatureExtractor(raw_dim=4, decay_rates=(0.5, 0.9))
        s = ex.init()
        chex.assert_shape(s.traces, (2, 4))
        chex.assert_trees_all_close(s.traces, jnp.zeros((2, 4)))

    def test_feature_dim_with_raw(self) -> None:
        ex = HistoryFeatureExtractor(
            raw_dim=4, decay_rates=(0.5, 0.9, 0.99), include_raw=True
        )
        assert ex.feature_dim() == 4 + 4 * 3

    def test_feature_dim_without_raw(self) -> None:
        ex = HistoryFeatureExtractor(
            raw_dim=4, decay_rates=(0.5, 0.9), include_raw=False
        )
        assert ex.feature_dim() == 4 * 2

    def test_feature_dim_subset_channels(self) -> None:
        ex = HistoryFeatureExtractor(
            raw_dim=10, decay_rates=(0.9,), channels=(0, 3, 5)
        )
        assert ex.feature_dim() == 10 + 3 * 1


class TestValidation:
    def test_invalid_decay_rate_negative(self) -> None:
        with pytest.raises(ValueError, match="decay_rates"):
            HistoryFeatureExtractor(raw_dim=4, decay_rates=(-0.1, 0.9))

    def test_invalid_decay_rate_one(self) -> None:
        with pytest.raises(ValueError, match="decay_rates"):
            HistoryFeatureExtractor(raw_dim=4, decay_rates=(0.5, 1.0))

    def test_invalid_decay_rate_nan(self) -> None:
        with pytest.raises(ValueError, match="decay_rates"):
            HistoryFeatureExtractor(raw_dim=4, decay_rates=(float("nan"), 0.9))

    def test_invalid_channel_index(self) -> None:
        with pytest.raises(ValueError, match="channels"):
            HistoryFeatureExtractor(raw_dim=4, channels=(0, 5))

    @pytest.mark.parametrize("value", [True, 1.5, 0, 2**31, "4"])
    def test_raw_dim_requires_supported_signed_int32(self, value: object) -> None:
        with pytest.raises(ValueError, match="raw_dim"):
            HistoryFeatureExtractor(raw_dim=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "rates",
        [
            [0.5],
            (True,),
            (float("inf"),),
            (1.0 - 1e-10,),
            (1e100,),
            (Fraction(-1, 10**50),),
        ],
    )
    def test_decay_rates_require_exact_tuple_and_float32_domain(
        self, rates: object
    ) -> None:
        with pytest.raises(ValueError, match="decay_rates"):
            HistoryFeatureExtractor(raw_dim=2, decay_rates=rates)  # type: ignore[arg-type]

    def test_scalar_class_spoofs_are_rejected_without_conversion(self) -> None:
        class Spoof:
            @property
            def __class__(self) -> type[float]:  # type: ignore[override]
                return float

            def __float__(self) -> float:
                raise RuntimeError("must not convert")

        with pytest.raises(ValueError, match="raw_dim"):
            HistoryFeatureExtractor(raw_dim=Spoof())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="decay_rates"):
            HistoryFeatureExtractor(raw_dim=2, decay_rates=(Spoof(),))  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "channels",
        [[0], (True,), (1.5,), ("0",), (2,)],
    )
    def test_channels_require_exact_tuple_of_in_range_integers(
        self, channels: object
    ) -> None:
        with pytest.raises(ValueError, match="channels"):
            HistoryFeatureExtractor(raw_dim=2, channels=channels)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [1, 0, np.bool_(True), "true"])
    def test_include_raw_requires_actual_bool(self, value: object) -> None:
        with pytest.raises(ValueError, match="include_raw"):
            HistoryFeatureExtractor(raw_dim=2, include_raw=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("integer_type", [np.longlong, np.ulonglong])
    def test_numpy_integer_scalars_are_canonicalized(
        self, integer_type: type[np.integer[object]]
    ) -> None:
        extractor = HistoryFeatureExtractor(
            raw_dim=integer_type(3),
            channels=(integer_type(0), integer_type(2)),
        )
        assert type(extractor.raw_dim) is int
        assert all(type(channel) is int for channel in extractor.channels)

    def test_nonbuiltin_real_rates_are_canonicalized_for_json(self) -> None:
        extractor = HistoryFeatureExtractor(
            raw_dim=2,
            decay_rates=(np.float32(0.5), Fraction(3, 4)),
        )
        assert all(type(rate) is float for rate in extractor.decay_rates)
        assert extractor.to_config()["decay_rates"] == [0.5, 0.75]

    def test_derived_feature_dim_requires_signed_int32(self) -> None:
        with pytest.raises(ValueError, match="feature_dim"):
            HistoryFeatureExtractor(
                raw_dim=2**31 - 1,
                decay_rates=(0.5,),
                channels=(0,),
            )
        boundary = HistoryFeatureExtractor(
            raw_dim=2**31 - 2,
            decay_rates=(0.5,),
            channels=(0,),
        )
        assert boundary.feature_dim() == 2**31 - 1


class TestStep:
    def test_first_step_traces(self) -> None:
        ex = HistoryFeatureExtractor(raw_dim=3, decay_rates=(0.9,))
        s = ex.init()
        obs = jnp.array([1.0, 2.0, 3.0])
        aug, s2 = ex.step(s, obs)
        # First step: trace = 0.9 * 0 + 0.1 * obs = 0.1 * obs
        chex.assert_trees_all_close(s2.traces[0], 0.1 * obs, atol=1e-7)
        # Augmented: raw + traces (raw_dim=3, decay rates=1, so 6 dims)
        chex.assert_shape(aug, (6,))
        chex.assert_trees_all_close(aug[:3], obs)
        chex.assert_trees_all_close(aug[3:], 0.1 * obs, atol=1e-7)

    def test_decay_dynamics(self) -> None:
        ex = HistoryFeatureExtractor(raw_dim=1, decay_rates=(0.5,))
        s = ex.init()
        obs1 = jnp.array([1.0])
        obs2 = jnp.array([0.0])

        # Step 1: trace = 0.5*0 + 0.5*1 = 0.5
        _, s1 = ex.step(s, obs1)
        chex.assert_trees_all_close(s1.traces[0], jnp.array([0.5]), atol=1e-7)

        # Step 2 with obs=0: trace = 0.5*0.5 + 0.5*0 = 0.25
        _, s2 = ex.step(s1, obs2)
        chex.assert_trees_all_close(s2.traces[0], jnp.array([0.25]), atol=1e-7)

    def test_multiple_decay_rates_independent(self) -> None:
        ex = HistoryFeatureExtractor(raw_dim=1, decay_rates=(0.0, 0.5, 0.9))
        s = ex.init()
        obs = jnp.array([1.0])
        _, s1 = ex.step(s, obs)
        # decay=0.0: trace = 0.0*0 + 1.0*1 = 1.0
        # decay=0.5: trace = 0.5
        # decay=0.9: trace = 0.1
        chex.assert_trees_all_close(
            s1.traces[:, 0], jnp.array([1.0, 0.5, 0.1]), atol=1e-7
        )

    def test_subset_channels_only(self) -> None:
        ex = HistoryFeatureExtractor(
            raw_dim=4, decay_rates=(0.5,), channels=(1, 3), include_raw=False
        )
        s = ex.init()
        obs = jnp.array([10.0, 20.0, 30.0, 40.0])
        aug, s1 = ex.step(s, obs)
        # Only channels 1, 3 are tracked
        # trace = 0.5*0 + 0.5*[20, 40] = [10, 20]
        chex.assert_shape(aug, (2,))
        chex.assert_trees_all_close(aug, jnp.array([10.0, 20.0]), atol=1e-7)


class TestStepShapeContract:
    """The docstring promises ``observation`` has shape ``(raw_dim,)`` and
    ``augmented`` has shape ``(out_dim,)``; ``step`` must enforce the input
    side rather than silently truncating/duplicating channels via JAX's
    clipped out-of-bounds indexing."""

    def test_rejects_short_observation(self) -> None:
        ex = HistoryFeatureExtractor(raw_dim=4, decay_rates=(0.5, 0.9))
        s = ex.init()
        short_obs = jnp.array([1.0, 2.0])
        with pytest.raises(ValueError, match="observation must have shape"):
            ex.step(s, short_obs)

    def test_rejects_long_observation(self) -> None:
        ex = HistoryFeatureExtractor(raw_dim=4, decay_rates=(0.5, 0.9))
        s = ex.init()
        long_obs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="observation must have shape"):
            ex.step(s, long_obs)

    def test_rejects_rank_two_observation(self) -> None:
        ex = HistoryFeatureExtractor(raw_dim=4, decay_rates=(0.5, 0.9))
        s = ex.init()
        rank2_obs = jnp.ones((4, 1))
        with pytest.raises(ValueError, match="observation must have shape"):
            ex.step(s, rank2_obs)

    def test_short_observation_would_silently_duplicate_channels_if_unchecked(
        self,
    ) -> None:
        """Documents the exact corruption this validation prevents: JAX
        clips out-of-bounds fancy-index reads instead of raising, so an
        under-shaped observation used to make channel 3 silently reuse the
        value at channel 1 instead of failing."""
        ex = HistoryFeatureExtractor(raw_dim=4, decay_rates=(0.5,), include_raw=False)
        s = ex.init()
        short_obs = jnp.array([10.0, 20.0])
        channel_indices = jnp.asarray(ex.channels, dtype=jnp.int32)
        clipped_read = short_obs[channel_indices]
        # Without the guard, channels 2 and 3 would clip to index 1 (=20.0)
        # instead of raising -- the corrupted trace this function must never
        # produce.
        chex.assert_trees_all_close(clipped_read, jnp.array([10.0, 20.0, 20.0, 20.0]))
        with pytest.raises(ValueError, match="observation must have shape"):
            ex.step(s, short_obs)


class TestJitAndScan:
    def test_step_jit_compiles(self) -> None:
        ex = HistoryFeatureExtractor(raw_dim=4, decay_rates=(0.5, 0.9))
        s = ex.init()
        # Decorator already JITs; just verify two calls produce same results
        obs = jnp.ones(4)
        out1, s1 = ex.step(s, obs)
        out2, s2 = ex.step(s, obs)
        chex.assert_trees_all_close(out1, out2)
        chex.assert_trees_all_close(s1.traces, s2.traces)

    def test_scan_compatibility(self) -> None:
        ex = HistoryFeatureExtractor(raw_dim=3, decay_rates=(0.9,))
        s0 = ex.init()
        observations = jr.normal(jr.key(0), (50, 3))

        def step_fn(
            state: HistoryFeatureState,
            obs: jax.Array,
        ) -> tuple[HistoryFeatureState, jax.Array]:
            aug, new_state = cast(
                tuple[jax.Array, HistoryFeatureState],
                ex.step(state, obs),
            )
            return new_state, aug

        final_state, augmented = jax.lax.scan(step_fn, s0, observations)
        chex.assert_shape(augmented, (50, 6))
        chex.assert_tree_all_finite(augmented)
        chex.assert_tree_all_finite(final_state.traces)

    def test_nonfinite_observation_holds_warm_traces_and_recovers(self) -> None:
        ex = HistoryFeatureExtractor(
            raw_dim=3,
            decay_rates=(0.5, 0.9),
            channels=(0, 2),
        )
        _, warm_state = ex.step(
            ex.init(),
            jnp.asarray([2.0, 4.0, 6.0], dtype=jnp.float32),
        )

        @jax.jit
        def run(
            state: HistoryFeatureState,
            observation: jax.Array,
        ) -> tuple[jax.Array, HistoryFeatureState]:
            return cast(
                tuple[jax.Array, HistoryFeatureState],
                ex.step(state, observation),
            )

        augmented, held_state = run(
            warm_state,
            jnp.asarray([jnp.inf, 8.0, jnp.nan], dtype=jnp.float32),
        )

        chex.assert_tree_all_finite((augmented, held_state))
        chex.assert_trees_all_equal(held_state, warm_state)
        chex.assert_trees_all_equal(augmented[:3], jnp.asarray([0.0, 8.0, 0.0]))
        chex.assert_trees_all_equal(augmented[3:], warm_state.traces.reshape(-1))

        recovered_augmented, recovered_state = run(
            held_state,
            jnp.asarray([4.0, -2.0, 2.0], dtype=jnp.float32),
        )

        chex.assert_tree_all_finite((recovered_augmented, recovered_state))
        chex.assert_trees_all_close(
            recovered_state.traces,
            jnp.asarray([[2.5, 2.5], [0.58, 0.74]], dtype=jnp.float32),
        )

    def test_zero_decay_does_not_multiply_inf_traces(self) -> None:
        """decay=0 keeps only the current obs; 0 * inf stored traces is NaN."""
        ex = HistoryFeatureExtractor(raw_dim=1, decay_rates=(0.0,), include_raw=False)
        poisoned = HistoryFeatureState(
            traces=jnp.asarray([[jnp.inf]], dtype=jnp.float32)
        )  # type: ignore[call-arg]
        observation = jnp.asarray([3.0], dtype=jnp.float32)
        raw = jnp.asarray(0.0, dtype=jnp.float32) * poisoned.traces
        assert not bool(jnp.all(jnp.isfinite(raw)))

        augmented, next_state = ex.step(poisoned, observation)
        chex.assert_tree_all_finite((augmented, next_state))
        chex.assert_trees_all_close(next_state.traces[0], observation)
        chex.assert_trees_all_close(augmented, observation)

    def test_untracked_nonfinite_coordinate_holds_raw_disabled_traces(self) -> None:
        ex = HistoryFeatureExtractor(
            raw_dim=3,
            decay_rates=(0.5,),
            channels=(0, 2),
            include_raw=False,
        )
        _, warm_state = ex.step(
            ex.init(),
            jnp.asarray([2.0, 4.0, 6.0], dtype=jnp.float32),
        )

        augmented, held_state = ex.step(
            warm_state,
            jnp.asarray([10.0, jnp.inf, 14.0], dtype=jnp.float32),
        )

        chex.assert_tree_all_finite((augmented, held_state))
        chex.assert_trees_all_equal(held_state, warm_state)
        chex.assert_trees_all_equal(augmented, warm_state.traces.reshape(-1))

    def test_finite_step_is_bitwise_unchanged(self) -> None:
        ex = HistoryFeatureExtractor(
            raw_dim=3,
            decay_rates=(0.25, 0.75),
            channels=(0, 2),
        )
        state = HistoryFeatureState(
            traces=jnp.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=jnp.float32)
        )  # type: ignore[call-arg]
        observation = jnp.asarray([8.0, -5.0, 6.0], dtype=jnp.float32)

        augmented, next_state = ex.step(state, observation)
        tracked = observation[jnp.asarray(ex.channels, dtype=jnp.int32)]
        decay = jnp.asarray(ex.decay_rates, dtype=jnp.float32)[:, None]
        expected_traces = decay * state.traces + (1.0 - decay) * tracked[None, :]
        expected_augmented = jnp.concatenate([observation, expected_traces.reshape(-1)])

        assert np.asarray(next_state.traces).tobytes() == np.asarray(
            expected_traces
        ).tobytes()
        assert np.asarray(augmented).tobytes() == np.asarray(
            expected_augmented
        ).tobytes()


class TestConfig:
    def test_roundtrip(self) -> None:
        original = HistoryFeatureExtractor(
            raw_dim=5,
            decay_rates=(0.1, 0.5, 0.9),
            channels=(0, 2, 4),
            include_raw=False,
        )
        config = original.to_config()
        restored = HistoryFeatureExtractor.from_config(config)

        assert restored.raw_dim == 5
        assert restored.decay_rates == (0.1, 0.5, 0.9)
        assert restored.channels == (0, 2, 4)
        assert restored.include_raw is False
        assert restored.feature_dim() == original.feature_dim()


# =============================================================================
# Smoke test: history features integrate with a learner
# =============================================================================


class TestIntegration:
    """Basic smoke test: a learner can consume the augmented observations
    and produce finite predictions / updates."""

    def test_extractor_plus_learner_runs(self) -> None:
        from alberta_framework import LMS, LinearLearner

        ex = HistoryFeatureExtractor(raw_dim=2, decay_rates=(0.5, 0.9))
        learner = LinearLearner(optimizer=LMS(step_size=0.05))
        l_state = learner.init(ex.feature_dim())
        h_state = ex.init()

        rng = np.random.default_rng(0)
        for _ in range(50):
            obs = jnp.asarray(rng.normal(size=2).astype(np.float32))
            target = jnp.atleast_1d(jnp.float32(rng.normal()))
            aug, h_state = ex.step(h_state, obs)
            res = learner.update(l_state, aug, target)
            l_state = res.state

        chex.assert_tree_all_finite(l_state.weights)
        chex.assert_tree_all_finite(h_state.traces)
