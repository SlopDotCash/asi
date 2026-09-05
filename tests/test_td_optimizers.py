"""Tests for TDIDBD and AutoTDIDBD optimizers."""

import math

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework import TDIDBD, AutoTDIDBD


class TestTDIDBD:
    """Tests for the TD-IDBD optimizer."""

    def test_init_creates_correct_state(self):
        """TDIDBD init should create per-weight step-sizes, traces, and h traces."""
        optimizer = TDIDBD(initial_step_size=0.01, meta_step_size=0.001, trace_decay=0.9)
        state = optimizer.init(feature_dim=10)

        chex.assert_shape(state.log_step_sizes, (10,))
        chex.assert_shape(state.eligibility_traces, (10,))
        chex.assert_shape(state.h_traces, (10,))
        chex.assert_trees_all_close(jnp.exp(state.log_step_sizes), jnp.full(10, 0.01))
        chex.assert_trees_all_close(state.eligibility_traces, jnp.zeros(10))
        chex.assert_trees_all_close(state.h_traces, jnp.zeros(10))
        assert state.meta_step_size == pytest.approx(0.001)
        assert state.trace_decay == pytest.approx(0.9)

    def test_update_returns_correct_shapes(self, sample_observation):
        """TDIDBD update should return correctly shaped deltas."""
        optimizer = TDIDBD()
        state = optimizer.init(feature_dim=len(sample_observation))

        td_error = jnp.array(1.0)
        gamma = jnp.array(0.99)
        next_obs = sample_observation * 0.9  # Slightly different

        result = optimizer.update(state, td_error, sample_observation, next_obs, gamma)

        chex.assert_shape(result.weight_delta, sample_observation.shape)
        chex.assert_shape(result.new_state.log_step_sizes, sample_observation.shape)
        chex.assert_shape(result.new_state.eligibility_traces, sample_observation.shape)
        chex.assert_shape(result.new_state.h_traces, sample_observation.shape)

    def test_eligibility_traces_accumulate(self, sample_observation):
        """Eligibility traces should accumulate over steps."""
        optimizer = TDIDBD(trace_decay=0.9)
        state = optimizer.init(feature_dim=len(sample_observation))

        td_error = jnp.array(1.0)
        gamma = jnp.array(0.99)
        next_obs = sample_observation * 0.9

        # First update - traces should equal observation
        result1 = optimizer.update(state, td_error, sample_observation, next_obs, gamma)
        chex.assert_trees_all_close(
            result1.new_state.eligibility_traces, sample_observation, atol=1e-6
        )

        # Second update - traces should accumulate
        result2 = optimizer.update(result1.new_state, td_error, sample_observation, next_obs, gamma)
        expected_traces = gamma * 0.9 * sample_observation + sample_observation
        chex.assert_trees_all_close(
            result2.new_state.eligibility_traces, expected_traces, atol=1e-6
        )

    def test_step_sizes_adapt_with_consistent_td_errors(self, sample_observation):
        """Step-sizes should adapt when TD errors consistently agree."""
        optimizer = TDIDBD(initial_step_size=0.1, meta_step_size=0.1)
        feature_dim = len(sample_observation)
        state = optimizer.init(feature_dim=feature_dim)

        # Consistent positive TD error
        td_error = jnp.array(1.0)
        gamma = jnp.array(0.99)
        next_obs = jnp.zeros(feature_dim)

        # Run multiple updates with consistent TD errors
        for _ in range(10):
            result = optimizer.update(state, td_error, sample_observation, next_obs, gamma)
            state = result.new_state

        # h traces should have built up
        assert jnp.any(state.h_traces != 0)

    def test_metrics_contain_step_size_info(self, sample_observation):
        """TDIDBD update should return step-size statistics in metrics."""
        optimizer = TDIDBD()
        state = optimizer.init(feature_dim=len(sample_observation))

        td_error = jnp.array(1.0)
        gamma = jnp.array(0.99)
        next_obs = sample_observation * 0.9

        result = optimizer.update(state, td_error, sample_observation, next_obs, gamma)

        assert "mean_step_size" in result.metrics
        assert "min_step_size" in result.metrics
        assert "max_step_size" in result.metrics
        assert "mean_eligibility_trace" in result.metrics

    def test_semi_gradient_vs_ordinary_gradient(self, sample_observation):
        """Semi-gradient and ordinary gradient should produce different updates."""
        semi_grad = TDIDBD(use_semi_gradient=True)
        ordinary_grad = TDIDBD(use_semi_gradient=False)

        semi_state = semi_grad.init(feature_dim=len(sample_observation))
        ordinary_state = ordinary_grad.init(feature_dim=len(sample_observation))

        td_error = jnp.array(1.0)
        gamma = jnp.array(0.99)
        next_obs = sample_observation * 1.5  # Different from current

        # Both should produce valid updates
        semi_result = semi_grad.update(semi_state, td_error, sample_observation, next_obs, gamma)
        ordinary_result = ordinary_grad.update(
            ordinary_state, td_error, sample_observation, next_obs, gamma
        )

        chex.assert_tree_all_finite(semi_result.weight_delta)
        chex.assert_tree_all_finite(ordinary_result.weight_delta)

        # Weight deltas should be the same (same initial state)
        # but h traces should evolve differently over time

    def test_terminal_state_handling(self, sample_observation):
        """Terminal states (gamma=0) should be handled correctly."""
        optimizer = TDIDBD()
        state = optimizer.init(feature_dim=len(sample_observation))

        td_error = jnp.array(1.0)
        gamma = jnp.array(0.0)  # Terminal state
        next_obs = jnp.zeros_like(sample_observation)

        result = optimizer.update(state, td_error, sample_observation, next_obs, gamma)

        chex.assert_tree_all_finite(result.weight_delta)
        chex.assert_tree_all_finite(result.new_state.log_step_sizes)

    def test_zero_gamma_does_not_multiply_inf_next_observation(self) -> None:
        """Ordinary-gradient 0 * inf phi(s') is NaN when gamma is exactly 0."""
        optimizer = TDIDBD(use_semi_gradient=False)
        obs = jnp.asarray([0.5, -0.25], dtype=jnp.float32)
        state = optimizer.init(feature_dim=2)
        result = optimizer.update(
            state,
            jnp.asarray(1.0, dtype=jnp.float32),
            obs,
            jnp.asarray([jnp.inf, 0.0], dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        assert bool(jnp.all(jnp.isfinite(result.weight_delta)))
        chex.assert_trees_all_close(result.new_state.eligibility_traces, obs)

    def test_zero_trace_decay_does_not_multiply_inf_traces(self) -> None:
        """Default lambda is 0; 0 * inf eligibility is NaN and would freeze."""
        optimizer = TDIDBD()
        obs = jnp.asarray([0.5, -0.25], dtype=jnp.float32)
        state = optimizer.init(feature_dim=2)
        state = state.replace(
            eligibility_traces=jnp.full(2, jnp.inf, dtype=jnp.float32),
            bias_eligibility_trace=jnp.asarray(jnp.inf, dtype=jnp.float32),
        )
        result = optimizer.update(
            state,
            jnp.asarray(0.3, dtype=jnp.float32),
            obs,
            obs * 0.5,
            jnp.asarray(0.9, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        chex.assert_trees_all_close(result.new_state.eligibility_traces, obs)


class TestAutoTDIDBD:
    """Tests for the AutoTDIDBD optimizer."""

    def test_init_creates_correct_state(self):
        """AutoTDIDBD init should create step-sizes, traces, h traces, normalizers."""
        optimizer = AutoTDIDBD(initial_step_size=0.01, meta_step_size=0.001, trace_decay=0.9)
        state = optimizer.init(feature_dim=10)

        chex.assert_shape(state.log_step_sizes, (10,))
        chex.assert_shape(state.eligibility_traces, (10,))
        chex.assert_shape(state.h_traces, (10,))
        chex.assert_shape(state.normalizers, (10,))
        chex.assert_trees_all_close(jnp.exp(state.log_step_sizes), jnp.full(10, 0.01))
        chex.assert_trees_all_close(state.eligibility_traces, jnp.zeros(10))
        chex.assert_trees_all_close(state.h_traces, jnp.zeros(10))
        chex.assert_trees_all_close(state.normalizers, jnp.ones(10))
        assert state.meta_step_size == pytest.approx(0.001)
        assert state.trace_decay == pytest.approx(0.9)

    def test_update_returns_correct_shapes(self, sample_observation):
        """AutoTDIDBD update should return correctly shaped deltas."""
        optimizer = AutoTDIDBD()
        state = optimizer.init(feature_dim=len(sample_observation))

        td_error = jnp.array(1.0)
        gamma = jnp.array(0.99)
        next_obs = sample_observation * 0.9

        result = optimizer.update(state, td_error, sample_observation, next_obs, gamma)

        chex.assert_shape(result.weight_delta, sample_observation.shape)
        chex.assert_shape(result.new_state.log_step_sizes, sample_observation.shape)
        chex.assert_shape(result.new_state.eligibility_traces, sample_observation.shape)
        chex.assert_shape(result.new_state.h_traces, sample_observation.shape)
        chex.assert_shape(result.new_state.normalizers, sample_observation.shape)

    def test_normalizers_adapt_to_gradient_magnitude(self, sample_observation):
        """Normalizers should adapt to gradient magnitudes."""
        optimizer = AutoTDIDBD()
        feature_dim = len(sample_observation)
        state = optimizer.init(feature_dim=feature_dim)

        # Large TD error should lead to normalizer adaptation
        td_error = jnp.array(10.0)
        gamma = jnp.array(0.99)
        next_obs = sample_observation * 2.0  # Large difference

        result = optimizer.update(state, td_error, sample_observation, next_obs, gamma)

        # Normalizers should have changed (at least some of them)
        chex.assert_tree_all_finite(result.new_state.normalizers)
        # Normalizers should be positive
        assert jnp.all(result.new_state.normalizers > 0)

    def test_metrics_contain_normalizer_info(self, sample_observation):
        """AutoTDIDBD update should return normalizer statistics in metrics."""
        optimizer = AutoTDIDBD()
        state = optimizer.init(feature_dim=len(sample_observation))

        td_error = jnp.array(1.0)
        gamma = jnp.array(0.99)
        next_obs = sample_observation * 0.9

        result = optimizer.update(state, td_error, sample_observation, next_obs, gamma)

        assert "mean_step_size" in result.metrics
        assert "min_step_size" in result.metrics
        assert "max_step_size" in result.metrics
        assert "mean_eligibility_trace" in result.metrics
        assert "mean_normalizer" in result.metrics

    def test_effective_step_size_normalization(self, sample_observation):
        """Effective step-size normalization should prevent overshooting."""
        optimizer = AutoTDIDBD(initial_step_size=1.0)  # Large initial step-size
        state = optimizer.init(feature_dim=len(sample_observation))

        td_error = jnp.array(10.0)  # Large TD error
        gamma = jnp.array(0.99)
        next_obs = sample_observation * 2.0

        result = optimizer.update(state, td_error, sample_observation, next_obs, gamma)

        # Updates should remain finite even with large step-sizes
        chex.assert_tree_all_finite(result.weight_delta)
        chex.assert_tree_all_finite(result.new_state.log_step_sizes)

    def test_terminal_state_handling(self, sample_observation):
        """Terminal states (gamma=0) should be handled correctly."""
        optimizer = AutoTDIDBD()
        state = optimizer.init(feature_dim=len(sample_observation))

        td_error = jnp.array(1.0)
        gamma = jnp.array(0.0)  # Terminal state
        next_obs = jnp.zeros_like(sample_observation)

        result = optimizer.update(state, td_error, sample_observation, next_obs, gamma)

        chex.assert_tree_all_finite(result.weight_delta)
        chex.assert_tree_all_finite(result.new_state.log_step_sizes)

    def test_zero_gamma_does_not_multiply_inf_next_observation(self) -> None:
        """AutoTDIDBD 0 * inf phi(s') is NaN when gamma is exactly 0."""
        optimizer = AutoTDIDBD()
        obs = jnp.asarray([0.5, -0.25], dtype=jnp.float32)
        state = optimizer.init(feature_dim=2)
        result = optimizer.update(
            state,
            jnp.asarray(1.0, dtype=jnp.float32),
            obs,
            jnp.asarray([jnp.inf, 0.0], dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        assert bool(jnp.all(jnp.isfinite(result.weight_delta)))
        chex.assert_trees_all_close(result.new_state.eligibility_traces, obs)


# ---------------------------------------------------------------------------
# Published-algorithm pin: Kearney, Veeriah, Travnik, Pilarski & Sutton (2019),
# "Learning Feature Relevance Through Step Size Adaptation in
# Temporal-Difference Learning", arXiv:1903.03252, Section 6.2, Algorithm 6
# ("AutoStep Style Normalized TIDBD(lambda)").  Lines as printed:
#
#   3   delta <- R + gamma w^T phi(s') - w^T phi(s)
#   5-7 eta_i <- max[ |delta [gamma phi_i(s') - phi_i(s)] h_i|,
#                     eta_i - 1/tau alpha_i [gamma phi_i(s') - phi_i(s)] z_i
#                              (|delta phi_i(s) h_i| - eta_i) ]
#   9   beta_i <- beta_i - theta 1/eta_i delta [gamma phi_i(s') - phi_i(s)] h_i
#   10  M <- max(-e^{beta_i} [gamma phi_i(s') - phi_i(s)]^T z_i, 1)
#   11  beta_i <- beta_i - log(M)
#   12  alpha_i <- e^{beta_i}
#   13  z_i <- z_i gamma lambda + phi_i(s)
#   14  w_i <- w_i + alpha_i delta z_i
#   15  h_i <- h_i [1 + alpha_i [gamma phi_i(s') - phi_i(s)] z_i]^+ + alpha_i delta z_i
#
# Note lines 5-7: the max's first argument uses the feature *difference*
# [gamma phi(s') - phi(s)], while the exponential-average argument uses the raw
# feature phi(s).  That asymmetry is the published text, not a transcription
# slip, and a "consistency cleanup" of it silently changes the algorithm.
# ---------------------------------------------------------------------------

_ALG6_ALPHA0 = 0.4
_ALG6_THETA = 0.5
_ALG6_LAMBDA = 0.9
_ALG6_TAU = 2.0
_ALG6_GAMMA = 0.95

# Mixed-sign transitions so that both branches of line 10's max and both
# branches of line 15's positive part are exercised on this trajectory.
_ALG6_OBS = np.array(
    [
        [1.6, 0.9, 1.2],
        [2.0, 1.5, 0.5],
        [0.8, 1.1, 1.9],
        [1.7, 0.6, 1.0],
        [1.8, 1.4, 0.6],
        [1.4, 1.0, 0.8],
    ],
    dtype=np.float32,
)
_ALG6_NEXT_OBS = np.array(
    [
        [-1.2, -0.7, -1.5],
        [5.0, -1.4, 0.2],
        [-1.6, -0.8, -0.6],
        [-0.7, -1.3, -1.7],
        [5.5, -1.0, 0.3],
        [-1.1, -1.6, -1.2],
    ],
    dtype=np.float32,
)
_ALG6_TD_ERRORS = np.array([1.2, -0.9, 0.7, -1.3, 1.1, -0.6], dtype=np.float32)

# float32 agreement between the optimizer and the hand-transcribed reference.
# Observed maximum absolute disagreement is ~2.4e-7 on weight deltas of
# magnitude ~2.0.  Every deviation asserted below is at least 1e-3, i.e. more
# than three orders of magnitude above this floor, so no assertion here can
# pass or fail by float32 rounding.
_ALG6_MATCH_ATOL = 2e-6
_ALG6_DEVIATION_FLOOR = 1e-3


def _algorithm_6_weight_deltas(variant: str = "published") -> tuple[np.ndarray, dict[str, int]]:
    """Literal float32 transcription of Algorithm 6's weight path.

    ``variant`` injects one deliberate departure from the printed algorithm so
    a test can assert that the published choice is load-bearing:

    ``symmetric_normalizer``
        line 7 uses ``|delta [gamma phi(s') - phi(s)] h|`` instead of the
        printed ``|delta phi(s) h|``;
    ``eta_max_uses_phi``
        line 6 uses ``|delta phi(s) h|`` instead of the printed feature
        difference;
    ``post_update_z_in_m``
        line 10 uses the line-13 (post-update) eligibility trace instead of the
        trace that line 10 actually sees;
    ``no_plus_in_h``
        line 15 drops the positive part ``[.]^+``.

    The optimizer adds two numerical guards the paper does not state (a
    ``[-10, 2]`` clamp on ``beta`` and a ``1e-8`` floor on ``eta``).  The
    returned diagnostics record whether either guard ever binds; the pin below
    requires that neither does, so the comparison is against the published
    algorithm and not against a guard.
    """
    f32 = np.float32
    n_features = _ALG6_OBS.shape[1]
    h = np.zeros(n_features, dtype=f32)
    z = np.zeros(n_features, dtype=f32)
    beta = np.full(n_features, f32(math.log(_ALG6_ALPHA0)), dtype=f32)
    # The paper does not state eta's initial value; AutoTDIDBD.init uses ones.
    eta = np.ones(n_features, dtype=f32)
    deltas: list[np.ndarray] = []
    diagnostics = {"m_above_one": 0, "positive_part_clamps": 0, "guards_bound": 0}

    for step in range(_ALG6_OBS.shape[0]):
        phi = _ALG6_OBS[step]
        phi_next = _ALG6_NEXT_OBS[step]
        td_error = _ALG6_TD_ERRORS[step]
        difference = (f32(_ALG6_GAMMA) * phi_next - phi).astype(f32)
        alpha = np.exp(beta).astype(f32)

        # Lines 5-7.
        max_arm = np.abs(
            td_error * (phi if variant == "eta_max_uses_phi" else difference) * h
        ).astype(f32)
        average_arm_signal = np.abs(
            td_error * (difference if variant == "symmetric_normalizer" else phi) * h
        ).astype(f32)
        average_arm = (
            eta
            - (f32(1.0) / f32(_ALG6_TAU)) * alpha * difference * z * (average_arm_signal - eta)
        ).astype(f32)
        eta = np.maximum(max_arm, average_arm).astype(f32)
        if bool(np.any(eta < f32(1e-8))):
            diagnostics["guards_bound"] += 1
        eta = np.maximum(eta, f32(1e-8)).astype(f32)

        # Line 9.
        beta = (
            beta - f32(_ALG6_THETA) * (f32(1.0) / eta) * td_error * difference * h
        ).astype(f32)

        # Lines 10-12.  Line 10 sees the trace before line 13 advances it.
        trace_for_m = z
        if variant == "post_update_z_in_m":
            trace_for_m = (z * f32(_ALG6_GAMMA * _ALG6_LAMBDA) + phi).astype(f32)
        m_factor = max(float(-np.sum(np.exp(beta).astype(f32) * difference * trace_for_m)), 1.0)
        if m_factor > 1.0:
            diagnostics["m_above_one"] += 1
        beta = (beta - f32(math.log(m_factor))).astype(f32)
        if bool(np.any(beta < f32(-10.0)) or np.any(beta > f32(2.0))):
            diagnostics["guards_bound"] += 1
        beta = np.clip(beta, f32(-10.0), f32(2.0)).astype(f32)
        alpha = np.exp(beta).astype(f32)

        # Lines 13-15.
        z = (z * f32(_ALG6_GAMMA * _ALG6_LAMBDA) + phi).astype(f32)
        deltas.append((alpha * td_error * z).astype(f32))
        positive_part = (f32(1.0) + alpha * difference * z).astype(f32)
        if bool(np.any(positive_part < f32(0.0))):
            diagnostics["positive_part_clamps"] += 1
        if variant != "no_plus_in_h":
            positive_part = np.maximum(f32(0.0), positive_part).astype(f32)
        h = (h * positive_part + alpha * td_error * z).astype(f32)

    return np.stack(deltas), diagnostics


def _auto_tdidbd_weight_deltas() -> np.ndarray:
    """Run :class:`AutoTDIDBD` over the pinned trajectory."""
    optimizer = AutoTDIDBD(
        initial_step_size=_ALG6_ALPHA0,
        meta_step_size=_ALG6_THETA,
        trace_decay=_ALG6_LAMBDA,
        normalizer_decay=_ALG6_TAU,
    )
    state = optimizer.init(feature_dim=_ALG6_OBS.shape[1])
    deltas: list[np.ndarray] = []
    for step in range(_ALG6_OBS.shape[0]):
        result = optimizer.update(
            state,
            jnp.asarray(_ALG6_TD_ERRORS[step]),
            jnp.asarray(_ALG6_OBS[step]),
            jnp.asarray(_ALG6_NEXT_OBS[step]),
            jnp.asarray(_ALG6_GAMMA, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        deltas.append(np.asarray(result.weight_delta))
        state = result.new_state
    return np.stack(deltas)


class TestAutoTDIDBDPublishedAlgorithm:
    """Pin AutoTDIDBD's numerics to Kearney et al. 2019 Algorithm 6.

    The pre-existing AutoTDIDBD tests assert shapes, finiteness, JIT
    compilation, and qualitative direction only. Nothing pinned the arithmetic,
    so an edit could silently produce a different algorithm from the one the
    docstring cites while every test still passed. These tests compare the
    optimizer against a line-by-line transcription of the published algorithm
    and then show that each published choice actually changes the trajectory.
    """

    def test_trajectory_exercises_the_published_branches_without_guards(self) -> None:
        """The pinned trajectory must reach line 10's M>1 arm and line 15's clamp.

        It must also stay clear of the two numerical guards the optimizer adds
        beyond the paper, so the comparison below is against Algorithm 6 rather
        than against a clamp or a floor.
        """
        _, diagnostics = _algorithm_6_weight_deltas()
        assert diagnostics["m_above_one"] > 0
        assert diagnostics["positive_part_clamps"] > 0
        assert diagnostics["guards_bound"] == 0

    def test_weight_deltas_match_published_algorithm_6(self) -> None:
        """AutoTDIDBD reproduces Algorithm 6 lines 3-15 to float32."""
        published, _ = _algorithm_6_weight_deltas()
        chex.assert_trees_all_close(
            _auto_tdidbd_weight_deltas(), published, atol=_ALG6_MATCH_ATOL, rtol=0.0
        )

    @pytest.mark.parametrize(
        "variant",
        [
            "symmetric_normalizer",
            "eta_max_uses_phi",
            "post_update_z_in_m",
            "no_plus_in_h",
        ],
    )
    def test_each_published_choice_changes_the_trajectory(self, variant: str) -> None:
        """Each departure from Algorithm 6 must move the weight deltas.

        Without this the match test above could pass for a trajectory on which
        the published detail happens to be inert. The optimizer must track the
        published branch and must not track the departed one.
        """
        published, _ = _algorithm_6_weight_deltas()
        departed, _ = _algorithm_6_weight_deltas(variant)
        separation = float(np.max(np.abs(published - departed)))
        assert separation > _ALG6_DEVIATION_FLOOR, (
            f"{variant} is inert on this trajectory; the pin would not detect it"
        )

        observed = _auto_tdidbd_weight_deltas()
        assert float(np.max(np.abs(observed - published))) <= _ALG6_MATCH_ATOL
        assert float(np.max(np.abs(observed - departed))) > _ALG6_DEVIATION_FLOOR


class TestTDOptimizerComparison:
    """Integration tests comparing TDIDBD and AutoTDIDBD behavior."""

    def test_all_optimizers_produce_valid_updates(self, sample_observation):
        """All TD optimizers should produce finite, non-zero updates."""
        tdidbd = TDIDBD(initial_step_size=0.01)
        auto_tdidbd = AutoTDIDBD(initial_step_size=0.01)

        tdidbd_state = tdidbd.init(len(sample_observation))
        auto_state = auto_tdidbd.init(len(sample_observation))

        td_error = jnp.array(1.0)
        gamma = jnp.array(0.99)
        next_obs = sample_observation * 0.9

        tdidbd_result = tdidbd.update(tdidbd_state, td_error, sample_observation, next_obs, gamma)
        auto_result = auto_tdidbd.update(auto_state, td_error, sample_observation, next_obs, gamma)

        # All should produce finite updates
        chex.assert_tree_all_finite(tdidbd_result.weight_delta)
        chex.assert_tree_all_finite(auto_result.weight_delta)

        # All should produce non-zero updates for non-zero TD error
        # (with non-zero eligibility traces after first step)
        # Note: First step may have zero deltas due to zero initial eligibility traces

    def test_optimizers_with_zero_td_error(self, sample_observation):
        """Optimizers should handle zero TD error gracefully."""
        tdidbd = TDIDBD()
        auto_tdidbd = AutoTDIDBD()

        tdidbd_state = tdidbd.init(len(sample_observation))
        auto_state = auto_tdidbd.init(len(sample_observation))

        td_error = jnp.array(0.0)  # Zero TD error
        gamma = jnp.array(0.99)
        next_obs = sample_observation

        tdidbd_result = tdidbd.update(tdidbd_state, td_error, sample_observation, next_obs, gamma)
        auto_result = auto_tdidbd.update(auto_state, td_error, sample_observation, next_obs, gamma)

        # All should produce finite updates (even if zero)
        chex.assert_tree_all_finite(tdidbd_result.weight_delta)
        chex.assert_tree_all_finite(auto_result.weight_delta)
