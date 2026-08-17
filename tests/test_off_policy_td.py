"""Tests for off-policy linear TD with importance sampling (Step 3 Phase E)."""

from __future__ import annotations

from fractions import Fraction

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.learners import LinearLearner
from alberta_framework.core.off_policy_td import (
    ETDLinearLearner,
    ETDUpdateResult,
    GradientTDLinearLearner,
    GradientTDUpdateResult,
    OffPolicyTDLinearLearner,
    OffPolicyTDUpdateResult,
    run_gradient_td_learning_loop,
)
from alberta_framework.core.optimizers import LMS

# =============================================================================
# Init / sanity
# =============================================================================


class TestInit:
    def test_init_zero(self) -> None:
        learner = OffPolicyTDLinearLearner(step_size=0.05)
        s = learner.init(7)
        chex.assert_shape(s.weights, (7,))
        chex.assert_trees_all_close(s.weights, jnp.zeros(7))
        chex.assert_trees_all_close(s.eligibility_traces, jnp.zeros(7))

    def test_invalid_args_raise(self) -> None:
        with pytest.raises(ValueError, match="step_size"):
            OffPolicyTDLinearLearner(step_size=-0.1)
        with pytest.raises(ValueError, match="trace_decay"):
            OffPolicyTDLinearLearner(trace_decay=1.5)
        with pytest.raises(ValueError, match="retrace_clip"):
            OffPolicyTDLinearLearner(retrace_clip=-1.0)
        with pytest.raises(ValueError, match="step_size"):
            ETDLinearLearner(step_size=0.0)
        with pytest.raises(ValueError, match="trace_decay"):
            ETDLinearLearner(trace_decay=-0.1)


@pytest.mark.parametrize(
    "learner",
    [
        OffPolicyTDLinearLearner(step_size=0.1),
        ETDLinearLearner(step_size=0.1),
        GradientTDLinearLearner(step_size=0.1),
    ],
)
@pytest.mark.parametrize("gamma", [-0.25, 1.25])
def test_update_rejects_discount_outside_unit_interval(learner, gamma: float) -> None:
    state = learner.init(1)

    result = learner.update(
        state,
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([2.0], dtype=jnp.float32),
        jnp.asarray(gamma, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )

    chex.assert_trees_all_equal(result.state, state)
    assert not bool(result.update_applied)
    assert float(result.td_error) == 0.0


# =============================================================================
# rho=1 reduces to on-policy TD
# =============================================================================


class TestOnPolicyEquivalence:
    def test_rho_one_lambda_zero_matches_lms_on_terminating_step(self) -> None:
        """With rho=1, gamma=0, the update reduces to LMS-style supervised:
        w += alpha * (R - V) * phi."""
        feature_dim = 4
        alpha = 0.1
        n_steps = 30

        learner = OffPolicyTDLinearLearner(step_size=alpha, trace_decay=0.0, retrace_clip=10.0)
        state = learner.init(feature_dim)

        rng = np.random.default_rng(0)
        observations = jnp.asarray(rng.normal(size=(n_steps, feature_dim)).astype(np.float32))
        rewards = jnp.asarray(rng.normal(size=n_steps).astype(np.float32))

        for t in range(n_steps):
            res = learner.update(
                state,
                observations[t],
                rewards[t],
                jnp.zeros(feature_dim),
                jnp.float32(0.0),
                jnp.float32(1.0),
            )
            state = res.state

        # Reference: pure LMS update from zero weights with the same data
        w_ref = jnp.zeros(feature_dim)
        b_ref = jnp.float32(0.0)
        for t in range(n_steps):
            v = jnp.dot(w_ref, observations[t]) + b_ref
            err = rewards[t] - v
            w_ref = w_ref + alpha * err * observations[t]
            b_ref = b_ref + alpha * err

        chex.assert_trees_all_close(state.weights, w_ref, atol=1e-5)
        chex.assert_trees_all_close(state.bias, b_ref, atol=1e-5)


# =============================================================================
# Per-decision importance sampling
# =============================================================================


class TestPerDecisionImportanceSampling:
    def test_time_varying_rho_enters_each_trace_once(self) -> None:
        """The current ratio scales the new feature; the previous ratio is retained."""
        learner = OffPolicyTDLinearLearner(
            step_size=0.1,
            trace_decay=0.8,
            retrace_clip=float("inf"),
        )
        state = learner.init(2)

        first = learner.update(
            state,
            jnp.array([1.0, 0.0], dtype=jnp.float32),
            jnp.float32(1.0),
            jnp.zeros(2, dtype=jnp.float32),
            jnp.float32(0.9),
            jnp.float32(2.0),
        )
        second = learner.update(
            first.state,
            jnp.array([0.0, 1.0], dtype=jnp.float32),
            jnp.float32(1.0),
            jnp.zeros(2, dtype=jnp.float32),
            jnp.float32(0.9),
            jnp.float32(0.5),
        )

        # z_1 = 2 * phi_1.  On the next transition,
        # z_2 = 0.5 * (0.9 * 0.8 * z_1 + phi_2) = [0.72, 0.5].
        chex.assert_trees_all_close(
            first.state.eligibility_traces,
            jnp.array([2.0, 0.0], dtype=jnp.float32),
        )
        chex.assert_trees_all_close(
            second.state.eligibility_traces,
            jnp.array([0.72, 0.5], dtype=jnp.float32),
        )
        chex.assert_trees_all_close(
            second.state.bias_eligibility_trace,
            jnp.float32(1.22),
        )
        # delta_1 = 1 and delta_2 = 1 + 0.9*0.2 - 0.2 = 0.98.
        chex.assert_trees_all_close(
            second.state.weights,
            jnp.array([0.27056, 0.049], dtype=jnp.float32),
            atol=1e-6,
        )
        chex.assert_trees_all_close(second.state.bias, jnp.float32(0.31956), atol=1e-6)

    def test_constant_rho_retains_prior_weight_updates(self) -> None:
        """The canonical trace is externally equivalent to the old placement at fixed rho."""
        learner = OffPolicyTDLinearLearner(
            step_size=0.1,
            trace_decay=0.8,
            retrace_clip=float("inf"),
        )
        state = learner.init(2)

        for observation in (
            jnp.array([1.0, 0.0], dtype=jnp.float32),
            jnp.array([0.0, 1.0], dtype=jnp.float32),
        ):
            state = learner.update(
                state,
                observation,
                jnp.float32(1.0),
                jnp.zeros(2, dtype=jnp.float32),
                jnp.float32(0.9),
                jnp.float32(2.0),
            ).state

        chex.assert_trees_all_close(
            state.weights,
            jnp.array([0.48224, 0.196], dtype=jnp.float32),
            atol=1e-6,
        )
        chex.assert_trees_all_close(state.bias, jnp.float32(0.67824), atol=1e-6)


# =============================================================================
# ETD(lambda)
# =============================================================================


class TestETDLambda:
    def test_lambda_zero_on_policy_terminating_matches_lms(self) -> None:
        """With rho=1, lambda=0, and gamma=0, ETD reduces to LMS/TD(0)."""
        feature_dim = 4
        alpha = 0.08
        n_steps = 40

        rng = np.random.default_rng(123)
        observations = jnp.asarray(rng.normal(size=(n_steps, feature_dim)).astype(np.float32))
        rewards = jnp.asarray(rng.normal(size=n_steps).astype(np.float32))

        etd = ETDLinearLearner(step_size=alpha, trace_decay=0.0)
        etd_state = etd.init(feature_dim)
        for t in range(n_steps):
            res = etd.update(
                etd_state,
                observations[t],
                rewards[t],
                jnp.zeros(feature_dim),
                jnp.float32(0.0),
                jnp.float32(1.0),
            )
            etd_state = res.state

        lms = LinearLearner(optimizer=LMS(step_size=alpha))
        lms_state = lms.init(feature_dim)
        for t in range(n_steps):
            res = lms.update(lms_state, observations[t], jnp.atleast_1d(rewards[t]))
            lms_state = res.state

        chex.assert_trees_all_close(etd_state.weights, lms_state.weights, atol=1e-5)
        chex.assert_trees_all_close(etd_state.bias, lms_state.bias, atol=1e-5)

    def test_follow_on_and_emphasis_evolve_under_off_policy_rho(self) -> None:
        learner = ETDLinearLearner(step_size=0.05, trace_decay=0.4)
        state = learner.init(2)

        first = learner.update(
            state,
            jnp.array([1.0, 0.0], dtype=jnp.float32),
            jnp.float32(0.0),
            jnp.zeros(2, dtype=jnp.float32),
            jnp.float32(0.9),
            jnp.float32(2.0),
        )
        second = learner.update(
            first.state,
            jnp.array([0.0, 1.0], dtype=jnp.float32),
            jnp.float32(0.0),
            jnp.zeros(2, dtype=jnp.float32),
            jnp.float32(0.8),
            jnp.float32(0.5),
        )

        # F_1 = 2 * 0.9 * 0 + 1 = 1
        # F_2 = 0.5 * 0.8 * 1 + 1 = 1.4
        # M_2 = lambda * i + (1 - lambda) * F_2 = 0.4 + 0.6 * 1.4 = 1.24
        chex.assert_trees_all_close(first.state.follow_on_trace, jnp.float32(1.0))
        chex.assert_trees_all_close(second.state.follow_on_trace, jnp.float32(1.4))
        chex.assert_trees_all_close(second.state.emphasis, jnp.float32(1.24))
        chex.assert_trees_all_close(
            second.state.eligibility_traces,
            jnp.array([0.32, 0.62], dtype=jnp.float32),
            atol=1e-6,
        )

    def test_update_is_jit_compatible(self) -> None:
        learner = ETDLinearLearner(step_size=0.05, trace_decay=0.5)
        state = learner.init(4)
        result = learner.update(
            state,
            jnp.ones(4),
            jnp.float32(1.0),
            jnp.ones(4),
            jnp.float32(0.9),
            jnp.float32(1.5),
        )
        assert isinstance(result, ETDUpdateResult)
        chex.assert_shape(result.metrics, (7,))

    def test_config_roundtrip(self) -> None:
        original = ETDLinearLearner(step_size=0.03, trace_decay=0.7)
        config = original.to_config()
        assert config["type"] == "ETDLinearLearner"
        restored = ETDLinearLearner.from_config(config)
        assert restored.to_config() == config

    def test_bounded_finite_updates(self) -> None:
        learner = ETDLinearLearner(step_size=0.001, trace_decay=0.6)
        state = learner.init(5)

        rng = np.random.default_rng(9)
        for _ in range(1000):
            phi = jnp.asarray(0.2 * rng.normal(size=5).astype(np.float32))
            phi_next = jnp.asarray(0.2 * rng.normal(size=5).astype(np.float32))
            reward = jnp.float32(0.1 * rng.normal())
            rho = jnp.float32(rng.uniform(0.0, 1.8))
            result = learner.update(
                state,
                phi,
                reward,
                phi_next,
                jnp.float32(0.7),
                rho,
            )
            state = result.state

        chex.assert_tree_all_finite(state.weights)
        chex.assert_tree_all_finite(state.eligibility_traces)
        chex.assert_tree_all_finite(state.follow_on_trace)
        chex.assert_tree_all_finite(state.emphasis)
        assert float(jnp.max(jnp.abs(state.weights))) < 5.0


# =============================================================================
# Gradient-TD / TDC
# =============================================================================


class TestGradientTD:
    def test_invalid_args_raise(self) -> None:
        with pytest.raises(ValueError, match="step_size"):
            GradientTDLinearLearner(step_size=0.0)
        with pytest.raises(ValueError, match="secondary_step_size"):
            GradientTDLinearLearner(secondary_step_size=-0.1)
        with pytest.raises(ValueError, match="trace_decay"):
            GradientTDLinearLearner(trace_decay=1.2)
        with pytest.raises(ValueError, match="ratio_clip"):
            GradientTDLinearLearner(ratio_clip=0.0)

    def test_config_roundtrip_and_exports(self) -> None:
        original = GradientTDLinearLearner(
            step_size=0.02,
            secondary_step_size=0.03,
            trace_decay=0.4,
            ratio_clip=2.0,
        )
        config = original.to_config()
        restored = GradientTDLinearLearner.from_config(config)
        assert restored.to_config() == config

    def test_update_shapes_and_secondary_weights_change(self) -> None:
        learner = GradientTDLinearLearner(
            step_size=0.01,
            secondary_step_size=0.05,
            trace_decay=0.2,
            ratio_clip=2.0,
        )
        state = learner.init(3)
        result = learner.update(
            state,
            jnp.array([1.0, 0.0, -1.0], dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
            jnp.array([0.0, 1.0, 0.5], dtype=jnp.float32),
            jnp.array(0.9, dtype=jnp.float32),
            jnp.array(3.0, dtype=jnp.float32),
        )

        assert isinstance(result, GradientTDUpdateResult)
        chex.assert_shape(result.state.weights, (4,))
        chex.assert_shape(result.state.secondary_weights, (4,))
        chex.assert_shape(result.metrics, (6,))
        assert float(result.rho_clipped) == pytest.approx(2.0)
        assert float(jnp.linalg.norm(result.state.secondary_weights)) > 0.0
        chex.assert_tree_all_finite(result.state)

    def test_scan_off_policy_positive_control(self) -> None:
        rng = np.random.default_rng(0)
        steps = 600
        actions = rng.integers(0, 2, size=steps)
        observations = jnp.ones((steps, 1), dtype=jnp.float32)
        next_observations = jnp.ones((steps, 1), dtype=jnp.float32)
        rewards = jnp.asarray((actions == 1).astype(np.float32))
        rhos = jnp.asarray(np.where(actions == 1, 2.0, 0.0).astype(np.float32))
        gammas = jnp.zeros((steps,), dtype=jnp.float32)

        learner = GradientTDLinearLearner(
            step_size=0.02,
            secondary_step_size=0.05,
            ratio_clip=10.0,
        )
        result = run_gradient_td_learning_loop(
            learner,
            learner.init(1),
            observations,
            rewards,
            next_observations,
            gammas,
            rhos,
        )
        pred = float(learner.predict(result.state, jnp.ones(1))[0])

        assert pred > 0.95
        chex.assert_shape(result.metrics, (steps, 6))
        chex.assert_tree_all_finite(result.state)


# =============================================================================
# rho clipping
# =============================================================================


class TestRetraceClip:
    def test_clip_at_one(self) -> None:
        learner = OffPolicyTDLinearLearner(retrace_clip=1.0)
        state = learner.init(3)
        # rho >> 1 should be clipped to 1
        result = learner.update(
            state,
            jnp.array([1.0, 0.0, 0.0]),
            jnp.float32(1.0),
            jnp.zeros(3),
            jnp.float32(0.0),
            jnp.float32(5.0),
        )
        assert float(result.rho_clipped) == 1.0

    def test_no_clip_when_below_threshold(self) -> None:
        learner = OffPolicyTDLinearLearner(retrace_clip=1.0)
        state = learner.init(3)
        result = learner.update(
            state,
            jnp.array([1.0, 0.0, 0.0]),
            jnp.float32(1.0),
            jnp.zeros(3),
            jnp.float32(0.0),
            jnp.float32(0.5),
        )
        assert float(result.rho_clipped) == pytest.approx(0.5)

    def test_inf_clip_disables(self) -> None:
        learner = OffPolicyTDLinearLearner(retrace_clip=float("inf"))
        state = learner.init(3)
        # Large rho stays unclipped
        result = learner.update(
            state,
            jnp.array([1.0, 0.0, 0.0]),
            jnp.float32(1.0),
            jnp.zeros(3),
            jnp.float32(0.0),
            jnp.float32(7.0),
        )
        assert float(result.rho_clipped) == pytest.approx(7.0)


# =============================================================================
# Off-policy convergence on a small chain
# =============================================================================


class TestOffPolicyConvergence:
    """Small bandit-with-state: 2 actions, 4 states.
    Behavior policy uniform; target policy always picks action 0.
    Reward depends on state and action.
    """

    @staticmethod
    def _generate_episode(
        rng: np.random.Generator,
        n_states: int = 4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Random walk where each step picks left or right uniformly.
        Reward 1 when reaching the right end; 0 otherwise.
        """
        eye = np.eye(n_states, dtype=np.float32)
        state = n_states // 2
        obs_list, next_obs_list, rew_list, gam_list, action_list = (
            [],
            [],
            [],
            [],
            [],
        )
        while True:
            obs = eye[state]
            action = int(rng.integers(0, 2))
            new_state = state - 1 if action == 0 else state + 1
            if new_state < 0:
                rew = 0.0
                term = True
            elif new_state >= n_states:
                rew = 1.0
                term = True
            else:
                rew = 0.0
                term = False

            obs_list.append(obs)
            action_list.append(action)
            rew_list.append(rew)
            if term:
                next_obs_list.append(np.zeros(n_states, dtype=np.float32))
                gam_list.append(0.0)
                break
            else:
                next_obs_list.append(eye[new_state])
                gam_list.append(1.0)
                state = new_state

        return (
            np.asarray(obs_list),
            np.asarray(rew_list),
            np.asarray(next_obs_list),
            np.asarray(gam_list),
            np.asarray(action_list),
        )

    def test_off_policy_converges_to_target_v(self) -> None:
        """Behavior: uniform random. Target: always go right.
        Under target policy, V(s_i) = 1.0 for every state (always reach
        right end). Off-policy TD with IS should converge there.
        """
        n_states = 4

        # rho_t = pi(a_t|s) / b(a_t|s)
        # target policy: always go right (action 1) => pi(1|s)=1, pi(0|s)=0
        # behavior:      uniform =>                     b(0|s)=b(1|s)=0.5
        # rho(action=1) = 1 / 0.5 = 2
        # rho(action=0) = 0 / 0.5 = 0  (the trajectory contributes nothing)

        learner = OffPolicyTDLinearLearner(step_size=0.05, trace_decay=0.0, retrace_clip=2.0)
        state = learner.init(n_states)

        rng = np.random.default_rng(42)
        for _ in range(2000):
            obs, rew, nxt, gam, actions = self._generate_episode(rng, n_states)
            for t in range(len(rew)):
                # Importance ratio per the policy definitions above
                rho = 2.0 if actions[t] == 1 else 0.0
                res = learner.update(
                    state,
                    jnp.asarray(obs[t]),
                    jnp.asarray(rew[t]),
                    jnp.asarray(nxt[t]),
                    jnp.asarray(gam[t]),
                    jnp.float32(rho),
                )
                state = res.state

        # Under target policy (always right), V(s_i) = 1 for all states
        eye = np.eye(n_states, dtype=np.float32)
        v_estimated = np.array(
            [
                float(jnp.dot(state.weights, jnp.asarray(eye[s])) + state.bias)
                for s in range(n_states)
            ]
        )
        # Target V per state
        v_true = np.ones(n_states, dtype=np.float32)
        rmse = float(np.sqrt(np.mean((v_estimated - v_true) ** 2)))
        assert rmse < 0.20, f"Off-policy TD did not converge: V_est={v_estimated}, RMSE={rmse}"

    def test_naive_is_finite_with_clipping(self) -> None:
        """Even with a high IS-ratio target/behavior mismatch, clipping at
        c=1 should keep weights finite over many steps."""
        learner = OffPolicyTDLinearLearner(step_size=0.01, trace_decay=0.7, retrace_clip=1.0)
        state = learner.init(5)

        rng = np.random.default_rng(7)
        for _ in range(2000):
            phi = jnp.asarray(rng.normal(size=5).astype(np.float32))
            phi_next = jnp.asarray(rng.normal(size=5).astype(np.float32))
            r = jnp.float32(rng.normal())
            # Wildly varying rho
            rho = jnp.float32(rng.uniform(0.0, 50.0))
            res = learner.update(state, phi, r, phi_next, jnp.float32(0.95), rho)
            state = res.state

        chex.assert_tree_all_finite(state.weights)
        chex.assert_tree_all_finite(state.eligibility_traces)


# =============================================================================
# JIT / scan
# =============================================================================


class TestJit:
    def test_predict_and_update_jit(self) -> None:
        learner = OffPolicyTDLinearLearner(step_size=0.05, trace_decay=0.5)
        state = learner.init(4)
        # Two calls should not retrace
        v1 = learner.predict(state, jnp.ones(4))
        v2 = learner.predict(state, jnp.ones(4))
        chex.assert_trees_all_close(v1, v2)

        result = learner.update(
            state,
            jnp.ones(4),
            jnp.float32(1.0),
            jnp.ones(4),
            jnp.float32(0.9),
            jnp.float32(1.5),
        )
        assert isinstance(result, OffPolicyTDUpdateResult)


# =============================================================================
# Config roundtrip
# =============================================================================


class TestConfig:
    def test_roundtrip(self) -> None:
        original = OffPolicyTDLinearLearner(step_size=0.07, trace_decay=0.6, retrace_clip=2.5)
        config = original.to_config()
        assert config["type"] == "OffPolicyTDLinearLearner"
        restored = OffPolicyTDLinearLearner.from_config(config)
        assert restored.step_size == 0.07
        assert restored.trace_decay == 0.6
        assert restored.retrace_clip == 2.5


# =============================================================================
# Baird-style: don't diverge with bounded clipping
# =============================================================================


class TestBairdStyle:
    """With Retrace clipping (c=1) on a moderate off-policy problem,
    the algorithm stays finite and bounded. This is NOT a guarantee of
    convergence on Baird's exact counterexample (which requires gradient-
    TD or emphatic methods); it's a sanity check that clipping prevents
    the most pathological IS-variance blowups.
    """

    def test_no_divergence_with_clip(self) -> None:
        learner = OffPolicyTDLinearLearner(step_size=0.01, trace_decay=0.5, retrace_clip=1.0)
        state = learner.init(4)

        rng = np.random.default_rng(11)
        for _ in range(3000):
            phi = jnp.asarray(rng.normal(size=4).astype(np.float32))
            phi_next = jnp.asarray(rng.normal(size=4).astype(np.float32))
            r = jnp.float32(rng.normal())
            # Heavy-tailed rho but mostly modest values
            rho = jnp.float32(rng.exponential(1.0))
            res = learner.update(state, phi, r, phi_next, jnp.float32(0.9), rho)
            state = res.state

        chex.assert_tree_all_finite(state.weights)
        # Without clipping this is unbounded; with c=1 it's bounded.
        assert float(jnp.max(jnp.abs(state.weights))) < 50.0


class TestInfiniteRewardDoesNotPoisonWeights:
    def test_off_policy_td(self) -> None:
        learner = OffPolicyTDLinearLearner(step_size=0.1, trace_decay=0.0)
        state = learner.init(2)
        obs = jnp.array([0.0, 1.0], dtype=jnp.float32)
        nxt = jnp.zeros(2, dtype=jnp.float32)
        gamma = jnp.array(0.0, dtype=jnp.float32)
        rho = jnp.array(1.0, dtype=jnp.float32)

        poisoned = learner.update(
            state, obs, jnp.array(jnp.inf, dtype=jnp.float32), nxt, gamma, rho
        )
        chex.assert_trees_all_close(poisoned.state.weights, state.weights)
        assert int(poisoned.state.step_count) == int(state.step_count)
        assert not bool(poisoned.update_applied)
        assert float(poisoned.td_error) == 0.0
        chex.assert_trees_all_close(poisoned.metrics, jnp.zeros_like(poisoned.metrics))

        recovered = learner.update(
            poisoned.state, obs, jnp.array(1.0, dtype=jnp.float32), nxt, gamma, rho
        )
        chex.assert_tree_all_finite(recovered.state.weights)
        assert int(recovered.state.step_count) == int(state.step_count) + 1
        assert bool(recovered.update_applied)

    def test_etd(self) -> None:
        learner = ETDLinearLearner(step_size=0.1, trace_decay=0.0)
        state = learner.init(2)
        obs = jnp.array([0.0, 1.0], dtype=jnp.float32)
        nxt = jnp.zeros(2, dtype=jnp.float32)
        gamma = jnp.array(0.0, dtype=jnp.float32)
        rho = jnp.array(1.0, dtype=jnp.float32)

        poisoned = learner.update(
            state, obs, jnp.array(jnp.inf, dtype=jnp.float32), nxt, gamma, rho
        )
        chex.assert_trees_all_close(poisoned.state.weights, state.weights)
        assert not bool(poisoned.update_applied)
        assert float(poisoned.td_error) == 0.0

        recovered = learner.update(
            poisoned.state, obs, jnp.array(1.0, dtype=jnp.float32), nxt, gamma, rho
        )
        chex.assert_tree_all_finite(recovered.state.weights)
        assert bool(recovered.update_applied)

    def test_gradient_td(self) -> None:
        learner = GradientTDLinearLearner(step_size=0.1)
        state = learner.init(2)
        obs = jnp.array([0.0, 1.0], dtype=jnp.float32)
        nxt = jnp.zeros(2, dtype=jnp.float32)
        gamma = jnp.array(0.0, dtype=jnp.float32)
        rho = jnp.array(1.0, dtype=jnp.float32)

        poisoned = learner.update(
            state, obs, jnp.array(jnp.inf, dtype=jnp.float32), nxt, gamma, rho
        )
        chex.assert_trees_all_close(poisoned.state.weights, state.weights)
        chex.assert_trees_all_close(poisoned.state.secondary_weights, state.secondary_weights)
        assert not bool(poisoned.update_applied)
        assert float(poisoned.td_error) == 0.0

        recovered = learner.update(
            poisoned.state, obs, jnp.array(1.0, dtype=jnp.float32), nxt, gamma, rho
        )
        chex.assert_tree_all_finite(recovered.state.weights)
        chex.assert_tree_all_finite(recovered.state.secondary_weights)
        assert bool(recovered.update_applied)


class TestZeroGammaDoesNotMultiplyInfBootstrap:
    def test_off_policy_td(self) -> None:
        learner = OffPolicyTDLinearLearner(step_size=0.1, trace_decay=0.0)
        huge = jnp.float32(1e38)
        state = learner.init(2).replace(  # type: ignore[attr-defined]
            weights=jnp.array([huge, 0.0], dtype=jnp.float32)
        )
        obs = jnp.array([0.0, 1.0], dtype=jnp.float32)
        nxt = jnp.array([huge, 0.0], dtype=jnp.float32)
        raw = jnp.asarray(0.0, dtype=jnp.float32) * (huge * huge)
        assert not bool(jnp.isfinite(raw))

        result = learner.update(
            state,
            obs,
            jnp.array(3.0, dtype=jnp.float32),
            nxt,
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        chex.assert_trees_all_close(result.td_error, jnp.array(3.0, dtype=jnp.float32))
        chex.assert_tree_all_finite(result.state.weights)

    def test_gradient_td_zero_gamma_branches_before_overflowing_correction(self) -> None:
        learner = GradientTDLinearLearner(step_size=0.1, secondary_step_size=0.01)
        state = learner.init(4).replace(  # type: ignore[attr-defined]
            secondary_weights=jnp.ones((5,), dtype=jnp.float32)
        )
        observation = jnp.ones((4,), dtype=jnp.float32)
        rho = jnp.array(1e38, dtype=jnp.float32)
        traces = rho * jnp.ones((5,), dtype=jnp.float32)
        assert not bool(jnp.isfinite(jnp.dot(state.secondary_weights, traces)))

        result = learner.update(
            state,
            observation,
            jnp.array(1.0, dtype=jnp.float32),
            jnp.zeros((4,), dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
            rho,
        )

        assert bool(result.update_applied)
        chex.assert_tree_all_finite(result.state)

    def test_etd(self) -> None:
        learner = ETDLinearLearner(step_size=0.1, trace_decay=0.0)
        huge = jnp.float32(1e38)
        state = learner.init(2).replace(  # type: ignore[attr-defined]
            weights=jnp.array([huge, 0.0], dtype=jnp.float32)
        )
        result = learner.update(
            state,
            jnp.array([0.0, 1.0], dtype=jnp.float32),
            jnp.array(3.0, dtype=jnp.float32),
            jnp.array([huge, 0.0], dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        chex.assert_trees_all_close(result.td_error, jnp.array(3.0, dtype=jnp.float32))
        chex.assert_tree_all_finite(result.state.weights)

    def test_gradient_td(self) -> None:
        learner = GradientTDLinearLearner(step_size=0.1, secondary_step_size=0.01)
        huge = jnp.float32(1e38)
        state = learner.init(2).replace(  # type: ignore[attr-defined]
            weights=jnp.array([huge, 0.0, 0.0], dtype=jnp.float32)
        )
        result = learner.update(
            state,
            jnp.array([0.0, 1.0], dtype=jnp.float32),
            jnp.array(3.0, dtype=jnp.float32),
            jnp.array([huge, 0.0], dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        chex.assert_trees_all_close(result.td_error, jnp.array(3.0, dtype=jnp.float32))
        chex.assert_tree_all_finite(result.state.weights)

    def test_off_policy_td_does_not_multiply_inf_traces(self) -> None:
        """gamma*lam=0 drops leftover traces; 0 * inf must not freeze."""
        learner = OffPolicyTDLinearLearner(step_size=0.1, trace_decay=0.9)
        state = learner.init(2).replace(  # type: ignore[attr-defined]
            eligibility_traces=jnp.full(2, jnp.inf, dtype=jnp.float32),
            bias_eligibility_trace=jnp.asarray(jnp.inf, dtype=jnp.float32),
        )
        raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
        assert not bool(jnp.isfinite(raw))

        result = learner.update(
            state,
            jnp.array([0.5, -0.25], dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
            jnp.array([jnp.inf, 0.0], dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        chex.assert_tree_all_finite(result.state.eligibility_traces)

        assert bool(jnp.isfinite(result.state.bias_eligibility_trace))

    def test_etd_does_not_multiply_inf_follow_on(self) -> None:
        """gamma=0 drops leftover F; 0 * inf must not freeze the ETD step."""
        learner = ETDLinearLearner(step_size=0.1, trace_decay=0.4)
        state = learner.init(2).replace(  # type: ignore[attr-defined]
            follow_on_trace=jnp.asarray(jnp.inf, dtype=jnp.float32),
            eligibility_traces=jnp.full(2, jnp.inf, dtype=jnp.float32),
            bias_eligibility_trace=jnp.asarray(jnp.inf, dtype=jnp.float32),
        )
        raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
        assert not bool(jnp.isfinite(raw))

        result = learner.update(
            state,
            jnp.array([0.5, -0.25], dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
            jnp.array([jnp.inf, 0.0], dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        assert bool(jnp.isfinite(result.state.follow_on_trace))
        chex.assert_tree_all_finite(result.state.eligibility_traces)

    def test_gradient_td_does_not_multiply_inf_traces(self) -> None:
        """gamma*lam=0 drops leftover GTD traces; 0 * inf must not freeze."""
        learner = GradientTDLinearLearner(step_size=0.1, trace_decay=0.9)
        state = learner.init(2).replace(  # type: ignore[attr-defined]
            eligibility_traces=jnp.full(3, jnp.inf, dtype=jnp.float32),
        )
        raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
        assert not bool(jnp.isfinite(raw))

        result = learner.update(
            state,
            jnp.array([0.5, -0.25], dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
            jnp.array([jnp.inf, 0.0], dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        chex.assert_tree_all_finite(result.state.eligibility_traces)


@pytest.mark.parametrize(
    "learner",
    [OffPolicyTDLinearLearner(), ETDLinearLearner(), GradientTDLinearLearner()],
)
@pytest.mark.parametrize("value", [True, np.bool_(True), 1.5, "4", object(), 0])
def test_feature_dim_rejects_non_integer_families_without_repr(
    learner: object, value: object
) -> None:
    with pytest.raises(ValueError, match="feature_dim"):
        learner.init(value)  # type: ignore[attr-defined]


@pytest.mark.parametrize("code", ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"))
def test_feature_dim_accepts_and_canonicalizes_numpy_integer_families(code: str) -> None:
    feature_dim = np.dtype(code).type(4)
    assert OffPolicyTDLinearLearner().init(feature_dim).weights.shape == (4,)
    assert ETDLinearLearner().init(feature_dim).weights.shape == (4,)
    assert GradientTDLinearLearner().init(feature_dim).weights.shape == (5,)


def test_init_preflights_state_bytes_and_augmented_width_before_allocation() -> None:
    for learner in (
        OffPolicyTDLinearLearner(),
        ETDLinearLearner(),
        GradientTDLinearLearner(),
    ):
        with pytest.raises(ValueError, match="state_nbytes"):
            learner.init(300_000_000)
    with pytest.raises(ValueError, match="feature_dim"):
        GradientTDLinearLearner().init(2**31 - 1)


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (lambda value: OffPolicyTDLinearLearner(step_size=value), "step_size"),
        (lambda value: ETDLinearLearner(trace_decay=value), "trace_decay"),
        (
            lambda value: GradientTDLinearLearner(secondary_step_size=value),
            "secondary_step_size",
        ),
    ],
)
def test_config_scalars_reject_hostile_and_float32_invalid_values(
    factory: object, field: str
) -> None:
    for value in (float("nan"), 1.0e100, object()):
        with pytest.raises(ValueError, match=field):
            factory(value)  # type: ignore[operator]


def test_config_scalars_canonicalize_reals_and_preserve_infinity_clip_sentinel() -> None:
    off_policy = OffPolicyTDLinearLearner(
        step_size=Fraction(1, 4),
        trace_decay=np.float64(0.5),
        retrace_clip=float("inf"),
    )
    gradient = GradientTDLinearLearner(
        step_size=np.float32(0.25),
        secondary_step_size=Fraction(1, 2),
        trace_decay=np.float64(0.5),
        ratio_clip=np.float64(np.inf),
    )
    assert off_policy.step_size == 0.25
    assert off_policy.trace_decay == 0.5
    assert off_policy.retrace_clip == float("inf")
    assert gradient.step_size == 0.25
    assert gradient.secondary_step_size == 0.5
    assert gradient.trace_decay == 0.5
    assert gradient.ratio_clip == float("inf")
    assert all(
        type(value) is float
        for value in (
            off_policy.step_size,
            off_policy.trace_decay,
            gradient.step_size,
            gradient.secondary_step_size,
            gradient.trace_decay,
        )
    )
    huge_finite = np.longdouble("1e400")
    with pytest.raises(ValueError, match="retrace_clip"):
        OffPolicyTDLinearLearner(retrace_clip=huge_finite)
    with pytest.raises(ValueError, match="ratio_clip"):
        GradientTDLinearLearner(ratio_clip=huge_finite)


def test_public_boundaries_validate_host_metadata_before_jax_conversion() -> None:
    class HostileVector:
        shape = (2,)
        dtype = np.dtype(np.float64)

        def __jax_array__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("conversion must not run")

    for learner in (OffPolicyTDLinearLearner(), ETDLinearLearner()):
        state = learner.init(2)
        with pytest.raises(ValueError, match="observation"):
            learner.predict(state, HostileVector())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="observation"):
            learner.update(  # type: ignore[attr-defined]
                state,
                HostileVector(),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.zeros(2, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(1.0, dtype=jnp.float32),
            )
    gradient = GradientTDLinearLearner()
    with pytest.raises(ValueError, match="observation"):
        gradient.predict(gradient.init(2), HostileVector())  # type: ignore[arg-type]


def test_config_real_subclasses_are_rejected_without_running_hooks() -> None:
    calls = 0

    class HostileFraction(Fraction):
        def as_integer_ratio(self) -> tuple[int, int]:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile ratio hook ran")

    for factory in (
        lambda: OffPolicyTDLinearLearner(step_size=HostileFraction(1, 4)),
        lambda: ETDLinearLearner(trace_decay=HostileFraction(1, 2)),
        lambda: GradientTDLinearLearner(ratio_clip=HostileFraction(2, 1)),
    ):
        with pytest.raises(ValueError):
            factory()
    assert calls == 0


def test_state_metadata_is_hostile_safe_and_counters_saturate() -> None:
    class HostileLeaf:
        @property
        def shape(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("shape hook")

        def __repr__(self) -> str:
            raise AssertionError("repr must not run")

    off_policy = OffPolicyTDLinearLearner()
    malformed = off_policy.init(2).replace(weights=HostileLeaf())
    with pytest.raises(ValueError, match="state.weights"):
        off_policy.predict(malformed, jnp.zeros(2, dtype=jnp.float32))

    maximum = jnp.asarray(2**31 - 1, dtype=jnp.int32)
    for learner in (off_policy, ETDLinearLearner(), GradientTDLinearLearner()):
        state = learner.init(2).replace(step_count=maximum)
        result = learner.update(
            state,
            jnp.zeros(2, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.zeros(2, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        assert int(result.state.step_count) == 2**31 - 1


def test_diagnostics_are_finite_and_transactional_at_float32_extremes() -> None:
    learner = GradientTDLinearLearner(step_size=0.01, secondary_step_size=0.0)
    state = learner.init(2).replace(
        weights=jnp.full(3, jnp.finfo(jnp.float32).max, dtype=jnp.float32),
        secondary_weights=jnp.full(3, jnp.finfo(jnp.float32).max, dtype=jnp.float32),
    )
    accepted = learner.update(
        state,
        jnp.zeros(2, dtype=jnp.float32),
        jnp.asarray(jnp.finfo(jnp.float32).max, dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert bool(accepted.update_applied)
    chex.assert_tree_all_finite(accepted.metrics)

    off_policy = OffPolicyTDLinearLearner(step_size=1e-6)
    initial = off_policy.init(2)
    rejected = off_policy.update(
        initial,
        jnp.zeros(2, dtype=jnp.float32),
        jnp.asarray(1e20, dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, initial)
    chex.assert_trees_all_equal(rejected.metrics, jnp.zeros(5, dtype=jnp.float32))


def test_gradient_scan_preflights_host_shapes_and_aggregate_resources() -> None:
    learner = GradientTDLinearLearner()
    state = learner.init(2)

    class Oversized:
        dtype = np.dtype(np.float32)

        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def __jax_array__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("conversion must not run")

    steps = 100_000_000
    with pytest.raises(ValueError, match="aggregate resources"):
        run_gradient_td_learning_loop(
            learner,
            state,
            Oversized((steps, 2)),  # type: ignore[arg-type]
            Oversized((steps,)),  # type: ignore[arg-type]
            Oversized((steps, 2)),  # type: ignore[arg-type]
            Oversized((steps,)),  # type: ignore[arg-type]
            Oversized((steps,)),  # type: ignore[arg-type]
        )
