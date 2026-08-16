"""Baird counterexample suite for the linear off-policy learners in ``off_policy_td.py``.

Canonical 7-state Baird star (Baird 1995; Sutton & Barto 2018, section 11.2):

- Six "upper" states (0..5) and one "lower" state (6). Two actions: **dashed**
  moves to one of the six upper states uniformly; **solid** moves to state 6.
- Behavior policy takes dashed with probability 6/7 and solid with 1/7, so the
  next-state distribution (and the stationary distribution) is uniform over all
  seven states. The target policy always takes solid, giving importance ratios
  rho = 0 on dashed transitions and rho = 7 on solid ones.
- All rewards are zero and gamma = 0.99, so the true value function is zero.
- Classic 8-feature representation: phi(s) = 2 e_s + e_7 for upper states,
  phi(6) = e_6 + 2 e_7; classic initial weights (1, 1, 1, 1, 1, 1, 10, 1).

Under this setup semi-gradient off-policy TD diverges, while Gradient-TD/TDC
and (in expectation) Emphatic TD converge — the properties asserted here.

Because the 7x8 feature matrix has full row rank, every value function over the
seven states is representable, the MSPBE projection is the identity, and
``MSPBE(w) = mean_s (gamma * V(6) - V(s))**2`` under the uniform distribution.
MSPBE = 0 iff V is identically zero.

Note: horde-level Baird coverage (``core/off_policy_horde.py``) is deliberately
deferred — that module is being repaired concurrently; this suite exercises the
linear learners in ``core/off_policy_td.py`` only.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from alberta_framework.core.off_policy_td import (
    ETDLinearLearner,
    ETDState,
    GradientTDLinearLearner,
    OffPolicyTDLinearLearner,
    OffPolicyTDState,
)

# =============================================================================
# Baird star: hand-computed dynamics and representation
# =============================================================================

GAMMA = 0.99
N_STATES = 7
N_FEATURES = 8
LOWER = 6  # index of the lower ("solid") state
RHO_SOLID = 7.0  # pi(solid|s) / b(solid|s) = 1 / (1/7)
RHO_DASHED = 0.0  # pi(dashed|s) / b(dashed|s) = 0 / (6/7)


def _baird_features() -> np.ndarray:
    """The classic 8-feature Baird star representation, one row per state."""
    phi = np.zeros((N_STATES, N_FEATURES), dtype=np.float32)
    for s in range(6):
        phi[s, s] = 2.0
        phi[s, N_FEATURES - 1] = 1.0
    phi[LOWER, LOWER] = 1.0
    phi[LOWER, N_FEATURES - 1] = 2.0
    return phi


PHI = _baird_features()
W_INIT = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=np.float32)


def _sample_trajectory(n_steps: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a behavior-policy trajectory; returns (obs, next_obs, rhos)."""
    rng = np.random.default_rng(seed)
    states = np.zeros(n_steps + 1, dtype=np.int64)
    rhos = np.zeros(n_steps, dtype=np.float32)
    states[0] = LOWER
    for t in range(n_steps):
        if rng.random() < 6.0 / 7.0:
            states[t + 1] = rng.integers(0, 6)  # dashed: uniform over upper states
            rhos[t] = RHO_DASHED
        else:
            states[t + 1] = LOWER  # solid: always to the lower state
            rhos[t] = RHO_SOLID
    return PHI[states[:-1]], PHI[states[1:]], rhos


def _scan_transitions(
    learner: OffPolicyTDLinearLearner | GradientTDLinearLearner,
    state: chex.ArrayTree,
    obs: np.ndarray,
    next_obs: np.ndarray,
    rhos: np.ndarray,
) -> tuple[chex.ArrayTree, np.ndarray]:
    """Run ``learner.update`` over the trajectory; returns (state, per-step ||w||)."""

    def step(carry: chex.ArrayTree, inputs: tuple[Array, Array, Array]) -> tuple:
        o, no, rho = inputs
        result = learner.update(carry, o, jnp.float32(0.0), no, jnp.float32(GAMMA), rho)
        return result.state, jnp.linalg.norm(result.state.weights)

    final_state, norms = jax.lax.scan(
        step, state, (jnp.asarray(obs), jnp.asarray(next_obs), jnp.asarray(rhos))
    )
    return final_state, np.asarray(norms)


def _values_linear(state: OffPolicyTDState | ETDState) -> np.ndarray:
    """V(s) for the weights+bias learners, evaluated at every star state."""
    return np.asarray(jnp.asarray(PHI) @ state.weights + state.bias)


def _mspbe(values: np.ndarray) -> float:
    """MSPBE under the uniform distribution (projection is the identity here)."""
    return float(np.mean((GAMMA * values[LOWER] - values) ** 2))


# =============================================================================
# (1) Semi-gradient off-policy TD diverges
# =============================================================================


class TestSemiGradientDivergence:
    def test_unclipped_is_td_diverges(self) -> None:
        """Per-decision IS TD(0) with no ratio clipping is Baird's counterexample:
        the weights grow without bound (alpha = 0.01 as in Sutton & Barto fig 11.2)."""
        learner = OffPolicyTDLinearLearner(
            step_size=0.01, trace_decay=0.0, retrace_clip=float("inf")
        )
        state = learner.init(N_FEATURES).replace(weights=jnp.asarray(W_INIT))
        obs, next_obs, rhos = _sample_trajectory(4000, seed=0)

        final_state, norms = _scan_transitions(learner, state, obs, next_obs, rhos)

        # Monotone explosive growth across 1000-step windows...
        checkpoints = norms[999::1000]
        previous = float(jnp.linalg.norm(state.weights))
        for norm in checkpoints:
            assert norm > 2.0 * previous, f"weight norm not exploding: {checkpoints}"
            previous = float(norm)
        # ...ending far above any bounded iterate (initial norm is ~10.3).
        chex.assert_tree_all_finite(final_state.weights)
        assert float(norms[-1]) > 1e4


# =============================================================================
# (2) Gradient-TD / TDC converges
# =============================================================================


class TestGradientTDConvergence:
    def test_tdc_drives_mspbe_to_near_zero(self) -> None:
        """TDC on the same off-policy stream keeps the weights bounded and drives
        the MSPBE from ~68 to ~5e-4 (alpha=0.005, beta=0.05, S&B fig 11.5).

        The MSPBE-flat direction (all values equal, curvature ~(1-gamma)^2) decays
        too slowly to assert V -> 0 within a CI budget, so MSPBE is the criterion.
        """
        learner = GradientTDLinearLearner(
            step_size=0.005, secondary_step_size=0.05, trace_decay=0.0, ratio_clip=10.0
        )
        # GradientTDState appends a bias feature; give it the classic init + zero bias.
        w0 = jnp.asarray(np.concatenate([W_INIT, [0.0]]).astype(np.float32))
        state = learner.init(N_FEATURES).replace(weights=w0)

        def values(s: chex.ArrayTree) -> np.ndarray:
            return np.array(
                [float(learner.predict(s, jnp.asarray(PHI[i]))[0]) for i in range(N_STATES)]
            )

        initial_mspbe = _mspbe(values(state))
        assert initial_mspbe > 10.0  # sanity: the classic init starts far from the fix-point

        obs, next_obs, rhos = _sample_trajectory(60_000, seed=1)
        final_state, norms = _scan_transitions(learner, state, obs, next_obs, rhos)

        chex.assert_tree_all_finite(final_state)
        assert float(np.max(norms)) < 20.0, "TDC iterates must stay bounded on Baird"
        final_mspbe = _mspbe(values(final_state))
        assert final_mspbe < 5e-3, f"TDC failed to reduce MSPBE: {final_mspbe}"
        assert final_mspbe < 1e-3 * initial_mspbe


# =============================================================================
# (3) Emphatic TD converges (in expectation)
# =============================================================================


class TestETDConvergence:
    def test_expected_emphatic_updates_converge_to_zero_values(self) -> None:
        """One-step ETD applied in expectation converges on Baird (S&B fig 11.6).

        Sampled ETD has unbounded follow-on variance here (E[(rho*gamma)^2] =
        7 * gamma^2 ~ 6.9 > 1), so — exactly as in the textbook figure — the
        expected update is used. The expected emphatic weighting solves
        m = d + gamma * P_pi^T m with d uniform: since every state maps to the
        lower state under the target policy, sum(m) = 1/(1-gamma) and the
        per-state emphasis is M(s) = 7 m(s), i.e. M(upper) = 1 and
        M(lower) = 1 + 7 * gamma / (1 - gamma).

        Each sweep applies one ``update`` per state with the target transition
        (s -> lower, r = 0, rho = 1 folds in E_b[rho * delta] = delta_pi) and the
        follow-on trace injected so the learner's own recursion
        F = rho * gamma * F_prev + i reproduces M(s).
        """
        learner = ETDLinearLearner(step_size=5e-4, trace_decay=0.0)
        state = learner.init(N_FEATURES).replace(weights=jnp.asarray(W_INIT))

        emphasis = np.ones(N_STATES, dtype=np.float32)
        emphasis[LOWER] = 1.0 + 7.0 * GAMMA / (1.0 - GAMMA)
        follow_on_inject = jnp.asarray((emphasis - 1.0) / GAMMA)
        obs_sweep = jnp.asarray(PHI)
        next_sweep = jnp.asarray(np.tile(PHI[LOWER], (N_STATES, 1)))

        def sweep(carry: ETDState, _: Array) -> tuple[ETDState, Array]:
            def one_state(inner: ETDState, inputs: tuple[Array, Array, Array]) -> tuple:
                o, no, f_prev = inputs
                result = learner.update(
                    inner.replace(follow_on_trace=f_prev),
                    o,
                    jnp.float32(0.0),
                    no,
                    jnp.float32(GAMMA),
                    jnp.float32(1.0),
                )
                return result.state, ()

            new_state, _ = jax.lax.scan(
                one_state, carry, (obs_sweep, next_sweep, follow_on_inject)
            )
            return new_state, jnp.linalg.norm(new_state.weights)

        final_state, norms = jax.lax.scan(sweep, state, jnp.arange(5000))

        chex.assert_tree_all_finite(final_state.weights)
        assert float(np.max(np.asarray(norms))) < 15.0
        values = _values_linear(final_state)
        # Unlike TDC, the emphatic weighting pulls the whole value function to the
        # true v = 0 (weight ~694 on the lower state kills the flat direction).
        assert float(np.max(np.abs(values))) < 1e-2, f"ETD values not at zero: {values}"
        assert _mspbe(values) < 1e-6


# =============================================================================
# (4) rho = 0 transitions leave primary weights invariant
# =============================================================================


class TestRhoZeroInvariance:
    """A dashed transition (rho = 0) must not move the primary weights: every
    method multiplies its weight update by rho (directly or through the trace)."""

    @pytest.mark.parametrize("trace_decay", [0.0, 0.7])
    def test_off_policy_td_weights_and_bias_invariant(self, trace_decay: float) -> None:
        learner = OffPolicyTDLinearLearner(
            step_size=0.1, trace_decay=trace_decay, retrace_clip=float("inf")
        )
        state = learner.init(N_FEATURES).replace(
            weights=jnp.asarray(W_INIT), bias=jnp.float32(0.5)
        )
        result = learner.update(
            state,
            jnp.asarray(PHI[3]),
            jnp.float32(1.3),
            jnp.asarray(PHI[2]),
            jnp.float32(GAMMA),
            jnp.float32(RHO_DASHED),
        )
        chex.assert_trees_all_equal(result.state.weights, state.weights)
        chex.assert_trees_all_equal(result.state.bias, state.bias)
        # The canonical per-decision trace z = rho * (...) vanishes.
        chex.assert_trees_all_close(result.state.eligibility_traces, jnp.zeros(N_FEATURES))

    def test_etd_weights_invariant_and_follow_on_resets(self) -> None:
        learner = ETDLinearLearner(step_size=0.1, trace_decay=0.0)
        state = learner.init(N_FEATURES).replace(
            weights=jnp.asarray(W_INIT),
            bias=jnp.float32(0.5),
            follow_on_trace=jnp.float32(3.0),
        )
        result = learner.update(
            state,
            jnp.asarray(PHI[0]),
            jnp.float32(1.3),
            jnp.asarray(PHI[5]),
            jnp.float32(GAMMA),
            jnp.float32(RHO_DASHED),
        )
        chex.assert_trees_all_equal(result.state.weights, state.weights)
        chex.assert_trees_all_equal(result.state.bias, state.bias)
        # The follow-on recursion F = rho * gamma * F_prev + i resets to interest.
        chex.assert_trees_all_close(result.state.follow_on_trace, jnp.float32(1.0))
        # e = rho * (...) vanishes, so nothing can leak into the next update.
        chex.assert_trees_all_close(result.state.eligibility_traces, jnp.zeros(N_FEATURES))

    def test_gradient_td_primary_weights_invariant(self) -> None:
        learner = GradientTDLinearLearner(
            step_size=0.05, secondary_step_size=0.1, trace_decay=0.0, ratio_clip=10.0
        )
        w0 = jnp.asarray(np.concatenate([W_INIT, [0.0]]).astype(np.float32))
        state = learner.init(N_FEATURES).replace(weights=w0)
        # Warm up with one solid transition so the secondary weights are nonzero.
        warm = learner.update(
            state,
            jnp.asarray(PHI[1]),
            jnp.float32(0.0),
            jnp.asarray(PHI[LOWER]),
            jnp.float32(GAMMA),
            jnp.float32(RHO_SOLID),
        ).state
        assert float(jnp.linalg.norm(warm.secondary_weights)) > 0.0

        result = learner.update(
            warm,
            jnp.asarray(PHI[1]),
            jnp.float32(1.3),
            jnp.asarray(PHI[4]),
            jnp.float32(GAMMA),
            jnp.float32(RHO_DASHED),
        )
        # rho = 0 zeroes the trace, so the primary update vanishes exactly; the
        # secondary weights still take their prescribed -(h . phi) phi decay step.
        chex.assert_trees_all_equal(result.state.weights, warm.weights)
        chex.assert_trees_all_close(result.state.eligibility_traces, jnp.zeros(N_FEATURES + 1))
        assert (
            float(jnp.linalg.norm(result.state.secondary_weights - warm.secondary_weights)) > 0.0
        )
