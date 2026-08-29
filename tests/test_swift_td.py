"""Tests for the SwiftTD optimizer.

SwiftTD (Javed, Sharifnassab & Sutton, RLC 2024) combines True Online
TD(lambda) with per-feature step-size optimization, an overshoot bound on
the update to the eligibility vector, and step-size decay.

Covers: exact trajectory equivalence with the author's dense reference
implementation (``SwiftTDNonSparse``), the three headline behaviors
(step-size optimization, overshoot bounding, step-size decay), and
learning quality/robustness on ``XDistShiftStream`` -- the input-scale
shift stream on which the repo's retired Step 1 development lane recorded
fixed and IDBD-style step-sizes diverging to NaN.
"""

import functools
import math

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core import (
    SwiftTD,
    SwiftTDState,
    TDLinearLearner,
    optimizer_from_config,
)
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.swift_td import _require_float32_resource
from alberta_framework.streams.alberta_plan_step1 import XDistShiftStream

# =============================================================================
# Reference implementation (line-by-line port of SwiftTDNonSparse::Step
# from https://github.com/khurramjaved96/SwiftTD, src/cpp/SwiftTD.cpp)
# =============================================================================


class _RefSwiftTDNonSparse:
    """Faithful float64 port of the author's dense C++ reference."""

    def __init__(self, n, lam, alpha, gamma, eta, decay, meta_step_size, eta_min):
        self.gamma = gamma
        self.w = np.zeros(n)
        self.z = np.zeros(n)
        self.z_delta = np.zeros(n)
        self.delta_w = np.zeros(n)
        self.h = np.zeros(n)
        self.h_old = np.zeros(n)
        self.h_temp = np.zeros(n)
        self.beta = np.full(n, np.log(alpha))
        self.z_bar = np.zeros(n)
        self.p = np.zeros(n)
        self.v_old = 0.0
        self.lam = lam
        self.v_delta = 0.0
        self.eta = eta
        self.eta_min = eta_min
        self.decay = decay
        self.meta_step_size = meta_step_size

    def step(self, features, reward):
        n = len(features)
        v = float(np.dot(self.w, features))
        delta = reward + self.gamma * v - self.v_old
        for i in range(n):
            self.delta_w[i] = delta * self.z[i] - self.z_delta[i] * self.v_delta
            self.w[i] += self.delta_w[i]
            self.beta[i] += (
                self.meta_step_size
                / np.exp(self.beta[i])
                * (delta - self.v_delta)
                * self.p[i]
            )
            if np.exp(self.beta[i]) > self.eta:
                self.beta[i] = np.log(self.eta)
            if np.exp(self.beta[i]) < self.eta_min:
                self.beta[i] = np.log(self.eta_min)
            self.h_old[i] = self.h[i]
            self.h[i] = (
                self.h_temp[i] + delta * self.z_bar[i] - self.z_delta[i] * self.v_delta
            )
            self.h_temp[i] = self.h[i]
            self.z_delta[i] = 0.0
            self.z[i] *= self.gamma * self.lam
            self.p[i] *= self.gamma * self.lam
            self.z_bar[i] *= self.gamma * self.lam
        self.v_delta = 0.0
        tau = 0.0
        for i in range(n):
            tau += np.exp(self.beta[i]) * features[i] * features[i]
        b = 0.0
        for i in range(n):
            b += self.z[i] * features[i]
        for i in range(n):
            self.v_delta += self.delta_w[i] * features[i]
            multiplier = min(1.0, self.eta / tau)
            self.z_delta[i] = multiplier * np.exp(self.beta[i]) * features[i]
            self.z[i] += self.z_delta[i] * (1.0 - b)
            self.p[i] += self.h_old[i] * features[i]
            self.z_bar[i] += self.z_delta[i] * (1.0 - b - self.z_bar[i] * features[i])
            self.h_temp[i] = (
                self.h[i]
                - self.h_old[i] * features[i] * (self.z[i] - self.z_delta[i])
                - self.h[i] * self.z_delta[i] * features[i]
            )
            if tau > self.eta:
                self.h_temp[i] = 0.0
                self.h[i] = 0.0
                self.h_old[i] = 0.0
                self.z_bar[i] = 0.0
                self.beta[i] += np.log(self.decay) * features[i] * features[i]
        self.v_old = v
        return v


def _supervised_update(optimizer, opt_state, weights, bias, observation, target):
    """One SwiftTD update in the supervised limit (gamma = 0)."""
    prediction = jnp.dot(weights, observation) + bias
    td_error = jnp.squeeze(target) - prediction
    upd = optimizer.update(opt_state, td_error, observation, observation, jnp.array(0.0))
    return upd, weights + upd.weight_delta, bias + upd.bias_delta, td_error


# =============================================================================
# Init and config
# =============================================================================


class TestSwiftTDInit:
    """Initialization and hyperparameter validation."""

    def test_init_creates_correct_state(self):
        """Arrays get feature_dim + 1 entries (bias last), traces start at zero."""
        optimizer = SwiftTD(
            initial_step_size=0.01, meta_step_size=0.001, trace_decay=0.9, eta=0.1
        )
        state = optimizer.init(feature_dim=10)

        chex.assert_shape(state.log_step_sizes, (11,))
        chex.assert_shape(state.eligibility_traces, (11,))
        chex.assert_shape(state.prev_weight_update, (11,))
        chex.assert_trees_all_close(jnp.exp(state.log_step_sizes), jnp.full(11, 0.01))
        chex.assert_trees_all_close(state.eligibility_traces, jnp.zeros(11))
        chex.assert_trees_all_close(state.z_bar_traces, jnp.zeros(11))
        chex.assert_trees_all_close(state.p_traces, jnp.zeros(11))
        chex.assert_trees_all_close(state.h_traces, jnp.zeros(11))
        assert state.meta_step_size == pytest.approx(0.001)
        assert state.trace_decay == pytest.approx(0.9)
        assert state.eta == pytest.approx(0.1)

    def test_init_clips_initial_step_size_to_eta(self):
        """Following the dense reference, step-sizes are capped at eta from step 0."""
        optimizer = SwiftTD(initial_step_size=1.0, eta=0.1)
        state = optimizer.init(feature_dim=3)
        chex.assert_trees_all_close(jnp.exp(state.log_step_sizes), jnp.full(4, 0.1))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("initial_step_size", 0.0),
            ("initial_step_size", math.nan),
            ("initial_step_size", math.inf),
            ("initial_step_size", -math.inf),
            ("initial_step_size", True),
            ("eta", -1.0),
            ("eta", math.nan),
            ("eta", math.inf),
            ("eta_min", math.nan),
            ("eta_min", math.inf),
            ("meta_step_size", -1.0),
            ("meta_step_size", math.nan),
            ("meta_step_size", math.inf),
            ("step_size_decay", 0.0),
            ("trace_decay", 1.5),
        ],
    )
    def test_invalid_hyperparameters_raise(self, field: str, value: object):
        with pytest.raises(ValueError, match=field):
            SwiftTD(**{field: value})  # type: ignore[arg-type]

    def test_rejects_float_subclasses_without_calling_numeric_or_repr_hooks(self):
        class HostileFloat(float):
            def __float__(self):
                raise AssertionError("numeric hook executed")

            def __repr__(self):
                raise AssertionError("repr hook executed")

        with pytest.raises(ValueError, match="eta"):
            SwiftTD(eta=HostileFloat(0.1))

    def test_numpy_scalars_are_canonicalized_to_builtin_floats(self):
        config = SwiftTD(
            initial_step_size=np.float32(0.01),
            meta_step_size=np.float64(0.001),
            trace_decay=np.float16(0.5),
            eta=np.int64(1),
        ).to_config()

        assert all(type(value) is float for key, value in config.items() if key != "type")

    def test_zero_meta_step_size_remains_supported(self):
        state = SwiftTD(meta_step_size=0.0).init(feature_dim=3)
        assert state.meta_step_size == 0.0

    def test_config_factory_rejects_nonfinite_swifttd_values(self):
        with pytest.raises(ValueError, match="initial_step_size"):
            optimizer_from_config({"type": "SwiftTD", "initial_step_size": math.nan})

    def test_update_returns_correct_shapes(self):
        optimizer = SwiftTD(initial_step_size=0.01)
        state = optimizer.init(feature_dim=5)
        obs = jnp.ones(5, dtype=jnp.float32)

        result = optimizer.update(state, jnp.array(1.0), obs, obs * 0.9, jnp.array(0.99))

        chex.assert_shape(result.weight_delta, (5,))
        chex.assert_shape(result.bias_delta, ())
        chex.assert_shape(result.new_state.log_step_sizes, (6,))
        assert isinstance(result.new_state, SwiftTDState)
        for key in ("mean_step_size", "mean_eligibility_trace", "bound_scale"):
            assert key in result.metrics


# =============================================================================
# Exact equivalence with the reference implementation
# =============================================================================


class TestSwiftTDMatchesReference:
    """Weight trajectories must match the author's dense C++ reference."""

    @pytest.mark.parametrize(
        "gamma,lam,alpha,theta",
        [
            (0.9, 0.7, 0.05, 0.01),  # moderate: bound mostly inactive
            (0.99, 0.9, 0.5, 0.05),  # aggressive: overshoot bound + decay active
        ],
    )
    def test_matches_reference_trajectory(self, gamma, lam, alpha, theta):
        dim, steps = 4, 40
        eta, decay, eta_min = 0.1, 0.9, float(np.exp(-15.0))
        rng = np.random.default_rng(7)
        xs = rng.normal(size=(steps, dim))
        rewards = rng.normal(size=(steps,))

        # Reference operates on augmented features [x, 1] (bias as a feature).
        ref = _RefSwiftTDNonSparse(dim + 1, lam, alpha, gamma, eta, decay, theta, eta_min)
        ref_weights = []
        for t in range(steps):
            ref.step(np.concatenate([xs[t], [1.0]]), rewards[t])
            ref_weights.append(ref.w.copy())

        # Ours: call t handles transition (x_{t-1}, r_t, x_t); reference
        # Step(phi_0, r_0) touches no weights (zero traces), it only extends
        # the traces with phi_0 -- which is our call 1's first half-step.
        optimizer = SwiftTD(
            initial_step_size=alpha,
            meta_step_size=theta,
            trace_decay=lam,
            eta=eta,
            step_size_decay=decay,
            eta_min=eta_min,
        )
        state = optimizer.init(dim)
        w = jnp.zeros(dim, dtype=jnp.float32)
        b = jnp.array(0.0, dtype=jnp.float32)
        max_err = 0.0
        for t in range(1, steps):
            obs = jnp.asarray(xs[t - 1], dtype=jnp.float32)
            nxt = jnp.asarray(xs[t], dtype=jnp.float32)
            td_error = (
                rewards[t] + gamma * (jnp.dot(w, nxt) + b) - (jnp.dot(w, obs) + b)
            )
            upd = optimizer.update(state, td_error, obs, nxt, jnp.array(gamma))
            w = w + upd.weight_delta
            b = b + upd.bias_delta
            state = upd.new_state
            # Our weights after call t == reference weights after Step t.
            err = max(
                float(np.max(np.abs(ref_weights[t][:-1] - np.asarray(w)))),
                abs(float(ref_weights[t][-1]) - float(b)),
            )
            max_err = max(max_err, err)
        assert max_err < 1e-3, f"trajectory diverged from reference: {max_err}"


# =============================================================================
# Headline behaviors
# =============================================================================


class TestSwiftTDBehavior:
    """The three SwiftTD ideas, tested in the supervised limit (gamma=0)."""

    def test_correction_ratio_matches_bound_formula(self):
        """One update moves the prediction by exactly min(tau, eta) * error.

        From zero weights and fresh traces, ``delta_w = min(1, eta/tau) *
        alpha * phi * error`` so the new prediction is
        ``min(tau, eta) * target`` (paper Eq. 7 with the overshoot bound).
        """
        obs = jnp.array([1.0, -1.0, 0.5], dtype=jnp.float32)
        target = jnp.array(2.0)
        phi_sq_sum = float(jnp.sum(obs**2)) + 1.0  # + bias feature

        for alpha, eta in ((0.01, 0.1), (0.05, 0.1)):  # tau < eta, tau > eta
            optimizer = SwiftTD(initial_step_size=alpha, meta_step_size=0.0, eta=eta)
            state = optimizer.init(3)
            w = jnp.zeros(3, dtype=jnp.float32)
            b = jnp.array(0.0, dtype=jnp.float32)
            tau = alpha * phi_sq_sum
            _, w, b, _ = _supervised_update(optimizer, state, w, b, obs, target)
            new_prediction = float(jnp.dot(w, obs) + b)
            expected = min(tau, eta) * float(target)
            assert new_prediction == pytest.approx(expected, rel=1e-5)

    def test_step_size_increases_under_correlated_errors(self):
        """Persistently same-sign errors on active features raise their
        step-sizes; features that are never active keep theirs unchanged."""
        optimizer = SwiftTD(initial_step_size=0.01, meta_step_size=0.01, eta=0.1)
        state = optimizer.init(3)
        initial_log = state.log_step_sizes
        obs = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)
        target = jnp.array(1.0)
        w = jnp.zeros(3, dtype=jnp.float32)
        b = jnp.array(0.0, dtype=jnp.float32)

        for _ in range(30):
            upd, w, b, td_error = _supervised_update(optimizer, state, w, b, obs, target)
            state = upd.new_state
            assert float(td_error) > 0.0  # errors stay same-sign (bounded updates)

        # Active features: step-sizes meta-learned upward.
        assert float(state.log_step_sizes[0]) > float(initial_log[0])
        assert float(state.log_step_sizes[1]) > float(initial_log[1])
        # Inactive feature (phi = 0): untouched.
        assert float(state.log_step_sizes[2]) == pytest.approx(float(initial_log[2]))

    def test_overshoot_bound_prevents_overshoot(self):
        """With absurdly large step-sizes, repeated updates on one sample
        must approach the target monotonically without ever crossing it."""
        optimizer = SwiftTD(initial_step_size=10.0, meta_step_size=0.01, eta=0.1)
        state = optimizer.init(5)
        obs = jnp.ones(5, dtype=jnp.float32)
        target = jnp.array(2.0)
        w = jnp.zeros(5, dtype=jnp.float32)
        b = jnp.array(0.0, dtype=jnp.float32)

        prev_error = float(target)
        for _ in range(20):
            upd, w, b, td_error = _supervised_update(optimizer, state, w, b, obs, target)
            state = upd.new_state
            error = float(td_error)
            assert jnp.isfinite(error)
            assert error > 0.0, "prediction overshot the target"
            assert error <= prev_error + 1e-6, "error must shrink monotonically"
            prev_error = error
        prediction = float(jnp.dot(w, obs) + b)
        assert 0.0 < prediction < float(target)

    def test_decay_triggers_when_bound_active(self):
        """When tau > eta the active step-sizes decay by ln(decay) * phi^2
        and the decay metric fires; when tau < eta nothing decays."""
        obs = jnp.array([1.0, 1.0], dtype=jnp.float32)
        target = jnp.array(1.0)

        # tau = 0.05 * 3 = 0.15 > eta = 0.1 -> decay triggers.
        optimizer = SwiftTD(
            initial_step_size=0.05, meta_step_size=0.0, eta=0.1, step_size_decay=0.99
        )
        state = optimizer.init(2)
        upd, _, _, _ = _supervised_update(
            optimizer, state, jnp.zeros(2), jnp.array(0.0), obs, target
        )
        assert float(upd.metrics["decay_triggered"]) == 1.0
        expected = float(jnp.log(0.05) + jnp.log(0.99))  # phi_i^2 = 1 for all entries
        chex.assert_trees_all_close(
            upd.new_state.log_step_sizes, jnp.full(3, expected), atol=1e-6
        )

        # tau = 0.02 * 3 = 0.06 < eta = 0.1 -> no decay, step-sizes unchanged.
        optimizer = SwiftTD(
            initial_step_size=0.02, meta_step_size=0.0, eta=0.1, step_size_decay=0.99
        )
        state = optimizer.init(2)
        upd, _, _, _ = _supervised_update(
            optimizer, state, jnp.zeros(2), jnp.array(0.0), obs, target
        )
        assert float(upd.metrics["decay_triggered"]) == 0.0
        chex.assert_trees_all_close(
            upd.new_state.log_step_sizes, jnp.full(3, jnp.log(0.02)), atol=1e-6
        )

    def test_traces_reset_at_terminal(self):
        """gamma = 0 zeroes the eligibility traces for the next transition;
        gamma * lambda > 0 keeps them alive."""
        optimizer = SwiftTD(initial_step_size=0.01, trace_decay=0.9)
        state = optimizer.init(3)
        obs = jnp.ones(3, dtype=jnp.float32)

        result = optimizer.update(state, jnp.array(1.0), obs, obs, jnp.array(0.0))
        chex.assert_trees_all_close(
            result.new_state.eligibility_traces, jnp.zeros(4), atol=1e-7
        )

        result = optimizer.update(state, jnp.array(1.0), obs, obs, jnp.array(0.9))
        assert float(jnp.max(jnp.abs(result.new_state.eligibility_traces))) > 0.0

    def test_zero_gamma_does_not_require_finite_next_observation(self) -> None:
        """gamma=0 still required a finite next obs even though V(s') is unused."""
        optimizer = SwiftTD(initial_step_size=0.01, trace_decay=0.9)
        state = optimizer.init(2)
        obs = jnp.asarray([0.5, -0.25], dtype=jnp.float32)
        result = optimizer.update(
            state,
            jnp.asarray(1.0, dtype=jnp.float32),
            obs,
            jnp.asarray([jnp.inf, 0.0], dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        assert bool(jnp.all(jnp.isfinite(result.weight_delta)))
        chex.assert_trees_all_close(
            result.new_state.eligibility_traces, jnp.zeros(3), atol=1e-7
        )

    def test_zero_gamma_does_not_disguise_consumed_inf_trace_state(self) -> None:
        optimizer = SwiftTD(initial_step_size=0.01, trace_decay=0.0)
        state = optimizer.init(2).replace(  # type: ignore[attr-defined]
            eligibility_traces=jnp.full((3,), jnp.inf, dtype=jnp.float32)
        )

        result = optimizer.update(
            state,
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray([0.5, -0.25], dtype=jnp.float32),
            jnp.asarray([jnp.inf, 0.0], dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )

        assert not bool(result.update_applied)
        chex.assert_trees_all_equal(result.new_state, state)

    def test_integrates_with_td_linear_learner(self):
        """SwiftTD follows the TDOptimizer interface and drives TDLinearLearner."""
        learner = TDLinearLearner(optimizer=SwiftTD(initial_step_size=0.01))
        state = learner.init(feature_dim=4)
        assert isinstance(state.optimizer_state, SwiftTDState)

        key = jr.key(3)
        for i in range(10):
            key, k_obs, k_next = jr.split(key, 3)
            obs = jr.normal(k_obs, (4,), dtype=jnp.float32)
            nxt = jr.normal(k_next, (4,), dtype=jnp.float32)
            result = learner.update(state, obs, jnp.array(1.0), nxt, jnp.array(0.9))
            state = result.state
        assert bool(jnp.all(jnp.isfinite(state.weights)))
        assert bool(jnp.all(jnp.isfinite(result.metrics)))


# =============================================================================
# Learning quality on XDistShiftStream
# =============================================================================

_FEATURE_DIM = 10
_NUM_RELEVANT = 3
_NUM_STEPS = 6000
_FINAL_WINDOW = 1500
_NUM_SEEDS = 10
_LMS_SWEEP = (1e-3, 1e-2, 1e-1)


def _make_stream() -> XDistShiftStream:
    return XDistShiftStream(
        feature_dim=_FEATURE_DIM,
        num_relevant=_NUM_RELEVANT,
        scale_change_interval=1500,
    )


def _seed_keys():
    return jr.split(jr.key(0), _NUM_SEEDS)


def _final_errors_and_finiteness(step_fn, init_carry):
    """Scan a supervised learning loop; return (final-window MSE, all-finite)."""
    _, sq_errors = jax.lax.scan(step_fn, init_carry, jnp.arange(_NUM_STEPS))
    return jnp.mean(sq_errors[-_FINAL_WINDOW:]), jnp.all(jnp.isfinite(sq_errors))


def _run_lms_seed(step_size, key):
    stream = _make_stream()
    optimizer = LMS(step_size=step_size)

    def step_fn(carry, idx):
        (w, b, opt_state), s_state = carry
        ts, s_new = stream.step(s_state, idx)
        err = jnp.squeeze(ts.target) - (jnp.dot(w, ts.observation) + b)
        upd = optimizer.update(opt_state, err, ts.observation)
        new_carry = ((w + upd.weight_delta, b + upd.bias_delta, upd.new_state), s_new)
        return new_carry, err**2

    init = (
        (jnp.zeros(_FEATURE_DIM, dtype=jnp.float32), jnp.array(0.0), optimizer.init(_FEATURE_DIM)),
        stream.init(key),
    )
    return _final_errors_and_finiteness(step_fn, init)


def _run_swift_seed(initial_step_size, key):
    stream = _make_stream()
    optimizer = SwiftTD(initial_step_size=initial_step_size, meta_step_size=1e-2, eta=0.1)

    def step_fn(carry, idx):
        (w, b, opt_state), s_state = carry
        ts, s_new = stream.step(s_state, idx)
        td_error = jnp.squeeze(ts.target) - (jnp.dot(w, ts.observation) + b)
        upd = optimizer.update(
            opt_state, td_error, ts.observation, ts.observation, jnp.array(0.0)
        )
        new_carry = ((w + upd.weight_delta, b + upd.bias_delta, upd.new_state), s_new)
        return new_carry, td_error**2

    init = (
        (jnp.zeros(_FEATURE_DIM, dtype=jnp.float32), jnp.array(0.0), optimizer.init(_FEATURE_DIM)),
        stream.init(key),
    )
    return _final_errors_and_finiteness(step_fn, init)


@functools.cache
def _best_fixed_lms_mean() -> float:
    """Mean final-window MSE of the best FIXED scalar step-size in the sweep.

    Non-finite seeds count as infinite error, so a step-size with any
    diverged seed cannot be "best".
    """
    keys = _seed_keys()
    means = []
    for alpha in _LMS_SWEEP:
        finals, finite = jax.vmap(lambda k, a=alpha: _run_lms_seed(a, k))(keys)
        finals = jnp.where(finite, finals, jnp.inf)
        means.append(float(jnp.mean(jnp.nan_to_num(finals, nan=jnp.inf))))
    return min(means)


class TestSwiftTDLearningQuality:
    """SwiftTD on the input-scale-shift stream (Alberta Plan Step 1)."""

    def test_no_nans_and_beats_best_fixed_step_size(self):
        """Zero NaN seeds AND lower final tracking error than the best
        fixed scalar step-size from the sweep {1e-3, 1e-2, 1e-1}."""
        finals, finite = jax.vmap(lambda k: _run_swift_seed(1e-3, k))(_seed_keys())

        assert int(jnp.sum(finite)) == _NUM_SEEDS, "SwiftTD produced non-finite errors"
        swift_mean = float(jnp.mean(finals))
        best_fixed = _best_fixed_lms_mean()
        assert swift_mean < best_fixed, (
            f"SwiftTD final MSE {swift_mean:.4f} should beat best fixed "
            f"step-size {best_fixed:.4f}"
        )
        # Sanity: close to the stream's noise floor (0.1^2), far from divergence.
        assert swift_mean < 0.05

    def test_robust_to_100x_too_large_initial_step_size(self):
        """With the initial step-size set 100x larger than the best fixed
        scalar (0.1 vs 1e-3), the bound + decay keep every seed stable --
        and it still beats the best fixed step-size."""
        finals, finite = jax.vmap(lambda k: _run_swift_seed(1e-1, k))(_seed_keys())

        assert int(jnp.sum(finite)) == _NUM_SEEDS, "SwiftTD diverged with large init"
        swift_mean = float(jnp.mean(finals))
        assert swift_mean < _best_fixed_lms_mean()
        assert swift_mean < 0.05


def test_swift_td_integer_validation() -> None:
    swift = SwiftTD(initial_step_size=0.01)

    with pytest.raises(ValueError, match="feature_dim"):
        swift.init(feature_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_dim"):
        swift.init(feature_dim=4.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_dim"):
        swift.init(feature_dim=0)

    state = swift.init(feature_dim=np.int32(4))
    assert state.log_step_sizes.shape == (5,)


@pytest.mark.parametrize(
    "integer_type",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    ],
)
def test_swift_td_accepts_full_numpy_integer_family(integer_type: type) -> None:
    assert SwiftTD().init(integer_type(2)).log_step_sizes.shape == (3,)


def test_swift_td_rejects_hostile_integer_subclasses_without_hooks() -> None:
    class HostileInt(int):
        def __index__(self) -> int:  # pragma: no cover
            raise AssertionError("conversion hook executed")

        def __repr__(self) -> str:  # pragma: no cover
            raise AssertionError("repr hook executed")

    with pytest.raises(ValueError, match="feature_dim"):
        SwiftTD().init(HostileInt(2))


def test_swift_td_state_resource_endpoint_and_adjacent_are_allocation_free() -> None:
    scalar_budget = (2**31 - 1) // 4
    _require_float32_resource("endpoint", float32_scalars=scalar_budget)
    with pytest.raises(ValueError, match="byte count"):
        _require_float32_resource("endpoint", float32_scalars=scalar_budget + 1)

    last_feature_dim = ((scalar_budget - 5) // 8) - 1
    with pytest.raises(ValueError, match="SwiftTD state byte count"):
        SwiftTD().init(last_feature_dim + 1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initial_step_size", 1e100),
        ("initial_step_size", 1e-100),
        ("eta", 1e100),
        ("eta_min", 1e-100),
        ("step_size_decay", 1e-100),
    ],
)
def test_swift_td_rejects_float32_overflow_and_underflow(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        SwiftTD(**{field: value})


def test_swift_td_rejects_inverted_eta_interval() -> None:
    with pytest.raises(ValueError, match="eta_min must be <= eta"):
        SwiftTD(eta=0.01, eta_min=0.02)


def test_swift_td_validates_adopted_state_static_contract() -> None:
    optimizer = SwiftTD(initial_step_size=0.01)
    state = optimizer.init(3)
    obs = jnp.ones((3,), dtype=jnp.float32)
    scalar = jnp.asarray(0.5, dtype=jnp.float32)

    with pytest.raises(ValueError, match="eligibility_traces must have shape"):
        optimizer.update(
            state.replace(eligibility_traces=jnp.zeros((3,), dtype=jnp.float32)),
            scalar,
            obs,
            obs,
            scalar,
        )
    with pytest.raises(TypeError, match="log_step_sizes must have dtype float32"):
        optimizer.update(
            state.replace(log_step_sizes=jnp.zeros((4,), dtype=jnp.int32)),
            scalar,
            obs,
            obs,
            scalar,
        )


@pytest.mark.parametrize(
    ("name", "replacement", "match"),
    [
        ("td_error", jnp.ones((1,), dtype=jnp.float32), "td_error must have shape"),
        ("observation", jnp.ones((2,), dtype=jnp.float32), "observation must have shape"),
        (
            "next_observation",
            jnp.ones((2,), dtype=jnp.float32),
            "next_observation must have shape",
        ),
        ("gamma", jnp.ones((1,), dtype=jnp.float32), "gamma must have shape"),
        ("observation", jnp.ones((3,), dtype=jnp.int32), "observation must have dtype"),
    ],
)
def test_swift_td_public_input_metadata_fails_before_tracing(
    name: str, replacement: jax.Array, match: str
) -> None:
    optimizer = SwiftTD(initial_step_size=0.01)
    state = optimizer.init(3)
    values = {
        "td_error": jnp.asarray(1.0, dtype=jnp.float32),
        "observation": jnp.ones((3,), dtype=jnp.float32),
        "next_observation": jnp.ones((3,), dtype=jnp.float32),
        "gamma": jnp.asarray(0.9, dtype=jnp.float32),
    }
    values[name] = replacement
    with pytest.raises((TypeError, ValueError), match=match):
        jax.jit(optimizer.update)(state, **values)


def test_swift_td_invalid_adopted_scalar_domain_rolls_back() -> None:
    optimizer = SwiftTD(initial_step_size=0.01)
    state = optimizer.init(3).replace(eta_min=jnp.asarray(0.2, dtype=jnp.float32))
    obs = jnp.ones((3,), dtype=jnp.float32)
    result = optimizer.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        obs,
        obs,
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.new_state, state)


@pytest.mark.parametrize("gamma", (-0.5, 1.5))
def test_swift_td_out_of_range_discount_rolls_back(gamma: float) -> None:
    optimizer = SwiftTD(initial_step_size=0.01, trace_decay=0.9)
    state = optimizer.init(3)
    observation = jnp.ones((3,), dtype=jnp.float32)
    result = jax.jit(optimizer.update)(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        observation,
        observation,
        jnp.asarray(gamma, dtype=jnp.float32),
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.new_state, state)
    chex.assert_trees_all_equal(result.weight_delta, jnp.zeros_like(result.weight_delta))
    chex.assert_trees_all_equal(result.bias_delta, jnp.zeros_like(result.bias_delta))
    for value in result.metrics.values():
        chex.assert_trees_all_equal(value, jnp.zeros_like(value))


def test_step_size_decay_clamped_to_eta_min_prevents_underflow() -> None:
    """Strong decay from large phi must clamp log_alphas to eta_min without NaN."""
    optimizer = SwiftTD(
        initial_step_size=0.1,
        meta_step_size=1e-3,
        eta=0.1,
        step_size_decay=0.5,
        eta_min=1e-6,
    )
    state = optimizer.init(2)
    obs = jnp.array([100.0, 100.0], dtype=jnp.float32)
    next_obs = jnp.zeros(2, dtype=jnp.float32)
    result = jax.jit(optimizer.update)(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        obs,
        next_obs,
        jnp.asarray(0.9, dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    assert jnp.all(jnp.isfinite(result.weight_delta))
    assert jnp.isfinite(result.bias_delta)
    assert jnp.all(result.new_state.log_step_sizes >= jnp.log(result.new_state.eta_min) - 1e-6)
    assert jnp.all(result.new_state.log_step_sizes <= jnp.log(result.new_state.eta) + 1e-6)
    assert bool(result.metrics["decay_triggered"])


def test_step_size_decay_maintains_eta_min_invariant_across_multiple_steps() -> None:
    """Consecutive large-magnitude steps must stay bounded and finite."""
    optimizer = SwiftTD(
        initial_step_size=0.1,
        meta_step_size=1e-3,
        eta=0.1,
        step_size_decay=0.1,
        eta_min=1e-5,
    )
    state = optimizer.init(3)
    update_jit = jax.jit(optimizer.update)

    for i in range(10):
        scale = float(10 ** (i % 3 + 1))
        obs = jnp.full(3, scale, dtype=jnp.float32)
        next_obs = jnp.full(3, scale * 0.5, dtype=jnp.float32)
        result = update_jit(
            state,
            jnp.asarray(float(i + 1), dtype=jnp.float32),
            obs,
            next_obs,
            jnp.asarray(0.95, dtype=jnp.float32),
        )
        assert bool(result.update_applied), f"Step {i} failed to apply"
        assert jnp.all(result.new_state.log_step_sizes >= jnp.log(result.new_state.eta_min) - 1e-6)
        assert jnp.all(result.new_state.log_step_sizes <= jnp.log(result.new_state.eta) + 1e-6)
        state = result.new_state


def test_step_size_decay_with_zero_meta_step_size() -> None:
    """Decay must commit cleanly and stay clamped when meta_step_size is 0."""
    optimizer = SwiftTD(
        initial_step_size=0.1,
        meta_step_size=0.0,
        eta=0.1,
        step_size_decay=0.5,
        eta_min=1e-6,
    )
    state = optimizer.init(2)
    obs = jnp.array([50.0, 50.0], dtype=jnp.float32)
    next_obs = jnp.zeros(2, dtype=jnp.float32)
    result = jax.jit(optimizer.update)(
        state,
        jnp.asarray(2.0, dtype=jnp.float32),
        obs,
        next_obs,
        jnp.asarray(0.9, dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    assert jnp.all(result.new_state.log_step_sizes >= jnp.log(result.new_state.eta_min) - 1e-6)
    assert jnp.all(result.new_state.log_step_sizes <= jnp.log(result.new_state.eta) + 1e-6)
