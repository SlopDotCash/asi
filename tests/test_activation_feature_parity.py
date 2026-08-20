"""Paper-property parity coverage for the #1566 activation and feature controls.

Companion to `test_activation_feature_ipmnist.py`. That file binds the lane's
harness contracts -- receipts, validators, schedule bounds, end-to-end execution.
This one binds the three primitives themselves to properties their papers state,
so an edit that silently changes a mechanism fails here rather than only shifting
a benchmark number that nothing cross-checks.

The primitives under test live in `plasticity_comparators`, which both the
screening lane and this file share, so these assertions hold regardless of which
lane consumes them.

Pinned sources:

- Smooth-Leaky: ICLR 2026 (OpenReview `XZf6wObHX4`), preprint arXiv:2509.22562v4.
- AID: ICML 2025, PMLR v267 pp. 47991-48026, preprint arXiv:2502.01342v2.
- Deep Fourier features: ICLR 2025 (OpenReview `NIkfix2eDQ`), preprint
  arXiv:2410.20634v1.
"""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.plasticity_comparators import (
    deep_fourier_features,
    interval_dropout,
    smooth_leaky,
)

# Figure 5 defaults from the Smooth-Leaky paper.
SL_ALPHA = 0.1
SL_POWER = 3.0
SL_CURVATURE = 5.0


def _sl(value):
    return smooth_leaky(value, alpha=SL_ALPHA, power=SL_POWER, curvature=SL_CURVATURE)


class TestSmoothLeakyParity:
    """Eq. 1: f(x) = a*x + (1 - a) * x * sigmoid(c * x / p)."""

    def test_matches_the_closed_form_of_equation_one(self) -> None:
        value = jnp.asarray(np.linspace(-8.0, 8.0, 65), dtype=jnp.float32)
        expected = SL_ALPHA * value + (1.0 - SL_ALPHA) * value * jax.nn.sigmoid(
            SL_CURVATURE * value / SL_POWER
        )
        assert jnp.allclose(_sl(value), expected, atol=1e-6)

    def test_is_asymptotically_linear_on_both_branches(self) -> None:
        # "asymptotically linear (f(x) ~ a*x for x << 0, f(x) ~ x for x >> 0)"
        far_negative = float(_sl(jnp.asarray([-1.0e4], dtype=jnp.float32))[0])
        far_positive = float(_sl(jnp.asarray([1.0e4], dtype=jnp.float32))[0])
        assert math.isclose(far_negative, SL_ALPHA * -1.0e4, rel_tol=1e-6)
        assert math.isclose(far_positive, 1.0e4, rel_tol=1e-6)

    def test_has_a_strict_non_zero_derivative_floor(self) -> None:
        # The paper calls an activation non-zero-floor when some a > 0 bounds
        # the derivative below on the negative branch. Contrast with ReLU, whose
        # negative-branch derivative is exactly zero.
        #
        # The margin here is thin and it is not the floor a=0.1 that binds: at
        # a=0.1, c=5, p=3 the derivative dips to roughly 0.01 near x=-1.4 before
        # rising back toward a. Any change to those three constants must re-check
        # this minimum rather than assume a bounds it, or the assertion can graze
        # zero without the grid ever revealing it.
        grad = jax.grad(lambda x: _sl(jnp.asarray([x], dtype=jnp.float32)).sum())
        derivatives = [float(grad(x)) for x in np.linspace(-40.0, -1e-3, 200)]
        assert min(derivatives) > 0.0

    def test_is_c1_continuous_where_leaky_relu_kinks(self) -> None:
        # The stated purpose is "removing the kink with a smooth, curved
        # transition", so the one-sided derivatives at the origin must agree.
        grad = jax.grad(lambda x: _sl(jnp.asarray([x], dtype=jnp.float32)).sum())
        left = float(grad(-1e-4))
        right = float(grad(1e-4))
        assert math.isclose(left, right, abs_tol=1e-3)

    def test_alpha_one_is_the_declared_mechanism_off_identity(self) -> None:
        value = jnp.asarray(np.linspace(-4.0, 4.0, 17), dtype=jnp.float32)
        assert jnp.allclose(
            smooth_leaky(value, alpha=1.0, power=SL_POWER, curvature=SL_CURVATURE), value
        )


class TestIntervalDropoutParity:
    """Property 1: AID_p applies ReLU with probability p, min(x, 0) otherwise."""

    def test_every_training_output_is_one_of_the_two_branches(self) -> None:
        value = jnp.asarray(np.linspace(-3.0, 3.0, 41), dtype=jnp.float32)
        positive = np.asarray(jnp.maximum(value, 0))
        negative = np.asarray(jnp.minimum(value, 0))
        for seed in range(24):
            drawn = np.asarray(interval_dropout(value, jr.key(seed), relu_probability=0.75))
            on_a_branch = np.isclose(drawn, positive) | np.isclose(drawn, negative)
            assert on_a_branch.all()

    def test_evaluation_is_the_exact_expectation_of_training(self) -> None:
        # Algorithm 1 scales interval j by (1 - p_j) at test time, which for the
        # simplified two-interval form is exactly E[AID_p(x)].
        probability = 0.75
        value = jnp.asarray(np.linspace(-3.0, 3.0, 41), dtype=jnp.float32)
        expectation = probability * jnp.maximum(value, 0) + (1.0 - probability) * jnp.minimum(
            value, 0
        )
        evaluated = interval_dropout(value, jr.key(0), relu_probability=probability, training=False)
        assert jnp.allclose(evaluated, expectation, atol=1e-6)

    def test_probability_one_reduces_to_relu(self) -> None:
        value = jnp.asarray(np.linspace(-3.0, 3.0, 41), dtype=jnp.float32)
        for training in (True, False):
            drawn = interval_dropout(value, jr.key(3), relu_probability=1.0, training=training)
            assert jnp.allclose(drawn, jnp.maximum(value, 0))

    def test_probability_one_half_is_a_linear_network(self) -> None:
        # "behaves like ReLU when p = 1 and a linear network when p = 0.5"
        value = jnp.asarray(np.linspace(-3.0, 3.0, 41), dtype=jnp.float32)
        evaluated = interval_dropout(value, jr.key(0), relu_probability=0.5, training=False)
        assert jnp.allclose(evaluated, 0.5 * value, atol=1e-6)

    def test_draws_vary_with_the_key(self) -> None:
        # A stochastic activation that ignores its key would silently become
        # deterministic and every other assertion here would still pass.
        value = jnp.asarray(np.linspace(-3.0, 3.0, 41), dtype=jnp.float32)
        first = np.asarray(interval_dropout(value, jr.key(1), relu_probability=0.5))
        second = np.asarray(interval_dropout(value, jr.key(2), relu_probability=0.5))
        assert not np.allclose(first, second)


class TestDeepFourierParity:
    """Proposition 1: one branch is near-linear on every quarter-period window."""

    def test_concatenates_sine_and_cosine_of_the_same_preactivation(self) -> None:
        value = jnp.asarray(np.linspace(-3.0, 3.0, 16), dtype=jnp.float32)
        features = deep_fourier_features(value)
        assert features.shape == (32,)
        assert jnp.allclose(features[:16], jnp.sin(value))
        assert jnp.allclose(features[16:], jnp.cos(value))

    def test_one_branch_is_linear_within_the_stated_bound(self) -> None:
        # For any z, sin or cos is within c = sqrt(2) * pi^2 / 28 of a linear
        # function on [z - pi/4, z + pi/4].
        bound = math.sqrt(2.0) * math.pi**2 / 28.0
        worst = 0.0
        for centre in np.linspace(-10.0, 10.0, 161):
            window = np.linspace(centre - math.pi / 4, centre + math.pi / 4, 48)
            best = math.inf
            for branch in (np.sin, np.cos):
                slope, intercept = np.polyfit(window, branch(window), 1)
                residual = float(np.abs(branch(window) - (slope * window + intercept)).max())
                best = min(best, residual)
            worst = max(worst, best)
        assert worst <= bound

    def test_disabled_is_the_declared_mechanism_off_passthrough(self) -> None:
        value = jnp.asarray(np.linspace(-3.0, 3.0, 16), dtype=jnp.float32)
        assert jnp.array_equal(deep_fourier_features(value, enabled=False), value)
