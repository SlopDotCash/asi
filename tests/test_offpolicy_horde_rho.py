"""Per-decision importance-sampling tests for the off-policy Horde backends.

Covers the canonical trace composition ``z_t = rho_t (gamma lambda z_{t-1}
+ grad)`` with update ``delta_t z_t`` (Sutton & Barto 2nd ed., eq. 12.23;
GQ(lambda), Maei & Sutton 2010), the rho=1 equivalence with the on-policy
``HordeLearner``, and the rho-weighted GTD correction term.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np

from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.off_policy_horde import (
    NonlinearSharedGTDHordeLearner,
    OffPolicyHordeLearner,
)
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.types import DemonType, GVFSpec, HordeSpec, create_horde_spec


def _spec(
    gammas: tuple[float, ...],
    lamdas: tuple[float, ...] | None = None,
) -> HordeSpec:
    if lamdas is None:
        lamdas = tuple(0.0 for _ in gammas)
    demons = tuple(
        GVFSpec(
            name=f"demon_{i}",
            demon_type=DemonType.PREDICTION,
            gamma=gamma,
            lamda=lamdas[i],
            cumulant_index=i,
        )  # type: ignore[call-arg]
        for i, gamma in enumerate(gammas)
    )
    return create_horde_spec(demons)


def test_two_step_canonical_per_decision_is_update() -> None:
    """Hand-computed 2-step check of canonical per-decision IS TD(lambda).

    Canonical composition: ``z_t = rho_t (gamma lambda z_{t-1} + grad_t)``
    and ``w += alpha delta_t z_t``, so the coefficient on ``grad_{t-1}`` in
    the step-2 update is ``gamma lambda rho_2 rho_1`` — one ratio per
    decision.  A composition that multiplies rho into both the trace decay
    and the update instead yields ``gamma lambda rho_2^2`` and fails here.
    """
    gamma, lamda, alpha = 0.5, 1.0, 0.1
    rho1, rho2 = 2.0, 0.5
    learner = OffPolicyHordeLearner(
        _spec(gammas=(gamma,), lamdas=(lamda,)),
        hidden_sizes=(),
        optimizer=LMS(step_size=alpha),
        ratio_clip=10.0,
        trace_ratio_clip=10.0,
        sparsity=0.0,
    )
    state = learner.init(2, jax.random.key(0))

    w = np.asarray(state.head_params.weights[0], dtype=np.float64).reshape(-1)
    b = float(state.head_params.biases[0][0])
    x1 = np.array([1.0, 0.0])
    x2 = np.array([0.0, 1.0])
    x3 = np.array([1.0, 1.0])

    def value(w_: np.ndarray, b_: float, x: np.ndarray) -> float:
        return float(w_ @ x + b_)

    # Step 1: z_1 = rho_1 * grad, update alpha * delta_1 * z_1.
    delta1 = 1.0 + gamma * value(w, b, x2) - value(w, b, x1)
    z_w = rho1 * x1
    z_b = rho1 * 1.0
    w = w + alpha * delta1 * z_w
    b = b + alpha * delta1 * z_b

    # Step 2: z_2 = rho_2 * (gamma * lamda * z_1 + grad), update delta_2 * z_2.
    delta2 = -0.5 + gamma * value(w, b, x3) - value(w, b, x2)
    z_w = rho2 * (gamma * lamda * z_w + x2)
    z_b = rho2 * (gamma * lamda * z_b + 1.0)
    w = w + alpha * delta2 * z_w
    b = b + alpha * delta2 * z_b

    result = learner.update_with_ratios(
        state,
        jnp.asarray(x1, dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.asarray(x2, dtype=jnp.float32),
        jnp.array([rho1], dtype=jnp.float32),
    )
    result = learner.update_with_ratios(
        result.state,
        jnp.asarray(x2, dtype=jnp.float32),
        jnp.array([-0.5], dtype=jnp.float32),
        jnp.asarray(x3, dtype=jnp.float32),
        jnp.array([rho2], dtype=jnp.float32),
    )

    final = result.state
    np.testing.assert_allclose(
        np.asarray(final.head_params.weights[0]).reshape(-1), w, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(final.head_params.biases[0]).reshape(-1), [b], atol=1e-5
    )
    # The stored trace must carry each decision's ratio exactly once.
    np.testing.assert_allclose(
        np.asarray(final.head_traces[0][0]).reshape(-1), z_w, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(final.head_traces[0][1]).reshape(-1), [z_b], atol=1e-5
    )


def test_rho_one_matches_on_policy_horde_learner() -> None:
    """With rho=1 everywhere, updates must equal the on-policy HordeLearner."""
    spec = _spec(gammas=(0.0, 0.7), lamdas=(0.0, 0.6))
    off = OffPolicyHordeLearner(
        spec,
        hidden_sizes=(8,),
        optimizer=LMS(step_size=0.05),
    )
    on = HordeLearner(
        spec,
        hidden_sizes=(8,),
        optimizer=LMS(step_size=0.05),
    )
    off_state = off.init(3, jax.random.key(11))
    on_state = on.init(3, jax.random.key(11))
    chex.assert_trees_all_close(off_state.head_params, on_state.head_params)

    rng = np.random.default_rng(3)
    ones = jnp.ones(2, dtype=jnp.float32)
    for t in range(6):
        obs = jnp.asarray(rng.normal(size=3), dtype=jnp.float32)
        next_obs = jnp.asarray(rng.normal(size=3), dtype=jnp.float32)
        cums = jnp.asarray(rng.normal(size=2), dtype=jnp.float32)
        if t == 3:
            cums = cums.at[1].set(jnp.nan)  # inactive demon step
        off_state = off.update_with_ratios(off_state, obs, cums, next_obs, ones).state
        on_state = on.update(on_state, obs, cums, next_obs).state

    chex.assert_trees_all_close(off_state.trunk_params, on_state.trunk_params, atol=1e-6)
    chex.assert_trees_all_close(off_state.head_params, on_state.head_params, atol=1e-6)
    chex.assert_trees_all_close(off_state.head_traces, on_state.head_traces, atol=1e-6)
    chex.assert_trees_all_close(off_state.trunk_traces, on_state.trunk_traces, atol=1e-6)


def test_gtd_backend_correction_term_carries_rho() -> None:
    """Hand-computed 2-step check that the TDC/GQ correction is rho-weighted.

    Canonical off-policy TDC (Sutton & Barto 2nd ed., Section 11.7) and
    GQ(0) (Maei & Sutton 2010) both use
    ``theta += alpha (rho delta grad - gamma rho (v . grad) grad')`` with
    secondary ``v += beta (rho delta grad - (v . grad) grad)``.  The first
    step leaves the secondary weights nonzero, so a step-2 update with
    rho != 1 exposes a correction term that dropped rho.
    """
    gamma, alpha, beta = 0.8, 0.01, 0.1
    rho1, rho2 = 2.0, 0.5
    learner = NonlinearSharedGTDHordeLearner(
        _spec(gammas=(gamma,)),
        hidden_size=3,
        primary_step_size=alpha,
        secondary_step_size=beta,
        ratio_clip=10.0,
    )
    state = learner.init(2, jax.random.key(21))

    trunk_w = np.asarray(state.trunk_w, dtype=np.float64)
    trunk_b = np.asarray(state.trunk_b, dtype=np.float64)
    head_w = np.asarray(state.head_w[0], dtype=np.float64)
    head_b = float(state.head_b[0])
    sec = [
        np.zeros_like(trunk_w),
        np.zeros_like(trunk_b),
        np.zeros_like(head_w),
        0.0,
    ]

    x1 = np.array([1.0, -0.5])
    x2 = np.array([-0.3, 1.0])
    x3 = np.array([0.7, 0.2])
    cums = [1.0, -0.5]
    rhos = [rho1, rho2]

    for step, (x, x_next) in enumerate([(x1, x2), (x2, x3)]):
        h = np.tanh(trunk_w @ x + trunk_b)
        h_next = np.tanh(trunk_w @ x_next + trunk_b)
        v = head_w @ h + head_b
        v_next = head_w @ h_next + head_b
        delta = cums[step] + gamma * v_next - v
        rho = rhos[step]

        g_hidden = head_w * (1.0 - h**2)
        grads = [g_hidden[:, None] * x[None, :], g_hidden, h, 1.0]
        g_hidden_next = head_w * (1.0 - h_next**2)
        next_grads = [g_hidden_next[:, None] * x_next[None, :], g_hidden_next, h_next, 1.0]
        sd = sum(float(np.vdot(s, g)) for s, g in zip(sec, grads))

        steps = [
            alpha * (rho * delta * g - gamma * rho * sd * g_next)
            for g, g_next in zip(grads, next_grads)
        ]
        sec = [s + beta * (rho * delta * g - sd * g) for s, g in zip(sec, grads)]
        trunk_w = trunk_w + steps[0]
        trunk_b = trunk_b + steps[1]
        head_w = head_w + steps[2]
        head_b = head_b + steps[3]

    result_state = state
    for step, (x, x_next) in enumerate([(x1, x2), (x2, x3)]):
        result_state = learner.update_with_ratios_and_discounts(
            result_state,
            jnp.asarray(x, dtype=jnp.float32),
            jnp.array([cums[step]], dtype=jnp.float32),
            jnp.asarray(x_next, dtype=jnp.float32),
            jnp.array([rhos[step]], dtype=jnp.float32),
            jnp.array([gamma], dtype=jnp.float32),
        ).state

    np.testing.assert_allclose(np.asarray(result_state.trunk_w), trunk_w, atol=1e-5)
    np.testing.assert_allclose(np.asarray(result_state.trunk_b), trunk_b, atol=1e-5)
    np.testing.assert_allclose(np.asarray(result_state.head_w[0]), head_w, atol=1e-5)
    np.testing.assert_allclose(float(result_state.head_b[0]), head_b, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(result_state.secondary_head_w[0]), sec[2], atol=1e-5
    )


def test_gtd_backend_masks_nan_cumulants() -> None:
    """A NaN cumulant must freeze that demon and not poison the shared trunk."""
    learner = NonlinearSharedGTDHordeLearner(
        _spec(gammas=(0.8, 0.8)),
        hidden_size=4,
        primary_step_size=0.01,
        secondary_step_size=0.01,
        ratio_clip=10.0,
    )
    state = learner.init(2, jax.random.key(5))
    obs = jnp.array([1.0, 0.2], dtype=jnp.float32)
    next_obs = jnp.array([-0.4, 1.0], dtype=jnp.float32)

    # One warm-up step so demon 0 has nonzero secondary weights.
    warm = learner.update_with_ratios_and_discounts(
        state,
        obs,
        jnp.array([1.0, 1.0], dtype=jnp.float32),
        next_obs,
        jnp.array([1.5, 1.5], dtype=jnp.float32),
        jnp.array([0.8, 0.8], dtype=jnp.float32),
    )
    result = learner.update_with_ratios_and_discounts(
        warm.state,
        obs,
        jnp.array([jnp.nan, 1.0], dtype=jnp.float32),
        next_obs,
        jnp.array([1.5, 1.5], dtype=jnp.float32),
        jnp.array([0.8, 0.8], dtype=jnp.float32),
    )

    chex.assert_tree_all_finite(result.state)
    chex.assert_trees_all_close(
        result.state.head_w[0], warm.state.head_w[0], atol=0.0
    )
    chex.assert_trees_all_close(
        result.state.secondary_trunk_w[0], warm.state.secondary_trunk_w[0], atol=0.0
    )
    assert float(jnp.linalg.norm(result.state.head_w[1] - warm.state.head_w[1])) > 0.0
    assert bool(jnp.isnan(result.td_errors[0]))
    assert bool(jnp.isfinite(result.correction_norms[0]))


def test_gtd_backend_rejects_non_finite_cumulant() -> None:
    """An inf (not NaN) cumulant must reject that head without poisoning it."""
    learner = NonlinearSharedGTDHordeLearner(
        _spec(gammas=(0.8, 0.8)),
        hidden_size=4,
        primary_step_size=0.01,
        secondary_step_size=0.01,
        ratio_clip=10.0,
    )
    state = learner.init(2, jax.random.key(7))
    obs = jnp.array([1.0, 0.2], dtype=jnp.float32)
    next_obs = jnp.array([-0.4, 1.0], dtype=jnp.float32)
    rhos = jnp.array([1.5, 1.5], dtype=jnp.float32)
    discounts = jnp.array([0.8, 0.8], dtype=jnp.float32)

    result = learner.update_with_ratios_and_discounts(
        state,
        obs,
        jnp.array([jnp.inf, 1.0], dtype=jnp.float32),
        next_obs,
        rhos,
        discounts,
    )

    chex.assert_tree_all_finite(result.state)
    assert not bool(result.head_updates_applied[0])
    assert bool(result.head_updates_applied[1])
    assert bool(result.update_applied)
    assert float(jnp.linalg.norm(result.state.head_w[1] - state.head_w[1])) > 0.0
    assert float(jnp.linalg.norm(result.state.head_w[0] - state.head_w[0])) == 0.0


def test_gtd_backend_rejects_non_finite_observation() -> None:
    """A NaN observation must fail closed and leave the state untouched."""
    learner = NonlinearSharedGTDHordeLearner(
        _spec(gammas=(0.8,)),
        hidden_size=4,
        primary_step_size=0.01,
        secondary_step_size=0.01,
        ratio_clip=10.0,
    )
    state = learner.init(2, jax.random.key(9))
    result = learner.update_with_ratios_and_discounts(
        state,
        jnp.array([jnp.nan, 0.2], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([-0.4, 1.0], dtype=jnp.float32),
        jnp.array([1.5], dtype=jnp.float32),
        jnp.array([0.8], dtype=jnp.float32),
    )

    chex.assert_trees_all_close(result.state.trunk_w, state.trunk_w, atol=0.0)
    chex.assert_trees_all_close(result.state.head_w, state.head_w, atol=0.0)
    assert not bool(result.update_applied)
