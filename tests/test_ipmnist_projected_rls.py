"""Projected covariance must follow dense RLS on its declared feature subspace."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    _projected_rls_gain,
    screening_spec,
)


def test_projected_gain_tracks_dense_psd_on_nondisible_width() -> None:
    width, rank = 12, 4
    basis = np.zeros((width, rank))
    for i in range(width - 1):
        basis[i, i % (rank - 1)] = 1.0
    basis[-1, -1] = 1.0
    basis /= np.linalg.norm(basis, axis=0)
    cov = basis @ basis.T / 0.03
    roots = jnp.eye(rank)[None] / np.sqrt(0.03)
    rng = np.random.default_rng(222)
    for phi in np.abs(rng.normal(size=(14, width))):
        phi[-1] = 1.0
        projected = cov @ phi
        expected_gain = projected / (1.0 + phi @ projected)
        cov -= np.outer(expected_gain, projected)
        gain, roots = jax.jit(_projected_rls_gain)(roots, jnp.asarray(phi, jnp.float32))
        np.testing.assert_allclose(gain, expected_gain, rtol=2e-4, atol=3e-5)
        factor = basis @ np.asarray(roots[0], np.float64)
        np.testing.assert_allclose(factor @ factor.T, cov, rtol=3e-4, atol=5e-5)


def test_zero_projected_rank_reduces_bitwise_to_parent() -> None:
    name = "rls_head_resid_identmap50_r_idet003_s0031640625_r02"
    parent = screening_spec(name)
    arm = screening_spec(name + "_projected8")
    params = {
        "w1": jnp.ones((4, 4)),
        "b1": jnp.zeros(4),
        "w2": jnp.eye(4),
        "b2": jnp.arange(4, dtype=jnp.float32),
        "w3": jnp.zeros((4, 2)),
        "b3": jnp.zeros(2),
    }

    def run(spec, hp):
        initialize, step = spec.factory(hp)
        op = jax.jit(step)
        pars, state = params, initialize(params)
        for i in range(3):
            pars, state, metrics = op(
                pars, state, jnp.arange(4, dtype=jnp.float32), jnp.int32(i % 2), jr.key(i)
            )
        return pars, state, metrics

    control = run(parent, parent.hyperparameters)
    inert = run(arm, {**arm.hyperparameters, "rls_projected_rank": 0.0})
    assert jax.tree.structure(control) == jax.tree.structure(inert)
    for a, b in zip(jax.tree.leaves(control), jax.tree.leaves(inert), strict=True):
        np.testing.assert_array_equal(a, b)


def test_live_rank_eight_uses_compact_state_and_resets_its_factor() -> None:
    spec = screening_spec("rls_head_resid_identmap50_r_idet003_s0031640625_r02_projected8")
    init, step = spec.factory(spec.hyperparameters)
    params = {
        "w1": jnp.ones((4, 3)),
        "b1": jnp.zeros(3),
        "w2": jnp.ones((3, 10)),
        "b2": jnp.arange(10, dtype=jnp.float32),
        "w3": jnp.zeros((10, 2)),
        "b3": jnp.zeros(2),
    }
    state = init(params)
    assert state.inner.p.shape == (1, 8, 8)
    args = jnp.zeros(4), jnp.int32(1), jr.key(0)
    _, after, _ = jax.jit(step)(params, state, *args)
    _, reset_step = spec.factory({**spec.hyperparameters, "rls_reset_frac": 0.0})
    _, reset, _ = jax.jit(reset_step)(params, state, *args)
    assert not np.array_equal(after.inner.p, state.inner.p)
    np.testing.assert_array_equal(reset.inner.p, state.inner.p)
    np.testing.assert_array_equal(reset.inner.wout, after.inner.wout)
