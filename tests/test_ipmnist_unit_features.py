"""Mechanism and exact control reduction for one nonpromoting RLS screen."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    _rls_head_forward,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, init_mlp_params

ARM = "rls_head_resid_identmap50_r_idet003_s005625_unitfeatures"
CONTROL = "rls_head_resid_identmap50_r_idet003_s005625"
CONFIG = IPMNISTConfig(
    n_tasks=1, task_length=12, input_dim=4, hidden1=5, hidden2=3, n_classes=2
)


def test_unit_features_match_hand_normalization_and_keep_bias():
    params = init_mlp_params(jr.key(28), CONFIG)
    x = jnp.array([0.5, -0.2, 0.3, 0.7], dtype=jnp.float32)
    wout = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    hidden1 = jax.nn.relu(x @ params["w1"] + params["b1"])
    hidden2 = jax.nn.relu(hidden1 @ params["w2"] + params["b2"])
    expected = jnp.concatenate(
        [hidden2 / jnp.sqrt(jnp.sum(hidden2**2) + 1e-8), jnp.ones(1)]
    )
    logits, features = _rls_head_forward(params, wout, x, unit_features=True)
    np.testing.assert_allclose(features, expected, rtol=1e-6)
    np.testing.assert_array_equal(logits, wout.T @ features)
    assert float(features[-1]) == 1.0


def test_mechanism_off_is_bit_exact_to_control_updates():
    control = screening_spec(CONTROL)
    candidate = screening_spec(ARM)
    init_a, step_a = control.factory(control.hyperparameters)
    init_b, step_b = candidate.factory({**candidate.hyperparameters, "unit_features": 0.0})
    params_a = init_mlp_params(jr.key(19), CONFIG)
    params_b = params_a
    state_a, state_b = init_a(params_a), init_b(params_b)
    step_a, step_b = jax.jit(step_a), jax.jit(step_b)
    for i in range(120):
        key = jr.fold_in(jr.key(73), i)
        x = jr.normal(key, (CONFIG.input_dim,)) + (8.0 if i >= 40 else 0.0)
        y = jnp.array(i % CONFIG.n_classes, dtype=jnp.int32)
        params_a, state_a, metrics_a = step_a(params_a, state_a, x, y, key)
        params_b, state_b, metrics_b = step_b(params_b, state_b, x, y, key)
        for a, b in zip(
            jax.tree.leaves((params_a, state_a, metrics_a)),
            jax.tree.leaves((params_b, state_b, metrics_b)), strict=True,
        ):
            np.testing.assert_array_equal(a, b)


def test_candidate_scores_before_update_and_changes_readout():
    params = init_mlp_params(jr.key(19), CONFIG)
    x = jnp.array([0.5, -0.2, 0.3, 0.7], dtype=jnp.float32)
    key = jr.key(4)
    candidate, control = screening_spec(ARM), screening_spec(CONTROL)
    init_a, step_a = candidate.factory(candidate.hyperparameters)
    init_b, step_b = control.factory(control.hyperparameters)
    _, state_a, metrics_a = jax.jit(step_a)(
        params, init_a(params), x, jnp.array(1), key
    )
    _, state_b, _ = jax.jit(step_b)(params, init_b(params), x, jnp.array(1), key)
    # Both initial heads predict class zero, before learning this class-one label.
    assert float(metrics_a[0]) == 0.0
    assert not np.array_equal(state_a.inner.wout, state_b.inner.wout)
