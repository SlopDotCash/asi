"""Regression tests for finite CBP outgoing-weight utility in plasticity_diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.plasticity_diagnostics import (
    INPUT_DIM,
    N_CLASSES,
    DiagnosticMLPState,
    MLPState,
    _diagnostic_step,
    _init_diagnostic_state,
    _init_state,
    _step,
)

pytestmark = pytest.mark.unit


def _step_diagnostic(
    state: DiagnosticMLPState,
    inputs: jax.Array,
    label: jax.Array,
    *,
    use_jit: bool,
    replacement_rate: float = 0.0,
) -> DiagnosticMLPState:
    updated: DiagnosticMLPState
    with jax.disable_jit(not use_jit):
        updated, _loss, _prediction, _counts = _diagnostic_step(
            state,
            inputs,
            label,
            jr.key(7),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(replacement_rate, dtype=jnp.float32),
            jnp.asarray(2, dtype=jnp.int32),
        )
    return updated


def _step_v1(
    state: MLPState,
    inputs: jax.Array,
    label: jax.Array,
    *,
    use_jit: bool,
    replacement_rate: float = 0.0,
) -> MLPState:
    updated: MLPState
    with jax.disable_jit(not use_jit):
        updated, _loss, _prediction, _count = _step(
            state,
            inputs,
            label,
            jr.key(7),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(replacement_rate, dtype=jnp.float32),
            jnp.asarray(2, dtype=jnp.int32),
        )
    return updated


@pytest.mark.parametrize("use_jit", [False, True])
@pytest.mark.parametrize("layer", [1, 2, 3])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_diagnostic_step_keeps_finite_utility_when_outgoing_weight_sum_overflows(
    use_jit: bool, layer: int, sign: float
) -> None:
    width = 8
    base = _init_diagnostic_state(jr.key(0), width)
    active_bias = jnp.zeros((width,), dtype=jnp.float32).at[0].set(1.0)
    large = jnp.float32(sign * 1e38)
    w_hidden = jnp.zeros((width, width), dtype=jnp.float32).at[0, :].set(large).at[1, :].set(large)
    w_out = (
        jnp.zeros((width, N_CLASSES), dtype=jnp.float32).at[0, :].set(large).at[1, :].set(large)
    )
    state = base._replace(
        w1=jnp.zeros((INPUT_DIM, width), dtype=jnp.float32),
        b1=active_bias if layer == 1 else jnp.zeros((width,), dtype=jnp.float32),
        w2=w_hidden if layer == 1 else jnp.zeros((width, width), dtype=jnp.float32),
        b2=active_bias if layer == 2 else jnp.zeros((width,), dtype=jnp.float32),
        w3=w_hidden if layer == 2 else jnp.zeros((width, width), dtype=jnp.float32),
        b3=active_bias if layer == 3 else jnp.zeros((width,), dtype=jnp.float32),
        w4=w_out if layer == 3 else jnp.zeros((width, N_CLASSES), dtype=jnp.float32),
        b4=jnp.zeros((N_CLASSES,), dtype=jnp.float32),
        utility1=jnp.full((width,), 2.0, dtype=jnp.float32),
        utility2=jnp.full((width,), 4.0, dtype=jnp.float32),
        utility3=jnp.full((width,), 6.0, dtype=jnp.float32),
    )
    inputs = jnp.zeros((INPUT_DIM,), dtype=jnp.float32)
    label = jnp.asarray(0, dtype=jnp.int32)

    updated = _step_diagnostic(state, inputs, label, use_jit=use_jit)

    utility, prior = (
        (updated.utility1, 2.0)
        if layer == 1
        else (updated.utility2, 4.0)
        if layer == 2
        else (updated.utility3, 6.0)
    )
    host = np.asarray(utility)
    expected_active = float(np.float32(0.01) * np.float32(1e38))
    assert np.all(np.isfinite(host))
    assert host[0] == pytest.approx(expected_active, rel=1e-6)
    # Dead ReLU unit 1 has 1e38 outgoing weights; 0.0 * mean(|w|) must be 0.0, not NaN.
    np.testing.assert_allclose(host[1:], np.float32(0.99 * prior), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("use_jit", [False, True])
@pytest.mark.parametrize("layer", [1, 2])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_v1_step_keeps_finite_utility_when_outgoing_weight_sum_overflows(
    use_jit: bool, layer: int, sign: float
) -> None:
    width = 8
    base = _init_state(jr.key(0), width)
    active_bias = jnp.zeros((width,), dtype=jnp.float32).at[0].set(1.0)
    large = jnp.float32(sign * 1e38)
    w_hidden = jnp.zeros((width, width), dtype=jnp.float32).at[0, :].set(large).at[1, :].set(large)
    w_out = (
        jnp.zeros((width, N_CLASSES), dtype=jnp.float32).at[0, :].set(large).at[1, :].set(large)
    )
    state = base._replace(
        w1=jnp.zeros((INPUT_DIM, width), dtype=jnp.float32),
        b1=active_bias if layer == 1 else jnp.zeros((width,), dtype=jnp.float32),
        w2=w_hidden if layer == 1 else jnp.zeros((width, width), dtype=jnp.float32),
        b2=active_bias if layer == 2 else jnp.zeros((width,), dtype=jnp.float32),
        w3=w_out if layer == 2 else jnp.zeros((width, N_CLASSES), dtype=jnp.float32),
        b3=jnp.zeros((N_CLASSES,), dtype=jnp.float32),
        utility1=jnp.full((width,), 3.0, dtype=jnp.float32),
        utility2=jnp.full((width,), 5.0, dtype=jnp.float32),
    )
    inputs = jnp.zeros((INPUT_DIM,), dtype=jnp.float32)
    label = jnp.asarray(0, dtype=jnp.int32)

    updated = _step_v1(state, inputs, label, use_jit=use_jit)

    utility, prior = (updated.utility1, 3.0) if layer == 1 else (updated.utility2, 5.0)
    host = np.asarray(utility)
    expected_active = float(np.float32(0.01) * np.float32(1e38))
    assert np.all(np.isfinite(host))
    assert host[0] == pytest.approx(expected_active, rel=1e-6)
    np.testing.assert_allclose(host[1:], np.float32(0.99 * prior), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("use_jit", [False, True])
def test_modest_outgoing_weights_match_direct_mean_bitwise(use_jit: bool) -> None:
    width = 8
    state = _init_diagnostic_state(jr.key(11), width)
    inputs = jnp.linspace(-0.5, 0.5, INPUT_DIM, dtype=jnp.float32)
    label = jnp.asarray(3, dtype=jnp.int32)

    hidden1 = jax.nn.relu(inputs @ state.w1 + state.b1)
    hidden2 = jax.nn.relu(hidden1 @ state.w2 + state.b2)
    hidden3 = jax.nn.relu(hidden2 @ state.w3 + state.b3)
    direct1 = 0.99 * state.utility1 + 0.01 * hidden1 * jnp.mean(jnp.abs(state.w2), axis=1)
    direct2 = 0.99 * state.utility2 + 0.01 * hidden2 * jnp.mean(jnp.abs(state.w3), axis=1)
    direct3 = 0.99 * state.utility3 + 0.01 * hidden3 * jnp.mean(jnp.abs(state.w4), axis=1)

    updated = _step_diagnostic(state, inputs, label, use_jit=use_jit)

    np.testing.assert_array_equal(np.asarray(updated.utility1), np.asarray(direct1))
    np.testing.assert_array_equal(np.asarray(updated.utility2), np.asarray(direct2))
    np.testing.assert_array_equal(np.asarray(updated.utility3), np.asarray(direct3))


def test_nonfinite_outgoing_weight_remains_nonfinite() -> None:
    width = 8
    base = _init_diagnostic_state(jr.key(0), width)
    state = base._replace(
        w1=jnp.zeros((INPUT_DIM, width), dtype=jnp.float32),
        b1=jnp.ones((width,), dtype=jnp.float32),
        w2=jnp.zeros((width, width), dtype=jnp.float32).at[0, 0].set(jnp.inf),
    )
    updated = _step_diagnostic(
        state,
        jnp.zeros((INPUT_DIM,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        use_jit=False,
    )
    assert np.isinf(float(updated.utility1[0]))
