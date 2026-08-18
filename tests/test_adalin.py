from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from alberta_framework.benchmarks.adalin import ADALIN_PROTOCOL, adalin_relu, adalin_tanh


def test_relu_reduction_and_prelu_identity() -> None:
    x = jnp.array([-2.0, 3.0])
    np.testing.assert_array_equal(adalin_relu(x, jnp.zeros(2)), jax.nn.relu(x))
    np.testing.assert_array_equal(adalin_relu(x, jnp.array([0.25, 0.25])), [-0.5, 3.0])


def test_gate_is_stop_gradient() -> None:
    value, derivative = jax.value_and_grad(lambda z: adalin_tanh(z, jnp.array(0.2)))(
        jnp.array(2.0)
    )
    gate = jnp.cos(0.5 * jnp.pi * jnp.abs(1.0 - jnp.tanh(2.0) ** 2))
    expected = (1.0 - jnp.tanh(2.0) ** 2) + 0.2 * gate
    assert jnp.isfinite(value)
    np.testing.assert_allclose(derivative, expected, rtol=1e-6)


def test_protocol_keeps_pmnist_difference_explicit() -> None:
    assert ADALIN_PROTOCOL["paper_revision"] == "arXiv:2505.09486v1"
    assert ADALIN_PROTOCOL["paper_pmnist_tasks"] == 400
    assert ADALIN_PROTOCOL["asi_target_tasks"] == 200
    assert ADALIN_PROTOCOL["mechanism_off"] == "alpha_zero_exact_base_activation"
    assert ADALIN_PROTOCOL["scientific_promotion_allowed"] is False
