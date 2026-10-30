"""Adam keeps a finite bias-corrected step when a gradient square overflows.

``(1 - beta2) * g**2`` is about ``4e35`` for ``|g| = 2e19`` and the default
``beta2``, which float32 can store. Squaring first overflows, and dividing
that moment by ``1 - beta2**t`` on the first step rebuilds ``g**2``. A still
unrepresentable moment, including ``|g| = 1e21``, stays rejected.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.baseline_optimizers import Adam

pytestmark = pytest.mark.unit

_OVERFLOW_MAGNITUDE = np.float32(2.0e19)
_UNREPRESENTABLE_MAGNITUDE = np.float32(1.0e21)
_MODES = ("linear", "loss_gradient", "prediction_gradient")


def _weighted_moment(beta2: np.floating, magnitude: np.floating) -> np.float64:
    scale = np.float64(np.float32(1.0) - np.float32(beta2))
    return scale * np.float64(magnitude) * np.float64(magnitude)


def _ideal_step(step_size: np.floating, magnitude: np.floating, eps: np.floating) -> np.float32:
    step_size = np.float32(step_size)
    magnitude = np.float32(magnitude)
    eps = np.float32(eps)
    return np.float32(step_size * magnitude / (magnitude + eps))


def _apply(mode: str, signed_signal: np.floating, *, compiled: bool):
    optimizer = Adam()
    signal = jnp.asarray(np.float32(signed_signal), dtype=jnp.float32)
    if mode == "linear":
        state = optimizer.init(1)
        observation = jnp.ones(1, dtype=jnp.float32)

        def apply(state, signal, observation):
            return optimizer.update(state, signal, observation)

        args = (state, signal, observation)
    elif mode == "loss_gradient":
        state = optimizer.init_for_shape((1,))
        gradient = jnp.asarray([signal], dtype=jnp.float32)

        def apply(state, gradient):
            return optimizer.update_from_gradient_checked(state, gradient)

        args = (state, gradient)
    elif mode == "prediction_gradient":
        state = optimizer.init_for_shape((1,))
        gradient = jnp.ones(1, dtype=jnp.float32)

        def apply(state, gradient, signal):
            return optimizer.update_from_gradient_checked(state, gradient, signal)

        args = (state, gradient, signal)
    else:
        raise AssertionError(mode)

    if compiled:
        return jax.jit(apply)(*args), state
    with jax.disable_jit():
        return apply(*args), state


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("compiled", [False, True])
def test_overflowing_gradient_square_keeps_the_adam_step(
    mode: str, sign: float, compiled: bool
) -> None:
    """A finite ``|g| = 2e19`` has a representable weighted second moment."""
    result, state = _apply(mode, sign * _OVERFLOW_MAGNITUDE, compiled=compiled)
    expected_moment = _weighted_moment(np.asarray(state.beta2), _OVERFLOW_MAGNITUDE)
    expected_step = np.copysign(
        _ideal_step(np.asarray(state.step_size), _OVERFLOW_MAGNITUDE, np.asarray(state.eps)),
        np.float32(sign),
    )

    assert bool(result.update_applied)
    assert float(result.new_state.t) == 1.0
    np.testing.assert_allclose(np.asarray(result.new_state.v), expected_moment, rtol=2e-6)
    if mode == "linear":
        np.testing.assert_allclose(np.asarray(result.new_state.bias_v), expected_moment, rtol=2e-6)
        np.testing.assert_allclose(np.asarray(result.weight_delta), expected_step, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(result.bias_delta), expected_step, rtol=1e-6)
        assert float(result.weight_delta[0]) != 0.0
    else:
        np.testing.assert_allclose(np.asarray(result.step), expected_step, rtol=1e-6)
        assert float(result.step[0]) != 0.0


def test_finite_gradient_square_keeps_the_direct_adam_step() -> None:
    """Ordinary squares stay on the historical left-to-right formula."""
    optimizer = Adam()
    error = np.float32(0.25)
    observation = np.float32(0.5)
    state = optimizer.init(1)
    beta1 = np.float32(state.beta1)
    beta2 = np.float32(state.beta2)
    step_size = np.float32(state.step_size)
    eps = np.float32(state.eps)
    g = np.float32(np.float32(-error) * observation)
    g_b = np.float32(-error)
    one = np.float32(1.0)
    new_t = np.float32(1.0)
    new_v = np.float32(np.float32(one - beta2) * np.float32(g * g))
    new_bias_v = np.float32(np.float32(one - beta2) * np.float32(g_b * g_b))
    m_hat = np.float32(np.float32(np.float32(one - beta1) * g) / np.float32(one - beta1**new_t))
    bias_m_hat = np.float32(
        np.float32(np.float32(one - beta1) * g_b) / np.float32(one - beta1**new_t)
    )
    v_hat = np.float32(new_v / np.float32(one - beta2**new_t))
    bias_v_hat = np.float32(new_bias_v / np.float32(one - beta2**new_t))
    weight_delta = np.float32(np.float32(-step_size) * m_hat / np.float32(np.sqrt(v_hat) + eps))
    bias_delta = np.float32(
        np.float32(-step_size) * bias_m_hat / np.float32(np.sqrt(bias_v_hat) + eps)
    )

    result = optimizer.update(
        state,
        jnp.asarray(error, dtype=jnp.float32),
        jnp.asarray([observation], dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    np.testing.assert_array_equal(np.asarray(result.new_state.v), np.asarray([new_v]))
    np.testing.assert_array_equal(np.asarray(result.new_state.bias_v), new_bias_v)
    np.testing.assert_array_equal(np.asarray(result.weight_delta), np.asarray([weight_delta]))
    np.testing.assert_array_equal(np.asarray(result.bias_delta), bias_delta)

    param_state = optimizer.init_for_shape((1,))
    gradient = jnp.asarray([g], dtype=jnp.float32)
    param_result = optimizer.update_from_gradient_checked(param_state, gradient)
    param_step = np.float32(step_size * m_hat / np.float32(np.sqrt(v_hat) + eps))
    assert bool(param_result.update_applied)
    np.testing.assert_array_equal(np.asarray(param_result.new_state.v), np.asarray([new_v]))
    np.testing.assert_array_equal(np.asarray(param_result.step), np.asarray([param_step]))


@pytest.mark.parametrize("mode", ("linear", "loss_gradient"))
@pytest.mark.parametrize(
    "bad",
    [np.float32(1.0e21), np.float32(np.inf), np.float32(np.nan)],
)
def test_unrepresentable_adam_square_still_rejected(mode: str, bad: np.float32) -> None:
    """``|g| = 1e21`` makes even the weighted moment overflow float32."""
    result, state = _apply(mode, bad, compiled=False)

    assert not bool(result.update_applied)
    np.testing.assert_array_equal(np.asarray(result.new_state.t), np.asarray(state.t))
    np.testing.assert_array_equal(np.asarray(result.new_state.v), np.asarray(state.v))
    np.testing.assert_array_equal(np.asarray(result.new_state.m), np.asarray(state.m))
    if mode == "linear":
        np.testing.assert_array_equal(
            np.asarray(result.weight_delta), np.zeros(1, dtype=np.float32)
        )
        np.testing.assert_array_equal(np.asarray(result.bias_delta), np.float32(0.0))
        np.testing.assert_array_equal(np.asarray(result.new_state.bias_v), np.asarray(state.bias_v))
    else:
        np.testing.assert_array_equal(np.asarray(result.step), np.zeros(1, dtype=np.float32))
