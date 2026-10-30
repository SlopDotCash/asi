"""Alberta AdaUPGD keeps a finite step when a gradient square overflows.

``(1 - beta2) * g**2`` is about ``4e35`` for ``|g| = 2e19`` and the default
second-moment decay. Float32 can store that moment, but the square overflows
first. Step-1 bias correction then divides by ``1 - beta2`` and rebuilds the
square in the denominator. ``|g| = 1e21`` makes the weighted moment itself
overflow and stays rejected.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.canonical_upgd import AlbertaAdaUPGD, AlbertaAdaUPGDConfig

pytestmark = pytest.mark.unit

_OVERFLOW_MAGNITUDE = np.float32(2.0e19)
_UNREPRESENTABLE_MAGNITUDE = np.float32(1.0e21)


def _optimizer() -> AlbertaAdaUPGD:
    return AlbertaAdaUPGD(
        AlbertaAdaUPGDConfig(
            noise_std=0.0,
            mode="non_protecting",
            weight_decay=0.0,
        )
    )


def _apply(gradient: np.floating, *, compiled: bool):
    optimizer = _optimizer()
    params = {"w": jnp.zeros(1, dtype=jnp.float32)}
    state = optimizer.init(params)
    gradients = {"w": jnp.asarray([np.float32(gradient)], dtype=jnp.float32)}
    noise = {"w": jnp.zeros(1, dtype=jnp.float32)}
    key = jr.key(0)

    def apply(state, params, gradients, noise, key):
        return optimizer.update(state, params, gradients, key, noise=noise)

    if compiled:
        return jax.jit(apply)(state, params, gradients, noise, key), state, params
    with jax.disable_jit():
        return apply(state, params, gradients, noise, key), state, params


def _expected_moment(decay: float, magnitude: np.floating) -> np.float64:
    """Return ``(1 - decay) * g**2`` in wide precision.

    The configured decay stays a Python float. Narrowing it before the
    subtraction is a different scale from the historical product.
    """

    scale = np.float64(1.0 - decay)
    return scale * np.float64(magnitude) * np.float64(magnitude)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("compiled", [False, True])
def test_overflowing_gradient_square_keeps_the_adaptive_step(
    sign: float, compiled: bool
) -> None:
    """A finite ``|g| = 2e19`` has a representable bias-corrected denominator."""
    optimizer = _optimizer()
    result, _, _ = _apply(sign * _OVERFLOW_MAGNITUDE, compiled=compiled)
    decay = float(optimizer.config.second_moment_decay)
    expected_moment = _expected_moment(decay, _OVERFLOW_MAGNITUDE)
    # Step 1 evaluates ``1 - beta**t`` in the parameter dtype. That rounding
    # does not match the Python ``1 - beta`` used to weight the square.
    correction = np.float64(1.0 - jnp.power(decay, jnp.asarray(1.0, dtype=jnp.float32)))
    rms = np.sqrt(expected_moment / correction)
    direction = np.float64(sign) * np.float64(_OVERFLOW_MAGNITUDE) / (
        rms + np.float64(optimizer.config.epsilon)
    )
    expected_param = -np.float64(optimizer.config.step_size) * direction

    assert bool(result.accepted)
    assert int(result.state.step) == 1
    np.testing.assert_allclose(
        np.asarray(result.state.gradient_second_moment["w"]),
        expected_moment,
        rtol=2e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(result.params["w"]),
        expected_param,
        rtol=2e-6,
        atol=0.0,
    )
    assert float(result.params["w"][0]) != 0.0


def test_finite_gradient_square_keeps_the_direct_adaptive_step() -> None:
    """Ordinary squares stay on the historical moment and root."""
    optimizer = _optimizer()
    gradient = jnp.asarray([-1.5], dtype=jnp.float32)
    param = jnp.asarray([0.25], dtype=jnp.float32)
    decay = optimizer.config.second_moment_decay
    step_size = optimizer.config.step_size
    epsilon = jnp.asarray(optimizer.config.epsilon, dtype=jnp.float32)
    moment = (1.0 - decay) * jnp.square(gradient)
    correction = 1.0 - jnp.power(decay, jnp.asarray(1.0, dtype=jnp.float32))
    divisor = jnp.maximum(correction, epsilon)
    direction = gradient / (jnp.sqrt(moment / divisor) + epsilon)
    expected_param = param - step_size * direction

    params = {"w": param}
    state = optimizer.init(params)
    result = optimizer.update(
        state,
        params,
        {"w": gradient},
        jr.key(0),
        noise={"w": jnp.zeros(1, dtype=jnp.float32)},
    )

    assert bool(result.accepted)
    np.testing.assert_array_equal(
        np.asarray(result.state.gradient_second_moment["w"]),
        np.asarray(moment),
    )
    np.testing.assert_array_equal(np.asarray(result.params["w"]), np.asarray(expected_param))


@pytest.mark.parametrize(
    "bad",
    [np.float32(1.0e21), np.float32(np.inf), np.float32(np.nan)],
)
def test_unrepresentable_adaptive_square_still_rejected(bad: np.float32) -> None:
    """``|g| = 1e21`` makes even the weighted moment overflow float32."""
    result, state, params = _apply(bad, compiled=False)

    assert not bool(result.accepted)
    np.testing.assert_array_equal(np.asarray(result.params["w"]), np.asarray(params["w"]))
    np.testing.assert_array_equal(
        np.asarray(result.state.gradient_second_moment["w"]),
        np.asarray(state.gradient_second_moment["w"]),
    )
    assert int(result.state.step) == 0
