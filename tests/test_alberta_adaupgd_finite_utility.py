"""Alberta AdaUPGD keeps a finite utility when a parameter product overflows.

``(1 - beta) * (-g * w)`` is about ``4e35`` for ``|g| = |w| = 2e19`` and the
default utility decay. Float32 can store that weighted utility, but ``g * w``
overflows first. On the first step the bias correction divides the weighted
utility back to the raw product, which still does not fit. After one ordinary
step the correction is near ``1 - beta**2``, and the corrected utility fits.
``|g| = |w| = 1e21`` makes the weighted product itself overflow and stays
rejected.
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


def _update(optimizer, state, params, gradient, key, *, compiled: bool):
    gradients = {"w": jnp.asarray(gradient, dtype=jnp.float32)}
    noise = {"w": jnp.zeros_like(gradients["w"])}

    def apply(state, params, gradients, noise, key):
        return optimizer.update(state, params, gradients, key, noise=noise)

    if compiled:
        return jax.jit(apply)(state, params, gradients, noise, key)
    with jax.disable_jit():
        return apply(state, params, gradients, noise, key)


def _warmup(optimizer, *, compiled: bool):
    params = {"w": jnp.asarray([1.0], dtype=jnp.float32)}
    state = optimizer.init(params)
    result = _update(
        optimizer,
        state,
        params,
        jnp.asarray([1.0], dtype=jnp.float32),
        jr.key(0),
        compiled=compiled,
    )
    assert bool(result.accepted)
    return result


def _expected_utility(
    decay: float,
    old_utility: np.floating,
    weight: np.floating,
    gradient: np.floating,
) -> np.float64:
    """Return ``beta * u + (1 - beta) * (-g * w)`` in wide precision."""

    scale = np.float64(1.0) - np.float64(decay)
    weighted = scale * np.float64(-gradient) * np.float64(weight)
    return np.float64(decay) * np.float64(old_utility) + weighted


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("compiled", [False, True])
def test_overflowing_utility_product_keeps_the_later_adaptive_step(
    sign: float, compiled: bool
) -> None:
    optimizer = _optimizer()
    warmed = _warmup(optimizer, compiled=compiled)
    magnitude = _OVERFLOW_MAGNITUDE
    weight = np.float32(sign) * magnitude
    gradient = magnitude
    params = {"w": jnp.asarray([weight], dtype=jnp.float32)}
    result = _update(
        optimizer,
        warmed.state,
        params,
        jnp.asarray([gradient], dtype=jnp.float32),
        warmed.next_key,
        compiled=compiled,
    )
    expected = _expected_utility(
        float(optimizer.config.utility_decay),
        np.asarray(warmed.state.utility_ema["w"])[0],
        weight,
        gradient,
    )
    assert abs(expected) < np.finfo(np.float32).max

    assert bool(result.accepted)
    assert int(result.state.step) == 2
    np.testing.assert_allclose(
        np.asarray(result.state.utility_ema["w"]),
        np.asarray([expected], dtype=np.float32),
        rtol=2e-6,
        atol=0.0,
    )
    assert not np.array_equal(
        np.asarray(result.state.gradient_second_moment["w"]),
        np.asarray(warmed.state.gradient_second_moment["w"]),
    )


def test_finite_utility_product_stays_bitwise() -> None:
    optimizer = _optimizer()
    param = jnp.asarray([0.25], dtype=jnp.float32)
    gradient = jnp.asarray([-1.5], dtype=jnp.float32)
    params = {"w": param}
    state = optimizer.init(params)
    result = _update(optimizer, state, params, gradient, jr.key(1), compiled=False)
    decay = optimizer.config.utility_decay
    expected = (1.0 - decay) * (-gradient * param)

    assert bool(result.accepted)
    np.testing.assert_array_equal(
        np.asarray(result.state.utility_ema["w"]),
        np.asarray(expected),
    )


def test_first_step_unrepresentable_corrected_utility_stays_rejected() -> None:
    """Step 1 divides the weighted utility back to the raw overflowing product."""

    optimizer = _optimizer()
    magnitude = _OVERFLOW_MAGNITUDE
    params = {"w": jnp.asarray([magnitude], dtype=jnp.float32)}
    state = optimizer.init(params)
    before = state
    result = _update(
        optimizer,
        state,
        params,
        jnp.asarray([magnitude], dtype=jnp.float32),
        jr.key(2),
        compiled=False,
    )
    raw_product = np.float64(magnitude) * np.float64(magnitude)
    assert raw_product > np.finfo(np.float32).max

    assert not bool(result.accepted)
    assert int(result.state.step) == int(before.step)
    np.testing.assert_array_equal(
        np.asarray(result.state.utility_ema["w"]),
        np.asarray(before.utility_ema["w"]),
    )
    np.testing.assert_array_equal(np.asarray(result.params["w"]), np.asarray(params["w"]))


@pytest.mark.parametrize("compiled", [False, True])
def test_weighted_utility_past_float32_stays_rejected(compiled: bool) -> None:
    optimizer = _optimizer()
    warmed = _warmup(optimizer, compiled=compiled)
    magnitude = _UNREPRESENTABLE_MAGNITUDE
    params = {"w": jnp.asarray([magnitude], dtype=jnp.float32)}
    result = _update(
        optimizer,
        warmed.state,
        params,
        jnp.asarray([magnitude], dtype=jnp.float32),
        warmed.next_key,
        compiled=compiled,
    )
    scale = np.float64(1.0) - np.float64(optimizer.config.utility_decay)
    weighted = scale * np.float64(magnitude) * np.float64(magnitude)
    assert weighted > np.finfo(np.float32).max

    assert not bool(result.accepted)
    assert int(result.state.step) == int(warmed.state.step)
    np.testing.assert_array_equal(
        np.asarray(result.state.utility_ema["w"]),
        np.asarray(warmed.state.utility_ema["w"]),
    )
