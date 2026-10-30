"""Adaptive ObGD keeps a finite step when a trace square overflows.

``(1 - beta2) * (signal * z)**2`` is about ``4e35`` for ``|signal| = 2e19``,
``z = 1``, and the default ``beta2``. Float32 can store that moment, but the
square overflows first. Step-1 bias correction then divides by ``1 - beta2``
and rebuilds the square in the denominator. ``|signal| = 1e21`` makes the
weighted moment itself overflow and stays rejected.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.recurrent_trace_actor_critic import adaptive_obgd_update

pytestmark = pytest.mark.unit

_OVERFLOW_MAGNITUDE = np.float32(2.0e19)
_UNREPRESENTABLE_MAGNITUDE = np.float32(1.0e21)
_BETA2 = 0.999
_ALPHA = 1.0
_KAPPA = 1.0
_EPSILON = 1e-8


def _apply(signal: np.floating, *, compiled: bool):
    traces = {"weight": jnp.ones(1, dtype=jnp.float32)}
    second_moment = {"weight": jnp.zeros(1, dtype=jnp.float32)}
    observed = jnp.asarray(np.float32(signal), dtype=jnp.float32)

    def apply(traces, second_moment, observed):
        return adaptive_obgd_update(
            traces,
            second_moment,
            observed,
            alpha=_ALPHA,
            kappa=_KAPPA,
            beta2=_BETA2,
            epsilon=_EPSILON,
            step=1,
        )

    if compiled:
        return jax.jit(apply)(traces, second_moment, observed), second_moment
    with jax.disable_jit():
        return apply(traces, second_moment, observed), second_moment


def _expected_moment(magnitude: np.floating) -> np.float64:
    beta2 = np.float32(_BETA2)
    scale = np.float64(np.float32(1.0) - beta2)
    return scale * np.float64(magnitude) * np.float64(magnitude)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("compiled", [False, True])
def test_overflowing_trace_square_keeps_the_obgd_step(sign: float, compiled: bool) -> None:
    """A finite ``|signal * z| = 2e19`` has a representable weighted moment."""
    result, _ = _apply(sign * _OVERFLOW_MAGNITUDE, compiled=compiled)
    expected_moment = _expected_moment(_OVERFLOW_MAGNITUDE)
    beta2 = np.float32(_BETA2)
    correction = np.float64(np.float32(1.0) - beta2)
    rms = np.sqrt(expected_moment / correction)
    normalized = 1.0 / (rms + np.float64(_EPSILON))
    bound = abs(float(_OVERFLOW_MAGNITUDE)) * abs(normalized) * _ALPHA * _KAPPA
    denominator = max(1.0, bound)
    expected_update = np.float64(sign) * np.float64(_OVERFLOW_MAGNITUDE) * (
        _ALPHA / denominator
    ) * normalized

    assert bool(result.update_applied)
    np.testing.assert_allclose(
        np.asarray(result.second_moment["weight"]),
        expected_moment,
        rtol=2e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(result.updates["weight"]),
        expected_update,
        rtol=2e-6,
        atol=0.0,
    )
    assert float(result.updates["weight"][0]) != 0.0


def test_finite_trace_square_keeps_the_direct_obgd_step() -> None:
    """Ordinary squares stay on the historical moment and root."""
    signal = np.float32(-3.0)
    trace = np.float32(2.0)
    previous = np.float32(0.25)
    beta2 = np.float32(0.9)
    alpha = np.float32(0.2)
    kappa = np.float32(2.0)
    epsilon = np.float32(1e-4)
    step = np.int32(4)
    one = np.float32(1.0)
    product = np.float32(signal * trace)
    moment = np.float32(
        beta2 * previous + np.float32(one - beta2) * np.float32(product * product)
    )
    correction = np.float32(one - np.float32(beta2**step))
    corrected = np.float32(moment / correction)
    normalized = np.float32(trace / np.float32(np.sqrt(corrected) + epsilon))
    bound = np.float32(np.maximum(np.abs(signal), one) * np.abs(normalized) * alpha * kappa)
    denominator = np.maximum(one, bound)
    step_size = np.float32(alpha / denominator)
    update = np.float32(step_size * signal * normalized)

    result = adaptive_obgd_update(
        {"weight": jnp.asarray([trace], dtype=jnp.float32)},
        {"weight": jnp.asarray([previous], dtype=jnp.float32)},
        jnp.asarray(signal, dtype=jnp.float32),
        alpha=float(alpha),
        kappa=float(kappa),
        beta2=float(beta2),
        epsilon=float(epsilon),
        step=int(step),
    )

    assert bool(result.update_applied)
    np.testing.assert_array_equal(np.asarray(result.second_moment["weight"]), np.asarray([moment]))
    np.testing.assert_array_equal(np.asarray(result.updates["weight"]), np.asarray([update]))
    np.testing.assert_array_equal(np.asarray(result.step_size), step_size)


@pytest.mark.parametrize(
    "bad",
    [np.float32(1.0e21), np.float32(np.inf), np.float32(np.nan)],
)
def test_unrepresentable_obgd_square_still_rejected(bad: np.float32) -> None:
    """``|signal| = 1e21`` makes even the weighted moment overflow float32."""
    result, previous = _apply(bad, compiled=False)

    assert not bool(result.update_applied)
    np.testing.assert_array_equal(
        np.asarray(result.second_moment["weight"]),
        np.asarray(previous["weight"]),
    )
    np.testing.assert_array_equal(
        np.asarray(result.updates["weight"]), np.zeros(1, dtype=np.float32)
    )
    np.testing.assert_array_equal(np.asarray(result.step_size), np.float32(0.0))
