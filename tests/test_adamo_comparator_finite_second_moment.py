"""AdamO keeps a finite step when a task-gradient square overflows.

``(1 - beta2) * g**2`` is about ``4e35`` for ``|g| = 2e19`` and the default
second-moment decay. Float32 can store that variance, but the square overflows
first. Step-1 bias correction then divides by ``1 - beta2`` and rebuilds the
square under the root. ``|g| = 1e21`` makes the weighted variance itself
overflow and stays rejected.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.plasticity_comparators import adamo_update

pytestmark = pytest.mark.unit

_OVERFLOW_MAGNITUDE = np.float32(2.0e19)
_UNREPRESENTABLE_MAGNITUDE = np.float32(1.0e21)


def _apply(gradient: np.floating, *, compiled: bool):
    weights = jnp.zeros((1, 1), dtype=jnp.float32)
    task_gradient = jnp.asarray([[np.float32(gradient)]], dtype=jnp.float32)
    zeros = jnp.zeros_like(weights)

    def apply(weights, task_gradient, zeros):
        return adamo_update(
            weights,
            task_gradient,
            zeros,
            zeros,
            step=1,
            learning_rate=1e-3,
            isometry_strength=0.0,
        )

    if compiled:
        return jax.jit(apply)(weights, task_gradient, zeros)
    with jax.disable_jit():
        return apply(weights, task_gradient, zeros)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("compiled", [False, True])
def test_overflowing_gradient_square_keeps_the_adamo_step(
    sign: float, compiled: bool
) -> None:
    """A finite ``|g| = 2e19`` has a representable bias-corrected denominator."""
    gradient = sign * _OVERFLOW_MAGNITUDE
    updated, moment, variance = _apply(gradient, compiled=compiled)
    beta2 = 0.999
    magnitude = np.float64(_OVERFLOW_MAGNITUDE)
    expected_variance = np.float64(1.0 - beta2) * magnitude * magnitude
    expected_weight = -np.float64(1e-3) * np.float64(sign)

    np.testing.assert_allclose(np.asarray(variance), expected_variance, rtol=2e-6, atol=0.0)
    np.testing.assert_allclose(np.asarray(moment), (1.0 - 0.9) * gradient, rtol=2e-6, atol=0.0)
    np.testing.assert_allclose(np.asarray(updated), expected_weight, rtol=2e-6, atol=0.0)
    assert float(np.asarray(updated).reshape(-1)[0]) != 0.0
    assert np.all(np.isfinite(np.asarray(updated)))


def test_finite_gradient_square_keeps_the_direct_adamo_step() -> None:
    """Ordinary squares stay on the historical moment, variance, and step."""
    gradient = jnp.asarray([[-1.5, 0.25]], dtype=jnp.float32)
    weights = jnp.asarray([[0.25, -0.5]], dtype=jnp.float32)
    zeros = jnp.zeros_like(weights)
    beta1 = 0.9
    beta2 = 0.999
    step = 1
    learning_rate = 0.01
    epsilon = 1e-8
    moment = (1.0 - beta1) * gradient
    variance = (1.0 - beta2) * jnp.square(gradient)
    corrected_moment = moment / (1.0 - beta1**step)
    corrected_variance = variance / (1.0 - beta2**step)
    task_delta = learning_rate * corrected_moment / (jnp.sqrt(corrected_variance) + epsilon)
    expected = weights - task_delta

    updated, got_moment, got_variance = adamo_update(
        weights,
        gradient,
        zeros,
        zeros,
        step=step,
        learning_rate=learning_rate,
        isometry_strength=0.0,
    )

    np.testing.assert_array_equal(np.asarray(got_moment), np.asarray(moment))
    np.testing.assert_array_equal(np.asarray(got_variance), np.asarray(variance))
    np.testing.assert_array_equal(np.asarray(updated), np.asarray(expected))


@pytest.mark.parametrize(
    "bad",
    [np.float32(1.0e21), np.float32(np.inf), np.float32(np.nan)],
)
def test_unrepresentable_adamo_square_still_rejected(bad: np.float32) -> None:
    """``|g| = 1e21`` overflows the weighted variance, and non-finite inputs fail."""
    if np.isfinite(bad):
        match = "AdamO update must produce only finite values"
    else:
        match = "task_gradient must contain only finite values"
    with pytest.raises(ValueError, match=match):
        _apply(bad, compiled=False)

    updated, moment, variance = _apply(bad, compiled=True)
    assert np.all(~np.isfinite(np.asarray(updated)))
    assert np.all(~np.isfinite(np.asarray(moment)))
    assert np.all(~np.isfinite(np.asarray(variance)))
