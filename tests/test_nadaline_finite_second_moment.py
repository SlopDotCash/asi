"""NADALINE keeps a finite feature scale when a feature square overflows.

``(1 - decay) * x**2`` is about ``4e36`` for ``|x| = 2e19`` and the default
decay ``0.99``, which float32 can store. Squaring first overflows, so the
normalized step and the plain bias step are both discarded. ``|x| = 2e20``
makes the weighted moment itself overflow and stays rejected.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.baseline_optimizers import NADALINE

pytestmark = pytest.mark.unit

_OVERFLOW_MAGNITUDE = np.float32(2.0e19)
_UNREPRESENTABLE_MAGNITUDE = np.float32(2.0e20)


def _weighted_moment(decay: np.floating, magnitude: np.floating) -> np.float64:
    scale = np.float64(np.float32(1.0) - np.float32(decay))
    return scale * np.float64(magnitude) * np.float64(magnitude)


def _apply(observation: np.floating, error: np.floating, *, compiled: bool):
    optimizer = NADALINE()
    state = optimizer.init(1)
    observed = jnp.asarray([np.float32(observation)], dtype=jnp.float32)
    residual = jnp.asarray(np.float32(error), dtype=jnp.float32)

    def apply(state, residual, observed):
        return optimizer.update(state, residual, observed)

    if compiled:
        return jax.jit(apply)(state, residual, observed), state
    with jax.disable_jit():
        return apply(state, residual, observed), state


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("compiled", [False, True])
def test_overflowing_feature_square_keeps_the_normalized_step(
    sign: float, compiled: bool
) -> None:
    """A finite ``|x| = 2e19`` has a representable weighted second moment."""
    result, state = _apply(sign * _OVERFLOW_MAGNITUDE, 1.0, compiled=compiled)
    expected_moment = _weighted_moment(np.asarray(state.decay), _OVERFLOW_MAGNITUDE)
    expected_weight = (
        np.float64(state.step_size) * np.float64(sign) * np.float64(_OVERFLOW_MAGNITUDE)
        / expected_moment
    )
    expected_bias = np.float64(state.step_size)

    assert bool(result.update_applied)
    np.testing.assert_allclose(
        np.asarray(result.new_state.feature_second_moment),
        expected_moment,
        rtol=2e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(result.weight_delta), expected_weight, rtol=2e-6, atol=0.0
    )
    np.testing.assert_allclose(np.asarray(result.bias_delta), expected_bias, rtol=0.0, atol=0.0)
    assert float(result.weight_delta[0]) != 0.0
    assert float(result.bias_delta) != 0.0


def test_finite_feature_square_keeps_the_direct_nadaline_step() -> None:
    """Ordinary squares stay on the historical left-to-right formula."""
    optimizer = NADALINE()
    state = optimizer.init(1)
    error = np.float32(0.25)
    observation = np.float32(0.5)
    decay = np.float32(state.decay)
    step_size = np.float32(state.step_size)
    eps = np.float32(state.eps)
    one = np.float32(1.0)
    moment = np.float32(np.float32(one - decay) * np.float32(observation * observation))
    denom = np.maximum(eps, moment)
    weight_delta = np.float32(np.float32(np.float32(step_size) * error) * observation / denom)
    bias_delta = np.float32(step_size * error)

    result = optimizer.update(
        state,
        jnp.asarray(error, dtype=jnp.float32),
        jnp.asarray([observation], dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    np.testing.assert_array_equal(
        np.asarray(result.new_state.feature_second_moment), np.asarray([moment])
    )
    np.testing.assert_array_equal(np.asarray(result.weight_delta), np.asarray([weight_delta]))
    np.testing.assert_array_equal(np.asarray(result.bias_delta), bias_delta)


@pytest.mark.parametrize(
    "bad",
    [np.float32(2.0e20), np.float32(np.inf), np.float32(np.nan)],
)
def test_unrepresentable_nadaline_square_still_rejected(bad: np.float32) -> None:
    """``|x| = 2e20`` makes even the weighted moment overflow float32."""
    result, state = _apply(bad, 1.0, compiled=False)

    assert not bool(result.update_applied)
    np.testing.assert_array_equal(
        np.asarray(result.new_state.feature_second_moment),
        np.asarray(state.feature_second_moment),
    )
    np.testing.assert_array_equal(np.asarray(result.weight_delta), np.zeros(1, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(result.bias_delta), np.float32(0.0))
