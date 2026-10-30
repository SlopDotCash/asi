"""Finite GVF TD cancellation for the independent-demon Horde."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework import DemonType, GVFSpec, create_horde_spec
from alberta_framework.core.independent_demon_horde import IndependentDemonHorde


def _horde(step_size: float) -> IndependentDemonHorde:
    spec = create_horde_spec(
        [
            GVFSpec(
                name="v",
                demon_type=DemonType.PREDICTION,
                gamma=1.0,
                lamda=0.0,
                cumulant_index=0,
                terminal_reward=0.0,
            )
        ]
    )
    return IndependentDemonHorde(
        horde_spec=spec,
        hidden_sizes=(),
        step_size=step_size,
        sparsity=0.0,
        use_layer_norm=False,
    )


def _state(horde: IndependentDemonHorde, weight: np.float32) -> object:
    state = horde.init(1, jr.key(0))
    demon = state.demon_states[0]
    return state.replace(
        demon_states=(
            demon.replace(
                params=demon.params.replace(
                    weights=(jnp.asarray([[weight]], dtype=jnp.float32),),
                    biases=(jnp.zeros((1,), dtype=jnp.float32),),
                )
            ),
        )
    )


def _updates(horde: IndependentDemonHorde, state: object, reward: np.float32, next_feature: float):
    observation = jnp.ones((1,), dtype=jnp.float32)
    next_observation = jnp.asarray([next_feature], dtype=jnp.float32)
    cumulant = jnp.asarray([reward], dtype=jnp.float32)
    with jax.disable_jit():
        eager = horde.update(state, observation, cumulant, next_observation)
    compiled = jax.jit(horde.update)(state, observation, cumulant, next_observation)
    return eager, compiled


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_overflowing_cumulant_sum_keeps_the_linear_step(sign: float) -> None:
    """``c + V(s')`` overflows; cancelling it against V(s) leaves ``delta = c``.

    A fresh linear trace equals the feature, so the weight and bias each step
    by ``alpha * c``.
    """
    magnitude = np.float32(sign * 2.0e38)
    alpha = np.float32(0.1)
    step = alpha * np.float32(1.0)
    horde = _horde(0.1)
    state = _state(horde, magnitude)
    expected_weight = np.float32(magnitude + magnitude * step)
    expected_bias = np.float32(magnitude * step)

    for result in _updates(horde, state, magnitude, 1.0):
        assert bool(result.update_applied)
        assert bool(result.head_updates_applied[0])
        np.testing.assert_array_equal(np.asarray(result.td_errors), magnitude)
        demon = result.state.demon_states[0]
        np.testing.assert_array_equal(
            np.asarray(demon.params.weights[0]),
            np.asarray([[expected_weight]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(demon.params.biases[0]),
            np.asarray([expected_bias], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(demon.traces[0]),
            np.ones((1, 1), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(demon.traces[1]),
            np.ones((1,), dtype=np.float32),
        )
        assert int(result.state.step_count) == 1


def test_finite_cumulant_sum_keeps_left_to_right_rounding() -> None:
    """A finite ``(c + V(s')) - V(s)`` keeps that association."""
    reward = np.float32(0.1)
    prediction = np.float32(1.0e10)
    alpha = np.float32(0.1)
    expected_td = np.float32((reward + prediction) - prediction)
    horde = _horde(0.1)
    state = _state(horde, prediction)

    result = _updates(horde, state, reward, 1.0)[0]
    assert bool(result.update_applied)
    np.testing.assert_array_equal(np.asarray(result.td_errors), expected_td)
    demon = result.state.demon_states[0]
    np.testing.assert_array_equal(
        np.asarray(demon.params.weights[0]),
        np.asarray([[np.float32(prediction + alpha * expected_td)]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(demon.params.biases[0]),
        np.asarray([np.float32(alpha * expected_td)], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("next_feature", "step_size"),
    [(0.0, 0.1), (1.0, 1.0)],
)
def test_unrepresentable_independent_horde_step_still_rejects(
    next_feature: float,
    step_size: float,
) -> None:
    """Reject a TD error or weight step that is still outside float32."""
    magnitude = np.float32(2.0e38)
    weight = np.float32(-magnitude if next_feature == 0.0 else magnitude)
    horde = _horde(step_size)
    state = _state(horde, weight)

    for result in _updates(horde, state, magnitude, next_feature):
        assert not bool(result.update_applied)
        assert not bool(result.head_updates_applied[0])
        np.testing.assert_array_equal(np.asarray(result.td_errors), np.float32(0.0))
        demon = result.state.demon_states[0]
        np.testing.assert_array_equal(
            np.asarray(demon.params.weights[0]),
            np.asarray([[weight]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(demon.params.biases[0]),
            np.zeros((1,), dtype=np.float32),
        )
        assert int(result.state.step_count) == 0
