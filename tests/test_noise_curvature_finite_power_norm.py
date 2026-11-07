"""Finite power-iteration norm for the noise-curvature controller."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.noise_curvature_ipmnist import (
    NoiseCurvatureConfig,
    NoiseCurvatureState,
    _layer_l2_norm,
    init_noise_curvature_state,
    noise_curvature_step,
)


def _zero_params() -> dict[str, jax.Array]:
    return {
        "w1": jnp.zeros((1, 1), dtype=jnp.float32),
        "b1": jnp.zeros((1,), dtype=jnp.float32),
        "w2": jnp.zeros((1, 1), dtype=jnp.float32),
        "b2": jnp.zeros((1,), dtype=jnp.float32),
        "w3": jnp.zeros((1, 1), dtype=jnp.float32),
        "b3": jnp.zeros((1,), dtype=jnp.float32),
    }


def _config() -> NoiseCurvatureConfig:
    return NoiseCurvatureConfig(
        mode="fixed",
        total_steps=1,
        control_interval=1,
        power_iterations=1,
        weight_decay=0.0,
    )


def _controller_step(
    loss_fn: object,
    power_w1: jax.Array,
    power_b1: jax.Array,
) -> NoiseCurvatureState:
    params = _zero_params()
    config = _config()
    state = init_noise_curvature_state(params, config)
    power = dict(state.power_vectors)
    power["w1"] = power_w1
    power["b1"] = power_b1
    state = state.replace(power_vectors=power)
    grads = {name: jnp.zeros_like(value) for name, value in params.items()}
    x = jnp.zeros((1,), dtype=jnp.float32)
    y = jnp.asarray(0, dtype=jnp.int32)

    def step(
        candidate: dict[str, jax.Array], current: NoiseCurvatureState
    ) -> tuple[dict[str, jax.Array], NoiseCurvatureState]:
        return noise_curvature_step(candidate, current, grads, x, y, loss_fn, config)

    _params, eager_state = step(params, state)
    _jit_params, jit_state = jax.jit(step)(params, state)
    assert int(eager_state.diagnostic_failures) == 0
    assert int(eager_state.controller_count) == 1
    assert all(
        bool(jnp.array_equal(eager_state.power_vectors[name], jit_state.power_vectors[name]))
        for name in params
    )
    return eager_state


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_power_iteration_normalizes_when_a_component_square_overflows(sign: float) -> None:
    scale = jnp.asarray(sign * 2.0e19, dtype=jnp.float32)

    def loss_fn(
        candidate: dict[str, jax.Array], _x: jax.Array, _y: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        value = scale * candidate["w1"][0, 0] * candidate["b1"][0]
        return value, value

    state = _controller_step(
        loss_fn,
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
    )
    assert jnp.array_equal(state.power_vectors["w1"], jnp.asarray([[sign]], dtype=jnp.float32))
    assert jnp.array_equal(state.power_vectors["b1"], jnp.zeros((1,), dtype=jnp.float32))


def test_power_iteration_keeps_the_direct_norm_for_modest_curvature() -> None:
    def loss_fn(
        candidate: dict[str, jax.Array], _x: jax.Array, _y: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        weight = candidate["w1"][0, 0]
        bias = candidate["b1"][0]
        value = 0.5 * 3.0 * jnp.square(weight) + 0.5 * 4.0 * jnp.square(bias)
        return value, value

    state = _controller_step(
        loss_fn,
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
    )
    hvp_w = jnp.asarray(3.0, dtype=jnp.float32)
    hvp_b = jnp.asarray(4.0, dtype=jnp.float32)
    direct = jnp.sqrt(jnp.asarray(0.0, dtype=jnp.float32) + jnp.square(hvp_w) + jnp.square(hvp_b))
    floor = jnp.asarray(1e-30, dtype=jnp.float32)
    unit_w = jnp.reshape(hvp_w / jnp.maximum(direct, floor), (1, 1))
    unit_b = jnp.reshape(hvp_b / jnp.maximum(direct, floor), (1,))
    assert jnp.array_equal(state.power_vectors["w1"], unit_w)
    assert jnp.array_equal(state.power_vectors["b1"], unit_b)


def test_power_iteration_leaves_the_vector_when_the_norm_does_not_fit() -> None:
    big = jnp.asarray(3.0e38, dtype=jnp.float32)

    def loss_fn(
        candidate: dict[str, jax.Array], _x: jax.Array, _y: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        weight = candidate["w1"][0, 0]
        bias = candidate["b1"][0]
        value = 0.5 * big * jnp.square(weight) - 0.5 * big * jnp.square(bias)
        return value, value

    state = _controller_step(
        loss_fn,
        jnp.ones((1, 1), dtype=jnp.float32),
        -jnp.ones((1,), dtype=jnp.float32),
    )
    assert jnp.array_equal(state.power_vectors["w1"], jnp.ones((1, 1), dtype=jnp.float32))
    assert jnp.array_equal(state.power_vectors["b1"], -jnp.ones((1,), dtype=jnp.float32))


def _layer_pair(weight: float, bias: float) -> dict[str, jax.Array]:
    return {
        "w1": jnp.asarray([[weight]], dtype=jnp.float32),
        "b1": jnp.asarray([bias], dtype=jnp.float32),
    }


def _float64_l2(weight: float, bias: float) -> np.float32:
    values = np.asarray([weight, bias], dtype=np.float32).astype(np.float64)
    return np.float32(np.sqrt(np.sum(np.square(values))))


def test_layer_l2_norm_matches_the_direct_square_root_when_it_is_finite() -> None:
    tree = _layer_pair(3.0, 4.0)
    expected = jnp.asarray(_float64_l2(3.0, 4.0))
    eager = _layer_l2_norm(tree, ("w1", "b1"))
    compiled = jax.jit(lambda candidate: _layer_l2_norm(candidate, ("w1", "b1")))(tree)
    assert jnp.array_equal(eager, expected)
    assert jnp.array_equal(compiled, expected)
    zeros = _layer_pair(0.0, 0.0)
    assert jnp.array_equal(
        _layer_l2_norm(zeros, ("w1", "b1")),
        jnp.asarray(0.0, dtype=jnp.float32),
    )


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_layer_l2_norm_recovers_one_overflowing_square(sign: float) -> None:
    tree = _layer_pair(sign * 2.0e19, 0.0)
    expected = jnp.asarray(2.0e19, dtype=jnp.float32)
    eager = _layer_l2_norm(tree, ("w1", "b1"))
    compiled = jax.jit(lambda candidate: _layer_l2_norm(candidate, ("w1", "b1")))(tree)
    assert jnp.array_equal(eager, expected)
    assert jnp.array_equal(compiled, expected)
    assert bool(jnp.isfinite(eager))


def test_layer_l2_norm_matches_float64_when_several_squares_overflow() -> None:
    tree = _layer_pair(2.0e19, -2.0e19)
    expected = jnp.asarray(_float64_l2(2.0e19, -2.0e19))
    assert bool(np.isfinite(expected))
    eager = _layer_l2_norm(tree, ("w1", "b1"))
    compiled = jax.jit(lambda candidate: _layer_l2_norm(candidate, ("w1", "b1")))(tree)
    assert jnp.array_equal(eager, expected)
    assert jnp.array_equal(compiled, expected)


def test_layer_l2_norm_stays_nonfinite_when_the_recovered_norm_overflows() -> None:
    tree = _layer_pair(3.0e38, 3.0e38)
    eager = _layer_l2_norm(tree, ("w1", "b1"))
    compiled = jax.jit(lambda candidate: _layer_l2_norm(candidate, ("w1", "b1")))(tree)
    assert not bool(jnp.isfinite(eager))
    assert not bool(jnp.isfinite(compiled))


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_layer_l2_norm_does_not_repair_a_nonfinite_entry(bad: float) -> None:
    tree = _layer_pair(bad, 2.0e19)
    eager = _layer_l2_norm(tree, ("w1", "b1"))
    compiled = jax.jit(lambda candidate: _layer_l2_norm(candidate, ("w1", "b1")))(tree)
    assert not bool(jnp.isfinite(eager))
    assert not bool(jnp.isfinite(compiled))
