"""Hidden RMS stays finite when an activation square overflows.

``_rms_mix`` divides by ``sqrt(mean(h**2) + eps)``. Float32 can represent the
root-mean-square of a ``2e19`` activation, but the square overflows first and
the mix returns zeros. The discovery step then applies weight decay without
the gradient. A modest activation keeps the direct square, and a non-finite
activation stays non-finite.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.rule_discovery import (
    _CHAMPION_CONFIG,
    FLAG_NAMES,
    PARAM_NAMES,
    _rms_mix,
    decode_genome,
    genome_from_config,
    init_rule_state,
    rule_step,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, init_mlp_params

pytestmark = pytest.mark.unit

_EPS = np.float64(1e-8)
_OVERFLOW = np.float32(2.0e19)


def _mix(hidden: jnp.ndarray, enabled: float, *, compiled: bool) -> jnp.ndarray:
    flag = jnp.asarray(enabled, dtype=jnp.float32)

    def apply(hidden: jnp.ndarray, flag: jnp.ndarray) -> jnp.ndarray:
        return _rms_mix(hidden, flag)

    if compiled:
        return jax.jit(apply)(hidden, flag)
    with jax.disable_jit():
        return apply(hidden, flag)


def _finite_rms(hidden: np.ndarray) -> np.ndarray:
    """Scale-safe RMS. A finite direct square uses that value."""

    values = np.asarray(hidden, dtype=np.float64)
    with np.errstate(over="ignore"):
        squares = np.square(np.asarray(hidden, dtype=np.float32)).astype(np.float64)
    direct = np.sqrt(np.mean(squares) + _EPS)
    max_abs = np.max(np.abs(values))
    safe = max_abs if max_abs > 0.0 else 1.0
    scaled = values / safe
    recovered = safe * np.sqrt(np.mean(np.square(scaled)) + _EPS / np.square(safe))
    rms = direct if np.isfinite(direct) else recovered
    return values / rms


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("compiled", [False, True])
def test_overflowing_activation_square_keeps_its_direction(
    sign: float, compiled: bool
) -> None:
    hidden = jnp.asarray([_OVERFLOW, _OVERFLOW, 0.0, 0.0], dtype=jnp.float32) * sign
    mixed = _mix(hidden, 1.0, compiled=compiled)
    expected = _finite_rms(np.asarray(hidden))
    assert bool(np.all(np.isfinite(mixed)))
    np.testing.assert_allclose(np.asarray(mixed), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("compiled", [False, True])
def test_modest_activation_rms_stays_bitwise(compiled: bool) -> None:
    hidden = jnp.asarray([0.5, -1.25, 0.25, 0.0], dtype=jnp.float32)
    mixed = _mix(hidden, 1.0, compiled=compiled)
    direct = hidden / jnp.sqrt(jnp.mean(hidden * hidden) + np.float32(_EPS))
    np.testing.assert_array_equal(np.asarray(mixed), np.asarray(direct))


@pytest.mark.parametrize("compiled", [False, True])
def test_nonfinite_activation_stays_nonfinite(compiled: bool) -> None:
    hidden = jnp.asarray([jnp.inf, 1.0, 0.0, -3.0], dtype=jnp.float32)
    mixed = _mix(hidden, 1.0, compiled=compiled)
    assert not bool(np.all(np.isfinite(mixed)))


def _genome(hidden_rms: float) -> jnp.ndarray:
    config = {name: 0.0 for name in FLAG_NAMES}
    config.update({name: _CHAMPION_CONFIG[name] for name in PARAM_NAMES})
    config["hidden_rms"] = hidden_rms
    return jnp.asarray(genome_from_config(config))


def _scaled_params(scale: float) -> dict[str, jnp.ndarray]:
    config = IPMNISTConfig(
        n_tasks=1, task_length=1, input_dim=4, hidden1=3, hidden2=3, n_classes=2
    )
    params = init_mlp_params(jr.key(0), config)
    return {
        "w1": jnp.full_like(params["w1"], np.float32(scale)),
        "b1": jnp.zeros_like(params["b1"]),
        "w2": jnp.full_like(params["w2"], np.float32(0.5)),
        "b2": jnp.zeros_like(params["b2"]),
        "w3": jnp.full_like(params["w3"], np.float32(0.25)),
        "b3": jnp.zeros_like(params["b3"]),
    }


def _expected_w3(params: dict[str, jnp.ndarray], hidden_rms: float) -> np.ndarray:
    """One SGD step on ``w3`` after a scale-safe hidden mix."""

    decoded = decode_genome(np.asarray(_genome(hidden_rms)))
    lr = np.float64(decoded["lr"])
    weight_decay = np.float64(decoded["weight_decay"])
    x = np.ones((4,), dtype=np.float64)
    w1 = np.asarray(params["w1"], dtype=np.float64)
    w2 = np.asarray(params["w2"], dtype=np.float64)
    w3 = np.asarray(params["w3"], dtype=np.float64)
    hidden = np.maximum(x @ w1, 0.0)
    if hidden_rms:
        hidden = _finite_rms(hidden)
    hidden = np.maximum(hidden @ w2, 0.0)
    if hidden_rms:
        hidden = _finite_rms(hidden)
    logits = hidden @ w3
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    softmax = exp_logits / np.sum(exp_logits)
    grad_logits = softmax - np.asarray([1.0, 0.0])
    grad_w3 = hidden[:, None] * grad_logits[None, :]
    decay = 1.0 - lr * weight_decay
    return w3 * decay - lr * grad_w3


@pytest.mark.parametrize("compiled", [False, True])
def test_discovery_step_uses_the_finite_hidden_scale(compiled: bool) -> None:
    params = _scaled_params(float(_OVERFLOW))
    genome = _genome(1.0)
    state = init_rule_state(params)
    x = jnp.ones((4,), dtype=jnp.float32)
    y = jnp.asarray(0, dtype=jnp.int32)

    def step(params, state, x, y):
        return rule_step(genome, params, state, x, y)

    if compiled:
        new_params, _, _, _ = jax.jit(step)(params, state, x, y)
    else:
        with jax.disable_jit():
            new_params, _, _, _ = step(params, state, x, y)
    expected = _expected_w3(params, 1.0)
    assert bool(np.all(np.isfinite(new_params["w3"])))
    np.testing.assert_allclose(np.asarray(new_params["w3"]), expected, rtol=1e-5, atol=1e-6)
    decay_only = np.asarray(params["w3"], dtype=np.float64) * (
        1.0 - np.float64(decode_genome(np.asarray(genome))["lr"])
        * np.float64(decode_genome(np.asarray(genome))["weight_decay"])
    )
    assert not np.allclose(expected, decay_only, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("compiled", [False, True])
def test_negative_overflow_weights_stay_relu_blocked(compiled: bool) -> None:
    params = _scaled_params(-float(_OVERFLOW))
    genome = _genome(1.0)
    state = init_rule_state(params)
    x = jnp.ones((4,), dtype=jnp.float32)
    y = jnp.asarray(0, dtype=jnp.int32)

    def step(params, state, x, y):
        return rule_step(genome, params, state, x, y)

    if compiled:
        new_params, _, _, _ = jax.jit(step)(params, state, x, y)
    else:
        with jax.disable_jit():
            new_params, _, _, _ = step(params, state, x, y)
    expected = _expected_w3(params, 1.0)
    np.testing.assert_allclose(np.asarray(new_params["w3"]), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("compiled", [False, True])
def test_disabled_rms_keeps_the_raw_activation_step(compiled: bool) -> None:
    params = _scaled_params(0.5)
    genome = _genome(0.0)
    state = init_rule_state(params)
    x = jnp.ones((4,), dtype=jnp.float32)
    y = jnp.asarray(0, dtype=jnp.int32)

    def step(params, state, x, y):
        return rule_step(genome, params, state, x, y)

    if compiled:
        new_params, _, _, _ = jax.jit(step)(params, state, x, y)
    else:
        with jax.disable_jit():
            new_params, _, _, _ = step(params, state, x, y)
    expected = _expected_w3(params, 0.0)
    np.testing.assert_array_equal(np.asarray(new_params["w3"]).shape, expected.shape)
    np.testing.assert_allclose(np.asarray(new_params["w3"]), expected, rtol=1e-5, atol=1e-6)
