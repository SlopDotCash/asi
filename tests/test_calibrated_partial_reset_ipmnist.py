from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.calibrated_partial_reset_ipmnist import (
    ARMS,
    _make_cpr_learner,
    calibrated_partial_reset_spec,
    persistent_numeric_bytes,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, init_mlp_params

SMALL = IPMNISTConfig(n_tasks=1, task_length=8, input_dim=4, hidden1=3, hidden2=2, n_classes=2)


def _hp() -> dict[str, float]:
    result = dict(calibrated_partial_reset_spec("cpr_utility").hyperparameters)
    result["update_frequency"] = 2.0
    return result


def _trajectory(mode: str) -> dict[str, jnp.ndarray]:
    init_fn, step_fn = _make_cpr_learner(_hp(), mode=mode)  # type: ignore[arg-type]
    params = init_mlp_params(jr.key(7), SMALL)
    state = init_fn(params)
    for index in range(4):
        params, state, _ = step_fn(
            params,
            state,
            jnp.asarray([0.2, -0.1, 0.3, 0.5], dtype=jnp.float32),
            jnp.asarray(index % 2, dtype=jnp.int32),
            jr.key(index),
        )
    return params


def test_registered_arm_roster_and_resource_formula_are_exact() -> None:
    assert tuple(calibrated_partial_reset_spec(name).name for name in ARMS) == ARMS
    parameter_values = 4 * 3 + 3 + 3 * 2 + 2 + 2 * 2 + 2
    assert persistent_numeric_bytes(input_dim=4, hidden1=3, hidden2=2, n_classes=2) == (
        (4 * parameter_values + 3 + 2 + 1) * 4
    )


def test_mechanism_off_is_bit_exact_when_reset_strengths_are_inert() -> None:
    off = _trajectory("off")
    hp = _hp()
    hp["replacement_rate"] = 0.0
    init_fn, step_fn = _make_cpr_learner(hp, mode="utility")
    params = init_mlp_params(jr.key(7), SMALL)
    state = init_fn(params)
    for index in range(4):
        params, state, _ = step_fn(
            params,
            state,
            jnp.asarray([0.2, -0.1, 0.3, 0.5], dtype=jnp.float32),
            jnp.asarray(index % 2, dtype=jnp.int32),
            jr.key(100 + index),
        )
    for name in off:
        np.testing.assert_array_equal(np.asarray(params[name]), np.asarray(off[name]))


@pytest.mark.parametrize("mode", ["utility", "utility_free", "l2_init", "hard_reset"])
def test_each_reduction_changes_the_end_to_end_parameter_trajectory(mode: str) -> None:
    off = _trajectory("off")
    changed = _trajectory(mode)
    assert any(not np.array_equal(np.asarray(off[name]), np.asarray(changed[name])) for name in off)


def test_hard_reset_is_a_full_below_mean_reduction() -> None:
    changed = _trajectory("hard_reset")
    assert all(np.isfinite(np.asarray(value)).all() for value in changed.values())
