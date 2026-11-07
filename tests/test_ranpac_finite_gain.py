"""RanPAC recursive ridge readout keeps a finite gain when ``phi @ pp`` overflows.

``make_frozen_feature_learner`` updates its recursive ridge readout (``method=1``)
with the Sherman-Morrison step ``gain = pp / (1.0 + phi @ pp)`` and
``inverse = state.inverse_covariance - jnp.outer(gain, pp)``. A finite feature
``phi`` of ``2e19`` (or two ``1.5e19`` features whose individual squares fit in
float32) overflows ``1.0 + phi @ pp`` to ``+inf``, collapsing ``gain`` to zero
so ``readout`` and ``inverse_covariance`` silently skip the update while
incrementing ``update_count``. Modest inputs keep the direct float32 arithmetic,
and non-finite or singular updates are still rejected.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.benchmarks.replay_frozen_ipmnist import (
    FROZEN_FEATURE_DIM,
    FrozenFeatureState,
    frozen_hyperparameters,
    make_frozen_feature_learner,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, init_mlp_params

pytestmark = pytest.mark.unit


def _config() -> IPMNISTConfig:
    return IPMNISTConfig(
        n_tasks=1, task_length=4, input_dim=4, hidden1=3, hidden2=2, n_classes=2
    )


def _setup() -> tuple[dict[str, jax.Array], FrozenFeatureState, Any]:
    params = init_mlp_params(jr.key(0), _config())
    hp = frozen_hyperparameters(method=1.0, mechanism=1.0)
    init_fn, step_fn = make_frozen_feature_learner(hp)
    return params, init_fn(params), step_fn


def _replace_state(state: FrozenFeatureState, **changes: jax.Array) -> FrozenFeatureState:
    return cast(FrozenFeatureState, dataclasses.replace(cast(Any, state), **changes))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("x_values", "proj_rows", "label", "expected_gain", "expected_inv_block"),
    [
        (
            [2.0e19, 0.0, 0.0, 0.0],
            [(0, 0, 1.0)],
            0,
            [5.0e-20, 0.0],
            [[0.0, 0.0], [0.0, 1.0]],
        ),
        (
            [-2.0e19, 0.0, 0.0, 0.0],
            [(0, 0, -1.0)],
            1,
            [5.0e-20, 0.0],
            [[0.0, 0.0], [0.0, 1.0]],
        ),
        (
            [1.5e19, 1.5e19, 0.0, 0.0],
            [(0, 0, 1.0), (1, 1, 1.0)],
            0,
            [1.0 / 3.0e19, 1.0 / 3.0e19],
            [[0.5, -0.5], [-0.5, 0.5]],
        ),
    ],
)
def test_overflowing_quadratic_form_keeps_finite_sherman_morrison_update(
    x_values: list[float],
    proj_rows: list[tuple[int, int, float]],
    label: int,
    expected_gain: list[float],
    expected_inv_block: list[list[float]],
    compiled: bool,
) -> None:
    params, state, step_fn = _setup()
    projection = jnp.zeros((FROZEN_FEATURE_DIM, 4), dtype=jnp.float32)
    for row, col, value in proj_rows:
        projection = projection.at[row, col].set(value)
    state = _replace_state(state, projection=projection)
    x = jnp.asarray(x_values, dtype=jnp.float32)
    y = jnp.asarray(label, dtype=jnp.int32)
    run = jax.jit(step_fn) if compiled else step_fn
    _, result, (_, _, plasticity) = run(params, state, x, y, jr.key(1))

    target = jax.nn.one_hot(y, 2, dtype=jnp.float32)
    expected_readout_top = jnp.outer(
        jnp.asarray(expected_gain, dtype=jnp.float32), target
    )
    assert jnp.allclose(result.readout[:2], expected_readout_top, rtol=1e-6, atol=0.0)
    assert jnp.array_equal(
        result.inverse_covariance[:2, :2],
        jnp.asarray(expected_inv_block, dtype=jnp.float32),
    )
    phi = jax.nn.relu(result.projection @ x)
    logits_after = result.readout.T @ phi
    assert jnp.allclose(logits_after, target, rtol=1e-6, atol=1e-6)
    assert float(plasticity) > 0.5
    assert int(result.update_count) == 1


@pytest.mark.parametrize("compiled", [False, True])
def test_modest_quadratic_form_matches_direct_update(compiled: bool) -> None:
    params, state, step_fn = _setup()
    projection = (
        jnp.zeros((FROZEN_FEATURE_DIM, 4), dtype=jnp.float32)
        .at[0, 0]
        .set(1.0)
        .at[1, 1]
        .set(2.0)
    )
    state = _replace_state(state, projection=projection)
    x = jnp.asarray([1.0, 0.5, -1.0, 1.5], dtype=jnp.float32)
    y = jnp.asarray(1, dtype=jnp.int32)
    phi = jax.nn.relu(state.projection @ x)
    logits = state.readout.T @ phi
    target = jax.nn.one_hot(y, 2, dtype=jnp.float32)
    pp = state.inverse_covariance @ phi
    direct_gain = pp / (1.0 + phi @ pp)
    expected_readout = state.readout + jnp.outer(direct_gain, target - logits)
    direct_inverse = state.inverse_covariance - jnp.outer(direct_gain, pp)
    expected_inverse = 0.5 * (direct_inverse + direct_inverse.T)

    run = jax.jit(step_fn) if compiled else step_fn
    _, result, _ = run(params, state, x, y, jr.key(2))
    assert jnp.array_equal(result.readout, expected_readout)
    assert jnp.array_equal(result.inverse_covariance, expected_inverse)


@pytest.mark.parametrize("compiled", [False, True])
def test_nonfinite_and_singular_updates_are_rejected(compiled: bool) -> None:
    params, state, step_fn = _setup()
    run = jax.jit(step_fn) if compiled else step_fn

    nonfinite_x = jnp.asarray([jnp.inf, 0.0, 0.0, 0.0], dtype=jnp.float32)
    _, rejected_nonfinite, _ = run(
        params, state, nonfinite_x, jnp.asarray(0, dtype=jnp.int32), jr.key(3)
    )
    assert int(rejected_nonfinite.update_count) == 0
    assert jnp.array_equal(rejected_nonfinite.readout, state.readout)
    assert jnp.array_equal(
        rejected_nonfinite.inverse_covariance, state.inverse_covariance
    )

    projection = jnp.zeros((FROZEN_FEATURE_DIM, 4), dtype=jnp.float32).at[0, 0].set(1.0)
    singular_state = _replace_state(
        state,
        projection=projection,
        inverse_covariance=state.inverse_covariance.at[0, 0].set(-1.0),
    )
    unit_x = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    _, rejected_singular, _ = run(
        params, singular_state, unit_x, jnp.asarray(0, dtype=jnp.int32), jr.key(4)
    )
    assert int(rejected_singular.update_count) == 0
    assert jnp.array_equal(rejected_singular.readout, singular_state.readout)
    assert jnp.array_equal(
        rejected_singular.inverse_covariance, singular_state.inverse_covariance
    )
