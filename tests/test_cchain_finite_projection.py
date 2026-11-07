"""C-CHAIN projection stays finite when the gradient dot overflows.

``cchain_gradient_component`` forms ``(churn · task) / (task · task) * task``.
A task entry of ``1.5e19`` has a finite square, but its dot with a parallel
``3e19`` churn overflows. The projection becomes non-finite, and a zero task
coordinate becomes NaN, so the orthogonal component drops a finite
perpendicular entry. A modest pair keeps the direct projection, and a
non-finite gradient is still rejected.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.benchmarks.cchain_ipmnist import cchain_gradient_component

pytestmark = pytest.mark.unit


def _mlp(w3: list[float]) -> dict[str, jax.Array]:
    row = jnp.asarray(w3, dtype=jnp.float32).reshape(1, -1)
    width = row.shape[1]
    return {
        "w1": jnp.zeros((1, 1), dtype=jnp.float32),
        "b1": jnp.zeros((1,), dtype=jnp.float32),
        "w2": jnp.zeros((1, 1), dtype=jnp.float32),
        "b2": jnp.zeros((1,), dtype=jnp.float32),
        "w3": row,
        "b3": jnp.zeros((width,), dtype=jnp.float32),
    }


def _components(
    task_w3: list[float],
    churn_w3: list[float],
    *,
    compiled: bool,
) -> tuple[jax.Array, jax.Array]:
    task = _mlp(task_w3)
    churn = _mlp(churn_w3)

    def project(
        task_gradient: dict[str, jax.Array],
        churn_gradient: dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        projection = cchain_gradient_component(
            task_gradient, churn_gradient, component="projective"
        )
        orthogonal = cchain_gradient_component(
            task_gradient, churn_gradient, component="orthogonal"
        )
        return projection["w3"], orthogonal["w3"]

    if compiled:
        project = jax.jit(project)
    return project(task, churn)


def _direct(task_w3: list[float], churn_w3: list[float]) -> tuple[jax.Array, jax.Array]:
    task = jnp.asarray(task_w3, dtype=jnp.float32)
    churn = jnp.asarray(churn_w3, dtype=jnp.float32)
    dot = jnp.vdot(churn, task)
    energy = jnp.vdot(task, task)
    scale = dot / jnp.maximum(energy, jnp.asarray(1e-8, dtype=jnp.float32))
    projection = (scale * task).reshape(1, -1)
    return projection, (churn - scale * task).reshape(1, -1)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("task_w3", "churn_w3", "projection_w3", "orthogonal_w3"),
    [
        ([1.5e19, 0.0], [3.0e19, 1.0], [3.0e19, 0.0], [0.0, 1.0]),
        ([-1.5e19, 0.0], [-3.0e19, -1.0], [-3.0e19, 0.0], [0.0, -1.0]),
        ([1.5e19, 0.0], [-3.0e19, 4.0], [-3.0e19, 0.0], [0.0, 4.0]),
    ],
)
def test_overflowing_dot_keeps_the_finite_projection(
    task_w3: list[float],
    churn_w3: list[float],
    projection_w3: list[float],
    orthogonal_w3: list[float],
    compiled: bool,
) -> None:
    projection, orthogonal = _components(task_w3, churn_w3, compiled=compiled)
    assert jnp.array_equal(projection, jnp.asarray(projection_w3, dtype=jnp.float32).reshape(1, -1))
    assert jnp.array_equal(orthogonal, jnp.asarray(orthogonal_w3, dtype=jnp.float32).reshape(1, -1))


@pytest.mark.parametrize("compiled", [False, True])
def test_modest_projection_matches_the_direct_product(compiled: bool) -> None:
    task_w3 = [2.0, -1.0]
    churn_w3 = [-0.5, 3.0]
    projection, orthogonal = _components(task_w3, churn_w3, compiled=compiled)
    expected_projection, expected_orthogonal = _direct(task_w3, churn_w3)
    assert jnp.array_equal(projection, expected_projection)
    assert jnp.array_equal(orthogonal, expected_orthogonal)


def test_nonfinite_gradient_is_rejected() -> None:
    task = _mlp([1.0, -2.0])
    churn = _mlp([0.25, 4.0])
    task["w3"] = task["w3"].at[0, 0].set(jnp.inf)
    with pytest.raises(ValueError, match="finite"):
        cchain_gradient_component(task, churn, component="projective")
