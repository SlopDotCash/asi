"""The supervised-limit wrapper must treat bias as another traced coordinate.

The reference is Table 1 of Mahmood et al. (2012), applied once to an
augmented feature vector. It uses NumPy float64, independently of the JAX
optimizer's separate weight/bias paths and numerical transaction guards.
"""

from dataclasses import fields
from pathlib import Path
from typing import cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from alberta_framework import (
    Autostep,
    AutostepGTDLambda,
    LinearLearner,
    load_checkpoint,
    save_checkpoint,
)
from alberta_framework.core.optimizers import OptimizerUpdate
from alberta_framework.core.types import AutostepGTDLambdaState, AutostepState

Vector = NDArray[np.float64]
pytestmark = pytest.mark.unit


def _reference(
    alpha: Vector, h: Vector, v: Vector, z: Vector, error: float, mu: float, tau: float
) -> tuple[Vector, Vector, Vector, Vector]:
    """One augmented-coordinate Table-1 step with ASI's existing alpha clip."""
    correlation = error * z * h
    v_next = np.maximum(np.abs(correlation), v + alpha * z**2 * (np.abs(correlation) - v) / tau)
    exponent = np.divide(mu * correlation, v_next, out=np.zeros_like(v), where=v_next > 0)
    proposed = np.clip(alpha * np.exp(exponent), 1e-8, 1.0)
    alpha_next = proposed / max(1.0, float(np.dot(proposed, z**2)))
    delta = alpha_next * error * z
    h_next = h * (1.0 - alpha_next * z**2) + delta
    return delta, alpha_next, h_next, v_next


def _coordinates(state: AutostepGTDLambdaState, field: str, bias_field: str) -> Vector:
    return np.append(np.asarray(getattr(state, field)), float(getattr(state, bias_field)))


def _check_result(
    result: OptimizerUpdate, expected: tuple[Vector, Vector, Vector, Vector], z: Vector
) -> AutostepGTDLambdaState:
    assert bool(result.update_applied)
    state = cast(AutostepGTDLambdaState, result.new_state)
    actual = (
        np.append(np.asarray(result.weight_delta), float(result.bias_delta)),
        _coordinates(state, "step_sizes", "bias_step_size"),
        _coordinates(state, "traces", "bias_trace"),
        _coordinates(state, "normalizers", "bias_normalizer"),
    )
    for obtained, wanted in zip(actual, expected, strict=True):
        np.testing.assert_allclose(obtained, wanted, rtol=3e-6, atol=2e-8)
    np.testing.assert_allclose(
        _coordinates(state, "eligibility_traces", "bias_eligibility_trace"), z, rtol=1e-6
    )
    return state


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_bias_delta_uses_accumulated_eligibility(compiled: bool) -> None:
    optimizer = AutostepGTDLambda(initial_step_size=0.1, meta_step_size=0, trace_decay=0.5)
    update = jax.jit(optimizer.update) if compiled else optimizer.update
    state = optimizer.init(1)
    for _ in range(2):
        result = update(state, jnp.float32(1), jnp.ones(1))
        state = cast(AutostepGTDLambdaState, result.new_state)
    np.testing.assert_allclose([state.eligibility_traces[0], state.bias_eligibility_trace], 1.5)
    # alpha=.1, error=1, z=1+.5*1: both coordinates must change by .15.
    np.testing.assert_allclose([result.weight_delta[0], result.bias_delta], 0.15, rtol=1e-6)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_bias_meta_gradient_normalizer_and_h_use_same_coordinate(compiled: bool) -> None:
    optimizer = AutostepGTDLambda(initial_step_size=0.1, meta_step_size=0.2, trace_decay=0.5)
    state = optimizer.init(1).replace(  # type: ignore[attr-defined]
        eligibility_traces=jnp.ones(1),
        bias_eligibility_trace=jnp.float32(1),
        traces=jnp.full(1, 0.1),
        bias_trace=jnp.float32(0.1),
        normalizers=jnp.ones(1),
        bias_normalizer=jnp.float32(1),
    )
    update = jax.jit(optimizer.update) if compiled else optimizer.update
    result = update(state, jnp.float32(1), jnp.ones(1))
    # g=.15; v=1+.1*2.25*(.15-1)/10000=.999980875.
    v = 0.999980875
    alpha = 0.1 * np.exp(0.2 * 0.15 / v)
    delta = 1.5 * alpha
    h = 0.1 * (1 - alpha * 2.25) + delta
    _check_result(
        result,
        (np.full(2, delta), np.full(2, alpha), np.full(2, h), np.full(2, v)),
        np.full(2, 1.5),
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_joint_overshoot_includes_bias_eligibility_squared(compiled: bool) -> None:
    optimizer = AutostepGTDLambda(initial_step_size=0.3, meta_step_size=0, trace_decay=0.5)
    update = jax.jit(optimizer.update) if compiled else optimizer.update
    state = optimizer.init(1)
    state = cast(AutostepGTDLambdaState, update(state, jnp.float32(1), jnp.ones(1)).new_state)
    result = update(state, jnp.float32(1), jnp.ones(1))
    state = cast(AutostepGTDLambdaState, result.new_state)
    # M=.3*(1.5**2+1.5**2)=1.35, alpha'=.3/M=2/9, delta=1/3.
    np.testing.assert_allclose([result.weight_delta[0], result.bias_delta], 1 / 3, rtol=1e-6)
    alpha = _coordinates(state, "step_sizes", "bias_step_size")
    np.testing.assert_allclose(alpha, 2 / 9, rtol=1e-6)
    np.testing.assert_allclose(np.dot(alpha, np.full(2, 1.5) ** 2), 1, rtol=1e-6)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("decay", [0.0, 0.5, 0.9])
def test_multistep_augmented_table_one_reference(compiled: bool, decay: float) -> None:
    optimizer = AutostepGTDLambda(
        initial_step_size=0.3, meta_step_size=0.07, tau=20, trace_decay=decay
    )
    update = jax.jit(optimizer.update) if compiled else optimizer.update
    state = optimizer.init(2)
    alpha, h, v, z = np.full(3, 0.3), np.zeros(3), np.zeros(3), np.zeros(3)
    samples: list[tuple[list[float], float]] = [
        ([1, -1], 1),
        ([0, 2], -0.2),
        ([2, 1], 0),
        ([-1, 0.5], 0.5),
        ([0.25, -0.75], -1.2),
        ([1, 1], 0.3),
    ]
    for x, error in samples:
        z = decay * z + np.array([*x, 1.0])
        expected = _reference(alpha, h, v, z, error, 0.07, 20)
        state = _check_result(update(state, jnp.float32(error), jnp.array(x)), expected, z)
        _, alpha, h, v = expected


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_public_linear_learner_prediction_is_point_four_four(compiled: bool) -> None:
    learner = LinearLearner(
        # LinearLearner's legacy annotation omits this supported optimizer.
        optimizer=AutostepGTDLambda(  # type: ignore[arg-type]
            initial_step_size=0.1, meta_step_size=0, trace_decay=0.5
        )
    )
    state = learner.init(1)
    update = jax.jit(learner.update) if compiled else learner.update
    for _ in range(2):
        result = update(state, jnp.ones(1), jnp.float32(1))
        assert bool(result.update_applied)
        state = result.state
    # First w=b=.1; second error=.8 and delta=.1*.8*1.5=.12 each.
    np.testing.assert_allclose(learner.predict(state, jnp.ones(1)), [0.44], rtol=1e-6)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_public_multistep_predictions_follow_independent_reference(compiled: bool) -> None:
    learner = LinearLearner(
        optimizer=AutostepGTDLambda(  # type: ignore[arg-type]
            initial_step_size=0.2, meta_step_size=0.04, tau=30, trace_decay=0.6
        )
    )
    update = jax.jit(learner.update) if compiled else learner.update
    state = learner.init(2)
    params, z = np.zeros(3), np.zeros(3)
    alpha, h, v = np.full(3, 0.2), np.zeros(3), np.zeros(3)
    samples: list[tuple[list[float], float]] = [
        ([1, -1], 1),
        ([0, 2], 0.5),
        ([2, 1], -0.4),
        ([-1, 0.5], 0.1),
    ]
    for x, target in samples:
        raw = np.array([*x, 1.0])
        prediction = float(params @ raw)
        z = 0.6 * z + raw
        delta, alpha, h, v = _reference(alpha, h, v, z, target - prediction, 0.04, 30)
        params += delta
        result = update(state, jnp.array(x), jnp.float32(target))
        assert bool(result.update_applied)
        np.testing.assert_allclose(result.prediction, [prediction], rtol=3e-6, atol=1e-7)
        state = result.state
        np.testing.assert_allclose(np.append(state.weights, state.bias), params, rtol=3e-6)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_zero_decay_matches_ordinary_autostep_all_common_state(compiled: bool) -> None:
    base = Autostep(initial_step_size=0.1, meta_step_size=0.03)
    traced = AutostepGTDLambda(initial_step_size=0.1, meta_step_size=0.03)
    base_state, state = base.init(2), traced.init(2)
    base_update = jax.jit(base.update) if compiled else base.update
    update = jax.jit(traced.update) if compiled else traced.update
    for x, error in [([1, -1], 1), ([0, 2], -0.2), ([2, 1], 0), ([-1, 0.5], 0.5)]:
        a = base_update(base_state, jnp.float32(error), jnp.array(x, dtype=jnp.float32))
        b = update(state, jnp.float32(error), jnp.array(x, dtype=jnp.float32))
        chex.assert_trees_all_equal(
            (a.weight_delta, a.bias_delta, a.metrics), (b.weight_delta, b.bias_delta, b.metrics)
        )
        for field in fields(base_state):  # type: ignore[arg-type]  # Chex dataclass at runtime.
            chex.assert_trees_all_equal(
                getattr(a.new_state, field.name), getattr(b.new_state, field.name)
            )
        base_state = cast(AutostepState, a.new_state)
        state = cast(AutostepGTDLambdaState, b.new_state)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "fault", ["error", "observation", "bias_eligibility", "bias_h", "overflow"]
)
def test_rejected_update_preserves_all_state_and_can_resume(compiled: bool, fault: str) -> None:
    optimizer = AutostepGTDLambda(initial_step_size=0.1, meta_step_size=0.03, trace_decay=0.5)
    update = jax.jit(optimizer.update) if compiled else optimizer.update
    state = optimizer.init(1)
    state = cast(AutostepGTDLambdaState, update(state, jnp.float32(1), jnp.ones(1)).new_state)
    error, x = jnp.float32(1), jnp.ones(1)
    bad = state
    if fault == "error":
        error = jnp.float32(jnp.nan)
    elif fault == "observation":
        x = jnp.full(1, jnp.inf)
    elif fault == "bias_eligibility":
        bad = state.replace(bias_eligibility_trace=jnp.float32(jnp.inf))  # type: ignore[attr-defined]
    elif fault == "bias_h":
        bad = state.replace(bias_trace=jnp.float32(jnp.inf))  # type: ignore[attr-defined]
    else:
        bad = state.replace(bias_eligibility_trace=jnp.float32(1e30))  # type: ignore[attr-defined]
    result = update(bad, error, x)
    assert not bool(result.update_applied)
    # Check bytes, including non-finite payloads, rather than NaN equality.
    for before, after in zip(jax.tree.leaves(bad), jax.tree.leaves(result.new_state), strict=True):
        assert np.asarray(before).tobytes() == np.asarray(after).tobytes()
    chex.assert_trees_all_equal(result.weight_delta, jnp.zeros(1))
    chex.assert_trees_all_equal(result.bias_delta, jnp.float32(0))
    assert all(float(value) == 0 for value in result.metrics.values())
    # A rejected input must not consume a trace update before the next good input.
    if fault in ("error", "observation"):
        chex.assert_trees_all_equal(
            update(cast(AutostepGTDLambdaState, result.new_state), jnp.float32(0.2), jnp.ones(1)),
            update(state, jnp.float32(0.2), jnp.ones(1)),
        )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_public_rejection_rolls_back_parameters_clock_and_optimizer(compiled: bool) -> None:
    learner = LinearLearner(optimizer=AutostepGTDLambda(trace_decay=0.5))  # type: ignore[arg-type]
    state = learner.init(1)
    update = jax.jit(learner.update) if compiled else learner.update
    state = update(state, jnp.ones(1), jnp.float32(1)).state
    rejected = update(state, jnp.ones(1), jnp.float32(jnp.inf))
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, state)
    chex.assert_trees_all_equal(
        update(rejected.state, jnp.ones(1), jnp.float32(0.2)),
        update(state, jnp.ones(1), jnp.float32(0.2)),
    )


@pytest.mark.integration
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("decay", [0.0, 0.5])
def test_checkpoint_restores_unchanged_schema_and_exact_continuation(
    tmp_path: Path, compiled: bool, decay: float
) -> None:
    learner = LinearLearner(
        optimizer=AutostepGTDLambda(  # type: ignore[arg-type]
            initial_step_size=0.1, meta_step_size=0.03, trace_decay=decay
        )
    )
    update = jax.jit(learner.update) if compiled else learner.update
    state = learner.init(2)
    samples: list[tuple[list[float], float]] = [([1, -1], 1), ([0, 2], -0.2), ([2, 1], 0.4)]
    for x, target in samples:
        state = update(state, jnp.array(x, dtype=jnp.float32), jnp.float32(target)).state
    assert len(jax.tree.leaves(state.optimizer_state)) == 11
    save_checkpoint(state, tmp_path / "checkpoint")
    restored, _ = load_checkpoint(learner.init(2), tmp_path / "checkpoint")
    chex.assert_trees_all_equal(restored, state)
    continuation: list[tuple[list[float], float]] = [([0.5, -1], 0.8), ([1, 1], -0.1)]
    for x, target in continuation:
        a = update(state, jnp.array(x), jnp.float32(target))
        b = update(restored, jnp.array(x), jnp.float32(target))
        assert bool(a.update_applied) and bool(b.update_applied)
        chex.assert_trees_all_equal(a, b)
        state, restored = a.state, b.state
