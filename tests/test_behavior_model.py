"""Tests for the online behavior/action prediction model."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import tree_util


class _HostileFloat(float):
    def as_integer_ratio(self) -> tuple[int, int]:
        raise AssertionError("scalar hook must not run")


class _HostileDict(dict[str, Any]):
    def __iter__(self):
        raise AssertionError("mapping hook must not run")

try:
    from alberta_framework.core.behavior_model import (
        _BEHAVIOR_MODEL_SEQUENCE_MAX_STEPS,
        BehaviorModel,
        BehaviorModelConfig,
        _behavior_model_update_working_set_bytes,
        _preflight_behavior_model_update_working_set,
        _require_behavior_model_sequence_length,
        _resource_counts,
        action_log_likelihoods,
        clipped_importance_ratios,
        epsilon_greedy_probabilities,
        floor_and_renormalize_probabilities,
        run_behavior_model_from_arrays,
        selected_action_probabilities,
    )
except ImportError:
    # Other in-flight Step 8/world-model lanes can temporarily break package
    # imports. Keep this focused behavior-model test runnable without touching
    # those files.
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "alberta_framework"
        / "core"
        / "behavior_model.py"
    )
    spec = importlib.util.spec_from_file_location(
        "alberta_framework_behavior_model_under_test",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise
    behavior_model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(behavior_model_module)
    BehaviorModel = behavior_model_module.BehaviorModel
    BehaviorModelConfig = behavior_model_module.BehaviorModelConfig
    action_log_likelihoods = behavior_model_module.action_log_likelihoods
    clipped_importance_ratios = behavior_model_module.clipped_importance_ratios
    epsilon_greedy_probabilities = behavior_model_module.epsilon_greedy_probabilities
    floor_and_renormalize_probabilities = behavior_model_module.floor_and_renormalize_probabilities
    run_behavior_model_from_arrays = behavior_model_module.run_behavior_model_from_arrays
    selected_action_probabilities = behavior_model_module.selected_action_probabilities
    _behavior_model_update_working_set_bytes = (
        behavior_model_module._behavior_model_update_working_set_bytes
    )
    _preflight_behavior_model_update_working_set = (
        behavior_model_module._preflight_behavior_model_update_working_set
    )
    _resource_counts = behavior_model_module._resource_counts
    _BEHAVIOR_MODEL_SEQUENCE_MAX_STEPS = (
        behavior_model_module._BEHAVIOR_MODEL_SEQUENCE_MAX_STEPS
    )
    _require_behavior_model_sequence_length = (
        behavior_model_module._require_behavior_model_sequence_length
    )


def _assert_behavior_update_finite(result: Any) -> None:
    """Check numeric leaves while handling JAX typed PRNG keys explicitly."""
    chex.assert_tree_all_finite(
        (
            result.state.weights,
            result.state.bias,
            jax.random.key_data(result.state.rng_key),
            result.state.step_count,
            result.state.nll_ema,
            result.state.accuracy_ema,
            result.state.confidence_ema,
            result.logits,
            result.probabilities,
            result.action_probability,
            result.log_likelihood,
            result.loss,
            result.entropy,
            result.confidence,
            result.predicted_action,
            result.correct,
        )
    )


def test_init_predict_update_finite_and_shapes() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=3, step_size=0.1))
    state = model.init(feature_dim=4, key=jax.random.key(0))
    obs = jnp.array([1.0, -1.0, 0.5, 2.0], dtype=jnp.float32)

    logits = model.predict_logits(state, obs)
    probs = model.predict_probabilities(state, obs)
    result = model.update(state, obs, jnp.array(2, dtype=jnp.int32))

    chex.assert_shape(logits, (3,))
    chex.assert_shape(probs, (3,))
    chex.assert_shape(result.probabilities, (3,))
    chex.assert_shape(result.action_probability, ())
    _assert_behavior_update_finite(result)
    assert int(result.state.step_count) == 1
    assert float(result.loss) > 0.0


def test_infinite_observation_does_not_poison_weights() -> None:
    """Inf obs makes softmax NaN and logit_error * x = 0*inf on a silent feature."""
    model = BehaviorModel(BehaviorModelConfig(n_actions=2, step_size=0.1))
    state = model.init(feature_dim=2, key=jax.random.key(0))
    obs = jnp.array([0.0, jnp.inf], dtype=jnp.float32)
    action = jnp.array(0, dtype=jnp.int32)

    poisoned = model.update(state, obs, action)
    chex.assert_trees_all_close(poisoned.state.weights, state.weights)
    chex.assert_trees_all_close(poisoned.state.bias, state.bias)
    assert int(poisoned.state.step_count) == int(state.step_count)
    assert not bool(poisoned.update_applied)
    assert float(poisoned.loss) == 0.0
    assert int(poisoned.predicted_action) == 0
    chex.assert_trees_all_close(poisoned.probabilities, jnp.zeros_like(poisoned.probabilities))

    recovered = model.update(
        poisoned.state,
        jnp.array([0.0, 1.0], dtype=jnp.float32),
        action,
    )
    chex.assert_tree_all_finite(recovered.state.weights)
    chex.assert_tree_all_finite(recovered.state.bias)
    assert bool(recovered.update_applied)


def test_zero_diagnostic_decay_does_not_multiply_inf_ema() -> None:
    """decay=0 times an infinite diagnostic EMA is NaN and would reject a finite sample."""
    model = BehaviorModel(
        BehaviorModelConfig(n_actions=2, step_size=0.1, diagnostic_decay=0.0)
    )
    state = model.init(feature_dim=2, key=jax.random.key(0))
    obs = jnp.array([0.5, -0.25], dtype=jnp.float32)
    action = jnp.array(0, dtype=jnp.int32)
    state = model.update(state, obs, action).state
    state = state.replace(  # type: ignore[attr-defined]
        nll_ema=jnp.asarray(jnp.inf, dtype=jnp.float32),
        accuracy_ema=jnp.asarray(jnp.inf, dtype=jnp.float32),
        confidence_ema=jnp.asarray(jnp.inf, dtype=jnp.float32),
    )
    raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
    assert not bool(jnp.isfinite(raw))

    result = model.update(state, obs, action)
    assert bool(result.update_applied)
    chex.assert_tree_all_finite(result.state.nll_ema)
    chex.assert_tree_all_finite(result.state.accuracy_ema)
    chex.assert_tree_all_finite(result.state.confidence_ema)
    chex.assert_trees_all_close(result.state.nll_ema, result.loss)
    chex.assert_trees_all_close(result.state.accuracy_ema, result.correct)
    chex.assert_trees_all_close(result.state.confidence_ema, result.confidence)


def test_infinite_observation_marks_input_loss_gradient_invalid() -> None:
    """The state-builder bridge returns a neutral payload with a false verdict.

    Inf observation makes softmax NaN and ``W.T @ (p - one_hot)`` non-finite,
    even on a silent feature coordinate. The explicit verdict prevents the
    neutral zero gradient from being mistaken for a valid training signal.
    """
    model = BehaviorModel(BehaviorModelConfig(n_actions=2, step_size=0.1))
    state = model.init(feature_dim=2, key=jax.random.key(0))
    obs = jnp.array([0.0, jnp.inf], dtype=jnp.float32)

    result = jax.jit(model.input_loss_gradient)(state, obs, jnp.array(0, dtype=jnp.int32))

    chex.assert_tree_all_finite(result)
    assert not bool(result.valid)
    chex.assert_trees_all_close(result.gradient, jnp.zeros(2, dtype=jnp.float32))
    assert float(result.loss) == 0.0
    assert float(result.gradient_norm) == 0.0


def test_input_loss_gradient_verdict_is_consistent_eager_jit_and_vmap() -> None:
    """The traced verdict gates mutation in every supported transform."""
    model = BehaviorModel(BehaviorModelConfig(n_actions=2, step_size=0.1))
    state = model.init(feature_dim=2, key=jax.random.key(1))
    valid_obs = jnp.array([0.25, -0.5], dtype=jnp.float32)
    invalid_obs = jnp.array([0.25, jnp.inf], dtype=jnp.float32)
    action = jnp.array(1, dtype=jnp.int32)
    builder_state = jnp.array([3.0, -2.0], dtype=jnp.float32)

    def consume(observation: jax.Array) -> tuple[Any, jax.Array]:
        result = model.input_loss_gradient(state, observation, action)
        proposal = builder_state - jnp.asarray(0.1, dtype=jnp.float32) * result.gradient
        committed = jnp.where(result.valid, proposal, builder_state)
        return result, committed

    with jax.disable_jit():
        eager_valid, eager_committed = consume(valid_obs)
        eager_invalid, eager_rejected = consume(invalid_obs)
    compiled_valid, compiled_committed = jax.jit(consume)(valid_obs)
    compiled_invalid, compiled_rejected = jax.jit(consume)(invalid_obs)
    batched_results, batched_committed = jax.vmap(consume)(
        jnp.stack([valid_obs, invalid_obs])
    )

    assert bool(eager_valid.valid)
    assert bool(compiled_valid.valid)
    assert bool(batched_results.valid[0])
    assert not bool(eager_invalid.valid)
    assert not bool(compiled_invalid.valid)
    assert not bool(batched_results.valid[1])
    chex.assert_trees_all_close(eager_valid, compiled_valid)
    chex.assert_trees_all_close(eager_valid, jax.tree.map(lambda x: x[0], batched_results))
    chex.assert_trees_all_close(eager_committed, compiled_committed)
    chex.assert_trees_all_close(eager_committed, batched_committed[0])
    chex.assert_trees_all_close(eager_rejected, builder_state)
    chex.assert_trees_all_close(compiled_rejected, builder_state)
    chex.assert_trees_all_close(batched_committed[1], builder_state)


def test_input_loss_gradient_rejects_finite_gradient_with_overflowed_norm() -> None:
    """The verdict covers every numeric field, including derived diagnostics."""
    model = BehaviorModel(BehaviorModelConfig(n_actions=2, step_size=0.1))
    state = model.init(feature_dim=2, key=jax.random.key(2))
    maximum = jnp.finfo(jnp.float32).max
    state = state.replace(  # type: ignore[attr-defined]
        weights=jnp.array(
            [[maximum, maximum], [-maximum, -maximum]],
            dtype=jnp.float32,
        ),
        bias=jnp.zeros((2,), dtype=jnp.float32),
    )

    result = model.input_loss_gradient(
        state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
    )

    assert not bool(result.valid)
    chex.assert_tree_all_finite(result)
    chex.assert_trees_all_close(result.gradient, jnp.zeros((2,), dtype=jnp.float32))
    assert float(result.gradient_norm) == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("step_size", float("nan")),
        ("step_size", float("inf")),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("l2_penalty", float("nan")),
        ("max_gradient_norm", float("inf")),
        ("min_probability", float("nan")),
        ("min_probability", 1.0),
        ("ratio_clip", float("inf")),
        ("diagnostic_decay", float("nan")),
    ],
)
def test_config_rejects_nonfinite_or_invalid_numeric_values(
    field: str,
    value: float,
) -> None:
    kwargs: dict[str, Any] = {"n_actions": 2, field: value}
    with pytest.raises(ValueError):
        BehaviorModelConfig(**kwargs)


def test_config_and_init_reject_boolean_or_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        BehaviorModelConfig(n_actions=True)
    model = BehaviorModel(BehaviorModelConfig(n_actions=2))
    for feature_dim in (True, 0, -1):
        with pytest.raises(ValueError, match="feature_dim"):
            model.init(feature_dim=feature_dim, key=jax.random.key(0))
        with pytest.raises(ValueError, match="feature_dim"):
            model.resource_budget(feature_dim)


def test_resource_budget_matches_initialized_state_arrays_exactly() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=3))
    feature_dim = 5
    state = model.init(feature_dim=feature_dim, key=jax.random.key(0))
    budget = model.resource_budget(feature_dim)
    actual_nbytes = sum(int(leaf.nbytes) for leaf in jax.tree_util.tree_leaves(state))

    assert budget.feature_dim == feature_dim
    assert budget.n_actions == 3
    assert budget.trainable_float32_scalars == 3 * 5 + 3
    assert budget.diagnostic_float32_scalars == 3
    assert budget.administrative_int32_scalars == 1
    assert budget.rng_uint32_scalars == 2
    assert budget.state_nbytes == actual_nbytes
    assert budget.learned_float32_scalars_touched_per_update == 3 * 5 + 3 + 3
    assert budget.replay_capacity == 0
    assert budget.to_dict()["state_nbytes"] == actual_nbytes


def test_preupdate_input_gradient_matches_autodiff_and_does_not_advance_state() -> None:
    model = BehaviorModel(
        BehaviorModelConfig(
            n_actions=3,
            step_size=0.1,
            temperature=0.7,
        )
    )
    state = model.init(feature_dim=4, key=jax.random.key(7)).replace(
        weights=jnp.array(
            [
                [0.2, -0.3, 0.5, 0.1],
                [-0.4, 0.6, 0.2, -0.5],
                [0.1, 0.4, -0.2, 0.3],
            ],
            dtype=jnp.float32,
        ),
        bias=jnp.array([0.1, -0.2, 0.05], dtype=jnp.float32),
    )
    observation = jnp.array([0.5, -1.0, 0.25, 2.0], dtype=jnp.float32)
    action = jnp.array(1, dtype=jnp.int32)

    def loss_fn(features):
        logits = (state.weights @ features + state.bias) / 0.7
        return -jax.nn.log_softmax(logits)[action]

    expected_loss, expected_gradient = jax.value_and_grad(loss_fn)(observation)
    before = jax.tree_util.tree_map(lambda value: value.copy(), state)
    result = jax.jit(model.input_loss_gradient)(state, observation, action)

    chex.assert_trees_all_close(result.loss, expected_loss, atol=1e-7, rtol=1e-6)
    chex.assert_trees_all_close(
        result.gradient,
        expected_gradient,
        atol=1e-7,
        rtol=1e-6,
    )
    chex.assert_trees_all_close(
        result.gradient_norm,
        jnp.linalg.norm(expected_gradient),
        atol=1e-7,
        rtol=1e-6,
    )
    chex.assert_trees_all_equal(state, before)
    assert int(state.step_count) == 0


def test_probability_simplex_and_helper_invariants() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=4))
    state = model.init(feature_dim=2, key=jax.random.key(1))
    probs = model.predict_probabilities(
        state,
        jnp.array([10.0, -3.0], dtype=jnp.float32),
    )
    floored = floor_and_renormalize_probabilities(
        jnp.array([0.0, 0.2, 0.3, 0.5], dtype=jnp.float32),
        min_probability=0.01,
    )

    chex.assert_trees_all_close(jnp.sum(probs), 1.0, atol=1e-6)
    chex.assert_trees_all_close(jnp.sum(floored), 1.0, atol=1e-6)
    assert float(jnp.min(floored)) >= 0.01 - 1e-7

    selected = selected_action_probabilities(
        jnp.array([[0.2, 0.8], [0.9, 0.1]], dtype=jnp.float32),
        jnp.array([1, 0], dtype=jnp.int32),
    )
    logs = action_log_likelihoods(
        jnp.array([[0.2, 0.8], [0.9, 0.1]], dtype=jnp.float32),
        jnp.array([1, 0], dtype=jnp.int32),
    )
    chex.assert_trees_all_close(selected, jnp.array([0.8, 0.9], dtype=jnp.float32))
    chex.assert_trees_all_close(logs, jnp.log(selected))


@pytest.mark.parametrize(
    "action",
    [
        jnp.array(1.9, dtype=jnp.float32),
        jnp.array(True, dtype=jnp.bool_),
    ],
)
def test_update_rejects_non_integer_action_dtypes(action: jax.Array) -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=3, step_size=0.1))
    state = model.init(feature_dim=2, key=jax.random.key(0))
    observation = jnp.array([1.0, -0.5], dtype=jnp.float32)

    with pytest.raises(ValueError, match="actions must have an integer dtype"):
        model.update(state, observation, action)


@pytest.mark.parametrize(
    "action",
    (
        np.int64(2**32),
        np.uint64(2**32),
        np.int64(-1),
    ),
)
def test_behavior_model_rejects_original_width_action_before_jax_narrowing(
    action: object,
) -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=3, step_size=0.1))
    state = model.init(feature_dim=2, key=jax.random.key(0))
    observation = jnp.array([1.0, -0.5], dtype=jnp.float32)
    probabilities = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32)

    with pytest.raises(ValueError, match="actions must lie"):
        model.update(state, observation, action)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="actions must lie"):
        model.input_loss_gradient(state, observation, action)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="actions must lie"):
        selected_action_probabilities(probabilities, action)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="actions must lie"):
        run_behavior_model_from_arrays(
            model,
            state,
            observations=observation[None, :],
            actions=np.asarray([action]),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("action", (-1, 3, 2**31 - 1))
def test_behavior_model_staged_invalid_actions_are_atomic_and_neutral(action: int) -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=3, step_size=0.1))
    state = model.init(feature_dim=2, key=jax.random.key(0))
    observation = jnp.array([1.0, -0.5], dtype=jnp.float32)

    staged_update = jax.jit(lambda current, selected: model.update(current, observation, selected))
    result = staged_update(state, jnp.asarray(action, dtype=jnp.int32))
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.probabilities, jnp.zeros((3,), dtype=jnp.float32))
    assert float(result.loss) == 0.0

    staged_gradient = jax.jit(
        lambda current, selected: model.input_loss_gradient(current, observation, selected)
    )
    gradient = staged_gradient(state, jnp.asarray(action, dtype=jnp.int32))
    assert not bool(gradient.valid)
    chex.assert_trees_all_equal(gradient.gradient, jnp.zeros((2,), dtype=jnp.float32))
    assert float(gradient.loss) == 0.0


def test_likelihood_improves_on_deterministic_policy_stream() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=2, step_size=0.2, diagnostic_decay=0.9))
    state = model.init(feature_dim=2, key=jax.random.key(2))
    obs0 = jnp.array([1.0, 0.0], dtype=jnp.float32)
    obs1 = jnp.array([0.0, 1.0], dtype=jnp.float32)

    start_p0 = model.action_probability(state, obs0, jnp.array(0, dtype=jnp.int32))
    start_p1 = model.action_probability(state, obs1, jnp.array(1, dtype=jnp.int32))
    for _ in range(160):
        state = model.update(state, obs0, jnp.array(0, dtype=jnp.int32)).state
        state = model.update(state, obs1, jnp.array(1, dtype=jnp.int32)).state

    end_p0 = model.action_probability(state, obs0, jnp.array(0, dtype=jnp.int32))
    end_p1 = model.action_probability(state, obs1, jnp.array(1, dtype=jnp.int32))

    assert float(end_p0) > float(start_p0) + 0.35
    assert float(end_p1) > float(start_p1) + 0.35
    assert float(end_p0) > 0.85
    assert float(end_p1) > 0.85
    assert float(state.accuracy_ema) > 0.85


def test_scan_loop_and_jit_compatibility() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=3, step_size=0.05))
    state = model.init(feature_dim=3, key=jax.random.key(3))
    observations = jnp.eye(3, dtype=jnp.float32).repeat(4, axis=0)
    actions = jnp.array([0, 1, 2] * 4, dtype=jnp.int32)

    jitted_update = jax.jit(model.update)
    update_result = jitted_update(state, observations[0], actions[0])
    result = run_behavior_model_from_arrays(
        model,
        state,
        observations,
        actions,
    )

    _assert_behavior_update_finite(update_result)
    chex.assert_shape(result.probabilities, (12, 3))
    chex.assert_shape(result.action_probabilities, (12,))
    chex.assert_shape(result.log_likelihoods, (12,))
    chex.assert_shape(result.correct, (12,))
    assert int(result.state.step_count) == 12


def test_config_roundtrip_and_sampling() -> None:
    model = BehaviorModel(
        BehaviorModelConfig(
            n_actions=3,
            step_size=0.03,
            temperature=0.8,
            l2_penalty=0.01,
            max_gradient_norm=1.5,
            min_probability=1e-5,
            ratio_clip=3.0,
            diagnostic_decay=0.8,
        )
    )
    restored = BehaviorModel.from_config(model.to_config())
    assert restored.to_config() == model.to_config()

    state = restored.init(feature_dim=2, key=jax.random.key(4))
    sample = restored.sample_action(state, jnp.ones(2, dtype=jnp.float32))
    chex.assert_shape(sample.probabilities, (3,))
    chex.assert_trees_all_close(jnp.sum(sample.probabilities), 1.0, atol=1e-6)
    assert 0 <= int(sample.action) < 3


def test_importance_ratio_and_epsilon_greedy_helpers() -> None:
    target = jnp.array([[0.8, 0.2], [0.1, 0.9]], dtype=jnp.float32)
    behavior = jnp.array([[0.4, 0.6], [0.5, 0.5]], dtype=jnp.float32)
    actions = jnp.array([0, 1], dtype=jnp.int32)

    ratios = clipped_importance_ratios(
        target,
        behavior,
        actions,
        clip=1.5,
    )
    chex.assert_trees_all_close(ratios, jnp.array([1.5, 1.5], dtype=jnp.float32))

    q_values = jnp.array([1.0, 3.0, 3.0, 0.0], dtype=jnp.float32)
    probs = epsilon_greedy_probabilities(q_values, jnp.array(0.2, dtype=jnp.float32))
    expected = jnp.array([0.05, 0.45, 0.45, 0.05], dtype=jnp.float32)
    chex.assert_trees_all_close(probs, expected, atol=1e-6)

    model = BehaviorModel(BehaviorModelConfig(n_actions=2, ratio_clip=1.25))
    state = model.init(feature_dim=2, key=jax.random.key(5))
    ratio = model.importance_ratio(
        state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.array(1, dtype=jnp.int32),
        jnp.array([0.1, 0.9], dtype=jnp.float32),
    )
    assert float(ratio) == 1.25


def test_behavior_model_config_rejects_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        BehaviorModelConfig(n_actions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_actions"):
        BehaviorModelConfig(n_actions=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="step_size"):
        BehaviorModelConfig(n_actions=2, step_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="diagnostic_decay"):
        BehaviorModelConfig(n_actions=2, diagnostic_decay=1.0 - 1e-10)


def test_behavior_model_config_accepts_and_canonicalizes_numpy_integers() -> None:
    config = BehaviorModelConfig(n_actions=np.int32(3))
    assert type(config.n_actions) is int
    assert config.n_actions == 3

    model = BehaviorModel(config)
    state = model.init(feature_dim=np.int64(4), key=jax.random.key(0))
    assert state.weights.shape == (3, 4)

    budget = model.resource_budget(np.uint16(4))
    assert budget.feature_dim == 4
    assert type(budget.feature_dim) is int


@pytest.mark.parametrize(
    "scalar_type",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    ],
)
def test_behavior_model_accepts_full_numpy_integer_family(scalar_type: type) -> None:
    config = BehaviorModelConfig(n_actions=scalar_type(2))  # type: ignore[call-arg,arg-type]
    assert type(config.n_actions) is int
    assert config.n_actions == 2


def test_behavior_model_rejects_hostile_scalar_without_running_hooks() -> None:
    with pytest.raises(ValueError, match="step_size"):
        BehaviorModelConfig(n_actions=2, step_size=_HostileFloat(0.1))


def test_behavior_model_deserialization_requires_exact_complete_dicts() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=2))
    construction = model.to_config()
    with pytest.raises(ValueError, match="actual dict"):
        BehaviorModel.from_config(_HostileDict(construction))

    wrong_type = model.to_config()
    wrong_type["type"] = _HostileFloat(0.0)
    with pytest.raises(ValueError, match="type"):
        BehaviorModel.from_config(wrong_type)
    missing = model.to_config()
    missing.pop("type")
    with pytest.raises(ValueError, match="fields"):
        BehaviorModel.from_config(missing)
    nested_subclass = model.to_config()
    nested = nested_subclass["config"]
    assert isinstance(nested, dict)
    nested_subclass["config"] = _HostileDict(nested)
    with pytest.raises(ValueError, match="nested config"):
        BehaviorModel.from_config(nested_subclass)


def test_behavior_model_preflights_resource_bytes_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=2))
    persist_last_legal = ((2**31 - 1) - 32) // 8
    update_last_legal = 48_806_441
    with pytest.raises(ValueError, match="update working set byte count"):
        model.resource_budget(persist_last_legal)
    with pytest.raises(ValueError, match="state_nbytes"):
        model.resource_budget(persist_last_legal + 1)
    budget = model.resource_budget(update_last_legal)
    assert budget.state_nbytes <= 2**31 - 1

    calls = 0

    def forbidden_zeros(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("allocator reached")

    monkeypatch.setattr("alberta_framework.core.behavior_model.jnp.zeros", forbidden_zeros)
    with pytest.raises(ValueError, match="state_nbytes"):
        model.init(persist_last_legal + 1, jax.random.key(0))
    assert calls == 0
    with pytest.raises(ValueError, match="update working set byte count"):
        model.init(persist_last_legal, jax.random.key(0))
    assert calls == 0
    with pytest.raises(AssertionError, match="allocator reached"):
        model.init(update_last_legal, jax.random.key(0))
    assert calls == 1


def test_action_ids_reject_untrusted_arrays_without_running_conversion_hooks() -> None:
    """The host path must gate on type before any conversion hook can run.

    ``_integer_action_ids`` reaches ``np.asarray(actions)`` before validating
    anything on the non-tracer path, so an untrusted ``__array__`` hook
    executes ahead of the dtype and range checks and can decide the value that
    is subsequently validated.
    """

    class _HostileArray:
        def __array__(self, dtype: object = None, copy: object = None) -> np.ndarray:
            raise AssertionError("array hook must not run")

    probabilities = jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    with pytest.raises(TypeError, match="trusted array"):
        selected_action_probabilities(probabilities, _HostileArray())

    model = BehaviorModel(BehaviorModelConfig(n_actions=4, step_size=0.1))
    state = model.init(feature_dim=2, key=jax.random.key(0))
    observation = jnp.array([1.0, -0.5], dtype=jnp.float32)
    with pytest.raises(TypeError, match="trusted array"):
        model.update(state, observation, _HostileArray())


def test_action_ids_still_accept_trusted_hosts_and_tracers() -> None:
    """The gate must not reject the array types the module already supports."""
    probabilities = jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    for action in (
        jnp.asarray(2, dtype=jnp.int32),
        np.array(2, dtype=np.int16),
        np.int32(1),
        2,
    ):
        chex.assert_tree_all_finite(selected_action_probabilities(probabilities, action))

    traced = jax.jit(lambda a: selected_action_probabilities(probabilities, a))(
        jnp.asarray(3, dtype=jnp.int32)
    )
    chex.assert_tree_all_finite(traced)


_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_behavior_model_persist_matches_jit_materialized_leaves() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=2))
    state = model.init(4, jax.random.key(0))
    actual = sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in tree_util.tree_leaves(state)
    )
    assert actual == model.resource_budget(4).state_nbytes
    _, persist_bytes = _resource_counts(2, 4)
    assert actual == persist_bytes


def test_behavior_model_persist_fits_while_update_working_set_does_not() -> None:
    feature_dim = 180_000_000
    _, persist_bytes = _resource_counts(1, feature_dim)
    working_set_bytes = _behavior_model_update_working_set_bytes(1, feature_dim)
    assert persist_bytes <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    model = BehaviorModel(BehaviorModelConfig(n_actions=1))
    with pytest.raises(ValueError, match="update working set byte count"):
        model.resource_budget(feature_dim)
    with pytest.raises(ValueError, match="update working set byte count"):
        model.init(feature_dim, jax.random.key(0))


def test_behavior_model_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for feature_dim in range(89_478_476, 89_478_480):
        working_set_bytes = _behavior_model_update_working_set_bytes(1, feature_dim)
        _, persist_bytes = _resource_counts(1, feature_dim)
        assert persist_bytes <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = feature_dim
        elif first_overflow is None:
            first_overflow = feature_dim
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    model = BehaviorModel(BehaviorModelConfig(n_actions=1))
    model.resource_budget(last_fit)
    with pytest.raises(ValueError, match="update working set byte count"):
        model.resource_budget(first_overflow)


def test_behavior_model_persistent_byte_bound_still_fires_first() -> None:
    feature_dim = 536_870_905
    with pytest.raises(ValueError, match="state_nbytes"):
        _resource_counts(1, feature_dim)
    model = BehaviorModel(BehaviorModelConfig(n_actions=1))
    with pytest.raises(ValueError, match="state_nbytes"):
        model.init(feature_dim, jax.random.key(0))


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_behavior_model_update_working_set(1, 180_000_000)


# =============================================================================
# Scan sequence-length ceiling (hang guard)
# =============================================================================
#
# ``run_behavior_model_from_arrays`` hands ``observations``/``actions``
# straight to ``jax.lax.scan`` with no bound on the leading (step) axis. A
# hostile or mistaken caller supplying a huge array forces JAX to materialize
# per-step outputs at that length, hanging the process well before any step
# executes -- the same hang class already fixed for other scan-driven array
# loops in ``core`` and ``utils`` (e.g. ``core/sarsa.py``,
# ``core/average_reward.py``, ``core/learners.py``, ``utils/nexting.py``).


def _spy_scan(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []

    def spy(fn, init, xs, **kwargs):  # type: ignore[no-untyped-def]
        first = xs[0] if isinstance(xs, tuple) else xs
        seen.append(int(first.shape[0]))
        raise AssertionError(f"jax.lax.scan must not run: T={first.shape[0]}")

    monkeypatch.setattr("alberta_framework.core.behavior_model.jax.lax.scan", spy)
    return seen


class TestBehaviorModelSequenceCeiling:
    def test_documented_protocol_ceiling(self) -> None:
        assert _BEHAVIOR_MODEL_SEQUENCE_MAX_STEPS == 50_000

    def test_last_fit_length_is_accepted(self) -> None:
        matrix = jnp.zeros((_BEHAVIOR_MODEL_SEQUENCE_MAX_STEPS, 2))
        assert (
            _require_behavior_model_sequence_length("observations", matrix)
            == _BEHAVIOR_MODEL_SEQUENCE_MAX_STEPS
        )

    def test_first_overflow_length_is_rejected(self) -> None:
        matrix = jnp.zeros((_BEHAVIOR_MODEL_SEQUENCE_MAX_STEPS + 1, 2))
        with pytest.raises(
            ValueError, match=r"observations length must be an integer in \[1, 50000\]"
        ):
            _require_behavior_model_sequence_length("observations", matrix)

    def test_empty_length_is_rejected(self) -> None:
        matrix = jnp.zeros((0, 2))
        with pytest.raises(ValueError, match=r"observations length must be an integer in"):
            _require_behavior_model_sequence_length("observations", matrix)

    def test_scalar_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="leading step axis"):
            _require_behavior_model_sequence_length("observations", jnp.array(1.0))

    def test_non_array_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a trusted array"):
            _require_behavior_model_sequence_length("observations", [[1.0, 2.0]])

        class _HostileArrayLike:
            shape = (3, 2)
            ndim = 2

        with pytest.raises(TypeError, match="must be a trusted array"):
            _require_behavior_model_sequence_length("observations", _HostileArrayLike())

    def test_run_behavior_model_from_arrays_rejects_overflow_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        model = BehaviorModel(BehaviorModelConfig(n_actions=3, step_size=0.05))
        state = model.init(feature_dim=3, key=jax.random.key(5))
        n = _BEHAVIOR_MODEL_SEQUENCE_MAX_STEPS + 1
        observations = jnp.zeros((n, 3), dtype=jnp.float32)
        actions = jnp.zeros((n,), dtype=jnp.int32)
        with pytest.raises(ValueError, match="observations length must be an integer in"):
            run_behavior_model_from_arrays(model, state, observations, actions)
        assert seen == []

    def test_run_behavior_model_from_arrays_rejects_mismatched_actions_before_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _spy_scan(monkeypatch)
        model = BehaviorModel(BehaviorModelConfig(n_actions=3, step_size=0.05))
        state = model.init(feature_dim=3, key=jax.random.key(6))
        observations = jnp.zeros((5, 3), dtype=jnp.float32)
        actions = jnp.zeros((4,), dtype=jnp.int32)
        with pytest.raises(ValueError, match="actions must broadcast"):
            run_behavior_model_from_arrays(model, state, observations, actions)
        assert seen == []

    def test_run_behavior_model_from_arrays_still_runs_inside_the_ceiling(self) -> None:
        model = BehaviorModel(BehaviorModelConfig(n_actions=3, step_size=0.05))
        state = model.init(feature_dim=3, key=jax.random.key(7))
        observations = jnp.eye(3, dtype=jnp.float32).repeat(4, axis=0)
        actions = jnp.array([0, 1, 2] * 4, dtype=jnp.int32)
        result = run_behavior_model_from_arrays(model, state, observations, actions)
        chex.assert_shape(result.probabilities, (12, 3))
        assert int(result.state.step_count) == 12
