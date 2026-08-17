"""Focused contracts for the bounded predict-before-update world-model ensemble."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.learning_signals import (
    LearningSignalEstimatorConfig,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    WorldModelEnsembleResourceBudget,
    save_world_model_ensemble_checkpoint,
)


@pytest.fixture(autouse=True)
def _bound_compilation_cache(request: pytest.FixtureRequest):
    """Compile the dedicated JIT parity case; run contract cases eagerly."""
    try:
        if request.node.name in {
            "test_action_dtype_contract_matches_eager_and_jit",
            "test_eager_and_explicit_jit_match",
            "test_replay_update_eager_and_explicit_jit_match",
        }:
            yield
        else:
            with jax.disable_jit():
                yield
    finally:
        jax.clear_caches()


def _config(
    *,
    ensemble_size: int = 2,
    bootstrap_probability: float = 0.5,
    max_input_magnitude: float = 100.0,
    max_predicted_variance: float = 10_000.0,
    max_observed_loss: float = 10_000.0,
    residual_variance_warmup_steps: int = 1,
) -> WorldModelEnsembleConfig:
    model = ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        gamma=0.95,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        use_layer_norm=False,
        error_decay=0.8,
    )
    signals = LearningSignalEstimatorConfig(
        ensemble_size=ensemble_size,
        target_dim=4,
        progress_warmup_steps=2,
        change_calibration_steps=2,
        fast_loss_decay=0.5,
        slow_loss_decay=0.9,
        max_input_magnitude=max_input_magnitude,
        max_predicted_variance=max_predicted_variance,
        max_observed_loss=max_observed_loss,
    )
    return WorldModelEnsembleConfig(
        model=model,
        signal_estimator=signals,
        ensemble_size=ensemble_size,
        bootstrap_probability=bootstrap_probability,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=residual_variance_warmup_steps,
        residual_variance_floor=1.0e-6,
    )


def _event(index: int = 0) -> tuple[jax.Array, ...]:
    observation = jnp.asarray(
        [0.1 + 0.02 * index, -0.2 + 0.01 * index], dtype=jnp.float32
    )
    action = jnp.asarray(index % 2, dtype=jnp.int32)
    reward = jnp.asarray(0.3 - 0.01 * index, dtype=jnp.float32)
    discount = jnp.asarray(0.9, dtype=jnp.float32)
    next_observation = observation + jnp.asarray([0.05, -0.03], dtype=jnp.float32)
    return observation, action, reward, discount, next_observation


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_structure = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_structure = jax.tree.flatten(_materialize_keys(right))
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_public_exports_and_schema() -> None:
    assert alberta.WorldModelEnsemble is core.WorldModelEnsemble
    assert alberta.WorldModelEnsembleConfig is core.WorldModelEnsembleConfig
    assert (
        alberta.WorldModelEnsembleResourceBudget
        is core.WorldModelEnsembleResourceBudget
        is WorldModelEnsembleResourceBudget
    )
    assert alberta.WorldModelEnsembleState is core.WorldModelEnsembleState
    assert alberta.WorldModelEnsembleUpdateResult is core.WorldModelEnsembleUpdateResult


def test_config_and_ensemble_round_trip_are_strict() -> None:
    config = _config(ensemble_size=3, residual_variance_warmup_steps=3)
    restored = WorldModelEnsembleConfig.from_config(config.to_config())
    assert restored == config
    assert config.to_config()["residual_variance_warmup_steps"] == 3

    ensemble = WorldModelEnsemble(config)
    reconstructed = WorldModelEnsemble.from_config(ensemble.to_config())
    assert reconstructed.config == config

    bad = config.to_config()
    bad["accepted_scientific_evidence"] = True
    with pytest.raises(ValueError, match="not accepted scientific evidence"):
        WorldModelEnsembleConfig.from_config(bad)

    with pytest.raises(ValueError, match="keys"):
        WorldModelEnsemble.from_config(
            {"type": "WorldModelEnsemble", "config": config.to_config(), "extra": 1}
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("ensemble_size", 1, "ensemble_size"),
        ("bootstrap_probability", 0.0, "bootstrap_probability"),
        ("bootstrap_probability", 1.0, "bootstrap_probability"),
        ("residual_variance_decay", 1.0, "residual_variance_decay"),
        ("residual_variance_warmup_steps", 0, "residual_variance_warmup_steps"),
        ("residual_variance_floor", 0.0, "residual_variance_floor"),
    ],
)
def test_config_rejects_invalid_static_contract(
    field: str,
    value: Any,
    match: str,
) -> None:
    base = _config()
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(base, **{field: value})


def test_config_rejects_signal_shape_mismatches() -> None:
    base = _config()
    wrong_ensemble = dataclasses.replace(
        base.signal_estimator,
        ensemble_size=3,
    )
    with pytest.raises(ValueError, match="ensemble_size"):
        dataclasses.replace(base, signal_estimator=wrong_ensemble)

    wrong_target = dataclasses.replace(base.signal_estimator, target_dim=3)
    with pytest.raises(ValueError, match="target_dim"):
        dataclasses.replace(base, signal_estimator=wrong_target)


def test_init_uses_distinct_member_keys_and_isolated_real_replay_mask_keys() -> None:
    ensemble = WorldModelEnsemble(_config(ensemble_size=3))
    state = ensemble.init(jr.key(7))
    assert bool(ensemble.state_valid(state))
    assert len(state.member_states) == 3

    member_weights = [
        state.member_states[index].learner_state.head_params.weights[0]
        for index in range(3)
    ]
    assert not bool(jnp.array_equal(member_weights[0], member_weights[1]))
    assert not bool(jnp.array_equal(member_weights[1], member_weights[2]))

    split_keys = jr.split(jr.key(7), 4)
    chex.assert_trees_all_equal(jr.key_data(state.bootstrap_key), jr.key_data(split_keys[-1]))
    chex.assert_trees_all_equal(
        jr.key_data(state.replay_bootstrap_key),
        jr.key_data(jr.fold_in(jr.key(7), 0x5245504C)),
    )
    assert not bool(jnp.array_equal(state.bootstrap_key, state.replay_bootstrap_key))
    chex.assert_trees_all_equal(state.last_bootstrap_mask, jnp.zeros(3, dtype=jnp.bool_))
    chex.assert_trees_all_equal(
        state.last_replay_bootstrap_mask, jnp.zeros(3, dtype=jnp.bool_)
    )


def test_resource_budget_counts_member_lifetime_words_and_matches_state() -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(0))
    budget = ensemble.resource_budget(state)
    leaves = jax.tree.leaves(_materialize_keys(state))
    expected_scalars = sum(int(jnp.asarray(leaf).size) for leaf in leaves)
    expected_bytes = sum(int(jnp.asarray(leaf).nbytes) for leaf in leaves)
    assert budget.persistent_state_scalars == expected_scalars
    assert budget.persistent_state_bytes == expected_bytes
    assert budget.persistent_uint32_scalars == 2 * budget.ensemble_size + 4


def test_predict_is_read_only_and_fail_closed() -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(0))
    state_before = _materialize_keys(state)
    observation, action, *_ = _event()
    prediction = ensemble.predict(state, observation, action)
    assert bool(prediction.valid)
    assert prediction.member_raw_predictions.shape == (2, 4)
    assert prediction.member_next_observations.shape == (2, 2)
    assert not bool(prediction.residual_proxy_ready)

    invalid = ensemble.predict(
        state,
        jnp.asarray([jnp.nan, 0.0], dtype=jnp.float32),
        action,
    )
    assert not bool(invalid.valid)
    chex.assert_trees_all_equal(invalid.member_raw_predictions, jnp.zeros((2, 4)))
    _assert_tree_equal(state, state_before)


def test_update_reports_exact_preupdate_prediction_loss_and_targets() -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(3))
    event = _event()
    prediction = ensemble.predict(state, event[0], event[1])
    targets = ensemble.member_model.targets(event[0], event[2], event[3], event[4])
    expected_member_losses = jnp.mean(
        jnp.square(prediction.member_raw_predictions - targets[None, :]), axis=1
    )

    result = ensemble.update(state, *event)
    assert bool(result.diagnostics.applied)
    chex.assert_trees_all_close(
        result.prediction.member_raw_predictions,
        prediction.member_raw_predictions,
    )
    chex.assert_trees_all_close(result.targets, targets)
    chex.assert_trees_all_close(
        result.member_prediction_losses,
        expected_member_losses,
    )
    chex.assert_trees_all_close(result.observed_loss, jnp.mean(expected_member_losses))
    chex.assert_trees_all_close(
        result.representation_objective,
        0.5 * jnp.mean(expected_member_losses),
    )
    assert bool(result.representation_gradient_valid)
    assert bool(result.diagnostics.representation_gradient_valid)
    chex.assert_trees_all_equal(
        result.state.member_update_counts,
        result.bootstrap_mask.astype(jnp.int32),
    )
    assert int(result.state.event_count) == 1
    assert int(result.state.signal_state.step_count) == 1
    chex.assert_trees_all_equal(result.state.last_bootstrap_mask, result.bootstrap_mask)

    # The first event used only the configured prior floor. Its signal event is
    # recorded, but uncertainty channels are conservatively unavailable.
    assert bool(result.signals.availability.input_valid)
    assert not bool(result.signals.availability.epistemic)
    assert not bool(result.signals.availability.aleatoric)
    assert not bool(result.signals.availability.normalized_residual)


def test_replay_update_changes_only_model_and_replay_accounting() -> None:
    ensemble = WorldModelEnsemble(
        _config(bootstrap_probability=0.999)
    )
    real = ensemble.update(ensemble.init(jr.key(71)), *_event(0))
    assert bool(real.diagnostics.applied)
    state = real.state
    prediction = ensemble.predict(state, _event(1)[0], _event(1)[1])

    replay = ensemble.replay_update(
        state,
        *_event(1),
        jnp.asarray(True),
    )

    assert bool(replay.diagnostics.applied)
    assert bool(replay.diagnostics.calibration_unchanged)
    chex.assert_trees_all_close(
        replay.prediction.member_raw_predictions,
        prediction.member_raw_predictions,
    )
    _assert_tree_equal(replay.state.signal_state, state.signal_state)
    _assert_tree_equal(replay.state.residual_variances, state.residual_variances)
    _assert_tree_equal(replay.state.bootstrap_key, state.bootstrap_key)
    _assert_tree_equal(replay.state.last_bootstrap_mask, state.last_bootstrap_mask)
    _assert_tree_equal(replay.state.member_update_counts, state.member_update_counts)
    assert int(replay.state.event_count) == int(state.event_count)
    assert int(replay.state.replay_event_count) == 1
    np.testing.assert_array_equal(
        np.asarray(replay.state.replay_member_update_counts),
        np.asarray(replay.member_updates_applied, dtype=np.int32),
    )
    assert bool(jnp.any(replay.member_updates_applied))
    assert not np.array_equal(
        np.asarray(jr.key_data(replay.state.replay_bootstrap_key)),
        np.asarray(jr.key_data(state.replay_bootstrap_key)),
    )


def test_replay_padding_and_invalid_available_sample_are_atomic() -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(73))
    event = _event(0)
    padding = ensemble.replay_update(
        state,
        *event,
        jnp.asarray(False),
    )
    assert not bool(padding.diagnostics.sample_available)
    assert not bool(padding.diagnostics.applied)
    assert not bool(padding.diagnostics.rejected)
    _assert_tree_equal(padding.state, state)

    invalid_event = (
        event[0],
        jnp.asarray(2, dtype=jnp.int32),
        *event[2:],
    )
    invalid = ensemble.replay_update(
        state,
        *invalid_event,
        jnp.asarray(True),
    )
    assert bool(invalid.diagnostics.sample_available)
    assert not bool(invalid.diagnostics.input_valid)
    assert not bool(invalid.diagnostics.applied)
    assert bool(invalid.diagnostics.rejected)
    _assert_tree_equal(invalid.state, state)


def test_replay_does_not_advance_real_bootstrap_stream() -> None:
    ensemble = WorldModelEnsemble(_config())
    base = ensemble.init(jr.key(79))
    replay_branch = base
    for index in range(3):
        replay_result = ensemble.replay_update(
            replay_branch,
            *_event(index),
            jnp.asarray(True),
        )
        assert bool(replay_result.diagnostics.applied)
        replay_branch = replay_result.state

    direct_real = ensemble.update(base, *_event(4))
    replay_then_real = ensemble.update(replay_branch, *_event(4))
    assert bool(direct_real.diagnostics.applied)
    assert bool(replay_then_real.diagnostics.applied)
    np.testing.assert_array_equal(
        np.asarray(direct_real.bootstrap_mask),
        np.asarray(replay_then_real.bootstrap_mask),
    )
    np.testing.assert_array_equal(
        np.asarray(jr.key_data(direct_real.state.bootstrap_key)),
        np.asarray(jr.key_data(replay_then_real.state.bootstrap_key)),
    )


def test_replay_counter_exhaustion_fails_closed() -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(83)).replace(
        replay_event_count=jnp.asarray(2**31 - 1, dtype=jnp.int32)
    )
    assert bool(ensemble.state_valid(state))
    result = ensemble.replay_update(
        state,
        *_event(0),
        jnp.asarray(True),
    )
    assert not bool(result.diagnostics.capacity_available)
    assert not bool(result.diagnostics.applied)
    assert bool(result.diagnostics.rejected)
    _assert_tree_equal(result.state, state)


def test_representation_gradient_matches_frozen_target_autodiff_and_difference() -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(41))
    observation, action, reward, discount, next_observation = _event(2)
    frozen_target = jax.lax.stop_gradient(
        ensemble.member_model.targets(
            observation,
            reward,
            discount,
            next_observation,
        )
    )

    def objective(representation: jax.Array) -> jax.Array:
        raw_predictions = jnp.stack(
            [
                ensemble.member_model.predict(member, representation, action).raw_predictions
                for member in state.member_states
            ]
        )
        return 0.5 * jnp.mean(
            jnp.square(raw_predictions - frozen_target[None, :])
        )

    expected_objective, expected_gradient = jax.value_and_grad(objective)(observation)
    result = ensemble.update(state, *_event(2))
    assert bool(result.representation_gradient_valid)
    chex.assert_trees_all_close(
        result.representation_objective,
        expected_objective,
    )
    chex.assert_trees_all_close(
        result.representation_gradient,
        expected_gradient,
    )

    epsilon = jnp.asarray(1.0e-3, dtype=jnp.float32)
    finite_difference = []
    for index in range(observation.shape[0]):
        basis = jnp.zeros_like(observation).at[index].set(epsilon)
        finite_difference.append(
            (objective(observation + basis) - objective(observation - basis))
            / (2.0 * epsilon)
        )
    chex.assert_trees_all_close(
        result.representation_gradient,
        jnp.stack(finite_difference),
        rtol=2.0e-3,
        atol=2.0e-4,
    )

    def objective_with_leaking_target(representation: jax.Array) -> jax.Array:
        raw_predictions = jnp.stack(
            [
                ensemble.member_model.predict(member, representation, action).raw_predictions
                for member in state.member_states
            ]
        )
        target = ensemble.member_model.targets(
            representation,
            reward,
            discount,
            next_observation,
        )
        return 0.5 * jnp.mean(jnp.square(raw_predictions - target[None, :]))

    leaking_gradient = jax.grad(objective_with_leaking_target)(observation)
    assert not bool(jnp.allclose(result.representation_gradient, leaking_gradient))


def test_learning_signal_observe_uses_preupdate_proxy_and_predictions() -> None:
    ensemble = WorldModelEnsemble(_config())
    first = ensemble.update(ensemble.init(jr.key(11)), *_event(0))
    state = first.state
    event = _event(1)
    prediction = ensemble.predict(state, event[0], event[1])
    target = ensemble.member_model.targets(event[0], event[2], event[3], event[4])
    member_losses = jnp.mean(
        jnp.square(prediction.member_raw_predictions - target[None, :]), axis=1
    )
    expected_signal_state, expected_signals = ensemble.signal_estimator.observe(
        state.signal_state,
        prediction.member_raw_predictions,
        state.residual_variances,
        target,
        jnp.mean(member_losses),
    )

    result = ensemble.update(state, *event)
    assert bool(result.diagnostics.applied)
    assert bool(result.prediction.residual_proxy_ready)
    _assert_tree_equal(result.state.signal_state, expected_signal_state)
    chex.assert_trees_all_close(
        result.signals.epistemic_disagreement,
        expected_signals.epistemic_disagreement,
    )
    chex.assert_trees_all_close(
        result.signals.aleatoric_uncertainty,
        expected_signals.aleatoric_uncertainty,
    )
    chex.assert_trees_all_close(
        result.prediction.residual_variances,
        state.residual_variances,
    )
    assert bool(result.signals.availability.epistemic)
    assert bool(result.signals.availability.aleatoric)
    assert bool(result.signals.availability.learning_progress)


def test_residual_proxy_updates_only_after_signal_observation() -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(5))
    event = _event()
    result = ensemble.update(state, *event)
    residuals = result.prediction.member_raw_predictions - result.targets[None, :]
    expected = jnp.maximum(jnp.square(residuals), ensemble.config.residual_variance_floor)
    chex.assert_trees_all_close(result.state.residual_variances, expected)
    chex.assert_trees_all_close(
        result.prediction.residual_variances,
        state.residual_variances,
    )


def test_residual_proxy_readiness_obeys_configured_warmup() -> None:
    ensemble = WorldModelEnsemble(
        _config(residual_variance_warmup_steps=2)
    )
    state = ensemble.init(jr.key(43))

    first = ensemble.update(state, *_event(0))
    assert bool(first.diagnostics.applied)
    assert not bool(first.prediction.residual_proxy_ready)
    assert not bool(first.signals.availability.aleatoric)

    second = ensemble.update(first.state, *_event(1))
    assert bool(second.diagnostics.applied)
    assert not bool(second.prediction.residual_proxy_ready)
    assert not bool(second.signals.availability.aleatoric)

    third = ensemble.update(second.state, *_event(2))
    assert bool(third.diagnostics.applied)
    assert bool(third.prediction.residual_proxy_ready)
    assert bool(third.signals.availability.aleatoric)


@pytest.mark.parametrize(
    "event",
    [
        (
            jnp.asarray([jnp.nan, 0.0], dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.9, dtype=jnp.float32),
            jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        ),
        (
            jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.9, dtype=jnp.float32),
            jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        ),
        (
            jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(1.1, dtype=jnp.float32),
            jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        ),
    ],
)
def test_invalid_dynamic_input_is_an_exact_atomic_noop(
    event: tuple[jax.Array, ...],
) -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(13))
    result = ensemble.update(state, *event)
    assert not bool(result.diagnostics.applied)
    assert bool(result.diagnostics.rejected)
    assert not bool(result.diagnostics.input_valid)
    _assert_tree_equal(result.state, state)


def test_corrupt_dynamic_state_is_an_exact_atomic_noop() -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(2))
    corrupt = state.replace(
        residual_variances=state.residual_variances.at[0, 0].set(jnp.nan)
    )
    result = ensemble.update(corrupt, *_event())
    assert not bool(result.diagnostics.state_valid)
    assert not bool(result.diagnostics.applied)
    _assert_tree_equal(result.state, corrupt)


def test_corrupt_dynamic_member_state_is_an_exact_atomic_noop() -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(47))
    member = state.member_states[0]
    head_params = member.learner_state.head_params
    corrupt_head_params = head_params.replace(
        weights=(
            head_params.weights[0].at[0, 0].set(jnp.nan),
            *head_params.weights[1:],
        )
    )
    corrupt_learner = member.learner_state.replace(head_params=corrupt_head_params)
    corrupt_member = member.replace(learner_state=corrupt_learner)
    corrupt = state.replace(
        member_states=(corrupt_member, *state.member_states[1:])
    )

    result = ensemble.update(corrupt, *_event())
    assert not bool(result.diagnostics.state_valid)
    assert not bool(result.diagnostics.applied)
    _assert_tree_equal(result.state, corrupt)
    _assert_tree_equal(result.state.signal_state, corrupt.signal_state)
    np.testing.assert_array_equal(
        np.asarray(jr.key_data(result.state.bootstrap_key)),
        np.asarray(jr.key_data(corrupt.bootstrap_key)),
    )


def test_candidate_numeric_failure_rolls_back_rng_models_signals_and_proxy() -> None:
    config = _config(
        max_input_magnitude=100.0,
        max_predicted_variance=10.0,
        max_observed_loss=10.0,
    )
    ensemble = WorldModelEnsemble(config)
    state = ensemble.init(jr.key(17))
    observation, action, _, discount, next_observation = _event()
    result = ensemble.update(
        state,
        observation,
        action,
        jnp.asarray(50.0, dtype=jnp.float32),
        discount,
        next_observation,
    )
    assert bool(result.diagnostics.input_valid)
    assert not bool(result.diagnostics.predictions_valid)
    assert not bool(result.diagnostics.applied)
    assert not bool(result.representation_gradient_valid)
    _assert_tree_equal(result.state, state)
    _assert_tree_equal(result.state.signal_state, state.signal_state)
    np.testing.assert_array_equal(
        np.asarray(jr.key_data(result.state.bootstrap_key)),
        np.asarray(jr.key_data(state.bootstrap_key)),
    )


def test_static_shape_and_dtype_drift_raise_before_execution() -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(0))
    with pytest.raises(ValueError, match="observation must have shape"):
        ensemble.update(
            state,
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.9, dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="action must be a scalar with dtype int32"):
        ensemble.update(
            state,
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.9, dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.float32),
        )


def test_member_static_shape_corruption_raises_before_execution() -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(53))
    member = state.member_states[0].replace(
        observation_min=jnp.zeros((1, 2), dtype=jnp.float32)
    )
    corrupt = state.replace(member_states=(member, *state.member_states[1:]))
    with pytest.raises(ValueError, match=r"member_states\[0\].*shape"):
        ensemble.update(corrupt, *_event())


def test_action_dtype_contract_matches_eager_and_jit() -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(59))
    observation, _, reward, discount, next_observation = _event()
    int16_action = jnp.asarray(0, dtype=jnp.int16)
    message = "action must be a scalar with dtype int32"

    with jax.disable_jit():
        with pytest.raises(ValueError, match=message):
            ensemble.update(
                state,
                observation,
                int16_action,
                reward,
                discount,
                next_observation,
            )

    with pytest.raises(ValueError, match=message):
        jax.jit(ensemble.update)(
            state,
            observation,
            int16_action,
            reward,
            discount,
            next_observation,
        )


def test_fixed_update_budget_and_persisted_masks() -> None:
    ensemble = WorldModelEnsemble(_config(ensemble_size=3, bootstrap_probability=0.5))
    state = ensemble.init(jr.key(23))
    masks = []
    for index in range(16):
        result = ensemble.update(state, *_event(index))
        assert bool(result.diagnostics.applied)
        assert int(jnp.sum(result.member_updates_applied)) <= 3
        masks.append(np.asarray(result.bootstrap_mask))
        state = result.state

    mask_array = np.stack(masks)
    assert np.unique(mask_array, axis=0).shape[0] > 1
    np.testing.assert_array_equal(
        np.asarray(state.member_update_counts),
        np.sum(mask_array, axis=0),
    )
    np.testing.assert_array_equal(np.asarray(state.last_bootstrap_mask), mask_array[-1])
    assert int(state.event_count) == 16


def test_eager_and_explicit_jit_match() -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(29))
    event = _event(2)
    eager = ensemble.update(state, *event)
    compiled = jax.jit(ensemble.update)(state, *event)
    _assert_tree_equal(eager, compiled)


def test_replay_update_eager_and_explicit_jit_match() -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(89))
    event = _event(2)
    eager = ensemble.replay_update(state, *event, jnp.asarray(True))
    compiled = jax.jit(ensemble.replay_update)(state, *event, jnp.asarray(True))
    _assert_tree_equal(eager, compiled)


def test_checkpoint_rejects_invalid_state(tmp_path: Path) -> None:
    ensemble = WorldModelEnsemble(_config())
    state = ensemble.init(jr.key(0))
    corrupt = state.replace(event_count=jnp.asarray(-1, dtype=jnp.int32))
    with pytest.raises(ValueError, match="invalid"):
        save_world_model_ensemble_checkpoint(
            ensemble,
            corrupt,
            tmp_path / "invalid",
        )


def test_config_preflights_complete_update_result_at_exact_boundary() -> None:
    # Linear 2-observation/2-action fixture: 324 * ensemble_size + 196 bytes.
    last_legal = (2**31 - 1 - 196) // 324
    _config(ensemble_size=last_legal)
    with pytest.raises(ValueError, match="update-result bytes"):
        _config(ensemble_size=last_legal + 1)


def test_prediction_reductions_remain_finite_at_full_float32_member_means() -> None:
    ensemble = WorldModelEnsemble(_config(ensemble_size=2))
    state = ensemble.init(jr.key(0))
    maximum = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
    members = tuple(
        dataclasses.replace(
            member,
            observation_min=jnp.full((2,), maximum, dtype=jnp.float32),
            observation_max=jnp.full((2,), maximum, dtype=jnp.float32),
            reward_min=maximum,
            reward_max=maximum,
            step_count=jnp.asarray(1, dtype=jnp.int32),
        )
        for member in state.member_states
    )
    prediction = ensemble._predict_unchecked(  # noqa: SLF001
        dataclasses.replace(state, member_states=members),
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert bool(prediction.valid)
    assert bool(jnp.all(jnp.isfinite(prediction.mean_next_observation)))
    assert bool(jnp.isfinite(prediction.mean_reward))
    np.testing.assert_array_equal(
        np.asarray(prediction.mean_next_observation),
        np.full((2,), np.finfo(np.float32).max, dtype=np.float32),
    )


def test_prediction_variance_reduction_scales_before_sum() -> None:
    magnitude = 8.0e18
    ensemble_size = 8
    model = ActionConditionedWorldModelConfig(
        observation_dim=2, n_actions=2, hidden_sizes=(), sparsity=0.0
    )
    signals = LearningSignalEstimatorConfig(
        ensemble_size=ensemble_size,
        target_dim=4,
        max_input_magnitude=magnitude,
    )
    ensemble = WorldModelEnsemble(
        WorldModelEnsembleConfig(
            model=model,
            signal_estimator=signals,
            ensemble_size=ensemble_size,
        )
    )
    state = ensemble.init(jr.key(0))
    members = []
    for index, member in enumerate(state.member_states):
        heads = member.learner_state.head_params
        signed = magnitude if index % 2 == 0 else -magnitude
        members.append(
            dataclasses.replace(
                member,
                learner_state=dataclasses.replace(
                    member.learner_state,
                    head_params=dataclasses.replace(
                        heads,
                        weights=tuple(jnp.zeros_like(weight) for weight in heads.weights),
                        biases=tuple(jnp.full_like(bias, signed) for bias in heads.biases),
                    ),
                ),
            )
        )
    prediction = ensemble._predict_unchecked(  # noqa: SLF001
        dataclasses.replace(state, member_states=tuple(members)),
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert bool(prediction.valid)
    assert bool(jnp.all(jnp.isfinite(prediction.per_head_epistemic_variance)))
    assert bool(jnp.isfinite(prediction.epistemic_disagreement))


def test_public_prediction_neutralizes_nonfinite_aggregate_variance() -> None:
    ensemble = WorldModelEnsemble(_config(ensemble_size=2))
    state = ensemble.init(jr.key(0))
    maximum = np.finfo(np.float32).max
    members = []
    for index, member in enumerate(state.member_states):
        heads = member.learner_state.head_params
        signed = maximum if index == 0 else -maximum
        members.append(
            dataclasses.replace(
                member,
                learner_state=dataclasses.replace(
                    member.learner_state,
                    head_params=dataclasses.replace(
                        heads,
                        weights=tuple(jnp.zeros_like(weight) for weight in heads.weights),
                        biases=tuple(jnp.full_like(bias, signed) for bias in heads.biases),
                    ),
                ),
            )
        )
    prediction = ensemble.predict(
        dataclasses.replace(state, member_states=tuple(members)),
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert not bool(prediction.valid)
    chex.assert_tree_all_finite(prediction)
    np.testing.assert_array_equal(
        np.asarray(prediction.member_raw_predictions),
        np.zeros((2, 4), dtype=np.float32),
    )
