"""Tests for causal, typed ensemble learning signals."""

from __future__ import annotations

import json
from dataclasses import replace

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.checkpoints import load_checkpoint, save_checkpoint
from alberta_framework.core.learning_signals import (
    LearningSignalEstimator,
    LearningSignalEstimatorConfig,
)


def test_learning_signal_producer_is_publicly_exported() -> None:
    assert alberta.LearningSignalEstimator is core.LearningSignalEstimator
    assert (
        alberta.LearningSignalEstimatorConfig
        is core.LearningSignalEstimatorConfig
    )
    assert alberta.TypedLearningSignals is core.TypedLearningSignals


def _estimator(**overrides: object) -> LearningSignalEstimator:
    values: dict[str, object] = {
        "ensemble_size": 2,
        "target_dim": 1,
        "fast_loss_decay": 0.0,
        "slow_loss_decay": 0.5,
        "progress_warmup_steps": 2,
        "change_calibration_steps": 4,
        "change_z_threshold": 3.0,
        "change_temperature": 0.5,
        "change_decay": 0.8,
        "calibration_scale_floor": 0.25,
    }
    values.update(overrides)
    return LearningSignalEstimator(LearningSignalEstimatorConfig(**values))  # type: ignore[arg-type]


def _observe_scalar(
    estimator: LearningSignalEstimator,
    state,
    *,
    means: tuple[float, float] = (0.0, 0.0),
    variances: tuple[float, float] = (1.0, 1.0),
    target: float = 1.0,
    loss: float = 1.0,
):
    return estimator.observe(
        state,
        jnp.asarray(means, dtype=jnp.float32).reshape(2, 1),
        jnp.asarray(variances, dtype=jnp.float32).reshape(2, 1),
        jnp.asarray([target], dtype=jnp.float32),
        jnp.asarray(loss, dtype=jnp.float32),
    )


def test_config_roundtrip_validation_and_exact_resource_budget() -> None:
    config = LearningSignalEstimatorConfig(
        ensemble_size=3,
        target_dim=2,
        change_calibration_steps=8,
    )
    restored = LearningSignalEstimatorConfig.from_config(config.to_config())
    assert restored == config
    json.dumps(config.to_config())

    budget = LearningSignalEstimator(config).resource_budget()
    assert budget.input_float_scalars_per_step == 2 * 3 * 2 + 2 + 1
    assert budget.persistent_float32_scalars == 5
    assert budget.persistent_int32_scalars == 4
    assert budget.persistent_state_scalars == 9
    assert budget.persistent_state_bytes == 36
    assert budget.output_float32_scalars == 8
    assert budget.output_bool_scalars == 6
    assert budget.output_logical_bytes == 38
    assert budget.trainable_scalars == 0
    json.dumps(budget.to_config())

    with pytest.raises(ValueError, match="ensemble_size"):
        LearningSignalEstimatorConfig(ensemble_size=1, target_dim=1)
    with pytest.raises(ValueError, match="target_dim"):
        LearningSignalEstimatorConfig(ensemble_size=2, target_dim=0)
    with pytest.raises(ValueError, match="smaller"):
        LearningSignalEstimatorConfig(
            ensemble_size=2,
            target_dim=1,
            fast_loss_decay=0.99,
            slow_loss_decay=0.9,
        )
    with pytest.raises(ValueError, match="change_calibration_steps"):
        LearningSignalEstimatorConfig(
            ensemble_size=2,
            target_dim=1,
            change_calibration_steps=1,
        )
    with pytest.raises(ValueError, match="fit in int32"):
        LearningSignalEstimatorConfig(
            ensemble_size=2,
            target_dim=1,
            change_calibration_steps=2_147_483_647,
        )
    with pytest.raises(ValueError, match="type"):
        LearningSignalEstimatorConfig.from_config(
            {
                "type": "OtherConfig",
                "ensemble_size": 2,
                "target_dim": 1,
            }
        )
    with pytest.raises(ValueError, match="accepted scientific evidence"):
        invalid_claim = config.to_config()
        invalid_claim["accepted_scientific_evidence"] = True
        LearningSignalEstimatorConfig.from_config(invalid_claim)


def test_hand_calculation_preserves_units_and_causal_progress() -> None:
    estimator = LearningSignalEstimator(
        LearningSignalEstimatorConfig(
            ensemble_size=2,
            target_dim=2,
            fast_loss_decay=0.0,
            slow_loss_decay=0.5,
            progress_warmup_steps=2,
            change_calibration_steps=4,
        )
    )
    state = estimator.init()
    means = jnp.asarray([[1.0, 3.0], [3.0, 5.0]], dtype=jnp.float32)
    variances = jnp.full((2, 2), 3.0, dtype=jnp.float32)
    target = jnp.asarray([4.0, 2.0], dtype=jnp.float32)

    state, first = estimator.observe(state, means, variances, target, 10.0)
    # Ensemble mean [2, 4], population disagreement [1, 1],
    # total predictive variance [4, 4], and squared residual [4, 4].
    chex.assert_trees_all_close(first.epistemic_disagreement, 1.0)
    chex.assert_trees_all_close(first.epistemic_surprise, 1.0 / 3.0)
    chex.assert_trees_all_close(first.aleatoric_uncertainty, 3.0)
    chex.assert_trees_all_close(first.normalized_residual, 1.0)
    assert bool(first.availability.epistemic)
    assert bool(first.availability.aleatoric)
    assert bool(first.availability.normalized_residual)
    assert not bool(first.availability.learning_progress)
    chex.assert_trees_all_close(first.learning_progress, 0.0)

    state, second = estimator.observe(state, means, variances, target, 0.0)
    # Fast loss is 0, slow loss is 5: progress is +5 observed-loss units.
    chex.assert_trees_all_close(second.learning_progress, 5.0)
    assert bool(second.availability.learning_progress)
    assert int(state.valid_count) == 2


def test_epistemic_and_aleatoric_channels_are_not_conflated() -> None:
    estimator = _estimator()
    state = estimator.init()

    _, epistemic = _observe_scalar(
        estimator,
        state,
        means=(-2.0, 2.0),
        variances=(0.25, 0.25),
        target=0.0,
    )
    _, aleatoric = _observe_scalar(
        estimator,
        state,
        means=(0.0, 0.0),
        variances=(16.0, 16.0),
        target=0.0,
    )

    chex.assert_trees_all_close(epistemic.epistemic_disagreement, 4.0)
    chex.assert_trees_all_close(epistemic.aleatoric_uncertainty, 0.25)
    chex.assert_trees_all_close(epistemic.epistemic_surprise, 16.0)
    chex.assert_trees_all_close(aleatoric.epistemic_disagreement, 0.0)
    chex.assert_trees_all_close(aleatoric.epistemic_surprise, 0.0)
    chex.assert_trees_all_close(aleatoric.aleatoric_uncertainty, 16.0)
    assert not hasattr(epistemic, "total")
    assert not hasattr(epistemic, "score")
    assert not hasattr(epistemic, "reward")


def test_high_aleatoric_noisy_tv_does_not_look_epistemic_or_changed() -> None:
    estimator = _estimator(change_calibration_steps=8, change_decay=0.8)
    state = estimator.init()
    signals = None
    # Squared error and predicted variance are both 100, so the normalized
    # residual remains one even though raw outcomes alternate dramatically.
    for index in range(24):
        state, signals = _observe_scalar(
            estimator,
            state,
            means=(0.0, 0.0),
            variances=(100.0, 100.0),
            target=10.0 if index % 2 == 0 else -10.0,
        )

    assert signals is not None
    chex.assert_trees_all_close(signals.epistemic_disagreement, 0.0)
    chex.assert_trees_all_close(signals.epistemic_surprise, 0.0)
    chex.assert_trees_all_close(signals.aleatoric_uncertainty, 100.0)
    chex.assert_trees_all_close(signals.normalized_residual, 1.0)
    assert bool(signals.availability.change_probability)
    assert float(signals.change_probability) < 0.01


def _calibrated_state(estimator: LearningSignalEstimator):
    state = estimator.init()
    for _ in range(estimator.config.change_calibration_steps):
        state, signal = _observe_scalar(estimator, state, target=1.0)
        assert not bool(signal.availability.change_probability)
    return state


def test_sustained_shift_is_separated_from_an_isolated_outlier() -> None:
    estimator = _estimator(change_calibration_steps=8, change_decay=0.8)
    calibrated = _calibrated_state(estimator)

    _, isolated = _observe_scalar(estimator, calibrated, target=5.0, loss=25.0)
    assert bool(isolated.availability.change_probability)
    assert 0.19 < float(isolated.change_probability) < 0.21

    sustained_state = calibrated
    sustained = None
    for _ in range(10):
        sustained_state, sustained = _observe_scalar(
            estimator,
            sustained_state,
            target=5.0,
            loss=25.0,
        )
    assert sustained is not None
    assert float(sustained.change_probability) > 0.85
    assert float(sustained.change_probability) > 4.0 * float(isolated.change_probability)


def test_warmup_flags_and_invalid_inputs_fail_closed_without_poisoning_state() -> None:
    estimator = _estimator(change_calibration_steps=4)
    state = estimator.init()
    state, valid = _observe_scalar(estimator, state)
    assert bool(valid.availability.input_valid)
    before = state

    state, invalid = estimator.observe(
        state,
        jnp.asarray([[jnp.nan], [0.0]], dtype=jnp.float32),
        jnp.asarray([[1.0], [1.0]], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert not bool(invalid.availability.input_valid)
    assert not any(bool(flag) for flag in jax.tree_util.tree_leaves(invalid.availability))
    invalid_values = (
        invalid.epistemic_disagreement,
        invalid.epistemic_surprise,
        invalid.aleatoric_uncertainty,
        invalid.normalized_residual,
        invalid.learning_progress,
        invalid.calibrated_residual_z,
        invalid.instantaneous_change_probability,
        invalid.change_probability,
    )
    for value in invalid_values:
        assert bool(jnp.all(jnp.isfinite(value)))
        chex.assert_trees_all_close(value, jnp.zeros_like(value))
    assert int(state.step_count) == int(before.step_count) + 1
    assert int(state.invalid_count) == int(before.invalid_count) + 1
    assert int(state.valid_count) == int(before.valid_count)
    chex.assert_trees_all_close(state.calibration_mean, before.calibration_mean)
    chex.assert_trees_all_close(state.fast_loss_ema, before.fast_loss_ema)

    state_after_negative, negative = _observe_scalar(
        estimator,
        state,
        variances=(-1.0, 1.0),
    )
    assert not bool(negative.availability.input_valid)
    assert int(state_after_negative.invalid_count) == int(state.invalid_count) + 1

    corrupt_state = replace(
        state,
        calibration_mean=jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    returned_corrupt_state, corrupt_signal = _observe_scalar(
        estimator,
        corrupt_state,
    )
    assert not any(bool(flag) for flag in jax.tree_util.tree_leaves(corrupt_signal.availability))
    chex.assert_trees_all_equal(returned_corrupt_state, corrupt_state)


def test_lifetime_counters_saturate_without_wraparound() -> None:
    estimator = _estimator(change_calibration_steps=4)
    maximum = 2_147_483_647
    near_limit = replace(
        estimator.init(),
        step_count=jnp.asarray(maximum - 1, dtype=jnp.int32),
        valid_count=jnp.asarray(maximum - 1, dtype=jnp.int32),
        calibration_count=jnp.asarray(4, dtype=jnp.int32),
    )

    saturated, signal = _observe_scalar(estimator, near_limit)
    assert bool(signal.availability.input_valid)
    assert int(saturated.step_count) == maximum
    assert int(saturated.valid_count) == maximum

    still_saturated, _ = estimator.observe(
        saturated,
        jnp.asarray([[jnp.nan], [0.0]], dtype=jnp.float32),
        jnp.ones((2, 1), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert int(still_saturated.step_count) == maximum
    assert int(still_saturated.valid_count) == maximum
    assert int(still_saturated.invalid_count) == 1


def test_shape_and_dtype_validation_is_strict() -> None:
    estimator = _estimator()
    state = estimator.init()
    with pytest.raises(ValueError, match="member_means"):
        estimator.observe(
            state,
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.ones((2, 1), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            0.0,
        )
    with pytest.raises(ValueError, match="floating dtype"):
        estimator.observe(
            state,
            jnp.zeros((2, 1), dtype=jnp.int32),
            jnp.ones((2, 1), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            0.0,
        )
    with pytest.raises(ValueError, match="observed_loss"):
        estimator.observe(
            state,
            jnp.zeros((2, 1), dtype=jnp.float32),
            jnp.ones((2, 1), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
        )


def test_jit_scan_parity_and_fixed_output_shapes() -> None:
    estimator = _estimator(change_calibration_steps=4)
    num_steps = 12
    means = jnp.zeros((num_steps, 2, 1), dtype=jnp.float32)
    variances = jnp.ones_like(means)
    targets = jnp.concatenate(
        [
            jnp.ones((6, 1), dtype=jnp.float32),
            4.0 * jnp.ones((6, 1), dtype=jnp.float32),
        ]
    )
    losses = jnp.square(targets[:, 0])

    eager_state, eager_signals = estimator.scan(
        estimator.init(),
        means,
        variances,
        targets,
        losses,
    )
    compiled_scan = jax.jit(estimator.scan)
    jit_state, jit_signals = compiled_scan(
        estimator.init(),
        means,
        variances,
        targets,
        losses,
    )
    chex.assert_trees_all_close(eager_state, jit_state)
    chex.assert_trees_all_close(eager_signals, jit_signals)

    for value in jax.tree_util.tree_leaves(eager_signals):
        assert value.shape == (num_steps,)
    assert len(jax.tree_util.tree_leaves(eager_signals)) == 14


def test_checkpoint_resume_matches_uninterrupted_scan(tmp_path) -> None:
    estimator = _estimator(change_calibration_steps=4)
    num_steps = 14
    means = jnp.zeros((num_steps, 2, 1), dtype=jnp.float32)
    variances = jnp.ones_like(means)
    targets = jnp.arange(num_steps, dtype=jnp.float32).reshape(num_steps, 1) / 4.0
    losses = jnp.square(targets[:, 0])

    expected_state, _ = estimator.scan(
        estimator.init(),
        means,
        variances,
        targets,
        losses,
    )
    midpoint = 7
    partial_state, _ = estimator.scan(
        estimator.init(),
        means[:midpoint],
        variances[:midpoint],
        targets[:midpoint],
        losses[:midpoint],
    )
    checkpoint_path = tmp_path / "learning-signals"
    save_checkpoint(
        partial_state,
        checkpoint_path,
        metadata={"estimator_config": estimator.to_config()},
    )
    resumed_state, metadata = load_checkpoint(estimator.init(), checkpoint_path)
    assert (
        LearningSignalEstimatorConfig.from_config(metadata["estimator_config"]) == estimator.config
    )
    actual_state, _ = estimator.scan(
        resumed_state,
        means[midpoint:],
        variances[midpoint:],
        targets[midpoint:],
        losses[midpoint:],
    )
    chex.assert_trees_all_close(expected_state, actual_state)
    np.testing.assert_equal(
        jax.tree_util.tree_structure(expected_state),
        jax.tree_util.tree_structure(actual_state),
    )


def test_zero_ema_decay_does_not_multiply_inf_trackers() -> None:
    """A disabled EMA decay times an infinite tracker is NaN."""
    estimator = _estimator(fast_loss_decay=0.0, change_decay=0.0)
    state, _ = _observe_scalar(estimator, estimator.init())
    state = state.replace(fast_loss_ema=jnp.asarray(jnp.inf, dtype=jnp.float32))
    raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
    assert not bool(jnp.isfinite(raw))

    next_state, _ = _observe_scalar(estimator, state, loss=0.25)
    chex.assert_trees_all_close(
        next_state.fast_loss_ema,
        jnp.asarray(0.25, dtype=jnp.float32),
    )

    calibrated = _calibrated_state(estimator).replace(
        sustained_change_probability=jnp.asarray(jnp.inf, dtype=jnp.float32),
    )
    next_calibrated, _ = _observe_scalar(estimator, calibrated)
    assert bool(jnp.isfinite(next_calibrated.sustained_change_probability))


def test_learning_signals_config_rejects_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="ensemble_size"):
        LearningSignalEstimatorConfig(ensemble_size=True, target_dim=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_dim"):
        LearningSignalEstimatorConfig(ensemble_size=2, target_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="progress_warmup_steps"):
        LearningSignalEstimatorConfig(ensemble_size=2, target_dim=1, progress_warmup_steps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="change_calibration_steps"):
        LearningSignalEstimatorConfig(ensemble_size=2, target_dim=1, change_calibration_steps=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="ensemble_size"):
        LearningSignalEstimatorConfig(ensemble_size=2.5, target_dim=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_dim"):
        LearningSignalEstimatorConfig(ensemble_size=2, target_dim=1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="fast_loss_decay"):
        LearningSignalEstimatorConfig(ensemble_size=2, target_dim=1, fast_loss_decay=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fast_loss_decay"):
        LearningSignalEstimatorConfig(ensemble_size=2, target_dim=1, fast_loss_decay=1.0 - 1e-10)
    with pytest.raises(ValueError, match="change_decay"):
        LearningSignalEstimatorConfig(ensemble_size=2, target_dim=1, change_decay=True)  # type: ignore[arg-type]


def test_learning_signals_config_accepts_and_canonicalizes_numpy_integers() -> None:
    config = LearningSignalEstimatorConfig(
        ensemble_size=np.int32(4),
        target_dim=np.int64(2),
        progress_warmup_steps=np.uint16(3),
        change_calibration_steps=np.int8(8),
    )
    assert type(config.ensemble_size) is int
    assert type(config.target_dim) is int
    assert type(config.progress_warmup_steps) is int
    assert type(config.change_calibration_steps) is int
    assert config.ensemble_size == 4
    assert config.target_dim == 2
    assert config.progress_warmup_steps == 3
    assert config.change_calibration_steps == 8
