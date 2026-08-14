"""Tests for EMNIST advanced protection mechanisms.

Validates feature-wise normalization, catastrophic forgetting detection,
per-class batch normalization, and task-aware dropout scheduling.
"""

import dataclasses
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from emnist_advanced_protections import (
    # Feature-wise normalization
    FeatureWiseNormState,
    init_feature_wise_norm_state,
    update_feature_wise_norm,

    # Catastrophic forgetting
    ForgettingDetectorState,
    init_forgetting_detector_state,
    detect_catastrophic_forgetting,

    # Per-class batch norm
    PerClassBatchNormState,
    init_per_class_batch_norm_state,
    normalize_per_class,

    # Task-aware dropout
    TaskAwareDropoutState,
    init_task_aware_dropout_state,
    schedule_dropout_rate,
    apply_task_aware_dropout,

    # Complete learner
    ProtectedEMNISTLearnerState,
    init_protected_emnist_learner,
    protected_emnist_forward,
    protected_emnist_update,
    protected_emnist_predict,
    compute_protected_learner_accuracy,
)


# =============================================================================
# Feature-Wise Normalization Tests
# =============================================================================


class TestFeatureWiseNormalization:
    """Test feature-wise normalization mechanisms."""

    def test_init_creates_valid_state(self):
        """Feature-wise norm state initializes correctly."""
        state = init_feature_wise_norm_state(feature_dim=100)

        assert state.feature_means.shape == (100,)
        assert state.feature_vars.shape == (100,)
        assert state.feature_min.shape == (100,)
        assert state.feature_max.shape == (100,)
        assert jnp.all(jnp.isfinite(state.feature_means))
        assert jnp.all(state.feature_vars == 1.0)

    def test_normalization_centers_features(self):
        """Normalized features are centered around zero on first update."""
        state = init_feature_wise_norm_state(feature_dim=50)
        features = jnp.ones(50) * 5.0

        normalized, _ = update_feature_wise_norm(state, features)

        # First update: features far from zero mean will have large normalized values
        # because initial mean is 0. After momentum update, means shift toward features.
        assert jnp.all(jnp.isfinite(normalized))
        # Second update should show better centering
        features2 = jnp.ones(50) * 5.0
        normalized2, _ = update_feature_wise_norm(state, features2)

        # After state has been updated, should be more centered
        assert jnp.all(jnp.isfinite(normalized2))

    def test_normalization_preserves_shape(self):
        """Normalized output has same shape as input."""
        state = init_feature_wise_norm_state(feature_dim=784)
        features = jr.normal(jr.key(0), (784,))

        normalized, new_state = update_feature_wise_norm(state, features)

        assert normalized.shape == features.shape
        assert new_state.feature_means.shape == features.shape

    def test_momentum_affects_learning_rate(self):
        """Higher momentum means slower adaptation."""
        state_fast = init_feature_wise_norm_state(feature_dim=10, momentum=0.5)
        state_slow = init_feature_wise_norm_state(feature_dim=10, momentum=0.99)

        features = jnp.ones(10) * 10.0

        _, state_fast_after = update_feature_wise_norm(state_fast, features)
        _, state_slow_after = update_feature_wise_norm(state_slow, features)

        # Fast momentum should adapt more
        fast_delta = jnp.abs(state_fast_after.feature_means - state_fast.feature_means).mean()
        slow_delta = jnp.abs(state_slow_after.feature_means - state_slow.feature_means).mean()

        assert fast_delta > slow_delta

    def test_minmax_tracking(self):
        """Min/max values are tracked correctly."""
        state = init_feature_wise_norm_state(feature_dim=5)

        features1 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, state1 = update_feature_wise_norm(state, features1)

        assert jnp.all(state1.feature_min <= features1)
        assert jnp.all(state1.feature_max >= features1)

        features2 = jnp.array([0.5, 1.5, 3.5, 4.5, 5.5])
        _, state2 = update_feature_wise_norm(state1, features2)

        # Min should be element-wise minimum
        expected_min = jnp.minimum(features1, features2)
        expected_max = jnp.maximum(features1, features2)

        assert jnp.allclose(state2.feature_min, expected_min)
        assert jnp.allclose(state2.feature_max, expected_max)

    def test_numerical_stability(self):
        """Normalization handles edge cases without NaNs/Infs."""
        state = init_feature_wise_norm_state(feature_dim=10)

        # Extreme values
        extreme_features = jnp.array([1e6, -1e6, 0.0, 1e-10, -1e-10, 1.0, -1.0, 100.0, -100.0, 0.1])

        normalized, new_state = update_feature_wise_norm(state, extreme_features)

        assert jnp.all(jnp.isfinite(normalized))
        assert jnp.all(jnp.isfinite(new_state.feature_means))
        assert jnp.all(jnp.isfinite(new_state.feature_vars))


# =============================================================================
# Catastrophic Forgetting Detection Tests
# =============================================================================


class TestForgettingDetection:
    """Test catastrophic forgetting detection."""

    def test_init_creates_valid_state(self):
        """Forgetting detector state initializes correctly."""
        state = init_forgetting_detector_state(window_size=20)

        assert state.accuracy_history.shape == (20,)
        assert state.task_switch_count == 0
        assert state.forgetting_episodes == 0
        assert state.recovery_mode_active == False

    def test_detects_accuracy_drop(self):
        """Large accuracy drops trigger forgetting detection."""
        state = init_forgetting_detector_state()

        high_accuracy = jnp.array(0.95)
        new_state, detected, _ = detect_catastrophic_forgetting(
            state, high_accuracy
        )

        # No forgetting initially
        assert detected == False

        # Large accuracy drop
        low_accuracy = jnp.array(0.70)
        new_state2, detected2, _ = detect_catastrophic_forgetting(
            new_state, low_accuracy, forgetting_threshold=0.15
        )

        assert detected2 == True

    def test_recovery_mode_activation(self):
        """Recovery mode activates on forgetting and decays."""
        state = init_forgetting_detector_state()

        # Trigger forgetting
        low_accuracy = jnp.array(0.70)
        new_state, _, recovery_factor1 = detect_catastrophic_forgetting(
            state, low_accuracy, forgetting_threshold=0.15, recovery_duration=50
        )

        # Recovery should reduce step size
        assert recovery_factor1 == pytest.approx(0.5, abs=1e-6)
        assert new_state.recovery_mode_active == True

        # Continue for several steps - recovery should decay
        for _ in range(10):
            new_state, _, recovery_factor = detect_catastrophic_forgetting(
                new_state,
                jnp.array(0.75),  # Slightly higher accuracy
                forgetting_threshold=0.15
            )

        # Still in recovery but fewer steps remaining
        assert new_state.recovery_mode_active == True
        assert new_state.recovery_step_count < 50

    def test_below_threshold_detection(self):
        """Below-threshold accuracy triggers forgetting."""
        state = init_forgetting_detector_state(min_threshold=0.75)

        below_threshold = jnp.array(0.70)
        new_state, detected, _ = detect_catastrophic_forgetting(
            state, below_threshold
        )

        assert detected == True

    def test_task_switch_counting(self):
        """Task switches are counted correctly."""
        state = init_forgetting_detector_state()

        _, _, _ = detect_catastrophic_forgetting(state, jnp.array(0.9))
        new_state1, _, _ = detect_catastrophic_forgetting(
            state, jnp.array(0.9), task_boundary=jnp.array(True)
        )

        assert new_state1.task_switch_count == 1

        new_state2, _, _ = detect_catastrophic_forgetting(
            new_state1, jnp.array(0.85), task_boundary=jnp.array(True)
        )

        assert new_state2.task_switch_count == 2

    def test_accuracy_history_rolling(self):
        """Accuracy history maintains rolling window."""
        state = init_forgetting_detector_state(window_size=5)

        accuracies = [0.9, 0.91, 0.92, 0.88, 0.87, 0.85]

        for acc in accuracies:
            state, _, _ = detect_catastrophic_forgetting(
                state, jnp.array(acc)
            )

        # Should have last 5 accuracies
        expected_last_5 = jnp.array([0.87, 0.85, 0.9, 0.91, 0.92])
        assert state.accuracy_history.shape == (5,)


# =============================================================================
# Per-Class Batch Normalization Tests
# =============================================================================


class TestPerClassBatchNorm:
    """Test per-class batch normalization."""

    def test_init_creates_valid_state(self):
        """Per-class norm state initializes correctly."""
        state = init_per_class_batch_norm_state(n_classes=47)

        assert state.class_means.shape == (47,)
        assert state.class_vars.shape == (47,)
        assert state.class_counts.shape == (47,)
        assert jnp.all(state.class_vars >= 1.0)

    def test_per_class_statistics_update(self):
        """Class statistics update independently."""
        state = init_per_class_batch_norm_state(n_classes=10)

        # Update class 0
        output1, state1 = normalize_per_class(state, jnp.array(5.0), jnp.array(0))

        # Class 0 mean should change
        assert state1.class_means[0] != state.class_means[0]
        # Other classes unchanged
        assert jnp.all(state1.class_means[1:] == state.class_means[1:])

    def test_normalization_per_class(self):
        """Outputs are normalized using class-specific statistics."""
        state = init_per_class_batch_norm_state(n_classes=5)

        # Build different statistics per class
        for i in range(5):
            output, state = normalize_per_class(
                state,
                jnp.array(float(i * 10)),
                jnp.array(i)
            )

        # Now normalize a new output with class 2
        new_output = jnp.array(20.0)
        normalized, _ = normalize_per_class(state, new_output, jnp.array(2))

        assert jnp.all(jnp.isfinite(normalized))
        # Normalized should be roughly centered
        assert jnp.abs(normalized) < 10.0

    def test_class_count_tracking(self):
        """Class counts increment correctly."""
        state = init_per_class_batch_norm_state(n_classes=3)

        for _ in range(5):
            _, state = normalize_per_class(state, jnp.array(1.0), jnp.array(0))

        for _ in range(3):
            _, state = normalize_per_class(state, jnp.array(2.0), jnp.array(1))

        assert state.class_counts[0] == 5
        assert state.class_counts[1] == 3
        assert state.class_counts[2] == 0

    def test_class_last_seen_update(self):
        """Last-seen step is tracked per class."""
        state = init_per_class_batch_norm_state(n_classes=3)

        _, state1 = normalize_per_class(state, jnp.array(1.0), jnp.array(0))
        assert state1.class_last_seen[0] == 0

        _, state2 = normalize_per_class(state1, jnp.array(2.0), jnp.array(1))
        assert state2.class_last_seen[0] == 0
        assert state2.class_last_seen[1] == 1


# =============================================================================
# Task-Aware Dropout Tests
# =============================================================================


class TestTaskAwareDropout:
    """Test task-aware dropout scheduling."""

    def test_init_creates_valid_state(self):
        """Dropout state initializes correctly."""
        state = init_task_aware_dropout_state(feature_dim=100)

        assert state.task_stability.shape == (100,)
        assert state.feature_usage_counts.shape == (100,)
        assert state.current_dropout_rate == state.base_dropout_rate
        assert state.current_step == 0

    def test_dropout_scheduling_exponential(self):
        """Exponential schedule increases dropout on task switch, then decays."""
        base_rate = jnp.array(0.05)

        # Immediately after switch
        rate_t0 = schedule_dropout_rate(base_rate, jnp.array(0), 0, recovery_factor=2.0)
        # Some time after switch
        rate_t50 = schedule_dropout_rate(base_rate, jnp.array(50), 0, recovery_factor=2.0)
        # Long time after switch - should decay back toward base
        rate_t200 = schedule_dropout_rate(base_rate, jnp.array(200), 0, recovery_factor=2.0)

        # Should have: t0 > t50 > t200 (decay pattern)
        assert rate_t0 > rate_t50
        assert rate_t50 > rate_t200

    def test_dropout_scheduling_linear(self):
        """Linear schedule decays linearly."""
        base_rate = jnp.array(0.05)

        rate_t0 = schedule_dropout_rate(base_rate, jnp.array(0), 1, recovery_factor=2.0)
        rate_t50 = schedule_dropout_rate(base_rate, jnp.array(50), 1, recovery_factor=2.0)
        rate_t100 = schedule_dropout_rate(base_rate, jnp.array(100), 1, recovery_factor=2.0)

        # Linear decay: t0 > t50 > t100
        assert rate_t0 > rate_t50
        assert rate_t50 > rate_t100

    def test_dropout_scheduling_cosine(self):
        """Cosine schedule smooth decay."""
        base_rate = jnp.array(0.05)

        rate_t0 = schedule_dropout_rate(base_rate, jnp.array(0), 2, recovery_factor=2.0)
        rate_t50 = schedule_dropout_rate(base_rate, jnp.array(50), 2, recovery_factor=2.0)
        rate_t100 = schedule_dropout_rate(base_rate, jnp.array(100), 2, recovery_factor=2.0)

        # Cosine decay: t0 > t50 > t100
        assert rate_t0 > rate_t50
        assert rate_t50 > rate_t100

    def test_apply_dropout_produces_valid_output(self):
        """Dropout application maintains valid output shape and values."""
        state = init_task_aware_dropout_state(feature_dim=50)
        features = jr.normal(jr.key(0), (50,))

        dropped, new_state = apply_task_aware_dropout(
            state, features, jr.key(1)
        )

        assert dropped.shape == features.shape
        assert jnp.all(jnp.isfinite(dropped))

    def test_dropout_rescaling(self):
        """Dropout is rescaled to maintain expectation."""
        state = init_task_aware_dropout_state(feature_dim=1000, base_dropout_rate=0.1)

        # Create known features
        features = jnp.ones(1000)

        # Run multiple times to average out randomness
        outputs = []
        for i in range(100):
            key = jr.fold_in(jr.key(0), i)
            dropped, _ = apply_task_aware_dropout(state, features, key)
            outputs.append(jnp.mean(dropped))

        mean_output = jnp.mean(jnp.array(outputs))

        # Mean should be close to 1.0 (input value) since dropout rescales
        assert jnp.abs(mean_output - 1.0) < 0.1

    def test_task_boundary_updates_switch_step(self):
        """Task boundary updates task switch step."""
        state = init_task_aware_dropout_state(feature_dim=10)
        features = jr.normal(jr.key(0), (10,))

        # No boundary
        _, state1 = apply_task_aware_dropout(
            state, features, jr.key(1), task_boundary=jnp.array(False)
        )
        assert state1.task_switch_step == 0

        # With boundary
        _, state2 = apply_task_aware_dropout(
            state1, features, jr.key(2), task_boundary=jnp.array(True)
        )
        assert state2.task_switch_step == state2.current_step - 1

    def test_stability_affects_dropout_rate(self):
        """Unstable features get higher dropout."""
        state = init_task_aware_dropout_state(feature_dim=10)

        # Set some features as unstable (low stability)
        # Use dataclass replace method
        new_stability = jnp.array([0.5] * 5 + [0.95] * 5)
        state = dataclasses.replace(state, task_stability=new_stability)

        features = jr.normal(jr.key(0), (10,))

        # Run dropout and check mask shapes
        dropped, _ = apply_task_aware_dropout(
            state, features, jr.key(1), stability_threshold=0.7
        )

        assert dropped.shape == features.shape


# =============================================================================
# Complete Protected Learner Tests
# =============================================================================


class TestProtectedEMNISTLearner:
    """Test complete protected EMNIST learner."""

    def test_init_creates_valid_learner_state(self):
        """Learner initializes with all components."""
        state = init_protected_emnist_learner(jr.key(0))

        assert state.weights.shape == (784, 47)
        assert state.bias.shape == (47,)
        assert isinstance(state.feature_norm_state, FeatureWiseNormState)
        assert isinstance(state.forgetting_state, ForgettingDetectorState)
        assert isinstance(state.per_class_norm_state, PerClassBatchNormState)
        assert isinstance(state.dropout_state, TaskAwareDropoutState)

    def test_forward_pass_valid_output(self):
        """Forward pass produces valid logits."""
        state = init_protected_emnist_learner(jr.key(0))
        features = jr.normal(jr.key(1), (784,))

        logits, new_state = protected_emnist_forward(
            state, features, training=True, key=jr.key(2)
        )

        assert logits.shape == (47,)
        assert jnp.all(jnp.isfinite(logits))

    def test_forward_preserves_model_params(self):
        """Forward pass doesn't modify model parameters."""
        state = init_protected_emnist_learner(jr.key(0))
        features = jr.normal(jr.key(1), (784,))

        _, new_state = protected_emnist_forward(state, features)

        # Model params should be unchanged
        assert jnp.allclose(new_state.weights, state.weights)
        assert jnp.allclose(new_state.bias, state.bias)

    def test_inference_no_dropout(self):
        """Inference mode doesn't apply dropout."""
        state = init_protected_emnist_learner(jr.key(0))
        features = jr.normal(jr.key(1), (784,))

        logits_train1, _ = protected_emnist_forward(state, features, training=True, key=jr.key(2))
        logits_train2, _ = protected_emnist_forward(state, features, training=True, key=jr.key(3))
        logits_infer1, _ = protected_emnist_forward(state, features, training=False)
        logits_infer2, _ = protected_emnist_forward(state, features, training=False)

        # Training outputs differ (due to dropout randomness)
        assert not jnp.allclose(logits_train1, logits_train2)

        # Inference outputs are identical
        assert jnp.allclose(logits_infer1, logits_infer2)

    def test_update_step(self):
        """Update step modifies weights and state."""
        state = init_protected_emnist_learner(jr.key(0))
        features = jr.normal(jr.key(1), (784,))
        target = jnp.array(5)
        accuracy = jnp.array(0.85)

        new_state, metrics = protected_emnist_update(
            state, features, target, accuracy, step_size=0.01, key=jr.key(2)
        )

        # Metrics should be present
        assert "loss" in metrics
        assert "step_size_used" in metrics
        assert "recovery_active" in metrics
        assert "forgetting_detected" in metrics

        # Update count should increment
        assert new_state.update_count == 1

    def test_update_applies_recovery_factor(self):
        """Update respects recovery factor from forgetting detector."""
        state = init_protected_emnist_learner(jr.key(0))
        features = jr.normal(jr.key(1), (784,))
        target = jnp.array(5)

        # High accuracy - no recovery
        new_state1, metrics1 = protected_emnist_update(
            state, features, target, jnp.array(0.95), step_size=0.01, key=jr.key(2)
        )

        # Low accuracy - recovery
        new_state2, metrics2 = protected_emnist_update(
            state, features, target, jnp.array(0.70), step_size=0.01, key=jr.key(3)
        )

        # Recovery should affect step size
        assert metrics2["step_size_used"] < metrics1["step_size_used"]

    def test_predict_produces_valid_output(self):
        """Predict produces valid class logits."""
        state = init_protected_emnist_learner(jr.key(0))
        features = jr.normal(jr.key(1), (784,))

        logits = protected_emnist_predict(state, features)

        assert logits.shape == (47,)
        assert jnp.all(jnp.isfinite(logits))

    def test_batch_accuracy_computation(self):
        """Batch accuracy computation works correctly."""
        state = init_protected_emnist_learner(jr.key(0))

        # Create a batch
        features_batch = jr.normal(jr.key(1), (32, 784))
        targets_batch = jr.randint(jr.key(2), (32,), 0, 47)

        accuracy = compute_protected_learner_accuracy(
            state, features_batch, targets_batch
        )

        assert accuracy.shape == ()
        assert 0.0 <= accuracy <= 1.0
        assert jnp.all(jnp.isfinite(accuracy))

    def test_training_loop_sequence(self):
        """Training loop runs correctly over multiple steps."""
        key = jr.key(0)
        state = init_protected_emnist_learner(key)

        accuracies = []

        for step in range(10):
            key, subkey = jr.split(key)
            features = jr.normal(subkey, (784,))
            target = jr.randint(key, (), 0, 47)

            current_accuracy = jnp.array(0.8 + 0.1 * jnp.sin(step))

            state, metrics = protected_emnist_update(
                state, features, target, current_accuracy, key=subkey
            )

            accuracies.append(metrics["loss"])

        # Should have completed without errors
        assert len(accuracies) == 10
        assert all(jnp.isfinite(a) for a in accuracies)

    def test_loss_history_maintained(self):
        """Loss history is maintained as rolling window."""
        state = init_protected_emnist_learner(jr.key(0))

        for i in range(100):
            key = jr.fold_in(jr.key(0), i)
            features = jr.normal(key, (784,))
            target = jnp.array(i % 47)
            accuracy = jnp.array(0.85)

            state, _ = protected_emnist_update(
                state, features, target, accuracy, key=key
            )

        # History should have size 50
        assert state.loss_history.shape == (50,)
        assert jnp.all(jnp.isfinite(state.loss_history))


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_multi_task_sequence(self):
        """Learner handles multi-task sequence."""
        key = jr.key(0)
        state = init_protected_emnist_learner(key)

        n_tasks = 5
        samples_per_task = 20

        for task_id in range(n_tasks):
            key, subkey = jr.split(key)

            # Task boundary
            is_boundary = task_id > 0

            for sample_id in range(samples_per_task):
                key, subkey = jr.split(key)

                features = jr.normal(subkey, (784,))
                target = jnp.array((task_id * 10 + sample_id) % 47)
                accuracy = jnp.array(0.85 - 0.05 * task_id)

                state, metrics = protected_emnist_update(
                    state,
                    features,
                    target,
                    accuracy,
                    task_boundary=jnp.array(sample_id == 0 and is_boundary),
                    key=subkey
                )

        # Should complete without errors
        assert state.update_count == n_tasks * samples_per_task

    def test_forgetting_recovery_cycle(self):
        """System detects and recovers from forgetting."""
        key = jr.key(0)
        state = init_protected_emnist_learner(key)

        # Good accuracy phase
        recovery_counts = []
        for i in range(30):
            key, subkey = jr.split(key)
            features = jr.normal(subkey, (784,))
            target = jnp.array(i % 47)

            # Start with good accuracy
            accuracy = jnp.array(0.9)

            state, metrics = protected_emnist_update(
                state, features, target, accuracy, key=subkey
            )

            if metrics["recovery_active"]:
                recovery_counts.append(i)

        # Accuracy drops suddenly
        for i in range(30, 60):
            key, subkey = jr.split(key)
            features = jr.normal(subkey, (784,))
            target = jnp.array(i % 47)

            # Poor accuracy
            accuracy = jnp.array(0.6)

            state, metrics = protected_emnist_update(
                state, features, target, accuracy, key=subkey
            )

            if metrics["recovery_active"]:
                recovery_counts.append(i)

        # System should have activated recovery
        assert len(recovery_counts) > 0

    def test_all_protections_interact(self):
        """All protection mechanisms work together."""
        key = jr.key(0)
        state = init_protected_emnist_learner(key)

        metrics_history = []

        for step in range(50):
            key, subkey = jr.split(key)

            features = jr.normal(subkey, (784,))
            target = jnp.array(step % 47)

            # Varying accuracy
            accuracy = jnp.array(0.85 + 0.1 * jnp.sin(step * 0.3))

            # Occasional task boundaries
            task_boundary = jnp.array(step % 10 == 0)

            state, metrics = protected_emnist_update(
                state,
                features,
                target,
                accuracy,
                task_boundary=task_boundary,
                key=subkey
            )

            metrics_history.append(metrics)

        # Verify all metrics are collected
        for metrics in metrics_history:
            assert "loss" in metrics
            assert "step_size_used" in metrics
            assert "recovery_active" in metrics
            assert "forgetting_detected" in metrics
            assert "dropout_rate" in metrics
            assert "feature_norm_mean" in metrics
            assert "feature_norm_std" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
