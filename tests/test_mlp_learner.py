"""Tests for the MLPLearner and run_mlp_learning_loop."""

import math
import time

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework import (
    IDBD,
    LMS,
    AGCBounding,
    Autostep,
    AutostepGTDLambda,
    BatchedMLPResult,
    EMANormalizer,
    MLPLearner,
    NormalizerTrackingConfig,
    ObGD,
    ObGDBounding,
    RandomWalkStream,
    WelfordNormalizer,
    run_mlp_learning_loop,
    run_mlp_learning_loop_batched,
)


class TestMLPLearner:
    """Tests for the MLPLearner class."""

    def test_rejects_optimizer_without_shape_generic_hooks_at_construction(self):
        with pytest.raises(ValueError, match="optimizer.*MLP"):
            MLPLearner(optimizer=ObGD())

        with pytest.raises(ValueError, match="head_optimizer.*MLP"):
            MLPLearner(head_optimizer=AutostepGTDLambda())

    def test_correct_param_shapes_single_hidden(self):
        """MLP with one hidden layer should have correct param shapes."""
        learner = MLPLearner(hidden_sizes=(32,), sparsity=0.0)
        state = learner.init(feature_dim=10, key=jr.key(42))

        # Layer 0: 10 -> 32
        chex.assert_shape(state.params.weights[0], (32, 10))
        chex.assert_shape(state.params.biases[0], (32,))
        # Layer 1: 32 -> 1
        chex.assert_shape(state.params.weights[1], (1, 32))
        chex.assert_shape(state.params.biases[1], (1,))

        assert len(state.params.weights) == 2
        assert len(state.params.biases) == 2

    def test_correct_param_shapes_two_hidden(self):
        """MLP with two hidden layers should have correct param shapes."""
        learner = MLPLearner(hidden_sizes=(64, 32), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))

        # Layer 0: 5 -> 64
        chex.assert_shape(state.params.weights[0], (64, 5))
        chex.assert_shape(state.params.biases[0], (64,))
        # Layer 1: 64 -> 32
        chex.assert_shape(state.params.weights[1], (32, 64))
        chex.assert_shape(state.params.biases[1], (32,))
        # Layer 2: 32 -> 1
        chex.assert_shape(state.params.weights[2], (1, 32))
        chex.assert_shape(state.params.biases[2], (1,))

        assert len(state.params.weights) == 3

    def test_predict_returns_scalar(self):
        """Predict should return a 1-d array (scalar prediction)."""
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.ones(5)
        prediction = learner.predict(state, observation)

        chex.assert_shape(prediction, (1,))
        chex.assert_tree_all_finite(prediction)

    def test_update_returns_correct_result(self):
        """Update should return MLPUpdateResult with correct shapes."""
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.ones(5)
        target = jnp.array([1.0])

        result = learner.update(state, observation, target)

        chex.assert_shape(result.prediction, (1,))
        chex.assert_shape(result.error, (1,))
        chex.assert_shape(result.metrics, (3,))
        chex.assert_tree_all_finite(result.metrics)

        # State should have same structure
        assert len(result.state.params.weights) == len(state.params.weights)

    def test_infinite_target_with_obgd_does_not_poison_weights(self):
        """Inf target makes ObGD scale 0, then error*step is 0*inf=NaN.

        MultiHead already refuses inf targets (NaN is the missing-head
        sentinel). MLPLearner used to commit the NaN weights. Keep the
        previous finite state so a later finite target can recover.
        """
        learner = MLPLearner(
            hidden_sizes=(4,),
            sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0),
            optimizer=LMS(0.1),
        )
        state = learner.init(feature_dim=3, key=jr.key(0))
        observation = jnp.ones(3, dtype=jnp.float32)

        poisoned = learner.update(
            state, observation, jnp.array(jnp.inf, dtype=jnp.float32)
        )
        for new_w, old_w in zip(
            poisoned.state.params.weights, state.params.weights, strict=True
        ):
            assert bool(jnp.all(jnp.isfinite(new_w)))
            chex.assert_trees_all_close(new_w, old_w)
        assert int(poisoned.state.step_count) == int(state.step_count)
        assert not bool(poisoned.update_applied)
        chex.assert_trees_all_close(poisoned.prediction, jnp.zeros_like(poisoned.prediction))
        chex.assert_trees_all_close(poisoned.error, jnp.zeros_like(poisoned.error))
        chex.assert_trees_all_close(poisoned.metrics, jnp.zeros_like(poisoned.metrics))

        recovered = learner.update(
            poisoned.state, observation, jnp.array(1.0, dtype=jnp.float32)
        )
        for new_w in recovered.state.params.weights:
            assert bool(jnp.all(jnp.isfinite(new_w)))
        assert bool(recovered.update_applied)

    def test_zero_trace_decay_does_not_multiply_inf_traces(self) -> None:
        """Default gamma*lamda is 0; 0 * inf traces is NaN and would freeze."""
        learner = MLPLearner(hidden_sizes=(4,), sparsity=0.0, optimizer=LMS(0.1))
        state = learner.init(feature_dim=3, key=jr.key(1))
        poisoned_traces = tuple(
            jnp.full_like(trace, jnp.inf) for trace in state.traces
        )
        state = state.replace(traces=poisoned_traces)
        raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
        assert not bool(jnp.isfinite(raw))

        observation = jnp.ones(3, dtype=jnp.float32)
        result = learner.update(
            state, observation, jnp.array(1.0, dtype=jnp.float32)
        )
        assert bool(result.update_applied)
        for trace in result.state.traces:
            assert bool(jnp.all(jnp.isfinite(trace)))
        for new_w in result.state.params.weights:
            assert bool(jnp.all(jnp.isfinite(new_w)))

    def test_zero_utility_decay_recovers_poisoned_utility(self) -> None:
        learner = MLPLearner(
            hidden_sizes=(4,),
            sparsity=0.0,
            optimizer=LMS(0.1),
            track_neuron_utility=True,
            neuron_utility_decay=0.0,
        )
        state = learner.init(feature_dim=3, key=jr.key(2))
        assert state.neuron_utility is not None
        state = state.replace(
            neuron_utility=tuple(
                jnp.full_like(utility, jnp.inf) for utility in state.neuron_utility
            )
        )

        result = learner.update(
            state,
            jnp.ones(3, dtype=jnp.float32),
            jnp.array(1.0, dtype=jnp.float32),
        )

        assert bool(result.update_applied)
        assert result.state.neuron_utility is not None
        for utility in result.state.neuron_utility:
            assert bool(jnp.all(jnp.isfinite(utility)))

    def test_update_reduces_error(self):
        """Multiple updates on a fixed target should reduce error."""
        learner = MLPLearner(
            hidden_sizes=(16,), step_size=0.1, bounder=ObGDBounding(kappa=2.0), sparsity=0.0
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.array([1.0, 0.5, -0.3, 0.2, 0.8])
        target = jnp.array([2.0])

        initial_error = abs(float(learner.predict(state, observation)[0]) - 2.0)

        # Run several updates
        for _ in range(50):
            result = learner.update(state, observation, target)
            state = result.state

        final_error = abs(float(learner.predict(state, observation)[0]) - 2.0)

        # Error should decrease
        assert final_error < initial_error

    def test_deterministic_with_same_key(self):
        """Same key should produce identical initial states."""
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.5)

        state1 = learner.init(feature_dim=5, key=jr.key(42))
        state2 = learner.init(feature_dim=5, key=jr.key(42))

        for w1, w2 in zip(state1.params.weights, state2.params.weights):
            chex.assert_trees_all_close(w1, w2)

    def test_sparse_init_applied(self):
        """Weights should be sparse when sparsity > 0."""
        learner = MLPLearner(hidden_sizes=(32,), sparsity=0.9)
        state = learner.init(feature_dim=10, key=jr.key(42))

        # First layer weights should be ~90% sparse
        zeros = jnp.sum(state.params.weights[0] == 0)
        total = state.params.weights[0].size
        sparsity = float(zeros) / total

        assert sparsity > 0.85  # Allow some tolerance

    def test_biases_initialized_to_zero(self):
        """All biases should be initialized to zero."""
        learner = MLPLearner(hidden_sizes=(32, 16), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))

        for bias in state.params.biases:
            chex.assert_trees_all_close(bias, jnp.zeros_like(bias))

    def test_traces_initialized_to_zero(self):
        """All eligibility traces should be initialized to zero."""
        learner = MLPLearner(hidden_sizes=(32,))
        state = learner.init(feature_dim=5, key=jr.key(42))

        for trace in state.traces:
            chex.assert_trees_all_close(trace, jnp.zeros_like(trace))


class TestRunMLPLearningLoop:
    """Tests for the run_mlp_learning_loop function."""

    def test_scan_loop_produces_correct_shapes(self):
        """Scan loop should return metrics with shape (num_steps, 3)."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, bounder=ObGDBounding(kappa=2.0)
        )

        state, metrics = run_mlp_learning_loop(
            learner, stream, num_steps=100, key=jr.key(42)
        )

        chex.assert_shape(metrics, (100, 3))
        chex.assert_tree_all_finite(metrics)

        # State should have correct param shapes
        chex.assert_shape(state.params.weights[0], (16, 5))
        chex.assert_shape(state.params.weights[1], (1, 16))

    def test_scan_loop_deterministic(self):
        """Same key should produce identical results."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, bounder=ObGDBounding(kappa=2.0)
        )

        _, metrics1 = run_mlp_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42)
        )
        _, metrics2 = run_mlp_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42)
        )

        chex.assert_trees_all_close(metrics1, metrics2)

    def test_scan_loop_with_provided_state(self):
        """Should accept a pre-initialized state."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, bounder=ObGDBounding(kappa=2.0)
        )
        initial_state = learner.init(feature_dim=5, key=jr.key(0))

        state, metrics = run_mlp_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42),
            learner_state=initial_state,
        )

        chex.assert_shape(metrics, (50, 3))
        chex.assert_tree_all_finite(metrics)


class TestBatchedMLPLearningLoop:
    """Tests for the run_mlp_learning_loop_batched function."""

    def test_batched_returns_correct_shapes(self):
        """Batched loop should return metrics with shape (num_seeds, num_steps, 3)."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, bounder=ObGDBounding(kappa=2.0)
        )
        num_seeds = 4
        num_steps = 50

        keys = jr.split(jr.key(42), num_seeds)
        result = run_mlp_learning_loop_batched(
            learner, stream, num_steps=num_steps, keys=keys
        )

        assert isinstance(result, BatchedMLPResult)
        chex.assert_shape(result.metrics, (num_seeds, num_steps, 3))
        chex.assert_tree_all_finite(result.metrics)

        # Check batched param shapes
        chex.assert_shape(result.states.params.weights[0], (num_seeds, 16, 5))
        chex.assert_shape(result.states.params.weights[1], (num_seeds, 1, 16))
        chex.assert_shape(result.states.params.biases[0], (num_seeds, 16))
        chex.assert_shape(result.states.params.biases[1], (num_seeds, 1))

    def test_batched_matches_sequential(self):
        """Batched results should match sequential results for each seed."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, bounder=ObGDBounding(kappa=2.0)
        )
        num_seeds = 3
        num_steps = 50

        keys = jr.split(jr.key(42), num_seeds)

        # Run batched
        batched_result = run_mlp_learning_loop_batched(
            learner, stream, num_steps=num_steps, keys=keys
        )

        # Run sequential
        for i in range(num_seeds):
            state_i, metrics_i = run_mlp_learning_loop(
                learner, stream, num_steps=num_steps, key=keys[i]
            )
            chex.assert_trees_all_close(
                batched_result.metrics[i], metrics_i, rtol=1e-4
            )

    def test_batched_deterministic(self):
        """Same keys should produce identical batched results."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, bounder=ObGDBounding(kappa=2.0)
        )

        keys = jr.split(jr.key(42), 3)

        result1 = run_mlp_learning_loop_batched(
            learner, stream, num_steps=50, keys=keys
        )
        result2 = run_mlp_learning_loop_batched(
            learner, stream, num_steps=50, keys=keys
        )

        chex.assert_trees_all_close(result1.metrics, result2.metrics)

    def test_batched_different_keys_different_results(self):
        """Different seeds should produce different metrics."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, bounder=ObGDBounding(kappa=2.0)
        )

        keys = jr.split(jr.key(42), 3)
        result = run_mlp_learning_loop_batched(
            learner, stream, num_steps=50, keys=keys
        )

        # Different seeds should give different final metrics
        assert not jnp.allclose(result.metrics[0], result.metrics[1])
        assert not jnp.allclose(result.metrics[0], result.metrics[2])


class TestNormalizedMLPLearner:
    """Tests for MLPLearner with normalizer parameter."""

    def test_correct_param_shapes(self):
        """MLPLearner with normalizer should have correct param shapes."""
        learner = MLPLearner(hidden_sizes=(32,), sparsity=0.0, normalizer=EMANormalizer())
        state = learner.init(feature_dim=10, key=jr.key(42))

        # MLP layer shapes
        chex.assert_shape(state.params.weights[0], (32, 10))
        chex.assert_shape(state.params.biases[0], (32,))
        chex.assert_shape(state.params.weights[1], (1, 32))
        chex.assert_shape(state.params.biases[1], (1,))

        # Normalizer state
        chex.assert_shape(state.normalizer_state.mean, (10,))
        chex.assert_shape(state.normalizer_state.var, (10,))

    def test_predict_returns_scalar(self):
        """Predict should return a 1-d array."""
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0), normalizer=EMANormalizer(),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.ones(5)
        prediction = learner.predict(state, observation)

        chex.assert_shape(prediction, (1,))
        chex.assert_tree_all_finite(prediction)

    def test_update_returns_correct_shapes(self):
        """Update should return MLPUpdateResult with 4-column metrics."""
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0), normalizer=EMANormalizer(),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.ones(5)
        target = jnp.array([1.0])

        result = learner.update(state, observation, target)

        chex.assert_shape(result.prediction, (1,))
        chex.assert_shape(result.error, (1,))
        chex.assert_shape(result.metrics, (4,))
        chex.assert_tree_all_finite(result.metrics)

    def test_normalizer_state_updates(self):
        """Normalizer state should change after an update."""
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0), normalizer=EMANormalizer(),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target = jnp.array([1.0])

        result = learner.update(state, observation, target)

        # Mean should have changed from zeros
        assert not jnp.allclose(
            result.state.normalizer_state.mean, state.normalizer_state.mean
        )

    def test_works_with_ema_normalizer(self):
        """Should work with EMANormalizer."""
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, normalizer=EMANormalizer(decay=0.95)
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.ones(5)
        target = jnp.array([1.0])

        result = learner.update(state, observation, target)
        chex.assert_shape(result.metrics, (4,))
        chex.assert_tree_all_finite(result.metrics)

    def test_works_with_welford_normalizer(self):
        """Should work with WelfordNormalizer."""
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, normalizer=WelfordNormalizer()
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.ones(5)
        target = jnp.array([1.0])

        result = learner.update(state, observation, target)
        chex.assert_shape(result.metrics, (4,))
        chex.assert_tree_all_finite(result.metrics)


class TestRunMLPNormalizedLearningLoop:
    """Tests for run_mlp_learning_loop with a normalized MLPLearner."""

    def test_scan_loop_produces_correct_shapes(self):
        """Scan loop should return metrics with shape (num_steps, 4)."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0), normalizer=EMANormalizer(),
        )

        state, metrics = run_mlp_learning_loop(
            learner, stream, num_steps=100, key=jr.key(42)
        )

        chex.assert_shape(metrics, (100, 4))
        chex.assert_tree_all_finite(metrics)

        # State should have correct param shapes
        chex.assert_shape(state.params.weights[0], (16, 5))
        chex.assert_shape(state.params.weights[1], (1, 16))

    def test_scan_loop_deterministic(self):
        """Same key should produce identical results."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0), normalizer=EMANormalizer(),
        )

        _, metrics1 = run_mlp_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42)
        )
        _, metrics2 = run_mlp_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42)
        )

        chex.assert_trees_all_close(metrics1, metrics2)

    def test_tracking_returns_3_tuple(self):
        """With normalizer tracking, should return 3-tuple."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0), normalizer=EMANormalizer(),
        )
        config = NormalizerTrackingConfig(interval=10)

        result = run_mlp_learning_loop(
            learner, stream, num_steps=100, key=jr.key(42),
            normalizer_tracking=config,
        )

        assert len(result) == 3
        state, metrics, norm_history = result

        chex.assert_shape(metrics, (100, 4))
        chex.assert_shape(norm_history.means, (10, 5))
        chex.assert_shape(norm_history.variances, (10, 5))
        chex.assert_shape(norm_history.recording_indices, (10,))

    def test_invalid_interval_raises(self):
        """Invalid tracking interval should raise ValueError."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0), normalizer=EMANormalizer(),
        )

        with pytest.raises(ValueError, match="must be >= 1"):
            run_mlp_learning_loop(
                learner, stream, num_steps=100, key=jr.key(42),
                normalizer_tracking=NormalizerTrackingConfig(interval=0),
            )

        with pytest.raises(ValueError, match="must be <= num_steps"):
            run_mlp_learning_loop(
                learner, stream, num_steps=100, key=jr.key(42),
                normalizer_tracking=NormalizerTrackingConfig(interval=200),
            )

    def test_with_provided_state(self):
        """Should accept a pre-initialized state."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0), normalizer=EMANormalizer(),
        )
        initial_state = learner.init(feature_dim=5, key=jr.key(0))

        state, metrics = run_mlp_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42),
            learner_state=initial_state,
        )

        chex.assert_shape(metrics, (50, 4))
        chex.assert_tree_all_finite(metrics)


class TestBatchedMLPNormalizedLearningLoop:
    """Tests for run_mlp_learning_loop_batched with a normalized MLPLearner."""

    def test_batched_returns_correct_shapes(self):
        """Batched loop should return metrics with shape (num_seeds, num_steps, 4)."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0), normalizer=EMANormalizer(),
        )
        num_seeds = 4
        num_steps = 50

        keys = jr.split(jr.key(42), num_seeds)
        result = run_mlp_learning_loop_batched(
            learner, stream, num_steps=num_steps, keys=keys
        )

        assert isinstance(result, BatchedMLPResult)
        chex.assert_shape(result.metrics, (num_seeds, num_steps, 4))
        chex.assert_tree_all_finite(result.metrics)
        assert result.normalizer_history is None

        # Check batched param shapes
        chex.assert_shape(
            result.states.params.weights[0], (num_seeds, 16, 5)
        )
        chex.assert_shape(
            result.states.params.weights[1], (num_seeds, 1, 16)
        )

    def test_batched_matches_sequential(self):
        """Batched results should match sequential results for each seed."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0), normalizer=EMANormalizer(),
        )
        num_seeds = 3
        num_steps = 50

        keys = jr.split(jr.key(42), num_seeds)

        # Run batched
        batched_result = run_mlp_learning_loop_batched(
            learner, stream, num_steps=num_steps, keys=keys
        )

        # Run sequential
        for i in range(num_seeds):
            state_i, metrics_i = run_mlp_learning_loop(
                learner, stream, num_steps=num_steps, key=keys[i]
            )
            chex.assert_trees_all_close(
                batched_result.metrics[i], metrics_i, rtol=1e-4
            )

    def test_batched_with_tracking(self):
        """Batched with tracking should return correct shapes."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            bounder=ObGDBounding(kappa=2.0), normalizer=EMANormalizer(),
        )
        num_seeds = 3
        num_steps = 50
        config = NormalizerTrackingConfig(interval=10)

        keys = jr.split(jr.key(42), num_seeds)
        result = run_mlp_learning_loop_batched(
            learner, stream, num_steps=num_steps, keys=keys,
            normalizer_tracking=config,
        )

        assert isinstance(result, BatchedMLPResult)
        chex.assert_shape(result.metrics, (num_seeds, num_steps, 4))
        assert result.normalizer_history is not None
        chex.assert_shape(result.normalizer_history.means, (num_seeds, 5, 5))
        chex.assert_shape(result.normalizer_history.variances, (num_seeds, 5, 5))
        chex.assert_shape(result.normalizer_history.recording_indices, (num_seeds, 5))


class TestAGCBounding:
    """Tests for the AGCBounding class."""

    def test_no_clipping_small_step(self):
        """Steps much smaller than weight norms should pass through unchanged."""
        # Large weights, tiny steps -> no clipping
        params = (jnp.ones((5, 3)) * 10.0, jnp.ones(3) * 10.0)
        steps = (jnp.ones((5, 3)) * 1e-6, jnp.ones(3) * 1e-6)
        error = jnp.array(1.0)

        bounder = AGCBounding(clip_factor=0.01, eps=1e-3)
        clipped, frac = bounder.bound(steps, error, params)

        # Steps should be unchanged
        for c, s in zip(clipped, steps):
            chex.assert_trees_all_close(c, s)

        # No units should be clipped
        assert float(frac) == 0.0

    def test_clipping_large_step(self):
        """Large steps relative to weight norms should be clipped down."""
        # Small weights, huge steps -> clipping
        params = (jnp.ones((5, 3)) * 0.1, jnp.ones(3) * 0.1)
        steps = (jnp.ones((5, 3)) * 100.0, jnp.ones(3) * 100.0)
        error = jnp.array(1.0)

        bounder = AGCBounding(clip_factor=0.01, eps=1e-3)
        clipped, frac = bounder.bound(steps, error, params)

        # Clipped steps should be smaller than original
        for c, s in zip(clipped, steps):
            assert float(jnp.max(jnp.abs(c))) < float(jnp.max(jnp.abs(s)))

        # Some units should be clipped
        assert float(frac) > 0.0

    def test_metric_fraction_clipped(self):
        """Metric should report correct fraction of clipped units."""
        # Create a scenario where exactly the 2D param gets clipped but the 1D doesn't
        # 2D: weight norm per unit = sqrt(2) * 0.01 ~ 0.014, step norm = sqrt(2) * 100 ~ 141
        # g_norm = 1.0 * 141 = 141, max_norm = max(0.014, 0.001) * 0.01 ~ 0.00014 -> clips
        # 1D: |weight| = 10.0, |step| = 1e-8
        # g_norm = 1.0 * 1e-8 = 1e-8, max_norm = max(10, 0.001) * 0.01 = 0.1 -> no clip
        params = (jnp.ones((2, 3)) * 0.01, jnp.ones(3) * 10.0)
        steps = (jnp.ones((2, 3)) * 100.0, jnp.ones(3) * 1e-8)
        error = jnp.array(1.0)

        bounder = AGCBounding(clip_factor=0.01, eps=1e-3)
        _, frac = bounder.bound(steps, error, params)

        # 2D (fan_out=2, fan_in=3) has 2 output units (all clipped), 1D has 3
        # elements (none clipped). Total units = 5, clipped = 2 -> frac = 0.4
        assert float(frac) == pytest.approx(0.4, abs=0.01)

    def test_clipping_is_per_output_unit_not_per_input_column(self):
        """Rows of (fan_out, fan_in) are the units; a loud row must not shield a quiet one."""
        bounder = AGCBounding(clip_factor=0.01, eps=1e-3)
        error = jnp.array(1.0)
        # Case 1: identical unit weights, unit 0 takes a huge step, unit 1 a tiny one.
        params = (jnp.ones((2, 2)),)
        steps = (jnp.array([[1e3, 0.0], [1e-3, 0.0]], dtype=jnp.float32),)
        (clipped,), frac = bounder.bound(steps, error, params)
        # unit 1's own budget is clip_factor * ||w_1|| = 0.01 * sqrt(2) > 1e-3: untouched
        assert float(clipped[1, 0]) == pytest.approx(1e-3, rel=1e-5)
        # unit 0 is clipped to its own budget
        assert float(jnp.linalg.norm(clipped[0])) == pytest.approx(0.01 * math.sqrt(2.0), rel=1e-4)
        assert float(frac) == pytest.approx(0.5)
        # Case 2: unit 0 has huge weights, unit 1 tiny weights; only unit 1 moves.
        params = (jnp.array([[1e3, 1e3], [1e-3, 1e-3]], dtype=jnp.float32),)
        steps = (jnp.array([[0.0, 0.0], [0.5, 0.5]], dtype=jnp.float32),)
        (clipped,), frac = bounder.bound(steps, error, params)
        # unit 1's budget is 0.01 * ||[1e-3, 1e-3]|| = 1.41e-5; its step must be clipped there
        assert float(jnp.linalg.norm(clipped[1])) == pytest.approx(0.01 * math.sqrt(2e-6), rel=1e-4)
        assert float(frac) == pytest.approx(0.5)

    def test_mlp_with_agc_runs(self):
        """MLPLearner with AGCBounding should run without error in a scan loop."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, bounder=AGCBounding(clip_factor=0.01)
        )

        state, metrics = run_mlp_learning_loop(
            learner, stream, num_steps=100, key=jr.key(42)
        )

        chex.assert_shape(metrics, (100, 3))
        chex.assert_tree_all_finite(metrics)

    def test_mlp_agc_reduces_error(self):
        """Multi-step MLP training with AGC should reduce error."""
        learner = MLPLearner(
            hidden_sizes=(16,), step_size=0.1,
            bounder=AGCBounding(clip_factor=0.01), sparsity=0.0,
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.array([1.0, 0.5, -0.3, 0.2, 0.8])
        target = jnp.array([2.0])

        initial_error = abs(float(learner.predict(state, observation)[0]) - 2.0)

        for _ in range(50):
            result = learner.update(state, observation, target)
            state = result.state

        final_error = abs(float(learner.predict(state, observation)[0]) - 2.0)

        assert final_error < initial_error


class TestLayerNormToggle:
    """Tests for the use_layer_norm parameter on MLPLearner."""

    def test_layer_norm_disabled_runs(self):
        """MLPLearner with use_layer_norm=False can init, predict, and update."""
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, use_layer_norm=False,
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        observation = jnp.ones(5)
        target = jnp.array([1.0])

        prediction = learner.predict(state, observation)
        chex.assert_shape(prediction, (1,))
        chex.assert_tree_all_finite(prediction)

        result = learner.update(state, observation, target)
        chex.assert_shape(result.prediction, (1,))
        chex.assert_shape(result.error, (1,))
        chex.assert_shape(result.metrics, (3,))
        chex.assert_tree_all_finite(result.metrics)

    def test_layer_norm_disabled_different_predictions(self):
        """Predictions differ between use_layer_norm=True and False."""
        learner_ln = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, use_layer_norm=True,
        )
        learner_no_ln = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, use_layer_norm=False,
        )

        # Same key -> same initial weights
        state_ln = learner_ln.init(feature_dim=5, key=jr.key(42))
        state_no_ln = learner_no_ln.init(feature_dim=5, key=jr.key(42))

        observation = jnp.array([1.0, 0.5, -0.3, 0.2, 0.8])

        pred_ln = learner_ln.predict(state_ln, observation)
        pred_no_ln = learner_no_ln.predict(state_no_ln, observation)

        assert not jnp.allclose(pred_ln, pred_no_ln)

    def test_layer_norm_default_true(self):
        """Default MLPLearner should use layer norm (backwards compatible)."""
        learner_default = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        learner_explicit = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, use_layer_norm=True,
        )

        state_default = learner_default.init(feature_dim=5, key=jr.key(42))
        state_explicit = learner_explicit.init(feature_dim=5, key=jr.key(42))

        observation = jnp.array([1.0, 0.5, -0.3, 0.2, 0.8])

        pred_default = learner_default.predict(state_default, observation)
        pred_explicit = learner_explicit.predict(state_explicit, observation)

        chex.assert_trees_all_close(pred_default, pred_explicit)

    def test_layer_norm_disabled_batched(self):
        """run_mlp_learning_loop_batched works with use_layer_norm=False."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0, use_layer_norm=False,
            bounder=ObGDBounding(kappa=2.0),
        )
        num_seeds = 3
        num_steps = 50

        keys = jr.split(jr.key(42), num_seeds)
        result = run_mlp_learning_loop_batched(
            learner, stream, num_steps=num_steps, keys=keys
        )

        assert isinstance(result, BatchedMLPResult)
        chex.assert_shape(result.metrics, (num_seeds, num_steps, 3))
        chex.assert_tree_all_finite(result.metrics)


class TestMLPLifecycleTracking:
    """Tests for MLP agent lifecycle tracking."""

    def test_step_count_starts_at_zero(self):
        """step_count should be 0 after init."""
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))
        assert int(state.step_count) == 0

    def test_step_count_increments(self):
        """step_count should increment on update."""
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        target = jnp.array([1.0])
        result = learner.update(state, obs, target)
        assert int(result.state.step_count) == 1

    def test_birth_timestamp_set(self):
        """birth_timestamp should be set at init."""
        before = time.time()
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))
        after = time.time()
        assert before <= state.birth_timestamp <= after

    def test_birth_timestamp_survives_update(self):
        """birth_timestamp should not change across updates."""
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        state = learner.init(feature_dim=5, key=jr.key(42))
        original_ts = state.birth_timestamp

        obs = jnp.ones(5)
        target = jnp.array([1.0])
        result = learner.update(state, obs, target)
        assert result.state.birth_timestamp == original_ts

    def test_uptime_increases_after_loop(self):
        """uptime_s should be > 0 after run_mlp_learning_loop."""
        stream = RandomWalkStream(feature_dim=5)
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)

        state, _ = run_mlp_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42)
        )
        assert state.uptime_s > 0.0

    def test_step_count_after_loop(self):
        """step_count should equal num_steps after learning loop."""
        stream = RandomWalkStream(feature_dim=5)
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)

        state, _ = run_mlp_learning_loop(
            learner, stream, num_steps=100, key=jr.key(42)
        )
        assert int(state.step_count) == 100


class TestHybridOptimizer:
    """Tests for MLPLearner with head_optimizer (hybrid trunk/head optimizers)."""

    def test_hybrid_init_creates_different_states(self):
        """Output layer optimizer state type should differ from trunk when using hybrid."""
        from alberta_framework.core.types import AutostepParamState, LMSState

        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            step_size=1.0,
            head_optimizer=Autostep(initial_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        # Interleaved: w0, b0, w1, b1
        # w0, b0 = trunk (LMS), w1, b1 = output (Autostep)
        assert isinstance(state.optimizer_states[0], LMSState)
        assert isinstance(state.optimizer_states[1], LMSState)
        assert isinstance(state.optimizer_states[2], AutostepParamState)
        assert isinstance(state.optimizer_states[3], AutostepParamState)

    def test_hybrid_update_runs(self):
        """Basic update with LMS trunk + Autostep head should work."""
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            step_size=1.0,
            head_optimizer=Autostep(initial_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        target = jnp.array([1.0])

        result = learner.update(state, obs, target)
        chex.assert_shape(result.prediction, (1,))
        chex.assert_shape(result.error, (1,))
        chex.assert_shape(result.metrics, (3,))
        chex.assert_tree_all_finite(result.metrics)

    def test_hybrid_scan_loop(self):
        """Full learning loop should work with hybrid optimizer."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            step_size=1.0,
            head_optimizer=Autostep(initial_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
        )

        state, metrics = run_mlp_learning_loop(
            learner, stream, num_steps=100, key=jr.key(42)
        )

        chex.assert_shape(metrics, (100, 3))
        chex.assert_tree_all_finite(metrics)
        assert int(state.step_count) == 100

    def test_hybrid_default_none_matches_uniform(self):
        """head_optimizer=None should produce same results as explicit single optimizer."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)

        learner_default = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            step_size=1.0,
            bounder=ObGDBounding(kappa=2.0),
        )
        learner_explicit = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            step_size=1.0,
            head_optimizer=None,
            bounder=ObGDBounding(kappa=2.0),
        )

        _, metrics_default = run_mlp_learning_loop(
            learner_default, stream, num_steps=50, key=jr.key(42)
        )
        _, metrics_explicit = run_mlp_learning_loop(
            learner_explicit, stream, num_steps=50, key=jr.key(42)
        )

        chex.assert_trees_all_close(metrics_default, metrics_explicit)

    def test_hybrid_two_hidden_layers(self):
        """Hybrid optimizer should work with two hidden layers."""
        from alberta_framework.core.types import AutostepParamState, LMSState

        learner = MLPLearner(
            hidden_sizes=(32, 16), sparsity=0.0,
            step_size=1.0,
            head_optimizer=Autostep(initial_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        # Interleaved: w0, b0, w1, b1, w2, b2
        # w0, b0, w1, b1 = trunk (LMS), w2, b2 = output (Autostep)
        assert isinstance(state.optimizer_states[0], LMSState)
        assert isinstance(state.optimizer_states[1], LMSState)
        assert isinstance(state.optimizer_states[2], LMSState)
        assert isinstance(state.optimizer_states[3], LMSState)
        assert isinstance(state.optimizer_states[4], AutostepParamState)
        assert isinstance(state.optimizer_states[5], AutostepParamState)

        # Should run without error
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        state, metrics = run_mlp_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42)
        )
        chex.assert_shape(metrics, (50, 3))
        chex.assert_tree_all_finite(metrics)

    def test_hybrid_batched(self):
        """Batched loop should work with hybrid optimizer."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,), sparsity=0.0,
            step_size=1.0,
            head_optimizer=Autostep(initial_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
        )

        keys = jr.split(jr.key(42), 3)
        result = run_mlp_learning_loop_batched(
            learner, stream, num_steps=50, keys=keys
        )

        assert isinstance(result, BatchedMLPResult)
        chex.assert_shape(result.metrics, (3, 50, 3))
        chex.assert_tree_all_finite(result.metrics)


class TestMLPLearnerWithIDBD:
    """Tests for MLPLearner with IDBD optimizer (Meyer MLP adaptation)."""

    def test_init_creates_idbd_param_states(self):
        """Init should create IDBDParamState for each parameter."""
        from alberta_framework.core.types import IDBDParamState

        learner = MLPLearner(
            hidden_sizes=(16,),
            sparsity=0.0,
            optimizer=IDBD(initial_step_size=0.01, meta_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        for opt_state in state.optimizer_states:
            assert isinstance(opt_state, IDBDParamState)

    def test_basic_update_runs(self):
        """Basic update with IDBD optimizer should run without error."""
        learner = MLPLearner(
            hidden_sizes=(16,),
            sparsity=0.0,
            optimizer=IDBD(initial_step_size=0.01, meta_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        obs = jnp.ones(5)
        target = jnp.array([1.0])

        result = learner.update(state, obs, target)
        chex.assert_shape(result.prediction, (1,))
        chex.assert_shape(result.error, (1,))
        chex.assert_shape(result.metrics, (3,))
        chex.assert_tree_all_finite(result.metrics)

    def test_learning_loop_produces_finite_metrics(self):
        """Learning loop with IDBD should produce finite metrics."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,),
            sparsity=0.0,
            optimizer=IDBD(initial_step_size=0.01, meta_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
        )

        state, metrics = run_mlp_learning_loop(
            learner, stream, num_steps=100, key=jr.key(42)
        )

        chex.assert_shape(metrics, (100, 3))
        chex.assert_tree_all_finite(metrics)
        assert int(state.step_count) == 100

    def test_loss_grads_mode(self):
        """IDBD with loss_grads h_decay_mode should work with MLP."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,),
            sparsity=0.0,
            optimizer=IDBD(
                initial_step_size=0.01,
                meta_step_size=0.01,
                h_decay_mode="loss_grads",
            ),
            bounder=ObGDBounding(kappa=2.0),
        )

        state, metrics = run_mlp_learning_loop(
            learner, stream, num_steps=100, key=jr.key(42)
        )

        chex.assert_shape(metrics, (100, 3))
        chex.assert_tree_all_finite(metrics)

    def test_hybrid_lms_trunk_idbd_head(self):
        """Hybrid LMS trunk + IDBD head should work."""
        from alberta_framework.core.types import IDBDParamState, LMSState

        learner = MLPLearner(
            hidden_sizes=(16,),
            sparsity=0.0,
            step_size=1.0,
            head_optimizer=IDBD(initial_step_size=0.01, meta_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
        )
        state = learner.init(feature_dim=5, key=jr.key(42))

        # Trunk: LMS, Head: IDBD
        assert isinstance(state.optimizer_states[0], LMSState)
        assert isinstance(state.optimizer_states[1], LMSState)
        assert isinstance(state.optimizer_states[2], IDBDParamState)
        assert isinstance(state.optimizer_states[3], IDBDParamState)

        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        state, metrics = run_mlp_learning_loop(
            learner, stream, num_steps=50, key=jr.key(42)
        )
        chex.assert_shape(metrics, (50, 3))
        chex.assert_tree_all_finite(metrics)

    def test_batched_learning_loop(self):
        """Batched loop should work with IDBD."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,),
            sparsity=0.0,
            optimizer=IDBD(initial_step_size=0.01, meta_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
        )

        keys = jr.split(jr.key(42), 3)
        result = run_mlp_learning_loop_batched(
            learner, stream, num_steps=50, keys=keys
        )

        assert isinstance(result, BatchedMLPResult)
        chex.assert_shape(result.metrics, (3, 50, 3))
        chex.assert_tree_all_finite(result.metrics)

    def test_with_normalizer(self):
        """IDBD + EMANormalizer should work together."""
        stream = RandomWalkStream(feature_dim=5, drift_rate=0.001)
        learner = MLPLearner(
            hidden_sizes=(16,),
            sparsity=0.0,
            optimizer=IDBD(initial_step_size=0.01, meta_step_size=0.01),
            bounder=ObGDBounding(kappa=2.0),
            normalizer=EMANormalizer(decay=0.99),
        )

        state, metrics = run_mlp_learning_loop(
            learner, stream, num_steps=100, key=jr.key(42)
        )

        chex.assert_shape(metrics, (100, 4))
        chex.assert_tree_all_finite(metrics)


class TestNeuronUtility:
    """Tests for per-neuron utility tracking and dormant-neuron reset."""

    def test_utility_disabled_by_default(self):
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0)
        state = learner.init(feature_dim=8, key=jr.key(0))
        assert state.neuron_utility is None

    def test_utility_shape_single_hidden(self):
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0, track_neuron_utility=True)
        state = learner.init(feature_dim=8, key=jr.key(0))
        assert state.neuron_utility is not None
        assert len(state.neuron_utility) == 1
        chex.assert_shape(state.neuron_utility[0], (16,))

    def test_utility_shape_two_hidden(self):
        learner = MLPLearner(hidden_sizes=(16, 8), sparsity=0.0, track_neuron_utility=True)
        state = learner.init(feature_dim=5, key=jr.key(1))
        assert state.neuron_utility is not None
        assert len(state.neuron_utility) == 2
        chex.assert_shape(state.neuron_utility[0], (16,))
        chex.assert_shape(state.neuron_utility[1], (8,))

    def test_utility_initialised_to_zero(self):
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0, track_neuron_utility=True)
        state = learner.init(feature_dim=8, key=jr.key(2))
        assert state.neuron_utility is not None
        assert float(jnp.sum(jnp.abs(state.neuron_utility[0]))) == 0.0

    def test_utility_increases_after_updates(self):
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0, track_neuron_utility=True)
        state = learner.init(feature_dim=8, key=jr.key(3))
        obs = jnp.ones(8, dtype=jnp.float32)
        for _ in range(20):
            result = learner.update(state, obs, jnp.array([1.0]))
            state = result.state
        assert state.neuron_utility is not None
        assert float(jnp.sum(state.neuron_utility[0])) > 0.0

    def test_utility_ema_decay(self):
        """After one update, utility should equal (1 - decay) * grad_norm."""
        decay = 0.9
        learner = MLPLearner(
            hidden_sizes=(4,), sparsity=0.0, track_neuron_utility=True,
            neuron_utility_decay=decay,
        )
        state = learner.init(feature_dim=4, key=jr.key(4))
        obs = jnp.ones(4, dtype=jnp.float32)
        result = learner.update(state, obs, jnp.array([1.0]))
        new_state = result.state
        assert new_state.neuron_utility is not None
        # First update: utility = 0 * decay + (1 - decay) * grad_norm
        # utility should be strictly positive for non-zero gradient
        assert float(jnp.sum(new_state.neuron_utility[0])) > 0.0

    def test_dormant_fraction_zero_after_init(self):
        learner = MLPLearner(hidden_sizes=(16,), sparsity=0.0, track_neuron_utility=True)
        state = learner.init(feature_dim=8, key=jr.key(5))
        frac = MLPLearner.dormant_neuron_fraction(state, threshold=0.01)
        assert frac == 1.0  # all zero at init → all dormant

    def test_dormant_fraction_decreases_after_updates(self):
        learner = MLPLearner(
            hidden_sizes=(8,), sparsity=0.0, track_neuron_utility=True,
            neuron_utility_decay=0.5,  # faster convergence
        )
        state = learner.init(feature_dim=4, key=jr.key(6))
        obs = jr.normal(jr.key(7), (4,))
        for _ in range(50):
            result = learner.update(state, obs, jnp.array([1.0]))
            state = result.state
        frac = MLPLearner.dormant_neuron_fraction(state, threshold=1e-6)
        assert frac < 1.0  # at least some neurons are active

    def test_dormant_fraction_no_tracking(self):
        learner = MLPLearner(hidden_sizes=(8,), sparsity=0.0)
        state = learner.init(feature_dim=4, key=jr.key(8))
        frac = MLPLearner.dormant_neuron_fraction(state)
        assert frac == 0.0

    def test_reset_dormant_neurons_reduces_dormant_fraction(self):
        learner = MLPLearner(
            hidden_sizes=(8,), sparsity=0.0, track_neuron_utility=True,
        )
        state = learner.init(feature_dim=4, key=jr.key(9))
        # All utility = 0 → all dormant at very low threshold
        assert MLPLearner.dormant_neuron_fraction(state, threshold=1.0) == 1.0
        state_reset = learner.reset_dormant_neurons(state, jr.key(10), threshold=1.0)
        # Weights should have changed
        assert not jnp.allclose(
            state_reset.params.weights[0], state.params.weights[0]
        )
        # Utility for reset neurons is zeroed
        assert state_reset.neuron_utility is not None
        assert float(jnp.sum(state_reset.neuron_utility[0])) == 0.0

    def test_config_roundtrip_with_tracking(self):
        learner = MLPLearner(
            hidden_sizes=(8,), sparsity=0.0, track_neuron_utility=True,
            neuron_utility_decay=0.95,
        )
        cfg = learner.to_config()
        assert cfg["track_neuron_utility"] is True
        assert cfg["neuron_utility_decay"] == pytest.approx(0.95)
        restored = MLPLearner.from_config(cfg)
        assert restored._track_neuron_utility is True
        assert restored._neuron_utility_decay == pytest.approx(0.95)


class TestResetDormantOptimizerState:
    """Collapse-reset-recover: dormant reset must reinitialise optimizer slices.

    ``reset_dormant_neurons`` must splice ``init_for_shape`` values into the
    optimizer state for reset neurons (not zeros: IDBD stores *log* step-sizes
    where zero means exp(0)=1.0, and Autostep step-sizes adapt multiplicatively
    so zero sticks at zero) and must leave shared scalar leaves (e.g.
    ``meta_step_size``) scalar.
    """

    N_HIDDEN = 8
    FEATURE_DIM = 4
    N_RESET = 3

    @pytest.mark.parametrize(
        "make_optimizer",
        [
            lambda: LMS(step_size=0.05),
            lambda: IDBD(initial_step_size=0.05, meta_step_size=0.01),
            lambda: Autostep(initial_step_size=0.05, meta_step_size=0.01),
        ],
        ids=["lms", "idbd", "autostep"],
    )
    def test_collapse_reset_recover(self, make_optimizer):
        optimizer = make_optimizer()
        learner = MLPLearner(
            hidden_sizes=(self.N_HIDDEN,),
            optimizer=optimizer,
            sparsity=0.0,
            track_neuron_utility=True,
            neuron_utility_decay=0.9,
        )
        state = learner.init(feature_dim=self.FEATURE_DIM, key=jr.key(0))

        # Learnable linear target so error can recover after the reset.
        w_true = jnp.array([0.5, -0.3, 0.8, 0.1], dtype=jnp.float32)
        obs_seq = jr.normal(jr.key(1), (1000, self.FEATURE_DIM), dtype=jnp.float32)

        for t in range(200):
            obs = obs_seq[t % obs_seq.shape[0]]
            state = learner.update(state, obs, jnp.dot(w_true, obs)[None]).state

        # Force exactly the first N_RESET neurons dormant.
        assert state.neuron_utility is not None
        forced = jnp.where(
            jnp.arange(self.N_HIDDEN) < self.N_RESET, 0.0, 1.0
        ).astype(jnp.float32)
        state = state.replace(neuron_utility=(forced,))
        reset_state = learner.reset_dormant_neurons(state, jr.key(2), threshold=0.5)

        mask = jnp.arange(self.N_HIDDEN) < self.N_RESET
        fresh_states = (
            optimizer.init_for_shape((self.N_HIDDEN, self.FEATURE_DIM)),
            optimizer.init_for_shape((self.N_HIDDEN,)),
        )
        for j, fresh in enumerate(fresh_states):
            old_leaves = jax.tree_util.tree_leaves(state.optimizer_states[j])
            new_leaves = jax.tree_util.tree_leaves(reset_state.optimizer_states[j])
            fresh_leaves = jax.tree_util.tree_leaves(fresh)
            for old, new, ref in zip(old_leaves, new_leaves, fresh_leaves):
                # Leaf shapes must be preserved: scalar leaves stay scalar.
                assert new.shape == old.shape
                if new.ndim == 0:
                    # Shared scalars (e.g. meta_step_size) must be untouched.
                    assert jnp.allclose(new, old)
                else:
                    sel = mask[:, None] if new.ndim == 2 else mask
                    # Reset slices equal fresh init_for_shape values.
                    assert jnp.allclose(
                        jnp.where(sel, new, 0.0), jnp.where(sel, ref, 0.0)
                    )
                    # Non-reset slices are untouched.
                    assert jnp.allclose(
                        jnp.where(sel, 0.0, new), jnp.where(sel, 0.0, old)
                    )

        # Recovery: 1000 further updates must stay finite and reduce error.
        state = reset_state
        sq_errors = []
        for t in range(1000):
            obs = obs_seq[t]
            result = learner.update(state, obs, jnp.dot(w_true, obs)[None])
            state = result.state
            sq_errors.append(float(jnp.squeeze(result.error)) ** 2)

        for leaf in jax.tree_util.tree_leaves(state):
            assert bool(jnp.all(jnp.isfinite(leaf)))
        for j in range(len(state.optimizer_states)):
            old_leaves = jax.tree_util.tree_leaves(reset_state.optimizer_states[j])
            new_leaves = jax.tree_util.tree_leaves(state.optimizer_states[j])
            for old, new in zip(old_leaves, new_leaves):
                assert new.shape == old.shape
        mse_first = sum(sq_errors[:100]) / 100.0
        mse_last = sum(sq_errors[-100:]) / 100.0
        assert mse_last < mse_first


class TestMLPLearnerConstructorScalars:
    """Leftover constructor scalars must not sit on the single-head MLP host."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("sparsity", True),
            ("sparsity", False),
            ("sparsity", float("nan")),
            ("sparsity", float("inf")),
            ("sparsity", 1.5),
            ("gamma", True),
            ("gamma", float("nan")),
            ("lamda", True),
            ("lamda", float("nan")),
            ("leaky_relu_slope", True),
            ("leaky_relu_slope", float("nan")),
            ("step_size", True),
            ("step_size", False),
            ("step_size", float("nan")),
            ("step_size", 0.0),
            ("neuron_utility_decay", True),
            ("neuron_utility_decay", float("nan")),
            ("neuron_utility_decay", 1.0),
        ],
    )
    def test_rejects_leftover_constructor_scalars(self, field: str, value: object) -> None:
        kwargs: dict[str, object] = {"hidden_sizes": (8,), "sparsity": 0.0}
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            MLPLearner(**kwargs)  # type: ignore[arg-type]

    def test_true_sparsity_does_not_serialize_as_full_mask(self) -> None:
        with pytest.raises(ValueError, match="sparsity"):
            MLPLearner(hidden_sizes=(8,), sparsity=True)

    def test_true_gamma_does_not_store_as_undiscounted_one(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            MLPLearner(hidden_sizes=(8,), gamma=True, lamda=0.0)

    def test_bool_flags_require_exact_bool(self) -> None:
        with pytest.raises(ValueError, match="use_layer_norm"):
            MLPLearner(hidden_sizes=(8,), use_layer_norm=1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="track_neuron_utility"):
            MLPLearner(hidden_sizes=(8,), track_neuron_utility=1)  # type: ignore[arg-type]

    def test_from_config_rejects_nan_sparsity_and_bool_gamma(self) -> None:
        config = MLPLearner(hidden_sizes=(8,), sparsity=0.0).to_config()
        poisoned = dict(config)
        poisoned["sparsity"] = float("nan")
        with pytest.raises(ValueError, match="sparsity"):
            MLPLearner.from_config(poisoned)
        poisoned = dict(config)
        poisoned["gamma"] = True
        with pytest.raises(ValueError, match="gamma"):
            MLPLearner.from_config(poisoned)

    def test_rejects_class_spoofed_sparsity_without_invoking_float(self) -> None:
        class Spoof:
            @property
            def __class__(self) -> type[float]:  # type: ignore[override]
                return float

            def __float__(self) -> float:
                raise RuntimeError("must not run")

        with pytest.raises(ValueError, match="sparsity"):
            MLPLearner(hidden_sizes=(8,), sparsity=Spoof())  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sparsity": 2.0**-150},
            {"leaky_relu_slope": 5e-324},
            {"gamma": 2.0**-150},
            {"lamda": 2.0**-150},
            {"neuron_utility_decay": 5e-324},
        ],
    )
    def test_rejects_nonzero_float32_underflow(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="must remain nonzero"):
            MLPLearner(hidden_sizes=(8,), **kwargs)  # type: ignore[arg-type]

    def test_rejects_trace_product_that_underflows_float32(self) -> None:
        with pytest.raises(ValueError, match=r"gamma \* lamda must remain nonzero"):
            MLPLearner(hidden_sizes=(), gamma=2.0**-100, lamda=2.0**-100)

    def test_legal_zeros_and_exact_bools_still_construct(self) -> None:
        smallest_float32 = 2.0**-149
        learner = MLPLearner(
            hidden_sizes=(8,),
            sparsity=0.0,
            leaky_relu_slope=0.0,
            gamma=0.0,
            lamda=smallest_float32,
            use_layer_norm=False,
            track_neuron_utility=True,
            neuron_utility_decay=0.0,
            step_size=0.1,
        )
        config = learner.to_config()
        assert config["sparsity"] == 0.0
        assert config["leaky_relu_slope"] == 0.0
        assert config["lamda"] == smallest_float32
        assert config["use_layer_norm"] is False
        assert config["track_neuron_utility"] is True
        assert config["neuron_utility_decay"] == 0.0


def test_mlp_dimensions_are_exact_canonical_and_preflighted() -> None:
    for invalid in ([8], (True,), (0,), (2**31,)):
        with pytest.raises(ValueError, match="hidden_sizes"):
            MLPLearner(hidden_sizes=invalid)  # type: ignore[arg-type]

    learner = MLPLearner(hidden_sizes=(np.longlong(8),))
    assert learner._hidden_sizes == (8,)
    assert type(learner._hidden_sizes[0]) is int

    with pytest.raises(ValueError, match="feature_dim"):
        learner.init(True, jr.key(0))
    with pytest.raises(ValueError, match="resource"):
        MLPLearner(hidden_sizes=(2**26,)).init(2, jr.key(0))
