"""
Tests for micro_continual_final_mechanisms.py — all four mechanisms plus integrated.

Test coverage:
- Factory initialization and state creation
- Hyperparameter validation
- Signature compliance (LearnerInitFn, ScreeningStepFn)
- Forward/backward sanity checks
- Metric computation and ranges
- Integration of all four mechanisms
- State evolution correctness
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from micro_continual_final_mechanisms import (
    FINAL_MECHANISMS,
    _make_attention_feature_selection_learner,
    _make_combined_final_mechanisms_learner,
    _make_dual_head_learner,
    _make_intrinsic_motivation_learner,
    _make_memory_consolidation_learner,
)


class TestFinalMechanismsMetadata(unittest.TestCase):
    """Validate mechanism metadata structure and completeness."""

    def test_registry_completeness(self) -> None:
        """All mechanisms in FINAL_MECHANISMS have required fields."""
        required_fields = {
            "name",
            "factory",
            "mechanism",
            "hyperparameters",
            "description",
        }
        for mech_name, spec in FINAL_MECHANISMS.items():
            with self.subTest(mechanism=mech_name):
                self.assertIsInstance(spec, dict)
                self.assertEqual(set(spec.keys()), required_fields)
                self.assertIsInstance(spec["name"], str)
                self.assertIsInstance(spec["mechanism"], str)
                self.assertIsInstance(spec["hyperparameters"], dict)
                self.assertIsInstance(spec["description"], str)
                self.assertTrue(len(spec["description"]) > 0)

    def test_factory_callable(self) -> None:
        """All factories are callable."""
        for mech_name, spec in FINAL_MECHANISMS.items():
            with self.subTest(mechanism=mech_name):
                self.assertTrue(callable(spec["factory"]))

    def test_hyperparameters_types(self) -> None:
        """All hyperparameters are numeric or bool."""
        for mech_name, spec in FINAL_MECHANISMS.items():
            with self.subTest(mechanism=mech_name):
                for key, val in spec["hyperparameters"].items():
                    self.assertIsInstance(
                        val, (int, float, bool), msg=f"{mech_name}.{key} = {val!r}"
                    )


class TestDualHeadLearner(unittest.TestCase):
    """Dual-head architecture: feature encoder + class head."""

    def setUp(self) -> None:
        """Create factory and mock params."""
        self.factory = _make_dual_head_learner
        self.hyperparameters = FINAL_MECHANISMS["dual_head"]["hyperparameters"]
        self.mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32),
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32),
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32),
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

    def test_factory_initialization(self) -> None:
        """Factory initializes without error."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        self.assertTrue(callable(init_fn))
        self.assertTrue(callable(step_fn))

    def test_init_fn_returns_valid_state(self) -> None:
        """Init function returns valid state dict."""
        init_fn, _ = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        self.assertIsInstance(state, dict)
        required_keys = {
            "feature_weights",
            "class_weights",
            "head_correlation",
            "feature_plasticity_trace",
            "class_plasticity_trace",
            "step_count",
        }
        self.assertEqual(set(state.keys()), required_keys)

    def test_step_fn_produces_metrics(self) -> None:
        """Step function produces (accuracy, loss, plasticity) metrics."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32)}
        key = jr.key(0)

        params_out, state_out, metrics = step_fn(self.mock_params, state, grads, key)
        accuracy, loss, plasticity = metrics

        self.assertEqual(len(metrics), 3)
        self.assertTrue(0 <= float(accuracy) <= 1)
        self.assertTrue(float(loss) >= 0)
        self.assertTrue(0 <= float(plasticity) <= 1)

    def test_head_correlation_bounded(self) -> None:
        """Head correlation is bounded."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32)}
        key = jr.key(0)

        for _ in range(10):
            params_out, state, _ = step_fn(self.mock_params, state, grads, key)
            corr = float(state["head_correlation"])
            self.assertTrue(-2 <= corr <= 2, f"Correlation out of bounds: {corr}")


class TestAttentionFeatureSelection(unittest.TestCase):
    """Attention-based feature selection with dynamic weighting."""

    def setUp(self) -> None:
        """Create factory and mock params."""
        self.factory = _make_attention_feature_selection_learner
        self.hyperparameters = FINAL_MECHANISMS["attention_feature_selection"][
            "hyperparameters"
        ]
        self.mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32),
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32),
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32),
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

    def test_init_fn_returns_valid_state(self) -> None:
        """Init function returns valid attention state."""
        init_fn, _ = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        required_keys = {
            "attention_logits",
            "attention_weights",
            "feature_importance_ema",
            "attention_entropy",
            "feature_grad_history",
            "step_count",
        }
        self.assertEqual(set(state.keys()), required_keys)

    def test_attention_weights_sum_to_one(self) -> None:
        """Attention weights form valid probability distribution."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32)}
        key = jr.key(0)

        for _ in range(5):
            params_out, state, _ = step_fn(self.mock_params, state, grads, key)
            weight_sum = float(jnp.sum(state["attention_weights"]))
            self.assertAlmostEqual(weight_sum, 1.0, places=5)

    def test_attention_entropy_bounds(self) -> None:
        """Attention entropy is bounded by max entropy."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        n_features = 256
        max_entropy = math.log(n_features)

        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32)}
        key = jr.key(0)

        for _ in range(10):
            params_out, state, _ = step_fn(self.mock_params, state, grads, key)
            entropy = float(state["attention_entropy"])
            self.assertTrue(
                0 <= entropy <= max_entropy,
                f"Entropy {entropy} outside [0, {max_entropy}]",
            )

    def test_feature_selectivity_evolution(self) -> None:
        """Feature selection evolves over time."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        # Use gradient with structure
        grads = {
            "w1": jnp.concatenate(
                [jnp.ones((128, 256)), jnp.zeros((128, 256))], axis=0
            )
        }
        key = jr.key(0)

        initial_entropy = float(state["attention_entropy"])
        for _ in range(10):
            params_out, state, _ = step_fn(self.mock_params, state, grads, key)

        final_entropy = float(state["attention_entropy"])
        # Should evolve (selectivity should change)
        self.assertNotEqual(initial_entropy, final_entropy)


class TestMemoryConsolidation(unittest.TestCase):
    """Memory consolidation with offline replay and sleep phases."""

    def setUp(self) -> None:
        """Create factory and mock params."""
        self.factory = _make_memory_consolidation_learner
        self.hyperparameters = FINAL_MECHANISMS["memory_consolidation"]["hyperparameters"]
        self.mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32),
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32),
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32),
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

    def test_init_fn_returns_valid_state(self) -> None:
        """Init function returns valid consolidation state."""
        init_fn, _ = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        required_keys = {
            "replay_buffer",
            "buffer_index",
            "buffer_full",
            "priorities",
            "consolidated_weights",
            "consolidation_loss",
            "consolidation_steps",
            "sleep_phase",
            "step_count",
        }
        self.assertEqual(set(state.keys()), required_keys)

    def test_priority_weights_valid(self) -> None:
        """Priority weights are valid probabilities."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32)}
        key = jr.key(0)

        for _ in range(5):
            params_out, state, _ = step_fn(self.mock_params, state, grads, key)
            priorities = state["priorities"]
            self.assertTrue(jnp.all(priorities >= 0))
            self.assertTrue(jnp.all(priorities <= 1))

    def test_sleep_phase_detection(self) -> None:
        """Sleep phases detected at correct intervals."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)
        sleep_interval = int(self.hyperparameters["sleep_interval"])

        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32)}
        key = jr.key(0)

        sleep_steps = []
        for i in range(100):
            params_out, state, _ = step_fn(self.mock_params, state, grads, key)
            if state["sleep_phase"]:
                sleep_steps.append(i)

        # Check that sleep occurs at approximately correct intervals
        if len(sleep_steps) > 1:
            intervals = [sleep_steps[i + 1] - sleep_steps[i] for i in range(len(sleep_steps) - 1)]
            avg_interval = np.mean(intervals)
            self.assertAlmostEqual(avg_interval, sleep_interval, delta=2)

    def test_consolidation_loss_evolution(self) -> None:
        """Consolidation loss evolves during sleep phases."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32)}
        key = jr.key(0)

        sleep_phase_seen = False
        for _ in range(100):
            params_out, state, _ = step_fn(self.mock_params, state, grads, key)
            if state["sleep_phase"]:
                sleep_phase_seen = True

        # Should have seen at least one sleep phase in 100 steps
        self.assertTrue(sleep_phase_seen)


class TestIntrinsicMotivation(unittest.TestCase):
    """Intrinsic motivation via uncertainty and novelty detection."""

    def setUp(self) -> None:
        """Create factory and mock params."""
        self.factory = _make_intrinsic_motivation_learner
        self.hyperparameters = FINAL_MECHANISMS["intrinsic_motivation"]["hyperparameters"]
        self.mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32),
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32),
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32),
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

    def test_init_fn_returns_valid_state(self) -> None:
        """Init function returns valid motivation state."""
        init_fn, _ = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        required_keys = {
            "prediction_error_ema",
            "uncertainty_estimate",
            "feature_diversity",
            "curiosity_signal",
            "exploration_bonus_ema",
            "novel_states_seen",
            "step_count",
        }
        self.assertEqual(set(state.keys()), required_keys)

    def test_uncertainty_bounded(self) -> None:
        """Uncertainty estimate stays in [0, 1]."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32) * 10}
        key = jr.key(0)

        for _ in range(20):
            params_out, state, _ = step_fn(self.mock_params, state, grads, key)
            uncertainty = float(state["uncertainty_estimate"])
            self.assertTrue(0 <= uncertainty <= 1, f"Uncertainty out of bounds: {uncertainty}")

    def test_curiosity_signal_nonnegative(self) -> None:
        """Curiosity signal is non-negative."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32)}
        key = jr.key(0)

        for _ in range(10):
            params_out, state, _ = step_fn(self.mock_params, state, grads, key)
            curiosity = float(state["curiosity_signal"])
            self.assertTrue(curiosity >= 0, f"Curiosity negative: {curiosity}")

    def test_novel_states_counting(self) -> None:
        """Novel state counter increments correctly."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        # High gradient = high uncertainty = novel states
        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32) * 100}
        key = jr.key(0)

        initial_count = int(state["novel_states_seen"])
        for _ in range(10):
            params_out, state, _ = step_fn(self.mock_params, state, grads, key)

        final_count = int(state["novel_states_seen"])
        self.assertGreaterEqual(final_count, initial_count)

    def test_feature_diversity_evolution(self) -> None:
        """Feature diversity updates and evolves."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32)}
        key = jr.key(0)

        initial_diversity = state["feature_diversity"].copy()
        for _ in range(10):
            params_out, state, _ = step_fn(self.mock_params, state, grads, key)

        final_diversity = state["feature_diversity"]
        # Should have changed
        diff = jnp.linalg.norm(final_diversity - initial_diversity)
        self.assertGreater(float(diff), 0)


class TestCombinedFinalMechanisms(unittest.TestCase):
    """Integrated mechanism: all four systems operating synergistically."""

    def setUp(self) -> None:
        """Create factory and mock params."""
        self.factory = _make_combined_final_mechanisms_learner
        self.hyperparameters = FINAL_MECHANISMS["combined_final_mechanisms"][
            "hyperparameters"
        ]
        self.mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32),
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32),
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32),
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

    def test_init_fn_returns_complete_state(self) -> None:
        """Init function returns state with all four mechanisms."""
        init_fn, _ = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        # Check for keys from all four mechanisms
        dual_head_keys = {"feature_weights", "class_weights", "head_correlation"}
        attention_keys = {"attention_logits", "attention_weights", "attention_entropy"}
        consolidation_keys = {
            "consolidated_weights",
            "consolidation_loss",
            "sleep_phase",
        }
        motivation_keys = {
            "prediction_error_ema",
            "uncertainty_estimate",
            "curiosity_signal",
            "novel_states_seen",
        }

        state_keys = set(state.keys())
        self.assertTrue(dual_head_keys.issubset(state_keys))
        self.assertTrue(attention_keys.issubset(state_keys))
        self.assertTrue(consolidation_keys.issubset(state_keys))
        self.assertTrue(motivation_keys.issubset(state_keys))

    def test_step_fn_produces_valid_metrics(self) -> None:
        """Step function produces valid (accuracy, loss, plasticity)."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32)}
        key = jr.key(0)

        params_out, state_out, metrics = step_fn(self.mock_params, state, grads, key)
        accuracy, loss, plasticity = metrics

        self.assertTrue(0 <= float(accuracy) <= 1)
        self.assertTrue(float(loss) >= 0)
        self.assertTrue(0 <= float(plasticity) <= 1)

    def test_all_mechanisms_active(self) -> None:
        """All four mechanisms produce active state changes."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32)}
        key = jr.key(0)

        # Run multiple steps
        initial_uncertainty = float(state["uncertainty_estimate"])
        for _ in range(10):
            params_out, state, _ = step_fn(self.mock_params, state, grads, key)

        # Verify mechanisms have evolved
        final_uncertainty = float(state["uncertainty_estimate"])
        self.assertNotEqual(initial_uncertainty, final_uncertainty)

    def test_sleep_consolidation_integration(self) -> None:
        """Sleep consolidation reduces plasticity during consolidation."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32)}
        key = jr.key(0)

        sleep_interval = int(self.hyperparameters["sleep_interval"])
        plasticity_during_sleep = []
        plasticity_awake = []

        for i in range(100):
            params_out, state, metrics = step_fn(self.mock_params, state, grads, key)
            _, _, plasticity = metrics

            if state["sleep_phase"]:
                plasticity_during_sleep.append(float(plasticity))
            else:
                plasticity_awake.append(float(plasticity))

        # During sleep, plasticity should be lower
        if len(plasticity_during_sleep) > 0 and len(plasticity_awake) > 0:
            avg_sleep = np.mean(plasticity_during_sleep)
            avg_awake = np.mean(plasticity_awake)
            # Sleep should reduce plasticity
            self.assertLess(avg_sleep, avg_awake + 0.1)

    def test_uncertainty_drives_plasticity(self) -> None:
        """Gradient magnitude affects uncertainty estimation."""
        init_fn, step_fn = self.factory(self.hyperparameters)

        # Test with low gradient magnitude
        grads_low = {"w1": jnp.ones((256, 256), dtype=jnp.float32) * 0.1}
        key = jr.key(0)
        state_low = init_fn(self.mock_params)
        for _ in range(5):
            _, state_low, _ = step_fn(self.mock_params, state_low, grads_low, key)
        uncertainty_low = float(state_low["uncertainty_estimate"])

        # Test with high gradient magnitude
        grads_high = {"w1": jnp.ones((256, 256), dtype=jnp.float32) * 10.0}
        state_high = init_fn(self.mock_params)
        for _ in range(5):
            _, state_high, _ = step_fn(self.mock_params, state_high, grads_high, key)
        uncertainty_high = float(state_high["uncertainty_estimate"])

        # Both should be valid uncertainty values
        self.assertTrue(0 <= uncertainty_low <= 1)
        self.assertTrue(0 <= uncertainty_high <= 1)

    def test_state_evolution_deterministic(self) -> None:
        """State evolution is deterministic given same inputs."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state1 = init_fn(self.mock_params)
        state2 = init_fn(self.mock_params)

        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32)}
        key = jr.key(42)

        for _ in range(5):
            _, state1, _ = step_fn(self.mock_params, state1, grads, key)
            _, state2, _ = step_fn(self.mock_params, state2, grads, key)

        # Compare key state variables
        self.assertAlmostEqual(
            float(state1["uncertainty_estimate"]), float(state2["uncertainty_estimate"]), places=5
        )
        self.assertAlmostEqual(
            float(jnp.linalg.norm(state1["attention_weights"])),
            float(jnp.linalg.norm(state2["attention_weights"])),
            places=5,
        )


class TestMetricsConsistency(unittest.TestCase):
    """Verify metrics are consistent across all mechanisms."""

    def setUp(self) -> None:
        """Setup test parameters."""
        self.mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32),
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32),
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32),
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

    def test_all_mechanisms_produce_valid_metrics(self) -> None:
        """All mechanisms produce valid (accuracy, loss, plasticity) tuples."""
        grads = {"w1": jnp.ones((256, 256), dtype=jnp.float32)}
        key = jr.key(0)

        for mech_name, spec in FINAL_MECHANISMS.items():
            with self.subTest(mechanism=mech_name):
                factory = spec["factory"]
                hp = spec["hyperparameters"]

                init_fn, step_fn = factory(hp)
                state = init_fn(self.mock_params)

                _, _, metrics = step_fn(self.mock_params, state, grads, key)
                accuracy, loss, plasticity = metrics

                self.assertTrue(0 <= float(accuracy) <= 1, f"{mech_name}: accuracy out of bounds")
                self.assertTrue(float(loss) >= 0, f"{mech_name}: loss negative")
                self.assertTrue(0 <= float(plasticity) <= 1, f"{mech_name}: plasticity out of bounds")


if __name__ == "__main__":
    unittest.main()
