"""
Unit tests for micro_continual_improvements.py — preregistered arm factories.

Test coverage:
- Factory initialization and state creation
- Hyperparameter validation
- Signature compliance (LearnerInitFn, ScreeningStepFn)
- Integration with micro_continual framework
- Basic forward/backward sanity checks
- Preregistration metadata validation
"""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from micro_continual_improvements import (
    PREREGISTERED_ARMS,
    _make_actor_critic_micro_learner,
    _make_alignment_first_learner,
    _make_dual_speed_rfs_rls_learner,
    _make_naive_bayes_extended_learner,
    _make_rls_head_resid_learner,
)


class TestPreregisteredArmsMetadata(unittest.TestCase):
    """Validate preregistration metadata structure and completeness."""

    def test_registry_completeness(self) -> None:
        """All arms in PREREGISTERED_ARMS have required fields."""
        required_fields = {
            "name",
            "factory",
            "mechanism",
            "hyperparameters",
            "description",
        }
        for arm_name, spec in PREREGISTERED_ARMS.items():
            with self.subTest(arm=arm_name):
                self.assertIsInstance(spec, dict)
                self.assertEqual(set(spec.keys()), required_fields)
                self.assertIsInstance(spec["name"], str)
                self.assertIsInstance(spec["mechanism"], str)
                self.assertIsInstance(spec["hyperparameters"], dict)
                self.assertIsInstance(spec["description"], str)
                self.assertTrue(len(spec["description"]) > 0)

    def test_factory_callable(self) -> None:
        """All factories are callable."""
        for arm_name, spec in PREREGISTERED_ARMS.items():
            with self.subTest(arm=arm_name):
                self.assertTrue(callable(spec["factory"]))

    def test_hyperparameters_types(self) -> None:
        """All hyperparameters are numeric or bool."""
        for arm_name, spec in PREREGISTERED_ARMS.items():
            with self.subTest(arm=arm_name):
                for key, val in spec["hyperparameters"].items():
                    self.assertIsInstance(
                        val, (int, float, bool), msg=f"{arm_name}.{key} = {val!r}"
                    )

    def test_preregistration_reference_in_description(self) -> None:
        """Each arm description cites its preregistration source."""
        valid_sources = {
            "CONTRIBUTION_PREREGISTRATION",
            "NEW_DIRECTIONS",
            "FORAGER_OPEN_BASELINES_PREREGISTRATION",
            "SUITE",
        }
        for arm_name, spec in PREREGISTERED_ARMS.items():
            with self.subTest(arm=arm_name):
                self.assertTrue(
                    any(src in spec["description"] for src in valid_sources),
                    msg=f"{arm_name} has no preregistration source cited",
                )


class TestRLSHeadResidLearner(unittest.TestCase):
    """RLS readout + residual head learning."""

    def setUp(self) -> None:
        """Create factory and mock params."""
        self.factory = _make_rls_head_resid_learner
        self.hyperparameters = PREREGISTERED_ARMS["rls_head_resid"]["hyperparameters"]
        self.mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32),
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32),
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32),
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

    def test_init_returns_callable(self) -> None:
        """Factory returns (init_fn, step_fn) tuple."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        self.assertTrue(callable(init_fn))
        self.assertTrue(callable(step_fn))

    def test_init_fn_creates_state(self) -> None:
        """init_fn creates state dict with required keys."""
        init_fn, _ = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        required_keys = {"norm_mean", "norm_var", "rls_P", "rls_w", "n_shifted"}
        self.assertEqual(set(state.keys()), required_keys)

        # Validate shapes
        self.assertEqual(state["norm_mean"].shape, (256,))
        self.assertEqual(state["norm_var"].shape, (256,))
        self.assertEqual(state["rls_P"].shape, (150, 150))
        self.assertEqual(state["rls_w"].shape, (150, 10))
        self.assertEqual(state["n_shifted"].shape, ())

    def test_step_fn_signature(self) -> None:
        """step_fn has correct signature: (params, state, grads, key) -> (params, state, metrics)."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)
        grads = {k: jnp.zeros_like(v) for k, v in self.mock_params.items()}
        key = jr.key(0)

        new_params, new_state, metrics = step_fn(self.mock_params, state, grads, key)

        # Validate return types
        self.assertIsInstance(new_params, dict)
        self.assertIsInstance(new_state, dict)
        self.assertIsInstance(metrics, tuple)
        self.assertEqual(len(metrics), 3)  # (accuracy, loss, plasticity)

    def test_state_preservation(self) -> None:
        """State structure preserved across steps."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)
        grads = {k: jnp.zeros_like(v) for k, v in self.mock_params.items()}
        key = jr.key(0)

        _, new_state, _ = step_fn(self.mock_params, state, grads, key)

        # State keys unchanged
        self.assertEqual(set(new_state.keys()), set(state.keys()))


class TestAlignmentFirstLearner(unittest.TestCase):
    """Permutation alignment detector."""

    def setUp(self) -> None:
        """Create factory and mock params."""
        self.factory = _make_alignment_first_learner
        self.hyperparameters = PREREGISTERED_ARMS["alignment_first"]["hyperparameters"]
        self.mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32),
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32),
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32),
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

    def test_init_creates_alignment_state(self) -> None:
        """init_fn creates state with alignment tracking."""
        init_fn, _ = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        required_keys = {
            "norm_mean",
            "norm_var",
            "alignment_buffer",
            "last_perm",
            "step_count",
        }
        self.assertEqual(set(state.keys()), required_keys)
        self.assertEqual(state["step_count"], jnp.array(0, dtype=jnp.int32))

    def test_step_increments_count(self) -> None:
        """step_fn increments internal step counter."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)
        grads = {k: jnp.zeros_like(v) for k, v in self.mock_params.items()}
        key = jr.key(0)

        _, new_state, _ = step_fn(self.mock_params, state, grads, key)

        self.assertEqual(new_state["step_count"], state["step_count"] + 1)


class TestNaiveBayesExtendedLearner(unittest.TestCase):
    """Context-conditioned streaming generative classifier."""

    def setUp(self) -> None:
        """Create factory and mock params."""
        self.factory = _make_naive_bayes_extended_learner
        self.hyperparameters = PREREGISTERED_ARMS["naive_bayes_extended"]["hyperparameters"]
        self.mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32),
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32),
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32),
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

    def test_init_creates_generative_state(self) -> None:
        """init_fn creates class-conditional statistics."""
        init_fn, _ = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        required_keys = {
            "class_means",
            "class_vars",
            "class_counts",
            "context_cache",
            "current_context",
        }
        self.assertEqual(set(state.keys()), required_keys)

        self.assertEqual(state["class_means"].shape, (10, 256))
        self.assertEqual(state["class_vars"].shape, (10, 256))
        self.assertEqual(state["class_counts"].shape, (10,))

    def test_plasticity_is_max(self) -> None:
        """No weight decay = maximum plasticity signal."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)
        grads = {k: jnp.zeros_like(v) for k, v in self.mock_params.items()}
        key = jr.key(0)

        _, _, metrics = step_fn(self.mock_params, state, grads, key)
        accuracy, loss, plasticity = metrics

        # Generative model (no gradients) = max plasticity
        self.assertAlmostEqual(float(plasticity), 1.0)


class TestDualSpeedRFSRLSLearner(unittest.TestCase):
    """Frozen random features + per-regime RLS cache."""

    def setUp(self) -> None:
        """Create factory and mock params."""
        self.factory = _make_dual_speed_rfs_rls_learner
        self.hyperparameters = PREREGISTERED_ARMS["dual_speed_rfs_rls"]["hyperparameters"]
        self.mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32),
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32),
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32),
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

    def test_init_creates_frozen_bank(self) -> None:
        """init_fn creates fixed random feature matrix."""
        init_fn, _ = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        required_keys = {
            "rfs_matrix",
            "rls_P",
            "rls_w",
            "context_fingerprint",
            "context_cache",
            "current_context_id",
        }
        self.assertEqual(set(state.keys()), required_keys)

        rfs_dim = self.hyperparameters["rfs_dim"]
        self.assertEqual(state["rfs_matrix"].shape, (rfs_dim, 256))
        self.assertEqual(state["rls_P"].shape, (rfs_dim, rfs_dim))

    def test_rls_p_initialized_scaled(self) -> None:
        """RLS P-matrix scaled by lambda."""
        init_fn, _ = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        rls_lambda = self.hyperparameters["rls_lambda"]
        expected_scale = 1.0 / rls_lambda
        self.assertAlmostEqual(float(state["rls_P"][0, 0]), expected_scale, places=5)

    def test_plasticity_reduced(self) -> None:
        """Frozen body = lower plasticity signal."""
        init_fn, step_fn = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)
        grads = {k: jnp.zeros_like(v) for k, v in self.mock_params.items()}
        key = jr.key(0)

        _, _, metrics = step_fn(self.mock_params, state, grads, key)
        _, _, plasticity = metrics

        # Fixed features = reduced plasticity
        self.assertLess(float(plasticity), 1.0)


class TestActorCriticMicroLearner(unittest.TestCase):
    """On-policy actor-critic adapted to supervised streams."""

    def setUp(self) -> None:
        """Create factory and mock params."""
        self.factory = _make_actor_critic_micro_learner
        self.hyperparameters = PREREGISTERED_ARMS["actor_critic_micro"]["hyperparameters"]
        self.mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32),
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32),
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32),
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

    def test_init_creates_actor_critic_state(self) -> None:
        """init_fn creates both actor and value-network params."""
        init_fn, _ = self.factory(self.hyperparameters)
        state = init_fn(self.mock_params)

        required_keys = {"norm_mean", "norm_var", "value_params"}
        self.assertEqual(set(state.keys()), required_keys)

        # Critic network has its own parameters
        value_params = state["value_params"]
        self.assertIn("v_w1", value_params)
        self.assertIn("v_b1", value_params)
        self.assertIn("v_out", value_params)


class TestFactorySignatureCompliance(unittest.TestCase):
    """All factories conform to MicroArmFactory signature."""

    def test_all_factories_return_init_step_pair(self) -> None:
        """Every factory returns (init_fn, step_fn)."""
        mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32),
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32),
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32),
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

        for arm_name, spec in PREREGISTERED_ARMS.items():
            with self.subTest(arm=arm_name):
                factory = spec["factory"]
                hp = spec["hyperparameters"]

                init_fn, step_fn = factory(hp)

                self.assertTrue(callable(init_fn))
                self.assertTrue(callable(step_fn))

                # init_fn(params) -> state
                state = init_fn(mock_params)
                self.assertIsInstance(state, dict)

                # step_fn(params, state, grads, key) -> (params, state, metrics)
                grads = {k: jnp.zeros_like(v) for k, v in mock_params.items()}
                key = jr.key(0)
                new_params, new_state, metrics = step_fn(mock_params, state, grads, key)

                self.assertIsInstance(new_params, dict)
                self.assertIsInstance(new_state, dict)
                self.assertIsInstance(metrics, tuple)
                self.assertEqual(len(metrics), 3)  # (accuracy, loss, plasticity)

    def test_metrics_are_arrays(self) -> None:
        """Metrics (accuracy, loss, plasticity) are JAX arrays."""
        mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32),
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32),
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32),
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

        for arm_name, spec in PREREGISTERED_ARMS.items():
            with self.subTest(arm=arm_name):
                factory = spec["factory"]
                hp = spec["hyperparameters"]

                init_fn, step_fn = factory(hp)
                state = init_fn(mock_params)
                grads = {k: jnp.zeros_like(v) for k, v in mock_params.items()}
                key = jr.key(0)

                _, _, metrics = step_fn(mock_params, state, grads, key)
                accuracy, loss, plasticity = metrics

                self.assertEqual(accuracy.dtype, jnp.float32)
                self.assertEqual(loss.dtype, jnp.float32)
                self.assertEqual(plasticity.dtype, jnp.float32)

                # Metrics are in reasonable ranges
                self.assertGreaterEqual(float(accuracy), 0.0)
                self.assertLessEqual(float(accuracy), 1.0)
                self.assertGreaterEqual(float(plasticity), 0.0)
                self.assertLessEqual(float(plasticity), 1.0)


class TestIntegrationWithMicroContinual(unittest.TestCase):
    """Integration checks with micro_continual framework."""

    def test_preregistered_arms_can_be_looked_up(self) -> None:
        """Arms in registry can be accessed by name."""
        for arm_name in PREREGISTERED_ARMS:
            spec = PREREGISTERED_ARMS[arm_name]
            self.assertEqual(spec["name"], arm_name)

    def test_hyperparameters_are_mappings(self) -> None:
        """Hyperparameter dicts are Mapping-compatible."""
        for arm_name, spec in PREREGISTERED_ARMS.items():
            hp = spec["hyperparameters"]
            # Should be passable to factories expecting Mapping[str, float]
            self.assertTrue(hasattr(hp, "get"))
            self.assertTrue(hasattr(hp, "__getitem__"))

    def test_factory_handles_missing_hyperparameters(self) -> None:
        """Factories use .get() with defaults, not direct access."""
        mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32),
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32),
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32),
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

        # Empty hyperparameters dict should not crash
        empty_hp: dict[str, float] = {}

        for arm_name, spec in PREREGISTERED_ARMS.items():
            with self.subTest(arm=arm_name):
                factory = spec["factory"]
                init_fn, step_fn = factory(empty_hp)

                # Should succeed with defaults
                state = init_fn(mock_params)
                self.assertIsInstance(state, dict)


class TestPreregistrationConsistency(unittest.TestCase):
    """Validate consistency between arms and preregistration docs."""

    def test_rls_head_resid_has_preregistration(self) -> None:
        """RLS arm references CONTRIBUTION_PREREGISTRATION."""
        arm = PREREGISTERED_ARMS["rls_head_resid"]
        self.assertIn("CONTRIBUTION_PREREGISTRATION", arm["description"])
        self.assertIn("0.87114", arm["description"])

    def test_alignment_first_has_preregistration(self) -> None:
        """Alignment arm references NEW_DIRECTIONS V2."""
        arm = PREREGISTERED_ARMS["alignment_first"]
        self.assertIn("NEW_DIRECTIONS", arm["description"])
        self.assertIn("V2", arm["description"])

    def test_naive_bayes_extended_has_placement(self) -> None:
        """NB arm includes baseline placement (SUITE)."""
        arm = PREREGISTERED_ARMS["naive_bayes_extended"]
        self.assertIn("SUITE", arm["description"])
        self.assertIn("0.7851", arm["description"])

    def test_dual_speed_has_v4_reference(self) -> None:
        """Dual-speed arm references NEW_DIRECTIONS V4."""
        arm = PREREGISTERED_ARMS["dual_speed_rfs_rls"]
        self.assertIn("NEW_DIRECTIONS", arm["description"])
        self.assertIn("V4", arm["description"])

    def test_actor_critic_has_forager_reference(self) -> None:
        """AC arm references Forager preregistration."""
        arm = PREREGISTERED_ARMS["actor_critic_micro"]
        self.assertIn("FORAGER_OPEN_BASELINES_PREREGISTRATION", arm["description"])


if __name__ == "__main__":
    unittest.main()
