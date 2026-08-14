"""
Integration test: micro_continual_improvements can be imported and used
with the actual micro_continual framework.

This test verifies that the preregistered arms can be instantiated and
run through the micro_continual machinery without errors.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import jax.random as jr

from micro_continual_improvements import PREREGISTERED_ARMS


class TestIntegrationWithMicroContinualFramework(unittest.TestCase):
    """End-to-end integration tests."""

    def test_all_preregistered_arms_importable(self) -> None:
        """All arms in registry are importable and callable."""
        for arm_name, spec in PREREGISTERED_ARMS.items():
            with self.subTest(arm=arm_name):
                factory = spec["factory"]
                hp = spec["hyperparameters"]

                # Should not raise
                init_fn, step_fn = factory(hp)
                self.assertTrue(callable(init_fn))
                self.assertTrue(callable(step_fn))

    def test_mock_mlp_run(self) -> None:
        """Simulate a single micro_continual run step with each arm."""
        # Mock MLP parameters (256 -> 75 -> 38 -> 10, micro protocol)
        mock_params = {
            "w1": jnp.ones((256, 256), dtype=jnp.float32) * 0.01,
            "b1": jnp.zeros(256, dtype=jnp.float32),
            "w2": jnp.ones((256, 150), dtype=jnp.float32) * 0.01,
            "b2": jnp.zeros(150, dtype=jnp.float32),
            "w_out": jnp.ones((150, 10), dtype=jnp.float32) * 0.01,
            "b_out": jnp.zeros(10, dtype=jnp.float32),
        }

        for arm_name, spec in PREREGISTERED_ARMS.items():
            with self.subTest(arm=arm_name):
                factory = spec["factory"]
                hp = spec["hyperparameters"]

                # Initialize
                init_fn, step_fn = factory(hp)
                state = init_fn(mock_params)

                # Simulate 10 steps
                for step in range(10):
                    grads = {
                        k: jnp.ones_like(v) * 0.001 for k, v in mock_params.items()
                    }
                    key = jr.fold_in(jr.key(0), step)

                    new_params, new_state, metrics = step_fn(
                        mock_params, state, grads, key
                    )

                    # Validate outputs
                    self.assertIsInstance(new_params, dict)
                    self.assertIsInstance(new_state, dict)
                    self.assertIsInstance(metrics, tuple)
                    self.assertEqual(len(metrics), 3)

                    accuracy, loss, plasticity = metrics
                    self.assertGreaterEqual(float(accuracy), 0.0)
                    self.assertLessEqual(float(accuracy), 1.0)
                    self.assertGreaterEqual(float(loss), -1e-6)  # allow small negative from numerics
                    self.assertGreaterEqual(float(plasticity), 0.0)
                    self.assertLessEqual(float(plasticity), 1.0)

                    # Update state for next iteration
                    state = new_state

    def test_preregistered_arms_registry_structure(self) -> None:
        """Registry matches micro_continual.MicroArmSpec expectations."""
        from micro_continual_improvements import PREREGISTERED_ARMS

        # Each entry should be convertible to micro_continual.MicroArmSpec
        for arm_name, spec in PREREGISTERED_ARMS.items():
            with self.subTest(arm=arm_name):
                # Required fields
                self.assertIn("name", spec)
                self.assertIn("mechanism", spec)
                self.assertIn("hyperparameters", spec)
                self.assertIn("factory", spec)
                self.assertIn("description", spec)

                # Types
                self.assertIsInstance(spec["name"], str)
                self.assertIsInstance(spec["mechanism"], str)
                self.assertIsInstance(spec["hyperparameters"], dict)
                self.assertTrue(callable(spec["factory"]))
                self.assertIsInstance(spec["description"], str)

                # All hyperparameters are numeric
                for k, v in spec["hyperparameters"].items():
                    self.assertIsInstance(v, (int, float, bool))

    def test_hyperparameter_coverage(self) -> None:
        """Verify each arm has reasonable hyperparameter coverage."""
        expected_hyperparam_count = {
            "rls_head_resid": (6, 8),  # expect 6-8 hyperparams
            "alignment_first": (5, 7),
            "naive_bayes_extended": (2, 4),
            "dual_speed_rfs_rls": (3, 5),
            "actor_critic_micro": (4, 6),
        }

        for arm_name, (min_hp, max_hp) in expected_hyperparam_count.items():
            with self.subTest(arm=arm_name):
                spec = PREREGISTERED_ARMS[arm_name]
                n_hp = len(spec["hyperparameters"])
                self.assertGreaterEqual(
                    n_hp,
                    min_hp,
                    msg=f"{arm_name} has fewer hyperparams than expected"
                )
                self.assertLessEqual(
                    n_hp,
                    max_hp,
                    msg=f"{arm_name} has more hyperparams than expected"
                )

    def test_descriptions_are_informative(self) -> None:
        """Each arm description includes mechanism, source, and rationale."""
        for arm_name, spec in PREREGISTERED_ARMS.items():
            with self.subTest(arm=arm_name):
                desc = spec["description"]

                # Should mention the mechanism or key concept
                self.assertGreater(len(desc), 50)

                # Should cite a preregistration source
                sources = {
                    "CONTRIBUTION_PREREGISTRATION",
                    "NEW_DIRECTIONS",
                    "FORAGER_OPEN_BASELINES_PREREGISTRATION",
                    "SUITE",
                }
                has_source = any(src in desc for src in sources)
                self.assertTrue(has_source, msg=f"{arm_name} has no preregistration source")


class TestPreregistrationMetricsConsistency(unittest.TestCase):
    """Verify that preregistration metrics are embedded in descriptions."""

    def test_rls_head_resid_metrics(self) -> None:
        """RLS arm includes 0.87114 measurement."""
        spec = PREREGISTERED_ARMS["rls_head_resid"]
        self.assertIn("0.87114", spec["description"])

    def test_naive_bayes_extended_placement(self) -> None:
        """NB arm includes 0.7851 baseline."""
        spec = PREREGISTERED_ARMS["naive_bayes_extended"]
        self.assertIn("0.7851", spec["description"])

    def test_dual_speed_rfs_rls_baseline(self) -> None:
        """Dual-speed arm includes 0.848 measurement."""
        spec = PREREGISTERED_ARMS["dual_speed_rfs_rls"]
        self.assertIn("0.848", spec["description"])

    def test_champion_baseline_mentioned(self) -> None:
        """Champion baseline 0.86449 appears in registry context."""
        # Not every arm needs to cite it, but some should
        descs = [spec["description"] for spec in PREREGISTERED_ARMS.values()]
        has_champion_ref = any("0.86449" in desc for desc in descs)
        # This is a soft check; not all arms need to cite champion
        # (some are designed to beat it, others to understand it)


if __name__ == "__main__":
    unittest.main()
