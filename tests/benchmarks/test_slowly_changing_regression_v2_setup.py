"""Smoke tests for SCR v2 infrastructure (unit-level, no benchmark execution).

These tests verify that the arm registry, learner factories, and setup functions
are self-consistent and ready for shard execution. They do NOT run full protocols
or consume preregistered seeds — they are cheap unit-level checks.

Run with: pytest tests/benchmarks/test_slowly_changing_regression_v2_setup.py -v
"""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.slowly_changing_regression import (
    SlowlyChangingRegressionConfig,
)
from alberta_framework.benchmarks.slowly_changing_regression_v2_arms import (
    ARM_REGISTRY,
    SCR_V2_ALL_ARMS,
    SCR_V2_ALBERTA_ARMS,
    SCR_V2_BASELINE_ARMS,
    get_arm_description,
    get_arm_hyperparameters,
)
from alberta_framework.benchmarks.slowly_changing_regression_v2_learners import (
    get_learner_factory,
)
from alberta_framework.benchmarks.slowly_changing_regression_v2_setup import (
    get_all_registered_arms,
    setup_arm_learner,
    validate_arm_name,
    validate_preregistration_config,
)


class TestArmRegistry:
    """Verify arm registry structure and completeness."""

    def test_registry_has_all_baseline_arms(self) -> None:
        """All preregistered baseline arms are in the registry."""
        for arm_name in SCR_V2_BASELINE_ARMS:
            assert arm_name in ARM_REGISTRY, f"baseline arm {arm_name!r} not in registry"

    def test_registry_has_all_alberta_arms(self) -> None:
        """All preregistered Alberta arms are in the registry."""
        for arm_name in SCR_V2_ALBERTA_ARMS:
            assert arm_name in ARM_REGISTRY, f"Alberta arm {arm_name!r} not in registry"

    def test_registry_completeness(self) -> None:
        """Registry contains exactly the preregistered set."""
        registry_names = set(ARM_REGISTRY.keys())
        assert registry_names == SCR_V2_ALL_ARMS

    def test_arm_specifications_have_required_fields(self) -> None:
        """Each arm spec has name, role, hyperparameters, description, reference."""
        for name, spec in ARM_REGISTRY.items():
            assert spec.name == name
            assert spec.role in (
                "baseline_publication",
                "baseline_control",
                "alberta_domain_transfer",
                "alberta_mechanism_extension",
            )
            assert bool(spec.hyperparameters), f"arm {name} has empty hyperparameters"
            assert bool(spec.description), f"arm {name} has empty description"
            assert bool(spec.reference), f"arm {name} has empty reference"

    def test_arm_roles_distribution(self) -> None:
        """Verify expected role distribution."""
        publication = sum(1 for s in ARM_REGISTRY.values() if s.role == "baseline_publication")
        control = sum(1 for s in ARM_REGISTRY.values() if s.role == "baseline_control")
        domain_transfer = sum(1 for s in ARM_REGISTRY.values() if s.role == "alberta_domain_transfer")
        mechanism = sum(1 for s in ARM_REGISTRY.values() if s.role == "alberta_mechanism_extension")

        assert publication == 1, "exactly one publication baseline"
        assert control == 2, "exactly two control baselines"
        assert domain_transfer == 1, "exactly one domain-transfer arm"
        assert mechanism == 2, "exactly two mechanism-extension arms"


class TestHyperparameterRetrieval:
    """Verify hyperparameter access and validation."""

    def test_get_hyperparameters_for_all_arms(self) -> None:
        """get_arm_hyperparameters works for every registered arm."""
        for arm_name in SCR_V2_ALL_ARMS:
            hp = get_arm_hyperparameters(arm_name)
            assert isinstance(hp, dict)
            assert bool(hp), f"arm {arm_name} returned empty hyperparameters"

    def test_hyperparameters_are_unfrozen_dicts(self) -> None:
        """Returned hyperparameters are mutable dicts (not frozen)."""
        for arm_name in SCR_V2_ALL_ARMS:
            hp = get_arm_hyperparameters(arm_name)
            # Should not raise; frozen types would raise TypeError
            hp["_test_key"] = 1.0
            del hp["_test_key"]

    def test_baseline_hyperparameters_have_core_fields(self) -> None:
        """All arms have the required core learner hyperparameters."""
        required_fields = {
            "hidden_units",
            "step_size",
            "cbp_replacement_rate",
            "cbp_maturity_threshold",
            "cbp_decay_rate",
            "upgd_utility_decay",
            "upgd_sigma",
            "upgd_beta",
        }
        for arm_name in SCR_V2_ALL_ARMS:
            hp = get_arm_hyperparameters(arm_name)
            missing = required_fields - set(hp.keys())
            assert not missing, f"arm {arm_name} missing fields: {missing}"

    def test_get_hyperparameters_raises_on_invalid_arm(self) -> None:
        """Raises KeyError for unregistered arms."""
        with pytest.raises(KeyError):
            get_arm_hyperparameters("nonexistent_arm")


class TestDescriptionAccess:
    """Verify arm description retrieval."""

    def test_get_description_for_all_arms(self) -> None:
        """get_arm_description works for every registered arm."""
        for arm_name in SCR_V2_ALL_ARMS:
            desc = get_arm_description(arm_name)
            assert isinstance(desc, str)
            assert len(desc) > 20, f"description for {arm_name} suspiciously short"

    def test_get_description_raises_on_invalid_arm(self) -> None:
        """Raises KeyError for unregistered arms."""
        with pytest.raises(KeyError):
            get_arm_description("nonexistent_arm")


class TestLearnerFactories:
    """Verify learner factory instantiation."""

    def test_learner_factory_exists_for_all_arms(self) -> None:
        """get_learner_factory returns a callable for every arm."""
        for arm_name in SCR_V2_ALL_ARMS:
            factory = get_learner_factory(arm_name)
            assert callable(factory), f"factory for {arm_name} is not callable"

    def test_learner_factory_raises_on_invalid_arm(self) -> None:
        """Raises KeyError for unregistered arms."""
        with pytest.raises(KeyError):
            get_learner_factory("nonexistent_arm")

    def test_learner_factory_returns_init_and_step(self) -> None:
        """Factories return (init_fn, step_fn) pairs."""
        for arm_name in SCR_V2_ALL_ARMS:
            factory = get_learner_factory(arm_name)
            hp = get_arm_hyperparameters(arm_name)
            init_fn, step_fn = factory(hp)
            assert callable(init_fn), f"init_fn for {arm_name} is not callable"
            assert callable(step_fn), f"step_fn for {arm_name} is not callable"


class TestValidation:
    """Verify validation functions."""

    def test_validate_arm_name_accepts_valid_arms(self) -> None:
        """validate_arm_name accepts all registered arms."""
        for arm_name in SCR_V2_ALL_ARMS:
            validate_arm_name(arm_name)  # Should not raise

    def test_validate_arm_name_rejects_invalid_arms(self) -> None:
        """validate_arm_name raises ValueError for unknown arms."""
        with pytest.raises(ValueError, match="not registered"):
            validate_arm_name("nonexistent_arm")

    def test_validate_preregistration_config_accepts_valid(self) -> None:
        """validate_preregistration_config accepts valid specifications."""
        config = SlowlyChangingRegressionConfig(
            num_bits=20,
            num_flipping_bits=15,
            flip_period=10_000,
            target_hidden_units=100,
            ltu_beta=0.7,
            num_examples=60_000,  # Phase 2 screening size
        )
        validate_preregistration_config(
            config,
            list(SCR_V2_ALL_ARMS),
            [100, 101, 102],
        )  # Should not raise

    def test_validate_preregistration_config_rejects_wrong_seeds(self) -> None:
        """validate_preregistration_config rejects out-of-range seeds."""
        config = SlowlyChangingRegressionConfig()
        with pytest.raises(ValueError, match="seed IDs must be in"):
            validate_preregistration_config(
                config,
                list(SCR_V2_ALL_ARMS),
                [0, 1, 2],  # Wrong: preregistration requires [100, 102]
            )

    def test_validate_preregistration_config_rejects_wrong_task_params(self) -> None:
        """validate_preregistration_config rejects mismatched task config."""
        config = SlowlyChangingRegressionConfig(
            num_bits=10,  # Wrong: should be 20
            num_flipping_bits=15,
            flip_period=10_000,
            target_hidden_units=100,
        )
        with pytest.raises(ValueError, match="num_bits"):
            validate_preregistration_config(
                config,
                list(SCR_V2_ALL_ARMS),
                [100, 101, 102],
            )


class TestSetupArmLearner:
    """Verify end-to-end arm setup."""

    def test_setup_arm_learner_returns_learner_and_metadata(self) -> None:
        """setup_arm_learner returns (init_fn, step_fn, metadata)."""
        config = SlowlyChangingRegressionConfig()
        for arm_name in SCR_V2_ALL_ARMS:
            init_fn, step_fn, metadata = setup_arm_learner(arm_name, config)
            assert callable(init_fn)
            assert callable(step_fn)
            assert isinstance(metadata, dict)

    def test_metadata_has_required_fields(self) -> None:
        """Returned metadata includes arm details."""
        config = SlowlyChangingRegressionConfig()
        for arm_name in SCR_V2_ALL_ARMS:
            _, _, metadata = setup_arm_learner(arm_name, config)
            assert metadata["arm_name"] == arm_name
            assert "arm_role" in metadata
            assert "arm_description" in metadata
            assert "arm_reference" in metadata
            assert "hyperparameters" in metadata

    def test_setup_arm_learner_raises_on_invalid_arm(self) -> None:
        """setup_arm_learner raises ValueError for unknown arms."""
        config = SlowlyChangingRegressionConfig()
        with pytest.raises(ValueError, match="not registered"):
            setup_arm_learner("nonexistent_arm", config)


class TestGetAllRegisteredArms:
    """Verify registry summary."""

    def test_get_all_registered_arms_returns_dict(self) -> None:
        """get_all_registered_arms returns a summary dict."""
        summary = get_all_registered_arms()
        assert isinstance(summary, dict)
        assert len(summary) == len(SCR_V2_ALL_ARMS)

    def test_summary_has_all_arms(self) -> None:
        """Summary includes all registered arms."""
        summary = get_all_registered_arms()
        assert set(summary.keys()) == SCR_V2_ALL_ARMS

    def test_summary_entries_have_required_fields(self) -> None:
        """Each summary entry has role, description, reference."""
        summary = get_all_registered_arms()
        for arm_name, entry in summary.items():
            assert "role" in entry
            assert "description" in entry
            assert "reference" in entry
            assert isinstance(entry["role"], str)
            assert isinstance(entry["description"], str)
            assert isinstance(entry["reference"], str)
