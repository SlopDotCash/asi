"""Setup and initialization for SCR v2 shard execution.

This module provides high-level setup functions used by shard executors and
merge pipelines to instantiate arms, validate configurations, and prepare
for deterministic execution.

Reference: SLOWLY_CHANGING_REGRESSION_V2_PREREGISTRATION.md
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from alberta_framework.benchmarks.slowly_changing_regression import (
    SCRLearnerParams,
    SlowlyChangingRegressionConfig,
)
from alberta_framework.benchmarks.slowly_changing_regression_v2_arms import (
    ARM_REGISTRY,
    get_arm_hyperparameters,
)
from alberta_framework.benchmarks.slowly_changing_regression_v2_learners import (
    LearnerInitFn,
    LearnerStepFn,
    get_learner_factory,
)

__all__ = [
    "validate_arm_name",
    "validate_preregistration_config",
    "setup_arm_learner",
    "get_all_registered_arms",
]


def validate_arm_name(arm_name: str) -> None:
    """Validate that an arm name is in the registry.

    Args:
        arm_name: The arm identifier to validate.

    Raises:
        ValueError: if the arm is not registered.
    """
    if arm_name not in ARM_REGISTRY:
        valid_arms = sorted(ARM_REGISTRY.keys())
        raise ValueError(
            f"arm {arm_name!r} is not registered. "
            f"Valid preregistered arms: {valid_arms}"
        )


def validate_preregistration_config(
    config: SlowlyChangingRegressionConfig,
    arm_names: list[str],
    seed_ids: list[int],
) -> None:
    """Validate that a configuration matches the preregistration spec.

    Checks:
    - All arm names are registered
    - Seed IDs are in the preregistration range [100, 102]
    - Task configuration uses expected protocol shape

    Args:
        config: The task configuration to validate.
        arm_names: List of arm identifiers to use.
        seed_ids: List of seed values for the run.

    Raises:
        ValueError: if any check fails.
    """
    # Validate arms
    for arm in arm_names:
        validate_arm_name(arm)

    # Validate seeds (preregistration specifies 100-102)
    if not arm_names:
        raise ValueError("at least one arm must be specified")
    if not seed_ids:
        raise ValueError("at least one seed must be specified")

    valid_seed_range = set(range(100, 103))
    invalid_seeds = set(seed_ids) - valid_seed_range
    if invalid_seeds:
        raise ValueError(
            f"seed IDs must be in [100, 102] per preregistration; "
            f"got {sorted(invalid_seeds)}"
        )

    # Validate task configuration (expected protocol shape)
    if config.num_bits != 20:
        raise ValueError(
            f"preregistration specifies num_bits=20; got {config.num_bits}"
        )
    if config.num_flipping_bits != 15:
        raise ValueError(
            f"preregistration specifies num_flipping_bits=15; "
            f"got {config.num_flipping_bits}"
        )
    if config.flip_period != 10_000:
        raise ValueError(
            f"preregistration specifies flip_period=10000; "
            f"got {config.flip_period}"
        )
    if config.target_hidden_units != 100:
        raise ValueError(
            f"preregistration specifies target_hidden_units=100; "
            f"got {config.target_hidden_units}"
        )


def setup_arm_learner(
    arm_name: str,
    task_config: SlowlyChangingRegressionConfig,
) -> tuple[LearnerInitFn, LearnerStepFn, dict[str, Any]]:
    """Instantiate a learner for an arm with its preregistered hyperparameters.

    Retrieves hyperparameters from the registry, instantiates the learner
    via the factory, and returns the init/step functions ready for the
    benchmark loop.

    Args:
        arm_name: The arm identifier (key in ARM_REGISTRY).
        task_config: The task configuration (for context; unused in current factories).

    Returns:
        A tuple of (init_fn, step_fn, metadata_dict) where:
        - init_fn: Callable to initialize learner state
        - step_fn: Callable to perform one learner update
        - metadata_dict: Dict with arm name, role, description, and hyperparameters

    Raises:
        ValueError: if the arm is not registered.
        KeyError: if the learner factory is not found.
    """
    validate_arm_name(arm_name)

    arm_spec = ARM_REGISTRY[arm_name]
    hp = get_arm_hyperparameters(arm_name)
    factory = get_learner_factory(arm_name)

    init_fn, step_fn = factory(hp)

    metadata = {
        "arm_name": arm_name,
        "arm_role": arm_spec.role,
        "arm_description": arm_spec.description,
        "arm_reference": arm_spec.reference,
        "hyperparameters": dict(hp),
    }

    return init_fn, step_fn, metadata


def get_all_registered_arms() -> dict[str, dict[str, str]]:
    """Return a summary of all registered arms.

    Useful for printing registry contents, building CLI help text, etc.

    Returns:
        A dict mapping arm name -> {role, description, reference}.
    """
    return {
        name: {
            "role": spec.role,
            "description": spec.description,
            "reference": spec.reference,
        }
        for name, spec in ARM_REGISTRY.items()
    }
