"""Regression coverage for #2214: Step 7 planning and Step 9 dreaming loop
lengths must be bounded before arange/scan materialization.

Oversized config values previously passed validation (up to int32 max) and
attempted multi-gigabyte arange buffers at trace time.
"""

import pytest

from alberta_framework.steps.step7 import (
    _PLANNING_SCAN_BUDGET,
    Step7DynaConfig,
)
from alberta_framework.steps.step9 import (
    _DREAM_SCAN_BUDGET,
    Step9DreamingConfig,
)


def test_step7_budget_cap() -> None:
    assert _PLANNING_SCAN_BUDGET.maximum_steps == 10_000


def test_step9_budget_cap() -> None:
    assert _DREAM_SCAN_BUDGET.maximum_steps == 10_000


def _step7_cfg(**overrides):
    from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
    from alberta_framework.steps.step8 import Step8WorldModelConfig

    base = {
        "control": Step6DifferentialSARSAConfig(n_actions=2),
        "world_model": Step8WorldModelConfig(observation_dim=2, n_actions=2),
    }
    base.update(overrides)
    return Step7DynaConfig(**base)


def test_step7_oversized_planning_steps_rejected() -> None:
    # Construction itself must fail (config validation runs in __post_init__).
    with pytest.raises(ValueError, match="Step 7 planning budget"):
        _step7_cfg(planning_steps=10**9)


def test_step7_oversized_rollout_depth_rejected() -> None:
    with pytest.raises(ValueError, match="Step 7 planning budget"):
        _step7_cfg(planning_rollout_depth=10**9)


def test_step9_oversized_planning_budget_rejected() -> None:
    with pytest.raises(ValueError, match="Step 9 dreaming budget"):
        Step9DreamingConfig(planning_budget=10**9)


def test_step9_oversized_rollout_horizon_rejected() -> None:
    with pytest.raises(ValueError, match="Step 9 dreaming budget"):
        Step9DreamingConfig(dream_rollout_horizon=10**9)


def test_step9_oversized_candidate_count_rejected() -> None:
    with pytest.raises(ValueError, match="Step 9 dreaming budget"):
        Step9DreamingConfig(dream_candidate_count=10**9)
