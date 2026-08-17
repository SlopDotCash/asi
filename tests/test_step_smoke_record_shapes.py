"""Cross-Step smoke-result shape identity and consistency tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from alberta_framework.steps._smoke_record_validation import require_step_shape
from alberta_framework.steps.step1 import Step1KernelConfig, Step1SmokeResult
from alberta_framework.steps.step2 import (
    Step2AssociativeConfig,
    Step2AssociativeSmokeResult,
    Step2KernelConfig,
    Step2SmokeResult,
)
from alberta_framework.steps.step4 import Step4SARSAConfig, Step4SmokeResult
from alberta_framework.steps.step5 import Step5AverageRewardTDConfig, Step5SmokeResult
from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig, Step6SmokeResult
from alberta_framework.steps.step7 import Step7DynaConfig, Step7SmokeResult
from alberta_framework.steps.step8 import Step8SmokeResult, Step8WorldModelConfig
from alberta_framework.steps.step9 import Step9DreamingConfig, Step9SmokeResult
from alberta_framework.steps.step10 import Step10SmokeResult, Step10STOMPConfig
from alberta_framework.steps.step11 import Step11OaKConfig, Step11SmokeResult
from alberta_framework.steps.step12 import Step12IAConfig, Step12SmokeResult


@pytest.mark.parametrize("value", ([], (), (True,), (-1,), (2**31,), (7,)))
def test_shared_shape_gate_rejects_noncanonical_or_inconsistent_values(value: object) -> None:
    with pytest.raises(ValueError):
        require_step_shape("shape", value, steps=8)


def _records_and_shape_fields() -> tuple[tuple[object, tuple[str, ...]], ...]:
    return (
        (Step1SmokeResult(Step1KernelConfig(), 8, 0, 0.0, (8,), True), ("metrics_shape",)),
        (
            Step2SmokeResult(Step2KernelConfig(), 8, 0, 0.0, (8,), True, {}),
            ("metrics_shape",),
        ),
        (
            Step2AssociativeSmokeResult(
                Step2AssociativeConfig(), 8, 0, 1.0, 0.5, (8,), True, {}
            ),
            ("metrics_shape",),
        ),
        (
            Step4SmokeResult(Step4SARSAConfig(), 8, 0, (8, 2), (8,), (8,), True, {}),
            ("q_values_shape", "td_errors_shape", "actions_shape"),
        ),
        (
            Step5SmokeResult(
                Step5AverageRewardTDConfig(), 8, 0, (8,), (8,), (8,), True, {}
            ),
            ("predictions_shape", "td_errors_shape", "average_rewards_shape"),
        ),
        (
            Step6SmokeResult(
                Step6DifferentialSARSAConfig(),
                8,
                0,
                (8, 2),
                (8,),
                (8,),
                (8,),
                True,
                {},
            ),
            ("q_values_shape", "td_errors_shape", "average_rewards_shape", "actions_shape"),
        ),
        (
            Step7SmokeResult(
                Step7DynaConfig(),
                8,
                0,
                (8,),
                (8, 1),
                (8, 1),
                (8, 1),
                (8, 1),
                (8,),
                True,
                0,
                {},
                {},
            ),
            (
                "real_td_errors_shape",
                "planning_td_errors_shape",
                "planning_priorities_shape",
                "planning_anchor_indices_shape",
                "planning_importance_ratios_shape",
                "actions_shape",
            ),
        ),
        (
            Step8SmokeResult(
                Step8WorldModelConfig(), 8, 0, (8,), (8, 4), (8,), (8, 4), True, {}
            ),
            (
                "reward_predictions_shape",
                "next_observation_predictions_shape",
                "reward_errors_shape",
                "next_observation_errors_shape",
            ),
        ),
        (
            Step9SmokeResult(
                Step9DreamingConfig(), 8, 0, (8,), (8, 1), (8,), True, 0, {}, {}
            ),
            ("real_td_errors_shape", "dream_td_errors_shape", "actions_shape"),
        ),
        (
            Step10SmokeResult(
                Step10STOMPConfig(),
                8,
                0,
                (8,),
                (8,),
                (8,),
                (8,),
                (8, 0),
                True,
                0,
                {},
            ),
            (
                "td_errors_shape",
                "average_rewards_shape",
                "primitive_actions_shape",
                "executing_options_shape",
                "pseudo_rewards_shape",
            ),
        ),
        (
            Step11SmokeResult(
                Step11OaKConfig(), 8, 0, (8,), (8,), (8,), (8, 0), True, 0, {}
            ),
            (
                "td_errors_shape",
                "average_rewards_shape",
                "primitive_actions_shape",
                "utility_emas_shape",
            ),
        ),
        (
            Step12SmokeResult(
                Step12IAConfig(), 8, 0, (8, 4), (8, 4), (8,), (8, 8), (8,), True, {}
            ),
            (
                "predictions_shape",
                "cerebellum_errors_shape",
                "recommendations_shape",
                "augmented_obs_shape",
                "cortex_td_errors_shape",
            ),
        ),
    )


def test_every_changed_step_smoke_record_rejects_bool_shape_identities() -> None:
    for record, fields in _records_and_shape_fields():
        for field in fields:
            with pytest.raises(ValueError, match=field):
                replace(record, **{field: (True,)})


def test_every_changed_step_smoke_record_rejects_step_shape_mismatch() -> None:
    for record, fields in _records_and_shape_fields():
        for field in fields:
            with pytest.raises(ValueError, match="must equal steps"):
                replace(record, **{field: (7,)})
