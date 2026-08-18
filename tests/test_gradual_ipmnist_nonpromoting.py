from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.evaluation.gradual_ipmnist_nonpromoting as gradual_report
from alberta_framework.benchmarks.ipmnist_gradual import run_gradual_input_pair
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig
from alberta_framework.evaluation.gradual_ipmnist_nonpromoting import (
    GradualInputDevelopmentPlan,
    build_gradual_input_development_report,
    retain_frozen_gradual_input_development_report,
    validate_gradual_input_development_report,
)


def _tiny() -> tuple[np.ndarray, np.ndarray, GradualInputDevelopmentPlan]:
    x = np.asarray([[-1.0, -0.5, 0.5, 1.0], [1.0, 0.5, -0.5, -1.0]], dtype=np.float32)
    y = np.asarray([0, 1], dtype=np.int32)
    return (
        x,
        y,
        GradualInputDevelopmentPlan(
            seeds=(19,),
            config=IPMNISTConfig(
                n_tasks=2, task_length=2, input_dim=4, hidden1=3, hidden2=2, n_classes=2
            ),
            transition_steps=1,
        ),
    )


def test_gradual_report_is_strict_derived_and_nonpromoting() -> None:
    x, y, plan = _tiny()
    run = run_gradual_input_pair(
        x,
        y,
        learner_name="adamw_control",
        seed=19,
        config=plan.config,
        transition_steps=1,
    )
    report = build_gradual_input_development_report(plan, (run,), x, y)
    validate_gradual_input_development_report(report, x, y)
    payload = cast(dict[str, Any], report)

    assert payload["policy"]["scientific_promotion_allowed"] is False
    assert payload["records"][0]["arms"][0]["metrics"]["correct"] == int(
        run.correct_counts[0].sum()
    )
    forged = copy.deepcopy(payload)
    forged["records"][0]["arms"][0]["metrics"]["online_accuracy"] = 1.0
    with pytest.raises(ValueError, match="derived"):
        validate_gradual_input_development_report(forged, x, y)


def test_gradual_report_revalidates_mutated_run_receipts() -> None:
    x, y, plan = _tiny()
    run = run_gradual_input_pair(
        x,
        y,
        learner_name="adamw_control",
        seed=19,
        config=plan.config,
        transition_steps=1,
    )
    object.__setattr__(run, "updates", 999)
    with pytest.raises(ValueError, match="counter"):
        build_gradual_input_development_report(plan, (run,), x, y)


def test_gradual_report_rejects_hostile_json_keys_without_hooks() -> None:
    class Hostile(str):
        calls = 0

        def __hash__(self) -> int:
            self.calls += 1
            return super().__hash__()

        def __eq__(self, other: object) -> bool:
            self.calls += 1
            raise AssertionError("must not compare")

    key = Hostile("schema")
    payload = {key: "x"}
    key.calls = 0
    with pytest.raises(ValueError, match="exact JSON"):
        validate_gradual_input_development_report(payload, np.zeros((2, 2)), np.zeros(2))
    assert key.calls == 0


def test_gradual_report_retention_is_exclusive_and_reload_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    x, y, plan = _tiny()
    run = run_gradual_input_pair(
        x,
        y,
        learner_name="adamw_control",
        seed=19,
        config=plan.config,
        transition_steps=1,
    )
    report = build_gradual_input_development_report(plan, (run,), x, y)
    monkeypatch.setattr(gradual_report, "FROZEN_GRADUAL_INPUT_PLAN", plan)
    destination = retain_frozen_gradual_input_development_report(
        report, x, y, repository_root=tmp_path
    )
    assert destination.read_bytes() == gradual_report.canonical_gradual_input_development_bytes(
        report, x, y
    )
    with pytest.raises(FileExistsError):
        retain_frozen_gradual_input_development_report(report, x, y, repository_root=tmp_path)
