"""Hostile input and boundary validation for IPMNIST screening dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_screening import (
    ScreeningRunResult,
    ScreeningSpec,
    _CBPLayerRefs,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig


def test_cbp_layer_refs_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="in_weight must be a non-empty string"):
        _CBPLayerRefs(
            in_weight="",
            in_bias="b1",
            out_weight="w2",
        )


def test_screening_spec_rejects_invalid_inputs() -> None:
    factory = screening_spec("upgd_w_control").factory
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        ScreeningSpec(
            name="",
            base_learner="upgd_w",
            mechanism="test",
            hyperparameters={},
            factory=factory,
        )

    with pytest.raises(TypeError, match="hyperparameters must be a dict"):
        ScreeningSpec(
            name="test",
            base_learner="upgd_w",
            mechanism="test",
            hyperparameters=None,  # type: ignore[arg-type]
            factory=factory,
        )


def test_screening_run_result_rejects_invalid_inputs() -> None:
    dummy_arr = np.zeros((1, 1))
    with pytest.raises(TypeError, match="config must be an IPMNISTConfig"):
        ScreeningRunResult(
            config_name="test",
            base_learner="upgd_w",
            hyperparameters={},
            seed=0,
            config=None,  # type: ignore[arg-type]
            per_task_accuracy=dummy_arr,
            per_task_loss=dummy_arr,
            per_task_plasticity=dummy_arr,
            wall_clock_seconds=1.0,
        )

    with pytest.raises(ValueError, match="wall_clock_seconds must be a finite float"):
        ScreeningRunResult(
            config_name="test",
            base_learner="upgd_w",
            hyperparameters={},
            seed=0,
            config=IPMNISTConfig(),
            per_task_accuracy=dummy_arr,
            per_task_loss=dummy_arr,
            per_task_plasticity=dummy_arr,
            wall_clock_seconds=float("inf"),
        )
