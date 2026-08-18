"""Hostile input and boundary validation for UPGD benchmark results."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    IPMNISTRunResult,
)
from alberta_framework.benchmarks.upgd_label_emnist import (
    LabelEMNISTConfig,
    LabelEMNISTRunResult,
)


def test_ipmnist_run_result_rejects_invalid_inputs() -> None:
    dummy_arr = np.zeros((1, 1))
    with pytest.raises(TypeError, match="hyperparameters must be a dict"):
        IPMNISTRunResult(
            learner="upgd",
            hyperparameters=None,  # type: ignore[arg-type]
            seeds=(1,),
            config=IPMNISTConfig(),
            per_task_accuracy=dummy_arr,
            per_task_loss=dummy_arr,
            per_task_plasticity=dummy_arr,
            average_online_accuracy=dummy_arr,
            wall_clock_seconds=1.0,
        )

    with pytest.raises(TypeError, match="config must be an IPMNISTConfig"):
        IPMNISTRunResult(
            learner="upgd",
            hyperparameters={},
            seeds=(1,),
            config=None,  # type: ignore[arg-type]
            per_task_accuracy=dummy_arr,
            per_task_loss=dummy_arr,
            per_task_plasticity=dummy_arr,
            average_online_accuracy=dummy_arr,
            wall_clock_seconds=1.0,
        )


def test_label_emnist_run_result_rejects_invalid_inputs() -> None:
    dummy_arr = np.zeros((1, 1))
    with pytest.raises(TypeError, match="hyperparameters must be a dict"):
        LabelEMNISTRunResult(
            learner="upgd",
            hyperparameters=None,  # type: ignore[arg-type]
            seeds=(1,),
            config=LabelEMNISTConfig(),
            per_task_accuracy=dummy_arr,
            per_task_loss=dummy_arr,
            per_task_plasticity=dummy_arr,
            average_online_accuracy=dummy_arr,
            wall_clock_seconds=1.0,
        )

    with pytest.raises(TypeError, match="config must be a LabelEMNISTConfig"):
        LabelEMNISTRunResult(
            learner="upgd",
            hyperparameters={},
            seeds=(1,),
            config=None,  # type: ignore[arg-type]
            per_task_accuracy=dummy_arr,
            per_task_loss=dummy_arr,
            per_task_plasticity=dummy_arr,
            average_online_accuracy=dummy_arr,
            wall_clock_seconds=1.0,
        )
