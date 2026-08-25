"""Receipt-only regression tests for the C-CHAIN development lane."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_screening import (
    cchain_development_result_payload,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig
from alberta_framework.evaluation.cchain_ipmnist_nonpromoting import (
    validate_cchain_development_result,
)


def test_receipt_rejects_coefficient_adaptation_before_delay() -> None:
    config = IPMNISTConfig(
        n_tasks=1,
        task_length=4,
        input_dim=4,
        hidden1=3,
        hidden2=2,
        n_classes=2,
    )
    features = np.asarray(
        [
            [-1.0, -0.5, 0.5, 1.0],
            [1.0, 0.5, -0.5, -1.0],
            [-0.5, 1.0, -1.0, 0.5],
            [0.5, -1.0, 1.0, -0.5],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 0, 1], dtype=np.int32)
    result = run_screening_config(
        features,
        labels,
        screening_spec("cchain_full"),
        0,
        config,
    )
    receipt = cchain_development_result_payload(result, outcome="inconclusive")

    assert receipt["metrics"]["diagnostic_updates"] == 2.0  # type: ignore[index]
    assert receipt["metrics"]["final_coefficient"] == 1.0  # type: ignore[index]
    receipt["metrics"]["final_coefficient"] = 2.0  # type: ignore[index]

    with pytest.raises(ValueError, match="coefficient cannot adapt before its delay"):
        validate_cchain_development_result(receipt)
