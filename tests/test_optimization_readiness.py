from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.evaluation.optimization_readiness import (
    OPTIMIZATION_READINESS_PROTOCOL,
    energy_rank,
    estimate_optimization_readiness,
)


def test_readiness_matches_paper_equations() -> None:
    gradients = np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    report = estimate_optimization_readiness(loss=2.0, batch_gradients=gradients)
    assert report.gradient_squared_norm == pytest.approx(4.0)
    assert report.expected_batch_gradient_squared_norm == pytest.approx(5.0)
    assert report.gradient_strength == pytest.approx(2.0)
    assert report.gradient_reliability == pytest.approx(0.8)
    assert report.optimization_readiness == pytest.approx(1.6)
    assert report.gradient_norm == pytest.approx(2.0)


def test_zero_and_mechanism_off_reductions() -> None:
    zero = estimate_optimization_readiness(
        loss=1.0, batch_gradients=np.zeros((2, 3), dtype=np.float64)
    )
    assert zero.optimization_readiness == 0.0
    strength_only = estimate_optimization_readiness(
        loss=2.0,
        batch_gradients=np.asarray([[1.0], [3.0]]),
        include_reliability=False,
    )
    assert strength_only.optimization_readiness == strength_only.gradient_strength


def test_energy_rank_baseline() -> None:
    matrix = np.diag([3.0, 1.0])
    assert energy_rank(matrix, threshold=0.9) == 1
    assert energy_rank(matrix, threshold=0.99) == 2
    assert energy_rank(np.zeros((2, 2)), threshold=0.99) == 0


@pytest.mark.parametrize("loss", [0.0, -1.0, float("nan"), True])
def test_readiness_rejects_invalid_loss(loss: object) -> None:
    with pytest.raises(ValueError, match="loss"):
        estimate_optimization_readiness(  # type: ignore[arg-type]
            loss=loss, batch_gradients=np.ones((2, 2))
        )


def test_protocol_is_prospective_and_nonpromoting() -> None:
    assert OPTIMIZATION_READINESS_PROTOCOL["paper_revision"] == "arXiv:2605.09044v1"
    assert OPTIMIZATION_READINESS_PROTOCOL["diagnostics"] == (
        "optimization_readiness",
        "gradient_norm",
        "representation_energy_rank_0.99",
        "curvature_energy_rank_0.99",
        "parameter_norm",
    )
    assert OPTIMIZATION_READINESS_PROTOCOL["development_only"] is True
    assert OPTIMIZATION_READINESS_PROTOCOL["scientific_promotion_allowed"] is False
