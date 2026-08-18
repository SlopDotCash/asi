"""Hostile input and boundary validation for Forager comparison dataclasses."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_results import (
    ForagerComparisonReport,
    ForagerPairedComparison,
)


def test_forager_paired_comparison_valid_construction() -> None:
    comp = ForagerPairedComparison(
        candidate="cand_1",
        baseline="base_1",
        candidate_privileged=False,
        baseline_privileged=False,
        metric="mean_reward",
        seeds=(1, 2, 3),
        mean_difference=0.5,
        ci_low=0.2,
        ci_high=0.8,
        confidence=0.95,
    )
    assert comp.candidate == "cand_1"
    assert comp.seeds == (1, 2, 3)


def test_forager_paired_comparison_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="candidate must be a non-empty string"):
        ForagerPairedComparison(
            candidate="",
            baseline="base_1",
            candidate_privileged=False,
            baseline_privileged=False,
            metric="mean_reward",
            seeds=(1, 2),
            mean_difference=0.5,
            ci_low=0.2,
            ci_high=0.8,
            confidence=0.95,
        )

    with pytest.raises(TypeError, match="candidate_privileged must be an exact boolean"):
        ForagerPairedComparison(
            candidate="cand_1",
            baseline="base_1",
            candidate_privileged=1,  # type: ignore[arg-type]
            baseline_privileged=False,
            metric="mean_reward",
            seeds=(1, 2),
            mean_difference=0.5,
            ci_low=0.2,
            ci_high=0.8,
            confidence=0.95,
        )

    with pytest.raises(ValueError, match="seeds must be unique"):
        ForagerPairedComparison(
            candidate="cand_1",
            baseline="base_1",
            candidate_privileged=False,
            baseline_privileged=False,
            metric="mean_reward",
            seeds=(1, 1),
            mean_difference=0.5,
            ci_low=0.2,
            ci_high=0.8,
            confidence=0.95,
        )

    with pytest.raises(ValueError, match="mean_difference must be a finite float"):
        ForagerPairedComparison(
            candidate="cand_1",
            baseline="base_1",
            candidate_privileged=False,
            baseline_privileged=False,
            metric="mean_reward",
            seeds=(1, 2),
            mean_difference=float("nan"),
            ci_low=0.2,
            ci_high=0.8,
            confidence=0.95,
        )


def test_forager_comparison_report_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="summaries must be a mapping"):
        ForagerComparisonReport(
            candidate="cand_1",
            metric="mean_reward",
            summaries=None,  # type: ignore[arg-type]
            paired_comparisons=(),
            unpaired_methods=(),
        )

    with pytest.raises(TypeError, match="paired_comparisons must be an exact tuple"):
        ForagerComparisonReport(
            candidate="cand_1",
            metric="mean_reward",
            summaries={},
            paired_comparisons=[],  # type: ignore[arg-type]
            unpaired_methods=(),
        )
