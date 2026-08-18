"""Hostile input and boundary validation for Forager matrix tuning contracts."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matrix import (
    ForagerMatrixManifestError,
    ForagerTuningRule,
    ForagerTuningSelection,
)


def test_forager_tuning_rule_valid_construction() -> None:
    rule = ForagerTuningRule(
        metric="episodic_reward_mean",
        direction="maximize",
        statistic="mean",
        confidence=0.95,
        bootstrap_resamples=1000,
        bootstrap_seed=42,
    )
    assert rule.metric == "episodic_reward_mean"
    assert rule.direction == "maximize"
    assert rule.statistic == "mean"
    assert rule.confidence == 0.95
    assert rule.bootstrap_resamples == 1000
    assert rule.bootstrap_seed == 42


def test_forager_tuning_rule_rejects_invalid_inputs() -> None:
    with pytest.raises(ForagerMatrixManifestError, match="metric must be a non-empty string"):
        ForagerTuningRule(
            metric="",
            direction="maximize",
            statistic="mean",
            confidence=0.95,
            bootstrap_resamples=1000,
            bootstrap_seed=42,
        )

    with pytest.raises(
        ForagerMatrixManifestError, match="direction must be 'maximize' or 'minimize'"
    ):
        ForagerTuningRule(
            metric="metric",
            direction="invalid",  # type: ignore[arg-type]
            statistic="mean",
            confidence=0.95,
            bootstrap_resamples=1000,
            bootstrap_seed=42,
        )

    with pytest.raises(ForagerMatrixManifestError, match="confidence must be a finite float in"):
        ForagerTuningRule(
            metric="metric",
            direction="maximize",
            statistic="mean",
            confidence=1.5,
            bootstrap_resamples=1000,
            bootstrap_seed=42,
        )

    with pytest.raises(
        ForagerMatrixManifestError, match="bootstrap_resamples must be an integer >= 1"
    ):
        ForagerTuningRule(
            metric="metric",
            direction="maximize",
            statistic="mean",
            confidence=0.95,
            bootstrap_resamples=0,
            bootstrap_seed=42,
        )

    with pytest.raises(ForagerMatrixManifestError, match="bootstrap_seed must be an integer in"):
        ForagerTuningRule(
            metric="metric",
            direction="maximize",
            statistic="mean",
            confidence=0.95,
            bootstrap_resamples=100,
            bootstrap_seed=-1,
        )


def test_forager_tuning_selection_validation() -> None:
    sel = ForagerTuningSelection(
        report_path="tuning/report.json",
        file_sha256="a" * 64,
        selected_variants={"eval_v1": "tune_v1"},
    )
    assert sel.report_path == "tuning/report.json"
    assert sel.file_sha256 == "a" * 64

    with pytest.raises(ForagerMatrixManifestError, match="report_path must be a non-empty string"):
        ForagerTuningSelection(
            report_path="",
            file_sha256="a" * 64,
            selected_variants={},
        )

    with pytest.raises(
        ForagerMatrixManifestError, match="file_sha256 must be a 64-character hex string"
    ):
        ForagerTuningSelection(
            report_path="tuning/report.json",
            file_sha256="invalid",
            selected_variants={},
        )
