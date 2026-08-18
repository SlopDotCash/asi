"""Hostile input and boundary validation for ForagerMatrixManifest."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matrix import (
    ForagerMatrixManifest,
    ForagerTuningRule,
)


@pytest.fixture
def dummy_rule() -> ForagerTuningRule:
    return ForagerTuningRule(
        metric="mean_reward",
        direction="maximize",
        statistic="mean",
        confidence=0.95,
        bootstrap_resamples=100,
        bootstrap_seed=0,
    )


def test_forager_matrix_manifest_rejects_invalid_inputs(
    dummy_rule: ForagerTuningRule,
) -> None:
    with pytest.raises(ValueError, match="schema_version must be a non-empty string"):
        ForagerMatrixManifest(
            schema_version="",
            preset="field_of_view",
            stage="evaluation",
            steps=100,
            seeds=(0,),
            jax_chunk_size=1,
            seed_batch_size=1,
            mode="strict",
            source_execution_mode="live_tree_unsealed",
            metric_evidence_mode="scalar_summary_unsealed",
            selection_rule=dummy_rule,
            variants={},
        )

    with pytest.raises(ValueError, match="steps must be a positive integer"):
        ForagerMatrixManifest(
            schema_version="2.3",
            preset="field_of_view",
            stage="evaluation",
            steps=0,
            seeds=(0,),
            jax_chunk_size=1,
            seed_batch_size=1,
            mode="strict",
            source_execution_mode="live_tree_unsealed",
            metric_evidence_mode="scalar_summary_unsealed",
            selection_rule=dummy_rule,
            variants={},
        )

    with pytest.raises(TypeError, match="selection_rule must be a ForagerTuningRule"):
        ForagerMatrixManifest(
            schema_version="2.3",
            preset="field_of_view",
            stage="evaluation",
            steps=100,
            seeds=(0,),
            jax_chunk_size=1,
            seed_batch_size=1,
            mode="strict",
            source_execution_mode="live_tree_unsealed",
            metric_evidence_mode="scalar_summary_unsealed",
            selection_rule=None,  # type: ignore[arg-type]
            variants={},
        )
