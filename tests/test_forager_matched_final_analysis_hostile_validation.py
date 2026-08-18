"""Hostile input and boundary validation for Forager matched final analysis bundle."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_final_analysis import (
    ContentVerifiedFinalAnalysisBundle,
    ForagerMatchedFinalAnalysisError,
    FreshFinalAnalysisBindings,
)


def test_final_analysis_bundle_rejects_invalid_root() -> None:
    with pytest.raises(ForagerMatchedFinalAnalysisError, match="output_root must be a Path"):
        ContentVerifiedFinalAnalysisBundle(
            output_root="invalid/path",  # type: ignore[arg-type]
            manifest={},
            seal_content=None,  # type: ignore[arg-type]
            evaluation_score_evidence=None,  # type: ignore[arg-type]
            evaluation_verification_request=None,  # type: ignore[arg-type]
            open_bindings_cache=None,  # type: ignore[arg-type]
            evaluation_bindings_cache=None,  # type: ignore[arg-type]
            analysis_runtime_source={},
            contract=None,  # type: ignore[arg-type]
            result=None,  # type: ignore[arg-type]
        )


def test_fresh_final_analysis_bindings_rejects_invalid_bindings() -> None:
    with pytest.raises(
        ForagerMatchedFinalAnalysisError,
        match="open_bindings must be an AuthenticatedEvidenceBindings",
    ):
        FreshFinalAnalysisBindings(
            open_bindings=None,  # type: ignore[arg-type]
            evaluation_bindings=None,  # type: ignore[arg-type]
        )
