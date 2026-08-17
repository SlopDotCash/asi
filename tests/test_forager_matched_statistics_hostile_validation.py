"""Hostile validation for forager matched statistics trust boundary."""
# mypy: disable-error-code="arg-type"

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.forager_matched_statistics import (
    PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256,
    SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256,
    BootstrapSpec,
    ComparisonSpec,
    DescriptiveDiagnosticScores,
    EvidenceBinding,
    LearningMethodScores,
    MatchedComparisonContract,
    PermutationSpec,
    _require_exact_str,
)

SEEDS = (101, 202, 303, 404)
EVIDENCE = EvidenceBinding(
    horizon=1_000,
    metric_sha256="1" * 64,
    environment_sha256="2" * 64,
    rng_schedule_sha256="3" * 64,
    runtime_profile_sha256="4" * 64,
    source_evidence_sha256="5" * 64,
    executor_evidence_sha256="6" * 64,
    score_evidence_sha256="7" * 64,
    execution_closure_sha256="8" * 64,
    authenticated_bindings_sha256="9" * 64,
    external_verification_subject_sha256="e" * 64,
    external_verification_receipt_sha256="a" * 64,
    sealed_protocol_sha256="b" * 64,
    selection_result_sha256="c" * 64,
    selection_report_sha256="d" * 64,
)
BOOTSTRAP = BootstrapSpec(resamples=257, seed=7_001, confidence=0.95)
PERMUTATION = PermutationSpec(monte_carlo_resamples=512, seed=8_001, familywise_alpha=0.05)


class _EvilStr(str):
    calls = 0

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


def _method(
    method_id: str,
    scores: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0),
    *,
    seeds: tuple[int, ...] = SEEDS,
    evidence: EvidenceBinding = EVIDENCE,
    preregistered: bool = True,
) -> LearningMethodScores:
    return LearningMethodScores(
        method_id=method_id,
        seeds=seeds,
        scores=scores,
        evidence=evidence,
        preregistered=preregistered,
    )


def _contract_with_methods(
    methods: tuple[LearningMethodScores, ...],
    primary_comparison: ComparisonSpec,
    secondary: tuple[ComparisonSpec, ...] = (),
    diagnostics: tuple[DescriptiveDiagnosticScores, ...] = (),
) -> MatchedComparisonContract:
    return MatchedComparisonContract(
        methods=methods,
        primary_comparison=primary_comparison,
        secondary_comparisons=secondary,
        fixed_descriptive_diagnostics=diagnostics,
        bootstrap=BOOTSTRAP,
        permutation=PERMUTATION,
        primary_margin=0.0,
        primary_analysis_implementation_sha256=PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256,
        secondary_analysis_implementation_sha256=SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256,
    )


def test_require_exact_str_rejects_evil() -> None:
    evil = _EvilStr("v")
    _EvilStr.calls = 0
    with pytest.raises(Exception, match="exact string") as exc:
        _require_exact_str("key", evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(Exception, match="exact string"):
        _require_exact_str("key", _StringSubclass("v"))


def test_method_rejects_evil_without_hooks() -> None:
    alberta = _method("alberta_candidate_v1", (2.0, 3.0, 4.0, 5.0))
    primary = _method("primary_learning_baseline", (1.0, 2.0, 3.0, 4.0))
    contract = _contract_with_methods(
        (alberta, primary),
        ComparisonSpec(
            hypothesis_id="primary_superiority",
            intervention_id=alberta.method_id,
            comparator_id=primary.method_id,
        ),
    )
    evil = _EvilStr("bad")
    _EvilStr.calls = 0
    with pytest.raises(Exception, match="exact string") as exc:
        contract.method(evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_method_rejects_subclass() -> None:
    alberta = _method("alberta_candidate_v1")
    primary = _method("primary_learning_baseline", (1.0, 2.0, 3.0, 4.0))
    contract = _contract_with_methods(
        (alberta, primary),
        ComparisonSpec(
            hypothesis_id="primary_superiority",
            intervention_id=alberta.method_id,
            comparator_id=primary.method_id,
        ),
    )
    with pytest.raises(Exception, match="exact string"):
        contract.method(_StringSubclass("bad"))  # type: ignore[arg-type]


def test_method_sanitized() -> None:
    alberta = _method("alberta_candidate_v1")
    primary = _method("primary_learning_baseline", (1.0, 2.0, 3.0, 4.0))
    contract = _contract_with_methods(
        (alberta, primary),
        ComparisonSpec(
            hypothesis_id="primary_superiority",
            intervention_id=alberta.method_id,
            comparator_id=primary.method_id,
        ),
    )
    with pytest.raises(Exception, match="unknown method identifier") as exc:
        contract.method("nope")
    assert "!r" not in str(exc.value)
    assert "nope" in str(exc.value)
    assert "'" in str(exc.value)


def test_contract_not_preregistered_sanitized() -> None:
    bad = _method("bad_method", (2.0, 3.0, 4.0, 5.0), preregistered=False)
    good = _method("good_method", (1.0, 2.0, 3.0, 4.0))
    with pytest.raises(Exception, match="not preregistered") as exc:
        _contract_with_methods(
            (bad, good),
            ComparisonSpec(
                hypothesis_id="hyp1",
                intervention_id=bad.method_id,
                comparator_id=good.method_id,
            ),
        )
    assert "!r" not in str(exc.value)
    assert "bad_method" in str(exc.value)
    assert "'" in str(exc.value)


def test_contract_seed_mismatch_sanitized() -> None:
    m1 = _method("m1", (2.0, 3.0, 4.0, 5.0), seeds=(1, 2, 3, 4))
    m2 = _method("m2", (1.0, 2.0, 3.0, 4.0), seeds=(5, 6, 7, 8))
    with pytest.raises(Exception, match="exact common seed") as exc:
        _contract_with_methods(
            (m1, m2),
            ComparisonSpec(hypothesis_id="hyp1", intervention_id="m1", comparator_id="m2"),
        )
    assert "!r" not in str(exc.value)
    assert "'" in str(exc.value)


def test_contract_evidence_mismatch_sanitized() -> None:
    other_evidence = EvidenceBinding(
        horizon=1_000,
        metric_sha256="f" * 64,
        environment_sha256="2" * 64,
        rng_schedule_sha256="3" * 64,
        runtime_profile_sha256="4" * 64,
        source_evidence_sha256="5" * 64,
        executor_evidence_sha256="6" * 64,
        score_evidence_sha256="7" * 64,
        execution_closure_sha256="8" * 64,
        authenticated_bindings_sha256="9" * 64,
        external_verification_subject_sha256="e" * 64,
        external_verification_receipt_sha256="a" * 64,
        sealed_protocol_sha256="b" * 64,
        selection_result_sha256="c" * 64,
        selection_report_sha256="d" * 64,
    )
    m1 = _method("m1", evidence=EVIDENCE)
    m2 = _method("m2", evidence=other_evidence)
    with pytest.raises(Exception, match="different evidence") as exc:
        _contract_with_methods(
            (m1, m2),
            ComparisonSpec(hypothesis_id="hyp1", intervention_id="m1", comparator_id="m2"),
        )
    assert "!r" not in str(exc.value)


def test_contract_diagnostic_seed_mismatch_sanitized() -> None:
    m1 = _method("m1")
    m2 = _method("m2", (1.0, 2.0, 3.0, 4.0))
    diag = DescriptiveDiagnosticScores(
        candidate_id="diag1",
        seeds=(9, 10, 11, 12),
        scores=(0.0, 0.0, 0.0, 0.0),
        exclusion_reasons=("reason1",),
    )
    with pytest.raises(Exception, match="exact common seed") as exc:
        _contract_with_methods(
            (m1, m2),
            ComparisonSpec(hypothesis_id="hyp1", intervention_id="m1", comparator_id="m2"),
            diagnostics=(diag,),
        )
    assert "!r" not in str(exc.value)
    assert "diag1" in str(exc.value)


def test_contract_unknown_methods_sanitized() -> None:
    m1 = _method("m1")
    m2 = _method("m2", (1.0, 2.0, 3.0, 4.0))
    with pytest.raises(Exception, match="references unknown methods") as exc:
        _contract_with_methods(
            (m1, m2),
            ComparisonSpec(hypothesis_id="hyp1", intervention_id="m1", comparator_id="unknown_bad"),
        )
    assert "!r" not in str(exc.value)
    assert "hyp1" in str(exc.value)
    assert "unknown_bad" in str(exc.value)
    assert "'" in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/benchmarks/forager_matched_statistics.py").read_text()
    assert "!r" not in text
