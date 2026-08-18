"""Trust-boundary validation for forager_matched_statistics sanitized errors."""

from __future__ import annotations

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
    MatchedStatisticsError,
    PermutationSpec,
)

SEEDS = (101, 202, 303, 404)
EVIDENCE = EvidenceBinding(
    horizon=1000,
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
BOOTSTRAP = BootstrapSpec(resamples=257, seed=7001, confidence=0.95)
PERMUTATION = PermutationSpec(
    monte_carlo_resamples=512, seed=8001, familywise_alpha=0.05
)


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")

    def __hash__(self) -> int:  # type: ignore[override]
        raise AssertionError("EvilStr.__hash__ must not be called")


class _StringSubclass(str):
    pass


def _method(
    method_id: str,
    scores: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
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


def _contract(
    methods: tuple[LearningMethodScores, ...],
    primary: ComparisonSpec,
    diagnostics: tuple[DescriptiveDiagnosticScores, ...] = (),
) -> MatchedComparisonContract:
    return MatchedComparisonContract(
        methods=methods,
        primary_comparison=primary,
        secondary_comparisons=(),
        fixed_descriptive_diagnostics=diagnostics,
        bootstrap=BOOTSTRAP,
        permutation=PERMUTATION,
        primary_margin=0.0,
        primary_analysis_implementation_sha256=(
            PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256
        ),
        secondary_analysis_implementation_sha256=(
            SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256
        ),
    )


def test_not_preregistered_sanitized() -> None:
    m1 = _method("evil_method", preregistered=False)
    m2 = _method("good_method")
    with pytest.raises(MatchedStatisticsError, match="is not preregistered") as exc:
        _contract(
            (m1, m2),
            ComparisonSpec("h1", "evil_method", "good_method"),
        )
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_method'" in msg


def test_seed_ordering_sanitized() -> None:
    m1 = _method("evil_method", seeds=SEEDS)
    m2 = _method("good_method", seeds=(1, 2, 3, 4))
    with pytest.raises(MatchedStatisticsError, match="exact common") as exc:
        _contract(
            (m1, m2),
            ComparisonSpec("h1", "evil_method", "good_method"),
        )
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'good_method'" in msg or "'evil_method'" in msg


def test_evidence_binding_sanitized() -> None:
    other_evidence = EvidenceBinding(
        horizon=1000,
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
    m1 = _method("evil_method", evidence=EVIDENCE)
    m2 = _method("good_method", evidence=other_evidence)
    with pytest.raises(MatchedStatisticsError, match="different evidence") as exc:
        _contract(
            (m1, m2),
            ComparisonSpec("h1", "evil_method", "good_method"),
        )
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'good_method'" in msg or "'evil_method'" in msg


def test_diagnostic_seed_sanitized() -> None:
    m1 = _method("m1")
    m2 = _method("m2")
    diag = DescriptiveDiagnosticScores(
        candidate_id="evil_diag",
        seeds=(1, 2, 3, 4),
        scores=(1.0, 2.0, 3.0, 4.0),
        exclusion_reasons=("shared_agent_environment_rng",),
    )
    with pytest.raises(MatchedStatisticsError, match="exact common") as exc:
        _contract(
            (m1, m2),
            ComparisonSpec("h1", "m1", "m2"),
            diagnostics=(diag,),
        )
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_diag'" in msg


def test_comparison_unknown_sanitized() -> None:
    m1 = _method("m1")
    m2 = _method("m2")
    with pytest.raises(MatchedStatisticsError, match="references unknown") as exc:
        _contract(
            (m1, m2),
            ComparisonSpec("evil_hypothesis", "m1", "missing"),
        )
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_hypothesis'" in msg


def test_unknown_method_identifier_sanitized() -> None:
    m1 = _method("m1")
    m2 = _method("m2")
    contract = _contract((m1, m2), ComparisonSpec("h1", "m1", "m2"))
    with pytest.raises(MatchedStatisticsError, match="unknown method identifier") as exc:
        contract.method("evil_unknown")
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_unknown'" in msg


def test_unknown_method_identifier_hostile_subclass() -> None:
    m1 = _method("m1")
    m2 = _method("m2")
    contract = _contract((m1, m2), ComparisonSpec("h1", "m1", "m2"))
    evil = _EvilStr("evil")
    with pytest.raises(MatchedStatisticsError, match="portable identifier") as exc:
        contract.method(evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)
    with pytest.raises(MatchedStatisticsError, match="portable identifier"):
        contract.method(_StringSubclass("evil"))  # type: ignore[arg-type]


def test_method_id_exact_str_rejected() -> None:
    with pytest.raises(MatchedStatisticsError, match="portable identifier"):
        _method(_StringSubclass("evil"))  # type: ignore[arg-type]
    evil = _EvilStr("evil")
    with pytest.raises(MatchedStatisticsError, match="portable identifier") as exc:
        _method(evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)


def test_valid_contract_still_passes() -> None:
    m1 = _method("m1")
    m2 = _method("m2")
    c = _contract((m1, m2), ComparisonSpec("h1", "m1", "m2"))
    assert c.method("m1").method_id == "m1"
