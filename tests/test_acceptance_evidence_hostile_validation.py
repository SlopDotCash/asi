"""Hostile input, leftover identity, and boundary validation for evaluation acceptance records."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.continual_ia import (
    IAAcceptanceEvidence,
    IAAcceptanceResult,
)
from alberta_framework.evaluation.continual_multiagent import (
    AcceptanceEvidence,
    AcceptanceResult,
)


class HostileFloat:
    """Object attempting to bypass finite float validation."""

    def __float__(self) -> float:
        return 1.0


class HostileBool:
    """Object attempting to bypass bool validation."""

    def __bool__(self) -> bool:
        return True


class StringSubclass(str):
    """A leftover string identity that must not cross strict record boundaries."""


def test_acceptance_evidence_valid_construction() -> None:
    evidence = AcceptanceEvidence(
        name="coadaptation_uplift",
        passed=True,
        actual=0.15,
        comparator=">=",
        threshold=0.05,
        detail="p-value: 0.001",
    )
    assert evidence.name == "coadaptation_uplift"
    assert evidence.passed is True
    assert evidence.actual == 0.15
    assert evidence.comparator == ">="
    assert evidence.threshold == 0.05
    assert evidence.detail == "p-value: 0.001"


def test_ia_acceptance_evidence_rejects_scope_string_subclass() -> None:
    with pytest.raises(ValueError, match="scope must be 'primary' or 'secondary'"):
        IAAcceptanceEvidence(
            name="check",
            scope=StringSubclass("primary"),  # type: ignore[arg-type]
            passed=True,
            actual=0.1,
            comparator=">=",
            threshold=0.0,
            detail="ok",
        )


@pytest.mark.parametrize(
    "invalid_name",
    ["", 123, None, True, False],
)
def test_acceptance_evidence_rejects_invalid_name(invalid_name: object) -> None:
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        AcceptanceEvidence(
            name=invalid_name,  # type: ignore[arg-type]
            passed=True,
            actual=0.1,
            comparator=">=",
            threshold=0.0,
            detail="ok",
        )


@pytest.mark.parametrize(
    "invalid_passed",
    [1, 0, "True", "False", None, HostileBool(), 1.0],
)
def test_acceptance_evidence_rejects_non_boolean_passed(invalid_passed: object) -> None:
    with pytest.raises(ValueError, match="passed must be a boolean"):
        AcceptanceEvidence(
            name="check",
            passed=invalid_passed,  # type: ignore[arg-type]
            actual=0.1,
            comparator=">=",
            threshold=0.0,
            detail="ok",
        )


@pytest.mark.parametrize(
    "invalid_actual",
    [
        True,
        False,
        "0.5",
        None,
        HostileFloat(),
    ],
)
def test_acceptance_evidence_rejects_invalid_actual(invalid_actual: object) -> None:
    with pytest.raises(ValueError, match="actual must be a real number"):
        AcceptanceEvidence(
            name="check",
            passed=True,
            actual=invalid_actual,  # type: ignore[arg-type]
            comparator=">=",
            threshold=0.0,
            detail="ok",
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        "0.5",
        None,
        HostileFloat(),
    ],
)
def test_acceptance_evidence_rejects_invalid_threshold(invalid_threshold: object) -> None:
    with pytest.raises(ValueError, match="threshold must be a finite real number"):
        AcceptanceEvidence(
            name="check",
            passed=True,
            actual=0.1,
            comparator=">=",
            threshold=invalid_threshold,  # type: ignore[arg-type]
            detail="ok",
        )


@pytest.mark.parametrize(
    "invalid_comparator",
    ["", 123, None, True, False],
)
def test_acceptance_evidence_rejects_invalid_comparator(invalid_comparator: object) -> None:
    with pytest.raises(ValueError, match="comparator must be a non-empty string"):
        AcceptanceEvidence(
            name="check",
            passed=True,
            actual=0.1,
            comparator=invalid_comparator,  # type: ignore[arg-type]
            threshold=0.0,
            detail="ok",
        )


def test_acceptance_result_validation() -> None:
    evidence = AcceptanceEvidence(
        name="check",
        passed=True,
        actual=0.1,
        comparator=">=",
        threshold=0.0,
        detail="ok",
    )
    result = AcceptanceResult(passed=True, checks=(evidence,))
    assert result.passed is True
    assert result.checks == (evidence,)
    assert len(result.failures) == 0

    with pytest.raises(ValueError, match="passed must be a boolean"):
        AcceptanceResult(passed=1, checks=(evidence,))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="checks must be a tuple of AcceptanceEvidence"):
        AcceptanceResult(passed=True, checks=[evidence])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="checks must be a tuple of AcceptanceEvidence"):
        AcceptanceResult(passed=True, checks=(evidence, "invalid"))  # type: ignore[arg-type]


def test_ia_acceptance_evidence_valid_construction() -> None:
    evidence = IAAcceptanceEvidence(
        name="primary_uplift",
        scope="primary",
        passed=True,
        actual=0.25,
        comparator=">=",
        threshold=0.1,
        detail="statistically significant",
    )
    assert evidence.name == "primary_uplift"
    assert evidence.scope == "primary"
    assert evidence.passed is True
    assert evidence.actual == 0.25
    assert evidence.comparator == ">="
    assert evidence.threshold == 0.1
    assert evidence.detail == "statistically significant"


@pytest.mark.parametrize(
    "invalid_scope",
    ["tertiary", "other", "", 123, None, True],
)
def test_ia_acceptance_evidence_rejects_invalid_scope(invalid_scope: object) -> None:
    with pytest.raises(ValueError, match="scope must be 'primary' or 'secondary'"):
        IAAcceptanceEvidence(
            name="check",
            scope=invalid_scope,  # type: ignore[arg-type]
            passed=True,
            actual=0.1,
            comparator=">=",
            threshold=0.0,
            detail="ok",
        )


@pytest.mark.parametrize(
    "invalid_actual",
    [
        True,
        False,
        "0.5",
        None,
        HostileFloat(),
    ],
)
def test_ia_acceptance_evidence_rejects_invalid_actual(invalid_actual: object) -> None:
    with pytest.raises(ValueError, match="actual must be a real number"):
        IAAcceptanceEvidence(
            name="check",
            scope="primary",
            passed=True,
            actual=invalid_actual,  # type: ignore[arg-type]
            comparator=">=",
            threshold=0.0,
            detail="ok",
        )


def test_ia_acceptance_result_validation() -> None:
    evidence = IAAcceptanceEvidence(
        name="check",
        scope="primary",
        passed=True,
        actual=0.1,
        comparator=">=",
        threshold=0.0,
        detail="ok",
    )
    result = IAAcceptanceResult(
        passed=True,
        primary_passed=True,
        secondary_passed=True,
        checks=(evidence,),
    )
    assert result.passed is True
    assert result.primary_passed is True
    assert result.secondary_passed is True
    assert result.checks == (evidence,)
    assert len(result.failures) == 0

    with pytest.raises(ValueError, match="passed must be a boolean"):
        IAAcceptanceResult(
            passed=1,  # type: ignore[arg-type]
            primary_passed=True,
            secondary_passed=True,
            checks=(evidence,),
        )

    with pytest.raises(ValueError, match="primary_passed must be a boolean"):
        IAAcceptanceResult(
            passed=True,
            primary_passed=1,  # type: ignore[arg-type]
            secondary_passed=True,
            checks=(evidence,),
        )

    with pytest.raises(ValueError, match="secondary_passed must be a boolean"):
        IAAcceptanceResult(
            passed=True,
            primary_passed=True,
            secondary_passed=1,  # type: ignore[arg-type]
            checks=(evidence,),
        )

    with pytest.raises(ValueError, match="checks must be a tuple of IAAcceptanceEvidence"):
        IAAcceptanceResult(
            passed=True,
            primary_passed=True,
            secondary_passed=True,
            checks=[evidence],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("nonfinite_actual", [float("nan"), float("inf"), float("-inf")])
def test_acceptance_evidence_records_nonfinite_actual(nonfinite_actual: float) -> None:
    """Non-finite observed values are recorded rejections, never construction crashes."""
    evidence = AcceptanceEvidence(
        name="mean_recurrence_recovery_steps",
        passed=False,
        actual=nonfinite_actual,
        comparator="<=",
        threshold=1.0,
        detail="non-finite evidence fails the comparison without a skipped state",
    )
    assert evidence.passed is False
    assert evidence.actual != evidence.actual or abs(evidence.actual) == float("inf")


@pytest.mark.parametrize("nonfinite_actual", [float("nan"), float("inf"), float("-inf")])
def test_ia_acceptance_evidence_records_nonfinite_actual(nonfinite_actual: float) -> None:
    evidence = IAAcceptanceEvidence(
        name="check",
        scope="primary",
        passed=False,
        actual=nonfinite_actual,
        comparator=">=",
        threshold=0.0,
        detail="recorded rejection",
    )
    assert evidence.passed is False


def test_minimum_check_records_failed_comparison_for_nonfinite() -> None:
    """The evaluator's documented contract: no skipped or 'not evaluated' state."""
    from alberta_framework.evaluation.continual_multiagent import (
        _maximum_check,
        _minimum_check,
    )

    for value in (float("nan"), float("inf"), float("-inf")):
        low = _minimum_check("check", value, 1.0, "detail")
        high = _maximum_check("check", value, 1.0, "detail")
        assert low.passed is False
        assert high.passed is False
