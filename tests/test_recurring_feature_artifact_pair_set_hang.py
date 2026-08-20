"""Recurring-feature artifacts reject oversized feature_dim before pair-set rebuild."""

from __future__ import annotations

import time

import pytest

from alberta_framework.evaluation.recurring_feature_artifact import (
    _result_integrity,
    build_recurring_feature_artifact,
)
from alberta_framework.recurring_feature_gate import (
    MAX_RECURRING_FEATURE_DIM,
    FeatureMemoryBudget,
    PhaseEvidence,
    RecurringFeatureGateResult,
    RecurringFeatureProtocol,
    RecurringFeatureSeedEvidence,
    RecurringFeatureVariantEvidence,
    TaskRecoveryEvidence,
)


def _seed() -> RecurringFeatureSeedEvidence:
    return RecurringFeatureSeedEvidence(
        seed=30,
        final_heldout_nmse=(0.1, 0.2, 0.3, 0.4),
        active_pairs=((0, 1),),
        candidate_pairs=((0, 1), (2, 3)),
        phase_evidence=(PhaseEvidence(0, "A", 1, 0.25, None),),
        task_recovery=(TaskRecoveryEvidence("A", 1, (2,)),),
        steps_seen=1,
    )


def _result(*, feature_dim: object) -> RecurringFeatureGateResult:
    seed = _seed()
    return RecurringFeatureGateResult(
        protocol=RecurringFeatureProtocol(feature_dim=feature_dim),  # type: ignore[arg-type]
        memory_budget=FeatureMemoryBudget(3, 15, 4),
        retained=RecurringFeatureVariantEvidence("retained", 0.999, (seed,)),
        no_retention=RecurringFeatureVariantEvidence("no_retention", None, (seed,)),
    )


def test_frozen_feature_dim_is_inside_the_reconstruction_bound() -> None:
    assert RecurringFeatureProtocol().feature_dim == 6
    assert 6 <= RecurringFeatureProtocol().feature_dim <= MAX_RECURRING_FEATURE_DIM


def test_last_fit_feature_dim_reconstructs_without_pair_set_hang() -> None:
    started = time.perf_counter()
    finite, archive = _result_integrity(_result(feature_dim=MAX_RECURRING_FEATURE_DIM))
    assert time.perf_counter() - started < 0.5
    assert finite is False
    assert archive is False


@pytest.mark.parametrize("feature_dim", [65, 5000])
def test_build_rejects_oversized_feature_dim_before_pair_set(
    feature_dim: int,
) -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match=r"integer in \[6, 64\]"):
        build_recurring_feature_artifact(
            _result(feature_dim=feature_dim),
            gate_wall_seconds=0.0,
        )
    assert time.perf_counter() - started < 0.5


def test_public_result_decision_rejects_oversized_feature_dim_before_pair_set() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match=r"feature_dim must be in \[6, 64\]"):
        _result(feature_dim=5000).decision()
    assert time.perf_counter() - started < 0.5


def test_build_rejects_non_int_feature_dim_before_pair_set() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="exact int"):
        build_recurring_feature_artifact(
            _result(feature_dim=True),
            gate_wall_seconds=0.0,
        )
    assert time.perf_counter() - started < 0.5


class _HostileInt(int):
    def __lt__(self, other: object) -> bool:
        raise AssertionError("hostile integer comparison ran")

    def __gt__(self, other: object) -> bool:
        raise AssertionError("hostile integer comparison ran")

    def __index__(self) -> int:
        raise AssertionError("hostile integer index conversion ran")


def test_build_rejects_hostile_int_subclass_before_hooks() -> None:
    with pytest.raises(ValueError, match="exact int"):
        build_recurring_feature_artifact(
            _result(feature_dim=_HostileInt(6)),
            gate_wall_seconds=0.0,
        )
