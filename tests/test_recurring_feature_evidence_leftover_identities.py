"""Constructor boundaries for recurring-feature evidence record identities."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from alberta_framework.recurring_feature_gate import (
    PAIRWISE_PROBE_SCOPE,
    FeatureMemoryBudget,
    PhaseEvidence,
    RecurringFeatureGateResult,
    RecurringFeatureProtocol,
    RecurringFeatureSeedEvidence,
    RecurringFeatureVariantEvidence,
    TaskRecoveryEvidence,
)


def _phase() -> PhaseEvidence:
    return PhaseEvidence(0, "A", 1, 0.25, None)


def _recovery() -> TaskRecoveryEvidence:
    return TaskRecoveryEvidence("A", None, (2, None))


def _seed() -> RecurringFeatureSeedEvidence:
    return RecurringFeatureSeedEvidence(
        seed=30,
        final_heldout_nmse=(0.1, 0.2, 0.3, math.inf),
        active_pairs=((0, 1),),
        candidate_pairs=((0, 1), (2, 3)),
        phase_evidence=(_phase(),),
        task_recovery=(_recovery(),),
        steps_seen=1,
    )


def _variant() -> RecurringFeatureVariantEvidence:
    return RecurringFeatureVariantEvidence("retained", 0.999, (_seed(),))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("phase_index", True),
        ("task", True),
        ("occurrence", True),
        ("prequential_nmse", True),
        ("recovery_steps", True),
    ),
)
def test_phase_evidence_rejects_leftover_identities(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(_phase(), **{field: value})


def test_task_recovery_rejects_leftover_identities() -> None:
    with pytest.raises(ValueError):
        replace(_recovery(), acquisition_steps=True)
    with pytest.raises(ValueError):
        replace(_recovery(), recurrence_steps=(True,))


def test_seed_evidence_rejects_nested_leftover_identities() -> None:
    with pytest.raises(ValueError):
        replace(_seed(), seed=True)
    with pytest.raises(ValueError):
        replace(_seed(), final_heldout_nmse=(True,))
    with pytest.raises(ValueError):
        replace(_seed(), active_pairs=((True, 1),))


def test_variant_and_result_reject_leftover_string_and_float_identities() -> None:
    with pytest.raises(ValueError):
        replace(_variant(), name=True)
    with pytest.raises(ValueError):
        replace(_variant(), utility_retention_decay=True)
    result = RecurringFeatureGateResult(
        protocol=RecurringFeatureProtocol(),
        memory_budget=FeatureMemoryBudget(3, 15, 4),
        retained=_variant(),
        no_retention=RecurringFeatureVariantEvidence("no_retention", None, (_seed(),)),
        scope=PAIRWISE_PROBE_SCOPE,
    )
    with pytest.raises(ValueError):
        replace(result, scope=True)


def test_unrecovered_and_degenerate_nmse_sentinels_remain_legal() -> None:
    phase = PhaseEvidence(0, "A", 1, math.inf, None)
    recovery = TaskRecoveryEvidence("A", None, (None,))
    seed = replace(
        _seed(),
        final_heldout_nmse=(math.inf,),
        phase_evidence=(phase,),
        task_recovery=(recovery,),
    )
    assert math.isinf(seed.final_heldout_nmse[0])
    assert seed.phase_evidence[0].recovery_steps is None
