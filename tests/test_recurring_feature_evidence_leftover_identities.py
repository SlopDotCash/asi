"""Leftover-identity gates for recurring-feature evidence records."""

from __future__ import annotations

import math

import pytest

from alberta_framework.recurring_feature_gate import (
    FeatureMemoryBudget,
    PhaseEvidence,
    RecurringFeatureGateResult,
    RecurringFeatureProtocol,
    RecurringFeatureSeedEvidence,
    RecurringFeatureVariantEvidence,
    TaskRecoveryEvidence,
)


def _phase(**overrides: object) -> PhaseEvidence:
    payload: dict[str, object] = {
        "phase_index": 0,
        "task": "A",
        "occurrence": 1,
        "prequential_nmse": 0.1,
        "recovery_steps": None,
    }
    payload.update(overrides)
    return PhaseEvidence(**payload)  # type: ignore[arg-type]


def _recovery(**overrides: object) -> TaskRecoveryEvidence:
    payload: dict[str, object] = {
        "task": "A",
        "acquisition_steps": None,
        "recurrence_steps": (None,),
    }
    payload.update(overrides)
    return TaskRecoveryEvidence(**payload)  # type: ignore[arg-type]


def _seed(**overrides: object) -> RecurringFeatureSeedEvidence:
    payload: dict[str, object] = {
        "seed": 30,
        "final_heldout_nmse": (0.1, 0.2, 0.3, math.inf),
        "active_pairs": ((0, 1), (2, 3)),
        "candidate_pairs": ((4, 5),),
        "phase_evidence": (_phase(),),
        "task_recovery": (_recovery(),),
        "steps_seen": 9,
    }
    payload.update(overrides)
    return RecurringFeatureSeedEvidence(**payload)  # type: ignore[arg-type]


def test_phase_evidence_rejects_leftover_identities() -> None:
    """Public phase records must not keep leftover index/task/nmse identities."""

    with pytest.raises(ValueError, match="prequential_nmse"):
        _phase(prequential_nmse=True)
    with pytest.raises(ValueError, match="phase_index"):
        _phase(phase_index=True)
    with pytest.raises(ValueError, match="occurrence"):
        _phase(occurrence=True)
    with pytest.raises(ValueError, match="recovery_steps"):
        _phase(recovery_steps=True)
    with pytest.raises(ValueError, match="task"):
        _phase(task=True)

    legal = _phase(prequential_nmse=math.inf)
    assert type(legal.prequential_nmse) is float
    assert legal.prequential_nmse == math.inf
    assert legal.recovery_steps is None


def test_task_recovery_and_seed_evidence_reject_leftover_identities() -> None:
    """Public seed records must not keep leftover seed/pair/recovery identities."""

    with pytest.raises(ValueError, match="acquisition_steps"):
        _recovery(acquisition_steps=True)
    with pytest.raises(ValueError, match="recurrence_steps"):
        _recovery(recurrence_steps=(True,))
    with pytest.raises(ValueError, match="seed"):
        _seed(seed=True)
    with pytest.raises(ValueError, match="final_heldout_nmse"):
        _seed(final_heldout_nmse=(True,))
    with pytest.raises(ValueError, match="active_pairs"):
        _seed(active_pairs=((True, 1),))
    with pytest.raises(ValueError, match="steps_seen"):
        _seed(steps_seen=True)

    legal = _seed()
    assert legal.seed == 30
    assert type(legal.final_heldout_nmse[-1]) is float
    assert legal.final_heldout_nmse[-1] == math.inf


def test_variant_and_result_reject_leftover_identities() -> None:
    """Public variant/result records must not keep leftover name/decay/scope identities."""

    with pytest.raises(ValueError, match="name"):
        RecurringFeatureVariantEvidence(name=True, utility_retention_decay=None, seeds=())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="utility_retention_decay"):
        RecurringFeatureVariantEvidence(
            name="retained",
            utility_retention_decay=True,
            seeds=(),
        )

    retained = RecurringFeatureVariantEvidence("retained", 0.999, ())
    baseline = RecurringFeatureVariantEvidence("no_retention", None, ())
    with pytest.raises(ValueError, match="scope"):
        RecurringFeatureGateResult(
            protocol=RecurringFeatureProtocol(),
            memory_budget=FeatureMemoryBudget(3, 15, 4),
            retained=retained,
            no_retention=baseline,
            scope=True,  # type: ignore[arg-type]
        )

    legal = RecurringFeatureGateResult(
        protocol=RecurringFeatureProtocol(),
        memory_budget=FeatureMemoryBudget(3, 15, 4),
        retained=retained,
        no_retention=baseline,
    )
    assert type(legal.scope) is str
    assert type(retained.utility_retention_decay) is float
    assert baseline.utility_retention_decay is None
