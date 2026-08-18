"""Hostile input and type validation for feature artifact and historical chain records."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.evidence_manifest import (
    _HistoricalFTLChainValidation,
    _HistoricalIAChainValidation,
)
from alberta_framework.evaluation.recurring_feature_artifact import (
    ArtifactValidation as RecurringArtifactValidation,
)
from alberta_framework.evaluation.scale_robust_feature_artifact import (
    ArtifactValidation as ScaleRobustArtifactValidation,
)


@pytest.mark.parametrize(
    "cls",
    [
        RecurringArtifactValidation,
        ScaleRobustArtifactValidation,
    ],
)
def test_feature_artifact_validation_construction_and_guards(cls: type) -> None:
    inst = cls(valid=True, accepted=True, errors=())
    assert inst.valid is True
    assert inst.accepted is True
    assert inst.errors == ()

    with pytest.raises(ValueError, match="valid must be a boolean"):
        cls(valid=1, accepted=True, errors=())

    with pytest.raises(ValueError, match="accepted must be a boolean"):
        cls(valid=True, accepted=1, errors=())

    with pytest.raises(ValueError, match="errors must be a tuple of strings"):
        cls(valid=True, accepted=True, errors=["error"])


def test_historical_ia_chain_validation() -> None:
    inst = _HistoricalIAChainValidation(
        valid=True,
        errors=(),
        artifacts=({"key": "val"},),
        record={"rec": 1},
    )
    assert inst.valid is True
    assert inst.errors == ()

    with pytest.raises(ValueError, match="valid must be a boolean"):
        _HistoricalIAChainValidation(
            valid=1,  # type: ignore[arg-type]
            errors=(),
            artifacts=(),
            record={},
        )

    with pytest.raises(ValueError, match="errors must be a tuple of strings"):
        _HistoricalIAChainValidation(
            valid=True,
            errors=["error"],  # type: ignore[arg-type]
            artifacts=(),
            record={},
        )

    with pytest.raises(ValueError, match="artifacts must be a tuple"):
        _HistoricalIAChainValidation(
            valid=True,
            errors=(),
            artifacts=[{"key": "val"}],  # type: ignore[arg-type]
            record={},
        )

    with pytest.raises(ValueError, match="record must be a dict"):
        _HistoricalIAChainValidation(
            valid=True,
            errors=(),
            artifacts=(),
            record=(),  # type: ignore[arg-type]
        )


def test_historical_ftl_chain_validation() -> None:
    inst = _HistoricalFTLChainValidation(
        valid=True,
        accepted=True,
        errors=(),
        artifacts=({"key": "val"},),
        record={"rec": 1},
    )
    assert inst.valid is True
    assert inst.accepted is True
    assert inst.errors == ()

    with pytest.raises(ValueError, match="valid must be a boolean"):
        _HistoricalFTLChainValidation(
            valid=1,  # type: ignore[arg-type]
            accepted=True,
            errors=(),
            artifacts=(),
            record={},
        )

    with pytest.raises(ValueError, match="accepted must be a boolean"):
        _HistoricalFTLChainValidation(
            valid=True,
            accepted="yes",  # type: ignore[arg-type]
            errors=(),
            artifacts=(),
            record={},
        )

    with pytest.raises(ValueError, match="artifacts must be a tuple"):
        _HistoricalFTLChainValidation(
            valid=True,
            accepted=True,
            errors=(),
            artifacts=[{"key": "val"}],  # type: ignore[arg-type]
            record={},
        )
