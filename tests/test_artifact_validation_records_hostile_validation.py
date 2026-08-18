"""Hostile input, leftover identity, and policy validation for artifact validation records."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.continual_ia_artifact import IAArtifactValidation
from alberta_framework.evaluation.continual_multiagent_artifact import (
    ArtifactValidation as MultiAgentArtifactValidation,
)
from alberta_framework.evaluation.ftl_decision_artifact import (
    ArtifactValidation as FTLArtifactValidation,
)
from alberta_framework.evaluation.upgd_ipmnist_nonpromoting import UPGDIPMNISTValidation


def test_upgd_ipmnist_validation_valid_construction() -> None:
    val = UPGDIPMNISTValidation(
        valid=True,
        errors=(),
        partial_sha256=(("sha", "val"),),
        artifact_sha256="a" * 64,
        observed_seed_pairs=(("learner", 42),),
    )
    assert val.valid is True
    assert val.development_only is True
    assert val.scientific_promotion_allowed is False


def test_upgd_ipmnist_validation_rejects_promotion_bypass() -> None:
    with pytest.raises(
        ValueError, match="scientific_promotion_allowed must permanently remain False"
    ):
        UPGDIPMNISTValidation(
            valid=True,
            errors=(),
            scientific_promotion_allowed=True,
        )

    with pytest.raises(ValueError, match="development_only must permanently remain True"):
        UPGDIPMNISTValidation(
            valid=True,
            errors=(),
            development_only=False,
        )


def test_upgd_ipmnist_validation_rejects_invalid_types() -> None:
    with pytest.raises(ValueError, match="valid must be a boolean"):
        UPGDIPMNISTValidation(valid=1, errors=())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="errors must be a tuple of strings"):
        UPGDIPMNISTValidation(valid=True, errors=["err"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="artifact_sha256 must be a string or None"):
        UPGDIPMNISTValidation(valid=True, errors=(), artifact_sha256=123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "cls",
    [
        FTLArtifactValidation,
        MultiAgentArtifactValidation,
        IAArtifactValidation,
    ],
)
def test_artifact_validation_classes_construction_and_validation(cls: type) -> None:
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
