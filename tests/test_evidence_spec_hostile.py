"""Complete EvidenceSpec identity contract: leftover, types, and roster gates."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from alberta_framework.evaluation.evidence_manifest import EVIDENCE_SPECS, EvidenceSpec


class StringSubclass(str):
    """Leftover string identity that must not cross the spec boundary."""


def _load(_path: Path) -> dict[str, object]:
    return {"ok": True}


def _validate(_artifact: Mapping[str, object]) -> object:
    class _Result:
        valid = True
        accepted = True
        errors: tuple[str, ...] = ()

    return _Result()


def _legal(**overrides: object) -> EvidenceSpec:
    payload: dict[str, object] = {
        "name": "fixture_claim",
        "claim_scope": "test-only scope",
        "evidence_class": "scientific",
        "evidence_level": "L2",
        "promotes_scientific_claim": True,
        "relative_path": Path("fixture.json"),
        "expected_schema": "test.schema.v1",
        "command_argv": ("python", "-m", "fixture"),
        "protocol": {"protocol_version": "test.protocol.v1"},
        "configuration": {"steps": 1},
        "seeds": {"development": (0,)},
        "thresholds": {"minimum_effect": 0.25},
        "limitations": ("test fixture only",),
        "source_paths": (Path("fixture.py"),),
        "required_environment_fields": ("python",),
        "loader": _load,
        "validator": _validate,
    }
    payload.update(overrides)
    return EvidenceSpec(**payload)  # type: ignore[arg-type]


def test_evidence_spec_accepts_canonical_and_registered_identities() -> None:
    spec = _legal()
    assert spec.name == "fixture_claim"
    assert spec.evidence_class == "scientific"
    assert spec.promotes_scientific_claim is True
    assert spec.relative_path == Path("fixture.json")
    names = {registered.name for registered in EVIDENCE_SPECS}
    assert names == {
        "recurring_pair_features",
        "scale_robust_pair_features",
        "ftl_world_model_decision_fidelity",
        "recurring_multiagent_coadaptation",
        "continual_intelligence_amplification",
    }


def test_evidence_spec_rejects_leftover_string_identities() -> None:
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        _legal(name=True)
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        _legal(name=StringSubclass("fixture_claim"))
    with pytest.raises(ValueError, match="claim_scope must be a non-empty string"):
        _legal(claim_scope=True)
    with pytest.raises(ValueError, match="expected_schema must be a non-empty string"):
        _legal(expected_schema="")


def test_evidence_spec_rejects_leftover_class_level_and_bool_identities() -> None:
    with pytest.raises(ValueError, match="evidence_class must be a known evidence class"):
        _legal(evidence_class=True)
    with pytest.raises(ValueError, match="evidence_class must be a known evidence class"):
        _legal(evidence_class=StringSubclass("scientific"))
    with pytest.raises(ValueError, match="evidence_class must be a known evidence class"):
        _legal(evidence_class="anecdote")
    with pytest.raises(ValueError, match="evidence_level must be a known evidence level"):
        _legal(evidence_level="L9")
    with pytest.raises(ValueError, match="promotes_scientific_claim must be a boolean"):
        _legal(promotes_scientific_claim=1)


def test_evidence_spec_rejects_leftover_path_tuple_and_mapping_identities() -> None:
    with pytest.raises(ValueError, match="relative_path must be a Path"):
        _legal(relative_path="fixture.json")
    with pytest.raises(ValueError, match="command_argv must be a non-empty tuple"):
        _legal(command_argv=["python"])
    with pytest.raises(ValueError, match="limitations must be a non-empty tuple"):
        _legal(limitations=["test fixture only"])
    with pytest.raises(ValueError, match="source_paths must be a non-empty tuple"):
        _legal(source_paths=["fixture.py"])
    with pytest.raises(ValueError, match="protocol must be a non-empty dict"):
        _legal(protocol=True)
    with pytest.raises(ValueError, match="seeds must be a non-empty dict"):
        _legal(seeds={True: 0})
    with pytest.raises(ValueError, match="loader must be callable"):
        _legal(loader=None)
    with pytest.raises(ValueError, match="validator must be callable"):
        _legal(validator="validate")
