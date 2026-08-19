"""Hostile string gate for evidence_manifest protocol_version before dispatch."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.evidence_manifest import EvidenceSpec, _validate_spec

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash")

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile len")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str")


def _hostile_spec_with_protocol_version(value):
    # Bypass frozen dataclass __post_init__ validation to test _validate_spec directly
    spec = object.__new__(EvidenceSpec)
    # minimal valid fields, using real spec as template then override protocol
    from alberta_framework.evaluation.evidence_manifest import EVIDENCE_SPECS
    base = EVIDENCE_SPECS[0]
    object.__setattr__(spec, "name", base.name)
    object.__setattr__(spec, "claim_scope", base.claim_scope)
    object.__setattr__(spec, "evidence_class", base.evidence_class)
    object.__setattr__(spec, "evidence_level", base.evidence_level)
    object.__setattr__(spec, "promotes_scientific_claim", base.promotes_scientific_claim)
    object.__setattr__(spec, "relative_path", base.relative_path)
    object.__setattr__(spec, "expected_schema", base.expected_schema)
    object.__setattr__(spec, "command_argv", base.command_argv)
    object.__setattr__(spec, "protocol", {"protocol_version": value})
    object.__setattr__(spec, "configuration", base.configuration)
    object.__setattr__(spec, "seeds", base.seeds)
    object.__setattr__(spec, "thresholds", base.thresholds)
    object.__setattr__(spec, "limitations", base.limitations)
    object.__setattr__(spec, "source_paths", base.source_paths)
    object.__setattr__(spec, "required_environment_fields", base.required_environment_fields)
    object.__setattr__(spec, "loader", base.loader)
    object.__setattr__(spec, "validator", base.validator)
    return spec


def test_protocol_version_rejects_hostile_before_dispatch() -> None:
    hostile = _HostileStr("1.0")
    _HostileStr.calls = 0
    spec = _hostile_spec_with_protocol_version(hostile)
    with pytest.raises(ValueError, match="protocol must carry"):
        _validate_spec(spec)
    assert _HostileStr.calls == 0


def test_protocol_version_benign_still_passes() -> None:
    spec = _hostile_spec_with_protocol_version("1.0")
    _validate_spec(spec)


def test_protocol_version_non_str_rejected() -> None:
    spec = _hostile_spec_with_protocol_version(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="protocol must carry"):
        _validate_spec(spec)
