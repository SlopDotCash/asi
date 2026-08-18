"""Hostile-identity tests for reference life scorecard consumer."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.reference_life_scorecard import validate_scorecard_artifact


class _HostileMapping(dict):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __iter__")

    def __getitem__(self, key):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __getitem__")

    def get(self, key, default=None):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile get")

    def __contains__(self, key):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __contains__")


class _HostileList(list):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __iter__")

    def __len__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __len__")

    def __getitem__(self, index):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile __getitem__")


def test_validate_rejects_hostile_mapping_without_iter() -> None:
    hostile = _HostileMapping(
        {
            "schema": "test",
            "schema_version": 1,
            "benchmark": "reference_life_matched_development_scorecard",
            "plan": {},
            "plan_sha256": "x",
            "evidence_policy": {},
            "source_identity": {},
            "runtime_identity": {},
            "dependency_identity": {},
            "identity_scope_note": (
                "consistency binding only; not authenticated execution attestation"
            ),
            "run_order": [],
            "runs": [],
            "summary": {},
            "artifact_sha256": "x",
        }
    )
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        validate_scorecard_artifact(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0


def test_validate_rejects_hostile_plan_without_get() -> None:
    # Top is builtin dict, but plan value is hostile mapping
    hostile_plan = _HostileMapping({"api_version": "v1"})
    payload: dict[str, object] = {
        "schema": "alberta.reference_life.matched_development_scorecard.v1",
        "schema_version": 1,
        "benchmark": "reference_life_matched_development_scorecard",
        "plan": hostile_plan,
        "plan_sha256": "0" * 64,
        "evidence_policy": {"promotion_allowed": False},
        "source_identity": {},
        "runtime_identity": {},
        "dependency_identity": {},
        "identity_scope_note": "consistency binding only; not authenticated execution attestation",
        "run_order": [],
        "runs": [],
        "summary": {},
        "artifact_sha256": "0" * 64,
    }
    _HostileMapping.calls = 0
    with pytest.raises((ValueError, TypeError)):
        validate_scorecard_artifact(payload)  # type: ignore[arg-type]
    # Hostile plan rejected via exact type check without dispatch
    assert _HostileMapping.calls == 0


def test_validate_rejects_hostile_list_without_len() -> None:
    hostile_runs = _HostileList([])
    payload: dict[str, object] = {
        "schema": "alberta.reference_life.matched_development_scorecard.v1",
        "schema_version": 1,
        "benchmark": "reference_life_matched_development_scorecard",
        "plan": {},
        "plan_sha256": "0" * 64,
        "evidence_policy": {"promotion_allowed": False},
        "source_identity": {},
        "runtime_identity": {},
        "dependency_identity": {},
        "identity_scope_note": "consistency binding only; not authenticated execution attestation",
        "run_order": [],
        "runs": hostile_runs,
        "summary": {},
        "artifact_sha256": "0" * 64,
    }
    _HostileList.calls = 0
    with pytest.raises((ValueError, TypeError)):
        validate_scorecard_artifact(payload)  # type: ignore[arg-type]
    assert _HostileList.calls == 0
