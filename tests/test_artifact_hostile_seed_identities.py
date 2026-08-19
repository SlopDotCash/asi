"""Failing-first tests for exact-type hostile-int seed rejection in artifact validators."""
from __future__ import annotations

import pytest


def _multiagent_bad_seed_payload(kind: str) -> dict:
    return {
        "seed": {"hostile": True, "numpy_bool": False, "float": 4.0, "string": "4", "int_subclass": type("H", (int,), {"__int__": lambda s: 1/0})(4)}[kind],
        "conditions": {"active": [0]},
    }


def test_multiagent_artifact_rejects_bool_int_float_string_seed_identities() -> None:
    from alberta_framework.evaluation.continual_multiagent_artifact import (
        _validate_operational as _validate,
    )
    for kind in ("bool", "numpy_bool", "float", "string", "int_subclass"):
        payload = {"condition_timings": [_multiagent_bad_seed_payload(kind)]}
        result = _validate(payload)
        assert "seed must be an integer" in str(result.get("errors"))


def test_ftl_decision_artifact_rejects_hostile_seed_identities() -> None:
    from alberta_framework.evaluation.ftl_decision_artifact import (
        _validate_ftl_decision_summaries,
    )
    for kind in ("bool", "float", "string"):
        payload = {"seed_summaries": [_multiagent_bad_seed_payload(kind)]}
        result = _validate_ftl_decision_summaries(payload)
        assert "seed must be an integer" in str(result.get("errors"))


def test_recurring_feature_artifact_rejects_hostile_seed_identities() -> None:
    from alberta_framework.evaluation.recurring_feature_artifact import (
        _validate_recurring_feature_evidence,
    )
    for kind in ("bool", "float", "string"):
        payload = {"evidence_summaries": [_multiagent_bad_seed_payload(kind)]}
        result = _validate_recurring_feature_evidence(payload)
        assert "seed must be an integer" in str(result.get("errors"))
