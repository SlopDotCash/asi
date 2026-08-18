"""Leftover-identity gates for recurring-feature gate decisions."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from alberta_framework.recurring_feature_gate import RecurringFeatureGateDecision


def test_gate_decision_rejects_leftover_identities() -> None:
    """Public decisions must not keep leftover accepted/summary/failures identities."""

    with pytest.raises(ValueError, match="accepted"):
        RecurringFeatureGateDecision(1, "ok", ())
    with pytest.raises(ValueError, match="accepted"):
        RecurringFeatureGateDecision(0, "fail", ("x",))
    with pytest.raises(ValueError, match="accepted"):
        RecurringFeatureGateDecision("FIXED", "ok", ())
    with pytest.raises(ValueError, match="summary"):
        RecurringFeatureGateDecision(True, True, ())
    with pytest.raises(ValueError, match="failures"):
        RecurringFeatureGateDecision(True, "ok", None)
    with pytest.raises(ValueError, match="failures"):
        RecurringFeatureGateDecision(True, "ok", ("x", 1))

    legal = RecurringFeatureGateDecision(True, "ok", ())
    dumped = json.dumps(asdict(legal), allow_nan=False)
    assert dumped == '{"accepted": true, "summary": "ok", "failures": []}'
    assert '"accepted": 1' not in dumped
    assert '"accepted": "FIXED"' not in dumped
    legal.require()
