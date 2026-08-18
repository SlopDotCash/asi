"""Hostile-identity tests for interaction features config consumer."""

from __future__ import annotations

import pytest

from alberta_framework.core.interaction_features import FixedBudgetInteractionLearner


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
    def keys(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile keys")
    def items(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile items")

def test_interaction_from_config_rejects_hostile_without_dispatch() -> None:
    hostile = _HostileMapping({"type": "FixedBudgetInteractionLearner"})
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        FixedBudgetInteractionLearner.from_config(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0
