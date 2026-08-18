"""Hostile-identity tests for feature discovery config consumer."""

from __future__ import annotations

import pytest

from alberta_framework.core.feature_discovery import FixedBudgetFeatureLearner


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

    def keys(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile keys")

    def items(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile items")


def test_feature_discovery_from_config_rejects_hostile_without_dispatch() -> None:
    hostile = _HostileMapping({"type": "FixedBudgetFeatureLearner"})
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        FixedBudgetFeatureLearner.from_config(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0


def test_feature_discovery_from_config_rejects_hostile_nested_without_dispatch() -> None:
    hostile = _HostileMapping(
        {
            "type": "FixedBudgetFeatureLearner",
            "generator_mix": [1.0, 0.0, 0.0],
        }
    )
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        FixedBudgetFeatureLearner.from_config(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0
