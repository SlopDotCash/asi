"""Hostile-identity tests for canonical UPGD config consumer."""

from __future__ import annotations

import pytest

from alberta_framework.core.canonical_upgd import CanonicalUPGDConfig, _copy_config_mapping


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


def test_copy_config_mapping_rejects_hostile_without_dispatch() -> None:
    hostile = _HostileMapping({"a": 1})
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        _copy_config_mapping("CanonicalUPGD config", hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0


def test_canonical_upgd_from_config_rejects_hostile_without_dispatch() -> None:
    hostile = _HostileMapping(
        {
            "type": "CanonicalUPGD",
            "step_size": 0.01,
            "utility_decay": 0.9,
            "noise_std": 0.0,
            "weight_decay": 0.0,
            "mode": "protecting",
            "profile": "paper_global",
            "normalization": "global",
            "epsilon": 1e-8,
        }
    )
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        CanonicalUPGDConfig.from_config(hostile)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0
