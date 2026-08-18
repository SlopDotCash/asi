"""Hostile-identity tests for recurrent latent ensemble config consumer."""

from __future__ import annotations

import pytest

from alberta_framework.core.recurrent_latent_world_model_ensemble import (
    RecurrentLatentWorldModelEnsemble,
)


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


def test_from_config_rejects_hostile_nested_without_dispatch() -> None:
    hostile = _HostileMapping({"latent_dim": 1})
    payload = {"type": "RecurrentLatentWorldModelEnsemble", "config": hostile}
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        RecurrentLatentWorldModelEnsemble.from_config(payload)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0


def test_from_config_rejects_hostile_nested_direct_without_dispatch() -> None:
    # Also verify dict(payload) not dispatching hostile when config is exact dict
    # hostile nested must be rejected before any get/set
    hostile = _HostileMapping({"ensemble_size": 1})
    payload = {"type": "RecurrentLatentWorldModelEnsemble", "config": hostile}
    _HostileMapping.calls = 0
    with pytest.raises(ValueError):
        RecurrentLatentWorldModelEnsemble.from_config(payload)  # type: ignore[arg-type]
    assert _HostileMapping.calls == 0
