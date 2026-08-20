"""Reject oversized UPGD hidden-size lists before per-layer validation hangs."""

from __future__ import annotations

import pytest

from alberta_framework.core.upgd import _MAX_UPGD_HIDDEN_LAYERS, UPGDLearner


def test_upgd_hidden_layer_cap_constant() -> None:
    assert _MAX_UPGD_HIDDEN_LAYERS == 4096


def test_upgd_accepts_max_hidden_layer_count() -> None:
    UPGDLearner(n_heads=1, hidden_sizes=(1,) * _MAX_UPGD_HIDDEN_LAYERS)


def test_upgd_rejects_oversized_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        UPGDLearner(n_heads=1, hidden_sizes=(1,) * (_MAX_UPGD_HIDDEN_LAYERS + 1))


def test_upgd_from_config_rejects_oversized_hidden_list() -> None:
    payload = UPGDLearner(n_heads=1, hidden_sizes=(4,)).to_config()
    payload["hidden_sizes"] = [1] * (_MAX_UPGD_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        UPGDLearner.from_config(payload)
