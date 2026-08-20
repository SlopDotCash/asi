"""Reject oversized UPGD-memory hidden-size lists before per-layer validation hangs."""

from __future__ import annotations

import pytest

from alberta_framework.core.upgd_memory import (
    _MAX_UPGD_MEMORY_HIDDEN_LAYERS,
    UPGDMemoryConfig,
)


def test_upgd_memory_hidden_layer_cap_constant() -> None:
    assert _MAX_UPGD_MEMORY_HIDDEN_LAYERS == 4096


def test_upgd_memory_accepts_max_hidden_layer_count() -> None:
    UPGDMemoryConfig(
        feature_dim=2, n_heads=2, hidden_sizes=(1,) * _MAX_UPGD_MEMORY_HIDDEN_LAYERS
    )


def test_upgd_memory_rejects_oversized_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        UPGDMemoryConfig(
            feature_dim=2,
            n_heads=2,
            hidden_sizes=(1,) * (_MAX_UPGD_MEMORY_HIDDEN_LAYERS + 1),
        )


def test_upgd_memory_from_config_rejects_oversized_hidden_list() -> None:
    payload = UPGDMemoryConfig(feature_dim=2, n_heads=2, hidden_sizes=(4,)).to_config()
    payload["hidden_sizes"] = [1] * (_MAX_UPGD_MEMORY_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        UPGDMemoryConfig.from_config(payload)
