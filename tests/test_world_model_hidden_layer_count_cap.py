"""Reject oversized world-model hidden-size lists before per-layer validation hangs."""

from __future__ import annotations

import pytest

from alberta_framework.core.world_model import (
    _MAX_WORLD_MODEL_HIDDEN_LAYERS,
    WorldModelConfig,
)


def test_world_model_hidden_layer_cap_constant() -> None:
    assert _MAX_WORLD_MODEL_HIDDEN_LAYERS == 4096


def test_world_model_accepts_max_hidden_layer_count() -> None:
    WorldModelConfig(
        observation_dim=2,
        n_actions=2,
        hidden_sizes=(1,) * _MAX_WORLD_MODEL_HIDDEN_LAYERS,
    )


def test_world_model_rejects_oversized_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        WorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(1,) * (_MAX_WORLD_MODEL_HIDDEN_LAYERS + 1),
        )


def test_world_model_from_config_rejects_oversized_hidden_list() -> None:
    payload = WorldModelConfig(observation_dim=2, n_actions=2, hidden_sizes=(4,)).to_config()
    payload["hidden_sizes"] = [1] * (_MAX_WORLD_MODEL_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        WorldModelConfig.from_config(payload)
