"""Reject oversized latent world-model hidden-size lists before layer-walk hang."""

from __future__ import annotations

import pytest

from alberta_framework.core.latent_world_model import (
    _MAX_LATENT_HIDDEN_LAYERS,
    LatentWorldModelConfig,
)


def test_latent_hidden_layer_cap_constant() -> None:
    assert _MAX_LATENT_HIDDEN_LAYERS == 4096


def test_latent_accepts_max_hidden_layer_count() -> None:
    LatentWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        hidden_sizes=(1,) * _MAX_LATENT_HIDDEN_LAYERS,
    )


def test_latent_rejects_oversized_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(1,) * (_MAX_LATENT_HIDDEN_LAYERS + 1),
        )


def test_latent_from_config_rejects_oversized_hidden_list() -> None:
    payload = LatentWorldModelConfig(observation_dim=2, n_actions=2, hidden_sizes=(4,)).to_config()
    payload["hidden_sizes"] = [1] * (_MAX_LATENT_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        LatentWorldModelConfig.from_config(payload)
