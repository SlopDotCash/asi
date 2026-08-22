"""Reject oversized Forager hidden-size lists before layer-walk hang."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager import (
    _MAX_FORAGER_HIDDEN_LAYERS,
    AlbertaForagerConfig,
)


def test_forager_hidden_layer_cap_constant() -> None:
    assert _MAX_FORAGER_HIDDEN_LAYERS == 4096


def test_forager_accepts_max_actor_hidden_layer_count() -> None:
    AlbertaForagerConfig(actor_hidden_sizes=(1,) * _MAX_FORAGER_HIDDEN_LAYERS)


def test_forager_rejects_oversized_actor_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="actor_hidden_sizes length"):
        AlbertaForagerConfig(
            actor_hidden_sizes=(1,) * (_MAX_FORAGER_HIDDEN_LAYERS + 1),
        )


def test_forager_rejects_oversized_critic_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="critic_hidden_sizes length"):
        AlbertaForagerConfig(
            critic_hidden_sizes=(1,) * (_MAX_FORAGER_HIDDEN_LAYERS + 1),
        )
