"""Reject oversized Horde actor-critic hidden-size lists before layer-walk hang."""

from __future__ import annotations

import pytest

from alberta_framework.core.horde_actor_critic import (
    _MAX_HORDE_ACTOR_HIDDEN_LAYERS,
    NonlinearHordeActorCriticConfig,
)


def test_horde_actor_hidden_layer_cap_constant() -> None:
    assert _MAX_HORDE_ACTOR_HIDDEN_LAYERS == 4096


def test_horde_actor_accepts_max_hidden_layer_count() -> None:
    NonlinearHordeActorCriticConfig(
        n_actions=2,
        hidden_sizes=(1,) * _MAX_HORDE_ACTOR_HIDDEN_LAYERS,
    )


def test_horde_actor_rejects_oversized_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        NonlinearHordeActorCriticConfig(
            n_actions=2,
            hidden_sizes=(1,) * (_MAX_HORDE_ACTOR_HIDDEN_LAYERS + 1),
        )


def test_horde_actor_from_config_rejects_oversized_hidden_list() -> None:
    payload = NonlinearHordeActorCriticConfig(n_actions=2, hidden_sizes=(4,)).to_config()
    payload["hidden_sizes"] = [1] * (_MAX_HORDE_ACTOR_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        NonlinearHordeActorCriticConfig.from_config(payload)
