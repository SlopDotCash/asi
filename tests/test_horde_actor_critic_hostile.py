"""Hostile string gates for Horde actor-critic agents."""

from __future__ import annotations

import types

import pytest

from alberta_framework.core.horde_actor_critic import (
    NonlinearQHordeActorCriticAgent,
    QHordeActorCriticAgent,
)
from alberta_framework.core.types import DemonType

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def _mock_critic(n_actions: int = 2):
    demons = [
        types.SimpleNamespace(demon_type=DemonType.CONTROL, gamma=0.0)
        for _ in range(n_actions)
    ]
    return types.SimpleNamespace(
        n_demons=n_actions,
        horde_spec=types.SimpleNamespace(demons=demons),
    )


def _mock_config(critic_target, actor_update="td_error"):
    return types.SimpleNamespace(
        n_actions=2,
        gamma=0.9,
        temperature=1.0,
        actor_td_error_clip=None,
        actor_gradient_clip_norm=None,
        critic_target=critic_target,
        actor_update=actor_update,
    )


def test_qhorde_critic_target_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("expected_sarsa")
    _HostileStr.calls = 0
    config = _mock_config(critic_target=hostile)
    critic = _mock_critic()
    with pytest.raises(ValueError, match="critic_target"):
        QHordeActorCriticAgent(config, critic)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
    # benign passes
    config2 = _mock_config(critic_target="sampled_sarsa")
    agent = QHordeActorCriticAgent(config2, critic)  # type: ignore[arg-type]
    assert agent is not None


def test_qhorde_actor_update_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("td_error")
    _HostileStr.calls = 0
    config = _mock_config(critic_target="expected_sarsa", actor_update=hostile)
    critic = _mock_critic()
    with pytest.raises(ValueError, match="actor_update"):
        QHordeActorCriticAgent(config, critic)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_nonlinear_qhorde_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("expected_sarsa")
    _HostileStr.calls = 0
    config = _mock_config(critic_target=hostile)
    critic = _mock_critic()
    with pytest.raises(ValueError, match="critic_target"):
        NonlinearQHordeActorCriticAgent(config, critic)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
    # benign
    config2 = _mock_config(critic_target="expected_sarsa", actor_update="expected_advantage")
    agent = NonlinearQHordeActorCriticAgent(config2, critic)  # type: ignore[arg-type]
    assert agent is not None


def test_non_string_rejects() -> None:
    config = _mock_config(critic_target=123)  # type: ignore[arg-type]
    critic = _mock_critic()
    with pytest.raises(ValueError, match="critic_target"):
        QHordeActorCriticAgent(config, critic)  # type: ignore[arg-type]


