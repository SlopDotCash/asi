"""Actor-critic key boundaries accept only typed scalar Threefry keys."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.actor_critic import (
    ActorCriticAgent,
    ActorCriticConfig,
    ContinuousActorCriticAgent,
    ContinuousActorCriticConfig,
)

pytestmark = pytest.mark.unit

_AGENTS: tuple[Callable[[], Any], ...] = (
    lambda: ActorCriticAgent(ActorCriticConfig(n_actions=2)),
    lambda: ContinuousActorCriticAgent(ContinuousActorCriticConfig(action_dim=1)),
)

_BAD_KEYS: dict[str, Callable[[], Any]] = {
    "legacy_prng_key": lambda: jr.PRNGKey(7),
    "raw_uint32_words": lambda: jnp.array([0, 7], dtype=jnp.uint32),
    "rbg": lambda: jr.key(7, impl="rbg"),
    "batched_typed": lambda: jr.split(jr.key(7)),
}


@pytest.mark.parametrize("make_agent", _AGENTS)
@pytest.mark.parametrize("bad_key", sorted(_BAD_KEYS))
def test_init_rejects_non_typed_scalar_threefry_keys(
    make_agent: Callable[[], Any], bad_key: str
) -> None:
    with pytest.raises(ValueError, match="Threefry"):
        make_agent().init(3, _BAD_KEYS[bad_key]())


@pytest.mark.parametrize("make_agent", _AGENTS)
def test_runtime_rejects_state_with_legacy_rng_key(make_agent: Callable[[], Any]) -> None:
    agent = make_agent()
    state = agent.init(3, jr.key(7))
    assert state.rng_key.shape == ()
    assert str(jr.key_impl(state.rng_key)) == "threefry2x32"
    legacy_state = state.replace(rng_key=jr.PRNGKey(7))
    with pytest.raises(ValueError, match="Threefry"):
        agent.select_action(legacy_state, jnp.ones((3,), dtype=jnp.float32))
