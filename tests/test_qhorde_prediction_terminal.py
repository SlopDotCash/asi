"""Q-Horde prediction demons must not bootstrap across a terminal transition."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.horde_actor_critic import (
    NonlinearQHordeActorCriticAgent,
    NonlinearQHordeActorCriticConfig,
    QHordeActorCriticAgent,
    QHordeActorCriticConfig,
)
from alberta_framework.core.optimizers import LMS
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec

_CHAIN = 3
_GAMMA = 0.9


def _obs(pos: int) -> jnp.ndarray:
    return jnp.eye(_CHAIN, dtype=jnp.float32)[pos]


@pytest.mark.parametrize("nonlinear", [False, True])
def test_prediction_demon_learns_episodic_return_under_auto_reset(nonlinear: bool) -> None:
    # Episodic chain of three states with Gymnasium-style auto-reset: the
    # terminal transition's next observation is the next episode's first
    # observation. A prediction demon with cumulant 1 and gamma 0.9 must learn
    # the episodic returns 2.71, 1.9 and 1.0. Bootstrapping across the
    # terminal onto the reset observation drives every value to 1 / (1 - 0.9).
    specs = [
        GVFSpec(name=f"q{i}", demon_type=DemonType.CONTROL, gamma=0.0, lamda=0.0, cumulant_index=-1)
        for i in range(2)
    ]
    specs.append(
        GVFSpec(
            name="steps",
            demon_type=DemonType.PREDICTION,
            gamma=_GAMMA,
            lamda=0.0,
            cumulant_index=-1,
        )
    )
    critic = HordeLearner(
        create_horde_spec(specs), hidden_sizes=(), use_layer_norm=False, step_size=0.1
    )
    kwargs = dict(n_actions=2, gamma=_GAMMA, actor_lamda=0.0, temperature=1.0)
    agent: QHordeActorCriticAgent | NonlinearQHordeActorCriticAgent
    if nonlinear:
        agent = NonlinearQHordeActorCriticAgent(
            NonlinearQHordeActorCriticConfig(hidden_sizes=(2,), use_layer_norm=False, **kwargs),
            critic,
            actor_optimizer=LMS(step_size=0.01),  # type: ignore[arg-type]
        )
    else:
        agent = QHordeActorCriticAgent(
            QHordeActorCriticConfig(actor_step_size=0.01, **kwargs), critic
        )
    state = agent.init(_CHAIN, jr.key(0))
    state = agent.start(state, _obs(0))[0]
    pos = 0
    cumulant = jnp.array([1.0], dtype=jnp.float32)
    for _ in range(3000):
        pos += 1
        terminated = pos == _CHAIN
        if terminated:
            pos = 0
        state = agent.update(
            state,
            jnp.float32(0.0),
            _obs(pos),
            jnp.bool_(terminated),
            prediction_cumulants=cumulant,
        ).state

    learned = [float(critic.predict(state.critic_state, _obs(p))[2]) for p in range(_CHAIN)]
    np.testing.assert_allclose(learned, [2.71, 1.9, 1.0], atol=0.02)
