"""Development calibration for RiverSwim differential SARSA.

The frozen reference-life scorecard plan stays unchanged. On that plan the
differential-SARSA arm uses a state-independent bias and decays epsilon from
0.5 to 0.02 over 2,500 steps. The retained 2026-10-09 run records zero
high-end visits on all 12 RiverSwim seeds (mean return about 96.6). Replaying
those hyperparameters on the direct RiverSwim chain reproduces that miss.

A separate, permanently nonpromoting calibration turns the bias off, because a
shared bias copies the small left reward into every unvisited state, and it
spreads exploration across the 20,000-step horizon. On fresh seeds 0 through 5
this direct replay visited the high end on every seed. Five seeds then earned
about 3,200 return; seed 2 still finished near the random return after a
single visit. This is not a scorecard gate, not `reference-dev`, and not
scientific evidence.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import random as jr

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
)
from alberta_framework.streams.closed_loop import RiverSwimConfig, RiverSwimMDP

FROZEN_DIFFERENTIAL_SARSA: dict[str, float | int | bool] = {
    "q_step_size": 0.1,
    "average_reward_step_size": 0.01,
    "trace_decay": 0.0,
    "epsilon_start": 0.5,
    "epsilon_end": 0.02,
    "epsilon_decay_steps": 2_500,
    "use_bias": True,
}

CALIBRATED_DIFFERENTIAL_SARSA: dict[str, float | int | bool] = {
    "q_step_size": 0.1,
    "average_reward_step_size": 0.01,
    "trace_decay": 0.0,
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay_steps": 20_000,
    "use_bias": False,
}


def riverswim_differential_return(
    seed: int,
    horizon: int,
    overrides: Mapping[str, float | int | bool],
) -> tuple[int, float]:
    """Return ``(high_end_visits, reward_sum)`` for one direct RiverSwim life.

    The chain is the default six-state RiverSwim, not the reference-life
    runner. The retained scorecard used that same transition and reward
    kernel. ``seed`` is the JAX key; it is not a consumed scorecard seed.
    """

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative int")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive int")
    config = DifferentialSARSAConfig(
        n_actions=2,
        q_step_size=float(overrides["q_step_size"]),
        average_reward_step_size=float(overrides["average_reward_step_size"]),
        trace_decay=float(overrides["trace_decay"]),
        epsilon_start=float(overrides["epsilon_start"]),
        epsilon_end=float(overrides["epsilon_end"]),
        epsilon_decay_steps=int(overrides["epsilon_decay_steps"]),
        use_bias=bool(overrides["use_bias"]),
    )
    env = RiverSwimMDP(RiverSwimConfig())
    agent = DifferentialSARSAAgent(config)
    key = jr.PRNGKey(seed)
    key, env_key, agent_key = jr.split(key, 3)
    env_state = env.init(env_key)
    observation = env.observe(env_state)
    learner, action = agent.start(agent.init(env.n_states, agent_key), observation)

    def body(carry, _):  # type: ignore[no-untyped-def]
        current_env, current_learner, current_action, current_key = carry
        current_key, step_key = jr.split(current_key)
        next_observation, reward, current_env = env.step(
            current_env,
            current_action,
            step_key,
        )
        result = agent.update(current_learner, reward, next_observation)
        visit = (current_env.state_index == env.n_states - 1).astype(jnp.float32)
        updated = (current_env, result.state, result.action, current_key)
        return updated, (visit, reward)

    _final, (visits, rewards) = jax.lax.scan(
        body,
        (env_state, learner, action, key),
        None,
        length=horizon,
    )
    return int(jnp.sum(visits)), float(jnp.sum(rewards))
