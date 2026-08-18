"""Saturating lifetime clock keeps Step7DynaState.step_count non-negative."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
from alberta_framework.steps.step7 import (
    Step7DynaConfig,
    Step7DynaState,
    init_step7_state,
    make_step7_components,
    step7_update,
)
from alberta_framework.steps.step8 import Step8WorldModelConfig

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def _cfg() -> Step7DynaConfig:
    return Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=2),
        world_model=Step8WorldModelConfig(observation_dim=2, n_actions=2),
        planning_steps=1,
        planning_warmup_steps=0,
        planning_memory_size=16,
    )


def test_bare_int32_increment_wraps_negative() -> None:
    wrap = jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    assert int(wrap) == _INT32_MIN


def test_step7_step_count_saturates_at_int32_max() -> None:
    cfg = _cfg()
    agent, model = make_step7_components(cfg)
    state = init_step7_state(
        agent,
        model,
        key=jr.key(0),
        initial_observation=jnp.zeros(2),
        memory_size=cfg.planning_memory_size,
    )
    near_max: Step7DynaState = state.replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )

    result = step7_update(
        cfg,
        agent,
        model,
        near_max,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.1, 0.2], dtype=jnp.float32),
    )

    assert int(result.state.step_count) == _INT32_MAX
    assert int(result.state.step_count) >= 0

    again = step7_update(
        cfg,
        agent,
        model,
        result.state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.1, 0.2], dtype=jnp.float32),
    )
    assert int(again.state.step_count) == _INT32_MAX
    assert int(again.state.step_count) >= 0


def test_step7_step_count_increments_below_max() -> None:
    cfg = _cfg()
    agent, model = make_step7_components(cfg)
    state = init_step7_state(
        agent,
        model,
        key=jr.key(1),
        initial_observation=jnp.zeros(2),
        memory_size=cfg.planning_memory_size,
    )
    result = step7_update(
        cfg,
        agent,
        model,
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.1, 0.2], dtype=jnp.float32),
    )
    assert int(result.state.step_count) == 1
