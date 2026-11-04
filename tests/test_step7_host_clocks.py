"""Step 7 facades preserve the host-only lifecycle clocks exactly.

The control and model updates are jitted, so ``birth_timestamp`` and
``uptime_s`` come back narrowed to float32 unless the facade restores them.
Near 1.79e9 a float32 birth time is off by up to 64 seconds, and the Step 7
scan previously restored only the world-model clocks, leaving the control
birth narrowed and its uptime never advanced.
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
from alberta_framework.steps.step7 import (
    Step7DynaConfig,
    init_step7_state,
    make_step7_components,
    run_step7_scan,
    step7_update,
)
from alberta_framework.steps.step8 import Step8WorldModelConfig

_BIRTH = 1793800123.456
_UPTIME = 7.123456


def _cfg() -> Step7DynaConfig:
    return Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=2),
        world_model=Step8WorldModelConfig(observation_dim=2, n_actions=2),
        planning_steps=1,
        planning_warmup_steps=0,
        planning_memory_size=16,
    )


def _state_with_clocks(cfg, agent, model):
    state = init_step7_state(
        agent,
        model,
        key=jr.key(0),
        initial_observation=jnp.zeros(2),
        memory_size=cfg.planning_memory_size,
    )
    control = state.control_state.replace(
        birth_timestamp=_BIRTH, uptime_s=_UPTIME
    )
    learner = state.world_model_state.learner_state.replace(
        birth_timestamp=_BIRTH, uptime_s=_UPTIME
    )
    world = state.world_model_state.replace(learner_state=learner)
    return state.replace(control_state=control, world_model_state=world)


def test_init_step7_state_keeps_exact_control_clocks() -> None:
    cfg = _cfg()
    agent, model = make_step7_components(cfg)
    t0 = time.time()
    state = init_step7_state(
        agent,
        model,
        key=jr.key(0),
        initial_observation=jnp.zeros(2),
        memory_size=cfg.planning_memory_size,
    )
    t1 = time.time()
    assert type(state.control_state.birth_timestamp) is float
    assert type(state.control_state.uptime_s) is float
    assert t0 <= state.control_state.birth_timestamp <= t1
    assert state.control_state.uptime_s == 0.0


def test_step7_update_restores_exact_host_clocks() -> None:
    cfg = _cfg()
    agent, model = make_step7_components(cfg)
    state = _state_with_clocks(cfg, agent, model)
    result = step7_update(
        cfg,
        agent,
        model,
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.1, 0.2], dtype=jnp.float32),
    )
    control = result.state.control_state
    assert type(control.birth_timestamp) is float
    assert control.birth_timestamp == _BIRTH
    assert control.uptime_s == _UPTIME
    learner = result.state.world_model_state.learner_state
    assert learner.birth_timestamp == _BIRTH
    assert learner.uptime_s == _UPTIME


def test_step7_scan_restores_control_clocks_exactly() -> None:
    cfg = _cfg()
    agent, model = make_step7_components(cfg)
    state = _state_with_clocks(cfg, agent, model)
    result = run_step7_scan(
        cfg,
        agent,
        model,
        state,
        jnp.array([1.0, 1.0], dtype=jnp.float32),
        jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32),
    )
    control = result.state.control_state
    assert type(control.birth_timestamp) is float
    assert control.birth_timestamp == _BIRTH
    assert control.uptime_s == _UPTIME
    learner = result.state.world_model_state.learner_state
    assert learner.birth_timestamp == _BIRTH
    assert learner.uptime_s > _UPTIME
