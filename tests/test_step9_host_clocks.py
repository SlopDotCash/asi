"""Step 9 facades preserve the host-only lifecycle clocks exactly.

Step 9 drives the same jitted differential SARSA control agent as Step 7,
and ``step9_update`` is itself jitted, so ``birth_timestamp`` and
``uptime_s`` come back narrowed to float32 unless the facade restores them.
Near 1.79e9 a float32 birth time is off by up to 64 seconds. The Step 9
scan previously restored only the world-model clocks, and neither
``init_step9_state`` nor ``step9_update`` restored any.
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.steps.step9 import (
    Step9DreamingConfig,
    init_step9_state,
    make_step9_components,
    run_step9_scan,
    step9_update,
)

_BIRTH = 1793800123.456
_UPTIME = 7.123456


def _cfg() -> Step9DreamingConfig:
    return Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=1,
        dreaming_warmup_steps=0,
    )


def _state_with_clocks(agent, model, buffer):
    state = init_step9_state(agent, model, buffer, key=jr.key(17), initial_observation=jnp.zeros(2))
    control = state.control_state.replace(birth_timestamp=_BIRTH, uptime_s=_UPTIME)
    learner = state.world_model_state.learner_state.replace(
        birth_timestamp=_BIRTH, uptime_s=_UPTIME
    )
    world = state.world_model_state.replace(learner_state=learner)
    return state.replace(control_state=control, world_model_state=world)


def test_init_step9_state_keeps_exact_control_clocks() -> None:
    agent, model, buffer = make_step9_components(_cfg())
    t0 = time.time()
    state = init_step9_state(agent, model, buffer, key=jr.key(17), initial_observation=jnp.zeros(2))
    t1 = time.time()
    assert type(state.control_state.birth_timestamp) is float
    assert type(state.control_state.uptime_s) is float
    assert t0 <= state.control_state.birth_timestamp <= t1
    assert state.control_state.uptime_s == 0.0


def test_step9_update_restores_exact_host_clocks() -> None:
    cfg = _cfg()
    agent, model, buffer = make_step9_components(cfg)
    state = _state_with_clocks(agent, model, buffer)
    result = step9_update(
        cfg,
        agent,
        model,
        buffer,
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.1, 0.2], dtype=jnp.float32),
    )
    control = result.state.control_state
    assert type(control.birth_timestamp) is float
    assert control.birth_timestamp == _BIRTH
    assert control.uptime_s == _UPTIME
    learner = result.state.world_model_state.learner_state
    assert type(learner.birth_timestamp) is float
    assert learner.birth_timestamp == _BIRTH
    assert learner.uptime_s == _UPTIME


def test_step9_scan_restores_control_clocks_exactly() -> None:
    cfg = _cfg()
    agent, model, buffer = make_step9_components(cfg)
    state = _state_with_clocks(agent, model, buffer)
    result = run_step9_scan(
        cfg,
        agent,
        model,
        buffer,
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
