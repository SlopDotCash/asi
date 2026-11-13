"""Step 10, 11, and 12 facades must preserve base learner host clocks."""

import time

import jax
import jax.numpy as jnp
import jax.random as jr

from alberta_framework.steps.step10 import (
    init_step10_state,
    init_step10_state_and_action,
    make_step10_stomp_agent,
    run_step10_scan,
    step10_update,
)
from alberta_framework.steps.step11 import (
    Step11OaKConfig,
    init_step11_state,
    make_step11_oak_agent,
    run_step11_scan,
    step11_update,
)
from alberta_framework.steps.step12 import (
    Step12IAConfig,
    init_step12_state,
    make_step12_ia_agent,
    run_step12_scan,
    step12_update,
)

_BIRTH = 1_700_000_000.125
_UPTIME = 16_777_216.25


def test_step10_facades_preserve_host_clocks() -> None:
    agent = make_step10_stomp_agent()
    obs_dim = agent.config.observation_dim
    t0 = time.time()
    started = init_step10_state_and_action(
        agent,
        key=jr.key(0),
        initial_observation=jnp.zeros(obs_dim, dtype=jnp.float32),
    )
    t1 = time.time()
    learner = started.state.base_learner_state
    assert type(learner.birth_timestamp) is float
    assert type(learner.uptime_s) is float
    assert t0 <= learner.birth_timestamp <= t1
    assert learner.uptime_s == 0.0

    state = init_step10_state(
        agent,
        key=jr.key(0),
        initial_observation=jnp.zeros(obs_dim, dtype=jnp.float32),
    )
    assert type(state.base_learner_state.birth_timestamp) is float
    assert type(state.base_learner_state.uptime_s) is float

    state = state.replace(  # type: ignore[attr-defined]
        base_learner_state=state.base_learner_state.replace(  # type: ignore[attr-defined]
            birth_timestamp=_BIRTH,
            uptime_s=_UPTIME,
        )
    )
    upd = step10_update(
        agent,
        state,
        jnp.float32(1.0),
        jnp.ones(obs_dim, dtype=jnp.float32),
    )
    jax.block_until_ready(upd)
    assert type(upd.state.base_learner_state.birth_timestamp) is float
    assert type(upd.state.base_learner_state.uptime_s) is float
    assert upd.state.base_learner_state.birth_timestamp == _BIRTH
    assert upd.state.base_learner_state.uptime_s == _UPTIME

    scan_res = run_step10_scan(
        agent,
        state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.ones((2, obs_dim), dtype=jnp.float32),
    )
    jax.block_until_ready(scan_res)
    assert type(scan_res.state.base_learner_state.birth_timestamp) is float
    assert type(scan_res.state.base_learner_state.uptime_s) is float
    assert scan_res.state.base_learner_state.birth_timestamp == _BIRTH
    assert scan_res.state.base_learner_state.uptime_s == _UPTIME


def test_step11_facades_preserve_host_clocks() -> None:
    agent = make_step11_oak_agent(
        Step11OaKConfig(observation_dim=2, n_primitive_actions=2)
    )
    t0 = time.time()
    state = init_step11_state(
        agent,
        key=jr.key(0),
        initial_observation=jnp.zeros(2, dtype=jnp.float32),
    )
    t1 = time.time()
    learner = state.stomp_state.base_learner_state
    assert type(learner.birth_timestamp) is float
    assert type(learner.uptime_s) is float
    assert t0 <= learner.birth_timestamp <= t1
    assert learner.uptime_s == 0.0

    state = state.replace(  # type: ignore[attr-defined]
        stomp_state=state.stomp_state.replace(  # type: ignore[attr-defined]
            base_learner_state=state.stomp_state.base_learner_state.replace(  # type: ignore[attr-defined]
                birth_timestamp=_BIRTH,
                uptime_s=_UPTIME,
            )
        )
    )
    upd = step11_update(
        agent,
        state,
        jnp.float32(1.0),
        jnp.ones(2, dtype=jnp.float32),
    )
    jax.block_until_ready(upd)
    upd_learner = upd.state.stomp_state.base_learner_state
    assert type(upd_learner.birth_timestamp) is float
    assert type(upd_learner.uptime_s) is float
    assert upd_learner.birth_timestamp == _BIRTH
    assert upd_learner.uptime_s == _UPTIME

    scan_res = run_step11_scan(
        agent,
        state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.ones((2, 2), dtype=jnp.float32),
    )
    jax.block_until_ready(scan_res)
    scan_learner = scan_res.state.stomp_state.base_learner_state
    assert type(scan_learner.birth_timestamp) is float
    assert type(scan_learner.uptime_s) is float
    assert scan_learner.birth_timestamp == _BIRTH
    assert scan_learner.uptime_s == _UPTIME


def test_step12_facades_preserve_host_clocks() -> None:
    agent = make_step12_ia_agent(
Step12IAConfig(observation_dim=2, n_primitive_actions=2)
    )
    state = init_step12_state(
        agent,
        key=jr.key(0),
        initial_observation=jnp.zeros(2, dtype=jnp.float32),
    )
    state = state.replace(  # type: ignore[attr-defined]
        cortex_state=state.cortex_state.replace(  # type: ignore[attr-defined]
            stomp_state=state.cortex_state.stomp_state.replace(  # type: ignore[attr-defined]
                base_learner_state=state.cortex_state.stomp_state.base_learner_state.replace(  # type: ignore[attr-defined]
                    birth_timestamp=_BIRTH,
                    uptime_s=_UPTIME,
                )
            )
        )
    )
    upd = step12_update(
        agent,
        state,
        jnp.zeros(2, dtype=jnp.float32),
        jnp.float32(1.0),
        jnp.ones(2, dtype=jnp.float32),
    )
    jax.block_until_ready(upd)
    upd_learner = upd.state.cortex_state.stomp_state.base_learner_state
    assert type(upd_learner.birth_timestamp) is float
    assert type(upd_learner.uptime_s) is float
    assert upd_learner.birth_timestamp == _BIRTH
    assert upd_learner.uptime_s == _UPTIME

    scan_res = run_step12_scan(
        agent,
        state,
        jnp.zeros((2, 2), dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32),
        jnp.ones((2, 2), dtype=jnp.float32),
    )
    jax.block_until_ready(scan_res)
    scan_learner = scan_res.state.cortex_state.stomp_state.base_learner_state
    assert type(scan_learner.birth_timestamp) is float
    assert type(scan_learner.uptime_s) is float
    assert scan_learner.birth_timestamp == _BIRTH
    assert scan_learner.uptime_s == _UPTIME
