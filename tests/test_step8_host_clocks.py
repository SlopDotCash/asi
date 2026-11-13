"""The host Step 8 update and scan must retain incoming lifecycle metadata."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.steps.step8 import (
    Step8WorldModelConfig,
    init_step8_state,
    make_step8_world_model,
    run_step8_scan,
    step8_update,
)


@pytest.mark.parametrize("obs_value", [1.0, float("nan")], ids=["finite", "rejected"])
def test_step8_update_retains_host_metadata(obs_value: float) -> None:
    model = make_step8_world_model(
        Step8WorldModelConfig(observation_dim=2, n_actions=2, hidden_sizes=(), sparsity=0.0)
    )
    state = init_step8_state(model, key=jr.key(0))
    birth, uptime = 1_700_000_000.125, 16_777_216.25
    state = state.replace(  # type: ignore[attr-defined]
        learner_state=state.learner_state.replace(  # type: ignore[attr-defined]
            birth_timestamp=birth,
            uptime_s=uptime,
        )
    )
    for iteration in (1, 2):
        result = step8_update(
            model,
            state,
            jnp.full(2, obs_value, dtype=jnp.float32),
            jnp.int32(0),
            jnp.float32(1.0),
            jnp.ones(2, dtype=jnp.float32),
        )
        jax.block_until_ready(result)
        metadata = result.state.learner_state
        assert float(metadata.birth_timestamp) == birth
        assert float(metadata.uptime_s) == uptime
        assert type(metadata.birth_timestamp) is float
        assert type(metadata.uptime_s) is float
        assert result.prediction.next_observation.shape == (2,)
        assert int(result.state.step_count) == (iteration if obs_value == 1.0 else 0)
        state = result.state


@pytest.mark.parametrize("obs_value", [1.0, float("nan")], ids=["finite", "rejected"])
def test_step8_scan_retains_host_metadata(obs_value: float) -> None:
    model = make_step8_world_model(
        Step8WorldModelConfig(observation_dim=2, n_actions=2, hidden_sizes=(), sparsity=0.0)
    )
    state = init_step8_state(model, key=jr.key(0))
    birth, uptime = 1_700_000_000.125, 16_777_216.25
    state = state.replace(  # type: ignore[attr-defined]
        learner_state=state.learner_state.replace(  # type: ignore[attr-defined]
            birth_timestamp=birth,
            uptime_s=uptime,
        )
    )
    result = run_step8_scan(
        model,
        state,
        jnp.full((2, 2), obs_value, dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.int32),
        jnp.ones(2, dtype=jnp.float32),
        jnp.ones((2, 2), dtype=jnp.float32),
    )
    jax.block_until_ready(result)
    metadata = result.state.learner_state
    assert float(metadata.birth_timestamp) == birth
    assert float(metadata.uptime_s) == uptime
    assert type(metadata.birth_timestamp) is float
    assert type(metadata.uptime_s) is float
