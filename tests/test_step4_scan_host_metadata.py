"""The host Step 4 scan and update must retain incoming lifecycle metadata."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.steps.step4 import (
    Step4SARSAConfig,
    init_step4_state,
    make_step4_sarsa_agent,
    run_step4_scan,
    step4_update,
)


@pytest.mark.parametrize("reward", [1.0, float("nan")], ids=["finite", "rejected"])
def test_step4_scan_retains_host_metadata(reward: float) -> None:
    agent = make_step4_sarsa_agent(Step4SARSAConfig(n_actions=2, hidden_sizes=()))
    state = init_step4_state(
        agent, feature_dim=2, key=jr.key(0), initial_features=jnp.ones(2)
    )
    birth, uptime = 1_700_000_000.125, 16_777_216.25
    state = state.replace(
        learner_state=state.learner_state.replace(birth_timestamp=birth, uptime_s=uptime)
    )
    for iteration in (1, 2):
        result = run_step4_scan(
            agent, state, jnp.ones((2, 2)), jnp.full((2,), reward), jnp.zeros(2)
        )
        jax.block_until_ready(result)
        metadata = result.state.learner_state
        print("HOST_METADATA", reward, iteration, metadata.birth_timestamp, metadata.uptime_s)
        assert float(metadata.birth_timestamp) == birth
        assert float(metadata.uptime_s) == uptime
        assert type(metadata.birth_timestamp) is float
        assert type(metadata.uptime_s) is float
        assert result.q_values.shape == (2, 2)
        assert result.actions.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(result.q_values)))
        assert int(metadata.step_count) == (2 * iteration if reward == 1.0 else 0)
        state = result.state


@pytest.mark.parametrize("reward", [1.0, float("nan")], ids=["finite", "rejected"])
def test_step4_update_retains_host_metadata(reward: float) -> None:
    agent = make_step4_sarsa_agent(Step4SARSAConfig(n_actions=2, hidden_sizes=()))
    state = init_step4_state(agent, feature_dim=2, key=jr.key(0), initial_features=jnp.ones(2))
    birth, uptime = 1_700_000_000.125, 16_777_216.25
    state = state.replace(
        learner_state=state.learner_state.replace(birth_timestamp=birth, uptime_s=uptime)
    )
    for iteration in (1, 2):
        result = step4_update(agent, state, jnp.float32(reward), jnp.ones(2), jnp.float32(0.0))
        jax.block_until_ready(result)
        metadata = result.state.learner_state
        print("HOST_METADATA", reward, iteration, metadata.birth_timestamp, metadata.uptime_s)
        assert float(metadata.birth_timestamp) == birth
        assert float(metadata.uptime_s) == uptime
        assert type(metadata.birth_timestamp) is float
        assert type(metadata.uptime_s) is float
        assert result.q_values.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(result.q_values)))
        assert int(metadata.step_count) == (iteration if reward == 1.0 else 0)
        state = result.state
