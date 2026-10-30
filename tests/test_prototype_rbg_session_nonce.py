"""PrototypeAgent session nonces must stay two words for every supported key."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeTransition,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)

pytestmark = pytest.mark.unit

OBS_DIM = 4


def _agent() -> PrototypeAgent:
    return PrototypeAgent(
        PrototypeAgentConfig(
            oak=OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=(
                        SubtaskSpec(
                            feature_index=0,
                            threshold=0.5,
                            pseudo_reward_scale=1.0,
                            max_option_steps=8,
                        ),
                    ),
                    observation_dim=OBS_DIM,
                    n_primitive_actions=2,
                )
            )
        )
    )


def _transition(state: PrototypeAgentState) -> PrototypeTransition:
    next_observation = jnp.asarray([1.0, 0.0, 0.5, -0.5], dtype=jnp.float32)
    return PrototypeTransition(
        observation=state.current_raw_observation,
        action=state.current_action,
        decision_id=state.current_decision_id,
        reward=jnp.asarray(1.0, dtype=jnp.float32),
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        terminated=jnp.asarray(False),
        truncated=jnp.asarray(False),
        next_observation=next_observation,
        next_decision_observation=next_observation,
    )


def _materialize_typed_keys(tree: Any) -> Any:
    def convert(value: Any) -> Any:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)
        return value

    return jax.tree.map(convert, tree)


def _explicit_rbg_key() -> jax.Array:
    return jr.key(0, impl="rbg")


def _ambient_rbg_key() -> jax.Array:
    with jax.default_prng_impl("rbg"):
        return jr.key(0)


@pytest.mark.parametrize(
    "make_key",
    [_explicit_rbg_key, _ambient_rbg_key],
    ids=["explicit-rbg", "ambient-rbg-default"],
)
def test_rbg_key_init_derives_a_two_word_session_nonce(
    make_key: Callable[[], jax.Array],
) -> None:
    agent = _agent()
    key = make_key()
    assert str(jr.key_impl(key)) == "rbg"

    state = agent.init(key)

    assert state.current_decision_id.shape == (4,)
    assert state.current_decision_id.dtype == jnp.uint32
    np.testing.assert_array_equal(state.current_decision_id[2:], np.zeros(2))
    started = agent.start(state, jnp.zeros((OBS_DIM,), dtype=jnp.float32))
    assert bool(started.started)
    result = agent.update_transition(started, _transition(started))
    assert bool(result.transition_diagnostics.valid)


def test_threefry_session_nonce_is_the_session_key_data() -> None:
    key = jr.key(7)
    state = _agent().init(key)

    session_key = jr.split(key, 7)[0]
    np.testing.assert_array_equal(
        state.current_decision_id[:2],
        jr.key_data(session_key),
    )


@pytest.mark.parametrize(
    "lifecycle_id",
    [None, jnp.asarray([1, 2], dtype=jnp.uint32)],
    ids=["derived-nonce", "caller-nonce"],
)
def test_rbg_checkpoint_restores_with_the_default_template(
    tmp_path: Path,
    lifecycle_id: jax.Array | None,
) -> None:
    agent = _agent()
    state = agent.start(
        agent.init(_explicit_rbg_key(), lifecycle_id=lifecycle_id),
        jnp.zeros((OBS_DIM,), dtype=jnp.float32),
    )
    state = agent.update_transition(state, _transition(state)).state

    checkpoint_path = tmp_path / "prototype-rbg"
    save_prototype_checkpoint(agent, state, checkpoint_path)
    restored_agent, restored_state = load_prototype_checkpoint(checkpoint_path)

    chex.assert_trees_all_equal(
        _materialize_typed_keys(restored_state),
        _materialize_typed_keys(state),
    )
    uninterrupted = agent.update_transition(state, _transition(state))
    resumed = restored_agent.update_transition(
        restored_state,
        _transition(restored_state),
    )
    assert bool(resumed.transition_diagnostics.valid)
    chex.assert_trees_all_equal(
        _materialize_typed_keys(resumed.state),
        _materialize_typed_keys(uninterrupted.state),
    )
