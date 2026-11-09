"""Learning-loop scans must keep exact host lifecycle clocks.

``birth_timestamp`` and ``uptime_s`` are host-only Python floats. A
``jax.lax.scan`` carries them through the traced update and narrows them to
float32, which cannot represent ``1700000000.125`` or an uptime of
``2**24 + 0.25``. Each loop restores the incoming Python values and adds
that scan's elapsed time to the original uptime.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework import (
    LMS,
    DemonType,
    EMANormalizer,
    GVFSpec,
    LinearLearner,
    MLPLearner,
    NormalizerTrackingConfig,
    OffPolicyHordeLearner,
    RandomWalkStream,
    SARSAAgent,
    SARSAConfig,
    StepSizeTrackingConfig,
    TDLinearLearner,
    TDTimeStep,
    TrueOnlineTDLearner,
    create_horde_spec,
    run_learning_loop,
    run_learning_loop_batched,
    run_mlp_learning_loop,
    run_mlp_learning_loop_batched,
    run_off_policy_horde_learning_loop,
    run_sarsa_from_arrays,
    run_sarsa_from_arrays_final_state,
    run_td_learning_loop,
    run_true_online_td_loop,
)

_BIRTH = 1700000000.125
_UPTIME = float(2**24) + 0.25


class _SimpleTDStream:
    """Deterministic TD stream small enough for a clock round-trip."""

    def __init__(self, feature_dim: int = 3) -> None:
        self.feature_dim = feature_dim

    def init(self, key: Any) -> dict[str, Any]:
        return {"key": key}

    def step(self, state: dict[str, Any], idx: Any) -> tuple[TDTimeStep, dict[str, Any]]:
        del idx
        key, obs_key, next_key, reward_key = jr.split(state["key"], 4)
        timestep = TDTimeStep(
            observation=jr.normal(obs_key, (self.feature_dim,), dtype=jnp.float32),
            reward=jr.normal(reward_key, (), dtype=jnp.float32),
            next_observation=jr.normal(next_key, (self.feature_dim,), dtype=jnp.float32),
            gamma=jnp.asarray(0.9, dtype=jnp.float32),
        )
        return timestep, {"key": key}


def _assert_exact_clocks(final_state: Any, *, steps: int) -> None:
    assert type(final_state.birth_timestamp) is float
    assert final_state.birth_timestamp == _BIRTH
    assert type(final_state.uptime_s) is float
    assert final_state.uptime_s > _UPTIME
    assert int(final_state.step_count) == steps


def test_linear_learning_loop_preserves_host_clocks() -> None:
    learner = LinearLearner()
    state = learner.init(3).replace(birth_timestamp=_BIRTH, uptime_s=_UPTIME)
    final, _metrics = run_learning_loop(
        learner,
        RandomWalkStream(feature_dim=3),
        num_steps=4,
        key=jr.key(0),
        learner_state=state,
    )
    _assert_exact_clocks(final, steps=4)


def test_linear_learning_loop_tracking_preserves_host_clocks() -> None:
    learner = LinearLearner()
    state = learner.init(3).replace(birth_timestamp=_BIRTH, uptime_s=_UPTIME)
    final, _metrics, _history = run_learning_loop(
        learner,
        RandomWalkStream(feature_dim=3),
        num_steps=4,
        key=jr.key(1),
        learner_state=state,
        step_size_tracking=StepSizeTrackingConfig(interval=2),
    )
    _assert_exact_clocks(final, steps=4)


def test_mlp_learning_loop_preserves_host_clocks() -> None:
    learner = MLPLearner(hidden_sizes=(4,), sparsity=0.0, use_layer_norm=False)
    state = learner.init(3, jr.key(2)).replace(birth_timestamp=_BIRTH, uptime_s=_UPTIME)
    final, _metrics = run_mlp_learning_loop(
        learner,
        RandomWalkStream(feature_dim=3),
        num_steps=4,
        key=jr.key(3),
        learner_state=state,
    )
    _assert_exact_clocks(final, steps=4)


def test_mlp_learning_loop_tracking_preserves_host_clocks() -> None:
    learner = MLPLearner(
        hidden_sizes=(4,),
        sparsity=0.0,
        use_layer_norm=False,
        normalizer=EMANormalizer(),
    )
    state = learner.init(3, jr.key(4)).replace(birth_timestamp=_BIRTH, uptime_s=_UPTIME)
    final, _metrics, _history = run_mlp_learning_loop(
        learner,
        RandomWalkStream(feature_dim=3),
        num_steps=4,
        key=jr.key(5),
        learner_state=state,
        normalizer_tracking=NormalizerTrackingConfig(interval=2),
    )
    _assert_exact_clocks(final, steps=4)


def test_learning_loop_with_jitted_learner_preserves_host_clocks() -> None:
    learner = MLPLearner(hidden_sizes=(4,), sparsity=0.0, use_layer_norm=False)
    state = learner.init(3, jr.key(10)).replace(birth_timestamp=_BIRTH, uptime_s=_UPTIME)
    final, _metrics = run_learning_loop(
        learner,
        RandomWalkStream(feature_dim=3),
        num_steps=4,
        key=jr.key(11),
        learner_state=state,
    )
    _assert_exact_clocks(final, steps=4)


class _JittedLinearLearner(LinearLearner):
    """Exercise the linear tracking interface with clocks crossing a JIT boundary."""

    update = jax.jit(LinearLearner.update, static_argnums=(0,))


@pytest.mark.parametrize("tracking", [False, True])
def test_learning_loop_with_jitted_linear_learner_preserves_host_clocks(tracking: bool) -> None:
    learner = _JittedLinearLearner(normalizer=EMANormalizer())
    state = learner.init(3).replace(birth_timestamp=_BIRTH, uptime_s=_UPTIME)
    result = run_learning_loop(
        learner,
        RandomWalkStream(feature_dim=3),
        num_steps=4,
        key=jr.key(13),
        learner_state=state,
        step_size_tracking=StepSizeTrackingConfig(interval=2) if tracking else None,
        normalizer_tracking=NormalizerTrackingConfig(interval=2) if tracking else None,
    )
    _assert_exact_clocks(result[0], steps=4)


def test_td_learning_loop_preserves_host_clocks() -> None:
    learner = TDLinearLearner()
    state = learner.init(3).replace(birth_timestamp=_BIRTH, uptime_s=_UPTIME)
    final, _metrics = run_td_learning_loop(
        learner,
        _SimpleTDStream(),
        num_steps=4,
        key=jr.key(6),
        learner_state=state,
    )
    _assert_exact_clocks(final, steps=4)


def test_td_learning_loop_with_true_online_td_preserves_host_clocks() -> None:
    learner = TrueOnlineTDLearner()
    state = learner.init(3).replace(birth_timestamp=_BIRTH, uptime_s=_UPTIME)
    final, _metrics = run_td_learning_loop(
        learner,
        _SimpleTDStream(),
        num_steps=4,
        key=jr.key(12),
        learner_state=state,
    )
    _assert_exact_clocks(final, steps=4)


def test_true_online_td_loop_preserves_host_clocks() -> None:
    learner = TrueOnlineTDLearner()
    state = learner.init(3).replace(birth_timestamp=_BIRTH, uptime_s=_UPTIME)
    final, _metrics = run_true_online_td_loop(
        learner,
        _SimpleTDStream(),
        num_steps=4,
        key=jr.key(7),
        learner_state=state,
    )
    _assert_exact_clocks(final, steps=4)


@pytest.mark.parametrize("final_state_only", [False, True])
def test_sarsa_array_runners_preserve_host_clocks(final_state_only: bool) -> None:
    agent = SARSAAgent(
        sarsa_config=SARSAConfig(n_actions=2, epsilon_decay_steps=0),  # type: ignore[call-arg]
        hidden_sizes=(4,),
        sparsity=0.0,
    )
    observations = jnp.ones((4, 3), dtype=jnp.float32)
    rewards = jnp.zeros((4,), dtype=jnp.float32)
    terminated = jnp.zeros((4,), dtype=jnp.float32)
    state = agent.init(feature_dim=3, key=jr.key(8))
    action, key = agent.select_action(state, observations[0])
    state = state.replace(  # type: ignore[attr-defined]
        last_action=action,
        last_observation=observations[0],
        rng_key=key,
        learner_state=state.learner_state.replace(  # type: ignore[attr-defined]
            birth_timestamp=_BIRTH, uptime_s=_UPTIME
        ),
    )
    if final_state_only:
        final = run_sarsa_from_arrays_final_state(
            agent, state, observations, rewards, terminated, observations
        )
    else:
        final = run_sarsa_from_arrays(
            agent, state, observations, rewards, terminated, observations
        ).state
    _assert_exact_clocks(final.learner_state, steps=4)


def test_off_policy_horde_learning_loop_preserves_host_clocks() -> None:
    spec = create_horde_spec(
        tuple(
            GVFSpec(
                name=f"demon_{i}",
                demon_type=DemonType.PREDICTION,
                gamma=gamma,
                lamda=0.0,
                cumulant_index=i,
            )  # type: ignore[call-arg]
            for i, gamma in enumerate((0.0, 0.5))
        )
    )
    learner = OffPolicyHordeLearner(spec, hidden_sizes=(4,), optimizer=LMS(step_size=0.01))
    state = learner.init(3, jr.key(9)).replace(  # type: ignore[attr-defined]
        birth_timestamp=_BIRTH, uptime_s=_UPTIME
    )
    observations = jnp.ones((4, 3), dtype=jnp.float32)
    cumulants = jnp.ones((4, 2), dtype=jnp.float32)
    rhos = jnp.ones((4, 2), dtype=jnp.float32)
    result = run_off_policy_horde_learning_loop(
        learner, state, observations, cumulants, observations, rhos
    )
    _assert_exact_clocks(result.state, steps=4)


def test_float32_narrowing_drops_the_planted_clocks() -> None:
    """Document the representation the scan would keep without a restore."""
    narrowed_birth = float(jnp.asarray(_BIRTH, dtype=jnp.float32))
    narrowed_uptime = float(jnp.asarray(_UPTIME, dtype=jnp.float32))
    assert narrowed_birth != _BIRTH
    assert narrowed_uptime != _UPTIME
    assert narrowed_uptime < _UPTIME


def test_linear_learning_loop_batched_preserves_host_clocks() -> None:
    learner = LinearLearner()
    state = learner.init(3).replace(birth_timestamp=_BIRTH, uptime_s=_UPTIME)
    stream = RandomWalkStream(feature_dim=3)
    keys = jr.split(jr.key(10), 2)
    result = run_learning_loop_batched(
        learner, stream, 4, keys, learner_state=state
    )
    assert type(result.states.birth_timestamp) is float
    assert result.states.birth_timestamp == _BIRTH
    assert type(result.states.uptime_s) is float
    assert result.states.uptime_s > _UPTIME


def test_mlp_learning_loop_batched_preserves_host_clocks() -> None:
    learner = MLPLearner(hidden_sizes=(4,))
    state = learner.init(3, key=jr.key(11)).replace(
        birth_timestamp=_BIRTH, uptime_s=_UPTIME
    )
    stream = RandomWalkStream(feature_dim=3)
    keys = jr.split(jr.key(12), 2)
    result = run_mlp_learning_loop_batched(
        learner, stream, 4, keys, learner_state=state
    )
    assert type(result.states.birth_timestamp) is float
    assert result.states.birth_timestamp == _BIRTH
    assert type(result.states.uptime_s) is float
    assert result.states.uptime_s > _UPTIME
