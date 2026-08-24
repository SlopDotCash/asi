"""Dream backups must not mutate real OaK option lifecycle state."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core import multi_head_learner
from alberta_framework.core.dreaming import DreamingConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    _guarded_dream_rng_root,
    _sample_one_hot_dream_observation,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

OBS = jnp.array([1.0, -0.5], dtype=jnp.float32)
N_DREAMS = 3


def test_guarded_dream_rng_root_is_disjoint_from_future_real_stream() -> None:
    stored_real_key = jr.key(5474)
    dream_root = _guarded_dream_rng_root(stored_real_key)

    assert not bool(jnp.array_equal(dream_root, stored_real_key))
    dream_keys = []
    dream_key = dream_root
    for _ in range(N_DREAMS):
        dream_key, sample_key, action_key = jr.split(dream_key, 3)
        dream_keys.extend((sample_key, action_key))

    future_real_key = stored_real_key
    for _ in range(N_DREAMS + 1):
        future_real_key, explore_key, noise_key = jr.split(future_real_key, 3)
        assert not bool(jnp.array_equal(dream_root, future_real_key))
        for dream_key in dream_keys:
            assert not bool(jnp.array_equal(dream_key, explore_key))
            assert not bool(jnp.array_equal(dream_key, noise_key))

    chex.assert_trees_all_equal(
        dream_root,
        _guarded_dream_rng_root(stored_real_key),
    )


def _dream_config(
    *,
    mode: Literal["model_prediction", "sample_one_hot"] = "model_prediction",
    predict_delta: bool = True,
) -> PrototypeAgentConfig:
    return PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(
                    SubtaskSpec(
                        feature_index=0,
                        threshold=1.0e6,
                        max_option_steps=8,
                    ),
                ),
                observation_dim=2,
                n_primitive_actions=2,
            )
        ),
        world_model=ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            step_size=0.1,
            predict_delta=predict_delta,
        ),
        dreaming=DreamingConfig(
            warmup_steps=0,
            max_model_error_ema=1.0e6,
        ),
        buffer_capacity=4,
        n_dreams_per_step=N_DREAMS,
        dream_next_observation_mode=mode,
    )


def _zero_world_model_heads(state: Any) -> Any:
    learner = state.world_model_state.learner_state
    zero_learner = learner.replace(
        head_params=learner.head_params.replace(
            weights=tuple(
                jnp.zeros_like(weight)
                for weight in learner.head_params.weights
            ),
            biases=tuple(
                jnp.zeros_like(bias)
                for bias in learner.head_params.biases
            ),
        )
    )
    return state.world_model_state.replace(learner_state=zero_learner)


class _KeySensitiveBuffer:
    """Expose the sampled anchor index through a one-hot observation."""

    def sample(self, _state: Any, key: Any) -> tuple[Any, Any]:
        index = jr.randint(key, (), 0, 2, dtype=jnp.int32)
        return jax.nn.one_hot(index, 2, dtype=jnp.float32), index


class _AnchorActionDreamer:
    """Encode both legacy random choices in the returned TD diagnostic."""

    def propose(
        self,
        _model: Any,
        _model_state: Any,
        observation: Any,
        action: Any,
    ) -> Any:
        reward = (
            1.0
            + 2.0 * jnp.argmax(observation).astype(jnp.float32)
            + action.astype(jnp.float32)
        )
        transition = SimpleNamespace(
            reward=reward,
            discount=jnp.array(0.0, dtype=jnp.float32),
            next_observation=jnp.array([1.0, 0.0], dtype=jnp.float32),
        )
        return SimpleNamespace(
            transition=transition,
            accepted=jnp.array(True),
        )


class _DiagnosticOak:
    """Return the encoded reward while leaving the learner unchanged."""

    def update(
        self,
        state: Any,
        reward: Any,
        _next_observation: Any,
        _discount: Any,
        *,
        enable_option_planning: bool,
    ) -> Any:
        del enable_option_planning
        return SimpleNamespace(state=state, td_error=reward)


def test_sample_one_hot_projection_has_exact_support_and_expected_frequency() -> None:
    prediction = jnp.array([0.0, 0.25, 0.75, 0.0], dtype=jnp.float32)
    keys = jr.split(jr.key(91), 4096)
    samples, valid = jax.vmap(
        _sample_one_hot_dream_observation,
        in_axes=(None, 0),
    )(prediction, keys)

    chex.assert_trees_all_equal(valid, jnp.ones((4096,), dtype=jnp.bool_))
    chex.assert_trees_all_equal(
        jnp.sum(samples, axis=1),
        jnp.ones((4096,), dtype=jnp.float32),
    )
    assert not bool(jnp.any(samples[:, 0]))
    assert not bool(jnp.any(samples[:, 3]))
    assert float(jnp.mean(samples[:, 2])) == pytest.approx(0.75, abs=0.03)


def test_sample_one_hot_projection_clips_finite_coordinates() -> None:
    sample, valid = _sample_one_hot_dream_observation(
        jnp.array([-3.0, 2.0, 0.0], dtype=jnp.float32),
        jr.key(3),
    )

    assert bool(valid)
    chex.assert_trees_all_equal(
        sample,
        jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32),
    )


@pytest.mark.parametrize(
    "prediction",
    [
        jnp.zeros((0,), dtype=jnp.float32),
        jnp.zeros((2, 2), dtype=jnp.float32),
    ],
)
def test_sample_one_hot_projection_rejects_non_vector_shapes(
    prediction: Any,
) -> None:
    with pytest.raises(ValueError, match="non-empty vector"):
        _sample_one_hot_dream_observation(prediction, jr.key(3))


@pytest.mark.parametrize(
    "prediction",
    [
        [0.0, 0.0],
        [-1.0, -2.0],
        [float("nan"), 1.0],
        [float("inf"), 1.0],
        [float("-inf"), 1.0],
    ],
)
def test_sample_one_hot_projection_fails_closed(
    prediction: list[float],
) -> None:
    sample, valid = _sample_one_hot_dream_observation(
        jnp.asarray(prediction, dtype=jnp.float32),
        jr.key(4),
    )

    assert not bool(valid)
    chex.assert_trees_all_equal(
        sample,
        jnp.array([1.0, 0.0], dtype=jnp.float32),
    )


def test_zero_mass_sampled_prediction_cannot_update_base_learner() -> None:
    agent = PrototypeAgent(
        _dream_config(
            mode="sample_one_hot",
            predict_delta=False,
        )
    )
    state = agent.start(
        agent.init(jr.key(0)),
        jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    assert agent._buffer is not None
    buffer_state = agent._buffer.add(
        state.buffer_state,
        jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    world_model_state = _zero_world_model_heads(state)

    dreamed, td_errors = agent._run_dreams(
        state.oak_state,
        world_model_state,
        buffer_state,
        jr.key(9),
    )

    chex.assert_trees_all_equal(dreamed, state.oak_state)
    chex.assert_trees_all_equal(
        td_errors,
        jnp.zeros((N_DREAMS,), dtype=jnp.float32),
    )


def test_sample_mode_does_not_shift_legacy_anchor_or_action_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``PrototypeAgent.init`` casts this host-only timestamp to float32. At
    # current epoch values, adjacent representable timestamps are 128 seconds
    # apart, so two otherwise identical initial states can occasionally land
    # on opposite sides of a rounding boundary.
    monkeypatch.setattr(
        multi_head_learner,
        "time",
        SimpleNamespace(time=lambda: 0.0),
    )
    legacy_agent = PrototypeAgent(_dream_config())
    sampled_agent = PrototypeAgent(_dream_config(mode="sample_one_hot"))
    legacy_state = legacy_agent.start(
        legacy_agent.init(jr.key(12)),
        jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    sampled_state = sampled_agent.start(
        sampled_agent.init(jr.key(12)),
        jnp.array([1.0, 0.0], dtype=jnp.float32),
    )
    legacy_agent._buffer = _KeySensitiveBuffer()  # type: ignore[assignment]
    sampled_agent._buffer = _KeySensitiveBuffer()  # type: ignore[assignment]
    legacy_agent._dreamer = _AnchorActionDreamer()  # type: ignore[assignment]
    sampled_agent._dreamer = _AnchorActionDreamer()  # type: ignore[assignment]
    legacy_agent._oak = _DiagnosticOak()  # type: ignore[assignment]
    sampled_agent._oak = _DiagnosticOak()  # type: ignore[assignment]
    root_key = jr.key(17)

    legacy_dreamed, legacy_td_errors = legacy_agent._run_dreams(
        legacy_state.oak_state,
        legacy_state.world_model_state,
        jnp.array(0, dtype=jnp.int32),
        root_key,
    )
    sampled_dreamed, sampled_td_errors = sampled_agent._run_dreams(
        sampled_state.oak_state,
        sampled_state.world_model_state,
        jnp.array(0, dtype=jnp.int32),
        root_key,
    )

    expected = []
    key = _guarded_dream_rng_root(root_key)
    for _ in range(N_DREAMS):
        key, sample_key, action_key = jr.split(key, 3)
        anchor_index = jr.randint(
            sample_key,
            (),
            0,
            2,
            dtype=jnp.int32,
        )
        action = jr.randint(
            action_key,
            (),
            0,
            2,
            dtype=jnp.int32,
        )
        expected.append(
            1.0
            + 2.0 * anchor_index.astype(jnp.float32)
            + action.astype(jnp.float32)
        )

    chex.assert_trees_all_equal(sampled_dreamed, legacy_dreamed)
    chex.assert_trees_all_equal(
        legacy_td_errors,
        jnp.stack(expected),
    )
    chex.assert_trees_all_equal(sampled_td_errors, legacy_td_errors)


def test_accepted_dreams_only_change_base_learner_state() -> None:
    config = PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(
                    SubtaskSpec(
                        feature_index=0,
                        threshold=1.0e6,
                        max_option_steps=8,
                    ),
                ),
                observation_dim=2,
                n_primitive_actions=2,
            )
        ),
        world_model=ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            step_size=0.1,
        ),
        dreaming=DreamingConfig(
            warmup_steps=0,
            max_model_error_ema=1.0e6,
        ),
        buffer_capacity=4,
        n_dreams_per_step=N_DREAMS,
    )
    agent = PrototypeAgent(config)
    state = agent.start(agent.init(jr.key(0)), OBS)
    assert agent._buffer is not None
    buffer_state = agent._buffer.add(state.buffer_state, OBS)

    # Populate real-trajectory/statistics fields while preserving the valid
    # aligned counters produced by ``start``. The option is deliberately active.
    learner = state.oak_state.stomp_state.base_learner_state
    zero_learner = learner.replace(
        head_params=learner.head_params.replace(
            weights=tuple(jnp.zeros_like(weight) for weight in learner.head_params.weights),
            biases=tuple(jnp.zeros_like(bias) for bias in learner.head_params.biases),
        )
    )
    stomp_state = state.oak_state.stomp_state.replace(
        base_learner_state=zero_learner,
        base_last_obs=jnp.array([0.25, 0.75], dtype=jnp.float32),
        base_last_action=jnp.array(2, dtype=jnp.int32),
        last_primitive_action=jnp.array(1, dtype=jnp.int32),
        rng_key=jr.key(21),
        executing_option=jnp.array(0, dtype=jnp.int32),
        option_start_obs=jnp.array([-0.25, 0.5], dtype=jnp.float32),
        option_last_intra_action=jnp.array(1, dtype=jnp.int32),
        option_cumreward=jnp.array(3.0, dtype=jnp.float32),
        option_env_cumreward=jnp.array(4.0, dtype=jnp.float32),
        option_baseline_mass=jnp.array(2.25, dtype=jnp.float32),
        option_discount=jnp.array(0.75, dtype=jnp.float32),
        option_steps=jnp.array(0, dtype=jnp.int32),
        base_average_reward=jnp.array(0.4, dtype=jnp.float32),
    )
    oak_state = state.oak_state.replace(
        stomp_state=stomp_state,
        execution_counts=jnp.array([1], dtype=jnp.int32),
        cumulative_pseudo_rewards=jnp.array([6.0], dtype=jnp.float32),
        utility_ema=jnp.array([7.0], dtype=jnp.float32),
    )

    dreamed, td_errors = agent._run_dreams(
        oak_state,
        state.world_model_state,
        buffer_state,
        jr.key(9),
    )

    # warmup_steps=0 and finite initialized model outputs accept every backup.
    assert int(dreamed.stomp_state.base_learner_state.step_count) == (
        int(oak_state.stomp_state.base_learner_state.step_count) + N_DREAMS
    )
    chex.assert_shape(td_errors, (N_DREAMS,))
    chex.assert_tree_all_finite(td_errors)
    before_q = agent.oak_agent.base_q_values(oak_state, OBS)
    after_q = agent.oak_agent.base_q_values(dreamed, OBS)
    assert bool(jnp.any(before_q != after_q))

    # Replace the one intentionally mutable subtree, then require bitwise
    # identity for option policies/models/lifecycle, average reward, action,
    # RNG, utility statistics, and both real-step counters.
    normalized = dreamed.replace(
        stomp_state=dreamed.stomp_state.replace(
            base_learner_state=oak_state.stomp_state.base_learner_state
        )
    )
    chex.assert_trees_all_equal(normalized, oak_state)
