"""Model replay transactions preserve behavior across action encodings."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import tree_util

from alberta_framework.core.dual_replay import DualReplayConfig
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.model_replay_rehearsal import (
    ModelReplayRehearsal,
    ModelReplayRehearsalConfig,
    RealModelReplayEvent,
    ReplayActionEncoding,
    load_model_replay_rehearsal_checkpoint,
    save_model_replay_rehearsal_checkpoint,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import WorldModelEnsembleConfig

_OBSERVATION_DIM = 2
_N_ACTIONS = 3


def _ensemble_config(*, ensemble_size: int = 2) -> WorldModelEnsembleConfig:
    target_dim = _OBSERVATION_DIM + 2
    model = ActionConditionedWorldModelConfig(
        observation_dim=_OBSERVATION_DIM,
        n_actions=_N_ACTIONS,
        hidden_sizes=(),
        gamma=0.95,
        step_size=0.05,
        sparsity=0.0,
        use_layer_norm=False,
        error_decay=0.8,
    )
    signals = LearningSignalEstimatorConfig(
        ensemble_size=ensemble_size,
        target_dim=target_dim,
        progress_warmup_steps=2,
        change_calibration_steps=2,
        fast_loss_decay=0.5,
        slow_loss_decay=0.9,
        max_input_magnitude=100.0,
        max_predicted_variance=10_000.0,
        max_observed_loss=10_000.0,
    )
    return WorldModelEnsembleConfig(
        model=model,
        signal_estimator=signals,
        ensemble_size=ensemble_size,
        bootstrap_probability=0.5,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=1,
        residual_variance_floor=1e-6,
    )


def _replay_config(*, action_dim: int, total_capacity: int = 8) -> DualReplayConfig:
    return DualReplayConfig(
        total_capacity=total_capacity,
        short_term_capacity=4,
        observation_dim=_OBSERVATION_DIM,
        action_dim=action_dim,
        short_term_sample_size=2,
        long_term_sample_size=2,
    )


def _rehearsal(encoding: ReplayActionEncoding) -> ModelReplayRehearsal:
    action_dim = 1 if encoding == "scalar_index" else _N_ACTIONS
    config = ModelReplayRehearsalConfig(
        ensemble=_ensemble_config(),
        replay=_replay_config(action_dim=action_dim),
        action_encoding=encoding,
    )
    return ModelReplayRehearsal(config)


def test_internal_template_keys_ignore_the_ambient_prng_default() -> None:
    with jax.default_prng_impl("threefry2x32"):
        expected = _rehearsal("scalar_index").resource_budget().to_config()
    with jax.default_prng_impl("rbg"):
        observed = _rehearsal("scalar_index").resource_budget().to_config()

    assert observed == expected


def test_default_checkpoint_template_ignores_the_ambient_prng_default(
    tmp_path: Path,
) -> None:
    rehearsal = _rehearsal("scalar_index")
    state = rehearsal.init(jr.key(7, impl="threefry2x32"))
    path = tmp_path / "model_replay"
    save_model_replay_rehearsal_checkpoint(rehearsal, state, path)

    with jax.default_prng_impl("rbg"):
        restored_rehearsal, restored_state = load_model_replay_rehearsal_checkpoint(path)

    assert restored_rehearsal.to_config() == rehearsal.to_config()
    assert all(
        bool(jnp.array_equal(expected, observed))
        for expected, observed in zip(
            tree_util.tree_leaves(state),
            tree_util.tree_leaves(restored_state),
            strict=True,
        )
    )


def _event(
    *, observation: jnp.ndarray, action: int, next_observation: jnp.ndarray
) -> RealModelReplayEvent:
    return RealModelReplayEvent(  # type: ignore[call-arg]
        observation=observation,
        action=jnp.asarray(action, dtype=jnp.int32),
        reward=jnp.asarray(0.5, dtype=jnp.float32),
        discount=jnp.asarray(0.99, dtype=jnp.float32),
        terminated=jnp.asarray(False),
        truncated=jnp.asarray(False),
        next_observation=next_observation,
        representation_version=jnp.asarray(0, dtype=jnp.int32),
        provenance_id=jnp.asarray(0, dtype=jnp.int32),
        source_id=jnp.asarray(0, dtype=jnp.int32),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True),
        valid=jnp.asarray(True),
    )


def test_both_action_encodings_run_equivalent_real_rehearsal_transactions() -> None:
    """The two ``action_encoding`` branches must be behaviorally equivalent.

    Feeding the identical observation/action/reward stream through the
    ``"scalar_index"`` and ``"one_hot"`` compositions must produce the same
    decoded action trace and the same real observed loss at every step: the
    storage encoding is not supposed to change what gets learned.
    """
    scalar_rehearsal = _rehearsal("scalar_index")
    one_hot_rehearsal = _rehearsal("one_hot")
    scalar_state = scalar_rehearsal.init(jr.key(0))
    one_hot_state = one_hot_rehearsal.init(jr.key(0))

    # Bind both public conversion branches before exercising the same conversion
    # inside the jitted replay transaction.
    for action in range(_N_ACTIONS):
        scalar_stored = scalar_rehearsal.encode_action(jnp.asarray(action, dtype=jnp.int32))
        one_hot_stored = one_hot_rehearsal.encode_action(jnp.asarray(action, dtype=jnp.int32))
        assert scalar_stored.tolist() == [float(action)]
        assert one_hot_stored.tolist() == [float(index == action) for index in range(_N_ACTIONS)]
        scalar_conversion = scalar_rehearsal.decode_action(scalar_stored)
        one_hot_conversion = one_hot_rehearsal.decode_action(one_hot_stored)
        assert bool(scalar_conversion.valid) and int(scalar_conversion.action) == action
        assert bool(one_hot_conversion.valid) and int(one_hot_conversion.action) == action

    key = jr.key(7)
    for step, action in enumerate(range(_N_ACTIONS)):
        key, subkey = jr.split(key)
        obs_key, next_key = jr.split(subkey)
        observation = jr.normal(obs_key, (_OBSERVATION_DIM,), dtype=jnp.float32)
        next_observation = jr.normal(next_key, (_OBSERVATION_DIM,), dtype=jnp.float32)
        event = _event(observation=observation, action=action, next_observation=next_observation)

        scalar_result = scalar_rehearsal.step(scalar_state, event)
        one_hot_result = one_hot_rehearsal.step(one_hot_state, event)
        scalar_state = scalar_result.state
        one_hot_state = one_hot_result.state

        assert bool(scalar_result.diagnostics.transaction_applied)
        assert bool(one_hot_result.diagnostics.transaction_applied)
        assert bool(jnp.any(scalar_result.trace.model_updates_applied))
        assert bool(jnp.any(one_hot_result.trace.model_updates_applied))
        assert int(scalar_state.rehearsal_applied_count) > 0
        assert int(one_hot_state.rehearsal_applied_count) > 0
        assert int(scalar_state.accepted_real_event_count) == step + 1
        assert int(one_hot_state.accepted_real_event_count) == step + 1
        assert scalar_result.trace.actions.tolist() == one_hot_result.trace.actions.tolist()
        assert bool(
            jnp.allclose(
                scalar_result.trace.observed_losses,
                one_hot_result.trace.observed_losses,
                rtol=1e-6,
                atol=1e-7,
            )
        )
        assert bool(
            jnp.allclose(
                scalar_result.real_observed_loss,
                one_hot_result.real_observed_loss,
                rtol=1e-6,
                atol=1e-7,
            )
        )
        assert all(
            bool(jnp.array_equal(scalar_leaf, one_hot_leaf))
            for scalar_leaf, one_hot_leaf in zip(
                tree_util.tree_leaves(scalar_state.ensemble_state),
                tree_util.tree_leaves(one_hot_state.ensemble_state),
                strict=True,
            )
        )

    # The real dispatcher committed genuinely different storage layouts while
    # preserving the same decoded action stream and learned trajectory.
    scalar_entries = scalar_state.replay_state.short_term
    one_hot_entries = one_hot_state.replay_state.short_term
    assert scalar_entries.actions[:_N_ACTIONS].tolist() == [[0.0], [1.0], [2.0]]
    assert one_hot_entries.actions[:_N_ACTIONS].tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert scalar_entries.valid[:_N_ACTIONS].tolist() == [True] * _N_ACTIONS
    assert one_hot_entries.valid[:_N_ACTIONS].tolist() == [True] * _N_ACTIONS

    scalar_rehearsal.validate_state(scalar_state)
    one_hot_rehearsal.validate_state(one_hot_state)
