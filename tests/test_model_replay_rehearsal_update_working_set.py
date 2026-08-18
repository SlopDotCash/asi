"""Complete update working-set preflight for model-replay rehearsal."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.dual_replay import DualReplayConfig, _allocation_sizes
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.model_replay_rehearsal import (
    ModelReplayRehearsal,
    ModelReplayRehearsalConfig,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsembleConfig,
    _ensemble_state_resource_counts,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_WORKING_SET_OVERFLOW = 9_000


def _ensemble(*, observation_dim: int, n_actions: int = 2, ensemble_size: int = 2):
    target_dim = observation_dim + 2
    model = ActionConditionedWorldModelConfig(
        observation_dim=observation_dim,
        n_actions=n_actions,
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


def _replay(*, observation_dim: int, action_dim: int = 2, total_capacity: int = 4):
    return DualReplayConfig(
        total_capacity=total_capacity,
        short_term_capacity=1,
        observation_dim=observation_dim,
        action_dim=action_dim,
        short_term_sample_size=1,
        long_term_sample_size=1,
    )


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_rehearsal_one_bank_and_persistent_fit_while_update_working_set_does_not() -> None:
    ensemble = _ensemble(observation_dim=_WORKING_SET_OVERFLOW)
    replay = _replay(observation_dim=_WORKING_SET_OVERFLOW)
    _, ensemble_bytes = _ensemble_state_resource_counts(
        model=ensemble.model, ensemble_size=ensemble.ensemble_size
    )
    _, replay_bytes = _allocation_sizes(replay)
    persistent_bytes = ensemble_bytes + replay_bytes + 28
    update_bytes = 2 * ensemble_bytes + replay_bytes
    assert ensemble_bytes <= _INT32_MAX
    assert persistent_bytes <= _INT32_MAX
    assert update_bytes > _INT32_MAX
    config = ModelReplayRehearsalConfig(
        ensemble=ensemble, replay=replay, action_encoding="one_hot"
    )
    with pytest.raises(ValueError, match="update working set byte count"):
        ModelReplayRehearsal(config)


def test_rehearsal_persistent_byte_bound_still_fires_first() -> None:
    ensemble = _ensemble(observation_dim=10_000)
    replay = _replay(observation_dim=10_000, total_capacity=7_000)
    _, ensemble_bytes = _ensemble_state_resource_counts(
        model=ensemble.model, ensemble_size=ensemble.ensemble_size
    )
    _, replay_bytes = _allocation_sizes(replay)
    assert ensemble_bytes <= _INT32_MAX
    assert ensemble_bytes + replay_bytes + 28 > _INT32_MAX
    config = ModelReplayRehearsalConfig(
        ensemble=ensemble, replay=replay, action_encoding="one_hot"
    )
    with pytest.raises(ValueError, match="state byte count"):
        ModelReplayRehearsal(config)


def test_legal_rehearsal_init_identity_is_unchanged() -> None:
    config = ModelReplayRehearsalConfig(
        ensemble=_ensemble(observation_dim=2),
        replay=_replay(observation_dim=2),
        action_encoding="one_hot",
    )
    rehearsal = ModelReplayRehearsal(config)
    state = rehearsal.init(jr.key(0))
    budget = rehearsal.resource_budget(state)
    assert budget.persistent_state_bytes <= _INT32_MAX
    assert state.ensemble_state is not None
    assert int(state.real_attempt_count) == 0
