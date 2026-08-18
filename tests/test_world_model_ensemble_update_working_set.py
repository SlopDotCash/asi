"""#1383-complete update working-set preflight for world-model ensemble."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import tree_util

from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    _ensemble_state_resource_counts,
    _ensemble_update_working_set_bytes,
    _preflight_ensemble_update_working_set,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
# Persist + extras last-legal N on origin/main: still constructs there, but
# 3 × persist + extras overflows signed int32.
_OVERFLOW_ENSEMBLE_SIZE = 6_628_035
_LAST_FIT_ENSEMBLE_SIZE = 2_485_513
_FIRST_OVERFLOW_ENSEMBLE_SIZE = 2_485_514


def _linear_model() -> ActionConditionedWorldModelConfig:
    return ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        gamma=0.95,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        use_layer_norm=False,
        error_decay=0.8,
    )


def _config(ensemble_size: int) -> WorldModelEnsembleConfig:
    return WorldModelEnsembleConfig(
        model=_linear_model(),
        signal_estimator=LearningSignalEstimatorConfig(
            ensemble_size=ensemble_size,
            target_dim=4,
            progress_warmup_steps=2,
            change_calibration_steps=2,
            fast_loss_decay=0.5,
            slow_loss_decay=0.9,
            max_input_magnitude=100.0,
            max_predicted_variance=10_000.0,
            max_observed_loss=10_000.0,
        ),
        ensemble_size=ensemble_size,
        bootstrap_probability=0.5,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=1,
        residual_variance_floor=1.0e-6,
    )


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_ensemble_persist_matches_jit_and_width_still_fits() -> None:
    config = _config(2)
    ensemble = WorldModelEnsemble(config)
    state = ensemble.init(jr.key(0))
    actual = 0
    for leaf in tree_util.tree_leaves(state):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jax.dtypes.prng_key):
            array = jr.key_data(leaf)
        else:
            array = jnp.asarray(leaf)
        actual += int(array.size) * int(array.dtype.itemsize)
    _, persist_bytes = _ensemble_state_resource_counts(
        model=config.model, ensemble_size=2
    )
    assert actual == persist_bytes == 600
    assert ensemble.resource_budget(state).persistent_state_bytes == persist_bytes
    working_set_bytes = _ensemble_update_working_set_bytes(
        model=config.model, ensemble_size=_OVERFLOW_ENSEMBLE_SIZE
    )
    persist_at_overflow, extras_fit = _persist_and_update(
        config.model, _OVERFLOW_ENSEMBLE_SIZE
    )
    assert persist_at_overflow <= _INT32_MAX
    assert extras_fit <= _INT32_MAX
    assert 4 * config.model.observation_dim <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        _config(_OVERFLOW_ENSEMBLE_SIZE)


def _persist_and_update(
    model: ActionConditionedWorldModelConfig, ensemble_size: int
) -> tuple[int, int]:
    _, persist_bytes = _ensemble_state_resource_counts(
        model=model, ensemble_size=ensemble_size
    )
    working_set_bytes = _ensemble_update_working_set_bytes(
        model=model, ensemble_size=ensemble_size
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    return persist_bytes, persist_bytes + extras_bytes


def test_ensemble_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    model = _linear_model()
    for ensemble_size in range(_LAST_FIT_ENSEMBLE_SIZE, _FIRST_OVERFLOW_ENSEMBLE_SIZE + 2):
        persist_bytes, update_bytes = _persist_and_update(model, ensemble_size)
        working_set_bytes = _ensemble_update_working_set_bytes(
            model=model, ensemble_size=ensemble_size
        )
        assert persist_bytes <= _INT32_MAX
        assert update_bytes <= _INT32_MAX
        assert 4 * model.observation_dim <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = ensemble_size
        elif first_overflow is None:
            first_overflow = ensemble_size
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _LAST_FIT_ENSEMBLE_SIZE
    config = _config(last_fit)
    assert config.ensemble_size == last_fit
    with pytest.raises(ValueError, match="update working set byte count"):
        _config(first_overflow)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_ensemble_update_working_set(
            model=_linear_model(),
            ensemble_size=_OVERFLOW_ENSEMBLE_SIZE,
        )


def test_legal_small_ensemble_still_constructs() -> None:
    config = _config(2)
    WorldModelEnsemble(config).init(jr.key(1))
