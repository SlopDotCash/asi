"""Complete update working-set preflight for action-conditioned world models."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
    OneStepWorldModel,
    WorldModelConfig,
    _preflight_world_model_update_working_set,
    _world_model_update_working_set_bytes,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_AC_LAST_PERSIST = 16_380
_AC_LAST_WORKING_SET = 11_581
_AC_FIRST_WORKING_SET_OVERFLOW = 11_582
_WM_LAST_PERSIST = 16_381
_WM_FIRST_WORKING_SET_OVERFLOW = 11_583


def _ac_kwargs(observation_dim: int) -> dict[str, object]:
    return {
        "observation_dim": observation_dim,
        "action_feature_dim": 2,
        "hidden_sizes": (),
        "n_heads": observation_dim + 2,
        "outer_state_scalars": 2 * observation_dim + 4,
    }


def _wm_kwargs(observation_dim: int) -> dict[str, object]:
    return {
        "observation_dim": observation_dim,
        "action_feature_dim": 2,
        "hidden_sizes": (),
        "n_heads": observation_dim + 1,
        "outer_state_scalars": 1,
    }


def _ac_persist_bytes(observation_dim: int) -> int:
    return 8 * observation_dim * observation_dim + 48 * observation_dim + 76


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_world_model_persist_fits_while_update_working_set_does_not() -> None:
    persist = _ac_persist_bytes(_AC_FIRST_WORKING_SET_OVERFLOW)
    working = _world_model_update_working_set_bytes(**_ac_kwargs(_AC_FIRST_WORKING_SET_OVERFLOW))
    assert persist <= _INT32_MAX
    assert persist == 1_073_697_804
    assert (
        _world_model_update_working_set_bytes(**_ac_kwargs(_AC_LAST_WORKING_SET))
        <= _INT32_MAX
    )
    assert working > _INT32_MAX
    config = ActionConditionedWorldModelConfig(
        observation_dim=_AC_FIRST_WORKING_SET_OVERFLOW,
        n_actions=2,
        hidden_sizes=(),
    )
    model = ActionConditionedWorldModel(config)
    with pytest.raises(ValueError, match="update working set byte count"):
        model.init(jr.key(0))


def test_world_model_last_legal_persistent_config_still_constructs() -> None:
    ActionConditionedWorldModelConfig(
        observation_dim=_AC_LAST_PERSIST,
        n_actions=2,
        hidden_sizes=(),
    )
    WorldModelConfig(
        observation_dim=_WM_LAST_PERSIST,
        n_actions=2,
        hidden_sizes=(),
    )
    with pytest.raises(ValueError, match="combined_direct_state_bytes"):
        ActionConditionedWorldModelConfig(
            observation_dim=_AC_LAST_PERSIST + 1,
            n_actions=2,
            hidden_sizes=(),
        )
    with pytest.raises(ValueError, match="combined_direct_state_bytes"):
        WorldModelConfig(
            observation_dim=_WM_LAST_PERSIST + 1,
            n_actions=2,
            hidden_sizes=(),
        )


def test_world_model_persistent_aggregate_bound_still_fires_first() -> None:
    with pytest.raises(ValueError, match="combined_direct_state_bytes"):
        ActionConditionedWorldModelConfig(
            observation_dim=20_000,
            n_actions=2,
            hidden_sizes=(),
        )
    with pytest.raises(ValueError, match="combined_direct_state_bytes"):
        WorldModelConfig(
            observation_dim=20_000,
            n_actions=2,
            hidden_sizes=(),
        )


def test_legal_world_model_init_and_update_identity_is_unchanged() -> None:
    ac = ActionConditionedWorldModel(
        ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
        )
    )
    ac_state = ac.init(jr.key(7))
    obs = jnp.array([0.1, -0.2], dtype=jnp.float32)
    next_obs = jnp.array([0.2, 0.1], dtype=jnp.float32)
    ac_result = ac.update(ac_state, obs, jnp.int32(1), 0.5, 0.9, next_obs)
    assert ac_result.prediction.next_observation.shape == (2,)
    assert ac_result.targets.shape == (4,)
    assert _world_model_update_working_set_bytes(**_ac_kwargs(2)) <= _INT32_MAX

    wm = OneStepWorldModel(
        WorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
        )
    )
    wm_state = wm.init(jr.key(8))
    wm_result = wm.update(wm_state, obs, jnp.int32(1), 0.5, next_obs)
    assert wm_result.prediction.next_observation.shape == (2,)
    assert wm_result.targets.shape == (3,)
    assert _world_model_update_working_set_bytes(**_wm_kwargs(2)) <= _INT32_MAX


def test_world_model_exact_last_legal_update_width_and_first_overflow() -> None:
    low, high = 1, _AC_LAST_PERSIST
    while low < high:
        middle = (low + high + 1) // 2
        if _world_model_update_working_set_bytes(**_ac_kwargs(middle)) <= _INT32_MAX:
            low = middle
        else:
            high = middle - 1
    assert low == _AC_LAST_WORKING_SET
    _preflight_world_model_update_working_set(**_ac_kwargs(low))
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_world_model_update_working_set(**_ac_kwargs(low + 1))

    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_world_model_update_working_set(
            **_wm_kwargs(_WM_FIRST_WORKING_SET_OVERFLOW)
        )
    OneStepWorldModel(
        WorldModelConfig(
            observation_dim=_WM_FIRST_WORKING_SET_OVERFLOW,
            n_actions=2,
            hidden_sizes=(),
        )
    )
