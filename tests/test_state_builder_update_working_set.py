"""Complete update working-set preflight for the online-gated state builder."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import tree_util

from alberta_framework.core.state_builder import (
    OnlineGatedStateBuilder,
    OnlineGatedStateBuilderConfig,
    _online_gated_update_working_set_bytes,
    _preflight_online_gated_update_working_set,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def _persist_bytes(observation_dim: int, hidden_dim: int, n_actions: int = 0) -> int:
    event_dim = observation_dim + n_actions + 2
    parameter_count = 2 * hidden_dim * (event_dim + 1)
    persist_scalars = (
        parameter_count + hidden_dim + hidden_dim * parameter_count + 3
    )
    return 4 * persist_scalars


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_online_gated_persist_matches_jit_materialized_leaves() -> None:
    config = OnlineGatedStateBuilderConfig(
        observation_dim=3,
        hidden_dim=2,
        n_actions=1,
    )
    builder = OnlineGatedStateBuilder(config)
    state = builder.init(jr.PRNGKey(0))
    actual = sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in tree_util.tree_leaves(state)
    )
    assert actual == builder.resource_budget().state_bytes
    assert actual == _persist_bytes(3, 2, 1)


def test_online_gated_persist_fits_while_update_working_set_does_not() -> None:
    observation_dim = 45_000_000
    persist_bytes = _persist_bytes(observation_dim, 1)
    working_set_bytes = _online_gated_update_working_set_bytes(
        observation_dim,
        0,
        1,
        include_raw_observation=True,
    )
    assert persist_bytes <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        OnlineGatedStateBuilderConfig(observation_dim=observation_dim, hidden_dim=1)


def test_online_gated_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for observation_dim in range(26_843_540, 26_843_545):
        working_set_bytes = _online_gated_update_working_set_bytes(
            observation_dim,
            0,
            1,
            include_raw_observation=True,
        )
        if working_set_bytes <= _INT32_MAX:
            last_fit = observation_dim
        elif first_overflow is None:
            first_overflow = observation_dim
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert _persist_bytes(last_fit, 1) <= _INT32_MAX
    OnlineGatedStateBuilderConfig(observation_dim=last_fit, hidden_dim=1)
    with pytest.raises(ValueError, match="update working set byte count"):
        OnlineGatedStateBuilderConfig(observation_dim=first_overflow, hidden_dim=1)


def test_online_gated_persistent_byte_bound_still_fires_first() -> None:
    observation_dim = 134_217_725
    assert _persist_bytes(observation_dim, 1) > _INT32_MAX
    with pytest.raises(ValueError, match="state_bytes"):
        OnlineGatedStateBuilderConfig(observation_dim=observation_dim, hidden_dim=1)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_online_gated_update_working_set(
            45_000_000,
            0,
            1,
            include_raw_observation=True,
        )
