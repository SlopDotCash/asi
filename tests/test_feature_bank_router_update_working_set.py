"""Complete update working-set preflight for feature-bank routing."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.feature_bank_router import (
    FeatureBankRouter,
    FeatureBankRouterConfig,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_LAST_LEGAL_PERSIST_SLOTS = 268_435_454
_FIRST_WORKING_SET_OVERFLOW = 26_753


def _persist_bytes(active_slots: int) -> int:
    return 4 * (2 * active_slots + 2)


def _route_working_set_bytes(active_slots: int) -> int:
    return 2 * _persist_bytes(active_slots) + 3 * active_slots * active_slots


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_router_persist_fits_while_route_working_set_does_not() -> None:
    persist = _persist_bytes(_FIRST_WORKING_SET_OVERFLOW)
    working = _route_working_set_bytes(_FIRST_WORKING_SET_OVERFLOW)
    assert persist <= _INT32_MAX
    assert _route_working_set_bytes(_FIRST_WORKING_SET_OVERFLOW - 1) <= _INT32_MAX
    assert working > _INT32_MAX
    config = FeatureBankRouterConfig(
        base_dim=2, active_slots=_FIRST_WORKING_SET_OVERFLOW
    )
    assert _persist_bytes(config.active_slots) == persist
    router = FeatureBankRouter(config)
    with pytest.raises(ValueError, match="update working set byte count"):
        router.init()


def test_router_last_legal_persistent_config_still_constructs() -> None:
    boundary = FeatureBankRouterConfig(
        base_dim=2, active_slots=_LAST_LEGAL_PERSIST_SLOTS
    )
    assert _persist_bytes(boundary.active_slots) == 2_147_483_640
    assert _persist_bytes(boundary.active_slots) <= _INT32_MAX
    assert _route_working_set_bytes(boundary.active_slots) > _INT32_MAX
    FeatureBankRouter(boundary)
    with pytest.raises(ValueError, match="router_state_nbytes"):
        FeatureBankRouterConfig(base_dim=2, active_slots=_LAST_LEGAL_PERSIST_SLOTS + 1)


def test_legal_router_init_and_route_identity_is_unchanged() -> None:
    router = FeatureBankRouter(FeatureBankRouterConfig(base_dim=4, active_slots=4))
    old = jnp.asarray(
        [[0, 1], [0, 2], [1, 3], [-1, -1]],
        dtype=jnp.int32,
    )
    new = jnp.asarray(
        [[1, 3], [0, 1], [2, 3], [-1, -1]],
        dtype=jnp.int32,
    )
    consumers = {
        "weights": jnp.arange(8, dtype=jnp.float32),
    }
    state = router.init(old)
    result = router.route(state, consumers, new)
    assert result.state.descriptors.shape == (4, 2)
    assert result.consumers["weights"].shape == (8,)
    assert _persist_bytes(4) <= _INT32_MAX
    assert _route_working_set_bytes(4) <= _INT32_MAX
