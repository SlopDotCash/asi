"""#1383-complete update working-set preflight for the OaK host."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.oak import (
    OaKAgent,
    OaKConfig,
    _oak_direct_state_bytes,
    _oak_update_working_set_bytes,
    _preflight_oak_update_working_set,
)
from alberta_framework.core.options import (
    STOMPConfig,
    SubtaskSpec,
    _stomp_direct_state_bytes,
    _stomp_update_working_set_bytes,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_LAST_OAK_FIT_HIDDEN = 19_884_101
_FIRST_OAK_OVERFLOW_HIDDEN = 19_884_102


def _boundary_stomp(hidden_width: int) -> STOMPConfig:
    """Return a valid STOMP config tuned to the narrower OaK-only boundary."""
    return STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),),
        observation_dim=1,
        n_primitive_actions=1,
        base_hidden_sizes=(hidden_width,),
    )


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_valid_stomp_reaches_the_oak_only_update_boundary() -> None:
    stomp = _boundary_stomp(_FIRST_OAK_OVERFLOW_HIDDEN)
    stomp_persist_bytes = _stomp_direct_state_bytes(stomp)
    stomp_working_set_bytes = _stomp_update_working_set_bytes(stomp)
    persist_bytes = _oak_direct_state_bytes(stomp)
    working_set_bytes = _oak_update_working_set_bytes(stomp)
    extras_bytes = working_set_bytes - 3 * persist_bytes

    assert stomp_persist_bytes == 715_827_852
    assert stomp_working_set_bytes == 2_147_483_620
    assert stomp_working_set_bytes <= _INT32_MAX
    assert persist_bytes == 715_827_876
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _FIRST_OAK_OVERFLOW_HIDDEN <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes == 2_147_483_684
    with pytest.raises(ValueError, match="update working set byte count"):
        OaKConfig(stomp=stomp)


def test_adjacent_hidden_widths_straddle_only_the_oak_boundary() -> None:
    assert _FIRST_OAK_OVERFLOW_HIDDEN == _LAST_OAK_FIT_HIDDEN + 1
    last_stomp = _boundary_stomp(_LAST_OAK_FIT_HIDDEN)
    first_stomp = _boundary_stomp(_FIRST_OAK_OVERFLOW_HIDDEN)

    assert _stomp_update_working_set_bytes(last_stomp) <= _INT32_MAX
    assert _stomp_update_working_set_bytes(first_stomp) <= _INT32_MAX
    assert _oak_update_working_set_bytes(last_stomp) == 2_147_483_576
    assert _oak_update_working_set_bytes(first_stomp) == 2_147_483_684

    config = OaKConfig(stomp=last_stomp)
    assert config.stomp.base_hidden_sizes == (_LAST_OAK_FIT_HIDDEN,)
    with pytest.raises(ValueError, match="update working set byte count"):
        OaKConfig(stomp=first_stomp)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_oak_update_working_set(
            _boundary_stomp(_FIRST_OAK_OVERFLOW_HIDDEN)
        )


def test_nested_stomp_overflow_is_not_mislabeled_as_oak_coverage() -> None:
    """The former observation-width fixture now fails in STOMP, before OaK."""
    with pytest.raises(ValueError, match="STOMP update working set"):
        STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0),),
            observation_dim=20_000,
            n_primitive_actions=1,
            base_hidden_sizes=(),
        )


def test_legal_small_oak_still_constructs() -> None:
    stomp = STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),),
        observation_dim=4,
        n_primitive_actions=2,
        base_hidden_sizes=(),
    )
    persist_bytes = _oak_direct_state_bytes(stomp)
    assert persist_bytes == 448
    agent = OaKAgent(OaKConfig(stomp=stomp))
    state = agent.init(jr.key(0))
    agent.update(
        state,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.zeros((4,), dtype=jnp.float32),
    )
