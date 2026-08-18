"""#1383-complete step working-set preflight for temporal context."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import tree_util

from alberta_framework.core.temporal_context import (
    TemporalContextConfig,
    TemporalContextFeaturizer,
    _preflight_temporal_context_update_working_set,
    _temporal_context_persist_bytes,
    _temporal_context_update_working_set_bytes,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_INPUT_DIM = 25_000_000
_PHASE_PRODUCT_KWARGS = {
    "include_raw": True,
    "include_ema": True,
    "include_delta": True,
    "include_phase_products": True,
    "n_periods": 3,
}


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_temporal_context_persist_matches_jit_and_width_still_fits() -> None:
    config = TemporalContextConfig(input_dim=3, include_phase_products=True)
    state = TemporalContextFeaturizer(config).init()
    actual = sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in tree_util.tree_leaves(state)
    )
    persist_bytes = _temporal_context_persist_bytes(3)
    assert actual == persist_bytes == 16
    working_set_bytes = _temporal_context_update_working_set_bytes(
        _OVERFLOW_INPUT_DIM,
        **_PHASE_PRODUCT_KWARGS,
    )
    persist_at_overflow = _temporal_context_persist_bytes(_OVERFLOW_INPUT_DIM)
    assert persist_at_overflow <= _INT32_MAX
    assert 4 * _OVERFLOW_INPUT_DIM <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        TemporalContextConfig(
            input_dim=_OVERFLOW_INPUT_DIM,
            include_phase_products=True,
        )


def test_temporal_context_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for input_dim in range(22_369_618, 22_369_624):
        working_set_bytes = _temporal_context_update_working_set_bytes(
            input_dim,
            **_PHASE_PRODUCT_KWARGS,
        )
        persist_bytes = _temporal_context_persist_bytes(input_dim)
        assert persist_bytes <= _INT32_MAX
        assert 4 * input_dim <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = input_dim
        elif first_overflow is None:
            first_overflow = input_dim
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    config = TemporalContextConfig(input_dim=last_fit, include_phase_products=True)
    assert config.input_dim == last_fit
    with pytest.raises(ValueError, match="update working set byte count"):
        TemporalContextConfig(input_dim=first_overflow, include_phase_products=True)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="update working set byte count"):
        _preflight_temporal_context_update_working_set(
            _OVERFLOW_INPUT_DIM,
            **_PHASE_PRODUCT_KWARGS,
        )


def test_legal_small_temporal_context_still_constructs() -> None:
    config = TemporalContextConfig(input_dim=4, include_phase_products=True)
    assert config.output_dim() == 3 * 4 + 6 + 6 * 4
    TemporalContextFeaturizer(config).init()
