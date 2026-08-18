"""Complete update working-set preflight for compositional features."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.compositional_features import CompositionalFeatureLearner

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_LAST_LEGAL_CONSTRUCTOR = (_INT32_MAX - 80) // 68
_FIRST_WORKING_SET_OVERFLOW = 12_859_182


def _persist_bytes(
    n_features: int,
    n_tasks: int = 1,
    candidate_count: int = 0,
    generator_resource_contexts: int = 1,
) -> int:
    return (
        20
        + 48 * generator_resource_contexts
        + 56 * n_features
        + 12 * n_tasks
        + 68 * candidate_count
        + 12 * n_features * n_tasks
        + 12 * n_tasks * candidate_count
        + 4 * n_features * candidate_count
    )


def _update_working_set_bytes(
    n_features: int,
    n_tasks: int = 1,
    candidate_count: int = 0,
    generator_resource_contexts: int = 1,
) -> int:
    extras = 227 + 31 * n_features + 8 * n_tasks + 32 * candidate_count
    return 2 * _persist_bytes(
        n_features, n_tasks, candidate_count, generator_resource_contexts
    ) + extras


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_compositional_persist_fits_while_update_working_set_does_not() -> None:
    persist = _persist_bytes(_FIRST_WORKING_SET_OVERFLOW)
    working = _update_working_set_bytes(_FIRST_WORKING_SET_OVERFLOW)
    assert persist <= _INT32_MAX
    assert _update_working_set_bytes(_FIRST_WORKING_SET_OVERFLOW - 1) <= _INT32_MAX
    assert working > _INT32_MAX
    learner = CompositionalFeatureLearner(_FIRST_WORKING_SET_OVERFLOW, 1)
    with pytest.raises(ValueError, match="update working set byte count"):
        learner.init(1, jr.key(0))


def test_compositional_last_legal_constructor_still_constructs() -> None:
    persist = _persist_bytes(_LAST_LEGAL_CONSTRUCTOR)
    assert persist <= _INT32_MAX
    assert persist == 2_147_483_600
    assert _update_working_set_bytes(_LAST_LEGAL_CONSTRUCTOR) > _INT32_MAX
    CompositionalFeatureLearner(_LAST_LEGAL_CONSTRUCTOR, 1)
    with pytest.raises(ValueError, match="persistent state bytes"):
        CompositionalFeatureLearner(_LAST_LEGAL_CONSTRUCTOR + 1, 1)


def test_legal_compositional_init_and_update_identity_is_unchanged() -> None:
    learner = CompositionalFeatureLearner(4, 1)
    state = learner.init(1, jr.key(0))
    result = learner.update(
        state,
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
    )
    assert state.ops.shape == (4,)
    assert result.predictions.shape == (1,)
    assert _persist_bytes(4) <= _INT32_MAX
    assert _update_working_set_bytes(4) <= _INT32_MAX
