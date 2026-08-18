"""Complete update working-set preflight for off-policy TD learners."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.off_policy_td import (
    ETDLinearLearner,
    GradientTDLinearLearner,
    OffPolicyTDLinearLearner,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_WORKING_SET_OVERFLOW = 100_000_000


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_off_policy_td_one_bank_and_persistent_fit_while_update_working_set_does_not() -> None:
    one_bank_bytes = 4 * _WORKING_SET_OVERFLOW
    persistent_bytes = 4 * (2 * _WORKING_SET_OVERFLOW + 3)
    update_bytes = 4 * (8 * _WORKING_SET_OVERFLOW + 16)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes <= _INT32_MAX
    assert update_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        OffPolicyTDLinearLearner().init(_WORKING_SET_OVERFLOW)


def test_etd_one_bank_and_persistent_fit_while_update_working_set_does_not() -> None:
    one_bank_bytes = 4 * _WORKING_SET_OVERFLOW
    persistent_bytes = 4 * (2 * _WORKING_SET_OVERFLOW + 5)
    update_bytes = 4 * (8 * _WORKING_SET_OVERFLOW + 16)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes <= _INT32_MAX
    assert update_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        ETDLinearLearner().init(_WORKING_SET_OVERFLOW)


def test_gradient_td_one_bank_and_persistent_fit_while_update_working_set_does_not() -> None:
    width = _WORKING_SET_OVERFLOW + 1
    one_bank_bytes = 4 * width
    persistent_bytes = 4 * (3 * width + 1)
    update_bytes = 4 * (9 * width + 16)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes <= _INT32_MAX
    assert update_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        GradientTDLinearLearner().init(_WORKING_SET_OVERFLOW)


def test_off_policy_td_persistent_byte_bound_still_fires_first() -> None:
    with pytest.raises(ValueError, match="state_nbytes"):
        OffPolicyTDLinearLearner().init(300_000_000)


def test_legal_off_policy_td_update_identity_is_unchanged() -> None:
    observation = jnp.ones(4, dtype=jnp.float32)
    next_observation = jnp.asarray([1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    reward = jnp.asarray(1.0, dtype=jnp.float32)
    gamma = jnp.asarray(0.9, dtype=jnp.float32)
    rho = jnp.asarray(1.0, dtype=jnp.float32)

    off_policy = OffPolicyTDLinearLearner(step_size=0.05, trace_decay=0.5)
    off_policy_state = off_policy.init(4)
    assert off_policy_state.weights.shape == (4,)
    assert off_policy_state.eligibility_traces.shape == (4,)
    assert 4 * (2 * 4 + 3) <= _INT32_MAX
    off_policy_result = off_policy.update(
        off_policy_state, observation, reward, next_observation, gamma, rho
    )
    assert bool(off_policy_result.update_applied)
    assert off_policy_result.state.weights.shape == (4,)
    assert off_policy_result.metrics.shape == (5,)

    etd = ETDLinearLearner(step_size=0.05, trace_decay=0.5)
    etd_state = etd.init(4)
    etd_result = etd.update(etd_state, observation, reward, next_observation, gamma, rho)
    assert bool(etd_result.update_applied)
    assert etd_result.state.weights.shape == (4,)
    assert etd_result.state.eligibility_traces.shape == (4,)

    gradient = GradientTDLinearLearner(step_size=0.05, trace_decay=0.5)
    gradient_state = gradient.init(4)
    assert gradient_state.weights.shape == (5,)
    gradient_result = gradient.update(
        gradient_state, observation, reward, next_observation, gamma, rho
    )
    assert bool(gradient_result.update_applied)
    assert gradient_result.state.weights.shape == (5,)
    assert gradient_result.state.secondary_weights.shape == (5,)
