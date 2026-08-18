"""Complete update working-set preflight for IDBD and Autostep."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.optimizers import IDBD, Autostep

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


def test_idbd_one_bank_and_persistent_fit_while_update_working_set_does_not() -> None:
    one_bank_bytes = 4 * _WORKING_SET_OVERFLOW
    persistent_bytes = 4 * (2 * _WORKING_SET_OVERFLOW + 3)
    update_bytes = 4 * (9 * _WORKING_SET_OVERFLOW + 8)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes <= _INT32_MAX
    assert update_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        IDBD().init(_WORKING_SET_OVERFLOW)


def test_autostep_one_bank_and_persistent_fit_while_update_working_set_does_not() -> None:
    one_bank_bytes = 4 * _WORKING_SET_OVERFLOW
    persistent_bytes = 4 * (3 * _WORKING_SET_OVERFLOW + 5)
    update_bytes = 4 * (10 * _WORKING_SET_OVERFLOW + 8)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes <= _INT32_MAX
    assert update_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        Autostep().init(_WORKING_SET_OVERFLOW)


def test_idbd_persistent_byte_bound_still_fires_first() -> None:
    float32_scalar_limit = _INT32_MAX // 4
    overflowing = (float32_scalar_limit - 3) // 2 + 1
    with pytest.raises(ValueError, match="IDBD state byte count"):
        IDBD().init(overflowing)


def test_legal_idbd_and_autostep_update_identity_is_unchanged() -> None:
    observation = jnp.ones(4, dtype=jnp.float32)
    error = jnp.asarray(0.5, dtype=jnp.float32)

    idbd = IDBD()
    idbd_state = idbd.init(4)
    assert idbd_state.log_step_sizes.shape == (4,)
    assert idbd_state.traces.shape == (4,)
    assert 4 * (2 * 4 + 3) <= _INT32_MAX
    idbd_result = idbd.update(idbd_state, error, observation)
    assert bool(idbd_result.update_applied)
    assert idbd_result.weight_delta.shape == (4,)
    assert idbd_result.new_state.traces.shape == (4,)

    autostep = Autostep()
    autostep_state = autostep.init(4)
    assert autostep_state.step_sizes.shape == (4,)
    assert autostep_state.traces.shape == (4,)
    assert autostep_state.normalizers.shape == (4,)
    autostep_result = autostep.update(autostep_state, error, observation)
    assert bool(autostep_result.update_applied)
    assert autostep_result.weight_delta.shape == (4,)
    assert autostep_result.new_state.normalizers.shape == (4,)
