"""Complete aggregate preflight for Step 1 baseline optimizer state."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.baseline_optimizers import AdaGain, Adam

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_ONE_BANK_FITS = _INT32_MAX // 4
_WORKING_SET_OVERFLOW = 100_000_000


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_adam_one_moment_bank_fits_while_persistent_aggregate_does_not() -> None:
    one_bank_bytes = 4 * _ONE_BANK_FITS
    persistent_bytes = 4 * (2 * _ONE_BANK_FITS + 7)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="persistent state byte count"):
        Adam().init(_ONE_BANK_FITS)


def test_adagain_one_gain_bank_fits_while_persistent_aggregate_does_not() -> None:
    one_bank_bytes = 4 * _ONE_BANK_FITS
    persistent_bytes = 4 * (2 * _ONE_BANK_FITS + 4)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="persistent state byte count"):
        AdaGain().init(_ONE_BANK_FITS)


def test_adam_rejects_simultaneous_update_working_set() -> None:
    one_bank_bytes = 4 * _WORKING_SET_OVERFLOW
    persistent_bytes = 4 * (2 * _WORKING_SET_OVERFLOW + 7)
    update_bytes = 4 * (9 * _WORKING_SET_OVERFLOW + 8)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes <= _INT32_MAX
    assert update_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        Adam().init(_WORKING_SET_OVERFLOW)


def test_adagain_rejects_simultaneous_update_working_set() -> None:
    one_bank_bytes = 4 * _WORKING_SET_OVERFLOW
    persistent_bytes = 4 * (2 * _WORKING_SET_OVERFLOW + 4)
    update_bytes = 4 * (9 * _WORKING_SET_OVERFLOW + 8)
    assert one_bank_bytes <= _INT32_MAX
    assert persistent_bytes <= _INT32_MAX
    assert update_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="update working set byte count"):
        AdaGain().init(_WORKING_SET_OVERFLOW)


def test_legal_adam_and_adagain_update_identity_is_unchanged() -> None:
    observation = jnp.ones(4, dtype=jnp.float32)
    error = jnp.asarray(0.5, dtype=jnp.float32)

    adam = Adam(step_size=0.01)
    adam_state = adam.init(4)
    assert adam_state.m.shape == (4,)
    assert adam_state.v.shape == (4,)
    assert 4 * (2 * 4 + 7) <= _INT32_MAX
    adam_result = adam.update(adam_state, error, observation)
    assert bool(adam_result.update_applied)
    assert adam_result.weight_delta.shape == (4,)
    assert adam_result.new_state.m.shape == (4,)
    assert float(adam_result.new_state.t) == pytest.approx(1.0)

    adagain = AdaGain()
    adagain_state = adagain.init(4)
    assert adagain_state.step_sizes.shape == (4,)
    assert adagain_state.gradient_trace.shape == (4,)
    adagain_result = adagain.update(adagain_state, error, observation)
    assert bool(adagain_result.update_applied)
    assert adagain_result.weight_delta.shape == (4,)
    assert adagain_result.new_state.gradient_trace.shape == (4,)
