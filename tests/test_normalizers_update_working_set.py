"""#1383-complete update working-set preflight for online normalizers."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.normalizers import (
    EMANormalizer,
    StreamingBatchNormalizer,
    WelfordNormalizer,
    _normalizer_update_working_set_bytes,
    _preflight_normalizer_update_working_set,
    normalizer_state_nbytes_formula,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_EMA_OVERFLOW_FEATURE_DIM = 80_000_000
_EMA_LAST_FIT = 76_695_842
_EMA_FIRST_OVERFLOW = 76_695_843
_WELFORD_OVERFLOW_FEATURE_DIM = 60_000_000
_WELFORD_LAST_FIT = 53_687_089
_WELFORD_FIRST_OVERFLOW = 53_687_090


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_ema_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = normalizer_state_nbytes_formula(
        "EMANormalizer", _EMA_OVERFLOW_FEATURE_DIM
    )
    working_set_bytes = _normalizer_update_working_set_bytes(
        persist_bytes, _EMA_OVERFLOW_FEATURE_DIM
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 640_000_016
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _EMA_OVERFLOW_FEATURE_DIM <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="working set byte count"):
        EMANormalizer().init(_EMA_OVERFLOW_FEATURE_DIM)
    with pytest.raises(ValueError, match="working set byte count"):
        StreamingBatchNormalizer().init(_EMA_OVERFLOW_FEATURE_DIM)


def test_ema_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for feature_dim in range(_EMA_LAST_FIT, _EMA_FIRST_OVERFLOW + 2):
        persist_bytes = normalizer_state_nbytes_formula("EMANormalizer", feature_dim)
        working_set_bytes = _normalizer_update_working_set_bytes(
            persist_bytes, feature_dim
        )
        extras_bytes = working_set_bytes - 3 * persist_bytes
        assert persist_bytes <= _INT32_MAX
        assert persist_bytes + extras_bytes <= _INT32_MAX
        assert 4 * feature_dim <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = feature_dim
        elif first_overflow is None:
            first_overflow = feature_dim
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _EMA_LAST_FIT
    _preflight_normalizer_update_working_set(
        "EMANormalizer",
        normalizer_state_nbytes_formula("EMANormalizer", last_fit),
        last_fit,
    )
    with pytest.raises(ValueError, match="working set byte count"):
        EMANormalizer().init(first_overflow)
    with pytest.raises(ValueError, match="working set byte count"):
        StreamingBatchNormalizer().init(first_overflow)


def test_welford_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = normalizer_state_nbytes_formula(
        "WelfordNormalizer", _WELFORD_OVERFLOW_FEATURE_DIM
    )
    working_set_bytes = _normalizer_update_working_set_bytes(
        persist_bytes, _WELFORD_OVERFLOW_FEATURE_DIM
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 720_000_012
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _WELFORD_OVERFLOW_FEATURE_DIM <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="working set byte count"):
        WelfordNormalizer().init(_WELFORD_OVERFLOW_FEATURE_DIM)


def test_welford_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for feature_dim in range(_WELFORD_LAST_FIT, _WELFORD_FIRST_OVERFLOW + 2):
        persist_bytes = normalizer_state_nbytes_formula("WelfordNormalizer", feature_dim)
        working_set_bytes = _normalizer_update_working_set_bytes(
            persist_bytes, feature_dim
        )
        extras_bytes = working_set_bytes - 3 * persist_bytes
        assert persist_bytes <= _INT32_MAX
        assert persist_bytes + extras_bytes <= _INT32_MAX
        assert 4 * feature_dim <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = feature_dim
        elif first_overflow is None:
            first_overflow = feature_dim
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    assert last_fit == _WELFORD_LAST_FIT
    _preflight_normalizer_update_working_set(
        "WelfordNormalizer",
        normalizer_state_nbytes_formula("WelfordNormalizer", last_fit),
        last_fit,
    )
    with pytest.raises(ValueError, match="working set byte count"):
        WelfordNormalizer().init(first_overflow)


def test_persist_formula_still_names_int32_max() -> None:
    persist_bytes = normalizer_state_nbytes_formula("EMANormalizer", _INT32_MAX)
    assert persist_bytes == 8 * _INT32_MAX + 16
    assert persist_bytes > _INT32_MAX


def test_legal_small_normalizers_still_update() -> None:
    observation = jnp.ones((5,), dtype=jnp.float32)
    ema = EMANormalizer()
    assert normalizer_state_nbytes_formula("EMANormalizer", 5) == 56
    ema.normalize_with_diagnostics(ema.init(5), observation)
    welford = WelfordNormalizer()
    assert normalizer_state_nbytes_formula("WelfordNormalizer", 5) == 72
    welford.normalize_with_diagnostics(welford.init(5), observation)
    streaming = StreamingBatchNormalizer()
    streaming.normalize_with_diagnostics(streaming.init(5), observation)
