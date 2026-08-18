"""#1383-complete step working-set preflight for synthetic streams."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.streams.synthetic import (
    DynamicScaleShiftStream,
    HiddenStateAR2Stream,
    ScaleDriftStream,
    SuttonExperiment1Stream,
    _hidden_state_ar2_persistent_bytes,
    _preflight_synthetic_step_working_set,
    _require_float32_resource,
    _scale_stream_persistent_bytes,
    _sutton_experiment1_persistent_bytes,
    _synthetic_step_working_set_bytes,
)

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_OVERFLOW_FEATURE_DIM = 80_000_000
_HIDDEN_LAST_FIT = 76_695_843
_HIDDEN_FIRST_OVERFLOW = 76_695_844
_SCALE_LAST_FIT = 76_695_843
_SCALE_FIRST_OVERFLOW = 76_695_844
_SUTTON_OVERFLOW_FEATURE_DIM = 200_000_000
_SUTTON_LAST_FIT = 134_217_725
_SUTTON_FIRST_OVERFLOW = 134_217_726


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_hidden_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _hidden_state_ar2_persistent_bytes(_OVERFLOW_FEATURE_DIM)
    working_set_bytes = _synthetic_step_working_set_bytes(
        persist_bytes, _OVERFLOW_FEATURE_DIM
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 640_000_008
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_FEATURE_DIM <= _INT32_MAX
    assert 4 * 1 <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="working set byte count"):
        HiddenStateAR2Stream(_OVERFLOW_FEATURE_DIM, visible_dim=1)


def test_hidden_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for feature_dim in range(_HIDDEN_LAST_FIT, _HIDDEN_FIRST_OVERFLOW + 2):
        persist_bytes = _hidden_state_ar2_persistent_bytes(feature_dim)
        working_set_bytes = _synthetic_step_working_set_bytes(persist_bytes, feature_dim)
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
    assert last_fit == _HIDDEN_LAST_FIT
    _preflight_synthetic_step_working_set(
        "HiddenStateAR2Stream",
        _hidden_state_ar2_persistent_bytes(last_fit),
        last_fit,
    )
    with pytest.raises(ValueError, match="working set byte count"):
        HiddenStateAR2Stream(first_overflow, visible_dim=1)


def test_scale_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _scale_stream_persistent_bytes(_OVERFLOW_FEATURE_DIM)
    working_set_bytes = _synthetic_step_working_set_bytes(
        persist_bytes, _OVERFLOW_FEATURE_DIM
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 640_000_012
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _OVERFLOW_FEATURE_DIM <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="working set byte count"):
        DynamicScaleShiftStream(_OVERFLOW_FEATURE_DIM)
    with pytest.raises(ValueError, match="working set byte count"):
        ScaleDriftStream(_OVERFLOW_FEATURE_DIM)


def test_scale_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for feature_dim in range(_SCALE_LAST_FIT, _SCALE_FIRST_OVERFLOW + 2):
        persist_bytes = _scale_stream_persistent_bytes(feature_dim)
        working_set_bytes = _synthetic_step_working_set_bytes(persist_bytes, feature_dim)
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
    assert last_fit == _SCALE_LAST_FIT
    _preflight_synthetic_step_working_set(
        "DynamicScaleShiftStream",
        _scale_stream_persistent_bytes(last_fit),
        last_fit,
    )
    with pytest.raises(ValueError, match="working set byte count"):
        DynamicScaleShiftStream(first_overflow)
    with pytest.raises(ValueError, match="working set byte count"):
        ScaleDriftStream(first_overflow)


def test_sutton_named_persist_and_width_still_fit_at_overflow() -> None:
    persist_bytes = _sutton_experiment1_persistent_bytes(_SUTTON_OVERFLOW_FEATURE_DIM)
    working_set_bytes = _synthetic_step_working_set_bytes(
        persist_bytes, _SUTTON_OVERFLOW_FEATURE_DIM
    )
    extras_bytes = working_set_bytes - 3 * persist_bytes
    assert persist_bytes == 800_000_012
    assert persist_bytes <= _INT32_MAX
    assert persist_bytes + extras_bytes <= _INT32_MAX
    assert 4 * _SUTTON_OVERFLOW_FEATURE_DIM <= _INT32_MAX
    assert working_set_bytes > _INT32_MAX
    with pytest.raises(ValueError, match="working set byte count"):
        SuttonExperiment1Stream(
            num_relevant=1, num_irrelevant=_SUTTON_OVERFLOW_FEATURE_DIM - 1
        )


def test_sutton_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for feature_dim in range(_SUTTON_LAST_FIT, _SUTTON_FIRST_OVERFLOW + 2):
        persist_bytes = _sutton_experiment1_persistent_bytes(feature_dim)
        working_set_bytes = _synthetic_step_working_set_bytes(persist_bytes, feature_dim)
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
    assert last_fit == _SUTTON_LAST_FIT
    _preflight_synthetic_step_working_set(
        "SuttonExperiment1Stream",
        _sutton_experiment1_persistent_bytes(last_fit),
        last_fit,
    )
    with pytest.raises(ValueError, match="working set byte count"):
        SuttonExperiment1Stream(num_relevant=1, num_irrelevant=first_overflow - 1)


def test_persist_bound_still_fires_before_working_set() -> None:
    last_legal = (2**29 - 1 - 3) // 2
    _require_float32_resource(
        "DynamicScaleShiftStream state",
        vector_scalars=2 * last_legal,
        fixed_scalars=3,
    )
    with pytest.raises(ValueError, match="byte count"):
        DynamicScaleShiftStream(feature_dim=last_legal + 1)
    hidden_last_legal = (2**29 - 1 - 2) // 2
    _require_float32_resource(
        "HiddenStateAR2Stream state",
        vector_scalars=2 * hidden_last_legal,
        fixed_scalars=2,
    )
    with pytest.raises(ValueError, match="byte count"):
        HiddenStateAR2Stream(feature_dim=hidden_last_legal + 1, visible_dim=2)
    last_legal_total = 2**29 - 1 - 3
    _require_float32_resource(
        "SuttonExperiment1Stream state",
        vector_scalars=last_legal_total,
        fixed_scalars=3,
    )
    with pytest.raises(ValueError, match="byte count"):
        SuttonExperiment1Stream(num_relevant=1, num_irrelevant=last_legal_total)


def test_legal_small_synthetic_streams_still_step() -> None:
    assert _hidden_state_ar2_persistent_bytes(8) == 72
    hidden = HiddenStateAR2Stream(8, visible_dim=2)
    hidden.step(hidden.init(jr.key(0)), jnp.asarray(0, dtype=jnp.int32))
    assert _scale_stream_persistent_bytes(4) == 44
    dynamic = DynamicScaleShiftStream(4)
    dynamic.step(dynamic.init(jr.key(1)), jnp.asarray(0, dtype=jnp.int32))
    drift = ScaleDriftStream(4)
    drift.step(drift.init(jr.key(2)), jnp.asarray(0, dtype=jnp.int32))
    assert _sutton_experiment1_persistent_bytes(5) == 32
    sutton = SuttonExperiment1Stream(num_relevant=2, num_irrelevant=3)
    sutton.step(sutton.init(jr.key(3)), jnp.asarray(0, dtype=jnp.int32))
