"""Overflow preflight keeps history-feature trace allocation in signed int32."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.history_features import HistoryFeatureExtractor

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_TRACE_BYTE_OVERFLOW_CHANNELS = _INT32_MAX // 4 + 1


def test_int32_wrap_forges_a_different_trace_allocation_identity() -> None:
    feature_width = _TRACE_BYTE_OVERFLOW_CHANNELS
    assert feature_width <= _INT32_MAX
    trace_bytes = 4 * feature_width
    assert trace_bytes == _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != trace_bytes


def test_extractor_rejects_trace_bank_byte_overflow() -> None:
    with pytest.raises(ValueError, match="byte count must fit signed int32"):
        HistoryFeatureExtractor(
            raw_dim=_TRACE_BYTE_OVERFLOW_CHANNELS,
            decay_rates=(0.5,),
            include_raw=False,
        )


def test_width_can_fit_while_output_bytes_do_not() -> None:
    with pytest.raises(ValueError, match="output byte count"):
        HistoryFeatureExtractor(
            raw_dim=_INT32_MAX - 1,
            decay_rates=(0.5,),
            channels=(0,),
        )


def test_step_working_set_is_preflighted_before_default_channel_expansion() -> None:
    # Trace/output banks each fit. The simultaneous trace temporaries and raw
    # observation vectors do not; rejection must precede tuple(range(raw_dim)).
    with pytest.raises(ValueError, match="step working set"):
        HistoryFeatureExtractor(
            raw_dim=40_000_000,
            decay_rates=(0.5,),
            include_raw=False,
        )


def test_legal_step_allocation_identity_is_unchanged() -> None:
    extractor = HistoryFeatureExtractor(
        raw_dim=4,
        decay_rates=(0.5, 0.9),
        include_raw=False,
    )
    assert extractor.feature_dim() == 8
    state = extractor.init()
    assert state.traces.shape == (2, 4)
    assert 4 * int(state.traces.size) <= _INT32_MAX
    augmented, advanced = extractor.step(state, jnp.ones(4, dtype=jnp.float32))
    assert augmented.shape == (8,)
    assert advanced.traces.shape == (2, 4)


@pytest.mark.parametrize(
    "bad_traces",
    [
        jnp.zeros((1, 4), dtype=jnp.float32),
        jnp.zeros((2, 4), dtype=jnp.float16),
    ],
)
def test_step_rejects_malformed_state_before_computation(bad_traces: jnp.ndarray) -> None:
    extractor = HistoryFeatureExtractor(raw_dim=4, decay_rates=(0.5, 0.9))
    state = extractor.init().replace(traces=bad_traces)
    with pytest.raises(ValueError, match="state.traces"):
        extractor.step(state, jnp.ones(4, dtype=jnp.float32))
