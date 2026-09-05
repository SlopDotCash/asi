"""Cardinality preflights for history-feature decay rates and channels."""

from __future__ import annotations

from typing import Any

import pytest

import alberta_framework.core.history_features as history_features
from alberta_framework.core.history_features import (
    _MAX_HISTORY_CHANNELS,
    _MAX_HISTORY_DECAY_RATES,
    HistoryFeatureExtractor,
)


class _HostileList(list[object]):
    calls = 0

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("list length hook executed")


def test_last_fit_decay_count_is_accepted() -> None:
    extractor = HistoryFeatureExtractor(
        raw_dim=1,
        decay_rates=(0.5,) * _MAX_HISTORY_DECAY_RATES,
        channels=(0,),
        include_raw=False,
    )
    assert len(extractor.decay_rates) == _MAX_HISTORY_DECAY_RATES


def test_last_fit_channel_count_is_accepted() -> None:
    extractor = HistoryFeatureExtractor(
        raw_dim=_MAX_HISTORY_CHANNELS,
        decay_rates=(0.5,),
        channels=tuple(range(_MAX_HISTORY_CHANNELS)),
        include_raw=False,
    )
    assert extractor.channels is not None
    assert len(extractor.channels) == _MAX_HISTORY_CHANNELS


def test_rejects_oversized_decay_rates_before_per_rate_walk() -> None:
    calls = 0

    class HostileFloat:
        def __float__(self) -> float:
            nonlocal calls
            calls += 1
            raise AssertionError("oversized decay tuple walked an element")

    hostile: Any = HostileFloat()
    with pytest.raises(ValueError, match="at most 4096"):
        HistoryFeatureExtractor(raw_dim=1, decay_rates=(hostile,) * 4097)
    assert calls == 0


def test_rejects_oversized_channels_before_per_index_walk() -> None:
    calls = 0

    class HostileIndex:
        def __index__(self) -> int:
            nonlocal calls
            calls += 1
            raise AssertionError("oversized channel tuple walked an element")

    hostile: Any = HostileIndex()
    with pytest.raises(ValueError, match="at most 4096"):
        HistoryFeatureExtractor(
            raw_dim=1,
            decay_rates=(0.5,),
            channels=(hostile,) * 4097,
        )
    assert calls == 0


def test_default_channels_reject_before_range_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def range_must_not_run(*args: object) -> range:
        raise AssertionError("default channel range expanded before its count bound")

    monkeypatch.setattr(history_features, "range", range_must_not_run, raising=False)
    with pytest.raises(ValueError, match="channels must contain at most 4096"):
        HistoryFeatureExtractor(
            raw_dim=_MAX_HISTORY_CHANNELS + 1,
            decay_rates=(0.5,),
            channels=None,
            include_raw=False,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decay_rates", [0.5] * 4097),
        ("channels", [0] * 4097),
    ],
)
def test_from_config_rejects_oversized_lists_before_tuple_copy(
    field: str, value: list[object]
) -> None:
    config = HistoryFeatureExtractor(raw_dim=1).to_config()
    config[field] = value
    with pytest.raises(ValueError, match="at most 4096"):
        HistoryFeatureExtractor.from_config(config)


@pytest.mark.parametrize("field", ["decay_rates", "channels"])
def test_from_config_rejects_list_subclasses_before_length_hooks(field: str) -> None:
    config = HistoryFeatureExtractor(raw_dim=1).to_config()
    config[field] = _HostileList()
    _HostileList.calls = 0
    with pytest.raises(ValueError, match="actual tuple"):
        HistoryFeatureExtractor.from_config(config)
    assert _HostileList.calls == 0
