"""Reject oversized temporal-context period lists before per-period validation hangs."""

from __future__ import annotations

import pytest

from alberta_framework.core.temporal_context import (
    _MAX_TEMPORAL_CONTEXT_PERIODS,
    TemporalContextConfig,
)


def test_temporal_period_cap_constant() -> None:
    assert _MAX_TEMPORAL_CONTEXT_PERIODS == 4096


def test_temporal_accepts_max_period_count() -> None:
    TemporalContextConfig(
        input_dim=1,
        include_phase_products=False,
        periods=(50.0,) * _MAX_TEMPORAL_CONTEXT_PERIODS,
    )


def test_temporal_rejects_oversized_period_count() -> None:
    with pytest.raises(ValueError, match="periods length"):
        TemporalContextConfig(
            input_dim=1,
            periods=(50.0,) * (_MAX_TEMPORAL_CONTEXT_PERIODS + 1),
        )


def test_temporal_from_config_rejects_oversized_period_list() -> None:
    payload = TemporalContextConfig(input_dim=1).to_config()
    payload["periods"] = [50.0] * (_MAX_TEMPORAL_CONTEXT_PERIODS + 1)
    with pytest.raises(ValueError, match="periods length"):
        TemporalContextConfig.from_config(payload)
