"""Reject oversized FeatureBankRouter consumer PyTrees before leaf-walk hang."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.core.feature_bank_router import (
    _MAX_CONSUMER_LEAVES,
    FeatureBankRouter,
    FeatureBankRouterConfig,
)


def _router() -> FeatureBankRouter:
    return FeatureBankRouter(FeatureBankRouterConfig(base_dim=2, active_slots=1))


def test_feature_bank_router_leaf_cap_constant() -> None:
    assert _MAX_CONSUMER_LEAVES == 4096


def test_feature_bank_router_accepts_max_consumer_leaves() -> None:
    width = _router()._config.total_feature_dim
    consumers = tuple(np.zeros((width,), dtype=np.float32) for _ in range(_MAX_CONSUMER_LEAVES))
    arrays, _, axes = _router()._consumer_layout(consumers, None)
    assert len(arrays) == _MAX_CONSUMER_LEAVES
    assert axes == (0,) * _MAX_CONSUMER_LEAVES


def test_feature_bank_router_rejects_oversized_consumer_leaves() -> None:
    width = _router()._config.total_feature_dim
    consumers = tuple(
        np.zeros((width,), dtype=np.float32) for _ in range(_MAX_CONSUMER_LEAVES + 1)
    )
    with pytest.raises(ValueError, match="consumer leaf count"):
        _router()._consumer_layout(consumers, None)
