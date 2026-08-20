"""Reject oversized multi-head hidden-size lists before per-layer validation hangs."""

from __future__ import annotations

import pytest

from alberta_framework.core.multi_head_learner import (
    _MAX_MULTI_HEAD_HIDDEN_LAYERS,
    MultiHeadMLPLearner,
)


def test_multi_head_hidden_layer_cap_constant() -> None:
    assert _MAX_MULTI_HEAD_HIDDEN_LAYERS == 4096


def test_multi_head_accepts_max_hidden_layer_count() -> None:
    MultiHeadMLPLearner(n_heads=1, hidden_sizes=(1,) * _MAX_MULTI_HEAD_HIDDEN_LAYERS)


def test_multi_head_rejects_oversized_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        MultiHeadMLPLearner(
            n_heads=1, hidden_sizes=(1,) * (_MAX_MULTI_HEAD_HIDDEN_LAYERS + 1)
        )


def test_multi_head_from_config_rejects_oversized_hidden_list() -> None:
    payload = MultiHeadMLPLearner(n_heads=1, hidden_sizes=(4,)).to_config()
    payload["hidden_sizes"] = [1] * (_MAX_MULTI_HEAD_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        MultiHeadMLPLearner.from_config(payload)
