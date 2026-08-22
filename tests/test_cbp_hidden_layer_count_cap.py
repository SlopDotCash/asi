"""Reject oversized CBP hidden-size lists before per-layer validation hangs."""

from __future__ import annotations

import pytest

from alberta_framework.core.continual_backprop import (
    _MAX_CBP_HIDDEN_LAYERS,
    CBPMultiHeadMLPLearner,
)


def test_cbp_hidden_layer_cap_constant() -> None:
    assert _MAX_CBP_HIDDEN_LAYERS == 4096


def test_cbp_accepts_max_hidden_layer_count() -> None:
    CBPMultiHeadMLPLearner(n_heads=1, hidden_sizes=(1,) * _MAX_CBP_HIDDEN_LAYERS)


def test_cbp_rejects_oversized_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        CBPMultiHeadMLPLearner(
            n_heads=1,
            hidden_sizes=(1,) * (_MAX_CBP_HIDDEN_LAYERS + 1),
        )


def test_cbp_from_config_rejects_oversized_hidden_list() -> None:
    payload = CBPMultiHeadMLPLearner(n_heads=1, hidden_sizes=(4,)).to_config()
    payload["hidden_sizes"] = [1] * (_MAX_CBP_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        CBPMultiHeadMLPLearner.from_config(payload)
