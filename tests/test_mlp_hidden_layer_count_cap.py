"""Reject oversized MLPLearner hidden-size lists before per-layer validation hangs."""

from __future__ import annotations

import pytest

from alberta_framework.core.learners import _MAX_MLP_HIDDEN_LAYERS, MLPLearner


def test_mlp_hidden_layer_cap_constant() -> None:
    assert _MAX_MLP_HIDDEN_LAYERS == 4096


def test_mlp_accepts_max_hidden_layer_count() -> None:
    MLPLearner(hidden_sizes=(1,) * _MAX_MLP_HIDDEN_LAYERS)


def test_mlp_rejects_oversized_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        MLPLearner(hidden_sizes=(1,) * (_MAX_MLP_HIDDEN_LAYERS + 1))


def test_mlp_from_config_rejects_oversized_hidden_list() -> None:
    payload = MLPLearner(hidden_sizes=(4,)).to_config()
    payload["hidden_sizes"] = [1] * (_MAX_MLP_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        MLPLearner.from_config(payload)
