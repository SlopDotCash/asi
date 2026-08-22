"""Reject oversized Step 8 world-model hidden-size lists before layer-walk hang."""

from __future__ import annotations

import pytest

from alberta_framework.steps.step8 import _MAX_STEP8_HIDDEN_LAYERS, Step8WorldModelConfig


def test_step8_hidden_layer_cap_constant() -> None:
    assert _MAX_STEP8_HIDDEN_LAYERS == 4096


def test_step8_accepts_max_hidden_layer_count() -> None:
    Step8WorldModelConfig(hidden_sizes=(1,) * _MAX_STEP8_HIDDEN_LAYERS)


def test_step8_rejects_oversized_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        Step8WorldModelConfig(hidden_sizes=(1,) * (_MAX_STEP8_HIDDEN_LAYERS + 1))


def test_step8_from_dict_rejects_oversized_hidden_list() -> None:
    payload = Step8WorldModelConfig(hidden_sizes=(4,)).to_dict()
    payload["hidden_sizes"] = [1] * (_MAX_STEP8_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        Step8WorldModelConfig.from_dict(payload)
