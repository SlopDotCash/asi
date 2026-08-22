"""Reject oversized Step 4 SARSA hidden-size lists before layer-walk hang."""

from __future__ import annotations

import pytest

from alberta_framework.steps.step4 import _MAX_STEP4_HIDDEN_LAYERS, Step4SARSAConfig


def test_step4_hidden_layer_cap_constant() -> None:
    assert _MAX_STEP4_HIDDEN_LAYERS == 4096


def test_step4_accepts_max_hidden_layer_count() -> None:
    Step4SARSAConfig(n_actions=2, hidden_sizes=(1,) * _MAX_STEP4_HIDDEN_LAYERS)


def test_step4_rejects_oversized_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        Step4SARSAConfig(
            n_actions=2,
            hidden_sizes=(1,) * (_MAX_STEP4_HIDDEN_LAYERS + 1),
        )


def test_step4_from_dict_rejects_oversized_hidden_list() -> None:
    payload = Step4SARSAConfig(n_actions=2, hidden_sizes=(4,)).to_dict()
    payload["hidden_sizes"] = [1] * (_MAX_STEP4_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        Step4SARSAConfig.from_dict(payload)
