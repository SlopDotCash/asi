"""Reject oversized off-policy Horde hidden-size lists before layer-walk hang."""

from __future__ import annotations

import pytest

from alberta_framework.core.off_policy_horde import (
    _MAX_OFF_POLICY_HORDE_HIDDEN_LAYERS,
    OffPolicyHordeLearner,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec


def _spec():
    return create_horde_spec(
        (
            GVFSpec(
                name="demon_0",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
            ),  # type: ignore[call-arg]
        )
    )


def test_off_policy_horde_hidden_layer_cap_constant() -> None:
    assert _MAX_OFF_POLICY_HORDE_HIDDEN_LAYERS == 4096


def test_off_policy_horde_accepts_max_hidden_layer_count() -> None:
    OffPolicyHordeLearner(
        horde_spec=_spec(),
        hidden_sizes=(1,) * _MAX_OFF_POLICY_HORDE_HIDDEN_LAYERS,
    )


def test_off_policy_horde_rejects_oversized_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        OffPolicyHordeLearner(
            horde_spec=_spec(),
            hidden_sizes=(1,) * (_MAX_OFF_POLICY_HORDE_HIDDEN_LAYERS + 1),
        )


def test_off_policy_horde_from_config_rejects_oversized_hidden_list() -> None:
    payload = OffPolicyHordeLearner(horde_spec=_spec(), hidden_sizes=(4,)).to_config()
    payload["hidden_sizes"] = [1] * (_MAX_OFF_POLICY_HORDE_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        OffPolicyHordeLearner.from_config(payload)
