"""Reject oversized independent-Horde hidden-size lists before layer-walk hang."""

from __future__ import annotations

import pytest

from alberta_framework.core.independent_demon_horde import (
    _MAX_INDEPENDENT_HORDE_HIDDEN_LAYERS,
    IndependentDemonHorde,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec


def _spec() -> object:
    return create_horde_spec(
        (
            GVFSpec(
                name="d0",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
            ),
        )
    )


def test_independent_horde_hidden_layer_cap_constant() -> None:
    assert _MAX_INDEPENDENT_HORDE_HIDDEN_LAYERS == 4096


def test_independent_horde_accepts_max_hidden_layer_count() -> None:
    IndependentDemonHorde(
        horde_spec=_spec(),  # type: ignore[arg-type]
        hidden_sizes=(1,) * _MAX_INDEPENDENT_HORDE_HIDDEN_LAYERS,
        sparsity=0.0,
    )


def test_independent_horde_rejects_oversized_hidden_layer_count() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length"):
        IndependentDemonHorde(
            horde_spec=_spec(),  # type: ignore[arg-type]
            hidden_sizes=(1,) * (_MAX_INDEPENDENT_HORDE_HIDDEN_LAYERS + 1),
            sparsity=0.0,
        )


def test_independent_horde_from_config_rejects_oversized_hidden_list() -> None:
    horde = IndependentDemonHorde(horde_spec=_spec(), hidden_sizes=(4,), sparsity=0.0)  # type: ignore[arg-type]
    payload = horde.to_config()
    payload["hidden_sizes"] = [1] * (_MAX_INDEPENDENT_HORDE_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="hidden_sizes length"):
        IndependentDemonHorde.from_config(payload)
