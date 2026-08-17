"""Leftover-identity gates for lifetime gauntlet resource-budget records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.streams.gauntlet import LifetimeGauntletResourceBudget


def test_gauntlet_resource_budget_rejects_leftover_identities() -> None:
    """Public resource-budget records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="state_nbytes"):
        LifetimeGauntletResourceBudget(
            state_nbytes=True,
            exact_clock_nbytes=12,
            exact_clock_delta_nbytes=8,
        )
    with pytest.raises(ValueError, match="trainable_scalars"):
        LifetimeGauntletResourceBudget(
            state_nbytes=1,
            exact_clock_nbytes=12,
            exact_clock_delta_nbytes=8,
            trainable_scalars=True,
        )
    with pytest.raises(ValueError, match="replay_capacity"):
        LifetimeGauntletResourceBudget(
            state_nbytes=1,
            exact_clock_nbytes=12,
            exact_clock_delta_nbytes=8,
            replay_capacity=True,
        )
    with pytest.raises(ValueError, match="state_nbytes"):
        LifetimeGauntletResourceBudget(
            state_nbytes=float("nan"),
            exact_clock_nbytes=12,
            exact_clock_delta_nbytes=8,
        )

    legal = LifetimeGauntletResourceBudget(
        state_nbytes=1,
        exact_clock_nbytes=12,
        exact_clock_delta_nbytes=8,
    )
    dumped = json.dumps(legal.to_dict(), allow_nan=False)
    assert '"state_nbytes": 1' in dumped
    assert '"exact_clock_nbytes": 12' in dumped
    assert '"trainable_scalars": 0' in dumped
    assert '"replay_capacity": 0' in dumped
    assert '"state_nbytes": true' not in dumped
    assert '"trainable_scalars": true' not in dumped
    assert '"replay_capacity": true' not in dumped
