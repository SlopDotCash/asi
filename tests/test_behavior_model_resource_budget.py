"""Leftover-identity gates for behavior-model resource-budget records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.core.behavior_model import BehaviorModelResourceBudget


def _legal_budget() -> BehaviorModelResourceBudget:
    return BehaviorModelResourceBudget(
        feature_dim=5,
        n_actions=3,
        trainable_float32_scalars=18,
        diagnostic_float32_scalars=3,
        administrative_int32_scalars=1,
        rng_uint32_scalars=2,
        state_nbytes=96,
        learned_float32_scalars_touched_per_update=21,
        replay_capacity=0,
    )


def test_behavior_model_resource_budget_rejects_leftover_identities() -> None:
    """Public resource-budget records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="feature_dim"):
        BehaviorModelResourceBudget(
            feature_dim=True,
            n_actions=3,
            trainable_float32_scalars=18,
            diagnostic_float32_scalars=3,
            administrative_int32_scalars=1,
            rng_uint32_scalars=2,
            state_nbytes=96,
            learned_float32_scalars_touched_per_update=21,
            replay_capacity=0,
        )
    with pytest.raises(ValueError, match="replay_capacity"):
        BehaviorModelResourceBudget(
            feature_dim=5,
            n_actions=3,
            trainable_float32_scalars=18,
            diagnostic_float32_scalars=3,
            administrative_int32_scalars=1,
            rng_uint32_scalars=2,
            state_nbytes=96,
            learned_float32_scalars_touched_per_update=21,
            replay_capacity=True,
        )
    with pytest.raises(ValueError, match="state_nbytes"):
        BehaviorModelResourceBudget(
            feature_dim=5,
            n_actions=3,
            trainable_float32_scalars=18,
            diagnostic_float32_scalars=3,
            administrative_int32_scalars=1,
            rng_uint32_scalars=2,
            state_nbytes=float("nan"),
            learned_float32_scalars_touched_per_update=21,
            replay_capacity=0,
        )

    legal = _legal_budget()
    dumped = json.dumps(legal.to_dict(), allow_nan=False)
    assert '"feature_dim": 5' in dumped
    assert '"n_actions": 3' in dumped
    assert '"replay_capacity": 0' in dumped
    assert '"feature_dim": true' not in dumped
    assert '"replay_capacity": true' not in dumped
    assert '"state_nbytes": true' not in dumped
