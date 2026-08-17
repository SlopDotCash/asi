"""Leftover-identity gates for world-model-ensemble resource-budget records."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alberta_framework.core.world_model_ensemble import WorldModelEnsembleResourceBudget


def _legal_budget() -> WorldModelEnsembleResourceBudget:
    return WorldModelEnsembleResourceBudget(
        ensemble_size=2,
        observation_dim=3,
        target_dim=4,
        member_state_scalars_per_member=5,
        member_state_bytes_per_member=20,
        member_trainable_scalars=6,
        total_trainable_scalars=12,
        persistent_float32_scalars=8,
        persistent_float64_scalars=0,
        persistent_int32_scalars=4,
        persistent_int64_scalars=0,
        persistent_uint32_scalars=4,
        persistent_bool_scalars=2,
        persistent_state_scalars=18,
        persistent_state_bytes=72,
        bootstrap_prng_keys=2,
        bootstrap_prng_uint32_scalars=4,
        bootstrap_prng_bytes=16,
        prediction_output_logical_scalars=10,
        prediction_output_logical_bytes=40,
        update_result_output_logical_scalars=20,
        update_result_output_logical_bytes=80,
        replay_update_result_output_logical_scalars=20,
        replay_update_result_output_logical_bytes=80,
        member_update_candidates_per_valid_event=2,
        max_member_updates_per_event=2,
        replay_member_update_candidates_per_available_sample=2,
        max_replay_member_updates_per_available_sample=2,
        max_event_count=100,
        max_member_update_count=100,
        max_replay_event_count=100,
        max_replay_member_update_count=100,
        replay_capacity=0,
    )


def test_world_model_ensemble_resource_budget_rejects_leftover_identities() -> None:
    """Public resource-budget records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="ensemble_size"):
        replace(_legal_budget(), ensemble_size=True)
    with pytest.raises(ValueError, match="replay_capacity"):
        replace(_legal_budget(), replay_capacity=True)
    with pytest.raises(ValueError, match="persistent_state_bytes"):
        replace(_legal_budget(), persistent_state_bytes=float("nan"))

    legal = _legal_budget()
    dumped = json.dumps(legal.to_config(), allow_nan=False)
    assert '"ensemble_size": 2' in dumped
    assert '"replay_capacity": 0' in dumped
    assert '"persistent_state_bytes": 72' in dumped
    assert '"ensemble_size": true' not in dumped
    assert '"replay_capacity": true' not in dumped
    assert '"persistent_state_bytes": true' not in dumped
