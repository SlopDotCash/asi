"""Leftover-identity gates for world-model-ensemble resource-budget records."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alberta_framework.core.world_model_ensemble import WorldModelEnsembleResourceBudget


def _legal_budget() -> WorldModelEnsembleResourceBudget:
    return WorldModelEnsembleResourceBudget(
        ensemble_size=2,
        observation_dim=2,
        target_dim=4,
        member_state_scalars_per_member=61,
        member_state_bytes_per_member=244,
        member_trainable_scalars=20,
        total_trainable_scalars=40,
        persistent_float32_scalars=127,
        persistent_float64_scalars=0,
        persistent_int32_scalars=14,
        persistent_int64_scalars=0,
        persistent_uint32_scalars=8,
        persistent_bool_scalars=4,
        persistent_state_scalars=153,
        persistent_state_bytes=600,
        bootstrap_prng_keys=2,
        bootstrap_prng_uint32_scalars=4,
        bootstrap_prng_bytes=16,
        prediction_output_logical_scalars=39,
        prediction_output_logical_bytes=150,
        update_result_output_logical_scalars=232,
        update_result_output_logical_bytes=844,
        replay_update_result_output_logical_scalars=213,
        replay_update_result_output_logical_bytes=792,
        member_update_candidates_per_valid_event=2,
        max_member_updates_per_event=2,
        replay_member_update_candidates_per_available_sample=2,
        max_replay_member_updates_per_available_sample=2,
        max_event_count=2_147_483_647,
        max_member_update_count=2_147_483_647,
        max_replay_event_count=2_147_483_647,
        max_replay_member_update_count=2_147_483_647,
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
    assert '"persistent_state_bytes": 600' in dumped
    assert '"ensemble_size": true' not in dumped
    assert '"replay_capacity": true' not in dumped
    assert '"persistent_state_bytes": true' not in dumped


@pytest.mark.parametrize(
    "field",
    [
        "ensemble_size",
        "observation_dim",
        "target_dim",
        "member_state_scalars_per_member",
        "member_state_bytes_per_member",
        "member_trainable_scalars",
        "total_trainable_scalars",
        "persistent_float32_scalars",
        "persistent_float64_scalars",
        "persistent_int32_scalars",
        "persistent_int64_scalars",
        "persistent_uint32_scalars",
        "persistent_bool_scalars",
        "persistent_state_scalars",
        "persistent_state_bytes",
        "bootstrap_prng_keys",
        "bootstrap_prng_uint32_scalars",
        "bootstrap_prng_bytes",
        "prediction_output_logical_scalars",
        "prediction_output_logical_bytes",
        "update_result_output_logical_scalars",
        "update_result_output_logical_bytes",
        "replay_update_result_output_logical_scalars",
        "replay_update_result_output_logical_bytes",
        "member_update_candidates_per_valid_event",
        "max_member_updates_per_event",
        "replay_member_update_candidates_per_available_sample",
        "max_replay_member_updates_per_available_sample",
        "max_event_count",
        "max_member_update_count",
        "max_replay_event_count",
        "max_replay_member_update_count",
        "replay_capacity",
    ],
)
def test_world_model_ensemble_budget_rejects_false_exact_identity(field: str) -> None:
    legal = _legal_budget()
    with pytest.raises(ValueError):
        replace(legal, **{field: getattr(legal, field) + 1})


class _HostileInt(int):
    def __index__(self) -> int:
        raise AssertionError("hostile index hook executed")


def test_world_model_ensemble_budget_rejects_hostile_integer_without_hook() -> None:
    with pytest.raises(ValueError, match="ensemble_size"):
        replace(_legal_budget(), ensemble_size=_HostileInt(2))
