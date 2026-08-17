"""Leftover-identity gates for model-replay resource-budget records."""

from __future__ import annotations

import dataclasses
import json

import pytest

from alberta_framework.core.model_replay_rehearsal import ModelReplayRehearsalResourceBudget


def _legal_budget() -> ModelReplayRehearsalResourceBudget:
    return ModelReplayRehearsalResourceBudget(
        persistent_state_scalars=15,
        persistent_state_bytes=60,
        ensemble_state_bytes=20,
        replay_state_bytes=12,
        composer_accounting_bytes=28,
        replay_total_capacity=4,
        short_term_capacity=2,
        long_term_capacity=2,
        fixed_replay_quota=1,
        max_real_model_update_candidates_per_event=2,
        max_replay_model_update_candidates_per_event=2,
        max_total_model_update_candidates_per_event=4,
        max_actor_updates_per_event=0,
        max_critic_updates_per_event=0,
        max_state_builder_updates_per_event=0,
        max_real_event_count=2_147_483_647,
        max_rehearsal_attempt_count=2_147_483_647,
    )


def test_model_replay_resource_budget_rejects_leftover_identities() -> None:
    """Public resource-budget records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="persistent_state_scalars"):
        ModelReplayRehearsalResourceBudget(
            persistent_state_scalars=True,
            persistent_state_bytes=40,
            ensemble_state_bytes=20,
            replay_state_bytes=12,
            composer_accounting_bytes=8,
            replay_total_capacity=4,
            short_term_capacity=2,
            long_term_capacity=2,
            fixed_replay_quota=1,
            max_real_model_update_candidates_per_event=2,
            max_replay_model_update_candidates_per_event=2,
            max_total_model_update_candidates_per_event=4,
            max_actor_updates_per_event=0,
            max_critic_updates_per_event=0,
            max_state_builder_updates_per_event=0,
            max_real_event_count=100,
            max_rehearsal_attempt_count=100,
        )
    with pytest.raises(ValueError, match="max_actor_updates_per_event"):
        ModelReplayRehearsalResourceBudget(
            persistent_state_scalars=10,
            persistent_state_bytes=40,
            ensemble_state_bytes=20,
            replay_state_bytes=12,
            composer_accounting_bytes=8,
            replay_total_capacity=4,
            short_term_capacity=2,
            long_term_capacity=2,
            fixed_replay_quota=1,
            max_real_model_update_candidates_per_event=2,
            max_replay_model_update_candidates_per_event=2,
            max_total_model_update_candidates_per_event=4,
            max_actor_updates_per_event=True,
            max_critic_updates_per_event=0,
            max_state_builder_updates_per_event=0,
            max_real_event_count=100,
            max_rehearsal_attempt_count=100,
        )
    with pytest.raises(ValueError, match="persistent_state_bytes"):
        ModelReplayRehearsalResourceBudget(
            persistent_state_scalars=10,
            persistent_state_bytes=float("nan"),
            ensemble_state_bytes=20,
            replay_state_bytes=12,
            composer_accounting_bytes=8,
            replay_total_capacity=4,
            short_term_capacity=2,
            long_term_capacity=2,
            fixed_replay_quota=1,
            max_real_model_update_candidates_per_event=2,
            max_replay_model_update_candidates_per_event=2,
            max_total_model_update_candidates_per_event=4,
            max_actor_updates_per_event=0,
            max_critic_updates_per_event=0,
            max_state_builder_updates_per_event=0,
            max_real_event_count=100,
            max_rehearsal_attempt_count=100,
        )

    legal = _legal_budget()
    dumped = json.dumps(legal.to_config(), allow_nan=False)
    assert '"persistent_state_scalars": 15' in dumped
    assert '"max_actor_updates_per_event": 0' in dumped
    assert '"persistent_state_bytes": 60' in dumped
    assert '"persistent_state_scalars": true' not in dumped
    assert '"max_actor_updates_per_event": true' not in dumped
    assert '"persistent_state_bytes": true' not in dumped


@pytest.mark.parametrize(
    "field",
    [
        "persistent_state_bytes",
        "ensemble_state_bytes",
        "replay_state_bytes",
        "composer_accounting_bytes",
        "replay_total_capacity",
        "short_term_capacity",
        "long_term_capacity",
        "fixed_replay_quota",
        "max_real_model_update_candidates_per_event",
        "max_replay_model_update_candidates_per_event",
        "max_total_model_update_candidates_per_event",
        "max_actor_updates_per_event",
        "max_critic_updates_per_event",
        "max_state_builder_updates_per_event",
        "max_real_event_count",
        "max_rehearsal_attempt_count",
    ],
)
def test_model_replay_budget_rejects_false_exact_identities(field: str) -> None:
    legal = _legal_budget()
    with pytest.raises(ValueError):
        dataclasses.replace(legal, **{field: getattr(legal, field) + 1})


class _HostileInt(int):
    def __index__(self) -> int:
        raise AssertionError("hostile index hook executed")


def test_model_replay_budget_rejects_hostile_integer_without_hook() -> None:
    with pytest.raises(ValueError, match="persistent_state_scalars"):
        dataclasses.replace(_legal_budget(), persistent_state_scalars=_HostileInt(15))
