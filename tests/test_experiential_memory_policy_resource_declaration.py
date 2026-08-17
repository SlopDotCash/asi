"""Leftover-identity gates for experiential-memory-policy resource declarations."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alberta_framework.core.experiential_memory_policy import (
    ExperientialMemoryPolicyResourceDeclaration,
)


def _legal_declaration() -> ExperientialMemoryPolicyResourceDeclaration:
    return ExperientialMemoryPolicyResourceDeclaration(
        n_actions=3,
        owned_trainable_float32_scalars=0,
        owned_persistent_state_bytes=0,
        external_memory_persistent_state_bytes=16,
        memory_queries_per_proposal=1,
        random_draws_per_proposal=0,
        score_mass_values_interpreted_per_proposal=3,
        hard_safety_values_interpreted_per_proposal=3,
        argmax_candidates_per_proposal=3,
    )


def test_experiential_memory_policy_resource_declaration_rejects_leftover_identities() -> None:
    """Public resource declarations must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="n_actions"):
        replace(_legal_declaration(), n_actions=True)
    with pytest.raises(ValueError, match="random_draws_per_proposal"):
        replace(_legal_declaration(), random_draws_per_proposal=True)
    with pytest.raises(ValueError, match="external_memory_persistent_state_bytes"):
        replace(_legal_declaration(), external_memory_persistent_state_bytes=float("nan"))

    legal = _legal_declaration()
    dumped = json.dumps(legal.to_config(), allow_nan=False)
    assert '"n_actions": 3' in dumped
    assert '"random_draws_per_proposal": 0' in dumped
    assert '"external_memory_persistent_state_bytes": 16' in dumped
    assert '"n_actions": true' not in dumped
    assert '"random_draws_per_proposal": true' not in dumped
    assert '"external_memory_persistent_state_bytes": true' not in dumped
