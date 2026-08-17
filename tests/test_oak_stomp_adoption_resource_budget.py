"""Leftover-identity gates for OaK STOMP-adoption resource-budget records."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alberta_framework.core.oak import OaKExternalSTOMPAdoptionResourceBudget


def _legal_budget() -> OaKExternalSTOMPAdoptionResourceBudget:
    return OaKExternalSTOMPAdoptionResourceBudget(
        persistent_state_nbytes_before=64,
        persistent_state_nbytes_after=64,
        persistent_state_growth_bytes=0,
        stomp_update_evaluations_per_adopt=0,
        stomp_update_evaluations_per_delegated_update=1,
        derivation_recomputed_on_adopt=False,
        source_result_integrity_checked=True,
        caller_authority_required=True,
        caller_authenticated=False,
    )


def test_oak_stomp_adoption_resource_budget_rejects_leftover_identities() -> None:
    """Public resource-budget records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="persistent_state_nbytes_before"):
        replace(_legal_budget(), persistent_state_nbytes_before=True)
    with pytest.raises(ValueError, match="stomp_update_evaluations_per_delegated_update"):
        replace(_legal_budget(), stomp_update_evaluations_per_delegated_update=True)
    with pytest.raises(ValueError, match="persistent_state_growth_bytes"):
        replace(_legal_budget(), persistent_state_growth_bytes=float("nan"))
    with pytest.raises(ValueError, match="caller_authenticated"):
        replace(_legal_budget(), caller_authenticated=1)
    with pytest.raises(ValueError, match="source_result_integrity_checked"):
        replace(_legal_budget(), source_result_integrity_checked=0)

    legal = _legal_budget()
    dumped = json.dumps(legal.to_config(), allow_nan=False)
    assert '"persistent_state_nbytes_before": 64' in dumped
    assert '"stomp_update_evaluations_per_delegated_update": 1' in dumped
    assert '"persistent_state_growth_bytes": 0' in dumped
    assert '"caller_authenticated": false' in dumped
    assert '"source_result_integrity_checked": true' in dumped
    assert '"persistent_state_nbytes_before": true' not in dumped
    assert '"stomp_update_evaluations_per_delegated_update": true' not in dumped
    assert '"caller_authenticated": 1' not in dumped
