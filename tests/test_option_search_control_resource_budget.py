"""Leftover-identity gates for option-search resource-budget records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.core.option_search_control import OptionSearchControlResourceBudget


def _legal_budget() -> OptionSearchControlResourceBudget:
    return OptionSearchControlResourceBudget(
        n_options=2,
        observation_dim=4,
        backup_budget=3,
        persistent_state_bytes=0,
        rng_draws_per_call=0,
        candidate_values_per_evaluation=2,
        max_candidate_evaluations_per_call=6,
        max_base_learner_updates_per_call=3,
        max_model_matrix_vector_products_per_call=6,
        max_base_value_forward_calls_per_call=12,
        max_base_value_backward_calls_per_call=3,
        stomp_self_audits_per_call=1,
        max_diagnostic_payload_bytes_per_call=191,
    )


def test_option_search_resource_budget_rejects_leftover_identities() -> None:
    """Public resource-budget records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="n_options"):
        OptionSearchControlResourceBudget(
            n_options=True,
            observation_dim=4,
            backup_budget=3,
            persistent_state_bytes=0,
            rng_draws_per_call=0,
            candidate_values_per_evaluation=2,
            max_candidate_evaluations_per_call=6,
            max_base_learner_updates_per_call=3,
            max_model_matrix_vector_products_per_call=6,
            max_base_value_forward_calls_per_call=12,
            max_base_value_backward_calls_per_call=3,
            stomp_self_audits_per_call=1,
            max_diagnostic_payload_bytes_per_call=191,
        )
    with pytest.raises(ValueError, match="rng_draws_per_call"):
        OptionSearchControlResourceBudget(
            n_options=2,
            observation_dim=4,
            backup_budget=3,
            persistent_state_bytes=0,
            rng_draws_per_call=True,
            candidate_values_per_evaluation=2,
            max_candidate_evaluations_per_call=6,
            max_base_learner_updates_per_call=3,
            max_model_matrix_vector_products_per_call=6,
            max_base_value_forward_calls_per_call=12,
            max_base_value_backward_calls_per_call=3,
            stomp_self_audits_per_call=1,
            max_diagnostic_payload_bytes_per_call=191,
        )
    with pytest.raises(ValueError, match="persistent_state_bytes"):
        OptionSearchControlResourceBudget(
            n_options=2,
            observation_dim=4,
            backup_budget=3,
            persistent_state_bytes=float("nan"),
            rng_draws_per_call=0,
            candidate_values_per_evaluation=2,
            max_candidate_evaluations_per_call=6,
            max_base_learner_updates_per_call=3,
            max_model_matrix_vector_products_per_call=6,
            max_base_value_forward_calls_per_call=12,
            max_base_value_backward_calls_per_call=3,
            stomp_self_audits_per_call=1,
            max_diagnostic_payload_bytes_per_call=191,
        )

    legal = _legal_budget()
    dumped = json.dumps(legal.to_config(), allow_nan=False)
    assert '"n_options": 2' in dumped
    assert '"backup_budget": 3' in dumped
    assert '"persistent_state_bytes": 0' in dumped
    assert '"n_options": true' not in dumped
    assert '"rng_draws_per_call": true' not in dumped
    assert '"persistent_state_bytes": true' not in dumped
