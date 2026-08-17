"""Host-boundary identities for the recurring-feature protocol and criteria."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from alberta_framework.recurring_feature_gate import (
    RecurringFeatureGateCriteria,
    RecurringFeatureProtocol,
    run_recurring_feature_gate,
)


def test_canonical_protocol_and_criteria_remain_legal() -> None:
    RecurringFeatureProtocol().validate()
    RecurringFeatureGateCriteria().validate()
    protocol = replace(
        RecurringFeatureProtocol(),
        step_size_output=0.0,
        utility_decay=0.0,
        retained_utility_decay=0.0,
        replacement_interval=0,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_blend=0.0,
        future_utility_mix=0.0,
        candidate_utility_retention_decay=None,
    )
    protocol.validate()


@pytest.mark.parametrize(
    "field",
    [
        "steps_per_phase",
        "active_pair_budget",
        "candidate_pair_budget",
        "heldout_samples",
        "recovery_window",
        "replacement_interval",
        "min_feature_age",
        "candidate_min_age",
    ],
)
@pytest.mark.parametrize("invalid", [True, False, np.bool_(True), 1.0, np.int64(1)])
def test_protocol_integer_fields_reject_bool_and_non_builtin(
    field: str, invalid: object
) -> None:
    overrides: dict[str, object] = {field: invalid}
    if field == "steps_per_phase":
        overrides["recovery_window"] = 1
    with pytest.raises(ValueError, match=f"{field} must"):
        replace(RecurringFeatureProtocol(), **overrides).validate()


@pytest.mark.parametrize(
    "field",
    [
        "target_amplitude",
        "recovery_nmse_threshold",
        "step_size_output",
        "utility_decay",
        "retained_utility_decay",
        "promotion_margin",
        "promotion_blend",
        "future_utility_mix",
        "obgd_kappa",
    ],
)
@pytest.mark.parametrize("invalid", [True, False, np.bool_(True), math.nan, math.inf])
def test_protocol_float_fields_reject_bool_and_non_finite(
    field: str, invalid: object
) -> None:
    with pytest.raises(ValueError, match=f"{field} must"):
        replace(RecurringFeatureProtocol(), **{field: invalid}).validate()


def test_protocol_optional_retention_decay_rejects_bool_and_non_finite() -> None:
    for invalid in (True, False, np.bool_(True), math.nan, math.inf):
        with pytest.raises(ValueError, match="candidate_utility_retention_decay"):
            replace(
                RecurringFeatureProtocol(),
                candidate_utility_retention_decay=invalid,
            ).validate()


@pytest.mark.parametrize(
    "field",
    ["refresh_candidates", "refresh_promoted_candidate", "use_obgd"],
)
def test_protocol_flags_require_exact_bool(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        replace(RecurringFeatureProtocol(), **{field: 1}).validate()
    with pytest.raises(ValueError, match=field):
        replace(RecurringFeatureProtocol(), **{field: np.bool_(True)}).validate()


@pytest.mark.parametrize(
    "field",
    [
        "minimum_seeds",
        "minimum_heldout_samples",
    ],
)
@pytest.mark.parametrize("invalid", [True, False, np.bool_(True), 1.0])
def test_criteria_integer_fields_reject_bool(field: str, invalid: object) -> None:
    with pytest.raises(ValueError, match=f"{field} must"):
        RecurringFeatureGateCriteria(**{field: invalid}).validate()


@pytest.mark.parametrize(
    "field",
    [
        "minimum_retained_all_critical_rate",
        "minimum_obsolete_eviction_rate",
        "maximum_median_critical_nmse",
        "minimum_median_obsolete_nmse",
        "minimum_retention_rate_gain_over_baseline",
        "minimum_critical_nmse_gain_over_baseline",
    ],
)
@pytest.mark.parametrize("invalid", [True, False, np.bool_(True), math.nan, math.inf])
def test_criteria_float_fields_reject_bool_and_non_finite(
    field: str, invalid: object
) -> None:
    with pytest.raises(ValueError, match=f"{field} must"):
        RecurringFeatureGateCriteria(**{field: invalid}).validate()


def test_criteria_recurrence_flag_requires_exact_bool() -> None:
    with pytest.raises(ValueError, match="require_recurrence_faster_than_acquisition"):
        RecurringFeatureGateCriteria(
            require_recurrence_faster_than_acquisition=1
        ).validate()


def test_run_rejects_boolean_seed_before_allocation() -> None:
    protocol = replace(
        RecurringFeatureProtocol(),
        steps_per_phase=1,
        recovery_window=1,
        heldout_samples=1,
    )
    with pytest.raises(ValueError, match="seeds"):
        run_recurring_feature_gate((True,), protocol=protocol)
