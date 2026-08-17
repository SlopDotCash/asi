"""Leftover-identity gates for prototype-lifecycle resource-budget records."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureLifecycleResourceBudget,
)


def _legal_budget() -> PrototypeFeatureLifecycleResourceBudget:
    return PrototypeFeatureLifecycleResourceBudget(
        mechanism_status="development_mechanism_only",
        scientific_promotion_allowed=False,
        base_feature_slots=4,
        active_pair_slots=1,
        candidate_pair_slots=1,
        managed_oak_feature_width=5,
        learner_persistent_state_nbytes=8,
        router_persistent_state_nbytes=8,
        lifecycle_counter_nbytes=16,
        lifecycle_state_nbytes=32,
        consumer_binding_persistent_nbytes=12,
        internal_learner_template_nbytes=8,
        internal_oak_template_nbytes=8,
        internal_template_nbytes=16,
        owned_persistent_state_nbytes=48,
        managed_oak_consumer_nbytes=60,
        rebuilt_base_cache_nbytes=20,
        input_route_feature_groups=2,
        output_route_feature_groups=1,
        router_calls_per_observe=2,
        router_calls_per_committed_curation=2,
        max_active_pair_products_per_observe=5,
        max_candidate_pair_products_per_observe=1,
        max_observations=10,
    )


def test_prototype_lifecycle_resource_budget_rejects_leftover_identities() -> None:
    """Public resource-budget records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="base_feature_slots"):
        replace(_legal_budget(), base_feature_slots=True)
    with pytest.raises(ValueError, match="max_observations"):
        replace(_legal_budget(), max_observations=True)
    with pytest.raises(ValueError, match="lifecycle_state_nbytes"):
        replace(_legal_budget(), lifecycle_state_nbytes=float("nan"))
    with pytest.raises(ValueError, match="scientific_promotion_allowed"):
        replace(_legal_budget(), scientific_promotion_allowed=1)
    with pytest.raises(ValueError, match="mechanism_status"):
        replace(_legal_budget(), mechanism_status=True)
    with pytest.raises(ValueError, match="mechanism_status"):
        replace(_legal_budget(), mechanism_status="other")
    with pytest.raises(ValueError, match="scientific_promotion_allowed"):
        replace(_legal_budget(), scientific_promotion_allowed=True)

    legal = _legal_budget()
    dumped = json.dumps(legal.to_config(), allow_nan=False)
    assert '"base_feature_slots": 4' in dumped
    assert '"max_observations": 10' in dumped
    assert '"lifecycle_state_nbytes": 32' in dumped
    assert '"scientific_promotion_allowed": false' in dumped
    assert '"base_feature_slots": true' not in dumped
    assert '"max_observations": true' not in dumped
    assert '"lifecycle_state_nbytes": true' not in dumped
    assert '"scientific_promotion_allowed": 1' not in dumped


@pytest.mark.parametrize(
    "field",
    (
        "managed_oak_feature_width",
        "lifecycle_state_nbytes",
        "consumer_binding_persistent_nbytes",
        "internal_template_nbytes",
        "owned_persistent_state_nbytes",
        "managed_oak_consumer_nbytes",
        "rebuilt_base_cache_nbytes",
        "router_calls_per_observe",
        "router_calls_per_committed_curation",
        "max_active_pair_products_per_observe",
        "max_candidate_pair_products_per_observe",
    ),
)
def test_prototype_lifecycle_resource_budget_rejects_formula_drift(field: str) -> None:
    budget = _legal_budget()
    with pytest.raises(ValueError, match=field):
        replace(budget, **{field: getattr(budget, field) + 1})


def test_prototype_lifecycle_resource_budget_rejects_hooks_and_subclasses() -> None:
    class HostileMeta(type):
        calls = 0

        def __hash__(cls) -> int:
            HostileMeta.calls += 1
            raise AssertionError("HostileMeta.__hash__ must not run")

    class HostileInt(int, metaclass=HostileMeta):
        def __int__(self) -> int:
            raise AssertionError("HostileInt.__int__ must not run")

        def __index__(self) -> int:
            raise AssertionError("HostileInt.__index__ must not run")

    HostileMeta.calls = 0
    with pytest.raises(ValueError, match="base_feature_slots"):
        replace(_legal_budget(), base_feature_slots=HostileInt(4))
    assert HostileMeta.calls == 0

    class HostileStr(str):
        calls = 0

        def __eq__(self, other: object) -> bool:
            type(self).calls += 1
            raise AssertionError("HostileStr.__eq__ must not run")

    HostileStr.calls = 0
    with pytest.raises(ValueError, match="mechanism_status"):
        replace(_legal_budget(), mechanism_status=HostileStr("development_mechanism_only"))
    assert HostileStr.calls == 0

    class BudgetSubclass(PrototypeFeatureLifecycleResourceBudget):
        pass

    with pytest.raises(ValueError, match="actual record"):
        BudgetSubclass(**_legal_budget().to_config())  # type: ignore[arg-type]
