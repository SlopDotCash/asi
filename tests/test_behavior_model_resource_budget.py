"""Leftover-identity gates for behavior-model resource-budget records."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
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


def test_behavior_budget_rejects_hostile_integer_facades_without_hooks() -> None:
    class HostileInt(int):
        comparison_calls = 0

        def __lt__(self, other: object) -> bool:
            type(self).comparison_calls += 1
            raise RuntimeError("comparison hook")

    with pytest.raises(ValueError, match="feature_dim"):
        replace(_legal_budget(), feature_dim=HostileInt(5))
    assert HostileInt.comparison_calls == 0


@pytest.mark.parametrize("field", ("feature_dim", "n_actions"))
@pytest.mark.parametrize("value", (0, 2**31, np.uint64(2**32)))
def test_behavior_budget_rejects_dimension_boundaries(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        replace(_legal_budget(), **{field: value})


@pytest.mark.parametrize(
    "field",
    (
        "trainable_float32_scalars",
        "diagnostic_float32_scalars",
        "administrative_int32_scalars",
        "rng_uint32_scalars",
        "state_nbytes",
        "learned_float32_scalars_touched_per_update",
        "replay_capacity",
    ),
)
def test_behavior_budget_rejects_cross_field_formula_drift(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        replace(_legal_budget(), **{field: getattr(_legal_budget(), field) + 1})
