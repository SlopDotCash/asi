"""Residual scalar and recursive-resource contracts for compositional features."""

from __future__ import annotations

from typing import Any, cast

import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import (
    GENERATION_ROBUST_RECURSIVE,
    CompositionalFeatureLearner,
    FiniteCandidateSelector,
)


class _HostileFloat(float):
    def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
        raise RuntimeError("ratio hook")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr executed")


def _learner(**overrides: Any) -> CompositionalFeatureLearner:
    values: dict[str, Any] = {"n_features": 6, "n_tasks": 2, "candidate_count": 2}
    values.update(overrides)
    return CompositionalFeatureLearner(**values)


@pytest.mark.parametrize(
    "field",
    [
        "step_size_output",
        "step_size_theta",
        "promotion_margin",
        "promotion_blend",
        "obgd_kappa",
        "parent_temperature",
        "parent_novelty_weight",
        "future_utility_mix",
        "future_utility_trace_decay",
        "candidate_score_energy_epsilon",
        "candidate_selector_learning_rate",
        "generator_resource_advantage_clip",
        "generator_resource_cost_weight",
    ],
)
def test_float32_sinks_contain_hostile_ratio_and_repr_hooks(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _learner(**{field: _HostileFloat(0.5)})


def test_selector_float_and_string_sinks_are_exact_and_canonical() -> None:
    selector = FiniteCandidateSelector(
        2,
        learning_rate=cast(Any, np.float32(0.5)),
        exploration=cast(Any, np.float64(0.25)),
    )
    assert type(selector.to_config()["learning_rate"]) is float
    for field in ("learning_rate", "exploration", "loss_lower_bound", "loss_upper_bound"):
        with pytest.raises(ValueError, match=field):
            FiniteCandidateSelector(2, **cast(Any, {field: _HostileFloat(0.5)}))
    with pytest.raises(ValueError, match="update_rule"):
        FiniteCandidateSelector(2, update_rule=np.str_("hedge"))


def test_sequence_and_string_contracts_roundtrip_canonically() -> None:
    learner = _learner(
        operation_prior=(0.0, 1.0, 0.0, 0.0, 0.0),
        generator_resource_initial_preferences=(0.0, 0.0, 0.0, 0.0),
    )
    assert CompositionalFeatureLearner.from_config(learner.to_config()).to_config() == (
        learner.to_config()
    )
    with pytest.raises(ValueError, match="operation_prior"):
        _learner(operation_prior=[0.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="generation_strategy"):
        _learner(generation_strategy=np.str_("utility"))


def test_signed_scaffolds_preflight_only_when_they_can_be_constructed() -> None:
    learner = _learner(
        n_features=20_001,
        n_tasks=1,
        candidate_count=0,
        generation_strategy=GENERATION_ROBUST_RECURSIVE,
        signed_tanh_scaffold_count=1,
    )
    with pytest.raises(ValueError, match="signed-tanh scaffold construction"):
        learner.init(20_000, jr.key(0))

    # With no composed slot, init never constructs the quadratic parent tables.
    no_composition = _learner(
        n_features=10,
        n_tasks=1,
        candidate_count=0,
        generation_strategy=GENERATION_ROBUST_RECURSIVE,
        signed_tanh_scaffold_count=1,
    )
    assert no_composition.init(10, jr.key(1)).ops.shape == (10,)
