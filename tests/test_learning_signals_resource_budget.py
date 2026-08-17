"""Leftover-identity gates for learning-signal resource-budget records."""

from __future__ import annotations

import dataclasses
import json

import pytest

from alberta_framework.core.learning_signals import (
    LearningSignalEstimator,
    LearningSignalEstimatorConfig,
    LearningSignalResourceBudget,
)


class _HostileInt(int):
    def __index__(self) -> int:
        raise AssertionError("hostile index hook executed")


def _legal_budget() -> LearningSignalResourceBudget:
    return LearningSignalResourceBudget(
        input_float_scalars_per_step=15,
        persistent_float32_scalars=5,
        persistent_int32_scalars=4,
        persistent_state_scalars=9,
        persistent_state_bytes=36,
        output_float32_scalars=8,
        output_bool_scalars=6,
        output_logical_bytes=38,
        trainable_scalars=0,
    )


def test_learning_signals_resource_budget_rejects_leftover_identities() -> None:
    """Public resource-budget records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="input_float_scalars_per_step"):
        LearningSignalResourceBudget(
            input_float_scalars_per_step=True,
            persistent_float32_scalars=5,
            persistent_int32_scalars=4,
            persistent_state_scalars=9,
            persistent_state_bytes=36,
            output_float32_scalars=8,
            output_bool_scalars=6,
            output_logical_bytes=38,
            trainable_scalars=0,
        )
    with pytest.raises(ValueError, match="trainable_scalars"):
        LearningSignalResourceBudget(
            input_float_scalars_per_step=15,
            persistent_float32_scalars=5,
            persistent_int32_scalars=4,
            persistent_state_scalars=9,
            persistent_state_bytes=36,
            output_float32_scalars=8,
            output_bool_scalars=6,
            output_logical_bytes=38,
            trainable_scalars=True,
        )
    with pytest.raises(ValueError, match="persistent_state_bytes"):
        LearningSignalResourceBudget(
            input_float_scalars_per_step=15,
            persistent_float32_scalars=5,
            persistent_int32_scalars=4,
            persistent_state_scalars=9,
            persistent_state_bytes=float("nan"),
            output_float32_scalars=8,
            output_bool_scalars=6,
            output_logical_bytes=38,
            trainable_scalars=0,
        )

    legal = _legal_budget()
    dumped = json.dumps(legal.to_config(), allow_nan=False)
    assert '"input_float_scalars_per_step": 15' in dumped
    assert '"output_bool_scalars": 6' in dumped
    assert '"trainable_scalars": 0' in dumped
    assert '"input_float_scalars_per_step": true' not in dumped
    assert '"trainable_scalars": true' not in dumped
    assert '"persistent_state_bytes": true' not in dumped


@pytest.mark.parametrize(
    "field",
    [
        "persistent_float32_scalars",
        "persistent_int32_scalars",
        "persistent_state_scalars",
        "persistent_state_bytes",
        "output_float32_scalars",
        "output_bool_scalars",
        "output_logical_bytes",
        "trainable_scalars",
    ],
)
def test_learning_signal_budget_requires_exact_derived_identity(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(_legal_budget(), **{field: getattr(_legal_budget(), field) + 1})


def test_learning_signal_budget_rejects_hostile_integer_subclass_without_hook() -> None:
    with pytest.raises(ValueError, match="input_float_scalars_per_step"):
        dataclasses.replace(_legal_budget(), input_float_scalars_per_step=_HostileInt(15))


@pytest.mark.parametrize("count", [6, 8, 10, 11, 2_147_483_647])
def test_learning_signal_budget_accepts_attainable_input_counts(count: int) -> None:
    assert dataclasses.replace(_legal_budget(), input_float_scalars_per_step=count)


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 7, 9, 13, 17])
def test_learning_signal_budget_rejects_unattainable_input_counts(count: int) -> None:
    with pytest.raises(ValueError, match="not attainable"):
        dataclasses.replace(_legal_budget(), input_float_scalars_per_step=count)


def test_learning_signal_config_gates_exact_derived_input_budget_bound() -> None:
    largest = LearningSignalEstimatorConfig(ensemble_size=1_073_741_822, target_dim=1)
    assert (
        LearningSignalEstimator(largest).resource_budget().input_float_scalars_per_step
        == 2_147_483_646
    )
    with pytest.raises(ValueError, match="input resource budget"):
        LearningSignalEstimatorConfig(ensemble_size=1_073_741_823, target_dim=1)
    with pytest.raises(ValueError, match="input resource budget"):
        LearningSignalEstimatorConfig(ensemble_size=50_000, target_dim=50_000)
