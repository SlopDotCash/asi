"""Leftover-identity gates for learning-signal resource-budget records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.core.learning_signals import LearningSignalResourceBudget


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
