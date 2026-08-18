"""Leftover-identity gates for learning-signal resource-budget records."""

from __future__ import annotations

import dataclasses
import json

import jax.numpy as jnp
import pytest
from jax import tree_util

from alberta_framework.core.learning_signals import (
    LearningSignalEstimator,
    LearningSignalEstimatorConfig,
    LearningSignalResourceBudget,
    _learning_signal_observe_working_set_bytes,
    _preflight_learning_signal_observe_working_set,
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
    with pytest.raises(ValueError, match="observe working set byte count"):
        LearningSignalEstimatorConfig(ensemble_size=1_073_741_822, target_dim=1)
    with pytest.raises(ValueError, match="input resource budget"):
        LearningSignalEstimatorConfig(ensemble_size=1_073_741_823, target_dim=1)
    with pytest.raises(ValueError, match="input resource budget"):
        LearningSignalEstimatorConfig(ensemble_size=50_000, target_dim=50_000)


_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_learning_signal_persist_matches_jit_and_input_bytes_still_fit() -> None:
    config = LearningSignalEstimatorConfig(ensemble_size=2, target_dim=1)
    estimator = LearningSignalEstimator(config)
    state = estimator.init()
    actual = sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in tree_util.tree_leaves(state)
    )
    assert actual == estimator.resource_budget().persistent_state_bytes == 36
    target_dim = 45_000_000
    input_scalars = 2 * 2 * target_dim + target_dim + 1
    assert input_scalars <= _INT32_MAX
    assert 4 * input_scalars <= _INT32_MAX
    assert _learning_signal_observe_working_set_bytes(2, target_dim) > _INT32_MAX
    with pytest.raises(ValueError, match="observe working set byte count"):
        LearningSignalEstimatorConfig(ensemble_size=2, target_dim=target_dim)


def test_learning_signal_last_fit_and_first_overflow_are_adjacent() -> None:
    last_fit = None
    first_overflow = None
    for target_dim in range(31_580_638, 31_580_642):
        working_set_bytes = _learning_signal_observe_working_set_bytes(2, target_dim)
        input_scalars = 2 * 2 * target_dim + target_dim + 1
        assert input_scalars <= _INT32_MAX
        assert 4 * input_scalars <= _INT32_MAX
        if working_set_bytes <= _INT32_MAX:
            last_fit = target_dim
        elif first_overflow is None:
            first_overflow = target_dim
            break
    assert last_fit is not None and first_overflow == last_fit + 1
    LearningSignalEstimatorConfig(ensemble_size=2, target_dim=last_fit)
    with pytest.raises(ValueError, match="observe working set byte count"):
        LearningSignalEstimatorConfig(ensemble_size=2, target_dim=first_overflow)


def test_learning_signal_input_scalar_bound_still_fires_first() -> None:
    with pytest.raises(ValueError, match="input resource budget"):
        LearningSignalEstimatorConfig(ensemble_size=1_073_741_823, target_dim=1)


def test_preflight_helper_rejects_the_same_working_set() -> None:
    with pytest.raises(ValueError, match="observe working set byte count"):
        _preflight_learning_signal_observe_working_set(2, 45_000_000)
