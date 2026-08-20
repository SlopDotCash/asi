"""Reject oversized subtask-index lists before unbounded from_config walks."""

from __future__ import annotations

import time

import pytest

from alberta_framework.core.prototype_feature_lifecycle import (
    _MAX_PYTHON_COLLECTION_LENGTH,
    PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA,
    PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS,
    PrototypeFeatureLifecycleConfig,
)

pytestmark = pytest.mark.unit


def _payload(*, indices: list[int]) -> dict[str, object]:
    return {
        "schema": PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA,
        "type": "PrototypeFeatureLifecycleConfig",
        "mechanism_status": PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS,
        "scientific_promotion_allowed": False,
        "base_feature_dim": 4,
        "active_pair_slots": 2,
        "candidate_pair_slots": 6,
        "n_tasks": 2,
        "n_options": 2,
        "n_primitive_actions": 2,
        "option_subtask_feature_indices": indices,
        "step_size_output": 0.05,
        "utility_decay": 0.9,
        "replacement_interval": 1,
        "min_feature_age": 0,
        "candidate_min_age": 0,
        "promotion_margin": 1.0,
        "scale_normalizer_decay": 0.9,
        "scale_normalizer_epsilon": 1.0e-6,
        "carry_survivors": True,
        "max_observations": 100,
    }


def test_from_config_rejects_first_overflow_list_length() -> None:
    with pytest.raises(ValueError, match="collection limit"):
        PrototypeFeatureLifecycleConfig.from_config(
            _payload(indices=[0] * (_MAX_PYTHON_COLLECTION_LENGTH + 1))
        )


def test_from_config_rejects_million_index_list_before_walk() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="collection limit"):
        PrototypeFeatureLifecycleConfig.from_config(_payload(indices=[0] * 5_000_000))
    assert time.perf_counter() - started < 0.25


def test_from_config_last_fit_pair_still_roundtrips() -> None:
    config = PrototypeFeatureLifecycleConfig.from_config(_payload(indices=[0, 1]))
    assert config.option_subtask_feature_indices == (0, 1)
