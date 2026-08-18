"""Focused scalar and allocation preflights for interaction features."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.interaction_features import FixedBudgetInteractionLearner

_INT32_MAX = 2**31 - 1


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover - exact-type rejection must win
        raise AssertionError("integer hook executed")

    def __repr__(self) -> str:  # pragma: no cover - errors must not interpolate values
        raise AssertionError("repr executed")


class _ClassSpoof:
    @property  # type: ignore[misc]
    def __class__(self) -> type:
        return int

    def __repr__(self) -> str:  # pragma: no cover - errors must not interpolate values
        raise AssertionError("repr executed")


def _construct(**overrides: Any) -> FixedBudgetInteractionLearner:
    values: dict[str, Any] = {"n_features": 4, "n_tasks": 2, "candidate_count": 2}
    values.update(overrides)
    return FixedBudgetInteractionLearner(**values)


@pytest.mark.parametrize(
    ("field", "valid"),
    [
        ("n_features", 4),
        ("n_tasks", 2),
        ("candidate_count", 2),
        ("replacement_interval", 0),
        ("min_feature_age", 0),
        ("candidate_min_age", 0),
        ("utility_top_k", 1),
        ("utility_retention_grace_steps", 0),
        ("utility_evidence_confirmation_steps", 0),
        ("stale_retirement_interval", 1),
        ("candidate_promotion_confirmation_steps", 1),
        ("candidate_reacquisition_confirmation_steps", 1),
    ],
)
def test_integer_fields_reject_spoofs_without_hooks(field: str, valid: int) -> None:
    valid_overrides: dict[str, Any] = {field: np.int64(valid)}
    if field == "utility_retention_grace_steps":
        valid_overrides["utility_evidence_threshold"] = 0.1
    assert _construct(**valid_overrides).to_config()[field] == valid
    assert type(_construct(**valid_overrides).to_config()[field]) is int
    for invalid in (True, np.bool_(True), float(valid), _HostileInt(valid), _ClassSpoof()):
        with pytest.raises(ValueError, match=field):
            _construct(**{field: invalid})


@pytest.mark.parametrize(
    "integer_type",
    [
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    ],
)
def test_full_numpy_integer_family_canonicalizes(integer_type: Callable[[int], Any]) -> None:
    learner = _construct(
        n_features=integer_type(4),
        n_tasks=integer_type(2),
        candidate_count=integer_type(2),
        replacement_interval=integer_type(0),
        min_feature_age=integer_type(0),
        candidate_min_age=integer_type(0),
        utility_top_k=integer_type(1),
    )
    payload = learner.to_config()
    for field in (
        "n_features",
        "n_tasks",
        "candidate_count",
        "replacement_interval",
        "min_feature_age",
        "candidate_min_age",
        "utility_top_k",
    ):
        assert type(payload[field]) is int


@pytest.mark.parametrize(
    "field",
    [
        "evidence_gated_active_output_memory",
        "independent_relevance_probe",
        "retire_stale_features",
        "refresh_candidates",
        "refresh_promoted_candidate",
        "include_squares",
        "use_obgd",
        "scale_robust",
    ],
)
def test_boolean_fields_require_exact_bool(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _construct(**{field: np.bool_(False)})


def test_persistent_state_resources_are_preflighted_without_allocation() -> None:
    # With one task, no candidates, and no normalizers, state bytes are 45F + 44.
    last_legal = (_INT32_MAX - 44) // 45
    with pytest.raises(ValueError, match="update working set byte count"):
        _construct(n_features=last_legal, n_tasks=1, candidate_count=0)
    with pytest.raises(ValueError, match="state byte count"):
        _construct(n_features=last_legal + 1, n_tasks=1, candidate_count=0)
    with pytest.raises(ValueError, match="state (scalar|byte) count"):
        _construct(n_features=50_000, n_tasks=50_000, candidate_count=1)


def test_all_pairs_resources_and_feature_dim_are_checked_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = _construct(
        n_features=1,
        n_tasks=1,
        candidate_count=1,
        candidate_strategy="all_pairs",
    )
    max_pairs = _INT32_MAX // 8
    last_legal_dim = (1 + math.isqrt(1 + 8 * max_pairs)) // 2

    def tiny_candidates(*_args: object) -> tuple[jnp.ndarray, jnp.ndarray]:
        zero = jnp.zeros((1,), dtype=jnp.int32)
        return zero, zero

    monkeypatch.setattr(learner, "_candidate_pairs", tiny_candidates)
    state = learner.init(last_legal_dim, jr.key(0))
    assert state.candidate_left.shape == (1,)
    with pytest.raises(ValueError, match="all-pairs candidate construction byte count"):
        learner.init(last_legal_dim + 1, jr.key(0))
    with pytest.raises(ValueError, match="feature_dim"):
        learner.init(True, jr.key(0))


def test_config_accepts_mapping_and_requires_exact_serialized_schema() -> None:
    payload = _construct().to_config()
    restored = FixedBudgetInteractionLearner.from_config(MappingProxyType(payload))
    assert restored.to_config() == payload

    for mutation, match in (
        ({"type": "OtherLearner"}, "type"),
        ({"n_features": np.int32(4)}, "n_features"),
        ({"step_size_output": np.float32(0.03)}, "step_size_output"),
        ({"refresh_candidates": np.bool_(True)}, "refresh_candidates"),
        ({"generator_mix": [1.0, 0, 0.0]}, "generator_mix"),
        ({"task_utility_weights": [1]}, "task_utility_weights"),
        ({"extra": 1}, "fields"),
    ):
        invalid = dict(payload)
        invalid.update(mutation)
        with pytest.raises((TypeError, ValueError), match=match):
            FixedBudgetInteractionLearner.from_config(invalid)


def test_config_normalizes_hostile_mapping_failure() -> None:
    class HostileMapping(Mapping[str, Any]):
        def __getitem__(self, key: str) -> Any:
            raise RuntimeError("hook executed")

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError("hook executed")

        def __len__(self) -> int:
            return 1

    with pytest.raises(ValueError, match="could not be read"):
        FixedBudgetInteractionLearner.from_config(HostileMapping())


@pytest.mark.parametrize(
    "field",
    [
        "step_size_output",
        "utility_decay",
        "promotion_margin",
        "promotion_blend",
        "task_activity_decay",
        "future_utility_mix",
        "utility_retention_decay",
        "utility_evidence_threshold",
        "candidate_promotion_floor",
        "candidate_utility_retention_decay",
        "obgd_kappa",
        "scale_normalizer_decay",
        "scale_normalizer_epsilon",
    ],
)
def test_leftover_constructor_scalars_reject_bool_and_nonfinite(field: str) -> None:
    for invalid in (True, False, np.bool_(True), float("nan"), float("inf")):
        with pytest.raises(ValueError, match=field):
            _construct(**{field: invalid})


def test_leftover_constructor_scalars_keep_legal_floats() -> None:
    learner = _construct(
        step_size_output=0.0,
        obgd_kappa=2.0,
        utility_decay=0.0,
        promotion_blend=1.0,
        future_utility_mix=0.0,
        task_activity_decay=0.0,
        scale_normalizer_decay=0.0,
        scale_normalizer_epsilon=1e-6,
    )
    payload = learner.to_config()
    assert payload["step_size_output"] == 0.0
    assert payload["obgd_kappa"] == 2.0
    assert payload["utility_decay"] == 0.0
    assert payload["promotion_blend"] == 1.0
    assert type(payload["step_size_output"]) is float
    assert type(payload["obgd_kappa"]) is float
