"""Hostile string gates for ipmnist screening domain and merge shards."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_screening import (
    _require_screening_curve_domain,
    merge_shards,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_curve_domain_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("per_task_accuracy")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="field must be an exact string"):
        _require_screening_curve_domain(np.array([0.5]), hostile, context="ctx")
    assert _HostileStr.calls == 0


def test_curve_domain_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="field must be an exact string"):
        _require_screening_curve_domain(np.array([0.5]), 123, context="ctx")  # type: ignore[arg-type]


def test_curve_domain_benign_passes() -> None:
    _require_screening_curve_domain(
        np.array([0.5]), "per_task_accuracy", context="ctx"
    )
    _require_screening_curve_domain(np.array([0.5]), "per_task_loss", context="ctx")


def test_merge_shards_rejects_hostile_before_in(tmp_path) -> None:
    hostile = _HostileStr("upgd_w_control")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="control is invalid"):
        merge_shards([], control_name=hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_merge_shards_rejects_non_string(tmp_path) -> None:
    with pytest.raises(ValueError, match="control is invalid"):
        merge_shards([], control_name=123)  # type: ignore[arg-type]


def test_merge_shards_benign_missing_control() -> None:
    with pytest.raises(ValueError, match="no shards given"):
        merge_shards([], control_name="upgd_w_control")
