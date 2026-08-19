"""Hostile-safe validation for metrics utilities.

Includes the boolean-trace depth ceiling: a 2000-deep nest must raise
ValueError rather than RecursionError.
"""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.utils.metrics import (
    ContinualLearningSummary,
    StabilityGap,
    compare_learners,
    compute_cumulative_error,
    compute_forward_transfer,
    compute_per_task_forgetting,
    compute_prequential_performance,
    compute_recovery_lengths,
    compute_running_mean,
    compute_stability_gap,
    extract_metric,
)


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr hook")


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:
        type(self).calls += 1
        raise RuntimeError("ratio hook")


class _StringSubclass(str):
    pass


def test_rejects_hostile_float_threshold_without_ratio() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="threshold"):
        compute_recovery_lengths([0.1, 0.9, 0.9], [0], _HostileFloat(0.8), window_size=1)
    assert _HostileFloat.calls == 0


def test_rejects_string_subclass_error_key() -> None:
    with pytest.raises(ValueError, match="error_key"):
        compute_cumulative_error([{"squared_error": 1.0}], _StringSubclass("squared_error"))


def test_rejects_string_subclass_metric_compare() -> None:
    with pytest.raises(ValueError, match="metric"):
        compare_learners({"a": [{"squared_error": 1.0}]}, metric=_StringSubclass("squared_error"))


def test_rejects_string_subclass_key_extract() -> None:
    with pytest.raises(ValueError, match="key"):
        extract_metric([{"x": 1.0}], _StringSubclass("x"))


def test_rejects_string_subclass_learner_name() -> None:
    with pytest.raises(ValueError, match="learner name"):
        compare_learners({_StringSubclass("a"): [{"squared_error": 1.0}]})


def test_rejects_hostile_int_window_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        compute_running_mean([1.0, 2.0, 3.0], window_size=_HostileInt(2))


def test_rejects_hostile_int_change_points() -> None:
    with pytest.raises(ValueError, match="change_points"):
        compute_recovery_lengths([0.1, 0.9, 0.9], [_HostileInt(0)], 0.8, window_size=1)


def test_rejects_bool_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        compute_recovery_lengths([0.1, 0.9], [0], True, window_size=1)


def test_rejects_deeply_nested_boolean_trace_without_recursion_error() -> None:
    nest: object = True
    for _ in range(2000):
        nest = [nest]
    with pytest.raises(ValueError, match="depth"):
        compute_stability_gap(nest, 0.0)  # type: ignore[arg-type]


def test_rejects_hostile_float_reference_performance() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="reference_performance"):
        compute_stability_gap([0.5, 0.6], _HostileFloat(0.8))
    assert _HostileFloat.calls == 0


def test_stability_gap_rejects_hostile_float_mean() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="mean"):
        StabilityGap(mean=_HostileFloat(0.1), maximum=0.2, per_step=np.array([0.1]))
    assert _HostileFloat.calls == 0


def test_rejects_hostile_repr_window_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        compute_running_mean([1.0, 2.0], window_size=_RaisingRepr())  # type: ignore[arg-type]


def test_numpy_and_fraction_threshold_canonicalizes() -> None:
    result = compute_recovery_lengths([0.1, 0.9, 0.9], [0], np.float64(0.8), window_size=1)
    assert int(result[0]) == 2
    from fractions import Fraction

    result2 = compute_recovery_lengths([0.1, 0.9, 0.9], [0], Fraction(4, 5), window_size=1)  # type: ignore[arg-type]
    assert int(result2[0]) == 2


def test_rejects_hostile_containers_before_iteration_hooks() -> None:
    class HostileList(list[float]):
        def __iter__(self):  # type: ignore[no-untyped-def, override]
            raise AssertionError("list hook executed")

    class HostileDict(dict[str, list[dict[str, float]]]):
        def items(self):  # type: ignore[no-untyped-def, override]
            raise AssertionError("mapping hook executed")

    with pytest.raises(ValueError, match="online_performance"):
        compute_prequential_performance(HostileList([0.1]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="results"):
        compare_learners(HostileDict())  # type: ignore[arg-type]


def test_rejects_coercive_baselines_and_direction_flags() -> None:
    with pytest.raises(ValueError, match="baseline_performance"):
        compute_forward_transfer([[0.1]], [0], [True])
    with pytest.raises(ValueError, match="higher_is_better"):
        compute_per_task_forgetting([[0.1]], [0], higher_is_better=1)  # type: ignore[arg-type]


def test_window_size_is_bounded_before_arithmetic() -> None:
    with pytest.raises(ValueError, match="window_size"):
        compute_running_mean([1.0], window_size=2**31)


def test_metric_history_rejects_mapping_subclasses_before_hooks() -> None:
    class HostileRecord(dict[str, float]):
        def __getitem__(self, key: str) -> float:
            raise AssertionError("getitem hook executed")

    with pytest.raises(ValueError, match=r"metrics_history\[0\]"):
        compute_cumulative_error([HostileRecord(squared_error=1.0)])


def test_public_metric_records_bind_exact_array_schemas_and_totals() -> None:
    with pytest.raises(ValueError, match="per_step"):
        StabilityGap(mean=0.0, maximum=0.0, per_step=np.array([False]))
    with pytest.raises(ValueError, match="match per_step"):
        StabilityGap(mean=0.0, maximum=0.2, per_step=np.array([0.2]))

    payload = dict(
        final_performance=0.8,
        prequential_performance=0.7,
        mean_forgetting=0.1,
        max_forgetting=0.1,
        backward_transfer=0.0,
        stability_gap_mean=0.05,
        stability_gap_max=0.1,
        per_task_final_performance=np.array([0.8]),
        per_task_forgetting=np.array([0.1]),
        per_task_backward_transfer=np.array([0.0]),
    )
    with pytest.raises(ValueError, match="final_performance"):
        ContinualLearningSummary(**{**payload, "final_performance": 0.7})
    with pytest.raises(ValueError, match="matching shapes"):
        ContinualLearningSummary(
            **{**payload, "per_task_backward_transfer": np.array([0.0, 0.0])}
        )
