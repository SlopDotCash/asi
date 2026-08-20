"""Machine-readable exports preserve finite measurements exactly and fail closed."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

import alberta_framework.utils.export as export_module
from alberta_framework.utils.experiments import AggregatedResults, MetricSummary
from alberta_framework.utils.export import (
    export_to_csv,
    export_to_json,
    generate_latex_table,
    generate_markdown_table,
    save_experiment_report,
)

pytestmark = pytest.mark.unit

_METRIC = "squared_error"
_ROUND_TRIP_VALUES = (
    1.0000000000000002,
    0.12345678901234566,
    1.785e-8,
    float.fromhex("0x0.0000000000001p-1022"),
)


def _constant_result(name: str, value: float = 1.0) -> AggregatedResults:
    values = np.asarray([value], dtype=np.float64)
    return AggregatedResults(
        config_name=name,
        seeds=[17],
        metric_arrays={_METRIC: values.reshape(1, 1)},
        summary={
            _METRIC: MetricSummary(
                mean=value,
                std=0.0,
                min=value,
                max=value,
                n_seeds=1,
                values=values,
            )
        },
    )


def _timeseries_result() -> AggregatedResults:
    series = np.asarray(
        [
            [_ROUND_TRIP_VALUES[0], _ROUND_TRIP_VALUES[2]],
            [_ROUND_TRIP_VALUES[1], _ROUND_TRIP_VALUES[3]],
        ],
        dtype=np.float64,
    )
    final_values = series[:, -1]
    return AggregatedResults(
        config_name="trace",
        seeds=[17, 29],
        metric_arrays={_METRIC: series},
        summary={
            _METRIC: MetricSummary(
                mean=float(np.mean(final_values)),
                std=float(np.std(final_values, ddof=1)),
                min=float(np.min(final_values)),
                max=float(np.max(final_values)),
                n_seeds=2,
                values=final_values,
            )
        },
    )


def _result_with_nonfinite(surface: str, number: float) -> AggregatedResults:
    result = _constant_result("invalid")
    summary = result.summary[_METRIC]
    if surface == "summary":
        summary = summary._replace(mean=number)
        return result._replace(summary={_METRIC: summary})
    if surface == "values":
        summary = summary._replace(values=np.asarray([number], dtype=np.float64))
        return result._replace(summary={_METRIC: summary})
    if surface == "timeseries":
        return result._replace(metric_arrays={_METRIC: np.asarray([[number]], dtype=np.float64)})
    raise AssertionError(f"unknown test surface: {surface}")


def _preflight_invalid_results(case: str) -> dict[str, AggregatedResults]:
    result = _constant_result("invalid")
    summary = result.summary[_METRIC]

    if case == "empty_results":
        return {}
    if case == "duplicate_seeds":
        duplicate = _timeseries_result()._replace(seeds=[17, 17])
        return {"invalid": duplicate}
    if case == "empty_seeds":
        empty_summary = summary._replace(
            mean=0.0,
            min=0.0,
            max=0.0,
            n_seeds=0,
            values=np.empty((0,), dtype=np.float64),
        )
        result = result._replace(
            seeds=[],
            metric_arrays={_METRIC: np.empty((0, 1), dtype=np.float64)},
            summary={_METRIC: empty_summary},
        )
    elif case == "bool_seed":
        result = result._replace(seeds=[True])
    elif case == "numpy_integer_seed":
        result = result._replace(seeds=[np.int64(17)])
    elif case == "negative_seed":
        result = result._replace(seeds=[-1])
    elif case == "seed_above_uint32":
        result = result._replace(seeds=[1 << 32])
    elif case == "zero_seed_axis":
        result = result._replace(metric_arrays={_METRIC: np.empty((0, 1), dtype=np.float64)})
    elif case == "zero_step_axis":
        result = result._replace(metric_arrays={_METRIC: np.empty((1, 0), dtype=np.float64)})
    elif case == "metric_ndim":
        result = result._replace(metric_arrays={_METRIC: np.ones((1, 1, 1), dtype=np.float64)})
    elif case == "metric_row_count":
        result = result._replace(metric_arrays={_METRIC: np.ones((2, 1), dtype=np.float64)})
    elif case == "summary_values_ndim":
        bad_summary = summary._replace(values=np.ones((1, 1), dtype=np.float64))
        result = result._replace(summary={_METRIC: bad_summary})
    elif case == "summary_count":
        bad_summary = summary._replace(n_seeds=2)
        result = result._replace(summary={_METRIC: bad_summary})
    elif case == "summary_value_count":
        bad_summary = summary._replace(values=np.ones((2,), dtype=np.float64))
        result = result._replace(summary={_METRIC: bad_summary})
    elif case == "nonfinite_unselected_array":
        result = result._replace(
            metric_arrays={
                _METRIC: result.metric_arrays[_METRIC],
                "unselected": np.asarray([[math.nan]], dtype=np.float64),
            }
        )
    elif case == "nonfinite_unselected_summary":
        bad_summary = summary._replace(mean=math.inf)
        result = result._replace(summary={_METRIC: summary, "unselected": bad_summary})
    elif case == "empty_metric_arrays":
        result = result._replace(metric_arrays={})
    elif case == "empty_summary":
        result = result._replace(summary={})
    else:
        raise AssertionError(f"unknown preflight test case: {case}")

    return {"invalid": result}


def _export_mode(
    mode: str,
    results: dict[str, AggregatedResults],
    path: Path,
) -> None:
    if mode == "summary_csv":
        export_to_csv(results, path)
    elif mode == "timeseries_csv":
        export_to_csv(results, path, include_timeseries=True)
    elif mode == "json":
        export_to_json(results, path, include_timeseries=True)
    else:
        raise AssertionError(f"unknown export mode: {mode}")


@pytest.mark.parametrize("value", _ROUND_TRIP_VALUES)
def test_summary_csv_uses_shortest_binary64_round_trip(value: float, tmp_path: Path) -> None:
    path = tmp_path / "nested" / "summary.csv"
    export_to_csv({"candidate": _constant_result("candidate", value)}, path)

    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["mean"] == repr(value)
    assert row["min"] == repr(value)
    assert row["max"] == repr(value)
    assert float(row["mean"]) == value


def test_timeseries_csv_round_trips_values_under_the_matching_seed_headers(
    tmp_path: Path,
) -> None:
    result = _timeseries_result()
    path = tmp_path / "timeseries.csv"
    export_to_csv({"trace": result}, path, include_timeseries=True)

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == ["step", "trace_seed17", "trace_seed29"]
    assert rows[0]["trace_seed17"] == repr(_ROUND_TRIP_VALUES[0])
    assert rows[0]["trace_seed29"] == repr(_ROUND_TRIP_VALUES[1])
    assert rows[1]["trace_seed17"] == repr(_ROUND_TRIP_VALUES[2])
    assert rows[1]["trace_seed29"] == repr(_ROUND_TRIP_VALUES[3])
    assert [[float(row["trace_seed17"]), float(row["trace_seed29"])] for row in rows] == [
        [_ROUND_TRIP_VALUES[0], _ROUND_TRIP_VALUES[1]],
        [_ROUND_TRIP_VALUES[2], _ROUND_TRIP_VALUES[3]],
    ]


@pytest.mark.parametrize("mode", ["summary_csv", "timeseries_csv", "json"])
def test_export_rejects_oversized_config_name_before_header_hang(
    mode: str,
    tmp_path: Path,
) -> None:
    name = "a" * 50_000
    result = _constant_result("placeholder")._replace(config_name=name)
    results = {name: result}
    suffix = "json" if mode == "json" else "csv"
    existing = tmp_path / f"existing.{suffix}"
    sentinel = "existing artifact\n"
    existing.write_text(sentinel, encoding="utf-8")

    with pytest.raises(ValueError, match="config name exceeds"):
        _export_mode(mode, results, existing)

    assert existing.read_text(encoding="utf-8") == sentinel


def test_timeseries_csv_accepts_max_config_name(tmp_path: Path) -> None:
    name = "a" * export_module._MAX_EXPORT_STRING_BYTES
    result = _constant_result("placeholder")._replace(config_name=name)
    path = tmp_path / "bounded.csv"
    export_to_csv({name: result}, path, include_timeseries=True)
    assert path.is_file()
    with path.open(encoding="utf-8", newline="") as handle:
        assert next(csv.DictReader(handle))[f"{name}_seed17"] == "1.0"


def test_report_rejects_oversized_experiment_name_before_path_join(
    tmp_path: Path,
) -> None:
    results = {"candidate": _constant_result("candidate")}
    with pytest.raises(ValueError, match="experiment_name exceeds"):
        save_experiment_report(
            results,
            tmp_path,
            "a" * (export_module._MAX_EXPORT_FILENAME_BYTES + 1),
            metric=_METRIC,
        )
    assert list(tmp_path.iterdir()) == []


def test_json_round_trips_summary_values_and_timeseries(tmp_path: Path) -> None:
    results = {
        f"value_{index}": _constant_result(f"value_{index}", value)
        for index, value in enumerate(_ROUND_TRIP_VALUES)
    }
    path = tmp_path / "nested" / "results.json"
    export_to_json(results, path, include_timeseries=True)

    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, result in results.items():
        exported = payload[name]
        summary = result.summary[_METRIC]
        assert exported["seeds"] == result.seeds
        assert exported["summary"][_METRIC] == {
            "mean": summary.mean,
            "std": summary.std,
            "min": summary.min,
            "max": summary.max,
            "n_seeds": summary.n_seeds,
            "values": summary.values.tolist(),
        }
        assert exported["timeseries"][_METRIC] == result.metric_arrays[_METRIC].tolist()


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize(
    ("format_name", "surface", "include_timeseries"),
    [
        ("csv", "summary", False),
        ("csv", "values", False),
        ("csv", "timeseries", True),
        ("json", "summary", False),
        ("json", "values", False),
        ("json", "timeseries", True),
    ],
)
def test_nonfinite_export_rejects_before_any_destination_mutation(
    format_name: str,
    surface: str,
    include_timeseries: bool,
    number: float,
    tmp_path: Path,
) -> None:
    result = _result_with_nonfinite(surface, number)

    def export(path: Path) -> None:
        if format_name == "csv":
            export_to_csv(
                {"invalid": result},
                path,
                include_timeseries=include_timeseries,
            )
        else:
            export_to_json(
                {"invalid": result},
                path,
                include_timeseries=include_timeseries,
            )

    suffix = format_name
    existing = tmp_path / f"existing.{suffix}"
    sentinel = "existing artifact\n"
    existing.write_text(sentinel, encoding="utf-8")
    with pytest.raises(ValueError):
        export(existing)
    assert existing.read_text(encoding="utf-8") == sentinel

    absent = tmp_path / "not-created" / f"absent.{suffix}"
    with pytest.raises(ValueError):
        export(absent)
    assert not absent.exists()
    assert not absent.parent.exists()


@pytest.mark.parametrize("mode", ["summary_csv", "timeseries_csv", "json"])
@pytest.mark.parametrize(
    "case",
    [
        "empty_results",
        "duplicate_seeds",
        "empty_seeds",
        "bool_seed",
        "numpy_integer_seed",
        "negative_seed",
        "seed_above_uint32",
        "zero_seed_axis",
        "zero_step_axis",
        "metric_ndim",
        "metric_row_count",
        "summary_values_ndim",
        "summary_count",
        "summary_value_count",
        "nonfinite_unselected_array",
        "nonfinite_unselected_summary",
        "empty_metric_arrays",
        "empty_summary",
    ],
)
def test_shared_preflight_rejects_invalid_aggregate_before_filesystem_mutation(
    mode: str,
    case: str,
    tmp_path: Path,
) -> None:
    results = _preflight_invalid_results(case)
    suffix = "json" if mode == "json" else "csv"

    existing = tmp_path / f"existing.{suffix}"
    sentinel = "existing artifact\n"
    existing.write_text(sentinel, encoding="utf-8")
    with pytest.raises(ValueError):
        _export_mode(mode, results, existing)
    assert existing.read_text(encoding="utf-8") == sentinel

    absent = tmp_path / "not-created" / f"absent.{suffix}"
    with pytest.raises(ValueError):
        _export_mode(mode, results, absent)
    assert not absent.exists()
    assert not absent.parent.exists()


@pytest.mark.parametrize("include_timeseries", [False, True], ids=["summary", "timeseries"])
def test_csv_preflight_requires_requested_metric_in_every_aggregate(
    include_timeseries: bool,
    tmp_path: Path,
) -> None:
    valid = _constant_result("valid")
    missing = _constant_result("missing")
    if include_timeseries:
        missing = missing._replace(metric_arrays={"other": missing.metric_arrays[_METRIC]})
    else:
        missing = missing._replace(summary={"other": missing.summary[_METRIC]})
    results = {"valid": valid, "missing": missing}

    existing = tmp_path / "existing.csv"
    sentinel = "existing artifact\n"
    existing.write_text(sentinel, encoding="utf-8")
    with pytest.raises(ValueError, match="requested metric"):
        export_to_csv(
            results,
            existing,
            metric=_METRIC,
            include_timeseries=include_timeseries,
        )
    assert existing.read_text(encoding="utf-8") == sentinel

    absent = tmp_path / "not-created" / "absent.csv"
    with pytest.raises(ValueError, match="requested metric"):
        export_to_csv(
            results,
            absent,
            metric=_METRIC,
            include_timeseries=include_timeseries,
        )
    assert not absent.parent.exists()


@pytest.mark.parametrize("mode", ["json", "csv"])
def test_preflight_rejects_metric_surface_mismatch(mode: str, tmp_path: Path) -> None:
    valid = _constant_result("mismatch")
    mismatched = valid._replace(
        metric_arrays={**valid.metric_arrays, "extra": valid.metric_arrays[_METRIC]}
    )
    destination = tmp_path / f"mismatch.{mode}"

    with pytest.raises(ValueError, match="same metric names"):
        if mode == "json":
            export_to_json({"mismatch": mismatched}, destination, include_timeseries=True)
        else:
            export_to_csv({"mismatch": mismatched}, destination, metric=_METRIC)

    assert not destination.exists()


@pytest.mark.parametrize("case", ["empty_results", "zero_step_axis", "missing_metric"])
def test_report_preflight_rejects_before_output_directory_mutation(
    case: str,
    tmp_path: Path,
) -> None:
    if case == "missing_metric":
        valid = _constant_result("valid")
        missing = _constant_result("missing")._replace(summary={})
        results = {"valid": valid, "missing": missing}
    else:
        results = _preflight_invalid_results(case)

    absent = tmp_path / "absent-report"
    with pytest.raises(ValueError):
        save_experiment_report(results, absent, "invalid", metric=_METRIC)
    assert not absent.exists()

    existing = tmp_path / "existing-report"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("existing artifact\n", encoding="utf-8")
    with pytest.raises(ValueError):
        save_experiment_report(results, existing, "invalid", metric=_METRIC)
    assert sentinel.read_text(encoding="utf-8") == "existing artifact\n"
    assert list(existing.iterdir()) == [sentinel]


@pytest.mark.parametrize("mode", ["summary_csv", "timeseries_csv", "json"])
def test_shared_preflight_accepts_uint32_seed_boundaries(mode: str, tmp_path: Path) -> None:
    result = _timeseries_result()._replace(seeds=[0, (1 << 32) - 1])
    suffix = "json" if mode == "json" else "csv"
    path = tmp_path / "nested" / f"valid.{suffix}"

    _export_mode(mode, {"trace": result}, path)

    assert path.is_file()
    if mode == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["trace"]["seeds"] == [0, (1 << 32) - 1]


def test_timeseries_csv_rejects_seed_row_misalignment_without_overwriting(
    tmp_path: Path,
) -> None:
    result = _timeseries_result()._replace(seeds=[17])
    path = tmp_path / "timeseries.csv"
    path.write_text("existing artifact\n", encoding="utf-8")

    with pytest.raises(ValueError, match="seed count"):
        export_to_csv({"trace": result}, path, include_timeseries=True)

    assert path.read_text(encoding="utf-8") == "existing artifact\n"


def test_timeseries_csv_rejects_duplicate_seed_headers_without_overwriting(
    tmp_path: Path,
) -> None:
    result = _timeseries_result()._replace(seeds=[17, 17])
    path = tmp_path / "timeseries.csv"
    path.write_text("existing artifact\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unique seeds"):
        export_to_csv({"trace": result}, path, include_timeseries=True)

    assert path.read_text(encoding="utf-8") == "existing artifact\n"


def test_atomic_publish_failure_preserves_destination_and_removes_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.csv"
    path.write_text("existing artifact\n", encoding="utf-8")

    def reject_replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        raise OSError(f"cannot publish {source!s} to {destination!s}")

    monkeypatch.setattr(export_module.os, "replace", reject_replace)
    with pytest.raises(OSError, match="cannot publish"):
        export_to_csv({"candidate": _constant_result("candidate")}, path)

    assert path.read_text(encoding="utf-8") == "existing artifact\n"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("field", ["mean", "std"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_display_tables_reject_nonfinite_summaries(field: str, value: float) -> None:
    result = _constant_result("invalid")
    summary = result.summary[_METRIC]._replace(**{field: value})
    results = {"invalid": result._replace(summary={_METRIC: summary})}

    for render in (generate_latex_table, generate_markdown_table):
        with pytest.raises(ValueError, match="non-finite"):
            render(results)


@pytest.mark.parametrize("field", ["mean", "std"])
def test_display_tables_reject_float_subclass_without_calling_hook(field: str) -> None:
    calls = 0

    class InfiniteFloatWithFiniteCoercion(float):
        def __float__(self) -> float:
            nonlocal calls
            calls += 1
            return 0.0

    value = InfiniteFloatWithFiniteCoercion(math.inf)

    result = _constant_result("candidate", 1.0)
    summary = result.summary[_METRIC]._replace(**{field: value})
    results = {"candidate": result._replace(summary={_METRIC: summary})}

    for render in (generate_latex_table, generate_markdown_table):
        with pytest.raises(ValueError, match="non-canonical"):
            render(results)
    assert calls == 0


def test_display_tables_reject_empty_results() -> None:
    for render in (generate_latex_table, generate_markdown_table):
        with pytest.raises(ValueError, match="non-empty"):
            render({})


def test_display_only_tables_keep_four_decimal_presentation() -> None:
    results = {"candidate": _constant_result("candidate", 0.12345678901234566)}

    latex = generate_latex_table(results)
    markdown = generate_markdown_table(results)

    assert r"\textbf{0.1235} $\pm$ 0.0000" in latex
    assert "**0.1235** ± 0.0000" in markdown


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
@pytest.mark.parametrize("field", ["mean", "std", "min", "max"])
def test_export_rejects_boolean_summary_statistics(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    result = _constant_result("candidate", 1.0)
    summary = result.summary[_METRIC]._replace(**{field: value})
    results = {"candidate": result._replace(summary={_METRIC: summary})}

    with pytest.raises(ValueError, match="refusing to export boolean as numeric measurement"):
        export_to_csv(results, tmp_path / "summary.csv")

    with pytest.raises(ValueError, match="refusing to export boolean as numeric measurement"):
        export_to_json(results, tmp_path / "summary.json")

    with pytest.raises(ValueError, match="refusing to export boolean as numeric measurement"):
        generate_latex_table(results)

    with pytest.raises(ValueError, match="refusing to export boolean as numeric measurement"):
        generate_markdown_table(results)
