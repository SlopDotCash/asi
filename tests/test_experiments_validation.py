"""Contract tests for validating public multi-seed experiment inputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Never, cast

import numpy as np
import pytest

from alberta_framework.core.learners import LinearLearner
from alberta_framework.core.types import LearnerState
from alberta_framework.streams.base import ScanStream
from alberta_framework.streams.synthetic import RandomWalkStream
from alberta_framework.utils.experiments import (
    ExperimentConfig,
    SingleRunResult,
    aggregate_metrics,
    get_final_performance,
    get_metric_timeseries,
    run_multi_seed_experiment,
)
from alberta_framework.utils.export import generate_latex_table, generate_markdown_table
from alberta_framework.utils.statistics import compute_statistics

pytestmark = pytest.mark.unit


def _fail_if_called() -> Never:
    raise AssertionError("experiment factory must not be called")


def _stream_factory() -> ScanStream[Any]:
    return RandomWalkStream(feature_dim=2)


def _config(
    name: str,
    *,
    learner_factory: Callable[[], LinearLearner] = LinearLearner,
    stream_factory: Callable[[], ScanStream[Any]] = _stream_factory,
    num_steps: int = 2,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        learner_factory=learner_factory,
        stream_factory=stream_factory,
        num_steps=num_steps,
    )


def test_duplicate_names_reject_before_distinct_factories_execute() -> None:
    configs = [
        _config(
            "baseline",
            learner_factory=_fail_if_called,
            stream_factory=_fail_if_called,
            num_steps=1,
        ),
        _config(
            "baseline",
            learner_factory=_fail_if_called,
            stream_factory=_fail_if_called,
            num_steps=2,
        ),
    ]

    with pytest.raises(
        ValueError,
        match=r"^Experiment configuration names must be unique; duplicates: 'baseline'$",
    ):
        run_multi_seed_experiment(configs, seeds=[0, 1], parallel=False, show_progress=False)


def test_repeated_config_object_rejects_before_factory_executes() -> None:
    config = _config(
        "baseline",
        learner_factory=_fail_if_called,
        stream_factory=_fail_if_called,
    )

    with pytest.raises(
        ValueError,
        match=r"^Experiment configuration names must be unique; duplicates: 'baseline'$",
    ):
        run_multi_seed_experiment([config, config], seeds=[0], parallel=False, show_progress=False)


def test_multiple_duplicate_names_are_reported_deterministically() -> None:
    configs = [
        _config(
            name,
            learner_factory=_fail_if_called,
            stream_factory=_fail_if_called,
        )
        for name in ("zeta", "alpha", "zeta", "beta", "alpha", "beta", "zeta")
    ]

    with pytest.raises(ValueError) as exc_info:
        run_multi_seed_experiment(configs, seeds=[0], parallel=False, show_progress=False)

    assert str(exc_info.value) == (
        "Experiment configuration names must be unique; duplicates: 'alpha', 'beta', 'zeta'"
    )


def test_unique_names_preserve_config_and_seed_order() -> None:
    results = run_multi_seed_experiment(
        [_config("second"), _config("first")],
        seeds=[7, 3],
        parallel=False,
        show_progress=False,
    )

    assert list(results) == ["second", "first"]
    assert results["second"].seeds == [7, 3]
    assert results["first"].seeds == [7, 3]
    assert all(
        summary.n_seeds == 2
        for result in results.values()
        for summary in result.summary.values()
    )


def test_empty_config_sequence_still_returns_empty_results() -> None:
    assert (
        run_multi_seed_experiment([], seeds=[0], parallel=False, show_progress=False) == {}
    )


# ---------------------------------------------------------------------------
# Across-seed spread: sample std (ddof=1), matching compute_statistics
# ---------------------------------------------------------------------------

# Issue #33 falsifier: population std understates sample std by sqrt((n-1)/n).
_SPREAD_VALUES = [0.10, 0.12, 0.30]
_POPULATION_STD = float(np.std(_SPREAD_VALUES))  # ddof=0 ~ 0.0899
_SAMPLE_STD = float(np.std(_SPREAD_VALUES, ddof=1))  # ddof=1 ~ 0.1102


def _constant_run(seed: int, value: float, *, n_steps: int = 1) -> SingleRunResult:
    """One seed whose metric is constant, so the final-window mean equals `value`."""
    return SingleRunResult(
        config_name="baseline",
        seed=seed,
        metrics_history=[{"squared_error": value} for _ in range(n_steps)],
        final_state=cast(LearnerState, None),
    )


def test_spread_estimators_disagree_on_issue_33_falsifier() -> None:
    """Document the two estimators; publication paths must pick sample std."""
    stats = compute_statistics(_SPREAD_VALUES)
    assert _POPULATION_STD == pytest.approx(0.08993825042154695)
    assert _SAMPLE_STD == pytest.approx(0.11015141094572203)
    assert _SAMPLE_STD != pytest.approx(_POPULATION_STD)
    assert stats.std == pytest.approx(_SAMPLE_STD)
    assert stats.std != pytest.approx(_POPULATION_STD)


def test_aggregate_metrics_std_matches_compute_statistics_sample_std() -> None:
    agg = aggregate_metrics(
        [_constant_run(seed, value) for seed, value in enumerate(_SPREAD_VALUES)]
    )
    summary = agg.summary["squared_error"]

    assert summary.std == pytest.approx(_SAMPLE_STD)
    assert summary.std == pytest.approx(compute_statistics(summary.values).std)
    assert summary.std != pytest.approx(_POPULATION_STD)
    assert summary.mean == pytest.approx(float(np.mean(_SPREAD_VALUES)))
    assert summary.n_seeds == 3


def test_aggregate_metrics_std_is_across_seed_window_means() -> None:
    """Std is of per-seed last-window means, not of flattened within-run steps."""
    runs = [
        SingleRunResult(
            config_name="baseline",
            seed=0,
            metrics_history=[{"squared_error": 0.00}, {"squared_error": 0.20}],
            final_state=cast(LearnerState, None),
        ),
        SingleRunResult(
            config_name="baseline",
            seed=1,
            metrics_history=[{"squared_error": 0.04}, {"squared_error": 0.20}],
            final_state=cast(LearnerState, None),
        ),
        SingleRunResult(
            config_name="baseline",
            seed=2,
            metrics_history=[{"squared_error": 0.20}, {"squared_error": 0.40}],
            final_state=cast(LearnerState, None),
        ),
    ]
    summary = aggregate_metrics(runs).summary["squared_error"]
    assert list(summary.values) == pytest.approx(_SPREAD_VALUES)
    assert summary.std == pytest.approx(_SAMPLE_STD)


def test_aggregate_metrics_single_seed_std_is_zero_not_nan() -> None:
    summary = aggregate_metrics([_constant_run(0, 0.10)]).summary["squared_error"]
    assert summary.std == 0.0
    assert np.isfinite(summary.std)
    assert summary.std == compute_statistics([0.10]).std
    assert summary.n_seeds == 1


def test_get_final_performance_uses_sample_std() -> None:
    agg = aggregate_metrics(
        [_constant_run(seed, value) for seed, value in enumerate(_SPREAD_VALUES)]
    )
    mean, std = get_final_performance({"baseline": agg})["baseline"]
    assert mean == pytest.approx(float(np.mean(_SPREAD_VALUES)))
    assert std == pytest.approx(_SAMPLE_STD)
    assert std == pytest.approx(compute_statistics(_SPREAD_VALUES).std)
    assert std != pytest.approx(_POPULATION_STD)


def test_get_final_performance_single_seed_std_is_zero_not_nan() -> None:
    agg = aggregate_metrics([_constant_run(0, 0.10)])
    mean, std = get_final_performance({"baseline": agg})["baseline"]
    assert mean == pytest.approx(0.10)
    assert std == 0.0
    assert np.isfinite(std)


def test_publication_tables_print_sample_std() -> None:
    agg = aggregate_metrics(
        [_constant_run(seed, value) for seed, value in enumerate(_SPREAD_VALUES)]
    )
    results = {"baseline": agg}
    markdown = generate_markdown_table(results)
    latex = generate_latex_table(results)
    assert f"{_SAMPLE_STD:.4f}" in markdown
    assert f"{_SAMPLE_STD:.4f}" in latex
    assert f"{_POPULATION_STD:.4f}" not in markdown
    assert f"{_POPULATION_STD:.4f}" not in latex


def test_get_metric_timeseries_remains_population_sd_band() -> None:
    """Issue #33 does not change the per-step ±1 SD band (explicitly out of #26)."""
    agg = aggregate_metrics(
        [_constant_run(seed, value) for seed, value in enumerate(_SPREAD_VALUES)]
    )
    mean, lower, upper = get_metric_timeseries(agg, "squared_error")
    assert mean[0] == pytest.approx(float(np.mean(_SPREAD_VALUES)))
    assert lower[0] == pytest.approx(mean[0] - _POPULATION_STD)
    assert upper[0] == pytest.approx(mean[0] + _POPULATION_STD)
