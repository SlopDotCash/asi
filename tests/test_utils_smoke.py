"""Smoke tests for utils.timing and utils.visualization.

Visualization tests run on the headless Agg backend, operate purely on
in-memory figure objects, and perform no file I/O.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alberta_framework.utils.experiments import (  # noqa: E402
    AggregatedResults,
    MetricSummary,
)
from alberta_framework.utils.timing import Timer, format_duration  # noqa: E402
from alberta_framework.utils.visualization import (  # noqa: E402
    create_comparison_figure,
    plot_final_performance_bars,
    plot_hyperparameter_heatmap,
    plot_learning_curves,
    plot_step_size_evolution,
    set_publication_style,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def test_format_duration_branches() -> None:
    assert format_duration(0.5) == "0.50s"
    assert format_duration(90.5) == "1m 30.50s"
    assert format_duration(3665) == "1h 1m 5.00s"
    assert format_duration(59.999) == "1m 0.00s"
    assert format_duration(3599.999) == "1h 0m 0.00s"


def test_timer_monotonic_accumulation() -> None:
    with Timer("smoke", verbose=False) as timer:
        first = timer.elapsed()
        time.sleep(0.01)
        second = timer.elapsed()

    assert 0.0 <= first <= second
    assert timer.end_time >= timer.start_time
    assert timer.duration >= second
    assert timer.duration == pytest.approx(timer.end_time - timer.start_time)

    # A second timed block accumulates independently and stays non-negative.
    with Timer("smoke-2", verbose=False) as second_timer:
        time.sleep(0.001)
    assert second_timer.duration > 0.0
    assert second_timer.start_time >= timer.end_time


def test_timer_print_fn_and_repr() -> None:
    messages: list[str] = []
    with Timer("Custom op", print_fn=messages.append):
        pass

    assert len(messages) == 1
    assert messages[0].startswith("Custom op completed in ")

    silent = Timer("silent", verbose=False)
    assert repr(silent) == "Timer(name='silent')"
    with silent:
        time.sleep(0.001)
    assert "duration=" in repr(silent)


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------


def _aggregated(name: str, seed_offset: int, n_steps: int = 120) -> AggregatedResults:
    rng = np.random.default_rng(seed_offset)
    error = rng.uniform(0.1, 1.0, size=(3, n_steps))
    step_size = rng.uniform(0.01, 0.1, size=(3, n_steps))
    finals = error[:, -1]
    summary = {
        "squared_error": MetricSummary(
            mean=float(finals.mean()),
            std=float(finals.std()),
            min=float(finals.min()),
            max=float(finals.max()),
            n_seeds=3,
            values=finals,
        )
    }
    return AggregatedResults(
        config_name=name,
        seeds=[0, 1, 2],
        metric_arrays={"squared_error": error, "mean_step_size": step_size},
        summary=summary,
    )


def _aggregated_from_error(name: str, error: np.ndarray) -> AggregatedResults:
    finals = error[:, -1]
    return AggregatedResults(
        config_name=name,
        seeds=list(range(error.shape[0])),
        metric_arrays={"squared_error": error},
        summary={
            "squared_error": MetricSummary(
                mean=float(finals.mean()),
                std=float(finals.std()),
                min=float(finals.min()),
                max=float(finals.max()),
                n_seeds=error.shape[0],
                values=finals,
            )
        },
    )


@pytest.fixture
def results() -> dict[str, AggregatedResults]:
    return {"lms": _aggregated("lms", 0), "idbd": _aggregated("idbd", 1)}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_learning_curves_returns_figure_and_axes(results) -> None:
    fig, ax = plot_learning_curves(results, window_size=5)
    assert isinstance(fig, plt.Figure)
    assert ax.figure is fig
    assert len(ax.lines) == 2
    assert {line.get_label() for line in ax.lines} == {"lms", "idbd"}


def test_plot_learning_curves_omits_future_informed_running_mean_padding() -> None:
    values = np.arange(6, dtype=float)[None, :]
    result = _aggregated_from_error("linear", values)

    _, ax = plot_learning_curves(
        {"linear": result},
        window_size=3,
        show_ci=False,
        log_scale=False,
    )

    np.testing.assert_array_equal(ax.lines[0].get_xdata(), np.arange(2, 6))
    np.testing.assert_allclose(ax.lines[0].get_ydata(), (1.0, 2.0, 3.0, 4.0))


def test_plot_learning_curves_preserves_short_input_coordinates() -> None:
    result = _aggregated_from_error("short", np.array([[2.0, 4.0]]))

    _, ax = plot_learning_curves(
        {"short": result},
        window_size=3,
        show_ci=False,
        log_scale=False,
    )

    np.testing.assert_array_equal(ax.lines[0].get_xdata(), (0, 1))
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), (2.0, 4.0))


def test_plot_learning_curves_aligns_multiseed_confidence_band() -> None:
    values = np.array(
        [
            [0.0, 2.0, 4.0, 6.0],
            [2.0, 4.0, 6.0, 8.0],
        ]
    )
    result = _aggregated_from_error("multiseed", values)

    _, ax = plot_learning_curves(
        {"multiseed": result},
        window_size=2,
        show_ci=True,
        log_scale=False,
    )

    expected_steps = np.arange(1, 4)
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), expected_steps)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), (2.0, 4.0, 6.0))
    assert len(ax.collections) == 1
    band_x = np.unique(ax.collections[0].get_paths()[0].vertices[:, 0])
    np.testing.assert_array_equal(band_x, expected_steps)


def test_create_comparison_figure_uses_causal_learning_curve_coordinates() -> None:
    values = np.vstack(
        [
            np.arange(1.0, 103.0),
            np.arange(3.0, 105.0),
        ]
    )
    result = _aggregated_from_error("comparison", values)

    fig = create_comparison_figure({"comparison": result})
    learning_ax = next(ax for ax in fig.axes if ax.get_title() == "Learning Curves")

    np.testing.assert_array_equal(learning_ax.lines[0].get_xdata(), (99, 100, 101))
    np.testing.assert_allclose(learning_ax.lines[0].get_ydata(), (51.5, 52.5, 53.5))


def test_plot_final_performance_bars(results) -> None:
    fig, ax = plot_final_performance_bars(results)
    assert isinstance(fig, plt.Figure)
    assert len(ax.patches) == 2
    assert [label.get_text() for label in ax.get_xticklabels()] == ["lms", "idbd"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("mean", float("nan")), ("std", float("inf"))],
)
def test_plot_final_performance_bars_rejects_nonfinite_statistic_without_figure(
    results, field: str, value: float
) -> None:
    """np.argmin treats NaN as the best bar, so a failed run would be crowned."""
    poisoned = dict(results)
    summary = results["lms"].summary["squared_error"]
    poisoned["lms"] = results["lms"]._replace(
        summary={
            "squared_error": summary._replace(**{field: value}),
        }
    )
    before = tuple(plt.get_fignums())

    with pytest.raises(ValueError, match="finite metric means and stds"):
        plot_final_performance_bars(poisoned)
    assert tuple(plt.get_fignums()) == before


def test_plot_final_performance_bars_rejects_empty_results_without_figure() -> None:
    before = tuple(plt.get_fignums())

    with pytest.raises(ValueError, match="at least one result"):
        plot_final_performance_bars({})
    assert tuple(plt.get_fignums()) == before


def test_plot_step_size_evolution(results) -> None:
    fig, ax = plot_step_size_evolution(results)
    assert isinstance(fig, plt.Figure)
    assert len(ax.lines) == 2
    assert ax.get_yscale() == "log"


def test_plot_hyperparameter_heatmap_handles_missing_cells(results) -> None:
    fig, ax = plot_hyperparameter_heatmap(
        results,
        param1_name="optimizer",
        param1_values=["lms", "idbd"],
        param2_name="suffix",
        param2_values=["", "_missing"],
        name_pattern="{p1}{p2}",
    )
    assert isinstance(fig, plt.Figure)
    assert len(ax.images) == 1
    # Only the first column resolves to a known config; the second is NaN.
    data = np.ma.filled(np.asarray(ax.images[0].get_array(), dtype=float), np.nan)
    assert np.isfinite(data[:, 0]).all()
    assert np.isnan(data[:, 1]).all()


def test_create_comparison_figure_has_four_panels(results) -> None:
    fig = create_comparison_figure(results)
    assert isinstance(fig, plt.Figure)
    # 2x2 panels plus the heatmap-free layout; colorbars would add axes.
    assert len(fig.axes) == 4
    titles = {ax.get_title() for ax in fig.axes}
    assert "Learning Curves" in titles
    assert "Final Performance" in titles


def _bars_with_means(low: float, high: float) -> dict[str, AggregatedResults]:
    return {
        "low": _aggregated_from_error("low", np.full((2, 4), low)),
        "high": _aggregated_from_error("high", np.full((2, 4), high)),
    }


def _gold_index(ax: plt.Axes) -> int:
    gold = mcolors.to_rgba("gold")
    for index, patch in enumerate(ax.patches):
        if patch.get_linewidth() >= 2 and np.allclose(patch.get_edgecolor(), gold, atol=1e-3):
            return index
    raise AssertionError("no gold-marked bar")


def test_plot_final_performance_bars_rejects_string_false_without_crowning_loser() -> None:
    """A non-empty string is truthy, so 'false' used to mark the lowest mean gold."""
    results = _bars_with_means(1.0, 2.0)
    before = tuple(plt.get_fignums())

    with pytest.raises(ValueError, match="exact bool"):
        plot_final_performance_bars(results, lower_is_better="false")
    assert tuple(plt.get_fignums()) == before


@pytest.mark.parametrize("value", [1, 0, "true", np.bool_(True)])
def test_plot_final_performance_bars_rejects_non_bool_identities_without_figure(
    value: object,
) -> None:
    results = _bars_with_means(1.0, 2.0)
    before = tuple(plt.get_fignums())

    with pytest.raises(ValueError, match="exact bool"):
        plot_final_performance_bars(results, lower_is_better=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact bool"):
        plot_final_performance_bars(results, show_significance=value)  # type: ignore[arg-type]
    assert tuple(plt.get_fignums()) == before


def test_plot_final_performance_bars_marks_highest_when_lower_is_better_is_false() -> None:
    results = _bars_with_means(1.0, 2.0)
    _, ax = plot_final_performance_bars(results, lower_is_better=False, show_significance=False)
    assert _gold_index(ax) == 1


def test_plot_final_performance_bars_marks_lowest_when_lower_is_better_is_true() -> None:
    results = _bars_with_means(1.0, 2.0)
    _, ax = plot_final_performance_bars(results, lower_is_better=True, show_significance=False)
    assert _gold_index(ax) == 0


def test_plot_learning_curves_rejects_string_false_ci_without_drawing_band(results) -> None:
    before = tuple(plt.get_fignums())

    with pytest.raises(ValueError, match="exact bool"):
        plot_learning_curves(results, show_ci="false", window_size=5)
    assert tuple(plt.get_fignums()) == before


def test_plot_learning_curves_omits_band_when_show_ci_is_false(results) -> None:
    _, ax = plot_learning_curves(results, show_ci=False, log_scale=False, window_size=5)
    assert len(ax.collections) == 0


def test_plot_step_size_evolution_rejects_non_bool_show_ci_without_figure(results) -> None:
    before = tuple(plt.get_fignums())

    with pytest.raises(ValueError, match="exact bool"):
        plot_step_size_evolution(results, show_ci="false")
    assert tuple(plt.get_fignums()) == before


def test_plot_hyperparameter_heatmap_rejects_non_bool_lower_is_better(results) -> None:
    before = tuple(plt.get_fignums())

    with pytest.raises(ValueError, match="exact bool"):
        plot_hyperparameter_heatmap(
            results,
            param1_name="optimizer",
            param1_values=["lms", "idbd"],
            param2_name="suffix",
            param2_values=[""],
            name_pattern="{p1}{p2}",
            lower_is_better="false",
        )
    assert tuple(plt.get_fignums()) == before


def test_set_publication_style_rejects_non_bool_and_nonfinite_hosts() -> None:
    before = dict(plt.rcParams)
    try:
        with pytest.raises(ValueError, match="exact bool"):
            set_publication_style(use_latex=1)
        with pytest.raises(ValueError, match="real number"):
            set_publication_style(font_size=True)
        with pytest.raises(ValueError, match="finite positive"):
            set_publication_style(figure_width=float("nan"))
    finally:
        plt.rcParams.update(before)


def test_set_publication_style_keeps_legal_hosts() -> None:
    before = dict(plt.rcParams)
    try:
        set_publication_style(font_size=10, use_latex=False, figure_width=3.5)
        assert plt.rcParams["font.size"] == 10
        assert plt.rcParams["text.usetex"] is False
    finally:
        plt.rcParams.update(before)


def test_plot_final_performance_bars_significance_offset_on_negative_metrics() -> None:
    """Significance markers must sit strictly above the error bar for negative metrics."""
    from alberta_framework.utils.statistics import SignificanceResult

    results = {
        "best": _aggregated_from_error("best", np.full((2, 4), -100.0)),
        "suboptimal": _aggregated_from_error("suboptimal", np.full((2, 4), -150.0)),
    }
    # For squared_error, override the summary with negative means and positive stds
    summary_best = MetricSummary(
        mean=-100.0, std=5.0, min=-110.0, max=-90.0, n_seeds=2, values=np.array([-105.0, -95.0])
    )
    summary_sub = MetricSummary(
        mean=-150.0, std=10.0, min=-165.0, max=-135.0, n_seeds=2, values=np.array([-160.0, -140.0])
    )
    results["best"] = results["best"]._replace(summary={"squared_error": summary_best})
    results["suboptimal"] = results["suboptimal"]._replace(summary={"squared_error": summary_sub})

    sig_results = {
        ("suboptimal", "best"): SignificanceResult(
            test_name="t_test",
            statistic=-5.0,
            p_value=0.0005,
            significant=True,
            alpha=0.05,
            effect_size=2.0,
            method_a="suboptimal",
            method_b="best",
        )
    }

    fig, ax = plot_final_performance_bars(
        results,
        metric="squared_error",
        show_significance=True,
        significance_results=sig_results,
        lower_is_better=False,  # -100 is better than -150
    )
    assert isinstance(fig, plt.Figure)
    annotations = [t for t in ax.texts if t.get_text() == "***"]
    assert len(annotations) == 1
    ann = annotations[0]
    # Bar top + std for suboptimal is -150.0 + 10.0 = -140.0.
    # The annotation position y must be strictly above -140.0.
    bar_top = -150.0 + 10.0
    ann_x, ann_y = ann.xy
    assert ann_x == 1  # suboptimal is second bar
    assert ann_y > bar_top, f"Annotation y ({ann_y}) must be above bar top ({bar_top})"


def test_plot_hyperparameter_heatmap_handles_missing_metric_in_summary(results) -> None:
    """Heatmap sets nan without KeyError when requested metric is missing in summary."""
    fig, ax = plot_hyperparameter_heatmap(
        results,
        param1_name="optimizer",
        param1_values=["lms", "idbd"],
        param2_name="suffix",
        param2_values=[""],
        name_pattern="{p1}{p2}",
        metric="nonexistent_metric",
    )
    assert isinstance(fig, plt.Figure)
    data = np.ma.filled(np.asarray(ax.images[0].get_array(), dtype=float), np.nan)
    assert np.isnan(data).all()
