"""Publication-style renderer switches affect subsequent figure creation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.texmanager import TexManager  # noqa: E402

from alberta_framework.utils import visualization  # noqa: E402
from alberta_framework.utils.experiments import (  # noqa: E402
    AggregatedResults,
    MetricSummary,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_publication_style(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(visualization, "_current_style", visualization._current_style.copy())
    with matplotlib.rc_context():
        yield


def test_publication_style_can_switch_latex_off_and_back_on() -> None:
    visualization.set_publication_style(use_latex=True)
    assert plt.rcParams["text.usetex"] is True

    visualization.set_publication_style(use_latex=False)
    assert plt.rcParams["text.usetex"] is False

    visualization.set_publication_style(use_latex=True)
    assert plt.rcParams["text.usetex"] is True


def test_default_publication_style_renders_without_latex_after_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latex_calls = 0

    def unavailable_latex(*args: Any, **kwargs: Any) -> None:
        nonlocal latex_calls
        latex_calls += 1
        raise FileNotFoundError("LaTeX renderer is unavailable")

    monkeypatch.setattr(TexManager, "get_text_width_height_descent", unavailable_latex)
    visualization.set_publication_style(use_latex=True)
    visualization.set_publication_style()

    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [1, 2], label="candidate")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Squared Error")
        ax.set_title("Native text rendering")
        ax.legend()
        fig.canvas.draw()
        assert latex_calls == 0
        assert ax.get_xlabel() == "Time Step"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"formats": []}, "formats"),
        ({"formats": "png"}, "formats"),
        ({"formats": ["png", "png"]}, "duplicate"),
        ({"formats": ["png", "zzz"]}, "unsupported"),
        ({"formats": ["png", 3]}, "formats"),
        ({"dpi": 0}, "dpi"),
        ({"dpi": True}, "dpi"),
        ({"dpi": float("nan")}, "dpi"),
        ({"transparent": "no"}, "transparent"),
    ],
)
def test_save_figure_rejects_malformed_options_before_any_write(
    tmp_path: Path,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    fig = plt.figure()
    target = tmp_path / "figures" / "curve"
    try:
        with pytest.raises(ValueError, match=message):
            visualization.save_figure(fig, target, **kwargs)
    finally:
        plt.close(fig)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["lr_0.001", "curve.png"])
def test_save_figure_rejects_dotted_base_name_that_would_be_truncated(
    tmp_path: Path,
    name: str,
) -> None:
    fig = plt.figure()
    try:
        with pytest.raises(ValueError, match="without an extension"):
            visualization.save_figure(fig, tmp_path / name, formats=["png"])
    finally:
        plt.close(fig)
    assert list(tmp_path.iterdir()) == []


def test_save_figure_writes_every_requested_format(tmp_path: Path) -> None:
    fig = plt.figure()
    try:
        saved = visualization.save_figure(fig, str(tmp_path / "nested" / "curve"))
    finally:
        plt.close(fig)
    assert saved == [tmp_path / "nested" / "curve.pdf", tmp_path / "nested" / "curve.png"]
    assert all(path.stat().st_size > 0 for path in saved)


def _make_aggregated(
    name: str,
    squared_error: np.ndarray,
    *,
    mean_step_size: np.ndarray | None = None,
    std: float = 0.0,
) -> AggregatedResults:
    metric_arrays = {"squared_error": squared_error}
    if mean_step_size is not None:
        metric_arrays["mean_step_size"] = mean_step_size
    final_vals = np.asarray(squared_error[:, -1], dtype=np.float64)
    mean_val = float(np.mean(final_vals)) if np.all(np.isfinite(final_vals)) else 1.0
    return AggregatedResults(
        config_name=name,
        seeds=list(range(squared_error.shape[0])),
        metric_arrays=metric_arrays,
        summary={
            "squared_error": MetricSummary(
                mean=mean_val,
                std=std,
                min=mean_val,
                max=mean_val,
                n_seeds=squared_error.shape[0],
                values=np.ones(squared_error.shape[0], dtype=np.float64),
            )
        },
    )


def test_cumulative_error_accumulates_float32_traces_in_float64() -> None:
    err_f32 = np.full((2, 4), 2.0e38, dtype=np.float32)
    results = {"f32": _make_aggregated("f32", err_f32)}
    fig = visualization.create_comparison_figure(results)
    try:
        cumulative_ax = next(
            ax for ax in fig.axes if ax.get_title() == "Cumulative Error"
        )
        ydata = np.asarray(cumulative_ax.lines[0].get_ydata(), dtype=np.float64)
        expected = np.cumsum(err_f32[0].astype(np.float64), dtype=np.float64)
        np.testing.assert_allclose(ydata, expected)
    finally:
        plt.close(fig)


def test_plotting_helpers_reject_invalid_series_without_leaking_figures_or_mutating_ax() -> None:
    good = _make_aggregated(
        "good",
        np.ones((2, 4), dtype=np.float64),
        mean_step_size=np.full((2, 4), 0.01, dtype=np.float64),
    )
    bad = _make_aggregated(
        "bad",
        np.array([[1.0, np.nan, 1.0, 1.0], [1.0, 2.0, 1.0, 1.0]], dtype=np.float64),
        mean_step_size=np.array(
            [[0.01, np.nan, 0.01, 0.01], [0.01, 0.02, 0.01, 0.01]],
            dtype=np.float64,
        ),
    )
    negative_std = _make_aggregated(
        "neg",
        np.ones((2, 4), dtype=np.float64),
        std=-1.0,
    )

    before = tuple(plt.get_fignums())
    with pytest.raises(ValueError, match="finite"):
        visualization.plot_learning_curves({"good": good, "bad": bad}, window_size=2)
    with pytest.raises(ValueError, match="finite"):
        visualization.plot_step_size_evolution({"good": good, "bad": bad})
    with pytest.raises(ValueError, match="finite metric means and stds"):
        visualization.plot_final_performance_bars({"neg": negative_std})
    with pytest.raises(ValueError, match="finite"):
        visualization.create_comparison_figure({"good": good, "bad": bad})
    assert tuple(plt.get_fignums()) == before

    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="finite"):
            visualization.plot_learning_curves(
                {"good": good, "bad": bad}, window_size=2, ax=ax
            )
        assert len(ax.lines) == 0
        with pytest.raises(ValueError, match="finite"):
            visualization.plot_step_size_evolution(
                {"good": good, "bad": bad}, ax=ax
            )
        assert len(ax.lines) == 0
    finally:
        plt.close(fig)


def test_hyperparameter_heatmap_scales_mean_for_large_finite_cells() -> None:
    cells = {
        "a_0": 5.0e307,
        "a_1": 5.0e307,
        "b_0": 5.0e307,
        "b_1": 8.0e307,
    }
    results = {}
    for key, val in cells.items():
        agg = _make_aggregated(key, np.ones((2, 4), dtype=np.float64))
        results[key] = agg._replace(
            summary={
                "squared_error": agg.summary["squared_error"]._replace(mean=val)
            }
        )
    fig, ax = visualization.plot_hyperparameter_heatmap(
        results,
        param1_name="p1",
        param1_values=["a", "b"],
        param2_name="p2",
        param2_values=[0, 1],
    )
    try:
        colors = [text.get_color() for text in ax.texts]
        assert colors == ["black", "black", "black", "white"]
    finally:
        plt.close(fig)
