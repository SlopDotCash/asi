"""Plot helpers must refuse a request that leaves nothing to draw."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alberta_framework.utils import visualization  # noqa: E402
from alberta_framework.utils.experiments import AggregatedResults  # noqa: E402

pytestmark = pytest.mark.unit


def _aggregate(metric: str) -> AggregatedResults:
    return AggregatedResults(
        config_name="candidate",
        seeds=[0, 1],
        metric_arrays={metric: np.ones((2, 4), dtype=np.float64)},
        summary={},
    )


def test_step_size_plot_rejects_a_metric_no_result_contains() -> None:
    figures = plt.get_fignums()
    with pytest.raises(ValueError, match="no result with metric 'mean_stepsize'"):
        visualization.plot_step_size_evolution(
            {"candidate": _aggregate("mean_step_size")}, metric="mean_stepsize"
        )
    assert plt.get_fignums() == figures


def test_step_size_plot_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="no result with metric"):
        visualization.plot_step_size_evolution({})


def test_step_size_plot_still_skips_a_result_without_the_metric() -> None:
    fig, ax = visualization.plot_step_size_evolution(
        {
            "candidate": _aggregate("mean_step_size"),
            "control": _aggregate("squared_error"),
        }
    )
    try:
        assert [line.get_label() for line in ax.get_lines()] == ["candidate"]
    finally:
        plt.close(fig)


def test_learning_curves_reject_empty_results() -> None:
    figures = plt.get_fignums()
    with pytest.raises(ValueError, match="at least one result"):
        visualization.plot_learning_curves({})
    assert plt.get_fignums() == figures
