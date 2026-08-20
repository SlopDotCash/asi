"""Protocol ceilings for visualization grid and smoothing lengths."""

from __future__ import annotations

import pytest

from alberta_framework.utils.visualization import (
    _VIS_GRID_MAX,
    plot_hyperparameter_heatmap,
    plot_learning_curves,
)


def test_documented_protocol_ceiling() -> None:
    assert _VIS_GRID_MAX == 10_000


def test_heatmap_rejects_oversized_axes() -> None:
    with pytest.raises(ValueError, match="heatmap axes"):
        plot_hyperparameter_heatmap(
            {},
            param1_name="a",
            param1_values=list(range(_VIS_GRID_MAX + 1)),
            param2_name="b",
            param2_values=[0],
        )


def test_learning_curves_reject_oversized_window() -> None:
    with pytest.raises(ValueError, match="window_size"):
        plot_learning_curves({}, window_size=_VIS_GRID_MAX + 1)
