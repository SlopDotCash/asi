"""Publication-style renderer switches affect subsequent figure creation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.texmanager import TexManager  # noqa: E402

from alberta_framework.utils import visualization  # noqa: E402

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
