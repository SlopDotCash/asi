"""Publication-style renderer switches affect subsequent figure creation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
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
