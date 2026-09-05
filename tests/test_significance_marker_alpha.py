"""Significance markers must reflect each result's stored decision threshold.

The rendering helpers historically re-derived stars from hardcoded raw-p tiers
(0.05 / 0.01 / 0.001), discarding the operative corrected alpha stored in
``SignificanceResult.alpha``.  A Holm/Bonferroni-corrected comparison whose
threshold sits above 0.05 was silently dropped, and stars were keyed to
uncorrected tiers even when the operative claim was stricter.  Issue #2227.
"""

from __future__ import annotations

import pytest

from alberta_framework.utils.export import (
    _LEGACY_LATEX_LEGEND,
    _LEGACY_MD_LEGEND,
    _get_md_significance_marker,
    _get_significance_marker,
    _significance_legend_latex,
    _significance_legend_markdown,
)
from alberta_framework.utils.statistics import SignificanceResult
from alberta_framework.utils.visualization import _get_significance_marker_for_plot

pytestmark = pytest.mark.unit


def _result(p_value: float, alpha: float) -> SignificanceResult:
    return SignificanceResult(
        test_name="ttest",
        statistic=1.0,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=0.5,
        method_a="a",
        method_b="b",
    )


def _dict(p_value: float, alpha: float) -> dict:
    return {("b", "a"): _result(p_value, alpha)}


def test_corrected_alpha_above_default_gets_marker() -> None:
    # Significant at its own threshold alpha=0.10, above the old 0.05 tier.
    d = _dict(0.08, 0.10)
    assert _get_md_significance_marker("b", "a", d) == " *"
    assert _get_significance_marker("b", "a", d) == r"$^{*}$"
    assert _get_significance_marker_for_plot("b", "a", d) == "*"


def test_holm_threshold_stars_consistent_with_own_alpha() -> None:
    # p=0.002 is significant at alpha=0.0025 but NOT an order of magnitude
    # below it, so exactly one star -- not the two the raw-p 0.01 tier gave.
    d = _dict(0.002, 0.0025)
    assert _get_md_significance_marker("b", "a", d) == " *"


def test_corrected_alpha_tiers_scale_by_own_threshold() -> None:
    # p an order of magnitude below alpha -> two stars.
    d = _dict(0.0002, 0.0025)
    assert _get_md_significance_marker("b", "a", d) == " **"
    # p two orders of magnitude below alpha -> three stars.
    d = _dict(0.00002, 0.0025)
    assert _get_md_significance_marker("b", "a", d) == " ***"


def test_default_alpha_dict_preserves_legacy_markers() -> None:
    d = {
        ("b", "a"): _result(0.03, 0.05),
        ("c", "a"): _result(0.0005, 0.05),
    }
    assert _get_md_significance_marker("b", "a", d) == " *"
    assert _get_md_significance_marker("c", "a", d) == " ***"
    assert _get_significance_marker("b", "a", d) == r"$^{*}$"
    assert _get_significance_marker("c", "a", d) == r"$^{***}$"


def test_non_significant_is_empty_regardless_of_alpha() -> None:
    d = _dict(0.5, 0.05)
    assert _get_md_significance_marker("b", "a", d) == ""
    assert _get_significance_marker("b", "a", d) == ""
    assert _get_significance_marker_for_plot("b", "a", d) == ""


def test_legacy_legend_unchanged() -> None:
    d = _dict(0.03, 0.05)
    assert _significance_legend_markdown(d) == _LEGACY_MD_LEGEND
    assert _significance_legend_latex(d) == _LEGACY_LATEX_LEGEND


def test_dynamic_legend_not_hardcoded_and_lists_alpha() -> None:
    d = _dict(0.08, 0.10)
    legend = _significance_legend_markdown(d)
    assert "0.05" not in legend
    assert "0.1" in legend
    latex = _significance_legend_latex(d)
    assert "0.05" not in latex
    assert "0.1" in latex


def test_plot_marker_matches_markdown_marker() -> None:
    for p_value, alpha in [
        (0.08, 0.10),
        (0.002, 0.0025),
        (0.0002, 0.0025),
        (0.03, 0.05),
        (0.0005, 0.05),
    ]:
        d = _dict(p_value, alpha)
        md = _get_md_significance_marker("b", "a", d).strip()
        plot = _get_significance_marker_for_plot("b", "a", d)
        assert md == plot


def test_unrelated_alpha_does_not_change_default_comparison_marker() -> None:
    results = _dict(0.007, 0.05)
    assert _get_md_significance_marker("b", "a", results) == " **"
    results[("c", "a")] = _result(0.02, 0.10)
    assert _get_md_significance_marker("b", "a", results) == " **"
    assert _get_significance_marker("b", "a", results) == r"$^{**}$"
    assert _get_significance_marker_for_plot("b", "a", results) == "**"
    legend = _significance_legend_markdown(results)
    assert "alpha = 0.05" in legend
    assert "p < 0.01" in legend
