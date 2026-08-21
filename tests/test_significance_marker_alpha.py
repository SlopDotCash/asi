"""Regression coverage for #2227: significance markers must reflect each
result's stored decision threshold (alpha), not hardcoded raw-p tiers.

A result declared significant at its own corrected alpha rendered with no
marker (or a misleading tier) whenever alpha differed from 0.05.
"""


from alberta_framework.utils.export import (
    _get_md_significance_marker,
    _get_significance_marker,
)
from alberta_framework.utils.statistics import SignificanceResult


def _result(p_value: float, alpha: float, significant: bool = True) -> SignificanceResult:
    return SignificanceResult(
        test_name="wilcoxon",
        statistic=1.0,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=0.5,
        method_a="a",
        method_b="b",
    )


def test_significant_at_alpha_0_10_renders_marker() -> None:
    # Holm-corrected: significant at its own threshold alpha=0.10 with p=0.08.
    r = _result(p_value=0.08, alpha=0.10)
    assert _get_md_significance_marker("b", "a", {("b", "a"): r}) != ""
    assert _get_significance_marker("b", "a", {("b", "a"): r}) != ""


def test_strict_alpha_tier_scales() -> None:
    # Holm step-down alpha=0.0025 with p=0.0002 → '**' (p < alpha/10 = 0.00025).
    r = _result(p_value=0.0002, alpha=0.0025)
    assert _get_md_significance_marker("b", "a", {("b", "a"): r}) == " **"
    assert _get_significance_marker("b", "a", {("b", "a"): r}) == r"$^{**}$"
    # p < alpha/100 → '***'.
    r3 = _result(p_value=0.00002, alpha=0.0025)
    assert _get_md_significance_marker("b", "a", {("b", "a"): r3}) == " ***"


def test_medium_alpha_tier() -> None:
    # alpha=0.10, p=0.02 → '**' (p < alpha/10 = 0.01? no: 0.02 > 0.01 → '*')
    # p=0.02 < alpha=0.10 → single star.
    r = _result(p_value=0.02, alpha=0.10)
    assert _get_md_significance_marker("b", "a", {("b", "a"): r}) == " *"


def test_alpha_0_05_legacy_behavior() -> None:
    # Default uncorrected path keeps exact legacy strings.
    r = _result(p_value=0.03, alpha=0.05)
    assert _get_md_significance_marker("b", "a", {("b", "a"): r}) == " *"
    r2 = _result(p_value=0.004, alpha=0.05)
    assert _get_md_significance_marker("b", "a", {("b", "a"): r2}) == " **"
    r3 = _result(p_value=0.0004, alpha=0.05)
    assert _get_md_significance_marker("b", "a", {("b", "a"): r3}) == " ***"


def test_not_significant_no_marker() -> None:
    r = _result(p_value=0.08, alpha=0.05, significant=False)
    assert _get_md_significance_marker("b", "a", {("b", "a"): r}) == ""


def test_plot_marker_parity() -> None:
    from alberta_framework.utils.visualization import _get_significance_marker_for_plot

    r = _result(p_value=0.08, alpha=0.10)
    assert _get_significance_marker_for_plot("b", "a", {("b", "a"): r}) == "*"
    r2 = _result(p_value=0.0002, alpha=0.0025)
    assert _get_significance_marker_for_plot("b", "a", {("b", "a"): r2}) == "**"
    r3 = _result(p_value=0.00002, alpha=0.0025)
    assert _get_significance_marker_for_plot("b", "a", {("b", "a"): r3}) == "***"
