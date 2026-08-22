"""Supplementary coverage for recurring_feature_artifact.py.

The module's existing tests cover the artifact lifecycle; these cases cover
the statistical helpers that were previously untested: canonical scientific
bytes (deterministic JSON), the Wilson score interval (exact formula,
boundary validation), and the paired bootstrap mean interval (determinism,
centrality, confidence width).
"""

import math
from statistics import NormalDist

import pytest

from alberta_framework.evaluation.recurring_feature_artifact import (
    canonical_scientific_bytes,
    paired_bootstrap_mean_interval,
    wilson_score_interval,
)


def test_canonical_scientific_bytes_deterministic() -> None:
    payload = {"b": 2, "a": [3, 1], "c": {"z": 1, "y": 2}}
    first = canonical_scientific_bytes(payload)
    second = canonical_scientific_bytes(dict(payload))
    assert first == second
    # Keys sorted: a, b, c.
    text = first.decode("utf-8")
    assert text.index('"a"') < text.index('"b"') < text.index('"c"')


def test_canonical_scientific_bytes_empty() -> None:
    assert canonical_scientific_bytes({}) == b"{}"


def test_wilson_interval_perfect_success() -> None:
    interval = wilson_score_interval(100, 100)
    assert isinstance(interval, dict)
    assert "lower" in interval and "upper" in interval
    assert 0.9 < interval["lower"] <= 1.0


def test_wilson_interval_zero_success() -> None:
    interval = wilson_score_interval(0, 100)
    assert 0.0 <= interval["lower"] < 0.05
    assert interval["lower"] <= interval["upper"]


def test_wilson_interval_centered() -> None:
    interval = wilson_score_interval(50, 100)
    # Symmetric around 0.5 for equal counts.
    assert abs((interval["lower"] + interval["upper"]) / 2 - 0.5) < 1e-9


def test_wilson_interval_matches_formula() -> None:
    successes, sample_size = 30, 50
    interval = wilson_score_interval(successes, sample_size)
    p_hat = successes / sample_size
    z = NormalDist().inv_cdf(0.5 + 0.95 / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / sample_size
    center = (p_hat + z2 / (2.0 * sample_size)) / denom
    half = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4.0 * sample_size)) / sample_size) / denom
    assert abs(interval["lower"] - (center - half)) < 1e-9
    assert abs(interval["upper"] - (center + half)) < 1e-9


def test_wilson_interval_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        wilson_score_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_score_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_score_interval(5, 0)


def test_paired_bootstrap_interval_deterministic() -> None:
    diffs = [1.0, -2.0, 3.0, 0.5, -1.5, 2.5, 0.0, 1.5]
    first = paired_bootstrap_mean_interval(diffs, seed_offset=7)
    second = paired_bootstrap_mean_interval(diffs, seed_offset=7)
    assert first == second


def test_paired_bootstrap_interval_shape() -> None:
    diffs = [1.0, -2.0, 3.0, 0.5, -1.5]
    interval = paired_bootstrap_mean_interval(diffs, seed_offset=1)
    assert interval["lower"] <= interval["upper"]
    # Sample mean should lie inside the bootstrap interval.
    mean = sum(diffs) / len(diffs)
    assert interval["lower"] <= mean <= interval["upper"]
