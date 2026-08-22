"""Supplementary coverage for continual_multiagent_artifact.py helpers.

Covers previously untested helpers: wilson_score_interval (with configurable
confidence level) and canonical_content_bytes (compact sorted JSON).
"""

import json
import math
from statistics import NormalDist

import pytest

from alberta_framework.evaluation.continual_multiagent_artifact import (
    canonical_content_bytes,
    wilson_score_interval,
)


def test_wilson_default_confidence() -> None:
    interval = wilson_score_interval(40, 100)
    assert interval["lower"] <= interval["upper"]
    assert 0.3 < interval["lower"] < 0.5


def test_wilson_confidence_level() -> None:
    wide = wilson_score_interval(50, 100, confidence_level=0.99)
    narrow = wilson_score_interval(50, 100, confidence_level=0.90)
    assert (wide["upper"] - wide["lower"]) > (narrow["upper"] - narrow["lower"])


def test_wilson_centered() -> None:
    interval = wilson_score_interval(50, 100)
    assert abs((interval["lower"] + interval["upper"]) / 2 - 0.5) < 1e-9


def test_wilson_matches_formula() -> None:
    s, n = 25, 80
    interval = wilson_score_interval(s, n)
    p_hat = s / n
    z = NormalDist().inv_cdf(0.5 + 0.95 / 2)
    z2 = z * z
    denom = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    half = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n) / denom
    assert abs(interval["lower"] - (center - half)) < 1e-9
    assert abs(interval["upper"] - (center + half)) < 1e-9


def test_wilson_invalid() -> None:
    with pytest.raises(ValueError):
        wilson_score_interval(5, 0)
    with pytest.raises(ValueError):
        wilson_score_interval(-1, 10)


def test_canonical_content_bytes_compact() -> None:
    payload = {"b": 2, "a": 1}
    out = canonical_content_bytes(payload)
    # Compact separators and sorted keys.
    assert out == b'{"a":1,"b":2}'


def test_canonical_content_bytes_unicode() -> None:
    payload = {"name": "测试"}
    out = canonical_content_bytes(payload)
    assert "测试".encode("utf-8") in out


def test_canonical_content_bytes_deterministic() -> None:
    payload = {"x": [3, 1], "y": {"k": "v"}}
    assert canonical_content_bytes(payload) == canonical_content_bytes(dict(payload))


def test_canonical_content_bytes_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_content_bytes({"v": float("nan")})
