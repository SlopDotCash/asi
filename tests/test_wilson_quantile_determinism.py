from __future__ import annotations

import statistics

import pytest

from alberta_framework.evaluation.continual_multiagent_artifact import (
    wilson_score_interval as multiagent_wilson,
)
from alberta_framework.evaluation.recurring_feature_artifact import (
    wilson_score_interval as recurring_wilson,
)


def test_published_wilson_intervals_do_not_call_host_normal_quantile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("published 95% quantile must not use the host accelerator")

    monkeypatch.setattr(statistics.NormalDist, "inv_cdf", fail)

    recurring = recurring_wilson(1, 1)
    multiagent = multiagent_wilson(1, 1)
    expected_lower = float.fromhex("0x1.a70353b217770p-3")
    assert recurring["lower"] == expected_lower
    assert multiagent["lower"] == expected_lower
    assert recurring["upper"] == multiagent["upper"] == 1.0


def test_nonpublished_confidence_level_retains_general_quantile_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def fixed(_self: statistics.NormalDist, probability: float) -> float:
        calls.append(probability)
        return 1.0

    monkeypatch.setattr(statistics.NormalDist, "inv_cdf", fixed)
    interval = multiagent_wilson(1, 2, confidence_level=0.8)

    assert calls == [0.9]
    assert interval["confidence_level"] == 0.8
