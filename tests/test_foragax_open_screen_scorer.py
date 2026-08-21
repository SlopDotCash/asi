"""Unit coverage for alberta_framework.benchmarks._foragax_open_screen_scorer.

Tests the frozen EMA scorer (shape/dtype/finiteness gates, sample-count
contract, tail window, deterministic trace hash) and the canonical
relative-path validation.
"""

import numpy as np
import pytest

from alberta_framework.benchmarks._foragax_open_screen_scorer import (
    _relative,
    score_rewards,
)


def test_score_rewards_basic() -> None:
    rewards = np.ones(100, dtype=np.float32)
    result = score_rewards(rewards, horizon=100)
    assert result["reward_shape"] == [100]
    assert result["reward_sum_float64"] == 100.0
    assert result["ema_sample_count"] > 0
    assert result["ema_tail_sample_count"] > 0
    assert len(result["reward_trace_sha256"]) == 64


def test_score_rewards_deterministic() -> None:
    rewards = np.random.default_rng(0).normal(size=100).astype(np.float64)
    r1 = score_rewards(rewards, horizon=100)
    r2 = score_rewards(rewards.copy(), horizon=100)
    assert r1["fov_last_10pct_ema_auc"] == r2["fov_last_10pct_ema_auc"]
    assert r1["reward_trace_sha256"] == r2["reward_trace_sha256"]


def test_score_rewards_constant_ema() -> None:
    # Constant rewards → EMA moves toward the constant over time.
    rewards = np.full(200, 5.0, dtype=np.float64)
    result = score_rewards(rewards, horizon=200)
    # EMA starts at 0 and decays toward 5; final EMA > 0 and below 5.
    assert 0.0 < result["final_unadjusted_ema"] < 5.0
    # Longer horizon converges closer to 5.
    long = score_rewards(np.full(10000, 5.0, dtype=np.float64), horizon=10000)
    assert long["final_unadjusted_ema"] > result["final_unadjusted_ema"]


def test_score_rewards_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="exact shape"):
        score_rewards(np.ones(99), horizon=100)


def test_score_rewards_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="numeric dtype"):
        score_rewards(np.zeros(100, dtype=object), horizon=100)


def test_score_rewards_rejects_nonfinite() -> None:
    rewards = np.ones(100)
    rewards[50] = np.nan
    with pytest.raises(ValueError, match="finite"):
        score_rewards(rewards, horizon=100)


def test_relative_accepts_canonical() -> None:
    assert _relative("results/run1").as_posix() == "results/run1"


def test_relative_rejects_bad() -> None:
    with pytest.raises(ValueError, match="canonical and relative"):
        _relative("/abs/path")
    with pytest.raises(ValueError, match="canonical and relative"):
        _relative("a/../b")
    with pytest.raises(ValueError, match="canonical and relative"):
        _relative("./a")
    with pytest.raises(ValueError, match="canonical and relative"):
        _relative("a//b")
