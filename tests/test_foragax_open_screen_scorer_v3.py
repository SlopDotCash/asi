"""Host-boundary identities for the pinned Foragax open-screen v3 scorer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alberta_framework.benchmarks._foragax_open_screen_scorer_v3 import (
    score_archives,
    score_rewards,
)


@pytest.mark.parametrize("horizon", [True, False, 0, -1, 1.0])
def test_score_rewards_rejects_noncanonical_horizon_before_aggregates(
    horizon: object,
) -> None:
    with pytest.raises(ValueError, match="horizon"):
        score_rewards(np.zeros(1, dtype=np.float64), horizon)  # type: ignore[arg-type]


def test_score_rewards_keeps_builtin_horizon_identity() -> None:
    scored = score_rewards(np.zeros(1, dtype=np.float64), 1)
    assert scored["reward_shape"] == [1]
    assert type(scored["reward_shape"][0]) is int


@pytest.mark.parametrize("horizon", [True, False, 0, -1, 1.0])
def test_score_archives_rejects_noncanonical_horizon_before_payload_io(
    tmp_path: Path,
    horizon: object,
) -> None:
    with pytest.raises(ValueError, match="horizon"):
        score_archives(tmp_path, "results", [0], horizon)  # type: ignore[arg-type]


@pytest.mark.parametrize("seeds", [True, False, 0, [True], [False], [1.0], [-1]])
def test_score_archives_rejects_noncanonical_seeds_before_payload_io(
    tmp_path: Path,
    seeds: object,
) -> None:
    with pytest.raises(ValueError, match="seeds"):
        score_archives(tmp_path, "results", seeds, 1)  # type: ignore[arg-type]
