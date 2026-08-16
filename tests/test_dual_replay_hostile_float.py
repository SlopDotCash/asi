"""Hostile float-hook regression coverage for dual replay."""

from __future__ import annotations

import pytest

from alberta_framework.core.dual_replay import DualReplayConfig


class _HostileFloat(float):
    def as_integer_ratio(self) -> tuple[int, int]:
        raise RuntimeError("untrusted ratio hook")


@pytest.mark.parametrize(
    "field",
    [
        "surprise_scale",
        "coverage_scale",
        "progress_scale",
        "surprise_weight",
        "coverage_weight",
        "progress_weight",
        "aleatoric_downweight_scale",
        "calibrated_priority_threshold",
        "calibrated_replacement_margin",
    ],
)
def test_dual_replay_wraps_hostile_float_hook_failures(field: str) -> None:
    kwargs: dict[str, object] = {
        "total_capacity": 6,
        "short_term_capacity": 2,
        "observation_dim": 2,
        "action_dim": 2,
        "short_term_sample_size": 1,
        "long_term_sample_size": 1,
        field: _HostileFloat(0.5),
    }
    with pytest.raises(ValueError, match=field):
        DualReplayConfig(**kwargs)  # type: ignore[arg-type]
