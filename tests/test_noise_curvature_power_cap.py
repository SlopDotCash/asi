"""Protocol ceilings for noise-curvature power iterations."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.noise_curvature_ipmnist import (
    _MAX_POWER_ITERATIONS,
    NoiseCurvatureConfig,
)


def test_documented_protocol_ceiling() -> None:
    assert _MAX_POWER_ITERATIONS == 10_000


@pytest.mark.parametrize("value", [10_001, 2**31 - 1])
def test_rejects_oversized_power_iterations(value: int) -> None:
    with pytest.raises(ValueError, match="power_iterations"):
        NoiseCurvatureConfig(
            mode="fixed",
            total_steps=2,
            control_interval=2,
            power_iterations=value,
        )
