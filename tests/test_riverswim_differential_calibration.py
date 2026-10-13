"""Direct RiverSwim replay of the frozen differential-SARSA miss.

These checks are development measurements on fresh JAX seeds. They do not
consume scorecard seeds 70000-70011 and they do not change the frozen plan.
"""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.reference_life_scorecard import (
    build_development_plan,
)
from alberta_framework.benchmarks.riverswim_differential_calibration import (
    CALIBRATED_DIFFERENTIAL_SARSA,
    FROZEN_DIFFERENTIAL_SARSA,
    riverswim_differential_return,
)


def test_calibration_stays_outside_the_frozen_scorecard_plan() -> None:
    """The frozen plan still asks for the control that missed the high end."""

    frozen = build_development_plan().arm_definition("differential_sarsa")["config"]
    assert frozen["use_bias"] is True
    assert frozen["epsilon_start"] == 0.5
    assert frozen["epsilon_end"] == 0.02
    assert frozen["epsilon_decay_steps"] == 2_500
    assert FROZEN_DIFFERENTIAL_SARSA["use_bias"] is True
    assert FROZEN_DIFFERENTIAL_SARSA["epsilon_decay_steps"] == 2_500
    assert CALIBRATED_DIFFERENTIAL_SARSA["use_bias"] is False
    assert CALIBRATED_DIFFERENTIAL_SARSA["epsilon_start"] == 1.0
    assert CALIBRATED_DIFFERENTIAL_SARSA["epsilon_end"] == 0.1
    assert CALIBRATED_DIFFERENTIAL_SARSA["epsilon_decay_steps"] == 20_000


@pytest.mark.integration
def test_frozen_differential_sarsa_misses_the_riverswim_high_end() -> None:
    """Seed 0 of the frozen hyperparameters never reaches the upstream reward."""

    visits, reward = riverswim_differential_return(
        0,
        20_000,
        FROZEN_DIFFERENTIAL_SARSA,
    )
    assert visits == 0
    assert reward < 150.0


@pytest.mark.integration
def test_calibrated_differential_sarsa_reaches_the_riverswim_high_end() -> None:
    """The same seed reaches the high end and collects the upstream reward."""

    visits, reward = riverswim_differential_return(
        0,
        20_000,
        CALIBRATED_DIFFERENTIAL_SARSA,
    )
    assert visits >= 3_000
    assert reward >= 2_500.0
