"""Spread-estimator contract for the ceiling analyzer shipped under ``outputs/``.

The analyzer prints the ``sd`` values quoted in
``outputs/ipmnist_screening/CEILING_ANALYSIS.md``. Issue #33 unified
across-seed spread on the sample estimator, but that unification covered
``alberta_framework`` only, so this script kept NumPy's population default.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_ANALYZER = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "ipmnist_screening"
    / "ceiling"
    / "ceiling_analyze.py"
)


def _load_analyzer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ceiling_analyze", _ANALYZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_across_seed_spread_is_the_sample_estimator() -> None:
    """Across-seed spread must use ddof=1, matching the screening merge."""
    module = _load_analyzer()
    values = [0.7814, 0.7842, 0.7870]

    expected = float(np.asarray(values, dtype=np.float64).std(ddof=1))
    assert module.across_seed_spread(values) == pytest.approx(expected)

    population = float(np.asarray(values, dtype=np.float64).std())
    assert module.across_seed_spread(values) > population


def test_across_seed_spread_matches_the_screening_merge_convention() -> None:
    """The helper agrees with the estimator the campaign merge already uses."""
    module = _load_analyzer()
    rng = np.random.default_rng(0)
    for n in (2, 3, 5, 20):
        sample = rng.normal(size=n)
        merge_convention = float(sample.std(ddof=1))
        assert module.across_seed_spread(sample) == pytest.approx(merge_convention)


@pytest.mark.parametrize("values", ([], [0.5]))
def test_across_seed_spread_reports_no_spread_below_two_seeds(values: list[float]) -> None:
    """One seed has no spread; ddof=1 would be undefined rather than zero."""
    module = _load_analyzer()
    assert module.across_seed_spread(values) == 0.0
