"""Unit coverage for alberta_framework.steps.step4.

Tests the fail-closed scalar gates (unit interval, GVF probability with
normal-float32 requirement, non-negative/positive reals, ints, bool,
choice) and the Step4SARSAConfig round-trip.
"""

import pytest

from alberta_framework.steps.step4 import (
    Step4SARSAConfig,
    _require_bool,
    _require_choice,
    _require_gvf_probability,
    _require_nonneg_int,
    _require_nonnegative_real,
    _require_positive_int,
    _require_positive_real,
    _require_unit_interval,
)


def test_unit_interval() -> None:
    assert _require_unit_interval("x", 0.5) == 0.5
    assert _require_unit_interval("x", 0.0) == 0.0
    assert _require_unit_interval("x", 1.0) == 1.0
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        _require_unit_interval("x", 1.5)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        _require_unit_interval("x", -0.1)


def test_gvf_probability() -> None:
    assert _require_gvf_probability("x", 0.5) == 0.5
    assert _require_gvf_probability("x", 0.0) == 0.0
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        _require_gvf_probability("x", 1.5)
    # Subnormal (below float32 min normal) must be rejected.
    with pytest.raises(ValueError, match="normal float32"):
        _require_gvf_probability("x", 1e-40)


def test_nonnegative_real() -> None:
    assert _require_nonnegative_real("x", 0.0) == 0.0
    assert _require_nonnegative_real("x", 2.5) == 2.5
    with pytest.raises(ValueError, match="non-negative"):
        _require_nonnegative_real("x", -1.0)


def test_positive_real() -> None:
    assert _require_positive_real("x", 0.1) == 0.1
    with pytest.raises(ValueError, match="positive"):
        _require_positive_real("x", 0.0)
    with pytest.raises(ValueError, match="positive"):
        _require_positive_real("x", -1.0)


def test_positive_int() -> None:
    assert _require_positive_int("x", 5) == 5
    with pytest.raises(ValueError, match="positive"):
        _require_positive_int("x", 0)
    with pytest.raises(ValueError, match="positive integer"):
        _require_positive_int("x", 1.5)


def test_nonneg_int() -> None:
    assert _require_nonneg_int("x", 0) == 0
    assert _require_nonneg_int("x", 5) == 5
    with pytest.raises(ValueError, match="non-negative"):
        _require_nonneg_int("x", -1)


def test_bool() -> None:
    assert _require_bool("x", True) is True
    with pytest.raises(ValueError, match="boolean"):
        _require_bool("x", 1)


def test_choice() -> None:
    assert _require_choice("x", "lms", ("lms", "idbd")) == "lms"
    with pytest.raises(ValueError, match="one of"):
        _require_choice("x", "bogus", ("lms", "idbd"))
    with pytest.raises(ValueError, match="one of"):
        _require_choice("x", 1, ("lms", "idbd"))


def test_config_round_trip() -> None:
    cfg = Step4SARSAConfig(n_actions=3, hidden_sizes=(8, 4), optimizer="idbd")
    payload = cfg.to_dict()
    assert payload["n_actions"] == 3
    assert payload["hidden_sizes"] == [8, 4]
    restored = Step4SARSAConfig.from_dict(payload)
    assert restored == cfg


def test_config_rejects_bad() -> None:
    with pytest.raises(ValueError, match="positive"):
        Step4SARSAConfig(n_actions=0)
    with pytest.raises(ValueError, match="one of"):
        Step4SARSAConfig(optimizer="bogus")
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        Step4SARSAConfig(gamma=1.5)
