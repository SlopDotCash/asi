"""Unit coverage for alberta_framework.steps.step8.

Tests the fail-closed validation gates: exact-string, non-negative and
half-open [0,1) unit-interval scalars, integer bounds, built-in bool,
and the Step8WorldModelConfig validation.
"""

import pytest

from alberta_framework.steps.step8 import (
    Step8WorldModelConfig,
    _require_bool,
    _require_exact_str,
    _require_half_open_unit_interval,
    _require_int,
    _require_nonnegative_real,
    _require_unit_interval,
)


def test_require_exact_str() -> None:
    assert _require_exact_str("x", "abc") == "abc"
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_str("x", 123)


def test_nonnegative_real() -> None:
    assert _require_nonnegative_real("x", 0.0) == 0.0
    assert _require_nonnegative_real("x", 2.5) == 2.5
    with pytest.raises(ValueError, match="non-negative"):
        _require_nonnegative_real("x", -1.0)


def test_unit_interval() -> None:
    assert _require_unit_interval("x", 0.5) == 0.5
    assert _require_unit_interval("x", 1.0) == 1.0
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        _require_unit_interval("x", 1.5)


def test_half_open_unit_interval() -> None:
    assert _require_half_open_unit_interval("x", 0.5) == 0.5
    assert _require_half_open_unit_interval("x", 0.0) == 0.0
    # 1.0 excluded in [0, 1)
    with pytest.raises(ValueError, match="\\[0, 1\\)"):
        _require_half_open_unit_interval("x", 1.0)
    with pytest.raises(ValueError, match="\\[0, 1\\)"):
        _require_half_open_unit_interval("x", 1.5)
    with pytest.raises(ValueError, match="\\[0, 1\\)"):
        _require_half_open_unit_interval("x", -0.1)


def test_require_int() -> None:
    assert _require_int("x", 5) == 5
    with pytest.raises(ValueError, match="positive"):
        _require_int("x", 0, minimum=1)
    with pytest.raises(ValueError, match="non-negative"):
        _require_int("x", -1, minimum=0)
    with pytest.raises(ValueError, match="integer"):
        _require_int("x", 1.5)


def test_require_bool() -> None:
    assert _require_bool("x", False) is False
    with pytest.raises(ValueError, match="built-in bool"):
        _require_bool("x", 1)


def test_world_model_config_valid() -> None:
    cfg = Step8WorldModelConfig(
        observation_dim=4,
        action_dim=2,
        hidden_sizes=(16,),
    )
    assert cfg.observation_dim == 4


def test_world_model_config_rejects_bad() -> None:
    with pytest.raises(ValueError, match="positive"):
        Step8WorldModelConfig(observation_dim=0)
    with pytest.raises(ValueError, match="positive"):
        Step8WorldModelConfig(action_dim=0)
