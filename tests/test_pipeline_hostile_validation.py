"""Hostile-safe validation for pipeline trusts."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.pipeline import (
    Step2FeatureConfig,
    Step2UPGDConfig,
    _require_bool,
    _require_int,
    _require_str_choice,
)


class _StringSubclass(str):
    pass


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise RuntimeError("str hook")

    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook")


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr hook")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:
        type(self).calls += 1
        raise RuntimeError("ratio hook")


def test_require_bool_rejects_string_subclass_name() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_bool(_StringSubclass("use_layer_norm"), True)
    with pytest.raises(ValueError, match="exact string"):
        _require_int(_StringSubclass("observation_dim"), 4)


def test_require_bool_does_not_invoke_hostile_name_repr() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_bool(_EvilStr("include_raw"), True)
    with pytest.raises(ValueError, match="exact string"):
        _require_str_choice(_EvilStr("step2"), "upgd", ("upgd", "identity"))


def test_require_bool_rejects_non_bool_without_repr() -> None:
    with pytest.raises(ValueError, match="must be a bool"):
        _require_bool("include_raw", 1)
    with pytest.raises(ValueError, match="must be a bool"):
        _require_bool("include_raw", _StringSubclass("true"))


def test_require_int_rejects_bool_and_hostile() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _require_int("observation_dim", True)
    with pytest.raises(ValueError, match="must be an integer"):
        _require_int("observation_dim", _HostileInt(4))


def test_require_int_does_not_invoke_hostile_repr() -> None:
    evil = _EvilStr("evil")
    # value is EvilStr but _require_int will reject before formatting it with !r
    with pytest.raises(ValueError, match="must be an integer"):
        _require_int("observation_dim", evil)


def test_require_str_choice_rejects_string_subclass() -> None:
    with pytest.raises(ValueError, match="unknown"):
        _require_str_choice("step2", _StringSubclass("upgd"), ("upgd", "identity"))
    with pytest.raises(ValueError, match="exact string"):
        _require_str_choice(_StringSubclass("step2"), "upgd", ("upgd",))


def test_step2feature_rejects_string_subclass_periods() -> None:
    with pytest.raises(ValueError, match="periods must be a tuple"):
        Step2FeatureConfig(periods=cast(Any, _StringSubclass("bad")))


def test_step2feature_does_not_invoke_hostile_periods_repr() -> None:
    # periods with hostile repr should not invoke it (no !r)
    with pytest.raises(ValueError, match="periods must be a tuple"):
        Step2FeatureConfig(periods=cast(Any, _EvilStr("bad")))


def test_step2upgd_rejects_string_subclass_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes must contain"):
        Step2UPGDConfig(hidden_sizes=cast(Any, _StringSubclass("bad")))
    with pytest.raises(ValueError, match="hidden_sizes must contain"):
        Step2UPGDConfig(hidden_sizes=cast(Any, _EvilStr("bad")))


def test_valid_configs_still_pass() -> None:
    cfg = Step2FeatureConfig(observation_dim=4, periods=(32.0, 64.0))
    assert cfg.observation_dim == 4
    cfg2 = Step2UPGDConfig(observation_dim=4, hidden_sizes=(8,), step_size=0.03)
    assert cfg2.observation_dim == 4
    assert _require_bool("flag", True) is True
    assert _require_int("n", cast(Any, np.int32(7))) == 7
    assert _require_str_choice("m", "a", ("a", "b")) == "a"


def test_numpy_int_and_domain_still_pass() -> None:
    # valid numpy int types still canonicalize
    assert _require_int("x", cast(Any, np.int64(5))) == 5
    cfg = Step2FeatureConfig(observation_dim=cast(Any, np.int32(3)), periods=())
    assert cfg.observation_dim == 3
