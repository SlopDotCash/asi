"""Hostile validation for Step 11 OaK facade."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.core.options import SubtaskSpec
from alberta_framework.steps.step11 import Step11OaKConfig


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


class _HostileInt(int):
    calls = 0

    def __index__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileInt.__index__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileInt.__repr__ must not be called")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat.as_integer_ratio must not be called")

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat.__float__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileFloat.__repr__ must not be called")


def test_rejects_string_subclass_for_observation_dim() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step11OaKConfig(observation_dim=_StringSubclass("4"))  # type: ignore[arg-type]


def test_hostile_str_for_observation_dim_without_repr_leak() -> None:
    evil = _EvilStr("4")
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step11OaKConfig(observation_dim=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step11OaKConfig(observation_dim=True)  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step11OaKConfig(observation_dim=_HostileInt(4))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        Step11OaKConfig(base_step_size=_HostileFloat(0.05))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_does_not_invoke_hostile_value_when_name_is_evil_via_sink() -> None:
    from alberta_framework.steps._float32_validation import finite_real_and_float32

    evil = _EvilStr("x")
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be an exact string"):
        finite_real_and_float32(evil, _HostileFloat(1.0))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_rejects_plain_string_for_option_gamma() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step11OaKConfig(option_gamma="0.99")  # type: ignore[arg-type]


def test_rejects_string_subclass_for_option_gamma() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step11OaKConfig(option_gamma=_StringSubclass("0.99"))  # type: ignore[arg-type]


def test_rejects_out_of_range_option_gamma_without_repr() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]") as exc:
        Step11OaKConfig(option_gamma=2.0)
    assert "2.0" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_subtask_specs_non_tuple_without_repr() -> None:
    with pytest.raises(ValueError, match="must be a tuple of SubtaskSpec") as exc:
        Step11OaKConfig(subtask_specs=[SubtaskSpec(feature_index=0)])  # type: ignore[arg-type]
    assert "!r" not in str(exc.value)


def test_rejects_feature_index_out_of_range_without_repr() -> None:
    with pytest.raises(ValueError, match="must be < observation_dim") as exc:
        Step11OaKConfig(
            observation_dim=2,
            subtask_specs=(SubtaskSpec(feature_index=5, threshold=1.0),),
        )
    assert "!r" not in str(exc.value)


def test_valid_configs_still_pass() -> None:
    cfg = Step11OaKConfig(observation_dim=4, option_gamma=0.99)
    assert cfg.observation_dim == 4
    assert cfg.option_gamma == pytest.approx(0.99)
    cfg2 = Step11OaKConfig(
        subtask_specs=(SubtaskSpec(feature_index=0, threshold=0.5),),
        observation_dim=4,
    )
    assert cfg2.subtask_specs[0].feature_index == 0


def test_numpy_scalars_pass() -> None:
    cfg = Step11OaKConfig(
        observation_dim=cast(Any, np.int32(4)),
        base_step_size=cast(Any, np.float32(0.05)),
        option_gamma=cast(Any, np.float64(0.9)),
    )
    assert cfg.observation_dim == 4
    cfg2 = Step11OaKConfig(base_step_size=cast(Any, Fraction(1, 20)))
    assert cfg2.base_step_size == pytest.approx(0.05)


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            return (3, 4)

    with pytest.raises(ValueError, match="must be finite"):
        Step11OaKConfig(base_step_size=RatioFloat(0.05))
    assert RatioFloat.calls == 0
