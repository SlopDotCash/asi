"""Hostile validation for Step 3 Horde facade."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.steps.step3 import Step3HordeConfig


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


def test_rejects_string_subclass_for_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        Step3HordeConfig(hidden_sizes=(_StringSubclass("4"),))  # type: ignore[arg-type]


def test_hostile_str_for_hidden_sizes_without_repr_leak() -> None:
    evil = _EvilStr("4")
    with pytest.raises(ValueError, match="must be a positive integer") as exc:
        Step3HordeConfig(hidden_sizes=(evil,))  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int() -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        Step3HordeConfig(hidden_sizes=(True,))  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be a positive integer") as exc:
        Step3HordeConfig(hidden_sizes=(_HostileInt(4),))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        Step3HordeConfig(step_size=_HostileFloat(0.05))  # type: ignore[arg-type]
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


def test_rejects_plain_string_for_step_size() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step3HordeConfig(step_size="0.05")  # type: ignore[arg-type]


def test_rejects_string_subclass_for_step_size() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step3HordeConfig(step_size=_StringSubclass("0.05"))  # type: ignore[arg-type]


def test_rejects_out_of_range_gammas_without_repr() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]") as exc:
        Step3HordeConfig(gammas=(2.0,), lamdas=(0.0,))
    assert "2.0" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_subnormal_gammas_without_repr() -> None:
    tiny = 1e-40
    with pytest.raises(ValueError, match="must be zero or a normal float32") as exc:
        Step3HordeConfig(gammas=(tiny,), lamdas=(0.0,))
    assert "!r" not in str(exc.value)


def test_rejects_nonnegative_step_size_negative_without_repr() -> None:
    with pytest.raises(ValueError, match="must be non-negative") as exc:
        Step3HordeConfig(step_size=-1.0)
    assert "!r" not in str(exc.value)


def test_rejects_bool_gate_without_repr() -> None:
    from alberta_framework.steps.step3 import _require_bool

    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_bool(_EvilStr("use_obgd"), True)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    with pytest.raises(ValueError, match="must be a boolean") as exc2:
        _require_bool("use_obgd", _StringSubclass("true"))  # type: ignore[arg-type]
    assert "StringSubclass" not in str(exc2.value)
    assert "!r" not in str(exc2.value)


def test_valid_configs_still_pass() -> None:
    cfg = Step3HordeConfig(gammas=(0.9,), lamdas=(0.8,), hidden_sizes=(8,))
    assert cfg.gammas[0] == pytest.approx(0.9)
    assert cfg.hidden_sizes == (8,)


def test_numpy_scalars_pass() -> None:
    cfg = Step3HordeConfig(
        gammas=(cast(Any, np.float32(0.5)),),
        lamdas=(cast(Any, np.float64(0.5)),),
        step_size=cast(Any, Fraction(1, 20)),
        hidden_sizes=(cast(Any, np.int32(4)),),
    )
    assert cfg.hidden_sizes[0] == 4


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            return (3, 4)

    with pytest.raises(ValueError, match="must be finite"):
        Step3HordeConfig(step_size=RatioFloat(0.05))
    assert RatioFloat.calls == 0
