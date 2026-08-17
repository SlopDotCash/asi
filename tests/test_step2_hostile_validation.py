"""Hostile validation for Step 2 kernel facade."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.steps.step2 import Step2KernelConfig


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

    def __int__(self) -> int:
        type(self).calls += 1
        raise AssertionError("HostileInt.__int__ must not be called")

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


def test_rejects_string_subclass_for_feature_dim() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step2KernelConfig(feature_dim=_StringSubclass("8"))  # type: ignore[arg-type]


def test_hostile_str_for_feature_dim_without_repr_leak() -> None:
    evil = _EvilStr("8")
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step2KernelConfig(feature_dim=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step2KernelConfig(feature_dim=True)  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step2KernelConfig(feature_dim=_HostileInt(8))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        Step2KernelConfig(step_size=_HostileFloat(0.03))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_plain_string_for_step_size() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step2KernelConfig(step_size="0.03")  # type: ignore[arg-type]


def test_rejects_string_subclass_for_step_size() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step2KernelConfig(step_size=_StringSubclass("0.03"))  # type: ignore[arg-type]


def test_rejects_negative_step_size_without_repr() -> None:
    with pytest.raises(ValueError, match="must be non-negative") as exc:
        Step2KernelConfig(step_size=-0.01)
    assert "-0.01" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_hidden_sizes_non_tuple_without_repr() -> None:
    with pytest.raises(ValueError, match="must be a tuple of integers") as exc:
        Step2KernelConfig(hidden_sizes=[32])  # type: ignore[arg-type]
    assert "!r" not in str(exc.value)


def test_rejects_unknown_stream_without_repr() -> None:
    with pytest.raises(ValueError, match="unknown Step 2 stream") as exc:
        Step2KernelConfig(stream="evil")  # type: ignore[arg-type]
    assert "evil" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_valid_configs_still_pass() -> None:
    cfg = Step2KernelConfig(feature_dim=8, step_size=0.03)
    assert cfg.feature_dim == 8
    assert cfg.step_size == pytest.approx(0.03)
    cfg2 = Step2KernelConfig(
        feature_dim=32,
        n_heads=4,
        hidden_sizes=(32, 16),
        step_size=0.02,
    )
    assert cfg2.n_heads == 4


def test_numpy_scalars_pass() -> None:
    cfg = Step2KernelConfig(
        feature_dim=cast(Any, np.int32(8)),
        step_size=cast(Any, np.float32(0.03)),
        n_heads=cast(Any, np.int64(3)),
    )
    assert cfg.feature_dim == 8
    cfg2 = Step2KernelConfig(step_size=cast(Any, Fraction(3, 100)))
    assert cfg2.step_size == pytest.approx(0.03)


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            return (3, 100)

    with pytest.raises(ValueError, match="must be finite"):
        Step2KernelConfig(step_size=RatioFloat(0.03))
    assert RatioFloat.calls == 0
