"""Hostile validation for Step 6 average-reward facade."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig


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


def test_rejects_string_subclass_for_n_actions() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step6DifferentialSARSAConfig(n_actions=_StringSubclass("4"))  # type: ignore[arg-type]


def test_hostile_str_for_n_actions_without_repr_leak() -> None:
    evil = _EvilStr("4")
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step6DifferentialSARSAConfig(n_actions=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step6DifferentialSARSAConfig(n_actions=True)  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step6DifferentialSARSAConfig(n_actions=_HostileInt(4))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_rejects_hostile_integer_metaclass_without_hash_hook() -> None:
    class HostileMeta(type):
        calls = 0

        def __hash__(cls) -> int:
            HostileMeta.calls += 1
            raise AssertionError("HostileMeta.__hash__ must not be called")

    class HostileMetaInt(int, metaclass=HostileMeta):
        pass

    with pytest.raises(ValueError, match="must be an integer"):
        Step6DifferentialSARSAConfig(n_actions=HostileMetaInt(2))  # type: ignore[arg-type]
    assert HostileMeta.calls == 0


def test_smoke_accepts_full_uint32_seed_contract() -> None:
    from alberta_framework.steps.step6 import run_step6_smoke

    result = run_step6_smoke(steps=1, feature_dim=1, seed=2**32 - 1)
    assert result.seed == 2**32 - 1


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        Step6DifferentialSARSAConfig(q_step_size=_HostileFloat(0.05))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_plain_string_for_q_step_size() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step6DifferentialSARSAConfig(q_step_size="0.05")  # type: ignore[arg-type]


def test_rejects_string_subclass_for_q_step_size() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step6DifferentialSARSAConfig(q_step_size=_StringSubclass("0.05"))  # type: ignore[arg-type]


def test_rejects_out_of_range_trace_decay_without_repr() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]") as exc:
        Step6DifferentialSARSAConfig(trace_decay=2.0)
    assert "2.0" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_negative_q_step_size_without_repr() -> None:
    with pytest.raises(ValueError, match="must be non-negative") as exc:
        Step6DifferentialSARSAConfig(q_step_size=-1.0)
    assert "!r" not in str(exc.value)


def test_rejects_bool_gate_without_repr() -> None:
    from alberta_framework.steps.step6 import _require_bool

    with pytest.raises(ValueError, match="must be a built-in bool") as exc2:
        _require_bool("finite", _StringSubclass("true"))  # type: ignore[arg-type]
    assert "StringSubclass" not in str(exc2.value)
    assert "!r" not in str(exc2.value)
    evil = _EvilStr("true")
    with pytest.raises(ValueError, match="must be a built-in bool") as exc3:
        _require_bool("finite", evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc3.value)


def test_valid_configs_still_pass() -> None:
    cfg = Step6DifferentialSARSAConfig(n_actions=2, q_step_size=0.05)
    assert cfg.n_actions == 2
    assert cfg.q_step_size == pytest.approx(0.05)


def test_numpy_scalars_pass() -> None:
    cfg = Step6DifferentialSARSAConfig(
        n_actions=cast(Any, np.int32(4)),
        q_step_size=cast(Any, np.float32(0.05)),
        trace_decay=cast(Any, np.float64(0.5)),
    )
    assert cfg.n_actions == 4
    cfg2 = Step6DifferentialSARSAConfig(q_step_size=cast(Any, Fraction(1, 20)))
    assert cfg2.q_step_size == pytest.approx(0.05)


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            return (3, 4)

    with pytest.raises(ValueError, match="must be finite"):
        Step6DifferentialSARSAConfig(q_step_size=RatioFloat(0.05))
    assert RatioFloat.calls == 0
