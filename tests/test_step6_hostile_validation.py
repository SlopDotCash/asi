"""Hostile validation for Step 6 average-reward facade."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.steps.step6 import (
    Step6DifferentialSARSAConfig,
    make_step6_differential_sarsa_agent,
    run_step6_smoke,
)


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


def test_rejects_rising_epsilon_schedule() -> None:
    with pytest.raises(ValueError, match="epsilon_end must not exceed epsilon_start"):
        Step6DifferentialSARSAConfig(epsilon_start=0.1, epsilon_end=0.5, epsilon_decay_steps=4)


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


def test_from_dict_requires_exact_complete_schema_without_mapping_hooks() -> None:
    class HostileDict(dict[object, object]):
        calls = 0

        def __iter__(self):  # pragma: no cover - must not run
            type(self).calls += 1
            raise AssertionError("mapping hook must not run")

    with pytest.raises(ValueError, match="exact dictionary"):
        Step6DifferentialSARSAConfig.from_dict(cast(Any, HostileDict()))
    assert HostileDict.calls == 0
    payload = Step6DifferentialSARSAConfig().to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="schema"):
        Step6DifferentialSARSAConfig.from_dict(payload)


def test_runtime_entry_points_require_exact_config_without_truthiness_hooks() -> None:
    calls = 0

    class HostileConfig:
        def __bool__(self) -> bool:  # pragma: no cover - must not run
            nonlocal calls
            calls += 1
            raise AssertionError("truthiness hook must not run")

    value = HostileConfig()
    with pytest.raises(TypeError, match="exact Step6DifferentialSARSAConfig"):
        make_step6_differential_sarsa_agent(cast(Any, value))
    with pytest.raises(TypeError, match="exact Step6DifferentialSARSAConfig"):
        run_step6_smoke(cast(Any, value), steps=1)
    assert calls == 0


def test_smoke_preflights_state_and_output_resources_before_allocation() -> None:
    huge_actions = Step6DifferentialSARSAConfig(n_actions=536_870_912)
    with pytest.raises(ValueError, match="state parameter bytes"):
        run_step6_smoke(huge_actions, steps=1, feature_dim=1)
    with pytest.raises(ValueError, match="observation row count"):
        run_step6_smoke(steps=2**31 - 1, feature_dim=1)
