"""Hostile validation for Step 7 Dyna planning facade."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
from alberta_framework.steps.step7 import Step7DynaConfig
from alberta_framework.steps.step8 import Step8WorldModelConfig


def _control_world() -> tuple[Step6DifferentialSARSAConfig, Step8WorldModelConfig]:
    control = Step6DifferentialSARSAConfig(n_actions=2)
    world = Step8WorldModelConfig(observation_dim=2, n_actions=2)
    return control, world


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


def _valid_config_kwargs() -> dict[str, Any]:
    control, world = _control_world()
    return {
        "control": control,
        "world_model": world,
        "planning_steps": 2,
        "planning_utility_step_size": 0.1,
    }


def test_rejects_string_subclass_for_planning_steps() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step7DynaConfig(**{**_valid_config_kwargs(), "planning_steps": _StringSubclass("2")})  # type: ignore[arg-type]


def test_hostile_str_for_planning_steps_without_repr_leak() -> None:
    evil = _EvilStr("2")
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step7DynaConfig(**{**_valid_config_kwargs(), "planning_steps": evil})  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step7DynaConfig(**{**_valid_config_kwargs(), "planning_steps": True})  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step7DynaConfig(**{**_valid_config_kwargs(), "planning_steps": _HostileInt(2)})  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be") as exc:
        Step7DynaConfig(**{**_valid_config_kwargs(), "planning_utility_step_size": _HostileFloat(0.1)})  # type: ignore[arg-type]  # noqa: E501
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


def test_rejects_plain_string_for_utility_step() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step7DynaConfig(**{**_valid_config_kwargs(), "planning_utility_step_size": "0.1"})  # type: ignore[arg-type]


def test_rejects_string_subclass_for_utility_step() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step7DynaConfig(  # noqa: E501
            **{**_valid_config_kwargs(), "planning_utility_step_size": _StringSubclass("0.1")}  # type: ignore[arg-type]
        )


def test_rejects_out_of_range_utility_step_without_repr() -> None:
    with pytest.raises(ValueError, match="must be in") as exc:
        Step7DynaConfig(**{**_valid_config_kwargs(), "planning_utility_step_size": 2.0})
    assert "2.0" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_planning_strategy_non_string_without_repr() -> None:
    with pytest.raises(ValueError, match="must be an actual string") as exc:
        Step7DynaConfig(**{**_valid_config_kwargs(), "planning_strategy": 123})  # type: ignore[arg-type]
    assert "!r" not in str(exc.value)


def test_rejects_importance_correction_non_bool_without_repr() -> None:
    with pytest.raises(ValueError, match="must be a built-in bool") as exc:
        Step7DynaConfig(  # noqa: E501
            **{**_valid_config_kwargs(), "planning_apply_importance_correction": "true"}  # type: ignore[arg-type]
        )
    assert "!r" not in str(exc.value)


def test_valid_configs_still_pass() -> None:
    cfg = Step7DynaConfig(**_valid_config_kwargs())
    assert cfg.planning_steps == 2
    cfg2 = Step7DynaConfig(**{**_valid_config_kwargs(), "planning_strategy": "reward"})
    assert cfg2.planning_strategy == "reward"


def test_numpy_scalars_pass() -> None:
    cfg = Step7DynaConfig(
        **{
            **_valid_config_kwargs(),
            "planning_steps": cast(Any, np.int32(2)),
            "planning_utility_step_size": cast(Any, np.float32(0.1)),
        }
    )
    assert cfg.planning_steps == 2
    cfg2 = Step7DynaConfig(
        **{**_valid_config_kwargs(), "planning_utility_step_size": cast(Any, Fraction(1, 10))}
    )
    assert cfg2.planning_utility_step_size == pytest.approx(0.1)


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            return (1, 10)

    with pytest.raises(ValueError, match="must be"):
        Step7DynaConfig(**{**_valid_config_kwargs(), "planning_utility_step_size": RatioFloat(0.1)})
    assert RatioFloat.calls == 0
