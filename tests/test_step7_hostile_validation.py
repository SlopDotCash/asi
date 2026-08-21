"""Hostile validation for Step 7 Dyna planning facade."""

from fractions import Fraction
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.steps.step6 import Step6DifferentialSARSAConfig
from alberta_framework.steps.step7 import (
    Step7DynaConfig,
    init_step7_state,
    make_step7_components,
    run_step7_scan,
    run_step7_smoke,
)
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

    def __int__(self) -> int:
        type(self).calls += 1
        raise AssertionError("HostileInt.__int__ must not be called")

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


class _HostileIntMeta(type):
    calls = 0

    def __hash__(cls) -> int:
        type(cls).calls += 1
        raise AssertionError("HostileIntMeta.__hash__ must not be called")


class _MetaclassHostileInt(int, metaclass=_HostileIntMeta):
    pass


class _HostileFloatMeta(type):
    calls = 0

    def __hash__(cls) -> int:
        type(cls).calls += 1
        raise AssertionError("HostileFloatMeta.__hash__ must not be called")


class _MetaclassHostileFloat(float, metaclass=_HostileFloatMeta):
    pass


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


def test_rejects_hostile_int_metaclass_without_hooks() -> None:
    _HostileIntMeta.calls = 0
    with pytest.raises(ValueError, match="must be an integer"):
        Step7DynaConfig(**{**_valid_config_kwargs(), "planning_steps": _MetaclassHostileInt(2)})
    assert _HostileIntMeta.calls == 0


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be") as exc:
        Step7DynaConfig(**{**_valid_config_kwargs(), "planning_utility_step_size": _HostileFloat(0.1)})  # type: ignore[arg-type]  # noqa: E501
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_hostile_float_metaclass_without_hooks() -> None:
    _HostileFloatMeta.calls = 0
    with pytest.raises(ValueError, match="must be finite"):
        Step7DynaConfig(
            **{
                **_valid_config_kwargs(),
                "planning_utility_step_size": _MetaclassHostileFloat(0.1),
            }
        )
    assert _HostileFloatMeta.calls == 0


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


def test_config_requires_exact_nested_records_and_closed_dict_schema() -> None:
    class ControlSubclass(Step6DifferentialSARSAConfig):
        pass

    hostile_control = object.__new__(ControlSubclass)
    with pytest.raises(ValueError, match="actual Step6"):
        Step7DynaConfig(control=hostile_control)
    payload = Step7DynaConfig().to_dict()
    with pytest.raises(ValueError, match="exact Step7"):
        Step7DynaConfig.from_dict({**payload, "extra": 1})

    class DictSubclass(dict[str, object]):
        pass

    with pytest.raises(ValueError, match="actual dict"):
        Step7DynaConfig.from_dict(DictSubclass(payload))


class _HostileArray:
    calls = 0

    @property
    def shape(self) -> tuple[int, ...]:
        type(self).calls += 1
        raise AssertionError("shape hook must not run")


def test_init_and_scan_reject_hostile_arrays_without_hooks() -> None:
    cfg = Step7DynaConfig(
        control=Step6DifferentialSARSAConfig(n_actions=2),
        world_model=Step8WorldModelConfig(observation_dim=2, n_actions=2),
        planning_memory_size=2,
    )
    agent, model = make_step7_components(cfg)
    _HostileArray.calls = 0
    with pytest.raises(TypeError, match="trusted array"):
        init_step7_state(
            agent,
            model,
            key=jr.key(0),
            initial_observation=cast(Any, _HostileArray()),
            memory_size=2,
        )
    assert _HostileArray.calls == 0
    state = init_step7_state(
        agent,
        model,
        key=jr.key(0),
        initial_observation=jnp.zeros(2, dtype=jnp.float32),
        memory_size=2,
    )
    with pytest.raises(TypeError, match="rewards.*float32"):
        run_step7_scan(
            cfg,
            agent,
            model,
            state,
            jnp.zeros(1, dtype=jnp.int32),
            jnp.zeros((1, 2), dtype=jnp.float32),
        )


def test_smoke_preflights_all_live_arrays_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_allocation(*args: object, **kwargs: object) -> None:
        raise AssertionError("allocation must not start before resource preflight")

    monkeypatch.setattr(jr, "normal", unexpected_allocation)
    with pytest.raises(ValueError, match="smoke resources"):
        run_step7_smoke(steps=50_000_000)


class TestStep7ScanBudget:
    """Oversized planning loop lengths must fail at config validation (#2214)."""

    def _cfg(self, **overrides: Any) -> Step7DynaConfig:
        control, world = _control_world()
        return Step7DynaConfig(
            control=control,
            world_model=world,
            planning_steps=1,
            planning_rollout_depth=1,
            **overrides,
        )

    def test_planning_steps_rejected_above_budget(self) -> None:
        with pytest.raises(
            ValueError, match="planning_steps must be an integer in \\[1, 10000\\]"
        ):
            self._cfg(planning_steps=10**9)

    def test_planning_rollout_depth_rejected_above_budget(self) -> None:
        with pytest.raises(
            ValueError,
            match="planning_rollout_depth must be an integer in \\[1, 10000\\]",
        ):
            self._cfg(planning_rollout_depth=10**9)

    def test_boundary_accepted(self) -> None:
        cfg = self._cfg(planning_steps=10_000, planning_rollout_depth=10_000)
        assert cfg.planning_steps == 10_000
        assert cfg.planning_rollout_depth == 10_000

    def test_zero_planning_steps_still_accepted(self) -> None:
        # planning_steps=0 is the documented "disabled" value; it must not trip
        # the >=1 scan-step guard.
        cfg = self._cfg(planning_steps=0)
        assert cfg.planning_steps == 0
