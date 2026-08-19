"""Hostile validation for Step 4 SARSA facade."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.steps.step4 import Step4SARSAConfig


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
    with pytest.raises(ValueError, match="must be a positive integer"):
        Step4SARSAConfig(n_actions=_StringSubclass("2"))  # type: ignore[arg-type]


def test_hostile_str_for_n_actions_without_repr_leak() -> None:
    evil = _EvilStr("2")
    with pytest.raises(ValueError, match="must be a positive integer") as exc:
        Step4SARSAConfig(n_actions=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int() -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        Step4SARSAConfig(n_actions=True)  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be a positive integer") as exc:
        Step4SARSAConfig(n_actions=_HostileInt(2))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_rejects_hostile_config_containers_and_choices_without_hooks() -> None:
    with pytest.raises(ValueError, match="hidden_sizes must be an actual tuple"):
        Step4SARSAConfig(hidden_sizes=[8])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="optimizer must be one of"):
        Step4SARSAConfig(optimizer=_EvilStr("lms"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bounder must be one of"):
        Step4SARSAConfig(bounder=_EvilStr("obgd"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="trace_mode must be one of"):
        Step4SARSAConfig(trace_mode=_EvilStr("accumulating"))  # type: ignore[arg-type]


def test_smoke_rejects_hostile_dimensions_before_numeric_hooks() -> None:
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="steps must be a positive integer"):
        from alberta_framework.steps.step4 import run_step4_smoke

        run_step4_smoke(steps=_HostileInt(1))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        Step4SARSAConfig(gamma=_HostileFloat(0.99))  # type: ignore[arg-type]
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


def test_rejects_plain_string_for_gamma() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step4SARSAConfig(gamma="0.99")  # type: ignore[arg-type]


def test_rejects_string_subclass_for_gamma() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step4SARSAConfig(gamma=_StringSubclass("0.99"))  # type: ignore[arg-type]


def test_rejects_out_of_range_gamma_without_repr() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]") as exc:
        Step4SARSAConfig(gamma=2.0)
    assert "2.0" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_subnormal_gamma_without_repr() -> None:
    tiny = 1e-40
    with pytest.raises(ValueError, match="must be zero or a normal float32") as exc:
        Step4SARSAConfig(gamma=tiny)
    assert "!r" not in str(exc.value)


def test_rejects_use_layer_norm_non_bool_without_repr() -> None:
    with pytest.raises(ValueError, match="must be a boolean") as exc:
        Step4SARSAConfig(use_layer_norm=1)  # type: ignore[arg-type]
    assert "!r" not in str(exc.value)
    with pytest.raises(ValueError, match="must be a boolean"):
        Step4SARSAConfig(use_layer_norm="true")  # type: ignore[arg-type]


def test_valid_configs_still_pass() -> None:
    cfg = Step4SARSAConfig(n_actions=2, gamma=0.99, sparsity=0.5)
    assert cfg.n_actions == 2
    assert cfg.gamma == pytest.approx(0.99)
    cfg2 = Step4SARSAConfig(n_actions=4, epsilon_decay_steps=0, use_layer_norm=False)
    assert cfg2.n_actions == 4


def test_numpy_scalars_pass() -> None:
    cfg = Step4SARSAConfig(
        n_actions=cast(Any, np.int32(2)),
        gamma=cast(Any, np.float32(0.5)),
        step_size=cast(Any, np.float64(0.03)),
    )
    assert cfg.n_actions == 2
    cfg2 = Step4SARSAConfig(bounder_kappa=cast(Any, Fraction(1, 2)))
    assert cfg2.bounder_kappa == pytest.approx(0.5)


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            return (3, 4)

    with pytest.raises(ValueError, match="must be finite"):
        Step4SARSAConfig(gamma=RatioFloat(0.99))
    assert RatioFloat.calls == 0


def test_from_dict_rejects_hostile_mapping_and_keys() -> None:
    class _HostileDict(dict[str, object]):
        pass

    with pytest.raises(ValueError, match="must be an exact dictionary"):
        Step4SARSAConfig.from_dict(_HostileDict())

    payload = Step4SARSAConfig().to_dict()
    bad_keys: dict[Any, Any] = dict(payload)
    bad_keys[_EvilStr("extra")] = 1
    with pytest.raises(ValueError, match="keys must be exact strings"):
        Step4SARSAConfig.from_dict(cast(Any, bad_keys))


