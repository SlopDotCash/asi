"""Hostile validation for Step 8 world-model facade."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.steps.step8 import Step8WorldModelConfig


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


class _HostileTypeName(type):
    calls = 0

    def __getattribute__(cls, name: str) -> Any:
        if name == "__name__":
            _HostileTypeName.calls += 1
            raise AssertionError("metaclass __name__ hook must not be called")
        return super().__getattribute__(name)


class _HostileHiddenSizes(metaclass=_HostileTypeName):
    pass


def test_rejects_string_subclass_for_observation_dim() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step8WorldModelConfig(observation_dim=_StringSubclass("4"))  # type: ignore[arg-type]


def test_hostile_str_for_observation_dim_without_repr_leak() -> None:
    evil = _EvilStr("4")
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step8WorldModelConfig(observation_dim=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step8WorldModelConfig(observation_dim=True)  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step8WorldModelConfig(observation_dim=_HostileInt(4))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        Step8WorldModelConfig(step_size=_HostileFloat(0.05))  # type: ignore[arg-type]
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


def test_rejects_plain_string_for_sparsity() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step8WorldModelConfig(sparsity="0.9")  # type: ignore[arg-type]


def test_rejects_string_subclass_for_sparsity() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step8WorldModelConfig(sparsity=_StringSubclass("0.9"))  # type: ignore[arg-type]


def test_rejects_out_of_range_sparsity_without_repr() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]") as exc:
        Step8WorldModelConfig(sparsity=2.0)
    assert "2.0" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_use_layer_norm_non_bool_without_repr() -> None:
    with pytest.raises(ValueError, match="must be a built-in bool") as exc:
        Step8WorldModelConfig(use_layer_norm=1)  # type: ignore[arg-type]
    assert "!r" not in str(exc.value)
    with pytest.raises(ValueError, match="must be a built-in bool"):
        Step8WorldModelConfig(use_layer_norm="true")  # type: ignore[arg-type]


def test_rejects_hidden_sizes_without_type_name_hook() -> None:
    _HostileTypeName.calls = 0
    with pytest.raises(ValueError, match="hidden_sizes must be an actual tuple"):
        Step8WorldModelConfig(hidden_sizes=_HostileHiddenSizes())  # type: ignore[arg-type]
    assert _HostileTypeName.calls == 0


def test_valid_configs_still_pass() -> None:
    cfg = Step8WorldModelConfig(observation_dim=4, step_size=0.05, sparsity=0.9)
    assert cfg.observation_dim == 4
    assert cfg.sparsity == pytest.approx(0.9)
    cfg2 = Step8WorldModelConfig(observation_dim=2, n_actions=None, utility_decay=0.5)
    assert cfg2.n_actions is None


def test_numpy_scalars_pass() -> None:
    cfg = Step8WorldModelConfig(
        observation_dim=cast(Any, np.int32(4)),
        step_size=cast(Any, np.float32(0.05)),
        sparsity=cast(Any, np.float64(0.9)),
    )
    assert cfg.observation_dim == 4
    cfg2 = Step8WorldModelConfig(leaky_relu_slope=cast(Any, Fraction(1, 100)))
    assert cfg2.leaky_relu_slope == pytest.approx(0.01)


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            return (3, 4)

    with pytest.raises(ValueError, match="must be finite"):
        Step8WorldModelConfig(step_size=RatioFloat(0.05))
    assert RatioFloat.calls == 0


def test_from_dict_rejects_hostile_mapping_and_keys() -> None:
    class _HostileDict(dict[str, object]):
        pass

    with pytest.raises(ValueError, match="must be an exact dictionary"):
        Step8WorldModelConfig.from_dict(_HostileDict())

    payload = Step8WorldModelConfig().to_dict()
    bad_keys: dict[Any, Any] = dict(payload)
    bad_keys[_EvilStr("extra")] = 1
    with pytest.raises(ValueError, match="keys must be exact strings"):
        Step8WorldModelConfig.from_dict(cast(Any, bad_keys))


