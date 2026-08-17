"""Hostile validation for Step 3 Horde facade."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.steps.step3 import Step3HordeConfig, run_step3_smoke


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


class _HostileTuple(tuple[object, ...]):
    calls = 0

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileTuple.__iter__ must not be called")

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("HostileTuple.__len__ must not be called")

    def __getitem__(self, key: object) -> object:
        type(self).calls += 1
        raise AssertionError("HostileTuple.__getitem__ must not be called")


class _HostileList(list[object]):
    calls = 0

    def __iter__(self):
        type(self).calls += 1
        raise AssertionError("HostileList.__iter__ must not be called")


class _HostileDict(dict[str, object]):
    calls = 0

    def __iter__(self):
        type(self).calls += 1
        raise AssertionError("HostileDict.__iter__ must not be called")

    def keys(self):
        type(self).calls += 1
        raise AssertionError("HostileDict.keys must not be called")


class _HostileMeta(type):
    calls = 0

    def __instancecheck__(cls, instance: object) -> bool:
        cls.calls += 1
        raise AssertionError("HostileMeta.__instancecheck__ must not be called")

    def __subclasscheck__(cls, subclass: type[object]) -> bool:
        cls.calls += 1
        raise AssertionError("HostileMeta.__subclasscheck__ must not be called")


class _MetaclassScalar(metaclass=_HostileMeta):
    @property
    def __class__(self) -> type[int]:
        _HostileMeta.calls += 1
        raise AssertionError("MetaclassScalar.__class__ must not be read")

    def __int__(self) -> int:
        _HostileMeta.calls += 1
        raise AssertionError("MetaclassScalar.__int__ must not be called")

    def __repr__(self) -> str:
        _HostileMeta.calls += 1
        raise AssertionError("MetaclassScalar.__repr__ must not be called")


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


def test_rejects_metaclass_scalar_without_spoofing_or_conversion_hooks() -> None:
    _HostileMeta.calls = 0
    with pytest.raises(ValueError, match="must be a positive integer"):
        Step3HordeConfig(hidden_sizes=(_MetaclassScalar(),))  # type: ignore[arg-type]
    assert _HostileMeta.calls == 0


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        Step3HordeConfig(step_size=_HostileFloat(0.05))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)


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
    with pytest.raises(ValueError, match="must be a boolean") as exc:
        Step3HordeConfig(use_obgd=_StringSubclass("true"))  # type: ignore[arg-type]
    assert "StringSubclass" not in str(exc.value)
    assert "!r" not in str(exc.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("normalizer", "ema"),
        ("trace_mode", "replacing"),
        ("routing", "mixed"),
    ],
)
def test_rejects_string_subclasses_for_choices(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be one of"):
        Step3HordeConfig(**{field: _StringSubclass(value)})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("normalizer", "median"),
        ("trace_mode", "dutch"),
        ("routing", "unknown"),
    ],
)
def test_rejects_unknown_exact_choices(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be one of"):
        Step3HordeConfig(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["gammas", "lamdas", "hidden_sizes"])
def test_rejects_non_exact_tuple_containers_without_hooks(field: str) -> None:
    value: object = _HostileTuple((0.5,))
    if field == "hidden_sizes":
        value = _HostileTuple((4,))
    _HostileTuple.calls = 0
    with pytest.raises(ValueError, match=f"{field} must be an actual tuple"):
        Step3HordeConfig(**{field: value})  # type: ignore[arg-type]
    assert _HostileTuple.calls == 0


def test_config_json_round_trip_and_exact_container_contract() -> None:
    config = Step3HordeConfig(
        gammas=(0.25, 0.75),
        lamdas=(0.0, 0.5),
        hidden_sizes=(4,),
        normalizer="ema",
        trace_mode="replacing",
        routing="mixed",
    )
    payload = config.to_dict()
    assert Step3HordeConfig.from_dict(payload) == config

    hostile_payload = _HostileDict(payload)
    _HostileDict.calls = 0
    with pytest.raises(ValueError, match="must be an actual dict"):
        Step3HordeConfig.from_dict(hostile_payload)
    assert _HostileDict.calls == 0

    for field in ("gammas", "lamdas", "hidden_sizes"):
        malformed = dict(payload)
        malformed[field] = _HostileList(cast(list[object], payload[field]))
        _HostileList.calls = 0
        with pytest.raises(ValueError, match=f"serialized {field} must be a JSON array"):
            Step3HordeConfig.from_dict(malformed)
        assert _HostileList.calls == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("steps", True),
        ("steps", 1.0),
        ("seed", -1),
        ("seed", 2**32),
        ("final_window", False),
        ("raw_feature_dim", 0),
        ("constructed_feature_dim", -1),
    ],
)
def test_smoke_rejects_noncanonical_or_out_of_range_integers(
    name: str, value: object
) -> None:
    kwargs = {name: value}
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match=name):
        run_step3_smoke(**kwargs)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_smoke_rejects_hostile_int_without_hooks() -> None:
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="steps must be an integer"):
        run_step3_smoke(steps=_HostileInt(2))
    assert _HostileInt.calls == 0


def test_smoke_accepts_exact_boundary_shaped_small_inputs() -> None:
    result = run_step3_smoke(
        steps=2,
        seed=2**32 - 1,
        final_window=1,
        raw_feature_dim=1,
        constructed_feature_dim=0,
    )
    assert result.steps == 2
    assert result.seed == 2**32 - 1
    assert result.finite


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
