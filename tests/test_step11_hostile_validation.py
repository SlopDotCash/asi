"""Hostile validation for Step 11 OaK facade."""

from fractions import Fraction
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.options import SubtaskSpec
from alberta_framework.steps.step11 import (
    Step11OaKConfig,
    init_step11_state,
    make_step11_oak_agent,
    run_step11_scan,
    step11_update,
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


def test_rejects_hostile_int_metaclass_without_hooks() -> None:
    _HostileIntMeta.calls = 0
    with pytest.raises(ValueError, match="must be an integer"):
        Step11OaKConfig(observation_dim=_MetaclassHostileInt(4))
    assert _HostileIntMeta.calls == 0


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        Step11OaKConfig(base_step_size=_HostileFloat(0.05))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_hostile_float_metaclass_without_hooks() -> None:
    _HostileFloatMeta.calls = 0
    with pytest.raises(ValueError, match="must be finite"):
        Step11OaKConfig(base_step_size=_MetaclassHostileFloat(0.05))
    assert _HostileFloatMeta.calls == 0


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


def test_rejects_tuple_and_subtask_spec_subclasses_without_hooks() -> None:
    calls: list[str] = []

    class HostileTuple(tuple[SubtaskSpec, ...]):
        def __iter__(self):  # type: ignore[override]
            calls.append("iter")
            raise AssertionError("HostileTuple.__iter__ must not be called")

        def __repr__(self) -> str:
            calls.append("repr")
            raise AssertionError("HostileTuple.__repr__ must not be called")

    class HostileSpec(SubtaskSpec):
        def __repr__(self) -> str:
            calls.append("spec-repr")
            raise AssertionError("HostileSpec.__repr__ must not be called")

    with pytest.raises(ValueError, match="must be a tuple of SubtaskSpec"):
        Step11OaKConfig(subtask_specs=HostileTuple((SubtaskSpec(feature_index=0),)))
    with pytest.raises(ValueError, match="must contain SubtaskSpec values"):
        Step11OaKConfig(subtask_specs=(HostileSpec(feature_index=0),))
    assert calls == []


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


def test_config_subclass_is_rejected_before_attribute_hooks() -> None:
    class HostileConfig(Step11OaKConfig):
        calls = 0

        def __getattribute__(self, name: str) -> object:
            if name not in {"calls", "__class__"}:
                type(self).calls += 1
                raise AssertionError("attribute hook must not run")
            return super().__getattribute__(name)

    hostile = object.__new__(HostileConfig)
    with pytest.raises(ValueError, match="actual Step11OaKConfig"):
        Step11OaKConfig.__post_init__(hostile)
    assert HostileConfig.calls == 0


def test_from_config_requires_complete_exact_schema_without_mapping_hooks() -> None:
    class HostileDict(dict[object, object]):
        calls = 0

        def __iter__(self):  # pragma: no cover - must not run
            type(self).calls += 1
            raise AssertionError("mapping hook must not run")

    with pytest.raises(ValueError, match="exact dictionary"):
        Step11OaKConfig.from_config(cast(Any, HostileDict()))
    assert HostileDict.calls == 0

    payload = Step11OaKConfig().to_config()
    for mutation in (
        lambda value: value.pop("type"),
        lambda value: value.__setitem__("extra", 1),
        lambda value: value.__setitem__("type", "wrong"),
    ):
        malformed = dict(payload)
        mutation(malformed)
        with pytest.raises(ValueError, match="(schema|payload type)"):
            Step11OaKConfig.from_config(malformed)


def test_from_config_requires_exact_nested_records_before_hooks() -> None:
    class HostileRecord(dict[object, object]):
        calls = 0

        def __iter__(self):  # pragma: no cover - must not run
            type(self).calls += 1
            raise AssertionError("record hook must not run")

    payload = Step11OaKConfig().to_config()
    payload["subtask_specs"] = [HostileRecord()]
    with pytest.raises(ValueError, match="exact dictionary"):
        Step11OaKConfig.from_config(payload)
    assert HostileRecord.calls == 0

    payload = Step11OaKConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),)
    ).to_config()
    payload["subtask_specs"] = [{**payload["subtask_specs"][0], "extra": 1}]
    with pytest.raises(ValueError, match="fields do not match"):
        Step11OaKConfig.from_config(payload)


def test_from_config_and_direct_paths_have_matching_canonical_values() -> None:
    direct = Step11OaKConfig(
        observation_dim=np.int32(4),  # type: ignore[arg-type]
        base_step_size=np.float64(0.05),  # type: ignore[arg-type]
        subtask_specs=(SubtaskSpec(feature_index=0, threshold=Fraction(1, 2)),),
    )
    parsed = Step11OaKConfig.from_config(direct.to_config())
    assert parsed == direct
    assert parsed.to_config() == direct.to_config()


def test_subtask_validation_work_is_bounded_before_iteration() -> None:
    spec = SubtaskSpec(feature_index=0)
    with pytest.raises(ValueError, match="at most 4096"):
        Step11OaKConfig(subtask_specs=(spec,) * 4_097)

    payload = Step11OaKConfig().to_config()
    raw_spec = {
        "feature_index": 0,
        "threshold": 0.5,
        "pseudo_reward_scale": 1.0,
        "max_option_steps": 8,
    }
    payload["subtask_specs"] = [raw_spec] * 4_097
    with pytest.raises(ValueError, match="at most 4096"):
        Step11OaKConfig.from_config(payload)


class _HostileArray:
    calls = 0

    @property
    def shape(self) -> tuple[int, ...]:
        type(self).calls += 1
        raise AssertionError("shape hook must not run")

    @property
    def dtype(self) -> np.dtype[Any]:
        type(self).calls += 1
        raise AssertionError("dtype hook must not run")


def test_runtime_rejects_hostile_arrays_without_hooks() -> None:
    agent = make_step11_oak_agent()
    _HostileArray.calls = 0
    with pytest.raises(TypeError, match="trusted array"):
        init_step11_state(
            agent,
            key=jr.key(0),
            initial_observation=cast(Any, _HostileArray()),
        )
    assert _HostileArray.calls == 0


def test_runtime_requires_typed_key_and_exact_float32_arrays() -> None:
    agent = make_step11_oak_agent()
    with pytest.raises(TypeError, match="typed JAX PRNG key"):
        init_step11_state(
            agent,
            key=cast(Any, jr.PRNGKey(0)),
            initial_observation=jnp.zeros(4, dtype=jnp.float32),
        )
    state = init_step11_state(
        agent,
        key=jr.key(0),
        initial_observation=jnp.zeros(4, dtype=jnp.float32),
    )
    with pytest.raises(TypeError, match="env_reward.*float32"):
        step11_update(
            agent,
            state,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.zeros(4, dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="next_observations must have shape"):
        run_step11_scan(
            agent,
            state,
            jnp.zeros(2, dtype=jnp.float32),
            jnp.zeros((3, 4), dtype=jnp.float32),
        )
