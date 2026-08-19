from __future__ import annotations

from typing import Any

import pytest

from alberta_framework.streams.recurring_multiagent import RecurringTwoAgentWorld

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile lt must not run")

    def __le__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile le must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_recurring_world_rejects_hostile_int_before_range() -> None:
    hostile = _HostileInt(64)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="context_length must be positive"):
        RecurringTwoAgentWorld(context_length=hostile)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0

    hostile2 = _HostileInt(4)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="nuisance_dim must be non-negative"):
        RecurringTwoAgentWorld(nuisance_dim=hostile2)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0

    with pytest.raises(ValueError, match="context_length must be positive"):
        RecurringTwoAgentWorld(context_length=True)  # type: ignore[arg-type]


def test_recurring_world_rejects_hostile_float_before_float() -> None:
    hostile = _HostileFloat(1.0)
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="nuisance_scale must be a finite real number"):
        RecurringTwoAgentWorld(nuisance_scale=hostile)  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0

    hostile_int = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="nuisance_scale must be a finite real number"):
        RecurringTwoAgentWorld(nuisance_scale=hostile_int)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_recurring_world_benign_still_works() -> None:
    world = RecurringTwoAgentWorld()
    assert world is not None
    world2 = RecurringTwoAgentWorld(context_length=32, nuisance_dim=2, nuisance_scale=0.5)
    assert world2 is not None

    with pytest.raises(ValueError, match="must be finite"):
        RecurringTwoAgentWorld(nuisance_scale=float("inf"))  # type: ignore[arg-type]


def test_recurring_world_hostile_not_in_repr() -> None:
    hostile = _HostileInt(64)
    _HostileInt.calls = 0
    try:
        RecurringTwoAgentWorld(context_length=hostile)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "_HostileInt" not in str(exc)
        assert _HostileInt.calls == 0
    else:
        raise AssertionError("should have raised")


def test_recurring_world_from_config_roundtrip() -> None:
    world = RecurringTwoAgentWorld(
        context_length=32,
        nuisance_dim=2,
        nuisance_scale=0.5,
        initial_positions=(-0.25, 0.25),
    )
    cfg = world.to_config()
    restored = RecurringTwoAgentWorld.from_config(cfg)
    assert restored._context_length == 32
    assert restored._nuisance_dim == 2
    assert restored._nuisance_scale == 0.5
    assert restored._initial_positions_tuple == (-0.25, 0.25)


def test_recurring_world_from_config_rejects_invalid_containers() -> None:
    with pytest.raises(ValueError, match="exact dictionary"):
        RecurringTwoAgentWorld.from_config([("type", "RecurringTwoAgentWorld")])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact dictionary"):
        RecurringTwoAgentWorld.from_config("not_a_dict")  # type: ignore[arg-type]


def test_recurring_world_from_config_rejects_non_string_keys() -> None:
    valid = RecurringTwoAgentWorld().to_config()
    bad_keys: dict[Any, Any] = dict(valid)
    bad_keys[123] = "invalid_key"
    with pytest.raises(ValueError, match="keys must be exact strings"):
        RecurringTwoAgentWorld.from_config(bad_keys)  # type: ignore[arg-type]


def test_recurring_world_from_config_rejects_invalid_schema_fields() -> None:
    valid = RecurringTwoAgentWorld().to_config()
    with pytest.raises(ValueError, match="fields are invalid"):
        RecurringTwoAgentWorld.from_config({**valid, "extra_field": 1})
    missing = dict(valid)
    del missing["context_length"]
    with pytest.raises(ValueError, match="fields are invalid"):
        RecurringTwoAgentWorld.from_config(missing)


def test_recurring_world_from_config_rejects_unsupported_type_or_schema_or_policy() -> None:
    valid = RecurringTwoAgentWorld().to_config()
    with pytest.raises(ValueError, match="type is unsupported"):
        RecurringTwoAgentWorld.from_config({**valid, "type": "WrongWorld"})
    with pytest.raises(ValueError, match="config schema is unsupported"):
        RecurringTwoAgentWorld.from_config({**valid, "config_schema": "wrong.schema"})
    with pytest.raises(ValueError, match="state schema is unsupported"):
        RecurringTwoAgentWorld.from_config({**valid, "state_schema": "wrong.schema"})
    with pytest.raises(ValueError, match="partner policy is unsupported"):
        RecurringTwoAgentWorld.from_config({**valid, "partner_policy": "custom_policy"})


def test_recurring_world_from_config_rejects_invalid_initial_positions() -> None:
    valid = RecurringTwoAgentWorld().to_config()
    with pytest.raises(ValueError, match="initial_positions must be an exact list"):
        RecurringTwoAgentWorld.from_config({**valid, "initial_positions": (-0.5, 0.5)})
    with pytest.raises(ValueError, match="initial_positions must have length 2"):
        RecurringTwoAgentWorld.from_config({**valid, "initial_positions": [-0.5]})
    with pytest.raises(ValueError, match="initial_positions must have length 2"):
        RecurringTwoAgentWorld.from_config({**valid, "initial_positions": [-0.5, 0.0, 0.5]})
