"""Hostile-safe validation for security contracts."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from alberta_framework.security import (
    SecurityAction,
    SecurityFeatureSchema,
    SecurityRolloutStep,
    ThroughputMeasurement,
    ThroughputMeter,
    coerce_security_action,
    security_reward,
    security_rollout_step_to_oracle_experience,
    to_security_gym_action,
)


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr hook")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:  # type: ignore[override]
        type(self).calls += 1
        raise RuntimeError("ratio hook")


class _StringSubclass(str):
    pass


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook")


def test_coerce_rejects_hostile_int_without_hook() -> None:
    with pytest.raises(ValueError, match="security action"):
        coerce_security_action(_HostileInt(1))  # type: ignore[arg-type]


def test_coerce_rejects_string_subclass() -> None:
    with pytest.raises(ValueError, match="security action"):
        coerce_security_action(_StringSubclass("pass"))  # type: ignore[arg-type]


def test_coerce_rejects_hostile_repr() -> None:
    with pytest.raises(ValueError, match="security action"):
        coerce_security_action(_RaisingRepr())  # type: ignore[arg-type]


def test_to_gym_rejects_hostile_float_without_ratio() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="risk_score"):
        to_security_gym_action("pass", risk_score=_HostileFloat(0.5))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_to_gym_rejects_bool_risk() -> None:
    with pytest.raises(ValueError, match="risk_score"):
        to_security_gym_action("pass", risk_score=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="risk_score"):
        to_security_gym_action("pass", risk_score=np.bool_(True))  # type: ignore[arg-type]


def test_to_gym_rejects_string_subclass_risk() -> None:
    with pytest.raises(ValueError, match="risk_score"):
        to_security_gym_action("pass", risk_score=_StringSubclass("0.5"))  # type: ignore[arg-type]


def test_tick_rejects_hostile_int_without_hook() -> None:
    meter = ThroughputMeter()
    with pytest.raises(ValueError, match="n_events"):
        meter.tick(_HostileInt(1))  # type: ignore[arg-type]


def test_tick_rejects_bool() -> None:
    meter = ThroughputMeter()
    with pytest.raises(ValueError, match="n_events"):
        meter.tick(True)  # type: ignore[arg-type]


def test_schema_from_dict_preserves_mapping_proxy() -> None:
    schema = SecurityFeatureSchema(names=("a", "b"))
    payload = schema.to_dict()
    restored = SecurityFeatureSchema.from_dict(MappingProxyType(payload))
    assert restored == schema


def test_schema_from_dict_rejects_string_subclass_key() -> None:
    schema = SecurityFeatureSchema(names=("a", "b"))
    payload = schema.to_dict()
    hostile: dict[Any, Any] = {_StringSubclass("names"): payload["names"]}
    for k, v in payload.items():
        if k != "names":
            hostile[k] = v
    with pytest.raises(ValueError, match="exact strings"):
        SecurityFeatureSchema.from_dict(hostile)  # type: ignore[arg-type]


def test_schema_from_dict_rejects_hostile_mapping() -> None:
    from collections.abc import Mapping

    class HostileMapping(Mapping[str, Any]):  # type: ignore[type-arg]
        def __getitem__(self, key: str) -> Any:
            raise RuntimeError("hook")

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("iter hook")

        def __len__(self) -> int:
            return 0

    with pytest.raises(ValueError, match="mapping"):
        SecurityFeatureSchema.from_dict(HostileMapping())  # type: ignore[arg-type]


def test_schema_rejects_inconsistent_derived_dimension() -> None:
    with pytest.raises(ValueError, match="feature_dim"):
        SecurityFeatureSchema.from_dict({"names": ["a"], "feature_dim": 2})


def test_rollout_from_dict_rejects_hostile_float() -> None:
    step = SecurityRolloutStep(
        state=(0.0, 1.0),
        action=SecurityAction.PASS,
        reward=0.0,
        next_state=(0.0, 1.0),
        terminated=False,
    )
    payload = step.to_dict()
    _HostileFloat.calls = 0
    payload["reward"] = _HostileFloat(0.5)
    with pytest.raises(ValueError, match="reward"):
        SecurityRolloutStep.from_dict(payload)  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_rollout_from_dict_rejects_laundered_metadata_and_action_name() -> None:
    payload = {
        "state": [0.0],
        "action": 0,
        "reward": 0.0,
        "next_state": [0.0],
        "terminated": False,
        "policy_metadata": None,
    }
    with pytest.raises(ValueError, match="policy_metadata"):
        SecurityRolloutStep.from_dict(payload)
    with pytest.raises(ValueError, match="action_name"):
        SecurityRolloutStep.from_dict(
            {**payload, "policy_metadata": {}, "action_name": "alert"}
        )


def test_policy_metadata_rejects_nested_hostile_values_without_hook() -> None:
    class HostileList(list[object]):
        def __iter__(self):  # type: ignore[no-untyped-def, override]
            raise AssertionError("container hook executed")

    with pytest.raises(ValueError, match="exact JSON values"):
        SecurityRolloutStep(
            state=(0.0,),
            action=SecurityAction.PASS,
            reward=0.0,
            next_state=(0.0,),
            terminated=False,
            policy_metadata={"nested": HostileList()},
        )


def test_security_reward_rejects_hostile_inputs_without_numeric_hooks() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="component"):
        security_reward({"threat_blocked": _HostileFloat(1.0)})
    assert _HostileFloat.calls == 0

    class HostileComponents(dict[str, float]):
        def __iter__(self):  # type: ignore[no-untyped-def, override]
            raise AssertionError("mapping hook executed")

    with pytest.raises(ValueError, match="components"):
        security_reward(HostileComponents())


def test_oracle_conversion_requires_exact_malicious_label() -> None:
    step = SecurityRolloutStep(
        state=(0.0,),
        action=SecurityAction.PASS,
        reward=0.0,
        next_state=(0.0,),
        terminated=False,
        policy_metadata={"is_malicious": 1},
    )
    with pytest.raises(ValueError, match="exact bool"):
        security_rollout_step_to_oracle_experience(step)


def test_throughput_canonicalizes_scalars_and_rejects_cumulative_overflow() -> None:
    measurement = ThroughputMeasurement(n_events=np.int32(3), elapsed_s=np.float64(2.0))
    assert type(measurement.n_events) is int
    assert type(measurement.elapsed_s) is float
    assert measurement.to_dict()["events_per_second"] == 1.5

    meter = ThroughputMeter()
    meter.tick(2**31 - 1)
    with pytest.raises(ValueError, match="cumulative"):
        meter.tick(1)


def test_numpy_risk_canonicalizes() -> None:
    res = to_security_gym_action("alert", risk_score=np.float64(5.5))
    assert res["risk_score"] == (5.5,)
    res2 = to_security_gym_action("pass", risk_score=np.int32(5))
    assert res2["risk_score"] == (5.0,)


def test_coerce_admits_every_numpy_integer_action_id() -> None:
    """Gymnasium and ``argmax`` hand callers numpy ids, so the gate must take them."""
    families = (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    )
    for family in families:
        coerced = coerce_security_action(family(3))  # type: ignore[arg-type]
        assert coerced is SecurityAction(3)
        assert type(coerced) is SecurityAction

    # The documented producer of an action id is a discrete argmax/sample.
    greedy = np.argmax(np.array([0.1, 0.4, 0.2, 0.9, 0.3, 0.0], dtype=np.float32))
    assert coerce_security_action(greedy) is SecurityAction(3)  # type: ignore[arg-type]

    # One call, one numpy type: ``risk_score`` already accepted it, so ``action``
    # must too.
    assert to_security_gym_action(np.int64(3), np.int64(3)) == {  # type: ignore[arg-type]
        "action": 3,
        "risk_score": (3.0,),
    }


def test_coerce_still_rejects_non_integer_lookalikes() -> None:
    """Widening the integer families must not admit bools, reals, or arrays."""
    for value in (
        np.bool_(True),
        np.float32(3.0),
        np.float64(3.0),
        np.array(3),
        np.array([3]),
        np.int64(99),
    ):
        with pytest.raises(ValueError, match="security action"):
            coerce_security_action(value)  # type: ignore[arg-type]

    # An exact-type gate keeps hostile subclasses out without touching a hook.
    with pytest.raises(ValueError, match="security action"):
        coerce_security_action(_HostileInt(3))  # type: ignore[arg-type]
