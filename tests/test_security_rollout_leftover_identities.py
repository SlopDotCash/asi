"""Leftover-identity gates for security rollout-step records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.security import (
    SecurityAction,
    SecurityFeatureSchema,
    SecurityRolloutStep,
    ThroughputMeasurement,
    to_security_gym_action,
    validate_security_rollout,
)


class _ExplodingHashMeta(type):
    def __hash__(cls) -> int:
        raise AssertionError("hostile runtime-class hash executed")


class _HostileScalar(metaclass=_ExplodingHashMeta):
    pass


def _legal_step(**overrides: object) -> SecurityRolloutStep:
    payload: dict[str, object] = {
        "state": (0.0, 1.0),
        "action": SecurityAction.PASS,
        "reward": 0.5,
        "next_state": (0.0, 1.0),
        "terminated": False,
        "truncated": False,
    }
    payload.update(overrides)
    return SecurityRolloutStep(**payload)  # type: ignore[arg-type]


def test_security_rollout_step_rejects_leftover_identities() -> None:
    """Public rollout records must not keep leftover bool/string identities."""

    with pytest.raises(ValueError, match="reward"):
        _legal_step(reward=True)
    with pytest.raises(ValueError, match="reward"):
        _legal_step(reward=False)
    with pytest.raises(ValueError, match="terminated"):
        _legal_step(terminated=1)
    with pytest.raises(ValueError, match="truncated"):
        _legal_step(truncated=0)
    with pytest.raises(ValueError, match="action"):
        _legal_step(action=True)
    with pytest.raises(ValueError, match="action"):
        _legal_step(action="FIXED")
    with pytest.raises(ValueError, match="action"):
        _legal_step(action="BLOCK")
    with pytest.raises(ValueError, match="exact tuples"):
        _legal_step(state=[0.0, 1.0])
    with pytest.raises(ValueError, match=r"state\[1\]"):
        _legal_step(state=(0.0, float("nan")))
    with pytest.raises(ValueError, match="reward"):
        _legal_step(reward=float("inf"))
    with pytest.raises(ValueError, match="policy_metadata"):
        _legal_step(policy_metadata={"bad": object()})

    legal = _legal_step()
    dumped = json.dumps(legal.to_dict(), allow_nan=False)
    assert '"reward": 0.5' in dumped
    assert '"terminated": false' in dumped
    assert '"reward": true' not in dumped
    assert '"terminated": 1' not in dumped
    assert legal.action is SecurityAction.PASS


def test_real_type_gates_do_not_hash_hostile_runtime_classes() -> None:
    hostile = _HostileScalar()

    with pytest.raises(ValueError, match=r"state\[0\]"):
        _legal_step(state=(hostile, 1.0))
    with pytest.raises(ValueError, match="risk_score"):
        to_security_gym_action(SecurityAction.PASS, hostile)
    with pytest.raises(ValueError, match="n_events"):
        ThroughputMeasurement(n_events=hostile, elapsed_s=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="elapsed_s"):
        ThroughputMeasurement(n_events=1, elapsed_s=hostile)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reward", float("nan")),
        ("action", 0),
        ("terminated", 0),
        ("state", [0.0, 1.0]),
    ],
)
def test_validate_security_rollout_revalidates_mutated_records(
    field: str,
    value: object,
) -> None:
    step = _legal_step()
    object.__setattr__(step, field, value)

    with pytest.raises(ValueError, match="invalid rollout step 0"):
        validate_security_rollout(
            [step],
            SecurityFeatureSchema(names=("first", "second")),
        )
