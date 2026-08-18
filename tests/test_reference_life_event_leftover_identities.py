"""Leftover-identity gates for reference-life event and regime records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.reference_life import (
    PendingOutcome,
    ReferenceEnvironmentExecution,
    ReferenceEnvironmentStart,
    ReferenceLifeEvent,
)

_DIGEST = "0" * 64


class _ExplodingHashMeta(type):
    def __hash__(cls) -> int:
        raise AssertionError("hostile runtime-class hash executed")


class _HostileScalar(metaclass=_ExplodingHashMeta):
    pass


def _legal_event(**overrides: object) -> ReferenceLifeEvent:
    payload: dict[str, object] = {
        "command": object(),
        "receipt": object(),
        "transaction": object(),
        "step_result": object(),
        "regime_id": 0,
        "oracle_reward": 0.5,
        "transcript_sha256": _DIGEST,
        "recovered": False,
    }
    payload.update(overrides)
    return ReferenceLifeEvent(**payload)  # type: ignore[arg-type]


def test_reference_life_event_rejects_leftover_identities() -> None:
    """Public life events must not keep leftover recovered/regime/reward identities."""

    with pytest.raises(ValueError, match="recovered"):
        _legal_event(recovered=1)
    with pytest.raises(ValueError, match="recovered"):
        _legal_event(recovered=0)
    with pytest.raises(ValueError, match="recovered"):
        _legal_event(recovered="FIXED")
    with pytest.raises(ValueError, match="regime_id"):
        _legal_event(regime_id=True)
    with pytest.raises(ValueError, match="regime_id"):
        _legal_event(regime_id=False)
    with pytest.raises(ValueError, match="oracle_reward"):
        _legal_event(oracle_reward=True)
    with pytest.raises(ValueError, match="oracle_reward"):
        _legal_event(oracle_reward=float("nan"))
    with pytest.raises(ValueError, match="transcript_sha256"):
        _legal_event(transcript_sha256="FIXED")
    with pytest.raises(ValueError, match="transcript_sha256"):
        _legal_event(transcript_sha256=True)

    legal = _legal_event(recovered=True, oracle_reward=0.5)
    dumped = json.dumps(
        {
            "recovered": legal.recovered,
            "regime_id": legal.regime_id,
            "oracle_reward": legal.oracle_reward,
        },
        allow_nan=False,
    )
    assert dumped == '{"recovered": true, "regime_id": 0, "oracle_reward": 0.5}'
    assert '"recovered": 1' not in dumped
    assert legal.recovered is True
    assert legal.regime_id == 0


def test_reference_life_hosts_reject_leftover_regime_and_reward_identities() -> None:
    """Sibling start/execution/pending hosts must reject leftover True==1 identities."""

    with pytest.raises(ValueError, match="regime_id"):
        ReferenceEnvironmentStart(state=object(), observation=object(), regime_id=True)
    with pytest.raises(ValueError, match="regime_id"):
        PendingOutcome(transaction=object(), regime_id=True, oracle_reward=0.0)
    with pytest.raises(ValueError, match="oracle reward"):
        PendingOutcome(transaction=object(), regime_id=0, oracle_reward=True)
    with pytest.raises(ValueError, match="reward"):
        ReferenceEnvironmentExecution(
            command=object(),
            state=object(),
            applied_action=object(),
            next_observation=object(),
            reward=True,
            discount=1.0,
            terminated=False,
            truncated=False,
            autoreset=False,
            regime_id=0,
            oracle_reward=0.0,
        )
    with pytest.raises(ValueError, match="terminated"):
        ReferenceEnvironmentExecution(
            command=object(),
            state=object(),
            applied_action=object(),
            next_observation=object(),
            reward=0.0,
            discount=1.0,
            terminated=1,
            truncated=False,
            autoreset=False,
            regime_id=0,
            oracle_reward=0.0,
        )


def test_real_type_gates_do_not_hash_hostile_runtime_classes() -> None:
    hostile = _HostileScalar()

    with pytest.raises(ValueError, match="recovered"):
        _legal_event(recovered=hostile)
    with pytest.raises(ValueError, match="regime_id"):
        _legal_event(regime_id=hostile)
    with pytest.raises(ValueError, match="oracle_reward"):
        _legal_event(oracle_reward=hostile)
