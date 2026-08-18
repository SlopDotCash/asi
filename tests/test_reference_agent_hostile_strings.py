"""Hostile string identities for reference-agent veto layers."""

from __future__ import annotations

import pytest

from alberta_framework.reference_agent import (
    AgentCapabilities,
    AgentManifest,
    AuthorizationStatus,
    Decision,
    DispatchAuthorization,
    ReferenceAgentUpdate,
    ReferenceTransactionLedger,
    ReferenceTransactionState,
    SpaceSpec,
    StepResult,
    TransactionPhase,
)

pytestmark = pytest.mark.unit


class _HostileString(str):
    calls = 0

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool executed")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq executed")

    __hash__ = str.__hash__

    def strip(self, chars: str | None = None) -> str:  # type: ignore[override]
        del chars
        type(self).calls += 1
        raise AssertionError("hostile strip executed")

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile len executed")

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("hostile str executed")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("hostile repr executed")


def _observation_spec() -> SpaceSpec:
    return SpaceSpec.box(
        shape=(3,),
        dtype="float32",
        low=None,
        high=None,
        semantic_id="tests.reference_observation.v1",
    )


def _action_spec() -> SpaceSpec:
    return SpaceSpec.box(
        shape=(2,),
        dtype="float32",
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        semantic_id="tests.normalized_action.v1",
    )


def _manifest() -> AgentManifest:
    return AgentManifest.from_config(
        schema="asi.reference_agent_manifest.preview1",
        implementation_id="tests.fake_reference_adapter.v1",
        state_schema="tests.fake_reference_state.v1",
        config={"agent": "fake", "action_scale": 1.0, "seed": 7},
        observation_spec=_observation_spec(),
        action_spec=_action_spec(),
        capabilities=AgentCapabilities(dispatch_rebinding=False),
    )


def _decision(manifest: AgentManifest) -> Decision:
    return manifest.make_decision(
        lifecycle_id="life-a",
        decision_index=0,
        observation_id="life-a:observation:0",
        observation=(0.0, 0.0, 0.0),
        proposed_action=(0.25, -0.25),
        armed=True,
    )


def test_validate_json_value_rejects_hostile_key_before_dispatch() -> None:
    from alberta_framework.reference_agent import _validate_json_value

    hostile = _HostileString("evil")
    _HostileString.calls = 0
    with pytest.raises(ValueError, match="keys must be strings"):
        _validate_json_value({hostile: 1}, path="config")  # type: ignore[dict-item]
    assert _HostileString.calls == 0


def test_dispatch_authorization_veto_reason_rejects_hostile_before_strip() -> None:
    manifest = _manifest()
    decision = _decision(manifest)
    hostile = _HostileString("veto me")
    _HostileString.calls = 0
    with pytest.raises(ValueError, match="vetoed authorization requires"):
        DispatchAuthorization(
            decision=decision,
            status=AuthorizationStatus.VETOED,
            authorized_action=None,
            authority_id="tests.safety_authority.v1",
            policy_version="tests.safety_policy.v1",
            authorization_id=f"{decision.decision_id}:authorization",
            veto_reason=hostile,  # type: ignore[arg-type]
        )
    assert _HostileString.calls == 0


def test_reference_agent_update_rejection_reason_rejects_hostile() -> None:
    hostile = _HostileString("reject")
    _HostileString.calls = 0
    with pytest.raises(ValueError, match="requires a nonempty reason"):
        ReferenceAgentUpdate(
            state=None,
            next_decision=None,
            accepted=False,
            parameters_changed=False,
            rejection_reason=hostile,  # type: ignore[arg-type]
        )
    assert _HostileString.calls == 0


def test_step_result_rejection_reason_rejects_hostile() -> None:
    from alberta_framework.reference_agent import Transaction as _Tx

    tx = object.__new__(_Tx)
    hostile = _HostileString("nope")
    _HostileString.calls = 0
    with pytest.raises(ValueError, match="rejection_reason"):
        StepResult(
            transaction=tx,  # type: ignore[arg-type]
            next_decision=None,
            transaction_accepted=False,
            parameters_changed=False,
            rejection_reason=hostile,  # type: ignore[arg-type]
        )
    assert _HostileString.calls == 0


def test_reference_transaction_state_halt_reason_rejects_hostile() -> None:
    hostile = _HostileString("halted")
    _HostileString.calls = 0
    with pytest.raises(ValueError, match="halted state requires"):
        ReferenceTransactionState(
            schema="asi.reference_transaction_state.preview1",
            manifest_id="0" * 64,
            phase=TransactionPhase.HALTED,
            lifecycle_id=None,
            next_decision_index=0,
            halt_reason=hostile,  # type: ignore[arg-type]
        )
    assert _HostileString.calls == 0


def test_reference_transaction_ledger_halt_rejects_hostile_before_commit() -> None:
    manifest = _manifest()
    ledger = ReferenceTransactionLedger(manifest)
    state = ledger.init()
    hostile = _HostileString("halt")
    _HostileString.calls = 0
    with pytest.raises(ValueError, match="halt reason must be nonempty"):
        ledger._halt(state, reason=hostile)  # type: ignore[arg-type]
    assert _HostileString.calls == 0


def test_reference_ledger_halt_via_reducer_rejects_hostile() -> None:
    hostile = _HostileString("abort")
    _HostileString.calls = 0
    assert type(hostile) is not str
    assert _HostileString.calls == 0
    assert (type(hostile) is not str or not hostile.strip()) is True
    assert _HostileString.calls == 0
