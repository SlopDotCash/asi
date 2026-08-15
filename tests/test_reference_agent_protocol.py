"""Failing-first tests for ASI's host-facing reference-agent transactions.

These contracts establish only L0 interoperability and ownership semantics.
They do not select ``reference-dev`` or establish learning, retention, safety,
robotics readiness, exact whole-life resume, or scientific evidence.
"""

from __future__ import annotations

import copy
import math
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import alberta_framework.reference_agent as reference_agent
from alberta_framework.reference_agent import (
    REFERENCE_AGENT_API_VERSION,
    REFERENCE_AGENT_MANIFEST_SCHEMA,
    REFERENCE_TRANSACTION_STATE_SCHEMA,
    AgentCapabilities,
    AgentManifest,
    ArrayValue,
    AuthorizationStatus,
    Decision,
    DecisionOwnershipError,
    DispatchAck,
    DispatchAuthorization,
    DispatchReceipt,
    DispatchStatus,
    ReferenceTransactionLedger,
    SpaceSpec,
    StepResult,
    Transaction,
    TransactionPhase,
    canonical_config_sha256,
)


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


def _manifest(*, dispatch_rebinding: bool = False) -> AgentManifest:
    return AgentManifest.from_config(
        schema=REFERENCE_AGENT_MANIFEST_SCHEMA,
        implementation_id="tests.fake_reference_adapter.v1",
        state_schema="tests.fake_reference_state.v1",
        config={"agent": "fake", "action_scale": 1.0, "seed": 7},
        observation_spec=_observation_spec(),
        action_spec=_action_spec(),
        capabilities=AgentCapabilities(dispatch_rebinding=dispatch_rebinding),
    )


def _decision(
    manifest: AgentManifest,
    *,
    lifecycle_id: str = "life-a",
    decision_index: int = 0,
    observation_id: str | None = None,
    observation: object = (0.0, 0.0, 0.0),
    action: object = (0.25, -0.25),
    armed: bool = True,
) -> Decision:
    return manifest.make_decision(
        lifecycle_id=lifecycle_id,
        decision_index=decision_index,
        observation_id=(
            f"{lifecycle_id}:observation:{decision_index}"
            if observation_id is None
            else observation_id
        ),
        observation=observation,
        proposed_action=action if armed else None,
        armed=armed,
    )


def _armed_ledger(
    *,
    dispatch_rebinding: bool = False,
) -> tuple[ReferenceTransactionLedger, object, Decision]:
    ledger = ReferenceTransactionLedger(_manifest(dispatch_rebinding=dispatch_rebinding))
    state = ledger.init()
    decision = _decision(ledger.manifest)
    return ledger, ledger.arm(state, decision), decision


def _authorized(
    *,
    dispatch_rebinding: bool = False,
    effective_action: object | None = None,
) -> tuple[
    ReferenceTransactionLedger,
    object,
    Decision,
    DispatchAuthorization,
]:
    ledger, state, decision = _armed_ledger(dispatch_rebinding=dispatch_rebinding)
    state, authorization = ledger.authorize(
        state,
        decision,
        authorized_action=effective_action,
        authority_id="tests.safety_authority.v1",
        policy_version="tests.safety_policy.v1",
        authorization_id=f"{decision.decision_id}:authorization",
    )
    return ledger, state, decision, authorization


def _settled(
    *,
    dispatch_rebinding: bool = False,
    effective_action: object | None = None,
) -> tuple[
    ReferenceTransactionLedger,
    object,
    Decision,
    DispatchAuthorization,
    DispatchAck,
]:
    ledger, state, decision, authorization = _authorized(
        dispatch_rebinding=dispatch_rebinding,
        effective_action=effective_action,
    )
    state, dispatch = ledger.settle_dispatch(
        state,
        authorization,
        rebinding_applied=(effective_action is not None),
        settlement_id=f"{decision.decision_id}:settlement",
    )
    assert dispatch is not None
    return ledger, state, decision, authorization, dispatch


def _dispatched(
    *,
    dispatch_rebinding: bool = False,
    effective_action: object | None = None,
) -> tuple[
    ReferenceTransactionLedger,
    object,
    Decision,
    DispatchAuthorization,
    DispatchAck,
    DispatchReceipt,
]:
    ledger, state, decision, authorization, dispatch = _settled(
        dispatch_rebinding=dispatch_rebinding,
        effective_action=effective_action,
    )
    state, receipt = ledger.record_dispatch(
        state,
        dispatch,
        receipt_id=f"{decision.decision_id}:receipt",
        executor_id="tests.environment_executor.v1",
    )
    return ledger, state, decision, authorization, dispatch, receipt


def _outcome(
    *,
    terminated: bool = False,
    truncated: bool = False,
    autoreset: bool = False,
    bootstrap_observation: object = (0.1, 0.2, 0.3),
    bootstrap_observation_id: str = "life-a:observation:1",
    next_decision_observation: object | None = (0.1, 0.2, 0.3),
    next_decision_observation_id: str | None = "life-a:observation:1",
) -> tuple[ReferenceTransactionLedger, object, Transaction]:
    ledger, state, _decision_value, _authorization, _dispatch, receipt = _dispatched()
    state, transaction = ledger.record_outcome(
        state,
        receipt,
        reward=1.0,
        discount=0.0 if terminated else 0.9,
        terminated=terminated,
        truncated=truncated,
        autoreset=autoreset,
        bootstrap_observation_id=bootstrap_observation_id,
        bootstrap_observation=bootstrap_observation,
        next_decision_observation_id=next_decision_observation_id,
        next_decision_observation=next_decision_observation,
    )
    return ledger, state, transaction


def test_versioned_manifest_binds_canonical_config_and_state_schema() -> None:
    assert REFERENCE_AGENT_API_VERSION == "asi.reference_agent.preview1"
    assert REFERENCE_AGENT_MANIFEST_SCHEMA == "asi.reference_agent_manifest.preview1"
    assert REFERENCE_TRANSACTION_STATE_SCHEMA == "asi.reference_transaction_state.preview1"

    first = {"seed": 7, "nested": {"beta": 2, "alpha": 1}}
    reordered = {"nested": {"alpha": 1, "beta": 2}, "seed": 7}
    assert canonical_config_sha256(first) == canonical_config_sha256(reordered)
    with pytest.raises(ValueError, match="finite|JSON"):
        canonical_config_sha256({"invalid": math.nan})

    manifest = _manifest()
    assert manifest.state_schema == "tests.fake_reference_state.v1"
    assert len(manifest.config_sha256) == 64
    assert len(manifest.manifest_id) == 64
    assert manifest.config == {"action_scale": 1.0, "agent": "fake", "seed": 7}
    with pytest.raises(FrozenInstanceError):
        manifest.implementation_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="schema"):
        AgentManifest.from_config(
            schema="alberta.reference_agent.v0",
            implementation_id="tests.fake_reference_adapter.v1",
            state_schema="tests.fake_reference_state.v1",
            config={},
            observation_spec=_observation_spec(),
            action_spec=_action_spec(),
            capabilities=AgentCapabilities(dispatch_rebinding=False),
        )


def test_space_encoding_is_exact_dtype_bounded_and_deeply_immutable() -> None:
    source = np.array([0.25, -0.5], dtype=np.float32)
    encoded = _action_spec().encode(source)
    assert isinstance(encoded, ArrayValue)
    source[:] = 9.0
    np.testing.assert_array_equal(encoded.to_numpy(), np.array([0.25, -0.5], np.float32))
    assert encoded.to_python() == (0.25, -0.5)
    with pytest.raises(FrozenInstanceError):
        encoded.payload = b""  # type: ignore[misc]

    with pytest.raises(ValueError, match="dtype|float32"):
        _action_spec().encode(np.array([0.25, -0.5], dtype=np.float64))
    with pytest.raises(ValueError, match="shape"):
        _action_spec().encode((0.25,))
    with pytest.raises(ValueError, match="finite"):
        _action_spec().encode((0.25, math.nan))
    with pytest.raises(ValueError, match="bounds|range"):
        _action_spec().encode((0.25, 1.5))

    discrete = SpaceSpec.discrete(
        cardinality=4,
        dtype="int32",
        semantic_id="tests.discrete_action.v1",
    )
    assert discrete.encode(3).to_python() == 3
    with pytest.raises((TypeError, ValueError), match="integer|bool"):
        discrete.encode(True)
    with pytest.raises(ValueError, match="cardinality|range"):
        discrete.encode(4)
    with pytest.raises(ValueError, match="represent|int8|cardinality"):
        SpaceSpec.discrete(
            cardinality=300,
            dtype="int8",
            semantic_id="tests.too_wide.v1",
        )


def test_manifest_factory_binds_codec_manifest_identity_and_armed_state() -> None:
    manifest = _manifest()
    decision = _decision(manifest)
    assert decision.manifest_id == manifest.manifest_id
    assert decision.decision_id == "life-a:0"
    assert decision.decision_index == 0
    assert decision.observation.semantic_id == manifest.observation_spec.semantic_id
    assert decision.proposed_action is not None
    assert decision.proposed_action.semantic_id == manifest.action_spec.semantic_id
    manifest.validate_decision(decision)

    with pytest.raises(ValueError, match="manifest"):
        manifest.validate_decision(replace(decision, manifest_id="0" * 64))
    with pytest.raises(ValueError, match="canonical|decision_id"):
        replace(decision, decision_id="life-a:reused")
    disarmed = _decision(manifest, armed=False)
    assert disarmed.proposed_action is None and not disarmed.armed


def test_authorization_settlement_and_receipt_are_distinct_owned_records() -> None:
    ledger, state, decision, authorization = _authorized()
    assert state.phase is TransactionPhase.AUTHORIZED
    assert authorization.status is AuthorizationStatus.EXACT
    assert authorization.authorized_action == decision.proposed_action
    assert authorization.authority_id == "tests.safety_authority.v1"

    state, dispatch = ledger.settle_dispatch(
        state,
        authorization,
        rebinding_applied=False,
        settlement_id="life-a:0:settlement",
    )
    assert state.phase is TransactionPhase.SETTLED
    assert dispatch is not None and dispatch.status is DispatchStatus.EXACT
    assert dispatch.authorized and dispatch.transition_expected
    assert not hasattr(dispatch, "dispatched")

    state, receipt = ledger.record_dispatch(
        state,
        dispatch,
        receipt_id="life-a:0:receipt",
        executor_id="tests.environment_executor.v1",
    )
    assert state.phase is TransactionPhase.DISPATCHED
    assert receipt.dispatch == dispatch
    assert receipt.effective_action == dispatch.effective_action

    with pytest.raises(DecisionOwnershipError, match="phase|dispatch"):
        ledger.record_dispatch(
            state,
            dispatch,
            receipt_id="life-a:0:duplicate-receipt",
            executor_id="tests.environment_executor.v1",
        )


def test_replacement_requires_declared_and_applied_credit_rebinding() -> None:
    ledger, state, _decision_value, authorization = _authorized(
        dispatch_rebinding=False,
        effective_action=(-0.5, 0.5),
    )
    assert authorization.status is AuthorizationStatus.REPLACED
    halted, dispatch = ledger.settle_dispatch(
        state,
        authorization,
        rebinding_applied=False,
        settlement_id="life-a:0:settlement",
    )
    assert dispatch is None
    assert halted.phase is TransactionPhase.HALTED
    assert "rebind" in (halted.halt_reason or "")

    ledger, state, _decision_value, authorization = _authorized(
        dispatch_rebinding=True,
        effective_action=(-0.5, 0.5),
    )
    with pytest.raises(ValueError, match="rebinding_applied"):
        ledger.settle_dispatch(
            state,
            authorization,
            rebinding_applied=False,
            settlement_id="life-a:0:settlement",
            halt_on_unsupported=False,
        )
    state, dispatch = ledger.settle_dispatch(
        state,
        authorization,
        rebinding_applied=True,
        settlement_id="life-a:0:settlement",
    )
    assert dispatch is not None and dispatch.status is DispatchStatus.REBOUND


def test_veto_halts_before_settlement_dispatch_or_learning() -> None:
    ledger, state, decision = _armed_ledger()
    halted, authorization = ledger.authorize(
        state,
        decision,
        authorized_action=None,
        authority_id="tests.safety_authority.v1",
        policy_version="tests.safety_policy.v1",
        authorization_id="life-a:0:authorization",
        veto_reason="safety envelope",
    )
    assert authorization.status is AuthorizationStatus.VETOED
    assert halted.phase is TransactionPhase.HALTED
    with pytest.raises(DecisionOwnershipError, match="phase|veto"):
        ledger.settle_dispatch(
            halted,
            authorization,
            rebinding_applied=False,
            settlement_id="life-a:0:settlement",
        )


def test_outcome_binds_exact_receipt_and_explicit_boundary_reset_identity() -> None:
    ledger, state, _decision_value, _authorization, _dispatch, receipt = _dispatched()
    wrong_receipt = replace(receipt, receipt_id="life-a:0:other-receipt")
    with pytest.raises(DecisionOwnershipError, match="receipt"):
        ledger.record_outcome(
            state,
            wrong_receipt,
            reward=1.0,
            discount=0.9,
            terminated=False,
            truncated=False,
            autoreset=False,
            bootstrap_observation_id="life-a:observation:1",
            bootstrap_observation=(0.1, 0.2, 0.3),
            next_decision_observation_id="life-a:observation:1",
            next_decision_observation=(0.1, 0.2, 0.3),
        )

    same_value = (0.0, 0.0, 0.0)
    state, transaction = ledger.record_outcome(
        state,
        receipt,
        reward=1.0,
        discount=0.0,
        terminated=True,
        truncated=False,
        autoreset=True,
        bootstrap_observation_id="life-a:terminal:0",
        bootstrap_observation=same_value,
        next_decision_observation_id="life-a:reset:1",
        next_decision_observation=same_value,
    )
    assert state.phase is TransactionPhase.OUTCOME
    assert transaction.is_boundary and transaction.is_autoreset
    assert transaction.bootstrap_observation == transaction.next_decision_observation


@pytest.mark.parametrize("discount", [-0.1, 1.1, math.nan, math.inf])
def test_outcome_rejects_invalid_discount_without_advancing(discount: float) -> None:
    ledger, state, _decision_value, _authorization, _dispatch, receipt = _dispatched()
    before = state
    with pytest.raises(ValueError, match="discount"):
        ledger.record_outcome(
            state,
            receipt,
            reward=1.0,
            discount=discount,
            terminated=False,
            truncated=False,
            autoreset=False,
            bootstrap_observation_id="life-a:observation:1",
            bootstrap_observation=(0.1, 0.2, 0.3),
            next_decision_observation_id="life-a:observation:1",
            next_decision_observation=(0.1, 0.2, 0.3),
        )
    assert state == before


def test_ledger_enforces_one_lifecycle_monotonic_decisions_and_exactly_once_phases() -> None:
    ledger, armed, decision = _armed_ledger()
    with pytest.raises(DecisionOwnershipError, match="phase|armed"):
        ledger.arm(armed, decision)
    other_ledger = ReferenceTransactionLedger(ledger.manifest)
    with pytest.raises(DecisionOwnershipError, match="decision index|expected"):
        other_ledger.arm(
            other_ledger.init(),
            _decision(other_ledger.manifest, decision_index=1),
        )

    ledger, state, transaction = _outcome()
    next_decision = _decision(
        ledger.manifest,
        decision_index=1,
        observation_id=transaction.next_decision_observation_id,
        observation=transaction.next_decision_observation.to_numpy(),
    )
    state, result = ledger.accept(
        state,
        next_decision=next_decision,
        parameters_changed=True,
    )
    assert state.phase is TransactionPhase.ARMED
    assert state.next_decision_index == 1
    assert result.transaction_accepted and result.next_decision == next_decision

    with pytest.raises(DecisionOwnershipError, match="phase|outcome"):
        ledger.accept(state, next_decision=None, parameters_changed=False)

    lifecycle_ledger, boundary, _transaction = _outcome(
        terminated=True,
        autoreset=False,
        bootstrap_observation=(0.0, 0.0, 0.0),
        bootstrap_observation_id="life-a:terminal:0",
        next_decision_observation=None,
        next_decision_observation_id=None,
    )
    ready, _result = lifecycle_ledger.accept(
        boundary,
        next_decision=None,
        parameters_changed=False,
    )
    with pytest.raises(DecisionOwnershipError, match="lifecycle"):
        lifecycle_ledger.arm(
            ready,
            _decision(lifecycle_ledger.manifest, lifecycle_id="life-b", decision_index=1),
        )


def test_ledger_rejects_reinitialization_and_stale_snapshot_forks() -> None:
    ledger = ReferenceTransactionLedger(_manifest(dispatch_rebinding=True))
    initial = ledger.init()
    with pytest.raises(DecisionOwnershipError, match="initialized"):
        ledger.init()

    decision = _decision(ledger.manifest)
    armed = ledger.arm(initial, decision)
    with pytest.raises(DecisionOwnershipError, match="stale|replayed|current"):
        ledger.arm(initial, _decision(ledger.manifest, lifecycle_id="life-b"))
    with pytest.raises(DecisionOwnershipError, match="stale|replayed|current"):
        ledger.authorize(
            copy.copy(armed),
            decision,
            authorized_action=None,
            authority_id="tests.safety_authority.v1",
            policy_version="tests.safety_policy.v1",
            authorization_id="life-a:0:copied-authorization",
        )

    ledger.authorize(
        armed,
        decision,
        authorized_action=None,
        authority_id="tests.safety_authority.v1",
        policy_version="tests.safety_policy.v1",
        authorization_id="life-a:0:first-authorization",
    )
    with pytest.raises(DecisionOwnershipError, match="stale|replayed|current"):
        ledger.authorize(
            armed,
            decision,
            authorized_action=(-0.5, 0.5),
            authority_id="tests.safety_authority.v1",
            policy_version="tests.safety_policy.v1",
            authorization_id="life-a:0:forked-authorization",
        )
    with pytest.raises((TypeError, pickle.PicklingError)):
        pickle.dumps(ledger)


def test_ledger_rejects_reusing_an_outcome_after_rejection() -> None:
    ledger, outcome, _transaction = _outcome()
    halted, rejected = ledger.reject(outcome, reason="atomic learner update rejected")
    assert halted.phase is TransactionPhase.HALTED
    assert rejected.retry_required

    with pytest.raises(DecisionOwnershipError, match="stale|replayed|current"):
        ledger.accept(outcome, next_decision=None, parameters_changed=True)


def test_ledger_serializes_concurrent_authorization_of_one_snapshot() -> None:
    ledger, armed, decision = _armed_ledger(dispatch_rebinding=True)
    barrier = threading.Barrier(2)

    def authorize(suffix: str) -> object:
        barrier.wait()
        return ledger.authorize(
            armed,
            decision,
            authorized_action=None,
            authority_id="tests.safety_authority.v1",
            policy_version="tests.safety_policy.v1",
            authorization_id=f"life-a:0:{suffix}",
        )

    successes = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(authorize, suffix) for suffix in ("first", "second")]
        for future in futures:
            try:
                future.result()
            except DecisionOwnershipError as exc:
                assert "stale" in str(exc) or "current" in str(exc)
                failures += 1
            else:
                successes += 1
    assert (successes, failures) == (1, 1)


def test_decision_counter_is_bounded() -> None:
    with pytest.raises(ValueError, match="decision_index"):
        _decision(_manifest(), decision_index=2**64)


def test_final_uint64_decision_is_consumed_once_then_exhausts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reference_agent, "_MAX_DECISION_INDEX", 0)
    ledger, outcome, transaction = _outcome()
    exhausted, result = ledger.accept(
        outcome,
        next_decision=None,
        parameters_changed=True,
    )
    assert result.transaction == transaction
    assert result.transaction_accepted
    assert exhausted.phase is TransactionPhase.EXHAUSTED
    assert exhausted.next_decision_index == 0
    assert exhausted.lifecycle_id == transaction.lifecycle_id
    assert exhausted.decision is None
    with pytest.raises(DecisionOwnershipError, match="phase|exhausted"):
        ledger.arm(exhausted, _decision(ledger.manifest))


def test_protocol_rejects_host_scalars_that_overflow_or_underflow_binary64() -> None:
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("platform longdouble has no wider exponent range than float64")
    _ledger, _state, transaction = _outcome()
    huge = np.longdouble("1e4000")
    tiny = np.nextafter(np.longdouble(0), np.longdouble(1))

    with pytest.raises(ValueError, match="reward.*binary64"):
        replace(transaction, reward=huge)
    with pytest.raises(ValueError, match="reward.*underflow"):
        replace(transaction, reward=tiny)
    with pytest.raises(ValueError, match="interoperable"):
        SpaceSpec.box(
            shape=(),
            dtype=np.dtype(np.longdouble).name,
            low=None,
            high=None,
            semantic_id="tests.unsupported_float.v1",
        )
    with pytest.raises(ValueError, match="interoperable"):
        ArrayValue(
            semantic_id="tests.unsupported_float.v1",
            dtype=np.dtype(np.longdouble).name,
            shape=(),
            payload=np.asarray(1.0, dtype=np.longdouble).tobytes(),
        )


def test_rejected_transaction_cannot_consume_event_or_arm_next_decision() -> None:
    ledger, state, transaction = _outcome()
    next_decision = _decision(
        ledger.manifest,
        decision_index=1,
        observation_id=transaction.next_decision_observation_id,
        observation=transaction.next_decision_observation.to_numpy(),
    )
    before_index = state.next_decision_index
    halted, result = ledger.reject(state, reason="atomic learner update rejected")
    assert halted.phase is TransactionPhase.HALTED
    assert halted.next_decision_index == before_index
    assert halted.transaction == transaction
    assert not result.transaction_accepted
    assert not result.parameters_changed
    assert result.retry_required
    assert result.next_decision is None

    with pytest.raises(ValueError, match="rejected|next decision|next_decision"):
        StepResult(
            transaction=transaction,
            next_decision=next_decision,
            transaction_accepted=False,
            parameters_changed=False,
            retry_required=True,
            rejection_reason="atomic learner update rejected",
        )


def test_step_result_rejects_disarmed_or_cross_manifest_next_decision() -> None:
    ledger, state, transaction = _outcome()
    disarmed = _decision(
        ledger.manifest,
        decision_index=1,
        observation_id=transaction.next_decision_observation_id,
        observation=transaction.next_decision_observation.to_numpy(),
        armed=False,
    )
    with pytest.raises(ValueError, match="armed"):
        StepResult(
            transaction=transaction,
            next_decision=disarmed,
            transaction_accepted=True,
            parameters_changed=False,
            retry_required=False,
            rejection_reason=None,
        )

    other_manifest = AgentManifest.from_config(
        schema=REFERENCE_AGENT_MANIFEST_SCHEMA,
        implementation_id="tests.other_reference_adapter.v1",
        state_schema="tests.other_reference_state.v1",
        config={},
        observation_spec=_observation_spec(),
        action_spec=_action_spec(),
        capabilities=AgentCapabilities(dispatch_rebinding=False),
    )
    foreign = _decision(
        other_manifest,
        decision_index=1,
        observation_id=transaction.next_decision_observation_id,
        observation=transaction.next_decision_observation.to_numpy(),
    )
    with pytest.raises(DecisionOwnershipError, match="manifest"):
        StepResult(
            transaction=transaction,
            next_decision=foreign,
            transaction_accepted=True,
            parameters_changed=False,
            retry_required=False,
            rejection_reason=None,
        )


def test_boundary_without_autoreset_waits_for_explicit_next_arm() -> None:
    ledger, state, transaction = _outcome(
        terminated=True,
        autoreset=False,
        bootstrap_observation=(0.0, 0.0, 0.0),
        bootstrap_observation_id="life-a:terminal:0",
        next_decision_observation=None,
        next_decision_observation_id=None,
    )
    state, result = ledger.accept(
        state,
        next_decision=None,
        parameters_changed=True,
    )
    assert state.phase is TransactionPhase.READY
    assert result.next_decision is None

    reset_decision = _decision(
        ledger.manifest,
        decision_index=1,
        observation_id="life-a:reset:1",
        observation=(0.0, 0.0, 0.0),
    )
    state = ledger.arm(state, reset_decision)
    assert state.phase is TransactionPhase.ARMED


def test_transaction_state_rejects_phase_skips_and_foreign_record_chains() -> None:
    ledger, settled, decision, authorization, dispatch = _settled()
    fresh = ReferenceTransactionLedger(ledger.manifest).init()

    with pytest.raises(DecisionOwnershipError, match="record chain|phase"):
        replace(
            fresh,
            phase=TransactionPhase.SETTLED,
            lifecycle_id=decision.lifecycle_id,
            dispatch=dispatch,
        )
    with pytest.raises(DecisionOwnershipError, match="record chain|phase"):
        replace(settled, phase=TransactionPhase.READY)
    with pytest.raises(DecisionOwnershipError, match="lifecycle"):
        replace(settled, lifecycle_id="life-foreign")
    with pytest.raises(DecisionOwnershipError, match="index"):
        replace(settled, next_decision_index=1)
    with pytest.raises(DecisionOwnershipError, match="exact authorization"):
        replace(
            settled,
            phase=TransactionPhase.HALTED,
            dispatch=None,
            halt_reason="forged pre-dispatch halt",
        )

    no_rebind_ledger, authorized, rebind_decision, replacement = _authorized(
        dispatch_rebinding=False,
        effective_action=(-0.5, 0.5),
    )
    assert replacement.authorized_action is not None
    forged_rebound = DispatchAck(
        authorization=replacement,
        status=DispatchStatus.REBOUND,
        effective_action=replacement.authorized_action,
        settlement_id=f"{rebind_decision.decision_id}:settlement",
    )
    forged_settled = replace(
        authorized,
        phase=TransactionPhase.SETTLED,
        dispatch=forged_rebound,
    )
    with pytest.raises(DecisionOwnershipError, match="dispatch_rebinding"):
        no_rebind_ledger.record_dispatch(
            forged_settled,
            forged_rebound,
            receipt_id=f"{rebind_decision.decision_id}:receipt",
            executor_id="tests.environment_executor.v1",
        )

    foreign_manifest = AgentManifest.from_config(
        schema=REFERENCE_AGENT_MANIFEST_SCHEMA,
        implementation_id="tests.foreign_reference_adapter.v1",
        state_schema="tests.foreign_reference_state.v1",
        config={"foreign": True},
        observation_spec=_observation_spec(),
        action_spec=_action_spec(),
        capabilities=AgentCapabilities(dispatch_rebinding=False),
    )
    foreign_decision = _decision(foreign_manifest, lifecycle_id="foreign-life")
    assert foreign_decision.proposed_action is not None
    foreign_authorization = DispatchAuthorization(
        decision=foreign_decision,
        status=AuthorizationStatus.EXACT,
        authorized_action=foreign_decision.proposed_action,
        authority_id="tests.safety_authority.v1",
        policy_version="tests.safety_policy.v1",
        authorization_id="foreign-life:0:authorization",
    )
    foreign_dispatch = DispatchAck(
        authorization=foreign_authorization,
        status=DispatchStatus.EXACT,
        effective_action=foreign_decision.proposed_action,
        settlement_id="foreign-life:0:settlement",
    )
    with pytest.raises(DecisionOwnershipError, match="manifest"):
        replace(
            fresh,
            phase=TransactionPhase.SETTLED,
            lifecycle_id="foreign-life",
            decision=foreign_decision,
            authorization=foreign_authorization,
            dispatch=foreign_dispatch,
        )


def test_ledger_revalidates_every_embedded_value_against_manifest_codecs() -> None:
    ledger = ReferenceTransactionLedger(_manifest(dispatch_rebinding=True))
    initial = ledger.init()
    decision = _decision(ledger.manifest)
    assert decision.proposed_action is not None
    foreign_observation = SpaceSpec.box(
        shape=(3,),
        dtype="float32",
        low=None,
        high=None,
        semantic_id="tests.foreign_observation.v1",
    ).encode(np.asarray((0.0, 0.0, 0.0), dtype=np.float32))
    bad_decision = replace(decision, observation=foreign_observation)
    bad_armed = replace(
        initial,
        phase=TransactionPhase.ARMED,
        lifecycle_id=bad_decision.lifecycle_id,
        decision=bad_decision,
    )
    with pytest.raises(DecisionOwnershipError, match="manifest codecs"):
        ledger.authorize(
            bad_armed,
            bad_decision,
            authorized_action=None,
            authority_id="tests.safety_authority.v1",
            policy_version="tests.safety_policy.v1",
            authorization_id="life-a:0:authorization",
        )

    armed = ledger.arm(initial, decision)
    foreign_action = SpaceSpec.box(
        shape=(2,),
        dtype="float32",
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        semantic_id="tests.foreign_action.v1",
    ).encode(np.asarray((-0.5, 0.5), dtype=np.float32))
    bad_authorization = DispatchAuthorization(
        decision=decision,
        status=AuthorizationStatus.REPLACED,
        authorized_action=foreign_action,
        authority_id="tests.safety_authority.v1",
        policy_version="tests.safety_policy.v1",
        authorization_id="life-a:0:authorization",
    )
    bad_authorized = replace(
        armed,
        phase=TransactionPhase.AUTHORIZED,
        authorization=bad_authorization,
    )
    with pytest.raises(DecisionOwnershipError, match="manifest codecs"):
        ledger.settle_dispatch(
            bad_authorized,
            bad_authorization,
            rebinding_applied=True,
            settlement_id="life-a:0:settlement",
        )

    ledger, dispatched, _decision_value, _authorization, _dispatch, receipt = _dispatched()
    bad_transaction = Transaction(
        receipt=receipt,
        reward=0.0,
        discount=0.0,
        terminated=True,
        truncated=False,
        autoreset=False,
        bootstrap_observation_id="life-a:terminal:0",
        bootstrap_observation=foreign_observation,
        next_decision_observation_id=None,
        next_decision_observation=None,
    )
    bad_outcome = replace(
        dispatched,
        phase=TransactionPhase.OUTCOME,
        transaction=bad_transaction,
    )
    with pytest.raises(DecisionOwnershipError, match="manifest codecs"):
        ledger.accept(
            bad_outcome,
            next_decision=None,
            parameters_changed=False,
        )
