"""Failing-first tests for ASI's host-facing reference-agent transactions.

These contracts establish only L0 interoperability and ownership semantics.
They do not select ``reference-dev`` or establish learning, retention, safety,
robotics readiness, exact whole-life resume, or scientific evidence.
"""

from __future__ import annotations

import math
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework.reference_agent as reference_agent
from alberta_framework.reference_agent import (
    MAX_DECISION_INDEX,
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
    ReferenceTransactionState,
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
) -> tuple[ReferenceTransactionLedger, ReferenceTransactionState, Decision]:
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
    ReferenceTransactionState,
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
    ReferenceTransactionState,
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
    ReferenceTransactionState,
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
) -> tuple[ReferenceTransactionLedger, ReferenceTransactionState, Transaction]:
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
    assert MAX_DECISION_INDEX == (1 << 64) - 1
    assert [field.name for field in fields(AgentCapabilities)] == ["dispatch_rebinding"]

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
    with pytest.raises(ValueError, match="dtype|float32"):
        _action_spec().encode(jnp.array([0.25, -0.5], dtype=jnp.float16))
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
    with pytest.raises((TypeError, ValueError), match="integer"):
        discrete.encode(1.0)
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
    with pytest.raises(ValueError, match="lifecycle_id|derived|length|long"):
        _decision(
            manifest,
            lifecycle_id="l" * 243,
            observation_id="tests.observation:0",
        )
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
    assert not hasattr(dispatch, "dispatched")
    assert not hasattr(dispatch, "transition_expected")

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
    with pytest.raises(ValueError, match="receipt_id|canonical"):
        replace(receipt, receipt_id="life-a:0:other-receipt")

    alternate_authorization = replace(
        receipt.dispatch.authorization,
        authority_id="tests.other_safety_authority.v1",
    )
    alternate_dispatch = replace(receipt.dispatch, authorization=alternate_authorization)
    alternate_receipt = replace(receipt, dispatch=alternate_dispatch)
    with pytest.raises(DecisionOwnershipError, match="receipt"):
        ledger.record_outcome(
            state,
            alternate_receipt,
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


def test_continuing_outcome_cannot_use_terminal_zero_discount() -> None:
    ledger, state, _decision_value, _authorization, _dispatch, receipt = _dispatched()
    with pytest.raises(ValueError, match="discount|terminated"):
        ledger.record_outcome(
            state,
            receipt,
            reward=1.0,
            discount=0.0,
            terminated=False,
            truncated=False,
            autoreset=False,
            bootstrap_observation_id="life-a:observation:1",
            bootstrap_observation=(0.1, 0.2, 0.3),
            next_decision_observation_id="life-a:observation:1",
            next_decision_observation=(0.1, 0.2, 0.3),
        )


def test_outcome_rejects_scalar_that_overflows_during_protocol_conversion() -> None:
    ledger, state, _decision_value, _authorization, _dispatch, receipt = _dispatched()
    huge = 10**400
    with pytest.raises(ValueError, match="reward|finite|representable"):
        ledger.record_outcome(
            state,
            receipt,
            reward=huge,
            discount=0.9,
            terminated=False,
            truncated=False,
            autoreset=False,
            bootstrap_observation_id="life-a:observation:1",
            bootstrap_observation=(0.1, 0.2, 0.3),
            next_decision_observation_id="life-a:observation:1",
            next_decision_observation=(0.1, 0.2, 0.3),
        )

    tiny = Fraction(1, 10**400)
    with pytest.raises(ValueError, match="reward|underflow|representable"):
        ledger.record_outcome(
            state,
            receipt,
            reward=tiny,
            discount=0.9,
            terminated=False,
            truncated=False,
            autoreset=False,
            bootstrap_observation_id="life-a:observation:1",
            bootstrap_observation=(0.1, 0.2, 0.3),
            next_decision_observation_id="life-a:observation:1",
            next_decision_observation=(0.1, 0.2, 0.3),
        )


def test_ledger_enforces_one_lifecycle_monotonic_decisions_and_exactly_once_phases() -> None:
    ledger, armed, decision = _armed_ledger()
    with pytest.raises(DecisionOwnershipError, match="phase|armed"):
        ledger.arm(armed, decision)
    with pytest.raises(DecisionOwnershipError, match="initialized|current|replay"):
        ledger.init()

    ledger, state, transaction = _outcome()
    assert transaction.next_decision_observation_id is not None
    assert transaction.next_decision_observation is not None
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
    with pytest.raises(DecisionOwnershipError, match="current|replay|stale"):
        ledger.arm(
            replace(state, phase=TransactionPhase.READY, decision=None),
            _decision(ledger.manifest, lifecycle_id="life-b", decision_index=1),
        )


def test_ledger_rejects_old_state_replay_and_phase_or_chain_forgery() -> None:
    ledger, settled, decision, _authorization, dispatch = _settled()
    dispatched, _receipt = ledger.record_dispatch(
        settled,
        dispatch,
        receipt_id="life-a:0:receipt",
        executor_id="tests.environment_executor.v1",
    )
    with pytest.raises(DecisionOwnershipError, match="current|replay|stale"):
        ledger.record_dispatch(
            settled,
            dispatch,
            receipt_id="life-a:0:second-receipt",
            executor_id="tests.environment_executor.v1",
        )
    with pytest.raises(DecisionOwnershipError, match="current|replay|stale"):
        ledger.record_outcome(
            replace(dispatched),
            _receipt,
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

    with pytest.raises(ValueError, match="phase|fields|records"):
        replace(dispatched, phase=TransactionPhase.READY)

    rolled_back = replace(
        settled,
        phase=TransactionPhase.READY,
        lifecycle_id=decision.lifecycle_id,
        decision=None,
        authorization=None,
        dispatch=None,
    )
    with pytest.raises(DecisionOwnershipError, match="current|replay|stale"):
        ledger.arm(rolled_back, decision)

    other = _manifest()
    foreign_decision = _decision(other, lifecycle_id="life-foreign")
    with pytest.raises(ValueError, match="manifest|lifecycle|ownership"):
        replace(settled, decision=foreign_decision)

    disarmed = _decision(
        ledger.manifest,
        lifecycle_id=decision.lifecycle_id,
        decision_index=decision.decision_index,
        armed=False,
    )
    with pytest.raises(ValueError, match="armed|disarmed"):
        replace(
            settled,
            phase=TransactionPhase.ARMED,
            decision=disarmed,
            authorization=None,
            dispatch=None,
        )


def test_ledger_serializes_concurrent_authorization_of_one_snapshot() -> None:
    ledger, armed, decision = _armed_ledger(dispatch_rebinding=True)
    barrier = threading.Barrier(2)

    def authorize() -> object:
        barrier.wait()
        return ledger.authorize(
            armed,
            decision,
            authorized_action=None,
            authority_id="tests.safety_authority.v1",
            policy_version="tests.safety_policy.v1",
            authorization_id="life-a:0:authorization",
        )

    successes = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(authorize) for _ in range(2)]
        for future in futures:
            try:
                future.result()
                successes += 1
            except DecisionOwnershipError:
                failures += 1

    assert (successes, failures) == (1, 1)


def test_live_ledger_owner_cannot_be_pickled_as_an_undeclared_resume_surface() -> None:
    ledger, state, _decision_value = _armed_ledger()
    with pytest.raises(TypeError, match="ledger|serialize|checkpoint|resume"):
        pickle.dumps((ledger, state))


def test_rejected_transaction_cannot_consume_event_or_arm_next_decision() -> None:
    ledger, state, transaction = _outcome()
    assert transaction.next_decision_observation_id is not None
    assert transaction.next_decision_observation is not None
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
    assert not result.event_consumed
    assert result.recovery_required
    assert result.next_decision is None

    with pytest.raises(ValueError, match="rejected|next decision|next_decision"):
        StepResult(
            transaction=transaction,
            next_decision=next_decision,
            transaction_accepted=False,
            parameters_changed=False,
            rejection_reason="atomic learner update rejected",
        )


def test_step_result_rejects_disarmed_or_cross_manifest_next_decision() -> None:
    ledger, state, transaction = _outcome()
    assert transaction.next_decision_observation_id is not None
    assert transaction.next_decision_observation is not None
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
            rejection_reason=None,
        )


def test_counter_is_bounded_and_final_event_is_consumed_before_disarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="decision_index|uint64|maximum"):
        _decision(manifest, decision_index=MAX_DECISION_INDEX + 1)

    monkeypatch.setattr(reference_agent, "MAX_DECISION_INDEX", 0)
    ledger, state, transaction = _outcome()
    state, result = ledger.accept(
        state,
        next_decision=None,
        parameters_changed=True,
    )
    assert result.transaction_accepted
    assert result.event_consumed
    assert result.life_exhausted
    assert state.phase is TransactionPhase.EXHAUSTED
    assert state.next_decision_index == 0


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
