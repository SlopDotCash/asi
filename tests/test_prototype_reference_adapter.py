"""Failing-first contracts for the development-only Prototype reference adapter.

This lane establishes only a primitive-action, exact-dispatch interoperability
slice.  It does not select ``reference-dev``, prove learning benefit, or provide
an aggregate runner, durable replay, exact whole-life resume, or safety claim.
"""

from __future__ import annotations

import dataclasses
import pickle

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.dreaming import DreamingConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
)
from alberta_framework.prototype_reference_adapter import (
    PROTOTYPE_REFERENCE_ADAPTER_SCHEMA,
    PROTOTYPE_REFERENCE_STATE_SCHEMA,
    PrototypeReferenceAdapter,
    PrototypeReferenceState,
)
from alberta_framework.reference_agent import (
    AuthorizationStatus,
    Decision,
    DecisionOwnershipError,
    DispatchAck,
    DispatchAuthorization,
    DispatchReceipt,
    DispatchStatus,
    ReferenceTransactionLedger,
    ReferenceTransactionState,
    Transaction,
    TransactionPhase,
)
from alberta_framework.streams.closed_loop import (
    PHASE_A,
    PHASE_B,
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
)

pytestmark = pytest.mark.unit

_LIFE_A = "prototype.0000000100000002"
_LIFE_B = "prototype.0000000300000004"


def _config(*, base_step_size: float = 0.05) -> PrototypeAgentConfig:
    return PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(),
                observation_dim=2,
                n_primitive_actions=2,
                base_step_size=base_step_size,
                epsilon_base=0.0,
            )
        )
    )


def _start(
    adapter: PrototypeReferenceAdapter,
    *,
    lifecycle_id: str = _LIFE_A,
    key_seed: int = 0,
    observation: np.ndarray | None = None,
) -> tuple[PrototypeReferenceState, Decision]:
    initial_observation = (
        np.asarray([1.0, 0.0], dtype=np.float32)
        if observation is None
        else observation
    )
    state = adapter.init(jr.key(key_seed), lifecycle_id=lifecycle_id)
    return adapter.start(
        state,
        observation_id=f"{lifecycle_id}:observation:0",
        observation=initial_observation,
    )


def _record_outcome(
    adapter: PrototypeReferenceAdapter,
    decision: Decision,
    *,
    agent_state: PrototypeReferenceState | None = None,
    reward: float = 1.0,
    discount: float = 1.0,
    terminated: bool = False,
    truncated: bool = False,
    autoreset: bool = False,
    bootstrap_observation: np.ndarray | None = None,
) -> tuple[
    ReferenceTransactionLedger,
    ReferenceTransactionState,
    Transaction,
]:
    ledger = ReferenceTransactionLedger(adapter.manifest)
    ledger_state = ledger.arm(ledger.init(), decision)
    ledger_state, authorization = ledger.authorize(
        ledger_state,
        decision,
        authorized_action=None,
        authority_id="tests.authority",
        policy_version="tests.policy.v1",
        authorization_id=f"{decision.decision_id}:authorization",
    )
    rebinding_applied = False
    if agent_state is not None:
        settled_agent_state, rebinding_applied = adapter.settle_dispatch(
            agent_state,
            authorization,
        )
        assert settled_agent_state is agent_state
    ledger_state, dispatch = ledger.settle_dispatch(
        ledger_state,
        authorization,
        rebinding_applied=rebinding_applied,
        settlement_id=f"{decision.decision_id}:settlement",
    )
    assert dispatch is not None
    ledger_state, receipt = ledger.record_dispatch(
        ledger_state,
        dispatch,
        receipt_id=f"{decision.decision_id}:receipt",
        executor_id="tests.executor",
    )
    bootstrap = (
        np.asarray([0.0, 1.0], dtype=np.float32)
        if bootstrap_observation is None
        else bootstrap_observation
    )
    boundary = terminated or truncated
    bootstrap_id = (
        f"{decision.lifecycle_id}:bootstrap:0"
        if boundary
        else f"{decision.lifecycle_id}:observation:1"
    )
    next_id = (
        None
        if boundary and not autoreset
        else (
            f"{decision.lifecycle_id}:observation:1"
            if autoreset
            else bootstrap_id
        )
    )
    next_observation = None if next_id is None else bootstrap
    ledger_state, transaction = ledger.record_outcome(
        ledger_state,
        receipt,
        reward=reward,
        discount=discount,
        terminated=terminated,
        truncated=truncated,
        autoreset=autoreset,
        bootstrap_observation_id=bootstrap_id,
        bootstrap_observation=bootstrap,
        next_decision_observation_id=next_id,
        next_decision_observation=next_observation,
    )
    return ledger, ledger_state, transaction


def _exact_transaction_for_decision(
    adapter: PrototypeReferenceAdapter,
    decision: Decision,
    *,
    next_observation_id: str,
    next_observation: np.ndarray,
    reward: float = 1.0,
) -> Transaction:
    """Build a structurally exact transaction without requiring a fresh index-zero ledger."""

    assert decision.proposed_action is not None
    authorization = DispatchAuthorization(
        decision=decision,
        status=AuthorizationStatus.EXACT,
        authorized_action=decision.proposed_action,
        authority_id="tests.authority",
        policy_version="tests.policy.v1",
        authorization_id=f"{decision.decision_id}:authorization",
    )
    dispatch = DispatchAck(
        authorization=authorization,
        status=DispatchStatus.EXACT,
        effective_action=decision.proposed_action,
        settlement_id=f"{decision.decision_id}:settlement",
    )
    receipt = DispatchReceipt(
        dispatch=dispatch,
        receipt_id=f"{decision.decision_id}:receipt",
        executor_id="tests.executor",
    )
    encoded_observation = adapter.manifest.observation_spec.encode(next_observation)
    return Transaction(
        receipt=receipt,
        reward=reward,
        discount=1.0,
        terminated=False,
        truncated=False,
        autoreset=False,
        bootstrap_observation_id=next_observation_id,
        bootstrap_observation=encoded_observation,
        next_decision_observation_id=next_observation_id,
        next_decision_observation=encoded_observation,
    )


@pytest.mark.parametrize(
    "config",
    [
        PrototypeAgentConfig(),
        PrototypeAgentConfig(
            oak=OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=(SubtaskSpec(feature_index=0),),
                    observation_dim=2,
                    n_primitive_actions=2,
                )
            )
        ),
        dataclasses.replace(_config(), dreaming=DreamingConfig()),
        dataclasses.replace(_config(), auto_curate_every=1),
    ],
)
def test_adapter_rejects_options_sidecars_and_python_curation(
    config: PrototypeAgentConfig,
) -> None:
    with pytest.raises(ValueError, match="primitive-only|unsupported|disabled"):
        PrototypeReferenceAdapter.from_config(config)


def test_manifest_binds_canonical_minimal_config_and_exact_codecs() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_config())
    manifest = adapter.manifest

    assert manifest.implementation_id == "asi.prototype_exact_adapter.preview1"
    assert manifest.state_schema == PROTOTYPE_REFERENCE_STATE_SCHEMA
    assert manifest.config["adapter_schema"] == PROTOTYPE_REFERENCE_ADAPTER_SCHEMA
    assert manifest.config["dispatch_mode"] == "exact_only"
    assert manifest.config["prototype_agent"] == _config().to_config()
    assert manifest.config["prototype_state_schema"] == "alberta.prototype_agent.v3"
    assert not manifest.capabilities.dispatch_rebinding
    assert manifest.observation_spec.kind == "box"
    assert manifest.observation_spec.shape == (2,)
    assert manifest.observation_spec.dtype == "float32"
    assert manifest.action_spec.kind == "discrete"
    assert manifest.action_spec.cardinality == 2
    assert manifest.action_spec.dtype == "int32"

    with pytest.raises(ValueError, match="dtype"):
        manifest.observation_spec.encode(np.asarray([1.0, 0.0], dtype=np.float64))
    with pytest.raises(ValueError, match="dtype"):
        manifest.action_spec.encode(np.asarray(0, dtype=np.int64))


def test_state_is_manifest_bound_even_when_other_config_has_identical_shapes() -> None:
    adapter_a = PrototypeReferenceAdapter.from_config(_config(base_step_size=0.05))
    adapter_b = PrototypeReferenceAdapter.from_config(_config(base_step_size=0.02))
    state_b = adapter_b.init(jr.key(3), lifecycle_id=_LIFE_A)

    assert isinstance(state_b, PrototypeReferenceState)
    assert state_b.schema == PROTOTYPE_REFERENCE_STATE_SCHEMA
    assert state_b.manifest_id == adapter_b.manifest.manifest_id
    assert state_b.config_sha256 == adapter_b.manifest.config_sha256
    assert state_b.lifecycle_id == _LIFE_A
    assert state_b.decision_index == 0
    assert state_b.current_observation_id is None
    assert adapter_a.manifest.manifest_id != adapter_b.manifest.manifest_id
    with pytest.raises(DecisionOwnershipError, match="manifest|config"):
        adapter_a.start(
            state_b,
            observation_id=f"{_LIFE_A}:observation:0",
            observation=np.asarray([1.0, 0.0], dtype=np.float32),
        )

    state_a, decision_a = _start(adapter_a, key_seed=7)
    state_b, _ = adapter_b.start(
        state_b,
        observation_id=f"{_LIFE_A}:observation:0",
        observation=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    _, _, transaction_a = _record_outcome(adapter_a, decision_a)
    rejected = adapter_a.apply_outcome(state_b, transaction_a)

    assert not rejected.accepted
    assert rejected.state is state_b
    assert rejected.next_decision is None
    assert rejected.rejection_reason is not None
    assert "manifest" in rejected.rejection_reason or "config" in rejected.rejection_reason
    assert state_a is not state_b


def test_relabelled_learned_state_cannot_cross_adapter_configurations() -> None:
    adapter_a = PrototypeReferenceAdapter.from_config(_config(base_step_size=0.05))
    adapter_b = PrototypeReferenceAdapter.from_config(_config(base_step_size=0.02))
    state_b, decision_b = _start(adapter_b, key_seed=3)
    _, _, transaction_b = _record_outcome(
        adapter_b,
        decision_b,
        agent_state=state_b,
        reward=4.0,
    )
    advanced_b = adapter_b.apply_outcome(state_b, transaction_b)
    assert advanced_b.accepted
    assert advanced_b.next_decision is not None

    relabelled_b = dataclasses.replace(
        advanced_b.state,
        manifest_id=adapter_a.manifest.manifest_id,
        config_sha256=adapter_a.manifest.config_sha256,
    )
    b_decision = advanced_b.next_decision
    relabelled_decision = adapter_a.manifest.make_decision(
        lifecycle_id=b_decision.lifecycle_id,
        decision_index=b_decision.decision_index,
        observation_id=b_decision.observation_id,
        observation=b_decision.observation,
        proposed_action=b_decision.proposed_action,
        armed=True,
    )
    transaction_a = _exact_transaction_for_decision(
        adapter_a,
        relabelled_decision,
        next_observation_id=f"{_LIFE_A}:observation:2",
        next_observation=np.asarray([1.0, 0.0], dtype=np.float32),
        reward=4.0,
    )

    with pytest.raises(DecisionOwnershipError, match="owner|adapter instance"):
        adapter_a.current_decision(relabelled_b)
    with pytest.raises(DecisionOwnershipError, match="owner|adapter instance"):
        adapter_a.settle_dispatch(
            relabelled_b,
            transaction_a.receipt.dispatch.authorization,
        )
    rejected = adapter_a.apply_outcome(relabelled_b, transaction_a)

    assert not rejected.accepted
    assert rejected.state is relabelled_b
    assert rejected.next_decision is None
    assert rejected.rejection_reason is not None
    assert "owner" in rejected.rejection_reason or "adapter instance" in rejected.rejection_reason


def test_equivalent_adapter_instances_do_not_share_process_local_state() -> None:
    adapter_a = PrototypeReferenceAdapter.from_config(_config())
    adapter_b = PrototypeReferenceAdapter.from_config(_config())
    state_a, decision_a = _start(adapter_a)
    assert adapter_a.manifest == adapter_b.manifest

    with pytest.raises(DecisionOwnershipError, match="owner|adapter instance"):
        adapter_b.current_decision(state_a)
    _, _, transaction = _record_outcome(adapter_b, decision_a)
    rejected = adapter_b.apply_outcome(state_a, transaction)

    assert not rejected.accepted
    assert rejected.state is state_a
    assert rejected.next_decision is None
    assert rejected.rejection_reason is not None
    assert "owner" in rejected.rejection_reason or "adapter instance" in rejected.rejection_reason


def test_adapter_and_reference_state_are_explicitly_nonportable() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_config())
    state = adapter.init(jr.key(0), lifecycle_id=_LIFE_A)

    with pytest.raises(TypeError, match="process-local|serialize|restore"):
        pickle.dumps(adapter)
    with pytest.raises(TypeError, match="process-local|serialize|restore"):
        pickle.dumps(state)


def test_lifecycle_and_internal_generation_map_exactly_to_host_decision() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_config())
    state, decision = _start(adapter)

    assert decision.lifecycle_id == _LIFE_A
    assert decision.decision_index == 0
    assert decision.observation_id == f"{_LIFE_A}:observation:0"
    assert state.lifecycle_id == decision.lifecycle_id
    assert state.decision_index == decision.decision_index
    assert state.current_observation_id == decision.observation_id
    assert decision.observation.to_numpy().dtype == np.dtype(np.float32)
    assert decision.proposed_action is not None
    assert decision.proposed_action.to_numpy().dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(
        np.asarray(state.agent_state.current_decision_id, dtype=np.uint32),
        np.asarray([1, 2, 0, 0], dtype=np.uint32),
    )

    with pytest.raises(ValueError, match="lifecycle"):
        adapter.init(jr.key(1), lifecycle_id="free-form-life")
    with pytest.raises(DecisionOwnershipError, match="index|generation"):
        adapter.current_decision(dataclasses.replace(state, decision_index=1))


def test_relabelled_decision_observation_id_is_rejected_without_learning() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_config())
    agent_state, decision = _start(adapter)
    _, _, transaction = _record_outcome(adapter, decision)
    relabelled_decision = dataclasses.replace(
        transaction.decision,
        observation_id=f"{_LIFE_A}:observation:forged",
    )
    relabelled_authorization = dataclasses.replace(
        transaction.receipt.dispatch.authorization,
        decision=relabelled_decision,
    )
    relabelled_dispatch = dataclasses.replace(
        transaction.receipt.dispatch,
        authorization=relabelled_authorization,
    )
    relabelled_receipt = dataclasses.replace(
        transaction.receipt,
        dispatch=relabelled_dispatch,
    )
    relabelled_transaction = dataclasses.replace(
        transaction,
        receipt=relabelled_receipt,
    )

    rejected = adapter.apply_outcome(agent_state, relabelled_transaction)

    assert not rejected.accepted
    assert rejected.state is agent_state
    assert rejected.next_decision is None
    assert rejected.rejection_reason is not None
    assert "observation" in rejected.rejection_reason


def test_exact_settlement_is_an_agent_state_identity_operation() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_config())
    agent_state, decision = _start(adapter)
    ledger = ReferenceTransactionLedger(adapter.manifest)
    ledger_state = ledger.arm(ledger.init(), decision)
    ledger_state, authorization = ledger.authorize(
        ledger_state,
        decision,
        authorized_action=None,
        authority_id="tests.authority",
        policy_version="tests.policy.v1",
        authorization_id=f"{decision.decision_id}:authorization",
    )

    settled_agent_state, rebinding_applied = adapter.settle_dispatch(
        agent_state,
        authorization,
    )

    assert settled_agent_state is agent_state
    assert rebinding_applied is False
    ledger_state, dispatch = ledger.settle_dispatch(
        ledger_state,
        authorization,
        rebinding_applied=rebinding_applied,
        settlement_id=f"{decision.decision_id}:settlement",
    )
    assert dispatch is not None
    assert ledger_state.phase is TransactionPhase.SETTLED


def test_replacement_and_veto_never_mutate_the_exact_only_agent() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_config())
    agent_state, decision = _start(adapter)
    assert decision.proposed_action is not None
    proposed_action = decision.proposed_action.to_python()
    assert isinstance(proposed_action, int)
    replacement = np.asarray(1 - proposed_action, dtype=np.int32)

    ledger = ReferenceTransactionLedger(adapter.manifest)
    ledger_state = ledger.arm(ledger.init(), decision)
    ledger_state, authorization = ledger.authorize(
        ledger_state,
        decision,
        authorized_action=replacement,
        authority_id="tests.authority",
        policy_version="tests.policy.v1",
        authorization_id=f"{decision.decision_id}:authorization",
    )
    assert authorization.status is AuthorizationStatus.REPLACED
    with pytest.raises(DecisionOwnershipError, match="exact|replacement|rebind"):
        adapter.settle_dispatch(agent_state, authorization)
    halted, dispatch = ledger.settle_dispatch(
        ledger_state,
        authorization,
        rebinding_applied=False,
        settlement_id=f"{decision.decision_id}:settlement",
    )
    assert dispatch is None
    assert halted.phase is TransactionPhase.HALTED

    other_ledger = ReferenceTransactionLedger(adapter.manifest)
    other_state = other_ledger.arm(other_ledger.init(), decision)
    vetoed_state, veto = other_ledger.authorize(
        other_state,
        decision,
        authorized_action=None,
        authority_id="tests.authority",
        policy_version="tests.policy.v1",
        authorization_id=f"{decision.decision_id}:authorization",
        veto_reason="test veto",
    )
    assert vetoed_state.phase is TransactionPhase.HALTED
    with pytest.raises(DecisionOwnershipError, match="exact|veto"):
        adapter.settle_dispatch(agent_state, veto)

    current = adapter.current_decision(agent_state)
    assert current == decision


def test_one_full_continuing_transaction_advances_agent_and_ledger_once() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_config())
    agent_state, decision = _start(adapter)
    ledger, ledger_state, transaction = _record_outcome(
        adapter,
        decision,
        agent_state=agent_state,
    )

    candidate = adapter.apply_outcome(agent_state, transaction)

    assert candidate.accepted
    assert candidate.rejection_reason is None
    assert candidate.next_decision is not None
    assert candidate.next_decision.decision_index == 1
    assert candidate.next_decision.observation_id == f"{_LIFE_A}:observation:1"
    assert int(candidate.state.agent_state.step_count) == 1
    assert int(agent_state.agent_state.step_count) == 0
    np.testing.assert_array_equal(
        np.asarray(candidate.state.agent_state.current_decision_id, dtype=np.uint32),
        np.asarray([1, 2, 0, 1], dtype=np.uint32),
    )
    ledger_state, step_result = ledger.accept(
        ledger_state,
        next_decision=candidate.next_decision,
        parameters_changed=candidate.parameters_changed,
    )
    assert step_result.transaction_accepted
    assert ledger_state.phase is TransactionPhase.ARMED
    assert ledger_state.next_decision_index == 1


@pytest.mark.integration
def test_switching_two_state_exact_chain_crosses_phase_boundary() -> None:
    """Exercise the real deterministic environment without claiming dispatch proof."""

    adapter = PrototypeReferenceAdapter.from_config(_config())
    environment = SwitchingTwoStateMDP(
        SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    )
    environment_state = environment.init(jr.key(7))
    adapter_state, decision = _start(
        adapter,
        observation=np.asarray(environment.observe(environment_state)),
    )
    ledger = ReferenceTransactionLedger(adapter.manifest)
    ledger_state = ledger.arm(ledger.init(), decision)
    observed_phases: list[int] = []
    observation_ids = {decision.observation_id}

    for event_index in range(3):
        phase = int(environment.phase_id(environment_state))
        observed_phases.append(phase)
        ledger_state, authorization = ledger.authorize(
            ledger_state,
            decision,
            authorized_action=None,
            authority_id="tests.authority",
            policy_version="tests.policy.v1",
            authorization_id=f"{decision.decision_id}:authorization",
        )
        settled_state, rebinding_applied = adapter.settle_dispatch(
            adapter_state,
            authorization,
        )
        assert settled_state is adapter_state
        ledger_state, dispatch = ledger.settle_dispatch(
            ledger_state,
            authorization,
            rebinding_applied=rebinding_applied,
            settlement_id=f"{decision.decision_id}:settlement",
        )
        assert dispatch is not None
        ledger_state, receipt = ledger.record_dispatch(
            ledger_state,
            dispatch,
            receipt_id=f"{decision.decision_id}:receipt",
            executor_id="tests.deterministic_environment",
        )

        # Validate the recorded effective action immediately before handing it
        # to the deterministic test environment. The host receipt remains an
        # acknowledgement, not proof of physical dispatch.
        effective_action = adapter.manifest.action_spec.encode(
            receipt.effective_action
        ).to_numpy()
        assert effective_action.shape == ()
        assert effective_action.dtype == np.dtype(np.int32)
        action_index = int(effective_action)
        environment_action = jnp.asarray(effective_action, dtype=jnp.int32)
        state_index = int(environment_state.state_index)
        expected_reward = (
            environment.config.payoffs_a
            if phase == PHASE_A
            else environment.config.payoffs_b
        )[state_index][action_index]
        next_observation, reward, candidate_environment_state = environment.step(
            environment_state,
            environment_action,
            jr.key(event_index),
        )
        assert float(reward) == expected_reward

        next_observation_id = f"{_LIFE_A}:observation:{event_index + 1}"
        assert next_observation_id not in observation_ids
        observation_ids.add(next_observation_id)
        ledger_state, transaction = ledger.record_outcome(
            ledger_state,
            receipt,
            reward=np.asarray(reward, dtype=np.float32),
            discount=np.asarray(1.0, dtype=np.float32),
            terminated=False,
            truncated=False,
            autoreset=False,
            bootstrap_observation_id=next_observation_id,
            bootstrap_observation=np.asarray(next_observation),
            next_decision_observation_id=next_observation_id,
            next_decision_observation=np.asarray(next_observation),
        )
        candidate = adapter.apply_outcome(adapter_state, transaction)
        assert candidate.accepted, candidate.rejection_reason
        assert candidate.next_decision is not None
        ledger_state, result = ledger.accept(
            ledger_state,
            next_decision=candidate.next_decision,
            parameters_changed=candidate.parameters_changed,
        )
        assert result.transaction_accepted

        environment_state = candidate_environment_state
        adapter_state = candidate.state
        decision = candidate.next_decision
        expected_count = event_index + 1
        assert int(environment_state.step_count) == expected_count
        assert int(adapter_state.agent_state.step_count) == expected_count
        assert adapter_state.decision_index == expected_count
        assert ledger_state.next_decision_index == expected_count
        assert decision.decision_index == expected_count
        assert adapter_state.current_observation_id == next_observation_id
        assert decision.observation_id == next_observation_id
        assert ledger_state.phase is TransactionPhase.ARMED

    assert observed_phases == [PHASE_A, PHASE_A, PHASE_B]
    assert len(observation_ids) == 4


def test_stale_and_cross_lifecycle_outcomes_preserve_the_supplied_state() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_config())
    initial_state, decision = _start(adapter)
    _, _, transaction = _record_outcome(adapter, decision)
    accepted = adapter.apply_outcome(initial_state, transaction)
    assert accepted.accepted

    stale = adapter.apply_outcome(accepted.state, transaction)
    assert not stale.accepted
    assert stale.state is accepted.state
    assert stale.next_decision is None
    assert stale.rejection_reason is not None

    other_state, other_decision = _start(
        adapter,
        lifecycle_id=_LIFE_B,
        key_seed=4,
    )
    _, _, other_transaction = _record_outcome(adapter, other_decision)
    mismatch = adapter.apply_outcome(initial_state, other_transaction)
    assert not mismatch.accepted
    assert mismatch.state is initial_state
    assert mismatch.next_decision is None
    chex.assert_trees_all_equal(initial_state.agent_state, mismatch.state.agent_state)
    assert other_state is not initial_state

    wrong_codec = dataclasses.replace(
        transaction.bootstrap_observation,
        semantic_id="tests.wrong_observation.preview1",
    )
    wrong_codec_transaction = dataclasses.replace(
        transaction,
        bootstrap_observation=wrong_codec,
        next_decision_observation=wrong_codec,
    )
    codec_mismatch = adapter.apply_outcome(initial_state, wrong_codec_transaction)
    assert not codec_mismatch.accepted
    assert codec_mismatch.state is initial_state
    assert codec_mismatch.next_decision is None


def test_boundary_without_a_next_decision_is_rejected_before_learning() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_config())
    agent_state, decision = _start(adapter)
    _, _, transaction = _record_outcome(
        adapter,
        decision,
        reward=0.0,
        discount=0.0,
        terminated=True,
    )

    rejected = adapter.apply_outcome(agent_state, transaction)

    assert not rejected.accepted
    assert rejected.state is agent_state
    assert rejected.next_decision is None
    assert rejected.rejection_reason is not None
    assert "boundary" in rejected.rejection_reason


def test_runtime_failure_discards_the_functional_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = PrototypeReferenceAdapter.from_config(_config())
    agent_state, decision = _start(adapter)
    _, _, transaction = _record_outcome(adapter, decision)

    def fail_update(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("synthetic Prototype update failure")

    monkeypatch.setattr(PrototypeAgent, "update_transition", fail_update)
    rejected = adapter.apply_outcome(agent_state, transaction)

    assert not rejected.accepted
    assert rejected.state is agent_state
    assert rejected.next_decision is None
    assert rejected.rejection_reason == "synthetic Prototype update failure"


def test_premature_prototype_disarm_discards_the_functional_candidate() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_config())
    agent_state, decision = _start(adapter)
    near_value = np.iinfo(np.int32).max - 1
    near_counter = jnp.asarray(near_value, dtype=jnp.int32)
    near_words = jnp.asarray([0, near_value], dtype=jnp.uint32)
    oak_state = agent_state.agent_state.oak_state
    stomp_state = oak_state.stomp_state
    base_state = stomp_state.base_learner_state.replace(
        step_count=near_counter,
        step_words=near_words,
    )
    stomp_state = stomp_state.replace(
        base_learner_state=base_state,
        step_count=near_counter,
        step_words=near_words,
    )
    oak_state = oak_state.replace(
        stomp_state=stomp_state,
        step_count=near_counter,
        step_words=near_words,
    )
    near_limit = dataclasses.replace(
        agent_state,
        agent_state=agent_state.agent_state.replace(  # type: ignore[attr-defined]
            oak_state=oak_state,
            step_count=near_counter,
        ),
    )
    _, _, transaction = _record_outcome(adapter, decision)

    rejected = adapter.apply_outcome(near_limit, transaction)

    assert not rejected.accepted
    assert rejected.state is near_limit
    assert rejected.next_decision is None
    assert rejected.rejection_reason is not None
    assert "disarmed" in rejected.rejection_reason or "capacity" in rejected.rejection_reason
    assert int(rejected.state.agent_state.step_count) == np.iinfo(np.int32).max - 1
