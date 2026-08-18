"""Failing-first contracts for the development-only aggregate reference life."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig
from alberta_framework.core.prototype_agent import PrototypeAgentConfig
from alberta_framework.prototype_reference_adapter import (
    PROTOTYPE_REFERENCE_MAX_ACCEPTED_EVENTS,
    PrototypeAdapterUpdate,
    PrototypeReferenceAdapter,
)
from alberta_framework.reference_agent import (
    AuthorizationStatus,
    DecisionOwnershipError,
    DispatchAck,
    DispatchAuthorization,
    DispatchCommand,
    DispatchStatus,
    ReferenceAgentUpdate,
    ReferenceTransactionLedger,
    ReferenceTransactionReducer,
    TransactionPhase,
)
from alberta_framework.reference_life import (
    ExactDispatchAdapter,
    HaltStage,
    LifePhase,
    RecoveryMode,
    ReferenceLifeConfig,
    ReferenceLifeMetricsAdapter,
    ReferenceLifeRunner,
    SwitchingEnvironmentExecution,
    SwitchingTwoStateReferenceEnvironment,
    build_prototype_switching_life,
)
from alberta_framework.streams.closed_loop import (
    PHASE_A,
    PHASE_B,
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
)

pytestmark = pytest.mark.unit

_LIFECYCLE_ID = "prototype.0000000100000002"


def _agent_config() -> PrototypeAgentConfig:
    return PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(),
                observation_dim=2,
                n_primitive_actions=2,
                base_step_size=0.05,
                epsilon_base=0.0,
            )
        )
    )


def _runner(
    *,
    horizon: int = 3,
    phase_length: int = 2,
) -> ReferenceLifeRunner:
    return build_prototype_switching_life(
        agent_config=_agent_config(),
        environment_config=SwitchingTwoStateConfig(  # type: ignore[call-arg]
            phase_length=phase_length
        ),
        lifecycle_id=_LIFECYCLE_ID,
        seed=7,
        max_accepted_events=horizon,
    )


def _corrupt_accepted_update(
    update: PrototypeAdapterUpdate,
    *,
    prior_state: Any,
    adapter: PrototypeReferenceAdapter,
    defect: str,
) -> PrototypeAdapterUpdate:
    assert update.accepted
    assert update.next_decision is not None
    if defect == "stale_state":
        return dataclasses.replace(update, state=prior_state)
    if defect == "mismatched_decision":
        action = update.next_decision.proposed_action
        assert action is not None
        action_value = action.to_python()
        assert isinstance(action_value, int)
        mismatched_action = adapter.manifest.action_spec.encode(
            np.asarray(1 - action_value, dtype=np.int32)
        )
        return dataclasses.replace(
            update,
            next_decision=dataclasses.replace(
                update.next_decision,
                proposed_action=mismatched_action,
            ),
        )
    raise AssertionError(f"unknown test defect: {defect}")


def test_pure_transaction_reducer_has_no_hidden_current_state() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_agent_config())
    agent_state = adapter.init(jr.key(7), lifecycle_id=_LIFECYCLE_ID)
    agent_state, decision = adapter.start(
        agent_state,
        observation_id=f"{_LIFECYCLE_ID}:observation:0",
        observation=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    reducer = ReferenceTransactionReducer(adapter.manifest)

    ready = reducer.init()
    first = reducer.arm(ready, decision)
    second = reducer.arm(ready, decision)

    assert ready.phase is TransactionPhase.READY
    assert first == second
    assert first is not second
    assert first.phase is TransactionPhase.ARMED

    ledger = ReferenceTransactionLedger(adapter.manifest)
    live_ready = ledger.init()
    ledger.arm(live_ready, decision)
    with pytest.raises(DecisionOwnershipError, match="stale|replayed|current"):
        ledger.arm(live_ready, decision)


def test_life_config_is_canonical_and_binds_complete_components() -> None:
    runner = _runner()
    config = runner.config
    payload = config.config

    assert payload["agent"]["manifest_id"] == runner.agent_adapter.manifest.manifest_id
    assert payload["agent"]["config"] == runner.agent_adapter.manifest.config
    assert payload["environment"]["config"]["phase_length"] == 2
    assert payload["dispatch"]["config"]["mode"] == "exact_only"
    assert payload["dispatch"]["config"]["authority_role"] == "declared_static_identity"
    assert payload["dispatch"]["config"]["safety_policy"] == "unimplemented"
    assert payload["dispatch"]["config"]["veto_conformance"] is False
    assert payload["metrics"]["config"]["oracle_regret"] is True
    assert payload["max_accepted_events"] == 3
    assert len(config.config_sha256) == 64
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_accepted_events = 4  # type: ignore[misc]


def test_runner_rejects_rehashed_nested_environment_descriptor_tamper() -> None:
    runner = _runner()
    payload = runner.config.config
    payload["environment"]["config"]["phase_length"] = 999
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    tampered = dataclasses.replace(
        runner.config,
        config_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        _config_json=encoded,
    )
    assert isinstance(tampered, ReferenceLifeConfig)

    with pytest.raises(ValueError, match="canonical|component|configuration"):
        ReferenceLifeRunner(
            config=tampered,
            agent_adapter=runner.agent_adapter,
            environment_adapter=runner.environment_adapter,
            dispatch_adapter=runner._dispatch_adapter,
            metrics_adapter=runner._metrics_adapter,
        )


def test_direct_runner_constructor_rejects_executor_identity_mismatch() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_agent_config())
    environment = SwitchingTwoStateReferenceEnvironment(
        SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
        observation_spec=adapter.manifest.observation_spec,
        action_spec=adapter.manifest.action_spec,
        executor_id="asi.switching_two_state.other_executor",
    )
    dispatch = ExactDispatchAdapter()
    metrics = ReferenceLifeMetricsAdapter()
    config = ReferenceLifeConfig.from_components(
        lifecycle_id=_LIFECYCLE_ID,
        seed=7,
        max_accepted_events=1,
        agent_manifest=adapter.manifest,
        agent_capacity=adapter.max_accepted_events,
        environment_manifest=environment.manifest,
        dispatch_config=dispatch.config,
        metrics_config=metrics.config,
    )

    with pytest.raises(ValueError, match="executor IDs"):
        ReferenceLifeRunner(
            config=config,
            agent_adapter=adapter,
            environment_adapter=environment,
            dispatch_adapter=dispatch,
            metrics_adapter=metrics,
        )


def test_current_state_must_match_configured_lifecycle() -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    forged = dataclasses.replace(
        initial,
        lifecycle_id="prototype.0000000100000003",
    )
    runner._current_state = forged

    with pytest.raises(DecisionOwnershipError, match="lifecycle"):
        runner.step(forged)


def test_runner_rejects_oversized_life_before_component_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = False

    def forbidden_init(self: PrototypeReferenceAdapter, *args: Any, **kwargs: Any) -> Any:
        del self, args, kwargs
        nonlocal initialized
        initialized = True
        raise AssertionError("agent init must not run")

    monkeypatch.setattr(PrototypeReferenceAdapter, "init", forbidden_init)
    at_capacity = build_prototype_switching_life(
        agent_config=_agent_config(),
        environment_config=SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
        lifecycle_id=_LIFECYCLE_ID,
        seed=7,
        max_accepted_events=PROTOTYPE_REFERENCE_MAX_ACCEPTED_EVENTS,
    )
    assert (
        at_capacity.config.max_accepted_events
        == PROTOTYPE_REFERENCE_MAX_ACCEPTED_EVENTS
    )
    with pytest.raises(ValueError, match="max_accepted_events|capacity"):
        build_prototype_switching_life(
            agent_config=_agent_config(),
            environment_config=SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
            lifecycle_id=_LIFECYCLE_ID,
            seed=7,
            max_accepted_events=PROTOTYPE_REFERENCE_MAX_ACCEPTED_EVENTS + 1,
        )
    assert initialized is False


@pytest.mark.parametrize(
    "lifecycle_id",
    (
        "other.0000000100000002",
        "prototype.000000010000000g",
    ),
)
def test_concrete_life_rejects_invalid_prototype_lifecycle_before_init(
    lifecycle_id: str,
) -> None:
    with pytest.raises(ValueError, match="Prototype lifecycle|lifecycle codec"):
        build_prototype_switching_life(
            agent_config=_agent_config(),
            environment_config=SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
            lifecycle_id=lifecycle_id,
            seed=7,
            max_accepted_events=1,
        )


def test_switching_phase_length_cannot_exceed_signed_int32_capacity() -> None:
    with pytest.raises(
        ValueError,
        match="phase_length.*(?:signed-int32|capacity|positive integer)",
    ):
        build_prototype_switching_life(
            agent_config=_agent_config(),
            environment_config=SwitchingTwoStateConfig(  # type: ignore[call-arg]
                phase_length=int(np.iinfo(np.int32).max) + 1
            ),
            lifecycle_id=_LIFECYCLE_ID,
            seed=7,
            max_accepted_events=1,
        )


def test_switching_reference_environment_behavior_is_immutable() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_agent_config())
    caller_payoffs: Any = [[0.0, 1.0], [1.0, 0.0]]
    environment = SwitchingTwoStateReferenceEnvironment(
        SwitchingTwoStateConfig(  # type: ignore[call-arg]
            phase_length=2,
            payoffs_a=caller_payoffs,
        ),
        observation_spec=adapter.manifest.observation_spec,
        action_spec=adapter.manifest.action_spec,
    )
    kernel = environment._environment
    oracle = kernel.optimal_average_reward(PHASE_A)

    caller_payoffs[0][0] = 999.0

    with pytest.raises(AttributeError, match="immutable"):
        environment._environment = SwitchingTwoStateMDP()  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        kernel._phase_length = 999  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        kernel._payoffs_np = np.zeros((2, 2, 2), dtype=np.float32)  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        kernel._payoffs_np[PHASE_A, 0, 0] = 999.0
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        kernel.config.payoffs_a = ((999.0, 999.0), (999.0, 999.0))  # type: ignore[misc]

    assert kernel.optimal_average_reward(PHASE_A) == oracle
    assert kernel.config.payoffs_a[0][0] == 0.0
    assert environment.manifest.config["phase_length"] == 2


@pytest.mark.integration
def test_authoritative_life_crosses_phase_and_completes_at_horizon() -> None:
    runner = _runner(horizon=3, phase_length=2)
    state = runner.init()
    initial_digest = state.transcript_sha256
    phases: list[int] = []

    for expected_count in range(1, 4):
        step = runner.step(state)
        assert step.accepted, step.rejection_reason
        assert step.event is not None
        phases.append(step.event.regime_id)
        assert step.event.receipt.command == step.event.command
        assert step.event.receipt.applied_action == step.event.command.effective_action
        state = step.state
        assert state.accepted_events == expected_count
        assert state.dispatch_attempts == expected_count
        assert state.executed_events == expected_count
        assert state.environment_rng_cursor == expected_count
        assert int(state.environment_state.step_count) == expected_count
        assert int(state.agent_state.agent_state.step_count) == expected_count
        assert state.transaction_state.next_decision_index == expected_count
        assert state.agent_state.current_observation_id == (
            f"{_LIFECYCLE_ID}:observation:{expected_count}"
        )

    assert phases == [PHASE_A, PHASE_A, PHASE_B]
    assert state.phase is LifePhase.COMPLETED
    assert state.pending_outcome is None
    assert state.halt is None
    assert state.transcript_sha256 != initial_digest
    assert state.metrics.accepted_events == 3
    assert state.metrics.phase_event_counts == (2, 1)
    assert state.metrics.oracle_reward_sum == pytest.approx(3.0)
    assert state.metrics.regret_sum == pytest.approx(
        state.metrics.oracle_reward_sum - state.metrics.reward_sum
    )
    with pytest.raises(DecisionOwnershipError, match="completed"):
        runner.step(state)


def test_environment_rejects_bad_action_before_clipping_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    state = runner.init()
    assert state.transaction_state.decision is not None
    decision = state.transaction_state.decision
    assert decision.proposed_action is not None
    bad_action = dataclasses.replace(
        decision.proposed_action,
        payload=np.asarray(2, dtype=np.int32).tobytes(),
    )
    forged_decision = dataclasses.replace(decision, proposed_action=bad_action)
    authorization = DispatchAuthorization(
        decision=forged_decision,
        status=AuthorizationStatus.EXACT,
        authorized_action=bad_action,
        authority_id="tests.authority",
        policy_version="tests.policy.v1",
        authorization_id=f"{decision.decision_id}:authorization",
    )
    dispatch = DispatchAck(
        authorization=authorization,
        status=DispatchStatus.EXACT,
        effective_action=bad_action,
        settlement_id=f"{decision.decision_id}:settlement",
    )
    command = DispatchCommand(
        dispatch=dispatch,
        command_id=f"{decision.decision_id}:command",
        executor_id="asi.switching_two_state.executor",
        executor_epoch="asi.switching_two_state.executor_epoch.1",
    )
    called = False

    def forbidden_step(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        nonlocal called
        called = True
        raise AssertionError("clipping environment must not receive invalid action")

    monkeypatch.setattr(SwitchingTwoStateMDP, "step", forbidden_step)
    with pytest.raises(ValueError, match="cardinality|outside|range"):
        runner.environment_adapter.execute(
            state.environment_state,
            command,
            key=jnp.asarray([0, 1], dtype=jnp.uint32),
        )
    assert called is False


def test_post_execution_agent_rejection_halts_then_recovers_without_redispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    original_apply = PrototypeReferenceAdapter.apply_outcome
    original_execute = SwitchingTwoStateReferenceEnvironment.execute
    apply_calls = 0
    execute_calls = 0

    def reject_once(
        self: PrototypeReferenceAdapter,
        state: Any,
        transaction: Any,
    ) -> PrototypeAdapterUpdate:
        nonlocal apply_calls
        apply_calls += 1
        if apply_calls == 1:
            return PrototypeAdapterUpdate(
                state=state,
                next_decision=None,
                accepted=False,
                parameters_changed=False,
                rejection_reason="synthetic post-execution rejection",
            )
        return original_apply(self, state, transaction)

    def count_execute(
        self: SwitchingTwoStateReferenceEnvironment,
        state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> SwitchingEnvironmentExecution:
        nonlocal execute_calls
        execute_calls += 1
        return original_execute(self, state, command, key=key)

    monkeypatch.setattr(PrototypeReferenceAdapter, "apply_outcome", reject_once)
    monkeypatch.setattr(SwitchingTwoStateReferenceEnvironment, "execute", count_execute)

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.phase is LifePhase.HALTED
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.OUTCOME_PENDING
    assert halted.halt.recovery_mode is RecoveryMode.RETRY_OUTCOME
    assert halted.pending_outcome is not None
    assert halted.accepted_events == 0
    assert halted.dispatch_attempts == 1
    assert halted.executed_events == 1
    assert int(halted.environment_state.step_count) == 1
    assert int(halted.agent_state.agent_state.step_count) == 0
    assert halted.metrics.accepted_events == 0
    assert execute_calls == 1

    recovered = runner.recover_pending_outcome(halted)
    assert recovered.accepted, recovered.rejection_reason
    assert recovered.state.phase is LifePhase.COMPLETED
    assert recovered.state.accepted_events == 1
    assert int(recovered.state.agent_state.agent_state.step_count) == 1
    assert recovered.state.metrics.accepted_events == 1
    assert recovered.state.pending_outcome is None
    assert execute_calls == 1
    assert apply_calls == 2
    with pytest.raises(DecisionOwnershipError, match="stale|current"):
        runner.recover_pending_outcome(halted)


@pytest.mark.parametrize("recover", (False, True))
def test_runner_accepts_generic_agent_updates_in_step_and_recovery(
    monkeypatch: pytest.MonkeyPatch,
    recover: bool,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    original_apply = PrototypeReferenceAdapter.apply_outcome
    apply_calls = 0

    def generic_update(
        self: PrototypeReferenceAdapter,
        state: Any,
        transaction: Any,
    ) -> ReferenceAgentUpdate:
        nonlocal apply_calls
        apply_calls += 1
        if recover and apply_calls == 1:
            return ReferenceAgentUpdate(
                state=state,
                next_decision=None,
                accepted=False,
                parameters_changed=False,
                rejection_reason="synthetic generic rejection",
            )
        update = original_apply(self, state, transaction)
        return ReferenceAgentUpdate(
            state=update.state,
            next_decision=update.next_decision,
            accepted=update.accepted,
            parameters_changed=update.parameters_changed,
            rejection_reason=update.rejection_reason,
        )

    monkeypatch.setattr(PrototypeReferenceAdapter, "apply_outcome", generic_update)

    first = runner.step(initial)
    if recover:
        assert not first.accepted
        assert first.state.pending_outcome is not None
        result = runner.recover_pending_outcome(first.state)
    else:
        result = first

    assert result.accepted, result.rejection_reason
    assert result.state.phase is LifePhase.COMPLETED
    assert result.state.accepted_events == 1
    assert int(result.state.agent_state.agent_state.step_count) == 1
    assert apply_calls == (2 if recover else 1)


def test_generic_agent_update_does_not_bypass_concrete_adapter_state_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    original_apply = PrototypeReferenceAdapter.apply_outcome
    metric_calls = 0

    def foreign_state_update(
        self: PrototypeReferenceAdapter,
        state: Any,
        transaction: Any,
    ) -> ReferenceAgentUpdate:
        update = original_apply(self, state, transaction)
        assert update.accepted
        return ReferenceAgentUpdate(
            state=object(),
            next_decision=update.next_decision,
            accepted=True,
            parameters_changed=update.parameters_changed,
            rejection_reason=None,
        )

    def forbidden_metric(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        nonlocal metric_calls
        metric_calls += 1
        raise AssertionError("foreign generic state must not reach metrics")

    monkeypatch.setattr(PrototypeReferenceAdapter, "apply_outcome", foreign_state_update)
    monkeypatch.setattr(ReferenceLifeMetricsAdapter, "observe", forbidden_metric)

    rejected = runner.step(initial)

    assert not rejected.accepted
    assert rejected.state.phase is LifePhase.HALTED
    assert rejected.state.pending_outcome is not None
    assert rejected.state.agent_state is initial.agent_state
    assert rejected.state.halt is not None
    assert "PrototypeReferenceState" in rejected.state.halt.reason
    assert metric_calls == 0


def test_generic_agent_adapter_cannot_enter_prototype_checkpoint_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    monkeypatch.setattr(runner, "_agent_adapter", object())

    with pytest.raises(DecisionOwnershipError, match="Prototype adapter"):
        runner.validate_checkpoint_state(initial)


@pytest.mark.parametrize("defect", ("stale_state", "mismatched_decision"))
def test_invalid_accepted_agent_update_retains_outcome_before_metrics(
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    original_apply = PrototypeReferenceAdapter.apply_outcome
    metric_calls = 0

    def corrupt_update(
        self: PrototypeReferenceAdapter,
        state: Any,
        transaction: Any,
    ) -> PrototypeAdapterUpdate:
        update = original_apply(self, state, transaction)
        return _corrupt_accepted_update(
            update,
            prior_state=state,
            adapter=self,
            defect=defect,
        )

    def forbidden_metric(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        nonlocal metric_calls
        metric_calls += 1
        raise AssertionError("invalid accepted agent state must not reach metrics")

    monkeypatch.setattr(PrototypeReferenceAdapter, "apply_outcome", corrupt_update)
    monkeypatch.setattr(ReferenceLifeMetricsAdapter, "observe", forbidden_metric)

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.phase is LifePhase.HALTED
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.OUTCOME_PENDING
    assert halted.halt.recovery_mode is RecoveryMode.RETRY_OUTCOME
    assert "accepted agent update" in halted.halt.reason
    assert halted.pending_outcome is not None
    assert halted.accepted_events == 0
    assert halted.executed_events == 1
    assert halted.agent_state is initial.agent_state
    assert halted.metrics is initial.metrics
    assert metric_calls == 0


@pytest.mark.parametrize("defect", ("stale_state", "mismatched_decision"))
def test_invalid_accepted_agent_recovery_retains_pending_outcome_before_metrics(
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    original_apply = PrototypeReferenceAdapter.apply_outcome
    apply_calls = 0
    metric_calls = 0

    def reject_then_corrupt(
        self: PrototypeReferenceAdapter,
        state: Any,
        transaction: Any,
    ) -> PrototypeAdapterUpdate:
        nonlocal apply_calls
        apply_calls += 1
        if apply_calls == 1:
            return PrototypeAdapterUpdate(
                state=state,
                next_decision=None,
                accepted=False,
                parameters_changed=False,
                rejection_reason="synthetic initial rejection",
            )
        update = original_apply(self, state, transaction)
        return _corrupt_accepted_update(
            update,
            prior_state=state,
            adapter=self,
            defect=defect,
        )

    def forbidden_metric(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        nonlocal metric_calls
        metric_calls += 1
        raise AssertionError("invalid recovered agent state must not reach metrics")

    monkeypatch.setattr(PrototypeReferenceAdapter, "apply_outcome", reject_then_corrupt)
    monkeypatch.setattr(ReferenceLifeMetricsAdapter, "observe", forbidden_metric)

    first = runner.step(initial)
    pending = first.state
    assert pending.halt is not None
    assert pending.halt.stage is HaltStage.OUTCOME_PENDING
    assert pending.pending_outcome is not None

    rejected = runner.recover_pending_outcome(pending)
    retained = rejected.state
    assert not rejected.accepted
    assert retained.phase is LifePhase.HALTED
    assert retained.halt is not None
    assert retained.halt.stage is HaltStage.OUTCOME_PENDING
    assert retained.halt.recovery_mode is RecoveryMode.RETRY_OUTCOME
    assert retained.halt.recovery_attempts == pending.halt.recovery_attempts + 1
    assert "accepted agent update" in retained.halt.reason
    assert retained.pending_outcome == pending.pending_outcome
    assert retained.accepted_events == 0
    assert retained.executed_events == 1
    assert retained.agent_state is pending.agent_state
    assert retained.metrics is pending.metrics
    assert apply_calls == 2
    assert metric_calls == 0
    with pytest.raises(DecisionOwnershipError, match="stale|current"):
        runner.recover_pending_outcome(pending)


def test_metric_staging_failure_recovers_without_rolling_back_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    original_observe = ReferenceLifeMetricsAdapter.observe
    original_execute = SwitchingTwoStateReferenceEnvironment.execute
    observe_calls = 0
    execute_calls = 0

    def reject_metric_once(
        self: ReferenceLifeMetricsAdapter,
        state: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal observe_calls
        observe_calls += 1
        if observe_calls == 1:
            raise ValueError("synthetic metric staging failure")
        return original_observe(self, state, **kwargs)

    def count_execute(
        self: SwitchingTwoStateReferenceEnvironment,
        state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> SwitchingEnvironmentExecution:
        nonlocal execute_calls
        execute_calls += 1
        return original_execute(self, state, command, key=key)

    monkeypatch.setattr(ReferenceLifeMetricsAdapter, "observe", reject_metric_once)
    monkeypatch.setattr(SwitchingTwoStateReferenceEnvironment, "execute", count_execute)

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.OUTCOME_PENDING
    assert halted.pending_outcome is not None
    assert "metric update" in halted.halt.reason
    assert halted.accepted_events == 0
    assert halted.executed_events == 1
    assert int(halted.environment_state.step_count) == 1
    assert int(halted.agent_state.agent_state.step_count) == 0
    assert halted.metrics.accepted_events == 0

    recovered = runner.recover_pending_outcome(halted)
    assert recovered.accepted
    assert recovered.state.phase is LifePhase.COMPLETED
    assert int(recovered.state.agent_state.agent_state.step_count) == 1
    assert recovered.state.metrics.accepted_events == 1
    assert execute_calls == 1
    assert observe_calls == 2


def test_applied_action_mismatch_halts_without_learning() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_agent_config())
    environment = _MismatchingEnvironment(
        SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
        observation_spec=adapter.manifest.observation_spec,
        action_spec=adapter.manifest.action_spec,
    )
    runner = ReferenceLifeRunner.create(
        agent_adapter=adapter,
        environment_adapter=environment,
        lifecycle_id=_LIFECYCLE_ID,
        seed=7,
        max_accepted_events=1,
    )
    initial = runner.init()

    rejected = runner.step(initial)

    assert not rejected.accepted
    assert rejected.state.phase is LifePhase.HALTED
    assert rejected.state.halt is not None
    assert rejected.state.halt.stage is HaltStage.POST_EXECUTION_DIVERGENCE
    assert rejected.state.halt.recovery_mode is RecoveryMode.RECONCILE_ONLY
    assert rejected.state.accepted_events == 0
    assert rejected.state.executed_events == 1
    assert int(rejected.state.environment_state.step_count) == 1
    assert int(rejected.state.agent_state.agent_state.step_count) == 0
    assert rejected.state.metrics.accepted_events == 0
    assert rejected.state.transaction_state.receipt is not None
    command = rejected.state.transaction_state.command
    assert command is not None
    assert rejected.state.transaction_state.receipt.applied_action != command.effective_action
    with pytest.raises(DecisionOwnershipError, match="complete outcome-pending"):
        runner.recover_pending_outcome(rejected.state)


class _MismatchingEnvironment(SwitchingTwoStateReferenceEnvironment):
    def execute(
        self,
        state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> SwitchingEnvironmentExecution:
        commanded_value = command.effective_action.to_python()
        assert isinstance(commanded_value, int)
        applied_value = 1 - commanded_value
        phase = int(self._environment.phase_id(state))
        next_observation, reward, next_state = self._environment.step(
            state,
            jnp.asarray(applied_value, dtype=jnp.int32),
            key,
        )
        replacement = self.manifest.action_spec.encode(
            np.asarray(applied_value, dtype=np.int32)
        )
        return SwitchingEnvironmentExecution(
            command=command,
            state=next_state,
            applied_action=replacement,
            next_observation=self.manifest.observation_spec.encode(
                np.asarray(next_observation, dtype=np.float32)
            ),
            reward=float(np.asarray(reward, dtype=np.float32)),
            discount=1.0,
            terminated=False,
            truncated=False,
            autoreset=False,
            regime_id=phase,
            oracle_reward=self._environment.optimal_average_reward(phase),
        )


def test_timeout_after_executor_submission_commits_uncertain_halt_and_stales_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    original_execute = SwitchingTwoStateReferenceEnvironment.execute
    execute_calls = 0

    def execute_then_timeout(
        self: SwitchingTwoStateReferenceEnvironment,
        state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> SwitchingEnvironmentExecution:
        nonlocal execute_calls
        execute_calls += 1
        original_execute(self, state, command, key=key)
        raise TimeoutError("synthetic lost executor response")

    monkeypatch.setattr(
        SwitchingTwoStateReferenceEnvironment,
        "execute",
        execute_then_timeout,
    )

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.DISPATCH_UNCERTAIN
    assert halted.halt.recovery_mode is RecoveryMode.RECONCILE_ONLY
    assert halted.dispatch_attempts == 1
    assert halted.executed_events == 0
    assert int(halted.environment_state.step_count) == 0
    assert halted.transaction_state.command is not None
    assert halted.transaction_state.receipt is None
    with pytest.raises(DecisionOwnershipError, match="stale|current"):
        runner.step(initial)
    assert execute_calls == 1


def test_post_execution_receipt_record_failure_retains_execution_and_stales_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    execute_calls = 0
    original_execute = SwitchingTwoStateReferenceEnvironment.execute

    def count_execute(
        self: SwitchingTwoStateReferenceEnvironment,
        state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> SwitchingEnvironmentExecution:
        nonlocal execute_calls
        execute_calls += 1
        return original_execute(self, state, command, key=key)

    def fail_record_dispatch(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("synthetic receipt persistence failure")

    monkeypatch.setattr(SwitchingTwoStateReferenceEnvironment, "execute", count_execute)
    monkeypatch.setattr(
        ReferenceTransactionReducer,
        "record_dispatch",
        fail_record_dispatch,
    )

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.POST_EXECUTION_DIVERGENCE
    assert halted.halt.recovery_mode is RecoveryMode.RECONCILE_ONLY
    assert halted.dispatch_attempts == 1
    assert halted.executed_events == 1
    assert int(halted.environment_state.step_count) == 1
    assert int(halted.agent_state.agent_state.step_count) == 0
    assert halted.metrics.accepted_events == 0
    assert halted.transaction_state.receipt is not None
    with pytest.raises(DecisionOwnershipError, match="stale|current"):
        runner.step(initial)
    assert execute_calls == 1


def test_outcome_record_failure_halts_with_advanced_environment_and_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()

    def fail_record_outcome(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ValueError("synthetic outcome record failure")

    monkeypatch.setattr(
        ReferenceTransactionReducer,
        "record_outcome",
        fail_record_outcome,
    )

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.POST_EXECUTION_DIVERGENCE
    assert halted.executed_events == 1
    assert int(halted.environment_state.step_count) == 1
    assert halted.transaction_state.receipt is not None
    assert halted.transaction_state.transaction is None
    with pytest.raises(DecisionOwnershipError, match="stale|current"):
        runner.step(initial)


def test_post_execution_accept_failure_discards_staged_learning_and_never_redispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    original_execute = SwitchingTwoStateReferenceEnvironment.execute
    execute_calls = 0

    def count_execute(
        self: SwitchingTwoStateReferenceEnvironment,
        state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> SwitchingEnvironmentExecution:
        nonlocal execute_calls
        execute_calls += 1
        return original_execute(self, state, command, key=key)

    def fail_accept(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("synthetic transaction acceptance failure")

    monkeypatch.setattr(SwitchingTwoStateReferenceEnvironment, "execute", count_execute)
    monkeypatch.setattr(ReferenceTransactionReducer, "accept", fail_accept)

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.POST_EXECUTION_DIVERGENCE
    assert halted.executed_events == 1
    assert int(halted.environment_state.step_count) == 1
    assert int(halted.agent_state.agent_state.step_count) == 0
    assert halted.metrics.accepted_events == 0
    assert halted.transaction_state.transaction is not None
    with pytest.raises(DecisionOwnershipError, match="stale|current"):
        runner.step(initial)
    assert execute_calls == 1


def test_post_accept_candidate_failure_halts_from_retained_outcome_and_stales_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    original_execute = SwitchingTwoStateReferenceEnvironment.execute
    execute_calls = 0

    def count_execute(
        self: SwitchingTwoStateReferenceEnvironment,
        state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> SwitchingEnvironmentExecution:
        nonlocal execute_calls
        execute_calls += 1
        return original_execute(self, state, command, key=key)

    def unchanged_metrics(
        self: ReferenceLifeMetricsAdapter,
        state: Any,
        **kwargs: Any,
    ) -> Any:
        del self, kwargs
        return state

    monkeypatch.setattr(SwitchingTwoStateReferenceEnvironment, "execute", count_execute)
    monkeypatch.setattr(ReferenceLifeMetricsAdapter, "observe", unchanged_metrics)

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.POST_EXECUTION_DIVERGENCE
    assert halted.transaction_state.transaction is not None
    assert halted.executed_events == 1
    assert int(halted.environment_state.step_count) == 1
    assert int(halted.agent_state.agent_state.step_count) == 0
    assert halted.metrics.accepted_events == 0
    with pytest.raises(DecisionOwnershipError, match="stale|current"):
        runner.step(initial)
    assert execute_calls == 1


def test_double_fault_in_receipt_record_and_normal_halt_uses_emergency_cas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    original_execute = SwitchingTwoStateReferenceEnvironment.execute
    execute_calls = 0

    def count_execute(
        self: SwitchingTwoStateReferenceEnvironment,
        state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> SwitchingEnvironmentExecution:
        nonlocal execute_calls
        execute_calls += 1
        return original_execute(self, state, command, key=key)

    def fail(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("synthetic double fault")

    monkeypatch.setattr(SwitchingTwoStateReferenceEnvironment, "execute", count_execute)
    monkeypatch.setattr(ReferenceTransactionReducer, "record_dispatch", fail)
    monkeypatch.setattr(ReferenceTransactionReducer, "halt_after_execution", fail)

    rejected = runner.step(initial)
    assert not rejected.accepted
    assert rejected.state.phase is LifePhase.HALTED
    assert rejected.state.executed_events == 1
    assert int(rejected.state.environment_state.step_count) == 1
    with pytest.raises(DecisionOwnershipError, match="stale|current"):
        runner.step(initial)
    assert execute_calls == 1


def test_malformed_dispatch_issued_state_halts_before_environment_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    execute_calls = 0

    def malformed_issued(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return object()

    def forbidden_execute(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        nonlocal execute_calls
        execute_calls += 1
        raise AssertionError("malformed dispatch state must prevent execution")

    monkeypatch.setattr(ExactDispatchAdapter, "issued", malformed_issued)
    monkeypatch.setattr(
        SwitchingTwoStateReferenceEnvironment,
        "execute",
        forbidden_execute,
    )

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.phase is LifePhase.HALTED
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.DISPATCH_UNCERTAIN
    assert halted.executed_events == 0
    assert int(halted.environment_state.step_count) == 0
    with pytest.raises(DecisionOwnershipError, match="stale|current"):
        runner.step(initial)
    assert execute_calls == 0


def test_malformed_transaction_issued_state_halts_before_environment_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    original_issue = ReferenceTransactionReducer.issue_dispatch
    execute_calls = 0

    def malformed_issue(
        self: ReferenceTransactionReducer,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        _, command = original_issue(self, *args, **kwargs)
        return object(), command

    def forbidden_execute(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        nonlocal execute_calls
        execute_calls += 1
        raise AssertionError("malformed issued transaction must prevent execution")

    monkeypatch.setattr(ReferenceTransactionReducer, "issue_dispatch", malformed_issue)
    monkeypatch.setattr(
        SwitchingTwoStateReferenceEnvironment,
        "execute",
        forbidden_execute,
    )

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.phase is LifePhase.HALTED
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.PRE_DISPATCH
    assert halted.dispatch_attempts == 0
    assert halted.executed_events == 0
    assert int(halted.environment_state.step_count) == 0
    with pytest.raises(DecisionOwnershipError, match="stale|current"):
        runner.step(initial)
    assert execute_calls == 0


def test_malformed_dispatch_receipted_state_retains_one_execution_and_stales_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    original_execute = SwitchingTwoStateReferenceEnvironment.execute
    execute_calls = 0

    def count_execute(
        self: SwitchingTwoStateReferenceEnvironment,
        state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> SwitchingEnvironmentExecution:
        nonlocal execute_calls
        execute_calls += 1
        return original_execute(self, state, command, key=key)

    def malformed_receipted(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return object()

    monkeypatch.setattr(SwitchingTwoStateReferenceEnvironment, "execute", count_execute)
    monkeypatch.setattr(ExactDispatchAdapter, "receipted", malformed_receipted)

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.phase is LifePhase.HALTED
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.POST_EXECUTION_DIVERGENCE
    assert halted.executed_events == 1
    assert int(halted.environment_state.step_count) == 1
    assert halted.dispatch_state.commands_issued == 1
    assert halted.dispatch_state.receipts_recorded == 0
    with pytest.raises(DecisionOwnershipError, match="stale|current"):
        runner.step(initial)
    assert execute_calls == 1


class _ForeignCommandEnvironment(SwitchingTwoStateReferenceEnvironment):
    def execute(
        self,
        state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> SwitchingEnvironmentExecution:
        result = super().execute(state, command, key=key)
        foreign = dataclasses.replace(
            command,
            executor_epoch="asi.switching_two_state.executor_epoch.2",
        )
        return dataclasses.replace(result, command=foreign)


def test_foreign_command_result_cannot_acknowledge_or_advance_local_execution() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_agent_config())
    environment = _ForeignCommandEnvironment(
        SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
        observation_spec=adapter.manifest.observation_spec,
        action_spec=adapter.manifest.action_spec,
    )
    runner = ReferenceLifeRunner.create(
        agent_adapter=adapter,
        environment_adapter=environment,
        lifecycle_id=_LIFECYCLE_ID,
        seed=7,
        max_accepted_events=1,
    )
    initial = runner.init()

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.DISPATCH_UNCERTAIN
    assert halted.executed_events == 0
    assert int(halted.environment_state.step_count) == 0
    assert halted.transaction_state.receipt is None
    assert halted.transcript_sha256 == initial.transcript_sha256


class _FalseObservationEnvironment(SwitchingTwoStateReferenceEnvironment):
    def execute(
        self,
        state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> SwitchingEnvironmentExecution:
        result = super().execute(state, command, key=key)
        observation = result.next_observation.to_numpy()[::-1].copy()
        forged = self.manifest.observation_spec.encode(observation)
        return dataclasses.replace(result, next_observation=forged)


def test_semantically_inconsistent_execution_result_never_reaches_learning() -> None:
    adapter = PrototypeReferenceAdapter.from_config(_agent_config())
    environment = _FalseObservationEnvironment(
        SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
        observation_spec=adapter.manifest.observation_spec,
        action_spec=adapter.manifest.action_spec,
    )
    runner = ReferenceLifeRunner.create(
        agent_adapter=adapter,
        environment_adapter=environment,
        lifecycle_id=_LIFECYCLE_ID,
        seed=7,
        max_accepted_events=1,
    )
    initial = runner.init()

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.DISPATCH_UNCERTAIN
    assert halted.executed_events == 0
    assert int(halted.environment_state.step_count) == 0
    assert int(halted.agent_state.agent_state.step_count) == 0
    assert halted.metrics.accepted_events == 0
    assert halted.transcript_sha256 == initial.transcript_sha256


def test_malformed_executor_result_commits_uncertain_halt_and_never_redispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    execute_calls = 0

    def malformed_result(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        nonlocal execute_calls
        execute_calls += 1
        return object()

    monkeypatch.setattr(
        SwitchingTwoStateReferenceEnvironment,
        "execute",
        malformed_result,
    )

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.DISPATCH_UNCERTAIN
    assert halted.executed_events == 0
    assert int(halted.environment_state.step_count) == 0
    with pytest.raises(DecisionOwnershipError, match="stale|current"):
        runner.step(initial)
    assert execute_calls == 1


def test_synthetic_safety_veto_executes_no_command_and_cannot_recover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    original_authorize = ReferenceTransactionReducer.authorize
    execute_calls = 0

    def veto(
        self: ReferenceTransactionReducer,
        state: Any,
        decision: Any,
        *,
        authorized_action: Any,
        authority_id: str,
        policy_version: str,
        authorization_id: str,
        veto_reason: str | None = None,
    ) -> Any:
        del authorized_action, veto_reason
        return original_authorize(
            self,
            state,
            decision,
            authorized_action=None,
            authority_id=authority_id,
            policy_version=policy_version,
            authorization_id=authorization_id,
            veto_reason="synthetic safety veto",
        )

    def forbidden_execute(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        nonlocal execute_calls
        execute_calls += 1
        raise AssertionError("a vetoed decision must not execute")

    monkeypatch.setattr(ReferenceTransactionReducer, "authorize", veto)
    monkeypatch.setattr(
        SwitchingTwoStateReferenceEnvironment,
        "execute",
        forbidden_execute,
    )

    rejected = runner.step(initial)
    halted = rejected.state
    assert not rejected.accepted
    assert halted.halt is not None
    assert halted.halt.stage is HaltStage.PRE_DISPATCH
    assert halted.accepted_events == 0
    assert halted.executed_events == 0
    assert halted.dispatch_attempts == 0
    assert halted.transaction_state.authorization is not None
    assert halted.transaction_state.authorization.status is AuthorizationStatus.VETOED
    assert execute_calls == 0
    with pytest.raises(DecisionOwnershipError, match="complete outcome-pending"):
        runner.recover_pending_outcome(halted)


def test_recovery_refuses_incomplete_current_state() -> None:
    runner = _runner(horizon=1)
    initial = runner.init()

    with pytest.raises(DecisionOwnershipError, match="complete outcome-pending"):
        runner.recover_pending_outcome(initial)


def test_command_precedes_execution_and_receipt_and_horizon_has_no_extra_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    initial = runner.init()
    original_issue = ReferenceTransactionReducer.issue_dispatch
    original_record = ReferenceTransactionReducer.record_dispatch
    original_execute = SwitchingTwoStateReferenceEnvironment.execute
    order: list[str] = []
    execute_calls = 0

    def issue(self: ReferenceTransactionReducer, *args: Any, **kwargs: Any) -> Any:
        result = original_issue(self, *args, **kwargs)
        order.append("command")
        return result

    def execute(
        self: SwitchingTwoStateReferenceEnvironment,
        *args: Any,
        **kwargs: Any,
    ) -> SwitchingEnvironmentExecution:
        nonlocal execute_calls
        assert order == ["command"]
        order.append("execute")
        execute_calls += 1
        return original_execute(self, *args, **kwargs)

    def record(self: ReferenceTransactionReducer, *args: Any, **kwargs: Any) -> Any:
        assert order == ["command", "execute"]
        order.append("receipt")
        return original_record(self, *args, **kwargs)

    monkeypatch.setattr(ReferenceTransactionReducer, "issue_dispatch", issue)
    monkeypatch.setattr(SwitchingTwoStateReferenceEnvironment, "execute", execute)
    monkeypatch.setattr(ReferenceTransactionReducer, "record_dispatch", record)

    completed = runner.step(initial).state
    assert completed.phase is LifePhase.COMPLETED
    assert order == ["command", "execute", "receipt"]
    with pytest.raises(DecisionOwnershipError, match="completed"):
        runner.step(completed)
    assert execute_calls == 1


def test_outer_runner_cas_prevents_concurrent_double_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=2)
    initial = runner.init()
    barrier = threading.Barrier(2)
    original_execute = SwitchingTwoStateReferenceEnvironment.execute
    calls_lock = threading.Lock()
    execute_calls = 0

    def count_execute(
        self: SwitchingTwoStateReferenceEnvironment,
        state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> SwitchingEnvironmentExecution:
        nonlocal execute_calls
        with calls_lock:
            execute_calls += 1
        return original_execute(self, state, command, key=key)

    monkeypatch.setattr(SwitchingTwoStateReferenceEnvironment, "execute", count_execute)

    def race() -> Any:
        barrier.wait()
        return runner.step(initial)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(race) for _ in range(2)]
        successes = []
        failures = []
        for future in futures:
            try:
                successes.append(future.result())
            except DecisionOwnershipError as exc:
                failures.append(exc)

    assert len(successes) == 1
    assert len(failures) == 1
    assert successes[0].state.dispatch_attempts == 1
    assert int(successes[0].state.environment_state.step_count) == 1
    assert execute_calls == 1


def test_abort_is_terminal_and_preserves_pending_execution() -> None:
    runner = _runner(horizon=1)
    state = runner.init()
    aborted = runner.abort(state, reason="operator stop")

    assert aborted.phase is LifePhase.ABORTED
    assert aborted.accepted_events == 0
    assert aborted.halt is not None
    assert aborted.halt.recovery_mode is RecoveryMode.ABORT_ONLY
    with pytest.raises(DecisionOwnershipError, match="aborted"):
        runner.step(aborted)
