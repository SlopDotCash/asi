"""Failing-first contracts for the stochastic RiverSwim reference life."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig
from alberta_framework.core.prototype_agent import PrototypeAgentConfig
from alberta_framework.reference_agent import (
    AuthorizationStatus,
    DecisionOwnershipError,
    DispatchAck,
    DispatchAuthorization,
    DispatchCommand,
    DispatchStatus,
)
from alberta_framework.reference_life import (
    RIVERSWIM_REFERENCE_MAX_STATES,
    HaltStage,
    LifePhase,
    ReferenceEnvironmentExecution,
    RiverSwimReferenceEnvironment,
    build_prototype_riverswim_life,
)
from alberta_framework.streams.closed_loop import RiverSwimConfig, RiverSwimMDP

pytestmark = pytest.mark.unit

_LIFECYCLE_ID = "prototype.0000002100000022"


def _agent_config(*, epsilon: float = 0.25, observation_dim: int = 3) -> PrototypeAgentConfig:
    return PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(),
                observation_dim=observation_dim,
                n_primitive_actions=2,
                base_step_size=0.05,
                epsilon_base=epsilon,
            )
        )
    )


def _runner(*, horizon: int = 5, seed: int = 41) -> Any:
    return build_prototype_riverswim_life(
        agent_config=_agent_config(),
        environment_config=RiverSwimConfig(  # type: ignore[call-arg]
            n_states=3,
            p_right_up=0.5,
            p_right_down=0.25,
            reward_left=0.01,
            reward_right=1.0,
            initial_state=0,
        ),
        lifecycle_id=_LIFECYCLE_ID,
        seed=seed,
        max_accepted_events=horizon,
    )


def _right_command(runner: Any, state: Any) -> DispatchCommand:
    decision = state.transaction_state.decision
    assert decision is not None
    right = runner.agent_adapter.manifest.action_spec.encode(np.asarray(1, dtype=np.int32))
    decision = dataclasses.replace(decision, proposed_action=right)
    authorization = DispatchAuthorization(
        decision=decision,
        status=AuthorizationStatus.EXACT,
        authorized_action=right,
        authority_id="tests.riverswim.authority",
        policy_version="tests.riverswim.policy.v1",
        authorization_id=f"{decision.decision_id}:authorization",
    )
    dispatch = DispatchAck(
        authorization=authorization,
        status=DispatchStatus.EXACT,
        effective_action=right,
        settlement_id=f"{decision.decision_id}:settlement",
    )
    return DispatchCommand(
        dispatch=dispatch,
        command_id=f"{decision.decision_id}:command",
        executor_id=runner.environment_adapter.executor_id,
        executor_epoch=runner.environment_adapter.executor_epoch,
    )


def test_riverswim_execute_and_validation_receive_the_identical_derived_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1)
    state = runner.init()
    observed: dict[str, tuple[str, bytes]] = {}
    original_execute = RiverSwimReferenceEnvironment.execute
    original_validate = RiverSwimReferenceEnvironment.validate_execution

    def key_bytes(key: Any) -> tuple[str, bytes]:
        return str(jr.key_impl(key)), np.asarray(jr.key_data(key)).tobytes(order="C")

    def execute(
        self: RiverSwimReferenceEnvironment,
        environment_state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> ReferenceEnvironmentExecution:
        observed["execute"] = key_bytes(key)
        return original_execute(self, environment_state, command, key=key)

    def validate(
        self: RiverSwimReferenceEnvironment,
        previous_state: Any,
        command: DispatchCommand,
        execution: ReferenceEnvironmentExecution,
        *,
        key: Any,
    ) -> None:
        observed["validate"] = key_bytes(key)
        original_validate(self, previous_state, command, execution, key=key)

    monkeypatch.setattr(RiverSwimReferenceEnvironment, "execute", execute)
    monkeypatch.setattr(RiverSwimReferenceEnvironment, "validate_execution", validate)

    step = runner.step(state)
    assert step.accepted, step.rejection_reason
    assert observed["execute"] == observed["validate"]
    assert observed["execute"] == key_bytes(runner._environment_key(0))


def test_riverswim_state_cap_rejects_before_exponential_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle_called = False

    def forbidden_oracle(self: RiverSwimMDP) -> float:
        del self
        nonlocal oracle_called
        oracle_called = True
        raise AssertionError("oversized chain must not enter 2**n oracle enumeration")

    monkeypatch.setattr(RiverSwimMDP, "optimal_average_reward", forbidden_oracle)
    with pytest.raises(ValueError, match=r"n_states|2\*\*n|oracle"):
        build_prototype_riverswim_life(
            agent_config=_agent_config(
                observation_dim=RIVERSWIM_REFERENCE_MAX_STATES + 1
            ),
            environment_config=RiverSwimConfig(  # type: ignore[call-arg]
                n_states=RIVERSWIM_REFERENCE_MAX_STATES + 1
            ),
            lifecycle_id=_LIFECYCLE_ID,
            seed=41,
            max_accepted_events=1,
        )
    assert oracle_called is False


def test_riverswim_reference_environment_behavior_is_immutable() -> None:
    runner = _runner(horizon=1)
    environment = runner.environment_adapter
    assert isinstance(environment, RiverSwimReferenceEnvironment)
    kernel = environment._environment
    oracle = kernel.optimal_average_reward()

    with pytest.raises(AttributeError, match="immutable"):
        setattr(environment, "_environment", RiverSwimMDP())
    with pytest.raises(AttributeError, match="immutable"):
        setattr(kernel, "_transitions_np", np.zeros((2, 3, 3), dtype=np.float32))
    with pytest.raises(AttributeError, match="immutable"):
        setattr(kernel, "_rewards_np", np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="read-only"):
        kernel._transitions_np[1, 0, 1] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        kernel._rewards_np[0, 0] = 999.0

    assert kernel.optimal_average_reward() == oracle


def test_riverswim_validator_rejects_an_alternate_possible_wrong_key_transition() -> None:
    runner = _runner(horizon=1)
    state = runner.init()
    command = _right_command(runner, state)
    correct_key = runner._environment_key(0)
    expected = runner.environment_adapter.execute(
        state.environment_state,
        command,
        key=correct_key,
    )

    forged = None
    for offset in range(1, 128):
        wrong_key = jr.fold_in(correct_key, offset)
        candidate = runner.environment_adapter.execute(
            state.environment_state,
            command,
            key=wrong_key,
        )
        if np.asarray(candidate.state.state_index).tobytes() != np.asarray(
            expected.state.state_index
        ).tobytes():
            forged = candidate
            break
    assert forged is not None, "test keys must include a different possible transition"

    with pytest.raises(DecisionOwnershipError, match="key|replay|stochastic|transition"):
        runner.environment_adapter.validate_execution(
            state.environment_state,
            command,
            forged,
            key=correct_key,
        )


def test_riverswim_wrong_key_result_commits_uncertain_halt_without_redispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1, seed=2)
    initial = runner.init()
    assert initial.transaction_state.decision is not None
    assert initial.transaction_state.decision.proposed_action is not None
    assert initial.transaction_state.decision.proposed_action.to_python() == 1
    original_execute = RiverSwimReferenceEnvironment.execute
    execute_calls = 0

    def wrong_key_execute(
        self: RiverSwimReferenceEnvironment,
        state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> ReferenceEnvironmentExecution:
        nonlocal execute_calls
        execute_calls += 1
        expected = original_execute(self, state, command, key=key)
        for offset in range(1, 128):
            candidate = original_execute(self, state, command, key=jr.fold_in(key, offset))
            if np.asarray(candidate.state.state_index).tobytes() != np.asarray(
                expected.state.state_index
            ).tobytes():
                return candidate
        raise AssertionError("test keys must include a different possible transition")

    monkeypatch.setattr(RiverSwimReferenceEnvironment, "execute", wrong_key_execute)
    rejected = runner.step(initial)

    assert not rejected.accepted
    assert rejected.state.phase is LifePhase.HALTED
    assert rejected.state.halt is not None
    assert rejected.state.halt.stage is HaltStage.DISPATCH_UNCERTAIN
    assert rejected.state.dispatch_attempts == 1
    assert rejected.state.environment_rng_cursor == 1
    assert rejected.state.executed_events == 0
    assert rejected.state.accepted_events == 0
    assert int(rejected.state.environment_state.step_count) == 0
    assert int(rejected.state.agent_state.agent_state.step_count) == 0
    assert rejected.state.metrics.accepted_events == 0
    with pytest.raises(DecisionOwnershipError, match="stale|current"):
        runner.step(initial)
    assert execute_calls == 1


def test_riverswim_applied_action_substitution_is_retained_without_learning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(horizon=1, seed=2)
    initial = runner.init()
    original_execute = RiverSwimReferenceEnvironment.execute
    execute_calls = 0

    def substituted_execute(
        self: RiverSwimReferenceEnvironment,
        state: Any,
        command: DispatchCommand,
        *,
        key: Any,
    ) -> ReferenceEnvironmentExecution:
        nonlocal execute_calls
        execute_calls += 1
        reference = original_execute(self, state, command, key=key)
        commanded = command.effective_action.to_python()
        assert isinstance(commanded, int) and not isinstance(commanded, bool)
        substituted = self.manifest.action_spec.encode(
            np.asarray(1 - commanded, dtype=np.int32)
        )
        observation, reward, next_state = self._environment.step(
            state,
            jnp.asarray(substituted.to_numpy(), dtype=jnp.int32),
            key,
        )
        return dataclasses.replace(
            reference,
            state=next_state,
            applied_action=substituted,
            next_observation=self.manifest.observation_spec.encode(
                np.asarray(observation, dtype=np.float32)
            ),
            reward=float(np.asarray(reward, dtype=np.float32)),
        )

    monkeypatch.setattr(RiverSwimReferenceEnvironment, "execute", substituted_execute)
    rejected = runner.step(initial)

    assert not rejected.accepted
    assert rejected.state.phase is LifePhase.HALTED
    assert rejected.state.halt is not None
    assert rejected.state.halt.stage is HaltStage.POST_EXECUTION_DIVERGENCE
    assert rejected.state.dispatch_attempts == 1
    assert rejected.state.environment_rng_cursor == 1
    assert rejected.state.executed_events == 1
    assert rejected.state.accepted_events == 0
    assert int(rejected.state.environment_state.step_count) == 1
    assert int(rejected.state.agent_state.agent_state.step_count) == 0
    assert rejected.state.metrics.accepted_events == 0
    assert rejected.state.pending_outcome is None
    with pytest.raises(DecisionOwnershipError, match="stale|current"):
        runner.step(initial)
    assert execute_calls == 1


@pytest.mark.integration
def test_riverswim_reference_life_runs_five_keyed_events_without_extra_dispatch() -> None:
    runner = _runner(horizon=5)
    state = runner.init()
    events = []
    while state.phase is LifePhase.QUIESCENT:
        step = runner.step(state)
        assert step.accepted, step.rejection_reason
        assert step.event is not None
        events.append(step.event)
        state = step.state

    assert state.phase is LifePhase.COMPLETED
    assert len(events) == 5
    assert state.accepted_events == state.executed_events == state.dispatch_attempts == 5
    assert state.environment_rng_cursor == 5
    assert int(state.environment_state.step_count) == 5
    assert int(state.agent_state.agent_state.step_count) == 5
    assert state.metrics.accepted_events == 5
    assert runner.config.config["metrics"]["config"]["mode"] == "stationary"
    assert state.metrics.phase_event_counts == (5, 0)
    assert state.metrics.phase_switches == 0
    assert state.metrics.current_phase == 0
    assert state.metrics.current_segment_events == 5
    assert np.isclose(
        state.metrics.oracle_reward_sum - state.metrics.reward_sum,
        state.metrics.regret_sum,
    )
    with pytest.raises(DecisionOwnershipError, match="completed"):
        runner.step(state)
