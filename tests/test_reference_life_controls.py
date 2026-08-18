"""Development contracts for exact-dispatch reference-life control adapters."""

from __future__ import annotations

import dataclasses
import itertools
import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.average_reward import DifferentialSARSAState
from alberta_framework.core.sarsa import SARSAState
from alberta_framework.core.types import LMSState
from alberta_framework.reference_agent import (
    AuthorizationStatus,
    DecisionOwnershipError,
    DispatchAuthorization,
)
from alberta_framework.reference_life import (
    ExactDispatchConfig,
    LifePhase,
    ReferenceLifeMetricsConfig,
    ReferenceLifeRunner,
    RiverSwimReferenceEnvironment,
    SwitchingTwoStateReferenceEnvironment,
)
from alberta_framework.reference_life_controls import (
    REFERENCE_CONTROL_MAX_ACCEPTED_EVENTS,
    AnalyticOracleReferenceAdapter,
    AnalyticOracleReferenceConfig,
    DifferentialSARSAReferenceAdapter,
    DifferentialSARSAReferenceConfig,
    DiscountedSARSAReferenceAdapter,
    DiscountedSARSAReferenceConfig,
    ReferenceLifeControlState,
    UniformRandomReferenceAdapter,
    UniformRandomReferenceConfig,
    control_state_resource_usage,
)
from alberta_framework.streams.closed_loop import (
    RiverSwimConfig,
    RiverSwimMDP,
    SwitchingTwoStateConfig,
)


def _switching_runner(
    adapter: object,
    environment_config: SwitchingTwoStateConfig,
    *,
    lifecycle_id: str,
    seed: int = 17,
    horizon: int = 4,
) -> ReferenceLifeRunner:
    manifest = adapter.manifest  # type: ignore[attr-defined]
    environment = SwitchingTwoStateReferenceEnvironment(
        environment_config,
        observation_spec=manifest.observation_spec,
        action_spec=manifest.action_spec,
    )
    return ReferenceLifeRunner.create(
        agent_adapter=adapter,  # type: ignore[arg-type]
        environment_adapter=environment,
        lifecycle_id=lifecycle_id,
        seed=seed,
        max_accepted_events=horizon,
        metrics_config=ReferenceLifeMetricsConfig(mode="switching_two_phase"),
    )


def _river_runner(
    adapter: object,
    environment_config: RiverSwimConfig,
    *,
    lifecycle_id: str,
    seed: int = 17,
    horizon: int = 4,
) -> ReferenceLifeRunner:
    manifest = adapter.manifest  # type: ignore[attr-defined]
    dispatch = ExactDispatchConfig(
        executor_id="asi.riverswim.executor",
        executor_epoch="asi.riverswim.executor_epoch.1",
    )
    environment = RiverSwimReferenceEnvironment(
        environment_config,
        observation_spec=manifest.observation_spec,
        action_spec=manifest.action_spec,
        executor_id=dispatch.executor_id,
        executor_epoch=dispatch.executor_epoch,
    )
    return ReferenceLifeRunner.create(
        agent_adapter=adapter,  # type: ignore[arg-type]
        environment_adapter=environment,
        lifecycle_id=lifecycle_id,
        seed=seed,
        max_accepted_events=horizon,
        dispatch_config=dispatch,
        metrics_config=ReferenceLifeMetricsConfig(mode="stationary"),
    )


def _uniform_switching() -> UniformRandomReferenceAdapter:
    environment = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    return UniformRandomReferenceAdapter(
        UniformRandomReferenceConfig.for_switching(environment)
    )


def _differential_switching() -> DifferentialSARSAReferenceAdapter:
    environment = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    return DifferentialSARSAReferenceAdapter(
        DifferentialSARSAReferenceConfig.for_switching(
            environment,
            q_step_size=0.1,
            average_reward_step_size=0.01,
            epsilon_start=0.25,
            epsilon_end=0.25,
        )
    )


def _discounted_switching() -> DiscountedSARSAReferenceAdapter:
    environment = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    return DiscountedSARSAReferenceAdapter(
        DiscountedSARSAReferenceConfig.for_switching(
            environment,
            gamma=0.9,
            epsilon_start=0.25,
            epsilon_end=0.25,
            hidden_sizes=(),
            step_size=0.05,
        )
    )


def _control_for_environment(
    algorithm: str,
    environment_kind: str,
) -> tuple[object, SwitchingTwoStateConfig | RiverSwimConfig]:
    if environment_kind == "switching_two_state":
        switching_environment = SwitchingTwoStateConfig(  # type: ignore[call-arg]
            phase_length=2
        )
        if algorithm == "uniform_random":
            config = UniformRandomReferenceConfig.for_switching(switching_environment)
            return UniformRandomReferenceAdapter(config), switching_environment
        if algorithm == "analytic_oracle":
            oracle_config = AnalyticOracleReferenceConfig.for_switching(
                switching_environment,
                horizon=4,
            )
            return AnalyticOracleReferenceAdapter(oracle_config), switching_environment
        if algorithm == "differential_sarsa":
            differential_config = DifferentialSARSAReferenceConfig.for_switching(
                switching_environment,
                epsilon_start=0.25,
                epsilon_end=0.25,
            )
            return (
                DifferentialSARSAReferenceAdapter(differential_config),
                switching_environment,
            )
        if algorithm == "discounted_sarsa":
            discounted_config = DiscountedSARSAReferenceConfig.for_switching(
                switching_environment,
                epsilon_start=0.25,
                epsilon_end=0.25,
                hidden_sizes=(),
            )
            return (
                DiscountedSARSAReferenceAdapter(discounted_config),
                switching_environment,
            )
    elif environment_kind == "riverswim":
        river_environment = RiverSwimConfig(n_states=3)  # type: ignore[call-arg]
        if algorithm == "uniform_random":
            config = UniformRandomReferenceConfig.for_riverswim(river_environment)
            return UniformRandomReferenceAdapter(config), river_environment
        if algorithm == "analytic_oracle":
            oracle_config = AnalyticOracleReferenceConfig.for_riverswim(
                river_environment,
                horizon=4,
            )
            return AnalyticOracleReferenceAdapter(oracle_config), river_environment
        if algorithm == "differential_sarsa":
            differential_config = DifferentialSARSAReferenceConfig.for_riverswim(
                river_environment,
                epsilon_start=0.25,
                epsilon_end=0.25,
            )
            return DifferentialSARSAReferenceAdapter(differential_config), river_environment
        if algorithm == "discounted_sarsa":
            discounted_config = DiscountedSARSAReferenceConfig.for_riverswim(
                river_environment,
                epsilon_start=0.25,
                epsilon_end=0.25,
                hidden_sizes=(),
            )
            return DiscountedSARSAReferenceAdapter(discounted_config), river_environment
    raise AssertionError(f"unsupported test control {algorithm}/{environment_kind}")


@pytest.mark.integration
@pytest.mark.parametrize(
    "algorithm",
    (
        "uniform_random",
        "analytic_oracle",
        "differential_sarsa",
        "discounted_sarsa",
    ),
)
@pytest.mark.parametrize("environment_kind", ("switching_two_state", "riverswim"))
def test_every_control_runs_one_complete_life_in_each_environment(
    algorithm: str,
    environment_kind: str,
) -> None:
    adapter, environment = _control_for_environment(algorithm, environment_kind)
    if isinstance(environment, RiverSwimConfig):
        runner = _river_runner(
            adapter,
            environment,
            lifecycle_id=f"control.{algorithm}.riverswim.life",
            horizon=3,
        )
    else:
        runner = _switching_runner(
            adapter,
            environment,
            lifecycle_id=f"control.{algorithm}.switching.life",
            horizon=3,
        )

    run = runner.run_to_completion(runner.init())

    assert run.state.phase is LifePhase.COMPLETED
    assert run.state.accepted_events == 3
    assert len(run.events) == 3
    assert all(event.step_result.transaction_accepted for event in run.events)
    assert all(event.transaction.decision.proposed_action is not None for event in run.events)


def test_executed_control_scalars_are_canonical_actual_float32_values() -> None:
    environment = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    differential_values = {
        "q_step_size": 0.123456789,
        "average_reward_step_size": 0.234567891,
        "trace_decay": 0.345678912,
        "epsilon_start": 0.456789123,
        "epsilon_end": 0.123456789,
    }
    differential = DifferentialSARSAReferenceConfig.for_switching(
        environment,
        **differential_values,
    )
    discounted_values = {
        "gamma": 0.912345678,
        "epsilon_start": 0.456789123,
        "epsilon_end": 0.123456789,
        "step_size": 0.234567891,
        "sparsity": 0.345678912,
        "leaky_relu_slope": 0.0123456789,
        "lamda": 0.567891234,
    }
    discounted = DiscountedSARSAReferenceConfig.for_switching(
        environment,
        hidden_sizes=(),
        **discounted_values,
    )

    for name, raw in differential_values.items():
        assert getattr(differential, name) == float(np.float32(raw))
    for name, raw in discounted_values.items():
        assert getattr(discounted, name) == float(np.float32(raw))
    assert differential.core_config().q_step_size == differential.q_step_size
    assert discounted.core_config().gamma == discounted.gamma


def test_oracle_environment_scalars_bind_the_executed_float32_values() -> None:
    switching_raw = 0.123456789
    switching = AnalyticOracleReferenceConfig.for_switching(
        SwitchingTwoStateConfig(  # type: ignore[call-arg]
            phase_length=2,
            payoffs_a=((switching_raw, 0.0), (0.0, switching_raw)),
        ),
        horizon=4,
    )
    assert switching.environment_config["payoffs_a"] == [
        [float(np.float32(switching_raw)), 0.0],
        [0.0, float(np.float32(switching_raw))],
    ]

    river_values = {
        "p_right_up": 0.345678912,
        "p_right_down": 0.123456789,
        "reward_left": 0.0123456789,
        "reward_right": 0.987654321,
    }
    river = AnalyticOracleReferenceConfig.for_riverswim(
        RiverSwimConfig(  # type: ignore[call-arg]
            n_states=3,
            **river_values,
        ),
        horizon=4,
    )
    for name, raw in river_values.items():
        assert river.environment_config[name] == float(np.float32(raw))


def test_control_scalar_validation_uses_post_narrowing_float32_bounds() -> None:
    environment = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    overflow = float(np.finfo(np.float32).max) * 2.0
    for field in ("q_step_size", "average_reward_step_size"):
        with pytest.raises(ValueError, match="float32"):
            DifferentialSARSAReferenceConfig.for_switching(
                environment,
                **{field: overflow},
            )
    for field in ("step_size", "leaky_relu_slope"):
        with pytest.raises(ValueError, match="float32"):
            DiscountedSARSAReferenceConfig.for_switching(
                environment,
                hidden_sizes=(),
                **{field: overflow},
            )

    rounds_to_one = math.nextafter(1.0, 0.0)
    with pytest.raises(ValueError, match="gamma|less than one|float32"):
        DiscountedSARSAReferenceConfig.for_switching(
            environment,
            gamma=rounds_to_one,
            hidden_sizes=(),
        )
    with pytest.raises(ValueError, match="sparsity|less than one|float32"):
        DiscountedSARSAReferenceConfig.for_switching(
            environment,
            sparsity=rounds_to_one,
            hidden_sizes=(),
        )


def test_environment_scalar_gate_rejects_non_real_overflow_and_positive_collapse() -> None:
    overflow = float(np.finfo(np.float32).max) * 2.0
    with pytest.raises(ValueError, match="float32|numeric"):
        UniformRandomReferenceConfig.for_switching(
            SwitchingTwoStateConfig(  # type: ignore[call-arg]
                phase_length=2,
                payoffs_a=((overflow, 0.0), (0.0, 0.0)),
            )
        )
    with pytest.raises(ValueError, match="real|float32|numeric"):
        UniformRandomReferenceConfig.for_switching(
            SwitchingTwoStateConfig(  # type: ignore[call-arg]
                phase_length=2,
                payoffs_a=(("0.5", 0.0), (0.0, 0.0)),
            )
        )
    with pytest.raises(ValueError, match="positive|float32"):
        UniformRandomReferenceConfig.for_riverswim(
            RiverSwimConfig(  # type: ignore[call-arg]
                n_states=3,
                p_right_up=math.nextafter(0.0, 1.0),
            )
        )


def test_decay_counts_and_discounted_network_shapes_are_resource_bounded() -> None:
    environment = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    int32_max = int(np.iinfo(np.int32).max)
    DifferentialSARSAReferenceConfig.for_switching(
        environment,
        epsilon_decay_steps=int32_max,
    )
    DiscountedSARSAReferenceConfig.for_switching(
        environment,
        epsilon_decay_steps=int32_max,
        hidden_sizes=(),
    )
    for config_type in (
        DifferentialSARSAReferenceConfig,
        DiscountedSARSAReferenceConfig,
    ):
        with pytest.raises(ValueError, match="epsilon_decay_steps|int32|capacity"):
            config_type(
                environment_kind="switching_two_state",
                observation_dim=2,
                epsilon_decay_steps=int32_max + 1,
            )

    with pytest.raises(ValueError, match="hidden|layer|resource"):
        DiscountedSARSAReferenceConfig.for_switching(
            environment,
            hidden_sizes=(1,) * 33,
        )
    with pytest.raises(ValueError, match="hidden|int32|resource"):
        DiscountedSARSAReferenceConfig.for_switching(
            environment,
            hidden_sizes=(int32_max + 1,),
        )
    with pytest.raises(ValueError, match="network|parameter|resource"):
        DiscountedSARSAReferenceConfig.for_switching(
            environment,
            hidden_sizes=(4096, 4096),
        )


def test_stochastic_controls_bind_explicit_threefry_roots_across_global_defaults() -> None:
    environment = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    adapter = UniformRandomReferenceAdapter(
        UniformRandomReferenceConfig.for_switching(environment)
    )
    assert adapter.manifest.config["rng_implementation"] == "threefry2x32"

    explicit_key = jr.key(91, impl="threefry2x32")
    baseline_state = adapter.init(explicit_key, lifecycle_id="control.rng.baseline")
    baseline_started, baseline_decision = adapter.start(
        baseline_state,
        observation_id="observation.0",
        observation=np.asarray((1.0, 0.0), dtype=np.float32),
    )
    assert baseline_decision.proposed_action is not None
    assert baseline_started.random_key is not None
    assert str(jr.key_impl(baseline_started.random_key)) == "threefry2x32"

    with jax.default_prng_impl("rbg"):
        alternate_state = adapter.init(
            explicit_key,
            lifecycle_id="control.rng.alternate_global",
        )
        _, alternate_decision = adapter.start(
            alternate_state,
            observation_id="observation.0",
            observation=np.asarray((1.0, 0.0), dtype=np.float32),
        )
        assert alternate_decision.proposed_action == baseline_decision.proposed_action
        with pytest.raises(DecisionOwnershipError, match="threefry2x32"):
            adapter.init(
                jr.key(91),
                lifecycle_id="control.rng.rbg_root",
            )


def test_control_resource_usage_includes_both_array_value_caches() -> None:
    adapter = _uniform_switching()
    fresh = adapter.init(
        jr.key(7, impl="threefry2x32"),
        lifecycle_id="control.resources.cache",
    )
    fresh_usage = control_state_resource_usage(fresh)
    started, _ = adapter.start(
        fresh,
        observation_id="observation.0",
        observation=np.asarray((1.0, 0.0), dtype=np.float32),
    )
    armed_usage = control_state_resource_usage(started)

    assert armed_usage.array_leaves == fresh_usage.array_leaves + 2
    assert armed_usage.array_elements == fresh_usage.array_elements + 3
    assert armed_usage.persistent_bytes == fresh_usage.persistent_bytes + 12
    assert armed_usage.floating_array_leaves == fresh_usage.floating_array_leaves + 1


def test_uniform_random_schedule_is_seed_deterministic_and_owner_bound() -> None:
    environment = SwitchingTwoStateConfig(phase_length=3)  # type: ignore[call-arg]
    config = UniformRandomReferenceConfig.for_switching(environment)
    first = UniformRandomReferenceAdapter(config)
    second = UniformRandomReferenceAdapter(config)
    first_runner = _switching_runner(
        first,
        environment,
        lifecycle_id="control.random.determinism",
        seed=123,
        horizon=24,
    )
    second_runner = _switching_runner(
        second,
        environment,
        lifecycle_id="control.random.determinism",
        seed=123,
        horizon=24,
    )

    first_run = first_runner.run_to_completion(first_runner.init())
    second_run = second_runner.run_to_completion(second_runner.init())
    first_actions = tuple(
        event.transaction.decision.proposed_action.to_python()  # type: ignore[union-attr]
        for event in first_run.events
    )
    second_actions = tuple(
        event.transaction.decision.proposed_action.to_python()  # type: ignore[union-attr]
        for event in second_run.events
    )

    assert first_actions == second_actions
    assert first_run.state.transcript_sha256 == second_run.state.transcript_sha256
    with pytest.raises(DecisionOwnershipError, match="owner|another adapter"):
        second.validate_state(first_run.state.agent_state)


def test_oracle_uses_exact_finite_horizon_dynamic_program() -> None:
    switching = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    switching_horizon = 6
    switching_config = AnalyticOracleReferenceConfig.for_switching(
        switching,
        horizon=switching_horizon,
    )
    switching_runner = _switching_runner(
        AnalyticOracleReferenceAdapter(switching_config),
        switching,
        lifecycle_id="control.oracle.switching.valid",
        horizon=switching_horizon,
    )
    valid_switching_run = switching_runner.run_to_completion(switching_runner.init())
    switching_payoffs = np.asarray(
        (switching.payoffs_a, switching.payoffs_b), dtype=np.float64
    )
    switching_values = np.zeros(2, dtype=np.float64)
    expected_switching = np.empty((switching_horizon, 2), dtype=np.int32)
    for index in range(switching_horizon - 1, -1, -1):
        phase = (index // switching.phase_length) % 2
        q_values = switching_payoffs[phase] + switching_values[np.newaxis, :]
        expected_switching[index] = np.argmax(q_values, axis=1)
        switching_values = q_values[np.arange(2), expected_switching[index]]
    for event in valid_switching_run.events:
        decision = event.transaction.decision
        observation = np.asarray(decision.observation.to_python())
        state_index = int(np.argmax(observation))
        assert decision.proposed_action is not None
        assert decision.proposed_action.to_python() == int(
            expected_switching[decision.decision_index, state_index]
        )

    river = RiverSwimConfig(n_states=4)  # type: ignore[call-arg]
    river_kernel = RiverSwimMDP(river)
    river_horizon = 8
    river_config = AnalyticOracleReferenceConfig.for_riverswim(
        river,
        horizon=river_horizon,
    )
    river_adapter = AnalyticOracleReferenceAdapter(river_config)
    river_runner = _river_runner(
        river_adapter,
        river,
        lifecycle_id="control.oracle.river",
        horizon=river_horizon,
    )
    river_run = river_runner.run_to_completion(river_runner.init())
    transitions = river_kernel.transition_tensor.astype(np.float64)
    rewards = river_kernel.reward_tensor.astype(np.float64)
    river_values = np.zeros(river.n_states, dtype=np.float64)
    expected_river = np.empty((river_horizon, river.n_states), dtype=np.int32)
    for index in range(river_horizon - 1, -1, -1):
        q_values = rewards + np.einsum("asn,n->sa", transitions, river_values)
        expected_river[index] = np.argmax(q_values, axis=1)
        river_values = q_values[np.arange(river.n_states), expected_river[index]]
    for event in river_run.events:
        decision = event.transaction.decision
        state_index = int(np.argmax(np.asarray(decision.observation.to_python())))
        assert decision.proposed_action is not None
        assert decision.proposed_action.to_python() == int(
            expected_river[decision.decision_index, state_index]
        )

    manifest_config = river_adapter.manifest.config
    assert manifest_config["privileged"] is True
    assert manifest_config["environment_config_sha256"] == river_config.environment_config_sha256


def test_stale_cross_config_and_cached_action_corruption_are_failure_atomic() -> None:
    environment = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    adapter = UniformRandomReferenceAdapter(
        UniformRandomReferenceConfig.for_switching(environment)
    )
    runner = _switching_runner(
        adapter,
        environment,
        lifecycle_id="control.random.ownership",
        horizon=2,
    )
    initial = runner.init()
    state = initial.agent_state
    assert isinstance(state, ReferenceLifeControlState)
    decision = adapter.current_decision(state)
    assert decision.proposed_action is not None
    proposed_action = decision.proposed_action.to_python()
    assert isinstance(proposed_action, int)
    other_action = adapter.manifest.action_spec.encode(
        np.asarray(1 - proposed_action, dtype=np.int32)
    )
    substituted_decision = dataclasses.replace(decision, proposed_action=other_action)
    substituted_authorization = DispatchAuthorization(
        decision=substituted_decision,
        status=AuthorizationStatus.EXACT,
        authorized_action=other_action,
        authority_id="control.test.authority",
        policy_version="control.test.policy.1",
        authorization_id=f"{substituted_decision.decision_id}:authorization",
    )
    with pytest.raises(DecisionOwnershipError, match="decision|action|cache"):
        adapter.settle_dispatch(state, substituted_authorization)

    corrupted = dataclasses.replace(state, current_action=other_action)
    with pytest.raises(DecisionOwnershipError, match="action|random"):
        adapter.validate_state(corrupted)

    different_config = UniformRandomReferenceAdapter(
        UniformRandomReferenceConfig.for_riverswim(
            RiverSwimConfig(n_states=3)  # type: ignore[call-arg]
        )
    )
    with pytest.raises(DecisionOwnershipError, match="owner|manifest|config"):
        different_config.validate_state(state)
    relabeled_config = dataclasses.replace(
        state,
        config_sha256=different_config.manifest.config_sha256,
    )
    with pytest.raises(DecisionOwnershipError, match="configuration"):
        adapter.validate_state(relabeled_config)

    first_step = runner.step(initial)
    assert first_step.event is not None
    stale_state = first_step.state.agent_state
    rejected = adapter.apply_outcome(stale_state, first_step.event.transaction)
    assert not rejected.accepted
    assert rejected.state is stale_state
    assert rejected.next_decision is None
    assert rejected.parameters_changed is False


def test_differential_sarsa_applies_hand_computed_first_update() -> None:
    rewarding = SwitchingTwoStateConfig(  # type: ignore[call-arg]
        phase_length=10,
        payoffs_a=((1.0, 1.0), (1.0, 1.0)),
        payoffs_b=((1.0, 1.0), (1.0, 1.0)),
    )
    adapter = DifferentialSARSAReferenceAdapter(
        DifferentialSARSAReferenceConfig.for_switching(
            rewarding,
            q_step_size=0.5,
            average_reward_step_size=0.25,
            epsilon_start=0.0,
            epsilon_end=0.0,
            use_bias=False,
        )
    )
    runner = _switching_runner(
        adapter,
        rewarding,
        lifecycle_id="control.differential.update",
        horizon=1,
    )
    initial = runner.init()
    initial_control = initial.agent_state
    assert isinstance(initial_control, ReferenceLifeControlState)
    initial_learning = initial_control.agent_state
    assert isinstance(initial_learning, DifferentialSARSAState)
    initial_decision = adapter.current_decision(initial_control)
    initial_state_index = int(np.argmax(initial_decision.observation.to_numpy()))
    assert initial_decision.proposed_action is not None
    initial_action = initial_decision.proposed_action.to_python()
    assert isinstance(initial_action, int)

    step = runner.step(initial)

    assert step.accepted
    final_control = step.state.agent_state
    assert isinstance(final_control, ReferenceLifeControlState)
    final_learning = final_control.agent_state
    assert isinstance(final_learning, DifferentialSARSAState)
    expected_weights = np.zeros((2, 2), dtype=np.float32)
    expected_weights[initial_action, initial_state_index] = 0.5
    np.testing.assert_array_equal(np.asarray(final_learning.q_weights), expected_weights)
    np.testing.assert_array_equal(np.asarray(final_learning.q_bias), np.zeros(2, dtype=np.float32))
    assert float(final_learning.average_reward) == pytest.approx(0.25)
    assert int(final_learning.step_count) == 1
    assert step.event is not None
    assert step.event.step_result.parameters_changed


def test_discounted_sarsa_updates_on_continuing_unit_discount_outcome() -> None:
    rewarding = SwitchingTwoStateConfig(  # type: ignore[call-arg]
        phase_length=10,
        payoffs_a=((1.0, 1.0), (1.0, 1.0)),
        payoffs_b=((1.0, 1.0), (1.0, 1.0)),
    )
    adapter = DiscountedSARSAReferenceAdapter(
        DiscountedSARSAReferenceConfig.for_switching(
            rewarding,
            gamma=0.5,
            epsilon_start=0.0,
            epsilon_end=0.0,
            hidden_sizes=(),
            step_size=0.25,
            sparsity=0.0,
            use_layer_norm=False,
        )
    )
    runner = _switching_runner(
        adapter,
        rewarding,
        lifecycle_id="control.discounted.update",
        horizon=2,
    )
    initial = runner.init()
    run = runner.run_to_completion(initial)

    assert run.state.phase is LifePhase.COMPLETED
    final_control = run.state.agent_state
    assert isinstance(final_control, ReferenceLifeControlState)
    final_learning = final_control.agent_state
    assert isinstance(final_learning, SARSAState)
    assert int(final_learning.step_count) == 2
    assert int(final_learning.learner_state.step_count) == 2
    assert all(not event.transaction.is_boundary for event in run.events)
    assert all(event.transaction.discount == 1.0 for event in run.events)
    assert any(event.step_result.parameters_changed for event in run.events)
    assert adapter.manifest.config["gamma"] == 0.5

    resources = control_state_resource_usage(final_control)
    assert resources.array_leaves > 0
    assert resources.array_elements > 0
    assert resources.persistent_bytes > 0
    assert resources.floating_array_leaves > 0


def test_learning_control_state_validation_requires_exact_persistent_dtypes() -> None:
    differential = _differential_switching()
    differential_runner = _switching_runner(
        differential,
        SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
        lifecycle_id="control.differential.dtype",
        horizon=1,
    )
    differential_state = differential_runner.init().agent_state
    assert isinstance(differential_state, ReferenceLifeControlState)
    differential_learner = differential_state.agent_state
    assert isinstance(differential_learner, DifferentialSARSAState)
    bad_differential_words = differential_learner.replace(  # type: ignore[attr-defined]
        step_words=jnp.asarray((0, 0), dtype=jnp.int32)
    )
    with pytest.raises(DecisionOwnershipError, match="step_words|uint32|dtype"):
        differential.validate_state(
            dataclasses.replace(
                differential_state,
                agent_state=bad_differential_words,
            )
        )

    discounted = _discounted_switching()
    discounted_runner = _switching_runner(
        discounted,
        SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
        lifecycle_id="control.discounted.dtype",
        horizon=1,
    )
    discounted_state = discounted_runner.init().agent_state
    assert isinstance(discounted_state, ReferenceLifeControlState)
    discounted_learner = discounted_state.agent_state
    assert isinstance(discounted_learner, SARSAState)
    inner = discounted_learner.learner_state
    bad_inner_words = inner.replace(  # type: ignore[attr-defined]
        step_words=jnp.asarray((0, 0), dtype=jnp.int32)
    )
    with pytest.raises(DecisionOwnershipError, match="step_words|uint32|dtype"):
        discounted.validate_state(
            dataclasses.replace(
                discounted_state,
                agent_state=discounted_learner.replace(  # type: ignore[attr-defined]
                    learner_state=bad_inner_words
                ),
            )
        )

    head_weights = inner.head_params.weights
    bad_head_params = inner.head_params.replace(  # type: ignore[attr-defined]
        weights=(head_weights[0].astype(jnp.float16), *head_weights[1:])
    )
    bad_float_tree = inner.replace(  # type: ignore[attr-defined]
        head_params=bad_head_params
    )
    with pytest.raises(DecisionOwnershipError, match="float32|dtype|contract"):
        discounted.validate_state(
            dataclasses.replace(
                discounted_state,
                agent_state=discounted_learner.replace(  # type: ignore[attr-defined]
                    learner_state=bad_float_tree
                ),
            )
        )


def test_all_control_adapters_declare_the_exact_signed_int32_capacity() -> None:
    adapters = (
        _uniform_switching(),
        AnalyticOracleReferenceAdapter(
            AnalyticOracleReferenceConfig.for_switching(
                SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
                horizon=2,
            )
        ),
        _differential_switching(),
        _discounted_switching(),
    )
    assert REFERENCE_CONTROL_MAX_ACCEPTED_EVENTS == int(np.iinfo(np.int32).max)
    assert adapters[1].max_accepted_events == 2
    assert all(
        adapter.max_accepted_events == REFERENCE_CONTROL_MAX_ACCEPTED_EVENTS
        for adapter in (adapters[0], adapters[2], adapters[3])
    )


def test_control_configs_and_states_are_immutable() -> None:
    config = UniformRandomReferenceConfig.for_switching(
        SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.observation_dim = 3  # type: ignore[misc]

    adapter = UniformRandomReferenceAdapter(config)
    runner = _switching_runner(
        adapter,
        SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
        lifecycle_id="control.random.immutable",
        horizon=1,
    )
    state = runner.init().agent_state
    assert isinstance(state, ReferenceLifeControlState)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.decision_index = 3  # type: ignore[misc]


def test_runner_binds_control_environment_kind_and_oracle_model() -> None:
    switching = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    uniform = UniformRandomReferenceAdapter(
        UniformRandomReferenceConfig.for_switching(switching)
    )
    with pytest.raises(ValueError, match="kinds differ"):
        _river_runner(
            uniform,
            RiverSwimConfig(n_states=2),  # type: ignore[call-arg]
            lifecycle_id="control.cross-kind",
            horizon=2,
        )

    oracle = AnalyticOracleReferenceAdapter(
        AnalyticOracleReferenceConfig.for_switching(switching, horizon=3)
    )
    mismatched = SwitchingTwoStateConfig(phase_length=3)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="bound environment differs"):
        _switching_runner(
            oracle,
            mismatched,
            lifecycle_id="control.oracle.cross-environment",
            horizon=3,
        )


def test_runner_uses_explicit_threefry_under_an_rbg_global_default() -> None:
    environment = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]

    def action_trace() -> tuple[tuple[int, ...], str]:
        adapter = UniformRandomReferenceAdapter(
            UniformRandomReferenceConfig.for_switching(environment)
        )
        runner = _switching_runner(
            adapter,
            environment,
            lifecycle_id="control.explicit-threefry",
            seed=123,
            horizon=12,
        )
        result = runner.run_to_completion(runner.init())
        actions = tuple(
            int(event.transaction.decision.proposed_action.to_python())
            for event in result.events
            if event.transaction.decision.proposed_action is not None
        )
        return actions, result.state.transcript_sha256

    expected = action_trace()
    with jax.default_prng_impl("rbg"):
        observed = action_trace()
    assert observed == expected


def test_nested_core_agent_has_stable_jit_identity_and_mutation_fails_closed() -> None:
    environment = SwitchingTwoStateConfig(  # type: ignore[call-arg]
        phase_length=4,
        payoffs_a=((1.0, 1.0), (1.0, 1.0)),
        payoffs_b=((1.0, 1.0), (1.0, 1.0)),
    )
    config = DifferentialSARSAReferenceConfig.for_switching(
        environment,
        q_step_size=0.5,
        average_reward_step_size=0.25,
        epsilon_start=0.0,
        epsilon_end=0.0,
    )
    adapter = DifferentialSARSAReferenceAdapter(config)
    stable = adapter._differential_agent
    assert adapter._differential_agent is stable
    stable._config = dataclasses.replace(  # type: ignore[attr-defined]
        stable.config,
        q_step_size=0.0,
        average_reward_step_size=0.0,
    )
    with pytest.raises(DecisionOwnershipError, match="configuration|mutat"):
        _ = adapter._differential_agent


def test_switching_oracle_is_finite_horizon_optimal_across_phase_boundaries() -> None:
    environment = SwitchingTwoStateConfig(  # type: ignore[call-arg]
        phase_length=1,
        payoffs_a=((10.0, 0.0), (0.0, 9.0)),
        payoffs_b=((9.0, 0.0), (0.0, 10.0)),
    )
    horizon = 6
    adapter = AnalyticOracleReferenceAdapter(
        AnalyticOracleReferenceConfig.for_switching(environment, horizon=horizon)
    )
    runner = _switching_runner(
        adapter,
        environment,
        lifecycle_id="control.oracle.periodic-optimal",
        seed=11,
        horizon=horizon,
    )
    result = runner.run_to_completion(runner.init())
    first_observation = result.events[0].transaction.decision.observation.to_numpy()
    initial_state = int(np.argmax(first_observation))

    def sequence_return(actions: tuple[int, ...]) -> float:
        state = initial_state
        total = 0.0
        for index, action in enumerate(actions):
            payoffs = environment.payoffs_a if index % 2 == 0 else environment.payoffs_b
            total += float(payoffs[state][action])
            state = action
        return total

    best = max(sequence_return(actions) for actions in itertools.product((0, 1), repeat=horizon))
    assert result.state.metrics.reward_sum == best


def test_riverswim_oracle_uses_stable_finite_horizon_expectations() -> None:
    environment = RiverSwimConfig(  # type: ignore[call-arg]
        n_states=2,
        p_right_up=1e-20,
        p_right_down=0.1,
        reward_left=1.0,
        reward_right=1e22,
        initial_state=0,
    )
    adapter = AnalyticOracleReferenceAdapter(
        AnalyticOracleReferenceConfig.for_riverswim(environment, horizon=2)
    )
    runner = _river_runner(
        adapter,
        environment,
        lifecycle_id="control.oracle.conditioned-river",
        seed=5,
        horizon=2,
    )
    initial = runner.init()
    decision = adapter.current_decision(initial.agent_state)
    assert decision.proposed_action is not None
    assert decision.proposed_action.to_python() == 1


def test_discounted_sarsa_rejects_relabeled_lms_optimizer_state() -> None:
    adapter = _discounted_switching()
    runner = _switching_runner(
        adapter,
        SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
        lifecycle_id="control.discounted.optimizer-binding",
        horizon=1,
    )
    state = runner.init().agent_state
    assert isinstance(state, ReferenceLifeControlState)
    learner = state.agent_state
    assert isinstance(learner, SARSAState)
    inner = learner.learner_state
    first_group = inner.head_optimizer_states[0]
    altered_group = (
        LMSState(step_size=jnp.asarray(0.0, dtype=jnp.float32)),
        first_group[1],
    )
    altered_inner = inner.replace(  # type: ignore[attr-defined]
        head_optimizer_states=(altered_group, *inner.head_optimizer_states[1:])
    )
    altered = dataclasses.replace(
        state,
        agent_state=learner.replace(learner_state=altered_inner),  # type: ignore[attr-defined]
    )
    with pytest.raises(DecisionOwnershipError, match="step size differs"):
        adapter.validate_state(altered)


@pytest.mark.parametrize(
    "factory",
    (DifferentialSARSAReferenceConfig.for_switching, DiscountedSARSAReferenceConfig.for_switching),
)
def test_learning_controls_reject_increasing_epsilon_schedules(factory: object) -> None:
    with pytest.raises(ValueError, match="epsilon_end must not exceed"):
        factory(  # type: ignore[operator]
            SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
            epsilon_start=0.1,
            epsilon_end=0.2,
        )
