# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Contracts for bounded discovery/install/maintenance scheduling."""

from __future__ import annotations

import copy
import dataclasses
import functools

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.cumulant_option_installation import (
    CumulantOptionInstallation,
    CumulantOptionInstallationConfig,
    CumulantOptionLiveInputs,
)
from alberta_framework.core.cumulant_option_scheduler import (
    CUMULANT_OPTION_SCHEDULER_ASSESSMENT,
    CUMULANT_OPTION_SCHEDULER_GO_NO_GO_AUTHORITY,
    CUMULANT_OPTION_SCHEDULER_PROMOTION_AUTHORITY,
    CUMULANT_OPTION_SCHEDULER_RETIREMENT_AUTHORITY,
    CUMULANT_OPTION_SCHEDULER_SAFETY_AUTHORITY,
    CUMULANT_OPTION_SCHEDULER_SCIENTIFIC_PROMOTION_ALLOWED,
    CumulantOptionInstallationAuthorityReceipt,
    CumulantOptionScheduler,
    CumulantOptionSchedulerArmInputs,
    CumulantOptionSchedulerConfig,
    CumulantOptionSchedulerObservation,
    CumulantOptionSchedulerResult,
    CumulantOptionSchedulerState,
)
from alberta_framework.core.cumulant_subtask_discovery import (
    CUMULANT_SOURCE_CONTROLLABLE_EVENT,
    CUMULANT_SOURCE_FEATURE_CHANGE,
    CUMULANT_SOURCE_HAND_AUTHORED,
    CUMULANT_SOURCE_PREDICTION_BOTTLENECK,
    CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
    CumulantSubtaskDiscovery,
    CumulantSubtaskDiscoveryConfig,
)
from alberta_framework.core.option_lifecycle_audit import (
    OptionLifecycleAudit,
    OptionLifecycleAuditConfig,
    option_semantic_digest,
)
from alberta_framework.core.options import STOMPConfig
from alberta_framework.core.stomp_option_lifecycle import STOMPOptionLifecycleConfig

GENERATION = 3
SOURCE = jnp.asarray([0xA11CE, 0x51DE], dtype=jnp.uint32)
CONSUMER_SOURCE = option_semantic_digest({"source": "scheduler-test"})
REPRESENTATION = option_semantic_digest({"representation": "raw-two-v1"})
ISSUER = option_semantic_digest({"authority": "scheduler-test-owner"})
LIFECYCLE_ID = jnp.asarray([0xC011, 0xA17E], dtype=jnp.uint32)
HAND_IDENTITY = jnp.asarray([0xCAFE, 0xBEEF], dtype=jnp.uint32)


def _discovery_config() -> CumulantSubtaskDiscoveryConfig:
    return CumulantSubtaskDiscoveryConfig(
        raw_feature_dim=2,
        probe_feature_dim=1,
        n_actions=2,
        controllable_event_dim=1,
        transition_atom_dim=1,
        prediction_bottleneck_dim=1,
        option_budget=4,
        family_quotas=(1, 1, 1, 1),
        controllable_event_descriptors=(
            (CUMULANT_SOURCE_CONTROLLABLE_EVENT, 0, 1, 10),
        ),
        feature_change_descriptors=(
            (CUMULANT_SOURCE_FEATURE_CHANGE, 0, 1, 20),
        ),
        reward_transition_descriptors=(
            (CUMULANT_SOURCE_REWARD_TRANSITION_ATOM, 0, 1, 30),
        ),
        prediction_bottleneck_descriptors=(
            (CUMULANT_SOURCE_PREDICTION_BOTTLENECK, 0, 1, 40),
        ),
        incumbent_descriptors=((90, 0, 1, 900),),
        hand_comparator_descriptors=tuple(
            (CUMULANT_SOURCE_HAND_AUTHORED, index, 1, 100 + index)
            for index in range(4)
        ),
        hand_comparator_identity=(0xCAFE, 0xBEEF),
        reward_task_weights=(0.5,),
        model_task_weights=(0.5,),
        probe_step_size=0.1,
        shadow_step_size=1.0e-6,
        learnability_evidence_floor=1,
        controllability_evidence_floor_per_action=1,
        novelty_evidence_floor=1,
        contribution_evidence_floor=1,
        bottleneck_evidence_floor=1,
        learnability_threshold=0.0,
        baseline_variance_floor=1.0e-8,
        controllability_threshold=0.0,
        novelty_threshold=1.0e-8,
        contribution_threshold=0.0,
        bottleneck_epistemic_floor=0.0,
        bottleneck_progress_floor=0.0,
        bottleneck_aleatoric_ceiling=1.0,
        max_observations=16,
    )


@functools.cache
def _scheduler(
    *,
    proposal_period: int = 1,
    maintenance_period: int = 2,
    max_steps: int = 12,
    max_retry_streak: int = 3,
    max_maintenance_handoffs: int = 3,
    max_install_attempts: int | None = None,
) -> CumulantOptionScheduler:
    discovery = CumulantSubtaskDiscovery(_discovery_config())
    stomp = STOMPConfig.for_dynamic_option_template(
        subtask_specs=(),
        observation_dim=2,
        n_primitive_actions=2,
        epsilon_base=0.0,
        epsilon_option=0.0,
        option_planning_backups_per_step=1,
    )
    audit = OptionLifecycleAudit(
        OptionLifecycleAuditConfig(
            n_options=4,
            n_contexts=1,
            outcome_dim=6,
            fixed_horizon=2,
            maintenance_budget=1,
            signature_scales=(1.0,) * 11,
            initiation_opportunity_floor=1,
            completion_evidence_floor=1,
            model_error_evidence_floor=1,
            comparison_treatment_evidence_floor=1,
            comparison_primitive_evidence_floor=1,
            signature_evidence_floor_per_context=1,
            redundancy_shared_context_floor=1,
            max_planning_uses_per_observation=4,
            max_compute_cost_per_observation=10.0,
            max_observations=32,
        )
    )
    installation = CumulantOptionInstallation(
        discovery,
        stomp,
        audit,
        STOMPOptionLifecycleConfig(audit_enabled=False),
        CumulantOptionInstallationConfig(
            polarized_cumulant_threshold=0.5,
            max_option_steps=3,
            max_installations=4,
        ),
    )
    return CumulantOptionScheduler(
        installation,
        CumulantOptionSchedulerConfig(
            proposal_period=proposal_period,
            maintenance_period=maintenance_period,
            max_steps=max_steps,
            max_install_attempts=(
                min(4, max_steps)
                if max_install_attempts is None
                else max_install_attempts
            ),
            max_retry_streak=max_retry_streak,
            max_maintenance_handoffs=max_maintenance_handoffs,
        ),
    )


def _init(scheduler: CumulantOptionScheduler, seed: int = 0) -> CumulantOptionSchedulerState:
    return scheduler.init(
        jr.key(seed, impl="threefry2x32"),
        semantic_generation=GENERATION,
        source_digest=SOURCE,
        consumer_source_digest=CONSUMER_SOURCE,
        consumer_representation_digest=REPRESENTATION,
        lifecycle_id=LIFECYCLE_ID,
        authority_issuer_digest=ISSUER,
    )


def _snapshot(step: int) -> dict[str, jax.Array]:
    value = float(step)
    action = float(step % 2)
    return {
        "raw": jnp.asarray([value, value * value + 0.25], dtype=jnp.float32),
        "event": jnp.asarray([1.0 + 2.0 * action + 0.1 * value], dtype=jnp.float32),
        "atom": jnp.asarray([0.5 + action + 0.2 * value], dtype=jnp.float32),
        "bottleneck": jnp.asarray([2.0 - action + 0.15 * value], dtype=jnp.float32),
        "probe": jnp.asarray([1.0 + 0.1 * value], dtype=jnp.float32),
        "incumbent": jnp.asarray([20.0 + value], dtype=jnp.float32),
        "hand": jnp.arange(4, dtype=jnp.float32) + value,
    }


def _transition_id(step: int) -> jax.Array:
    return jnp.asarray([0xD15C, step], dtype=jnp.uint32)


def _transition(
    state: CumulantOptionSchedulerState,
    step: int,
) -> tuple[
    CumulantOptionSchedulerArmInputs,
    CumulantOptionSchedulerObservation,
    CumulantOptionLiveInputs,
]:
    current = _snapshot(step)
    successor = _snapshot(step + 1)
    intervention = jnp.zeros((2,), dtype=jnp.bool_).at[step % 2].set(True)
    arm = CumulantOptionSchedulerArmInputs(
        current_raw_features=current["raw"],
        current_raw_available=jnp.ones((2,), dtype=jnp.bool_),
        current_controllable_events=current["event"],
        current_controllable_events_available=jnp.ones((1,), dtype=jnp.bool_),
        current_transition_atoms=current["atom"],
        current_transition_atoms_available=jnp.full((1,), step > 0, dtype=jnp.bool_),
        current_bottleneck_values=current["bottleneck"],
        current_bottleneck_available=jnp.ones((1,), dtype=jnp.bool_),
        probe_features=current["probe"],
        current_incumbent_values=current["incumbent"],
        current_incumbent_available=jnp.ones((1,), dtype=jnp.bool_),
        current_hand_values=current["hand"],
        current_hand_available=jnp.ones((4,), dtype=jnp.bool_),
        hand_comparator_identity=HAND_IDENTITY,
        reward_base_predictions=jnp.zeros((1,), dtype=jnp.float32),
        model_base_predictions=jnp.zeros((1,), dtype=jnp.float32),
        action=jnp.asarray(step % 2, dtype=jnp.int32),
        behavior_propensity=jnp.asarray(0.5, dtype=jnp.float32),
        randomized=jnp.asarray(True, dtype=jnp.bool_),
        transition_id=_transition_id(step),
        semantic_generation=jnp.asarray(GENERATION, dtype=jnp.int32),
        source_digest=SOURCE,
    )
    observation = CumulantOptionSchedulerObservation(
        next_raw_features=successor["raw"],
        next_raw_available=jnp.ones((2,), dtype=jnp.bool_),
        next_controllable_events=successor["event"],
        next_controllable_events_available=jnp.ones((1,), dtype=jnp.bool_),
        next_transition_atoms=successor["atom"],
        next_transition_atoms_available=jnp.ones((1,), dtype=jnp.bool_),
        next_bottleneck_values=successor["bottleneck"],
        next_bottleneck_available=jnp.ones((1,), dtype=jnp.bool_),
        bottleneck_epistemic=jnp.asarray([0.5], dtype=jnp.float32),
        bottleneck_progress=jnp.asarray([0.25], dtype=jnp.float32),
        bottleneck_aleatoric=jnp.asarray([0.1], dtype=jnp.float32),
        bottleneck_evidence_available=jnp.ones((1,), dtype=jnp.bool_),
        randomized_action_evidence=intervention,
        next_incumbent_values=successor["incumbent"],
        next_incumbent_available=jnp.ones((1,), dtype=jnp.bool_),
        next_hand_values=successor["hand"],
        next_hand_available=jnp.ones((4,), dtype=jnp.bool_),
        hand_comparator_identity=HAND_IDENTITY,
        reward_targets=jnp.zeros((1,), dtype=jnp.float32),
        reward_targets_available=jnp.ones((1,), dtype=jnp.bool_),
        model_targets=jnp.zeros((1,), dtype=jnp.float32),
        model_targets_available=jnp.ones((1,), dtype=jnp.bool_),
        transition_id=_transition_id(step),
        semantic_generation=jnp.asarray(GENERATION, dtype=jnp.int32),
        source_digest=SOURCE,
    )
    live = CumulantOptionLiveInputs(
        raw_features=successor["raw"],
        raw_available=jnp.ones((2,), dtype=jnp.bool_),
        controllable_events=successor["event"],
        controllable_events_available=jnp.ones((1,), dtype=jnp.bool_),
        transition_atoms=successor["atom"],
        transition_atoms_available=jnp.ones((1,), dtype=jnp.bool_),
        bottleneck_values=successor["bottleneck"],
        bottleneck_available=jnp.ones((1,), dtype=jnp.bool_),
        semantic_generation=jnp.asarray(GENERATION, dtype=jnp.int32),
        source_digest=SOURCE,
        canonical_digest=state.discovery_state.canonical_digest,
        transition_id=_transition_id(step),
        state_observation_count=state.discovery_state.observation_count + 1,
    )
    return arm, observation, live


def _receipt(
    state: CumulantOptionSchedulerState,
    *,
    authorized: bool,
    issuer: jax.Array = ISSUER,
    authority_revision: int | None = None,
) -> CumulantOptionInstallationAuthorityReceipt:
    revision = (
        int(state.install_applied_words[1]) + 1
        if authority_revision is None
        else authority_revision
    )
    return CumulantOptionInstallationAuthorityReceipt(
        go_no_go_authorized=jnp.asarray(authorized, dtype=jnp.bool_),
        safety_boundary_authorized=jnp.asarray(authorized, dtype=jnp.bool_),
        semantic_generation=jnp.asarray(GENERATION, dtype=jnp.int32),
        source_digest=SOURCE,
        canonical_digest=state.discovery_state.canonical_digest,
        valid_from_step_words=jnp.asarray([0, 0], dtype=jnp.uint32),
        valid_through_step_words=jnp.asarray([0, 16], dtype=jnp.uint32),
        issuer_digest=issuer,
        authority_revision_words=jnp.asarray([0, revision], dtype=jnp.uint32),
    )


def _step(
    scheduler: CumulantOptionScheduler,
    state: CumulantOptionSchedulerState,
    step: int,
    *,
    authorized: bool,
) -> CumulantOptionSchedulerResult:
    arm_inputs, observation, live = _transition(state, step)
    arm = scheduler.arm(state, arm_inputs)
    return scheduler.observe(state, arm, observation, live, _receipt(state, authorized=authorized))


def _force_extended_action(
    scheduler: CumulantOptionScheduler,
    state: CumulantOptionSchedulerState,
    extended_action: int,
) -> CumulantOptionSchedulerState:
    installation = state.installation_state
    learner = installation.lifecycle_state.stomp_state.base_learner_state
    parameters = learner.head_params.replace(
        weights=tuple(jnp.zeros_like(weight) for weight in learner.head_params.weights),
        biases=tuple(
            jnp.full_like(bias, 1_000.0 if index == extended_action else 0.0)
            for index, bias in enumerate(learner.head_params.biases)
        ),
    )
    stomp = installation.lifecycle_state.stomp_state.replace(
        base_learner_state=learner.replace(head_params=parameters)
    )
    lifecycle = scheduler.installation.lifecycle.with_external_semantic_digests(
        installation.installed_semantic_digests
    )._with_checksum(
        dataclasses.replace(installation.lifecycle_state, stomp_state=stomp)
    )
    installation = scheduler.installation._with_checksum(
        dataclasses.replace(installation, lifecycle_state=lifecycle)
    )
    return scheduler._with_checksum(
        dataclasses.replace(state, installation_state=installation)
    )


@pytest.mark.unit
def test_config_clock_resources_and_authority_are_strict() -> None:
    config = CumulantOptionSchedulerConfig(
        proposal_period=2,
        maintenance_period=3,
        max_steps=6,
        max_install_attempts=2,
        max_retry_streak=2,
        max_maintenance_handoffs=2,
    )
    assert CumulantOptionSchedulerConfig.from_config(config.to_config()) == config
    with pytest.raises(ValueError, match="proposal_period"):
        dataclasses.replace(config, proposal_period=7)
    payload = config.to_config()
    payload["retirement_authority"] = True
    with pytest.raises(ValueError, match="retirement_authority"):
        CumulantOptionSchedulerConfig.from_config(payload)

    scheduler = _scheduler(proposal_period=2, maintenance_period=3, max_steps=6)
    state = _init(scheduler)
    assert bool(scheduler.state_valid(state))
    eager = scheduler.schedule_clock(jnp.asarray([0, 0], jnp.uint32), jnp.asarray(False))
    compiled = jax.jit(scheduler.schedule_clock)(
        jnp.asarray([0, 0], jnp.uint32), jnp.asarray(False)
    )
    chex.assert_trees_all_equal(eager, compiled)
    assert bool(eager.proposal_due)
    assert not bool(eager.maintenance_due_after_step)

    def scan_body(words: jax.Array, _: jax.Array) -> tuple[jax.Array, jax.Array]:
        clock = scheduler.schedule_clock(words, jnp.asarray(False))
        return clock.next_step_words, clock.proposal_due

    final_words, due = jax.jit(
        lambda: jax.lax.scan(
            scan_body,
            jnp.asarray([0, 0], dtype=jnp.uint32),
            jnp.arange(5, dtype=jnp.int32),
        )
    )()
    np.testing.assert_array_equal(final_words, [0, 5])
    np.testing.assert_array_equal(due, [True, False, True, False, True])

    budget = scheduler.resource_budget(state)
    leaves = jax.tree_util.tree_leaves(state)
    expected_bytes = 0
    for leaf in leaves:
        leaf_array = jnp.asarray(leaf)
        array = (
            jr.key_data(leaf_array)
            if jax.dtypes.issubdtype(leaf_array.dtype, jax.dtypes.prng_key)
            else leaf_array
        )
        expected_bytes += int(array.size * array.dtype.itemsize)
    assert budget.persistent_state_nbytes == expected_bytes
    assert budget.pending_proposal_slots == 0
    assert budget.exact_counter_words == 14
    assert budget.max_materializations_per_step == 2
    assert budget.assessment == CUMULANT_OPTION_SCHEDULER_ASSESSMENT
    assert not CUMULANT_OPTION_SCHEDULER_GO_NO_GO_AUTHORITY
    assert not CUMULANT_OPTION_SCHEDULER_SAFETY_AUTHORITY
    assert not CUMULANT_OPTION_SCHEDULER_RETIREMENT_AUTHORITY
    assert not CUMULANT_OPTION_SCHEDULER_PROMOTION_AUTHORITY
    assert not CUMULANT_OPTION_SCHEDULER_SCIENTIFIC_PROMOTION_ALLOWED


@pytest.mark.unit
def test_live_successor_mismatch_is_an_atomic_noop() -> None:
    scheduler = _scheduler()
    state = _init(scheduler, 1)
    arm_inputs, observation, live = _transition(state, 0)
    arm = scheduler.arm(state, arm_inputs)
    mismatched = dataclasses.replace(live, raw_features=live.raw_features.at[0].add(1.0))
    result = scheduler.observe(
        state,
        arm,
        observation,
        mismatched,
        _receipt(state, authorized=False),
    )
    assert not bool(result.transaction_applied)
    chex.assert_trees_all_equal(result.state, state)


@pytest.mark.integration
def test_cold_mask_retry_then_complete_authorized_install_and_checkpoint() -> None:
    scheduler = _scheduler()
    state = _init(scheduler, 2)
    last = None
    for step in range(8):
        last = _step(scheduler, state, step, authorized=False)
        assert bool(last.transaction_applied)
        assert bool(last.cold_option_mask_active)
        assert not bool(last.installation_applied)
        assert not bool(last.retirement_applied)
        state = last.state
        if bool(last.proposal_ready):
            break
    assert last is not None and bool(last.proposal_ready)
    assert bool(last.authorization_requested)
    assert bool(last.retry_scheduled)
    assert not bool(last.materialization.tail_available.any())
    assert not bool(state.installation_state.installed)

    first_retry = _step(scheduler, state, step + 1, authorized=False)
    assert bool(first_retry.retry_scheduled)
    state = first_retry.state
    exhausted = _step(scheduler, state, step + 2, authorized=False)
    assert bool(exhausted.retry_exhausted_this_step)
    assert not bool(exhausted.state.retry_due)
    assert not bool(exhausted.state.installation_state.installed)
    state = exhausted.state
    last = exhausted
    step += 2

    cold_start = scheduler.start_control(state, last.materialization)
    assert cold_start.applied and cold_start.cold_path
    assert cold_start.control is not None
    assert cold_start.control.lifecycle_result is not None
    assert int(cold_start.control.lifecycle_result.primitive_action) < 2
    state = cold_start.state

    installed = _step(scheduler, state, step + 1, authorized=True)
    assert bool(installed.transaction_applied)
    assert bool(installed.installation_attempted)
    assert bool(installed.installation_applied)
    assert bool(installed.scheduler_rng_advanced)
    assert not bool(installed.cold_option_mask_active)
    assert bool(installed.state.installation_state.installed)
    assert bool(installed.materialization.tail_available.all())
    np.testing.assert_array_equal(
        installed.state.installation_state.installed_bundle.transition_id,
        _transition_id(step + 1),
    )

    payload = scheduler.checkpoint_payload(installed.state)
    restored = scheduler.restore_checkpoint(
        copy.deepcopy(payload),
        expected_semantic_generation=GENERATION,
        expected_source_digest=SOURCE,
        expected_consumer_source_digest=CONSUMER_SOURCE,
        expected_consumer_representation_digest=REPRESENTATION,
        expected_lifecycle_id=LIFECYCLE_ID,
        expected_authority_issuer_digest=ISSUER,
        expected_installed_bundle=installed.state.installation_state.installed_bundle,
    )
    chex.assert_trees_all_equal(restored, installed.state)
    with pytest.raises(ValueError, match="stale"):
        scheduler.restore_checkpoint(
            payload,
            expected_semantic_generation=GENERATION,
            expected_source_digest=SOURCE,
            expected_consumer_source_digest=CONSUMER_SOURCE,
            expected_consumer_representation_digest=REPRESENTATION,
            expected_lifecycle_id=LIFECYCLE_ID,
            expected_authority_issuer_digest=ISSUER + jnp.uint32(1),
            expected_installed_bundle=installed.state.installation_state.installed_bundle,
        )

    # Equal revisions are replay, not renewed authority.  The real live path
    # still advances, but the installed provenance cannot refresh.
    arm_inputs, observation, live = _transition(installed.state, step + 2)
    arm = scheduler.arm(installed.state, arm_inputs)
    replay = scheduler.observe(
        installed.state,
        arm,
        observation,
        live,
        _receipt(
            installed.state,
            authorized=True,
            authority_revision=1,
        ),
    )
    assert bool(replay.transaction_applied)
    assert not bool(replay.authority_receipt_valid)
    assert bool(replay.authorization_requested)
    assert not bool(replay.installation_attempted)
    np.testing.assert_array_equal(
        replay.state.installation_state.installed_bundle.transition_id,
        installed.state.installation_state.installed_bundle.transition_id,
    )


@pytest.mark.integration
def test_active_option_defers_without_queue_then_fresh_retry_and_maintenance_cap() -> None:
    scheduler = _scheduler(maintenance_period=1, max_maintenance_handoffs=2)
    state = _init(scheduler, 3)
    installed_result = None
    step = 0
    while step < 8:
        installed_result = _step(scheduler, state, step, authorized=True)
        assert bool(installed_result.transaction_applied)
        state = installed_result.state
        if bool(installed_result.installation_applied):
            break
        step += 1
    assert installed_result is not None and bool(installed_result.installation_applied)
    assert not bool(installed_result.retirement_handoff.retirement_authority)
    assert not bool(installed_result.retirement_handoff.go_no_go_authority)

    state = _force_extended_action(scheduler, state, 2)
    step += 1
    live_for_start = _step(scheduler, state, step, authorized=False)
    assert bool(live_for_start.transaction_applied)
    state = live_for_start.state
    started = scheduler.start_control(state, live_for_start.materialization)
    assert started.applied and not started.cold_path
    assert int(started.state.installation_state.lifecycle_state.stomp_state.executing_option) == 0
    # The next base decision must be primitive so natural termination does not
    # immediately select the same option again.
    state = _force_extended_action(scheduler, started.state, 0)

    step += 1
    deferred = _step(scheduler, state, step, authorized=True)
    assert bool(deferred.transaction_applied)
    assert not bool(deferred.quiescent_boundary)
    assert bool(deferred.installation_deferred)
    assert bool(deferred.retry_scheduled)
    assert not bool(deferred.installation_attempted)
    assert int(deferred.state.install_attempt_words[1]) == int(
        state.install_attempt_words[1]
    )
    state = deferred.state

    base = scheduler.installation.stomp_agent._base_learner
    before_q = base.predict(
        state.installation_state.lifecycle_state.stomp_state.base_learner_state,
        deferred.materialization.observation,
    )
    assert int(jnp.argmax(before_q)) == 0

    ended = scheduler.update_control(
        state,
        deferred.materialization,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        enable_planning=False,
    )
    assert ended.applied
    assert ended.control is not None and ended.control.lifecycle_result is not None
    assert bool(ended.control.lifecycle_result.option_terminated)
    assert int(ended.state.installation_state.lifecycle_state.stomp_state.executing_option) == -1
    state = ended.state

    step += 1
    retried = _step(scheduler, state, step, authorized=True)
    assert bool(retried.transaction_applied)
    assert bool(retried.proposal_due)
    assert bool(retried.installation_attempted)
    assert bool(retried.installation_applied)
    assert bool(retried.scheduler_rng_advanced)
    np.testing.assert_array_equal(
        retried.state.installation_state.installed_bundle.transition_id,
        _transition_id(step),
    )
    assert int(retried.state.maintenance_handoff_words[1]) == 2
    assert not bool(retried.maintenance_handoff_emitted)
    assert not bool(retried.retirement_applied)


@pytest.mark.integration
def test_exact_schedule_capacity_is_fail_stop_but_control_remains_available() -> None:
    scheduler = _scheduler(max_steps=4, max_maintenance_handoffs=0)
    state = _init(scheduler, 4)
    last = None
    for step in range(4):
        last = _step(scheduler, state, step, authorized=True)
        assert bool(last.transaction_applied)
        state = last.state
    assert last is not None
    assert bool(state.schedule_unavailable)
    np.testing.assert_array_equal(state.step_words, [0, 4])
    assert bool(state.installation_state.installed)

    arm_inputs, observation, live = _transition(state, 4)
    arm = scheduler.arm(state, arm_inputs)
    capped = scheduler.observe(
        state,
        arm,
        observation,
        live,
        _receipt(state, authorized=True),
    )
    assert not bool(capped.transaction_applied)
    chex.assert_trees_all_equal(capped.state, state)

    # Scheduling capacity cannot veto the already-installed real controller.
    started = scheduler.start_control(state, last.materialization)
    assert started.applied
    assert not started.cold_path
    np.testing.assert_array_equal(started.state.step_words, [0, 4])


@pytest.mark.integration
def test_zero_install_attempt_budget_is_explicit_and_never_retries() -> None:
    scheduler = _scheduler(
        max_install_attempts=0,
        max_maintenance_handoffs=0,
    )
    state = _init(scheduler, 5)
    result = None
    for step in range(8):
        result = _step(scheduler, state, step, authorized=True)
        assert bool(result.transaction_applied)
        state = result.state
        if bool(result.proposal_ready):
            break
    assert result is not None and bool(result.proposal_ready)
    assert not bool(result.installation_attempt_capacity_available)
    assert bool(result.installation_attempt_capacity_exhausted)
    assert not bool(result.installation_attempted)
    assert not bool(result.installation_deferred)
    assert not bool(result.retry_scheduled)
    assert not bool(state.installation_state.installed)
    np.testing.assert_array_equal(state.install_attempt_words, [0, 0])
