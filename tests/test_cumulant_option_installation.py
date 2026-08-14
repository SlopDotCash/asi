# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Unit contracts for strict discovered-cumulant option installation."""

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
    CUMULANT_OPTION_INSTALLATION_ASSESSMENT,
    CUMULANT_OPTION_INSTALLATION_AUTONOMOUS_DISCOVERY_CLAIM,
    CUMULANT_OPTION_INSTALLATION_BENEFIT_CLAIM,
    CUMULANT_OPTION_INSTALLATION_CONTROL_HOST_ONLY,
    CUMULANT_OPTION_INSTALLATION_EVIDENCE_AUTHORITY,
    CUMULANT_OPTION_INSTALLATION_OUTPUT_WRITES,
    CUMULANT_OPTION_INSTALLATION_PROMOTION_AUTHORITY,
    CUMULANT_OPTION_INSTALLATION_SCIENTIFIC_PROMOTION_ALLOWED,
    CumulantOptionInstallation,
    CumulantOptionInstallationConfig,
    CumulantOptionInstallationState,
    CumulantOptionLiveInputs,
)
from alberta_framework.core.cumulant_subtask_discovery import (
    CUMULANT_SOURCE_CONTROLLABLE_EVENT,
    CUMULANT_SOURCE_FEATURE_CHANGE,
    CUMULANT_SOURCE_HAND_AUTHORED,
    CUMULANT_SOURCE_PREDICTION_BOTTLENECK,
    CUMULANT_SOURCE_REWARD_TRANSITION_ATOM,
    CumulantSubtaskDiscovery,
    CumulantSubtaskDiscoveryConfig,
    CumulantSubtaskProposalBundle,
)
from alberta_framework.core.option_lifecycle_audit import (
    OptionLifecycleAudit,
    OptionLifecycleAuditConfig,
    option_semantic_digest,
)
from alberta_framework.core.options import STOMPConfig
from alberta_framework.core.stomp_option_lifecycle import STOMPOptionLifecycleConfig

pytestmark = [pytest.mark.unit, pytest.mark.slow]

GENERATION = 3
SOURCE = jnp.asarray([0xA11CE, 0x51DE], dtype=jnp.uint32)
CANONICAL = jnp.arange(1, 33, dtype=jnp.uint8)
CONSUMER_SOURCE = option_semantic_digest({"source": "installer-test"})
REPRESENTATION = option_semantic_digest({"representation": "raw-two-v1"})
LIFECYCLE_ID = jnp.asarray([0xC011, 0xA17E], dtype=jnp.uint32)


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
            (CUMULANT_SOURCE_CONTROLLABLE_EVENT, 0, -1, 11),
        ),
        feature_change_descriptors=(
            (CUMULANT_SOURCE_FEATURE_CHANGE, 0, 1, 20),
            (CUMULANT_SOURCE_FEATURE_CHANGE, 1, -1, 21),
        ),
        reward_transition_descriptors=(
            (CUMULANT_SOURCE_REWARD_TRANSITION_ATOM, 0, 1, 30),
            (CUMULANT_SOURCE_REWARD_TRANSITION_ATOM, 0, -1, 31),
        ),
        prediction_bottleneck_descriptors=(
            (CUMULANT_SOURCE_PREDICTION_BOTTLENECK, 0, 1, 40),
            (CUMULANT_SOURCE_PREDICTION_BOTTLENECK, 0, -1, 41),
        ),
        incumbent_descriptors=((90, 0, 1, 900),),
        hand_comparator_descriptors=tuple(
            (CUMULANT_SOURCE_HAND_AUTHORED, index, 1, 100 + index)
            for index in range(4)
        ),
        hand_comparator_identity=(0xCAFE, 0xBEEF),
        reward_task_weights=(0.5,),
        model_task_weights=(0.5,),
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
        max_observations=32,
    )


@functools.cache
def _composition(
    *,
    epsilon: float = 0.0,
    max_installations: int = 2,
    audit_enabled: bool = False,
    reserved_observation_suffix: int = 0,
) -> CumulantOptionInstallation:
    discovery = CumulantSubtaskDiscovery(_discovery_config())
    stomp = STOMPConfig.for_dynamic_option_template(
        subtask_specs=(),
        observation_dim=2 + reserved_observation_suffix,
        n_primitive_actions=2,
        epsilon_base=epsilon,
        epsilon_option=0.0,
        option_planning_backups_per_step=1,
    )
    audit = OptionLifecycleAudit(
        OptionLifecycleAuditConfig(
            n_options=4,
            n_contexts=1,
            outcome_dim=6 + reserved_observation_suffix,
            fixed_horizon=2,
            maintenance_budget=1,
            signature_scales=(1.0,) * (11 + reserved_observation_suffix),
            initiation_opportunity_floor=1,
            completion_evidence_floor=1,
            model_error_evidence_floor=1,
            comparison_treatment_evidence_floor=1,
            comparison_primitive_evidence_floor=1,
            signature_evidence_floor_per_context=1,
            redundancy_shared_context_floor=1,
            max_planning_uses_per_observation=4,
            max_compute_cost_per_observation=10.0,
            max_observations=64,
        )
    )
    return CumulantOptionInstallation(
        discovery,
        stomp,
        audit,
        STOMPOptionLifecycleConfig(audit_enabled=audit_enabled),
        CumulantOptionInstallationConfig(
            polarized_cumulant_threshold=0.5,
            max_option_steps=3,
            max_installations=max_installations,
        ),
    )


def test_reserved_observation_suffix_is_zero_in_standalone_materializations() -> None:
    composition = _composition(reserved_observation_suffix=2)
    assert composition.stomp_agent.config.observation_dim == 8
    assert tuple(spec.feature_index for spec in composition.subtask_specs) == (2, 3, 4, 5)

    prior = _inputs(1, raw=(0.1, 0.2), event=0.0, atom=0.0, bottleneck=0.0)
    cold = composition.materialize_cold(_init(composition), prior)
    assert bool(cold.applied)
    np.testing.assert_array_equal(cold.materialization.observation[:2], prior.raw_features)
    np.testing.assert_array_equal(cold.materialization.observation[2:], 0.0)

    current = _inputs(2, raw=(0.3, 0.4), event=0.25, atom=0.35, bottleneck=0.45)
    installed = composition.install(
        cold.state,
        _bundle(composition, current, prior.raw_features),
        jr.key(0x5155),
        inputs=current,
    )
    assert bool(installed.applied)
    np.testing.assert_array_equal(
        installed.materialization.observation[:2],
        current.raw_features,
    )
    np.testing.assert_array_equal(
        installed.materialization.observation[2:6],
        installed.materialization.tail_values,
    )
    np.testing.assert_array_equal(installed.materialization.observation[6:], 0.0)

    externally_populated = dataclasses.replace(
        installed.materialization,
        observation=installed.materialization.observation.at[6:].set(
            jnp.asarray((0.5, -0.25), dtype=jnp.float32)
        ),
    )
    refused = composition.start(installed.state, externally_populated)
    assert not refused.applied
    chex.assert_trees_all_equal(refused.state, installed.state)


def test_template_observation_width_cannot_be_smaller_than_raw_prefix() -> None:
    base = _composition()
    undersized = dataclasses.replace(
        base.stomp_agent.config,
        subtask_specs=(),
        observation_dim=1,
    )
    with pytest.raises(ValueError, match="observation_dim.*raw_feature_dim"):
        CumulantOptionInstallation(
            base.discovery,
            undersized,
            base.lifecycle.audit,
            base._lifecycle_config,
            base.config,
        )


def _init(composition: CumulantOptionInstallation, seed: int = 0):
    return composition.init(
        jr.key(seed),
        consumer_source_digest=CONSUMER_SOURCE,
        consumer_representation_digest=REPRESENTATION,
        lifecycle_id=LIFECYCLE_ID,
    )


def _inputs(
    step: int,
    *,
    raw: tuple[float, float],
    event: float,
    atom: float,
    bottleneck: float,
) -> CumulantOptionLiveInputs:
    return CumulantOptionLiveInputs(
        raw_features=jnp.asarray(raw, dtype=jnp.float32),
        raw_available=jnp.ones((2,), dtype=jnp.bool_),
        controllable_events=jnp.asarray([event], dtype=jnp.float32),
        controllable_events_available=jnp.ones((1,), dtype=jnp.bool_),
        transition_atoms=jnp.asarray([atom], dtype=jnp.float32),
        transition_atoms_available=jnp.ones((1,), dtype=jnp.bool_),
        bottleneck_values=jnp.asarray([bottleneck], dtype=jnp.float32),
        bottleneck_available=jnp.ones((1,), dtype=jnp.bool_),
        semantic_generation=jnp.asarray(GENERATION, dtype=jnp.int32),
        source_digest=SOURCE,
        canonical_digest=CANONICAL,
        transition_id=jnp.asarray([0xD15C, step], dtype=jnp.uint32),
        state_observation_count=jnp.asarray(step, dtype=jnp.int32),
    )


def _descriptor_values(
    descriptors: tuple[tuple[int, int, int, int], ...],
    previous_raw: jax.Array,
    inputs: CumulantOptionLiveInputs,
) -> jax.Array:
    values: list[jax.Array] = []
    for family, index, polarity, _tag in descriptors:
        if family == CUMULANT_SOURCE_CONTROLLABLE_EVENT:
            value = inputs.controllable_events[index]
        elif family == CUMULANT_SOURCE_FEATURE_CHANGE:
            value = inputs.raw_features[index] - previous_raw[index]
        elif family == CUMULANT_SOURCE_REWARD_TRANSITION_ATOM:
            value = inputs.transition_atoms[index]
        else:
            value = inputs.bottleneck_values[index]
        values.append(value * jnp.asarray(polarity, dtype=jnp.float32))
    return jnp.stack(values).astype(jnp.float32)


def _bundle(
    composition: CumulantOptionInstallation,
    inputs: CumulantOptionLiveInputs,
    previous_raw: jax.Array,
    indices: tuple[int, int, int, int] = (0, 2, 4, 6),
    *,
    values_override: jax.Array | None = None,
) -> CumulantSubtaskProposalBundle:
    candidates = composition.discovery.config.candidate_descriptors
    descriptors = tuple(candidates[index] for index in indices)
    cumulants = (
        _descriptor_values(descriptors, previous_raw, inputs)
        if values_override is None
        else values_override
    )
    descriptor_array = jnp.asarray(descriptors, dtype=jnp.int32)
    # This test-only fixture seals the same complete payload emitted by observe().
    return composition.discovery._make_bundle(
        ready=jnp.asarray(True, dtype=jnp.bool_),
        cohort_id=-1,
        indices=jnp.asarray(indices, dtype=jnp.int32),
        family_ids=descriptor_array[:, 0],
        descriptors=descriptor_array,
        scores=jnp.arange(4, dtype=jnp.float32),
        cumulants=cumulants,
        semantic_generation=inputs.semantic_generation,
        source_digest=inputs.source_digest,
        canonical_digest=inputs.canonical_digest,
        transition_id=inputs.transition_id,
        state_observation_count=inputs.state_observation_count,
    )


def _force_extended_action(
    composition: CumulantOptionInstallation,
    state: CumulantOptionInstallationState,
    extended_action: int,
) -> CumulantOptionInstallationState:
    learner = state.lifecycle_state.stomp_state.base_learner_state
    total = composition.stomp_agent.config.n_total_actions
    params = learner.head_params.replace(
        weights=tuple(jnp.zeros_like(weight) for weight in learner.head_params.weights),
        biases=tuple(
            jnp.full_like(bias, 1000.0 if index == extended_action else -1000.0)
            for index, bias in enumerate(learner.head_params.biases)
        ),
    )
    stomp = state.lifecycle_state.stomp_state.replace(
        base_learner_state=learner.replace(head_params=params)
    )
    bound = composition.lifecycle.with_external_semantic_digests(
        state.installed_semantic_digests
    )
    lifecycle = bound._with_checksum(
        dataclasses.replace(state.lifecycle_state, stomp_state=stomp)
    )
    assert total == len(learner.head_params.biases)
    return composition._with_checksum(
        dataclasses.replace(state, lifecycle_state=lifecycle)
    )


def test_cold_path_masks_huge_option_head_and_advances_raw_prior() -> None:
    composition = _composition(epsilon=0.0)
    state = _force_extended_action(composition, _init(composition), 2)
    assert bool(composition.lifecycle.audit.state_valid(state.lifecycle_state.audit_state))
    first = _inputs(1, raw=(0.1, 0.2), event=0.0, atom=0.0, bottleneck=0.0)
    cold = composition.materialize_cold(state, first)
    assert bool(cold.applied)
    assert not np.any(cold.materialization.tail_available)
    np.testing.assert_array_equal(cold.materialization.observation[2:], 0.0)

    started = composition.start_cold(cold.state, cold.materialization)
    assert started.applied
    assert started.lifecycle_result is not None
    assert int(started.lifecycle_result.primitive_action) < 2
    assert int(started.state.lifecycle_state.stomp_state.executing_option) == -1

    second = _inputs(2, raw=(0.3, 0.4), event=0.2, atom=0.3, bottleneck=0.4)
    advanced = composition.materialize_cold(started.state, second)
    assert bool(advanced.applied)
    updated = composition.update_cold(
        advanced.state,
        advanced.materialization,
        jnp.asarray(0.25, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert updated.applied
    assert updated.lifecycle_result is not None
    assert int(updated.lifecycle_result.planning_usage.sum()) == 0
    assert int(updated.state.lifecycle_state.stomp_state.executing_option) == -1

    cutover_inputs = _inputs(
        3,
        raw=(0.5, 0.6),
        event=0.25,
        atom=0.35,
        bottleneck=0.45,
    )
    cutover_bundle = _bundle(
        composition,
        cutover_inputs,
        second.raw_features,
    )
    cutover = composition.install(
        updated.state,
        cutover_bundle,
        jr.key(41),
        inputs=cutover_inputs,
    )
    assert bool(cutover.applied)
    assert bool(cutover.state.lifecycle_state.started)
    continuing_inputs = _inputs(
        4,
        raw=(0.7, 0.8),
        event=0.2,
        atom=0.3,
        bottleneck=0.4,
    )
    continuing = composition.materialize_live(cutover.state, continuing_inputs)
    cutover_update = composition.update(
        continuing.state,
        continuing.materialization,
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert cutover_update.applied


@pytest.mark.integration
def test_install_live_materialization_and_real_threshold_termination() -> None:
    composition = _composition()
    state = _init(composition)
    prior = _inputs(1, raw=(0.1, 0.2), event=0.0, atom=0.0, bottleneck=0.0)
    cold = composition.materialize_cold(state, prior)
    current = _inputs(2, raw=(0.3, 0.4), event=0.25, atom=0.35, bottleneck=0.45)
    bundle = _bundle(composition, current, prior.raw_features)
    installed = composition.install(cold.state, bundle, jr.key(8), inputs=current)
    assert bool(installed.applied)
    assert bool(installed.semantics_changed)
    assert int(installed.state.installation_count) == 1
    np.testing.assert_allclose(
        installed.materialization.tail_values,
        [0.25, 0.2, 0.35, 0.45],
    )
    assert all(spec.pseudo_reward_scale == 1.0 for spec in composition.subtask_specs)
    assert tuple(spec.feature_index for spec in composition.subtask_specs) == (2, 3, 4, 5)

    state = _force_extended_action(composition, installed.state, 2)
    live = _inputs(3, raw=(0.5, 0.7), event=0.1, atom=0.2, bottleneck=0.3)
    rematerialized = composition.materialize_live(state, live)
    assert bool(rematerialized.applied)
    np.testing.assert_allclose(
        rematerialized.materialization.tail_values,
        [0.1, 0.2, 0.2, 0.3],
    )
    started = composition.start(rematerialized.state, rematerialized.materialization)
    assert started.applied
    assert int(started.state.lifecycle_state.stomp_state.executing_option) == 0

    terminal = _inputs(4, raw=(0.6, 0.8), event=0.75, atom=0.1, bottleneck=0.2)
    next_live = composition.materialize_live(started.state, terminal)
    with pytest.raises(ValueError, match="not bound"):
        composition.update(
            next_live.state,
            next_live.materialization,
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.9, dtype=jnp.float32),
            decision_observation=next_live.materialization.observation,
        )
    updated = composition.update(
        next_live.state,
        next_live.materialization,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert updated.applied
    assert updated.lifecycle_result is not None
    assert bool(updated.lifecycle_result.option_terminated)
    assert float(updated.lifecycle_result.pseudo_reward) == pytest.approx(0.75)


def test_changed_install_defers_exactly_during_option_then_applies_quiescent() -> None:
    composition = _composition()
    prior = _inputs(1, raw=(0.1, 0.2), event=0.0, atom=0.0, bottleneck=0.0)
    cold = composition.materialize_cold(_init(composition), prior)
    first_inputs = _inputs(2, raw=(0.3, 0.4), event=0.2, atom=0.3, bottleneck=0.4)
    installed = composition.install(
        cold.state,
        _bundle(composition, first_inputs, prior.raw_features),
        jr.key(50),
        inputs=first_inputs,
    )
    state = _force_extended_action(composition, installed.state, 2)
    active_inputs = _inputs(3, raw=(0.5, 0.6), event=0.1, atom=0.2, bottleneck=0.3)
    active_live = composition.materialize_live(state, active_inputs)
    started = composition.start(active_live.state, active_live.materialization)
    assert started.applied
    assert int(started.state.lifecycle_state.stomp_state.executing_option) == 0

    deferred_inputs = _inputs(4, raw=(0.7, 0.8), event=0.75, atom=0.1, bottleneck=0.2)
    deferred_bundle = _bundle(
        composition,
        deferred_inputs,
        active_inputs.raw_features,
        (0, 3, 4, 6),
    )
    deferred = composition.install(
        started.state,
        deferred_bundle,
        jr.key(51),
        inputs=deferred_inputs,
    )
    assert bool(deferred.transaction_valid)
    assert bool(deferred.deferred)
    assert not bool(deferred.applied)
    assert bool(deferred.live_policy_rng_preserved)
    chex.assert_trees_all_equal(deferred.state, started.state)

    ending_state = _force_extended_action(composition, started.state, 0)
    old_live = composition.materialize_live(ending_state, deferred_inputs)
    ended = composition.update(
        old_live.state,
        old_live.materialization,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert ended.applied
    assert ended.lifecycle_result is not None
    assert bool(ended.lifecycle_result.option_terminated)
    assert int(ended.state.lifecycle_state.stomp_state.executing_option) == -1

    quiescent_inputs = _inputs(5, raw=(0.9, 1.0), event=0.2, atom=0.3, bottleneck=0.4)
    quiescent_bundle = _bundle(
        composition,
        quiescent_inputs,
        deferred_inputs.raw_features,
        (0, 3, 4, 6),
    )
    applied = composition.install(
        ended.state,
        quiescent_bundle,
        jr.key(52),
        inputs=quiescent_inputs,
    )
    assert bool(applied.quiescent)
    assert bool(applied.applied)
    np.testing.assert_array_equal(applied.reset_slots, [False, True, False, False])


def test_changed_install_defers_during_comparator_trial_then_applies() -> None:
    composition = _composition(audit_enabled=True)
    prior = _inputs(1, raw=(0.1, 0.2), event=0.0, atom=0.0, bottleneck=0.0)
    cold = composition.materialize_cold(_init(composition), prior)
    first_inputs = _inputs(2, raw=(0.3, 0.4), event=0.2, atom=0.3, bottleneck=0.4)
    installed = composition.install(
        cold.state,
        _bundle(composition, first_inputs, prior.raw_features),
        jr.key(60),
        inputs=first_inputs,
    )
    primitive = _force_extended_action(composition, installed.state, 0)
    start_inputs = _inputs(3, raw=(0.5, 0.6), event=0.1, atom=0.2, bottleneck=0.3)
    start_live = composition.materialize_live(primitive, start_inputs)
    started = composition.start(start_live.state, start_live.materialization)
    assert started.applied
    assert int(started.state.lifecycle_state.stomp_state.executing_option) == -1

    trial_inputs = _inputs(4, raw=(0.6, 0.7), event=0.2, atom=0.3, bottleneck=0.4)
    trial_live = composition.materialize_live(started.state, trial_inputs)
    trial_update = composition.update(
        trial_live.state,
        trial_live.materialization,
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
        context=0,
        idle_candidate_option=1,
        idle_initiation_eligible=True,
        comparator_randomized=True,
        treatment_propensity=0.5,
    )
    assert trial_update.applied
    assert bool(trial_update.state.lifecycle_state.audit_state.trial_active)

    deferred_inputs = _inputs(5, raw=(0.7, 0.8), event=0.3, atom=0.2, bottleneck=0.1)
    deferred_bundle = _bundle(
        composition,
        deferred_inputs,
        trial_inputs.raw_features,
        (0, 3, 4, 6),
    )
    deferred = composition.install(
        trial_update.state,
        deferred_bundle,
        jr.key(61),
        inputs=deferred_inputs,
    )
    assert bool(deferred.transaction_valid)
    assert bool(deferred.deferred)
    assert not bool(deferred.applied)
    chex.assert_trees_all_equal(deferred.state, trial_update.state)

    trial_finish_live = composition.materialize_live(trial_update.state, deferred_inputs)
    trial_finished = composition.update(
        trial_finish_live.state,
        trial_finish_live.materialization,
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
        context=0,
        idle_candidate_option=1,
    )
    assert trial_finished.applied
    assert not bool(trial_finished.state.lifecycle_state.audit_state.trial_active)

    quiescent_inputs = _inputs(6, raw=(0.8, 0.9), event=0.2, atom=0.3, bottleneck=0.4)
    quiescent_bundle = _bundle(
        composition,
        quiescent_inputs,
        deferred_inputs.raw_features,
        (0, 3, 4, 6),
    )
    applied = composition.install(
        trial_finished.state,
        quiescent_bundle,
        jr.key(62),
        inputs=quiescent_inputs,
    )
    assert bool(applied.quiescent)
    assert bool(applied.applied)


def test_same_descriptors_refresh_without_rebind_changed_slot_resets_and_exhausts() -> None:
    composition = _composition(max_installations=2)
    prior = _inputs(1, raw=(0.1, 0.2), event=0.0, atom=0.0, bottleneck=0.0)
    cold = composition.materialize_cold(_init(composition), prior)
    first_inputs = _inputs(2, raw=(0.3, 0.4), event=0.25, atom=0.35, bottleneck=0.45)
    first = composition.install(
        cold.state,
        _bundle(composition, first_inputs, prior.raw_features),
        jr.key(3),
        inputs=first_inputs,
    )
    assert bool(first.applied)

    refresh_inputs = _inputs(3, raw=(0.5, 0.6), event=0.2, atom=0.3, bottleneck=0.4)
    refresh_bundle = _bundle(
        composition,
        refresh_inputs,
        first_inputs.raw_features,
    )
    refreshed = composition.install(
        first.state,
        refresh_bundle,
        jr.key(4),
        inputs=refresh_inputs,
    )
    assert bool(refreshed.applied)
    assert bool(refreshed.provenance_refreshed)
    assert not bool(refreshed.semantics_changed)
    assert int(refreshed.state.installation_count) == 1
    chex.assert_trees_all_equal(
        refreshed.state.lifecycle_state,
        first.state.lifecycle_state,
    )

    changed_inputs = _inputs(4, raw=(0.7, 0.9), event=0.1, atom=0.2, bottleneck=0.3)
    changed_bundle = _bundle(
        composition,
        changed_inputs,
        refresh_inputs.raw_features,
        (0, 3, 4, 6),
    )
    changed = composition.install(
        refreshed.state,
        changed_bundle,
        jr.key(5),
        inputs=changed_inputs,
    )
    assert bool(changed.applied)
    assert bool(changed.semantics_changed)
    np.testing.assert_array_equal(changed.reset_slots, [False, True, False, False])
    np.testing.assert_array_equal(changed.preserved_slots, [True, False, True, True])
    assert int(changed.state.installation_count) == 2
    assert bool(changed.state.installer_unavailable)

    control_state = _force_extended_action(composition, changed.state, 2)
    later = _inputs(5, raw=(0.8, 1.0), event=0.2, atom=0.4, bottleneck=0.2)
    live = composition.materialize_live(control_state, later)
    assert bool(live.applied), "installer exhaustion must not freeze live STOMP control"
    started = composition.start(live.state, live.materialization)
    assert started.applied
    after = _inputs(6, raw=(0.9, 1.1), event=0.75, atom=0.3, bottleneck=0.1)
    next_live = composition.materialize_live(started.state, after)
    controlled = composition.update(
        next_live.state,
        next_live.materialization,
        jnp.asarray(0.2, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
    )
    assert controlled.applied
    assert bool(controlled.state.installer_unavailable)


def test_invalid_stale_misattributed_and_corrupt_transactions_are_exact_noops() -> None:
    composition = _composition()
    prior = _inputs(1, raw=(0.1, 0.2), event=0.0, atom=0.0, bottleneck=0.0)
    cold = composition.materialize_cold(_init(composition), prior)
    current = _inputs(2, raw=(0.3, 0.4), event=0.25, atom=0.35, bottleneck=0.45)
    wrong_values = jnp.zeros((4,), dtype=jnp.float32)
    invalid_install = composition.install(
        cold.state,
        _bundle(
            composition,
            current,
            prior.raw_features,
            values_override=wrong_values,
        ),
        jr.key(4),
        inputs=current,
    )
    assert not bool(invalid_install.applied)
    chex.assert_trees_all_equal(invalid_install.state, cold.state)

    non_emitted = composition.install(
        cold.state,
        composition.discovery.empty_proposal_bundle(-1),
        jr.key(4),
        inputs=current,
    )
    assert not bool(non_emitted.applied)
    chex.assert_trees_all_equal(non_emitted.state, cold.state)

    installed = composition.install(
        cold.state,
        _bundle(composition, current, prior.raw_features),
        jr.key(5),
        inputs=current,
    )
    stale = composition.materialize_live(installed.state, current)
    assert not bool(stale.applied)
    chex.assert_trees_all_equal(stale.state, installed.state)

    next_inputs = _inputs(3, raw=(0.5, 0.6), event=0.2, atom=0.3, bottleneck=0.4)
    misattributed = dataclasses.replace(
        next_inputs,
        source_digest=jnp.asarray([9, 10], dtype=jnp.uint32),
    )
    rejected = composition.materialize_live(installed.state, misattributed)
    assert not bool(rejected.applied)
    chex.assert_trees_all_equal(rejected.state, installed.state)

    unavailable = dataclasses.replace(
        next_inputs,
        controllable_events_available=jnp.asarray([False], dtype=jnp.bool_),
    )
    unavailable_result = composition.materialize_live(installed.state, unavailable)
    assert not bool(unavailable_result.applied)
    chex.assert_trees_all_equal(unavailable_result.state, installed.state)

    corrupt = dataclasses.replace(
        installed.state,
        last_raw_features=installed.state.last_raw_features.at[0].set(jnp.nan),
    )
    control = composition.start(corrupt, installed.materialization)
    assert not control.applied
    chex.assert_trees_all_equal(control.state, corrupt)


@pytest.mark.integration
def test_checkpoint_and_jit_materialization_roundtrip_fail_closed() -> None:
    composition = _composition()
    prior = _inputs(1, raw=(0.1, 0.2), event=0.0, atom=0.0, bottleneck=0.0)
    with jax.disable_jit(False):
        cold_fn = jax.jit(composition.materialize_cold)
        cold = cold_fn(_init(composition), prior)
        assert bool(cold.applied)
        current = _inputs(2, raw=(0.3, 0.4), event=0.25, atom=0.35, bottleneck=0.45)
        bundle = _bundle(composition, current, prior.raw_features)
        install_fn = jax.jit(
            lambda state, proposal, key, live: composition.install(
                state,
                proposal,
                key,
                inputs=live,
            )
        )
        installed = install_fn(cold.state, bundle, jr.key(9), current)
    assert bool(installed.applied)
    payload = composition.checkpoint_payload(installed.state)
    restored = composition.restore_checkpoint(
        payload,
        expected_consumer_source_digest=CONSUMER_SOURCE,
        expected_consumer_representation_digest=REPRESENTATION,
        expected_lifecycle_id=LIFECYCLE_ID,
        expected_installed_bundle=bundle,
    )
    chex.assert_trees_all_equal(restored, installed.state)
    resources = composition.resource_budget(installed.state)
    assert resources.control_host_only is True
    assert resources.installer_capacity_can_block_valid_stomp_control is False

    tampered = copy.deepcopy(payload)
    tampered["state_digest"] = jnp.zeros((32,), dtype=jnp.uint8)
    with pytest.raises(ValueError, match="digest differs"):
        composition.restore_checkpoint(
            tampered,
            expected_consumer_source_digest=CONSUMER_SOURCE,
            expected_consumer_representation_digest=REPRESENTATION,
            expected_lifecycle_id=LIFECYCLE_ID,
            expected_installed_bundle=bundle,
        )

    assert CUMULANT_OPTION_INSTALLATION_ASSESSMENT == "not_assessed"
    assert CUMULANT_OPTION_INSTALLATION_OUTPUT_WRITES is False
    assert CUMULANT_OPTION_INSTALLATION_EVIDENCE_AUTHORITY is False
    assert CUMULANT_OPTION_INSTALLATION_PROMOTION_AUTHORITY is False
    assert CUMULANT_OPTION_INSTALLATION_BENEFIT_CLAIM is False
    assert CUMULANT_OPTION_INSTALLATION_AUTONOMOUS_DISCOVERY_CLAIM is False
    assert CUMULANT_OPTION_INSTALLATION_SCIENTIFIC_PROMOTION_ALLOWED is False
    assert CUMULANT_OPTION_INSTALLATION_CONTROL_HOST_ONLY is True
