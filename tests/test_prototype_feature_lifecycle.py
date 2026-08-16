# mypy: disable-error-code="attr-defined,call-arg,no-untyped-def"
"""Standalone L0 contracts for bounded prototype feature discovery."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.core.prototype_feature_lifecycle as lifecycle_module
from alberta_framework.core.checkpoints import load_checkpoint_metadata
from alberta_framework.core.oak import OaKAgent, OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_feature_lifecycle import (
    PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS,
    PROTOTYPE_FEATURE_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED,
    PrototypeFeatureConsumerBinding,
    PrototypeFeatureLifecycle,
    PrototypeFeatureLifecycleConfig,
    PrototypeFeatureLifecycleEvent,
    load_prototype_feature_lifecycle_checkpoint,
    save_prototype_feature_lifecycle_checkpoint,
)

pytestmark = pytest.mark.unit


def _config(*, replacement_interval: int = 1) -> PrototypeFeatureLifecycleConfig:
    return PrototypeFeatureLifecycleConfig(
        base_feature_dim=4,
        active_pair_slots=2,
        candidate_pair_slots=6,
        n_tasks=2,
        n_options=2,
        n_primitive_actions=2,
        option_subtask_feature_indices=(0, 1),
        step_size_output=0.05,
        utility_decay=0.9,
        replacement_interval=replacement_interval,
        min_feature_age=0,
        candidate_min_age=0,
        promotion_margin=1.0,
        scale_normalizer_decay=0.9,
        scale_normalizer_epsilon=1.0e-6,
        carry_survivors=True,
        max_observations=100,
    )


def _oak_agent(config: PrototypeFeatureLifecycleConfig, *, hidden: bool = False) -> OaKAgent:
    specs = tuple(
        SubtaskSpec(feature_index=index)
        for index in config.option_subtask_feature_indices
    )
    return OaKAgent(
        OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=specs,
                observation_dim=config.total_feature_dim,
                n_primitive_actions=config.n_primitive_actions,
                base_hidden_sizes=((3,) if hidden else ()),
                epsilon_base=0.0,
                epsilon_option=0.0,
            )
        )
    )


def _materialize_keys(tree: Any) -> Any:
    def convert(value: Any) -> Any:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)
        # MultiHeadMLPState's two host-only timing leaves are Python floats in
        # eager state and traced float32 scalars under JIT.  They are not
        # scientific state; normalize only those host scalars for comparison.
        if type(value) is float:
            return jnp.asarray(value, dtype=jnp.float32)
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_exact(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_tree = jax.tree.flatten(_materialize_keys(right))
    assert left_tree == right_tree  # type: ignore[operator]
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _assert_tree_close(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_tree = jax.tree.flatten(_materialize_keys(right))
    assert left_tree == right_tree  # type: ignore[operator]
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        if np.issubdtype(left_array.dtype, np.inexact):
            np.testing.assert_allclose(left_array, right_array, rtol=1.0e-6, atol=1.0e-7)
        else:
            np.testing.assert_array_equal(left_array, right_array)


def _force_promotion(lifecycle: PrototypeFeatureLifecycle, state):
    learner_state = state.learner_state
    active = set(
        zip(
            np.asarray(learner_state.feature_left).tolist(),
            np.asarray(learner_state.feature_right).tolist(),
            strict=True,
        )
    )
    candidates = list(
        zip(
            np.asarray(learner_state.candidate_left).tolist(),
            np.asarray(learner_state.candidate_right).tolist(),
            strict=True,
        )
    )
    candidate_index = next(index for index, pair in enumerate(candidates) if pair not in active)
    candidate_utilities = jnp.zeros_like(learner_state.candidate_utilities)
    candidate_utilities = candidate_utilities.at[candidate_index].set(0.9)
    learner_state = learner_state.replace(
        utilities=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
        ages=jnp.full_like(learner_state.ages, 10),
        candidate_utilities=candidate_utilities,
        candidate_ages=jnp.full_like(learner_state.candidate_ages, 10),
        step_count=jnp.asarray(10, dtype=jnp.int32),
    )
    forced = state.replace(
        learner_state=learner_state,
        observe_count=jnp.asarray(10, dtype=jnp.int32),
    )
    assert bool(lifecycle.state_valid(forced))
    return forced, candidates[candidate_index]


def _consumer_binding(state) -> PrototypeFeatureConsumerBinding:
    return PrototypeFeatureConsumerBinding(
        semantic_generation=state.router_state.generation_count,
        descriptors=state.router_state.descriptors,
    )


def _seed_oak(
    lifecycle: PrototypeFeatureLifecycle,
    lifecycle_state,
    next_base_observation: jax.Array,
) -> OaKState:
    config = lifecycle.config
    oak = _oak_agent(config).init(jr.key(71))
    stomp = oak.stomp_state
    width = config.total_feature_dim

    head_weights = tuple(
        jnp.arange(width, dtype=jnp.float32)[None, :] + 100.0 * (index + 1)
        for index in range(config.n_total_actions)
    )
    head_traces = tuple(
        (
            jnp.arange(width, dtype=jnp.float32)[None, :] + 1_000.0 * (index + 1),
            trace_pair[1],
        )
        for index, trace_pair in enumerate(stomp.base_learner_state.head_traces)
    )
    base_learner = stomp.base_learner_state.replace(
        head_params=stomp.base_learner_state.head_params.replace(weights=head_weights),
        head_traces=head_traces,
    )
    policy_values = jnp.arange(
        config.n_options * config.n_primitive_actions * width,
        dtype=jnp.float32,
    ).reshape(config.n_options, config.n_primitive_actions, width)
    model_values = jnp.arange(
        config.n_options * width * width,
        dtype=jnp.float32,
    ).reshape(config.n_options, width, width)
    policies = stomp.option_policies.replace(
        q_weights=policy_values + 2_000.0,
        traces=policy_values + 3_000.0,
    )
    models = stomp.option_models.replace(next_state_weights=model_values + 4_000.0)
    stomp = stomp.replace(
        base_learner_state=base_learner,
        base_last_obs=lifecycle.augment(lifecycle_state, next_base_observation),
        base_last_action=jnp.asarray(0, dtype=jnp.int32),
        option_policies=policies,
        option_models=models,
        executing_option=jnp.asarray(-1, dtype=jnp.int32),
        option_start_obs=jnp.arange(width, dtype=jnp.float32) + 5_000.0,
    )
    return cast(OaKState, oak.replace(stomp_state=stomp))


def _event(*, allow_curation: bool = True) -> PrototypeFeatureLifecycleEvent:
    return PrototypeFeatureLifecycleEvent(
        observation=jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float32),
        targets=jnp.asarray([0.75, jnp.nan], dtype=jnp.float32),
        next_observation=jnp.asarray([-1.0, 2.0, 4.0, 0.25], dtype=jnp.float32),
        allow_curation=jnp.asarray(allow_curation, dtype=jnp.bool_),
    )


def test_config_is_strict_versioned_mechanism_only_and_round_trips() -> None:
    class ConfigSubclass(PrototypeFeatureLifecycleConfig):
        pass

    class OaKConfigSubclass(OaKConfig):
        pass

    class STOMPConfigSubclass(STOMPConfig):
        pass

    class TupleSubclass(tuple[int, ...]):
        pass

    class ListSubclass(list[int]):
        pass

    config = _config()
    lifecycle = PrototypeFeatureLifecycle(config)
    assert PrototypeFeatureLifecycleConfig.from_config(config.to_config()) == config
    lifecycle.require_compatible_oak_config(_oak_agent(config).config)
    assert config.to_config()["mechanism_status"] == (
        PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS
    )
    assert config.to_config()["scientific_promotion_allowed"] is False
    assert PROTOTYPE_FEATURE_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED is False

    config_subclass = ConfigSubclass(**dataclasses.asdict(config))
    with pytest.raises(TypeError, match="PrototypeFeatureLifecycleConfig"):
        PrototypeFeatureLifecycle(config_subclass)
    assert type(config_subclass.from_config(config.to_config())) is (
        PrototypeFeatureLifecycleConfig
    )
    oak_subclass = OaKConfigSubclass(stomp=_oak_agent(config).config.stomp)
    with pytest.raises(TypeError, match="OaKConfig"):
        lifecycle.require_compatible_oak_config(oak_subclass)
    exact_stomp = _oak_agent(config).config.stomp
    stomp_subclass = STOMPConfigSubclass(
        **{
            field.name: getattr(exact_stomp, field.name)
            for field in dataclasses.fields(exact_stomp)
        }
    )
    # OaKConfig is now the authoritative first boundary: it must reject an
    # unvalidated STOMP subclass before derived OaK resource accounting runs.
    with pytest.raises(ValueError, match="actual STOMPConfig"):
        OaKConfig(stomp=stomp_subclass)

    payload = config.to_config()
    payload["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="cannot claim promotion"):
        PrototypeFeatureLifecycleConfig.from_config(payload)
    payload = config.to_config()
    payload["active_pair_slots"] = True
    with pytest.raises(ValueError, match="active_pair_slots"):
        PrototypeFeatureLifecycleConfig.from_config(payload)
    with pytest.raises(ValueError, match="pair space"):
        dataclasses.replace(config, active_pair_slots=7)
    with pytest.raises(ValueError, match="requires candidate_pair_slots"):
        dataclasses.replace(config, candidate_pair_slots=0)
    with pytest.raises(ValueError, match="strict float"):
        dataclasses.replace(config, utility_decay=1)
    with pytest.raises(ValueError, match="stable base"):
        dataclasses.replace(config, option_subtask_feature_indices=(0, 4))
    with pytest.raises(ValueError, match="exact tuple"):
        dataclasses.replace(
            config,
            option_subtask_feature_indices=TupleSubclass((0, 1)),
        )
    payload = config.to_config()
    payload["option_subtask_feature_indices"] = ListSubclass([0, 1])
    with pytest.raises(ValueError, match="JSON integer list"):
        PrototypeFeatureLifecycleConfig.from_config(payload)
    with pytest.raises(ValueError, match="float32"):
        dataclasses.replace(config, step_size_output=1.0e300)
    with pytest.raises(ValueError, match="float32"):
        dataclasses.replace(config, scale_normalizer_epsilon=1.0e-300)
    with pytest.raises(ValueError, match="float32"):
        dataclasses.replace(config, utility_decay=0.999999999)
    with pytest.raises(ValueError, match="float32"):
        dataclasses.replace(config, scale_normalizer_decay=0.999999999)
    mismatched_oak = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(SubtaskSpec(feature_index=0), SubtaskSpec(feature_index=4)),
            observation_dim=config.total_feature_dim,
            n_primitive_actions=config.n_primitive_actions,
        )
    )
    with pytest.raises(ValueError, match="subtask attestation"):
        lifecycle.require_compatible_oak_config(mismatched_oak)
    for malformed_index in (True, 1.0):
        with pytest.raises(ValueError, match="feature_index"):
            SubtaskSpec(feature_index=cast(int, malformed_index))
    for malformed_dimension in (float(config.total_feature_dim),):
        with pytest.raises(ValueError, match="observation_dim"):
            STOMPConfig(
                subtask_specs=(
                    SubtaskSpec(feature_index=0),
                    SubtaskSpec(feature_index=1),
                ),
                observation_dim=cast(int, malformed_dimension),
                n_primitive_actions=config.n_primitive_actions,
            )
    with pytest.raises(ValueError, match="n_primitive_actions"):
        STOMPConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0),
                SubtaskSpec(feature_index=1),
            ),
            observation_dim=config.total_feature_dim,
            n_primitive_actions=cast(int, float(config.n_primitive_actions)),
        )
    single_action_config = dataclasses.replace(config, n_primitive_actions=1)
    PrototypeFeatureLifecycle(single_action_config)
    with pytest.raises(ValueError, match="n_primitive_actions"):
        STOMPConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0),
                SubtaskSpec(feature_index=1),
            ),
            observation_dim=single_action_config.total_feature_dim,
            n_primitive_actions=cast(int, True),
        )


def test_allocation_and_python_collection_ceilings_fail_before_construction() -> None:
    config = _config()
    with pytest.raises(ValueError, match="active descriptor comparison"):
        dataclasses.replace(
            config,
            base_feature_dim=100,
            active_pair_slots=2_049,
            candidate_pair_slots=0,
            replacement_interval=0,
        )
    with pytest.raises(ValueError, match="all-pairs candidate enumeration"):
        dataclasses.replace(
            config,
            base_feature_dim=400,
            active_pair_slots=1,
            candidate_pair_slots=1,
        )
    with pytest.raises(ValueError, match="base-head collection"):
        dataclasses.replace(
            config,
            n_options=3_000,
            n_primitive_actions=2_000,
            option_subtask_feature_indices=(0,) * 3_000,
        )


def test_init_augmentation_state_and_exact_resource_contracts() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(3))

    np.testing.assert_array_equal(
        state.router_state.descriptors,
        np.asarray([[0, 1], [0, 2]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        state.router_state.descriptors[:, 0],
        state.learner_state.feature_left,
    )
    np.testing.assert_array_equal(
        state.router_state.descriptors[:, 1],
        state.learner_state.feature_right,
    )
    assert bool(lifecycle.state_valid(state))
    binding = _consumer_binding(state)
    assert bool(lifecycle.consumer_binding_valid(state, binding))

    observation = jnp.asarray([2.0, 3.0, 5.0, 7.0], dtype=jnp.float32)
    np.testing.assert_array_equal(
        lifecycle.augment(state, observation),
        np.asarray([2.0, 3.0, 5.0, 7.0, 6.0, 10.0], dtype=np.float32),
    )
    budget = lifecycle.resource_budget(state)
    expected_lifecycle_state_bytes = sum(
        int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree.leaves(state)
    )
    expected_learner_template_bytes = sum(
        int(getattr(leaf, "nbytes", 0))
        for leaf in jax.tree.leaves(state.learner_state)
    )
    expected_oak_template_bytes = sum(
        int(getattr(leaf, "nbytes", 0))
        for leaf in jax.tree.leaves(_oak_agent(lifecycle.config).init(jr.key(0)))
    )
    assert budget.lifecycle_state_nbytes == expected_lifecycle_state_bytes
    assert budget.consumer_binding_persistent_nbytes == sum(
        int(getattr(leaf, "nbytes", 0))
        for leaf in jax.tree.leaves(binding)
    )
    assert (
        budget.internal_learner_template_nbytes
        == expected_learner_template_bytes
    )
    assert budget.internal_oak_template_nbytes == expected_oak_template_bytes
    assert budget.internal_template_nbytes == (
        expected_learner_template_bytes + expected_oak_template_bytes
    )
    assert budget.owned_persistent_state_nbytes == (
        expected_lifecycle_state_bytes
        + expected_learner_template_bytes
        + expected_oak_template_bytes
    )
    assert budget.router_calls_per_committed_curation == 2
    assert budget.router_calls_per_observe == 2
    assert budget.max_active_pair_products_per_observe == 10
    assert budget.max_candidate_pair_products_per_observe == 6
    assert budget.managed_oak_feature_width == lifecycle.config.total_feature_dim
    assert budget.rebuilt_base_cache_nbytes == 4 * lifecycle.config.total_feature_dim
    assert budget.scientific_promotion_allowed is False


def test_unavailable_diagnostics_are_finite_neutral_and_jax_shape_compatible() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(4))
    generation = state.router_state.generation_count

    eager = lifecycle.unavailable_diagnostics(generation)
    compiled = jax.jit(lifecycle.unavailable_diagnostics)(generation)

    _assert_tree_exact(compiled, eager)
    assert not bool(eager.available)
    assert not bool(eager.consumer_binding_valid)
    assert not bool(eager.transaction_applied)
    assert not bool(eager.routing_attempted)
    assert not bool(eager.postcondition_checked)
    assert not bool(eager.postcondition_valid)
    assert not bool(eager.postcondition_rolled_back)
    assert int(eager.semantic_generation_before) == int(generation)
    assert int(eager.semantic_generation_after) == int(generation)
    for leaf in jax.tree.leaves(eager):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.inexact):
            assert np.all(np.isfinite(array))


def test_pair_gradient_pullback_matches_jax_autodiff_oracle() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(5))
    observation = jnp.asarray([0.5, -2.0, 3.0, 4.0], dtype=jnp.float32)
    augmented_gradient = jnp.asarray(
        [0.2, -0.3, 0.5, 1.0, 1.5, -2.0],
        dtype=jnp.float32,
    )

    result = lifecycle.pullback_pair_gradient(
        state,
        observation,
        augmented_gradient,
        state.router_state.generation_count,
        state.router_state.descriptors,
    )
    expected = jax.grad(
        lambda value: jnp.vdot(
            lifecycle.augment(state, value),
            augmented_gradient,
        )
    )(observation)

    assert bool(result.valid)
    assert int(result.semantic_generation) == 0
    np.testing.assert_allclose(result.gradient, expected, rtol=0.0, atol=0.0)

    invalid = lifecycle.pullback_pair_gradient(
        state,
        observation.at[0].set(jnp.inf),
        augmented_gradient,
        state.router_state.generation_count,
        state.router_state.descriptors,
    )
    assert not bool(invalid.valid)
    np.testing.assert_array_equal(invalid.gradient, jnp.zeros_like(observation))

    stale = lifecycle.pullback_pair_gradient(
        state,
        observation,
        augmented_gradient,
        jnp.asarray(1, dtype=jnp.int32),
        state.router_state.descriptors,
    )
    assert not bool(stale.valid)
    np.testing.assert_array_equal(stale.gradient, jnp.zeros_like(observation))


def test_descriptor_semantics_cannot_advance_ahead_of_generation() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(6))
    mutated_descriptors = jnp.asarray(
        [[1, 2], [2, 3]],
        dtype=jnp.int32,
    )
    mutated = state.replace(
        learner_state=state.learner_state.replace(
            feature_left=mutated_descriptors[:, 0],
            feature_right=mutated_descriptors[:, 1],
        ),
        router_state=dataclasses.replace(
            state.router_state,
            descriptors=mutated_descriptors,
        ),
    )
    assert not bool(lifecycle.state_valid(mutated))
    pullback = lifecycle.pullback_pair_gradient(
        mutated,
        jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        jnp.ones((lifecycle.config.total_feature_dim,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        mutated.router_state.descriptors,
    )
    assert not bool(pullback.valid)
    np.testing.assert_array_equal(
        pullback.gradient,
        np.zeros((lifecycle.config.base_feature_dim,), dtype=np.float32),
    )

    one_generation = mutated.replace(
        learner_state=mutated.learner_state.replace(
            step_count=jnp.asarray(1, dtype=jnp.int32)
        ),
        router_state=dataclasses.replace(
            mutated.router_state,
            route_count=jnp.asarray(1, dtype=jnp.int32),
            generation_count=jnp.asarray(1, dtype=jnp.int32),
        ),
        observe_count=jnp.asarray(1, dtype=jnp.int32),
        committed_curation_count=jnp.asarray(1, dtype=jnp.int32),
    )
    assert not bool(lifecycle.state_valid(one_generation))


def test_same_generation_descriptor_forks_cannot_cross_bind_or_pull_back() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    canonical = lifecycle.init(jr.key(61))

    def branch(descriptors: jax.Array):
        return canonical.replace(
            learner_state=canonical.learner_state.replace(
                feature_left=descriptors[:, 0],
                feature_right=descriptors[:, 1],
                step_count=jnp.asarray(1, dtype=jnp.int32),
            ),
            router_state=dataclasses.replace(
                canonical.router_state,
                descriptors=descriptors,
                route_count=jnp.asarray(1, dtype=jnp.int32),
                generation_count=jnp.asarray(1, dtype=jnp.int32),
            ),
            observe_count=jnp.asarray(1, dtype=jnp.int32),
            committed_curation_count=jnp.asarray(1, dtype=jnp.int32),
        )

    branch_a = branch(jnp.asarray([[0, 1], [0, 3]], dtype=jnp.int32))
    branch_b = branch(jnp.asarray([[0, 1], [1, 2]], dtype=jnp.int32))
    assert bool(lifecycle.state_valid(branch_a))
    assert bool(lifecycle.state_valid(branch_b))
    binding_b = _consumer_binding(branch_b)
    assert not bool(lifecycle.consumer_binding_valid(branch_a, binding_b))

    pullback = lifecycle.pullback_pair_gradient(
        branch_a,
        jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        jnp.ones((lifecycle.config.total_feature_dim,), dtype=jnp.float32),
        branch_b.router_state.generation_count,
        branch_b.router_state.descriptors,
    )
    assert not bool(pullback.valid)
    np.testing.assert_array_equal(
        pullback.gradient,
        np.zeros((lifecycle.config.base_feature_dim,), dtype=np.float32),
    )


def test_augmentation_rejects_finite_coordinates_whose_products_overflow() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(7))
    observation = jnp.full((4,), 1.0e20, dtype=jnp.float32)

    augmented = lifecycle.augment(state, observation)

    np.testing.assert_array_equal(
        augmented,
        jnp.zeros((lifecycle.config.total_feature_dim,), dtype=jnp.float32),
    )


def test_invalid_dynamic_input_is_an_exact_atomic_noop() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(17))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)
    invalid = event.replace(observation=event.observation.at[1].set(jnp.inf))

    result = lifecycle.observe_and_route(state, oak, _consumer_binding(state), invalid)

    assert not bool(result.diagnostics.event_values_valid)
    assert not bool(result.diagnostics.transaction_applied)
    assert not bool(result.diagnostics.curation_committed)
    _assert_tree_exact(result.state, state)
    _assert_tree_exact(result.oak_state, oak)
    np.testing.assert_array_equal(
        result.next_augmented_observation,
        lifecycle.augment(state, event.next_observation),
    )


@pytest.mark.parametrize(
    "counter_name",
    (
        "ages",
        "candidate_ages",
        "evidence_idle_steps",
        "utility_evidence_streak",
        "candidate_promotion_evidence_streak",
    ),
)
def test_owned_learner_counters_cannot_exceed_observed_steps(
    counter_name: str,
) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(43))
    counter = getattr(state.learner_state, counter_name)
    corrupt = state.replace(
        learner_state=state.learner_state.replace(
            **{
                counter_name: jnp.full_like(counter, 2_147_483_647),
            }
        )
    )
    assert not bool(lifecycle.state_valid(corrupt))

    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)
    result = lifecycle.observe_and_route(
        corrupt,
        oak,
        _consumer_binding(corrupt),
        event,
    )
    assert not bool(result.diagnostics.state_values_valid)
    assert not bool(result.diagnostics.transaction_applied)
    _assert_tree_exact(result.state, corrupt)
    _assert_tree_exact(result.oak_state, oak)


@pytest.mark.parametrize("timer_name", ("birth_timestamp", "uptime_s"))
def test_owned_learner_timers_must_be_nonnegative(timer_name: str) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(45))
    corrupt = state.replace(
        learner_state=state.learner_state.replace(
            **{
                timer_name: jnp.asarray(-1.0, dtype=jnp.float32),
            }
        )
    )
    assert not bool(lifecycle.state_valid(corrupt))


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("feature_parent_a", 0),
        ("feature_parent_b", 0),
        ("candidate_parent_a", 0),
        ("candidate_parent_b", 0),
        ("feature_generator", 1),
        ("candidate_generator", 1),
    ),
)
def test_pair_only_lifecycle_rejects_compositional_provenance(
    field: str,
    replacement: int,
) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(46))
    current = getattr(state.learner_state, field)
    corrupt = state.replace(
        learner_state=state.learner_state.replace(
            **{field: jnp.full_like(current, replacement)}
        )
    )
    assert not bool(lifecycle.state_valid(corrupt))


def test_fixed_disabled_learner_substate_must_remain_reachable() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(48))
    state = state.replace(
        learner_state=state.learner_state.replace(
            step_count=jnp.asarray(1, dtype=jnp.int32)
        ),
        observe_count=jnp.asarray(1, dtype=jnp.int32),
    )
    assert bool(lifecycle.state_valid(state))
    learner = state.learner_state
    corruptions = (
        learner.replace(
            relevance_probe_weights=learner.relevance_probe_weights.at[0, 0].set(1.0)
        ),
        learner.replace(
            relevance_probe_biases=learner.relevance_probe_biases.at[0].set(1.0)
        ),
        learner.replace(
            evidence_idle_steps=learner.evidence_idle_steps.at[0].set(1)
        ),
        learner.replace(
            utility_evidence_streak=learner.utility_evidence_streak.at[0].set(1)
        ),
        learner.replace(
            candidate_promotion_evidence_streak=(
                learner.candidate_promotion_evidence_streak.at[0].set(1)
            )
        ),
        learner.replace(
            active_output_memory_committed=(
                learner.active_output_memory_committed.at[0].set(True)
            )
        ),
        learner.replace(
            candidate_reacquisition_required=(
                learner.candidate_reacquisition_required.at[0].set(True)
            )
        ),
    )
    for corrupt_learner in corruptions:
        corrupt = state.replace(learner_state=corrupt_learner)
        assert not bool(lifecycle.state_valid(corrupt))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("utilities", -1.0),
        ("utilities", 2.0),
        ("candidate_utilities", -1.0),
        ("candidate_utilities", 2.0),
    ),
)
def test_scale_robust_utility_emas_remain_bounded(
    field: str,
    value: float,
) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(49))
    current = getattr(state.learner_state, field)
    corrupt = state.replace(
        learner_state=state.learner_state.replace(
            **{field: current.at[0].set(value)}
        )
    )
    assert not bool(lifecycle.state_valid(corrupt))


def test_full_oak_audit_rejects_ownership_model_counter_and_outer_corruption() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(47))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)
    stomp = oak.stomp_state
    corruptions = (
        oak.replace(
            stomp_state=stomp.replace(
                base_last_action=jnp.asarray(1, dtype=jnp.int32)
            )
        ),
        oak.replace(
            stomp_state=stomp.replace(option_steps=jnp.asarray(1, dtype=jnp.int32))
        ),
        oak.replace(
            stomp_state=stomp.replace(
                option_models=stomp.option_models.replace(
                    duration_ema=stomp.option_models.duration_ema.at[0].set(-1.0)
                )
            )
        ),
        oak.replace(
            execution_counts=jnp.asarray([2, 0], dtype=jnp.int32)
        ),
        oak.replace(
            stomp_state=stomp.replace(
                option_models=stomp.option_models.replace(
                    n_completions=jnp.asarray(
                        [2_147_483_647, 0],
                        dtype=jnp.int32,
                    )
                )
            )
        ),
        oak.replace(
            stomp_state=stomp.replace(
                base_learner_state=stomp.base_learner_state.replace(
                    birth_timestamp=-1.0
                )
            )
        ),
        oak.replace(
            stomp_state=stomp.replace(
                base_learner_state=stomp.base_learner_state.replace(
                    uptime_s=jnp.asarray(-1.0, dtype=jnp.float32)
                )
            )
        ),
    )

    for corrupt in corruptions:
        result = lifecycle.observe_and_route(
            state,
            corrupt,
            _consumer_binding(state),
            event,
        )
        assert not bool(result.diagnostics.oak_values_valid)
        assert not bool(result.diagnostics.transaction_applied)
        _assert_tree_exact(result.state, state)
        _assert_tree_exact(result.oak_state, corrupt)


def test_max_observation_capacity_is_an_exact_atomic_noop() -> None:
    config = dataclasses.replace(_config(replacement_interval=0), max_observations=1)
    lifecycle = PrototypeFeatureLifecycle(config)
    state = lifecycle.init(jr.key(53))
    state = state.replace(
        learner_state=state.learner_state.replace(
            step_count=jnp.asarray(1, dtype=jnp.int32)
        ),
        observe_count=jnp.asarray(1, dtype=jnp.int32),
    )
    assert bool(lifecycle.state_valid(state))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)

    result = lifecycle.observe_and_route(state, oak, _consumer_binding(state), event)

    assert not bool(result.diagnostics.update_capacity_available)
    assert not bool(result.diagnostics.transaction_applied)
    _assert_tree_exact(result.state, state)
    _assert_tree_exact(result.oak_state, oak)


def test_static_contract_rejects_wrong_event_dtype_and_nonlinear_oak() -> None:
    class EventSubclass(PrototypeFeatureLifecycleEvent):
        pass

    class BindingSubclass(PrototypeFeatureConsumerBinding):
        pass

    lifecycle = PrototypeFeatureLifecycle(_config())
    state = lifecycle.init(jr.key(19))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)

    with pytest.raises(TypeError, match="event.observation"):
        lifecycle.observe_and_route(
            state,
            oak,
            _consumer_binding(state),
            event.replace(observation=event.observation.astype(jnp.float16)),
        )
    subclass_event = EventSubclass(
        observation=event.observation,
        targets=event.targets,
        next_observation=event.next_observation,
        allow_curation=event.allow_curation,
    )
    with pytest.raises(TypeError, match="PrototypeFeatureLifecycleEvent"):
        lifecycle.observe_and_route(
            state,
            oak,
            _consumer_binding(state),
            subclass_event,
        )
    binding = _consumer_binding(state)
    binding_subclass = BindingSubclass(
        semantic_generation=binding.semantic_generation,
        descriptors=binding.descriptors,
    )
    with pytest.raises(TypeError, match="PrototypeFeatureConsumerBinding"):
        lifecycle.observe_and_route(state, oak, binding_subclass, event)
    with pytest.raises(TypeError, match="consumer_binding.descriptors"):
        lifecycle.observe_and_route(
            state,
            oak,
            binding.replace(descriptors=binding.descriptors.astype(jnp.float32)),
            event,
        )
    nonlinear = _oak_agent(lifecycle.config, hidden=True).init(jr.key(20))
    nonlinear = nonlinear.replace(
        stomp_state=nonlinear.stomp_state.replace(
            base_last_obs=lifecycle.augment(state, event.next_observation)
        )
    )
    with pytest.raises(ValueError, match="linear OaK"):
        lifecycle.observe_and_route(
            state,
            nonlinear,
            _consumer_binding(state),
            event,
        )
    for malformed_timer in (
        jnp.asarray(1.0, dtype=jnp.float16),
        jnp.asarray(1.0 + 2.0j, dtype=jnp.complex64),
    ):
        malformed_oak = oak.replace(
            stomp_state=oak.stomp_state.replace(
                base_learner_state=oak.stomp_state.base_learner_state.replace(
                    birth_timestamp=malformed_timer
                )
            )
        )
        with pytest.raises(ValueError, match="linear OaK"):
            lifecycle.observe_and_route(
                state,
                malformed_oak,
                _consumer_binding(state),
                event,
            )


def test_eager_and_jit_transactions_are_exact_and_scan_is_recurrent() -> None:
    lifecycle = PrototypeFeatureLifecycle(_config())
    state, _ = _force_promotion(lifecycle, lifecycle.init(jr.key(23)))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)

    binding = _consumer_binding(state)
    eager = lifecycle.observe_and_route(state, oak, binding, event)
    compiled = jax.jit(lifecycle.observe_and_route)(state, oak, binding, event)
    _assert_tree_exact(compiled, eager)

    scan_lifecycle = PrototypeFeatureLifecycle(_config(replacement_interval=0))
    scan_state, scan_binding = scan_lifecycle.init_bound(jr.key(29))
    scan_oak = _seed_oak(scan_lifecycle, scan_state, event.next_observation)
    events = jax.tree.map(lambda value: jnp.stack([value, value, value]), event)

    def body(carry, one_event):
        one_state, one_oak, one_binding = carry
        result = scan_lifecycle.observe_and_route(
            one_state,
            one_oak,
            one_binding,
            one_event,
        )
        return (
            result.state,
            result.oak_state,
            result.consumer_binding,
        ), result.diagnostics.transaction_applied

    (scanned_state, scanned_oak, scanned_binding), applied = jax.jit(
        lambda initial, values: jax.lax.scan(body, initial, values)
    )((scan_state, scan_oak, scan_binding), events)
    loop_state, loop_oak, loop_binding = scan_state, scan_oak, scan_binding
    for _ in range(3):
        result = scan_lifecycle.observe_and_route(
            loop_state,
            loop_oak,
            loop_binding,
            event,
        )
        loop_state, loop_oak, loop_binding = (
            result.state,
            result.oak_state,
            result.consumer_binding,
        )
    # XLA may reassociate the scan's float32 EMA arithmetic by one ULP; all
    # integer, descriptor, counter, and key leaves remain bit exact.
    _assert_tree_close(scanned_state, loop_state)
    _assert_tree_close(scanned_oak, loop_oak)
    _assert_tree_exact(scanned_binding, loop_binding)
    np.testing.assert_array_equal(applied, np.ones(3, dtype=np.bool_))
    assert int(scanned_state.observe_count) == 3
    assert int(scanned_state.router_state.generation_count) == 0


def test_checkpoint_round_trip_is_strict_and_resource_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = PrototypeFeatureLifecycle(_config(replacement_interval=0))
    state = lifecycle.init(jr.key(31))
    event = _event()
    oak = _seed_oak(lifecycle, state, event.next_observation)
    state = lifecycle.observe_and_route(
        state,
        oak,
        _consumer_binding(state),
        event,
    ).state
    path = tmp_path / "feature-lifecycle"

    save_prototype_feature_lifecycle_checkpoint(lifecycle, state, path)
    restored_lifecycle, restored_state = load_prototype_feature_lifecycle_checkpoint(path)

    assert restored_lifecycle.config == lifecycle.config
    assert restored_lifecycle.resource_budget(restored_state) == lifecycle.resource_budget(state)
    assert bool(restored_lifecycle.state_valid(restored_state))
    _assert_tree_exact(restored_state, state)

    metadata = load_checkpoint_metadata(path)
    raw_budget = cast(dict[str, object], metadata["resource_budget"])
    type_tampers = (
        ("scientific_promotion_allowed", 0),
        (
            "owned_persistent_state_nbytes",
            float(cast(int, raw_budget["owned_persistent_state_nbytes"])),
        ),
    )
    for field, replacement in type_tampers:
        tampered = copy.deepcopy(metadata)
        tampered_budget = cast(dict[str, object], tampered["resource_budget"])
        tampered_budget[field] = replacement
        monkeypatch.setattr(
            lifecycle_module,
            "load_checkpoint_metadata",
            lambda _path, payload=tampered: payload,
        )
        monkeypatch.setattr(
            lifecycle_module,
            "load_checkpoint",
            lambda _template, _path, payload=tampered: (state, payload),
        )
        with pytest.raises(ValueError, match="resource contract changed"):
            load_prototype_feature_lifecycle_checkpoint(path)

    changed_between_reads = copy.deepcopy(metadata)
    changed_budget = cast(
        dict[str, object],
        changed_between_reads["resource_budget"],
    )
    changed_budget["scientific_promotion_allowed"] = 0
    monkeypatch.setattr(
        lifecycle_module,
        "load_checkpoint_metadata",
        lambda _path: metadata,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "load_checkpoint",
        lambda _template, _path: (state, changed_between_reads),
    )
    with pytest.raises(ValueError, match="metadata changed between reads"):
        load_prototype_feature_lifecycle_checkpoint(path)

    corrupt = state.replace(observe_count=jnp.asarray(-1, dtype=jnp.int32))
    with pytest.raises(ValueError, match="state is invalid"):
        save_prototype_feature_lifecycle_checkpoint(
            lifecycle,
            corrupt,
            tmp_path / "corrupt",
        )
