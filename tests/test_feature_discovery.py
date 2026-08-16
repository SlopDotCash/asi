"""Tests for Step 2 fixed-budget feature discovery."""

from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.experimental import checkify

from alberta_framework import (
    FixedBudgetFeatureLearner,
    FixedBudgetInteractionLearner,
    InteractionFeatureDiscoveryStream,
    NonlinearFeatureDiscoveryStream,
    collect_feature_discovery_stream,
    run_feature_discovery_arrays,
    run_feature_discovery_loop,
    run_interaction_feature_arrays,
)
from alberta_framework.core.checkpoints import load_checkpoint_metadata, save_checkpoint
from alberta_framework.core.feature_discovery import (
    GENERATOR_IMPRINT,
    GENERATOR_MUTATE_PARENT,
    GENERATOR_RANDOM,
)
from alberta_framework.core.future_utility import (
    one_step_output_loss_reduction,
    trace_output_loss_reduction,
)
from alberta_framework.core.interaction_features import (
    RELEVANCE_PROBE_MODE_CONDITIONAL_V1,
    RELEVANCE_PROBE_MODE_TARGET_ONLY_V1,
    InteractionCurationPriorityOverride,
    load_interaction_feature_checkpoint,
    save_interaction_feature_checkpoint,
)
from alberta_framework.core.update_safety import floating_tree_is_finite


def test_one_step_output_loss_reduction_is_causal_lms_counterfactual() -> None:
    reductions = one_step_output_loss_reduction(
        errors=jnp.array([2.0, 0.0], dtype=jnp.float32),
        feature_values=jnp.array([1.0, 2.0], dtype=jnp.float32),
        active_mask=jnp.array([True, False]),
        step_size_output=0.5,
        active_count=1.0,
    )

    chex.assert_shape(reductions, (2, 2))
    # Feature 0: delta_y = 0.5 * 2 * 1**2 = 1, so loss reduction is
    # 2 * 1 - 0.5 * 1**2 = 1.5.
    assert float(reductions[0, 0]) == 1.5
    # Feature 1 overshoots the residual: delta_y = 4, so the signed reduction
    # would be zero after clipping.
    assert float(reductions[0, 1]) == 0.0
    assert float(reductions[1, 0]) == 0.0


def test_trace_output_loss_reduction_matches_one_step_at_zero_decay() -> None:
    errors = jnp.array([2.0, 0.0], dtype=jnp.float32)
    features = jnp.array([1.0, 2.0], dtype=jnp.float32)
    active_mask = jnp.array([True, False])

    one_step = one_step_output_loss_reduction(
        errors=errors,
        feature_values=features,
        active_mask=active_mask,
        step_size_output=0.5,
        active_count=1.0,
    )
    traced, error_trace, feature_trace, feature_energy_trace = trace_output_loss_reduction(
        errors=errors,
        feature_values=features,
        active_mask=active_mask,
        step_size_output=0.5,
        active_count=1.0,
        error_trace=jnp.zeros(2, dtype=jnp.float32),
        feature_trace=jnp.zeros(2, dtype=jnp.float32),
        feature_energy_trace=jnp.zeros(2, dtype=jnp.float32),
        trace_decay=0.0,
    )

    chex.assert_trees_all_close(traced, one_step)
    chex.assert_trees_all_close(error_trace, errors)
    chex.assert_trees_all_close(feature_trace, features)
    chex.assert_trees_all_close(feature_energy_trace, features**2)


def test_trace_output_loss_reduction_credits_recurring_alignment() -> None:
    _, error_trace, feature_trace, feature_energy_trace = trace_output_loss_reduction(
        errors=jnp.array([1.0], dtype=jnp.float32),
        feature_values=jnp.array([1.0], dtype=jnp.float32),
        active_mask=jnp.array([True]),
        step_size_output=0.1,
        active_count=1.0,
        error_trace=jnp.zeros(1, dtype=jnp.float32),
        feature_trace=jnp.zeros(1, dtype=jnp.float32),
        feature_energy_trace=jnp.zeros(1, dtype=jnp.float32),
        trace_decay=0.9,
    )
    traced, _, _, _ = trace_output_loss_reduction(
        errors=jnp.array([1.0], dtype=jnp.float32),
        feature_values=jnp.array([1.0], dtype=jnp.float32),
        active_mask=jnp.array([True]),
        step_size_output=0.1,
        active_count=1.0,
        error_trace=error_trace,
        feature_trace=feature_trace,
        feature_energy_trace=feature_energy_trace,
        trace_decay=0.9,
    )
    one_step = one_step_output_loss_reduction(
        errors=jnp.array([1.0], dtype=jnp.float32),
        feature_values=jnp.array([1.0], dtype=jnp.float32),
        active_mask=jnp.array([True]),
        step_size_output=0.1,
        active_count=1.0,
    )

    assert float(traced[0, 0]) > float(one_step[0, 0])


class TestNonlinearFeatureDiscoveryStream:
    """Tests for the Step 2 nonlinear multitask benchmark stream."""

    def test_step_shapes(self) -> None:
        stream = NonlinearFeatureDiscoveryStream(
            feature_dim=6,
            n_tasks=3,
            n_latents=8,
            context_length=5,
        )
        state = stream.init(jr.key(0))
        timestep, new_state = stream.step(state, jnp.array(0))

        chex.assert_shape(timestep.observation, (6,))
        chex.assert_shape(timestep.target, (3,))
        chex.assert_tree_all_finite(timestep.observation)
        chex.assert_tree_all_finite(timestep.target)
        assert int(new_state.step_count) == 1

    def test_collect_stream_shapes(self) -> None:
        stream = NonlinearFeatureDiscoveryStream(
            feature_dim=5,
            n_tasks=2,
            n_latents=6,
        )
        observations, targets = collect_feature_discovery_stream(
            stream, num_steps=12, key=jr.key(1)
        )

        chex.assert_shape(observations, (12, 5))
        chex.assert_shape(targets, (12, 2))
        chex.assert_tree_all_finite(observations)
        chex.assert_tree_all_finite(targets)


class TestInteractionFeatureDiscoveryStream:
    """Tests for the hidden pair-product Step 2 benchmark stream."""

    def test_init_selects_exact_active_count_when_scores_tie(self, monkeypatch) -> None:
        stream = InteractionFeatureDiscoveryStream(
            feature_dim=4, n_tasks=1, n_contexts=1, active_pairs_per_context=2
        )

        def tied_uniform(key, shape, dtype=jnp.float32, **kwargs):
            return jnp.zeros(shape, dtype=dtype)

        monkeypatch.setattr(
            "alberta_framework.streams.feature_discovery.jr.uniform", tied_uniform
        )
        state = stream.init(jr.key(0))
        assert int(jnp.count_nonzero(state.context_weights)) == 2

    @pytest.mark.parametrize("active_count", [0, -1, True, 1.0, np.int64(1)])
    def test_active_pair_count_requires_positive_builtin_int(
        self, active_count: object
    ) -> None:
        with pytest.raises(ValueError, match="positive built-in integer"):
            InteractionFeatureDiscoveryStream(
                feature_dim=4,
                active_pairs_per_context=active_count,  # type: ignore[arg-type]
            )

    def test_active_pair_count_caps_at_available_pairs(self) -> None:
        stream = InteractionFeatureDiscoveryStream(
            feature_dim=4,
            n_tasks=2,
            n_contexts=2,
            active_pairs_per_context=100,
        )
        state = stream.init(jr.key(91))

        assert state.context_weights.shape == (2, 2, 6)
        assert int(jnp.count_nonzero(state.context_weights)) == state.context_weights.size

    def test_unique_finite_scores_preserve_legacy_context_weights(self) -> None:
        stream = InteractionFeatureDiscoveryStream(
            feature_dim=4,
            n_tasks=2,
            n_contexts=2,
            active_pairs_per_context=3,
        )
        root_key = jr.key(314)
        state = stream.init(root_key)
        _, context_key, mask_key, _ = jr.split(root_key, 4)
        pair_count = state.pair_left.shape[0]
        dense_weights = jr.normal(
            context_key,
            (2, 2, pair_count),
            dtype=jnp.float32,
        )
        scores = jr.uniform(mask_key, (2, 2, pair_count), dtype=jnp.float32)
        for row in np.asarray(scores).reshape((-1, pair_count)):
            assert np.unique(row).size == pair_count
        threshold = jnp.sort(scores, axis=-1)[..., 2:3]
        legacy_mask = scores <= threshold
        expected = dense_weights * legacy_mask.astype(jnp.float32) / jnp.sqrt(
            jnp.sum(legacy_mask, axis=-1, keepdims=True)
        )

        np.testing.assert_array_equal(
            np.asarray(state.context_weights),
            np.asarray(expected),
        )

    @pytest.mark.parametrize("compiled", [False, True])
    @pytest.mark.parametrize(
        ("scores", "expected_mask"),
        [
            ([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [True, True, False, False, False, False]),
            ([0.1, 0.5, 0.5, 0.2, 0.9, 0.8], [True, True, False, True, False, False]),
        ],
    )
    def test_init_selects_exact_stable_active_count_under_ties(
        self,
        monkeypatch: pytest.MonkeyPatch,
        compiled: bool,
        scores: list[float],
        expected_mask: list[bool],
    ) -> None:
        stream = InteractionFeatureDiscoveryStream(
            feature_dim=4,
            n_tasks=2,
            n_contexts=2,
            active_pairs_per_context=3 if scores[0] != scores[1] else 2,
            noise_std=0.0,
        )

        def fixed_normal(key, shape, dtype=jnp.float32, **kwargs):
            del key, kwargs
            return jnp.ones(shape, dtype=dtype)

        def fixed_uniform(key, shape, dtype=jnp.float32, **kwargs):
            del key, kwargs
            return jnp.broadcast_to(jnp.asarray(scores, dtype=dtype), shape)

        monkeypatch.setattr(
            "alberta_framework.streams.feature_discovery.jr.normal", fixed_normal
        )
        monkeypatch.setattr(
            "alberta_framework.streams.feature_discovery.jr.uniform", fixed_uniform
        )
        init = jax.jit(stream.init) if compiled else stream.init
        state = init(jr.key(0))

        actual_mask = state.context_weights != 0.0
        expected = jnp.broadcast_to(
            jnp.asarray(expected_mask), state.context_weights.shape
        )
        chex.assert_trees_all_equal(actual_mask, expected)

        def body(carry, idx):
            timestep, next_state = stream.step(carry, idx)
            return next_state, (timestep.observation, timestep.target)

        final_state, outputs = jax.lax.scan(body, state, jnp.arange(4))
        assert int(final_state.step_count) == 4
        chex.assert_tree_all_finite(outputs)

    def test_step_shapes(self) -> None:
        stream = InteractionFeatureDiscoveryStream(
            feature_dim=6,
            n_tasks=3,
            context_length=5,
            active_pairs_per_context=2,
        )
        state = stream.init(jr.key(8))
        timestep, new_state = stream.step(state, jnp.array(0))

        chex.assert_shape(timestep.observation, (6,))
        chex.assert_shape(timestep.target, (3,))
        chex.assert_tree_all_finite(timestep.observation)
        chex.assert_tree_all_finite(timestep.target)
        assert int(new_state.step_count) == 1


class TestFixedBudgetFeatureLearner:
    """Tests for explicit feature construction, utility, and replacement."""

    def test_active_topk_excludes_inactive_zero_placeholders(self) -> None:
        """Mirror of the #275 fix for the interaction learner: inactive heads are not zeros."""
        learner = FixedBudgetFeatureLearner(
            n_features=1,
            n_tasks=4,
            candidate_count=0,
            utility_aggregation="topk",
            utility_top_k=2,
            utility_task_balancing="active",
        )
        active_mask = jnp.asarray((True, False, False, False), dtype=jnp.bool_)
        activity = jnp.ones((4,), dtype=jnp.float32)
        signed_signal = jnp.asarray(((-4.0,), (99.0,), (99.0,), (99.0,)), dtype=jnp.float32)
        utility = learner._aggregate_task_feature_signal(signed_signal, active_mask, activity)
        np.testing.assert_array_equal(utility, np.asarray((-4.0,), dtype=np.float32))

        output_weights = jnp.asarray(((6.0,), (5.0,), (5.0,), (5.0,)), dtype=jnp.float32)
        features = jnp.ones((1,), dtype=jnp.float32)
        for top_k in (1, 2, 4):
            probe = FixedBudgetFeatureLearner(
                n_features=1,
                n_tasks=4,
                candidate_count=0,
                utility_aggregation="topk",
                utility_top_k=top_k,
                utility_task_balancing="active",
            )
            utility = probe._output_utility_signal(output_weights, features, active_mask, activity)
            np.testing.assert_array_equal(utility, np.asarray((6.0,), dtype=np.float32))

    @pytest.mark.parametrize(
        "reducer_name", ["_aggregate_task_feature_signal", "_output_utility_signal"]
    )
    def test_active_topk_all_inactive_avoids_zero_division(self, reducer_name: str) -> None:
        learner = FixedBudgetFeatureLearner(
            n_features=1,
            n_tasks=4,
            candidate_count=0,
            utility_aggregation="topk",
            utility_top_k=2,
            utility_task_balancing="active",
        )
        active_mask = jnp.zeros((4,), dtype=jnp.bool_)
        activity = jnp.ones((4,), dtype=jnp.float32)
        if reducer_name == "_aggregate_task_feature_signal":
            args: tuple[Any, ...] = (
                jnp.asarray(((1.0,), (2.0,), (3.0,), (4.0,)), dtype=jnp.float32),
                active_mask,
                activity,
            )
        else:
            args = (
                jnp.ones((4, 1), dtype=jnp.float32),
                jnp.ones((1,), dtype=jnp.float32),
                active_mask,
                activity,
            )
        checked = checkify.checkify(getattr(learner, reducer_name), errors=checkify.float_checks)
        error, utility = checked(*args)
        error.throw()
        np.testing.assert_array_equal(utility, np.zeros((1,), dtype=np.float32))

    def test_unbalanced_topk_still_averages_largest_heads(self) -> None:
        learner = FixedBudgetFeatureLearner(
            n_features=1,
            n_tasks=4,
            candidate_count=0,
            utility_aggregation="topk",
            utility_top_k=2,
            utility_task_balancing="none",
        )
        signal = jnp.asarray(((4.0,), (2.0,), (0.5,), (0.0,)), dtype=jnp.float32)
        utility = learner._aggregate_task_feature_signal(
            signal, jnp.ones((4,), dtype=jnp.bool_), jnp.ones((4,), dtype=jnp.float32)
        )
        np.testing.assert_array_equal(utility, np.asarray((3.0,), dtype=np.float32))

    def test_init_shapes(self) -> None:
        learner = FixedBudgetFeatureLearner(
            n_features=7,
            n_tasks=3,
            candidate_count=4,
        )
        state = learner.init(feature_dim=5, key=jr.key(2))

        chex.assert_shape(state.feature_weights, (7, 5))
        chex.assert_shape(state.output_weights, (3, 7))
        chex.assert_shape(state.utilities, (7,))
        chex.assert_shape(state.task_activity_ema, (3,))
        chex.assert_shape(state.candidate_weights, (4, 5))
        chex.assert_shape(state.candidate_output_weights, (3, 4))

    def test_update_returns_finite_metrics(self) -> None:
        learner = FixedBudgetFeatureLearner(
            n_features=8,
            n_tasks=2,
            candidate_count=3,
            replacement_interval=10,
        )
        state = learner.init(feature_dim=4, key=jr.key(3))

        result = learner.update(
            state,
            jnp.array([0.1, -0.2, 0.3, 0.4], dtype=jnp.float32),
            jnp.array([1.0, -1.0], dtype=jnp.float32),
        )

        chex.assert_shape(result.predictions, (2,))
        chex.assert_shape(result.errors, (2,))
        chex.assert_shape(result.metrics, (7,))
        chex.assert_tree_all_finite(result.metrics)
        assert int(result.state.step_count) == 1

    def test_zero_utility_decay_does_not_multiply_inf_utilities(self) -> None:
        """utility_decay=0 times an infinite tracker is NaN and would freeze."""
        learner = FixedBudgetFeatureLearner(
            n_features=4,
            n_tasks=2,
            candidate_count=2,
            replacement_interval=0,
            utility_decay=0.0,
        )
        state = learner.init(feature_dim=3, key=jr.key(4))
        state = state.replace(
            utilities=jnp.full_like(state.utilities, jnp.inf),
            candidate_utilities=jnp.full_like(state.candidate_utilities, jnp.inf),
        )
        raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
        assert not bool(jnp.isfinite(raw))

        result = learner.update(
            state,
            jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32),
            jnp.array([1.0, -1.0], dtype=jnp.float32),
        )
        assert bool(result.update_applied)
        assert bool(jnp.all(jnp.isfinite(result.state.utilities)))
        assert bool(jnp.all(jnp.isfinite(result.state.candidate_utilities)))

    def test_zero_decays_recover_poisoned_forgotten_trackers(self) -> None:
        learner = FixedBudgetFeatureLearner(
            n_features=3,
            n_tasks=2,
            candidate_count=2,
            replacement_interval=0,
            utility_decay=0.0,
            task_activity_decay=0.0,
            learn_feature_resources=True,
            resource_discount=0.0,
        )
        state = learner.init(feature_dim=3, key=jr.key(40)).replace(
            utilities=jnp.full((3,), jnp.inf, dtype=jnp.float32),
            candidate_utilities=jnp.full((2,), -jnp.inf, dtype=jnp.float32),
            task_activity_ema=jnp.array([jnp.inf, jnp.nan], dtype=jnp.float32),
            generator_log_weights=jnp.array(
                [jnp.inf, jnp.nan, -jnp.inf], dtype=jnp.float32
            ),
            generator_utility_ema=jnp.array(
                [jnp.inf, jnp.nan, -jnp.inf], dtype=jnp.float32
            ),
            plasticity_log_weights=jnp.array(
                [jnp.nan, jnp.inf, -jnp.inf], dtype=jnp.float32
            ),
            plasticity_signal_ema=jnp.array(
                [jnp.nan, jnp.inf, -jnp.inf], dtype=jnp.float32
            ),
            birth_timestamp=0.0,
        )
        recovered_weights = learner._resource_weights(state.generator_log_weights)

        chex.assert_tree_all_finite(recovered_weights)
        assert float(jnp.sum(recovered_weights)) == pytest.approx(1.0)

        result = jax.jit(learner.update)(
            state,
            jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32),
            jnp.array([1.0, -1.0], dtype=jnp.float32),
        )

        assert bool(result.update_applied)
        assert bool(floating_tree_is_finite(result.state))
        chex.assert_tree_all_finite(result.metrics)
        chex.assert_trees_all_equal(
            result.state.generator_utility_ema[1:],
            jnp.zeros((2,), dtype=jnp.float32),
        )

    def test_zero_utility_decay_rejects_poison_consumed_by_retention(self) -> None:
        learner = FixedBudgetFeatureLearner(
            n_features=1,
            n_tasks=1,
            replacement_interval=1,
            min_feature_age=0,
            utility_decay=0.0,
            utility_retention_decay=0.9,
        )
        state = learner.init(feature_dim=2, key=jr.key(41)).replace(
            utilities=jnp.array([jnp.inf], dtype=jnp.float32),
            birth_timestamp=0.0,
        )

        result = jax.jit(learner.update)(
            state,
            jnp.ones((2,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        )

        assert not bool(result.update_applied)
        chex.assert_trees_all_equal(result.state, state)
        chex.assert_trees_all_equal(result.predictions, jnp.zeros((1,), jnp.float32))
        chex.assert_trees_all_equal(result.errors, jnp.zeros((1,), jnp.float32))
        chex.assert_trees_all_equal(result.metrics, jnp.zeros((7,), jnp.float32))
        assert int(result.replaced_slot) == -1

    def test_zero_resource_discount_preserves_finite_resource_state_semantics(
        self,
    ) -> None:
        learner = FixedBudgetFeatureLearner(
            n_features=2,
            n_tasks=1,
            replacement_interval=0,
            learn_feature_resources=True,
            resource_discount=0.0,
            resource_exploration=0.0,
        )
        logits = jnp.array([2.0, -1.0, 0.5], dtype=jnp.float32)
        state = learner.init(feature_dim=2, key=jr.key(42)).replace(
            generator_log_weights=logits,
            generator_utility_ema=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        )

        chex.assert_trees_all_equal(
            learner._resource_weights(logits),
            jax.nn.softmax(logits),
        )
        result = learner.update(
            state,
            jnp.ones((2,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        )

        assert bool(result.update_applied)
        # Only generator zero has members; unavailable finite EMA slots retain
        # their previous values even though the configured discount is zero.
        chex.assert_trees_all_equal(
            result.state.generator_utility_ema[1:],
            state.generator_utility_ema[1:],
        )

    def test_constructed_and_augmented_feature_shapes(self) -> None:
        learner = FixedBudgetFeatureLearner(n_features=6, n_tasks=2)
        state = learner.init(feature_dim=4, key=jr.key(14))
        observation = jnp.array([0.1, -0.2, 0.3, 0.4], dtype=jnp.float32)

        features = learner.constructed_features(state, observation)
        augmented = learner.augmented_observation(state, observation)

        chex.assert_shape(features, (6,))
        chex.assert_shape(augmented, (10,))
        chex.assert_tree_all_finite(features)
        chex.assert_tree_all_finite(augmented)

    def test_random_replacement_event_occurs(self) -> None:
        learner = FixedBudgetFeatureLearner(
            n_features=5,
            n_tasks=2,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=0,
            generator_mix=(1.0, 0.0, 0.0),
        )
        state = learner.init(feature_dim=4, key=jr.key(4))
        result = learner.update(
            state,
            jnp.ones(4, dtype=jnp.float32),
            jnp.array([0.5, -0.25], dtype=jnp.float32),
        )

        assert float(result.metrics[5]) == 1.0
        assert int(result.replaced_slot) >= 0
        assert int(result.state.ages[result.replaced_slot]) == 0
        assert int(result.state.feature_generator[result.replaced_slot]) in {
            GENERATOR_RANDOM,
            GENERATOR_MUTATE_PARENT,
            GENERATOR_IMPRINT,
        }

    def test_age_corrected_promotion_restarts_raw_utility_ema(self) -> None:
        learner = FixedBudgetFeatureLearner(
            n_features=1,
            n_tasks=1,
            step_size_output=0.0,
            step_size_feature=0.0,
            utility_decay=0.5,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=1,
            candidate_min_age=0,
            promotion_margin=1.0,
            future_utility_mix=1.0,
            future_utility_normalization="age",
            use_obgd=False,
        )
        state = learner.init(feature_dim=2, key=jr.key(431)).replace(
            utilities=jnp.array([0.1], dtype=jnp.float32),
            ages=jnp.array([5], dtype=jnp.int32),
            candidate_utilities=jnp.array([1.0], dtype=jnp.float32),
            candidate_ages=jnp.array([5], dtype=jnp.int32),
        )

        promoted = jax.jit(learner.update)(
            state,
            jnp.ones((2,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
        )

        assert bool(promoted.update_applied)
        assert int(promoted.promoted_candidate) == 0
        assert int(promoted.state.ages[0]) == 0
        assert float(promoted.state.utilities[0]) == 0.0
        assert float(promoted.state.candidate_utilities[0]) == 0.0
        chex.assert_tree_all_finite(promoted.metrics)

    def test_scan_loop_shapes(self) -> None:
        stream = NonlinearFeatureDiscoveryStream(
            feature_dim=4,
            n_tasks=2,
            n_latents=8,
            context_length=8,
        )
        learner = FixedBudgetFeatureLearner(
            n_features=8,
            n_tasks=2,
            replacement_interval=5,
            min_feature_age=3,
            candidate_count=2,
            candidate_min_age=2,
        )
        result = run_feature_discovery_loop(learner, stream, num_steps=15, key=jr.key(5))

        chex.assert_shape(result.metrics, (15, 7))
        chex.assert_tree_all_finite(result.metrics)
        assert int(result.state.step_count) == 15

    def test_array_loop_shapes(self) -> None:
        stream = NonlinearFeatureDiscoveryStream(
            feature_dim=4,
            n_tasks=2,
            n_latents=8,
        )
        observations, targets = collect_feature_discovery_stream(
            stream, num_steps=10, key=jr.key(6)
        )
        learner = FixedBudgetFeatureLearner(
            n_features=8,
            n_tasks=2,
            replacement_interval=0,
        )
        state = learner.init(feature_dim=4, key=jr.key(7))
        result = run_feature_discovery_arrays(learner, state, observations, targets)

        chex.assert_shape(result.metrics, (10, 7))
        chex.assert_tree_all_finite(result.metrics)


def _conditional_probe_learner() -> FixedBudgetInteractionLearner:
    return FixedBudgetInteractionLearner(
        n_features=2,
        n_tasks=1,
        step_size_output=0.1,
        utility_decay=0.0,
        utility_retention_grace_steps=8,
        utility_evidence_threshold=0.01,
        evidence_gated_active_output_memory=True,
        utility_evidence_confirmation_steps=1,
        independent_relevance_probe=True,
        replacement_interval=0,
        candidate_count=1,
        candidate_min_age=0,
        refresh_candidates=False,
        refresh_promoted_candidate=False,
        scale_robust=True,
    )


def test_interaction_update_contract_is_static_and_dynamic_failure_is_atomic() -> None:
    learner = _conditional_probe_learner()
    state = learner.init(feature_dim=3, key=jr.key(700))
    observation = jnp.ones((3,), dtype=jnp.float32)
    target = jnp.ones((1,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="observation must be rank one"):
        learner.update(state, observation.reshape((1, 3)), target)
    with pytest.raises(ValueError, match="targets must have shape"):
        learner.update(state, observation, target.reshape((1, 1)))
    with pytest.raises(TypeError, match="observation must have dtype float32"):
        learner.update(state, observation.astype(jnp.int32), target)
    with pytest.raises(TypeError, match="external_read_mask must have dtype bool"):
        learner.update(
            state,
            observation,
            target,
            jnp.ones((2,), dtype=jnp.int32),
        )

    invalid_observation = learner.update(
        state,
        observation.at[0].set(jnp.inf),
        target,
    )
    invalid_target = learner.update(
        state,
        observation,
        target.at[0].set(jnp.inf),
    )
    for result in (invalid_observation, invalid_target):
        assert bool(result.update_rejected)
        chex.assert_trees_all_equal(result.state, state)
        chex.assert_trees_all_equal(result.pre_curation_state, state)
        assert int(result.replaced_slot) == -1
        assert int(result.promoted_candidate) == -1
        assert int(result.refreshed_candidate) == -1
        assert int(result.retired_slot) == -1

    compiled_invalid = jax.jit(learner.update)(
        state,
        observation.at[0].set(jnp.inf),
        target,
    )
    chex.assert_trees_all_equal(compiled_invalid, invalid_observation)

    invalid_observations = jnp.stack(
        (
            observation.at[0].set(jnp.inf),
            observation.at[1].set(-jnp.inf),
        )
    )

    def reject_step(
        carry: Any,
        invalid_input: Any,
    ) -> tuple[Any, tuple[Any, Any]]:
        rejected = learner.update(carry, invalid_input, target)
        return rejected.state, (
            rejected.pre_curation_state,
            rejected.update_rejected,
        )

    final_state, (pre_curation_states, rejected) = jax.jit(
        lambda initial: jax.lax.scan(
            reject_step,
            initial,
            invalid_observations,
        )
    )(state)
    chex.assert_trees_all_equal(final_state, state)
    chex.assert_trees_all_equal(
        pre_curation_states,
        jax.tree_util.tree_map(
            lambda leaf: jnp.stack((leaf, leaf)),
            state,
        ),
    )
    chex.assert_trees_all_equal(rejected, jnp.ones((2,), dtype=jnp.bool_))

    missing_target = learner.update(
        state,
        observation,
        jnp.asarray([jnp.nan], dtype=jnp.float32),
    )
    assert not bool(missing_target.update_rejected)
    assert int(missing_target.state.step_count) == 1


def test_curation_priority_override_has_exact_no_cadence_learning_parity() -> None:
    learner = FixedBudgetInteractionLearner(
        n_features=2,
        n_tasks=1,
        replacement_interval=4,
        min_feature_age=0,
        candidate_count=2,
        candidate_min_age=0,
        refresh_candidates=True,
        refresh_promoted_candidate=False,
        use_obgd=False,
    )
    state = learner.init(feature_dim=4, key=jr.key(702)).replace(
        feature_left=jnp.asarray((0, 0), dtype=jnp.int32),
        feature_right=jnp.asarray((1, 2), dtype=jnp.int32),
        candidate_left=jnp.asarray((1, 2), dtype=jnp.int32),
        candidate_right=jnp.asarray((3, 3), dtype=jnp.int32),
        utilities=jnp.asarray((0.25, 0.5), dtype=jnp.float32),
        candidate_utilities=jnp.asarray((2.0, 3.0), dtype=jnp.float32),
    )
    active_ranks = jnp.asarray((10.0, -7.0), dtype=jnp.float32)
    candidate_ranks = jnp.asarray((-50.0, 80.0), dtype=jnp.float32)
    disabled = InteractionCurationPriorityOverride(
        enabled=jnp.asarray(False, dtype=jnp.bool_),
        active_ranks=active_ranks,
        candidate_ranks=candidate_ranks,
    )
    enabled = disabled.replace(enabled=jnp.asarray(True, dtype=jnp.bool_))
    observation = jnp.ones((4,), dtype=jnp.float32)
    target = jnp.ones((1,), dtype=jnp.float32)

    full = learner.update(
        state,
        observation,
        target,
        curation_priority_override=disabled,
    )
    random = learner.update(
        state,
        observation,
        target,
        curation_priority_override=enabled,
    )

    chex.assert_trees_all_equal(full.pre_curation_state, random.pre_curation_state)
    chex.assert_trees_all_equal(full.state, random.state)
    assert not bool(full.curation_attempted)
    assert not bool(random.curation_attempted)
    assert not bool(full.curation_priority_override_applied)
    assert not bool(random.curation_priority_override_applied)
    assert not bool(jnp.array_equal(random.state.utilities, active_ranks))
    assert not bool(
        jnp.array_equal(random.state.candidate_utilities, candidate_ranks)
    )


def test_curation_priority_override_changes_only_forced_transaction_selection() -> None:
    learner = FixedBudgetInteractionLearner(
        n_features=2,
        n_tasks=1,
        step_size_output=0.0,
        utility_decay=0.999,
        replacement_interval=1,
        min_feature_age=0,
        candidate_count=2,
        candidate_min_age=0,
        promotion_margin=1.0,
        refresh_candidates=False,
        refresh_promoted_candidate=False,
        use_obgd=False,
    )
    state = learner.init(feature_dim=4, key=jr.key(703)).replace(
        feature_left=jnp.asarray((0, 0), dtype=jnp.int32),
        feature_right=jnp.asarray((1, 2), dtype=jnp.int32),
        utilities=jnp.asarray((0.0, 10.0), dtype=jnp.float32),
        ages=jnp.asarray((5, 5), dtype=jnp.int32),
        candidate_left=jnp.asarray((1, 0), dtype=jnp.int32),
        candidate_right=jnp.asarray((2, 3), dtype=jnp.int32),
        candidate_utilities=jnp.asarray((100.0, 20.0), dtype=jnp.float32),
        candidate_ages=jnp.asarray((5, 5), dtype=jnp.int32),
    )
    active_ranks = jnp.asarray((10.0, 0.0), dtype=jnp.float32)
    candidate_ranks = jnp.asarray((0.0, 10.0), dtype=jnp.float32)
    full_override = InteractionCurationPriorityOverride(
        enabled=jnp.asarray(False, dtype=jnp.bool_),
        active_ranks=active_ranks,
        candidate_ranks=candidate_ranks,
    )
    random_override = full_override.replace(
        enabled=jnp.asarray(True, dtype=jnp.bool_)
    )
    observation = jnp.ones((4,), dtype=jnp.float32)
    target = jnp.zeros((1,), dtype=jnp.float32)

    full = learner.update(
        state,
        observation,
        target,
        curation_priority_override=full_override,
    )
    random = learner.update(
        state,
        observation,
        target,
        curation_priority_override=random_override,
    )
    compiled_random = jax.jit(learner.update)(
        state,
        observation,
        target,
        curation_priority_override=random_override,
    )

    chex.assert_trees_all_equal(full.pre_curation_state, random.pre_curation_state)
    chex.assert_trees_all_equal(random, compiled_random)
    assert int(full.curation_selected_active_worst_slot) == 0
    assert int(random.curation_selected_active_worst_slot) == 1
    assert int(full.curation_selected_promotion_candidate) == 0
    assert int(random.curation_selected_promotion_candidate) == 1
    assert int(full.replaced_slot) == 0
    assert int(random.replaced_slot) == 1
    assert int(full.promoted_candidate) == 0
    assert int(random.promoted_candidate) == 1
    assert bool(random.curation_priority_override_applied)
    assert not bool(jnp.array_equal(random.state.utilities, active_ranks))
    assert not bool(
        jnp.array_equal(random.state.candidate_utilities, candidate_ranks)
    )


def test_curation_priority_override_controls_candidate_refresh_argmin() -> None:
    learner = FixedBudgetInteractionLearner(
        n_features=1,
        n_tasks=1,
        step_size_output=0.0,
        replacement_interval=1,
        min_feature_age=0,
        candidate_count=2,
        candidate_min_age=100,
        refresh_candidates=True,
        refresh_promoted_candidate=False,
        use_obgd=False,
    )
    state = learner.init(feature_dim=4, key=jr.key(704)).replace(
        feature_left=jnp.asarray((0,), dtype=jnp.int32),
        feature_right=jnp.asarray((1,), dtype=jnp.int32),
        ages=jnp.asarray((5,), dtype=jnp.int32),
        candidate_left=jnp.asarray((1, 2), dtype=jnp.int32),
        candidate_right=jnp.asarray((3, 3), dtype=jnp.int32),
        candidate_utilities=jnp.asarray((0.0, 10.0), dtype=jnp.float32),
        candidate_ages=jnp.asarray((0, 0), dtype=jnp.int32),
    )
    ranks = InteractionCurationPriorityOverride(
        enabled=jnp.asarray(False, dtype=jnp.bool_),
        active_ranks=jnp.asarray((0.0,), dtype=jnp.float32),
        candidate_ranks=jnp.asarray((10.0, 0.0), dtype=jnp.float32),
    )
    observation = jnp.ones((4,), dtype=jnp.float32)
    target = jnp.zeros((1,), dtype=jnp.float32)

    full = learner.update(
        state,
        observation,
        target,
        curation_priority_override=ranks,
    )
    random = learner.update(
        state,
        observation,
        target,
        curation_priority_override=ranks.replace(
            enabled=jnp.asarray(True, dtype=jnp.bool_)
        ),
    )

    chex.assert_trees_all_equal(full.pre_curation_state, random.pre_curation_state)
    assert int(full.refreshed_candidate) == 0
    assert int(random.refreshed_candidate) == 1
    assert int(full.curation_selected_refresh_candidate) == 0
    assert int(random.curation_selected_refresh_candidate) == 1


def test_nonfinite_curation_priority_override_rejects_atomically_in_all_modes() -> None:
    learner = _conditional_probe_learner()
    state = learner.init(feature_dim=3, key=jr.key(705))
    observation = jnp.ones((3,), dtype=jnp.float32)
    target = jnp.ones((1,), dtype=jnp.float32)
    invalid_override = InteractionCurationPriorityOverride(
        enabled=jnp.asarray(True, dtype=jnp.bool_),
        active_ranks=jnp.asarray((0.0, jnp.nan), dtype=jnp.float32),
        candidate_ranks=jnp.asarray((0.0,), dtype=jnp.float32),
    )

    eager = learner.update(
        state,
        observation,
        target,
        curation_priority_override=invalid_override,
    )
    compiled = jax.jit(learner.update)(
        state,
        observation,
        target,
        curation_priority_override=invalid_override,
    )
    chex.assert_trees_all_equal(eager, compiled)
    chex.assert_trees_all_equal(eager.state, state)
    chex.assert_trees_all_equal(eager.pre_curation_state, state)
    assert bool(eager.update_rejected)
    assert not bool(eager.curation_priority_override_applied)

    def reject_step(carry: Any, _: Any) -> tuple[Any, Any]:
        result = learner.update(
            carry,
            observation,
            target,
            curation_priority_override=invalid_override,
        )
        return result.state, (
            result.pre_curation_state,
            result.update_rejected,
        )

    final_state, (pre_curation_states, rejected) = jax.jit(
        lambda initial: jax.lax.scan(reject_step, initial, xs=None, length=2)
    )(state)
    chex.assert_trees_all_equal(final_state, state)
    chex.assert_trees_all_equal(
        pre_curation_states,
        jax.tree_util.tree_map(lambda leaf: jnp.stack((leaf, leaf)), state),
    )
    chex.assert_trees_all_equal(rejected, jnp.ones((2,), dtype=jnp.bool_))

    with pytest.raises(ValueError, match="active_ranks"):
        learner.update(
            state,
            observation,
            target,
            curation_priority_override=invalid_override.replace(
                active_ranks=jnp.zeros((3,), dtype=jnp.float32)
            ),
        )
    with pytest.raises(TypeError, match="candidate_ranks"):
        learner.update(
            state,
            observation,
            target,
            curation_priority_override=invalid_override.replace(
                candidate_ranks=jnp.zeros((1,), dtype=jnp.int32)
            ),
        )
    with pytest.raises(TypeError, match="enabled"):
        learner.update(
            state,
            observation,
            target,
            curation_priority_override=invalid_override.replace(
                enabled=jnp.asarray(1.0, dtype=jnp.float32)
            ),
        )


def test_conditional_candidate_relevance_rejects_collinear_redundancy() -> None:
    learner = _conditional_probe_learner()
    base = learner.init(feature_dim=3, key=jr.key(701)).replace(
        feature_left=jnp.asarray((0, 1), dtype=jnp.int32),
        feature_right=jnp.asarray((1, 2), dtype=jnp.int32),
        output_weights=jnp.asarray(((1.0, 0.0),), dtype=jnp.float32),
        relevance_probe_weights=jnp.asarray(((1.0, 1.0),), dtype=jnp.float32),
        active_output_memory_committed=jnp.asarray((True, True), dtype=jnp.bool_),
        candidate_left=jnp.asarray((0,), dtype=jnp.int32),
        candidate_right=jnp.asarray((2,), dtype=jnp.int32),
        candidate_output_weights=jnp.asarray(((1.0,),), dtype=jnp.float32),
        feature_second_moments=jnp.ones((2,), dtype=jnp.float32),
        candidate_second_moments=jnp.ones((1,), dtype=jnp.float32),
        target_second_moments=jnp.ones((1,), dtype=jnp.float32),
    )
    observation = jnp.ones((3,), dtype=jnp.float32)
    target = jnp.ones((1,), dtype=jnp.float32)

    redundant = learner.update(base, observation, target)
    absent_bank = learner.update(
        base.replace(output_weights=jnp.zeros((1, 2), dtype=jnp.float32)),
        observation,
        target,
    )

    assert not bool(redundant.update_rejected)
    assert float(redundant.candidate_promotion_signal[0]) == 0.0
    assert not bool(redundant.candidate_promotion_raw_evidence[0])
    assert float(absent_bank.candidate_promotion_signal[0]) > 0.0
    assert bool(absent_bank.candidate_promotion_raw_evidence[0])
    assert float(redundant.relevance_probe_scores[0]) > 0.0
    assert float(redundant.relevance_probe_scores[1]) == 0.0


def test_default_probe_mode_is_bit_exact_to_explicit_conditional_v1() -> None:
    common = {
        "n_features": 2,
        "n_tasks": 1,
        "step_size_output": 0.1,
        "utility_decay": 0.0,
        "utility_retention_grace_steps": 8,
        "utility_evidence_threshold": 0.01,
        "evidence_gated_active_output_memory": True,
        "utility_evidence_confirmation_steps": 1,
        "independent_relevance_probe": True,
        "replacement_interval": 0,
        "candidate_count": 1,
        "candidate_min_age": 0,
        "refresh_candidates": False,
        "refresh_promoted_candidate": False,
        "scale_robust": True,
    }
    default = FixedBudgetInteractionLearner(**common)
    explicit = FixedBudgetInteractionLearner(
        **common,
        relevance_probe_mode=RELEVANCE_PROBE_MODE_CONDITIONAL_V1,
    )
    default_state = default.init(feature_dim=3, key=jr.key(703))
    explicit_state = explicit.init(feature_dim=3, key=jr.key(703))
    observation = jnp.asarray((0.5, -1.0, 2.0), dtype=jnp.float32)
    target = jnp.asarray((0.75,), dtype=jnp.float32)

    chex.assert_trees_all_equal(default_state, explicit_state)
    chex.assert_trees_all_equal(
        default.update(default_state, observation, target),
        explicit.update(explicit_state, observation, target),
    )
    assert default.to_config() == explicit.to_config()
    assert default.to_config()["relevance_probe_mode"] == "conditional_v1"


def test_target_only_probe_is_invariant_to_durable_bank_and_learns_separate_bias() -> None:
    learner = FixedBudgetInteractionLearner(
        n_features=2,
        n_tasks=1,
        step_size_output=0.1,
        utility_decay=0.0,
        utility_retention_grace_steps=8,
        utility_evidence_threshold=0.01,
        evidence_gated_active_output_memory=True,
        utility_evidence_confirmation_steps=2,
        independent_relevance_probe=True,
        relevance_probe_mode=RELEVANCE_PROBE_MODE_TARGET_ONLY_V1,
        replacement_interval=0,
        candidate_count=1,
        candidate_min_age=0,
        refresh_candidates=False,
        refresh_promoted_candidate=False,
        scale_robust=True,
    )
    base = learner.init(feature_dim=3, key=jr.key(704)).replace(
        feature_left=jnp.asarray((0, 0), dtype=jnp.int32),
        feature_right=jnp.asarray((1, 2), dtype=jnp.int32),
        output_weights=jnp.asarray(((0.0, 0.0),), dtype=jnp.float32),
        output_biases=jnp.asarray((0.3,), dtype=jnp.float32),
        relevance_probe_weights=jnp.asarray(((0.5, 0.25),), dtype=jnp.float32),
        relevance_probe_biases=jnp.asarray((0.2,), dtype=jnp.float32),
        active_output_memory_committed=jnp.asarray((True, True), dtype=jnp.bool_),
        candidate_left=jnp.asarray((1,), dtype=jnp.int32),
        candidate_right=jnp.asarray((2,), dtype=jnp.int32),
        candidate_output_weights=jnp.asarray(((0.5,),), dtype=jnp.float32),
        feature_second_moments=jnp.ones((2,), dtype=jnp.float32),
        candidate_second_moments=jnp.ones((1,), dtype=jnp.float32),
        target_second_moments=jnp.ones((1,), dtype=jnp.float32),
    )
    changed_bank = base.replace(
        output_weights=jnp.asarray(((2.0, -0.5),), dtype=jnp.float32)
    )
    observation = jnp.ones((3,), dtype=jnp.float32)
    target = jnp.ones((1,), dtype=jnp.float32)
    reference = learner.update(base, observation, target)
    changed = learner.update(changed_bank, observation, target)
    changed_probe = learner.update(
        base.replace(
            relevance_probe_weights=base.relevance_probe_weights.at[0, 0].set(9.0)
        ),
        observation,
        target,
    )

    assert float(reference.predictions[0]) != float(changed.predictions[0])
    chex.assert_trees_all_equal(
        reference.relevance_probe_errors,
        jnp.full((1, 2), 0.8, dtype=jnp.float32),
    )
    chex.assert_trees_all_equal(
        reference.relevance_probe_errors,
        changed.relevance_probe_errors,
    )
    chex.assert_trees_all_equal(
        reference.relevance_probe_scores,
        changed.relevance_probe_scores,
    )
    chex.assert_trees_all_equal(
        reference.state.relevance_probe_weights,
        changed.state.relevance_probe_weights,
    )
    chex.assert_trees_all_equal(
        reference.candidate_promotion_signal,
        changed.candidate_promotion_signal,
    )
    chex.assert_trees_all_equal(
        reference.state.candidate_output_weights,
        changed.state.candidate_output_weights,
    )
    chex.assert_trees_all_equal(
        reference.state.relevance_probe_biases,
        changed.state.relevance_probe_biases,
    )
    chex.assert_trees_all_equal(
        reference.relevance_probe_errors,
        changed_probe.relevance_probe_errors,
    )
    chex.assert_trees_all_equal(
        reference.relevance_probe_scores[1:],
        changed_probe.relevance_probe_scores[1:],
    )
    chex.assert_trees_all_equal(
        reference.state.relevance_probe_weights[:, 1:],
        changed_probe.state.relevance_probe_weights[:, 1:],
    )
    chex.assert_trees_all_equal(
        reference.state.candidate_output_weights,
        changed_probe.state.candidate_output_weights,
    )
    assert float(reference.state.relevance_probe_biases[0]) == pytest.approx(0.28)
    assert float(reference.state.relevance_probe_biases[0]) != float(
        reference.state.output_biases[0]
    )


def test_probe_modes_have_identical_fixed_resource_shape() -> None:
    common = {
        "n_features": 3,
        "n_tasks": 2,
        "candidate_count": 4,
        "utility_retention_grace_steps": 8,
        "utility_evidence_threshold": 0.1,
        "evidence_gated_active_output_memory": True,
        "independent_relevance_probe": True,
        "scale_robust": True,
    }
    conditional = FixedBudgetInteractionLearner(
        **common,
        relevance_probe_mode=RELEVANCE_PROBE_MODE_CONDITIONAL_V1,
    )
    target_only = FixedBudgetInteractionLearner(
        **common,
        relevance_probe_mode=RELEVANCE_PROBE_MODE_TARGET_ONLY_V1,
    )
    conditional_state = conditional.init(feature_dim=3, key=jr.key(705))
    target_only_state = target_only.init(feature_dim=3, key=jr.key(705))

    chex.assert_trees_all_equal(conditional_state, target_only_state)
    assert conditional.memory_accounting(conditional_state) == target_only.memory_accounting(
        target_only_state
    )


def test_probe_mode_config_and_checkpoint_round_trip_fail_closed(
    tmp_path: Path,
) -> None:
    learner = FixedBudgetInteractionLearner(
        n_features=2,
        n_tasks=1,
        candidate_count=1,
        utility_retention_grace_steps=8,
        utility_evidence_threshold=0.1,
        evidence_gated_active_output_memory=True,
        independent_relevance_probe=True,
        relevance_probe_mode=RELEVANCE_PROBE_MODE_TARGET_ONLY_V1,
        scale_robust=True,
    )
    restored_config = FixedBudgetInteractionLearner.from_config(learner.to_config())
    assert restored_config.to_config()["relevance_probe_mode"] == "target_only_v1"
    state = learner.init(feature_dim=3, key=jr.key(706))
    updated = learner.update(
        state,
        jnp.asarray((0.5, -1.0, 2.0), dtype=jnp.float32),
        jnp.asarray((0.75,), dtype=jnp.float32),
    ).state
    path = tmp_path / "target_only"
    save_interaction_feature_checkpoint(learner, updated, path, feature_dim=3)
    metadata = load_checkpoint_metadata(path)
    assert metadata["learner_config"]["relevance_probe_mode"] == "target_only_v1"
    loaded_learner, loaded_state = load_interaction_feature_checkpoint(path)
    assert loaded_learner.to_config()["relevance_probe_mode"] == "target_only_v1"
    chex.assert_trees_all_equal(loaded_state, updated)

    ambiguous = learner.to_config()
    ambiguous.pop("relevance_probe_mode")
    with pytest.raises(ValueError, match="ambiguous"):
        FixedBudgetInteractionLearner.from_config(ambiguous)
    disabled_legacy = FixedBudgetInteractionLearner(
        n_features=1,
        n_tasks=1,
    ).to_config()
    disabled_legacy.pop("relevance_probe_mode")
    migrated = FixedBudgetInteractionLearner.from_config(disabled_legacy)
    assert migrated.to_config()["relevance_probe_mode"] == "conditional_v1"
    unknown = learner.to_config()
    unknown["relevance_probe_mode"] = "unknown_v1"
    with pytest.raises(ValueError, match="relevance_probe_mode"):
        FixedBudgetInteractionLearner.from_config(unknown)

    invalid_metadata = dict(metadata)
    invalid_config = dict(invalid_metadata["learner_config"])
    invalid_config["relevance_probe_mode"] = "unknown_v1"
    invalid_metadata["learner_config"] = invalid_config
    invalid_path = tmp_path / "invalid_mode"
    save_checkpoint(updated, invalid_path, metadata=invalid_metadata)
    with pytest.raises(ValueError, match="relevance_probe_mode"):
        load_interaction_feature_checkpoint(invalid_path)


def test_target_only_probe_runs_under_jit_and_scan() -> None:
    learner = FixedBudgetInteractionLearner(
        n_features=2,
        n_tasks=1,
        candidate_count=1,
        utility_retention_grace_steps=8,
        utility_evidence_threshold=0.01,
        evidence_gated_active_output_memory=True,
        independent_relevance_probe=True,
        relevance_probe_mode=RELEVANCE_PROBE_MODE_TARGET_ONLY_V1,
        replacement_interval=0,
        scale_robust=True,
    )
    state = learner.init(feature_dim=3, key=jr.key(707))
    observation = jnp.asarray((0.5, -1.0, 2.0), dtype=jnp.float32)
    target = jnp.asarray((0.75,), dtype=jnp.float32)
    jitted = jax.jit(learner.update)(state, observation, target)
    scanned = run_interaction_feature_arrays(
        learner,
        state,
        jnp.stack((observation, observation)),
        jnp.stack((target, target)),
    )

    assert not bool(jitted.update_rejected)
    chex.assert_tree_all_finite(jitted.metrics)
    chex.assert_tree_all_finite(jitted.state.relevance_probe_weights)
    chex.assert_tree_all_finite(jitted.state.relevance_probe_biases)
    chex.assert_shape(scanned.metrics, (2, 7))
    chex.assert_tree_all_finite(scanned.metrics)
    assert int(scanned.state.step_count) == 2


class TestFixedBudgetInteractionLearner:
    """Tests for pairwise feature construction, utility, and replacement."""

    def test_init_shapes(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=7,
            n_tasks=3,
            candidate_count=4,
        )
        state = learner.init(feature_dim=5, key=jr.key(9))

        chex.assert_shape(state.feature_left, (7,))
        chex.assert_shape(state.feature_right, (7,))
        chex.assert_shape(state.output_weights, (3, 7))
        chex.assert_shape(state.utilities, (7,))
        chex.assert_shape(state.utility_evidence_streak, (7,))
        chex.assert_shape(state.active_output_memory_committed, (7,))
        assert state.utility_evidence_streak.dtype == jnp.int32
        assert state.active_output_memory_committed.dtype == jnp.bool_
        chex.assert_shape(state.task_activity_ema, (3,))
        chex.assert_shape(state.candidate_left, (4,))
        chex.assert_shape(state.candidate_output_weights, (3, 4))

    def test_all_pairs_candidate_strategy_covers_pair_space(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=4,
            n_tasks=2,
            candidate_count=6,
            candidate_strategy="all_pairs",
            refresh_candidates=False,
            refresh_promoted_candidate=False,
        )
        state = learner.init(feature_dim=4, key=jr.key(14))

        candidate_pairs = {
            (int(left), int(right))
            for left, right in zip(state.candidate_left, state.candidate_right, strict=True)
        }

        assert candidate_pairs == {
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        }

    def test_candidate_matching_active_pair_cannot_waste_promotion_slot(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=2,
            n_tasks=1,
            utility_decay=0.999,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=2,
            candidate_min_age=0,
            promotion_margin=0.5,
            refresh_candidates=False,
            refresh_promoted_candidate=False,
            use_obgd=False,
        )
        state = learner.init(feature_dim=4, key=jr.key(30))
        state = state.replace(  # type: ignore[attr-defined]
            feature_left=jnp.array([0, 1], dtype=jnp.int32),
            feature_right=jnp.array([1, 2], dtype=jnp.int32),
            utilities=jnp.array([0.0, 1.0], dtype=jnp.float32),
            ages=jnp.array([10, 10], dtype=jnp.int32),
            candidate_left=jnp.array([0, 2], dtype=jnp.int32),
            candidate_right=jnp.array([1, 3], dtype=jnp.int32),
            candidate_utilities=jnp.array([100.0, 10.0], dtype=jnp.float32),
            candidate_ages=jnp.array([10, 10], dtype=jnp.int32),
        )

        result = learner.update(
            state,
            jnp.ones(4, dtype=jnp.float32),
            jnp.array([0.0], dtype=jnp.float32),
        )

        assert int(result.promoted_candidate) == 1
        promoted = int(result.replaced_slot)
        assert (
            int(result.state.feature_left[promoted]),
            int(result.state.feature_right[promoted]),
        ) == (2, 3)

    def test_constructed_and_augmented_feature_shapes(self) -> None:
        learner = FixedBudgetInteractionLearner(n_features=6, n_tasks=2)
        state = learner.init(feature_dim=4, key=jr.key(15))
        observation = jnp.array([0.1, -0.2, 0.3, 0.4], dtype=jnp.float32)

        features = learner.constructed_features(state, observation)
        augmented = learner.augmented_observation(state, observation)

        chex.assert_shape(features, (6,))
        chex.assert_shape(augmented, (10,))
        chex.assert_tree_all_finite(features)
        chex.assert_tree_all_finite(augmented)

    def test_update_returns_finite_metrics(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=8,
            n_tasks=2,
            candidate_count=3,
            replacement_interval=10,
        )
        state = learner.init(feature_dim=4, key=jr.key(10))

        result = learner.update(
            state,
            jnp.array([0.1, -0.2, 0.3, 0.4], dtype=jnp.float32),
            jnp.array([1.0, -1.0], dtype=jnp.float32),
        )

        chex.assert_shape(result.predictions, (2,))
        chex.assert_shape(result.errors, (2,))
        chex.assert_shape(result.metrics, (7,))
        chex.assert_tree_all_finite(result.metrics)
        assert int(result.state.step_count) == 1

    def test_max_utility_aggregation_does_not_dilute_rare_task_head(self) -> None:
        mean_learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=4,
            utility_decay=0.0,
            replacement_interval=0,
            utility_aggregation="mean",
        )
        max_learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=4,
            utility_decay=0.0,
            replacement_interval=0,
            utility_aggregation="max",
        )
        mean_state = mean_learner.init(feature_dim=2, key=jr.key(16))
        max_state = max_learner.init(feature_dim=2, key=jr.key(16))
        rare_head_weights = jnp.array([[2.0], [0.0], [0.0], [0.0]], dtype=jnp.float32)
        mean_state = mean_state.replace(output_weights=rare_head_weights)  # type: ignore[attr-defined]
        max_state = max_state.replace(output_weights=rare_head_weights)  # type: ignore[attr-defined]
        observation = jnp.ones(2, dtype=jnp.float32)
        targets = jnp.array([0.0, jnp.nan, jnp.nan, jnp.nan], dtype=jnp.float32)

        mean_result = mean_learner.update(mean_state, observation, targets)
        max_result = max_learner.update(max_state, observation, targets)

        assert float(mean_result.state.utilities[0]) == 0.5
        assert float(max_result.state.utilities[0]) == 2.0

    def test_topk_utility_aggregation_averages_largest_heads(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=4,
            utility_decay=0.0,
            replacement_interval=0,
            utility_aggregation="topk",
            utility_top_k=2,
        )
        state = learner.init(feature_dim=2, key=jr.key(18))
        state = state.replace(  # type: ignore[attr-defined]
            output_weights=jnp.array([[4.0], [2.0], [0.5], [0.0]], dtype=jnp.float32)
        )

        result = learner.update(
            state,
            jnp.ones(2, dtype=jnp.float32),
            jnp.zeros(4, dtype=jnp.float32),
        )

        assert float(result.state.utilities[0]) == 3.0

    def test_active_task_balancing_removes_nan_head_dilution(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=4,
            utility_decay=0.0,
            replacement_interval=0,
            utility_task_balancing="active",
        )
        state = learner.init(feature_dim=2, key=jr.key(19))
        state = state.replace(  # type: ignore[attr-defined]
            output_weights=jnp.array([[2.0], [0.0], [0.0], [0.0]], dtype=jnp.float32)
        )

        result = learner.update(
            state,
            jnp.ones(2, dtype=jnp.float32),
            jnp.array([0.0, jnp.nan, jnp.nan, jnp.nan], dtype=jnp.float32),
        )

        assert float(result.state.utilities[0]) == 2.0
        assert float(result.state.task_activity_ema[0]) > 0.0
        assert float(result.state.task_activity_ema[1]) == 0.0

    def test_inverse_frequency_replacement_keeps_rare_oracle_pair(self) -> None:
        mean_learner = FixedBudgetInteractionLearner(
            n_features=2,
            n_tasks=5,
            utility_decay=0.99,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=0,
            generator_mix=(1.0, 0.0, 0.0),
            utility_task_balancing="none",
            task_activity_decay=0.99,
            use_obgd=False,
        )
        protected_learner = FixedBudgetInteractionLearner(
            n_features=2,
            n_tasks=5,
            utility_decay=0.99,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=0,
            generator_mix=(1.0, 0.0, 0.0),
            utility_task_balancing="active_inverse_frequency",
            task_activity_decay=0.99,
            use_obgd=False,
        )
        state = mean_learner.init(feature_dim=4, key=jr.key(25))
        output_weights = jnp.zeros((5, 2), dtype=jnp.float32).at[4, 0].set(1.0)
        state = state.replace(  # type: ignore[attr-defined]
            feature_left=jnp.array([0, 2], dtype=jnp.int32),
            feature_right=jnp.array([1, 3], dtype=jnp.int32),
            output_weights=output_weights,
            utilities=jnp.array([0.0, 0.5], dtype=jnp.float32),
            ages=jnp.array([10, 10], dtype=jnp.int32),
        )
        protected_state = protected_learner.init(feature_dim=4, key=jr.key(25))
        protected_state = protected_state.replace(  # type: ignore[attr-defined]
            feature_left=state.feature_left,
            feature_right=state.feature_right,
            output_weights=state.output_weights,
            utilities=state.utilities,
            ages=state.ages,
        )
        observation = jnp.array([1.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
        targets = jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan, 0.0])

        mean_result = mean_learner.update(state, observation, targets)
        protected_result = protected_learner.update(
            protected_state,
            observation,
            targets,
        )

        assert int(mean_result.replaced_slot) == 0
        assert int(protected_result.replaced_slot) == 1

    def test_future_utility_mix_credits_new_candidate_weights(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            step_size_output=0.5,
            utility_decay=0.0,
            replacement_interval=0,
            candidate_count=1,
            future_utility_mix=1.0,
            use_obgd=False,
        )
        state = learner.init(feature_dim=2, key=jr.key(20))

        result = learner.update(
            state,
            jnp.ones(2, dtype=jnp.float32),
            jnp.array([2.0], dtype=jnp.float32),
        )

        assert float(result.state.utilities[0]) == 1.5
        assert float(result.state.candidate_utilities[0]) == 1.5

    def test_interaction_config_roundtrip_keeps_utility_knobs(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=3,
            n_tasks=2,
            utility_aggregation="topk",
            utility_top_k=2,
            utility_task_balancing="active_inverse_frequency",
            task_activity_decay=0.9,
            future_utility_mix=0.25,
        )

        restored = FixedBudgetInteractionLearner.from_config(learner.to_config())

        assert restored.to_config()["utility_aggregation"] == "topk"
        assert restored.to_config()["utility_top_k"] == 2
        assert restored.to_config()["utility_task_balancing"] == "active_inverse_frequency"
        assert restored.to_config()["task_activity_decay"] == 0.9
        assert restored.to_config()["future_utility_mix"] == 0.25

    def test_utility_retention_slows_off_context_decay(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            utility_decay=0.0,
            utility_retention_decay=0.9,
            replacement_interval=0,
        )
        state = learner.init(feature_dim=2, key=jr.key(17))
        state = state.replace(utilities=jnp.array([1.0], dtype=jnp.float32))  # type: ignore[attr-defined]

        result = learner.update(
            state,
            jnp.ones(2, dtype=jnp.float32),
            jnp.array([0.0], dtype=jnp.float32),
        )

        assert abs(float(result.state.utilities[0]) - 0.9) < 1e-6

    def test_active_retention_does_not_make_stale_candidates_immortal(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            utility_decay=0.5,
            utility_retention_decay=0.9,
            replacement_interval=0,
            candidate_count=1,
            use_obgd=False,
        )
        state = learner.init(feature_dim=2, key=jr.key(31))
        state = state.replace(  # type: ignore[attr-defined]
            utilities=jnp.array([1.0], dtype=jnp.float32),
            candidate_utilities=jnp.array([1.0], dtype=jnp.float32),
        )

        result = learner.update(
            state,
            jnp.zeros(2, dtype=jnp.float32),
            jnp.array([0.0], dtype=jnp.float32),
        )

        assert abs(float(result.state.utilities[0]) - 0.9) < 1e-6
        assert abs(float(result.state.candidate_utilities[0]) - 0.5) < 1e-6

    def test_inactive_pair_descriptor_constructs_exact_zero(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=2,
            n_tasks=1,
            replacement_interval=0,
        )
        state = learner.init(feature_dim=2, key=jr.key(41)).replace(
            feature_left=jnp.array([-1, 0], dtype=jnp.int32),
            feature_right=jnp.array([-1, 1], dtype=jnp.int32),
        )

        features = learner.constructed_features(
            state,
            jnp.array([2.0, 3.0], dtype=jnp.float32),
        )

        chex.assert_trees_all_equal(
            features,
            jnp.array([0.0, 6.0], dtype=jnp.float32),
        )

    def test_evidence_lease_refreshes_then_expires_retention_floor(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            utility_decay=0.0,
            utility_retention_decay=0.9,
            utility_retention_grace_steps=2,
            utility_evidence_threshold=0.5,
            replacement_interval=0,
            use_obgd=False,
        )
        state = learner.init(feature_dim=2, key=jr.key(42)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            output_weights=jnp.array([[1.0]], dtype=jnp.float32),
            utilities=jnp.array([1.0], dtype=jnp.float32),
            evidence_idle_steps=jnp.array([2], dtype=jnp.int32),
        )
        refreshed = learner.update(
            state,
            jnp.ones(2, dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
        )
        assert bool(refreshed.evidence_refreshed[0])
        assert int(refreshed.state.evidence_idle_steps[0]) == 0

        stale = refreshed.state.replace(
            output_weights=jnp.zeros((1, 1), dtype=jnp.float32),
            utilities=jnp.ones((1,), dtype=jnp.float32),
        )
        first = learner.update(
            stale,
            jnp.ones(2, dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
        )
        second = learner.update(
            first.state,
            jnp.ones(2, dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
        )
        expired = learner.update(
            second.state,
            jnp.ones(2, dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
        )

        assert int(first.state.evidence_idle_steps[0]) == 1
        assert int(second.state.evidence_idle_steps[0]) == 2
        assert int(expired.state.evidence_idle_steps[0]) == 3
        assert float(first.state.utilities[0]) == pytest.approx(0.9)
        assert float(second.state.utilities[0]) == pytest.approx(0.81)
        assert float(expired.state.utilities[0]) == pytest.approx(0.0)

    def test_disabled_confirmed_memory_is_update_compatible_and_shape_matched(self) -> None:
        default = FixedBudgetInteractionLearner(
            n_features=2,
            n_tasks=1,
            candidate_count=1,
            replacement_interval=0,
            utility_retention_grace_steps=3,
            utility_evidence_threshold=0.1,
            use_obgd=False,
        )
        explicit_disabled = FixedBudgetInteractionLearner(
            n_features=2,
            n_tasks=1,
            candidate_count=1,
            replacement_interval=0,
            utility_retention_grace_steps=3,
            utility_evidence_threshold=0.1,
            evidence_gated_active_output_memory=False,
            utility_evidence_confirmation_steps=0,
            use_obgd=False,
        )
        state = default.init(feature_dim=3, key=jr.key(52))
        observation = jnp.array([0.5, -1.0, 2.0], dtype=jnp.float32)
        target = jnp.array([0.75], dtype=jnp.float32)

        default_result = default.update(state, observation, target)
        disabled_result = explicit_disabled.update(state, observation, target)

        chex.assert_trees_all_equal(default_result, disabled_result)
        chex.assert_trees_all_equal(
            disabled_result.state.utility_evidence_streak,
            jnp.zeros((2,), dtype=jnp.int32),
        )
        chex.assert_trees_all_equal(
            disabled_result.state.active_output_memory_committed,
            jnp.zeros((2,), dtype=jnp.bool_),
        )

    def test_confirmed_memory_bootstraps_zero_head_then_protects_committed_head(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            step_size_output=0.5,
            utility_decay=0.0,
            utility_retention_decay=0.9,
            utility_retention_grace_steps=4,
            utility_evidence_threshold=0.1,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            replacement_interval=0,
            use_obgd=False,
        )
        state = learner.init(feature_dim=2, key=jr.key(53)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            evidence_idle_steps=jnp.array([3], dtype=jnp.int32),
        )
        observation = jnp.ones((2,), dtype=jnp.float32)
        first = learner.update(
            state,
            observation,
            jnp.array([1.0], dtype=jnp.float32),
        )
        second = learner.update(
            first.state,
            observation,
            jnp.array([2.0], dtype=jnp.float32),
        )
        confirmed = learner.update(
            second.state,
            observation,
            jnp.array([3.0], dtype=jnp.float32),
        )

        assert float(first.state.output_weights[0, 0]) == pytest.approx(0.5)
        assert not bool(first.evidence_refreshed[0])
        assert not bool(first.state.active_output_memory_committed[0])
        assert float(second.state.output_weights[0, 0]) == pytest.approx(1.0)
        assert bool(second.evidence_refreshed[0])
        assert not bool(second.retention_evidence_refreshed[0])
        assert int(second.state.utility_evidence_streak[0]) == 1
        assert int(second.state.evidence_idle_steps[0]) == 5
        assert float(confirmed.state.output_weights[0, 0]) == pytest.approx(1.5)
        assert bool(confirmed.retention_evidence_refreshed[0])
        assert bool(confirmed.state.active_output_memory_committed[0])
        assert int(confirmed.state.utility_evidence_streak[0]) == 2
        assert int(confirmed.state.evidence_idle_steps[0]) == 0

        unconfirmed = learner.update(
            confirmed.state.replace(
                output_weights=jnp.array([[0.05]], dtype=jnp.float32),
                utility_evidence_streak=jnp.array([1], dtype=jnp.int32),
                evidence_idle_steps=jnp.array([2], dtype=jnp.int32),
            ),
            observation,
            jnp.array([10.0], dtype=jnp.float32),
        )

        assert not bool(unconfirmed.evidence_refreshed[0])
        assert not bool(unconfirmed.retention_evidence_refreshed[0])
        assert float(unconfirmed.state.output_weights[0, 0]) == pytest.approx(0.05)
        assert int(unconfirmed.state.utility_evidence_streak[0]) == 0
        assert int(unconfirmed.state.evidence_idle_steps[0]) == 3

    def test_confirmed_memory_streak_saturates_and_replacement_resets_identity(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            utility_decay=0.0,
            utility_retention_grace_steps=4,
            utility_evidence_threshold=0.1,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            replacement_interval=2,
            min_feature_age=0,
            use_obgd=False,
        )
        state = learner.init(feature_dim=3, key=jr.key(54)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            output_weights=jnp.array([[1.0]], dtype=jnp.float32),
            utility_evidence_streak=jnp.array([2**31 - 2], dtype=jnp.int32),
            active_output_memory_committed=jnp.array([True], dtype=jnp.bool_),
            ages=jnp.array([4], dtype=jnp.int32),
        )

        saturated = learner.update(
            state,
            jnp.ones((3,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        )
        result = learner.update(
            saturated.state,
            jnp.ones((3,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        )

        assert int(saturated.replaced_slot) == -1
        assert int(saturated.state.utility_evidence_streak[0]) == 2**31 - 1
        assert bool(saturated.state.active_output_memory_committed[0])
        assert bool(result.retention_evidence_refreshed[0])
        assert int(result.replaced_slot) == 0
        assert int(result.state.utility_evidence_streak[0]) == 0
        assert not bool(result.state.active_output_memory_committed[0])

    def test_robust_confirmed_memory_bootstraps_then_freezes_unconfirmed_write(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            step_size_output=0.5,
            utility_decay=0.0,
            utility_retention_decay=0.9,
            utility_retention_grace_steps=4,
            utility_evidence_threshold=0.1,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            replacement_interval=0,
            scale_robust=True,
        )
        state = learner.init(feature_dim=2, key=jr.key(60)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
        )
        observation = jnp.ones((2,), dtype=jnp.float32)
        target = jnp.ones((1,), dtype=jnp.float32)

        bootstrap = learner.update(state, observation, target)
        first_evidence = learner.update(bootstrap.state, observation, target)
        confirmed = learner.update(first_evidence.state, observation, target)

        assert float(bootstrap.state.output_weights[0, 0]) > 0.0
        assert not bool(bootstrap.evidence_refreshed[0])
        assert bool(first_evidence.evidence_refreshed[0])
        assert not bool(first_evidence.retention_evidence_refreshed[0])
        assert int(first_evidence.state.utility_evidence_streak[0]) == 1
        assert bool(confirmed.retention_evidence_refreshed[0])
        assert bool(confirmed.state.active_output_memory_committed[0])
        committed_weight = float(confirmed.state.output_weights[0, 0])

        wrong_sign = learner.update(
            confirmed.state,
            observation,
            jnp.array([-10.0], dtype=jnp.float32),
        )

        assert not bool(wrong_sign.evidence_refreshed[0])
        assert not bool(wrong_sign.retention_evidence_refreshed[0])
        assert float(wrong_sign.state.output_weights[0, 0]) == pytest.approx(
            committed_weight
        )
        assert int(wrong_sign.state.utility_evidence_streak[0]) == 0
        assert int(wrong_sign.state.evidence_idle_steps[0]) == 1

    def test_candidate_probe_and_promotion_bypass_old_active_write_gate(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            step_size_output=0.5,
            utility_decay=0.995,
            utility_retention_grace_steps=4,
            utility_evidence_threshold=0.9,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=3,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=1,
            candidate_min_age=0,
            promotion_margin=1.0,
            refresh_candidates=False,
            refresh_promoted_candidate=False,
            use_obgd=False,
        )
        state = learner.init(feature_dim=3, key=jr.key(55)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            output_weights=jnp.zeros((1, 1), dtype=jnp.float32),
            utilities=jnp.zeros((1,), dtype=jnp.float32),
            utility_evidence_streak=jnp.array([2], dtype=jnp.int32),
            active_output_memory_committed=jnp.array([True], dtype=jnp.bool_),
            ages=jnp.array([4], dtype=jnp.int32),
            candidate_left=jnp.array([1], dtype=jnp.int32),
            candidate_right=jnp.array([2], dtype=jnp.int32),
            candidate_output_weights=jnp.zeros((1, 1), dtype=jnp.float32),
            candidate_utilities=jnp.array([10.0], dtype=jnp.float32),
            candidate_ages=jnp.array([4], dtype=jnp.int32),
        )

        result = learner.update(
            state,
            jnp.ones((3,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        )

        assert not bool(result.retention_evidence_refreshed[0])
        assert int(result.promoted_candidate) == 0
        assert int(result.replaced_slot) == 0
        assert float(result.state.output_weights[0, 0]) == pytest.approx(0.5)
        assert int(result.state.utility_evidence_streak[0]) == 0
        assert not bool(result.state.active_output_memory_committed[0])
        # The matched lifecycle-freeze control commits this exact learned
        # snapshot: ordinary active/candidate learning and recurrence advance,
        # but the promotion transaction has not changed identities or reset
        # either bank yet.
        pre_curation = result.pre_curation_state
        chex.assert_trees_all_equal(pre_curation.feature_left, state.feature_left)
        chex.assert_trees_all_equal(pre_curation.feature_right, state.feature_right)
        chex.assert_trees_all_equal(pre_curation.candidate_left, state.candidate_left)
        chex.assert_trees_all_equal(pre_curation.candidate_right, state.candidate_right)
        assert float(pre_curation.output_weights[0, 0]) == 0.0
        assert float(pre_curation.candidate_output_weights[0, 0]) == pytest.approx(0.5)
        assert int(pre_curation.utility_evidence_streak[0]) == 0
        assert bool(pre_curation.active_output_memory_committed[0])
        assert int(pre_curation.ages[0]) == 5
        assert int(pre_curation.candidate_ages[0]) == 5
        assert int(pre_curation.step_count) == int(state.step_count) + 1
        assert not bool(jnp.array_equal(pre_curation.key, state.key))

    def test_conditional_probe_signal_tracks_the_deployed_durable_bank(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=2,
            n_tasks=1,
            step_size_output=0.1,
            utility_decay=0.0,
            utility_retention_grace_steps=8,
            utility_evidence_threshold=0.01,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            independent_relevance_probe=True,
            replacement_interval=0,
            scale_robust=True,
        )
        base = learner.init(feature_dim=3, key=jr.key(61)).replace(
            feature_left=jnp.array([0, 0], dtype=jnp.int32),
            feature_right=jnp.array([1, 2], dtype=jnp.int32),
            output_weights=jnp.array([[0.0, 0.0]], dtype=jnp.float32),
            relevance_probe_weights=jnp.array([[0.5, 0.5]], dtype=jnp.float32),
            relevance_probe_biases=jnp.array([0.2], dtype=jnp.float32),
            active_output_memory_committed=jnp.array([True, True]),
            feature_second_moments=jnp.ones((2,), dtype=jnp.float32),
            target_second_moments=jnp.ones((1,), dtype=jnp.float32),
        )
        perturbed = base.replace(
            output_weights=jnp.array([[10.0, -100.0]], dtype=jnp.float32)
        )
        observation = jnp.ones((3,), dtype=jnp.float32)
        target = jnp.ones((1,), dtype=jnp.float32)
        external = jnp.ones((2,), dtype=jnp.bool_)
        reference = learner.update(base, observation, target, external)
        changed = learner.update(perturbed, observation, target, external)

        assert float(reference.predictions[0]) != float(changed.predictions[0])
        chex.assert_trees_all_close(
            reference.relevance_probe_errors,
            jnp.asarray(((1.0, 1.0),), dtype=jnp.float32),
        )
        chex.assert_trees_all_close(
            changed.relevance_probe_errors,
            jnp.asarray(((101.0, -9.0),), dtype=jnp.float32),
        )
        assert not bool(
            jnp.all(reference.relevance_probe_scores == changed.relevance_probe_scores)
        )
        assert not bool(
            jnp.all(
                reference.state.relevance_probe_weights
                == changed.state.relevance_probe_weights
            )
        )

    def test_conditional_probe_heads_remain_isolated_from_each_other(
        self,
    ) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=3,
            n_tasks=1,
            step_size_output=0.5,
            utility_decay=0.0,
            utility_retention_grace_steps=8,
            utility_evidence_threshold=0.01,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            independent_relevance_probe=True,
            replacement_interval=0,
            scale_robust=True,
        )
        base = learner.init(feature_dim=3, key=jr.key(611)).replace(
            feature_left=jnp.array([0, 0, 1], dtype=jnp.int32),
            feature_right=jnp.array([1, 2, 2], dtype=jnp.int32),
            output_weights=jnp.array([[0.0, 0.0, -0.0]], dtype=jnp.float32),
            relevance_probe_weights=jnp.array([[0.5, 0.4, 0.3]], dtype=jnp.float32),
            relevance_probe_biases=jnp.array([0.2], dtype=jnp.float32),
            output_biases=jnp.array([0.3], dtype=jnp.float32),
            active_output_memory_committed=jnp.array([True, True, False]),
            feature_second_moments=jnp.ones((3,), dtype=jnp.float32),
            target_second_moments=jnp.ones((1,), dtype=jnp.float32),
        )
        perturbed_bank = base.replace(
            output_weights=jnp.array([[0.2, -0.4, 0.0]], dtype=jnp.float32)
        )
        perturbed_probe = base.replace(
            relevance_probe_weights=base.relevance_probe_weights.at[0, 0].set(10.5)
        )
        observation = jnp.ones((3,), dtype=jnp.float32)
        target = jnp.ones((1,), dtype=jnp.float32)
        external = jnp.ones((3,), dtype=jnp.bool_)
        reference = learner.update(base, observation, target, external)
        bank_changed = learner.update(perturbed_bank, observation, target, external)
        probe_changed = learner.update(perturbed_probe, observation, target, external)

        assert not bool(
            jnp.all(reference.relevance_probe_errors == bank_changed.relevance_probe_errors)
        )
        chex.assert_trees_all_equal(
            reference.relevance_probe_errors,
            probe_changed.relevance_probe_errors,
        )
        chex.assert_trees_all_equal(
            reference.relevance_probe_scores[1:],
            probe_changed.relevance_probe_scores[1:],
        )
        chex.assert_trees_all_equal(
            reference.state.relevance_probe_weights[:, 1:],
            probe_changed.state.relevance_probe_weights[:, 1:],
        )

    def test_conditional_probe_candidate_seed_respects_deployed_residual(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            step_size_output=0.5,
            utility_decay=0.995,
            utility_retention_grace_steps=8,
            utility_evidence_threshold=0.9,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            independent_relevance_probe=True,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=1,
            candidate_min_age=0,
            promotion_margin=1.0,
            refresh_candidates=False,
            refresh_promoted_candidate=False,
            scale_robust=True,
        )
        base = learner.init(feature_dim=3, key=jr.key(613)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            output_weights=jnp.array([[0.0]], dtype=jnp.float32),
            relevance_probe_weights=jnp.array([[0.5]], dtype=jnp.float32),
            active_output_memory_committed=jnp.array([True]),
            ages=jnp.array([4], dtype=jnp.int32),
            candidate_left=jnp.array([1], dtype=jnp.int32),
            candidate_right=jnp.array([2], dtype=jnp.int32),
            candidate_output_weights=jnp.array([[0.5]], dtype=jnp.float32),
            candidate_utilities=jnp.array([10.0], dtype=jnp.float32),
            candidate_ages=jnp.array([4], dtype=jnp.int32),
            feature_second_moments=jnp.ones((1,), dtype=jnp.float32),
            candidate_second_moments=jnp.ones((1,), dtype=jnp.float32),
            target_second_moments=jnp.ones((1,), dtype=jnp.float32),
        )
        perturbed = base.replace(
            output_weights=jnp.array([[100.0]], dtype=jnp.float32)
        )
        observation = jnp.ones((3,), dtype=jnp.float32)
        target = jnp.ones((1,), dtype=jnp.float32)
        reference = learner.update(base, observation, target)
        changed = learner.update(perturbed, observation, target)

        assert float(reference.predictions[0]) != float(changed.predictions[0])
        assert int(reference.promoted_candidate) == 0
        assert int(changed.promoted_candidate) == 0
        chex.assert_trees_all_equal(
            reference.state.feature_left,
            changed.state.feature_left,
        )
        chex.assert_trees_all_equal(
            reference.state.feature_right,
            changed.state.feature_right,
        )
        assert not bool(
            jnp.all(
                reference.state.relevance_probe_weights
                == changed.state.relevance_probe_weights
            )
        )
        assert not bool(
            jnp.all(
                reference.candidate_promotion_signal
                == changed.candidate_promotion_signal
            )
        )
        chex.assert_trees_all_equal(
            reference.state.output_weights,
            jnp.zeros((1, 1), dtype=jnp.float32),
        )
        chex.assert_trees_all_equal(
            changed.state.output_weights,
            jnp.zeros((1, 1), dtype=jnp.float32),
        )

    def test_candidate_promotion_requires_consecutive_marginal_evidence(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            step_size_output=0.0,
            utility_decay=0.995,
            utility_retention_grace_steps=8,
            utility_evidence_threshold=0.1,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            independent_relevance_probe=True,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=1,
            candidate_min_age=0,
            promotion_margin=1.0,
            candidate_promotion_confirmation_steps=3,
            refresh_candidates=False,
            refresh_promoted_candidate=False,
            scale_robust=True,
        )
        state = learner.init(feature_dim=3, key=jr.key(614)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            output_weights=jnp.zeros((1, 1), dtype=jnp.float32),
            relevance_probe_weights=jnp.zeros((1, 1), dtype=jnp.float32),
            active_output_memory_committed=jnp.array([True]),
            ages=jnp.array([4], dtype=jnp.int32),
            candidate_left=jnp.array([1], dtype=jnp.int32),
            candidate_right=jnp.array([2], dtype=jnp.int32),
            candidate_output_weights=jnp.array([[0.5]], dtype=jnp.float32),
            candidate_utilities=jnp.array([10.0], dtype=jnp.float32),
            candidate_ages=jnp.array([4], dtype=jnp.int32),
            feature_second_moments=jnp.ones((1,), dtype=jnp.float32),
            candidate_second_moments=jnp.ones((1,), dtype=jnp.float32),
            target_second_moments=jnp.ones((1,), dtype=jnp.float32),
        )
        observation = jnp.ones((3,), dtype=jnp.float32)
        jitted_update = jax.jit(learner.update)
        target_sequence = (1.0, 1.0, 0.0, 1.0, 1.0, 1.0)
        expected_updated_streak = (1, 2, 0, 1, 2, 3)
        expected_post_streak = (1, 2, 0, 1, 2, 0)
        results = []

        for target_value, expected_updated, expected_post in zip(
            target_sequence,
            expected_updated_streak,
            expected_post_streak,
            strict=True,
        ):
            result = jitted_update(
                state,
                observation,
                jnp.array([target_value], dtype=jnp.float32),
            )
            results.append(result)
            assert int(result.candidate_promotion_evidence_streak_updated[0]) == (
                expected_updated
            )
            assert int(result.state.candidate_promotion_evidence_streak[0]) == (
                expected_post
            )
            state = result.state

        assert bool(results[0].candidate_promotion_raw_evidence[0])
        assert bool(results[1].candidate_promotion_raw_evidence[0])
        assert not bool(results[2].candidate_promotion_raw_evidence[0])
        assert int(results[1].promoted_candidate) == -1
        assert not bool(results[1].candidate_promotion_confirmed[0])
        assert int(results[4].promoted_candidate) == -1
        assert not bool(results[4].candidate_promotion_confirmed[0])
        assert int(results[5].promoted_candidate) == 0
        assert bool(results[5].candidate_promotion_confirmed[0])
        assert float(results[5].candidate_promotion_signal[0]) > 0.1
        assert int(
            np.asarray(
                results[5].state.candidate_output_weights[0, 0]
            ).view(np.uint32)
        ) == 0

    def test_reacquisition_confirmation_does_not_delay_first_acquisition(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            step_size_output=0.0,
            utility_decay=0.995,
            utility_retention_grace_steps=100,
            utility_evidence_threshold=0.1,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            independent_relevance_probe=True,
            retire_stale_features=True,
            candidate_promotion_floor=0.1,
            candidate_reacquisition_confirmation_steps=3,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=1,
            candidate_min_age=0,
            promotion_margin=1.0,
            refresh_candidates=False,
            refresh_promoted_candidate=False,
            scale_robust=True,
        )
        state = learner.init(feature_dim=3, key=jr.key(618)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            ages=jnp.array([4], dtype=jnp.int32),
            candidate_left=jnp.array([1], dtype=jnp.int32),
            candidate_right=jnp.array([2], dtype=jnp.int32),
            candidate_output_weights=jnp.array([[0.5]], dtype=jnp.float32),
            candidate_utilities=jnp.array([10.0], dtype=jnp.float32),
            candidate_ages=jnp.array([4], dtype=jnp.int32),
            feature_second_moments=jnp.ones((1,), dtype=jnp.float32),
            candidate_second_moments=jnp.ones((1,), dtype=jnp.float32),
            target_second_moments=jnp.ones((1,), dtype=jnp.float32),
        )
        result = jax.jit(learner.update)(
            state,
            jnp.ones((3,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        )

        assert bool(result.candidate_promotion_raw_evidence[0])
        assert bool(result.candidate_promotion_confirmed[0])
        assert not bool(result.candidate_reacquisition_required_pre[0])
        assert not bool(result.candidate_reacquisition_required_post[0])
        assert not bool(result.candidate_reacquisition_confirmed[0])
        assert int(result.candidate_promotion_evidence_streak_updated[0]) == 0
        assert int(result.promoted_candidate) == 0

    def test_unrelated_lower_utility_candidate_can_win_during_reacquisition(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            step_size_output=0.0,
            utility_decay=0.995,
            utility_retention_grace_steps=0,
            utility_evidence_threshold=0.1,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            independent_relevance_probe=True,
            retire_stale_features=True,
            candidate_promotion_floor=0.1,
            candidate_reacquisition_confirmation_steps=3,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=2,
            candidate_min_age=0,
            promotion_margin=1.0,
            refresh_candidates=False,
            refresh_promoted_candidate=False,
            scale_robust=True,
        )
        state = learner.init(feature_dim=3, key=jr.key(620)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            active_output_memory_committed=jnp.array([True]),
            evidence_idle_steps=jnp.array([1], dtype=jnp.int32),
            ages=jnp.array([4], dtype=jnp.int32),
            candidate_left=jnp.array([0, 1], dtype=jnp.int32),
            candidate_right=jnp.array([1, 2], dtype=jnp.int32),
            candidate_output_weights=jnp.array([[0.5, 0.5]], dtype=jnp.float32),
            candidate_utilities=jnp.array([100.0, 10.0], dtype=jnp.float32),
            candidate_ages=jnp.array([4, 4], dtype=jnp.int32),
            feature_second_moments=jnp.ones((1,), dtype=jnp.float32),
            candidate_second_moments=jnp.ones((2,), dtype=jnp.float32),
            target_second_moments=jnp.ones((1,), dtype=jnp.float32),
        )
        observation = jnp.ones((3,), dtype=jnp.float32)
        jitted_update = jax.jit(learner.update)
        retired = jitted_update(
            state,
            observation,
            jnp.zeros((1,), dtype=jnp.float32),
        )
        assert int(retired.retired_slot) == 0
        chex.assert_trees_all_equal(
            retired.state.candidate_reacquisition_required,
            jnp.array([True, False]),
        )

        reacquiring = retired.state.replace(
            candidate_output_weights=jnp.array([[0.5, 0.5]], dtype=jnp.float32),
            candidate_utilities=jnp.array([100.0, 10.0], dtype=jnp.float32),
            candidate_ages=jnp.array([4, 4], dtype=jnp.int32),
            candidate_second_moments=jnp.ones((2,), dtype=jnp.float32),
        )
        result = jitted_update(
            reacquiring,
            observation,
            jnp.ones((1,), dtype=jnp.float32),
        )

        chex.assert_trees_all_equal(
            result.candidate_promotion_confirmed,
            jnp.array([False, True]),
        )
        chex.assert_trees_all_equal(
            result.candidate_reacquisition_required_pre,
            jnp.array([True, False]),
        )
        assert int(result.promoted_candidate) == 1
        assert bool(result.state.candidate_reacquisition_required[0])
        assert not bool(result.state.candidate_reacquisition_required[1])

    def test_refresh_and_invalid_descriptor_clear_reacquisition_state(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            step_size_output=0.0,
            utility_decay=0.995,
            utility_retention_grace_steps=100,
            utility_evidence_threshold=0.1,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            independent_relevance_probe=True,
            retire_stale_features=True,
            candidate_promotion_floor=0.1,
            candidate_reacquisition_confirmation_steps=3,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=1,
            candidate_min_age=0,
            promotion_margin=1.0,
            refresh_candidates=True,
            refresh_promoted_candidate=False,
            scale_robust=True,
        )
        base = learner.init(feature_dim=3, key=jr.key(621)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            ages=jnp.array([4], dtype=jnp.int32),
            candidate_left=jnp.array([1], dtype=jnp.int32),
            candidate_right=jnp.array([2], dtype=jnp.int32),
            candidate_output_weights=jnp.array([[0.5]], dtype=jnp.float32),
            candidate_utilities=jnp.array([10.0], dtype=jnp.float32),
            candidate_ages=jnp.array([4], dtype=jnp.int32),
            candidate_promotion_evidence_streak=jnp.array([1], dtype=jnp.int32),
            candidate_reacquisition_required=jnp.array([True]),
            feature_second_moments=jnp.ones((1,), dtype=jnp.float32),
            candidate_second_moments=jnp.ones((1,), dtype=jnp.float32),
            target_second_moments=jnp.ones((1,), dtype=jnp.float32),
        )
        observation = jnp.ones((3,), dtype=jnp.float32)
        target = jnp.ones((1,), dtype=jnp.float32)
        jitted_update = jax.jit(learner.update)
        refreshed = jitted_update(base, observation, target)

        assert bool(refreshed.candidate_reacquisition_required_pre[0])
        assert int(refreshed.candidate_promotion_evidence_streak_updated[0]) == 2
        assert int(refreshed.promoted_candidate) == -1
        assert int(refreshed.refreshed_candidate) == 0
        assert not bool(refreshed.state.candidate_reacquisition_required[0])
        assert int(refreshed.state.candidate_promotion_evidence_streak[0]) == 0
        pre_curation = refreshed.pre_curation_state
        chex.assert_trees_all_equal(pre_curation.feature_left, base.feature_left)
        chex.assert_trees_all_equal(pre_curation.feature_right, base.feature_right)
        chex.assert_trees_all_equal(pre_curation.candidate_left, base.candidate_left)
        chex.assert_trees_all_equal(pre_curation.candidate_right, base.candidate_right)
        chex.assert_trees_all_equal(
            pre_curation.candidate_parent_a,
            base.candidate_parent_a,
        )
        chex.assert_trees_all_equal(
            pre_curation.candidate_parent_b,
            base.candidate_parent_b,
        )
        chex.assert_trees_all_equal(
            pre_curation.candidate_generator,
            base.candidate_generator,
        )
        assert float(pre_curation.candidate_output_weights[0, 0]) == 0.5
        assert int(pre_curation.candidate_promotion_evidence_streak[0]) == 2
        assert bool(pre_curation.candidate_reacquisition_required[0])
        assert int(pre_curation.candidate_ages[0]) == 5

        invalid = jitted_update(
            base.replace(
                candidate_left=jnp.array([-1], dtype=jnp.int32),
                candidate_right=jnp.array([-1], dtype=jnp.int32),
                candidate_promotion_evidence_streak=jnp.array([2], dtype=jnp.int32),
            ),
            observation,
            target,
        )
        assert bool(invalid.candidate_reacquisition_required_pre[0])
        assert not bool(invalid.candidate_reacquisition_required_post[0])
        assert int(invalid.state.candidate_promotion_evidence_streak[0]) == 0

    def test_confirmed_lower_utility_candidate_is_ranked_before_unconfirmed(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            step_size_output=0.0,
            utility_decay=0.995,
            utility_retention_grace_steps=8,
            utility_evidence_threshold=0.1,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            independent_relevance_probe=True,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=2,
            candidate_min_age=0,
            promotion_margin=1.0,
            candidate_promotion_confirmation_steps=3,
            refresh_candidates=False,
            refresh_promoted_candidate=False,
            scale_robust=True,
        )
        state = learner.init(feature_dim=3, key=jr.key(617)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            ages=jnp.array([4], dtype=jnp.int32),
            candidate_left=jnp.array([0, 1], dtype=jnp.int32),
            candidate_right=jnp.array([2, 2], dtype=jnp.int32),
            candidate_output_weights=jnp.array([[0.5, 0.5]], dtype=jnp.float32),
            candidate_utilities=jnp.array([100.0, 10.0], dtype=jnp.float32),
            candidate_ages=jnp.array([4, 4], dtype=jnp.int32),
            candidate_promotion_evidence_streak=jnp.array([0, 2], dtype=jnp.int32),
            feature_second_moments=jnp.ones((1,), dtype=jnp.float32),
            candidate_second_moments=jnp.ones((2,), dtype=jnp.float32),
            target_second_moments=jnp.ones((1,), dtype=jnp.float32),
        )
        result = learner.update(
            state,
            jnp.ones((3,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        )

        chex.assert_trees_all_equal(
            result.candidate_promotion_confirmed,
            jnp.array([False, True]),
        )
        chex.assert_trees_all_equal(
            result.candidate_promotion_evidence_streak_updated,
            jnp.array([1, 3], dtype=jnp.int32),
        )
        assert int(result.promoted_candidate) == 1
        chex.assert_trees_all_equal(
            result.state.candidate_promotion_evidence_streak,
            jnp.array([1, 0], dtype=jnp.int32),
        )

    @pytest.mark.parametrize(
        "value",
        [True, 0, -1, 2**31 - 1, 1.5],
    )
    def test_candidate_promotion_confirmation_rejects_unsafe_values(
        self,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match="candidate_promotion_confirmation_steps"):
            FixedBudgetInteractionLearner(
                n_features=1,
                n_tasks=1,
                candidate_promotion_confirmation_steps=value,  # type: ignore[arg-type]
            )

        configured = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            candidate_promotion_confirmation_steps=3,
        )
        restored = FixedBudgetInteractionLearner.from_config(configured.to_config())
        assert restored.to_config()["candidate_promotion_confirmation_steps"] == 3

    @pytest.mark.parametrize(
        "value",
        [True, 0, -1, 2**31 - 1, 1.5],
    )
    def test_candidate_reacquisition_confirmation_rejects_unsafe_values(
        self,
        value: object,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="candidate_reacquisition_confirmation_steps",
        ):
            FixedBudgetInteractionLearner(
                n_features=1,
                n_tasks=1,
                candidate_reacquisition_confirmation_steps=value,  # type: ignore[arg-type]
            )

    def test_candidate_reacquisition_confirmation_requires_probe_not_retirement(
        self,
    ) -> None:
        common = {
            "n_features": 1,
            "n_tasks": 1,
            "utility_retention_grace_steps": 1,
            "utility_evidence_threshold": 0.1,
            "evidence_gated_active_output_memory": True,
            "replacement_interval": 1,
            "candidate_promotion_floor": 0.1,
        }
        with pytest.raises(ValueError, match="requires independent_relevance_probe"):
            FixedBudgetInteractionLearner(
                **common,
                retire_stale_features=True,
                candidate_reacquisition_confirmation_steps=2,
            )
        no_retirement = FixedBudgetInteractionLearner(
            **common,
            independent_relevance_probe=True,
            candidate_reacquisition_confirmation_steps=2,
        )
        no_retirement_state = no_retirement.init(feature_dim=3, key=jr.key(618))
        assert no_retirement.to_config()["retire_stale_features"] is False
        chex.assert_trees_all_equal(
            no_retirement_state.candidate_reacquisition_required,
            jnp.zeros_like(no_retirement_state.candidate_reacquisition_required),
        )

        configured = FixedBudgetInteractionLearner(
            **common,
            independent_relevance_probe=True,
            retire_stale_features=True,
            candidate_reacquisition_confirmation_steps=3,
        )
        restored = FixedBudgetInteractionLearner.from_config(configured.to_config())
        assert restored.to_config()["candidate_reacquisition_confirmation_steps"] == 3

    def test_candidate_compatibility_mode_keeps_zero_signal_streak_canonical(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            candidate_count=1,
            replacement_interval=0,
            utility_evidence_threshold=0.0,
            candidate_promotion_confirmation_steps=1,
            candidate_reacquisition_confirmation_steps=1,
        )
        state = learner.init(feature_dim=3, key=jr.key(616)).replace(
            candidate_output_weights=jnp.zeros((1, 1), dtype=jnp.float32),
            candidate_promotion_evidence_streak=jnp.array([7], dtype=jnp.int32),
            candidate_reacquisition_required=jnp.array([True]),
        )
        result = learner.update(
            state,
            jnp.ones((3,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
        )

        assert float(result.candidate_promotion_signal[0]) == 0.0
        assert not bool(result.candidate_promotion_raw_evidence[0])
        assert int(result.candidate_promotion_evidence_streak_updated[0]) == 0
        assert int(result.state.candidate_promotion_evidence_streak[0]) == 0
        assert not bool(result.state.candidate_reacquisition_required[0])
        assert not bool(result.candidate_reacquisition_confirmed[0])
        assert bool(result.candidate_promotion_confirmed[0])

    def test_independent_probe_scores_before_update_and_commits_exact_preupdate_value(
        self,
    ) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            step_size_output=0.5,
            utility_decay=0.0,
            utility_retention_grace_steps=8,
            utility_evidence_threshold=0.1,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=1,
            independent_relevance_probe=True,
            replacement_interval=0,
            scale_robust=True,
        )
        state = learner.init(feature_dim=2, key=jr.key(62)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            output_weights=jnp.array([[-0.0]], dtype=jnp.float32),
            relevance_probe_weights=jnp.array([[0.5]], dtype=jnp.float32),
            feature_second_moments=jnp.ones((1,), dtype=jnp.float32),
            target_second_moments=jnp.ones((1,), dtype=jnp.float32),
        )
        result = learner.update(
            state,
            jnp.ones((2,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.bool_),
        )

        assert bool(result.retention_evidence_refreshed[0])
        assert bool(result.state.active_output_memory_committed[0])
        assert float(result.relevance_probe_scores[0]) == pytest.approx(0.27272728)
        assert float(result.state.output_weights[0, 0]) == 0.5
        assert float(result.state.relevance_probe_weights[0, 0]) == pytest.approx(0.75)
        chex.assert_trees_all_equal(
            result.state.output_weights,
            state.relevance_probe_weights,
        )

        frozen = learner.update(
            result.state,
            jnp.ones((2,), dtype=jnp.float32),
            jnp.array([-1.0], dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.bool_),
        )
        chex.assert_trees_all_equal(
            frozen.state.output_weights,
            result.state.output_weights,
        )

    def test_independent_probe_uncommitted_and_closed_heads_are_zero_contribution(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=2,
            n_tasks=1,
            utility_decay=0.0,
            utility_retention_grace_steps=8,
            utility_evidence_threshold=0.9,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            independent_relevance_probe=True,
            replacement_interval=0,
            scale_robust=True,
        )
        state = learner.init(feature_dim=3, key=jr.key(63)).replace(
            feature_left=jnp.array([0, 0], dtype=jnp.int32),
            feature_right=jnp.array([1, 2], dtype=jnp.int32),
            output_weights=jnp.array([[-0.0, 100.0]], dtype=jnp.float32),
            active_output_memory_committed=jnp.array([False, True]),
        )
        external = jnp.array([True, False])
        prediction = learner.predict(state, jnp.ones((3,), dtype=jnp.float32), external)
        result = learner.update(
            state,
            jnp.ones((3,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            external,
        )

        chex.assert_trees_all_equal(prediction, jnp.zeros((1,), dtype=jnp.float32))
        chex.assert_trees_all_equal(
            result.durable_read_mask,
            jnp.zeros((2,), dtype=jnp.bool_),
        )
        assert int(np.asarray(result.state.output_weights[0, 0]).view(np.uint32)) == 0
        assert float(result.state.output_weights[0, 1]) == 100.0

    def test_independent_probe_promotion_seeds_probe_and_retirement_resets_it(self) -> None:
        promote = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            step_size_output=0.5,
            utility_decay=0.995,
            utility_retention_grace_steps=8,
            utility_evidence_threshold=0.9,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            independent_relevance_probe=True,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=1,
            candidate_min_age=0,
            promotion_margin=1.0,
            refresh_candidates=False,
            refresh_promoted_candidate=False,
            scale_robust=True,
        )
        promoted_state = promote.init(feature_dim=3, key=jr.key(64)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            output_weights=jnp.zeros((1, 1), dtype=jnp.float32),
            relevance_probe_weights=jnp.array([[9.0]], dtype=jnp.float32),
            active_output_memory_committed=jnp.array([True]),
            ages=jnp.array([4], dtype=jnp.int32),
            candidate_left=jnp.array([1], dtype=jnp.int32),
            candidate_right=jnp.array([2], dtype=jnp.int32),
            candidate_output_weights=jnp.ones((1, 1), dtype=jnp.float32),
            candidate_utilities=jnp.array([10.0], dtype=jnp.float32),
            candidate_ages=jnp.array([4], dtype=jnp.int32),
            feature_second_moments=jnp.ones((1,), dtype=jnp.float32),
            candidate_second_moments=jnp.ones((1,), dtype=jnp.float32),
            target_second_moments=jnp.ones((1,), dtype=jnp.float32),
        )
        promoted = promote.update(
            promoted_state,
            jnp.ones((3,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        )

        assert int(promoted.promoted_candidate) == 0
        assert float(promoted.state.relevance_probe_weights[0, 0]) == 1.0
        assert int(np.asarray(promoted.state.output_weights[0, 0]).view(np.uint32)) == 0
        assert not bool(promoted.state.active_output_memory_committed[0])

        retire = FixedBudgetInteractionLearner(
            n_features=1,
            n_tasks=1,
            utility_decay=0.0,
            utility_retention_grace_steps=0,
            utility_evidence_threshold=0.9,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            independent_relevance_probe=True,
            retire_stale_features=True,
            candidate_promotion_floor=1.0,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=1,
            candidate_min_age=0,
            refresh_candidates=False,
            refresh_promoted_candidate=False,
            scale_robust=True,
        )
        retired_state = retire.init(feature_dim=2, key=jr.key(65)).replace(
            feature_left=jnp.array([0], dtype=jnp.int32),
            feature_right=jnp.array([1], dtype=jnp.int32),
            output_weights=jnp.array([[2.0]], dtype=jnp.float32),
            relevance_probe_weights=jnp.array([[-0.0]], dtype=jnp.float32),
            active_output_memory_committed=jnp.array([True]),
            evidence_idle_steps=jnp.array([1], dtype=jnp.int32),
            ages=jnp.array([4], dtype=jnp.int32),
            candidate_left=jnp.array([0], dtype=jnp.int32),
            candidate_right=jnp.array([1], dtype=jnp.int32),
        )
        retired = jax.jit(retire.update)(
            retired_state,
            jnp.ones((2,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.bool_),
        )

        assert int(retired.retired_slot) == 0
        assert int(np.asarray(retired.state.output_weights[0, 0]).view(np.uint32)) == 0
        assert int(
            np.asarray(retired.state.relevance_probe_weights[0, 0]).view(np.uint32)
        ) == 0

    def test_relevance_probe_state_is_always_fixed_shape_and_accounted(self) -> None:
        disabled = FixedBudgetInteractionLearner(
            n_features=3,
            n_tasks=2,
            candidate_count=4,
        )
        enabled = FixedBudgetInteractionLearner(
            n_features=3,
            n_tasks=2,
            candidate_count=4,
            utility_retention_grace_steps=8,
            utility_evidence_threshold=0.1,
            evidence_gated_active_output_memory=True,
            independent_relevance_probe=True,
            scale_robust=True,
        )
        disabled_state = disabled.init(feature_dim=3, key=jr.key(66))
        enabled_state = enabled.init(feature_dim=3, key=jr.key(66))
        disabled_budget = disabled.memory_accounting(disabled_state)
        enabled_budget = enabled.memory_accounting(enabled_state)

        chex.assert_shape(disabled_state.relevance_probe_weights, (2, 3))
        chex.assert_shape(enabled_state.relevance_probe_weights, (2, 3))
        chex.assert_shape(disabled_state.relevance_probe_biases, (2,))
        chex.assert_shape(enabled_state.relevance_probe_biases, (2,))
        chex.assert_shape(
            disabled_state.candidate_promotion_evidence_streak,
            (4,),
        )
        chex.assert_shape(
            enabled_state.candidate_promotion_evidence_streak,
            (4,),
        )
        chex.assert_shape(disabled_state.candidate_reacquisition_required, (4,))
        chex.assert_shape(enabled_state.candidate_reacquisition_required, (4,))
        assert disabled_state.candidate_reacquisition_required.dtype == jnp.bool_
        assert enabled_state.candidate_reacquisition_required.dtype == jnp.bool_
        assert disabled_budget["relevance_probe_weight_scalars"] == 6
        assert disabled_budget["relevance_probe_weight_bytes"] == 24
        assert enabled_budget["relevance_probe_weight_bytes"] == 24
        assert disabled_budget["relevance_probe_bias_scalars"] == 2
        assert disabled_budget["relevance_probe_bias_bytes"] == 8
        assert enabled_budget["relevance_probe_bias_bytes"] == 8
        assert enabled_budget["relevance_probe_bytes"] == 32
        assert disabled_budget["candidate_promotion_evidence_streak_scalars"] == 4
        assert disabled_budget["candidate_promotion_evidence_streak_bytes"] == 16
        assert enabled_budget["candidate_promotion_evidence_streak_bytes"] == 16
        assert disabled_budget["candidate_reacquisition_required_scalars"] == 4
        assert disabled_budget["candidate_reacquisition_required_bytes"] == 4
        assert enabled_budget["candidate_reacquisition_required_bytes"] == 4

        with pytest.raises(ValueError, match="requires evidence_gated_active_output_memory"):
            FixedBudgetInteractionLearner(
                n_features=1,
                n_tasks=1,
                independent_relevance_probe=True,
            )

    def test_stale_retirement_resets_slot_and_every_matching_candidate(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=2,
            n_tasks=1,
            utility_decay=0.0,
            utility_retention_decay=0.9,
            utility_retention_grace_steps=1,
            utility_evidence_threshold=0.1,
            evidence_gated_active_output_memory=True,
            utility_evidence_confirmation_steps=2,
            independent_relevance_probe=True,
            retire_stale_features=True,
            candidate_promotion_floor=0.1,
            candidate_promotion_confirmation_steps=3,
            candidate_reacquisition_confirmation_steps=3,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=8,
            candidate_min_age=0,
            candidate_strategy="all_pairs",
            candidate_utility_retention_decay=0.9,
            refresh_candidates=False,
            refresh_promoted_candidate=False,
            scale_robust=True,
        )
        state = learner.init(feature_dim=3, key=jr.key(43)).replace(
            feature_left=jnp.array([0, 0], dtype=jnp.int32),
            feature_right=jnp.array([1, 2], dtype=jnp.int32),
            output_weights=jnp.ones((1, 2), dtype=jnp.float32),
            utilities=jnp.ones((2,), dtype=jnp.float32),
            evidence_idle_steps=jnp.array([1, 0], dtype=jnp.int32),
            ages=jnp.full((2,), 10, dtype=jnp.int32),
            candidate_output_weights=jnp.ones((1, 8), dtype=jnp.float32),
            candidate_utilities=jnp.ones((8,), dtype=jnp.float32),
            candidate_ages=jnp.full((8,), 10, dtype=jnp.int32),
            candidate_promotion_evidence_streak=jnp.full(
                (8,),
                2,
                dtype=jnp.int32,
            ),
            feature_second_moments=jnp.ones((2,), dtype=jnp.float32),
            candidate_second_moments=jnp.ones((8,), dtype=jnp.float32),
        )
        matching = (state.candidate_left == 0) & (state.candidate_right == 1)

        result = learner.update(
            state,
            jnp.ones(3, dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        )

        assert int(result.retired_slot) == 0
        assert int(result.retired_left) == 0
        assert int(result.retired_right) == 1
        assert int(result.replaced_slot) == -1
        assert int(result.promoted_candidate) == -1
        assert int(result.refreshed_candidate) == -1
        assert int(result.live_feature_count) == 1
        chex.assert_trees_all_equal(
            result.state.feature_left,
            jnp.array([-1, 0], dtype=jnp.int32),
        )
        chex.assert_trees_all_equal(
            result.state.feature_right,
            jnp.array([-1, 2], dtype=jnp.int32),
        )
        assert float(result.state.output_weights[0, 0]) == 0.0
        assert float(result.state.utilities[0]) == 0.0
        assert int(result.state.ages[0]) == 0
        assert int(result.state.evidence_idle_steps[0]) == 0
        assert float(result.state.feature_second_moments[0]) == 0.0
        chex.assert_trees_all_equal(
            result.state.candidate_output_weights[:, matching],
            jnp.zeros((1, int(jnp.sum(matching))), dtype=jnp.float32),
        )
        chex.assert_trees_all_equal(
            result.state.candidate_utilities[matching],
            jnp.zeros((int(jnp.sum(matching)),), dtype=jnp.float32),
        )
        chex.assert_trees_all_equal(
            result.state.candidate_ages[matching],
            jnp.zeros((int(jnp.sum(matching)),), dtype=jnp.int32),
        )
        chex.assert_trees_all_equal(
            result.state.candidate_second_moments[matching],
            jnp.zeros((int(jnp.sum(matching)),), dtype=jnp.float32),
        )
        chex.assert_trees_all_equal(
            result.candidate_promotion_evidence_streak_updated,
            jnp.full((8,), 3, dtype=jnp.int32),
        )
        chex.assert_trees_all_equal(
            result.state.candidate_promotion_evidence_streak[matching],
            jnp.zeros((int(jnp.sum(matching)),), dtype=jnp.int32),
        )
        chex.assert_trees_all_equal(
            result.state.candidate_promotion_evidence_streak[~matching],
            jnp.full((int(jnp.sum(~matching)),), 3, dtype=jnp.int32),
        )
        chex.assert_trees_all_equal(
            result.candidate_reacquisition_required_pre,
            jnp.zeros((8,), dtype=jnp.bool_),
        )
        chex.assert_trees_all_equal(
            result.state.candidate_reacquisition_required[matching],
            jnp.ones((int(jnp.sum(matching)),), dtype=jnp.bool_),
        )
        chex.assert_trees_all_equal(
            result.state.candidate_reacquisition_required[~matching],
            jnp.zeros((int(jnp.sum(~matching)),), dtype=jnp.bool_),
        )
        assert not bool(jnp.any(result.candidate_reacquisition_confirmed))
        pre_curation = result.pre_curation_state
        chex.assert_trees_all_equal(pre_curation.feature_left, state.feature_left)
        chex.assert_trees_all_equal(pre_curation.feature_right, state.feature_right)
        chex.assert_trees_all_equal(pre_curation.feature_parent_a, state.feature_parent_a)
        chex.assert_trees_all_equal(pre_curation.feature_parent_b, state.feature_parent_b)
        chex.assert_trees_all_equal(pre_curation.feature_generator, state.feature_generator)
        chex.assert_trees_all_equal(pre_curation.candidate_left, state.candidate_left)
        chex.assert_trees_all_equal(pre_curation.candidate_right, state.candidate_right)
        assert float(pre_curation.candidate_output_weights[0, matching][0]) != 0.0
        assert int(pre_curation.ages[0]) == 11
        assert int(pre_curation.candidate_ages[matching][0]) == 11
        assert int(pre_curation.candidate_promotion_evidence_streak[matching][0]) == 3
        assert not bool(pre_curation.candidate_reacquisition_required[matching][0])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "utility_retention_grace_steps": 2,
                "utility_evidence_threshold": 0.0,
            },
            {
                "retire_stale_features": True,
                "utility_retention_grace_steps": None,
                "candidate_promotion_floor": 0.1,
            },
            {
                "retire_stale_features": True,
                "utility_retention_grace_steps": 2,
                "utility_evidence_threshold": 0.1,
                "candidate_promotion_floor": 0.0,
            },
        ],
    )
    def test_evidence_lease_rejects_unsafe_controls(
        self,
        kwargs: dict[str, object],
    ) -> None:
        with pytest.raises(ValueError):
            FixedBudgetInteractionLearner(
                n_features=2,
                n_tasks=1,
                **kwargs,
            )

    def test_random_replacement_event_occurs(self) -> None:
        learner = FixedBudgetInteractionLearner(
            n_features=5,
            n_tasks=2,
            replacement_interval=1,
            min_feature_age=0,
            candidate_count=0,
            generator_mix=(1.0, 0.0, 0.0),
        )
        state = learner.init(feature_dim=4, key=jr.key(11))
        result = learner.update(
            state,
            jnp.ones(4, dtype=jnp.float32),
            jnp.array([0.5, -0.25], dtype=jnp.float32),
        )

        assert float(result.metrics[5]) == 1.0
        assert int(result.replaced_slot) >= 0
        assert int(result.state.ages[result.replaced_slot]) == 0

    def test_array_loop_shapes(self) -> None:
        stream = InteractionFeatureDiscoveryStream(
            feature_dim=5,
            n_tasks=2,
            context_length=8,
            active_pairs_per_context=2,
        )
        observations, targets = collect_feature_discovery_stream(
            stream, num_steps=10, key=jr.key(12)
        )
        learner = FixedBudgetInteractionLearner(
            n_features=8,
            n_tasks=2,
            replacement_interval=0,
        )
        state = learner.init(feature_dim=5, key=jr.key(13))
        result = run_interaction_feature_arrays(learner, state, observations, targets)

        chex.assert_shape(result.metrics, (10, 7))
        chex.assert_tree_all_finite(result.metrics)


class TestReplaceFractionRemoval:
    """The vestigial replace_fraction knob is gone but legacy configs load."""

    def test_constructor_rejects_replace_fraction(self) -> None:
        with pytest.raises(TypeError):
            FixedBudgetFeatureLearner(n_features=4, n_tasks=1, replace_fraction=0.5)  # type: ignore[call-arg]

    def test_to_config_omits_replace_fraction(self) -> None:
        learner = FixedBudgetFeatureLearner(n_features=4, n_tasks=1)

        assert "replace_fraction" not in learner.to_config()

    def test_from_config_drops_legacy_replace_fraction(self) -> None:
        learner = FixedBudgetFeatureLearner(n_features=4, n_tasks=2, candidate_count=1)
        legacy = dict(learner.to_config())
        legacy["replace_fraction"] = 0.25

        restored = FixedBudgetFeatureLearner.from_config(legacy)

        assert restored.to_config() == learner.to_config()
