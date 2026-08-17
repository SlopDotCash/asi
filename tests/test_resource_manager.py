"""Tests for learned resource managers."""

from __future__ import annotations

import math
from types import MappingProxyType

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from numpy.typing import NDArray

from alberta_framework import (
    GeneratorMetaResourceManager,
    LearnedResourceManager,
    finite_candidate_hedge_regret_bound,
    optimal_hedge_learning_rate,
)
from alberta_framework.core.compositional_features import (
    CompositionalFeatureLearner,
    run_compositional_arrays,
)


class TestLearnedResourceManager:
    """Behavioral checks for the contextual Hedge manager."""

    @staticmethod
    def _mixture_loss(
        manager: LearnedResourceManager,
        losses: NDArray[np.float64],
    ) -> float:
        state = manager.init()
        total = 0.0
        for row in losses:
            weights = manager.weights(state)
            total += float(np.dot(np.asarray(weights), row))
            state = manager.update(
                state,
                jnp.asarray(row, dtype=jnp.float32),
            ).state
        return total

    def test_init_shapes_and_uniform_weights(self) -> None:
        manager = LearnedResourceManager(n_actions=3, n_contexts=2)
        state = manager.init()

        chex.assert_shape(state.log_weights, (2, 3))
        chex.assert_shape(state.loss_ema, (2, 3))
        chex.assert_shape(state.action_counts, (2, 3))
        weights = manager.weights(state, 1)
        assert weights.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_weights_shift_toward_lower_loss_action(self) -> None:
        manager = LearnedResourceManager(
            n_actions=3,
            learning_rate=2.0,
            discount=1.0,
            exploration=0.0,
        )
        state = manager.init()
        for _ in range(20):
            result = manager.update(state, jnp.asarray([1.0, 0.1, 0.8]))
            state = result.state

        weights = manager.weights(state)
        assert int(jnp.argmax(weights)) == 1
        assert float(weights[1]) > 0.95

    def test_contexts_learn_independently(self) -> None:
        manager = LearnedResourceManager(
            n_actions=2,
            n_contexts=2,
            learning_rate=2.0,
            discount=1.0,
        )
        state = manager.init()
        for _ in range(10):
            state = manager.update(state, jnp.asarray([0.1, 1.0]), context_id=0).state
            state = manager.update(state, jnp.asarray([1.0, 0.1]), context_id=1).state

        assert int(jnp.argmax(manager.weights(state, 0))) == 0
        assert int(jnp.argmax(manager.weights(state, 1))) == 1

    def test_resource_cost_can_break_loss_tie(self) -> None:
        manager = LearnedResourceManager(
            n_actions=2,
            learning_rate=2.0,
            discount=1.0,
            cost_weight=1.0,
        )
        state = manager.init()
        losses = jnp.asarray([0.1, 0.1])
        costs = jnp.asarray([0.0, 1.0])
        for _ in range(10):
            state = manager.update(state, losses, resource_costs=costs).state

        weights = manager.weights(state)
        assert int(jnp.argmax(weights)) == 0
        assert float(weights[0]) > 0.99

    def test_nan_loss_is_ignored(self) -> None:
        manager = LearnedResourceManager(n_actions=2, learning_rate=1.0)
        state = manager.init()
        result = manager.update(state, jnp.asarray([0.1, jnp.nan]))

        assert float(result.advantages[1]) == 0.0
        assert result.state.action_counts[0, 0] == pytest.approx(1.0)
        assert result.state.action_counts[0, 1] == pytest.approx(0.0)

    def test_action_counts_increment_past_float32_exact_integer_limit(self) -> None:
        manager = LearnedResourceManager(n_actions=2)
        state = manager.init().replace(
            action_counts=manager.init().action_counts.at[0, 0].set(2**24)
        )

        result = manager.update(state, jnp.asarray([0.0, jnp.nan], dtype=jnp.float32))

        assert int(result.state.action_counts[0, 0]) == 2**24 + 1

    def test_zero_discount_recovers_nonfinite_logits(self) -> None:
        manager = LearnedResourceManager(
            n_actions=2,
            learning_rate=1.0,
            discount=0.0,
            exploration=0.0,
        )
        finite_state = manager.init().replace(
            log_weights=jnp.asarray([[2.0, -2.0]], dtype=jnp.float32)
        )
        chex.assert_trees_all_equal(
            manager.weights(finite_state),
            jax.nn.softmax(finite_state.log_weights[0]),
        )

        state = finite_state.replace(log_weights=jnp.full((1, 2), jnp.inf, dtype=jnp.float32))
        raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
        assert not bool(jnp.isfinite(raw))
        assert manager.weights(state).tolist() == pytest.approx([0.5, 0.5])

        result = manager.update(state, jnp.asarray([0.1, 1.0], dtype=jnp.float32))
        assert bool(result.update_applied)
        assert bool(jnp.all(jnp.isfinite(result.state.log_weights)))
        assert bool(jnp.all(jnp.isfinite(result.weights)))

    def test_nonzero_discount_rejects_nonfinite_logits(self) -> None:
        manager = LearnedResourceManager(n_actions=2, discount=0.5)
        state = manager.init().replace(log_weights=jnp.full((1, 2), jnp.inf, dtype=jnp.float32))

        assert not bool(jnp.all(jnp.isfinite(manager.weights(state))))
        result = manager.update(state, jnp.asarray([0.1, 1.0], dtype=jnp.float32))

        assert not bool(result.update_applied)
        chex.assert_trees_all_equal(result.state, state)

    def test_zero_loss_decay_recovers_observed_nonfinite_ema(self) -> None:
        manager = LearnedResourceManager(n_actions=2, loss_decay=0.0)
        state = manager.init().replace(loss_ema=jnp.asarray([[jnp.inf, 7.0]], dtype=jnp.float32))

        result = manager.update(state, jnp.asarray([0.1, jnp.nan], dtype=jnp.float32))

        assert bool(result.update_applied)
        assert result.state.loss_ema[0].tolist() == pytest.approx([0.1, 7.0])

    def test_zero_loss_decay_rejects_poisoned_ema_in_ignored_slot(self) -> None:
        manager = LearnedResourceManager(n_actions=2, loss_decay=0.0)
        state = manager.init().replace(loss_ema=jnp.asarray([[0.0, jnp.inf]], dtype=jnp.float32))

        result = manager.update(state, jnp.asarray([0.1, jnp.nan], dtype=jnp.float32))

        assert not bool(result.update_applied)
        chex.assert_trees_all_equal(result.state, state)

    def test_nonzero_loss_decay_rejects_consumed_nonfinite_ema(self) -> None:
        manager = LearnedResourceManager(n_actions=2, loss_decay=0.5)
        state = manager.init().replace(loss_ema=jnp.asarray([[jnp.inf, 0.0]], dtype=jnp.float32))

        result = manager.update(state, jnp.asarray([0.1, 1.0], dtype=jnp.float32))

        assert not bool(result.update_applied)
        chex.assert_trees_all_equal(result.state, state)

    def test_config_roundtrip(self) -> None:
        manager = LearnedResourceManager(
            n_actions=4,
            n_contexts=3,
            learning_rate=0.7,
            discount=0.9,
            exploration=0.05,
            loss_decay=0.8,
            cost_weight=0.2,
            advantage_clip=3.0,
        )
        clone = LearnedResourceManager.from_config(manager.to_config())

        assert clone.to_config() == manager.to_config()

    def test_fixed_candidate_regret_bound_matches_hedge_theorem(self) -> None:
        losses = np.asarray(
            [
                [0.10, 0.70, 0.40],
                [0.20, 0.60, 0.30],
                [0.15, 0.90, 0.20],
                [0.25, 0.20, 0.50],
                [0.20, 0.30, 0.45],
                [0.10, 0.80, 0.35],
                [0.30, 0.40, 0.40],
                [0.15, 0.70, 0.25],
            ],
            dtype=np.float64,
        )
        horizon, n_actions = losses.shape
        eta = optimal_hedge_learning_rate(n_actions, horizon)
        manager = LearnedResourceManager(
            n_actions=n_actions,
            learning_rate=eta,
            discount=1.0,
            exploration=0.0,
            advantage_clip=10.0,
        )

        mixture_loss = self._mixture_loss(manager, losses)
        best_fixed_loss = float(np.min(np.sum(losses, axis=0)))
        regret = mixture_loss - best_fixed_loss

        assert regret <= manager.fixed_candidate_regret_bound(horizon)
        assert math.isclose(
            manager.fixed_candidate_regret_bound(horizon),
            finite_candidate_hedge_regret_bound(n_actions, horizon, eta),
        )

    def test_regret_helpers_validate_theorem_preconditions(self) -> None:
        assert optimal_hedge_learning_rate(1, 10) == 0.0
        assert finite_candidate_hedge_regret_bound(1, 10, 0.0) == 0.0
        assert math.isinf(finite_candidate_hedge_regret_bound(2, 10, 0.0))

        with pytest.raises(ValueError):
            optimal_hedge_learning_rate(0, 10)
        with pytest.raises(ValueError):
            optimal_hedge_learning_rate(2, 0)
        with pytest.raises(ValueError):
            finite_candidate_hedge_regret_bound(2, 10, -0.1)
        with pytest.raises(ValueError):
            finite_candidate_hedge_regret_bound(2, 10, 0.1, loss_bound=0.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_actions": True, "horizon": 8},
            {"n_actions": 3, "horizon": True},
            {"n_actions": 3.0, "horizon": 8},
            {"n_actions": float("nan"), "horizon": 8},
            {"n_actions": 3, "horizon": float("inf")},
            {"n_actions": 3, "horizon": 8, "loss_bound": float("nan")},
            {"n_actions": 3, "horizon": 8, "loss_bound": float("inf")},
            {"n_actions": 3, "horizon": 8, "loss_bound": True},
        ],
    )
    def test_optimal_hedge_rate_rejects_bool_and_nonfinite_preconditions(
        self,
        kwargs: dict[str, object],
    ) -> None:
        with pytest.raises(ValueError):
            optimal_hedge_learning_rate(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_actions": True, "horizon": 8, "learning_rate": 0.1},
            {"n_actions": 3, "horizon": True, "learning_rate": 0.1},
            {"n_actions": 3, "horizon": 8, "learning_rate": float("nan")},
            {"n_actions": 3, "horizon": 8, "learning_rate": float("inf")},
            {"n_actions": 3, "horizon": 8, "learning_rate": True},
            {"n_actions": 3, "horizon": 8, "learning_rate": 0.1, "loss_bound": float("nan")},
            {"n_actions": 3, "horizon": 8, "learning_rate": 0.1, "loss_bound": True},
        ],
    )
    def test_hedge_regret_bound_rejects_bool_and_nonfinite_preconditions(
        self,
        kwargs: dict[str, object],
    ) -> None:
        with pytest.raises(ValueError):
            finite_candidate_hedge_regret_bound(**kwargs)  # type: ignore[arg-type]

    def test_manager_regret_bound_rejects_nonfinite_horizon_and_loss_bound(self) -> None:
        manager = LearnedResourceManager(
            n_actions=3,
            learning_rate=0.1,
            discount=1.0,
            exploration=0.0,
        )
        with pytest.raises(ValueError):
            manager.fixed_candidate_regret_bound(float("nan"))  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            manager.fixed_candidate_regret_bound(True)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            manager.fixed_candidate_regret_bound(8, loss_bound=float("nan"))

    def test_theorem_helpers_fail_closed_or_saturate_at_finite_extremes(self) -> None:
        with pytest.raises(ValueError, match="finite learning rate"):
            optimal_hedge_learning_rate(
                2,
                1,
                loss_bound=float.fromhex("0x0.0000000000001p-1022"),
            )

        bound = finite_candidate_hedge_regret_bound(
            2,
            1,
            learning_rate=1.0,
            loss_bound=1e308,
        )
        assert math.isinf(bound)


class TestGeneratorMetaResourceManager:
    """Behavioral checks for generator-internal meta-resource policies."""

    @staticmethod
    def _manager(
        *,
        discount: float = 0.995,
        reward_decay: float = 0.99,
    ) -> GeneratorMetaResourceManager:
        return GeneratorMetaResourceManager(
            policy_names=("safe", "residual"),
            op_ids=(1, 3),
            parent_modes=(0, 3),
            replacement_multipliers=(0.5, 2.0),
            promotion_margin_multipliers=(1.25, 0.75),
            candidate_min_age_multipliers=(1.5, 0.5),
            imprint_scales=(0.0, 1.0),
            learning_rate=1.0,
            discount=discount,
            exploration=0.0,
            reward_decay=reward_decay,
        )

    def test_action_counts_increment_past_float32_exact_integer_limit(self) -> None:
        manager = self._manager()
        state = manager.init().replace(  # type: ignore[attr-defined]
            action_counts=manager.init().action_counts.at[0, 0].set(2**24)
        )

        result = manager.update(
            state,
            jnp.asarray([1.0, jnp.nan], dtype=jnp.float32),
        )

        assert int(result.state.action_counts[0, 0]) == 2**24 + 1

    def test_zero_discount_recovers_nonfinite_logits(self) -> None:
        manager = self._manager(discount=0.0)
        finite_state = manager.init().replace(  # type: ignore[attr-defined]
            log_weights=jnp.asarray([[2.0, -2.0]], dtype=jnp.float32)
        )
        chex.assert_trees_all_equal(
            manager.weights(finite_state),
            jax.nn.softmax(finite_state.log_weights[0]),
        )

        state = finite_state.replace(  # type: ignore[attr-defined]
            log_weights=jnp.full((1, 2), jnp.inf, dtype=jnp.float32)
        )
        assert manager.weights(state).tolist() == pytest.approx([0.5, 0.5])

        result = manager.update(state, jnp.asarray([1.0, 0.1], dtype=jnp.float32))

        assert bool(result.update_applied)
        assert bool(jnp.all(jnp.isfinite(result.state.log_weights)))
        assert bool(jnp.all(jnp.isfinite(result.weights)))

    def test_nonzero_discount_rejects_nonfinite_logits(self) -> None:
        manager = self._manager(discount=0.5)
        state = manager.init().replace(  # type: ignore[attr-defined]
            log_weights=jnp.full((1, 2), jnp.inf, dtype=jnp.float32)
        )

        assert not bool(jnp.all(jnp.isfinite(manager.weights(state))))
        result = manager.update(state, jnp.asarray([1.0, 0.1], dtype=jnp.float32))

        assert not bool(result.update_applied)
        chex.assert_trees_all_equal(result.state, state)

    def test_zero_reward_decay_recovers_observed_nonfinite_ema(self) -> None:
        manager = self._manager(reward_decay=0.0)
        state = manager.init().replace(  # type: ignore[attr-defined]
            reward_ema=jnp.asarray([[jnp.inf, 7.0]], dtype=jnp.float32)
        )

        result = manager.update(
            state,
            jnp.asarray([1.0, 0.1], dtype=jnp.float32),
            finite_mask=jnp.asarray([True, False]),
        )

        assert bool(result.update_applied)
        assert result.state.reward_ema[0].tolist() == pytest.approx([1.0, 7.0])

    def test_zero_reward_decay_rejects_poisoned_ema_in_masked_slot(self) -> None:
        manager = self._manager(reward_decay=0.0)
        state = manager.init().replace(  # type: ignore[attr-defined]
            reward_ema=jnp.asarray([[0.0, jnp.inf]], dtype=jnp.float32)
        )

        result = manager.update(
            state,
            jnp.asarray([1.0, 0.1], dtype=jnp.float32),
            finite_mask=jnp.asarray([True, False]),
        )

        assert not bool(result.update_applied)
        chex.assert_trees_all_equal(result.state, state)

    def test_nonzero_reward_decay_rejects_consumed_nonfinite_ema(self) -> None:
        manager = self._manager(reward_decay=0.5)
        state = manager.init().replace(  # type: ignore[attr-defined]
            reward_ema=jnp.asarray([[jnp.inf, 0.0]], dtype=jnp.float32)
        )

        result = manager.update(state, jnp.asarray([1.0, 0.1], dtype=jnp.float32))

        assert not bool(result.update_applied)
        chex.assert_trees_all_equal(result.state, state)

    def test_contexts_learn_independently_from_rewards(self) -> None:
        manager = GeneratorMetaResourceManager(
            policy_names=("product", "tanh"),
            op_ids=(1, 3),
            parent_modes=(1, 3),
            replacement_multipliers=(1.0, 2.0),
            promotion_margin_multipliers=(1.0, 0.8),
            candidate_min_age_multipliers=(1.0, 0.5),
            imprint_scales=(0.0, 1.0),
            n_contexts=2,
            learning_rate=2.0,
            discount=1.0,
            exploration=0.0,
        )
        state = manager.init()

        for _ in range(10):
            state = manager.update(
                state,
                jnp.asarray([1.0, 0.1], dtype=jnp.float32),
                context_id=0,
            ).state
            state = manager.update(
                state,
                jnp.asarray([0.1, 1.0], dtype=jnp.float32),
                context_id=1,
            ).state

        assert int(jnp.argmax(manager.weights(state, 0))) == 0
        assert int(jnp.argmax(manager.weights(state, 1))) == 1

    def test_policy_probabilities_are_normalized_with_priors(self) -> None:
        manager = GeneratorMetaResourceManager(
            policy_names=("safe", "product", "residual"),
            op_ids=(1, 1, 3),
            parent_modes=(0, 2, 3),
            replacement_multipliers=(0.5, 1.0, 2.0),
            promotion_margin_multipliers=(1.25, 1.0, 0.75),
            candidate_min_age_multipliers=(1.5, 1.0, 0.5),
            imprint_scales=(0.0, 0.25, 1.0),
            exploration=0.1,
            initial_preferences=(-1.0, 0.0, 1.0),
        )
        weights = manager.weights(manager.init())

        assert float(jnp.sum(weights)) == pytest.approx(1.0)
        assert jnp.all(weights > 0.0)
        assert int(jnp.argmax(weights)) == 2

    def test_exp3_credit_updates_selected_reward_direction(self) -> None:
        manager = GeneratorMetaResourceManager(
            policy_names=("safe", "residual"),
            op_ids=(1, 3),
            parent_modes=(0, 3),
            replacement_multipliers=(0.5, 2.0),
            promotion_margin_multipliers=(1.25, 0.75),
            candidate_min_age_multipliers=(1.5, 0.5),
            imprint_scales=(0.0, 1.0),
            learning_rate=0.5,
            discount=1.0,
            exploration=0.1,
            update_rule="exp3",
        )
        state = manager.init()
        before = manager.weights(state)
        result = manager.update(
            state,
            jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            selected_action=1,
            selected_probability=before[1],
        )
        after = manager.weights(result.state)

        assert float(after[1]) > float(before[1])
        assert float(result.advantages[1]) > 0.0

    def test_select_returns_policy_knobs(self) -> None:
        manager = GeneratorMetaResourceManager(
            policy_names=("safe", "aggressive"),
            op_ids=(1, 4),
            parent_modes=(0, 3),
            replacement_multipliers=(0.5, 2.0),
            promotion_margin_multipliers=(1.25, 0.75),
            candidate_min_age_multipliers=(1.5, 0.5),
            imprint_scales=(0.0, 1.0),
            exploration=0.0,
        )
        state = manager.init().replace(  # type: ignore[attr-defined]
            log_weights=jnp.asarray([[10.0, -10.0]], dtype=jnp.float32)
        )

        decision = manager.select(state, jr.key(0))

        assert int(decision.action) == 0
        assert int(decision.op_id) == 1
        assert int(decision.parent_mode) == 0
        assert float(decision.replacement_multiplier) == pytest.approx(0.5)
        assert float(decision.promotion_margin_multiplier) == pytest.approx(1.25)
        assert float(decision.candidate_min_age_multiplier) == pytest.approx(1.5)
        assert float(decision.imprint_scale) == pytest.approx(0.0)

    def test_config_roundtrip(self) -> None:
        manager = GeneratorMetaResourceManager(
            policy_names=("a", "b", "c"),
            op_ids=(1, 3, 4),
            parent_modes=(0, 2, 3),
            replacement_multipliers=(0.5, 1.0, 2.0),
            promotion_margin_multipliers=(1.2, 1.0, 0.8),
            candidate_min_age_multipliers=(2.0, 1.0, 0.5),
            imprint_scales=(0.0, 0.5, 1.0),
            n_contexts=3,
            learning_rate=0.7,
            discount=0.9,
            exploration=0.05,
            reward_decay=0.8,
            cost_weight=0.1,
            advantage_clip=2.0,
            update_rule="exp3",
            initial_preferences=(-0.5, 0.0, 0.5),
        )

        clone = GeneratorMetaResourceManager.from_config(manager.to_config())

        assert clone.to_config() == manager.to_config()

    def test_generator_resource_training_metrics_are_finite(self) -> None:
        observations = jnp.asarray(
            [
                [0.2, 0.3, 0.1],
                [0.4, -0.5, 0.2],
                [-0.3, 0.7, -0.1],
                [0.6, 0.2, 0.4],
                [-0.5, -0.4, 0.3],
                [0.1, 0.8, -0.2],
            ],
            dtype=jnp.float32,
        )
        targets = (observations[:, 0] * observations[:, 1])[:, None]
        learner = CompositionalFeatureLearner(
            n_features=8,
            n_tasks=1,
            candidate_count=4,
            replacement_interval=2,
            min_feature_age=1,
            candidate_min_age=1,
            learn_generator_resources=True,
            generator_resource_update_rule="exp3",
            generator_resource_promotion_credit=0.5,
            generator_resource_cost_weight=0.1,
        )
        state = learner.init(feature_dim=3, key=jr.key(123))
        result = run_compositional_arrays(learner, state, observations, targets)

        assert jnp.all(jnp.isfinite(result.metrics))
        assert jnp.all(jnp.isfinite(result.state.generator_resource_state.log_weights))


def test_learned_resource_manager_integer_validation() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        LearnedResourceManager(n_actions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_actions"):
        LearnedResourceManager(n_actions=3.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_contexts"):
        LearnedResourceManager(n_actions=3, n_contexts=True)  # type: ignore[arg-type]

    rm = LearnedResourceManager(n_actions=np.int32(4), n_contexts=np.int64(2))
    assert rm.n_actions == 4
    assert rm.n_contexts == 2
    assert type(rm.n_actions) is int
    assert type(rm.n_contexts) is int


def test_generator_meta_resource_manager_integer_validation() -> None:
    with pytest.raises(ValueError, match="n_contexts"):
        GeneratorMetaResourceManager(
            policy_names=("p1",),
            op_ids=(0,),
            parent_modes=(0,),
            replacement_multipliers=(1.0,),
            promotion_margin_multipliers=(1.0,),
            candidate_min_age_multipliers=(1.0,),
            imprint_scales=(1.0,),
            n_contexts=True,  # type: ignore[arg-type]
        )

    mgr = GeneratorMetaResourceManager(
        policy_names=("p1",),
        op_ids=(np.int32(1),),
        parent_modes=(np.int64(0),),
        replacement_multipliers=(1.0,),
        promotion_margin_multipliers=(1.0,),
        candidate_min_age_multipliers=(1.0,),
        imprint_scales=(1.0,),
        n_contexts=np.int32(2),
    )
    assert mgr.n_contexts == 2
    assert mgr._op_ids == (1,)
    assert type(mgr._op_ids[0]) is int
    assert type(mgr.n_contexts) is int


class _FloatSubclass(float):
    def as_integer_ratio(self) -> tuple[int, int]:
        raise RuntimeError("hostile hook executed")


class _TupleSpoof:
    @property
    def __class__(self) -> type[tuple[object, ...]]:
        return tuple

    def __iter__(self) -> object:
        raise RuntimeError("hostile iterator executed")


def _minimal_generator_manager(**overrides: object) -> GeneratorMetaResourceManager:
    kwargs: dict[str, object] = {
        "policy_names": ("p1", "p2"),
        "op_ids": (0, 1),
        "parent_modes": (0, 1),
        "replacement_multipliers": (1.0, 1.0),
        "promotion_margin_multipliers": (1.0, 1.0),
        "candidate_min_age_multipliers": (1.0, 1.0),
        "imprint_scales": (0.0, 1.0),
    }
    kwargs.update(overrides)
    return GeneratorMetaResourceManager(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", _FloatSubclass(0.5)),
        ("discount", 1e100),
        ("discount", np.float64(1e-100)),
        ("exploration", np.float64(1e-100)),
        ("exploration", np.float64(1.0 - 1e-10)),
        ("loss_decay", np.float64(1.0 - 1e-10)),
        ("cost_weight", np.float64(1e-100)),
        ("advantage_clip", True),
    ],
)
def test_learned_resource_manager_rejects_invalid_float32_config(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        LearnedResourceManager(n_actions=2, **{field: value})  # type: ignore[arg-type]


def test_resource_manager_float32_config_is_canonical_and_json_safe() -> None:
    manager = LearnedResourceManager(
        n_actions=np.int16(2),
        learning_rate=np.float32(0.5),
        discount=np.float64(0.75),
        exploration=np.float16(0.125),
        loss_decay=np.float32(0.5),
        cost_weight=np.float64(0.25),
        advantage_clip=np.float32(2.0),
    )
    config = manager.to_config()
    assert all(type(config[name]) is float for name in (
        "learning_rate", "discount", "exploration", "loss_decay", "cost_weight",
        "advantage_clip",
    ))


def test_resource_manager_preflights_complete_state_before_allocation() -> None:
    # Conservative update/select working set is 49*n_actions+35 for one context.
    last_legal = ((2**31 - 1) // 4 - 35) // 49
    LearnedResourceManager(n_actions=last_legal)
    with pytest.raises(ValueError, match="state exceeds"):
        LearnedResourceManager(n_actions=last_legal + 1)
    # Multiple contexts are covered by the same aggregate formula.
    generator_last = ((2**31 - 1) // 4 - 115) // 18
    _minimal_generator_manager(n_contexts=generator_last)
    with pytest.raises(ValueError, match="state exceeds"):
        _minimal_generator_manager(n_contexts=generator_last + 1)


def test_generator_config_rejects_hostile_sequences_and_accepts_mapping_proxy() -> None:
    with pytest.raises(ValueError, match="policy_names"):
        _minimal_generator_manager(policy_names=_TupleSpoof())

    config = _minimal_generator_manager().to_config()
    clone = GeneratorMetaResourceManager.from_config(MappingProxyType(config))
    assert clone.to_config() == config
    config["op_ids"] = range(2)
    with pytest.raises(ValueError, match="op_ids"):
        GeneratorMetaResourceManager.from_config(config)


@pytest.mark.parametrize("shape", [(), (1,), (1, 2), (2, 1), (3,)])
def test_resource_manager_updates_reject_wrong_vector_shapes_under_jit(
    shape: tuple[int, ...]
) -> None:
    learned = LearnedResourceManager(n_actions=2)
    generator = _minimal_generator_manager()
    malformed = jnp.zeros(shape, dtype=jnp.float32)
    with pytest.raises(ValueError, match="losses"):
        learned.update(learned.init(), malformed)
    with pytest.raises(ValueError, match="rewards"):
        generator.update(generator.init(), malformed)


def test_generator_float_sequences_are_validated_at_float32_sink() -> None:
    with pytest.raises(ValueError, match="replacement_multipliers"):
        _minimal_generator_manager(replacement_multipliers=(1.0, 1e100))
    with pytest.raises(ValueError, match="initial_preferences"):
        _minimal_generator_manager(initial_preferences=(0.0, _FloatSubclass(1.0)))
    with pytest.raises(ValueError, match="initial_preferences span"):
        _minimal_generator_manager(initial_preferences=(-3e38, 3e38))
    manager = _minimal_generator_manager(
        replacement_multipliers=(np.float32(0.5), np.float64(2.0)),
        initial_preferences=(np.float32(-0.5), np.float64(0.5)),
    )
    config = manager.to_config()
    assert all(type(value) is float for value in config["replacement_multipliers"])
    assert all(type(value) is float for value in config["initial_preferences"])


def test_resource_manager_state_contract_and_counter_saturation() -> None:
    learned = LearnedResourceManager(n_actions=2)
    state = learned.init()
    malformed = state.replace(log_weights=jnp.zeros((2,), dtype=jnp.float32))
    with pytest.raises(ValueError, match="state.log_weights"):
        learned.weights(malformed)

    saturated = state.replace(
        action_counts=jnp.full_like(state.action_counts, 2**31 - 1),
        step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
    )
    result = learned.update(saturated, jnp.asarray([0.0, 1.0], dtype=jnp.float32))
    assert bool(result.update_applied)
    assert int(result.state.step_count) == 2**31 - 1
    assert bool(jnp.all(result.state.action_counts == 2**31 - 1))

    invalid = state.replace(step_count=jnp.asarray(-1, dtype=jnp.int32))
    result = learned.update(invalid, jnp.asarray([0.0, 1.0], dtype=jnp.float32))
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, invalid)

    invalid_counts = state.replace(action_counts=-jnp.ones_like(state.action_counts))
    result = learned.update(invalid_counts, jnp.asarray([0.0, 1.0], dtype=jnp.float32))
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, invalid_counts)

    generator = _minimal_generator_manager()
    generator_state = generator.init()
    generator_saturated = generator_state.replace(
        action_counts=jnp.full_like(generator_state.action_counts, 2**31 - 1),
        step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32),
    )
    generator_result = generator.update(
        generator_saturated, jnp.asarray([1.0, 0.5], dtype=jnp.float32)
    )
    assert bool(generator_result.update_applied)
    assert bool(jnp.all(generator_result.state.action_counts == 2**31 - 1))


def test_resource_manager_host_preflight_and_hostile_state_metadata() -> None:
    learned = LearnedResourceManager(n_actions=2)
    generator = _minimal_generator_manager()

    class HostVector:
        shape = (2,)
        dtype = np.dtype(np.float64)

        def __jax_array__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("conversion must not run")

    with pytest.raises(ValueError, match="losses"):
        learned.update(learned.init(), HostVector())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rewards"):
        generator.update(generator.init(), HostVector())  # type: ignore[arg-type]

    class HostileLeaf:
        @property
        def shape(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("shape hook")

        def __repr__(self) -> str:
            raise AssertionError("repr hook must not run")

    malformed = learned.init().replace(log_weights=HostileLeaf())
    with pytest.raises(ValueError, match="state.log_weights"):
        learned.weights(malformed)

    malformed_generator = generator.init().replace(log_weights=HostileLeaf())
    with pytest.raises(ValueError, match="state.log_weights"):
        generator.select(malformed_generator, jr.key(0))


def test_resource_manager_loader_failures_are_normalized() -> None:
    learned = LearnedResourceManager(n_actions=2)
    with pytest.raises(ValueError, match="serialized LearnedResourceManager"):
        LearnedResourceManager.from_config({**learned.to_config(), "unknown": 1})

    generator = _minimal_generator_manager()
    payload = generator.to_config()
    del payload["policy_names"]
    with pytest.raises(ValueError, match="serialized GeneratorMetaResourceManager"):
        GeneratorMetaResourceManager.from_config(payload)


def test_resource_manager_serialized_schemas_are_exact() -> None:
    learned_payload = LearnedResourceManager(n_actions=2).to_config()
    learned_clone = LearnedResourceManager.from_config(MappingProxyType(learned_payload))
    assert learned_clone.to_config() == learned_payload
    for mutation, match in (
        ({"type": "Other"}, "type"),
        ({"n_actions": np.int32(2)}, "n_actions"),
        ({"learning_rate": np.float32(0.1)}, "learning_rate"),
        ({"extra": 1}, "fields"),
    ):
        invalid = dict(learned_payload)
        invalid.update(mutation)
        with pytest.raises(ValueError, match=match):
            LearnedResourceManager.from_config(invalid)

    generator_payload = _minimal_generator_manager().to_config()
    for mutation, match in (
        ({"type": "Other"}, "type"),
        ({"n_contexts": np.int32(1)}, "n_contexts"),
        ({"policy_names": ("p1", "p2")}, "policy_names"),
        ({"op_ids": [np.int32(0), 1]}, "policy ids"),
        ({"replacement_multipliers": [1, 1.0]}, "replacement_multipliers"),
        ({"initial_preferences": [0, 0.0]}, "initial_preferences"),
        ({"extra": 1}, "fields"),
    ):
        invalid = dict(generator_payload)
        invalid.update(mutation)
        with pytest.raises(ValueError, match=match):
            GeneratorMetaResourceManager.from_config(invalid)
