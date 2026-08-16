"""Tests for action-conditioned environment prediction and dream guards."""

from __future__ import annotations

import dataclasses
import warnings
from fractions import Fraction

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.dreaming import (
    ActionConditionedDreamWorld,
    DreamBehaviorModelPrediction,
    DreamingConfig,
    DreamRolloutConfig,
    GuardedDreamer,
    RecentObservationBuffer,
    dream_rollout,
    imagined_rollout_to_gvf_items,
    init_dream_rollout_state,
)
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
    ActionConditionedWorldModelState,
    WorldModelPrediction,
    run_action_conditioned_world_model_learning_loop,
)


def test_action_conditioned_world_model_update_and_prediction_shapes() -> None:
    config = ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        observation_scale=(1.0, 2.0),
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        gamma=0.95,
    )
    model = ActionConditionedWorldModel(config)
    state = model.init(jr.key(0))

    result = model.update(
        state,
        jnp.array([0.2, -0.1], dtype=jnp.float32),
        jnp.array(1, dtype=jnp.int32),
        jnp.array(0.5, dtype=jnp.float32),
        jnp.array(0.95, dtype=jnp.float32),
        jnp.array([0.3, 0.1], dtype=jnp.float32),
    )

    chex.assert_shape(result.prediction.next_observation, (2,))
    chex.assert_shape(result.prediction.raw_predictions, (4,))
    chex.assert_shape(result.targets, (4,))
    chex.assert_shape(result.per_head_metrics, (4, 3))
    assert int(result.state.step_count) == 1
    chex.assert_tree_all_finite(result.prediction.raw_predictions)
    chex.assert_tree_all_finite(result.prediction_error)


def test_diagnostics_reflect_true_decodes_not_guard_clips() -> None:
    """Regression test for #391.

    The public prediction remains guarded for dream rollouts, while update
    diagnostics score both decoded observation and reward before those guards.
    """
    config = ActionConditionedWorldModelConfig(
        observation_dim=1,
        n_actions=2,
        hidden_sizes=(),
        sparsity=0.0,
        observation_clip_margin=0.05,
    )
    model = ActionConditionedWorldModel(config)
    init_state = model.init(jr.key(0))
    reward_head = config.observation_dim  # heads: [obs deltas..., reward, discount]

    def warmed_state_with_biases(
        observation_bias: float, reward_bias: float
    ) -> ActionConditionedWorldModelState:
        head_params = init_state.learner_state.head_params
        new_weights = tuple(
            jnp.zeros_like(w) if i in (0, reward_head) else w
            for i, w in enumerate(head_params.weights)
        )
        new_biases = tuple(
            jnp.full_like(
                b,
                observation_bias if i == 0 else reward_bias,
            )
            if i in (0, reward_head)
            else b
            for i, b in enumerate(head_params.biases)
        )
        learner_state = dataclasses.replace(
            init_state.learner_state,
            head_params=dataclasses.replace(
                head_params, weights=new_weights, biases=new_biases
            ),
        )
        return dataclasses.replace(
            init_state,
            learner_state=learner_state,
            observation_min=jnp.array([0.0], dtype=jnp.float32),
            observation_max=jnp.array([0.0], dtype=jnp.float32),
            reward_min=jnp.array(5.0, dtype=jnp.float32),
            reward_max=jnp.array(5.0, dtype=jnp.float32),
            step_count=jnp.array(1, dtype=jnp.int32),
        )

    obs = jnp.array([0.0], dtype=jnp.float32)
    action = jnp.array(0, dtype=jnp.int32)
    true_reward = jnp.array(5.0, dtype=jnp.float32)
    discount = jnp.array(0.9, dtype=jnp.float32)

    near_state = warmed_state_with_biases(0.05, 5.05)
    far_state = warmed_state_with_biases(50.0, 50.0)

    near_result = model.update(near_state, obs, action, true_reward, discount, obs)
    far_result = model.update(far_state, obs, action, true_reward, discount, obs)

    assert bool(near_result.update_applied)
    assert bool(far_result.update_applied)

    # The guard published to dream rollouts stays clipped to the observed
    # reward range +/- the margin, unchanged by this fix.
    assert float(far_result.prediction.reward) == pytest.approx(5.05, abs=1e-4)
    assert float(near_result.prediction.reward) == pytest.approx(5.05, abs=1e-4)
    assert float(far_result.prediction.next_observation[0]) == pytest.approx(
        0.05, abs=1e-4
    )
    assert float(near_result.prediction.next_observation[0]) == pytest.approx(
        0.05, abs=1e-4
    )

    # reward_error must separate a badly wrong head from a nearly-correct one
    # instead of both saturating at observation_clip_margin.
    assert float(near_result.reward_error) == pytest.approx(0.05, abs=1e-3)
    assert float(far_result.reward_error) == pytest.approx(45.0, abs=1e-3)
    assert float(near_result.next_observation_errors[0]) == pytest.approx(
        0.05, abs=1e-3
    )
    assert float(far_result.next_observation_errors[0]) == pytest.approx(
        config.max_delta_scale, abs=1e-3
    )


def test_action_conditioned_world_model_config_roundtrip() -> None:
    config = ActionConditionedWorldModelConfig(
        observation_dim=3,
        n_actions=4,
        observation_scale=(1.0, 2.0, 3.0),
        hidden_sizes=(8,),
        error_decay=0.9,
    )
    model = ActionConditionedWorldModel(config)
    restored = ActionConditionedWorldModel.from_config(model.to_config())

    assert restored.config == config
    features = restored.input_features(
        jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        jnp.array(2, dtype=jnp.int32),
    )
    chex.assert_shape(features, (7,))
    chex.assert_trees_all_close(features[-4:], jnp.array([0.0, 0.0, 1.0, 0.0]))


def test_action_conditioned_world_model_optional_interaction_features() -> None:
    config = ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=3,
        hidden_sizes=(),
        include_action_interactions=True,
    )
    model = ActionConditionedWorldModel(config)

    assert model.input_dim == 11
    features = model.input_features(
        jnp.array([2.0, -3.0], dtype=jnp.float32),
        jnp.array(1, dtype=jnp.int32),
    )

    chex.assert_shape(features, (11,))
    chex.assert_trees_all_close(features[:2], jnp.array([2.0, -3.0]))
    chex.assert_trees_all_close(features[2:5], jnp.array([0.0, 1.0, 0.0]))
    chex.assert_trees_all_close(
        features[5:],
        jnp.array([0.0, 2.0, 0.0, -0.0, -3.0, -0.0]),
    )


def test_action_interactions_keep_invalid_public_results_fail_visible() -> None:
    """Internal operands stay finite while invalid public results remain NaN."""
    model = ActionConditionedWorldModel(
        ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=3,
            hidden_sizes=(),
            include_action_interactions=True,
        )
    )
    obs = jnp.array([jnp.inf, 1.0], dtype=jnp.float32)
    raw = obs[:, None] * jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)[None, :]
    assert not bool(jnp.all(jnp.isfinite(raw)))

    state = model.init(jr.key(0))
    safe_features, features_valid, safe_obs = model._safe_input_features(
        obs,
        jnp.array(0, dtype=jnp.int32),
    )
    assert not bool(features_valid)
    chex.assert_tree_all_finite((safe_features, safe_obs))

    def evaluate(observation: jax.Array) -> tuple[jax.Array, WorldModelPrediction]:
        features = model.input_features(observation, jnp.array(0, dtype=jnp.int32))
        return features, model.predict(state, observation, jnp.array(0, dtype=jnp.int32))

    with jax.disable_jit():
        eager_features, eager_prediction = evaluate(obs)
    compiled_features, compiled_prediction = jax.jit(evaluate)(obs)

    assert bool(jnp.all(jnp.isnan(eager_features)))
    assert bool(jnp.all(jnp.isnan(compiled_features)))
    for prediction in (eager_prediction, compiled_prediction):
        assert bool(jnp.all(jnp.isnan(prediction.next_observation)))
        assert bool(jnp.isnan(prediction.reward))
        assert bool(jnp.isnan(prediction.discount))
        assert bool(jnp.all(jnp.isnan(prediction.raw_predictions)))


def test_world_model_feature_and_prediction_validity_parity_under_vmap() -> None:
    """Valid rows retain exact results while invalid rows stay fail-visible."""
    model = ActionConditionedWorldModel(
        ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=3,
            hidden_sizes=(),
            include_action_interactions=True,
        )
    )
    state = model.init(jr.key(2))
    valid_obs = jnp.array([2.0, -3.0], dtype=jnp.float32)
    invalid_obs = jnp.array([jnp.inf, -3.0], dtype=jnp.float32)
    observations = jnp.stack([valid_obs, invalid_obs, valid_obs])
    actions = jnp.array([1, 1, 3], dtype=jnp.int32)

    with jax.disable_jit():
        eager_features = model.input_features(valid_obs, actions[0])
        eager_prediction = model.predict(state, valid_obs, actions[0])
    compiled_features = model.input_features(valid_obs, actions[0])
    compiled_prediction = model.predict(state, valid_obs, actions[0])
    batched_features = jax.vmap(model.input_features)(
        observations,
        actions,
    )
    batched_predictions = jax.vmap(lambda obs, action: model.predict(state, obs, action))(
        observations,
        actions,
    )

    chex.assert_trees_all_equal(eager_features, compiled_features)
    chex.assert_trees_all_equal(eager_features, batched_features[0])
    chex.assert_trees_all_equal(eager_prediction, compiled_prediction)
    chex.assert_trees_all_equal(
        eager_prediction,
        jax.tree.map(lambda value: value[0], batched_predictions),
    )
    for row in (1, 2):
        assert bool(jnp.all(jnp.isnan(batched_features[row])))
        assert bool(jnp.all(jnp.isnan(batched_predictions.next_observation[row])))
        assert bool(jnp.isnan(batched_predictions.reward[row]))
        assert bool(jnp.isnan(batched_predictions.discount[row]))
        assert bool(jnp.all(jnp.isnan(batched_predictions.raw_predictions[row])))


def test_action_conditioned_world_model_scan_loop_shapes() -> None:
    config = ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
    )
    model = ActionConditionedWorldModel(config)
    state = model.init(jr.key(8))
    observations = jnp.array(
        [[0.0, 0.0], [0.1, 0.0], [0.1, 0.2]],
        dtype=jnp.float32,
    )
    next_observations = jnp.array(
        [[0.1, 0.0], [0.1, 0.2], [0.2, 0.2]],
        dtype=jnp.float32,
    )
    result = run_action_conditioned_world_model_learning_loop(
        model,
        state,
        observations,
        jnp.array([0, 1, 0], dtype=jnp.int32),
        jnp.array([1.0, 0.5, 0.25], dtype=jnp.float32),
        next_observations,
        jnp.array([0.99, 0.99, 0.0], dtype=jnp.float32),
    )

    assert int(result.state.step_count) == 3
    chex.assert_shape(result.next_observation_predictions, (3, 2))
    chex.assert_shape(result.reward_predictions, (3,))
    chex.assert_shape(result.discount_predictions, (3,))
    chex.assert_shape(result.per_head_metrics, (3, 4, 3))
    chex.assert_tree_all_finite(result.prediction_errors)


def test_default_discounts_do_not_inherit_the_reward_dtype() -> None:
    """An integer or bool reward array must not truncate gamma into the default discounts."""
    config = ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
    )
    model = ActionConditionedWorldModel(config)
    state = model.init(jr.key(8))
    observations = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.2]], dtype=jnp.float32)
    next_observations = jnp.array([[0.1, 0.0], [0.1, 0.2], [0.2, 0.2]], dtype=jnp.float32)
    actions = jnp.array([0, 1, 0], dtype=jnp.int32)
    integer_rewards = jnp.array([1, 0, 1], dtype=jnp.int32)
    float_rewards = integer_rewards.astype(jnp.float32)
    explicit = run_action_conditioned_world_model_learning_loop(
        model,
        state,
        observations,
        actions,
        float_rewards,
        next_observations,
        jnp.full((3,), config.gamma, dtype=jnp.float32),
    )
    for rewards in (float_rewards, integer_rewards, integer_rewards.astype(jnp.bool_)):
        defaulted = run_action_conditioned_world_model_learning_loop(
            model, state, observations, actions, rewards, next_observations
        )
        chex.assert_trees_all_close(defaulted.discount_errors, explicit.discount_errors)
        chex.assert_trees_all_close(
            defaulted.discount_predictions, explicit.discount_predictions
        )
        assert bool(jnp.all(defaulted.updates_applied))
@pytest.mark.parametrize(
    "overrides",
    [
        {"observation_clip_margin": float("nan")},
        {"max_delta_scale": float("nan")},
        {"max_delta_scale": float("inf")},
        {"reward_scale": float("nan")},
        {"reward_scale": float("inf")},
        {"observation_scale": (float("nan"), 1.0)},
        {"utility_decay": float("nan")},
        {"error_decay": float("nan")},
        {"gamma": float("nan")},
    ],
)
def test_config_rejects_non_finite_scalars(overrides: dict[str, object]) -> None:
    """A NaN scalar passes a bare `< 0` check and yields an all-rejected run scoring 0.0."""
    config = ActionConditionedWorldModelConfig(
        observation_dim=2, n_actions=2, hidden_sizes=(), **overrides  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="finite"):
        ActionConditionedWorldModel(config)


class _FloatSpoof:
    """Not a Real: reports float through __class__ and never compares as out of range."""

    @property
    def __class__(self) -> type[float]:  # type: ignore[override]
        return float

    def as_integer_ratio(self) -> tuple[int, int]:
        return (1, 2)

    def __float__(self) -> float:
        return 0.5

    def __le__(self, other: object) -> bool:
        return True

    def __lt__(self, other: object) -> bool:
        return True

    def __ge__(self, other: object) -> bool:
        return True

    def __gt__(self, other: object) -> bool:
        return True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"reward_scale": _FloatSpoof()}, "reward_scale must be a finite real number"),
        ({"observation_clip_margin": _FloatSpoof()}, "must be a finite real number"),
        ({"reward_scale": 1e100}, "reward_scale must remain finite once narrowed"),
        ({"max_delta_scale": 1e100}, "max_delta_scale must remain finite once narrowed"),
        ({"reward_scale": 1e-100}, "reward_scale must remain positive once narrowed"),
        ({"max_delta_scale": 1e-100}, "max_delta_scale must remain positive once narrowed"),
        ({"observation_scale": (1e-100, 1.0)}, "must remain positive once narrowed"),
        (
            {"utility_decay": 1.0 - 1e-10},
            r"utility_decay must remain in \[0.0, 1.0\) once narrowed",
        ),
        ({"error_decay": 1.0 - 1e-10}, r"error_decay must remain in \[0.0, 1.0\) once narrowed"),
        ({"gamma": Fraction(1, 1) + Fraction(1, 2**60)}, r"gamma must be in \[0.0, 1.0\]"),
    ],
)
def test_config_rejects_scalars_that_leave_the_float32_domain(
    overrides: dict[str, object], message: str
) -> None:
    """Host-finite values must also stay finite and in range once narrowed to float32."""
    config = ActionConditionedWorldModelConfig(
        observation_dim=2, n_actions=2, hidden_sizes=(), **overrides  # type: ignore[arg-type]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=message):
            ActionConditionedWorldModel(config)


def test_config_canonicalizes_real_scalars_and_preserves_builtin_floats() -> None:
    model = ActionConditionedWorldModel(
        ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            reward_scale=Fraction(1, 4),
            observation_scale=(np.float64(2.0), 1),
            utility_decay=0.99,
        )
    )
    assert type(model.config.reward_scale) is float and model.config.reward_scale == 0.25
    assert model.config.observation_scale == (2.0, 1.0)
    assert all(type(value) is float for value in model.config.observation_scale)
    assert model.config.utility_decay == 0.99
    restored = ActionConditionedWorldModel.from_config(model.to_config())
    assert restored.config == model.config


def test_config_normalizes_real_comparison_hooks_and_conversion_failures() -> None:
    class LyingFraction(Fraction):
        def __gt__(self, other: object) -> bool:
            return True

        def __ge__(self, other: object) -> bool:
            return True

        def __le__(self, other: object) -> bool:
            return True

    class BrokenFraction(Fraction):
        def as_integer_ratio(self) -> tuple[int, int]:
            raise RuntimeError("conversion hook failed")

    with pytest.raises(ValueError, match="reward_scale must be positive"):
        ActionConditionedWorldModel(
            ActionConditionedWorldModelConfig(
                observation_dim=2,
                n_actions=2,
                hidden_sizes=(),
                reward_scale=LyingFraction(-1, 1),
            )
        )
    with pytest.raises(ValueError, match="gamma must be a finite real number"):
        ActionConditionedWorldModel(
            ActionConditionedWorldModelConfig(
                observation_dim=2,
                n_actions=2,
                hidden_sizes=(),
                gamma=BrokenFraction(1, 2),
            )
        )


@pytest.mark.parametrize("discount", [1.5, -0.5, 5.0])
def test_update_rejects_out_of_range_discounts(discount: float) -> None:
    config = ActionConditionedWorldModelConfig(
        observation_dim=2, n_actions=2, hidden_sizes=(), step_size=0.05, sparsity=0.0
    )
    model = ActionConditionedWorldModel(config)
    state = model.init(jr.key(8))
    obs = jnp.array([0.1, -0.2], dtype=jnp.float32)
    result = model.update(state, obs, jnp.int32(1), 0.5, discount, jnp.array([0.2, 0.1]))
    assert not bool(result.update_applied)
    assert int(result.state.step_count) == int(state.step_count)
    chex.assert_trees_all_equal(
        result.state.learner_state.trunk_params, state.learner_state.trunk_params
    )
    accepted = model.update(state, obs, jnp.int32(1), 0.5, 0.9, jnp.array([0.2, 0.1]))
    assert bool(accepted.update_applied)


def test_guarded_dreamer_rejects_warmup_and_accepts_after_real_updates() -> None:
    config = ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        error_decay=0.0,
    )
    model = ActionConditionedWorldModel(config)
    model_state = model.init(jr.key(1))
    dreamer = GuardedDreamer(
        DreamingConfig(warmup_steps=1, max_model_error_ema=100.0, max_uncertainty=0.1)
    )
    obs = jnp.array([0.0, 0.0], dtype=jnp.float32)
    action = jnp.array(0, dtype=jnp.int32)

    cold = dreamer.propose(model, model_state, obs, action)
    assert int(cold.reject_code) == GuardedDreamer.REJECT_WARMUP
    assert not bool(cold.accepted)

    update = model.update(
        model_state,
        obs,
        action,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array(0.99, dtype=jnp.float32),
        jnp.array([0.1, 0.0], dtype=jnp.float32),
    )
    warm = dreamer.propose(
        model,
        update.state,
        obs,
        action,
        uncertainty=jnp.array(0.0, dtype=jnp.float32),
    )

    assert int(warm.reject_code) == GuardedDreamer.ACCEPT
    assert bool(warm.accepted)
    chex.assert_shape(warm.transition.next_observation, (2,))


def test_recent_observation_buffer_ring_and_sample() -> None:
    buffer = RecentObservationBuffer(capacity=2, observation_dim=3)
    state = buffer.init()
    state = buffer.add(state, jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32))
    state = buffer.add(state, jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32))
    state = buffer.add(state, jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32))

    assert int(state.size) == 2
    sample, idx = buffer.sample(state, jr.key(3))
    chex.assert_shape(sample, (3,))
    assert 0 <= int(idx) < 2


class _ConstantBehavior:
    def sample_action(
        self,
        state: object,
        observation: jnp.ndarray,
        key: jnp.ndarray,
    ) -> DreamBehaviorModelPrediction:
        del state, observation, key
        return DreamBehaviorModelPrediction(
            action=jnp.array(1, dtype=jnp.int32),
            action_probability=jnp.array(1.0, dtype=jnp.float32),
            log_probability=jnp.array(0.0, dtype=jnp.float32),
        )


def test_action_conditioned_dream_rollout_converts_to_gvf_items() -> None:
    config = ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        error_decay=0.0,
    )
    model = ActionConditionedWorldModel(config)
    state = model.init(jr.key(4))
    obs = jnp.array([0.0, 0.0], dtype=jnp.float32)
    update = model.update(
        state,
        obs,
        jnp.array(1, dtype=jnp.int32),
        jnp.array(0.25, dtype=jnp.float32),
        jnp.array(0.99, dtype=jnp.float32),
        jnp.array([0.1, 0.0], dtype=jnp.float32),
    )

    rollout = dream_rollout(
        ActionConditionedDreamWorld(model),
        update.state,
        _ConstantBehavior(),
        None,
        init_dream_rollout_state(obs, jr.key(5)),
        DreamRolloutConfig(rollout_horizon=2, max_model_error=100.0),
    )
    gvf_item = imagined_rollout_to_gvf_items(rollout)

    chex.assert_shape(rollout.transitions.observation, (2, 2))
    chex.assert_shape(gvf_item.observations, (2, 2))
    chex.assert_shape(gvf_item.cumulants, (2, 1))
    chex.assert_shape(gvf_item.discounts, (2,))
