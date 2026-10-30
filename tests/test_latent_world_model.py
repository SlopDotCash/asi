"""Tests for latent predictive world models and dream selection."""

from __future__ import annotations

import math
import warnings
from types import MappingProxyType

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from alberta_framework.core.behavior_model import BehaviorModel, BehaviorModelConfig
from alberta_framework.core.dreaming import (
    BehaviorModelDreamPolicy,
    DreamSelectionConfig,
    score_dream_candidates,
)
from alberta_framework.core.latent_world_model import (
    EVIDENCE_LEVEL,
    SCIENTIFIC_PROMOTION_ALLOWED,
    LatentWorldModel,
    LatentWorldModelConfig,
    run_latent_world_model_learning_loop,
)

_ENCODER_CONFIG_KEYS = (
    "encoder_learning",
    "encoder_step_size",
    "max_encoder_update",
    "encoder_collapse_gate_threshold",
)


def _toy_rotation_stream(
    num_steps: int,
) -> tuple[Array, Array, Array, Array]:
    """Deterministic learnable dynamics: damped 2D rotation + action shift."""
    angle = 0.35
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x = [1.0, 0.0, 0.5]
    observations: list[list[float]] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_observations: list[list[float]] = []
    for step in range(num_steps):
        action = step % 2
        shift = 0.2 if action == 1 else -0.2
        next_x = [
            0.98 * (cos_a * x[0] - sin_a * x[1]),
            0.98 * (sin_a * x[0] + cos_a * x[1]),
            0.9 * x[2] + 0.1 * shift,
        ]
        observations.append(list(x))
        actions.append(action)
        rewards.append(x[0])
        next_observations.append(list(next_x))
        x = next_x
    return (
        jnp.array(observations, dtype=jnp.float32),
        jnp.array(actions, dtype=jnp.int32),
        jnp.array(rewards, dtype=jnp.float32),
        jnp.array(next_observations, dtype=jnp.float32),
    )


def test_latent_world_model_update_and_prediction_shapes() -> None:
    config = LatentWorldModelConfig(
        observation_dim=3,
        n_actions=2,
        latent_dim=5,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        gamma=0.95,
        observation_scale=(1.0, 2.0, 3.0),
    )
    model = LatentWorldModel(config)
    state = model.init(jr.key(0))

    result = model.update(
        state,
        jnp.array([0.2, -0.1, 0.3], dtype=jnp.float32),
        jnp.array(1, dtype=jnp.int32),
        jnp.array(0.5, dtype=jnp.float32),
        jnp.array(0.95, dtype=jnp.float32),
        jnp.array([0.3, 0.0, 0.2], dtype=jnp.float32),
    )

    assert int(result.state.step_count) == 1
    chex.assert_shape(result.prediction.latent, (5,))
    chex.assert_shape(result.prediction.next_latent, (5,))
    chex.assert_shape(result.prediction.raw_predictions, (7,))
    chex.assert_shape(result.targets, (7,))
    chex.assert_tree_all_finite(result.surprise)
    chex.assert_tree_all_finite(result.latent_std_mean)


@pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
def test_latent_world_model_rejects_wrong_rank_observation_vectors(
    shape: tuple[int, int],
) -> None:
    """Per-event APIs must not flatten row or column matrices into vectors."""
    model = LatentWorldModel(
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            latent_dim=2,
            hidden_sizes=(),
            sparsity=0.0,
        )
    )
    state = model.init(jr.key(320))
    observation = jnp.asarray([[0.25, -0.5]], dtype=jnp.float32).reshape(shape)
    vector = jnp.asarray([0.25, -0.5], dtype=jnp.float32)
    action = jnp.asarray(1, dtype=jnp.int32)
    reward = jnp.asarray(0.5, dtype=jnp.float32)
    discount = jnp.asarray(0.9, dtype=jnp.float32)

    calls = (
        lambda: model.encode(state, observation),
        lambda: model.input_features(state, observation, action),
        lambda: model.targets(state, observation, reward, discount, vector),
        lambda: model.targets(state, vector, reward, discount, observation),
        lambda: model.predict(state, observation, action),
        lambda: model.update(state, observation, action, reward, discount, vector),
        lambda: model.update(state, vector, action, reward, discount, observation),
    )
    for call in calls:
        with pytest.raises(ValueError, match=r"must have shape \(2,\)"):
            call()
        with pytest.raises(ValueError, match=r"must have shape \(2,\)"):
            jax.jit(call)()


@pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
def test_latent_world_model_rejects_wrong_rank_latent_vectors(
    shape: tuple[int, int],
) -> None:
    model = LatentWorldModel(
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            latent_dim=2,
            hidden_sizes=(),
            sparsity=0.0,
        )
    )
    state = model.init(jr.key(321))
    latent = jnp.asarray([[0.25, -0.5]], dtype=jnp.float32).reshape(shape)
    action = jnp.asarray(1, dtype=jnp.int32)

    calls = (
        lambda: model.input_features_from_latent(latent, action),
        lambda: model.predict_from_latent(state, latent, action),
    )
    for call in calls:
        with pytest.raises(ValueError, match=r"must have shape \(2,\)"):
            call()
        with pytest.raises(ValueError, match=r"must have shape \(2,\)"):
            jax.jit(call)()


@pytest.mark.parametrize(
    ("operation", "field"),
    [
        ("input_features_from_latent", "action"),
        ("input_features", "action"),
        ("targets", "reward"),
        ("targets", "discount"),
        ("predict_from_latent", "action"),
        ("predict", "action"),
        ("update", "action"),
        ("update", "reward"),
        ("update", "discount"),
    ],
)
def test_latent_world_model_rejects_size_one_scalar_aliases(
    operation: str,
    field: str,
) -> None:
    model = LatentWorldModel(
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            latent_dim=2,
            hidden_sizes=(),
            sparsity=0.0,
        )
    )
    state = model.init(jr.key(322))
    values = {
        "observation": jnp.asarray([0.25, -0.5], dtype=jnp.float32),
        "latent": jnp.asarray([0.1, -0.2], dtype=jnp.float32),
        "action": jnp.asarray(1, dtype=jnp.int32),
        "reward": jnp.asarray(0.5, dtype=jnp.float32),
        "discount": jnp.asarray(0.9, dtype=jnp.float32),
        "next_observation": jnp.asarray([0.5, 0.75], dtype=jnp.float32),
    }
    values[field] = jnp.ones((1,), dtype=jnp.float32)

    def call() -> object:
        if operation == "input_features_from_latent":
            return model.input_features_from_latent(values["latent"], values["action"])
        if operation == "input_features":
            return model.input_features(state, values["observation"], values["action"])
        if operation == "targets":
            return model.targets(
                state,
                values["observation"],
                values["reward"],
                values["discount"],
                values["next_observation"],
            )
        if operation == "predict_from_latent":
            return model.predict_from_latent(state, values["latent"], values["action"])
        if operation == "predict":
            return model.predict(state, values["observation"], values["action"])
        return model.update(
            state,
            values["observation"],
            values["action"],
            values["reward"],
            values["discount"],
            values["next_observation"],
        )

    with pytest.raises(ValueError, match=field):
        call()
    with pytest.raises(ValueError, match=field):
        jax.jit(call)()


def test_latent_default_discounts_do_not_inherit_the_reward_dtype() -> None:
    config = LatentWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        latent_dim=4,
        hidden_sizes=(8,),
    )
    model = LatentWorldModel(config)
    state = model.init(jr.key(3))
    observations = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.2]], dtype=jnp.float32)
    next_observations = jnp.array([[0.1, 0.0], [0.1, 0.2], [0.2, 0.2]], dtype=jnp.float32)
    actions = jnp.array([0, 1, 0], dtype=jnp.int32)
    integer_rewards = jnp.array([1, 0, 1], dtype=jnp.int32)
    explicit = run_latent_world_model_learning_loop(
        model,
        state,
        observations,
        actions,
        integer_rewards.astype(jnp.float32),
        next_observations,
        jnp.full((3,), config.gamma, dtype=jnp.float32),
    )
    defaulted = run_latent_world_model_learning_loop(
        model, state, observations, actions, integer_rewards, next_observations
    )
    chex.assert_trees_all_close(defaulted.discount_errors, explicit.discount_errors)
    chex.assert_trees_all_close(defaulted.discount_predictions, explicit.discount_predictions)


def test_latent_learning_loop_rejects_fractional_discrete_action_array() -> None:
    config = LatentWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        latent_dim=4,
        hidden_sizes=(),
    )
    model = LatentWorldModel(config)
    state = model.init(jr.key(4))
    observations = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
    next_observations = jnp.array([[0.1, 0.0]], dtype=jnp.float32)

    with pytest.raises(ValueError, match="actions must have an integer dtype"):
        run_latent_world_model_learning_loop(
            model,
            state,
            observations,
            jnp.array([0.5], dtype=jnp.float32),
            jnp.array([0.0], dtype=jnp.float32),
            next_observations,
        )


class _FloatSpoof:
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
        ({"encoder_scale": _FloatSpoof()}, "encoder_scale must be a finite real number"),
        ({"reward_scale": 1e100}, "reward_scale must remain finite once narrowed"),
        ({"encoder_scale": 1e-100}, "encoder_scale must remain positive once narrowed"),
        ({"max_latent_delta": 1e-100}, "max_latent_delta must remain positive once narrowed"),
        (
            {"surprise_decay": 1.0 - 1e-10},
            r"surprise_decay must remain in \[0.0, 1.0\) once narrowed",
        ),
        (
            {"collapse_decay": 1.0 - 1e-10},
            r"collapse_decay must remain in \[0.0, 1.0\) once narrowed",
        ),
        ({"encoder_step_size": 1e100}, "encoder_step_size must remain finite once narrowed"),
    ],
)
def test_latent_config_rejects_scalars_that_leave_the_float32_domain(
    overrides: dict[str, object], message: str
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=message):
            LatentWorldModelConfig(
                observation_dim=2,
                n_actions=2,
                latent_dim=4,
                hidden_sizes=(8,),
                **overrides,  # type: ignore[arg-type]
            )


def test_latent_config_canonicalizes_real_scalars() -> None:
    from fractions import Fraction

    model = LatentWorldModel(
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            latent_dim=4,
            hidden_sizes=(8,),
            encoder_scale=Fraction(1, 2),
            min_latent_std=np.float64(0.05),
        )
    )
    assert type(model.config.encoder_scale) is float and model.config.encoder_scale == 0.5
    assert type(model.config.min_latent_std) is float
    assert model.config.min_latent_std == float(np.float32(0.05))
    restored = LatentWorldModel.from_config(model.to_config())
    assert restored.config == model.config


@pytest.mark.parametrize("discount", [1.5, -0.5])
def test_latent_update_rejects_out_of_range_discounts(discount: float) -> None:
    config = LatentWorldModelConfig(observation_dim=2, n_actions=2, latent_dim=4, hidden_sizes=(8,))
    model = LatentWorldModel(config)
    state = model.init(jr.key(3))
    obs = jnp.array([0.1, -0.2], dtype=jnp.float32)
    result = model.update(state, obs, jnp.int32(1), 0.5, discount, jnp.array([0.2, 0.1]))
    assert not bool(result.update_applied)
    assert int(result.state.step_count) == int(state.step_count)
    chex.assert_trees_all_equal(
        result.state.learner_state.trunk_params, state.learner_state.trunk_params
    )


def test_latent_update_saturates_step_count_at_int32_max() -> None:
    config = LatentWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        latent_dim=2,
        hidden_sizes=(),
    )
    model = LatentWorldModel(config)
    state = model.init(jr.key(0)).replace(
        step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32)
    )

    result = model.update(
        state,
        jnp.asarray([0.1, -0.2], dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(0.9, dtype=jnp.float32),
        jnp.asarray([0.2, -0.1], dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    assert int(result.state.step_count) == 2**31 - 1


def test_latent_world_model_scan_loop_and_config_roundtrip() -> None:
    config = LatentWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        latent_dim=4,
        hidden_sizes=(8,),
        include_action_interactions=True,
        surprise_decay=0.9,
    )
    model = LatentWorldModel(config)
    restored = LatentWorldModel.from_config(model.to_config())
    assert restored.config == config

    state = restored.init(jr.key(3))
    observations = jnp.array(
        [[0.0, 0.0], [0.1, 0.0], [0.1, 0.2]],
        dtype=jnp.float32,
    )
    next_observations = jnp.array(
        [[0.1, 0.0], [0.1, 0.2], [0.2, 0.2]],
        dtype=jnp.float32,
    )
    result = run_latent_world_model_learning_loop(
        restored,
        state,
        observations,
        jnp.array([0, 1, 0], dtype=jnp.int32),
        jnp.array([1.0, 0.5, 0.25], dtype=jnp.float32),
        next_observations,
        jnp.array([0.99, 0.99, 0.0], dtype=jnp.float32),
    )

    assert int(result.state.step_count) == 3
    chex.assert_shape(result.latent_predictions, (3, 4))
    chex.assert_shape(result.next_latent_predictions, (3, 4))
    chex.assert_shape(result.reward_predictions, (3,))
    chex.assert_shape(result.surprises, (3,))
    chex.assert_shape(result.per_head_metrics, (3, 6, 3))
    chex.assert_tree_all_finite(result.prediction_errors)


def test_latent_action_interactions_do_not_form_zero_times_inf() -> None:
    """Inf latent times a silent action one-hot is 0*inf = NaN in the product."""
    model = LatentWorldModel(
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=3,
            latent_dim=2,
            hidden_sizes=(),
            include_action_interactions=True,
        )
    )
    latent = jnp.array([jnp.inf, 1.0], dtype=jnp.float32)
    raw = latent[:, None] * jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)[None, :]
    assert not bool(jnp.all(jnp.isfinite(raw)))

    features = model.input_features_from_latent(latent, jnp.array(0, dtype=jnp.int32))
    assert bool(jnp.all(jnp.isnan(features)))


def test_score_dream_candidates_selects_surprising_useful_valid_items() -> None:
    result = score_dream_candidates(
        surprises=jnp.array([0.1, 0.9, 0.7, 0.3], dtype=jnp.float32),
        utilities=jnp.array([1.0, -1.0, 0.5, 0.4], dtype=jnp.float32),
        confidences=jnp.array([1.0, 1.0, 0.2, 1.0], dtype=jnp.float32),
        model_errors=jnp.array([0.0, 0.0, 0.0, 2.0], dtype=jnp.float32),
        config=DreamSelectionConfig(
            max_items=2,
            surprise_weight=1.0,
            utility_weight=2.0,
            min_surprise=0.2,
            min_utility=0.0,
            min_confidence=0.5,
            max_model_error=1.0,
        ),
    )

    chex.assert_shape(result.selected_indices, (2,))
    assert bool(result.accepted[0]) is False
    assert bool(result.accepted[1]) is False
    assert bool(result.accepted[2]) is False
    assert bool(result.accepted[3]) is False
    assert not bool(jnp.any(result.selected_mask))

    permissive = score_dream_candidates(
        surprises=jnp.array([0.1, 0.9, 0.7, 0.3], dtype=jnp.float32),
        utilities=jnp.array([1.0, -1.0, 0.5, 0.4], dtype=jnp.float32),
        config=DreamSelectionConfig(max_items=2, min_utility=0.0),
    )
    assert set(map(int, permissive.selected_indices.tolist())) == {0, 2}
    assert int(jnp.sum(permissive.selected_mask)) == 2


def test_behavior_model_dream_policy_samples_from_learned_agent_model() -> None:
    model = BehaviorModel(BehaviorModelConfig(n_actions=2, step_size=0.1))
    state = model.init(feature_dim=3, key=jr.key(9))
    state = state.replace(  # type: ignore[attr-defined]
        weights=jnp.array(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=jnp.float32,
        )
    )
    policy = BehaviorModelDreamPolicy(model)
    sample = policy.sample_action(
        state,
        jnp.array([2.0, 0.0, 0.0], dtype=jnp.float32),
        jr.key(10),
    )

    assert int(sample.action) in {0, 1}
    chex.assert_tree_all_finite(sample.action_probability)
    chex.assert_tree_all_finite(sample.log_probability)


@pytest.mark.unit
def test_latent_world_model_evidence_level_is_development_only() -> None:
    assert EVIDENCE_LEVEL == "L0"
    assert SCIENTIFIC_PROMOTION_ALLOWED is False


@pytest.mark.unit
def test_trainable_encoder_default_off_matches_legacy_fixed_encoder_path() -> None:
    config = LatentWorldModelConfig(
        observation_dim=3,
        n_actions=2,
        latent_dim=4,
        hidden_sizes=(8,),
        sparsity=0.0,
    )
    assert config.encoder_learning is False

    legacy_payload = config.to_config()
    for key in _ENCODER_CONFIG_KEYS:
        del legacy_payload[key]
    legacy_config = LatentWorldModelConfig.from_config(legacy_payload)
    assert legacy_config == config

    model = LatentWorldModel(config)
    legacy_model = LatentWorldModel(legacy_config)
    # One shared initial state removes init wall-clock metadata differences,
    # so any trajectory difference below must come from the config flag.
    state = model.init(jr.key(7))

    observations, actions, rewards, next_observations = _toy_rotation_stream(16)
    result = run_latent_world_model_learning_loop(
        model, state, observations, actions, rewards, next_observations
    )
    legacy_result = run_latent_world_model_learning_loop(
        legacy_model, state, observations, actions, rewards, next_observations
    )

    # Bitwise-identical trajectories and final state with the flag off.
    chex.assert_trees_all_equal(result, legacy_result)
    assert bool(jnp.array_equal(result.state.encoder_matrix, state.encoder_matrix))
    assert bool(jnp.array_equal(result.state.encoder_bias, state.encoder_bias))
    assert not bool(jnp.any(result.encoder_updates_applied))
    assert not bool(jnp.any(result.encoder_collapse_gates))
    assert bool(jnp.all(result.encoder_gradient_norms == 0.0))


@pytest.mark.unit
def test_trainable_encoder_learning_smoke_loss_decreases() -> None:
    config = LatentWorldModelConfig(
        observation_dim=3,
        n_actions=2,
        latent_dim=4,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        encoder_learning=True,
        encoder_step_size=0.05,
        min_latent_std=0.0,  # collapse never detected: encoder gate stays open
    )
    model = LatentWorldModel(config)
    state = model.init(jr.key(1))
    observations, actions, rewards, next_observations = _toy_rotation_stream(300)
    result = run_latent_world_model_learning_loop(
        model, state, observations, actions, rewards, next_observations
    )

    assert int(result.state.step_count) == 300
    chex.assert_tree_all_finite(result.prediction_errors)
    chex.assert_tree_all_finite(result.state.encoder_matrix)
    chex.assert_tree_all_finite(result.state.encoder_bias)
    assert int(jnp.sum(result.encoder_updates_applied)) > 0
    assert bool(jnp.any(result.encoder_gradient_norms > 0.0))
    assert not bool(jnp.array_equal(result.state.encoder_matrix, state.encoder_matrix))

    quarter = 75
    early_loss = float(jnp.mean(result.prediction_errors[:quarter]))
    late_loss = float(jnp.mean(result.prediction_errors[-quarter:]))
    assert late_loss < early_loss


@pytest.mark.unit
def test_trainable_encoder_first_step_is_collapse_gated() -> None:
    config = LatentWorldModelConfig(
        observation_dim=3,
        n_actions=2,
        latent_dim=4,
        hidden_sizes=(),
        sparsity=0.0,
        encoder_learning=True,
    )
    model = LatentWorldModel(config)
    state = model.init(jr.key(2))

    # At step 0 the variance EMA holds no evidence, so collapse_score is 1.0
    # and the strict default threshold blocks the encoder step fail-closed.
    result = model.update(
        state,
        jnp.array([0.2, -0.1, 0.3], dtype=jnp.float32),
        jnp.array(1, dtype=jnp.int32),
        jnp.array(0.5, dtype=jnp.float32),
        jnp.array(0.95, dtype=jnp.float32),
        jnp.array([0.3, 0.0, 0.2], dtype=jnp.float32),
    )

    assert float(result.collapse_score) == 1.0
    assert bool(result.encoder_collapse_gated) is True
    assert bool(result.encoder_update_applied) is False
    assert bool(jnp.array_equal(result.state.encoder_matrix, state.encoder_matrix))
    assert bool(jnp.array_equal(result.state.encoder_bias, state.encoder_bias))


@pytest.mark.unit
def test_trainable_encoder_collapse_gate_blocks_all_updates() -> None:
    base: dict[str, object] = {
        "observation_dim": 3,
        "n_actions": 2,
        "latent_dim": 4,
        "hidden_sizes": (),
        "step_size": 0.05,
        "sparsity": 0.0,
        # tanh latents always have std far below 5.0, so every step reports
        # full collapse and the gate must block every encoder update.
        "min_latent_std": 5.0,
    }
    gated_model = LatentWorldModel(
        LatentWorldModelConfig(encoder_learning=True, **base)  # type: ignore[arg-type]
    )
    disabled_model = LatentWorldModel(
        LatentWorldModelConfig(encoder_learning=False, **base)  # type: ignore[arg-type]
    )
    # One shared initial state removes init wall-clock metadata differences.
    gated_state = gated_model.init(jr.key(3))

    observations, actions, rewards, next_observations = _toy_rotation_stream(24)
    gated_result = run_latent_world_model_learning_loop(
        gated_model, gated_state, observations, actions, rewards, next_observations
    )
    disabled_result = run_latent_world_model_learning_loop(
        disabled_model, gated_state, observations, actions, rewards, next_observations
    )

    assert bool(jnp.all(gated_result.encoder_collapse_gates))
    assert not bool(jnp.any(gated_result.encoder_updates_applied))
    assert bool(jnp.array_equal(gated_result.state.encoder_matrix, gated_state.encoder_matrix))
    assert bool(jnp.array_equal(gated_result.state.encoder_bias, gated_state.encoder_bias))
    # Predictor keeps learning while the encoder gate is closed.
    assert int(gated_result.state.step_count) == 24

    # A fully gated trainable encoder leaves the whole trajectory bitwise
    # identical to the disabled path.
    chex.assert_trees_all_equal(gated_result.state, disabled_result.state)
    chex.assert_trees_all_equal(
        gated_result.latent_predictions, disabled_result.latent_predictions
    )
    chex.assert_trees_all_equal(gated_result.surprises, disabled_result.surprises)
    chex.assert_trees_all_equal(
        gated_result.prediction_errors, disabled_result.prediction_errors
    )


@pytest.mark.unit
def test_trainable_encoder_serialization_roundtrip() -> None:
    config = LatentWorldModelConfig(
        observation_dim=2,
        n_actions=3,
        latent_dim=3,
        hidden_sizes=(8,),
        sparsity=0.0,
        encoder_learning=True,
        encoder_step_size=0.02,
        max_encoder_update=0.05,
        encoder_collapse_gate_threshold=0.25,
        min_latent_std=0.0,
    )
    assert LatentWorldModelConfig.from_config(config.to_config()) == config

    model = LatentWorldModel(config)
    restored = LatentWorldModel.from_config(model.to_config())
    assert restored.config == config

    # One shared initial state removes init wall-clock metadata differences.
    state = model.init(jr.key(5))

    args = (
        jnp.array([0.4, -0.2], dtype=jnp.float32),
        jnp.array(2, dtype=jnp.int32),
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array(0.99, dtype=jnp.float32),
        jnp.array([0.5, -0.1], dtype=jnp.float32),
    )
    result = model.update(state, *args)
    restored_result = restored.update(state, *args)
    chex.assert_trees_all_equal(result.state, restored_result.state)
    assert bool(result.encoder_update_applied) == bool(restored_result.encoder_update_applied)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"encoder_step_size": 0.0},
        {"encoder_step_size": -0.01},
        {"max_encoder_update": 0.0},
        {"max_encoder_update": -1.0},
        {"encoder_collapse_gate_threshold": -0.1},
        {"encoder_collapse_gate_threshold": 1.5},
        {"encoder_step_size": float("nan")},
        {"max_encoder_update": float("nan")},
        {"min_latent_std": float("nan")},
        {"min_latent_std": float("inf")},
        {"max_latent_delta": float("nan")},
        {"encoder_scale": float("nan")},
        {"encoder_bias_scale": float("nan")},
        {"reward_scale": float("inf")},
        {"observation_scale": (float("nan"), 1.0)},
    ],
)
def test_trainable_encoder_config_validation_fails_closed(
    overrides: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            encoder_learning=True,
            **overrides,  # type: ignore[arg-type]
        )


def test_latent_world_model_config_rejects_booleans_and_non_integers() -> None:
    with pytest.raises(ValueError, match="observation_dim"):
        LatentWorldModelConfig(observation_dim=True, n_actions=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_actions"):
        LatentWorldModelConfig(observation_dim=2, n_actions=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="latent_dim"):
        LatentWorldModelConfig(observation_dim=2, n_actions=2, latent_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_sizes"):
        LatentWorldModelConfig(observation_dim=2, n_actions=2, hidden_sizes=(True,))  # type: ignore[arg-type]


def test_latent_world_model_config_accepts_and_canonicalizes_numpy_integers() -> None:
    cfg = LatentWorldModelConfig(
        observation_dim=np.int32(4),
        n_actions=np.int64(2),
        latent_dim=np.uint16(8),
        hidden_sizes=(np.int32(32), np.int64(16)),
    )
    assert type(cfg.observation_dim) is int
    assert type(cfg.n_actions) is int
    assert type(cfg.latent_dim) is int
    assert type(cfg.hidden_sizes[0]) is int
    assert type(cfg.hidden_sizes[1]) is int
    assert cfg.observation_dim == 4
    assert cfg.n_actions == 2
    assert cfg.latent_dim == 8
    assert cfg.hidden_sizes == (32, 16)


@pytest.mark.parametrize(
    "field",
    ("predict_delta", "use_layer_norm", "include_action_interactions", "encoder_learning"),
)
def test_latent_world_model_config_requires_exact_booleans(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            **{field: 1},  # type: ignore[arg-type]
        )


def test_latent_world_model_config_requires_exact_containers_and_schema() -> None:
    with pytest.raises(ValueError, match="hidden_sizes"):
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=[8],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="observation_scale"):
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            observation_scale=[1.0, 1.0],  # type: ignore[arg-type]
        )

    payload = LatentWorldModelConfig(observation_dim=2, n_actions=2).to_config()
    assert LatentWorldModelConfig.from_config(MappingProxyType(payload)).to_config() == payload
    malformed = dict(payload)
    malformed["hidden_sizes"] = (64,)
    with pytest.raises(ValueError, match="hidden_sizes"):
        LatentWorldModelConfig.from_config(malformed)
    malformed = dict(payload)
    malformed["task_id"] = 3
    with pytest.raises(ValueError, match="fields"):
        LatentWorldModelConfig.from_config(malformed)

    model_payload = LatentWorldModel(
        LatentWorldModelConfig(observation_dim=2, n_actions=2, hidden_sizes=())
    ).to_config()
    assert LatentWorldModel.from_config(MappingProxyType(model_payload)).config.to_config() == (
        model_payload["config"]
    )
    malformed_model = dict(model_payload)
    malformed_model["task_id"] = 3
    with pytest.raises(ValueError, match="fields"):
        LatentWorldModel.from_config(malformed_model)


def test_latent_config_requires_complete_schema_except_legacy_encoder_defaults() -> None:
    payload = LatentWorldModelConfig(observation_dim=2, n_actions=2).to_config()
    missing = dict(payload)
    del missing["gamma"]
    with pytest.raises(ValueError, match="fields"):
        LatentWorldModelConfig.from_config(missing)
    wrong_type = dict(payload)
    wrong_type["type"] = type("StringSubclass", (str,), {})("LatentWorldModelConfig")
    with pytest.raises(ValueError, match="type"):
        LatentWorldModelConfig.from_config(wrong_type)


def test_latent_config_preflights_combined_direct_state_without_allocation() -> None:
    with pytest.raises(ValueError, match="combined_direct_state_bytes"):
        LatentWorldModelConfig(
            observation_dim=1,
            n_actions=1,
            latent_dim=8,
            hidden_sizes=(16_384, 16_384),
        )


def test_latent_config_mapping_gate_does_not_consult_spoofed_class() -> None:
    class SpoofedMapping:
        @property
        def __class__(self) -> type:
            raise AssertionError("__class__ hook must not run")

    with pytest.raises(ValueError, match="actual mapping"):
        LatentWorldModelConfig.from_config(SpoofedMapping())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"observation_dim": 50_000, "latent_dim": 50_000}, "encoder_matrix"),
        ({"observation_dim": 1, "latent_dim": 2**31 - 2}, "n_heads"),
        (
            {"observation_dim": 1, "latent_dim": 2**31 - 3, "n_actions": 3},
            "input_dim",
        ),
        (
            {
                "observation_dim": 1,
                "latent_dim": 50_000,
                "n_actions": 50_000,
                "include_action_interactions": True,
            },
            "action_interaction",
        ),
        ({"observation_dim": 1, "n_actions": 1, "hidden_sizes": (2**31 - 1,)}, "hidden_layer"),
        ({"observation_dim": 1, "n_actions": 1, "hidden_sizes": (50_000, 50_000)}, "hidden_layer"),
        (
            {
                "observation_dim": 1,
                "n_actions": 1,
                "latent_dim": 50_000,
                "hidden_sizes": (1, 50_000),
            },
            "head_weight",
        ),
    ],
)
def test_latent_world_model_config_rejects_derived_allocation_overflow(
    overrides: dict[str, object], message: str
) -> None:
    base: dict[str, object] = {"observation_dim": 2, "n_actions": 2}
    base.update(overrides)
    with pytest.raises(ValueError, match=message):
        LatentWorldModelConfig(**base)  # type: ignore[arg-type]


def test_latent_world_model_init_rejects_nonfinite_encoder_draw() -> None:
    model = LatentWorldModel(
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            latent_dim=2,
            hidden_sizes=(),
            encoder_scale=np.finfo(np.float32).max,
        )
    )

    with pytest.raises(ValueError, match="encoder initialization"):
        model.init(jr.key(0))


def test_latent_world_model_observation_scale_normalizes_without_floor() -> None:
    """Observation scale below 1e-6 normalizes by exact configured scale."""
    key = jr.key(42)
    latents = []
    for scale in (1.0, 1e-6, 1e-9, 1e-20):
        config = LatentWorldModelConfig(
            observation_dim=2,
            latent_dim=4,
            n_actions=2,
            observation_scale=(scale, scale),
        )
        model = LatentWorldModel(config)
        state = model.init(key)
        obs = jnp.asarray([1.5 * scale, -0.75 * scale], dtype=jnp.float32)
        latents.append(np.asarray(model.encode(state, obs)))

    for i in range(1, len(latents)):
        np.testing.assert_allclose(latents[i], latents[0], atol=1e-6)

    # Subnormal scale whose float32 reciprocal overflows to inf must be rejected
    subnormal = float(np.finfo(np.float32).smallest_subnormal)
    with pytest.raises(ValueError, match="observation_scale"):
        LatentWorldModelConfig(
            observation_dim=1,
            latent_dim=2,
            n_actions=2,
            observation_scale=(subnormal,),
        )


@pytest.mark.unit
def test_absolute_next_latent_is_not_bounded_by_the_delta_clip() -> None:
    """``max_latent_delta`` bounds a latent delta; it must not truncate absolute latents.

    With ``predict_delta=False`` the training target built by ``targets`` is
    ``encode(next_observation)`` with no ``max_latent_delta`` bound applied.
    Clipping the same head to +/-``max_latent_delta`` at prediction time makes
    the model structurally unable to return what it was trained to produce
    whenever the encoded latent exceeds that bound, and reports a permanent
    surprise on a transition the head has learned exactly.
    """
    config = LatentWorldModelConfig(
        observation_dim=3,
        n_actions=2,
        latent_dim=4,
        hidden_sizes=(),
        step_size=0.2,
        sparsity=0.0,
        use_layer_norm=False,
        encoder_scale=3.0,
        predict_delta=False,
        max_latent_delta=0.5,
    )
    model = LatentWorldModel(config)
    state = model.init(jr.key(0))
    observation = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32)
    next_observation = jnp.array([0.9, -0.8, 0.7], dtype=jnp.float32)
    action = jnp.array(1, dtype=jnp.int32)
    reward = jnp.asarray(0.0, dtype=jnp.float32)
    discount = jnp.asarray(0.99, dtype=jnp.float32)

    target = np.asarray(model.encode(state, next_observation))
    # The fixed tanh encoder must actually place the target outside the clip.
    assert float(np.max(np.abs(target))) > config.max_latent_delta

    for _ in range(600):
        result = model.update(state, observation, action, reward, discount, next_observation)
        state = result.state

    prediction = model.predict(state, observation, action)
    learned = np.asarray(prediction.raw_predictions)[: config.latent_dim]
    next_latent = np.asarray(prediction.next_latent)

    # The head really did learn the absolute target, so any gap is a decode bug.
    np.testing.assert_allclose(learned, target, atol=1e-3)
    np.testing.assert_allclose(next_latent, learned, atol=1e-6)
    assert float(result.surprise) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.unit
def test_delta_next_latent_remains_bounded_by_the_delta_clip() -> None:
    """In delta mode the predicted step stays elementwise within ``max_latent_delta``."""
    config = LatentWorldModelConfig(
        observation_dim=3,
        n_actions=2,
        latent_dim=4,
        hidden_sizes=(),
        step_size=0.2,
        sparsity=0.0,
        use_layer_norm=False,
        encoder_scale=3.0,
        predict_delta=True,
        max_latent_delta=0.5,
    )
    model = LatentWorldModel(config)
    state = model.init(jr.key(0))
    observation = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32)
    next_observation = jnp.array([-0.9, 0.8, -0.7], dtype=jnp.float32)
    action = jnp.array(1, dtype=jnp.int32)
    reward = jnp.asarray(0.0, dtype=jnp.float32)
    discount = jnp.asarray(0.99, dtype=jnp.float32)
    for _ in range(200):
        state = model.update(
            state, observation, action, reward, discount, next_observation
        ).state

    prediction = model.predict(state, observation, action)
    latent = np.asarray(prediction.latent)
    raw = np.asarray(prediction.raw_predictions)[: config.latent_dim]
    step = np.asarray(prediction.next_latent) - latent
    assert float(np.max(np.abs(raw))) > config.max_latent_delta
    assert float(np.max(np.abs(step))) <= config.max_latent_delta + 1e-6
