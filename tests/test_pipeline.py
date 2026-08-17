# mypy: disable-error-code="no-untyped-def"
"""Production Step 1-4 pipeline tests."""

import json

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework as af
from alberta_framework.pipeline import (
    AlbertaPipeline,
    AlbertaPipelineConfig,
    AlbertaPipelineSmokeResult,
    HordeActorCriticPipelineConfig,
    Step2AssociativePipelineConfig,
    Step2FeatureConfig,
    Step2UPGDConfig,
    make_alberta_pipeline,
    observation_channel_cumulant_fn,
    run_pipeline_smoke,
)
from alberta_framework.steps import (
    Step3HordeConfig,
    Step4SARSAConfig,
    run_step3_smoke,
    run_step4_smoke,
)


def _small_pipeline_config() -> AlbertaPipelineConfig:
    return AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=3),
        horde=Step3HordeConfig(
            gammas=(0.0, 0.5),
            lamdas=(0.0, 0.0),
            hidden_sizes=(),
            step_size=0.05,
            use_obgd=True,
            obgd_kappa=1.0,
        ),
        control=Step4SARSAConfig(
            n_actions=2,
            hidden_sizes=(),
            epsilon_start=0.0,
            epsilon_end=0.0,
            step_size=0.05,
            bounder_kappa=1.0,
        ),
    )


def _small_upgd_config() -> AlbertaPipelineConfig:
    return AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=3),
        upgd=Step2UPGDConfig(
            observation_dim=3,
            n_heads=1,
            hidden_sizes=(8,),
            step_size=0.03,
        ),
        horde=Step3HordeConfig(
            gammas=(0.0, 0.5),
            lamdas=(0.0, 0.0),
            hidden_sizes=(),
            step_size=0.05,
            use_obgd=True,
            obgd_kappa=1.0,
        ),
        control=Step4SARSAConfig(
            n_actions=2,
            hidden_sizes=(),
            epsilon_start=0.0,
            epsilon_end=0.0,
            step_size=0.05,
            bounder_kappa=1.0,
        ),
        step2="upgd",
    )


def _small_horde_ac_config() -> AlbertaPipelineConfig:
    return AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=3),
        horde=Step3HordeConfig(
            gammas=(0.95, 0.5),
            lamdas=(0.0, 0.0),
            hidden_sizes=(),
            step_size=0.05,
            use_obgd=True,
            obgd_kappa=1.0,
        ),
        control=Step4SARSAConfig(
            n_actions=2,
            hidden_sizes=(),
            epsilon_start=0.0,
            epsilon_end=0.0,
            step_size=0.05,
            bounder_kappa=1.0,
        ),
        horde_ac=HordeActorCriticPipelineConfig(
            n_actions=2,
            actor_step_size=0.02,
            actor_lamda=0.0,
            value_head_index=0,
        ),
        control_mode="horde_ac",
    )


def _small_associative_config() -> AlbertaPipelineConfig:
    return AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=5),
        associative=Step2AssociativePipelineConfig(
            vocab_size=8,
            block_size=5,
            suffix_length=3,
            max_features=128,
            adaptive_feature_family=True,
            adaptive_window=True,
            adaptive_budget=True,
            initial_budget_fraction=0.5,
        ),
        horde=Step3HordeConfig(
            gammas=(0.0,),
            lamdas=(0.0,),
            hidden_sizes=(),
            step_size=0.05,
        ),
        control=Step4SARSAConfig(
            n_actions=2,
            hidden_sizes=(),
            epsilon_start=0.0,
            epsilon_end=0.0,
            step_size=0.05,
        ),
        step2="associative",
    )


def test_pipeline_config_roundtrip_is_json_serializable() -> None:
    config = _small_pipeline_config()
    payload = config.to_dict()
    encoded = json.dumps(payload)
    roundtrip = AlbertaPipelineConfig.from_dict(json.loads(encoded))

    assert roundtrip == config
    assert roundtrip.feature_dim() == 3
    assert af.AlbertaPipeline is AlbertaPipeline


def test_pipeline_config_roundtrip_with_upgd_and_horde_ac() -> None:
    config = AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=3),
        upgd=Step2UPGDConfig(observation_dim=3, n_heads=1, hidden_sizes=(8,)),
        horde=Step3HordeConfig(
            gammas=(0.95, 0.5),
            lamdas=(0.0, 0.0),
            hidden_sizes=(),
            step_size=0.05,
        ),
        control=Step4SARSAConfig(n_actions=2, hidden_sizes=()),
        horde_ac=HordeActorCriticPipelineConfig(n_actions=2, value_head_index=0),
        step2="upgd",
        control_mode="horde_ac",
    )
    payload = config.to_dict()
    roundtrip = AlbertaPipelineConfig.from_dict(json.loads(json.dumps(payload)))
    assert roundtrip == config
    assert roundtrip.feature_dim() == 8


def test_pipeline_config_roundtrip_with_associative_step2() -> None:
    config = _small_associative_config()
    payload = config.to_dict()
    roundtrip = AlbertaPipelineConfig.from_dict(json.loads(json.dumps(payload)))

    assert roundtrip == config
    assert roundtrip.feature_dim() == 8
    assert roundtrip.associative is not None
    assert roundtrip.associative.adaptive_feature_family
    assert roundtrip.associative.adaptive_window
    assert roundtrip.associative.adaptive_budget


def test_step3_and_step4_facade_smokes_are_finite() -> None:
    step3_config = Step3HordeConfig(
        gammas=(0.0, 0.5),
        lamdas=(0.0, 0.0),
        hidden_sizes=(),
        step_size=0.05,
    )
    step3_result = run_step3_smoke(
        step3_config,
        steps=8,
        final_window=2,
        raw_feature_dim=3,
        constructed_feature_dim=2,
    )
    assert step3_result.finite
    assert step3_result.per_demon_metrics_shape == (8, 2, 3)
    assert step3_result.handoff.feature_dim == 5

    step4_config = Step4SARSAConfig(
        n_actions=2,
        hidden_sizes=(),
        epsilon_start=0.0,
        epsilon_end=0.0,
        step_size=0.05,
    )
    step4_result = run_step4_smoke(step4_config, steps=8, feature_dim=3)
    assert step4_result.finite
    assert step4_result.q_values_shape == (8, 2)
    assert step4_result.actions_shape == (8,)


def test_pipeline_init_predict_and_one_step_update_are_finite() -> None:
    config = _small_pipeline_config()
    pipeline = make_alberta_pipeline(config)
    initial_observation = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
    state = pipeline.init(jr.key(0), initial_observation)

    horde_predictions, q_values = pipeline.predict(state)
    chex.assert_shape(horde_predictions, (2,))
    chex.assert_shape(q_values, (2,))
    chex.assert_tree_all_finite((horde_predictions, q_values))

    result = pipeline.update(
        state,
        jnp.asarray([0.1, 0.3, -0.2], dtype=jnp.float32),
        jnp.asarray(0.25, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([0.3, -0.2], dtype=jnp.float32),
    )

    assert int(result.state.step_count) == 1
    chex.assert_shape(result.features, (3,))
    chex.assert_shape(result.horde_predictions, (2,))
    chex.assert_shape(result.q_values, (2,))
    chex.assert_tree_all_finite(
        (
            result.features,
            result.horde_predictions,
            result.horde_td_errors,
            result.q_values,
            result.control_td_error,
        )
    )
    assert 0 <= int(result.action) < config.control.n_actions


def test_pipeline_sarsa_control_contains_step3_prediction_demons() -> None:
    """SARSA control mirrors Step 3 GVFs as prediction demons."""
    config = _small_pipeline_config()
    pipeline = make_alberta_pipeline(config)
    assert pipeline.config.control_mode == "sarsa"
    assert pipeline.control.horde.n_demons == (
        config.control.n_actions + config.horde.n_demons
    )
    assert pipeline.control.horde.horde_spec.demons[config.control.n_actions].name == (
        "gvf_0"
    )

    initial_observation = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
    state = pipeline.init(jr.key(0), initial_observation)
    prediction_head_index = config.control.n_actions
    old_prediction_head = (
        state.control_state.learner_state.head_params.weights[prediction_head_index]
    )

    result = pipeline.update(
        state,
        jnp.asarray([0.1, 0.3, -0.2], dtype=jnp.float32),
        jnp.asarray(0.25, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([1.0, -0.5], dtype=jnp.float32),
    )

    new_prediction_head = (
        result.state.control_state.learner_state.head_params.weights[
            prediction_head_index
        ]
    )
    assert not jnp.allclose(old_prediction_head, new_prediction_head)


def test_pipeline_scan_smoke_is_finite() -> None:
    config = _small_pipeline_config()
    result = run_pipeline_smoke(config, steps=8, seed=3)

    assert result.finite
    assert result.feature_shape == (8, 3)
    assert result.horde_predictions_shape == (8, 2)
    assert result.q_values_shape == (8, 2)
    assert result.actions_shape == (8,)
    assert result.to_dict()["config"] == config.to_dict()


def test_pipeline_with_upgd_step2_smoke() -> None:
    """UPGD-backed Step 2 produces finite features that drive Step 3 and Step 4."""
    config = _small_upgd_config()
    pipeline = make_alberta_pipeline(config)
    assert pipeline.upgd is not None

    initial_observation = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
    state = pipeline.init(jr.key(7), initial_observation)
    chex.assert_shape(state.last_features, (8,))
    assert state.upgd_state is not None

    horde_predictions, q_values = pipeline.predict(state)
    chex.assert_shape(horde_predictions, (2,))
    chex.assert_shape(q_values, (2,))
    chex.assert_tree_all_finite((horde_predictions, q_values))

    result = pipeline.update(
        state,
        jnp.asarray([0.1, 0.3, -0.2], dtype=jnp.float32),
        jnp.asarray(0.25, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([0.3, -0.2], dtype=jnp.float32),
        upgd_targets=jnp.asarray([0.5], dtype=jnp.float32),
    )
    assert int(result.state.step_count) == 1
    chex.assert_shape(result.features, (8,))
    chex.assert_tree_all_finite(
        (result.features, result.horde_predictions, result.q_values)
    )
    smoke = run_pipeline_smoke(config, steps=4, seed=11)
    assert smoke.finite
    assert smoke.feature_shape == (4, 8)


def test_pipeline_upgd_config_is_honored() -> None:
    """UPGD-backed pipeline forwards supported learner config fields."""
    config = AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=3),
        upgd=Step2UPGDConfig(
            observation_dim=3,
            n_heads=2,
            hidden_sizes=(8,),
            step_size=0.02,
            sparsity=0.25,
            use_layer_norm=False,
            loss_normalization="target_density",
            readout_mode="softmax_ce",
        ),
        horde=Step3HordeConfig(gammas=(0.0,), lamdas=(0.0,), hidden_sizes=()),
        control=Step4SARSAConfig(n_actions=2, hidden_sizes=()),
        step2="upgd",
    )
    pipeline = make_alberta_pipeline(config)
    assert pipeline.upgd is not None

    upgd_config = pipeline.upgd.to_config()
    assert upgd_config["step_size"] == 0.02
    assert upgd_config["sparsity"] == 0.25
    assert upgd_config["use_layer_norm"] is False
    assert upgd_config["loss_normalization"] == "target_density"
    assert upgd_config["readout_mode"] == "softmax_ce"


def test_pipeline_upgd_strict_digit_readout_preset() -> None:
    config = AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=64),
        upgd=Step2UPGDConfig.strict_digit_readout(
            observation_dim=64,
            n_heads=10,
            hidden_sizes=(16, 16),
            step_size=0.018,
        ),
        horde=Step3HordeConfig(gammas=(0.0,), lamdas=(0.0,), hidden_sizes=()),
        control=Step4SARSAConfig(n_actions=2, hidden_sizes=()),
        step2="upgd",
    )
    pipeline = make_alberta_pipeline(config)
    assert pipeline.upgd is not None

    upgd_config = pipeline.upgd.to_config()
    assert upgd_config["hidden_sizes"] == [16, 16]
    assert upgd_config["readout_mode"] == "two_timescale_simplex"
    assert upgd_config["readout_fast_head_bounder_mode"] == "separate"
    assert upgd_config["adaptive_kappa_mode"] == "loss_ratio"


def test_pipeline_with_horde_ac_control_smoke() -> None:
    """Horde actor-critic control returns sensible policies and updates."""
    config = _small_horde_ac_config()
    pipeline = make_alberta_pipeline(config)
    assert pipeline.config.control_mode == "horde_ac"

    initial_observation = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
    state = pipeline.init(jr.key(0), initial_observation)
    ac_state = state.control_state
    assert hasattr(ac_state, "critic_state")
    chex.assert_trees_all_close(state.horde_state, ac_state.critic_state)

    horde_predictions, policy = pipeline.predict(state)
    chex.assert_shape(horde_predictions, (2,))
    chex.assert_shape(policy, (2,))
    chex.assert_tree_all_finite((horde_predictions, policy))
    assert float(jnp.abs(jnp.sum(policy) - 1.0)) < 1e-4

    result = pipeline.update(
        state,
        jnp.asarray([0.1, 0.3, -0.2], dtype=jnp.float32),
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([0.3, -0.2], dtype=jnp.float32),
    )
    assert int(result.state.step_count) == 1
    chex.assert_shape(result.q_values, (2,))
    next_ac_state = result.state.control_state
    assert hasattr(next_ac_state, "critic_state")
    chex.assert_trees_all_close(
        result.state.horde_state,
        next_ac_state.critic_state,
    )
    chex.assert_tree_all_finite(
        (result.features, result.horde_predictions, result.q_values)
    )
    assert config.horde_ac is not None
    assert 0 <= int(result.action) < config.horde_ac.n_actions

    smoke = run_pipeline_smoke(config, steps=4, seed=2)
    assert smoke.finite
    assert smoke.q_values_shape == (4, 2)


def test_pipeline_with_associative_step2_smoke() -> None:
    """Associative Step 2 exposes finite probability features and updates."""
    config = _small_associative_config()
    pipeline = make_alberta_pipeline(config)
    assert pipeline.associative is not None
    assert pipeline.associative.config.adaptive_feature_family
    assert pipeline.associative.config.adaptive_window
    assert pipeline.associative.config.adaptive_budget

    initial_observation = jnp.asarray([1, 2, 3, 4, 5], dtype=jnp.int32)
    state = pipeline.init(jr.key(0), initial_observation)
    chex.assert_shape(state.last_features, (8,))
    assert state.associative_state is not None

    result = pipeline.update(
        state,
        jnp.asarray([1, 2, 3, 4, 5], dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        associative_label=jnp.asarray(6, dtype=jnp.int32),
    )
    assert int(result.state.step_count) == 1
    chex.assert_shape(result.features, (8,))
    chex.assert_tree_all_finite(
        (result.features, result.horde_predictions, result.q_values)
    )

    smoke = run_pipeline_smoke(config, steps=4, seed=3)
    assert smoke.finite
    assert smoke.feature_shape == (4, 8)


def test_pipeline_behavioral_learns() -> None:
    """A 2000-step run on a fixed-target stream should reduce final-window MSE.

    The temporal-context Step 3 path tracks a single deterministic cumulant
    derived from the first observation channel. Final-window MSE is required
    to be strictly lower than initial-window MSE: a real learning signal.
    """
    config = AlbertaPipelineConfig(
        features=Step2FeatureConfig.identity(observation_dim=3),
        horde=Step3HordeConfig(
            gammas=(0.0,),
            lamdas=(0.0,),
            hidden_sizes=(),
            step_size=0.1,
            use_obgd=True,
            obgd_kappa=1.0,
        ),
        control=Step4SARSAConfig(
            n_actions=2,
            hidden_sizes=(),
            epsilon_start=0.0,
            epsilon_end=0.0,
            step_size=0.05,
            bounder_kappa=1.0,
        ),
    )
    pipeline = make_alberta_pipeline(config)

    n_steps = 2000
    key = jr.key(0)
    obs_key, _ = jr.split(key)
    observations = jr.normal(obs_key, (n_steps + 1, 3), dtype=jnp.float32)
    # Cumulant: the first channel of the next observation. The Horde must
    # learn to track this from the previous observation.
    cumulants = observations[1:, :1]
    rewards = jnp.zeros(n_steps, dtype=jnp.float32)
    terminated = jnp.zeros(n_steps, dtype=jnp.float32)

    state = pipeline.init(jr.key(99), observations[0])
    result = pipeline.run_arrays(
        state, observations[1:], rewards, terminated, cumulants
    )
    # Per-step squared error between the (single) demon's TD target and prediction.
    sq_err = jnp.square(
        result.horde_predictions[:, 0] - cumulants[:, 0]
    )
    initial_mse = float(jnp.mean(sq_err[:200]))
    final_mse = float(jnp.mean(sq_err[-200:]))

    assert jnp.isfinite(initial_mse)
    assert jnp.isfinite(final_mse)
    assert final_mse < initial_mse, (
        f"final-window MSE ({final_mse:.4f}) should be lower than "
        f"initial-window MSE ({initial_mse:.4f})"
    )


def test_pipeline_cumulant_fn_overrides_default() -> None:
    """Caller-provided cumulant_fn is used instead of the default channel map."""
    sentinel = jnp.array([0.123, 0.456], dtype=jnp.float32)

    def cumulant_fn(_obs, _reward, _terminated):
        return sentinel

    config = _small_pipeline_config()
    pipeline = make_alberta_pipeline(config, cumulant_fn=cumulant_fn)
    state = pipeline.init(
        jr.key(0), jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
    )
    result = pipeline.update(
        state,
        jnp.asarray([0.1, 0.3, -0.2], dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    # With gamma=0 in our test config, td_target ≈ cumulant. Verify the demon
    # 0 target equals the sentinel.
    chex.assert_trees_all_close(
        result.horde_td_targets[0], sentinel[0], atol=1e-5
    )


def test_observation_channel_cumulant_fn_wraps_channels() -> None:
    """Default cumulants are deterministic next-observation channel signals."""
    cumulant_fn = observation_channel_cumulant_fn(n_demons=5, observation_dim=3)

    cumulants = cumulant_fn(
        jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
        jnp.asarray(99.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )

    chex.assert_trees_all_close(
        cumulants,
        jnp.asarray([1.0, 2.0, 3.0, 1.0, 2.0], dtype=jnp.float32),
    )


def test_observation_channel_cumulant_fn_rejects_invalid_shapes() -> None:
    """Invalid default cumulant dimensions fail at construction time."""
    with pytest.raises(ValueError, match="n_demons must be positive"):
        observation_channel_cumulant_fn(n_demons=0, observation_dim=3)

    with pytest.raises(ValueError, match="observation_dim must be positive"):
        observation_channel_cumulant_fn(n_demons=1, observation_dim=0)


@pytest.mark.parametrize(
    ("n_demons", "observation_dim"),
    [
        (True, 3),
        (1.5, 3),
        ("3", 3),
        (1, True),
        (1, 1.5),
        (1, "3"),
    ],
)
def test_observation_channel_cumulant_fn_rejects_non_integer_identities(
    n_demons: object,
    observation_dim: object,
) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        observation_channel_cumulant_fn(
            n_demons=n_demons,  # type: ignore[arg-type]
            observation_dim=observation_dim,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field", ["n_demons", "observation_dim"])
def test_observation_channel_cumulant_fn_rejects_int32_overflow(field: str) -> None:
    kwargs = {"n_demons": 5, "observation_dim": 3, field: 2**31}
    with pytest.raises(ValueError, match="must be <="):
        observation_channel_cumulant_fn(**kwargs)


_INVALID_PIPELINE_FEATURE_FIELDS: tuple[tuple[str, object], ...] = (
    ("observation_dim", 0),
    ("observation_dim", -1),
    ("observation_dim", 2**31),
    ("observation_dim", True),
    ("observation_dim", "4"),
    ("ema_decay", -0.1),
    ("ema_decay", 1.0),
    ("ema_decay", 1.1),
    ("ema_decay", 1e100),
    ("ema_decay", float("nan")),
    ("ema_decay", True),
    ("ema_decay", "0.95"),
    ("periods", (0.0,)),
    ("periods", (-1.0,)),
    ("periods", (1e100,)),
    ("periods", (float("nan"),)),
    ("periods", (True,)),
    ("periods", [32.0]),
)


@pytest.mark.parametrize(("field", "value"), _INVALID_PIPELINE_FEATURE_FIELDS)
def test_pipeline_feature_fields_reject_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        Step2FeatureConfig(**{field: value})


_INVALID_PIPELINE_UPGD_FIELDS: tuple[tuple[str, object], ...] = (
    ("observation_dim", 0),
    ("observation_dim", 2**31),
    ("observation_dim", True),
    ("n_heads", 0),
    ("n_heads", 2**31),
    ("n_heads", True),
    ("hidden_sizes", ()),
    ("hidden_sizes", (0,)),
    ("hidden_sizes", (2**31,)),
    ("hidden_sizes", (True,)),
    ("step_size", -0.01),
    ("step_size", 1e100),
    ("step_size", float("nan")),
    ("step_size", True),
    ("sparsity", -0.1),
    ("sparsity", 1.1),
    ("sparsity", 1e100),
    ("sparsity", float("nan")),
    ("sparsity", True),
    ("use_layer_norm", 1),
    ("learner_preset", "unknown_preset"),
    ("loss_normalization", "unknown_norm"),
    ("readout_mode", "unknown_mode"),
)


@pytest.mark.parametrize(("field", "value"), _INVALID_PIPELINE_UPGD_FIELDS)
def test_pipeline_upgd_fields_reject_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        Step2UPGDConfig(**{field: value})


_INVALID_PIPELINE_ASSOCIATIVE_FIELDS: tuple[tuple[str, object], ...] = (
    ("vocab_size", 1),
    ("vocab_size", 2**31),
    ("vocab_size", True),
    ("block_size", 0),
    ("block_size", 2**31),
    ("block_size", True),
    ("suffix_length", 1),
    ("suffix_length", 9),
    ("suffix_length", True),
    ("max_features", 0),
    ("max_features", 2**31),
    ("max_features", True),
    ("write_lr", -0.1),
    ("write_lr", 0.0),
    ("write_lr", 1e100),
    ("write_lr", True),
    ("retention", -0.1),
    ("retention", 1.1),
    ("retention", 1e100),
    ("retention", True),
    ("utility_lr", -0.1),
    ("utility_lr", 1e100),
    ("utility_lr", True),
    ("utility_decay", -0.1),
    ("utility_decay", 1.1),
    ("utility_decay", 1e100),
    ("utility_decay", True),
    ("min_weight", -0.01),
    ("min_weight", 0.0),
    ("min_weight", 1e100),
    ("min_weight", True),
    ("max_weight", 0.0),
    ("max_weight", -1.0),
    ("max_weight", 1e100),
    ("max_weight", True),
    ("logit_scale", 0.0),
    ("logit_scale", -1.0),
    ("logit_scale", 1e100),
    ("logit_scale", True),
    ("normalize_by_weight", 1),
    ("adaptive_feature_family", 1),
    ("adaptive_window", 1),
    ("adaptive_budget", 1),
    ("scope_lr", -0.1),
    ("scope_lr", 1e100),
    ("scope_lr", True),
    ("budget_lr", -0.1),
    ("budget_lr", 1e100),
    ("budget_lr", True),
    ("initial_budget_fraction", 0.0),
    ("initial_budget_fraction", -0.1),
    ("initial_budget_fraction", 1.1),
    ("initial_budget_fraction", 1e100),
    ("initial_budget_fraction", True),
    ("min_effective_budget", 0),
    ("min_effective_budget", 513),
    ("min_effective_budget", True),
    ("scope_logit_clip", 0.0),
    ("scope_logit_clip", -1.0),
    ("scope_logit_clip", 1e100),
    ("scope_logit_clip", True),
)


@pytest.mark.parametrize(("field", "value"), _INVALID_PIPELINE_ASSOCIATIVE_FIELDS)
def test_pipeline_associative_fields_reject_invalid_inputs(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        Step2AssociativePipelineConfig(**{field: value})


_INVALID_PIPELINE_HORDE_AC_FIELDS: tuple[tuple[str, object], ...] = (
    ("n_actions", 0),
    ("n_actions", 2**31),
    ("n_actions", True),
    ("actor_step_size", -0.01),
    ("actor_step_size", 1e100),
    ("actor_step_size", True),
    ("actor_lamda", -0.1),
    ("actor_lamda", 1.1),
    ("actor_lamda", 1e100),
    ("actor_lamda", True),
    ("temperature", 0.0),
    ("temperature", -1.0),
    ("temperature", 1e100),
    ("temperature", True),
    ("value_head_index", -1),
    ("value_head_index", 2**31),
    ("value_head_index", True),
    ("actor_obgd_kappa", 0.0),
    ("actor_obgd_kappa", -1.0),
    ("actor_obgd_kappa", 1e100),
    ("actor_obgd_kappa", True),
)


@pytest.mark.parametrize(("field", "value"), _INVALID_PIPELINE_HORDE_AC_FIELDS)
def test_pipeline_horde_ac_fields_reject_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        HordeActorCriticPipelineConfig(**{field: value})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"steps": 0}, "steps"),
        ({"steps": -1}, "steps"),
        ({"steps": 2**31}, "steps"),
        ({"steps": True}, "steps"),
        ({"steps": "24"}, "steps"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"seed": True}, "seed"),
    ],
)
def test_pipeline_smoke_rejects_invalid_inputs(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        run_pipeline_smoke(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [2**31, 2**32 - 1])
def test_pipeline_smoke_accepts_full_uint32_seed_domain(seed: int) -> None:
    result = run_pipeline_smoke(steps=2, seed=seed)
    assert result.seed == seed


def test_pipeline_associative_requires_ordered_weight_bounds() -> None:
    with pytest.raises(ValueError, match="max_weight must be >= min_weight"):
        Step2AssociativePipelineConfig(min_weight=2.0, max_weight=1.0)

    config = Step2AssociativePipelineConfig(min_weight=1.0, max_weight=1.0)
    assert config.min_weight == config.max_weight == 1.0


@pytest.mark.parametrize(
    "ratio",
    [
        pytest.param((-1, 1), id="negative-ratio"),
        pytest.param((2, 1), id="above-unit-ratio"),
        pytest.param((-1, 2**200), id="negative-rounds-to-negative-zero"),
        pytest.param((2**200 + 1, 2**200), id="above-one-rounds-to-one"),
    ],
)
def test_pipeline_unit_interval_rejects_adversarial_ratio_floats(
    ratio: tuple[int, int]
) -> None:
    class HiddenBoundaryFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return ratio

    with pytest.raises(ValueError, match=r"sparsity must be in \[0, 1\]"):
        Step2UPGDConfig(sparsity=HiddenBoundaryFloat(0.5))


def test_pipeline_nonnegative_rejects_adversarial_negative_ratio() -> None:
    class HiddenNegativeFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return (-1, 1)

    with pytest.raises(ValueError, match=r"step_size must be non-negative"):
        Step2UPGDConfig(step_size=HiddenNegativeFloat(0.5))


def test_pipeline_rejects_class_property_spoofing_float() -> None:
    class ClassSpoof:
        @property
        def __class__(self) -> type[float]:
            return float

        def as_integer_ratio(self) -> tuple[int, int]:
            return (1, 2)

    value = ClassSpoof()
    with pytest.raises(ValueError, match="must be a real number"):
        Step2UPGDConfig(step_size=value)  # type: ignore[arg-type]


def test_pipeline_require_int_rejects_lying_int_subclass() -> None:
    """int subclasses are rejected before their __int__/__index__ hooks run."""

    class LieInt(int):
        def __int__(self) -> int:
            return 4

        def __index__(self) -> int:
            return 4

    with pytest.raises(ValueError, match="observation_dim must be an integer"):
        Step2UPGDConfig(observation_dim=LieInt(-1))


def test_pipeline_accepts_numpy_integers_and_stores_builtin_int() -> None:
    config = Step2UPGDConfig(observation_dim=np.int32(3), n_heads=np.int64(2))
    assert config.observation_dim == 3
    assert type(config.observation_dim) is int
    assert type(config.n_heads) is int
    json.dumps(config.to_dict())


class _SpoofedBool:
    @property
    def __class__(self) -> type[bool]:  # type: ignore[override]
        return bool

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "SpoofedBool()"


def test_pipeline_bool_flags_reject_class_spoofing() -> None:
    with pytest.raises(ValueError, match="use_layer_norm must be a bool"):
        Step2UPGDConfig(use_layer_norm=_SpoofedBool())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="normalize_by_weight must be a bool"):
        Step2AssociativePipelineConfig(
            normalize_by_weight=_SpoofedBool()  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="include_raw must be a bool"):
        Step2FeatureConfig(include_raw=_SpoofedBool())  # type: ignore[arg-type]


class _EqualsString:
    """Non-str object that compares equal to one target string."""

    def __init__(self, target: str) -> None:
        self._target = target

    def __eq__(self, other: object) -> bool:
        return other == self._target

    def __hash__(self) -> int:
        return hash(self._target)


def test_pipeline_string_discriminators_require_actual_str() -> None:
    with pytest.raises(ValueError, match="unknown learner_preset"):
        Step2UPGDConfig(
            learner_preset=_EqualsString("default")  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unknown loss_normalization"):
        Step2UPGDConfig(
            loss_normalization=_EqualsString("target_structure")  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unknown readout_mode"):
        Step2UPGDConfig(
            readout_mode=_EqualsString("linear_mse")  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unknown feature_family"):
        Step2AssociativePipelineConfig(
            feature_family=_EqualsString("token_suffix_pair")  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unknown step2 mode"):
        AlbertaPipelineConfig(
            step2=_EqualsString("temporal_context")  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unknown control_mode"):
        AlbertaPipelineConfig(
            control_mode=_EqualsString("sarsa")  # type: ignore[arg-type]
        )


def test_pipeline_associative_rejects_unknown_feature_family() -> None:
    with pytest.raises(ValueError, match="unknown feature_family"):
        Step2AssociativePipelineConfig(
            feature_family="unknown_family"  # type: ignore[arg-type]
        )


def test_pipeline_full_config_json_roundtrip() -> None:
    config = AlbertaPipelineConfig(
        upgd=Step2UPGDConfig(),
        associative=Step2AssociativePipelineConfig(),
        horde_ac=HordeActorCriticPipelineConfig(),
        step2="upgd",
        control_mode="horde_ac",
    )
    payload = json.loads(json.dumps(config.to_dict()))
    assert AlbertaPipelineConfig.from_dict(payload) == config


def _legal_pipeline_smoke_result(**overrides: object) -> AlbertaPipelineSmokeResult:
    payload: dict[str, object] = {
        "config": _small_pipeline_config(),
        "steps": 8,
        "seed": 0,
        "feature_shape": (8, 3),
        "horde_predictions_shape": (8, 2),
        "q_values_shape": (8, 2),
        "actions_shape": (8,),
        "finite": True,
    }
    payload.update(overrides)
    return AlbertaPipelineSmokeResult(**payload)  # type: ignore[arg-type]


def test_pipeline_smoke_result_rejects_leftover_identities() -> None:
    """Public pipeline smoke records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="steps"):
        _legal_pipeline_smoke_result(steps=True)
    with pytest.raises(ValueError, match="steps"):
        _legal_pipeline_smoke_result(steps=float("nan"))
    with pytest.raises(ValueError, match="seed"):
        _legal_pipeline_smoke_result(seed=True)
    with pytest.raises(ValueError, match="finite"):
        _legal_pipeline_smoke_result(finite=1)

    legal = _legal_pipeline_smoke_result()
    dumped = json.dumps(
        {
            "steps": legal.steps,
            "seed": legal.seed,
            "finite": legal.finite,
        },
        allow_nan=False,
    )
    assert '"steps": 8' in dumped
    assert '"seed": 0' in dumped
    assert '"finite": true' in dumped
    assert '"steps": true' not in dumped
    assert '"seed": true' not in dumped
    assert '"finite": 1' not in dumped


@pytest.mark.parametrize(
    "bad_context",
    [
        pytest.param(jnp.asarray([1.75, 2.25, 3.5, 4.5, 5.5], dtype=jnp.float32), id="float"),
        pytest.param(jnp.asarray([True, False, True, False, True]), id="bool"),
    ],
)
def test_associative_pipeline_rejects_non_integer_context(bad_context) -> None:
    """Associative contexts must not be laundered into int32 by the pipeline.

    ``AssociativeMemoryLearner`` documents ``Int[Array, " block_size"]`` and
    rejects a non-integer context itself. The pipeline must not defeat that
    validator by narrowing the caller's array first: a fractional or boolean
    observation is an invalid input, not a truncated one.
    """
    pipeline = make_alberta_pipeline(_small_associative_config())
    good_context = jnp.asarray([1, 2, 3, 4, 5], dtype=jnp.int32)

    with pytest.raises(ValueError, match="integer dtype"):
        pipeline.init(jr.key(0), bad_context)

    state = pipeline.init(jr.key(0), good_context)
    with pytest.raises(ValueError, match="integer dtype"):
        pipeline.update(
            state,
            bad_context,
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray([1.0], dtype=jnp.float32),
            associative_label=jnp.asarray(6, dtype=jnp.int32),
        )


@pytest.mark.parametrize(
    "bad_label",
    [
        pytest.param(jnp.asarray(6.9, dtype=jnp.float32), id="float"),
        pytest.param(jnp.asarray(True), id="bool"),
    ],
)
def test_associative_pipeline_rejects_non_integer_label(bad_label) -> None:
    """A fractional or boolean associative label must raise, not truncate."""
    pipeline = make_alberta_pipeline(_small_associative_config())
    context = jnp.asarray([1, 2, 3, 4, 5], dtype=jnp.int32)
    state = pipeline.init(jr.key(0), context)

    with pytest.raises(ValueError, match="integer dtype"):
        pipeline.update(
            state,
            context,
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray([1.0], dtype=jnp.float32),
            associative_label=bad_label,
        )


def test_associative_run_arrays_rejects_non_integer_observations_and_labels() -> None:
    """The array runner must reject non-integer contexts and label tables."""
    pipeline = make_alberta_pipeline(_small_associative_config())
    state = pipeline.init(jr.key(0), jnp.asarray([1, 2, 3, 4, 5], dtype=jnp.int32))
    observations = jnp.asarray([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=jnp.int32)
    steps = observations.shape[0]
    zeros = jnp.zeros((steps,), dtype=jnp.float32)
    cumulants = jnp.ones((steps, 1), dtype=jnp.float32)
    labels = jnp.asarray([6, 7], dtype=jnp.int32)

    with pytest.raises(ValueError, match="integer dtype"):
        pipeline.run_arrays(
            state,
            observations.astype(jnp.float32) + 0.5,
            zeros,
            zeros,
            cumulants,
            associative_labels=labels,
        )

    for bad_labels in (
        jnp.asarray([6.9, 7.1], dtype=jnp.float32),
        jnp.asarray([True, False]),
    ):
        with pytest.raises(ValueError, match="integer dtype"):
            pipeline.run_arrays(
                state,
                observations,
                zeros,
                zeros,
                cumulants,
                associative_labels=bad_labels,
            )


def test_associative_pipeline_accepts_documented_integer_contract() -> None:
    """The guard must not reject the documented integer contract."""
    pipeline = make_alberta_pipeline(_small_associative_config())
    context = jnp.asarray([1, 2, 3, 4, 5], dtype=jnp.int32)
    state = pipeline.init(jr.key(0), context)

    result = pipeline.update(
        state,
        context,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        associative_label=jnp.asarray(6, dtype=jnp.int32),
    )
    assert int(result.state.step_count) == 1


# silence the import lint warnings used in the test runner
_ = jax
