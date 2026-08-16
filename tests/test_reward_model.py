"""Tests for online reward models."""

from __future__ import annotations

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.reward_model import RLSRewardModel, RLSRewardModelConfig


def test_rls_reward_model_config_roundtrip() -> None:
    """Config serialization should preserve all fields."""
    config = RLSRewardModelConfig(
        feature_dim=4,
        forgetting=0.99,
        ridge=3.0,
        error_decay=0.9,
    )

    restored = RLSRewardModelConfig.from_config(config.to_config())

    assert restored == config


def test_rls_reward_model_learns_linear_reward() -> None:
    """RLS should quickly fit a deterministic linear reward surface."""
    model = RLSRewardModel(
        RLSRewardModelConfig(feature_dim=3, forgetting=1.0, ridge=0.1)
    )
    state = model.init()
    true_weights = jnp.array([0.25, -0.5, 0.75], dtype=jnp.float32)
    features = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 2.0, -1.0],
        ],
        dtype=jnp.float32,
    )

    for _ in range(16):
        for feature in features:
            reward = jnp.dot(true_weights, feature)
            state = model.update(state, feature, reward).state

    predictions = jnp.array([model.predict(state, feature) for feature in features])
    targets = features @ true_weights

    chex.assert_trees_all_close(predictions, targets, atol=2e-3, rtol=2e-3)
    chex.assert_tree_all_finite(state.covariance)
    assert int(state.step_count) == 80


def test_rls_reward_model_rejects_invalid_config() -> None:
    """Invalid numerical settings should fail early."""
    for config in (
        RLSRewardModelConfig(feature_dim=0),
        RLSRewardModelConfig(feature_dim=1, forgetting=0.0),
        RLSRewardModelConfig(feature_dim=1, ridge=0.0),
        RLSRewardModelConfig(feature_dim=1, error_decay=1.0),
    ):
        try:
            RLSRewardModel(config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {config}")


@pytest.mark.parametrize("value", [True, 1.0, "1", np.int64(1)])
def test_rls_reward_model_rejects_non_builtin_feature_dim(value: object) -> None:
    """Dimensions must not accept bool or integer-like aliases."""
    payload = RLSRewardModelConfig(feature_dim=1).to_config()
    payload["feature_dim"] = value

    with pytest.raises(ValueError, match="feature_dim"):
        RLSRewardModel(RLSRewardModelConfig.from_config(payload))


class _FloatSpoof:
    """Non-real object that spoofs ``float`` through ``__class__``."""

    @property
    def __class__(self) -> type[float]:  # type: ignore[override]
        return float

    def __float__(self) -> float:
        return 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("forgetting", True),
        ("forgetting", "0.5"),
        ("forgetting", float("nan")),
        ("forgetting", float("inf")),
        ("forgetting", float("-inf")),
        ("ridge", True),
        ("ridge", "1.0"),
        ("ridge", float("nan")),
        ("ridge", float("inf")),
        ("ridge", float("-inf")),
        ("error_decay", True),
        ("error_decay", "0.5"),
        ("error_decay", float("nan")),
        ("error_decay", float("inf")),
        ("error_decay", float("-inf")),
        pytest.param("forgetting", _FloatSpoof(), id="forgetting-class-spoof"),
        pytest.param("ridge", _FloatSpoof(), id="ridge-class-spoof"),
        pytest.param("error_decay", _FloatSpoof(), id="error-decay-class-spoof"),
    ],
)
def test_rls_reward_model_rejects_non_real_or_non_finite_scalars(
    field: str,
    value: object,
) -> None:
    """Serialized config must reject booleans, objects, NaN, and infinities."""
    payload = RLSRewardModelConfig(feature_dim=1).to_config()
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        RLSRewardModel(RLSRewardModelConfig.from_config(payload))


def test_rls_infinite_reward_on_zero_feature_does_not_poison_weights() -> None:
    """Inf reward * a silent feature's zero gain is 0*inf = NaN."""
    model = RLSRewardModel(
        RLSRewardModelConfig(feature_dim=2, forgetting=1.0, ridge=1.0)
    )
    state = model.init()
    features = jnp.array([0.0, 1.0], dtype=jnp.float32)

    poisoned = model.update(state, features, jnp.array(jnp.inf, dtype=jnp.float32))
    chex.assert_trees_all_close(poisoned.state.weights, state.weights)
    chex.assert_trees_all_close(poisoned.state.covariance, state.covariance)
    assert int(poisoned.state.step_count) == int(state.step_count)
    assert not bool(poisoned.update_applied)
    assert float(poisoned.prediction) == 0.0
    assert float(poisoned.error) == 0.0
    chex.assert_trees_all_close(poisoned.gain, jnp.zeros_like(poisoned.gain))

    recovered = model.update(
        poisoned.state, features, jnp.array(1.0, dtype=jnp.float32)
    )
    chex.assert_tree_all_finite(recovered.state.weights)
    chex.assert_tree_all_finite(recovered.state.covariance)
    assert bool(recovered.update_applied)


def test_zero_error_decay_does_not_multiply_inf_ema() -> None:
    """error_decay=0 times an infinite abs-error EMA is NaN."""
    model = RLSRewardModel(
        RLSRewardModelConfig(feature_dim=2, forgetting=1.0, ridge=1.0, error_decay=0.0)
    )
    features = jnp.array([0.0, 1.0], dtype=jnp.float32)
    state = model.update(
        model.init(), features, jnp.array(1.0, dtype=jnp.float32)
    ).state
    state = state.replace(abs_error_ema=jnp.asarray(jnp.inf, dtype=jnp.float32))
    raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
    assert not bool(jnp.isfinite(raw))

    result = model.update(state, features, jnp.array(0.5, dtype=jnp.float32))
    assert bool(result.update_applied)
    assert bool(jnp.isfinite(result.state.abs_error_ema))
