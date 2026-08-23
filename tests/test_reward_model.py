"""Tests for online reward models."""

from __future__ import annotations

import math

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.reward_model import (
    RLSRewardModel,
    RLSRewardModelConfig,
    RLSRewardModelState,
)


@pytest.mark.parametrize("operation", ["predict", "update"])
@pytest.mark.parametrize("shape", [(), (4, 1), (1, 4), (2, 2), (3,), (5,)])
def test_rls_reward_model_rejects_non_vector_feature_shapes(
    operation: str,
    shape: tuple[int, ...],
) -> None:
    model = RLSRewardModel(RLSRewardModelConfig(feature_dim=4))
    state = model.init()
    features = jnp.zeros(shape, dtype=jnp.float32)

    with pytest.raises(ValueError, match=r"features must have shape \(4,\)"):
        if operation == "predict":
            model.predict(state, features)
        else:
            model.update(state, features, jnp.array(0.0, dtype=jnp.float32))


def test_rls_reward_model_feature_shape_guard_is_jit_stable() -> None:
    model = RLSRewardModel(RLSRewardModelConfig(feature_dim=4))
    state = model.init()
    predict = jax.jit(lambda features: model.predict(state, features))
    update = jax.jit(
        lambda features: (
            model.update(
                state,
                features,
                jnp.array(0.0, dtype=jnp.float32),
            ).state.weights
        )
    )

    chex.assert_shape(predict(jnp.ones((4,), dtype=jnp.float32)), ())
    chex.assert_shape(update(jnp.ones((4,), dtype=jnp.float32)), (4,))
    for call in (predict, update):
        with pytest.raises(ValueError, match=r"features must have shape \(4,\)"):
            call(jnp.ones((2, 2), dtype=jnp.float32))


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
    model = RLSRewardModel(RLSRewardModelConfig(feature_dim=3, forgetting=1.0, ridge=0.1))
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
    for kwargs in (
        {"feature_dim": 0},
        {"feature_dim": 1, "forgetting": 0.0},
        {"feature_dim": 1, "ridge": 0.0},
        {"feature_dim": 1, "error_decay": 1.0},
    ):
        with pytest.raises(ValueError):
            RLSRewardModelConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, 1.0, "1", [1]])
def test_rls_reward_model_rejects_non_builtin_feature_dim(value: object) -> None:
    """Dimensions must not accept bool or non-integer aliases."""
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("forgetting", 1e-50),
        ("ridge", 1e-50),
        ("ridge", 1e50),
        ("error_decay", 1.0 - 1e-10),
    ],
)
def test_rls_reward_model_rejects_values_invalid_after_float32_narrowing(
    field: str,
    value: float,
) -> None:
    payload = RLSRewardModelConfig(feature_dim=1).to_config()
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        RLSRewardModel(RLSRewardModelConfig.from_config(payload))


def test_rls_reward_model_canonicalizes_numpy_scalars() -> None:
    model = RLSRewardModel(
        RLSRewardModelConfig(
            feature_dim=1,
            forgetting=np.float32(0.75),
            ridge=np.float32(0.5),
            error_decay=np.float32(0.25),
        )
    )

    assert type(model.config.forgetting) is float
    assert type(model.config.ridge) is float
    assert type(model.config.error_decay) is float
    assert RLSRewardModel.from_config(model.to_config()).to_config() == model.to_config()


def test_rls_infinite_reward_on_zero_feature_does_not_poison_weights() -> None:
    """Inf reward * a silent feature's zero gain is 0*inf = NaN."""
    model = RLSRewardModel(RLSRewardModelConfig(feature_dim=2, forgetting=1.0, ridge=1.0))
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

    recovered = model.update(poisoned.state, features, jnp.array(1.0, dtype=jnp.float32))
    chex.assert_tree_all_finite(recovered.state.weights)
    chex.assert_tree_all_finite(recovered.state.covariance)
    assert bool(recovered.update_applied)


def test_zero_error_decay_does_not_multiply_inf_ema() -> None:
    """error_decay=0 times an infinite abs-error EMA is NaN."""
    model = RLSRewardModel(
        RLSRewardModelConfig(feature_dim=2, forgetting=1.0, ridge=1.0, error_decay=0.0)
    )
    features = jnp.array([0.0, 1.0], dtype=jnp.float32)
    state = model.update(model.init(), features, jnp.array(1.0, dtype=jnp.float32)).state
    state = state.replace(abs_error_ema=jnp.asarray(jnp.inf, dtype=jnp.float32))
    raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
    assert not bool(jnp.isfinite(raw))

    result = model.update(state, features, jnp.array(0.5, dtype=jnp.float32))
    assert bool(result.update_applied)
    assert bool(jnp.isfinite(result.state.abs_error_ema))


def test_reward_model_step_count_saturates_at_int32_max() -> None:
    model = RLSRewardModel(RLSRewardModelConfig(feature_dim=1, forgetting=1.0))
    state = model.init().replace(step_count=jnp.array(2**31 - 1, dtype=jnp.int32))

    result = model.update(state, jnp.array([1.0], dtype=jnp.float32), jnp.array(1.0))

    assert bool(result.update_applied)
    assert int(result.state.step_count) == 2**31 - 1


def test_rls_reward_model_config_integer_and_scalar_validation() -> None:
    with pytest.raises(ValueError, match="feature_dim"):
        RLSRewardModelConfig(feature_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_dim"):
        RLSRewardModelConfig(feature_dim=4.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feature_dim"):
        RLSRewardModelConfig(feature_dim=0)
    with pytest.raises(ValueError, match="forgetting"):
        RLSRewardModelConfig(feature_dim=2, forgetting=float("nan"))
    with pytest.raises(ValueError, match="ridge"):
        RLSRewardModelConfig(feature_dim=2, ridge=0.0)

    cfg = RLSRewardModelConfig(
        feature_dim=np.int32(8),
        forgetting=np.float32(0.95),
        ridge=np.float32(5.0),
        error_decay=np.float32(0.8),
    )
    assert type(cfg.feature_dim) is int
    assert type(cfg.forgetting) is float
    assert type(cfg.ridge) is float
    assert type(cfg.error_decay) is float
    assert cfg.feature_dim == 8


_NUMPY_INTEGER_TYPES = tuple(dict.fromkeys(np.dtype(code).type for code in "bhilqBHILQpP"))


@pytest.mark.parametrize("integer_type", _NUMPY_INTEGER_TYPES)
def test_rls_reward_model_config_canonicalizes_every_numpy_integer_family(
    integer_type: type,
) -> None:
    config = RLSRewardModelConfig(feature_dim=integer_type(4))

    assert type(config.feature_dim) is int
    assert config.feature_dim == 4


def test_rls_reward_model_config_rejects_hostile_integer_types_without_repr() -> None:
    class HostileInt(int):
        def __repr__(self) -> str:
            raise AssertionError("repr must not run")

    class ClassSpoof:
        @property
        def __class__(self) -> type:
            return int

        def __repr__(self) -> str:
            raise AssertionError("repr must not run")

    for value in (HostileInt(4), ClassSpoof()):
        with pytest.raises(ValueError, match="feature_dim"):
            RLSRewardModelConfig(feature_dim=value)  # type: ignore[arg-type]


def test_reward_model_exact_schema_and_hostile_float_hooks() -> None:
    config = RLSRewardModelConfig(feature_dim=4)
    payload = config.to_config()

    class HostileDict(dict[str, object]):
        def __iter__(self):
            raise AssertionError("iteration must not run")

    class HostileFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            raise AssertionError("ratio must not run")

        def __repr__(self) -> str:
            raise AssertionError("repr must not run")

    with pytest.raises(ValueError, match="exact built-in dict"):
        RLSRewardModelConfig.from_config(HostileDict(payload))
    with pytest.raises(ValueError, match="serialized schema"):
        RLSRewardModelConfig.from_config({**payload, "extra": 1})
    with pytest.raises(ValueError, match="payload type"):
        RLSRewardModelConfig.from_config({**payload, "type": "wrong"})
    for field in ("forgetting", "ridge", "error_decay"):
        with pytest.raises(ValueError, match=field):
            RLSRewardModelConfig(feature_dim=1, **{field: HostileFloat(0.5)})


def test_rls_reward_model_preflights_quadratic_state_before_allocation() -> None:
    scalar_limit = (2**31 - 1) // 4
    last_legal = (math.isqrt(1 + 4 * (scalar_limit - 2)) - 1) // 2

    RLSRewardModelConfig(feature_dim=last_legal)
    with pytest.raises(ValueError, match="state bytes"):
        RLSRewardModelConfig(feature_dim=last_legal + 1)


@pytest.mark.parametrize("shape", ((1,), (1, 1)))
def test_rls_reward_model_rejects_non_scalar_rewards_eager_and_outer_jit(
    shape: tuple[int, ...],
) -> None:
    model = RLSRewardModel(RLSRewardModelConfig(feature_dim=2))
    state = model.init()
    features = jnp.ones((2,), dtype=jnp.float32)
    reward = jnp.zeros(shape, dtype=jnp.float32)

    with pytest.raises(ValueError, match="reward must be a scalar"):
        model.update(state, features, reward)
    compiled = jax.jit(lambda value: model.update(state, features, value).state)
    with pytest.raises(ValueError, match="reward must be a scalar"):
        compiled(reward)


def test_rls_reward_model_covariance_preserves_exact_symmetry_single_step() -> None:
    """Covariance update must remain exactly symmetric in float32."""
    model = RLSRewardModel(RLSRewardModelConfig(feature_dim=4, forgetting=0.99, ridge=1.0))
    state = model.init()
    features = jnp.array([1.2345678, 2.3456789, -3.4567891, 0.5678901], dtype=jnp.float32)
    reward = jnp.array(1.0, dtype=jnp.float32)

    result = model.update(state, features, reward)
    cov = np.array(result.state.covariance)
    assert np.array_equal(cov, cov.T)
    assert np.max(np.abs(cov - cov.T)) == 0.0


def test_rls_reward_model_covariance_preserves_exact_symmetry_multi_step() -> None:
    """Covariance update must remain exactly symmetric at every compiled step."""
    model = RLSRewardModel(RLSRewardModelConfig(feature_dim=4, forgetting=0.99, ridge=1.0))
    state = model.init()
    rng = np.random.default_rng(12345)

    for step in range(100):
        features = jnp.array(rng.standard_normal(4), dtype=jnp.float32)
        reward = jnp.array(rng.standard_normal(), dtype=jnp.float32)
        state = model.update(state, features, reward).state
        cov = np.array(state.covariance)
        assert np.array_equal(cov, cov.T), f"Covariance asymmetry at step {step}"
        assert np.max(np.abs(cov - cov.T)) == 0.0


def test_rls_reward_model_covariance_symmetrization_avoids_finite_overflow() -> None:
    """Finite extreme covariance does not overflow or trigger false rollback."""
    model = RLSRewardModel(RLSRewardModelConfig(feature_dim=2, forgetting=1.0, ridge=1.0))
    cov = jnp.diag(jnp.array([2e38, 2e38], dtype=jnp.float32))
    state = RLSRewardModelState(
        weights=jnp.zeros(2, dtype=jnp.float32),
        covariance=cov,
        abs_error_ema=jnp.array(0.0, dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
    )
    features = jnp.zeros(2, dtype=jnp.float32)
    reward = jnp.array(1.0, dtype=jnp.float32)

    result = model.update(state, features, reward)
    assert bool(result.update_applied)
    assert int(result.state.step_count) == 1
    assert float(result.error) == 1.0
    cov_out = np.array(result.state.covariance)
    assert np.all(np.isfinite(cov_out))
    assert np.array_equal(cov_out, cov_out.T)
