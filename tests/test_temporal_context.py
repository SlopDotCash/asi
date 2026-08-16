# mypy: disable-error-code="call-arg,untyped-decorator"
"""Tests for causal temporal/context features."""

from typing import cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.temporal_context import (
    TemporalContextConfig,
    TemporalContextFeaturizer,
    TemporalContextState,
    transform_temporal_context_arrays,
)


def test_temporal_context_shapes_and_roundtrip() -> None:
    config = TemporalContextConfig(input_dim=3, periods=(5.0, 10.0))
    featurizer = TemporalContextFeaturizer(config)
    state = featurizer.init()

    features = featurizer.features(state, jnp.ones(3))

    assert config.output_dim() == 13
    chex.assert_shape(features, (13,))
    chex.assert_tree_all_finite(features)
    assert TemporalContextConfig.from_config(config.to_config()) == config


def test_temporal_context_step_is_causal() -> None:
    config = TemporalContextConfig(input_dim=2, ema_decay=0.5, periods=())
    featurizer = TemporalContextFeaturizer(config)
    state = featurizer.init()
    observation = jnp.asarray([2.0, -2.0], dtype=jnp.float32)

    next_state, features = featurizer.step(state, observation)

    chex.assert_trees_all_close(features[:2], observation)
    chex.assert_trees_all_close(features[2:4], jnp.zeros(2))
    chex.assert_trees_all_close(features[4:6], observation)
    chex.assert_trees_all_close(next_state.observation_ema, observation * 0.5)
    assert int(next_state.step_count) == 1


def test_zero_ema_decay_does_not_multiply_inf_ema() -> None:
    """ema_decay=0 times an infinite tracker is NaN and would be committed."""
    config = TemporalContextConfig(input_dim=2, ema_decay=0.0, periods=())
    featurizer = TemporalContextFeaturizer(config)
    state = featurizer.init().replace(
        observation_ema=jnp.asarray([jnp.inf, jnp.inf], dtype=jnp.float32)
    )
    raw = jnp.asarray(0.0, dtype=jnp.float32) * jnp.asarray(jnp.inf, dtype=jnp.float32)
    assert not bool(jnp.isfinite(raw))

    observation = jnp.asarray([2.0, -1.0], dtype=jnp.float32)
    next_state = featurizer.update(state, observation)
    chex.assert_trees_all_close(next_state.observation_ema, observation)
    assert int(next_state.step_count) == 1


def test_temporal_context_phase_products_expand_with_input() -> None:
    config = TemporalContextConfig(
        input_dim=2,
        include_phase_products=True,
        periods=(4.0,),
    )
    featurizer = TemporalContextFeaturizer(config)

    features = featurizer.features(featurizer.init(), jnp.asarray([3.0, -1.0]))

    assert config.output_dim() == 12
    chex.assert_shape(features, (12,))
    chex.assert_tree_all_finite(features)


def test_temporal_context_array_transform_is_jittable() -> None:
    config = TemporalContextConfig(input_dim=2, periods=(4.0,))
    featurizer = TemporalContextFeaturizer(config)
    observations = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)

    @jax.jit
    def run(
        initial_state: TemporalContextState,
    ) -> tuple[TemporalContextState, jax.Array]:
        return transform_temporal_context_arrays(
            featurizer,
            observations,
            state=initial_state,
        )

    state, features = run(featurizer.init())

    chex.assert_shape(features, (2, config.output_dim()))
    assert int(state.step_count) == 2
    chex.assert_tree_all_finite(features)


def test_nonfinite_observation_holds_warm_state_and_recovers_under_jit() -> None:
    config = TemporalContextConfig(
        input_dim=3,
        ema_decay=0.5,
        periods=(4.0,),
        include_phase_products=True,
    )
    featurizer = TemporalContextFeaturizer(config)
    warm_state, _ = featurizer.step(
        featurizer.init(),
        jnp.asarray([2.0, -4.0, 6.0], dtype=jnp.float32),
    )

    @jax.jit
    def run(
        state: TemporalContextState,
        observation: jax.Array,
    ) -> tuple[TemporalContextState, jax.Array]:
        return cast(
            tuple[TemporalContextState, jax.Array],
            featurizer.step(state, observation),
        )

    held_state, invalid_features = run(
        warm_state,
        jnp.asarray([jnp.inf, 8.0, jnp.nan], dtype=jnp.float32),
    )

    chex.assert_tree_all_finite(invalid_features)
    chex.assert_trees_all_equal(held_state, warm_state)
    chex.assert_trees_all_equal(invalid_features[:3], jnp.asarray([0.0, 8.0, 0.0]))
    chex.assert_trees_all_equal(invalid_features[3:6], warm_state.observation_ema)
    # Invalid delta coordinates are exactly zero, not the negated warm EMA.
    chex.assert_trees_all_equal(invalid_features[6:9], jnp.asarray([0.0, 10.0, 0.0]))
    phase_products = invalid_features[11:].reshape(2, 3)
    chex.assert_trees_all_equal(phase_products[:, (0, 2)], jnp.zeros((2, 2)))

    recovered_state, recovered_features = run(
        held_state,
        jnp.asarray([4.0, 2.0, -2.0], dtype=jnp.float32),
    )

    chex.assert_tree_all_finite((recovered_state, recovered_features))
    chex.assert_trees_all_close(
        recovered_state.observation_ema,
        jnp.asarray([2.5, 0.0, 0.5], dtype=jnp.float32),
    )
    assert int(recovered_state.step_count) == 2


def test_finite_temporal_context_path_is_bitwise_unchanged() -> None:
    config = TemporalContextConfig(input_dim=3, ema_decay=0.75, periods=())
    featurizer = TemporalContextFeaturizer(config)
    state = TemporalContextState(
        observation_ema=jnp.asarray([0.25, -1.5, 3.0], dtype=jnp.float32),
        step_count=jnp.asarray(17, dtype=jnp.int32),
    )
    observation = jnp.asarray([1.5, 2.0, -0.5], dtype=jnp.float32)

    next_state, features = featurizer.step(state, observation)
    expected_features = jnp.concatenate(
        [observation, state.observation_ema, observation - state.observation_ema]
    )
    expected_ema = jnp.float32(0.75) * state.observation_ema + jnp.float32(
        0.25
    ) * observation

    assert np.asarray(features).tobytes() == np.asarray(expected_features).tobytes()
    assert np.asarray(next_state.observation_ema).tobytes() == np.asarray(
        expected_ema
    ).tobytes()
    assert int(next_state.step_count) == 18


_INVALID_TEMPORAL_CONTEXT_CONFIGS: tuple[dict[str, object], ...] = (
    {"input_dim": 0},
    {"input_dim": -1},
    {"input_dim": 2**31},
    {"input_dim": True},
    {"input_dim": "4"},
    {"input_dim": 4, "include_raw": 1},
    {"input_dim": 4, "include_ema": 1},
    {"input_dim": 4, "include_delta": 1},
    {"input_dim": 4, "include_phase_products": 1},
    {"input_dim": 4, "include_raw": False, "include_ema": False, "include_delta": False},
    {"input_dim": 4, "ema_decay": -0.1},
    {"input_dim": 4, "ema_decay": 1.0},
    {"input_dim": 4, "ema_decay": 1.1},
    {"input_dim": 4, "ema_decay": 1e100},
    {"input_dim": 4, "ema_decay": float("nan")},
    {"input_dim": 4, "ema_decay": True},
    {"input_dim": 4, "periods": (0.0,)},
    {"input_dim": 4, "periods": (-1.0,)},
    {"input_dim": 4, "periods": (1e100,)},
    {"input_dim": 4, "periods": (float("nan"),)},
    {"input_dim": 4, "periods": (True,)},
)


@pytest.mark.parametrize("kwargs", _INVALID_TEMPORAL_CONTEXT_CONFIGS)
def test_temporal_context_config_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TemporalContextConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ratio",
    [
        pytest.param((-1, 1), id="negative-ratio"),
        pytest.param((1, 1), id="one-ratio"),
        pytest.param((2, 1), id="above-unit-ratio"),
        pytest.param((-1, 2**200), id="negative-rounds-to-negative-zero"),
        pytest.param((2**200 + 1, 2**200), id="above-one-rounds-to-one"),
    ],
)
def test_temporal_context_rejects_adversarial_ratio_floats(
    ratio: tuple[int, int]
) -> None:
    class HiddenBoundaryFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return ratio

    with pytest.raises(ValueError, match=r"ema_decay must be in \[0, 1\)"):
        TemporalContextConfig(
            input_dim=4,
            ema_decay=HiddenBoundaryFloat(0.5),
        )


def test_temporal_context_rejects_class_property_spoofing_float() -> None:
    class ClassSpoof:
        @property
        def __class__(self) -> type[float]:
            return float

        def as_integer_ratio(self) -> tuple[int, int]:
            return (1, 2)

    value = ClassSpoof()
    with pytest.raises(ValueError, match="must be a real number"):
        TemporalContextConfig(
            input_dim=4,
            ema_decay=value,  # type: ignore[arg-type]
        )


def test_temporal_context_rejects_spoofed_bool_flags() -> None:
    class SpoofedBool:
        @property
        def __class__(self) -> type[bool]:
            return bool

        def __bool__(self) -> bool:
            return True

    with pytest.raises(ValueError, match="include_raw"):
        TemporalContextConfig(input_dim=4, include_raw=SpoofedBool())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="include_ema"):
        TemporalContextConfig(input_dim=4, include_ema=SpoofedBool())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="include_delta"):
        TemporalContextConfig(input_dim=4, include_delta=SpoofedBool())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="include_phase_products"):
        TemporalContextConfig(input_dim=4, include_phase_products=SpoofedBool())  # type: ignore[arg-type]


def test_temporal_context_rejects_spoofed_periods_container() -> None:
    class SpoofedTuple(list):
        @property
        def __class__(self) -> type[tuple]:
            return tuple

    with pytest.raises(ValueError, match="periods"):
        TemporalContextConfig(input_dim=4, periods=SpoofedTuple([50.0]))  # type: ignore[arg-type]


def test_temporal_context_rejects_spoofed_int_class_and_negative_ratios() -> None:
    class SpoofedIntFloat(float):
        @property
        def __class__(self) -> type[int]:
            return int

        def as_integer_ratio(self) -> tuple[int, int]:
            return (-1, 2**200)

    with pytest.raises(ValueError, match="ema_decay"):
        TemporalContextConfig(input_dim=4, ema_decay=SpoofedIntFloat(0.5))
