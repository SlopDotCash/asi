# mypy: disable-error-code="call-arg,untyped-decorator,no-any-return"
"""Tests for fixed-budget Step 2 prototype memory."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.prototype_memory import (
    PrototypeMemoryConfig,
    PrototypeMemoryLearner,
    PrototypeMemoryState,
    run_prototype_memory_arrays,
)


def test_prototype_memory_init_shapes() -> None:
    """Initial state should match configured budget."""
    config = PrototypeMemoryConfig(feature_dim=4, n_classes=3, slots_per_class=5)
    learner = PrototypeMemoryLearner(config)
    state = learner.init()

    chex.assert_shape(state.means, (3, 5, 4))
    chex.assert_shape(state.counts, (3, 5))
    chex.assert_shape(state.last_update, (3, 5))
    assert int(state.step_count) == 0
    chex.assert_tree_all_finite(state)


def test_prototype_memory_hit_count_is_int32_and_saturates() -> None:
    """Hit counts must be exact past 2**24 and saturate instead of stalling."""

    config = PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=1)
    learner = PrototypeMemoryLearner(config)
    state = learner.init()
    assert state.counts.dtype == jnp.dtype(jnp.int32)
    target = jnp.asarray([1.0, 0.0], dtype=jnp.float32)

    planted = state.replace(
        counts=jnp.array([[16777216], [0]], dtype=jnp.int32),
        means=jnp.array([[[1.0, 1.0]], [[0.0, 0.0]]], dtype=jnp.float32),
    )
    result = learner.update(planted, jnp.ones((2,), dtype=jnp.float32), target)
    assert int(result.state.counts[0, 0]) == 16777217

    exhausted = result.state.replace(
        counts=result.state.counts.at[0, 0].set(2**31 - 1)
    )
    final = learner.update(exhausted, jnp.ones((2,), dtype=jnp.float32), target)
    assert int(final.state.counts[0, 0]) == 2**31 - 1


def test_replacement_uses_exact_integer_minimum_past_float32_precision() -> None:
    """An older, higher-count slot must not join the least-used tie set."""

    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=2)
    )
    state = learner.init().replace(
        counts=jnp.asarray([[2**24, 2**24 + 1], [0, 0]], dtype=jnp.int32),
        last_update=jnp.asarray([[2, 1], [0, 0]], dtype=jnp.int32),
        step_count=jnp.asarray(2, dtype=jnp.int32),
    )
    head = jnp.asarray(0, dtype=jnp.int32)

    eager_slot = learner._replacement_slot(state, head)
    jitted_slot = jax.jit(learner._replacement_slot)(state, head)

    assert int(eager_slot) == 0
    assert int(jitted_slot) == 0


def test_empty_memory_predicts_uniformly() -> None:
    """With no prototypes, softmax logits should be neutral."""
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=2, n_classes=4, slots_per_class=2)
    )
    state = learner.init()
    prediction = learner.predict(state, jnp.asarray([1.0, -1.0], dtype=jnp.float32))

    chex.assert_trees_all_close(prediction, jnp.full((4,), 0.25, dtype=jnp.float32))


def test_occupied_class_keeps_mass_when_distance_is_below_empty_sentinel() -> None:
    """An occupied class worse than -1e9 must not lose to an empty class.

    ``class_logits`` used to replace empty-class max-slots with ``-1e9``.
    A legal ``bandwidth=1e-8`` and a finite observation of ``5`` then scored
    the occupied prototype at ``-2.5e9`` and the empty class at ``-1e9``, so
    ``predict`` returned ``[0, 1]`` for a class that was never observed.
    """
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(
            feature_dim=1,
            n_classes=2,
            slots_per_class=1,
            bandwidth=1e-8,
        )
    )
    state = learner.init().replace(
        means=learner.init().means.at[0, 0, 0].set(0.0),
        counts=learner.init().counts.at[0, 0].set(1),
    )
    observation = jnp.asarray([5.0], dtype=jnp.float32)

    logits = learner.class_logits(state, observation)
    prediction = learner.predict(state, observation)

    assert float(logits[0]) == pytest.approx(-2.5e9)
    assert not bool(jnp.isfinite(logits[1]))
    assert float(logits[1]) < 0.0
    chex.assert_trees_all_close(
        prediction,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
    )
    assert int(jnp.argmax(prediction)) == 0


def test_repeated_update_moves_prediction_to_target_class() -> None:
    """A repeated class example should become confidently classified."""
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(
            feature_dim=2,
            n_classes=3,
            slots_per_class=2,
            novelty_threshold=0.5,
            bandwidth=0.05,
        )
    )
    state = learner.init()
    observation = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
    target = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32)

    for _ in range(3):
        result = learner.update(state, observation, target)
        state = result.state

    prediction = learner.predict(state, observation)
    assert int(jnp.argmax(prediction)) == 1
    assert float(prediction[1]) > 0.95
    assert int(jnp.sum(state.counts > 0.0)) == 1


def test_novelty_allocates_multiple_prototypes_per_class() -> None:
    """Far examples with the same class should occupy different slots."""
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(
            feature_dim=2,
            n_classes=2,
            slots_per_class=3,
            novelty_threshold=0.01,
        )
    )
    state = learner.init()
    target = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    for observation in (
        jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        jnp.asarray([1.0, 1.0], dtype=jnp.float32),
    ):
        state = learner.update(state, observation, target).state

    assert int(jnp.sum(state.counts[0] > 0.0)) == 2


def test_invalid_target_advances_time_without_allocating() -> None:
    """Non-simplex targets should not corrupt memory slots."""
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=2)
    )
    state = learner.init()
    result = learner.update(
        state,
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
    )

    assert int(result.state.step_count) == 1
    assert int(jnp.sum(result.state.counts > 0.0)) == 0
    assert float(result.metrics[4]) == 0.0


def test_run_prototype_memory_arrays_is_scan_compatible() -> None:
    """Array runner should return fixed-width predictions and metrics."""
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=2)
    )
    observations = jnp.asarray(
        [[0.0, 0.0], [1.0, 1.0], [0.1, 0.0]],
        dtype=jnp.float32,
    )
    targets = jnp.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=jnp.float32,
    )

    result = run_prototype_memory_arrays(learner, observations, targets)

    chex.assert_shape(result.predictions, (3, 2))
    chex.assert_shape(result.metrics, (3, 6))
    assert int(result.state.step_count) == 3
    chex.assert_tree_all_finite(result)


def test_update_can_be_wrapped_by_jit() -> None:
    """Single-step update should work inside an outer JIT."""
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=2)
    )
    state = learner.init()
    observation = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    target = jnp.asarray([0.0, 1.0], dtype=jnp.float32)

    @jax.jit
    def update_once(inner_state: PrototypeMemoryState) -> PrototypeMemoryState:
        return learner.update(inner_state, observation, target).state

    updated = update_once(state)
    assert int(updated.step_count) == 1


def test_config_roundtrip() -> None:
    """Config serialization should be reversible."""
    config = PrototypeMemoryConfig(
        feature_dim=7,
        n_classes=5,
        slots_per_class=4,
        update_rate=0.25,
        novelty_threshold=0.2,
        bandwidth=0.03,
    )
    learner = PrototypeMemoryLearner(config)

    assert PrototypeMemoryConfig.from_config(config.to_config()) == config
    assert PrototypeMemoryLearner.from_config(learner.to_config()).config == config


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bandwidth": float("inf")}, "bandwidth"),
        ({"bandwidth": float("nan")}, "bandwidth"),
        ({"novelty_threshold": float("inf")}, "novelty_threshold"),
        ({"novelty_threshold": float("nan")}, "novelty_threshold"),
        ({"update_rate": float("nan")}, "update_rate"),
    ],
)
def test_config_rejects_nonfinite_floats(kwargs: dict[str, float], match: str) -> None:
    """Non-finite kernel hyperparameters must fail closed at construction."""
    with pytest.raises(ValueError, match=match):
        PrototypeMemoryLearner(
            PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=1, **kwargs)
        )


def test_infinite_observation_prediction_stays_nonfinite() -> None:
    """Inf observations corrupt logits; predict must stay fail-visible, not a uniform simplex."""
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=2, n_classes=3, slots_per_class=2)
    )
    state = learner.init()
    target = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32)
    state = learner.update(
        state, jnp.asarray([0.25, 0.75], dtype=jnp.float32), target
    ).state
    obs = jnp.asarray([jnp.inf, 0.0], dtype=jnp.float32)

    logits = learner.class_logits(state, obs)
    assert not bool(jnp.all(jnp.isfinite(logits)))
    prediction = learner.predict(state, obs)
    assert not bool(jnp.all(jnp.isfinite(prediction)))
_INVALID_PROTOTYPE_CONFIGS: tuple[dict[str, object], ...] = (
    {"feature_dim": 0, "n_classes": 2},
    {"feature_dim": -1, "n_classes": 2},
    {"feature_dim": 2**31, "n_classes": 2},
    {"feature_dim": True, "n_classes": 2},
    {"feature_dim": "4", "n_classes": 2},
    {"feature_dim": 4, "n_classes": 1},
    {"feature_dim": 4, "n_classes": 0},
    {"feature_dim": 4, "n_classes": -1},
    {"feature_dim": 4, "n_classes": 2**31},
    {"feature_dim": 4, "n_classes": True},
    {"feature_dim": 4, "n_classes": "2"},
    {"feature_dim": 4, "n_classes": 2, "slots_per_class": 0},
    {"feature_dim": 4, "n_classes": 2, "slots_per_class": -1},
    {"feature_dim": 4, "n_classes": 2, "slots_per_class": 2**31},
    {"feature_dim": 4, "n_classes": 2, "slots_per_class": True},
    {"feature_dim": 4, "n_classes": 2, "update_rate": 0.0},
    {"feature_dim": 4, "n_classes": 2, "update_rate": -0.1},
    {"feature_dim": 4, "n_classes": 2, "update_rate": 1.1},
    {"feature_dim": 4, "n_classes": 2, "update_rate": 1e100},
    {"feature_dim": 4, "n_classes": 2, "update_rate": float("nan")},
    {"feature_dim": 4, "n_classes": 2, "update_rate": True},
    {"feature_dim": 4, "n_classes": 2, "novelty_threshold": -0.1},
    {"feature_dim": 4, "n_classes": 2, "novelty_threshold": 1e100},
    {"feature_dim": 4, "n_classes": 2, "novelty_threshold": float("nan")},
    {"feature_dim": 4, "n_classes": 2, "novelty_threshold": True},
    {"feature_dim": 4, "n_classes": 2, "bandwidth": 0.0},
    {"feature_dim": 4, "n_classes": 2, "bandwidth": -0.1},
    {"feature_dim": 4, "n_classes": 2, "bandwidth": 1e100},
    {"feature_dim": 4, "n_classes": 2, "bandwidth": float("nan")},
    {"feature_dim": 4, "n_classes": 2, "bandwidth": True},
)


@pytest.mark.parametrize("kwargs", _INVALID_PROTOTYPE_CONFIGS)
def test_prototype_memory_config_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PrototypeMemoryConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("numpy_type", [np.longlong, np.ulonglong])
@pytest.mark.parametrize("field", ["feature_dim", "n_classes", "slots_per_class"])
def test_prototype_memory_config_accepts_extended_numpy_integers(
    field: str,
    numpy_type: type[np.generic],
) -> None:
    kwargs: dict[str, object] = {"feature_dim": 4, "n_classes": 2}
    kwargs[field] = numpy_type(4)

    config = PrototypeMemoryConfig(**kwargs)

    assert getattr(config, field) == 4
    assert type(getattr(config, field)) is int
    assert PrototypeMemoryConfig.from_config(config.to_config()) == config


@pytest.mark.parametrize("numpy_type", [np.longlong, np.ulonglong])
@pytest.mark.parametrize("field", ["feature_dim", "n_classes", "slots_per_class"])
def test_prototype_memory_config_bounds_extended_numpy_integers(
    field: str,
    numpy_type: type[np.generic],
) -> None:
    kwargs: dict[str, object] = {"feature_dim": 4, "n_classes": 2}
    kwargs[field] = numpy_type(2**31)

    with pytest.raises(ValueError, match=field):
        PrototypeMemoryConfig(**kwargs)


@pytest.mark.parametrize("field", ["feature_dim", "n_classes", "slots_per_class"])
def test_prototype_memory_config_rejects_negative_numpy_longlong(field: str) -> None:
    kwargs: dict[str, object] = {"feature_dim": 4, "n_classes": 2}
    kwargs[field] = np.longlong(-1)

    with pytest.raises(ValueError, match=field):
        PrototypeMemoryConfig(**kwargs)


@pytest.mark.parametrize(
    "ratio",
    [
        pytest.param((-1, 1), id="negative-ratio"),
        pytest.param((2, 1), id="above-unit-ratio"),
        pytest.param((-1, 2**200), id="negative-rounds-to-negative-zero"),
        pytest.param((2**200 + 1, 2**200), id="above-one-rounds-to-one"),
    ],
)
def test_prototype_memory_rejects_adversarial_ratio_floats(
    ratio: tuple[int, int]
) -> None:
    class HiddenBoundaryFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return ratio

    with pytest.raises(ValueError, match="update_rate"):
        PrototypeMemoryConfig(
            feature_dim=4,
            n_classes=2,
            update_rate=HiddenBoundaryFloat(0.5),
        )


def test_prototype_memory_rejects_class_property_spoofing_float() -> None:
    class ClassSpoof:
        @property
        def __class__(self) -> type[float]:
            return float

        def as_integer_ratio(self) -> tuple[int, int]:
            return (1, 2)

    value = ClassSpoof()
    with pytest.raises(ValueError, match="finite real"):
        PrototypeMemoryConfig(
            feature_dim=4,
            n_classes=2,
            update_rate=value,  # type: ignore[arg-type]
        )


def test_prototype_memory_rejects_hostile_integral_subclasses() -> None:
    class LieInt(int):
        def __int__(self) -> int:
            return 4

    for field in ("feature_dim", "n_classes", "slots_per_class"):
        with pytest.raises(ValueError, match=field):
            PrototypeMemoryConfig(
                **{"feature_dim": 4, "n_classes": 2, field: LieInt(-1)}
            )


@pytest.mark.parametrize(
    "integer_type",
    tuple(
        dict.fromkeys(
            np.dtype(code).type
            for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q")
        )
    ),
)
def test_prototype_memory_canonicalizes_every_numpy_integer_type(
    integer_type: type,
) -> None:
    config = PrototypeMemoryConfig(
        feature_dim=integer_type(4),
        n_classes=integer_type(4),
        slots_per_class=integer_type(4),
    )

    assert type(config.feature_dim) is int
    assert type(config.n_classes) is int
    assert type(config.slots_per_class) is int


def test_prototype_memory_integer_errors_do_not_interpolate_hostile_repr() -> None:
    class ClassSpoof:
        @property
        def __class__(self) -> type[int]:  # type: ignore[override]
            return int

        def __repr__(self) -> str:
            raise RuntimeError("repr must not run")

    with pytest.raises(ValueError, match="feature_dim"):
        PrototypeMemoryConfig(
            feature_dim=ClassSpoof(),  # type: ignore[arg-type]
            n_classes=2,
        )
