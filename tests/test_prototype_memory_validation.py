"""Validation hardening for prototype memory (int/float bounds + resource preflights)."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.prototype_memory import (
    PrototypeMemoryConfig,
    PrototypeMemoryLearner,
    run_prototype_memory_arrays,
)

_INT32_MAX = 2**31 - 1


class _LyingIntSubclass(int):
    def __int__(self) -> int:  # pragma: no cover
        return 2

    def __index__(self) -> int:  # pragma: no cover
        return 2


class _RaisingIntSubclass(int):
    def __int__(self) -> int:  # pragma: no cover
        raise RuntimeError("conversion hook must not run")

    def __index__(self) -> int:  # pragma: no cover
        raise RuntimeError("conversion hook must not run")


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook must not run")


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: PrototypeMemoryConfig(feature_dim=v, n_classes=2, slots_per_class=2),
        lambda v: PrototypeMemoryConfig(feature_dim=2, n_classes=v, slots_per_class=2),
        lambda v: PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=v),
    ],
)
def test_prototype_int_validators_reject_hostile_subclass_without_running_hook(
    ctor,
) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        ctor(_LyingIntSubclass(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        ctor(_RaisingIntSubclass(4))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: PrototypeMemoryConfig(feature_dim=v, n_classes=2, slots_per_class=2),
        lambda v: PrototypeMemoryConfig(feature_dim=2, n_classes=v, slots_per_class=2),
        lambda v: PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=v),
    ],
)
def test_prototype_int_validators_do_not_run_repr_hook(ctor) -> None:
    with pytest.raises(ValueError):
        ctor(_RaisingRepr())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "np_type",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,  # noqa: E501
        np.ulonglong,
    ],
)
def test_prototype_int_validators_canonicalize_numpy_scalars(np_type: type) -> None:
    cfg = PrototypeMemoryConfig(
        feature_dim=np_type(4),
        n_classes=np_type(3),
        slots_per_class=np_type(5),
    )
    assert cfg.feature_dim == 4
    assert cfg.n_classes == 3
    assert cfg.slots_per_class == 5
    assert type(cfg.feature_dim) is int
    assert type(cfg.n_classes) is int
    assert type(cfg.slots_per_class) is int


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: PrototypeMemoryConfig(feature_dim=v, n_classes=2, slots_per_class=2),
        lambda v: PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=v),
    ],
)
@pytest.mark.parametrize(
    "value", [True, np.bool_(True), 4.0, np.float64(4.0), "4", None, 0, -1, _INT32_MAX + 1, 10**100]
)
def test_prototype_int_validators_reject_non_integer_and_out_of_range(
    ctor, value: object
) -> None:
    with pytest.raises(ValueError, match="must be"):
        ctor(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value", [True, np.bool_(True), 4.0, "4", None, 1, 0, -1, _INT32_MAX + 1]
)
def test_prototype_n_classes_rejects_out_of_range(value: object) -> None:
    with pytest.raises(ValueError, match="must be"):
        PrototypeMemoryConfig(feature_dim=2, n_classes=value, slots_per_class=2)  # type: ignore[arg-type]


def test_prototype_float_validators_reject_nonfinite_and_hostile() -> None:
    class HostileFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            raise RuntimeError("untrusted ratio hook executed")

    class ClassSpoof:
        @property
        def __class__(self):  # type: ignore[no-untyped-def]
            return float

        def __float__(self) -> float:  # pragma: no cover
            return 0.1

    for field, bad in [
        ("update_rate", float("nan")),
        ("update_rate", float("inf")),
        ("update_rate", -0.1),
        ("update_rate", 0.0),
        ("update_rate", 2.0),
        ("update_rate", HostileFloat(0.5)),
        ("novelty_threshold", -0.1),
        ("novelty_threshold", float("nan")),
        ("novelty_threshold", HostileFloat(0.5)),
        ("bandwidth", 0.0),
        ("bandwidth", -1.0),
        ("bandwidth", float("nan")),
        ("bandwidth", float("inf")),
        ("bandwidth", HostileFloat(0.5)),
    ]:
        with pytest.raises(ValueError, match=field):
            PrototypeMemoryConfig(
                feature_dim=2,
                n_classes=2,
                slots_per_class=2,
                **{field: bad},  # type: ignore[arg-type]
            )


def test_prototype_float_validators_reject_hostile_ratio() -> None:
    class HostileFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            type(self).calls += 1
            raise RuntimeError("ratio hook")

    with pytest.raises(ValueError, match="update_rate"):
        PrototypeMemoryConfig(
            feature_dim=2,
            n_classes=2,
            slots_per_class=2,
            update_rate=HostileFloat(0.5),  # type: ignore[arg-type]
        )
    assert HostileFloat.calls == 1


def test_prototype_float_validators_accept_valid_values() -> None:
    cfg = PrototypeMemoryConfig(
        feature_dim=2,
        n_classes=2,
        slots_per_class=2,
        update_rate=0.3,
        novelty_threshold=0.08,
        bandwidth=0.01,
    )
    assert cfg.update_rate == pytest.approx(0.3)
    assert cfg.novelty_threshold == pytest.approx(0.08)


def test_prototype_dimensions_preflight_without_allocation() -> None:
    # n_classes * slots_per_class overflows signed int32.
    with pytest.raises(ValueError, match="dimensions must fit signed int32"):
        PrototypeMemoryConfig(
            feature_dim=2, n_classes=_INT32_MAX, slots_per_class=2
        )
    # n_classes * slots_per_class * feature_dim overflows.
    with pytest.raises(ValueError, match="dimensions must fit signed int32"):
        PrototypeMemoryConfig(
            feature_dim=_INT32_MAX, n_classes=2, slots_per_class=2
        )
    # Scalar-count preflight via large product but individual dims legal.
    with pytest.raises(ValueError, match="dimensions must fit signed|scalar count|byte count"):
        PrototypeMemoryConfig(
            feature_dim=1, n_classes=50000, slots_per_class=50000
        )


def test_prototype_state_preflight_bytes_without_allocation() -> None:
    # Minimal dimensions, varying slots: update bound = 36*slots + 65.
    last_legal = (_INT32_MAX // 4 - 65) // 36
    PrototypeMemoryConfig(feature_dim=1, n_classes=2, slots_per_class=last_legal)
    with pytest.raises(ValueError, match="update byte count"):
        PrototypeMemoryConfig(feature_dim=1, n_classes=2, slots_per_class=last_legal + 1)
    # Non-minimal vector should also be allocation-free.
    with pytest.raises(ValueError, match="byte count|scalar count|dimensions"):
        PrototypeMemoryConfig(feature_dim=5000, n_classes=5000, slots_per_class=5000)


def test_prototype_state_preflight_feature_dim_boundary() -> None:
    # n_classes=2, slots=1, varying features: update bound = 16*fd + 85.
    last_legal = (_INT32_MAX // 4 - 85) // 16
    PrototypeMemoryConfig(feature_dim=last_legal, n_classes=2, slots_per_class=1)
    with pytest.raises(ValueError, match="update byte count"):
        PrototypeMemoryConfig(feature_dim=last_legal + 1, n_classes=2, slots_per_class=1)


def test_prototype_bandwidth_must_remain_normal_at_float32_sink() -> None:
    minimum_normal = float.fromhex("0x1.0p-126")
    config = PrototypeMemoryConfig(feature_dim=2, n_classes=2, bandwidth=minimum_normal)
    assert config.bandwidth == minimum_normal
    with pytest.raises(ValueError, match="bandwidth"):
        PrototypeMemoryConfig(
            feature_dim=2,
            n_classes=2,
            bandwidth=float.fromhex("0x1.0p-149"),
        )


def test_prototype_config_mapping_compatibility_rejects_spoofs_before_hooks() -> None:
    class MappingSpoof:
        @property  # type: ignore[misc]
        def __class__(self) -> type:
            return dict

        def __iter__(self) -> object:
            raise AssertionError("iteration hook executed")

        def __repr__(self) -> str:
            raise AssertionError("repr hook executed")

    class HostileMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("getitem hook")

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError("iteration hook")

        def __len__(self) -> int:
            return 1

    config = PrototypeMemoryConfig(feature_dim=2, n_classes=2)
    assert PrototypeMemoryConfig.from_config(MappingProxyType(config.to_config())) == config
    learner = PrototypeMemoryLearner(config)
    restored = PrototypeMemoryLearner.from_config(MappingProxyType(learner.to_config()))
    assert restored.config == config
    with pytest.raises(ValueError, match="mapping"):
        PrototypeMemoryConfig.from_config(MappingSpoof())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mapping"):
        PrototypeMemoryLearner.from_config(MappingSpoof())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mapping could not be read"):
        PrototypeMemoryConfig.from_config(HostileMapping())
    with pytest.raises(ValueError, match="mapping could not be read"):
        PrototypeMemoryLearner.from_config(HostileMapping())


def test_prototype_serialization_preserves_historical_constructor_compatibility() -> None:
    config = PrototypeMemoryConfig(feature_dim=2, n_classes=2)
    payload = config.to_config()
    payload["type"] = "historical-marker"
    payload["feature_dim"] = np.int32(2)
    payload["n_classes"] = np.uint16(2)
    payload["update_rate"] = np.float32(0.3)
    restored = PrototypeMemoryConfig.from_config(MappingProxyType(payload))
    assert restored.feature_dim == config.feature_dim
    assert restored.n_classes == config.n_classes
    assert restored.slots_per_class == config.slots_per_class
    assert restored.update_rate == float(np.float32(0.3))
    assert restored.novelty_threshold == config.novelty_threshold
    assert restored.bandwidth == config.bandwidth
    assert type(restored.feature_dim) is int
    assert type(restored.n_classes) is int
    assert type(restored.update_rate) is float

    partial = PrototypeMemoryConfig.from_config({"feature_dim": 2, "n_classes": 2})
    assert partial == config

    with pytest.raises(ValueError, match="extra"):
        PrototypeMemoryConfig.from_config({**config.to_config(), "extra": 1})

    learner_payload = PrototypeMemoryLearner(config).to_config()
    learner_payload["type"] = "historical-marker"
    learner_payload["extra"] = "ignored legacy metadata"
    learner = PrototypeMemoryLearner.from_config(MappingProxyType(learner_payload))
    assert learner.config == config


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("means", jnp.zeros((2, 2, 2), dtype=jnp.float32), "state.means"),
        ("counts", jnp.zeros((2, 2), dtype=jnp.float16), "state.counts"),
        ("last_update", jnp.zeros((2, 2), dtype=jnp.float32), "state.last_update"),
        ("step_count", jnp.zeros((1,), dtype=jnp.int32), "state.step_count"),
    ],
)
def test_prototype_public_methods_reject_malformed_state_before_tracing(
    field: str,
    replacement: jax.Array,
    message: str,
) -> None:
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=3, n_classes=2, slots_per_class=2)
    )
    state = dataclasses.replace(learner.init(), **{field: replacement})
    observation = jnp.zeros((3,), dtype=jnp.float32)
    target = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    for call in (
        lambda: learner.predict(state, observation),
        lambda: learner.update(state, observation, target),
        lambda: jax.jit(lambda s: learner.predict(s, observation))(state),
        lambda: jax.jit(lambda s: learner.update(s, observation, target))(state),
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            call()


@pytest.mark.parametrize(
    ("operand", "shape", "message"),
    [
        ("observation", (), "observation"),
        ("observation", (1, 3), "observation"),
        ("target", (), "target"),
        ("target", (1, 2), "target"),
        ("threshold", (1,), "novelty_threshold"),
    ],
)
def test_prototype_update_rejects_scalar_and_vector_shape_aliases(
    operand: str,
    shape: tuple[int, ...],
    message: str,
) -> None:
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=3, n_classes=2, slots_per_class=2)
    )
    state = learner.init()
    observation = jnp.zeros((3,), dtype=jnp.float32)
    target = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    threshold = jnp.asarray(0.1, dtype=jnp.float32)
    malformed = jnp.zeros(shape, dtype=jnp.float32)
    if operand == "observation":
        observation = malformed
    elif operand == "target":
        target = malformed
    else:
        threshold = malformed
    for call in (
        lambda: learner.update_with_novelty_threshold(
            state, observation, target, threshold
        ),
        lambda: jax.jit(
            lambda s, x, y, value: learner.update_with_novelty_threshold(
                s, x, y, value
            )
        )(state, observation, target, threshold),
    ):
        with pytest.raises(ValueError, match=message):
            call()


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int32, np.bool_])
def test_prototype_public_operations_reject_non_float32_operands(dtype: object) -> None:
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=3, n_classes=2, slots_per_class=2)
    )
    state = learner.init()
    observation = np.zeros((3,), dtype=dtype)
    target = np.zeros((2,), dtype=dtype)
    threshold = np.asarray(0, dtype=dtype)
    with pytest.raises(TypeError, match="observation.*dtype"):
        learner.predict(state, observation)
    with pytest.raises(TypeError, match="observation.*dtype"):
        learner.update(state, observation, jnp.zeros((2,), dtype=jnp.float32))
    with pytest.raises(TypeError, match="target.*dtype"):
        learner.update(state, jnp.zeros((3,), dtype=jnp.float32), target)
    with pytest.raises(TypeError, match="novelty_threshold.*dtype"):
        learner.update_with_novelty_threshold(
            state,
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.float32),
            threshold,
        )


def test_prototype_invalid_state_and_threshold_are_atomic_noops() -> None:
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=2)
    )
    state = learner.init()
    observation = jnp.zeros((2,), dtype=jnp.float32)
    target = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    invalid_states = (
        dataclasses.replace(state, counts=state.counts.at[0, 0].set(-1.0)),
        dataclasses.replace(state, last_update=state.last_update.at[0, 0].set(-1)),
        dataclasses.replace(state, last_update=state.last_update.at[0, 0].set(1)),
        dataclasses.replace(state, step_count=jnp.asarray(-1, dtype=jnp.int32)),
    )
    for invalid_state in invalid_states:
        result = learner.update(invalid_state, observation, target)
        assert not bool(result.update_applied)
        assert all(
            bool(equal)
            for equal in jax.tree.leaves(
                jax.tree.map(
                    lambda left, right: jnp.array_equal(left, right),
                    result.state,
                    invalid_state,
                )
            )
        )
    negative_threshold = learner.update_with_novelty_threshold(
        state,
        observation,
        target,
        jnp.asarray(-0.1, dtype=jnp.float32),
    )
    assert not bool(negative_threshold.update_applied)
    assert all(
        bool(equal)
        for equal in jax.tree.leaves(
            jax.tree.map(
                lambda left, right: jnp.array_equal(left, right),
                negative_threshold.state,
                state,
            )
        )
    )


def test_prototype_step_counter_saturates_without_invalidating_state() -> None:
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=2)
    )
    state = dataclasses.replace(
        learner.init(),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    result = learner.update(
        state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert int(result.state.step_count) == _INT32_MAX
    assert int(result.state.last_update[0, 0]) == _INT32_MAX


def test_prototype_array_runner_preflights_shapes_and_output_resources() -> None:
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=2)
    )
    with pytest.raises(ValueError, match="observations"):
        run_prototype_memory_arrays(
            learner,
            jnp.zeros((3, 1, 2), dtype=jnp.float32),
            jnp.zeros((3, 2), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="targets"):
        run_prototype_memory_arrays(
            learner,
            jnp.zeros((3, 2), dtype=jnp.float32),
            jnp.zeros((3, 1, 2), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="same step count"):
        run_prototype_memory_arrays(
            learner,
            jnp.zeros((3, 2), dtype=jnp.float32),
            jnp.zeros((2, 2), dtype=jnp.float32),
        )

    first_overflow = _INT32_MAX // (4 * (learner.config.n_classes + 6) + 1) + 1
    observations = jax.ShapeDtypeStruct((first_overflow, 2), jnp.float32)
    targets = jax.ShapeDtypeStruct((first_overflow, 2), jnp.float32)
    with pytest.raises(ValueError, match="byte count"):
        jax.eval_shape(
            lambda x, y: run_prototype_memory_arrays(learner, x, y),
            observations,
            targets,
        )


def test_prototype_array_runner_preflights_before_conversion() -> None:
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=2)
    )

    class OversizedArrayStub:
        shape = (_INT32_MAX, 2)
        dtype = np.dtype(np.float32)

        def __jax_array__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("conversion must not run")

    stub = OversizedArrayStub()
    with pytest.raises(ValueError, match="scalar count|byte count"):
        run_prototype_memory_arrays(learner, stub, stub)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int32, np.bool_])
def test_prototype_array_runner_rejects_non_float32_inputs(dtype: object) -> None:
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=2)
    )
    with pytest.raises(TypeError, match="observations.*dtype"):
        run_prototype_memory_arrays(
            learner,
            np.zeros((2, 2), dtype=dtype),
            np.zeros((2, 2), dtype=np.float32),
        )
    with pytest.raises(TypeError, match="targets.*dtype"):
        run_prototype_memory_arrays(
            learner,
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=dtype),
        )
