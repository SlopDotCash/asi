"""Validation hardening for associative memory (int/float bounds + resources)."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.associative_memory import (
    AssociativeMemoryConfig,
    AssociativeMemoryLearner,
    run_associative_memory_arrays,
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


def _base_cfg(**overrides: object) -> AssociativeMemoryConfig:
    base: dict[str, object] = {
        "vocab_size": 4,
        "block_size": 8,
        "suffix_length": 2,
        "max_features": 4,
    }
    base.update(overrides)
    return AssociativeMemoryConfig(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _base_cfg(vocab_size=v),
        lambda v: _base_cfg(block_size=v),
        lambda v: _base_cfg(suffix_length=v),
        lambda v: _base_cfg(max_features=v),
        lambda v: _base_cfg(min_effective_budget=v),
    ],
)
def test_associative_int_validators_reject_hostile_subclass_without_running_hook(
    ctor,
) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        ctor(_LyingIntSubclass(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        ctor(_RaisingIntSubclass(4))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _base_cfg(vocab_size=v),
        lambda v: _base_cfg(block_size=v),
    ],
)
def test_associative_int_validators_do_not_run_repr_hook(ctor) -> None:
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
def test_associative_int_validators_canonicalize_numpy_scalars(
    np_type: type,
) -> None:
    cfg = AssociativeMemoryConfig(
        vocab_size=np_type(8),
        block_size=np_type(8),
        suffix_length=np_type(2),
        max_features=np_type(16),
        min_effective_budget=np_type(1),
    )
    assert cfg.vocab_size == 8
    assert type(cfg.vocab_size) is int
    assert type(cfg.block_size) is int
    assert type(cfg.max_features) is int
    assert type(cfg.min_effective_budget) is int


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _base_cfg(vocab_size=v),
        lambda v: _base_cfg(block_size=v),
        lambda v: _base_cfg(max_features=v),
    ],
)
@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), 4.0, np.float64(4.0), "4", None, 0, -1, _INT32_MAX + 1],
)
def test_associative_int_validators_reject_non_integer_and_out_of_range(
    ctor,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="must be"):
        ctor(value)  # type: ignore[arg-type]


def test_associative_float_validators_reject_nonfinite_and_hostile() -> None:
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
        ("write_lr", float("nan")),
        ("write_lr", float("inf")),
        ("write_lr", 0.0),
        ("write_lr", -1.0),
        ("write_lr", HostileFloat(0.5)),
        ("retention", -0.1),
        ("retention", 1.5),
        ("retention", ClassSpoof()),  # type: ignore[arg-type]
        ("utility_decay", -0.1),
        ("utility_decay", 1.5),
        ("min_weight", 0.0),
        ("max_weight", 0.0),
        ("logit_scale", 0.0),
        ("scope_lr", -0.1),
        ("budget_lr", -0.1),
        ("initial_budget_fraction", 0.0),
        ("initial_budget_fraction", 1.5),
        ("scope_logit_clip", 0.0),
    ]:
        with pytest.raises(ValueError, match=field):
            _base_cfg(**{field: bad})  # type: ignore[arg-type]


def test_associative_float_validators_reject_hostile_ratio() -> None:
    class HostileFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            type(self).calls += 1
            raise RuntimeError("ratio hook")

    with pytest.raises(ValueError, match="write_lr"):
        _base_cfg(write_lr=HostileFloat(1.0))  # type: ignore[arg-type]
    assert HostileFloat.calls == 1


def test_associative_dimensions_preflight_without_allocation() -> None:
    with pytest.raises(ValueError, match="dimensions must fit signed|scalar count|byte count"):
        _base_cfg(vocab_size=_INT32_MAX, max_features=2, block_size=_INT32_MAX)
    with pytest.raises(ValueError, match="dimensions must fit signed|scalar count|byte count"):
        _base_cfg(vocab_size=50000, max_features=50000, block_size=8)


def test_associative_state_preflight_bytes_without_allocation() -> None:
    # With vocab=block=suffix=2, the conservative update bound is 34*max+244.
    last_legal = (_INT32_MAX // 4 - 244) // 34
    _base_cfg(
        vocab_size=2,
        block_size=2,
        suffix_length=2,
        max_features=last_legal,
    )
    with pytest.raises(ValueError, match="scalar count|byte count"):
        _base_cfg(
            vocab_size=2,
            block_size=2,
            suffix_length=2,
            max_features=last_legal + 1,
        )
    with pytest.raises(ValueError, match="dimensions must fit signed|scalar count|byte count"):
        _base_cfg(vocab_size=5000, block_size=8, suffix_length=2, max_features=500_000)


def test_associative_float_validators_accept_valid_values() -> None:
    cfg = _base_cfg(
        write_lr=1.0,
        retention=0.9,
        utility_decay=0.99,
        min_weight=0.1,
        max_weight=2.0,
        logit_scale=1.0,
        scope_lr=0.05,
        budget_lr=0.05,
        initial_budget_fraction=0.5,
        scope_logit_clip=8.0,
    )
    assert cfg.write_lr == 1.0
    assert cfg.retention == 0.9


def test_associative_pair_descriptors_preflight_before_learner_construction() -> None:
    with pytest.raises(
        ValueError, match="pair count|active feature|feature-key|descriptor|query"
    ):
        AssociativeMemoryConfig(
            vocab_size=2,
            block_size=50_000,
            suffix_length=50_000,
            max_features=1,
        )


def test_associative_serialization_preserves_historical_compatibility() -> None:
    config = _base_cfg()
    payload = config.to_config()
    payload["type"] = "historical-marker"
    payload["vocab_size"] = np.int32(config.vocab_size)
    restored = AssociativeMemoryConfig.from_config(MappingProxyType(payload))
    assert restored == config
    partial = AssociativeMemoryConfig.from_config(
        {"vocab_size": 4, "block_size": 8, "suffix_length": 2, "max_features": 4}
    )
    assert partial == config
    learner_payload = AssociativeMemoryLearner(config).to_config()
    learner_payload["type"] = "historical-marker"
    learner_payload["metadata"] = 1
    assert AssociativeMemoryLearner.from_config(learner_payload).config == config
    with pytest.raises(ValueError, match="serialized AssociativeMemoryConfig"):
        AssociativeMemoryConfig.from_config({**config.to_config(), "unknown": 1})


def test_associative_public_contracts_and_counters() -> None:
    learner = AssociativeMemoryLearner(_base_cfg())
    state = learner.init()
    context = jnp.zeros((8,), dtype=jnp.int32)
    label = jnp.asarray(1, dtype=jnp.int32)
    with pytest.raises(ValueError, match="context"):
        learner.predict(state, jnp.zeros((1, 8), dtype=jnp.int32))
    with pytest.raises(TypeError, match="context"):
        learner.predict(state, jnp.zeros((8,), dtype=jnp.int16))
    malformed = dataclasses.replace(
        state, values=jnp.zeros((4, 4), dtype=jnp.float16)
    )
    with pytest.raises(TypeError, match="state.values"):
        learner.update(malformed, context, label)

    maximum = dataclasses.replace(
        state,
        allocations=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        replacements=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    result = learner.update(maximum, context, label)
    assert bool(result.update_applied)
    assert int(result.state.step_count) == _INT32_MAX
    assert int(result.state.allocations) == _INT32_MAX
    assert int(result.state.replacements) == _INT32_MAX


def test_associative_scan_preflight_precedes_conversion() -> None:
    learner = AssociativeMemoryLearner(_base_cfg())
    first_overflow = _INT32_MAX // (4 * (learner.config.vocab_size + 9)) + 1

    class HostArray:
        dtype = np.dtype(np.int32)

        def __init__(self, shape: tuple[int, ...]):
            self.shape = shape

        def __jax_array__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("conversion must not run")

    with pytest.raises(ValueError, match="scalar count|byte count"):
        run_associative_memory_arrays(
            learner,
            learner.init(),
            HostArray((first_overflow, 8)),  # type: ignore[arg-type]
            HostArray((first_overflow,)),  # type: ignore[arg-type]
        )
