"""Validation hardening for associative memory (int/float bounds + resources)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.associative_memory import (
    AssociativeMemoryConfig,
    AssociativeMemoryLearner,
    _require_resource,
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
    # Simplified total = max_features*vocab + 8*max_features + vocab + suffix +5
    # With vocab=2,suffix=2 -> total=10*max+9, persistent=40*max+36
    last_legal = (2**31 - 1 - 36) // 40
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


def test_associative_static_pair_indices_preflight_before_python_lists() -> None:
    with pytest.raises(ValueError, match="static indices.*scalar count"):
        _base_cfg(block_size=50_000, suffix_length=50_000, max_features=1)


def test_associative_lookup_and_gather_products_are_preflighted() -> None:
    with pytest.raises(ValueError, match="lookup workspace.*scalar count"):
        _base_cfg(block_size=1_000, suffix_length=2, max_features=400_000)
    with pytest.raises(ValueError, match="gathered rows.*byte count"):
        _base_cfg(
            vocab_size=1_000_000,
            block_size=1_000,
            suffix_length=2,
            max_features=1,
        )


def test_associative_adaptive_window_product_is_conditional() -> None:
    # Other exact products fit; the pair-by-window matrix alone exceeds the bound.
    _base_cfg(block_size=1_050, suffix_length=1_050, max_features=1)
    with pytest.raises(ValueError, match="adaptive-window workspace.*byte count"):
        _base_cfg(
            block_size=1_050,
            suffix_length=1_050,
            max_features=1,
            adaptive_window=True,
        )


def test_associative_resource_endpoint_is_exact_and_allocation_free() -> None:
    _require_resource("endpoint", float32_scalars=_INT32_MAX // 4)
    with pytest.raises(ValueError, match="byte count"):
        _require_resource("endpoint", float32_scalars=_INT32_MAX // 4 + 1)
    with pytest.raises(ValueError, match="scalar count"):
        _require_resource("endpoint", bool_scalars=_INT32_MAX + 1)


def test_associative_serialized_schema_is_exact() -> None:
    learner = AssociativeMemoryLearner(_base_cfg())
    config_payload = learner.config.to_config()
    learner_payload = learner.to_config()
    assert AssociativeMemoryConfig.from_config(config_payload) == learner.config
    assert AssociativeMemoryLearner.from_config(learner_payload).config == learner.config

    for payload in (
        {**config_payload, "extra": 1},
        {key: value for key, value in config_payload.items() if key != "vocab_size"},
        {**config_payload, "type": "Wrong"},
        {**config_payload, "vocab_size": np.int32(4)},
    ):
        with pytest.raises(ValueError):
            AssociativeMemoryConfig.from_config(payload)
    with pytest.raises(ValueError):
        AssociativeMemoryLearner.from_config({**learner_payload, "extra": 1})


def test_associative_public_shapes_and_dtypes_fail_before_tracing() -> None:
    learner = AssociativeMemoryLearner(_base_cfg())
    state = learner.init()
    with pytest.raises(ValueError, match="context must have shape"):
        learner.predict(state, jnp.zeros((7,), dtype=jnp.int32))
    with pytest.raises(TypeError, match="context must have dtype int32"):
        learner.predict(state, jnp.zeros((8,), dtype=jnp.float32))
    with pytest.raises(ValueError, match="state.values must have shape"):
        learner.predict(
            state.replace(values=jnp.zeros((4, 3), dtype=jnp.float32)),
            jnp.zeros((8,), dtype=jnp.int32),
        )
    with pytest.raises(ValueError, match="labels must have shape"):
        run_associative_memory_arrays(
            learner,
            state,
            jnp.zeros((2, 8), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
        )


def test_associative_counters_saturate_without_wrapping() -> None:
    learner = AssociativeMemoryLearner(
        _base_cfg(
            block_size=2,
            suffix_length=2,
            max_features=1,
            feature_family="position_token",
        )
    )
    state = learner.init().replace(
        allocations=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        replacements=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    result = learner.update(
        state,
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    assert bool(result.update_applied)
    assert int(result.state.allocations) == _INT32_MAX
    assert int(result.state.replacements) == _INT32_MAX
    assert int(result.state.step_count) == _INT32_MAX


def test_associative_invalid_adopted_counter_rolls_back_transaction() -> None:
    learner = AssociativeMemoryLearner(_base_cfg())
    state = learner.init().replace(step_count=jnp.asarray(-1, dtype=jnp.int32))
    result = learner.update(
        state,
        jnp.zeros((8,), dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )
    assert not bool(result.update_applied)
    assert int(result.state.step_count) == -1
