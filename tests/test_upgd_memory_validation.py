"""Validation hardening for UPGD memory (int/float bounds + resources)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.upgd_memory import (
    UPGDMemoryConfig,
    UPGDMemoryLearner,
    _combined_state_resource_counts,
    _require_resource,
    run_upgd_memory_arrays,
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


class _RaisingMetadata:
    @property
    def shape(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("shape hook must not escape")

    @property
    def dtype(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("dtype hook must not escape")


class _RaisingEquality:
    def __eq__(self, other: object) -> bool:  # pragma: no cover
        raise RuntimeError("equality hook must not run")


def _base_cfg(**overrides: object) -> UPGDMemoryConfig:
    base: dict[str, object] = {
        "feature_dim": 4,
        "n_heads": 2,
        "hidden_sizes": (4,),
    }
    base.update(overrides)
    return UPGDMemoryConfig(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _base_cfg(feature_dim=v),
        lambda v: _base_cfg(n_heads=v),
        lambda v: _base_cfg(slots_per_class=v),
        lambda v: _base_cfg(upgd_head_loss_pressure_warmup_steps=v),
        lambda v: _base_cfg(upgd_head_repetition_warmup_steps=v),
    ],
)
def test_upgd_memory_int_validators_reject_hostile_subclass_without_running_hook(
    ctor,
) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        ctor(_LyingIntSubclass(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        ctor(_RaisingIntSubclass(4))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _base_cfg(feature_dim=v),
        lambda v: _base_cfg(n_heads=v),
    ],
)
def test_upgd_memory_int_validators_do_not_run_repr_hook(ctor) -> None:
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
def test_upgd_memory_int_validators_canonicalize_numpy_scalars(
    np_type: type,
) -> None:
    cfg = UPGDMemoryConfig(
        feature_dim=np_type(4),
        n_heads=np_type(2),
        hidden_sizes=(np_type(4),),  # type: ignore[arg-type]
        slots_per_class=np_type(4),
        upgd_head_loss_pressure_warmup_steps=np_type(1),
        upgd_head_repetition_warmup_steps=np_type(1),
    )
    assert cfg.feature_dim == 4
    assert type(cfg.feature_dim) is int
    assert type(cfg.n_heads) is int
    assert type(cfg.slots_per_class) is int


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _base_cfg(feature_dim=v),
        lambda v: _base_cfg(n_heads=v),
        lambda v: _base_cfg(slots_per_class=v),
    ],
)
@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), 4.0, np.float64(4.0), "4", None, 0, -1, _INT32_MAX + 1],
)
def test_upgd_memory_int_validators_reject_non_integer_and_out_of_range(
    ctor,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="must be"):
        ctor(value)  # type: ignore[arg-type]


def test_upgd_memory_float_validators_reject_nonfinite_and_hostile() -> None:
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
        ("upgd_step_size", float("nan")),
        ("upgd_step_size", float("inf")),
        ("upgd_step_size", 0.0),
        ("upgd_step_size", -1.0),
        ("upgd_step_size", HostileFloat(0.5)),
        ("memory_update_rate", -0.1),
        ("memory_update_rate", 1.5),
        ("memory_update_rate", ClassSpoof()),  # type: ignore[arg-type]
        ("memory_bandwidth", 0.0),
        ("memory_bandwidth", float("nan")),
        ("initial_novelty_threshold", 0.0),
        ("reliability_decay", -0.1),
        ("reliability_decay", 1.0),
        ("target_trace_blend_scale", -0.1),
        ("target_trace_blend_scale", 1.5),
        ("novelty_adaptation_rate", -0.1),
        ("target_allocation_rate", -0.1),
        ("target_allocation_rate", 1.5),
        ("min_novelty_threshold", 0.0),
        ("max_novelty_threshold", 0.0),
    ]:
        with pytest.raises(ValueError, match=field):
            _base_cfg(**{field: bad})  # type: ignore[arg-type]


def test_upgd_memory_float_validators_reject_hostile_ratio() -> None:
    class HostileFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            type(self).calls += 1
            raise RuntimeError("ratio hook")

    with pytest.raises(ValueError, match="upgd_step_size"):
        _base_cfg(upgd_step_size=HostileFloat(1.0))  # type: ignore[arg-type]
    assert HostileFloat.calls == 1


def test_upgd_memory_dimensions_preflight_without_allocation() -> None:
    with pytest.raises(ValueError, match="dimensions must fit signed|scalar count|byte count"):
        _base_cfg(feature_dim=_INT32_MAX, n_heads=2, slots_per_class=2)
    with pytest.raises(ValueError, match="dimensions must fit signed|scalar count|byte count"):
        _base_cfg(feature_dim=50000, n_heads=50000, slots_per_class=50000)


def test_upgd_memory_state_preflight_bytes_without_allocation() -> None:
    with pytest.raises(ValueError, match="persistent state"):
        _base_cfg(feature_dim=2, n_heads=2, slots_per_class=20_000_000)
    with pytest.raises(ValueError, match="dimensions|scalar|byte|persistent"):
        _base_cfg(feature_dim=5000, n_heads=100, slots_per_class=500_000)


def test_upgd_memory_float_validators_accept_valid_values() -> None:
    cfg = _base_cfg(
        upgd_step_size=0.03,
        memory_update_rate=0.3,
        memory_bandwidth=0.01,
        reliability_decay=0.98,
        target_trace_blend_scale=0.8,
        target_allocation_rate=0.18,
    )
    assert cfg.upgd_step_size == 0.03
    assert cfg.memory_update_rate == 0.3


@pytest.mark.parametrize(
    "field",
    [
        "initial_novelty_threshold",
        "memory_bandwidth",
        "min_novelty_threshold",
        "max_novelty_threshold",
    ],
)
def test_upgd_memory_log_division_fields_require_positive_normal_float32(field: str) -> None:
    minimum = np.finfo(np.float32).tiny
    overrides = {field: minimum}
    if field == "max_novelty_threshold":
        overrides["min_novelty_threshold"] = minimum
    assert getattr(_base_cfg(**overrides), field) == float(minimum)
    with pytest.raises(ValueError, match=field):
        _base_cfg(**{field: np.nextafter(np.float32(minimum), np.float32(0.0))})
    with pytest.raises(ValueError, match=field):
        _base_cfg(**{field: np.longdouble("1e-500")})


def test_upgd_memory_combined_state_count_includes_all_components() -> None:
    # F=2,D=2,H=(4,),slots/class=3: 63+6*3 floats, 3+2*3 ints, two key words.
    assert _combined_state_resource_counts(2, 2, (4,), 3) == (81, 9, 2)
    with pytest.raises(ValueError, match="combined state.*(scalar|byte) count"):
        _base_cfg(feature_dim=50_000, hidden_sizes=(50_000,), slots_per_class=1)


def test_upgd_memory_combined_resource_formula_matches_materialized_state() -> None:
    for hidden_sizes in ((), (4,), (3, 5)):
        learner = UPGDMemoryLearner(_base_cfg(hidden_sizes=hidden_sizes, slots_per_class=3))
        state = learner.init(jax.random.key(1))
        measured_bytes = sum(int(leaf.nbytes) for leaf in jax.tree_util.tree_leaves(state))
        counts = _combined_state_resource_counts(4, 2, hidden_sizes, 3)
        assert measured_bytes == 4 * sum(counts)


def test_upgd_memory_resource_endpoint_is_exact_and_allocation_free() -> None:
    scalar_budget = _INT32_MAX // 4
    _require_resource("endpoint", float32_scalars=scalar_budget)
    with pytest.raises(ValueError, match="byte count"):
        _require_resource("endpoint", float32_scalars=scalar_budget + 1)
    with pytest.raises(ValueError, match="scalar count"):
        _require_resource("endpoint", bool_scalars=_INT32_MAX + 1)


def test_upgd_memory_historical_mapping_envelopes_are_safe_and_compatible() -> None:
    learner = UPGDMemoryLearner(_base_cfg())
    config_payload = learner.config.to_config()
    learner_payload = learner.to_config()
    assert UPGDMemoryConfig.from_config(MappingProxyType(config_payload)) == learner.config
    assert UPGDMemoryLearner.from_config(MappingProxyType(learner_payload)).config == learner.config
    partial = UPGDMemoryConfig.from_config(
        MappingProxyType({"type": "historical", "feature_dim": 4, "n_heads": 2})
    )
    assert partial.hidden_sizes == (64,)
    assert UPGDMemoryConfig.from_config(
        {"feature_dim": np.int32(4), "n_heads": np.int32(2), "hidden_sizes": ()}
    ).feature_dim == 4

    class HostileMapping(Mapping):
        def __getitem__(self, key):  # type: ignore[no-untyped-def]
            raise RuntimeError("hostile mapping hook")

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("hostile mapping hook")

        def __len__(self) -> int:
            return 2

    with pytest.raises(ValueError, match="could not be read"):
        UPGDMemoryConfig.from_config(HostileMapping())
    with pytest.raises(ValueError, match="list or tuple"):
        UPGDMemoryConfig.from_config(
            {"feature_dim": 4, "n_heads": 2, "hidden_sizes": "4"}
        )
    with pytest.raises(ValueError, match="type marker"):
        UPGDMemoryConfig.from_config(
            {"type": _RaisingEquality(), "feature_dim": 4, "n_heads": 2}
        )
    with pytest.raises(ValueError, match="type marker"):
        UPGDMemoryLearner.from_config(
            {"type": _RaisingEquality(), "config": config_payload}
        )


@pytest.mark.parametrize("implementation", ["rbg", "unsafe_rbg"])
def test_upgd_memory_requires_exact_threefry_key_resource_contract(
    implementation: str,
) -> None:
    learner = UPGDMemoryLearner(_base_cfg())
    incompatible = jax.random.key(0, impl=implementation)
    with pytest.raises(TypeError, match="threefry2x32"):
        learner.init(incompatible)
    state = learner.init(jax.random.key(0))
    with pytest.raises(TypeError, match="threefry2x32"):
        learner.predict(
            state.replace(
                upgd_state=state.upgd_state.replace(key=incompatible)  # type: ignore[attr-defined]
            ),
            jnp.ones((4,), dtype=jnp.float32),
        )


def test_upgd_memory_validates_nested_and_wrapper_state_metadata() -> None:
    learner = UPGDMemoryLearner(_base_cfg())
    state = learner.init(jax.random.key(0))
    observation = jnp.ones((4,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="state.memory_logit must have shape"):
        learner.predict(
            state.replace(memory_logit=jnp.zeros((1,), dtype=jnp.float32)),
            observation,
        )
    malformed_upgd = state.upgd_state.replace(  # type: ignore[attr-defined]
        utilities=(jnp.zeros((4, 3), dtype=jnp.float32),)
    )
    with pytest.raises(ValueError, match=r"utilities\[0\].*shape"):
        learner.predict(state.replace(upgd_state=malformed_upgd), observation)
    with pytest.raises(TypeError, match="valid array shape and dtype metadata"):
        learner.predict(
            state.replace(memory_logit=_RaisingMetadata()),  # type: ignore[arg-type]
            observation,
        )


def test_upgd_memory_public_input_metadata_fails_before_tracing() -> None:
    learner = UPGDMemoryLearner(_base_cfg())
    state = learner.init(jax.random.key(0))
    with pytest.raises(ValueError, match="observation must have shape"):
        learner.predict(state, jnp.ones((3,), dtype=jnp.float32))
    with pytest.raises(TypeError, match="observation must have dtype"):
        learner.predict(state, jnp.ones((4,), dtype=jnp.int32))
    with pytest.raises(ValueError, match="target must have shape"):
        learner.update(
            state,
            jnp.ones((4,), dtype=jnp.float32),
            jnp.ones((3,), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="targets must have shape"):
        run_upgd_memory_arrays(
            learner,
            state,
            jnp.ones((2, 4), dtype=jnp.float32),
            jnp.ones((1, 2), dtype=jnp.float32),
        )
    with pytest.raises(TypeError, match="observations must expose valid array metadata"):
        run_upgd_memory_arrays(
            learner,
            state,
            _RaisingMetadata(),  # type: ignore[arg-type]
            jnp.ones((1, 2), dtype=jnp.float32),
        )


def test_upgd_memory_scan_preflights_aggregate_working_set_without_allocation() -> None:
    learner = UPGDMemoryLearner(_base_cfg())
    state = learner.init(jax.random.key(0))

    class HugeObservations:
        shape = (10_000_000, 4)
        dtype = jnp.float32

    class HugeTargets:
        shape = (10_000_000, 2)
        dtype = jnp.float32

    with pytest.raises(ValueError, match="working set"):
        run_upgd_memory_arrays(
            learner,
            state,
            HugeObservations(),  # type: ignore[arg-type]
            HugeTargets(),  # type: ignore[arg-type]
        )


def test_upgd_memory_all_lifetime_counters_saturate() -> None:
    learner = UPGDMemoryLearner(_base_cfg())
    initial = learner.init(jax.random.key(0))
    state = initial.replace(
        upgd_state=initial.upgd_state.replace(  # type: ignore[attr-defined]
            step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        ),
        memory_state=initial.memory_state.replace(  # type: ignore[attr-defined]
            step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
        ),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    result = learner.update(
        state,
        jnp.ones((4,), dtype=jnp.float32),
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert int(result.state.step_count) == _INT32_MAX
    assert int(result.state.upgd_state.step_count) == _INT32_MAX
    assert int(result.state.memory_state.step_count) == _INT32_MAX


def test_upgd_memory_negative_adopted_counter_rolls_back() -> None:
    learner = UPGDMemoryLearner(_base_cfg())
    state = learner.init(jax.random.key(0)).replace(
        step_count=jnp.asarray(-1, dtype=jnp.int32)
    )
    result = learner.update(
        state,
        jnp.ones((4,), dtype=jnp.float32),
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
    )
    assert not bool(result.update_applied)
    assert int(result.state.step_count) == -1


def test_upgd_memory_finite_operation_overflow_rolls_back_atomically() -> None:
    learner = UPGDMemoryLearner(_base_cfg())
    state = learner.init(jax.random.key(7))
    result = learner.update(
        state,
        jnp.zeros((4,), dtype=jnp.float32),
        jnp.asarray([np.finfo(np.float32).max, 0.0], dtype=jnp.float32),
    )
    assert not bool(result.update_applied)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda left, right: jnp.array_equal(left, right),
            result.state,
            state,
        )
    )
