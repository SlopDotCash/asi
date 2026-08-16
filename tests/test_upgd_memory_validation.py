"""Validation hardening for UPGD memory (int/float bounds + resources)."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.core.upgd_memory import UPGDMemoryConfig

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
    # total_means = n_heads*slots*feature_dim, fixed=2*n_heads*slots+9
    # With n_heads=2, slots var, feature_dim=2 -> total_state=8*slots+9
    # persistent=32slots+36 -> last_legal=(INT32-36)//32
    last_legal = (2**31 - 1 - 36) // 32
    _base_cfg(feature_dim=2, n_heads=2, slots_per_class=last_legal)
    with pytest.raises(ValueError, match="scalar count|byte count"):
        _base_cfg(feature_dim=2, n_heads=2, slots_per_class=last_legal + 1)
    with pytest.raises(ValueError, match="dimensions must fit signed|scalar count|byte count"):
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
