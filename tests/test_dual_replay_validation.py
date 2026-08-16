"""Validation hardening for dual replay (int/float bounds + resource preflight)."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.core.dual_replay import DualReplayConfig

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


def _base_cfg(**overrides):  # type: ignore[no-untyped-def]
    vals = dict(
        total_capacity=6,
        short_term_capacity=2,
        observation_dim=2,
        action_dim=2,
        short_term_sample_size=1,
        long_term_sample_size=1,
    )
    vals.update(overrides)
    return DualReplayConfig(**vals)


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _base_cfg(total_capacity=v),
        lambda v: _base_cfg(short_term_capacity=v),
        lambda v: _base_cfg(observation_dim=v),
        lambda v: _base_cfg(action_dim=v),
        lambda v: _base_cfg(short_term_sample_size=v),
        lambda v: _base_cfg(long_term_sample_size=v),
    ],
)
def test_dual_int_validators_reject_hostile_subclass_without_running_hook(ctor) -> None:
    with pytest.raises(ValueError, match=r"must be an integer in \["):
        ctor(_LyingIntSubclass(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"must be an integer in \["):
        ctor(_RaisingIntSubclass(4))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _base_cfg(max_representation_lag=v),
        lambda v: _base_cfg(max_persistent_bytes=v),
    ],
)
def test_dual_optional_int_validators_reject_hostile_subclass_without_running_hook(
    ctor,
) -> None:
    with pytest.raises(ValueError, match=r"must be an integer in \["):
        ctor(_LyingIntSubclass(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"must be an integer in \["):
        ctor(_RaisingIntSubclass(4))  # type: ignore[arg-type]


def test_dual_int_validators_do_not_run_repr_hook() -> None:
    with pytest.raises(ValueError):
        _base_cfg(total_capacity=_RaisingRepr())  # type: ignore[arg-type]


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
def test_dual_int_validators_canonicalize_numpy_scalars(np_type: type) -> None:
    cfg = _base_cfg(
        total_capacity=np_type(7),
        short_term_capacity=np_type(3),
        observation_dim=np_type(2),
        action_dim=np_type(2),
        short_term_sample_size=np_type(1),
        long_term_sample_size=np_type(1),
        max_representation_lag=np_type(0),
    )
    assert cfg.total_capacity == 7
    assert type(cfg.total_capacity) is int
    assert type(cfg.observation_dim) is int


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _base_cfg(total_capacity=v),
        lambda v: _base_cfg(observation_dim=v),
    ],
)
@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), 4.0, np.float64(4.0), "4", None, 0, -1, _INT32_MAX + 1, 10**100],
)
def test_dual_int_validators_reject_non_integer_and_out_of_range(ctor, value: object) -> None:
    with pytest.raises(ValueError, match=r"must be an integer in \["):
        ctor(value)  # type: ignore[arg-type]


def test_dual_float_validators_reject_hostile_and_nonfinite() -> None:
    class HostileFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            raise RuntimeError("ratio hook")

    class ClassSpoof:
        @property
        def __class__(self):  # type: ignore[no-untyped-def]
            return float

        def __float__(self) -> float:  # pragma: no cover
            return 0.1

    for field, bad in [
        ("surprise_scale", float("nan")),
        ("surprise_scale", HostileFloat(1.0)),  # type: ignore[arg-type]
        ("coverage_scale", float("inf")),
        ("progress_scale", ClassSpoof()),  # type: ignore[arg-type]
        ("calibrated_priority_threshold", float("nan")),
        ("calibrated_replacement_margin", HostileFloat(0.1)),  # type: ignore[arg-type]
    ]:
        with pytest.raises(ValueError, match=field):
            _base_cfg(**{field: bad})  # type: ignore[arg-type]


def test_dual_dimensions_preflight_without_allocation() -> None:
    with pytest.raises(ValueError, match="dimensions must fit signed int32"):
        _base_cfg(observation_dim=_INT32_MAX, action_dim=2)
    with pytest.raises(ValueError, match="scalar count|byte count"):
        _base_cfg(
            total_capacity=2,
            short_term_capacity=1,
            observation_dim=536870912,
            action_dim=1,
        )


def test_dual_state_preflight_bytes_without_allocation() -> None:
    last_legal = (2**31 - 1 - 60) // 79
    _base_cfg(
        total_capacity=last_legal,
        short_term_capacity=1,
        observation_dim=1,
        action_dim=1,
    )
    with pytest.raises(ValueError, match="byte count"):
        _base_cfg(
            total_capacity=last_legal + 1,
            short_term_capacity=1,
            observation_dim=1,
            action_dim=1,
        )
    with pytest.raises(ValueError, match="byte count|scalar count"):
        _base_cfg(
            total_capacity=100_000,
            short_term_capacity=1,
            observation_dim=5000,
            action_dim=5000,
        )
