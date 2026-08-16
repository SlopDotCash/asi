"""Validation hardening for prototype memory (int/float bounds + resource preflights)."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.core.prototype_memory import PrototypeMemoryConfig

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
    # Minimal sizes: n_classes=2, feature_dim=1, vary slots_per_class.
    # total = 6*slots +1, byte = 24*slots+4
    last_legal = (2**31 - 1 - 4) // 24
    PrototypeMemoryConfig(feature_dim=1, n_classes=2, slots_per_class=last_legal)
    with pytest.raises(ValueError, match="byte count"):
        PrototypeMemoryConfig(feature_dim=1, n_classes=2, slots_per_class=last_legal + 1)
    # Non-minimal vector should also be allocation-free.
    with pytest.raises(ValueError, match="byte count|scalar count|dimensions"):
        PrototypeMemoryConfig(feature_dim=5000, n_classes=5000, slots_per_class=5000)


def test_prototype_state_preflight_feature_dim_boundary() -> None:
    # Vary feature_dim with n_classes=2, slots=1: byte = 8*fd+20
    last_legal = (2**31 - 1 - 20) // 8
    PrototypeMemoryConfig(feature_dim=last_legal, n_classes=2, slots_per_class=1)
    with pytest.raises(ValueError, match="byte count"):
        PrototypeMemoryConfig(feature_dim=last_legal + 1, n_classes=2, slots_per_class=1)
