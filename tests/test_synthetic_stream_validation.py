"""Validation hardening for synthetic streams (issue synthetic int/float bounds)."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.streams.synthetic import (
    DynamicScaleShiftStream,
    HiddenStateAR2Stream,
    ScaleDriftStream,
    SuttonExperiment1Stream,
    _require_float32_resource,
    make_scale_range,
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
        lambda v: HiddenStateAR2Stream(feature_dim=v, visible_dim=2),
        lambda v: HiddenStateAR2Stream(feature_dim=8, visible_dim=v),
        lambda v: SuttonExperiment1Stream(num_relevant=v),
        lambda v: SuttonExperiment1Stream(num_irrelevant=v),
        lambda v: SuttonExperiment1Stream(change_interval=v),
        lambda v: DynamicScaleShiftStream(feature_dim=v),
        lambda v: ScaleDriftStream(feature_dim=v),
        lambda v: make_scale_range(feature_dim=v),
    ],
)
def test_synthetic_int_validators_reject_hostile_subclass_without_running_hook(
    ctor,
) -> None:
    with pytest.raises(ValueError, match=r"must be a positive integer in \[1, 2147483647\]"):
        ctor(_LyingIntSubclass(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"must be a positive integer in \[1, 2147483647\]"):
        ctor(_RaisingIntSubclass(4))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: HiddenStateAR2Stream(feature_dim=v, visible_dim=2),
        lambda v: SuttonExperiment1Stream(num_relevant=v),
    ],
)
def test_synthetic_int_validators_do_not_run_repr_hook(ctor) -> None:
    with pytest.raises(ValueError, match=r"must be a positive integer in \[1, 2147483647\]"):
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
def test_synthetic_int_validators_canonicalize_numpy_scalars(np_type: type) -> None:
    s = HiddenStateAR2Stream(feature_dim=np_type(8), visible_dim=np_type(2))
    assert s.feature_dim == 8
    assert type(s.feature_dim) is int
    t = SuttonExperiment1Stream(num_relevant=np_type(3), num_irrelevant=np_type(4))
    assert t.feature_dim == 7


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: HiddenStateAR2Stream(feature_dim=v, visible_dim=2),
        lambda v: SuttonExperiment1Stream(num_relevant=v),
        lambda v: DynamicScaleShiftStream(feature_dim=v),
        lambda v: make_scale_range(feature_dim=v),
    ],
)
@pytest.mark.parametrize(
    "value", [True, np.bool_(True), 4.0, np.float64(4.0), "4", None, 0, -1, _INT32_MAX + 1, 10**100]
)
def test_synthetic_int_validators_reject_non_integer_and_out_of_range(ctor, value: object) -> None:
    with pytest.raises(ValueError, match=r"must be a positive integer in \[1, 2147483647\]"):
        ctor(value)  # type: ignore[arg-type]


def test_hidden_ar2_float_validators_reject_nonfinite_and_hostile() -> None:
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
        ("phi1", float("nan")),
        ("phi1", float("inf")),
        ("phi2", HostileFloat(0.5)),
        ("innovation_std", -0.1),
        ("innovation_std", float("nan")),
        ("nonlinear_coeff", ClassSpoof()),  # type: ignore[arg-type]
        ("target_noise_std", -1.0),
    ]:
        with pytest.raises(ValueError, match=field):
            HiddenStateAR2Stream(feature_dim=8, visible_dim=2, **{field: bad})  # type: ignore[arg-type]


def test_sutton_float_validators_reject_nonfinite() -> None:
    for field, bad in [
        ("noise_std", float("nan")),
        ("noise_std", float("inf")),
        ("bias_drift_rate", -0.1),
    ]:
        with pytest.raises(ValueError, match=field):
            SuttonExperiment1Stream(**{field: bad})  # type: ignore[arg-type]


def test_scale_drift_float_validators_reject_nonfinite() -> None:
    for field, bad in [
        ("weight_drift_rate", float("nan")),
        ("scale_drift_rate", float("inf")),
        ("noise_std", -0.5),
    ]:
        with pytest.raises(ValueError, match=field):
            ScaleDriftStream(feature_dim=4, **{field: bad})  # type: ignore[arg-type]


def test_dynamic_scale_shift_float_validators_reject_nonfinite() -> None:
    with pytest.raises(ValueError, match="noise_std"):
        DynamicScaleShiftStream(feature_dim=4, noise_std=float("nan"))


def test_make_scale_range_rejects_invalid_feature_dim_hostile() -> None:
    with pytest.raises(ValueError, match="feature_dim"):
        make_scale_range(feature_dim=_RaisingIntSubclass(5))  # type: ignore[arg-type]


def test_hidden_and_scale_streams_preflight_state_bytes_without_allocation() -> None:
    last_legal = (2**29 - 1 - 3) // 2

    _require_float32_resource(
        "DynamicScaleShiftStream state",
        vector_scalars=2 * last_legal,
        fixed_scalars=3,
    )
    _require_float32_resource(
        "ScaleDriftStream state",
        vector_scalars=2 * last_legal,
        fixed_scalars=3,
    )
    with pytest.raises(ValueError, match="byte count"):
        DynamicScaleShiftStream(feature_dim=last_legal + 1)
    with pytest.raises(ValueError, match="byte count"):
        ScaleDriftStream(feature_dim=last_legal + 1)

    hidden_last_legal = (2**29 - 1 - 2) // 2
    _require_float32_resource(
        "HiddenStateAR2Stream state",
        vector_scalars=2 * hidden_last_legal,
        fixed_scalars=2,
    )
    with pytest.raises(ValueError, match="byte count"):
        HiddenStateAR2Stream(feature_dim=hidden_last_legal + 1, visible_dim=2)


def test_sutton_stream_preflights_derived_feature_and_state_bytes() -> None:
    last_legal_total = 2**29 - 1 - 3

    _require_float32_resource(
        "SuttonExperiment1Stream state",
        vector_scalars=last_legal_total,
        fixed_scalars=3,
    )
    with pytest.raises(ValueError, match="byte count"):
        SuttonExperiment1Stream(num_relevant=1, num_irrelevant=last_legal_total)
    with pytest.raises(ValueError, match="feature_dim"):
        SuttonExperiment1Stream(num_relevant=2**30, num_irrelevant=2**30)


def test_make_scale_range_rejects_oversize_before_numpy_allocation(monkeypatch) -> None:
    def allocation_must_not_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("allocation ran before resource validation")

    monkeypatch.setattr(np, "geomspace", allocation_must_not_run)
    with pytest.raises(ValueError, match="byte count"):
        make_scale_range(feature_dim=2**29)
