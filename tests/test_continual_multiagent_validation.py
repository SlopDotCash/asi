"""Adversarial configuration and resource gates for the multi-agent benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from alberta_framework.evaluation.continual_multiagent import (
    _MAX_CONFIGURED_ARRAY_NBYTES,
    AcceptanceThresholds,
    ContinualMultiAgentConfig,
    _bootstrap_working_nbytes,
    _condition_result_array_nbytes,
    _condition_working_nbytes,
    _world_array_nbytes,
    paired_bootstrap_mean_interval,
    run_continual_multiagent_benchmark,
)

_INT32_MAX = 2**31 - 1
_NUMPY_INTEGER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.longlong,
    np.ulonglong,
)
_CONFIG_INTEGER_VALUES = {
    "phase_steps": 64,
    "nuisance_dim": 4,
    "probe_horizon": 12,
    "probe_tail_steps": 4,
    "recovery_window": 4,
    "bootstrap_resamples": 10_000,
    "bootstrap_seed": 123,
}
_THRESHOLD_INTEGER_VALUES = {
    "minimum_seed_count": 30,
    "evidence_seed_start": 30,
}


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover - must not run
        raise AssertionError("untrusted index hook executed")

    def __int__(self) -> int:  # pragma: no cover - must not run
        raise AssertionError("untrusted int hook executed")

    def __repr__(self) -> str:  # pragma: no cover - must not run
        raise AssertionError("untrusted repr hook executed")


class _ClassSpoof:
    @property
    def __class__(self) -> type[int]:  # pragma: no cover - validator must ignore
        return int

    def __repr__(self) -> str:  # pragma: no cover - must not run
        raise AssertionError("untrusted repr hook executed")


def _bad_integer_scalar(kind: str) -> object:
    return {
        "bool": True,
        "numpy_bool": np.bool_(True),
        "float": 4.0,
        "string": "4",
        "int_subclass": _HostileInt(4),
        "class_spoof": _ClassSpoof(),
    }[kind]


@pytest.mark.parametrize("field,value", _CONFIG_INTEGER_VALUES.items())
@pytest.mark.parametrize("integer_type", _NUMPY_INTEGER_TYPES)
def test_config_accepts_every_numpy_integer_family(
    field: str, value: int, integer_type: type[np.integer],
) -> None:
    if value > np.iinfo(integer_type).max:
        pytest.skip("field domain has no value representable by this NumPy dtype")
    config = ContinualMultiAgentConfig(**{field: integer_type(value)})  # type: ignore[arg-type]
    assert getattr(config, field) == value
    assert type(getattr(config, field)) is int


@pytest.mark.parametrize("field,value", _THRESHOLD_INTEGER_VALUES.items())
@pytest.mark.parametrize("integer_type", _NUMPY_INTEGER_TYPES)
def test_thresholds_accept_every_numpy_integer_family(
    field: str, value: int, integer_type: type[np.integer],
) -> None:
    thresholds = AcceptanceThresholds(**{field: integer_type(value)})  # type: ignore[arg-type]
    assert getattr(thresholds, field) == value
    assert type(getattr(thresholds, field)) is int


@pytest.mark.parametrize("field", tuple(_CONFIG_INTEGER_VALUES))
@pytest.mark.parametrize(
    "bad_kind", ("bool", "numpy_bool", "float", "string", "int_subclass", "class_spoof")
)
def test_config_rejects_noncanonical_integer_scalars_without_hooks(
    field: str, bad_kind: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        ContinualMultiAgentConfig(  # type: ignore[arg-type]
            **{field: _bad_integer_scalar(bad_kind)}
        )


@pytest.mark.parametrize("field", tuple(_THRESHOLD_INTEGER_VALUES))
@pytest.mark.parametrize(
    "bad_kind", ("bool", "numpy_bool", "float", "string", "int_subclass", "class_spoof")
)
def test_thresholds_reject_noncanonical_integer_scalars_without_hooks(
    field: str, bad_kind: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        AcceptanceThresholds(  # type: ignore[arg-type]
            **{field: _bad_integer_scalar(bad_kind)}
        )


def test_phase_resource_boundary_is_exact_and_allocation_free() -> None:
    maximum = _MAX_CONFIGURED_ARRAY_NBYTES // 48
    assert _condition_working_nbytes(maximum) <= _MAX_CONFIGURED_ARRAY_NBYTES
    assert _condition_working_nbytes(maximum + 1) > _MAX_CONFIGURED_ARRAY_NBYTES
    assert ContinualMultiAgentConfig(
        phase_steps=maximum,
        probe_horizon=1,
        probe_tail_steps=1,
        recovery_window=1,
    ).phase_steps == maximum
    with pytest.raises(ValueError, match="phase work arrays"):
        ContinualMultiAgentConfig(
            phase_steps=maximum + 1,
            probe_horizon=1,
            probe_tail_steps=1,
            recovery_window=1,
        )


def test_world_resource_boundary_is_exact_and_allocation_free() -> None:
    maximum = (_MAX_CONFIGURED_ARRAY_NBYTES - 48) // 8
    state_nbytes, observation_nbytes = _world_array_nbytes(maximum)
    assert state_nbytes <= _MAX_CONFIGURED_ARRAY_NBYTES
    assert observation_nbytes <= _MAX_CONFIGURED_ARRAY_NBYTES
    assert ContinualMultiAgentConfig(nuisance_dim=maximum).nuisance_dim == maximum
    with pytest.raises(ValueError, match="recurring world observation"):
        ContinualMultiAgentConfig(nuisance_dim=maximum + 1)


def test_bootstrap_resource_formula_and_resample_boundary_are_exact() -> None:
    maximum = _MAX_CONFIGURED_ARRAY_NBYTES // 24
    assert _bootstrap_working_nbytes(maximum, 1) <= _MAX_CONFIGURED_ARRAY_NBYTES
    assert _bootstrap_working_nbytes(maximum + 1, 1) > _MAX_CONFIGURED_ARRAY_NBYTES
    assert ContinualMultiAgentConfig(bootstrap_resamples=maximum).bootstrap_resamples == maximum
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        ContinualMultiAgentConfig(bootstrap_resamples=maximum + 1)

    values = np.ones((2,), dtype=np.float64)
    overflowing_resamples = _MAX_CONFIGURED_ARRAY_NBYTES // (40) + 1
    with pytest.raises(ValueError, match="paired bootstrap working arrays"):
        paired_bootstrap_mean_interval(
            values,
            confidence_level=0.95,
            resamples=overflowing_resamples,
            seed=0,
        )


def test_retained_result_resource_gate_precedes_seed_materialization() -> None:
    bytes_per_seed = 3 * _condition_result_array_nbytes(64)
    first_overflowing_count = _MAX_CONFIGURED_ARRAY_NBYTES // bytes_per_seed + 1
    with pytest.raises(ValueError, match="retained condition-result arrays"):
        run_continual_multiagent_benchmark(seeds=range(first_overflowing_count))


def test_seed_container_and_scalars_are_hostile_safe() -> None:
    class HostileTuple(tuple):
        def __len__(self) -> int:  # pragma: no cover - must not run
            raise AssertionError("untrusted length hook executed")

        def __repr__(self) -> str:  # pragma: no cover - must not run
            raise AssertionError("untrusted repr hook executed")

    with pytest.raises(ValueError, match="actual list, tuple, or range"):
        run_continual_multiagent_benchmark(seeds=HostileTuple((0,)))
    with pytest.raises(ValueError, match="seed"):
        run_continual_multiagent_benchmark(seeds=(_HostileInt(0),))


def test_threshold_seed_schedule_stays_inside_signed_jax_seed_domain() -> None:
    accepted = AcceptanceThresholds(
        evidence_seed_start=_INT32_MAX,
        minimum_seed_count=1,
    )
    assert accepted.evidence_seed_start + accepted.minimum_seed_count == 2**31
    with pytest.raises(ValueError, match=r"evidence_seed_start \+ minimum_seed_count"):
        AcceptanceThresholds(
            evidence_seed_start=_INT32_MAX,
            minimum_seed_count=2,
        )


def test_default_configuration_and_thresholds_round_trip_through_canonical_json() -> None:
    config_payload = json.loads(json.dumps(asdict(ContinualMultiAgentConfig()), sort_keys=True))
    threshold_payload = json.loads(json.dumps(asdict(AcceptanceThresholds()), sort_keys=True))
    assert ContinualMultiAgentConfig(**config_payload) == ContinualMultiAgentConfig()
    assert AcceptanceThresholds(**threshold_payload) == AcceptanceThresholds()
