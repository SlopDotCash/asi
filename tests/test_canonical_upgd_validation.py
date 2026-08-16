"""Focused validation for canonical UPGD configs — hostile-safe and float32."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.canonical_upgd import (
    AlbertaAdaUPGDConfig,
    CanonicalUPGDConfig,
    OfficialAdaUPGDConfig,
    _require_float32_resource,
)

_INT32_MAX = 2_147_483_647


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:
        type(self).calls += 1
        raise RuntimeError("untrusted ratio hook executed")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr executed")


class _RaisingRepr:
    def __float__(self) -> float:  # pragma: no cover
        return 1.0

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr executed")


class _ClassSpoof:
    @property
    def __class__(self) -> type:
        return float  # type: ignore[return-value]

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr executed")


def _canon(**overrides: Any) -> CanonicalUPGDConfig:
    base: dict[str, Any] = {
        "step_size": 1e-3,
        "utility_decay": 0.5,
        "noise_std": 1e-3,
        "weight_decay": 0.0,
        "mode": "protecting",
        "profile": "safe_extended",
        "normalization": "global",
        "epsilon": 1e-8,
    }
    base.update(overrides)
    return CanonicalUPGDConfig(**base)


def _alberta(**overrides: Any) -> AlbertaAdaUPGDConfig:
    base: dict[str, Any] = {
        "step_size": 1e-3,
        "utility_decay": 0.5,
        "second_moment_decay": 0.9,
        "noise_std": 1e-3,
        "weight_decay": 0.0,
        "mode": "protecting",
        "normalization": "global",
        "epsilon": 1e-8,
    }
    base.update(overrides)
    return AlbertaAdaUPGDConfig(**base)


def _official(**overrides: Any) -> OfficialAdaUPGDConfig:
    base: dict[str, Any] = {
        "step_size": 1e-5,
        "weight_decay": 0.001,
        "utility_decay": 0.999,
        "noise_std": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-5,
    }
    base.update(overrides)
    return OfficialAdaUPGDConfig(**base)


def test_canonical_validators_accept_valid_values() -> None:
    cfg = _canon()
    assert cfg.step_size == pytest.approx(1e-3)
    assert cfg.utility_decay == pytest.approx(0.5)
    # float32 canonicalization check
    assert jnp.isfinite(jnp.asarray(cfg.step_size, dtype=jnp.float32))


def test_alberta_validators_accept_valid_values() -> None:
    cfg = _alberta()
    assert cfg.second_moment_decay == pytest.approx(0.9)
    assert jnp.isfinite(jnp.asarray(cfg.epsilon, dtype=jnp.float32))


def test_official_validators_accept_valid_values() -> None:
    cfg = _official()
    assert cfg.beta1 == pytest.approx(0.9)
    assert jnp.isfinite(jnp.asarray(cfg.step_size, dtype=jnp.float32))


@pytest.mark.parametrize(
    "field",
    ["step_size", "utility_decay", "noise_std", "weight_decay", "epsilon"],
)
def test_canonical_rejects_bool_and_spoof(field: str) -> None:
    for bad in (True, np.bool_(True), _ClassSpoof()):
        with pytest.raises((ValueError, TypeError), match="must be"):
            _canon(**{field: bad})


@pytest.mark.parametrize(
    "field",
    ["step_size", "utility_decay", "second_moment_decay", "noise_std", "weight_decay", "epsilon"],
)
def test_alberta_rejects_bool_and_spoof(field: str) -> None:
    for bad in (True, np.bool_(True), _ClassSpoof()):
        with pytest.raises((ValueError, TypeError), match="must be"):
            _alberta(**{field: bad})


@pytest.mark.parametrize(
    "field",
    ["step_size", "weight_decay", "utility_decay", "noise_std", "beta1", "beta2", "epsilon"],
)
def test_official_rejects_bool_and_spoof(field: str) -> None:
    for bad in (True, np.bool_(True), _ClassSpoof()):
        with pytest.raises((ValueError, TypeError), match="must be"):
            _official(**{field: bad})


def test_canonical_hostile_ratio_is_caught_and_counts_once() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be a finite real"):
        _canon(step_size=_HostileFloat(1.0))
    assert _HostileFloat.calls == 1


def test_alberta_hostile_ratio_is_caught_and_counts_once() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be a finite real"):
        _alberta(step_size=_HostileFloat(1.0))
    assert _HostileFloat.calls == 1


def test_official_hostile_ratio_is_caught_and_counts_once() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be a finite real"):
        _official(step_size=_HostileFloat(1.0))
    assert _HostileFloat.calls == 1


def test_hostile_repr_is_not_executed_for_float_fields() -> None:
    # validated_float32_scalar must not interpolate !r
    for bad in (_RaisingRepr(),):
        # Canonical uses validated helper which should not call __repr__
        try:
            _canon(step_size=bad)  # type: ignore
        except ValueError:
            pass
        except TypeError:
            pass
        else:
            raise AssertionError("should have raised")
        # if repr had been called, AssertionError from __repr__ would propagate


def test_canonical_domain_rejects_invalid_ranges() -> None:
    # step_size must be positive
    with pytest.raises(ValueError, match="must be"):
        _canon(step_size=0.0)
    with pytest.raises(ValueError, match="must be"):
        _canon(step_size=-1.0)
    with pytest.raises(ValueError, match="must be"):
        _canon(step_size=float("inf"))
    with pytest.raises(ValueError, match="must be"):
        _canon(step_size=float("nan"))
    # utility_decay must be in [0,1)
    with pytest.raises(ValueError, match="must be"):
        _canon(utility_decay=1.0)
    with pytest.raises(ValueError, match="must be"):
        _canon(utility_decay=-0.1)
    # epsilon must be positive
    with pytest.raises(ValueError, match="must be"):
        _canon(epsilon=0.0)
    # noise_std must be >=0
    with pytest.raises(ValueError, match="must be"):
        _canon(noise_std=-0.1)


def test_alberta_domain_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="must be"):
        _alberta(step_size=0.0)
    with pytest.raises(ValueError, match="must be"):
        _alberta(utility_decay=1.0)
    with pytest.raises(ValueError, match="must be"):
        _alberta(second_moment_decay=1.0)
    with pytest.raises(ValueError, match="must be"):
        _alberta(epsilon=0.0)
    with pytest.raises(ValueError, match="must be"):
        _alberta(noise_std=-1.0)


def test_official_domain_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="must be"):
        _official(step_size=0.0)
    with pytest.raises(ValueError, match="must be"):
        _official(utility_decay=1.0)
    with pytest.raises(ValueError, match="must be"):
        _official(beta1=1.0)
    with pytest.raises(ValueError, match="must be"):
        _official(beta2=1.0)
    with pytest.raises(ValueError, match="must be"):
        _official(epsilon=0.0)


def test_float32_overflow_is_rejected() -> None:
    # 1e40 is finite in host float64 but overflows float32 -> inf
    with pytest.raises(ValueError, match="must remain finite once narrowed"):
        _canon(step_size=1e40)
    with pytest.raises(ValueError, match="must remain finite once narrowed"):
        _alberta(step_size=1e40)
    with pytest.raises(ValueError, match="must remain finite once narrowed"):
        _official(step_size=1e40)


def test_float32_underflow_is_rejected_for_positive() -> None:
    # tiny positive that underflows float32 to 0.0 while host is non-zero
    tiny = 1e-50  # way below float32 subnormal, rounds to 0.0
    # only positive fields should reject underflow
    with pytest.raises(ValueError, match="must remain"):
        _canon(step_size=tiny)
    with pytest.raises(ValueError, match="must remain"):
        _canon(epsilon=tiny)


def test_config_string_validators_reject_invalid_and_hostile_repr() -> None:
    # mode must be protecting/non_protecting without !r
    class BadRepr:
        def __repr__(self) -> str:
            raise AssertionError("repr executed")

    with pytest.raises(ValueError, match="mode must be"):
        _canon(mode="bad")  # type: ignore
    # normalization
    with pytest.raises(ValueError, match="normalization"):
        _canon(normalization="bad")  # type: ignore
    # profile
    with pytest.raises(ValueError, match="profile must"):
        _canon(profile="bad")  # type: ignore
    # ensure hostile repr in mode string does not call repr
    # mode string compare does not call repr
    bad = BadRepr()
    # pass bad object as mode; comparison calls __eq__, not __repr__
    with pytest.raises(ValueError):
        _canon(mode=bad)  # type: ignore


def test_from_config_hostile_repr_not_executed() -> None:
    # canonical from_config previously used !r for type_name
    class HostileType:
        def __repr__(self) -> str:
            raise AssertionError("repr executed")

    payload = {
        "type": HostileType(),  # type: ignore
        "step_size": 1e-3,
        "utility_decay": 0.5,
        "noise_std": 1e-3,
        "weight_decay": 0.0,
        "mode": "protecting",
        "profile": "safe_extended",
        "normalization": "global",
        "epsilon": 1e-8,
    }
    with pytest.raises(ValueError, match="expected CanonicalUPGD"):
        CanonicalUPGDConfig.from_config(payload)  # type: ignore


def test_alberta_from_config_hostile_repr_not_executed() -> None:
    class HostileType:
        def __repr__(self) -> str:
            raise AssertionError("repr executed")

    payload = {
        "type": HostileType(),  # type: ignore
        "profile": "alberta_derived_first_order_adaptive_v1",
        "step_size": 1e-3,
        "utility_decay": 0.5,
        "second_moment_decay": 0.9,
        "noise_std": 1e-3,
        "weight_decay": 0.0,
        "mode": "protecting",
        "normalization": "global",
        "epsilon": 1e-8,
    }
    with pytest.raises(ValueError, match="expected AlbertaAdaUPGD"):
        AlbertaAdaUPGDConfig.from_config(payload)  # type: ignore


def test_official_from_config_hostile_repr_not_executed() -> None:
    class HostileType:
        def __repr__(self) -> str:
            raise AssertionError("repr executed")

    payload = {
        "type": HostileType(),  # type: ignore
        "profile": "official_rl_adaptive_upgd_b75e90a",
        "source_commit": "b75e90ad4b09c28971ac9dbb902a8fd86709b28c",
        "source_path": "core/run/rl/adaupgd.py",
        "step_size": 1e-5,
        "weight_decay": 0.001,
        "utility_decay": 0.999,
        "noise_std": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-5,
    }
    with pytest.raises(ValueError, match="expected OfficialAdaUPGD"):
        OfficialAdaUPGDConfig.from_config(payload)  # type: ignore


def test_canonical_normalization_fixed_by_profile() -> None:
    # paper_global fixes normalization to global, passing local should fail
    with pytest.raises(ValueError, match="fixes normalization"):
        _canon(profile="paper_global", normalization="local")  # type: ignore
    # official_experiment_local fixes to local
    with pytest.raises(ValueError, match="fixes normalization"):
        _canon(profile="official_experiment_local", normalization="global")  # type: ignore
    # safe_extended allows explicit global/local
    _canon(profile="safe_extended", normalization="global")
    _canon(profile="safe_extended", normalization="local")


def test_resource_helper_allocation_free_boundaries() -> None:
    legal_scalars = _INT32_MAX // 4
    # scalar count must fit signed int32
    _require_float32_resource("test", vector_scalars=legal_scalars)
    with pytest.raises(ValueError, match="scalar count must fit signed int32"):
        _require_float32_resource("test", vector_scalars=_INT32_MAX + 1)
    # byte count must fit signed int32 (4*scalars)
    with pytest.raises(ValueError, match="byte count must fit signed int32"):
        _require_float32_resource("test", vector_scalars=legal_scalars + 1)
    # combined vector + fixed
    _require_float32_resource("test", vector_scalars=legal_scalars - 5, fixed_scalars=5)
    with pytest.raises(ValueError, match="byte count"):
        _require_float32_resource("test", vector_scalars=legal_scalars, fixed_scalars=1)
