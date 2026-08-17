"""Focused validation for canonical UPGD configs — hostile-safe and float32."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.canonical_upgd import (
    AlbertaAdaUPGD,
    AlbertaAdaUPGDConfig,
    CanonicalUPGD,
    CanonicalUPGDConfig,
    OfficialAdaUPGD,
    OfficialAdaUPGDConfig,
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


class _MetadataOnlyArray:
    dtype = np.dtype(np.float32)

    def __init__(self, size: int):
        self.shape = (size,)
        self.conversion_calls = 0

    def __jax_array__(self) -> jax.Array:
        self.conversion_calls += 1
        raise RuntimeError("conversion must not run before resource preflight")


class _UnreadableMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise RuntimeError("hostile mapping hook")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("hostile mapping hook")

    def __len__(self) -> int:
        return 1


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


def test_canonical_hostile_ratio_is_rejected_without_executing_hook() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be a finite real"):
        _canon(step_size=_HostileFloat(1.0))
    assert _HostileFloat.calls == 0


def test_alberta_hostile_ratio_is_rejected_without_executing_hook() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be a finite real"):
        _alberta(step_size=_HostileFloat(1.0))
    assert _HostileFloat.calls == 0


def test_official_hostile_ratio_is_rejected_without_executing_hook() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be a finite real"):
        _official(step_size=_HostileFloat(1.0))
    assert _HostileFloat.calls == 0


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
    with pytest.raises(ValueError, match="remain in"):
        _canon(utility_decay=0.999999999)
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


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (_canon, "utility_decay"),
        (_canon, "noise_std"),
        (_canon, "weight_decay"),
        (_alberta, "second_moment_decay"),
        (_official, "beta1"),
    ],
)
def test_nonnegative_nonzero_scalars_cannot_silently_underflow(
    factory: Any, field: str
) -> None:
    with pytest.raises(ValueError, match="remain nonzero"):
        factory(**{field: 1e-50})
    assert getattr(factory(**{field: 0.0}), field) == 0.0


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


@pytest.mark.parametrize(
    ("optimizer", "bytes_per_scalar", "fixed_bytes"),
    [
        (CanonicalUPGD(), 24, 33),
        (AlbertaAdaUPGD(), 32, 50),
        (OfficialAdaUPGD(), 36, 24),
    ],
)
def test_production_init_preflights_complete_update_result_before_conversion(
    optimizer: object, bytes_per_scalar: int, fixed_bytes: int
) -> None:
    last_legal = (_INT32_MAX - fixed_bytes) // bytes_per_scalar
    legal = _MetadataOnlyArray(last_legal)
    with pytest.raises(ValueError, match="could not be converted"):
        optimizer.init({"w": legal})  # type: ignore[attr-defined]
    assert legal.conversion_calls == 1

    oversized = _MetadataOnlyArray(last_legal + 1)
    with pytest.raises(ValueError, match="derived .* update_result_nbytes"):
        optimizer.init({"w": oversized})  # type: ignore[attr-defined]
    assert oversized.conversion_calls == 0


def _tree_nbytes(tree: object) -> int:
    return sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in jax.tree.leaves(tree))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.float64])
def test_resource_formulas_match_initialized_state_and_update_results(dtype: Any) -> None:
    with jax.enable_x64():
        params = {"w": jnp.ones((2, 3), dtype=dtype)}
        gradients = jax.tree.map(jnp.ones_like, params)
        noise = jax.tree.map(jnp.zeros_like, params)
        parameter_nbytes = _tree_nbytes(params)
        cases = [
            (
                CanonicalUPGD(),
                parameter_nbytes + 4 * 6 + 4,
                5 * parameter_nbytes + 4 * 6 + 33,
            ),
            (
                AlbertaAdaUPGD(),
                2 * parameter_nbytes + 4 * 6 + 4,
                7 * parameter_nbytes + 4 * 6 + 50,
            ),
            (
                OfficialAdaUPGD(),
                3 * parameter_nbytes + 4,
                9 * parameter_nbytes + 24,
            ),
        ]
        for optimizer, expected_state, expected_update in cases:
            state = optimizer.init(params)
            result = optimizer.update(
                state, params, gradients, jr.key(0), noise=noise
            )
            assert _tree_nbytes(state) == expected_state
            assert _tree_nbytes(result) == expected_update


@pytest.mark.parametrize("optimizer", [CanonicalUPGD(), AlbertaAdaUPGD(), OfficialAdaUPGD()])
def test_python_float_parameter_leaf_compatibility(optimizer: object) -> None:
    params = {"w": 1.0}
    state = optimizer.init(params)  # type: ignore[attr-defined]
    result = optimizer.update(  # type: ignore[attr-defined]
        state, params, {"w": 0.5}, jr.key(4), noise={"w": 0.0}
    )
    assert jnp.asarray(result.params["w"]).shape == ()


@pytest.mark.parametrize(
    "config_type",
    [CanonicalUPGDConfig, AlbertaAdaUPGDConfig, OfficialAdaUPGDConfig],
)
def test_from_config_preserves_mapping_compatibility_and_normalizes_hooks(
    config_type: type[object],
) -> None:
    original = config_type()  # type: ignore[call-arg]
    payload = original.to_config()  # type: ignore[attr-defined]
    assert config_type.from_config(MappingProxyType(payload)) == original  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="readable mapping"):
        config_type.from_config(_UnreadableMapping())  # type: ignore[attr-defined]


def test_hostile_string_hooks_are_not_invoked() -> None:
    class HostileString:
        def __hash__(self) -> int:
            raise AssertionError("hash executed")

        def __eq__(self, other: object) -> bool:
            raise AssertionError("equality executed")

        def __repr__(self) -> str:
            raise AssertionError("repr executed")

    hostile = HostileString()
    with pytest.raises(ValueError, match="mode"):
        _canon(mode=hostile)
    with pytest.raises(ValueError, match="profile"):
        _alberta(profile=hostile)
    with pytest.raises(ValueError, match="profile"):
        _official(profile=hostile)
