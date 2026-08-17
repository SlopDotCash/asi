"""Focused validation for canonical UPGD configs — hostile-safe and float32."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.canonical_upgd import (
    ALBERTA_ADAUPGD_PROFILE,
    OFFICIAL_ADAUPGD_COMMIT,
    OFFICIAL_ADAUPGD_PATH,
    OFFICIAL_ADAUPGD_PROFILE,
    AlbertaAdaUPGD,
    AlbertaAdaUPGDConfig,
    AlbertaAdaUPGDResources,
    CanonicalUPGD,
    CanonicalUPGDConfig,
    OfficialAdaUPGD,
    OfficialAdaUPGDConfig,
    OfficialAdaUPGDResources,
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


def _floating_result_is_finite(result: object) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree.leaves(result)
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)
    )


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
        (CanonicalUPGD(), 96, 128),
        (AlbertaAdaUPGD(), 140, 128),
        (OfficialAdaUPGD(), 140, 128),
    ],
)
def test_production_init_preflights_complete_working_set_before_conversion(
    optimizer: object, bytes_per_scalar: int, fixed_bytes: int
) -> None:
    last_legal = (_INT32_MAX - fixed_bytes) // bytes_per_scalar
    legal = _MetadataOnlyArray(last_legal)
    with pytest.raises(ValueError, match="could not be converted"):
        optimizer.init({"w": legal})  # type: ignore[attr-defined]
    assert legal.conversion_calls == 1

    oversized = _MetadataOnlyArray(last_legal + 1)
    with pytest.raises(ValueError, match="derived .* working_set_nbytes"):
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


def test_safe_canonical_operation_overflow_rolls_back_atomically() -> None:
    maximum = np.finfo(np.float32).max
    params = {"w": jnp.asarray([maximum], dtype=jnp.float32)}
    gradients = {"w": jnp.asarray([-maximum], dtype=jnp.float32)}
    noise = {"w": jnp.zeros((1,), dtype=jnp.float32)}
    optimizer = CanonicalUPGD(
        CanonicalUPGDConfig(
            utility_decay=0.0,
            noise_std=0.0,
            profile="safe_extended",
            normalization="global",
        )
    )
    state = optimizer.init(params)
    key = jr.key(11)
    result = optimizer.update(state, params, gradients, key, noise=noise)
    assert not bool(result.metrics["update_applied"])
    chex.assert_trees_all_equal(result.params, params)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(jr.key_data(result.next_key), jr.key_data(key))
    assert _floating_result_is_finite(result)


def test_alberta_float64_metric_narrowing_overflow_rolls_back() -> None:
    with jax.enable_x64():
        maximum32 = float(np.finfo(np.float32).max)
        params = {"w": jnp.asarray([maximum32 * 2.0], dtype=jnp.float64)}
        gradients = {"w": jnp.asarray([-1.0], dtype=jnp.float64)}
        noise = {"w": jnp.zeros((1,), dtype=jnp.float64)}
        optimizer = AlbertaAdaUPGD(
            AlbertaAdaUPGDConfig(
                utility_decay=0.0,
                second_moment_decay=0.0,
                noise_std=0.0,
                normalization="global",
            )
        )
        state = optimizer.init(params)
        key = jr.key(12)
        result = optimizer.update(state, params, gradients, key, noise=noise)
        assert not bool(result.accepted)
        chex.assert_trees_all_equal(result.params, params)
        chex.assert_trees_all_equal(result.state, state)
        chex.assert_trees_all_equal(jr.key_data(result.next_key), jr.key_data(key))
        assert _floating_result_is_finite(result)


def test_official_counter_saturates_without_changing_source_equations() -> None:
    params = {"w": jnp.ones((1,), dtype=jnp.float32)}
    gradients = {"w": jnp.ones((1,), dtype=jnp.float32)}
    noise = {"w": jnp.zeros((1,), dtype=jnp.float32)}
    optimizer = OfficialAdaUPGD()
    state = optimizer.init(params).replace(  # type: ignore[attr-defined]
        step=jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    )
    result = optimizer.update(state, params, gradients, jr.key(13), noise=noise)
    assert int(result.state.step) == _INT32_MAX


def test_canonical_upgd_resource_records_reject_leftover_identities() -> None:
    """Public AdaUPGD resource records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="official_reference_parity"):
        AlbertaAdaUPGDResources(
            profile=ALBERTA_ADAUPGD_PROFILE,
            official_reference_parity=1,
            parameter_count=1,
            persistent_array_count=2,
            persistent_state_nbytes=3,
        )
    with pytest.raises(ValueError, match="parameter_count"):
        AlbertaAdaUPGDResources(
            profile=ALBERTA_ADAUPGD_PROFILE,
            official_reference_parity=False,
            parameter_count=True,
            persistent_array_count=2,
            persistent_state_nbytes=3,
        )
    with pytest.raises(ValueError, match="parameter_count"):
        OfficialAdaUPGDResources(
            profile=OFFICIAL_ADAUPGD_PROFILE,
            source_commit=OFFICIAL_ADAUPGD_COMMIT,
            source_path=OFFICIAL_ADAUPGD_PATH,
            official_reference_parity=True,
            parameter_count=True,
            persistent_array_count=2,
            persistent_state_nbytes=3,
        )
    with pytest.raises(ValueError, match="persistent_state_nbytes"):
        OfficialAdaUPGDResources(
            profile=OFFICIAL_ADAUPGD_PROFILE,
            source_commit=OFFICIAL_ADAUPGD_COMMIT,
            source_path=OFFICIAL_ADAUPGD_PATH,
            official_reference_parity=True,
            parameter_count=1,
            persistent_array_count=2,
            persistent_state_nbytes=float("nan"),
        )

    legal_alberta = AlbertaAdaUPGDResources(
        profile=ALBERTA_ADAUPGD_PROFILE,
        official_reference_parity=False,
        parameter_count=1,
        persistent_array_count=2,
        persistent_state_nbytes=3,
    )
    legal_official = OfficialAdaUPGDResources(
        profile=OFFICIAL_ADAUPGD_PROFILE,
        source_commit=OFFICIAL_ADAUPGD_COMMIT,
        source_path=OFFICIAL_ADAUPGD_PATH,
        official_reference_parity=True,
        parameter_count=1,
        persistent_array_count=2,
        persistent_state_nbytes=3,
    )
    dumped = json.dumps(
        {
            "alberta": legal_alberta.to_dict(),
            "official": legal_official.to_dict(),
        },
        allow_nan=False,
    )
    assert '"official_reference_parity": false' in dumped
    assert '"official_reference_parity": true' in dumped
    assert '"parameter_count": 1' in dumped
    assert '"parameter_count": true' not in dumped
    assert '"official_reference_parity": 1' not in dumped


class _HostileRecordIdentity:
    def __eq__(self, other: object) -> bool:
        del other
        raise AssertionError("untrusted equality hook executed")

    def __index__(self) -> int:
        raise AssertionError("untrusted index hook executed")

    def __int__(self) -> int:
        raise AssertionError("untrusted integer hook executed")


class _HostileStringIdentity(str):
    def __eq__(self, other: object) -> bool:
        del other
        raise AssertionError("untrusted string equality hook executed")


def _legal_alberta_resources(**overrides: object) -> AlbertaAdaUPGDResources:
    values: dict[str, object] = {
        "profile": ALBERTA_ADAUPGD_PROFILE,
        "official_reference_parity": False,
        "parameter_count": 1,
        "persistent_array_count": 2,
        "persistent_state_nbytes": 3,
    }
    values.update(overrides)
    return AlbertaAdaUPGDResources(**values)  # type: ignore[arg-type]


def _legal_official_resources(**overrides: object) -> OfficialAdaUPGDResources:
    values: dict[str, object] = {
        "profile": OFFICIAL_ADAUPGD_PROFILE,
        "source_commit": OFFICIAL_ADAUPGD_COMMIT,
        "source_path": OFFICIAL_ADAUPGD_PATH,
        "official_reference_parity": True,
        "parameter_count": 1,
        "persistent_array_count": 2,
        "persistent_state_nbytes": 3,
    }
    values.update(overrides)
    return OfficialAdaUPGDResources(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    [
        (_legal_alberta_resources, "profile", "safe_extended"),
        (_legal_alberta_resources, "profile", _HostileStringIdentity(ALBERTA_ADAUPGD_PROFILE)),
        (_legal_alberta_resources, "official_reference_parity", True),
        (_legal_official_resources, "profile", ALBERTA_ADAUPGD_PROFILE),
        (_legal_official_resources, "source_commit", "main"),
        (
            _legal_official_resources,
            "source_path",
            _HostileStringIdentity(OFFICIAL_ADAUPGD_PATH),
        ),
        (_legal_official_resources, "official_reference_parity", False),
        (_legal_official_resources, "parameter_count", _HostileRecordIdentity()),
        (_legal_official_resources, "persistent_array_count", np.int32(2)),
        (_legal_official_resources, "persistent_state_nbytes", -1),
        (_legal_official_resources, "persistent_state_nbytes", _INT32_MAX + 1),
    ],
)
def test_adaupgd_resource_records_bind_exact_hostile_safe_identities(
    factory: Any,
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        factory(**{field: value})


def test_adaupgd_resource_records_dump_only_bound_json_primitives() -> None:
    records = {
        "alberta": _legal_alberta_resources(
            parameter_count=_INT32_MAX,
            persistent_array_count=0,
            persistent_state_nbytes=_INT32_MAX,
        ).to_dict(),
        "official": _legal_official_resources().to_dict(),
    }
    dumped = json.dumps(records, allow_nan=False, sort_keys=True)
    decoded = json.loads(dumped)

    assert decoded["alberta"]["profile"] == ALBERTA_ADAUPGD_PROFILE
    assert decoded["alberta"]["official_reference_parity"] is False
    assert decoded["official"]["profile"] == OFFICIAL_ADAUPGD_PROFILE
    assert decoded["official"]["source_commit"] == OFFICIAL_ADAUPGD_COMMIT
    assert decoded["official"]["source_path"] == OFFICIAL_ADAUPGD_PATH
    assert decoded["official"]["official_reference_parity"] is True
