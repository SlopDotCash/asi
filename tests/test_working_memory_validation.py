"""Validation hardening for working memory (int/float/bool bounds + resource)."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.working_memory import (
    WorkingMemoryConfig,
    WorkingMemoryFeaturizer,
    transform_working_memory_arrays,
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


def _base_cfg(**overrides):
    cfg = {
        "observation_dim": 2,
        "action_dim": 1,
        "reward_dim": 1,
    }
    cfg.update(overrides)
    return WorkingMemoryConfig(**cfg)


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _base_cfg(observation_dim=v),
        lambda v: _base_cfg(action_dim=v),
        lambda v: _base_cfg(reward_dim=v),
    ],
)
def test_working_int_validators_reject_hostile_subclass_without_running_hook(
    ctor,
) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        ctor(_LyingIntSubclass(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        ctor(_RaisingIntSubclass(4))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _base_cfg(observation_dim=v),
        lambda v: _base_cfg(action_dim=v),
    ],
)
def test_working_int_validators_do_not_run_repr_hook(ctor) -> None:
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
def test_working_int_validators_canonicalize_numpy_scalars(np_type: type) -> None:
    cfg = _base_cfg(
        observation_dim=np_type(4),
        action_dim=np_type(2),
        reward_dim=np_type(2),
    )
    assert cfg.observation_dim == 4
    assert cfg.action_dim == 2
    assert cfg.reward_dim == 2
    assert type(cfg.observation_dim) is int
    assert type(cfg.action_dim) is int


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), 4.0, np.float64(4.0), "4", None, 0, -1, _INT32_MAX + 1, 10**100],
)
def test_working_observation_dim_rejects_non_integer_and_out_of_range(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="must be"):
        _base_cfg(observation_dim=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, np.bool_(True), 4.0, "4", None, -1, _INT32_MAX + 1])
def test_working_action_reward_rejects_non_integer_and_out_of_range(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="must be"):
        _base_cfg(action_dim=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be"):
        _base_cfg(reward_dim=value)  # type: ignore[arg-type]


def test_working_decay_rates_reject_not_tuple() -> None:
    with pytest.raises(ValueError, match="must be an actual tuple"):
        _base_cfg(observation_decay_rates=[0.5])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an actual tuple"):
        _base_cfg(action_decay_rates="0.5")  # type: ignore[arg-type]


def test_working_decay_rates_reject_nonfinite_and_hostile() -> None:
    class HostileFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            raise RuntimeError("ratio hook")

    class ClassSpoof:
        @property
        def __class__(self):  # type: ignore[no-untyped-def]
            return float

        def __float__(self) -> float:  # pragma: no cover
            return 0.5

    for bad in [float("nan"), float("inf"), -0.1, 1.0, 2.0, HostileFloat(0.5), ClassSpoof()]:
        with pytest.raises(ValueError):
            _base_cfg(observation_decay_rates=(bad,))  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            _base_cfg(action_decay_rates=(bad,))  # type: ignore[arg-type]


def test_working_float_validators_reject_nonfinite_and_hostile() -> None:
    class HostileFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            raise RuntimeError("ratio hook")

    for field, bad in [
        ("gate_threshold", float("nan")),
        ("gate_threshold", float("inf")),
        ("gate_threshold", -0.1),
        ("gate_threshold", HostileFloat(0.5)),
        ("gate_temperature", 0.0),
        ("gate_temperature", -1.0),
        ("gate_temperature", float("nan")),
        ("gate_temperature", float("inf")),
        ("gate_temperature", HostileFloat(0.5)),
    ]:
        with pytest.raises(ValueError, match=field):
            _base_cfg(**{field: bad})  # type: ignore[arg-type]


def test_working_bool_exact_type() -> None:
    for field in [
        "include_current_observation",
        "include_current_action",
        "include_current_reward",
        "include_traces",
        "include_innovations",
        "gated_update",
    ]:
        with pytest.raises(ValueError, match=field):
            _base_cfg(**{field: 1})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=field):
            _base_cfg(**{field: np.bool_(True)})  # type: ignore[arg-type]


def test_working_float_validators_accept_valid_values() -> None:
    cfg = _base_cfg(gate_threshold=0.0, gate_temperature=1.0)
    assert cfg.gate_threshold == pytest.approx(0.0)
    cfg2 = _base_cfg(
        observation_decay_rates=(0.5, 0.9),
        action_decay_rates=(0.5,),
        reward_decay_rates=(0.9,),
    )
    assert cfg2.observation_decay_rates == (0.5, 0.9)


def test_working_dimensions_preflight_without_allocation() -> None:
    # Single product overflow: observation_dim * len > INT32
    # Use len=3 default, so need obs_dim > INT32//3
    big = _INT32_MAX // 3 + 10
    with pytest.raises(
        ValueError,
        match="dimensions must fit signed|scalar count|byte count|configuration feature_dim",
    ):
        _base_cfg(observation_dim=big)
    # Scalar count via trace_scalars overflow
    with pytest.raises(
        ValueError,
        match="dimensions must fit signed|scalar count|byte count|configuration feature_dim",
    ):
        _base_cfg(observation_dim=_INT32_MAX, action_dim=1, reward_dim=1)


def test_working_state_preflight_bytes_without_allocation() -> None:
    # Minimal custom decays to make math tractable: len_obs=3 default
    # Use observation_dim variation with other dims 0 to isolate
    # With this configuration, the conservative update bound is 35*obs+47.
    last_legal = (_INT32_MAX // 4 - 47) // 35
    # Need to set action/reward 0 to keep trace minimal
    cfg = WorkingMemoryConfig(
        observation_dim=last_legal,
        action_dim=0,
        reward_dim=0,
        observation_decay_rates=(0.5, 0.9, 0.99),
        action_decay_rates=(),
        reward_decay_rates=(),
    )
    assert cfg.observation_dim == last_legal
    with pytest.raises(ValueError, match="byte count"):
        WorkingMemoryConfig(
            observation_dim=last_legal + 1,
            action_dim=0,
            reward_dim=0,
            observation_decay_rates=(0.5, 0.9, 0.99),
            action_decay_rates=(),
            reward_decay_rates=(),
        )
    # Non-minimal should also be allocation-free
    with pytest.raises(ValueError, match="byte count|scalar count|dimensions"):
        _base_cfg(
            observation_dim=200_000_000,
            action_dim=200_000_000,
            reward_dim=200_000_000,
        )


def test_working_state_preflight_feature_dim_boundary() -> None:
    # Feature dim boundary with default config (obs=2, act=1, rew=1)
    # Just ensure large dims overflow
    with pytest.raises(
        ValueError,
        match="configuration feature_dim|byte count|scalar count|dimensions",
    ):
        _base_cfg(observation_dim=600_000_000, action_dim=600_000_000, reward_dim=600_000_000)


def test_working_serialized_schema_preserves_only_exact_list_tuple_compatibility() -> None:
    config = _base_cfg()
    payload = config.to_config()
    payload["observation_decay_rates"] = tuple(payload["observation_decay_rates"])
    assert WorkingMemoryConfig.from_config(MappingProxyType(payload)) == config

    for mutation, match in (
        ({"type": "OtherConfig"}, "type"),
        ({"observation_dim": np.int32(2)}, "observation_dim"),
        ({"gate_threshold": np.float32(0.0)}, "gate_threshold"),
        ({"include_traces": 1}, "include_traces"),
        ({"observation_decay_rates": [np.float32(0.5)]}, "JSON numbers"),
        ({"extra": 1}, "fields"),
    ):
        invalid = config.to_config()
        invalid.update(mutation)
        with pytest.raises(ValueError, match=match):
            WorkingMemoryConfig.from_config(invalid)
    missing = config.to_config()
    missing.pop("gate_temperature")
    with pytest.raises(ValueError, match="fields"):
        WorkingMemoryConfig.from_config(missing)


def test_working_mapping_hooks_are_normalized_without_class_spoofing() -> None:
    class HostileMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("getitem hook")

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError("iteration hook")

        def __len__(self) -> int:
            return 1

    class MappingSpoof:
        @property
        def __class__(self) -> type:
            return dict

        def __iter__(self) -> object:
            raise AssertionError("iteration hook executed")

    for loader in (WorkingMemoryConfig.from_config, WorkingMemoryFeaturizer.from_config):
        with pytest.raises(ValueError, match="mapping"):
            loader(HostileMapping())
        with pytest.raises(ValueError, match="mapping"):
            loader(MappingSpoof())  # type: ignore[arg-type]


def test_working_featurizer_serialized_envelope_is_exact() -> None:
    memory = WorkingMemoryFeaturizer(_base_cfg())
    payload = memory.to_config()
    assert WorkingMemoryFeaturizer.from_config(MappingProxyType(payload)).config == memory.config
    with pytest.raises(ValueError, match="type"):
        WorkingMemoryFeaturizer.from_config({**payload, "type": "OtherFeaturizer"})
    with pytest.raises(ValueError, match="fields"):
        WorkingMemoryFeaturizer.from_config({**payload, "extra": 1})


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("observation_traces", jnp.zeros((3, 1), dtype=jnp.float32), "observation_traces"),
        ("action_traces", jnp.zeros((2, 1), dtype=jnp.float16), "action_traces"),
        ("reward_traces", jnp.zeros((2,), dtype=jnp.float32), "reward_traces"),
        ("step_count", jnp.zeros((1,), dtype=jnp.int32), "step_count"),
        ("last_gate", jnp.zeros((1, 3), dtype=jnp.float32), "last_gate"),
    ],
)
def test_working_public_methods_reject_malformed_state_static_contract(
    field: str, replacement: jax.Array, match: str
) -> None:
    memory = WorkingMemoryFeaturizer(_base_cfg())
    state = dataclasses.replace(memory.init(), **{field: replacement})
    args = (state, jnp.zeros((2,)), jnp.zeros((1,)), jnp.zeros((1,)))
    for call in (
        lambda: memory.features(*args),
        lambda: memory.update_checked(*args),
        lambda: memory.step(*args),
        lambda: jax.jit(memory.update_checked)(*args),
    ):
        with pytest.raises((TypeError, ValueError), match=match):
            call()


def test_working_invalid_state_rolls_back_and_lifetime_counter_saturates() -> None:
    memory = WorkingMemoryFeaturizer(_base_cfg())
    args = (jnp.zeros((2,)), jnp.zeros((1,)), jnp.zeros((1,)))
    invalid = dataclasses.replace(
        memory.init(),
        step_count=jnp.asarray(-1, dtype=jnp.int32),
    )
    rejected = memory.update_checked(invalid, *args)
    assert not bool(rejected.update_applied)
    assert int(rejected.state.step_count) == -1

    maximum = dataclasses.replace(
        memory.init(),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    accepted = memory.update_checked(maximum, *args)
    assert bool(accepted.update_applied)
    assert int(accepted.state.step_count) == _INT32_MAX
    with pytest.raises(ValueError, match="external_gate"):
        memory.update_checked(memory.init(), *args, external_gate=jnp.ones((1,)))


def test_working_transform_preflights_output_bytes_without_allocation() -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=1,
            action_dim=0,
            reward_dim=0,
            observation_decay_rates=(),
            action_decay_rates=(),
            reward_decay_rates=(),
            include_current_action=False,
            include_current_reward=False,
            include_traces=False,
        )
    )
    first_overflow = _INT32_MAX // 5 + 1
    observations = jax.ShapeDtypeStruct((first_overflow, 1), jnp.float32)
    empty = jax.ShapeDtypeStruct((first_overflow, 0), jnp.float32)
    with pytest.raises(ValueError, match="byte count"):
        jax.eval_shape(
            lambda obs, zero: transform_working_memory_arrays(memory, obs, zero, zero),
            observations,
            empty,
        )


def test_working_gate_temperature_must_be_positive_normal_float32() -> None:
    minimum_normal = float.fromhex("0x1.0p-126")
    assert _base_cfg(gate_temperature=minimum_normal).gate_temperature == minimum_normal
    with pytest.raises(ValueError, match="gate_temperature"):
        _base_cfg(gate_temperature=float.fromhex("0x1.0p-149"))


def test_working_large_finite_diagnostics_are_operation_safe() -> None:
    memory = WorkingMemoryFeaturizer(_base_cfg())
    limit = jnp.asarray(jnp.finfo(jnp.float32).max, dtype=jnp.float32)
    state = dataclasses.replace(
        memory.init(),
        observation_traces=jnp.full((3, 2), limit, dtype=jnp.float32),
    )
    diagnostics = memory.diagnostics(state)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(diagnostics))
    assert float(diagnostics.trace_energy) > 0.0
    assert float(diagnostics.observation_energy) == float(limit)


def test_working_large_finite_innovation_rolls_back_step() -> None:
    memory = WorkingMemoryFeaturizer(
        _base_cfg(include_innovations=True, observation_decay_rates=(0.5,))
    )
    limit = jnp.asarray(jnp.finfo(jnp.float32).max, dtype=jnp.float32)
    state = dataclasses.replace(
        memory.init(), observation_traces=jnp.full((1, 2), -limit, dtype=jnp.float32)
    )
    result = memory.step(
        state,
        jnp.full((2,), limit, dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
    )
    assert not bool(result.update_applied)
    assert bool(jnp.all(result.features == 0.0))
    assert all(
        bool(equal)
        for equal in jax.tree.leaves(jax.tree.map(jnp.array_equal, result.state, state))
    )


def test_working_transform_resource_preflight_precedes_jax_conversion() -> None:
    memory = WorkingMemoryFeaturizer(_base_cfg())
    first_overflow = _INT32_MAX // (4 * memory.feature_dim() + 1) + 1

    class HostArray:
        dtype = np.dtype(np.float32)

        def __init__(self, width: int):
            self.shape = (first_overflow, width)

        def __jax_array__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("conversion must not run")

    with pytest.raises(ValueError, match="byte count"):
        transform_working_memory_arrays(
            memory,
            HostArray(2),  # type: ignore[arg-type]
            HostArray(1),  # type: ignore[arg-type]
            HostArray(1),  # type: ignore[arg-type]
        )
