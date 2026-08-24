"""Focused validation for PrototypeAgent integration (hostile + resource)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    _MAX_N_DREAMS_PER_STEP,
    GRUPerceptionConfig,
    PrototypeAgent,
    PrototypeAgentConfig,
    _require_float32_resource,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

_INT32_MAX = 2**31 - 1


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover - exact-type rejection must win
        raise AssertionError("integer hook executed")

    def __repr__(self) -> str:  # pragma: no cover - errors must not interpolate values
        raise AssertionError("repr executed")


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover - errors must not interpolate values
        raise AssertionError("repr executed")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover - must not run
        type(self).calls += 1
        raise RuntimeError("ratio hook executed")


class _ClassSpoof:
    @property  # type: ignore[misc]
    def __class__(self) -> type:  # type: ignore[no-untyped-def]
        return float  # type: ignore[return-value]

    def __float__(self) -> float:  # pragma: no cover - must not run
        return 0.1

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr executed")


def _oak(obs_dim: int = 4, n_prim: int = 2) -> OaKConfig:
    stomp = STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),),
        observation_dim=obs_dim,
        n_primitive_actions=n_prim,
    )
    return OaKConfig(stomp=stomp)


def _cfg(**overrides: Any) -> PrototypeAgentConfig:
    base: dict[str, Any] = {"oak": _oak()}
    base.update(overrides)
    return PrototypeAgentConfig(**base)  # type: ignore[arg-type]


def _world_model(observation_dim: int) -> ActionConditionedWorldModelConfig:
    return ActionConditionedWorldModelConfig(
        observation_dim=observation_dim,
        n_actions=2,
        hidden_sizes=(),
    )


def test_prototype_int_validators_reject_hostile_without_hook() -> None:
    for ctor in [
        lambda v: _cfg(buffer_capacity=v),
        lambda v: _cfg(n_dreams_per_step=v),
        lambda v: _cfg(auto_curate_every=v),
        lambda v: _cfg(horde_hidden_sizes=(v, 64)),
        lambda v: GRUPerceptionConfig(observation_dim=v, hidden_dim=4),
        lambda v: GRUPerceptionConfig(observation_dim=4, hidden_dim=v),
    ]:
        with pytest.raises(ValueError):
            ctor(_HostileInt(2))  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            ctor(_RaisingRepr())  # type: ignore[arg-type]


def test_prototype_int_validators_canonicalize_numpy_scalars() -> None:
    cfg = _cfg(
        buffer_capacity=np.int64(200),
        n_dreams_per_step=np.int32(0),
        auto_curate_every=np.int16(0),
        horde_hidden_sizes=(np.int64(32), np.int32(16)),
    )
    assert type(cfg.buffer_capacity) is int
    assert type(cfg.n_dreams_per_step) is int
    assert type(cfg.horde_hidden_sizes[0]) is int
    gru = GRUPerceptionConfig(observation_dim=np.int32(4), hidden_dim=np.int64(8))
    assert type(gru.observation_dim) is int


@pytest.mark.parametrize(
    "value", [True, np.bool_(True), 4.0, np.float64(4.0), "4", None, 0, -1, _INT32_MAX + 1]
)
def test_prototype_int_validators_reject_non_integer_and_out_of_range(value: object) -> None:
    with pytest.raises(ValueError, match="must be"):
        _cfg(buffer_capacity=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GRUPerceptionConfig(observation_dim=value, hidden_dim=4)  # type: ignore[arg-type]


def test_prototype_float_validators_reject_hostile_ratio() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="horde_step_size"):
        _cfg(horde_step_size=_HostileFloat(0.1))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_prototype_float_validators_reject_spoof_and_nonfinite() -> None:
    for bad in [
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.1,
        0.0,
        _ClassSpoof(),
        _HostileFloat(0.2),
    ]:
        with pytest.raises(ValueError, match="horde_step_size"):
            _cfg(horde_step_size=bad)  # type: ignore[arg-type]  # type: ignore[arg-type]


def test_prototype_hidden_sizes_hostile_and_range() -> None:
    with pytest.raises(ValueError, match="horde_hidden_sizes"):
        _cfg(horde_hidden_sizes=[32])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horde_hidden_sizes"):
        _cfg(horde_hidden_sizes=(_HostileInt(32),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horde_hidden_sizes"):
        _cfg(horde_hidden_sizes=(_RaisingRepr(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horde_hidden_sizes"):
        _cfg(horde_hidden_sizes=(0,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horde_hidden_sizes"):
        _cfg(horde_hidden_sizes=(1, _INT32_MAX + 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horde_hidden_sizes"):
        _cfg(horde_hidden_sizes="64")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horde_hidden_sizes"):
        _cfg(horde_hidden_sizes=(True,))  # type: ignore[arg-type]


def test_prototype_serialized_hidden_sizes_require_json_list() -> None:
    payload = _cfg(horde_hidden_sizes=(32, 16)).to_config()
    payload["horde_hidden_sizes"] = (32, 16)
    with pytest.raises(ValueError, match="actual list"):
        PrototypeAgentConfig.from_config(payload)


def test_prototype_require_float32_resource_boundaries() -> None:
    legal = _INT32_MAX // 4
    _require_float32_resource("x", vector_scalars=legal)
    with pytest.raises(ValueError, match="byte count"):
        _require_float32_resource("x", vector_scalars=legal + 1)
    with pytest.raises(ValueError, match="scalar count"):
        _require_float32_resource("x", vector_scalars=_INT32_MAX + 1)
    with pytest.raises(ValueError, match="non-negative"):
        _require_float32_resource("x", vector_scalars=-1)


def test_prototype_buffer_resource_preflight_without_allocation() -> None:
    # buffer: capacity * obs_dim + 2 must fit int32 scalar and byte
    obs_dim = 2
    oak = _oak(obs_dim=obs_dim)
    legal = (_INT32_MAX // 4 - 2) // obs_dim  # byte boundary
    cfg = _cfg(oak=oak, world_model=_world_model(obs_dim), buffer_capacity=legal)
    assert cfg.buffer_capacity == legal
    with pytest.raises(ValueError, match="byte count|scalar count"):
        _cfg(oak=oak, world_model=_world_model(obs_dim), buffer_capacity=legal + 1)
    with pytest.raises(ValueError, match="fit signed int32"):
        _cfg(oak=oak, world_model=_world_model(obs_dim), buffer_capacity=600_000_000)

    dormant = _cfg(oak=oak, buffer_capacity=600_000_000)
    assert dormant.world_model is None

    # GRU: 3*h*obs +3*h*h +4*h byte overflow
    gru = GRUPerceptionConfig(observation_dim=4, hidden_dim=4)
    assert gru.hidden_dim == 4


def test_prototype_gru_resource_overflow_without_allocation() -> None:
    with pytest.raises(ValueError, match="fit signed int32"):
        GRUPerceptionConfig(observation_dim=1_000_000, hidden_dim=1_000_000)


def test_prototype_horde_uses_exact_learner_resource_formula() -> None:
    horde = create_horde_spec(
        (
            GVFSpec(
                name="prediction",
                demon_type=DemonType.PREDICTION,
                cumulant_index=0,
                gamma=0.0,
                lamda=0.0,
            ),
        )
    )
    with pytest.raises(ValueError, match="direct_state_bytes"):
        _cfg(
            oak=_oak(obs_dim=2),
            horde_spec=horde,
            horde_hidden_sizes=(70_000_000,),
        )


def test_prototype_mapping_loaders_preserve_markers_and_exact_keys() -> None:
    cfg = _cfg()
    payload = cfg.to_config()
    restored = PrototypeAgentConfig.from_config(MappingProxyType(payload))
    assert restored.buffer_capacity == cfg.buffer_capacity

    class StringSubclass(str):
        pass

    with pytest.raises(ValueError, match="exact strings"):
        PrototypeAgentConfig.from_config(
            {StringSubclass("type"): "PrototypeAgentConfig", "oak": payload["oak"]}  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="unknown fields"):
        PrototypeAgentConfig.from_config({**payload, "extra": 1})

    gru_payload = GRUPerceptionConfig(observation_dim=4, hidden_dim=8).to_config()
    with pytest.raises(ValueError, match="fields"):
        GRUPerceptionConfig.from_config({**gru_payload, "extra": 1})
    with pytest.raises(ValueError, match="type"):
        GRUPerceptionConfig.from_config({**gru_payload, "type": "Wrong"})
    with pytest.raises(ValueError, match="fields"):
        GRUPerceptionConfig.from_config(
            {"type": "GRUPerceptionConfig", "hidden_dim": 8}
        )


def test_prototype_config_rejects_hostile_mapping() -> None:
    class HostileMapping(Mapping[str, Any]):
        def __getitem__(self, key: str) -> Any:
            raise RuntimeError("hook executed")

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError("hook executed")

        def __len__(self) -> int:
            return 1

    with pytest.raises(ValueError, match="could not be read"):
        PrototypeAgentConfig.from_config(HostileMapping())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="could not be read"):
        GRUPerceptionConfig.from_config(HostileMapping())  # type: ignore[arg-type]

    class MappingSpoof:
        calls = 0

        @property
        def __class__(self) -> type:
            return dict

        def __iter__(self) -> Iterator[str]:
            type(self).calls += 1
            raise AssertionError("mapping hook executed")

    with pytest.raises(ValueError, match="must be a mapping"):
        PrototypeAgentConfig.from_config(MappingSpoof())  # type: ignore[arg-type]
    assert MappingSpoof.calls == 0


def test_prototype_agent_requires_exact_config_type() -> None:
    class DerivedConfig(PrototypeAgentConfig):
        pass

    derived = DerivedConfig(oak=_oak())
    with pytest.raises(ValueError, match="exact PrototypeAgentConfig"):
        PrototypeAgent(derived)


def test_prototype_valid_construction() -> None:
    cfg = _cfg(
        oak=_oak(obs_dim=4),
        buffer_capacity=10,
        n_dreams_per_step=0,
        horde_hidden_sizes=(32, 16),
        horde_step_size=0.05,
        auto_curate_every=0,
    )
    assert cfg.buffer_capacity == 10
    assert cfg.horde_step_size > 0
    payload = cfg.to_config()
    restored = PrototypeAgentConfig.from_config(payload)
    assert restored == cfg
    gru = GRUPerceptionConfig(observation_dim=4, hidden_dim=4)
    assert gru.augmented_dim() == 8


def test_n_dreams_per_step_is_capped_before_it_drives_a_scan() -> None:
    """An INT32-legal dream count hangs the agent instead of being rejected.

    ``PrototypeAgent._dream`` scans ``jnp.arange(n_dreams_per_step)`` and
    materializes one float32 td-error per imagined transition, so a value near
    ``_INT32_MAX`` costs an 8.6 GB output before any dream executes. The
    config must fail closed at construction, matching the dream-rollout
    ceiling already enforced in ``core.dreaming``.
    """
    # Match the cap's own message: a bare "n_dreams_per_step" would also be
    # satisfied by the unrelated "requires the legacy world_model" guard.
    expected = f"n_dreams_per_step must be <= {_MAX_N_DREAMS_PER_STEP}"
    for oversized in (_INT32_MAX, _MAX_N_DREAMS_PER_STEP + 1):
        with pytest.raises(ValueError) as excinfo:
            _cfg(n_dreams_per_step=oversized, world_model=_world_model(4))
        assert str(excinfo.value) == expected

    # The ceiling itself, and the counts the suite actually exercises, stay legal.
    # ``n_dreams_per_step > 0`` additionally requires the legacy world model.
    dreaming = _cfg(
        n_dreams_per_step=_MAX_N_DREAMS_PER_STEP,
        world_model=_world_model(4),
    )
    assert dreaming.n_dreams_per_step == _MAX_N_DREAMS_PER_STEP
    assert _cfg(n_dreams_per_step=4, world_model=_world_model(4)).n_dreams_per_step == 4
