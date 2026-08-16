"""Validation hardening for working memory (int/float/bool bounds + resource)."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.core.working_memory import WorkingMemoryConfig

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
    # With obs_dim variable, trace = 3*obs, total_state=3*obs+4, byte=12*obs+16
    last_legal = (_INT32_MAX - 16) // 12
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
