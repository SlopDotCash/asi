"""Validation hardening for dreaming configs and buffer (int/float bounds + resources)."""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.dreaming import (
    DreamingConfig,
    DreamRolloutConfig,
    DreamSelectionConfig,
    GuardedDreamer,
    RecentObservationBuffer,
    action_features,
    imagined_transition_to_supervised_item,
)

_INT32_MAX = 2**31 - 1


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook must not run")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr hook must not run")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
        type(self).calls += 1
        raise RuntimeError("ratio hook")


class _ClassSpoof:
    @property
    def __class__(self) -> type:  # type: ignore[no-untyped-def]
        return float  # type: ignore[return-value]

    def __float__(self) -> float:  # pragma: no cover
        return 0.1


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook must not run")


class _HostileMapping(dict):  # type: ignore[type-arg]
    def __iter__(self):  # type: ignore[override]
        raise RuntimeError("hostile iter")

    def __getitem__(self, key):  # type: ignore[override]
        raise RuntimeError("hostile getitem")

    def keys(self):  # type: ignore[override]
        raise RuntimeError("hostile keys")


def _dreaming_cfg(**overrides: object) -> DreamingConfig:
    base: dict[str, object] = {
        "warmup_steps": 100,
        "rollout_horizon": 1,
    }
    base.update(overrides)
    return DreamingConfig(**base)  # type: ignore[arg-type]


def _selection_cfg(**overrides: object) -> DreamSelectionConfig:
    base: dict[str, object] = {"max_items": 1}
    base.update(overrides)
    return DreamSelectionConfig(**base)  # type: ignore[arg-type]


def _rollout_cfg(**overrides: object) -> DreamRolloutConfig:
    base: dict[str, object] = {"rollout_horizon": 1}
    base.update(overrides)
    return DreamRolloutConfig(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _dreaming_cfg(warmup_steps=v),
        lambda v: _dreaming_cfg(rollout_horizon=v),
        lambda v: _selection_cfg(max_items=v),
        lambda v: _rollout_cfg(rollout_horizon=v),
        lambda v: RecentObservationBuffer(capacity=v, observation_dim=2),
        lambda v: RecentObservationBuffer(capacity=2, observation_dim=v),
        lambda v: action_features(
            __import__("jax.numpy").numpy.zeros(2), n_actions=v
        ),
    ],
)
def test_dreaming_int_validators_reject_hostile_without_hook(ctor) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        ctor(_HostileInt(4))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _dreaming_cfg(warmup_steps=v),
        lambda v: _dreaming_cfg(rollout_horizon=v),
        lambda v: _selection_cfg(max_items=v),
    ],
)
def test_dreaming_int_validators_do_not_run_repr_hook(ctor) -> None:
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
        np.longlong,
        np.ulonglong,
    ],
)
def test_dreaming_int_validators_canonicalize_numpy_scalars(np_type: type) -> None:
    cfg = _dreaming_cfg(warmup_steps=np_type(5), rollout_horizon=np_type(2))
    assert cfg.warmup_steps == 5
    assert type(cfg.warmup_steps) is int
    assert cfg.rollout_horizon == 2
    assert type(cfg.rollout_horizon) is int
    sel = _selection_cfg(max_items=np_type(3))
    assert sel.max_items == 3
    assert type(sel.max_items) is int
    buf = RecentObservationBuffer(capacity=np_type(4), observation_dim=np_type(3))
    assert buf.capacity == 4
    assert type(buf.capacity) is int


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _dreaming_cfg(warmup_steps=v),
        lambda v: _dreaming_cfg(rollout_horizon=v),
        lambda v: _selection_cfg(max_items=v),
    ],
)
@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), 4.0, np.float64(4.0), "4", None, -1, _INT32_MAX + 1],
)
def test_dreaming_int_validators_reject_non_integer_and_out_of_range(
    ctor, value: object
) -> None:
    # rollout_horizon/max_items min 1, warmup min 0 — 0 invalid for min1 (covered separately)
    with pytest.raises(ValueError, match="must be"):
        ctor(value)  # type: ignore[arg-type]


def test_dreaming_int_validators_reject_zero_for_min_one() -> None:
    with pytest.raises(ValueError, match="must be"):
        _dreaming_cfg(rollout_horizon=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be"):
        _selection_cfg(max_items=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be"):
        _rollout_cfg(rollout_horizon=0)  # type: ignore[arg-type]
    # warmup_steps 0 is valid
    assert _dreaming_cfg(warmup_steps=0).warmup_steps == 0


def test_dreaming_float_validators_reject_hostile_ratio() -> None:
    class HostileFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            type(self).calls += 1
            raise RuntimeError("ratio hook")

    with pytest.raises(ValueError, match="must be a finite"):
        _dreaming_cfg(max_model_error_ema=HostileFloat(0.5))  # type: ignore[arg-type]
    assert HostileFloat.calls == 0
    with pytest.raises(ValueError, match="must be a finite"):
        _selection_cfg(surprise_weight=HostileFloat(0.5))  # type: ignore[arg-type]
    assert HostileFloat.calls == 0
    with pytest.raises(ValueError, match="must be a finite"):
        _rollout_cfg(confidence_threshold=HostileFloat(0.5))  # type: ignore[arg-type]
    assert HostileFloat.calls == 0


def test_dreaming_float_validators_reject_spoof_and_nonfinite() -> None:
    for ctor, field, bad in [
        (_dreaming_cfg, "max_model_error_ema", float("nan")),
        (_dreaming_cfg, "max_model_error_ema", float("inf")),
        (_dreaming_cfg, "max_model_error_ema", -0.1),
        (_dreaming_cfg, "max_model_error_ema", _ClassSpoof()),  # type: ignore[arg-type]
        (_dreaming_cfg, "max_model_error_ema", _HostileFloat(0.5)),
        (_dreaming_cfg, "max_uncertainty", -0.1),
        (_selection_cfg, "surprise_weight", float("nan")),
        (_selection_cfg, "min_confidence", -0.1),
        (_selection_cfg, "surprise_weight", _ClassSpoof()),  # type: ignore[arg-type]
        (_rollout_cfg, "confidence_threshold", -0.1),
        (_rollout_cfg, "confidence_threshold", _HostileFloat(0.5)),
    ]:
        with pytest.raises(ValueError, match=field):
            ctor(**{field: bad})  # type: ignore[arg-type]


def test_dreaming_bool_validators_reject_non_bool() -> None:
    with pytest.raises(ValueError, match="must be a bool"):
        _dreaming_cfg(stop_on_terminal="true")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a bool"):
        _rollout_cfg(stop_on_terminal=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a bool"):
        DreamingConfig(warmup_steps=1, stop_on_terminal=1)  # type: ignore[arg-type]


def test_dreaming_require_float32_resource_boundaries() -> None:
    from alberta_framework.core.dreaming import _require_float32_resource

    legal = _INT32_MAX // 4
    _require_float32_resource("test", vector_scalars=legal)
    with pytest.raises(ValueError, match="byte count"):
        _require_float32_resource("test", vector_scalars=legal + 1)
    with pytest.raises(ValueError, match="scalar count"):
        _require_float32_resource("test", vector_scalars=_INT32_MAX + 1)


def test_dreaming_buffer_resource_preflight_without_allocation() -> None:
    legal = _INT32_MAX // 4
    # capacity*observation_dim byte overflow without allocation
    # choose capacity=legal, observation_dim=2 => scalars=~1B, bytes ~4B > INT32
    with pytest.raises(ValueError, match="byte count"):
        RecentObservationBuffer(capacity=legal, observation_dim=2)
    with pytest.raises(ValueError, match="scalar count"):
        RecentObservationBuffer(capacity=_INT32_MAX, observation_dim=2)
    # init also checks
    buf = RecentObservationBuffer(capacity=4, observation_dim=4)
    # mutate via object.__setattr__ to bypass __init__ validation, then init should still fail
    # Instead test that legal init succeeds and init allocates
    state = buf.init()
    assert state.observations.shape == (4, 4)


def test_dreaming_action_features_and_target_validation() -> None:
    # hostile int for n_actions
    with pytest.raises(ValueError, match="must be an integer"):
        action_features(jnp.zeros(2), n_actions=_HostileInt(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        action_features(jnp.zeros(2), n_actions=_RaisingRepr())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        action_features(jnp.zeros(2), n_actions=True)  # type: ignore[arg-type]
    # target unsupported should not leak repr
    from alberta_framework.core.dreaming import ImaginedTransition

    trans = ImaginedTransition(
        observation=jnp.zeros(2),
        action=jnp.array(0, dtype=jnp.int32),
        reward=jnp.array(0.0),
        next_observation=jnp.zeros(2),
        discount=jnp.array(0.9),
        terminated=jnp.array(False),
        confidence=jnp.array(1.0),
        model_error=jnp.array(0.0),
        behavior_probability=jnp.array(1.0),
        valid=jnp.array(True),
        step_index=jnp.array(0, dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match="is unsupported"):
        imagined_transition_to_supervised_item(trans, target="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is unsupported"):
        imagined_transition_to_supervised_item(
            trans, target=_HostileFloat(0.5)  # type: ignore[arg-type]
        )


def test_dreaming_mapping_loaders_preserve_markers_and_exact_keys() -> None:
    cfg = _dreaming_cfg()
    payload = cfg.to_config()
    restored = DreamingConfig.from_config(MappingProxyType(payload))
    assert restored.warmup_steps == cfg.warmup_steps
    assert restored.rollout_horizon == cfg.rollout_horizon
    with pytest.raises(ValueError, match="type"):
        DreamingConfig.from_config({**payload, "type": "wrong"})
    # String subclass key should be rejected
    class StringSubclass(str):
        pass

    bad = {StringSubclass("type"): "DreamingConfig", "warmup_steps": 1}
    with pytest.raises(ValueError, match="exact strings"):
        DreamingConfig.from_config(bad)  # type: ignore[arg-type]
    # Hostile mapping should be rejected
    hostile = _HostileMapping({"type": "DreamingConfig", "warmup_steps": 1})
    with pytest.raises(ValueError, match="payload could not be read"):
        DreamingConfig.from_config(hostile)  # type: ignore[arg-type]

    # DreamSelectionConfig
    sel = _selection_cfg()
    sel_payload = sel.to_config()
    assert DreamSelectionConfig.from_config(MappingProxyType(sel_payload)) == sel
    with pytest.raises(ValueError, match="exact strings"):
        DreamSelectionConfig.from_config(
            {StringSubclass("type"): "DreamSelectionConfig", "max_items": 1}
        )

    # DreamRolloutConfig
    roll = _rollout_cfg()
    roll_payload = roll.to_config()
    assert DreamRolloutConfig.from_config(MappingProxyType(roll_payload)) == roll

    # GuardedDreamer
    dreamer = GuardedDreamer(cfg)
    outer = dreamer.to_config()
    assert GuardedDreamer.from_config(MappingProxyType(outer)).config == cfg
    with pytest.raises(ValueError, match="exact strings"):
        GuardedDreamer.from_config(
            {StringSubclass("type"): "GuardedDreamer", "config": payload}
        )
    with pytest.raises(ValueError, match="payload has unknown fields"):
        GuardedDreamer.from_config({"type": "GuardedDreamer", "config": payload, "extra": 1})


@pytest.mark.parametrize(
    ("loader", "payload"),
    [
        (DreamingConfig.from_config, DreamingConfig(max_model_error=0.25).to_config()),
        (
            DreamSelectionConfig.from_config,
            DreamSelectionConfig(utility_weight=0.25).to_config(),
        ),
        (
            DreamRolloutConfig.from_config,
            DreamRolloutConfig(confidence_threshold=0.25).to_config(),
        ),
    ],
)
def test_dreaming_config_loaders_reject_every_missing_field(
    loader: Callable[[object], object], payload: dict[str, object]
) -> None:
    for field in payload.keys() - {"type"}:
        incomplete = dict(payload)
        del incomplete[field]
        with pytest.raises(ValueError, match="missing fields"):
            loader(incomplete)


def test_dreaming_valid_construction_and_roundtrip() -> None:
    cfg = DreamingConfig(
        warmup_steps=10,
        max_model_error_ema=0.5,
        max_uncertainty=0.8,
        rollout_horizon=2,
        stop_on_terminal=True,
    )
    assert cfg.warmup_steps == 10
    sel = DreamSelectionConfig(
        max_items=2, surprise_weight=0.5, min_confidence=0.1
    )
    assert sel.max_items == 2
    roll = DreamRolloutConfig(rollout_horizon=3, confidence_threshold=0.2)
    assert roll.rollout_horizon == 3
    buf = RecentObservationBuffer(capacity=8, observation_dim=4)
    state = buf.init()
    assert state.observations.shape == (8, 4)
    # numpy canonicalization should survive
    cfg2 = _dreaming_cfg(warmup_steps=np.int32(7))
    assert type(cfg2.warmup_steps) is int
    assert cfg2.warmup_steps == 7
    # float narrowing
    cfg3 = _dreaming_cfg(max_model_error_ema=0.5)
    assert cfg3.max_model_error_ema == 0.5
