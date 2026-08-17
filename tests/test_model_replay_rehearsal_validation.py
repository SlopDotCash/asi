"""Focused validation for model-replay rehearsal composition."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from alberta_framework.core.dual_replay import DualReplayConfig
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.model_replay_rehearsal import (
    ModelReplayRehearsal,
    ModelReplayRehearsalConfig,
    _require_float32_resource,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import WorldModelEnsembleConfig

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 4_294_967_295


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr executed")


class _HostileStr(str):
    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr executed")


class _ClassSpoof:
    @property  # type: ignore[misc]
    def __class__(self) -> type:  # type: ignore[no-untyped-def]
        return str

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr executed")


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook must not run")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:  # type: ignore[override]
        type(self).calls += 1
        raise RuntimeError("ratio hook")


def _ensemble(
    *,
    observation_dim: int = 2,
    n_actions: int = 2,
    ensemble_size: int = 2,
    **overrides: Any,
) -> WorldModelEnsembleConfig:
    target_dim = observation_dim + 2
    model = ActionConditionedWorldModelConfig(
        observation_dim=observation_dim,
        n_actions=n_actions,
        hidden_sizes=(),
        gamma=0.95,
        step_size=0.05,
        sparsity=0.0,
        use_layer_norm=False,
        error_decay=0.8,
    )
    signals = LearningSignalEstimatorConfig(
        ensemble_size=ensemble_size,
        target_dim=target_dim,
        progress_warmup_steps=2,
        change_calibration_steps=2,
        fast_loss_decay=0.5,
        slow_loss_decay=0.9,
        max_input_magnitude=100.0,
        max_predicted_variance=10_000.0,
        max_observed_loss=10_000.0,
    )
    base: dict[str, Any] = {
        "model": model,
        "signal_estimator": signals,
        "ensemble_size": ensemble_size,
        "bootstrap_probability": 0.5,
        "residual_variance_decay": 0.8,
        "residual_variance_warmup_steps": 1,
        "residual_variance_floor": 1e-6,
    }
    base.update(overrides)
    if "ensemble_size" in overrides:
        es = int(overrides["ensemble_size"])  # type: ignore[arg-type]
        # keep signals in sync unless explicitly overridden
        if "signal_estimator" not in overrides:
            signals = LearningSignalEstimatorConfig(
                ensemble_size=es,
                target_dim=target_dim,
                progress_warmup_steps=2,
                change_calibration_steps=2,
                fast_loss_decay=0.5,
                slow_loss_decay=0.9,
                max_input_magnitude=100.0,
                max_predicted_variance=10_000.0,
                max_observed_loss=10_000.0,
            )
            base["signal_estimator"] = signals
    return WorldModelEnsembleConfig(**base)  # type: ignore[arg-type]


def _replay(
    *,
    observation_dim: int = 2,
    action_dim: int = 2,
    total_capacity: int = 2,
    short_term_capacity: int = 1,
    short_term_sample_size: int = 1,
    long_term_sample_size: int = 1,
) -> DualReplayConfig:
    return DualReplayConfig(
        total_capacity=total_capacity,
        short_term_capacity=short_term_capacity,
        observation_dim=observation_dim,
        action_dim=action_dim,
        short_term_sample_size=short_term_sample_size,
        long_term_sample_size=long_term_sample_size,
    )


def _composer(**overrides: Any) -> ModelReplayRehearsalConfig:
    ensemble = _ensemble()
    replay = _replay()
    values: dict[str, Any] = {
        "ensemble": ensemble,
        "replay": replay,
        "action_encoding": "one_hot",
    }
    values.update(overrides)
    return ModelReplayRehearsalConfig(**values)  # type: ignore[arg-type]


def test_composer_accepts_both_encodings() -> None:
    one_hot = _composer(action_encoding="one_hot")
    assert one_hot.action_encoding == "one_hot"
    scalar = _composer(replay=_replay(action_dim=1), action_encoding="scalar_index")
    assert scalar.action_encoding == "scalar_index"
    # round-trip via mapping preserves exact strings
    payload = one_hot.to_config()
    restored = ModelReplayRehearsalConfig.from_config(MappingProxyType(payload))
    assert restored.action_encoding == "one_hot"


def test_action_encoding_requires_exact_str_without_repr() -> None:
    with pytest.raises(ValueError, match="action_encoding"):
        _composer(action_encoding=_HostileStr("one_hot"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="action_encoding"):
        _composer(action_encoding=_ClassSpoof())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _composer(action_encoding=_RaisingRepr())  # type: ignore[arg-type]
    for bad in (True, 1, 1.0, None, b"one_hot"):
        with pytest.raises(ValueError, match="action_encoding"):
            _composer(action_encoding=bad)  # type: ignore[arg-type]


def test_composer_rejects_invalid_encoding_value() -> None:
    with pytest.raises(ValueError, match="action_encoding"):
        _composer(action_encoding="invalid")  # type: ignore[arg-type]


def test_composer_rejects_ensemble_replay_type_mismatch() -> None:
    with pytest.raises(TypeError, match="ensemble"):
        ModelReplayRehearsalConfig(
            ensemble="not_ensemble",  # type: ignore[arg-type]
            replay=_replay(),
            action_encoding="one_hot",
        )
    with pytest.raises(TypeError, match="replay"):
        ModelReplayRehearsalConfig(
            ensemble=_ensemble(),
            replay="not_replay",  # type: ignore[arg-type]
            action_encoding="one_hot",
        )


def test_composer_rejects_observation_dim_mismatch() -> None:
    ensemble = _ensemble(observation_dim=3)
    replay = _replay(observation_dim=2, action_dim=2)
    with pytest.raises(ValueError, match="observation dimensions must match"):
        ModelReplayRehearsalConfig(
            ensemble=ensemble, replay=replay, action_encoding="one_hot"
        )


def test_composer_rejects_n_actions_exceeds_exact_float32() -> None:
    # _MAX_EXACT_FLOAT32_INTEGER is 16_777_216
    ensemble = _ensemble(n_actions=16_777_217)
    replay = _replay(action_dim=16_777_217)
    with pytest.raises(ValueError, match="exceeds exact float32"):
        ModelReplayRehearsalConfig(
            ensemble=ensemble, replay=replay, action_encoding="one_hot"
        )


def test_composer_rejects_action_dim_mismatch_for_encodings() -> None:
    ensemble = _ensemble(n_actions=4)
    # scalar_index requires action_dim == 1
    with pytest.raises(ValueError, match="scalar_index"):
        ModelReplayRehearsalConfig(
            ensemble=ensemble,
            replay=_replay(action_dim=2),
            action_encoding="scalar_index",
        )
    # one_hot requires action_dim == n_actions
    with pytest.raises(ValueError, match="one_hot"):
        ModelReplayRehearsalConfig(
            ensemble=ensemble,
            replay=_replay(action_dim=3),
            action_encoding="one_hot",
        )


def test_int_validators_reject_hostile_subclass_without_hook() -> None:
    # Directly exercise the hostile int path via DualReplayConfig (shared gate)
    with pytest.raises(ValueError, match="must be an integer"):
        _replay(action_dim=_HostileInt(2))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        _ensemble(ensemble_size=_HostileInt(2))  # type: ignore[arg-type]


def test_int_validators_do_not_run_repr_hook() -> None:
    with pytest.raises(ValueError):
        _replay(action_dim=_RaisingRepr())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _ensemble(ensemble_size=_RaisingRepr())  # type: ignore[arg-type]


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
def test_numpy_integer_family_canonicalizes(np_type: type) -> None:
    cfg = _composer(
        ensemble=_ensemble(ensemble_size=np_type(2)),
        replay=_replay(action_dim=np_type(2)),
    )
    assert cfg.ensemble.ensemble_size == 2
    assert type(cfg.ensemble.ensemble_size) is int
    assert cfg.replay.action_dim == 2
    assert type(cfg.replay.action_dim) is int


def test_float_hostile_ratio_is_suppressed() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="bootstrap_probability"):
        _ensemble(bootstrap_probability=_HostileFloat(0.5))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_require_float32_resource_boundaries_without_allocation() -> None:
    legal = _INT32_MAX // 4
    _require_float32_resource("test", vector_scalars=legal)
    with pytest.raises(ValueError, match="byte count must fit signed int32"):
        _require_float32_resource("test", vector_scalars=legal + 1)
    with pytest.raises(ValueError, match="scalar count must fit signed int32"):
        _require_float32_resource("test", vector_scalars=_INT32_MAX + 1)
    with pytest.raises(ValueError, match="scalar count must fit signed int32"):
        _require_float32_resource("test", vector_scalars=_INT32_MAX, fixed_scalars=1)


def test_composer_persistent_bytes_preflight_without_allocation() -> None:
    # Small composer is well within limits
    comp = _composer()
    rehearsal = ModelReplayRehearsal(comp)
    budget = rehearsal.resource_budget()
    assert budget.persistent_state_bytes <= _INT32_MAX
    assert budget.persistent_state_bytes <= _UINT32_MAX
    # Helper reflects the same bound used in __init__/resource_budget
    _require_float32_resource(
        "ModelReplayRehearsal state",
        vector_scalars=budget.persistent_state_scalars,
    )


def test_config_rejects_derived_per_event_candidate_overflow_before_allocation() -> None:
    ensemble = _ensemble(ensemble_size=50_000)
    replay = _replay(
        total_capacity=100_000,
        short_term_capacity=50_000,
        short_term_sample_size=50_000,
        long_term_sample_size=50_000,
    )
    with pytest.raises(ValueError, match="per-event model update candidates"):
        ModelReplayRehearsalConfig(
            ensemble=ensemble,
            replay=replay,
            action_encoding="one_hot",
        )


def test_from_config_requires_exact_schema() -> None:
    cfg = _composer()
    payload = cfg.to_config()
    restored = ModelReplayRehearsalConfig.from_config(MappingProxyType(payload))
    assert restored.to_config() == payload
    bad = dict(payload)
    bad["type"] = "Wrong"
    with pytest.raises(ValueError, match="type"):
        ModelReplayRehearsalConfig.from_config(bad)
    bad2 = dict(payload)
    bad2["extra"] = 1
    with pytest.raises(ValueError, match="fields do not match"):
        ModelReplayRehearsalConfig.from_config(bad2)


def test_hostile_mapping_is_normalized() -> None:
    from collections.abc import Mapping

    class HostileMapping(Mapping[str, Any]):  # type: ignore[type-arg]
        def __getitem__(self, key: str) -> Any:
            raise RuntimeError("hook")

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("hook")

        def __len__(self) -> int:
            return 1

    with pytest.raises(ValueError, match="mapping"):
        ModelReplayRehearsal(  # type: ignore[arg-type]
            ModelReplayRehearsalConfig.from_config(HostileMapping())  # type: ignore[arg-type]
        )
