"""Validation hardening for world-model ensemble (int/float bounds + resources)."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import WorldModelEnsembleConfig

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


def _base_cfg(**overrides: object) -> WorldModelEnsembleConfig:
    model = ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        gamma=0.95,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        use_layer_norm=False,
        error_decay=0.8,
    )
    signals = LearningSignalEstimatorConfig(
        ensemble_size=2,
        target_dim=4,
        progress_warmup_steps=2,
        change_calibration_steps=2,
        fast_loss_decay=0.5,
        slow_loss_decay=0.9,
        max_input_magnitude=100.0,
        max_predicted_variance=10_000.0,
        max_observed_loss=10_000.0,
    )
    base: dict[str, object] = {
        "model": model,
        "signal_estimator": signals,
        "ensemble_size": 2,
        "bootstrap_probability": 0.5,
        "residual_variance_decay": 0.8,
        "residual_variance_warmup_steps": 1,
        "residual_variance_floor": 1.0e-6,
    }
    base.update(overrides)
    # Keep signal_estimator in sync when ensemble_size overridden
    if "ensemble_size" in overrides:
        signals = LearningSignalEstimatorConfig(
            ensemble_size=overrides["ensemble_size"],  # type: ignore[arg-type]
            target_dim=4,
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


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _base_cfg(ensemble_size=v),
        lambda v: _base_cfg(residual_variance_warmup_steps=v),
    ],
)
def test_wm_ensemble_int_validators_reject_hostile_subclass_without_running_hook(
    ctor,
) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        ctor(_LyingIntSubclass(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        ctor(_RaisingIntSubclass(4))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _base_cfg(ensemble_size=v),
        lambda v: _base_cfg(residual_variance_warmup_steps=v),
    ],
)
def test_wm_ensemble_int_validators_do_not_run_repr_hook(ctor) -> None:
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
def test_wm_ensemble_int_validators_canonicalize_numpy_scalars(
    np_type: type,
) -> None:
    cfg = _base_cfg(
        ensemble_size=np_type(2),
        residual_variance_warmup_steps=np_type(2),
    )
    assert cfg.ensemble_size == 2
    assert type(cfg.ensemble_size) is int
    assert type(cfg.residual_variance_warmup_steps) is int


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _base_cfg(ensemble_size=v),
        lambda v: _base_cfg(residual_variance_warmup_steps=v),
    ],
)
@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), 4.0, np.float64(4.0), "4", None, 0, -1, _INT32_MAX + 1],
)
def test_wm_ensemble_int_validators_reject_non_integer_and_out_of_range(
    ctor,
    value: object,
) -> None:
    # ensemble_size min 2, warmup min 1 -> 0/1 invalid for ensemble_size
    with pytest.raises(ValueError, match="must be"):
        ctor(value)  # type: ignore[arg-type]


def test_wm_ensemble_float_validators_reject_nonfinite_and_hostile() -> None:
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
        ("bootstrap_probability", float("nan")),
        ("bootstrap_probability", float("inf")),
        ("bootstrap_probability", 0.0),
        ("bootstrap_probability", 1.0),
        ("bootstrap_probability", -0.1),
        ("bootstrap_probability", HostileFloat(0.5)),
        ("residual_variance_decay", float("nan")),
        ("residual_variance_decay", 1.0),
        ("residual_variance_decay", -0.1),
        ("residual_variance_decay", ClassSpoof()),  # type: ignore[arg-type]
        ("residual_variance_floor", 0.0),
        ("residual_variance_floor", float("inf")),
        ("residual_variance_floor", HostileFloat(0.5)),
    ]:
        with pytest.raises(ValueError, match=field):
            _base_cfg(**{field: bad})  # type: ignore[arg-type]


def test_wm_ensemble_float_validators_reject_hostile_ratio() -> None:
    class HostileFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            type(self).calls += 1
            raise RuntimeError("ratio hook")

    with pytest.raises(ValueError, match="bootstrap_probability"):
        _base_cfg(bootstrap_probability=HostileFloat(0.5))  # type: ignore[arg-type]
    assert HostileFloat.calls == 1


def test_wm_ensemble_dimensions_preflight_without_allocation() -> None:
    with pytest.raises(ValueError, match="dimensions must fit signed|scalar count|byte count"):
        _base_cfg(ensemble_size=_INT32_MAX)
    # Large ensemble with observation_dim 2 => target_dim 4, product 4*ensemble
    with pytest.raises(ValueError, match="dimensions must fit signed|scalar count|byte count"):
        _base_cfg(ensemble_size=600_000_000)


def test_wm_ensemble_state_preflight_bytes_without_allocation() -> None:
    # target_dim=4, total=ensemble*4, fixed=ensemble*2+10, persistent=4*(6*ensemble+10)
    last_legal = (2**31 - 1 - 40) // 24
    _base_cfg(ensemble_size=last_legal)
    with pytest.raises(ValueError, match="scalar count|byte count"):
        _base_cfg(ensemble_size=last_legal + 1)
    with pytest.raises(ValueError, match="scalar count|byte count|dimensions"):
        _base_cfg(ensemble_size=500_000_000)


def test_wm_ensemble_float_validators_accept_valid_values() -> None:
    cfg = _base_cfg(
        bootstrap_probability=0.5,
        residual_variance_decay=0.8,
        residual_variance_floor=1e-6,
    )
    assert cfg.bootstrap_probability == 0.5
    assert cfg.residual_variance_decay == 0.8
