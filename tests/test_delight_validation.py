"""Validation hardening for delight configs (float bounds + choice/bool + mappings)."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from alberta_framework.core.delight import (
    DelightfulPolicyGradientConfig,
    GradientJoyConfig,
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


def _grad_cfg(**overrides: object) -> GradientJoyConfig:
    base: dict[str, object] = {
        "candidate_semantics": "gradient",
        "gradient_step_size": 1.0,
        "max_update_norm": 1.0,
        "min_objective_decrease": 0.0,
        "max_retention_loss_increase": 0.0,
        "max_safety_cost_increase": 0.0,
        "min_objective_descent_alignment": 0.0,
        "min_retention_descent_alignment": 0.0,
        "min_safety_descent_alignment": 0.0,
        "alignment_temperature": 0.1,
        "norm_temperature": 0.1,
        "diagnostics_epsilon": 1.0e-8,
    }
    base.update(overrides)
    return GradientJoyConfig(**base)  # type: ignore[arg-type]


def _delight_cfg(**overrides: object) -> DelightfulPolicyGradientConfig:
    base: dict[str, object] = {
        "mode": "delightful_pg",
        "temperature": 1.0,
        "actor_trace_lambda": 0.0,
        "diagnostics_epsilon": 1.0e-8,
        "kondo_enabled": False,
    }
    base.update(overrides)
    return DelightfulPolicyGradientConfig(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hostile float / class spoof
# ---------------------------------------------------------------------------

def test_delight_float_validators_reject_hostile_subclass_without_running_hook() -> None:
    class HostileFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            raise RuntimeError("untrusted ratio hook executed")

    class ClassSpoof:
        @property
        def __class__(self):  # type: ignore[no-untyped-def]
            return float

        def __float__(self) -> float:  # pragma: no cover
            return 0.5

    with pytest.raises(ValueError, match="must be a finite real number"):
        _grad_cfg(gradient_step_size=HostileFloat(0.5))
    with pytest.raises(ValueError, match="must be a finite real number"):
        _grad_cfg(gradient_step_size=ClassSpoof())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a finite real number"):
        _delight_cfg(temperature=HostileFloat(0.5))
    with pytest.raises(ValueError, match="must be a finite real number"):
        _delight_cfg(temperature=ClassSpoof())  # type: ignore[arg-type]


def test_delight_float_validators_do_not_run_repr_hook() -> None:
    with pytest.raises(ValueError):
        _delight_cfg(kondo_enabled=_RaisingRepr())  # type: ignore[arg-type]


def test_delight_float_validators_reject_hostile_ratio_without_calling() -> None:
    class HostileFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            type(self).calls += 1
            raise RuntimeError("ratio hook")

    with pytest.raises(ValueError, match="gradient_step_size"):
        _grad_cfg(gradient_step_size=HostileFloat(0.5))  # type: ignore[arg-type]
    assert HostileFloat.calls == 0
    HostileFloat.calls = 0
    with pytest.raises(ValueError, match="temperature"):
        _delight_cfg(temperature=HostileFloat(0.5))  # type: ignore[arg-type]
    assert HostileFloat.calls == 0


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
def test_delight_float_validators_canonicalize_numpy_int_scalars(np_type: type) -> None:
    cfg = _grad_cfg(gradient_step_size=np_type(1), alignment_temperature=np_type(1))
    assert cfg.gradient_step_size == 1.0
    assert type(cfg.gradient_step_size) is float


@pytest.mark.parametrize(
    "np_type",
    [
        np.float16,
        np.float32,
        np.float64,
    ],
)
def test_delight_float_validators_canonicalize_numpy_float_scalars(np_type: type) -> None:
    cfg = _grad_cfg(gradient_step_size=np_type(0.5))
    assert cfg.gradient_step_size == pytest.approx(0.5)
    assert type(cfg.gradient_step_size) is float
    cfg2 = _delight_cfg(temperature=np_type(0.5))
    assert cfg2.temperature == pytest.approx(0.5)


def test_delight_float_validators_reject_nonfinite_and_domain() -> None:
    class HostileFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            raise RuntimeError

    bad_grad = [
        ("gradient_step_size", float("nan")),
        ("gradient_step_size", float("inf")),
        ("gradient_step_size", 0.0),
        ("gradient_step_size", -1.0),
        ("gradient_step_size", HostileFloat(0.5)),
        ("max_update_norm", 0.0),
        ("alignment_temperature", 0.0),
        ("norm_temperature", -0.1),
        ("diagnostics_epsilon", 0.0),
        ("min_objective_descent_alignment", 1.5),
        ("min_objective_descent_alignment", -1.5),
        ("min_objective_descent_alignment", float("nan")),
    ]
    for field, bad in bad_grad:
        with pytest.raises(ValueError, match=field):
            _grad_cfg(**{field: bad})  # type: ignore[arg-type]

    bad_delight = [
        ("temperature", float("nan")),
        ("temperature", 0.0),
        ("temperature", -1.0),
        ("temperature", HostileFloat(0.5)),
        ("diagnostics_epsilon", 0.0),
        ("actor_trace_lambda", 1.0),
        ("actor_trace_lambda", float("nan")),
    ]
    for field, bad in bad_delight:
        with pytest.raises(ValueError, match=field):
            _delight_cfg(**{field: bad})  # type: ignore[arg-type]


def test_delight_float_validators_reject_bool_and_string() -> None:
    for ctor in [_grad_cfg, _delight_cfg]:
        for bad in [True, np.bool_(True), "0.5", None]:
            with pytest.raises(ValueError, match="must be"):
                ctor(gradient_step_size=bad) if ctor is _grad_cfg else ctor(temperature=bad)  # type: ignore[arg-type]


def test_delight_float_validators_reject_subnormal_flush_to_zero() -> None:
    # 1e-45 flushes to zero in float32, 1e-44 does not
    with pytest.raises(ValueError, match="must remain nonzero"):
        _grad_cfg(gradient_step_size=1e-45)
    with pytest.raises(ValueError, match="must remain nonzero"):
        _delight_cfg(temperature=1e-45)
    # 1e-44 is smallest non-flush subnormal-ish that passes
    _grad_cfg(gradient_step_size=1e-44)
    _delight_cfg(temperature=1e-44)


def test_delight_float_validators_accept_valid_values() -> None:
    cfg = _grad_cfg(
        gradient_step_size=0.5,
        max_update_norm=2.0,
        alignment_temperature=0.2,
        min_objective_descent_alignment=0.5,
        min_retention_descent_alignment=-0.5,
        diagnostics_epsilon=1e-7,
    )
    assert cfg.gradient_step_size == 0.5
    cfg2 = _delight_cfg(temperature=0.5, diagnostics_epsilon=1e-7)
    assert cfg2.temperature == 0.5
    # Fraction accepted
    from fractions import Fraction

    cfg3 = _grad_cfg(gradient_step_size=Fraction(1, 2))
    assert cfg3.gradient_step_size == 0.5


def test_delight_choice_validators_reject_non_exact() -> None:
    class StringSubclass(str):
        pass

    with pytest.raises(ValueError, match="is unsupported"):
        _grad_cfg(candidate_semantics=StringSubclass("gradient"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is unsupported"):
        _grad_cfg(candidate_semantics="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is unsupported"):
        _delight_cfg(mode=StringSubclass("delightful_pg"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is unsupported"):
        _delight_cfg(mode="bad_mode")  # type: ignore[arg-type]
    # Hostile int subclass must not run hook
    with pytest.raises(ValueError, match="is unsupported"):
        _grad_cfg(candidate_semantics=_LyingIntSubclass(1))  # type: ignore[arg-type]


def test_delight_bool_validators_reject_non_bool() -> None:
    for bad in [1, 0, "true", np.bool_(True), 1.0]:
        with pytest.raises(ValueError, match="must be.*bool"):
            _delight_cfg(kondo_enabled=bad)  # type: ignore[arg-type]
    # True kondo is rejected with specific message but still bool
    with pytest.raises(ValueError, match="Kondo"):
        _delight_cfg(kondo_enabled=True)


def test_delight_mapping_loaders_preserve_markers_and_exact_keys() -> None:
    # GradientJoyConfig
    config = _grad_cfg()
    payload = config.to_config()
    restored = GradientJoyConfig.from_config(MappingProxyType(payload))
    assert restored == config
    with pytest.raises(ValueError, match="type"):
        GradientJoyConfig.from_config({**payload, "type": "wrong"})
    without_type = dict(payload)
    without_type.pop("type")
    with pytest.raises(ValueError, match="type"):
        GradientJoyConfig.from_config(without_type)
    with pytest.raises(ValueError, match="unsupported fields"):
        GradientJoyConfig.from_config({**payload, "unknown": 1})
    # Delightful
    dconfig = _delight_cfg()
    dpayload = dconfig.to_config()
    drestored = DelightfulPolicyGradientConfig.from_config(MappingProxyType(dpayload))
    assert drestored == dconfig
    with pytest.raises(ValueError, match="type"):
        DelightfulPolicyGradientConfig.from_config({**dpayload, "type": "wrong"})
    without_type = dict(dpayload)
    without_type.pop("type")
    with pytest.raises(ValueError, match="type"):
        DelightfulPolicyGradientConfig.from_config(without_type)
    with pytest.raises(ValueError, match="unsupported fields"):
        DelightfulPolicyGradientConfig.from_config({**dpayload, "unknown": 1})

    class StringSubclass(str):
        pass

    with pytest.raises(ValueError, match="exact strings"):
        GradientJoyConfig.from_config({StringSubclass("type"): "GradientJoyConfig", **payload})
    with pytest.raises(ValueError, match="exact strings"):
        DelightfulPolicyGradientConfig.from_config(
            {StringSubclass("type"): "DelightfulPolicyGradientConfig", **dpayload}
        )
    class HostileStringValue(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("hostile equality hook executed")

    with pytest.raises(ValueError, match="type"):
        GradientJoyConfig.from_config(
            {**payload, "type": HostileStringValue("GradientJoyConfig")}
        )

    class HostileMapping(dict):  # type: ignore[type-arg]
        def __iter__(self):  # type: ignore[override]
            raise RuntimeError("iter hook")

        def __getitem__(self, key):  # type: ignore[override]
            raise RuntimeError("get hook")

    with pytest.raises(ValueError, match="could not be read"):
        GradientJoyConfig.from_config(HostileMapping(payload))  # type: ignore[arg-type]


def test_delight_mapping_preserves_fraction_and_numpy() -> None:
    from fractions import Fraction

    cfg = _grad_cfg(gradient_step_size=Fraction(1, 4))
    payload = cfg.to_config()
    # to_config stores float, roundtrip should preserve
    restored = GradientJoyConfig.from_config(payload)
    assert restored.gradient_step_size == 0.25


def test_delight_max_update_norm_gt_diagnostics_epsilon() -> None:
    with pytest.raises(ValueError, match="greater than diagnostics_epsilon"):
        _grad_cfg(max_update_norm=1e-8, diagnostics_epsilon=1e-8)
    with pytest.raises(ValueError, match="greater than diagnostics_epsilon"):
        _grad_cfg(max_update_norm=1e-9, diagnostics_epsilon=1e-8)
    # valid when strictly greater
    _grad_cfg(max_update_norm=2e-8, diagnostics_epsilon=1e-8)


def test_delight_min_objective_decrease_nonnegative() -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        _grad_cfg(min_objective_decrease=-1e-8)
    _grad_cfg(min_objective_decrease=0.0)
