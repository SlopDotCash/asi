"""Hostile validation for micro continual trust boundary."""
# mypy: disable-error-code="arg-type"

from __future__ import annotations

import json
import pathlib

import pytest

from alberta_framework.benchmarks.micro_continual import (
    _freeze_micro_hyperparameters,
    _require_exact_str,
    _validated_curve,
    micro_arm_spec,
    micro_shard_payload,
)


class _EvilStr(str):
    calls = 0

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


class _HostileInt(int):
    calls = 0

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("HostileInt.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("HostileInt.__repr__ must not be called")


def test_require_exact_str_rejects_evil() -> None:
    evil = _EvilStr("v")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        _require_exact_str("key", evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_str("key", _StringSubclass("v"))


def test_freeze_rejects_evil_key_without_hooks() -> None:
    evil = _EvilStr("k")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="non-empty strings") as exc:
        _freeze_micro_hyperparameters({evil: 1.0}, context="ctx")
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_freeze_rejects_subclass_key() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        _freeze_micro_hyperparameters({_StringSubclass("k"): 1.0}, context="ctx")


def test_freeze_valid() -> None:
    assert dict(_freeze_micro_hyperparameters({"a": 1.0}, context="ctx")) == {"a": 1.0}


def test_micro_arm_spec_rejects_evil() -> None:
    evil = _EvilStr("bad")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        micro_arm_spec(evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_micro_arm_spec_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        micro_arm_spec(_StringSubclass("bad"))


def test_micro_arm_spec_sanitized() -> None:
    with pytest.raises(KeyError, match="unknown micro arm") as exc:
        micro_arm_spec("nope")
    assert "!r" not in str(exc.value)
    assert "nope" in str(exc.value)
    assert "'" in str(exc.value)


def test_micro_arm_spec_valid() -> None:
    from alberta_framework.benchmarks.micro_continual import MICRO_ARM_REGISTRY

    name = next(iter(MICRO_ARM_REGISTRY))
    spec = micro_arm_spec(name)
    assert spec.name == name


def test_shard_payload_rejects_evil_without_hooks() -> None:
    from alberta_framework.benchmarks.micro_continual import MicroRunResult, MicroStreamConfig

    cfg = MicroStreamConfig()
    evil = _EvilStr("bad_arm")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="non-empty string") as exc:
        MicroRunResult(
            family=cfg.family,
            arm_name=evil,
            mechanism="m",
            hyperparameters={},
            seed=0,
            hidden1=1,
            hidden2=1,
            stream_config=cfg,
            per_regime_accuracy=[0.5],
            per_regime_loss=[0.5],
            per_regime_plasticity=[0.5],
            overall_accuracy=0.5,
            wall_clock_seconds=0.1,
        )
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_shard_payload_rejects_subclass() -> None:
    from alberta_framework.benchmarks.micro_continual import MicroRunResult, MicroStreamConfig

    cfg = MicroStreamConfig()
    with pytest.raises(ValueError, match="non-empty string"):
        MicroRunResult(
            family=cfg.family,
            arm_name=_StringSubclass("bad_arm"),
            mechanism="m",
            hyperparameters={},
            seed=0,
            hidden1=1,
            hidden2=1,
            stream_config=cfg,
            per_regime_accuracy=[0.5],
            per_regime_loss=[0.5],
            per_regime_plasticity=[0.5],
            overall_accuracy=0.5,
            wall_clock_seconds=0.1,
        )


def test_validated_curve_rejects_evil_context_without_hooks() -> None:
    # context is str, but number validation should sanitize
    with pytest.raises(ValueError, match="must lie in") as exc:
        _validated_curve([2.0], n_regimes=1, lower=0.0, upper=1.0, context="ctx")
    assert "!r" not in str(exc.value)
    assert "2.0" in str(exc.value)


def test_load_shard_suite_version_sanitized(tmp_path: pathlib.Path) -> None:
    from alberta_framework.benchmarks.micro_continual import (
        MICRO_ARM_REGISTRY,
        MicroRunResult,
        MicroStreamConfig,
        load_micro_shard,
    )

    # Create valid shard then corrupt suite_version with evil
    cfg = MicroStreamConfig()
    name = next(iter(MICRO_ARM_REGISTRY))
    spec = MICRO_ARM_REGISTRY[name]
    result = MicroRunResult(
        family=cfg.family,
        arm_name=spec.name,
        mechanism=spec.mechanism,
        hyperparameters=dict(spec.hyperparameters),
        seed=0,
        hidden1=1,
        hidden2=1,
        stream_config=cfg,
        per_regime_accuracy=[0.5] * cfg.n_regimes,
        per_regime_loss=[0.5] * cfg.n_regimes,
        per_regime_plasticity=[0.5] * cfg.n_regimes,
        overall_accuracy=0.5,
        wall_clock_seconds=0.1,
    )
    payload = micro_shard_payload(result)
    payload["suite_version"] = "bad-suite"
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="suite_version mismatch") as exc:
        load_micro_shard(path)
    assert "!r" not in str(exc.value)
    assert "bad-suite" in str(exc.value)
    assert "gauss-v1" in str(exc.value)


def test_load_shard_arm_sanitized(tmp_path: pathlib.Path) -> None:
    from alberta_framework.benchmarks.micro_continual import (
        MICRO_ARM_REGISTRY,
        MicroRunResult,
        MicroStreamConfig,
        load_micro_shard,
    )
    cfg = MicroStreamConfig()
    name = next(iter(MICRO_ARM_REGISTRY))
    spec = MICRO_ARM_REGISTRY[name]
    result = MicroRunResult(
        family=cfg.family,
        arm_name=spec.name,
        mechanism=spec.mechanism,
        hyperparameters=dict(spec.hyperparameters),
        seed=0,
        hidden1=1,
        hidden2=1,
        stream_config=cfg,
        per_regime_accuracy=[0.5] * cfg.n_regimes,
        per_regime_loss=[0.5] * cfg.n_regimes,
        per_regime_plasticity=[0.5] * cfg.n_regimes,
        overall_accuracy=0.5,
        wall_clock_seconds=0.1,
    )
    payload = micro_shard_payload(result)
    payload["arm_name"] = "bad_arm"
    path = tmp_path / "shard2.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unknown arm") as exc:
        load_micro_shard(path)
    assert "!r" not in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/benchmarks/micro_continual.py").read_text()
    assert "!r" not in text
