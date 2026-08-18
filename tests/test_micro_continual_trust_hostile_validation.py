"""Trust-boundary validation for micro_continual sanitized errors."""

from __future__ import annotations

import json
import pathlib
from pathlib import Path

import numpy as np
import pytest

from alberta_framework.benchmarks.micro_continual import (
    MICRO_ARM_REGISTRY,
    MicroRunResult,
    MicroStreamConfig,
    _require_exact_str,
    load_micro_shard,
    micro_shard_payload,
)


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")

    def __hash__(self) -> int:  # type: ignore[override]
        raise AssertionError("EvilStr.__hash__ must not be called")


class _EvilStrNoHash(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStrNoHash.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStrNoHash.__repr__ must not be called")


class _StringSubclass(str):
    pass


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("name", _StringSubclass("x"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("arm_name", _StringSubclass("x"))  # type: ignore[arg-type]


def test_require_exact_str_hostile_without_repr_leak() -> None:
    evil = _EvilStr("evil")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_exact_str("name", evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_unknown_arm_sanitized() -> None:
    from alberta_framework.benchmarks.micro_continual import micro_arm_spec

    with pytest.raises(KeyError, match="unknown micro arm") as exc:
        micro_arm_spec("no_such_arm")
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'no_such_arm'" in msg
    assert "known:" in msg


def test_unknown_arm_hostile_blocked_before_hash() -> None:
    from alberta_framework.benchmarks.micro_continual import micro_arm_spec

    evil = _EvilStr("evil")
    with pytest.raises((ValueError, TypeError), match="must be an exact string") as exc:
        micro_arm_spec(evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    # NoHash variant must be rejected before formatting
    evil2 = _EvilStrNoHash("evil2")
    with pytest.raises((ValueError, TypeError), match="must be an exact string"):
        micro_arm_spec(evil2)  # type: ignore[arg-type]


def test_micro_shard_payload_hostile_before_hash() -> None:
    config = MicroStreamConfig(n_regimes=1, regime_length=1, dim=10)
    arm = next(iter(MICRO_ARM_REGISTRY.values()))
    # EvilStrNoHash to avoid hash assertion before gate
    evil = _EvilStrNoHash("evil")
    result = MicroRunResult(
        family=config.family,
        arm_name=arm.name,
        mechanism=arm.mechanism,
        hyperparameters=arm.hyperparameters,
        seed=0,
        hidden1=1,
        hidden2=1,
        stream_config=config,
        per_regime_accuracy=np.asarray([0.5]),
        per_regime_loss=np.asarray([0.5]),
        per_regime_plasticity=np.asarray([0.5]),
        overall_accuracy=0.5,
        wall_clock_seconds=0.1,
    )
    # Inject hostile via object.__setattr__ to bypass exact-str check in constructor
    object.__setattr__(result, "arm_name", evil)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        micro_shard_payload(result)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)


def test_suite_version_mismatch_sanitized(tmp_path: Path) -> None:
    config = MicroStreamConfig(n_regimes=1, regime_length=1, dim=10)
    arm = next(iter(MICRO_ARM_REGISTRY.values()))
    result = MicroRunResult(
        family=config.family,
        arm_name=arm.name,
        mechanism=arm.mechanism,
        hyperparameters=arm.hyperparameters,
        seed=0,
        hidden1=1,
        hidden2=1,
        stream_config=config,
        per_regime_accuracy=np.asarray([0.5]),
        per_regime_loss=np.asarray([0.5]),
        per_regime_plasticity=np.asarray([0.5]),
        overall_accuracy=0.5,
        wall_clock_seconds=0.1,
    )
    payload = micro_shard_payload(result)
    payload["suite_version"] = "evil_suite"
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="suite_version mismatch") as exc:
        load_micro_shard(path)
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_suite'" in msg
    # hostile suite_version is handled without repr leak via fallback placeholder
    # (payload with non-string suite_version is rejected with sanitized message)


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/benchmarks/micro_continual.py")
    text = p.read_text(encoding="utf-8")
    assert "unknown micro arm {name!r}" not in text
    assert "arm_name {result.arm_name!r}" not in text
    assert "got {number!r}" not in text
    assert "suite_version mismatch (expected {MICRO_GAUSS_SUITE_VERSION!r}" not in text
    assert "unknown arm {arm_name!r}" not in text
    assert "mechanism does not match registered arm {arm_spec.name!r}" not in text
    assert "hyperparameters do not match registered arm {arm_spec.name!r}" not in text
    assert "family {payload.get('family')!r}" not in text
    assert "arm {arm_name!r} has inconsistent" not in text
    assert "context=f\"arm {arm_name!r}\"" not in text
    # sanitized forms exist
    assert "unknown micro arm '{host_name}'" in text
    assert "arm_name '{host_arm}'" in text
    assert "suite_version mismatch (expected '{MICRO_GAUSS_SUITE_VERSION}'" in text


def test_valid_still_passes() -> None:
    assert _require_exact_str("name", "ok") == "ok"
    from alberta_framework.benchmarks.micro_continual import micro_arm_spec

    spec = micro_arm_spec("sgd_raw")
    assert spec.name == "sgd_raw"
