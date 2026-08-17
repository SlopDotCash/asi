"""Hostile validation for benchmarks/upgd_label_emnist trust-boundary."""

import json
import pathlib

import numpy as np
import pytest

from alberta_framework.benchmarks.upgd_label_emnist import (
    LabelEMNISTConfig,
    _require_exact_str,
    _strict_json_object,
    _validated_hyperparameter,
    resolve_hyperparameters,
)


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")

    def __hash__(self) -> int:  # type: ignore[override]
        raise AssertionError("EvilStr.__hash__ must not be called")


class _StringSubclass(str):
    pass


class _HostileInt(int):
    calls = 0

    def __index__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileInt.__index__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileInt.__repr__ must not be called")


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat.__float__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileFloat.__repr__ must not be called")


def test_validated_hyperparameter_rejects_string_subclass() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        _validated_hyperparameter(_StringSubclass("step_size"), 0.01)  # type: ignore[arg-type]


def test_validated_hyperparameter_hostile_str_without_repr_leak() -> None:
    evil = _EvilStr("step_size")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _validated_hyperparameter(evil, 0.01)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_validated_hyperparameter_does_not_invoke_hostile_value_when_name_evil() -> None:
    evil = _EvilStr("step_size")
    hostile = _HostileFloat(0.01)
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be an exact string"):
        _validated_hyperparameter(evil, hostile)  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_validated_hyperparameter_rejects_hostile_float_without_hook() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="hyperparameter 'step_size'") as exc:
        _validated_hyperparameter("step_size", _HostileFloat(0.1))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)


def test_label_config_rejects_bool_and_hostile_int() -> None:
    with pytest.raises(ValueError, match="n_tasks must be a positive integer"):
        LabelEMNISTConfig(n_tasks=True)  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="n_tasks must be a positive integer") as exc:
        LabelEMNISTConfig(n_tasks=_HostileInt(4))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_label_config_string_subclass_for_name_not_applicable_but_int_subclass_blocked() -> None:
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be a positive integer"):
        LabelEMNISTConfig(task_length=_HostileInt(8))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_label_config_numpy_scalars_pass() -> None:
    cfg = LabelEMNISTConfig(n_tasks=np.int32(3), task_length=np.int64(8))
    assert cfg.n_tasks == 3
    assert cfg.task_length == 8
    assert resolve_hyperparameters("upgd_w", {})["step_size"] == 0.01


def test_resolve_hyperparameters_rejects_string_subclass_learner() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        resolve_hyperparameters(_StringSubclass("upgd_w"))  # type: ignore[arg-type]


def test_resolve_hyperparameters_hostile_str_without_repr_leak() -> None:
    evil = _EvilStr("upgd_w")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        resolve_hyperparameters(evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_resolve_hyperparameters_unknown_learner_sanitized_without_repr() -> None:
    with pytest.raises(ValueError, match="unknown learner") as exc:
        resolve_hyperparameters("bogus_learner")
    assert "!r" not in str(exc.value)
    assert "'bogus_learner'" in str(exc.value)
    evil = _EvilStr("bogus_learner")
    with pytest.raises(ValueError, match="must be an exact string") as exc2:
        resolve_hyperparameters(evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc2.value)


def test_require_exact_str_for_key_hostile_without_repr() -> None:
    evil = _EvilStr("source")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_exact_str("key", evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)


def test_require_exact_str_for_key_string_subclass_rejected() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("key", _StringSubclass("source"))  # type: ignore[arg-type]


def test_strict_json_duplicate_key_sanitized(tmp_path: pathlib.Path) -> None:
    # Duplicate at top level via raw JSON text
    p = tmp_path / "dup.json"
    p.write_text('{"source": "a", "source": "b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key") as exc:
        _strict_json_object(p)
    assert "!r" not in str(exc.value)
    assert "'source'" in str(exc.value)


def test_valid_resolve_and_strict_json_still_pass(tmp_path: pathlib.Path) -> None:
    assert resolve_hyperparameters("upgd_w")["step_size"] == 0.01
    assert resolve_hyperparameters("adamw", {"beta1": 0.1})["beta1"] == 0.1
    p = tmp_path / "valid.json"
    p.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")
    obj = _strict_json_object(p)
    assert obj == {"a": 1, "b": 2}
    cfg = LabelEMNISTConfig(
        n_tasks=3, task_length=8, input_dim=6, hidden1=8, hidden2=4, n_classes=5
    )
    assert cfg.n_tasks == 3
