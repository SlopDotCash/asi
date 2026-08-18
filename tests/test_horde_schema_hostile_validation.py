"""Hostile validation for Horde schema gate."""

from __future__ import annotations

import pytest

from alberta_framework import DemonType, GVFSpec, HordeLearner, create_horde_spec
from alberta_framework.core.horde import _require_exact_str
from alberta_framework.core.multi_head_learner import MULTI_HEAD_MLP_STATE_SCHEMA


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


def _valid_config() -> dict:
    spec = create_horde_spec(
        [
            GVFSpec(
                name="d0",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
            )
        ]
    )
    learner = HordeLearner(horde_spec=spec, hidden_sizes=(4,), sparsity=0.0)
    return learner.to_config()


def test_require_exact_str_rejects_evil() -> None:
    evil = _EvilStr("value")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        _require_exact_str("state_schema", evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_str("state_schema", _StringSubclass("v"))  # type: ignore[arg-type]


def test_horde_rejects_evil_schema_without_hooks() -> None:
    cfg = _valid_config()
    evil = _EvilStr(MULTI_HEAD_MLP_STATE_SCHEMA)
    _EvilStr.calls = 0
    cfg["state_schema"] = evil  # type: ignore[assignment]
    with pytest.raises(ValueError, match="exact string") as exc:
        HordeLearner.from_config(cfg)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)
    assert "EvilStr" not in str(exc.value)


def test_horde_rejects_string_subclass_schema() -> None:
    cfg = _valid_config()
    cfg["state_schema"] = _StringSubclass(MULTI_HEAD_MLP_STATE_SCHEMA)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="exact string"):
        HordeLearner.from_config(cfg)  # type: ignore[arg-type]


def test_horde_mismatch_sanitized_without_leak() -> None:
    cfg = _valid_config()
    bad = "bad_schema"
    cfg["state_schema"] = bad
    with pytest.raises(ValueError, match="unsupported Horde state schema") as exc:
        HordeLearner.from_config(cfg)
    assert "!r" not in str(exc.value)
    assert bad not in str(exc.value)
    assert MULTI_HEAD_MLP_STATE_SCHEMA in str(exc.value)


def test_horde_valid_schema_passes() -> None:
    cfg = _valid_config()
    cfg["state_schema"] = MULTI_HEAD_MLP_STATE_SCHEMA
    learner = HordeLearner.from_config(cfg)
    assert learner is not None


def test_require_exact_str_rejects_hostile_name() -> None:
    evil = _EvilStr("state_schema")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_str(evil, "value")  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
