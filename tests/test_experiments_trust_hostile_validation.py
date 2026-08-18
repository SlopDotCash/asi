"""Trust-boundary validation for experiments sanitized errors."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.core.learners import LinearLearner
from alberta_framework.utils.experiments import (
    ExperimentConfig,
    SingleRunResult,
    _require_exact_str,
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


def _legal_config() -> ExperimentConfig:
    def _learner() -> LinearLearner:
        return LinearLearner()
    return ExperimentConfig(
        name="fixture",
        learner_factory=_learner,
        stream_factory=_learner,
        num_steps=2,
    )


def _legal_run() -> SingleRunResult:
    return SingleRunResult(
        config_name="fixture",
        seed=0,
        metrics_history=[{"squared_error": 0.1}],
        final_state=LinearLearner().init(2),
    )


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("field", _StringSubclass("x"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("name", _StringSubclass("x"))  # type: ignore[arg-type]


def test_require_exact_str_hostile_without_repr_leak() -> None:
    evil = _EvilStr("evil")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_exact_str("field", evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)
    evil2 = _EvilStr("evil2")
    with pytest.raises(ValueError, match="must be an exact string") as exc2:
        _require_exact_str("name", evil2)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc2.value)


def test_replace_sanitized() -> None:
    cfg = _legal_config()
    with pytest.raises(ValueError, match="Got unexpected field names") as exc:
        cfg._replace(**{"evil_field": 1})  # type: ignore[arg-type]
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_field'" in msg
    run = _legal_run()
    with pytest.raises(ValueError, match="Got unexpected field names") as exc2:
        run._replace(**{"evil_field": 1})  # type: ignore[arg-type]
    msg2 = str(exc2.value)
    assert "!r" not in msg2
    assert "'evil_field'" in msg2


def test_replace_hostile_blocked_before_repr() -> None:
    cfg = _legal_config()
    evil = _EvilStrNoHash("evil")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        cfg._replace(**{evil: 1})  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)
    # Subclass also rejected before formatting
    with pytest.raises(ValueError, match="must be an exact string"):
        cfg._replace(**{_StringSubclass("evil"): 1})  # type: ignore[arg-type]
    run = _legal_run()
    evil2 = _EvilStrNoHash("evil2")
    with pytest.raises(ValueError, match="must be an exact string") as exc2:
        run._replace(**{evil2: 1})  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc2.value)


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/utils/experiments.py")
    text = p.read_text(encoding="utf-8")
    assert "Got unexpected field names: {sorted(unexpected)!r}" not in text
    assert "{sorted(unexpected)!r}" not in text
    assert "Got unexpected field names: [{sanitized}]" in text
    assert "f\"'{_require_exact_str('field', k)}'\"" in text


def test_valid_still_passes() -> None:
    assert _require_exact_str("field", "ok") == "ok"
    cfg = _legal_config()
    assert cfg._replace(name="ok2").name == "ok2"
    run = _legal_run()
    assert run._replace(seed=1).seed == 1
