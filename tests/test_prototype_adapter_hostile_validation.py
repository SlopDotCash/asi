"""Hostile validation for prototype adapter facade."""

from __future__ import annotations

import dataclasses

import jax.random as jr
import pytest

from alberta_framework import prototype_reference_adapter as adapter
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig
from alberta_framework.core.prototype_agent import PrototypeAgentConfig
from alberta_framework.prototype_reference_adapter import (
    PROTOTYPE_REFERENCE_STATE_SCHEMA,
    PrototypeReferenceAdapter,
    PrototypeReferenceState,
)


def _valid_state() -> PrototypeReferenceState:
    config = PrototypeAgentConfig(
        oak=OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(),
                observation_dim=2,
                n_primitive_actions=2,
                base_step_size=0.05,
                epsilon_base=0.0,
            )
        )
    )
    adapter = PrototypeReferenceAdapter(config)
    return adapter.init(jr.key(0), lifecycle_id="prototype.0000000100000002")


class _EvilStr(str):
    calls = 0

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__repr__ must not be called")

    def strip(self, _chars: str | None = None) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.strip must not be called")


class _ExplodingPattern:
    calls = 0

    def __init__(self, pattern: object) -> None:
        self._pattern = pattern

    def fullmatch(self, value: str) -> object:
        if type(value) is not str:
            type(self).calls += 1
            raise AssertionError("pattern matching must follow exact-type validation")
        return self._pattern.fullmatch(value)  # type: ignore[attr-defined,no-any-return]


class _StringSubclass(str):
    pass


def test_schema_rejects_evil_str_without_hooks() -> None:
    state = _valid_state()
    evil = _EvilStr(PROTOTYPE_REFERENCE_STATE_SCHEMA)
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        dataclasses.replace(state, schema=evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)
    assert "EvilStr" not in str(exc.value)


def test_schema_rejects_string_subclass() -> None:
    state = _valid_state()
    with pytest.raises(ValueError, match="exact string"):
        dataclasses.replace(state, schema=_StringSubclass(PROTOTYPE_REFERENCE_STATE_SCHEMA))  # type: ignore[arg-type]


def test_schema_mismatch_sanitized_without_repr() -> None:
    state = _valid_state()
    bad = "bad_schema_value"
    with pytest.raises(ValueError, match="schema must be") as exc:
        dataclasses.replace(state, schema=bad)  # type: ignore[arg-type]
    assert "!r" not in str(exc.value)
    assert bad not in str(exc.value)
    assert PROTOTYPE_REFERENCE_STATE_SCHEMA in str(exc.value)


def test_schema_rejects_non_string() -> None:
    state = _valid_state()
    with pytest.raises(ValueError, match="exact string"):
        dataclasses.replace(state, schema=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact string"):
        dataclasses.replace(state, schema=None)  # type: ignore[arg-type]


def test_valid_schema_passes() -> None:
    state = _valid_state()
    assert state.schema == PROTOTYPE_REFERENCE_STATE_SCHEMA
    copy = dataclasses.replace(state, schema=PROTOTYPE_REFERENCE_STATE_SCHEMA)
    assert copy.schema == PROTOTYPE_REFERENCE_STATE_SCHEMA


def test_require_exact_str_rejects_hostile_name() -> None:
    from alberta_framework.prototype_reference_adapter import _require_exact_str

    evil = _EvilStr("schema")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_str(evil, "value")  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def test_prototype_state_repr_not_used() -> None:
    state = _valid_state()
    assert isinstance(state, PrototypeReferenceState)


def test_state_identity_fields_reject_string_subclasses_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _valid_state()
    evil = _EvilStr("0" * 64)
    sha_pattern = _ExplodingPattern(adapter._SHA256_PATTERN)
    lifecycle_pattern = _ExplodingPattern(adapter._LIFECYCLE_PATTERN)
    monkeypatch.setattr(adapter, "_SHA256_PATTERN", sha_pattern)
    monkeypatch.setattr(adapter, "_LIFECYCLE_PATTERN", lifecycle_pattern)
    _EvilStr.calls = 0
    _ExplodingPattern.calls = 0

    for field in ("manifest_id", "config_sha256", "lifecycle_id"):
        with pytest.raises(ValueError, match=field):
            dataclasses.replace(state, **{field: evil})
    with pytest.raises(ValueError, match="current_observation_id"):
        dataclasses.replace(state, current_observation_id=evil)
    with pytest.raises(ValueError, match="lifecycle_id"):
        adapter._lifecycle_words(evil)

    assert _EvilStr.calls == _ExplodingPattern.calls == 0
