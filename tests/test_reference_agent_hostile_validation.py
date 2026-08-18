"""Hostile validation for reference agent trust boundary."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.reference_agent import _require_exact_str


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


def test_require_exact_str_rejects_evil() -> None:
    evil = _EvilStr("v")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        _require_exact_str("key", evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc)


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_str("key", _StringSubclass("v"))


def test_dtype_sanitized() -> None:
    from alberta_framework.reference_agent import ArrayValue

    evil = _EvilStr("bad_dtype")
    _EvilStr.calls = 0
    try:
        ArrayValue(
            semantic_id="a.b",
            dtype=evil,
            shape=(),
            payload=b"\x00\x00\x80\x3f",
        )
    except ValueError as exc:
        assert _EvilStr.calls == 0
        assert "!r" not in str(exc)
    else:
        raise AssertionError("should have raised")


def test_schema_sanitized() -> None:
    from alberta_framework.reference_agent import AgentCapabilities, AgentManifest, SpaceSpec

    with pytest.raises(ValueError) as exc:
        AgentManifest(
            schema="bad_schema",
            implementation_id="a.b",
            state_schema="c.d",
            config_sha256="0" * 64,
            manifest_id="1" * 64,
            observation_spec=SpaceSpec(
                kind="box", shape=(), dtype="float32", semantic_id="a.b"
            ),
            action_spec=SpaceSpec(
                kind="box", shape=(), dtype="float32", semantic_id="a.b"
            ),
            capabilities=AgentCapabilities(),
            _config_json="{}",
        )
    assert "!r" not in str(exc)
    assert "bad_schema" in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/reference_agent.py").read_text()
    assert "!r" not in text


def test_valid_array_value() -> None:
    from alberta_framework.reference_agent import ArrayValue

    v = ArrayValue(
        semantic_id="a.b", dtype="float32", shape=(), payload=b"\x00\x00\x80\x3f"
    )
    assert v.dtype == "float32"
