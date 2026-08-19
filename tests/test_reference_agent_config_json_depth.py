"""Reject deep manifest config JSON before json.loads RecursionError."""

from __future__ import annotations

from typing import Any

import pytest

from alberta_framework.reference_agent import (
    _MAX_JSON_NESTING_DEPTH,
    REFERENCE_AGENT_MANIFEST_SCHEMA,
    AgentCapabilities,
    AgentManifest,
    SpaceSpec,
    _load_manifest_config_json,
)


def _nested_array_json(depth: int) -> str:
    return "[" * depth + "0" + "]" * depth


def _manifest_config_json(depth: int) -> str:
    return '{"k":' + _nested_array_json(depth) + "}"


def _attempt_manifest(config_json: str) -> AgentManifest:
    return AgentManifest(
        schema=REFERENCE_AGENT_MANIFEST_SCHEMA,
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
        _config_json=config_json,
    )


def test_deep_config_json_never_reaches_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """10_000-deep JSON must ValueError before json.loads RecursionError."""
    import json as json_module

    calls: list[str] = []
    real_loads = json_module.loads

    def spy(raw: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(raw if type(raw) is str else type(raw).__name__)
        return real_loads(raw, *args, **kwargs)

    monkeypatch.setattr("alberta_framework.reference_agent.json.loads", spy)
    deep = _manifest_config_json(10_000)
    assert len(deep.encode("utf-8")) < 1 << 20
    with pytest.raises(ValueError, match="nesting depth"):
        _load_manifest_config_json(deep)
    with pytest.raises(ValueError, match="nesting depth"):
        _attempt_manifest(deep)
    assert calls == []


def _nested_python_list(depth: int) -> Any:
    value: Any = 0
    for _ in range(depth):
        value = [value]
    return value


def test_last_fit_nesting_still_parses() -> None:
    # Object + 63 arrays = depth 64, the last-fit for the protocol cap.
    depth = _MAX_JSON_NESTING_DEPTH - 1
    loaded = _load_manifest_config_json(_manifest_config_json(depth))
    assert loaded == {"k": _nested_python_list(depth)}


def test_first_overflow_nesting_rejects_before_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    import json as json_module

    calls: list[int] = []
    real_loads = json_module.loads

    def spy(raw: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real_loads(raw, *args, **kwargs)

    monkeypatch.setattr("alberta_framework.reference_agent.json.loads", spy)
    with pytest.raises(ValueError, match="nesting depth"):
        _load_manifest_config_json(_manifest_config_json(_MAX_JSON_NESTING_DEPTH))
    assert calls == []


def test_recursionerror_from_loads_is_valueerror(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(raw: Any, *args: Any, **kwargs: Any) -> Any:
        del raw, args, kwargs
        raise RecursionError("simulated parser overflow")

    monkeypatch.setattr("alberta_framework.reference_agent.json.loads", boom)
    with pytest.raises(ValueError, match="nesting depth"):
        _load_manifest_config_json(_manifest_config_json(1))
