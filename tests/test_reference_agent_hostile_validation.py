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


class _BytesSubclass(bytes):
    pass


class _FakeDtypeList(list):
    dtype = "float32"


def test_comma_dtype_and_bytes_subclass_rejected() -> None:
    from alberta_framework.reference_agent import ArrayValue, SpaceSpec

    with pytest.raises(ValueError, match="unsupported dtype"):
        ArrayValue(semantic_id="a.b", dtype=",", shape=(), payload=b"")
    with pytest.raises(ValueError, match="unsupported dtype"):
        SpaceSpec(kind="box", shape=(1,), dtype=",", semantic_id="a.b")
    with pytest.raises(ValueError, match="immutable bytes"):
        ArrayValue(
            semantic_id="a.b",
            dtype="float32",
            shape=(),
            payload=_BytesSubclass(b"\x00\x00\x80\x3f"),
        )


def test_integer_box_bounds_reject_inexact_float64_above_2_pow_53() -> None:
    from alberta_framework.reference_agent import SpaceSpec

    with pytest.raises(ValueError, match="representable by int64"):
        SpaceSpec(
            kind="box",
            shape=(1,),
            dtype="int64",
            semantic_id="a.b",
            low=(0,),
            high=(float((1 << 53) + 2),),
        )


def test_space_encode_rejects_dtype_decorated_sequence_with_bool() -> None:
    from alberta_framework.reference_agent import SpaceSpec

    spec = SpaceSpec(kind="box", shape=(2,), dtype="float32", semantic_id="a.b")
    with pytest.raises(ValueError, match="bounded sequence|not bool"):
        spec.encode(_FakeDtypeList([True, 2.0]))


def test_halt_after_execution_rejects_blank_reason_on_invalid_receipt() -> None:
    from alberta_framework.reference_agent import (
        AgentCapabilities,
        AgentManifest,
        ArrayValue,
        DispatchReceipt,
        ReferenceTransactionReducer,
        SpaceSpec,
    )

    obs_spec = SpaceSpec(kind="box", shape=(1,), dtype="float32", semantic_id="obs")
    act_spec = SpaceSpec(kind="discrete", shape=(), dtype="int32", semantic_id="act", cardinality=2)
    manifest = AgentManifest.from_config(
        schema="asi.reference_agent_manifest.preview1",
        implementation_id="test.agent",
        state_schema="test.state",
        config={"lr": 0.1},
        observation_spec=obs_spec,
        action_spec=act_spec,
        capabilities=AgentCapabilities(dispatch_rebinding=False),
    )
    reducer = ReferenceTransactionReducer(manifest)
    state = reducer.init()
    decision = manifest.make_decision(
        lifecycle_id="life0",
        decision_index=0,
        observation_id="obs0",
        observation=(1.0,),
        proposed_action=0,
        armed=True,
    )
    state = reducer.arm(state, decision)
    state, auth = reducer.authorize(
        state,
        decision,
        authorized_action=None,
        authority_id="auth0",
        policy_version="v1",
        authorization_id="life0:0:authorization",
    )
    state, ack = reducer.settle_dispatch(
        state,
        auth,
        rebinding_applied=False,
        settlement_id="life0:0:settlement",
    )
    assert ack is not None
    state, cmd = reducer.issue_dispatch(
        state,
        ack,
        command_id="life0:0:command",
        executor_id="exec0",
        executor_epoch="ep0",
    )
    bad_action = ArrayValue(
        semantic_id="wrong",
        dtype="int32",
        shape=(),
        payload=(0).to_bytes(4, "little"),
    )
    bad_receipt = DispatchReceipt(
        command=cmd,
        applied_action=bad_action,
        receipt_id="life0:0:receipt",
    )
    with pytest.raises(ValueError, match="halt reason must be nonempty"):
        reducer.halt_after_execution(state, bad_receipt, reason="   ")
