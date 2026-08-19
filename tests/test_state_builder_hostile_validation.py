"""Hostile validation for state builder type gate."""

from __future__ import annotations

from typing import Any, cast

import pytest

from alberta_framework.core.state_builder import (
    FixedTraceStateBuilderConfig,
    IdentityStateBuilderConfig,
    OnlineGatedStateBuilderConfig,
    _require_exact_str,
    state_builder_config_from_config,
    state_builder_from_config,
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


def test_require_exact_str_rejects_evil() -> None:
    evil = _EvilStr("v")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        _require_exact_str("payload.type", evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_state_builder_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("IdentityStateBuilder")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        state_builder_config_from_config({"type": evil})
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)
    assert "EvilStr" not in str(exc.value)


def test_state_builder_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        state_builder_config_from_config({"type": _StringSubclass("IdentityStateBuilder")})


def test_state_builder_mismatch_sanitized() -> None:
    bad = "BadBuilder"
    with pytest.raises(ValueError, match="unknown state builder type") as exc:
        state_builder_config_from_config({"type": bad})
    assert "!r" not in str(exc.value)
    assert bad in str(exc.value)
    # However we sanitize to not use !r, but value is still leaked via plain; ensure !r not used
    assert "!r" not in str(exc.value)


def test_state_builder_valid_passes() -> None:
    cfg = state_builder_config_from_config({"type": "IdentityStateBuilder", "observation_dim": 2})
    assert cfg is not None


class _HostileMapping(dict[object, object]):
    calls = 0

    def __iter__(self) -> Any:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("HostileMapping.__iter__ must not be called")


def test_state_builder_from_config_rejects_non_dict_payload() -> None:
    for factory in (
        state_builder_config_from_config,
        state_builder_from_config,
        IdentityStateBuilderConfig.from_config,
        FixedTraceStateBuilderConfig.from_config,
        OnlineGatedStateBuilderConfig.from_config,
    ):
        with pytest.raises(ValueError, match="exact dict"):
            factory(cast(Any, _HostileMapping()))
        with pytest.raises(ValueError, match="exact dict"):
            factory(cast(Any, [("type", "IdentityStateBuilder")]))
        with pytest.raises(ValueError, match="exact dict"):
            factory(cast(Any, "IdentityStateBuilder"))


def test_state_builder_from_config_rejects_non_string_keys() -> None:
    payload_bad_keys = {1: "IdentityStateBuilder", "observation_dim": 2}
    with pytest.raises(ValueError, match="exact strings"):
        state_builder_config_from_config(cast(Any, payload_bad_keys))
    with pytest.raises(ValueError, match="exact strings"):
        IdentityStateBuilderConfig.from_config(cast(Any, payload_bad_keys))


def test_identity_state_builder_from_config_schema_validation() -> None:
    # Missing observation_dim
    with pytest.raises(ValueError, match="fields do not match the schema"):
        IdentityStateBuilderConfig.from_config({"type": "IdentityStateBuilder"})
    # Extra field
    with pytest.raises(ValueError, match="fields do not match the schema"):
        IdentityStateBuilderConfig.from_config(
            {"type": "IdentityStateBuilder", "observation_dim": 2, "extra": 1}
        )
    # Type mismatch
    with pytest.raises(ValueError, match="payload type must be 'IdentityStateBuilder'"):
        IdentityStateBuilderConfig.from_config(
            {"type": "FixedTraceStateBuilder", "observation_dim": 2}
        )


def test_fixed_trace_state_builder_from_config_schema_validation() -> None:
    base = FixedTraceStateBuilderConfig(observation_dim=2).to_config()
    # Missing field
    missing = dict(base)
    del missing["n_actions"]
    with pytest.raises(ValueError, match="fields do not match the schema"):
        FixedTraceStateBuilderConfig.from_config(missing)
    # Extra field
    extra = {**base, "extra": 1}
    with pytest.raises(ValueError, match="fields do not match the schema"):
        FixedTraceStateBuilderConfig.from_config(extra)
    # Type mismatch
    wrong_type = {**base, "type": "Wrong"}
    with pytest.raises(ValueError, match="payload type must be 'FixedTraceStateBuilder'"):
        FixedTraceStateBuilderConfig.from_config(wrong_type)
    # Non-sequence decay rates
    bad_decays = {**base, "observation_decay_rates": 123}
    with pytest.raises(ValueError, match="decay rates must be lists or tuples"):
        FixedTraceStateBuilderConfig.from_config(bad_decays)


def test_online_gated_state_builder_from_config_schema_validation() -> None:
    base = OnlineGatedStateBuilderConfig(observation_dim=2).to_config()
    # Missing field
    missing = dict(base)
    del missing["hidden_dim"]
    with pytest.raises(ValueError, match="fields do not match the schema"):
        OnlineGatedStateBuilderConfig.from_config(missing)
    # Extra field
    extra = {**base, "extra": 1}
    with pytest.raises(ValueError, match="fields do not match the schema"):
        OnlineGatedStateBuilderConfig.from_config(extra)
    # Type mismatch
    wrong_type = {**base, "type": "Wrong"}
    with pytest.raises(ValueError, match="payload type must be 'OnlineGatedStateBuilder'"):
        OnlineGatedStateBuilderConfig.from_config(wrong_type)


def test_state_builder_roundtrips() -> None:
    c1 = IdentityStateBuilderConfig(observation_dim=4)
    assert IdentityStateBuilderConfig.from_config(c1.to_config()) == c1
    c2 = FixedTraceStateBuilderConfig(
        observation_dim=4,
        n_actions=2,
        observation_decay_rates=(0.9,),
        action_decay_rates=(0.8,),
        outcome_decay_rates=(0.5,),
        include_raw_observation=True,
    )
    assert FixedTraceStateBuilderConfig.from_config(c2.to_config()) == c2
    c3 = OnlineGatedStateBuilderConfig(
        observation_dim=4,
        n_actions=2,
        hidden_dim=8,
        step_size=0.01,
        include_raw_observation=True,
    )
    assert OnlineGatedStateBuilderConfig.from_config(c3.to_config()) == c3
