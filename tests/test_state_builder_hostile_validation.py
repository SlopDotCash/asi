"""Hostile validation for state builder type gate."""

from __future__ import annotations

import pytest

from alberta_framework.core.state_builder import (
    _require_exact_str,
    state_builder_config_from_config,
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
        _require_exact_str("payload.type", evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_state_builder_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("IdentityStateBuilder")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        state_builder_config_from_config({"type": evil})  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)
    assert "EvilStr" not in str(exc.value)


def test_state_builder_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        state_builder_config_from_config({"type": _StringSubclass("IdentityStateBuilder")})  # type: ignore[arg-type]


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
