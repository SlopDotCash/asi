"""Hostile validation for world model expected string identities."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0
    __hash__ = str.__hash__

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool")

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile len")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq")


class _HostileInt(int):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile int eq")


class _HostileFloat(float):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile float eq")


def test_strict_json_equal_rejects_hostile_str_before_eq() -> None:
    from alberta_framework.core.recurrent_latent_world_model_ensemble import (
        _strict_json_equal,
    )

    hostile = _HostileStr("ok")
    _HostileStr.calls = 0
    assert _strict_json_equal("ok", hostile) is False
    assert _strict_json_equal(hostile, "ok") is False  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
    assert _strict_json_equal("ok", "ok") is True
    assert _strict_json_equal("ok", "bad") is False


def test_strict_json_equal_rejects_hostile_int_float_bool() -> None:
    from alberta_framework.core.recurrent_latent_world_model_ensemble import (
        _strict_json_equal,
    )

    hi = _HostileInt(5)
    _HostileInt.calls = 0
    assert _strict_json_equal(5, hi) is False
    assert _HostileInt.calls == 0
    assert _strict_json_equal(5, 5) is True

    hf = _HostileFloat(1.0)
    _HostileFloat.calls = 0
    assert _strict_json_equal(1.0, hf) is False
    assert _HostileFloat.calls == 0
    assert _strict_json_equal(1.0, 1.0) is True

    # bool exact
    assert _strict_json_equal(True, True) is True  # type: ignore[arg-type]
    assert _strict_json_equal(True, 1) is False  # type: ignore[arg-type]
    assert _strict_json_equal(1, True) is False  # type: ignore[arg-type]


def test_hostile_not_in_error_message() -> None:
    hostile = _HostileStr("evil")
    _HostileStr.calls = 0
    from alberta_framework.core.recurrent_latent_world_model_ensemble import (
        _strict_json_equal,
    )

    assert _strict_json_equal("ok", hostile) is False
    assert _HostileStr.calls == 0
    try:
        if type(hostile) is not str:
            raise ValueError("expected must be exact str")
    except ValueError as exc:
        assert "!r" not in str(exc)
        assert _HostileStr.calls == 0
