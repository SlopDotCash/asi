"""Hostile validation for Step2 exact keys."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.steps.step2 import _require_exact_keys, _require_exact_str


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
        _require_exact_str("config_name", evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_require_exact_keys_rejects_evil_config_without_hooks() -> None:
    evil = _EvilStr("my_config")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        _require_exact_keys(evil, {}, frozenset())  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)
    assert "EvilStr" not in str(exc.value)


def test_require_exact_keys_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_keys(_StringSubclass("cfg"), {}, frozenset())  # type: ignore[arg-type]


def test_require_exact_keys_rejects_non_dict() -> None:
    with pytest.raises(ValueError, match="exact dict"):
        _require_exact_keys("cfg", [], frozenset())  # type: ignore[arg-type]


def test_require_exact_keys_rejects_non_exact_keys() -> None:
    with pytest.raises(ValueError, match="exact strings"):
        _require_exact_keys("cfg", {_StringSubclass("k"): 1}, frozenset({"k"}))  # type: ignore[arg-type]


def test_require_exact_keys_mismatch_sanitized() -> None:
    with pytest.raises(ValueError, match="exactly the expected keys") as exc:
        _require_exact_keys("cfg", {"a": 1}, frozenset({"a", "b"}))
    assert "!r" not in str(exc.value)
    assert "cfg" in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/steps/step2.py").read_text()
    assert "!r" not in text


def test_valid_keys_pass() -> None:
    _require_exact_keys("cfg", {"a": 1, "b": 2}, frozenset({"a", "b"}))
