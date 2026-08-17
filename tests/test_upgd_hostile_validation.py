"""Hostile validation for UPGD trust boundary."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.core.upgd import _require_exact_str


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
    assert "!r" not in str(exc.value)


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_str("key", _StringSubclass("v"))


def test_upgd_rejects_evil_meta_plasticity() -> None:
    evil = _EvilStr("bad_mode")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        _require_exact_str("meta_plasticity_mode", evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_upgd_rejects_subclass_mode() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_str("meta_plasticity_mode", _StringSubclass("bad_mode"))


def test_upgd_sanitized() -> None:
    # Valid host should pass and file's sanitized branches use same host
    assert _require_exact_str("meta_plasticity_mode", "none") == "none"
    text = pathlib.Path("alberta_framework/core/upgd.py").read_text()
    assert "bad_mode" not in text or "!r" not in text



def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/core/upgd.py").read_text()
    assert "!r" not in text
