"""Hostile validation for optimizers trust boundary."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.core.optimizers import (
    _require_exact_str,
    bounder_from_config,
    optimizer_from_config,
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
        _require_exact_str("key", evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_exact_str("key", _StringSubclass("v"))


def test_optimizer_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("BadOpt")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        optimizer_from_config({"type": evil})
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_optimizer_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="exact string"):
        optimizer_from_config({"type": _StringSubclass("BadOpt")})


def test_optimizer_sanitized() -> None:
    with pytest.raises(ValueError, match="Unknown optimizer type") as exc:
        optimizer_from_config({"type": "BadOpt"})
    assert "!r" not in str(exc.value)
    assert "BadOpt" in str(exc.value)


def test_bounder_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("BadBounder")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        bounder_from_config({"type": evil})
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_bounder_sanitized() -> None:
    with pytest.raises(ValueError, match="Unknown bounder type") as exc:
        bounder_from_config({"type": "BadBounder"})
    assert "!r" not in str(exc.value)
    assert "BadBounder" in str(exc.value)


def test_h_decay_rejects_evil_without_hooks() -> None:
    from alberta_framework.core.optimizers import IDBD

    evil = _EvilStr("bad_mode")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string") as exc:
        IDBD(h_decay_mode=evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/core/optimizers.py").read_text()
    assert "!r" not in text
