"""Hostile validation for official Foragax image helper."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from alberta_framework.benchmarks._official_foragax_image_helper import (
    ImageHelperError,
    _require_exact_str,
    _strict_json,
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


def test_require_exact_str_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("key")
    _EvilStr.calls = 0
    with pytest.raises(ImageHelperError, match="exact string") as exc:
        _require_exact_str("key", evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)
    assert "EvilStr" not in str(exc.value)


def test_require_exact_str_rejects_string_subclass() -> None:
    with pytest.raises(ImageHelperError, match="exact string"):
        _require_exact_str("key", _StringSubclass("value"))  # type: ignore[arg-type]


def test_require_exact_str_rejects_hostile_name() -> None:
    evil = _EvilStr("key")
    _EvilStr.calls = 0
    with pytest.raises(ImageHelperError, match="exact string"):
        _require_exact_str(evil, "value")  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def test_strict_json_rejects_duplicate_key_sanitized() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dup.json"
        path.write_text('{"a": 1, "a": 2}')
        with pytest.raises(ImageHelperError, match="repeats JSON key") as exc:
            _strict_json(path)
        assert "!r" not in str(exc.value)


def test_strict_json_rejects_nan_constant_sanitized() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nonfinite.json"
        path.write_text('{"a": NaN}')
        with pytest.raises(ImageHelperError, match="contains JSON constant") as exc:
            _strict_json(path)
        assert "!r" not in str(exc.value)
        assert "NaN" not in str(exc.value)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "inf.json"
        path.write_text('{"a": Infinity}')
        with pytest.raises(ImageHelperError, match="contains JSON constant") as exc:
            _strict_json(path)
        assert "!r" not in str(exc.value)
        assert "Infinity" not in str(exc.value)


def test_strict_json_rejects_nonfinite_via_parse_float() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "float.json"
        path.write_text('{"a": 1e999}')
        with pytest.raises(ImageHelperError, match="contains non-finite JSON number") as exc:
            _strict_json(path)
        assert "!r" not in str(exc.value)


def test_strict_json_valid_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "valid.json"
        path.write_text('{"a": 1, "b": 2}')
        data = _strict_json(path)
        assert data == {"a": 1, "b": 2}
