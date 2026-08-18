"""Trust-boundary validation for forager_cli sanitized errors."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.forager_cli import _require_exact_str


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")

    def __hash__(self) -> int:  # type: ignore[override]
        raise AssertionError("EvilStr.__hash__ must not be called")


class _StringSubclass(str):
    pass


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("name", _StringSubclass("x"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("manifest_kind", _StringSubclass("x"))  # type: ignore[arg-type]


def test_require_exact_str_hostile_without_repr_leak() -> None:
    evil = _EvilStr("evil")
    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_exact_str("name", evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)
    evil2 = _EvilStr("manifest")
    with pytest.raises(ValueError, match="must be an exact string") as exc2:
        _require_exact_str("manifest_kind", evil2)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc2.value)


def test_conflicting_config_error_sanitized() -> None:
    # Simulate the sanitized error path for conflicting reference-config
    host = _require_exact_str("name", "evil_config")
    msg = f"conflicting --reference-config paths for '{host}'"
    assert "!r" not in msg
    assert "'evil_config'" in msg
    # Hostile subclass must be rejected before formatting
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("name", _StringSubclass("evil"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("name", _EvilStr("evil"))  # type: ignore[arg-type]


def test_manifest_kind_error_sanitized() -> None:
    host = _require_exact_str("manifest_kind", "evil_kind")
    msg = f"manifest_kind '{host}'"
    assert "!r" not in msg
    assert "'evil_kind'" in msg
    with pytest.raises(ValueError, match="must be an exact string"):
        _require_exact_str("manifest_kind", _EvilStr("evil"))  # type: ignore[arg-type]


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/forager_cli.py")
    text = p.read_text(encoding="utf-8")
    assert "!r" not in text
    # Check sanitized forms exist
    assert "conflicting --reference-config paths for '{host_name}'" in text
    assert "manifest_kind '{host_manifest_kind}'" in text
    assert "Forager family '{HISTORICAL_FORAGER_FAMILY_ID}'" in text


def test_valid_helper_still_passes() -> None:
    assert _require_exact_str("name", "ok") == "ok"
    assert _require_exact_str("manifest_kind", "official_foragax_single") == (
        "official_foragax_single"
    )
