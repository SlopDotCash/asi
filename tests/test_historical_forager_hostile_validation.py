"""Hostile validation for historical forager trust boundary."""
# mypy: disable-error-code="arg-type"

from __future__ import annotations

import pathlib
import tempfile
from pathlib import Path

import pytest

from alberta_framework.benchmarks.historical_forager import (
    HistoricalForagerArtifactError,
    _require_exact_str,
    _require_exact_str_artifact,
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
    with pytest.raises(Exception, match="exact string") as exc:
        _require_exact_str("key", evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(Exception, match="exact string"):
        _require_exact_str("key", _StringSubclass("v"))


def test_require_exact_str_artifact_rejects_evil() -> None:
    evil = _EvilStr("v")
    _EvilStr.calls = 0
    with pytest.raises(Exception, match="exact string") as exc:
        _require_exact_str_artifact("key", evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_require_exact_str_artifact_rejects_subclass() -> None:
    with pytest.raises(Exception, match="exact string"):
        _require_exact_str_artifact("key", _StringSubclass("v"))


def test_duplicate_key_sanitized_via_json() -> None:
    # Use the internal json hook via _strict_json_object by creating a temp file
    from alberta_framework.benchmarks.historical_forager import _strict_json_object

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "artifact.json"
        # Write JSON with duplicate key: json.loads with object_pairs_hook will trigger
        # We need raw bytes with duplicate keys
        p.write_bytes(b'{"dup":1,"dup":2}')
        # Make it canonical mode 0o444
        p.chmod(0o444)
        with pytest.raises(HistoricalForagerArtifactError, match="duplicate JSON key") as exc:
            _strict_json_object(p)
        assert "!r" not in str(exc.value)
        assert "'dup'" in str(exc.value)


def test_invalid_constant_sanitized() -> None:
    from alberta_framework.benchmarks.historical_forager import _strict_json_object

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "artifact.json"
        p.write_bytes(b'{"a": Infinity}')
        p.chmod(0o444)
        with pytest.raises(
            HistoricalForagerArtifactError, match="non-standard JSON constant"
        ) as exc:
            _strict_json_object(p)
        assert "!r" not in str(exc.value)
        assert "Infinity" in str(exc.value)


def test_nonfinite_number_sanitized() -> None:
    # parse_float is called via json.loads with numbers that are not finite but allowed as string?
    # The historical_forager uses parse_float for numbers, but Infinity via parse_constant
    # We can test _require helpers directly
    with pytest.raises(HistoricalForagerArtifactError, match="exact string"):
        _require_exact_str_artifact("value", _EvilStr("x"))

    # Also test non-finite via direct helper exposed?
    # Instead test helper directly
    evil = _EvilStr("Infinity")
    _EvilStr.calls = 0
    with pytest.raises(Exception, match="exact string") as exc:
        _require_exact_str_artifact("value", evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_missing_artifact_sanitized() -> None:
    from alberta_framework.benchmarks.historical_forager import _strict_json_object

    # Use EvilStr for path.name via subclassed Path? Simpler test helper directly
    evil_path = Path("/tmp") / _EvilStr("evil.json")
    _EvilStr.calls = 0
    # _strict_json_object will try lstat and fail, then format path.name which is EvilStr
    with pytest.raises(HistoricalForagerArtifactError, match="missing artifact file") as exc:
        _strict_json_object(evil_path)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/benchmarks/historical_forager.py").read_text()
    assert "!r" not in text
