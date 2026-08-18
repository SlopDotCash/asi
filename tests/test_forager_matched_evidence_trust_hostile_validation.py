"""Trust-boundary validation for forager_matched_evidence sanitized errors."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.forager_matched_evidence import (
    ForagerMatchedEvidenceError,
    _parse_json_float,
    _reject_duplicate_keys,
    _reject_nonfinite,
    _require_exact_str,
)


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
    with pytest.raises(ForagerMatchedEvidenceError, match="must be an exact string"):
        _require_exact_str("key", _StringSubclass("x"))  # type: ignore[arg-type]


def test_require_exact_str_hostile_without_repr_leak() -> None:
    evil = _EvilStr("evil")
    with pytest.raises(ForagerMatchedEvidenceError, match="must be an exact string") as exc:
        _require_exact_str("key", evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_duplicate_key_sanitized() -> None:
    with pytest.raises(ForagerMatchedEvidenceError, match="duplicate JSON object key") as exc:
        _reject_duplicate_keys([("evil_key", 1), ("evil_key", 2)])
    msg = str(exc.value)
    assert "!r" not in msg
    assert "'evil_key'" in msg


def test_duplicate_key_hostile_blocked_before_hash() -> None:
    evil = _EvilStr("evil")
    with pytest.raises(ForagerMatchedEvidenceError, match="must be an exact string") as exc:
        _reject_duplicate_keys([(evil, 1)])  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)


def test_nonfinite_sanitized() -> None:
    with pytest.raises(ForagerMatchedEvidenceError, match="non-finite JSON constant") as exc:
        _reject_nonfinite("NaN")
    assert "!r" not in str(exc.value)
    assert "'NaN'" in str(exc.value)
    with pytest.raises(ForagerMatchedEvidenceError, match="invalid JSON number") as exc2:
        _parse_json_float("bad")
    assert "!r" not in str(exc2.value)


def test_nonfinite_hostile() -> None:
    evil = _EvilStr("NaN")
    with pytest.raises(ForagerMatchedEvidenceError, match="must be an exact string"):
        _reject_nonfinite(evil)  # type: ignore[arg-type]
    with pytest.raises(ForagerMatchedEvidenceError, match="must be an exact string"):
        _parse_json_float(evil)  # type: ignore[arg-type]


def test_source_contains_no_repr_leak() -> None:
    p = pathlib.Path("alberta_framework/benchmarks/forager_matched_evidence.py")
    text = p.read_text(encoding="utf-8")
    assert "duplicate JSON object key {key!r}" not in text
    assert "non-finite JSON constant {value!r}" not in text
    assert "invalid JSON number {value!r}" not in text
    assert "non-finite JSON number {value!r}" not in text
    assert "candidate {candidate.candidate_id!r}" not in text
    assert "candidate {candidate_scores.candidate_id!r}" not in text
    assert "duplicate JSON object key '{host_key}'" in text
    assert "candidate '{host_candidate_id}'" in text


def test_valid_still_passes() -> None:
    assert _require_exact_str("key", "ok") == "ok"
    assert _reject_duplicate_keys([("a", 1)]) == {"a": 1}
    assert _parse_json_float("1.5") == 1.5
