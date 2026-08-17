"""Hostile validation for forager matched final analysis trust boundary."""
# mypy: disable-error-code="arg-type"

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.forager_matched_final_analysis import (
    _require_exact_str,
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


def test_candidate_binding_rejects_evil() -> None:
    from alberta_framework.benchmarks.forager_matched_final_analysis import (
        _expected_entrypoint_binding,
    )

    evil = _EvilStr("bad_candidate")
    _EvilStr.calls = 0
    with pytest.raises(Exception, match="exact string") as exc:
        _expected_entrypoint_binding(evil)
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)


def test_candidate_binding_sanitized() -> None:
    from alberta_framework.benchmarks.forager_matched_final_analysis import (
        _expected_entrypoint_binding,
    )

    with pytest.raises(Exception, match="has no frozen entrypoint binding") as exc:
        _expected_entrypoint_binding("unknown_candidate_xyz")
    assert "!r" not in str(exc.value)
    assert "unknown_candidate_xyz" in str(exc.value)
    assert "'" in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path(
        "alberta_framework/benchmarks/forager_matched_final_analysis.py"
    ).read_text()
    assert "!r" not in text
