"""Hostile string gate for forager container sha before len."""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks._forager_matched_container import (
    ContainerError,
    _safe_extract,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile len must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")

    def __contains__(self, item: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile contains must not run")

    def __iter__(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile iter must not run")


def test_safe_extract_rejects_hostile_before_len(tmp_path: pathlib.Path) -> None:
    hostile = _HostileStr("a" * 64)
    _HostileStr.calls = 0
    dummy = tmp_path / "dummy.tar"
    dummy.write_bytes(b"dummy")
    with pytest.raises(ContainerError, match="SHA-256 is invalid"):
        _safe_extract(dummy, hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_safe_extract_rejects_non_string_before_len(tmp_path: pathlib.Path) -> None:
    dummy = tmp_path / "dummy.tar"
    dummy.write_bytes(b"dummy")
    with pytest.raises(ContainerError, match="SHA-256 is invalid"):
        _safe_extract(dummy, 123)  # type: ignore[arg-type]


def test_safe_extract_rejects_short_benign(tmp_path: pathlib.Path) -> None:
    dummy = tmp_path / "dummy.tar"
    dummy.write_bytes(b"dummy")
    with pytest.raises(ContainerError, match="SHA-256 is invalid"):
        _safe_extract(dummy, "short")


def test_benign_valid_sha_passes_gate(tmp_path: pathlib.Path) -> None:
    dummy = tmp_path / "dummy.tar"
    dummy.write_bytes(b"dummy")
    # valid sha passes the sha check, then fails on missing raw archive contract (not sha error)
    try:
        _safe_extract(dummy, "a" * 64)
    except ContainerError as exc:
        assert "SHA-256 is invalid" not in str(exc)
    except Exception:
        pass

