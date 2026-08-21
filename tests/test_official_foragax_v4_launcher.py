"""Unit coverage for alberta_framework.benchmarks._official_foragax_v4_launcher.

Tests the fail-closed launcher validation: canonical absolute-path checks,
root-escape rejection, argument validation (flags, index expression,
allowlisted paths), and the immutable regular-file contract.
"""

import os
from pathlib import Path

import pytest

from alberta_framework.benchmarks._official_foragax_v4_launcher import (
    LauncherError,
    _absolute_path,
    _under,
    _validate_arguments,
)


def test_absolute_path_accepts_canonical() -> None:
    assert _absolute_path("/a/b/c", label="x") == Path("/a/b/c")


def test_absolute_path_rejects_bad() -> None:
    for bad in ["relative", "/a/../b", "/a//b", "/a\x00b"]:
        with pytest.raises(LauncherError, match="canonical absolute"):
            _absolute_path(bad, label="x")


def test_under_accepts_within() -> None:
    root = Path("/root")
    assert _under(Path("/root/sub/file"), root, label="x") == Path("/root/sub/file")


def test_under_rejects_escape() -> None:
    root = Path("/root")
    with pytest.raises(LauncherError, match="escapes"):
        _under(Path("/other/file"), root, label="x")


def test_validate_arguments_index_formats(tmp_path: Path) -> None:
    # Build minimal args with valid paths; only the index expression varies.
    def make_args(index: str):
        return type(
            "Args",
            (),
            {
                "python_flag": ["-I", "-B"],
                "trusted_python_path_mode": "immutable",
                "export_format": "v4",
                "max_steps": 100,
                "index": index,
            },
        )()

    # These need real file paths; the index check happens before file checks.
    with pytest.raises(LauncherError):
        _validate_arguments(make_args("bad-index"))
    with pytest.raises(LauncherError):
        _validate_arguments(make_args("2:1"))  # start >= end
    with pytest.raises(LauncherError):
        _validate_arguments(make_args("-1"))


def test_validate_arguments_rejects_wrong_flags(tmp_path: Path) -> None:
    args = type(
        "Args",
        (),
        {
            "python_flag": ["-B", "-I"],  # wrong order
            "trusted_python_path_mode": "immutable",
            "export_format": "v4",
            "max_steps": 1,
            "index": "0",
        },
    )()
    with pytest.raises(LauncherError, match="exactly -I then -B"):
        _validate_arguments(args)


def test_regular_immutable_contract(tmp_path: Path) -> None:
    from alberta_framework.benchmarks import _official_foragax_v4_launcher as mod

    # A file with 0o644 (not the required 0o444) must be rejected regardless
    # of uid — this covers the mode check without depending on the uid.
    f = tmp_path / "file"
    f.write_text("x", encoding="utf-8")
    os.chmod(f, 0o644)
    with pytest.raises(LauncherError, match="immutable regular"):
        mod._regular_immutable(f, label="test")
