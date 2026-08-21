"""Unit coverage for alberta_framework.benchmarks.ipmnist_provenance.

Tests the provenance helpers: file hashing/identity (relative naming,
non-file rejection), array identity determinism, dependency version
reporting (present/absent), source identity ordering, and repo metadata
binding.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_provenance import (
    array_identity,
    dependency_versions,
    file_identity,
    input_file_identities,
    repository_specification_identities,
    sha256_file,
    source_identities,
)


def test_sha256_file(tmp_path: Path) -> None:
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello")
    assert sha256_file(f) == hashlib.sha256(b"hello").hexdigest()


def test_file_identity(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("content", encoding="utf-8")
    ident = file_identity(f, relative_to=tmp_path)
    assert ident["path"] == "a.txt"
    assert ident["bytes"] == 7
    assert ident["sha256"] == hashlib.sha256(b"content").hexdigest()


def test_file_identity_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a regular file"):
        file_identity(tmp_path)


def test_file_identity_absolute_without_relative(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    ident = file_identity(f)
    assert ident["path"] == str(f.resolve())


def test_input_file_identities_sorted_unique(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("1", encoding="utf-8")
    b.write_text("2", encoding="utf-8")
    idents = input_file_identities([b, a, a], relative_to=tmp_path)
    assert len(idents) == 2
    assert [i["path"] for i in idents] == ["a.txt", "b.txt"]


def test_array_identity_deterministic() -> None:
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    i1 = array_identity(arr)
    i2 = array_identity(arr.copy())
    assert i1 == i2
    assert i1["shape"] == [3, 4]
    assert i1["dtype"] == "<f4"


def test_dependency_versions_present_absent() -> None:
    versions = dependency_versions(["numpy"])
    assert isinstance(versions["numpy"], str)
    missing = dependency_versions(["definitely-not-installed-xyz"])
    assert missing["definitely-not-installed-xyz"] is None


def test_source_identities_sorted(tmp_path: Path) -> None:
    b = tmp_path / "b.py"
    a = tmp_path / "a.py"
    a.write_text("x", encoding="utf-8")
    b.write_text("y", encoding="utf-8")
    sources = source_identities({"z": b, "a": a}, repository_root=tmp_path)
    assert list(sources.keys()) == ["a", "z"]


def test_repository_specification(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    idents = repository_specification_identities(tmp_path)
    assert len(idents) == 1
    assert idents[0]["path"] == "pyproject.toml"
