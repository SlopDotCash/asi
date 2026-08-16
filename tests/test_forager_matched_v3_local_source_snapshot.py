"""Focused contract tests for the pure-stdlib matched-v3 source snapshot."""

from __future__ import annotations

import copy
import errno
import hashlib
import importlib.util
import inspect
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

_MODULE_NAME = "_alberta_forager_matched_v3_local_source_snapshot_test_v1"
_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "alberta_framework"
    / "benchmarks"
    / "forager_matched_v3_local_source_snapshot.py"
)
_DESCRIPTOR_SHA256 = "5ba69445a00dfc0bc36a4d05dafcc534b291430d491c3f71560570d7eb862899"


def _load_direct_module() -> Any:
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


snapshot = _load_direct_module()


class _InjectedAbort(BaseException):
    pass


def _canonical(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _make_repository(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    repository = tmp_path / "repository"
    core = repository / "alberta_framework" / "core"
    cache = core / "__pycache__"
    core.mkdir(parents=True)
    cache.mkdir()
    payloads = {
        "pyproject.toml": b"[project]\nname = 'fixture'\n",
        "uv.lock": b"version = 1\n",
        "FORAGER_BENCHMARK.md": b"# Fixture Forager protocol\n",
        "alberta_framework/__init__.py": b'"""fixture"""\n',
        "alberta_framework/core/a.txt": b"alpha\n",
        "alberta_framework/core/z.py": b"VALUE = 7\n",
    }
    for relative_path, payload in payloads.items():
        target = repository / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (core / "ignored.pyc").write_bytes(b"excluded-cache-file")
    (cache / "ignored.cpython-312.pyc").write_bytes(b"excluded-cache-tree")
    return repository, payloads


def _rehash_manifest(manifest: dict[str, Any]) -> tuple[bytes, str]:
    tree_payload = {
        "schema_version": snapshot.LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION,
        "directories": manifest["directories"],
        "files": manifest["files"],
    }
    manifest["tree"]["sha256"] = hashlib.sha256(_canonical(tree_payload)).hexdigest()
    body = copy.deepcopy(manifest)
    body.pop("manifest_body_sha256", None)
    manifest["manifest_body_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    raw = _canonical(manifest)
    return raw, hashlib.sha256(raw).hexdigest()


def test_descriptor_has_frozen_canonical_identity_and_fail_closed_scope() -> None:
    raw = snapshot.canonical_matched_v3_local_source_snapshot_descriptor_bytes()
    descriptor = snapshot.matched_v3_local_source_snapshot_descriptor()

    assert len(raw) == 4_003
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert hashlib.sha256(raw).hexdigest() == _DESCRIPTOR_SHA256
    assert snapshot.LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256 == _DESCRIPTOR_SHA256
    assert snapshot.matched_v3_local_source_snapshot_descriptor_sha256() == (_DESCRIPTOR_SHA256)
    assert snapshot.parse_matched_v3_local_source_snapshot_descriptor(raw) == descriptor
    assert descriptor["repository_root"] == {
        "caller_supplied": True,
        "default_path": False,
        "recorded_in_manifest": False,
        "required_form": "exact_absolute_ascii_path_without_dot_segments",
        "ancestor_symlinks_allowed": False,
    }
    assert descriptor["inventory"]["root_files"] == [
        "pyproject.toml",
        "uv.lock",
        "FORAGER_BENCHMARK.md",
    ]
    assert descriptor["inventory"]["recursive_directory"] == "alberta_framework"
    assert descriptor["measurement"]["passes"] == 2
    assert descriptor["measurement"]["directory_descriptor_anchored"] is True
    assert descriptor["measurement"]["nofollow_required"] is True
    assert descriptor["runner_relationship"] == {
        "current_runner_identity_embedded": False,
        "runner_imported": False,
        "runner_bootstrap_performed": False,
        "runner_capability_requested": False,
    }
    assert "cannot establish execution linkage alone" in " ".join(descriptor["limitations"])
    assert all(value is False for value in descriptor["claims"].values())


def test_direct_file_execution_imports_no_heavy_or_alberta_module_and_observes_no_fs() -> None:
    script = r"""
import builtins
import dataclasses
import hashlib
import hmac
import json
import os
import pathlib
import re
import stat
import sys
import types
import typing

path = pathlib.Path(sys.argv[1])
source = path.read_bytes()
calls = []

def forbidden(*args, **kwargs):
    calls.append([str(args[0]) if args else "", sorted(kwargs)])
    raise AssertionError("filesystem observation during source module execution")

builtins.open = forbidden
os.open = forbidden
os.listdir = forbidden
os.stat = forbidden
os.fstat = forbidden
os.read = forbidden
name = "_snapshot_clean_direct_probe"
module = types.ModuleType(name)
module.__file__ = str(path)
module.__package__ = ""
sys.modules[name] = module
exec(compile(source, str(path), "exec"), module.__dict__)
print(json.dumps({
    "calls": calls,
    "alberta": any(
        key == "alberta_framework" or key.startswith("alberta_framework.")
        for key in sys.modules
    ),
    "jax": any(key == "jax" or key.startswith("jax.") for key in sys.modules),
    "numpy": any(key == "numpy" or key.startswith("numpy.") for key in sys.modules),
    "foragax": any(key == "foragax" or key.startswith("foragax.") for key in sys.modules),
    "runner": any("forager_matched_v3_local_runner" in key for key in sys.modules),
    "descriptor": module.LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(_MODULE_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "calls": [],
        "alberta": False,
        "jax": False,
        "numpy": False,
        "foragax": False,
        "runner": False,
        "descriptor": _DESCRIPTOR_SHA256,
    }


def test_measurement_is_deterministic_complete_and_cache_excluding(tmp_path: Path) -> None:
    repository, payloads = _make_repository(tmp_path)

    first = snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)
    second = snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)
    manifest = first.manifest()
    paths = [record["path"] for record in manifest["files"]]

    assert first == second
    assert paths == sorted(payloads)
    assert "alberta_framework/core/ignored.pyc" not in paths
    assert not any("__pycache__" in path for path in paths)
    assert manifest["inventory"] == {
        "directory_count": 2,
        "file_count": len(payloads),
        "total_size_bytes": sum(map(len, payloads.values())),
    }
    by_path = {record["path"]: record for record in manifest["files"]}
    for relative_path, payload in payloads.items():
        assert by_path[relative_path] == {
            "path": relative_path,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    tree_payload = {
        "schema_version": snapshot.LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION,
        "directories": ["alberta_framework", "alberta_framework/core"],
        "files": manifest["files"],
    }
    assert manifest["directories"] == tree_payload["directories"]
    assert first.directory_count == 2
    assert first.tree_sha256 == hashlib.sha256(_canonical(tree_payload)).hexdigest()
    body = copy.deepcopy(manifest)
    body_sha256 = body.pop("manifest_body_sha256")
    assert body_sha256 == hashlib.sha256(_canonical(body)).hexdigest()
    assert first.full_sha256 == hashlib.sha256(first.canonical_manifest_bytes).hexdigest()
    assert str(repository) not in first.canonical_manifest_bytes.decode("ascii")
    assert all(value is False for value in manifest["claims"].values())


def test_parse_is_detached_and_verifier_measures_only_after_expected_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)
    parsed = snapshot.parse_matched_v3_local_source_snapshot_manifest(
        measured.canonical_manifest_bytes,
        expected_full_sha256=measured.full_sha256,
    )
    parsed["claims"]["qualification_granted"] = True
    assert measured.manifest()["claims"]["qualification_granted"] is False

    def forbidden_observation(_repository_root: object) -> dict[str, Any]:
        raise AssertionError("filesystem observation preceded expected-artifact validation")

    monkeypatch.setattr(snapshot, "_measure_repository", forbidden_observation)
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="nonzero"):
        snapshot.verify_matched_v3_local_source_snapshot(
            repository_root=repository,
            expected_canonical_manifest_bytes=measured.canonical_manifest_bytes,
            expected_full_sha256="0" * 64,
        )
    noncanonical = b" " + measured.canonical_manifest_bytes
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="canonical"):
        snapshot.verify_matched_v3_local_source_snapshot(
            repository_root=repository,
            expected_canonical_manifest_bytes=noncanonical,
            expected_full_sha256=hashlib.sha256(noncanonical).hexdigest(),
        )


def test_verifier_accepts_exact_expectation_then_rejects_fresh_tree_drift(
    tmp_path: Path,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    expected = snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)

    verified = snapshot.verify_matched_v3_local_source_snapshot(
        repository_root=repository,
        expected_canonical_manifest_bytes=expected.canonical_manifest_bytes,
        expected_full_sha256=expected.full_sha256,
    )
    assert verified == expected

    (repository / "alberta_framework" / "core" / "a.txt").write_bytes(b"drift\n")
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="differs"):
        snapshot.verify_matched_v3_local_source_snapshot(
            repository_root=repository,
            expected_canonical_manifest_bytes=expected.canonical_manifest_bytes,
            expected_full_sha256=expected.full_sha256,
        )


def test_symlink_hardlink_and_nonregular_inventory_nodes_fail_closed(tmp_path: Path) -> None:
    symlink_root = tmp_path / "symlink-case"
    symlink_root.mkdir()
    repository, _payloads = _make_repository(symlink_root)
    (repository / "alberta_framework" / "core" / "link.py").symlink_to("z.py")
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="symbolic"):
        snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)

    hardlink_root = tmp_path / "hardlink-case"
    hardlink_root.mkdir()
    repository, _payloads = _make_repository(hardlink_root)
    source = repository / "alberta_framework" / "core" / "a.txt"
    os.link(source, repository / "alberta_framework" / "core" / "hard.txt")
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="hardlink"):
        snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)

    fifo_root = tmp_path / "fifo-case"
    fifo_root.mkdir()
    repository, _payloads = _make_repository(fifo_root)
    fifo = repository / "alberta_framework" / "core" / "pipe"
    try:
        os.mkfifo(fifo)
    except (AttributeError, NotImplementedError, OSError) as exc:
        pytest.skip(f"FIFO creation unavailable: {exc}")
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="nonregular"):
        snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)


def test_repository_root_symlink_traversal_and_nonconcrete_paths_fail_closed(
    tmp_path: Path,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    alias = tmp_path / "repository-alias"
    alias.symlink_to(repository, target_is_directory=True)

    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="symlink"):
        snapshot.measure_matched_v3_local_source_snapshot(repository_root=alias)
    traversing = repository / ".." / "repository"
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="aliases"):
        snapshot.measure_matched_v3_local_source_snapshot(repository_root=traversing)
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="exact concrete"):
        snapshot.measure_matched_v3_local_source_snapshot(
            repository_root=cast(Any, str(repository))
        )


def test_open_ancestor_locator_replacement_is_detected(tmp_path: Path) -> None:
    ancestor = tmp_path / "ancestor"
    ancestor.mkdir()
    repository, _payloads = _make_repository(ancestor)
    anchored = snapshot._open_anchored_repository_root(repository)
    moved = tmp_path / "moved-ancestor"
    try:
        ancestor.rename(moved)
        ancestor.mkdir()
        with pytest.raises(
            snapshot.ForagerMatchedV3LocalSourceSnapshotError,
            match="root locator changed",
        ):
            anchored.verify()
    finally:
        anchored.close()


def test_final_root_verification_stat_failure_closes_every_opened_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    real_open = snapshot.os.open
    real_stat = snapshot.os.stat
    real_fstat = snapshot.os.fstat
    opened: list[int] = []
    stat_calls: dict[tuple[str, int], int] = {}

    def tracking_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if dir_fd is None:
            descriptor = cast(int, real_open(path, flags, mode))
        else:
            descriptor = cast(int, real_open(path, flags, mode, dir_fd=dir_fd))
        opened.append(descriptor)
        return descriptor

    def fail_first_final_verification_stat(
        path: Any,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if dir_fd is not None:
            key = (os.fspath(path), dir_fd)
            stat_calls[key] = stat_calls.get(key, 0) + 1
            if stat_calls[key] == 3:
                raise OSError(errno.EIO, "injected final verification stat failure")
        return cast(
            os.stat_result,
            real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks),
        )

    monkeypatch.setattr(snapshot.os, "open", tracking_open)
    monkeypatch.setattr(snapshot.os, "stat", fail_first_final_verification_stat)
    with pytest.raises(
        snapshot.ForagerMatchedV3LocalSourceSnapshotError,
        match="root locator changed during source observation",
    ):
        snapshot._open_anchored_repository_root(repository)

    assert len(opened) == len(repository.parts)
    for descriptor in opened:
        with pytest.raises(OSError) as closed:
            real_fstat(descriptor)
        assert closed.value.errno == errno.EBADF


def test_repository_anchor_first_fstat_baseexception_closes_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    real_open = snapshot.os.open
    real_fstat = snapshot.os.fstat
    anchor = -1
    failure = _InjectedAbort("injected anchor fstat failure")

    def tracking_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal anchor
        if dir_fd is None:
            descriptor = cast(int, real_open(path, flags, mode))
        else:
            descriptor = cast(int, real_open(path, flags, mode, dir_fd=dir_fd))
        if anchor < 0:
            anchor = descriptor
        return descriptor

    def fail_anchor_fstat(descriptor: int) -> os.stat_result:
        if descriptor == anchor:
            raise failure
        return cast(os.stat_result, real_fstat(descriptor))

    monkeypatch.setattr(snapshot.os, "open", tracking_open)
    monkeypatch.setattr(snapshot.os, "fstat", fail_anchor_fstat)
    with pytest.raises(_InjectedAbort) as caught:
        snapshot._open_anchored_repository_root(repository)

    assert caught.value is failure
    with pytest.raises(OSError) as closed:
        real_fstat(anchor)
    assert closed.value.errno == errno.EBADF


def test_checked_child_first_fstat_baseexception_closes_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = tmp_path / "child"
    child.write_bytes(b"payload")
    parent = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    before = os.stat("child", dir_fd=parent, follow_symlinks=False)
    real_open = snapshot.os.open
    real_fstat = snapshot.os.fstat
    opened = -1
    failure = _InjectedAbort("injected first child fstat failure")

    def tracking_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal opened
        if dir_fd is None:
            opened = cast(int, real_open(path, flags, mode))
        else:
            opened = cast(int, real_open(path, flags, mode, dir_fd=dir_fd))
        return opened

    def fail_opened_fstat(descriptor: int) -> os.stat_result:
        if descriptor == opened:
            raise failure
        return cast(os.stat_result, real_fstat(descriptor))

    monkeypatch.setattr(snapshot.os, "open", tracking_open)
    monkeypatch.setattr(snapshot.os, "fstat", fail_opened_fstat)
    try:
        with pytest.raises(_InjectedAbort) as caught:
            snapshot._open_checked_child(parent, "child", before, directory=False)
        assert caught.value is failure
        assert stat.S_ISDIR(real_fstat(parent).st_mode)
        with pytest.raises(OSError) as closed:
            real_fstat(opened)
        assert closed.value.errno == errno.EBADF
    finally:
        os.close(parent)


def test_anchored_close_attempts_every_descriptor_and_preserves_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    real_open_anchored = snapshot._open_anchored_repository_root
    real_close = snapshot.os.close
    real_fstat = snapshot.os.fstat
    descriptors: tuple[int, ...] = ()
    failed_descriptor = -1
    anchored_result: Any = None
    primary = _InjectedAbort("injected observation failure")

    def tracking_open_anchored(root: Path) -> Any:
        nonlocal anchored_result, descriptors, failed_descriptor
        anchored_result = real_open_anchored(root)
        descriptors = tuple(anchored_result.descriptors)
        failed_descriptor = descriptors[-1]
        return anchored_result

    def fail_measurement(_descriptor: int) -> dict[str, Any]:
        raise primary

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor == failed_descriptor:
            raise OSError("injected anchored close failure")

    monkeypatch.setattr(snapshot, "_open_anchored_repository_root", tracking_open_anchored)
    monkeypatch.setattr(snapshot, "_measure_once", fail_measurement)
    monkeypatch.setattr(snapshot.os, "close", close_then_fail)
    with pytest.raises(_InjectedAbort) as caught:
        snapshot._measure_repository(repository)

    assert caught.value is primary
    assert anchored_result.descriptors == []
    assert any("anchored repository cleanup also failed" in note for note in primary.__notes__)
    for descriptor in descriptors:
        with pytest.raises(OSError) as closed:
            real_fstat(descriptor)
        assert closed.value.errno == errno.EBADF


@pytest.mark.parametrize("name", ["caf\N{LATIN SMALL LETTER E WITH ACUTE}.py", "white space.py"])
def test_ambiguous_or_non_ascii_inventory_paths_fail_closed(tmp_path: Path, name: str) -> None:
    repository, _payloads = _make_repository(tmp_path)
    (repository / "alberta_framework" / "core" / name).write_bytes(b"ambiguous\n")
    with pytest.raises(
        snapshot.ForagerMatchedV3LocalSourceSnapshotError,
        match="ASCII path component",
    ):
        snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)


def test_casefold_path_aliases_fail_closed(tmp_path: Path) -> None:
    repository, _payloads = _make_repository(tmp_path)
    core = repository / "alberta_framework" / "core"
    (core / "Alias.py").write_bytes(b"upper\n")
    (core / "alias.py").write_bytes(b"lower\n")
    if len({path.name for path in core.iterdir() if path.name.casefold() == "alias.py"}) < 2:
        pytest.skip("filesystem is case-insensitive")
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="casefold"):
        snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "message"),
    [
        ("_MAX_FILE_BYTES", 4, "file exceeds"),
        ("_MAX_TOTAL_BYTES", 8, "byte bound"),
        ("_MAX_FILES", 2, "file bound"),
        ("_MAX_DIRECTORIES", 1, "directory bound"),
        ("_MAX_DEPTH", 1, "depth bound"),
        ("_MAX_ENTRIES", 2, "entry bound"),
    ],
)
def test_size_tree_count_and_depth_bounds_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: int,
    message: str,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    monkeypatch.setattr(snapshot, limit_name, limit_value)
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match=message):
        snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)


def test_directory_enumeration_is_bounded_before_sorting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEntry:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeScandir:
        def __enter__(self) -> FakeScandir:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self) -> FakeScandir:
            return self

        def __next__(self) -> FakeEntry:
            current = self.consumed
            self.consumed += 1
            return FakeEntry(f"entry{current}")

        consumed = 0

    iterator = FakeScandir()
    monkeypatch.setattr(snapshot, "_MAX_ENTRIES", 3)
    monkeypatch.setattr(snapshot.os, "scandir", lambda _descriptor: iterator)

    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="entry bound"):
        snapshot._safe_sorted_names(7, maximum_entries=3)
    assert iterator.consumed == 4


def test_openat_path_swap_is_detected_before_content_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    target = repository / "alberta_framework" / "core" / "a.txt"
    replacement = tmp_path / "replacement.txt"
    replacement.write_bytes(b"replacement\n")
    original_open = snapshot.os.open
    swapped = False

    def racing_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "a.txt" and dir_fd is not None and not swapped:
            swapped = True
            os.replace(replacement, target)
        if dir_fd is None:
            return cast(int, original_open(path, flags, mode))
        return cast(int, original_open(path, flags, mode, dir_fd=dir_fd))

    monkeypatch.setattr(snapshot.os, "open", racing_open)
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="changed"):
        snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)
    assert swapped is True


def test_open_read_path_replacement_is_detected_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    target = repository / "alberta_framework" / "core" / "large.bin"
    payload = b"A" * (snapshot._READ_CHUNK_BYTES + 257)
    target.write_bytes(payload)
    replacement = tmp_path / "large-replacement.bin"
    replacement.write_bytes(b"B" * len(payload))
    original_open_checked = snapshot._open_checked_child
    original_read = snapshot.os.read
    target_descriptor: int | None = None
    replaced = False

    def tracked_open(
        parent_descriptor: int,
        name: str,
        before: os.stat_result,
        *,
        directory: bool,
    ) -> tuple[int, os.stat_result]:
        nonlocal target_descriptor
        result = original_open_checked(
            parent_descriptor,
            name,
            before,
            directory=directory,
        )
        if name == "large.bin":
            target_descriptor = result[0]
        return cast(tuple[int, os.stat_result], result)

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        chunk = cast(bytes, original_read(descriptor, count))
        if descriptor == target_descriptor and chunk and not replaced:
            replaced = True
            os.replace(replacement, target)
        return chunk

    monkeypatch.setattr(snapshot, "_open_checked_child", tracked_open)
    monkeypatch.setattr(snapshot.os, "read", racing_read)
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="changed"):
        snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)
    assert replaced is True


def test_disagreement_between_complete_measurement_passes_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    target = repository / "alberta_framework" / "core" / "a.txt"
    original_measure_once = snapshot._measure_once
    calls = 0

    def changing_measurement(repository_descriptor: int) -> dict[str, Any]:
        nonlocal calls
        result = cast(dict[str, Any], original_measure_once(repository_descriptor))
        calls += 1
        if calls == 1:
            target.write_bytes(b"changed between passes\n")
        return result

    monkeypatch.setattr(snapshot, "_measure_once", changing_measurement)
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="disagreed"):
        snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)
    assert calls == 2


@pytest.mark.parametrize("change", ["appear", "disappear"])
def test_empty_directory_change_between_complete_passes_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    empty_directory = repository / "alberta_framework" / "empty"
    if change == "disappear":
        empty_directory.mkdir()
    original_measure_once = snapshot._measure_once
    calls = 0

    def changing_measurement(repository_descriptor: int) -> dict[str, Any]:
        nonlocal calls
        result = cast(dict[str, Any], original_measure_once(repository_descriptor))
        calls += 1
        if calls == 1:
            if change == "appear":
                empty_directory.mkdir()
            else:
                empty_directory.rmdir()
        return result

    monkeypatch.setattr(snapshot, "_measure_once", changing_measurement)
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="disagreed"):
        snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)
    assert calls == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("traversal", "path component"),
        ("duplicate", "duplicate or aliased"),
        ("orphan_directory", "bound parent"),
        ("bool_size", "exact integer"),
        ("bool_count", "exact integer"),
    ],
)
def test_coherently_rehashed_semantic_manifest_drift_fails_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)
    manifest = measured.manifest()
    if mutation == "traversal":
        manifest["files"][0]["path"] = "alberta_framework/../escape.py"
    elif mutation == "duplicate":
        manifest["files"][1]["path"] = manifest["files"][0]["path"]
    elif mutation == "orphan_directory":
        manifest["directories"].append("alberta_framework/missing/child")
    elif mutation == "bool_size":
        manifest["files"][0]["size_bytes"] = False
    else:
        manifest["inventory"]["file_count"] = True
    raw, full_sha256 = _rehash_manifest(manifest)

    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match=message):
        snapshot.parse_matched_v3_local_source_snapshot_manifest(
            raw,
            expected_full_sha256=full_sha256,
        )


@pytest.mark.parametrize("mutation", ["directory_depth", "file_depth", "node_collision"])
def test_manifest_rejects_out_of_bound_depth_and_file_directory_collisions(
    tmp_path: Path,
    mutation: str,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)
    manifest = measured.manifest()
    if mutation == "directory_depth":
        path = "alberta_framework"
        for index in range(snapshot._MAX_DEPTH):
            path = f"{path}/d{index}"
            manifest["directories"].append(path)
        manifest["directories"].sort()
        manifest["inventory"]["directory_count"] = len(manifest["directories"])
    elif mutation == "file_depth":
        path = "alberta_framework"
        for index in range(snapshot._MAX_DEPTH):
            path = f"{path}/d{index}"
            if index < snapshot._MAX_DEPTH - 1:
                manifest["directories"].append(path)
        manifest["directories"].sort()
        manifest["inventory"]["directory_count"] = len(manifest["directories"])
        manifest["files"].append(
            {
                "path": f"{path}/too_deep.py",
                "size_bytes": 1,
                "sha256": hashlib.sha256(b"x").hexdigest(),
            }
        )
        manifest["files"].sort(key=lambda record: cast(str, record["path"]).encode("ascii"))
        manifest["inventory"]["file_count"] = len(manifest["files"])
        manifest["inventory"]["total_size_bytes"] += 1
    else:
        manifest["directories"].append("alberta_framework/collision")
        manifest["directories"].sort()
        manifest["files"].append(
            {
                "path": "alberta_framework/collision",
                "size_bytes": 1,
                "sha256": hashlib.sha256(b"x").hexdigest(),
            }
        )
        manifest["files"].sort(key=lambda record: cast(str, record["path"]).encode("ascii"))
        manifest["inventory"]["directory_count"] = len(manifest["directories"])
        manifest["inventory"]["file_count"] = len(manifest["files"])
        manifest["inventory"]["total_size_bytes"] += 1
    raw, full_sha256 = _rehash_manifest(manifest)

    message = "depth bound" if mutation != "node_collision" else "both a file and a directory"
    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match=message):
        snapshot.parse_matched_v3_local_source_snapshot_manifest(
            raw,
            expected_full_sha256=full_sha256,
        )


def test_manifest_requires_framework_initializer(tmp_path: Path) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)
    manifest = measured.manifest()
    manifest["files"] = [
        record for record in manifest["files"] if record["path"] != "alberta_framework/__init__.py"
    ]
    raw, full_sha256 = _rehash_manifest(manifest)

    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="initializer"):
        snapshot.parse_matched_v3_local_source_snapshot_manifest(
            raw,
            expected_full_sha256=full_sha256,
        )


@pytest.mark.parametrize("location", ["file", "tree", "body", "external"])
def test_zero_placeholder_digests_fail_closed(tmp_path: Path, location: str) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)
    manifest = measured.manifest()
    if location == "external":
        raw = measured.canonical_manifest_bytes
        full_sha256 = "0" * 64
    elif location == "file":
        manifest["files"][0]["sha256"] = "0" * 64
        raw, full_sha256 = _rehash_manifest(manifest)
    else:
        manifest[f"manifest_{location}_sha256" if location == "body" else "tree"] = (
            "0" * 64
            if location == "body"
            else {
                "schema_version": snapshot.LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION,
                "sha256": "0" * 64,
            }
        )
        if location == "tree":
            body = copy.deepcopy(manifest)
            body.pop("manifest_body_sha256")
            manifest["manifest_body_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        raw = _canonical(manifest)
        full_sha256 = hashlib.sha256(raw).hexdigest()

    with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError, match="nonzero"):
        snapshot.parse_matched_v3_local_source_snapshot_manifest(
            raw,
            expected_full_sha256=full_sha256,
        )


def test_strict_parser_rejects_duplicate_noncanonical_float_nonfinite_and_unicode_json(
    tmp_path: Path,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)
    raw = measured.canonical_manifest_bytes
    variants = [
        b'{"schema_version":"duplicate",' + raw[1:],
        b" " + raw,
        raw.replace(b'"size_bytes":6', b'"size_bytes":6.0', 1),
        raw.replace(b'"size_bytes":6', b'"size_bytes":NaN', 1),
        raw.replace(b'"path":"alberta_framework/', b'"path":"\\u00e9/', 1),
    ]
    for variant in variants:
        with pytest.raises(snapshot.ForagerMatchedV3LocalSourceSnapshotError):
            snapshot.parse_matched_v3_local_source_snapshot_manifest(
                variant,
                expected_full_sha256=hashlib.sha256(variant).hexdigest(),
            )


def test_public_apis_require_explicit_caller_values_and_expose_no_default_root() -> None:
    measure_signature = inspect.signature(snapshot.measure_matched_v3_local_source_snapshot)
    verify_signature = inspect.signature(snapshot.verify_matched_v3_local_source_snapshot)
    parse_signature = inspect.signature(snapshot.parse_matched_v3_local_source_snapshot_manifest)

    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in measure_signature.parameters.values()
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in verify_signature.parameters.values()
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in parse_signature.parameters.values()
    )
    assert not hasattr(snapshot, "DEFAULT_REPOSITORY_ROOT")
    assert not hasattr(snapshot, "CURRENT_REPOSITORY_MANIFEST")
    assert not hasattr(snapshot, "issue_matched_v3_local_execution_capability")
