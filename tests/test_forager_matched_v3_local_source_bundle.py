"""Adversarial tests for the matched-v3 local source USTAR producer."""

from __future__ import annotations

import copy
import errno
import hashlib
import importlib.util
import inspect
import io
import json
import os
import pickle
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]
_BUNDLE_PATH = (
    _ROOT / "alberta_framework" / "benchmarks" / "forager_matched_v3_local_source_bundle.py"
)
_SNAPSHOT_PATH = (
    _ROOT / "alberta_framework" / "benchmarks" / "forager_matched_v3_local_source_snapshot.py"
)


def _load_direct(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


snapshot = _load_direct("_matched_v3_bundle_snapshot_test", _SNAPSHOT_PATH)
bundle = _load_direct("_matched_v3_local_source_bundle_test", _BUNDLE_PATH)


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
        "alberta_framework/core/empty.bin": b"",
        "alberta_framework/core/z.py": b"VALUE = 7\n",
    }
    for relative_path, payload in payloads.items():
        target = repository / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (core / "ignored.pyc").write_bytes(b"excluded-cache-file")
    (cache / "ignored.cpython-312.pyc").write_bytes(b"excluded-cache-tree")
    return repository, payloads


def _expected(repository: Path) -> Any:
    return snapshot.measure_matched_v3_local_source_snapshot(repository_root=repository)


def _open_bundle(repository: Path, measured: Any) -> Any:
    return bundle.retain_matched_v3_local_source_bundle(
        repository_root=repository,
        expected_canonical_snapshot_manifest_bytes=measured.canonical_manifest_bytes,
        expected_snapshot_manifest_sha256=measured.full_sha256,
        expected_snapshot_tree_sha256=measured.tree_sha256,
    )


def _read_capability(capability: Any) -> bytes:
    descriptor = capability.subprocess_pass_fds[0]
    return os.pread(descriptor, capability.archive_size_bytes, 0)


def _rehash_snapshot_manifest(manifest: dict[str, Any]) -> tuple[bytes, str, str]:
    tree_payload = {
        "schema_version": snapshot.LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION,
        "directories": manifest["directories"],
        "files": manifest["files"],
    }
    tree_sha256 = hashlib.sha256(_canonical(tree_payload)).hexdigest()
    manifest["tree"]["sha256"] = tree_sha256
    body = copy.deepcopy(manifest)
    body.pop("manifest_body_sha256", None)
    manifest["manifest_body_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    raw = _canonical(manifest)
    return raw, hashlib.sha256(raw).hexdigest(), tree_sha256


def _rehash_receipt(receipt: dict[str, Any]) -> tuple[bytes, str]:
    body = copy.deepcopy(receipt)
    body.pop("receipt_body_sha256", None)
    receipt["receipt_body_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    raw = _canonical(receipt)
    return raw, hashlib.sha256(raw).hexdigest()


def test_descriptor_is_frozen_and_denies_every_authority() -> None:
    raw = bundle.canonical_matched_v3_local_source_bundle_descriptor_bytes()
    descriptor = bundle.matched_v3_local_source_bundle_descriptor()

    assert hashlib.sha256(raw).hexdigest() == bundle.LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SHA256
    assert bundle.parse_matched_v3_local_source_bundle_descriptor(raw) == descriptor
    assert descriptor["archive"]["format"] == "canonical_posix_ustar_uncompressed"
    assert descriptor["archive"]["members"] == "exact_snapshot_regular_file_inventory_only"
    assert descriptor["archive"]["member_mode"] == "0444"
    assert descriptor["archive"]["uid_gid_mtime"] == 0
    assert descriptor["retention"]["sealed_anonymous_read_only_descriptor"] is True
    assert all(value is False for value in descriptor["claims"].values())


def test_import_is_pure_stdlib_and_observes_no_filesystem() -> None:
    script = r"""
import builtins, fcntl, hashlib, hmac, json, os, pathlib, re, stat, sys, types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import *
calls = []
def forbidden(*args, **kwargs):
    calls.append(str(args[0]) if args else "")
    raise AssertionError("filesystem observation at import")
builtins.open = forbidden
os.open = forbidden
os.stat = forbidden
os.fstat = forbidden
os.scandir = forbidden
path = pathlib.Path(sys.argv[1])
source = path.read_bytes()
name = "_matched_v3_source_bundle_clean_probe"
module = types.ModuleType(name)
module.__file__ = str(path)
module.__package__ = ""
sys.modules[name] = module
exec(compile(source, str(path), "exec"), module.__dict__)
print(json.dumps({"calls": calls, "heavy": any(
    key == "alberta_framework" or key.startswith(("alberta_framework.", "jax", "numpy"))
    for key in sys.modules
)}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", inspect.cleandoc(script), str(_BUNDLE_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"calls": [], "heavy": False}


def test_bundle_is_deterministic_exact_and_independently_replayable(tmp_path: Path) -> None:
    repository, payloads = _make_repository(tmp_path)
    measured = _expected(repository)

    with _open_bundle(repository, measured) as first:
        first_raw = _read_capability(first)
        first_sha256 = first.archive_sha256
        first_size = first.archive_size_bytes
        first_receipt = first.receipt_bytes
        receipt = first.reverify()
        assert first.read_archive_bytes() == first_raw
        assert first.source_manifest_sha256 == measured.full_sha256
        assert first.source_tree_sha256 == measured.tree_sha256
        assert first.source_snapshot_manifest_sha256 == measured.full_sha256
        assert first.source_snapshot_tree_sha256 == measured.tree_sha256
        assert first.member_count == len(payloads)
        assert any(member["path"] == "FORAGER_BENCHMARK.md" for member in receipt["members"])
        assert receipt["archive"]["sha256"] == first_sha256
        assert receipt["source_snapshot"]["manifest_sha256"] == measured.full_sha256
        assert receipt["source_snapshot"]["tree_sha256"] == measured.tree_sha256
        assert all(value is False for value in receipt["claims"].values())
        independently = bundle.verify_matched_v3_local_source_bundle_archive(
            descriptor=first.subprocess_pass_fds[0],
            expected_archive_size_bytes=first_size,
            expected_archive_sha256=first_sha256,
            expected_receipt_bytes=first_receipt,
            expected_receipt_sha256=first.receipt_sha256,
            expected_source_snapshot_manifest_sha256=measured.full_sha256,
            expected_source_snapshot_tree_sha256=measured.tree_sha256,
        )
        assert independently == receipt

    with _open_bundle(repository, measured) as second:
        assert _read_capability(second) == first_raw
        assert second.archive_sha256 == first_sha256
        assert second.archive_size_bytes == first_size
        assert second.receipt_bytes == first_receipt

    assert len(first_raw) % 10_240 == 0
    with tarfile.open(fileobj=io.BytesIO(first_raw), mode="r:") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == sorted(payloads)
        for member in members:
            assert member.isfile()
            assert member.mode == 0o444
            assert member.uid == member.gid == member.mtime == 0
            assert member.uname == member.gname == ""
            extracted = archive.extractfile(member)
            assert extracted is not None
            assert extracted.read() == payloads[member.name]


def test_yielded_bundle_releases_source_descriptors_and_construction_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = _expected(repository)
    real_open_anchored = bundle._open_anchored_repository_root
    source_descriptors: list[int] = []

    def tracking_open_anchored(root: Path) -> Any:
        anchored = real_open_anchored(root)
        source_descriptors.extend(anchored.descriptors)
        return anchored

    monkeypatch.setattr(bundle, "_open_anchored_repository_root", tracking_open_anchored)
    manager = _open_bundle(repository, measured)
    generator = manager.gen
    with manager as retained:
        assert source_descriptors
        for descriptor in source_descriptors:
            with pytest.raises(OSError) as closed:
                os.fstat(descriptor)
            assert closed.value.errno == errno.EBADF

        retained_descriptor = retained.subprocess_pass_fds[0]
        assert retained_descriptor not in source_descriptors
        assert stat.S_ISREG(os.fstat(retained_descriptor).st_mode)
        assert retained.reverify() == retained.receipt()

        frame = generator.gi_frame
        assert frame is not None
        released_locals = {
            "expected_canonical_snapshot_manifest_bytes",
            "expected_tree",
            "observed_after",
            "observed_before",
            "receipt",
            "receipt_bytes",
            "records",
            "snapshot",
        }
        assert released_locals.isdisjoint(frame.f_locals)
        assert frame.f_locals["anchored"] is None


def test_retain_failure_cleanup_attempts_writable_and_every_anchored_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = _expected(repository)
    real_open_anchored = bundle._open_anchored_repository_root
    real_create_memfd = bundle._create_private_memfd
    real_close = bundle.os.close
    real_fstat = bundle.os.fstat
    anchored_descriptors: tuple[int, ...] = ()
    failed_anchor = -1
    writable = -1
    primary = _InjectedAbort("injected archive construction failure")

    def tracking_open_anchored(root: Path) -> Any:
        nonlocal anchored_descriptors, failed_anchor
        anchored = real_open_anchored(root)
        anchored_descriptors = tuple(anchored.descriptors)
        failed_anchor = anchored_descriptors[-1]
        return anchored

    def tracking_create_memfd() -> int:
        nonlocal writable
        writable = cast(int, real_create_memfd())
        return writable

    def fail_archive(*_args: Any, **_kwargs: Any) -> tuple[int, str]:
        raise primary

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor in {writable, failed_anchor}:
            raise OSError("injected retained-producer cleanup close failure")

    monkeypatch.setattr(bundle, "_open_anchored_repository_root", tracking_open_anchored)
    monkeypatch.setattr(bundle, "_create_private_memfd", tracking_create_memfd)
    monkeypatch.setattr(bundle, "_write_archive", fail_archive)
    monkeypatch.setattr(bundle.os, "close", close_then_fail)
    with pytest.raises(_InjectedAbort) as caught:
        with _open_bundle(repository, measured):
            pass

    assert caught.value is primary
    notes = getattr(primary, "__notes__", [])
    assert any("writable source bundle descriptor" in note for note in notes)
    assert any("anchored repository root" in note for note in notes)
    for descriptor in (*anchored_descriptors, writable):
        with pytest.raises(OSError) as closed:
            real_fstat(descriptor)
        assert closed.value.errno == errno.EBADF


def test_exact_capability_handoff_is_accepted_by_cpu_oci_plan(tmp_path: Path) -> None:
    from alberta_framework.benchmarks import forager_matched_v3_cpu_oci_build_plan as oci_plan

    repository, payloads = _make_repository(tmp_path)
    measured = _expected(repository)
    with _open_bundle(repository, measured) as retained:
        source = oci_plan.CanonicalSourceBundleInput(
            archive_bytes=retained.read_archive_bytes(),
            expected_archive_sha256=retained.archive_sha256,
            expected_archive_size_bytes=retained.archive_size_bytes,
            expected_member_count=retained.member_count,
            receipt_bytes=retained.receipt_bytes,
            expected_receipt_sha256=retained.receipt_sha256,
            source_manifest_sha256=retained.source_manifest_sha256,
            source_tree_sha256=retained.source_tree_sha256,
            staging_manifest_sha256=None,
        )
    validated = oci_plan._validate_source_bundle(
        source,
        role="local_alberta",
        context_path="inputs/local-alberta-source.v1.tar",
    )

    assert validated.archive_sha256 == source.expected_archive_sha256
    assert validated.archive_size_bytes == source.expected_archive_size_bytes
    assert validated.member_count == len(payloads)
    assert validated.source_manifest_sha256 == measured.full_sha256
    assert validated.source_tree_sha256 == measured.tree_sha256
    assert validated.receipt_sha256 == source.expected_receipt_sha256
    assert validated.receipt_size_bytes == len(source.receipt_bytes)
    assert validated.producer_descriptor_sha256 == bundle.LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SHA256
    assert validated.staging_manifest_sha256 is None
    assert [member["path"] for member in validated.members] == sorted(payloads)


def test_expected_artifact_validation_precedes_filesystem_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = _expected(repository)

    def forbidden(_root: object) -> object:
        raise AssertionError("filesystem observed before expected artifact validation")

    monkeypatch.setattr(bundle, "_open_anchored_repository_root", forbidden)
    with pytest.raises(bundle.ForagerMatchedV3LocalSourceBundleError, match="digest"):
        with bundle.retain_matched_v3_local_source_bundle(
            repository_root=repository,
            expected_canonical_snapshot_manifest_bytes=measured.canonical_manifest_bytes,
            expected_snapshot_manifest_sha256="1" * 64,
            expected_snapshot_tree_sha256=measured.tree_sha256,
        ):
            pass


def test_final_root_verification_stat_failure_closes_every_opened_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    real_open = bundle.os.open
    real_stat = bundle.os.stat
    real_fstat = bundle.os.fstat
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

    monkeypatch.setattr(bundle.os, "open", tracking_open)
    monkeypatch.setattr(bundle.os, "stat", fail_first_final_verification_stat)
    with pytest.raises(
        bundle.ForagerMatchedV3LocalSourceBundleError,
        match="root locator changed during bundling",
    ):
        bundle._open_anchored_repository_root(repository)

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
    real_open = bundle.os.open
    real_fstat = bundle.os.fstat
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

    monkeypatch.setattr(bundle.os, "open", tracking_open)
    monkeypatch.setattr(bundle.os, "fstat", fail_anchor_fstat)
    with pytest.raises(_InjectedAbort) as caught:
        bundle._open_anchored_repository_root(repository)

    assert caught.value is failure
    with pytest.raises(OSError) as closed:
        real_fstat(anchor)
    assert closed.value.errno == errno.EBADF


def test_private_memfd_first_fstat_baseexception_closes_and_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_creator = bundle.os.memfd_create
    real_fstat = bundle.os.fstat
    real_close = bundle.os.close
    created = -1
    failure = _InjectedAbort("injected first memfd fstat failure")

    def tracking_creator(name: str, flags: int) -> int:
        nonlocal created
        created = cast(int, real_creator(name, flags))
        return created

    def fail_created_fstat(descriptor: int) -> os.stat_result:
        if descriptor == created:
            raise failure
        return cast(os.stat_result, real_fstat(descriptor))

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor == created:
            raise OSError("injected memfd cleanup close failure")

    monkeypatch.setattr(bundle.os, "memfd_create", tracking_creator)
    monkeypatch.setattr(bundle.os, "fstat", fail_created_fstat)
    monkeypatch.setattr(bundle.os, "close", close_then_fail)
    with pytest.raises(_InjectedAbort) as caught:
        bundle._create_private_memfd()

    assert caught.value is failure
    assert any("cleanup close also failed" in note for note in failure.__notes__)
    with pytest.raises(OSError) as closed:
        real_fstat(created)
    assert closed.value.errno == errno.EBADF


def test_sealed_readonly_first_fstat_baseexception_closes_only_new_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writable = os.memfd_create(
        "bundle-seal-baseexception",
        os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
    )
    os.write(writable, b"payload")
    real_open = bundle.os.open
    real_fstat = bundle.os.fstat
    readonly = -1
    failure = _InjectedAbort("injected first read-only fstat failure")

    def tracking_open(path: Any, flags: int, mode: int = 0o777) -> int:
        nonlocal readonly
        readonly = cast(int, real_open(path, flags, mode))
        return readonly

    def fail_readonly_fstat(descriptor: int) -> os.stat_result:
        if descriptor == readonly:
            raise failure
        return cast(os.stat_result, real_fstat(descriptor))

    monkeypatch.setattr(bundle.os, "open", tracking_open)
    monkeypatch.setattr(bundle.os, "fstat", fail_readonly_fstat)
    try:
        with pytest.raises(_InjectedAbort) as caught:
            bundle._seal_and_reopen_readonly(writable, expected_size=7)
        assert caught.value is failure
        assert stat.S_ISREG(real_fstat(writable).st_mode)
        with pytest.raises(OSError) as closed:
            real_fstat(readonly)
        assert closed.value.errno == errno.EBADF
    finally:
        os.close(writable)


def test_duplicate_first_fstat_baseexception_closes_only_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    real_fcntl = bundle.fcntl.fcntl
    real_fstat = bundle.os.fstat
    duplicate = -1
    failure = _InjectedAbort("injected first duplicate fstat failure")

    def tracking_fcntl(descriptor: int, command: int, argument: int = 0) -> int:
        nonlocal duplicate
        result = cast(int, real_fcntl(descriptor, command, argument))
        if command == bundle.fcntl.F_DUPFD_CLOEXEC:
            duplicate = result
        return result

    def fail_duplicate_fstat(descriptor: int) -> os.stat_result:
        if descriptor == duplicate:
            raise failure
        return cast(os.stat_result, real_fstat(descriptor))

    monkeypatch.setattr(bundle.fcntl, "fcntl", tracking_fcntl)
    monkeypatch.setattr(bundle.os, "fstat", fail_duplicate_fstat)
    try:
        with pytest.raises(_InjectedAbort) as caught:
            bundle._duplicate_directory_descriptor(source, "test source")
        assert caught.value is failure
        assert stat.S_ISDIR(real_fstat(source).st_mode)
        with pytest.raises(OSError) as closed:
            real_fstat(duplicate)
        assert closed.value.errno == errno.EBADF
    finally:
        os.close(source)


def test_checked_child_first_fstat_baseexception_closes_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = tmp_path / "child"
    child.write_bytes(b"payload")
    parent = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    before = os.stat("child", dir_fd=parent, follow_symlinks=False)
    real_open = bundle.os.open
    real_fstat = bundle.os.fstat
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

    monkeypatch.setattr(bundle.os, "open", tracking_open)
    monkeypatch.setattr(bundle.os, "fstat", fail_opened_fstat)
    try:
        with pytest.raises(_InjectedAbort) as caught:
            bundle._open_checked_child(parent, "child", before, directory=False)
        assert caught.value is failure
        assert stat.S_ISDIR(real_fstat(parent).st_mode)
        with pytest.raises(OSError) as closed:
            real_fstat(opened)
        assert closed.value.errno == errno.EBADF
    finally:
        os.close(parent)


def test_relative_parent_child_baseexception_closes_child_and_root_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "child").mkdir()
    root = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    real_duplicate = bundle._duplicate_directory_descriptor
    real_open = bundle.os.open
    real_fstat = bundle.os.fstat
    duplicate = -1
    child = -1
    failure = _InjectedAbort("injected relative-parent child fstat failure")

    def tracking_duplicate(descriptor: int, label: str) -> int:
        nonlocal duplicate
        duplicate = cast(int, real_duplicate(descriptor, label))
        return duplicate

    def tracking_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal child
        if dir_fd is None:
            child = cast(int, real_open(path, flags, mode))
        else:
            child = cast(int, real_open(path, flags, mode, dir_fd=dir_fd))
        return child

    def fail_child_fstat(descriptor: int) -> os.stat_result:
        if descriptor == child:
            raise failure
        return cast(os.stat_result, real_fstat(descriptor))

    monkeypatch.setattr(bundle, "_duplicate_directory_descriptor", tracking_duplicate)
    monkeypatch.setattr(bundle.os, "open", tracking_open)
    monkeypatch.setattr(bundle.os, "fstat", fail_child_fstat)
    try:
        with pytest.raises(_InjectedAbort) as caught:
            bundle._open_relative_parent(root, "child/file")
        assert caught.value is failure
        assert stat.S_ISDIR(real_fstat(root).st_mode)
        for descriptor in (duplicate, child):
            with pytest.raises(OSError) as closed:
                real_fstat(descriptor)
            assert closed.value.errno == errno.EBADF
    finally:
        os.close(root)


@pytest.mark.parametrize("mutation", ["content", "extra", "symlink", "hardlink", "fifo"])
def test_post_snapshot_source_mutation_or_substitution_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = _expected(repository)
    core = repository / "alberta_framework" / "core"
    if mutation == "content":
        (core / "a.txt").write_bytes(b"drift\n")
    elif mutation == "extra":
        (core / "extra.py").write_bytes(b"EXTRA = True\n")
    elif mutation == "symlink":
        (core / "a.txt").unlink()
        (core / "a.txt").symlink_to("z.py")
    elif mutation == "hardlink":
        os.link(core / "a.txt", core / "extra.txt")
    else:
        try:
            os.mkfifo(core / "extra.pipe")
        except (AttributeError, NotImplementedError, OSError) as exc:
            pytest.skip(f"FIFO creation unavailable: {exc}")

    with pytest.raises(bundle.ForagerMatchedV3LocalSourceBundleError):
        with _open_bundle(repository, measured):
            pass


def test_coherently_rehashed_duplicate_and_traversal_manifests_fail_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = _expected(repository)

    def forbidden(_root: object) -> object:
        raise AssertionError("invalid manifest reached filesystem observation")

    monkeypatch.setattr(bundle, "_open_anchored_repository_root", forbidden)
    for mutation in ("duplicate", "traversal"):
        manifest = measured.manifest()
        if mutation == "duplicate":
            manifest["files"][1]["path"] = manifest["files"][0]["path"]
        else:
            manifest["files"][1]["path"] = "alberta_framework/../escape.py"
        raw, full_sha256, tree_sha256 = _rehash_snapshot_manifest(manifest)
        with pytest.raises(bundle.ForagerMatchedV3LocalSourceBundleError):
            with bundle.retain_matched_v3_local_source_bundle(
                repository_root=repository,
                expected_canonical_snapshot_manifest_bytes=raw,
                expected_snapshot_manifest_sha256=full_sha256,
                expected_snapshot_tree_sha256=tree_sha256,
            ):
                pass


def test_partial_source_read_fails_and_never_returns_a_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = _expected(repository)
    original_read = bundle.os.read
    injected = False

    def truncated_read(descriptor: int, count: int) -> bytes:
        nonlocal injected
        metadata = os.fstat(descriptor)
        if stat.S_ISREG(metadata.st_mode) and metadata.st_size == len(b"alpha\n") and not injected:
            injected = True
            return b""
        return cast(bytes, original_read(descriptor, count))

    monkeypatch.setattr(bundle.os, "read", truncated_read)
    with pytest.raises(bundle.ForagerMatchedV3LocalSourceBundleError, match="ended"):
        with _open_bundle(repository, measured):
            pass
    assert injected is True


def test_change_between_archive_and_post_measurement_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = _expected(repository)
    original_write_archive = bundle._write_archive

    def mutate_after_archive(*args: object, **kwargs: object) -> tuple[int, str]:
        result = cast(tuple[int, str], original_write_archive(*args, **kwargs))
        (repository / "alberta_framework" / "core" / "a.txt").write_bytes(b"late drift\n")
        return result

    monkeypatch.setattr(bundle, "_write_archive", mutate_after_archive)
    with pytest.raises(bundle.ForagerMatchedV3LocalSourceBundleError, match="post-archive"):
        with _open_bundle(repository, measured):
            pass


def test_receipt_parser_rejects_coherently_rehashed_archive_size_and_duplicate_member(
    tmp_path: Path,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = _expected(repository)
    with _open_bundle(repository, measured) as retained:
        original = retained.receipt()

    wrong_size = copy.deepcopy(original)
    wrong_size["archive"]["size_bytes"] += 10_240
    raw, digest = _rehash_receipt(wrong_size)
    with pytest.raises(bundle.ForagerMatchedV3LocalSourceBundleError, match="complete member"):
        bundle.parse_matched_v3_local_source_bundle_receipt(raw, expected_receipt_sha256=digest)

    duplicate = copy.deepcopy(original)
    duplicate["members"][1]["path"] = duplicate["members"][0]["path"]
    raw, digest = _rehash_receipt(duplicate)
    with pytest.raises(bundle.ForagerMatchedV3LocalSourceBundleError, match="path/mode"):
        bundle.parse_matched_v3_local_source_bundle_receipt(raw, expected_receipt_sha256=digest)


def test_independent_archive_verifier_rejects_header_and_payload_tampering(tmp_path: Path) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = _expected(repository)
    with _open_bundle(repository, measured) as retained:
        raw = retained.read_archive_bytes()
        values = {
            "size": retained.archive_size_bytes,
            "archive_sha256": retained.archive_sha256,
            "receipt": retained.receipt_bytes,
            "receipt_sha256": retained.receipt_sha256,
            "manifest_sha256": retained.source_manifest_sha256,
            "tree_sha256": retained.source_tree_sha256,
        }

    for offset in (0, 512):
        descriptor = os.memfd_create("tampered-source-bundle", os.MFD_CLOEXEC)
        try:
            tampered = bytearray(raw)
            tampered[offset] ^= 1
            os.write(descriptor, tampered)
            with pytest.raises(bundle.ForagerMatchedV3LocalSourceBundleError):
                bundle.verify_matched_v3_local_source_bundle_archive(
                    descriptor=descriptor,
                    expected_archive_size_bytes=cast(int, values["size"]),
                    expected_archive_sha256=cast(str, values["archive_sha256"]),
                    expected_receipt_bytes=cast(bytes, values["receipt"]),
                    expected_receipt_sha256=cast(str, values["receipt_sha256"]),
                    expected_source_snapshot_manifest_sha256=cast(str, values["manifest_sha256"]),
                    expected_source_snapshot_tree_sha256=cast(str, values["tree_sha256"]),
                )
        finally:
            os.close(descriptor)


def test_retained_capability_is_read_only_sealed_pid_bound_and_nonserializable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = _expected(repository)
    with _open_bundle(repository, measured) as retained:
        descriptor = retained.subprocess_pass_fds[0]
        assert os.get_inheritable(descriptor) is False
        assert os.fstat(descriptor).st_nlink == 0
        assert stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o400
        with pytest.raises(OSError):
            os.write(descriptor, b"x")
        with pytest.raises(TypeError):
            pickle.dumps(retained)
        owner_pid = retained.owner_pid
        monkeypatch.setattr(bundle.os, "getpid", lambda: owner_pid + 1)
        with pytest.raises(bundle.ForagerMatchedV3LocalSourceBundleError, match="PID"):
            retained.reverify()
        assert retained.closed is True
        with pytest.raises(OSError) as closed:
            os.fstat(descriptor)
        assert closed.value.errno == errno.EBADF


def test_retained_capability_metadata_failure_closes_owned_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = _expected(repository)
    with _open_bundle(repository, measured) as retained:
        descriptor = retained.subprocess_pass_fds[0]
        monkeypatch.setattr(
            bundle.fcntl,
            "fcntl",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("synthetic fcntl failure")),
        )
        with pytest.raises(bundle.ForagerMatchedV3LocalSourceBundleError, match="inaccessible"):
            retained.reverify()
        assert retained.closed is True
        with pytest.raises(OSError) as closed:
            os.fstat(descriptor)
        assert closed.value.errno == errno.EBADF


def test_descriptor_substitution_invalidates_without_closing_foreign_fd(tmp_path: Path) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = _expected(repository)
    with _open_bundle(repository, measured) as retained:
        descriptor = retained.subprocess_pass_fds[0]
        foreign = os.memfd_create("foreign", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            os.write(foreign, b"foreign")
            os.close(descriptor)
            os.dup2(foreign, descriptor, inheritable=False)
            with pytest.raises(bundle.ForagerMatchedV3LocalSourceBundleError, match="identity"):
                retained.reverify()
            assert os.pread(descriptor, 7, 0) == b"foreign"
        finally:
            os.close(descriptor)
            os.close(foreign)


def test_context_cleanup_does_not_close_substituted_foreign_descriptor(tmp_path: Path) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = _expected(repository)
    descriptor = -1
    foreign = os.memfd_create("foreign-context-cleanup", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        os.write(foreign, b"foreign")
        with _open_bundle(repository, measured) as retained:
            descriptor = retained.subprocess_pass_fds[0]
            os.close(descriptor)
            os.dup2(foreign, descriptor, inheritable=False)
        assert os.pread(descriptor, 7, 0) == b"foreign"
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(foreign)


def test_context_cleanup_preserves_caller_failure_when_retained_close_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _payloads = _make_repository(tmp_path)
    measured = _expected(repository)
    real_close = bundle.os.close
    real_fstat = bundle.os.fstat
    descriptor = -1
    primary = _InjectedAbort("injected caller failure")
    cleanup_failure = _InjectedAbort("injected retained close failure")

    def close_then_abort(closing: int) -> None:
        real_close(closing)
        if closing == descriptor:
            raise cleanup_failure

    monkeypatch.setattr(bundle.os, "close", close_then_abort)
    with pytest.raises(_InjectedAbort) as caught:
        with _open_bundle(repository, measured) as retained:
            descriptor = retained.subprocess_pass_fds[0]
            raise primary

    assert caught.value is primary
    assert any("retained source bundle" in note for note in getattr(primary, "__notes__", []))
    with pytest.raises(OSError) as closed:
        real_fstat(descriptor)
    assert closed.value.errno == errno.EBADF


def test_public_api_requires_all_caller_pins_and_exposes_no_authority_aliases() -> None:
    signature = inspect.signature(bundle.retain_matched_v3_local_source_bundle)
    assert all(
        parameter.default is inspect.Parameter.empty for parameter in signature.parameters.values()
    )
    assert not hasattr(bundle, "DEFAULT_REPOSITORY_ROOT")
    assert not hasattr(bundle, "CURRENT_SOURCE_BUNDLE")
    assert not hasattr(bundle.RetainedMatchedV3LocalSourceBundle, "extract")
    assert not hasattr(bundle.RetainedMatchedV3LocalSourceBundle, "publish")
    assert not hasattr(bundle.RetainedMatchedV3LocalSourceBundle, "execute")
