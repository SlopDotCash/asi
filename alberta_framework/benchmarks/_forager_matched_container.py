#!/usr/bin/env python3
"""Container-only launcher and scorer bridge for matched Forager runs.

This file intentionally depends only on the Python standard library.  The host
executor mounts these exact bytes into the qualified OCI image.  ``run`` emits
one USTAR stream and no other stdout.  ``score`` extracts that stream inside
the networkless container and delegates reward-array access to the frozen
scorer; reward arrays are never exposed to the host process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

CONTRACT = "alberta.forager_matched_container.v1"
OUTPUT_ROOT = Path("/run/alberta/output")
EXTRACT_ROOT = Path("/run/alberta/scoring-input")
EXECUTION_SOURCE_ROOT = Path("/run/alberta/source")
EXECUTION_CONFIG = Path("/run/alberta/configuration.json")
SOURCE_TREE_HASH_SCHEME = "canonical-entry-json+mode+size+bytes-v1"
# Verify-then-exec mini-program passed to ``python -c``.  It closes the
# check-to-use (TOCTOU) window that hashing a script and then letting Python
# re-open it by path would leave: the file is read once through an
# O_NOFOLLOW descriptor, fstat identity before/after the read rejects a
# concurrent rewrite, the SHA-256 is checked on the bytes actually read, and
# exec runs those same in-memory bytes — the path is never opened again.
VERIFIED_SCRIPT_LAUNCHER = (
    "import hashlib,os,stat,sys\n"
    "path,expected,*args=sys.argv[1:]\n"
    "fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))\n"
    "before=os.fstat(fd)\n"
    "assert stat.S_ISREG(before.st_mode) and before.st_nlink==1\n"
    "chunks=[]\n"
    "while True:\n"
    " chunk=os.read(fd,1048576)\n"
    " if not chunk: break\n"
    " chunks.append(chunk)\n"
    "after=os.fstat(fd);os.close(fd);raw=b''.join(chunks)\n"
    "assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)=="
    "(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)\n"
    "assert hashlib.sha256(raw).hexdigest()==expected\n"
    "sys.argv=[path,*args]\n"
    "scope={'__name__':'__main__','__file__':path,'__package__':None,'__cached__':None}\n"
    "exec(compile(raw,path,'exec'),scope,scope)"
)
# Defensive resource caps for untrusted-shaped inputs.  The payload cap is
# deliberately below the raw-archive cap: 480 MiB of payload plus worst-case
# USTAR overhead (per-member header block and padding for MAX_MEMBERS entries,
# end-of-archive blocks, record padding) must still fit inside the 512 MiB
# archive bound — checked at import time next to
# ``_maximum_ustar_stream_size`` below.
MAX_MEMBERS = 20_000
MAX_OUTPUT_DIRECTORIES = 20_000
MAX_OUTPUT_DEPTH = 64
MAX_MEMBER_BYTES = 256 * 1024**2
MAX_SOURCE_BYTES = 512 * 1024**2
MAX_OUTPUT_PAYLOAD_BYTES = 480 * 1024**2
MAX_RAW_ARCHIVE_BYTES = 512 * 1024**2
USTAR_BLOCK_BYTES = 512
USTAR_RECORD_BYTES = 10 * 1024
USTAR_END_BYTES = 2 * USTAR_BLOCK_BYTES


class ContainerError(RuntimeError):
    """The in-container invocation or artifact stream violated its contract."""


def _require_exact_str(name: object, value: object) -> str:
    if type(name) is not str:
        raise ContainerError("name must be an exact string")
    if type(value) is not str:
        raise ContainerError(f"{name} must be an exact string")
    return value


@dataclass(frozen=True)
class _OutputMember:
    name: str
    descriptor: int
    size: int
    identity: tuple[int, ...]


def _absolute(value: str, *, label: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or "\x00" in value
    ):
        raise ContainerError(f"{label} must be a canonical absolute path")
    return path


def _under(path: Path, root: Path, *, label: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ContainerError(f"{label} escapes {root}") from exc
    return path


def _relative(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != value
        or "." in path.parts
        or ".." in path.parts
        or "\x00" in value
    ):
        raise ContainerError(f"{label} must be a canonical relative path")
    return path


def _runpy_command(
    *,
    python: Path,
    python_import_root: Path,
    entrypoint: Path,
    arguments: list[str],
) -> list[str]:
    wrapper = (
        "import resource,runpy,sys;"
        "_,hard=resource.getrlimit(resource.RLIMIT_FSIZE);"
        f"limit={MAX_MEMBER_BYTES} if hard==resource.RLIM_INFINITY "
        f"else min({MAX_MEMBER_BYTES},hard);"
        "assert limit>0;"
        "resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit));"
        "trusted,entry,*args=sys.argv[1:];"
        "sys.path.insert(0,trusted);"
        "sys.argv=[entry,*args];"
        "runpy.run_path(entry,run_name='__main__')"
    )
    return [
        python.as_posix(),
        "-I",
        "-B",
        "-c",
        wrapper,
        python_import_root.as_posix(),
        entrypoint.as_posix(),
        *arguments,
    ]


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _output_members(stdout_log: Path, stderr_log: Path) -> list[_OutputMember]:
    members: list[_OutputMember] = []
    identities: set[tuple[int, int]] = set()
    total = 0
    entry_count = 0
    directory_count = 0
    seen_paths: set[str] = set()
    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        root_before = os.lstat(OUTPUT_ROOT)
        root_descriptor = os.open(OUTPUT_ROOT, root_flags)
    except OSError as exc:
        raise ContainerError("cannot safely open workload output root") from exc
    try:
        root_opened = os.fstat(root_descriptor)
    except BaseException:
        os.close(root_descriptor)
        raise
    if (
        not stat.S_ISDIR(root_before.st_mode)
        or not stat.S_ISDIR(root_opened.st_mode)
        or _stat_identity(root_before) != _stat_identity(root_opened)
    ):
        os.close(root_descriptor)
        raise ContainerError("workload output root changed before traversal")

    def walk(directory_descriptor: int, prefix: str, depth: int) -> None:
        nonlocal directory_count, entry_count, total
        if depth > MAX_OUTPUT_DEPTH:
            raise ContainerError("workload output exceeds its directory-depth bound")
        names: list[str] = []
        try:
            iterator = os.scandir(directory_descriptor)
        except OSError as exc:
            raise ContainerError("cannot enumerate workload output") from exc
        with iterator:
            for entry in iterator:
                entry_count += 1
                if entry_count > MAX_MEMBERS + MAX_OUTPUT_DIRECTORIES:
                    raise ContainerError("workload output exceeds its entry-count bound")
                names.append(entry.name)
        for name in sorted(names):
            if not name or name in {".", ".."} or "/" in name or "\x00" in name:
                raise ContainerError("workload output contains an unsafe name")
            relative = f"{prefix}/{name}" if prefix else name
            try:
                metadata = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ContainerError("cannot inspect workload output entry") from exc
            if stat.S_ISDIR(metadata.st_mode):
                directory_count += 1
                if directory_count > MAX_OUTPUT_DIRECTORIES:
                    raise ContainerError(
                        "workload output exceeds its directory-count bound"
                    )
                try:
                    child_descriptor = os.open(
                        name,
                        root_flags,
                        dir_fd=directory_descriptor,
                    )
                except OSError as exc:
                    raise ContainerError(
                        "cannot safely open workload output directory"
                    ) from exc
                try:
                    opened = os.fstat(child_descriptor)
                    if _stat_identity(opened) != _stat_identity(metadata):
                        raise ContainerError(
                            "workload output directory changed before traversal"
                        )
                    walk(child_descriptor, relative, depth + 1)
                    after = os.fstat(child_descriptor)
                    current = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        _stat_identity(opened) != _stat_identity(after)
                        or _stat_identity(after) != _stat_identity(current)
                    ):
                        raise ContainerError(
                            "workload output directory changed during traversal"
                        )
                finally:
                    os.close(child_descriptor)
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ContainerError(
                    "workload output contains a symlink, hardlink, or non-regular file"
                )
            if len(members) >= MAX_MEMBERS:
                raise ContainerError("workload output member count is outside its bound")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                descriptor = os.open(name, flags, dir_fd=directory_descriptor)
            except OSError as exc:
                raise ContainerError("cannot safely open workload output member") from exc
            try:
                opened = os.fstat(descriptor)
            except BaseException:
                os.close(descriptor)
                raise
            if _stat_identity(opened) != _stat_identity(metadata):
                os.close(descriptor)
                raise ContainerError("workload output changed before it was opened")
            inode = (opened.st_dev, opened.st_ino)
            if inode in identities:
                os.close(descriptor)
                raise ContainerError("workload output contains an inode alias")
            identities.add(inode)
            if opened.st_size > MAX_MEMBER_BYTES:
                os.close(descriptor)
                raise ContainerError("workload output member exceeds its size bound")
            total += opened.st_size
            if total > MAX_OUTPUT_PAYLOAD_BYTES:
                os.close(descriptor)
                raise ContainerError("workload output exceeds its total size bound")
            seen_paths.add(relative)
            members.append(
                _OutputMember(
                    relative,
                    descriptor,
                    opened.st_size,
                    _stat_identity(opened),
                )
            )

    try:
        walk(root_descriptor, "", 0)
        root_after = os.fstat(root_descriptor)
        root_current = os.lstat(OUTPUT_ROOT)
        if (
            _stat_identity(root_opened) != _stat_identity(root_after)
            or _stat_identity(root_after) != _stat_identity(root_current)
        ):
            raise ContainerError("workload output root changed during traversal")
        expected_logs = {
            stdout_log.relative_to(OUTPUT_ROOT).as_posix(),
            stderr_log.relative_to(OUTPUT_ROOT).as_posix(),
        }
        if not expected_logs <= seen_paths:
            raise ContainerError("workload output is missing captured process logs")
        if not 2 <= len(members) <= MAX_MEMBERS:
            raise ContainerError("workload output member count is outside its bound")
        return members
    except BaseException:
        for member in members:
            os.close(member.descriptor)
        raise
    finally:
        os.close(root_descriptor)


def _round_up(value: int, block_size: int) -> int:
    return ((value + block_size - 1) // block_size) * block_size


def _ustar_stream_size(member_sizes: list[int]) -> int:
    """Return the exact byte count emitted by tarfile's default stream buffer."""
    if any(size < 0 for size in member_sizes):
        raise ContainerError("USTAR member size must be nonnegative")
    unbuffered_size = USTAR_END_BYTES + sum(
        USTAR_BLOCK_BYTES + _round_up(size, USTAR_BLOCK_BYTES)
        for size in member_sizes
    )
    return _round_up(unbuffered_size, USTAR_RECORD_BYTES)


def _maximum_ustar_stream_size(*, payload_bytes: int, member_count: int) -> int:
    """Bound a USTAR stream including headers, payload padding, and record padding."""
    if payload_bytes < 0 or member_count < 0:
        raise ContainerError("USTAR bound inputs must be nonnegative")
    unbuffered_size = (
        USTAR_END_BYTES
        + payload_bytes
        + member_count * (USTAR_BLOCK_BYTES + USTAR_BLOCK_BYTES - 1)
    )
    return _round_up(unbuffered_size, USTAR_RECORD_BYTES)


if _maximum_ustar_stream_size(
    payload_bytes=MAX_OUTPUT_PAYLOAD_BYTES,
    member_count=MAX_MEMBERS,
) > MAX_RAW_ARCHIVE_BYTES:
    raise RuntimeError("output payload/member caps cannot fit in the raw USTAR cap")


def _write_ustar(members: list[_OutputMember]) -> None:
    stream_size = _ustar_stream_size([member.size for member in members])
    if stream_size > MAX_RAW_ARCHIVE_BYTES:
        for member in members:
            os.close(member.descriptor)
        raise ContainerError("USTAR stream exceeds the host/scorer raw archive bound")
    try:
        with tarfile.open(
            fileobj=sys.stdout.buffer, mode="w|", format=tarfile.USTAR_FORMAT
        ) as archive:
            for member in members:
                if _stat_identity(os.fstat(member.descriptor)) != member.identity:
                    raise ContainerError("workload output changed before archive streaming")
                info = tarfile.TarInfo(member.name)
                info.type = tarfile.REGTYPE
                info.mode = 0o600
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.size = member.size
                os.lseek(member.descriptor, 0, os.SEEK_SET)
                with os.fdopen(os.dup(member.descriptor), "rb", closefd=True) as stream:
                    archive.addfile(info, stream)
                if _stat_identity(os.fstat(member.descriptor)) != member.identity:
                    raise ContainerError("workload output changed during archive streaming")
    finally:
        for member in members:
            os.close(member.descriptor)


def _validate_workload_output(
    *,
    result_root: str,
    seed: int,
    stdout_log: Path,
    stderr_log: Path,
) -> None:
    relative_root = _relative(result_root, label="result root")
    expected_archive = OUTPUT_ROOT.joinpath(
        *relative_root.parts,
        "data",
        f"{seed}.npz",
    )
    for path, label, allow_empty in (
        (stdout_log, "captured workload stdout", True),
        (stderr_log, "captured workload stderr", True),
        (expected_archive, "expected reward archive", False),
    ):
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ContainerError(f"{label} is missing") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > MAX_MEMBER_BYTES
            or (not allow_empty and metadata.st_size == 0)
        ):
            raise ContainerError(f"{label} violates the regular-file contract")


def _copy_stable_file(source: Path, destination: Path, expected_sha256: str) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ContainerError("configuration is not a single-link regular file")
        digest = hashlib.sha256()
        output = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(output, view)
                    if written <= 0:
                        raise ContainerError("configuration snapshot write made no progress")
                    view = view[written:]
        finally:
            os.close(output)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise ContainerError("configuration changed while being snapshotted")
        if digest.hexdigest() != expected_sha256:
            raise ContainerError("configuration differs from its receipt-bound digest")
    finally:
        os.close(descriptor)


def _snapshot_source(source_root: Path, expected_sha256: str) -> None:
    if EXECUTION_SOURCE_ROOT.exists():
        raise ContainerError("execution source snapshot root unexpectedly exists")
    EXECUTION_SOURCE_ROOT.mkdir(mode=0o700, parents=True)
    entries: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    total = 0
    count = 0
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    root_fd = os.open(source_root, directory_flags)

    def walk(source_fd: int, destination: Path, prefix: str) -> None:
        nonlocal count, total
        for name in sorted(os.listdir(source_fd), key=lambda value: value.encode("utf-8")):
            if not name or name in {".", ".."} or "/" in name or "\x00" in name:
                raise ContainerError("source snapshot contains an unsafe name")
            relative = f"{prefix}/{name}" if prefix else name
            metadata = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
            target = destination / name
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = os.open(name, directory_flags, dir_fd=source_fd)
                try:
                    opened = os.fstat(child_fd)
                    if _stat_identity(metadata) != _stat_identity(opened):
                        raise ContainerError("source directory changed before snapshotting")
                    target.mkdir(mode=0o700)
                    entries.append(
                        {"mode": 0o775, "path": relative, "type": "directory"}
                    )
                    walk(child_fd, target, relative)
                    current = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
                    if _stat_identity(opened) != _stat_identity(os.fstat(child_fd)) or (
                        _stat_identity(opened) != _stat_identity(current)
                    ):
                        raise ContainerError("source directory changed while snapshotting")
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ContainerError("source snapshot contains a link or special file")
            inode = (metadata.st_dev, metadata.st_ino)
            if inode in identities:
                raise ContainerError("source snapshot contains an inode alias")
            identities.add(inode)
            count += 1
            total += metadata.st_size
            if count > MAX_MEMBERS or total > MAX_SOURCE_BYTES:
                raise ContainerError("source snapshot exceeds its file or byte bound")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            source_descriptor = os.open(name, flags, dir_fd=source_fd)
            try:
                opened = os.fstat(source_descriptor)
                if _stat_identity(metadata) != _stat_identity(opened):
                    raise ContainerError("source file changed before snapshotting")
                output_descriptor = os.open(
                    target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                    0o700 if stat.S_IMODE(opened.st_mode) & 0o111 else 0o600,
                )
                digest = hashlib.sha256()
                copied = 0
                try:
                    while chunk := os.read(source_descriptor, 1024 * 1024):
                        digest.update(chunk)
                        copied += len(chunk)
                        view = memoryview(chunk)
                        while view:
                            written = os.write(output_descriptor, view)
                            if written <= 0:
                                raise ContainerError("source snapshot write made no progress")
                            view = view[written:]
                finally:
                    os.close(output_descriptor)
                current = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
                if (
                    copied != opened.st_size
                    or _stat_identity(opened) != _stat_identity(os.fstat(source_descriptor))
                    or _stat_identity(opened) != _stat_identity(current)
                ):
                    raise ContainerError("source file changed while snapshotting")
                entries.append(
                    {
                        "mode": 0o775 if stat.S_IMODE(opened.st_mode) & 0o111 else 0o664,
                        "path": relative,
                        "sha256": digest.hexdigest(),
                        "size": copied,
                        "type": "file",
                    }
                )
            finally:
                os.close(source_descriptor)

    try:
        walk(root_fd, EXECUTION_SOURCE_ROOT, "")
    finally:
        os.close(root_fd)
    entries.sort(
        key=lambda item: (
            str(item["path"]) + ("/" if item["type"] == "directory" else "")
        ).encode("utf-8")
    )
    payload = {"entries": entries, "hash_scheme": SOURCE_TREE_HASH_SCHEME}
    raw = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ContainerError("source snapshot differs from its protocol-bound inventory")


def _run(args: argparse.Namespace) -> None:
    source_root = _absolute(args.source_root, label="source root")
    entrypoint = _under(
        _absolute(args.entrypoint, label="entrypoint"),
        source_root,
        label="entrypoint",
    )
    config = _absolute(args.config, label="configuration")
    python_import_root = _under(
        _absolute(args.python_import_root, label="Python import root"),
        source_root,
        label="Python import root",
    )
    python = _absolute(args.python, label="Python executable")
    if not source_root.is_dir() or source_root.is_symlink():
        raise ContainerError("source root must be a non-symlink directory")
    if not entrypoint.is_file() or entrypoint.is_symlink():
        raise ContainerError("entrypoint must be a non-symlink regular file")
    if not config.is_file() or config.is_symlink():
        raise ContainerError("configuration must be a non-symlink regular file")
    if args.seed < 0 or args.horizon < 1:
        raise ContainerError("seed and horizon are outside their domains")
    _snapshot_source(source_root, args.source_inventory_sha256)
    _copy_stable_file(config, EXECUTION_CONFIG, args.configuration_sha256)
    relative_entrypoint = entrypoint.relative_to(source_root)
    relative_import_root = python_import_root.relative_to(source_root)
    entrypoint = EXECUTION_SOURCE_ROOT / relative_entrypoint
    python_import_root = EXECUTION_SOURCE_ROOT / relative_import_root
    config = EXECUTION_CONFIG
    if OUTPUT_ROOT.exists():
        raise ContainerError("output root unexpectedly exists")
    results = OUTPUT_ROOT / "results"
    checkpoints = OUTPUT_ROOT / "checkpoints"
    OUTPUT_ROOT.mkdir(mode=0o700, parents=True)
    results.mkdir(mode=0o700)
    checkpoints.mkdir(mode=0o700)
    stdout_log = OUTPUT_ROOT / "stdout.log"
    stderr_log = OUTPUT_ROOT / "stderr.log"
    stdout_log.touch(mode=0o600)
    stderr_log.touch(mode=0o600)
    if not python_import_root.is_dir() or python_import_root.is_symlink():
        raise ContainerError("Python import root must be a non-symlink directory")
    if args.invocation_style == "official_foragax_continuing_main_v4":
        workload_arguments = [
            "-e",
            config.as_posix(),
            "-i",
            str(args.seed),
            "--save_path",
            results.as_posix(),
            "--checkpoint_path",
            checkpoints.as_posix(),
            "--max_steps",
            str(args.horizon),
        ]
    elif args.invocation_style == "official_foragax_ppo_frozen_updates_v1":
        # rtu_ppo.py interprets --max_steps as the number of PPO updates, not
        # environment transitions.  Qualification separately proves that the
        # frozen configuration's rollout_steps*num_updates equals horizon.
        workload_arguments = [
            "-e",
            config.as_posix(),
            "-i",
            str(args.seed),
            "--save_path",
            results.as_posix(),
            "--checkpoint_path",
            checkpoints.as_posix(),
        ]
    elif args.invocation_style == "alberta_single_seed_v1":
        workload_arguments = [
            "--configuration",
            config.as_posix(),
            "--seed",
            str(args.seed),
            "--horizon",
            str(args.horizon),
            "--output-root",
            results.as_posix(),
        ]
    else:
        raise ContainerError("invocation style is unsupported")
    command = _runpy_command(
        python=python,
        python_import_root=python_import_root,
        entrypoint=entrypoint,
        arguments=workload_arguments,
    )
    with stdout_log.open("wb") as stdout, stderr_log.open("wb") as stderr:
        completed = subprocess.run(
            command,
            cwd=EXECUTION_SOURCE_ROOT,
            env=dict(os.environ),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        raise ContainerError(f"workload exited with status {completed.returncode}")
    _validate_workload_output(
        result_root=args.result_root,
        seed=args.seed,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )
    _write_ustar(_output_members(stdout_log, stderr_log))


def _safe_extract(raw_archive: Path, expected_sha256: str) -> None:
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ContainerError("expected raw archive SHA-256 is invalid")
    if EXTRACT_ROOT.exists():
        raise ContainerError("scoring extraction root unexpectedly exists")
    EXTRACT_ROOT.mkdir(mode=0o700, parents=True)
    names: set[str] = set()
    total = 0
    count = 0
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(raw_archive, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > MAX_RAW_ARCHIVE_BYTES
        ):
            raise ContainerError("raw archive violates the bounded regular-file contract")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ContainerError("raw archive differs from the host-bound digest")
        os.lseek(descriptor, 0, os.SEEK_SET)
        stream = os.fdopen(os.dup(descriptor), "rb", closefd=True)
        archive_context = tarfile.open(fileobj=stream, mode="r:")
        with stream, archive_context as archive:
            for member in archive:
                count += 1
                relative = _relative(member.name, label="USTAR member")
                if member.name in names:
                    raise ContainerError("USTAR contains a duplicate member")
                names.add(member.name)
                if not member.isreg() or member.size < 0 or member.size > MAX_MEMBER_BYTES:
                    raise ContainerError("USTAR contains a non-regular or oversized member")
                total += member.size
                if count > MAX_MEMBERS or total > MAX_OUTPUT_PAYLOAD_BYTES:
                    raise ContainerError("USTAR exceeds its member or byte bound")
                destination = EXTRACT_ROOT.joinpath(*relative.parts)
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ContainerError("USTAR regular member has no readable payload")
                with destination.open("xb") as target:
                    remaining = member.size
                    while remaining:
                        chunk = source.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise ContainerError("USTAR member ended before its declared size")
                        target.write(chunk)
                        remaining -= len(chunk)
                    if source.read(1):
                        raise ContainerError("USTAR member exceeds its declared size")
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise ContainerError("raw archive changed during scoring extraction")
    finally:
        os.close(descriptor)
    if count < 3:
        raise ContainerError("USTAR omits result data or captured logs")


def _strict_json(raw: bytes) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContainerError("scorer JSON contains duplicate keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        _require_exact_str("value", value)
        raise ContainerError("scorer JSON contains non-finite number")

    def parse_float(value: str) -> float:
        host_value = _require_exact_str("value", value)
        parsed = float(host_value)
        if not math.isfinite(parsed):
            raise ContainerError("scorer JSON contains non-finite number")
        return parsed

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
            parse_float=parse_float,
        )
    except ContainerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContainerError("scorer did not return strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ContainerError("scorer result must be a JSON object")
    return value


def _score(args: argparse.Namespace) -> None:
    raw_archive = _absolute(args.raw_archive, label="raw archive")
    scorer = _absolute(args.scorer, label="scorer")
    python = _absolute(args.python, label="Python executable")
    if raw_archive.is_symlink() or not raw_archive.is_file():
        raise ContainerError("raw archive must be a non-symlink regular file")
    if scorer.is_symlink() or not scorer.is_file():
        raise ContainerError("scorer must be a non-symlink regular file")
    _relative(args.result_root, label="result root")
    if args.seed < 0 or args.horizon < 1:
        raise ContainerError("seed and horizon are outside their domains")
    _safe_extract(raw_archive, args.raw_archive_sha256)
    command = [
        python.as_posix(),
        "-I",
        "-B",
        "-c",
        VERIFIED_SCRIPT_LAUNCHER,
        scorer.as_posix(),
        args.scorer_sha256,
        "--payload-root",
        EXTRACT_ROOT.as_posix(),
        "--result-root",
        args.result_root,
        "--horizon",
        str(args.horizon),
        "--seed",
        str(args.seed),
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0 or completed.stderr:
        raise ContainerError(f"qualified scorer failed with status {completed.returncode}")
    payload = _strict_json(completed.stdout)
    sys.stdout.write(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--contract", required=True)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    run = subparsers.add_parser("run", allow_abbrev=False)
    run.add_argument("--python", required=True)
    run.add_argument("--source-root", required=True)
    run.add_argument("--entrypoint", required=True)
    run.add_argument("--python-import-root", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--source-inventory-sha256", required=True)
    run.add_argument("--configuration-sha256", required=True)
    run.add_argument("--invocation-style", required=True)
    run.add_argument("--result-root", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--horizon", type=int, required=True)
    score = subparsers.add_parser("score", allow_abbrev=False)
    score.add_argument("--python", required=True)
    score.add_argument("--raw-archive", required=True)
    score.add_argument("--raw-archive-sha256", required=True)
    score.add_argument("--scorer", required=True)
    score.add_argument("--scorer-sha256", required=True)
    score.add_argument("--result-root", required=True)
    score.add_argument("--seed", type=int, required=True)
    score.add_argument("--horizon", type=int, required=True)
    return parser


def main() -> int:
    try:
        args = _parser().parse_args()
        if args.contract != CONTRACT:
            raise ContainerError("container contract identity is unsupported")
        if args.operation == "run":
            _run(args)
        else:
            _score(args)
        return 0
    except (ContainerError, OSError, subprocess.SubprocessError, tarfile.TarError) as exc:
        sys.stderr.write(f"matched Forager container: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
