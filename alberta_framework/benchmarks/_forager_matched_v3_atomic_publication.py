"""Private atomic persistence for exact, flat matched-v3 byte inventories.

This standard-library-only module is a content primitive.  It does not interpret an
address, parse publisher manifests, reconstruct a live capability, select a default
root, or grant execution, qualification, ingestion, evidence, or promotion authority.

Publication is Linux-local and fail closed.  Files are written through a held staging
directory descriptor, replayed, fsynced, moved with ``renameat2(RENAME_NOREPLACE)``,
fsynced through the held parent descriptor, and replayed again.  Loading likewise opens
the absolute parent first and performs every child operation relative to held directory
descriptors.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, cast

ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.atomic_publication_descriptor.v1"
)
ATOMIC_PUBLICATION_STATUS: Final = "private_content_only_primitive"

DIRECTORY_MODE: Final = 0o700
FILE_MODE: Final = 0o600
MAX_FILES: Final = 64
MAX_FILE_BYTES: Final = 512 * 1024 * 1024
MAX_TOTAL_BYTES: Final = 1024 * 1024 * 1024

_STAGING_ATTEMPTS: Final = 64
_COPY_CHUNK_BYTES: Final = 1024 * 1024
_ADDRESS_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_NAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_STAGING_PREFIX: Final = ".forager-matched-v3-atomic-partial-"
_RENAME_NOREPLACE: Final = 1


class ForagerMatchedV3AtomicPublicationError(ValueError):
    """Exact content, namespace, filesystem, or durability validation failed."""


class ForagerMatchedV3AtomicPublicationCollisionError(
    ForagerMatchedV3AtomicPublicationError
):
    """The requested content address already names an entry."""


class ForagerMatchedV3AtomicPublicationUncertainError(
    ForagerMatchedV3AtomicPublicationError
):
    """Publication committed, or may have committed, before verification failed."""

    def __init__(
        self,
        destination: Path,
        address: str,
        detail: str,
        *,
        committed: Literal[True] | None,
    ) -> None:
        if committed is not True and committed is not None:
            raise TypeError("committed must be exactly True or None")
        self.destination = destination
        self.address = address
        self.committed = committed
        state = "committed" if committed is True else "commit state unknown"
        super().__init__(f"atomic publication {address} at {destination}: {detail}; {state}")


@dataclass(frozen=True, slots=True)
class ExactFileRecord:
    """One caller-carried filename, exact byte length, and exact content digest."""

    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ContentVerifiedFlatPublication:
    """Reloaded structural bytes; this object carries no execution authority."""

    root: Path
    address: str
    records: tuple[ExactFileRecord, ...]
    files: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class _OpenDirectory:
    path: Path
    descriptor: int
    inode_identity: tuple[int, int, int]
    owner: tuple[int, int]


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION,
        "status": ATOMIC_PUBLICATION_STATUS,
        "classification": "private_content_only_non_authorizing_primitive",
        "scope": {
            "layout": "one_sha256_named_directory_with_exact_flat_files",
            "address_semantics": "externally_carried_opaque_sha256_namespace_key",
            "publisher_specific_semantics": False,
            "default_publication_root": False,
            "recursive_deletion": False,
            "network_process_or_workload_execution": False,
        },
        "bounds": {
            "maximum_files": MAX_FILES,
            "maximum_file_bytes": MAX_FILE_BYTES,
            "maximum_total_bytes": MAX_TOTAL_BYTES,
            "zero_length_regular_files_allowed": True,
        },
        "filesystem_contract": {
            "absolute_non_root_parent_required": True,
            "parent_directory_mode": "0700",
            "staging_and_published_directory_mode": "0700",
            "file_mode": "0600",
            "held_directory_descriptors": True,
            "single_link_regular_files": True,
            "exclusive_move": "renameat2_RENAME_NOREPLACE_no_fallback",
            "file_staging_and_parent_fsync": True,
            "strict_replay_before_and_after_move": True,
        },
        "claims": {
            "authority_granted": False,
            "campaign_ingestion_authorized": False,
            "evidence_authority": False,
            "execution_authorized": False,
            "qualification_authority": False,
            "scientific_promotion_allowed": False,
        },
    }


_DESCRIPTOR_BYTES: Final = json.dumps(
    _descriptor(),
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("ascii")
ATOMIC_PUBLICATION_DESCRIPTOR_SHA256: Final = (
    "b224fe9fdc438ccab0df5bfd3199e1d264feacbb99147970cc68a9c703b9e98e"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    ATOMIC_PUBLICATION_DESCRIPTOR_SHA256,
):
    raise RuntimeError("private atomic publication descriptor identity drifted")


def atomic_publication_descriptor() -> dict[str, Any]:
    """Return a detached description of this private content-only primitive."""

    return cast(dict[str, Any], json.loads(_DESCRIPTOR_BYTES.decode("ascii")))


def canonical_atomic_publication_descriptor_bytes() -> bytes:
    """Return the exact canonical descriptor bytes."""

    return bytes(_DESCRIPTOR_BYTES)


def _require_address(value: object) -> str:
    if (
        type(value) is not str
        or _ADDRESS_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ForagerMatchedV3AtomicPublicationError(
            "address must be a nonzero lowercase SHA-256 digest"
        )
    return value


def _require_digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _ADDRESS_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ForagerMatchedV3AtomicPublicationError(
            f"{label} must be a nonzero lowercase SHA-256 digest"
        )
    return value


def _validated_records(
    records: object,
) -> tuple[tuple[ExactFileRecord, ...], dict[str, ExactFileRecord]]:
    if type(records) is not tuple:
        raise ForagerMatchedV3AtomicPublicationError(
            "expected file records must be an exact tuple"
        )
    supplied = cast(tuple[object, ...], records)
    if not 1 <= len(supplied) <= MAX_FILES:
        raise ForagerMatchedV3AtomicPublicationError(
            f"expected file count must be in [1, {MAX_FILES}]"
        )
    by_name: dict[str, ExactFileRecord] = {}
    total = 0
    for item in supplied:
        if type(item) is not ExactFileRecord:
            raise ForagerMatchedV3AtomicPublicationError(
                "each expected file record must be an exact ExactFileRecord"
            )
        record = item
        if (
            type(record.name) is not str
            or _SAFE_NAME_RE.fullmatch(record.name) is None
            or record.name in {".", ".."}
            or os.path.basename(record.name) != record.name
        ):
            raise ForagerMatchedV3AtomicPublicationError(
                "expected file name is not a safe portable flat name"
            )
        if record.name in by_name:
            raise ForagerMatchedV3AtomicPublicationError(
                f"duplicate expected file name {record.name!r}"
            )
        if (
            type(record.size_bytes) is not int
            or not 0 <= record.size_bytes <= MAX_FILE_BYTES
        ):
            raise ForagerMatchedV3AtomicPublicationError(
                f"expected file {record.name!r} size is outside the byte bound"
            )
        _require_digest(record.sha256, f"expected file {record.name!r} digest")
        total += record.size_bytes
        if total > MAX_TOTAL_BYTES:
            raise ForagerMatchedV3AtomicPublicationError(
                "expected file inventory exceeds the total byte bound"
            )
        by_name[record.name] = record
    normalized = tuple(by_name[name] for name in sorted(by_name, key=os.fsencode))
    return normalized, by_name


def _validated_payloads(
    payloads: object,
    expected: Mapping[str, ExactFileRecord],
) -> dict[str, bytes]:
    if type(payloads) is not dict:
        raise ForagerMatchedV3AtomicPublicationError(
            "publication payloads must be an exact dictionary"
        )
    supplied = cast(dict[object, object], payloads)
    if set(supplied) != set(expected):
        actual_names = {name for name in supplied if type(name) is str}
        raise ForagerMatchedV3AtomicPublicationError(
            "publication payload inventory differs; "
            f"missing={sorted(set(expected) - actual_names)!r}, "
            f"extra={sorted(actual_names - set(expected))!r}"
        )
    result: dict[str, bytes] = {}
    for name, record in expected.items():
        raw = supplied[name]
        if type(raw) is not bytes:
            raise ForagerMatchedV3AtomicPublicationError(
                f"publication payload {name!r} must be exact bytes"
            )
        payload = raw
        if len(payload) != record.size_bytes or not hmac.compare_digest(
            hashlib.sha256(payload).hexdigest(), record.sha256
        ):
            raise ForagerMatchedV3AtomicPublicationError(
                f"publication payload {name!r} differs from its expected record"
            )
        result[name] = payload
    return result


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _inode_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)


def _require_linux_open_flags() -> None:
    if os.name != "posix" or any(
        not hasattr(os, name) for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    ):
        raise ForagerMatchedV3AtomicPublicationError(
            "Linux O_CLOEXEC/O_DIRECTORY/O_NOFOLLOW support is required"
        )


def _directory_open_flags() -> int:
    _require_linux_open_flags()
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _file_read_flags() -> int:
    _require_linux_open_flags()
    return (
        os.O_RDONLY
        | os.O_CLOEXEC
        | os.O_NOFOLLOW
        | getattr(os, "O_NONBLOCK", 0)
    )


def _close_no_raise(descriptor: int) -> None:
    """Close once; cleanup failure must not mask a primary result or exception."""

    try:
        os.close(descriptor)
    except OSError:
        pass


def _open_parent(path: Path) -> _OpenDirectory:
    if not isinstance(path, Path):
        raise ForagerMatchedV3AtomicPublicationError(
            "publication parent must be a pathlib Path"
        )
    if not path.is_absolute() or path == Path("/"):
        raise ForagerMatchedV3AtomicPublicationError(
            "publication parent must be an absolute non-root path"
        )
    try:
        canonical = path.resolve(strict=True)
        path_metadata = path.lstat()
    except (OSError, RuntimeError) as exc:
        raise ForagerMatchedV3AtomicPublicationError(
            "cannot resolve publication parent"
        ) from exc
    if canonical != path or path.is_symlink():
        raise ForagerMatchedV3AtomicPublicationError(
            "publication parent must be a canonical non-symlink path"
        )
    try:
        descriptor = os.open(path, _directory_open_flags())
    except OSError as exc:
        raise ForagerMatchedV3AtomicPublicationError(
            "cannot safely open publication parent"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(path)
        identity = _inode_identity(opened)
        if (
            not stat.S_ISDIR(path_metadata.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _inode_identity(path_metadata) != identity
            or _inode_identity(current) != identity
            or stat.S_IMODE(opened.st_mode) != DIRECTORY_MODE
            or opened.st_uid != os.geteuid()
        ):
            raise ForagerMatchedV3AtomicPublicationError(
                "publication parent changed or has unsafe ownership/mode"
            )
        return _OpenDirectory(
            path=path,
            descriptor=descriptor,
            inode_identity=identity,
            owner=(opened.st_uid, opened.st_gid),
        )
    except ForagerMatchedV3AtomicPublicationError:
        _close_no_raise(descriptor)
        raise
    except OSError as exc:
        _close_no_raise(descriptor)
        raise ForagerMatchedV3AtomicPublicationError(
            "cannot verify publication parent"
        ) from exc
    except BaseException:
        _close_no_raise(descriptor)
        raise


def _assert_parent_path(parent: _OpenDirectory) -> None:
    try:
        opened = os.fstat(parent.descriptor)
        current = os.lstat(parent.path)
    except OSError as exc:
        raise ForagerMatchedV3AtomicPublicationError(
            "publication parent is no longer reachable"
        ) from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or parent.path.is_symlink()
        or _inode_identity(opened) != parent.inode_identity
        or _inode_identity(current) != parent.inode_identity
        or stat.S_IMODE(opened.st_mode) != DIRECTORY_MODE
        or (opened.st_uid, opened.st_gid) != parent.owner
    ):
        raise ForagerMatchedV3AtomicPublicationError(
            "publication parent no longer names the opened safe inode"
        )


def _open_directory_at(
    parent: _OpenDirectory,
    name: str,
    *,
    label: str,
) -> _OpenDirectory:
    try:
        path_metadata = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent.descriptor)
    except OSError as exc:
        raise ForagerMatchedV3AtomicPublicationError(
            f"cannot safely open {label}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        identity = _inode_identity(opened)
        if (
            not stat.S_ISDIR(path_metadata.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _inode_identity(path_metadata) != identity
            or _inode_identity(current) != identity
            or stat.S_IMODE(opened.st_mode) != DIRECTORY_MODE
            or (opened.st_uid, opened.st_gid) != parent.owner
        ):
            raise ForagerMatchedV3AtomicPublicationError(
                f"{label} changed or has unsafe ownership/mode"
            )
        return _OpenDirectory(
            path=parent.path / name,
            descriptor=descriptor,
            inode_identity=identity,
            owner=parent.owner,
        )
    except ForagerMatchedV3AtomicPublicationError:
        _close_no_raise(descriptor)
        raise
    except OSError as exc:
        _close_no_raise(descriptor)
        raise ForagerMatchedV3AtomicPublicationError(
            f"cannot verify {label}"
        ) from exc
    except BaseException:
        _close_no_raise(descriptor)
        raise


def _entry_matches_open_directory(
    parent: _OpenDirectory,
    name: str,
    opened: _OpenDirectory,
) -> bool:
    try:
        entry = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        descriptor = os.fstat(opened.descriptor)
    except OSError:
        return False
    return (
        stat.S_ISDIR(entry.st_mode)
        and _inode_identity(entry) == opened.inode_identity
        and _inode_identity(descriptor) == opened.inode_identity
    )


def _entry_exists(parent: _OpenDirectory, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def _root_inventory(
    root: _OpenDirectory,
    expected: Mapping[str, ExactFileRecord],
) -> dict[str, tuple[int, ...]]:
    root_metadata = os.fstat(root.descriptor)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != DIRECTORY_MODE
        or (root_metadata.st_uid, root_metadata.st_gid) != root.owner
    ):
        raise ForagerMatchedV3AtomicPublicationError(
            "publication root has unsafe ownership, mode, or type"
        )
    names: set[str] = set()
    inventory: dict[str, tuple[int, ...]] = {}
    total = 0
    try:
        iterator = os.scandir(root.descriptor)
    except OSError as exc:
        raise ForagerMatchedV3AtomicPublicationError(
            "cannot enumerate publication root"
        ) from exc
    with iterator:
        for entry in iterator:
            if len(names) >= MAX_FILES:
                raise ForagerMatchedV3AtomicPublicationError(
                    "publication root exceeds its entry bound"
                )
            if entry.name in names:
                raise ForagerMatchedV3AtomicPublicationError(
                    "publication root repeats an entry"
                )
            names.add(entry.name)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ForagerMatchedV3AtomicPublicationError(
                    "cannot inspect publication entry"
                ) from exc
            record = expected.get(entry.name)
            if (
                record is None
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != FILE_MODE
                or (metadata.st_uid, metadata.st_gid) != root.owner
                or metadata.st_size != record.size_bytes
            ):
                raise ForagerMatchedV3AtomicPublicationError(
                    "publication contains an unexpected, linked, special, mis-sized, "
                    "mis-owned, or mis-moded entry"
                )
            total += metadata.st_size
            if total > MAX_TOTAL_BYTES:
                raise ForagerMatchedV3AtomicPublicationError(
                    "publication root exceeds its total byte bound"
                )
            inventory[entry.name] = _stat_identity(metadata)
    if names != set(expected):
        raise ForagerMatchedV3AtomicPublicationError(
            "publication inventory differs; "
            f"missing={sorted(set(expected) - names)!r}, "
            f"extra={sorted(names - set(expected))!r}"
        )
    return inventory


def _read_stable_regular_at(
    root: _OpenDirectory,
    record: ExactFileRecord,
) -> bytes:
    try:
        path_metadata = os.stat(
            record.name,
            dir_fd=root.descriptor,
            follow_symlinks=False,
        )
        descriptor = os.open(
            record.name,
            _file_read_flags(),
            dir_fd=root.descriptor,
        )
    except OSError as exc:
        raise ForagerMatchedV3AtomicPublicationError(
            f"cannot safely open publication file {record.name!r}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != record.size_bytes
            or stat.S_IMODE(before.st_mode) != FILE_MODE
            or (before.st_uid, before.st_gid) != root.owner
            or _stat_identity(path_metadata) != _stat_identity(before)
        ):
            raise ForagerMatchedV3AtomicPublicationError(
                f"publication file {record.name!r} is not the expected safe inode"
            )
        chunks: list[bytes] = []
        remaining = record.size_bytes
        while remaining:
            chunk = os.read(descriptor, min(remaining, _COPY_CHUNK_BYTES))
            if not chunk:
                raise ForagerMatchedV3AtomicPublicationError(
                    f"publication file {record.name!r} ended while being read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ForagerMatchedV3AtomicPublicationError(
                f"publication file {record.name!r} grew while being read"
            )
        after = os.fstat(descriptor)
        current = os.stat(
            record.name,
            dir_fd=root.descriptor,
            follow_symlinks=False,
        )
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(current)
        ):
            raise ForagerMatchedV3AtomicPublicationError(
                f"publication file {record.name!r} changed while being read"
            )
        raw = b"".join(chunks)
        if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), record.sha256):
            raise ForagerMatchedV3AtomicPublicationError(
                f"publication file {record.name!r} differs from its expected digest"
            )
        return raw
    finally:
        _close_no_raise(descriptor)


def _load_from_open_root(
    root: _OpenDirectory,
    *,
    address: str,
    records: tuple[ExactFileRecord, ...],
    expected: Mapping[str, ExactFileRecord],
) -> ContentVerifiedFlatPublication:
    initial_inventory = _root_inventory(root, expected)
    loaded = {record.name: _read_stable_regular_at(root, record) for record in records}
    if _root_inventory(root, expected) != initial_inventory:
        raise ForagerMatchedV3AtomicPublicationError(
            "publication inventory changed during strict replay"
        )
    return ContentVerifiedFlatPublication(
        root=root.path,
        address=address,
        records=records,
        files=MappingProxyType(loaded),
    )


def load_exact_flat_publication(
    parent: Path,
    *,
    address: str,
    expected_files: tuple[ExactFileRecord, ...],
) -> ContentVerifiedFlatPublication:
    """Load an exact flat inventory under an externally carried address."""

    validated_address = _require_address(address)
    records, expected = _validated_records(expected_files)
    opened_parent = _open_parent(parent)
    root: _OpenDirectory | None = None
    try:
        root = _open_directory_at(
            opened_parent,
            validated_address,
            label="published content directory",
        )
        result = _load_from_open_root(
            root,
            address=validated_address,
            records=records,
            expected=expected,
        )
        if not _entry_matches_open_directory(opened_parent, validated_address, root):
            raise ForagerMatchedV3AtomicPublicationError(
                "publication address changed during replay"
            )
        _assert_parent_path(opened_parent)
        return result
    except ForagerMatchedV3AtomicPublicationError:
        raise
    except OSError as exc:
        raise ForagerMatchedV3AtomicPublicationError(
            "publication filesystem replay failed"
        ) from exc
    finally:
        if root is not None:
            _close_no_raise(root.descriptor)
        _close_no_raise(opened_parent.descriptor)


def _write_exclusive_at(
    root: _OpenDirectory,
    record: ExactFileRecord,
    raw: bytes,
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(
            record.name,
            flags,
            FILE_MODE,
            dir_fd=root.descriptor,
        )
    except OSError as exc:
        raise ForagerMatchedV3AtomicPublicationError(
            f"cannot stage publication file {record.name!r}"
        ) from exc
    try:
        os.fchmod(descriptor, FILE_MODE)
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written < 1:
                raise ForagerMatchedV3AtomicPublicationError(
                    f"cannot completely stage publication file {record.name!r}"
                )
            remaining = remaining[written:]
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        current = os.stat(
            record.name,
            dir_fd=root.descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != record.size_bytes
            or stat.S_IMODE(opened.st_mode) != FILE_MODE
            or (opened.st_uid, opened.st_gid) != root.owner
            or _stat_identity(opened) != _stat_identity(current)
        ):
            raise ForagerMatchedV3AtomicPublicationError(
                f"staged publication file {record.name!r} changed while writing"
            )
    except ForagerMatchedV3AtomicPublicationError:
        raise
    except OSError as exc:
        raise ForagerMatchedV3AtomicPublicationError(
            f"cannot stage publication file {record.name!r}"
        ) from exc
    finally:
        _close_no_raise(descriptor)


def _durably_sync_open_tree(
    root: _OpenDirectory,
    expected: Mapping[str, ExactFileRecord],
) -> None:
    initial_inventory = _root_inventory(root, expected)
    try:
        for name in sorted(initial_inventory, key=os.fsencode):
            path_metadata = os.stat(
                name,
                dir_fd=root.descriptor,
                follow_symlinks=False,
            )
            descriptor = os.open(name, _file_read_flags(), dir_fd=root.descriptor)
            try:
                opened = os.fstat(descriptor)
                if _stat_identity(path_metadata) != _stat_identity(opened):
                    raise ForagerMatchedV3AtomicPublicationError(
                        "staged publication file changed before fsync"
                    )
                os.fsync(descriptor)
                current = os.stat(
                    name,
                    dir_fd=root.descriptor,
                    follow_symlinks=False,
                )
                if (
                    _stat_identity(opened) != _stat_identity(os.fstat(descriptor))
                    or _stat_identity(opened) != _stat_identity(current)
                ):
                    raise ForagerMatchedV3AtomicPublicationError(
                        "staged publication file changed during fsync"
                    )
            finally:
                _close_no_raise(descriptor)
        os.fsync(root.descriptor)
    except ForagerMatchedV3AtomicPublicationError:
        raise
    except OSError as exc:
        raise ForagerMatchedV3AtomicPublicationError(
            "cannot durably sync publication staging tree"
        ) from exc
    if _root_inventory(root, expected) != initial_inventory:
        raise ForagerMatchedV3AtomicPublicationError(
            "staged publication inventory changed during fsync"
        )


def _rename_no_replace(
    parent: _OpenDirectory,
    source_name: str,
    destination_name: str,
) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
    except (OSError, TypeError, ValueError) as exc:
        raise ForagerMatchedV3AtomicPublicationError(
            "cannot resolve Linux renameat2; no overwrite fallback is permitted"
        ) from exc
    if renameat2 is None:
        raise ForagerMatchedV3AtomicPublicationError(
            "Linux renameat2 is required; no overwrite fallback is permitted"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        parent.descriptor,
        os.fsencode(source_name),
        parent.descriptor,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise ForagerMatchedV3AtomicPublicationCollisionError(
            "publication address was created concurrently"
        )
    raise ForagerMatchedV3AtomicPublicationError(
        f"exclusive renameat2 publication failed with errno {error}; "
        "no overwrite fallback is permitted"
    )


def _sync_publication_parent(parent: _OpenDirectory) -> None:
    os.fsync(parent.descriptor)


def _publish_verified_no_replace(
    parent: _OpenDirectory,
    staging: _OpenDirectory,
    staging_name: str,
    destination: Path,
    address: str,
) -> None:
    _assert_parent_path(parent)
    if not _entry_matches_open_directory(parent, staging_name, staging):
        raise ForagerMatchedV3AtomicPublicationError(
            "staging name no longer refers to the verified publication inode"
        )
    try:
        _rename_no_replace(parent, staging_name, address)
    except BaseException as exc:
        destination_matches = _entry_matches_open_directory(parent, address, staging)
        source_matches = _entry_matches_open_directory(parent, staging_name, staging)
        if destination_matches:
            raise ForagerMatchedV3AtomicPublicationUncertainError(
                destination,
                address,
                "exclusive move reported failure after destination became visible",
                committed=True,
            ) from exc
        if not source_matches:
            raise ForagerMatchedV3AtomicPublicationUncertainError(
                destination,
                address,
                "exclusive move outcome cannot be established",
                committed=None,
            ) from exc
        raise
    try:
        if not _entry_matches_open_directory(parent, address, staging):
            raise ForagerMatchedV3AtomicPublicationError(
                "published destination differs from the verified staging inode"
            )
        if _entry_exists(parent, staging_name):
            raise ForagerMatchedV3AtomicPublicationError(
                "staging name survived exclusive publication"
            )
        _sync_publication_parent(parent)
        if not _entry_matches_open_directory(parent, address, staging):
            raise ForagerMatchedV3AtomicPublicationError(
                "published destination changed during parent fsync"
            )
        _assert_parent_path(parent)
    except BaseException as exc:
        raise ForagerMatchedV3AtomicPublicationUncertainError(
            destination,
            address,
            "publication committed before durability or inode verification failed",
            committed=True,
        ) from exc


def _cleanup_owned_staging(
    parent: _OpenDirectory,
    name: str,
    staging: _OpenDirectory,
) -> None:
    """Remove only a still-owned flat staging inode; never recurse or follow links."""

    try:
        if not _entry_matches_open_directory(parent, name, staging):
            return
        root_metadata = os.fstat(staging.descriptor)
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or stat.S_IMODE(root_metadata.st_mode) != DIRECTORY_MODE
            or (root_metadata.st_uid, root_metadata.st_gid) != parent.owner
        ):
            return
        entries: list[tuple[str, tuple[int, ...]]] = []
        with os.scandir(staging.descriptor) as iterator:
            for entry in iterator:
                if len(entries) >= MAX_FILES:
                    return
                metadata = entry.stat(follow_symlinks=False)
                if (
                    _SAFE_NAME_RE.fullmatch(entry.name) is None
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != FILE_MODE
                    or (metadata.st_uid, metadata.st_gid) != staging.owner
                ):
                    return
                entries.append((entry.name, _stat_identity(metadata)))
        for entry_name, expected_identity in entries:
            current = os.stat(
                entry_name,
                dir_fd=staging.descriptor,
                follow_symlinks=False,
            )
            if _stat_identity(current) != expected_identity:
                return
        for entry_name, _ in entries:
            os.unlink(entry_name, dir_fd=staging.descriptor)
        if _entry_matches_open_directory(parent, name, staging):
            os.rmdir(name, dir_fd=parent.descriptor)
    except BaseException:
        return


def _best_effort_cleanup(
    parent: _OpenDirectory,
    name: str,
    staging: _OpenDirectory,
) -> None:
    try:
        _cleanup_owned_staging(parent, name, staging)
    except BaseException:
        pass


def _create_owned_staging(parent: _OpenDirectory) -> tuple[str, _OpenDirectory]:
    for _ in range(_STAGING_ATTEMPTS):
        name = f"{_STAGING_PREFIX}{secrets.token_hex(16)}"
        try:
            os.mkdir(name, DIRECTORY_MODE, dir_fd=parent.descriptor)
        except FileExistsError:
            continue
        except OSError as exc:
            raise ForagerMatchedV3AtomicPublicationError(
                "cannot create publication staging directory"
            ) from exc
        staging: _OpenDirectory | None = None
        created_identity: tuple[int, int, int] | None = None
        try:
            created = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
            created_identity = _inode_identity(created)
            if (
                not stat.S_ISDIR(created.st_mode)
                or created.st_uid != os.geteuid()
                or created.st_gid != parent.owner[1]
            ):
                raise ForagerMatchedV3AtomicPublicationError(
                    "new publication staging inode is not exclusively owned"
                )
            os.chmod(
                name,
                DIRECTORY_MODE,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            normalized = os.stat(
                name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            if (
                _inode_identity(normalized) != created_identity
                or stat.S_IMODE(normalized.st_mode) != DIRECTORY_MODE
                or (normalized.st_uid, normalized.st_gid) != parent.owner
            ):
                raise ForagerMatchedV3AtomicPublicationError(
                    "publication staging inode changed during normalization"
                )
            staging = _open_directory_at(
                parent,
                name,
                label="publication staging directory",
            )
            os.fchmod(staging.descriptor, DIRECTORY_MODE)
            verified = os.fstat(staging.descriptor)
            if (
                _inode_identity(verified) != created_identity
                or stat.S_IMODE(verified.st_mode) != DIRECTORY_MODE
                or (verified.st_uid, verified.st_gid) != parent.owner
            ):
                raise ForagerMatchedV3AtomicPublicationError(
                    "publication staging ownership or mode is unsafe"
                )
            return name, staging
        except BaseException:
            if staging is not None:
                _best_effort_cleanup(parent, name, staging)
                _close_no_raise(staging.descriptor)
            elif created_identity is not None:
                cleanup: _OpenDirectory | None = None
                try:
                    cleanup = _open_directory_at(
                        parent,
                        name,
                        label="failed publication staging directory",
                    )
                    if cleanup.inode_identity == created_identity:
                        _best_effort_cleanup(parent, name, cleanup)
                except BaseException:
                    pass
                finally:
                    if cleanup is not None:
                        _close_no_raise(cleanup.descriptor)
            else:
                try:
                    os.rmdir(name, dir_fd=parent.descriptor)
                except OSError:
                    pass
            raise
    raise ForagerMatchedV3AtomicPublicationError(
        "cannot allocate a unique publication staging directory"
    )


def publish_exact_flat_publication(
    parent: Path,
    *,
    address: str,
    expected_files: tuple[ExactFileRecord, ...],
    payloads: dict[str, bytes],
) -> ContentVerifiedFlatPublication:
    """Durably publish exact bytes once without interpreting or authorizing them."""

    validated_address = _require_address(address)
    records, expected = _validated_records(expected_files)
    validated_payloads = _validated_payloads(payloads, expected)
    opened_parent = _open_parent(parent)
    destination = opened_parent.path / validated_address
    staging: _OpenDirectory | None = None
    staging_name = ""
    committed = False
    try:
        if _entry_exists(opened_parent, validated_address):
            raise ForagerMatchedV3AtomicPublicationCollisionError(
                "publication address already exists"
            )
        staging_name, staging = _create_owned_staging(opened_parent)
        for record in records:
            _write_exclusive_at(staging, record, validated_payloads[record.name])
        _load_from_open_root(
            staging,
            address=validated_address,
            records=records,
            expected=expected,
        )
        _durably_sync_open_tree(staging, expected)
        _publish_verified_no_replace(
            opened_parent,
            staging,
            staging_name,
            destination,
            validated_address,
        )
        committed = True
        final_root = _open_directory_at(
            opened_parent,
            validated_address,
            label="published content directory",
        )
        try:
            if (
                not _entry_matches_open_directory(
                    opened_parent, validated_address, staging
                )
                or not _entry_matches_open_directory(
                    opened_parent, validated_address, final_root
                )
            ):
                raise ForagerMatchedV3AtomicPublicationError(
                    "published inode changed before final replay"
                )
            result = _load_from_open_root(
                final_root,
                address=validated_address,
                records=records,
                expected=expected,
            )
            if not _entry_matches_open_directory(
                opened_parent, validated_address, final_root
            ):
                raise ForagerMatchedV3AtomicPublicationError(
                    "published address changed during final replay"
                )
            _assert_parent_path(opened_parent)
            return result
        except BaseException as exc:
            raise ForagerMatchedV3AtomicPublicationUncertainError(
                destination,
                validated_address,
                "final strict replay failed after publication committed",
                committed=True,
            ) from exc
        finally:
            _close_no_raise(final_root.descriptor)
    except ForagerMatchedV3AtomicPublicationUncertainError:
        raise
    except BaseException as exc:
        destination_matches = (
            staging is not None
            and _entry_matches_open_directory(
                opened_parent, validated_address, staging
            )
        )
        source_matches = (
            staging is not None
            and bool(staging_name)
            and _entry_matches_open_directory(opened_parent, staging_name, staging)
        )
        if committed or destination_matches:
            raise ForagerMatchedV3AtomicPublicationUncertainError(
                destination,
                validated_address,
                "publication is visible but final state is uncertain",
                committed=True,
            ) from exc
        if staging is not None and not source_matches:
            raise ForagerMatchedV3AtomicPublicationUncertainError(
                destination,
                validated_address,
                "staging disappeared before commit state could be established",
                committed=None,
            ) from exc
        if staging is not None:
            _best_effort_cleanup(opened_parent, staging_name, staging)
        if isinstance(exc, OSError):
            raise ForagerMatchedV3AtomicPublicationError(
                "publication filesystem operation failed before commit"
            ) from exc
        raise
    finally:
        if staging is not None:
            _close_no_raise(staging.descriptor)
        _close_no_raise(opened_parent.descriptor)


__all__ = [
    "ATOMIC_PUBLICATION_DESCRIPTOR_SCHEMA_VERSION",
    "ATOMIC_PUBLICATION_DESCRIPTOR_SHA256",
    "ATOMIC_PUBLICATION_STATUS",
    "ContentVerifiedFlatPublication",
    "ExactFileRecord",
    "ForagerMatchedV3AtomicPublicationCollisionError",
    "ForagerMatchedV3AtomicPublicationError",
    "ForagerMatchedV3AtomicPublicationUncertainError",
    "atomic_publication_descriptor",
    "canonical_atomic_publication_descriptor_bytes",
    "load_exact_flat_publication",
    "publish_exact_flat_publication",
]
