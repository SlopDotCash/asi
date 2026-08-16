"""Durable content-only publication for matched-v3 external source USTARs.

The staging boundary deliberately returns only a PID-bound, sealed, unlinked
descriptor.  This module is the separate explicit filesystem-publication boundary.  It
accepts only that exact live capability, independently replays the raw canonical USTAR,
and atomically publishes two read-only files below a caller-opened ``sha256`` namespace.

Publication makes bytes available to a later, separately authorized OCI builder.  It
does not execute or import staged source, trust a daemon, issue seeds, accept results,
qualify a runtime or candidate, create evidence, or authorize scientific claims.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, NoReturn, cast

from alberta_framework.benchmarks import (
    forager_matched_v3_external_materialization as materialization,
)
from alberta_framework.benchmarks import forager_matched_v3_external_staging as staging

EXTERNAL_SOURCE_PUBLICATION_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_source_publication_contract.v1"
)
EXTERNAL_SOURCE_PUBLICATION_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.external_source_publication_receipt.v1"
)
EXTERNAL_SOURCE_PUBLICATION_STATUS: Final = "content_published_unqualified_non_authorizing"
EXTERNAL_SOURCE_ARCHIVE_FILENAME: Final = "external-source.v1.tar"
EXTERNAL_SOURCE_RECEIPT_FILENAME: Final = "receipt.v1.json"

_STAGING_SOURCE_PATH: Final = "alberta_framework/benchmarks/forager_matched_v3_external_staging.py"
_STAGING_SOURCE_SHA256: Final = "675d54edcf2f87c1847712e7a480e2e5134312d040a68a1102c10c4829f8fba0"
_MATERIALIZER_SOURCE_PATH: Final = (
    "alberta_framework/benchmarks/forager_matched_v3_external_materialization.py"
)
_MATERIALIZER_SOURCE_SHA256: Final = (
    "3ff59a9f88d79b122fa66a1cdca009a68ff524806a7a7c58e5d565cd30ecaafe"
)

_MAX_ARCHIVE_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_RECEIPT_BYTES: Final = 64 * 1024 * 1024
_MAX_MANIFEST_BYTES: Final = 64 * 1024 * 1024
_MAX_MEMBERS: Final = 20_002
_MAX_MEMBER_BYTES: Final = 256 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 500_000
_MAX_JSON_STRING_BYTES: Final = 64 * 1024 * 1024
_MAX_PATH_BYTES: Final = 255
_MAX_COMPONENT_BYTES: Final = 255
_MAX_PATH_COMPONENTS: Final = 256
_READ_CHUNK_BYTES: Final = 1024 * 1024
_USTAR_BLOCK_BYTES: Final = 512
_USTAR_RECORD_BYTES: Final = 10 * 1024
_STAGING_ATTEMPTS: Final = 64
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA1_RE: Final = re.compile(r"[0-9a-f]{40}\Z")


class ForagerMatchedV3ExternalSourcePublicationError(RuntimeError):
    """The source capability, receipt, archive, or filesystem failed closed."""


class PublishedMatchedV3ExternalSourceUncertainError(
    ForagerMatchedV3ExternalSourcePublicationError
):
    """The atomic destination became visible but final durability is uncertain."""

    def __init__(
        self,
        destination: Path,
        detail: str,
        *,
        archive_sha256: str,
        receipt_sha256: str,
    ) -> None:
        self.destination = destination
        self.archive_sha256 = archive_sha256
        self.receipt_sha256 = receipt_sha256
        super().__init__(f"external source publication at {destination}, but {detail}")


@dataclass(frozen=True, slots=True)
class PublishedMatchedV3ExternalSource:
    """Named content pins with no execution, daemon-trust, or scientific authority."""

    directory: Path
    archive: Path
    receipt: Path
    archive_sha256: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class _OpenDirectory:
    path: Path
    descriptor: int
    identity: tuple[int, ...]


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3ExternalSourcePublicationError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _source_sha256(module_file: object, expected_suffix: str) -> str:
    if (
        type(module_file) is not str
        or not module_file.endswith(expected_suffix)
        or not Path(module_file).is_absolute()
    ):
        raise RuntimeError(f"cannot resolve exact source path for {expected_suffix}")
    descriptor = -1
    try:
        before = os.stat(module_file, follow_symlinks=False)
        descriptor = os.open(
            module_file,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_uid,
            opened.st_gid,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_gid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not 1 <= opened.st_size <= _MAX_MANIFEST_BYTES
            or identity != before_identity
        ):
            raise RuntimeError(f"unsafe exact source file for {expected_suffix}")
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
            if not chunk:
                raise RuntimeError(f"exact source file ended early for {expected_suffix}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"exact source file grew while read for {expected_suffix}")
        after = os.fstat(descriptor)
        located = os.stat(module_file, follow_symlinks=False)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        located_identity = (
            located.st_dev,
            located.st_ino,
            located.st_mode,
            located.st_nlink,
            located.st_uid,
            located.st_gid,
            located.st_size,
            located.st_mtime_ns,
            located.st_ctime_ns,
        )
        if identity != after_identity or identity != located_identity:
            raise RuntimeError(f"exact source file changed while read for {expected_suffix}")
        return digest.hexdigest()
    except OSError as exc:
        raise RuntimeError(f"cannot read exact source bytes for {expected_suffix}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


if not hmac.compare_digest(
    _source_sha256(staging.__file__, _STAGING_SOURCE_PATH),
    _STAGING_SOURCE_SHA256,
):
    raise RuntimeError("external source publication staging-source binding drifted")
if not hmac.compare_digest(
    _source_sha256(materialization.__file__, _MATERIALIZER_SOURCE_PATH),
    _MATERIALIZER_SOURCE_SHA256,
):
    raise RuntimeError("external source publication materializer-source binding drifted")


def _claims() -> dict[str, bool]:
    return {
        "acceptance_authority_granted": False,
        "artifact_accepted": False,
        "candidate_qualified": False,
        "daemon_trust_granted": False,
        "evidence_authority_granted": False,
        "execution_authority_granted": False,
        "execution_ready": False,
        "performance_claim_allowed": False,
        "qualification_authority_granted": False,
        "result_acceptance_authority_granted": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "seed_authority_granted": False,
        "universal_sota_claim_allowed": False,
        "workload_executed": False,
    }


def _limitations() -> list[str]:
    return [
        "Publication preserves content identity but does not authenticate or trust an OCI daemon.",
        "The published archive is an input capability, not execution or qualification authority.",
        "No workload source is imported or executed by this publisher or validator.",
        "No runtime, dependency installation, hardware, RNG trace, or result is qualified here.",
        "A valid publication grants no acceptance, evidence, promotion, or performance authority.",
        (
            "Named-path stability assumes no concurrent process with the same effective UID "
            "mutates publication roots, namespaces, staging entries, or published paths."
        ),
    ]


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": EXTERNAL_SOURCE_PUBLICATION_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
        "status": EXTERNAL_SOURCE_PUBLICATION_STATUS,
        "classification": "durable_external_source_content_non_evidence",
        "dependency": {
            "materializer_identity_sha256": (
                materialization.PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256
            ),
            "materializer_source_path": _MATERIALIZER_SOURCE_PATH,
            "materializer_source_sha256": _MATERIALIZER_SOURCE_SHA256,
            "staging_descriptor_sha256": (staging.EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SHA256),
            "staging_source_path": _STAGING_SOURCE_PATH,
            "staging_source_sha256": _STAGING_SOURCE_SHA256,
        },
        "layout": {
            "namespace": "sha256",
            "directory_name": "exact_archive_sha256",
            "archive_filename": EXTERNAL_SOURCE_ARCHIVE_FILENAME,
            "receipt_filename": EXTERNAL_SOURCE_RECEIPT_FILENAME,
            "publication_directory_mode": "0555",
            "file_mode": "0444",
            "atomic_commit": "renameat2_RENAME_NOREPLACE",
        },
        "verification": {
            "exact_live_staging_capability_required": True,
            "pre_and_post_copy_capability_replay": True,
            "independent_raw_canonical_ustar_replay": True,
            "embedded_stage_manifest_replay": True,
            "embedded_materializer_manifest_replay": True,
            "descriptor_relative_nofollow_single_link_reads": True,
            "caller_external_receipt_sha256_required": True,
            "new_only_collision_refusal": True,
            "owned_partial_cleanup": True,
            "postcommit_failure_reported_as_uncertain": True,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _raise_json_constant(value: str) -> NoReturn:
    _fail(f"publication JSON contains forbidden constant {value!r}")


def _raise_json_float(value: str) -> NoReturn:
    _fail(f"publication JSON contains forbidden float {value!r}")


def _parse_json_int(value: str) -> int:
    if len(value.lstrip("-")) > 20:
        _fail("publication JSON integer exceeds its lexical bound")
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"publication JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _assert_plain_json(value: Any) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            _fail("publication JSON exceeds its node bound")
        if depth > _MAX_JSON_DEPTH:
            _fail("publication JSON exceeds its depth bound")
        if type(item) in {dict, list}:
            identity = id(item)
            if identity in seen:
                _fail("publication JSON must be an unaliased acyclic tree")
            seen.add(identity)
            if type(item) is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        _fail("publication JSON keys must be exact strings")
                    pending.append((child, depth + 1))
            else:
                pending.extend((child, depth + 1) for child in cast(list[Any], item))
        elif type(item) is str:
            try:
                encoded = item.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ForagerMatchedV3ExternalSourcePublicationError(
                    "publication JSON strings must be ASCII"
                ) from exc
            if len(encoded) > _MAX_JSON_STRING_BYTES:
                _fail("publication JSON string exceeds its byte bound")
        elif item is not None and type(item) not in {bool, int}:
            _fail("publication JSON contains a non-plain scalar")


def _canonical_json(value: Mapping[str, Any], *, newline: bool = True) -> bytes:
    if type(value) is not dict:
        _fail("publication canonical JSON root must be one plain object")
    _assert_plain_json(value)
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3ExternalSourcePublicationError(
            "publication value is not canonical ASCII JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def _strict_json(raw: bytes, *, label: str, maximum: int) -> dict[str, Any]:
    if type(raw) is not bytes or not 0 < len(raw) <= maximum:
        _fail(f"{label} must be bounded exact bytes")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail(f"{label} must have one canonical trailing newline")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_float=_raise_json_float,
            parse_int=_parse_json_int,
        )
    except ForagerMatchedV3ExternalSourcePublicationError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise ForagerMatchedV3ExternalSourcePublicationError(
            f"{label} is not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        _fail(f"{label} root must be one plain object")
    result = cast(dict[str, Any], value)
    _assert_plain_json(result)
    if not hmac.compare_digest(_canonical_json(result), raw):
        _fail(f"{label} is not in exact canonical form")
    return result


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
EXTERNAL_SOURCE_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "d76657b2f0d65adae377e21fa391628aa5749acb476c69aa64ce542a716f146d"
)
if not hmac.compare_digest(
    _sha256(_DESCRIPTOR_BYTES),
    EXTERNAL_SOURCE_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256,
):
    raise AssertionError("external source publication descriptor hash drifted")


def external_source_publication_contract_descriptor() -> dict[str, Any]:
    """Return a detached, authority-denying publisher descriptor."""

    return _strict_json(
        _DESCRIPTOR_BYTES,
        label="external source publication descriptor",
        maximum=_MAX_RECEIPT_BYTES,
    )


def canonical_external_source_publication_contract_descriptor_bytes() -> bytes:
    """Return the exact canonical publisher descriptor bytes."""

    external_source_publication_contract_descriptor()
    return _DESCRIPTOR_BYTES


def parse_external_source_publication_contract_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact frozen publisher descriptor."""

    value = _strict_json(
        raw,
        label="external source publication descriptor",
        maximum=_MAX_RECEIPT_BYTES,
    )
    if raw != _DESCRIPTOR_BYTES or _sha256(raw) != (
        EXTERNAL_SOURCE_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256
    ):
        _fail("external source publication descriptor identity differs")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be one lowercase SHA-256")
    return value


def _require_git_sha1(value: object, *, label: str) -> str:
    if type(value) is not str or _GIT_SHA1_RE.fullmatch(value) is None:
        _fail(f"{label} must be one lowercase Git SHA-1")
    return value


def _require_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be one bounded exact integer")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], *, label: str) -> None:
    if type(value) is not dict or set(value) != keys:
        _fail(f"{label} fields differ")


def _validate_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be one nonempty exact path")
    path = value
    if unicodedata.normalize("NFKC", path) != path:
        _fail(f"{label} must use NFKC Unicode")
    if "\\" in path or any(ord(character) < 32 or ord(character) == 127 for character in path):
        _fail(f"{label} contains a forbidden character")
    if any(character in '<>:"|?*' for character in path):
        _fail(f"{label} contains a reserved portable character")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or pure.as_posix() != path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail(f"{label} is not a canonical relative path")
    try:
        encoded = path.encode("utf-8", "strict")
        components = tuple(part.encode("utf-8", "strict") for part in pure.parts)
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3ExternalSourcePublicationError(
            f"{label} is not strict UTF-8"
        ) from exc
    if (
        len(encoded) > _MAX_PATH_BYTES
        or len(components) > _MAX_PATH_COMPONENTS
        or any(len(component) > _MAX_COMPONENT_BYTES for component in components)
    ):
        _fail(f"{label} exceeds its path bound")
    _split_ustar_path(path)
    return path


def _split_ustar_path(path: str) -> tuple[bytes, bytes]:
    encoded = path.encode("utf-8", "strict")
    if len(encoded) <= 100:
        return b"", encoded
    for slash in reversed([index for index, byte in enumerate(encoded) if byte == ord("/")]):
        prefix = encoded[:slash]
        name = encoded[slash + 1 :]
        if prefix and name and len(prefix) <= 155 and len(name) <= 100:
            return prefix, name
    _fail(f"path is not exactly representable in POSIX USTAR: {path}")


def _ustar_octal(value: int, width: int, *, label: str) -> bytes:
    if type(value) is not int or value < 0:
        _fail(f"{label} is not nonnegative")
    token = format(value, "o").encode("ascii")
    if len(token) > width - 1:
        _fail(f"{label} exceeds its USTAR field")
    return token.rjust(width - 1, b"0") + b"\0"


def _canonical_ustar_header(path: str, size: int, mode: int) -> bytes:
    if mode not in {0o444, 0o555}:
        _fail("USTAR member mode is not frozen")
    prefix, name = _split_ustar_path(path)
    header = bytearray(_USTAR_BLOCK_BYTES)
    header[0 : len(name)] = name
    header[100:108] = _ustar_octal(mode, 8, label="member mode")
    header[108:116] = _ustar_octal(0, 8, label="member uid")
    header[116:124] = _ustar_octal(0, 8, label="member gid")
    header[124:136] = _ustar_octal(size, 12, label="member size")
    header[136:148] = _ustar_octal(0, 12, label="member mtime")
    header[148:156] = b"        "
    header[156:157] = b"0"
    header[257:263] = b"ustar\0"
    header[263:265] = b"00"
    header[345 : 345 + len(prefix)] = prefix
    checksum = format(sum(header), "06o").encode("ascii")
    if len(checksum) != 6:
        _fail("USTAR checksum overflowed")
    header[148:156] = checksum + b"\0 "
    return bytes(header)


def _complete_inventory(
    manifest_raw: bytes,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    inventory = [dict(item) for item in cast(list[dict[str, Any]], manifest["payload_inventory"])]
    inventory.append(
        {
            "mode": "0444",
            "path": staging.EXTERNAL_STAGING_FINAL_MANIFEST_PATH,
            "provenance": "final_staging_manifest_self",
            "sha256": _sha256(manifest_raw),
            "size_bytes": len(manifest_raw),
        }
    )
    inventory.sort(key=lambda item: cast(str, item["path"]).encode("utf-8"))
    return inventory


def _source_binding(base_raw: bytes, base: Mapping[str, Any]) -> dict[str, Any]:
    identity = base.get("identity")
    source_tree = base.get("source_tree")
    claims = base.get("claims")
    if type(identity) is not dict or type(source_tree) is not dict or type(claims) is not dict:
        _fail("embedded materializer manifest identity fields differ")
    identity_value = cast(dict[str, Any], identity)
    source_value = cast(dict[str, Any], source_tree)
    claim_value = cast(dict[str, Any], claims)
    if not claim_value or any(value is not False for value in claim_value.values()):
        _fail("embedded materializer manifest authority denial differs")
    return {
        "commit_git_sha1": _require_git_sha1(
            identity_value.get("commit_git_sha1"), label="external source commit"
        ),
        "excluded_gitlink_count": _require_int(
            source_value.get("excluded_gitlink_count"),
            label="external source excluded gitlinks",
            maximum=_MAX_MEMBERS,
        ),
        "full_file_sha256": _sha256(base_raw),
        "identity_sha256": _require_sha256(
            base.get("identity_sha256"), label="external source identity"
        ),
        "materialized_regular_file_count": _require_int(
            source_value.get("materialized_regular_file_count"),
            label="external source file count",
            minimum=1,
            maximum=_MAX_MEMBERS,
        ),
        "materialized_total_size_bytes": _require_int(
            source_value.get("materialized_total_size_bytes"),
            label="external source payload size",
            maximum=_MAX_ARCHIVE_BYTES,
        ),
        "payload_sha256": _require_sha256(
            base.get("payload_sha256"), label="external source manifest body"
        ),
        "schema_version": base.get("schema_version"),
        "size_bytes": len(base_raw),
        "source_tree_sha256": _sha256(_canonical_json(source_value, newline=False)),
        "tracked_entry_count": _require_int(
            source_value.get("tracked_entry_count"),
            label="external source tracked entry count",
            minimum=1,
            maximum=_MAX_MEMBERS,
        ),
        "tree_git_sha1": _require_git_sha1(
            identity_value.get("tree_git_sha1"), label="external source tree"
        ),
    }


def _receipt_bytes(
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    inventory: list[dict[str, Any]],
    stage_raw: bytes,
    stage_manifest: Mapping[str, Any],
    base_raw: bytes,
    base_manifest: Mapping[str, Any],
) -> bytes:
    body: dict[str, Any] = {
        "archive": {
            "format": "canonical_posix_ustar_uncompressed",
            "inventory_sha256": _sha256(
                json.dumps(
                    inventory,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
            ),
            "member_count": len(inventory),
            "members": inventory,
            "payload_size_bytes": sum(cast(int, item["size_bytes"]) for item in inventory),
            "record_size_bytes": _USTAR_RECORD_BYTES,
            "sha256": archive_sha256,
            "size_bytes": archive_size_bytes,
        },
        "claims": _claims(),
        "classification": "durable_external_source_content_non_evidence",
        "external_source_manifest": _source_binding(base_raw, base_manifest),
        "limitations": _limitations(),
        "publication_contract": {
            "descriptor_sha256": EXTERNAL_SOURCE_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256,
            "materializer_source_sha256": _MATERIALIZER_SOURCE_SHA256,
            "staging_descriptor_sha256": (staging.EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SHA256),
            "staging_source_sha256": _STAGING_SOURCE_SHA256,
        },
        "schema_version": EXTERNAL_SOURCE_PUBLICATION_RECEIPT_SCHEMA_VERSION,
        "staging_manifest": {
            "body_sha256": stage_manifest["manifest_body_sha256"],
            "full_file_sha256": _sha256(stage_raw),
            "schema_version": stage_manifest["schema_version"],
            "size_bytes": len(stage_raw),
            "status": stage_manifest["status"],
        },
        "status": EXTERNAL_SOURCE_PUBLICATION_STATUS,
    }
    body["receipt_body_sha256"] = _sha256(_canonical_json(body, newline=False))
    raw = _canonical_json(body)
    if len(raw) > _MAX_RECEIPT_BYTES:
        _fail("external source publication receipt exceeds its byte bound")
    return raw


def _validate_member_inventory(value: object) -> list[dict[str, Any]]:
    if type(value) is not list or not 1 <= len(value) <= _MAX_MEMBERS:
        _fail("publication member inventory is invalid")
    records: list[dict[str, Any]] = []
    previous: bytes | None = None
    paths: set[str] = set()
    for index, item in enumerate(value):
        if type(item) is not dict:
            _fail(f"publication member {index} must be one object")
        record = cast(dict[str, Any], item)
        _exact_keys(
            record,
            {"mode", "path", "provenance", "sha256", "size_bytes"},
            label=f"publication member {index}",
        )
        path = _validate_path(record["path"], label=f"publication member {index} path")
        encoded = path.encode("utf-8")
        if previous is not None and encoded <= previous:
            _fail("publication members must be UTF-8-path-sorted and unique")
        if path in paths:
            _fail("publication member paths repeat")
        if record["mode"] not in {"0444", "0555"}:
            _fail("publication member mode differs")
        if record["provenance"] not in {
            "derived_configuration_overlay",
            "final_staging_manifest_self",
            "materializer_v2_regular_file",
            "relocated_exact_materializer_v2_manifest",
        }:
            _fail("publication member provenance differs")
        _require_sha256(record["sha256"], label=f"publication member {path}")
        _require_int(
            record["size_bytes"],
            label=f"publication member {path} size",
            maximum=_MAX_MEMBER_BYTES,
        )
        records.append(copy.deepcopy(record))
        paths.add(path)
        previous = encoded
    return records


def parse_external_source_publication_receipt(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Parse one externally pinned canonical, authority-denying receipt."""

    expected = _require_sha256(expected_file_sha256, label="publication receipt file")
    if type(raw) is not bytes or not hmac.compare_digest(_sha256(raw), expected):
        _fail("publication receipt full-file SHA-256 differs")
    receipt = _strict_json(
        raw, label="external source publication receipt", maximum=_MAX_RECEIPT_BYTES
    )
    _exact_keys(
        receipt,
        {
            "archive",
            "claims",
            "classification",
            "external_source_manifest",
            "limitations",
            "publication_contract",
            "receipt_body_sha256",
            "schema_version",
            "staging_manifest",
            "status",
        },
        label="external source publication receipt",
    )
    if (
        receipt["schema_version"] != EXTERNAL_SOURCE_PUBLICATION_RECEIPT_SCHEMA_VERSION
        or receipt["status"] != EXTERNAL_SOURCE_PUBLICATION_STATUS
        or receipt["classification"] != "durable_external_source_content_non_evidence"
    ):
        _fail("publication receipt schema/status/classification differs")
    claims = receipt["claims"]
    if (
        type(claims) is not dict
        or set(claims) != set(_claims())
        or any(claims[key] is not False for key in _claims())
        or receipt["limitations"] != _limitations()
    ):
        _fail("publication receipt authority denial differs")
    contract = receipt["publication_contract"]
    if type(contract) is not dict or contract != {
        "descriptor_sha256": EXTERNAL_SOURCE_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256,
        "materializer_source_sha256": _MATERIALIZER_SOURCE_SHA256,
        "staging_descriptor_sha256": staging.EXTERNAL_STAGING_CONTRACT_DESCRIPTOR_SHA256,
        "staging_source_sha256": _STAGING_SOURCE_SHA256,
    }:
        _fail("publication receipt contract binding differs")
    stage_value = receipt["staging_manifest"]
    if type(stage_value) is not dict:
        _fail("publication stage-manifest binding must be one object")
    stage = cast(dict[str, Any], stage_value)
    _exact_keys(
        stage,
        {"body_sha256", "full_file_sha256", "schema_version", "size_bytes", "status"},
        label="publication stage-manifest binding",
    )
    if (
        stage["schema_version"] != staging.EXTERNAL_STAGING_MANIFEST_SCHEMA_VERSION
        or stage["status"] != staging.EXTERNAL_STAGING_STATUS
    ):
        _fail("publication stage-manifest schema/status differs")
    _require_sha256(stage["body_sha256"], label="stage manifest body")
    _require_sha256(stage["full_file_sha256"], label="stage manifest file")
    _require_int(
        stage["size_bytes"],
        label="stage manifest size",
        minimum=1,
        maximum=_MAX_MANIFEST_BYTES,
    )
    source_value = receipt["external_source_manifest"]
    if type(source_value) is not dict:
        _fail("publication source-manifest binding must be one object")
    source = cast(dict[str, Any], source_value)
    _exact_keys(
        source,
        {
            "commit_git_sha1",
            "excluded_gitlink_count",
            "full_file_sha256",
            "identity_sha256",
            "materialized_regular_file_count",
            "materialized_total_size_bytes",
            "payload_sha256",
            "schema_version",
            "size_bytes",
            "source_tree_sha256",
            "tracked_entry_count",
            "tree_git_sha1",
        },
        label="publication source-manifest binding",
    )
    if source["schema_version"] != materialization.EXTERNAL_MATERIALIZATION_SCHEMA_VERSION:
        _fail("publication source-manifest schema differs")
    if source["identity_sha256"] != (
        materialization.PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256
    ):
        _fail("publication source-manifest identity differs")
    for key in ("full_file_sha256", "identity_sha256", "payload_sha256", "source_tree_sha256"):
        _require_sha256(source[key], label=f"source manifest {key}")
    _require_git_sha1(source["commit_git_sha1"], label="source manifest commit")
    _require_git_sha1(source["tree_git_sha1"], label="source manifest tree")
    for key, minimum, maximum in (
        ("excluded_gitlink_count", 0, _MAX_MEMBERS),
        ("materialized_regular_file_count", 1, _MAX_MEMBERS),
        ("materialized_total_size_bytes", 0, _MAX_ARCHIVE_BYTES),
        ("size_bytes", 1, _MAX_MANIFEST_BYTES),
        ("tracked_entry_count", 1, _MAX_MEMBERS),
    ):
        _require_int(source[key], label=f"source manifest {key}", minimum=minimum, maximum=maximum)
    archive_value = receipt["archive"]
    if type(archive_value) is not dict:
        _fail("publication archive binding must be one object")
    archive = cast(dict[str, Any], archive_value)
    _exact_keys(
        archive,
        {
            "format",
            "inventory_sha256",
            "member_count",
            "members",
            "payload_size_bytes",
            "record_size_bytes",
            "sha256",
            "size_bytes",
        },
        label="publication archive binding",
    )
    if (
        archive["format"] != "canonical_posix_ustar_uncompressed"
        or archive["record_size_bytes"] != _USTAR_RECORD_BYTES
    ):
        _fail("publication archive format differs")
    members = _validate_member_inventory(archive["members"])
    if archive["member_count"] != len(members):
        _fail("publication archive member count differs")
    inventory_sha256 = _require_sha256(
        archive["inventory_sha256"], label="publication archive inventory"
    )
    encoded_inventory = json.dumps(
        members,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    if inventory_sha256 != _sha256(encoded_inventory):
        _fail("publication archive inventory digest differs")
    payload_size = _require_int(
        archive["payload_size_bytes"],
        label="publication archive payload",
        maximum=_MAX_ARCHIVE_BYTES,
    )
    if payload_size != sum(cast(int, item["size_bytes"]) for item in members):
        _fail("publication archive payload total differs")
    archive_size = _require_int(
        archive["size_bytes"],
        label="publication archive size",
        minimum=_USTAR_RECORD_BYTES,
        maximum=_MAX_ARCHIVE_BYTES,
    )
    _require_sha256(archive["sha256"], label="publication archive")
    canonical_size = sum(
        _USTAR_BLOCK_BYTES
        + cast(int, item["size_bytes"])
        + (-cast(int, item["size_bytes"])) % _USTAR_BLOCK_BYTES
        for item in members
    )
    canonical_size += 2 * _USTAR_BLOCK_BYTES
    canonical_size += (-canonical_size) % _USTAR_RECORD_BYTES
    if archive_size != canonical_size:
        _fail("publication archive canonical size differs")
    by_path = {cast(str, item["path"]): item for item in members}
    final = by_path.get(staging.EXTERNAL_STAGING_FINAL_MANIFEST_PATH)
    base = by_path.get(staging.EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH)
    if (
        final is None
        or final["sha256"] != stage["full_file_sha256"]
        or final["size_bytes"] != stage["size_bytes"]
        or final["mode"] != "0444"
        or final["provenance"] != "final_staging_manifest_self"
        or base is None
        or base["sha256"] != source["full_file_sha256"]
        or base["size_bytes"] != source["size_bytes"]
        or base["mode"] != "0444"
        or base["provenance"] != "relocated_exact_materializer_v2_manifest"
    ):
        _fail("publication embedded manifest member bindings differ")
    supplied_body = _require_sha256(receipt["receipt_body_sha256"], label="receipt body")
    body = copy.deepcopy(receipt)
    del body["receipt_body_sha256"]
    if supplied_body != _sha256(_canonical_json(body, newline=False)):
        _fail("publication receipt body digest differs")
    return copy.deepcopy(receipt)


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


def _inode_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Identity fields stable across owned entry creation and removal."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


def _pread_exact(descriptor: int, size: int, offset: int, *, label: str) -> bytes:
    chunks: list[bytes] = []
    cursor = offset
    remaining = size
    while remaining:
        try:
            chunk = os.pread(descriptor, min(remaining, _READ_CHUNK_BYTES), cursor)
        except InterruptedError:
            continue
        if not chunk:
            _fail(f"{label} ended early")
        chunks.append(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _hash_fd(descriptor: int, size: int, *, label: str) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = _pread_exact(
            descriptor,
            min(_READ_CHUNK_BYTES, size - offset),
            offset,
            label=label,
        )
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, size):
        _fail(f"{label} exceeds its opened size")
    return digest.hexdigest()


def _verify_external_source_ustar_fd(
    descriptor: int,
    *,
    expected_size: int,
    expected_sha256: str,
    members: Sequence[Mapping[str, Any]],
) -> tuple[bytes, bytes]:
    """Independently replay exact USTAR headers, bytes, padding, and manifests."""

    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size != expected_size
        or not _USTAR_RECORD_BYTES <= expected_size <= _MAX_ARCHIVE_BYTES
        or expected_size % _USTAR_RECORD_BYTES
    ):
        _fail("external source USTAR descriptor metadata differs")
    offset = 0
    base_raw: bytes | None = None
    stage_raw: bytes | None = None
    for record in members:
        path = cast(str, record["path"])
        size = cast(int, record["size_bytes"])
        mode = 0o444 if record["mode"] == "0444" else 0o555
        header = _pread_exact(
            descriptor,
            _USTAR_BLOCK_BYTES,
            offset,
            label=f"USTAR header {path}",
        )
        if not hmac.compare_digest(header, _canonical_ustar_header(path, size, mode)):
            _fail(f"external source USTAR header is noncanonical: {path}")
        offset += _USTAR_BLOCK_BYTES
        digest = hashlib.sha256()
        retained_chunks: list[bytes] | None = (
            []
            if path
            in {
                staging.EXTERNAL_STAGING_FINAL_MANIFEST_PATH,
                staging.EXTERNAL_STAGING_MATERIALIZER_MANIFEST_PATH,
            }
            else None
        )
        remaining = size
        while remaining:
            chunk = _pread_exact(
                descriptor,
                min(_READ_CHUNK_BYTES, remaining),
                offset,
                label=f"USTAR payload {path}",
            )
            digest.update(chunk)
            if retained_chunks is not None:
                retained_chunks.append(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        if digest.hexdigest() != record["sha256"]:
            _fail(f"external source USTAR payload digest differs: {path}")
        if retained_chunks is not None:
            retained_raw = b"".join(retained_chunks)
            if path == staging.EXTERNAL_STAGING_FINAL_MANIFEST_PATH:
                stage_raw = retained_raw
            else:
                base_raw = retained_raw
        padding = (-size) % _USTAR_BLOCK_BYTES
        if padding and any(
            _pread_exact(
                descriptor,
                padding,
                offset,
                label=f"USTAR padding {path}",
            )
        ):
            _fail(f"external source USTAR padding is nonzero: {path}")
        offset += padding
    if any(
        _pread_exact(
            descriptor,
            2 * _USTAR_BLOCK_BYTES,
            offset,
            label="USTAR end blocks",
        )
    ):
        _fail("external source USTAR end blocks are nonzero")
    offset += 2 * _USTAR_BLOCK_BYTES
    final_size = offset + (-offset) % _USTAR_RECORD_BYTES
    if final_size != expected_size:
        _fail("external source USTAR record-padding length differs")
    tail = expected_size - offset
    if tail and any(_pread_exact(descriptor, tail, offset, label="USTAR record padding")):
        _fail("external source USTAR record padding is nonzero")
    actual_sha256 = _hash_fd(descriptor, expected_size, label="complete external source USTAR")
    after = os.fstat(descriptor)
    if (
        not hmac.compare_digest(actual_sha256, expected_sha256)
        or _stat_identity(before) != _stat_identity(after)
        or base_raw is None
        or stage_raw is None
    ):
        _fail("external source USTAR identity or embedded manifests differ")
    return base_raw, stage_raw


def _validate_embedded_manifests(
    receipt: Mapping[str, Any],
    *,
    base_raw: bytes,
    stage_raw: bytes,
) -> None:
    stage_binding = cast(dict[str, Any], receipt["staging_manifest"])
    source_binding = cast(dict[str, Any], receipt["external_source_manifest"])
    stage_manifest = staging.parse_external_staging_manifest(
        stage_raw,
        expected_manifest_sha256=cast(str, stage_binding["full_file_sha256"]),
    )
    if (
        stage_binding
        != {
            "body_sha256": stage_manifest["manifest_body_sha256"],
            "full_file_sha256": _sha256(stage_raw),
            "schema_version": stage_manifest["schema_version"],
            "size_bytes": len(stage_raw),
            "status": stage_manifest["status"],
        }
        or _complete_inventory(stage_raw, stage_manifest)
        != cast(dict[str, Any], receipt["archive"])["members"]
    ):
        _fail("embedded staging manifest does not reconstruct the receipt")
    try:
        base_manifest = materialization.parse_matched_v3_external_materialization_manifest(
            base_raw,
            expected_manifest_sha256=cast(str, source_binding["full_file_sha256"]),
        )
    except (AssertionError, materialization.ExternalMaterializationError) as exc:
        raise ForagerMatchedV3ExternalSourcePublicationError(
            "embedded materializer manifest failed exact replay"
        ) from exc
    if _source_binding(base_raw, base_manifest) != source_binding:
        _fail("embedded materializer manifest does not reconstruct the receipt")


def _build_receipt(
    retained: staging.RetainedExternalStagingBundle,
) -> tuple[bytes, str, dict[str, Any]]:
    try:
        stage_manifest = retained.reverify()
        stage_raw = retained.manifest_bytes
        stage_sha256 = retained.manifest_sha256
        archive_sha256 = retained.archive_sha256
        archive_size = retained.archive_size_bytes
        descriptors = retained.subprocess_pass_fds
    except staging.ForagerMatchedV3ExternalStagingError as exc:
        raise ForagerMatchedV3ExternalSourcePublicationError(
            "retained external source capability failed prepublication replay"
        ) from exc
    if (
        type(descriptors) is not tuple
        or len(descriptors) != 1
        or type(descriptors[0]) is not int
        or descriptors[0] <= 2
        or _sha256(stage_raw) != stage_sha256
    ):
        _fail("retained external source capability descriptor facts differ")
    inventory = _complete_inventory(stage_raw, stage_manifest)
    base_raw, embedded_stage_raw = _verify_external_source_ustar_fd(
        descriptors[0],
        expected_size=archive_size,
        expected_sha256=archive_sha256,
        members=inventory,
    )
    if embedded_stage_raw != stage_raw:
        _fail("retained archive embeds a different staging manifest")
    base_binding = cast(dict[str, Any], stage_manifest["base_materialization"])
    if (
        len(base_raw) != base_binding["manifest_size_bytes"]
        or _sha256(base_raw) != base_binding["manifest_sha256"]
    ):
        _fail("retained archive embeds a different materializer manifest")
    try:
        base_manifest = materialization.parse_matched_v3_external_materialization_manifest(
            base_raw,
            expected_manifest_sha256=cast(str, base_binding["manifest_sha256"]),
        )
    except (AssertionError, materialization.ExternalMaterializationError) as exc:
        raise ForagerMatchedV3ExternalSourcePublicationError(
            "retained archive materializer manifest failed exact replay"
        ) from exc
    raw = _receipt_bytes(
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size,
        inventory=inventory,
        stage_raw=stage_raw,
        stage_manifest=stage_manifest,
        base_raw=base_raw,
        base_manifest=base_manifest,
    )
    receipt_sha256 = _sha256(raw)
    receipt = parse_external_source_publication_receipt(
        raw,
        expected_file_sha256=receipt_sha256,
    )
    _validate_embedded_manifests(
        receipt,
        base_raw=base_raw,
        stage_raw=stage_raw,
    )
    return raw, receipt_sha256, receipt


def _open_directory(path: Path, *, label: str) -> _OpenDirectory:
    if type(path) is not type(Path()) or not path.is_absolute():
        _fail(f"{label} must be one exact absolute pathlib.Path")
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ForagerMatchedV3ExternalSourcePublicationError(
            f"{label} is not one exact non-symlink directory"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        after = path.lstat()
        identity = _directory_identity(opened)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _directory_identity(before) != identity
            or _directory_identity(after) != identity
        ):
            _fail(f"{label} identity changed while opened")
        return _OpenDirectory(path=path, descriptor=descriptor, identity=identity)
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_at(
    parent: _OpenDirectory,
    name: str,
    *,
    label: str,
) -> _OpenDirectory:
    try:
        before = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent.descriptor,
        )
    except OSError as exc:
        raise ForagerMatchedV3ExternalSourcePublicationError(
            f"{label} is not one exact non-symlink directory"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        after = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        identity = _directory_identity(opened)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _directory_identity(before) != identity
            or _directory_identity(after) != identity
        ):
            _fail(f"{label} identity changed while opened")
        return _OpenDirectory(
            path=parent.path / name,
            descriptor=descriptor,
            identity=identity,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _directory_stable(directory: _OpenDirectory) -> bool:
    try:
        return _directory_identity(os.fstat(directory.descriptor)) == directory.identity and (
            _directory_identity(directory.path.lstat()) == directory.identity
        )
    except OSError:
        return False


def _name_matches_directory(parent: _OpenDirectory, name: str, child: _OpenDirectory) -> bool:
    try:
        located = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        opened = os.fstat(child.descriptor)
    except OSError:
        return False
    return stat.S_ISDIR(located.st_mode) and _inode_identity(located) == _inode_identity(opened)


def _write_all(descriptor: int, raw: bytes) -> None:
    remaining = memoryview(raw)
    while remaining:
        try:
            written = os.write(descriptor, remaining)
        except InterruptedError:
            continue
        if written <= 0:
            _fail("external source publication write made no progress")
        remaining = remaining[written:]


def _write_new_file(directory_fd: int, name: str, raw: bytes) -> None:
    if type(raw) is not bytes or not raw:
        _fail("external source publication files must be nonempty exact bytes")
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        _write_all(descriptor, raw)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        located = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != len(raw)
            or stat.S_IMODE(opened.st_mode) != 0o444
            or _stat_identity(opened) != _stat_identity(located)
        ):
            _fail(f"staged external source publication file changed: {name}")
    except ForagerMatchedV3ExternalSourcePublicationError:
        raise
    except OSError as exc:
        raise ForagerMatchedV3ExternalSourcePublicationError(
            f"cannot stage external source publication file {name}"
        ) from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _copy_retained_archive(
    retained: staging.RetainedExternalStagingBundle,
    directory_fd: int,
) -> None:
    try:
        descriptors = retained.subprocess_pass_fds
        expected_size = retained.archive_size_bytes
        expected_sha256 = retained.archive_sha256
    except staging.ForagerMatchedV3ExternalStagingError as exc:
        raise ForagerMatchedV3ExternalSourcePublicationError(
            "retained external source capability is inactive"
        ) from exc
    if type(descriptors) is not tuple or len(descriptors) != 1:
        _fail("retained external source descriptor fact differs")
    source = descriptors[0]
    source_before = _stat_identity(os.fstat(source))
    output = -1
    try:
        output = os.open(
            EXTERNAL_SOURCE_ARCHIVE_FILENAME,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        digest = hashlib.sha256()
        offset = 0
        while offset < expected_size:
            chunk = _pread_exact(
                source,
                min(_READ_CHUNK_BYTES, expected_size - offset),
                offset,
                label="retained external source archive",
            )
            _write_all(output, chunk)
            digest.update(chunk)
            offset += len(chunk)
        if os.pread(source, 1, expected_size):
            _fail("retained external source archive exceeds its declared size")
        if digest.hexdigest() != expected_sha256:
            _fail("copied external source digest differs from retained capability")
        os.fchmod(output, 0o444)
        os.fsync(output)
        opened = os.fstat(output)
        located = os.stat(
            EXTERNAL_SOURCE_ARCHIVE_FILENAME,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != expected_size
            or stat.S_IMODE(opened.st_mode) != 0o444
            or _stat_identity(opened) != _stat_identity(located)
            or _stat_identity(os.fstat(source)) != source_before
        ):
            _fail("copied external source filesystem identity differs")
    except ForagerMatchedV3ExternalSourcePublicationError:
        raise
    except OSError as exc:
        raise ForagerMatchedV3ExternalSourcePublicationError(
            "cannot copy retained external source archive"
        ) from exc
    finally:
        if output >= 0:
            try:
                os.close(output)
            except OSError:
                pass


def _open_hashed_file_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    maximum: int,
) -> tuple[int, int, str, tuple[int, ...]]:
    descriptor = -1
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or not 1 <= before.st_size <= maximum
        ):
            _fail(f"{label} must be one bounded read-only single-link file")
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        identity = _stat_identity(opened)
        if identity != _stat_identity(before):
            _fail(f"{label} changed while opened")
        digest = _hash_fd(descriptor, opened.st_size, label=label)
        after = os.fstat(descriptor)
        located = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if identity != _stat_identity(after) or identity != _stat_identity(located):
            _fail(f"{label} changed while hashed")
        return descriptor, opened.st_size, digest, identity
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _reverify_file(
    descriptor: int,
    *,
    expected_identity: tuple[int, ...],
    expected_sha256: str,
    label: str,
) -> None:
    before = os.fstat(descriptor)
    if _stat_identity(before) != expected_identity:
        _fail(f"{label} descriptor identity changed")
    digest = _hash_fd(descriptor, before.st_size, label=label)
    if _stat_identity(os.fstat(descriptor)) != expected_identity or digest != expected_sha256:
        _fail(f"{label} descriptor bytes changed")


def _validate_published_directory_fd(
    directory_fd: int,
    *,
    expected_receipt_sha256: str,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    expected_receipt = _require_sha256(expected_receipt_sha256, label="published receipt")
    expected_archive = _require_sha256(expected_archive_sha256, label="published archive")
    directory_before = _stat_identity(os.fstat(directory_fd))
    if (
        not stat.S_ISDIR(os.fstat(directory_fd).st_mode)
        or stat.S_IMODE(os.fstat(directory_fd).st_mode) != 0o555
        or sorted(os.listdir(directory_fd))
        != [EXTERNAL_SOURCE_ARCHIVE_FILENAME, EXTERNAL_SOURCE_RECEIPT_FILENAME]
    ):
        _fail("published external source directory structure differs")
    receipt_fd = -1
    archive_fd = -1
    try:
        receipt_fd, receipt_size, receipt_sha256, receipt_identity = _open_hashed_file_at(
            directory_fd,
            EXTERNAL_SOURCE_RECEIPT_FILENAME,
            label="published external source receipt",
            maximum=_MAX_RECEIPT_BYTES,
        )
        if receipt_sha256 != expected_receipt:
            _fail("published external source receipt identity differs")
        receipt_raw = _pread_exact(
            receipt_fd,
            receipt_size,
            0,
            label="published external source receipt",
        )
        receipt = parse_external_source_publication_receipt(
            receipt_raw,
            expected_file_sha256=expected_receipt,
        )
        archive_binding = cast(dict[str, Any], receipt["archive"])
        if archive_binding["sha256"] != expected_archive:
            _fail("published receipt archive digest differs from directory identity")
        archive_fd, archive_size, archive_sha256, archive_identity = _open_hashed_file_at(
            directory_fd,
            EXTERNAL_SOURCE_ARCHIVE_FILENAME,
            label="published external source archive",
            maximum=_MAX_ARCHIVE_BYTES,
        )
        if archive_size != archive_binding["size_bytes"] or archive_sha256 != expected_archive:
            _fail("published external source archive identity differs")
        base_raw, stage_raw = _verify_external_source_ustar_fd(
            archive_fd,
            expected_size=archive_size,
            expected_sha256=archive_sha256,
            members=cast(list[dict[str, Any]], archive_binding["members"]),
        )
        _validate_embedded_manifests(
            receipt,
            base_raw=base_raw,
            stage_raw=stage_raw,
        )
        _reverify_file(
            receipt_fd,
            expected_identity=receipt_identity,
            expected_sha256=expected_receipt,
            label="published external source receipt",
        )
        _reverify_file(
            archive_fd,
            expected_identity=archive_identity,
            expected_sha256=expected_archive,
            label="published external source archive",
        )
        if _stat_identity(os.fstat(directory_fd)) != directory_before or sorted(
            os.listdir(directory_fd)
        ) != [EXTERNAL_SOURCE_ARCHIVE_FILENAME, EXTERNAL_SOURCE_RECEIPT_FILENAME]:
            _fail("published external source directory changed during replay")
        return receipt
    finally:
        for descriptor in (receipt_fd, archive_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _rename_new_only(directory_fd: int, source: str, target: str) -> None:
    renameat2 = cast(Any, getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None))
    if renameat2 is None:
        _fail("renameat2 is required for atomic external source publication")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    if renameat2(directory_fd, os.fsencode(source), directory_fd, os.fsencode(target), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(target)
    raise OSError(error_number, os.strerror(error_number), target)


def _cleanup_owned_staging(
    namespace: _OpenDirectory,
    name: str,
    opened: _OpenDirectory,
) -> bool:
    if not _name_matches_directory(namespace, name, opened):
        try:
            os.stat(name, dir_fd=namespace.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return False
    try:
        observed = sorted(os.listdir(opened.descriptor))
        allowed = {EXTERNAL_SOURCE_ARCHIVE_FILENAME, EXTERNAL_SOURCE_RECEIPT_FILENAME}
        if any(item not in allowed for item in observed):
            return False
        identities: dict[str, tuple[int, ...]] = {}
        for item in observed:
            metadata = os.stat(item, dir_fd=opened.descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) not in {0o600, 0o444}
            ):
                return False
            identities[item] = _stat_identity(metadata)
        os.fchmod(opened.descriptor, 0o700)
        for item in observed:
            current = os.stat(item, dir_fd=opened.descriptor, follow_symlinks=False)
            if _stat_identity(current) != identities[item]:
                return False
            os.unlink(item, dir_fd=opened.descriptor)
        os.fsync(opened.descriptor)
        if not _name_matches_directory(namespace, name, opened) or os.listdir(opened.descriptor):
            return False
        os.rmdir(name, dir_fd=namespace.descriptor)
        os.fsync(namespace.descriptor)
        try:
            os.stat(name, dir_fd=namespace.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return True
        return False
    except OSError:
        return False


def _open_namespace(publication_root: Path) -> tuple[_OpenDirectory, _OpenDirectory]:
    root = _open_directory(publication_root, label="external source publication root")
    namespace: _OpenDirectory | None = None
    try:
        root_metadata = os.fstat(root.descriptor)
        if root_metadata.st_uid != os.geteuid() or stat.S_IMODE(root_metadata.st_mode) & 0o022:
            _fail("publication root must be effective-UID-owned and not group/world writable")
        try:
            os.mkdir("sha256", 0o755, dir_fd=root.descriptor)
            os.fsync(root.descriptor)
        except FileExistsError:
            pass
        namespace = _open_directory_at(root, "sha256", label="external source sha256 namespace")
        namespace_metadata = os.fstat(namespace.descriptor)
        if (
            namespace_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(namespace_metadata.st_mode) & 0o022
        ):
            _fail("publication namespace must be effective-UID-owned and not group/world writable")
        if not _directory_stable(root) or not _directory_stable(namespace):
            _fail("publication root or namespace changed while opened")
        return root, namespace
    except BaseException:
        if namespace is not None:
            os.close(namespace.descriptor)
        os.close(root.descriptor)
        raise


def _create_staging(namespace: _OpenDirectory, digest: str) -> tuple[str, _OpenDirectory]:
    for _attempt in range(_STAGING_ATTEMPTS):
        name = f".staging-{digest}-{os.getpid()}-{secrets.token_hex(16)}"
        try:
            os.mkdir(name, 0o700, dir_fd=namespace.descriptor)
        except FileExistsError:
            continue
        except OSError as exc:
            raise ForagerMatchedV3ExternalSourcePublicationError(
                "cannot create external source publication staging directory"
            ) from exc
        opened: _OpenDirectory | None = None
        try:
            opened = _open_directory_at(
                namespace,
                name,
                label="external source publication staging directory",
            )
            metadata = os.fstat(opened.descriptor)
            if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
                _fail("external source staging ownership or mode differs")
            return name, opened
        except BaseException as exc:
            if opened is not None:
                if not _cleanup_owned_staging(namespace, name, opened):
                    exc.add_note(f"owned partial staging directory may remain: {name}")
                os.close(opened.descriptor)
            else:
                try:
                    os.rmdir(name, dir_fd=namespace.descriptor)
                    os.fsync(namespace.descriptor)
                except OSError as cleanup_exc:
                    exc.add_note(
                        f"owned empty staging directory may remain ({name}): {cleanup_exc}"
                    )
            raise
    _fail("cannot allocate a unique external source staging directory")


def publish_matched_v3_external_source(
    retained: staging.RetainedExternalStagingBundle,
    publication_root: Path,
    *,
    authorize_non_evidence_publication: bool,
) -> PublishedMatchedV3ExternalSource:
    """Publish exact USTAR bytes at ``sha256/<archive-sha256>`` new-only."""

    if authorize_non_evidence_publication is not True:
        _fail("external source publication requires explicit non-evidence authorization")
    if type(retained) is not staging.RetainedExternalStagingBundle:
        _fail("publication requires one exact live external staging capability")
    receipt_raw, receipt_sha256, receipt = _build_receipt(retained)
    archive = cast(dict[str, Any], receipt["archive"])
    archive_sha256 = cast(str, archive["sha256"])
    root, namespace = _open_namespace(publication_root)
    staging_name = ""
    staging_directory: _OpenDirectory | None = None
    destination = namespace.path / archive_sha256
    committed = False
    try:
        staging_name, staging_directory = _create_staging(namespace, archive_sha256)
        _copy_retained_archive(retained, staging_directory.descriptor)
        _write_new_file(
            staging_directory.descriptor,
            EXTERNAL_SOURCE_RECEIPT_FILENAME,
            receipt_raw,
        )
        try:
            retained.reverify()
        except staging.ForagerMatchedV3ExternalStagingError as exc:
            raise ForagerMatchedV3ExternalSourcePublicationError(
                "retained capability failed post-copy replay"
            ) from exc
        os.fchmod(staging_directory.descriptor, 0o555)
        os.fsync(staging_directory.descriptor)
        _validate_published_directory_fd(
            staging_directory.descriptor,
            expected_receipt_sha256=receipt_sha256,
            expected_archive_sha256=archive_sha256,
        )
        os.fsync(namespace.descriptor)
        try:
            _rename_new_only(namespace.descriptor, staging_name, archive_sha256)
        except BaseException as exc:
            destination_matches = _name_matches_directory(
                namespace, archive_sha256, staging_directory
            )
            source_matches = _name_matches_directory(namespace, staging_name, staging_directory)
            if destination_matches or not source_matches:
                raise PublishedMatchedV3ExternalSourceUncertainError(
                    destination,
                    "exclusive move outcome is uncertain",
                    archive_sha256=archive_sha256,
                    receipt_sha256=receipt_sha256,
                ) from exc
            raise
        committed = True
        try:
            if not _name_matches_directory(namespace, archive_sha256, staging_directory):
                _fail("published destination differs from the verified staging inode")
            os.fsync(namespace.descriptor)
            _validate_published_directory_fd(
                staging_directory.descriptor,
                expected_receipt_sha256=receipt_sha256,
                expected_archive_sha256=archive_sha256,
            )
            if (
                not _name_matches_directory(namespace, archive_sha256, staging_directory)
                or not _directory_stable(root)
                or not _directory_stable(namespace)
            ):
                _fail("published destination changed during final replay")
        except BaseException as exc:
            raise PublishedMatchedV3ExternalSourceUncertainError(
                destination,
                "final durability or content replay is uncertain",
                archive_sha256=archive_sha256,
                receipt_sha256=receipt_sha256,
            ) from exc
        return PublishedMatchedV3ExternalSource(
            directory=destination,
            archive=destination / EXTERNAL_SOURCE_ARCHIVE_FILENAME,
            receipt=destination / EXTERNAL_SOURCE_RECEIPT_FILENAME,
            archive_sha256=archive_sha256,
            receipt_sha256=receipt_sha256,
        )
    except PublishedMatchedV3ExternalSourceUncertainError:
        raise
    except FileExistsError as exc:
        cleanup_succeeded = staging_directory is None or _cleanup_owned_staging(
            namespace, staging_name, staging_directory
        )
        collision = FileExistsError(
            f"refusing to overwrite external source publication {archive_sha256}"
        )
        if not cleanup_succeeded:
            collision.add_note(f"owned partial staging directory may remain: {staging_name}")
        raise collision from exc
    except BaseException as exc:
        destination_matches = staging_directory is not None and _name_matches_directory(
            namespace, archive_sha256, staging_directory
        )
        source_matches = (
            staging_directory is not None
            and bool(staging_name)
            and _name_matches_directory(namespace, staging_name, staging_directory)
        )
        if (
            committed
            or destination_matches
            or (staging_directory is not None and not source_matches)
        ):
            raise PublishedMatchedV3ExternalSourceUncertainError(
                destination,
                "final publication state is uncertain",
                archive_sha256=archive_sha256,
                receipt_sha256=receipt_sha256,
            ) from exc
        cleanup_succeeded = staging_directory is None or _cleanup_owned_staging(
            namespace, staging_name, staging_directory
        )
        if isinstance(exc, ForagerMatchedV3ExternalSourcePublicationError):
            if not cleanup_succeeded:
                exc.add_note(f"owned partial staging directory may remain: {staging_name}")
            raise
        failure = ForagerMatchedV3ExternalSourcePublicationError(
            "external source publication failed before atomic commit"
        )
        if not cleanup_succeeded:
            failure.add_note(f"owned partial staging directory may remain: {staging_name}")
        raise failure from exc
    finally:
        for opened in (staging_directory, namespace, root):
            if opened is not None:
                try:
                    os.close(opened.descriptor)
                except OSError:
                    pass


def validate_published_matched_v3_external_source(
    directory: Path,
    *,
    expected_receipt_sha256: str,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    """Reopen and fully replay one exact, location-independent publication directory."""

    expected_archive = _require_sha256(expected_archive_sha256, label="published archive")
    if type(directory) is not type(Path()) or directory.name != expected_archive:
        _fail("published directory is not addressed by the expected archive digest")
    opened = _open_directory(directory, label="published external source directory")
    try:
        return _validate_published_directory_fd(
            opened.descriptor,
            expected_receipt_sha256=expected_receipt_sha256,
            expected_archive_sha256=expected_archive,
        )
    except ForagerMatchedV3ExternalSourcePublicationError:
        raise
    except OSError as exc:
        raise ForagerMatchedV3ExternalSourcePublicationError(
            "published external source filesystem replay failed"
        ) from exc
    finally:
        os.close(opened.descriptor)


__all__ = [
    "EXTERNAL_SOURCE_ARCHIVE_FILENAME",
    "EXTERNAL_SOURCE_PUBLICATION_CONTRACT_DESCRIPTOR_SCHEMA_VERSION",
    "EXTERNAL_SOURCE_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256",
    "EXTERNAL_SOURCE_PUBLICATION_RECEIPT_SCHEMA_VERSION",
    "EXTERNAL_SOURCE_PUBLICATION_STATUS",
    "EXTERNAL_SOURCE_RECEIPT_FILENAME",
    "ForagerMatchedV3ExternalSourcePublicationError",
    "PublishedMatchedV3ExternalSource",
    "PublishedMatchedV3ExternalSourceUncertainError",
    "canonical_external_source_publication_contract_descriptor_bytes",
    "external_source_publication_contract_descriptor",
    "parse_external_source_publication_contract_descriptor",
    "parse_external_source_publication_receipt",
    "publish_matched_v3_external_source",
    "validate_published_matched_v3_external_source",
]
