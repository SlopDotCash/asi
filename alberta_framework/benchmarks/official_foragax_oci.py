"""Audited build, launch, inspection, and qualification tooling for OCI v4.

"v4" names the launcher contract ``oci-read-only-stdout-tar-v4`` defined in
:mod:`alberta_framework.benchmarks.official_foragax` — run results leave the
container only as a tar stream on stdout — together with the matching
``alberta.official_foragax.oci_build.v4`` build-spec schema.  The lifecycle
maps onto the CLI subcommands, in file order:

1. **prepare** — :func:`prepare_build_context` checks every input against
   the audited pins below (source commit and Git tree, digest-pinned CUDA
   base image, uv binary, dependency lock) and atomically materializes a
   path-neutral Docker build context whose Dockerfile runs every stage with
   ``--network=none`` and carries its own canonical build attestation.
2. **archive-cache** — :func:`create_regular_cache_archive` rewrites the uv
   cache into one canonical USTAR archive so the cache identity is
   byte-stable across hosts.
3. **build** — :func:`build_image` runs the digest-pinned BuildKit frontend
   over a prepared context.
4. **inspect** — :func:`inspect_image` regenerates and verifies the image
   config, rootfs inventory, native-runtime inventory, and SBOM against the
   expected build spec.
5. **cpu-probe / emit-launch** — read-only, networkless, cap-dropped
   ``docker run`` command lines addressed by config digest only; GPU
   launches bind the host driver explicitly via
   :class:`DriverLaunchContract` (the image itself sets
   ``NVIDIA_VISIBLE_DEVICES=void`` to refuse implicit runtime injection).
6. **qualify** — :func:`qualify_v4_runs` seals two independent runs of the
   same workload into one determinism-evidence envelope.

The production trust and endorsement descriptors are intentionally outside
this utility.  It emits candidate attestations that a later human-reviewed
descriptor update may consume; it never grants official status itself.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import io
import json
import math
import os
import posixpath
import re
import shlex
import shutil
import sqlite3
import stat
import subprocess
import sys
import tarfile
import tomllib
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

from alberta_framework.benchmarks.official_foragax import (
    OFFICIAL_FORAGAX_AUDIT_COMMIT,
    OFFICIAL_FORAGAX_CUBLAS_WORKSPACE_CONFIG,
    OFFICIAL_FORAGAX_DETERMINISM_QUALIFICATION_SCHEMA,
    OFFICIAL_FORAGAX_GPU_XLA_FLAGS,
    OFFICIAL_FORAGAX_NATIVE_RUNTIME_INVENTORY_HASH_SCHEME,
    OFFICIAL_FORAGAX_OCI_LAUNCHER_CONTRACT,
    OFFICIAL_FORAGAX_QUALIFICATION_WORKLOAD_SCHEMA,
    OFFICIAL_FORAGAX_RESULTS_DB_COLUMNS,
    OFFICIAL_FORAGAX_XLA_PYTHON_CLIENT_PREALLOCATE,
)

BUILD_SPEC_SCHEMA = "alberta.official_foragax.oci_build.v4"
BUILD_REPORT_SCHEMA = "alberta.official_foragax.oci_image_report.v1"
DEBIAN_BUNDLE_SCHEMA = "alberta.official_foragax.debian_bundle.v1"
DEBIAN_INVENTORY_SCHEMA = "alberta.official_foragax.debian_inventory.v1"
NATIVE_INVENTORY_SCHEMA = "alberta.official_foragax.native_inventory.v1"
ROOTFS_INVENTORY_SCHEMA = "alberta.official_foragax.rootfs_inventory.v1"
QUALIFICATION_ENVELOPE_SCHEMA = (
    "alberta.official_foragax.determinism_evidence.v2"
)
QUALIFICATION_WORKLOAD_SCHEMA = OFFICIAL_FORAGAX_QUALIFICATION_WORKLOAD_SCHEMA
TREE_HASH_SCHEME = "canonical-entry-json+mode+size+bytes-v1"
# SHA-256 of the audited uv 0.9.24 release binary; ``prepare_build_context``
# refuses any other uv, pinning dependency resolution to one audited
# resolver build.
UV_BINARY_SHA256 = (
    "5c021e58e83d7fab046137a02e5f459df30955756afda74e86e38150fdc781c1"
)
DOCKERFILE_FRONTEND = (
    "docker/dockerfile:1.7@"
    "sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e"
)
SOURCE_ROOT = PurePosixPath("/opt/continual-foragax-agents")
RUNTIME_ROOT = PurePosixPath("/opt/alberta-runtime")
ATTESTATION_ROOT = PurePosixPath("/opt/alberta-attestations")
LAUNCHER_PATH = PurePosixPath("/opt/alberta/launcher")
PYTHON_EXECUTABLE = RUNTIME_ROOT / "bin/python"
# Audited identity pins enforced by ``prepare_build_context``: the
# digest-pinned CUDA base image, the Git tree SHA-1 reconstructed from the
# supplied source archive bytes, and the gitlink commit expected at each
# empty submodule placeholder.  Refreshing a pin is a source edit here that
# re-opens the audit — never a CLI option (``--base-image`` and
# ``--uv-binary-sha256`` merely default to the audited values; any other
# value is rejected).
_AUDITED_BASE_IMAGE = (
    "nvidia/cuda:12.8.1-base-ubuntu24.04@"
    "sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2"
)
_AUDITED_SOURCE_TREE_GIT_SHA1 = "a5ad878ac4be0567c43dfd9177471c4b5a910bfa"
_AUDITED_SOURCE_GITLINKS = {
    "continual-foragax-loss-of-plasticity": (
        "8880f3f241ec441e584416b61b0579fca3bc1ef4"
    ),
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_OCI_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_OCI_REFERENCE_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9._/-]*(?::[A-Za-z0-9._-]+)?@sha256:[0-9a-f]{64}$"
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_DEBIAN_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9+.-]*$")
# Substrings that only exist in build-time locations: ``/input/`` is the
# read-only bind-mount target for every build input in the generated
# Dockerfile, ``/build/`` and ``/root/.cache`` are conventional builder
# workspace and cache homes, and ``file://`` marks content addressed by
# local path rather than by digest.  Finding any of these in the build
# attestation or in the image environment means a builder-local location
# leaked into the artifact — its identity would then depend on where it was
# built — so both checks fail closed.
_FORBIDDEN_ATTESTATION_TEXT = (
    "/build/",
    "/input/",
    "/root/.cache",
    "file://",
)


class OfficialForagaxOciError(ValueError):
    """Raised when an OCI build or qualification contract fails closed."""


@dataclasses.dataclass(frozen=True)
class OciBuildInputs:
    """Caller-supplied immutable inputs for a production candidate context."""

    source_archive: Path
    source_archive_sha256: str
    dependency_lock: Path
    dependency_lock_sha256: str
    source_commit: str
    source_tree_git_sha1: str
    base_image: str
    uv_binary: Path
    uv_binary_sha256: str
    uv_cache_archive: Path
    uv_cache_archive_sha256: str
    debian_bundle: Path
    debian_manifest: Path
    output_context: Path


@dataclasses.dataclass(frozen=True)
class PreparedOciBuild:
    """Identity of one completely materialized Docker build context."""

    context: Path
    build_spec: Mapping[str, Any]
    build_spec_sha256: str
    dockerfile_sha256: str
    launcher_sha256: str


@dataclasses.dataclass(frozen=True)
class DriverLaunchContract:
    """Exact host-controlled NVIDIA device and read-only library binding."""

    driver_host_path: str
    driver_container_path: str
    device_paths: tuple[str, ...]
    device_indices: tuple[int, ...]
    cuda_wheel_library_paths: tuple[str, ...]
    driver_user_library_paths: tuple[str, ...]


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return encoded + (b"\n" if newline else b"")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise OfficialForagaxOciError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise OfficialForagaxOciError(f"{label} must be a lowercase SHA-256")
    return value


def _require_exact_str(name: object, value: object) -> str:
    if type(name) is not str:
        raise OfficialForagaxOciError("name must be an exact string")
    if type(value) is not str:
        raise OfficialForagaxOciError(f"{name} must be an exact string")
    return value


def _strict_json_bytes(contents: bytes, *, label: str) -> Any:
    host_label = _require_exact_str("label", label)
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            host_key = _require_exact_str("key", key)
            if host_key in result:
                raise OfficialForagaxOciError(
                    f"{host_label} repeats object key"
                )
            result[host_key] = value
        return result

    def constant(value: str) -> Any:
        _require_exact_str("value", value)
        raise OfficialForagaxOciError(
            f"{host_label} contains non-finite constant"
        )

    def parse_float(value: str) -> float:
        host_value = _require_exact_str("value", value)
        parsed = float(host_value)
        if not math.isfinite(parsed):
            raise OfficialForagaxOciError(
                f"{label} contains non-finite JSON number {value}"
            )
        return parsed

    try:
        return json.loads(
            contents,
            object_pairs_hook=pairs,
            parse_constant=constant,
            parse_float=parse_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfficialForagaxOciError(f"{label} is not strict JSON") from exc


def _strict_json_file(path: Path, *, label: str) -> Any:
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise OfficialForagaxOciError(f"cannot read {label}: {exc}") from exc
    return _strict_json_bytes(contents, label=label)


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise OfficialForagaxOciError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _safe_relative(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != value
        or "." in path.parts
        or ".." in path.parts
        or "\x00" in value
    ):
        raise OfficialForagaxOciError(
            f"{label} is not a canonical relative path"
        )
    return path


def _verify_hash(path: Path, expected: str, *, label: str) -> None:
    _require_sha256(expected, label=f"{label} expected digest")
    actual = _sha256(path)
    if actual != expected:
        raise OfficialForagaxOciError(
            f"{label} SHA-256 differs: expected {expected}, observed {actual}"
        )


def _source_archive_identity(
    archive_path: Path,
    *,
    source_commit: str,
    dependency_lock: bytes,
) -> dict[str, Any]:
    """Validate the deterministic PAX stream emitted by ``git archive``."""
    if _COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise OfficialForagaxOciError("source commit must be a full Git SHA-1")
    try:
        size = archive_path.stat().st_size
    except OSError as exc:
        raise OfficialForagaxOciError("source archive cannot be statted") from exc
    if size < 1536 or size > 2 * 1024**3 or size % 512:
        raise OfficialForagaxOciError(
            "source archive size is outside the deterministic tar bound"
        )
    entries: list[dict[str, Any]] = []
    names: list[str] = []
    mtimes: set[int] = set()
    lock_bytes: bytes | None = None
    git_directories = {""}
    git_children: dict[
        str,
        list[tuple[bytes, bytes, bytes, bool]],
    ] = {"": []}
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            if archive.pax_headers != {"comment": source_commit}:
                raise OfficialForagaxOciError(
                    "source archive does not bind the requested Git commit"
                )
            members = archive.getmembers()
            if not members or len(members) > 100_000:
                raise OfficialForagaxOciError(
                    "source archive member count is outside its bound"
                )
            for member in members:
                relative = _safe_relative(
                    member.name,
                    label="source archive member",
                )
                relative_name = relative.as_posix()
                parent = (
                    ""
                    if relative.parent.as_posix() == "."
                    else relative.parent.as_posix()
                )
                try:
                    basename = relative.name.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise OfficialForagaxOciError(
                        f"source path is not UTF-8: {member.name}"
                    ) from exc
                if (
                    member.uid != 0
                    or member.gid != 0
                    or member.uname != "root"
                    or member.gname != "root"
                    or member.pax_headers != {"comment": source_commit}
                    or member.sparse
                ):
                    raise OfficialForagaxOciError(
                        f"source archive metadata is not canonical: {member.name}"
                    )
                mtimes.add(int(member.mtime))
                mode = stat.S_IMODE(member.mode)
                if member.isdir() and member.type == tarfile.DIRTYPE:
                    if mode != 0o775:
                        raise OfficialForagaxOciError(
                            f"source directory mode is not Git-canonical: {member.name}"
                        )
                    entry: dict[str, Any] = {
                        "mode": mode,
                        "path": member.name,
                        "type": "directory",
                    }
                    git_directories.add(relative_name)
                    git_children.setdefault(relative_name, [])
                elif member.isfile() and member.type in {
                    tarfile.REGTYPE,
                    tarfile.AREGTYPE,
                }:
                    if mode not in {0o664, 0o775}:
                        raise OfficialForagaxOciError(
                            f"source file mode is not Git-canonical: {member.name}"
                        )
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise OfficialForagaxOciError(
                            f"source file cannot be read: {member.name}"
                        )
                    digest = hashlib.sha256()
                    git_digest = hashlib.sha1(usedforsecurity=False)
                    git_digest.update(
                        b"blob "
                        + str(member.size).encode("ascii")
                        + b"\x00"
                    )
                    captured = bytearray() if member.name == "uv.lock" else None
                    remaining = member.size
                    while remaining:
                        block = handle.read(min(1024 * 1024, remaining))
                        if not block:
                            raise OfficialForagaxOciError(
                                f"source file is truncated: {member.name}"
                            )
                        digest.update(block)
                        git_digest.update(block)
                        if captured is not None:
                            captured.extend(block)
                        remaining -= len(block)
                    if handle.read(1):
                        raise OfficialForagaxOciError(
                            f"source file exceeds header: {member.name}"
                        )
                    if captured is not None:
                        lock_bytes = bytes(captured)
                    entry = {
                        "mode": mode,
                        "path": member.name,
                        "sha256": digest.hexdigest(),
                        "size": member.size,
                        "type": "file",
                    }
                    git_children.setdefault(parent, []).append(
                        (
                            basename,
                            b"100755" if mode & 0o111 else b"100644",
                            git_digest.digest(),
                            False,
                        )
                    )
                elif member.issym() and member.type == tarfile.SYMTYPE:
                    target = PurePosixPath(member.linkname)
                    if (
                        mode != 0o777
                        or not member.linkname
                        or target.is_absolute()
                        or ".." in target.parts
                    ):
                        raise OfficialForagaxOciError(
                            f"source symlink is unsafe: {member.name}"
                        )
                    entry = {
                        "mode": mode,
                        "path": member.name,
                        "target": member.linkname,
                        "type": "symlink",
                    }
                    target_bytes = member.linkname.encode("utf-8")
                    git_digest = hashlib.sha1(usedforsecurity=False)
                    git_digest.update(
                        b"blob "
                        + str(len(target_bytes)).encode("ascii")
                        + b"\x00"
                        + target_bytes
                    )
                    git_children.setdefault(parent, []).append(
                        (basename, b"120000", git_digest.digest(), False)
                    )
                else:
                    raise OfficialForagaxOciError(
                        f"source archive contains forbidden member: {member.name}"
                    )
                names.append(member.name)
                entries.append(entry)
    except (OSError, tarfile.TarError) as exc:
        raise OfficialForagaxOciError(
            "source archive is not a valid deterministic Git archive"
        ) from exc
    git_sorted_names = [
        cast(str, entry["path"])
        for entry in sorted(
            entries,
            key=lambda entry: (
                cast(str, entry["path"])
                + ("/" if entry["type"] == "directory" else "")
            ).encode(),
        )
    ]
    if names != git_sorted_names or len(names) != len(set(names)):
        raise OfficialForagaxOciError(
            "source archive member order/set is not Git-deterministic"
        )
    if len(mtimes) != 1 or next(iter(mtimes)) < 1:
        raise OfficialForagaxOciError(
            "source archive does not use one deterministic commit timestamp"
        )
    required = {
        "pyproject.toml",
        "uv.lock",
        "src/continuing_main.py",
        "src/rtu_ppo.py",
    }
    if not required.issubset(names):
        raise OfficialForagaxOciError(
            "source archive lacks required project/entrypoint files"
        )
    if lock_bytes != dependency_lock:
        raise OfficialForagaxOciError(
            "caller-supplied lock differs from source archive uv.lock"
        )
    required_git_directories = set(git_children)
    required_git_directories.update(
        (
            ""
            if PurePosixPath(directory).parent.as_posix() == "."
            else PurePosixPath(directory).parent.as_posix()
        )
        for directory in git_directories
        if directory
    )
    missing_git_directories = required_git_directories - git_directories
    if missing_git_directories:
        raise OfficialForagaxOciError(
            "source archive omits explicit parent directories"
        )
    tree_sha1: bytes | None = None
    for directory in sorted(
        git_directories,
        key=lambda value: (value.count("/"), value.encode("utf-8")),
        reverse=True,
    ):
        children = git_children[directory]
        if len({name for name, _mode, _digest, _is_dir in children}) != len(
            children
        ):
            raise OfficialForagaxOciError(
                f"source Git tree repeats a child under {directory or '/'}"
            )
        gitlink = _AUDITED_SOURCE_GITLINKS.get(directory)
        if gitlink is not None:
            if children:
                raise OfficialForagaxOciError(
                    f"source Gitlink archive placeholder is not empty: {directory}"
                )
            parsed_directory = PurePosixPath(directory)
            parent = (
                ""
                if parsed_directory.parent.as_posix() == "."
                else parsed_directory.parent.as_posix()
            )
            git_children[parent].append(
                (
                    parsed_directory.name.encode("utf-8"),
                    b"160000",
                    bytes.fromhex(gitlink),
                    False,
                )
            )
            continue
        children.sort(
            key=lambda child: child[0] + (b"/" if child[3] else b"")
        )
        payload = b"".join(
            mode + b" " + name + b"\x00" + digest
            for name, mode, digest, _is_dir in children
        )
        tree_digest = hashlib.sha1(usedforsecurity=False)
        tree_digest.update(
            b"tree " + str(len(payload)).encode("ascii") + b"\x00" + payload
        )
        tree_sha1 = tree_digest.digest()
        if directory:
            parsed_directory = PurePosixPath(directory)
            parent = (
                ""
                if parsed_directory.parent.as_posix() == "."
                else parsed_directory.parent.as_posix()
            )
            git_children[parent].append(
                (
                    parsed_directory.name.encode("utf-8"),
                    b"40000",
                    tree_sha1,
                    True,
                )
            )
    if tree_sha1 is None:  # pragma: no cover - required files make this unreachable
        raise OfficialForagaxOciError("source Git tree cannot be reconstructed")
    identity = {"entries": entries, "hash_scheme": TREE_HASH_SCHEME}
    return {
        "commit_timestamp": next(iter(mtimes)),
        "entry_count": len(entries),
        "inventory_sha256": _json_sha256(identity),
        "source_tree_git_sha1": tree_sha1.hex(),
    }


def _validate_lock(contents: bytes) -> dict[str, str]:
    try:
        lock = tomllib.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise OfficialForagaxOciError("dependency lock is not valid TOML") from exc
    if lock.get("version") != 1 or lock.get("revision") != 3:
        raise OfficialForagaxOciError(
            "dependency lock format/revision is not the audited uv lock"
        )
    packages = lock.get("package")
    if type(packages) is not list:
        raise OfficialForagaxOciError("dependency lock package list is invalid")
    versions: dict[str, str] = {}
    for raw_package in packages:
        if type(raw_package) is not dict:
            raise OfficialForagaxOciError(
                "dependency lock contains a non-object package"
            )
        package = cast(dict[str, Any], raw_package)
        name = package.get("name")
        version = package.get("version")
        source = package.get("source")
        if type(name) is not str or type(version) is not str or type(source) is not dict:
            raise OfficialForagaxOciError(
                "dependency lock package identity is invalid"
            )
        if name in versions:
            raise OfficialForagaxOciError(
                f"dependency lock repeats package {name}"
            )
        versions[name] = version
        source_mapping = cast(dict[str, Any], source)
        if source_mapping == {"editable": "."}:
            if name != "continual-foragax-agents":
                raise OfficialForagaxOciError(
                    "dependency lock contains an unexpected editable package"
                )
            continue
        if set(source_mapping) != {"registry"}:
            raise OfficialForagaxOciError(
                f"dependency lock contains a non-registry package: {name}"
            )
        artifacts = []
        sdist = package.get("sdist")
        if type(sdist) is dict:
            artifacts.append(cast(dict[str, Any], sdist))
        wheels = package.get("wheels", [])
        if type(wheels) is not list:
            raise OfficialForagaxOciError(
                f"dependency lock wheel list is invalid: {name}"
            )
        artifacts.extend(
            cast(dict[str, Any], wheel)
            for wheel in wheels
            if type(wheel) is dict
        )
        if not artifacts or len(artifacts) != (
            (1 if type(sdist) is dict else 0) + len(wheels)
        ):
            raise OfficialForagaxOciError(
                f"dependency lock lacks hashed artifacts: {name}"
            )
        for artifact in artifacts:
            digest = artifact.get("hash")
            if (
                type(digest) is not str
                or not digest.startswith("sha256:")
                or _SHA256_PATTERN.fullmatch(digest.removeprefix("sha256:"))
                is None
            ):
                raise OfficialForagaxOciError(
                    f"dependency artifact is not SHA-256 pinned: {name}"
                )
    required = {
        "continual-foragax": "0.55.0",
        "imageio-ffmpeg": "0.6.0",
        "jax": "0.9.0.1",
        "jax-cuda12-pjrt": "0.9.0.1",
        "jax-cuda12-plugin": "0.9.0.1",
        "jaxlib": "0.9.0.1",
    }
    if any(versions.get(name) != version for name, version in required.items()):
        raise OfficialForagaxOciError(
            "dependency lock differs from the canonical Foragax/JAX CUDA 12 set"
        )
    return required


def validate_debian_manifest(
    manifest_path: Path,
    bundle: Path,
) -> dict[str, Any]:
    value = _strict_json_file(manifest_path, label="Debian bundle manifest")
    if type(value) is not dict:
        raise OfficialForagaxOciError(
            "Debian bundle manifest must contain an object"
        )
    manifest = cast(dict[str, Any], value)
    _exact_keys(
        manifest,
        {
            "architecture",
            "base_image",
            "installed_file_inventory_sha256",
            "packages",
            "python_executable",
            "python_executable_sha256",
            "repositories",
            "schema_version",
        },
        label="Debian bundle manifest",
    )
    if (
        manifest["schema_version"] != DEBIAN_BUNDLE_SCHEMA
        or manifest["architecture"] != "amd64"
        or manifest["base_image"] != _AUDITED_BASE_IMAGE
        or manifest["python_executable"] != "/usr/bin/python3.12"
    ):
        raise OfficialForagaxOciError(
            "Debian bundle platform/base/Python contract is unsupported"
        )
    _require_sha256(
        manifest["installed_file_inventory_sha256"],
        label="Debian installed-file inventory",
    )
    _require_sha256(
        manifest["python_executable_sha256"],
        label="Debian Python executable",
    )
    repositories = manifest["repositories"]
    if (
        type(repositories) is not list
        or not repositories
        or not all(
            type(repository) is str
            and repository.startswith(("http://", "https://"))
            and "\x00" not in repository
            for repository in repositories
        )
        or len(repositories) != len(set(cast(list[str], repositories)))
    ):
        raise OfficialForagaxOciError(
            "Debian repository provenance is invalid"
        )
    packages_value = manifest["packages"]
    if type(packages_value) is not list or not packages_value:
        raise OfficialForagaxOciError("Debian bundle package list is empty")
    names: list[str] = []
    filenames: list[str] = []
    for position, raw_package in enumerate(packages_value):
        if type(raw_package) is not dict:
            raise OfficialForagaxOciError(
                f"Debian package {position} is not an object"
            )
        package = cast(dict[str, Any], raw_package)
        _exact_keys(
            package,
            {"architecture", "filename", "package", "sha256", "version"},
            label=f"Debian package {position}",
        )
        name = package["package"]
        filename = package["filename"]
        architecture = package["architecture"]
        version = package["version"]
        if (
            type(name) is not str
            or _DEBIAN_NAME_PATTERN.fullmatch(name) is None
            or type(filename) is not str
            or PurePosixPath(filename).name != filename
            or not filename.endswith(".deb")
            or architecture not in {"all", "amd64"}
            or type(version) is not str
            or not version
        ):
            raise OfficialForagaxOciError(
                f"Debian package {position} identity is invalid"
            )
        digest = _require_sha256(
            package["sha256"],
            label=f"Debian package {name}",
        )
        _verify_hash(bundle / filename, digest, label=f"Debian package {name}")
        names.append(name)
        filenames.append(filename)
    if (
        names != sorted(names)
        or len(names) != len(set(names))
        or len(filenames) != len(set(filenames))
    ):
        raise OfficialForagaxOciError(
            "Debian packages must be unique and sorted by package name"
        )
    required = {
        "ca-certificates",
        "python3.12",
        "python3.12-minimal",
        "python3.12-venv",
    }
    if not required.issubset(names):
        raise OfficialForagaxOciError(
            "Debian bundle lacks the audited Python/CA packages"
        )
    actual_files = sorted(
        path.name
        for path in bundle.iterdir()
        if path.is_file() and not path.is_symlink()
    )
    if actual_files != sorted(filenames):
        raise OfficialForagaxOciError(
            "Debian bundle file set differs from its manifest"
        )
    return manifest


def validate_regular_cache_archive(path: Path) -> dict[str, Any]:
    """Validate a deterministic uv cache archive with internal symlinks."""
    entries: list[dict[str, Any]] = []
    names: list[str] = []
    try:
        with tarfile.open(path, mode="r:") as archive:
            if archive.pax_headers:
                raise OfficialForagaxOciError(
                    "uv cache archive contains global PAX metadata"
                )
            for member in archive:
                _safe_relative(member.name, label="uv cache member")
                if (
                    member.uid != 0
                    or member.gid != 0
                    or member.uname
                    or member.gname
                    or member.mtime != 0
                    or member.pax_headers
                    or member.sparse
                ):
                    raise OfficialForagaxOciError(
                        f"uv cache metadata is noncanonical: {member.name}"
                    )
                mode = stat.S_IMODE(member.mode)
                if member.isdir() and member.type == tarfile.DIRTYPE:
                    if mode != 0o700:
                        raise OfficialForagaxOciError(
                            f"uv cache directory mode differs: {member.name}"
                        )
                    entry: dict[str, Any] = {
                        "mode": mode,
                        "path": member.name,
                        "type": "directory",
                    }
                elif member.isfile() and member.type in {
                    tarfile.REGTYPE,
                    tarfile.AREGTYPE,
                }:
                    if mode not in {0o600, 0o700}:
                        raise OfficialForagaxOciError(
                            f"uv cache file mode differs: {member.name}"
                        )
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise OfficialForagaxOciError(
                            f"uv cache member cannot be read: {member.name}"
                        )
                    digest = hashlib.sha256()
                    remaining = member.size
                    while remaining:
                        block = handle.read(min(1024 * 1024, remaining))
                        if not block:
                            raise OfficialForagaxOciError(
                                f"uv cache member is truncated: {member.name}"
                            )
                        digest.update(block)
                        remaining -= len(block)
                    entry = {
                        "mode": mode,
                        "path": member.name,
                        "sha256": digest.hexdigest(),
                        "size": member.size,
                        "type": "file",
                    }
                elif member.issym() and member.type == tarfile.SYMTYPE:
                    if (
                        mode != 0o777
                        or not member.linkname
                        or PurePosixPath(member.linkname).is_absolute()
                    ):
                        raise OfficialForagaxOciError(
                            f"uv cache symlink is noncanonical: {member.name}"
                        )
                    normalized_target = posixpath.normpath(
                        posixpath.join(
                            posixpath.dirname(member.name),
                            member.linkname,
                        )
                    )
                    if normalized_target == ".." or normalized_target.startswith(
                        "../"
                    ):
                        raise OfficialForagaxOciError(
                            f"uv cache symlink escapes: {member.name}"
                        )
                    entry = {
                        "mode": mode,
                        "path": member.name,
                        "target": member.linkname,
                        "type": "symlink",
                    }
                else:
                    raise OfficialForagaxOciError(
                        f"uv cache archive has a special member: {member.name}"
                    )
                names.append(member.name)
                entries.append(entry)
    except (OSError, tarfile.TarError) as exc:
        raise OfficialForagaxOciError(
            "uv cache archive cannot be validated"
        ) from exc
    if names != sorted(names) or len(names) != len(set(names)):
        raise OfficialForagaxOciError(
            "uv cache archive order/set is not deterministic"
        )
    if not entries:
        raise OfficialForagaxOciError("uv cache archive is empty")
    identity = {"entries": entries, "hash_scheme": TREE_HASH_SCHEME}
    return {
        "entry_count": len(entries),
        "inventory_sha256": _json_sha256(identity),
    }


def create_regular_cache_archive(cache_root: Path, output: Path) -> str:
    """Rewrite cache symlinks internally and write one canonical USTAR."""
    if output.exists():
        raise OfficialForagaxOciError(f"cache archive already exists: {output}")
    paths = sorted(
        cache_root.rglob("*"),
        key=lambda path: path.relative_to(cache_root).as_posix(),
    )
    with output.open("xb") as output_handle, tarfile.open(
        fileobj=output_handle,
        mode="w:",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for path in paths:
            relative = path.relative_to(cache_root).as_posix()
            metadata = path.lstat()
            info = tarfile.TarInfo(relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                try:
                    resolved_relative = resolved.relative_to(
                        cache_root.resolve()
                    )
                except ValueError as exc:
                    raise OfficialForagaxOciError(
                        f"uv cache symlink escapes: {relative}"
                    ) from exc
                target = os.path.relpath(
                    resolved_relative.as_posix(),
                    PurePosixPath(relative).parent.as_posix(),
                )
                info.type = tarfile.SYMTYPE
                info.mode = 0o777
                info.size = 0
                info.linkname = target
                archive.addfile(info)
            elif path.is_dir():
                info.type = tarfile.DIRTYPE
                info.mode = 0o700
                info.size = 0
                archive.addfile(info)
            elif path.is_file() and stat.S_ISREG(metadata.st_mode):
                info.type = tarfile.REGTYPE
                info.mode = 0o700 if metadata.st_mode & 0o111 else 0o600
                info.size = metadata.st_size
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                raise OfficialForagaxOciError(
                    f"uv cache contains special path {relative}"
                )
    validate_regular_cache_archive(output)
    return _sha256(output)


def _resource_path(name: str) -> Path:
    return Path(__file__).resolve().with_name(name)


def _dockerfile(build: Mapping[str, Any]) -> str:
    base = cast(str, build["base_image"])
    commit = cast(str, build["source_commit"])
    source_sha = cast(str, build["source_archive_sha256"])
    lock_sha = cast(str, build["dependency_lock_sha256"])
    epoch = cast(int, build["source_commit_timestamp"])
    created = datetime.fromtimestamp(epoch, tz=UTC).isoformat().replace(
        "+00:00",
        "Z",
    )
    return f"""\
# syntax={DOCKERFILE_FRONTEND}
ARG SOURCE_DATE_EPOCH={epoch}
FROM {base} AS python-base
ARG SOURCE_DATE_EPOCH
RUN --network=none \\
    --mount=type=bind,source=debs,target=/input/debs,readonly \\
    --mount=type=bind,source=debian-manifest.json,target=/input/debian-manifest.json,readonly \\
    --mount=type=bind,source=image-helper.py,target=/input/image-helper.py,readonly \\
    set -eu; \\
    dpkg --unpack /input/debs/*.deb; \\
    DEBIAN_FRONTEND=noninteractive TZ=UTC dpkg --configure -a; \\
    /usr/bin/python3.12 -I -B /input/image-helper.py verify-debian \\
      --manifest /input/debian-manifest.json >/dev/null

FROM python-base AS dependency-builder
ARG SOURCE_DATE_EPOCH
RUN --network=none \\
    --mount=type=bind,source=source.tar,target=/input/source.tar,readonly \\
    --mount=type=bind,source=uv,target=/input/uv,readonly \\
    --mount=type=bind,source=uv-cache.tar,target=/input/uv-cache.tar,readonly \\
    --mount=type=bind,source=image-helper.py,target=/input/image-helper.py,readonly \\
    set -eu; \\
    /usr/bin/python3.12 -I -B /input/image-helper.py extract-source \\
      --archive /input/source.tar --destination /work/source \\
      --commit {commit} --lock-sha256 {lock_sha}; \\
    /usr/bin/python3.12 -I -B /input/image-helper.py extract-cache \\
      --archive /input/uv-cache.tar --destination /work/uv-cache; \\
    install -m 0555 /input/uv /work/uv; \\
    UV_CACHE_DIR=/work/uv-cache \\
    UV_LINK_MODE=copy \\
    UV_NO_PROGRESS=1 \\
    UV_OFFLINE=1 \\
    UV_PROJECT_ENVIRONMENT={RUNTIME_ROOT} \\
    UV_PYTHON_DOWNLOADS=never \\
      /work/uv sync --offline --frozen --no-dev --group cuda \\
      --no-install-project --python /usr/bin/python3.12 --project /work/source; \\
    rm {RUNTIME_ROOT}/bin/python; \\
    install -m 0555 /usr/bin/python3.12 {RUNTIME_ROOT}/bin/python; \\
    test ! -e {RUNTIME_ROOT}/lib/python3.12/site-packages/continual_foragax_agents-0.0.0.dist-info

FROM python-base AS source-builder
ARG SOURCE_DATE_EPOCH
RUN --network=none \\
    --mount=type=bind,source=source.tar,target=/input/source.tar,readonly \\
    --mount=type=bind,source=image-helper.py,target=/input/image-helper.py,readonly \\
    /usr/bin/python3.12 -I -B /input/image-helper.py extract-source \\
      --archive /input/source.tar --destination {SOURCE_ROOT} \\
      --commit {commit} --lock-sha256 {lock_sha}

FROM python-base AS finalizer
COPY --from=dependency-builder --chown=0:0 {RUNTIME_ROOT} {RUNTIME_ROOT}
COPY --from=source-builder --chown=0:0 {SOURCE_ROOT} {SOURCE_ROOT}
RUN --network=none \\
    --mount=type=bind,source=launcher.py,target=/input/launcher.py,readonly \\
    --mount=type=bind,source=build-attestation.json,target=/input/build-attestation.json,readonly \\
    --mount=type=bind,source=image-helper.py,target=/input/image-helper.py,readonly \\
    set -eu; \\
    install -d -m 0555 /opt/alberta; \\
    install -m 0555 /input/launcher.py {LAUNCHER_PATH}; \\
    /usr/bin/python3.12 -I -B /input/image-helper.py finalize \\
      --source-root {SOURCE_ROOT} --runtime-root {RUNTIME_ROOT} \\
      --launcher {LAUNCHER_PATH} \\
      --build-attestation /input/build-attestation.json \\
      --attestation-root {ATTESTATION_ROOT}

FROM python-base AS runtime
ARG SOURCE_DATE_EPOCH
COPY --from=finalizer --chown=0:0 {RUNTIME_ROOT} {RUNTIME_ROOT}
COPY --from=finalizer --chown=0:0 {SOURCE_ROOT} {SOURCE_ROOT}
COPY --from=finalizer --chown=0:0 /opt/alberta /opt/alberta
COPY --from=finalizer --chown=0:0 {ATTESTATION_ROOT} {ATTESTATION_ROOT}
ENV HOME=/run/alberta/home \\
    LANG=C.UTF-8 \\
    LC_ALL=C.UTF-8 \\
    NVIDIA_VISIBLE_DEVICES=void \\
    PATH={RUNTIME_ROOT}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONHASHSEED=0 \\
    PYTHONHOME= \\
    PYTHONNOUSERSITE=1 \\
    PYTHONPATH= \\
    PYTHONUTF8=1 \\
    TZ=UTC
LABEL org.opencontainers.image.created="{created}" \\
      org.opencontainers.image.revision="{commit}" \\
      org.opencontainers.image.source="https://github.com/steventango/continual-foragax-agents" \\
      io.elizaos.alberta.foragax.source-archive-sha256="{source_sha}" \\
      io.elizaos.alberta.foragax.dependency-lock-sha256="{lock_sha}" \\
      io.elizaos.alberta.foragax.launcher-contract="{OFFICIAL_FORAGAX_OCI_LAUNCHER_CONTRACT}"
USER 65532:65532
WORKDIR {SOURCE_ROOT}
CMD []
"""


def prepare_build_context(inputs: OciBuildInputs) -> PreparedOciBuild:
    """Validate all inputs and atomically create a path-neutral build context."""
    for name in (
        "source_archive",
        "dependency_lock",
        "uv_binary",
        "uv_cache_archive",
        "debian_manifest",
    ):
        path = cast(Path, getattr(inputs, name))
        if not path.is_file() or path.is_symlink():
            raise OfficialForagaxOciError(
                f"{name} must be a regular non-symlink file"
            )
    if not inputs.debian_bundle.is_dir() or inputs.debian_bundle.is_symlink():
        raise OfficialForagaxOciError(
            "Debian bundle must be a non-symlink directory"
        )
    if inputs.output_context.exists():
        raise OfficialForagaxOciError(
            f"output context already exists: {inputs.output_context}"
        )
    if inputs.source_commit != OFFICIAL_FORAGAX_AUDIT_COMMIT:
        raise OfficialForagaxOciError(
            "source commit is not the audited official Foragax commit"
        )
    if inputs.source_tree_git_sha1 != _AUDITED_SOURCE_TREE_GIT_SHA1:
        raise OfficialForagaxOciError(
            "source tree is not the audited official Foragax Git tree"
        )
    if (
        inputs.base_image != _AUDITED_BASE_IMAGE
        or _OCI_REFERENCE_PATTERN.fullmatch(inputs.base_image) is None
    ):
        raise OfficialForagaxOciError(
            "base image is not the audited digest-pinned CUDA base"
        )
    for path, expected, label in (
        (
            inputs.source_archive,
            inputs.source_archive_sha256,
            "source archive",
        ),
        (
            inputs.dependency_lock,
            inputs.dependency_lock_sha256,
            "dependency lock",
        ),
        (inputs.uv_binary, inputs.uv_binary_sha256, "uv binary"),
        (
            inputs.uv_cache_archive,
            inputs.uv_cache_archive_sha256,
            "uv cache archive",
        ),
    ):
        _verify_hash(path, expected, label=label)
    if inputs.uv_binary_sha256 != UV_BINARY_SHA256:
        raise OfficialForagaxOciError("uv binary is not the audited uv 0.9.24")
    lock_bytes = inputs.dependency_lock.read_bytes()
    locked_versions = _validate_lock(lock_bytes)
    source_identity = _source_archive_identity(
        inputs.source_archive,
        source_commit=inputs.source_commit,
        dependency_lock=lock_bytes,
    )
    if (
        source_identity["source_tree_git_sha1"]
        != inputs.source_tree_git_sha1
    ):
        raise OfficialForagaxOciError(
            "source archive bytes do not reconstruct the audited Git tree"
        )
    cache_identity = validate_regular_cache_archive(inputs.uv_cache_archive)
    debian_manifest = validate_debian_manifest(
        inputs.debian_manifest,
        inputs.debian_bundle,
    )
    launcher = _resource_path("_official_foragax_v4_launcher.py")
    helper = _resource_path("_official_foragax_image_helper.py")
    launcher_sha = _sha256(launcher)
    helper_sha = _sha256(helper)
    build_spec: dict[str, Any] = {
        "base_image": inputs.base_image,
        "debian_bundle_manifest_sha256": _sha256(inputs.debian_manifest),
        "debian_installed_file_inventory_sha256": debian_manifest[
            "installed_file_inventory_sha256"
        ],
        "dependency_lock_sha256": inputs.dependency_lock_sha256,
        "dockerfile_frontend": DOCKERFILE_FRONTEND,
        "image_helper_sha256": helper_sha,
        "launcher_contract": OFFICIAL_FORAGAX_OCI_LAUNCHER_CONTRACT,
        "launcher_sha256": launcher_sha,
        "locked_scientific_versions": locked_versions,
        "native_runtime_inventory_hash_scheme": (
            OFFICIAL_FORAGAX_NATIVE_RUNTIME_INVENTORY_HASH_SCHEME
        ),
        "native_runtime_inventory_root": RUNTIME_ROOT.as_posix(),
        "python_executable": PYTHON_EXECUTABLE.as_posix(),
        "python_executable_sha256": debian_manifest[
            "python_executable_sha256"
        ],
        "schema_version": BUILD_SPEC_SCHEMA,
        "source_archive_inventory_sha256": source_identity[
            "inventory_sha256"
        ],
        "source_archive_sha256": inputs.source_archive_sha256,
        "source_commit": inputs.source_commit,
        "source_commit_timestamp": source_identity["commit_timestamp"],
        "source_root": SOURCE_ROOT.as_posix(),
        "source_tree_git_sha1": inputs.source_tree_git_sha1,
        "uv_binary_sha256": inputs.uv_binary_sha256,
        "uv_cache_archive_inventory_sha256": cache_identity[
            "inventory_sha256"
        ],
        "uv_cache_archive_sha256": inputs.uv_cache_archive_sha256,
    }
    for forbidden in _FORBIDDEN_ATTESTATION_TEXT:
        if forbidden in _canonical_json_bytes(build_spec).decode("ascii"):
            raise OfficialForagaxOciError(
                "build attestation leaks a build/cache path"
            )
    context = inputs.output_context
    temporary = context.with_name(f".{context.name}.{uuid.uuid4().hex}.partial")
    temporary.mkdir(mode=0o700, parents=True)
    try:
        shutil.copyfile(inputs.source_archive, temporary / "source.tar")
        shutil.copyfile(inputs.uv_binary, temporary / "uv")
        (temporary / "uv").chmod(0o555)
        shutil.copyfile(inputs.uv_cache_archive, temporary / "uv-cache.tar")
        shutil.copyfile(inputs.debian_manifest, temporary / "debian-manifest.json")
        debs = temporary / "debs"
        debs.mkdir(mode=0o700)
        for package in cast(list[dict[str, Any]], debian_manifest["packages"]):
            filename = cast(str, package["filename"])
            shutil.copyfile(inputs.debian_bundle / filename, debs / filename)
        shutil.copyfile(launcher, temporary / "launcher.py")
        shutil.copyfile(helper, temporary / "image-helper.py")
        (temporary / "launcher.py").chmod(0o555)
        (temporary / "image-helper.py").chmod(0o444)
        (temporary / "build-attestation.json").write_bytes(
            _canonical_json_bytes(build_spec, newline=True)
        )
        dockerfile = _dockerfile(build_spec)
        (temporary / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        (temporary / ".dockerignore").write_text(
            "\n".join(
                (
                    "**",
                    "!.dockerignore",
                    "!Dockerfile",
                    "!build-attestation.json",
                    "!debian-manifest.json",
                    "!debs",
                    "!debs/**",
                    "!image-helper.py",
                    "!launcher.py",
                    "!source.tar",
                    "!uv",
                    "!uv-cache.tar",
                    "",
                )
            ),
            encoding="utf-8",
        )
        os.replace(temporary, context)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PreparedOciBuild(
        context=context,
        build_spec=build_spec,
        build_spec_sha256=_sha256(context / "build-attestation.json"),
        dockerfile_sha256=_sha256(context / "Dockerfile"),
        launcher_sha256=launcher_sha,
    )


def docker_build_command(
    prepared: PreparedOciBuild,
    *,
    image_tag: str,
    docker: Path = Path("/usr/bin/docker"),
) -> tuple[str, ...]:
    if (
        not image_tag
        or "@" in image_tag
        or any(character.isspace() for character in image_tag)
    ):
        raise OfficialForagaxOciError("build image tag is invalid")
    epoch = prepared.build_spec.get("source_commit_timestamp")
    if type(epoch) is not int or epoch < 0:
        raise OfficialForagaxOciError(
            "prepared source commit timestamp is invalid"
        )
    return (
        str(docker),
        "buildx",
        "build",
        "--load",
        "--build-arg",
        f"SOURCE_DATE_EPOCH={epoch}",
        "--network=none",
        "--provenance=false",
        "--sbom=false",
        "--tag",
        image_tag,
        str(prepared.context),
    )


def build_image(
    prepared: PreparedOciBuild,
    *,
    image_tag: str,
    docker: Path = Path("/usr/bin/docker"),
) -> None:
    """Build through BuildKit with all build-step networking disabled."""
    completed = subprocess.run(
        docker_build_command(prepared, image_tag=image_tag, docker=docker),
        check=False,
    )
    if completed.returncode != 0:
        raise OfficialForagaxOciError(
            f"offline OCI build failed with status {completed.returncode}"
        )


def _scratch_arguments() -> tuple[str, ...]:
    return (
        (
            "--mount=type=tmpfs,destination=/tmp/src,"
            "tmpfs-mode=0555,tmpfs-size=1048576"
        ),
        (
            "--tmpfs=/run/alberta:"
            "rw,noexec,nosuid,nodev,size=8g,uid=65532,gid=65532,mode=0700"
        ),
        (
            "--tmpfs=/run/alberta/home:"
            "rw,noexec,nosuid,nodev,size=64m,uid=65532,gid=65532,mode=0700"
        ),
        (
            "--tmpfs=/run/alberta/tmp:"
            "rw,noexec,nosuid,nodev,size=1g,uid=65532,gid=65532,mode=0700"
        ),
        (
            "--tmpfs=/run/alberta/matplotlib:"
            "rw,noexec,nosuid,nodev,size=64m,uid=65532,gid=65532,mode=0700"
        ),
        (
            "--tmpfs=/run/alberta/cache:"
            "rw,noexec,nosuid,nodev,size=1g,uid=65532,gid=65532,mode=0700"
        ),
        (
            "--tmpfs=/run/alberta/cuda-cache:"
            "rw,noexec,nosuid,nodev,size=256m,uid=65532,gid=65532,mode=0700"
        ),
        (
            "--tmpfs=/run/alberta/jax-cache:"
            "rw,noexec,nosuid,nodev,size=256m,uid=65532,gid=65532,mode=0700"
        ),
    )


def _environment_arguments() -> tuple[str, ...]:
    return (
        "--env=CUDA_CACHE_DISABLE=1",
        "--env=CUDA_CACHE_MAXSIZE=268435456",
        "--env=CUDA_CACHE_PATH=/run/alberta/cuda-cache",
        "--env=HOME=/run/alberta/home",
        "--env=JAX_COMPILATION_CACHE_DIR=/run/alberta/jax-cache",
        "--env=JAX_ENABLE_COMPILATION_CACHE=false",
        "--env=LANG=C.UTF-8",
        "--env=LC_ALL=C.UTF-8",
        "--env=MPLCONFIGDIR=/run/alberta/matplotlib",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--env=PYTHONHASHSEED=0",
        "--env=PYTHONHOME=",
        "--env=PYTHONNOUSERSITE=1",
        "--env=PYTHONPATH=",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=PYTHONUTF8=1",
        "--env=TMPDIR=/run/alberta/tmp",
        "--env=TZ=UTC",
        "--env=XDG_CACHE_HOME=/run/alberta/cache",
    )


def emit_launch_command(
    *,
    image_id: str,
    entrypoint: str,
    config_path: str,
    index_expression: str,
    gpu: bool,
    max_steps: int | None = None,
    driver: DriverLaunchContract | None = None,
    docker: Path = Path("/usr/bin/docker"),
) -> tuple[str, ...]:
    """Emit the exact digest-only production sandbox and v4 launcher command."""
    if _OCI_DIGEST_PATTERN.fullmatch(image_id) is None:
        raise OfficialForagaxOciError(
            "launch image must be a sha256 image-config digest"
        )
    if entrypoint not in {"src/continuing_main.py", "src/rtu_ppo.py"}:
        raise OfficialForagaxOciError("launch entrypoint is not allowlisted")
    config = PurePosixPath(config_path)
    if (
        not config.is_absolute()
        or config.as_posix() != config_path
        or ".." in config.parts
        or not config.is_relative_to(SOURCE_ROOT)
    ):
        raise OfficialForagaxOciError(
            "configuration must be an immutable source-root path"
        )
    indices = _parse_index_expression(index_expression)
    del indices
    if max_steps is not None and (
        type(max_steps) is not int or max_steps < 1
    ):
        raise OfficialForagaxOciError("launch max steps must be positive")
    command = [
        str(docker),
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65532:65532",
        "--pids-limit=512",
        *_scratch_arguments(),
        f"--workdir={SOURCE_ROOT}",
        *_environment_arguments(),
    ]
    if gpu:
        if driver is None:
            raise OfficialForagaxOciError(
                "GPU launch requires an exact driver/device contract"
            )
        command.extend(_driver_arguments(driver))
    else:
        if driver is not None:
            raise OfficialForagaxOciError(
                "CPU launch must not receive a driver contract"
            )
        command.extend(
            (
                "--env=JAX_PLATFORM_NAME=cpu",
                "--env=JAX_PLATFORMS=cpu",
                "--env=JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1",
            )
        )
    command.extend(
        (
            image_id,
            LAUNCHER_PATH.as_posix(),
            "--python",
            PYTHON_EXECUTABLE.as_posix(),
            "--python-flag=-I",
            "--python-flag=-B",
            "--trusted-python-path",
            (SOURCE_ROOT / "src").as_posix(),
            "--trusted-python-path-mode=isolated-runpy-prepend-v1",
            "--entrypoint",
            (SOURCE_ROOT / entrypoint).as_posix(),
            "--config",
            config_path,
            "--index",
            index_expression,
            "--save-path",
            "/run/alberta/output/official-results",
            "--checkpoint-path",
            "/run/alberta/output/official-checkpoints",
            "--stdout-log",
            "/run/alberta/output/stdout.log",
            "--stderr-log",
            "/run/alberta/output/stderr.log",
            "--export-format",
            OFFICIAL_FORAGAX_OCI_LAUNCHER_CONTRACT,
        )
    )
    if gpu:
        command.append("--gpu")
    if max_steps is not None:
        command.extend(("--max-steps", str(max_steps)))
    return tuple(command)


def _parse_index_expression(value: str) -> tuple[int, ...]:
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
        index = int(value)
        if index <= (1 << 32) - 1:
            return (index,)
    match = re.fullmatch(
        r"(0|[1-9][0-9]*):(0|[1-9][0-9]*)",
        value,
    )
    if match is not None:
        start, stop = (int(part) for part in match.groups())
        if 0 <= start < stop <= (1 << 32):
            return tuple(range(start, stop))
    raise OfficialForagaxOciError("index expression is not canonical")


def _canonical_host_path(value: str, *, label: str) -> str:
    path = Path(value)
    if (
        not path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or any(character in value for character in ",:=\x00")
    ):
        raise OfficialForagaxOciError(
            f"{label} is not a safe canonical host path"
        )
    return value


def _canonical_container_path(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
        or any(character in value for character in ",:=\x00")
    ):
        raise OfficialForagaxOciError(
            f"{label} is not a safe canonical container path"
        )
    return value


def _driver_arguments(contract: DriverLaunchContract) -> tuple[str, ...]:
    host = _canonical_host_path(
        contract.driver_host_path,
        label="driver host path",
    )
    container = _canonical_container_path(
        contract.driver_container_path,
        label="driver container path",
    )
    expected_devices = tuple(
        f"/dev/nvidia{index}" for index in contract.device_indices
    )
    if (
        not contract.device_indices
        or tuple(sorted(contract.device_indices)) != contract.device_indices
        or len(set(contract.device_indices)) != len(contract.device_indices)
        or not set(expected_devices).issubset(contract.device_paths)
        or "/dev/nvidiactl" not in contract.device_paths
        or "/dev/nvidia-uvm" not in contract.device_paths
        or any(
            re.fullmatch(
                r"/dev/nvidia(?:[0-9]+|ctl|-uvm|-uvm-tools|-modeset)",
                path,
            )
            is None
            for path in contract.device_paths
        )
    ):
        raise OfficialForagaxOciError("NVIDIA device contract is invalid")
    cuda_paths = tuple(
        _canonical_container_path(path, label="CUDA wheel library path")
        for path in contract.cuda_wheel_library_paths
    )
    driver_paths = tuple(
        _canonical_container_path(path, label="driver library path")
        for path in contract.driver_user_library_paths
    )
    if not cuda_paths or driver_paths != (container,):
        raise OfficialForagaxOciError(
            "GPU library path roles/order are invalid"
        )
    library_path = ":".join((*cuda_paths, *driver_paths))
    return (
        (
            "--mount=type=bind,"
            f"source={host},destination={container},readonly"
        ),
        *(f"--device={path}" for path in contract.device_paths),
        (
            "--env=CUDA_VISIBLE_DEVICES="
            + ",".join(str(index) for index in contract.device_indices)
        ),
        f"--env=CUBLAS_WORKSPACE_CONFIG={OFFICIAL_FORAGAX_CUBLAS_WORKSPACE_CONFIG}",
        f"--env=LD_LIBRARY_PATH={library_path}",
        f"--env=XLA_FLAGS={OFFICIAL_FORAGAX_GPU_XLA_FLAGS}",
        (
            "--env=XLA_PYTHON_CLIENT_PREALLOCATE="
            f"{OFFICIAL_FORAGAX_XLA_PYTHON_CLIENT_PREALLOCATE}"
        ),
    )


def _run_bytes(
    command: Sequence[str],
    *,
    stdin: bytes | None = None,
) -> bytes:
    completed = subprocess.run(
        tuple(command),
        input=stdin,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace").strip()
        raise OfficialForagaxOciError(
            f"command failed ({completed.returncode}): {' '.join(command)}"
            + (f": {diagnostic}" if diagnostic else "")
        )
    if completed.stderr:
        raise OfficialForagaxOciError(
            f"successful command emitted stderr: {' '.join(command)}"
        )
    return completed.stdout


def _inspect_image_json(
    image: str,
    *,
    docker: Path,
) -> dict[str, Any]:
    raw = _run_bytes(
        (
            str(docker),
            "image",
            "inspect",
            "--format={{json .}}",
            image,
        )
    )
    value = _strict_json_bytes(raw, label="Docker image inspection")
    if type(value) is not dict:
        raise OfficialForagaxOciError(
            "Docker image inspection did not return an object"
        )
    return cast(dict[str, Any], value)


def _inspected_config_digest(inspected: Mapping[str, Any]) -> str:
    """Resolve the config digest across legacy and containerd image stores."""
    image_id = inspected.get("Id")
    if (
        type(image_id) is not str
        or _OCI_DIGEST_PATTERN.fullmatch(image_id) is None
    ):
        raise OfficialForagaxOciError("inspected image ID is invalid")
    descriptor = inspected.get("Descriptor")
    if descriptor is None:
        # The legacy Docker graphdriver uses the config digest as `.Id`.
        return image_id
    if type(descriptor) is not dict:
        raise OfficialForagaxOciError("image descriptor is invalid")
    descriptor_mapping = cast(dict[str, Any], descriptor)
    annotations = descriptor_mapping.get("annotations")
    if (
        descriptor_mapping.get("digest") != image_id
        or descriptor_mapping.get("mediaType")
        not in {
            "application/vnd.docker.distribution.manifest.v2+json",
            "application/vnd.oci.image.manifest.v1+json",
        }
        or type(descriptor_mapping.get("size")) is not int
        or cast(int, descriptor_mapping["size"]) < 1
        or type(annotations) is not dict
    ):
        raise OfficialForagaxOciError(
            "containerd image descriptor identity is invalid"
        )
    config_digest = cast(dict[str, Any], annotations).get("config.digest")
    if (
        type(config_digest) is not str
        or _OCI_DIGEST_PATTERN.fullmatch(config_digest) is None
    ):
        raise OfficialForagaxOciError(
            "containerd image descriptor lacks its config digest"
        )
    return config_digest


def _saved_config_identity(
    image: str,
    *,
    config_digest: str,
    docker: Path,
) -> tuple[str, dict[str, Any]]:
    """Verify the config blob from a Docker/OCI image-save stream."""
    if _OCI_DIGEST_PATTERN.fullmatch(config_digest) is None:
        raise OfficialForagaxOciError("saved config digest is invalid")
    process = subprocess.Popen(
        (str(docker), "image", "save", image),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:  # pragma: no cover
        raise OfficialForagaxOciError("Docker image save pipes are unavailable")
    config_bytes: bytes | None = None
    digest_hex = config_digest.removeprefix("sha256:")
    expected_names = {
        digest_hex + ".json",
        "blobs/sha256/" + digest_hex,
    }
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                name = member.name.removeprefix("./")
                if name not in expected_names:
                    continue
                if config_bytes is not None:
                    raise OfficialForagaxOciError(
                        "Docker image save stream repeats its config blob"
                    )
                if not member.isfile() or member.size > 16 * 1024 * 1024:
                    raise OfficialForagaxOciError(
                        "Docker image config member is not a bounded file"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise OfficialForagaxOciError(
                        "Docker image config member cannot be read"
                    )
                config_bytes = handle.read(member.size + 1)
                if len(config_bytes) != member.size:
                    raise OfficialForagaxOciError(
                        "Docker image config member is truncated"
                    )
    except tarfile.TarError as exc:
        process.kill()
        raise OfficialForagaxOciError(
            "Docker image save stream is not a valid tar"
        ) from exc
    stderr = process.stderr.read()
    returncode = process.wait()
    if returncode != 0 or stderr:
        raise OfficialForagaxOciError(
            "Docker image save failed or emitted stderr"
        )
    if config_bytes is None:
        raise OfficialForagaxOciError(
            "Docker image save stream lacks its config blob"
        )
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    if f"sha256:{config_sha}" != config_digest:
        raise OfficialForagaxOciError(
            "Docker image config digest does not verify against config bytes"
        )
    value = _strict_json_bytes(config_bytes, label="Docker image config")
    if type(value) is not dict:
        raise OfficialForagaxOciError("Docker image config is not an object")
    return config_sha, cast(dict[str, Any], value)


def _rootfs_stream_inventory(
    stream: Any,
    *,
    capture_paths: set[str],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Hash a flattened Docker-export tar without extracting it."""
    entries: list[dict[str, Any]] = []
    captures: dict[str, bytes] = {}
    names: set[str] = set()
    try:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                raw_name = member.name.removeprefix("./").rstrip("/")
                if not raw_name:
                    continue
                relative = _safe_relative(raw_name, label="rootfs member")
                name = relative.as_posix()
                if name in names:
                    raise OfficialForagaxOciError(
                        f"rootfs export repeats path {name}"
                    )
                names.add(name)
                mode = stat.S_IMODE(member.mode)
                if member.isdir():
                    entry: dict[str, Any] = {
                        "mode": mode,
                        "path": name,
                        "type": "directory",
                    }
                elif member.isfile() and member.type in {
                    tarfile.REGTYPE,
                    tarfile.AREGTYPE,
                }:
                    if member.size < 0 or member.size > 16 * 1024**3:
                        raise OfficialForagaxOciError(
                            f"rootfs member size is unbounded: {name}"
                        )
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise OfficialForagaxOciError(
                            f"rootfs member cannot be read: {name}"
                        )
                    digest = hashlib.sha256()
                    captured = bytearray() if name in capture_paths else None
                    remaining = member.size
                    while remaining:
                        block = handle.read(min(1024 * 1024, remaining))
                        if not block:
                            raise OfficialForagaxOciError(
                                f"rootfs member is truncated: {name}"
                            )
                        digest.update(block)
                        if captured is not None:
                            captured.extend(block)
                        remaining -= len(block)
                    if captured is not None:
                        captures[name] = bytes(captured)
                    entry = {
                        "mode": mode,
                        "path": name,
                        "sha256": digest.hexdigest(),
                        "size": member.size,
                        "type": "file",
                    }
                elif member.issym():
                    entry = {
                        "mode": mode,
                        "path": name,
                        "target": member.linkname,
                        "type": "symlink",
                    }
                elif member.islnk():
                    target = PurePosixPath(member.linkname.removeprefix("./"))
                    if (
                        target.is_absolute()
                        or ".." in target.parts
                        or target.as_posix() not in names
                    ):
                        raise OfficialForagaxOciError(
                            f"rootfs hardlink target is unsafe: {name}"
                        )
                    entry = {
                        "mode": mode,
                        "path": name,
                        "target": target.as_posix(),
                        "type": "hardlink",
                    }
                else:
                    raise OfficialForagaxOciError(
                        f"rootfs contains a special entry: {name}"
                    )
                entries.append(entry)
    except tarfile.TarError as exc:
        raise OfficialForagaxOciError(
            "Docker rootfs export is not a valid tar"
        ) from exc
    entries.sort(key=lambda entry: cast(str, entry["path"]))
    identity = {"entries": entries, "hash_scheme": TREE_HASH_SCHEME}
    return (
        {
            **identity,
            "root": "/",
            "schema_version": ROOTFS_INVENTORY_SCHEMA,
            "tree_sha256": _json_sha256(identity),
        },
        captures,
    )


def _export_rootfs_inventory(
    image: str,
    *,
    docker: Path,
    capture_paths: set[str],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    name = f"alberta-foragax-inspect-{uuid.uuid4().hex}"
    create = subprocess.run(
        (
            str(docker),
            "container",
            "create",
            "--name",
            name,
            "--network=none",
            image,
            "/bin/true",
        ),
        check=False,
        capture_output=True,
    )
    if create.returncode != 0 or create.stderr:
        raise OfficialForagaxOciError(
            "Docker could not create a stopped inspection container"
        )
    try:
        process = subprocess.Popen(
            (str(docker), "container", "export", name),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None or process.stderr is None:  # pragma: no cover
            raise OfficialForagaxOciError(
                "Docker rootfs export pipes are unavailable"
            )
        inventory, captures = _rootfs_stream_inventory(
            process.stdout,
            capture_paths=capture_paths,
        )
        stderr = process.stderr.read()
        returncode = process.wait()
        if returncode != 0 or stderr:
            raise OfficialForagaxOciError(
                "Docker rootfs export failed or emitted stderr"
            )
        return inventory, captures
    finally:
        remove = subprocess.run(
            (str(docker), "container", "rm", name),
            check=False,
            capture_output=True,
        )
        if remove.returncode != 0:
            raise OfficialForagaxOciError(
                "Docker inspection container could not be removed"
            )


def _entry_map(inventory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries = inventory.get("entries")
    if type(entries) is not list:
        raise OfficialForagaxOciError("tree inventory entries are invalid")
    result: dict[str, dict[str, Any]] = {}
    for raw_entry in entries:
        if type(raw_entry) is not dict:
            raise OfficialForagaxOciError(
                "tree inventory contains a non-object entry"
            )
        entry = cast(dict[str, Any], raw_entry)
        path = entry.get("path")
        if type(path) is not str or path in result:
            raise OfficialForagaxOciError(
                "tree inventory contains an invalid/repeated path"
            )
        result[path] = entry
    return result


def _verify_native_inventory(
    rootfs: Mapping[str, Any],
    native: Mapping[str, Any],
) -> str:
    _exact_keys(
        native,
        {
            "entries",
            "hash_scheme",
            "root",
            "schema_version",
            "tree_sha256",
        },
        label="native runtime inventory",
    )
    if (
        native["schema_version"] != NATIVE_INVENTORY_SCHEMA
        or native["hash_scheme"] != TREE_HASH_SCHEME
        or native["root"] != RUNTIME_ROOT.as_posix()
    ):
        raise OfficialForagaxOciError(
            "native runtime inventory identity is unsupported"
        )
    native_identity = {
        "entries": native["entries"],
        "hash_scheme": native["hash_scheme"],
    }
    tree_sha = _json_sha256(native_identity)
    if native["tree_sha256"] != tree_sha:
        raise OfficialForagaxOciError(
            "native runtime inventory digest does not verify"
        )
    root_entries = _entry_map(rootfs)
    native_entries = _entry_map(native)
    prefix = RUNTIME_ROOT.as_posix().removeprefix("/") + "/"
    observed = {
        path.removeprefix(prefix): entry
        for path, entry in root_entries.items()
        if path.startswith(prefix)
    }
    if set(observed) != set(native_entries):
        raise OfficialForagaxOciError(
            "native inventory path set differs from exported rootfs"
        )
    for path, expected in native_entries.items():
        actual = dict(observed[path])
        actual["path"] = path
        if _canonical_json_bytes(actual) != _canonical_json_bytes(expected):
            raise OfficialForagaxOciError(
                f"native inventory entry differs from rootfs: {path}"
            )
    return tree_sha


def _validate_image_config(
    inspected: Mapping[str, Any],
    raw_config: Mapping[str, Any],
    build: Mapping[str, Any],
) -> tuple[str, list[str], str]:
    image_id = inspected.get("Id")
    if type(image_id) is not str or _OCI_DIGEST_PATTERN.fullmatch(image_id) is None:
        raise OfficialForagaxOciError("inspected image ID is invalid")
    rootfs = inspected.get("RootFS")
    if type(rootfs) is not dict:
        raise OfficialForagaxOciError("inspected image RootFS is invalid")
    layers = cast(dict[str, Any], rootfs).get("Layers")
    if (
        cast(dict[str, Any], rootfs).get("Type") != "layers"
        or type(layers) is not list
        or not layers
        or not all(
            type(layer) is str and _OCI_DIGEST_PATTERN.fullmatch(layer)
            for layer in layers
        )
    ):
        raise OfficialForagaxOciError("image rootfs diff IDs are invalid")
    config = inspected.get("Config")
    if type(config) is not dict:
        raise OfficialForagaxOciError("inspected image Config is invalid")
    config_mapping = cast(dict[str, Any], config)
    entrypoint = config_mapping.get("Entrypoint")
    command = config_mapping.get("Cmd")
    if (
        config_mapping.get("User") != "65532:65532"
        or config_mapping.get("WorkingDir") != SOURCE_ROOT.as_posix()
        or (entrypoint is not None and entrypoint != [])
        or (command is not None and command != [])
    ):
        raise OfficialForagaxOciError(
            "image default user/workdir/command contract is invalid"
        )
    environment = config_mapping.get("Env")
    if type(environment) is not list or not all(
        type(item) is str for item in environment
    ):
        raise OfficialForagaxOciError("image environment is invalid")
    environment_mapping: dict[str, str] = {}
    for item in cast(list[str], environment):
        name, separator, value = item.partition("=")
        if not separator or not name or name in environment_mapping:
            raise OfficialForagaxOciError(
                "image environment contains an invalid/repeated name"
            )
        environment_mapping[name] = value
    if environment_mapping.get("NVIDIA_VISIBLE_DEVICES") != "void":
        raise OfficialForagaxOciError(
            "image does not disable implicit NVIDIA runtime injection"
        )
    environment_text = "\n".join(cast(list[str], environment))
    if any(value in environment_text for value in _FORBIDDEN_ATTESTATION_TEXT):
        raise OfficialForagaxOciError(
            "image environment exposes a build/cache path"
        )
    labels = config_mapping.get("Labels")
    if type(labels) is not dict:
        raise OfficialForagaxOciError("image labels are missing")
    label_mapping = cast(dict[str, Any], labels)
    expected_epoch = build.get("source_commit_timestamp")
    if type(expected_epoch) is not int or expected_epoch < 0:
        raise OfficialForagaxOciError(
            "expected source commit timestamp is invalid"
        )
    expected_created = datetime.fromtimestamp(
        expected_epoch,
        tz=UTC,
    ).isoformat().replace("+00:00", "Z")
    expected_labels = {
        "io.elizaos.alberta.foragax.dependency-lock-sha256": build[
            "dependency_lock_sha256"
        ],
        "io.elizaos.alberta.foragax.launcher-contract": (
            OFFICIAL_FORAGAX_OCI_LAUNCHER_CONTRACT
        ),
        "io.elizaos.alberta.foragax.source-archive-sha256": build[
            "source_archive_sha256"
        ],
        "org.opencontainers.image.created": expected_created,
        "org.opencontainers.image.revision": build["source_commit"],
    }
    if any(label_mapping.get(key) != value for key, value in expected_labels.items()):
        raise OfficialForagaxOciError("image provenance labels differ")
    raw_rootfs = raw_config.get("rootfs")
    if (
        type(raw_rootfs) is not dict
        or cast(dict[str, Any], raw_rootfs).get("diff_ids") != layers
    ):
        raise OfficialForagaxOciError(
            "raw config diff IDs differ from image inspection"
        )
    raw_runtime_config = raw_config.get("config")
    if type(raw_runtime_config) is not dict or any(
        cast(dict[str, Any], raw_runtime_config).get(key)
        != config_mapping.get(key)
        for key in ("Cmd", "Entrypoint", "Env", "Labels", "User", "WorkingDir")
    ):
        raise OfficialForagaxOciError(
            "raw runtime config differs from image inspection"
        )
    if (
        raw_config.get("created") != expected_created
        or inspected.get("Created") != expected_created
    ):
        raise OfficialForagaxOciError(
            "image creation timestamp is not SOURCE_DATE_EPOCH"
        )
    repo_digests = inspected.get("RepoDigests")
    if (
        type(repo_digests) is not list
        or not repo_digests
        or not all(
            type(reference) is str
            and re.search(r"@sha256:[0-9a-f]{64}$", reference) is not None
            for reference in repo_digests
        )
    ):
        raise OfficialForagaxOciError(
            "image lacks a digest-only repository identity"
        )
    reference = sorted(cast(list[str], repo_digests))[0]
    reference_digest = reference.rsplit("@", 1)[1]
    if (
        inspected.get("Descriptor") is not None
        and reference_digest != image_id
    ):
        raise OfficialForagaxOciError(
            "containerd repository digest differs from image descriptor"
        )
    return image_id, cast(list[str], layers), reference_digest


def inspect_image(
    *,
    image: str,
    expected_build_spec: Mapping[str, Any],
    output_directory: Path,
    docker: Path = Path("/usr/bin/docker"),
) -> dict[str, Any]:
    """Generate and verify config, image, rootfs, native inventory, and SBOM."""
    if output_directory.exists():
        raise OfficialForagaxOciError(
            f"inspection output already exists: {output_directory}"
        )
    inspected = _inspect_image_json(image, docker=docker)
    inspected_id = inspected.get("Id")
    if type(inspected_id) is not str:
        raise OfficialForagaxOciError("image inspection lacks an ID")
    config_digest = _inspected_config_digest(inspected)
    config_sha, raw_config = _saved_config_identity(
        image,
        config_digest=config_digest,
        docker=docker,
    )
    image_id, diff_ids, reference_digest = _validate_image_config(
        inspected,
        raw_config,
        expected_build_spec,
    )
    capture_paths = {
        ATTESTATION_ROOT.as_posix().removeprefix("/")
        + "/build-attestation.json",
        ATTESTATION_ROOT.as_posix().removeprefix("/")
        + "/native-inventory.json",
        ATTESTATION_ROOT.as_posix().removeprefix("/")
        + "/runtime-metadata.json",
        ATTESTATION_ROOT.as_posix().removeprefix("/") + "/sbom.spdx.json",
        LAUNCHER_PATH.as_posix().removeprefix("/"),
    }
    rootfs, captures = _export_rootfs_inventory(
        image,
        docker=docker,
        capture_paths=capture_paths,
    )
    missing = capture_paths - set(captures)
    if missing:
        raise OfficialForagaxOciError(
            f"image rootfs lacks attested files: {sorted(missing)}"
        )
    build_bytes = captures[
        ATTESTATION_ROOT.as_posix().removeprefix("/")
        + "/build-attestation.json"
    ]
    build_value = _strict_json_bytes(
        build_bytes,
        label="embedded build attestation",
    )
    if build_value != dict(expected_build_spec):
        raise OfficialForagaxOciError(
            "embedded build attestation differs from prepared inputs"
        )
    native_value = _strict_json_bytes(
        captures[
            ATTESTATION_ROOT.as_posix().removeprefix("/")
            + "/native-inventory.json"
        ],
        label="embedded native inventory",
    )
    if type(native_value) is not dict:
        raise OfficialForagaxOciError(
            "embedded native inventory is not an object"
        )
    native_sha = _verify_native_inventory(
        rootfs,
        cast(dict[str, Any], native_value),
    )
    sbom_bytes = captures[
        ATTESTATION_ROOT.as_posix().removeprefix("/") + "/sbom.spdx.json"
    ]
    sbom_value = _strict_json_bytes(sbom_bytes, label="embedded SPDX SBOM")
    if (
        type(sbom_value) is not dict
        or cast(dict[str, Any], sbom_value).get("spdxVersion") != "SPDX-2.3"
        or cast(dict[str, Any], sbom_value).get("dataLicense") != "CC0-1.0"
    ):
        raise OfficialForagaxOciError("embedded SPDX SBOM is invalid")
    packages = cast(dict[str, Any], sbom_value).get("packages")
    if type(packages) is not list:
        raise OfficialForagaxOciError("embedded SPDX package list is invalid")
    versions = {
        re.sub(r"[-_.]+", "-", cast(str, package.get("name"))).casefold(): (
            package.get("versionInfo")
        )
        for package in packages
        if type(package) is dict and type(package.get("name")) is str
    }
    for name, version in cast(
        dict[str, str],
        expected_build_spec["locked_scientific_versions"],
    ).items():
        if versions.get(name) != version:
            raise OfficialForagaxOciError(
                f"SBOM version differs for locked package {name}"
            )
    if "continual-foragax-agents" in versions:
        raise OfficialForagaxOciError(
            "SBOM shows the upstream source was installed"
        )
    runtime_bytes = captures[
        ATTESTATION_ROOT.as_posix().removeprefix("/")
        + "/runtime-metadata.json"
    ]
    runtime_value = _strict_json_bytes(
        runtime_bytes,
        label="embedded runtime metadata",
    )
    if type(runtime_value) is not dict:
        raise OfficialForagaxOciError(
            "embedded runtime metadata is not an object"
        )
    runtime = cast(dict[str, Any], runtime_value)
    if (
        runtime.get("python_executable") != PYTHON_EXECUTABLE.as_posix()
        or runtime.get("python_executable_sha256")
        != expected_build_spec["python_executable_sha256"]
    ):
        raise OfficialForagaxOciError(
            "embedded runtime Python identity differs"
        )
    bundled = runtime.get("bundled_executables")
    if type(bundled) is not dict or type(
        cast(dict[str, Any], bundled).get("imageio-ffmpeg")
    ) is not dict:
        raise OfficialForagaxOciError(
            "runtime metadata lacks imageio-ffmpeg identity"
        )
    ffmpeg = cast(
        dict[str, Any],
        cast(dict[str, Any], bundled)["imageio-ffmpeg"],
    )
    ffmpeg_path = ffmpeg.get("path")
    if (
        type(ffmpeg_path) is not str
        or not ffmpeg_path.startswith(RUNTIME_ROOT.as_posix() + "/")
        or _SHA256_PATTERN.fullmatch(cast(str, ffmpeg.get("sha256"))) is None
    ):
        raise OfficialForagaxOciError(
            "runtime imageio-ffmpeg identity is invalid"
        )
    root_entries = _entry_map(rootfs)
    ffmpeg_entry = root_entries.get(ffmpeg_path.removeprefix("/"))
    if (
        ffmpeg_entry is None
        or ffmpeg_entry.get("type") != "file"
        or ffmpeg_entry.get("mode") != 0o555
        or ffmpeg_entry.get("sha256") != ffmpeg["sha256"]
    ):
        raise OfficialForagaxOciError(
            "rootfs imageio-ffmpeg executable differs from metadata"
        )
    launcher_bytes = captures[LAUNCHER_PATH.as_posix().removeprefix("/")]
    launcher_sha = hashlib.sha256(launcher_bytes).hexdigest()
    if launcher_sha != expected_build_spec["launcher_sha256"]:
        raise OfficialForagaxOciError("embedded launcher digest differs")
    rootfs_entry = root_entries.get(LAUNCHER_PATH.as_posix().removeprefix("/"))
    if rootfs_entry is None or rootfs_entry.get("mode") != 0o555:
        raise OfficialForagaxOciError("embedded launcher mode is not 0555")
    report: dict[str, Any] = {
        "build_attestation_sha256": hashlib.sha256(build_bytes).hexdigest(),
        "config_sha256": config_sha,
        "image_id": image_id,
        "image_reference_digest": reference_digest,
        "launcher_sha256": launcher_sha,
        "native_runtime_inventory_hash_scheme": TREE_HASH_SCHEME,
        "native_runtime_inventory_root": RUNTIME_ROOT.as_posix(),
        "native_runtime_inventory_sha256": native_sha,
        "rootfs_diff_ids": diff_ids,
        "rootfs_inventory_sha256": rootfs["tree_sha256"],
        "runtime_binary_sha256": runtime["python_executable_sha256"],
        "sbom_sha256": hashlib.sha256(sbom_bytes).hexdigest(),
        "schema_version": BUILD_REPORT_SCHEMA,
        "source_archive_sha256": expected_build_spec[
            "source_archive_sha256"
        ],
    }
    output_directory.mkdir(mode=0o700, parents=True)
    (output_directory / "rootfs-inventory.json").write_bytes(
        _canonical_json_bytes(rootfs, newline=True)
    )
    (output_directory / "image-report.json").write_bytes(
        _canonical_json_bytes(report, newline=True)
    )
    (output_directory / "sbom.spdx.json").write_bytes(sbom_bytes)
    (output_directory / "native-inventory.json").write_bytes(
        captures[
            ATTESTATION_ROOT.as_posix().removeprefix("/")
            + "/native-inventory.json"
        ]
    )
    return report


@dataclasses.dataclass(frozen=True)
class _V4ArchiveSnapshot:
    device: int
    inode: int
    payloads: dict[str, bytes]
    sha256: str
    size: int


def _read_v4_archive(path: Path) -> _V4ArchiveSnapshot:
    """Snapshot and read one canonical launcher USTAR exactly once."""
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise OfficialForagaxOciError(
            "v4 archive must be a directly named regular file"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        size = metadata.st_size
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or size < 1536
            or size > 512 * 1024**2
            or size % 512
        ):
            raise OfficialForagaxOciError(
                "v4 archive framing or file identity is invalid"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            archive_bytes = handle.read(size + 1)
    except OSError as exc:
        raise OfficialForagaxOciError("v4 archive cannot be read") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(archive_bytes) != size:
        raise OfficialForagaxOciError(
            "v4 archive changed while its byte snapshot was read"
        )
    payloads: dict[str, bytes] = {}
    stream_cursor = 0
    try:
        with tarfile.open(
            fileobj=io.BytesIO(archive_bytes),
            mode="r:",
        ) as archive:
            if archive.pax_headers:
                raise OfficialForagaxOciError(
                    "v4 archive contains global PAX metadata"
                )
            for member in archive.getmembers():
                _safe_relative(member.name, label="v4 archive member")
                header = archive_bytes[stream_cursor : stream_cursor + 512]
                if (
                    member.name in payloads
                    or member.offset != stream_cursor
                    or member.offset_data != stream_cursor + 512
                    or header[257:263] != b"ustar\x00"
                    or header[263:265] != b"00"
                    or not member.isfile()
                    or member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}
                    or member.linkname
                    or member.pax_headers
                    or member.sparse
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname
                    or member.gname
                    or member.mtime != 0
                    or stat.S_IMODE(member.mode) != 0o600
                    or member.size < 0
                    or member.size > 2 * 1024**3
                ):
                    raise OfficialForagaxOciError(
                        f"v4 archive metadata is noncanonical: {member.name}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise OfficialForagaxOciError(
                        f"v4 archive member cannot be read: {member.name}"
                    )
                contents = extracted.read(member.size + 1)
                if len(contents) != member.size:
                    raise OfficialForagaxOciError(
                        f"v4 archive member is truncated: {member.name}"
                    )
                payloads[member.name] = contents
                stream_cursor = member.offset_data + (
                    (member.size + 511) // 512
                ) * 512
            trailer_size = size - stream_cursor
            if trailer_size < 1024:
                raise OfficialForagaxOciError(
                    "v4 archive trailer is missing or nonzero"
                )
            trailer_cursor = stream_cursor
            trailer_remaining = trailer_size
            while trailer_remaining:
                block_size = min(1024 * 1024, trailer_remaining)
                block = archive_bytes[
                    trailer_cursor : trailer_cursor + block_size
                ]
                if not block or any(block):
                    raise OfficialForagaxOciError(
                        "v4 archive trailer is missing or nonzero"
                    )
                trailer_cursor += len(block)
                trailer_remaining -= len(block)
    except (OSError, tarfile.TarError) as exc:
        raise OfficialForagaxOciError(
            "v4 output is not a valid canonical USTAR"
        ) from exc
    if not payloads:
        raise OfficialForagaxOciError("v4 archive is empty")
    paths = list(payloads)
    if paths[-2:] != ["stdout.log", "stderr.log"]:
        raise OfficialForagaxOciError(
            "v4 archive does not end in the canonical diagnostic logs"
        )
    for log_path in ("stdout.log", "stderr.log"):
        try:
            decoded = payloads[log_path].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OfficialForagaxOciError(
                f"v4 diagnostic {log_path} is not UTF-8"
            ) from exc
        if "\x00" in decoded:
            raise OfficialForagaxOciError(
                f"v4 diagnostic {log_path} contains NUL bytes"
            )
    return _V4ArchiveSnapshot(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        payloads=payloads,
        sha256=hashlib.sha256(archive_bytes).hexdigest(),
        size=size,
    )


def _canonical_sqlite(
    contents: bytes,
    *,
    expected_seed: int,
) -> tuple[str, dict[str, Any]]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(contents)
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise OfficialForagaxOciError(
                "qualification SQLite integrity check failed"
            )
        schema_rows = connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        if (
            len(schema_rows) != 1
            or schema_rows[0][0:3] != ("table", "_metadata_", "_metadata_")
        ):
            raise OfficialForagaxOciError(
                "qualification SQLite schema is not the exact Foragax table"
            )
        columns_raw = connection.execute(
            'PRAGMA table_info("_metadata_")'
        ).fetchall()
        columns = [
            {
                "cid": int(row[0]),
                "default": row[4],
                "name": str(row[1]),
                "notnull": int(row[3]),
                "pk": int(row[5]),
                "type": str(row[2]),
            }
            for row in columns_raw
        ]
        column_names = [cast(str, column["name"]) for column in columns]
        if column_names != list(OFFICIAL_FORAGAX_RESULTS_DB_COLUMNS):
            raise OfficialForagaxOciError(
                "qualification SQLite columns differ from the exact Foragax "
                "metadata schema"
            )
        seed_rows = connection.execute(
            """
            SELECT seed, id, typeof(seed), typeof(id)
            FROM "_metadata_"
            ORDER BY id, seed
            """
        ).fetchall()
        if seed_rows != [
            (expected_seed, expected_seed, "integer", "integer")
        ]:
            raise OfficialForagaxOciError(
                "qualification SQLite seed/id row differs from the effective seed"
            )
        quoted = ", ".join(
            '"' + name.replace('"', '""') + '"' for name in column_names
        )
        rows_raw = connection.execute(
            f'SELECT {quoted} FROM "_metadata_" ORDER BY "id", "seed"'
        ).fetchall()

        def canonical_value(value: Any) -> Mapping[str, Any]:
            if value is None:
                return {"type": "null", "value": None}
            if type(value) is int:
                return {"type": "integer", "value": value}
            if type(value) is float:
                return {"type": "real", "value": value.hex()}
            if type(value) is str:
                return {"type": "text", "value": value}
            if type(value) is bytes:
                return {
                    "type": "blob",
                    "value": base64.b64encode(value).decode("ascii"),
                }
            raise OfficialForagaxOciError(
                "qualification SQLite contains an unsupported value type"
            )

        payload = {
            "columns": columns,
            "rows": [
                [canonical_value(value) for value in row] for row in rows_raw
            ],
            "schema": [
                {
                    "name": row[1],
                    "sql": row[3],
                    "table": row[2],
                    "type": row[0],
                }
                for row in schema_rows
            ],
        }
        return _json_sha256(payload), payload
    except sqlite3.DatabaseError as exc:
        raise OfficialForagaxOciError(
            "qualification SQLite is invalid"
        ) from exc
    finally:
        connection.close()


def _npz_reward_identity(
    contents: bytes,
    *,
    expected_steps: int,
) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(io.BytesIO(contents), mode="r") as archive:
            infos = archive.infolist()
            if (
                not infos
                or len(infos) != len({info.filename for info in infos})
                or any(
                    info.is_dir()
                    or not info.filename.endswith(".npy")
                    or PurePosixPath(info.filename).name != info.filename
                    or info.flag_bits & 0x1
                    for info in infos
                )
            ):
                raise OfficialForagaxOciError(
                    "qualification NPZ member contract is unsafe"
                )
        with np.load(io.BytesIO(contents), allow_pickle=False) as archive:
            if "rewards" not in archive.files:
                raise OfficialForagaxOciError(
                    "qualification NPZ lacks rewards"
                )
            rewards = np.asarray(archive["rewards"])
            if (
                rewards.dtype.hasobject
                or rewards.dtype.kind not in {"f", "i", "u"}
                or rewards.shape != (expected_steps,)
                or not rewards.flags.c_contiguous
                or not np.all(np.isfinite(rewards))
            ):
                raise OfficialForagaxOciError(
                    "qualification rewards do not match the finite real numeric "
                    "one-value-per-step contract"
                )
            data = rewards.tobytes(order="C")
            return {
                "c_contiguous": True,
                "dtype": rewards.dtype.str,
                "sha256": hashlib.sha256(data).hexdigest(),
                "shape": list(rewards.shape),
            }
    except OfficialForagaxOciError:
        raise
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        raise OfficialForagaxOciError(
            "qualification result is not a strict NPZ"
        ) from exc


def _qualification_result_layout(
    paths: Sequence[str],
    *,
    effective_seed: int,
) -> tuple[str, str]:
    """Derive one single-run result NPZ and its sibling database."""
    canonical_paths = [
        _safe_relative(path, label="qualification result member")
        for path in paths
    ]
    result_paths = [
        path
        for path in canonical_paths
        if path.suffix == ".npz"
    ]
    database_paths = [
        path
        for path in canonical_paths
        if path.suffix == ".db"
    ]
    if len(result_paths) != 1 or len(database_paths) != 1:
        raise OfficialForagaxOciError(
            "qualification must contain exactly one result NPZ and one database"
        )
    result_path = result_paths[0]
    root = result_path.parent.parent
    if (
        len(result_path.parts) < 4
        or result_path.parts[0] != "official-results"
        or result_path.parent.name != "data"
        or result_path.name != f"{effective_seed}.npz"
        or len(root.parts) < 2
    ):
        raise OfficialForagaxOciError(
            "qualification result NPZ must be "
            "<root>/data/<effective-seed>.npz"
        )
    expected_database = root / "results.db"
    if database_paths[0] != expected_database:
        raise OfficialForagaxOciError(
            "qualification database must be the result NPZ's sibling results.db"
        )
    expected_paths = [
        result_path,
        expected_database,
        PurePosixPath("stdout.log"),
        PurePosixPath("stderr.log"),
    ]
    if canonical_paths != expected_paths:
        raise OfficialForagaxOciError(
            "qualification archive members must be exactly the ordered result "
            "NPZ, sibling results.db, stdout.log, and stderr.log"
        )
    return result_path.as_posix(), expected_database.as_posix()


def _qualification_workload_identity(
    value: Mapping[str, Any],
    *,
    backend: str,
    config_sha256: str,
    effective_seed: int,
    steps: int,
    result_path: str,
    database_path: str,
) -> dict[str, Any]:
    """Validate the reviewable workload projection behind its digest."""
    workload = dict(value)
    _exact_keys(
        workload,
        {
            "backend",
            "configuration",
            "entrypoint",
            "invocation",
            "run",
            "schema_version",
        },
        label="qualification workload identity",
    )
    if workload["schema_version"] != QUALIFICATION_WORKLOAD_SCHEMA:
        raise OfficialForagaxOciError(
            "qualification workload schema is unsupported"
        )
    backend_identity = workload["backend"]
    if type(backend_identity) is not dict:
        raise OfficialForagaxOciError(
            "qualification workload backend is not an object"
        )
    backend_identity = cast(dict[str, Any], backend_identity)
    _exact_keys(
        backend_identity,
        {"kind", "launcher_contract", "runtime_arguments"},
        label="qualification workload backend",
    )
    runtime_arguments = backend_identity["runtime_arguments"]
    if (
        backend_identity["kind"] != backend
        or backend_identity["launcher_contract"]
        != OFFICIAL_FORAGAX_OCI_LAUNCHER_CONTRACT
        or type(runtime_arguments) is not list
        or not all(type(argument) is str and argument for argument in runtime_arguments)
    ):
        raise OfficialForagaxOciError(
            "qualification workload backend contract differs from the run"
        )
    if backend == "cpu" and runtime_arguments != [
        "--env=JAX_PLATFORM_NAME=cpu",
        "--env=JAX_PLATFORMS=cpu",
        "--env=JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1",
    ]:
        raise OfficialForagaxOciError(
            "qualification CPU workload runtime arguments are not canonical"
        )

    configuration = workload["configuration"]
    if type(configuration) is not dict:
        raise OfficialForagaxOciError(
            "qualification workload configuration is not an object"
        )
    configuration = cast(dict[str, Any], configuration)
    _exact_keys(
        configuration,
        {
            "agent",
            "config_path",
            "config_sha256",
            "entrypoint_family",
            "problem",
        },
        label="qualification workload configuration",
    )
    config_path = configuration["config_path"]
    parsed_config_path = (
        PurePosixPath(config_path) if type(config_path) is str else None
    )
    if (
        type(configuration["agent"]) is not str
        or not configuration["agent"]
        or configuration["entrypoint_family"] not in {"continuing", "ppo"}
        or configuration["problem"] != "Foragax"
        or configuration["config_sha256"] != config_sha256
        or parsed_config_path is None
        or not parsed_config_path.is_absolute()
        or parsed_config_path.as_posix() != config_path
        or ".." in parsed_config_path.parts
        or config_path.startswith(("/tmp/", "/run/"))
    ):
        raise OfficialForagaxOciError(
            "qualification workload configuration differs from the run"
        )

    entrypoint = workload["entrypoint"]
    if type(entrypoint) is not dict:
        raise OfficialForagaxOciError(
            "qualification workload entrypoint is not an object"
        )
    entrypoint = cast(dict[str, Any], entrypoint)
    _exact_keys(
        entrypoint,
        {"family", "path", "sha256"},
        label="qualification workload entrypoint",
    )
    entrypoint_path = entrypoint["path"]
    parsed_entrypoint = (
        _safe_relative(
            entrypoint_path,
            label="qualification workload entrypoint path",
        )
        if type(entrypoint_path) is str
        else None
    )
    if (
        entrypoint["family"] != configuration["entrypoint_family"]
        or parsed_entrypoint is None
        or not parsed_entrypoint.parts
        or parsed_entrypoint.parts[0] != "src"
    ):
        raise OfficialForagaxOciError(
            "qualification workload entrypoint differs from the configuration"
        )
    _require_sha256(
        entrypoint["sha256"],
        label="qualification workload entrypoint",
    )

    run = workload["run"]
    if type(run) is not dict:
        raise OfficialForagaxOciError(
            "qualification workload run is not an object"
        )
    run = cast(dict[str, Any], run)
    _exact_keys(
        run,
        {
            "applied_seed_offset",
            "applied_seed_offset_source",
            "effective_seed",
            "index",
            "nested_seed_offset",
            "stored_seed",
            "top_level_seed_offset",
        },
        label="qualification workload run",
    )
    integer_keys = (
        "applied_seed_offset",
        "effective_seed",
        "index",
        "nested_seed_offset",
        "stored_seed",
        "top_level_seed_offset",
    )
    if any(type(run[key]) is not int for key in integer_keys):
        raise OfficialForagaxOciError(
            "qualification workload run seed fields are not integers"
        )
    offset_source = run["applied_seed_offset_source"]
    source_key = {
        "nested": "nested_seed_offset",
        "top_level": "top_level_seed_offset",
    }.get(offset_source)
    if (
        source_key is None
        or run["applied_seed_offset"] != run[source_key]
        or run["stored_seed"] + run["applied_seed_offset"]
        != run["effective_seed"]
        or run["index"] != effective_seed
        or run["effective_seed"] != effective_seed
    ):
        raise OfficialForagaxOciError(
            "qualification workload seed arithmetic differs from the run"
        )

    invocation = workload["invocation"]
    if type(invocation) is not dict:
        raise OfficialForagaxOciError(
            "qualification workload invocation is not an object"
        )
    invocation = cast(dict[str, Any], invocation)
    _exact_keys(
        invocation,
        {
            "expected_result_env_steps",
            "index_expression",
            "indices",
            "max_steps_argument",
            "members",
        },
        label="qualification workload invocation",
    )
    expected_members = [
        {
            "content_policy": "strict_npz",
            "path": result_path,
            "role": "result_npz",
        },
        {
            "content_policy": "sqlite_foragax_metadata_v1",
            "path": database_path,
            "role": "auxiliary",
        },
        {
            "content_policy": "bounded_utf8_log",
            "path": "stdout.log",
            "role": "stdout_log",
        },
        {
            "content_policy": "bounded_utf8_diagnostic",
            "path": "stderr.log",
            "role": "stderr_log",
        },
    ]
    if (
        invocation["index_expression"] != str(effective_seed)
        or invocation["indices"] != [effective_seed]
        or invocation["expected_result_env_steps"] != steps
        or (
            invocation["max_steps_argument"] is not None
            and (
                type(invocation["max_steps_argument"]) is not int
                or invocation["max_steps_argument"] < 1
            )
        )
        or invocation["members"] != expected_members
    ):
        raise OfficialForagaxOciError(
            "qualification workload invocation differs from the archive"
        )
    return workload


def qualify_v4_runs(
    first_archive: Path,
    second_archive: Path,
    *,
    backend: str,
    image_id: str,
    runtime_profile_id: str,
    effective_seed: int,
    steps: int,
    config_sha256: str,
    source_archive_sha256: str,
    workload_identity: Mapping[str, Any],
    environment_profile_sha256: str,
) -> dict[str, Any]:
    """Seal exact two-run deterministic evidence without granting endorsement."""
    if backend not in {"cpu", "gpu"}:
        raise OfficialForagaxOciError(
            "qualification backend must be exactly cpu or gpu"
        )
    if _OCI_DIGEST_PATTERN.fullmatch(image_id) is None:
        raise OfficialForagaxOciError("qualification image ID is invalid")
    if not runtime_profile_id:
        raise OfficialForagaxOciError(
            "qualification runtime profile ID is empty"
        )
    if (
        type(effective_seed) is not int
        or effective_seed < 0
        or effective_seed > (1 << 32) - 1
        or type(steps) is not int
        or steps < 1
    ):
        raise OfficialForagaxOciError(
            "qualification seed/horizon binding is invalid"
        )
    for value, label in (
        (config_sha256, "configuration"),
        (source_archive_sha256, "source archive"),
        (environment_profile_sha256, "environment profile"),
    ):
        _require_sha256(value, label=f"qualification {label}")
    first_snapshot = _read_v4_archive(first_archive)
    second_snapshot = _read_v4_archive(second_archive)
    if (
        first_snapshot.device,
        first_snapshot.inode,
    ) == (
        second_snapshot.device,
        second_snapshot.inode,
    ):
        raise OfficialForagaxOciError(
            "qualification requires two distinct archive file identities"
        )
    first = first_snapshot.payloads
    second = second_snapshot.payloads
    if list(first) != list(second):
        raise OfficialForagaxOciError(
            "qualification archive member paths/order differ"
        )
    result_path, database_path = _qualification_result_layout(
        list(first),
        effective_seed=effective_seed,
    )
    normalized_workload_identity = _qualification_workload_identity(
        workload_identity,
        backend=backend,
        config_sha256=config_sha256,
        effective_seed=effective_seed,
        steps=steps,
        result_path=result_path,
        database_path=database_path,
    )
    workload_identity_sha256 = _json_sha256(normalized_workload_identity)
    result_paths = [result_path]
    scientific_map: list[dict[str, Any]] = []
    rewards: list[dict[str, Any]] = []
    sqlite_details: dict[str, Any] | None = None
    diagnostics: list[dict[str, Any]] = []
    for path in first:
        first_bytes = first[path]
        second_bytes = second[path]
        first_sha = hashlib.sha256(first_bytes).hexdigest()
        second_sha = hashlib.sha256(second_bytes).hexdigest()
        if path in {"stdout.log", "stderr.log"}:
            diagnostics.append(
                {
                    "first_sha256": first_sha,
                    "path": path,
                    "second_sha256": second_sha,
                }
            )
            continue
        if path == database_path:
            first_canonical_sha, first_canonical = _canonical_sqlite(
                first_bytes,
                expected_seed=effective_seed,
            )
            second_canonical_sha, second_canonical = _canonical_sqlite(
                second_bytes,
                expected_seed=effective_seed,
            )
            if (
                first_canonical_sha != second_canonical_sha
                or first_canonical != second_canonical
            ):
                raise OfficialForagaxOciError(
                    "qualification SQLite canonical schema/table/values differ"
                )
            classification = (
                "raw_and_canonical_equal"
                if first_sha == second_sha
                else "canonical_equal_raw_diff"
            )
            sqlite_details = {
                "canonical_payload_sha256": first_canonical_sha,
                "classification": classification,
                "first_raw_sha256": first_sha,
                "first_size": len(first_bytes),
                "second_raw_sha256": second_sha,
                "second_size": len(second_bytes),
            }
            scientific_map.append(
                {
                    "canonical_sha256": first_canonical_sha,
                    "path": path,
                    "payload_kind": "sqlite_foragax_metadata_v1",
                }
            )
            continue
        if first_bytes != second_bytes:
            raise OfficialForagaxOciError(
                f"qualification deterministic member bytes differ: {path}"
            )
        payload_kind = "strict_npz" if path in result_paths else "opaque"
        scientific_map.append(
            {
                "path": path,
                "payload_kind": payload_kind,
                "sha256": first_sha,
                "size": len(first_bytes),
            }
        )
        if path in result_paths:
            first_reward = _npz_reward_identity(
                first_bytes,
                expected_steps=steps,
            )
            second_reward = _npz_reward_identity(
                second_bytes,
                expected_steps=steps,
            )
            if first_reward != second_reward:
                raise OfficialForagaxOciError(
                    f"qualification reward identity differs: {path}"
                )
            rewards.append({"path": path, **first_reward})
    if sqlite_details is None:
        raise OfficialForagaxOciError(
            "qualification archives lack the exact results SQLite"
        )
    if len(result_paths) == 1:
        artifact_sha = hashlib.sha256(first[result_paths[0]]).hexdigest()
        rewards_sha = cast(str, rewards[0]["sha256"])
    else:
        artifact_sha = _json_sha256(
            [
                {
                    "path": path,
                    "sha256": hashlib.sha256(first[path]).hexdigest(),
                }
                for path in result_paths
            ]
        )
        rewards_sha = _json_sha256(rewards)
    member_payloads_sha = _json_sha256(scientific_map)
    bindings = {
        "backend": backend,
        "config_sha256": config_sha256,
        "effective_seed": effective_seed,
        "environment_profile_sha256": environment_profile_sha256,
        "image_id": image_id,
        "runtime_profile_id": runtime_profile_id,
        "source_archive_sha256": source_archive_sha256,
        "steps": steps,
        "workload_identity_sha256": workload_identity_sha256,
    }
    evidence_core = {
        "bindings": bindings,
        "diagnostic_logs": diagnostics,
        "first_archive_sha256": first_snapshot.sha256,
        "member_payloads": scientific_map,
        "rewards": rewards,
        "schema_version": QUALIFICATION_ENVELOPE_SCHEMA,
        "second_archive_sha256": second_snapshot.sha256,
        "sqlite": sqlite_details,
        "workload_identity": normalized_workload_identity,
    }
    evidence_sha = _json_sha256(evidence_core)
    qualification = {
        "artifact_sha256": artifact_sha,
        "backend": backend,
        "config_sha256": config_sha256,
        "effective_seed": effective_seed,
        "environment_profile_sha256": environment_profile_sha256,
        "evidence_envelope_sha256": evidence_sha,
        "executor_kind": "oci",
        "image_id": image_id,
        "member_payloads_sha256": member_payloads_sha,
        "repeat_count": 2,
        "rewards_sha256": rewards_sha,
        "runtime_profile_id": runtime_profile_id,
        "schema_version": OFFICIAL_FORAGAX_DETERMINISM_QUALIFICATION_SCHEMA,
        "seed_class": "open_development",
        "source_archive_sha256": source_archive_sha256,
        "state": "sealed_oci_two_run_exact",
        "steps": steps,
        "workload_identity_sha256": workload_identity_sha256,
    }
    return {
        "evidence": evidence_core,
        "evidence_envelope_sha256": evidence_sha,
        "qualification": qualification,
        "qualification_sha256": _json_sha256(qualification),
    }


_CPU_PROBE = r"""
import hashlib, json, os, stat, sys
from pathlib import Path
import imageio_ffmpeg
import jax
source = Path("/opt/continual-foragax-agents")
runtime = Path("/opt/alberta-runtime")
tmp_src = Path("/tmp/src")
scratch_expected = {
    "CUDA_CACHE_PATH": "/run/alberta/cuda-cache",
    "HOME": "/run/alberta/home",
    "JAX_COMPILATION_CACHE_DIR": "/run/alberta/jax-cache",
    "MPLCONFIGDIR": "/run/alberta/matplotlib",
    "TMPDIR": "/run/alberta/tmp",
    "XDG_CACHE_HOME": "/run/alberta/cache",
}
def write_fails(path):
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError:
        return True
    else:
        os.close(descriptor)
        Path(path).unlink()
        return False
scratch = {}
for variable, expected in scratch_expected.items():
    path = Path(expected)
    metadata = path.stat()
    probe = path / (".probe-" + str(os.getpid()))
    if write_fails(probe):
        raise RuntimeError("scratch directory is not writable: " + variable)
    scratch[variable] = {
        "environment": os.environ.get(variable),
        "is_mount": path.is_mount(),
        "mode": stat.S_IMODE(metadata.st_mode),
    }
base_sys_path = []
for value in sys.path:
    if not value or not Path(value).is_absolute():
        raise RuntimeError("isolated sys.path contains a relative entry")
    resolved = Path(value).resolve()
    writable = resolved.exists() and os.access(resolved, os.W_OK)
    if writable or resolved.as_posix().startswith(("/tmp/", "/run/")):
        raise RuntimeError("isolated sys.path contains writable scratch")
    base_sys_path.append(resolved.as_posix())
ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
ffmpeg_metadata = ffmpeg.stat()
payload = {
    "base_sys_path": base_sys_path,
    "cwd": Path.cwd().as_posix(),
    "ffmpeg": {
        "mode": stat.S_IMODE(ffmpeg_metadata.st_mode),
        "path": ffmpeg.as_posix(),
        "sha256": hashlib.sha256(ffmpeg.read_bytes()).hexdigest(),
    },
    "gid": os.getgid(),
    "jax_backend": jax.default_backend(),
    "nvidia_device_paths": sorted(
        path.as_posix() for path in Path("/dev").glob("nvidia*")
    ),
    "python_flags": {
        "dont_write_bytecode": sys.flags.dont_write_bytecode,
        "isolated": sys.flags.isolated,
        "no_user_site": sys.flags.no_user_site,
        "safe_path": sys.flags.safe_path,
    },
    "runtime_write_fails": write_fails(runtime / ".probe"),
    "scratch": scratch,
    "source_write_fails": write_fails(source / ".probe"),
    "tmp_root_write_fails": write_fails(Path("/tmp") / ".probe"),
    "tmp_src": {
        "entries": sorted(path.name for path in tmp_src.iterdir()),
        "is_mount": tmp_src.is_mount(),
        "mode": stat.S_IMODE(tmp_src.stat().st_mode),
        "write_fails": write_fails(tmp_src / ".probe"),
    },
    "uid": os.getuid(),
}
print(json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True))
"""


def cpu_probe_command(
    image_id: str,
    *,
    docker: Path = Path("/usr/bin/docker"),
) -> tuple[str, ...]:
    if _OCI_DIGEST_PATTERN.fullmatch(image_id) is None:
        raise OfficialForagaxOciError("CPU probe image ID is invalid")
    return (
        str(docker),
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65532:65532",
        "--pids-limit=512",
        *_scratch_arguments(),
        f"--workdir={SOURCE_ROOT}",
        *_environment_arguments(),
        "--env=JAX_PLATFORM_NAME=cpu",
        "--env=JAX_PLATFORMS=cpu",
        "--env=JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1",
        image_id,
        PYTHON_EXECUTABLE.as_posix(),
        "-I",
        "-B",
        "-c",
        _CPU_PROBE,
    )


def run_cpu_probe(
    image_id: str,
    *,
    docker: Path = Path("/usr/bin/docker"),
) -> dict[str, Any]:
    output = _run_bytes(cpu_probe_command(image_id, docker=docker))
    value = _strict_json_bytes(output, label="CPU isolation probe")
    if type(value) is not dict:
        raise OfficialForagaxOciError("CPU isolation probe is not an object")
    probe = cast(dict[str, Any], value)
    expected_flags = {
        "dont_write_bytecode": 1,
        "isolated": 1,
        "no_user_site": 1,
        "safe_path": True,
    }
    tmp_src = probe.get("tmp_src")
    scratch = probe.get("scratch")
    if (
        probe.get("uid") != 65532
        or probe.get("gid") != 65532
        or probe.get("cwd") != SOURCE_ROOT.as_posix()
        or probe.get("jax_backend") != "cpu"
        or probe.get("nvidia_device_paths") != []
        or probe.get("python_flags") != expected_flags
        or probe.get("runtime_write_fails") is not True
        or probe.get("source_write_fails") is not True
        or probe.get("tmp_root_write_fails") is not True
        or type(tmp_src) is not dict
        or cast(dict[str, Any], tmp_src)
        != {
            "entries": [],
            "is_mount": True,
            "mode": 0o555,
            "write_fails": True,
        }
        or type(scratch) is not dict
        or any(
            cast(dict[str, Any], scratch).get(variable)
            != {
                "environment": path,
                "is_mount": True,
                "mode": 0o700,
            }
            for variable, path in {
                "CUDA_CACHE_PATH": "/run/alberta/cuda-cache",
                "HOME": "/run/alberta/home",
                "JAX_COMPILATION_CACHE_DIR": "/run/alberta/jax-cache",
                "MPLCONFIGDIR": "/run/alberta/matplotlib",
                "TMPDIR": "/run/alberta/tmp",
                "XDG_CACHE_HOME": "/run/alberta/cache",
            }.items()
        )
    ):
        raise OfficialForagaxOciError(
            "CPU nonroot/read-only/scratch/import isolation contract failed"
        )
    ffmpeg = probe.get("ffmpeg")
    if (
        type(ffmpeg) is not dict
        or cast(dict[str, Any], ffmpeg).get("mode") != 0o555
        or not cast(str, cast(dict[str, Any], ffmpeg).get("path", "")).startswith(
            RUNTIME_ROOT.as_posix() + "/"
        )
        or _SHA256_PATTERN.fullmatch(
            cast(str, cast(dict[str, Any], ffmpeg).get("sha256"))
        )
        is None
    ):
        raise OfficialForagaxOciError(
            "CPU probe imageio-ffmpeg identity failed"
        )
    return probe


def _load_prepared_context(context: Path) -> PreparedOciBuild:
    value = _strict_json_file(
        context / "build-attestation.json",
        label="prepared build attestation",
    )
    if type(value) is not dict:
        raise OfficialForagaxOciError(
            "prepared build attestation is not an object"
        )
    build = cast(dict[str, Any], value)
    if build.get("schema_version") != BUILD_SPEC_SCHEMA:
        raise OfficialForagaxOciError(
            "prepared build attestation schema is unsupported"
        )
    return PreparedOciBuild(
        context=context,
        build_spec=build,
        build_spec_sha256=_sha256(context / "build-attestation.json"),
        dockerfile_sha256=_sha256(context / "Dockerfile"),
        launcher_sha256=cast(str, build["launcher_sha256"]),
    )


def _write_stdout(value: Any) -> None:
    sys.stdout.buffer.write(_canonical_json_bytes(value, newline=True))


def _write_new_file(path: Path, contents: bytes) -> None:
    """Create one directly named file without following or replacing anything."""
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as exc:
        raise OfficialForagaxOciError(
            f"qualification output cannot be created exclusively: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", allow_abbrev=False)
    prepare.add_argument("--source-archive", type=Path, required=True)
    prepare.add_argument("--source-archive-sha256", required=True)
    prepare.add_argument("--dependency-lock", type=Path, required=True)
    prepare.add_argument("--dependency-lock-sha256", required=True)
    prepare.add_argument(
        "--source-commit",
        default=OFFICIAL_FORAGAX_AUDIT_COMMIT,
    )
    prepare.add_argument("--source-tree-git-sha1", required=True)
    prepare.add_argument("--base-image", default=_AUDITED_BASE_IMAGE)
    prepare.add_argument("--uv-binary", type=Path, required=True)
    prepare.add_argument("--uv-binary-sha256", default=UV_BINARY_SHA256)
    prepare.add_argument("--uv-cache-archive", type=Path, required=True)
    prepare.add_argument("--uv-cache-archive-sha256", required=True)
    prepare.add_argument("--debian-bundle", type=Path, required=True)
    prepare.add_argument("--debian-manifest", type=Path, required=True)
    prepare.add_argument("--output-context", type=Path, required=True)
    cache = subparsers.add_parser("archive-cache", allow_abbrev=False)
    cache.add_argument("--cache-root", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    build = subparsers.add_parser("build", allow_abbrev=False)
    build.add_argument("--context", type=Path, required=True)
    build.add_argument("--tag", required=True)
    build.add_argument("--docker", type=Path, default=Path("/usr/bin/docker"))
    inspect = subparsers.add_parser("inspect", allow_abbrev=False)
    inspect.add_argument("--context", type=Path, required=True)
    inspect.add_argument("--image", required=True)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.add_argument("--docker", type=Path, default=Path("/usr/bin/docker"))
    probe = subparsers.add_parser("cpu-probe", allow_abbrev=False)
    probe.add_argument("--image-id", required=True)
    probe.add_argument("--docker", type=Path, default=Path("/usr/bin/docker"))
    launch = subparsers.add_parser("emit-launch", allow_abbrev=False)
    launch.add_argument("--image-id", required=True)
    launch.add_argument("--entrypoint", required=True)
    launch.add_argument("--config", required=True)
    launch.add_argument("--index", required=True)
    launch.add_argument("--max-steps", type=int)
    launch.add_argument("--gpu", action="store_true")
    launch.add_argument("--driver-host-path")
    launch.add_argument("--driver-container-path")
    launch.add_argument("--device-path", action="append")
    launch.add_argument("--device-index", action="append", type=int)
    launch.add_argument("--cuda-wheel-library-path", action="append")
    launch.add_argument("--driver-user-library-path", action="append")
    launch.add_argument("--docker", type=Path, default=Path("/usr/bin/docker"))
    qualify = subparsers.add_parser("qualify", allow_abbrev=False)
    qualify.add_argument("--first", type=Path, required=True)
    qualify.add_argument("--second", type=Path, required=True)
    qualify.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    qualify.add_argument("--image-id", required=True)
    qualify.add_argument("--runtime-profile-id", required=True)
    qualify.add_argument("--effective-seed", type=int, required=True)
    qualify.add_argument("--steps", type=int, required=True)
    qualify.add_argument("--config-sha256", required=True)
    qualify.add_argument("--source-archive-sha256", required=True)
    qualify.add_argument("--workload-identity", type=Path, required=True)
    qualify.add_argument("--environment-profile-sha256", required=True)
    qualify.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    try:
        args = _main_parser().parse_args()
        if args.command == "prepare":
            prepared = prepare_build_context(
                OciBuildInputs(
                    source_archive=args.source_archive,
                    source_archive_sha256=args.source_archive_sha256,
                    dependency_lock=args.dependency_lock,
                    dependency_lock_sha256=args.dependency_lock_sha256,
                    source_commit=args.source_commit,
                    source_tree_git_sha1=args.source_tree_git_sha1,
                    base_image=args.base_image,
                    uv_binary=args.uv_binary,
                    uv_binary_sha256=args.uv_binary_sha256,
                    uv_cache_archive=args.uv_cache_archive,
                    uv_cache_archive_sha256=args.uv_cache_archive_sha256,
                    debian_bundle=args.debian_bundle,
                    debian_manifest=args.debian_manifest,
                    output_context=args.output_context,
                )
            )
            _write_stdout(
                {
                    "build_spec_sha256": prepared.build_spec_sha256,
                    "context": prepared.context.as_posix(),
                    "dockerfile_sha256": prepared.dockerfile_sha256,
                    "launcher_sha256": prepared.launcher_sha256,
                }
            )
        elif args.command == "archive-cache":
            digest = create_regular_cache_archive(
                args.cache_root,
                args.output,
            )
            _write_stdout({"sha256": digest})
        elif args.command == "build":
            prepared = _load_prepared_context(args.context)
            build_image(
                prepared,
                image_tag=args.tag,
                docker=args.docker,
            )
            _write_stdout(
                {
                    "argv": list(
                        docker_build_command(
                            prepared,
                            image_tag=args.tag,
                            docker=args.docker,
                        )
                    )
                }
            )
        elif args.command == "inspect":
            prepared = _load_prepared_context(args.context)
            _write_stdout(
                inspect_image(
                    image=args.image,
                    expected_build_spec=prepared.build_spec,
                    output_directory=args.output,
                    docker=args.docker,
                )
            )
        elif args.command == "cpu-probe":
            _write_stdout(run_cpu_probe(args.image_id, docker=args.docker))
        elif args.command == "emit-launch":
            driver_values = (
                args.driver_host_path,
                args.driver_container_path,
                args.device_path,
                args.device_index,
                args.cuda_wheel_library_path,
                args.driver_user_library_path,
            )
            driver: DriverLaunchContract | None
            if args.gpu:
                if any(value is None for value in driver_values):
                    raise OfficialForagaxOciError(
                        "GPU CLI launch requires the complete driver contract"
                    )
                driver = DriverLaunchContract(
                    driver_host_path=args.driver_host_path,
                    driver_container_path=args.driver_container_path,
                    device_paths=tuple(args.device_path),
                    device_indices=tuple(args.device_index),
                    cuda_wheel_library_paths=tuple(
                        args.cuda_wheel_library_path
                    ),
                    driver_user_library_paths=tuple(
                        args.driver_user_library_path
                    ),
                )
            else:
                if any(value is not None for value in driver_values):
                    raise OfficialForagaxOciError(
                        "CPU CLI launch rejects a driver contract"
                    )
                driver = None
            command = emit_launch_command(
                image_id=args.image_id,
                entrypoint=args.entrypoint,
                config_path=args.config,
                index_expression=args.index,
                gpu=args.gpu,
                max_steps=args.max_steps,
                driver=driver,
                docker=args.docker,
            )
            _write_stdout(
                {"argv": list(command), "shell": shlex.join(command)}
            )
        elif args.command == "qualify":
            result = qualify_v4_runs(
                args.first,
                args.second,
                backend=args.backend,
                image_id=args.image_id,
                runtime_profile_id=args.runtime_profile_id,
                effective_seed=args.effective_seed,
                steps=args.steps,
                config_sha256=args.config_sha256,
                source_archive_sha256=args.source_archive_sha256,
                workload_identity=_strict_json_file(
                    args.workload_identity,
                    label="qualification workload identity",
                ),
                environment_profile_sha256=args.environment_profile_sha256,
            )
            _write_new_file(
                args.output,
                _canonical_json_bytes(result, newline=True),
            )
            _write_stdout(
                {
                    "evidence_envelope_sha256": result[
                        "evidence_envelope_sha256"
                    ],
                    "qualification_sha256": result[
                        "qualification_sha256"
                    ],
                }
            )
        else:  # pragma: no cover - argparse owns command choices
            raise OfficialForagaxOciError("unsupported OCI utility command")
        return 0
    except (OfficialForagaxOciError, OSError, subprocess.SubprocessError) as exc:
        sys.stderr.write(f"official Foragax OCI: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
