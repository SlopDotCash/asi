#!/usr/bin/python3.12
"""Standard-library image construction helper for the official Foragax OCI.

Runs inside the network-disabled BuildKit stages of the
:mod:`official_foragax_oci` image build, bind-mounted read-only as
``/input/image-helper.py``.  It executes under ``python3.12 -I -B`` before
the scientific environment exists, so only the standard library may be
imported.  The helper is never copied into the final image, but its exact
bytes are hashed into the build spec (``image_helper_sha256``), so any edit
here changes the identity of future image builds.

Subcommands mirror the build stages: ``extract-source`` / ``extract-cache``
unpack the validated inputs, ``verify-debian`` checks installed apt state
against the audited Debian bundle, and ``finalize`` hardens the sealed
layout and writes the attestation set under ``/opt/alberta-attestations``.
All outputs use canonical JSON and a reconstructible tree hashing scheme
shared with the runtime verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import posixpath
import re
import stat
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

TREE_HASH_SCHEME = "canonical-entry-json+mode+size+bytes-v1"
DEBIAN_SCHEMA = "alberta.official_foragax.debian_bundle.v1"
NATIVE_INVENTORY_SCHEMA = "alberta.official_foragax.native_inventory.v1"
RUNTIME_METADATA_SCHEMA = "alberta.official_foragax.runtime_metadata.v1"
SOURCE_ROOT = Path("/opt/continual-foragax-agents")
RUNTIME_ROOT = Path("/opt/alberta-runtime")
ATTESTATION_ROOT = Path("/opt/alberta-attestations")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_exact_str(name: object, value: object) -> str:
    if type(name) is not str:
        raise ImageHelperError("name must be an exact string")
    if type(value) is not str:
        raise ImageHelperError(f"{name} must be an exact string")
    return value


class ImageHelperError(RuntimeError):
    """Fail-closed image construction error."""


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value).rstrip(b"\n")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            host_key = _require_exact_str("key", key)
            if host_key in result:
                raise ImageHelperError(f"{path} repeats JSON key")
            result[host_key] = value
        return result

    def parse_float(value: str) -> float:
        host_value = _require_exact_str("value", value)
        parsed = float(host_value)
        if not math.isfinite(parsed):
            raise ImageHelperError(f"{path} contains non-finite JSON number")
        return parsed

    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ImageHelperError(f"{path} contains JSON constant")
            ),
            parse_float=parse_float,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImageHelperError(f"{path} is not strict JSON") from exc
    if type(value) is not dict:
        raise ImageHelperError(f"{path} must contain a JSON object")
    return value


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
        raise ImageHelperError(f"{label} is not a canonical relative path")
    return path


def extract_source(
    archive_path: Path,
    destination: Path,
    *,
    commit: str,
    lock_sha256: str,
) -> None:
    """Safely extract the validated deterministic Git archive."""
    if destination.exists():
        raise ImageHelperError(f"source destination already exists: {destination}")
    destination.mkdir(mode=0o755, parents=True)
    symlinks: set[PurePosixPath] = set()
    seen: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            if archive.pax_headers != {"comment": commit}:
                raise ImageHelperError("source archive commit PAX header differs")
            for member in archive:
                relative = _safe_relative(member.name, label="source archive member")
                if member.name in seen:
                    raise ImageHelperError("source archive repeats a member")
                seen.add(member.name)
                if any(parent in symlinks for parent in relative.parents):
                    raise ImageHelperError(
                        f"source member traverses a symlink: {member.name}"
                    )
                target = destination.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(mode=member.mode & 0o777, parents=False)
                elif member.isfile() and member.type in {
                    tarfile.REGTYPE,
                    tarfile.AREGTYPE,
                }:
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ImageHelperError(
                            f"source member cannot be read: {member.name}"
                        )
                    with target.open("xb") as output:
                        remaining = member.size
                        while remaining:
                            block = extracted.read(min(1024 * 1024, remaining))
                            if not block:
                                raise ImageHelperError(
                                    f"source member is truncated: {member.name}"
                                )
                            output.write(block)
                            remaining -= len(block)
                        if extracted.read(1):
                            raise ImageHelperError(
                                f"source member exceeds its header: {member.name}"
                            )
                    target.chmod(member.mode & 0o777)
                elif member.issym():
                    link = _safe_relative(
                        member.linkname,
                        label=f"source symlink target {member.name}",
                    )
                    if link.is_absolute() or ".." in link.parts:
                        raise ImageHelperError(
                            f"source symlink escapes the tree: {member.name}"
                        )
                    target.symlink_to(member.linkname)
                    symlinks.add(relative)
                else:
                    raise ImageHelperError(
                        f"source archive contains forbidden member {member.name}"
                    )
    except (OSError, tarfile.TarError) as exc:
        raise ImageHelperError("source archive cannot be extracted safely") from exc
    lock = destination / "uv.lock"
    if not lock.is_file() or _sha256(lock) != lock_sha256:
        raise ImageHelperError("extracted source lock differs from the frozen lock")


def _cache_link_target(member_name: str, link_name: str) -> str:
    if not link_name or PurePosixPath(link_name).is_absolute():
        raise ImageHelperError(f"cache symlink is absolute: {member_name}")
    normalized = posixpath.normpath(
        posixpath.join(posixpath.dirname(member_name), link_name)
    )
    if normalized == ".." or normalized.startswith("../"):
        raise ImageHelperError(f"cache symlink escapes the tree: {member_name}")
    return normalized


def extract_regular_tree_archive(archive_path: Path, destination: Path) -> None:
    """Extract a host-validated cache archive containing dirs/files/symlinks."""
    if destination.exists():
        raise ImageHelperError(f"cache destination already exists: {destination}")
    destination.mkdir(mode=0o700, parents=True)
    seen: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                relative = _safe_relative(member.name, label="cache archive member")
                if member.name in seen:
                    raise ImageHelperError("cache archive repeats a member")
                seen.add(member.name)
                target = destination.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(mode=0o700, parents=False)
                elif member.isfile() and member.type in {
                    tarfile.REGTYPE,
                    tarfile.AREGTYPE,
                }:
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ImageHelperError(
                            f"cache member cannot be read: {member.name}"
                        )
                    with target.open("xb") as output:
                        remaining = member.size
                        while remaining:
                            block = extracted.read(min(1024 * 1024, remaining))
                            if not block:
                                raise ImageHelperError(
                                    f"cache member is truncated: {member.name}"
                                )
                            output.write(block)
                            remaining -= len(block)
                    target.chmod(0o700 if member.mode & 0o111 else 0o600)
                elif member.issym() and member.type == tarfile.SYMTYPE:
                    _cache_link_target(member.name, member.linkname)
                    target.symlink_to(member.linkname)
                else:
                    raise ImageHelperError(
                        f"cache archive contains forbidden member {member.name}"
                    )
    except (OSError, tarfile.TarError) as exc:
        raise ImageHelperError("cache archive cannot be extracted safely") from exc


def _tree_entries(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise ImageHelperError(f"inventory root is not a directory: {root}")
    entries: list[dict[str, Any]] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        names = sorted([*directories, *filenames])
        for name in names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISDIR(metadata.st_mode):
                entry: dict[str, Any] = {
                    "mode": mode,
                    "path": relative,
                    "type": "directory",
                }
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise ImageHelperError(
                        f"inventory contains a hard-linked file: {relative}"
                    )
                entry = {
                    "mode": mode,
                    "path": relative,
                    "sha256": _sha256(path),
                    "size": metadata.st_size,
                    "type": "file",
                }
            elif stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(path)
                parsed = PurePosixPath(target)
                if not target or parsed.is_absolute() or ".." in parsed.parts:
                    raise ImageHelperError(
                        f"inventory contains an escaping symlink: {relative}"
                    )
                entry = {
                    "mode": mode,
                    "path": relative,
                    "target": target,
                    "type": "symlink",
                }
                if name in directories:
                    directories.remove(name)
            else:
                raise ImageHelperError(
                    f"inventory contains a special file: {relative}"
                )
            entries.append(entry)
    entries.sort(key=lambda entry: str(entry["path"]))
    return entries


def tree_inventory(root: Path, *, recorded_root: str) -> dict[str, Any]:
    entries = _tree_entries(root)
    identity = {
        "entries": entries,
        "hash_scheme": TREE_HASH_SCHEME,
    }
    return {
        **identity,
        "root": recorded_root,
        "schema_version": NATIVE_INVENTORY_SCHEMA,
        "tree_sha256": _json_sha256(identity),
    }


def harden_tree(root: Path) -> None:
    """Make every directory read/execute-only and retain executable file modes."""
    entries = _tree_entries(root)
    for entry in entries:
        path = root / str(entry["path"])
        if entry["type"] == "file":
            path.chmod(0o555 if int(entry["mode"]) & 0o111 else 0o444)
    for entry in reversed(entries):
        if entry["type"] == "directory":
            (root / str(entry["path"])).chmod(0o555)
    root.chmod(0o555)


def _debian_installed_paths(packages: list[dict[str, Any]]) -> list[Path]:
    paths: set[Path] = set()
    for package in packages:
        name = str(package["package"])
        query = subprocess.run(
            ("dpkg-query", "-L", name),
            check=False,
            capture_output=True,
            text=True,
        )
        if query.returncode != 0:
            raise ImageHelperError(f"dpkg cannot list installed package {name}")
        for line in query.stdout.splitlines():
            path = Path(line)
            if path.is_absolute() and path != Path("/"):
                paths.add(path)
    return sorted(paths, key=lambda path: path.as_posix())


def debian_inventory(manifest_path: Path) -> dict[str, Any]:
    """Inventory every file dpkg attributes to the audited Debian bundle.

    Two dpkg checks feed the inventory.  ``dpkg-query -W`` must report each
    manifest package as exactly ``install ok installed`` at the pinned
    version/architecture -- any other status (wrong version, half-configured,
    absent) fails the build.  ``dpkg-query -L`` then enumerates the paths
    those packages own; every path present on disk becomes a canonical
    directory/file/symlink entry (files carry sha256 and size) and special
    files fail closed.  Paths dpkg lists that are absent on disk are skipped:
    the inventory records on-disk state only.  Entries are hashed into
    ``tree_sha256`` under the scheme shared with the runtime verifier.
    """
    manifest = _strict_json(manifest_path)
    if manifest.get("schema_version") != DEBIAN_SCHEMA:
        raise ImageHelperError("Debian bundle schema is unsupported")
    packages_value = manifest.get("packages")
    if type(packages_value) is not list or not packages_value:
        raise ImageHelperError("Debian bundle package list is empty")
    packages = [item for item in packages_value if type(item) is dict]
    if len(packages) != len(packages_value):
        raise ImageHelperError("Debian bundle contains a non-object package")
    entries: list[dict[str, Any]] = []
    for package in packages:
        name = str(package["package"])
        query = subprocess.run(
            (
                "dpkg-query",
                "-W",
                "-f=${Package}\\t${Version}\\t${Architecture}\\t${Status}",
                name,
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        expected = (
            f"{name}\t{package['version']}\t{package['architecture']}"
            "\tinstall ok installed"
        )
        if query.returncode != 0 or query.stdout != expected:
            raise ImageHelperError(f"installed Debian identity differs for {name}")
    for path in _debian_installed_paths(packages):
        if not path.exists() and not path.is_symlink():
            continue
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        relative = path.as_posix().removeprefix("/")
        if stat.S_ISDIR(metadata.st_mode):
            entry: dict[str, Any] = {
                "mode": mode,
                "path": relative,
                "type": "directory",
            }
        elif stat.S_ISREG(metadata.st_mode):
            entry = {
                "mode": mode,
                "path": relative,
                "sha256": _sha256(path),
                "size": metadata.st_size,
                "type": "file",
            }
        elif stat.S_ISLNK(metadata.st_mode):
            entry = {
                "mode": mode,
                "path": relative,
                "target": os.readlink(path),
                "type": "symlink",
            }
        else:
            raise ImageHelperError(
                f"Debian installed-file inventory contains special path {path}"
            )
        entries.append(entry)
    entries.sort(key=lambda entry: str(entry["path"]))
    identity = {
        "entries": entries,
        "hash_scheme": TREE_HASH_SCHEME,
    }
    return {
        **identity,
        "schema_version": "alberta.official_foragax.debian_inventory.v1",
        "tree_sha256": _json_sha256(identity),
    }


def verify_debian(manifest_path: Path, *, allow_unbound: bool = False) -> dict[str, Any]:
    """Check installed apt state against the audited Debian bundle manifest.

    Fails when the recomputed installed-file inventory hash differs from the
    manifest's ``installed_file_inventory_sha256``, or when the interpreter
    at ``python_executable`` does not hash to ``python_executable_sha256``.
    ``allow_unbound`` skips only the inventory-hash comparison; it exists for
    authoring a new manifest (the inventory JSON is printed so its hash can
    be captured) and is never passed by the in-image build step.
    """
    manifest = _strict_json(manifest_path)
    inventory = debian_inventory(manifest_path)
    expected_inventory = manifest.get("installed_file_inventory_sha256")
    if (
        not allow_unbound
        and (
            type(expected_inventory) is not str
            or inventory["tree_sha256"] != expected_inventory
        )
    ):
        raise ImageHelperError("Debian installed-file inventory differs")
    python = Path(str(manifest.get("python_executable")))
    expected_python_sha = manifest.get("python_executable_sha256")
    if (
        not python.is_absolute()
        or not python.is_file()
        or type(expected_python_sha) is not str
        or _sha256(python) != expected_python_sha
    ):
        raise ImageHelperError("Debian Python executable identity differs")
    return inventory


def _runtime_probe(python: Path) -> dict[str, Any]:
    """Interrogate the built runtime venv from an isolated interpreter.

    The probe runs under ``-I -B`` with a pinned minimal environment and
    gates the image on four hermeticity checks:

    - no installed distribution's ``direct_url.json`` references a build
      path, so build-host locations cannot leak into runtime metadata;
    - the upstream ``continual-foragax-agents`` project itself is not
      installed -- it must execute from the read-only source mount, never
      from site-packages;
    - the only ``.pth`` file is the canonical ``_virtualenv.pth``, ruling
      out startup-time import injection;
    - the bundled imageio-ffmpeg binary is a regular, non-hardlinked file.

    Returns the SPDX-shaped package list (with per-distribution RECORD
    hashes), the ffmpeg identity, and the interpreter identity consumed by
    :func:`finalize_image`.
    """
    script = r"""
import hashlib, importlib.metadata, json, os, re, stat, sys
from pathlib import Path
root = Path(sys.prefix).resolve()
packages = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    if not name:
        continue
    direct_url = distribution.read_text("direct_url.json")
    if direct_url and ("file:" in direct_url or "/input/" in direct_url or "/build/" in direct_url):
        raise RuntimeError(f"distribution {name} exposes a build path")
    record = distribution.read_text("RECORD")
    packages.append({
        "SPDXID": "SPDXRef-Package-" + re.sub(r"[^A-Za-z0-9.-]", "-", name),
        "checksums": [] if record is None else [{
            "algorithm": "SHA256",
            "checksumValue": hashlib.sha256(record.encode()).hexdigest(),
        }],
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False,
        "licenseConcluded": "NOASSERTION",
        "name": name,
        "versionInfo": distribution.version,
    })
packages.sort(key=lambda item: (item["name"].casefold(), item["versionInfo"]))
if any(
    re.sub(r"[-_.]+", "-", item["name"]).casefold()
    == "continual-foragax-agents"
    for item in packages
):
    raise RuntimeError("upstream project was installed into the runtime")
pth = sorted(root.rglob("*.pth"))
if (
    len(pth) != 1
    or pth[0].name != "_virtualenv.pth"
    or pth[0].read_bytes() != b"import _virtualenv"
):
    raise RuntimeError(
        "runtime contains noncanonical .pth import injection: "
        + repr([path.as_posix() for path in pth])
    )
import imageio_ffmpeg
ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
metadata = ffmpeg.lstat()
if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
    raise RuntimeError("imageio-ffmpeg executable is not a regular file")
payload = {
    "ffmpeg": {
        "path": ffmpeg.resolve().as_posix(),
        "sha256": hashlib.sha256(ffmpeg.read_bytes()).hexdigest(),
    },
    "packages": packages,
    "python_executable": Path(sys.executable).as_posix(),
    "python_executable_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
    "python_version": ".".join(str(value) for value in sys.version_info[:3]),
}
print(json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True))
"""
    completed = subprocess.run(
        (str(python), "-I", "-B", "-c", script),
        check=False,
        capture_output=True,
        text=True,
        env={
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
        },
    )
    if completed.returncode != 0 or completed.stderr:
        raise ImageHelperError(
            "runtime package probe failed: " + completed.stderr.strip()
        )
    def _probe_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if type(key) is not str:
                raise ImageHelperError("runtime package probe repeats JSON key")
            if key in result:
                raise ImageHelperError("runtime package probe repeats JSON key")
            result[key] = item
        return result

    def _probe_parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ImageHelperError("runtime package probe contains non-finite JSON number")
        return parsed

    try:
        value = json.loads(
            completed.stdout,
            object_pairs_hook=_probe_pairs,
            parse_constant=lambda v: (_ for _ in ()).throw(
                ImageHelperError("runtime package probe contains JSON constant")
            ),
            parse_float=_probe_parse_float,
        )
    except (ImageHelperError, json.JSONDecodeError) as exc:
        if isinstance(exc, ImageHelperError):
            raise
        raise ImageHelperError("runtime package probe returned invalid JSON") from exc
    if type(value) is not dict:
        raise ImageHelperError("runtime package probe returned a non-object")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def finalize_image(
    *,
    source_root: Path,
    runtime_root: Path,
    launcher: Path,
    build_attestation: Path,
    attestation_root: Path,
) -> None:
    """Harden the sealed layout and write the image's attestation set.

    Rejects any deviation from the sealed roots, leftover Git metadata, and
    cache outputs (``__pycache__``/``.pyc``/``.cache``), then makes the
    source and runtime trees read-only and probes the hardened runtime
    (:func:`_runtime_probe`).  Emits four read-only artifacts under
    ``/opt/alberta-attestations``:

    - ``sbom.spdx.json`` -- SPDX 2.3 document over the probed packages, with
      a fixed epoch timestamp and a namespace derived from the dependency
      lock hash so identical builds produce identical bytes;
    - ``native-inventory.json`` -- canonical tree inventory of the runtime
      root, with ``tree_sha256`` under the shared hashing scheme;
    - ``runtime-metadata.json`` -- interpreter and bundled-ffmpeg identities;
    - ``build-attestation.json`` -- byte copy of the host build attestation.
    """
    if source_root != SOURCE_ROOT or runtime_root != RUNTIME_ROOT:
        raise ImageHelperError("final image roots differ from the sealed layout")
    if attestation_root != ATTESTATION_ROOT:
        raise ImageHelperError("attestation root differs from the sealed layout")
    if not launcher.is_file() or not build_attestation.is_file():
        raise ImageHelperError("launcher or build attestation is missing")
    if any(source_root.rglob(".git")):
        raise ImageHelperError("source tree unexpectedly contains Git metadata")
    for root in (source_root, runtime_root):
        for path in root.rglob("*"):
            if path.name in {"__pycache__", ".cache"} or path.suffix in {
                ".pyc",
                ".pyo",
            }:
                raise ImageHelperError(f"runtime contains cache output: {path}")
    launcher.chmod(0o555)
    harden_tree(source_root)
    harden_tree(runtime_root)
    probe = _runtime_probe(runtime_root / "bin/python")
    ffmpeg_path = Path(str(probe["ffmpeg"]["path"]))
    try:
        ffmpeg_path.relative_to(runtime_root)
    except ValueError as exc:
        raise ImageHelperError("imageio-ffmpeg executable escapes the runtime") from exc
    ffmpeg_metadata = ffmpeg_path.stat()
    if stat.S_IMODE(ffmpeg_metadata.st_mode) != 0o555:
        raise ImageHelperError("imageio-ffmpeg executable mode was not preserved")

    build = _strict_json(build_attestation)
    lock_sha = str(build.get("dependency_lock_sha256"))
    if _SHA256.fullmatch(lock_sha) is None:
        raise ImageHelperError("build attestation lock identity is invalid")
    namespace = f"https://elizaos.github.io/alberta/foragax/sbom/{lock_sha}"
    packages = probe["packages"]
    sbom = {
        "SPDXID": "SPDXRef-DOCUMENT",
        "creationInfo": {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Tool: alberta-official-foragax-oci-v4"],
        },
        "dataLicense": "CC0-1.0",
        "documentNamespace": namespace,
        "name": "alberta-official-foragax-runtime",
        "packages": packages,
        "relationships": [
            {
                "relatedSpdxElement": package["SPDXID"],
                "relationshipType": "DESCRIBES",
                "spdxElementId": "SPDXRef-DOCUMENT",
            }
            for package in packages
        ],
        "spdxVersion": "SPDX-2.3",
    }
    runtime_metadata = {
        "bundled_executables": {"imageio-ffmpeg": probe["ffmpeg"]},
        "python_executable": probe["python_executable"],
        "python_executable_sha256": probe["python_executable_sha256"],
        "python_version": probe["python_version"],
        "schema_version": RUNTIME_METADATA_SCHEMA,
    }
    inventory = tree_inventory(runtime_root, recorded_root=runtime_root.as_posix())
    _write(attestation_root / "sbom.spdx.json", sbom)
    _write(attestation_root / "native-inventory.json", inventory)
    _write(attestation_root / "runtime-metadata.json", runtime_metadata)
    (attestation_root / "build-attestation.json").write_bytes(
        build_attestation.read_bytes()
    )
    for path in attestation_root.iterdir():
        path.chmod(0o444)
    attestation_root.chmod(0o555)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract-source", allow_abbrev=False)
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--destination", type=Path, required=True)
    extract.add_argument("--commit", required=True)
    extract.add_argument("--lock-sha256", required=True)
    cache = subparsers.add_parser("extract-cache", allow_abbrev=False)
    cache.add_argument("--archive", type=Path, required=True)
    cache.add_argument("--destination", type=Path, required=True)
    debian = subparsers.add_parser("verify-debian", allow_abbrev=False)
    debian.add_argument("--manifest", type=Path, required=True)
    debian.add_argument("--allow-unbound", action="store_true")
    finalize = subparsers.add_parser("finalize", allow_abbrev=False)
    finalize.add_argument("--source-root", type=Path, required=True)
    finalize.add_argument("--runtime-root", type=Path, required=True)
    finalize.add_argument("--launcher", type=Path, required=True)
    finalize.add_argument("--build-attestation", type=Path, required=True)
    finalize.add_argument("--attestation-root", type=Path, required=True)
    return parser


def main() -> int:
    try:
        args = _parser().parse_args()
        if args.command == "extract-source":
            extract_source(
                args.archive,
                args.destination,
                commit=args.commit,
                lock_sha256=args.lock_sha256,
            )
        elif args.command == "extract-cache":
            extract_regular_tree_archive(args.archive, args.destination)
        elif args.command == "verify-debian":
            inventory = verify_debian(
                args.manifest,
                allow_unbound=args.allow_unbound,
            )
            sys.stdout.buffer.write(_json_bytes(inventory))
        elif args.command == "finalize":
            finalize_image(
                source_root=args.source_root,
                runtime_root=args.runtime_root,
                launcher=args.launcher,
                build_attestation=args.build_attestation,
                attestation_root=args.attestation_root,
            )
        else:  # pragma: no cover - argparse owns the choices
            raise ImageHelperError("unsupported image helper command")
        return 0
    except (ImageHelperError, OSError, subprocess.SubprocessError) as exc:
        sys.stderr.write(f"official Foragax image helper: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
