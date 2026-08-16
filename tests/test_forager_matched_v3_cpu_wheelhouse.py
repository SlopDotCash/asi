"""Focused synthetic tests for the matched-v3 CPU wheelhouse boundary."""

from __future__ import annotations

import base64
import copy
import dataclasses
import fcntl
import hashlib
import json
import os
import pickle
import stat
import struct
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matched_v3_cpu_wheelhouse as wheelhouse

pytestmark = pytest.mark.unit


def _canonical(value: Any, *, newline: bool = True) -> bytes:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return raw + (b"\n" if newline else b"")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


_FAKE_PACKAGING_FILES = {
    "packaging/__init__.py": b'__version__ = "1.test"\n',
    "packaging/version.py": b"""
import re
class Version:
    def __init__(self, value):
        if not re.fullmatch(r"[0-9]+(?:\\.[0-9]+)*(?:[A-Za-z0-9_.+-]*)", value):
            raise ValueError(value)
        self.raw = value
        release = re.match(r"[0-9]+(?:\\.[0-9]+)*", value).group()
        self.parts = tuple(int(x) for x in release.split("."))
    def __str__(self): return self.raw
    def __hash__(self): return hash(self.parts)
    def __eq__(self, other): return isinstance(other, Version) and self.parts == other.parts
    def __lt__(self, other): return self.parts < other.parts
    def __le__(self, other): return self.parts <= other.parts
    def __gt__(self, other): return self.parts > other.parts
    def __ge__(self, other): return self.parts >= other.parts
""",
    "packaging/specifiers.py": b"""
from .version import Version
class SpecifierSet:
    def __init__(self, raw=""):
        self.raw = raw.strip()
        self.items = []
        for token in filter(None, (x.strip() for x in self.raw.split(","))):
            for operator in ("==", "!=", ">=", "<=", ">", "<"):
                if token.startswith(operator):
                    self.items.append((operator, Version(token[len(operator):])))
                    break
            else: raise ValueError(token)
    def __str__(self): return self.raw
    def __contains__(self, value):
        value = value if isinstance(value, Version) else Version(str(value))
        operations = {"==": lambda a,b:a==b, "!=":lambda a,b:a!=b, ">=":lambda a,b:a>=b,
                      "<=":lambda a,b:a<=b, ">":lambda a,b:a>b, "<":lambda a,b:a<b}
        return all(operations[op](value, expected) for op, expected in self.items)
""",
    "packaging/tags.py": b"""
class Tag:
    def __init__(self, interpreter, abi, platform):
        self.interpreter, self.abi, self.platform = interpreter, abi, platform
    def __str__(self): return f"{self.interpreter}-{self.abi}-{self.platform}"
    def __hash__(self): return hash(str(self))
    def __eq__(self, other): return isinstance(other, Tag) and str(self) == str(other)
def parse_tag(raw):
    interpreter, abi, platform = raw.split("-")
    return {
        Tag(i, a, p)
        for i in interpreter.split(".")
        for a in abi.split(".")
        for p in platform.split(".")
    }
""",
    "packaging/markers.py": b"""
import re
class Marker:
    def __init__(self, raw): self.raw = raw.strip()
    def __str__(self): return self.raw
    def evaluate(self, environment=None, context=None):
        del context
        environment = environment or {}
        outcomes = []
        for expression in self.raw.split(" and "):
            pattern = r'([A-Za-z_]+)\\s*(==|!=|>=|<=|>|<)\\s*["\\\']([^"\\\']+)["\\\']'
            match = re.fullmatch(pattern, expression.strip())
            if not match: raise ValueError(expression)
            left, operator, right = match.groups()
            observed = environment[left]
            operations = {"==":lambda a,b:a==b, "!=":lambda a,b:a!=b, ">=":lambda a,b:a>=b,
                          "<=":lambda a,b:a<=b, ">":lambda a,b:a>b, "<":lambda a,b:a<b}
            outcomes.append(operations[operator](observed, right))
        return all(outcomes)
""",
    "packaging/requirements.py": b"""
import re
from .markers import Marker
from .specifiers import SpecifierSet
class Requirement:
    def __init__(self, raw):
        requirement, separator, marker = raw.partition(";")
        requirement = requirement.strip()
        direct, at, url = requirement.partition(" @ ")
        self.url = url.strip() if at else None
        token = direct.strip() if at else requirement
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\\[([^]]+)\\])?\\s*(.*)", token)
        if not match: raise ValueError(raw)
        self.name, extras, specifier = match.groups()
        self.extras = set() if not extras else {x.strip() for x in extras.split(",")}
        self.specifier = SpecifierSet(specifier.strip())
        self.marker = Marker(marker.strip()) if separator else None
""",
    "packaging/utils.py": b"""
import re
from .tags import parse_tag
from .version import Version
def canonicalize_name(value): return re.sub(r"[-_.]+", "-", value).lower()
def parse_wheel_filename(filename):
    if not filename.endswith(".whl"): raise ValueError(filename)
    parts = filename[:-4].split("-")
    if len(parts) != 5: raise ValueError(filename)
    name, version, python, abi, platform = parts
    return canonicalize_name(name), Version(version), (), parse_tag(f"{python}-{abi}-{platform}")
""",
}


def _write_packaging_tool(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, raw in sorted(_FAKE_PACKAGING_FILES.items()):
            archive.writestr(name, raw)


def _record_hash(raw: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
    return f"sha256={encoded}"


def _wheel_bytes(
    name: str,
    version: str,
    *,
    requires_dist: tuple[str, ...] = (),
    provides_extra: tuple[str, ...] = (),
    dynamic: tuple[str, ...] = (),
    tag: str = "py3-none-any",
    root_is_purelib: bool = True,
    corrupt_record: bool = False,
    extra_member: tuple[str, bytes] | None = None,
) -> tuple[str, bytes]:
    escaped = name.replace("-", "_")
    dist_info = f"{escaped}-{version}.dist-info"
    metadata_lines = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
        "Requires-Python: >=3.12,<3.13",
    ]
    metadata_lines.extend(f"Provides-Extra: {item}" for item in provides_extra)
    metadata_lines.extend(f"Requires-Dist: {item}" for item in requires_dist)
    metadata_lines.extend(f"Dynamic: {item}" for item in dynamic)
    metadata = ("\n".join(metadata_lines) + "\n\nsynthetic\n").encode()
    wheel_metadata = (
        "Wheel-Version: 1.0\n"
        "Generator: synthetic-test\n"
        f"Root-Is-Purelib: {str(root_is_purelib).lower()}\n"
        f"Tag: {tag}\n\n"
    ).encode()
    files = {
        f"{escaped}/__init__.py": b"VALUE = 1\n",
        f"{dist_info}/METADATA": metadata,
        f"{dist_info}/WHEEL": wheel_metadata,
    }
    if extra_member is not None:
        files[extra_member[0]] = extra_member[1]
    record_path = f"{dist_info}/RECORD"
    rows = []
    for path, raw in sorted(files.items()):
        digest = _record_hash(raw)
        if corrupt_record and path.endswith("/__init__.py"):
            digest = "sha256=" + "A" * 43
        rows.append(f"{path},{digest},{len(raw)}")
    rows.append(f"{record_path},,")
    files[record_path] = ("\n".join(rows) + "\n").encode()
    stream = __import__("io").BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.comment = b""
        for path, raw in sorted(files.items()):
            info = zipfile.ZipInfo(path, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    return f"{escaped}-{version}-{tag}.whl", stream.getvalue()


def _with_unindexed_zip_gap(raw: bytes) -> bytes:
    eocd = raw.rfind(b"PK\x05\x06")
    assert eocd >= 0
    central_offset = struct.unpack_from("<L", raw, eocd + 16)[0]
    gap = b"UNINDEXED"
    changed = bytearray(raw[:central_offset] + gap + raw[central_offset:])
    struct.pack_into("<L", changed, eocd + len(gap) + 16, central_offset + len(gap))
    return bytes(changed)


def _marker_environment() -> dict[str, str]:
    return {
        "implementation_name": "cpython",
        "implementation_version": "3.12.3",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "platform_release": "synthetic",
        "platform_system": "Linux",
        "platform_version": "synthetic",
        "python_full_version": "3.12.3",
        "python_version": "3.12",
        "sys_platform": "linux",
    }


def _manifest_bytes(wheels: dict[str, bytes], roots: list[str]) -> tuple[bytes, str]:
    value: dict[str, Any] = {
        "capture": {
            "network_used": True,
            "resolver_argv": ["synthetic-resolver", "--wheel-only"],
            "resolver_binary_sha256": "1" * 64,
            "resolver_name": "synthetic-resolver",
            "resolver_version": "1",
        },
        "claims": copy.deepcopy(wheelhouse.cpu_wheelhouse_contract_descriptor()["claims"]),
        "classification": "networked_solver_output_non_authorizing",
        "root_requirements": sorted(roots),
        "schema_version": wheelhouse.CPU_WHEEL_CAPTURE_MANIFEST_SCHEMA_VERSION,
        "status": "untrusted_network_capture_candidate_only",
        "target": {
            "abi": "cp312",
            "compatible_tags": ["py3-none-any"],
            "implementation": "CPython",
            "libc": {"family": "glibc", "version": "2.39"},
            "marker_environment": _marker_environment(),
            "oci_platform": "linux/amd64",
            "platform": "linux_x86_64",
            "python_version": "3.12.3",
        },
        "wheels": [
            {
                "filename": name,
                "origin_url": f"https://example.invalid/files/{name}",
                "sha256": _sha(raw),
                "size_bytes": len(raw),
            }
            for name, raw in sorted(wheels.items())
        ],
    }
    value["manifest_body_sha256"] = _sha(_canonical(value, newline=False))
    raw = _canonical(value)
    return raw, _sha(raw)


def _base_wheels() -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for name, version in (
        ("continual-foragax", "0.55.0"),
        ("jax", "0.11.0"),
        ("jaxlib", "0.11.0"),
    ):
        filename, raw = _wheel_bytes(name, version)
        result[filename] = raw
    return result


def _write_wheels(directory: Path, wheels: dict[str, bytes]) -> None:
    directory.mkdir()
    for name, raw in wheels.items():
        (directory / name).write_bytes(raw)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_fd_snapshot() -> dict[int, tuple[int, int, int]]:
    """Return stable identities for descriptors open around one synchronous call."""

    result: dict[int, tuple[int, int, int]] = {}
    for raw_descriptor in os.listdir("/proc/self/fd"):
        descriptor = int(raw_descriptor)
        try:
            metadata = os.fstat(descriptor)
        except OSError:
            continue
        result[descriptor] = (metadata.st_dev, metadata.st_ino, metadata.st_mode)
    return result


def _binding(tmp_path: Path) -> wheelhouse.WheelhouseVerifierToolBinding:
    tool = tmp_path / "packaging-tool.whl"
    _write_packaging_tool(tool)
    python = Path(sys.executable).resolve()
    completed = subprocess.run(
        (str(python), "--version"),
        check=False,
        capture_output=True,
        env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 0
    return wheelhouse.WheelhouseVerifierToolBinding(
        python_executable=python,
        python_executable_sha256=_file_sha(python),
        python_version_output_sha256=_sha(completed.stdout + completed.stderr),
        packaging_wheel=tool,
        packaging_wheel_sha256=_file_sha(tool),
        packaging_version="1.test",
    )


@pytest.fixture
def frozen_current_hashes() -> None:
    helper = Path(wheelhouse.__file__).with_name("_forager_matched_v3_cpu_wheelhouse_helper.py")
    helper_sha = _file_sha(helper)
    assert helper_sha == wheelhouse.CPU_WHEELHOUSE_HELPER_SOURCE_SHA256
    descriptor_raw = wheelhouse.canonical_cpu_wheelhouse_contract_descriptor_bytes()
    assert _sha(descriptor_raw) == wheelhouse.CPU_WHEELHOUSE_CONTRACT_DESCRIPTOR_SHA256


def _reseal_receipt(receipt: dict[str, Any], *, rebuild_report: bool = True) -> bytes:
    if rebuild_report:
        closure = receipt["closure"]
        graph = {
            "activated_extras": closure["activated_extras"],
            "edges": closure["edges"],
            "root_requirements": closure["root_requirements"],
        }
        closure["dependency_graph_sha256"] = _sha(_canonical(graph, newline=False))
        packages = receipt["packages"]
        inventory = [
            {
                "filename": package["filename"],
                "name": package["name"],
                "sha256": package["sha256"],
                "size_bytes": package["size_bytes"],
                "version": package["version"],
            }
            for package in packages
        ]
        report = {
            "capture_manifest_body_sha256": receipt["capture_manifest"]["body_sha256"],
            "claims": copy.deepcopy(receipt["claims"]),
            "classification": "disconnected_wheel_bytes_validation_non_authorizing",
            "closure": closure,
            "inventory_sha256": _sha(_canonical(inventory, newline=False)),
            "package_count": len(packages),
            "packages": packages,
            "packaging_tool": {
                "sha256": receipt["verifier"]["packaging_tool_wheel_sha256"],
                "version": receipt["verifier"]["packaging_tool_version"],
            },
            "schema_version": wheelhouse.CPU_WHEEL_VALIDATION_REPORT_SCHEMA_VERSION,
            "status": "content_verified_unqualified_non_authorizing",
            "total_uncompressed_bytes": sum(
                package["uncompressed_size_bytes"] for package in packages
            ),
            "total_wheel_bytes": sum(package["size_bytes"] for package in packages),
            "zip_member_count": sum(package["zip_member_count"] for package in packages),
        }
        report_body_sha = _sha(_canonical(report, newline=False))
        report["report_body_sha256"] = report_body_sha
        receipt["validation_report"]["body_sha256"] = report_body_sha
        receipt["validation_report"]["full_file_sha256"] = _sha(_canonical(report))
    body = copy.deepcopy(receipt)
    body.pop("receipt_body_sha256", None)
    receipt["receipt_body_sha256"] = _sha(_canonical(body, newline=False))
    return _canonical(receipt)


def _parse_resealed_receipt(receipt: dict[str, Any], *, rebuild_report: bool = True) -> None:
    raw = _reseal_receipt(receipt, rebuild_report=rebuild_report)
    wheelhouse.parse_cpu_wheelhouse_receipt(raw, expected_file_sha256=_sha(raw))


def _stage(
    tmp_path: Path,
    *,
    wheels: dict[str, bytes] | None = None,
    roots: list[str] | None = None,
) -> wheelhouse.RetainedMatchedV3CpuWheelhouse:
    selected = _base_wheels() if wheels is None else wheels
    requirements = (
        ["continual-foragax==0.55.0", "jax==0.11.0", "jaxlib==0.11.0"] if roots is None else roots
    )
    candidate = tmp_path / "candidate"
    _write_wheels(candidate, selected)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    manifest, digest = _manifest_bytes(selected, requirements)
    return wheelhouse.stage_matched_v3_cpu_wheelhouse(
        candidate_directory=candidate,
        capture_manifest_raw=manifest,
        expected_capture_manifest_sha256=digest,
        verifier=_binding(tmp_path),
        scratch_directory=scratch,
    )


def test_descriptor_is_canonical_nonauthorizing_and_has_no_large_artifact_path(
    frozen_current_hashes: None,
) -> None:
    raw = wheelhouse.canonical_cpu_wheelhouse_contract_descriptor_bytes()
    descriptor = wheelhouse.cpu_wheelhouse_contract_descriptor()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert _sha(raw) == wheelhouse.cpu_wheelhouse_contract_descriptor_sha256()
    assert descriptor["publication"]["default_root"] is False
    assert descriptor["phase_boundary"]["network_capture_implemented_here"] is False
    assert descriptor["helper"]["ambient_packaging_allowed"] is False
    assert descriptor["helper"]["packaging_tool_code_executed"] is True
    assert descriptor["helper"]["packaging_tool_trust_and_authentication_external"] is True
    assert descriptor["helper"]["packaging_tool_side_effect_isolation_claimed"] is False
    assert descriptor["helper"]["source_sealed_snapshot_inherited"] is True
    assert descriptor["helper"]["python_executable_sealed_snapshot_inherited"] is True
    assert descriptor["helper"]["python_runtime_dependency_bytes_bound"] is False
    assert descriptor["helper"]["packaging_tool_sealed_snapshot_inherited"] is True
    assert (
        descriptor["helper"]["record_entries_sha256_scheme"]
        == "canonical-json-sorted-verified-regular-member-path-size-sha256-v1"
    )
    assert descriptor["archive"]["members"] == (
        "root_level_sha256_wheel_names_with_original_filename_mapping"
    )
    assert (
        descriptor["phase_boundary"]["candidate_wheels_imported_or_executed_by_first_party_logic"]
        is False
    )
    assert descriptor["phase_boundary"]["supplied_tool_candidate_side_effects_attested"] is False
    assert all(value is False for value in descriptor["claims"].values())
    assert b"outputs/" not in raw
    detached = wheelhouse.cpu_wheelhouse_contract_descriptor()
    detached["claims"]["runtime_qualified"] = True
    assert wheelhouse.cpu_wheelhouse_contract_descriptor()["claims"]["runtime_qualified"] is False


def test_capture_manifest_parser_requires_canonical_independently_hashed_bytes() -> None:
    raw, digest = _manifest_bytes(_base_wheels(), ["jax==0.11.0"])
    parsed = wheelhouse.parse_cpu_wheel_capture_manifest(
        raw,
        expected_file_sha256=digest,
    )
    assert parsed["status"] == "untrusted_network_capture_candidate_only"
    assert all(value is False for value in parsed["claims"].values())
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="full-file"):
        wheelhouse.parse_cpu_wheel_capture_manifest(raw, expected_file_sha256="0" * 64)
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="canonical"):
        wheelhouse.parse_cpu_wheel_capture_manifest(
            raw.rstrip(b"\n"),
            expected_file_sha256=_sha(raw.rstrip(b"\n")),
        )


def test_capture_manifest_rejects_contradictory_implementation_version() -> None:
    raw, _digest = _manifest_bytes(_base_wheels(), ["jax==0.11.0"])
    manifest = json.loads(raw)
    manifest["target"]["marker_environment"]["implementation_version"] = "9.9.9"
    del manifest["manifest_body_sha256"]
    manifest["manifest_body_sha256"] = _sha(_canonical(manifest, newline=False))
    changed = _canonical(manifest)
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="contradicts"):
        wheelhouse.parse_cpu_wheel_capture_manifest(
            changed,
            expected_file_sha256=_sha(changed),
        )


def test_capture_manifest_rejects_nonfrozen_python_patch_version() -> None:
    raw, _digest = _manifest_bytes(_base_wheels(), ["jax==0.11.0"])
    manifest = json.loads(raw)
    manifest["target"]["python_version"] = "3.12.4"
    manifest["target"]["marker_environment"]["python_full_version"] = "3.12.4"
    manifest["target"]["marker_environment"]["implementation_version"] = "3.12.4"
    del manifest["manifest_body_sha256"]
    manifest["manifest_body_sha256"] = _sha(_canonical(manifest, newline=False))
    changed = _canonical(manifest)
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="exact CPython"):
        wheelhouse.parse_cpu_wheel_capture_manifest(
            changed,
            expected_file_sha256=_sha(changed),
        )


def test_stage_verifies_synthetic_wheels_and_retains_canonical_ustar(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    retained = _stage(tmp_path)
    try:
        receipt = retained.reverify()
        assert receipt["status"] == "content_verified_unqualified_non_authorizing"
        assert receipt["archive"]["member_count"] == 3
        assert receipt["closure"]["reachable_distributions"] == [
            "continual-foragax",
            "jax",
            "jaxlib",
        ]
        assert all(value is False for value in receipt["claims"].values())
        assert retained.archive_size_bytes % 10_240 == 0
        with tarfile.open(retained.proc_fd_path, mode="r:") as archive:
            assert [member.name for member in archive] == [
                item["archive_name"] for item in receipt["archive"]["members"]
            ]
            assert all(member.mode == 0o444 for member in archive.getmembers())
    finally:
        retained.close()
    assert retained.closed


def test_receipt_packages_carry_runtime_lock_file_identities(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    retained = _stage(tmp_path)
    try:
        receipt = retained.reverify()
        package = next(item for item in receipt["packages"] if item["name"] == "jax")
        dist_info = package["dist_info_directory"]
        assert package["metadata"]["path"] == f"{dist_info}/METADATA"
        assert package["metadata"]["size_bytes"] > 0
        assert package["metadata"]["sha256"] == package["metadata_sha256"]
        assert package["wheel"] == {
            **package["wheel_metadata"],
            "path": f"{dist_info}/WHEEL",
            "size_bytes": package["wheel"]["size_bytes"],
            "sha256": package["wheel_metadata_sha256"],
        }
        assert package["wheel"]["generator"] == "synthetic-test"
        assert package["record"]["path"] == f"{dist_info}/RECORD"
        assert package["record"]["size_bytes"] > 0
        assert package["record"]["sha256"] == package["record_sha256"]
        assert package["record"]["entry_count"] == package["payload_file_count"]
        assert package["record"]["entries_sha256"] == package["payload_inventory_sha256"]
    finally:
        retained.close()


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("metadata_path", "METADATA file identity"),
        ("wheel_generator", "WHEEL record differs"),
        ("wheel_sha256", "WHEEL file identity"),
        ("record_entry_count", "RECORD file identity"),
        ("record_entries_sha256", "RECORD file identity"),
    ],
)
def test_receipt_replay_rejects_runtime_lock_file_identity_tampering(
    tmp_path: Path,
    frozen_current_hashes: None,
    case: str,
    match: str,
) -> None:
    retained = _stage(tmp_path)
    try:
        receipt = retained.receipt()
        package = receipt["packages"][0]
        if case == "metadata_path":
            package["metadata"]["path"] = "wrong.dist-info/METADATA"
        elif case == "wheel_generator":
            package["wheel_metadata"]["generator"] = "bad\nGenerator"
            package["wheel"]["generator"] = "bad\nGenerator"
        elif case == "wheel_sha256":
            package["wheel"]["sha256"] = "0" * 64
        elif case == "record_entry_count":
            package["record"]["entry_count"] += 1
        elif case == "record_entries_sha256":
            package["record"]["entries_sha256"] = "0" * 64
        else:  # pragma: no cover - parametrization is closed above
            raise AssertionError(case)
        with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match=match):
            _parse_resealed_receipt(receipt)
    finally:
        retained.close()


def test_long_wheel_filename_uses_digest_archive_member_name(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    long_name = "long" + "x" * 100
    wheels = _base_wheels()
    continual_filename, continual_raw = _wheel_bytes(
        "continual-foragax",
        "0.55.0",
        requires_dist=(f"{long_name}==1.0.0",),
    )
    wheels[continual_filename] = continual_raw
    long_filename, long_raw = _wheel_bytes(long_name, "1.0.0")
    assert 100 < len(long_filename.encode("ascii")) <= 255
    wheels[long_filename] = long_raw
    retained = _stage(tmp_path, wheels=wheels)
    try:
        receipt = retained.reverify()
        member = next(
            item for item in receipt["archive"]["members"] if item["filename"] == long_filename
        )
        assert member["archive_name"] == f"{_sha(long_raw)}.whl"
        assert len(member["archive_name"].encode("ascii")) <= 100
        with tarfile.open(retained.proc_fd_path, mode="r:") as archive:
            assert [item.name for item in archive] == [
                item["archive_name"] for item in receipt["archive"]["members"]
            ]
    finally:
        retained.close()


def test_retained_capability_is_not_copyable_picklable_or_usable_after_close(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    retained = _stage(tmp_path)
    with pytest.raises(TypeError, match="copied"):
        copy.copy(retained)
    with pytest.raises(TypeError, match="serialized"):
        pickle.dumps(retained)
    retained.close()
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="closed"):
        _ = retained.archive_sha256


def test_retained_capability_invalidates_on_descriptor_mode_drift(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    retained = _stage(tmp_path)
    path = retained.proc_fd_path
    Path(path).chmod(0o600)
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="drifted"):
        _ = retained.archive_sha256
    assert retained.closed


def test_stage_cleanup_close_error_does_not_leak_transferred_archive_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _open_fd_snapshot()
    original_close = os.close
    injected = False

    def close_then_fail_once(descriptor: int) -> None:
        nonlocal injected
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            target = ""
        original_close(descriptor)
        if not injected and "/scratch/.matched-v3-wheelhouse-" in target:
            injected = True
            raise OSError("synthetic staging descriptor close failure")

    monkeypatch.setattr(os, "close", close_then_fail_once)
    with pytest.raises(OSError, match="synthetic staging descriptor close failure"):
        _stage(tmp_path)
    assert injected
    assert _open_fd_snapshot() == before


def test_stage_cleanup_does_not_delete_a_substituted_temporary_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement: Path | None = None
    displaced: Path | None = None

    def substitute_then_fail(**_arguments: Any) -> tuple[dict[str, Any], bytes]:
        nonlocal replacement, displaced
        scratch = tmp_path / "scratch"
        staging = next(scratch.glob(".matched-v3-wheelhouse-*"))
        displaced = scratch / "displaced-private-staging"
        staging.rename(displaced)
        replacement = staging
        replacement.mkdir()
        (replacement / "do-not-delete.txt").write_text("replacement", encoding="utf-8")
        raise RuntimeError("synthetic helper failure after path substitution")

    monkeypatch.setattr(wheelhouse, "_invoke_helper", substitute_then_fail)
    with pytest.raises(RuntimeError, match="synthetic helper failure"):
        _stage(tmp_path)
    assert replacement is not None
    assert displaced is not None
    assert (replacement / "do-not-delete.txt").read_text(encoding="utf-8") == "replacement"
    assert displaced.is_dir()


def test_publication_is_content_addressed_new_only_and_replayable(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    retained = _stage(tmp_path)
    publication = tmp_path / "publication"
    publication.mkdir()
    try:
        with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="authorization"):
            wheelhouse.publish_matched_v3_cpu_wheelhouse(
                retained,
                publication,
                authorize_non_evidence_publication=False,
            )
        published = wheelhouse.publish_matched_v3_cpu_wheelhouse(
            retained,
            publication,
            authorize_non_evidence_publication=True,
        )
        assert published.directory == publication / "sha256" / retained.archive_sha256
        assert sorted(path.name for path in published.directory.iterdir()) == [
            "receipt.v1.json",
            "wheelhouse.v1.tar",
        ]
        replayed = wheelhouse.validate_published_matched_v3_cpu_wheelhouse(
            published.directory,
            expected_receipt_sha256=published.receipt_sha256,
            expected_archive_sha256=published.archive_sha256,
        )
        assert replayed["archive"]["sha256"] == retained.archive_sha256
        with pytest.raises(FileExistsError, match="overwrite"):
            wheelhouse.publish_matched_v3_cpu_wheelhouse(
                retained,
                publication,
                authorize_non_evidence_publication=True,
            )
    finally:
        retained.close()


def test_publication_cleanup_preserves_primary_error_and_closes_every_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = _stage(tmp_path)
    publication = tmp_path / "publication"
    publication.mkdir()
    before = _open_fd_snapshot()
    original_close = os.close
    injected = False

    def close_then_fail_once(descriptor: int) -> None:
        nonlocal injected
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            target = ""
        original_close(descriptor)
        if not injected and "/sha256/.staging-" in target:
            injected = True
            raise OSError("synthetic publication descriptor close failure")

    def fail_copy(*_arguments: Any, **_keywords: Any) -> None:
        raise RuntimeError("synthetic publication copy failure")

    monkeypatch.setattr(os, "close", close_then_fail_once)
    monkeypatch.setattr(wheelhouse, "_copy_retained_archive", fail_copy)
    try:
        with pytest.raises(RuntimeError, match="synthetic publication copy failure"):
            wheelhouse.publish_matched_v3_cpu_wheelhouse(
                retained,
                publication,
                authorize_non_evidence_publication=True,
            )
        assert injected
        assert _open_fd_snapshot() == before
        assert list((publication / "sha256").iterdir()) == []
    finally:
        retained.close()


def test_post_rename_validation_failure_rolls_back_exact_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = _stage(tmp_path)
    publication = tmp_path / "publication"
    publication.mkdir()
    original_rename = wheelhouse._rename_new_only
    original_validate = wheelhouse._validate_published_directory_fd
    committed = False

    def record_rename(root_fd: int, source: str, target: str) -> None:
        nonlocal committed
        original_rename(root_fd, source, target)
        committed = True

    def fail_only_after_rename(*arguments: Any, **keywords: Any) -> dict[str, Any]:
        if committed:
            raise RuntimeError("synthetic post-rename validation failure")
        return original_validate(*arguments, **keywords)

    monkeypatch.setattr(wheelhouse, "_rename_new_only", record_rename)
    monkeypatch.setattr(wheelhouse, "_validate_published_directory_fd", fail_only_after_rename)
    try:
        with pytest.raises(RuntimeError, match="synthetic post-rename validation failure"):
            wheelhouse.publish_matched_v3_cpu_wheelhouse(
                retained,
                publication,
                authorize_non_evidence_publication=True,
            )
        assert committed
        assert list((publication / "sha256").iterdir()) == []
        assert retained.reverify()["archive"]["sha256"] == retained.archive_sha256
    finally:
        retained.close()


def test_successful_publication_close_error_does_not_report_a_false_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = _stage(tmp_path)
    publication = tmp_path / "publication"
    publication.mkdir()
    before = _open_fd_snapshot()
    original_close = os.close
    injected = False

    def close_then_fail_once(descriptor: int) -> None:
        nonlocal injected
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            target = ""
        original_close(descriptor)
        if not injected and target.endswith(f"/sha256/{retained.archive_sha256}"):
            injected = True
            raise OSError("synthetic post-commit descriptor close failure")

    monkeypatch.setattr(os, "close", close_then_fail_once)
    try:
        published = wheelhouse.publish_matched_v3_cpu_wheelhouse(
            retained,
            publication,
            authorize_non_evidence_publication=True,
        )
        assert injected
        assert _open_fd_snapshot() == before
        replayed = wheelhouse.validate_published_matched_v3_cpu_wheelhouse(
            published.directory,
            expected_receipt_sha256=published.receipt_sha256,
            expected_archive_sha256=published.archive_sha256,
        )
        assert replayed["archive"]["sha256"] == retained.archive_sha256
    finally:
        retained.close()


def test_failed_rollback_cleanup_never_leaves_the_content_address_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = _stage(tmp_path)
    publication = tmp_path / "publication"
    publication.mkdir()
    original_validate = wheelhouse._validate_published_directory_fd
    original_remove = wheelhouse._remove_owned_directory_entry
    committed = False

    def validate_then_fail(*arguments: Any, **keywords: Any) -> dict[str, Any]:
        nonlocal committed
        original_validate(*arguments, **keywords)
        committed = True
        raise RuntimeError("synthetic validation failure before rollback")

    def fail_committed_cleanup(*arguments: Any, **keywords: Any) -> None:
        if keywords.get("label") == "committed wheelhouse publication":
            raise OSError("synthetic rollback-tree cleanup failure")
        original_remove(*arguments, **keywords)

    monkeypatch.setattr(wheelhouse, "_validate_published_directory_fd", validate_then_fail)
    monkeypatch.setattr(wheelhouse, "_remove_owned_directory_entry", fail_committed_cleanup)
    try:
        with pytest.raises(RuntimeError, match="synthetic validation failure"):
            wheelhouse.publish_matched_v3_cpu_wheelhouse(
                retained,
                publication,
                authorize_non_evidence_publication=True,
            )
        assert committed
        namespace_names = [path.name for path in (publication / "sha256").iterdir()]
        assert retained.archive_sha256 not in namespace_names
        assert len(namespace_names) == 1
        assert namespace_names[0].startswith(f".rollback-{retained.archive_sha256}-")
    finally:
        retained.close()


def test_namespace_substitution_after_commit_is_rejected_without_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = _stage(tmp_path)
    publication = tmp_path / "publication"
    publication.mkdir()
    original_validate = wheelhouse._validate_published_directory_fd
    substituted = False

    def validate_then_substitute(*arguments: Any, **keywords: Any) -> dict[str, Any]:
        nonlocal substituted
        receipt = original_validate(*arguments, **keywords)
        namespace = publication / "sha256"
        namespace.rename(publication / "displaced-sha256")
        namespace.mkdir()
        (namespace / "do-not-delete.txt").write_text("replacement", encoding="utf-8")
        substituted = True
        return receipt

    monkeypatch.setattr(wheelhouse, "_validate_published_directory_fd", validate_then_substitute)
    try:
        with pytest.raises(
            wheelhouse.ForagerMatchedV3CpuWheelhouseError,
            match="no longer names",
        ):
            wheelhouse.publish_matched_v3_cpu_wheelhouse(
                retained,
                publication,
                authorize_non_evidence_publication=True,
            )
        assert substituted
        assert list((publication / "displaced-sha256").iterdir()) == []
        assert (publication / "sha256" / "do-not-delete.txt").read_text(
            encoding="utf-8"
        ) == "replacement"
    finally:
        retained.close()


def test_stage_rejects_wrong_packaging_tool_binding(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    candidate = tmp_path / "candidate"
    _write_wheels(candidate, wheels)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    manifest, digest = _manifest_bytes(wheels, ["jax==0.11.0"])
    binding = _binding(tmp_path)
    bad = dataclasses.replace(binding, packaging_wheel_sha256="0" * 64)
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="tool wheel"):
        wheelhouse.stage_matched_v3_cpu_wheelhouse(
            candidate_directory=candidate,
            capture_manifest_raw=manifest,
            expected_capture_manifest_sha256=digest,
            verifier=bad,
            scratch_directory=scratch,
        )


def test_verifier_rejects_caller_self_consistent_nonfrozen_version_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    version_raw = b"Python 3.12.4\n"
    binding = dataclasses.replace(
        _binding(tmp_path),
        python_version_output_sha256=_sha(version_raw),
    )

    def substituted_version(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=version_raw,
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", substituted_version)
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="Python version"):
        wheelhouse._validate_verifier(binding)


def test_verifier_executes_from_sealed_immutable_code_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    helper_source = (
        Path(wheelhouse.__file__)
        .with_name("_forager_matched_v3_cpu_wheelhouse_helper.py")
        .read_bytes()
    )
    helper_path = tmp_path / "helper.py"
    helper_path.write_bytes(helper_source)
    binding = _binding(tmp_path)
    packaging_source = binding.packaging_wheel.read_bytes()
    descriptors = wheelhouse._validate_verifier(binding)
    staged = tmp_path / "staged"
    staged.mkdir()
    staged_descriptor = os.open(staged, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    required_seals = sum(
        getattr(fcntl, name)
        for name in ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
    )

    class InspectionCompleteError(Exception):
        pass

    def inspect_run(
        command: tuple[str, ...],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        python_descriptor = int(command[0].rsplit("/", 1)[-1])
        helper_descriptor = int(command[4].rsplit("/", 1)[-1])
        packaging_descriptor = int(command[-1])
        helper_writer = os.open(helper_path, os.O_WRONLY | os.O_CLOEXEC)
        packaging_writer = os.open(binding.packaging_wheel, os.O_WRONLY | os.O_CLOEXEC)
        try:
            os.pwrite(helper_writer, bytes([helper_source[0] ^ 1]), 0)
            os.pwrite(packaging_writer, bytes([packaging_source[0] ^ 1]), 0)
            os.fsync(helper_writer)
            os.fsync(packaging_writer)
            assert os.pread(helper_descriptor, len(helper_source), 0) == helper_source
            assert os.pread(packaging_descriptor, len(packaging_source), 0) == packaging_source
            for descriptor, expected_mode in (
                (python_descriptor, 0o500),
                (helper_descriptor, 0o400),
                (packaging_descriptor, 0o400),
            ):
                metadata = os.fstat(descriptor)
                assert stat.S_ISREG(metadata.st_mode)
                assert metadata.st_nlink == 0
                assert stat.S_IMODE(metadata.st_mode) == expected_mode
                assert fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
                assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & required_seals == required_seals
        finally:
            os.pwrite(helper_writer, helper_source[:1], 0)
            os.pwrite(packaging_writer, packaging_source[:1], 0)
            os.fsync(helper_writer)
            os.fsync(packaging_writer)
            os.close(helper_writer)
            os.close(packaging_writer)
        raise InspectionCompleteError

    monkeypatch.setattr(wheelhouse, "_helper_path", lambda: helper_path)
    monkeypatch.setattr(
        wheelhouse,
        "CPU_WHEELHOUSE_HELPER_SOURCE_SHA256",
        _sha(helper_source),
    )
    monkeypatch.setattr(subprocess, "run", inspect_run)
    try:
        with pytest.raises(InspectionCompleteError):
            wheelhouse._invoke_helper(
                manifest={"synthetic": True},
                staged_directory_fd=staged_descriptor,
                verifier=binding,
                descriptors=descriptors,
            )
    finally:
        os.close(staged_descriptor)
        os.close(descriptors.python)
        os.close(descriptors.packaging_tool)


def test_stage_rejects_extra_candidate_file(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    candidate = tmp_path / "candidate"
    _write_wheels(candidate, wheels)
    (candidate / "unexpected.whl").write_bytes(b"unexpected")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    manifest, digest = _manifest_bytes(wheels, ["jax==0.11.0"])
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="enumeration"):
        wheelhouse.stage_matched_v3_cpu_wheelhouse(
            candidate_directory=candidate,
            capture_manifest_raw=manifest,
            expected_capture_manifest_sha256=digest,
            verifier=_binding(tmp_path),
            scratch_directory=scratch,
        )


def test_helper_rejects_corrupt_record_payload_hash(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    filename, raw = _wheel_bytes("jax", "0.11.0", corrupt_record=True)
    wheels[filename] = raw
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="RECORD"):
        _stage(tmp_path, wheels=wheels)


def test_helper_rejects_unindexed_zip_gap(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    filename = next(name for name in wheels if name.startswith("jax-"))
    wheels[filename] = _with_unindexed_zip_gap(wheels[filename])
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="gap"):
        _stage(tmp_path, wheels=wheels)


def test_helper_rejects_regular_path_prefix_conflict(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    filename, raw = _wheel_bytes("jax", "0.11.0", extra_member=("jax", b"conflict\n"))
    wheels[filename] = raw
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="prefix conflict"):
        _stage(tmp_path, wheels=wheels)


def test_closure_rejects_unreachable_extra_distribution(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    filename, raw = _wheel_bytes("unused", "1.0.0")
    wheels[filename] = raw
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="unreachable"):
        _stage(tmp_path, wheels=wheels)


def test_closure_propagates_active_extra_marker(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    filename, raw = _wheel_bytes(
        "continual-foragax",
        "0.55.0",
        requires_dist=('optional-dep==1.0.0; extra == "feature"',),
        provides_extra=("feature",),
    )
    wheels[filename] = raw
    optional_filename, optional_raw = _wheel_bytes("optional-dep", "1.0.0")
    wheels[optional_filename] = optional_raw
    retained = _stage(
        tmp_path,
        wheels=wheels,
        roots=[
            "continual-foragax[feature]==0.55.0",
            "jax==0.11.0",
            "jaxlib==0.11.0",
        ],
    )
    try:
        receipt = retained.receipt()
        assert receipt["closure"]["activated_extras"]["continual-foragax"] == ["feature"]
        assert any(edge["target"] == "optional-dep" for edge in receipt["closure"]["edges"])
    finally:
        retained.close()


def test_closure_overwrites_obsolete_edge_contexts_after_late_extra(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    continual_filename, continual_raw = _wheel_bytes(
        "continual-foragax",
        "0.55.0",
        requires_dist=(
            "jaxlib==0.11.0",
            'optional-dep==1.0.0; extra == "feature"',
        ),
        provides_extra=("feature",),
    )
    jax_filename, jax_raw = _wheel_bytes(
        "jax",
        "0.11.0",
        requires_dist=("continual-foragax[feature]==0.55.0",),
    )
    optional_filename, optional_raw = _wheel_bytes("optional-dep", "1.0.0")
    wheels[continual_filename] = continual_raw
    wheels[jax_filename] = jax_raw
    wheels[optional_filename] = optional_raw
    retained = _stage(tmp_path, wheels=wheels)
    try:
        edges = retained.receipt()["closure"]["edges"]
        stable_edges = [
            edge
            for edge in edges
            if edge["source"] == "continual-foragax" and edge["target"] == "jaxlib"
        ]
        assert len(stable_edges) == 1
        assert stable_edges[0]["active_contexts"] == ["", "feature"]
    finally:
        retained.close()


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("duplicate_package", "package names"),
        ("package_member_size", "archive member differs"),
        ("wrong_tag_rank", "compatible-tag rank"),
        ("critical_version", "critical version differs"),
        ("noncanonical_archive_size", "archive size is not canonical"),
        ("validation_report_identity", "validation-report body identity"),
    ],
)
def test_receipt_parser_rejects_resealed_nested_semantic_tampering(
    tmp_path: Path,
    frozen_current_hashes: None,
    case: str,
    match: str,
) -> None:
    retained = _stage(tmp_path)
    try:
        receipt = retained.receipt()
        rebuild_report = True
        if case == "duplicate_package":
            package = receipt["packages"][1]
            old_filename = package["filename"]
            package["name"] = "continual-foragax"
            package["metadata"]["name"] = "continual-foragax"
            package["filename"] = old_filename.replace("jax-", "continual_foragax-", 1)
            package["dist_info_directory"] = package["dist_info_directory"].replace(
                "jax-",
                "continual_foragax-",
                1,
            )
            package["metadata"]["path"] = f"{package['dist_info_directory']}/METADATA"
            package["wheel"]["path"] = f"{package['dist_info_directory']}/WHEEL"
            package["record"]["path"] = f"{package['dist_info_directory']}/RECORD"
            member = next(
                item for item in receipt["archive"]["members"] if item["filename"] == old_filename
            )
            member["filename"] = package["filename"]
        elif case == "package_member_size":
            receipt["packages"][0]["size_bytes"] += 1
        elif case == "wrong_tag_rank":
            receipt["packages"][0]["best_compatible_tag_rank"] = 1
        elif case == "critical_version":
            package = next(item for item in receipt["packages"] if item["name"] == "jax")
            old_filename = package["filename"]
            new_filename = old_filename.replace("0.11.0", "999.0")
            package["filename"] = new_filename
            package["version"] = "999.0"
            package["metadata"]["version"] = "999.0"
            package["dist_info_directory"] = package["dist_info_directory"].replace(
                "0.11.0",
                "999.0",
            )
            package["metadata"]["path"] = f"{package['dist_info_directory']}/METADATA"
            package["wheel"]["path"] = f"{package['dist_info_directory']}/WHEEL"
            package["record"]["path"] = f"{package['dist_info_directory']}/RECORD"
            member = next(
                item for item in receipt["archive"]["members"] if item["filename"] == old_filename
            )
            member["filename"] = new_filename
        elif case == "noncanonical_archive_size":
            receipt["archive"]["size_bytes"] += 10_240
        elif case == "validation_report_identity":
            receipt["validation_report"]["body_sha256"] = "0" * 64
            rebuild_report = False
        else:  # pragma: no cover - parametrization is closed above
            raise AssertionError(case)
        with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match=match):
            _parse_resealed_receipt(receipt, rebuild_report=rebuild_report)
    finally:
        retained.close()


def test_receipt_parser_rejects_resealed_edge_absent_from_source_metadata(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    filename, raw = _wheel_bytes(
        "continual-foragax",
        "0.55.0",
        requires_dist=("jax==0.11.0",),
    )
    wheels[filename] = raw
    retained = _stage(tmp_path, wheels=wheels)
    try:
        receipt = retained.receipt()
        edge = next(
            item
            for item in receipt["closure"]["edges"]
            if item["source"] == "continual-foragax" and item["target"] == "jax"
        )
        edge["requirement"]["raw"] = "jax>=0.11.0"
        edge["requirement"]["specifier"] = ">=0.11.0"
        with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="source METADATA"):
            _parse_resealed_receipt(receipt)
    finally:
        retained.close()


def test_receipt_parser_rejects_resealed_missing_unconditional_edge(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    filename, raw = _wheel_bytes(
        "continual-foragax",
        "0.55.0",
        requires_dist=("optional-dep==1.0.0",),
    )
    wheels[filename] = raw
    optional_filename, optional_raw = _wheel_bytes("optional-dep", "1.0.0")
    wheels[optional_filename] = optional_raw
    retained = _stage(tmp_path, wheels=wheels)
    try:
        receipt = retained.receipt()
        receipt["closure"]["edges"] = [
            edge
            for edge in receipt["closure"]["edges"]
            if not (edge["source"] == "continual-foragax" and edge["target"] == "optional-dep")
        ]
        with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="unconditional"):
            _parse_resealed_receipt(receipt)
    finally:
        retained.close()


def test_receipt_parser_rejects_resealed_unrequested_activated_extra(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    filename, raw = _wheel_bytes(
        "continual-foragax",
        "0.55.0",
        requires_dist=('optional-dep==1.0.0; extra == "feature"',),
        provides_extra=("feature", "unused"),
    )
    wheels[filename] = raw
    optional_filename, optional_raw = _wheel_bytes("optional-dep", "1.0.0")
    wheels[optional_filename] = optional_raw
    retained = _stage(
        tmp_path,
        wheels=wheels,
        roots=[
            "continual-foragax[feature]==0.55.0",
            "jax==0.11.0",
            "jaxlib==0.11.0",
        ],
    )
    try:
        receipt = retained.receipt()
        receipt["closure"]["activated_extras"]["continual-foragax"] = [
            "feature",
            "unused",
        ]
        with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="activated extras"):
            _parse_resealed_receipt(receipt)
    finally:
        retained.close()


def test_helper_rejects_forbidden_nvidia_distribution(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    filename, raw = _wheel_bytes(
        "continual-foragax",
        "0.55.0",
        requires_dist=("nvidia-cublas==1.0.0",),
    )
    wheels[filename] = raw
    nvidia_filename, nvidia_raw = _wheel_bytes("nvidia-cublas", "1.0.0")
    wheels[nvidia_filename] = nvidia_raw
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="forbidden"):
        _stage(tmp_path, wheels=wheels)


def test_helper_canonicalizes_repeated_identical_metadata_headers(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    requirement = 'optional-dep==1.0.0; extra == "feature"'
    filename, raw = _wheel_bytes(
        "continual-foragax",
        "0.55.0",
        provides_extra=("feature", "feature"),
        requires_dist=(requirement, requirement),
    )
    wheels[filename] = raw

    retained = _stage(tmp_path, wheels=wheels)
    try:
        package = next(
            item for item in retained.receipt()["packages"] if item["name"] == "continual-foragax"
        )
        assert package["metadata"]["provides_extra"] == ["feature"]
        assert package["metadata"]["requires_dist"] == [
            {
                "extras": [],
                "marker": 'extra == "feature"',
                "name": "optional-dep",
                "raw": requirement,
                "specifier": "==1.0.0",
            }
        ]
    finally:
        retained.close()


def test_helper_treats_dynamic_headers_as_informational_for_immutable_wheel_metadata(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    filename, raw = _wheel_bytes(
        "continual-foragax",
        "0.55.0",
        provides_extra=("feature",),
        requires_dist=('optional-dep==1.0.0; extra == "feature"',),
        dynamic=("requires-python", "requires-dist", "provides-extra"),
    )
    wheels[filename] = raw

    retained = _stage(tmp_path, wheels=wheels)
    retained.close()


@pytest.mark.parametrize("accelerator_name", ["jax-rocm60-plugin", "cupy-cuda12x"])
def test_helper_rejects_embedded_accelerator_distribution_segment(
    tmp_path: Path,
    frozen_current_hashes: None,
    accelerator_name: str,
) -> None:
    wheels = _base_wheels()
    filename, raw = _wheel_bytes(
        "continual-foragax",
        "0.55.0",
        requires_dist=(f"{accelerator_name}==1.0.0",),
    )
    wheels[filename] = raw
    accelerator_filename, accelerator_raw = _wheel_bytes(accelerator_name, "1.0.0")
    wheels[accelerator_filename] = accelerator_raw
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="forbidden"):
        _stage(tmp_path, wheels=wheels)


def test_receipt_replay_rejects_embedded_accelerator_distribution_segment(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    filename, raw = _wheel_bytes(
        "continual-foragax",
        "0.55.0",
        requires_dist=("optional-dep==1.0.0",),
    )
    wheels[filename] = raw
    optional_filename, optional_raw = _wheel_bytes("optional-dep", "1.0.0")
    wheels[optional_filename] = optional_raw
    retained = _stage(tmp_path, wheels=wheels)
    try:
        receipt = retained.receipt()
        package = next(item for item in receipt["packages"] if item["name"] == "optional-dep")
        old_filename = package["filename"]
        package["name"] = "jax-rocm60-plugin"
        package["metadata"]["name"] = "jax-rocm60-plugin"
        package["filename"] = old_filename.replace("optional_dep-", "jax_rocm60_plugin-", 1)
        package["dist_info_directory"] = package["dist_info_directory"].replace(
            "optional_dep-",
            "jax_rocm60_plugin-",
            1,
        )
        package["metadata"]["path"] = f"{package['dist_info_directory']}/METADATA"
        package["wheel"]["path"] = f"{package['dist_info_directory']}/WHEEL"
        package["record"]["path"] = f"{package['dist_info_directory']}/RECORD"
        member = next(
            item for item in receipt["archive"]["members"] if item["filename"] == old_filename
        )
        member["filename"] = package["filename"]
        receipt["packages"].sort(key=lambda item: item["name"])
        with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="forbidden"):
            _parse_resealed_receipt(receipt)
    finally:
        retained.close()


def test_helper_rejects_any_platform_wheel_with_non_none_abi_marked_pure(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    original_name = next(name for name in wheels if name.startswith("jax-"))
    del wheels[original_name]
    filename, raw = _wheel_bytes(
        "jax",
        "0.11.0",
        tag="cp312-cp312-any",
        root_is_purelib=True,
    )
    wheels[filename] = raw
    candidate = tmp_path / "candidate"
    _write_wheels(candidate, wheels)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    manifest_raw, _digest = _manifest_bytes(
        wheels,
        ["continual-foragax==0.55.0", "jax==0.11.0", "jaxlib==0.11.0"],
    )
    manifest = json.loads(manifest_raw)
    manifest["target"]["compatible_tags"].insert(0, "cp312-cp312-any")
    del manifest["manifest_body_sha256"]
    manifest["manifest_body_sha256"] = _sha(_canonical(manifest, newline=False))
    changed = _canonical(manifest)
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="platform-any"):
        wheelhouse.stage_matched_v3_cpu_wheelhouse(
            candidate_directory=candidate,
            capture_manifest_raw=changed,
            expected_capture_manifest_sha256=_sha(changed),
            verifier=_binding(tmp_path),
            scratch_directory=scratch,
        )


def test_helper_preserves_purelib_install_scheme_on_platform_wheel(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    original_name = next(name for name in wheels if name.startswith("jax-"))
    del wheels[original_name]
    filename, raw = _wheel_bytes(
        "jax",
        "0.11.0",
        tag="py3-none-manylinux2014_x86_64",
        root_is_purelib=True,
    )
    wheels[filename] = raw
    candidate = tmp_path / "candidate"
    _write_wheels(candidate, wheels)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    manifest_raw, _digest = _manifest_bytes(
        wheels,
        ["continual-foragax==0.55.0", "jax==0.11.0", "jaxlib==0.11.0"],
    )
    manifest = json.loads(manifest_raw)
    manifest["target"]["compatible_tags"].insert(0, "py3-none-manylinux2014_x86_64")
    del manifest["manifest_body_sha256"]
    manifest["manifest_body_sha256"] = _sha(_canonical(manifest, newline=False))
    changed = _canonical(manifest)
    retained = wheelhouse.stage_matched_v3_cpu_wheelhouse(
        candidate_directory=candidate,
        capture_manifest_raw=changed,
        expected_capture_manifest_sha256=_sha(changed),
        verifier=_binding(tmp_path),
        scratch_directory=scratch,
    )
    try:
        package = next(item for item in retained.receipt()["packages"] if item["name"] == "jax")
        assert package["wheel_metadata"]["root_is_purelib"] is True
        _parse_resealed_receipt(retained.receipt())
    finally:
        retained.close()


def test_incompatible_wheel_tag_fails_closed(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    wheels = _base_wheels()
    original_name = next(name for name in wheels if name.startswith("jax-"))
    del wheels[original_name]
    filename, raw = _wheel_bytes("jax", "0.11.0", tag="cp311-cp311-win_amd64")
    wheels[filename] = raw
    with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="incompatible"):
        _stage(tmp_path, wheels=wheels)


def test_published_archive_mutation_is_rejected(
    tmp_path: Path,
    frozen_current_hashes: None,
) -> None:
    retained = _stage(tmp_path)
    publication = tmp_path / "publication"
    publication.mkdir()
    try:
        published = wheelhouse.publish_matched_v3_cpu_wheelhouse(
            retained,
            publication,
            authorize_non_evidence_publication=True,
        )
        published.archive.chmod(0o644)
        with published.archive.open("r+b") as handle:
            handle.seek(513)
            original = handle.read(1)
            handle.seek(513)
            handle.write(bytes([original[0] ^ 1]))
        with pytest.raises(wheelhouse.ForagerMatchedV3CpuWheelhouseError, match="identity"):
            wheelhouse.validate_published_matched_v3_cpu_wheelhouse(
                published.directory,
                expected_receipt_sha256=published.receipt_sha256,
                expected_archive_sha256=published.archive_sha256,
            )
    finally:
        retained.close()


def test_public_api_has_no_download_install_extract_build_or_execute_surface() -> None:
    forbidden = ("download", "install", "extract", "docker", "build_image", "execute", "qualify")
    exported = tuple(wheelhouse.__all__)
    assert not any(token in name for name in exported for token in forbidden)
    source = Path(wheelhouse.__file__).read_text(encoding="utf-8")
    assert "requests" not in source
    assert "urllib.request" not in source
    assert "import packaging" not in source
    assert "shutil.rmtree" not in source
