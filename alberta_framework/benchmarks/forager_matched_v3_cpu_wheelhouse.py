"""Content-only CPU wheelhouse boundary for matched-v3 Forager qualification.

The module separates an untrusted network-capture manifest from disconnected
wheel-byte verification.  Its first-party logic does not resolve, download,
install, extract, or import candidate wheels; build an image; execute a
benchmark; or grant artifact, runtime, qualification, evidence, or promotion
authority.

Wheel semantics are checked from sealed snapshots of the adjacent frozen-hash
helper and a caller-bound CPython executable under ``-I -S -B``.  The helper
loads ``packaging`` only from a sealed snapshot of a separately supplied
exact-hash tool wheel.  Verified candidate bytes are streamed into a canonical
uncompressed POSIX USTAR and returned through a PID-bound read-only retained
capability.  Publication is a separate explicit, new-only, non-evidence
operation with no repository-default path.
"""

from __future__ import annotations

import copy
import ctypes
import dataclasses
import errno
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Final, Never, NoReturn, SupportsIndex, cast
from urllib.parse import urlsplit

CPU_WHEELHOUSE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_wheelhouse_contract.v1"
)
CPU_WHEEL_CAPTURE_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_wheel_capture_manifest.v1"
)
CPU_WHEEL_VALIDATION_REPORT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_wheel_validation_report.v1"
)
CPU_WHEELHOUSE_RECEIPT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_wheelhouse_receipt.v1"
)
CPU_WHEELHOUSE_HELPER_REQUEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_wheel_helper_request.v1"
)
CPU_WHEELHOUSE_STATUS: Final = "implemented_unqualified_non_authorizing"

_HELPER_FILENAME: Final = "_forager_matched_v3_cpu_wheelhouse_helper.py"
# Frozen after the helper and focused tests are stable.
CPU_WHEELHOUSE_HELPER_SOURCE_SHA256: Final = (
    "ea80e1860a0af0d376ed1be0b1c09ef74a34db2d7acd983db53ef4ffa09e99f9"
)

_MAX_WHEELS: Final = 256
_MAX_ROOT_REQUIREMENTS: Final = 512
_MAX_HELPER_SOURCE_BYTES: Final = 4 * 1024 * 1024
_MAX_VERIFIER_PYTHON_BYTES: Final = 256 * 1024 * 1024
_MAX_WHEEL_BYTES: Final = 256 * 1024 * 1024
_MAX_TOTAL_WHEEL_BYTES: Final = 1024 * 1024 * 1024
_MAX_ARCHIVE_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES_PER_WHEEL: Final = 2 * 1024 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES_TOTAL: Final = 8 * 1024 * 1024 * 1024
_MAX_ZIP_MEMBERS_PER_WHEEL: Final = 100_000
_MAX_ZIP_MEMBERS_TOTAL: Final = 1_000_000
_MAX_ZIP_PATH_BYTES: Final = 4096
_MAX_METADATA_BYTES: Final = 8 * 1024 * 1024
_MAX_WHEEL_METADATA_BYTES: Final = 256 * 1024
_MAX_RECORD_BYTES: Final = 64 * 1024 * 1024
_MAX_REQUIRES_DIST: Final = 4096
_MAX_EXTRAS: Final = 512
_MAX_MANIFEST_BYTES: Final = 16 * 1024 * 1024
_MAX_REPORT_BYTES: Final = 16 * 1024 * 1024
_MAX_RECEIPT_BYTES: Final = 16 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 250_000
_MAX_JSON_STRING_BYTES: Final = 8 * 1024
_MAX_COMPATIBLE_TAGS: Final = 4096
_READ_CHUNK_BYTES: Final = 1024 * 1024
_USTAR_BLOCK_BYTES: Final = 512
_USTAR_RECORD_BYTES: Final = 10 * 1024
_WORKER_TIMEOUT_SECONDS: Final = 1800.0

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_PYTHON_VERSION: Final = "3.12.3"
_EXPECTED_PYTHON_VERSION_OUTPUT: Final = b"Python 3.12.3\n"
_EXPECTED_PYTHON_VERSION_OUTPUT_SHA256: Final = (
    "5b3e43dc38ca01e3f5a7854ba50d4330864b0c1e0f646650b2c78cf072acc366"
)
_LIBC_VERSION_RE: Final = re.compile(r"[0-9]+\.[0-9]+\Z")
_WHEEL_FILENAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,250}\.whl\Z")
_ARCHIVE_MEMBER_RE: Final = re.compile(r"[0-9a-f]{64}\.whl\Z")
_TAG_RE: Final = re.compile(r"[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+\Z")
_PACKAGING_VERSION_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}\Z")
_CANONICAL_NAME_RE: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_PEP440_VERSION_RE: Final = re.compile(
    r"""
    v?
    (?:([0-9]+)!)?
    ([0-9]+(?:\.[0-9]+)*)
    (?:[-_.]?(alpha|a|beta|b|preview|pre|c|rc)[-_.]?([0-9]+)?)?
    (?:(?:-([0-9]+))|(?:[-_.]?(post|rev|r)[-_.]?([0-9]+)?))?
    (?:[-_.]?(dev)[-_.]?([0-9]+)?)?
    (?:\+([a-z0-9]+(?:[-_.][a-z0-9]+)*))?
    \Z
    """,
    re.IGNORECASE | re.VERBOSE,
)
_BUILD_TAG_RE: Final = re.compile(r"([0-9]+)([A-Za-z0-9_.]*)\Z")
_ACCELERATOR_SEGMENT_RE: Final = re.compile(
    r"(?:cuda|rocm|nvidia|cublas|cufft|curand|cusolver|cusparse|cudnn|nccl|hip|gpu|xpu|tpu|cupy)"
    r"[0-9a-z]*\Z"
)
_MARKER_KEYS: Final = frozenset(
    {
        "implementation_name",
        "implementation_version",
        "os_name",
        "platform_machine",
        "platform_python_implementation",
        "platform_release",
        "platform_system",
        "platform_version",
        "python_full_version",
        "python_version",
        "sys_platform",
    }
)
_CLAIM_KEYS: Final = frozenset(
    {
        "artifact_accepted",
        "execution_authority_granted",
        "image_qualified",
        "network_isolation_attested",
        "publication_authority_granted",
        "qualification_granted",
        "runtime_qualified",
        "scientific_evidence_created",
        "scientific_promotion_allowed",
        "wheelhouse_installation_reproduced",
    }
)
_CRITICAL_VERSIONS: Final = {
    "continual-foragax": "0.55.0",
    "jax": "0.11.0",
    "jaxlib": "0.11.0",
}
_FORBIDDEN_DISTRIBUTIONS: Final = frozenset(
    {
        "alberta-framework",
        "continual-foragax-agents",
        "jax-cuda12-pjrt",
        "jax-cuda12-plugin",
    }
)


class ForagerMatchedV3CpuWheelhouseError(RuntimeError):
    """The wheelhouse descriptor, input, archive, or publication failed closed."""


def _fail(message: str) -> NoReturn:
    raise ForagerMatchedV3CpuWheelhouseError(message)


def _is_forbidden_accelerator_distribution(name: str) -> bool:
    return name in _FORBIDDEN_DISTRIBUTIONS or any(
        _ACCELERATOR_SEGMENT_RE.fullmatch(segment) is not None for segment in name.split("-")
    )


@dataclasses.dataclass(frozen=True, slots=True)
class WheelhouseVerifierToolBinding:
    """Caller-authenticated interpreter executable and packaging-tool wheel bytes."""

    python_executable: Path
    python_executable_sha256: str
    python_version_output_sha256: str
    packaging_wheel: Path
    packaging_wheel_sha256: str
    packaging_version: str


@dataclasses.dataclass(frozen=True, slots=True)
class PublishedMatchedV3CpuWheelhouse:
    """Paths and content identities of one explicit new-only non-evidence publication."""

    directory: Path
    archive: Path
    receipt: Path
    archive_sha256: str
    receipt_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class _VerifierDescriptors:
    python: int
    python_identity: tuple[int, int, int, int, int, int, int]
    packaging_tool: int
    packaging_tool_identity: tuple[int, int, int, int, int, int, int]


def _claims() -> dict[str, bool]:
    return {key: False for key in sorted(_CLAIM_KEYS)}


def _limitations() -> list[str]:
    return [
        "A capture manifest is untrusted provenance from a separate networked phase.",
        (
            "The host/helper contains no network client and initiates no network operation; "
            "the supplied verifier Python and packaging tool are executable "
            "caller-authenticated inputs whose provenance and side effects are neither "
            "qualified nor isolated here."
        ),
        (
            "The verifier executable file, helper source, and packaging wheel execute from "
            "exact sealed snapshots, but the dynamic loader, shared libraries, standard "
            "library, runtime prefix, kernel, and other interpreter dependencies are not "
            "content-bound by this contract."
        ),
        (
            "Compatible tags and non-derivable marker/platform values are caller-carried "
            "target facts until an exact base runtime replays them."
        ),
        (
            "The receipt links the capture manifest's external full-file identity but does "
            "not embed or independently reconstruct the manifest bytes."
        ),
        (
            "Receipt replay checks wheel filename identity, static critical/forbidden "
            "policy, unconditional edges, and graph reachability; full PEP 508 marker and "
            "specifier semantics remain bound to the exact supplied packaging tool report."
        ),
        "Wheel RECORD identity is distinct from a future installer-produced RECORD identity.",
        "A receipt without the retained archive bytes is insufficient.",
        (
            "No lock, image, runtime, execution, qualification, evidence, or promotion "
            "claim is granted."
        ),
    ]


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": CPU_WHEELHOUSE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION,
        "status": CPU_WHEELHOUSE_STATUS,
        "classification": "cpu_wheel_content_verification_contract_non_authorizing",
        "phase_boundary": {
            "network_capture_implemented_here": False,
            "capture_manifest_status": "untrusted_network_capture_candidate_only",
            "candidate_directory_copied_before_validation": True,
            "candidate_wheels_imported_or_executed_by_first_party_logic": False,
            "supplied_tool_candidate_side_effects_attested": False,
            "index_metadata_consulted_during_validation": False,
            "network_isolation_claimed": False,
        },
        "helper": {
            "relative_path": f"alberta_framework/benchmarks/{_HELPER_FILENAME}",
            "source_sha256": CPU_WHEELHOUSE_HELPER_SOURCE_SHA256,
            "source_sealed_snapshot_inherited": True,
            "interpreter_flags": ["-I", "-S", "-B"],
            "python_executable_patch_version": _EXPECTED_PYTHON_VERSION,
            "python_executable_sealed_snapshot_inherited": True,
            "python_runtime_dependency_bytes_bound": False,
            "ambient_packaging_allowed": False,
            "packaging_tool_wheel_separately_supplied": True,
            "packaging_tool_full_sha256_required_before_import": True,
            "packaging_tool_sealed_snapshot_inherited": True,
            "packaging_tool_code_executed": True,
            "packaging_tool_trust_and_authentication_external": True,
            "packaging_tool_distribution_identity_validated_here": False,
            "packaging_tool_side_effect_isolation_claimed": False,
            "runtime_lock_metadata_wheel_record_file_identities_carried": True,
            "repeated_identical_provides_extra_canonicalized": True,
            "repeated_identical_requires_dist_canonicalized": True,
            "dynamic_metadata_headers_informational_for_immutable_wheel": True,
            "raw_metadata_file_identity_retained": True,
            "root_is_purelib_exact_declaration_retained": True,
            "wheel_tag_compatibility_validated_independently": True,
            "record_entries_sha256_scheme": (
                "canonical-json-sorted-verified-regular-member-path-size-sha256-v1"
            ),
            "candidate_wheels_imported_by_first_party_helper_logic": False,
            "supplied_tool_candidate_side_effects_attested": False,
        },
        "schemas": {
            "capture_manifest": CPU_WHEEL_CAPTURE_MANIFEST_SCHEMA_VERSION,
            "helper_request": CPU_WHEELHOUSE_HELPER_REQUEST_SCHEMA_VERSION,
            "validation_report": CPU_WHEEL_VALIDATION_REPORT_SCHEMA_VERSION,
            "receipt": CPU_WHEELHOUSE_RECEIPT_SCHEMA_VERSION,
        },
        "archive": {
            "format": "canonical_posix_ustar_uncompressed",
            "members": "root_level_sha256_wheel_names_with_original_filename_mapping",
            "member_order": "ascending_ascii_archive_name_bytes",
            "member_type": "regular",
            "mode": "0444",
            "uid": 0,
            "gid": 0,
            "uname": "",
            "gname": "",
            "mtime": 0,
            "payload_padding": "zero_to_512_byte_block",
            "end_blocks": 2,
            "record_padding": "zero_to_exact_10240_byte_multiple",
            "receipt_embedded": False,
        },
        "retained_capability": {
            "pid_bound": True,
            "copyable": False,
            "picklable": False,
            "read_only_descriptor": True,
            "reverify_required": True,
            "download_extract_install_execute_methods": False,
        },
        "publication": {
            "caller_supplied_root": True,
            "default_root": False,
            "explicit_authorization_required": True,
            "content_address": "sha256/<archive_sha256>",
            "files": ["receipt.v1.json", "wheelhouse.v1.tar"],
            "new_only": True,
            "overwrite_allowed": False,
            "classification": "runtime_input_non_evidence",
        },
        "limits": {
            "maximum_wheels": _MAX_WHEELS,
            "maximum_root_requirements": _MAX_ROOT_REQUIREMENTS,
            "maximum_wheel_filename_bytes": 255,
            "archive_member_name_scheme": "lowercase-wheel-sha256-dot-whl",
            "maximum_helper_source_bytes": _MAX_HELPER_SOURCE_BYTES,
            "maximum_verifier_python_bytes": _MAX_VERIFIER_PYTHON_BYTES,
            "maximum_wheel_bytes": _MAX_WHEEL_BYTES,
            "maximum_total_wheel_bytes": _MAX_TOTAL_WHEEL_BYTES,
            "maximum_archive_bytes": _MAX_ARCHIVE_BYTES,
            "maximum_uncompressed_bytes_per_wheel": _MAX_UNCOMPRESSED_BYTES_PER_WHEEL,
            "maximum_uncompressed_bytes_total": _MAX_UNCOMPRESSED_BYTES_TOTAL,
            "maximum_zip_members_per_wheel": _MAX_ZIP_MEMBERS_PER_WHEEL,
            "maximum_zip_members_total": _MAX_ZIP_MEMBERS_TOTAL,
            "maximum_zip_path_bytes": _MAX_ZIP_PATH_BYTES,
            "maximum_metadata_bytes": _MAX_METADATA_BYTES,
            "maximum_wheel_metadata_bytes": _MAX_WHEEL_METADATA_BYTES,
            "maximum_record_bytes": _MAX_RECORD_BYTES,
            "maximum_manifest_bytes": _MAX_MANIFEST_BYTES,
            "maximum_report_bytes": _MAX_REPORT_BYTES,
            "maximum_receipt_bytes": _MAX_RECEIPT_BYTES,
            "maximum_json_depth": _MAX_JSON_DEPTH,
            "maximum_json_nodes": _MAX_JSON_NODES,
            "maximum_json_string_bytes": _MAX_JSON_STRING_BYTES,
            "worker_timeout_seconds": int(_WORKER_TIMEOUT_SECONDS),
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _canonical_json(value: Any, *, newline: bool = True) -> bytes:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return raw + (b"\n" if newline else b"")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
CPU_WHEELHOUSE_CONTRACT_DESCRIPTOR_SHA256: Final = (
    "b74224c7bb0523b87458cb4a08aaf9967b5fd11574927d9635cf9a93bc417331"
)


def cpu_wheelhouse_contract_descriptor() -> dict[str, Any]:
    """Return a detached nonauthorizing descriptor."""

    return cast(dict[str, Any], json.loads(_DESCRIPTOR_BYTES))


def canonical_cpu_wheelhouse_contract_descriptor_bytes() -> bytes:
    """Return the exact canonical descriptor bytes."""

    return _DESCRIPTOR_BYTES


def cpu_wheelhouse_contract_descriptor_sha256() -> str:
    """Return the literal-frozen descriptor SHA-256."""

    observed = _sha256_bytes(_DESCRIPTOR_BYTES)
    if not hmac.compare_digest(observed, CPU_WHEELHOUSE_CONTRACT_DESCRIPTOR_SHA256):
        _fail("CPU wheelhouse contract descriptor literal is stale")
    return CPU_WHEELHOUSE_CONTRACT_DESCRIPTOR_SHA256


def parse_cpu_wheelhouse_contract_descriptor(raw: bytes) -> dict[str, Any]:
    """Accept only the exact literal-frozen contract descriptor."""

    if type(raw) is not bytes or not hmac.compare_digest(raw, _DESCRIPTOR_BYTES):
        _fail("CPU wheelhouse contract descriptor bytes differ")
    cpu_wheelhouse_contract_descriptor_sha256()
    return cpu_wheelhouse_contract_descriptor()


def _strict_json(raw: bytes, *, label: str, maximum: int) -> Any:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        _fail(f"{label} bytes exceed their exact bound")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> NoReturn:
        _fail(f"{label} contains non-finite constant {value}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForagerMatchedV3CpuWheelhouseError(f"{label} is not strict JSON") from exc
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail(f"{label} JSON structure exceeds its bound")
        if item is None or type(item) in {bool, int}:
            return
        if type(item) is str:
            if len(item.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
                _fail(f"{label} JSON string exceeds its bound")
            return
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
            return
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    _fail(f"{label} JSON object key is not text")
                visit(key, depth + 1)
                visit(child, depth + 1)
            return
        _fail(f"{label} JSON contains an unsupported value type")

    visit(value, 0)
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be one lowercase SHA-256")
    return value


def _require_string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        _fail(f"{label} must be one nonempty string")
    return value


def _require_int(value: Any, *, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} is outside its integer bound")
    return value


def _validate_claims(value: Any, *, label: str) -> None:
    if type(value) is not dict:
        _fail(f"{label} claims must be one object")
    claims = cast(dict[str, Any], value)
    _exact_keys(claims, set(_CLAIM_KEYS), label=f"{label} claims")
    if any(item is not False for item in claims.values()):
        _fail(f"{label} must keep every authority claim false")


def _validate_target(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("capture target must be one object")
    target = cast(dict[str, Any], value)
    _exact_keys(
        target,
        {
            "abi",
            "compatible_tags",
            "implementation",
            "libc",
            "marker_environment",
            "oci_platform",
            "platform",
            "python_version",
        },
        label="capture target",
    )
    if (
        target["implementation"] != "CPython"
        or type(target["python_version"]) is not str
        or target["python_version"] != _EXPECTED_PYTHON_VERSION
        or target["abi"] != "cp312"
        or target["platform"] != "linux_x86_64"
        or target["oci_platform"] != "linux/amd64"
    ):
        _fail("capture target is not exact CPython 3.12 Linux x86_64")
    libc = target["libc"]
    if type(libc) is not dict:
        _fail("capture target libc must be one object")
    libc_value = cast(dict[str, Any], libc)
    _exact_keys(libc_value, {"family", "version"}, label="capture target libc")
    if (
        libc_value["family"] != "glibc"
        or type(libc_value["version"]) is not str
        or _LIBC_VERSION_RE.fullmatch(libc_value["version"]) is None
    ):
        _fail("capture target libc identity is invalid")
    environment = target["marker_environment"]
    if type(environment) is not dict or set(environment) != _MARKER_KEYS:
        _fail("capture target marker environment has the wrong key set")
    marker = cast(dict[str, Any], environment)
    if not all(type(item) is str and item for item in marker.values()):
        _fail("capture target marker environment values must be nonempty exact strings")
    python_version = target["python_version"]
    if (
        marker["python_full_version"] != python_version
        or marker["implementation_version"] != python_version
        or marker["python_version"] != "3.12"
        or marker["platform_python_implementation"] != "CPython"
        or marker["implementation_name"] != "cpython"
        or marker["os_name"] != "posix"
        or marker["platform_machine"] != "x86_64"
        or marker["platform_system"] != "Linux"
        or marker["sys_platform"] != "linux"
    ):
        _fail("capture target marker environment contradicts its platform")
    tags = target["compatible_tags"]
    if (
        type(tags) is not list
        or not 1 <= len(tags) <= _MAX_COMPATIBLE_TAGS
        or not all(type(item) is str and _TAG_RE.fullmatch(item) is not None for item in tags)
        or len(tags) != len(set(cast(list[str], tags)))
    ):
        _fail("capture target compatible tags are invalid or duplicate")
    return target


def _validate_capture_manifest(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("capture manifest must contain one object")
    manifest = cast(dict[str, Any], value)
    _exact_keys(
        manifest,
        {
            "capture",
            "claims",
            "classification",
            "manifest_body_sha256",
            "root_requirements",
            "schema_version",
            "status",
            "target",
            "wheels",
        },
        label="capture manifest",
    )
    if (
        manifest["schema_version"] != CPU_WHEEL_CAPTURE_MANIFEST_SCHEMA_VERSION
        or manifest["status"] != "untrusted_network_capture_candidate_only"
        or manifest["classification"] != "networked_solver_output_non_authorizing"
    ):
        _fail("capture manifest schema/status/classification is unsupported")
    _validate_claims(manifest["claims"], label="capture manifest")
    _validate_target(manifest["target"])
    roots = manifest["root_requirements"]
    if (
        type(roots) is not list
        or not 1 <= len(roots) <= _MAX_ROOT_REQUIREMENTS
        or not all(type(item) is str and 0 < len(item) <= _MAX_JSON_STRING_BYTES for item in roots)
        or roots != sorted(set(cast(list[str], roots)))
    ):
        _fail("capture root requirements must be unique sorted bounded strings")
    wheels = manifest["wheels"]
    if type(wheels) is not list or not 1 <= len(wheels) <= _MAX_WHEELS:
        _fail("capture wheel list count is outside its bound")
    filenames: list[str] = []
    total_bytes = 0
    for index, raw_wheel in enumerate(wheels):
        if type(raw_wheel) is not dict:
            _fail(f"capture wheel {index} must be one object")
        wheel = cast(dict[str, Any], raw_wheel)
        _exact_keys(
            wheel,
            {"filename", "origin_url", "sha256", "size_bytes"},
            label=f"capture wheel {index}",
        )
        filename = _require_string(wheel["filename"], label=f"capture wheel {index} filename")
        if (
            len(filename.encode("ascii", "ignore")) != len(filename)
            or len(filename.encode("ascii")) > 255
            or _WHEEL_FILENAME_RE.fullmatch(filename) is None
        ):
            _fail(f"capture wheel {index} filename is not canonical bounded ASCII")
        origin = _require_string(wheel["origin_url"], label=f"capture wheel {index} URL")
        parsed = urlsplit(origin)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            _fail(f"capture wheel {index} origin URL is not canonical HTTPS provenance")
        _require_sha256(wheel["sha256"], label=f"capture wheel {index}")
        size = _require_int(
            wheel["size_bytes"],
            label=f"capture wheel {index} size",
            minimum=1,
            maximum=_MAX_WHEEL_BYTES,
        )
        total_bytes += size
        if total_bytes > _MAX_TOTAL_WHEEL_BYTES:
            _fail("capture wheel bytes exceed their global bound")
        filenames.append(filename)
    if filenames != sorted(filenames) or len(filenames) != len(set(filenames)):
        _fail("capture wheel filenames must be unique and sorted")
    capture = manifest["capture"]
    if type(capture) is not dict:
        _fail("capture provenance must be one object")
    capture_value = cast(dict[str, Any], capture)
    _exact_keys(
        capture_value,
        {
            "network_used",
            "resolver_argv",
            "resolver_binary_sha256",
            "resolver_name",
            "resolver_version",
        },
        label="capture provenance",
    )
    if capture_value["network_used"] is not True:
        _fail("capture provenance must remain explicitly networked and untrusted")
    _require_string(capture_value["resolver_name"], label="capture resolver name")
    _require_string(capture_value["resolver_version"], label="capture resolver version")
    _require_sha256(capture_value["resolver_binary_sha256"], label="capture resolver binary")
    argv = capture_value["resolver_argv"]
    if (
        type(argv) is not list
        or not 1 <= len(argv) <= 512
        or not all(type(item) is str and item and "\x00" not in item for item in argv)
    ):
        _fail("capture resolver argv is invalid")
    supplied_body_sha = _require_sha256(
        manifest["manifest_body_sha256"],
        label="capture manifest body",
    )
    body = copy.deepcopy(manifest)
    del body["manifest_body_sha256"]
    observed_body_sha = _sha256_bytes(_canonical_json(body, newline=False))
    if not hmac.compare_digest(supplied_body_sha, observed_body_sha):
        _fail("capture manifest body SHA-256 differs")
    return manifest


def parse_cpu_wheel_capture_manifest(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Parse exact canonical capture bytes with an independent full-file digest."""

    expected = _require_sha256(expected_file_sha256, label="capture manifest file")
    if type(raw) is not bytes or not hmac.compare_digest(_sha256_bytes(raw), expected):
        _fail("capture manifest full-file SHA-256 differs")
    value = _strict_json(raw, label="capture manifest", maximum=_MAX_MANIFEST_BYTES)
    manifest = _validate_capture_manifest(value)
    if not hmac.compare_digest(raw, _canonical_json(manifest)):
        _fail("capture manifest bytes are not canonical with one trailing newline")
    return copy.deepcopy(manifest)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_descriptor(descriptor: int, size: int, *, label: str) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(_READ_CHUNK_BYTES, size - offset), offset)
        if not block:
            _fail(f"{label} was truncated while hashed")
        digest.update(block)
        offset += len(block)
    if os.pread(descriptor, 1, size):
        _fail(f"{label} exceeds its opened size")
    return digest.hexdigest()


def _open_hashed_path(
    path: Path,
    *,
    label: str,
    maximum: int | None = None,
) -> tuple[int, int, str, tuple[int, int, int, int, int, int, int]]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ForagerMatchedV3CpuWheelhouseError(f"cannot stat {label}") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (maximum is not None and not 1 <= before.st_size <= maximum)
    ):
        _fail(f"{label} must be one bounded single-link regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        digest = _hash_descriptor(descriptor, opened.st_size, label=label)
        after = os.fstat(descriptor)
        located = path.lstat()
        identity = _stat_identity(opened)
        if _stat_identity(before) != identity or identity != _stat_identity(after):
            _fail(f"{label} changed while hashed")
        if identity != _stat_identity(located):
            _fail(f"{label} path changed while hashed")
        return descriptor, opened.st_size, digest, identity
    except BaseException:
        os.close(descriptor)
        raise


def _hash_path(path: Path, *, label: str, maximum: int | None = None) -> tuple[int, str]:
    descriptor, size, digest, _identity = _open_hashed_path(
        path,
        label=label,
        maximum=maximum,
    )
    os.close(descriptor)
    return size, digest


def _open_hashed_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    maximum: int,
) -> tuple[int, int, str, tuple[int, int, int, int, int, int, int]]:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise ForagerMatchedV3CpuWheelhouseError(f"cannot stat {label}") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o444
        or not 1 <= before.st_size <= maximum
    ):
        _fail(f"{label} identity must be one bounded single-link read-only regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        opened = os.fstat(descriptor)
        identity = _stat_identity(opened)
        if identity != _stat_identity(before):
            _fail(f"{label} identity changed while opened")
        digest = _hash_descriptor(descriptor, opened.st_size, label=label)
        after = os.fstat(descriptor)
        located = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if identity != _stat_identity(after) or identity != _stat_identity(located):
            _fail(f"{label} identity changed while hashed")
        return descriptor, opened.st_size, digest, identity
    except BaseException:
        os.close(descriptor)
        raise


def _reverify_bound_descriptor(
    descriptor: int,
    *,
    expected_identity: tuple[int, int, int, int, int, int, int],
    expected_sha256: str,
    label: str,
    sealed_mode: int | None = None,
) -> None:
    before = os.fstat(descriptor)
    if _stat_identity(before) != expected_identity:
        _fail(f"{label} descriptor identity changed")
    if sealed_mode is not None:
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 0
            or stat.S_IMODE(before.st_mode) != sealed_mode
            or fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
            or fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC == 0
            or fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & _required_snapshot_seals()
            != _required_snapshot_seals()
            or os.get_inheritable(descriptor)
        ):
            _fail(f"{label} sealed snapshot metadata changed")
    observed = _hash_descriptor(descriptor, before.st_size, label=label)
    after = os.fstat(descriptor)
    if _stat_identity(after) != expected_identity or not hmac.compare_digest(
        observed,
        expected_sha256,
    ):
        _fail(f"{label} descriptor bytes changed")


def _required_snapshot_seals() -> int:
    values = [
        getattr(fcntl, name, None)
        for name in ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
    ]
    if any(type(value) is not int for value in values):
        _fail("CPU wheelhouse requires full Linux memfd sealing support")
    return sum(cast(int, value) for value in values)


def _sealed_snapshot(
    source_descriptor: int,
    *,
    expected_identity: tuple[int, int, int, int, int, int, int],
    expected_size: int,
    expected_sha256: str,
    label: str,
    mode: int,
) -> tuple[int, tuple[int, int, int, int, int, int, int]]:
    if mode not in {0o400, 0o500}:
        _fail(f"{label} sealed snapshot mode is invalid")
    creator = getattr(os, "memfd_create", None)
    cloexec = getattr(os, "MFD_CLOEXEC", None)
    allow_sealing = getattr(os, "MFD_ALLOW_SEALING", None)
    if creator is None or type(cloexec) is not int or type(allow_sealing) is not int:
        _fail("CPU wheelhouse requires sealed anonymous memfd support")
    writable = -1
    readonly = -1
    try:
        writable = creator(f"alberta-{label.replace(' ', '-')}", cloexec | allow_sealing)
        os.fchmod(writable, 0o600)
        source_before = os.fstat(source_descriptor)
        if (
            _stat_identity(source_before) != expected_identity
            or source_before.st_size != expected_size
        ):
            _fail(f"{label} source descriptor changed before snapshot")
        digest = hashlib.sha256()
        offset = 0
        while offset < expected_size:
            block = os.pread(
                source_descriptor,
                min(_READ_CHUNK_BYTES, expected_size - offset),
                offset,
            )
            if not block:
                _fail(f"{label} source descriptor was truncated during snapshot")
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(writable, view)
                if written <= 0:
                    _fail(f"{label} sealed snapshot write made no progress")
                view = view[written:]
            offset += len(block)
        source_after = os.fstat(source_descriptor)
        if (
            os.pread(source_descriptor, 1, expected_size)
            or _stat_identity(source_after) != expected_identity
            or not hmac.compare_digest(digest.hexdigest(), expected_sha256)
        ):
            _fail(f"{label} source bytes changed during snapshot")
        os.fsync(writable)
        os.fchmod(writable, mode)
        fcntl.fcntl(writable, fcntl.F_ADD_SEALS, _required_snapshot_seals())
        readonly = os.open(
            f"/proc/self/fd/{writable}",
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        snapshot = os.fstat(readonly)
        snapshot_identity = _stat_identity(snapshot)
        if (
            not stat.S_ISREG(snapshot.st_mode)
            or snapshot.st_nlink != 0
            or snapshot.st_size != expected_size
            or stat.S_IMODE(snapshot.st_mode) != mode
            or fcntl.fcntl(readonly, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
            or fcntl.fcntl(readonly, fcntl.F_GETFD) & fcntl.FD_CLOEXEC == 0
            or fcntl.fcntl(readonly, fcntl.F_GET_SEALS) & _required_snapshot_seals()
            != _required_snapshot_seals()
            or os.get_inheritable(readonly)
            or not hmac.compare_digest(
                _hash_descriptor(readonly, expected_size, label=f"sealed {label}"),
                expected_sha256,
            )
            or _stat_identity(os.fstat(readonly)) != snapshot_identity
        ):
            _fail(f"{label} sealed snapshot differs from its exact source bytes")
        result = readonly
        try:
            os.close(writable)
        finally:
            writable = -1
        readonly = -1
        return result, snapshot_identity
    except OSError as exc:
        raise ForagerMatchedV3CpuWheelhouseError(
            f"{label} cannot be copied into an exact sealed snapshot"
        ) from exc
    finally:
        if readonly >= 0:
            os.close(readonly)
        if writable >= 0:
            os.close(writable)


def _validate_verifier(binding: WheelhouseVerifierToolBinding) -> _VerifierDescriptors:
    if type(binding) is not WheelhouseVerifierToolBinding:
        _fail("verifier binding must have the exact frozen dataclass type")
    for value, label in (
        (binding.python_executable_sha256, "verifier Python"),
        (binding.python_version_output_sha256, "verifier Python version output"),
        (binding.packaging_wheel_sha256, "packaging tool wheel"),
    ):
        _require_sha256(value, label=label)
    if (
        type(binding.packaging_version) is not str
        or _PACKAGING_VERSION_RE.fullmatch(binding.packaging_version) is None
    ):
        _fail("packaging tool version is invalid")
    if not hmac.compare_digest(
        binding.python_version_output_sha256,
        _EXPECTED_PYTHON_VERSION_OUTPUT_SHA256,
    ):
        _fail("verifier Python version output is not the frozen CPython 3.12.3 output")
    python_linked = -1
    python_snapshot = -1
    tool_linked = -1
    tool_snapshot = -1
    try:
        python_linked, python_size, python_sha, python_linked_identity = _open_hashed_path(
            binding.python_executable,
            label="verifier Python executable",
            maximum=_MAX_VERIFIER_PYTHON_BYTES,
        )
        if not hmac.compare_digest(python_sha, binding.python_executable_sha256):
            _fail("verifier Python executable SHA-256 differs")
        python_snapshot, python_snapshot_identity = _sealed_snapshot(
            python_linked,
            expected_identity=python_linked_identity,
            expected_size=python_size,
            expected_sha256=binding.python_executable_sha256,
            label="verifier Python executable",
            mode=0o500,
        )
        try:
            os.close(python_linked)
        finally:
            python_linked = -1
        completed = subprocess.run(
            (f"/proc/self/fd/{python_snapshot}", "--version"),
            check=False,
            capture_output=True,
            env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin"},
            pass_fds=(python_snapshot,),
            timeout=30,
        )
        version_raw = completed.stdout + completed.stderr
        if (
            completed.returncode != 0
            or not hmac.compare_digest(version_raw, _EXPECTED_PYTHON_VERSION_OUTPUT)
            or not hmac.compare_digest(
                _sha256_bytes(version_raw),
                binding.python_version_output_sha256,
            )
        ):
            _fail("verifier Python version output differs")
        _reverify_bound_descriptor(
            python_snapshot,
            expected_identity=python_snapshot_identity,
            expected_sha256=binding.python_executable_sha256,
            label="verifier Python executable",
            sealed_mode=0o500,
        )
        tool_linked, tool_size, tool_sha, tool_linked_identity = _open_hashed_path(
            binding.packaging_wheel,
            label="packaging tool wheel",
            maximum=_MAX_WHEEL_BYTES,
        )
        if not hmac.compare_digest(tool_sha, binding.packaging_wheel_sha256):
            _fail("packaging tool wheel SHA-256 differs")
        tool_snapshot, tool_snapshot_identity = _sealed_snapshot(
            tool_linked,
            expected_identity=tool_linked_identity,
            expected_size=tool_size,
            expected_sha256=binding.packaging_wheel_sha256,
            label="packaging tool wheel",
            mode=0o400,
        )
        try:
            os.close(tool_linked)
        finally:
            tool_linked = -1
        result = _VerifierDescriptors(
            python=python_snapshot,
            python_identity=python_snapshot_identity,
            packaging_tool=tool_snapshot,
            packaging_tool_identity=tool_snapshot_identity,
        )
        python_snapshot = -1
        tool_snapshot = -1
        return result
    finally:
        for descriptor in (tool_snapshot, tool_linked, python_snapshot, python_linked):
            if descriptor >= 0:
                os.close(descriptor)


def _directory_fd(path: Path, *, label: str) -> int:
    if type(path) is not type(Path()) or not path.is_absolute():
        _fail(f"{label} must be one exact absolute pathlib.Path")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ForagerMatchedV3CpuWheelhouseError(
            f"{label} is not an exact non-symlink directory"
        ) from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        _fail(f"{label} is not a directory")
    return descriptor


def _copy_candidate_file(
    source_directory_fd: int,
    target_directory_fd: int,
    *,
    filename: str,
    expected_size: int,
    expected_sha256: str,
) -> None:
    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    read_flags |= getattr(os, "O_NOFOLLOW", 0)
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    write_flags |= getattr(os, "O_NOFOLLOW", 0)
    before = os.stat(filename, dir_fd=source_directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != expected_size:
        _fail(f"candidate wheel is not a stable expected regular file: {filename}")
    source = os.open(filename, read_flags, dir_fd=source_directory_fd)
    target = -1
    try:
        opened = os.fstat(source)
        if (before.st_dev, before.st_ino, before.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            _fail(f"candidate wheel changed while opened: {filename}")
        target = os.open(filename, write_flags, 0o600, dir_fd=target_directory_fd)
        digest = hashlib.sha256()
        total = 0
        while total < expected_size:
            block = os.read(source, min(_READ_CHUNK_BYTES, expected_size - total))
            if not block:
                _fail(f"candidate wheel was truncated: {filename}")
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(target, view)
                if written <= 0:
                    _fail("candidate wheel staging write made no progress")
                view = view[written:]
            total += len(block)
        if os.read(source, 1):
            _fail(f"candidate wheel exceeds its captured size: {filename}")
        after = os.fstat(source)
        located = os.stat(filename, dir_fd=source_directory_fd, follow_symlinks=False)

        def identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if identity(before) != identity(opened) or identity(opened) != identity(after):
            _fail(f"candidate wheel changed while copied: {filename}")
        if identity(after) != identity(located):
            _fail(f"candidate wheel path changed while copied: {filename}")
        if total != expected_size or not hmac.compare_digest(digest.hexdigest(), expected_sha256):
            _fail(f"candidate wheel bytes differ from capture manifest: {filename}")
        os.fsync(target)
        os.fchmod(target, 0o444)
    finally:
        os.close(source)
        if target >= 0:
            os.close(target)


def _helper_path() -> Path:
    return Path(__file__).resolve().with_name(_HELPER_FILENAME)


def _invoke_helper(
    *,
    manifest: dict[str, Any],
    staged_directory_fd: int,
    verifier: WheelhouseVerifierToolBinding,
    descriptors: _VerifierDescriptors,
) -> tuple[dict[str, Any], bytes]:
    helper = _helper_path()
    helper_linked, helper_size, helper_sha, helper_identity = _open_hashed_path(
        helper,
        label="wheelhouse helper source",
        maximum=_MAX_HELPER_SOURCE_BYTES,
    )
    helper_descriptor = -1
    if not hmac.compare_digest(helper_sha, CPU_WHEELHOUSE_HELPER_SOURCE_SHA256):
        os.close(helper_linked)
        _fail("wheelhouse helper source SHA-256 differs from the frozen binding")
    try:
        helper_descriptor, helper_snapshot_identity = _sealed_snapshot(
            helper_linked,
            expected_identity=helper_identity,
            expected_size=helper_size,
            expected_sha256=CPU_WHEELHOUSE_HELPER_SOURCE_SHA256,
            label="wheelhouse helper source",
            mode=0o400,
        )
    except BaseException:
        os.close(helper_linked)
        raise
    try:
        os.close(helper_linked)
    except BaseException:
        os.close(helper_descriptor)
        raise
    request = {
        "manifest": manifest,
        "packaging_tool": {
            "sha256": verifier.packaging_wheel_sha256,
            "version": verifier.packaging_version,
        },
        "schema_version": CPU_WHEELHOUSE_HELPER_REQUEST_SCHEMA_VERSION,
    }
    request_raw = _canonical_json(request, newline=False)
    if len(request_raw) > _MAX_MANIFEST_BYTES:
        os.close(helper_descriptor)
        _fail("wheelhouse helper request exceeds its byte bound")
    try:
        request_handle = tempfile.TemporaryFile(mode="w+b")
    except BaseException:
        os.close(helper_descriptor)
        raise
    try:
        request_handle.write(request_raw)
        request_handle.flush()
        os.fsync(request_handle.fileno())
        command = (
            f"/proc/self/fd/{descriptors.python}",
            "-I",
            "-S",
            "-B",
            f"/proc/self/fd/{helper_descriptor}",
            "verify",
            "--request-fd",
            str(request_handle.fileno()),
            "--wheel-directory-fd",
            str(staged_directory_fd),
            "--packaging-wheel-fd",
            str(descriptors.packaging_tool),
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env={
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": "",
            },
            pass_fds=(
                helper_descriptor,
                request_handle.fileno(),
                staged_directory_fd,
                descriptors.packaging_tool,
                descriptors.python,
            ),
            timeout=_WORKER_TIMEOUT_SECONDS,
        )
        _reverify_bound_descriptor(
            helper_descriptor,
            expected_identity=helper_snapshot_identity,
            expected_sha256=CPU_WHEELHOUSE_HELPER_SOURCE_SHA256,
            label="wheelhouse helper source",
            sealed_mode=0o400,
        )
    except subprocess.TimeoutExpired as exc:
        raise ForagerMatchedV3CpuWheelhouseError("wheelhouse helper timed out") from exc
    finally:
        os.close(helper_descriptor)
        request_handle.close()
    _reverify_bound_descriptor(
        descriptors.python,
        expected_identity=descriptors.python_identity,
        expected_sha256=verifier.python_executable_sha256,
        label="verifier Python executable",
        sealed_mode=0o500,
    )
    _reverify_bound_descriptor(
        descriptors.packaging_tool,
        expected_identity=descriptors.packaging_tool_identity,
        expected_sha256=verifier.packaging_wheel_sha256,
        label="packaging tool wheel",
        sealed_mode=0o400,
    )
    if _stat_identity(helper.lstat()) != helper_identity:
        _fail("wheelhouse helper source path changed during execution")
    if completed.returncode != 0 or completed.stderr or not completed.stdout:
        diagnostic = completed.stderr[:4096].decode("utf-8", "replace").strip()
        _fail(f"wheelhouse helper failed closed: {diagnostic}")
    if len(completed.stdout) > _MAX_REPORT_BYTES:
        _fail("wheelhouse helper report exceeds its byte bound")
    report = _strict_json(
        completed.stdout,
        label="wheelhouse helper report",
        maximum=_MAX_REPORT_BYTES,
    )
    if type(report) is not dict:
        _fail("wheelhouse helper report must be one object")
    report_value = cast(dict[str, Any], report)
    _exact_keys(
        report_value,
        {
            "capture_manifest_body_sha256",
            "claims",
            "classification",
            "closure",
            "inventory_sha256",
            "package_count",
            "packages",
            "packaging_tool",
            "report_body_sha256",
            "schema_version",
            "status",
            "total_uncompressed_bytes",
            "total_wheel_bytes",
            "zip_member_count",
        },
        label="wheelhouse helper report",
    )
    if not hmac.compare_digest(completed.stdout, _canonical_json(report_value)):
        _fail("wheelhouse helper report is not canonical")
    if (
        report_value.get("schema_version") != CPU_WHEEL_VALIDATION_REPORT_SCHEMA_VERSION
        or report_value.get("status") != "content_verified_unqualified_non_authorizing"
        or report_value.get("classification")
        != "disconnected_wheel_bytes_validation_non_authorizing"
        or report_value.get("capture_manifest_body_sha256") != manifest["manifest_body_sha256"]
    ):
        _fail("wheelhouse helper report schema/status differs")
    _validate_claims(report_value["claims"], label="wheelhouse helper report")
    if report_value["packaging_tool"] != {
        "sha256": verifier.packaging_wheel_sha256,
        "version": verifier.packaging_version,
    }:
        _fail("wheelhouse helper report packaging-tool binding differs")
    packages = report_value["packages"]
    if (
        type(packages) is not list
        or report_value["package_count"] != len(packages)
        or report_value["package_count"] != len(manifest["wheels"])
        or type(report_value["closure"]) is not dict
    ):
        _fail("wheelhouse helper report package/closure shape differs")
    inventory = [
        {
            "filename": package["filename"],
            "name": package["name"],
            "sha256": package["sha256"],
            "size_bytes": package["size_bytes"],
            "version": package["version"],
        }
        for package in packages
        if type(package) is dict
    ]
    if len(inventory) != len(packages) or report_value["inventory_sha256"] != _sha256_bytes(
        _canonical_json(inventory, newline=False)
    ):
        _fail("wheelhouse helper report inventory identity differs")
    supplied = _require_sha256(
        report_value.get("report_body_sha256"),
        label="wheelhouse helper report body",
    )
    body = copy.deepcopy(report_value)
    del body["report_body_sha256"]
    if not hmac.compare_digest(supplied, _sha256_bytes(_canonical_json(body, newline=False))):
        _fail("wheelhouse helper report body SHA-256 differs")
    return report_value, completed.stdout


def _ustar_octal(value: int, width: int, *, label: str) -> bytes:
    token = format(value, "o").encode("ascii")
    if value < 0 or len(token) > width - 1:
        _fail(f"{label} exceeds its POSIX USTAR field")
    return token.rjust(width - 1, b"0") + b"\0"


def _ustar_header(filename: str, size: int) -> bytes:
    try:
        name = filename.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3CpuWheelhouseError("USTAR filename is not ASCII") from exc
    if len(name) > 100 or _ARCHIVE_MEMBER_RE.fullmatch(filename) is None:
        _fail("digest-addressed wheel name is not exactly representable in root-level USTAR")
    header = bytearray(_USTAR_BLOCK_BYTES)
    header[0 : len(name)] = name
    header[100:108] = _ustar_octal(0o444, 8, label="USTAR mode")
    header[108:116] = _ustar_octal(0, 8, label="USTAR uid")
    header[116:124] = _ustar_octal(0, 8, label="USTAR gid")
    header[124:136] = _ustar_octal(size, 12, label="USTAR size")
    header[136:148] = _ustar_octal(0, 12, label="USTAR mtime")
    header[148:156] = b"        "
    header[156:157] = b"0"
    header[257:263] = b"ustar\0"
    header[263:265] = b"00"
    checksum = format(sum(header), "06o").encode("ascii")
    if len(checksum) != 6:
        _fail("USTAR checksum exceeds its field")
    header[148:156] = checksum + b"\0 "
    return bytes(header)


class _ArchiveWriter:
    def __init__(self, descriptor: int) -> None:
        self.descriptor = descriptor
        self.size = 0
        self.digest = hashlib.sha256()

    def write(self, raw: bytes) -> None:
        if type(raw) is not bytes or self.size + len(raw) > _MAX_ARCHIVE_BYTES:
            _fail("canonical wheelhouse archive exceeds its byte bound")
        view = memoryview(raw)
        while view:
            written = os.write(self.descriptor, view)
            if written <= 0:
                _fail("canonical wheelhouse archive write made no progress")
            view = view[written:]
        self.size += len(raw)
        self.digest.update(raw)


def _write_archive(
    descriptor: int,
    staged_directory_fd: int,
    members: list[dict[str, Any]],
) -> tuple[int, str]:
    writer = _ArchiveWriter(descriptor)
    names = [cast(str, member["archive_name"]) for member in members]
    if names != sorted(names) or len(names) != len(set(names)):
        _fail("USTAR members must be unique and sorted")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    for member in members:
        filename = cast(str, member["filename"])
        archive_name = cast(str, member["archive_name"])
        size = cast(int, member["size_bytes"])
        expected_sha = cast(str, member["sha256"])
        descriptor_in = os.open(filename, flags, dir_fd=staged_directory_fd)
        try:
            before = os.fstat(descriptor_in)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o444
                or before.st_size != size
            ):
                _fail("staged wheel identity differs before archiving")
            writer.write(_ustar_header(archive_name, size))
            digest = hashlib.sha256()
            remaining = size
            while remaining:
                block = os.read(descriptor_in, min(_READ_CHUNK_BYTES, remaining))
                if not block:
                    _fail("staged wheel was truncated during archive creation")
                writer.write(block)
                digest.update(block)
                remaining -= len(block)
            if os.read(descriptor_in, 1) or digest.hexdigest() != expected_sha:
                _fail("staged wheel bytes differ during archive creation")
            padding = (-size) % _USTAR_BLOCK_BYTES
            if padding:
                writer.write(bytes(padding))
            after = os.fstat(descriptor_in)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                _fail("staged wheel changed during archive creation")
        finally:
            os.close(descriptor_in)
    writer.write(bytes(2 * _USTAR_BLOCK_BYTES))
    record_padding = (-writer.size) % _USTAR_RECORD_BYTES
    if record_padding:
        writer.write(bytes(record_padding))
    os.fsync(descriptor)
    return writer.size, writer.digest.hexdigest()


def _pread_exact(descriptor: int, size: int, offset: int, *, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    cursor = offset
    while remaining:
        block = os.pread(descriptor, min(_READ_CHUNK_BYTES, remaining), cursor)
        if not block:
            _fail(f"{label} was truncated")
        chunks.append(block)
        remaining -= len(block)
        cursor += len(block)
    return b"".join(chunks)


def _verify_archive_fd(
    descriptor: int,
    *,
    expected_size: int,
    expected_sha256: str,
    members: list[dict[str, Any]],
) -> None:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size != expected_size
        or expected_size <= 0
        or expected_size > _MAX_ARCHIVE_BYTES
        or expected_size % _USTAR_RECORD_BYTES != 0
    ):
        _fail("retained USTAR descriptor metadata differs")
    offset = 0
    for member in members:
        filename = cast(str, member["archive_name"])
        size = cast(int, member["size_bytes"])
        header = _pread_exact(
            descriptor,
            _USTAR_BLOCK_BYTES,
            offset,
            label=f"USTAR header {filename}",
        )
        if not hmac.compare_digest(header, _ustar_header(filename, size)):
            _fail(f"USTAR header is noncanonical: {filename}")
        offset += _USTAR_BLOCK_BYTES
        digest = hashlib.sha256()
        remaining = size
        while remaining:
            block = _pread_exact(
                descriptor,
                min(_READ_CHUNK_BYTES, remaining),
                offset,
                label=f"USTAR payload {filename}",
            )
            digest.update(block)
            remaining -= len(block)
            offset += len(block)
        if not hmac.compare_digest(digest.hexdigest(), cast(str, member["sha256"])):
            _fail(f"USTAR payload SHA-256 differs: {filename}")
        padding = (-size) % _USTAR_BLOCK_BYTES
        if padding:
            raw_padding = _pread_exact(
                descriptor,
                padding,
                offset,
                label=f"USTAR padding {filename}",
            )
            if any(raw_padding):
                _fail(f"USTAR payload padding is nonzero: {filename}")
            offset += padding
    if any(
        _pread_exact(
            descriptor,
            2 * _USTAR_BLOCK_BYTES,
            offset,
            label="USTAR end blocks",
        )
    ):
        _fail("USTAR end blocks are nonzero")
    offset += 2 * _USTAR_BLOCK_BYTES
    final_size = offset + ((-offset) % _USTAR_RECORD_BYTES)
    if final_size != expected_size:
        _fail("USTAR final record padding length differs")
    tail = expected_size - offset
    if tail and any(_pread_exact(descriptor, tail, offset, label="USTAR record padding")):
        _fail("USTAR record padding is nonzero")
    digest = hashlib.sha256()
    cursor = 0
    while cursor < expected_size:
        block = _pread_exact(
            descriptor,
            min(_READ_CHUNK_BYTES, expected_size - cursor),
            cursor,
            label="complete USTAR",
        )
        digest.update(block)
        cursor += len(block)
    after = os.fstat(descriptor)
    if not hmac.compare_digest(digest.hexdigest(), expected_sha256) or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        _fail("retained USTAR digest or descriptor stability differs")


_RETAINED_CREATION_TOKEN: Final = object()


class RetainedMatchedV3CpuWheelhouse:
    """PID-bound read-only capability for verified nonauthorizing wheel bytes."""

    __slots__ = (
        "_archive_sha256",
        "_archive_size",
        "_descriptor",
        "_device",
        "_ctime_ns",
        "_inode",
        "_members",
        "_mode",
        "_mtime_ns",
        "_owner_pid",
        "_receipt_raw",
        "_receipt_sha256",
    )

    def __init__(
        self,
        token: object,
        *,
        descriptor: int,
        archive_size: int,
        archive_sha256: str,
        members: list[dict[str, Any]],
        receipt_raw: bytes,
    ) -> None:
        if token is not _RETAINED_CREATION_TOKEN:
            raise TypeError("retained wheelhouses require the staging boundary")
        metadata = os.fstat(descriptor)
        self._descriptor = descriptor
        self._device = metadata.st_dev
        self._inode = metadata.st_ino
        self._mode = metadata.st_mode
        self._mtime_ns = metadata.st_mtime_ns
        self._ctime_ns = metadata.st_ctime_ns
        self._archive_size = archive_size
        self._archive_sha256 = archive_sha256
        self._members = copy.deepcopy(members)
        self._receipt_raw = receipt_raw
        self._receipt_sha256 = _sha256_bytes(receipt_raw)
        self._owner_pid = os.getpid()

    def __reduce__(self) -> Never:
        raise TypeError("retained wheelhouses cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("retained wheelhouses cannot be serialized")

    def __copy__(self) -> Never:
        raise TypeError("retained wheelhouses cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("retained wheelhouses cannot be copied")

    def _invalidate(self) -> None:
        descriptor = self._descriptor
        self._descriptor = -1
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _require_active(self) -> int:
        if os.getpid() != self._owner_pid:
            self._invalidate()
            _fail("retained wheelhouse is invalid after a PID change")
        descriptor = self._descriptor
        if descriptor < 0:
            _fail("retained wheelhouse is closed")
        try:
            metadata = os.fstat(descriptor)
            flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
        except OSError as exc:
            self._invalidate()
            raise ForagerMatchedV3CpuWheelhouseError(
                "retained wheelhouse descriptor is inaccessible"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode != self._mode
            or metadata.st_nlink != 0
            or metadata.st_size != self._archive_size
            or (metadata.st_dev, metadata.st_ino) != (self._device, self._inode)
            or metadata.st_mtime_ns != self._mtime_ns
            or metadata.st_ctime_ns != self._ctime_ns
            or flags & os.O_ACCMODE != os.O_RDONLY
            or descriptor_flags & fcntl.FD_CLOEXEC == 0
            or os.get_inheritable(descriptor)
        ):
            self._invalidate()
            _fail("retained wheelhouse descriptor identity drifted")
        return descriptor

    @property
    def closed(self) -> bool:
        return self._descriptor < 0

    @property
    def proc_fd_path(self) -> str:
        return f"/proc/self/fd/{self._require_active()}"

    @property
    def subprocess_pass_fds(self) -> tuple[int, ...]:
        return (self._require_active(),)

    @property
    def archive_size_bytes(self) -> int:
        self._require_active()
        return self._archive_size

    @property
    def archive_sha256(self) -> str:
        self._require_active()
        return self._archive_sha256

    @property
    def receipt_bytes(self) -> bytes:
        self._require_active()
        return self._receipt_raw

    @property
    def receipt_sha256(self) -> str:
        self._require_active()
        return self._receipt_sha256

    def receipt(self) -> dict[str, Any]:
        self._require_active()
        return cast(dict[str, Any], json.loads(self._receipt_raw))

    def reverify(self) -> dict[str, Any]:
        descriptor = self._require_active()
        try:
            _verify_archive_fd(
                descriptor,
                expected_size=self._archive_size,
                expected_sha256=self._archive_sha256,
                members=self._members,
            )
            receipt = parse_cpu_wheelhouse_receipt(
                self._receipt_raw,
                expected_file_sha256=self._receipt_sha256,
            )
            self._require_active()
            return receipt
        except BaseException:
            self._invalidate()
            raise

    def close(self) -> None:
        self._invalidate()

    def __enter__(self) -> RetainedMatchedV3CpuWheelhouse:
        self._require_active()
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()


def _canonical_name(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) > 255 or _CANONICAL_NAME_RE.fullmatch(value) is None:
        _fail(f"{label} must be one bounded canonical distribution name")
    return value


def _canonical_pep440_version(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) > 512:
        _fail(f"{label} must be one bounded PEP 440 version")
    match = _PEP440_VERSION_RE.fullmatch(value)
    if match is None:
        _fail(f"{label} is not a supported PEP 440 version")
    (
        epoch,
        release,
        pre_label,
        pre_number,
        post_number_short,
        post_label,
        post_number,
        dev_label,
        dev_number,
        local,
    ) = match.groups()
    result = ""
    if epoch is not None and int(epoch) != 0:
        result += f"{int(epoch)}!"
    result += ".".join(str(int(item)) for item in release.split("."))
    if pre_label is not None:
        normalized_pre = {
            "alpha": "a",
            "a": "a",
            "beta": "b",
            "b": "b",
            "preview": "rc",
            "pre": "rc",
            "c": "rc",
            "rc": "rc",
        }[pre_label.casefold()]
        result += normalized_pre + str(int(pre_number or "0"))
    if post_number_short is not None or post_label is not None:
        result += ".post" + str(int(post_number_short or post_number or "0"))
    if dev_label is not None:
        result += ".dev" + str(int(dev_number or "0"))
    if local is not None:
        segments = re.split(r"[-_.]", local.casefold())
        normalized_local = [str(int(item)) if item.isdecimal() else item for item in segments]
        result += "+" + ".".join(normalized_local)
    return result


def _validate_wheel_filename_identity(
    filename: str,
    *,
    expected_name: str,
    expected_version: str,
    expected_build: str | None,
    expected_tags: list[str],
) -> None:
    stem = filename.removesuffix(".whl")
    try:
        identity, interpreter, abi, platform = stem.rsplit("-", 3)
    except ValueError as exc:
        raise ForagerMatchedV3CpuWheelhouseError(
            "receipt wheel filename lacks its tag fields"
        ) from exc
    identity_parts = identity.split("-")
    if len(identity_parts) == 2:
        raw_name, raw_version = identity_parts
        filename_build: str | None = None
    elif len(identity_parts) == 3:
        raw_name, raw_version, raw_build = identity_parts
        build_match = _BUILD_TAG_RE.fullmatch(raw_build)
        if build_match is None:
            _fail("receipt wheel filename build tag is invalid")
        filename_build = f"{int(build_match.group(1))}{build_match.group(2)}"
    else:
        _fail("receipt wheel filename identity field count differs")
    filename_name = re.sub(r"[-_.]+", "-", raw_name).casefold()
    filename_version = _canonical_pep440_version(
        raw_version,
        label="receipt wheel filename version",
    )
    filename_tags = {
        f"{item_interpreter}-{item_abi}-{item_platform}"
        for item_interpreter in interpreter.split(".")
        for item_abi in abi.split(".")
        for item_platform in platform.split(".")
    }
    if (
        filename_name != expected_name
        or filename_version != expected_version
        or filename_build != expected_build
        or filename_tags != set(expected_tags)
    ):
        _fail("receipt wheel filename identity differs from its package record")


def _validate_dist_info_identity(
    directory: str,
    *,
    expected_name: str,
    expected_version: str,
) -> None:
    stem = directory.removesuffix(".dist-info")
    if "-" not in stem:
        _fail("receipt dist-info directory lacks its version separator")
    raw_name, raw_version = stem.rsplit("-", 1)
    if (
        re.sub(r"[-_.]+", "-", raw_name).casefold() != expected_name
        or _canonical_pep440_version(
            raw_version,
            label="receipt dist-info version",
        )
        != expected_version
    ):
        _fail("receipt dist-info identity differs from its package record")


def _sorted_unique_canonical_names(
    value: Any,
    *,
    label: str,
    maximum: int,
) -> list[str]:
    if type(value) is not list or len(value) > maximum:
        _fail(f"{label} must be one bounded list")
    names = value
    if any(
        type(item) is not str or len(item) > 255 or _CANONICAL_NAME_RE.fullmatch(item) is None
        for item in names
    ):
        _fail(f"{label} contains a noncanonical distribution name")
    result = cast(list[str], names)
    if result != sorted(set(result)):
        _fail(f"{label} must be unique and sorted")
    return result


def _validate_requirement_record(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be one requirement object")
    record = cast(dict[str, Any], value)
    _exact_keys(record, {"extras", "marker", "name", "raw", "specifier"}, label=label)
    _canonical_name(record["name"], label=f"{label} name")
    _sorted_unique_canonical_names(
        record["extras"],
        label=f"{label} extras",
        maximum=_MAX_EXTRAS,
    )
    _require_string(record["raw"], label=f"{label} raw requirement")
    if type(record["specifier"]) is not str or "\x00" in record["specifier"]:
        _fail(f"{label} specifier must be one exact string")
    marker = record["marker"]
    if marker is not None and (type(marker) is not str or not marker or "\x00" in marker):
        _fail(f"{label} marker must be null or one nonempty exact string")
    return record


def _validate_package_record(value: Any, *, index: int) -> dict[str, Any]:
    label = f"receipt package {index}"
    if type(value) is not dict:
        _fail(f"{label} must be one object")
    package = cast(dict[str, Any], value)
    _exact_keys(
        package,
        {
            "best_compatible_tag_rank",
            "build_tag",
            "compressed_size_bytes",
            "dist_info_directory",
            "filename",
            "metadata",
            "metadata_sha256",
            "name",
            "payload_file_count",
            "payload_inventory_sha256",
            "record",
            "record_sha256",
            "sha256",
            "size_bytes",
            "tags",
            "uncompressed_size_bytes",
            "version",
            "wheel",
            "wheel_metadata",
            "wheel_metadata_sha256",
            "zip_member_count",
            "zip_structure_sha256",
        },
        label=label,
    )
    filename = _require_string(package["filename"], label=f"{label} filename")
    if _WHEEL_FILENAME_RE.fullmatch(filename) is None or len(filename.encode("ascii")) > 255:
        _fail(f"{label} filename is invalid")
    name = _canonical_name(package["name"], label=f"{label} name")
    version = _require_string(package["version"], label=f"{label} version")
    if _canonical_pep440_version(version, label=f"{label} version") != version:
        _fail(f"{label} version is not canonical PEP 440 text")
    size = _require_int(
        package["size_bytes"],
        label=f"{label} size",
        minimum=1,
        maximum=_MAX_WHEEL_BYTES,
    )
    _require_int(
        package["best_compatible_tag_rank"],
        label=f"{label} compatible tag rank",
        minimum=0,
        maximum=_MAX_COMPATIBLE_TAGS - 1,
    )
    _require_int(
        package["compressed_size_bytes"],
        label=f"{label} compressed bytes",
        minimum=0,
        maximum=size,
    )
    payload_count = _require_int(
        package["payload_file_count"],
        label=f"{label} payload count",
        minimum=3,
        maximum=_MAX_ZIP_MEMBERS_PER_WHEEL,
    )
    _require_int(
        package["uncompressed_size_bytes"],
        label=f"{label} uncompressed bytes",
        minimum=1,
        maximum=_MAX_UNCOMPRESSED_BYTES_PER_WHEEL,
    )
    zip_count = _require_int(
        package["zip_member_count"],
        label=f"{label} ZIP member count",
        minimum=3,
        maximum=_MAX_ZIP_MEMBERS_PER_WHEEL,
    )
    if payload_count > zip_count:
        _fail(f"{label} payload count exceeds its ZIP member count")
    for key in (
        "metadata_sha256",
        "payload_inventory_sha256",
        "record_sha256",
        "sha256",
        "wheel_metadata_sha256",
        "zip_structure_sha256",
    ):
        _require_sha256(package[key], label=f"{label} {key}")
    build = package["build_tag"]
    if build is not None and (type(build) is not str or not build or "\x00" in build):
        _fail(f"{label} build tag must be null or one nonempty string")
    dist_info = _require_string(
        package["dist_info_directory"],
        label=f"{label} dist-info directory",
    )
    if "/" in dist_info or "\\" in dist_info or not dist_info.endswith(".dist-info"):
        _fail(f"{label} dist-info directory is not root-level")
    _validate_dist_info_identity(
        dist_info,
        expected_name=name,
        expected_version=version,
    )
    tags = package["tags"]
    if (
        type(tags) is not list
        or not 1 <= len(tags) <= _MAX_COMPATIBLE_TAGS
        or not all(type(item) is str and _TAG_RE.fullmatch(item) is not None for item in tags)
        or tags != sorted(set(cast(list[str], tags)))
    ):
        _fail(f"{label} tags are invalid, duplicate, or unsorted")
    _validate_wheel_filename_identity(
        filename,
        expected_name=name,
        expected_version=version,
        expected_build=build,
        expected_tags=cast(list[str], tags),
    )
    metadata = package["metadata"]
    if type(metadata) is not dict:
        _fail(f"{label} METADATA record must be one object")
    metadata_value = cast(dict[str, Any], metadata)
    _exact_keys(
        metadata_value,
        {
            "metadata_version",
            "name",
            "path",
            "provides_extra",
            "requires_dist",
            "requires_python",
            "sha256",
            "size_bytes",
            "version",
        },
        label=f"{label} METADATA record",
    )
    if metadata_value["metadata_version"] not in {"2.1", "2.2", "2.3", "2.4", "2.5"}:
        _fail(f"{label} METADATA version is unsupported")
    if metadata_value["name"] != name or metadata_value["version"] != version:
        _fail(f"{label} METADATA identity differs")
    if (
        metadata_value["path"] != f"{dist_info}/METADATA"
        or _require_int(
            metadata_value["size_bytes"],
            label=f"{label} METADATA size",
            minimum=1,
            maximum=_MAX_METADATA_BYTES,
        )
        > cast(int, package["uncompressed_size_bytes"])
        or _require_sha256(metadata_value["sha256"], label=f"{label} METADATA")
        != package["metadata_sha256"]
    ):
        _fail(f"{label} METADATA file identity differs")
    _sorted_unique_canonical_names(
        metadata_value["provides_extra"],
        label=f"{label} provided extras",
        maximum=_MAX_EXTRAS,
    )
    requires_python = metadata_value["requires_python"]
    if requires_python is not None and (
        type(requires_python) is not str or not requires_python or "\x00" in requires_python
    ):
        _fail(f"{label} Requires-Python is invalid")
    requirements = metadata_value["requires_dist"]
    if type(requirements) is not list or len(requirements) > _MAX_REQUIRES_DIST:
        _fail(f"{label} Requires-Dist list exceeds its bound")
    requirement_values = [
        _validate_requirement_record(item, label=f"{label} requirement {item_index}")
        for item_index, item in enumerate(requirements)
    ]
    if len({_canonical_json(item, newline=False) for item in requirement_values}) != len(
        requirement_values
    ):
        _fail(f"{label} repeats a requirement record")
    wheel_metadata = package["wheel_metadata"]
    if type(wheel_metadata) is not dict:
        _fail(f"{label} WHEEL record must be one object")
    wheel_value = cast(dict[str, Any], wheel_metadata)
    _exact_keys(
        wheel_value,
        {"build", "generator", "root_is_purelib", "tags", "wheel_version"},
        label=f"{label} WHEEL record",
    )
    generator = wheel_value["generator"]
    if (
        wheel_value["build"] != build
        or type(generator) is not str
        or not 1 <= len(generator) <= 512
        or any(not 32 <= ord(character) <= 126 for character in generator)
        or type(wheel_value["root_is_purelib"]) is not bool
        or wheel_value["tags"] != tags
        or wheel_value["wheel_version"] != "1.0"
    ):
        _fail(f"{label} WHEEL record differs from its package identity")
    if any(
        abi != "none" and platform == "any"
        for _python, abi, platform in (tag.rsplit("-", 2) for tag in tags)
    ):
        _fail(f"{label} WHEEL platform-any tag must use the none ABI")
    wheel = package["wheel"]
    if type(wheel) is not dict:
        _fail(f"{label} WHEEL file identity must be one object")
    wheel_file = cast(dict[str, Any], wheel)
    _exact_keys(
        wheel_file,
        {
            "build",
            "generator",
            "path",
            "root_is_purelib",
            "sha256",
            "size_bytes",
            "tags",
            "wheel_version",
        },
        label=f"{label} WHEEL file identity",
    )
    if (
        {key: wheel_file[key] for key in wheel_value} != wheel_value
        or wheel_file["path"] != f"{dist_info}/WHEEL"
        or _require_int(
            wheel_file["size_bytes"],
            label=f"{label} WHEEL size",
            minimum=1,
            maximum=_MAX_WHEEL_METADATA_BYTES,
        )
        > cast(int, package["uncompressed_size_bytes"])
        or _require_sha256(wheel_file["sha256"], label=f"{label} WHEEL")
        != package["wheel_metadata_sha256"]
    ):
        _fail(f"{label} WHEEL file identity differs")
    record = package["record"]
    if type(record) is not dict:
        _fail(f"{label} RECORD file identity must be one object")
    record_file = cast(dict[str, Any], record)
    _exact_keys(
        record_file,
        {"entries_sha256", "entry_count", "path", "sha256", "size_bytes"},
        label=f"{label} RECORD file identity",
    )
    if (
        record_file["path"] != f"{dist_info}/RECORD"
        or _require_int(
            record_file["size_bytes"],
            label=f"{label} RECORD size",
            minimum=1,
            maximum=_MAX_RECORD_BYTES,
        )
        > cast(int, package["uncompressed_size_bytes"])
        or _require_sha256(record_file["sha256"], label=f"{label} RECORD")
        != package["record_sha256"]
        or _require_int(
            record_file["entry_count"],
            label=f"{label} RECORD entry count",
            minimum=3,
            maximum=_MAX_ZIP_MEMBERS_PER_WHEEL,
        )
        != payload_count
        or _require_sha256(
            record_file["entries_sha256"],
            label=f"{label} RECORD entries",
        )
        != package["payload_inventory_sha256"]
    ):
        _fail(f"{label} RECORD file identity differs")
    return package


def _validate_closure_record(
    value: Any,
    *,
    packages: list[dict[str, Any]],
    root_requirements: list[str],
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("receipt closure must be one object")
    closure = cast(dict[str, Any], value)
    _exact_keys(
        closure,
        {
            "activated_extras",
            "dependency_graph_sha256",
            "edges",
            "reachable_distributions",
            "root_requirements",
        },
        label="receipt closure",
    )
    package_by_name = {cast(str, package["name"]): package for package in packages}
    package_names = sorted(package_by_name)
    reachable = _sorted_unique_canonical_names(
        closure["reachable_distributions"],
        label="receipt reachable distributions",
        maximum=_MAX_WHEELS,
    )
    if reachable != package_names:
        _fail("receipt closure reachability differs from its package set")
    activated = closure["activated_extras"]
    if type(activated) is not dict or sorted(activated) != package_names:
        _fail("receipt closure activated-extra keys differ from its package set")
    activated_value = cast(dict[str, Any], activated)
    for name in package_names:
        values = _sorted_unique_canonical_names(
            activated_value[name],
            label=f"receipt activated extras for {name}",
            maximum=_MAX_EXTRAS,
        )
        provided = set(
            cast(
                list[str],
                cast(dict[str, Any], package_by_name[name]["metadata"])["provides_extra"],
            )
        )
        if not set(values).issubset(provided):
            _fail(f"receipt closure activates an undeclared extra for {name}")
    roots = closure["root_requirements"]
    if type(roots) is not list or len(roots) != len(root_requirements):
        _fail("receipt closure root requirement count differs")
    root_values = [
        _validate_requirement_record(item, label=f"receipt root requirement {index}")
        for index, item in enumerate(roots)
    ]
    if [item["raw"] for item in root_values] != root_requirements:
        _fail("receipt closure root records differ from the capture roots")
    derived_activated: dict[str, set[str]] = {name: set() for name in package_names}
    for record in root_values:
        target = cast(str, record["name"])
        if target not in package_by_name or not set(cast(list[str], record["extras"])).issubset(
            cast(
                list[str],
                cast(dict[str, Any], package_by_name[target]["metadata"])["provides_extra"],
            )
        ):
            _fail("receipt root requirement target or extras differ")
        derived_activated[target].update(cast(list[str], record["extras"]))
    edges = closure["edges"]
    if type(edges) is not list:
        _fail("receipt dependency edges must be one list")
    edge_values: list[dict[str, Any]] = []
    for index, value_edge in enumerate(edges):
        if type(value_edge) is not dict:
            _fail(f"receipt dependency edge {index} must be one object")
        edge = cast(dict[str, Any], value_edge)
        _exact_keys(
            edge,
            {"active_contexts", "requirement", "source", "target"},
            label=f"receipt dependency edge {index}",
        )
        source = _canonical_name(edge["source"], label=f"receipt edge {index} source")
        target = _canonical_name(edge["target"], label=f"receipt edge {index} target")
        if source not in package_by_name or target not in package_by_name:
            _fail("receipt dependency edge references an absent package")
        requirement = _validate_requirement_record(
            edge["requirement"],
            label=f"receipt dependency edge {index} requirement",
        )
        if requirement["name"] != target or not set(
            cast(list[str], requirement["extras"])
        ).issubset(
            cast(
                list[str],
                cast(dict[str, Any], package_by_name[target]["metadata"])["provides_extra"],
            )
        ):
            _fail("receipt dependency edge target/extras differ")
        source_requirements = cast(
            list[dict[str, Any]],
            cast(dict[str, Any], package_by_name[source]["metadata"])["requires_dist"],
        )
        requirement_raw = _canonical_json(requirement, newline=False)
        if requirement_raw not in {
            _canonical_json(item, newline=False) for item in source_requirements
        }:
            _fail("receipt dependency edge is absent from source METADATA")
        derived_activated[target].update(cast(list[str], requirement["extras"]))
        contexts = edge["active_contexts"]
        if (
            type(contexts) is not list
            or not contexts
            or not all(type(item) is str for item in contexts)
            or contexts != sorted(set(cast(list[str], contexts)))
            or not set(cast(list[str], contexts)).issubset(
                {"", *cast(list[str], activated_value[source])}
            )
        ):
            _fail("receipt dependency edge contexts are invalid")
        edge_values.append(edge)
    encoded_edges = [_canonical_json(item, newline=False) for item in edge_values]
    if encoded_edges != sorted(set(encoded_edges)):
        _fail("receipt dependency edges are duplicate or unsorted")
    supplied_edge_requirements = {
        (
            cast(str, edge["source"]),
            _canonical_json(edge["requirement"], newline=False),
        )
        for edge in edge_values
    }
    for source, package in package_by_name.items():
        source_requirements = cast(
            list[dict[str, Any]],
            cast(dict[str, Any], package["metadata"])["requires_dist"],
        )
        for requirement in source_requirements:
            if (
                requirement["marker"] is None
                and (
                    source,
                    _canonical_json(requirement, newline=False),
                )
                not in supplied_edge_requirements
            ):
                _fail("receipt closure omits an unconditional dependency edge")
    derived_reachable = {cast(str, record["name"]) for record in root_values}
    for _iteration in range(len(package_names) + 1):
        before_reachable = frozenset(derived_reachable)
        for edge in edge_values:
            if edge["source"] in derived_reachable:
                derived_reachable.add(cast(str, edge["target"]))
        if frozenset(derived_reachable) == before_reachable:
            break
    else:  # pragma: no cover - a finite package set must converge within this bound
        _fail("receipt dependency reachability did not converge")
    if derived_reachable != set(package_names):
        _fail("receipt closure contains a package unreachable from its roots and edges")
    if any(
        set(cast(list[str], activated_value[name])) != derived_activated[name]
        for name in package_names
    ):
        _fail("receipt activated extras differ from root/edge declarations")
    graph_identity = {
        "activated_extras": activated_value,
        "edges": edge_values,
        "root_requirements": root_values,
    }
    graph_sha = _require_sha256(
        closure["dependency_graph_sha256"],
        label="receipt dependency graph",
    )
    if graph_sha != _sha256_bytes(_canonical_json(graph_identity, newline=False)):
        _fail("receipt dependency graph identity differs")
    return closure


def _canonical_ustar_size(members: list[dict[str, Any]]) -> int:
    size = 0
    for member in members:
        payload_size = cast(int, member["size_bytes"])
        size += _USTAR_BLOCK_BYTES + payload_size + (-payload_size) % _USTAR_BLOCK_BYTES
    size += 2 * _USTAR_BLOCK_BYTES
    return size + (-size) % _USTAR_RECORD_BYTES


def _receipt(
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
    report: dict[str, Any],
    report_raw: bytes,
    verifier: WheelhouseVerifierToolBinding,
    archive_size: int,
    archive_sha256: str,
    members: list[dict[str, Any]],
) -> bytes:
    archive_identity = {
        "format": "canonical_posix_ustar_uncompressed",
        "hash_scheme": "sha256-member-name+original-filename+size+sha256-v1",
        "inventory_sha256": _sha256_bytes(_canonical_json(members, newline=False)),
        "member_count": len(members),
        "members": members,
        "record_size_bytes": _USTAR_RECORD_BYTES,
        "sha256": archive_sha256,
        "size_bytes": archive_size,
    }
    body: dict[str, Any] = {
        "archive": archive_identity,
        "capture_manifest": {
            "body_sha256": manifest["manifest_body_sha256"],
            "full_file_sha256": manifest_sha256,
            "status": manifest["status"],
        },
        "claims": _claims(),
        "classification": "retained_runtime_input_content_non_evidence",
        "closure": report["closure"],
        "contract": {
            "descriptor_sha256": CPU_WHEELHOUSE_CONTRACT_DESCRIPTOR_SHA256,
            "helper_source_sha256": CPU_WHEELHOUSE_HELPER_SOURCE_SHA256,
        },
        "limitations": _limitations(),
        "packages": report["packages"],
        "root_requirements": manifest["root_requirements"],
        "schema_version": CPU_WHEELHOUSE_RECEIPT_SCHEMA_VERSION,
        "status": "content_verified_unqualified_non_authorizing",
        "target": manifest["target"],
        "validation_report": {
            "body_sha256": report["report_body_sha256"],
            "full_file_sha256": _sha256_bytes(report_raw),
            "schema_version": report["schema_version"],
        },
        "verifier": {
            "packaging_tool_version": verifier.packaging_version,
            "packaging_tool_wheel_sha256": verifier.packaging_wheel_sha256,
            "python_executable_sha256": verifier.python_executable_sha256,
            "python_version_output_sha256": verifier.python_version_output_sha256,
        },
    }
    body["receipt_body_sha256"] = _sha256_bytes(_canonical_json(body, newline=False))
    raw = _canonical_json(body)
    if len(raw) > _MAX_RECEIPT_BYTES:
        _fail("wheelhouse receipt exceeds its byte bound")
    return raw


def parse_cpu_wheelhouse_receipt(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Parse a canonical receipt and replay every internally reconstructible identity."""

    cpu_wheelhouse_contract_descriptor_sha256()
    expected = _require_sha256(expected_file_sha256, label="wheelhouse receipt file")
    if type(raw) is not bytes or not hmac.compare_digest(_sha256_bytes(raw), expected):
        _fail("wheelhouse receipt full-file SHA-256 differs")
    value = _strict_json(raw, label="wheelhouse receipt", maximum=_MAX_RECEIPT_BYTES)
    if type(value) is not dict:
        _fail("wheelhouse receipt must contain one object")
    receipt = cast(dict[str, Any], value)
    _exact_keys(
        receipt,
        {
            "archive",
            "capture_manifest",
            "claims",
            "classification",
            "closure",
            "contract",
            "limitations",
            "packages",
            "receipt_body_sha256",
            "root_requirements",
            "schema_version",
            "status",
            "target",
            "validation_report",
            "verifier",
        },
        label="wheelhouse receipt",
    )
    if (
        receipt["schema_version"] != CPU_WHEELHOUSE_RECEIPT_SCHEMA_VERSION
        or receipt["status"] != "content_verified_unqualified_non_authorizing"
        or receipt["classification"] != "retained_runtime_input_content_non_evidence"
    ):
        _fail("wheelhouse receipt schema/status/classification differs")
    _validate_claims(receipt["claims"], label="wheelhouse receipt")
    if receipt["limitations"] != _limitations():
        _fail("wheelhouse receipt limitations differ")
    if type(receipt["contract"]) is not dict:
        _fail("receipt contract must be one object")
    contract = cast(dict[str, Any], receipt["contract"])
    _exact_keys(contract, {"descriptor_sha256", "helper_source_sha256"}, label="receipt contract")
    if contract != {
        "descriptor_sha256": CPU_WHEELHOUSE_CONTRACT_DESCRIPTOR_SHA256,
        "helper_source_sha256": CPU_WHEELHOUSE_HELPER_SOURCE_SHA256,
    }:
        _fail("wheelhouse receipt contract binding differs")
    target = _validate_target(receipt["target"])
    capture = receipt["capture_manifest"]
    if type(capture) is not dict:
        _fail("receipt capture-manifest identity must be one object")
    capture_value = cast(dict[str, Any], capture)
    _exact_keys(
        capture_value,
        {"body_sha256", "full_file_sha256", "status"},
        label="receipt capture-manifest identity",
    )
    capture_body_sha = _require_sha256(
        capture_value["body_sha256"],
        label="receipt capture-manifest body",
    )
    _require_sha256(
        capture_value["full_file_sha256"],
        label="receipt capture-manifest file",
    )
    if capture_value["status"] != "untrusted_network_capture_candidate_only":
        _fail("receipt capture-manifest status differs")
    roots = receipt["root_requirements"]
    if (
        type(roots) is not list
        or not 1 <= len(roots) <= _MAX_ROOT_REQUIREMENTS
        or not all(type(item) is str and item and "\x00" not in item for item in roots)
        or roots != sorted(set(cast(list[str], roots)))
    ):
        _fail("receipt root requirements must be unique sorted bounded strings")
    root_values = cast(list[str], roots)
    verifier = receipt["verifier"]
    if type(verifier) is not dict:
        _fail("receipt verifier binding must be one object")
    verifier_value = cast(dict[str, Any], verifier)
    _exact_keys(
        verifier_value,
        {
            "packaging_tool_version",
            "packaging_tool_wheel_sha256",
            "python_executable_sha256",
            "python_version_output_sha256",
        },
        label="receipt verifier binding",
    )
    if (
        type(verifier_value["packaging_tool_version"]) is not str
        or _PACKAGING_VERSION_RE.fullmatch(verifier_value["packaging_tool_version"]) is None
    ):
        _fail("receipt packaging-tool version is invalid")
    for key in (
        "packaging_tool_wheel_sha256",
        "python_executable_sha256",
        "python_version_output_sha256",
    ):
        _require_sha256(verifier_value[key], label=f"receipt verifier {key}")
    packages = receipt["packages"]
    if type(packages) is not list or not 1 <= len(packages) <= _MAX_WHEELS:
        _fail("receipt packages must be one bounded nonempty list")
    package_values = [
        _validate_package_record(item, index=index) for index, item in enumerate(packages)
    ]
    package_names = [cast(str, package["name"]) for package in package_values]
    package_filenames = [cast(str, package["filename"]) for package in package_values]
    if package_names != sorted(package_names) or len(package_names) != len(set(package_names)):
        _fail("receipt package names must be unique and sorted")
    if len(package_filenames) != len(set(package_filenames)):
        _fail("receipt package filenames must be unique")
    packages_by_name = {cast(str, package["name"]): package for package in package_values}
    for critical_name, critical_version in _CRITICAL_VERSIONS.items():
        if (
            critical_name not in packages_by_name
            or packages_by_name[critical_name]["version"] != critical_version
        ):
            _fail(f"receipt critical version differs: {critical_name}=={critical_version}")
    if any(_is_forbidden_accelerator_distribution(name) for name in package_names):
        _fail("receipt contains a forbidden root/accelerator distribution")
    compatible_tags = cast(list[str], target["compatible_tags"])
    for package in package_values:
        intersections = set(cast(list[str], package["tags"])).intersection(compatible_tags)
        if not intersections or package["best_compatible_tag_rank"] != min(
            compatible_tags.index(tag) for tag in intersections
        ):
            _fail("receipt package compatible-tag rank differs from its target")
    closure = _validate_closure_record(
        receipt["closure"],
        packages=package_values,
        root_requirements=root_values,
    )
    if type(receipt["archive"]) is not dict:
        _fail("receipt archive must be one object")
    archive = cast(dict[str, Any], receipt["archive"])
    _exact_keys(
        archive,
        {
            "format",
            "hash_scheme",
            "inventory_sha256",
            "member_count",
            "members",
            "record_size_bytes",
            "sha256",
            "size_bytes",
        },
        label="receipt archive",
    )
    if (
        archive["format"] != "canonical_posix_ustar_uncompressed"
        or archive["hash_scheme"] != "sha256-member-name+original-filename+size+sha256-v1"
        or archive["record_size_bytes"] != _USTAR_RECORD_BYTES
    ):
        _fail("wheelhouse receipt archive contract differs")
    _require_sha256(archive["sha256"], label="receipt archive")
    members = archive["members"]
    if type(members) is not list or not 1 <= len(members) <= _MAX_WHEELS:
        _fail("wheelhouse receipt archive members are invalid")
    if archive["member_count"] != len(members):
        _fail("wheelhouse receipt archive member count differs")
    member_values = cast(list[dict[str, Any]], members)
    archive_names: list[str] = []
    filenames: list[str] = []
    for index, member in enumerate(member_values):
        if type(member) is not dict:
            _fail("wheelhouse receipt archive member is not an object")
        _exact_keys(
            member,
            {"archive_name", "filename", "mode", "sha256", "size_bytes"},
            label=f"member {index}",
        )
        archive_name = _require_string(
            member["archive_name"],
            label=f"member {index} archive name",
        )
        filename = _require_string(member["filename"], label=f"member {index} filename")
        member_sha256 = _require_sha256(member["sha256"], label=f"member {index}")
        if (
            _ARCHIVE_MEMBER_RE.fullmatch(archive_name) is None
            or archive_name != f"{member_sha256}.whl"
            or _WHEEL_FILENAME_RE.fullmatch(filename) is None
            or len(filename.encode("ascii")) > 255
        ):
            _fail("wheelhouse receipt archive filename is invalid")
        if member["mode"] != "0444":
            _fail("wheelhouse receipt archive mode differs")
        _require_int(
            member["size_bytes"],
            label=f"member {index} size",
            minimum=1,
            maximum=_MAX_WHEEL_BYTES,
        )
        archive_names.append(archive_name)
        filenames.append(filename)
    if archive_names != sorted(archive_names) or len(archive_names) != len(set(archive_names)):
        _fail("wheelhouse receipt archive members are not unique and sorted")
    if len(filenames) != len(set(filenames)):
        _fail("wheelhouse receipt archive original filenames are not unique")
    if len(member_values) != len(package_values):
        _fail("wheelhouse receipt archive/package counts differ")
    packages_by_filename = {cast(str, package["filename"]): package for package in package_values}
    for member in member_values:
        linked_package = packages_by_filename.get(cast(str, member["filename"]))
        if linked_package is None or (
            member["sha256"],
            member["size_bytes"],
        ) != (
            linked_package["sha256"],
            linked_package["size_bytes"],
        ):
            _fail("wheelhouse receipt archive member differs from its package")
    inventory_sha = _require_sha256(archive["inventory_sha256"], label="archive inventory")
    if inventory_sha != _sha256_bytes(_canonical_json(member_values, newline=False)):
        _fail("wheelhouse receipt archive inventory SHA-256 differs")
    archive_size = _require_int(
        archive["size_bytes"],
        label="archive size",
        minimum=_USTAR_RECORD_BYTES,
        maximum=_MAX_ARCHIVE_BYTES,
    )
    if archive_size != _canonical_ustar_size(member_values):
        _fail("wheelhouse receipt archive size is not canonical for its members")
    total_wheel_bytes = sum(cast(int, package["size_bytes"]) for package in package_values)
    total_uncompressed_bytes = sum(
        cast(int, package["uncompressed_size_bytes"]) for package in package_values
    )
    total_zip_members = sum(cast(int, package["zip_member_count"]) for package in package_values)
    if (
        total_wheel_bytes > _MAX_TOTAL_WHEEL_BYTES
        or total_uncompressed_bytes > _MAX_UNCOMPRESSED_BYTES_TOTAL
        or total_zip_members > _MAX_ZIP_MEMBERS_TOTAL
    ):
        _fail("wheelhouse receipt aggregate package limits are exceeded")
    validation = receipt["validation_report"]
    if type(validation) is not dict:
        _fail("receipt validation-report identity must be one object")
    validation_value = cast(dict[str, Any], validation)
    _exact_keys(
        validation_value,
        {"body_sha256", "full_file_sha256", "schema_version"},
        label="receipt validation-report identity",
    )
    report_body_sha = _require_sha256(
        validation_value["body_sha256"],
        label="receipt validation-report body",
    )
    report_file_sha = _require_sha256(
        validation_value["full_file_sha256"],
        label="receipt validation-report file",
    )
    if validation_value["schema_version"] != CPU_WHEEL_VALIDATION_REPORT_SCHEMA_VERSION:
        _fail("receipt validation-report schema differs")
    report_inventory = [
        {
            "filename": package["filename"],
            "name": package["name"],
            "sha256": package["sha256"],
            "size_bytes": package["size_bytes"],
            "version": package["version"],
        }
        for package in package_values
    ]
    reconstructed_report: dict[str, Any] = {
        "capture_manifest_body_sha256": capture_body_sha,
        "claims": _claims(),
        "classification": "disconnected_wheel_bytes_validation_non_authorizing",
        "closure": closure,
        "inventory_sha256": _sha256_bytes(_canonical_json(report_inventory, newline=False)),
        "package_count": len(package_values),
        "packages": package_values,
        "packaging_tool": {
            "sha256": verifier_value["packaging_tool_wheel_sha256"],
            "version": verifier_value["packaging_tool_version"],
        },
        "schema_version": CPU_WHEEL_VALIDATION_REPORT_SCHEMA_VERSION,
        "status": "content_verified_unqualified_non_authorizing",
        "total_uncompressed_bytes": total_uncompressed_bytes,
        "total_wheel_bytes": total_wheel_bytes,
        "zip_member_count": total_zip_members,
    }
    observed_report_body_sha = _sha256_bytes(_canonical_json(reconstructed_report, newline=False))
    if not hmac.compare_digest(report_body_sha, observed_report_body_sha):
        _fail("receipt validation-report body identity differs")
    reconstructed_report["report_body_sha256"] = observed_report_body_sha
    if not hmac.compare_digest(
        report_file_sha,
        _sha256_bytes(_canonical_json(reconstructed_report)),
    ):
        _fail("receipt validation-report full-file identity differs")
    supplied_body = _require_sha256(receipt["receipt_body_sha256"], label="receipt body")
    body = copy.deepcopy(receipt)
    del body["receipt_body_sha256"]
    if supplied_body != _sha256_bytes(_canonical_json(body, newline=False)):
        _fail("wheelhouse receipt body SHA-256 differs")
    if not hmac.compare_digest(raw, _canonical_json(receipt)):
        _fail("wheelhouse receipt is not canonical")
    return copy.deepcopy(receipt)


def _remove_owned_directory_entry(
    parent_fd: int,
    name: str,
    directory_fd: int,
    *,
    expected_names: tuple[str, ...],
    label: str,
) -> None:
    """Remove only a still-named directory capability and its expected flat files."""

    opened = os.fstat(directory_fd)
    located = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(located.st_mode)
        or (opened.st_dev, opened.st_ino) != (located.st_dev, located.st_ino)
    ):
        _fail(f"{label} cleanup path no longer names its retained directory")
    observed = sorted(os.listdir(directory_fd))
    allowed = frozenset(expected_names)
    if len(allowed) != len(expected_names) or any(item not in allowed for item in observed):
        _fail(f"{label} cleanup encountered an unexpected entry")
    os.fchmod(directory_fd, 0o700)
    for item in observed:
        os.unlink(item, dir_fd=directory_fd)
    os.fsync(directory_fd)
    located_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(located_after.st_mode)
        or (opened.st_dev, opened.st_ino) != (located_after.st_dev, located_after.st_ino)
        or os.listdir(directory_fd)
    ):
        _fail(f"{label} cleanup path changed before removal")
    os.rmdir(name, dir_fd=parent_fd)


def stage_matched_v3_cpu_wheelhouse(
    *,
    candidate_directory: Path,
    capture_manifest_raw: bytes,
    expected_capture_manifest_sha256: str,
    verifier: WheelhouseVerifierToolBinding,
    scratch_directory: Path,
) -> RetainedMatchedV3CpuWheelhouse:
    """Verify caller-enumerated wheel bytes and retain one canonical USTAR.

    First-party logic contains no network client and grants no authority.  The
    caller remains responsible for authenticating the verifier Python, its
    runtime dependencies, and the executable packaging tool, and for enforcing
    an actually disconnected process environment; this module deliberately
    records no such claim.
    """

    cpu_wheelhouse_contract_descriptor_sha256()
    manifest = parse_cpu_wheel_capture_manifest(
        capture_manifest_raw,
        expected_file_sha256=expected_capture_manifest_sha256,
    )
    descriptors = _validate_verifier(verifier)
    source_fd = -1
    scratch_fd = -1
    staging_fd = -1
    retained_fd = -1
    staging_name: str | None = None
    archive_handle: Any | None = None
    archive_size = -1
    archive_sha = ""
    members: list[dict[str, Any]] | None = None
    receipt_raw: bytes | None = None
    failure: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
        source_fd = _directory_fd(candidate_directory, label="candidate wheel directory")
        scratch_fd = _directory_fd(scratch_directory, label="wheelhouse scratch directory")
        scratch_proc_path = Path(f"/proc/self/fd/{scratch_fd}")
        expected_names = [cast(str, item["filename"]) for item in manifest["wheels"]]
        observed_names = sorted(os.listdir(source_fd))
        if observed_names != expected_names:
            _fail("candidate wheel directory differs from its explicit enumeration")
        staging_path = Path(
            tempfile.mkdtemp(prefix=".matched-v3-wheelhouse-", dir=scratch_proc_path)
        )
        staging_name = staging_path.name
        created = os.stat(staging_name, dir_fd=scratch_fd, follow_symlinks=False)
        staging_fd = os.open(
            staging_name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=scratch_fd,
        )
        opened = os.fstat(staging_fd)
        located = os.stat(staging_name, dir_fd=scratch_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (created.st_dev, created.st_ino) != (opened.st_dev, opened.st_ino)
            or (opened.st_dev, opened.st_ino) != (located.st_dev, located.st_ino)
        ):
            _fail("private wheel staging directory identity changed while opened")
        os.fchmod(staging_fd, 0o700)
        for raw_wheel in manifest["wheels"]:
            wheel = cast(dict[str, Any], raw_wheel)
            _copy_candidate_file(
                source_fd,
                staging_fd,
                filename=cast(str, wheel["filename"]),
                expected_size=cast(int, wheel["size_bytes"]),
                expected_sha256=cast(str, wheel["sha256"]),
            )
        os.fsync(staging_fd)
        report, report_raw = _invoke_helper(
            manifest=manifest,
            staged_directory_fd=staging_fd,
            verifier=verifier,
            descriptors=descriptors,
        )
        members = sorted(
            [
                {
                    "archive_name": f"{wheel['sha256']}.whl",
                    "filename": wheel["filename"],
                    "mode": "0444",
                    "sha256": wheel["sha256"],
                    "size_bytes": wheel["size_bytes"],
                }
                for wheel in manifest["wheels"]
            ],
            key=lambda member: cast(str, member["archive_name"]).encode("ascii"),
        )
        archive_handle = tempfile.TemporaryFile(mode="w+b", dir=scratch_proc_path)
        archive_size, archive_sha = _write_archive(
            archive_handle.fileno(),
            staging_fd,
            members,
        )
        os.fchmod(archive_handle.fileno(), 0o400)
        retained_fd = os.open(
            f"/proc/self/fd/{archive_handle.fileno()}",
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        archive_handle.close()
        archive_handle = None
        metadata = os.fstat(retained_fd)
        if metadata.st_nlink != 0 or fcntl.fcntl(retained_fd, fcntl.F_GETFL) & os.O_ACCMODE:
            _fail("retained wheelhouse archive is not an unlinked read-only file")
        _verify_archive_fd(
            retained_fd,
            expected_size=archive_size,
            expected_sha256=archive_sha,
            members=members,
        )
        receipt_raw = _receipt(
            manifest=manifest,
            manifest_sha256=expected_capture_manifest_sha256,
            report=report,
            report_raw=report_raw,
            verifier=verifier,
            archive_size=archive_size,
            archive_sha256=archive_sha,
            members=members,
        )
        parse_cpu_wheelhouse_receipt(
            receipt_raw,
            expected_file_sha256=_sha256_bytes(receipt_raw),
        )
    except BaseException as exc:
        failure = exc

    if archive_handle is not None:
        try:
            archive_handle.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
    if staging_fd >= 0 and staging_name is not None and scratch_fd >= 0:
        try:
            _remove_owned_directory_entry(
                scratch_fd,
                staging_name,
                staging_fd,
                expected_names=tuple(cast(str, item["filename"]) for item in manifest["wheels"]),
                label="private wheel staging directory",
            )
        except BaseException as exc:
            cleanup_errors.append(exc)
    for descriptor, label in (
        (staging_fd, "private wheel staging directory descriptor"),
        (source_fd, "candidate wheel directory descriptor"),
        (scratch_fd, "wheelhouse scratch directory descriptor"),
        (descriptors.python, "verifier Python descriptor"),
        (descriptors.packaging_tool, "packaging tool descriptor"),
    ):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as exc:
                exc.add_note(f"while closing {label}")
                cleanup_errors.append(exc)

    if failure is not None or cleanup_errors:
        if retained_fd >= 0:
            try:
                os.close(retained_fd)
            except OSError as exc:
                exc.add_note("while closing an untransferred retained archive descriptor")
                cleanup_errors.append(exc)
        if failure is not None:
            for cleanup_error in cleanup_errors:
                failure.add_note(f"cleanup also failed: {cleanup_error!r}")
            raise failure
        primary_cleanup_error = cleanup_errors[0]
        for cleanup_error in cleanup_errors[1:]:
            primary_cleanup_error.add_note(f"cleanup also failed: {cleanup_error!r}")
        raise primary_cleanup_error

    if retained_fd < 0 or members is None or receipt_raw is None:
        _fail("wheelhouse staging completed without a retained archive payload")
    try:
        retained = RetainedMatchedV3CpuWheelhouse(
            _RETAINED_CREATION_TOKEN,
            descriptor=retained_fd,
            archive_size=archive_size,
            archive_sha256=archive_sha,
            members=members,
            receipt_raw=receipt_raw,
        )
    except BaseException:
        os.close(retained_fd)
        raise
    retained_fd = -1
    return retained


def _write_new_file(directory_fd: int, name: str, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("new-only publication write made no progress")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_retained_archive(
    retained: RetainedMatchedV3CpuWheelhouse,
    directory_fd: int,
) -> None:
    source = retained._require_active()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    output = os.open("wheelhouse.v1.tar", flags, 0o600, dir_fd=directory_fd)
    try:
        digest = hashlib.sha256()
        offset = 0
        while offset < retained.archive_size_bytes:
            block = os.pread(
                source,
                min(_READ_CHUNK_BYTES, retained.archive_size_bytes - offset),
                offset,
            )
            if not block:
                _fail("retained wheelhouse was truncated during publication")
            view = memoryview(block)
            while view:
                written = os.write(output, view)
                if written <= 0:
                    _fail("wheelhouse publication write made no progress")
                view = view[written:]
            digest.update(block)
            offset += len(block)
        if digest.hexdigest() != retained.archive_sha256:
            _fail("published wheelhouse digest differs from retained capability")
        os.fchmod(output, 0o444)
        os.fsync(output)
    finally:
        os.close(output)


def _rename_new_only(root_fd: int, source: str, target: str) -> None:
    renameat2 = cast(Any, getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None))
    if renameat2 is None:
        _fail("atomic new-only directory publication is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(root_fd, os.fsencode(source), root_fd, os.fsencode(target), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(target)
    raise OSError(error_number, os.strerror(error_number), target)


def publish_matched_v3_cpu_wheelhouse(
    retained: RetainedMatchedV3CpuWheelhouse,
    publication_root: Path,
    *,
    authorize_non_evidence_publication: bool,
) -> PublishedMatchedV3CpuWheelhouse:
    """Publish ``sha256/<archive digest>`` atomically without overwriting.

    The caller supplies the directory that will contain the ``sha256``
    namespace.  Publication remains a non-evidence content operation.
    """

    if authorize_non_evidence_publication is not True:
        _fail("non-evidence wheelhouse publication requires explicit authorization")
    if type(retained) is not RetainedMatchedV3CpuWheelhouse:
        _fail("publication requires one exact retained wheelhouse capability")
    retained.reverify()
    root_fd = _directory_fd(publication_root, label="wheelhouse publication root")
    namespace = "sha256"
    namespace_fd = -1
    staging_fd = -1
    digest = retained.archive_sha256
    receipt_sha256 = retained.receipt_sha256
    staging = f".staging-{digest}-{os.getpid()}-{secrets.token_hex(16)}"
    entry_name: str | None = None
    committed = False
    succeeded = False
    failure: BaseException | None = None
    collision: FileExistsError | None = None
    cleanup_errors: list[BaseException] = []
    try:
        try:
            os.mkdir(namespace, 0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        namespace_fd = os.open(
            namespace,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
        root_identity = _stat_identity(os.fstat(root_fd))
        retained_namespace = os.fstat(namespace_fd)
        namespace_identity = (retained_namespace.st_dev, retained_namespace.st_ino)
        os.mkdir(staging, 0o700, dir_fd=namespace_fd)
        entry_name = staging
        created = os.stat(staging, dir_fd=namespace_fd, follow_symlinks=False)
        staging_fd = os.open(
            staging,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=namespace_fd,
        )
        opened = os.fstat(staging_fd)
        located = os.stat(staging, dir_fd=namespace_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (created.st_dev, created.st_ino) != (opened.st_dev, opened.st_ino)
            or (opened.st_dev, opened.st_ino) != (located.st_dev, located.st_ino)
        ):
            _fail("wheelhouse publication staging identity changed while opened")
        _copy_retained_archive(retained, staging_fd)
        _write_new_file(staging_fd, "receipt.v1.json", retained.receipt_bytes)
        os.fsync(staging_fd)
        os.fchmod(staging_fd, 0o555)
        os.fsync(staging_fd)
        os.fsync(namespace_fd)
        _rename_new_only(namespace_fd, staging, digest)
        committed = True
        entry_name = digest
        os.fsync(namespace_fd)
        _validate_published_directory_fd(
            staging_fd,
            expected_receipt_sha256=receipt_sha256,
            expected_archive_sha256=digest,
        )
        named_publication = os.stat(digest, dir_fd=namespace_fd, follow_symlinks=False)
        retained_publication = os.fstat(staging_fd)
        named_namespace = os.stat(namespace, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(named_publication.st_mode)
            or (named_publication.st_dev, named_publication.st_ino)
            != (retained_publication.st_dev, retained_publication.st_ino)
            or not stat.S_ISDIR(named_namespace.st_mode)
            or (named_namespace.st_dev, named_namespace.st_ino) != namespace_identity
            or _stat_identity(publication_root.lstat()) != root_identity
        ):
            _fail("published wheelhouse path no longer names the retained publication")
        succeeded = True
    except FileExistsError as exc:
        collision = exc
    except BaseException as exc:
        failure = exc

    if (
        not succeeded
        and committed
        and staging_fd >= 0
        and entry_name is not None
        and namespace_fd >= 0
    ):
        try:
            located = os.stat(entry_name, dir_fd=namespace_fd, follow_symlinks=False)
            opened = os.fstat(staging_fd)
            if not stat.S_ISDIR(located.st_mode) or (located.st_dev, located.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                _fail("committed wheelhouse rollback path no longer names its directory")
            rollback = f".rollback-{digest}-{os.getpid()}-{secrets.token_hex(16)}"
            _rename_new_only(namespace_fd, entry_name, rollback)
            entry_name = rollback
            os.fsync(namespace_fd)
        except BaseException as exc:
            cleanup_errors.append(exc)
    if not succeeded and staging_fd >= 0 and entry_name is not None and namespace_fd >= 0:
        try:
            _remove_owned_directory_entry(
                namespace_fd,
                entry_name,
                staging_fd,
                expected_names=("receipt.v1.json", "wheelhouse.v1.tar"),
                label=(
                    "committed wheelhouse publication"
                    if committed
                    else "wheelhouse publication staging directory"
                ),
            )
            os.fsync(namespace_fd)
        except BaseException as exc:
            cleanup_errors.append(exc)
    for descriptor, label in (
        (staging_fd, "wheelhouse publication directory descriptor"),
        (namespace_fd, "wheelhouse publication namespace descriptor"),
        (root_fd, "wheelhouse publication root descriptor"),
    ):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as exc:
                exc.add_note(f"while closing {label}")
                cleanup_errors.append(exc)

    if collision is not None:
        wrapped = FileExistsError(f"refusing to overwrite wheelhouse publication {digest}")
        for cleanup_error in cleanup_errors:
            wrapped.add_note(f"cleanup also failed: {cleanup_error!r}")
        raise wrapped from collision
    if failure is not None:
        for cleanup_error in cleanup_errors:
            failure.add_note(f"cleanup also failed: {cleanup_error!r}")
        raise failure
    if not succeeded:
        _fail("wheelhouse publication stopped without a result")
    # The rename is already the atomic commit point.  Linux releases these
    # read-only descriptors even when close reports a late error; do not turn
    # that report into an ambiguous failed call with a committed directory.
    directory = publication_root / namespace / digest
    result = PublishedMatchedV3CpuWheelhouse(
        directory=directory,
        archive=directory / "wheelhouse.v1.tar",
        receipt=directory / "receipt.v1.json",
        archive_sha256=digest,
        receipt_sha256=receipt_sha256,
    )
    return result


def _validate_published_directory_fd(
    directory_fd: int,
    *,
    expected_receipt_sha256: str,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    expected_receipt = _require_sha256(
        expected_receipt_sha256,
        label="published receipt",
    )
    expected_archive = _require_sha256(
        expected_archive_sha256,
        label="published archive",
    )
    receipt_descriptor = -1
    archive_descriptor = -1
    try:
        directory_identity = _stat_identity(os.fstat(directory_fd))
        if stat.S_IMODE(os.fstat(directory_fd).st_mode) != 0o555:
            _fail("published wheelhouse directory mode differs")
        if sorted(os.listdir(directory_fd)) != ["receipt.v1.json", "wheelhouse.v1.tar"]:
            _fail("published wheelhouse file set differs")
        receipt_descriptor, receipt_size, receipt_sha, receipt_identity = _open_hashed_at(
            directory_fd,
            "receipt.v1.json",
            label="published wheelhouse receipt",
            maximum=_MAX_RECEIPT_BYTES,
        )
        if receipt_size <= 0 or receipt_sha != expected_receipt:
            _fail("published wheelhouse receipt identity differs")
        receipt_raw = _pread_exact(
            receipt_descriptor,
            receipt_size,
            0,
            label="published wheelhouse receipt",
        )
        receipt = parse_cpu_wheelhouse_receipt(
            receipt_raw,
            expected_file_sha256=expected_receipt,
        )
        archive = cast(dict[str, Any], receipt["archive"])
        if archive["sha256"] != expected_archive:
            _fail("published receipt archive digest differs from its directory")
        archive_descriptor, archive_size, archive_sha, archive_identity = _open_hashed_at(
            directory_fd,
            "wheelhouse.v1.tar",
            label="published wheelhouse archive",
            maximum=_MAX_ARCHIVE_BYTES,
        )
        if archive_size != archive["size_bytes"] or archive_sha != expected_archive:
            _fail("published wheelhouse archive identity differs")
        _verify_archive_fd(
            archive_descriptor,
            expected_size=archive_size,
            expected_sha256=archive_sha,
            members=cast(list[dict[str, Any]], archive["members"]),
        )
        _reverify_bound_descriptor(
            receipt_descriptor,
            expected_identity=receipt_identity,
            expected_sha256=expected_receipt,
            label="published wheelhouse receipt",
        )
        _reverify_bound_descriptor(
            archive_descriptor,
            expected_identity=archive_identity,
            expected_sha256=expected_archive,
            label="published wheelhouse archive",
        )
        if (
            sorted(os.listdir(directory_fd)) != ["receipt.v1.json", "wheelhouse.v1.tar"]
            or _stat_identity(os.fstat(directory_fd)) != directory_identity
        ):
            _fail("published wheelhouse directory identity changed during validation")
        return receipt
    finally:
        if receipt_descriptor >= 0:
            os.close(receipt_descriptor)
        if archive_descriptor >= 0:
            os.close(archive_descriptor)


def validate_published_matched_v3_cpu_wheelhouse(
    directory: Path,
    *,
    expected_receipt_sha256: str,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    """Reopen and fully replay one exact content-addressed publication."""

    expected_archive = _require_sha256(
        expected_archive_sha256,
        label="published archive",
    )
    if directory.name != expected_archive:
        _fail("published directory is not addressed by the expected archive digest")
    directory_fd = _directory_fd(directory, label="published wheelhouse directory")
    try:
        return _validate_published_directory_fd(
            directory_fd,
            expected_receipt_sha256=expected_receipt_sha256,
            expected_archive_sha256=expected_archive,
        )
    finally:
        os.close(directory_fd)


__all__ = [
    "CPU_WHEELHOUSE_CONTRACT_DESCRIPTOR_SCHEMA_VERSION",
    "CPU_WHEELHOUSE_CONTRACT_DESCRIPTOR_SHA256",
    "CPU_WHEELHOUSE_HELPER_REQUEST_SCHEMA_VERSION",
    "CPU_WHEELHOUSE_HELPER_SOURCE_SHA256",
    "CPU_WHEELHOUSE_RECEIPT_SCHEMA_VERSION",
    "CPU_WHEELHOUSE_STATUS",
    "CPU_WHEEL_CAPTURE_MANIFEST_SCHEMA_VERSION",
    "CPU_WHEEL_VALIDATION_REPORT_SCHEMA_VERSION",
    "ForagerMatchedV3CpuWheelhouseError",
    "PublishedMatchedV3CpuWheelhouse",
    "RetainedMatchedV3CpuWheelhouse",
    "WheelhouseVerifierToolBinding",
    "canonical_cpu_wheelhouse_contract_descriptor_bytes",
    "cpu_wheelhouse_contract_descriptor",
    "cpu_wheelhouse_contract_descriptor_sha256",
    "parse_cpu_wheel_capture_manifest",
    "parse_cpu_wheelhouse_contract_descriptor",
    "parse_cpu_wheelhouse_receipt",
    "publish_matched_v3_cpu_wheelhouse",
    "stage_matched_v3_cpu_wheelhouse",
    "validate_published_matched_v3_cpu_wheelhouse",
]
