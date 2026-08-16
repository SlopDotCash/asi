"""Pure-stdlib source snapshot contract for matched-v3 local execution.

This module is safe to load directly by file path.  Importing it does not import
the Alberta package, JAX, NumPy, Foragax, or a benchmark runner; inspect a
filesystem; create a runtime; or grant execution authority.  Repository roots are
always caller supplied.

The measured inventory contains the exact root files ``pyproject.toml``,
``uv.lock``, and ``FORAGER_BENCHMARK.md`` plus every non-cache regular file below
``alberta_framework/``.  The
walk is anchored by directory descriptors, rejects links and special files, reads
regular files through ``O_NOFOLLOW``, checks pathname/descriptor metadata before
and after reads, and requires two identical complete measurements.  The resulting
manifest is an unqualified, nonauthorizing description only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn, cast

LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_source_snapshot_descriptor.v1"
)
LOCAL_SOURCE_SNAPSHOT_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_source_snapshot_manifest.v1"
)
LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.local_source_snapshot_tree.v1"
)
LOCAL_SOURCE_SNAPSHOT_STATUS: Final = "implemented_unqualified_non_authorizing"

_ROOT_FILE_NAMES: Final = ("pyproject.toml", "uv.lock", "FORAGER_BENCHMARK.md")
_FRAMEWORK_DIRECTORY_NAME: Final = "alberta_framework"
_CACHE_DIRECTORY_NAMES: Final = frozenset(
    {".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
)
_CACHE_FILE_SUFFIXES: Final = (".pyc", ".pyo")

_MAX_FILES: Final = 20_000
_MAX_DIRECTORIES: Final = 10_000
_MAX_ENTRIES: Final = 50_000
_MAX_DEPTH: Final = 64
_MAX_FILE_BYTES: Final = 128 * 1024 * 1024
_MAX_TOTAL_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES: Final = 16 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 250_000
_MAX_JSON_STRING_BYTES: Final = 2_048
_MAX_PATH_BYTES: Final = 1_024
_MAX_COMPONENT_BYTES: Final = 255
_READ_CHUNK_BYTES: Final = 1024 * 1024

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_ASCII_COMPONENT_RE: Final = re.compile(r"[A-Za-z0-9_.+-]{1,255}\Z")


class ForagerMatchedV3LocalSourceSnapshotError(RuntimeError):
    """The snapshot descriptor, manifest, path, or measured tree failed closed."""


def _claims() -> dict[str, bool]:
    return {
        "execution_authority_granted": False,
        "execution_linkage_established": False,
        "executed_bytecode_attested": False,
        "import_behavior_attested": False,
        "publication_authority_granted": False,
        "qualification_granted": False,
        "runtime_qualified": False,
        "source_snapshot_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "A measured manifest describes caller-supplied local source only.",
        "No current-repository manifest or source identity is embedded here.",
        (
            "Excluded cache subtrees and pyc/pyo files are not bound; this snapshot "
            "does not attest executed bytecode or import behavior and cannot establish "
            "execution linkage alone."
        ),
        (
            "A standalone execution bootstrap must separately neutralize bytecode "
            "caches before linking execution to this source snapshot."
        ),
        "A manifest grants no execution, publication, qualification, or promotion authority.",
        "Runtime, dependency, toolchain, and hardware closure remain external.",
    ]


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SCHEMA_VERSION,
        "status": LOCAL_SOURCE_SNAPSHOT_STATUS,
        "classification": "local_source_measurement_contract_non_authorizing",
        "repository_root": {
            "caller_supplied": True,
            "default_path": False,
            "recorded_in_manifest": False,
            "required_form": "exact_absolute_ascii_path_without_dot_segments",
            "ancestor_symlinks_allowed": False,
        },
        "inventory": {
            "root_files": list(_ROOT_FILE_NAMES),
            "recursive_directory": _FRAMEWORK_DIRECTORY_NAME,
            "included_node_type": "single_link_regular_file",
            "included_directory_type": "non_cache_directory",
            "cache_directory_names": sorted(_CACHE_DIRECTORY_NAMES),
            "cache_file_suffixes": list(_CACHE_FILE_SUFFIXES),
            "cache_directories_are_uninspected_excluded_subtrees": True,
            "all_other_directory_entries_must_be_regular_files_or_directories": True,
            "symlinks_allowed": False,
            "hardlinks_allowed": False,
            "special_files_allowed": False,
        },
        "path_identity": {
            "encoding": "printable_ascii",
            "component_pattern": "[A-Za-z0-9_.+-]{1,255}",
            "separator": "/",
            "dot_segments_allowed": False,
            "backslash_allowed": False,
            "duplicate_paths_allowed": False,
            "alias_key": "ascii_casefold",
            "casefold_aliases_allowed": False,
            "unicode_allowed": False,
        },
        "measurement": {
            "passes": 2,
            "identical_passes_required": True,
            "directory_descriptor_anchored": True,
            "open_relative_to_parent_descriptor": True,
            "nofollow_required": True,
            "pre_open_stat_open_fstat_post_read_fstat_restat_required": True,
            "stable_single_link_required_before_open_and_after_read": True,
            "directory_entry_set_stability_required": True,
            "root_locator_stability_required": True,
        },
        "limits": {
            "maximum_files": _MAX_FILES,
            "maximum_directories": _MAX_DIRECTORIES,
            "maximum_entries": _MAX_ENTRIES,
            "maximum_depth": _MAX_DEPTH,
            "maximum_file_bytes": _MAX_FILE_BYTES,
            "maximum_total_bytes": _MAX_TOTAL_BYTES,
            "maximum_manifest_bytes": _MAX_MANIFEST_BYTES,
            "maximum_path_bytes": _MAX_PATH_BYTES,
            "maximum_component_bytes": _MAX_COMPONENT_BYTES,
        },
        "file_record": {
            "fields": ["path", "size_bytes", "sha256"],
            "order": "ascending_ascii_path_bytes",
            "content_digest": "sha256",
        },
        "directory_record": {
            "representation": "relative_path_string",
            "order": "ascending_ascii_path_bytes",
        },
        "tree_identity": {
            "schema_version": LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION,
            "canonical_payload_fields": ["schema_version", "directories", "files"],
            "digest": "sha256_of_canonical_payload_bytes",
        },
        "manifest_identity": {
            "body_digest_field": "manifest_body_sha256",
            "body_digest": "sha256_of_canonical_manifest_without_body_digest",
            "full_digest": "caller_carried_sha256_of_canonical_manifest_bytes",
            "canonical_encoding": "ascii_sorted_keys_compact_one_trailing_newline",
        },
        "verification": {
            "expected_canonical_manifest_bytes_required": True,
            "expected_full_sha256_required": True,
            "expected_artifact_validated_before_filesystem_observation": True,
            "fresh_two_pass_measurement_required": True,
            "exact_canonical_bytes_match_required": True,
        },
        "runner_relationship": {
            "current_runner_identity_embedded": False,
            "runner_imported": False,
            "runner_bootstrap_performed": False,
            "runner_capability_requested": False,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


def _raise_json_constant(value: str) -> NoReturn:
    raise ForagerMatchedV3LocalSourceSnapshotError(
        f"source snapshot JSON contains non-finite constant {value!r}"
    )


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > 20:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot JSON integer exceeds its lexical bound"
        )
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                f"source snapshot JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: Any) -> None:
    seen: set[int] = set()
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source snapshot JSON exceeds its node bound"
            )
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source snapshot JSON exceeds its depth bound"
            )
        if type(item) is str:
            try:
                encoded = item.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ForagerMatchedV3LocalSourceSnapshotError(
                    "source snapshot JSON strings must be ASCII"
                ) from exc
            if len(encoded) > _MAX_JSON_STRING_BYTES or any(
                byte < 0x20 or byte > 0x7E for byte in encoded
            ):
                raise ForagerMatchedV3LocalSourceSnapshotError(
                    "source snapshot JSON strings must be bounded printable ASCII"
                )
            return
        if item is None or type(item) in {bool, int}:
            return
        if type(item) not in {dict, list}:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source snapshot JSON contains a non-plain value"
            )
        identity = id(item)
        if identity in seen:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source snapshot JSON contains a container alias"
            )
        seen.add(identity)
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
        else:
            for key, child in cast(dict[Any, Any], item).items():
                if type(key) is not str:
                    raise ForagerMatchedV3LocalSourceSnapshotError(
                        "source snapshot JSON object keys must be exact strings"
                    )
                visit(key, depth + 1)
                visit(child, depth + 1)

    visit(value, 0)


def _canonical_json(value: dict[str, Any]) -> bytes:
    if type(value) is not dict:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot canonical root must be a plain object"
        )
    _assert_plain_unaliased_json(value)
    try:
        raw = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot value is not canonical finite ASCII JSON"
        ) from exc
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot artifact exceeds its byte bound"
        )
    return raw


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_MANIFEST_BYTES:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot artifact must be bounded exact bytes"
        )
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot artifact must have one trailing newline"
        )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot artifact must be ASCII"
        ) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_json_constant,
            parse_int=_parse_bounded_int,
        )
    except ForagerMatchedV3LocalSourceSnapshotError:
        raise
    except (RecursionError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot artifact is not strict JSON"
        ) from exc
    if type(value) is not dict:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot artifact root must be a plain object"
        )
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if not hmac.compare_digest(_canonical_json(result), raw):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot artifact is not exactly canonical"
        )
    return result


def _exact_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if left is None or type(left) in {bool, int, str}:
        return bool(left == right)
    if type(left) is list:
        exact_left = left
        exact_right = cast(list[Any], right)
        return len(exact_left) == len(exact_right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(exact_left, exact_right, strict=True)
        )
    if type(left) is dict:
        exact_left_map = cast(dict[str, Any], left)
        exact_right_map = cast(dict[str, Any], right)
        return exact_left_map.keys() == exact_right_map.keys() and all(
            _exact_json_equal(exact_left_map[key], exact_right_map[key]) for key in exact_left_map
        )
    return False


def _require_exact_keys(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != expected:
        raise ForagerMatchedV3LocalSourceSnapshotError(f"{label} keys are not exact")
    return cast(dict[str, Any], value)


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            f"{label} must be one nonzero lowercase SHA-256"
        )
    return value


def _require_bounded_int(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            f"{label} must be an exact integer in [{minimum}, {maximum}]"
        )
    return value


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256: Final = (
    "5ba69445a00dfc0bc36a4d05dafcc534b291430d491c3f71560570d7eb862899"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(),
    LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256,
):
    raise AssertionError("matched-v3 local source snapshot descriptor identity drifted")


def _validate_component(component: Any, label: str) -> str:
    if (
        type(component) is not str
        or component in {"", ".", ".."}
        or "/" in component
        or "\\" in component
        or "\x00" in component
        or _ASCII_COMPONENT_RE.fullmatch(component) is None
    ):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            f"{label} is not one unambiguous ASCII path component"
        )
    if len(component.encode("ascii")) > _MAX_COMPONENT_BYTES:
        raise ForagerMatchedV3LocalSourceSnapshotError(f"{label} exceeds the component byte bound")
    return component


def _validate_relative_path(path: Any, label: str) -> str:
    if type(path) is not str:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            f"{label} must be one exact relative path string"
        )
    try:
        encoded = path.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3LocalSourceSnapshotError(f"{label} must be ASCII") from exc
    if not encoded or len(encoded) > _MAX_PATH_BYTES or path.startswith("/"):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            f"{label} is outside its relative path bounds"
        )
    components = path.split("/")
    if not components or any(not component for component in components):
        raise ForagerMatchedV3LocalSourceSnapshotError(f"{label} contains an empty path component")
    for component in components:
        _validate_component(component, label)
    if path in _ROOT_FILE_NAMES:
        return path
    if len(components) < 2 or components[0] != _FRAMEWORK_DIRECTORY_NAME:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            f"{label} is outside the frozen inventory roots"
        )
    if len(components) - 1 > _MAX_DEPTH:
        raise ForagerMatchedV3LocalSourceSnapshotError(f"{label} exceeds the global depth bound")
    if any(component in _CACHE_DIRECTORY_NAMES for component in components[:-1]) or path.endswith(
        _CACHE_FILE_SUFFIXES
    ):
        raise ForagerMatchedV3LocalSourceSnapshotError(f"{label} names excluded cache material")
    return path


def _validate_relative_directory_path(path: Any, label: str) -> str:
    if type(path) is not str:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            f"{label} must be one exact relative directory path string"
        )
    try:
        encoded = path.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3LocalSourceSnapshotError(f"{label} must be ASCII") from exc
    if not encoded or len(encoded) > _MAX_PATH_BYTES or path.startswith("/"):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            f"{label} is outside its relative path bounds"
        )
    components = path.split("/")
    if not components or any(not component for component in components):
        raise ForagerMatchedV3LocalSourceSnapshotError(f"{label} contains an empty path component")
    for component in components:
        _validate_component(component, label)
    if components[0] != _FRAMEWORK_DIRECTORY_NAME:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            f"{label} is outside the frozen recursive directory"
        )
    if len(components) > _MAX_DEPTH:
        raise ForagerMatchedV3LocalSourceSnapshotError(f"{label} exceeds the global depth bound")
    if any(component in _CACHE_DIRECTORY_NAMES for component in components):
        raise ForagerMatchedV3LocalSourceSnapshotError(f"{label} names an excluded cache directory")
    return path


def _path_alias_key(path: str) -> str:
    return path.casefold()


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


def _locator_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode))


def _close_untransferred_descriptor(
    descriptor: int,
    *,
    failure: BaseException | None,
    label: str,
) -> None:
    if descriptor < 0:
        return
    try:
        os.close(descriptor)
    except BaseException as cleanup_error:
        if failure is not None:
            failure.add_note(f"{label} cleanup close also failed: {cleanup_error!r}")
            return
        raise


def _directory_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot traversal requires O_DIRECTORY and O_NOFOLLOW"
        )
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot traversal requires O_NOFOLLOW"
        )
    return os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)


@dataclass(slots=True)
class _AnchoredRoot:
    descriptors: list[int]
    components: list[str]
    metadata: list[os.stat_result]

    @property
    def root_descriptor(self) -> int:
        return self.descriptors[-1]

    def verify(self) -> None:
        for index in range(1, len(self.descriptors)):
            parent = self.descriptors[index - 1]
            descriptor = self.descriptors[index]
            component = self.components[index]
            expected = self.metadata[index]
            try:
                current = os.stat(component, dir_fd=parent, follow_symlinks=False)
                opened = os.fstat(descriptor)
            except OSError as exc:
                raise ForagerMatchedV3LocalSourceSnapshotError(
                    "repository root locator changed during source observation"
                ) from exc
            if _locator_identity(current) != _locator_identity(expected) or _locator_identity(
                opened
            ) != _locator_identity(expected):
                raise ForagerMatchedV3LocalSourceSnapshotError(
                    "repository root locator changed during source observation"
                )

    def close(self) -> None:
        descriptors = self.descriptors
        self.descriptors = []
        failure: BaseException | None = None
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                if failure is None:
                    failure = cleanup_error
                    failure.add_note("while closing an anchored repository descriptor")
                else:
                    failure.add_note(
                        "another anchored repository descriptor close also failed: "
                        f"{cleanup_error!r}"
                    )
        if failure is not None:
            raise failure


def _open_anchored_repository_root(repository_root: Any) -> _AnchoredRoot:
    concrete_path_type = type(Path())
    if type(repository_root) is not concrete_path_type:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "repository_root must be one exact concrete pathlib.Path"
        )
    root = repository_root
    if not root.is_absolute() or root.anchor != os.sep or root == Path(root.anchor):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "repository_root must be a non-root absolute path"
        )
    if os.path.abspath(str(root)) != str(root):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "repository_root must not contain path aliases or traversal"
        )
    raw_components = list(root.parts)
    if not raw_components or raw_components[0] != root.anchor:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "repository_root has no exact absolute anchor"
        )
    components = [root.anchor]
    for index, component in enumerate(raw_components[1:]):
        components.append(_validate_component(component, f"repository_root component {index}"))

    flags = _directory_flags()
    descriptors: list[int] = []
    metadata: list[os.stat_result] = []
    pending_descriptor = -1
    try:
        pending_descriptor = os.open(root.anchor, flags)
        anchor_descriptor = pending_descriptor
        descriptors.append(anchor_descriptor)
        pending_descriptor = -1
        anchor_metadata = os.fstat(anchor_descriptor)
        if not stat.S_ISDIR(anchor_metadata.st_mode):
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "repository_root anchor is not a directory"
            )
        metadata.append(anchor_metadata)
        for component in components[1:]:
            parent = descriptors[-1]
            try:
                before = os.stat(component, dir_fd=parent, follow_symlinks=False)
            except OSError as exc:
                raise ForagerMatchedV3LocalSourceSnapshotError(
                    "cannot inspect repository_root without following links"
                ) from exc
            if not stat.S_ISDIR(before.st_mode):
                raise ForagerMatchedV3LocalSourceSnapshotError(
                    "repository_root contains a symlink or non-directory component"
                )
            try:
                pending_descriptor = os.open(component, flags, dir_fd=parent)
                child = pending_descriptor
            except OSError as exc:
                raise ForagerMatchedV3LocalSourceSnapshotError(
                    "cannot open repository_root without following links"
                ) from exc
            descriptors.append(child)
            pending_descriptor = -1
            opened = os.fstat(child)
            try:
                current = os.stat(component, dir_fd=parent, follow_symlinks=False)
            except OSError as exc:
                raise ForagerMatchedV3LocalSourceSnapshotError(
                    "repository_root changed while being opened"
                ) from exc
            if _stat_identity(before) != _stat_identity(opened) or _stat_identity(
                opened
            ) != _stat_identity(current):
                raise ForagerMatchedV3LocalSourceSnapshotError(
                    "repository_root changed while being opened"
                )
            metadata.append(opened)
    except BaseException as failure:
        if pending_descriptor >= 0:
            _close_untransferred_descriptor(
                pending_descriptor,
                failure=failure,
                label="unrecorded repository root descriptor",
            )
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                failure.add_note(
                    f"repository root descriptor cleanup also failed: {cleanup_error!r}"
                )
        raise
    anchored = _AnchoredRoot(descriptors, components, metadata)
    try:
        anchored.verify()
    except BaseException as failure:
        try:
            anchored.close()
        except BaseException as cleanup_error:
            failure.add_note(f"anchored repository cleanup also failed: {cleanup_error!r}")
        raise
    return anchored


@dataclass(slots=True)
class _WalkState:
    files: int = 0
    directories: int = 0
    entries: int = 0
    total_bytes: int = 0

    def observe_entries(self, count: int) -> None:
        self.entries += count
        if self.entries > _MAX_ENTRIES:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source tree exceeds the global entry bound"
            )

    def observe_directory(self, depth: int) -> None:
        self.directories += 1
        if self.directories > _MAX_DIRECTORIES:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source tree exceeds the global directory bound"
            )
        if depth > _MAX_DEPTH:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source tree exceeds the global depth bound"
            )

    def observe_file(self, size: int) -> None:
        if size < 0 or size > _MAX_FILE_BYTES:
            raise ForagerMatchedV3LocalSourceSnapshotError("source file exceeds its byte bound")
        self.files += 1
        self.total_bytes += size
        if self.files > _MAX_FILES:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source tree exceeds the global file bound"
            )
        if self.total_bytes > _MAX_TOTAL_BYTES:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source tree exceeds the global byte bound"
            )


def _safe_sorted_names(
    directory_descriptor: int,
    *,
    maximum_entries: int,
) -> list[str]:
    if type(maximum_entries) is not int or not 0 <= maximum_entries <= _MAX_ENTRIES:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source directory enumeration received an invalid entry budget"
        )
    result: list[str] = []
    aliases: set[str] = set()
    try:
        with os.scandir(directory_descriptor) as entries:
            for index, entry in enumerate(entries):
                if index >= maximum_entries:
                    raise ForagerMatchedV3LocalSourceSnapshotError(
                        "source directory exceeds the global entry bound"
                    )
                exact = _validate_component(entry.name, f"source entry {index}")
                alias = exact.casefold()
                if alias in aliases:
                    raise ForagerMatchedV3LocalSourceSnapshotError(
                        "source directory contains duplicate or casefold-aliased names"
                    )
                aliases.add(alias)
                result.append(exact)
    except ForagerMatchedV3LocalSourceSnapshotError:
        raise
    except OSError as exc:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "cannot enumerate an anchored source directory"
        ) from exc
    return sorted(result, key=lambda item: item.encode("ascii"))


def _open_checked_child(
    parent_descriptor: int,
    name: str,
    before: os.stat_result,
    *,
    directory: bool,
) -> tuple[int, os.stat_result]:
    flags = _directory_flags() if directory else _file_flags()
    descriptor = -1
    failure: BaseException | None = None
    try:
        try:
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "cannot safely open an anchored source entry"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source entry changed while being opened"
            ) from exc
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if (
            not expected_type(opened.st_mode)
            or _stat_identity(before) != _stat_identity(opened)
            or _stat_identity(opened) != _stat_identity(current)
        ):
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source entry changed while being opened"
            )
        result = (descriptor, opened)
        descriptor = -1
        return result
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if descriptor >= 0:
            _close_untransferred_descriptor(
                descriptor,
                failure=failure,
                label="source entry descriptor",
            )


def _verify_opened_child(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    expected: os.stat_result,
) -> None:
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source entry changed during observation"
        ) from exc
    if _stat_identity(opened) != _stat_identity(expected) or _stat_identity(
        current
    ) != _stat_identity(expected):
        raise ForagerMatchedV3LocalSourceSnapshotError("source entry changed during observation")


def _read_regular_file(
    parent_descriptor: int,
    name: str,
    relative_path: str,
    before: os.stat_result,
    state: _WalkState,
    *,
    include: bool,
) -> dict[str, Any] | None:
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source inventory contains a hardlink, symlink, or special file"
        )
    state.observe_file(before.st_size)
    descriptor, opened = _open_checked_child(
        parent_descriptor,
        name,
        before,
        directory=False,
    )
    failure: BaseException | None = None
    try:
        if opened.st_nlink != 1:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source inventory contains a multiply linked file"
            )
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            try:
                chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
            except OSError as exc:
                raise ForagerMatchedV3LocalSourceSnapshotError(
                    "source file could not be read exactly"
                ) from exc
            if not chunk:
                raise ForagerMatchedV3LocalSourceSnapshotError("source file ended while being read")
            digest.update(chunk)
            remaining -= len(chunk)
        try:
            if os.read(descriptor, 1):
                raise ForagerMatchedV3LocalSourceSnapshotError("source file grew while being read")
        except OSError as exc:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source file could not be checked after reading"
            ) from exc
        _verify_opened_child(parent_descriptor, name, descriptor, opened)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        closing_descriptor = descriptor
        descriptor = -1
        _close_untransferred_descriptor(
            closing_descriptor,
            failure=failure,
            label="measured source file descriptor",
        )
    if not include:
        return None
    exact_path = _validate_relative_path(relative_path, "measured source path")
    return {
        "path": exact_path,
        "size_bytes": opened.st_size,
        "sha256": digest.hexdigest(),
    }


def _walk_framework_directory(
    directory_descriptor: int,
    prefix: str,
    depth: int,
    opened_metadata: os.stat_result,
    state: _WalkState,
    directories: list[str],
    records: list[dict[str, Any]],
) -> None:
    state.observe_directory(depth)
    directories.append(_validate_relative_directory_path(prefix, "measured source directory"))
    names_before = _safe_sorted_names(
        directory_descriptor,
        maximum_entries=_MAX_ENTRIES - state.entries,
    )
    state.observe_entries(len(names_before))
    for name in names_before:
        relative = f"{prefix}/{name}"
        try:
            before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "cannot inspect an anchored source entry"
            ) from exc
        if stat.S_ISLNK(before.st_mode):
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source inventory contains a symbolic link"
            )
        if stat.S_ISDIR(before.st_mode):
            child, child_opened = _open_checked_child(
                directory_descriptor,
                name,
                before,
                directory=True,
            )
            child_failure: BaseException | None = None
            try:
                if name not in _CACHE_DIRECTORY_NAMES:
                    _walk_framework_directory(
                        child,
                        relative,
                        depth + 1,
                        child_opened,
                        state,
                        directories,
                        records,
                    )
                _verify_opened_child(
                    directory_descriptor,
                    name,
                    child,
                    child_opened,
                )
            except BaseException as exc:
                child_failure = exc
                raise
            finally:
                closing_child = child
                child = -1
                _close_untransferred_descriptor(
                    closing_child,
                    failure=child_failure,
                    label="measured source directory descriptor",
                )
            continue
        if not stat.S_ISREG(before.st_mode):
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source inventory contains a nonregular filesystem entry"
            )
        include = not name.endswith(_CACHE_FILE_SUFFIXES)
        record = _read_regular_file(
            directory_descriptor,
            name,
            relative,
            before,
            state,
            include=include,
        )
        if record is not None:
            records.append(record)
    names_after = _safe_sorted_names(
        directory_descriptor,
        maximum_entries=len(names_before),
    )
    try:
        after = os.fstat(directory_descriptor)
    except OSError as exc:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source directory changed during observation"
        ) from exc
    if names_before != names_after or _stat_identity(after) != _stat_identity(opened_metadata):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source directory changed during observation"
        )


def _measure_once(repository_descriptor: int) -> dict[str, Any]:
    state = _WalkState()
    directories: list[str] = []
    records: list[dict[str, Any]] = []
    aliases: set[str] = set()

    for root_name in _ROOT_FILE_NAMES:
        state.observe_entries(1)
        try:
            before = os.stat(
                root_name,
                dir_fd=repository_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                f"required root source file is unavailable: {root_name}"
            ) from exc
        record = _read_regular_file(
            repository_descriptor,
            root_name,
            root_name,
            before,
            state,
            include=True,
        )
        if record is None:  # pragma: no cover - include is statically true
            raise AssertionError("required root file was unexpectedly excluded")
        records.append(record)

    state.observe_entries(1)
    try:
        framework_before = os.stat(
            _FRAMEWORK_DIRECTORY_NAME,
            dir_fd=repository_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "required alberta_framework source directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(framework_before.st_mode):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "alberta_framework source root must be one non-symlink directory"
        )
    framework_descriptor, framework_opened = _open_checked_child(
        repository_descriptor,
        _FRAMEWORK_DIRECTORY_NAME,
        framework_before,
        directory=True,
    )
    failure: BaseException | None = None
    try:
        _walk_framework_directory(
            framework_descriptor,
            _FRAMEWORK_DIRECTORY_NAME,
            1,
            framework_opened,
            state,
            directories,
            records,
        )
        _verify_opened_child(
            repository_descriptor,
            _FRAMEWORK_DIRECTORY_NAME,
            framework_descriptor,
            framework_opened,
        )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        closing_framework = framework_descriptor
        framework_descriptor = -1
        _close_untransferred_descriptor(
            closing_framework,
            failure=failure,
            label="framework source root descriptor",
        )

    records.sort(key=lambda record: cast(str, record["path"]).encode("ascii"))
    directories.sort(key=lambda path: path.encode("ascii"))
    for record in records:
        path = cast(str, record["path"])
        alias = _path_alias_key(path)
        if alias in aliases:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source inventory contains duplicate or aliased relative paths"
            )
        aliases.add(alias)
    directory_aliases: set[str] = set()
    for directory in directories:
        alias = _path_alias_key(directory)
        if alias in directory_aliases:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source inventory contains duplicate or aliased directory paths"
            )
        directory_aliases.add(alias)
    included_bytes = sum(cast(int, record["size_bytes"]) for record in records)
    if len(records) > _MAX_FILES or included_bytes > _MAX_TOTAL_BYTES:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "included source inventory exceeds its global bounds"
        )
    return {"directories": directories, "files": records}


def _manifest_from_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    directories = list(cast(list[str], inventory["directories"]))
    detached_records = [dict(record) for record in cast(list[dict[str, Any]], inventory["files"])]
    tree_payload = {
        "schema_version": LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION,
        "directories": directories,
        "files": detached_records,
    }
    tree_sha256 = hashlib.sha256(_canonical_json(tree_payload)).hexdigest()
    total_size = sum(cast(int, record["size_bytes"]) for record in detached_records)
    body: dict[str, Any] = {
        "schema_version": LOCAL_SOURCE_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "status": "measured_unqualified_non_authorizing",
        "classification": "local_source_snapshot_content_only",
        "descriptor_binding": {
            "schema_version": LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256,
        },
        "observation": {
            "measurement_passes": 2,
            "identical_passes_required": True,
            "repository_path_recorded": False,
        },
        "inventory": {
            "directory_count": len(directories),
            "file_count": len(detached_records),
            "total_size_bytes": total_size,
        },
        "directories": directories,
        "files": detached_records,
        "tree": {
            "schema_version": LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION,
            "sha256": tree_sha256,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }
    return {
        **body,
        "manifest_body_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }


def _measure_repository(repository_root: Any) -> dict[str, Any]:
    anchored = _open_anchored_repository_root(repository_root)
    failure: BaseException | None = None
    try:
        try:
            root_before = os.fstat(anchored.root_descriptor)
            first = _measure_once(anchored.root_descriptor)
            anchored.verify()
            second = _measure_once(anchored.root_descriptor)
            anchored.verify()
            root_after = os.fstat(anchored.root_descriptor)
            if _stat_identity(root_before) != _stat_identity(root_after):
                raise ForagerMatchedV3LocalSourceSnapshotError(
                    "repository root changed during source observation"
                )
            if not _exact_json_equal(first, second):
                raise ForagerMatchedV3LocalSourceSnapshotError(
                    "two complete source measurements disagreed"
                )
            return _manifest_from_inventory(second)
        except OSError as exc:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "repository root could not be checked after source observation"
            ) from exc
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            anchored.close()
        except BaseException as cleanup_error:
            if failure is not None:
                failure.add_note(f"anchored repository cleanup also failed: {cleanup_error!r}")
            else:
                raise


def _validate_manifest(value: dict[str, Any]) -> None:
    manifest = _require_exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "descriptor_binding",
                "observation",
                "inventory",
                "directories",
                "files",
                "tree",
                "claims",
                "limitations",
                "manifest_body_sha256",
            }
        ),
        "source snapshot manifest",
    )
    if (
        manifest["schema_version"] != LOCAL_SOURCE_SNAPSHOT_MANIFEST_SCHEMA_VERSION
        or manifest["status"] != "measured_unqualified_non_authorizing"
        or manifest["classification"] != "local_source_snapshot_content_only"
    ):
        raise ForagerMatchedV3LocalSourceSnapshotError("source snapshot manifest identity drifted")
    if not _exact_json_equal(
        manifest["descriptor_binding"],
        {
            "schema_version": LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256,
        },
    ):
        raise ForagerMatchedV3LocalSourceSnapshotError("source snapshot descriptor binding drifted")
    if not _exact_json_equal(
        manifest["observation"],
        {
            "measurement_passes": 2,
            "identical_passes_required": True,
            "repository_path_recorded": False,
        },
    ):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot observation contract drifted"
        )

    directories = manifest["directories"]
    if type(directories) is not list or not 1 <= len(directories) <= _MAX_DIRECTORIES:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot directories must be one bounded array"
        )
    exact_directories = directories
    directory_paths: list[str] = []
    directory_aliases: set[str] = set()
    for index, item in enumerate(exact_directories):
        path = _validate_relative_directory_path(item, f"source snapshot directory {index}")
        alias = _path_alias_key(path)
        if alias in directory_aliases:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source snapshot contains duplicate or aliased directory paths"
            )
        directory_aliases.add(alias)
        directory_paths.append(path)
    if directory_paths != sorted(directory_paths, key=lambda path: path.encode("ascii")):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot directory records are not in exact ASCII order"
        )
    if _FRAMEWORK_DIRECTORY_NAME not in directory_paths:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot omits the required framework directory"
        )
    directory_path_set = set(directory_paths)
    for path in directory_paths:
        if path == _FRAMEWORK_DIRECTORY_NAME:
            continue
        parent = path.rsplit("/", 1)[0]
        if parent not in directory_path_set:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source snapshot directory has no bound parent directory"
            )

    files = manifest["files"]
    if type(files) is not list or not 2 <= len(files) <= _MAX_FILES:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot files must be one bounded array"
        )
    exact_files = files
    paths: list[str] = []
    aliases: set[str] = set()
    total_size = 0
    for index, item in enumerate(exact_files):
        record = _require_exact_keys(
            item,
            frozenset({"path", "size_bytes", "sha256"}),
            f"source snapshot file {index}",
        )
        path = _validate_relative_path(record["path"], f"source snapshot file {index} path")
        alias = _path_alias_key(path)
        if alias in aliases:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source snapshot contains duplicate or aliased paths"
            )
        aliases.add(alias)
        size = _require_bounded_int(
            record["size_bytes"],
            f"source snapshot file {index} size",
            minimum=0,
            maximum=_MAX_FILE_BYTES,
        )
        _require_sha256(record["sha256"], f"source snapshot file {index}")
        total_size += size
        if total_size > _MAX_TOTAL_BYTES:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source snapshot file bytes exceed the global bound"
            )
        paths.append(path)
    if paths != sorted(paths, key=lambda path: path.encode("ascii")):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot file records are not in exact ASCII order"
        )
    if not all(root_file in paths for root_file in _ROOT_FILE_NAMES):
        raise ForagerMatchedV3LocalSourceSnapshotError("source snapshot omits a required root file")
    required_initializer = f"{_FRAMEWORK_DIRECTORY_NAME}/__init__.py"
    if required_initializer not in paths:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot omits the required alberta_framework initializer"
        )
    for path in paths:
        if path in _ROOT_FILE_NAMES:
            continue
        parent = path.rsplit("/", 1)[0]
        if parent not in directory_path_set:
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source snapshot file has no bound parent directory"
            )
    if aliases & directory_aliases:
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot path cannot identify both a file and a directory"
        )

    inventory = _require_exact_keys(
        manifest["inventory"],
        frozenset({"directory_count", "file_count", "total_size_bytes"}),
        "source snapshot inventory",
    )
    directory_count = _require_bounded_int(
        inventory["directory_count"],
        "source snapshot directory count",
        minimum=1,
        maximum=_MAX_DIRECTORIES,
    )
    file_count = _require_bounded_int(
        inventory["file_count"],
        "source snapshot file count",
        minimum=2,
        maximum=_MAX_FILES,
    )
    recorded_total = _require_bounded_int(
        inventory["total_size_bytes"],
        "source snapshot total bytes",
        minimum=0,
        maximum=_MAX_TOTAL_BYTES,
    )
    if (
        directory_count != len(exact_directories)
        or file_count != len(exact_files)
        or recorded_total != total_size
    ):
        raise ForagerMatchedV3LocalSourceSnapshotError("source snapshot inventory totals drifted")

    tree = _require_exact_keys(
        manifest["tree"],
        frozenset({"schema_version", "sha256"}),
        "source snapshot tree",
    )
    tree_sha256 = _require_sha256(tree["sha256"], "source snapshot tree")
    expected_tree = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION,
                "directories": exact_directories,
                "files": exact_files,
            }
        )
    ).hexdigest()
    if tree[
        "schema_version"
    ] != LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION or not hmac.compare_digest(
        tree_sha256, expected_tree
    ):
        raise ForagerMatchedV3LocalSourceSnapshotError("source snapshot tree identity drifted")
    claims = _require_exact_keys(
        manifest["claims"],
        frozenset(_claims()),
        "source snapshot claims",
    )
    if not _exact_json_equal(claims, _claims()) or any(
        claim is not False for claim in claims.values()
    ):
        raise ForagerMatchedV3LocalSourceSnapshotError("source snapshot claim became true")
    if not _exact_json_equal(manifest["limitations"], _limitations()):
        raise ForagerMatchedV3LocalSourceSnapshotError("source snapshot limitations drifted")
    body = dict(manifest)
    supplied_body_sha256 = _require_sha256(
        body.pop("manifest_body_sha256"), "source snapshot manifest body"
    )
    expected_body_sha256 = hashlib.sha256(_canonical_json(body)).hexdigest()
    if not hmac.compare_digest(supplied_body_sha256, expected_body_sha256):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot manifest body digest drifted"
        )
    _assert_plain_unaliased_json(manifest)
    _canonical_json(manifest)


@dataclass(frozen=True, slots=True)
class MatchedV3LocalSourceSnapshot:
    """Canonical nonauthorizing source manifest and its external identities."""

    canonical_manifest_bytes: bytes
    full_sha256: str
    tree_sha256: str
    directory_count: int
    file_count: int
    total_size_bytes: int

    def __post_init__(self) -> None:
        full_sha256 = _require_sha256(self.full_sha256, "source snapshot full manifest")
        tree_sha256 = _require_sha256(self.tree_sha256, "source snapshot tree")
        _require_bounded_int(
            self.directory_count,
            "source snapshot result directory count",
            minimum=1,
            maximum=_MAX_DIRECTORIES,
        )
        _require_bounded_int(
            self.file_count,
            "source snapshot result file count",
            minimum=2,
            maximum=_MAX_FILES,
        )
        _require_bounded_int(
            self.total_size_bytes,
            "source snapshot result total bytes",
            minimum=0,
            maximum=_MAX_TOTAL_BYTES,
        )
        manifest = parse_matched_v3_local_source_snapshot_manifest(
            self.canonical_manifest_bytes,
            expected_full_sha256=full_sha256,
        )
        if (
            not hmac.compare_digest(cast(str, manifest["tree"]["sha256"]), tree_sha256)
            or manifest["inventory"]["directory_count"] != self.directory_count
            or manifest["inventory"]["file_count"] != self.file_count
            or manifest["inventory"]["total_size_bytes"] != self.total_size_bytes
        ):
            raise ForagerMatchedV3LocalSourceSnapshotError(
                "source snapshot result identity drifted"
            )

    def manifest(self) -> dict[str, Any]:
        """Return detached structural content; this grants no authority."""

        return parse_matched_v3_local_source_snapshot_manifest(
            self.canonical_manifest_bytes,
            expected_full_sha256=self.full_sha256,
        )


def matched_v3_local_source_snapshot_descriptor() -> dict[str, Any]:
    """Return a detached snapshot of the frozen measurement descriptor."""

    return _strict_json_load(_DESCRIPTOR_BYTES)


def canonical_matched_v3_local_source_snapshot_descriptor_bytes() -> bytes:
    """Return exact canonical descriptor bytes."""

    return _DESCRIPTOR_BYTES


def matched_v3_local_source_snapshot_descriptor_sha256() -> str:
    """Return the exact descriptor SHA-256."""

    return LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256


def parse_matched_v3_local_source_snapshot_descriptor(raw: bytes) -> dict[str, Any]:
    """Parse only the exact frozen measurement descriptor."""

    value = _strict_json_load(raw)
    if not _exact_json_equal(value, _descriptor()) or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256
    ):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot descriptor differs from its frozen identity"
        )
    return value


def parse_matched_v3_local_source_snapshot_manifest(
    raw: bytes,
    *,
    expected_full_sha256: str,
) -> dict[str, Any]:
    """Parse one manifest only with its caller-carried exact full-file digest."""

    exact_expected = _require_sha256(expected_full_sha256, "expected source snapshot full manifest")
    if type(raw) is not bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), exact_expected
    ):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "source snapshot manifest full-file digest disagrees"
        )
    value = _strict_json_load(raw)
    _validate_manifest(value)
    return value


def measure_matched_v3_local_source_snapshot(
    *,
    repository_root: Path,
) -> MatchedV3LocalSourceSnapshot:
    """Measure one caller-supplied source tree twice through anchored descriptors."""

    manifest = _measure_repository(repository_root)
    raw = _canonical_json(manifest)
    full_sha256 = hashlib.sha256(raw).hexdigest()
    return MatchedV3LocalSourceSnapshot(
        canonical_manifest_bytes=raw,
        full_sha256=full_sha256,
        tree_sha256=cast(str, manifest["tree"]["sha256"]),
        directory_count=cast(int, manifest["inventory"]["directory_count"]),
        file_count=cast(int, manifest["inventory"]["file_count"]),
        total_size_bytes=cast(int, manifest["inventory"]["total_size_bytes"]),
    )


def verify_matched_v3_local_source_snapshot(
    *,
    repository_root: Path,
    expected_canonical_manifest_bytes: bytes,
    expected_full_sha256: str,
) -> MatchedV3LocalSourceSnapshot:
    """Validate caller-carried expectations, then compare one fresh measurement."""

    parse_matched_v3_local_source_snapshot_manifest(
        expected_canonical_manifest_bytes,
        expected_full_sha256=expected_full_sha256,
    )
    observed = measure_matched_v3_local_source_snapshot(repository_root=repository_root)
    if not hmac.compare_digest(
        observed.full_sha256, expected_full_sha256
    ) or not hmac.compare_digest(
        observed.canonical_manifest_bytes,
        expected_canonical_manifest_bytes,
    ):
        raise ForagerMatchedV3LocalSourceSnapshotError(
            "fresh local source measurement differs from caller-carried expectations"
        )
    return observed


__all__ = [
    "ForagerMatchedV3LocalSourceSnapshotError",
    "LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SCHEMA_VERSION",
    "LOCAL_SOURCE_SNAPSHOT_DESCRIPTOR_SHA256",
    "LOCAL_SOURCE_SNAPSHOT_MANIFEST_SCHEMA_VERSION",
    "LOCAL_SOURCE_SNAPSHOT_STATUS",
    "LOCAL_SOURCE_SNAPSHOT_TREE_SCHEMA_VERSION",
    "MatchedV3LocalSourceSnapshot",
    "canonical_matched_v3_local_source_snapshot_descriptor_bytes",
    "matched_v3_local_source_snapshot_descriptor",
    "matched_v3_local_source_snapshot_descriptor_sha256",
    "measure_matched_v3_local_source_snapshot",
    "parse_matched_v3_local_source_snapshot_descriptor",
    "parse_matched_v3_local_source_snapshot_manifest",
    "verify_matched_v3_local_source_snapshot",
]
