"""Strict development-only contract for the namespaced continual-IA v2 study.

This module does not issue a plan at import time and does not reserve, consume,
or execute a seed implicitly.  Its public lifecycle is deliberately split into
an immutable plan, one deterministic-replay shard per seed, and an exact merge.
It cannot promote because it has no external chronology anchor. The v1
evaluator and immutable v1 outputs remain separate and unchanged.
"""

from __future__ import annotations

import ast
import errno
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import secrets
import stat
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import jax
import numpy as np

from alberta_framework.evaluation.continual_ia import (
    ContinualIAConfig,
    IAAcceptanceThresholds,
    IAConditionResult,
    _phase_means,
    _recovery_lengths,
    condition_controller_budgets,
    paired_bootstrap_mean_interval,
    run_continual_ia_benchmark,
)

PLAN_SCHEMA = "alberta.continual_ia.plan.v2"
SHARD_SCHEMA = "alberta.continual_ia.seed_shard.v2"
ARTIFACT_SCHEMA = "alberta.continual_ia.evidence.v2"
TRACE_SCHEMA = "alberta.continual_ia.primitive_trace.v2"
SOURCE_SCHEMA = "alberta.continual_ia.source_closure.v2"
RUNTIME_SCHEMA = "alberta.continual_ia.runtime.v3"
RESERVATION_SCHEMA = "alberta.continual_ia.seed_reservation.v2"
NAMESPACE = "continual_ia_v2_seed_60_89"
PROTOCOL_VERSION = "step12-hidden-phase-causal-ia.v2"

V2_EVIDENCE_SEEDS = tuple(range(60, 90))
V2_CONDITIONS = (
    "partner_alone",
    "observe_only",
    "recommendation_p075",
    "accept_always",
    "augmented_predictions",
    "augmented_noise",
)
V2_RECOMMENDATION_CONDITIONS = (
    "observe_only",
    "recommendation_p075",
    "accept_always",
)

V2_CONFIG = ContinualIAConfig(recommendation_acceptance_probability=0.75)
V2_THRESHOLDS = IAAcceptanceThresholds(evidence_seed_start=60)

_MAX_JSON_INTEGER = 2**53 - 1
_MAX_JSON_BYTES = 512 * 1024 * 1024
_REPO_ROOT = Path(__file__).absolute().parents[2]
_SOURCE_ROOT_MODULES = (
    "alberta_framework.evaluation.continual_ia_v2",
    "alberta_framework.evaluation.continual_ia_v2_cli",
)
_LOCKFILES = (Path("pyproject.toml"), Path("uv.lock"))
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | _NOFOLLOW
    | getattr(os, "O_CLOEXEC", 0)
)
_REGULAR_READ_FLAGS = (
    os.O_RDONLY
    | _NOFOLLOW
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_RUNTIME_ENVIRONMENT_NAMES = (
    "JAX_DEFAULT_MATMUL_PRECISION",
    "JAX_DEFAULT_PRNG_IMPL",
    "JAX_ENABLE_X64",
    "JAX_PLATFORM_NAME",
    "JAX_PLATFORMS",
    "XLA_FLAGS",
)
_RUNTIME_DISTRIBUTIONS = (
    "absl-py",
    "aiofiles",
    "chex",
    "cloudpickle",
    "etils",
    "humanize",
    "jax",
    "jaxlib",
    "jaxtyping",
    "ml-dtypes",
    "msgpack",
    "numpy",
    "opt-einsum",
    "orbax-checkpoint",
    "prometheus-client",
    "protobuf",
    "psutil",
    "pygments",
    "pyyaml",
    "scipy",
    "simplejson",
    "tensorstore",
    "toolz",
    "typing-extensions",
    "uvloop",
    "wadler-lindig",
)
_RUNTIME_DISTRIBUTION_CONTENT_SCOPE = (
    "explicit_clean_import_observed_plus_required_dependency_set"
)
_UNBOUND_RUNTIME_SCOPE = (
    "system_shared_libraries_loaded_by_python_or_extension_modules",
    "device_drivers_and_firmware",
    "dynamically_loaded_code_outside_distribution_file_manifests",
)

_EVIDENCE_POLICY: dict[str, object] = {
    "automatic_registry_promotion_allowed": False,
    "development_only": True,
    "fresh_seed_execution_required": True,
    "internal_l2_candidate_if_all_gates_pass": False,
    "external_prerun_chronology_attested": False,
    "independent_replication_present": False,
    "general_step12_claim_allowed": False,
    "sota_claim_allowed": False,
}

_TRACE_SEMANTICS: dict[str, object] = {
    "alignment": (
        "index t records reward_t from executed_action_t; the action, partner proposal, "
        "recommendation, and acceptance were selected before transition t and before any "
        "update using reward_t"
    ),
    "credit": "credited_actions[t] is the primitive action credited for transition t",
    "first_transition": (
        "recommendation/proposal equal the initial executed action and acceptance is false"
    ),
    "update_order": "predict/select before transition; observe reward; then update",
}

_DEVELOPMENT_SELECTION_PROVENANCE: dict[str, object] = {
    "role": "development_selection_only_nonpromoting",
    "v1_development_and_calibration_seed_ids": list(range(12)),
    "p075_selection_probe_consumed_seed_ids": list(range(30, 60)),
    "relationship_to_v1": (
        "the probe reused the already-consumed v1 evidence schedule and therefore cannot "
        "support v2 promotion"
    ),
    "selected_treatment": {
        "condition_id": "recommendation_p075",
        "recommendation_acceptance_probability": 0.75,
        "development_probe": {
            "changed_action_intervention_rate": 0.1083056,
            "primary_uplift": 0.27453,
            "paired_95_percent_lower_bound": 0.25386,
        },
    },
    "selection_rule": (
        "freeze p_accept=0.75 and retain every v1 gate, changing only the evidence seed "
        "start from 30 to 60 before any v2 evidence execution"
    ),
    "forbidden_uses": [
        "promotion",
        "threshold_retuning",
        "independent_replication",
        "general Step-12 claim",
        "state-of-the-art claim",
    ],
}


class ContinualIAV2Error(ValueError):
    """Raised when a v2 lifecycle payload or filesystem boundary fails closed."""


@dataclass(frozen=True)
class ContinualIAV2Validation:
    """Validation result with a deliberately narrow acceptance interpretation."""

    valid: bool
    internally_accepted: bool
    errors: tuple[str, ...]


def _fail(message: str) -> NoReturn:
    raise ContinualIAV2Error(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _is_int(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and -_MAX_JSON_INTEGER <= value <= _MAX_JSON_INTEGER
    )


def _is_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value)) and abs(float(value)) <= _MAX_JSON_INTEGER
    except (OverflowError, ValueError):
        return False


def _expect_dict(value: object, where: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{where} must be an object")
    return cast(dict[str, Any], value)


def _expect_list(value: object, where: str) -> list[Any]:
    _require(isinstance(value, list), f"{where} must be an array")
    return cast(list[Any], value)


def _expect_exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    _require(
        actual == expected,
        f"{where} keys differ: missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}",
    )


def _json_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _json_exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _parse_int(token: str) -> int:
    try:
        value = int(token)
    except ValueError as exc:  # pragma: no cover - json supplies integer tokens
        raise ContinualIAV2Error(f"invalid JSON integer: {token!r}") from exc
    _require(abs(value) <= _MAX_JSON_INTEGER, "JSON integer exceeds the exact safe bound")
    return value


def _parse_float(token: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:  # pragma: no cover - json supplies float tokens
        raise ContinualIAV2Error(f"invalid JSON number: {token!r}") from exc
    _require(math.isfinite(value), "non-finite JSON number is forbidden")
    _require(abs(value) <= _MAX_JSON_INTEGER, "JSON number exceeds the exact safe bound")
    return value


def _reject_constant(token: str) -> NoReturn:
    _fail(f"non-finite JSON constant is forbidden: {token}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    """Load the closed JSON subset: no duplicates, non-finite values, or huge ints."""

    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_int=_parse_int,
            parse_float=_parse_float,
        )
    except ContinualIAV2Error:
        raise
    except (json.JSONDecodeError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ContinualIAV2Error(f"invalid strict JSON: {exc}") from exc


def canonical_json_bytes(value: object) -> bytes:
    """Return the only accepted byte encoding for v2 lifecycle files."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContinualIAV2Error(f"value is not strict JSON: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def _compact_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContinualIAV2Error(f"value is not strict JSON: {exc}") from exc
    return encoded.encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha256(value: object) -> str:
    return sha256_bytes(_compact_json_bytes(value))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _absolute_lexical(path: Path) -> Path:
    """Make a path absolute without resolving a symlink."""

    _require(not any(part == ".." for part in path.parts), "parent traversal is forbidden")
    return Path(os.path.abspath(os.fspath(path)))


def _open_parent(path: Path, *, create: bool) -> tuple[int, str, Path]:
    absolute = _absolute_lexical(path)
    name = absolute.name
    _require(name not in {"", ".", ".."}, "file name is invalid")
    parts = absolute.parent.parts
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in parts[1:]:
            _require(component not in {"", ".", ".."}, "directory component is invalid")
            try:
                next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o755, dir_fd=descriptor)
                os.fsync(descriptor)
                next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, name, absolute


def _assert_parent_locator_stable(absolute: Path, directory_fd: int) -> None:
    """Require the descriptor's directory to remain at the requested locator."""

    verification_fd, _name, verified_absolute = _open_parent(absolute, create=False)
    try:
        opened = os.fstat(directory_fd)
        current = os.fstat(verification_fd)
        _require(
            (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)
            and verified_absolute == absolute,
            f"ancestor directory changed while accessing lifecycle path: {absolute}",
        )
    finally:
        os.close(verification_fd)


def _preflight_new_output(path: Path) -> Path:
    """Reject an occupied destination before an expensive execution begins."""

    directory_fd, name, absolute = _open_parent(path, create=True)
    try:
        _assert_parent_locator_stable(absolute, directory_fd)
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return absolute
        raise FileExistsError(f"refusing to overwrite immutable output: {absolute}")
    finally:
        os.close(directory_fd)


def _atomic_publish_new(path: Path, data: bytes) -> Path:
    """Durably publish immutable 0444 bytes without following or replacing links."""

    directory_fd, name, absolute = _open_parent(path, create=True)
    temporary = f".{name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    file_fd: int | None = None
    target_linked = False
    completed = False
    try:
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite immutable output: {absolute}")
        file_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        view = memoryview(data)
        while view:
            written = os.write(file_fd, view)
            _require(written > 0, "short write while publishing immutable output")
            view = view[written:]
        os.fchmod(file_fd, 0o444)
        os.fsync(file_fd)
        _assert_parent_locator_stable(absolute, directory_fd)
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite immutable output: {absolute}") from exc
        target_linked = True
        source_stat = os.fstat(file_fd)
        target_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        source_identity = (source_stat.st_dev, source_stat.st_ino, source_stat.st_size)
        target_identity = (target_stat.st_dev, target_stat.st_ino, target_stat.st_size)
        if (
            source_identity != target_identity
            or stat.S_IMODE(target_stat.st_mode) != 0o444
            or not stat.S_ISREG(target_stat.st_mode)
        ):
            if (source_stat.st_dev, source_stat.st_ino) == (
                target_stat.st_dev,
                target_stat.st_ino,
            ):
                os.unlink(name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            target_linked = False
            _fail("published path does not identify the descriptor-anchored temporary file")
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        temporary = ""
        os.fsync(directory_fd)
        final_target = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _require(
            (final_target.st_dev, final_target.st_ino, final_target.st_size)
            == source_identity
            and final_target.st_nlink == 1
            and stat.S_IMODE(final_target.st_mode) == 0o444
            and stat.S_ISREG(final_target.st_mode),
            "published output identity or link count changed",
        )
        _assert_parent_locator_stable(absolute, directory_fd)
        try:
            published = _read_regular_bytes(absolute, require_immutable=True)
            _require(published == data, "published output bytes differ from supplied bytes")
            post_read_target = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            _require(
                (
                    post_read_target.st_dev,
                    post_read_target.st_ino,
                    post_read_target.st_size,
                )
                == source_identity
                and post_read_target.st_nlink == 1
                and stat.S_IMODE(post_read_target.st_mode) == 0o444
                and stat.S_ISREG(post_read_target.st_mode),
                "published output changed during byte verification",
            )
            _assert_parent_locator_stable(absolute, directory_fd)
        except BaseException:
            # The descriptor-identity guard in ``finally`` removes only our
            # inode. Never unlink a concurrently substituted path by name.
            raise
        completed = True
    finally:
        if target_linked and not completed and file_fd is not None:
            try:
                source = os.fstat(file_fd)
                target = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (source.st_dev, source.st_ino) == (target.st_dev, target.st_ino):
                    os.unlink(name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
            except FileNotFoundError:
                pass
        if file_fd is not None:
            os.close(file_fd)
        if temporary:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            else:
                os.fsync(directory_fd)
        os.close(directory_fd)
    return absolute


def _atomic_publish_new_json(path: Path, payload: object) -> Path:
    return _atomic_publish_new(path, canonical_json_bytes(payload))


def _read_regular_bytes(
    path: Path,
    *,
    require_immutable: bool,
    max_bytes: int = _MAX_JSON_BYTES,
) -> bytes:
    """Read one descriptor-anchored regular file and detect locator replacement."""

    directory_fd, name, absolute = _open_parent(path, create=False)
    file_fd: int | None = None
    try:
        _assert_parent_locator_stable(absolute, directory_fd)
        file_fd = os.open(name, _REGULAR_READ_FLAGS, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        _require(stat.S_ISREG(before.st_mode), f"not a regular file: {absolute}")
        if require_immutable:
            _require(
                stat.S_IMODE(before.st_mode) == 0o444,
                f"immutable lifecycle file must have mode 0444: {absolute}",
            )
            _require(
                before.st_nlink == 1,
                f"immutable lifecycle file must have exactly one hard link: {absolute}",
            )
        _require(before.st_size <= max_bytes, f"file exceeds byte limit: {absolute}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_fd, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            _require(total <= max_bytes, f"file exceeds byte limit: {absolute}")
        after = os.fstat(file_fd)
        locator = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        identity_locator = (
            locator.st_dev,
            locator.st_ino,
            locator.st_size,
            locator.st_mtime_ns,
            locator.st_ctime_ns,
        )
        _require(
            identity_before == identity_after == identity_locator,
            f"file changed or locator was replaced during read: {absolute}",
        )
        _assert_parent_locator_stable(absolute, directory_fd)
        return b"".join(chunks)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ContinualIAV2Error(f"symlinked lifecycle path is forbidden: {absolute}") from exc
        raise
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def _read_canonical_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = _read_regular_bytes(path, require_immutable=True)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContinualIAV2Error(f"lifecycle file is not UTF-8: {path}") from exc
    value = strict_json_loads(text)
    payload = _expect_dict(value, str(path))
    _require(raw == canonical_json_bytes(payload), f"non-canonical JSON bytes: {path}")
    return raw, payload


def _module_candidates(module: str) -> tuple[Path, Path]:
    relative = Path(*module.split("."))
    return _REPO_ROOT / relative.with_suffix(".py"), _REPO_ROOT / relative / "__init__.py"


def _module_path(module: str) -> Path | None:
    if module != "alberta_framework" and not module.startswith("alberta_framework."):
        return None
    for candidate in _module_candidates(module):
        try:
            _read_regular_bytes(candidate, require_immutable=False, max_bytes=16 * 1024 * 1024)
        except FileNotFoundError:
            continue
        return candidate
    return None


def _module_name(path: Path) -> tuple[str, bool]:
    relative = path.relative_to(_REPO_ROOT)
    if relative.name == "__init__.py":
        return ".".join(relative.parent.parts), True
    return ".".join(relative.with_suffix("").parts), False


def _parent_packages(module: str) -> set[str]:
    parts = module.split(".")
    return {".".join(parts[:index]) for index in range(1, len(parts))}


def _resolve_local_imports(path: Path, raw: bytes) -> set[str]:
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    except (SyntaxError, UnicodeError) as exc:
        raise ContinualIAV2Error(f"cannot parse source closure member {path}: {exc}") from exc
    module, is_package = _module_name(path)
    package = module if is_package else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = package.split(".") if package else []
                keep = len(package_parts) - node.level + 1
                if keep < 0:
                    continue
                base_parts = package_parts[:keep]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                candidates.append(base)
                candidates.extend(f"{base}.{alias.name}" for alias in node.names)
        for candidate in candidates:
            parts = candidate.split(".")
            while parts:
                possible = ".".join(parts)
                if _module_path(possible) is not None:
                    found.add(possible)
                    found.update(_parent_packages(possible))
                    break
                parts.pop()
    return found


def _build_source_manifest() -> dict[str, Any]:
    pending = set(_SOURCE_ROOT_MODULES)
    pending.update(parent for module in _SOURCE_ROOT_MODULES for parent in _parent_packages(module))
    visited: set[str] = set()
    bytes_by_module: dict[str, bytes] = {}
    while pending:
        module = min(pending)
        pending.remove(module)
        if module in visited:
            continue
        path = _module_path(module)
        _require(path is not None, f"source closure module is missing: {module}")
        assert path is not None
        visited.add(module)
        raw = _read_regular_bytes(path, require_immutable=False, max_bytes=16 * 1024 * 1024)
        bytes_by_module[module] = raw
        pending.update(_resolve_local_imports(path, raw) - visited)
    files: list[dict[str, object]] = []
    for module in sorted(visited):
        path = _module_path(module)
        assert path is not None
        raw = bytes_by_module[module]
        files.append(
            {
                "module": module,
                "locator": path.relative_to(_REPO_ROOT).as_posix(),
                "byte_size": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )
    lockfiles: list[dict[str, object]] = []
    lockfile_bytes: dict[Path, bytes] = {}
    for relative in _LOCKFILES:
        raw = _read_regular_bytes(_REPO_ROOT / relative, require_immutable=False)
        lockfile_bytes[relative] = raw
        lockfiles.append(
            {
                "locator": relative.as_posix(),
                "byte_size": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )
    for entry in files:
        locator = Path(cast(str, entry["locator"]))
        current = _read_regular_bytes(
            _REPO_ROOT / locator,
            require_immutable=False,
            max_bytes=16 * 1024 * 1024,
        )
        _require(
            len(current) == entry["byte_size"]
            and sha256_bytes(current) == entry["sha256"],
            f"source file changed while its closure was being built: {locator}",
        )
    for entry in lockfiles:
        locator = Path(cast(str, entry["locator"]))
        current = _read_regular_bytes(_REPO_ROOT / locator, require_immutable=False)
        _require(
            current == lockfile_bytes[locator],
            f"lockfile changed while its closure was being built: {locator}",
        )
    return {
        "schema": SOURCE_SCHEMA,
        "closure_kind": "static_transitive_local_python_imports",
        "repository_subtree": "research/alberta",
        "root_modules": list(_SOURCE_ROOT_MODULES),
        "files": files,
        "lockfiles": lockfiles,
    }


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _distribution_content_identity(name: str) -> dict[str, object]:
    """Hash the actual installed distribution bytes, not only its version string."""

    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {
            "status": "not_installed",
            "file_count": 0,
            "total_bytes": 0,
            "sha256": sha256_bytes(b""),
        }
    files = distribution.files
    _require(files is not None, f"installed distribution {name!r} has no file manifest")
    assert files is not None
    digest = hashlib.sha256()
    count = 0
    total = 0
    for relative in sorted(files, key=lambda item: str(item)):
        path = Path(cast(Any, distribution.locate_file(relative))).resolve()
        if not path.is_file():
            continue
        raw = _read_regular_bytes(path, require_immutable=False)
        locator = str(relative).replace(os.sep, "/")
        digest.update(locator.encode("utf-8") + b"\0")
        digest.update(str(len(raw)).encode("ascii") + b"\0")
        digest.update(hashlib.sha256(raw).digest())
        count += 1
        total += len(raw)
    _require(count > 0, f"installed distribution {name!r} has no regular files")
    return {
        "status": "content_hashed",
        "file_count": count,
        "total_bytes": total,
        "sha256": digest.hexdigest(),
    }


def _module_content_identity(name: str) -> dict[str, object]:
    spec = importlib.util.find_spec(name)
    _require(
        spec is not None and isinstance(spec.origin, str) and bool(spec.origin),
        f"runtime module {name!r} has no file origin",
    )
    assert spec is not None and spec.origin is not None
    path = Path(spec.origin).resolve()
    raw = _read_regular_bytes(path, require_immutable=False)
    return {
        "locator": path.as_posix(),
        "byte_size": len(raw),
        "sha256": sha256_bytes(raw),
    }


def _runtime_json_value(value: object) -> str | int | float | bool | None:
    if value is None or type(value) in {str, int, bool}:
        return cast(str | int | bool | None, value)
    if type(value) is float:
        _require(math.isfinite(value), "JAX config contains a non-finite float")
        return value
    return str(value)


def _build_runtime_manifest() -> dict[str, Any]:
    executable_path = Path(sys.executable).resolve()
    executable_raw = _read_regular_bytes(executable_path, require_immutable=False)
    distribution_content = {
        name: _distribution_content_identity(name) for name in _RUNTIME_DISTRIBUTIONS
    }
    missing_distributions = sorted(
        name
        for name, identity in distribution_content.items()
        if identity["status"] != "content_hashed"
    )
    _require(
        not missing_distributions,
        f"required runtime distributions are not installed: {missing_distributions}",
    )
    flag_names = (
        "debug",
        "inspect",
        "interactive",
        "optimize",
        "dont_write_bytecode",
        "no_user_site",
        "no_site",
        "ignore_environment",
        "verbose",
        "bytes_warning",
        "quiet",
        "hash_randomization",
        "isolated",
        "dev_mode",
        "utf8_mode",
        "warn_default_encoding",
        "safe_path",
        "int_max_str_digits",
    )
    return {
        "schema": RUNTIME_SCHEMA,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": {
                "locator": executable_path.as_posix(),
                "byte_size": len(executable_raw),
                "sha256": sha256_bytes(executable_raw),
            },
            "flags": {name: int(getattr(sys.flags, name, 0)) for name in flag_names},
            "sys_path": [str(item) for item in sys.path],
        },
        "platform": {
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "dependencies": {
            "alberta-framework": _distribution_version("alberta-framework"),
            **{name: _distribution_version(name) for name in _RUNTIME_DISTRIBUTIONS},
        },
        "distribution_content_scope": _RUNTIME_DISTRIBUTION_CONTENT_SCOPE,
        "distribution_content": distribution_content,
        "unbound_runtime_scope": list(_UNBOUND_RUNTIME_SCOPE),
        "module_origins": {
            name: _module_content_identity(name) for name in ("jax", "jaxlib", "numpy")
        },
        "jax": {
            "backend": jax.default_backend(),
            "enable_x64": bool(jax.config.jax_enable_x64),
            "config": {
                name: _runtime_json_value(value)
                for name, value in sorted(jax.config.values.items())
            },
            "environment": {
                name: os.environ.get(name) for name in _RUNTIME_ENVIRONMENT_NAMES
            },
            "devices": [
                {
                    "device_kind": device.device_kind,
                    "id": int(device.id),
                    "platform": device.platform,
                    "process_index": int(device.process_index),
                    "runtime_type": str(getattr(device.client, "runtime_type", "unknown")),
                }
                for device in jax.devices()
            ],
        },
    }


def _capture_runtime_manifest() -> dict[str, Any]:
    try:
        return _build_runtime_manifest()
    except ContinualIAV2Error:
        raise
    except Exception as exc:
        raise ContinualIAV2Error(f"runtime identity discovery failed: {exc}") from exc


def _config_payload() -> dict[str, object]:
    return {
        "num_steps": V2_CONFIG.num_steps,
        "phase_length": V2_CONFIG.phase_length,
        "observation_dim": V2_CONFIG.observation_dim,
        "n_actions": V2_CONFIG.n_actions,
        "n_demons": V2_CONFIG.n_demons,
        "partner_q_step_size": V2_CONFIG.partner_q_step_size,
        "partner_average_reward_step_size": V2_CONFIG.partner_average_reward_step_size,
        "partner_epsilon": V2_CONFIG.partner_epsilon,
        "cortex_base_step_size": V2_CONFIG.cortex_base_step_size,
        "recommendation_acceptance_probability": 0.75,
        "recovery_window": V2_CONFIG.recovery_window,
        "recovery_mean_reward_threshold": V2_CONFIG.recovery_mean_reward_threshold,
        "bootstrap_resamples": V2_CONFIG.bootstrap_resamples,
        "confidence_level": V2_CONFIG.confidence_level,
        "bootstrap_seed": V2_CONFIG.bootstrap_seed,
    }


def _threshold_payload() -> dict[str, object]:
    return {
        "minimum_seed_count": 30,
        "evidence_seed_start": 60,
        "minimum_primary_uplift_lower_ci": 0.10,
        "minimum_changed_action_intervention_rate": 0.10,
        "minimum_augmentation_vs_alone_lower_ci": 0.05,
        "minimum_augmentation_vs_noise_lower_ci": 0.05,
        "require_observe_only_exact_identity": True,
        "maximum_executed_action_credit_mismatches": 0,
    }


def _condition_specs() -> list[dict[str, object]]:
    return [
        {
            "id": "partner_alone",
            "acceptance_probability": None,
            "text": "partner without IA",
        },
        {
            "id": "observe_only",
            "acceptance_probability": 0.0,
            "text": "IA runs and learns; recommendations are never accepted",
        },
        {
            "id": "recommendation_p075",
            "acceptance_probability": 0.75,
            "text": "IA recommendation treatment with exact p_accept=0.75",
        },
        {
            "id": "accept_always",
            "acceptance_probability": 1.0,
            "text": "full-delegation negative diagnostic; no uplift is required",
        },
        {
            "id": "augmented_predictions",
            "acceptance_probability": None,
            "text": "partner observes raw state plus learned cerebellum predictions",
        },
        {
            "id": "augmented_noise",
            "acceptance_probability": None,
            "text": "equal-shape control with prediction features replaced by uniform noise",
        },
    ]


def _budget_payload() -> dict[str, dict[str, object]]:
    raw = condition_controller_budgets(V2_CONFIG)
    renamed = {
        "partner_alone": raw["partner_alone"],
        "observe_only": raw["observe_only"],
        "recommendation_p075": raw["recommendation_p05"],
        "accept_always": raw["accept_always"],
        "augmented_predictions": raw["augmented_predictions"],
        "augmented_noise": raw["augmented_noise"],
    }
    return {
        condition: {
            "state_scalars": budget.state_scalars,
            "state_bytes": budget.state_bytes,
            "observation_scalars": budget.observation_scalars,
            "action_scalars_per_step": budget.action_scalars_per_step,
            "interaction_steps": budget.interaction_steps,
            "ia_attached": budget.ia_attached,
        }
        for condition, budget in renamed.items()
    }


def _run_spec() -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "primary_estimand": ("paired seed mean reward of recommendation_p075 minus observe_only"),
        "configuration": _config_payload(),
        "thresholds": _threshold_payload(),
        "conditions": _condition_specs(),
        "seed_schedule": {
            "seed_ids": list(V2_EVIDENCE_SEEDS),
            "seed_count": 30,
            "known_consumed_seed_ids": [*range(12), *range(30, 60)],
            "freshness": "operator_asserted_unexecuted_development_namespace_60_89",
            "freshness_attestation": (
                "self-reported operator assertion that seeds 60-89 were not previously "
                "executed; this is not externally anchored chronology"
            ),
        },
        "planned_shard_count": 30,
        "one_seed_per_shard": True,
        "controller_budgets": _budget_payload(),
        "trace_schema": TRACE_SCHEMA,
        "trace_semantics": dict(_TRACE_SEMANTICS),
        "statistics": {
            "pairing_unit": "seed",
            "interval": "paired-percentile-bootstrap",
            "resamples": 10_000,
            "confidence_level": 0.95,
            "bootstrap_seed": V2_CONFIG.bootstrap_seed,
        },
    }


def _claim_scope() -> dict[str, object]:
    return {
        "supported_if_all_gates_pass": (
            "reproducible development-only causal recommendation-channel diagnostic for the "
            "selected IA/partner pair in the deterministic hidden-phase micro-MDP"
        ),
        "independent_replication": "not established",
        "general_step12": "not established",
        "alberta_plan_completion": "not established",
        "state_of_the_art": "not established",
        "limitations": [
            "deterministic hand-designed two-state hidden-phase MDP",
            "p_accept=0.75 selected using a nonpromoting consumed-seed development probe",
            "single selected IA/partner pair",
            "no autonomous feature discovery",
            "internal source/runtime binding is not independent execution attestation",
            "self-issued v2 has no externally verifiable pre-run chronology",
        ],
    }


def _output_layout(
    plan_path: Path, shard_directory: Path, artifact_path: Path
) -> dict[str, object]:
    plan = _absolute_lexical(plan_path).as_posix()
    shard_dir = _absolute_lexical(shard_directory).as_posix()
    artifact = _absolute_lexical(artifact_path).as_posix()
    shards = [
        {
            "seed": seed,
            "path": (_absolute_lexical(shard_directory) / f"seed-{seed:03d}.v2.json").as_posix(),
        }
        for seed in V2_EVIDENCE_SEEDS
    ]
    return {
        "plan_path": plan,
        "shard_directory": shard_dir,
        "shards": shards,
        "artifact_path": artifact,
    }


def _commands(layout: Mapping[str, object]) -> dict[str, object]:
    plan_path = cast(str, layout["plan_path"])
    shard_dir = cast(str, layout["shard_directory"])
    artifact_path = cast(str, layout["artifact_path"])
    shards = cast(list[dict[str, object]], layout["shards"])
    return {
        "plan": [
            "plan",
            "--plan-out",
            plan_path,
            "--shard-dir",
            shard_dir,
            "--artifact-out",
            artifact_path,
            "--attest-fresh-seeds-60-89",
        ],
        "shards": [
            [
                "shard",
                "--plan",
                plan_path,
                "--seed",
                str(item["seed"]),
                "--output",
                str(item["path"]),
            ]
            for item in shards
        ],
        "merge": [
            "merge",
            "--plan",
            plan_path,
            "--output",
            artifact_path,
        ],
    }


def _build_plan_payload(
    plan_path: Path,
    shard_directory: Path,
    artifact_path: Path,
    *,
    issued_unix: int | None = None,
) -> dict[str, object]:
    """Build, but do not publish, the exact future v2 execution plan."""

    issued = int(time.time()) if issued_unix is None else issued_unix
    _require(_is_int(issued) and issued >= 0, "issued_unix must be a safe nonnegative integer")
    _require(
        issued <= int(time.time()) + 5,
        "issued_unix cannot be in the future",
    )
    layout = _output_layout(plan_path, shard_directory, artifact_path)
    _validate_layout(layout)
    commands = _commands(layout)
    source = _build_source_manifest()
    runtime = _capture_runtime_manifest()
    body: dict[str, object] = {
        "run_spec": _run_spec(),
        "run_spec_sha256": canonical_json_sha256(_run_spec()),
        "development_selection_provenance": dict(_DEVELOPMENT_SELECTION_PROVENANCE),
        "claim_scope": _claim_scope(),
        "output_layout": layout,
        "commands": commands,
        "source_manifest": source,
        "source_manifest_sha256": canonical_json_sha256(source),
        "runtime_manifest": runtime,
        "runtime_manifest_sha256": canonical_json_sha256(runtime),
        "issuance": {
            "issued_unix": issued,
            "kind": "self_issued_development_plan_without_external_chronology",
            "prescribed_argv": commands["plan"],
            "external_execution_attestation_present": False,
        },
    }
    return {
        "schema": PLAN_SCHEMA,
        "namespace": NAMESPACE,
        "evidence_policy": dict(_EVIDENCE_POLICY),
        "plan": body,
        "plan_sha256": canonical_json_sha256(body),
    }


def write_plan(
    path: Path,
    shard_directory: Path,
    artifact_path: Path,
    *,
    attest_fresh_seeds: bool,
) -> Path:
    """Explicitly issue one immutable v2 plan; never called implicitly."""

    _require(attest_fresh_seeds is True, "fresh seeds 60-89 must be explicitly attested")
    _preflight_new_output(path)
    payload = _build_plan_payload(path, shard_directory, artifact_path)
    _validate_plan_or_raise(payload, locator=path, recheck_current=True)
    body = cast(dict[str, Any], payload["plan"])
    layout = cast(dict[str, Any], body["output_layout"])
    _preflight_new_output(Path(cast(str, layout["artifact_path"])))
    for entry in cast(list[dict[str, Any]], layout["shards"]):
        shard_path = Path(cast(str, entry["path"]))
        _preflight_new_output(shard_path)
        _preflight_new_output(_reservation_path(shard_path))
    return _atomic_publish_new_json(path, payload)


def _validate_source_manifest_shape(value: object) -> dict[str, Any]:
    manifest = _expect_dict(value, "plan.source_manifest")
    _expect_exact_keys(
        manifest,
        {
            "schema",
            "closure_kind",
            "repository_subtree",
            "root_modules",
            "files",
            "lockfiles",
        },
        "plan.source_manifest",
    )
    _require(manifest["schema"] == SOURCE_SCHEMA, "source manifest schema differs")
    _require(
        manifest["closure_kind"] == "static_transitive_local_python_imports",
        "source closure kind differs",
    )
    _require(
        manifest["repository_subtree"] == "research/alberta",
        "source repository subtree differs",
    )
    _require(
        _json_exact_equal(manifest["root_modules"], list(_SOURCE_ROOT_MODULES)),
        "source root modules differ",
    )
    for collection, with_module in (("files", True), ("lockfiles", False)):
        items = _expect_list(manifest[collection], f"plan.source_manifest.{collection}")
        _require(bool(items), f"plan.source_manifest.{collection} cannot be empty")
        locators: list[str] = []
        for index, raw_item in enumerate(items):
            where = f"plan.source_manifest.{collection}[{index}]"
            item = _expect_dict(raw_item, where)
            keys = {"locator", "byte_size", "sha256"}
            if with_module:
                keys.add("module")
            _expect_exact_keys(item, keys, where)
            locator = item["locator"]
            _require(isinstance(locator, str) and bool(locator), f"{where}.locator invalid")
            _require(".." not in Path(locator).parts, f"{where}.locator traverses parents")
            locators.append(locator)
            _require(
                _is_int(item["byte_size"]) and item["byte_size"] >= 0,
                f"{where}.byte_size invalid",
            )
            _require(_is_sha256(item["sha256"]), f"{where}.sha256 invalid")
            if with_module:
                _require(
                    isinstance(item["module"], str) and bool(item["module"]),
                    f"{where}.module invalid",
                )
        _require(locators == sorted(locators), f"{collection} locators must be sorted")
        _require(len(locators) == len(set(locators)), f"{collection} locators must be unique")
    lock_locators = [item["locator"] for item in cast(list[dict[str, Any]], manifest["lockfiles"])]
    _require(
        lock_locators == [path.as_posix() for path in _LOCKFILES],
        "source lockfile set differs",
    )
    return manifest


def _validate_runtime_manifest_shape(value: object) -> dict[str, Any]:
    manifest = _expect_dict(value, "plan.runtime_manifest")
    _expect_exact_keys(
        manifest,
        {
            "schema",
            "python",
            "platform",
            "dependencies",
            "distribution_content_scope",
            "distribution_content",
            "unbound_runtime_scope",
            "module_origins",
            "jax",
        },
        "plan.runtime_manifest",
    )
    _require(manifest["schema"] == RUNTIME_SCHEMA, "runtime schema differs")
    _require(
        manifest["distribution_content_scope"]
        == _RUNTIME_DISTRIBUTION_CONTENT_SCOPE,
        "runtime distribution-content scope differs",
    )
    unbound_runtime_scope = _expect_list(
        manifest["unbound_runtime_scope"],
        "runtime.unbound_runtime_scope",
    )
    _require(
        unbound_runtime_scope == list(_UNBOUND_RUNTIME_SCOPE),
        "runtime unbound-runtime scope differs",
    )
    python = _expect_dict(manifest["python"], "plan.runtime_manifest.python")
    _expect_exact_keys(
        python,
        {"implementation", "version", "executable", "flags", "sys_path"},
        "runtime.python",
    )
    system = _expect_dict(manifest["platform"], "plan.runtime_manifest.platform")
    _expect_exact_keys(system, {"machine", "release", "system"}, "runtime.platform")
    dependencies = _expect_dict(manifest["dependencies"], "plan.runtime_manifest.dependencies")
    _expect_exact_keys(
        dependencies,
        {"alberta-framework", *_RUNTIME_DISTRIBUTIONS},
        "runtime.dependencies",
    )
    for where, mapping in (("runtime.platform", system), ("runtime.dependencies", dependencies)):
        _require(
            all(isinstance(item, str) and bool(item) for item in mapping.values()),
            f"{where} values must be nonempty strings",
        )
    _require(
        isinstance(python["implementation"], str)
        and bool(python["implementation"])
        and isinstance(python["version"], str)
        and bool(python["version"]),
        "runtime.python implementation/version invalid",
    )
    executable = _expect_dict(python["executable"], "runtime.python.executable")
    _expect_exact_keys(
        executable,
        {"locator", "byte_size", "sha256"},
        "runtime.python.executable",
    )
    _require(
        isinstance(executable["locator"], str)
        and Path(executable["locator"]).is_absolute(),
        "runtime.python.executable.locator invalid",
    )
    _require(
        _is_int(executable["byte_size"]) and executable["byte_size"] > 0,
        "runtime.python.executable.byte_size invalid",
    )
    _require(_is_sha256(executable["sha256"]), "runtime.python.executable.sha256 invalid")
    flags = _expect_dict(python["flags"], "runtime.python.flags")
    _require(bool(flags), "runtime.python.flags cannot be empty")
    _require(
        all(isinstance(name, str) and _is_int(value) for name, value in flags.items()),
        "runtime.python.flags invalid",
    )
    sys_path = _expect_list(python["sys_path"], "runtime.python.sys_path")
    _require(
        all(isinstance(item, str) for item in sys_path),
        "runtime.python.sys_path values invalid",
    )
    distribution_content = _expect_dict(
        manifest["distribution_content"],
        "runtime.distribution_content",
    )
    _expect_exact_keys(
        distribution_content,
        set(_RUNTIME_DISTRIBUTIONS),
        "runtime.distribution_content",
    )
    for name, raw_identity in distribution_content.items():
        identity = _expect_dict(raw_identity, f"runtime.distribution_content.{name}")
        _expect_exact_keys(
            identity,
            {"status", "file_count", "total_bytes", "sha256"},
            f"runtime.distribution_content.{name}",
        )
        _require(
            identity["status"] in {"content_hashed", "not_installed"},
            f"runtime.distribution_content.{name}.status invalid",
        )
        _require(
            _is_int(identity["file_count"]) and identity["file_count"] >= 0,
            f"runtime.distribution_content.{name}.file_count invalid",
        )
        _require(
            _is_int(identity["total_bytes"]) and identity["total_bytes"] >= 0,
            f"runtime.distribution_content.{name}.total_bytes invalid",
        )
        _require(
            _is_sha256(identity["sha256"]),
            f"runtime.distribution_content.{name}.sha256 invalid",
        )
    module_origins = _expect_dict(manifest["module_origins"], "runtime.module_origins")
    _expect_exact_keys(module_origins, {"jax", "jaxlib", "numpy"}, "runtime.module_origins")
    for name, raw_identity in module_origins.items():
        identity = _expect_dict(raw_identity, f"runtime.module_origins.{name}")
        _expect_exact_keys(
            identity,
            {"locator", "byte_size", "sha256"},
            f"runtime.module_origins.{name}",
        )
        _require(
            isinstance(identity["locator"], str) and Path(identity["locator"]).is_absolute(),
            f"runtime.module_origins.{name}.locator invalid",
        )
        _require(
            _is_int(identity["byte_size"]) and identity["byte_size"] > 0,
            f"runtime.module_origins.{name}.byte_size invalid",
        )
        _require(_is_sha256(identity["sha256"]), f"runtime.module_origins.{name}.sha256 invalid")
    jax_payload = _expect_dict(manifest["jax"], "plan.runtime_manifest.jax")
    _expect_exact_keys(
        jax_payload,
        {"backend", "enable_x64", "config", "devices", "environment"},
        "runtime.jax",
    )
    _require(
        isinstance(jax_payload["backend"], str) and bool(jax_payload["backend"]),
        "runtime.jax.backend invalid",
    )
    _require(type(jax_payload["enable_x64"]) is bool, "runtime.jax.enable_x64 invalid")
    config = _expect_dict(jax_payload["config"], "runtime.jax.config")
    _require(bool(config), "runtime.jax.config cannot be empty")
    for name, item in config.items():
        _require(
            isinstance(name, str)
            and type(item) in {str, int, float, bool, type(None)}
            and (type(item) is not float or math.isfinite(item)),
            f"runtime.jax.config[{name!r}] invalid",
        )
    environment = _expect_dict(jax_payload["environment"], "runtime.jax.environment")
    _expect_exact_keys(
        environment,
        set(_RUNTIME_ENVIRONMENT_NAMES),
        "runtime.jax.environment",
    )
    _require(
        all(item is None or isinstance(item, str) for item in environment.values()),
        "runtime.jax.environment values invalid",
    )
    devices = _expect_list(jax_payload["devices"], "runtime.jax.devices")
    _require(bool(devices), "runtime.jax.devices cannot be empty")
    for index, raw_device in enumerate(devices):
        device = _expect_dict(raw_device, f"runtime.jax.devices[{index}]")
        _expect_exact_keys(
            device,
            {"device_kind", "id", "platform", "process_index", "runtime_type"},
            f"runtime.jax.devices[{index}]",
        )
        _require(_is_int(device["id"]) and device["id"] >= 0, "runtime device id invalid")
        _require(
            _is_int(device["process_index"]) and device["process_index"] >= 0,
            "runtime device process_index invalid",
        )
        _require(
            isinstance(device["device_kind"], str) and bool(device["device_kind"]),
            "runtime device kind invalid",
        )
        _require(
            isinstance(device["platform"], str) and bool(device["platform"]),
            "runtime device platform invalid",
        )
        _require(
            isinstance(device["runtime_type"], str) and bool(device["runtime_type"]),
            "runtime device runtime_type invalid",
        )
    return manifest


def _validate_layout(value: object) -> dict[str, Any]:
    layout = _expect_dict(value, "plan.output_layout")
    _expect_exact_keys(
        layout,
        {"plan_path", "shard_directory", "shards", "artifact_path"},
        "plan.output_layout",
    )
    for field in ("plan_path", "shard_directory", "artifact_path"):
        raw = layout[field]
        _require(isinstance(raw, str) and Path(raw).is_absolute(), f"{field} must be absolute")
        _require(".." not in Path(raw).parts, f"{field} cannot traverse parents")
        _require(
            _absolute_lexical(Path(raw)).as_posix() == raw,
            f"{field} must be a canonical lexical absolute path",
        )
    shards = _expect_list(layout["shards"], "plan.output_layout.shards")
    _require(len(shards) == 30, "output layout must contain exactly thirty shards")
    observed: list[int] = []
    shard_paths: list[str] = []
    for index, raw_item in enumerate(shards):
        item = _expect_dict(raw_item, f"plan.output_layout.shards[{index}]")
        _expect_exact_keys(item, {"seed", "path"}, f"output_layout.shards[{index}]")
        seed = item["seed"]
        path = item["path"]
        _require(_is_int(seed), f"output_layout.shards[{index}].seed invalid")
        _require(isinstance(path, str) and Path(path).is_absolute(), "shard path invalid")
        _require(
            _absolute_lexical(Path(path)).as_posix() == path,
            f"output_layout.shards[{index}].path must be canonical",
        )
        observed.append(seed)
        shard_paths.append(path)
    _require(tuple(observed) == V2_EVIDENCE_SEEDS, "output shard schedule must be 60-89")
    expected = _output_layout(
        Path(cast(str, layout["plan_path"])),
        Path(cast(str, layout["shard_directory"])),
        Path(cast(str, layout["artifact_path"])),
    )
    _require(_json_exact_equal(layout, expected), "output layout differs from canonical layout")
    file_locators = [
        cast(str, layout["plan_path"]),
        cast(str, layout["artifact_path"]),
        *shard_paths,
    ]
    reservation_locators = [f"{path}.reservation" for path in shard_paths]
    all_locators = [
        cast(str, layout["shard_directory"]),
        *file_locators,
        *reservation_locators,
    ]
    _require(
        len(all_locators) == len(set(all_locators)),
        "plan, shard directory, shard, reservation, and artifact locators must be distinct",
    )
    file_paths = [Path(path) for path in (*file_locators, *reservation_locators)]
    all_paths = [Path(path) for path in all_locators]
    _require(
        not any(
            file_path in other.parents
            for file_path in file_paths
            for other in all_paths
            if file_path != other
        ),
        "a lifecycle file locator cannot be an ancestor of another bound locator",
    )
    return layout


def _validate_plan_or_raise(
    payload: Mapping[str, object],
    *,
    locator: Path | None,
    recheck_current: bool,
) -> dict[str, Any]:
    plan_payload = cast(dict[str, Any], dict(payload))
    _expect_exact_keys(
        plan_payload,
        {"schema", "namespace", "evidence_policy", "plan", "plan_sha256"},
        "plan payload",
    )
    _require(plan_payload["schema"] == PLAN_SCHEMA, "wrong v2 plan schema")
    _require(plan_payload["namespace"] == NAMESPACE, "wrong v2 namespace")
    _require(
        _json_exact_equal(plan_payload["evidence_policy"], _EVIDENCE_POLICY),
        "evidence policy differs",
    )
    body = _expect_dict(plan_payload["plan"], "plan")
    _expect_exact_keys(
        body,
        {
            "run_spec",
            "run_spec_sha256",
            "development_selection_provenance",
            "claim_scope",
            "output_layout",
            "commands",
            "source_manifest",
            "source_manifest_sha256",
            "runtime_manifest",
            "runtime_manifest_sha256",
            "issuance",
        },
        "plan",
    )
    expected_run_spec = _run_spec()
    _require(_json_exact_equal(body["run_spec"], expected_run_spec), "run spec differs")
    _require(
        body["run_spec_sha256"] == canonical_json_sha256(expected_run_spec),
        "run spec digest differs",
    )
    _require(
        _json_exact_equal(
            body["development_selection_provenance"], _DEVELOPMENT_SELECTION_PROVENANCE
        ),
        "development-selection provenance differs",
    )
    _require(_json_exact_equal(body["claim_scope"], _claim_scope()), "claim scope differs")
    layout = _validate_layout(body["output_layout"])
    if locator is not None:
        _require(
            cast(str, layout["plan_path"]) == _absolute_lexical(locator).as_posix(),
            "plan locator differs from its bound path",
        )
    commands = _expect_dict(body["commands"], "plan.commands")
    _require(_json_exact_equal(commands, _commands(layout)), "bound commands/argv differ")
    source = _validate_source_manifest_shape(body["source_manifest"])
    runtime = _validate_runtime_manifest_shape(body["runtime_manifest"])
    _require(
        body["source_manifest_sha256"] == canonical_json_sha256(source),
        "source manifest digest differs",
    )
    _require(
        body["runtime_manifest_sha256"] == canonical_json_sha256(runtime),
        "runtime manifest digest differs",
    )
    if recheck_current:
        _require(
            _json_exact_equal(source, _build_source_manifest()),
            "source closure no longer matches the current checkout",
        )
        _require(
            _json_exact_equal(runtime, _capture_runtime_manifest()),
            "runtime/devices/dependencies no longer match the issued plan",
        )
    issuance = _expect_dict(body["issuance"], "plan.issuance")
    _expect_exact_keys(
        issuance,
        {
            "issued_unix",
            "kind",
            "prescribed_argv",
            "external_execution_attestation_present",
        },
        "plan.issuance",
    )
    _require(
        _is_int(issuance["issued_unix"])
        and 0 <= issuance["issued_unix"] <= int(time.time()) + 5,
        "issuance timestamp invalid",
    )
    _require(
        issuance["kind"] == "self_issued_development_plan_without_external_chronology",
        "issuance kind differs",
    )
    _require(
        _json_exact_equal(issuance["prescribed_argv"], commands["plan"]),
        "issuance prescribed argv differs",
    )
    _require(
        issuance["external_execution_attestation_present"] is False,
        "unavailable external attestation cannot be claimed",
    )
    _require(_is_sha256(plan_payload["plan_sha256"]), "plan digest is invalid")
    _require(
        plan_payload["plan_sha256"] == canonical_json_sha256(body),
        "plan digest does not match the plan body",
    )
    return plan_payload


def validate_plan_payload(
    payload: Mapping[str, object],
    *,
    locator: Path | None = None,
    recheck_current: bool = True,
) -> ContinualIAV2Validation:
    try:
        _require(
            recheck_current,
            "public plan validity requires current source/runtime binding verification",
        )
        _validate_plan_or_raise(payload, locator=locator, recheck_current=recheck_current)
    except Exception as exc:
        return ContinualIAV2Validation(False, False, (str(exc),))
    return ContinualIAV2Validation(True, False, ())


def load_plan(path: Path, *, recheck_current: bool = True) -> tuple[bytes, dict[str, Any]]:
    _require(
        recheck_current,
        "public plan loading requires current source/runtime binding verification",
    )
    return _load_plan(path, recheck_current=True)


def _load_plan(path: Path, *, recheck_current: bool) -> tuple[bytes, dict[str, Any]]:
    raw, payload = _read_canonical_json(path)
    validated = _validate_plan_or_raise(
        payload,
        locator=path,
        recheck_current=recheck_current,
    )
    final_raw, final_payload = _read_canonical_json(path)
    _require(raw == final_raw, f"plan bytes changed during validation: {path}")
    _require(
        _json_exact_equal(payload, final_payload),
        f"plan payload changed during validation: {path}",
    )
    if recheck_current:
        _validate_plan_or_raise(validated, locator=path, recheck_current=True)
    return raw, validated


def _trace_condition(result: IAConditionResult) -> dict[str, object]:
    return {
        "rewards": [float(value) for value in result.rewards],
        "executed_actions": [int(value) for value in result.executed_actions],
        "credited_actions": [int(value) for value in result.credited_actions],
        "pre_update_recommendations": [int(value) for value in result.recommendations],
        "pre_update_partner_proposals": [int(value) for value in result.partner_proposals],
        "accepted_recommendations": [bool(value) for value in result.accepted_recommendations],
    }


def _run_seed_trace(seed: int) -> dict[str, object]:
    """Execute one reserved v2 seed; called only by the explicit shard command."""

    _require(_is_int(seed) and seed in V2_EVIDENCE_SEEDS, "seed must be reserved v2 seed 60-89")
    report = run_continual_ia_benchmark(
        seeds=(seed,),
        config=V2_CONFIG,
        thresholds=V2_THRESHOLDS,
    )
    _require(report.config == V2_CONFIG, "seed runner returned a different v2 configuration")
    _require(report.thresholds == V2_THRESHOLDS, "seed runner returned different v2 thresholds")
    expected_conditions = (
        "partner_alone",
        "observe_only",
        "recommendation_p05",
        "accept_always",
        "augmented_predictions",
        "augmented_noise",
    )
    _require(
        len(report.condition_results) == len(expected_conditions),
        "seed runner must return exactly six condition results",
    )
    _require(
        tuple(result.condition for result in report.condition_results) == expected_conditions,
        "seed runner condition order/set differs from the frozen v2 adapter contract",
    )
    _require(
        all(result.seed == seed for result in report.condition_results),
        "seed runner returned a result for a different seed",
    )
    by_condition = {result.condition: result for result in report.condition_results}
    return {
        "schema": TRACE_SCHEMA,
        "semantics": dict(_TRACE_SEMANTICS),
        "conditions": {
            "partner_alone": _trace_condition(by_condition["partner_alone"]),
            "observe_only": _trace_condition(by_condition["observe_only"]),
            "recommendation_p075": _trace_condition(by_condition["recommendation_p05"]),
            "accept_always": _trace_condition(by_condition["accept_always"]),
            "augmented_predictions": _trace_condition(by_condition["augmented_predictions"]),
            "augmented_noise": _trace_condition(by_condition["augmented_noise"]),
        },
    }


def _numeric_vector(value: object, where: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    items = _expect_list(value, where)
    _require(len(items) == V2_CONFIG.num_steps, f"{where} must have 1200 entries")
    _require(all(_is_number(item) for item in items), f"{where} must be finite numeric data")
    array = np.asarray(items, dtype=np.float64)
    _require(bool(np.all((array == 0.0) | (array == 1.0))), f"{where} rewards must be binary")
    return array


def _integer_vector(
    value: object,
    where: str,
    *,
    allowed: set[int],
) -> np.ndarray[Any, np.dtype[np.int64]]:
    items = _expect_list(value, where)
    _require(len(items) == V2_CONFIG.num_steps, f"{where} must have 1200 entries")
    _require(
        all(_is_int(item) and item in allowed for item in items),
        f"{where} contains an invalid integer",
    )
    return np.asarray(items, dtype=np.int64)


def _boolean_vector(value: object, where: str) -> np.ndarray[Any, np.dtype[np.bool_]]:
    items = _expect_list(value, where)
    _require(len(items) == V2_CONFIG.num_steps, f"{where} must have 1200 entries")
    _require(all(type(item) is bool for item in items), f"{where} must contain booleans")
    return np.asarray(items, dtype=np.bool_)


def _validate_trace_or_raise(value: object) -> dict[str, Any]:
    trace = _expect_dict(value, "primitive_trace")
    _expect_exact_keys(trace, {"schema", "semantics", "conditions"}, "primitive_trace")
    _require(trace["schema"] == TRACE_SCHEMA, "primitive trace schema differs")
    _require(_json_exact_equal(trace["semantics"], _TRACE_SEMANTICS), "trace semantics differ")
    conditions = _expect_dict(trace["conditions"], "primitive_trace.conditions")
    _require(
        set(conditions) == set(V2_CONDITIONS),
        "trace conditions must contain exactly the six canonical v2 arms",
    )
    expected_fields = {
        "rewards",
        "executed_actions",
        "credited_actions",
        "pre_update_recommendations",
        "pre_update_partner_proposals",
        "accepted_recommendations",
    }
    for condition in V2_CONDITIONS:
        where = f"primitive_trace.conditions.{condition}"
        record = _expect_dict(conditions[condition], where)
        _expect_exact_keys(record, expected_fields, where)
        _numeric_vector(record["rewards"], f"{where}.rewards")
        actions = _integer_vector(
            record["executed_actions"], f"{where}.executed_actions", allowed={0, 1}
        )
        _integer_vector(record["credited_actions"], f"{where}.credited_actions", allowed={0, 1})
        recommendation_allowed = {0, 1} if condition in V2_RECOMMENDATION_CONDITIONS else {-1}
        recommendations = _integer_vector(
            record["pre_update_recommendations"],
            f"{where}.pre_update_recommendations",
            allowed=recommendation_allowed,
        )
        proposals = _integer_vector(
            record["pre_update_partner_proposals"],
            f"{where}.pre_update_partner_proposals",
            allowed=recommendation_allowed,
        )
        accepted = _boolean_vector(
            record["accepted_recommendations"], f"{where}.accepted_recommendations"
        )
        if condition in V2_RECOMMENDATION_CONDITIONS:
            expected_actions = np.where(accepted, recommendations, proposals)
            _require(
                np.array_equal(actions, expected_actions),
                f"{where} violates executed-action provenance",
            )
            _require(not bool(accepted[0]), f"{where} first acceptance must be false")
            _require(
                recommendations[0] == actions[0] and proposals[0] == actions[0],
                f"{where} first decision provenance differs",
            )
            if condition == "observe_only":
                _require(not bool(np.any(accepted)), "observe_only cannot accept recommendations")
            if condition == "accept_always":
                _require(
                    bool(np.all(accepted[1:])),
                    "accept_always must accept every executable post-initial recommendation",
                )
        else:
            _require(not bool(np.any(accepted)), f"{condition} cannot record acceptances")
    return trace


def validate_trace(value: object) -> ContinualIAV2Validation:
    try:
        _validate_trace_or_raise(value)
    except Exception as exc:
        return ContinualIAV2Validation(False, False, (str(exc),))
    return ContinualIAV2Validation(True, False, ())


def _bound_shard_entry(plan: Mapping[str, Any], seed: int) -> tuple[int, dict[str, Any]]:
    body = cast(dict[str, Any], plan["plan"])
    layout = cast(dict[str, Any], body["output_layout"])
    entries = cast(list[dict[str, Any]], layout["shards"])
    index = seed - V2_EVIDENCE_SEEDS[0]
    _require(0 <= index < len(entries), "seed is outside the v2 shard schedule")
    entry = entries[index]
    _require(entry["seed"] == seed, "bound shard seed differs")
    return index, entry


def _shard_command(plan: Mapping[str, Any], seed: int) -> list[str]:
    index, _entry = _bound_shard_entry(plan, seed)
    body = cast(dict[str, Any], plan["plan"])
    commands = cast(dict[str, Any], body["commands"])
    return cast(list[str], commands["shards"][index])


def _build_shard_payload(
    plan_payload: Mapping[str, object],
    seed: int,
    primitive_trace: Mapping[str, object],
    replay_trace: Mapping[str, object],
    *,
    recheck_current: bool = True,
) -> dict[str, object]:
    """Build one seed shard only after an exact second-run trace replay."""

    plan = _validate_plan_or_raise(
        plan_payload,
        locator=None,
        recheck_current=recheck_current,
    )
    _require(_is_int(seed) and seed in V2_EVIDENCE_SEEDS, "seed must be reserved v2 seed 60-89")
    trace = _validate_trace_or_raise(primitive_trace)
    replay = _validate_trace_or_raise(replay_trace)
    trace_digest = canonical_json_sha256(trace)
    replay_digest = canonical_json_sha256(replay)
    _require(
        trace_digest == replay_digest and _json_exact_equal(trace, replay),
        "deterministic replay differs from the first primitive trace",
    )
    plan_bytes = canonical_json_bytes(plan)
    body: dict[str, object] = {
        "namespace": NAMESPACE,
        "plan_binding": {
            "plan_file_sha256": sha256_bytes(plan_bytes),
            "plan_sha256": plan["plan_sha256"],
        },
        "seed": seed,
        "prescribed_argv": _shard_command(plan, seed),
        "primitive_trace": trace,
        "primitive_trace_sha256": trace_digest,
        "deterministic_replay": {
            "performed": True,
            "exact_match": True,
            "first_trace_sha256": trace_digest,
            "replay_trace_sha256": replay_digest,
        },
    }
    return {
        "schema": SHARD_SCHEMA,
        "namespace": NAMESPACE,
        "shard": body,
        "shard_sha256": canonical_json_sha256(body),
    }


def _validate_shard_or_raise(
    payload: Mapping[str, object],
    plan_payload: Mapping[str, object],
    *,
    locator: Path | None,
    recheck_current: bool,
) -> dict[str, Any]:
    plan = _validate_plan_or_raise(
        plan_payload,
        locator=None,
        recheck_current=recheck_current,
    )
    shard_payload = cast(dict[str, Any], dict(payload))
    _expect_exact_keys(
        shard_payload,
        {"schema", "namespace", "shard", "shard_sha256"},
        "shard payload",
    )
    _require(shard_payload["schema"] == SHARD_SCHEMA, "wrong v2 shard schema")
    _require(shard_payload["namespace"] == NAMESPACE, "wrong v2 shard namespace")
    body = _expect_dict(shard_payload["shard"], "shard")
    _expect_exact_keys(
        body,
        {
            "namespace",
            "plan_binding",
            "seed",
            "prescribed_argv",
            "primitive_trace",
            "primitive_trace_sha256",
            "deterministic_replay",
        },
        "shard",
    )
    _require(body["namespace"] == NAMESPACE, "shard body namespace differs")
    seed = body["seed"]
    _require(_is_int(seed) and seed in V2_EVIDENCE_SEEDS, "shard seed is outside 60-89")
    _index, entry = _bound_shard_entry(plan, seed)
    if locator is not None:
        _require(
            cast(str, entry["path"]) == _absolute_lexical(locator).as_posix(),
            "shard locator differs from its exact bound path",
        )
    binding = _expect_dict(body["plan_binding"], "shard.plan_binding")
    _expect_exact_keys(binding, {"plan_file_sha256", "plan_sha256"}, "shard.plan_binding")
    _require(
        binding["plan_file_sha256"] == sha256_bytes(canonical_json_bytes(plan)),
        "shard plan-file binding differs",
    )
    _require(binding["plan_sha256"] == plan["plan_sha256"], "shard plan digest differs")
    _require(
        _json_exact_equal(body["prescribed_argv"], _shard_command(plan, seed)),
        "shard prescribed argv differs",
    )
    trace = _validate_trace_or_raise(body["primitive_trace"])
    trace_digest = canonical_json_sha256(trace)
    _require(_is_sha256(body["primitive_trace_sha256"]), "trace digest is invalid")
    _require(body["primitive_trace_sha256"] == trace_digest, "trace digest differs")
    replay = _expect_dict(body["deterministic_replay"], "shard.deterministic_replay")
    _expect_exact_keys(
        replay,
        {
            "performed",
            "exact_match",
            "first_trace_sha256",
            "replay_trace_sha256",
        },
        "shard.deterministic_replay",
    )
    _require(replay["performed"] is True, "deterministic replay was not performed")
    _require(replay["exact_match"] is True, "deterministic replay did not match")
    _require(
        replay["first_trace_sha256"] == trace_digest
        and replay["replay_trace_sha256"] == trace_digest,
        "deterministic replay digests differ from the primitive trace",
    )
    _require(_is_sha256(shard_payload["shard_sha256"]), "shard body digest is invalid")
    _require(
        shard_payload["shard_sha256"] == canonical_json_sha256(body),
        "shard body digest differs",
    )
    return shard_payload


def _replay_shard_or_raise(
    shard_payload: Mapping[str, Any],
    replay_runner: Callable[[int], dict[str, object]],
) -> None:
    body = cast(dict[str, Any], shard_payload["shard"])
    seed = cast(int, body["seed"])
    recorded = cast(dict[str, Any], body["primitive_trace"])
    try:
        replay = _validate_trace_or_raise(replay_runner(seed))
    except Exception as exc:
        raise ContinualIAV2Error(f"computational replay failed for seed {seed}: {exc}") from exc
    _require(
        _json_exact_equal(recorded, replay)
        and canonical_json_sha256(recorded) == canonical_json_sha256(replay),
        f"recorded primitive trace differs from exact computational replay for seed {seed}",
    )


def validate_shard_payload(
    payload: Mapping[str, object],
    plan_payload: Mapping[str, object],
    *,
    locator: Path | None = None,
    recheck_current: bool = True,
) -> ContinualIAV2Validation:
    try:
        _require(
            recheck_current,
            "public shard validity requires current source/runtime binding verification",
        )
        validated = _validate_shard_or_raise(
            payload,
            plan_payload,
            locator=locator,
            recheck_current=recheck_current,
        )
        _replay_shard_or_raise(validated, _run_seed_trace)
        if recheck_current:
            _validate_plan_or_raise(
                plan_payload,
                locator=None,
                recheck_current=True,
            )
    except Exception as exc:
        return ContinualIAV2Validation(False, False, (str(exc),))
    return ContinualIAV2Validation(True, False, ())


def write_shard(
    plan_path: Path,
    seed: int,
    output_path: Path,
) -> Path:
    """Run one seed twice and publish its exact bound shard if replay matches."""

    return _write_shard(plan_path, seed, output_path, runner=_run_seed_trace)


def _reservation_path(output_path: Path) -> Path:
    absolute = _absolute_lexical(output_path)
    return absolute.with_name(f"{absolute.name}.reservation")


def _build_reservation_payload(
    plan: Mapping[str, Any],
    plan_raw: bytes,
    seed: int,
    output_path: Path,
) -> dict[str, object]:
    reservation_path = _reservation_path(output_path)
    body: dict[str, object] = {
        "role": "development_seed_consumption_marker",
        "state": "execution_started_seed_irrevocably_consumed",
        "seed": seed,
        "target_path": _absolute_lexical(output_path).as_posix(),
        "reservation_path": reservation_path.as_posix(),
        "plan_file_sha256": sha256_bytes(plan_raw),
        "plan_sha256": plan["plan_sha256"],
        "prescribed_argv": _shard_command(plan, seed),
        "reserved_unix": int(time.time()),
        "external_chronology_attestation_present": False,
    }
    return {
        "schema": RESERVATION_SCHEMA,
        "namespace": NAMESPACE,
        "reservation": body,
        "reservation_sha256": canonical_json_sha256(body),
    }


def _validate_reservation_or_raise(
    payload: Mapping[str, object],
    plan: Mapping[str, Any],
    plan_raw: bytes,
    seed: int,
    output_path: Path,
    *,
    locator: Path,
) -> dict[str, Any]:
    reservation_payload = cast(dict[str, Any], dict(payload))
    _expect_exact_keys(
        reservation_payload,
        {"schema", "namespace", "reservation", "reservation_sha256"},
        "reservation payload",
    )
    _require(
        reservation_payload["schema"] == RESERVATION_SCHEMA,
        "wrong v2 reservation schema",
    )
    _require(
        reservation_payload["namespace"] == NAMESPACE,
        "wrong v2 reservation namespace",
    )
    body = _expect_dict(reservation_payload["reservation"], "reservation")
    _expect_exact_keys(
        body,
        {
            "role",
            "state",
            "seed",
            "target_path",
            "reservation_path",
            "plan_file_sha256",
            "plan_sha256",
            "prescribed_argv",
            "reserved_unix",
            "external_chronology_attestation_present",
        },
        "reservation",
    )
    expected_output = _absolute_lexical(output_path)
    expected_reservation = _reservation_path(expected_output)
    _require(
        _absolute_lexical(locator) == expected_reservation,
        "reservation locator differs from the bound shard reservation path",
    )
    _require(
        body["role"] == "development_seed_consumption_marker"
        and body["state"] == "execution_started_seed_irrevocably_consumed",
        "reservation role/state differs",
    )
    _require(body["seed"] == seed, "reservation seed differs")
    _require(
        body["target_path"] == expected_output.as_posix()
        and body["reservation_path"] == expected_reservation.as_posix(),
        "reservation target or locator differs",
    )
    _require(
        body["plan_file_sha256"] == sha256_bytes(plan_raw)
        and body["plan_sha256"] == plan["plan_sha256"],
        "reservation plan binding differs",
    )
    _require(
        _json_exact_equal(body["prescribed_argv"], _shard_command(plan, seed)),
        "reservation prescribed argv differs",
    )
    plan_body = cast(dict[str, Any], plan["plan"])
    issuance = cast(dict[str, Any], plan_body["issuance"])
    reserved_unix = body["reserved_unix"]
    _require(
        _is_int(reserved_unix)
        and issuance["issued_unix"] <= reserved_unix <= int(time.time()) + 5,
        "reservation timestamp invalid",
    )
    _require(
        body["external_chronology_attestation_present"] is False,
        "unavailable reservation chronology cannot be claimed",
    )
    _require(
        _is_sha256(reservation_payload["reservation_sha256"])
        and reservation_payload["reservation_sha256"] == canonical_json_sha256(body),
        "reservation digest differs",
    )
    return reservation_payload


def _write_shard(
    plan_path: Path,
    seed: int,
    output_path: Path,
    *,
    runner: Callable[[int], dict[str, object]],
) -> Path:
    _require(_is_int(seed) and seed in V2_EVIDENCE_SEEDS, "seed must be reserved v2 seed 60-89")
    plan_raw, plan = load_plan(plan_path, recheck_current=True)
    _index, entry = _bound_shard_entry(plan, seed)
    _require(
        _absolute_lexical(output_path).as_posix() == entry["path"],
        "requested shard output is not the exact plan-bound path",
    )
    output = _preflight_new_output(output_path)
    reservation_path = _reservation_path(output)
    _preflight_new_output(reservation_path)
    reservation = _build_reservation_payload(plan, plan_raw, seed, output)
    reservation_raw = canonical_json_bytes(reservation)
    _atomic_publish_new(reservation_path, reservation_raw)
    _preflight_new_output(output)
    try:
        first = runner(seed)
        replay = runner(seed)
    except Exception as exc:
        raise ContinualIAV2Error(
            f"seed execution failed after reservation for seed {seed}: {exc}"
        ) from exc
    payload = _build_shard_payload(plan, seed, first, replay, recheck_current=True)
    _validate_shard_or_raise(
        payload,
        plan,
        locator=output_path,
        recheck_current=True,
    )
    final_plan_raw, final_plan = _load_plan(plan_path, recheck_current=True)
    _require(
        plan_raw == final_plan_raw and _json_exact_equal(plan, final_plan),
        "plan bytes changed during shard execution",
    )
    final_reservation_raw, final_reservation = _read_canonical_json(reservation_path)
    _require(
        reservation_raw == final_reservation_raw
        and _json_exact_equal(reservation, final_reservation),
        "seed reservation changed during shard execution",
    )
    _validate_reservation_or_raise(
        final_reservation,
        plan,
        plan_raw,
        seed,
        output,
        locator=reservation_path,
    )
    _validate_plan_or_raise(plan, locator=None, recheck_current=True)
    return _atomic_publish_new_json(output, payload)


def load_shard(
    path: Path,
    plan_payload: Mapping[str, object],
    *,
    recheck_current: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    _require(
        recheck_current,
        "public shard loading requires current source/runtime binding verification",
    )
    validated_plan = _validate_plan_or_raise(
        plan_payload,
        locator=None,
        recheck_current=True,
    )
    plan_body = cast(dict[str, Any], validated_plan["plan"])
    layout = cast(dict[str, Any], plan_body["output_layout"])
    external_plan_path = Path(cast(str, layout["plan_path"]))
    external_plan_raw, external_plan = _load_plan(
        external_plan_path,
        recheck_current=True,
    )
    _require(
        _json_exact_equal(validated_plan, external_plan),
        "supplied shard plan differs from the external plan file",
    )
    shard_body = cast(dict[str, Any], validated_plan["plan"])
    layout = cast(dict[str, Any], shard_body["output_layout"])
    entries = cast(list[dict[str, Any]], layout["shards"])
    requested = _absolute_lexical(path).as_posix()
    matching = [entry for entry in entries if entry["path"] == requested]
    _require(len(matching) == 1, "shard locator is not in the exact bound schedule")
    entry = matching[0]
    seed = cast(int, entry["seed"])
    reservation_path = _reservation_path(path)
    reservation_raw, reservation_payload = _read_canonical_json(reservation_path)
    _validate_reservation_or_raise(
        reservation_payload,
        external_plan,
        external_plan_raw,
        seed,
        path,
        locator=reservation_path,
    )
    raw, shard = _load_shard(
        path,
        validated_plan,
        recheck_current=recheck_current,
        replay_runner=_run_seed_trace,
    )
    final_plan_raw, final_plan = _load_plan(external_plan_path, recheck_current=True)
    _require(
        external_plan_raw == final_plan_raw
        and _json_exact_equal(external_plan, final_plan),
        "external plan changed during shard validation",
    )
    final_reservation_raw, final_reservation = _read_canonical_json(reservation_path)
    _require(
        reservation_raw == final_reservation_raw
        and _json_exact_equal(reservation_payload, final_reservation),
        "external reservation changed during shard validation",
    )
    return raw, shard


def _load_shard(
    path: Path,
    plan_payload: Mapping[str, object],
    *,
    recheck_current: bool,
    replay_runner: Callable[[int], dict[str, object]] | None,
) -> tuple[bytes, dict[str, Any]]:
    raw, payload = _read_canonical_json(path)
    validated = _validate_shard_or_raise(
        payload,
        plan_payload,
        locator=path,
        recheck_current=recheck_current,
    )
    if replay_runner is not None:
        _replay_shard_or_raise(validated, replay_runner)
    if recheck_current:
        _validate_plan_or_raise(
            plan_payload,
            locator=None,
            recheck_current=True,
        )
    final_raw, final_payload = _read_canonical_json(path)
    _require(raw == final_raw, f"shard bytes changed during validation: {path}")
    _require(
        _json_exact_equal(payload, final_payload),
        f"shard payload changed during validation: {path}",
    )
    if recheck_current:
        _validate_plan_or_raise(
            plan_payload,
            locator=None,
            recheck_current=True,
        )
    return raw, validated


def _condition_arrays(trace: Mapping[str, Any], condition: str) -> tuple[np.ndarray[Any, Any], ...]:
    record = cast(dict[str, Any], cast(dict[str, Any], trace["conditions"])[condition])
    rewards = np.asarray(record["rewards"], dtype=np.float64)
    actions = np.asarray(record["executed_actions"], dtype=np.int64)
    credits = np.asarray(record["credited_actions"], dtype=np.int64)
    recommendations = np.asarray(record["pre_update_recommendations"], dtype=np.int64)
    proposals = np.asarray(record["pre_update_partner_proposals"], dtype=np.int64)
    accepted = np.asarray(record["accepted_recommendations"], dtype=np.bool_)
    return rewards, actions, credits, recommendations, proposals, accepted


def _per_seed_result(seed: int, trace: Mapping[str, Any]) -> dict[str, object]:
    conditions: dict[str, object] = {}
    for condition in V2_CONDITIONS:
        rewards, actions, credits, recommendations, proposals, accepted = _condition_arrays(
            trace, condition
        )
        changed = accepted & (recommendations != proposals)
        conditions[condition] = {
            "mean_reward": float(np.mean(rewards)),
            "phase_mean_rewards": [float(value) for value in _phase_means(rewards, V2_CONFIG)],
            "recovery_lengths": [int(value) for value in _recovery_lengths(rewards, V2_CONFIG)],
            "executed_accepted_recommendations": int(np.count_nonzero(accepted)),
            "action_changing_interventions": int(np.count_nonzero(changed)),
            "changed_action_intervention_rate": float(np.mean(changed)),
            "executed_action_credit_mismatches": int(np.count_nonzero(actions != credits)),
        }
    return {"seed": seed, "conditions": conditions}


def _interval_payload(
    values: np.ndarray[Any, np.dtype[np.float64]], offset: int
) -> dict[str, object]:
    interval = paired_bootstrap_mean_interval(
        values,
        confidence_level=V2_CONFIG.confidence_level,
        resamples=V2_CONFIG.bootstrap_resamples,
        seed=(V2_CONFIG.bootstrap_seed + offset) % 2**32,
    )
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "confidence_level": interval.confidence_level,
        "resamples": interval.resamples,
        "sample_size": interval.sample_size,
        "method": interval.method,
        "pairing_unit": "seed",
    }


def _aggregate(
    seed_results: Sequence[Mapping[str, Any]], traces: Sequence[Mapping[str, Any]]
) -> dict[str, object]:
    _require(len(seed_results) == 30 and len(traces) == 30, "aggregate requires thirty seeds")
    means = {
        condition: np.asarray(
            [result["conditions"][condition]["mean_reward"] for result in seed_results],
            dtype=np.float64,
        )
        for condition in V2_CONDITIONS
    }
    primary = means["recommendation_p075"] - means["observe_only"]
    accept_always = means["accept_always"] - means["observe_only"]
    augmentation_alone = means["augmented_predictions"] - means["partner_alone"]
    augmentation_noise = means["augmented_predictions"] - means["augmented_noise"]
    alone_rewards: list[np.ndarray[Any, Any]] = []
    observe_rewards: list[np.ndarray[Any, Any]] = []
    alone_actions: list[np.ndarray[Any, Any]] = []
    observe_actions: list[np.ndarray[Any, Any]] = []
    for trace in traces:
        alone = _condition_arrays(trace, "partner_alone")
        observe = _condition_arrays(trace, "observe_only")
        alone_rewards.append(alone[0])
        observe_rewards.append(observe[0])
        alone_actions.append(alone[1])
        observe_actions.append(observe[1])
    budgets = _budget_payload()
    treatment_results = [result["conditions"]["recommendation_p075"] for result in seed_results]
    credit_mismatches = sum(
        int(result["conditions"][condition]["executed_action_credit_mismatches"])
        for result in seed_results
        for condition in V2_CONDITIONS
        if condition != "partner_alone"
    )
    phase_means = {
        condition: [
            float(value)
            for value in np.mean(
                np.asarray(
                    [
                        result["conditions"][condition]["phase_mean_rewards"]
                        for result in seed_results
                    ],
                    dtype=np.float64,
                ),
                axis=0,
            )
        ]
        for condition in V2_CONDITIONS
    }
    recovery: dict[str, object] = {}
    for condition in V2_CONDITIONS:
        values = np.asarray(
            [
                value
                for result in seed_results
                for value in result["conditions"][condition]["recovery_lengths"]
            ],
            dtype=np.int64,
        )
        recovered = values >= 0
        recovery[condition] = {
            "fraction": float(np.mean(recovered)),
            "mean_steps": float(np.mean(values[recovered])) if np.any(recovered) else None,
        }
    all_numbers = np.concatenate([values for values in means.values()])
    return {
        "seeds": list(V2_EVIDENCE_SEEDS),
        "seed_count": 30,
        "mean_rewards": {
            condition: float(np.mean(means[condition])) for condition in V2_CONDITIONS
        },
        "primary_uplift": _interval_payload(primary, 0),
        "accept_always_uplift": _interval_payload(accept_always, 1),
        "augmentation_vs_alone": _interval_payload(augmentation_alone, 2),
        "augmentation_vs_noise": _interval_payload(augmentation_noise, 3),
        "mean_changed_action_intervention_rate": float(
            np.mean(
                np.asarray(
                    [result["changed_action_intervention_rate"] for result in treatment_results],
                    dtype=np.float64,
                )
            )
        ),
        "total_action_changing_interventions": sum(
            int(result["action_changing_interventions"]) for result in treatment_results
        ),
        "observe_only_exact_reward_identity": all(
            np.array_equal(left, right)
            for left, right in zip(alone_rewards, observe_rewards, strict=True)
        ),
        "observe_only_exact_action_identity": all(
            np.array_equal(left, right)
            for left, right in zip(alone_actions, observe_actions, strict=True)
        ),
        "executed_action_credit_mismatches": credit_mismatches,
        "primary_state_budget_matched": (budgets["recommendation_p075"] == budgets["observe_only"]),
        "primary_interaction_budget_matched": (
            budgets["recommendation_p075"]["interaction_steps"]
            == budgets["observe_only"]["interaction_steps"]
            and budgets["recommendation_p075"]["action_scalars_per_step"]
            == budgets["observe_only"]["action_scalars_per_step"]
        ),
        "augmentation_noise_state_budget_matched": (
            budgets["augmented_predictions"] == budgets["augmented_noise"]
        ),
        "augmentation_state_bytes_above_alone": (
            cast(int, budgets["augmented_predictions"]["state_bytes"])
            - cast(int, budgets["partner_alone"]["state_bytes"])
        ),
        "condition_budgets": budgets,
        "condition_phase_mean_rewards": phase_means,
        "condition_recovery": recovery,
        "all_values_finite": bool(np.all(np.isfinite(all_numbers))),
    }


def _check(
    name: str,
    scope: str,
    actual: float,
    comparator: str,
    threshold: float,
    detail: str,
) -> dict[str, object]:
    finite = math.isfinite(actual) and math.isfinite(threshold)
    passed = finite and (actual >= threshold if comparator == ">=" else actual <= threshold)
    return {
        "name": name,
        "scope": scope,
        "passed": bool(passed),
        "actual": actual,
        "comparator": comparator,
        "threshold": threshold,
        "detail": detail,
    }


def _acceptance(aggregate: Mapping[str, Any]) -> dict[str, object]:
    identity = bool(
        aggregate["observe_only_exact_reward_identity"]
        and aggregate["observe_only_exact_action_identity"]
    )
    primary_interval = cast(dict[str, float], aggregate["primary_uplift"])
    augmentation_alone = cast(dict[str, float], aggregate["augmentation_vs_alone"])
    augmentation_noise = cast(dict[str, float], aggregate["augmentation_vs_noise"])
    checks = [
        _check(
            "seed_count",
            "primary",
            float(aggregate["seed_count"]),
            ">=",
            30.0,
            "Exactly thirty paired seeds are required.",
        ),
        _check(
            "evidence_seed_schedule",
            "primary",
            float(aggregate["seeds"] == list(V2_EVIDENCE_SEEDS)),
            ">=",
            1.0,
            "Evidence must use exactly newly reserved seeds 60-89.",
        ),
        _check(
            "all_values_finite",
            "primary",
            float(aggregate["all_values_finite"]),
            ">=",
            1.0,
            "Every scientific value must be finite.",
        ),
        _check(
            "observe_only_exact_identity",
            "primary",
            float(identity),
            ">=",
            1.0,
            "Observe-only rewards/actions must equal partner-alone bitwise.",
        ),
        _check(
            "primary_state_budget_matched",
            "primary",
            float(aggregate["primary_state_budget_matched"]),
            ">=",
            1.0,
            "p=.75 treatment and p=0 control must have identical state budgets.",
        ),
        _check(
            "primary_interaction_budget_matched",
            "primary",
            float(aggregate["primary_interaction_budget_matched"]),
            ">=",
            1.0,
            "p=.75 treatment and p=0 control must have identical interaction budgets.",
        ),
        _check(
            "executed_action_credit_mismatches",
            "primary",
            float(aggregate["executed_action_credit_mismatches"]),
            "<=",
            0.0,
            "Every IA update must credit the executed primitive action.",
        ),
        _check(
            "primary_recommendation_uplift",
            "primary",
            float(primary_interval["lower"]),
            ">=",
            0.10,
            "Lower paired 95% CI for recommendation_p075 minus observe_only.",
        ),
        _check(
            "changed_action_intervention_rate",
            "primary",
            float(aggregate["mean_changed_action_intervention_rate"]),
            ">=",
            0.10,
            "Mean primitive-step rate of accepted action-changing p=.75 recommendations.",
        ),
        _check(
            "augmentation_noise_state_budget_matched",
            "secondary",
            float(aggregate["augmentation_noise_state_budget_matched"]),
            ">=",
            1.0,
            "Prediction/noise augmentation controls must have equal state budgets.",
        ),
        _check(
            "augmentation_state_budget_above_alone",
            "secondary",
            float(aggregate["augmentation_state_bytes_above_alone"]),
            ">=",
            1.0,
            "Equal-budget augmentation arms must exceed partner-alone state bytes.",
        ),
        _check(
            "augmentation_vs_alone",
            "secondary",
            float(augmentation_alone["lower"]),
            ">=",
            0.05,
            "Lower paired 95% CI for prediction augmentation minus partner-alone.",
        ),
        _check(
            "augmentation_vs_noise",
            "secondary",
            float(augmentation_noise["lower"]),
            ">=",
            0.05,
            "Lower paired 95% CI for predictions minus same-shape noise.",
        ),
    ]
    primary_passed = all(bool(check["passed"]) for check in checks if check["scope"] == "primary")
    secondary_passed = all(
        bool(check["passed"]) for check in checks if check["scope"] == "secondary"
    )
    scientific_gates_passed = primary_passed and secondary_passed
    return {
        "internally_accepted": False,
        "scientific_gates_passed": scientific_gates_passed,
        "chronology_attestation_present": False,
        "primary_passed": primary_passed,
        "secondary_passed": secondary_passed,
        "checks": checks,
        "interpretation": {
            "if_accepted": (
                "not available in self-issued v2; an externally anchored future protocol "
                "is required"
            ),
            "if_scientific_gates_pass": (
                "reproducible development-only diagnostic, not held-out or preregistered evidence"
            ),
            "independent_replication": False,
            "general_step12": False,
            "state_of_the_art": False,
        },
    }


def _artifact_from_validated_shards(
    plan: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    _require(len(shards) == 30, "artifact requires exactly thirty shards")
    observed_seeds = [cast(dict[str, Any], shard["shard"])["seed"] for shard in shards]
    _require(tuple(observed_seeds) == V2_EVIDENCE_SEEDS, "artifact shard order must be 60-89")
    traces = [
        cast(dict[str, Any], cast(dict[str, Any], shard["shard"])["primitive_trace"])
        for shard in shards
    ]
    seed_results = [
        _per_seed_result(seed, trace) for seed, trace in zip(V2_EVIDENCE_SEEDS, traces, strict=True)
    ]
    aggregate = _aggregate(seed_results, traces)
    acceptance = _acceptance(aggregate)
    plan_bytes = canonical_json_bytes(plan)
    shard_manifest = [
        {
            "seed": seed,
            "byte_size": len(canonical_json_bytes(shard)),
            "sha256": sha256_bytes(canonical_json_bytes(shard)),
        }
        for seed, shard in zip(V2_EVIDENCE_SEEDS, shards, strict=True)
    ]
    body: dict[str, object] = {
        "namespace": NAMESPACE,
        "plan": dict(plan),
        "plan_file_sha256": sha256_bytes(plan_bytes),
        "shard_manifest": shard_manifest,
        "merge_replay_manifest": [
            {
                "seed": seed,
                "exact_match": True,
                "primitive_trace_sha256": cast(dict[str, Any], shard["shard"])[
                    "primitive_trace_sha256"
                ],
            }
            for seed, shard in zip(V2_EVIDENCE_SEEDS, shards, strict=True)
        ],
        "shards": [dict(shard) for shard in shards],
        "per_seed_results": seed_results,
        "aggregate": aggregate,
        "acceptance": acceptance,
        "claim_scope": _claim_scope(),
    }
    return {
        "schema": ARTIFACT_SCHEMA,
        "namespace": NAMESPACE,
        "artifact": body,
        "artifact_sha256": canonical_json_sha256(body),
    }


def _build_artifact_payload(
    plan_payload: Mapping[str, object],
    shard_payloads: Sequence[Mapping[str, object]],
    *,
    recheck_current: bool = True,
) -> dict[str, object]:
    """Build a standalone artifact after validating exact shard coverage."""

    plan = _validate_plan_or_raise(
        plan_payload,
        locator=None,
        recheck_current=recheck_current,
    )
    _require(len(shard_payloads) == 30, "merge requires exactly thirty shard payloads")
    validated: list[dict[str, Any]] = []
    for expected_seed, payload in zip(V2_EVIDENCE_SEEDS, shard_payloads, strict=True):
        shard = _validate_shard_or_raise(
            payload,
            plan,
            locator=None,
            recheck_current=False,
        )
        _require(
            cast(dict[str, Any], shard["shard"])["seed"] == expected_seed,
            "merge shard coverage must be exact, ordered, and duplicate-free",
        )
        validated.append(shard)
    return _artifact_from_validated_shards(plan, validated)


def _validate_artifact_or_raise(
    payload: Mapping[str, object],
    *,
    locator: Path | None,
    recheck_current: bool,
    replay_runner: Callable[[int], dict[str, object]] | None = None,
) -> tuple[dict[str, Any], bool]:
    artifact_payload = cast(dict[str, Any], dict(payload))
    _expect_exact_keys(
        artifact_payload,
        {"schema", "namespace", "artifact", "artifact_sha256"},
        "artifact payload",
    )
    _require(artifact_payload["schema"] == ARTIFACT_SCHEMA, "wrong v2 artifact schema")
    _require(artifact_payload["namespace"] == NAMESPACE, "wrong v2 artifact namespace")
    body = _expect_dict(artifact_payload["artifact"], "artifact")
    _expect_exact_keys(
        body,
        {
            "namespace",
            "plan",
            "plan_file_sha256",
            "shard_manifest",
            "merge_replay_manifest",
            "shards",
            "per_seed_results",
            "aggregate",
            "acceptance",
            "claim_scope",
        },
        "artifact",
    )
    _require(body["namespace"] == NAMESPACE, "artifact body namespace differs")
    plan = _validate_plan_or_raise(
        _expect_dict(body["plan"], "artifact.plan"),
        locator=None,
        recheck_current=recheck_current,
    )
    plan_body = cast(dict[str, Any], plan["plan"])
    layout = cast(dict[str, Any], plan_body["output_layout"])
    if locator is not None:
        _require(
            _absolute_lexical(locator).as_posix() == layout["artifact_path"],
            "artifact locator differs from the plan-bound path",
        )
    _require(
        body["plan_file_sha256"] == sha256_bytes(canonical_json_bytes(plan)),
        "artifact plan-file digest differs",
    )
    raw_shards = _expect_list(body["shards"], "artifact.shards")
    _require(len(raw_shards) == 30, "artifact must embed exactly thirty shards")
    validated_shards: list[dict[str, Any]] = []
    for expected_seed, raw_shard in zip(V2_EVIDENCE_SEEDS, raw_shards, strict=True):
        shard = _validate_shard_or_raise(
            _expect_dict(raw_shard, f"artifact.shards[{expected_seed - 60}]"),
            plan,
            locator=None,
            recheck_current=False,
        )
        _require(
            cast(dict[str, Any], shard["shard"])["seed"] == expected_seed,
            "artifact shard coverage is not exactly ordered seeds 60-89",
        )
        validated_shards.append(shard)
    external_inputs: list[tuple[Path, bytes, dict[str, Any]]] = []
    if locator is not None:
        external_plan_path = Path(cast(str, layout["plan_path"]))
        external_plan_raw, external_plan_payload = _read_canonical_json(external_plan_path)
        external_plan = _validate_plan_or_raise(
            external_plan_payload,
            locator=external_plan_path,
            recheck_current=recheck_current,
        )
        _require(
            _json_exact_equal(plan, external_plan)
            and body["plan_file_sha256"] == sha256_bytes(external_plan_raw),
            "embedded artifact plan differs from the external plan file",
        )
        external_inputs.append((external_plan_path, external_plan_raw, external_plan_payload))
        for index, (entry, embedded_shard) in enumerate(
            zip(
                cast(list[dict[str, Any]], layout["shards"]),
                validated_shards,
                strict=True,
            )
        ):
            shard_path = Path(cast(str, entry["path"]))
            external_raw, external_payload = _read_canonical_json(shard_path)
            external_shard = _validate_shard_or_raise(
                external_payload,
                plan,
                locator=shard_path,
                recheck_current=False,
            )
            _require(
                _json_exact_equal(embedded_shard, external_shard),
                f"embedded artifact shard differs from external shard {index}",
            )
            external_inputs.append((shard_path, external_raw, external_payload))
            seed = cast(int, entry["seed"])
            reservation_path = _reservation_path(shard_path)
            reservation_raw, reservation_payload = _read_canonical_json(reservation_path)
            _validate_reservation_or_raise(
                reservation_payload,
                external_plan,
                external_plan_raw,
                seed,
                shard_path,
                locator=reservation_path,
            )
            external_inputs.append(
                (reservation_path, reservation_raw, reservation_payload)
            )
    recomputed = _artifact_from_validated_shards(plan, validated_shards)
    recomputed_body = cast(dict[str, Any], recomputed["artifact"])
    for field in (
        "shard_manifest",
        "merge_replay_manifest",
        "per_seed_results",
        "aggregate",
        "acceptance",
        "claim_scope",
    ):
        _require(
            _json_exact_equal(body[field], recomputed_body[field]),
            f"artifact.{field} differs from primitive shard recomputation",
        )
    if replay_runner is not None:
        for shard in validated_shards:
            _replay_shard_or_raise(shard, replay_runner)
    _require(_is_sha256(artifact_payload["artifact_sha256"]), "artifact digest invalid")
    _require(
        artifact_payload["artifact_sha256"] == canonical_json_sha256(body),
        "artifact digest differs",
    )
    if recheck_current:
        _validate_plan_or_raise(
            plan,
            locator=None,
            recheck_current=True,
        )
    for path, expected_raw, expected_payload in external_inputs:
        final_raw, final_payload = _read_canonical_json(path)
        _require(expected_raw == final_raw, f"external lifecycle bytes changed: {path}")
        _require(
            _json_exact_equal(expected_payload, final_payload),
            f"external lifecycle payload changed: {path}",
        )
    if recheck_current:
        _validate_plan_or_raise(
            plan,
            locator=None,
            recheck_current=True,
        )
    acceptance = _expect_dict(body["acceptance"], "artifact.acceptance")
    return artifact_payload, bool(acceptance["internally_accepted"])


def validate_artifact_payload(
    payload: Mapping[str, object],
    *,
    locator: Path | None = None,
    recheck_current: bool = True,
) -> ContinualIAV2Validation:
    try:
        _require(
            recheck_current,
            "public artifact validity requires current source/runtime binding verification",
        )
        _validated, accepted = _validate_artifact_or_raise(
            payload,
            locator=locator,
            recheck_current=recheck_current,
            replay_runner=_run_seed_trace,
        )
    except Exception as exc:
        return ContinualIAV2Validation(False, False, (str(exc),))
    return ContinualIAV2Validation(True, accepted, ())


def merge_shards(plan_path: Path, output_path: Path) -> Path:
    """Read the exact 30 bound shards, recompute everything, and publish once."""

    output, _validation = _merge_shards(
        plan_path,
        output_path,
        replay_runner=_run_seed_trace,
        recheck_current=True,
    )
    return output


def merge_shards_with_validation(
    plan_path: Path,
    output_path: Path,
) -> tuple[Path, ContinualIAV2Validation]:
    """Merge once and return its already-computed decision without replaying again."""

    return _merge_shards(
        plan_path,
        output_path,
        replay_runner=_run_seed_trace,
        recheck_current=True,
    )


def _merge_shards_for_testing(
    plan_path: Path,
    output_path: Path,
    *,
    replay_runner: Callable[[int], dict[str, object]],
) -> Path:
    """Synthetic-fixture seam; it cannot be reached from the public v2 CLI."""

    output, _validation = _merge_shards(
        plan_path,
        output_path,
        replay_runner=replay_runner,
        recheck_current=False,
    )
    return output


def _merge_shards(
    plan_path: Path,
    output_path: Path,
    *,
    replay_runner: Callable[[int], dict[str, object]],
    recheck_current: bool,
) -> tuple[Path, ContinualIAV2Validation]:
    plan_raw, plan = _load_plan(plan_path, recheck_current=recheck_current)
    plan_body = cast(dict[str, Any], plan["plan"])
    layout = cast(dict[str, Any], plan_body["output_layout"])
    _require(
        _absolute_lexical(output_path).as_posix() == layout["artifact_path"],
        "requested artifact output is not the exact plan-bound path",
    )
    output = _preflight_new_output(output_path)
    entries = cast(list[dict[str, Any]], layout["shards"])
    shards: list[dict[str, Any]] = []
    raw_manifest: list[tuple[int, int, str]] = []
    lifecycle_inputs: list[tuple[Path, bytes, dict[str, Any]]] = []
    for entry in entries:
        seed = cast(int, entry["seed"])
        shard_path = Path(cast(str, entry["path"]))
        reservation_path = _reservation_path(shard_path)
        reservation_raw, reservation_payload = _read_canonical_json(reservation_path)
        _validate_reservation_or_raise(
            reservation_payload,
            plan,
            plan_raw,
            seed,
            shard_path,
            locator=reservation_path,
        )
        lifecycle_inputs.append(
            (reservation_path, reservation_raw, reservation_payload)
        )
        raw, shard = _load_shard(
            shard_path,
            plan,
            recheck_current=False,
            replay_runner=None,
        )
        _replay_shard_or_raise(shard, replay_runner)
        shards.append(shard)
        raw_manifest.append((seed, len(raw), sha256_bytes(raw)))
        lifecycle_inputs.append((shard_path, raw, shard))
    artifact = _build_artifact_payload(plan, shards, recheck_current=recheck_current)
    body = cast(dict[str, Any], artifact["artifact"])
    manifest = cast(list[dict[str, Any]], body["shard_manifest"])
    _require(
        raw_manifest == [(item["seed"], item["byte_size"], item["sha256"]) for item in manifest],
        "artifact shard byte manifest differs from descriptor-anchored input bytes",
    )
    _validated, accepted = _validate_artifact_or_raise(
        artifact,
        locator=output,
        recheck_current=recheck_current,
    )
    final_plan_raw, final_plan = _load_plan(plan_path, recheck_current=recheck_current)
    _require(
        plan_raw == final_plan_raw and _json_exact_equal(plan, final_plan),
        "plan bytes changed during merge replay",
    )
    for input_path, expected_raw, expected_payload in lifecycle_inputs:
        final_raw, final_payload = _read_canonical_json(input_path)
        _require(
            expected_raw == final_raw,
            f"lifecycle input bytes changed during merge: {input_path}",
        )
        _require(
            _json_exact_equal(expected_payload, final_payload),
            f"lifecycle input payload changed during merge: {input_path}",
        )
    if recheck_current:
        _validate_plan_or_raise(plan, locator=plan_path, recheck_current=True)
    published = _atomic_publish_new_json(output, artifact)
    return published, ContinualIAV2Validation(True, accepted, ())


def load_artifact(
    path: Path,
    *,
    recheck_current: bool = True,
) -> tuple[bytes, dict[str, Any], ContinualIAV2Validation]:
    _require(
        recheck_current,
        "public artifact loading requires current source/runtime binding verification",
    )
    raw, payload = _read_canonical_json(path)
    validation = validate_artifact_payload(
        payload,
        locator=path,
        recheck_current=recheck_current,
    )
    if validation.valid:
        final_raw, final_payload = _read_canonical_json(path)
        if raw != final_raw or not _json_exact_equal(payload, final_payload):
            validation = ContinualIAV2Validation(
                False,
                False,
                (f"artifact bytes changed during validation: {path}",),
            )
        else:
            body = cast(dict[str, Any], payload["artifact"])
            try:
                _validate_plan_or_raise(
                    _expect_dict(body["plan"], "artifact.plan"),
                    locator=None,
                    recheck_current=True,
                )
            except Exception as exc:
                validation = ContinualIAV2Validation(False, False, (str(exc),))
    return raw, payload, validation


__all__ = [
    "ARTIFACT_SCHEMA",
    "NAMESPACE",
    "PLAN_SCHEMA",
    "PROTOCOL_VERSION",
    "RESERVATION_SCHEMA",
    "SHARD_SCHEMA",
    "TRACE_SCHEMA",
    "V2_CONDITIONS",
    "V2_EVIDENCE_SEEDS",
    "ContinualIAV2Error",
    "ContinualIAV2Validation",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "load_artifact",
    "load_plan",
    "load_shard",
    "merge_shards",
    "merge_shards_with_validation",
    "sha256_bytes",
    "strict_json_loads",
    "validate_artifact_payload",
    "validate_plan_payload",
    "validate_shard_payload",
    "validate_trace",
    "write_plan",
    "write_shard",
]
