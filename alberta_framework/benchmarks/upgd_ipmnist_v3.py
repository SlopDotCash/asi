"""Strict active execution contract for the UPGD IPMNIST development lane.

The benchmark science lives in :mod:`alberta_framework.benchmarks.upgd_ipmnist`
(the Input-permuted MNIST replication of Elsayed & Mahmood, ICLR 2024,
including its documented deviations); this module governs only *how* that
protocol may be executed.  Operator procedure — prerequisites, the external
seed-reservation ledger, and the exact per-stage commands — is specified in
``UPGD_IPMNIST_V3_RUNBOOK.md`` at the alberta tree root.

The v3 lifecycle is deliberately split into three commands: issue an immutable
pre-run plan, execute exactly one learner/seed shard, and merge the exact
planned Cartesian product.  Every file is published atomically at a new path.
The schema is permanently nonpromoting because the execution envelope is
self-recorded and has no independent attestation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import secrets
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast
from unittest.mock import patch

import jax
import numpy as np

from alberta_framework.benchmarks.upgd_ipmnist import (
    ADAMW_PROTOCOL_HYPERPARAMETERS,
    PAPER_REFERENCE,
    UPGD_W_PROTOCOL_HYPERPARAMETERS,
    IPMNISTConfig,
    IPMNISTRunResult,
    build_comparison,
    default_openml_data_home,
    run_ipmnist,
    summarize_result,
)

logger = logging.getLogger(__name__)

# Platform detection for cross-platform atomicity
# Windows doesn't support opening drive letters as file descriptors,
# so we use pathlib-based operations instead of POSIX directory FDs.
_IS_WINDOWS = platform.system() == "Windows"

PLAN_SCHEMA = "alberta.upgd_ipmnist.plan.v3"
PARTIAL_SCHEMA = "alberta.upgd_ipmnist.partial.v3"
ARTIFACT_SCHEMA = "alberta.upgd_ipmnist.artifact.v3"
RESERVATION_SCHEMA = "alberta.upgd_ipmnist.seed_reservation.v3"
SOURCE_CLOSURE_SCHEMA = "alberta.upgd_ipmnist.source_import_closure.v1"
RUNTIME_SCHEMA = "alberta.upgd_ipmnist.runtime.v2"
DATA_SCHEMA = "alberta.upgd_ipmnist.data.v1"
BENCHMARK = "upgd_input_permuted_mnist"

LEARNER_IDS = ("upgd_w", "adamw")
# Seed IDs 0-9 were consumed by the completed v1 diagnostic run of this
# benchmark; plan validation rejects them so no v3 plan can reuse a seed
# whose results have already been observed.
KNOWN_CONSUMED_SEED_IDS = tuple(range(10))
MNIST_ARCHIVE_RELATIVE_PATH = Path("openml/openml.org/data/v1/download/52667/mnist_784.arff.gz")
MNIST_ARCHIVE_BYTE_SIZE = 15_469_256
MNIST_ARCHIVE_SHA256 = "fe4410d8dbb50f6db6482b187557c5cb8bccfbcec74eeb6abc47c858f4ffab78"
MNIST_TRAIN_X_SHA256 = "dfd3c4418425b5f47b77c6b155c891f596f3ee03c324b2feb874116f0d0214cc"
MNIST_TRAIN_Y_SHA256 = "f710a62c2453a1aa8490a45dc62252abf838ca4ef2961f12770698a17bd74d37"
OPENML_CACHE_MEMBER_IDENTITIES: tuple[tuple[str, int, str], ...] = (
    (
        "openml/openml.org/api/v1/json/data/554.gz",
        1_929,
        "8e943aee97b9d706ccc321825570f01e6eb4ea096da5788783d980f0a676b423",
    ),
    (
        "openml/openml.org/api/v1/json/data/features/554.gz",
        4_650,
        "a2ea1a5660222291db76d6b0b2a25cce70b5741e7ac794c7888710b5784c71c7",
    ),
    (
        "openml/openml.org/api/v1/json/data/qualities/554.gz",
        1_285,
        "18adba5ffc0517f2f349eedf08d922af4b35c0a1d7211633668e8a11ce993cdf",
    ),
    (
        MNIST_ARCHIVE_RELATIVE_PATH.as_posix(),
        MNIST_ARCHIVE_BYTE_SIZE,
        MNIST_ARCHIVE_SHA256,
    ),
)
# Synthetic unit fixtures patch this private switch. Active publication and
# validation require the exact selected paper configuration.
_ALLOW_SYNTHETIC_CONFIG_FOR_TESTING = False

_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_REGULAR_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_OPENML_NETWORK_GUARD_LOCK = threading.RLock()

EVIDENCE_POLICY: dict[str, object] = {
    "evidence_role": "development_replication_diagnostic",
    "development_only": True,
    "external_execution_attestation_present": False,
    "scientific_promotion_allowed": False,
}

_SOURCE_ROOT_MODULES = (
    "alberta_framework.benchmarks.upgd_ipmnist",
    "alberta_framework.benchmarks.upgd_ipmnist_v3",
)
_SOURCE_AUXILIARY_PATHS = (
    "pyproject.toml",
    "uv.lock",
)
_RUNTIME_ENVIRONMENT_NAMES = (
    "JAX_DEFAULT_MATMUL_PRECISION",
    "JAX_DEFAULT_PRNG_IMPL",
    "JAX_ENABLE_X64",
    "JAX_PLATFORM_NAME",
    "JAX_PLATFORMS",
    "XLA_FLAGS",
)

# This is an explicit content-bound execution set, not a claim that Python
# package metadata captures native drivers, the kernel, libc, or every dynamic
# import. It covers the benchmark, JAX numerical stack, exact pandas OpenML
# parsing path, and selected active import-time dependencies in the locked
# environment. The manifest below remains the complete statement of scope.
_RUNTIME_CONTENT_DISTRIBUTIONS = (
    "absl-py",
    "chex",
    "etils",
    "jax",
    "jaxlib",
    "jaxtyping",
    "joblib",
    "ml-dtypes",
    "msgpack",
    "narwhals",
    "numpy",
    "opt-einsum",
    "orbax-checkpoint",
    "pandas",
    "protobuf",
    "python-dateutil",
    "scikit-learn",
    "scipy",
    "six",
    "tensorstore",
    "threadpoolctl",
    "toolz",
    "typing-extensions",
)

_COMMAND_TEMPLATES: dict[str, list[str]] = {
    "plan": [
        "python",
        "-m",
        "alberta_framework.benchmarks.upgd_ipmnist",
        "plan",
        "--plan-out",
        "<PLAN_LOCATOR>",
        "--seed-list",
        "<FRESH_SEED_IDS>",
        "--data-home",
        "<DATA_HOME_LOCATOR>",
        "--data-archive",
        "<DATA_ARCHIVE_LOCATOR>",
        "--n-tasks",
        "<N_TASKS>",
        "--task-length",
        "<TASK_LENGTH>",
        "--input-dim",
        "<INPUT_DIM>",
        "--hidden1",
        "<HIDDEN1>",
        "--hidden2",
        "<HIDDEN2>",
        "--n-classes",
        "<N_CLASSES>",
    ],
    "shard": [
        "python",
        "-m",
        "alberta_framework.benchmarks.upgd_ipmnist",
        "shard",
        "--plan",
        "<PLAN_LOCATOR>",
        "--learner-id",
        "<LEARNER_ID>",
        "--seed-id",
        "<SEED_ID>",
        "--partial-out",
        "<PARTIAL_LOCATOR>",
        "--data-home",
        "<DATA_HOME_LOCATOR>",
        "--data-archive",
        "<DATA_ARCHIVE_LOCATOR>",
    ],
    "merge": [
        "python",
        "-m",
        "alberta_framework.benchmarks.upgd_ipmnist",
        "merge",
        "--plan",
        "<PLAN_LOCATOR>",
        "--partials",
        "<EXACT_PLANNED_PARTIAL_LOCATORS>",
        "--output",
        "<ARTIFACT_LOCATOR>",
        "--data-home",
        "<DATA_HOME_LOCATOR>",
        "--data-archive",
        "<DATA_ARCHIVE_LOCATOR>",
    ],
}

_DATA_LOADER_CONTRACT: dict[str, object] = {
    "provider": "sklearn.datasets.fetch_openml",
    "data_id": 554,
    "parser": "pandas",
    "network_guard": "sklearn_openml_urlopen_denied_after_complete_cache_verification",
    "train_rows": 60_000,
    "input_scaling": "(x/255 - 0.5) / 0.5",
}

_MEASUREMENT_CONTRACT: dict[str, object] = {
    "loss": "single_example_softmax_cross_entropy",
    "online_accuracy": "pre_update_prediction_task_mean",
    "plasticity": "clip_one_minus_post_loss_over_pre_loss_floor_1e_minus_8",
    "partial_array_rank": 1,
}
_DURATION_SEMANTICS = (
    "self-reported monotonic worker diagnostic; cross-checked against integer Unix "
    "timestamps; excluded from learner summaries, comparisons, and scientific claims"
)
_PAIRED_COMPARISON_CONTRACT: dict[str, object] = {
    "estimand": "per_seed_average_online_accuracy_upgd_w_minus_adamw",
    "pairing_unit": "seed_id",
    "confidence_level": 0.95,
    "interval": "two_sided_paired_student_t",
    "degrees_of_freedom": 19,
    "t_critical": 2.093024054408263,
    "post_hoc_acceptance_gate": False,
}


class UPGDIPMNISTV3Error(ValueError):
    """Raised when a v3 payload fails its closed contract."""


@dataclass(frozen=True)
class UPGDIPMNISTV3Validation:
    """A structural validation result that can never authorize promotion."""

    valid: bool
    scientific_promotion_allowed: bool
    errors: tuple[str, ...]


def _fail(message: str) -> NoReturn:
    raise UPGDIPMNISTV3Error(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _expect_dict(value: object, where: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{where} must be an object")
    return cast(dict[str, Any], value)


def _expect_list(value: object, where: str) -> list[Any]:
    _require(isinstance(value, list), f"{where} must be an array")
    return cast(list[Any], value)


def _expect_exact_keys(value: Mapping[str, Any], keys: set[str], where: str) -> None:
    actual = set(value)
    _require(
        actual == keys,
        f"{where} keys differ: missing={sorted(keys - actual)}, extra={sorted(actual - keys)}",
    )


def _reject_constant(token: str) -> NoReturn:
    _fail(f"non-finite JSON constant is forbidden: {token}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _check_finite_tree(value: object, where: str = "$") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"{where} contains a non-finite float")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite_tree(item, f"{where}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_finite_tree(item, f"{where}.{key}")


def _json_exact_equal(actual: object, expected: object) -> bool:
    """Compare JSON trees without Python's bool/int or int/float coercions."""

    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _json_exact_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_exact_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _reject_legacy_marker(value: object, where: str = "$") -> None:
    if isinstance(value, dict):
        _require("is_protocol_exact" not in value, f"{where} contains forbidden legacy marker")
        for key, item in value.items():
            _reject_legacy_marker(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_legacy_marker(item, f"{where}[{index}]")


def strict_json_loads(text: str) -> Any:
    """Decode strict JSON, rejecting duplicate keys and non-finite numbers."""

    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except UPGDIPMNISTV3Error:
        raise
    except (json.JSONDecodeError, OverflowError, RecursionError, UnicodeError) as exc:
        raise UPGDIPMNISTV3Error(f"invalid JSON: {exc}") from exc
    try:
        _check_finite_tree(value)
    except RecursionError as exc:
        raise UPGDIPMNISTV3Error("JSON nesting is too deep") from exc
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Return the one accepted on-disk JSON encoding for v3 payloads."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise UPGDIPMNISTV3Error(f"value is not strict JSON: {exc}") from exc
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
    except (TypeError, ValueError) as exc:
        raise UPGDIPMNISTV3Error(f"value is not strict JSON: {exc}") from exc
    return encoded.encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase SHA-256 content digest."""

    return hashlib.sha256(data).hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON value independently of its locator or pretty encoding."""

    return sha256_bytes(_compact_json_bytes(value))


def _validated_process_argv(value: Sequence[str], where: str) -> list[str]:
    argv = list(value)
    _require(
        bool(argv) and all(isinstance(item, str) and bool(item) for item in argv),
        f"{where} must be a nonempty array of nonempty strings",
    )
    return argv


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _lexical_absolute(path: Path) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError) as exc:
        raise UPGDIPMNISTV3Error(f"invalid filesystem path {path!r}: {exc}") from exc


def _canonical_absolute_locator(value: object, where: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{where} must be a nonempty string")
    path = Path(cast(str, value))
    _require(path.is_absolute(), f"{where} must be absolute")
    canonical = _lexical_absolute(path).as_posix()
    _require(canonical == value, f"{where} must be lexically canonical")
    return canonical


def _validate_data_locators(value: object, where: str) -> dict[str, str]:
    locators = _expect_dict(value, where)
    _expect_exact_keys(locators, {"data_home", "archive"}, where)
    home = _canonical_absolute_locator(locators["data_home"], f"{where}.data_home")
    archive = _canonical_absolute_locator(locators["archive"], f"{where}.archive")
    _require(
        Path(archive) == _lexical_absolute(Path(home) / MNIST_ARCHIVE_RELATIVE_PATH),
        f"{where}.archive is not the canonical member of data_home",
    )
    return {"data_home": home, "archive": archive}


def _open_parent_directory(path: Path, *, create: bool) -> tuple[Path, int | None]:
    """Open a stable parent descriptor without following symlink components.

    Cross-platform implementation:
    - On POSIX (Linux/macOS): Returns a directory file descriptor for atomicity
    - On Windows: Returns None; operations use pathlib instead

    Preserves atomicity guarantees:
    - POSIX: Descriptor anchors inode, prevents TOCTOU races
    - Windows: pathlib + tempfile in same directory ensures filesystem atomicity
    """

    destination = _lexical_absolute(path)
    _require(destination != destination.parent, "filesystem path must name a file")

    # Windows doesn't support opening drive letters as file descriptors.
    # Fall back to pathlib-based directory operations.
    if _IS_WINDOWS:
        parent = destination.parent
        if create:
            parent.mkdir(parents=True, exist_ok=True)
        # Return None as sentinel; callers check for this and use pathlib
        return destination, None

    # POSIX: Use directory file descriptors for maximum safety
    root = destination.anchor or os.sep
    directory_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)
    try:
        for component in destination.parent.parts[1:]:
            created = False
            if create:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=directory_fd)
                    created = True
                except FileExistsError:
                    pass
            next_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd)
            if created:
                os.fsync(next_fd)
                os.fsync(directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
    except BaseException:
        os.close(directory_fd)
        raise
    return destination, directory_fd


def _stable_stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _assert_parent_locator_stable(destination: Path, directory_fd: int | None) -> None:
    """Require the opened parent to remain at the requested lexical locator.

    On Windows (directory_fd=None), this is a no-op since we can't anchor to
    an inode. The atomicity is ensured by pathlib's cross-platform rename.
    """

    if directory_fd is None:
        # Windows: pathlib operations are already atomic via OS rename semantics
        return

    verified_destination, verification_fd = _open_parent_directory(
        destination,
        create=False,
    )
    try:
        opened = os.fstat(directory_fd)
        current = os.fstat(verification_fd)
        _require(
            verified_destination == destination
            and (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino),
            f"ancestor directory changed while accessing lifecycle path: {destination}",
        )
    finally:
        if verification_fd is not None:
            os.close(verification_fd)


def _read_regular_bytes(path: Path, *, require_immutable: bool) -> bytes:
    """Read one stable regular file through descriptor-anchored path traversal.

    On POSIX: Uses directory FDs to anchor parent inode and prevent TOCTOU.
    On Windows: Uses pathlib stat directly (atomic via OS semantics).
    """

    destination, directory_fd = _open_parent_directory(path, create=False)
    file_fd = -1
    try:
        _assert_parent_locator_stable(destination, directory_fd)

        # Windows: directory_fd is None, use pathlib directly
        if directory_fd is None:
            # Verify file exists and is regular
            stat_result = destination.stat()
            _require(stat.S_ISREG(stat_result.st_mode), f"{destination} must be a regular file")
            if require_immutable:
                _require(
                    stat.S_IMODE(stat_result.st_mode) & 0o222 == 0,
                    f"{destination} must have no write permission bits",
                )
                _require(
                    stat_result.st_nlink == 1,
                    f"{destination} must have exactly one hard link",
                )
            # Read file content
            raw = destination.read_bytes()
            # Verify file wasn't modified during read
            stat_after = destination.stat()
            _require(
                stat.S_ISREG(stat_after.st_mode)
                and stat_result.st_size == stat_after.st_size
                and stat_result.st_mtime_ns == stat_after.st_mtime_ns,
                f"{destination} changed while it was being read",
            )
            return raw

        # POSIX: Use directory FD for stability
        file_fd = os.open(destination.name, _REGULAR_READ_FLAGS, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        _require(stat.S_ISREG(before.st_mode), f"{destination} must be a regular file")
        if require_immutable:
            _require(
                stat.S_IMODE(before.st_mode) & 0o222 == 0,
                f"{destination} must have no write permission bits",
            )
            _require(
                before.st_nlink == 1,
                f"{destination} must have exactly one hard link",
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        locator = os.stat(
            destination.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        _require(
            _stable_stat_signature(before)
            == _stable_stat_signature(after)
            == _stable_stat_signature(locator),
            f"{destination} changed or its locator was replaced while it was being read",
        )
        raw = b"".join(chunks)
        _require(
            len(raw) == before.st_size,
            f"{destination} size changed while it was being read",
        )
        _assert_parent_locator_stable(destination, directory_fd)
        return raw
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _unlink_if_identity(
    directory_fd: int | None,
    name: str,
    source: os.stat_result | None,
    parent_path: Path | None = None,
) -> None:
    """Remove ``name`` only while it still identifies ``source``.

    Cleanup must not delete a path that another actor substituted after a
    failure. The supplied identity can be the open descriptor or the exact
    post-link target observed before rejecting a pathname swap.

    On Windows (directory_fd=None): Uses pathlib operations instead.
    On POSIX: Uses directory FD to anchor inode.
    """

    if directory_fd is None:
        # Windows: Use pathlib to verify identity before unlinking
        if parent_path is None or source is None:
            return
        target_path = parent_path / name
        try:
            target_stat = target_path.stat()
            # On Windows, compare by size and mtime as proxy for identity
            # (inode is not reliable cross-platform)
            if source.st_size == target_stat.st_size:
                target_path.unlink()
        except (FileNotFoundError, OSError):
            pass
        return

    # POSIX: Use directory FD for stable identity check
    try:
        target = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if target.st_dev == source.st_dev and target.st_ino == source.st_ino:
        os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)


def atomic_write_new(path: Path, data: bytes) -> Path:
    """Atomically publish immutable bytes without replacing an existing path.

    Cross-platform atomicity strategy:
    - POSIX: Uses directory FDs and hard links for maximum safety
    - Windows: Uses pathlib.Path.replace() for atomic rename semantics

    Both approaches ensure:
    1. Temporary file created in same directory (same filesystem)
    2. File marked read-only before publication
    3. Atomic rename/link operation
    4. Verification that published data matches input
    """

    destination, directory_fd = _open_parent_directory(path, create=True)
    temporary_name = ""
    file_fd = -1
    target_linked = False
    publication_complete = False
    source: os.stat_result | None = None

    try:
        _assert_parent_locator_stable(destination, directory_fd)

        # Windows: Use tempfile + pathlib.replace() for atomicity
        if directory_fd is None:
            try:
                destination.stat()
                raise FileExistsError(f"refusing to overwrite immutable output: {destination}")
            except FileNotFoundError:
                pass

            # Create temporary file in same directory
            tmp_fd, tmp_path_str = tempfile.mkstemp(
                dir=str(destination.parent),
                prefix=f".{destination.name}.",
                suffix=".tmp",
            )
            tmp_path = Path(tmp_path_str)
            temporary_name = tmp_path.name

            try:
                # Write data
                remaining = memoryview(data)
                while remaining:
                    written = os.write(tmp_fd, remaining)
                    _require(written > 0, "short write while publishing immutable output")
                    remaining = remaining[written:]
                os.close(tmp_fd)
                file_fd = -1

                # Make read-only
                tmp_path.chmod(0o444)
                source = tmp_path.stat()

                # Atomic rename (Windows NTFS guarantees all-or-nothing)
                tmp_path.replace(destination)
                target_linked = True

                # Verify published content
                published = destination.read_bytes()
                _require(published == data, "published output bytes differ from supplied bytes")

                # Final verification
                final_stat = destination.stat()
                _require(
                    stat.S_ISREG(final_stat.st_mode)
                    and stat.S_IMODE(final_stat.st_mode) == 0o444,
                    "published output mode or type incorrect",
                )

                publication_complete = True
                return destination

            finally:
                if file_fd >= 0:
                    try:
                        os.close(file_fd)
                    except OSError:
                        pass
                if temporary_name and tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

        # POSIX: Use directory FDs and hard links (original logic)
        for _ in range(128):
            candidate = f".{destination.name}.{secrets.token_hex(16)}.tmp"
            try:
                file_fd = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        _require(file_fd >= 0, "could not allocate a unique temporary output")
        remaining = memoryview(data)
        while remaining:
            written = os.write(file_fd, remaining)
            _require(written > 0, "short write while publishing immutable output")
            remaining = remaining[written:]
        os.fchmod(file_fd, 0o444)
        os.fsync(file_fd)
        source = os.fstat(file_fd)
        _assert_parent_locator_stable(destination, directory_fd)
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite immutable output: {destination}") from exc
        target_linked = True
        target = os.stat(destination.name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            source.st_dev != target.st_dev
            or source.st_ino != target.st_ino
            or source.st_size != target.st_size
            or not stat.S_ISREG(target.st_mode)
            or stat.S_IMODE(target.st_mode) != 0o444
        ):
            if source.st_dev == target.st_dev and source.st_ino == target.st_ino:
                _unlink_if_identity(directory_fd, destination.name, source, destination.parent)
            target_linked = False
            _fail("published output does not identify the descriptor-anchored temporary file")
        _unlink_if_identity(directory_fd, temporary_name, source, destination.parent)
        temporary_name = ""
        os.fsync(directory_fd)
        final_target = os.stat(
            destination.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        _require(
            final_target.st_dev == source.st_dev
            and final_target.st_ino == source.st_ino
            and final_target.st_nlink == 1
            and stat.S_IMODE(final_target.st_mode) == 0o444,
            "published output identity or link count changed",
        )
        _assert_parent_locator_stable(destination, directory_fd)
        published = _read_regular_bytes(destination, require_immutable=True)
        _require(published == data, "published output bytes differ from supplied bytes")
        post_read_target = os.stat(
            destination.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        _require(
            post_read_target.st_dev == source.st_dev
            and post_read_target.st_ino == source.st_ino
            and post_read_target.st_size == source.st_size
            and post_read_target.st_nlink == 1
            and stat.S_IMODE(post_read_target.st_mode) == 0o444
            and stat.S_ISREG(post_read_target.st_mode),
            "published output changed during byte verification",
        )
        _assert_parent_locator_stable(destination, directory_fd)
        publication_complete = True
        return destination
    finally:
        if target_linked and not publication_complete and source is not None:
            try:
                _unlink_if_identity(directory_fd, destination.name, source, destination.parent)
            except OSError:
                pass
        if temporary_name:
            try:
                temporary_source = source
                if temporary_source is None and file_fd >= 0:
                    temporary_source = os.fstat(file_fd)
                if temporary_source is not None:
                    _unlink_if_identity(directory_fd, temporary_name, temporary_source, destination.parent)
            except OSError:
                pass
        if file_fd >= 0:
            try:
                os.close(file_fd)
            except OSError:
                pass
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass


def _preflight_new_output(path: Path) -> Path:
    """Reject an occupied output before an expensive worker starts."""

    destination, directory_fd = _open_parent_directory(path, create=True)
    try:
        _assert_parent_locator_stable(destination, directory_fd)

        # Windows: directory_fd is None, use pathlib
        if directory_fd is None:
            try:
                destination.stat()
                raise FileExistsError(f"refusing to overwrite immutable output: {destination}")
            except FileNotFoundError:
                return destination

        # POSIX: Use directory FD
        try:
            os.stat(destination.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return destination
        raise FileExistsError(f"refusing to overwrite immutable output: {destination}")
    finally:
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass


def atomic_write_new_json(path: Path, value: object) -> Path:
    """Publish a canonical v3 or explicit compatibility JSON file once."""

    return atomic_write_new(path, canonical_json_bytes(value))


def _read_strict_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = _read_regular_bytes(path, require_immutable=True)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UPGDIPMNISTV3Error(f"{path} is not UTF-8 JSON") from exc
    value = strict_json_loads(text)
    payload = _expect_dict(value, str(path))
    _reject_legacy_marker(payload)
    return raw, payload


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _module_path(module: str) -> Path | None:
    if module != "alberta_framework" and not module.startswith("alberta_framework."):
        return None
    relative = Path(*module.split("."))
    source = _repo_root() / relative.with_suffix(".py")
    if source.is_file():
        return source
    package = _repo_root() / relative / "__init__.py"
    return package if package.is_file() else None


def _module_name(path: Path) -> tuple[str, bool]:
    relative = path.relative_to(_repo_root())
    if relative.name == "__init__.py":
        return ".".join(relative.parent.parts), True
    return ".".join(relative.with_suffix("").parts), False


def _parent_packages(module: str) -> list[str]:
    parts = module.split(".")
    return [".".join(parts[:index]) for index in range(1, len(parts))]


def _resolve_imports(path: Path, raw: bytes) -> set[str]:
    module, is_package = _module_name(path)
    package = module if is_package else module.rpartition(".")[0]
    try:
        tree = ast.parse(raw, filename=str(path))
    except (SyntaxError, UnicodeError) as exc:
        raise UPGDIPMNISTV3Error(
            f"cannot parse source import closure member {path}: {exc}"
        ) from exc
    discovered: set[str] = set()
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
                name = ".".join(parts)
                if _module_path(name) is not None:
                    discovered.add(name)
                    discovered.update(_parent_packages(name))
                    break
                parts.pop()
    return discovered


def _build_source_import_closure() -> dict[str, Any]:
    pending = set(_SOURCE_ROOT_MODULES)
    for root in _SOURCE_ROOT_MODULES:
        pending.update(_parent_packages(root))
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
        raw = _read_regular_bytes(path, require_immutable=False)
        bytes_by_module[module] = raw
        pending.update(_resolve_imports(path, raw) - visited)
    files: list[dict[str, Any]] = []
    source_root = _repo_root()
    for module in sorted(visited):
        path = _module_path(module)
        assert path is not None
        raw = bytes_by_module[module]
        files.append(
            {
                "module": module,
                "locator": path.relative_to(source_root).as_posix(),
                "byte_size": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )
    for relative in _SOURCE_AUXILIARY_PATHS:
        raw = _read_regular_bytes(source_root / relative, require_immutable=False)
        files.append(
            {
                "module": f"file:{relative}",
                "locator": relative,
                "byte_size": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )
    files.sort(key=lambda entry: cast(str, entry["module"]))
    for entry in files:
        locator = cast(str, entry["locator"])
        current = _read_regular_bytes(source_root / locator, require_immutable=False)
        _require(
            len(current) == entry["byte_size"]
            and sha256_bytes(current) == entry["sha256"],
            f"source file changed while its closure was being built: {locator}",
        )
    return {
        "schema": SOURCE_CLOSURE_SCHEMA,
        "closure_kind": "static_transitive_local_python_imports_plus_lockfiles",
        "root_modules": list(_SOURCE_ROOT_MODULES),
        "files": files,
    }


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _distribution_content_identity(name: str) -> dict[str, object]:
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


def _runtime_json_value(value: object) -> str | int | float | bool | None:
    if value is None or type(value) in {str, int, bool}:
        return cast(str | int | bool | None, value)
    if type(value) is float:
        _require(math.isfinite(value), "JAX config contains a non-finite float")
        return value
    return str(value)


def _build_runtime_manifest() -> dict[str, Any]:
    devices = list(jax.devices())
    executable_path = Path(sys.executable).resolve()
    executable_raw = _read_regular_bytes(executable_path, require_immutable=False)
    return {
        "schema": RUNTIME_SCHEMA,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": {
            "locator": executable_path.as_posix(),
            "byte_size": len(executable_raw),
            "sha256": sha256_bytes(executable_raw),
        },
        "jax": jax.__version__,
        "jaxlib": _distribution_version("jaxlib"),
        "numpy": np.__version__,
        "scikit_learn": _distribution_version("scikit-learn"),
        "distribution_content": {
            name: _distribution_content_identity(name)
            for name in _RUNTIME_CONTENT_DISTRIBUTIONS
        },
        "distribution_content_scope": {
            "kind": "explicit_python_execution_distribution_set",
            "distribution_names": list(_RUNTIME_CONTENT_DISTRIBUTIONS),
            "dynamic_import_closure_claimed": False,
            "native_system_library_closure_claimed": False,
        },
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in devices],
        "jax_device_details": [
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "process_index": int(device.process_index),
                "runtime_type": str(getattr(device.client, "runtime_type", "unknown")),
            }
            for device in devices
        ],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_config": {
            name: _runtime_json_value(value)
            for name, value in sorted(jax.config.values.items())
        },
        "execution_environment": {
            name: os.environ.get(name) for name in _RUNTIME_ENVIRONMENT_NAMES
        },
    }


def _verify_offline_cache(
    data_home: Path,
    archive: Path | None = None,
) -> list[dict[str, object]]:
    """Verify every fetch_openml(data_id=554) cache input before sklearn runs."""

    home_path = _lexical_absolute(data_home)
    expected_archive = _lexical_absolute(home_path / MNIST_ARCHIVE_RELATIVE_PATH)
    if archive is not None:
        _require(
            _lexical_absolute(archive) == expected_archive,
            "MNIST archive must be the exact fetch_openml archive inside data_home",
        )
    cache_files: list[dict[str, object]] = []
    for relative, expected_size, expected_sha256 in OPENML_CACHE_MEMBER_IDENTITIES:
        member_raw = _read_regular_bytes(home_path / relative, require_immutable=True)
        _require(
            len(member_raw) == expected_size
            and sha256_bytes(member_raw) == expected_sha256,
            f"OpenML cache member differs from its pinned identity: {relative}",
        )
        cache_files.append(
            {
                "locator": relative,
                "byte_size": len(member_raw),
                "sha256": sha256_bytes(member_raw),
            }
        )
    _require(
        any(
            entry[0] == MNIST_ARCHIVE_RELATIVE_PATH.as_posix()
            for entry in OPENML_CACHE_MEMBER_IDENTITIES
        ),
        "complete OpenML cache identities omit the pinned archive",
    )
    return cache_files


def _build_data_manifest(data_home: Path, archive: Path) -> dict[str, Any]:
    home_path = _lexical_absolute(data_home)
    archive_path = _lexical_absolute(archive)
    expected_archive = _lexical_absolute(home_path / MNIST_ARCHIVE_RELATIVE_PATH)
    cache_files = _verify_offline_cache(home_path, archive_path)
    raw = _read_regular_bytes(expected_archive, require_immutable=True)
    _require(
        len(raw) == MNIST_ARCHIVE_BYTE_SIZE,
        "MNIST archive byte size differs from the pinned OpenML payload",
    )
    _require(
        sha256_bytes(raw) == MNIST_ARCHIVE_SHA256,
        "MNIST archive SHA-256 differs from the pinned OpenML payload",
    )
    return {
        "schema": DATA_SCHEMA,
        "dataset_id": "openml_mnist_784_v1_first_60000",
        "content": {
            "byte_size": len(raw),
            "sha256": sha256_bytes(raw),
        },
        "locators": {
            "data_home": home_path.as_posix(),
            "archive": archive_path.as_posix(),
        },
        "complete_offline_cache": {
            "network_access_required": False,
            "files": cache_files,
        },
        "materialized_arrays": {
            "x": {
                "dtype": "<f4",
                "shape": [60_000, 784],
                "sha256": MNIST_TRAIN_X_SHA256,
            },
            "y": {
                "dtype": "<i4",
                "shape": [60_000],
                "sha256": MNIST_TRAIN_Y_SHA256,
            },
        },
        "loader_contract": dict(_DATA_LOADER_CONTRACT),
    }


def _materialized_array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_loaded_mnist(data_x: np.ndarray, data_y: np.ndarray) -> None:
    x = np.asarray(data_x)
    y = np.asarray(data_y)
    _require(
        x.dtype.str == "<f4"
        and x.shape == (60_000, 784)
        and _materialized_array_sha256(x) == MNIST_TRAIN_X_SHA256,
        "materialized MNIST input array differs from the pinned train split",
    )
    _require(
        y.dtype.str == "<i4"
        and y.shape == (60_000,)
        and _materialized_array_sha256(y) == MNIST_TRAIN_Y_SHA256,
        "materialized MNIST label array differs from the pinned train split",
    )


def _load_pinned_mnist(
    data_home: Path,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    home = _lexical_absolute(data_home)
    try:
        # This check occurs before importing/calling sklearn, so an incomplete
        # cache fails without initiating any HTTP path.
        _verify_offline_cache(home)
        from sklearn.datasets import _openml as openml_module  # type: ignore[import-untyped]
        from sklearn.datasets import fetch_openml

        def deny_network(*_args: object, **_kwargs: object) -> NoReturn:
            raise UPGDIPMNISTV3Error(
                f"{context} attempted forbidden OpenML network access"
            )

        # fetch_openml's sole URL entry point calls this module-global
        # ``urlopen``. All expected data_id=554 members are already pinned;
        # any unexpected cache miss therefore fails closed instead of HTTP.
        with _OPENML_NETWORK_GUARD_LOCK:
            with patch.object(openml_module, "urlopen", deny_network):
                raw = fetch_openml(
                    data_id=554,
                    as_frame=False,
                    data_home=str(home),
                    n_retries=1,
                    delay=0.0,
                    parser="pandas",
                )
        data_x = np.asarray(raw.data, dtype=np.float32)[:60_000]
        data_y = np.asarray(raw.target, dtype=np.int32)[:60_000]
        data_x = (data_x / 255.0 - 0.5) / 0.5
        _validate_loaded_mnist(data_x, data_y)
    except Exception as exc:
        raise UPGDIPMNISTV3Error(f"{context} dataset load failed: {exc}") from exc
    return data_x, data_y


def _validate_seed_ids(seed_ids: Sequence[int]) -> tuple[int, ...]:
    _require(bool(seed_ids), "the pre-run plan requires explicit fresh seed IDs")
    _require(
        all(_is_int(seed) and 0 <= seed <= 0xFFFF_FFFF for seed in seed_ids),
        "seed IDs must be uint32 integers",
    )
    seeds = tuple(int(seed) for seed in seed_ids)
    _require(
        len(seeds) == int(PAPER_REFERENCE["n_seeds"]),
        f"v3 replication plans require exactly {PAPER_REFERENCE['n_seeds']} fresh seed IDs",
    )
    _require(len(set(seeds)) == len(seeds), "seed IDs must be unique")
    _require(seeds == tuple(sorted(seeds)), "seed IDs must be sorted")
    consumed = sorted(set(seeds) & set(KNOWN_CONSUMED_SEED_IDS))
    _require(
        not consumed,
        f"seed IDs were consumed by the completed v1 diagnostic: {consumed}",
    )
    return seeds


def _validate_learner_ids(learner_ids: Sequence[str]) -> tuple[str, ...]:
    learners = tuple(learner_ids)
    _require(
        learners == LEARNER_IDS,
        f"v3 plans require the exact canonical learner pair {list(LEARNER_IDS)}",
    )
    return learners


def _default_hyperparameters() -> dict[str, dict[str, float]]:
    return {
        "upgd_w": dict(UPGD_W_PROTOCOL_HYPERPARAMETERS),
        "adamw": dict(ADAMW_PROTOCOL_HYPERPARAMETERS),
    }


def _validated_hyperparameters(
    learner_ids: Sequence[str],
    hyperparameters: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    _require(set(hyperparameters) == set(learner_ids), "hyperparameter learner keys differ")
    defaults = _default_hyperparameters()
    validated: dict[str, dict[str, float]] = {}
    for learner in learner_ids:
        values = hyperparameters[learner]
        _require(
            set(values) == set(defaults[learner]),
            f"{learner} hyperparameter keys differ",
        )
        converted: dict[str, float] = {}
        for name, value in values.items():
            _require(_is_number(value), f"{learner}.{name} must be finite")
            converted[name] = float(value)
        _require(
            _json_exact_equal(converted, defaults[learner]),
            f"{learner} hyperparameters must equal the selected canonical arm",
        )
        validated[learner] = converted
    return validated


def _closed_deviations(seed_count: int) -> list[dict[str, object]]:
    return [
        {
            "id": "rng_schedule",
            "published": "unseeded_pixel_permutations",
            "implementation": "seed_derived_all_streams",
        },
        {
            "id": "metric_blocks",
            "published": "one_step_shifted_with_final_tail_omitted",
            "implementation": "task_boundary_aligned_complete_blocks",
        },
        {
            "id": "bias_correction_dtype",
            "published": "python_scalar_mixed_precision",
            "implementation": "float32",
        },
        {
            "id": "upgd_inner_loop",
            "published": "released_pytorch_optimizer",
            "implementation": "jax_lean_parity_tested_restatement",
        },
        {
            "id": "seed_count",
            "published": int(PAPER_REFERENCE["n_seeds"]),
            "planned": seed_count,
            "relation": "match" if seed_count == int(PAPER_REFERENCE["n_seeds"]) else "mismatch",
        },
    ]


def _selected_publication_match(
    config: IPMNISTConfig,
    hyperparameters: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    defaults = _default_hyperparameters()
    per_learner = {
        learner: (
            "match"
            if _json_exact_equal(dict(hyperparameters[learner]), defaults[learner])
            else "mismatch"
        )
        for learner in LEARNER_IDS
    }
    config_status = "match" if config.matches_selected_publication_configuration else "mismatch"
    return {
        "scope": "network_task_horizon_and_selected_learner_hyperparameters",
        "configuration": config_status,
        "hyperparameters_by_learner": per_learner,
        "all_selected_fields": (
            "match"
            if config_status == "match" and all(value == "match" for value in per_learner.values())
            else "mismatch"
        ),
    }


def build_run_spec(
    config: IPMNISTConfig,
    learner_ids: Sequence[str],
    seed_ids: Sequence[int],
    hyperparameters: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Build the closed execution specification shared by every v3 shard."""

    _require(config.input_dim == 784, "v3 MNIST plans require input_dim=784")
    _require(config.n_classes == 10, "v3 MNIST plans require n_classes=10")
    _require(
        config.task_length <= 60_000,
        "v3 MNIST task_length cannot exceed the 60000-row train split",
    )
    _require(
        _ALLOW_SYNTHETIC_CONFIG_FOR_TESTING or config == IPMNISTConfig(),
        "active v3 plans require the exact selected publication configuration",
    )
    learners = _validate_learner_ids(learner_ids)
    seeds = _validate_seed_ids(seed_ids)
    hp = _validated_hyperparameters(
        learners,
        _default_hyperparameters() if hyperparameters is None else hyperparameters,
    )
    return {
        "reference": {
            "id": "elsayed_mahmood_2024_ipmnist_selected_arms",
            "official_commit": PAPER_REFERENCE["official_commit"],
            "published_seed_count": int(PAPER_REFERENCE["n_seeds"]),
        },
        "learner_ids": list(learners),
        "seed_schedule": {
            "seed_ids": list(seeds),
            "seed_count": len(seeds),
            "known_consumed_seed_ids_excluded": list(KNOWN_CONSUMED_SEED_IDS),
            "freshness": "operator_reserved_pre_run_not_independently_attested",
        },
        "config": {**config.to_config(), "n_steps": config.n_steps},
        "hyperparameters": hp,
        "selected_publication_match": _selected_publication_match(config, hp),
        "measurement": dict(_MEASUREMENT_CONTRACT),
        "paired_comparison": dict(_PAIRED_COMPARISON_CONTRACT),
        "deviations": _closed_deviations(len(seeds)),
        "planned_shard_count": len(learners) * len(seeds),
    }


def _canonical_plan_argv(
    plan_locator: str,
    run_spec: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
) -> list[str]:
    config = cast(dict[str, int], run_spec["config"])
    locators = cast(dict[str, str], data_manifest["locators"])
    seed_ids = cast(list[int], run_spec["seed_schedule"]["seed_ids"])
    return [
        "plan",
        "--plan-out",
        plan_locator,
        "--seed-list",
        ",".join(str(seed) for seed in seed_ids),
        "--data-home",
        locators["data_home"],
        "--data-archive",
        locators["archive"],
        "--n-tasks",
        str(config["n_tasks"]),
        "--task-length",
        str(config["task_length"]),
        "--input-dim",
        str(config["input_dim"]),
        "--hidden1",
        str(config["hidden1"]),
        "--hidden2",
        str(config["hidden2"]),
        "--n-classes",
        str(config["n_classes"]),
    ]


def _build_plan_payload(
    plan_path: Path,
    config: IPMNISTConfig,
    seed_ids: Sequence[int],
    data_home: Path,
    data_archive: Path,
    *,
    learner_ids: Sequence[str] = LEARNER_IDS,
    hyperparameters: Mapping[str, Mapping[str, float]] | None = None,
    issued_unix: int | None = None,
    issuer_argv: Sequence[str] | None = None,
    invocation_origin: str = "payload_builder",
) -> dict[str, Any]:
    """Build a self-issued pre-run plan with exact data/source/runtime bindings."""

    issued = int(time.time()) if issued_unix is None else issued_unix
    _require(_is_int(issued) and issued >= 0, "issued_unix must be nonnegative")
    _require(issued <= int(time.time()) + 5, "issued_unix cannot be in the future")
    run_spec = build_run_spec(config, learner_ids, seed_ids, hyperparameters)
    data_manifest = _build_data_manifest(data_home, data_archive)
    _load_pinned_mnist(_lexical_absolute(data_home), context="pre-run offline preflight")
    source_closure = _build_source_import_closure()
    runtime = _build_runtime_manifest()
    process_argv = (
        list(issuer_argv)
        if issuer_argv is not None
        else ["python", "-m", "alberta_framework.benchmarks.upgd_ipmnist", "plan"]
    )
    process_argv = _validated_process_argv(process_argv, "issuer process argv")
    _require(
        invocation_origin in {"payload_builder", "direct_python_api", "cli"},
        "unsupported plan invocation origin",
    )
    prescribed_argv = _canonical_plan_argv(
        _lexical_absolute(plan_path).as_posix(),
        run_spec,
        data_manifest,
    )
    body = {
        "run_spec": run_spec,
        "run_spec_sha256": canonical_json_sha256(run_spec),
        "data_manifest": data_manifest,
        "data_manifest_sha256": canonical_json_sha256(data_manifest),
        "source_import_closure": source_closure,
        "source_import_closure_sha256": canonical_json_sha256(source_closure),
        "runtime_manifest": runtime,
        "runtime_manifest_sha256": canonical_json_sha256(runtime),
        "command_templates": {name: list(parts) for name, parts in _COMMAND_TEMPLATES.items()},
        "issuance": {
            "kind": "self_issued_pre_run_plan",
            "invocation_origin": invocation_origin,
            "prescribed_argv": prescribed_argv,
            "unattested_caller_argv": process_argv,
            "unattested_caller_argv_sha256": canonical_json_sha256(process_argv),
            "external_attestation_required_for_promotion": True,
            "external_attestation_present": False,
        },
    }
    return {
        "schema": PLAN_SCHEMA,
        "benchmark": BENCHMARK,
        "evidence_policy": dict(EVIDENCE_POLICY),
        "issued_unix": issued,
        "plan": body,
        "plan_sha256": canonical_json_sha256(body),
    }


def build_plan_payload(
    plan_path: Path,
    config: IPMNISTConfig,
    seed_ids: Sequence[int],
    data_home: Path,
    data_archive: Path,
    *,
    learner_ids: Sequence[str] = LEARNER_IDS,
    hyperparameters: Mapping[str, Mapping[str, float]] | None = None,
    issued_unix: int | None = None,
    issuer_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build an unissued payload through the direct payload-builder API."""

    return _build_plan_payload(
        plan_path,
        config,
        seed_ids,
        data_home,
        data_archive,
        learner_ids=learner_ids,
        hyperparameters=hyperparameters,
        issued_unix=issued_unix,
        issuer_argv=issuer_argv,
        invocation_origin="payload_builder",
    )


def _write_plan(
    path: Path,
    config: IPMNISTConfig,
    seed_ids: Sequence[int],
    data_home: Path,
    data_archive: Path,
    *,
    learner_ids: Sequence[str] = LEARNER_IDS,
    hyperparameters: Mapping[str, Mapping[str, float]] | None = None,
    issued_unix: int | None = None,
    issuer_argv: Sequence[str] | None = None,
    invocation_origin: str,
) -> Path:
    """Issue a canonical immutable pre-run plan at a new path."""

    destination = _preflight_new_output(path)
    payload = _build_plan_payload(
        destination,
        config,
        seed_ids,
        data_home,
        data_archive,
        learner_ids=learner_ids,
        hyperparameters=hyperparameters,
        issued_unix=issued_unix,
        issuer_argv=issuer_argv,
        invocation_origin=invocation_origin,
    )
    _validate_plan_payload(
        payload,
        verify_current_bindings=True,
        data_home=data_home,
        data_archive=data_archive,
    )
    return atomic_write_new_json(destination, payload)


def write_plan(
    path: Path,
    config: IPMNISTConfig,
    seed_ids: Sequence[int],
    data_home: Path,
    data_archive: Path,
    *,
    learner_ids: Sequence[str] = LEARNER_IDS,
    hyperparameters: Mapping[str, Mapping[str, float]] | None = None,
    issued_unix: int | None = None,
    issuer_argv: Sequence[str] | None = None,
) -> Path:
    """Issue a plan through the direct Python API entry point."""

    return _write_plan(
        path,
        config,
        seed_ids,
        data_home,
        data_archive,
        learner_ids=learner_ids,
        hyperparameters=hyperparameters,
        issued_unix=issued_unix,
        issuer_argv=issuer_argv,
        invocation_origin="direct_python_api",
    )


def _validate_config(value: object) -> IPMNISTConfig:
    config = _expect_dict(value, "run_spec.config")
    expected = set(IPMNISTConfig().to_config()) | {"n_steps"}
    _expect_exact_keys(config, expected, "run_spec.config")
    raw = {name: config[name] for name in IPMNISTConfig().to_config()}
    _require(all(_is_int(item) for item in raw.values()), "config values must be integers")
    try:
        parsed = IPMNISTConfig(**raw)
    except (TypeError, ValueError) as exc:
        raise UPGDIPMNISTV3Error(f"invalid run_spec.config: {exc}") from exc
    _require(config["n_steps"] == parsed.n_steps, "run_spec.config.n_steps is inconsistent")
    return parsed


def _validate_run_spec(value: object) -> tuple[dict[str, Any], IPMNISTConfig, tuple[int, ...]]:
    run_spec = _expect_dict(value, "run_spec")
    _expect_exact_keys(
        run_spec,
        {
            "reference",
            "learner_ids",
            "seed_schedule",
            "config",
            "hyperparameters",
            "selected_publication_match",
            "measurement",
            "paired_comparison",
            "deviations",
            "planned_shard_count",
        },
        "run_spec",
    )
    learners_raw = _expect_list(run_spec["learner_ids"], "run_spec.learner_ids")
    _require(all(isinstance(item, str) for item in learners_raw), "learner IDs must be strings")
    learners = _validate_learner_ids(cast(list[str], learners_raw))
    schedule = _expect_dict(run_spec["seed_schedule"], "run_spec.seed_schedule")
    _expect_exact_keys(
        schedule,
        {
            "seed_ids",
            "seed_count",
            "known_consumed_seed_ids_excluded",
            "freshness",
        },
        "run_spec.seed_schedule",
    )
    seed_values = _expect_list(schedule["seed_ids"], "run_spec.seed_schedule.seed_ids")
    seeds = _validate_seed_ids(cast(list[int], seed_values))
    config = _validate_config(run_spec["config"])
    hyperparameters_raw = _expect_dict(run_spec["hyperparameters"], "run_spec.hyperparameters")
    hp_values: dict[str, Mapping[str, float]] = {}
    for learner, raw_values in hyperparameters_raw.items():
        hp_values[learner] = cast(Mapping[str, float], _expect_dict(raw_values, learner))
    expected = build_run_spec(config, learners, seeds, hp_values)
    _require(
        _json_exact_equal(run_spec, expected),
        "run_spec differs from its derived closed specification",
    )
    return run_spec, config, seeds


def _validate_data_manifest(value: object) -> dict[str, Any]:
    manifest = _expect_dict(value, "data_manifest")
    _expect_exact_keys(
        manifest,
        {
            "schema",
            "dataset_id",
            "content",
            "locators",
            "complete_offline_cache",
            "materialized_arrays",
            "loader_contract",
        },
        "data_manifest",
    )
    _require(manifest["schema"] == DATA_SCHEMA, "wrong data manifest schema")
    _require(
        manifest["dataset_id"] == "openml_mnist_784_v1_first_60000",
        "wrong dataset identity",
    )
    content = _expect_dict(manifest["content"], "data_manifest.content")
    _expect_exact_keys(content, {"byte_size", "sha256"}, "data_manifest.content")
    _require(
        _is_int(content["byte_size"])
        and content["byte_size"] == MNIST_ARCHIVE_BYTE_SIZE,
        "dataset byte_size differs from the pinned OpenML payload",
    )
    _require(
        content["sha256"] == MNIST_ARCHIVE_SHA256,
        "dataset SHA-256 differs from the pinned OpenML payload",
    )
    _validate_data_locators(manifest["locators"], "data_manifest.locators")
    cache = _expect_dict(
        manifest["complete_offline_cache"],
        "data_manifest.complete_offline_cache",
    )
    _expect_exact_keys(
        cache,
        {"network_access_required", "files"},
        "data_manifest.complete_offline_cache",
    )
    _require(
        cache["network_access_required"] is False,
        "v3 data loading may not require network access",
    )
    cache_files = _expect_list(
        cache["files"],
        "data_manifest.complete_offline_cache.files",
    )
    expected_cache = [
        {
            "locator": relative,
            "byte_size": byte_size,
            "sha256": digest,
        }
        for relative, byte_size, digest in OPENML_CACHE_MEMBER_IDENTITIES
    ]
    _require(
        _json_exact_equal(cache_files, expected_cache),
        "complete OpenML cache identities differ",
    )
    arrays = _expect_dict(
        manifest["materialized_arrays"], "data_manifest.materialized_arrays"
    )
    _require(
        _json_exact_equal(
            arrays,
            {
                "x": {
                    "dtype": "<f4",
                    "shape": [60_000, 784],
                    "sha256": MNIST_TRAIN_X_SHA256,
                },
                "y": {
                    "dtype": "<i4",
                    "shape": [60_000],
                    "sha256": MNIST_TRAIN_Y_SHA256,
                },
            },
        ),
        "materialized MNIST array identities differ",
    )
    loader = _expect_dict(manifest["loader_contract"], "data_manifest.loader_contract")
    _require(_json_exact_equal(loader, _DATA_LOADER_CONTRACT), "data loader contract differs")
    return manifest


def _validate_source_closure(value: object) -> dict[str, Any]:
    closure = _expect_dict(value, "source_import_closure")
    _expect_exact_keys(
        closure,
        {"schema", "closure_kind", "root_modules", "files"},
        "source_import_closure",
    )
    _require(closure["schema"] == SOURCE_CLOSURE_SCHEMA, "wrong source closure schema")
    _require(
        closure["closure_kind"] == "static_transitive_local_python_imports_plus_lockfiles",
        "wrong source closure kind",
    )
    _require(
        _json_exact_equal(closure["root_modules"], list(_SOURCE_ROOT_MODULES)),
        "source roots differ",
    )
    files = _expect_list(closure["files"], "source_import_closure.files")
    _require(bool(files), "source import closure cannot be empty")
    modules: list[str] = []
    for index, raw_entry in enumerate(files):
        where = f"source_import_closure.files[{index}]"
        entry = _expect_dict(raw_entry, where)
        _expect_exact_keys(entry, {"module", "locator", "byte_size", "sha256"}, where)
        _require(
            isinstance(entry["module"], str) and bool(entry["module"]),
            f"{where}.module invalid",
        )
        _require(
            isinstance(entry["locator"], str) and bool(entry["locator"]),
            f"{where}.locator invalid",
        )
        locator = PurePosixPath(cast(str, entry["locator"]))
        _require(
            not locator.is_absolute()
            and ".." not in locator.parts
            and "." not in locator.parts
            and locator.as_posix() == entry["locator"],
            f"{where}.locator is unsafe or noncanonical",
        )
        _require(
            _is_int(entry["byte_size"]) and entry["byte_size"] >= 0,
            f"{where}.byte_size invalid",
        )
        _require(_is_sha256(entry["sha256"]), f"{where}.sha256 invalid")
        modules.append(entry["module"])
    _require(modules == sorted(set(modules)), "source modules must be unique and sorted")
    _require(set(_SOURCE_ROOT_MODULES) <= set(modules), "source root modules are absent")
    _require(
        {f"file:{relative}" for relative in _SOURCE_AUXILIARY_PATHS} <= set(modules),
        "source closure is missing lock/configuration files",
    )
    entries_by_module = {
        cast(str, _expect_dict(entry, "source closure entry")["module"]): cast(
            str, _expect_dict(entry, "source closure entry")["locator"]
        )
        for entry in files
    }
    for relative in _SOURCE_AUXILIARY_PATHS:
        _require(
            entries_by_module[f"file:{relative}"] == relative,
            f"source closure locator differs for {relative}",
        )
    return closure


def _validate_runtime(value: object, where: str) -> dict[str, Any]:
    runtime = _expect_dict(value, where)
    keys = {
        "schema",
        "python",
        "python_implementation",
        "python_executable",
        "jax",
        "jaxlib",
        "numpy",
        "scikit_learn",
        "distribution_content",
        "distribution_content_scope",
        "platform_system",
        "platform_machine",
        "platform_release",
        "jax_backend",
        "jax_devices",
        "jax_device_details",
        "jax_enable_x64",
        "jax_config",
        "execution_environment",
    }
    _expect_exact_keys(runtime, keys, where)
    _require(runtime["schema"] == RUNTIME_SCHEMA, f"{where} has wrong schema")
    for key in keys - {
        "schema",
        "jax_enable_x64",
        "jax_devices",
        "jax_device_details",
        "jax_config",
        "execution_environment",
        "python_executable",
        "distribution_content",
        "distribution_content_scope",
    }:
        _require(isinstance(runtime[key], str) and runtime[key], f"{where}.{key} invalid")
    executable = _expect_dict(runtime["python_executable"], f"{where}.python_executable")
    _expect_exact_keys(
        executable,
        {"locator", "byte_size", "sha256"},
        f"{where}.python_executable",
    )
    _canonical_absolute_locator(
        executable["locator"],
        f"{where}.python_executable.locator",
    )
    _require(
        _is_int(executable["byte_size"]) and executable["byte_size"] > 0,
        f"{where}.python_executable.byte_size invalid",
    )
    _require(
        _is_sha256(executable["sha256"]),
        f"{where}.python_executable.sha256 invalid",
    )
    distributions = _expect_dict(
        runtime["distribution_content"],
        f"{where}.distribution_content",
    )
    _expect_exact_keys(
        distributions,
        set(_RUNTIME_CONTENT_DISTRIBUTIONS),
        f"{where}.distribution_content",
    )
    for name, raw_identity in distributions.items():
        identity = _expect_dict(raw_identity, f"{where}.distribution_content.{name}")
        _expect_exact_keys(
            identity,
            {"status", "file_count", "total_bytes", "sha256"},
            f"{where}.distribution_content.{name}",
        )
        _require(
            identity["status"] == "content_hashed",
            f"{where}.distribution_content.{name}.status invalid",
        )
        _require(
            _is_int(identity["file_count"]) and identity["file_count"] >= 0,
            f"{where}.distribution_content.{name}.file_count invalid",
        )
        _require(
            _is_int(identity["total_bytes"]) and identity["total_bytes"] >= 0,
            f"{where}.distribution_content.{name}.total_bytes invalid",
        )
        _require(
            _is_sha256(identity["sha256"]),
            f"{where}.distribution_content.{name}.sha256 invalid",
        )
    scope = _expect_dict(
        runtime["distribution_content_scope"],
        f"{where}.distribution_content_scope",
    )
    _require(
        _json_exact_equal(
            scope,
            {
                "kind": "explicit_python_execution_distribution_set",
                "distribution_names": list(_RUNTIME_CONTENT_DISTRIBUTIONS),
                "dynamic_import_closure_claimed": False,
                "native_system_library_closure_claimed": False,
            },
        ),
        f"{where}.distribution_content_scope differs",
    )
    devices = _expect_list(runtime["jax_devices"], f"{where}.jax_devices")
    _require(
        bool(devices) and all(isinstance(item, str) and bool(item) for item in devices),
        f"{where}.jax_devices invalid",
    )
    details = _expect_list(runtime["jax_device_details"], f"{where}.jax_device_details")
    _require(len(details) == len(devices), f"{where}.jax_device_details width differs")
    for index, raw_detail in enumerate(details):
        detail = _expect_dict(raw_detail, f"{where}.jax_device_details[{index}]")
        _expect_exact_keys(
            detail,
            {"id", "platform", "device_kind", "process_index", "runtime_type"},
            f"{where}.jax_device_details[{index}]",
        )
        _require(_is_int(detail["id"]) and detail["id"] >= 0, "device id invalid")
        _require(
            _is_int(detail["process_index"]) and detail["process_index"] >= 0,
            "device process_index invalid",
        )
        for key in ("platform", "device_kind", "runtime_type"):
            _require(
                isinstance(detail[key], str) and bool(detail[key]),
                f"{where}.jax_device_details[{index}].{key} invalid",
            )
    _require(type(runtime["jax_enable_x64"]) is bool, f"{where}.jax_enable_x64 invalid")
    jax_config = _expect_dict(runtime["jax_config"], f"{where}.jax_config")
    _require(bool(jax_config), f"{where}.jax_config must not be empty")
    for name, item in jax_config.items():
        _require(
            isinstance(name, str)
            and type(item) in {str, int, float, bool, type(None)}
            and (type(item) is not float or math.isfinite(item)),
            f"{where}.jax_config[{name!r}] invalid",
        )
    environment = _expect_dict(
        runtime["execution_environment"], f"{where}.execution_environment"
    )
    _expect_exact_keys(
        environment,
        set(_RUNTIME_ENVIRONMENT_NAMES),
        f"{where}.execution_environment",
    )
    _require(
        all(item is None or isinstance(item, str) for item in environment.values()),
        f"{where}.execution_environment values invalid",
    )
    return runtime


def _data_manifest_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return every data binding except the intentionally relocatable paths."""

    return {key: item for key, item in value.items() if key != "locators"}


def _validate_current_bindings_against_plan(
    plan: Mapping[str, Any],
    *,
    data_home: Path | None,
    data_archive: Path | None,
) -> tuple[Path, Path]:
    """Rebuild and compare all current source, runtime, and data bindings."""

    body = cast(dict[str, Any], plan["plan"])
    closure = cast(dict[str, Any], body["source_import_closure"])
    runtime = cast(dict[str, Any], body["runtime_manifest"])
    data_manifest = cast(dict[str, Any], body["data_manifest"])
    home, archive = _effective_data_paths(plan, data_home, data_archive)
    # Materialize first. The loader performs its own pre-load complete-cache
    # check and network denial; the manifest rebuild below then proves the
    # cache also retained its exact identity after sklearn finished reading it.
    _load_pinned_mnist(home, context="current binding verification")
    _require(
        _json_exact_equal(closure, _build_source_import_closure()),
        "current source import closure differs from the issued plan",
    )
    _require(
        _json_exact_equal(runtime, _build_runtime_manifest()),
        "current runtime differs from the plan",
    )
    current_data = _build_data_manifest(home, archive)
    _require(
        _json_exact_equal(
            _data_manifest_identity(current_data),
            _data_manifest_identity(data_manifest),
        ),
        "current dataset/cache/loader bindings differ from the issued plan",
    )
    return home, archive


def _validate_plan_payload(
    value: object,
    *,
    verify_current_bindings: bool,
    data_home: Path | None = None,
    data_archive: Path | None = None,
) -> dict[str, Any]:
    plan = _expect_dict(value, "run_plan")
    _reject_legacy_marker(plan)
    _expect_exact_keys(
        plan,
        {"schema", "benchmark", "evidence_policy", "issued_unix", "plan", "plan_sha256"},
        "run_plan",
    )
    _require(plan["schema"] == PLAN_SCHEMA, "wrong v3 plan schema")
    _require(plan["benchmark"] == BENCHMARK, "wrong benchmark")
    _require(
        _json_exact_equal(plan["evidence_policy"], EVIDENCE_POLICY),
        "evidence policy differs",
    )
    _require(
        _is_int(plan["issued_unix"])
        and 0 <= plan["issued_unix"] <= int(time.time()) + 5,
        "issued_unix must be a nonnegative timestamp no more than five seconds in the future",
    )
    body = _expect_dict(plan["plan"], "run_plan.plan")
    _expect_exact_keys(
        body,
        {
            "run_spec",
            "run_spec_sha256",
            "data_manifest",
            "data_manifest_sha256",
            "source_import_closure",
            "source_import_closure_sha256",
            "runtime_manifest",
            "runtime_manifest_sha256",
            "command_templates",
            "issuance",
        },
        "run_plan.plan",
    )
    run_spec, _, _ = _validate_run_spec(body["run_spec"])
    _require(
        body["run_spec_sha256"] == canonical_json_sha256(run_spec),
        "run_spec_sha256 mismatch",
    )
    data_manifest = _validate_data_manifest(body["data_manifest"])
    _require(
        body["data_manifest_sha256"] == canonical_json_sha256(data_manifest),
        "data_manifest_sha256 mismatch",
    )
    closure = _validate_source_closure(body["source_import_closure"])
    _require(
        body["source_import_closure_sha256"] == canonical_json_sha256(closure),
        "source_import_closure_sha256 mismatch",
    )
    runtime = _validate_runtime(body["runtime_manifest"], "runtime_manifest")
    _require(
        body["runtime_manifest_sha256"] == canonical_json_sha256(runtime),
        "runtime_manifest_sha256 mismatch",
    )
    commands = _expect_dict(body["command_templates"], "command_templates")
    _require(_json_exact_equal(commands, _COMMAND_TEMPLATES), "command templates differ")
    issuance = _expect_dict(body["issuance"], "issuance")
    _expect_exact_keys(
        issuance,
        {
            "kind",
            "invocation_origin",
            "prescribed_argv",
            "unattested_caller_argv",
            "unattested_caller_argv_sha256",
            "external_attestation_required_for_promotion",
            "external_attestation_present",
        },
        "issuance",
    )
    prescribed_argv = _expect_list(
        issuance["prescribed_argv"],
        "issuance.prescribed_argv",
    )
    process_argv = _expect_list(
        issuance["unattested_caller_argv"],
        "issuance.unattested_caller_argv",
    )
    process_argv = _validated_process_argv(
        cast(list[str], process_argv),
        "issuer process argv",
    )
    _require(
        issuance["unattested_caller_argv_sha256"] == canonical_json_sha256(process_argv),
        "issuer unattested caller argv digest mismatch",
    )
    _require(
        issuance["invocation_origin"]
        in {"payload_builder", "direct_python_api", "cli"},
        "plan invocation origin invalid",
    )
    _require(
        len(prescribed_argv) == 21
        and prescribed_argv[:2] == ["plan", "--plan-out"]
        and isinstance(prescribed_argv[2], str)
        and bool(prescribed_argv[2]),
        "prescribed plan argv shape differs",
    )
    plan_locator = _canonical_absolute_locator(
        prescribed_argv[2],
        "issuance.prescribed_argv plan locator",
    )
    expected_canonical_argv = _canonical_plan_argv(
        plan_locator,
        run_spec,
        data_manifest,
    )
    _require(
        _json_exact_equal(prescribed_argv, expected_canonical_argv),
        "prescribed plan argv differs from the plan",
    )
    _require(
        _json_exact_equal(
            issuance,
            {
                "kind": "self_issued_pre_run_plan",
                "invocation_origin": issuance["invocation_origin"],
                "prescribed_argv": prescribed_argv,
                "unattested_caller_argv": process_argv,
                "unattested_caller_argv_sha256": canonical_json_sha256(process_argv),
                "external_attestation_required_for_promotion": True,
                "external_attestation_present": False,
            },
        ),
        "issuance envelope differs",
    )
    _require(plan["plan_sha256"] == canonical_json_sha256(body), "plan_sha256 mismatch")
    if verify_current_bindings:
        _validate_current_bindings_against_plan(
            plan,
            data_home=data_home,
            data_archive=data_archive,
        )
    return plan


def _read_validated_plan(
    path: Path,
    *,
    verify_current_bindings: bool,
    data_home: Path | None = None,
    data_archive: Path | None = None,
) -> tuple[bytes, dict[str, Any]]:
    raw, plan = _read_strict_json(path)
    _validate_plan_payload(
        plan,
        verify_current_bindings=verify_current_bindings,
        data_home=data_home,
        data_archive=data_archive,
    )
    _require(raw == canonical_json_bytes(plan), "run plan is not canonically encoded")
    issuance = cast(dict[str, Any], plan["plan"]["issuance"])
    prescribed_argv = cast(list[str], issuance["prescribed_argv"])
    _require(
        prescribed_argv[2] == _lexical_absolute(path).as_posix(),
        "run plan locator differs from its prescribed issuance output",
    )
    return raw, plan


def _effective_data_paths(
    plan: Mapping[str, Any],
    data_home: Path | None,
    data_archive: Path | None,
) -> tuple[Path, Path]:
    manifest = cast(dict[str, Any], plan["plan"]["data_manifest"])
    planned = _validate_data_locators(manifest["locators"], "data_manifest.locators")
    home = (
        _lexical_absolute(data_home)
        if data_home is not None
        else Path(planned["data_home"])
    )
    archive = (
        _lexical_absolute(data_archive)
        if data_archive is not None
        else _lexical_absolute(home / MNIST_ARCHIVE_RELATIVE_PATH)
    )
    _require(
        archive == _lexical_absolute(home / MNIST_ARCHIVE_RELATIVE_PATH),
        "effective data archive must be the canonical member of data_home",
    )
    return home, archive


def validate_plan(
    path: Path,
    *,
    verify_current_bindings: bool = True,
    data_home: Path | None = None,
    data_archive: Path | None = None,
) -> UPGDIPMNISTV3Validation:
    """Validate a v3 plan without granting evidence promotion."""

    try:
        _require(
            verify_current_bindings,
            "public plan validity requires current source/runtime/data bindings",
        )
        raw, plan = _read_validated_plan(
            path,
            verify_current_bindings=verify_current_bindings,
            data_home=data_home,
            data_archive=data_archive,
        )
        final_raw, final_plan = _read_validated_plan(
            path,
            verify_current_bindings=True,
            data_home=data_home,
            data_archive=data_archive,
        )
        _require(final_raw == raw, "plan bytes changed during validation")
        _require(_json_exact_equal(final_plan, plan), "plan changed during validation")
        _validate_current_bindings_against_plan(
            final_plan,
            data_home=data_home,
            data_archive=data_archive,
        )
    except Exception as exc:
        return UPGDIPMNISTV3Validation(False, False, (str(exc),))
    return UPGDIPMNISTV3Validation(True, False, ())


def _plan_binding(plan_path: Path, raw: bytes, plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "locator": _lexical_absolute(plan_path).as_posix(),
        "byte_size": len(raw),
        "sha256": sha256_bytes(raw),
        "plan_sha256": plan["plan_sha256"],
    }


def _seed_reservation_path(
    plan_path: Path,
    plan: Mapping[str, Any],
    learner_id: str,
    seed_id: int,
) -> Path:
    plan_locator = _lexical_absolute(plan_path)
    return (
        plan_locator.parent
        / f".{plan_locator.name}.seed-reservations"
        / cast(str, plan["plan_sha256"])
        / f"{learner_id}-{seed_id}.reservation.json"
    )


def _validate_reservation_payload(
    value: object,
    *,
    plan_path: Path,
    plan_raw: bytes,
    plan: Mapping[str, Any],
    learner_id: str,
    seed_id: int,
    partial_locator: str,
) -> dict[str, Any]:
    reservation = _expect_dict(value, "seed_reservation")
    _expect_exact_keys(
        reservation,
        {
            "schema",
            "benchmark",
            "evidence_policy",
            "plan_binding",
            "learner_id",
            "seed_id",
            "partial_locator",
            "reserved_unix",
            "reservation_sha256",
        },
        "seed_reservation",
    )
    _require(reservation["schema"] == RESERVATION_SCHEMA, "wrong seed reservation schema")
    _require(reservation["benchmark"] == BENCHMARK, "wrong seed reservation benchmark")
    _require(
        _json_exact_equal(reservation["evidence_policy"], EVIDENCE_POLICY),
        "seed reservation evidence policy differs",
    )
    _validate_plan_binding(reservation["plan_binding"], plan_path, plan_raw, plan)
    _require(reservation["learner_id"] == learner_id, "seed reservation learner differs")
    _require(reservation["seed_id"] == seed_id, "seed reservation seed differs")
    _require(
        reservation["partial_locator"] == partial_locator,
        "seed reservation partial locator differs",
    )
    _canonical_absolute_locator(
        reservation["partial_locator"],
        "seed_reservation.partial_locator",
    )
    _require(
        _is_int(reservation["reserved_unix"])
        and plan["issued_unix"]
        <= reservation["reserved_unix"]
        <= int(time.time()) + 5,
        "seed reservation time invalid",
    )
    body = {key: item for key, item in reservation.items() if key != "reservation_sha256"}
    _require(
        reservation["reservation_sha256"] == canonical_json_sha256(body),
        "seed reservation digest mismatch",
    )
    return reservation


def _acquire_seed_reservation(
    *,
    plan_path: Path,
    plan_raw: bytes,
    plan: Mapping[str, Any],
    learner_id: str,
    seed_id: int,
    partial_path: Path,
) -> dict[str, Any]:
    """Consume one learner/seed exactly once before any benchmark execution."""

    locator = _seed_reservation_path(plan_path, plan, learner_id, seed_id)
    partial_locator = _lexical_absolute(partial_path).as_posix()
    _require(
        locator != _lexical_absolute(partial_path),
        "partial output cannot occupy its persistent seed reservation path",
    )
    body = {
        "schema": RESERVATION_SCHEMA,
        "benchmark": BENCHMARK,
        "evidence_policy": dict(EVIDENCE_POLICY),
        "plan_binding": _plan_binding(plan_path, plan_raw, plan),
        "learner_id": learner_id,
        "seed_id": seed_id,
        "partial_locator": partial_locator,
        "reserved_unix": int(time.time()),
    }
    reservation = {**body, "reservation_sha256": canonical_json_sha256(body)}
    atomic_write_new_json(locator, reservation)
    raw, persisted = _read_strict_json(locator)
    _require(raw == canonical_json_bytes(persisted), "seed reservation is not canonical")
    _validate_reservation_payload(
        persisted,
        plan_path=plan_path,
        plan_raw=plan_raw,
        plan=plan,
        learner_id=learner_id,
        seed_id=seed_id,
        partial_locator=partial_locator,
    )
    return {
        "locator": locator.as_posix(),
        "byte_size": len(raw),
        "sha256": sha256_bytes(raw),
        "reservation_sha256": persisted["reservation_sha256"],
    }


def _validate_reservation_binding(
    value: object,
    *,
    plan_path: Path,
    plan_raw: bytes,
    plan: Mapping[str, Any],
    learner_id: str,
    seed_id: int,
    partial_locator: str,
) -> dict[str, Any]:
    binding = _expect_dict(value, "execution.seed_reservation_binding")
    _expect_exact_keys(
        binding,
        {"locator", "byte_size", "sha256", "reservation_sha256"},
        "execution.seed_reservation_binding",
    )
    locator = _canonical_absolute_locator(
        binding["locator"],
        "execution.seed_reservation_binding.locator",
    )
    raw, reservation = _read_strict_json(Path(locator))
    _require(raw == canonical_json_bytes(reservation), "seed reservation is not canonical")
    _require(binding["byte_size"] == len(raw), "seed reservation byte_size mismatch")
    _require(binding["sha256"] == sha256_bytes(raw), "seed reservation byte hash mismatch")
    validated = _validate_reservation_payload(
        reservation,
        plan_path=plan_path,
        plan_raw=plan_raw,
        plan=plan,
        learner_id=learner_id,
        seed_id=seed_id,
        partial_locator=partial_locator,
    )
    expected_locator = _seed_reservation_path(
        plan_path,
        plan,
        learner_id,
        seed_id,
    ).as_posix()
    _require(
        locator == expected_locator,
        "seed reservation locator differs from the exact plan-scoped path",
    )
    _require(
        binding["reservation_sha256"] == validated["reservation_sha256"],
        "seed reservation content digest mismatch",
    )
    return validated


def _canonical_worker_argv(
    execution_origin: str,
    plan_locator: str,
    learner_id: str,
    seed_id: int,
    partial_locator: str,
    data_home_locator: str,
    data_archive_locator: str,
) -> list[str]:
    if execution_origin in {"direct_supplied_result", "direct_supplied_result_builder"}:
        return [
            "direct_api",
            (
                "build_partial_payload"
                if execution_origin == "direct_supplied_result_builder"
                else "write_partial_for_result"
            ),
            "--plan",
            plan_locator,
            "--learner-id",
            learner_id,
            "--seed-id",
            str(seed_id),
            "--partial-out",
            partial_locator,
            "--data-home",
            data_home_locator,
            "--data-archive",
            data_archive_locator,
        ]
    _require(execution_origin == "benchmark_runner", "unsupported shard execution origin")
    return [
        "shard",
        "--plan",
        plan_locator,
        "--learner-id",
        learner_id,
        "--seed-id",
        str(seed_id),
        "--partial-out",
        partial_locator,
        "--data-home",
        data_home_locator,
        "--data-archive",
        data_archive_locator,
    ]


def _validate_result_against_plan(
    result: IPMNISTRunResult,
    run_spec: Mapping[str, Any],
) -> None:
    _require(result.learner in LEARNER_IDS, "result learner is unsupported")
    _require(len(result.seeds) == 1, "a v3 partial contains exactly one seed")
    _require(result.learner in run_spec["learner_ids"], "result learner is not planned")
    _require(result.seeds[0] in run_spec["seed_schedule"]["seed_ids"], "seed is not planned")
    config = _validate_config(run_spec["config"])
    _require(result.config == config, "result config differs from the plan")
    _require(
        _json_exact_equal(
            result.hyperparameters,
            run_spec["hyperparameters"][result.learner],
        ),
        "result hyperparameters differ from the plan",
    )
    shape = (1, config.n_tasks)
    for name, values in (
        ("per_task_accuracy", result.per_task_accuracy),
        ("per_task_loss", result.per_task_loss),
        ("per_task_plasticity", result.per_task_plasticity),
    ):
        array = np.asarray(values, dtype=np.float64)
        _require(array.shape == shape, f"{name} must have shape {shape} before serialization")
        _require(bool(np.all(np.isfinite(array))), f"{name} contains non-finite values")


def _build_partial_payload(
    plan_path: Path,
    result: IPMNISTRunResult,
    partial_path: Path,
    *,
    data_home: Path | None = None,
    data_archive: Path | None = None,
    process_argv: Sequence[str] = (),
    execution_origin: str,
    reservation_binding: Mapping[str, Any] | None,
    started_unix: int | None = None,
    finished_unix: int | None = None,
) -> dict[str, Any]:
    """Build a one-learner/one-seed shard bound to an issued plan."""

    plan_raw, plan = _read_validated_plan(
        plan_path,
        verify_current_bindings=True,
        data_home=data_home,
        data_archive=data_archive,
    )
    body = cast(dict[str, Any], plan["plan"])
    run_spec = cast(dict[str, Any], body["run_spec"])
    _validate_result_against_plan(result, run_spec)
    started = int(time.time()) if started_unix is None else started_unix
    finished = int(time.time()) if finished_unix is None else finished_unix
    _require(_is_int(started) and started >= 0, "started_unix invalid")
    _require(_is_int(finished) and finished >= started, "finished_unix invalid")
    _require(started >= plan["issued_unix"], "shard cannot start before plan issuance")
    _require(
        _is_number(result.wall_clock_seconds) and result.wall_clock_seconds >= 0.0,
        "result wall_clock_seconds must be finite and nonnegative",
    )
    argv = _validated_process_argv(process_argv, "self-reported process argv")
    learner = result.learner
    seed = result.seeds[0]
    plan_locator = _lexical_absolute(plan_path).as_posix()
    partial_locator = _lexical_absolute(partial_path).as_posix()
    planned_locators = cast(dict[str, str], body["data_manifest"]["locators"])
    effective_data_home = (
        _lexical_absolute(data_home).as_posix()
        if data_home is not None
        else planned_locators["data_home"]
    )
    effective_data_archive = (
        _lexical_absolute(data_archive).as_posix()
        if data_archive is not None
        else _lexical_absolute(
            Path(effective_data_home) / MNIST_ARCHIVE_RELATIVE_PATH
        ).as_posix()
    )
    runtime = _build_runtime_manifest()
    _require(
        execution_origin
        in {
            "benchmark_runner",
            "direct_supplied_result_builder",
            "direct_supplied_result",
        },
        "unsupported shard execution origin",
    )
    prescribed_worker_argv = _canonical_worker_argv(
        execution_origin,
        plan_locator,
        learner,
        seed,
        partial_locator,
        effective_data_home,
        effective_data_archive,
    )
    return {
        "schema": PARTIAL_SCHEMA,
        "benchmark": BENCHMARK,
        "evidence_policy": dict(EVIDENCE_POLICY),
        "plan_binding": _plan_binding(plan_path, plan_raw, plan),
        "learner_id": learner,
        "seed_id": seed,
        "execution": {
            "execution_origin": execution_origin,
            "partial_locator": partial_locator,
            "started_unix": started,
            "finished_unix": finished,
            "duration_seconds": float(result.wall_clock_seconds),
            "duration_semantics": _DURATION_SEMANTICS,
            "prescribed_worker_argv": prescribed_worker_argv,
            "unattested_caller_argv": argv,
            "unattested_caller_argv_sha256": canonical_json_sha256(argv),
            "data_locators_used": {
                "data_home": effective_data_home,
                "archive": effective_data_archive,
            },
            "runtime_manifest": runtime,
            "runtime_manifest_sha256": canonical_json_sha256(runtime),
            "source_import_closure_sha256": body["source_import_closure_sha256"],
            "data_content": dict(body["data_manifest"]["content"]),
            "seed_reservation_binding": (
                None if reservation_binding is None else dict(reservation_binding)
            ),
            "external_execution_attestation_present": False,
        },
        "measurements": _measurement_payload_from_result(result),
    }


def build_partial_payload(
    plan_path: Path,
    result: IPMNISTRunResult,
    partial_path: Path,
    *,
    data_home: Path | None = None,
    data_archive: Path | None = None,
    process_argv: Sequence[str] = (),
    started_unix: int | None = None,
    finished_unix: int | None = None,
) -> dict[str, Any]:
    """Build an explicitly supplied-result diagnostic payload."""

    return _build_partial_payload(
        plan_path,
        result,
        partial_path,
        data_home=data_home,
        data_archive=data_archive,
        process_argv=process_argv,
        execution_origin="direct_supplied_result_builder",
        reservation_binding=None,
        started_unix=started_unix,
        finished_unix=finished_unix,
    )


def _write_partial_for_result(
    plan_path: Path,
    result: IPMNISTRunResult,
    partial_path: Path,
    *,
    data_home: Path | None = None,
    data_archive: Path | None = None,
    process_argv: Sequence[str] = (),
    execution_origin: str,
    reservation_binding: Mapping[str, Any] | None,
    started_unix: int | None = None,
    finished_unix: int | None = None,
) -> Path:
    """Write a supplied single-seed result as one immutable v3 shard."""

    argv = _validated_process_argv(process_argv, "worker process argv")
    destination = _preflight_new_output(partial_path)
    payload = _build_partial_payload(
        plan_path,
        result,
        destination,
        data_home=data_home,
        data_archive=data_archive,
        process_argv=argv,
        execution_origin=execution_origin,
        reservation_binding=reservation_binding,
        started_unix=started_unix,
        finished_unix=finished_unix,
    )
    plan_raw, plan = _read_validated_plan(
        plan_path,
        verify_current_bindings=True,
        data_home=data_home,
        data_archive=data_archive,
    )
    _validate_partial_payload(payload, plan_path, plan_raw, plan)
    return atomic_write_new_json(destination, payload)


def write_partial_for_result(
    plan_path: Path,
    result: IPMNISTRunResult,
    partial_path: Path,
    *,
    data_home: Path | None = None,
    data_archive: Path | None = None,
    process_argv: Sequence[str] = (),
    started_unix: int | None = None,
    finished_unix: int | None = None,
) -> Path:
    """Publish an explicitly supplied-result diagnostic shard."""

    return _write_partial_for_result(
        plan_path,
        result,
        partial_path,
        data_home=data_home,
        data_archive=data_archive,
        process_argv=process_argv,
        execution_origin="direct_supplied_result",
        reservation_binding=None,
        started_unix=started_unix,
        finished_unix=finished_unix,
    )


def _validate_plan_binding(
    value: object,
    plan_path: Path,
    plan_raw: bytes,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _expect_dict(value, "plan_binding")
    _expect_exact_keys(
        binding,
        {"locator", "byte_size", "sha256", "plan_sha256"},
        "plan_binding",
    )
    locator = _canonical_absolute_locator(binding["locator"], "plan_binding.locator")
    _require(
        locator == _lexical_absolute(plan_path).as_posix(),
        "plan binding locator differs from the immutable external plan",
    )
    _require(
        _is_int(binding["byte_size"]) and binding["byte_size"] == len(plan_raw),
        "plan byte_size mismatch",
    )
    _require(binding["sha256"] == sha256_bytes(plan_raw), "plan byte hash mismatch")
    _require(binding["plan_sha256"] == plan["plan_sha256"], "plan digest mismatch")
    return binding


def _validate_measurement_vector(
    value: object,
    name: str,
    n_tasks: int,
    *,
    upper: float | None,
) -> np.ndarray:
    values = _expect_list(value, f"measurements.{name}")
    _require(len(values) == n_tasks, f"measurements.{name} length differs from n_tasks")
    _require(all(_is_number(item) for item in values), f"measurements.{name} must be finite")
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1, f"measurements.{name} must be one-dimensional")
    _require(bool(np.all(array >= 0.0)), f"measurements.{name} is below zero")
    if upper is not None:
        _require(bool(np.all(array <= upper)), f"measurements.{name} exceeds {upper}")
    return array


def _validate_partial_payload(
    value: object,
    plan_path: Path,
    plan_raw: bytes,
    plan: dict[str, Any],
) -> dict[str, Any]:
    partial = _expect_dict(value, "partial")
    _reject_legacy_marker(partial)
    _expect_exact_keys(
        partial,
        {
            "schema",
            "benchmark",
            "evidence_policy",
            "plan_binding",
            "learner_id",
            "seed_id",
            "execution",
            "measurements",
        },
        "partial",
    )
    _require(partial["schema"] == PARTIAL_SCHEMA, "wrong v3 partial schema")
    _require(partial["benchmark"] == BENCHMARK, "wrong partial benchmark")
    _require(
        _json_exact_equal(partial["evidence_policy"], EVIDENCE_POLICY),
        "partial evidence policy differs",
    )
    binding = _validate_plan_binding(partial["plan_binding"], plan_path, plan_raw, plan)
    body = cast(dict[str, Any], plan["plan"])
    run_spec, config, seeds = _validate_run_spec(body["run_spec"])
    learner = partial["learner_id"]
    seed = partial["seed_id"]
    _require(isinstance(learner, str) and learner in run_spec["learner_ids"], "unplanned learner")
    _require(_is_int(seed) and seed in seeds, "unplanned seed")
    execution = _expect_dict(partial["execution"], "execution")
    _expect_exact_keys(
        execution,
        {
            "execution_origin",
            "partial_locator",
            "started_unix",
            "finished_unix",
            "duration_seconds",
            "duration_semantics",
            "prescribed_worker_argv",
            "unattested_caller_argv",
            "unattested_caller_argv_sha256",
            "data_locators_used",
            "runtime_manifest",
            "runtime_manifest_sha256",
            "source_import_closure_sha256",
            "data_content",
            "seed_reservation_binding",
            "external_execution_attestation_present",
        },
        "execution",
    )
    _require(
        _is_int(execution["started_unix"]) and execution["started_unix"] >= 0,
        "started_unix invalid",
    )
    _require(
        execution["execution_origin"]
        in {
            "benchmark_runner",
            "direct_supplied_result",
            "direct_supplied_result_builder",
        },
        "shard execution origin invalid",
    )
    partial_locator = _canonical_absolute_locator(
        execution["partial_locator"],
        "execution.partial_locator",
    )
    _require(
        execution["started_unix"] >= plan["issued_unix"],
        "shard started before plan issuance",
    )
    _require(
        _is_int(execution["finished_unix"])
        and execution["started_unix"]
        <= execution["finished_unix"]
        <= int(time.time()) + 5,
        "finished_unix invalid",
    )
    _require(
        _is_number(execution["duration_seconds"]) and float(execution["duration_seconds"]) >= 0.0,
        "duration_seconds invalid",
    )
    _require(
        execution["duration_semantics"] == _DURATION_SEMANTICS,
        "duration semantics differ",
    )
    unix_elapsed = execution["finished_unix"] - execution["started_unix"]
    _require(
        abs(float(execution["duration_seconds"]) - float(unix_elapsed)) <= 2.0,
        "self-reported duration is inconsistent with worker timestamps",
    )
    plan_locator = cast(str, binding["locator"])
    data_locators = _validate_data_locators(
        execution["data_locators_used"],
        "execution.data_locators_used",
    )
    worker_argv = _expect_list(
        execution["prescribed_worker_argv"],
        "prescribed_worker_argv",
    )
    _require(
        _json_exact_equal(
            worker_argv,
            _canonical_worker_argv(
                cast(str, execution["execution_origin"]),
                plan_locator,
                cast(str, learner),
                cast(int, seed),
                partial_locator,
                data_locators["data_home"],
                data_locators["archive"],
            ),
        ),
        "prescribed worker argv differs from shard identity",
    )
    process_argv = _expect_list(
        execution["unattested_caller_argv"],
        "unattested_caller_argv",
    )
    process_argv = _validated_process_argv(
        cast(list[str], process_argv),
        "self-reported process argv",
    )
    _require(
        execution["unattested_caller_argv_sha256"] == canonical_json_sha256(process_argv),
        "worker unattested caller argv digest mismatch",
    )
    runtime = _validate_runtime(execution["runtime_manifest"], "execution.runtime_manifest")
    _require(
        _json_exact_equal(runtime, body["runtime_manifest"]),
        "worker runtime differs from planned runtime",
    )
    _require(
        execution["runtime_manifest_sha256"] == canonical_json_sha256(runtime),
        "worker runtime digest mismatch",
    )
    _require(
        execution["source_import_closure_sha256"] == body["source_import_closure_sha256"],
        "worker source closure digest mismatch",
    )
    data_content = _expect_dict(execution["data_content"], "execution.data_content")
    _require(
        _json_exact_equal(data_content, body["data_manifest"]["content"]),
        "worker data identity mismatch",
    )
    reservation_binding = execution["seed_reservation_binding"]
    if execution["execution_origin"] == "benchmark_runner":
        _require(
            reservation_binding is not None,
            "benchmark runner shard requires a persistent seed reservation",
        )
        reservation = _validate_reservation_binding(
            reservation_binding,
            plan_path=plan_path,
            plan_raw=plan_raw,
            plan=plan,
            learner_id=cast(str, learner),
            seed_id=cast(int, seed),
            partial_locator=partial_locator,
        )
        _require(
            reservation["reserved_unix"] <= execution["started_unix"],
            "benchmark execution started before its seed reservation",
        )
    else:
        _require(
            reservation_binding is None,
            "supplied-result shard cannot claim a benchmark seed reservation",
        )
    _require(
        execution["external_execution_attestation_present"] is False,
        "v3 shard cannot self-assert external attestation",
    )
    measurements = _expect_dict(partial["measurements"], "measurements")
    _expect_exact_keys(
        measurements,
        {"per_task_accuracy", "per_task_loss", "per_task_plasticity"},
        "measurements",
    )
    _validate_measurement_vector(
        measurements["per_task_accuracy"], "per_task_accuracy", config.n_tasks, upper=1.0
    )
    _validate_measurement_vector(
        measurements["per_task_loss"], "per_task_loss", config.n_tasks, upper=None
    )
    _validate_measurement_vector(
        measurements["per_task_plasticity"],
        "per_task_plasticity",
        config.n_tasks,
        upper=1.0,
    )
    return partial


def _read_validated_partial(
    path: Path,
    plan_path: Path,
    plan_raw: bytes,
    plan: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    raw, partial = _read_strict_json(path)
    _validate_partial_payload(partial, plan_path, plan_raw, plan)
    _require(raw == canonical_json_bytes(partial), f"{path}: partial is not canonically encoded")
    return raw, partial


def validate_partial(
    path: Path,
    plan_path: Path,
    *,
    verify_current_bindings: bool = True,
    data_home: Path | None = None,
    data_archive: Path | None = None,
) -> UPGDIPMNISTV3Validation:
    """Validate one shard structurally and by exact computational replay."""

    try:
        _require(
            verify_current_bindings,
            "public shard validity requires current source/runtime/data bindings",
        )
        plan_raw, plan = _read_validated_plan(
            plan_path,
            verify_current_bindings=verify_current_bindings,
            data_home=data_home,
            data_archive=data_archive,
        )
        partial_raw, partial = _read_validated_partial(path, plan_path, plan_raw, plan)
        run_spec, _, _ = _validate_run_spec(plan["plan"]["run_spec"])
        effective_home, effective_archive = _effective_data_paths(
            plan,
            data_home,
            data_archive,
        )
        _replay_partial_measurements({_identity(partial): partial}, run_spec, effective_home)
        if verify_current_bindings:
            final_plan_raw, final_plan = _read_validated_plan(
                plan_path,
                verify_current_bindings=True,
                data_home=effective_home,
                data_archive=effective_archive,
            )
            _require(final_plan_raw == plan_raw, "plan bytes changed during shard validation")
            _require(
                _json_exact_equal(final_plan, plan),
                "plan changed during shard validation",
            )
            final_partial_raw, final_partial = _read_validated_partial(
                path,
                plan_path,
                plan_raw,
                plan,
            )
            _require(
                final_partial_raw == partial_raw,
                "partial bytes changed during shard validation",
            )
            _require(
                _json_exact_equal(final_partial, partial),
                "partial changed during shard validation",
            )
            _validate_current_bindings_against_plan(
                final_plan,
                data_home=effective_home,
                data_archive=effective_archive,
            )
    except Exception as exc:
        return UPGDIPMNISTV3Validation(False, False, (str(exc),))
    return UPGDIPMNISTV3Validation(True, False, ())


def run_shard(
    plan_path: Path,
    learner_id: str,
    seed_id: int,
    partial_path: Path,
    *,
    data_home: Path | None = None,
    data_archive: Path | None = None,
    progress_every: int | None = 10,
    process_argv: Sequence[str] = (),
) -> Path:
    """Execute exactly one planned learner/seed and publish one valid shard."""

    _require(
        progress_every is None or (_is_int(progress_every) and progress_every > 0),
        "progress_every must be None or a positive integer",
    )
    argv = _validated_process_argv(process_argv, "worker process argv")
    destination = _preflight_new_output(partial_path)
    # The loader and digest verifier must address the same fetch_openml cache.
    preliminary_raw, preliminary_plan = _read_validated_plan(
        plan_path,
        verify_current_bindings=False,
    )
    preliminary_data = cast(dict[str, Any], preliminary_plan["plan"]["data_manifest"])
    planned_home = Path(cast(dict[str, str], preliminary_data["locators"])["data_home"])
    effective_data_home = (
        _lexical_absolute(data_home) if data_home is not None else planned_home
    )
    expected_archive = _lexical_absolute(
        effective_data_home / MNIST_ARCHIVE_RELATIVE_PATH
    )
    effective_archive = (
        _lexical_absolute(data_archive)
        if data_archive is not None
        else expected_archive
    )
    _require(
        effective_archive == expected_archive,
        "--data-archive must identify the fetch_openml archive inside --data-home",
    )
    preliminary_run_spec, config, seeds = _validate_run_spec(
        preliminary_plan["plan"]["run_spec"]
    )
    _require(
        learner_id in preliminary_run_spec["learner_ids"],
        f"learner {learner_id!r} is not planned",
    )
    _require(_is_int(seed_id) and seed_id in seeds, f"seed {seed_id!r} is not planned")
    reservation_binding = _acquire_seed_reservation(
        plan_path=plan_path,
        plan_raw=preliminary_raw,
        plan=preliminary_plan,
        learner_id=learner_id,
        seed_id=seed_id,
        partial_path=destination,
    )
    plan_raw, plan = _read_validated_plan(
        plan_path,
        verify_current_bindings=True,
        data_home=effective_data_home,
        data_archive=effective_archive,
    )
    _require(plan_raw == preliminary_raw, "plan bytes changed after seed reservation")
    _require(
        _json_exact_equal(plan, preliminary_plan),
        "plan changed after seed reservation",
    )
    run_spec, final_config, final_seeds = _validate_run_spec(plan["plan"]["run_spec"])
    _require(final_config == config and final_seeds == seeds, "plan run spec changed")
    # The reservation is deliberately persistent even if this check or any
    # later execution step fails: an attempted seed is consumed, not retried.
    _validate_current_bindings_against_plan(
        plan,
        data_home=effective_data_home,
        data_archive=effective_archive,
    )
    logger.info("loading planned MNIST dataset from data_home=%s", effective_data_home)
    data_x, data_y = _load_pinned_mnist(
        effective_data_home,
        context="worker",
    )
    started = int(time.time())
    result = run_ipmnist(
        data_x,
        data_y,
        cast(Any, learner_id),
        (seed_id,),
        config=config,
        hyperparameters=run_spec["hyperparameters"][learner_id],
        progress_every=progress_every,
    )
    finished = int(time.time())
    return _write_partial_for_result(
        plan_path,
        result,
        destination,
        data_home=effective_data_home,
        data_archive=effective_archive,
        process_argv=argv,
        execution_origin="benchmark_runner",
        reservation_binding=reservation_binding,
        started_unix=started,
        finished_unix=finished,
    )


def _planned_coverage(run_spec: Mapping[str, Any]) -> list[dict[str, object]]:
    return [
        {"learner_id": learner, "seed_id": seed}
        for learner in run_spec["learner_ids"]
        for seed in run_spec["seed_schedule"]["seed_ids"]
    ]


def _identity(payload: Mapping[str, Any]) -> tuple[str, int]:
    return cast(str, payload["learner_id"]), cast(int, payload["seed_id"])


def _results_from_partials(
    partials: Mapping[tuple[str, int], Mapping[str, Any]],
    run_spec: Mapping[str, Any],
) -> dict[str, IPMNISTRunResult]:
    config = _validate_config(run_spec["config"])
    seeds = tuple(cast(list[int], run_spec["seed_schedule"]["seed_ids"]))
    results: dict[str, IPMNISTRunResult] = {}
    for learner in cast(list[str], run_spec["learner_ids"]):
        learner_partials = [partials[(learner, seed)] for seed in seeds]
        accuracy = np.stack(
            [np.asarray(item["measurements"]["per_task_accuracy"]) for item in learner_partials]
        )
        loss = np.stack(
            [np.asarray(item["measurements"]["per_task_loss"]) for item in learner_partials]
        )
        plasticity = np.stack(
            [np.asarray(item["measurements"]["per_task_plasticity"]) for item in learner_partials]
        )
        results[learner] = IPMNISTRunResult(
            learner=learner,
            hyperparameters=dict(run_spec["hyperparameters"][learner]),
            seeds=seeds,
            config=config,
            per_task_accuracy=accuracy,
            per_task_loss=loss,
            per_task_plasticity=plasticity,
            average_online_accuracy=accuracy.mean(axis=1),
            wall_clock_seconds=sum(
                float(item["execution"]["duration_seconds"]) for item in learner_partials
            ),
        )
    return results


def _scientific_summary(result: IPMNISTRunResult) -> dict[str, Any]:
    """Summarize measurements while excluding unverified worker timing."""

    summary = summarize_result(result)
    _require("wall_clock_seconds" in summary, "base summary omitted its timing diagnostic")
    del summary["wall_clock_seconds"]
    return summary


def _v3_comparison(
    results: Mapping[str, IPMNISTRunResult],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    comparison = build_comparison(summaries)
    upgd = np.asarray(results["upgd_w"].average_online_accuracy, dtype=np.float64)
    adamw = np.asarray(results["adamw"].average_online_accuracy, dtype=np.float64)
    _require(upgd.shape == adamw.shape == (20,), "paired comparison requires twenty seed pairs")
    deltas = upgd - adamw
    _require(bool(np.all(np.isfinite(deltas))), "paired comparison contains non-finite values")
    mean = float(np.mean(deltas))
    standard_deviation = float(np.std(deltas, ddof=1))
    standard_error = standard_deviation / math.sqrt(float(deltas.size))
    t_critical = float(cast(float, _PAIRED_COMPARISON_CONTRACT["t_critical"]))
    comparison["paired_seed_comparison"] = {
        "contract": dict(_PAIRED_COMPARISON_CONTRACT),
        "per_seed_deltas": [float(value) for value in deltas],
        "mean_delta": mean,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "confidence_interval": {
            "lower": mean - t_critical * standard_error,
            "upper": mean + t_critical * standard_error,
        },
        "sign_counts": {
            "upgd_w_higher": int(np.count_nonzero(deltas > 0.0)),
            "equal": int(np.count_nonzero(deltas == 0.0)),
            "adamw_higher": int(np.count_nonzero(deltas < 0.0)),
        },
        "interpretation": "preregistered paired descriptive evidence; no post-hoc gate",
    }
    return comparison


def _measurement_payload_from_result(result: IPMNISTRunResult) -> dict[str, Any]:
    return {
        "per_task_accuracy": np.asarray(result.per_task_accuracy)[0].tolist(),
        "per_task_loss": np.asarray(result.per_task_loss)[0].tolist(),
        "per_task_plasticity": np.asarray(result.per_task_plasticity)[0].tolist(),
    }


def _replay_partial_measurements(
    partials: Mapping[tuple[str, int], Mapping[str, Any]],
    run_spec: Mapping[str, Any],
    data_home: Path,
) -> None:
    """Reexecute every supplied planned learner/seed and require exact measurements."""

    _require(bool(partials), "computational replay requires at least one shard")
    planned = {
        (cast(str, item["learner_id"]), cast(int, item["seed_id"]))
        for item in _planned_coverage(run_spec)
    }
    supplied = set(partials)
    _require(
        supplied <= planned,
        f"computational replay contains unplanned identities: {sorted(supplied - planned)}",
    )
    data_x, data_y = _load_pinned_mnist(
        data_home,
        context="computational replay",
    )
    config = _validate_config(run_spec["config"])
    for item in _planned_coverage(run_spec):
        learner = cast(str, item["learner_id"])
        seed = cast(int, item["seed_id"])
        if (learner, seed) not in supplied:
            continue
        try:
            replay = run_ipmnist(
                data_x,
                data_y,
                cast(Any, learner),
                (seed,),
                config=config,
                hyperparameters=run_spec["hyperparameters"][learner],
                progress_every=None,
            )
        except Exception as exc:
            raise UPGDIPMNISTV3Error(
                f"computational replay failed for {(learner, seed)}: {exc}"
            ) from exc
        _validate_result_against_plan(replay, run_spec)
        _require(
            _json_exact_equal(
                partials[(learner, seed)]["measurements"],
                _measurement_payload_from_result(replay),
            ),
            f"recorded measurements differ from exact replay for {(learner, seed)}",
        )


def _coverage_payload(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    coverage = _planned_coverage(run_spec)
    return {
        "planned": coverage,
        "observed": coverage,
        "planned_count": len(coverage),
        "observed_count": len(coverage),
        "complete": True,
    }


def _canonical_merge_argv(
    plan_locator: str,
    manifest: Sequence[Mapping[str, Any]],
    artifact_locator: str,
    data_home_locator: str,
    data_archive_locator: str,
) -> list[str]:
    return [
        "merge",
        "--plan",
        plan_locator,
        "--partials",
        *(cast(str, entry["locator"]) for entry in manifest),
        "--output",
        artifact_locator,
        "--data-home",
        data_home_locator,
        "--data-archive",
        data_archive_locator,
    ]


def _merge_partials(
    plan_path: Path,
    partial_paths: Sequence[Path],
    output_path: Path,
    *,
    created_unix: int | None = None,
    process_argv: Sequence[str] = (),
    verify_current_bindings: bool = True,
    data_home: Path | None = None,
    data_archive: Path | None = None,
    invocation_origin: str,
) -> Path:
    """Merge the exact planned Cartesian shard set into one immutable artifact.

    Publication is forbidden when current source/runtime/data verification is
    disabled. The keyword remains only so older callers fail explicitly before
    any replay rather than silently weakening the artifact.
    """

    _require(
        verify_current_bindings,
        "artifact publication requires current source/runtime/data bindings",
    )
    argv = _validated_process_argv(process_argv, "merge process argv")
    _require(
        invocation_origin in {"direct_python_api", "cli"},
        "unsupported merge invocation origin",
    )
    destination = _preflight_new_output(output_path)
    plan_raw, plan = _read_validated_plan(
        plan_path,
        verify_current_bindings=verify_current_bindings,
        data_home=data_home,
        data_archive=data_archive,
    )
    effective_home, effective_archive = _effective_data_paths(
        plan,
        data_home,
        data_archive,
    )
    run_spec, _, _ = _validate_run_spec(plan["plan"]["run_spec"])
    _require(bool(partial_paths), "merge requires shard paths")
    by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    raw_by_identity: dict[tuple[str, int], bytes] = {}
    manifest: list[dict[str, Any]] = []
    for path_value in partial_paths:
        path = Path(path_value)
        raw, partial = _read_validated_partial(path, plan_path, plan_raw, plan)
        identity = _identity(partial)
        _require(identity not in by_identity, f"duplicate shard identity: {identity}")
        by_identity[identity] = partial
        raw_by_identity[identity] = raw
        manifest.append(
            {
                "learner_id": identity[0],
                "seed_id": identity[1],
                "locator": _lexical_absolute(path).as_posix(),
                "byte_size": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )
    planned = {(item["learner_id"], item["seed_id"]) for item in _planned_coverage(run_spec)}
    observed = set(by_identity)
    _require(
        observed == planned,
        f"shard coverage differs: missing={sorted(planned - observed)}, "
        f"extra={sorted(observed - planned)}",
    )
    _replay_partial_measurements(by_identity, run_spec, effective_home)
    results = _results_from_partials(by_identity, run_spec)
    summaries = {
        learner: _scientific_summary(results[learner])
        for learner in cast(list[str], run_spec["learner_ids"])
    }
    manifest.sort(key=lambda item: (item["learner_id"], item["seed_id"]))
    created = int(time.time()) if created_unix is None else created_unix
    _require(_is_int(created) and created >= 0, "created_unix must be nonnegative")
    _require(created <= int(time.time()) + 5, "created_unix cannot be in the future")
    latest_finished = max(
        cast(int, partial["execution"]["finished_unix"]) for partial in by_identity.values()
    )
    _require(created >= latest_finished, "artifact cannot predate a merged shard")
    canonical_merge_argv = _canonical_merge_argv(
        _lexical_absolute(plan_path).as_posix(),
        manifest,
        destination.as_posix(),
        effective_home.as_posix(),
        effective_archive.as_posix(),
    )
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "benchmark": BENCHMARK,
        "evidence_policy": dict(EVIDENCE_POLICY),
        "created_unix": created,
        "run_plan": plan,
        "plan_binding": _plan_binding(plan_path, plan_raw, plan),
        "coverage": _coverage_payload(run_spec),
        "partial_manifest": manifest,
        "computational_replay": {
            "kind": "exact_full_reexecution",
            "completed": True,
            "shard_count": len(by_identity),
            "source_import_closure_sha256": plan["plan"][
                "source_import_closure_sha256"
            ],
            "runtime_manifest_sha256": plan["plan"]["runtime_manifest_sha256"],
            "data_manifest_sha256": plan["plan"]["data_manifest_sha256"],
            "data_locators_used": {
                "data_home": effective_home.as_posix(),
                "archive": effective_archive.as_posix(),
            },
        },
        "learners": summaries,
        "comparison": _v3_comparison(results, summaries),
        "merge_execution": {
            "invocation_origin": invocation_origin,
            "prescribed_merge_argv": canonical_merge_argv,
            "unattested_caller_argv": argv,
            "unattested_caller_argv_sha256": canonical_json_sha256(argv),
            "runtime_manifest_sha256": plan["plan"]["runtime_manifest_sha256"],
            "source_import_closure_sha256": plan["plan"]["source_import_closure_sha256"],
            "data_locators_used": {
                "data_home": effective_home.as_posix(),
                "archive": effective_archive.as_posix(),
            },
            "external_execution_attestation_present": False,
        },
    }
    if verify_current_bindings:
        final_plan_raw, final_plan = _read_validated_plan(
            plan_path,
            verify_current_bindings=True,
            data_home=effective_home,
            data_archive=effective_archive,
        )
        _require(final_plan_raw == plan_raw, "plan bytes changed during merge")
        _require(_json_exact_equal(final_plan, plan), "plan changed during merge")
        for entry in manifest:
            identity = (cast(str, entry["learner_id"]), cast(int, entry["seed_id"]))
            final_raw, final_partial = _read_validated_partial(
                Path(cast(str, entry["locator"])),
                plan_path,
                plan_raw,
                plan,
            )
            _require(
                final_raw == raw_by_identity[identity],
                f"{identity}: shard bytes changed during merge",
            )
            _require(
                _json_exact_equal(final_partial, by_identity[identity]),
                f"{identity}: shard payload changed during merge",
            )
        _validate_current_bindings_against_plan(
            final_plan,
            data_home=effective_home,
            data_archive=effective_archive,
        )
    return atomic_write_new_json(destination, artifact)


def merge_partials(
    plan_path: Path,
    partial_paths: Sequence[Path],
    output_path: Path,
    *,
    created_unix: int | None = None,
    process_argv: Sequence[str] = (),
    verify_current_bindings: bool = True,
    data_home: Path | None = None,
    data_archive: Path | None = None,
) -> Path:
    """Merge shards through the direct Python API entry point."""

    return _merge_partials(
        plan_path,
        partial_paths,
        output_path,
        created_unix=created_unix,
        process_argv=process_argv,
        verify_current_bindings=verify_current_bindings,
        data_home=data_home,
        data_archive=data_archive,
        invocation_origin="direct_python_api",
    )


def _validate_coverage(value: object, run_spec: Mapping[str, Any]) -> dict[str, Any]:
    coverage = _expect_dict(value, "coverage")
    _expect_exact_keys(
        coverage,
        {"planned", "observed", "planned_count", "observed_count", "complete"},
        "coverage",
    )
    expected = _coverage_payload(run_spec)
    _require(
        _json_exact_equal(coverage, expected),
        "coverage differs from the exact planned Cartesian product",
    )
    return coverage


def _validate_partial_manifest(
    value: object,
    run_spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entries = _expect_list(value, "partial_manifest")
    expected_identities = [
        (cast(str, item["learner_id"]), cast(int, item["seed_id"]))
        for item in _planned_coverage(run_spec)
    ]
    identities: list[tuple[str, int]] = []
    validated: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(entries):
        where = f"partial_manifest[{index}]"
        entry = _expect_dict(raw_entry, where)
        _expect_exact_keys(
            entry,
            {"learner_id", "seed_id", "locator", "byte_size", "sha256"},
            where,
        )
        _require(isinstance(entry["learner_id"], str), f"{where}.learner_id invalid")
        _require(_is_int(entry["seed_id"]), f"{where}.seed_id invalid")
        _canonical_absolute_locator(entry["locator"], f"{where}.locator")
        _require(
            _is_int(entry["byte_size"]) and entry["byte_size"] >= 0,
            f"{where}.byte_size invalid",
        )
        _require(_is_sha256(entry["sha256"]), f"{where}.sha256 invalid")
        identities.append((entry["learner_id"], entry["seed_id"]))
        validated.append(entry)
    expected_sorted = sorted(expected_identities)
    _require(identities == expected_sorted, "partial manifest identities or order differ")
    _require(len(set(identities)) == len(identities), "partial manifest identities duplicate")
    return validated


def _validate_artifact_payload(
    artifact: object,
    artifact_path: Path,
    partial_paths: Sequence[Path] | None,
    plan_path: Path,
    *,
    verify_current_bindings: bool,
    data_home: Path | None,
    data_archive: Path | None,
) -> dict[str, Any]:
    _require(
        verify_current_bindings,
        "artifact validation requires current source/runtime/data bindings",
    )
    payload = _expect_dict(artifact, "artifact")
    _reject_legacy_marker(payload)
    _expect_exact_keys(
        payload,
        {
            "schema",
            "benchmark",
            "evidence_policy",
            "created_unix",
            "run_plan",
            "plan_binding",
            "coverage",
            "partial_manifest",
            "computational_replay",
            "learners",
            "comparison",
            "merge_execution",
        },
        "artifact",
    )
    _require(payload["schema"] == ARTIFACT_SCHEMA, "wrong v3 artifact schema")
    _require(payload["benchmark"] == BENCHMARK, "wrong artifact benchmark")
    _require(
        _json_exact_equal(payload["evidence_policy"], EVIDENCE_POLICY),
        "artifact evidence policy differs",
    )
    _require(
        _is_int(payload["created_unix"])
        and 0 <= payload["created_unix"] <= int(time.time()) + 5,
        "created_unix invalid",
    )
    plan = _validate_plan_payload(
        payload["run_plan"],
        verify_current_bindings=verify_current_bindings,
        data_home=data_home,
        data_archive=data_archive,
    )
    plan_raw = canonical_json_bytes(plan)
    plan_binding = _validate_plan_binding(payload["plan_binding"], plan_path, plan_raw, plan)
    external_plan_raw, external_plan = _read_validated_plan(
        plan_path,
        verify_current_bindings=True,
        data_home=data_home,
        data_archive=data_archive,
    )
    _require(
        external_plan_raw == plan_raw and _json_exact_equal(external_plan, plan),
        "supplied external plan differs from the embedded plan",
    )
    external_plan_locator = _lexical_absolute(plan_path).as_posix()
    _require(
        plan_binding["locator"] == external_plan_locator,
        "artifact plan binding locator differs from the immutable external plan",
    )
    run_spec, _, _ = _validate_run_spec(plan["plan"]["run_spec"])
    _validate_coverage(payload["coverage"], run_spec)
    manifest = _validate_partial_manifest(payload["partial_manifest"], run_spec)
    shard_paths = (
        [Path(item["locator"]) for item in manifest]
        if partial_paths is None
        else [Path(path) for path in partial_paths]
    )
    _require(len(shard_paths) == len(manifest), "supplied shard count differs from manifest")
    manifest_by_identity = {(entry["learner_id"], entry["seed_id"]): entry for entry in manifest}
    partials: dict[tuple[str, int], dict[str, Any]] = {}
    partial_raw_by_identity: dict[tuple[str, int], bytes] = {}
    partial_path_by_identity: dict[tuple[str, int], Path] = {}
    for path in shard_paths:
        raw, partial = _read_validated_partial(path, plan_path, plan_raw, plan)
        identity = _identity(partial)
        _require(identity not in partials, f"duplicate supplied shard identity: {identity}")
        _require(identity in manifest_by_identity, f"supplied shard identity is extra: {identity}")
        entry = manifest_by_identity[identity]
        _require(entry["byte_size"] == len(raw), f"{identity}: shard byte_size mismatch")
        _require(entry["sha256"] == sha256_bytes(raw), f"{identity}: shard byte hash mismatch")
        partials[identity] = partial
        partial_raw_by_identity[identity] = raw
        partial_path_by_identity[identity] = path
    _require(set(partials) == set(manifest_by_identity), "supplied shards do not cover manifest")
    latest_finished = max(
        cast(int, partial["execution"]["finished_unix"]) for partial in partials.values()
    )
    _require(
        payload["created_unix"] >= latest_finished,
        "artifact predates a bound shard",
    )
    effective_home, effective_archive = _effective_data_paths(
        plan,
        data_home,
        data_archive,
    )
    effective_data_locators = {
        "data_home": effective_home.as_posix(),
        "archive": effective_archive.as_posix(),
    }
    _replay_partial_measurements(partials, run_spec, effective_home)
    replay = _expect_dict(payload["computational_replay"], "computational_replay")
    _expect_exact_keys(
        replay,
        {
            "kind",
            "completed",
            "shard_count",
            "source_import_closure_sha256",
            "runtime_manifest_sha256",
            "data_manifest_sha256",
            "data_locators_used",
        },
        "computational_replay",
    )
    replay_data_locators = _validate_data_locators(
        replay["data_locators_used"],
        "computational_replay.data_locators_used",
    )
    expected_replay = {
        "kind": "exact_full_reexecution",
        "completed": True,
        "shard_count": len(partials),
        "source_import_closure_sha256": plan["plan"][
            "source_import_closure_sha256"
        ],
        "runtime_manifest_sha256": plan["plan"]["runtime_manifest_sha256"],
        "data_manifest_sha256": plan["plan"]["data_manifest_sha256"],
        "data_locators_used": effective_data_locators,
    }
    _require(
        _json_exact_equal(replay, expected_replay),
        "computational replay receipt differs from the exact contract",
    )
    results = _results_from_partials(partials, run_spec)
    expected_summaries = {
        learner: _scientific_summary(results[learner])
        for learner in cast(list[str], run_spec["learner_ids"])
    }
    _require(
        _json_exact_equal(payload["learners"], expected_summaries),
        "learner summaries do not recompute",
    )
    _require(
        _json_exact_equal(
            payload["comparison"],
            _v3_comparison(results, expected_summaries),
        ),
        "comparison does not recompute",
    )
    execution = _expect_dict(payload["merge_execution"], "merge_execution")
    _expect_exact_keys(
        execution,
        {
            "invocation_origin",
            "prescribed_merge_argv",
            "unattested_caller_argv",
            "unattested_caller_argv_sha256",
            "runtime_manifest_sha256",
            "source_import_closure_sha256",
            "data_locators_used",
            "external_execution_attestation_present",
        },
        "merge_execution",
    )
    argv = _expect_list(
        execution["unattested_caller_argv"],
        "merge_execution.unattested_caller_argv",
    )
    argv = _validated_process_argv(
        cast(list[str], argv),
        "merge self-reported process argv",
    )
    _require(
        execution["unattested_caller_argv_sha256"] == canonical_json_sha256(argv),
        "merge unattested caller argv digest mismatch",
    )
    _require(
        execution["invocation_origin"] in {"direct_python_api", "cli"},
        "merge invocation origin invalid",
    )
    canonical_merge_argv = _expect_list(
        execution["prescribed_merge_argv"],
        "merge_execution.prescribed_merge_argv",
    )
    merge_data_locators = _validate_data_locators(
        execution["data_locators_used"],
        "merge_execution.data_locators_used",
    )
    expected_merge_argv = _canonical_merge_argv(
        external_plan_locator,
        manifest,
        _lexical_absolute(artifact_path).as_posix(),
        effective_home.as_posix(),
        effective_archive.as_posix(),
    )
    _require(
        _json_exact_equal(canonical_merge_argv, expected_merge_argv),
        "prescribed merge argv differs from external plan, manifest, output, or data bindings",
    )
    _require(
        execution["runtime_manifest_sha256"] == plan["plan"]["runtime_manifest_sha256"],
        "merge runtime digest differs",
    )
    _require(
        execution["source_import_closure_sha256"] == plan["plan"]["source_import_closure_sha256"],
        "merge source digest differs",
    )
    _require(
        _json_exact_equal(replay_data_locators, effective_data_locators)
        and _json_exact_equal(merge_data_locators, effective_data_locators),
        "merge/replay data locators differ from the exact effective data cache",
    )
    _require(
        execution["external_execution_attestation_present"] is False,
        "artifact cannot self-assert external execution attestation",
    )
    _validate_plan_payload(
        payload["run_plan"],
        verify_current_bindings=True,
        data_home=effective_home,
        data_archive=effective_archive,
    )
    final_plan_raw, final_plan = _read_validated_plan(
        plan_path,
        verify_current_bindings=True,
        data_home=effective_home,
        data_archive=effective_archive,
    )
    _require(
        final_plan_raw == external_plan_raw,
        "external plan bytes changed during artifact validation",
    )
    _require(
        _json_exact_equal(final_plan, plan),
        "external plan changed during artifact validation",
    )
    for identity, path in partial_path_by_identity.items():
        final_raw, final_partial = _read_validated_partial(
            path,
            plan_path,
            plan_raw,
            plan,
        )
        _require(
            final_raw == partial_raw_by_identity[identity],
            f"{identity}: shard bytes changed during artifact validation",
        )
        _require(
            _json_exact_equal(final_partial, partials[identity]),
            f"{identity}: shard changed during artifact validation",
        )
    _validate_current_bindings_against_plan(
        plan,
        data_home=effective_home,
        data_archive=effective_archive,
    )
    return payload


def validate_artifact(
    path: Path,
    *,
    partial_paths: Sequence[Path] | None = None,
    plan_path: Path | None = None,
    verify_current_bindings: bool = True,
    data_home: Path | None = None,
    data_archive: Path | None = None,
) -> UPGDIPMNISTV3Validation:
    """Validate an artifact and recompute it from exact content-bound shards."""

    try:
        _require(
            verify_current_bindings,
            "public artifact validity requires current source/runtime/data bindings",
        )
        _require(
            plan_path is not None,
            "public artifact validity requires an immutable external plan",
        )
        assert plan_path is not None
        raw, artifact = _read_strict_json(path)
        _validate_artifact_payload(
            artifact,
            path,
            partial_paths,
            plan_path,
            verify_current_bindings=verify_current_bindings,
            data_home=data_home,
            data_archive=data_archive,
        )
        _require(raw == canonical_json_bytes(artifact), "artifact is not canonically encoded")
        final_raw, final_artifact = _read_strict_json(path)
        _require(final_raw == raw, "artifact bytes changed during validation")
        _require(
            _json_exact_equal(final_artifact, artifact),
            "artifact payload changed during validation",
        )
        final_plan = _expect_dict(final_artifact["run_plan"], "artifact.run_plan")
        _validate_current_bindings_against_plan(
            final_plan,
            data_home=data_home,
            data_archive=data_archive,
        )
        terminal_plan_raw, terminal_plan = _read_validated_plan(
            plan_path,
            verify_current_bindings=True,
            data_home=data_home,
            data_archive=data_archive,
        )
        _require(
            terminal_plan_raw == canonical_json_bytes(final_plan)
            and _json_exact_equal(terminal_plan, final_plan),
            "external plan differs after the final artifact reread",
        )
        _validate_plan_binding(
            final_artifact["plan_binding"],
            plan_path,
            terminal_plan_raw,
            terminal_plan,
        )
        _validate_current_bindings_against_plan(
            terminal_plan,
            data_home=data_home,
            data_archive=data_archive,
        )
    except Exception as exc:
        return UPGDIPMNISTV3Validation(False, False, (str(exc),))
    return UPGDIPMNISTV3Validation(True, False, ())


def _parse_seed_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed list must contain comma-separated integers") from exc
    try:
        return _validate_seed_ids(values)
    except UPGDIPMNISTV3Error as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UPGD IPMNIST v3: immutable plan -> one-seed shards -> exact merge"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="issue a new immutable v3 plan")
    plan_parser.add_argument("--plan-out", type=Path, required=True)
    plan_parser.add_argument("--seed-list", type=_parse_seed_list, required=True)
    plan_parser.add_argument("--data-home", type=Path, default=None)
    plan_parser.add_argument("--data-archive", type=Path, default=None)
    plan_parser.add_argument("--n-tasks", type=int, default=200)
    plan_parser.add_argument("--task-length", type=int, default=5000)
    plan_parser.add_argument("--input-dim", type=int, default=784)
    plan_parser.add_argument("--hidden1", type=int, default=300)
    plan_parser.add_argument("--hidden2", type=int, default=150)
    plan_parser.add_argument("--n-classes", type=int, default=10)

    shard_parser = subparsers.add_parser(
        "shard", help="execute exactly one planned learner/seed shard"
    )
    shard_parser.add_argument("--plan", type=Path, required=True)
    shard_parser.add_argument("--learner-id", choices=LEARNER_IDS, required=True)
    shard_parser.add_argument("--seed-id", type=int, required=True)
    shard_parser.add_argument("--partial-out", type=Path, required=True)
    shard_parser.add_argument("--data-home", type=Path, default=None)
    shard_parser.add_argument("--data-archive", type=Path, default=None)
    shard_parser.add_argument("--progress-every", type=int, default=10)

    merge_parser = subparsers.add_parser("merge", help="merge the exact planned shard set")
    merge_parser.add_argument("--plan", type=Path, required=True)
    merge_parser.add_argument("--partials", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--data-home", type=Path, default=None)
    merge_parser.add_argument("--data-archive", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    """Run only the active v3 lifecycle; no direct aggregate mode exists."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    semantic_argv = list(argv) if argv is not None else list(sys.argv[1:])
    try:
        if args.command == "plan":
            data_home = args.data_home if args.data_home is not None else default_openml_data_home()
            archive = (
                args.data_archive
                if args.data_archive is not None
                else data_home / MNIST_ARCHIVE_RELATIVE_PATH
            )
            return _write_plan(
                args.plan_out,
                IPMNISTConfig(
                    n_tasks=args.n_tasks,
                    task_length=args.task_length,
                    input_dim=args.input_dim,
                    hidden1=args.hidden1,
                    hidden2=args.hidden2,
                    n_classes=args.n_classes,
                ),
                args.seed_list,
                data_home,
                archive,
                issuer_argv=semantic_argv,
                invocation_origin="cli",
            )
        if args.command == "shard":
            return run_shard(
                args.plan,
                args.learner_id,
                args.seed_id,
                args.partial_out,
                data_home=args.data_home,
                data_archive=args.data_archive,
                progress_every=args.progress_every,
                process_argv=semantic_argv,
            )
        if args.command == "merge":
            return _merge_partials(
                args.plan,
                args.partials,
                args.output,
                process_argv=semantic_argv,
                data_home=args.data_home,
                data_archive=args.data_archive,
                invocation_origin="cli",
            )
    except Exception as exc:
        parser.error(str(exc))
    parser.error("an explicit v3 lifecycle command is required")


if __name__ == "__main__":
    main()
