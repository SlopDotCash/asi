"""Strict ingestion for the three-seed, published-scale OPMNIST development run.

The upstream runner predates Alberta's fail-closed evidence envelopes.  Its
result-finalization manifest records useful command, runtime, and two-file
source metadata, but there was no externally issued pre-run manifest binding
the full import closure or dataset bytes.  This module therefore preserves and
validates the run as a development-only published-scale confirmation.  It can
never promote a scientific, SOTA, Step-2, or Alberta Plan completion claim.

The ingester reads inputs through descriptor-anchored no-symlink paths, pins the
known OpenML gzip identity, validates all three seed results without importing
the upstream runner, independently recomputes the six metric comparisons, and
durably publishes a read-only, digested v1 receipt with atomic no-replace
semantics. Validation snapshots the exact closed tree through anchored
descriptors, rejects permissive JSON extensions, and rebuilds the receipt from
the retained bytes.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import logging
import math
import os
import secrets
import shlex
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

logger = logging.getLogger(__name__)

OPMNIST_DEVELOPMENT_RECEIPT_SCHEMA = (
    "alberta.opmnist-published-scale-development-receipt.v1"
)
UPSTREAM_PLAN_SCHEMA = "alberta.step2.opmnist_full_run_plan.v1"
UPSTREAM_RESULT_MANIFEST_SCHEMA = "alberta.step2.upgd_memory_opmnist.manifest.v1"
UPSTREAM_SOLUTION_STATUS_SCHEMA = "alberta.step2.opmnist_solution_status.v1"
UPSTREAM_SOLUTION_GATE_SCHEMA = "alberta.step2.opmnist_solution_gate.audit.v1"
UPSTREAM_PROVENANCE_STATUS_SCHEMA = (
    "alberta.step2.opmnist_artifact_provenance_status.v1"
)

EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_STEPS_PER_SEED = 48_000_000
EXPECTED_TASKS = 800
EXPECTED_TASK_BLOCK_SIZE = 60_000
EXPECTED_TEST_VIEWS = 800
EXPECTED_OPENML_DATASET_SHA256 = (
    "fe4410d8dbb50f6db6482b187557c5cb8bccfbcec74eeb6abc47c858f4ffab78"
)
EXPECTED_OPENML_DATASET_BYTE_SIZE = 15_469_256
_OPENML_CACHE_RELATIVE_PATH = Path(
    "openml/openml.org/data/v1/download/52667/mnist_784.arff.gz"
)
_BUNDLE_FILE_MODE = 0o440
_BUNDLE_DIRECTORY_MODE = 0o550
_STAGING_DIRECTORY_MODE = 0o700
_READ_CHUNK_SIZE = 1024 * 1024
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_OPEN_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_OPEN_REGULAR_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
EXPECTED_METHODS = (
    "step2_hybrid_memory_trace",
    "step2_hybrid_memory_trace_adaptive_sharp",
    "mlp_h64",
    "mlp_h128",
    "mlp_h64_sharp",
    "mlp_h128_sharp",
)
EXPECTED_MLP_METHODS = (
    "mlp_h64",
    "mlp_h128",
    "mlp_h64_sharp",
    "mlp_h128_sharp",
)
EXPECTED_CANDIDATE_METHODS = (
    "step2_hybrid_memory_trace",
    "step2_hybrid_memory_trace_adaptive_sharp",
)
CORE_METRICS = (
    "online_mean_mse",
    "online_mean_accuracy",
    "final_window_mse",
    "final_window_accuracy",
    "test_mse",
    "test_accuracy",
)
HIGHER_IS_BETTER = {
    "online_mean_mse": False,
    "online_mean_accuracy": True,
    "final_window_mse": False,
    "final_window_accuracy": True,
    "test_mse": False,
    "test_accuracy": True,
}

_RESULT_TOP_FIELDS = {
    "aggregate",
    "candidate_methods",
    "config",
    "datasets",
    "evidence_level",
    "manifest",
    "mlp_methods",
    "primary_method",
    "records",
    "solution_status",
    "wall_clock_s",
}
_ARG_CONFIG_FIELDS = {
    "allow_openml_download",
    "allow_torchvision_download",
    "chunk_size",
    "evaluate_all_permutation_views",
    "final_window",
    "force_restart",
    "include_adaptive_primary_sharpened",
    "include_brier_single_upgd",
    "include_centroid_candidates",
    "include_delight_candidates",
    "include_dreaming_candidates",
    "include_identity_permutation",
    "include_primary_sharpened",
    "include_prototype_memory",
    "include_rls_calibrated",
    "include_sharpened_mlp",
    "include_single_upgd",
    "include_smoothed_single_upgd",
    "include_temperature_single_upgd",
    "max_test_examples",
    "max_test_permutation_views",
    "max_train_examples",
    "mnist_published_scale",
    "mnist_source",
    "mnist_split",
    "n_permutations",
    "n_seeds",
    "note_path",
    "only_methods",
    "openml_data_home",
    "openml_n_retries",
    "openml_retry_delay",
    "opmnist_fraction",
    "output_dir",
    "result_prefix",
    "resume",
    "resume_path",
    "sample_with_replacement",
    "seed",
    "status_path",
    "steps",
    "stop_after_chunks",
    "task_block_size",
    "task_sampling",
    "torchvision_data_home",
    "train_fraction",
}
_RESULT_CONFIG_FIELDS = _ARG_CONFIG_FIELDS | {"created_at", "methods", "runner"}
_DATASET_FIELDS = {
    "all_experts_update_every_step",
    "benchmark",
    "candidate_methods",
    "checkpoint_loaded",
    "completed_full_task_blocks",
    "dataset",
    "description",
    "evaluate_all_permutation_views",
    "feature_dim",
    "full_mnist_task_blocks",
    "heldout_deployment_objective",
    "include_identity_permutation",
    "is_full_mnist_split",
    "is_true_mnist",
    "limitations",
    "matches_dohare_opmnist_core_protocol",
    "matches_dohare_opmnist_published_task_count",
    "max_test_examples",
    "max_test_permutation_views",
    "max_train_examples",
    "methods",
    "mlp_methods",
    "n_classes",
    "n_permutations",
    "n_test",
    "n_total",
    "n_train",
    "observed_task_blocks",
    "openml_data_home",
    "opmnist_completed_full_60000_task_blocks",
    "opmnist_elapsed_s",
    "opmnist_eta_to_800_tasks_s",
    "opmnist_overall_steps_per_second",
    "partial_task_steps",
    "permutations_are_random_pixel_orders",
    "prediction_before_update_every_step",
    "protocol",
    "published_protocol_delta",
    "random_pixel_permutation_seed",
    "resumable_runner",
    "resume_checkpoint_path",
    "sample_with_replacement",
    "sequence_seed",
    "sequential_single_pass_task_epochs",
    "single_pass_examples_within_task",
    "source_kind",
    "split",
    "split_seed",
    "steps",
    "stream_chunk_size",
    "streaming_runner",
    "task_block_size",
    "task_id_provided_to_learner",
    "task_ids_observed",
    "task_sampling",
    "test_permutation_views",
    "test_task_ids_evaluated",
    "test_views_cover_all_permutations",
    "test_views_cover_observed_permutations",
    "train_fraction",
}
_PLAN_FIELDS = {
    "audit_command",
    "audit_command_shell",
    "claim",
    "expected_result",
    "expected_solution_status",
    "merge_command",
    "merge_command_shell",
    "methods",
    "n_seeds",
    "promotion_rule",
    "protocol",
    "runner_command",
    "runner_command_shell",
    "schema",
    "seed_start",
    "split_seed_runner_command_shells",
    "split_seed_runner_commands",
}
_PLAN_PROTOCOL_FIELDS = {
    "mnist_source",
    "mnist_split",
    "n_permutations",
    "prediction_before_update",
    "task_block_size",
    "task_id_provided_to_learner",
    "test_views",
    "updates_per_seed",
}
_MANIFEST_FIELDS = {
    "argv",
    "config",
    "created_at_utc",
    "environment",
    "git",
    "methods",
    "schema",
    "source_sha256",
}
_GIT_FIELDS = {"branch", "commit", "describe", "dirty", "status_porcelain"}
_ENVIRONMENT_FIELDS = {
    "jax",
    "jax_default_backend",
    "jax_devices",
    "jaxlib",
    "numpy",
    "platform",
    "python",
    "python_executable",
}
_SOLUTION_STATUS_FIELDS = {
    "artifact_provenance",
    "candidate_metric_wins",
    "candidates_winning_all_metrics",
    "claim_scope",
    "completed_record_seed_count",
    "configured_seed_count",
    "min_seeds",
    "multi_seed_full_scale",
    "protocol_complete",
    "schema",
    "solved_opmnist_step2",
}
_PROVENANCE_STATUS_FIELDS = {
    "direct_runner_manifest_complete",
    "manifest_schema",
    "merged_split_manifest_complete",
    "mixed_method_manifest_complete",
    "provenance_complete",
    "schema",
}
_STATUS_TOP_FIELDS = {
    "checkpoint_path",
    "completed_steps",
    "dohare_target_steps",
    "latest_progress",
    "protocol",
    "requested_steps",
    "schema",
    "seed",
    "status",
    "updated_at_utc",
}
_PROGRESS_STATUS_FIELDS = {
    "completed_full_task_blocks",
    "completed_steps",
    "elapsed_s",
    "eta_human",
    "eta_seconds",
    "overall_steps_per_second",
    "progress_fraction",
    "recent_steps_per_second",
    "remaining_full_task_blocks",
    "remaining_steps",
    "target_full_task_blocks",
    "target_steps",
}
_LATEST_PROGRESS_FIELDS = {
    "chunk_elapsed_s",
    "chunk_steps",
    "chunks_run_this_invocation",
    "completed_full_task_blocks",
    "completed_steps",
    "dohare_target_steps",
    "elapsed_s",
    "eta_to_dohare_800_s",
    "requested_steps",
    "seed",
    "steps_per_second",
    "stop_after_chunks",
    "timestamp_utc",
}
_MERGED_RESULT_TOP_FIELDS = {
    "aggregate",
    "candidate_methods",
    "config",
    "datasets",
    "evidence_level",
    "manifest",
    "mlp_methods",
    "primary_method",
    "records",
    "solution_status",
    "split_results",
}
_MERGE_MANIFEST_FIELDS = {
    "created_at_utc",
    "merge_script",
    "merge_script_sha256",
    "methods",
    "runner_path",
    "runner_sha256",
    "schema",
    "seeds",
    "split_results",
}
_MERGE_SPLIT_RESULT_FIELDS = {"manifest", "path", "seeds", "sha256"}
_GATE_FIELDS = {"artifact_path", "schema", "status"}

_BUNDLE_FILES = {
    "receipt.v1.json",
    "inputs/plan.json",
    "inputs/RUNBOOK.md",
    "inputs/merged-result.json",
    "inputs/solution-gate.json",
    "inputs/source/step2_upgd_memory_opmnist.py",
    "inputs/source/step2_published_stressors.py",
    "inputs/source/step2_opmnist_merge_seed_results.py",
    "inputs/source/step2_opmnist_solution_gate.py",
    "inputs/data/mnist_784.arff.gz",
    *{f"inputs/seed-results/seed-{seed}.json" for seed in EXPECTED_SEEDS},
    *{f"inputs/seed-status/seed-{seed}.json" for seed in EXPECTED_SEEDS},
}
_BUNDLE_DIRECTORIES = {
    "inputs",
    "inputs/data",
    "inputs/seed-results",
    "inputs/seed-status",
    "inputs/source",
}

_EVIDENCE_POLICY: dict[str, object] = {
    "evidence_class": "development_published_scale_confirmation",
    "development_only": True,
    "scientific_promotion_allowed": False,
    "externally_issued_prerun_full_closure_manifest": False,
    "consumed_seeds_may_not_be_reused_for_promotion": True,
}
_LIMITATIONS = [
    (
        "No externally issued pre-run manifest bound the commands, complete import "
        "closure, runtime, and dataset bytes before seeds 0, 1, and 2 were consumed."
    ),
    (
        "Runner manifests were created during result finalization and hash only the "
        "runner and published-stressor files; their source and data checks are post-hoc."
    ),
    (
        "Three-seed comparisons are descriptive and do not establish a SOTA, "
        "inferential, Step-2, or Alberta Plan completion claim."
    ),
    (
        "Protocol completion and an all-six-metric mean win are separate facts; "
        "neither changes this receipt's permanently nonpromoting evidence class."
    ),
]


class OPMNISTDevelopmentIngestError(ValueError):
    """Raised when an input cannot enter the development bundle."""


def _absolute_lexical_path(path: Path) -> Path:
    """Return an absolute normalized path without resolving symbolic links."""
    return Path(os.path.abspath(os.fspath(path)))


def _path_components(path: Path) -> tuple[Path, tuple[str, ...]]:
    absolute = _absolute_lexical_path(path)
    if not absolute.is_absolute():  # pragma: no cover - abspath is absolute
        raise OPMNISTDevelopmentIngestError(f"path is not absolute: {path}")
    return absolute, tuple(absolute.parts[1:])


def _path_open_error(*, path: Path, label: str, exc: OSError) -> OPMNISTDevelopmentIngestError:
    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
        detail = "contains a symbolic link or non-directory component"
    else:
        detail = exc.strerror or type(exc).__name__
    return OPMNISTDevelopmentIngestError(f"{label} path {path} {detail}")


def _open_directory_path_no_symlinks(path: Path, *, label: str) -> int:
    """Open an existing directory by walking every component with ``O_NOFOLLOW``."""
    absolute, components = _path_components(path)
    current_fd = os.open("/", _OPEN_DIRECTORY_FLAGS)
    try:
        for component in components:
            try:
                next_fd = os.open(component, _OPEN_DIRECTORY_FLAGS, dir_fd=current_fd)
            except OSError as exc:
                raise _path_open_error(path=absolute, label=label, exc=exc) from exc
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _open_or_create_directory_path_no_symlinks(path: Path, *, label: str) -> int:
    """Create missing directory components while refusing every symbolic link."""
    absolute, components = _path_components(path)
    current_fd = os.open("/", _OPEN_DIRECTORY_FLAGS)
    try:
        for component in components:
            try:
                next_fd = os.open(component, _OPEN_DIRECTORY_FLAGS, dir_fd=current_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o750, dir_fd=current_fd)
                    os.fsync(current_fd)
                except FileExistsError:
                    pass
                try:
                    next_fd = os.open(component, _OPEN_DIRECTORY_FLAGS, dir_fd=current_fd)
                except OSError as exc:
                    raise _path_open_error(path=absolute, label=label, exc=exc) from exc
            except OSError as exc:
                raise _path_open_error(path=absolute, label=label, exc=exc) from exc
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _stable_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_open_regular_file(fd: int, *, label: str) -> bytes:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise OPMNISTDevelopmentIngestError(f"{label} must be a regular file")
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, _READ_CHUNK_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(fd)
    raw = b"".join(chunks)
    if _stable_stat_identity(before) != _stable_stat_identity(after) or len(raw) != after.st_size:
        raise OPMNISTDevelopmentIngestError(f"{label} changed while it was being read")
    return raw


def _read_regular_file_no_symlinks(path: Path, *, label: str) -> bytes:
    """Read one stable regular file through descriptor-anchored no-symlink traversal."""
    absolute, components = _path_components(path)
    if not components:
        raise OPMNISTDevelopmentIngestError(f"{label} path must name a regular file")
    parent_fd = os.open("/", _OPEN_DIRECTORY_FLAGS)
    try:
        for component in components[:-1]:
            try:
                next_fd = os.open(component, _OPEN_DIRECTORY_FLAGS, dir_fd=parent_fd)
            except OSError as exc:
                raise _path_open_error(path=absolute, label=label, exc=exc) from exc
            os.close(parent_fd)
            parent_fd = next_fd
        try:
            file_fd = os.open(components[-1], _OPEN_REGULAR_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise _path_open_error(path=absolute, label=label, exc=exc) from exc
        try:
            return _read_open_regular_file(file_fd, label=label)
        finally:
            os.close(file_fd)
    finally:
        os.close(parent_fd)


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    path = Path(relative_path)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise OPMNISTDevelopmentIngestError(
            f"bundle path must be a normalized relative path: {relative_path!r}"
        )
    return tuple(path.parts)


def _open_relative_directory(root_fd: int, parts: Sequence[str]) -> int:
    current_fd = os.dup(root_fd)
    try:
        for component in parts:
            next_fd = os.open(component, _OPEN_DIRECTORY_FLAGS, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _write_file_at(root_fd: int, relative_path: str, raw: bytes) -> None:
    """Create, fsync, and seal one staging file beneath an anchored directory."""
    parts = _relative_parts(relative_path)
    parent_fd = _open_relative_directory(root_fd, parts[:-1])
    try:
        file_fd = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            view = memoryview(raw)
            written = 0
            while written < len(view):
                written += os.write(file_fd, view[written:])
            os.fsync(file_fd)
            os.fchmod(file_fd, _BUNDLE_FILE_MODE)
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _create_bundle_directories(root_fd: int) -> None:
    for relative in sorted(_BUNDLE_DIRECTORIES, key=lambda item: (item.count("/"), item)):
        parts = _relative_parts(relative)
        parent_fd = _open_relative_directory(root_fd, parts[:-1])
        try:
            os.mkdir(parts[-1], _STAGING_DIRECTORY_MODE, dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)


def _seal_bundle_directories(root_fd: int) -> None:
    for relative in sorted(
        _BUNDLE_DIRECTORIES,
        key=lambda item: (item.count("/"), item),
        reverse=True,
    ):
        directory_fd = _open_relative_directory(root_fd, _relative_parts(relative))
        try:
            os.fchmod(directory_fd, _BUNDLE_DIRECTORY_MODE)
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    os.fchmod(root_fd, _BUNDLE_DIRECTORY_MODE)
    os.fsync(root_fd)


def _rename_noreplace_at(parent_fd: int, source_name: str, target_name: str) -> None:
    """Atomically publish a directory without replacing any existing node."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:  # pragma: no cover - current Linux/glibc exposes it
        raise OSError(errno.ENOTSUP, "renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(target_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error_number, "target already exists", target_name)
        raise OSError(error_number, os.strerror(error_number), target_name)


def _remove_tree_at(parent_fd: int, name: str) -> None:
    """Remove a private staging tree without following inserted links."""
    root_fd = os.open(name, _OPEN_DIRECTORY_FLAGS, dir_fd=parent_fd)
    try:
        os.fchmod(root_fd, _STAGING_DIRECTORY_MODE)

        def remove_children(directory_fd: int) -> None:
            for child in os.listdir(directory_fd):
                child_stat = os.stat(child, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISDIR(child_stat.st_mode):
                    child_fd = os.open(child, _OPEN_DIRECTORY_FLAGS, dir_fd=directory_fd)
                    try:
                        os.fchmod(child_fd, _STAGING_DIRECTORY_MODE)
                        remove_children(child_fd)
                    finally:
                        os.close(child_fd)
                    os.rmdir(child, dir_fd=directory_fd)
                else:
                    os.unlink(child, dir_fd=directory_fd)

        remove_children(root_fd)
    finally:
        os.close(root_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _directory_stable(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _snapshot_bundle_no_symlinks(bundle_dir: Path) -> tuple[dict[str, bytes], list[str]]:
    """Read a closed bundle tree once through anchored descriptors."""
    root_fd = _open_directory_path_no_symlinks(bundle_dir, label="bundle root")
    files: dict[str, bytes] = {}
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    errors: list[str] = []

    def walk(directory_fd: int, prefix: str, *, is_root: bool = False) -> None:
        before = os.fstat(directory_fd)
        expected_mode = _BUNDLE_DIRECTORY_MODE
        if stat.S_IMODE(before.st_mode) != expected_mode:
            label = "bundle root" if is_root else f"bundle directory {prefix}"
            errors.append(f"{label} mode must be {oct(expected_mode)}")
        names_before = sorted(os.listdir(directory_fd))
        for name in names_before:
            relative = f"{prefix}/{name}" if prefix else name
            node_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(node_stat.st_mode):
                errors.append(f"bundle paths must not be symbolic links: {relative}")
                continue
            if stat.S_ISDIR(node_stat.st_mode):
                actual_directories.add(relative)
                try:
                    child_fd = os.open(name, _OPEN_DIRECTORY_FLAGS, dir_fd=directory_fd)
                except OSError as exc:
                    errors.append(f"could not open bundle directory {relative}: {exc}")
                    continue
                try:
                    opened = os.fstat(child_fd)
                    if (opened.st_dev, opened.st_ino) != (node_stat.st_dev, node_stat.st_ino):
                        errors.append(f"bundle directory changed during validation: {relative}")
                    else:
                        walk(child_fd, relative)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(node_stat.st_mode):
                errors.append(f"bundle filesystem node must be regular: {relative}")
                continue
            actual_files.add(relative)
            if stat.S_IMODE(node_stat.st_mode) != _BUNDLE_FILE_MODE:
                errors.append(f"bundle file {relative} mode must be {oct(_BUNDLE_FILE_MODE)}")
            try:
                file_fd = os.open(name, _OPEN_REGULAR_FLAGS, dir_fd=directory_fd)
                try:
                    opened = os.fstat(file_fd)
                    if _stable_stat_identity(opened) != _stable_stat_identity(node_stat):
                        errors.append(f"bundle file changed during validation: {relative}")
                    else:
                        files[relative] = _read_open_regular_file(
                            file_fd,
                            label=f"bundle file {relative}",
                        )
                finally:
                    os.close(file_fd)
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
        names_after = sorted(os.listdir(directory_fd))
        after = os.fstat(directory_fd)
        if names_before != names_after or not _directory_stable(before, after):
            errors.append(f"bundle directory changed during validation: {prefix or '.'}")

    try:
        walk(root_fd, "", is_root=True)
    finally:
        os.close(root_fd)
    if actual_files != _BUNDLE_FILES:
        errors.append(
            "bundle file set differs from v1 schema; "
            f"missing={sorted(_BUNDLE_FILES - actual_files)}, "
            f"extra={sorted(actual_files - _BUNDLE_FILES)}"
        )
    if actual_directories != _BUNDLE_DIRECTORIES:
        errors.append(
            "bundle directory set differs from v1 schema; "
            f"missing={sorted(_BUNDLE_DIRECTORIES - actual_directories)}, "
            f"extra={sorted(actual_directories - _BUNDLE_DIRECTORIES)}"
        )
    return files, errors


@dataclass(frozen=True)
class OPMNISTDevelopmentBundleValidation:
    """Fail-closed bundle validation with claims kept explicitly separate."""

    valid: bool
    protocol_complete: bool
    all_metric_mean_win: bool
    solved_opmnist_step2: bool
    errors: tuple[str, ...]
    receipt_sha256: str | None = None
    development_only: bool = True
    scientific_promotion_allowed: bool = False


@dataclass(frozen=True)
class _InputBytes:
    """One input decoded from the same bytes that are later copied and hashed."""

    path: Path
    raw: bytes
    payload: dict[str, object]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _decode_strict_json_object(raw: bytes, *, label: str) -> dict[str, object]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{label}: non-finite JSON constant is forbidden: {value}")

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: invalid UTF-8 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label}: payload must contain one JSON object")
    _reject_nonfinite_tree(parsed, label=label)
    return cast(dict[str, object], parsed)


def _reject_nonfinite_tree(value: object, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label}: all JSON numbers must be finite")
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_nonfinite_tree(child, label=label)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite_tree(child, label=label)


def _read_json(path: Path, *, label: str) -> _InputBytes:
    raw = _read_regular_file_no_symlinks(path, label=label)
    return _InputBytes(path=path, raw=raw, payload=_decode_strict_json_object(raw, label=label))


def _mapping(value: object, *, label: str, errors: list[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return {}
    return cast(Mapping[str, object], value)


def _expect_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
    errors: list[str],
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        errors.append(f"{label} fields differ from v1 schema; missing={missing}, extra={extra}")


def _finite_number(
    value: object,
    *,
    label: str,
    errors: list[str],
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label} must be a finite JSON number")
        return None
    number = float(value)
    if not math.isfinite(number):
        errors.append(f"{label} must be a finite JSON number")
        return None
    if minimum is not None and number < minimum:
        errors.append(f"{label} must be >= {minimum}")
    if maximum is not None and number > maximum:
        errors.append(f"{label} must be <= {maximum}")
    return number


def _required_finite_float(value: object, *, label: str) -> float:
    """Return a finite non-boolean number or raise for a recomputation primitive."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite JSON number")
    return number


def _iso_timestamp(value: object, *, label: str, errors: list[str]) -> None:
    if not isinstance(value, str):
        errors.append(f"{label} must be an ISO-8601 timestamp")
        return
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        errors.append(f"{label} must be an ISO-8601 timestamp")
        return
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        errors.append(f"{label} must include an explicit +00:00 UTC offset")


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _string_list(value: object, *, label: str, errors: list[str]) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        errors.append(f"{label} must be an array of strings")
        return []
    return cast(list[str], value)


def _exact_integer_list(
    value: object,
    expected: Sequence[int],
    *,
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        errors.append(f"{label} must be an array of JSON integers")
        return
    if value != list(expected):
        errors.append(f"{label} does not match the exact required integer sequence")


def _compare_recomputed(
    actual: object,
    expected: object,
    *,
    label: str,
    errors: list[str],
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            errors.append(f"{label} must be an object")
            return
        actual_mapping = cast(Mapping[str, object], actual)
        if set(actual_mapping) != set(expected):
            errors.append(f"{label} fields do not match the independently recomputed value")
            return
        for key in expected:
            _compare_recomputed(
                actual_mapping[key],
                expected[key],
                label=f"{label}.{key}",
                errors=errors,
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            errors.append(f"{label} does not match the independently recomputed value")
            return
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _compare_recomputed(
                actual_item,
                expected_item,
                label=f"{label}[{index}]",
                errors=errors,
            )
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected or type(actual) is not type(expected):
            errors.append(f"{label} does not match the independently recomputed value")
        return
    if isinstance(expected, (int, float)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            errors.append(f"{label} must be numeric")
            return
        if not math.isfinite(float(actual)) or not math.isclose(
            float(actual),
            float(expected),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            errors.append(f"{label} does not match the independently recomputed value")
        return
    if actual != expected:
        errors.append(f"{label} does not match the independently recomputed value")


def _sample_stderr(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def _paired_diff(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    higher_is_better: bool,
) -> dict[str, object]:
    diffs = [
        candidate_value - baseline_value
        if higher_is_better
        else baseline_value - candidate_value
        for candidate_value, baseline_value in zip(candidate, baseline, strict=True)
    ]
    return {
        "diff_mean_positive_favors_candidate": sum(diffs) / len(diffs),
        "diff_stderr": _sample_stderr(diffs),
        "wins_for_candidate": sum(value > 0.0 for value in diffs),
        "wins_for_baseline": sum(value < 0.0 for value in diffs),
        "ties": sum(value == 0.0 for value in diffs),
        "diffs": diffs,
    }


def recompute_opmnist_metrics(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Recompute all upstream summaries from per-seed primitive metrics."""
    if not records:
        raise ValueError("cannot aggregate an empty OPMNIST record sequence")
    method_rows: dict[str, list[Mapping[str, object]]] = {}
    for method in EXPECTED_METHODS:
        rows: list[Mapping[str, object]] = []
        for record in records:
            methods = cast(Mapping[str, object], record["methods"])
            rows.append(cast(Mapping[str, object], methods[method]))
        method_rows[method] = rows

    aggregate: dict[str, object] = {}
    for method in EXPECTED_METHODS:
        method_summary: dict[str, object] = {}
        for metric in CORE_METRICS:
            values = [
                _required_finite_float(
                    row[metric],
                    label=f"records.methods.{method}.{metric}",
                )
                for row in method_rows[method]
            ]
            method_summary[metric] = {
                "mean": sum(values) / len(values),
                "stderr": _sample_stderr(values),
                "per_seed": values,
            }
        aggregate[method] = method_summary

    comparisons: dict[str, object] = {}
    for metric in CORE_METRICS:
        higher = HIGHER_IS_BETTER[metric]
        best_mlp = (max if higher else min)(
            EXPECTED_MLP_METHODS,
            key=lambda method: _required_finite_float(
                cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], aggregate[method])[metric],
                )["mean"],
                label=f"aggregate.{method}.{metric}.mean",
            ),
        )
        baseline_values = [
            _required_finite_float(
                row[metric],
                label=f"records.methods.{best_mlp}.{metric}",
            )
            for row in method_rows[best_mlp]
        ]
        candidate_diffs = {
            candidate: _paired_diff(
                [
                    _required_finite_float(
                        row[metric],
                        label=f"records.methods.{candidate}.{metric}",
                    )
                    for row in method_rows[candidate]
                ],
                baseline_values,
                higher_is_better=higher,
            )
            for candidate in EXPECTED_CANDIDATE_METHODS
        }
        comparisons[metric] = {
            "best_mlp": best_mlp,
            "candidate_vs_best_mlp": candidate_diffs,
            "primary_vs_best_mlp": candidate_diffs[EXPECTED_CANDIDATE_METHODS[0]],
        }
    aggregate["comparisons"] = comparisons
    aggregate["method_order"] = list(EXPECTED_METHODS)
    aggregate["mlp_methods"] = list(EXPECTED_MLP_METHODS)
    aggregate["candidate_methods"] = list(EXPECTED_CANDIDATE_METHODS)
    return aggregate


def _metric_wins(aggregate: Mapping[str, object]) -> dict[str, dict[str, bool]]:
    comparisons = cast(Mapping[str, object], aggregate["comparisons"])
    result: dict[str, dict[str, bool]] = {}
    for candidate in EXPECTED_CANDIDATE_METHODS:
        result[candidate] = {}
        for metric in CORE_METRICS:
            comparison = cast(Mapping[str, object], comparisons[metric])
            rows = cast(Mapping[str, object], comparison["candidate_vs_best_mlp"])
            row = cast(Mapping[str, object], rows[candidate])
            result[candidate][metric] = _required_finite_float(
                row["diff_mean_positive_favors_candidate"],
                label=f"aggregate.comparisons.{metric}.{candidate}.mean_diff",
            ) > 0.0
    return result


def _expected_upstream_seed_status(aggregate: Mapping[str, object]) -> dict[str, object]:
    wins = _metric_wins(aggregate)
    winners = [candidate for candidate, metric_wins in wins.items() if all(metric_wins.values())]
    return {
        "schema": UPSTREAM_SOLUTION_STATUS_SCHEMA,
        "min_seeds": 3,
        "protocol_complete": True,
        "configured_seed_count": 1,
        "completed_record_seed_count": 1,
        "multi_seed_full_scale": False,
        "artifact_provenance": {
            "schema": UPSTREAM_PROVENANCE_STATUS_SCHEMA,
            "manifest_schema": UPSTREAM_RESULT_MANIFEST_SCHEMA,
            "direct_runner_manifest_complete": True,
            "merged_split_manifest_complete": False,
            "mixed_method_manifest_complete": False,
            "provenance_complete": True,
        },
        "candidate_metric_wins": wins,
        "candidates_winning_all_metrics": winners,
        "solved_opmnist_step2": False,
        "claim_scope": "limited_opmnist_evidence_not_step2_solution",
    }


def _command_flag(command: Sequence[str], flag: str) -> str | None:
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) != 1:
        return None
    position = positions[0]
    if position + 1 >= len(command):
        return None
    return command[position + 1]


def _validate_seed_command(command: Sequence[str], seed: int, *, errors: list[str]) -> None:
    label = f"plan.split_seed_runner_commands[{seed}]"
    if len(command) < 3 or not command[1].endswith("step2_upgd_memory_opmnist.py"):
        errors.append(f"{label} must invoke the OPMNIST runner")
        return
    expected_flags = {
        "--n-seeds": "1",
        "--seed": str(seed),
        "--final-window": "5000",
        "--chunk-size": "60000",
        "--max-test-permutation-views": "800",
        "--only-methods": ",".join(EXPECTED_METHODS),
    }
    for flag, expected in expected_flags.items():
        if _command_flag(command, flag) != expected:
            errors.append(f"{label} must bind {flag}={expected}")
    for flag in (
        "--mnist-published-scale",
        "--allow-openml-download",
        "--include-sharpened-mlp",
        "--include-adaptive-primary-sharpened",
        "--evaluate-all-permutation-views",
    ):
        if command.count(flag) != 1:
            errors.append(f"{label} must contain {flag} exactly once")


def _validate_plan(plan: Mapping[str, object], *, errors: list[str]) -> list[list[str]]:
    _expect_fields(plan, _PLAN_FIELDS, label="plan", errors=errors)
    if plan.get("schema") != UPSTREAM_PLAN_SCHEMA:
        errors.append("plan.schema is unsupported")
    if (
        type(plan.get("n_seeds")) is not int
        or plan.get("n_seeds") != 3
        or type(plan.get("seed_start")) is not int
        or plan.get("seed_start") != 0
    ):
        errors.append("plan must specify exactly seeds 0, 1, and 2")
    if plan.get("methods") != list(EXPECTED_METHODS):
        errors.append("plan.methods does not match the frozen six-method order")
    protocol = _mapping(plan.get("protocol"), label="plan.protocol", errors=errors)
    _expect_fields(protocol, _PLAN_PROTOCOL_FIELDS, label="plan.protocol", errors=errors)
    expected_protocol: dict[str, object] = {
        "mnist_source": "openml",
        "mnist_split": "canonical_60000_10000",
        "n_permutations": EXPECTED_TASKS,
        "prediction_before_update": True,
        "task_block_size": EXPECTED_TASK_BLOCK_SIZE,
        "task_id_provided_to_learner": False,
        "test_views": "all_800_permutation_views",
        "updates_per_seed": EXPECTED_STEPS_PER_SEED,
    }
    _compare_recomputed(protocol, expected_protocol, label="plan.protocol", errors=errors)

    commands_value = plan.get("split_seed_runner_commands")
    commands: list[list[str]] = []
    if not isinstance(commands_value, list) or len(commands_value) != 3:
        errors.append("plan.split_seed_runner_commands must contain exactly three commands")
    else:
        for seed, command_value in zip(EXPECTED_SEEDS, commands_value, strict=True):
            command = _string_list(
                command_value,
                label=f"plan.split_seed_runner_commands[{seed}]",
                errors=errors,
            )
            _validate_seed_command(command, seed, errors=errors)
            commands.append(command)

    shells = plan.get("split_seed_runner_command_shells")
    if not isinstance(shells, list) or len(shells) != len(commands):
        errors.append("plan split command shell forms must match the command arrays")
    else:
        expected_shells = [shlex.join(command) for command in commands]
        if shells != expected_shells:
            errors.append("plan split command shell forms do not reconstruct from arrays")

    plan_commands: dict[str, list[str]] = {}
    for array_field, shell_field in (
        ("runner_command", "runner_command_shell"),
        ("merge_command", "merge_command_shell"),
        ("audit_command", "audit_command_shell"),
    ):
        command = _string_list(plan.get(array_field), label=f"plan.{array_field}", errors=errors)
        plan_commands[array_field] = command
        if plan.get(shell_field) != shlex.join(command):
            errors.append(f"plan.{shell_field} does not reconstruct from {array_field}")

    runner_command = plan_commands["runner_command"]
    if len(runner_command) < 3 or not runner_command[1].endswith(
        "step2_upgd_memory_opmnist.py"
    ):
        errors.append("plan.runner_command must invoke the OPMNIST runner")
    for flag, expected in {
        "--n-seeds": "3",
        "--seed": "0",
        "--final-window": "5000",
        "--chunk-size": "60000",
        "--max-test-permutation-views": "800",
        "--only-methods": ",".join(EXPECTED_METHODS),
    }.items():
        if _command_flag(runner_command, flag) != expected:
            errors.append(f"plan.runner_command must bind {flag}={expected}")

    merge_command = plan_commands["merge_command"]
    if len(merge_command) < 3 or not merge_command[1].endswith(
        "step2_opmnist_merge_seed_results.py"
    ):
        errors.append("plan.merge_command must invoke the audited merge script")
    if _command_flag(merge_command, "--output") != plan.get("expected_result"):
        errors.append("plan.merge_command output must equal plan.expected_result")
    if _command_flag(merge_command, "--write-summary") is None:
        errors.append("plan.merge_command must bind --write-summary")

    audit_command = plan_commands["audit_command"]
    if len(audit_command) < 3 or not audit_command[1].endswith(
        "step2_opmnist_solution_gate.py"
    ):
        errors.append("plan.audit_command must invoke the audited solution gate")
    if len(audit_command) < 3 or audit_command[2] != plan.get("expected_result"):
        errors.append("plan.audit_command input must equal plan.expected_result")
    if _command_flag(audit_command, "--min-seeds") != "3":
        errors.append("plan.audit_command must bind --min-seeds=3")
    if _command_flag(audit_command, "--write-status") != plan.get(
        "expected_solution_status"
    ):
        errors.append(
            "plan.audit_command --write-status must equal plan.expected_solution_status"
        )
    if plan.get("claim") != "Step 2 OPMNIST solution candidate run":
        errors.append("plan.claim is not the audited v1 claim text")
    if plan.get("promotion_rule") != (
        "The audit command must exit 0 without --allow-unsolved and report "
        "status.solved_opmnist_step2=true."
    ):
        errors.append("plan.promotion_rule is not the audited v1 rule")
    return commands


def _validate_runbook(raw: bytes, *, errors: list[str]) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        errors.append("RUNBOOK.md must be UTF-8")
        return
    required_phrases = (
        "800-task 3-seed",
        "48,000,000",
        "Seeds 0, 1, 2",
        "all 800",
        "solved_opmnist_step2",
        "NOT as a solved-Step-2 claim",
    )
    for phrase in required_phrases:
        if phrase not in text:
            errors.append(f"RUNBOOK.md is missing audited phrase {phrase!r}")


def _validate_config(
    config: Mapping[str, object],
    *,
    seed: int,
    errors: list[str],
) -> None:
    _expect_fields(config, _RESULT_CONFIG_FIELDS, label=f"seed {seed} config", errors=errors)
    expected_values: dict[str, object] = {
        "allow_openml_download": True,
        "allow_torchvision_download": False,
        "chunk_size": EXPECTED_TASK_BLOCK_SIZE,
        "evaluate_all_permutation_views": True,
        "final_window": 5_000,
        "force_restart": False,
        "include_adaptive_primary_sharpened": True,
        "include_brier_single_upgd": False,
        "include_centroid_candidates": False,
        "include_delight_candidates": False,
        "include_dreaming_candidates": False,
        "include_identity_permutation": False,
        "include_primary_sharpened": False,
        "include_prototype_memory": False,
        "include_rls_calibrated": False,
        "include_sharpened_mlp": True,
        "include_single_upgd": False,
        "include_smoothed_single_upgd": False,
        "include_temperature_single_upgd": False,
        "max_test_examples": None,
        "max_test_permutation_views": EXPECTED_TEST_VIEWS,
        "max_train_examples": None,
        "mnist_published_scale": True,
        "mnist_source": "openml",
        "mnist_split": "canonical",
        "n_permutations": EXPECTED_TASKS,
        "n_seeds": 1,
        "only_methods": ",".join(EXPECTED_METHODS),
        "openml_data_home": None,
        "openml_n_retries": 2,
        "openml_retry_delay": 1.0,
        "opmnist_fraction": None,
        "resume": True,
        "resume_path": None,
        "sample_with_replacement": False,
        "seed": seed,
        "steps": EXPECTED_STEPS_PER_SEED,
        "stop_after_chunks": None,
        "task_block_size": EXPECTED_TASK_BLOCK_SIZE,
        "task_sampling": "sequential_epoch",
        "torchvision_data_home": None,
        "train_fraction": 0.7,
        "runner": "step2_upgd_memory_opmnist",
        "methods": list(EXPECTED_METHODS),
    }
    for field, expected in expected_values.items():
        if config.get(field) != expected or type(config.get(field)) is not type(expected):
            errors.append(f"seed {seed} config.{field} must equal {expected!r}")
    for field in ("note_path", "output_dir", "result_prefix", "status_path"):
        value = config.get(field)
        if not isinstance(value, str) or not value:
            errors.append(f"seed {seed} config.{field} must be a non-empty string")
    _iso_timestamp(config.get("created_at"), label=f"seed {seed} config.created_at", errors=errors)


def _validate_dataset(
    dataset: Mapping[str, object],
    *,
    seed: int,
    errors: list[str],
) -> None:
    label = f"seed {seed} dataset"
    _expect_fields(dataset, _DATASET_FIELDS, label=label, errors=errors)
    expected_values: dict[str, object] = {
        "all_experts_update_every_step": True,
        "benchmark": "permuted_mnist_like",
        "candidate_methods": list(EXPECTED_CANDIDATE_METHODS),
        "completed_full_task_blocks": EXPECTED_TASKS,
        "dataset": "sklearn.datasets.fetch_openml('mnist_784', version=1)",
        "evaluate_all_permutation_views": True,
        "feature_dim": 784,
        "full_mnist_task_blocks": True,
        "include_identity_permutation": False,
        "is_full_mnist_split": True,
        "is_true_mnist": True,
        "matches_dohare_opmnist_core_protocol": True,
        "matches_dohare_opmnist_published_task_count": True,
        "max_test_examples": None,
        "max_test_permutation_views": EXPECTED_TEST_VIEWS,
        "max_train_examples": None,
        "methods": list(EXPECTED_METHODS),
        "mlp_methods": list(EXPECTED_MLP_METHODS),
        "n_classes": 10,
        "n_permutations": EXPECTED_TASKS,
        "n_test": 10_000,
        "n_total": 70_000,
        "n_train": EXPECTED_TASK_BLOCK_SIZE,
        "observed_task_blocks": EXPECTED_TASKS,
        "opmnist_completed_full_60000_task_blocks": EXPECTED_TASKS,
        "opmnist_eta_to_800_tasks_s": 0.0,
        "partial_task_steps": 0,
        "permutations_are_random_pixel_orders": True,
        "prediction_before_update_every_step": True,
        "protocol": "chunked_online_permuted_pixels",
        "random_pixel_permutation_seed": seed + 10_000,
        "resumable_runner": True,
        "sample_with_replacement": False,
        "sequence_seed": seed + 10_000,
        "sequential_single_pass_task_epochs": True,
        "single_pass_examples_within_task": True,
        "source_kind": "openml_mnist_784",
        "split": "openml_canonical_60000_10000",
        "split_seed": seed,
        "steps": EXPECTED_STEPS_PER_SEED,
        "stream_chunk_size": EXPECTED_TASK_BLOCK_SIZE,
        "streaming_runner": True,
        "task_block_size": EXPECTED_TASK_BLOCK_SIZE,
        "task_id_provided_to_learner": False,
        "task_ids_observed": list(range(EXPECTED_TASKS)),
        "task_sampling": "sequential_epoch",
        "test_permutation_views": EXPECTED_TEST_VIEWS,
        "test_task_ids_evaluated": list(range(EXPECTED_TEST_VIEWS)),
        "test_views_cover_all_permutations": True,
        "test_views_cover_observed_permutations": True,
        "train_fraction": 0.7,
    }
    for field, expected in expected_values.items():
        if dataset.get(field) != expected or type(dataset.get(field)) is not type(expected):
            errors.append(f"{label}.{field} must equal the exact published-scale value")
    _exact_integer_list(
        dataset.get("task_ids_observed"),
        range(EXPECTED_TASKS),
        label=f"{label}.task_ids_observed",
        errors=errors,
    )
    _exact_integer_list(
        dataset.get("test_task_ids_evaluated"),
        range(EXPECTED_TEST_VIEWS),
        label=f"{label}.test_task_ids_evaluated",
        errors=errors,
    )
    for field in (
        "description",
        "heldout_deployment_objective",
        "limitations",
        "openml_data_home",
        "published_protocol_delta",
        "resume_checkpoint_path",
    ):
        value = dataset.get(field)
        if not isinstance(value, str) or not value:
            errors.append(f"{label}.{field} must be a non-empty string")
    if type(dataset.get("checkpoint_loaded")) is not bool:
        errors.append(f"{label}.checkpoint_loaded must be boolean")
    _finite_number(
        dataset.get("opmnist_elapsed_s"),
        label=f"{label}.opmnist_elapsed_s",
        errors=errors,
        minimum=0.0,
    )
    _finite_number(
        dataset.get("opmnist_overall_steps_per_second"),
        label=f"{label}.opmnist_overall_steps_per_second",
        errors=errors,
        minimum=0.0,
    )


def _validate_final_status(
    status_payload: Mapping[str, object],
    dataset: Mapping[str, object],
    *,
    seed: int,
    errors: list[str],
) -> None:
    """Validate the dynamic completion sidecar against the final result record."""
    label = f"seed {seed} final status"
    _expect_fields(status_payload, _STATUS_TOP_FIELDS, label=label, errors=errors)
    if status_payload.get("schema") != "alberta.upgd_memory_opmnist.status.v1":
        errors.append(f"{label}.schema is unsupported")
    exact_top: dict[str, object] = {
        "seed": seed,
        "requested_steps": EXPECTED_STEPS_PER_SEED,
        "dohare_target_steps": EXPECTED_STEPS_PER_SEED,
        "completed_steps": EXPECTED_STEPS_PER_SEED,
    }
    for field, expected in exact_top.items():
        if status_payload.get(field) != expected or type(status_payload.get(field)) is not type(
            expected
        ):
            errors.append(f"{label}.{field} must equal {expected!r}")
    checkpoint_path = status_payload.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        errors.append(f"{label}.checkpoint_path must be a non-empty string")
    if checkpoint_path != dataset.get("resume_checkpoint_path"):
        errors.append(f"{label}.checkpoint_path does not match the final record")
    _iso_timestamp(
        status_payload.get("updated_at_utc"),
        label=f"{label}.updated_at_utc",
        errors=errors,
    )

    protocol = _mapping(status_payload.get("protocol"), label=f"{label}.protocol", errors=errors)
    status_protocol_fields = _DATASET_FIELDS - {
        "opmnist_elapsed_s",
        "opmnist_eta_to_800_tasks_s",
        "opmnist_overall_steps_per_second",
    }
    _expect_fields(protocol, status_protocol_fields, label=f"{label}.protocol", errors=errors)
    for field in status_protocol_fields - {"checkpoint_loaded"}:
        if protocol.get(field) != dataset.get(field):
            errors.append(f"{label}.protocol.{field} does not match the final record")
    if type(protocol.get("checkpoint_loaded")) is not bool:
        errors.append(f"{label}.protocol.checkpoint_loaded must be boolean")

    progress_status = _mapping(status_payload.get("status"), label=f"{label}.status", errors=errors)
    _expect_fields(
        progress_status,
        _PROGRESS_STATUS_FIELDS,
        label=f"{label}.status",
        errors=errors,
    )
    expected_status: dict[str, object] = {
        "completed_full_task_blocks": EXPECTED_TASKS,
        "completed_steps": EXPECTED_STEPS_PER_SEED,
        "eta_seconds": 0.0,
        "progress_fraction": 1.0,
        "remaining_full_task_blocks": 0,
        "remaining_steps": 0,
        "target_full_task_blocks": EXPECTED_TASKS,
        "target_steps": EXPECTED_STEPS_PER_SEED,
    }
    for field, expected in expected_status.items():
        actual = progress_status.get(field)
        if actual != expected or type(actual) is not type(expected):
            errors.append(f"{label}.status.{field} must equal {expected!r}")
    if progress_status.get("eta_human") != "0s":
        errors.append(f"{label}.status.eta_human must equal '0s'")
    elapsed = _finite_number(
        progress_status.get("elapsed_s"),
        label=f"{label}.status.elapsed_s",
        errors=errors,
        minimum=0.0,
    )
    overall = _finite_number(
        progress_status.get("overall_steps_per_second"),
        label=f"{label}.status.overall_steps_per_second",
        errors=errors,
        minimum=0.0,
    )
    recent = _finite_number(
        progress_status.get("recent_steps_per_second"),
        label=f"{label}.status.recent_steps_per_second",
        errors=errors,
        minimum=0.0,
    )
    if elapsed is not None and elapsed <= 0.0:
        errors.append(f"{label}.status.elapsed_s must be positive")
    if overall is not None and overall <= 0.0:
        errors.append(f"{label}.status.overall_steps_per_second must be positive")
    if recent is not None and recent <= 0.0:
        errors.append(f"{label}.status.recent_steps_per_second must be positive")
    if elapsed is not None and elapsed > 0.0 and overall is not None and not math.isclose(
        overall,
        EXPECTED_STEPS_PER_SEED / elapsed,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        errors.append(f"{label}.status.overall_steps_per_second does not reconstruct")
    result_elapsed = _required_finite_float(
        dataset.get("opmnist_elapsed_s"),
        label=f"{label}.result.opmnist_elapsed_s",
    )
    result_overall = _required_finite_float(
        dataset.get("opmnist_overall_steps_per_second"),
        label=f"{label}.result.opmnist_overall_steps_per_second",
    )
    if elapsed is not None and not math.isclose(result_elapsed, elapsed, rel_tol=1e-12):
        errors.append(f"{label} elapsed time does not match the final result")
    if overall is not None and not math.isclose(result_overall, overall, rel_tol=1e-12):
        errors.append(f"{label} overall rate does not match the final result")

    latest = _mapping(
        status_payload.get("latest_progress"),
        label=f"{label}.latest_progress",
        errors=errors,
    )
    _expect_fields(
        latest,
        _LATEST_PROGRESS_FIELDS,
        label=f"{label}.latest_progress",
        errors=errors,
    )
    expected_latest: dict[str, object] = {
        "seed": seed,
        "completed_steps": EXPECTED_STEPS_PER_SEED,
        "requested_steps": EXPECTED_STEPS_PER_SEED,
        "dohare_target_steps": EXPECTED_STEPS_PER_SEED,
        "stop_after_chunks": None,
        "chunk_steps": EXPECTED_TASK_BLOCK_SIZE,
        "completed_full_task_blocks": EXPECTED_TASKS,
        "eta_to_dohare_800_s": 0.0,
    }
    for field, expected in expected_latest.items():
        if latest.get(field) != expected or type(latest.get(field)) is not type(expected):
            errors.append(f"{label}.latest_progress.{field} must equal {expected!r}")
    chunks = latest.get("chunks_run_this_invocation")
    if type(chunks) is not int or not 1 <= chunks <= EXPECTED_TASKS:
        errors.append(f"{label}.latest_progress.chunks_run_this_invocation is invalid")
    latest_elapsed = _finite_number(
        latest.get("elapsed_s"),
        label=f"{label}.latest_progress.elapsed_s",
        errors=errors,
        minimum=0.0,
    )
    chunk_elapsed = _finite_number(
        latest.get("chunk_elapsed_s"),
        label=f"{label}.latest_progress.chunk_elapsed_s",
        errors=errors,
        minimum=0.0,
    )
    latest_rate = _finite_number(
        latest.get("steps_per_second"),
        label=f"{label}.latest_progress.steps_per_second",
        errors=errors,
        minimum=0.0,
    )
    if chunk_elapsed is not None and chunk_elapsed <= 0.0:
        errors.append(f"{label}.latest_progress.chunk_elapsed_s must be positive")
    if latest_rate is not None and latest_rate <= 0.0:
        errors.append(f"{label}.latest_progress.steps_per_second must be positive")
    if (
        chunk_elapsed is not None
        and chunk_elapsed > 0.0
        and latest_rate is not None
        and not math.isclose(
            latest_rate,
            EXPECTED_TASK_BLOCK_SIZE / chunk_elapsed,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        errors.append(f"{label}.latest_progress chunk rate does not reconstruct")
    if elapsed is not None and latest_elapsed is not None and not math.isclose(
        elapsed,
        latest_elapsed,
        rel_tol=1e-12,
    ):
        errors.append(f"{label}.latest_progress.elapsed_s does not match status")
    if (
        latest_rate is not None
        and recent is not None
        and not math.isclose(latest_rate, recent, rel_tol=1e-12)
    ):
        errors.append(f"{label}.latest_progress.steps_per_second does not match status")
    _iso_timestamp(
        latest.get("timestamp_utc"),
        label=f"{label}.latest_progress.timestamp_utc",
        errors=errors,
    )


def _validate_manifest(
    manifest: Mapping[str, object],
    config: Mapping[str, object],
    command: Sequence[str],
    *,
    seed: int,
    source_sha256: Mapping[str, str],
    errors: list[str],
) -> None:
    label = f"seed {seed} manifest"
    _expect_fields(manifest, _MANIFEST_FIELDS, label=label, errors=errors)
    if manifest.get("schema") != UPSTREAM_RESULT_MANIFEST_SCHEMA:
        errors.append(f"{label}.schema is unsupported")
    if manifest.get("argv") != list(command[2:]):
        errors.append(f"{label}.argv does not exactly match the planned seed command")
    if manifest.get("methods") != list(EXPECTED_METHODS):
        errors.append(f"{label}.methods does not match the frozen order")
    _iso_timestamp(manifest.get("created_at_utc"), label=f"{label}.created_at_utc", errors=errors)

    manifest_config = _mapping(manifest.get("config"), label=f"{label}.config", errors=errors)
    _expect_fields(manifest_config, _ARG_CONFIG_FIELDS, label=f"{label}.config", errors=errors)
    expected_config = {field: config.get(field) for field in _ARG_CONFIG_FIELDS}
    _compare_recomputed(
        manifest_config,
        expected_config,
        label=f"{label}.config",
        errors=errors,
    )

    git = _mapping(manifest.get("git"), label=f"{label}.git", errors=errors)
    _expect_fields(git, _GIT_FIELDS, label=f"{label}.git", errors=errors)
    for field in ("branch", "commit", "describe"):
        if not isinstance(git.get(field), str):
            errors.append(f"{label}.git.{field} must be a string")
    if type(git.get("dirty")) is not bool:
        errors.append(f"{label}.git.dirty must be boolean")
    _string_list(git.get("status_porcelain"), label=f"{label}.git.status_porcelain", errors=errors)

    environment = _mapping(
        manifest.get("environment"),
        label=f"{label}.environment",
        errors=errors,
    )
    _expect_fields(environment, _ENVIRONMENT_FIELDS, label=f"{label}.environment", errors=errors)
    for field in (
        "jax",
        "jax_default_backend",
        "jaxlib",
        "numpy",
        "platform",
        "python",
        "python_executable",
    ):
        value = environment.get(field)
        if not isinstance(value, str) or not value:
            errors.append(f"{label}.environment.{field} must be a non-empty string")
    devices = _string_list(
        environment.get("jax_devices"),
        label=f"{label}.environment.jax_devices",
        errors=errors,
    )
    if not devices:
        errors.append(f"{label}.environment.jax_devices must not be empty")

    source = _mapping(
        manifest.get("source_sha256"),
        label=f"{label}.source_sha256",
        errors=errors,
    )
    if set(source) != {"runner", "published_stressors"}:
        errors.append(f"{label}.source_sha256 must bind exactly the two recorded files")
    for role, expected_digest in source_sha256.items():
        digest = source.get(role)
        if not _valid_sha256(digest) or digest != expected_digest:
            errors.append(f"{label}.source_sha256.{role} does not match bundled bytes")


def _validate_seed_result(
    payload: Mapping[str, object],
    status_payload: Mapping[str, object],
    *,
    seed: int,
    command: Sequence[str],
    source_sha256: Mapping[str, str],
    errors: list[str],
) -> Mapping[str, object]:
    label = f"seed {seed} result"
    initial_error_count = len(errors)
    _expect_fields(payload, _RESULT_TOP_FIELDS, label=label, errors=errors)
    config = _mapping(payload.get("config"), label=f"{label}.config", errors=errors)
    _validate_config(config, seed=seed, errors=errors)

    if payload.get("primary_method") != EXPECTED_CANDIDATE_METHODS[0]:
        errors.append(f"{label}.primary_method is not the frozen primary")
    if payload.get("mlp_methods") != list(EXPECTED_MLP_METHODS):
        errors.append(f"{label}.mlp_methods does not match the frozen order")
    if payload.get("candidate_methods") != list(EXPECTED_CANDIDATE_METHODS):
        errors.append(f"{label}.candidate_methods does not match the frozen order")
    if payload.get("evidence_level") != "single_upgd_memory_opmnist_resumable":
        errors.append(f"{label}.evidence_level is unsupported")
    _finite_number(
        payload.get("wall_clock_s"),
        label=f"{label}.wall_clock_s",
        errors=errors,
        minimum=0.0,
    )

    datasets = _mapping(payload.get("datasets"), label=f"{label}.datasets", errors=errors)
    if set(datasets) != {"permuted_mnist_like"}:
        errors.append(f"{label}.datasets must contain exactly permuted_mnist_like")
    dataset = _mapping(
        datasets.get("permuted_mnist_like"),
        label=f"{label}.datasets.permuted_mnist_like",
        errors=errors,
    )
    _validate_dataset(dataset, seed=seed, errors=errors)
    _validate_final_status(status_payload, dataset, seed=seed, errors=errors)

    records_value = payload.get("records")
    if not isinstance(records_value, list) or len(records_value) != 1:
        errors.append(f"{label}.records must contain exactly one seed record")
        record: Mapping[str, object] = {}
    else:
        record = _mapping(records_value[0], label=f"{label}.records[0]", errors=errors)
    _expect_fields(
        record,
        {"dataset", "dataset_name", "methods", "seed"},
        label=f"{label}.record",
        errors=errors,
    )
    if (
        record.get("dataset_name") != "permuted_mnist_like"
        or type(record.get("seed")) is not int
        or record.get("seed") != seed
    ):
        errors.append(f"{label}.record identity does not match seed {seed}")
    if record.get("dataset") != dataset:
        errors.append(f"{label}.record.dataset must equal the top-level dataset metadata")

    methods = _mapping(record.get("methods"), label=f"{label}.record.methods", errors=errors)
    if set(methods) != set(EXPECTED_METHODS):
        errors.append(f"{label}.record.methods must contain exactly the frozen methods")
    for method in EXPECTED_METHODS:
        method_metrics = _mapping(
            methods.get(method),
            label=f"{label}.record.methods.{method}",
            errors=errors,
        )
        if set(method_metrics) != set(CORE_METRICS):
            errors.append(f"{label}.record.methods.{method} fields must be the six metrics")
        for metric in CORE_METRICS:
            _finite_number(
                method_metrics.get(metric),
                label=f"{label}.record.methods.{method}.{metric}",
                errors=errors,
                minimum=0.0,
                maximum=1.0 if metric.endswith("accuracy") else None,
            )

    if (
        len(errors) == initial_error_count
        and set(methods) == set(EXPECTED_METHODS)
        and all(isinstance(methods.get(method), Mapping) for method in EXPECTED_METHODS)
    ):
        recomputed = recompute_opmnist_metrics([record])
        aggregate_outer = _mapping(
            payload.get("aggregate"),
            label=f"{label}.aggregate",
            errors=errors,
        )
        if set(aggregate_outer) != {"permuted_mnist_like"}:
            errors.append(f"{label}.aggregate must contain exactly permuted_mnist_like")
        _compare_recomputed(
            aggregate_outer.get("permuted_mnist_like"),
            recomputed,
            label=f"{label}.aggregate.permuted_mnist_like",
            errors=errors,
        )
        solution_status = _mapping(
            payload.get("solution_status"),
            label=f"{label}.solution_status",
            errors=errors,
        )
        _expect_fields(
            solution_status,
            _SOLUTION_STATUS_FIELDS,
            label=f"{label}.solution_status",
            errors=errors,
        )
        provenance_status = _mapping(
            solution_status.get("artifact_provenance"),
            label=f"{label}.solution_status.artifact_provenance",
            errors=errors,
        )
        _expect_fields(
            provenance_status,
            _PROVENANCE_STATUS_FIELDS,
            label=f"{label}.solution_status.artifact_provenance",
            errors=errors,
        )
        _compare_recomputed(
            solution_status,
            _expected_upstream_seed_status(recomputed),
            label=f"{label}.solution_status",
            errors=errors,
        )

    manifest = _mapping(payload.get("manifest"), label=f"{label}.manifest", errors=errors)
    _validate_manifest(
        manifest,
        config,
        command,
        seed=seed,
        source_sha256=source_sha256,
        errors=errors,
    )
    return record


def _expected_upstream_merged_status(aggregate: Mapping[str, object]) -> dict[str, object]:
    wins = _metric_wins(aggregate)
    winners = [candidate for candidate, metric_wins in wins.items() if all(metric_wins.values())]
    solved = bool(winners)
    return {
        "schema": UPSTREAM_SOLUTION_STATUS_SCHEMA,
        "min_seeds": 3,
        "protocol_complete": True,
        "configured_seed_count": 3,
        "completed_record_seed_count": 3,
        "multi_seed_full_scale": True,
        "artifact_provenance": {
            "schema": UPSTREAM_PROVENANCE_STATUS_SCHEMA,
            "manifest_schema": "alberta.step2.upgd_memory_opmnist.merge_manifest.v1",
            "direct_runner_manifest_complete": False,
            "merged_split_manifest_complete": True,
            "mixed_method_manifest_complete": False,
            "provenance_complete": True,
        },
        "candidate_metric_wins": wins,
        "candidates_winning_all_metrics": winners,
        "solved_opmnist_step2": solved,
        "claim_scope": (
            "multi_seed_full_scale_opmnist_solution"
            if solved
            else "limited_opmnist_evidence_not_step2_solution"
        ),
    }


def _validate_merged_result_and_gate(
    *,
    merged: Mapping[str, object],
    gate: Mapping[str, object],
    seed_results: Sequence[_InputBytes],
    records: Sequence[Mapping[str, object]],
    plan: Mapping[str, object],
    runner_sha256: str,
    merge_source_sha256: str,
    errors: list[str],
) -> None:
    """Cross-check the upstream merge and gate without trusting their booleans."""
    _expect_fields(merged, _MERGED_RESULT_TOP_FIELDS, label="merged result", errors=errors)
    aggregate = recompute_opmnist_metrics(records)
    expected_status = _expected_upstream_merged_status(aggregate)

    merge_command = _string_list(
        plan.get("merge_command"),
        label="plan.merge_command",
        errors=errors,
    )
    try:
        output_flag = merge_command.index("--output")
    except ValueError:
        errors.append("plan.merge_command must contain --output")
        output_flag = 0
    expected_input_paths = merge_command[2:output_flag] if output_flag >= 2 else []
    if len(expected_input_paths) != len(EXPECTED_SEEDS):
        errors.append("plan.merge_command must name exactly three seed results")
    expected_merged_path = plan.get("expected_result")
    if output_flag and (
        output_flag + 1 >= len(merge_command)
        or merge_command[output_flag + 1] != expected_merged_path
    ):
        errors.append("plan.merge_command output does not match plan.expected_result")

    if merged.get("primary_method") != EXPECTED_CANDIDATE_METHODS[0]:
        errors.append("merged result primary_method is not the frozen primary")
    if merged.get("mlp_methods") != list(EXPECTED_MLP_METHODS):
        errors.append("merged result mlp_methods does not match the frozen order")
    if merged.get("candidate_methods") != list(EXPECTED_CANDIDATE_METHODS):
        errors.append("merged result candidate_methods does not match the frozen order")
    if merged.get("evidence_level") != "merged_upgd_memory_opmnist_seed_splits":
        errors.append("merged result evidence_level is unsupported")
    if merged.get("split_results") != expected_input_paths:
        errors.append("merged result split_results does not match the planned inputs")

    expected_records = [
        cast(list[object], result.payload["records"])[0]
        for result in seed_results
    ]
    _compare_recomputed(
        merged.get("records"),
        expected_records,
        label="merged result.records",
        errors=errors,
    )
    aggregate_outer = _mapping(
        merged.get("aggregate"),
        label="merged result.aggregate",
        errors=errors,
    )
    if set(aggregate_outer) != {"permuted_mnist_like"}:
        errors.append("merged result.aggregate must contain exactly permuted_mnist_like")
    _compare_recomputed(
        aggregate_outer.get("permuted_mnist_like"),
        aggregate,
        label="merged result.aggregate.permuted_mnist_like",
        errors=errors,
    )

    merged_config = _mapping(merged.get("config"), label="merged result.config", errors=errors)
    first_config = cast(Mapping[str, object], seed_results[0].payload["config"])
    expected_config = dict(first_config)
    expected_config.update(
        {
            "created_at": merged_config.get("created_at"),
            "merged_from_seed_splits": True,
            "split_result_paths": expected_input_paths,
            "n_seeds": 3,
            "seeds": list(EXPECTED_SEEDS),
        }
    )
    _iso_timestamp(
        merged_config.get("created_at"),
        label="merged result.config.created_at",
        errors=errors,
    )
    _compare_recomputed(
        merged_config,
        expected_config,
        label="merged result.config",
        errors=errors,
    )

    datasets = _mapping(merged.get("datasets"), label="merged result.datasets", errors=errors)
    if set(datasets) != {"permuted_mnist_like"}:
        errors.append("merged result.datasets must contain exactly permuted_mnist_like")
    merged_dataset = _mapping(
        datasets.get("permuted_mnist_like"),
        label="merged result.datasets.permuted_mnist_like",
        errors=errors,
    )
    expected_dataset = dict(cast(Mapping[str, object], records[-1]["dataset"]))
    expected_dataset.update(
        {
            "merged_from_seed_splits": True,
            "split_result_paths": expected_input_paths,
        }
    )
    _compare_recomputed(
        merged_dataset,
        expected_dataset,
        label="merged result.datasets.permuted_mnist_like",
        errors=errors,
    )

    manifest = _mapping(merged.get("manifest"), label="merged result.manifest", errors=errors)
    _expect_fields(manifest, _MERGE_MANIFEST_FIELDS, label="merged result.manifest", errors=errors)
    if manifest.get("schema") != "alberta.step2.upgd_memory_opmnist.merge_manifest.v1":
        errors.append("merged result manifest schema is unsupported")
    _iso_timestamp(
        manifest.get("created_at_utc"),
        label="merged result.manifest.created_at_utc",
        errors=errors,
    )
    for field in ("merge_script", "runner_path"):
        if not isinstance(manifest.get(field), str) or not manifest.get(field):
            errors.append(f"merged result.manifest.{field} must be a non-empty string")
    if manifest.get("merge_script_sha256") != merge_source_sha256:
        errors.append("merged result manifest does not match bundled merge source")
    if manifest.get("runner_sha256") != runner_sha256:
        errors.append("merged result manifest does not match bundled runner source")
    if manifest.get("methods") != list(EXPECTED_METHODS):
        errors.append("merged result manifest methods do not match the frozen order")
    _exact_integer_list(
        manifest.get("seeds"),
        EXPECTED_SEEDS,
        label="merged result.manifest.seeds",
        errors=errors,
    )
    split_rows = manifest.get("split_results")
    if not isinstance(split_rows, list) or len(split_rows) != len(EXPECTED_SEEDS):
        errors.append("merged result manifest must contain exactly three split rows")
    else:
        for seed, row_value, result, expected_path in zip(
            EXPECTED_SEEDS,
            split_rows,
            seed_results,
            expected_input_paths,
            strict=True,
        ):
            row = _mapping(
                row_value,
                label=f"merged result.manifest.split_results[{seed}]",
                errors=errors,
            )
            _expect_fields(
                row,
                _MERGE_SPLIT_RESULT_FIELDS,
                label=f"merged result.manifest.split_results[{seed}]",
                errors=errors,
            )
            expected_row: dict[str, object] = {
                "path": expected_path,
                "sha256": _sha256(result.raw),
                "seeds": [seed],
                "manifest": result.payload["manifest"],
            }
            _compare_recomputed(
                row,
                expected_row,
                label=f"merged result.manifest.split_results[{seed}]",
                errors=errors,
            )

    merged_status = _mapping(
        merged.get("solution_status"),
        label="merged result.solution_status",
        errors=errors,
    )
    _expect_fields(
        merged_status,
        _SOLUTION_STATUS_FIELDS,
        label="merged result.solution_status",
        errors=errors,
    )
    provenance_status = _mapping(
        merged_status.get("artifact_provenance"),
        label="merged result.solution_status.artifact_provenance",
        errors=errors,
    )
    _expect_fields(
        provenance_status,
        _PROVENANCE_STATUS_FIELDS,
        label="merged result.solution_status.artifact_provenance",
        errors=errors,
    )
    _compare_recomputed(
        merged_status,
        expected_status,
        label="merged result.solution_status",
        errors=errors,
    )

    _expect_fields(gate, _GATE_FIELDS, label="solution gate", errors=errors)
    if gate.get("schema") != UPSTREAM_SOLUTION_GATE_SCHEMA:
        errors.append("solution gate schema is unsupported")
    if gate.get("artifact_path") != expected_merged_path:
        errors.append("solution gate artifact_path does not match the planned merged result")
    _compare_recomputed(
        gate.get("status"),
        expected_status,
        label="solution gate.status",
        errors=errors,
    )


def _file_binding(role: str, relative_path: str, raw: bytes) -> dict[str, object]:
    return {
        "role": role,
        "path": relative_path,
        "byte_size": len(raw),
        "sha256": _sha256(raw),
    }


def _bundle_inputs(
    *,
    plan: _InputBytes,
    runbook_raw: bytes,
    result_inputs: Sequence[_InputBytes],
    status_inputs: Sequence[_InputBytes],
    merged_result: _InputBytes,
    solution_gate: _InputBytes,
    runner_source: bytes,
    published_stressors_source: bytes,
    merge_source: bytes,
    solution_gate_source: bytes,
    dataset: bytes,
) -> dict[str, bytes]:
    files = {
        "inputs/plan.json": plan.raw,
        "inputs/RUNBOOK.md": runbook_raw,
        "inputs/source/step2_upgd_memory_opmnist.py": runner_source,
        "inputs/source/step2_published_stressors.py": published_stressors_source,
        "inputs/source/step2_opmnist_merge_seed_results.py": merge_source,
        "inputs/source/step2_opmnist_solution_gate.py": solution_gate_source,
        "inputs/data/mnist_784.arff.gz": dataset,
        "inputs/merged-result.json": merged_result.raw,
        "inputs/solution-gate.json": solution_gate.raw,
    }
    for seed, result in zip(EXPECTED_SEEDS, result_inputs, strict=True):
        files[f"inputs/seed-results/seed-{seed}.json"] = result.raw
    for seed, status in zip(EXPECTED_SEEDS, status_inputs, strict=True):
        files[f"inputs/seed-status/seed-{seed}.json"] = status.raw
    return files


def _build_scientific_payload(
    *,
    plan: _InputBytes,
    runbook_raw: bytes,
    result_inputs: Sequence[_InputBytes],
    status_inputs: Sequence[_InputBytes],
    merged_result: _InputBytes,
    solution_gate: _InputBytes,
    runner_source: bytes,
    published_stressors_source: bytes,
    merge_source: bytes,
    solution_gate_source: bytes,
    dataset: bytes,
) -> dict[str, object]:
    errors: list[str] = []
    commands = _validate_plan(plan.payload, errors=errors)
    _validate_runbook(runbook_raw, errors=errors)
    if not runner_source:
        errors.append("runner source snapshot must not be empty")
    if not published_stressors_source:
        errors.append("published-stressor source snapshot must not be empty")
    if not merge_source:
        errors.append("merge source snapshot must not be empty")
    if not solution_gate_source:
        errors.append("solution-gate source snapshot must not be empty")
    dataset_sha256 = _sha256(dataset)
    if len(dataset) != EXPECTED_OPENML_DATASET_BYTE_SIZE:
        errors.append(
            "OpenML dataset byte size does not match the pinned dataset identity: "
            f"{len(dataset)} != {EXPECTED_OPENML_DATASET_BYTE_SIZE}"
        )
    if dataset_sha256 != EXPECTED_OPENML_DATASET_SHA256:
        errors.append(
            "OpenML dataset SHA-256 does not match the pinned dataset identity: "
            f"{dataset_sha256} != {EXPECTED_OPENML_DATASET_SHA256}"
        )
    if not dataset.startswith(b"\x1f\x8b"):
        errors.append("OpenML dataset identity must be a gzip object")
    source_sha256 = {
        "runner": _sha256(runner_source),
        "published_stressors": _sha256(published_stressors_source),
    }
    if len(result_inputs) != len(EXPECTED_SEEDS):
        errors.append("exactly three seed result files are required")
    if len(status_inputs) != len(EXPECTED_SEEDS):
        errors.append("exactly three final status sidecars are required")
    records: list[Mapping[str, object]] = []
    if (
        len(commands) == len(EXPECTED_SEEDS)
        and len(result_inputs) == len(EXPECTED_SEEDS)
        and len(status_inputs) == len(EXPECTED_SEEDS)
    ):
        for seed, result, status, command in zip(
            EXPECTED_SEEDS,
            result_inputs,
            status_inputs,
            commands,
            strict=True,
        ):
            records.append(
                _validate_seed_result(
                    result.payload,
                    status.payload,
                    seed=seed,
                    command=command,
                    source_sha256=source_sha256,
                    errors=errors,
                )
            )

    if errors:
        raise OPMNISTDevelopmentIngestError("\n".join(errors))

    if len(records) == len(EXPECTED_SEEDS):
        _validate_merged_result_and_gate(
            merged=merged_result.payload,
            gate=solution_gate.payload,
            seed_results=result_inputs,
            records=records,
            plan=plan.payload,
            runner_sha256=source_sha256["runner"],
            merge_source_sha256=_sha256(merge_source),
            errors=errors,
        )

    if errors:
        raise OPMNISTDevelopmentIngestError("\n".join(errors))

    if records:
        cache_roots = {
            cast(str, cast(Mapping[str, object], record["dataset"])["openml_data_home"])
            for record in records
        }
        if len(cache_roots) != 1:
            errors.append("all seed results must report one shared OpenML cache root")

    if errors:
        raise OPMNISTDevelopmentIngestError("\n".join(errors))

    aggregate = recompute_opmnist_metrics(records)
    candidate_metric_wins = _metric_wins(aggregate)
    winners = [
        candidate
        for candidate, wins in candidate_metric_wins.items()
        if all(wins.values())
    ]
    bundled_files = _bundle_inputs(
        plan=plan,
        runbook_raw=runbook_raw,
        result_inputs=result_inputs,
        status_inputs=status_inputs,
        merged_result=merged_result,
        solution_gate=solution_gate,
        runner_source=runner_source,
        published_stressors_source=published_stressors_source,
        merge_source=merge_source,
        solution_gate_source=solution_gate_source,
        dataset=dataset,
    )
    role_by_path = {
        "inputs/plan.json": "upstream_plan",
        "inputs/RUNBOOK.md": "upstream_runbook",
        "inputs/source/step2_upgd_memory_opmnist.py": "runner_source_posthoc",
        "inputs/source/step2_published_stressors.py": "published_stressors_source_posthoc",
        "inputs/source/step2_opmnist_merge_seed_results.py": "merge_source_posthoc",
        "inputs/source/step2_opmnist_solution_gate.py": "solution_gate_source_posthoc",
        "inputs/data/mnist_784.arff.gz": "openml_dataset_cache_posthoc",
        "inputs/merged-result.json": "upstream_merged_result",
        "inputs/solution-gate.json": "upstream_solution_gate",
        **{
            f"inputs/seed-results/seed-{seed}.json": f"upstream_seed_{seed}_result"
            for seed in EXPECTED_SEEDS
        },
        **{
            f"inputs/seed-status/seed-{seed}.json": f"upstream_seed_{seed}_final_status"
            for seed in EXPECTED_SEEDS
        },
    }
    file_bindings = [
        _file_binding(role_by_path[path], path, raw)
        for path, raw in sorted(bundled_files.items())
    ]
    runtime_bindings: list[dict[str, object]] = []
    command_bindings: list[dict[str, object]] = []
    for seed, result, command in zip(EXPECTED_SEEDS, result_inputs, commands, strict=True):
        manifest = cast(Mapping[str, object], result.payload["manifest"])
        environment = cast(Mapping[str, object], manifest["environment"])
        git = cast(Mapping[str, object], manifest["git"])
        runtime_bindings.append(
            {
                "seed": seed,
                "environment_sha256": _sha256(_canonical_json_bytes(environment)),
                "git_metadata_sha256": _sha256(_canonical_json_bytes(git)),
                "manifest_created_at_utc": manifest["created_at_utc"],
            }
        )
        command_bindings.append(
            {
                "seed": seed,
                "planned_command_sha256": _sha256(_canonical_json_bytes(list(command))),
                "result_argv_sha256": _sha256(_canonical_json_bytes(manifest["argv"])),
                "argv_matches_plan": True,
            }
        )

    coverage_by_seed = []
    for seed, record, status_input in zip(
        EXPECTED_SEEDS,
        records,
        status_inputs,
        strict=True,
    ):
        dataset_meta = cast(Mapping[str, object], record["dataset"])
        coverage_by_seed.append(
            {
                "seed": seed,
                "updates": dataset_meta["steps"],
                "dynamic_completed_steps": status_input.payload["completed_steps"],
                "completed_task_blocks": dataset_meta["completed_full_task_blocks"],
                "dynamic_completed_task_blocks": cast(
                    Mapping[str, object],
                    status_input.payload["status"],
                )["completed_full_task_blocks"],
                "heldout_permutation_views": dataset_meta["test_permutation_views"],
                "observed_task_ids_sha256": _sha256(
                    _canonical_json_bytes(dataset_meta["task_ids_observed"])
                ),
                "heldout_task_ids_sha256": _sha256(
                    _canonical_json_bytes(dataset_meta["test_task_ids_evaluated"])
                ),
            }
        )

    return {
        "evidence_policy": dict(_EVIDENCE_POLICY),
        "protocol": {
            "benchmark": "online_permuted_mnist",
            "dataset": "OpenML mnist_784 version 1 canonical 60000/10000 split",
            "expected_seeds": list(EXPECTED_SEEDS),
            "seed_count": len(EXPECTED_SEEDS),
            "updates_per_seed": EXPECTED_STEPS_PER_SEED,
            "total_updates": EXPECTED_STEPS_PER_SEED * len(EXPECTED_SEEDS),
            "task_count_per_seed": EXPECTED_TASKS,
            "task_block_size": EXPECTED_TASK_BLOCK_SIZE,
            "heldout_permutation_views_per_seed": EXPECTED_TEST_VIEWS,
            "prediction_before_update_every_step": True,
            "task_id_provided_to_learner": False,
            "method_order": list(EXPECTED_METHODS),
        },
        "coverage": {
            "protocol_complete": True,
            "multi_seed_full_scale": True,
            "result_count": len(records),
            "seeds": list(EXPECTED_SEEDS),
            "per_seed": coverage_by_seed,
        },
        "metric_recomputation": {
            "core_metrics": list(CORE_METRICS),
            "aggregate": aggregate,
            "candidate_metric_wins": candidate_metric_wins,
            "candidates_winning_all_metrics": winners,
            "all_metric_mean_win": bool(winners),
        },
        "provenance": {
            "files": file_bindings,
            "commands": command_bindings,
            "runtime": runtime_bindings,
            "source_files_bound_posthoc": ["runner", "published_stressors"],
            "source_hashes_match_all_result_manifests": True,
            "merge_and_gate_sources_bound_posthoc": True,
            "merged_result_and_gate_bytes_bound": True,
            "final_status_sidecars_bound": True,
            "dataset_bytes_bound_posthoc": True,
            "dataset_identity": {
                "sha256": dataset_sha256,
                "byte_size": len(dataset),
                "pinned_sha256": EXPECTED_OPENML_DATASET_SHA256,
                "pinned_byte_size": EXPECTED_OPENML_DATASET_BYTE_SIZE,
                "gzip_magic_verified": True,
            },
            "dataset_source_matched_reported_cache_location_at_ingest": True,
            "dataset_digest_recorded_during_execution": False,
            "plan_and_runbook_are_posthoc_locators": True,
            "command_execution_attested": False,
            "full_import_closure_bound_before_execution": False,
            "externally_issued_prerun_manifest_present": False,
        },
        "claims": {
            "protocol_complete": True,
            "multi_seed_full_scale": True,
            "all_metric_mean_win": bool(winners),
            "protocol_completion_is_distinct_from_all_metric_mean_win": True,
            "solved_opmnist_step2": False,
            "sota_claimed": False,
            "alberta_plan_step2_solved": False,
            "claim_scope": "development_published_scale_confirmation_only",
        },
        "limitations": list(_LIMITATIONS),
    }


def _receipt(scientific_payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": OPMNIST_DEVELOPMENT_RECEIPT_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "scientific_payload": dict(scientific_payload),
        "scientific_digest": {
            "algorithm": "sha256",
            "scope": "$.scientific_payload",
            "sha256": _sha256(_canonical_json_bytes(scientific_payload)),
        },
    }


def opmnist_development_receipt_json(receipt: Mapping[str, object]) -> str:
    """Serialize the v1 receipt without permissive non-finite JSON values."""
    return (
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def ingest_opmnist_development_bundle(
    *,
    plan_path: Path,
    runbook_path: Path,
    result_paths: Sequence[Path],
    status_paths: Sequence[Path],
    merged_result_path: Path,
    solution_gate_path: Path,
    runner_source_path: Path,
    published_stressors_source_path: Path,
    merge_source_path: Path,
    solution_gate_source_path: Path,
    dataset_path: Path,
    output_dir: Path,
) -> Path:
    """Validate inputs and durably publish one new read-only bundle directory."""
    if len(result_paths) != len(EXPECTED_SEEDS):
        raise OPMNISTDevelopmentIngestError("exactly three result paths are required")
    if len(status_paths) != len(EXPECTED_SEEDS):
        raise OPMNISTDevelopmentIngestError("exactly three status paths are required")
    plan = _read_json(plan_path, label="plan")
    result_inputs = [
        _read_json(path, label=f"seed result input {index}")
        for index, path in enumerate(result_paths)
    ]
    reported_cache_roots: set[Path] = set()
    for result in result_inputs:
        datasets = result.payload.get("datasets")
        if not isinstance(datasets, Mapping):
            continue
        metadata = datasets.get("permuted_mnist_like")
        if not isinstance(metadata, Mapping):
            continue
        cache_root = metadata.get("openml_data_home")
        if isinstance(cache_root, str) and cache_root:
            reported_cache_roots.add(_absolute_lexical_path(Path(cache_root)))
    if len(reported_cache_roots) != 1:
        raise OPMNISTDevelopmentIngestError(
            "seed results must report exactly one shared OpenML cache root"
    )
    expected_dataset_path = reported_cache_roots.pop() / _OPENML_CACHE_RELATIVE_PATH
    if _absolute_lexical_path(dataset_path) != expected_dataset_path:
        raise OPMNISTDevelopmentIngestError(
            "--dataset-file must be the cache file named by the seed result metadata"
        )
    status_inputs = [
        _read_json(path, label=f"final seed status input {index}")
        for index, path in enumerate(status_paths)
    ]
    merged_result = _read_json(merged_result_path, label="merged result input")
    solution_gate = _read_json(solution_gate_path, label="solution gate input")
    runbook_raw = _read_regular_file_no_symlinks(runbook_path, label="runbook")
    runner_source = _read_regular_file_no_symlinks(runner_source_path, label="runner source")
    published_stressors_source = _read_regular_file_no_symlinks(
        published_stressors_source_path,
        label="published-stressors source",
    )
    merge_source = _read_regular_file_no_symlinks(merge_source_path, label="merge source")
    solution_gate_source = _read_regular_file_no_symlinks(
        solution_gate_source_path,
        label="solution-gate source",
    )
    dataset = _read_regular_file_no_symlinks(dataset_path, label="OpenML dataset")
    scientific_payload = _build_scientific_payload(
        plan=plan,
        runbook_raw=runbook_raw,
        result_inputs=result_inputs,
        status_inputs=status_inputs,
        merged_result=merged_result,
        solution_gate=solution_gate,
        runner_source=runner_source,
        published_stressors_source=published_stressors_source,
        merge_source=merge_source,
        solution_gate_source=solution_gate_source,
        dataset=dataset,
    )
    receipt = _receipt(scientific_payload)
    files = _bundle_inputs(
        plan=plan,
        runbook_raw=runbook_raw,
        result_inputs=result_inputs,
        status_inputs=status_inputs,
        merged_result=merged_result,
        solution_gate=solution_gate,
        runner_source=runner_source,
        published_stressors_source=published_stressors_source,
        merge_source=merge_source,
        solution_gate_source=solution_gate_source,
        dataset=dataset,
    )
    files["receipt.v1.json"] = opmnist_development_receipt_json(receipt).encode("utf-8")

    output_absolute = _absolute_lexical_path(output_dir)
    output_name = output_absolute.name
    if not output_name or output_name in {".", ".."}:
        raise OPMNISTDevelopmentIngestError("output must name a child directory")
    parent_fd = _open_or_create_directory_path_no_symlinks(
        output_absolute.parent,
        label="output parent",
    )
    staging_name = f".{output_name}.staging-{secrets.token_hex(12)}"
    staging_created = False
    published = False
    staging_fd: int | None = None
    try:
        try:
            os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"refusing to overwrite existing output directory: {output_dir}"
            )
        os.mkdir(staging_name, _STAGING_DIRECTORY_MODE, dir_fd=parent_fd)
        staging_created = True
        staging_fd = os.open(staging_name, _OPEN_DIRECTORY_FLAGS, dir_fd=parent_fd)
        _create_bundle_directories(staging_fd)
        for relative, raw in sorted(files.items()):
            _write_file_at(staging_fd, relative, raw)
        _seal_bundle_directories(staging_fd)
        os.fsync(parent_fd)
        try:
            _rename_noreplace_at(parent_fd, staging_name, output_name)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite output directory created during ingestion: {output_dir}"
            ) from exc
        published = True
        os.fsync(parent_fd)
    except BaseException as exc:
        if staging_created and not published:
            try:
                _remove_tree_at(parent_fd, staging_name)
                os.fsync(parent_fd)
            except OSError as cleanup_error:
                exc.add_note(f"staging cleanup also failed: {cleanup_error}")
        raise
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(parent_fd)
    return output_dir / "receipt.v1.json"


def _input_bytes_from_snapshot(
    files: Mapping[str, bytes],
    relative_path: str,
    *,
    label: str,
) -> _InputBytes:
    raw = files[relative_path]
    return _InputBytes(
        path=Path(relative_path),
        raw=raw,
        payload=_decode_strict_json_object(raw, label=label),
    )


def _load_bundle_inputs(files: Mapping[str, bytes]) -> tuple[
    _InputBytes,
    bytes,
    list[_InputBytes],
    list[_InputBytes],
    _InputBytes,
    _InputBytes,
    bytes,
    bytes,
    bytes,
    bytes,
    bytes,
]:
    plan = _input_bytes_from_snapshot(files, "inputs/plan.json", label="bundled plan")
    runbook_raw = files["inputs/RUNBOOK.md"]
    results = [
        _input_bytes_from_snapshot(
            files,
            f"inputs/seed-results/seed-{seed}.json",
            label=f"bundled seed {seed} result",
        )
        for seed in EXPECTED_SEEDS
    ]
    statuses = [
        _input_bytes_from_snapshot(
            files,
            f"inputs/seed-status/seed-{seed}.json",
            label=f"bundled seed {seed} final status",
        )
        for seed in EXPECTED_SEEDS
    ]
    merged_result = _input_bytes_from_snapshot(
        files,
        "inputs/merged-result.json",
        label="bundled merged result",
    )
    solution_gate = _input_bytes_from_snapshot(
        files,
        "inputs/solution-gate.json",
        label="bundled solution gate",
    )
    runner_source = files["inputs/source/step2_upgd_memory_opmnist.py"]
    published_stressors_source = files["inputs/source/step2_published_stressors.py"]
    merge_source = files["inputs/source/step2_opmnist_merge_seed_results.py"]
    solution_gate_source = files["inputs/source/step2_opmnist_solution_gate.py"]
    dataset = files["inputs/data/mnist_784.arff.gz"]
    return (
        plan,
        runbook_raw,
        results,
        statuses,
        merged_result,
        solution_gate,
        runner_source,
        published_stressors_source,
        merge_source,
        solution_gate_source,
        dataset,
    )


def validate_opmnist_development_bundle(
    bundle_dir: Path,
) -> OPMNISTDevelopmentBundleValidation:
    """Rebuild and validate a bundle; never authorize promotion."""
    errors: list[str] = []
    files: dict[str, bytes] = {}
    try:
        files, filesystem_errors = _snapshot_bundle_no_symlinks(bundle_dir)
        errors.extend(filesystem_errors)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    receipt_sha256: str | None = None
    receipt: Mapping[str, object] = {}
    try:
        raw_receipt = files["receipt.v1.json"]
        receipt_sha256 = _sha256(raw_receipt)
        receipt = _decode_strict_json_object(raw_receipt, label="receipt")
    except (KeyError, ValueError) as exc:
        errors.append(str(exc))

    expected_top = {
        "development_only",
        "schema_version",
        "scientific_digest",
        "scientific_payload",
        "scientific_promotion_allowed",
    }
    if receipt:
        _expect_fields(receipt, expected_top, label="receipt", errors=errors)
        if receipt.get("schema_version") != OPMNIST_DEVELOPMENT_RECEIPT_SCHEMA:
            errors.append("receipt schema version is unsupported")
        if receipt.get("development_only") is not True:
            errors.append("receipt must remain development-only")
        if receipt.get("scientific_promotion_allowed") is not False:
            errors.append("receipt must forbid scientific promotion")

    payload = _mapping(receipt.get("scientific_payload"), label="scientific_payload", errors=errors)
    digest = _mapping(receipt.get("scientific_digest"), label="scientific_digest", errors=errors)
    _expect_fields(
        digest,
        {"algorithm", "scope", "sha256"},
        label="scientific_digest",
        errors=errors,
    )
    if digest.get("algorithm") != "sha256" or digest.get("scope") != "$.scientific_payload":
        errors.append("scientific_digest algorithm or scope is unsupported")
    expected_digest = _sha256(_canonical_json_bytes(payload))
    if digest.get("sha256") != expected_digest:
        errors.append("scientific payload digest mismatch")

    rebuilt: Mapping[str, object] = {}
    try:
        (
            plan,
            runbook_raw,
            result_inputs,
            status_inputs,
            merged_result,
            solution_gate,
            runner_source,
            published_stressors_source,
            merge_source,
            solution_gate_source,
            dataset,
        ) = _load_bundle_inputs(files)
        rebuilt = _build_scientific_payload(
            plan=plan,
            runbook_raw=runbook_raw,
            result_inputs=result_inputs,
            status_inputs=status_inputs,
            merged_result=merged_result,
            solution_gate=solution_gate,
            runner_source=runner_source,
            published_stressors_source=published_stressors_source,
            merge_source=merge_source,
            solution_gate_source=solution_gate_source,
            dataset=dataset,
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(f"malformed bundled input: {exc}")
    if rebuilt:
        _compare_recomputed(payload, rebuilt, label="scientific_payload", errors=errors)

    claims = _mapping(payload.get("claims"), label="scientific_payload.claims", errors=errors)
    protocol_complete = claims.get("protocol_complete") is True
    all_metric_mean_win = claims.get("all_metric_mean_win") is True
    if claims.get("solved_opmnist_step2") is not False:
        errors.append("development receipt must never claim solved_opmnist_step2")
    return OPMNISTDevelopmentBundleValidation(
        valid=not errors,
        protocol_complete=protocol_complete,
        all_metric_mean_win=all_metric_mean_win,
        solved_opmnist_step2=False,
        errors=tuple(errors),
        receipt_sha256=receipt_sha256,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="create a new development bundle")
    ingest.add_argument("--plan", type=Path, required=True)
    ingest.add_argument("--runbook", type=Path, required=True)
    ingest.add_argument("--result", type=Path, action="append", required=True)
    ingest.add_argument("--status", type=Path, action="append", required=True)
    ingest.add_argument("--merged-result", type=Path, required=True)
    ingest.add_argument("--solution-gate", type=Path, required=True)
    ingest.add_argument("--runner-source", type=Path, required=True)
    ingest.add_argument("--published-stressors-source", type=Path, required=True)
    ingest.add_argument("--merge-source", type=Path, required=True)
    ingest.add_argument("--solution-gate-source", type=Path, required=True)
    ingest.add_argument("--dataset-file", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="validate an existing bundle")
    validate.add_argument("bundle", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the strict development ingestion or validation command."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    if args.command == "ingest":
        receipt_path = ingest_opmnist_development_bundle(
            plan_path=args.plan,
            runbook_path=args.runbook,
            result_paths=args.result,
            status_paths=args.status,
            merged_result_path=args.merged_result,
            solution_gate_path=args.solution_gate,
            runner_source_path=args.runner_source,
            published_stressors_source_path=args.published_stressors_source,
            merge_source_path=args.merge_source,
            solution_gate_source_path=args.solution_gate_source,
            dataset_path=args.dataset_file,
            output_dir=args.output,
        )
        validation = validate_opmnist_development_bundle(args.output)
        if not validation.valid:
            raise OPMNISTDevelopmentIngestError("\n".join(validation.errors))
        logger.info(
            json.dumps(
                {
                    "receipt": str(receipt_path),
                    "receipt_sha256": validation.receipt_sha256,
                    "protocol_complete": validation.protocol_complete,
                    "all_metric_mean_win": validation.all_metric_mean_win,
                    "solved_opmnist_step2": False,
                    "development_only": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    validation = validate_opmnist_development_bundle(args.bundle)
    logger.info(
        json.dumps(
            {
                "valid": validation.valid,
                "errors": list(validation.errors),
                "receipt_sha256": validation.receipt_sha256,
                "protocol_complete": validation.protocol_complete,
                "all_metric_mean_win": validation.all_metric_mean_win,
                "solved_opmnist_step2": False,
                "development_only": True,
                "scientific_promotion_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if validation.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
