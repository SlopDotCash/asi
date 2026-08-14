"""Adversarial contract tests for the active UPGD IPMNIST v3 lifecycle.

"Active" means the strict v3 execution contract in
:mod:`alberta_framework.benchmarks.upgd_ipmnist_v3` — the lifecycle the
public ``upgd_ipmnist.main`` dispatches to exclusively; the legacy v2 CLI
survives only as ``main_v2_compat`` for historical shard merging, with its
unverifiable direct-aggregate mode removed.  v3 splits execution into three
commands (issue an immutable plan -> execute exactly one learner/seed shard
-> merge the exact planned Cartesian product), publishes every file
atomically at a new path, and is permanently nonpromoting because the
execution envelope is self-recorded.  The benchmark science itself (the
Elsayed & Mahmood ICLR 2024 Input-permuted MNIST replication) lives in
:mod:`alberta_framework.benchmarks.upgd_ipmnist` and is not re-tested here;
these tests attack the lifecycle: dispatch, immutable publication, plan
issuance, single-seed partials, and exact merge.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import stat
import statistics
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.benchmarks.upgd_ipmnist_v3 as v3
from alberta_framework.benchmarks.upgd_ipmnist import (
    ADAMW_PROTOCOL_HYPERPARAMETERS,
    UPGD_W_PROTOCOL_HYPERPARAMETERS,
    IPMNISTConfig,
    IPMNISTRunResult,
    main_v2_compat,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    main as active_main,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    run_ipmnist as run_ipmnist_exact,
)
from alberta_framework.benchmarks.upgd_ipmnist_v3 import (
    ARTIFACT_SCHEMA,
    PARTIAL_SCHEMA,
    PLAN_SCHEMA,
    UPGDIPMNISTV3Error,
    atomic_write_new,
    canonical_json_bytes,
    canonical_json_sha256,
    main,
    merge_partials,
    validate_artifact,
    validate_partial,
    validate_plan,
    write_partial_for_result,
    write_plan,
)
from alberta_framework.evaluation.upgd_ipmnist_v3 import main as evaluation_main

REAL_REPLAY_PARTIAL_MEASUREMENTS = v3._replay_partial_measurements
REAL_LOAD_PINNED_MNIST = v3._load_pinned_mnist

# Shrinks only what v3 leaves free: plan validation pins the pinned-MNIST
# data identity (input_dim=784, n_classes=10, task_length within the
# 60,000-row train split), so a tiny fixture may reduce task count/length
# and hidden widths but not the input/output dimensions.
TINY = IPMNISTConfig(
    n_tasks=3,
    task_length=2,
    input_dim=784,
    hidden1=3,
    hidden2=2,
    n_classes=10,
)
SEEDS = tuple(range(100, 120))
FAKE_MNIST_ARCHIVE = b"deterministic fake MNIST archive bytes"
FAKE_OPENML_CACHE_BYTES = {
    "openml/openml.org/api/v1/json/data/554.gz": b"fake-data-description",
    "openml/openml.org/api/v1/json/data/features/554.gz": b"fake-feature-description",
    "openml/openml.org/api/v1/json/data/qualities/554.gz": b"fake-quality-description",
    v3.MNIST_ARCHIVE_RELATIVE_PATH.as_posix(): FAKE_MNIST_ARCHIVE,
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: object, *, bind_artifact_output: bool = True) -> Path:
    persisted = copy.deepcopy(value)
    if (
        bind_artifact_output
        and isinstance(persisted, dict)
        and persisted.get("schema") == ARTIFACT_SCHEMA
        and isinstance(persisted.get("partial_manifest"), list)
        and isinstance(persisted.get("merge_execution"), dict)
    ):
        argv = persisted["merge_execution"].get("prescribed_merge_argv")
        if isinstance(argv, list):
            output_index = 5 + len(persisted["partial_manifest"])
            if output_index < len(argv):
                argv[output_index] = v3._lexical_absolute(path).as_posix()
    path.write_bytes(canonical_json_bytes(persisted))
    path.chmod(0o444)
    return path


def _copy_complete_cache(source_home: Path, destination_home: Path) -> Path:
    for relative in FAKE_OPENML_CACHE_BYTES:
        source = source_home / relative
        destination = destination_home / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        destination.chmod(0o444)
    return destination_home / v3.MNIST_ARCHIVE_RELATIVE_PATH


def _result(learner: str, seed: int) -> IPMNISTRunResult:
    learner_offset = 0.1 if learner == "upgd_w" else 0.0
    seed_offset = (seed - SEEDS[0]) * 0.001
    accuracy = np.asarray(
        [[0.50 + learner_offset + seed_offset, 0.55 + learner_offset, 0.60]],
        dtype=np.float64,
    )
    loss = np.asarray([[1.2, 1.0, 0.8]], dtype=np.float64)
    plasticity = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64)
    hyperparameters = (
        dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
        if learner == "upgd_w"
        else dict(ADAMW_PROTOCOL_HYPERPARAMETERS)
    )
    return IPMNISTRunResult(
        learner=learner,
        hyperparameters=hyperparameters,
        seeds=(seed,),
        config=TINY,
        per_task_accuracy=accuracy,
        per_task_loss=loss,
        per_task_plasticity=plasticity,
        average_online_accuracy=accuracy.mean(axis=1),
        wall_clock_seconds=1.25,
    )


def _contains_key(value: object, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(_contains_key(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


@pytest.fixture(scope="module", autouse=True)
def pinned_test_archive() -> Iterator[None]:
    original_size = v3.MNIST_ARCHIVE_BYTE_SIZE
    original_sha256 = v3.MNIST_ARCHIVE_SHA256
    original_cache_identities = v3.OPENML_CACHE_MEMBER_IDENTITIES
    original_synthetic_config = v3._ALLOW_SYNTHETIC_CONFIG_FOR_TESTING
    v3.MNIST_ARCHIVE_BYTE_SIZE = len(FAKE_MNIST_ARCHIVE)
    v3.MNIST_ARCHIVE_SHA256 = v3.sha256_bytes(FAKE_MNIST_ARCHIVE)
    v3.OPENML_CACHE_MEMBER_IDENTITIES = tuple(
        (relative, len(raw), v3.sha256_bytes(raw))
        for relative, raw in FAKE_OPENML_CACHE_BYTES.items()
    )
    v3._ALLOW_SYNTHETIC_CONFIG_FOR_TESTING = True
    try:
        yield
    finally:
        v3.MNIST_ARCHIVE_BYTE_SIZE = original_size
        v3.MNIST_ARCHIVE_SHA256 = original_sha256
        v3.OPENML_CACHE_MEMBER_IDENTITIES = original_cache_identities
        v3._ALLOW_SYNTHETIC_CONFIG_FOR_TESTING = original_synthetic_config


@pytest.fixture(scope="module")
def bundle(
    tmp_path_factory: pytest.TempPathFactory,
    pinned_test_archive: None,
) -> Iterator[dict[str, Any]]:
    del pinned_test_archive
    # Freeze one internally coherent source/runtime capture for this synthetic
    # unit fixture. Production code still rebuilds live bindings; dedicated
    # adversarial tests replace this frozen provider to exercise drift.
    source_closure: dict[str, Any] | None = None
    for _ in range(20):
        try:
            source_closure = v3._build_source_import_closure()
            break
        except UPGDIPMNISTV3Error as exc:
            if "changed while its closure was being built" not in str(exc):
                raise
    assert source_closure is not None
    runtime_manifest = v3._build_runtime_manifest()
    patch = pytest.MonkeyPatch()
    patch.setattr(v3, "_build_source_import_closure", lambda: copy.deepcopy(source_closure))
    patch.setattr(v3, "_build_runtime_manifest", lambda: copy.deepcopy(runtime_manifest))
    patch.setattr(
        v3,
        "_load_pinned_mnist",
        lambda _home, *, context: (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        ),
    )

    def synthetic_replay(
        partials: dict[tuple[str, int], dict[str, Any]],
        run_spec: dict[str, Any],
        _data_home: Path,
    ) -> None:
        planned = {
            (learner, seed)
            for learner in run_spec["learner_ids"]
            for seed in run_spec["seed_schedule"]["seed_ids"]
        }
        assert set(partials) <= planned
        for learner, seed in sorted(partials):
            expected = v3._measurement_payload_from_result(_result(learner, seed))
            if not v3._json_exact_equal(partials[(learner, seed)]["measurements"], expected):
                raise UPGDIPMNISTV3Error(
                    f"recorded measurements differ from exact replay for {(learner, seed)}"
                )

    patch.setattr(v3, "_replay_partial_measurements", synthetic_replay)
    root = tmp_path_factory.mktemp("upgd_ipmnist_v3")
    data_home = root / "data"
    archive = data_home / v3.MNIST_ARCHIVE_RELATIVE_PATH
    for relative, raw in FAKE_OPENML_CACHE_BYTES.items():
        member = data_home / relative
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(raw)
        member.chmod(0o444)
    plan = write_plan(
        root / "plan.v3.json",
        TINY,
        SEEDS,
        data_home,
        archive,
        issued_unix=10,
        issuer_argv=("pytest", "issue-plan"),
    )
    partials: list[Path] = []
    for learner in v3.LEARNER_IDS:
        for seed in SEEDS:
            partial = root / "shards" / f"{learner}-{seed}.json"
            write_partial_for_result(
                plan,
                _result(learner, seed),
                partial,
                process_argv=("pytest", learner, str(seed)),
                started_unix=20,
                finished_unix=22,
            )
            partials.append(partial)
    artifact = merge_partials(
        plan,
        tuple(reversed(partials)),
        root / "artifact.v3.json",
        created_unix=30,
        process_argv=("pytest", "merge"),
    )
    try:
        yield {
            "root": root,
            "data_home": data_home,
            "archive": archive,
            "plan": plan,
            "partials": tuple(partials),
            "artifact": artifact,
        }
    finally:
        patch.undo()


class TestActiveDispatch:
    def test_active_cli_requires_an_explicit_v3_lifecycle_command(self) -> None:
        assert PLAN_SCHEMA == "alberta.upgd_ipmnist.plan.v3"
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2

    def test_public_runner_dispatches_to_v3_without_v2_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        observed: list[list[str]] = []

        def fake_v3(argv: list[str]) -> Path:
            observed.append(argv)
            return Path("unused")

        monkeypatch.setattr(v3, "main", fake_v3)
        active_main(["plan", "--plan-out", "p", "--seed-list", "7"])
        assert observed == [["plan", "--plan-out", "p", "--seed-list", "7"]]

    def test_legacy_v2_direct_aggregate_mode_is_disabled(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main_v2_compat([])
        assert exc_info.value.code == 2

    def test_shard_command_cannot_omit_its_only_valid_output(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["shard", "--plan", "p", "--learner-id", "upgd_w", "--seed-id", "7"])
        assert exc_info.value.code == 2

    def test_cli_plan_origin_is_derived_and_argv_is_only_prescribed(
        self, bundle: dict[str, Any]
    ) -> None:
        output = bundle["root"] / "cli-origin-plan.json"
        main(
            [
                "plan",
                "--plan-out",
                str(output),
                "--seed-list",
                ",".join(str(seed) for seed in SEEDS),
                "--data-home",
                str(bundle["data_home"]),
                "--data-archive",
                str(bundle["archive"]),
                "--n-tasks",
                str(TINY.n_tasks),
                "--task-length",
                str(TINY.task_length),
                "--input-dim",
                str(TINY.input_dim),
                "--hidden1",
                str(TINY.hidden1),
                "--hidden2",
                str(TINY.hidden2),
                "--n-classes",
                str(TINY.n_classes),
            ]
        )
        issuance = _read(output)["plan"]["issuance"]
        assert issuance["invocation_origin"] == "cli"
        assert issuance["prescribed_argv"][0] == "plan"
        assert issuance["unattested_caller_argv"][0] == "plan"


class TestImmutablePublication:
    def test_atomic_writer_refuses_overwrite_and_preserves_original(self, tmp_path: Path) -> None:
        target = atomic_write_new(tmp_path / "sealed.json", b"first\n")
        assert stat.S_IMODE(target.stat().st_mode) == 0o444
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            atomic_write_new(target, b"second\n")
        assert target.read_bytes() == b"first\n"

    def test_atomic_writer_cleans_temporary_file_after_publish_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # This test uses os.link() which is POSIX-only. On Windows,
        # atomic_write_new uses pathlib.Path.replace() instead.
        if platform.system() == "Windows":
            pytest.skip("os.link() monkeypatch not applicable on Windows")

        def fail_link(*_args: object, **_kwargs: object) -> None:
            raise OSError("simulated link failure")

        monkeypatch.setattr(os, "link", fail_link)
        with pytest.raises(OSError, match="simulated link failure"):
            atomic_write_new(tmp_path / "never-published.json", b"payload")
        assert list(tmp_path.iterdir()) == []

    def test_atomic_writer_removes_its_target_after_post_link_readback_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # This test relies on os.link() which is POSIX-only
        if platform.system() == "Windows":
            pytest.skip("os.link() not applicable on Windows")

        target = tmp_path / "post-link-failure.json"
        real_reader = v3._read_regular_bytes

        def fail_target_read(path: Path, *, require_immutable: bool) -> bytes:
            if v3._lexical_absolute(path) == v3._lexical_absolute(target):
                raise OSError("simulated post-link readback failure")
            return real_reader(path, require_immutable=require_immutable)

        monkeypatch.setattr(v3, "_read_regular_bytes", fail_target_read)
        with pytest.raises(OSError, match="post-link readback"):
            atomic_write_new(target, b"payload")
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_atomic_writer_detects_temporary_name_swap_before_link(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # This test uses os.link() which is POSIX-only
        if platform.system() == "Windows":
            pytest.skip("os.link() not applicable on Windows")

        real_link = os.link
        substituted_name = ""

        def swap_then_link(
            source: str,
            destination: str,
            *,
            src_dir_fd: int,
            dst_dir_fd: int,
            follow_symlinks: bool,
        ) -> None:
            nonlocal substituted_name
            substituted_name = source
            os.unlink(source, dir_fd=src_dir_fd)
            replacement_fd = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=src_dir_fd,
            )
            try:
                os.write(replacement_fd, b"attacker bytes")
                os.fchmod(replacement_fd, 0o444)
            finally:
                os.close(replacement_fd)
            real_link(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )

        monkeypatch.setattr(os, "link", swap_then_link)
        target = tmp_path / "must-not-publish.json"
        with pytest.raises(UPGDIPMNISTV3Error, match="descriptor-anchored"):
            atomic_write_new(target, b"trusted bytes")
        assert target.read_bytes() == b"attacker bytes"
        assert stat.S_IMODE(target.stat().st_mode) == 0o444
        substituted = tmp_path / substituted_name
        assert substituted.read_bytes() == b"attacker bytes"
        assert set(tmp_path.iterdir()) == {target, substituted}

    def test_atomic_writer_preserves_temporary_name_substitution_after_link(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # This test uses os.link() and directory FDs which are POSIX-only
        if platform.system() == "Windows":
            pytest.skip("os.link() and dir_fd not applicable on Windows")

        real_link = os.link
        substituted_name = ""

        def link_then_substitute(
            source: str,
            destination: str,
            *,
            src_dir_fd: int,
            dst_dir_fd: int,
            follow_symlinks: bool,
        ) -> None:
            nonlocal substituted_name
            real_link(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )
            substituted_name = source
            os.unlink(source, dir_fd=src_dir_fd)
            replacement_fd = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=src_dir_fd,
            )
            try:
                os.write(replacement_fd, b"attacker replacement")
            finally:
                os.close(replacement_fd)

        monkeypatch.setattr(os, "link", link_then_substitute)
        target = atomic_write_new(tmp_path / "published.json", b"trusted bytes")

        assert target.read_bytes() == b"trusted bytes"
        substituted = tmp_path / substituted_name
        assert substituted.read_bytes() == b"attacker replacement"

    def test_atomic_writer_preserves_temporary_name_substitution_on_link_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # This test uses os.link() and directory FDs which are POSIX-only
        if platform.system() == "Windows":
            pytest.skip("os.link() and dir_fd not applicable on Windows")

        substituted_name = ""

        def substitute_then_fail(
            source: str,
            _destination: str,
            *,
            src_dir_fd: int,
            **_kwargs: object,
        ) -> None:
            nonlocal substituted_name
            substituted_name = source
            os.unlink(source, dir_fd=src_dir_fd)
            replacement_fd = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=src_dir_fd,
            )
            try:
                os.write(replacement_fd, b"attacker replacement")
            finally:
                os.close(replacement_fd)
            raise OSError("simulated link failure after substitution")

        monkeypatch.setattr(os, "link", substitute_then_fail)
        target = tmp_path / "unpublished.json"
        with pytest.raises(OSError, match="after substitution"):
            atomic_write_new(target, b"trusted bytes")

        assert not target.exists()
        substituted = tmp_path / substituted_name
        assert substituted.read_bytes() == b"attacker replacement"

    def test_atomic_writer_does_not_follow_destination_or_ancestor_symlinks(
        self, tmp_path: Path
    ) -> None:
        # Symlinks require elevated privileges on Windows
        if platform.system() == "Windows":
            pytest.skip("Symlink creation requires elevated privileges on Windows")

        sentinel = tmp_path / "sentinel"
        sentinel.write_bytes(b"keep")
        destination_link = tmp_path / "destination-link"
        destination_link.symlink_to(sentinel)
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            atomic_write_new(destination_link, b"replace")
        assert sentinel.read_bytes() == b"keep"

        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        parent_link = tmp_path / "parent-link"
        parent_link.symlink_to(real_parent, target_is_directory=True)
        with pytest.raises(OSError):
            atomic_write_new(parent_link / "escaped.json", b"payload")
        assert not (real_parent / "escaped.json").exists()

    def test_atomic_writer_detects_ancestor_replacement_and_cleans_owned_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # This test relies on directory FDs which are POSIX-only
        if platform.system() == "Windows":
            pytest.skip("Directory FDs not applicable on Windows")

        parent = tmp_path / "parent"
        parent.mkdir()
        moved = tmp_path / "moved"
        original = v3._assert_parent_locator_stable
        checks = 0

        def replace_before_link(destination: Path, directory_fd: int | None) -> None:
            nonlocal checks
            checks += 1
            if checks == 2:
                parent.rename(moved)
                parent.mkdir()
            original(destination, directory_fd)

        monkeypatch.setattr(v3, "_assert_parent_locator_stable", replace_before_link)
        with pytest.raises(UPGDIPMNISTV3Error, match="ancestor directory changed"):
            atomic_write_new(parent / "evidence.json", b"trusted")
        assert not (parent / "evidence.json").exists()
        assert not (moved / "evidence.json").exists()
        assert not list(moved.glob(".*.tmp"))

    def test_reader_detects_same_byte_locator_substitution(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # This test relies on directory FDs which are POSIX-only
        if platform.system() == "Windows":
            pytest.skip("Directory FDs not applicable on Windows")

        target = atomic_write_new(tmp_path / "target.json", b"payload")
        moved = tmp_path / "moved.json"
        real_stat = os.stat
        substituted = False

        def substitute_before_locator_stat(
            path: object,
            *args: object,
            **kwargs: object,
        ) -> os.stat_result:
            nonlocal substituted
            if (
                not substituted
                and path == target.name
                and kwargs.get("dir_fd") is not None
            ):
                substituted = True
                target.rename(moved)
                target.write_bytes(b"payload")
                target.chmod(0o444)
            return real_stat(path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(os, "stat", substitute_before_locator_stat)
        with pytest.raises(UPGDIPMNISTV3Error, match="locator was replaced"):
            v3._read_regular_bytes(target, require_immutable=True)
        assert target.read_bytes() == moved.read_bytes() == b"payload"

    def test_atomic_writer_detects_same_byte_substitution_during_readback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # This test relies on directory FDs which are POSIX-only
        if platform.system() == "Windows":
            pytest.skip("Directory FDs not applicable on Windows")

        target = tmp_path / "target.json"
        moved = tmp_path / "moved.json"
        real_reader = v3._read_regular_bytes
        substituted = False

        def substitute_after_read(path: Path, *, require_immutable: bool) -> bytes:
            nonlocal substituted
            raw = real_reader(path, require_immutable=require_immutable)
            if v3._lexical_absolute(path) == target and not substituted:
                substituted = True
                target.rename(moved)
                target.write_bytes(raw)
                target.chmod(0o444)
            return raw

        monkeypatch.setattr(v3, "_read_regular_bytes", substitute_after_read)
        with pytest.raises(UPGDIPMNISTV3Error, match="changed during byte verification"):
            atomic_write_new(target, b"payload")
        assert target.read_bytes() == moved.read_bytes() == b"payload"

    def test_strict_reader_rejects_symlink_hardlink_writable_and_nonregular_plan(
        self, bundle: dict[str, Any]
    ) -> None:
        root = bundle["root"]
        plan_bytes = bundle["plan"].read_bytes()

        writable = root / "writable-plan.json"
        writable.write_bytes(plan_bytes)
        writable.chmod(0o644)
        report = validate_plan(writable)
        assert not report.valid
        assert "write permission" in report.errors[0]

        sealed = root / "hardlinked-plan.json"
        sealed.write_bytes(plan_bytes)
        sealed.chmod(0o444)
        alias = root / "hardlinked-plan-alias.json"
        alias.hardlink_to(sealed)
        report = validate_plan(alias)
        assert not report.valid
        assert "exactly one hard link" in report.errors[0]

        symlink = root / "symlink-plan.json"
        symlink.symlink_to(bundle["plan"])
        report = validate_plan(symlink)
        assert not report.valid

        fifo = root / "plan.fifo"
        os.mkfifo(fifo)
        report = validate_plan(fifo)
        assert not report.valid
        assert "regular file" in report.errors[0]

    def test_worker_preflight_rejects_occupied_output_before_any_data_or_seed_work(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        occupied = atomic_write_new(bundle["root"] / "occupied-partial.json", b"occupied")
        effects: list[str] = []

        def forbidden_loader(
            _path: Path, *, context: str
        ) -> tuple[np.ndarray, np.ndarray]:
            del context
            effects.append("dataset")
            raise AssertionError("dataset loader must not run")

        monkeypatch.setattr(v3, "_load_pinned_mnist", forbidden_loader)
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            v3.run_shard(
                bundle["plan"],
                "upgd_w",
                SEEDS[0],
                occupied,
                process_argv=("pytest", "must-not-run"),
            )
        assert effects == []

    def test_dataset_reader_rejects_symlink_hardlink_and_writable_archive(
        self, bundle: dict[str, Any]
    ) -> None:
        root = bundle["root"] / "adversarial-data"

        writable_home = root / "writable"
        writable = _copy_complete_cache(bundle["data_home"], writable_home)
        writable.chmod(0o644)
        with pytest.raises(UPGDIPMNISTV3Error, match="write permission"):
            v3._build_data_manifest(writable_home, writable)

        hardlink_home = root / "hardlink"
        hardlink = _copy_complete_cache(bundle["data_home"], hardlink_home)
        alias = root / "archive-alias.gz"
        alias.hardlink_to(hardlink)
        with pytest.raises(UPGDIPMNISTV3Error, match="exactly one hard link"):
            v3._build_data_manifest(hardlink_home, hardlink)

        symlink_home = root / "symlink"
        symlink = _copy_complete_cache(bundle["data_home"], symlink_home)
        symlink.unlink()
        target = root / "archive-target.gz"
        target.write_bytes(FAKE_MNIST_ARCHIVE)
        target.chmod(0o444)
        symlink.symlink_to(target)
        with pytest.raises(OSError):
            v3._build_data_manifest(symlink_home, symlink)


class TestIssuedPlan:
    def test_plan_binds_full_closed_execution_spec(self, bundle: dict[str, Any]) -> None:
        report = validate_plan(bundle["plan"])
        assert report.valid
        assert report.scientific_promotion_allowed is False
        payload = _read(bundle["plan"])
        assert payload["schema"] == PLAN_SCHEMA
        assert payload["plan_sha256"] == canonical_json_sha256(payload["plan"])
        spec = payload["plan"]["run_spec"]
        assert spec["learner_ids"] == ["upgd_w", "adamw"]
        assert spec["seed_schedule"]["seed_ids"] == list(SEEDS)
        assert spec["seed_schedule"]["known_consumed_seed_ids_excluded"] == list(range(10))
        assert spec["planned_shard_count"] == 40
        assert spec["selected_publication_match"] == {
            "scope": "network_task_horizon_and_selected_learner_hyperparameters",
            "configuration": "mismatch",
            "hyperparameters_by_learner": {"upgd_w": "match", "adamw": "match"},
            "all_selected_fields": "mismatch",
        }
        assert [item["id"] for item in spec["deviations"]] == [
            "rng_schedule",
            "metric_blocks",
            "bias_correction_dtype",
            "upgd_inner_loop",
            "seed_count",
        ]
        assert payload["plan"]["source_import_closure"]["closure_kind"] == (
            "static_transitive_local_python_imports_plus_lockfiles"
        )
        assert payload["plan"]["data_manifest"]["content"]["sha256"]
        assert payload["plan"]["runtime_manifest"]["scikit_learn"]
        assert payload["plan"]["runtime_manifest"]["jax_devices"]
        runtime = payload["plan"]["runtime_manifest"]
        assert set(runtime["distribution_content"]) == set(v3._RUNTIME_CONTENT_DISTRIBUTIONS)
        assert {
            "etils",
            "msgpack",
            "orbax-checkpoint",
            "protobuf",
            "tensorstore",
        } <= set(runtime["distribution_content"])
        assert runtime["distribution_content_scope"] == {
            "kind": "explicit_python_execution_distribution_set",
            "distribution_names": list(v3._RUNTIME_CONTENT_DISTRIBUTIONS),
            "dynamic_import_closure_claimed": False,
            "native_system_library_closure_claimed": False,
        }
        assert payload["plan"]["issuance"]["prescribed_argv"][0] == "plan"
        assert payload["plan"]["issuance"]["invocation_origin"] == "direct_python_api"
        assert not _contains_key(payload, "is_protocol_exact")

    def test_dataset_locator_can_change_when_bound_bytes_do_not(
        self, bundle: dict[str, Any]
    ) -> None:
        relocated_home = bundle["root"] / "relocated-data-home"
        relocated = _copy_complete_cache(bundle["data_home"], relocated_home)
        report = validate_plan(
            bundle["plan"],
            data_home=relocated_home,
            data_archive=relocated,
        )
        assert report.valid

    def test_plan_requires_complete_offline_cache_and_materialized_preflight(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        missing_member = "openml/openml.org/api/v1/json/data/554-missing.gz"
        monkeypatch.setattr(
            v3,
            "OPENML_CACHE_MEMBER_IDENTITIES",
            (
                (
                    missing_member,
                    4,
                    v3.sha256_bytes(b"meta"),
                ),
                (
                    v3.MNIST_ARCHIVE_RELATIVE_PATH.as_posix(),
                    len(FAKE_MNIST_ARCHIVE),
                    v3.sha256_bytes(FAKE_MNIST_ARCHIVE),
                ),
            ),
        )
        with pytest.raises(FileNotFoundError):
            v3._build_data_manifest(bundle["data_home"], bundle["archive"])

        monkeypatch.setattr(
            v3,
            "OPENML_CACHE_MEMBER_IDENTITIES",
            (
                (
                    v3.MNIST_ARCHIVE_RELATIVE_PATH.as_posix(),
                    len(FAKE_MNIST_ARCHIVE),
                    v3.sha256_bytes(FAKE_MNIST_ARCHIVE),
                ),
            ),
        )
        monkeypatch.setattr(
            v3,
            "_load_pinned_mnist",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                UPGDIPMNISTV3Error("offline materialized preflight failed")
            ),
        )
        output = bundle["root"] / "preflight-failure-plan.json"
        with pytest.raises(UPGDIPMNISTV3Error, match="materialized preflight"):
            write_plan(
                output,
                TINY,
                SEEDS,
                bundle["data_home"],
                bundle["archive"],
                issuer_argv=("pytest", "preflight-failure"),
            )
        assert not output.exists()

    def test_exact_data_id_loader_denies_network_after_complete_cache_check(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sklearn import datasets  # type: ignore[import-untyped]
        from sklearn.datasets import _openml  # type: ignore[import-untyped]

        observed: list[dict[str, object]] = []

        def fake_fetch_openml(**kwargs: object) -> SimpleNamespace:
            observed.append(dict(kwargs))
            with pytest.raises(UPGDIPMNISTV3Error, match="forbidden OpenML network"):
                _openml.urlopen("https://openml.org/must-not-run")
            return SimpleNamespace(
                data=np.zeros((1, 1), dtype=np.float32),
                target=np.zeros((1,), dtype=np.int32),
            )

        monkeypatch.setattr(datasets, "fetch_openml", fake_fetch_openml)
        monkeypatch.setattr(v3, "_validate_loaded_mnist", lambda _x, _y: None)
        REAL_LOAD_PINNED_MNIST(bundle["data_home"], context="network-denial-test")
        assert observed == [
            {
                "data_id": 554,
                "as_frame": False,
                "data_home": str(bundle["data_home"]),
                "n_retries": 1,
                "delay": 0.0,
                "parser": "pandas",
            }
        ]

    def test_missing_cache_member_fails_before_sklearn_or_http(
        self,
        bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sklearn import datasets

        incomplete_home = tmp_path / "incomplete-cache"
        _copy_complete_cache(bundle["data_home"], incomplete_home)
        missing = incomplete_home / "openml/openml.org/api/v1/json/data/features/554.gz"
        missing.unlink()
        calls: list[str] = []
        def forbidden_fetch(**_kwargs: object) -> SimpleNamespace:
            calls.append("sklearn")
            return SimpleNamespace()

        monkeypatch.setattr(datasets, "fetch_openml", forbidden_fetch)
        with pytest.raises(UPGDIPMNISTV3Error, match="dataset load failed"):
            REAL_LOAD_PINNED_MNIST(incomplete_home, context="missing-member-test")
        assert calls == []

    @pytest.mark.parametrize(
        "bad_seeds",
        [
            SEEDS[:-1],
            (*SEEDS[:-1], True),
            (*SEEDS[:-1], SEEDS[-2]),
            (*SEEDS[:-2], SEEDS[-1], SEEDS[-2]),
            (-1, *SEEDS[1:]),
            (0, *SEEDS[1:]),
        ],
    )
    def test_plan_rejects_invalid_or_implicit_seed_schedules(
        self, bundle: dict[str, Any], bad_seeds: tuple[int, ...]
    ) -> None:
        with pytest.raises(UPGDIPMNISTV3Error):
            v3.build_run_spec(TINY, v3.LEARNER_IDS, bad_seeds)

    def test_plan_rejects_incomplete_or_mismatched_learner_set(self) -> None:
        with pytest.raises(UPGDIPMNISTV3Error, match="exact canonical learner pair"):
            v3.build_run_spec(TINY, ("upgd_w",), SEEDS)

    @pytest.mark.parametrize(
        ("config", "message"),
        [
            (replace(TINY, input_dim=4), "input_dim=784"),
            (replace(TINY, n_classes=2), "n_classes=10"),
            (replace(TINY, task_length=60_001), "60000-row train split"),
        ],
    )
    def test_plan_rejects_dataset_incompatible_configuration(
        self, config: IPMNISTConfig, message: str
    ) -> None:
        with pytest.raises(UPGDIPMNISTV3Error, match=message):
            v3.build_run_spec(config, v3.LEARNER_IDS, SEEDS)

    def test_active_contract_requires_exact_selected_publication_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(v3, "_ALLOW_SYNTHETIC_CONFIG_FOR_TESTING", False)
        exact = v3.build_run_spec(IPMNISTConfig(), v3.LEARNER_IDS, SEEDS)
        assert exact["selected_publication_match"]["all_selected_fields"] == "match"
        with pytest.raises(UPGDIPMNISTV3Error, match="exact selected publication"):
            v3.build_run_spec(TINY, v3.LEARNER_IDS, SEEDS)

    def test_schedule_tamper_fails_even_after_digest_rewrite(self, bundle: dict[str, Any]) -> None:
        payload = copy.deepcopy(_read(bundle["plan"]))
        payload["plan"]["run_spec"]["seed_schedule"]["seed_count"] = len(SEEDS) + 1
        payload["plan"]["run_spec_sha256"] = canonical_json_sha256(payload["plan"]["run_spec"])
        payload["plan_sha256"] = canonical_json_sha256(payload["plan"])
        path = _write(bundle["root"] / "tampered-schedule.json", payload)
        report = validate_plan(path)
        assert not report.valid
        assert any("derived closed specification" in error for error in report.errors)

    def test_hyperparameter_tamper_fails_after_digest_rewrite(self, bundle: dict[str, Any]) -> None:
        payload = copy.deepcopy(_read(bundle["plan"]))
        payload["plan"]["run_spec"]["hyperparameters"]["upgd_w"]["step_size"] = 0.02
        payload["plan"]["run_spec_sha256"] = canonical_json_sha256(payload["plan"]["run_spec"])
        payload["plan_sha256"] = canonical_json_sha256(payload["plan"])
        path = _write(bundle["root"] / "tampered-hyperparameters.json", payload)
        report = validate_plan(path)
        assert not report.valid
        assert any(
            "selected canonical arm" in error or "derived closed specification" in error
            for error in report.errors
        )

    def test_fully_rederived_noncanonical_hyperparameter_is_rejected(
        self, bundle: dict[str, Any]
    ) -> None:
        payload = copy.deepcopy(_read(bundle["plan"]))
        spec = payload["plan"]["run_spec"]
        spec["hyperparameters"]["upgd_w"]["step_size"] = -999.0
        spec["selected_publication_match"]["hyperparameters_by_learner"]["upgd_w"] = "mismatch"
        spec["selected_publication_match"]["all_selected_fields"] = "mismatch"
        payload["plan"]["run_spec_sha256"] = canonical_json_sha256(spec)
        payload["plan_sha256"] = canonical_json_sha256(payload["plan"])
        path = _write(bundle["root"] / "rederived-hyperparameters.json", payload)
        report = validate_plan(path)
        assert not report.valid
        assert any("selected canonical arm" in error for error in report.errors)

    def test_prescribed_plan_argv_is_recomputed(self, bundle: dict[str, Any]) -> None:
        payload = copy.deepcopy(_read(bundle["plan"]))
        payload["plan"]["issuance"]["prescribed_argv"][4] = "999"
        payload["plan_sha256"] = canonical_json_sha256(payload["plan"])
        path = _write(bundle["root"] / "tampered-plan-argv.json", payload)
        report = validate_plan(path)
        assert not report.valid
        assert any("prescribed plan argv" in error for error in report.errors)

    def test_closed_deviation_tamper_fails_after_digest_rewrite(
        self, bundle: dict[str, Any]
    ) -> None:
        payload = copy.deepcopy(_read(bundle["plan"]))
        payload["plan"]["run_spec"]["deviations"][0]["implementation"] = "free prose"
        payload["plan"]["run_spec_sha256"] = canonical_json_sha256(payload["plan"]["run_spec"])
        payload["plan_sha256"] = canonical_json_sha256(payload["plan"])
        path = _write(bundle["root"] / "tampered-deviation.json", payload)
        report = validate_plan(path)
        assert not report.valid
        assert any("derived closed specification" in error for error in report.errors)

    def test_recursive_legacy_marker_is_rejected_first(self, bundle: dict[str, Any]) -> None:
        payload = copy.deepcopy(_read(bundle["plan"]))
        payload["plan"]["runtime_manifest"]["nested"] = {"is_protocol_exact": True}
        payload["plan_sha256"] = canonical_json_sha256(payload["plan"])
        path = _write(bundle["root"] / "legacy-marker.json", payload)
        report = validate_plan(path)
        assert not report.valid
        assert any("forbidden legacy marker" in error for error in report.errors)

    @pytest.mark.parametrize(
        "schema",
        ["upgd_ipmnist.plan.v1", "alberta.upgd_ipmnist.plan.v2"],
    )
    def test_v1_and_v2_plans_are_rejected(self, bundle: dict[str, Any], schema: str) -> None:
        path = _write(
            bundle["root"] / f"legacy-plan-{schema.rsplit('.', 1)[-1]}.json",
            {"schema": schema},
        )
        assert not validate_plan(path).valid


class TestSingleSeedPartials:
    def test_partials_are_one_dimensional_and_share_plan_digest(
        self, bundle: dict[str, Any]
    ) -> None:
        plan = _read(bundle["plan"])
        for path in bundle["partials"]:
            report = validate_partial(path, bundle["plan"])
            assert report.valid
            payload = _read(path)
            assert payload["schema"] == PARTIAL_SCHEMA
            assert payload["plan_binding"]["plan_sha256"] == plan["plan_sha256"]
            assert payload["evidence_policy"]["scientific_promotion_allowed"] is False
            assert payload["execution"]["execution_origin"] == "direct_supplied_result"
            assert payload["execution"]["seed_reservation_binding"] is None
            assert payload["execution"]["prescribed_worker_argv"][0] == "direct_api"
            for values in payload["measurements"].values():
                assert np.asarray(values).shape == (TINY.n_tasks,)

    def test_partial_validation_replays_measurements(
        self, bundle: dict[str, Any]
    ) -> None:
        payload = copy.deepcopy(_read(bundle["partials"][0]))
        payload["measurements"]["per_task_accuracy"][0] = 0.0
        path = _write(bundle["root"] / "forged-but-structural-partial.json", payload)
        report = validate_partial(path, bundle["plan"])
        assert not report.valid
        assert any(
            "recorded measurements differ from exact replay" in error
            for error in report.errors
        )

    def test_supplied_result_cannot_claim_runner_origin_without_reservation(
        self, bundle: dict[str, Any]
    ) -> None:
        payload = copy.deepcopy(_read(bundle["partials"][0]))
        payload["execution"]["execution_origin"] = "benchmark_runner"
        payload["execution"]["prescribed_worker_argv"][0] = "shard"
        payload["execution"]["prescribed_worker_argv"].pop(1)
        path = _write(bundle["root"] / "fake-runner-origin.json", payload)
        report = validate_partial(path, bundle["plan"])
        assert not report.valid
        assert any("requires a persistent seed reservation" in error for error in report.errors)

    def test_worker_refuses_existing_shard_before_execution(self, bundle: dict[str, Any]) -> None:
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            v3.run_shard(
                bundle["plan"],
                "upgd_w",
                SEEDS[0],
                bundle["partials"][0],
                process_argv=("pytest", "occupied-output"),
            )

    def test_plan_scoped_reservation_allows_only_one_concurrent_seed_execution(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seed = SEEDS[2]
        barrier = threading.Barrier(2)
        real_acquire = v3._acquire_seed_reservation
        executions: list[tuple[str, int]] = []

        def synchronized_acquire(**kwargs: Any) -> dict[str, Any]:
            barrier.wait(timeout=10)
            return real_acquire(**kwargs)

        def runner(
            _x: np.ndarray,
            _y: np.ndarray,
            learner: str,
            seeds: tuple[int, ...],
            **_kwargs: object,
        ) -> IPMNISTRunResult:
            executions.append((learner, seeds[0]))
            return _result(learner, seeds[0])

        monkeypatch.setattr(v3, "_acquire_seed_reservation", synchronized_acquire)
        monkeypatch.setattr(v3, "run_ipmnist", runner)

        def invoke(index: int) -> Path | BaseException:
            try:
                return v3.run_shard(
                    bundle["plan"],
                    "upgd_w",
                    seed,
                    bundle["root"] / f"reserved-race-{index}.json",
                    process_argv=("pytest", "reservation-race", str(index)),
                )
            except BaseException as exc:  # returned for deterministic thread assertion
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(invoke, range(2)))
        published = [item for item in outcomes if isinstance(item, Path)]
        rejected = [item for item in outcomes if isinstance(item, FileExistsError)]
        assert len(published) == 1
        assert len(rejected) == 1
        assert executions == [("upgd_w", seed)]
        assert validate_partial(published[0], bundle["plan"]).valid
        payload = _read(published[0])
        binding = payload["execution"]["seed_reservation_binding"]
        reservation = Path(binding["locator"])
        assert reservation.is_file()
        assert stat.S_IMODE(reservation.stat().st_mode) == 0o444
        assert reservation.stat().st_nlink == 1

    def test_failed_seed_execution_leaves_a_consuming_reservation(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seed = SEEDS[3]
        executions: list[str] = []

        def fail_runner(*_args: object, **_kwargs: object) -> IPMNISTRunResult:
            executions.append("failed")
            raise RuntimeError("simulated runner failure")

        monkeypatch.setattr(v3, "run_ipmnist", fail_runner)
        with pytest.raises(RuntimeError, match="runner failure"):
            v3.run_shard(
                bundle["plan"],
                "adamw",
                seed,
                bundle["root"] / "failed-reserved-seed.json",
                process_argv=("pytest", "failed-reservation"),
            )
        reservation = v3._seed_reservation_path(
            bundle["plan"],
            _read(bundle["plan"]),
            "adamw",
            seed,
        )
        assert reservation.is_file()
        assert stat.S_IMODE(reservation.stat().st_mode) == 0o444

        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            v3.run_shard(
                bundle["plan"],
                "adamw",
                seed,
                bundle["root"] / "different-output-same-consumed-seed.json",
                process_argv=("pytest", "retry-must-not-run"),
            )
        assert executions == ["failed"]

    @pytest.mark.parametrize("progress_every", [0, -1, True])
    def test_worker_preflights_argv_and_progress_before_seed_execution(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        progress_every: object,
    ) -> None:
        effects: list[str] = []
        def observed_loader(
            _home: Path, *, context: str
        ) -> tuple[np.ndarray, np.ndarray]:
            del context
            effects.append("dataset")
            return np.empty((0, 0)), np.empty((0,))

        monkeypatch.setattr(v3, "_load_pinned_mnist", observed_loader)
        monkeypatch.setattr(
            v3,
            "run_ipmnist",
            lambda *_args, **_kwargs: effects.append("seed"),
        )
        output = bundle["root"] / f"invalid-progress-{progress_every!r}.json"
        with pytest.raises(UPGDIPMNISTV3Error, match="progress_every"):
            v3.run_shard(
                bundle["plan"],
                "upgd_w",
                SEEDS[0],
                output,
                progress_every=cast(Any, progress_every),
                process_argv=("pytest", "invalid-progress"),
            )
        assert effects == []
        assert not output.exists()

        empty_argv_output = bundle["root"] / f"empty-argv-{progress_every!r}.json"
        with pytest.raises(UPGDIPMNISTV3Error, match="process argv"):
            v3.run_shard(
                bundle["plan"],
                "upgd_w",
                SEEDS[0],
                empty_argv_output,
                progress_every=1,
            )
        assert effects == []
        assert not empty_argv_output.exists()

    def test_writer_refuses_preplan_clock_and_invalid_duration(
        self, bundle: dict[str, Any]
    ) -> None:
        with pytest.raises(UPGDIPMNISTV3Error, match="before plan issuance"):
            write_partial_for_result(
                bundle["plan"],
                _result("upgd_w", SEEDS[0]),
                bundle["root"] / "preplan-write.json",
                process_argv=("pytest", "preplan"),
                started_unix=9,
                finished_unix=10,
            )
        invalid_duration = replace(
            _result("upgd_w", SEEDS[0]),
            wall_clock_seconds=float("nan"),
        )
        with pytest.raises(UPGDIPMNISTV3Error, match="wall_clock_seconds"):
            write_partial_for_result(
                bundle["plan"],
                invalid_duration,
                bundle["root"] / "nonfinite-duration-write.json",
                process_argv=("pytest", "nonfinite-duration"),
                started_unix=20,
                finished_unix=21,
            )

    def test_partial_cannot_be_validated_against_a_different_plan(
        self, bundle: dict[str, Any]
    ) -> None:
        other_plan = write_plan(
            bundle["root"] / "different-plan.v3.json",
            TINY,
            tuple(range(200, 220)),
            bundle["data_home"],
            bundle["archive"],
            issued_unix=11,
            issuer_argv=("pytest", "different-plan"),
        )
        report = validate_partial(bundle["partials"][0], other_plan)
        assert not report.valid
        assert any(
            "plan binding locator" in error
            or "plan byte" in error
            or "plan digest" in error
            for error in report.errors
        )

    def test_extra_nested_execution_key_is_rejected(self, bundle: dict[str, Any]) -> None:
        payload = copy.deepcopy(_read(bundle["partials"][0]))
        payload["execution"]["unplanned"] = "value"
        path = _write(bundle["root"] / "extra-nested-key.json", payload)
        report = validate_partial(path, bundle["plan"])
        assert not report.valid
        assert any("execution keys differ" in error for error in report.errors)

    def test_preplan_timestamp_and_huge_integer_fail_closed(self, bundle: dict[str, Any]) -> None:
        before_plan = copy.deepcopy(_read(bundle["partials"][0]))
        before_plan["execution"]["started_unix"] = 9
        path = _write(bundle["root"] / "before-plan.json", before_plan)
        report = validate_partial(path, bundle["plan"])
        assert not report.valid
        assert any("before plan issuance" in error for error in report.errors)

        huge = copy.deepcopy(_read(bundle["partials"][0]))
        huge["execution"]["duration_seconds"] = 10**1000
        huge_path = _write(bundle["root"] / "huge-integer.json", huge)
        huge_report = validate_partial(huge_path, bundle["plan"])
        assert not huge_report.valid
        assert any("duration_seconds invalid" in error for error in huge_report.errors)

    def test_worker_argv_and_self_reported_digest_are_recomputed(
        self, bundle: dict[str, Any]
    ) -> None:
        prescribed = copy.deepcopy(_read(bundle["partials"][0]))
        prescribed["execution"]["prescribed_worker_argv"][4] = "adamw"
        prescribed_path = _write(bundle["root"] / "tampered-worker-argv.json", prescribed)
        report = validate_partial(prescribed_path, bundle["plan"])
        assert not report.valid
        assert any("prescribed worker argv" in error for error in report.errors)

        raw = copy.deepcopy(_read(bundle["partials"][0]))
        raw["execution"]["unattested_caller_argv"].append("tampered")
        raw_path = _write(bundle["root"] / "tampered-raw-argv.json", raw)
        raw_report = validate_partial(raw_path, bundle["plan"])
        assert not raw_report.valid
        assert any("caller argv digest" in error for error in raw_report.errors)

        locator = copy.deepcopy(_read(bundle["partials"][0]))
        locator["execution"]["partial_locator"] = 123
        locator_path = _write(bundle["root"] / "numeric-partial-locator.json", locator)
        locator_report = validate_partial(locator_path, bundle["plan"])
        assert not locator_report.valid
        assert any("partial_locator" in error for error in locator_report.errors)

        relative = copy.deepcopy(_read(bundle["partials"][0]))
        relative["execution"]["partial_locator"] = "relative.json"
        relative["execution"]["prescribed_worker_argv"][9] = "relative.json"
        relative_path = _write(bundle["root"] / "relative-partial-locator.json", relative)
        relative_report = validate_partial(relative_path, bundle["plan"])
        assert not relative_report.valid
        assert any("must be absolute" in error for error in relative_report.errors)

        relative_plan = copy.deepcopy(_read(bundle["partials"][0]))
        relative_plan["plan_binding"]["locator"] = "../plan.json"
        relative_plan_path = _write(
            bundle["root"] / "relative-plan-binding.json",
            relative_plan,
        )
        relative_plan_report = validate_partial(relative_plan_path, bundle["plan"])
        assert not relative_plan_report.valid
        assert any("plan_binding.locator" in error for error in relative_plan_report.errors)

    def test_plan_binding_byte_size_requires_an_integer(self, bundle: dict[str, Any]) -> None:
        payload = copy.deepcopy(_read(bundle["partials"][0]))
        payload["plan_binding"]["byte_size"] = float(payload["plan_binding"]["byte_size"])
        path = _write(bundle["root"] / "float-plan-byte-size.json", payload)
        report = validate_partial(path, bundle["plan"])
        assert not report.valid
        assert any("plan byte_size mismatch" in error for error in report.errors)

    def test_reservation_binding_requires_the_exact_plan_scoped_locator(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        learner = "adamw"
        seed = SEEDS[-1]
        monkeypatch.setattr(
            v3,
            "run_ipmnist",
            lambda *_args, **_kwargs: _result(learner, seed),
        )
        published = v3.run_shard(
            bundle["plan"],
            learner,
            seed,
            bundle["root"] / "canonical-reservation-partial.json",
            process_argv=("pytest", "canonical-reservation"),
        )
        payload = copy.deepcopy(_read(published))
        binding = payload["execution"]["seed_reservation_binding"]
        original = Path(binding["locator"])
        relocated = bundle["root"] / "relocated-reservation.json"
        relocated.write_bytes(original.read_bytes())
        relocated.chmod(0o444)
        binding["locator"] = relocated.as_posix()
        path = _write(bundle["root"] / "relocated-reservation-binding.json", payload)

        report = validate_partial(path, bundle["plan"])

        assert not report.valid
        assert any("exact plan-scoped path" in error for error in report.errors)

    def test_reservation_binding_rejects_noncanonical_immutable_bytes_even_when_rebound(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        learner = "adamw"
        seed = SEEDS[5]
        monkeypatch.setattr(
            v3,
            "run_ipmnist",
            lambda *_args, **_kwargs: _result(learner, seed),
        )
        published = v3.run_shard(
            bundle["plan"],
            learner,
            seed,
            bundle["root"] / "noncanonical-reservation-source.json",
            process_argv=("pytest", "noncanonical-reservation"),
        )
        payload = copy.deepcopy(_read(published))
        binding = payload["execution"]["seed_reservation_binding"]
        reservation_path = Path(binding["locator"])
        reservation = _read(reservation_path)
        noncanonical = json.dumps(reservation, separators=(",", ":")).encode("utf-8")
        assert noncanonical != canonical_json_bytes(reservation)
        reservation_path.chmod(0o644)
        reservation_path.write_bytes(noncanonical)
        reservation_path.chmod(0o444)
        binding["byte_size"] = len(noncanonical)
        binding["sha256"] = v3.sha256_bytes(noncanonical)
        rebound = _write(bundle["root"] / "noncanonical-reservation-rebound.json", payload)

        report = validate_partial(rebound, bundle["plan"])

        assert not report.valid
        assert any("seed reservation is not canonical" in error for error in report.errors)

    def test_future_lifecycle_timestamps_fail_closed(
        self,
        bundle: dict[str, Any],
    ) -> None:
        future = int(time.time()) + 60
        future_plan = copy.deepcopy(_read(bundle["plan"]))
        future_plan["issued_unix"] = future
        future_plan_path = _write(bundle["root"] / "future-plan.json", future_plan)
        assert not validate_plan(future_plan_path).valid

        future_partial = copy.deepcopy(_read(bundle["partials"][0]))
        future_partial["execution"]["started_unix"] = future
        future_partial["execution"]["finished_unix"] = future
        future_partial["execution"]["duration_seconds"] = 0.0
        future_partial_path = _write(
            bundle["root"] / "future-partial.json",
            future_partial,
        )
        assert not validate_partial(future_partial_path, bundle["plan"]).valid

        future_artifact = copy.deepcopy(_read(bundle["artifact"]))
        future_artifact["created_unix"] = future
        future_artifact_path = _write(
            bundle["root"] / "future-artifact.json",
            future_artifact,
        )
        assert not validate_artifact(
            future_artifact_path,
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
        ).valid

    def test_public_validators_wrap_unexpected_failures(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected validation failure")

        monkeypatch.setattr(v3, "_read_validated_plan", fail)
        assert not validate_plan(bundle["plan"]).valid
        monkeypatch.undo()

        monkeypatch.setattr(v3, "_replay_partial_measurements", fail)
        assert not validate_partial(bundle["partials"][0], bundle["plan"]).valid
        monkeypatch.undo()

        monkeypatch.setattr(v3, "_validate_artifact_payload", fail)
        assert not validate_artifact(bundle["artifact"], plan_path=bundle["plan"]).valid

    def test_public_artifact_validation_requires_an_external_plan(
        self, bundle: dict[str, Any]
    ) -> None:
        report = validate_artifact(
            bundle["artifact"],
            partial_paths=bundle["partials"],
        )
        assert not report.valid
        assert any("requires an immutable external plan" in error for error in report.errors)

        with pytest.raises(SystemExit) as exc_info:
            evaluation_main(["artifact", str(bundle["artifact"])])
        assert exc_info.value.code == 2

    def test_duplicate_json_keys_and_nested_nonfinite_values_are_rejected(
        self, bundle: dict[str, Any]
    ) -> None:
        duplicate = bundle["root"] / "duplicate-key.json"
        duplicate.write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
        duplicate.chmod(0o444)
        assert not validate_partial(duplicate, bundle["plan"]).valid

        payload = copy.deepcopy(_read(bundle["partials"][0]))
        payload["measurements"]["per_task_loss"][0] = float("nan")
        nonfinite = bundle["root"] / "nonfinite.json"
        nonfinite.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
        nonfinite.chmod(0o444)
        assert not validate_partial(nonfinite, bundle["plan"]).valid

    @pytest.mark.parametrize(
        "schema",
        ["upgd_ipmnist.partial.v1", "alberta.upgd_ipmnist.partial.v2"],
    )
    def test_v1_and_v2_partials_are_explicitly_rejected(
        self, bundle: dict[str, Any], schema: str
    ) -> None:
        path = _write(
            bundle["root"] / f"legacy-{schema.rsplit('.', 1)[-1]}.json",
            {"schema": schema},
        )
        report = validate_partial(path, bundle["plan"])
        assert not report.valid
        assert any("partial keys differ" in error for error in report.errors)


class TestExactMergeAndArtifact:
    def test_merge_preflights_argv_and_binding_policy_before_replay(
        self, bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        effects: list[str] = []
        monkeypatch.setattr(
            v3,
            "_replay_partial_measurements",
            lambda *_args: effects.append("replay"),
        )
        empty_output = bundle["root"] / "merge-empty-argv.json"
        with pytest.raises(UPGDIPMNISTV3Error, match="process argv"):
            merge_partials(bundle["plan"], bundle["partials"], empty_output)
        assert effects == []
        assert not empty_output.exists()

        disabled_output = bundle["root"] / "merge-disabled-bindings.json"
        with pytest.raises(UPGDIPMNISTV3Error, match="requires current"):
            merge_partials(
                bundle["plan"],
                bundle["partials"],
                disabled_output,
                process_argv=("pytest", "disabled-bindings"),
                verify_current_bindings=False,
            )
        assert effects == []
        assert not disabled_output.exists()

    def test_partial_merge_and_artifact_validation_accept_pinned_relocated_cache(
        self, bundle: dict[str, Any]
    ) -> None:
        relocated_home = bundle["root"] / "relocated-complete-cache"
        relocated_archive = _copy_complete_cache(bundle["data_home"], relocated_home)

        assert validate_partial(
            bundle["partials"][0],
            bundle["plan"],
            data_home=relocated_home,
            data_archive=relocated_archive,
        ).valid
        relocated_artifact = merge_partials(
            bundle["plan"],
            bundle["partials"],
            bundle["root"] / "relocated-data-artifact.json",
            process_argv=("pytest", "merge-relocated-data"),
            data_home=relocated_home,
            data_archive=relocated_archive,
        )
        payload = _read(relocated_artifact)
        assert payload["computational_replay"]["data_locators_used"] == {
            "data_home": relocated_home.as_posix(),
            "archive": relocated_archive.as_posix(),
        }
        assert validate_artifact(
            relocated_artifact,
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
            data_home=relocated_home,
            data_archive=relocated_archive,
        ).valid
        mismatched_cache_report = validate_artifact(
            bundle["artifact"],
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
            data_home=relocated_home,
            data_archive=relocated_archive,
        )
        assert not mismatched_cache_report.valid
        assert any(
            "computational replay receipt" in error
            or "exact effective data cache" in error
            for error in mismatched_cache_report.errors
        )

    def test_artifact_recomputes_from_relocated_shards(self, bundle: dict[str, Any]) -> None:
        relocated_dir = bundle["root"] / "relocated-shards"
        relocated_dir.mkdir(exist_ok=True)
        relocated: list[Path] = []
        for index, original in enumerate(reversed(bundle["partials"])):
            copy_path = relocated_dir / f"copy-{index}.json"
            copy_path.write_bytes(original.read_bytes())
            copy_path.chmod(0o444)
            relocated.append(copy_path)
        report = validate_artifact(
            bundle["artifact"],
            partial_paths=relocated,
            plan_path=bundle["plan"],
        )
        assert report.valid
        assert report.scientific_promotion_allowed is False
        artifact = _read(bundle["artifact"])
        assert artifact["schema"] == ARTIFACT_SCHEMA
        assert artifact["coverage"]["complete"] is True
        assert artifact["coverage"]["observed_count"] == 40
        assert len(artifact["partial_manifest"]) == 40
        assert artifact["computational_replay"] == {
            "kind": "exact_full_reexecution",
            "completed": True,
            "shard_count": 40,
            "source_import_closure_sha256": artifact["run_plan"]["plan"][
                "source_import_closure_sha256"
            ],
            "runtime_manifest_sha256": artifact["run_plan"]["plan"][
                "runtime_manifest_sha256"
            ],
            "data_manifest_sha256": artifact["run_plan"]["plan"][
                "data_manifest_sha256"
            ],
            "data_locators_used": {
                "data_home": bundle["data_home"].as_posix(),
                "archive": bundle["archive"].as_posix(),
            },
        }
        paired = artifact["comparison"]["paired_seed_comparison"]
        assert paired["contract"]["pairing_unit"] == "seed_id"
        assert paired["contract"]["post_hoc_acceptance_gate"] is False
        assert len(paired["per_seed_deltas"]) == 20
        assert sum(paired["sign_counts"].values()) == 20
        assert paired["confidence_interval"]["lower"] <= paired["mean_delta"]
        assert paired["confidence_interval"]["upper"] >= paired["mean_delta"]
        assert artifact["merge_execution"]["invocation_origin"] == "direct_python_api"
        assert artifact["merge_execution"]["prescribed_merge_argv"][0] == "merge"
        assert "wall_clock_seconds" not in artifact["learners"]["upgd_w"]

    def test_paired_statistics_match_an_independent_numerical_oracle(self) -> None:
        deltas = [float(index - 10) / 100.0 for index in range(20)]
        adamw_accuracy = np.full((20,), 0.5, dtype=np.float64)
        upgd_accuracy = adamw_accuracy + np.asarray(deltas, dtype=np.float64)

        def aggregate(learner: str, accuracy: np.ndarray) -> IPMNISTRunResult:
            per_task = np.repeat(accuracy[:, None], TINY.n_tasks, axis=1)
            return IPMNISTRunResult(
                learner=learner,
                hyperparameters=(
                    dict(UPGD_W_PROTOCOL_HYPERPARAMETERS)
                    if learner == "upgd_w"
                    else dict(ADAMW_PROTOCOL_HYPERPARAMETERS)
                ),
                seeds=SEEDS,
                config=TINY,
                per_task_accuracy=per_task,
                per_task_loss=np.ones_like(per_task),
                per_task_plasticity=np.zeros_like(per_task),
                average_online_accuracy=accuracy,
                wall_clock_seconds=0.0,
            )

        results = {
            "upgd_w": aggregate("upgd_w", upgd_accuracy),
            "adamw": aggregate("adamw", adamw_accuracy),
        }
        summaries = {
            learner: v3._scientific_summary(result)
            for learner, result in results.items()
        }
        paired = v3._v3_comparison(results, summaries)["paired_seed_comparison"]
        expected_mean = statistics.fmean(deltas)
        expected_sd = statistics.stdev(deltas)
        expected_se = expected_sd / (20.0**0.5)
        half_width = 2.093024054408263 * expected_se
        assert paired["per_seed_deltas"] == pytest.approx(deltas, abs=1e-15)
        assert paired["mean_delta"] == pytest.approx(expected_mean, abs=1e-15)
        assert paired["sample_standard_deviation"] == pytest.approx(expected_sd, abs=1e-15)
        assert paired["standard_error"] == pytest.approx(expected_se, abs=1e-15)
        assert paired["confidence_interval"] == pytest.approx(
            {"lower": expected_mean - half_width, "upper": expected_mean + half_width},
            abs=1e-15,
        )
        assert paired["sign_counts"] == {
            "upgd_w_higher": 9,
            "equal": 1,
            "adamw_higher": 10,
        }

    def test_merge_rejects_structurally_valid_fabricated_measurements(
        self, bundle: dict[str, Any]
    ) -> None:
        forged_payload = copy.deepcopy(_read(bundle["partials"][0]))
        forged_payload["measurements"]["per_task_accuracy"] = [0.0] * TINY.n_tasks
        forged = _write(bundle["root"] / "forged-measurements.json", forged_payload)
        paths = [forged, *bundle["partials"][1:]]
        with pytest.raises(
            UPGDIPMNISTV3Error,
            match="recorded measurements differ from exact replay",
        ):
            merge_partials(
                bundle["plan"],
                paths,
                bundle["root"] / "forged-artifact.json",
                process_argv=("pytest", "merge-forged"),
            )

    def test_real_replay_oracle_reexecutes_every_identity_and_compares_exactly(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = _read(bundle["plan"])
        run_spec = plan["plan"]["run_spec"]
        partials = {
            (payload["learner_id"], payload["seed_id"]): payload
            for payload in (_read(path) for path in bundle["partials"])
        }
        calls: list[tuple[str, int]] = []

        monkeypatch.setattr(
            v3,
            "_load_pinned_mnist",
            lambda _home, *, context: (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            ),
        )
        monkeypatch.setattr(v3, "_validate_loaded_mnist", lambda _x, _y: None)

        def replay_runner(
            _x: np.ndarray,
            _y: np.ndarray,
            learner: str,
            seeds: tuple[int, ...],
            **_kwargs: object,
        ) -> IPMNISTRunResult:
            calls.append((learner, seeds[0]))
            return _result(learner, seeds[0])

        monkeypatch.setattr(v3, "run_ipmnist", replay_runner)
        REAL_REPLAY_PARTIAL_MEASUREMENTS(partials, run_spec, bundle["data_home"])
        assert calls == [
            (learner, seed)
            for learner in v3.LEARNER_IDS
            for seed in SEEDS
        ]

        forged = copy.deepcopy(partials)
        first = (v3.LEARNER_IDS[0], SEEDS[0])
        forged[first]["measurements"]["per_task_loss"][0] = 99.0
        with pytest.raises(
            UPGDIPMNISTV3Error,
            match="recorded measurements differ from exact replay",
        ):
            REAL_REPLAY_PARTIAL_MEASUREMENTS(forged, run_spec, bundle["data_home"])

    def test_actual_tiny_jax_run_round_trips_through_partial_and_real_replay(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rng = np.random.default_rng(17)
        data_x = rng.normal(size=(16, TINY.input_dim)).astype(np.float32)
        data_y = np.arange(16, dtype=np.int32) % TINY.n_classes
        plan = _read(bundle["plan"])
        run_spec = plan["plan"]["run_spec"]
        seed = SEEDS[0]
        result = run_ipmnist_exact(
            data_x,
            data_y,
            "upgd_w",
            (seed,),
            config=TINY,
            hyperparameters=run_spec["hyperparameters"]["upgd_w"],
            progress_every=None,
        )
        partial_path = write_partial_for_result(
            bundle["plan"],
            result,
            bundle["root"] / "actual-tiny-jax-partial.json",
            process_argv=("pytest", "actual-tiny-jax"),
            started_unix=20,
            finished_unix=20 + int(round(result.wall_clock_seconds)),
        )
        plan_raw, validated_plan = v3._read_validated_plan(
            bundle["plan"],
            verify_current_bindings=True,
        )
        serialized_raw, partial = v3._read_validated_partial(
            partial_path,
            bundle["plan"],
            plan_raw,
            validated_plan,
        )
        assert serialized_raw == canonical_json_bytes(partial)
        assert stat.S_IMODE(partial_path.stat().st_mode) == 0o444
        assert partial_path.stat().st_nlink == 1
        monkeypatch.setattr(
            v3,
            "_load_pinned_mnist",
            lambda _home, *, context: (data_x, data_y),
        )
        REAL_REPLAY_PARTIAL_MEASUREMENTS(
            {("upgd_w", seed): partial},
            run_spec,
            bundle["data_home"],
        )

    def test_merge_rechecks_source_after_long_replay_independently(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        copied_dir = bundle["root"] / "merge-race-copies"
        copied_dir.mkdir()
        copied: list[Path] = []
        for index, original in enumerate(bundle["partials"]):
            path = copied_dir / f"{index:02d}.json"
            path.write_bytes(original.read_bytes())
            path.chmod(0o444)
            copied.append(path)

        expected_source = cast(
            dict[str, Any],
            copy.deepcopy(_read(bundle["plan"])["plan"]["source_import_closure"]),
        )
        drifted = False

        def source_closure() -> dict[str, Any]:
            current = copy.deepcopy(expected_source)
            if drifted:
                current["files"][0]["sha256"] = "0" * 64
            return current

        def mutate_during_replay(*_args: object) -> None:
            nonlocal drifted
            drifted = True

        monkeypatch.setattr(v3, "_build_source_import_closure", source_closure)
        monkeypatch.setattr(v3, "_replay_partial_measurements", mutate_during_replay)
        output = bundle["root"] / "merge-race-artifact.json"
        with pytest.raises(UPGDIPMNISTV3Error, match="current source"):
            merge_partials(
                bundle["plan"],
                copied,
                output,
                process_argv=("pytest", "merge-race"),
            )
        assert not output.exists()

    def test_merge_rechecks_partial_bytes_after_long_replay_independently(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        copied_dir = bundle["root"] / "merge-partial-only-race-copies"
        copied_dir.mkdir()
        copied: list[Path] = []
        for index, original in enumerate(bundle["partials"]):
            path = copied_dir / f"{index:02d}.json"
            path.write_bytes(original.read_bytes())
            path.chmod(0o444)
            copied.append(path)

        def mutate_partial(*_args: object) -> None:
            first = copied[0]
            replacement = copy.deepcopy(_read(first))
            replacement["execution"]["duration_seconds"] = 3.25
            first.rename(first.with_suffix(".original"))
            _write(first, replacement)

        monkeypatch.setattr(v3, "_replay_partial_measurements", mutate_partial)
        output = bundle["root"] / "merge-partial-only-race-artifact.json"
        with pytest.raises(UPGDIPMNISTV3Error, match="changed during merge"):
            merge_partials(
                bundle["plan"],
                copied,
                output,
                process_argv=("pytest", "merge-partial-only-race"),
            )
        assert not output.exists()

    @pytest.mark.parametrize("binding_kind", ["runtime", "data"])
    def test_merge_rechecks_runtime_and_data_independently_after_replay(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        binding_kind: str,
    ) -> None:
        drifted = False

        if binding_kind == "runtime":
            expected_runtime = cast(
                dict[str, Any],
                copy.deepcopy(_read(bundle["plan"])["plan"]["runtime_manifest"]),
            )

            def runtime_manifest() -> dict[str, Any]:
                current = copy.deepcopy(expected_runtime)
                if drifted:
                    current["python"] = "late-runtime-drift"
                return current

            monkeypatch.setattr(v3, "_build_runtime_manifest", runtime_manifest)
        else:
            real_data_manifest = v3._build_data_manifest

            def data_manifest(home: Path, archive: Path) -> dict[str, Any]:
                current = real_data_manifest(home, archive)
                if drifted:
                    current["materialized_arrays"]["x"]["sha256"] = "0" * 64
                return current

            monkeypatch.setattr(v3, "_build_data_manifest", data_manifest)

        def drift_after_replay(*_args: object) -> None:
            nonlocal drifted
            drifted = True

        monkeypatch.setattr(v3, "_replay_partial_measurements", drift_after_replay)
        output = bundle["root"] / f"merge-{binding_kind}-late-drift.json"
        with pytest.raises(UPGDIPMNISTV3Error, match=f"current {binding_kind}|dataset/cache"):
            merge_partials(
                bundle["plan"],
                bundle["partials"],
                output,
                process_argv=("pytest", f"merge-{binding_kind}-drift"),
            )
        assert not output.exists()

    def test_partial_rechecks_current_bindings_after_its_final_reread(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        partial_copy = bundle["root"] / "partial-final-binding-reread.json"
        partial_copy.write_bytes(bundle["partials"][0].read_bytes())
        partial_copy.chmod(0o444)
        expected_source = cast(
            dict[str, Any],
            copy.deepcopy(_read(bundle["plan"])["plan"]["source_import_closure"]),
        )
        drifted = False
        partial_reads = 0
        real_read = v3._read_strict_json

        def source_closure() -> dict[str, Any]:
            current = copy.deepcopy(expected_source)
            if drifted:
                current["files"][0]["sha256"] = "0" * 64
            return current

        def tracked_read(path: Path) -> tuple[bytes, dict[str, Any]]:
            nonlocal drifted, partial_reads
            result = real_read(path)
            if v3._lexical_absolute(path) == v3._lexical_absolute(partial_copy):
                partial_reads += 1
                if partial_reads == 2:
                    drifted = True
            return result

        monkeypatch.setattr(v3, "_build_source_import_closure", source_closure)
        monkeypatch.setattr(v3, "_read_strict_json", tracked_read)
        report = validate_partial(partial_copy, bundle["plan"])
        assert not report.valid
        assert any("current source" in error for error in report.errors)
        assert partial_reads == 2

    def test_artifact_rechecks_current_bindings_after_its_final_reread(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        artifact_copy = bundle["root"] / "artifact-final-binding-reread.json"
        _write(artifact_copy, _read(bundle["artifact"]))
        expected_source = cast(
            dict[str, Any],
            copy.deepcopy(_read(bundle["plan"])["plan"]["source_import_closure"]),
        )
        drifted = False
        artifact_reads = 0
        real_read = v3._read_strict_json

        def source_closure() -> dict[str, Any]:
            current = copy.deepcopy(expected_source)
            if drifted:
                current["files"][0]["sha256"] = "0" * 64
            return current

        def tracked_read(path: Path) -> tuple[bytes, dict[str, Any]]:
            nonlocal artifact_reads, drifted
            result = real_read(path)
            if v3._lexical_absolute(path) == v3._lexical_absolute(artifact_copy):
                artifact_reads += 1
                if artifact_reads == 2:
                    drifted = True
            return result

        monkeypatch.setattr(v3, "_build_source_import_closure", source_closure)
        monkeypatch.setattr(v3, "_read_strict_json", tracked_read)
        report = validate_artifact(
            artifact_copy,
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
        )
        assert not report.valid
        assert any("current source" in error for error in report.errors)
        assert artifact_reads == 2

    def test_artifact_rereads_external_plan_after_its_final_artifact_read(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = bundle["plan"]
        backup = bundle["root"] / "terminal-plan-reread.original"
        artifact_reads = 0
        real_read = v3._read_strict_json

        def tracked_read(path: Path) -> tuple[bytes, dict[str, Any]]:
            nonlocal artifact_reads
            result = real_read(path)
            if v3._lexical_absolute(path) == v3._lexical_absolute(bundle["artifact"]):
                artifact_reads += 1
                if artifact_reads == 2:
                    plan_path.rename(backup)
                    _write(plan_path, {"schema": "changed-after-final-artifact-read"})
            return result

        monkeypatch.setattr(v3, "_read_strict_json", tracked_read)
        try:
            report = validate_artifact(
                bundle["artifact"],
                partial_paths=bundle["partials"],
                plan_path=plan_path,
            )
        finally:
            if plan_path.exists():
                plan_path.unlink()
            backup.rename(plan_path)

        assert not report.valid
        assert artifact_reads == 2

    def test_artifact_validator_rereads_artifact_and_external_shards_after_replay(
        self,
        bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        artifact_copy = bundle["root"] / "artifact-race-copy.json"
        _write(artifact_copy, _read(bundle["artifact"]))
        shard_dir = bundle["root"] / "artifact-race-shards"
        shard_dir.mkdir()
        shards: list[Path] = []
        for index, original in enumerate(bundle["partials"]):
            path = shard_dir / f"{index:02d}.json"
            path.write_bytes(original.read_bytes())
            path.chmod(0o444)
            shards.append(path)

        def mutate_shard(*_args: object) -> None:
            first = shards[0]
            replacement = copy.deepcopy(_read(first))
            replacement["execution"]["duration_seconds"] = 3.25
            first.rename(first.with_suffix(".original"))
            _write(first, replacement)

        monkeypatch.setattr(v3, "_replay_partial_measurements", mutate_shard)
        shard_report = validate_artifact(
            artifact_copy,
            partial_paths=shards,
            plan_path=bundle["plan"],
        )
        assert not shard_report.valid
        assert any("changed during artifact validation" in error for error in shard_report.errors)

        artifact_copy_2 = bundle["root"] / "artifact-race-copy-2.json"
        _write(artifact_copy_2, _read(bundle["artifact"]))

        def mutate_artifact(*_args: object) -> None:
            replacement = copy.deepcopy(_read(artifact_copy_2))
            replacement["created_unix"] = 31
            artifact_copy_2.rename(artifact_copy_2.with_suffix(".original"))
            _write(artifact_copy_2, replacement)

        monkeypatch.setattr(v3, "_replay_partial_measurements", mutate_artifact)
        artifact_report = validate_artifact(
            artifact_copy_2,
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
        )
        assert not artifact_report.valid
        assert any("artifact bytes changed" in error for error in artifact_report.errors)

    def test_duplicate_shard_identity_is_rejected_before_merge(
        self, bundle: dict[str, Any]
    ) -> None:
        paths = [*bundle["partials"][:-1], bundle["partials"][0]]
        with pytest.raises(UPGDIPMNISTV3Error, match="duplicate shard identity"):
            merge_partials(
                bundle["plan"],
                paths,
                bundle["root"] / "duplicate-artifact.json",
                process_argv=("pytest", "merge-duplicate"),
            )

    def test_missing_shard_fails_exact_cartesian_coverage(self, bundle: dict[str, Any]) -> None:
        with pytest.raises(UPGDIPMNISTV3Error, match="coverage differs"):
            merge_partials(
                bundle["plan"],
                bundle["partials"][:-1],
                bundle["root"] / "missing-artifact.json",
                process_argv=("pytest", "merge-missing"),
            )

    def test_same_count_with_unplanned_identity_is_rejected(self, bundle: dict[str, Any]) -> None:
        payload = copy.deepcopy(_read(bundle["partials"][-1]))
        payload["seed_id"] = 999
        payload["execution"]["prescribed_worker_argv"][6] = "999"
        extra = _write(bundle["root"] / "unplanned-seed.json", payload)
        paths = [*bundle["partials"][:-1], extra]
        with pytest.raises(UPGDIPMNISTV3Error, match="unplanned seed"):
            merge_partials(
                bundle["plan"],
                paths,
                bundle["root"] / "extra-artifact.json",
                process_argv=("pytest", "merge-extra"),
            )

    def test_manifest_binds_exact_shard_bytes_not_locator(self, bundle: dict[str, Any]) -> None:
        changed = copy.deepcopy(_read(bundle["partials"][0]))
        changed["execution"]["duration_seconds"] = 3.25
        replacement = _write(bundle["root"] / "relocated-changed-bytes.json", changed)
        supplied = [replacement, *bundle["partials"][1:]]
        report = validate_artifact(
            bundle["artifact"],
            partial_paths=supplied,
            plan_path=bundle["plan"],
        )
        assert not report.valid
        assert any("byte hash mismatch" in error for error in report.errors), report.errors

    def test_mismatched_learner_summary_set_is_rejected(self, bundle: dict[str, Any]) -> None:
        payload = copy.deepcopy(_read(bundle["artifact"]))
        del payload["learners"]["adamw"]
        path = _write(bundle["root"] / "missing-learner-summary.json", payload)
        report = validate_artifact(
            path,
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
        )
        assert not report.valid
        assert any("summaries do not recompute" in error for error in report.errors)

    def test_artifact_rejects_noncanonical_manifest_and_merge_locators(
        self, bundle: dict[str, Any]
    ) -> None:
        manifest_payload = copy.deepcopy(_read(bundle["artifact"]))
        manifest_payload["partial_manifest"][0]["locator"] = "../relative-shard.json"
        manifest_path = _write(bundle["root"] / "relative-manifest-locator.json", manifest_payload)
        manifest_report = validate_artifact(
            manifest_path,
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
        )
        assert not manifest_report.valid
        assert any("must be absolute" in error for error in manifest_report.errors)

        merge_payload = copy.deepcopy(_read(bundle["artifact"]))
        merge_payload["merge_execution"]["prescribed_merge_argv"][2] = "../plan.json"
        merge_path = _write(bundle["root"] / "relative-merge-plan.json", merge_payload)
        merge_report = validate_artifact(
            merge_path,
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
        )
        assert not merge_report.valid
        assert any("prescribed merge argv" in error for error in merge_report.errors)

    def test_artifact_rejects_canonical_but_unbound_merge_provenance(
        self, bundle: dict[str, Any]
    ) -> None:
        wrong_plan = copy.deepcopy(_read(bundle["artifact"]))
        wrong_plan["merge_execution"]["prescribed_merge_argv"][2] = (
            bundle["root"] / "unbound-plan.json"
        ).as_posix()
        wrong_plan_path = _write(bundle["root"] / "wrong-merge-plan.json", wrong_plan)
        wrong_plan_report = validate_artifact(
            wrong_plan_path,
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
        )
        assert not wrong_plan_report.valid
        assert any("prescribed merge argv" in error for error in wrong_plan_report.errors)

        wrong_output = copy.deepcopy(_read(bundle["artifact"]))
        output_index = 5 + len(wrong_output["partial_manifest"])
        wrong_output["merge_execution"]["prescribed_merge_argv"][output_index] = (
            bundle["root"] / "unbound-artifact.json"
        ).as_posix()
        wrong_output_path = _write(
            bundle["root"] / "wrong-merge-output.json",
            wrong_output,
            bind_artifact_output=False,
        )
        wrong_output_report = validate_artifact(
            wrong_output_path,
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
        )
        assert not wrong_output_report.valid
        assert any("prescribed merge argv" in error for error in wrong_output_report.errors)

        wrong_data = copy.deepcopy(_read(bundle["artifact"]))
        claimed_home = bundle["root"] / "unbound-cache"
        claimed_archive = claimed_home / v3.MNIST_ARCHIVE_RELATIVE_PATH
        claimed_locators = {
            "data_home": claimed_home.as_posix(),
            "archive": claimed_archive.as_posix(),
        }
        wrong_data["computational_replay"]["data_locators_used"] = claimed_locators
        wrong_data["merge_execution"]["data_locators_used"] = claimed_locators
        wrong_data["merge_execution"]["prescribed_merge_argv"][-3] = claimed_home.as_posix()
        wrong_data["merge_execution"]["prescribed_merge_argv"][-1] = (
            claimed_archive.as_posix()
        )
        wrong_data_path = _write(bundle["root"] / "wrong-merge-data.json", wrong_data)
        wrong_data_report = validate_artifact(
            wrong_data_path,
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
        )
        assert not wrong_data_report.valid
        assert any(
            "computational replay receipt" in error
            or "exact effective data cache" in error
            for error in wrong_data_report.errors
        )

    def test_artifact_rejects_a_relocated_external_plan(self, bundle: dict[str, Any]) -> None:
        relocated_plan = bundle["root"] / "relocated-plan.json"
        relocated_plan.write_bytes(bundle["plan"].read_bytes())
        relocated_plan.chmod(0o444)

        report = validate_artifact(
            bundle["artifact"],
            partial_paths=bundle["partials"],
            plan_path=relocated_plan,
        )

        assert not report.valid
        assert any(
            "plan binding locator" in error or "run plan locator" in error
            for error in report.errors
        )

    def test_artifact_nested_extra_and_legacy_schemas_are_rejected(
        self, bundle: dict[str, Any]
    ) -> None:
        payload = copy.deepcopy(_read(bundle["artifact"]))
        payload["coverage"]["extra"] = False
        extra = _write(bundle["root"] / "artifact-extra-key.json", payload)
        report = validate_artifact(
            extra,
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
        )
        assert not report.valid
        assert any("coverage keys differ" in error for error in report.errors)

        for schema in ("upgd_ipmnist.artifact.v1", "alberta.upgd_ipmnist.artifact.v2"):
            legacy = _write(
                bundle["root"] / f"artifact-{schema.rsplit('.', 1)[-1]}.json",
                {"schema": schema},
            )
            assert not validate_artifact(legacy, partial_paths=()).valid

    def test_bool_integer_coercion_cannot_fake_derived_coverage(
        self, bundle: dict[str, Any]
    ) -> None:
        payload = copy.deepcopy(_read(bundle["artifact"]))
        payload["coverage"]["complete"] = 1
        path = _write(bundle["root"] / "bool-as-int.json", payload)
        report = validate_artifact(
            path,
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
        )
        assert not report.valid
        assert any("coverage differs" in error for error in report.errors)

    def test_artifact_cannot_predate_shards_or_tamper_merge_argv(
        self, bundle: dict[str, Any]
    ) -> None:
        payload = copy.deepcopy(_read(bundle["artifact"]))
        payload["created_unix"] = 21
        early = _write(bundle["root"] / "early-artifact.json", payload)
        early_report = validate_artifact(
            early,
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
        )
        assert not early_report.valid
        assert any("predates a bound shard" in error for error in early_report.errors)

        argv_payload = copy.deepcopy(_read(bundle["artifact"]))
        argv_payload["merge_execution"]["prescribed_merge_argv"][-1] = "different.json"
        argv_path = _write(bundle["root"] / "tampered-merge-argv.json", argv_payload)
        argv_report = validate_artifact(
            argv_path,
            partial_paths=bundle["partials"],
            plan_path=bundle["plan"],
        )
        assert not argv_report.valid
        assert any(
            "prescribed merge argv" in error
            or "canonical member" in error
            or "must be absolute" in error
            for error in argv_report.errors
        )


@pytest.fixture(autouse=True)
def _restore_logger_state() -> Iterator[None]:
    """Keep argparse/logging tests isolated from unrelated suite configuration."""

    yield
