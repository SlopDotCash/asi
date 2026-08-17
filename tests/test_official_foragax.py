"""Tests for the hash-attested official continual-Foragax runner.

The runner (:mod:`alberta_framework.benchmarks.official_foragax`) executes
the upstream ``continual-foragax-agents`` entry points unmodified inside a
networkless read-only OCI sandbox and treats a result as evidence only when
an atomic manifest binds the artifact bytes to a clean descriptor-pinned
source checkout, config blob, lock file, interpreter, and effective seed —
see that module's docstring for the five-stage trust/plan/harden/execute/
verify lifecycle these tests walk in order.

The attestation flow exercised here: build a synthetic git checkout and a
hash-pinned trust/endorsement descriptor, prepare a frozen plan, run a fake
(or stubbed-OCI) experiment, publish the manifest atomically, then verify —
and every deviation must fail closed: dirty or drifted sources, tampered
manifests or rewards, symlink swaps in outputs, ambiguous result layouts,
seed mismatches, and interrupted publication (which must remove both
manifest and lock).  Marked ``slow``: tests construct real git repositories,
tar archives, and SQLite result databases on disk.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import alberta_framework.benchmarks.official_foragax as official_foragax_module
from alberta_framework.benchmarks.official_foragax import (
    OFFICIAL_FORAGAX_REPOSITORY,
    OfficialForagaxBatchRun,
    OfficialForagaxBatchRunPlan,
    OfficialForagaxBatchRunRequest,
    OfficialForagaxRun,
    OfficialForagaxRunPlan,
    OfficialForagaxRunRequest,
    OfficialForagaxValidationError,
    official_foragax_batch_run_specs_from_manifest,
    official_foragax_run_spec_from_manifest,
    prepare_official_foragax_batch_run,
    prepare_official_foragax_run,
    run_official_foragax,
    run_official_foragax_batch,
    verify_official_foragax_batch_manifest,
    verify_official_foragax_manifest,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _endorse_candidate(
    monkeypatch: pytest.MonkeyPatch,
    manifest_path: Path,
) -> None:
    """Install a test-only, hash-pinned endorsement for one synthetic result."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = official_foragax_module._artifact_endorsement_identities(
        manifest
    )
    endorsement = {
        "artifact_identities_sha256": official_foragax_module._json_sha256(
            artifacts
        ),
        "artifacts": artifacts,
        "endorsement_id": "synthetic-test-endorsement",
        "executor_image_id": None,
        "manifest_kind": manifest["manifest_kind"],
        "manifest_sha256": manifest["manifest_sha256"],
        "output_tree_sha256": manifest["output_tree"]["sha256"],
        "profile_sha256": manifest["trust"]["profile_sha256"],
        "trust_descriptor_sha256": manifest["trust"]["descriptor_sha256"],
    }
    descriptor = {
        "descriptor_id": (
            official_foragax_module.OFFICIAL_FORAGAX_ENDORSEMENT_DESCRIPTOR_ID
        ),
        "endorsements": [endorsement],
        "manifest_schema_version": "1.4",
        "schema_version": "1.0",
    }
    descriptor_bytes = (
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n"
    ).encode()
    descriptor_path = (
        manifest_path.parent.parent
        / f".{manifest_path.parent.name}-test-endorsements.json"
    )
    descriptor_path.write_bytes(descriptor_bytes)
    monkeypatch.setattr(
        official_foragax_module,
        "_ENDORSEMENT_DESCRIPTOR_PATH",
        descriptor_path,
    )
    monkeypatch.setattr(
        official_foragax_module,
        "OFFICIAL_FORAGAX_ENDORSEMENT_DESCRIPTOR_SHA256",
        hashlib.sha256(descriptor_bytes).hexdigest(),
    )


def test_harness_digest_closes_over_transitive_validator_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_paths = (
        "alberta_framework/benchmarks/official_foragax.py",
        "alberta_framework/benchmarks/runtime_profile.py",
        "alberta_framework/benchmarks/forager_results.py",
    )
    assert (
        official_foragax_module._HARNESS_SOURCE_RELATIVE_PATHS
        == expected_paths
    )
    for index, relative_path in enumerate(expected_paths):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"validator-{index}\n".encode())

    with monkeypatch.context() as context:
        context.setattr(
            official_foragax_module,
            "_HARNESS_SOURCE_ROOT",
            tmp_path,
        )
        baseline = official_foragax_module._harness_sha256()
        for relative_path in expected_paths:
            path = tmp_path / relative_path
            original = path.read_bytes()
            path.write_bytes(original + b"# mutation\n")
            assert official_foragax_module._harness_sha256() != baseline
            path.write_bytes(original)
            assert official_foragax_module._harness_sha256() == baseline


def test_driver_user_library_tree_hash_is_reconstructible(
    tmp_path: Path,
) -> None:
    driver_root = tmp_path / "driver"
    driver_root.mkdir()
    libcuda = driver_root / "libcuda.so.595.71.05"
    libcuda.write_bytes(b"audited-libcuda")
    (driver_root / "libcuda.so.1").symlink_to(libcuda.name)

    first = official_foragax_module._driver_user_library_tree_identity(
        driver_root,
        libcuda_relative_path=libcuda.name,
    )
    second = official_foragax_module._driver_user_library_tree_identity(
        driver_root,
        libcuda_relative_path=libcuda.name,
    )
    assert first == second
    assert first[1] == hashlib.sha256(b"audited-libcuda").hexdigest()

    libcuda.write_bytes(b"mutated-libcuda")
    mutated = official_foragax_module._driver_user_library_tree_identity(
        driver_root,
        libcuda_relative_path=libcuda.name,
    )
    assert mutated != first

    mismatched = driver_root / "libnvidia-gpucomp.so.595.84"
    mismatched.write_bytes(b"mismatched-driver")
    with pytest.raises(
        OfficialForagaxValidationError,
        match="version differs from the kernel contract",
    ):
        official_foragax_module._driver_user_library_tree_identity(
            driver_root,
            libcuda_relative_path=libcuda.name,
            expected_driver_version="595.71.05",
        )
    mismatched.unlink()

    (driver_root / "escape").symlink_to("../outside")
    with pytest.raises(
        OfficialForagaxValidationError,
        match="escapes its bundle",
    ):
        official_foragax_module._driver_user_library_tree_identity(
            driver_root,
            libcuda_relative_path=libcuda.name,
        )
    with pytest.raises(
        OfficialForagaxValidationError,
        match="root-owned mode 0555",
    ):
        official_foragax_module._driver_user_library_tree_identity(
            driver_root,
            libcuda_relative_path=libcuda.name,
            require_root_owned_read_only=True,
            expected_driver_version="595.71.05",
        )


def test_strict_json_and_archive_contracts_reject_lossy_inputs(
    tmp_path: Path,
) -> None:
    official_foragax_module._require_empty_oci_host_stderr(
        subprocess.CompletedProcess(("runtime",), 0, b"", b""),
        operation="test",
    )
    with pytest.raises(
        OfficialForagaxValidationError,
        match="unframed host stderr",
    ):
        official_foragax_module._require_empty_oci_host_stderr(
            subprocess.CompletedProcess(
                ("runtime",),
                0,
                b"",
                b"host warning\n",
            ),
            operation="test",
        )

    canonical_packages = [
        "continual-foragax==0.55.0",
        "jax==0.9.0.1",
        "jax-cuda12-pjrt==0.9.0.1",
        "jax-cuda12-plugin==0.9.0.1",
        "jaxlib==0.9.0.1",
    ]
    official_foragax_module._validate_oci_scientific_package_inventory(
        canonical_packages,
        executor={"kind": "oci"},
    )
    with pytest.raises(
        OfficialForagaxValidationError,
        match="canonical Foragax 0.55/JAX 0.9.0.1 CUDA 12 lock",
    ):
        official_foragax_module._validate_oci_scientific_package_inventory(
            [
                *canonical_packages,
                "jax-cuda13-plugin==0.9.0.1",
            ],
            executor={"kind": "oci"},
        )

    minimal_executor = {
        "cpu_runtime_arguments": [
            "--env=JAX_PLATFORM_NAME=cpu",
            "--env=JAX_PLATFORMS=cpu",
            "--env=JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1",
        ],
        "image_id": "sha256:" + "1" * 64,
        "launcher_path": "/opt/alberta/launcher",
        "python_executable": "/usr/bin/python3.12",
        "source_root": "/opt/continual-foragax-agents",
    }
    base_command = official_foragax_module._oci_base_command(
        runtime=Path("/usr/bin/docker"),
        executor=minimal_executor,
        gpu=False,
    )
    assert (
        "--mount=type=tmpfs,destination=/tmp/src,"
        "tmpfs-mode=0555,tmpfs-size=1048576"
    ) in base_command
    assert not any(
        argument.startswith("--tmpfs=/tmp:") for argument in base_command
    )
    assert (
        "--tmpfs=/run/alberta/tmp:"
        "rw,noexec,nosuid,nodev,size=1g,uid=65532,gid=65532,mode=0700"
    ) in base_command
    assert "--env=HOME=/run/alberta/home" in base_command
    assert "--env=MPLCONFIGDIR=/run/alberta/matplotlib" in base_command
    assert "--env=NVIDIA_VISIBLE_DEVICES=void" in base_command
    assert "--env=TMPDIR=/run/alberta/tmp" in base_command
    assert "--env=XDG_CACHE_HOME=/run/alberta/cache" in base_command
    assert "--env=JAX_ENABLE_COMPILATION_CACHE=false" in base_command
    assert "--env=PYTHONHASHSEED=0" in base_command
    assert "--env=PYTHONHOME=" in base_command
    assert "--env=PYTHONPATH=" in base_command
    assert "--workdir=/opt/continual-foragax-agents" in base_command
    launcher_command = official_foragax_module._oci_official_command(
        runtime=Path("/usr/bin/docker"),
        executor=minimal_executor,
        gpu=False,
        entrypoint="src/continuing_main.py",
        config_path="/opt/config.json",
        index_expression="0",
        max_steps_argument=10,
    )
    assert "--python-flag=-I" in launcher_command
    assert "--python-flag=-B" in launcher_command
    assert "--trusted-python-path" in launcher_command
    assert "/opt/continual-foragax-agents/src" in launcher_command
    assert (
        "--trusted-python-path-mode=isolated-runpy-prepend-v1"
        in launcher_command
    )

    gpu_executor = {
        **minimal_executor,
        "gpu_runtime_arguments": [
            (
                "--mount=type=bind,"
                "source=/tmp/nvidia-driver,"
                "destination=/opt/nvidia-driver,readonly"
            ),
            "--device=/dev/nvidia0",
            "--device=/dev/nvidiactl",
            "--device=/dev/nvidia-uvm",
            "--env=CUDA_VISIBLE_DEVICES=0",
            "--env=CUBLAS_WORKSPACE_CONFIG=:4096:8",
            (
                "--env=LD_LIBRARY_PATH="
                "/opt/cuda-wheels:/opt/nvidia-driver"
            ),
            (
                "--env=XLA_FLAGS=--xla_gpu_enable_triton_gemm=false "
                "--xla_gpu_deterministic_ops=true"
            ),
            "--env=XLA_PYTHON_CLIENT_PREALLOCATE=false",
        ],
    }
    gpu_command = official_foragax_module._oci_base_command(
        runtime=Path("/usr/bin/docker"),
        executor=gpu_executor,
        gpu=True,
    )
    assert not any(argument == "--gpus" for argument in gpu_command)
    assert "--device=/dev/nvidia0" in gpu_command
    assert (
        "--mount=type=bind,source=/tmp/nvidia-driver,"
        "destination=/opt/nvidia-driver,readonly"
    ) in gpu_command
    assert "--env=CUBLAS_WORKSPACE_CONFIG=:4096:8" in gpu_command
    assert "--env=XLA_PYTHON_CLIENT_PREALLOCATE=false" in gpu_command

    with pytest.raises(
        OfficialForagaxValidationError,
        match="duplicate object key",
    ):
        official_foragax_module._strict_json_loads(
            '{"seed":1,"seed":2}',
            label="test JSON",
        )
    with pytest.raises(
        OfficialForagaxValidationError,
        match="non-finite JSON constant",
    ):
        official_foragax_module._strict_json_loads(
            '{"reward":NaN}',
            label="test JSON",
        )

    redacted = official_foragax_module._sanitize_package_freeze_line(
        'pkg==1.0 ; direct_url={"url":"file:/tmp/local-wheel"}'
    )
    assert redacted == (
        'pkg==1.0 ; direct_url={"url":"<LOCAL_PATH>"}'
    )
    malformed = official_foragax_module._sanitize_package_freeze_line(
        "pkg==1.0 ; direct_url={not-json"
    )
    assert malformed.endswith(" ; direct_url=<REDACTED>")
    with pytest.raises(
        OfficialForagaxValidationError,
        match="package freeze direct_url is not finite JSON",
    ):
        official_foragax_module._sanitize_package_freeze_line(
            'pkg==1.0 ; direct_url={"url":NaN}'
        )
    with pytest.raises(
        OfficialForagaxValidationError,
        match="package freeze direct_url is not finite JSON",
    ):
        official_foragax_module._sanitize_package_freeze_line(
            'pkg==1.0 ; direct_url={"url":Infinity}'
        )
    freeze_source = official_foragax_module._package_freeze.__code__.co_consts
    assert any(
        isinstance(constant, str) and "allow_nan=False" in constant
        for constant in freeze_source
    )

    contracts = [
        {
            "name": "rewards",
            "dtype": "float16",
            "shape_tail": [],
            "semantic_role": "trusted_metric_payload",
            "finite_policy": "all_finite",
        },
        {
            "name": "churn_norm",
            "dtype": "float16",
            "shape_tail": [],
            "semantic_role": "diagnostic",
            "finite_policy": "allow_nonfinite",
        },
    ]
    valid = tmp_path / "valid.npz"
    np.savez_compressed(
        valid,
        rewards=np.ones((4,), dtype=np.float16),
        churn_norm=np.full((4,), np.nan, dtype=np.float16),
    )
    metadata = official_foragax_module._inspect_npz(
        tmp_path,
        valid.name,
        expected_steps=4,
        expected_members=contracts,
    )
    assert metadata["validated_consumed_arrays"] == {
        "rewards": {
            "all_finite": True,
            "dtype": "float16",
            "real_numeric": True,
            "shape": [4],
        }
    }

    malformed_arrays = {
        "squeezed": np.ones((1, 4), dtype=np.float16),
        "boolean": np.ones((4,), dtype=np.bool_),
        "complex": np.ones((4,), dtype=np.complex64),
        "nonfinite": np.full((4,), np.nan, dtype=np.float16),
    }
    for name, rewards in malformed_arrays.items():
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(
            path,
            rewards=rewards,
            churn_norm=np.full((4,), np.nan, dtype=np.float16),
        )
        with pytest.raises(
            OfficialForagaxValidationError,
            match="shape|real numeric|dtype|non-finite",
        ):
            official_foragax_module._inspect_npz(
                tmp_path,
                path.name,
                expected_steps=4,
                expected_members=contracts,
            )


def _foragax_results_database_bytes(
    path: Path,
    *,
    indices: tuple[int, ...],
) -> bytes:
    connection = sqlite3.connect(path)
    try:
        columns = ", ".join(
            f'"{name}" INTEGER'
            for name in official_foragax_module.OFFICIAL_FORAGAX_RESULTS_DB_COLUMNS
        )
        connection.execute(f'CREATE TABLE "_metadata_" ({columns})')
        connection.executemany(
            'INSERT INTO "_metadata_" (seed, id) VALUES (?, ?)',
            [(index, index) for index in indices],
        )
        connection.commit()
    finally:
        connection.close()
    return path.read_bytes()


def _oci_result_layout_invocation(
    *,
    indices: tuple[int, ...],
    result_paths: tuple[str, ...],
    database_paths: tuple[str, ...],
    extra_members: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    return {
        "indices": list(indices),
        "members": [
            *(
                {
                    "content_policy": "strict_npz",
                    "max_bytes": 1024,
                    "path": path,
                    "role": "result_npz",
                }
                for path in result_paths
            ),
            *(
                {
                    "content_policy": "sqlite_foragax_metadata_v1",
                    "max_bytes": 1024,
                    "path": path,
                    "role": "auxiliary",
                }
                for path in database_paths
            ),
            *extra_members,
            {
                "content_policy": "bounded_utf8_log",
                "max_bytes": 1024,
                "path": "stdout.log",
                "role": "stdout_log",
            },
            {
                "content_policy": "bounded_utf8_diagnostic",
                "max_bytes": 1024,
                "path": "stderr.log",
                "role": "stderr_log",
            },
        ],
    }


@pytest.mark.parametrize(
    ("root", "indices"),
    (
        ("official-results/result", (7,)),
        (
            "official-results/results/E138-two-biome-large/foragax/"
            "ForagaxTwoBiomeLarge-v1/9/DQN",
            (2_000_001, 2_000_002),
        ),
    ),
)
def test_descriptor_result_layout_derives_one_sibling_database(
    root: str,
    indices: tuple[int, ...],
) -> None:
    invocation = _oci_result_layout_invocation(
        indices=indices,
        result_paths=tuple(f"{root}/data/{index}.npz" for index in indices),
        database_paths=(f"{root}/results.db",),
    )

    layout = official_foragax_module._descriptor_result_layout(
        invocation,
        label="test invocation",
    )

    assert layout.experiment_root == root
    assert layout.result_paths == tuple(
        f"{root}/data/{index}.npz" for index in indices
    )
    assert layout.database_path == f"{root}/results.db"


def _qualification_trust_binding_fixture() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    seed = 2_000_001
    root = (
        "official-results/results/E138-two-biome-large/foragax/"
        "ForagaxTwoBiomeLarge-v1/9/DQN"
    )
    invocation = _oci_result_layout_invocation(
        indices=(seed,),
        result_paths=(f"{root}/data/{seed}.npz",),
        database_paths=(f"{root}/results.db",),
    )
    invocation.update(
        {
            "expected_result_env_steps": 1,
            "index_expression": str(seed),
            "max_steps_argument": 1,
            "max_total_bytes": 16 * 1024,
        }
    )
    run = {
        "effective_seed": seed,
        "index": seed,
        "nested_seed_offset": 0,
        "stored_seed": seed,
        "top_level_seed_offset": 0,
    }
    configuration = {
        "agent": "DQN",
        "config_sha256": "2" * 64,
        "container_config_path": (
            "/opt/continual-foragax-agents/experiments/"
            "E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/9/"
            "DQN.json"
        ),
        "entrypoint_family": "continuing",
        "invocations": [invocation],
        "problem": "Foragax",
        "runs": [run],
    }
    entrypoints = {
        "continuing": {
            "path": "src/continuing_main.py",
            "sha256": "6" * 64,
        },
        "ppo": {
            "path": "src/rtu_ppo.py",
            "sha256": "7" * 64,
        },
    }
    executor = {
        "cpu_runtime_arguments": [
            "--env=JAX_PLATFORM_NAME=cpu",
            "--env=JAX_PLATFORMS=cpu",
            "--env=JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1",
        ],
        "environment_profile_sha256": "5" * 64,
        "gpu_runtime_arguments": ["--device=/dev/nvidia0"],
        "launcher_contract": "oci-read-only-stdout-tar-v4",
        "source_archive_sha256": "3" * 64,
    }
    projection = official_foragax_module._qualification_workload_projection(
        executor=executor,
        entrypoints=entrypoints,
        configuration=configuration,
        run=run,
        invocation=invocation,
        backend="cpu",
    )
    executor["determinism_qualification"] = {
        "backend": "cpu",
        "config_sha256": configuration["config_sha256"],
        "effective_seed": seed,
        "environment_profile_sha256": executor[
            "environment_profile_sha256"
        ],
        "source_archive_sha256": executor["source_archive_sha256"],
        "steps": 1,
        "workload_identity_sha256": (
            official_foragax_module._json_sha256(projection)
        ),
    }
    return executor, entrypoints, configuration


def test_qualification_trust_binding_is_unique_and_fail_closed() -> None:
    executor, entrypoints, configuration = (
        _qualification_trust_binding_fixture()
    )
    official_foragax_module._validate_qualification_trust_binding(
        executor=executor,
        entrypoints=entrypoints,
        configurations=[configuration],
        label="test profile",
    )

    for mutation in (
        "workload_hash",
        "source_hash",
        "environment_hash",
        "config_hash",
        "seed",
        "horizon",
        "result_root",
        "member_order",
        "backend_arguments",
        "entrypoint_hash",
    ):
        changed_executor = copy.deepcopy(executor)
        changed_entrypoints = copy.deepcopy(entrypoints)
        changed_configuration = copy.deepcopy(configuration)
        determinism = changed_executor["determinism_qualification"]
        invocation = changed_configuration["invocations"][0]
        if mutation == "workload_hash":
            determinism["workload_identity_sha256"] = "0" * 64
        elif mutation == "source_hash":
            changed_executor["source_archive_sha256"] = "0" * 64
        elif mutation == "environment_hash":
            changed_executor["environment_profile_sha256"] = "0" * 64
        elif mutation == "config_hash":
            determinism["config_sha256"] = "0" * 64
        elif mutation == "seed":
            determinism["effective_seed"] += 1
        elif mutation == "horizon":
            determinism["steps"] += 1
        elif mutation == "result_root":
            invocation["members"][0]["path"] = (
                "official-results/other/data/2000001.npz"
            )
        elif mutation == "member_order":
            invocation["members"][0], invocation["members"][1] = (
                invocation["members"][1],
                invocation["members"][0],
            )
        elif mutation == "backend_arguments":
            changed_executor["cpu_runtime_arguments"].append(
                "--env=UNTRUSTED=1"
            )
        else:
            changed_entrypoints["continuing"]["sha256"] = "0" * 64
        with pytest.raises(OfficialForagaxValidationError):
            official_foragax_module._validate_qualification_trust_binding(
                executor=changed_executor,
                entrypoints=changed_entrypoints,
                configurations=[changed_configuration],
                label=f"test profile {mutation}",
            )


def test_trust_configuration_accepts_descriptor_bound_nested_e138_layout() -> None:
    indices = (2_000_001, 2_000_002)
    root = (
        "official-results/results/E138-two-biome-large/foragax/"
        "ForagaxTwoBiomeLarge-v1/9/DQN"
    )
    invocation = _oci_result_layout_invocation(
        indices=indices,
        result_paths=tuple(f"{root}/data/{index}.npz" for index in indices),
        database_paths=(f"{root}/results.db",),
    )
    invocation.update(
        {
            "expected_result_env_steps": 10_000,
            "index_expression": "2000001:2000003",
            "max_steps_argument": 10_000,
            "max_total_bytes": 5 * 1024,
        }
    )
    archive_members = [
        dict(member)
        for member in official_foragax_module._TEST_NATIVE_ARCHIVE_ARRAY_CONTRACTS
    ]
    runs = [
        {
            "agent_access_sha256": "a" * 64,
            "archive_members": archive_members,
            "effective_configuration_sha256": "b" * 64,
            "effective_seed": index,
            "environment_sha256": "c" * 64,
            "environment_rng_schedule": "shared_agent_environment_rng_v1",
            "index": index,
            "jax_key_sha256": "d" * 64,
            "nested_seed_offset": 0,
            "registry_sha256": "e" * 64,
            "resolved_hyperparameters_sha256": "f" * 64,
            "stored_seed": index,
            "top_level_seed_offset": 0,
        }
        for index in indices
    ]
    configuration = {
        "agent": "DQN",
        "config_commit": "1" * 40,
        "config_git_blob_sha1": "2" * 40,
        "config_lock_git_blob_sha1": "3" * 40,
        "config_lock_sha256": "4" * 64,
        "config_path": "experiments/E138-two-biome-large/DQN.json",
        "config_sha256": "5" * 64,
        "container_config_path": (
            "/opt/continual-foragax-agents/experiments/"
            "E138-two-biome-large/DQN.json"
        ),
        "entrypoint_family": "continuing",
        "invocations": [invocation],
        "problem": "Foragax",
        "runs": runs,
        "scientific_track": "matched_current_environment_comparator",
    }

    validated = official_foragax_module._validate_trust_configuration(
        configuration,
        profile_label="test OCI profile",
        executor_kind="oci",
    )

    assert validated is configuration


@pytest.mark.parametrize(
    "invocation",
    (
        _oci_result_layout_invocation(
            indices=(2, 3),
            result_paths=(
                "official-results/result/data/3.npz",
                "official-results/result/data/2.npz",
            ),
            database_paths=("official-results/result/results.db",),
        ),
        _oci_result_layout_invocation(
            indices=(2, 2),
            result_paths=(
                "official-results/result/data/2.npz",
                "official-results/result/data/2.npz",
            ),
            database_paths=("official-results/result/results.db",),
        ),
        _oci_result_layout_invocation(
            indices=(3, 2),
            result_paths=(
                "official-results/result/data/3.npz",
                "official-results/result/data/2.npz",
            ),
            database_paths=("official-results/result/results.db",),
        ),
        _oci_result_layout_invocation(
            indices=(2, 3),
            result_paths=(
                "official-results/first/data/2.npz",
                "official-results/second/data/3.npz",
            ),
            database_paths=("official-results/first/results.db",),
        ),
        _oci_result_layout_invocation(
            indices=(2,),
            result_paths=("official-results/result/data/2.npz",),
            database_paths=("official-results/other/results.db",),
        ),
        _oci_result_layout_invocation(
            indices=(2,),
            result_paths=("official-results/result/data/2.npz",),
            database_paths=("official-results/result/results.db",),
            extra_members=(
                {
                    "content_policy": "opaque_bound",
                    "max_bytes": 1024,
                    "path": "official-results/result/data/extra.npz",
                    "role": "auxiliary",
                },
            ),
        ),
        _oci_result_layout_invocation(
            indices=(2,),
            result_paths=("official-results/result/data/2.npz",),
            database_paths=(
                "official-results/result/results.db",
                "official-results/other/results.db",
            ),
        ),
        _oci_result_layout_invocation(
            indices=(2,),
            result_paths=(
                "official-results/result/data/../data/2.npz",
            ),
            database_paths=("official-results/result/results.db",),
        ),
        _oci_result_layout_invocation(
            indices=(2,),
            result_paths=(
                "official-results/result/data/../../escape/2.npz",
            ),
            database_paths=("official-results/result/results.db",),
        ),
    ),
)
def test_descriptor_result_layout_rejects_ambiguous_or_aliased_paths(
    invocation: dict[str, Any],
) -> None:
    with pytest.raises(OfficialForagaxValidationError):
        official_foragax_module._descriptor_result_layout(
            invocation,
            label="test invocation",
        )


def test_results_database_and_ustar_boundaries_fail_closed(
    tmp_path: Path,
) -> None:
    database_bytes = _foragax_results_database_bytes(
        tmp_path / "results.db",
        indices=(2, 3),
    )
    official_foragax_module._validate_foragax_results_database_bytes(
        database_bytes,
        expected_indices=(2, 3),
    )
    nested_database_path = (
        "official-results/results/E138-two-biome-large/foragax/"
        "ForagaxTwoBiomeLarge-v1/9/DQN/results.db"
    )
    nested_database = tmp_path / nested_database_path
    nested_database.parent.mkdir(parents=True)
    nested_database.write_bytes(database_bytes)
    official_foragax_module._verify_foragax_results_database(
        tmp_path,
        database_path=nested_database_path,
        expected_indices=(2, 3),
    )
    with pytest.raises(
        OfficialForagaxValidationError,
        match="must be named results.db",
    ):
        official_foragax_module._verify_foragax_results_database(
            tmp_path,
            database_path=nested_database_path.removesuffix("results.db")
            + "metadata.db",
            expected_indices=(2, 3),
        )
    with pytest.raises(
        OfficialForagaxValidationError,
        match="invocation index set",
    ):
        official_foragax_module._validate_foragax_results_database_bytes(
            database_bytes,
            expected_indices=(2, 4),
        )
    with pytest.raises(
        OfficialForagaxValidationError,
        match="valid trusted SQLite database",
    ):
        official_foragax_module._validate_foragax_results_database_bytes(
            b"not SQLite",
            expected_indices=(2, 3),
        )

    root = tmp_path / "tar-output"
    root.mkdir()
    archive_path = tmp_path / "output.tar"
    payload = b"bounded UTF-8 log\n"
    stream = io.BytesIO()
    with tarfile.open(
        fileobj=stream,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        member = tarfile.TarInfo("logs/stdout.log")
        member.size = len(payload)
        member.mode = 0o600
        member.uid = 0
        member.gid = 0
        member.mtime = 0
        archive.addfile(member, io.BytesIO(payload))
    archive_path.write_bytes(stream.getvalue())
    invocation = {
        "indices": [2],
        "max_total_bytes": 128,
        "members": [
            {
                "content_policy": "bounded_utf8_log",
                "max_bytes": 64,
                "path": "logs/stdout.log",
                "role": "stdout",
            }
        ],
    }
    root_descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    archive_descriptor = os.open(archive_path, os.O_RDONLY)
    try:
        official_foragax_module._extract_trusted_oci_tar_at(
            root_descriptor=root_descriptor,
            archive_descriptor=archive_descriptor,
            invocation=invocation,
        )
    finally:
        os.close(archive_descriptor)
        os.close(root_descriptor)
    assert (root / "logs/stdout.log").read_bytes() == payload

    malicious_stream = io.BytesIO()
    with tarfile.open(
        fileobj=malicious_stream,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        member = tarfile.TarInfo("logs/stdout.log")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        member.mode = 0o600
        member.uid = 0
        member.gid = 0
        member.mtime = 0
        archive.addfile(member)
    malicious_path = tmp_path / "malicious.tar"
    malicious_path.write_bytes(malicious_stream.getvalue())
    malicious_root = tmp_path / "malicious-output"
    malicious_root.mkdir()
    root_descriptor = os.open(
        malicious_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    archive_descriptor = os.open(malicious_path, os.O_RDONLY)
    try:
        with pytest.raises(
            OfficialForagaxValidationError,
            match="member metadata is unsafe",
        ):
            official_foragax_module._extract_trusted_oci_tar_at(
                root_descriptor=root_descriptor,
                archive_descriptor=archive_descriptor,
                invocation=invocation,
            )
    finally:
        os.close(archive_descriptor)
        os.close(root_descriptor)
    assert not (tmp_path / "outside").exists()


@pytest.fixture
def official_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str, str]:
    """Build a tiny repository with the same runner/config contracts."""
    repository = tmp_path / "continual-foragax-agents"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Foragax Test")
    _git(repository, "config", "user.email", "foragax@example.invalid")
    _git(repository, "remote", "add", "origin", f"{OFFICIAL_FORAGAX_REPOSITORY}.git")

    _write(repository / "uv.lock", "version = 1\n")
    _write(repository / "config.json", '{"save_path": "results/{name}/{agent}"}\n')
    _write(repository / "src/experiment/__init__.py", "")
    _write(repository / "src/algorithms/__init__.py", "")
    _write(repository / "src/algorithms/nn/__init__.py", "")
    _write(
        repository / "src/algorithms/nn/DQN.py",
        "class DQN:\n    pass\n",
    )
    _write(
        repository / "src/algorithms/nn/RealTimeACConv.py",
        "class RealTimeActorCriticConv:\n    pass\n",
    )
    _write(
        repository / "src/algorithms/SearchAgent.py",
        "class SearchAgent:\n    pass\n",
    )
    _write(
        repository / "src/algorithms/MCTSAgent.py",
        "class MCTSAgent:\n    pass\n",
    )
    _write(
        repository / "src/algorithms/RandomAgent.py",
        "class RandomAgent:\n    pass\n",
    )
    _write(
        repository / "src/algorithms/DebugAgent.py",
        "class DebugAgent:\n    pass\n",
    )
    _write(
        repository / "src/algorithms/registry.py",
        """
from algorithms.DebugAgent import DebugAgent
from algorithms.MCTSAgent import MCTSAgent
from algorithms.RandomAgent import RandomAgent
from algorithms.SearchAgent import SearchAgent
from algorithms.nn.DQN import DQN


def getAgent(name):
    if name.startswith("DQN"):
        return DQN
    if name.startswith("Search"):
        return SearchAgent
    if name.startswith("MCTS"):
        return MCTSAgent
    if name.startswith("Random"):
        return RandomAgent
    return DebugAgent
""".lstrip(),
    )
    _write(
        repository / "src/algorithms/PPORegistry.py",
        """
from algorithms.nn.RealTimeACConv import RealTimeActorCriticConv


def getAgent(name):
    del name
    return RealTimeActorCriticConv
""".lstrip(),
    )
    _write(
        repository / "src/experiment/ExperimentModel.py",
        """
import json


class _Experiment:
    def __init__(self, data):
        self.data = data

    def get_hypers(self, index):
        del index
        return self.data["metaParameters"]

    def getRun(self, index):
        return index

    def numPermutations(self):
        return 1


def load(path):
    with open(path, encoding="utf-8") as handle:
        return _Experiment(json.load(handle))
""".lstrip(),
    )
    runner = """
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-e", required=True)
parser.add_argument("-i", required=True)
parser.add_argument("--save_path", required=True)
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--silent", action="store_true")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--max_steps", type=int)
args = parser.parse_args()
if ":" in args.i:
    start, stop = (int(value) for value in args.i.split(":"))
    indices = list(range(start, stop))
else:
    indices = [int(args.i)]
with open(args.e, encoding="utf-8") as handle:
    config = json.load(handle)
meta = config["metaParameters"]
if __RUNNER_FAMILY__ == "ppo":
    rollout = int(meta["rollout_steps"])
    updates = (
        args.max_steps
        if args.max_steps is not None
        else config["total_steps"] // rollout + 1
    )
    steps = updates * rollout
else:
    steps = args.max_steps if args.max_steps is not None else config["total_steps"]
rewards = np.arange(steps, dtype=np.float32)
if meta.get("emit_nonfinite"):
    rewards[-1] = np.nan
for index in indices:
    destination = Path(args.save_path) / "result" / "data" / f"{index}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, rewards=rewards)
    print(f"wrote {destination}")
database = Path(args.save_path) / "result" / "results.db"
database.write_bytes(b"synthetic sqlite payload")
checkpoint = Path(args.checkpoint_path) / "state" / "checkpoint.bin"
checkpoint.parent.mkdir(parents=True, exist_ok=True)
checkpoint.write_bytes(b"synthetic checkpoint payload")
"""
    _write(
        repository / "src/continuing_main.py",
        runner.replace("__RUNNER_FAMILY__", repr("continuing")).lstrip(),
    )
    _write(
        repository / "src/rtu_ppo.py",
        runner.replace("__RUNNER_FAMILY__", repr("ppo")).lstrip(),
    )
    _write(
        repository / "configs/dqn.json",
        json.dumps(
            {
                "agent": "DQN",
                "problem": "Foragax",
                "total_steps": 100,
                "metaParameters": {
                    "experiment": {"seed_offset": 7},
                    "environment": {"env_id": "ForagaxSquareWaveTwoBiome-v11"},
                },
            },
            indent=2,
        )
        + "\n",
    )
    _write(
        repository / "configs/dqn_zero.json",
        json.dumps(
            {
                "agent": "DQN",
                "problem": "Foragax",
                "total_steps": 100,
                "metaParameters": {
                    "experiment": {"seed_offset": 0},
                    "environment": {
                        "env_id": "ForagaxSquareWaveTwoBiome-v11"
                    },
                },
            },
            indent=2,
        )
        + "\n",
    )
    _write(
        repository / "configs/dqn_nonfinite.json",
        json.dumps(
            {
                "agent": "DQN",
                "problem": "Foragax",
                "total_steps": 100,
                "metaParameters": {
                    "emit_nonfinite": True,
                    "experiment": {"seed_offset": 7},
                    "environment": {
                        "env_id": "ForagaxSquareWaveTwoBiome-v11"
                    },
                },
            },
            indent=2,
        )
        + "\n",
    )
    _write(
        repository / "configs/ppo.json",
        json.dumps(
            {
                "agent": "PPO-RTU",
                "problem": "Foragax",
                "total_steps": 10,
                "metaParameters": {
                    "seed_offset": 0,
                    "rollout_steps": 4,
                    "environment": {"env_id": "ForagaxBig-v5"},
                },
            },
            indent=2,
        )
        + "\n",
    )
    classification_configs = {
        "dqn_fov9.json": {
            "agent": "DQN",
            "environment": {
                "env_id": "ForagaxTwoBiomeLarge-v1",
                "aperture_size": 9,
                "observation_type": "color",
            },
        },
        "dqn_world.json": {
            "agent": "DQN",
            "mode": "world",
            "environment": {
                "env_id": "ForagaxTwoBiomeLarge-v1",
                "aperture_size": -1,
                "observation_type": "object",
            },
        },
        "search_nearest.json": {
            "agent": "Search-Nearest",
            "mode": "world",
            "reward_prioritization": False,
            "environment": {
                "env_id": "ForagaxTwoBiomeLarge-v1",
                "aperture_size": -1,
                "observation_type": "object",
            },
        },
        "search_oracle.json": {
            "agent": "Search-Oracle",
            "mode": "world",
            "reward_prioritization": True,
            "channel_priorities": {"apple": 1.0},
            "environment": {
                "env_id": "ForagaxTwoBiomeLarge-v1",
                "aperture_size": -1,
                "observation_type": "object",
            },
        },
        "dqn_reward_grid.json": {
            "agent": "DQN",
            "reward_prioritization": True,
            "environment": {
                "env_id": "ForagaxTwoBiomeLarge-v1",
                "aperture_size": 9,
                "observation_type": "color",
            },
        },
        "dqn_temperature.json": {
            "agent": "DQN",
            "temperature_prioritization": True,
            "environment": {
                "env_id": "ForagaxTwoBiomeLarge-v1",
                "aperture_size": 9,
                "observation_type": "color",
            },
        },
        "mcts.json": {
            "agent": "MCTS",
            "environment": {
                "env_id": "ForagaxTwoBiomeLarge-v1",
                "aperture_size": 9,
                "observation_type": "color",
            },
        },
        "unknown.json": {
            "agent": "Unknown-Debug",
            "environment": {
                "env_id": "ForagaxTwoBiomeLarge-v1",
                "aperture_size": 9,
                "observation_type": "color",
            },
        },
    }
    for name, values in classification_configs.items():
        agent = values["agent"]
        meta_parameters = {
            key: value for key, value in values.items() if key != "agent"
        }
        _write(
            repository / "configs" / name,
            json.dumps(
                {
                    "agent": agent,
                    "problem": "Foragax",
                    "total_steps": 5,
                    "metaParameters": meta_parameters,
                },
                indent=2,
            )
            + "\n",
        )
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "paper configs")
    config_commit = _git(repository, "rev-parse", "HEAD")

    with (repository / "src/continuing_main.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# corrected execution source\n")
    _write(repository / "uv.lock", "version = 2\n")
    _git(repository, "add", "src/continuing_main.py", "uv.lock")
    _git(repository, "commit", "-qm", "correct runner")
    execution_commit = _git(repository, "rev-parse", "HEAD")
    historical_lock = subprocess.run(
        ("git", "show", f"{config_commit}:uv.lock"),
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    configurations = []
    for config_path in sorted((repository / "configs").glob("*.json")):
        relative = config_path.relative_to(repository).as_posix()
        config_bytes = subprocess.run(
            ("git", "show", f"{config_commit}:{relative}"),
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        config = json.loads(config_bytes)
        configurations.append(
            {
                "agent": config["agent"],
                "config_commit": config_commit,
                "config_git_blob_sha1": _git(
                    repository,
                    "rev-parse",
                    f"{config_commit}:{relative}",
                ),
                "config_lock_git_blob_sha1": _git(
                    repository,
                    "rev-parse",
                    f"{config_commit}:uv.lock",
                ),
                "config_lock_sha256": hashlib.sha256(
                    historical_lock
                ).hexdigest(),
                "config_path": relative,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "container_config_path": "test/config.snapshot.json",
                "entrypoint_family": (
                    "ppo"
                    if str(config["agent"]).startswith(
                        ("PPO", "RealTimeActorCritic", "ActorCritic")
                    )
                    else "continuing"
                ),
                "invocations": [],
                "problem": config["problem"],
                "runs": [],
                "scientific_track": "synthetic_test",
            }
        )
    descriptor = {
        "descriptor_id": official_foragax_module.OFFICIAL_FORAGAX_TRUST_DESCRIPTOR_ID,
        "manifest_schema_version": "1.4",
        "profiles": [
            {
                "configurations": configurations,
                "entrypoints": {
                    family: {
                        "path": path,
                        "sha256": hashlib.sha256(
                            (repository / path).read_bytes()
                        ).hexdigest(),
                    }
                    for family, path in {
                        "continuing": "src/continuing_main.py",
                        "ppo": "src/rtu_ppo.py",
                    }.items()
                },
                "execution_commit": execution_commit,
                "execution_config_git_blob_sha1": _git(
                    repository,
                    "rev-parse",
                    "HEAD:config.json",
                ),
                "execution_config_sha256": hashlib.sha256(
                    (repository / "config.json").read_bytes()
                ).hexdigest(),
                "execution_lock_git_blob_sha1": _git(
                    repository,
                    "rev-parse",
                    "HEAD:uv.lock",
                ),
                "execution_lock_sha256": hashlib.sha256(
                    (repository / "uv.lock").read_bytes()
                ).hexdigest(),
                "execution_tree_git_sha1": _git(
                    repository,
                    "rev-parse",
                    "HEAD^{tree}",
                ),
                "executor": {
                    "interpreter_sha256": hashlib.sha256(
                        Path(sys.executable).resolve().read_bytes()
                    ).hexdigest(),
                    "kind": "test-native",
                },
                "profile_id": "synthetic-test-profile",
                "source_tree_sha256": (
                    official_foragax_module._tracked_tree_sha256(
                        repository,
                        "src",
                    )
                ),
            }
        ],
        "repository": OFFICIAL_FORAGAX_REPOSITORY,
        "schema_version": "1.0",
    }
    descriptor_path = tmp_path / "official-foragax-test-trust.json"
    descriptor_bytes = (
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n"
    ).encode()
    descriptor_path.write_bytes(descriptor_bytes)
    monkeypatch.setattr(
        official_foragax_module,
        "_TRUST_DESCRIPTOR_PATH",
        descriptor_path,
    )
    monkeypatch.setattr(
        official_foragax_module,
        "OFFICIAL_FORAGAX_TRUST_DESCRIPTOR_SHA256",
        hashlib.sha256(descriptor_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        official_foragax_module,
        "_ALLOW_TEST_NATIVE_EXECUTION",
        True,
    )
    return repository, config_commit, execution_commit


@pytest.fixture
def request_factory(
    official_repository: tuple[Path, str, str],
    tmp_path: Path,
) -> Callable[..., OfficialForagaxRunRequest]:
    repository, config_commit, execution_commit = official_repository

    def factory(**changes: Any) -> OfficialForagaxRunRequest:
        defaults: dict[str, Any] = {
            "repository": repository,
            "execution_commit": execution_commit,
            "config_path": Path("configs/dqn.json"),
            "config_commit": config_commit,
            "interpreter": Path(sys.executable),
            "output_dir": tmp_path / "run",
            "index": 2,
            "expected_seed": 9,
            "max_env_steps": 5,
        }
        defaults.update(changes)
        return OfficialForagaxRunRequest(**defaults)

    return factory


@pytest.fixture
def batch_request_factory(
    official_repository: tuple[Path, str, str],
    tmp_path: Path,
) -> Callable[..., OfficialForagaxBatchRunRequest]:
    repository, config_commit, execution_commit = official_repository

    def factory(**changes: Any) -> OfficialForagaxBatchRunRequest:
        defaults: dict[str, Any] = {
            "repository": repository,
            "execution_commit": execution_commit,
            "config_path": Path("configs/dqn_zero.json"),
            "config_commit": config_commit,
            "interpreter": Path(sys.executable),
            "output_dir": tmp_path / "batch-run",
            "indices": (0, 1),
            "expected_seeds": (0, 1),
            "max_env_steps": 5,
        }
        defaults.update(changes)
        return OfficialForagaxBatchRunRequest(**defaults)

    return factory


def test_dry_run_separates_config_and_execution_provenance(
    request_factory: Callable[..., OfficialForagaxRunRequest],
) -> None:
    plan = run_official_foragax(request_factory(), dry_run=True)
    assert isinstance(plan, OfficialForagaxRunPlan)
    assert plan.source["execution_commit"] != plan.source["config_commit"]
    assert plan.source["config_sha256"]
    assert plan.source["config_git_blob_sha1"]
    assert plan.source["lock_sha256"]
    assert (
        plan.source["config_commit_lock_sha256"]
        != plan.source["lock_sha256"]
    )
    assert plan.source["source_tree_sha256"]
    assert plan.run["entrypoint_family"] == "continuing"
    assert plan.run["stored_seed"] == 2
    assert plan.run["effective_seed"] == 9
    assert plan.run["expected_result_env_steps"] == 5
    assert plan.command[-2:] == ("--max_steps", "5")
    assert not plan.output_dir.exists()
    assert plan.run["resolved_hyperparameters_sha256"]
    assert plan.run["registry"]["class"] == "algorithms.nn.DQN.DQN"
    assert plan.run["registry_sha256"]
    assert plan.run["agent_access"]["classified"] is True
    assert plan.run["agent_access"]["method_family"] == "learning"
    assert plan.run["agent_access"]["privileged"] is False
    assert plan.run["agent_access_binding_sha256"]


@pytest.mark.parametrize(
    (
        "config_name",
        "method_family",
        "role",
        "privileged",
        "information_flag",
    ),
    [
        (
            "dqn_fov9.json",
            "learning",
            "learning_baseline",
            False,
            None,
        ),
        (
            "dqn_world.json",
            "learning",
            "privileged_learning_control",
            True,
            "uses_object_identity_observation",
        ),
        (
            "search_nearest.json",
            "search_control",
            "privileged_control",
            True,
            None,
        ),
        (
            "search_oracle.json",
            "search_control",
            "privileged_control",
            True,
            "uses_reward_grid",
        ),
        (
            "dqn_reward_grid.json",
            "learning",
            "privileged_learning_control",
            True,
            "uses_reward_grid",
        ),
        (
            "dqn_temperature.json",
            "learning",
            "privileged_learning_control",
            True,
            "uses_temperature_info",
        ),
        (
            "mcts.json",
            "planning_control",
            "privileged_control",
            True,
            "uses_simulator_state",
        ),
        (
            "unknown.json",
            "unclassified",
            "unclassified",
            None,
            None,
        ),
    ],
)
def test_agent_access_classification_is_registry_and_hyperparameter_bound(
    request_factory: Callable[..., OfficialForagaxRunRequest],
    config_name: str,
    method_family: str,
    role: str,
    privileged: bool | None,
    information_flag: str | None,
) -> None:
    plan = prepare_official_foragax_run(
        request_factory(
            config_path=Path("configs") / config_name,
            index=0,
            expected_seed=0,
            output_dir=request_factory().output_dir.parent
            / f"classify-{Path(config_name).stem}",
        )
    )
    access = plan.run["agent_access"]
    assert access["method_family"] == method_family
    assert access["role"] == role
    assert access["classified"] is (privileged is not None)
    assert access["privileged"] is privileged
    if information_flag is not None:
        assert access["information_access"][information_flag] is True
    assert (
        plan.run["resolved_hyperparameters_sha256"]
        == official_foragax_module._json_sha256(
            plan.run["resolved_hyperparameters"]
        )
    )
    assert plan.run["registry_sha256"] == official_foragax_module._json_sha256(
        plan.run["registry"]
    )
    assert plan.run["agent_access_sha256"] == official_foragax_module._json_sha256(
        access
    )


@pytest.mark.parametrize(
    ("config_name", "expected_agent", "expected_privileged"),
    [
        ("dqn_fov9.json", "DQN", False),
        ("search_nearest.json", "Search-Nearest", True),
        ("search_oracle.json", "Search-Oracle", True),
    ],
)
def test_strict_manifest_import_derives_scientific_identity_and_access(
    request_factory: Callable[..., OfficialForagaxRunRequest],
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    expected_agent: str,
    expected_privileged: bool,
) -> None:
    request = request_factory(
        config_path=Path("configs") / config_name,
        index=0,
        expected_seed=0,
        output_dir=request_factory().output_dir.parent
        / f"strict-{Path(config_name).stem}",
    )
    completed = run_official_foragax(request)
    assert isinstance(completed, OfficialForagaxRun)
    _endorse_candidate(monkeypatch, completed.manifest_path)
    spec = official_foragax_run_spec_from_manifest(completed.manifest_path)
    assert spec.agent == expected_agent
    assert spec.privileged is expected_privileged
    assert spec.protocol_attested is True
    assert spec.agent_access["official_agent"] == expected_agent

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        official_foragax_run_spec_from_manifest(
            completed.manifest_path,
            agent="caller-relabel",
        )


def test_strict_manifest_import_refuses_unknown_registry_class(
    request_factory: Callable[..., OfficialForagaxRunRequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = request_factory(
        config_path=Path("configs/unknown.json"),
        index=0,
        expected_seed=0,
        output_dir=request_factory().output_dir.parent / "strict-unknown",
    )
    completed = run_official_foragax(request)
    assert isinstance(completed, OfficialForagaxRun)
    assert completed.manifest["run"]["agent_access"]["classified"] is False
    _endorse_candidate(monkeypatch, completed.manifest_path)
    verify_official_foragax_manifest(completed.manifest_path)
    with pytest.raises(
        OfficialForagaxValidationError,
        match="refuses an unclassified agent",
    ):
        official_foragax_run_spec_from_manifest(completed.manifest_path)


def test_native_range_plan_uses_one_half_open_official_expression(
    batch_request_factory: Callable[..., OfficialForagaxBatchRunRequest],
) -> None:
    indices = tuple(range(30))
    plan = prepare_official_foragax_batch_run(
        batch_request_factory(
            indices=indices,
            expected_seeds=tuple(range(30)),
        )
    )
    assert isinstance(plan, OfficialForagaxBatchRunPlan)
    position = plan.command.index("-i")
    assert plan.command[position + 1] == "0:30"
    assert plan.command.count("-i") == 1
    assert plan.run["indices"] == list(range(30))
    assert plan.run["effective_seeds"] == list(range(30))
    assert plan.run["native_single_process_batch"] is True
    assert "half-open" in str(plan.run["index_expression_semantics"])


def test_request_rejects_untyped_paths_and_non_boolean_gpu(
    request_factory: Callable[..., OfficialForagaxRunRequest],
) -> None:
    with pytest.raises(OfficialForagaxValidationError, match="gpu must be a boolean"):
        request_factory(gpu=1)
    with pytest.raises(
        OfficialForagaxValidationError,
        match="config_path must be a pathlib.Path",
    ):
        request_factory(config_path="configs/dqn.json")


def test_batch_preflight_rejects_seed_mismatch_and_duplicate_effective_seed(
    batch_request_factory: Callable[..., OfficialForagaxBatchRunRequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        OfficialForagaxValidationError,
        match="index 1 has effective seed 1; expected 999",
    ):
        prepare_official_foragax_batch_run(
            batch_request_factory(expected_seeds=(0, 999))
        )

    original_probe = official_foragax_module._probe_experiment

    def duplicate_probe(**kwargs: Any) -> Any:
        payload = dict(original_probe(**kwargs))
        if kwargs["index"] == 1:
            payload["stored_seed"] = 0
            payload["effective_seed"] = 0
            payload["jax_key_words"] = [0, 0]
        return payload

    monkeypatch.setattr(
        official_foragax_module,
        "_probe_experiment",
        duplicate_probe,
    )
    with pytest.raises(
        OfficialForagaxValidationError,
        match="duplicate (stored|effective) seeds",
    ):
        prepare_official_foragax_batch_run(
            batch_request_factory(expected_seeds=None)
        )


def test_native_batch_success_resume_and_exact_artifact_set(
    batch_request_factory: Callable[..., OfficialForagaxBatchRunRequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = batch_request_factory()
    completed = run_official_foragax_batch(request)
    assert isinstance(completed, OfficialForagaxBatchRun)
    assert not completed.resumed
    assert [path.name for path in completed.artifact_paths] == ["0.npz", "1.npz"]
    assert completed.manifest["artifact_set"]["ordered_indices"] == [0, 1]
    assert completed.manifest["run"]["effective_seeds"] == [0, 1]
    static_run = completed.manifest["run"]
    assert all(
        entry["resolved_hyperparameters_sha256"]
        == static_run["resolved_hyperparameters_sha256"]
        and entry["registry_sha256"] == static_run["registry_sha256"]
        and entry["agent_access_sha256"] == static_run["agent_access_sha256"]
        and entry["agent_access_binding_sha256"]
        == static_run["agent_access_binding_sha256"]
        for entry in static_run["runs"]
    )
    command = completed.manifest["execution"]["command"]
    assert command[command.index("-i") + 1] == "0:2"
    assert completed.manifest["execution"]["logs"]["stdout"]["sha256"]
    assert completed.manifest["attestation_state"] == (
        "protocol_conformant_candidate"
    )
    assert "verified" not in completed.manifest["environment"]["implementation"]
    execution_environment = completed.manifest["execution"][
        "relevant_environment"
    ]
    assert "HOME" not in execution_environment
    assert "PYTHONPATH" not in execution_environment
    assert execution_environment["PYTHONHASHSEED"] == "0"
    assert execution_environment["PATH"].startswith(
        "<REDACTED_PATH_VALUE sha256="
    )
    assert completed.manifest["source_at_completion"][
        "execution_environment_sha256"
    ]
    auxiliary_by_path = {
        item["path"]: item for item in completed.manifest["auxiliary_files"]
    }
    assert set(auxiliary_by_path) == {
        "official-checkpoints/state/checkpoint.bin",
        "official-results/result/results.db",
    }
    assert all(item["type"] == "file" for item in auxiliary_by_path.values())
    assert {
        item["path"] for item in completed.manifest["output_directories"]
    } == {
        "experiment",
        "official-checkpoints",
        "official-checkpoints/state",
        "official-results",
        "official-results/result",
        "official-results/result/data",
    }
    assert all(
        item["type"] == "directory"
        for item in completed.manifest["output_directories"]
    )
    assert completed.manifest["output_tree"]["entry_count"] == 14
    assert completed.manifest["output_tree"]["file_count"] == 8
    assert completed.manifest["output_tree"]["directory_count"] == 6
    assert (
        completed.manifest["output_tree"]["hash_scheme"]
        == "relative-path+type+size+bytes-v2"
    )
    assert completed.manifest["output_tree"]["sha256"]

    serialized = completed.manifest_path.read_text(encoding="utf-8")
    assert str(request.repository.resolve()) not in serialized
    assert str(request.output_dir.resolve()) not in serialized
    assert str(Path(sys.executable).resolve()) not in serialized
    stdout_text = (completed.manifest_path.parent / "stdout.log").read_text(
        encoding="utf-8"
    )
    assert str(request.repository.resolve()) not in stdout_text
    assert str(request.output_dir.resolve()) not in stdout_text

    original_mtimes = tuple(path.stat().st_mtime_ns for path in completed.artifact_paths)
    resumed = run_official_foragax_batch(request, resume=True)
    assert isinstance(resumed, OfficialForagaxBatchRun)
    assert resumed.resumed
    assert tuple(path.stat().st_mtime_ns for path in resumed.artifact_paths) == (
        original_mtimes
    )
    _endorse_candidate(monkeypatch, completed.manifest_path)
    standalone = verify_official_foragax_batch_manifest(completed.manifest_path)
    assert standalone.artifact_paths == completed.artifact_paths
    specs = official_foragax_batch_run_specs_from_manifest(
        completed.manifest_path
    )
    assert tuple(spec.seed for spec in specs) == (0, 1)
    assert tuple(spec.path for spec in specs) == completed.artifact_paths
    assert all(spec.environment.preset == "relearning" for spec in specs)
    assert all(
        spec.environment.resolved_env_id == "ForagaxSquareWaveTwoBiome-v11"
        for spec in specs
    )
    assert all(spec.config_commit == request.config_commit for spec in specs)
    assert all(spec.protocol_attested for spec in specs)
    from alberta_framework.benchmarks.forager import ForagerEnvConfig

    with pytest.raises(
        ValueError,
        match="environment semantics do not match",
    ):
        official_foragax_batch_run_specs_from_manifest(
            completed.manifest_path,
            environment=ForagerEnvConfig.paper_unending(),
        )

    database = (
        completed.manifest_path.parent
        / "official-results/result/results.db"
    )
    original_database = database.read_bytes()
    database.write_bytes(original_database + b"tampered")
    with pytest.raises(
        OfficialForagaxValidationError,
        match="auxiliary file set, metadata, or hash",
    ):
        verify_official_foragax_batch_manifest(completed.manifest_path)
    database.write_bytes(original_database)

    extra_output = completed.manifest_path.parent / "unbound-output.bin"
    extra_output.write_bytes(b"extra")
    with pytest.raises(
        OfficialForagaxValidationError,
        match="auxiliary file set",
    ):
        verify_official_foragax_batch_manifest(completed.manifest_path)
    extra_output.unlink()

    extra_directory = completed.manifest_path.parent / "unbound-empty-directory"
    extra_directory.mkdir()
    with pytest.raises(
        OfficialForagaxValidationError,
        match="output directory set",
    ):
        verify_official_foragax_batch_manifest(completed.manifest_path)
    extra_directory.rmdir()

    temporary_leftover = completed.manifest_path.parent / ".orphan.tmp"
    temporary_leftover.write_bytes(b"partial")
    with pytest.raises(
        OfficialForagaxValidationError,
        match="construction leftover",
    ):
        verify_official_foragax_batch_manifest(completed.manifest_path)
    temporary_leftover.unlink()

    temporary_directory = completed.manifest_path.parent / ".orphan-dir.tmp"
    temporary_directory.mkdir()
    with pytest.raises(
        OfficialForagaxValidationError,
        match="construction leftover",
    ):
        verify_official_foragax_batch_manifest(completed.manifest_path)
    temporary_directory.rmdir()

    stale_lock = completed.manifest_path.parent / ".running"
    stale_lock.write_bytes(b"")
    with pytest.raises(
        OfficialForagaxValidationError,
        match="stale running lock",
    ):
        verify_official_foragax_batch_manifest(completed.manifest_path)
    stale_lock.unlink()

    stale_lock.mkdir()
    with pytest.raises(
        OfficialForagaxValidationError,
        match="stale running lock",
    ):
        verify_official_foragax_batch_manifest(completed.manifest_path)
    stale_lock.rmdir()

    symlink_file = completed.manifest_path.parent / "linked-output.bin"
    symlink_file.symlink_to(database)
    with pytest.raises(
        OfficialForagaxValidationError,
        match="contains a symlink",
    ):
        verify_official_foragax_batch_manifest(completed.manifest_path)
    symlink_file.unlink()

    symlink_directory = completed.manifest_path.parent / "linked-output-dir"
    symlink_directory.symlink_to(database.parent, target_is_directory=True)
    with pytest.raises(
        OfficialForagaxValidationError,
        match="contains a symlink",
    ):
        verify_official_foragax_batch_manifest(completed.manifest_path)
    symlink_directory.unlink()

    if hasattr(os, "mkfifo"):
        fifo = completed.manifest_path.parent / "unexpected.fifo"
        os.mkfifo(fifo)
        with pytest.raises(
            OfficialForagaxValidationError,
            match="non-regular file",
        ):
            verify_official_foragax_batch_manifest(completed.manifest_path)
        fifo.unlink()

    first, second = completed.artifact_paths
    saved_first = first.read_bytes()
    first.unlink()
    with pytest.raises((FileNotFoundError, OfficialForagaxValidationError)):
        verify_official_foragax_batch_manifest(completed.manifest_path)
    first.write_bytes(saved_first)

    extra = first.parent / "99.npz"
    extra.write_bytes(second.read_bytes())
    with pytest.raises(
        OfficialForagaxValidationError,
        match="missing, extra, or duplicate",
    ):
        verify_official_foragax_batch_manifest(completed.manifest_path)
    extra.unlink()

    duplicate = first.parent.parent / "duplicate" / "data" / first.name
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(first.read_bytes())
    plan = prepare_official_foragax_batch_run(request)
    with pytest.raises(
        OfficialForagaxValidationError,
        match="duplicate=\\[0\\]",
    ):
        official_foragax_module._find_batch_results(plan)
    duplicate.unlink()

    with np.load(second, allow_pickle=False) as archive:
        rewards = np.asarray(archive["rewards"])
    rewards[0] = -321.0
    np.savez_compressed(second, rewards=rewards)
    with pytest.raises(
        OfficialForagaxValidationError,
        match="artifact 1 metadata or hash",
    ):
        verify_official_foragax_batch_manifest(completed.manifest_path)


def test_run_writes_atomic_manifest_and_resume_verifies_every_hash(
    request_factory: Callable[..., OfficialForagaxRunRequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = request_factory()
    completed = run_official_foragax(request)
    assert isinstance(completed, OfficialForagaxRun)
    assert not completed.resumed
    assert completed.manifest_path.name == "manifest.json"
    assert not list(completed.manifest_path.parent.glob(".*.tmp"))
    rewards_metadata = completed.manifest["artifact"][
        "validated_consumed_arrays"
    ]["rewards"]
    assert rewards_metadata["shape"] == [5]
    assert rewards_metadata["all_finite"] is True
    assert completed.manifest["execution"]["package_freeze"]
    assert completed.manifest["execution"]["runtime"]["jax_devices"]
    assert (
        completed.manifest["execution"]["runtime"]["python_hash_seed"]
        == "0"
    )
    assert completed.manifest["execution"]["runtime"]["python_soabi"]
    assert completed.manifest["execution"]["runtime"]["distribution_records"]
    assert completed.manifest["execution"]["runtime"]["jax_config"]
    assert completed.manifest["source"]["harness_module_sha256"]
    assert completed.manifest["run"]["index"] == 2
    assert completed.manifest["run"]["stored_seed"] == 2
    assert completed.manifest["run"]["effective_seed"] == 9
    manifest_link = completed.manifest_path.parent.parent / "manifest-link.json"
    manifest_link.symlink_to(completed.manifest_path)
    with pytest.raises(
        OfficialForagaxValidationError,
        match="manifest is not a regular file",
    ):
        verify_official_foragax_manifest(manifest_link)
    manifest_link.unlink()
    with monkeypatch.context() as context:
        stale_closure = "0" * 64
        context.setattr(
            official_foragax_module,
            "_HARNESS_SHA256_AT_IMPORT",
            stale_closure,
        )
        context.setattr(
            official_foragax_module,
            "_harness_sha256",
            lambda: stale_closure,
        )
        with pytest.raises(
            OfficialForagaxValidationError,
            match="validator source closure",
        ):
            verify_official_foragax_manifest(
                completed.manifest_path,
                _require_endorsement=False,
            )
    _endorse_candidate(monkeypatch, completed.manifest_path)
    with pytest.raises(
        OfficialForagaxValidationError,
        match="index \\(2\\) differs from effective seed \\(9\\)",
    ):
        official_foragax_run_spec_from_manifest(completed.manifest_path)

    original_mtime = completed.artifact_path.stat().st_mtime_ns
    resumed = run_official_foragax(request, resume=True)
    assert isinstance(resumed, OfficialForagaxRun)
    assert resumed.resumed
    assert resumed.artifact_path.stat().st_mtime_ns == original_mtime

    with monkeypatch.context() as context:
        context.setattr(
            official_foragax_module,
            "_harness_sha256",
            lambda: "0" * 64,
        )
        with pytest.raises(
            OfficialForagaxValidationError,
            match="runner harness changed",
        ):
            run_official_foragax(request, resume=True)

    snapshot = resumed.manifest_path.parent / "experiment/config.snapshot.json"
    original_snapshot = snapshot.read_bytes()
    snapshot.write_bytes(original_snapshot + b"\n")
    with pytest.raises(
        OfficialForagaxValidationError,
        match="historical config snapshot",
    ):
        run_official_foragax(request, resume=True)
    snapshot.write_bytes(original_snapshot)

    snapshot_copy = snapshot.parent.parent.parent / "snapshot-copy.json"
    snapshot_copy.write_bytes(original_snapshot)
    snapshot.unlink()
    snapshot.symlink_to(snapshot_copy)
    with pytest.raises(
        OfficialForagaxValidationError,
        match="historical config snapshot is not a regular file",
    ):
        run_official_foragax(request, resume=True)
    snapshot.unlink()
    snapshot.write_bytes(original_snapshot)

    stdout = resumed.manifest_path.parent / "stdout.log"
    original_stdout = stdout.read_bytes()
    stdout.write_bytes(original_stdout + b"tampered\n")
    with pytest.raises(
        OfficialForagaxValidationError,
        match="stdout log metadata or hash",
    ):
        run_official_foragax(request, resume=True)
    stdout.write_bytes(original_stdout)

    stdout_copy = stdout.parent.parent / "stdout-copy.log"
    stdout_copy.write_bytes(original_stdout)
    stdout.unlink()
    stdout.symlink_to(stdout_copy)
    with pytest.raises(
        OfficialForagaxValidationError,
        match="log is not a regular file",
    ):
        run_official_foragax(request, resume=True)
    stdout.unlink()
    stdout.write_bytes(original_stdout)

    extra = resumed.artifact_path.parent / "99.npz"
    extra.write_bytes(resumed.artifact_path.read_bytes())
    with pytest.raises(
        OfficialForagaxValidationError,
        match=(
            "auxiliary file set, metadata, or hash|"
            "missing, extra, or duplicate data artifacts"
        ),
    ):
        run_official_foragax(request, resume=True)
    extra.unlink()

    artifact_copy = resumed.manifest_path.parent.parent / "artifact-copy.npz"
    artifact_copy.write_bytes(resumed.artifact_path.read_bytes())
    resumed.artifact_path.unlink()
    resumed.artifact_path.symlink_to(artifact_copy)
    with pytest.raises(
        OfficialForagaxValidationError,
        match="result artifact is not a regular file",
    ):
        run_official_foragax(request, resume=True)
    resumed.artifact_path.unlink()
    resumed.artifact_path.write_bytes(artifact_copy.read_bytes())

    with np.load(resumed.artifact_path, allow_pickle=False) as archive:
        rewards = np.asarray(archive["rewards"])
    rewards[0] = -123.0
    np.savez_compressed(resumed.artifact_path, rewards=rewards)
    with pytest.raises(
        OfficialForagaxValidationError,
        match="artifact metadata or hash",
    ):
        run_official_foragax(request, resume=True)


def test_ppo_env_step_request_is_converted_to_updates(
    request_factory: Callable[..., OfficialForagaxRunRequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = request_factory(
        config_path=Path("configs/ppo.json"),
        index=3,
        expected_seed=3,
        max_env_steps=8,
        output_dir=request_factory().output_dir.parent / "ppo-run",
    )
    plan = prepare_official_foragax_run(request)
    assert plan.run["entrypoint_family"] == "ppo"
    assert plan.run["rollout_steps"] == 4
    assert plan.run["max_steps_argument"] == 2
    assert plan.run["expected_result_env_steps"] == 8
    assert plan.command[-2:] == ("--max_steps", "2")

    completed = run_official_foragax(request)
    assert isinstance(completed, OfficialForagaxRun)
    assert completed.manifest["artifact"]["validated_consumed_arrays"][
        "rewards"
    ]["shape"] == [8]
    _endorse_candidate(monkeypatch, completed.manifest_path)
    imported = official_foragax_run_spec_from_manifest(completed.manifest_path)
    assert imported.seed == 3
    assert imported.path == completed.artifact_path
    assert imported.protocol_attested is True

    with pytest.raises(OfficialForagaxValidationError, match="cannot be represented exactly"):
        prepare_official_foragax_run(
            dataclasses.replace(
                request,
                output_dir=request.output_dir.parent / "bad-ppo-run",
                max_env_steps=7,
            )
        )


def test_ppo_default_records_actual_rollout_rounded_horizon(
    request_factory: Callable[..., OfficialForagaxRunRequest],
) -> None:
    plan = prepare_official_foragax_run(
        request_factory(
            config_path=Path("configs/ppo.json"),
            index=0,
            expected_seed=0,
            max_env_steps=None,
        )
    )
    assert plan.run["configured_env_steps"] == 10
    assert plan.run["configured_updates"] == 3
    assert plan.run["expected_result_env_steps"] == 12
    assert "--max_steps" not in plan.command


def test_preflight_rejects_dirty_and_unallowlisted_execution_source(
    request_factory: Callable[..., OfficialForagaxRunRequest],
    official_repository: tuple[Path, str, str],
) -> None:
    repository, config_commit, _execution_commit = official_repository
    with pytest.raises(OfficialForagaxValidationError, match="effective seed"):
        prepare_official_foragax_run(request_factory(expected_seed=8))

    _write(repository / "untracked.txt", "dirty\n")
    with pytest.raises(OfficialForagaxValidationError, match="must be clean"):
        prepare_official_foragax_run(request_factory())
    (repository / "untracked.txt").unlink()

    config = repository / "configs/dqn.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    data["total_steps"] = 101
    config.write_text(json.dumps(data) + "\n", encoding="utf-8")
    _git(repository, "add", "configs/dqn.json")
    _git(repository, "commit", "-qm", "repurpose config")
    current = _git(repository, "rev-parse", "HEAD")
    historical_request = request_factory(
        execution_commit=current,
        config_commit=config_commit,
    )
    with pytest.raises(
        OfficialForagaxValidationError,
        match="not uniquely allowlisted",
    ):
        prepare_official_foragax_run(historical_request)


def test_manifest_tampering_and_nonfinite_rewards_fail_closed(
    request_factory: Callable[..., OfficialForagaxRunRequest],
) -> None:
    request = request_factory()
    completed = run_official_foragax(request)
    assert isinstance(completed, OfficialForagaxRun)
    payload = json.loads(completed.manifest_path.read_text(encoding="utf-8"))
    for schema_version in ("1.1", "1.2", "1.3"):
        legacy_payload = json.loads(json.dumps(payload))
        legacy_payload["schema_version"] = schema_version
        legacy_payload["manifest_sha256"] = (
            official_foragax_module._canonical_json_sha256(legacy_payload)
        )
        legacy_path = (
            completed.manifest_path.parent.parent
            / f"legacy-manifest-{schema_version}.json"
        )
        legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
        with pytest.raises(
            OfficialForagaxValidationError,
            match="archival evidence and rerun with schema 1.4",
        ):
            verify_official_foragax_manifest(legacy_path)

    for unsafe_path in (
        "../outside.npz",
        "/tmp/outside.npz",
        r"C:\outside.npz",
        r"..\outside.npz",
    ):
        unsafe = json.loads(json.dumps(payload))
        unsafe["artifact"]["path"] = unsafe_path
        unsafe["manifest_sha256"] = (
            official_foragax_module._canonical_json_sha256(unsafe)
        )
        completed.manifest_path.write_text(
            json.dumps(unsafe),
            encoding="utf-8",
        )
        with pytest.raises(
            OfficialForagaxValidationError,
            match="canonical path inside its run directory",
        ):
            verify_official_foragax_manifest(completed.manifest_path)

    relabelled = json.loads(json.dumps(payload))
    relabelled["run"]["agent"] = "caller-relabelled"
    relabelled["manifest_sha256"] = (
        official_foragax_module._canonical_json_sha256(relabelled)
    )
    completed.manifest_path.write_text(
        json.dumps(relabelled),
        encoding="utf-8",
    )
    with pytest.raises(
        OfficialForagaxValidationError,
        match="(agent-access classification|effective configuration) does not verify",
    ):
        verify_official_foragax_manifest(completed.manifest_path)

    forged_access = json.loads(json.dumps(payload))
    forged_access["run"]["agent_access"]["privileged"] = True
    forged_access["run"]["agent_access_sha256"] = (
        official_foragax_module._json_sha256(
            forged_access["run"]["agent_access"]
        )
    )
    forged_access["manifest_sha256"] = (
        official_foragax_module._canonical_json_sha256(forged_access)
    )
    completed.manifest_path.write_text(
        json.dumps(forged_access),
        encoding="utf-8",
    )
    with pytest.raises(
        OfficialForagaxValidationError,
        match="(agent-access classification|effective configuration) does not verify",
    ):
        verify_official_foragax_manifest(completed.manifest_path)

    payload["run"]["effective_seed"] = 999
    completed.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        OfficialForagaxValidationError,
        match="(manifest hash|seed-offset arithmetic)",
    ):
        verify_official_foragax_manifest(completed.manifest_path)

    bad_request = request_factory(
        config_path=Path("configs/dqn_nonfinite.json"),
        output_dir=request.output_dir.parent / "nonfinite-run",
    )
    with pytest.raises(OfficialForagaxValidationError, match="non-finite"):
        run_official_foragax(bad_request)
    assert not (bad_request.output_dir / "manifest.json").exists()


def test_mid_run_source_mutation_fails_before_manifest(
    request_factory: Callable[..., OfficialForagaxRunRequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = request_factory(output_dir=request_factory().output_dir.parent / "source-race")
    original_run = subprocess.run

    def mutating_run(*args: Any, **kwargs: Any) -> Any:
        result = original_run(*args, **kwargs)
        command = args[0] if args else kwargs["args"]
        if (
            isinstance(command, (list, tuple))
            and len(command) > 1
            and Path(command[1]).name == "continuing_main.py"
        ):
            entrypoint = request.repository / "src/continuing_main.py"
            entrypoint.write_text(
                entrypoint.read_text(encoding="utf-8") + "\n# mid-run mutation\n",
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(subprocess, "run", mutating_run)
    with pytest.raises(
        OfficialForagaxValidationError,
        match="source/runtime changed during execution",
    ):
        run_official_foragax(request)
    assert not (request.output_dir / "manifest.json").exists()


def test_mid_run_foragax_tree_hash_drift_fails_before_manifest(
    request_factory: Callable[..., OfficialForagaxRunRequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = request_factory(output_dir=request_factory().output_dir.parent / "package-race")
    original_probe = official_foragax_module._probe_runtime
    probes = 0

    def drifting_probe(**kwargs: Any) -> Any:
        nonlocal probes
        probes += 1
        payload = json.loads(json.dumps(original_probe(**kwargs)))
        if probes > 1:
            payload["foragax_implementation"]["install_tree_sha256"] = "0" * 64
        return payload

    monkeypatch.setattr(
        official_foragax_module,
        "_probe_runtime",
        drifting_probe,
    )
    with pytest.raises(
        OfficialForagaxValidationError,
        match="foragax_install_tree_sha256|runtime_sha256",
    ):
        run_official_foragax(request)
    assert not (request.output_dir / "manifest.json").exists()


def test_output_scan_rejects_deterministic_file_to_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "race-root"
    root.mkdir()
    target = root / "aux.bin"
    target.write_bytes(b"same bytes")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"same bytes")
    original_open = os.open
    swapped = False

    def swapping_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "aux.bin" and dir_fd is not None and not swapped:
            swapped = True
            target.unlink()
            target.symlink_to(outside)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(official_foragax_module.os, "open", swapping_open)
    with pytest.raises(
        OfficialForagaxValidationError,
        match="cannot be read safely|changed while it was opened",
    ):
        official_foragax_module._scan_bound_output_tree(
            root,
            allow_running_lock=False,
        )
    assert swapped


def test_bound_reader_rejects_symlinked_output_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_root = real_parent / "run"
    real_root.mkdir(parents=True)
    (real_root / "artifact.bin").write_bytes(b"artifact")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(
        OfficialForagaxValidationError,
        match="is not a real directory",
    ):
        official_foragax_module._read_bound_regular_file(
            linked_parent / "run",
            "artifact.bin",
            label="test artifact",
        )


def test_manifest_publication_failure_removes_manifest_and_lock(
    request_factory: Callable[..., OfficialForagaxRunRequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = request_factory(
        output_dir=request_factory().output_dir.parent / "manifest-fsync-failure"
    )
    output_dir = request.output_dir.resolve()
    original_atomic_write_json_at = (
        official_foragax_module._atomic_write_json_at
    )
    injected = False

    def failing_manifest_publication(
        root_descriptor: int,
        relative_value: str,
        payload: dict[str, Any],
    ) -> None:
        nonlocal injected
        original_atomic_write_json_at(
            root_descriptor,
            relative_value,
            payload,
        )
        if not injected and relative_value == "manifest.json":
            injected = True
            raise OSError("injected manifest publication failure")

    monkeypatch.setattr(
        official_foragax_module,
        "_atomic_write_json_at",
        failing_manifest_publication,
    )
    with pytest.raises(OSError, match="injected manifest publication"):
        run_official_foragax(request)
    assert injected
    assert not (output_dir / "manifest.json").exists()
    assert not (output_dir / ".running").exists()


def test_manifest_postpublication_keyboard_interrupt_removes_manifest_and_lock(
    request_factory: Callable[..., OfficialForagaxRunRequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = request_factory(
        output_dir=request_factory().output_dir.parent
        / "manifest-keyboard-interrupt"
    )

    def interrupted_verification(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(
        official_foragax_module,
        "verify_official_foragax_manifest",
        interrupted_verification,
    )
    with pytest.raises(KeyboardInterrupt):
        run_official_foragax(request)
    assert not (request.output_dir / "manifest.json").exists()
    assert not (request.output_dir / ".running").exists()


def test_agent_access_rejects_nonfinite_channel_priorities() -> None:
    base = {
        "agent": "Search-Oracle",
        "semantic_environment": {
            "aperture_size": -1,
            "observation_type": "object",
        },
        "registry": {
            "class": "algorithms.MCTSAgent.MCTSAgent",
            "module": "algorithms.MCTSAgent",
        },
    }
    for value in (float("nan"), float("inf"), float("-inf")):
        result = official_foragax_module._classify_official_foragax_agent_access(
            resolved_hyperparameters={
                "mode": "world",
                "channel_priorities": {"apple": value},
            },
            **base,
        )
        assert result["classified"] is False
        assert result["role"] == "unclassified"
