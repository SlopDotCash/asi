"""Hostile input and boundary validation for Forager matched executor records."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.forager_matched_executor import (
    ForagerMatchedExecutorError,
    LiveRuntimeIdentity,
    PreparedCandidate,
)


def test_live_runtime_identity_validation() -> None:
    ident = LiveRuntimeIdentity(
        executable=Path("/usr/bin/podman"),
        executable_sha256="a" * 64,
        version={"major": 4},
        image_inspection={"id": "image_id"},
        executor_manifest_sha256="b" * 64,
    )
    assert ident.executable_sha256 == "a" * 64

    with pytest.raises(ForagerMatchedExecutorError, match="executable must be a Path"):
        LiveRuntimeIdentity(
            executable="/usr/bin/podman",  # type: ignore[arg-type]
            executable_sha256="a" * 64,
            version={},
            image_inspection={},
            executor_manifest_sha256="b" * 64,
        )

    with pytest.raises(ForagerMatchedExecutorError, match="must be a lowercase SHA-256"):
        LiveRuntimeIdentity(
            executable=Path("/usr/bin/podman"),
            executable_sha256="invalid",
            version={},
            image_inspection={},
            executor_manifest_sha256="b" * 64,
        )


def test_prepared_candidate_validation() -> None:
    with pytest.raises(ForagerMatchedExecutorError, match="candidate must be a MatchedCandidate"):
        PreparedCandidate(
            candidate=None,  # type: ignore[arg-type]
            source_root=Path("/source"),
            source_archive=Path("/source.tar"),
            original_configuration=Path("/config.json"),
            configuration=Path("/config_mod.json"),
            entrypoint_path="main.py",
            python_import_root=".",
            invocation_style="alberta_single_seed_v1",
            result_root="results",
            rng_isolation_patch_sha256=None,
            capability_receipt={},
            capability_receipt_sha256="c" * 64,
            source_inventory={},
        )
