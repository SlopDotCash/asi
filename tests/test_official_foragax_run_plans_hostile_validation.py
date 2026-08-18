"""Hostile input and boundary validation for official foragax run plan dataclasses."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.official_foragax import (
    OfficialForagaxBatchRunPlan,
    OfficialForagaxBatchRunRequest,
    OfficialForagaxRunPlan,
    OfficialForagaxRunRequest,
    OfficialForagaxValidationError,
)


@pytest.fixture
def dummy_run_request(tmp_path: Path) -> OfficialForagaxRunRequest:
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "configs/dqn.json"
    config.parent.mkdir()
    config.write_text("{}", encoding="utf-8")
    return OfficialForagaxRunRequest(
        repository=repo,
        execution_commit="a" * 40,
        config_path=config,
        config_commit="b" * 40,
        interpreter=Path("/usr/bin/python3"),
        output_dir=tmp_path / "output",
        index=0,
    )


@pytest.fixture
def dummy_batch_request(tmp_path: Path) -> OfficialForagaxBatchRunRequest:
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "configs/dqn.json"
    config.parent.mkdir()
    config.write_text("{}", encoding="utf-8")
    return OfficialForagaxBatchRunRequest(
        repository=repo,
        execution_commit="a" * 40,
        config_path=config,
        config_commit="b" * 40,
        interpreter=Path("/usr/bin/python3"),
        output_dir=tmp_path / "output",
        indices=(0, 1),
    )


def test_official_foragax_run_plan_rejects_invalid_inputs(
    dummy_run_request: OfficialForagaxRunRequest,
) -> None:
    with pytest.raises(TypeError, match="request must be an OfficialForagaxRunRequest"):
        OfficialForagaxRunPlan(
            request=None,  # type: ignore[arg-type]
            trust={},
            source={},
            run={},
            claim={},
            command=("python",),
            environment_overrides={},
            relevant_environment={},
            interpreter_sha256="a" * 64,
            package_freeze=(),
            package_freeze_sha256="b" * 64,
            runtime={},
            config_snapshot_bytes=b"{}",
            execution_config_bytes=b"{}",
        )

    with pytest.raises(
        OfficialForagaxValidationError, match="interpreter_sha256 must be a lowercase SHA-256"
    ):
        OfficialForagaxRunPlan(
            request=dummy_run_request,
            trust={},
            source={},
            run={},
            claim={},
            command=("python",),
            environment_overrides={},
            relevant_environment={},
            interpreter_sha256="invalid",
            package_freeze=(),
            package_freeze_sha256="b" * 64,
            runtime={},
            config_snapshot_bytes=b"{}",
            execution_config_bytes=b"{}",
        )


def test_official_foragax_batch_run_plan_rejects_invalid_inputs(
    dummy_batch_request: OfficialForagaxBatchRunRequest,
) -> None:
    with pytest.raises(TypeError, match="request must be an OfficialForagaxBatchRunRequest"):
        OfficialForagaxBatchRunPlan(
            request=None,  # type: ignore[arg-type]
            trust={},
            source={},
            run={},
            claim={},
            command=("python",),
            environment_overrides={},
            relevant_environment={},
            interpreter_sha256="a" * 64,
            package_freeze=(),
            package_freeze_sha256="b" * 64,
            runtime={},
            config_snapshot_bytes=b"{}",
            execution_config_bytes=b"{}",
        )

    with pytest.raises(TypeError, match="config_snapshot_bytes must be bytes"):
        OfficialForagaxBatchRunPlan(
            request=dummy_batch_request,
            trust={},
            source={},
            run={},
            claim={},
            command=("python",),
            environment_overrides={},
            relevant_environment={},
            interpreter_sha256="a" * 64,
            package_freeze=(),
            package_freeze_sha256="b" * 64,
            runtime={},
            config_snapshot_bytes="not bytes",  # type: ignore[arg-type]
            execution_config_bytes=b"{}",
        )
