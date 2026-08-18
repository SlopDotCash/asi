"""Hostile input and boundary validation for official foragax run plan dataclasses."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks import official_foragax
from alberta_framework.benchmarks.official_foragax import (
    OfficialForagaxBatchRunPlan,
    OfficialForagaxBatchRunRequest,
    OfficialForagaxRunPlan,
    OfficialForagaxRunRequest,
    OfficialForagaxValidationError,
)


class _HostileString(str):
    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile truth hook executed")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile comparison hook executed")

    def strip(self, chars: object = None) -> str:
        del chars
        type(self).calls += 1
        raise AssertionError("hostile strip hook executed")

    def startswith(self, prefix: object, *args: object) -> bool:
        del prefix, args
        type(self).calls += 1
        raise AssertionError("hostile startswith hook executed")

    __hash__ = str.__hash__


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_repository", official_foragax.OFFICIAL_FORAGAX_REPOSITORY, "repository"),
        ("execution_commit", "a" * 40, "execution_commit"),
        ("config_commit", "b" * 40, "config_commit"),
    ],
)
def test_run_request_rejects_hostile_string_identities_before_hooks(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    kwargs = {
        "repository": tmp_path,
        "execution_commit": "a" * 40,
        "config_path": Path("config.json"),
        "config_commit": "b" * 40,
        "interpreter": Path("python"),
        "output_dir": tmp_path / "output",
        "index": 0,
        field: _HostileString(value),
    }
    _HostileString.calls = 0

    with pytest.raises(OfficialForagaxValidationError, match=message):
        OfficialForagaxRunRequest(**kwargs)  # type: ignore[arg-type]

    assert _HostileString.calls == 0


def test_manifest_identity_helpers_reject_or_normalize_hostile_strings(
    tmp_path: Path,
) -> None:
    hostile = _HostileString("algorithms.registry")
    registry = {
        "module": hostile,
        "class": "algorithms.RandomAgent.RandomAgent",
        "registry_source_path": "src/algorithms/registry.py",
        "registry_source_sha256": "a" * 64,
        "class_source_path": "src/algorithms/RandomAgent.py",
        "class_source_sha256": "b" * 64,
    }
    _HostileString.calls = 0
    with pytest.raises(OfficialForagaxValidationError, match="registry module"):
        official_foragax._validated_registry_identity(registry)
    with pytest.raises(OfficialForagaxValidationError, match="manifest artifact"):
        official_foragax._manifest_relative_file(
            tmp_path, _HostileString("artifact.json"), label="artifact"
        )
    assert _HostileString.calls == 0

    runtime = {
        "foragax_implementation": {
            "direct_url": {"url": _HostileString("file:///host/path")}
        }
    }
    sanitized = official_foragax._sanitized_runtime(runtime)
    assert sanitized["foragax_implementation"]["direct_url"]["url"] == "<LOCAL_PATH>"
    assert _HostileString.calls == 0
