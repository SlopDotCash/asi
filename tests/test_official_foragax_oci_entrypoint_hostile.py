"""Hostile string gates for OCI launch and qualification."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks import official_foragax_oci as oci

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def test_emit_launch_entrypoint_rejects_hostile_before_membership() -> None:
    hostile = _HostileStr("src/continuing_main.py")
    _HostileStr.calls = 0
    image_id = "sha256:" + "1" * 64
    with pytest.raises(oci.OfficialForagaxOciError, match="allowlisted"):
        oci.emit_launch_command(
            image_id=image_id,
            entrypoint=hostile,  # type: ignore[arg-type]
            config_path="/opt/continual-foragax-agents/config.json",
            index_expression="0",
            gpu=False,
        )
    assert _HostileStr.calls == 0
    # benign passes
    cmd = oci.emit_launch_command(
        image_id=image_id,
        entrypoint="src/continuing_main.py",
        config_path="/opt/continual-foragax-agents/config.json",
        index_expression="0",
        gpu=False,
    )
    assert "src/continuing_main.py" in " ".join(cmd)


def test_emit_launch_rejects_non_string_before_membership() -> None:
    image_id = "sha256:" + "1" * 64
    with pytest.raises(oci.OfficialForagaxOciError, match="allowlisted"):
        oci.emit_launch_command(
            image_id=image_id,
            entrypoint=123,  # type: ignore[arg-type]
            config_path="/opt/continual-foragax-agents/config.json",
            index_expression="0",
            gpu=False,
        )


def test_qualify_backend_rejects_hostile_before_membership() -> None:
    hostile = _HostileStr("cpu")
    _HostileStr.calls = 0
    with pytest.raises(oci.OfficialForagaxOciError, match="cpu or gpu"):
        oci.qualify_v4_runs(
            first_archive=Path("/tmp/a"),
            second_archive=Path("/tmp/b"),
            backend=hostile,  # type: ignore[arg-type]
            image_id="sha256:" + "1" * 64,
            runtime_profile_id="prof",
            effective_seed=0,
            steps=1,
            config_sha256="sha256:" + "2" * 64,
            source_archive_sha256="sha256:" + "3" * 64,
            workload_identity={},
            environment_profile_sha256="sha256:" + "4" * 64,
        )
    assert _HostileStr.calls == 0
    # benign unknown
    with pytest.raises(oci.OfficialForagaxOciError, match="cpu or gpu"):
        oci.qualify_v4_runs(
            first_archive=Path("/tmp/a"),
            second_archive=Path("/tmp/b"),
            backend="unknown_backend_xyz",
            image_id="sha256:" + "1" * 64,
            runtime_profile_id="prof",
            effective_seed=0,
            steps=1,
            config_sha256="sha256:" + "2" * 64,
            source_archive_sha256="sha256:" + "3" * 64,
            workload_identity={},
            environment_profile_sha256="sha256:" + "4" * 64,
        )


