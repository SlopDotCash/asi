"""Hostile input and boundary validation for foragax open screen dataclasses."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.foragax_open_screen import (
    FrozenConfiguration,
    ProcessCapture,
    ProtocolSnapshot,
    ScreenError,
    load_frozen_protocol,
)

_ROOT = Path(__file__).resolve().parent.parent


def test_frozen_configuration_rejects_invalid_inputs() -> None:
    with pytest.raises(ScreenError, match="path must be a non-empty string"):
        FrozenConfiguration(
            path="",
            sha256="a" * 64,
            agent="dqn",
            entrypoint="main.py",
        )

    with pytest.raises(ScreenError, match="sha256 must be a lowercase SHA-256 hex digest"):
        FrozenConfiguration(
            path="agent.py",
            sha256="invalid",
            agent="dqn",
            entrypoint="main.py",
        )


def test_protocol_snapshot_rejects_invalid_inputs() -> None:
    protocol = load_frozen_protocol(_ROOT / "outputs/forager/fov_baseline_screening_cpu_v3")
    with pytest.raises(ScreenError, match="protocol must be a FrozenProtocol"):
        ProtocolSnapshot(
            protocol=None,  # type: ignore[arg-type]
            inventory=(),
            inventory_sha256="a" * 64,
        )

    with pytest.raises(ScreenError, match="inventory must be a tuple"):
        ProtocolSnapshot(
            protocol=protocol,
            inventory=[],  # type: ignore[arg-type]
            inventory_sha256="a" * 64,
        )


def test_process_capture_rejects_invalid_inputs() -> None:
    with pytest.raises(ScreenError, match="returncode must be an integer"):
        ProcessCapture(
            returncode=True,
            stdout=b"",
            stderr=b"",
        )

    with pytest.raises(ScreenError, match="stdout must be bytes"):
        ProcessCapture(
            returncode=0,
            stdout="not bytes",  # type: ignore[arg-type]
            stderr=b"",
        )
