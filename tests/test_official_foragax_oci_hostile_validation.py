"""Hostile input and boundary validation for official Foragax OCI dataclasses."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.official_foragax_oci import (
    DriverLaunchContract,
    OfficialForagaxOciError,
    PreparedOciBuild,
)


def test_prepared_oci_build_valid_construction() -> None:
    build = PreparedOciBuild(
        context=Path("/context"),
        build_spec={},
        build_spec_sha256="a" * 64,
        dockerfile_sha256="b" * 64,
        launcher_sha256="c" * 64,
    )
    assert build.build_spec_sha256 == "a" * 64


def test_prepared_oci_build_rejects_invalid_inputs() -> None:
    with pytest.raises(OfficialForagaxOciError, match="context must be a Path"):
        PreparedOciBuild(
            context="/context",  # type: ignore[arg-type]
            build_spec={},
            build_spec_sha256="a" * 64,
            dockerfile_sha256="b" * 64,
            launcher_sha256="c" * 64,
        )

    with pytest.raises(OfficialForagaxOciError, match="build_spec must be a mapping"):
        PreparedOciBuild(
            context=Path("/context"),
            build_spec=None,  # type: ignore[arg-type]
            build_spec_sha256="a" * 64,
            dockerfile_sha256="b" * 64,
            launcher_sha256="c" * 64,
        )

    with pytest.raises(
        OfficialForagaxOciError, match="build_spec_sha256 must be a lowercase SHA-256"
    ):
        PreparedOciBuild(
            context=Path("/context"),
            build_spec={},
            build_spec_sha256="invalid",
            dockerfile_sha256="b" * 64,
            launcher_sha256="c" * 64,
        )


def test_driver_launch_contract_rejects_invalid_inputs() -> None:
    with pytest.raises(
        OfficialForagaxOciError, match="driver_host_path must be a non-empty string"
    ):
        DriverLaunchContract(
            driver_host_path="",
            driver_container_path="/usr/lib",
            device_paths=(),
            device_indices=(),
            cuda_wheel_library_paths=(),
            driver_user_library_paths=(),
        )

    with pytest.raises(OfficialForagaxOciError, match="device_paths must be an exact tuple"):
        DriverLaunchContract(
            driver_host_path="/usr/lib",
            driver_container_path="/usr/lib",
            device_paths=[],  # type: ignore[arg-type]
            device_indices=(),
            cuda_wheel_library_paths=(),
            driver_user_library_paths=(),
        )
