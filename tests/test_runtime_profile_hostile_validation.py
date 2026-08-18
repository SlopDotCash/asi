"""Hostile input and boundary validation for runtime profile dataclasses."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.runtime_profile import (
    EnvironmentRuntimeIdentity,
)


def test_environment_runtime_identity_valid_construction() -> None:
    ident = EnvironmentRuntimeIdentity(
        runtime_profile_id="profile_1",
        environment_runtime_profile_sha256="a" * 64,
        environment_rng_schedule="dedicated_environment_split_chain_v1",
        environment_rng_schedule_sha256="b" * 64,
    )
    assert ident.runtime_profile_id == "profile_1"


def test_environment_runtime_identity_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="runtime_profile_id must be a non-empty string"):
        EnvironmentRuntimeIdentity(
            runtime_profile_id="",
            environment_runtime_profile_sha256="a" * 64,
            environment_rng_schedule="dedicated_environment_split_chain_v1",
            environment_rng_schedule_sha256="b" * 64,
        )

    with pytest.raises(
        ValueError, match="environment_runtime_profile_sha256 must be a lowercase SHA-256"
    ):
        EnvironmentRuntimeIdentity(
            runtime_profile_id="profile_1",
            environment_runtime_profile_sha256="invalid",
            environment_rng_schedule="dedicated_environment_split_chain_v1",
            environment_rng_schedule_sha256="b" * 64,
        )

    with pytest.raises(
        ValueError, match="environment_rng_schedule_sha256 must be a lowercase SHA-256"
    ):
        EnvironmentRuntimeIdentity(
            runtime_profile_id="profile_1",
            environment_runtime_profile_sha256="a" * 64,
            environment_rng_schedule="dedicated_environment_split_chain_v1",
            environment_rng_schedule_sha256="invalid",
        )
