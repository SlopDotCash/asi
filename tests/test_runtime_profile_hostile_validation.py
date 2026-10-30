"""Hostile input and boundary validation for runtime profile dataclasses."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.runtime_profile import (
    EnvironmentRuntimeIdentity,
    environment_rng_schedule_sha256,
)


def test_environment_runtime_identity_valid_construction() -> None:
    ident = EnvironmentRuntimeIdentity(
        runtime_profile_id="profile_1",
        environment_runtime_profile_sha256="a" * 64,
        environment_rng_schedule="dedicated_environment_split_chain_v1",
        environment_rng_schedule_sha256=environment_rng_schedule_sha256(
            "dedicated_environment_split_chain_v1"
        ),
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


@pytest.mark.parametrize(
    ("schedule", "schedule_sha256"),
    [
        ("unsupported_schedule", "b" * 64),
        (
            "dedicated_environment_split_chain_v1",
            environment_rng_schedule_sha256("shared_agent_environment_rng_v1"),
        ),
    ],
)
def test_environment_runtime_identity_binds_rng_schedule(
    schedule: str,
    schedule_sha256: str,
) -> None:
    with pytest.raises(ValueError, match="environment RNG schedule"):
        EnvironmentRuntimeIdentity(
            runtime_profile_id="profile_1",
            environment_runtime_profile_sha256="a" * 64,
            environment_rng_schedule=schedule,  # type: ignore[arg-type]
            environment_rng_schedule_sha256=schedule_sha256,
        )
