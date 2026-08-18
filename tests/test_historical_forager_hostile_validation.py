"""Hostile input and boundary validation for historical Forager dataclasses."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.historical_forager import (
    HistoricalForagerContractError,
    HistoricalForagerExecution,
    HistoricalForagerPairingIdentity,
    HistoricalForagerRunResult,
)


def _make_run_result() -> HistoricalForagerRunResult:
    return HistoricalForagerRunResult(
        seed=1,
        aperture_size=9,
        steps=100,
        metrics={},
        reward_sidecar={},
        environment_adapter={},
        runtime={},
        kernel={},
    )


def test_historical_forager_execution_rejects_invalid_inputs() -> None:
    with pytest.raises(HistoricalForagerContractError, match="result must be a"):
        HistoricalForagerExecution(
            result=None,  # type: ignore[arg-type]
            final_kernel_state={},
            next_action=0,
            manifest_sha256="a" * 64,
        )

    with pytest.raises(HistoricalForagerContractError, match="next_action must be an integer"):
        HistoricalForagerExecution(
            result=_make_run_result(),
            final_kernel_state={},
            next_action=True,
            manifest_sha256="a" * 64,
        )


def test_historical_forager_pairing_identity_valid_construction() -> None:
    ident = HistoricalForagerPairingIdentity(
        family_id="family_1",
        provenance_sha256="a" * 64,
        seed=1,
        aperture_size=9,
        steps=100,
        semantic_contract_sha256="b" * 64,
        environment_adapter_mode="golden_verified_read_only_source",
        runtime_sha256="c" * 64,
    )
    assert ident.family_id == "family_1"
    assert ident.aperture_size == 9


def test_historical_forager_pairing_identity_rejects_invalid_inputs() -> None:
    with pytest.raises(HistoricalForagerContractError, match="family_id must be a"):
        HistoricalForagerPairingIdentity(
            family_id="",
            provenance_sha256="a" * 64,
            seed=1,
            aperture_size=9,
            steps=100,
            semantic_contract_sha256="b" * 64,
            environment_adapter_mode="golden_verified_read_only_source",
            runtime_sha256="c" * 64,
        )

    with pytest.raises(HistoricalForagerContractError, match="aperture_size must be one of"):
        HistoricalForagerPairingIdentity(
            family_id="family_1",
            provenance_sha256="a" * 64,
            seed=1,
            aperture_size=8,
            steps=100,
            semantic_contract_sha256="b" * 64,
            environment_adapter_mode="golden_verified_read_only_source",
            runtime_sha256="c" * 64,
        )

    with pytest.raises(HistoricalForagerContractError, match="environment_adapter_mode is invalid"):
        HistoricalForagerPairingIdentity(
            family_id="family_1",
            provenance_sha256="a" * 64,
            seed=1,
            aperture_size=9,
            steps=100,
            semantic_contract_sha256="b" * 64,
            environment_adapter_mode="invalid_mode",  # type: ignore[arg-type]
            runtime_sha256="c" * 64,
        )
