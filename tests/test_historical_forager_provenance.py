"""Unit coverage for the historical Forager provenance module.

Covers the fail-closed invariants: canonical SHA-256 identity, detached
copy semantics, strict JSON-tree validation (depth/node/string-byte
budgets, finite numbers, exact-string keys), byte-equivalent validation,
and cross-family pairing rejection.
"""

import hashlib

import pytest

from alberta_framework.benchmarks.historical_forager_provenance import (
    CURRENT_FORAGAX_055_FAMILY_ID,
    HISTORICAL_FORAGER_FAMILY_ID,
    HISTORICAL_FORAGER_PROVENANCE_SCHEMA,
    HISTORICAL_FORAGER_PROVENANCE_SHA256,
    HistoricalForagerFamilyMismatchError,
    HistoricalForagerProvenanceError,
    assert_historical_family_pairing,
    historical_forager_provenance,
    validate_historical_forager_provenance,
)


def test_provenance_returns_detached_copy() -> None:
    first = historical_forager_provenance()
    second = historical_forager_provenance()
    assert first == second
    assert first is not second
    first["tampered"] = True
    assert "tampered" not in second


def test_provenance_has_expected_schema_and_family() -> None:
    provenance = historical_forager_provenance()
    assert provenance["schema_version"] == HISTORICAL_FORAGER_PROVENANCE_SCHEMA
    assert provenance["family_id"] == HISTORICAL_FORAGER_FAMILY_ID
    assert provenance["environment_resolution_attested"] is False
    assert (
        provenance["agents"]["repository_url"]
        == "https://github.com/steventango/forager-agents"
    )


def test_canonical_sha256_matches_module_constant() -> None:
    import json

    provenance = historical_forager_provenance()
    canonical = json.dumps(
        provenance,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == HISTORICAL_FORAGER_PROVENANCE_SHA256


def test_validate_accepts_byte_equivalent_provenance() -> None:
    provenance = historical_forager_provenance()
    validate_historical_forager_provenance(provenance)  # no raise


def test_validate_rejects_non_dict() -> None:
    with pytest.raises(HistoricalForagerProvenanceError, match="actual dictionary"):
        validate_historical_forager_provenance([])
    with pytest.raises(HistoricalForagerProvenanceError, match="actual dictionary"):
        validate_historical_forager_provenance("not a dict")


def test_validate_rejects_altered_provenance() -> None:
    provenance = historical_forager_provenance()
    provenance["family_id"] = "tampered"
    with pytest.raises(
        HistoricalForagerProvenanceError, match="differs from the audited"
    ):
        validate_historical_forager_provenance(provenance)


def test_family_pairing_accepts_historical_pair() -> None:
    assert_historical_family_pairing(
        HISTORICAL_FORAGER_FAMILY_ID, HISTORICAL_FORAGER_FAMILY_ID
    )


def test_family_pairing_rejects_cross_family() -> None:
    with pytest.raises(HistoricalForagerFamilyMismatchError, match="pair only with"):
        assert_historical_family_pairing(
            HISTORICAL_FORAGER_FAMILY_ID, CURRENT_FORAGAX_055_FAMILY_ID
        )


def test_family_pairing_rejects_unknown_family() -> None:
    with pytest.raises(HistoricalForagerFamilyMismatchError):
        assert_historical_family_pairing("unknown-family", HISTORICAL_FORAGER_FAMILY_ID)


def test_family_pairing_rejects_non_strings() -> None:
    with pytest.raises(HistoricalForagerProvenanceError, match="must be a string"):
        assert_historical_family_pairing(123, HISTORICAL_FORAGER_FAMILY_ID)


def test_validation_bounds_depth() -> None:
    # A nested structure beyond _MAX_PROVENANCE_DEPTH (8) must be rejected.
    value = historical_forager_provenance()
    deep: dict[str, object] = {}
    cursor = deep
    for _ in range(12):
        nested: dict[str, object] = {}
        cursor["deep"] = nested
        cursor = nested
    with pytest.raises(HistoricalForagerProvenanceError, match="too large"):
        validate_historical_forager_provenance(deep)


def test_validation_rejects_nonfinite_numbers() -> None:
    with pytest.raises(HistoricalForagerProvenanceError, match="finite"):
        validate_historical_forager_provenance({"x": float("nan")})
    with pytest.raises(HistoricalForagerProvenanceError, match="finite"):
        validate_historical_forager_provenance({"x": float("inf")})


def test_validation_rejects_non_string_keys() -> None:
    with pytest.raises(HistoricalForagerProvenanceError, match="exact strings"):
        validate_historical_forager_provenance({1: "value"})


def test_validation_rejects_unsupported_types() -> None:
    with pytest.raises(HistoricalForagerProvenanceError, match="exact JSON values"):
        validate_historical_forager_provenance({"x": object()})
