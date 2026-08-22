"""Supplementary coverage for forager.py and forager_matched_trust.py
helpers.

Covers previously untested helpers: forager_rng_contract (documented seed
schedule) and load_trust_anchor_document (strict schema validation of a
trust-anchor JSON document).
"""

import json

import pytest

from alberta_framework.benchmarks.forager import (
    FORAGER_ENVIRONMENT_RNG_SCHEDULE,
    forager_rng_contract,
)
from alberta_framework.benchmarks.forager_matched_trust import (
    MATCHED_TRUST_ANCHOR_SCHEMA_VERSION,
    TRUST_ANCHOR_ALGORITHM,
    ForagerMatchedTrustError,
    load_trust_anchor_document,
)


def test_forager_rng_contract_schema() -> None:
    contract = forager_rng_contract()
    assert contract["schema_version"] == "alberta.forager_rng_schedule.v1"
    assert contract["identity"] == FORAGER_ENVIRONMENT_RNG_SCHEDULE
    assert "environment" in contract
    assert "root" in contract["environment"]


def test_forager_rng_contract_reset_transition() -> None:
    contract = forager_rng_contract()
    assert "split(env_key" in contract["environment"]["reset"]
    assert "split(env_key" in contract["environment"]["transition"]


def _anchor_payload() -> dict:
    return {
        "schema_version": MATCHED_TRUST_ANCHOR_SCHEMA_VERSION,
        "algorithm": TRUST_ANCHOR_ALGORITHM,
        "trust_anchor_identity": "forager-test",
        "key_id": "k1",
        "key_sha256": "a" * 64,
    }


def test_load_trust_anchor_valid(tmp_path) -> None:
    path = tmp_path / "anchor.json"
    path.write_text(json.dumps(_anchor_payload()), encoding="utf-8")
    anchor = load_trust_anchor_document(path)
    assert anchor.trust_anchor_identity == "forager-test"


def test_load_trust_anchor_wrong_keys(tmp_path) -> None:
    path = tmp_path / "anchor.json"
    payload = _anchor_payload()
    del payload["key_sha256"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ForagerMatchedTrustError):
        load_trust_anchor_document(path)


def test_load_trust_anchor_unsupported_schema(tmp_path) -> None:
    path = tmp_path / "anchor.json"
    payload = _anchor_payload()
    payload["schema_version"] = "999"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ForagerMatchedTrustError, match="schema_version"):
        load_trust_anchor_document(path)


def test_load_trust_anchor_invalid_json(tmp_path) -> None:
    path = tmp_path / "anchor.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(Exception, match="JSON"):
        load_trust_anchor_document(path)
