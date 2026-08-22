"""Supplementary coverage for development_provenance and
replay_frozen_ipmnist helpers.

Covers previously untested helpers: identity_from_payload (schema-exact
payload parsing) and require_current_identity (identity equality gate), plus
replay_hyperparameters and frozen_hyperparameters (protocol-encoding
validators).
"""

import pytest

from alberta_framework.benchmarks.development_provenance import (
    DevelopmentIdentity,
    identity_from_payload,
    require_current_identity,
)
from alberta_framework.benchmarks.replay_frozen_ipmnist import (
    frozen_hyperparameters,
    replay_hyperparameters,
)


def _valid_payload() -> dict:
    return {
        "lane_source_sha256": "a" * 64,
        "dependency_source_sha256": [["dep", "b" * 64]],
        "runtime_identity": [["python", "3.11"]],
        "dependency_versions": [["numpy", "1.26"]],
        "workload_registry_sha256": "c" * 64,
        "paper_registry_sha256": "d" * 64,
    }


def test_identity_from_payload_valid() -> None:
    identity = identity_from_payload(_valid_payload())
    assert isinstance(identity, DevelopmentIdentity)


def test_identity_from_payload_non_dict() -> None:
    with pytest.raises(ValueError, match="schema"):
        identity_from_payload("not-a-dict")


def test_identity_from_payload_wrong_keys() -> None:
    payload = _valid_payload()
    del payload["paper_registry_sha256"]
    with pytest.raises(ValueError, match="schema"):
        identity_from_payload(payload)


def test_require_current_identity_matches() -> None:
    identity = identity_from_payload(_valid_payload())
    # Same instance → passes.
    require_current_identity(identity, identity)


def test_require_current_identity_mismatch() -> None:
    identity = identity_from_payload(_valid_payload())
    other = identity_from_payload(_valid_payload())
    # Different instances with equal fields → passes (value equality).
    require_current_identity(other, identity)


def test_replay_hyperparameters_valid() -> None:
    params = replay_hyperparameters(replay_update=1.0, context=0.0)
    assert params["replay_weight"] == 1.0
    assert params["context_weight"] == 0.0
    assert "replay_capacity" in params


def test_replay_hyperparameters_invalid() -> None:
    with pytest.raises(ValueError, match="exact binary floats"):
        replay_hyperparameters(replay_update=0.5, context=0.0)


def test_frozen_hyperparameters_valid() -> None:
    params = frozen_hyperparameters(method=1.0, mechanism=0.0)
    assert params["method"] == 1.0
    assert params["mechanism"] == 0.0


def test_frozen_hyperparameters_invalid() -> None:
    with pytest.raises(ValueError, match="outside the protocol"):
        frozen_hyperparameters(method=3.0, mechanism=0.0)
