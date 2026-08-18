import pytest

from alberta_framework.benchmarks.ipmnist_ceilings import (
    CEILING_PROTOCOL,
    CeilingResourceLedger,
    FrozenFeatureCeiling,
)


def test_resource_ledger_charges_replay_pretraining_and_extractor() -> None:
    ledger = CeilingResourceLedger(
        persistent_bytes=100,
        replay_bytes=20,
        environment_steps=10,
        pretraining_steps=30,
        model_queries=40,
        extractor_queries=50,
    )
    assert ledger.total_persistent_bytes == 120
    assert ledger.total_steps == 40
    assert ledger.total_model_queries == 90


def test_mechanism_off_is_zero_replay_frozen_random_features() -> None:
    config = FrozenFeatureCeiling(method="randumb", feature_dim=64, replay_capacity=0)
    assert config.mechanism_off is True
    assert config.persistent_replay_bytes(example_bytes=100) == 0


def test_replay_bytes_are_explicit_and_bounded() -> None:
    config = FrozenFeatureCeiling(method="ranpac", feature_dim=64, replay_capacity=5)
    assert config.persistent_replay_bytes(example_bytes=100) == 500
    with pytest.raises(ValueError, match="example_bytes"):
        config.persistent_replay_bytes(example_bytes=-1)


def test_protocol_keeps_ceilings_separate_and_nonpromoting() -> None:
    assert CEILING_PROTOCOL["methods"] == ("replay", "in_context", "randumb", "ranpac", "prol")
    assert CEILING_PROTOCOL["pretraining_allowed_but_charged"] is True
    assert CEILING_PROTOCOL["scientific_promotion_allowed"] is False
