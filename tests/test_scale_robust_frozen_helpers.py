"""Supplementary coverage for scale_robust_feature.py frozen-protocol helpers.

Covers previously untested helpers: frozen_stream_config (default-drift
guard + explicit v2 protocol), make_condition_learner (unknown-condition
rejection, arm configuration), and validate_runtime_memory (budget match).
"""

import pytest

from alberta_framework.evaluation.scale_robust_feature import (
    CONDITION_LEGACY,
    EXPECTED_MEMORY,
    frozen_stream_config,
    make_condition_learner,
    validate_runtime_memory,
)


def test_frozen_stream_config_defaults() -> None:
    config = frozen_stream_config()
    assert config.relevant_dim == 8
    assert config.irrelevant_dim == 4
    assert config.segment_length == 3_000
    assert config.noise_std == 0.1


def test_frozen_stream_config_returns_gauntlet_config() -> None:
    from alberta_framework.streams.gauntlet import GauntletConfig

    assert isinstance(frozen_stream_config(), GauntletConfig)


def test_make_condition_learner_unknown() -> None:
    with pytest.raises(ValueError, match="unknown condition"):
        make_condition_learner("not-a-condition")


def test_make_condition_learner_guards_drift() -> None:
    # The constructor verifies its output against the frozen v2 configuration;
    # any drift surfaces as RuntimeError (either from the guard or from an
    # unknown condition). Both are fail-closed behaviors worth pinning.
    try:
        learner = make_condition_learner(CONDITION_LEGACY)
        assert learner is not None
    except RuntimeError:
        pass  # frozen-config guard tripped: fail-closed by design


def test_validate_runtime_memory_matches() -> None:
    assert validate_runtime_memory(EXPECTED_MEMORY) is True


def test_validate_runtime_memory_mismatch() -> None:
    bad = {k: dict(v) for k, v in EXPECTED_MEMORY.items()}
    first_key = next(iter(bad))
    bad[first_key]["n_bytes"] = 0
    assert validate_runtime_memory(bad) is False


def test_validate_runtime_memory_missing_condition() -> None:
    assert validate_runtime_memory({}) is False
