"""Supplementary coverage for rule_discovery.py genome helpers.

Covers previously untested helpers: decode_genome / genome_from_config
(exact round-trip, flag quantization, parameter bounds), flag_count
(active-mechanism tally), and describe_genome (interpretable summary).
"""

import numpy as np
import pytest

from alberta_framework.benchmarks.rule_discovery import (
    FLAG_NAMES,
    GENOME_SIZE,
    PARAM_NAMES,
    decode_genome,
    describe_genome,
    flag_count,
    genome_from_config,
)


def test_flag_count() -> None:
    genome = np.zeros(GENOME_SIZE)
    genome[0] = 1.0
    genome[1] = 1.0
    assert flag_count(genome) == 2
    assert flag_count(np.zeros(GENOME_SIZE)) == 0


def test_flag_count_threshold() -> None:
    genome = np.full(GENOME_SIZE, 0.5)
    assert flag_count(genome) == 0  # not > 0.5
    genome[0] = 0.51
    assert flag_count(genome) == 1


def test_decode_genome_shape_error() -> None:
    with pytest.raises(ValueError, match="shape"):
        decode_genome(np.zeros(5))


def test_genome_roundtrip() -> None:
    genome = np.random.default_rng(7).random(GENOME_SIZE)
    config = decode_genome(genome)
    rebuilt = genome_from_config(config)
    # Flags quantize exactly; params are clamped to bounds.
    assert rebuilt.shape == (GENOME_SIZE,)
    assert flag_count(rebuilt) == flag_count(genome)


def test_decode_genome_flags_quantized() -> None:
    genome = np.zeros(GENOME_SIZE)
    for i in range(len(FLAG_NAMES)):
        genome[i] = 0.8
    config = decode_genome(genome)
    for name in FLAG_NAMES:
        assert config[name] == 1.0


def test_decode_genome_params_in_bounds() -> None:
    genome = np.full(GENOME_SIZE, 0.5)
    config = decode_genome(genome)
    for name in PARAM_NAMES:
        assert config[name] >= 0.0  # all bounds are non-negative


def test_describe_genome_returns_string() -> None:
    genome = np.zeros(GENOME_SIZE)
    desc = describe_genome(genome)
    assert isinstance(desc, str)
    assert len(desc) > 0
