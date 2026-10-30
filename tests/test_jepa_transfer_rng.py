"""JEPA execution and receipt validation own the same Threefry roots."""

import dataclasses

import jax
import jax.random as jr
import pytest

from alberta_framework.benchmarks import jepa_transfer_feasibility as lane

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def baseline() -> lane.JEPATransferResult:
    protocol = lane.JEPATransferProtocol(
        steps=4, phase_length=2, pretraining_steps=1, warmup_steps=1, exploration_period=2
    )
    with jax.default_prng_impl("threefry2x32"):
        return lane.run_jepa_transfer_feasibility(protocol)


@pytest.mark.parametrize("implementation", ["rbg", "unsafe_rbg"])
def test_jepa_execution_and_validation_ignore_default_rng(
    baseline: lane.JEPATransferResult, implementation: str
) -> None:
    with jax.default_prng_impl(implementation):
        actual = lane.run_jepa_transfer_feasibility(baseline.protocol)
        assert str(jr.key_impl(jr.key(0))) == implementation
        assert lane.validate_jepa_transfer_payload(actual.to_payload()) == actual
        assert lane.validate_jepa_transfer_payload(baseline.to_payload()) == baseline
    with jax.default_prng_impl("threefry2x32"):
        assert lane.validate_jepa_transfer_payload(actual.to_payload()) == actual
    for observed, expected in zip(actual.arms, baseline.arms, strict=True):
        assert dataclasses.replace(observed, timing=expected.timing) == expected
