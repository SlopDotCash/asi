"""Host-boundary identities for continual-IA operational timings."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.continual_ia_artifact import _validate_operational

pytestmark = pytest.mark.unit

_ENV = {
    "python": {"implementation": "CPython", "version": "3.12.0"},
    "platform": {"system": "Linux", "release": "1", "machine": "x86_64"},
    "packages": {"jax": "0", "jaxlib": "0", "numpy": "0"},
    "jax_default_backend": "cpu",
    "jax_devices": [{"platform": "cpu", "device_kind": "cpu"}],
}


def _operational(seed: object, condition: object = "baseline") -> dict[str, object]:
    return {
        "digest_exclusion_reason": (
            "host environment and wall-clock timing are non-deterministic"
        ),
        "environment": _ENV,
        "condition_timings": [
            {
                "seed": seed,
                "condition": condition,
                "wall_seconds": 1.0,
                "mean_step_latency_ms": 1.0,
            }
        ],
        "overall_acceptance_passed": False,
    }


@pytest.mark.parametrize("seed", [True, False, 1.0, "1"])
def test_condition_timing_rejects_noncanonical_seed(seed: object) -> None:
    errors: list[str] = []
    _validate_operational(
        _operational(seed),
        expected_results=None,
        expected_acceptance=False,
        errors=errors,
    )
    assert any("invalid identity" in error for error in errors)


def test_condition_timing_accepts_builtin_int_seed() -> None:
    errors: list[str] = []
    _validate_operational(
        _operational(7),
        expected_results=None,
        expected_acceptance=False,
        errors=errors,
    )
    assert not any("invalid identity" in error for error in errors)
