"""The completed discount-horizon validator enforces its resource policy."""

import runpy
from pathlib import Path

import pytest

_VALIDATOR = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "reference_life_control_calibration"
    / "discounted-sarsa-horizon-2026-10-15"
    / "validate.py"
)


def test_validator_accepts_only_the_declared_thread_settings() -> None:
    namespace = runpy.run_path(str(_VALIDATOR))
    expected = dict(namespace["_REQUIRED_THREAD_SETTINGS"])
    namespace["require_fixed_thread_settings"]({"thread_settings": expected})
    for name in expected:
        invalid = dict(expected)
        invalid[name] = "2"
        with pytest.raises(ValueError, match="fixed thread settings"):
            namespace["require_fixed_thread_settings"]({"thread_settings": invalid})
