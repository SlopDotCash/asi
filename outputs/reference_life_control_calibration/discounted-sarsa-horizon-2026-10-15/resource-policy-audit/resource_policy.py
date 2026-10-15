"""Exact resource-policy checks for the frozen discount-horizon diagnostic."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

EXPECTED_SETTINGS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
FROZEN_INPUTS = Path(__file__).resolve().parent.parent
FROZEN_SCRIPT_HASHES = {
    "run.py": "5f38fb5c9aafd5b667814cfaa7a6757b2310ca30c79e78585b00c2842f39d696",
    "validate.py": "44b15dcd9db8d26850715e83e5b3f9e671b0e1285588e229f8cf93e0f5c2263b",
}


class ResourcePolicyError(ValueError):
    """A missing or noncanonical setting violates the declared resource policy."""


def require_settings(settings: object, *, context: str) -> None:
    if not isinstance(settings, Mapping):
        raise ResourcePolicyError(f"{context}: thread_settings must be a mapping")
    for key, expected in EXPECTED_SETTINGS.items():
        actual = settings.get(key)
        if type(actual) is not str or actual != expected:
            raise ResourcePolicyError(
                f"{context}: {key} must be exactly {expected!r}; got {actual!r}"
            )


def frozen_script(name: str) -> Path:
    path = FROZEN_INPUTS / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_SCRIPT_HASHES[name]:
        raise ResourcePolicyError(f"Frozen {name} bytes changed")
    return path
