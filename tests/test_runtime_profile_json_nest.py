"""Reject deep runtime-profile mappings before json.dumps RecursionError.

Origin ``validate_environment_runtime_profile`` clones the caller mapping
with ``json.dumps`` and no nesting preflight. A 16_000-deep object nest
RecursionError's the C encoder on origin/main. Overlay fail-closes at the
shared 32-deep JSON ceiling before dumps.
"""

from __future__ import annotations

import json
import time

import pytest
from test_runtime_profile import _matched_gpu_profile

from alberta_framework.benchmarks.runtime_profile import (
    _JSON_MAX_DEPTH,
    _json_copy,
    validate_environment_runtime_profile,
)

pytestmark = pytest.mark.unit


def _nest(depth: int) -> dict[str, object]:
    node: dict[str, object] = {"leaf": 1}
    for _ in range(depth):
        node = {"x": node}
    return node


def test_frozen_runtime_profile_json_nest_bound() -> None:
    assert _JSON_MAX_DEPTH == 32


def test_last_fit_runtime_profile_still_validates() -> None:
    profile = _matched_gpu_profile()
    validated = validate_environment_runtime_profile(profile)
    assert validated["schema_version"] == profile["schema_version"]


def test_last_fit_json_chain_still_encodes() -> None:
    copied = _json_copy(_nest(_JSON_MAX_DEPTH - 1), label="runtime profile nest")
    assert isinstance(copied, dict)


def test_origin_recursion_class_rejects_before_dumps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dumps(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("json.dumps ran before the runtime-profile nest gate")

    monkeypatch.setattr(json, "dumps", fail_dumps)
    started = time.perf_counter()
    with pytest.raises(ValueError, match="nesting depth"):
        validate_environment_runtime_profile(_nest(16_000))
    assert time.perf_counter() - started < 0.25
