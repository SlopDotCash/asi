"""Reject deep dual-replay checkpoint mappings before json.dumps RecursionError.

Origin ``DualReplayMemory.from_checkpoint_payload`` digest-binds the caller
``memory`` mapping with ``json.dumps`` and no nesting preflight. A 16_000-deep
object nest RecursionError's the C encoder on origin/main. Overlay fail-closes
at the shared 32-deep JSON ceiling before dumps.
"""

from __future__ import annotations

import json
import time

import jax.random as jr
import pytest

from alberta_framework.core.dual_replay import (
    _CHECKPOINT_JSON_MAX_DEPTH,
    DUAL_REPLAY_CHECKPOINT_SCHEMA,
    MECHANISM_STATUS,
    DualReplayConfig,
    DualReplayMemory,
    _canonical_json,
)

pytestmark = pytest.mark.unit


def _nest(depth: int) -> dict[str, object]:
    node: dict[str, object] = {"leaf": 1}
    for _ in range(depth):
        node = {"x": node}
    return node


def _hostile_checkpoint(memory: object) -> dict[str, object]:
    return {
        "schema": DUAL_REPLAY_CHECKPOINT_SCHEMA,
        "mechanism_status": MECHANISM_STATUS,
        "memory": memory,
        "config_digest": "0" * 64,
        "state": {"ok": True},
        "state_digest": "0" * 64,
    }


def test_frozen_checkpoint_json_nest_bound() -> None:
    assert _CHECKPOINT_JSON_MAX_DEPTH == 32


def test_last_fit_checkpoint_still_roundtrips() -> None:
    memory = DualReplayMemory(
        DualReplayConfig(
            total_capacity=4,
            short_term_capacity=2,
            observation_dim=1,
            action_dim=1,
            short_term_sample_size=1,
            long_term_sample_size=1,
        )
    )
    payload = memory.checkpoint_payload(memory.init(jr.key(7)))
    restored_memory, restored_state = DualReplayMemory.from_checkpoint_payload(payload)
    assert restored_memory.to_config() == memory.to_config()
    assert int(restored_state.accepted_transition_count) == 0


def test_last_fit_json_chain_still_encodes() -> None:
    encoded = _canonical_json(_nest(_CHECKPOINT_JSON_MAX_DEPTH - 1))
    assert encoded.startswith("{")


def test_origin_recursion_class_rejects_before_dumps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dumps(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("json.dumps ran before the checkpoint nest gate")

    monkeypatch.setattr(json, "dumps", fail_dumps)
    started = time.perf_counter()
    with pytest.raises(ValueError, match="nesting depth"):
        DualReplayMemory.from_checkpoint_payload(_hostile_checkpoint(_nest(16_000)))
    assert time.perf_counter() - started < 0.25
