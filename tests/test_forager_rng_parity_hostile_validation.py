"""Hostile input and boundary validation for Forager RNG parity records."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_rng_parity import (
    EnvironmentTraceDigest,
    ForagerRngParityError,
    KeyFrame,
    TransitionDigest,
    TreeDigest,
)


def _make_key_frame() -> KeyFrame:
    return KeyFrame(
        input_key=(1, 2),
        next_key=(3, 4),
        environment_key=(5, 6),
    )


def _make_tree_digest() -> TreeDigest:
    return TreeDigest(
        leaf_count=10,
        structure_sha256="a" * 64,
        content_sha256="b" * 64,
    )


def _make_transition_digest() -> TransitionDigest:
    td = _make_tree_digest()
    return TransitionDigest(
        index=0,
        action=1,
        keys=_make_key_frame(),
        observation=td,
        reward=td,
        done=td,
        info=td,
        state=td,
    )


def _make_trace_digest() -> EnvironmentTraceDigest:
    td = _make_tree_digest()
    return EnvironmentTraceDigest(
        seed=42,
        action_sequence_sha256="c" * 64,
        reset_keys=_make_key_frame(),
        reset_observation=td,
        reset_state=td,
        transitions=(_make_transition_digest(),),
        trace_sha256="d" * 64,
    )


def test_transition_digest_validation() -> None:
    trans = _make_transition_digest()
    assert trans.index == 0
    assert trans.action == 1

    with pytest.raises(ForagerRngParityError, match="transition.action must lie in"):
        TransitionDigest(
            index=0,
            action=4,
            keys=_make_key_frame(),
            observation=_make_tree_digest(),
            reward=_make_tree_digest(),
            done=_make_tree_digest(),
            info=_make_tree_digest(),
            state=_make_tree_digest(),
        )

    with pytest.raises(ForagerRngParityError, match="keys must be a KeyFrame"):
        TransitionDigest(
            index=0,
            action=1,
            keys={"invalid": "key"},  # type: ignore[arg-type]
            observation=_make_tree_digest(),
            reward=_make_tree_digest(),
            done=_make_tree_digest(),
            info=_make_tree_digest(),
            state=_make_tree_digest(),
        )


def test_environment_trace_digest_validation() -> None:
    trace = _make_trace_digest()
    assert trace.seed == 42

    with pytest.raises(
        ForagerRngParityError, match="transitions must be a tuple of TransitionDigest"
    ):
        td = _make_tree_digest()
        EnvironmentTraceDigest(
            seed=42,
            action_sequence_sha256="c" * 64,
            reset_keys=_make_key_frame(),
            reset_observation=td,
            reset_state=td,
            transitions=[_make_transition_digest()],  # type: ignore[arg-type]
            trace_sha256="d" * 64,
        )
