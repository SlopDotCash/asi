"""Hostile ``int`` identity containment at the seed/recovery consistency gate.

``_validate_seed_and_aggregate_consistency`` gated ``seed`` and
``recurrence_recovery_steps`` with ``isinstance(x, bool) or not isinstance(x,
int)``. ``isinstance`` accepts subclasses, so a hostile ``int`` subclass with
an overridden ``__eq__``/``__ge__`` passed the gate, was appended into
``observed_seeds``/``recurrence_steps``, and its overridden dunder then ran
during a later "trusted" comparison — ``tuple(observed_seeds) !=
_EXPECTED_EVIDENCE_SEEDS``, ``aggregate_seeds != observed_seeds``, or
``value >= 0`` over ``recurrence_steps`` — before the value's type was ever
confirmed safe. This is the same spoofable-identity shape already closed at
other gates in this exact file (e.g. the ``check.name`` gate in
``_validate_checks``, and the numeric artifact gates fixed elsewhere in this
module), just missed at this particular boundary.

Both sites now gate on an exact-type membership check (``type(value) is not
int``), which rejects both ``bool`` (``type(True) is bool``, never ``int``)
and any hostile ``int`` subclass before any dunder of the value is ever
invoked, and before the value is ever appended into a list that later feeds a
rich comparison.
"""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.continual_multiagent_artifact import (
    _validate_seed_and_aggregate_consistency,
)

pytestmark = pytest.mark.unit


class _HostileInt(int):
    """An ``int`` subclass whose comparison dunders must never run."""

    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile __eq__ ran")

    def __ne__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile __ne__ ran")

    def __ge__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile __ge__ ran")

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile __lt__ ran")

    def __hash__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile __hash__ ran")


def _joint_adaptive_condition(recovery: object) -> dict[str, object]:
    return {
        "learning_mask": [True, True],
        "prequential_reward": 0.0,
        "recurrence_recovery_steps": recovery,
    }


def test_seed_gate_rejects_hostile_int_before_comparison() -> None:
    hostile = _HostileInt(30)
    _HostileInt.calls = 0
    content = {
        "seed_summaries": [{"seed": hostile, "conditions": {}}],
        "aggregate": {},
        "configuration": {},
    }
    errors: list[str] = []

    _validate_seed_and_aggregate_consistency(content, errors)

    assert _HostileInt.calls == 0
    assert any("seed must be an integer" in error for error in errors)


def test_seed_gate_rejects_bool() -> None:
    content = {
        "seed_summaries": [{"seed": True, "conditions": {}}],
        "aggregate": {},
        "configuration": {},
    }
    errors: list[str] = []

    _validate_seed_and_aggregate_consistency(content, errors)

    assert any("seed must be an integer" in error for error in errors)


def test_seed_gate_accepts_genuine_int() -> None:
    content = {
        "seed_summaries": [{"seed": 30, "conditions": {}}],
        "aggregate": {},
        "configuration": {},
    }
    errors: list[str] = []

    _validate_seed_and_aggregate_consistency(content, errors)

    assert not any("seed must be an integer" in error for error in errors)


def test_recovery_gate_rejects_hostile_int_before_comparison() -> None:
    hostile = _HostileInt(3)
    _HostileInt.calls = 0
    content = {
        "seed_summaries": [
            {
                "seed": 30,
                "conditions": {
                    "frozen": {"learning_mask": [False, False], "prequential_reward": 0.0},
                    "learner_only": {
                        "learning_mask": [True, False],
                        "prequential_reward": 0.0,
                    },
                    "joint_adaptive": _joint_adaptive_condition(hostile),
                },
            }
        ],
        "aggregate": {},
        "configuration": {},
    }
    errors: list[str] = []

    _validate_seed_and_aggregate_consistency(content, errors)

    assert _HostileInt.calls == 0
    assert any("recurrence_recovery_steps must be an integer" in error for error in errors)


def test_recovery_gate_rejects_bool() -> None:
    content = {
        "seed_summaries": [
            {
                "seed": 30,
                "conditions": {
                    "frozen": {"learning_mask": [False, False], "prequential_reward": 0.0},
                    "learner_only": {
                        "learning_mask": [True, False],
                        "prequential_reward": 0.0,
                    },
                    "joint_adaptive": _joint_adaptive_condition(True),
                },
            }
        ],
        "aggregate": {},
        "configuration": {},
    }
    errors: list[str] = []

    _validate_seed_and_aggregate_consistency(content, errors)

    assert any("recurrence_recovery_steps must be an integer" in error for error in errors)


def test_recovery_gate_accepts_genuine_int() -> None:
    content = {
        "seed_summaries": [
            {
                "seed": 30,
                "conditions": {
                    "frozen": {"learning_mask": [False, False], "prequential_reward": 0.0},
                    "learner_only": {
                        "learning_mask": [True, False],
                        "prequential_reward": 0.0,
                    },
                    "joint_adaptive": _joint_adaptive_condition(3),
                },
            }
        ],
        "aggregate": {},
        "configuration": {},
    }
    errors: list[str] = []

    _validate_seed_and_aggregate_consistency(content, errors)

    assert not any(
        "recurrence_recovery_steps must be an integer" in error for error in errors
    )
