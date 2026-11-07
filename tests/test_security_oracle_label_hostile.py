"""Hostile string validation for security oracle label."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from alberta_framework.security import (
    SecurityAction,
    SecurityFeatureSchema,
    SecurityOracleExperience,
    validate_security_oracle_experience,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    __hash__ = str.__hash__

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool executed")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq executed")

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile len executed")

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("hostile str executed")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("hostile repr executed")


def _schema() -> SecurityFeatureSchema:
    return SecurityFeatureSchema(names=("x",))


def test_validate_oracle_experience_rejects_hostile_label_before_bool() -> None:
    schema = _schema()
    hostile = _HostileStr("safe")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="exact JSON values"):
        record = SecurityOracleExperience(
            state=(0.5,),
            action=SecurityAction.PASS,
            reward=0.0,
            outcome={"label": hostile},  # type: ignore[dict-item]
        )
        validate_security_oracle_experience([record], schema)
    assert _HostileStr.calls == 0


def test_validate_oracle_experience_accepts_builtin_label() -> None:
    schema = _schema()
    record = SecurityOracleExperience(
        state=(0.5,),
        action=SecurityAction.PASS,
        reward=0.0,
        outcome={"label": "safe"},
    )
    # Should not raise for valid builtin label
    validate_security_oracle_experience([record], schema)


def test_validate_oracle_experience_rejects_empty_label() -> None:
    schema = _schema()
    record = SecurityOracleExperience(
        state=(0.5,),
        action=SecurityAction.PASS,
        reward=0.0,
        outcome={"label": ""},
    )
    with pytest.raises(ValueError, match="missing outcome label"):
        validate_security_oracle_experience([record], schema)


def test_hostile_empty_string_still_rejects_before_len() -> None:
    # Empty hostile subclass with empty value; type check should fail before len/bool
    hostile = _HostileStr("")
    _HostileStr.calls = 0
    schema = _schema()
    with pytest.raises(ValueError, match="exact JSON values"):
        record = SecurityOracleExperience(
            state=(0.5,),
            action=SecurityAction.PASS,
            reward=0.0,
            outcome={"label": hostile},  # type: ignore[dict-item]
        )
        validate_security_oracle_experience([record], schema)
    assert _HostileStr.calls == 0


@pytest.mark.parametrize(
    ("action", "label"),
    [
        (SecurityAction.PASS, "true_positive"),
        (SecurityAction.PASS, "false_positive"),
        (SecurityAction.ALERT, "true_positive"),
        (SecurityAction.ALERT, "false_positive"),
        (SecurityAction.UNBLOCK, "true_positive"),
        (SecurityAction.UNBLOCK, "false_positive"),
        (SecurityAction.THROTTLE, "false_negative"),
        (SecurityAction.THROTTLE, "true_negative"),
        (SecurityAction.BLOCK, "false_negative"),
        (SecurityAction.BLOCK, "true_negative"),
        (SecurityAction.ISOLATE, "false_negative"),
        (SecurityAction.ISOLATE, "true_negative"),
    ],
)
def test_oracle_experience_rejects_confusion_label_impossible_for_action(
    action: SecurityAction, label: str
) -> None:
    """A confusion-matrix label outside the action's possible set is rejected.

    Defensive actions (THROTTLE/BLOCK/ISOLATE) can only be true/false
    positive; non-defensive actions (PASS/ALERT/UNBLOCK) can only be
    false/true negative — whichever the ground truth, so the check needs
    no ``is_malicious`` metadata.
    """
    with pytest.raises(ValueError, match="impossible for action"):
        SecurityOracleExperience(
            state=(0.5,),
            action=action,
            reward=0.0,
            outcome={"label": label, "terminated": False, "truncated": False},
        )


@pytest.mark.parametrize(
    ("action", "label"),
    [
        (SecurityAction.PASS, "false_negative"),
        (SecurityAction.PASS, "true_negative"),
        (SecurityAction.ALERT, "false_negative"),
        (SecurityAction.ALERT, "true_negative"),
        (SecurityAction.UNBLOCK, "false_negative"),
        (SecurityAction.UNBLOCK, "true_negative"),
        (SecurityAction.THROTTLE, "true_positive"),
        (SecurityAction.THROTTLE, "false_positive"),
        (SecurityAction.BLOCK, "true_positive"),
        (SecurityAction.BLOCK, "false_positive"),
        (SecurityAction.ISOLATE, "true_positive"),
        (SecurityAction.ISOLATE, "false_positive"),
    ],
)
def test_oracle_experience_accepts_confusion_label_possible_for_action(
    action: SecurityAction, label: str
) -> None:
    schema = _schema()
    record = SecurityOracleExperience(
        state=(0.5,),
        action=action,
        reward=0.0,
        outcome={"label": label, "terminated": False, "truncated": False},
    )
    validate_security_oracle_experience([record], schema)


def test_validate_oracle_experience_rejects_forged_impossible_label() -> None:
    """A forged record with an impossible label is caught on revalidation."""
    schema = _schema()
    record = SecurityOracleExperience(
        state=(0.5,),
        action=SecurityAction.PASS,
        reward=0.0,
        outcome={"label": "safe"},
    )
    object.__setattr__(
        record,
        "outcome",
        MappingProxyType({"label": "true_positive", "terminated": False, "truncated": False}),
    )
    with pytest.raises(ValueError, match="impossible for action"):
        validate_security_oracle_experience([record], schema)
