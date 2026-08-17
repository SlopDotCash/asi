"""Tests for security-gym / rlsecd integration contracts."""

import json
import math

import pytest

from alberta_framework import (
    N_SECURITY_ACTIONS,
    SECURITY_ACTION_NAMES,
    SECURITY_GYM_ACTION_NAMES,
    SecurityAction,
    SecurityFeatureSchema,
    SecurityRewardWeights,
    SecurityRolloutStep,
    ThroughputMeter,
    coerce_security_action,
    security_gym_action_name,
    security_gym_action_reward,
    security_reward,
    to_security_gym_action,
    validate_security_rollout,
)
from alberta_framework.security import (
    SecurityOracleExperience,
    ThroughputMeasurement,
)


def test_security_action_indices_are_stable() -> None:
    assert N_SECURITY_ACTIONS == 6
    assert SECURITY_ACTION_NAMES == (
        "pass",
        "alert",
        "throttle",
        "block",
        "unblock",
        "isolate",
    )
    assert [int(action) for action in SecurityAction] == list(range(6))
    assert coerce_security_action("block") == SecurityAction.BLOCK
    assert coerce_security_action("block_source") == SecurityAction.BLOCK
    assert coerce_security_action(5) == SecurityAction.ISOLATE


@pytest.mark.parametrize("action", [True, False, 1.0, 99])
def test_coerce_security_action_rejects_noncanonical_actions(action: object) -> None:
    with pytest.raises(ValueError, match="security action"):
        coerce_security_action(action)  # type: ignore[arg-type]


def test_to_security_gym_action_rejects_boolean_and_nonfinite_risk() -> None:
    with pytest.raises(ValueError, match="risk_score"):
        to_security_gym_action("pass", risk_score=True)
    with pytest.raises(ValueError, match="risk_score"):
        to_security_gym_action("pass", risk_score=float("nan"))
    with pytest.raises(ValueError, match="risk_score"):
        to_security_gym_action("pass", risk_score=float("inf"))


def test_security_gym_action_adapter_matches_sibling_contract() -> None:
    assert SECURITY_GYM_ACTION_NAMES == (
        "pass",
        "alert",
        "throttle",
        "block_source",
        "unblock",
        "isolate",
    )
    assert security_gym_action_name(SecurityAction.BLOCK) == "block_source"
    assert to_security_gym_action("block_source", risk_score=11.0) == {
        "action": 3,
        "risk_score": (10.0,),
    }
    assert security_gym_action_reward(SecurityAction.BLOCK, is_malicious=True) == pytest.approx(
        1.0
    )
    assert security_gym_action_reward(SecurityAction.BLOCK, is_malicious=False) == pytest.approx(
        -1.0
    )


def test_security_reward_uses_named_components() -> None:
    weights = SecurityRewardWeights(
        threat_blocked=2.0,
        false_positive=-1.0,
        service_disruption=-0.25,
        alert_cost=-0.1,
        latency_cost=0.0,
        compromise_cost=-3.0,
        recovery=0.5,
    )
    reward = security_reward(
        {
            "threat_blocked": 1.0,
            "false_positive": 0.0,
            "alert_cost": 1.0,
            "unknown_diagnostic": 100.0,
        },
        weights,
    )
    assert reward == pytest.approx(1.9)


def test_feature_schema_roundtrip_and_validation() -> None:
    schema = SecurityFeatureSchema(names=("src_reputation", "dst_port_risk"))
    assert schema.feature_dim == 2
    schema.validate_observation((0.1, 0.2))

    restored = SecurityFeatureSchema.from_dict(schema.to_dict())
    assert restored == schema

    with pytest.raises(ValueError, match="observation length"):
        schema.validate_observation((0.1,))


def test_rollout_step_roundtrip_and_validation() -> None:
    schema = SecurityFeatureSchema(names=("x0", "x1"))
    step = SecurityRolloutStep(
        state=(0.0, 1.0),
        action=SecurityAction.THROTTLE,
        reward=-0.2,
        next_state=(0.5, 1.0),
        terminated=False,
        policy_metadata={"epsilon": 0.05, "q_values": [0.0, 0.1, 0.2, 0.0, 0.0, 0.0]},
    )

    restored = SecurityRolloutStep.from_dict(step.to_dict())
    assert restored == step
    validate_security_rollout([restored], schema)

    invalid = SecurityRolloutStep(
        state=(0.0,),
        action=SecurityAction.PASS,
        reward=0.0,
        next_state=(0.0, 1.0),
        terminated=False,
    )
    with pytest.raises(ValueError, match="invalid rollout step 0"):
        validate_security_rollout([invalid], schema)


def test_throughput_meter_records_events() -> None:
    meter = ThroughputMeter()
    meter.tick(3)
    measurement = meter.measure()

    assert measurement.n_events == 3
    assert measurement.elapsed_s >= 0.0
    assert measurement.events_per_second > 0.0
    assert measurement.to_dict()["n_events"] == 3


@pytest.mark.parametrize("elapsed_s", [0.0, -1.0, float("-inf")])
def test_throughput_measurement_to_dict_rejects_nonpositive_elapsed(
    elapsed_s: float,
) -> None:
    measurement = ThroughputMeasurement(n_events=3, elapsed_s=elapsed_s)
    assert measurement.events_per_second == float("inf")
    leaked = json.dumps(
        {
            "n_events": measurement.n_events,
            "elapsed_s": measurement.elapsed_s,
            "events_per_second": measurement.events_per_second,
        }
    )
    assert "Infinity" in leaked
    with pytest.raises(ValueError, match="RFC-compliant JSON"):
        measurement.to_dict()


def test_throughput_measurement_to_dict_rejects_nan_elapsed() -> None:
    measurement = ThroughputMeasurement(n_events=3, elapsed_s=float("nan"))
    assert math.isnan(measurement.events_per_second)
    leaked = json.dumps(
        {
            "n_events": measurement.n_events,
            "elapsed_s": measurement.elapsed_s,
            "events_per_second": measurement.events_per_second,
        }
    )
    assert "NaN" in leaked
    with pytest.raises(ValueError, match="RFC-compliant JSON"):
        measurement.to_dict()


def test_throughput_measurement_to_dict_is_strict_json() -> None:
    payload = ThroughputMeasurement(n_events=4, elapsed_s=2.0).to_dict()
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == {
        "elapsed_s": 2.0,
        "events_per_second": 2.0,
        "n_events": 4,
    }
    assert "Infinity" not in encoded
    assert "NaN" not in encoded


def test_security_to_dict_rejects_nonfinite_numbers() -> None:
    weights = SecurityRewardWeights(threat_blocked=float("inf"))
    step = SecurityRolloutStep(
        state=(0.0, 1.0),
        action=SecurityAction.PASS,
        reward=float("nan"),
        next_state=(0.0, 1.0),
        terminated=False,
    )
    experience = SecurityOracleExperience(
        state=(0.0, 1.0),
        action=SecurityAction.PASS,
        reward=0.0,
        outcome={"label": "true_negative", "score": float("nan")},
    )
    with pytest.raises(ValueError, match="RFC-compliant JSON"):
        weights.to_dict()
    with pytest.raises(ValueError, match="RFC-compliant JSON"):
        step.to_dict()
    with pytest.raises(ValueError, match="RFC-compliant JSON"):
        experience.to_dict()
