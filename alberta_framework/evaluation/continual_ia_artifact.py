"""Versioned, internally bound evidence artifacts for continual IA.

The SHA-256 digest covers deterministic scientific content: frozen protocol,
configuration and thresholds, primitive per-step records, recomputed
aggregates and paired intervals, acceptance checks, and relevant source hashes.
Host environment and wall-clock timing remain inspectable outside that digest.

The validator does not trust declared summaries.  It reconstructs condition
results from primitive traces, re-derives action-changing interventions and
executed-action credit, recomputes paired aggregates and confidence intervals,
and binds acceptance checks to that evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import jax
import numpy as np
from numpy.typing import NDArray

from alberta_framework.evaluation.continual_ia import (
    CONDITION_NAMES,
    PROMOTED_EVIDENCE_SEEDS,
    RECOMMENDATION_CONDITIONS,
    ConditionTiming,
    ContinualIAConfig,
    ContinualIAReport,
    ControllerBudget,
    IAAcceptanceEvidence,
    IAAcceptanceResult,
    IAAcceptanceThresholds,
    IAAggregateEvidence,
    IAConditionName,
    IAConditionResult,
    PairedBootstrapInterval,
    _phase_means,
    _recovery_lengths,
    aggregate_ia_evidence,
    condition_controller_budgets,
    evaluate_ia_acceptance,
)

SCHEMA_VERSION = "alberta.continual_ia_evidence.v1"
PROTOCOL_VERSION = "step12-hidden-phase-causal-ia.v1"
DIGEST_SCOPE = "$.content"
CANONICALIZATION = "utf8-json-sort-keys-compact-no-nan"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATHS = (
    Path("alberta_framework/evaluation/continual_ia.py"),
    Path("alberta_framework/evaluation/continual_ia_artifact.py"),
    Path("alberta_framework/evaluation/continual_ia_cli.py"),
    Path("alberta_framework/core/intelligence_amplification.py"),
    Path("alberta_framework/core/average_reward.py"),
    Path("alberta_framework/core/oak.py"),
    Path("alberta_framework/core/options.py"),
    Path("alberta_framework/streams/closed_loop.py"),
)


@dataclass(frozen=True)
class IAArtifactValidation:
    """Fail-closed schema, integrity, internal-binding, and acceptance result."""

    valid: bool
    accepted: bool
    errors: tuple[str, ...]


def _finite_float(value: float) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _config_payload(config: ContinualIAConfig) -> dict[str, object]:
    return {
        "num_steps": config.num_steps,
        "phase_length": config.phase_length,
        "observation_dim": config.observation_dim,
        "n_actions": config.n_actions,
        "n_demons": config.n_demons,
        "partner_q_step_size": config.partner_q_step_size,
        "partner_average_reward_step_size": (config.partner_average_reward_step_size),
        "partner_epsilon": config.partner_epsilon,
        "cortex_base_step_size": config.cortex_base_step_size,
        "recommendation_acceptance_probability": (config.recommendation_acceptance_probability),
        "recovery_window": config.recovery_window,
        "recovery_mean_reward_threshold": (config.recovery_mean_reward_threshold),
        "bootstrap_resamples": config.bootstrap_resamples,
        "confidence_level": config.confidence_level,
        "bootstrap_seed": config.bootstrap_seed,
    }


def _threshold_payload(
    thresholds: IAAcceptanceThresholds,
) -> dict[str, object]:
    return {
        "minimum_seed_count": thresholds.minimum_seed_count,
        "evidence_seed_start": thresholds.evidence_seed_start,
        "minimum_primary_uplift_lower_ci": (thresholds.minimum_primary_uplift_lower_ci),
        "minimum_changed_action_intervention_rate": (
            thresholds.minimum_changed_action_intervention_rate
        ),
        "minimum_augmentation_vs_alone_lower_ci": (
            thresholds.minimum_augmentation_vs_alone_lower_ci
        ),
        "minimum_augmentation_vs_noise_lower_ci": (
            thresholds.minimum_augmentation_vs_noise_lower_ci
        ),
        "require_observe_only_exact_identity": (thresholds.require_observe_only_exact_identity),
        "maximum_executed_action_credit_mismatches": (
            thresholds.maximum_executed_action_credit_mismatches
        ),
    }


def _budget_payload(budget: ControllerBudget) -> dict[str, object]:
    return {
        "state_scalars": budget.state_scalars,
        "state_bytes": budget.state_bytes,
        "observation_scalars": budget.observation_scalars,
        "action_scalars_per_step": budget.action_scalars_per_step,
        "interaction_steps": budget.interaction_steps,
        "ia_attached": budget.ia_attached,
    }


def _interval_payload(interval: PairedBootstrapInterval) -> dict[str, object]:
    return {
        "estimate": _finite_float(interval.estimate),
        "lower": _finite_float(interval.lower),
        "upper": _finite_float(interval.upper),
        "confidence_level": interval.confidence_level,
        "resamples": interval.resamples,
        "sample_size": interval.sample_size,
        "method": interval.method,
        "pairing_unit": "seed",
    }


def _condition_payload(result: IAConditionResult) -> dict[str, object]:
    return {
        "primitive_records": {
            "rewards": [float(value) for value in result.rewards],
            "executed_actions": [int(value) for value in result.executed_actions],
            "credited_actions": [int(value) for value in result.credited_actions],
            "recommendations": [int(value) for value in result.recommendations],
            "partner_proposals": [int(value) for value in result.partner_proposals],
            "accepted_recommendations": [bool(value) for value in result.accepted_recommendations],
        },
        "summary": {
            "mean_reward": result.mean_reward,
            "phase_mean_rewards": [float(value) for value in result.phase_mean_rewards],
            "recovery_lengths": [int(value) for value in result.recovery_lengths],
            "nominal_recommendation_decisions": (result.nominal_recommendation_decisions),
            "nominal_accepted_recommendations": (result.nominal_accepted_recommendations),
            "executed_accepted_recommendations": (result.executed_accepted_recommendations),
            "action_changing_interventions": (result.action_changing_interventions),
            "changed_action_intervention_rate": (result.changed_action_intervention_rate),
            "executed_action_credit_mismatches": (result.executed_action_credit_mismatches),
            "controller_budget": _budget_payload(result.controller_budget),
        },
    }


def _seed_payloads(
    results: tuple[IAConditionResult, ...],
) -> list[dict[str, object]]:
    grouped: dict[int, dict[IAConditionName, IAConditionResult]] = {}
    for result in results:
        conditions = grouped.setdefault(result.seed, {})
        if result.condition in conditions:
            raise ValueError(f"duplicate condition {result.condition} for seed {result.seed}")
        conditions[result.condition] = result
    payloads: list[dict[str, object]] = []
    for seed in sorted(grouped):
        conditions = grouped[seed]
        if set(conditions) != set(CONDITION_NAMES):
            raise ValueError(f"seed {seed} does not contain every condition")
        payloads.append(
            {
                "seed": seed,
                "conditions": {
                    condition: _condition_payload(conditions[condition])
                    for condition in CONDITION_NAMES
                },
            }
        )
    return payloads


def _aggregate_payload(aggregate: IAAggregateEvidence) -> dict[str, object]:
    return {
        "seeds": list(aggregate.seeds),
        "seed_count": len(aggregate.seeds),
        "mean_rewards": {
            condition: aggregate.mean_rewards[condition] for condition in CONDITION_NAMES
        },
        "primary_uplift": aggregate.primary_uplift,
        "primary_uplift_interval": _interval_payload(aggregate.primary_uplift_interval),
        "accept_always_uplift": aggregate.accept_always_uplift,
        "accept_always_uplift_interval": _interval_payload(aggregate.accept_always_uplift_interval),
        "augmentation_vs_alone": aggregate.augmentation_vs_alone,
        "augmentation_vs_alone_interval": _interval_payload(
            aggregate.augmentation_vs_alone_interval
        ),
        "augmentation_vs_noise": aggregate.augmentation_vs_noise,
        "augmentation_vs_noise_interval": _interval_payload(
            aggregate.augmentation_vs_noise_interval
        ),
        "mean_changed_action_intervention_rate": (aggregate.mean_changed_action_intervention_rate),
        "total_action_changing_interventions": (aggregate.total_action_changing_interventions),
        "observe_only_exact_reward_identity": (aggregate.observe_only_exact_reward_identity),
        "observe_only_exact_action_identity": (aggregate.observe_only_exact_action_identity),
        "executed_action_credit_mismatches": (aggregate.executed_action_credit_mismatches),
        "primary_state_budget_matched": aggregate.primary_state_budget_matched,
        "primary_interaction_budget_matched": (aggregate.primary_interaction_budget_matched),
        "augmentation_noise_state_budget_matched": (
            aggregate.augmentation_noise_state_budget_matched
        ),
        "augmentation_state_bytes_above_alone": (aggregate.augmentation_state_bytes_above_alone),
        "condition_budgets": {
            condition: _budget_payload(aggregate.condition_budgets[condition])
            for condition in CONDITION_NAMES
        },
        "condition_phase_mean_rewards": {
            condition: [float(value) for value in aggregate.condition_phase_mean_rewards[condition]]
            for condition in CONDITION_NAMES
        },
        "condition_recovery_fraction": {
            condition: aggregate.condition_recovery_fraction[condition]
            for condition in CONDITION_NAMES
        },
        "condition_mean_recovery_steps": {
            condition: _finite_float(aggregate.condition_mean_recovery_steps[condition])
            for condition in CONDITION_NAMES
        },
        "all_values_finite": aggregate.all_values_finite,
    }


def _check_payload(check: IAAcceptanceEvidence) -> dict[str, object]:
    return {
        "name": check.name,
        "scope": check.scope,
        "passed": check.passed,
        "actual": _finite_float(check.actual),
        "comparator": check.comparator,
        "threshold": _finite_float(check.threshold),
        "detail": check.detail,
    }


def _acceptance_payload(
    acceptance: IAAcceptanceResult,
) -> dict[str, object]:
    return {
        "passed": acceptance.passed,
        "primary_passed": acceptance.primary_passed,
        "secondary_passed": acceptance.secondary_passed,
        "checks": [_check_payload(check) for check in acceptance.checks],
    }


def _protocol_payload() -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "primary_estimand": (
            "paired mean reward of recommendation_p05 minus observe_only; "
            "both arms run identical IA computation/state and credit the "
            "executed primitive action"
        ),
        "action_changing_intervention": (
            "an executed primitive transition whose accepted recommendation "
            "differs from the partner proposal"
        ),
        "conditions": {
            "partner_alone": "partner without IA",
            "observe_only": "IA runs and learns; p_accept=0",
            "recommendation_p05": "IA recommendation treatment; p_accept=0.5",
            "accept_always": (
                "p_accept=1 negative/full-delegation diagnostic; no improvement required"
            ),
            "augmented_predictions": (
                "partner observes raw state plus learned cerebellum predictions"
            ),
            "augmented_noise": (
                "same partner/IA state shape with predictions replaced by uniform noise"
            ),
        },
        "seed_roles": {
            "development_and_calibration": list(range(12)),
            "promoted_held_out_evidence": list(PROMOTED_EVIDENCE_SEEDS),
            "rule": (
                "p=.5 and all thresholds were frozen after seeds 0-11; "
                "seeds 30-59 are evaluated without retuning"
            ),
        },
        "budget_interpretation": (
            "recommendation_p05 and observe_only have identical state and "
            "interaction budgets; augmented_predictions and augmented_noise "
            "have equal budgets to each other but intentionally exceed partner_alone"
        ),
        "supported_claim": (
            "causal recommendation-channel uplift for the selected IA/partner "
            "pair in this deterministic hidden-phase micro-MDP"
        ),
        "limitations": [
            "deterministic hand-designed two-state hidden-phase MDP",
            "p_accept=0.5 and hyperparameters selected after development seeds 0-11",
            "no general partner, environment-family, or population claim",
            "no autonomous feature discovery",
            "not completion of Step 12 across realistic partners",
            "not completion of the Alberta Plan",
            "digest and source hashes provide integrity, not origin authentication",
        ],
    }


def _source_provenance() -> dict[str, object]:
    return {
        "repository_subtree": "research/alberta",
        "source_sha256": {
            relative.as_posix(): hashlib.sha256((_REPO_ROOT / relative).read_bytes()).hexdigest()
            for relative in _SOURCE_PATHS
        },
    }


def _environment_payload() -> dict[str, object]:
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "jax": jax.__version__,
            "jaxlib": importlib.metadata.version("jaxlib"),
            "numpy": np.__version__,
        },
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [
            {
                "platform": device.platform,
                "device_kind": device.device_kind,
            }
            for device in jax.devices()
        ],
    }


def _operational_payload(report: ContinualIAReport) -> dict[str, object]:
    return {
        "digest_exclusion_reason": ("host environment and wall-clock timing are non-deterministic"),
        "environment": _environment_payload(),
        "condition_timings": [
            {
                "seed": result.seed,
                "condition": result.condition,
                "wall_seconds": result.timing.wall_seconds,
                "mean_step_latency_ms": result.timing.mean_step_latency_ms,
            }
            for result in report.condition_results
        ],
        "overall_acceptance_passed": report.acceptance.passed,
    }


def canonical_content_bytes(content: Mapping[str, object]) -> bytes:
    return json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def scientific_content_sha256(content: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_content_bytes(content)).hexdigest()


def build_ia_evidence_artifact(
    report: ContinualIAReport,
) -> dict[str, object]:
    """Build one deterministic-content evidence candidate."""

    if tuple(sorted(report.aggregate.seeds)) != report.aggregate.seeds:
        raise ValueError("artifact seeds must be strictly increasing")
    content: dict[str, object] = {
        "protocol": _protocol_payload(),
        "configuration": _config_payload(report.config),
        "thresholds": _threshold_payload(report.thresholds),
        "seed_summaries": _seed_payloads(report.condition_results),
        "aggregate": _aggregate_payload(report.aggregate),
        "acceptance": _acceptance_payload(report.acceptance),
        "source_provenance": _source_provenance(),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "content": content,
        "operational_diagnostics": _operational_payload(report),
        "content_digest": {
            "algorithm": "sha256",
            "scope": DIGEST_SCOPE,
            "canonicalization": CANONICALIZATION,
            "sha256": scientific_content_sha256(content),
        },
    }


def ia_artifact_json(artifact: Mapping[str, object]) -> str:
    return (
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _reject_nonstandard_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON object key: {key}")
        parsed[key] = value
    return parsed


def load_ia_evidence_artifact(path: Path) -> dict[str, object]:
    parsed = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_constant,
        parse_float=_parse_finite_json_float,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(parsed, dict):
        raise ValueError("IA evidence artifact must be a JSON object")
    return parsed


def write_ia_evidence_artifact(
    path: Path,
    report: ContinualIAReport,
) -> dict[str, object]:
    """Exclusively create an internally valid evidence artifact."""

    artifact = build_ia_evidence_artifact(report)
    validation = validate_ia_evidence_artifact(artifact)
    if not validation.valid:
        raise ValueError(
            "refusing to write invalid IA evidence artifact: " + "; ".join(validation.errors)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(ia_artifact_json(artifact))
    return artifact


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _numbers_match(first: object, second: object) -> bool:
    left = _number(first)
    right = _number(second)
    return (
        left is not None
        and right is not None
        and bool(np.isclose(left, right, rtol=0.0, atol=1e-12))
    )


def _numeric_vector(
    value: object,
    *,
    length: int,
    location: str,
    errors: list[str],
) -> NDArray[np.float64] | None:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(_number(item) is None for item in value)
    ):
        errors.append(f"{location} must contain {length} finite numbers")
        return None
    return np.asarray(value, dtype=np.float64)


def _integer_vector(
    value: object,
    *,
    length: int,
    location: str,
    allowed: set[int],
    errors: list[str],
) -> NDArray[np.int64] | None:
    if not isinstance(value, list) or len(value) != length:
        errors.append(f"{location} must contain {length} integers")
        return None
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item not in allowed for item in value
    ):
        errors.append(f"{location} contains an invalid integer")
        return None
    return np.asarray(value, dtype=np.int64)


def _boolean_vector(
    value: object,
    *,
    length: int,
    location: str,
    errors: list[str],
) -> NDArray[np.bool_] | None:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(not isinstance(item, bool) for item in value)
    ):
        errors.append(f"{location} must contain {length} booleans")
        return None
    return np.asarray(value, dtype=np.bool_)


def _integer(
    value: object,
    *,
    location: str,
    errors: list[str],
) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{location} must be an integer")
        return None
    return value


def _parse_budget(
    value: object,
    *,
    location: str,
    errors: list[str],
) -> ControllerBudget | None:
    if not isinstance(value, Mapping):
        errors.append(f"{location} must be an object")
        return None
    expected_fields = {
        "state_scalars",
        "state_bytes",
        "observation_scalars",
        "action_scalars_per_step",
        "interaction_steps",
        "ia_attached",
    }
    if set(value) != expected_fields:
        errors.append(f"{location} keys do not match the v1 schema")
    integer_fields = (
        "state_scalars",
        "state_bytes",
        "observation_scalars",
        "action_scalars_per_step",
        "interaction_steps",
    )
    parsed: dict[str, int] = {}
    for field_name in integer_fields:
        field_value = _integer(
            value.get(field_name),
            location=f"{location}.{field_name}",
            errors=errors,
        )
        if field_value is not None:
            parsed[field_name] = field_value
    attached = value.get("ia_attached")
    if not isinstance(attached, bool):
        errors.append(f"{location}.ia_attached must be boolean")
    if len(parsed) != len(integer_fields) or not isinstance(attached, bool):
        return None
    try:
        return ControllerBudget(
            state_scalars=parsed["state_scalars"],
            state_bytes=parsed["state_bytes"],
            observation_scalars=parsed["observation_scalars"],
            action_scalars_per_step=parsed["action_scalars_per_step"],
            interaction_steps=parsed["interaction_steps"],
            ia_attached=attached,
        )
    except ValueError as error:
        errors.append(f"{location} is invalid: {error}")
        return None


def _parse_config(
    value: object,
    errors: list[str],
) -> ContinualIAConfig | None:
    if not isinstance(value, Mapping):
        errors.append("content.configuration must be an object")
        return None
    canonical_payload = _config_payload(ContinualIAConfig())
    if set(value) != set(canonical_payload) or any(
        type(value.get(field_name)) is not type(expected) or value.get(field_name) != expected
        for field_name, expected in canonical_payload.items()
    ):
        errors.append("content.configuration must equal the frozen v1 protocol")
    try:
        config = ContinualIAConfig(
            **cast(dict[str, Any], dict(value)),
        )
    except (TypeError, ValueError) as error:
        errors.append(f"invalid content.configuration: {error}")
        return None
    return config


def _parse_thresholds(
    value: object,
    errors: list[str],
) -> IAAcceptanceThresholds | None:
    if not isinstance(value, Mapping):
        errors.append("content.thresholds must be an object")
        return None
    canonical_payload = _threshold_payload(IAAcceptanceThresholds())
    if set(value) != set(canonical_payload) or any(
        type(value.get(field_name)) is not type(expected) or value.get(field_name) != expected
        for field_name, expected in canonical_payload.items()
    ):
        errors.append("content.thresholds must equal the exact frozen v1 thresholds")
    try:
        thresholds = IAAcceptanceThresholds(
            **cast(dict[str, Any], dict(value)),
        )
    except (TypeError, ValueError) as error:
        errors.append(f"invalid content.thresholds: {error}")
        return None
    return thresholds


def _compare_claimed_summary(
    summary: Mapping[str, object],
    result: IAConditionResult,
    *,
    location: str,
    errors: list[str],
) -> None:
    numeric_expected = {
        "mean_reward": result.mean_reward,
        "changed_action_intervention_rate": (result.changed_action_intervention_rate),
    }
    for field_name, expected in numeric_expected.items():
        if not _numbers_match(summary.get(field_name), expected):
            errors.append(f"{location}.{field_name} is inconsistent with primitives")
    integer_expected = {
        "executed_accepted_recommendations": (result.executed_accepted_recommendations),
        "action_changing_interventions": (result.action_changing_interventions),
        "executed_action_credit_mismatches": (result.executed_action_credit_mismatches),
    }
    for field_name, expected in integer_expected.items():
        if summary.get(field_name) != expected:
            errors.append(f"{location}.{field_name} is inconsistent with primitives")
    if summary.get("phase_mean_rewards") != [float(value) for value in result.phase_mean_rewards]:
        errors.append(f"{location}.phase_mean_rewards is inconsistent")
    if summary.get("recovery_lengths") != [int(value) for value in result.recovery_lengths]:
        errors.append(f"{location}.recovery_lengths is inconsistent")


def _parse_condition(
    value: object,
    *,
    seed: int,
    condition: IAConditionName,
    config: ContinualIAConfig,
    location: str,
    errors: list[str],
) -> IAConditionResult | None:
    if not isinstance(value, Mapping):
        errors.append(f"{location} must be an object")
        return None
    if set(value) != {"primitive_records", "summary"}:
        errors.append(f"{location} keys do not match the v1 schema")
    primitive = value.get("primitive_records")
    summary = value.get("summary")
    if not isinstance(primitive, Mapping) or not isinstance(summary, Mapping):
        errors.append(f"{location} requires primitive_records and summary objects")
        return None
    if set(primitive) != {
        "rewards",
        "executed_actions",
        "credited_actions",
        "recommendations",
        "partner_proposals",
        "accepted_recommendations",
    }:
        errors.append(f"{location}.primitive_records keys do not match the v1 schema")
    if set(summary) != {
        "mean_reward",
        "phase_mean_rewards",
        "recovery_lengths",
        "nominal_recommendation_decisions",
        "nominal_accepted_recommendations",
        "executed_accepted_recommendations",
        "action_changing_interventions",
        "changed_action_intervention_rate",
        "executed_action_credit_mismatches",
        "controller_budget",
    }:
        errors.append(f"{location}.summary keys do not match the v1 schema")
    rewards = _numeric_vector(
        primitive.get("rewards"),
        length=config.num_steps,
        location=f"{location}.primitive_records.rewards",
        errors=errors,
    )
    actions = _integer_vector(
        primitive.get("executed_actions"),
        length=config.num_steps,
        location=f"{location}.primitive_records.executed_actions",
        allowed={0, 1},
        errors=errors,
    )
    credits = _integer_vector(
        primitive.get("credited_actions"),
        length=config.num_steps,
        location=f"{location}.primitive_records.credited_actions",
        allowed={0, 1},
        errors=errors,
    )
    recommendation_allowed = {0, 1} if condition in RECOMMENDATION_CONDITIONS else {-1}
    recommendations = _integer_vector(
        primitive.get("recommendations"),
        length=config.num_steps,
        location=f"{location}.primitive_records.recommendations",
        allowed=recommendation_allowed,
        errors=errors,
    )
    proposals = _integer_vector(
        primitive.get("partner_proposals"),
        length=config.num_steps,
        location=f"{location}.primitive_records.partner_proposals",
        allowed=recommendation_allowed,
        errors=errors,
    )
    accepted = _boolean_vector(
        primitive.get("accepted_recommendations"),
        length=config.num_steps,
        location=f"{location}.primitive_records.accepted_recommendations",
        errors=errors,
    )
    arrays = (rewards, actions, credits, recommendations, proposals, accepted)
    if any(array is None for array in arrays):
        return None
    assert rewards is not None
    assert actions is not None
    assert credits is not None
    assert recommendations is not None
    assert proposals is not None
    assert accepted is not None

    if condition in RECOMMENDATION_CONDITIONS:
        expected_actions = np.where(accepted, recommendations, proposals)
        if not np.array_equal(actions, expected_actions):
            errors.append(f"{location} executed actions violate recommendation protocol")
        if bool(accepted[0]):
            errors.append(f"{location} first transition cannot have a prior acceptance")
        if recommendations[0] != actions[0] or proposals[0] != actions[0]:
            errors.append(f"{location} first transition provenance must equal the initial action")
    elif np.any(accepted):
        errors.append(f"{location} non-recommendation arm cannot accept recommendations")

    changed = accepted & (recommendations != proposals)
    nominal_decisions = _integer(
        summary.get("nominal_recommendation_decisions"),
        location=f"{location}.summary.nominal_recommendation_decisions",
        errors=errors,
    )
    nominal_accepted = _integer(
        summary.get("nominal_accepted_recommendations"),
        location=f"{location}.summary.nominal_accepted_recommendations",
        errors=errors,
    )
    budget = _parse_budget(
        summary.get("controller_budget"),
        location=f"{location}.summary.controller_budget",
        errors=errors,
    )
    if nominal_decisions is None or nominal_accepted is None or budget is None:
        return None
    if condition in RECOMMENDATION_CONDITIONS:
        if nominal_decisions != config.num_steps:
            errors.append(f"{location} nominal decision count must equal num_steps")
        if not 0 <= nominal_accepted <= nominal_decisions:
            errors.append(f"{location} nominal accepted count is out of range")
        executed_accepted = int(np.count_nonzero(accepted))
        # The primitive trace records each acceptance one step after the
        # decision: a decision at step t selects the action executed at step
        # t + 1, so the run's final decision never appears as an executed
        # transition.  The nominal counter may therefore lead the executed
        # count by exactly one; any other discrepancy is a protocol violation.
        if nominal_accepted not in {
            executed_accepted,
            executed_accepted + 1,
        }:
            errors.append(f"{location} nominal accepted count is not bound to executed primitives")
        if condition == "observe_only" and nominal_accepted != 0:
            errors.append(f"{location} observe-only nominal accepts must be zero")
        if condition == "accept_always" and nominal_accepted != config.num_steps:
            errors.append(f"{location} accept-always must accept every decision")
    elif nominal_decisions != 0 or nominal_accepted != 0:
        errors.append(f"{location} non-recommendation counters must be zero")

    result = IAConditionResult(
        seed=seed,
        condition=condition,
        rewards=rewards,
        executed_actions=actions,
        credited_actions=credits,
        recommendations=recommendations,
        partner_proposals=proposals,
        accepted_recommendations=accepted,
        mean_reward=float(np.mean(rewards)),
        phase_mean_rewards=_phase_means(rewards, config),
        recovery_lengths=_recovery_lengths(rewards, config),
        nominal_recommendation_decisions=nominal_decisions,
        nominal_accepted_recommendations=nominal_accepted,
        executed_accepted_recommendations=int(np.count_nonzero(accepted)),
        action_changing_interventions=int(np.count_nonzero(changed)),
        changed_action_intervention_rate=float(np.mean(changed)),
        executed_action_credit_mismatches=int(np.count_nonzero(actions != credits)),
        controller_budget=budget,
        timing=ConditionTiming(wall_seconds=0.0, mean_step_latency_ms=0.0),
    )
    _compare_claimed_summary(
        summary,
        result,
        location=f"{location}.summary",
        errors=errors,
    )
    return result


def _parse_results(
    value: object,
    *,
    config: ContinualIAConfig,
    errors: list[str],
) -> tuple[IAConditionResult, ...] | None:
    if not isinstance(value, list):
        errors.append("content.seed_summaries must be an array")
        return None
    observed_seeds: list[int] = []
    results: list[IAConditionResult] = []
    for index, raw_seed in enumerate(value):
        location = f"content.seed_summaries[{index}]"
        if not isinstance(raw_seed, Mapping):
            errors.append(f"{location} must be an object")
            continue
        if set(raw_seed) != {"seed", "conditions"}:
            errors.append(f"{location} keys do not match the v1 schema")
        seed = _integer(raw_seed.get("seed"), location=f"{location}.seed", errors=errors)
        conditions = raw_seed.get("conditions")
        if seed is None or not isinstance(conditions, Mapping):
            if not isinstance(conditions, Mapping):
                errors.append(f"{location}.conditions must be an object")
            continue
        observed_seeds.append(seed)
        if set(conditions) != set(CONDITION_NAMES):
            errors.append(f"{location}.conditions must contain exactly all six arms")
        for condition in CONDITION_NAMES:
            result = _parse_condition(
                conditions.get(condition),
                seed=seed,
                condition=condition,
                config=config,
                location=f"{location}.conditions.{condition}",
                errors=errors,
            )
            if result is not None:
                results.append(result)
    seed_schedule_valid = tuple(observed_seeds) == PROMOTED_EVIDENCE_SEEDS
    if not seed_schedule_valid:
        errors.append("content.seed_summaries must contain exactly unique seeds 30-59 in order")
    expected_results = len(PROMOTED_EVIDENCE_SEEDS) * len(CONDITION_NAMES)
    if len(results) != expected_results:
        errors.append(f"content.seed_summaries must yield {expected_results} conditions")
        return None
    if not seed_schedule_valid:
        return None
    expected_budgets = condition_controller_budgets(config)
    budgets_valid = True
    for result in results:
        if result.controller_budget != expected_budgets[result.condition]:
            budgets_valid = False
            errors.append(
                "content seed condition controller budget does not match "
                f"the canonical {result.condition} state"
            )
    if not budgets_valid:
        return None
    return tuple(results)


def _validate_provenance(value: object, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append("content.source_provenance must be an object")
        return
    try:
        expected = _source_provenance()
    except OSError as error:
        errors.append(f"cannot bind source provenance: {error}")
        return
    if value != expected:
        errors.append("content.source_provenance does not match the current pinned sources")


def _validate_environment(value: object, errors: list[str]) -> None:
    location = "operational_diagnostics.environment"
    if not isinstance(value, Mapping):
        errors.append(f"{location} must be an object")
        return
    if set(value) != {
        "python",
        "platform",
        "packages",
        "jax_default_backend",
        "jax_devices",
    }:
        errors.append(f"{location} keys do not match the v1 schema")
    expected_string_mappings = {
        "python": {"implementation", "version"},
        "platform": {"system", "release", "machine"},
        "packages": {"jax", "jaxlib", "numpy"},
    }
    for field_name, expected_keys in expected_string_mappings.items():
        field = value.get(field_name)
        if (
            not isinstance(field, Mapping)
            or set(field) != expected_keys
            or any(
                not isinstance(field.get(key), str) or not field.get(key) for key in expected_keys
            )
        ):
            errors.append(f"{location}.{field_name} is invalid")
    backend = value.get("jax_default_backend")
    if not isinstance(backend, str) or not backend:
        errors.append(f"{location}.jax_default_backend is invalid")
    devices = value.get("jax_devices")
    if not isinstance(devices, list) or not devices:
        errors.append(f"{location}.jax_devices must be a non-empty array")
        return
    for index, device in enumerate(devices):
        if (
            not isinstance(device, Mapping)
            or set(device) != {"platform", "device_kind"}
            or not isinstance(device.get("platform"), str)
            or not device.get("platform")
            or not isinstance(device.get("device_kind"), str)
            or not device.get("device_kind")
        ):
            errors.append(f"{location}.jax_devices[{index}] is invalid")


def _validate_operational(
    value: object,
    *,
    expected_results: tuple[IAConditionResult, ...] | None,
    expected_acceptance: bool,
    errors: list[str],
) -> None:
    if not isinstance(value, Mapping):
        errors.append("operational_diagnostics must be an object")
        return
    if set(value) != {
        "digest_exclusion_reason",
        "environment",
        "condition_timings",
        "overall_acceptance_passed",
    }:
        errors.append("operational_diagnostics keys do not match the v1 schema")
    if value.get("digest_exclusion_reason") != (
        "host environment and wall-clock timing are non-deterministic"
    ):
        errors.append("operational_diagnostics digest exclusion reason is invalid")
    _validate_environment(value.get("environment"), errors)
    timings = value.get("condition_timings")
    if not isinstance(timings, list):
        errors.append("operational_diagnostics.condition_timings must be an array")
    else:
        observed_pairs: list[tuple[int, str]] = []
        for index, timing in enumerate(timings):
            if not isinstance(timing, Mapping):
                errors.append(f"condition_timings[{index}] must be an object")
                continue
            if set(timing) != {
                "seed",
                "condition",
                "wall_seconds",
                "mean_step_latency_ms",
            }:
                errors.append(f"condition_timings[{index}] keys do not match the v1 schema")
            seed = timing.get("seed")
            condition = timing.get("condition")
            wall = _number(timing.get("wall_seconds"))
            latency = _number(timing.get("mean_step_latency_ms"))
            if not isinstance(seed, int) or not isinstance(condition, str):
                errors.append(f"condition_timings[{index}] has invalid identity")
            else:
                observed_pairs.append((seed, condition))
            if wall is None or latency is None or wall < 0.0 or latency < 0.0:
                errors.append(f"condition_timings[{index}] has invalid timing")
        expected_pairs = (
            [(result.seed, result.condition) for result in expected_results]
            if expected_results is not None
            else None
        )
        if expected_pairs is not None and observed_pairs != expected_pairs:
            errors.append("operational timing identities do not match scientific results")
    if value.get("overall_acceptance_passed") is not expected_acceptance:
        errors.append("operational overall acceptance is inconsistent")


def validate_ia_evidence_artifact(
    artifact: Mapping[str, object],
) -> IAArtifactValidation:
    """Validate digest plus primitive-to-acceptance internal consistency."""

    errors: list[str] = []
    if set(artifact) != {
        "schema_version",
        "content",
        "operational_diagnostics",
        "content_digest",
    }:
        errors.append("artifact top-level keys do not match the v1 schema")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    content = artifact.get("content")
    digest = artifact.get("content_digest")
    operational = artifact.get("operational_diagnostics")
    if not isinstance(content, Mapping):
        errors.append("content must be an object")
        content = None
    if not isinstance(digest, Mapping):
        errors.append("content_digest must be an object")
        digest = None

    expected_results: tuple[IAConditionResult, ...] | None = None
    expected_acceptance = False
    if content is not None:
        if set(content) != {
            "protocol",
            "configuration",
            "thresholds",
            "seed_summaries",
            "aggregate",
            "acceptance",
            "source_provenance",
        }:
            errors.append("content keys do not match the v1 schema")
        if content.get("protocol") != _protocol_payload():
            errors.append("content.protocol is not the canonical frozen v1 protocol")
        config = _parse_config(content.get("configuration"), errors)
        thresholds = _parse_thresholds(content.get("thresholds"), errors)
        _validate_provenance(content.get("source_provenance"), errors)
        if config is not None:
            expected_results = _parse_results(
                content.get("seed_summaries"),
                config=config,
                errors=errors,
            )
        if config is not None and thresholds is not None and expected_results is not None:
            try:
                recomputed_aggregate = aggregate_ia_evidence(
                    expected_results,
                    config=config,
                )
            except ValueError as error:
                errors.append(f"cannot aggregate primitive seed evidence: {error}")
            else:
                recomputed_acceptance = evaluate_ia_acceptance(
                    recomputed_aggregate,
                    thresholds,
                )
                expected_acceptance = recomputed_acceptance.passed
                if content.get("aggregate") != _aggregate_payload(recomputed_aggregate):
                    errors.append("content.aggregate is inconsistent with primitive seed evidence")
                if content.get("acceptance") != _acceptance_payload(recomputed_acceptance):
                    errors.append("content.acceptance is inconsistent with recomputed evidence")

    if digest is not None:
        if set(digest) != {
            "algorithm",
            "scope",
            "canonicalization",
            "sha256",
        }:
            errors.append("content_digest keys do not match the v1 schema")
        if digest.get("algorithm") != "sha256":
            errors.append("content_digest.algorithm must be 'sha256'")
        if digest.get("scope") != DIGEST_SCOPE:
            errors.append(f"content_digest.scope must be {DIGEST_SCOPE!r}")
        if digest.get("canonicalization") != CANONICALIZATION:
            errors.append("content_digest canonicalization is unsupported")
        recorded = digest.get("sha256")
        if (
            not isinstance(recorded, str)
            or len(recorded) != 64
            or any(character not in "0123456789abcdef" for character in recorded)
        ):
            errors.append("content_digest.sha256 must be lowercase hexadecimal")
        elif content is not None and recorded != scientific_content_sha256(content):
            errors.append("content_digest.sha256 does not match content")

    _validate_operational(
        operational,
        expected_results=expected_results,
        expected_acceptance=expected_acceptance,
        errors=errors,
    )
    valid = not errors
    return IAArtifactValidation(
        valid=valid,
        accepted=valid and expected_acceptance,
        errors=tuple(errors),
    )
