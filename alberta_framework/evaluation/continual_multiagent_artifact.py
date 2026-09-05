"""Versioned evidence artifacts for the recurring multi-agent benchmark.

The artifact deliberately separates deterministic scientific content from
operational diagnostics.  Controller-update and wall-clock timing depend on
the host and are retained for inspection, but they are outside the SHA-256
scope.  The digest covers the exact protocol description, configuration,
thresholds, paired seed summaries, aggregate statistics, scientific acceptance
checks, and relevant source-file hashes.

This is evidence for a narrow, visibly cued A-B-A coadaptation sanity check.  It
is not evidence of general feature discovery, intelligence amplification, or
completion of the Alberta Plan.
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
from typing import NoReturn

import jax
import numpy as np

from alberta_framework._bounded_containers import require_json_text_nesting
from alberta_framework.evaluation.continual_multiagent import (
    AcceptanceEvidence,
    AcceptanceThresholds,
    BootstrapInterval,
    ConditionResult,
    ContinualMultiAgentConfig,
    ContinualMultiAgentReport,
    paired_bootstrap_mean_interval,
)

SCHEMA_VERSION = "alberta.continual_multiagent_evidence.v1"
_MAX_JSON_NESTING_DEPTH = 64
PROTOCOL_VERSION = "recurring-two-agent-contextual-bandit-aba.v1"
DIGEST_ALGORITHM = "sha256"
DIGEST_SCOPE = "$.content"
CANONICALIZATION = "utf8-json-sort-keys-compact-no-nan"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATHS = (
    Path("alberta_framework/evaluation/continual_multiagent.py"),
    Path("alberta_framework/evaluation/continual_multiagent_artifact.py"),
    Path("alberta_framework/evaluation/continual_multiagent_cli.py"),
    Path("alberta_framework/streams/recurring_multiagent.py"),
    Path("alberta_framework/utils/metrics.py"),
)
_SCIENTIFIC_CHECK_RENAMES = {
    "partner_uplift": "coadaptation_uplift_over_learner_only",
}
_OPERATIONAL_CHECKS = frozenset({"update_latency_ms"})
_REQUIRED_SCIENTIFIC_CHECKS = frozenset(
    {
        "seed_count",
        "evidence_seed_schedule",
        "all_values_finite",
        "budgets_identical",
        "reward_uplift_over_frozen",
        "coadaptation_uplift_over_learner_only",
        "recurrent_a_probe_reward",
        "mean_forgetting",
        "interference_forgetting",
        "recurrence_recovery_fraction",
        "mean_recurrence_recovery_steps",
        "mean_stability_gap",
    }
)
_EXPECTED_EVIDENCE_SEEDS = tuple(range(30, 60))
_EXPECTED_CONDITIONS = ("frozen", "learner_only", "joint_adaptive")
_EXPECTED_LEARNING_MASKS = {
    "frozen": [False, False],
    "learner_only": [True, False],
    "joint_adaptive": [True, True],
}
_REQUIRED_CONDITION_FIELDS = frozenset(
    {
        "learning_mask",
        "prequential_reward",
        "final_probe_reward",
        "phase_mean_rewards",
        "read_only_probe_matrix",
        "mean_forgetting",
        "maximum_forgetting",
        "backward_transfer",
        "mean_stability_gap",
        "maximum_stability_gap",
        "per_task_final_performance",
        "per_task_forgetting",
        "per_task_backward_transfer",
        "recovery_lengths",
        "recurrence_recovery_steps",
        "interference_forgetting",
        "controller_budget",
    }
)


@dataclass(frozen=True)
class ArtifactValidation:
    """Fail-closed integrity and acceptance result for one parsed artifact."""

    valid: bool
    accepted: bool
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.valid) is not bool:
            raise ValueError("valid must be a boolean")
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be a boolean")
        if type(self.errors) is not tuple or not all(isinstance(e, str) for e in self.errors):
            raise ValueError("errors must be a tuple of strings")


def _finite_float(value: float) -> float | None:
    """Encode non-finite evidence as JSON ``null`` rather than non-standard NaN."""

    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _vector(values: np.ndarray) -> list[float | None]:
    return [_finite_float(float(value)) for value in np.asarray(values).reshape(-1)]


def _matrix(values: np.ndarray) -> list[list[float | None]]:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("matrix evidence must be two-dimensional")
    return [[_finite_float(float(value)) for value in row] for row in array]


def _config_payload(config: ContinualMultiAgentConfig) -> dict[str, object]:
    return {
        "phase_steps": config.phase_steps,
        "nuisance_dim": config.nuisance_dim,
        "learning_rate": config.learning_rate,
        "exploration_rate": config.exploration_rate,
        "probe_horizon": config.probe_horizon,
        "probe_tail_steps": config.probe_tail_steps,
        "recovery_reward_threshold": config.recovery_reward_threshold,
        "recovery_window": config.recovery_window,
        "stability_reference_reward": config.stability_reference_reward,
        "bootstrap_resamples": config.bootstrap_resamples,
        "confidence_level": config.confidence_level,
        "bootstrap_seed": config.bootstrap_seed,
    }


def _threshold_payload(thresholds: AcceptanceThresholds) -> dict[str, object]:
    return {
        "minimum_seed_count": thresholds.minimum_seed_count,
        "evidence_seed_start": thresholds.evidence_seed_start,
        "minimum_reward_uplift_over_frozen": (
            thresholds.minimum_reward_uplift_over_frozen
        ),
        "minimum_coadaptation_uplift_over_learner_only": (
            thresholds.minimum_partner_uplift
        ),
        "minimum_recurrent_a_probe_reward": (
            thresholds.minimum_recurrent_a_probe_reward
        ),
        "maximum_mean_forgetting": thresholds.maximum_mean_forgetting,
        "maximum_interference_forgetting": (
            thresholds.maximum_interference_forgetting
        ),
        "minimum_recurrence_recovery_fraction": (
            thresholds.minimum_recurrence_recovery_fraction
        ),
        "maximum_mean_recurrence_recovery_steps": (
            thresholds.maximum_mean_recurrence_recovery_steps
        ),
        "maximum_mean_stability_gap": thresholds.maximum_mean_stability_gap,
        "maximum_update_latency_ms": thresholds.maximum_update_latency_ms,
    }


def _interval_payload(interval: BootstrapInterval) -> dict[str, object]:
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


def wilson_score_interval(
    successes: int,
    sample_size: int,
    *,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Return a Wilson score interval for one observed Bernoulli proportion."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if not 0 <= successes <= sample_size:
        raise ValueError("successes must lie in [0, sample_size]")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")

    from statistics import NormalDist

    proportion = successes / sample_size
    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    z_squared = z_score * z_score
    denominator = 1.0 + z_squared / sample_size
    center = (proportion + z_squared / (2.0 * sample_size)) / denominator
    half_width = (
        z_score
        * (
            proportion * (1.0 - proportion) / sample_size
            + z_squared / (4.0 * sample_size * sample_size)
        )
        ** 0.5
        / denominator
    )
    return {
        "estimate": proportion,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
        "confidence_level": confidence_level,
        "successes": successes,
        "sample_size": sample_size,
        "method": "wilson-score",
        "interpretation": (
            "uncertainty interval for population recovery probability; the "
            "observed-fraction acceptance threshold does not prove its lower "
            "bound exceeds that threshold"
        ),
    }


def _condition_payload(result: ConditionResult) -> dict[str, object]:
    summary = result.summary
    return {
        "learning_mask": list(result.learning_mask),
        "prequential_reward": _finite_float(summary.prequential_performance),
        "final_probe_reward": _finite_float(summary.final_performance),
        "phase_mean_rewards": _vector(result.phase_mean_rewards),
        "read_only_probe_matrix": _matrix(result.performance_matrix),
        "mean_forgetting": _finite_float(summary.mean_forgetting),
        "maximum_forgetting": _finite_float(summary.max_forgetting),
        "backward_transfer": _finite_float(summary.backward_transfer),
        "mean_stability_gap": _finite_float(summary.stability_gap_mean),
        "maximum_stability_gap": _finite_float(summary.stability_gap_max),
        "per_task_final_performance": _vector(
            summary.per_task_final_performance,
        ),
        "per_task_forgetting": _vector(summary.per_task_forgetting),
        "per_task_backward_transfer": _vector(
            summary.per_task_backward_transfer,
        ),
        "recovery_lengths": [
            int(value) for value in np.asarray(result.recovery_lengths).reshape(-1)
        ],
        "recurrence_recovery_steps": result.recurrence_recovery_steps,
        "interference_forgetting": _finite_float(
            result.interference_forgetting,
        ),
        "controller_budget": {
            "state_scalars": result.controller_budget.state_scalars,
            "state_bytes": result.controller_budget.state_bytes,
            "action_scalars_per_step": (
                result.controller_budget.action_scalars_per_step
            ),
        },
    }


def _seed_payloads(results: tuple[ConditionResult, ...]) -> list[dict[str, object]]:
    grouped: dict[int, dict[str, dict[str, object]]] = {}
    for result in results:
        conditions = grouped.setdefault(result.seed, {})
        if result.condition in conditions:
            raise ValueError(
                f"duplicate condition {result.condition!r} for seed {result.seed}"
            )
        conditions[result.condition] = _condition_payload(result)

    required = {"frozen", "learner_only", "joint_adaptive"}
    payloads: list[dict[str, object]] = []
    for seed in sorted(grouped):
        conditions = grouped[seed]
        if set(conditions) != required:
            missing = sorted(required - set(conditions))
            extra = sorted(set(conditions) - required)
            raise ValueError(
                f"seed {seed} condition mismatch: missing={missing}, extra={extra}"
            )
        payloads.append(
            {
                "seed": seed,
                "conditions": {
                    name: conditions[name]
                    for name in ("frozen", "learner_only", "joint_adaptive")
                },
            }
        )
    return payloads


def _aggregate_payload(report: ContinualMultiAgentReport) -> dict[str, object]:
    aggregate = report.aggregate
    seed_count = len(aggregate.seeds)
    recovery_successes = sum(
        result.recurrence_recovery_steps >= 0
        for result in report.condition_results
        if result.condition == "joint_adaptive"
    )
    return {
        "seed_count": seed_count,
        "seeds": list(aggregate.seeds),
        "mean_prequential_reward": {
            "frozen": _finite_float(aggregate.frozen_prequential_reward),
            "learner_only": _finite_float(
                aggregate.learner_only_prequential_reward
            ),
            "joint_adaptive": _finite_float(
                aggregate.joint_adaptive_prequential_reward
            ),
        },
        "reward_uplift_over_frozen": _finite_float(
            aggregate.reward_uplift_over_frozen
        ),
        "reward_uplift_over_frozen_paired_interval": _interval_payload(
            aggregate.reward_uplift_interval
        ),
        "coadaptation_uplift_over_learner_only": _finite_float(
            aggregate.partner_uplift
        ),
        "coadaptation_uplift_over_learner_only_paired_interval": (
            _interval_payload(aggregate.partner_uplift_interval)
        ),
        "joint_adaptive_phase_rewards": _vector(
            aggregate.joint_adaptive_phase_rewards
        ),
        "joint_adaptive_read_only_probe_matrix": _matrix(
            aggregate.joint_adaptive_performance_matrix
        ),
        "recurrent_a_probe_reward": _finite_float(
            aggregate.recurrent_a_probe_reward
        ),
        "mean_forgetting": _finite_float(aggregate.mean_forgetting),
        "maximum_forgetting": _finite_float(aggregate.max_forgetting),
        "mean_interference_forgetting": _finite_float(
            aggregate.mean_interference_forgetting
        ),
        "recurrence_recovery_fraction": _finite_float(
            aggregate.recurrence_recovery_fraction
        ),
        "recurrence_recovery_fraction_interval": wilson_score_interval(
            recovery_successes,
            seed_count,
            confidence_level=report.config.confidence_level,
        ),
        "mean_recurrence_recovery_steps_among_recovered_seeds": _finite_float(
            aggregate.mean_recurrence_recovery_steps
        ),
        "mean_stability_gap": _finite_float(aggregate.mean_stability_gap),
        "resource_budget": {
            "state_scalars": aggregate.state_scalars,
            "state_bytes": aggregate.state_bytes,
            "action_scalars_per_step": aggregate.action_scalars_per_step,
            "identical_across_conditions": aggregate.budgets_identical,
        },
        "all_scientific_and_timing_values_finite": aggregate.all_values_finite,
    }


def _check_payload(check: AcceptanceEvidence) -> dict[str, object]:
    name = _SCIENTIFIC_CHECK_RENAMES.get(check.name, check.name)
    detail = check.detail
    if check.name == "partner_uplift":
        detail = (
            "Lower confidence bound for paired joint-adaptive minus "
            "learner-only reward; this measures coadaptation from partner "
            "learning, not an IA intervention."
        )
    return {
        "name": name,
        "passed": check.passed,
        "actual": _finite_float(check.actual),
        "comparator": check.comparator,
        "threshold": _finite_float(check.threshold),
        "detail": detail,
    }


def _scientific_acceptance_payload(
    report: ContinualMultiAgentReport,
) -> dict[str, object]:
    checks = [
        _check_payload(check)
        for check in report.acceptance.checks
        if check.name not in _OPERATIONAL_CHECKS
    ]
    return {
        "passed": all(check["passed"] is True for check in checks),
        "checks": checks,
        "excluded_operational_checks": sorted(_OPERATIONAL_CHECKS),
    }


def _source_provenance() -> dict[str, object]:
    source_hashes: dict[str, str] = {}
    for relative_path in _SOURCE_PATHS:
        path = _REPO_ROOT / relative_path
        source_hashes[relative_path.as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return {
        "repository_subtree": "research/alberta",
        "source_sha256": source_hashes,
    }


def _validate_source_provenance(value: object, errors: list[str]) -> None:
    """Require exact hashes for every source that defines this evidence."""

    location = "content.source_provenance"
    if not isinstance(value, Mapping):
        errors.append(f"{location} must be an object")
        return
    try:
        expected = _source_provenance()
    except OSError as error:
        errors.append(f"cannot bind current source provenance: {error}")
        return
    if set(value) != set(expected):
        errors.append(f"{location} keys do not match the v1 schema")
    if value.get("repository_subtree") != expected["repository_subtree"]:
        errors.append(f"{location}.repository_subtree is invalid")
    if value.get("source_sha256") != expected["source_sha256"]:
        errors.append(f"{location} does not match the current pinned source hashes")


def _protocol_payload() -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "schedule": ["A1-meet", "B-avoid", "A2-meet"],
        "conditions": {
            "frozen": "neither controller applies learning writes",
            "learner_only": "agent 0 learns and agent 1 is frozen",
            "joint_adaptive": "both agents learn online",
        },
        "pairing_unit": "seed",
        "interaction": (
            "one uninterrupted predict-act-observe-update life with no replay "
            "or task-boundary callback"
        ),
        "supported_claim": (
            "fixed-memory online coadaptation and retention in a visibly cued "
            "two-agent A-B-A sanity benchmark"
        ),
        "excluded_claims": [
            "general feature discovery",
            "uncued task inference",
            "intelligence amplification or recommendation intervention",
            "general catastrophic-forgetting resistance",
            "completion of the Alberta Plan",
        ],
        "coadaptation_metric": (
            "paired mean prequential reward of joint_adaptive minus "
            "learner_only; this is partner-learning uplift, not causal IA"
        ),
        "seed_roles": {
            "development_and_calibration": list(range(30)),
            "promoted_held_out_evidence": list(range(30, 60)),
            "smoke_subset": [0, 1, 2],
            "rule": (
                "thresholds were frozen after seeds 0-29; promoted evidence "
                "uses disjoint seeds 30-59 without retuning"
            ),
        },
        "open_ablation": (
            "A same-state-budget joint learner with aliased context values is "
            "not in this frozen protocol. It requires a newly preregistered, "
            "disjoint evidence set before supporting a context-memory uplift "
            "claim."
        ),
    }


def _environment_payload() -> dict[str, object]:
    devices = [
        {
            "platform": device.platform,
            "device_kind": device.device_kind,
        }
        for device in jax.devices()
    ]
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
        "jax_devices": devices,
    }


def _operational_payload(report: ContinualMultiAgentReport) -> dict[str, object]:
    condition_timings = [
        {
            "seed": result.seed,
            "condition": result.condition,
            "wall_seconds": _finite_float(result.timing.wall_seconds),
            "mean_step_latency_ms": _finite_float(
                result.timing.mean_step_latency_ms
            ),
            "mean_update_latency_ms": _finite_float(
                result.timing.mean_update_latency_ms
            ),
            "p95_update_latency_ms": _finite_float(
                result.timing.p95_update_latency_ms
            ),
        }
        for result in report.condition_results
    ]
    operational_checks = [
        _check_payload(check)
        for check in report.acceptance.checks
        if check.name in _OPERATIONAL_CHECKS
    ]
    return {
        "digest_exclusion_reason": (
            "host environment and measured wall-clock timing are retained for "
            "inspection but are not deterministic scientific evidence"
        ),
        "environment": _environment_payload(),
        "condition_timings": condition_timings,
        "maximum_update_latency_ms": _finite_float(
            report.aggregate.maximum_update_latency_ms
        ),
        "checks": operational_checks,
        "passed": all(check["passed"] is True for check in operational_checks),
        "overall_acceptance_passed": report.acceptance.passed,
    }


def canonical_content_bytes(content: Mapping[str, object]) -> bytes:
    """Return the documented canonical byte representation for digesting."""

    return json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def scientific_content_sha256(content: Mapping[str, object]) -> str:
    """Digest deterministic scientific content using SHA-256."""

    return hashlib.sha256(canonical_content_bytes(content)).hexdigest()


def build_evidence_artifact(
    report: ContinualMultiAgentReport,
) -> dict[str, object]:
    """Build a deterministic-content, fail-closed evidence artifact."""

    if tuple(sorted(report.aggregate.seeds)) != report.aggregate.seeds:
        raise ValueError("artifact seeds must be in strictly increasing order")
    if len(set(report.aggregate.seeds)) != len(report.aggregate.seeds):
        raise ValueError("artifact seeds must be unique")

    content: dict[str, object] = {
        "protocol": _protocol_payload(),
        "configuration": _config_payload(report.config),
        "thresholds": _threshold_payload(report.thresholds),
        "seed_summaries": _seed_payloads(report.condition_results),
        "aggregate": _aggregate_payload(report),
        "acceptance": _scientific_acceptance_payload(report),
        "source_provenance": _source_provenance(),
    }
    digest = scientific_content_sha256(content)
    return {
        "schema_version": SCHEMA_VERSION,
        "content": content,
        "operational_diagnostics": _operational_payload(report),
        "content_digest": {
            "algorithm": DIGEST_ALGORITHM,
            "scope": DIGEST_SCOPE,
            "canonicalization": CANONICALIZATION,
            "sha256": digest,
        },
    }


def artifact_json(artifact: Mapping[str, object]) -> str:
    """Serialize a human-readable artifact using strict standard JSON."""

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


def write_evidence_artifact(
    path: Path,
    report: ContinualMultiAgentReport,
) -> dict[str, object]:
    """Build, validate, and exclusively create an evidence artifact."""

    artifact = build_evidence_artifact(report)
    validation = validate_evidence_artifact(artifact)
    if not validation.valid:
        raise ValueError(
            "refusing to write invalid evidence artifact: "
            + "; ".join(validation.errors)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(artifact_json(artifact))
    return artifact


def _reject_nonstandard_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_evidence_artifact(path: Path) -> dict[str, object]:
    """Load strict JSON and require a top-level object."""

    text = path.read_text(encoding="utf-8")
    require_json_text_nesting(
        text,
        max_depth=_MAX_JSON_NESTING_DEPTH,
        name="evidence artifact",
    )
    try:
        parsed = json.loads(
            text,
            parse_constant=_reject_nonstandard_json_constant,
            parse_float=_parse_finite_json_float,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except RecursionError as exc:
        raise ValueError("evidence artifact exceeds the JSON nesting limit") from exc
    if not isinstance(parsed, dict):
        raise ValueError("evidence artifact must be a JSON object")
    return parsed


def _required_mapping(
    parent: Mapping[str, object],
    key: str,
    errors: list[str],
) -> Mapping[str, object] | None:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be an object")
        return None
    return value


def _finite_number(value: object) -> float | None:
    if type(value) is bool or (type(value) is not int and type(value) is not float):
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _validate_threshold_policy(
    thresholds: Mapping[str, object],
    errors: list[str],
) -> None:
    """Require canonical seed roles and forbid weaker v1 acceptance limits."""

    canonical = _threshold_payload(AcceptanceThresholds())
    for key in ("minimum_seed_count", "evidence_seed_start"):
        if thresholds.get(key) != canonical[key]:
            errors.append(
                f"content.thresholds.{key} must equal the v1 canonical value"
            )

    minimum_keys = (
        "minimum_reward_uplift_over_frozen",
        "minimum_coadaptation_uplift_over_learner_only",
        "minimum_recurrent_a_probe_reward",
        "minimum_recurrence_recovery_fraction",
    )
    maximum_keys = (
        "maximum_mean_forgetting",
        "maximum_interference_forgetting",
        "maximum_mean_recurrence_recovery_steps",
        "maximum_mean_stability_gap",
        "maximum_update_latency_ms",
    )
    for key in minimum_keys:
        actual = _finite_number(thresholds.get(key))
        floor = _finite_number(canonical[key])
        if actual is None or floor is None or actual < floor:
            errors.append(
                f"content.thresholds.{key} is weaker than the v1 canonical gate"
            )
    for key in maximum_keys:
        actual = _finite_number(thresholds.get(key))
        ceiling = _finite_number(canonical[key])
        if actual is None or ceiling is None or actual > ceiling:
            errors.append(
                f"content.thresholds.{key} is weaker than the v1 canonical gate"
            )


def _numbers_match(first: object, second: object) -> bool:
    left = _finite_number(first)
    right = _finite_number(second)
    return (
        left is not None
        and right is not None
        and bool(np.isclose(left, right, rtol=0.0, atol=1e-12))
    )


def _validate_interval_record(
    value: object,
    *,
    location: str,
    config: Mapping[str, object],
    expected: BootstrapInterval | None,
    errors: list[str],
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{location} must be an object")
        return
    if value.get("sample_size") != len(_EXPECTED_EVIDENCE_SEEDS):
        errors.append(f"{location}.sample_size must be 30")
    if value.get("method") != "paired-percentile-bootstrap":
        errors.append(f"{location}.method must be paired-percentile-bootstrap")
    if value.get("pairing_unit") != "seed":
        errors.append(f"{location}.pairing_unit must be seed")
    if value.get("resamples") != config.get("bootstrap_resamples"):
        errors.append(
            f"{location}.resamples must match content.configuration"
        )
    if not _numbers_match(
        value.get("confidence_level"),
        config.get("confidence_level"),
    ):
        errors.append(
            f"{location}.confidence_level must match content.configuration"
        )
    if expected is not None:
        for field_name in ("estimate", "lower", "upper"):
            if not _numbers_match(
                value.get(field_name),
                getattr(expected, field_name),
            ):
                errors.append(
                    f"{location}.{field_name} is inconsistent with paired "
                    "seed summaries"
                )


def _validate_seed_and_aggregate_consistency(
    content: Mapping[str, object],
    errors: list[str],
) -> None:
    """Cross-check exact seed roles, conditions, aggregate means, and intervals."""

    seed_summaries = content.get("seed_summaries")
    aggregate = content.get("aggregate")
    config = content.get("configuration")
    if not isinstance(seed_summaries, list):
        errors.append("content.seed_summaries must be an array")
        return
    if not isinstance(aggregate, Mapping) or not isinstance(config, Mapping):
        return

    observed_seeds: list[int] = []
    prequential: dict[str, list[float]] = {
        condition: [] for condition in _EXPECTED_CONDITIONS
    }
    recurrence_steps: list[int] = []
    for index, raw_seed in enumerate(seed_summaries):
        location = f"content.seed_summaries[{index}]"
        if not isinstance(raw_seed, Mapping):
            errors.append(f"{location} must be an object")
            continue
        seed = raw_seed.get("seed")
        if type(seed) is not int:
            errors.append(f"{location}.seed must be an integer")
        else:
            observed_seeds.append(seed)

        conditions = raw_seed.get("conditions")
        if not isinstance(conditions, Mapping):
            errors.append(f"{location}.conditions must be an object")
            continue
        if set(conditions) != set(_EXPECTED_CONDITIONS):
            errors.append(
                f"{location}.conditions must contain exactly "
                + ", ".join(_EXPECTED_CONDITIONS)
            )
        for condition in _EXPECTED_CONDITIONS:
            result = conditions.get(condition)
            result_location = f"{location}.conditions.{condition}"
            if not isinstance(result, Mapping):
                errors.append(f"{result_location} must be an object")
                continue
            missing_fields = sorted(
                _REQUIRED_CONDITION_FIELDS - set(result)
            )
            if missing_fields:
                errors.append(
                    f"{result_location} is missing fields: "
                    + ", ".join(missing_fields)
                )
            if result.get("learning_mask") != _EXPECTED_LEARNING_MASKS[condition]:
                errors.append(
                    f"{result_location}.learning_mask is inconsistent "
                    "with the protocol"
                )
            reward = _finite_number(result.get("prequential_reward"))
            if reward is None:
                errors.append(
                    f"{result_location}.prequential_reward must be finite"
                )
            else:
                prequential[condition].append(reward)
            if condition == "joint_adaptive":
                recovery = result.get("recurrence_recovery_steps")
                if type(recovery) is not int:
                    errors.append(
                        f"{result_location}.recurrence_recovery_steps "
                        "must be an integer"
                    )
                else:
                    recurrence_steps.append(recovery)

    if tuple(observed_seeds) != _EXPECTED_EVIDENCE_SEEDS:
        errors.append(
            "content.seed_summaries must contain exactly unique held-out "
            "seeds 30-59 in order"
        )
    aggregate_seeds = aggregate.get("seeds")
    if aggregate_seeds != list(_EXPECTED_EVIDENCE_SEEDS):
        errors.append("content.aggregate.seeds must equal held-out seeds 30-59")
    if aggregate.get("seed_count") != len(_EXPECTED_EVIDENCE_SEEDS):
        errors.append("content.aggregate.seed_count must equal 30")
    if aggregate_seeds != observed_seeds:
        errors.append(
            "content.aggregate.seeds must match content.seed_summaries"
        )

    mean_rewards = aggregate.get("mean_prequential_reward")
    if isinstance(mean_rewards, Mapping):
        for condition in _EXPECTED_CONDITIONS:
            values = prequential[condition]
            if len(values) == len(_EXPECTED_EVIDENCE_SEEDS) and not _numbers_match(
                mean_rewards.get(condition),
                float(np.mean(values)),
            ):
                errors.append(
                    "content.aggregate.mean_prequential_reward."
                    f"{condition} is inconsistent with seed summaries"
                )
    else:
        errors.append("content.aggregate.mean_prequential_reward must be an object")

    confidence = _finite_number(config.get("confidence_level"))
    resamples = config.get("bootstrap_resamples")
    bootstrap_seed = config.get("bootstrap_seed")
    can_bootstrap = (
        confidence is not None
        and isinstance(resamples, int)
        and not isinstance(resamples, bool)
        and resamples > 0
        and isinstance(bootstrap_seed, int)
        and not isinstance(bootstrap_seed, bool)
        and all(
            len(prequential[condition]) == len(_EXPECTED_EVIDENCE_SEEDS)
            for condition in _EXPECTED_CONDITIONS
        )
    )
    reward_interval: BootstrapInterval | None = None
    coadaptation_interval: BootstrapInterval | None = None
    if can_bootstrap:
        assert confidence is not None
        assert isinstance(resamples, int)
        assert isinstance(bootstrap_seed, int)
        frozen = np.asarray(prequential["frozen"], dtype=np.float64)
        learner_only = np.asarray(
            prequential["learner_only"],
            dtype=np.float64,
        )
        joint = np.asarray(prequential["joint_adaptive"], dtype=np.float64)
        reward_interval = paired_bootstrap_mean_interval(
            joint - frozen,
            confidence_level=confidence,
            resamples=resamples,
            seed=bootstrap_seed,
        )
        coadaptation_interval = paired_bootstrap_mean_interval(
            joint - learner_only,
            confidence_level=confidence,
            resamples=resamples,
            seed=(bootstrap_seed + 1) % 2**32,
        )
    _validate_interval_record(
        aggregate.get("reward_uplift_over_frozen_paired_interval"),
        location=(
            "content.aggregate.reward_uplift_over_frozen_paired_interval"
        ),
        config=config,
        expected=reward_interval,
        errors=errors,
    )
    _validate_interval_record(
        aggregate.get(
            "coadaptation_uplift_over_learner_only_paired_interval"
        ),
        location=(
            "content.aggregate."
            "coadaptation_uplift_over_learner_only_paired_interval"
        ),
        config=config,
        expected=coadaptation_interval,
        errors=errors,
    )
    if reward_interval is not None and not _numbers_match(
        aggregate.get("reward_uplift_over_frozen"),
        reward_interval.estimate,
    ):
        errors.append(
            "content.aggregate.reward_uplift_over_frozen is inconsistent "
            "with paired seed summaries"
        )
    if coadaptation_interval is not None and not _numbers_match(
        aggregate.get("coadaptation_uplift_over_learner_only"),
        coadaptation_interval.estimate,
    ):
        errors.append(
            "content.aggregate.coadaptation_uplift_over_learner_only is "
            "inconsistent with paired seed summaries"
        )

    if len(recurrence_steps) == len(_EXPECTED_EVIDENCE_SEEDS):
        successes = sum(value >= 0 for value in recurrence_steps)
        fraction = successes / len(recurrence_steps)
        if not _numbers_match(
            aggregate.get("recurrence_recovery_fraction"),
            fraction,
        ):
            errors.append(
                "content.aggregate.recurrence_recovery_fraction is "
                "inconsistent with seed summaries"
            )
        recovered = [value for value in recurrence_steps if value >= 0]
        expected_mean = float(np.mean(recovered)) if recovered else None
        if expected_mean is None or not _numbers_match(
            aggregate.get(
                "mean_recurrence_recovery_steps_among_recovered_seeds"
            ),
            expected_mean,
        ):
            errors.append(
                "content.aggregate mean recurrence recovery is inconsistent "
                "with seed summaries"
            )
        recovery_interval = aggregate.get(
            "recurrence_recovery_fraction_interval"
        )
        if not isinstance(recovery_interval, Mapping):
            errors.append(
                "content.aggregate.recurrence_recovery_fraction_interval "
                "must be an object"
            )
        else:
            expected_wilson = wilson_score_interval(
                successes,
                len(recurrence_steps),
                confidence_level=confidence if confidence is not None else 0.95,
            )
            for field_name in (
                "estimate",
                "lower",
                "upper",
                "confidence_level",
            ):
                if not _numbers_match(
                    recovery_interval.get(field_name),
                    expected_wilson[field_name],
                ):
                    errors.append(
                        "content.aggregate.recurrence_recovery_fraction_interval."
                        f"{field_name} is inconsistent with seed summaries"
                    )
            if recovery_interval.get("successes") != successes:
                errors.append(
                    "content.aggregate.recurrence_recovery_fraction_interval."
                    "successes is inconsistent with seed summaries"
                )
            if recovery_interval.get("sample_size") != len(
                _EXPECTED_EVIDENCE_SEEDS
            ):
                errors.append(
                    "content.aggregate.recurrence_recovery_fraction_interval."
                    "sample_size must equal 30"
                )


def _validate_scientific_check_bindings(
    checks: Mapping[str, Mapping[str, object]],
    *,
    thresholds: Mapping[str, object],
    aggregate: Mapping[str, object],
    errors: list[str],
) -> None:
    reward_interval = aggregate.get(
        "reward_uplift_over_frozen_paired_interval"
    )
    coadaptation_interval = aggregate.get(
        "coadaptation_uplift_over_learner_only_paired_interval"
    )
    resources = aggregate.get("resource_budget")
    aggregate_seeds = aggregate.get("seeds")
    expected_schedule = float(
        aggregate_seeds == list(_EXPECTED_EVIDENCE_SEEDS)
    )
    expected: dict[str, tuple[object, str, object]] = {
        "seed_count": (
            aggregate.get("seed_count"),
            ">=",
            thresholds.get("minimum_seed_count"),
        ),
        "evidence_seed_schedule": (expected_schedule, ">=", 1.0),
        "all_values_finite": (
            float(
                aggregate.get("all_scientific_and_timing_values_finite")
                is True
            ),
            ">=",
            1.0,
        ),
        "budgets_identical": (
            float(
                isinstance(resources, Mapping)
                and resources.get("identical_across_conditions") is True
            ),
            ">=",
            1.0,
        ),
        "reward_uplift_over_frozen": (
            reward_interval.get("lower")
            if isinstance(reward_interval, Mapping)
            else None,
            ">=",
            thresholds.get("minimum_reward_uplift_over_frozen"),
        ),
        "coadaptation_uplift_over_learner_only": (
            coadaptation_interval.get("lower")
            if isinstance(coadaptation_interval, Mapping)
            else None,
            ">=",
            thresholds.get(
                "minimum_coadaptation_uplift_over_learner_only"
            ),
        ),
        "recurrent_a_probe_reward": (
            aggregate.get("recurrent_a_probe_reward"),
            ">=",
            thresholds.get("minimum_recurrent_a_probe_reward"),
        ),
        "mean_forgetting": (
            aggregate.get("mean_forgetting"),
            "<=",
            thresholds.get("maximum_mean_forgetting"),
        ),
        "interference_forgetting": (
            aggregate.get("mean_interference_forgetting"),
            "<=",
            thresholds.get("maximum_interference_forgetting"),
        ),
        "recurrence_recovery_fraction": (
            aggregate.get("recurrence_recovery_fraction"),
            ">=",
            thresholds.get("minimum_recurrence_recovery_fraction"),
        ),
        "mean_recurrence_recovery_steps": (
            aggregate.get(
                "mean_recurrence_recovery_steps_among_recovered_seeds"
            ),
            "<=",
            thresholds.get("maximum_mean_recurrence_recovery_steps"),
        ),
        "mean_stability_gap": (
            aggregate.get("mean_stability_gap"),
            "<=",
            thresholds.get("maximum_mean_stability_gap"),
        ),
    }
    for name, (actual, comparator, threshold) in expected.items():
        check = checks.get(name)
        if check is None:
            continue
        if not _numbers_match(check.get("actual"), actual):
            errors.append(
                f"content.acceptance check {name!r} actual is inconsistent "
                "with aggregate evidence"
            )
        if check.get("comparator") != comparator:
            errors.append(
                f"content.acceptance check {name!r} comparator is inconsistent"
            )
        if not _numbers_match(check.get("threshold"), threshold):
            errors.append(
                f"content.acceptance check {name!r} threshold is inconsistent "
                "with content.thresholds"
            )


def _validate_checks(
    value: object,
    *,
    location: str,
    errors: list[str],
) -> tuple[bool, dict[str, Mapping[str, object]]]:
    if not isinstance(value, list) or not value:
        errors.append(f"{location} must be a non-empty array")
        return False, {}

    all_passed = True
    by_name: dict[str, Mapping[str, object]] = {}
    for index, raw_check in enumerate(value):
        check_location = f"{location}[{index}]"
        if not isinstance(raw_check, Mapping):
            errors.append(f"{check_location} must be an object")
            all_passed = False
            continue
        if set(raw_check) != {
            "name",
            "passed",
            "actual",
            "comparator",
            "threshold",
            "detail",
        }:
            errors.append(f"{check_location} keys do not match the v1 schema")
            all_passed = False
        name = raw_check.get("name")
        if type(name) is not str or not name:
            errors.append(f"{check_location}.name must be a non-empty string")
            all_passed = False
            continue
        if name in by_name:
            errors.append(f"{location} contains duplicate check {name!r}")
            all_passed = False
            continue
        by_name[name] = raw_check

        actual = _finite_number(raw_check.get("actual"))
        threshold = _finite_number(raw_check.get("threshold"))
        comparator = raw_check.get("comparator")
        declared_passed = raw_check.get("passed")
        if comparator == ">=":
            expected_passed = (
                actual is not None
                and threshold is not None
                and actual >= threshold
            )
        elif comparator == "<=":
            expected_passed = (
                actual is not None
                and threshold is not None
                and actual <= threshold
            )
        else:
            errors.append(f"{check_location}.comparator must be '>=' or '<='")
            expected_passed = False
        if not isinstance(declared_passed, bool):
            errors.append(f"{check_location}.passed must be boolean")
        elif declared_passed != expected_passed:
            errors.append(
                f"{check_location}.passed is inconsistent with its comparison"
            )
        all_passed = all_passed and expected_passed
    return all_passed, by_name


def validate_evidence_artifact(
    artifact: Mapping[str, object],
) -> ArtifactValidation:
    """Validate schema, required content, digest integrity, and acceptance."""

    errors: list[str] = []
    if set(artifact) != {
        "schema_version",
        "content",
        "content_digest",
        "operational_diagnostics",
    }:
        errors.append("artifact top-level keys do not match the v1 schema")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")

    content = _required_mapping(artifact, "content", errors)
    digest = _required_mapping(artifact, "content_digest", errors)
    operational = _required_mapping(
        artifact,
        "operational_diagnostics",
        errors,
    )

    scientific_passed = False
    operational_passed = False
    overall_passed = False

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
        for key in (
            "protocol",
            "configuration",
            "thresholds",
            "aggregate",
            "acceptance",
            "source_provenance",
        ):
            _required_mapping(content, key, errors)
        _validate_source_provenance(content.get("source_provenance"), errors)
        threshold_mapping = content.get("thresholds")
        if isinstance(threshold_mapping, Mapping):
            _validate_threshold_policy(threshold_mapping, errors)
        aggregate_mapping = content.get("aggregate")
        _validate_seed_and_aggregate_consistency(content, errors)

        acceptance = content.get("acceptance")
        if isinstance(acceptance, Mapping):
            if set(acceptance) != {
                "passed",
                "checks",
                "excluded_operational_checks",
            }:
                errors.append("content.acceptance keys do not match the v1 schema")
            passed = acceptance.get("passed")
            checks = acceptance.get("checks")
            if not isinstance(passed, bool):
                errors.append("content.acceptance.passed must be boolean")
            computed_passed, scientific_checks = _validate_checks(
                checks,
                location="content.acceptance.checks",
                errors=errors,
            )
            missing_checks = sorted(
                _REQUIRED_SCIENTIFIC_CHECKS - set(scientific_checks)
            )
            if missing_checks:
                errors.append(
                    "content.acceptance.checks is missing required checks: "
                    + ", ".join(missing_checks)
                )
            if isinstance(threshold_mapping, Mapping) and isinstance(
                aggregate_mapping,
                Mapping,
            ):
                _validate_scientific_check_bindings(
                    scientific_checks,
                    thresholds=threshold_mapping,
                    aggregate=aggregate_mapping,
                    errors=errors,
                )
            if isinstance(passed, bool) and passed != computed_passed:
                errors.append(
                    "content.acceptance.passed is inconsistent with its checks"
                )
            scientific_passed = computed_passed

    if digest is not None:
        if set(digest) != {
            "algorithm",
            "scope",
            "canonicalization",
            "sha256",
        }:
            errors.append("content_digest keys do not match the v1 schema")
        if digest.get("algorithm") != DIGEST_ALGORITHM:
            errors.append(f"content_digest.algorithm must be {DIGEST_ALGORITHM!r}")
        if digest.get("scope") != DIGEST_SCOPE:
            errors.append(f"content_digest.scope must be {DIGEST_SCOPE!r}")
        if digest.get("canonicalization") != CANONICALIZATION:
            errors.append(
                "content_digest.canonicalization has an unsupported value"
            )
        recorded = digest.get("sha256")
        if type(recorded) is not str or len(recorded) != 64:
            errors.append("content_digest.sha256 must be a 64-character string")
        elif content is not None:
            expected = scientific_content_sha256(content)
            if recorded != expected:
                errors.append("content_digest.sha256 does not match content")

    if operational is not None:
        if set(operational) != {
            "digest_exclusion_reason",
            "environment",
            "condition_timings",
            "maximum_update_latency_ms",
            "checks",
            "passed",
            "overall_acceptance_passed",
        }:
            errors.append("operational_diagnostics keys do not match the v1 schema")
        passed = operational.get("passed")
        overall = operational.get("overall_acceptance_passed")
        if not isinstance(passed, bool):
            errors.append("operational_diagnostics.passed must be boolean")
        computed_passed, operational_checks = _validate_checks(
            operational.get("checks"),
            location="operational_diagnostics.checks",
            errors=errors,
        )
        if isinstance(passed, bool) and passed != computed_passed:
            errors.append(
                "operational_diagnostics.passed is inconsistent with its checks"
            )
        operational_passed = computed_passed

        latency = _finite_number(
            operational.get("maximum_update_latency_ms")
        )
        latency_check = operational_checks.get("update_latency_ms")
        if latency_check is None:
            errors.append(
                "operational_diagnostics.checks must include update_latency_ms"
            )
        elif latency != _finite_number(latency_check.get("actual")):
            errors.append(
                "operational_diagnostics maximum latency is inconsistent "
                "with its check"
            )
        operational_thresholds = (
            content.get("thresholds") if content is not None else None
        )
        if latency_check is not None and isinstance(
            operational_thresholds,
            Mapping,
        ):
            if not _numbers_match(
                latency_check.get("threshold"),
                operational_thresholds.get("maximum_update_latency_ms"),
            ):
                errors.append(
                    "operational update latency threshold is inconsistent "
                    "with content.thresholds"
                )
            if latency_check.get("comparator") != "<=":
                errors.append(
                    "operational update latency comparator must be '<='"
                )

        if not isinstance(overall, bool):
            errors.append(
                "operational_diagnostics.overall_acceptance_passed must be boolean"
            )
        expected_overall = scientific_passed and operational_passed
        if isinstance(overall, bool) and overall != expected_overall:
            errors.append(
                "operational_diagnostics.overall_acceptance_passed is "
                "inconsistent with scientific and operational checks"
            )
        overall_passed = expected_overall

    valid = not errors
    accepted = (
        valid
        and scientific_passed
        and operational_passed
        and overall_passed
    )
    return ArtifactValidation(
        valid=valid,
        accepted=accepted,
        errors=tuple(errors),
    )
