"""Versioned, fail-closed artifacts for the recurring pair-feature gate.

Only deterministic scientific content is covered by the canonical SHA-256
digest.  Generation time, benchmark wall time, runtime packages, devices, and
worktree dirtiness remain inspectable under ``operational_metadata`` but are
outside the digest.

The frozen v1 protocol was calibrated on development seeds 0--29.  Promoted
evidence uses the disjoint schedule 30--59 exactly.  Descriptive bootstrap
intervals do not alter the already-frozen acceptance thresholds.

This artifact supports one narrow claim: under supplied output heads and
Gaussian pair-product/L2 targets, utility retention preserves three recurring
pair descriptors in a three-slot deployed bank while evicting a one-off pair
from that bank.  The exhaustive all-pairs candidate archive, its descriptors,
and its per-head candidate weights remain counted memory.  Active-bank eviction
is not erasure, autonomous or general feature discovery, continual control, or
evidence of indefinite memory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import NoReturn

import jax
import numpy as np

from alberta_framework.recurring_feature_gate import (
    CRITICAL_TASKS,
    DEVELOPMENT_SEEDS,
    EVIDENCE_SEEDS,
    PAIRWISE_PROBE_SCOPE,
    PHASE_TASKS,
    TASK_NAMES,
    TASK_PAIRS,
    FeatureMemoryBudget,
    RecurringFeatureGateCriteria,
    RecurringFeatureGateResult,
    RecurringFeatureProtocol,
    RecurringFeatureSeedEvidence,
    RecurringFeatureVariantEvidence,
)

SCHEMA_VERSION = "alberta.recurring_feature_evidence.v1"
PROTOCOL_VERSION = "recurring-pair-feature-active-bank-aba.v1"
DIGEST_ALGORITHM = "sha256"
DIGEST_SCOPE = "$.scientific_payload"
CANONICALIZATION = "utf8-json-sort-keys-compact-no-nan"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 2_026_073_002

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATHS = (
    Path("alberta_framework/recurring_feature_gate.py"),
    Path("alberta_framework/core/interaction_features.py"),
    Path("alberta_framework/evaluation/recurring_feature_artifact.py"),
    Path("alberta_framework/evaluation/recurring_feature_cli.py"),
)
_REQUIRED_CHECK_ORDER = (
    "canonical_protocol",
    "evidence_seed_schedule",
    "minimum_seed_count",
    "matched_unique_seeds",
    "minimum_heldout_samples",
    "finite_complete_seed_evidence",
    "exhaustive_counted_candidate_archive",
    "memory_budget_consistent",
    "retained_all_critical_rate",
    "obsolete_pair_active_bank_eviction_rate",
    "maximum_critical_task_median_nmse",
    "minimum_obsolete_task_median_nmse",
    "retention_rate_gain_over_no_retention",
    "critical_nmse_gain_over_no_retention",
    "recurrence_faster_than_acquisition",
    "upstream_gate_decision",
)
_REQUIRED_CHECKS = frozenset(_REQUIRED_CHECK_ORDER)


@dataclass(frozen=True)
class ArtifactValidation:
    """Integrity and scientific-acceptance decision for one parsed artifact."""

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
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def canonical_scientific_bytes(payload: Mapping[str, object]) -> bytes:
    """Return the declared canonical representation for digesting."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def scientific_payload_sha256(payload: Mapping[str, object]) -> str:
    """Hash canonical deterministic scientific content."""

    return hashlib.sha256(canonical_scientific_bytes(payload)).hexdigest()


def _git_head() -> str:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to resolve git HEAD for provenance") from error
    commit = result.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("git HEAD is not a canonical 40-character SHA-1")
    return commit


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=normal"),
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to inspect git worktree provenance") from error
    return bool(result.stdout)


def _source_provenance() -> dict[str, object]:
    hashes: dict[str, str] = {}
    for relative_path in _SOURCE_PATHS:
        source_path = _REPO_ROOT / relative_path
        try:
            source_bytes = source_path.read_bytes()
        except OSError as error:
            raise RuntimeError(f"unable to hash source {relative_path}") from error
        hashes[relative_path.as_posix()] = hashlib.sha256(source_bytes).hexdigest()
    return {
        "repository_subtree": "research/alberta",
        "git_head": _git_head(),
        "source_sha256": hashes,
        "interpretation": (
            "reproducibility and integrity identifiers only; hashes are not "
            "signatures and do not establish artifact authenticity"
        ),
    }


def _runtime_provenance() -> dict[str, object]:
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
            "alberta-framework": importlib.metadata.version("alberta-framework"),
            "jax": jax.__version__,
            "jaxlib": importlib.metadata.version("jaxlib"),
            "numpy": np.__version__,
        },
        "jax": {
            "default_backend": jax.default_backend(),
            "devices": [
                {
                    "platform": device.platform,
                    "device_kind": device.device_kind,
                }
                for device in jax.devices()
            ],
        },
    }


def _protocol_payload() -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "supported_claim": (
            "retention and deployed-bank allocation of supplied pair-product "
            "features in the frozen Gaussian/L2 recurring probe"
        ),
        "schedule": list(PHASE_TASKS),
        "stream": (
            "one uninterrupted predict-before-update stream of iid Gaussian "
            "vectors with supplied output heads and squared-error targets"
        ),
        "search_space": (
            "exhaustive all-pairs candidate descriptors over six supplied input coordinates"
        ),
        "memory_semantics": (
            "active-bank eviction removes a pair only from the deployed "
            "three-slot bank; all 15 candidate descriptors and their 60 "
            "per-head candidate weights remain in the counted exhaustive archive"
        ),
        "seed_roles": {
            "development_and_threshold_calibration": list(DEVELOPMENT_SEEDS),
            "promoted_held_out_evidence": list(EVIDENCE_SEEDS),
            "rule": (
                "configuration and thresholds were calibrated on seeds 0-29; "
                "the promoted schedule is seeds 30-59 and is not selected or "
                "retuned after observing its results"
            ),
        },
        "supplied_structure": {
            "output_heads": list(TASK_NAMES),
            "target_pairs": [list(pair) for pair in TASK_PAIRS],
            "critical_recurring_heads": list(CRITICAL_TASKS),
            "obsolete_one_off_head": "D",
            "observation_distribution": "iid standard Gaussian",
            "loss": "per-head squared error with normalized-MSE evaluation",
        },
        "excluded_claims": [
            "erasure from the exhaustive candidate archive",
            "autonomous target or output-head discovery",
            "general feature discovery",
            "general catastrophic-forgetting resistance",
            "continual control",
            "indefinite-memory learning",
            "completion of the Alberta Plan",
        ],
        "upstream_scope": PAIRWISE_PROBE_SCOPE,
    }


def _configuration_payload(protocol: RecurringFeatureProtocol) -> dict[str, object]:
    return dict(dataclasses.asdict(protocol))


def _criteria_payload(criteria: RecurringFeatureGateCriteria) -> dict[str, object]:
    return dict(dataclasses.asdict(criteria))


def _memory_payload(budget: FeatureMemoryBudget) -> dict[str, object]:
    return {
        "active_pair_descriptor_slots": budget.active_pair_slots,
        "candidate_pair_descriptor_slots": budget.candidate_pair_slots,
        "total_pair_descriptor_slots": budget.total_pair_slots,
        "output_heads": budget.output_heads,
        "active_output_weight_slots": budget.active_output_weight_slots,
        "candidate_output_weight_slots": budget.candidate_output_weight_slots,
        "total_output_weight_slots": budget.total_output_weight_slots,
        "candidate_archive_is_counted_memory": True,
    }


def _phase_payload(seed: RecurringFeatureSeedEvidence) -> list[dict[str, object]]:
    return [
        {
            "phase_index": phase.phase_index,
            "task": phase.task,
            "occurrence": phase.occurrence,
            "prequential_nmse": _finite_float(phase.prequential_nmse),
            "recovery_steps": phase.recovery_steps,
        }
        for phase in seed.phase_evidence
    ]


def _recovery_payload(seed: RecurringFeatureSeedEvidence) -> list[dict[str, object]]:
    return [
        {
            "task": recovery.task,
            "acquisition_steps": recovery.acquisition_steps,
            "recurrence_steps": list(recovery.recurrence_steps),
        }
        for recovery in seed.task_recovery
    ]


def _variant_seed_payload(seed: RecurringFeatureSeedEvidence) -> dict[str, object]:
    candidate_set = set(seed.candidate_pairs)
    expected_candidates = {(left, right) for left in range(6) for right in range(left + 1, 6)}
    return {
        "final_heldout_nmse": {
            task: _finite_float(seed.final_heldout_nmse[index])
            for index, task in enumerate(TASK_NAMES)
        },
        "active_pairs": [list(pair) for pair in seed.active_pairs],
        "candidate_pairs": [list(pair) for pair in seed.candidate_pairs],
        "critical_pairs_retained": list(seed.critical_pairs_retained),
        "retained_all_critical_pairs": seed.retained_all_critical_pairs,
        "obsolete_pair_evicted_from_active_bank": seed.obsolete_pair_evicted,
        "obsolete_pair_remains_in_candidate_archive": TASK_PAIRS[3] in candidate_set,
        "candidate_archive_is_exhaustive_all_pairs": candidate_set == expected_candidates,
        "steps_seen": seed.steps_seen,
        "phase_evidence": _phase_payload(seed),
        "task_recovery": _recovery_payload(seed),
    }


def _seed_payloads(result: RecurringFeatureGateResult) -> list[dict[str, object]]:
    retained = {seed.seed: seed for seed in result.retained.seeds}
    baseline = {seed.seed: seed for seed in result.no_retention.seeds}
    if len(retained) != len(result.retained.seeds):
        raise ValueError("retained evidence contains duplicate seeds")
    if len(baseline) != len(result.no_retention.seeds):
        raise ValueError("no-retention evidence contains duplicate seeds")
    if set(retained) != set(baseline):
        raise ValueError("retained and no-retention seed schedules differ")
    return [
        {
            "seed": seed,
            "retained": _variant_seed_payload(retained[seed]),
            "no_retention": _variant_seed_payload(baseline[seed]),
        }
        for seed in sorted(retained)
    ]


def _variant_aggregate_payload(
    variant: RecurringFeatureVariantEvidence,
) -> dict[str, object]:
    return {
        "name": variant.name,
        "utility_retention_decay": variant.utility_retention_decay,
        "all_critical_retention_rate": _finite_float(variant.all_critical_retention_rate),
        "obsolete_active_bank_eviction_rate": _finite_float(variant.obsolete_eviction_rate),
        "median_heldout_nmse_by_task": {
            task: _finite_float(value)
            for task, value in zip(
                TASK_NAMES,
                variant.median_heldout_nmse_by_task,
                strict=True,
            )
        },
        "maximum_critical_task_median_nmse": _finite_float(
            variant.maximum_critical_task_median_nmse
        ),
        "median_obsolete_heldout_nmse": _finite_float(variant.median_obsolete_heldout_nmse),
        "median_acquisition_recovery_steps": _finite_float(
            variant.median_acquisition_recovery_steps
        ),
        "median_recurrence_recovery_steps": _finite_float(variant.median_recurrence_recovery_steps),
    }


def paired_bootstrap_mean_interval(
    differences: Sequence[float],
    *,
    seed_offset: int,
) -> dict[str, object]:
    """Deterministic percentile interval over paired per-seed differences."""

    values = np.asarray(tuple(differences), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired bootstrap requires a non-empty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("paired bootstrap values must be finite")
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = generator.integers(
        0,
        values.size,
        size=(BOOTSTRAP_RESAMPLES, values.size),
    )
    means = values[indices].mean(axis=1)
    tail = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    lower, upper = np.quantile(means, (tail, 1.0 - tail))
    return {
        "estimate": float(values.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "resamples": BOOTSTRAP_RESAMPLES,
        "sample_size": int(values.size),
        "method": "paired-percentile-bootstrap",
        "pairing_unit": "seed",
        "acceptance_role": "descriptive-only; frozen gate uses point criteria",
    }


def wilson_score_interval(successes: int, sample_size: int) -> dict[str, object]:
    """Wilson interval for a reported seed-level proportion."""

    if sample_size < 1 or not 0 <= successes <= sample_size:
        raise ValueError("invalid Wilson interval counts")
    proportion = successes / sample_size
    z_score = NormalDist().inv_cdf(0.5 + BOOTSTRAP_CONFIDENCE_LEVEL / 2.0)
    z_squared = z_score**2
    denominator = 1.0 + z_squared / sample_size
    center = (proportion + z_squared / (2.0 * sample_size)) / denominator
    half_width = (
        z_score
        * math.sqrt(
            proportion * (1.0 - proportion) / sample_size + z_squared / (4.0 * sample_size**2)
        )
        / denominator
    )
    return {
        "estimate": proportion,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "successes": successes,
        "sample_size": sample_size,
        "method": "wilson-score",
        "acceptance_role": "descriptive-only; frozen gate uses observed rate",
    }


def _per_seed_recovery_medians(
    seed: RecurringFeatureSeedEvidence,
) -> tuple[float, float] | None:
    acquisitions = [
        recovery.acquisition_steps
        for recovery in seed.task_recovery
        if recovery.task in CRITICAL_TASKS and recovery.acquisition_steps is not None
    ]
    recurrences = [
        step
        for recovery in seed.task_recovery
        if recovery.task in CRITICAL_TASKS
        for step in recovery.recurrence_steps
        if step is not None
    ]
    if not acquisitions or not recurrences:
        return None
    return (
        float(np.median(np.asarray(acquisitions, dtype=np.float64))),
        float(np.median(np.asarray(recurrences, dtype=np.float64))),
    )


def _paired_effect_payload(result: RecurringFeatureGateResult) -> dict[str, object]:
    retained = {seed.seed: seed for seed in result.retained.seeds}
    baseline = {seed.seed: seed for seed in result.no_retention.seeds}
    seeds = sorted(set(retained) & set(baseline))
    retention_differences = [
        float(retained[seed].retained_all_critical_pairs)
        - float(baseline[seed].retained_all_critical_pairs)
        for seed in seeds
    ]
    critical_nmse_differences = [
        max(baseline[seed].final_heldout_nmse[:3]) - max(retained[seed].final_heldout_nmse[:3])
        for seed in seeds
    ]
    obsolete_nmse_differences = [
        retained[seed].final_heldout_nmse[3] - baseline[seed].final_heldout_nmse[3]
        for seed in seeds
    ]
    recovery_differences: list[float] = []
    for seed in seeds:
        recovery = _per_seed_recovery_medians(retained[seed])
        if recovery is not None:
            acquisition, recurrence = recovery
            recovery_differences.append(acquisition - recurrence)
    return {
        "retention_rate_gain_over_no_retention": paired_bootstrap_mean_interval(
            retention_differences,
            seed_offset=0,
        ),
        "per_seed_maximum_critical_nmse_reduction": (
            paired_bootstrap_mean_interval(
                critical_nmse_differences,
                seed_offset=1,
            )
        ),
        "obsolete_nmse_increase_over_no_retention": (
            paired_bootstrap_mean_interval(
                obsolete_nmse_differences,
                seed_offset=2,
            )
        ),
        "retained_acquisition_minus_recurrence_steps": (
            paired_bootstrap_mean_interval(
                recovery_differences,
                seed_offset=3,
            )
        ),
    }


def _aggregate_payload(result: RecurringFeatureGateResult) -> dict[str, object]:
    seeds = tuple(seed.seed for seed in result.retained.seeds)
    retained_successes = sum(seed.retained_all_critical_pairs for seed in result.retained.seeds)
    eviction_successes = sum(seed.obsolete_pair_evicted for seed in result.retained.seeds)
    return {
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "retained": _variant_aggregate_payload(result.retained),
        "no_retention": _variant_aggregate_payload(result.no_retention),
        "retained_all_critical_rate_interval": wilson_score_interval(
            retained_successes,
            len(seeds),
        ),
        "retained_obsolete_active_bank_eviction_rate_interval": (
            wilson_score_interval(eviction_successes, len(seeds))
        ),
        "paired_effects": _paired_effect_payload(result),
    }


def _comparison_passes(
    actual: float | None,
    comparator: str,
    threshold: float,
) -> bool:
    if actual is None or not math.isfinite(actual) or not math.isfinite(threshold):
        return False
    if comparator == ">=":
        return actual >= threshold
    if comparator == "<=":
        return actual <= threshold
    if comparator == ">":
        return actual > threshold
    if comparator == "==":
        return actual == threshold
    raise ValueError(f"unsupported comparator {comparator!r}")


def _check(
    name: str,
    actual: float | None,
    comparator: str,
    threshold: float,
    detail: str,
) -> dict[str, object]:
    return {
        "name": name,
        "passed": _comparison_passes(actual, comparator, threshold),
        "actual": actual,
        "comparator": comparator,
        "threshold": threshold,
        "detail": detail,
    }


def _result_integrity(result: RecurringFeatureGateResult) -> tuple[bool, bool]:
    expected_candidates = {
        (left, right)
        for left in range(result.protocol.feature_dim)
        for right in range(left + 1, result.protocol.feature_dim)
    }
    finite_complete = True
    archive_complete = True
    for variant in (result.retained, result.no_retention):
        for seed in variant.seeds:
            finite_complete = finite_complete and (
                len(seed.final_heldout_nmse) == result.protocol.output_heads
                and all(math.isfinite(value) and value >= 0.0 for value in seed.final_heldout_nmse)
                and len(seed.phase_evidence) == len(PHASE_TASKS)
                and all(
                    math.isfinite(phase.prequential_nmse) and phase.prequential_nmse >= 0.0
                    for phase in seed.phase_evidence
                )
                and seed.steps_seen == result.protocol.total_steps
            )
            archive_complete = archive_complete and (
                len(seed.candidate_pairs) == result.protocol.candidate_pair_budget
                and set(seed.candidate_pairs) == expected_candidates
                and TASK_PAIRS[3] in set(seed.candidate_pairs)
            )
    return finite_complete, archive_complete


def _acceptance_payload(
    result: RecurringFeatureGateResult,
    criteria: RecurringFeatureGateCriteria,
) -> dict[str, object]:
    retained_ids = tuple(seed.seed for seed in result.retained.seeds)
    baseline_ids = tuple(seed.seed for seed in result.no_retention.seeds)
    finite_complete, archive_complete = _result_integrity(result)
    budget = result.memory_budget
    protocol = result.protocol
    canonical_protocol = protocol == RecurringFeatureProtocol()
    matched_unique = (
        retained_ids == baseline_ids
        and len(set(retained_ids)) == len(retained_ids)
        and tuple(sorted(retained_ids)) == retained_ids
    )
    schedule_matches = retained_ids == EVIDENCE_SEEDS and baseline_ids == EVIDENCE_SEEDS
    budget_consistent = (
        budget.active_pair_slots == protocol.active_pair_budget
        and budget.candidate_pair_slots == protocol.candidate_pair_budget
        and budget.output_heads == protocol.output_heads
    )
    retention_gain = (
        result.retained.all_critical_retention_rate
        - result.no_retention.all_critical_retention_rate
    )
    nmse_gain = (
        result.no_retention.maximum_critical_task_median_nmse
        - result.retained.maximum_critical_task_median_nmse
    )
    recovery_gain = (
        result.retained.median_acquisition_recovery_steps
        - result.retained.median_recurrence_recovery_steps
    )
    upstream = result.decision(criteria)
    checks = [
        _check(
            "canonical_protocol",
            float(canonical_protocol),
            "==",
            1.0,
            "Every v1 protocol field, including 4096 held-out samples, is frozen.",
        ),
        _check(
            "evidence_seed_schedule",
            float(schedule_matches),
            "==",
            1.0,
            "Promoted evidence must use seeds 30-59 exactly.",
        ),
        _check(
            "minimum_seed_count",
            float(len(retained_ids)),
            ">=",
            float(criteria.minimum_seeds),
            "At least thirty unique paired evidence seeds are required.",
        ),
        _check(
            "matched_unique_seeds",
            float(matched_unique),
            "==",
            1.0,
            "Retained and no-retention variants must use the same ordered unique seeds.",
        ),
        _check(
            "minimum_heldout_samples",
            float(protocol.heldout_samples),
            ">=",
            float(criteria.minimum_heldout_samples),
            "Each seed requires the frozen independent held-out sample count.",
        ),
        _check(
            "finite_complete_seed_evidence",
            float(finite_complete),
            "==",
            1.0,
            "Every seed must contain finite NMSE, all phases, and all online updates.",
        ),
        _check(
            "exhaustive_counted_candidate_archive",
            float(archive_complete),
            "==",
            1.0,
            "All 15 pair descriptors, including obsolete D, remain in counted archive memory.",
        ),
        _check(
            "memory_budget_consistent",
            float(budget_consistent),
            "==",
            1.0,
            "Reported descriptor and per-head weight slots must match the protocol.",
        ),
        _check(
            "retained_all_critical_rate",
            result.retained.all_critical_retention_rate,
            ">=",
            criteria.minimum_retained_all_critical_rate,
            "Observed fraction with A, B, and C in the deployed active bank.",
        ),
        _check(
            "obsolete_pair_active_bank_eviction_rate",
            result.retained.obsolete_eviction_rate,
            ">=",
            criteria.minimum_obsolete_eviction_rate,
            "D must leave the deployed bank; this is not archive erasure.",
        ),
        _check(
            "maximum_critical_task_median_nmse",
            result.retained.maximum_critical_task_median_nmse,
            "<=",
            criteria.maximum_median_critical_nmse,
            "Worst task-wise held-out median across recurring heads A, B, and C.",
        ),
        _check(
            "minimum_obsolete_task_median_nmse",
            result.retained.median_obsolete_heldout_nmse,
            ">=",
            criteria.minimum_median_obsolete_nmse,
            "One-off head D must no longer be accurately served by the deployed bank.",
        ),
        _check(
            "retention_rate_gain_over_no_retention",
            retention_gain,
            ">=",
            criteria.minimum_retention_rate_gain_over_baseline,
            "Matched seed-level active-bank retention-rate difference.",
        ),
        _check(
            "critical_nmse_gain_over_no_retention",
            nmse_gain,
            ">=",
            criteria.minimum_critical_nmse_gain_over_baseline,
            "Matched protocol difference in worst task-wise critical median NMSE.",
        ),
        _check(
            "recurrence_faster_than_acquisition",
            recovery_gain,
            ">",
            0.0,
            "Retained median acquisition steps minus recurrence steps must be positive.",
        ),
        _check(
            "upstream_gate_decision",
            float(upstream.accepted),
            "==",
            1.0,
            "The pinned upstream gate must independently accept the same result.",
        ),
    ]
    passed = all(check["passed"] is True for check in checks)
    if upstream.accepted != passed:
        raise RuntimeError("versioned artifact checks disagree with the pinned upstream gate")
    return {
        "passed": passed,
        "checks": checks,
        "upstream_summary": upstream.summary,
        "upstream_failures": list(upstream.failures),
    }


def _scientific_payload(result: RecurringFeatureGateResult) -> dict[str, object]:
    criteria = RecurringFeatureGateCriteria()
    return {
        "protocol": _protocol_payload(),
        "configuration": _configuration_payload(result.protocol),
        "criteria": _criteria_payload(criteria),
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "seed": BOOTSTRAP_SEED,
            "role": "descriptive paired uncertainty; not threshold calibration",
        },
        "memory_budget": _memory_payload(result.memory_budget),
        "seed_summaries": _seed_payloads(result),
        "aggregate": _aggregate_payload(result),
        "acceptance": _acceptance_payload(result, criteria),
        "source_provenance": _source_provenance(),
    }


def _operational_payload(
    *,
    gate_wall_seconds: float,
    generated_at: datetime | None,
) -> dict[str, object]:
    if not math.isfinite(gate_wall_seconds) or gate_wall_seconds < 0.0:
        raise ValueError("gate_wall_seconds must be finite and non-negative")
    timestamp = datetime.now(UTC) if generated_at is None else generated_at
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    return {
        "digest_exclusion_reason": (
            "timestamp, host timing, runtime environment, devices, and worktree "
            "dirtiness are operational provenance rather than deterministic evidence"
        ),
        "generated_at_utc": timestamp.astimezone(UTC).isoformat(),
        "gate_wall_seconds": gate_wall_seconds,
        "runtime": _runtime_provenance(),
        "git_worktree": {
            "head": _git_head(),
            "dirty": _git_dirty(),
        },
    }


def build_recurring_feature_artifact(
    result: RecurringFeatureGateResult,
    *,
    gate_wall_seconds: float,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Build a versioned artifact without trusting a cached pass flag."""

    scientific_payload = _scientific_payload(result)
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_payload": scientific_payload,
        "scientific_digest": {
            "algorithm": DIGEST_ALGORITHM,
            "scope": DIGEST_SCOPE,
            "canonicalization": CANONICALIZATION,
            "sha256": scientific_payload_sha256(scientific_payload),
        },
        "operational_metadata": _operational_payload(
            gate_wall_seconds=gate_wall_seconds,
            generated_at=generated_at,
        ),
    }


def artifact_json(artifact: Mapping[str, object]) -> str:
    """Serialize strict, human-readable standard JSON."""

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
    parsed: dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON object key: {key}")
        parsed[key] = value
    return parsed


def load_recurring_feature_artifact(path: Path) -> dict[str, object]:
    """Load strict JSON and require a top-level object."""

    parsed = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
        parse_float=_parse_finite_json_float,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(parsed, dict):
        raise ValueError("evidence artifact must be a JSON object")
    return parsed


def _mapping(
    parent: Mapping[str, object],
    key: str,
    location: str,
    errors: list[str],
) -> Mapping[str, object] | None:
    value = parent.get(key)
    if type(value) is not dict:
        errors.append(f"{location}.{key} must be an object")
        return None
    return value


def _finite_number(value: object) -> float | None:
    if type(value) is not int and type(value) is not float:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _validate_source_provenance(
    provenance: Mapping[str, object],
    errors: list[str],
) -> None:
    try:
        expected = _source_provenance()
    except RuntimeError as error:
        errors.append(str(error))
        return
    if set(provenance) != set(expected):
        errors.append("scientific_payload.source_provenance keys do not match the v1 schema")
    if provenance.get("repository_subtree") != expected["repository_subtree"]:
        errors.append("scientific_payload.source_provenance repository is invalid")
    provenance_head = provenance.get("git_head")
    if (
        not isinstance(provenance_head, str)
        or len(provenance_head) != 40
        or any(character not in "0123456789abcdef" for character in provenance_head)
    ):
        errors.append("scientific_payload.source_provenance.git_head must be a canonical SHA-1")
    if provenance.get("source_sha256") != expected["source_sha256"]:
        errors.append(
            "scientific_payload.source_provenance does not match current pinned source hashes"
        )
    if provenance.get("interpretation") != expected["interpretation"]:
        errors.append("scientific_payload.source_provenance interpretation is invalid")


def _extract_seed_metrics(
    summaries: object,
    errors: list[str],
) -> tuple[list[int], dict[str, list[dict[str, object]]]]:
    if not isinstance(summaries, list) or not summaries:
        errors.append("scientific_payload.seed_summaries must be a non-empty array")
        return [], {"retained": [], "no_retention": []}
    seeds: list[int] = []
    variants: dict[str, list[dict[str, object]]] = {
        "retained": [],
        "no_retention": [],
    }
    for index, raw_summary in enumerate(summaries):
        location = f"scientific_payload.seed_summaries[{index}]"
        if not isinstance(raw_summary, Mapping):
            errors.append(f"{location} must be an object")
            continue
        if set(raw_summary) != {"seed", "retained", "no_retention"}:
            errors.append(f"{location} keys do not match the v1 schema")
        seed = raw_summary.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            errors.append(f"{location}.seed must be an integer")
            continue
        seeds.append(seed)
        for name in variants:
            raw_variant = raw_summary.get(name)
            if not isinstance(raw_variant, Mapping):
                errors.append(f"{location}.{name} must be an object")
                continue
            variants[name].append(dict(raw_variant))
    if seeds != sorted(seeds) or len(set(seeds)) != len(seeds):
        errors.append("scientific_payload seed summaries must be strictly increasing and unique")
    if any(len(values) != len(seeds) for values in variants.values()):
        errors.append("each seed summary must contain both learner variants")
    return seeds, variants


def _nmse_values(
    variants: Sequence[dict[str, object]],
    errors: list[str],
    name: str,
) -> dict[str, list[float]]:
    nmse_values: dict[str, list[float]] = {task: [] for task in TASK_NAMES}
    for index, variant in enumerate(variants):
        raw_nmse = variant.get("final_heldout_nmse")
        if not isinstance(raw_nmse, Mapping):
            errors.append(f"{name}[{index}].final_heldout_nmse must be an object")
            continue
        if set(raw_nmse) != set(TASK_NAMES):
            errors.append(f"{name}[{index}].final_heldout_nmse keys are not A/B/C/D")
        for task in TASK_NAMES:
            value = _finite_number(raw_nmse.get(task))
            if value is None or value < 0.0:
                errors.append(f"{name}[{index}] has invalid {task} NMSE")
            else:
                nmse_values[task].append(value)
    return nmse_values


def _pair_set(value: object) -> set[tuple[int, int]] | None:
    if type(value) is not list:
        return None
    pairs: set[tuple[int, int]] = set()
    for pair in value:
        if (
            type(pair) is not list
            or len(pair) != 2
            or any(type(item) is not int for item in pair)
        ):
            return None
        pairs.add((pair[0], pair[1]))
    return pairs


def _valid_optional_recovery_step(value: object, *, maximum: int) -> bool:
    return value is None or (
        type(value) is int and 1 <= value <= maximum
    )


def _valid_task_recovery_record(
    value: object,
    *,
    expected_task: str,
    maximum_step: int,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "task",
        "acquisition_steps",
        "recurrence_steps",
    }:
        return False
    recurrence_steps = value.get("recurrence_steps")
    return (
        value.get("task") == expected_task
        and _valid_optional_recovery_step(
            value.get("acquisition_steps"),
            maximum=maximum_step,
        )
        and isinstance(recurrence_steps, list)
        and all(
            _valid_optional_recovery_step(step, maximum=maximum_step) for step in recurrence_steps
        )
    )


def _recomputed_aggregate(
    seeds: list[int],
    variants: dict[str, list[dict[str, object]]],
    errors: list[str],
) -> dict[str, object] | None:
    if not seeds or any(len(values) != len(seeds) for values in variants.values()):
        return None
    initial_error_count = len(errors)
    protocol = RecurringFeatureProtocol()
    expected_candidates = {(left, right) for left in range(6) for right in range(left + 1, 6)}
    parsed_nmse: dict[str, dict[str, list[float]]] = {
        variant_name: _nmse_values(variant_records, errors, variant_name)
        for variant_name, variant_records in variants.items()
    }
    for variant_name, task_values_by_name in parsed_nmse.items():
        if any(len(task_values) != len(seeds) for task_values in task_values_by_name.values()):
            errors.append(f"{variant_name} NMSE evidence is incomplete")
            return None

    retention: dict[str, list[float]] = {variant_name: [] for variant_name in variants}
    evictions: dict[str, list[float]] = {variant_name: [] for variant_name in variants}
    expected_variant_keys = {
        "final_heldout_nmse",
        "active_pairs",
        "candidate_pairs",
        "critical_pairs_retained",
        "retained_all_critical_pairs",
        "obsolete_pair_evicted_from_active_bank",
        "obsolete_pair_remains_in_candidate_archive",
        "candidate_archive_is_exhaustive_all_pairs",
        "steps_seen",
        "phase_evidence",
        "task_recovery",
    }
    phase_occurrences: dict[str, int] = {task: 0 for task in TASK_NAMES}
    expected_occurrences: list[int] = []
    for task in PHASE_TASKS:
        phase_occurrences[task] += 1
        expected_occurrences.append(phase_occurrences[task])

    for variant_name, variant_records in variants.items():
        for index, variant_record in enumerate(variant_records):
            location = f"{variant_name}[{index}]"
            if set(variant_record) != expected_variant_keys:
                errors.append(f"{location} keys do not match the v1 schema")

            raw_candidates = variant_record.get("candidate_pairs")
            raw_active = variant_record.get("active_pairs")
            candidates = _pair_set(raw_candidates)
            active = _pair_set(raw_active)
            candidate_archive_valid = (
                isinstance(raw_candidates, list)
                and len(raw_candidates) == protocol.candidate_pair_budget
                and candidates == expected_candidates
            )
            active_bank_valid = (
                isinstance(raw_active, list)
                and len(raw_active) == protocol.active_pair_budget
                and active is not None
                and len(active) == protocol.active_pair_budget
                and active <= expected_candidates
            )
            if not candidate_archive_valid:
                errors.append(f"{location} does not preserve the exhaustive counted archive")
            if not active_bank_valid:
                errors.append(f"{location} has an invalid three-slot active bank")

            active_pairs = active if active_bank_valid and active is not None else set()
            expected_critical = [pair in active_pairs for pair in TASK_PAIRS[:3]]
            expected_retained = active_bank_valid and all(expected_critical)
            expected_evicted = active_bank_valid and TASK_PAIRS[3] not in active_pairs
            expected_obsolete_present = (
                candidate_archive_valid and candidates is not None and TASK_PAIRS[3] in candidates
            )
            retention[variant_name].append(float(expected_retained))
            evictions[variant_name].append(float(expected_evicted))
            declared_critical = variant_record.get("critical_pairs_retained")
            if (
                not isinstance(declared_critical, list)
                or len(declared_critical) != len(CRITICAL_TASKS)
                or not all(isinstance(value, bool) for value in declared_critical)
                or declared_critical != expected_critical
            ):
                errors.append(f"{location}.critical_pairs_retained disagrees with active_pairs")
            if variant_record.get("retained_all_critical_pairs") is not expected_retained:
                errors.append(f"{location}.retained_all_critical_pairs disagrees with active_pairs")
            if variant_record.get("obsolete_pair_evicted_from_active_bank") is not expected_evicted:
                errors.append(
                    f"{location}.obsolete_pair_evicted_from_active_bank disagrees with active_pairs"
                )
            if (
                variant_record.get("candidate_archive_is_exhaustive_all_pairs")
                is not candidate_archive_valid
            ):
                errors.append(
                    f"{location}.candidate_archive_is_exhaustive_all_pairs "
                    "disagrees with candidate_pairs"
                )
            if (
                variant_record.get("obsolete_pair_remains_in_candidate_archive")
                is not expected_obsolete_present
            ):
                errors.append(
                    f"{location}.obsolete_pair_remains_in_candidate_archive "
                    "disagrees with candidate_pairs"
                )

            steps_seen = variant_record.get("steps_seen")
            if (
                isinstance(steps_seen, bool)
                or not isinstance(steps_seen, int)
                or steps_seen != protocol.total_steps
            ):
                errors.append(f"{location}.steps_seen is not the uninterrupted update count")

            raw_phases = variant_record.get("phase_evidence")
            derived_recovery: dict[str, list[int | None]] = {task: [] for task in TASK_NAMES}
            phases_valid = isinstance(raw_phases, list) and len(raw_phases) == len(PHASE_TASKS)
            if not phases_valid:
                errors.append(f"{location}.phase_evidence is not the complete frozen schedule")
            if isinstance(raw_phases, list):
                for phase_index, raw_phase in enumerate(raw_phases):
                    phase_location = f"{location}.phase_evidence[{phase_index}]"
                    if not isinstance(raw_phase, Mapping):
                        errors.append(f"{phase_location} must be an object")
                        continue
                    if set(raw_phase) != {
                        "phase_index",
                        "task",
                        "occurrence",
                        "prequential_nmse",
                        "recovery_steps",
                    }:
                        errors.append(f"{phase_location} keys do not match the v1 schema")
                    if phase_index >= len(PHASE_TASKS):
                        errors.append(f"{phase_location} exceeds the frozen schedule")
                        continue
                    expected_task = PHASE_TASKS[phase_index]
                    declared_phase_index = raw_phase.get("phase_index")
                    if (
                        isinstance(declared_phase_index, bool)
                        or not isinstance(declared_phase_index, int)
                        or declared_phase_index != phase_index
                    ):
                        errors.append(f"{phase_location}.phase_index is inconsistent")
                    if raw_phase.get("task") != expected_task:
                        errors.append(f"{phase_location}.task is inconsistent")
                    occurrence = raw_phase.get("occurrence")
                    if (
                        isinstance(occurrence, bool)
                        or not isinstance(occurrence, int)
                        or occurrence != expected_occurrences[phase_index]
                    ):
                        errors.append(f"{phase_location}.occurrence is inconsistent")
                    phase_nmse = _finite_number(raw_phase.get("prequential_nmse"))
                    if phase_nmse is None or phase_nmse < 0.0:
                        errors.append(f"{phase_location}.prequential_nmse is invalid")
                    recovery_step = raw_phase.get("recovery_steps")
                    recovery_valid = _valid_optional_recovery_step(
                        recovery_step,
                        maximum=protocol.steps_per_phase,
                    )
                    if not recovery_valid:
                        errors.append(f"{phase_location}.recovery_steps is invalid")
                    else:
                        derived_recovery[expected_task].append(recovery_step)

            expected_task_recovery = [
                {
                    "task": task,
                    "acquisition_steps": (
                        derived_recovery[task][0] if derived_recovery[task] else None
                    ),
                    "recurrence_steps": (
                        derived_recovery[task][1:] if derived_recovery[task] else []
                    ),
                }
                for task in TASK_NAMES
            ]
            raw_task_recovery = variant_record.get("task_recovery")
            task_recovery_schema_valid = (
                isinstance(raw_task_recovery, list)
                and len(raw_task_recovery) == len(TASK_NAMES)
                and all(
                    _valid_task_recovery_record(
                        record,
                        expected_task=TASK_NAMES[task_index],
                        maximum_step=protocol.steps_per_phase,
                    )
                    for task_index, record in enumerate(raw_task_recovery)
                )
            )
            if not task_recovery_schema_valid:
                errors.append(f"{location}.task_recovery does not match the v1 schema")
            if raw_task_recovery != expected_task_recovery:
                errors.append(f"{location}.task_recovery is not derived from phase_evidence")

    if len(errors) != initial_error_count:
        return None

    def variant_payload(name: str) -> dict[str, object]:
        medians = {task: float(np.median(parsed_nmse[name][task])) for task in TASK_NAMES}
        acquisition: list[float] = []
        recurrence: list[float] = []
        for index, variant_record in enumerate(variants[name]):
            raw_recovery = variant_record.get("task_recovery")
            if not isinstance(raw_recovery, list):
                errors.append(f"{name}[{index}].task_recovery must be an array")
                continue
            for task_record in raw_recovery:
                if not isinstance(task_record, Mapping):
                    continue
                if task_record.get("task") not in CRITICAL_TASKS:
                    continue
                acquisition_value = _finite_number(task_record.get("acquisition_steps"))
                if acquisition_value is not None:
                    acquisition.append(acquisition_value)
                recurrence_values = task_record.get("recurrence_steps")
                if isinstance(recurrence_values, list):
                    recurrence.extend(
                        value
                        for raw_value in recurrence_values
                        if (value := _finite_number(raw_value)) is not None
                    )
        return {
            "name": name,
            "utility_retention_decay": (
                RecurringFeatureProtocol().retained_utility_decay if name == "retained" else None
            ),
            "all_critical_retention_rate": float(np.mean(retention[name])),
            "obsolete_active_bank_eviction_rate": float(np.mean(evictions[name])),
            "median_heldout_nmse_by_task": medians,
            "maximum_critical_task_median_nmse": max(medians[task] for task in CRITICAL_TASKS),
            "median_obsolete_heldout_nmse": medians["D"],
            "median_acquisition_recovery_steps": (
                float(np.median(acquisition)) if acquisition else None
            ),
            "median_recurrence_recovery_steps": (
                float(np.median(recurrence)) if recurrence else None
            ),
        }

    retained_successes = int(sum(retention["retained"]))
    eviction_successes = int(sum(evictions["retained"]))
    critical_nmse_differences = [
        max(parsed_nmse["no_retention"][task][index] for task in CRITICAL_TASKS)
        - max(parsed_nmse["retained"][task][index] for task in CRITICAL_TASKS)
        for index in range(len(seeds))
    ]
    obsolete_differences = [
        parsed_nmse["retained"]["D"][index] - parsed_nmse["no_retention"]["D"][index]
        for index in range(len(seeds))
    ]
    recovery_differences: list[float] = []
    for retained_record in variants["retained"]:
        raw_recovery = retained_record.get("task_recovery")
        if not isinstance(raw_recovery, list):
            continue
        acquisitions: list[float] = []
        recurrences: list[float] = []
        for task_record in raw_recovery:
            if not isinstance(task_record, Mapping):
                continue
            if task_record.get("task") not in CRITICAL_TASKS:
                continue
            acquisition = _finite_number(task_record.get("acquisition_steps"))
            if acquisition is not None:
                acquisitions.append(acquisition)
            raw_steps = task_record.get("recurrence_steps")
            if isinstance(raw_steps, list):
                recurrences.extend(
                    step for raw_step in raw_steps if (step := _finite_number(raw_step)) is not None
                )
        if acquisitions and recurrences:
            recovery_differences.append(float(np.median(acquisitions) - np.median(recurrences)))
    retention_differences = [
        left - right
        for left, right in zip(
            retention["retained"],
            retention["no_retention"],
            strict=True,
        )
    ]
    return {
        "seed_count": len(seeds),
        "seeds": seeds,
        "retained": variant_payload("retained"),
        "no_retention": variant_payload("no_retention"),
        "retained_all_critical_rate_interval": wilson_score_interval(
            retained_successes,
            len(seeds),
        ),
        "retained_obsolete_active_bank_eviction_rate_interval": (
            wilson_score_interval(eviction_successes, len(seeds))
        ),
        "paired_effects": {
            "retention_rate_gain_over_no_retention": (
                paired_bootstrap_mean_interval(
                    retention_differences,
                    seed_offset=0,
                )
            ),
            "per_seed_maximum_critical_nmse_reduction": (
                paired_bootstrap_mean_interval(
                    critical_nmse_differences,
                    seed_offset=1,
                )
            ),
            "obsolete_nmse_increase_over_no_retention": (
                paired_bootstrap_mean_interval(
                    obsolete_differences,
                    seed_offset=2,
                )
            ),
            "retained_acquisition_minus_recurrence_steps": (
                paired_bootstrap_mean_interval(
                    recovery_differences,
                    seed_offset=3,
                )
            ),
        },
    }


def _validate_checks(
    value: object,
    *,
    expected_actuals: Mapping[str, float],
    errors: list[str],
) -> bool:
    if not isinstance(value, list):
        errors.append("scientific_payload.acceptance.checks must be an array")
        return False
    criteria = RecurringFeatureGateCriteria()
    expected_specs: dict[str, tuple[str, float]] = {
        "canonical_protocol": ("==", 1.0),
        "evidence_seed_schedule": ("==", 1.0),
        "minimum_seed_count": (">=", float(criteria.minimum_seeds)),
        "matched_unique_seeds": ("==", 1.0),
        "minimum_heldout_samples": (">=", float(criteria.minimum_heldout_samples)),
        "finite_complete_seed_evidence": ("==", 1.0),
        "exhaustive_counted_candidate_archive": ("==", 1.0),
        "memory_budget_consistent": ("==", 1.0),
        "retained_all_critical_rate": (
            ">=",
            criteria.minimum_retained_all_critical_rate,
        ),
        "obsolete_pair_active_bank_eviction_rate": (
            ">=",
            criteria.minimum_obsolete_eviction_rate,
        ),
        "maximum_critical_task_median_nmse": (
            "<=",
            criteria.maximum_median_critical_nmse,
        ),
        "minimum_obsolete_task_median_nmse": (
            ">=",
            criteria.minimum_median_obsolete_nmse,
        ),
        "retention_rate_gain_over_no_retention": (
            ">=",
            criteria.minimum_retention_rate_gain_over_baseline,
        ),
        "critical_nmse_gain_over_no_retention": (
            ">=",
            criteria.minimum_critical_nmse_gain_over_baseline,
        ),
        "recurrence_faster_than_acquisition": (">", 0.0),
        "upstream_gate_decision": ("==", 1.0),
    }
    by_name: dict[str, Mapping[str, object]] = {}
    encountered_names: list[str] = []
    all_passed = True
    for index, raw_check in enumerate(value):
        location = f"scientific_payload.acceptance.checks[{index}]"
        if not isinstance(raw_check, Mapping):
            errors.append(f"{location} must be an object")
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
            errors.append(f"{location} keys do not match the v1 schema")
        name = raw_check.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"{location}.name must be a non-empty string")
            all_passed = False
            continue
        encountered_names.append(name)
        if name in by_name:
            errors.append(f"duplicate acceptance check {name!r}")
            all_passed = False
            continue
        by_name[name] = raw_check
        actual = _finite_number(raw_check.get("actual"))
        threshold = _finite_number(raw_check.get("threshold"))
        comparator = raw_check.get("comparator")
        declared = raw_check.get("passed")
        detail = raw_check.get("detail")
        if not isinstance(detail, str) or not detail:
            errors.append(f"{location}.detail must be a non-empty string")
        expected_actual = expected_actuals.get(name)
        if expected_actual is not None and actual != expected_actual:
            errors.append(f"{location}.actual is inconsistent with raw evidence")
        expected_spec = expected_specs.get(name)
        if expected_spec is None:
            errors.append(f"{location}.name is not a canonical v1 check")
        elif comparator != expected_spec[0] or threshold != expected_spec[1]:
            errors.append(f"{location} does not use the frozen comparison")
        if (
            actual is None
            or threshold is None
            or not isinstance(comparator, str)
            or comparator not in {">=", "<=", ">", "=="}
        ):
            computed = False
            errors.append(f"{location} has an invalid numeric comparison")
        else:
            computed = _comparison_passes(actual, comparator, threshold)
        if not isinstance(declared, bool) or declared != computed:
            errors.append(f"{location}.passed is inconsistent with its comparison")
        all_passed = all_passed and computed
    missing = sorted(_REQUIRED_CHECKS - set(by_name))
    if missing:
        errors.append("acceptance checks missing required names: " + ", ".join(missing))
        all_passed = False
    if encountered_names != list(_REQUIRED_CHECK_ORDER):
        errors.append("acceptance checks are not in the canonical v1 order")
        all_passed = False
    return all_passed


def _expected_check_actuals(
    scientific: Mapping[str, object],
    aggregate: Mapping[str, object],
    seeds: list[int],
    variants: dict[str, list[dict[str, object]]],
) -> dict[str, float]:
    retained = aggregate.get("retained")
    baseline = aggregate.get("no_retention")
    retained_map = retained if isinstance(retained, Mapping) else {}
    baseline_map = baseline if isinstance(baseline, Mapping) else {}
    configuration = scientific.get("configuration")
    memory = scientific.get("memory_budget")
    canonical = _configuration_payload(RecurringFeatureProtocol())
    canonical_protocol = isinstance(configuration, Mapping) and configuration == canonical
    matched = (
        len(seeds) == len(set(seeds))
        and seeds == sorted(seeds)
        and all(len(values) == len(seeds) for values in variants.values())
    )
    archive_ok = True
    finite_ok = True
    active_ok = True
    expected_candidates = {(left, right) for left in range(6) for right in range(left + 1, 6)}
    protocol = RecurringFeatureProtocol()
    for variant_records in variants.values():
        for variant_record in variant_records:
            raw_candidates = variant_record.get("candidate_pairs")
            raw_active = variant_record.get("active_pairs")
            archive_ok = archive_ok and (
                isinstance(raw_candidates, list)
                and len(raw_candidates) == protocol.candidate_pair_budget
                and _pair_set(raw_candidates) == expected_candidates
                and variant_record.get("candidate_archive_is_exhaustive_all_pairs") is True
                and variant_record.get("obsolete_pair_remains_in_candidate_archive") is True
            )
            active_pairs = _pair_set(raw_active)
            active_ok = active_ok and (
                isinstance(raw_active, list)
                and len(raw_active) == protocol.active_pair_budget
                and active_pairs is not None
                and len(active_pairs) == protocol.active_pair_budget
                and active_pairs <= expected_candidates
            )
            raw_nmse = variant_record.get("final_heldout_nmse")
            raw_phases = variant_record.get("phase_evidence")
            phases_complete = isinstance(raw_phases, list) and len(raw_phases) == len(PHASE_TASKS)
            phases_finite = (
                all(
                    isinstance(raw_phase, Mapping)
                    and (phase_nmse := _finite_number(raw_phase.get("prequential_nmse")))
                    is not None
                    and phase_nmse >= 0.0
                    for raw_phase in raw_phases
                )
                if isinstance(raw_phases, list) and phases_complete
                else False
            )
            steps_seen = variant_record.get("steps_seen")
            finite_ok = (
                finite_ok
                and isinstance(raw_nmse, Mapping)
                and all(
                    (value := _finite_number(raw_nmse.get(task))) is not None and value >= 0.0
                    for task in TASK_NAMES
                )
                and phases_finite
                and not isinstance(steps_seen, bool)
                and isinstance(steps_seen, int)
                and steps_seen == protocol.total_steps
            )
    expected_memory = _memory_payload(
        FeatureMemoryBudget(
            active_pair_slots=RecurringFeatureProtocol().active_pair_budget,
            candidate_pair_slots=RecurringFeatureProtocol().candidate_pair_budget,
            output_heads=RecurringFeatureProtocol().output_heads,
        )
    )
    retained_rate = _finite_number(retained_map.get("all_critical_retention_rate"))
    eviction_rate = _finite_number(retained_map.get("obsolete_active_bank_eviction_rate"))
    retained_nmse = _finite_number(retained_map.get("maximum_critical_task_median_nmse"))
    obsolete_nmse = _finite_number(retained_map.get("median_obsolete_heldout_nmse"))
    baseline_rate = _finite_number(baseline_map.get("all_critical_retention_rate"))
    baseline_nmse = _finite_number(baseline_map.get("maximum_critical_task_median_nmse"))
    acquisition = _finite_number(retained_map.get("median_acquisition_recovery_steps"))
    recurrence = _finite_number(retained_map.get("median_recurrence_recovery_steps"))
    heldout_samples = _finite_number(
        configuration.get("heldout_samples") if isinstance(configuration, Mapping) else None
    )
    memory_ok = isinstance(memory, Mapping) and memory == expected_memory
    actuals = {
        "canonical_protocol": float(canonical_protocol),
        "evidence_seed_schedule": float(tuple(seeds) == EVIDENCE_SEEDS),
        "minimum_seed_count": float(len(seeds)),
        "matched_unique_seeds": float(matched),
        "minimum_heldout_samples": heldout_samples if heldout_samples is not None else -1.0,
        "finite_complete_seed_evidence": float(finite_ok),
        "exhaustive_counted_candidate_archive": float(archive_ok),
        "memory_budget_consistent": float(memory_ok),
        "retained_all_critical_rate": (retained_rate if retained_rate is not None else 0.0),
        "obsolete_pair_active_bank_eviction_rate": (
            eviction_rate if eviction_rate is not None else 0.0
        ),
        "maximum_critical_task_median_nmse": (
            retained_nmse if retained_nmse is not None else math.inf
        ),
        "minimum_obsolete_task_median_nmse": (
            obsolete_nmse if obsolete_nmse is not None else -math.inf
        ),
        "retention_rate_gain_over_no_retention": (
            retained_rate - baseline_rate
            if retained_rate is not None and baseline_rate is not None
            else -math.inf
        ),
        "critical_nmse_gain_over_no_retention": (
            baseline_nmse - retained_nmse
            if baseline_nmse is not None and retained_nmse is not None
            else -math.inf
        ),
        "recurrence_faster_than_acquisition": (
            acquisition - recurrence
            if acquisition is not None and recurrence is not None
            else -math.inf
        ),
    }
    criteria = RecurringFeatureGateCriteria()
    retained_name_ok = retained_map.get("name") == "retained"
    retained_decay = _finite_number(retained_map.get("utility_retention_decay"))
    baseline_name_ok = baseline_map.get("name") == "no_retention"
    baseline_decay_ok = baseline_map.get("utility_retention_decay") is None
    upstream_accepted = (
        canonical_protocol
        and matched
        and len(seeds) >= criteria.minimum_seeds
        and heldout_samples is not None
        and heldout_samples >= criteria.minimum_heldout_samples
        and finite_ok
        and archive_ok
        and active_ok
        and memory_ok
        and retained_name_ok
        and retained_decay == protocol.retained_utility_decay
        and baseline_name_ok
        and baseline_decay_ok
        and retained_rate is not None
        and retained_rate >= criteria.minimum_retained_all_critical_rate
        and eviction_rate is not None
        and eviction_rate >= criteria.minimum_obsolete_eviction_rate
        and retained_nmse is not None
        and retained_nmse <= criteria.maximum_median_critical_nmse
        and obsolete_nmse is not None
        and obsolete_nmse >= criteria.minimum_median_obsolete_nmse
        and baseline_rate is not None
        and retained_rate - baseline_rate >= criteria.minimum_retention_rate_gain_over_baseline
        and baseline_nmse is not None
        and baseline_nmse - retained_nmse >= criteria.minimum_critical_nmse_gain_over_baseline
        and acquisition is not None
        and recurrence is not None
        and recurrence < acquisition
    )
    actuals["upstream_gate_decision"] = float(upstream_accepted)
    return actuals


def _validate_operational(
    operational: Mapping[str, object],
    *,
    scientific: Mapping[str, object] | None,
    errors: list[str],
) -> None:
    if set(operational) != {
        "digest_exclusion_reason",
        "generated_at_utc",
        "gate_wall_seconds",
        "runtime",
        "git_worktree",
    }:
        errors.append("operational_metadata keys do not match the v1 schema")
    if operational.get("digest_exclusion_reason") != (
        "timestamp, host timing, runtime environment, devices, and worktree "
        "dirtiness are operational provenance rather than deterministic evidence"
    ):
        errors.append("operational_metadata digest exclusion reason is invalid")
    timestamp = operational.get("generated_at_utc")
    if not isinstance(timestamp, str):
        errors.append("operational_metadata.generated_at_utc must be a string")
    else:
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError:
            errors.append("operational_metadata.generated_at_utc is invalid ISO-8601")
        else:
            utc_offset = parsed.utcoffset()
            if parsed.tzinfo is None or utc_offset is None:
                errors.append("operational_metadata.generated_at_utc must include a timezone")
            elif utc_offset.total_seconds() != 0.0:
                errors.append("operational_metadata.generated_at_utc must be normalized to UTC")
    wall_seconds = _finite_number(operational.get("gate_wall_seconds"))
    if wall_seconds is None or wall_seconds < 0.0:
        errors.append("operational_metadata.gate_wall_seconds must be non-negative")
    runtime = operational.get("runtime")
    if not isinstance(runtime, Mapping):
        errors.append("operational_metadata.runtime must be an object")
    else:
        if set(runtime) != {"python", "platform", "packages", "jax"}:
            errors.append("operational_metadata.runtime keys do not match the v1 schema")
        python_runtime = runtime.get("python")
        if (
            not isinstance(python_runtime, Mapping)
            or set(python_runtime)
            != {
                "implementation",
                "version",
            }
            or not all(
                isinstance(python_runtime.get(key), str) and python_runtime.get(key)
                for key in ("implementation", "version")
            )
        ):
            errors.append("operational_metadata.runtime.python is invalid")
        platform_runtime = runtime.get("platform")
        if (
            not isinstance(platform_runtime, Mapping)
            or set(platform_runtime)
            != {
                "system",
                "release",
                "machine",
            }
            or not all(
                isinstance(platform_runtime.get(key), str)
                for key in ("system", "release", "machine")
            )
        ):
            errors.append("operational_metadata.runtime.platform is invalid")
        packages = runtime.get("packages")
        if (
            not isinstance(packages, Mapping)
            or set(packages)
            != {
                "alberta-framework",
                "jax",
                "jaxlib",
                "numpy",
            }
            or not all(
                isinstance(packages.get(key), str) and packages.get(key)
                for key in ("alberta-framework", "jax", "jaxlib", "numpy")
            )
        ):
            errors.append("operational_metadata.runtime.packages is invalid")
        jax_runtime = runtime.get("jax")
        if not isinstance(jax_runtime, Mapping) or set(jax_runtime) != {
            "default_backend",
            "devices",
        }:
            errors.append("operational_metadata.runtime.jax is invalid")
        else:
            backend = jax_runtime.get("default_backend")
            devices = jax_runtime.get("devices")
            if not isinstance(backend, str) or not backend:
                errors.append("operational_metadata.runtime.jax.default_backend is invalid")
            if (
                not isinstance(devices, list)
                or not devices
                or not all(
                    isinstance(device, Mapping)
                    and set(device) == {"platform", "device_kind"}
                    and isinstance(device.get("platform"), str)
                    and bool(device.get("platform"))
                    and isinstance(device.get("device_kind"), str)
                    and bool(device.get("device_kind"))
                    for device in devices
                )
            ):
                errors.append("operational_metadata.runtime.jax.devices is invalid")
    git_worktree = operational.get("git_worktree")
    if not isinstance(git_worktree, Mapping):
        errors.append("operational_metadata.git_worktree must be an object")
    else:
        if set(git_worktree) != {"head", "dirty"}:
            errors.append("operational_metadata.git_worktree keys do not match the v1 schema")
        recorded_head = git_worktree.get("head")
        if (
            not isinstance(recorded_head, str)
            or len(recorded_head) != 40
            or any(character not in "0123456789abcdef" for character in recorded_head)
        ):
            errors.append("operational_metadata.git_worktree.head must be a canonical SHA-1")
        provenance_head: object = None
        if scientific is not None:
            provenance = scientific.get("source_provenance")
            if isinstance(provenance, Mapping):
                provenance_head = provenance.get("git_head")
        if recorded_head != provenance_head:
            errors.append(
                "operational_metadata.git_worktree.head must match "
                "scientific_payload.source_provenance.git_head"
            )
        if not isinstance(git_worktree.get("dirty"), bool):
            errors.append("operational_metadata.git_worktree.dirty must be boolean")


def validate_recurring_feature_artifact(
    artifact: Mapping[str, object],
) -> ArtifactValidation:
    """Validate schema, provenance, raw evidence, aggregates, checks, and digest."""

    errors: list[str] = []
    if type(artifact) is not dict:
        errors.append("artifact must be a mapping with exact dict identities")
        return ArtifactValidation(valid=False, accepted=False, errors=tuple(errors))
    expected_top_level = {
        "schema_version",
        "scientific_payload",
        "scientific_digest",
        "operational_metadata",
    }
    if set(artifact) != expected_top_level:
        errors.append("artifact top-level keys do not match the v1 schema")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    scientific = _mapping(artifact, "scientific_payload", "artifact", errors)
    digest = _mapping(artifact, "scientific_digest", "artifact", errors)
    operational = _mapping(artifact, "operational_metadata", "artifact", errors)

    scientific_passed = False
    if scientific is not None:
        if set(scientific) != {
            "protocol",
            "configuration",
            "criteria",
            "bootstrap",
            "memory_budget",
            "seed_summaries",
            "aggregate",
            "acceptance",
            "source_provenance",
        }:
            errors.append("scientific_payload keys do not match the v1 schema")
        protocol = _mapping(scientific, "protocol", "scientific_payload", errors)
        configuration = _mapping(
            scientific,
            "configuration",
            "scientific_payload",
            errors,
        )
        criteria = _mapping(scientific, "criteria", "scientific_payload", errors)
        bootstrap = _mapping(scientific, "bootstrap", "scientific_payload", errors)
        memory = _mapping(scientific, "memory_budget", "scientific_payload", errors)
        aggregate = _mapping(scientific, "aggregate", "scientific_payload", errors)
        acceptance = _mapping(scientific, "acceptance", "scientific_payload", errors)
        provenance = _mapping(
            scientific,
            "source_provenance",
            "scientific_payload",
            errors,
        )
        if protocol is not None:
            if protocol != _protocol_payload():
                errors.append("scientific_payload protocol is not canonical v1")
        if configuration is not None and configuration != _configuration_payload(
            RecurringFeatureProtocol()
        ):
            errors.append("scientific_payload configuration is not canonical v1")
        if criteria is not None and criteria != _criteria_payload(RecurringFeatureGateCriteria()):
            errors.append("scientific_payload criteria are not canonical v1")
        if bootstrap is not None and bootstrap != {
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "seed": BOOTSTRAP_SEED,
            "role": "descriptive paired uncertainty; not threshold calibration",
        }:
            errors.append("scientific_payload bootstrap policy is not canonical v1")
        if memory is not None and memory != _memory_payload(
            FeatureMemoryBudget(
                active_pair_slots=RecurringFeatureProtocol().active_pair_budget,
                candidate_pair_slots=RecurringFeatureProtocol().candidate_pair_budget,
                output_heads=RecurringFeatureProtocol().output_heads,
            )
        ):
            errors.append("scientific_payload memory budget is not canonical v1")
        if provenance is not None:
            _validate_source_provenance(provenance, errors)

        seeds, variants = _extract_seed_metrics(
            scientific.get("seed_summaries"),
            errors,
        )
        recomputed = _recomputed_aggregate(seeds, variants, errors)
        if aggregate is not None and recomputed is not None and aggregate != recomputed:
            errors.append("scientific_payload.aggregate is inconsistent with seed summaries")
        expected_actuals = (
            _expected_check_actuals(scientific, aggregate, seeds, variants)
            if aggregate is not None
            else {}
        )
        if acceptance is not None:
            if set(acceptance) != {
                "passed",
                "checks",
                "upstream_summary",
                "upstream_failures",
            }:
                errors.append("scientific_payload.acceptance keys do not match the v1 schema")
            if not isinstance(acceptance.get("upstream_summary"), str):
                errors.append("scientific_payload.acceptance.upstream_summary must be a string")
            upstream_failures = acceptance.get("upstream_failures")
            if not isinstance(upstream_failures, list) or not all(
                isinstance(failure, str) for failure in upstream_failures
            ):
                errors.append(
                    "scientific_payload.acceptance.upstream_failures must be a string array"
                )
            declared_passed = acceptance.get("passed")
            computed_passed = _validate_checks(
                acceptance.get("checks"),
                expected_actuals=expected_actuals,
                errors=errors,
            )
            if not isinstance(declared_passed, bool) or (declared_passed != computed_passed):
                errors.append("scientific_payload.acceptance.passed is inconsistent with checks")
            scientific_passed = computed_passed

    if digest is not None:
        if set(digest) != {
            "algorithm",
            "scope",
            "canonicalization",
            "sha256",
        }:
            errors.append("scientific_digest keys do not match the v1 schema")
        if digest.get("algorithm") != DIGEST_ALGORITHM:
            errors.append(f"scientific_digest.algorithm must be {DIGEST_ALGORITHM!r}")
        if digest.get("scope") != DIGEST_SCOPE:
            errors.append(f"scientific_digest.scope must be {DIGEST_SCOPE!r}")
        if digest.get("canonicalization") != CANONICALIZATION:
            errors.append("scientific_digest canonicalization is invalid")
        recorded = digest.get("sha256")
        if (
            not isinstance(recorded, str)
            or len(recorded) != 64
            or any(character not in "0123456789abcdef" for character in recorded)
        ):
            errors.append("scientific_digest.sha256 must be lowercase hexadecimal SHA-256")
        elif scientific is not None:
            try:
                computed_digest = scientific_payload_sha256(scientific)
            except (TypeError, ValueError) as error:
                errors.append(f"scientific_payload is not canonical JSON: {error}")
            else:
                if recorded != computed_digest:
                    errors.append("scientific_digest.sha256 does not match scientific payload")

    if operational is not None:
        _validate_operational(
            operational,
            scientific=scientific,
            errors=errors,
        )
    valid = not errors
    return ArtifactValidation(
        valid=valid,
        accepted=valid and scientific_passed,
        errors=tuple(errors),
    )


def write_recurring_feature_artifact(
    path: Path,
    result: RecurringFeatureGateResult,
    *,
    gate_wall_seconds: float,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Build, validate, and atomically create one artifact without overwriting."""

    artifact = build_recurring_feature_artifact(
        result,
        gate_wall_seconds=gate_wall_seconds,
        generated_at=generated_at,
    )
    validation = validate_recurring_feature_artifact(artifact)
    if not validation.valid:
        raise ValueError(
            "refusing to write invalid recurring-feature artifact: " + "; ".join(validation.errors)
        )
    expanded = path.expanduser()
    destination = expanded.resolve()
    if os.path.lexists(expanded) or os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing output path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(artifact_json(artifact))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite concurrently created output path: {destination}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return artifact


__all__ = [
    "ArtifactValidation",
    "BOOTSTRAP_CONFIDENCE_LEVEL",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CANONICALIZATION",
    "DIGEST_ALGORITHM",
    "DIGEST_SCOPE",
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "artifact_json",
    "build_recurring_feature_artifact",
    "canonical_scientific_bytes",
    "load_recurring_feature_artifact",
    "paired_bootstrap_mean_interval",
    "scientific_payload_sha256",
    "validate_recurring_feature_artifact",
    "wilson_score_interval",
    "write_recurring_feature_artifact",
]
