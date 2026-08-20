"""Versioned, fail-closed evidence for the frozen FTL decision probe.

The scientific digest covers the exact deterministic protocol, its frozen
configuration and thresholds, primitive per-seed metrics, reconstructed
5,000-resample bootstrap summaries, acceptance checks, and pinned source
provenance.  Host timing, runtime details, generation time, and worktree
dirtiness are retained outside the digest.

This artifact supports a deliberately narrow result.  The environment is
deterministic, fully observed, and one-dimensional; reward is known; planning
chooses among five hand-designed horizon-six open-loop menus.  The raw online
ridge baseline is not compute-, feature-, or memory-capacity matched.  The
probe is inspired by WorldModelGym, not an official WorldModelGym protocol.
It is not evidence for closed-loop MPC, reward learning, a lifetime-retention
theorem, general control, or completion of the Alberta Plan.

SHA-256 and source hashes are reproducibility and integrity identifiers.  They
are not signatures and do not establish who produced an artifact.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import jax
import numpy as np
from numpy.typing import NDArray

from alberta_framework.evaluation.ftl_decision_fidelity import (
    CONDITION_NAMES,
    DEVELOPMENT_SEEDS,
    EVIDENCE_SEEDS,
    DecisionFidelityConfig,
    DecisionFidelityReport,
)

SCHEMA_VERSION = "alberta.ftl_decision_fidelity_evidence.v1"
PROTOCOL_VERSION = "sparse-ftl-known-reward-open-loop-aba.v1"
DIGEST_ALGORITHM = "sha256"
DIGEST_SCOPE = "$.scientific_payload"
CANONICALIZATION = "utf8-json-sort-keys-compact-no-nan"

# These values are part of the already-inspected protocol.  In particular,
# do not silently replace the 5,000-resample intervals with a post-look 10,000
# resample analysis.
FROZEN_BOOTSTRAP_RESAMPLES = 5_000
FROZEN_CONFIDENCE_LEVEL = 0.95
FROZEN_BOOTSTRAP_SEED = 2_026_073_001

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATHS = (
    Path("alberta_framework/core/ftl_world_model.py"),
    Path("alberta_framework/evaluation/ftl_decision_fidelity.py"),
    Path("alberta_framework/evaluation/ftl_decision_artifact.py"),
    Path("alberta_framework/evaluation/ftl_decision_cli.py"),
)
_EXPECTED_CONDITION_NAMES = (
    "sparse_untrained",
    "sparse_after_a1",
    "sparse_after_b",
    "sparse_after_a2",
    "linear_after_a1",
    "linear_after_b",
    "linear_after_a2",
)
_EXPECTED_DEVELOPMENT_SEEDS = tuple(range(30))
_EXPECTED_EVIDENCE_SEEDS = tuple(range(30, 60))
_FROZEN_CONFIGURATION: dict[str, object] = {
    "phase_steps": 180,
    "horizon": 6,
    "probes_per_domain": 12,
    "menu_amplitudes": [-0.85, -0.45, 0.0, 0.45, 0.85],
    "projection_dim": 12,
    "bins": 7,
    "ridge": 0.01,
    "prediction_clip": 3.0,
    "state_bound": 1.75,
    "action_cost": 0.025,
    "bootstrap_resamples": FROZEN_BOOTSTRAP_RESAMPLES,
    "confidence_level": FROZEN_CONFIDENCE_LEVEL,
    "bootstrap_seed": FROZEN_BOOTSTRAP_SEED,
}
_METRIC_FIELDS = (
    "normalized_regret",
    "domain_a_normalized_regret",
    "domain_b_normalized_regret",
    "oracle_pick_rate",
    "reward_mae",
    "return_mae",
    "normalized_return_mae",
)
_COMPARISON_SPECS = (
    (
        "sparse_after_b_vs_untrained_regret_reduction",
        "sparse_untrained regret - sparse_after_b regret; positive favors learning",
        "sparse_untrained",
        "sparse_after_b",
        "normalized_regret",
    ),
    (
        "sparse_after_b_vs_linear_regret_reduction",
        "linear_after_b regret - sparse_after_b regret; positive favors sparse",
        "linear_after_b",
        "sparse_after_b",
        "normalized_regret",
    ),
    (
        "sparse_domain_a_regret_change_after_b",
        "sparse_after_b A regret - sparse_after_a1 A regret; positive is interference",
        "sparse_after_b",
        "sparse_after_a1",
        "domain_a_normalized_regret",
    ),
    (
        "sparse_domain_a_recovery_after_a2",
        "sparse_after_b A regret - sparse_after_a2 A regret; positive is recovery",
        "sparse_after_b",
        "sparse_after_a2",
        "domain_a_normalized_regret",
    ),
)
_ACCEPTANCE_CHECK_ORDER = (
    "untrained_normalized_regret",
    "sparse_after_b_normalized_regret",
    "sparse_after_b_vs_untrained_regret_reduction",
    "sparse_after_b_vs_linear_regret_reduction",
    "sparse_after_b_oracle_pick_rate",
    "sparse_after_b_reward_mae_ratio_to_linear",
    "sparse_after_b_normalized_return_mae_ratio_to_linear",
    "sparse_after_a1_domain_a_regret",
    "sparse_after_a1_domain_b_regret",
    "sparse_after_b_domain_b_regret",
    "sparse_domain_a_interference",
    "sparse_domain_a_recovery_after_a2",
    "linear_after_a1_domain_a_regret",
    "linear_improvement_after_b",
    "linear_non_degradation_after_a2",
    "linear_after_b_normalized_regret",
)


@dataclass(frozen=True)
class ArtifactValidation:
    """Integrity and scientific-acceptance result for a parsed artifact."""

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


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def canonical_scientific_bytes(payload: Mapping[str, object]) -> bytes:
    """Return the documented canonical representation for hashing."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def scientific_payload_sha256(payload: Mapping[str, object]) -> str:
    """Hash canonical deterministic scientific content with SHA-256."""

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
        path = _REPO_ROOT / relative_path
        try:
            source = path.read_bytes()
        except OSError as error:
            raise RuntimeError(f"unable to hash source {relative_path}") from error
        hashes[relative_path.as_posix()] = hashlib.sha256(source).hexdigest()
    return {
        "repository_subtree": "research/alberta",
        "git_head": _git_head(),
        "source_sha256": hashes,
        "interpretation": (
            "reproducibility and integrity identifiers only; hashes are not "
            "signatures and do not establish artifact authenticity"
        ),
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


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
            "alberta-framework": _package_version("alberta-framework"),
            "jax": jax.__version__,
            "jaxlib": _package_version("jaxlib"),
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
            "a sparse fixed-shape online transition model preserves low "
            "known-reward open-loop menu regret across a deterministic A-B-A "
            "visitation stream and outperforms the included untrained and raw "
            "online-ridge decision baselines"
        ),
        "environment": ("deterministic, fully observed, one-dimensional nonlinear dynamics"),
        "training_schedule": ["negative-domain A1", "positive-domain B", "negative-domain A2"],
        "interaction": (
            "one uninterrupted predict-before-update transition stream with no "
            "replay or phase label supplied to either learner"
        ),
        "decision_probe": (
            "known reward applied to immutable model rollouts over five "
            "hand-designed horizon-six open-loop action menus sharing their "
            "first action"
        ),
        "statistical_protocol": (
            "condition means and paired differences use the frozen 5,000-"
            "resample percentile bootstrap and original deterministic seed offsets"
        ),
        "baseline_scope": (
            "the raw online ridge baseline sees the same transitions and ridge "
            "coefficient but is not compute-, feature-count-, or memory-capacity matched"
        ),
        "external_protocol_relationship": (
            "WorldModelGym-inspired decision fidelity, not an official WorldModelGym evaluation"
        ),
        "seed_roles": {
            "development_and_threshold_calibration": list(_EXPECTED_DEVELOPMENT_SEEDS),
            "promoted_held_out_evidence": list(_EXPECTED_EVIDENCE_SEEDS),
            "rule": (
                "configuration and thresholds were inspected on seeds 0-29; "
                "promoted evidence is the disjoint frozen schedule 30-59 with "
                "no retuning after its single held-out look"
            ),
        },
        "excluded_claims": [
            "closed-loop acting or model-predictive control",
            "reward-model learning",
            "stochastic or partially observed control",
            "visual world-model fidelity",
            "compute- or capacity-matched baseline superiority",
            "exact global follow-the-leader guarantees",
            "indefinite-lifetime retention or scaling theorem",
            "general catastrophic-forgetting resistance",
            "completion of the Alberta Plan",
        ],
    }


def _configuration_payload(config: DecisionFidelityConfig) -> dict[str, object]:
    return {
        "phase_steps": config.phase_steps,
        "horizon": config.horizon,
        "probes_per_domain": config.probes_per_domain,
        "menu_amplitudes": list(config.menu_amplitudes),
        "projection_dim": config.projection_dim,
        "bins": config.bins,
        "ridge": config.ridge,
        "prediction_clip": config.prediction_clip,
        "state_bound": config.state_bound,
        "action_cost": config.action_cost,
        "bootstrap_resamples": config.bootstrap_resamples,
        "confidence_level": config.confidence_level,
        "bootstrap_seed": config.bootstrap_seed,
    }


def _canonical_config() -> DecisionFidelityConfig:
    """Resolve v1 only if upstream constants still match the frozen protocol."""

    if CONDITION_NAMES != _EXPECTED_CONDITION_NAMES:
        raise RuntimeError("upstream condition names drifted from artifact v1")
    if DEVELOPMENT_SEEDS != _EXPECTED_DEVELOPMENT_SEEDS:
        raise RuntimeError("upstream development seeds drifted from artifact v1")
    if EVIDENCE_SEEDS != _EXPECTED_EVIDENCE_SEEDS:
        raise RuntimeError("upstream evidence seeds drifted from artifact v1")
    config = DecisionFidelityConfig()
    if _configuration_payload(config) != _FROZEN_CONFIGURATION:
        raise RuntimeError("upstream default configuration drifted from artifact v1")
    return config


def _threshold_payload() -> dict[str, float]:
    """Encode the exact comparisons already asserted by the frozen protocol."""

    return {
        "minimum_untrained_normalized_regret_lower": 0.25,
        "maximum_sparse_after_b_normalized_regret_upper": 0.01,
        "minimum_sparse_vs_untrained_regret_reduction_lower": 0.25,
        "minimum_sparse_vs_linear_regret_reduction_lower": 0.03,
        "minimum_sparse_after_b_oracle_pick_rate": 0.85,
        "maximum_sparse_reward_mae_ratio_to_linear": 0.20,
        "maximum_sparse_normalized_return_mae_ratio_to_linear": 0.15,
        "maximum_sparse_after_a1_domain_a_regret": 1.0e-3,
        "minimum_sparse_after_a1_domain_b_regret": 0.10,
        "maximum_sparse_after_b_domain_b_regret_upper": 0.01,
        "maximum_sparse_domain_a_interference_upper": 0.01,
        "minimum_sparse_domain_a_recovery": 0.0,
        "maximum_linear_after_a1_domain_a_regret": 0.01,
        "minimum_linear_improvement_after_b": 0.0,
        "minimum_linear_non_degradation_after_a2": 0.0,
        "maximum_linear_after_b_normalized_regret": 0.10,
    }


def _bootstrap_payload(config: DecisionFidelityConfig) -> dict[str, object]:
    return {
        "resamples": config.bootstrap_resamples,
        "confidence_level": config.confidence_level,
        "seed": config.bootstrap_seed,
        "condition_seed_offsets": {
            condition: {
                field: condition_index * len(_METRIC_FIELDS) + field_index
                for field_index, field in enumerate(_METRIC_FIELDS)
            }
            for condition_index, condition in enumerate(_EXPECTED_CONDITION_NAMES)
        },
        "paired_comparison_seed_offsets": {
            spec[0]: 100 + index for index, spec in enumerate(_COMPARISON_SPECS)
        },
        "method": "percentile bootstrap of the seed-level mean",
        "pairing_unit": "seed",
        "freeze_note": (
            "the original 5,000-resample analysis is retained; no post-held-out "
            "10,000-resample replacement is added"
        ),
    }


def _seed_payloads(report: DecisionFidelityReport) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for seed_result in report.seed_results:
        conditions: dict[str, dict[str, float | None]] = {}
        for metrics in seed_result.metrics:
            if metrics.condition in conditions:
                raise ValueError(
                    f"duplicate condition {metrics.condition!r} for seed {seed_result.seed}"
                )
            conditions[metrics.condition] = {
                field: _finite_float(getattr(metrics, field)) for field in _METRIC_FIELDS
            }
        payloads.append({"seed": seed_result.seed, "conditions": conditions})
    return payloads


def _bootstrap_mean(
    values: NDArray[np.float64],
    *,
    config: DecisionFidelityConfig,
    seed_offset: int,
    paired: bool,
) -> dict[str, object]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be finite")
    generator = np.random.default_rng(config.bootstrap_seed + seed_offset)
    indices = generator.integers(
        0,
        values.size,
        size=(config.bootstrap_resamples, values.size),
    )
    resampled_means = values[indices].mean(axis=1)
    tail = (1.0 - config.confidence_level) / 2.0
    lower, upper = np.quantile(resampled_means, (tail, 1.0 - tail))
    return {
        "estimate": float(values.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": config.confidence_level,
        "resamples": config.bootstrap_resamples,
        "sample_size": int(values.size),
        "method": ("paired-percentile-bootstrap" if paired else "percentile-bootstrap"),
        "pairing_unit": "seed",
    }


def _aggregate_payload(
    vectors: Mapping[str, Mapping[str, NDArray[np.float64]]],
    config: DecisionFidelityConfig,
) -> dict[str, object]:
    conditions: dict[str, object] = {}
    seed_offset = 0
    for condition in _EXPECTED_CONDITION_NAMES:
        metrics: dict[str, object] = {}
        for field in _METRIC_FIELDS:
            metrics[field] = _bootstrap_mean(
                vectors[condition][field],
                config=config,
                seed_offset=seed_offset,
                paired=False,
            )
            seed_offset += 1
        conditions[condition] = metrics
    return {
        "seed_count": len(_EXPECTED_EVIDENCE_SEEDS),
        "seeds": list(_EXPECTED_EVIDENCE_SEEDS),
        "conditions": conditions,
    }


def _comparison_payload(
    vectors: Mapping[str, Mapping[str, NDArray[np.float64]]],
    config: DecisionFidelityConfig,
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    for index, (name, definition, left, right, field) in enumerate(_COMPARISON_SPECS):
        differences = vectors[left][field] - vectors[right][field]
        comparisons[name] = {
            "definition": definition,
            "interval": _bootstrap_mean(
                differences,
                config=config,
                seed_offset=100 + index,
                paired=True,
            ),
        }
    return comparisons


def _interval_metric(
    aggregate: Mapping[str, object],
    condition: str,
    field: str,
    statistic: str,
) -> float | None:
    conditions = aggregate.get("conditions")
    if not isinstance(conditions, Mapping):
        return None
    condition_payload = conditions.get(condition)
    if not isinstance(condition_payload, Mapping):
        return None
    interval = condition_payload.get(field)
    if not isinstance(interval, Mapping):
        return None
    return _finite_number(interval.get(statistic))


def _comparison_metric(
    comparisons: Mapping[str, object],
    name: str,
    statistic: str,
) -> float | None:
    comparison = comparisons.get(name)
    if not isinstance(comparison, Mapping):
        return None
    interval = comparison.get("interval")
    if not isinstance(interval, Mapping):
        return None
    return _finite_number(interval.get(statistic))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if (
        numerator is None
        or denominator is None
        or denominator <= 0.0
        or not math.isfinite(numerator)
        or not math.isfinite(denominator)
    ):
        return None
    return numerator / denominator


def _comparison_passes(
    actual: float | None,
    comparator: str,
    threshold: float,
) -> bool:
    if actual is None or not math.isfinite(actual):
        return False
    if comparator == ">":
        return actual > threshold
    if comparator == "<":
        return actual < threshold
    if comparator == ">=":
        return actual >= threshold
    if comparator == "<=":
        return actual <= threshold
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
        "actual": _finite_float(actual) if actual is not None else None,
        "comparator": comparator,
        "threshold": threshold,
        "detail": detail,
    }


def _acceptance_payload(
    aggregate: Mapping[str, object],
    comparisons: Mapping[str, object],
) -> dict[str, object]:
    thresholds = _threshold_payload()
    learned_reward = _interval_metric(aggregate, "sparse_after_b", "reward_mae", "estimate")
    linear_reward = _interval_metric(aggregate, "linear_after_b", "reward_mae", "estimate")
    learned_normalized_return = _interval_metric(
        aggregate, "sparse_after_b", "normalized_return_mae", "estimate"
    )
    linear_normalized_return = _interval_metric(
        aggregate, "linear_after_b", "normalized_return_mae", "estimate"
    )
    linear_a1 = _interval_metric(aggregate, "linear_after_a1", "normalized_regret", "estimate")
    linear_b = _interval_metric(aggregate, "linear_after_b", "normalized_regret", "estimate")
    linear_a2 = _interval_metric(aggregate, "linear_after_a2", "normalized_regret", "estimate")
    linear_improvement = (
        linear_a1 - linear_b if linear_a1 is not None and linear_b is not None else None
    )
    linear_non_degradation = (
        linear_b - linear_a2 if linear_b is not None and linear_a2 is not None else None
    )
    checks = [
        _check(
            "untrained_normalized_regret",
            _interval_metric(
                aggregate,
                "sparse_untrained",
                "normalized_regret",
                "lower",
            ),
            ">",
            float(thresholds["minimum_untrained_normalized_regret_lower"]),
            "Lower confidence bound; the untrained decision problem must be nontrivial.",
        ),
        _check(
            "sparse_after_b_normalized_regret",
            _interval_metric(
                aggregate,
                "sparse_after_b",
                "normalized_regret",
                "upper",
            ),
            "<",
            float(thresholds["maximum_sparse_after_b_normalized_regret_upper"]),
            "Upper confidence bound after learning both visited domains.",
        ),
        _check(
            "sparse_after_b_vs_untrained_regret_reduction",
            _comparison_metric(
                comparisons,
                "sparse_after_b_vs_untrained_regret_reduction",
                "lower",
            ),
            ">",
            float(thresholds["minimum_sparse_vs_untrained_regret_reduction_lower"]),
            "Lower paired confidence bound; positive favors online learning.",
        ),
        _check(
            "sparse_after_b_vs_linear_regret_reduction",
            _comparison_metric(
                comparisons,
                "sparse_after_b_vs_linear_regret_reduction",
                "lower",
            ),
            ">",
            float(thresholds["minimum_sparse_vs_linear_regret_reduction_lower"]),
            (
                "Lower paired confidence bound; the included raw ridge baseline "
                "is not compute or capacity matched."
            ),
        ),
        _check(
            "sparse_after_b_oracle_pick_rate",
            _interval_metric(
                aggregate,
                "sparse_after_b",
                "oracle_pick_rate",
                "estimate",
            ),
            ">",
            float(thresholds["minimum_sparse_after_b_oracle_pick_rate"]),
            "Seed-mean fraction of menu choices matching the true best return.",
        ),
        _check(
            "sparse_after_b_reward_mae_ratio_to_linear",
            _safe_ratio(learned_reward, linear_reward),
            "<",
            float(thresholds["maximum_sparse_reward_mae_ratio_to_linear"]),
            "Point-estimate known-reward MAE ratio to the raw ridge baseline.",
        ),
        _check(
            "sparse_after_b_normalized_return_mae_ratio_to_linear",
            _safe_ratio(learned_normalized_return, linear_normalized_return),
            "<",
            float(thresholds["maximum_sparse_normalized_return_mae_ratio_to_linear"]),
            "Point-estimate normalized-return MAE ratio to raw ridge.",
        ),
        _check(
            "sparse_after_a1_domain_a_regret",
            _interval_metric(
                aggregate,
                "sparse_after_a1",
                "domain_a_normalized_regret",
                "estimate",
            ),
            "<",
            float(thresholds["maximum_sparse_after_a1_domain_a_regret"]),
            "Point regret in the initially visited A domain.",
        ),
        _check(
            "sparse_after_a1_domain_b_regret",
            _interval_metric(
                aggregate,
                "sparse_after_a1",
                "domain_b_normalized_regret",
                "estimate",
            ),
            ">",
            float(thresholds["minimum_sparse_after_a1_domain_b_regret"]),
            "Unvisited B must remain a nontrivial decision domain after A1.",
        ),
        _check(
            "sparse_after_b_domain_b_regret",
            _interval_metric(
                aggregate,
                "sparse_after_b",
                "domain_b_normalized_regret",
                "upper",
            ),
            "<",
            float(thresholds["maximum_sparse_after_b_domain_b_regret_upper"]),
            "Upper confidence bound for newly visited B.",
        ),
        _check(
            "sparse_domain_a_interference",
            _comparison_metric(
                comparisons,
                "sparse_domain_a_regret_change_after_b",
                "upper",
            ),
            "<",
            float(thresholds["maximum_sparse_domain_a_interference_upper"]),
            "Upper paired bound on A regret increase during B.",
        ),
        _check(
            "sparse_domain_a_recovery_after_a2",
            _comparison_metric(
                comparisons,
                "sparse_domain_a_recovery_after_a2",
                "estimate",
            ),
            ">=",
            float(thresholds["minimum_sparse_domain_a_recovery"]),
            "Paired point estimate; nonnegative means A2 does not worsen A.",
        ),
        _check(
            "linear_after_a1_domain_a_regret",
            _interval_metric(
                aggregate,
                "linear_after_a1",
                "domain_a_normalized_regret",
                "estimate",
            ),
            "<",
            float(thresholds["maximum_linear_after_a1_domain_a_regret"]),
            "The raw ridge control must be a nontrivial learned baseline.",
        ),
        _check(
            "linear_improvement_after_b",
            linear_improvement,
            ">",
            float(thresholds["minimum_linear_improvement_after_b"]),
            "Raw ridge overall regret after A1 minus after B.",
        ),
        _check(
            "linear_non_degradation_after_a2",
            linear_non_degradation,
            ">=",
            float(thresholds["minimum_linear_non_degradation_after_a2"]),
            "Raw ridge overall regret after B minus after A2.",
        ),
        _check(
            "linear_after_b_normalized_regret",
            linear_b,
            "<",
            float(thresholds["maximum_linear_after_b_normalized_regret"]),
            "Raw ridge overall regret after B must remain below the loose ceiling.",
        ),
    ]
    if tuple(check["name"] for check in checks) != _ACCEPTANCE_CHECK_ORDER:
        raise RuntimeError("acceptance check order drifted from the v1 schema")
    return {
        "passed": all(check["passed"] is True for check in checks),
        "checks": checks,
    }


def _scientific_payload(report: DecisionFidelityReport) -> dict[str, object]:
    canonical_config = _canonical_config()
    if report.config != canonical_config:
        raise ValueError("promoted artifact requires the canonical frozen configuration")
    if report.seeds != _EXPECTED_EVIDENCE_SEEDS:
        raise ValueError("promoted artifact requires held-out seeds 30-59 exactly")
    if tuple(result.seed for result in report.seed_results) != _EXPECTED_EVIDENCE_SEEDS:
        raise ValueError("primitive seed results must be held-out seeds 30-59 in order")

    seed_payloads = _seed_payloads(report)
    primitive_errors: list[str] = []
    vectors = _validate_and_extract_seed_vectors(seed_payloads, primitive_errors)
    if primitive_errors or vectors is None:
        raise ValueError("invalid primitive decision evidence: " + "; ".join(primitive_errors))
    aggregate = _aggregate_payload(vectors, canonical_config)
    comparisons = _comparison_payload(vectors, canonical_config)
    return {
        "protocol": _protocol_payload(),
        "configuration": _configuration_payload(canonical_config),
        "thresholds": _threshold_payload(),
        "bootstrap": _bootstrap_payload(canonical_config),
        "seed_summaries": seed_payloads,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "acceptance": _acceptance_payload(aggregate, comparisons),
        "source_provenance": _source_provenance(),
    }


def _operational_payload(
    *,
    evaluation_wall_seconds: float,
    generated_at: datetime | None,
) -> dict[str, object]:
    if not math.isfinite(evaluation_wall_seconds) or evaluation_wall_seconds < 0.0:
        raise ValueError("evaluation_wall_seconds must be finite and non-negative")
    timestamp = datetime.now(UTC) if generated_at is None else generated_at
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    return {
        "digest_exclusion_reason": (
            "timestamp, host timing, runtime environment, devices, and worktree "
            "dirtiness are operational provenance rather than deterministic evidence"
        ),
        "generated_at_utc": timestamp.astimezone(UTC).isoformat(),
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "runtime": _runtime_provenance(),
        "git_worktree": {
            "head": _git_head(),
            "dirty": _git_dirty(),
        },
    }


def build_ftl_decision_artifact(
    report: DecisionFidelityReport,
    *,
    evaluation_wall_seconds: float,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Build an artifact from primitive seed metrics, not cached aggregates."""

    scientific_payload = _scientific_payload(report)
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
            evaluation_wall_seconds=evaluation_wall_seconds,
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
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_ftl_decision_artifact(path: Path) -> dict[str, object]:
    """Load strict JSON, rejecting NaN, infinities, duplicates, and non-objects."""

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
    if not isinstance(value, Mapping):
        errors.append(f"{location}.{key} must be an object")
        return None
    return value


def _numbers_match(first: object, second: object) -> bool:
    left = _finite_number(first)
    right = _finite_number(second)
    return left is not None and right is not None and left == right


def _compare_structure(
    actual: object,
    expected: object,
    location: str,
    errors: list[str],
) -> None:
    """Compare exact schemas and finite values after a JSON float round trip."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            errors.append(f"{location} must be an object")
            return
        if set(actual) != set(expected):
            errors.append(f"{location} keys do not match the v1 schema")
        for key in sorted(set(actual) & set(expected)):
            _compare_structure(
                actual[key],
                expected[key],
                f"{location}.{key}",
                errors,
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            errors.append(f"{location} must be an array")
            return
        if len(actual) != len(expected):
            errors.append(f"{location} length does not match the v1 schema")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=False)):
            _compare_structure(
                actual_item,
                expected_item,
                f"{location}[{index}]",
                errors,
            )
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            errors.append(f"{location} does not match the v1 value")
        return
    if type(expected) is bool or type(expected) is not int:
        errors.append(f"{location} must be an integer")
        return
    if type(actual) is bool or type(actual) is not int or actual != expected:
        errors.append(f"{location} does not match the v1 integer value")
        return
    if isinstance(expected, float):
        if not _numbers_match(actual, expected):
            errors.append(f"{location} does not match reconstructed evidence")
        return
    if actual != expected:
        errors.append(f"{location} does not match the v1 value")


def _validate_source_provenance(
    value: object,
    errors: list[str],
) -> None:
    """Bind exact source bytes without self-invalidating after artifact commit.

    Evidence is normally generated immediately before it is committed.  That
    commit advances ``HEAD`` even if every pinned source byte is unchanged.  The
    hashes are therefore the scientific binding, while the generation commit is
    internally consistent contextual provenance.
    """

    location = "scientific_payload.source_provenance"
    if not isinstance(value, Mapping):
        errors.append(f"{location} must be an object")
        return
    try:
        expected = _source_provenance()
    except RuntimeError as error:
        errors.append(str(error))
        return
    if set(value) != set(expected):
        errors.append(f"{location} keys do not match the v1 schema")
    if value.get("repository_subtree") != expected["repository_subtree"]:
        errors.append(f"{location}.repository_subtree is invalid")
    generation_head = value.get("git_head")
    if (
        not isinstance(generation_head, str)
        or len(generation_head) != 40
        or any(character not in "0123456789abcdef" for character in generation_head)
    ):
        errors.append(f"{location}.git_head must be a canonical SHA-1")
    if value.get("source_sha256") != expected["source_sha256"]:
        errors.append(f"{location} does not match the current pinned source hashes")
    if value.get("interpretation") != expected["interpretation"]:
        errors.append(f"{location}.interpretation is invalid")


def _validate_and_extract_seed_vectors(
    summaries: object,
    errors: list[str],
) -> dict[str, dict[str, NDArray[np.float64]]] | None:
    if not isinstance(summaries, list):
        errors.append("scientific_payload.seed_summaries must be an array")
        return None
    if len(summaries) != len(_EXPECTED_EVIDENCE_SEEDS):
        errors.append("scientific_payload.seed_summaries must contain exactly 30 seeds")

    observed_seeds: list[int] = []
    raw_vectors: dict[str, dict[str, list[float]]] = {
        condition: {field: [] for field in _METRIC_FIELDS}
        for condition in _EXPECTED_CONDITION_NAMES
    }
    for index, summary in enumerate(summaries):
        location = f"scientific_payload.seed_summaries[{index}]"
        if not isinstance(summary, Mapping):
            errors.append(f"{location} must be an object")
            continue
        if set(summary) != {"seed", "conditions"}:
            errors.append(f"{location} keys do not match the v1 schema")
        seed = summary.get("seed")
        if type(seed) is bool or type(seed) is not int:
            errors.append(f"{location}.seed must be an integer")
        else:
            observed_seeds.append(seed)

        conditions = summary.get("conditions")
        if not isinstance(conditions, Mapping):
            errors.append(f"{location}.conditions must be an object")
            continue
        if set(conditions) != set(_EXPECTED_CONDITION_NAMES):
            errors.append(f"{location}.conditions must contain exactly the seven frozen conditions")
        for condition in _EXPECTED_CONDITION_NAMES:
            metric_payload = conditions.get(condition)
            metric_location = f"{location}.conditions.{condition}"
            if not isinstance(metric_payload, Mapping):
                errors.append(f"{metric_location} must be an object")
                continue
            if set(metric_payload) != set(_METRIC_FIELDS):
                errors.append(f"{metric_location} keys do not match the v1 metric schema")
            parsed: dict[str, float] = {}
            for field in _METRIC_FIELDS:
                value = _finite_number(metric_payload.get(field))
                if value is None:
                    errors.append(f"{metric_location}.{field} must be finite")
                    continue
                parsed[field] = value
                raw_vectors[condition][field].append(value)

            for field in (
                "normalized_regret",
                "domain_a_normalized_regret",
                "domain_b_normalized_regret",
                "oracle_pick_rate",
            ):
                value = parsed.get(field)
                if value is not None and not 0.0 <= value <= 1.0:
                    errors.append(f"{metric_location}.{field} must lie in [0, 1]")
            for field in ("reward_mae", "return_mae", "normalized_return_mae"):
                value = parsed.get(field)
                if value is not None and value < 0.0:
                    errors.append(f"{metric_location}.{field} must be non-negative")

            overall = parsed.get("normalized_regret")
            domain_a = parsed.get("domain_a_normalized_regret")
            domain_b = parsed.get("domain_b_normalized_regret")
            if (
                overall is not None
                and domain_a is not None
                and domain_b is not None
                and not math.isclose(
                    overall,
                    0.5 * (domain_a + domain_b),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                errors.append(
                    f"{metric_location}.normalized_regret is inconsistent "
                    "with the equal-sized domain means"
                )
            oracle = parsed.get("oracle_pick_rate")
            # 24 = two domains x the frozen probes_per_domain=12, so a genuine
            # pick rate must be an exact multiple of 1/24.
            probe_count = 24
            if oracle is not None and not math.isclose(
                oracle * probe_count,
                round(oracle * probe_count),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ):
                errors.append(
                    f"{metric_location}.oracle_pick_rate is not a multiple of 1/{probe_count}"
                )

    if tuple(observed_seeds) != _EXPECTED_EVIDENCE_SEEDS:
        errors.append(
            "scientific_payload.seed_summaries must contain held-out seeds "
            "30-59 exactly, in strictly increasing order"
        )
    expected_count = len(_EXPECTED_EVIDENCE_SEEDS)
    if any(
        len(values) != expected_count
        for condition in raw_vectors.values()
        for values in condition.values()
    ):
        return None
    return {
        condition: {field: np.asarray(values, dtype=np.float64) for field, values in fields.items()}
        for condition, fields in raw_vectors.items()
    }


def _validate_operational(
    operational: Mapping[str, object],
    *,
    scientific: Mapping[str, object] | None,
    errors: list[str],
) -> None:
    if set(operational) != {
        "digest_exclusion_reason",
        "generated_at_utc",
        "evaluation_wall_seconds",
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
            offset = parsed.utcoffset()
            if parsed.tzinfo is None or offset is None:
                errors.append("operational_metadata.generated_at_utc must include a timezone")
            elif offset.total_seconds() != 0.0:
                errors.append("operational_metadata.generated_at_utc must be normalized to UTC")

    wall_seconds = _finite_number(operational.get("evaluation_wall_seconds"))
    if wall_seconds is None or wall_seconds < 0.0:
        errors.append("operational_metadata.evaluation_wall_seconds must be non-negative")

    runtime = operational.get("runtime")
    if not isinstance(runtime, Mapping):
        errors.append("operational_metadata.runtime must be an object")
    else:
        if set(runtime) != {"python", "platform", "packages", "jax"}:
            errors.append("operational_metadata.runtime keys do not match the v1 schema")
        python_runtime = runtime.get("python")
        if (
            not isinstance(python_runtime, Mapping)
            or set(python_runtime) != {"implementation", "version"}
            or not all(
                isinstance(python_runtime.get(key), str) and bool(python_runtime.get(key))
                for key in ("implementation", "version")
            )
        ):
            errors.append("operational_metadata.runtime.python is invalid")
        platform_runtime = runtime.get("platform")
        if (
            not isinstance(platform_runtime, Mapping)
            or set(platform_runtime) != {"system", "release", "machine"}
            or not all(
                isinstance(platform_runtime.get(key), str)
                for key in ("system", "release", "machine")
            )
        ):
            errors.append("operational_metadata.runtime.platform is invalid")
        packages = runtime.get("packages")
        if (
            not isinstance(packages, Mapping)
            or set(packages) != {"alberta-framework", "jax", "jaxlib", "numpy"}
            or not all(
                isinstance(packages.get(key), str) and bool(packages.get(key))
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
        if not isinstance(git_worktree.get("dirty"), bool):
            errors.append("operational_metadata.git_worktree.dirty must be boolean")
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


def validate_ftl_decision_artifact(
    artifact: Mapping[str, object],
) -> ArtifactValidation:
    """Reconstruct all summaries and checks; reject unknown or stale content."""

    errors: list[str] = []
    if set(artifact) != {
        "schema_version",
        "scientific_payload",
        "scientific_digest",
        "operational_metadata",
    }:
        errors.append("artifact top-level keys do not match the v1 schema")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")

    scientific = _mapping(artifact, "scientific_payload", "artifact", errors)
    digest = _mapping(artifact, "scientific_digest", "artifact", errors)
    operational = _mapping(artifact, "operational_metadata", "artifact", errors)
    reconstructed_acceptance_passed = False

    if scientific is not None:
        expected_scientific_keys = {
            "protocol",
            "configuration",
            "thresholds",
            "bootstrap",
            "seed_summaries",
            "aggregate",
            "paired_comparisons",
            "acceptance",
            "source_provenance",
        }
        if set(scientific) != expected_scientific_keys:
            errors.append("scientific_payload keys do not match the v1 schema")

        _compare_structure(
            scientific.get("protocol"),
            _protocol_payload(),
            "scientific_payload.protocol",
            errors,
        )
        try:
            canonical_config = _canonical_config()
        except RuntimeError as error:
            errors.append(str(error))
            canonical_config = DecisionFidelityConfig()
        _compare_structure(
            scientific.get("configuration"),
            _configuration_payload(canonical_config),
            "scientific_payload.configuration",
            errors,
        )
        _compare_structure(
            scientific.get("thresholds"),
            _threshold_payload(),
            "scientific_payload.thresholds",
            errors,
        )
        _compare_structure(
            scientific.get("bootstrap"),
            _bootstrap_payload(canonical_config),
            "scientific_payload.bootstrap",
            errors,
        )
        _validate_source_provenance(
            scientific.get("source_provenance"),
            errors,
        )

        vectors = _validate_and_extract_seed_vectors(
            scientific.get("seed_summaries"),
            errors,
        )
        if vectors is not None:
            expected_aggregate = _aggregate_payload(vectors, canonical_config)
            expected_comparisons = _comparison_payload(vectors, canonical_config)
            expected_acceptance = _acceptance_payload(
                expected_aggregate,
                expected_comparisons,
            )
            _compare_structure(
                scientific.get("aggregate"),
                expected_aggregate,
                "scientific_payload.aggregate",
                errors,
            )
            _compare_structure(
                scientific.get("paired_comparisons"),
                expected_comparisons,
                "scientific_payload.paired_comparisons",
                errors,
            )
            _compare_structure(
                scientific.get("acceptance"),
                expected_acceptance,
                "scientific_payload.acceptance",
                errors,
            )
            reconstructed_acceptance_passed = expected_acceptance["passed"] is True

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
            errors.append("scientific_digest canonicalization is unsupported")
        recorded = digest.get("sha256")
        if (
            not isinstance(recorded, str)
            or len(recorded) != 64
            or any(character not in "0123456789abcdef" for character in recorded)
        ):
            errors.append("scientific_digest.sha256 must be 64 lowercase hexadecimal characters")
        elif scientific is not None:
            expected_digest = scientific_payload_sha256(scientific)
            if recorded != expected_digest:
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
        accepted=valid and reconstructed_acceptance_passed,
        errors=tuple(errors),
    )


def write_ftl_decision_artifact(
    path: Path,
    report: DecisionFidelityReport,
    *,
    evaluation_wall_seconds: float,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Build, validate, and write accepted or validly rejected evidence.

    Existing files — including the pinned canonical artifact — are immutable:
    a path that already exists is refused before any bytes are written.
    """

    expanded = path.expanduser()
    destination = expanded.resolve()
    if os.path.lexists(expanded) or os.path.lexists(destination):
        raise FileExistsError(
            f"refusing to overwrite existing evidence artifact path: {path}; "
            "reruns must write to a new path"
        )
    artifact = build_ftl_decision_artifact(
        report,
        evaluation_wall_seconds=evaluation_wall_seconds,
        generated_at=generated_at,
    )
    validation = validate_ftl_decision_artifact(artifact)
    if not validation.valid:
        raise ValueError(
            "refusing to write invalid evidence artifact: " + "; ".join(validation.errors)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(artifact_json(artifact))
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing evidence artifact path: {destination}; "
            "reruns must write to a new path"
        ) from error
    return artifact


__all__ = [
    "ArtifactValidation",
    "CANONICALIZATION",
    "DIGEST_ALGORITHM",
    "DIGEST_SCOPE",
    "FROZEN_BOOTSTRAP_RESAMPLES",
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "artifact_json",
    "build_ftl_decision_artifact",
    "canonical_scientific_bytes",
    "load_ftl_decision_artifact",
    "scientific_payload_sha256",
    "validate_ftl_decision_artifact",
    "write_ftl_decision_artifact",
]
