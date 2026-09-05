"""Fail-closed v2 artifacts for the scale-robust pair-feature protocol.

The SHA-256 digest covers the exact frozen protocol, configuration,
thresholds, counted memory, primitive paired-seed records, reconstructed
aggregates and bootstrap intervals, acceptance checks, and source hashes.
Generation time, host runtime details, worktree dirtiness, and wall time are
retained outside the digest.  The generation commit is copied into both
scientific and operational provenance and those copies must agree; validation
binds the exact current source-file hashes but deliberately does not require
the validator's later Git HEAD to equal the recorded generation commit.

Thresholds were frozen once from the direct-key development schedule 8--15.
The exact available calibration summary and structural per-seed differences
are retained in a hash-bound package record; primitive rows that were not
emitted by that run are explicitly marked unavailable rather than inferred.
An older primary-only confirmation already exposed seeds 30--45.  V2
therefore does not treat 30--59 as held out: its promoted schedule is a
fresh, deterministically namespace-derived set of 30 keys that remains
unrun.

The eventual supported claim is intentionally narrow.  This is a finite nine-phase
regression stream with an exhaustive degree-two candidate archive and a
visible, supplied task cue.  It is not evidence of general or open-ended
feature discovery, autonomous task inference, an indefinite retention
guarantee, continual control, or completion of the Alberta Plan.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import jax
import numpy as np
from numpy.typing import NDArray

from alberta_framework.evaluation.scale_robust_feature import (
    ASYMPTOTIC_WINDOW_STEPS,
    CONDITION_LEGACY,
    CONDITION_NAMES,
    CONDITION_NO_RETENTION,
    CONDITION_PRIMARY,
    DEVELOPMENT_SEEDS,
    EARLY_WINDOW_STEPS,
    EVIDENCE_SEED_DERIVATION,
    EVIDENCE_SEED_NAMESPACE,
    EVIDENCE_SEEDS,
    EXPECTED_MEMORY,
    FROZEN_STREAM_CONFIGURATION,
    TAIL_WINDOW_STEPS,
    ScaleRobustFeatureReport,
    count_relevant_context_pairs_by_task,
    frozen_configuration_payload,
)
from alberta_framework.streams.gauntlet import SEGMENT_NAMES

SCHEMA_VERSION = "alberta.scale_robust_pair_feature_evidence.v2"
PROTOCOL_VERSION = "gauntlet-exhaustive-pair-scale-robustness.v2"
DIGEST_ALGORITHM = "sha256"
DIGEST_SCOPE = "$.scientific_payload"
CANONICALIZATION = "utf8-json-sort-keys-compact-no-nan"

BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 2_026_073_004
BOOTSTRAP_METHOD = "paired-percentile-bootstrap"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATHS = (
    Path("alberta_framework/core/interaction_features.py"),
    Path("alberta_framework/streams/gauntlet.py"),
    Path("alberta_framework/evaluation/scale_robust_feature.py"),
    Path("alberta_framework/evaluation/scale_robust_feature_artifact.py"),
    Path("alberta_framework/evaluation/scale_robust_feature_cli.py"),
)
CALIBRATION_RECORD_RELATIVE_PATH = Path(
    "alberta_framework/evaluation/scale_robust_feature_calibration_v2.json"
)
CALIBRATION_RECORD_SHA256 = "a8414dad0ede1ad490292a1ebb809a0ecb42088adf36c5196ec006a2bcaae792"
_SOURCE_PROVENANCE_INTERPRETATION = (
    "reproducibility and integrity identifiers only; hashes are not "
    "signatures and do not establish artifact authenticity"
)

_PHASE_RECORD_KEYS = frozenset(
    {
        "phase_index",
        "phase_name",
        "step_count",
        "nonfinite_steps",
        "phase_squared_error_sum",
        "early_squared_error_sum",
        "early_count",
        "tail_squared_error_sum",
        "tail_count",
        "asymptotic_squared_error_sum",
        "asymptotic_count",
    }
)
_METRIC_KEYS = frozenset(
    {
        "total_nonfinite_steps",
        "savings_c",
        "savings_d",
        "savings_c_final",
        "first_c_early_mse",
        "first_c_tail_mse",
        "first_d_early_mse",
        "first_d_tail_mse",
        "recurrent_c_early_mse",
        "recurrent_c_tail_mse",
        "recurrent_d_early_mse",
        "recurrent_d_tail_mse",
        "scaled_early_mse",
        "scaled_cumulative_mse",
        "scaled_tail_mse",
        "nonlinear_mse",
        "final_c_early_mse",
        "final_c_tail_mse",
        "end_segment_5_unique_relevant_c_context_pairs",
        "end_segment_5_unique_relevant_d_context_pairs",
        "end_segment_5_unique_relevant_context_pairs_total",
        "end_segment_5_unique_relevant_context_pairs_task_minimum",
        "end_segment_7_unique_relevant_c_context_pairs",
        "end_segment_7_unique_relevant_d_context_pairs",
        "end_segment_7_unique_relevant_context_pairs_total",
        "end_segment_7_unique_relevant_context_pairs_task_minimum",
        "final_unique_relevant_c_context_pairs",
        "final_unique_relevant_d_context_pairs",
        "final_unique_relevant_context_pairs_total",
        "final_unique_relevant_context_pairs_task_minimum",
    }
)
_CONDITION_RECORD_KEYS = frozenset(
    {
        "phase_windows",
        "end_segment_5_active_pairs",
        "end_segment_7_active_pairs",
        "final_active_pairs",
        "metrics",
    }
)
_COMPARISON_NAMES = (
    "primary_minus_legacy_final_savings",
    "legacy_minus_primary_final_tail_mse",
    "primary_minus_legacy_final_unique_relevant_c_context_pairs",
    "primary_minus_legacy_final_unique_relevant_d_context_pairs",
    "primary_minus_no_retention_end_segment_7_unique_relevant_c_context_pairs",
    "primary_minus_no_retention_end_segment_7_unique_relevant_d_context_pairs",
    "no_retention_minus_primary_final_tail_mse",
)

_CALIBRATED_THRESHOLD_NAMES = frozenset(
    {
        "maximum_median_first_d_early_mse",
        "maximum_median_first_d_tail_mse",
        "maximum_median_recurrent_d_early_mse",
        "maximum_median_recurrent_d_tail_mse",
        "maximum_median_scaled_early_mse",
        "maximum_per_seed_scaled_early_mse",
        "maximum_median_scaled_cumulative_mse",
        "maximum_per_seed_scaled_cumulative_mse",
        "minimum_unique_relevant_c_context_pairs_end_segment_5",
        "minimum_unique_relevant_d_context_pairs_end_segment_5",
        "minimum_unique_relevant_c_context_pairs_end_segment_7",
        "minimum_unique_relevant_d_context_pairs_end_segment_7",
        "minimum_unique_relevant_c_context_pairs_final",
        "minimum_unique_relevant_d_context_pairs_final",
        "minimum_all_seed_primary_minus_legacy_final_unique_c_context",
        "minimum_all_seed_primary_minus_legacy_final_unique_d_context",
        ("minimum_all_seed_primary_minus_no_retention_end_segment_7_unique_c_context"),
        ("minimum_all_seed_primary_minus_no_retention_end_segment_7_unique_d_context"),
        "minimum_paired_mean_primary_minus_legacy_final_unique_c_context_ci_lower",
        "minimum_paired_mean_primary_minus_legacy_final_unique_d_context_ci_lower",
        ("minimum_paired_mean_primary_minus_no_retention_end_segment_7_unique_c_context_ci_lower"),
        ("minimum_paired_mean_primary_minus_no_retention_end_segment_7_unique_d_context_ci_lower"),
    }
)


@dataclass(frozen=True)
class ArtifactValidation:
    """Integrity and frozen-gate decision for one parsed artifact."""

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


def _threshold_payload() -> dict[str, object]:
    return {
        "calibration": {
            "status": "frozen_direct_key_development_8_15",
            "required_seed_schedule": list(DEVELOPMENT_SEEDS),
            "learner_init_seed": 123,
            "unset_thresholds": [],
            "record": {
                "path": CALIBRATION_RECORD_RELATIVE_PATH.as_posix(),
                "sha256": CALIBRATION_RECORD_SHA256,
            },
            "rule": (
                "thresholds were frozen once from direct-key development "
                "seeds 8-15; the fresh namespace-derived evidence schedule "
                "remains unrun, must not be used for tuning, and a failed "
                "evidence gate remains a valid rejection"
            ),
        },
        "minimum_seed_count": 30,
        "minimum_unique_relevant_c_context_pairs_end_segment_5": 4.0,
        "minimum_unique_relevant_d_context_pairs_end_segment_5": 4.0,
        "minimum_unique_relevant_c_context_pairs_end_segment_7": 4.0,
        "minimum_unique_relevant_d_context_pairs_end_segment_7": 4.0,
        "minimum_unique_relevant_c_context_pairs_final": 4.0,
        "minimum_unique_relevant_d_context_pairs_final": 4.0,
        "minimum_median_savings_c": 8.0,
        "minimum_median_savings_d": 5.0,
        "minimum_median_savings_c_final": 5.0,
        "maximum_median_first_c_early_mse": 15.0,
        "maximum_median_first_c_tail_mse": 2.0,
        "maximum_median_first_d_early_mse": 15.0,
        "maximum_median_first_d_tail_mse": 2.0,
        "maximum_median_recurrent_c_early_mse": 2.0,
        "maximum_median_recurrent_c_tail_mse": 2.0,
        "maximum_median_recurrent_d_early_mse": 2.0,
        "maximum_median_recurrent_d_tail_mse": 2.0,
        "maximum_median_scaled_early_mse": 100.0,
        "maximum_per_seed_scaled_early_mse": 200.0,
        "maximum_median_scaled_cumulative_mse": 50.0,
        "maximum_per_seed_scaled_cumulative_mse": 100.0,
        "maximum_median_scaled_tail_mse": 10.0,
        "maximum_per_seed_scaled_tail_mse": 50.0,
        "maximum_median_final_c_early_mse": 3.0,
        "maximum_median_final_c_tail_mse": 0.1,
        "maximum_per_seed_final_c_tail_mse": 0.1,
        "maximum_median_nonlinear_mse": 0.1,
        "minimum_all_seed_primary_minus_legacy_final_unique_c_context": 4.0,
        "minimum_all_seed_primary_minus_legacy_final_unique_d_context": 4.0,
        ("minimum_all_seed_primary_minus_no_retention_end_segment_7_unique_c_context"): 0.0,
        ("minimum_all_seed_primary_minus_no_retention_end_segment_7_unique_d_context"): 0.0,
        "minimum_all_seed_legacy_minus_primary_final_tail_mse": 0.0,
        "minimum_all_seed_primary_minus_legacy_final_savings": 0.0,
        "minimum_paired_mean_primary_minus_legacy_final_savings_ci_lower": 2.0,
        "minimum_paired_mean_legacy_minus_primary_final_tail_mse_ci_lower": 5.0,
        "minimum_paired_mean_primary_minus_legacy_final_unique_c_context_ci_lower": 4.0,
        "minimum_paired_mean_primary_minus_legacy_final_unique_d_context_ci_lower": 4.0,
        (
            "minimum_paired_mean_primary_minus_no_retention_end_segment_7_unique_c_context_ci_lower"
        ): 0.0,
        (
            "minimum_paired_mean_primary_minus_no_retention_end_segment_7_unique_d_context_ci_lower"
        ): 0.0,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "seed": BOOTSTRAP_SEED,
            "method": BOOTSTRAP_METHOD,
            "quantile_method": "linear",
            "statistic": "paired mean of per-seed differences",
        },
    }


def threshold_calibration_ready() -> bool:
    """Return whether every v2 contract threshold has development calibration."""

    thresholds = _threshold_payload()
    calibration = thresholds.get("calibration")
    status = calibration.get("status") if isinstance(calibration, Mapping) else None
    unset = calibration.get("unset_thresholds") if isinstance(calibration, Mapping) else None
    try:
        calibration_digest = _calibration_record_sha256()
    except RuntimeError:
        return False
    return (
        status == "frozen_direct_key_development_8_15"
        and unset == []
        and calibration_digest == CALIBRATION_RECORD_SHA256
        and all(
            _finite_number(thresholds.get(name)) is not None for name in _CALIBRATED_THRESHOLD_NAMES
        )
    )


def _protocol_payload() -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "supported_claim": (
            "online scale-robust selection and structural active-bank "
            "retention of unique ground-truth-relevant pair products from a "
            "counted exhaustive degree-two archive in the frozen visibly-cued "
            "nine-phase regression gauntlet"
        ),
        "schedule": list(SEGMENT_NAMES),
        "seed_roles": {
            "development_and_threshold_calibration": list(DEVELOPMENT_SEEDS),
            "fresh_promoted_evidence": list(EVIDENCE_SEEDS),
            "fresh_evidence_namespace": EVIDENCE_SEED_NAMESPACE,
            "fresh_evidence_derivation": EVIDENCE_SEED_DERIVATION,
            "historical_exposure": {
                "seeds": list(range(30, 46)),
                "protocol": "older primary-only scale-robust confirmation",
                "consequence": (
                    "seeds 30-59 are excluded from v2 evidence; prior exposure "
                    "prevents calling that namespace untouched"
                ),
            },
            "rule": (
                "mechanism development and the direct-key calibration run used "
                "seeds 8-15 to freeze the v2 contract; promoted evidence uses "
                "the fresh namespace-derived schedule exactly, with no "
                "post-evidence tuning"
            ),
        },
        "paired_design": (
            "all three conditions receive the same stream seed; every arm and "
            "every stream seed independently starts from the one fixed learner "
            "initialization key jax.random.key(123), so learner initialization "
            "does not vary across pairing units"
        ),
        "prequential_semantics": (
            "each squared error is measured from the prediction before the "
            "same observation-target transition updates the learner"
        ),
        "feature_search_space": (
            "all 91 distinct non-square products over the 14 supplied "
            "observation coordinates; 24 product slots are deployed"
        ),
        "task_information": (
            "the stream supplies two noisy visible context coordinates; the "
            "learner is not required to infer latent task identity"
        ),
        "representation_metric": (
            "separate counts for task-C context index 12 and task-D context "
            "index 13 of unique canonical active products joining that cue to "
            "one relevant input coordinate; duplicate descriptors count once"
        ),
        "representation_snapshots": (
            "after segments 5 and 7 and at program end; the end-segment-7 "
            "snapshot is after the nonlinear phase and immediately before "
            "final task-C reacquisition, so its retained-vs-no-retention "
            "comparison is not contaminated by that final exposure"
        ),
        "error_guard_windows": (
            "first 200 and last 500 prequential errors for first and recurrent "
            "C/D phases; first 200, all 3000, and last 500 errors for the "
            "scaled-C phase"
        ),
        "memory_semantics": (
            "the 91 pair descriptors and all 91 dormant output heads remain "
            "resident and are counted even when only 24 pairs are deployed"
        ),
        "excluded_claims": [
            "general or open-ended feature discovery",
            "features outside the exhaustive degree-two archive",
            "autonomous task or context inference",
            "erasure from the resident candidate archive",
            "an indefinite-lifetime retention theorem",
            "a performance benefit caused by the retention floor",
            "robustness across learner initialization keys",
            "an uncontaminated result from numeric seeds 30-59, because an "
            "older primary-only protocol exposed seeds 30-45",
            "causal attribution to scale normalization alone",
            "continual control or reinforcement learning",
            "completion of the Alberta Plan",
        ],
    }


def canonical_scientific_bytes(payload: Mapping[str, object]) -> bytes:
    """Return the canonical deterministic JSON representation."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _same(actual: object, expected: object) -> bool:
    """Type-exact equality for JSON subtrees.

    Python's ``==`` treats ``True == 1.0`` and ``30 == 30.0`` as equal, so a
    re-digested payload with punned scalar types would pass a plain ``!=``
    comparison. Comparing canonical bytes rejects any byte-level drift.
    """
    try:
        return canonical_scientific_bytes({"v": actual}) == canonical_scientific_bytes(
            {"v": expected}
        )
    except (TypeError, ValueError):
        return False


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


def _source_sha256() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path in _SOURCE_PATHS:
        try:
            source = (_REPO_ROOT / relative_path).read_bytes()
        except OSError as error:
            raise RuntimeError(f"unable to hash source {relative_path}") from error
        hashes[relative_path.as_posix()] = hashlib.sha256(source).hexdigest()
    return hashes


def current_source_fingerprint() -> dict[str, str]:
    """Hash the exact five executable sources bound by the v2 artifact."""

    return _source_sha256()


def _calibration_record_sha256() -> str:
    try:
        record = (_REPO_ROOT / CALIBRATION_RECORD_RELATIVE_PATH).read_bytes()
    except OSError as error:
        raise RuntimeError("unable to hash the frozen v2 calibration record") from error
    return hashlib.sha256(record).hexdigest()


def _source_provenance() -> dict[str, object]:
    return {
        "repository_subtree": "research/alberta",
        "git_head": _git_head(),
        "source_sha256": _source_sha256(),
        "interpretation": _SOURCE_PROVENANCE_INTERPRETATION,
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


def _phase_payload(phase: object) -> dict[str, object]:
    return {
        "phase_index": getattr(phase, "phase_index"),
        "phase_name": getattr(phase, "phase_name"),
        "step_count": getattr(phase, "step_count"),
        "nonfinite_steps": getattr(phase, "nonfinite_steps"),
        "phase_squared_error_sum": getattr(phase, "phase_squared_error_sum"),
        "early_squared_error_sum": getattr(phase, "early_squared_error_sum"),
        "early_count": getattr(phase, "early_count"),
        "tail_squared_error_sum": getattr(phase, "tail_squared_error_sum"),
        "tail_count": getattr(phase, "tail_count"),
        "asymptotic_squared_error_sum": getattr(
            phase,
            "asymptotic_squared_error_sum",
        ),
        "asymptotic_count": getattr(phase, "asymptotic_count"),
    }


def _pair_payload(pairs: Sequence[tuple[int, int]]) -> list[list[int]]:
    return [[int(left), int(right)] for left, right in pairs]


def _window_mean(
    phase: Mapping[str, object],
    sum_key: str,
    count_key: str,
) -> float | None:
    total = _finite_number(phase.get(sum_key))
    count = _strict_int(phase.get(count_key))
    if total is None or count is None or count <= 0:
        return None
    value = total / count
    return value if math.isfinite(value) else None


def _metrics_from_condition(
    condition: Mapping[str, object],
) -> dict[str, object]:
    phases_object = condition.get("phase_windows")
    if not isinstance(phases_object, list) or len(phases_object) != len(SEGMENT_NAMES):
        return {key: None for key in _METRIC_KEYS}
    phases: list[Mapping[str, object]] = []
    for phase in phases_object:
        if not isinstance(phase, Mapping):
            return {key: None for key in _METRIC_KEYS}
        phases.append(phase)

    early = [
        _window_mean(
            phase,
            "early_squared_error_sum",
            "early_count",
        )
        for phase in phases
    ]
    full = [
        _window_mean(
            phase,
            "phase_squared_error_sum",
            "step_count",
        )
        for phase in phases
    ]
    tail = [
        _window_mean(
            phase,
            "tail_squared_error_sum",
            "tail_count",
        )
        for phase in phases
    ]
    asymptotic = [
        _window_mean(
            phase,
            "asymptotic_squared_error_sum",
            "asymptotic_count",
        )
        for phase in phases
    ]
    nonfinite = [_strict_int(phase.get("nonfinite_steps")) for phase in phases]

    end_segment_5_pairs = _parsed_pairs(condition.get("end_segment_5_active_pairs"))
    end_segment_7_pairs = _parsed_pairs(condition.get("end_segment_7_active_pairs"))
    final_pairs = _parsed_pairs(condition.get("final_active_pairs"))
    if (
        any(value is None for value in (*full, *early, *tail, *asymptotic))
        or any(value is None for value in nonfinite)
        or end_segment_5_pairs is None
        or end_segment_7_pairs is None
        or final_pairs is None
    ):
        return {key: None for key in _METRIC_KEYS}

    finite_full = [float(value) for value in full if value is not None]
    finite_early = [float(value) for value in early if value is not None]
    finite_tail = [float(value) for value in tail if value is not None]
    finite_asymptotic = [float(value) for value in asymptotic if value is not None]
    finite_nonfinite = [int(value) for value in nonfinite if value is not None]
    end_5_c, end_5_d = count_relevant_context_pairs_by_task(end_segment_5_pairs)
    end_7_c, end_7_d = count_relevant_context_pairs_by_task(end_segment_7_pairs)
    final_c, final_d = count_relevant_context_pairs_by_task(final_pairs)
    return {
        "total_nonfinite_steps": sum(finite_nonfinite),
        "savings_c": finite_early[2] / max(finite_early[4], 1e-8),
        "savings_d": finite_early[3] / max(finite_early[5], 1e-8),
        "savings_c_final": finite_early[2] / max(finite_early[8], 1e-8),
        "first_c_early_mse": finite_early[2],
        "first_c_tail_mse": finite_tail[2],
        "first_d_early_mse": finite_early[3],
        "first_d_tail_mse": finite_tail[3],
        "recurrent_c_early_mse": finite_early[4],
        "recurrent_c_tail_mse": finite_tail[4],
        "recurrent_d_early_mse": finite_early[5],
        "recurrent_d_tail_mse": finite_tail[5],
        "scaled_early_mse": finite_early[6],
        "scaled_cumulative_mse": finite_full[6],
        "scaled_tail_mse": finite_tail[6],
        # gauntlet_scorecard()["nonlinear_mse"] uses the trailing half.
        "nonlinear_mse": finite_asymptotic[7],
        "final_c_early_mse": finite_early[8],
        "final_c_tail_mse": finite_tail[8],
        "end_segment_5_unique_relevant_c_context_pairs": end_5_c,
        "end_segment_5_unique_relevant_d_context_pairs": end_5_d,
        "end_segment_5_unique_relevant_context_pairs_total": end_5_c + end_5_d,
        "end_segment_5_unique_relevant_context_pairs_task_minimum": min(
            end_5_c,
            end_5_d,
        ),
        "end_segment_7_unique_relevant_c_context_pairs": end_7_c,
        "end_segment_7_unique_relevant_d_context_pairs": end_7_d,
        "end_segment_7_unique_relevant_context_pairs_total": end_7_c + end_7_d,
        "end_segment_7_unique_relevant_context_pairs_task_minimum": min(
            end_7_c,
            end_7_d,
        ),
        "final_unique_relevant_c_context_pairs": final_c,
        "final_unique_relevant_d_context_pairs": final_d,
        "final_unique_relevant_context_pairs_total": final_c + final_d,
        "final_unique_relevant_context_pairs_task_minimum": min(
            final_c,
            final_d,
        ),
    }


def _seed_payloads(report: ScaleRobustFeatureReport) -> list[dict[str, object]]:
    seed_schedule = list(report.seeds)
    if len(set(seed_schedule)) != len(seed_schedule):
        raise ValueError("report.seeds must not contain duplicate seeds")

    grouped: dict[int, dict[str, dict[str, object]]] = {}
    for record in report.records:
        conditions = grouped.setdefault(record.seed, {})
        if record.condition in conditions:
            raise ValueError(f"duplicate {record.condition!r} record for seed {record.seed}")
        primitive: dict[str, object] = {
            "phase_windows": [_phase_payload(phase) for phase in record.phases],
            "end_segment_5_active_pairs": _pair_payload(record.end_segment_5_active_pairs),
            "end_segment_7_active_pairs": _pair_payload(record.end_segment_7_active_pairs),
            "final_active_pairs": _pair_payload(record.final_active_pairs),
        }
        primitive["metrics"] = _metrics_from_condition(primitive)
        conditions[record.condition] = primitive

    expected_seeds = set(seed_schedule)
    observed_seeds = set(grouped)
    missing = [seed for seed in seed_schedule if seed not in observed_seeds]
    extra = [seed for seed in grouped if seed not in expected_seeds]
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing primitive records for seeds {missing!r}")
        if extra:
            details.append(f"extra primitive records for seeds {extra!r}")
        raise ValueError("report.seeds must exactly match primitive records: " + "; ".join(details))

    payloads: list[dict[str, object]] = []
    for seed in seed_schedule:
        conditions = grouped[seed]
        if set(conditions) != set(CONDITION_NAMES):
            raise ValueError(f"seed {seed} must contain exactly {CONDITION_NAMES!r}")
        payloads.append(
            {
                "seed": seed,
                "conditions": {condition: conditions[condition] for condition in CONDITION_NAMES},
            }
        )
    return payloads


def _metric_vector(
    seed_records: Sequence[Mapping[str, object]],
    condition: str,
    metric: str,
) -> list[float] | None:
    values: list[float] = []
    for seed_record in seed_records:
        conditions = seed_record.get("conditions")
        if not isinstance(conditions, Mapping):
            return None
        condition_record = conditions.get(condition)
        if not isinstance(condition_record, Mapping):
            return None
        metrics = condition_record.get("metrics")
        if not isinstance(metrics, Mapping):
            return None
        value = _finite_number(metrics.get(metric))
        if value is None:
            return None
        values.append(value)
    return values


def _median(values: Sequence[float] | None) -> float | None:
    if values is None or not values:
        return None
    array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    if not bool(np.all(np.isfinite(array))):
        return None
    return float(np.median(array))


def _minimum(values: Sequence[float] | None) -> float | None:
    if values is None or not values:
        return None
    return float(min(values))


def _maximum(values: Sequence[float] | None) -> float | None:
    if values is None or not values:
        return None
    return float(max(values))


def _aggregate_payload(
    seed_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    metric_names = tuple(key for key in _METRIC_KEYS if key != "total_nonfinite_steps")
    conditions: dict[str, object] = {}
    for condition in CONDITION_NAMES:
        conditions[condition] = {
            "median_metrics": {
                metric: _median(_metric_vector(seed_records, condition, metric))
                for metric in metric_names
            },
            "total_nonfinite_steps": (
                None
                if (
                    nonfinite := _metric_vector(
                        seed_records,
                        condition,
                        "total_nonfinite_steps",
                    )
                )
                is None
                else int(sum(nonfinite))
            ),
        }
    primary_scaled_early = _metric_vector(
        seed_records,
        CONDITION_PRIMARY,
        "scaled_early_mse",
    )
    primary_scaled_cumulative = _metric_vector(
        seed_records,
        CONDITION_PRIMARY,
        "scaled_cumulative_mse",
    )
    primary_scaled_tail = _metric_vector(
        seed_records,
        CONDITION_PRIMARY,
        "scaled_tail_mse",
    )
    primary_final_tail = _metric_vector(
        seed_records,
        CONDITION_PRIMARY,
        "final_c_tail_mse",
    )
    representation_metrics = {
        "minimum_unique_relevant_c_context_pairs_end_segment_5": (
            "end_segment_5_unique_relevant_c_context_pairs"
        ),
        "minimum_unique_relevant_d_context_pairs_end_segment_5": (
            "end_segment_5_unique_relevant_d_context_pairs"
        ),
        "minimum_unique_relevant_c_context_pairs_end_segment_7": (
            "end_segment_7_unique_relevant_c_context_pairs"
        ),
        "minimum_unique_relevant_d_context_pairs_end_segment_7": (
            "end_segment_7_unique_relevant_d_context_pairs"
        ),
        "minimum_unique_relevant_c_context_pairs_final": ("final_unique_relevant_c_context_pairs"),
        "minimum_unique_relevant_d_context_pairs_final": ("final_unique_relevant_d_context_pairs"),
    }
    representation_extrema = {
        extreme_name: _minimum(
            _metric_vector(
                seed_records,
                CONDITION_PRIMARY,
                metric_name,
            )
        )
        for extreme_name, metric_name in representation_metrics.items()
    }
    return {
        "seed_count": len(seed_records),
        "seeds": [seed_record.get("seed") for seed_record in seed_records],
        "conditions": conditions,
        "primary_extrema": {
            **representation_extrema,
            "maximum_scaled_early_mse": _maximum(primary_scaled_early),
            "maximum_scaled_cumulative_mse": _maximum(primary_scaled_cumulative),
            "maximum_scaled_tail_mse": _maximum(primary_scaled_tail),
            "maximum_final_c_tail_mse": _maximum(primary_final_tail),
        },
    }


def paired_bootstrap_mean_interval(
    differences: Sequence[float],
) -> dict[str, object]:
    """Deterministic paired percentile interval for a mean difference."""

    values: NDArray[np.float64] = np.asarray(tuple(differences), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired bootstrap requires a non-empty vector")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("paired bootstrap values must all be finite")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0,
        values.size,
        size=(BOOTSTRAP_RESAMPLES, values.size),
    )
    sampled_means = values[indices].mean(axis=1)
    tail = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    lower, upper = np.quantile(
        sampled_means,
        (tail, 1.0 - tail),
        method="linear",
    )
    return {
        "estimate": float(values.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "resamples": BOOTSTRAP_RESAMPLES,
        "sample_size": int(values.size),
        "seed": BOOTSTRAP_SEED,
        "method": BOOTSTRAP_METHOD,
        "quantile_method": "linear",
        "pairing_unit": "seed",
        "statistic": "mean of paired per-seed differences",
    }


def _comparison(
    seed_records: Sequence[Mapping[str, object]],
    *,
    left_condition: str,
    left_metric: str,
    right_condition: str,
    right_metric: str,
    interpretation: str,
) -> dict[str, object]:
    left = _metric_vector(
        seed_records,
        left_condition,
        left_metric,
    )
    right = _metric_vector(
        seed_records,
        right_condition,
        right_metric,
    )
    if left is None or right is None or len(left) != len(right):
        differences: list[float] = []
    else:
        differences = [
            left_value - right_value for left_value, right_value in zip(left, right, strict=True)
        ]
    per_seed = (
        [
            {
                "seed": seed_record.get("seed"),
                "difference": difference,
            }
            for seed_record, difference in zip(
                seed_records,
                differences,
                strict=True,
            )
        ]
        if len(differences) == len(seed_records)
        else []
    )
    interval: dict[str, object] | None
    if differences:
        interval = paired_bootstrap_mean_interval(differences)
    else:
        interval = None
    return {
        "definition": {
            "left_condition": left_condition,
            "left_metric": left_metric,
            "right_condition": right_condition,
            "right_metric": right_metric,
            "operation": "left minus right",
            "interpretation": interpretation,
        },
        "per_seed": per_seed,
        "minimum_difference": _minimum(differences),
        "median_difference": _median(differences),
        "paired_mean_interval": interval,
    }


def _comparisons_payload(
    seed_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "primary_minus_legacy_final_savings": _comparison(
            seed_records,
            left_condition=CONDITION_PRIMARY,
            left_metric="savings_c_final",
            right_condition=CONDITION_LEGACY,
            right_metric="savings_c_final",
            interpretation="positive favors the scale-robust primary arm",
        ),
        "legacy_minus_primary_final_tail_mse": _comparison(
            seed_records,
            left_condition=CONDITION_LEGACY,
            left_metric="final_c_tail_mse",
            right_condition=CONDITION_PRIMARY,
            right_metric="final_c_tail_mse",
            interpretation="positive is an error reduction for the primary arm",
        ),
        "primary_minus_legacy_final_unique_relevant_c_context_pairs": _comparison(
            seed_records,
            left_condition=CONDITION_PRIMARY,
            left_metric="final_unique_relevant_c_context_pairs",
            right_condition=CONDITION_LEGACY,
            right_metric="final_unique_relevant_c_context_pairs",
            interpretation=(
                "positive means the scale-robust primary arm retains more "
                "unique relevant task-C context products at program end"
            ),
        ),
        "primary_minus_legacy_final_unique_relevant_d_context_pairs": _comparison(
            seed_records,
            left_condition=CONDITION_PRIMARY,
            left_metric="final_unique_relevant_d_context_pairs",
            right_condition=CONDITION_LEGACY,
            right_metric="final_unique_relevant_d_context_pairs",
            interpretation=(
                "positive means the scale-robust primary arm retains more "
                "unique relevant task-D context products at program end"
            ),
        ),
        "primary_minus_no_retention_end_segment_7_unique_relevant_c_context_pairs": _comparison(
            seed_records,
            left_condition=CONDITION_PRIMARY,
            left_metric="end_segment_7_unique_relevant_c_context_pairs",
            right_condition=CONDITION_NO_RETENTION,
            right_metric="end_segment_7_unique_relevant_c_context_pairs",
            interpretation=(
                "retention ablation before final task-C reacquisition; "
                "positive means the floor preserves more unique task-C products"
            ),
        ),
        "primary_minus_no_retention_end_segment_7_unique_relevant_d_context_pairs": _comparison(
            seed_records,
            left_condition=CONDITION_PRIMARY,
            left_metric="end_segment_7_unique_relevant_d_context_pairs",
            right_condition=CONDITION_NO_RETENTION,
            right_metric="end_segment_7_unique_relevant_d_context_pairs",
            interpretation=(
                "retention ablation before final task-C reacquisition; "
                "positive means the floor preserves more unique task-D products"
            ),
        ),
        "no_retention_minus_primary_final_tail_mse": _comparison(
            seed_records,
            left_condition=CONDITION_NO_RETENTION,
            left_metric="final_c_tail_mse",
            right_condition=CONDITION_PRIMARY,
            right_metric="final_c_tail_mse",
            interpretation=(
                "descriptive retention ablation; positive is an error "
                "reduction for the retained primary arm and is not an "
                "acceptance-bound claim"
            ),
        ),
    }


def _comparison_value(
    comparisons: Mapping[str, object],
    comparison: str,
    field: str,
) -> float | None:
    record = comparisons.get(comparison)
    if not isinstance(record, Mapping):
        return None
    if field == "paired_mean_interval.lower":
        interval = record.get("paired_mean_interval")
        if not isinstance(interval, Mapping):
            return None
        return _finite_number(interval.get("lower"))
    return _finite_number(record.get(field))


def _aggregate_value(
    aggregate: Mapping[str, object],
    condition: str,
    metric: str,
) -> float | None:
    conditions = aggregate.get("conditions")
    if not isinstance(conditions, Mapping):
        return None
    condition_record = conditions.get(condition)
    if not isinstance(condition_record, Mapping):
        return None
    medians = condition_record.get("median_metrics")
    if not isinstance(medians, Mapping):
        return None
    return _finite_number(medians.get(metric))


def _extreme_value(
    aggregate: Mapping[str, object],
    name: str,
) -> float | None:
    extrema = aggregate.get("primary_extrema")
    if not isinstance(extrema, Mapping):
        return None
    return _finite_number(extrema.get(name))


def _check(
    name: str,
    *,
    actual: object,
    comparator: str,
    threshold: object,
    passed: bool,
) -> dict[str, object]:
    return {
        "name": name,
        "actual": actual,
        "comparator": comparator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _at_most(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def _greater(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold


def _threshold_float(
    thresholds: Mapping[str, object],
    name: str,
) -> float | None:
    value = _finite_number(thresholds.get(name))
    if value is None:
        if name in _CALIBRATED_THRESHOLD_NAMES and thresholds.get(name) is None:
            return None
        raise RuntimeError(f"frozen threshold {name!r} must be numeric")
    return value


def _threshold_int(
    thresholds: Mapping[str, object],
    name: str,
) -> int:
    value = _strict_int(thresholds.get(name))
    if value is None:
        raise RuntimeError(f"frozen threshold {name!r} must be integer")
    return value


def _acceptance_payload(
    *,
    seed_records: Sequence[Mapping[str, object]],
    memory: Mapping[str, object],
    aggregate: Mapping[str, object],
    comparisons: Mapping[str, object],
) -> dict[str, object]:
    thresholds = _threshold_payload()
    seeds = [seed_record.get("seed") for seed_record in seed_records]
    total_nonfinite = 0.0
    nonfinite_complete = True
    for condition in CONDITION_NAMES:
        conditions = aggregate.get("conditions")
        condition_record = conditions.get(condition) if isinstance(conditions, Mapping) else None
        value = (
            _finite_number(condition_record.get("total_nonfinite_steps"))
            if isinstance(condition_record, Mapping)
            else None
        )
        if value is None:
            nonfinite_complete = False
        else:
            total_nonfinite += value

    primary = CONDITION_PRIMARY
    calibration = thresholds.get("calibration")
    calibration_status = calibration.get("status") if isinstance(calibration, Mapping) else None
    all_required_thresholds_frozen = threshold_calibration_ready()
    checks = [
        _check(
            "all_required_thresholds_frozen",
            actual=calibration_status,
            comparator="==",
            threshold="frozen_direct_key_development_8_15",
            passed=all_required_thresholds_frozen,
        ),
        _check(
            "canonical_evidence_seed_schedule",
            actual=seeds,
            comparator="==",
            threshold=list(EVIDENCE_SEEDS),
            passed=seeds == list(EVIDENCE_SEEDS),
        ),
        _check(
            "minimum_seed_count",
            actual=len(seed_records),
            comparator=">=",
            threshold=thresholds["minimum_seed_count"],
            passed=len(seed_records) >= _threshold_int(thresholds, "minimum_seed_count"),
        ),
        _check(
            "all_prequential_errors_finite",
            actual=total_nonfinite if nonfinite_complete else None,
            comparator="==",
            threshold=0,
            passed=nonfinite_complete and total_nonfinite == 0.0,
        ),
        _check(
            "counted_memory_matches_frozen_budget",
            actual=memory,
            comparator="==",
            threshold=EXPECTED_MEMORY,
            passed=memory == EXPECTED_MEMORY,
        ),
    ]

    minimum_checks = (
        (
            "primary_unique_relevant_c_context_pairs_end_segment_5",
            _extreme_value(
                aggregate,
                "minimum_unique_relevant_c_context_pairs_end_segment_5",
            ),
            "minimum_unique_relevant_c_context_pairs_end_segment_5",
        ),
        (
            "primary_unique_relevant_d_context_pairs_end_segment_5",
            _extreme_value(
                aggregate,
                "minimum_unique_relevant_d_context_pairs_end_segment_5",
            ),
            "minimum_unique_relevant_d_context_pairs_end_segment_5",
        ),
        (
            "primary_unique_relevant_c_context_pairs_end_segment_7",
            _extreme_value(
                aggregate,
                "minimum_unique_relevant_c_context_pairs_end_segment_7",
            ),
            "minimum_unique_relevant_c_context_pairs_end_segment_7",
        ),
        (
            "primary_unique_relevant_d_context_pairs_end_segment_7",
            _extreme_value(
                aggregate,
                "minimum_unique_relevant_d_context_pairs_end_segment_7",
            ),
            "minimum_unique_relevant_d_context_pairs_end_segment_7",
        ),
        (
            "primary_unique_relevant_c_context_pairs_final",
            _extreme_value(
                aggregate,
                "minimum_unique_relevant_c_context_pairs_final",
            ),
            "minimum_unique_relevant_c_context_pairs_final",
        ),
        (
            "primary_unique_relevant_d_context_pairs_final",
            _extreme_value(
                aggregate,
                "minimum_unique_relevant_d_context_pairs_final",
            ),
            "minimum_unique_relevant_d_context_pairs_final",
        ),
        (
            "primary_median_savings_c",
            _aggregate_value(aggregate, primary, "savings_c"),
            "minimum_median_savings_c",
        ),
        (
            "primary_median_savings_d",
            _aggregate_value(aggregate, primary, "savings_d"),
            "minimum_median_savings_d",
        ),
        (
            "primary_median_savings_c_final",
            _aggregate_value(aggregate, primary, "savings_c_final"),
            "minimum_median_savings_c_final",
        ),
    )
    for name, actual, threshold_name in minimum_checks:
        threshold = _threshold_float(thresholds, threshold_name)
        checks.append(
            _check(
                name,
                actual=actual,
                comparator=">=" if threshold is not None else "calibration_pending",
                threshold=threshold,
                passed=threshold is not None and _at_least(actual, threshold),
            )
        )

    maximum_checks = (
        (
            "primary_median_first_c_early_mse",
            _aggregate_value(aggregate, primary, "first_c_early_mse"),
            "maximum_median_first_c_early_mse",
        ),
        (
            "primary_median_first_c_tail_mse",
            _aggregate_value(aggregate, primary, "first_c_tail_mse"),
            "maximum_median_first_c_tail_mse",
        ),
        (
            "primary_median_first_d_early_mse",
            _aggregate_value(aggregate, primary, "first_d_early_mse"),
            "maximum_median_first_d_early_mse",
        ),
        (
            "primary_median_first_d_tail_mse",
            _aggregate_value(aggregate, primary, "first_d_tail_mse"),
            "maximum_median_first_d_tail_mse",
        ),
        (
            "primary_median_recurrent_c_early_mse",
            _aggregate_value(aggregate, primary, "recurrent_c_early_mse"),
            "maximum_median_recurrent_c_early_mse",
        ),
        (
            "primary_median_recurrent_c_tail_mse",
            _aggregate_value(aggregate, primary, "recurrent_c_tail_mse"),
            "maximum_median_recurrent_c_tail_mse",
        ),
        (
            "primary_median_recurrent_d_early_mse",
            _aggregate_value(aggregate, primary, "recurrent_d_early_mse"),
            "maximum_median_recurrent_d_early_mse",
        ),
        (
            "primary_median_recurrent_d_tail_mse",
            _aggregate_value(aggregate, primary, "recurrent_d_tail_mse"),
            "maximum_median_recurrent_d_tail_mse",
        ),
        (
            "primary_median_scaled_early_mse",
            _aggregate_value(aggregate, primary, "scaled_early_mse"),
            "maximum_median_scaled_early_mse",
        ),
        (
            "primary_maximum_scaled_early_mse",
            _extreme_value(aggregate, "maximum_scaled_early_mse"),
            "maximum_per_seed_scaled_early_mse",
        ),
        (
            "primary_median_scaled_cumulative_mse",
            _aggregate_value(aggregate, primary, "scaled_cumulative_mse"),
            "maximum_median_scaled_cumulative_mse",
        ),
        (
            "primary_maximum_scaled_cumulative_mse",
            _extreme_value(aggregate, "maximum_scaled_cumulative_mse"),
            "maximum_per_seed_scaled_cumulative_mse",
        ),
        (
            "primary_median_scaled_tail_mse",
            _aggregate_value(aggregate, primary, "scaled_tail_mse"),
            "maximum_median_scaled_tail_mse",
        ),
        (
            "primary_maximum_scaled_tail_mse",
            _extreme_value(aggregate, "maximum_scaled_tail_mse"),
            "maximum_per_seed_scaled_tail_mse",
        ),
        (
            "primary_median_final_c_early_mse",
            _aggregate_value(aggregate, primary, "final_c_early_mse"),
            "maximum_median_final_c_early_mse",
        ),
        (
            "primary_median_final_c_tail_mse",
            _aggregate_value(aggregate, primary, "final_c_tail_mse"),
            "maximum_median_final_c_tail_mse",
        ),
        (
            "primary_maximum_final_c_tail_mse",
            _extreme_value(aggregate, "maximum_final_c_tail_mse"),
            "maximum_per_seed_final_c_tail_mse",
        ),
        (
            "primary_median_nonlinear_mse",
            _aggregate_value(aggregate, primary, "nonlinear_mse"),
            "maximum_median_nonlinear_mse",
        ),
    )
    for name, actual, threshold_name in maximum_checks:
        threshold = _threshold_float(thresholds, threshold_name)
        checks.append(
            _check(
                name,
                actual=actual,
                comparator="<=" if threshold is not None else "calibration_pending",
                threshold=threshold,
                passed=threshold is not None and _at_most(actual, threshold),
            )
        )

    all_seed_strict_checks = (
        (
            "all_seed_primary_final_tail_reduction_over_legacy",
            "legacy_minus_primary_final_tail_mse",
            "minimum_all_seed_legacy_minus_primary_final_tail_mse",
        ),
        (
            "all_seed_primary_final_savings_improvement_over_legacy",
            "primary_minus_legacy_final_savings",
            "minimum_all_seed_primary_minus_legacy_final_savings",
        ),
    )
    for name, comparison, threshold_name in all_seed_strict_checks:
        actual = _comparison_value(
            comparisons,
            comparison,
            "minimum_difference",
        )
        threshold = _threshold_float(thresholds, threshold_name)
        checks.append(
            _check(
                name,
                actual=actual,
                comparator=">" if threshold is not None else "calibration_pending",
                threshold=threshold,
                passed=threshold is not None and _greater(actual, threshold),
            )
        )

    all_seed_at_least_checks = (
        (
            "all_seed_primary_final_unique_c_context_improvement_over_legacy",
            "primary_minus_legacy_final_unique_relevant_c_context_pairs",
            "minimum_all_seed_primary_minus_legacy_final_unique_c_context",
        ),
        (
            "all_seed_primary_final_unique_d_context_improvement_over_legacy",
            "primary_minus_legacy_final_unique_relevant_d_context_pairs",
            "minimum_all_seed_primary_minus_legacy_final_unique_d_context",
        ),
        (
            "all_seed_primary_pre_final_c_context_noninferior_to_no_retention",
            "primary_minus_no_retention_end_segment_7_unique_relevant_c_context_pairs",
            ("minimum_all_seed_primary_minus_no_retention_end_segment_7_unique_c_context"),
        ),
        (
            "all_seed_primary_pre_final_d_context_noninferior_to_no_retention",
            "primary_minus_no_retention_end_segment_7_unique_relevant_d_context_pairs",
            ("minimum_all_seed_primary_minus_no_retention_end_segment_7_unique_d_context"),
        ),
    )
    for name, comparison, threshold_name in all_seed_at_least_checks:
        actual = _comparison_value(
            comparisons,
            comparison,
            "minimum_difference",
        )
        threshold = _threshold_float(thresholds, threshold_name)
        checks.append(
            _check(
                name,
                actual=actual,
                comparator=">=" if threshold is not None else "calibration_pending",
                threshold=threshold,
                passed=threshold is not None and _at_least(actual, threshold),
            )
        )

    interval_at_least_checks = (
        (
            "paired_mean_primary_minus_legacy_final_savings_ci_lower",
            "primary_minus_legacy_final_savings",
            "minimum_paired_mean_primary_minus_legacy_final_savings_ci_lower",
        ),
        (
            "paired_mean_legacy_minus_primary_final_tail_mse_ci_lower",
            "legacy_minus_primary_final_tail_mse",
            "minimum_paired_mean_legacy_minus_primary_final_tail_mse_ci_lower",
        ),
        (
            "paired_mean_primary_minus_legacy_final_unique_c_context_ci_lower",
            "primary_minus_legacy_final_unique_relevant_c_context_pairs",
            "minimum_paired_mean_primary_minus_legacy_final_unique_c_context_ci_lower",
        ),
        (
            "paired_mean_primary_minus_legacy_final_unique_d_context_ci_lower",
            "primary_minus_legacy_final_unique_relevant_d_context_pairs",
            "minimum_paired_mean_primary_minus_legacy_final_unique_d_context_ci_lower",
        ),
    )
    for name, comparison, threshold_name in interval_at_least_checks:
        actual = _comparison_value(
            comparisons,
            comparison,
            "paired_mean_interval.lower",
        )
        threshold = _threshold_float(thresholds, threshold_name)
        checks.append(
            _check(
                name,
                actual=actual,
                comparator=">=" if threshold is not None else "calibration_pending",
                threshold=threshold,
                passed=threshold is not None and _at_least(actual, threshold),
            )
        )

    interval_strict_checks = (
        (
            ("paired_mean_primary_minus_no_retention_end_segment_7_unique_c_context_ci_lower"),
            "primary_minus_no_retention_end_segment_7_unique_relevant_c_context_pairs",
            (
                "minimum_paired_mean_primary_minus_no_retention_"
                "end_segment_7_unique_c_context_ci_lower"
            ),
        ),
        (
            ("paired_mean_primary_minus_no_retention_end_segment_7_unique_d_context_ci_lower"),
            "primary_minus_no_retention_end_segment_7_unique_relevant_d_context_pairs",
            (
                "minimum_paired_mean_primary_minus_no_retention_"
                "end_segment_7_unique_d_context_ci_lower"
            ),
        ),
    )
    for name, comparison, threshold_name in interval_strict_checks:
        actual = _comparison_value(
            comparisons,
            comparison,
            "paired_mean_interval.lower",
        )
        threshold = _threshold_float(thresholds, threshold_name)
        checks.append(
            _check(
                name,
                actual=actual,
                comparator=">" if threshold is not None else "calibration_pending",
                threshold=threshold,
                passed=threshold is not None and _greater(actual, threshold),
            )
        )

    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "retention_ablation_interpretation": (
            "retention is assessed separately for task-C and task-D products "
            "at end segment 7, immediately before final task-C reacquisition; "
            "every paired seed must be structurally non-inferior and each "
            "paired-mean interval must exclude zero; the no-retention "
            "final-tail comparison remains descriptive, no retention-savings "
            "or retention-caused performance claim is made"
        ),
    }


def build_evidence_artifact(
    report: ScaleRobustFeatureReport,
) -> dict[str, object]:
    """Build a versioned artifact without upgrading failed evidence."""

    seed_records = _seed_payloads(report)
    memory = {
        condition: dict(report.memory_by_condition.get(condition, {}))
        for condition in CONDITION_NAMES
    }
    aggregate = _aggregate_payload(seed_records)
    comparisons = _comparisons_payload(seed_records)
    acceptance = _acceptance_payload(
        seed_records=seed_records,
        memory=memory,
        aggregate=aggregate,
        comparisons=comparisons,
    )
    source_provenance = _source_provenance()
    scientific_payload: dict[str, object] = {
        "protocol": _protocol_payload(),
        "configuration": frozen_configuration_payload(),
        "thresholds": _threshold_payload(),
        "memory_by_condition": memory,
        "seed_records": seed_records,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "acceptance": acceptance,
        "source_provenance": source_provenance,
    }
    digest = scientific_payload_sha256(scientific_payload)
    wall_times = {
        condition: float(report.wall_time_seconds_by_condition.get(condition, math.nan))
        for condition in CONDITION_NAMES
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_payload": scientific_payload,
        "content_digest": {
            "algorithm": DIGEST_ALGORITHM,
            "scope": DIGEST_SCOPE,
            "canonicalization": CANONICALIZATION,
            "sha256": digest,
        },
        "operational_metadata": {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "git_head": source_provenance["git_head"],
            "wall_time_seconds_by_condition": wall_times,
            "worktree_dirty": _git_dirty(),
            "runtime_provenance": _runtime_provenance(),
            "interpretation": (
                "host-dependent diagnostics outside the scientific digest and outside acceptance"
            ),
        },
    }


def artifact_json(artifact: Mapping[str, object]) -> str:
    """Serialize an artifact as strict, human-readable JSON."""

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


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_evidence_artifact(path: Path) -> dict[str, object]:
    """Load strict JSON, rejecting duplicate keys and NaN/Infinity."""

    loaded = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
        parse_float=_parse_finite_json_float,
        object_pairs_hook=_object_without_duplicates,
    )
    if not isinstance(loaded, dict):
        raise ValueError("artifact root must be an object")
    return loaded


def _finite_number(value: object) -> float | None:
    if type(value) is bool or (type(value) is not int and type(value) is not float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _strict_int(value: object) -> int | None:
    if type(value) is bool or type(value) is not int:
        return None
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str] | frozenset[str],
    location: str,
    errors: list[str],
) -> None:
    actual = set(value)
    missing = sorted(set(expected) - actual)
    unknown = sorted(actual - set(expected))
    if missing:
        errors.append(f"{location} is missing keys: {', '.join(missing)}")
    if unknown:
        errors.append(f"{location} has unknown keys: {', '.join(unknown)}")


def _mapping(
    value: object,
    location: str,
    errors: list[str],
) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{location} must be an object")
        return None
    return value


def _parsed_pairs(value: object) -> tuple[tuple[int, int], ...] | None:
    if not isinstance(value, list):
        return None
    pairs: list[tuple[int, int]] = []
    for pair in value:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or _strict_int(pair[0]) is None
            or _strict_int(pair[1]) is None
        ):
            return None
        pairs.append((int(pair[0]), int(pair[1])))
    return tuple(pairs)


def _validate_pair_bank(
    value: object,
    location: str,
    errors: list[str],
) -> None:
    pairs = _parsed_pairs(value)
    if pairs is None:
        errors.append(f"{location} must be a list of integer pairs")
        return
    if len(pairs) != 24:
        errors.append(f"{location} must contain exactly 24 active pairs")
    for index, (left, right) in enumerate(pairs):
        if not 0 <= left < right < 14:
            errors.append(f"{location}[{index}] must be canonical with 0 <= left < right < 14")


def _validate_phase(
    value: object,
    *,
    phase_index: int,
    location: str,
    errors: list[str],
) -> None:
    phase = _mapping(value, location, errors)
    if phase is None:
        return
    _exact_keys(phase, _PHASE_RECORD_KEYS, location, errors)
    if phase.get("phase_index") != phase_index:
        errors.append(f"{location}.phase_index is inconsistent")
    if phase.get("phase_name") != SEGMENT_NAMES[phase_index]:
        errors.append(f"{location}.phase_name is inconsistent")
    if phase.get("step_count") != FROZEN_STREAM_CONFIGURATION["segment_length"]:
        errors.append(f"{location}.step_count is inconsistent")
    nonfinite = _strict_int(phase.get("nonfinite_steps"))
    if nonfinite is None or not 0 <= nonfinite <= 3_000:
        errors.append(f"{location}.nonfinite_steps must be in [0, 3000]")
    expected_counts = {
        "early_count": EARLY_WINDOW_STEPS,
        "tail_count": TAIL_WINDOW_STEPS,
        "asymptotic_count": ASYMPTOTIC_WINDOW_STEPS,
    }
    for key, expected in expected_counts.items():
        if phase.get(key) != expected:
            errors.append(f"{location}.{key} must equal {expected}")
    for key in (
        "phase_squared_error_sum",
        "early_squared_error_sum",
        "tail_squared_error_sum",
        "asymptotic_squared_error_sum",
    ):
        value_number = _finite_number(phase.get(key))
        if value_number is None or value_number < 0.0:
            errors.append(f"{location}.{key} must be finite and non-negative")


def _validate_metrics(
    condition: Mapping[str, object],
    location: str,
    errors: list[str],
) -> None:
    metrics = _mapping(condition.get("metrics"), f"{location}.metrics", errors)
    if metrics is None:
        return
    _exact_keys(metrics, _METRIC_KEYS, f"{location}.metrics", errors)
    recomputed = _metrics_from_condition(condition)
    if not _same(metrics, recomputed):
        errors.append(f"{location}.metrics do not match primitive phase/pair records")
    nonfinite = _strict_int(metrics.get("total_nonfinite_steps"))
    if nonfinite is None or nonfinite < 0:
        errors.append(f"{location}.metrics.total_nonfinite_steps must be non-negative integer")
    for key in _METRIC_KEYS - {"total_nonfinite_steps"}:
        if _finite_number(metrics.get(key)) is None:
            errors.append(f"{location}.metrics.{key} must be finite")


def _validate_condition_record(
    value: object,
    location: str,
    errors: list[str],
) -> None:
    condition = _mapping(value, location, errors)
    if condition is None:
        return
    _exact_keys(condition, _CONDITION_RECORD_KEYS, location, errors)
    phases = condition.get("phase_windows")
    if not isinstance(phases, list) or len(phases) != len(SEGMENT_NAMES):
        errors.append(f"{location}.phase_windows must contain all nine phases")
    else:
        for phase_index, phase in enumerate(phases):
            _validate_phase(
                phase,
                phase_index=phase_index,
                location=f"{location}.phase_windows[{phase_index}]",
                errors=errors,
            )
    _validate_pair_bank(
        condition.get("end_segment_5_active_pairs"),
        f"{location}.end_segment_5_active_pairs",
        errors,
    )
    _validate_pair_bank(
        condition.get("end_segment_7_active_pairs"),
        f"{location}.end_segment_7_active_pairs",
        errors,
    )
    _validate_pair_bank(
        condition.get("final_active_pairs"),
        f"{location}.final_active_pairs",
        errors,
    )
    _validate_metrics(condition, location, errors)


def _validated_seed_records(
    value: object,
    errors: list[str],
) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        errors.append("scientific_payload.seed_records must be a list")
        return []
    records: list[Mapping[str, object]] = []
    observed_seeds: list[int] = []
    for index, item in enumerate(value):
        location = f"scientific_payload.seed_records[{index}]"
        record = _mapping(item, location, errors)
        if record is None:
            continue
        _exact_keys(record, {"seed", "conditions"}, location, errors)
        seed = _strict_int(record.get("seed"))
        if seed is None or seed < 0:
            errors.append(f"{location}.seed must be a non-negative integer")
        else:
            observed_seeds.append(seed)
        conditions = _mapping(
            record.get("conditions"),
            f"{location}.conditions",
            errors,
        )
        if conditions is not None:
            _exact_keys(
                conditions,
                set(CONDITION_NAMES),
                f"{location}.conditions",
                errors,
            )
            for condition in CONDITION_NAMES:
                _validate_condition_record(
                    conditions.get(condition),
                    f"{location}.conditions.{condition}",
                    errors,
                )
        records.append(record)
    if len(set(observed_seeds)) != len(observed_seeds):
        errors.append("scientific_payload.seed_records contain duplicate seeds")
    elif set(observed_seeds) == set(EVIDENCE_SEEDS) and observed_seeds != list(EVIDENCE_SEEDS):
        errors.append(
            "scientific_payload.seed_records must follow the frozen "
            "namespace-derived evidence-seed order"
        )
    return records


def _validate_source_provenance(
    value: object,
    errors: list[str],
) -> None:
    source = _mapping(
        value,
        "scientific_payload.source_provenance",
        errors,
    )
    if source is None:
        return
    _exact_keys(
        source,
        {"repository_subtree", "git_head", "source_sha256", "interpretation"},
        "scientific_payload.source_provenance",
        errors,
    )
    if source.get("repository_subtree") != "research/alberta":
        errors.append("source_provenance.repository_subtree is inconsistent")
    git_head = source.get("git_head")
    if (
        type(git_head) is not str
        or len(git_head) != 40
        or any(character not in "0123456789abcdef" for character in git_head)
    ):
        errors.append("source_provenance.git_head must be a canonical SHA-1")
    hashes = _mapping(
        source.get("source_sha256"),
        "scientific_payload.source_provenance.source_sha256",
        errors,
    )
    if hashes is not None:
        expected_paths = {path.as_posix() for path in _SOURCE_PATHS}
        _exact_keys(
            hashes,
            expected_paths,
            "scientific_payload.source_provenance.source_sha256",
            errors,
        )
        for path, digest in hashes.items():
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                errors.append(f"source SHA-256 for {path} is malformed")
    if source.get("interpretation") != _SOURCE_PROVENANCE_INTERPRETATION:
        errors.append("source_provenance.interpretation is inconsistent")
    try:
        current_hashes = _source_sha256()
    except RuntimeError as error:
        errors.append(f"unable to bind current source provenance: {error}")
    else:
        if hashes != current_hashes:
            errors.append(
                "scientific_payload.source_provenance does not match current pinned source hashes"
            )


def _validate_memory(value: object, errors: list[str]) -> Mapping[str, object]:
    memory = _mapping(value, "scientific_payload.memory_by_condition", errors)
    if memory is None:
        return {}
    _exact_keys(
        memory,
        set(CONDITION_NAMES),
        "scientific_payload.memory_by_condition",
        errors,
    )
    expected_fields = set(next(iter(EXPECTED_MEMORY.values())))
    for condition in CONDITION_NAMES:
        condition_memory = _mapping(
            memory.get(condition),
            f"scientific_payload.memory_by_condition.{condition}",
            errors,
        )
        if condition_memory is None:
            continue
        _exact_keys(
            condition_memory,
            expected_fields,
            f"scientific_payload.memory_by_condition.{condition}",
            errors,
        )
        for key in expected_fields:
            value_field = condition_memory.get(key)
            if key == "candidate_archive_is_counted":
                if not isinstance(value_field, bool):
                    errors.append(f"memory_by_condition.{condition}.{key} must be boolean")
            else:
                integer_value = _strict_int(value_field)
                if integer_value is None or integer_value < 0:
                    errors.append(
                        f"memory_by_condition.{condition}.{key} must be a non-negative integer"
                    )
    return memory


def _validate_comparison_shape(
    value: object,
    *,
    sample_size: int,
    location: str,
    errors: list[str],
) -> None:
    comparison = _mapping(value, location, errors)
    if comparison is None:
        return
    _exact_keys(
        comparison,
        {
            "definition",
            "per_seed",
            "minimum_difference",
            "median_difference",
            "paired_mean_interval",
        },
        location,
        errors,
    )
    definition = _mapping(
        comparison.get("definition"),
        f"{location}.definition",
        errors,
    )
    if definition is not None:
        _exact_keys(
            definition,
            {
                "left_condition",
                "left_metric",
                "right_condition",
                "right_metric",
                "operation",
                "interpretation",
            },
            f"{location}.definition",
            errors,
        )
    per_seed = comparison.get("per_seed")
    if not isinstance(per_seed, list) or len(per_seed) != sample_size:
        errors.append(f"{location}.per_seed has wrong sample size")
    else:
        for index, item in enumerate(per_seed):
            record = _mapping(
                item,
                f"{location}.per_seed[{index}]",
                errors,
            )
            if record is not None:
                _exact_keys(
                    record,
                    {"seed", "difference"},
                    f"{location}.per_seed[{index}]",
                    errors,
                )
                if _strict_int(record.get("seed")) is None:
                    errors.append(f"{location}.per_seed[{index}].seed must be integer")
                if _finite_number(record.get("difference")) is None:
                    errors.append(f"{location}.per_seed[{index}].difference must be finite")
    for key in ("minimum_difference", "median_difference"):
        if _finite_number(comparison.get(key)) is None:
            errors.append(f"{location}.{key} must be finite")
    interval = _mapping(
        comparison.get("paired_mean_interval"),
        f"{location}.paired_mean_interval",
        errors,
    )
    if interval is not None:
        _exact_keys(
            interval,
            {
                "estimate",
                "lower",
                "upper",
                "confidence_level",
                "resamples",
                "sample_size",
                "seed",
                "method",
                "quantile_method",
                "pairing_unit",
                "statistic",
            },
            f"{location}.paired_mean_interval",
            errors,
        )
        for key in ("estimate", "lower", "upper", "confidence_level"):
            if _finite_number(interval.get(key)) is None:
                errors.append(f"{location}.paired_mean_interval.{key} must be finite")
        expected = {
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "resamples": BOOTSTRAP_RESAMPLES,
            "sample_size": sample_size,
            "seed": BOOTSTRAP_SEED,
            "method": BOOTSTRAP_METHOD,
            "quantile_method": "linear",
            "pairing_unit": "seed",
            "statistic": "mean of paired per-seed differences",
        }
        for key, expected_value in expected.items():
            if interval.get(key) != expected_value:
                errors.append(f"{location}.paired_mean_interval.{key} is inconsistent")


def _validate_acceptance_shape(
    value: object,
    errors: list[str],
) -> None:
    acceptance = _mapping(value, "scientific_payload.acceptance", errors)
    if acceptance is None:
        return
    _exact_keys(
        acceptance,
        {"passed", "checks", "retention_ablation_interpretation"},
        "scientific_payload.acceptance",
        errors,
    )
    if not isinstance(acceptance.get("passed"), bool):
        errors.append("scientific_payload.acceptance.passed must be boolean")
    if not isinstance(acceptance.get("retention_ablation_interpretation"), str):
        errors.append("acceptance.retention_ablation_interpretation must be a string")
    checks = acceptance.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("scientific_payload.acceptance.checks must be non-empty list")
        return
    names: list[str] = []
    for index, item in enumerate(checks):
        location = f"scientific_payload.acceptance.checks[{index}]"
        check = _mapping(item, location, errors)
        if check is None:
            continue
        _exact_keys(
            check,
            {"name", "actual", "comparator", "threshold", "passed"},
            location,
            errors,
        )
        name = check.get("name")
        if not isinstance(name, str):
            errors.append(f"{location}.name must be string")
        else:
            names.append(name)
        if check.get("comparator") not in {
            "==",
            ">=",
            "<=",
            ">",
            "calibration_pending",
        }:
            errors.append(f"{location}.comparator is unsupported")
        if not isinstance(check.get("passed"), bool):
            errors.append(f"{location}.passed must be boolean")
    if len(names) != len(set(names)):
        errors.append("scientific_payload.acceptance checks contain duplicate names")


def _validate_operational_metadata(
    value: object,
    errors: list[str],
) -> None:
    operational = _mapping(value, "operational_metadata", errors)
    if operational is None:
        return
    _exact_keys(
        operational,
        {
            "generated_at_utc",
            "git_head",
            "wall_time_seconds_by_condition",
            "worktree_dirty",
            "runtime_provenance",
            "interpretation",
        },
        "operational_metadata",
        errors,
    )
    generated = operational.get("generated_at_utc")
    if not isinstance(generated, str):
        errors.append("operational_metadata.generated_at_utc must be string")
    else:
        try:
            parsed = datetime.fromisoformat(generated)
        except ValueError:
            errors.append("operational_metadata.generated_at_utc is invalid ISO-8601")
        else:
            if parsed.tzinfo is None:
                errors.append("operational_metadata.generated_at_utc must include timezone")
    git_head = operational.get("git_head")
    if (
        type(git_head) is not str
        or len(git_head) != 40
        or any(character not in "0123456789abcdef" for character in git_head)
    ):
        errors.append("operational_metadata.git_head must be a canonical SHA-1")
    timings = _mapping(
        operational.get("wall_time_seconds_by_condition"),
        "operational_metadata.wall_time_seconds_by_condition",
        errors,
    )
    if timings is not None:
        _exact_keys(
            timings,
            set(CONDITION_NAMES),
            "operational_metadata.wall_time_seconds_by_condition",
            errors,
        )
        for condition in CONDITION_NAMES:
            timing = _finite_number(timings.get(condition))
            if timing is None or timing < 0.0:
                errors.append(
                    f"operational wall time for {condition} must be finite and non-negative"
                )
    if not isinstance(operational.get("worktree_dirty"), bool):
        errors.append("operational_metadata.worktree_dirty must be boolean")
    if not isinstance(operational.get("interpretation"), str):
        errors.append("operational_metadata.interpretation must be string")

    runtime = _mapping(
        operational.get("runtime_provenance"),
        "operational_metadata.runtime_provenance",
        errors,
    )
    if runtime is None:
        return
    _exact_keys(
        runtime,
        {"python", "platform", "packages", "jax"},
        "operational_metadata.runtime_provenance",
        errors,
    )
    nested_schemas = {
        "python": {"implementation", "version"},
        "platform": {"system", "release", "machine"},
        "packages": {"alberta-framework", "jax", "jaxlib", "numpy"},
        "jax": {"default_backend", "devices"},
    }
    for name, keys in nested_schemas.items():
        nested = _mapping(
            runtime.get(name),
            f"operational_metadata.runtime_provenance.{name}",
            errors,
        )
        if nested is not None:
            _exact_keys(
                nested,
                keys,
                f"operational_metadata.runtime_provenance.{name}",
                errors,
            )
    jax_payload = runtime.get("jax")
    if isinstance(jax_payload, Mapping):
        devices = jax_payload.get("devices")
        if not isinstance(devices, list):
            errors.append("runtime_provenance.jax.devices must be list")
        else:
            for index, device in enumerate(devices):
                device_mapping = _mapping(
                    device,
                    f"runtime_provenance.jax.devices[{index}]",
                    errors,
                )
                if device_mapping is not None:
                    _exact_keys(
                        device_mapping,
                        {"platform", "device_kind"},
                        f"runtime_provenance.jax.devices[{index}]",
                        errors,
                    )


def validate_evidence_artifact(
    artifact: Mapping[str, object],
) -> ArtifactValidation:
    """Strictly validate structure, digest, internal bindings, and gate."""

    errors: list[str] = []
    _exact_keys(
        artifact,
        {
            "schema_version",
            "scientific_payload",
            "content_digest",
            "operational_metadata",
        },
        "$",
        errors,
    )
    if artifact.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")

    scientific = _mapping(
        artifact.get("scientific_payload"),
        "scientific_payload",
        errors,
    )
    accepted_by_payload = False
    if scientific is not None:
        _exact_keys(
            scientific,
            {
                "protocol",
                "configuration",
                "thresholds",
                "memory_by_condition",
                "seed_records",
                "aggregate",
                "comparisons",
                "acceptance",
                "source_provenance",
            },
            "scientific_payload",
            errors,
        )
        if not _same(scientific.get("protocol"), _protocol_payload()):
            errors.append("scientific_payload.protocol is not the frozen v2 protocol")
        if not _same(scientific.get("configuration"), frozen_configuration_payload()):
            errors.append("scientific_payload.configuration is not the frozen v2 configuration")
        if not _same(scientific.get("thresholds"), _threshold_payload()):
            errors.append("scientific_payload.thresholds are not the frozen v2 thresholds")
        memory = _validate_memory(
            scientific.get("memory_by_condition"),
            errors,
        )
        records = _validated_seed_records(
            scientific.get("seed_records"),
            errors,
        )
        recomputed_aggregate = _aggregate_payload(records)
        if not _same(scientific.get("aggregate"), recomputed_aggregate):
            errors.append("scientific_payload.aggregate does not match primitive records")
        comparisons = scientific.get("comparisons")
        comparisons_mapping = _mapping(
            comparisons,
            "scientific_payload.comparisons",
            errors,
        )
        if comparisons_mapping is not None:
            _exact_keys(
                comparisons_mapping,
                set(_COMPARISON_NAMES),
                "scientific_payload.comparisons",
                errors,
            )
            for name in _COMPARISON_NAMES:
                _validate_comparison_shape(
                    comparisons_mapping.get(name),
                    sample_size=len(records),
                    location=f"scientific_payload.comparisons.{name}",
                    errors=errors,
                )
        recomputed_comparisons = _comparisons_payload(records)
        if not _same(comparisons, recomputed_comparisons):
            errors.append("scientific_payload.comparisons do not match primitive records")
        recomputed_acceptance = _acceptance_payload(
            seed_records=records,
            memory=memory,
            aggregate=recomputed_aggregate,
            comparisons=recomputed_comparisons,
        )
        _validate_acceptance_shape(scientific.get("acceptance"), errors)
        if not _same(scientific.get("acceptance"), recomputed_acceptance):
            errors.append("scientific_payload.acceptance does not match recomputed gate")
        accepted_by_payload = bool(recomputed_acceptance["passed"])
        _validate_source_provenance(
            scientific.get("source_provenance"),
            errors,
        )

    digest = _mapping(
        artifact.get("content_digest"),
        "content_digest",
        errors,
    )
    if digest is not None:
        _exact_keys(
            digest,
            {"algorithm", "scope", "canonicalization", "sha256"},
            "content_digest",
            errors,
        )
        if digest.get("algorithm") != DIGEST_ALGORITHM:
            errors.append(f"content_digest.algorithm must be {DIGEST_ALGORITHM!r}")
        if digest.get("scope") != DIGEST_SCOPE:
            errors.append(f"content_digest.scope must be {DIGEST_SCOPE!r}")
        if digest.get("canonicalization") != CANONICALIZATION:
            errors.append("content_digest.canonicalization is unsupported")
        recorded = digest.get("sha256")
        if (
            type(recorded) is not str
            or len(recorded) != 64
            or any(character not in "0123456789abcdef" for character in recorded)
        ):
            errors.append("content_digest.sha256 must be canonical SHA-256")
        elif scientific is not None:
            try:
                expected_digest = scientific_payload_sha256(scientific)
            except (TypeError, ValueError) as error:
                errors.append(f"scientific_payload is not canonical finite JSON: {error}")
            else:
                if recorded != expected_digest:
                    errors.append("content_digest.sha256 does not match scientific_payload")

    _validate_operational_metadata(
        artifact.get("operational_metadata"),
        errors,
    )
    if scientific is not None:
        source = scientific.get("source_provenance")
        operational = artifact.get("operational_metadata")
        if isinstance(source, Mapping) and isinstance(operational, Mapping):
            scientific_head = source.get("git_head")
            operational_head = operational.get("git_head")
            if (
                isinstance(scientific_head, str)
                and isinstance(operational_head, str)
                and scientific_head != operational_head
            ):
                errors.append(
                    "scientific and operational generation git_head values are inconsistent"
                )
    valid = not errors
    return ArtifactValidation(
        valid=valid,
        accepted=valid and accepted_by_payload,
        errors=tuple(errors),
    )


__all__ = [
    "ArtifactValidation",
    "BOOTSTRAP_CONFIDENCE_LEVEL",
    "BOOTSTRAP_METHOD",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CALIBRATION_RECORD_RELATIVE_PATH",
    "CALIBRATION_RECORD_SHA256",
    "CANONICALIZATION",
    "DIGEST_ALGORITHM",
    "DIGEST_SCOPE",
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "artifact_json",
    "build_evidence_artifact",
    "canonical_scientific_bytes",
    "current_source_fingerprint",
    "load_evidence_artifact",
    "paired_bootstrap_mean_interval",
    "scientific_payload_sha256",
    "threshold_calibration_ready",
    "validate_evidence_artifact",
]
