"""CLI for reproducible recurring multi-agent evidence artifacts.

The historical promoted run is pinned immutably at
``outputs/continual_multiagent/evidence.json``.  Reruns are reproducibility
evidence, never a new promotion, and must write to a NEW path::

    python -m alberta_framework.evaluation.continual_multiagent_cli \
        --output /new/path/continual-multiagent-evidence.json

Generation refuses the pinned canonical path and any already-existing file
before the benchmark runs.

The default run uses thirty paired seeds.  Passing a small ``--seed-count``
gives a quick development run, but it exits nonzero because the artifact
cannot satisfy the promoted minimum-seed acceptance threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from alberta_framework.evaluation.continual_multiagent import (
    AcceptanceThresholds,
    ContinualMultiAgentConfig,
    ContinualMultiAgentReport,
    evaluate_acceptance,
    run_continual_multiagent_benchmark,
)
from alberta_framework.evaluation.continual_multiagent_artifact import (
    artifact_json,
    build_evidence_artifact,
    load_evidence_artifact,
    validate_evidence_artifact,
)

# The pinned immutable promoted artifact.  Generation refuses this path (and
# any existing file) before running; pass --output with a new path.
DEFAULT_OUTPUT = Path("outputs/continual_multiagent/evidence.json")


def _resolved_new_output(path: Path) -> Path:
    expanded = path.expanduser()
    resolved = expanded.resolve()
    canonical = DEFAULT_OUTPUT.expanduser().resolve()
    if resolved == canonical:
        raise FileExistsError(
            f"refusing to write pinned canonical artifact path: {resolved}; "
            "pass --output with a new path — reruns are reproducibility "
            "evidence and never overwrite the promoted artifact"
        )
    if os.path.lexists(expanded) or os.path.lexists(resolved):
        raise FileExistsError(
            f"refusing to overwrite existing output path: {resolved}; "
            "pass --output with a new path"
        )
    return resolved


def _parser() -> argparse.ArgumentParser:
    defaults = AcceptanceThresholds()
    parser = argparse.ArgumentParser(
        description=(
            "Run or verify the narrow recurring two-agent A-B-A evidence gate."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "required for generation; must be a new path (the pinned canonical "
            f"artifact {DEFAULT_OUTPUT} and existing files are never overwritten)"
        ),
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="validate an existing artifact instead of running the benchmark",
    )
    parser.add_argument("--seed-start", type=int, default=30)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2_026_073_000)
    parser.add_argument(
        "--minimum-reward-uplift-over-frozen",
        type=float,
        default=defaults.minimum_reward_uplift_over_frozen,
    )
    parser.add_argument(
        "--minimum-coadaptation-uplift",
        type=float,
        default=defaults.minimum_partner_uplift,
        help=(
            "minimum paired joint-adaptive minus learner-only reward; this "
            "does not measure IA"
        ),
    )
    parser.add_argument(
        "--minimum-recurrent-a-probe-reward",
        type=float,
        default=defaults.minimum_recurrent_a_probe_reward,
    )
    parser.add_argument(
        "--maximum-mean-forgetting",
        type=float,
        default=defaults.maximum_mean_forgetting,
    )
    parser.add_argument(
        "--maximum-interference-forgetting",
        type=float,
        default=defaults.maximum_interference_forgetting,
    )
    parser.add_argument(
        "--minimum-recurrence-recovery-fraction",
        type=float,
        default=defaults.minimum_recurrence_recovery_fraction,
    )
    parser.add_argument(
        "--maximum-mean-recurrence-recovery-steps",
        type=float,
        default=defaults.maximum_mean_recurrence_recovery_steps,
    )
    parser.add_argument(
        "--maximum-mean-stability-gap",
        type=float,
        default=defaults.maximum_mean_stability_gap,
    )
    parser.add_argument(
        "--maximum-update-latency-ms",
        type=float,
        default=defaults.maximum_update_latency_ms,
    )
    return parser


def _thresholds(args: argparse.Namespace) -> AcceptanceThresholds:
    defaults = AcceptanceThresholds()
    return AcceptanceThresholds(
        minimum_seed_count=defaults.minimum_seed_count,
        evidence_seed_start=defaults.evidence_seed_start,
        minimum_reward_uplift_over_frozen=(
            args.minimum_reward_uplift_over_frozen
        ),
        minimum_partner_uplift=args.minimum_coadaptation_uplift,
        minimum_recurrent_a_probe_reward=(
            args.minimum_recurrent_a_probe_reward
        ),
        maximum_mean_forgetting=args.maximum_mean_forgetting,
        maximum_interference_forgetting=(
            args.maximum_interference_forgetting
        ),
        minimum_recurrence_recovery_fraction=(
            args.minimum_recurrence_recovery_fraction
        ),
        maximum_mean_recurrence_recovery_steps=(
            args.maximum_mean_recurrence_recovery_steps
        ),
        maximum_mean_stability_gap=args.maximum_mean_stability_gap,
        maximum_update_latency_ms=args.maximum_update_latency_ms,
    )


def _weakened_thresholds(
    thresholds: AcceptanceThresholds,
) -> tuple[str, ...]:
    """Identify CLI overrides weaker than the versioned canonical gate."""

    defaults = AcceptanceThresholds()
    errors: list[str] = []
    minimum_fields = (
        "minimum_reward_uplift_over_frozen",
        "minimum_partner_uplift",
        "minimum_recurrent_a_probe_reward",
        "minimum_recurrence_recovery_fraction",
    )
    maximum_fields = (
        "maximum_mean_forgetting",
        "maximum_interference_forgetting",
        "maximum_mean_recurrence_recovery_steps",
        "maximum_mean_stability_gap",
        "maximum_update_latency_ms",
    )
    for field_name in minimum_fields:
        if getattr(thresholds, field_name) < getattr(defaults, field_name):
            errors.append(f"{field_name} cannot be weaker than the canonical gate")
    for field_name in maximum_fields:
        if getattr(thresholds, field_name) > getattr(defaults, field_name):
            errors.append(f"{field_name} cannot be weaker than the canonical gate")
    return tuple(errors)


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _verify(path: Path) -> int:
    try:
        artifact = load_evidence_artifact(path)
    except FileNotFoundError as error:
        _emit(
            {
                "accepted": False,
                "artifact": str(path),
                "errors": [str(error)],
                "valid": False,
            }
        )
        return 1
    except (OSError, ValueError, json.JSONDecodeError) as error:
        _emit(
            {
                "accepted": False,
                "artifact": str(path),
                "errors": [str(error)],
                "valid": False,
            }
        )
        return 2

    try:
        validation = validate_evidence_artifact(artifact)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        _emit(
            {
                "accepted": False,
                "artifact": str(path),
                "errors": [str(error)],
                "valid": False,
            }
        )
        return 2

    _emit(
        {
            "accepted": validation.accepted,
            "artifact": str(path),
            "errors": list(validation.errors),
            "valid": validation.valid,
        }
    )
    if not validation.valid:
        return 2
    return 0 if validation.accepted else 1


def main(
    argv: Sequence[str] | None = None,
    *,
    report: ContinualMultiAgentReport | None = None,
) -> int:
    """Run/write or verify evidence; injected reports keep CLI tests cheap."""

    args = _parser().parse_args(argv)
    if args.verify is not None:
        return _verify(args.verify)
    if args.output is None:
        _emit(
            {
                "accepted": False,
                "artifact": None,
                "errors": [
                    "generation requires --output with a new path; pinned "
                    "artifacts are immutable; pass --output with a new path"
                ],
                "valid": False,
            }
        )
        return 2

    try:
        output_path = _resolved_new_output(args.output)
    except OSError as error:
        _emit(
            {
                "accepted": False,
                "artifact": str(args.output),
                "errors": [str(error)],
                "valid": False,
            }
        )
        return 2

    if args.seed_count < 1:
        _emit(
            {
                "accepted": False,
                "errors": ["seed-count must be positive"],
                "valid": False,
            }
        )
        return 2

    thresholds = _thresholds(args)
    weakened = _weakened_thresholds(thresholds)
    if weakened:
        _emit(
            {
                "accepted": False,
                "errors": list(weakened),
                "valid": False,
            }
        )
        return 2
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    if report is None:
        config = ContinualMultiAgentConfig(
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        evidence_report = run_continual_multiagent_benchmark(
            seeds=seeds,
            config=config,
            thresholds=thresholds,
        )
    else:
        if report.aggregate.seeds != seeds:
            _emit(
                {
                    "accepted": False,
                    "errors": [
                        "injected report seeds do not match the requested "
                        "seed-start/seed-count schedule"
                    ],
                    "valid": False,
                }
            )
            return 2
        acceptance = evaluate_acceptance(report.aggregate, thresholds)
        evidence_report = replace(
            report,
            thresholds=thresholds,
            acceptance=acceptance,
        )

    artifact = build_evidence_artifact(evidence_report)
    validation = validate_evidence_artifact(artifact)
    # Rejected runs are still useful parseable diagnostics.  Only the exit
    # status and validator decide promotion; writing never upgrades a failure.
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(artifact_json(artifact))
    except OSError as error:
        _emit(
            {
                "accepted": False,
                "artifact": str(args.output),
                "errors": [str(error)],
                "valid": False,
            }
        )
        return 2
    digest_record = artifact["content_digest"]
    digest = (
        digest_record.get("sha256")
        if isinstance(digest_record, dict)
        else None
    )
    _emit(
        {
            "accepted": validation.accepted,
            "artifact": str(args.output),
            "content_sha256": digest,
            "errors": list(validation.errors),
            "schema_version": artifact["schema_version"],
            "seed_count": len(evidence_report.aggregate.seeds),
            "valid": validation.valid,
        }
    )
    return 0 if validation.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
