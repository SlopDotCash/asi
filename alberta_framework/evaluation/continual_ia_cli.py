"""CLI for validating historical IA evidence and rerunning consumed seeds.

The historical promoted run is pinned immutably at
``outputs/continual_ia/evidence.json``.  Rerunning the frozen 30-seed
protocol is reproducibility evidence, never a new promotion, and must write
to a NEW path::

    python -m alberta_framework.evaluation.continual_ia_cli \
        --output /new/path/continual-ia-evidence.json

Generation refuses the pinned canonical path and any already-existing file
before the protocol runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from alberta_framework.evaluation.continual_ia import (
    PROMOTED_EVIDENCE_SEEDS,
    ContinualIAReport,
    IAAcceptanceThresholds,
    run_continual_ia_benchmark,
)
from alberta_framework.evaluation.continual_ia_artifact import (
    NONPROMOTING_REPLAY_POLICY,
    build_ia_consumed_seed_replay,
    ia_artifact_json,
    load_ia_evidence_artifact,
    validate_ia_evidence_artifact,
)

# The pinned immutable promoted artifact.  Generation refuses this path (and
# any existing file) before running; pass --output with a new path.
DEFAULT_OUTPUT = Path("outputs/continual_ia/evidence.json")


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
    parser = argparse.ArgumentParser(
        description=("Run a nonpromoting consumed-seed replay or verify an IA artifact.")
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
        help="validate an existing artifact instead of running",
    )
    return parser


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _verify(path: Path) -> int:
    try:
        artifact = load_ia_evidence_artifact(path)
        validation = validate_ia_evidence_artifact(artifact)
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
    report: ContinualIAReport | None = None,
) -> int:
    """Run/write or verify evidence; report injection avoids test reruns."""

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

    thresholds = IAAcceptanceThresholds()
    seeds = PROMOTED_EVIDENCE_SEEDS
    if report is None:
        evidence_report = run_continual_ia_benchmark(
            seeds=seeds,
            thresholds=thresholds,
        )
    else:
        if report.aggregate.seeds != seeds:
            _emit(
                {
                    "accepted": False,
                    "errors": ["injected report seeds do not match the requested schedule"],
                    "valid": False,
                }
            )
            return 2
        if report.thresholds != thresholds:
            _emit(
                {
                    "accepted": False,
                    "errors": ["injected report thresholds are not the frozen v1 thresholds"],
                    "valid": False,
                }
            )
            return 2
        evidence_report = report

    artifact = build_ia_consumed_seed_replay(evidence_report)
    validation = validate_ia_evidence_artifact(artifact)
    if not validation.valid:
        _emit(
            {
                "accepted": False,
                "artifact": str(args.output),
                "errors": list(validation.errors),
                "valid": False,
            }
        )
        return 2
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(ia_artifact_json(artifact))
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
    digest = artifact["content_digest"]
    digest_value = digest.get("sha256") if isinstance(digest, dict) else None
    _emit(
        {
            "accepted": validation.accepted,
            "artifact": str(args.output),
            "content_sha256": digest_value,
            "evidence_policy": NONPROMOTING_REPLAY_POLICY,
            "errors": list(validation.errors),
            "primary_passed": evidence_report.acceptance.primary_passed,
            "schema_version": artifact["schema_version"],
            "secondary_passed": evidence_report.acceptance.secondary_passed,
            "seed_count": len(evidence_report.aggregate.seeds),
            "valid": validation.valid,
        }
    )
    return 0 if validation.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
