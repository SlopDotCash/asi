"""Generate or verify the scale-robust pair-feature evidence artifact.

Generation intentionally exposes no seed, configuration, or threshold tuning
flags.  Direct-key development seeds 8--15 froze the v2 thresholds once; the
historical promotion used the then-fresh frozen 30-key
SHA-256-namespace-derived set. Executing that now-consumed schedule at a new
path is reproducibility evidence, not a new promotion::

    python -m alberta_framework.evaluation.scale_robust_feature_cli \
        --output /new/path/scale-robust-evidence.json

Generation requires a new path and never overwrites the pinned canonical
artifact or any other existing file. Verification never executes the
scientific protocol::

    python -m alberta_framework.evaluation.scale_robust_feature_cli \
        --verify outputs/scale_robust_feature/evidence.v2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from alberta_framework.evaluation.scale_robust_feature import (
    EVIDENCE_SEEDS,
    ScaleRobustFeatureReport,
    run_scale_robust_feature_evaluation,
)
from alberta_framework.evaluation.scale_robust_feature_artifact import (
    artifact_json,
    build_evidence_artifact,
    current_source_fingerprint,
    load_evidence_artifact,
    threshold_calibration_ready,
    validate_evidence_artifact,
)

DEFAULT_OUTPUT = Path("outputs/scale_robust_feature/evidence.v2.json")


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


def _atomic_create_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite concurrently created output path: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or verify the frozen exhaustive-pair scale-robustness evidence artifact."
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
        help="validate an existing artifact without running the protocol",
    )
    return parser


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
    report: ScaleRobustFeatureReport | None = None,
) -> int:
    """Generate an evidence artifact at a new path or verify an existing one.

    ``report`` is an injection seam for synthetic tests.  Production CLI use
    cannot alter the evidence seed schedule, learner configurations, windows,
    bootstrap policy, or frozen thresholds.
    """

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

    production_run = report is None
    if production_run and not threshold_calibration_ready():
        _emit(
            {
                "accepted": False,
                "errors": [
                    "frozen v2 thresholds or their calibration record are "
                    "missing or inconsistent; the fresh evidence schedule "
                    "was not executed"
                ],
                "valid": False,
            }
        )
        return 2

    source_before: dict[str, str] | None = None
    if production_run:
        try:
            source_before = current_source_fingerprint()
            evidence_report = run_scale_robust_feature_evaluation()
            source_after = current_source_fingerprint()
        except (OSError, RuntimeError, ValueError) as error:
            _emit(
                {
                    "accepted": False,
                    "errors": [str(error)],
                    "valid": False,
                }
            )
            return 2
        if source_after != source_before:
            _emit(
                {
                    "accepted": False,
                    "errors": [
                        "one or more registered protocol sources changed "
                        "during evidence execution; no artifact was written"
                    ],
                    "valid": False,
                }
            )
            return 2
        if not threshold_calibration_ready():
            _emit(
                {
                    "accepted": False,
                    "errors": [
                        "the frozen v2 calibration record changed or became "
                        "unavailable during evidence execution; no artifact "
                        "was written"
                    ],
                    "valid": False,
                }
            )
            return 2
    else:
        assert report is not None
        evidence_report = report

    if evidence_report.seeds != EVIDENCE_SEEDS:
        _emit(
            {
                "accepted": False,
                "errors": [
                    "report seed schedule must be the frozen fresh "
                    "namespace-derived 30-seed schedule"
                ],
                "valid": False,
            }
        )
        return 2

    try:
        artifact = build_evidence_artifact(evidence_report)
        validation = validate_evidence_artifact(artifact)
        if source_before is not None:
            scientific = artifact.get("scientific_payload")
            provenance = (
                scientific.get("source_provenance") if isinstance(scientific, dict) else None
            )
            artifact_fingerprint = (
                provenance.get("source_sha256") if isinstance(provenance, dict) else None
            )
            source_before_write = current_source_fingerprint()
            if artifact_fingerprint != source_before or source_before_write != source_before:
                raise RuntimeError(
                    "one or more registered protocol sources changed "
                    "before artifact serialization; no artifact was written"
                )
            if not threshold_calibration_ready():
                raise RuntimeError(
                    "the frozen v2 calibration record changed or became "
                    "unavailable before artifact serialization; no artifact "
                    "was written"
                )
        serialized = artifact_json(artifact)
    except (OSError, RuntimeError, ValueError) as error:
        _emit(
            {
                "accepted": False,
                "errors": [str(error)],
                "valid": False,
            }
        )
        return 2

    if not validation.valid:
        _emit(
            {
                "accepted": False,
                "artifact": str(args.output),
                "errors": list(validation.errors),
                "schema_version": artifact["schema_version"],
                "seed_count": len(evidence_report.seeds),
                "valid": False,
            }
        )
        return 2

    try:
        _atomic_create_text(output_path, serialized)
    except OSError as error:
        _emit(
            {
                "accepted": False,
                "artifact": str(args.output),
                "errors": [str(error)],
                "schema_version": artifact["schema_version"],
                "seed_count": len(evidence_report.seeds),
                "valid": False,
            }
        )
        return 2
    digest_record = artifact["content_digest"]
    digest = digest_record.get("sha256") if isinstance(digest_record, dict) else None
    _emit(
        {
            "accepted": validation.accepted,
            "artifact": str(args.output),
            "content_sha256": digest,
            "errors": list(validation.errors),
            "schema_version": artifact["schema_version"],
            "seed_count": len(evidence_report.seeds),
            "valid": validation.valid,
        }
    )
    return 0 if validation.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
