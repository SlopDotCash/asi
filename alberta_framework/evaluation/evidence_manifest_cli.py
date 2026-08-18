"""CLI for the versioned, fail-closed evidence-claim manifest."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from alberta_framework.evaluation.evidence_manifest import (
    REPO_ROOT,
    build_evidence_manifest,
    evidence_manifest_exit_code,
    evidence_manifest_json,
)

DEFAULT_OUTPUT = REPO_ROOT / "outputs/evidence_manifest.json"


def _resolved_new_output(path: Path) -> Path:
    expanded = path.expanduser()
    resolved = expanded.resolve()
    canonical = DEFAULT_OUTPUT.expanduser().resolve()
    if resolved == canonical:
        raise FileExistsError(
            f"refusing to write pinned canonical artifact path: {resolved}; "
            "pass --output with a new path"
        )
    if os.path.lexists(expanded) or os.path.lexists(resolved):
        raise FileExistsError(
            f"refusing to overwrite existing output path: {resolved}; "
            "pass --output with a new path"
        )
    return resolved


def _emit_error(error: OSError) -> None:
    sys.stderr.write(f"error: {error}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate every registered evidence claim and distinguish accepted, "
            "valid-rejection, not-run, invalid, and nonpromoting evidence."
        )
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "optionally write the versioned JSON manifest to a new path as well as "
            f"printing it; {DEFAULT_OUTPUT} and existing paths are never overwritten"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate registered artifacts without running scientific protocols."""

    args = _parser().parse_args(argv)
    output_path: Path | None = None
    if args.output is not None:
        try:
            output_path = _resolved_new_output(args.output)
        except OSError as error:
            _emit_error(error)
            return 2

    manifest = build_evidence_manifest(args.root)
    serialized = evidence_manifest_json(manifest)
    if output_path is not None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
        except OSError as error:
            _emit_error(error)
            return 2
    sys.stdout.write(serialized)
    return evidence_manifest_exit_code(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
