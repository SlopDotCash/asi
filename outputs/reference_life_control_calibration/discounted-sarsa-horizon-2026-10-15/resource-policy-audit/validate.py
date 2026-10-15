"""Reject every noncompliant shard before the unchanged full validator runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from resource_policy import (
    EXPECTED_SETTINGS,
    FROZEN_INPUTS,
    ResourcePolicyError,
    frozen_script,
    require_settings,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=FROZEN_INPUTS)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path)
    args = parser.parse_args()
    try:
        if args.inputs.resolve() != FROZEN_INPUTS:
            raise ResourcePolicyError("This guard applies only to the frozen horizon inputs")
        if args.out.exists() or (args.audit_report and args.audit_report.exists()):
            raise ResourcePolicyError("Output already exists; overwriting is forbidden")
        frozen_script("run.py")
        script = frozen_script("validate.py")
        comparison_paths = sorted(args.records.glob("*-seed*.json"))
        preflight_paths = sorted((args.records / "preflight").glob("preflight-*.json"))
        expected_preflights = [
            args.records / "preflight" / "preflight-gamma_090.json",
            args.records / "preflight" / "preflight-gamma_099.json",
        ]
        if preflight_paths != expected_preflights:
            raise ResourcePolicyError("Expected both frozen preflight records")
        paths = comparison_paths + preflight_paths
        if not comparison_paths:
            raise ResourcePolicyError("No comparison records")
        checked = []
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            runtime = record.get("runtime")
            settings = runtime.get("thread_settings") if isinstance(runtime, dict) else None
            relative_path = path.relative_to(args.records).as_posix()
            require_settings(settings, context=relative_path)
            checked.append({
                "path": relative_path,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "recorded_settings": settings,
            })
    except (ResourcePolicyError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    # Resource-invalid inputs cannot reach aggregation or create an output file.
    result = subprocess.run([
        sys.executable, str(script), "--records", str(args.records),
        "--inputs", str(args.inputs), "--source-root", str(args.source_root),
        "--preregistration", str(args.preregistration), "--out", str(args.out),
    ], check=False)
    if result.returncode:
        return result.returncode
    if args.audit_report:
        report = {
            "schema": "asi.riverswim_sarsa_discount_horizon.resource_audit.v1",
            "development_only": True,
            "scientific_promotion_allowed": False,
            "post_collection_audit": True,
            "required_settings": EXPECTED_SETTINGS,
            "compliant_comparison_records": len(comparison_paths),
            "compliant_preflight_records": len(preflight_paths),
            "records": checked,
            "recomputed_summary_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
            "limits": "Recorded environment settings, not independently measured native "
            "thread counts, actual peak memory, execution attestation or a retroactive "
            "pre-execution guard. Frozen seed consumption and scientific status unchanged.",
        }
        with args.audit_report.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
