"""Environment check on consumed scorecard seed 70000 (DEVELOPMENT_PLAN.md, "Environment check").

* ``differential_sarsa``, ``random``, ``privileged_oracle``: their agent manifests are unchanged
  since the stored scorecard run, so every outcome field (rewards, oracle and regret sums,
  phase sums, windows, parameter changes, persistent bytes) and the transcript SHA-256 and
  manifest id must match the stored shard bit-exactly.
* ``prototype``, ``prototype_frozen``, ``sarsa``: their manifests have since gained fields
  that carry their old behaviour as defaults (``host_timing_metadata`` = disabled for the
  Prototype; ``initial_q_value`` = None for SARSA). The manifest config must differ from the
  stored one in exactly those added keys, and every outcome field must match bit-exactly. The
  transcript digest may differ only if the manifest differs (it binds the manifest id).
* ``probe_step010_70000``: the probe path must complete, learn, run at base step 0.10, and
  have a manifest config that differs from the default Prototype only in
  ``oak.stomp.base_step_size``. Its outcome must differ from the default life on the same seed
  (the new step is actually exercised).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as plan_run  # noqa: E402

OUTCOME_FIELDS = (
    "accepted_events",
    "parameter_change_events",
    "reward_sum",
    "oracle_reward_sum",
    "regret_sum",
    "phase_event_counts",
    "phase_reward_sums",
    "windows",
    "environment_rng_cursor",
    "parameter_change_check",
)
ADDED_DEFAULT_KEYS = {
    "prototype": {"host_timing_metadata": "disabled_canonical_float32_positive_zero"},
    "prototype_frozen": {"host_timing_metadata": "disabled_canonical_float32_positive_zero"},
    "sarsa": {"initial_q_value": None},
}


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out |= flatten(item, f"{prefix}.{key}" if prefix else key)
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            out |= flatten(item, f"{prefix}[{index}]")
        return out
    return {prefix: value}


def config_diff(a: dict, b: dict) -> dict[str, tuple[Any, Any]]:
    fa, fb = flatten(a), flatten(b)
    missing = "<absent>"
    return {
        k: (fa.get(k, missing), fb.get(k, missing))
        for k in sorted(set(fa) | set(fb))
        if fa.get(k, missing) != fb.get(k, missing)
    }


def persistent_bytes(outcome: dict) -> tuple[Any, Any]:
    resources = outcome.get("resources") or {}
    return tuple(
        (resources.get(phase) or {}).get("persistent_numeric_bytes_total")
        for phase in ("initial", "final")
    )


def check_stored(arm: str, life: dict) -> list[str]:
    path = plan_run.SCORECARD_SHARDS / f"{plan_run.ENVIRONMENT}__{arm}__70000.json"
    record = json.loads(path.read_text())
    stored, got = record["outcome"], life["outcome"]
    if got.get("status") != "completed":
        return [f"life did not complete: {got.get('error')}"]
    problems = [
        f"{k}: got {got.get(k)!r} want {stored[k]!r}"
        for k in OUTCOME_FIELDS
        if got.get(k) != stored[k]
    ]
    if persistent_bytes(got) != persistent_bytes(stored):
        have, want = persistent_bytes(got), persistent_bytes(stored)
        problems.append(f"persistent bytes: got {have} want {want}")
    if life["lifecycle_id"] != record["lifecycle_id"]:
        problems.append("lifecycle id differs from the stored shard")
    stored_manifest = record["resolved"]["agent_manifest"]
    if arm in ADDED_DEFAULT_KEYS:
        diff = config_diff(stored_manifest["config"], life["agent_manifest"]["config"])
        want = {k: ("<absent>", v) for k, v in ADDED_DEFAULT_KEYS[arm].items()}
        if diff != want:
            problems.append(f"manifest config drift beyond the known added defaults: {diff}")
        manifest_same = life["agent_manifest_id"] == stored_manifest["manifest_id"]
    else:
        manifest_same = True
        if life["agent_manifest_id"] != stored_manifest["manifest_id"]:
            problems.append("manifest id differs from the stored shard")
    if manifest_same and got.get("transcript_sha256") != stored["transcript_sha256"]:
        problems.append("transcript differs although the manifest is unchanged")
    return problems


def check_probe(life: dict, default: dict) -> list[str]:
    got = life["outcome"]
    if got.get("status") != "completed":
        return [f"probe life did not complete: {got.get('error')}"]
    problems = []
    if life["prototype_base_step_size"] != plan_run.PROBE_STEP:
        problems.append(f"probe base step {life['prototype_base_step_size']!r}")
    if not got["parameter_change_check"]["passed"]:
        problems.append("probe arm did not learn")
    diff = config_diff(default["agent_manifest"]["config"], life["agent_manifest"]["config"])
    if [k for k in diff if not k.endswith("base_step_size")] or len(diff) != 1:
        problems.append(f"probe manifest differs from the default beyond the base step: {diff}")
    if got.get("transcript_sha256") == default["outcome"].get("transcript_sha256"):
        problems.append("probe transcript equals the default: new step not exercised")
    return problems


def main(argv: list[str]) -> int:
    directory = Path(argv[1])
    lives = {
        case: json.loads((directory / f"{case}.json").read_text())
        for case in plan_run.ENVCHECK_CASES
    }
    ok = True
    for case, (arm, _seed) in plan_run.ENVCHECK_CASES.items():
        life = lives[case]
        if arm == plan_run.PROBE_ARM:
            problems = check_probe(life, lives["scorecard_prototype_70000"])
        else:
            problems = check_stored(arm, life)
        ok &= not problems
        out = life["outcome"]
        print(
            f"{case}: {'MATCH' if not problems else 'MISMATCH'} reward {out.get('reward_sum')!r} "
            f"transcript {out.get('transcript_sha256')}"
        )
        for problem in problems:
            print(f"  {problem}")
    print("ENVCHECK", "PASS" if ok else "VOID")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
