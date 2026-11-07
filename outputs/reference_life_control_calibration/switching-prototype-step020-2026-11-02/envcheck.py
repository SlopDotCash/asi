"""Environment and replay checks (DEVELOPMENT_PLAN.md, "Checks before any probe life").

* ``scorecard_prototype_70000``: the stored scorecard default Prototype life on consumed seed
  70000; every outcome field must match the stored shard bit-exactly and the manifest config may
  differ from the stored one only by the known added default (``host_timing_metadata``), as in
  our own #3656.
* ``probe_step020_70000``: completes, learns, runs at base step 0.20, its config differs from
  the default Prototype only in ``oak.stomp.base_step_size``, and its transcript differs from the
  default's (the new step is exercised).
* ``replay/<arm>.json``: every #3656 arm re-run on seed 93610 with #3656's own runner; outcome
  fields, transcript, manifest id and lifecycle id must equal the committed #3656 evidence, so the
  reused lives are what this head produces.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# Loaded under a unique name: a bare ``import run`` would collide with #3656's run.py.
_spec = importlib.util.spec_from_file_location("switching_step020_run", HERE / "run.py")
assert _spec is not None and _spec.loader is not None
plan_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan_run)

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
ADDED_DEFAULT_KEYS = {"host_timing_metadata": "disabled_canonical_float32_positive_zero"}


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


def compare_outcomes(got: dict, want: dict) -> list[str]:
    if got.get("status") != "completed":
        return [f"life did not complete: {got.get('error')}"]
    problems = [
        f"{k}: got {got.get(k)!r} want {want[k]!r}" for k in OUTCOME_FIELDS if got.get(k) != want[k]
    ]
    if persistent_bytes(got) != persistent_bytes(want):
        problems.append(
            f"persistent bytes: got {persistent_bytes(got)} want {persistent_bytes(want)}"
        )
    return problems


def check_stored_prototype(life: dict) -> list[str]:
    path = plan_run.SCORECARD_SHARDS / f"{plan_run.ENVIRONMENT}__prototype__70000.json"
    record = json.loads(path.read_text())
    problems = compare_outcomes(life["outcome"], record["outcome"])
    if life["lifecycle_id"] != record["lifecycle_id"]:
        problems.append("lifecycle id differs from the stored shard")
    diff = config_diff(
        record["resolved"]["agent_manifest"]["config"], life["agent_manifest"]["config"]
    )
    if diff != {k: ("<absent>", v) for k, v in ADDED_DEFAULT_KEYS.items()}:
        problems.append(f"manifest config drift beyond the known added default: {diff}")
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


def check_replay(arm: str, life: dict) -> list[str]:
    packed = json.loads(
        (plan_run.PARENT_DIR / "evidence" / f"seed{plan_run.REPLAY_SEED}.json").read_text()
    )
    want = next(x for x in packed["lives"] if x["arm"] == arm)
    problems = compare_outcomes(life["outcome"], want["outcome"])
    for key in (
        "arm",
        "seed",
        "horizon",
        "lifecycle_id",
        "agent_manifest_id",
        "prototype_base_step_size",
    ):
        if life[key] != want[key]:
            problems.append(f"{key}: got {life[key]!r} want {want[key]!r}")
    if life["outcome"].get("transcript_sha256") != want["outcome"]["transcript_sha256"]:
        problems.append("transcript differs from the committed #3656 life")
    return problems


def main(argv: list[str]) -> int:
    directory = Path(argv[1])
    ok = True

    def report(name: str, life: dict, problems: list[str]) -> None:
        nonlocal ok
        ok &= not problems
        out = life["outcome"]
        print(
            f"{name}: {'MATCH' if not problems else 'MISMATCH'} reward {out.get('reward_sum')!r} "
            f"transcript {out.get('transcript_sha256')}"
        )
        for problem in problems:
            print(f"  {problem}")

    default = json.loads((directory / "scorecard_prototype_70000.json").read_text())
    probe = json.loads((directory / "probe_step020_70000.json").read_text())
    report("scorecard_prototype_70000", default, check_stored_prototype(default))
    report("probe_step020_70000", probe, check_probe(probe, default))
    for arm in plan_run.PARENT_ARMS:
        life = json.loads((directory / "replay" / f"{arm}.json").read_text())
        report(f"replay_{arm}_{plan_run.REPLAY_SEED}", life, check_replay(arm, life))
    print("ENVCHECK", "PASS" if ok else "VOID")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
