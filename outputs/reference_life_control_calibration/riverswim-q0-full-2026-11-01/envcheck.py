"""Compare env-check lives bit-exactly with stored scorecard and #3308 records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as plan_run  # noqa: E402

FIELDS = (
    "accepted_events",
    "parameter_change_events",
    "reward_sum",
    "oracle_reward_sum",
    "regret_sum",
    "transcript_sha256",
)


def stored(case: str) -> dict:
    arm, seed, _, _ = plan_run.ENVCHECK_CASES[case]
    if case.startswith("scorecard"):
        record = json.loads(
            (plan_run.SCORECARD_SHARDS / f"riverswim__{arm}__{seed}.json").read_text()
        )
        return {f: record["outcome"][f] for f in FIELDS} | {
            "manifest": record["resolved"]["agent_manifest"]["manifest_id"]
        }
    record = json.loads(plan_run.Q0_REFERENCE.read_text())
    return {f: record["result"][f] for f in FIELDS} | {"manifest": record["agent_manifest_id"]}


def main(argv: list[str]) -> int:
    directory = Path(argv[1])
    ok = True
    for case in plan_run.ENVCHECK_CASES:
        life = json.loads((directory / f"{case}.json").read_text())
        got = {f: life["outcome"].get(f) for f in FIELDS} | {"manifest": life["agent_manifest_id"]}
        want = stored(case)
        match = got == want
        ok &= match
        verdict = "MATCH" if match else "MISMATCH"
        transcript = got["transcript_sha256"]
        print(f"{case}: {verdict} transcript {transcript} reward {got['reward_sum']!r}")
        if not match:
            for key in want:
                if got[key] != want[key]:
                    print(f"  {key}: got {got[key]!r} want {want[key]!r}")
    print("ENVCHECK", "PASS" if ok else "VOID")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
