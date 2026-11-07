"""Environment check against stored scorecard and #3308 records.

* ``scorecard_differential_70000``: the default (knob unset) differential-SARSA life must
  match the stored scorecard shard bit-exactly in accepted events, parameter changes, reward
  sum, transcript SHA-256, agent manifest id and persistent numeric bytes.
* ``scorecard_random_70000`` / ``scorecard_oracle_70000``: the same decision fields,
  transcript and manifest; persistent bytes for random only (the scorecard measures the
  oracle's resources with its own accounting, which plays no role in the gate).
* ``oracle_reward_sum`` for all three uses the rule from the published Q0 calibration: the
  scorecard recorder adds the per-event oracle gain once per event
  (``reference_life_scorecard.py:743``), this runner reads the stationary-metrics closed form
  ``N * gain`` (``reference_life.py:698-701``). Each must equal its own formula applied to
  the exact RiverSwim gain; ``regret_sum`` follows from each.
* ``pr3308_q0_81010``: #3308's Q0 = 10 life must match every recorded field bit-exactly.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as plan_run  # noqa: E402

from alberta_framework.benchmarks import reference_life_scorecard as sc  # noqa: E402
from alberta_framework.streams.closed_loop import RiverSwimMDP  # noqa: E402

ALL_FIELDS = (
    "accepted_events",
    "parameter_change_events",
    "reward_sum",
    "oracle_reward_sum",
    "regret_sum",
    "transcript_sha256",
)
DECISION_FIELDS = ("accepted_events", "parameter_change_events", "reward_sum")


def riverswim_gain() -> float:
    config = sc._river_environment_config(sc.build_development_plan())
    return float(RiverSwimMDP(config).optimal_average_reward())


def sequential_sum(value: float, count: int) -> float:
    total = 0.0
    for _ in range(count):
        total += value
    return total


def check_q0(life: dict) -> list[str]:
    record = json.loads(plan_run.Q0_REFERENCE.read_text())
    want = {f: record["result"][f] for f in ALL_FIELDS} | {"manifest": record["agent_manifest_id"]}
    got = {f: life["outcome"].get(f) for f in ALL_FIELDS} | {"manifest": life["agent_manifest_id"]}
    return [f"{k}: got {got[k]!r} want {want[k]!r}" for k in want if got[k] != want[k]]


def check_scorecard(arm: str, seed: int, life: dict, gain: float) -> list[str]:
    path = plan_run.SCORECARD_SHARDS / f"riverswim__{arm}__{seed}.json"
    record = json.loads(path.read_text())
    stored, got = record["outcome"], life["outcome"]
    problems = [
        f"{k}: got {got.get(k)!r} want {stored[k]!r}"
        for k in DECISION_FIELDS
        if got.get(k) != stored[k]
    ]
    resources = stored["resources"]
    for phase in ("initial", "final") if arm != "privileged_oracle" else ():
        want = resources[phase]["persistent_numeric_bytes_total"]
        have = got.get(f"persistent_bytes_{phase}")
        if have != want:
            problems.append(f"persistent_bytes_{phase}: got {have!r} want {want!r}")
    events = stored["accepted_events"]
    if stored["oracle_reward_sum"] != sequential_sum(gain, events):
        problems.append("stored oracle_reward_sum is not the per-event sum of the exact gain")
    if got.get("oracle_reward_sum") != float(events) * gain:
        problems.append("runner oracle_reward_sum is not N * exact gain")
    if stored["regret_sum"] != stored["oracle_reward_sum"] - stored["reward_sum"]:
        problems.append("stored regret_sum is not stored oracle minus stored reward")
    # The stationary-metrics regret is accumulated by the runner, not formed as a difference
    # (they differ in the last bits for the oracle), so it is required present and finite.
    regret = got.get("regret_sum")
    if not isinstance(regret, float) or not math.isfinite(regret):
        problems.append("runner regret_sum missing or non-finite")
    manifest = record["resolved"]["agent_manifest"]["manifest_id"]
    if life["agent_manifest_id"] != manifest:
        problems.append(f"manifest: got {life['agent_manifest_id']!r} want {manifest!r}")
    if got.get("transcript_sha256") != stored["transcript_sha256"]:
        have = got.get("transcript_sha256")
        problems.append(f"transcript_sha256: got {have!r} want {stored['transcript_sha256']!r}")
    return problems


def main(argv: list[str]) -> int:
    directory = Path(argv[1])
    gain = riverswim_gain()
    ok = True
    for case, (arm, seed, _, _) in plan_run.ENVCHECK_CASES.items():
        life = json.loads((directory / f"{case}.json").read_text())
        if case.startswith("scorecard"):
            problems = check_scorecard(arm, seed, life, gain)
        else:
            problems = check_q0(life)
        ok &= not problems
        verdict = "MATCH" if not problems else "MISMATCH"
        transcript = life["outcome"].get("transcript_sha256")
        reward = life["outcome"].get("reward_sum")
        print(f"{case}: {verdict} transcript {transcript} reward {reward!r}")
        for problem in problems:
            print(f"  {problem}")
    print("ENVCHECK", "PASS" if ok else "VOID")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
