"""Environment check against stored scorecard and #3308 records (amended 2026-11-01).

The #3308 Q0 = 10 life must match every recorded field bit-exactly. The three scorecard
seed-70000 lives must match the decision fields bit-exactly (accepted events, parameter
changes, reward sum; persistent bytes for sarsa and random). Two known, explained
differences are checked exactly instead of being skipped:

* ``oracle_reward_sum``: the scorecard recorder adds the per-event oracle gain once per
  event (``reference_life_scorecard.py:743``), while this runner reads the stationary
  metrics closed form ``N * gain`` (``reference_life.py:698-701``). Both must equal their
  own formula applied to the exact RiverSwim gain. ``regret_sum`` follows from each.
* ``sarsa`` manifest id and transcript: #3308 (commit e9b0760f) added
  ``"initial_q_value": None`` to the discounted-SARSA manifest config
  (``reference_life_controls.py:760``), which changes the manifest id and, through it,
  the transcript hash. The values reproduced at the base commit are pinned below.
"""

from __future__ import annotations

import json
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
SARSA_MANIFEST_DRIFT = {
    "stored_manifest": "5687f91a2bad89247810b3b74113f6d521b916b60db2242054cea090a883ea27",
    "current_manifest": "b25a382e0ea6b3928fe037d9912a7067f9f91dda5dd001d7eb627f1aae88427a",
    "stored_transcript": "a47228d61e94bfd6500d71656a7948c2c9c73b2d2f163d937736151292754580",
    "current_transcript": "3c4301d015659dc19419e50c69943cda3d7ae6863c9756f0310bc05d6ebbb59a",
}


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
    # The scorecard measures the oracle's resources with its own accounting, which this
    # runner's control_state_resource_usage does not reproduce; the oracle's bytes play no
    # role in the gate, so bytes are compared for the sarsa and random arms only.
    for phase in ("initial", "final") if arm != "privileged_oracle" else ():
        want = resources[phase]["persistent_numeric_bytes_total"]
        have = got.get(f"persistent_bytes_{phase}")
        if have != want:
            problems.append(f"persistent_bytes_{phase}: got {have!r} want {want!r}")
    events = stored["accepted_events"]
    expected_stored_oracle = sequential_sum(gain, events)
    expected_got_oracle = float(events) * gain
    if stored["oracle_reward_sum"] != expected_stored_oracle:
        problems.append("stored oracle_reward_sum is not the per-event sum of the exact gain")
    if got.get("oracle_reward_sum") != expected_got_oracle:
        problems.append("runner oracle_reward_sum is not N * exact gain")
    if stored["regret_sum"] != stored["oracle_reward_sum"] - stored["reward_sum"]:
        problems.append("stored regret_sum is not stored oracle minus stored reward")
    if got.get("regret_sum") is None:
        problems.append("runner regret_sum missing")
    manifest = record["resolved"]["agent_manifest"]["manifest_id"]
    transcript = stored["transcript_sha256"]
    if arm == "sarsa":
        if (manifest, transcript) != (
            SARSA_MANIFEST_DRIFT["stored_manifest"],
            SARSA_MANIFEST_DRIFT["stored_transcript"],
        ):
            problems.append("stored sarsa manifest or transcript is not the recorded value")
        manifest = SARSA_MANIFEST_DRIFT["current_manifest"]
        transcript = SARSA_MANIFEST_DRIFT["current_transcript"]
    if life["agent_manifest_id"] != manifest:
        problems.append(f"manifest: got {life['agent_manifest_id']!r} want {manifest!r}")
    if got.get("transcript_sha256") != transcript:
        have = got.get("transcript_sha256")
        problems.append(f"transcript_sha256: got {have!r} want {transcript!r}")
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
