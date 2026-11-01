"""Fixed Switching analysis per DEVELOPMENT_PLAN.md: the scorecard's own Switching rule,
applied to ``prototype_step010`` in place of ``prototype``.

Exit 0 = PASS, 1 = FAIL, 2 = VOID. Formulas and thresholds are taken from the scorecard module
(``reference_life_scorecard.py``), not restated:

* scale = mean over seeds of (oracle - random) reward sums; must be finite and positive;
* normalized score = (arm - random) / scale; control LCB = mean - 2.201 * sd / sqrt(12);
* control qualification: differential SARSA or SARSA with LCB > 0.10;
* beats frozen: paired LCB of (probe - frozen) / scale >= 0.05;
* noninferiority: paired LCB of (probe - best qualifying SARSA control) / scale >= -0.05,
  where "best" is the qualifying control with the higher normalized mean (ties: roster order).

PASS needs all three. The default ``prototype`` arm gets the same summary as context only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as plan_run  # noqa: E402

from alberta_framework.benchmarks import reference_life_scorecard as sc  # noqa: E402

CONTROLS = ("differential_sarsa", "sarsa")
BEATS_FROZEN_LCB = 0.05
NONINFERIORITY_LCB = -0.05


def load(directory: Path) -> dict[str, dict[int, dict[str, Any]]]:
    lives: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in plan_run.ARMS}
    for seed in plan_run.SEEDS:
        packed = json.loads((directory / f"seed{seed}.json").read_text())
        if packed["seed"] != seed or [x["arm"] for x in packed["lives"]] != list(plan_run.ARMS):
            raise ValueError(f"seed{seed}.json does not hold the fixed arm order")
        for life in packed["lives"]:
            if life["seed"] != seed or life["horizon"] != plan_run.HORIZON:
                raise ValueError("life record does not match its seed file or horizon")
            lives[life["arm"]][seed] = life
    return lives


def candidate_gate(
    candidate: str, returns: dict[str, list[float]], scale: float, best: str | None
) -> dict[str, Any]:
    frozen = [
        (c - f) / scale
        for c, f in zip(returns[candidate], returns["prototype_frozen"], strict=True)
    ]
    frozen_lcb = sc._paired_lcb(frozen)
    sarsa_lcb = None
    if best is not None:
        paired = [(c - b) / scale for c, b in zip(returns[candidate], returns[best], strict=True)]
        sarsa_lcb = sc._paired_lcb(paired)
    beats = frozen_lcb is not None and frozen_lcb >= BEATS_FROZEN_LCB
    noninferior = sarsa_lcb is not None and sarsa_lcb >= NONINFERIORITY_LCB
    return {
        "frozen_paired_lcb": frozen_lcb,
        "beats_frozen": beats,
        "best_sarsa_control": best,
        "sarsa_noninferiority_paired_lcb": sarsa_lcb,
        "sarsa_noninferior": noninferior,
        "passes": beats and noninferior,
    }


def analyze(lives: dict[str, dict[int, dict[str, Any]]]) -> tuple[str, dict[str, Any]]:
    void: list[str] = []
    for arm, by_seed in lives.items():
        for seed, life in by_seed.items():
            outcome = life["outcome"]
            if outcome.get("status") != "completed":
                void.append(f"{arm}/{seed} did not complete")
            elif not outcome["parameter_change_check"]["passed"]:
                void.append(f"{arm}/{seed} failed the scorecard parameter-change check")
            elif life["arm"] in ("prototype", plan_run.PROBE_ARM) and life[
                "prototype_base_step_size"
            ] != (plan_run.PROBE_STEP if arm == plan_run.PROBE_ARM else 0.05):
                void.append(f"{arm}/{seed} ran at the wrong base step")
    if void:
        return "VOID", {"void_reasons": void}
    returns = {
        arm: [float(lives[arm][s]["outcome"]["reward_sum"]) for s in plan_run.SEEDS]
        for arm in plan_run.ARMS
    }
    if not all(math.isfinite(v) for values in returns.values() for v in values):
        return "VOID", {"void_reasons": ["non-finite reward sum"]}
    scale = sc._mean(
        [o - r for o, r in zip(returns["privileged_oracle"], returns["random"], strict=True)]
    )
    if not (math.isfinite(scale) and scale > 0.0):
        return "VOID", {"void_reasons": [f"normalization scale {scale!r} invalid"]}
    arms: dict[str, Any] = {}
    for arm in plan_run.ARMS:
        normalized = [(a - r) / scale for a, r in zip(returns[arm], returns["random"], strict=True)]
        mean = sc._mean(normalized)
        sd = sc._sample_sd(normalized)
        arms[arm] = {
            "reward_sum_mean": sc._mean(returns[arm]),
            "normalized_score_mean": mean,
            "normalized_score_sample_sd": sd,
            "paired_t_lcb_95": mean - sc.CONTROL_GATE_T_CRITICAL * sd / math.sqrt(len(normalized)),
        }
    qualified = [
        arm for arm in CONTROLS if arms[arm]["paired_t_lcb_95"] > sc.CONTROL_GATE_LCB_THRESHOLD
    ]
    summary: dict[str, Any] = {
        "seeds": list(plan_run.SEEDS),
        "scale": scale,
        "arms": arms,
        "qualified_controls": qualified,
    }
    if not qualified:
        summary["void_reasons"] = ["no SARSA control qualified (LCB > 0.10)"]
        return "VOID", summary
    best = max(
        qualified,
        key=lambda a: (arms[a]["normalized_score_mean"], -sc.ARM_ROSTER.index(a)),
    )
    probe = candidate_gate(plan_run.PROBE_ARM, returns, scale, best)
    summary["probe_gate"] = probe
    summary["default_prototype_context_only"] = candidate_gate("prototype", returns, scale, best)
    deltas = [
        (p - d) / scale
        for p, d in zip(returns[plan_run.PROBE_ARM], returns["prototype"], strict=True)
    ]
    summary["probe_minus_default_descriptive"] = {
        "normalized_mean": sc._mean(deltas),
        "paired_t_lcb_95": sc._paired_lcb(deltas),
        "predicted_direction": "positive",
        "direction_observed": "positive" if sc._mean(deltas) > 0 else "not positive",
    }
    return ("PASS" if probe["passes"] else "FAIL"), summary


def main(argv: list[str]) -> int:
    verdict, summary = analyze(load(Path(argv[1])))
    print(json.dumps({"verdict": verdict, **summary}, indent=2, sort_keys=True, allow_nan=False))
    return {"PASS": 0, "FAIL": 1}.get(verdict, 2)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
