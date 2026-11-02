"""Fixed Switching analysis per DEVELOPMENT_PLAN.md: the scorecard's own Switching rule,
applied to ``prototype_step020`` in place of ``prototype``, exactly as in our own #3656.

The probe lives come from this directory's evidence; every other arm on the same seeds comes
from the committed #3656 evidence (reused after the replay check). Usage:
``analyze.py <probe evidence dir> [<#3656 evidence dir>]``.

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

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# Loaded under a unique name: a bare ``import run`` would collide with #3656's run.py.
_spec = importlib.util.spec_from_file_location("switching_step020_run", HERE / "run.py")
assert _spec is not None and _spec.loader is not None
plan_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan_run)

from alberta_framework.benchmarks import reference_life_scorecard as sc  # noqa: E402

CONTROLS = ("differential_sarsa", "sarsa")
BEATS_FROZEN_LCB = 0.05
NONINFERIORITY_LCB = -0.05


ARMS = (plan_run.PROBE_ARM, *plan_run.PARENT_ARMS)


def _load_dir(directory: Path, arms: tuple[str, ...], keep: tuple[str, ...], lives: dict) -> None:
    for seed in plan_run.SEEDS:
        packed = json.loads((directory / f"seed{seed}.json").read_text())
        if packed["seed"] != seed or [x["arm"] for x in packed["lives"]] != list(arms):
            raise ValueError(f"{directory.name}/seed{seed}.json does not hold the fixed arm order")
        for life in packed["lives"]:
            if life["seed"] != seed or life["horizon"] != plan_run.HORIZON:
                raise ValueError("life record does not match its seed file or horizon")
            if life["arm"] in keep:
                lives[life["arm"]][seed] = life


def load(directory: Path, parent: Path | None = None) -> dict[str, dict[int, dict[str, Any]]]:
    lives: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    _load_dir(directory, plan_run.RUN_ARMS, plan_run.RUN_ARMS, lives)
    parent = parent or plan_run.PARENT_DIR / "evidence"
    _load_dir(parent, plan_run.PARENT_ARMS, plan_run.PARENT_ARMS, lives)
    return lives


EXPECTED_STEP = {
    plan_run.PROBE_ARM: plan_run.PROBE_STEP,
    "prototype_step010": 0.10,
    "prototype": 0.05,
}


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
            elif arm in EXPECTED_STEP and life["prototype_base_step_size"] != EXPECTED_STEP[arm]:
                void.append(f"{arm}/{seed} ran at the wrong base step")
    if void:
        return "VOID", {"void_reasons": void}
    returns = {
        arm: [float(lives[arm][s]["outcome"]["reward_sum"]) for s in plan_run.SEEDS] for arm in ARMS
    }
    if not all(math.isfinite(v) for values in returns.values() for v in values):
        return "VOID", {"void_reasons": ["non-finite reward sum"]}
    scale = sc._mean(
        [o - r for o, r in zip(returns["privileged_oracle"], returns["random"], strict=True)]
    )
    if not (math.isfinite(scale) and scale > 0.0):
        return "VOID", {"void_reasons": [f"normalization scale {scale!r} invalid"]}
    arms: dict[str, Any] = {}
    for arm in ARMS:
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
    summary["step010_context_only"] = candidate_gate("prototype_step010", returns, scale, best)

    def delta(other: str, predicted: str) -> dict[str, Any]:
        d = [
            (p - o) / scale
            for p, o in zip(returns[plan_run.PROBE_ARM], returns[other], strict=True)
        ]
        return {
            "normalized_mean": sc._mean(d),
            "paired_t_lcb_95": sc._paired_lcb(d),
            "predicted_direction": predicted,
            "direction_observed": "positive" if sc._mean(d) > 0 else "not positive",
        }

    summary["probe_minus_default_descriptive"] = delta("prototype", "positive")
    summary["probe_minus_step010_descriptive"] = delta("prototype_step010", "not positive")
    return ("PASS" if probe["passes"] else "FAIL"), summary


def main(argv: list[str]) -> int:
    parent = Path(argv[2]) if len(argv) > 2 else None
    verdict, summary = analyze(load(Path(argv[1]), parent))
    print(json.dumps({"verdict": verdict, **summary}, indent=2, sort_keys=True, allow_nan=False))
    return {"PASS": 0, "FAIL": 1}.get(verdict, 2)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
