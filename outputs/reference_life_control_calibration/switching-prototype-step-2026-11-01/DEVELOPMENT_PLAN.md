# Switching: Prototype base step 0.05 -> 0.10, a development probe

Committed before any life of this probe was run. Times are GMT. Base
`ea6eee9d0da3fce024028810b60edfb7fa4389b4`.

**This tunes the candidate, not a control.** The changed arm is the Prototype (STOMP) arm. Every
control, the frozen Prototype, random and the oracle are run with their scorecard settings.

## The one change, fixed in advance

- `alberta_framework/benchmarks/reference_life_scorecard.py:258@ea6eee9d` sets the Prototype arm's
  `base_step_size = 0.05`; `_prototype_agent_config` (`:3118-3143`) passes it to
  `STOMPConfig.base_step_size` (`alberta_framework/core/options.py:1568`, default 0.05).
- The probe arm `prototype_step010` sets **base_step_size = 0.10**. One value, chosen before any
  life, and no other value will be tried. No other field changes (tested).
- The override lives in this directory's runner (`dataclasses.replace` on the arm's config). No
  registered evidence source and no library file is edited. The default arms' manifest ids and life
  config hashes equal those built by `build_scorecard_runner` on main (tested).

## Hypothesis: tracking lag

Switching (`alberta_framework/streams/closed_loop.py`): next state = action; reward =
`payoffs[phase, s, a]`, phase = ⌊step / 250⌋ mod 2 (`closed_loop.py:398,513`; scorecard
`SWITCHING_PHASE_LENGTH` `:108`); phase A pays toggling, phase B pays staying (scorecard `:115-116`);
horizon 4000 (`:107`), 16 phases. Each state is visited about 125 times per phase.

Effective step. The base learner is linear with a per-head bias
(`options.py:~1776-1784`, `base_hidden_sizes=()` at `:1571`); `MultiHeadMLPLearner`
(`alberta_framework/core/multi_head_learner.py:444`, LMS at `:603`) updates weight and bias with the
same error (`:1330-1378`): w += αδx, b += αδ. With a one-hot state input the visited (s, a) value
moves by 2αδ, and the same action in the other state moves by αδ through the shared bias. Checked
numerically on base: α = 0.05 gives Δq = 0.1δ (other state 0.05δ); α = 0.10 gives 0.2δ (0.1δ).

Memory ≈ 1 / effective step, in own-cell visits:

| arm | α | effective step | memory (visits) | phase length (visits per state) |
|---|---|---|---|---|
| Prototype, scorecard | 0.05 | 0.10 | ≈ 10 | ≈ 125 |
| Prototype, probe | 0.10 | 0.20 | ≈ 5 | ≈ 125 |
| differential SARSA control | 0.1 (`scorecard:305`, `reference_life_controls.py:485`) | 0.20 (weights + bias, `average_reward.py:2558-2577`, `scorecard:311`) | ≈ 5 | ≈ 125 |

So the probe gives the Prototype exactly the control's effective step. After a swap, the old value
decays as (1 − step)^n: to 10% after ≈ 22 visits at 0.10 and ≈ 11 visits at 0.20, i.e. inside the
first ≈ 9% vs ≈ 18% of a phase. The 50-step post-switch window (`:109`) is about 25 visits per
state, so the probe should recover within it while the default is still re-learning.

**Predicted direction: positive** (probe normalized score above the scorecard Prototype on the same
seeds). Low confidence: the shared bias couples the two states, and a larger step adds noise in the
steady part of each phase. The probe-minus-default difference is reported as descriptive only.

## Gate (the scorecard's own, unchanged)

From `reference_life_scorecard.py@ea6eee9d`:

- normalized score = (arm − random) / scale, scale = mean(oracle − random), finite and > 0;
- a SARSA control qualifies if its one-sided 95% LCB (t = 2.201, `:136-137`, `:1206`) is > 0.10;
- the probe must beat the frozen Prototype: paired LCB ≥ 0.05 (`:428`, `:1236`);
- and be noninferior to the best qualifying SARSA control (higher normalized mean): paired LCB
  ≥ −0.05 (`:429`, `:1267-1268`); environment pass as at `:1271`.

PASS iff both hold. FAIL otherwise; FAIL is kept and published as a clean negative, with no retune
and no second value. The scorecard Prototype's gate on the same seeds is reported as context only.

**VOID** if any life fails, any parameter-change check fails (Prototype arms and SARSA controls must
change; frozen, random and oracle must not), any Prototype life ran at the wrong base step, the scale
is invalid, or no control qualifies. VOID means stop: no re-run, no replacement seeds.

## Seeds

93610-93621 (12). Fresh at staging time, with a boundary regex that also accepts `93_610` forms:
0 hits in main's tree, 0 in main's history (`git log -G`), and 0 in the diffs of all 153 open pull
requests (the 6 too large for the API were fetched and diffed against their merge base). Disjoint
from every roster used by the scorecard and by our own #3652/#3655.

## Arms and lives

Per seed: `prototype_step010`, `prototype`, `prototype_frozen`, `differential_sarsa`, `sarsa`,
`random`, `privileged_oracle` at horizon 4000. 84 lives, plus 7 environment-check lives.

## Environment check (before any probe life)

Re-runs the stored scorecard Switching lives at consumed seed 70000:

- differential SARSA, random, oracle: bit-exact outcome, persistent bytes, manifest and transcript;
- Prototype, frozen Prototype, SARSA: manifest ids moved since those shards were stored, by additive
  defaults only (`host_timing_metadata` on the Prototype configs, `initial_q_value = None` on SARSA),
  the same kind of drift explained in our own #3652. The check requires the config diff to be exactly
  those fields and the outcome to be bit-exact;
- the probe at seed 70000: completes, learns, ran at 0.10, its config differs from the default only
  in `base_step_size`, and its transcript differs from the default's (the new path is exercised).

Any failure: VOID, stop.

## Cost

Timing smoke (100 events, consumed seed 70000): Prototype arms ≈ 0.078 s per warm step plus ≈ 3 s
setup and first step; controls ≈ 0.004-0.007 s per step. Estimate ≈ 1 h wall clock with 4
one-thread workers (39 Prototype lives × ≈ 5.3 min dominate).
