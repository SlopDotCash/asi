**Switching: Prototype base step 0.05 -> 0.20, a development probe (step 2 of our own #3656)**

Committed before any life of this probe was run. Times are GMT. Base: the head of our own #3656,
`fe1bfe232fd3606cdabd5aedb1b064b4afb988c2`, which ran 0.05 -> 0.10 and passed the gate.

**What changes.** One field of the scorecard Prototype (STOMP) arm, `oak.stomp.base_step_size`,
set to **0.20** in the probe arm `prototype_step020`. One value, chosen before any life, and no
other value will be tried. No other field changes (tested). The override is the same runner-level
`dataclasses.replace` as in #3656; no registered source or library file is edited. This tunes the
candidate, not a control.

**Why 0.20.** #3656 found that 0.10 (effective step 0.20, matching the differential SARSA control)
beats the frozen Prototype and is noninferior to differential SARSA. The next doubling asks whether
the gain keeps growing or the step is already past its useful range. With the one-hot state input and
shared bias, α = 0.20 moves the visited value by 2αδ = 0.40δ (other state 0.20δ): memory about 2.5
own-cell visits against about 125 visits per state per phase.

**Predictions (low confidence, stated before any life).**

- Probe minus scorecard Prototype (0.05): positive.
- Probe minus #3656's 0.10 arm: not positive. A 0.40 effective step makes each value mostly the last
  one or two targets, and the bias couples the two states at 0.20δ, so steady-phase noise should cost
  more than the faster recovery after each switch gains.
- Gate: no prediction of PASS or FAIL; both are published.

**Gate (the scorecard's own, unchanged from #3656).** From `reference_life_scorecard.py`:

- normalized score = (arm − random) / scale, scale = mean(oracle − random), finite and > 0;
- a SARSA control qualifies if its one-sided 95% LCB (t = 2.201) is > 0.10;
- the probe must beat the frozen Prototype: paired LCB ≥ 0.05;
- and be noninferior to the best qualifying SARSA control (higher normalized mean): paired LCB
  ≥ −0.05.

PASS iff both hold. FAIL otherwise; a FAIL is kept and published as a clean negative, with no retune
and no third value. The scorecard Prototype's and the 0.10 arm's gates on the same seeds are reported
as context only, as are the two paired differences above.

**VOID** if any life fails, any parameter-change check fails, any Prototype life ran at the wrong
base step, the scale is invalid, no control qualifies, or any check below fails. VOID means stop: no
re-run, no replacement seeds.

**Seeds and reuse.** The same 12 seeds as #3656, 93610-93621, so the comparison with 0.10 is paired.
These seeds are no longer fresh: the 0.10 result on them is known, and 0.20 is chosen knowing it.
That is disclosed here and is why this stays a development probe. Only the probe arm is run (12
lives). The other seven arms on these seeds (0.10 probe, scorecard Prototype, frozen Prototype,
differential SARSA, SARSA, random, oracle) are the committed #3656 lives, reused unchanged, and
reused only if the replay check passes. Each life is a deterministic single process, so a life that
replays bit-exactly is the life this head would produce.

**Checks before any probe life.**

- Environment: the stored scorecard default Prototype life on consumed seed 70000 must reproduce
  bit-exactly in every outcome field, with the manifest config differing from the stored one only by
  the additive default `host_timing_metadata` (as explained in #3656).
- Probe path on seed 70000: completes, learns, ran at 0.20, config differs from the default only in
  `base_step_size`, and its transcript differs from the default's.
- Replay: all 7 #3656 arms re-run on seed 93610 with #3656's own runner; outcome fields, persistent
  bytes, transcript, manifest id and lifecycle id must equal the committed #3656 evidence.

Any failure: VOID, stop.

**Cost.** About 5.3 min per Prototype life at one thread. Checks: 5 Prototype lives plus 4 short
control lives; probe: 12 Prototype lives. About 30 min wall clock with 4 one-thread workers.
