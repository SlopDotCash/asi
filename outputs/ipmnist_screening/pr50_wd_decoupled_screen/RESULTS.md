# RESULTS: weight-decay-decoupled control for partial reset f=0.5 (development screen)

**Verdict (preregistered rule, 5e601e5f): DECAY-DRIVEN.** The f=0.5 run at 200 tasks is not recommended.

## Run
- Launched at HEAD e007c366 on 2026-10-31 01:40:37 UK time (GMT); finished 02:04:06 UK time (GMT).
- Env check PASS: the control seed 0 and the parent seed 0 match 3c38b025 bit for bit, and pr50 seed 4 matches e0612611 bit for bit.
- Parent (612491db, cut to 60 tasks) and pr50 (e0612611) shards are reused verbatim, with SHA256SUMS.

## Bars
S = step-only mean gain vs the parent and C = pr50 mean gain vs the parent.

| Bar | Requirement | Result | Pass? |
|---|---|---|---|
| STEP-REAL | S ≥ 0.5·C = +0.000307 and CI lower bound of S > 0 | S = −0.001071, CI lower bound −0.001163 | fails both |
| DECAY-DRIVEN | CI lower bound of S ≤ 0, or S < 0.25·C | met on both conditions; S/C = −1.75 | met |

## Component gains (paired, 60 tasks, seeds 4-13; 95% t CI with df=9)

| Comparison | Mean | 95% CI | Seeds improved | Gain by 15-task block |
|---|---|---|---|---|
| pr50 vs parent | +0.000613 | [+0.000547, +0.000680] | 10/10 | −0.000609 / +0.000427 / +0.001111 / +0.001525 |
| step-only vs parent | −0.001071 | [−0.001163, −0.000979] | 0/10 | −0.001921 / −0.001411 / −0.000820 / −0.000133 |
| decay-only vs parent | +0.001656 | [+0.001565, +0.001748] | 10/10 | +0.001268 / +0.001860 / +0.001760 / +0.001737 |
| step-only minus pr50 | −0.001685 | [−0.001773, −0.001596] | 0/10 | −0.001312 / −0.001837 / −0.001931 / −0.001659 |
| decay-only minus pr50 | +0.001043 | [+0.000935, +0.001151] | 10/10 | +0.001877 / +0.001433 / +0.000649 / +0.000212 |
| interaction | +0.000028 | [−0.000020, +0.000077] | 7/10 | +0.000044 / −0.000023 / +0.000171 / −0.000079 |

The interaction is pr50 − step-only − decay-only + parent.

Per-seed values are in DECOMPOSITION.txt.

## Reading
- The pr50 gain is the sum of two roughly additive parts; the interaction CI includes 0.
- Its step part (a higher post-shift step) **hurts**, by −0.00107.
- Its decay part (smaller post-shift effective weight decay) **helps**, by +0.00166.
- Decay-only, on its own, beats pr50 by +0.00104 on all 10 seeds.

## Caveats
- The decay-only arm is **descriptive and post hoc**. It is not a preregistered candidate, and seeds 4-13 have now been seen many times.
- Any follow-up (for example "slower decay after a shift" at full step) needs its own preregistration on fresh seeds.
- Development screen only; nonpromoting.
- The arm was chosen after the partial-reset result.
- The step-decay coupling (`ipmnist_screening.py` lines 5049-5053 at 3c38b025) is what this screen decouples.
