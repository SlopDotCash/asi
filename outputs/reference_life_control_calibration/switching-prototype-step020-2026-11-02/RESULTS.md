**Switching: Prototype base step 0.05 -> 0.20, results**

Run 2026-11-02 13:52-14:20 GMT at 334874758c62dc6e1e11ead8d2335c797d8f1de5, per DEVELOPMENT_PLAN.md
(committed before any life). Development-only and permanently nonpromoting.

**Verdict: PASS.** `prototype_step020` beats the frozen Prototype (paired LCB 0.7639 ≥ 0.05) and is
noninferior to the best qualifying control, differential SARSA (paired LCB 0.1353 ≥ −0.05). Both SARSA
controls qualify. No VOID condition was hit; all 12 probe lives completed, learned and ran at 0.20.

**Checks: PASS.** The stored scorecard default Prototype life on seed 70000 reproduced bit-exactly
(reward 2959.0), the probe path on seed 70000 ran at 0.20 with a different transcript (reward 3503.0),
and all 7 #3656 arms replayed on seed 93610 matched the committed #3656 lives bit-exactly (outcome
fields, transcript, manifest id, lifecycle id), so the reused lives are what this head produces.

**Arms (normalized, scale 2000.5833, 12 seeds 93610-93621, horizon 4000).** Only the 0.20 arm is new;
the rest are our own #3656's lives, reused.

| arm | mean | sd | one-sided 95% LCB |
|---|---|---|---|
| Prototype, base step 0.20 (probe) | 0.7822 | 0.0244 | 0.7667 |
| Prototype, base step 0.10 (#3656) | 0.6897 | 0.0316 | 0.6696 |
| differential SARSA | 0.6343 | 0.0201 | 0.6215 |
| Prototype, scorecard 0.05 | 0.5051 | 0.0513 | 0.4726 |
| SARSA | 0.4659 | 0.0291 | 0.4474 |
| frozen Prototype | 0.0035 | 0.0147 | −0.0059 |

**Predictions against outcome.** Probe minus scorecard Prototype: predicted positive, observed
+0.2771 (paired LCB 0.2538). Probe minus 0.10: predicted not positive, observed **+0.0926** (paired
LCB 0.0829), positive on all 12 seeds. That prediction was wrong: doubling the step again still helps
on this environment.

| seed | 0.20 | 0.10 | 0.05 | diff. SARSA | frozen | 0.20 − 0.10 |
|---|---|---|---|---|---|---|
| 93610 | 0.8083 | 0.7143 | 0.5463 | 0.6653 | 0.0180 | +0.0940 |
| 93611 | 0.7898 | 0.7118 | 0.5238 | 0.6243 | −0.0020 | +0.0780 |
| 93612 | 0.7888 | 0.7078 | 0.5338 | 0.6263 | −0.0090 | +0.0810 |
| 93613 | 0.8148 | 0.7108 | 0.5643 | 0.6368 | −0.0075 | +0.1040 |
| 93614 | 0.7723 | 0.6858 | 0.5388 | 0.6253 | 0.0015 | +0.0865 |
| 93615 | 0.7388 | 0.6318 | 0.4219 | 0.6378 | −0.0005 | +0.1070 |
| 93616 | 0.7613 | 0.6783 | 0.5278 | 0.6228 | 0.0050 | +0.0830 |
| 93617 | 0.7738 | 0.6438 | 0.4209 | 0.6373 | 0.0270 | +0.1300 |
| 93618 | 0.7588 | 0.6598 | 0.4359 | 0.5998 | −0.0105 | +0.0990 |
| 93619 | 0.7643 | 0.6813 | 0.4829 | 0.6183 | −0.0195 | +0.0830 |
| 93620 | 0.8058 | 0.7218 | 0.5278 | 0.6438 | 0.0145 | +0.0840 |
| 93621 | 0.8103 | 0.7288 | 0.5373 | 0.6738 | 0.0245 | +0.0815 |

**Limits.**

- Same seeds as #3656, chosen knowing the 0.10 result: the seeds are not fresh for this question, so
  this is development evidence about direction, not a confirmation. A confirmation needs fresh seeds.
- One environment, one horizon, one new value. Where the gain stops is not located; nothing here says
  0.20 is the best step, only that it beats 0.10 on these seeds.
- Seven of eight arms are reused, not re-run; the replay check covers one seed per arm, and every life
  is a deterministic single process.
- It does not change scorecard defaults or stored results.
